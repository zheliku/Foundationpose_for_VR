"""
本文件功能概述
================
利用 FoundationPose 对给定 RGB-D 序列中的单个已知 3D 网格模型进行 6D 姿态（位姿）初始化与逐帧跟踪，
并可选地结合 2D 跟踪器与 6D 卡尔曼滤波进行稳定化与先验引导，同时输出可视化结果与位姿序列。

核心流程
--------
1. 读取 RGB 与深度序列，按文件名排序配对；深度转换为米并过滤无效值。
2. 载入 3D 网格，按需要缩放到米单位，若模型无颜色可强制填充纯色纹理；
   同时计算模型的有向包围盒（OBB）以便后续可视化。
3. 初始化 FoundationPose（包含评分器、精炼器与 CUDA 光栅化上下文）。
4. 第一帧：利用初始掩码进行姿态注册（register），得到初始 6D 姿态；
   同时初始化 2D 跟踪器与（可选）卡尔曼滤波器。
5. 后续帧：
   - 用 2D 跟踪器得到目标在图像中的 2D 框/中心；
   - 若启用 2D 跟踪与/或卡尔曼滤波，则将该 2D 信息转化为对上一帧 6D 位姿的先验/约束；
   - 调用 FoundationPose 的 track_one 做多次迭代的位姿精炼与跟踪；
   - 可选地用卡尔曼滤波进行一步预测，使滤波器状态与跟踪保持同步。
6. 将每帧位姿保存到列表，并绘制 3D 盒与坐标轴到图像上进行在线展示/保存。
7. 将整段位姿序列写入 .npy 文件，释放显存并关闭窗口。

注意事项
--------
- 深度单位固定转换为米（除以 1000），并将过小/无穷深度置零以避免异常值影响。
- _prepare_3d_mesh 使用全局参数 args.apply_color 来强制着色（当 force_apply_color 为真时才生效）。
- 2D 跟踪器支持 Cutie 与一个基础 Tracker_2D，默认在 activate_2d_tracker 为真时使用 Cutie。
- 卡尔曼滤波通过位置像素观测（图像中心点）与 6D 姿态观测联合更新，预测-更新节奏与 FoundationPose 同步。
"""

import argparse
import os
from typing import Sequence
from pathlib import Path
import torch
import json
import cv2
import sys
import trimesh
from VOT import Cutie, Tracker_2D
from utils.kalman_filter_6d import KalmanFilter6D
from utils.pose_tool import (
    adjust_pose_to_image_point,
    get_pose_xy_from_image_point,
    get_6d_pose_arr_from_mat,
    get_mat_from_6d_pose_arr,
    rs_intrinsics_to_cv,
)
import pyrealsense2 as rs
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from hydra.utils import instantiate

print(sys.path)

# 将项目根目录和 FoundationPose 库的路径添加到系统路径中
# 这样 Python 解释器才能找到并导入这些模块
src_path = os.getcwd()
foundationpose_path = Path(src_path) / "FoundationPose"
sam2_path = Path(src_path) / "sam2"
if src_path not in sys.path:
    sys.path.append(src_path)
if foundationpose_path not in sys.path:
    sys.path.append(str(foundationpose_path.resolve()))
if sam2_path not in sys.path:
    sys.path.append(str(sam2_path.resolve()))

from FoundationPose.estimater import (
    ScorePredictor,
    PoseRefinePredictor,
    dr,
    FoundationPose,
    draw_posed_3d_box,
    draw_xyz_axis,
    trimesh_add_pure_colored_texture
)
import numpy as np

try:
    from sam2.build_sam import build_sam2, _load_checkpoint
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    raise ImportError("请确保已正确安装 SAM 2 相关依赖，并将 sam2 目录添加到 PYTHONPATH 中。")


def _prepare_3d_mesh(
        mesh_path: str,
        apply_scale: bool,
        force_apply_color: bool,
) -> tuple[trimesh.Geometry, np.ndarray, np.ndarray]:
    """
    载入并规范化 3D 网格模型，按需缩放与着色，并计算有向包围盒用于可视化。

    参数
    ----
    mesh_path : str
        3D 网格文件路径（如 .stl/.obj）。
    apply_scale : float
        模型缩放比例（米尺度），例如 0.01 表示将单位从厘米转米。
    force_apply_color : bool
        若为 True 且模型没有顶点色/纹理，则用纯色纹理进行着色（颜色来自全局 args.apply_color）。

    返回
    ----
    (mesh, to_origin, bbox) : tuple
        - mesh: 处理后的 trimesh 对象；
        - to_origin: 将 OBB 对齐到原点的 4x4 仿射矩阵；
        - bbox: 形状为 (2, 3) 的包围盒顶点范围（最小/最大），用于 3D 框可视化。
    """
    mesh = trimesh.load(mesh_path)

    # 将模型单位转换为米（按需缩放）
    mesh.apply_scale(apply_scale)

    # 如果模型没有颜色，强制为其应用一个纯色纹理
    if force_apply_color:
        mesh = trimesh_add_pure_colored_texture(mesh, color=np.array(args.apply_color), resolution=10)

    # 计算模型的有向边界框 (OBB)，用于可视化
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    return mesh, to_origin, bbox


def _prepare_foundationpose(
        mesh: trimesh.Geometry,
        debug_dir: str = "debug/"
) -> FoundationPose:
    """
    构建 FoundationPose 估计器，包括评分器、精炼器与 CUDA 光栅化上下文。

    参数
    ----
    mesh : trimesh.Geometry
        目标物体的三角网格（需包含顶点与法线）。
    debug_dir : str
        FoundationPose 的调试输出目录。

    返回
    ----
    FoundationPose
        已初始化的姿态估计器，可进行 register 与 track_one。
    """
    # 初始化 FoundationPose 需要的评分器和精炼器
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()

    # 初始化 CUDA 光栅化上下文
    glctx = dr.RasterizeCudaContext()

    # 实例化 FoundationPose 估计器
    fp = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        glctx=glctx,
        debug_dir=debug_dir
    )
    return fp


def _prepare_realsense() -> tuple[rs.pipeline, rs.align, np.ndarray, np.ndarray]:
    # 创建 RealSense 管道
    pipeline = rs.pipeline()
    config = rs.config()

    # 配置彩色图和深度图的流
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # 启动管道
    pipeline.start(config)

    # 创建对齐对象，将深度图对齐到彩色图
    align_to = rs.stream.color
    align = rs.align(align_to)

    # 获取相机内参
    profile = pipeline.get_active_profile()
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_stream.get_intrinsics()

    # 将 RealSense 内参转换为 OpenCV 格式（供姿态估计/画坐标轴使用）
    camera_matrix, dist_coeffs = rs_intrinsics_to_cv(intr)

    return pipeline, align, camera_matrix, dist_coeffs


def _prepare_sam2():
    # 1) 构建/加载 SAM 2 模型与图像预测器
    # 你的配置和权重路径
    model_cfg_path = "./sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
    checkpoint = "./sam2/checkpoints/sam2.1_hiera_large.pt"

    # 1. 读取配置文件（不用 hydra.initialize）
    cfg = OmegaConf.load(model_cfg_path)
    OmegaConf.resolve(cfg)

    # 2. 手动实例化模型
    # Hydra 原本做的就是 instantiate(cfg.model, _recursive_=True)
    model = instantiate(cfg.model, _recursive_=True)

    # 3. 加载权重（同原函数）
    # from sam2.sam2.build_sam import _load_checkpoint  # 这个函数可直接用
    _load_checkpoint(model, checkpoint)

    # 4. 上设备 & eval 模式
    model = model.to("cuda")
    model.eval()

    # 5. 构建 predictor
    predictor = SAM2ImagePredictor(model)

    return predictor


def _prepare_aruco_detector(aruco_dict: int):
    # 准备 ArUco 检测器（OpenCV 4.11 新式 API：ArucoDetector）
    dictionary = cv2.aruco.getPredefinedDictionary(aruco_dict)
    params = cv2.aruco.DetectorParameters()  # 4.11 新写法：直接构造，不再用 *_create()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX  # 亚像素角点精炼
    detector = cv2.aruco.ArucoDetector(dictionary, params)

    return detector


def _read_rgbd(
        pipeline: rs.pipeline,
        align: rs.align,
) -> tuple[np.ndarray, np.ndarray]:
    # 等待获取下一帧
    frames = pipeline.wait_for_frames()

    # 对齐深度帧到彩色帧
    aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()

    # 将图像转换为 numpy 数组
    color = np.asanyarray(color_frame.get_data())
    depth = np.asanyarray(depth_frame.get_data()) / 1000.0  # 转换为米
    depth[(depth < 0.001) | (depth >= np.inf)] = 0  # 无效深度置 0

    return color, depth


def _try_get_mask(
        color: np.ndarray,
        predictor: SAM2ImagePredictor,
        corners: Sequence[np.ndarray],
        ids: np.ndarray,
        rejected: Sequence[np.ndarray]
):
    if ids is None or len(ids) == 0:
        print("未检测到 ArUco 标记，无法生成初始掩码。")
        return None

    print(corners)
    print(np.asarray(corners).reshape(-1, 2))

    # 准备点提示（可多个）。点坐标格式为 (x, y)，标签 1=前景，0=背景。
    # 这里示例仅用一个前景点；你可以添加更多点以细化结果，例如：
    # input_point = np.array([[300, 275], [x2, y2], ...])
    # input_label = np.array([1, 0, ...])
    input_point = np.asarray(corners).reshape(-1, 2)
    input_label = np.array([1 for _ in range(len(input_point))])

    for pt in input_point:
        print(pt)
        cv2.circle(color, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)
    cv2.imshow("Mask Points", color)

    # 进行预测：先设置当前图像，再基于提示进行多掩码输出
    predictor.set_image(color)

    # multimask_output=True 会返回多个候选掩码及其分数，便于从中挑选最佳结果
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True
    )

    # 根据得分从高到低对结果进行排序，便于优先查看高置信掩码
    sorted_ind = np.argsort(scores)[::-1]
    masks = masks[sorted_ind]
    scores = scores[sorted_ind]
    logits = logits[sorted_ind]

    first_mask: np.ndarray = masks[0].astype(np.uint8) * 255
    cv2.imshow("Initial Mask", first_mask)

    return first_mask


def _track_first_frame(
        color: np.ndarray,
        depth: np.ndarray,
        init_mask: np.ndarray,
        cam_K: np.ndarray,
        fp: FoundationPose,
        tracker_2D: Tracker_2D,
        kf: KalmanFilter6D | None,
        mask_vis_filename: str | None,
        bbox_vis_filename: str | None,
        est_refine_iter: int,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    pose = fp.register(K=cam_K, rgb=color, depth=depth, ob_mask=init_mask, iteration=est_refine_iter)

    if kf:
        # 以首帧姿态作为滤波器初始观测
        kf_mean, kf_covariance = kf.initiate(get_6d_pose_arr_from_mat(pose))
    else:
        kf_mean, kf_covariance = None, None

    tracker_2D.initialize(
        color,
        init_info={"mask": init_mask},
        mask_visualization_path=mask_vis_filename,
        bbox_visualization_path=bbox_vis_filename
    )

    return pose, kf_mean, kf_covariance
    # return pose, None, None


def _track_subsequent_frame(
        color: np.ndarray,
        depth: np.ndarray,
        cam_K: np.ndarray,
        fp: FoundationPose,
        tracker_2D: Tracker_2D,
        kf: KalmanFilter6D | None,
        kf_mean: np.ndarray | None,
        kf_covariance: np.ndarray | None,
        mask_vis_filename: str | None,
        bbox_vis_filename: str | None,
        track_refine_iter: int,
        activate_2d_tracker: bool,
        activate_kalman_filter: bool,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    bbox_2d = tracker_2D.track(
        color,
        mask_visualization_path=mask_vis_filename,
        bbox_visualization_path=bbox_vis_filename
    )
    if activate_2d_tracker:
        if not activate_kalman_filter:
            # 仅用 2D 框中心修正上一帧位姿的投影中心（快速启发，不改变深度/旋转）
            fp.pose_last = adjust_pose_to_image_point(ob_in_cam=fp.pose_last, K=cam_K,
                                                      x=bbox_2d[0] + bbox_2d[2] / 2,
                                                      y=bbox_2d[1] + bbox_2d[3] / 2)
        else:
            # 同时使用卡尔曼滤波：
            # 1) 用上一帧位姿做一次 6D 观测更新（减弱抖动）；
            kf_mean, kf_covariance = kf.update(kf_mean, kf_covariance, get_6d_pose_arr_from_mat(fp.pose_last))
            # 2) 将 2D 框中心转为像素观测（与当前位姿的投影中心对齐），再做一次 xy 更新；
            measurement_xy = np.array(
                get_pose_xy_from_image_point(ob_in_cam=fp.pose_last, K=cam_K, x=bbox_2d[0] + bbox_2d[2] / 2,
                                             y=bbox_2d[1] + bbox_2d[3] / 2))
            kf_mean, kf_covariance = kf.update_from_xy(kf_mean, kf_covariance, measurement_xy)
            # 3) 将滤波后的 6D 状态还原为 4x4 位姿矩阵，作为 FoundationPose 的上一帧初值
            fp.pose_last = torch.from_numpy(get_mat_from_6d_pose_arr(kf_mean[:6])).unsqueeze(0).to(
                fp.pose_last.device)

    # 核心：用 FoundationPose 在当前帧进行迭代精炼与跟踪，得到该帧位姿
    pose = fp.track_one(rgb=color, depth=depth, K=cam_K, iteration=track_refine_iter)

    if activate_2d_tracker and kf:
        # 跟踪完成后做一次预测，使滤波器状态与下一帧节奏对齐（kf 总是“慢一拍”）
        kf_mean, kf_covariance = kf.predict(kf_mean, kf_covariance)

    return pose, kf_mean, kf_covariance
    # return pose, None, None


def track_pose(
        mesh_path: str,
        cam_K: np.ndarray,
        pose_output_path: str,
        enable_visualizations: bool,
        mask_visualization_path: str,
        bbox_visualization_path: str,
        pose_visualization_path: str,
        est_refine_iter: int,
        track_refine_iter: int,
        apply_scale: bool,
        force_apply_color: bool,
        activate_2d_tracker: bool = False,
        activate_kalman_filter: bool = False,
) -> None:
    """
    主函数：对整段 RGB-D 序列进行 6D 姿态初始化与逐帧跟踪，并输出可视化与位姿序列。

    参数
    ----
    rgb_seq_path / depth_seq_path : str
        RGB 与深度序列目录（文件名需一一对应，函数内部将按字典序排序）。
    mesh_path : str
        物体 CAD 网格路径（支持 .stl/.obj 等）。
    init_mask_path : str
        首帧目标掩码路径；用于 FoundationPose 的 register 初始化。
    cam_K : numpy.ndarray
        3x3 相机内参矩阵。
    pose_output_path : str
        位姿序列保存路径（.npy）。
    mask_visualization_path / bbox_visualization_path / pose_visualization_path : str
        可视化输出目录（分别为掩码、2D 框、姿态渲染）。若为 None 则不保存。
    est_refine_iter / track_refine_iter : int
        初始化/跟踪阶段的 Refinement 迭代次数；数值越大，精度可能更高但速度更慢。
    activate_2d_tracker : bool
        是否启用 2D 跟踪器（Cutie/Tracker_2D），用于提供图像中心先验与可视化。
    activate_kalman_filter : bool
        是否启用 6D 卡尔曼滤波（与 2D 观测融合），以平滑/稳健跟踪。

    返回
    ----
    None
    """
    # ============== 可调参数 ==============
    ARUCO_DICT = cv2.aruco.DICT_4X4_50  # 你也可以改成 DICT_6X6_250 / DICT_7X7_1000 等
    MARKER_LENGTH_M = 0.045  # 单个 ArUco 标记实物边长（米）, 用于姿态估计
    SHOW_POSE_AXES = True  # 如果为 True 且拿到相机内参，将绘制姿态坐标轴
    # ====================================

    # 1. 准备相机数据
    pipeline, align, camera_matrix, dist_coeffs = _prepare_realsense()

    # 2. 准备 3D 网格模型（含单位缩放、可选纯色）与 OBB
    mesh, to_origin, bbox = _prepare_3d_mesh(
        mesh_path,
        apply_scale=apply_scale,
        force_apply_color=force_apply_color,
    )

    # 3. 实例化 FoundationPose 估计器
    fp = _prepare_foundationpose(mesh, debug_dir="FoundationPose/debug/")

    # 4. 选择 2D 目标跟踪器（开启时默认使用 Cutie，否则使用基础 Tracker_2D）
    if activate_2d_tracker:
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        # 默认使用 Cutie 作为 2D 跟踪器
        tracker_2D = Cutie()
    else:
        tracker_2D = Tracker_2D()

    # 5. 初始化卡尔曼滤波器（如果启用），kf_mean/kf_covariance 为滤波器内部状态
    kf = KalmanFilter6D(args.kf_measurement_noise_scale) if activate_kalman_filter \
        else None
    kf_mean, kf_covariance = None, None

    # 6. 准备可视化输出目录
    mask_visualization_path = Path(mask_visualization_path) if mask_visualization_path else None
    bbox_visualization_path = Path(bbox_visualization_path) if bbox_visualization_path else None
    pose_visualization_path = Path(pose_visualization_path) if pose_visualization_path else None

    if enable_visualizations:
        if mask_visualization_path:
            mask_visualization_path.mkdir(parents=True, exist_ok=True)
        if bbox_visualization_path:
            bbox_visualization_path.mkdir(parents=True, exist_ok=True)
        if pose_visualization_path:
            pose_visualization_path.mkdir(parents=True, exist_ok=True)

    # 7. 准备 SAM 2 图像预测器
    predictor = _prepare_sam2()

    # 8. 准备 ArUco 检测器
    detector = _prepare_aruco_detector(ARUCO_DICT)

    # 9. 遍历每一帧进行 6D 姿态跟踪
    pose_seq = []  # 存储每一帧的 4x4
    try:
        frame_index = 0
        init_mask = None

        while True:
            # 为该帧准备可视化输出文件名（若对应路径非空）
            mask_vis_path_str = str(
                mask_visualization_path / f"{frame_index:04d}.png") if mask_visualization_path else None
            bbox_vis_path_str = str(
                bbox_visualization_path / f"{frame_index:04d}.png") if bbox_visualization_path else None
            pose_vis_path_str = str(
                pose_visualization_path / f"{frame_index:04d}.png") if pose_visualization_path else None

            color, depth = _read_rgbd(pipeline, align)
            corners, ids, rejected = detector.detectMarkers(color)

            print(f"corners: {corners}, ids: {ids}")

            if init_mask is None:
                init_mask = _try_get_mask(color, predictor, corners, ids, rejected)
                print(f"init_mask: {init_mask}")
            else:
                # cv2.imwrite("initial_mask.png", init_mask)

                # 处理首帧：用 register 根据掩码估计初始位姿，并初始化 2D 跟踪器/卡尔曼滤波
                if frame_index == 0:
                    pose, kf_mean, kf_covariance = _track_first_frame(
                        color,
                        depth,
                        init_mask,
                        cam_K,
                        fp,
                        tracker_2D,
                        kf,
                        mask_vis_path_str,
                        bbox_vis_path_str,
                        est_refine_iter,
                    )

                # 处理后续帧：先用 2D 跟踪器得到目标位置，再作为先验约束引导 FoundationPose
                else:
                    pose, kf_mean, kf_covariance = _track_subsequent_frame(
                        color,
                        depth,
                        cam_K,
                        fp,
                        tracker_2D,
                        kf,
                        kf_mean,
                        kf_covariance,
                        mask_vis_path_str,
                        bbox_vis_path_str,
                        track_refine_iter,
                        activate_2d_tracker,
                        activate_kalman_filter,
                    )

                # 记录该帧位姿（确保为 4x4 矩阵）
                pose_seq.append(pose.reshape(4, 4))

                # 可视化：将位姿对齐到 OBB 中心后绘制 3D 盒与坐标轴
                center_pose = pose @ np.linalg.inv(to_origin)
                vis_color = draw_posed_3d_box(cam_K, img=color, ob_in_cam=center_pose, bbox=bbox)
                vis_color = draw_xyz_axis(
                    vis_color,
                    ob_in_cam=center_pose,
                    scale=0.1,
                    K=cam_K,
                    thickness=3,
                    transparency=0,
                    is_input_rgb=True,
                )

                # 实时显示：按 "q" 键即可提前退出
                cv2.imshow("Live Pose Estimation", vis_color)

                # # 若指定了路径，则保存姿态可视化结果
                if pose_vis_path_str:
                    cv2.imwrite(pose_vis_path_str, vis_color)

                frame_index += 1

            if SHOW_POSE_AXES:
                cv2.aruco.drawDetectedMarkers(color, corners, ids)
                if ids is not None and len(ids) > 0:
                    # 估计每个标记的 rvec / tvec（按单标记假设）
                    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                        corners, MARKER_LENGTH_M, camera_matrix, dist_coeffs
                    )
                    for rvec, tvec in zip(rvecs, tvecs):
                        # 画坐标轴（长度设为标记边长的 0.75 倍）
                        cv2.drawFrameAxes(color, camera_matrix, dist_coeffs,
                                          rvec, tvec, MARKER_LENGTH_M * 0.75)
                cv2.imshow("Aruco Detection", color)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        # 捕获 Ctrl+C 中断
        print("正在停止...")

    finally:
        # 停止管道
        pipeline.stop()
        #################################################
        # 保存整段位姿序列（单位为米，坐标系与输入 K 一致）
        #################################################
        np.save(pose_output_path, np.array(pose_seq))

        # 清理资源
        torch.cuda.empty_cache()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 定义命令行参数
    parser.add_argument("--mesh_path", type=str, required=True, help="要跟踪物体的 3D 模型文件路径 (.stl, .ply, etc.)")
    parser.add_argument("--pose_output_path", type=str, default="pose_sequence.npy",
                        help="保存姿态序列的 .npy 文件路径")
    # (以下可视化路径参数在当前脚本中未直接使用，但为扩展性保留)
    parser.add_argument("--enable_visualizations", type=bool, default=False, help="启用可视化")
    parser.add_argument("--mask_visualization_path", type=str, default="visualizations/masks")
    parser.add_argument("--bbox_visualization_path", type=str, default="visualizations/bbox")
    parser.add_argument("--pose_visualization_path", type=str, default="visualizations/pose")

    parser.add_argument("--cam_K", type=json.loads, default='[[609.45, 0, 322.08], [0, 608.35, 242.44], [0, 0, 1]]',
                        help="相机内参矩阵，以 JSON 字符串形式提供")
    parser.add_argument("--est_refine_iter", type=int, default=10, help="FoundationPose 初始姿态估计的精炼迭代次数")
    parser.add_argument("--track_refine_iter", type=int, default=5, help="FoundationPose 跟踪过程中的精炼迭代次数")
    parser.add_argument("--activate_2d_tracker", action='store_true', help="激活 2D 跟踪器以辅助 6D 跟踪")
    parser.add_argument("--activate_kalman_filter", action='store_true', help="激活卡尔曼滤波器以平滑姿态")
    parser.add_argument("--kf_measurement_noise_scale", type=float, default=0.05,
                        help="卡尔曼滤波器的测量噪声比例，值越大滤波效果越强")
    parser.add_argument("--apply_scale", type=float, default=1.0,
                        help="应用于 3D 模型的缩放因子 (例如，从毫米到米用 0.001)")
    parser.add_argument("--force_apply_color", action='store_true', help="为无色模型强制应用指定颜色")
    parser.add_argument("--apply_color", type=json.loads, default="[0, 159, 237]",
                        help="要应用的 RGB 颜色, 格式为 '[r,g,b]'")
    parser.add_argument("--class_name", type=str, help="YOLO 物体检测的目标类别名称")
    parser.add_argument("--yolo_model_path", type=str, help="YOLO 模型权重文件的路径 (.pt)")

    args = parser.parse_args()

    track_pose(
        args.mesh_path,
        np.array(args.cam_K),
        args.pose_output_path,
        args.enable_visualizations,
        args.mask_visualization_path,
        args.bbox_visualization_path,
        args.pose_visualization_path,
        args.est_refine_iter,
        args.track_refine_iter,
        args.apply_scale,
        args.force_apply_color,
        args.activate_2d_tracker,
        args.activate_kalman_filter,
    )

    torch.cuda.empty_cache()
