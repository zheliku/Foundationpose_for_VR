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
from typing import Generator
from pathlib import Path
import torch
import json
import cv2
import sys
import numpy as np
import trimesh
from VOT import Cutie, Tracker_2D
from utils.kalman_filter_6d import KalmanFilter6D
from utils.pose_tool import (
    adjust_pose_to_image_point,
    get_pose_xy_from_image_point,
    get_6d_pose_arr_from_mat,
    get_mat_from_6d_pose_arr
)

# 将项目根目录和 FoundationPose 库的路径添加到系统路径中
# 这样 Python 解释器才能找到并导入这些模块

print(sys.path)

# sys.exit()

src_path = os.getcwd()
foundationpose_path = os.path.join(src_path, "FoundationPose")
if src_path not in sys.path:
    sys.path.append(src_path)
if foundationpose_path not in sys.path:
    sys.path.append(foundationpose_path)

from FoundationPose.estimater import (
    ScorePredictor,
    PoseRefinePredictor,
    dr,
    FoundationPose,
    draw_posed_3d_box,
    draw_xyz_axis,
    trimesh_add_pure_colored_texture
)


def _prepare_rgbd(
        rgb_seq_path: str,
        depth_seq_path: str,
) -> Generator[tuple[int, np.ndarray, np.ndarray, str, str], None, None]:
    """
    读取并配对一组 RGB 与深度序列。

    参数
    ----
    rgb_seq_path : str
        RGB 图像序列所在目录路径。
    depth_seq_path : str
        深度图像序列所在目录路径（与 RGB 一一对应）。

    返回
    ----
    list[tuple]
        形如 [(i, color, depth, color_name, depth_name), ...] 的列表：
        - i: 帧序号（从 0 开始）；
        - color: BGR 彩色图（numpy.ndarray）；
        - depth: 深度图（单位：米，numpy.ndarray，已将异常值置为 0）；
        - color_name/depth_name: 文件名，便于可视化输出时复用。

    说明
    ----
    - 深度图以 OpenCV 读取后除以 1e3 转米；
    - 将 < 1mm 或 >= inf 的深度置为 0，避免无效深度参与后续估计。
    """
    color_frame_list = os.listdir(rgb_seq_path)
    depth_frame_list = os.listdir(depth_seq_path)

    color_frame_list.sort()
    depth_frame_list.sort()

    for i in range(len(color_frame_list)):
        color = cv2.imread(os.path.join(rgb_seq_path, color_frame_list[i]))
        depth = cv2.imread(os.path.join(depth_seq_path, depth_frame_list[i]), -1) / 1e3  # 转换为米
        depth[(depth < 0.001) | (depth >= np.inf)] = 0  # 无效深度置 0

        yield i, color, depth, color_frame_list[i], depth_frame_list[i]


def _prepare_mask(
        init_mask_path: str,
) -> np.ndarray:
    """
    读取初始目标掩码（灰度图），并标准化到 0/255 范围。

    参数
    ----
    init_mask_path : str
        初始掩码图像路径（通常只用于首帧的目标初始化）。

    返回
    ----
    numpy.ndarray
        单通道 8-bit 掩码图，前景为 255，背景为 0。
    """
    init_mask = cv2.imread(init_mask_path, cv2.IMREAD_GRAYSCALE)

    return init_mask.astype(np.uint8) * 255


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
        rgb_seq_path: str,
        depth_seq_path: str,
        mesh_path: str,
        init_mask_path: str,
        cam_K: np.ndarray,
        pose_output_path: str,
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
    # 1. 获取 RGB-D 帧生成器
    frame_generator = _prepare_rgbd(
        rgb_seq_path,
        depth_seq_path,
    )

    # 2. 读取初始帧的物体掩码
    init_mask = _prepare_mask(init_mask_path)

    # 3. 准备 3D 网格模型（含单位缩放、可选纯色）与 OBB
    mesh, to_origin, bbox = _prepare_3d_mesh(
        mesh_path,
        apply_scale=apply_scale,
        force_apply_color=force_apply_color,
    )

    # 4. 实例化 FoundationPose 估计器
    fp = _prepare_foundationpose(mesh, debug_dir="debug/")

    # 5. 选择 2D 目标跟踪器（开启时默认使用 Cutie，否则使用基础 Tracker_2D）
    if activate_2d_tracker:  # 默认使用 Cutie 作为 2D 跟踪器
        tracker_2D = Cutie()
    else:
        tracker_2D = Tracker_2D()

    # 6. 初始化卡尔曼滤波器（如果启用），kf_mean/kf_covariance 为滤波器内部状态
    kf = KalmanFilter6D(args.kf_measurement_noise_scale) if activate_kalman_filter \
        else None
    kf_mean, kf_covariance = None, None

    # 7. 准备可视化输出目录
    mask_visualization_path = Path(mask_visualization_path) if mask_visualization_path else None
    bbox_visualization_path = Path(bbox_visualization_path) if bbox_visualization_path else None
    pose_visualization_path = Path(pose_visualization_path) if pose_visualization_path else None

    if mask_visualization_path:
        mask_visualization_path.mkdir(parents=True, exist_ok=True)
    if bbox_visualization_path:
        bbox_visualization_path.mkdir(parents=True, exist_ok=True)
    if pose_visualization_path:
        pose_visualization_path.mkdir(parents=True, exist_ok=True)

    # 8. 遍历每一帧进行 6D 姿态跟踪
    pose_seq = []  # 存储每一帧的 4x4
    for frame_index, color, depth, color_frame_path_str, depth_frame_path_str in frame_generator:

        # 为该帧准备可视化输出文件名（若对应路径非空）
        mask_vis_path_str = str(mask_visualization_path / color_frame_path_str) if mask_visualization_path else None
        bbox_vis_path_str = str(bbox_visualization_path / color_frame_path_str) if bbox_visualization_path else None
        pose_vis_path_str = str(pose_visualization_path / color_frame_path_str) if pose_visualization_path else None

        print(mask_vis_path_str)
        print(bbox_vis_path_str)

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
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # # 若指定了路径，则保存姿态可视化结果
        if pose_vis_path_str:
            cv2.imwrite(pose_vis_path_str, vis_color)

    #################################################
    # 保存整段位姿序列（单位为米，坐标系与输入 K 一致）
    #################################################
    np.save(pose_output_path, np.array(pose_seq))

    # 清理资源
    torch.cuda.empty_cache()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--rgb_seq_path", type=str,
                        default="/workspace/foundationpose/FoundationPose-plus-plus/$TESTCASE/rgb")
    parser.add_argument("--depth_seq_path", type=str,
                        default="/workspace/foundationpose/FoundationPose-plus-plus/$TESTCASE/depth")
    parser.add_argument("--mesh_path", type=str,
                        default="/workspace/foundationpose/FoundationPose-plus-plus/$TESTCASE/mesh/$TESTCASE.stl")
    parser.add_argument("--init_mask_path", type=str,
                        default="/workspace/foundationpose/FoundationPose-plus-plus/$TESTCASE/0_mask.png")

    parser.add_argument("--pose_output_path", type=str,
                        default="/workspace/yanwenhao/detection/FoundationPose++/pose.npy")
    parser.add_argument("--mask_visualization_path", type=str,
                        default="/workspace/foundationpose/FoundationPose-plus-plus/$TESTCASE/masks_visualization")
    parser.add_argument("--bbox_visualization_path", type=str,
                        default="/workspace/foundationpose/FoundationPose-plus-plus/$TESTCASE/bbox_visualization")
    parser.add_argument("--pose_visualization_path", type=str,
                        default="/workspace/foundationpose/FoundationPose-plus-plus/$TESTCASE/pose_visualization")
    parser.add_argument("--cam_K", type=json.loads,
                        default="[[387.88845825, 0.0, 323.28192139], [0.0, 387.46902466, 237.11705017], [0.0, 0.0, 1.0]]",
                        help="Camera intrinsic parameters")
    parser.add_argument("--est_refine_iter", type=int, default=5,
                        help="FoundationPose initial refine iterations, see https://github.com/NVlabs/FoundationPose")
    parser.add_argument("--track_refine_iter", type=int, default=2,
                        help="FoundationPose tracking refine iterations, see https://github.com/NVlabs/FoundationPose")
    parser.add_argument("--activate_2d_tracker", action='store_true', help="activate 2d tracker")
    parser.add_argument("--activate_kalman_filter", action='store_true', help="activate kalman_filter")
    parser.add_argument("--kf_measurement_noise_scale", type=float, default=0.05,
                        help="The scale of measurement noise relative to prediction in kalman filter, greater value means more filtering. Only effective if activate_kalman_filter")
    parser.add_argument("--apply_scale", type=float, default=0.01,
                        help="Mesh scale factor in meters (1.0 means no scaling), commonly use 0.01")
    parser.add_argument("--force_apply_color", action='store_true', help="force a color for colorless mesh")
    parser.add_argument("--apply_color", type=json.loads, default="[0, 159, 237]",
                        help="RGB color to apply, in format 'r,g,b'. Only effective if force_apply_color")
    args = parser.parse_args()

    track_pose(
        args.rgb_seq_path,
        args.depth_seq_path,
        args.mesh_path,
        args.init_mask_path,
        np.array(args.cam_K),
        args.pose_output_path,
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
