import argparse
import os
import time
import torch
import json
import cv2
import sys
import numpy as np
import trimesh
from VOT import Cutie, Tracker_2D
from utils.kalman_filter_6d import KalmanFilter6D
import pyrealsense2 as rs
from ultralytics import YOLO
from utils.pose_tool import (
    adjust_pose_to_image_point,
    get_pose_xy_from_image_point,
    get_6d_pose_arr_from_mat,
    get_mat_from_6d_pose_arr
)

# 将项目根目录和 FoundationPose 库的路径添加到系统路径中
# 这样 Python 解释器才能找到并导入这些模块
src_path = os.getcwd()
foundationpose_path = os.path.join(src_path, "FoundationPose")
if src_path not in sys.path:
    sys.path.append(src_path)
if foundationpose_path not in sys.path:
    sys.path.append(foundationpose_path)

print(sys.path)

from FoundationPose.estimater import (
    ScorePredictor,
    PoseRefinePredictor,
    dr,
    FoundationPose,
    logging,
    draw_posed_3d_box,
    draw_xyz_axis,
    trimesh_add_pure_colored_texture
)

# sys.exit(0)

# 定义从机器人基座坐标系到工具末端（tool0）坐标系的变换矩阵
# 这个矩阵描述了机器人在“home”位置时，工具末端相对于基座的姿态
T_base_tool0 = np.array([
    [-0.815, -0.465, -0.345, -0.520],
    [-0.496, 0.868, 0.001, 0.133],
    [0.299, 0.172, -0.939, 0.529],
    [0.000, 0.000, 0.000, 1.000]
])

# 定义从相机坐标系到夹爪（gripper）坐标系的变换矩阵
# 这个矩阵通常通过手眼标定过程获得
T_cam2gripper = np.array([
    [0.57123, 0.81900076, -0.05416683, 0.06254428],
    [-0.82046694, 0.56790884, -0.06567765, 0.03732711],
    [-0.02302823, 0.08195913, 0.99636961, 0.01224618],
    [0.0, 0.0, 0.0, 1.0]
])


def get_object_mask_from_frame(
        frame: np.ndarray,
        target_class_name: str,
        yolo_model: YOLO,
        confidence_threshold: float = 0.9,
        visualize: bool = True) -> np.ndarray:
    """
    使用 YOLO 模型从单帧图像中检测目标物体并生成其二进制掩码。

    :param frame: 输入的图像帧 (numpy array)。
    :param target_class_name: 目标物体的类别名称 (str)。
    :param yolo_model: 预加载的 YOLO 模型。
    :param confidence_threshold: 检测结果的置信度阈值。
    :param visualize: 是否可视化检测结果。
    :return: 一个与输入帧同样大小的二进制掩码 (numpy array)，目标物体区域为 255，背景为 0。
    """
    frame_h, frame_w = frame.shape[:2]
    # 创建一个全黑的空白掩码
    mask_full = np.zeros((frame_h, frame_w), dtype=np.uint8)

    # 使用 YOLO 模型进行预测
    results = yolo_model(frame)

    # 遍历检测结果
    for result in results:
        classes_names = result.names
        if result.masks is not None:
            masks = result.masks.xy
            for mask, box in zip(masks, result.boxes):
                cls = int(box.cls[0])
                class_name = classes_names[cls]
                conf = float(box.conf[0])

                # 如果类别匹配且置信度高于阈值
                if class_name.lower() == target_class_name.lower() and conf > confidence_threshold:
                    # 将检测到的多边形轮廓转换为 numpy 数组
                    mask_np = np.array(mask, dtype=np.int32)
                    # 在空白掩码上填充该物体的区域
                    cv2.fillPoly(mask_full, [mask_np], 255)

                    # 如果需要可视化
                    if visualize:
                        overlay_color = (255, 0, 0)  # 蓝色
                        # 在原图上绘制物体轮廓
                        cv2.polylines(frame, [mask_np], isClosed=True, color=overlay_color, thickness=2)
                        # 在原图上显示类别名称和置信度
                        cv2.putText(frame, f'{class_name} {conf:.2f}',
                                    (mask_np[0][0], mask_np[0][1]),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, overlay_color, 2)

    # 如果需要可视化，显示带有检测结果的图像
    if visualize:
        cv2.imshow('YOLO Segmentation', frame)
        cv2.waitKey(1)  # 等待 1ms，用于刷新窗口

    return mask_full


def _prepare_3d_mesh(
        mesh_path: str,
        apply_scale: bool,
        force_apply_color: bool,
) -> tuple[trimesh.Geometry, np.ndarray, np.ndarray]:
    mesh_file: str = os.path.join(mesh_path)

    if not os.path.exists(mesh_file):
        # print(f"3D 模型文件未找到: {mesh_file}")
        raise FileNotFoundError(f"3D 模型文件未找到: {mesh_file}")

    mesh = trimesh.load(mesh_file)

    # 如果模型是场景图，则合并为一个单独的网格
    # if isinstance(mesh, trimesh.Scene):
    #     mesh: list[trimesh.Geometry] = mesh.dump(concatenate=True)

    # 将模型单位转换为米
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
    logging.info("FoundationPose 估计器初始化完成")
    return fp


def _prepare_realsense() -> tuple[rs.pipeline, rs.align]:
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
    return pipeline, align


def _get_mask(
        pipeline: rs.pipeline,
        class_name: str,
        yolo_model: YOLO,
        timeout_seconds: int = 120
):
    start_time = time.time()
    i = -1  # 初始化帧计数器
    print("正在等待检测目标物体...")

    while True:
        # 等待相机帧
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())

        # 使用 YOLO 检测物体并获取掩码
        mask_full = get_object_mask_from_frame(
            frame=frame,
            target_class_name=class_name,
            yolo_model=yolo_model,
            confidence_threshold=0.9,
            visualize=True
        )

        init_mask = mask_full.astype(bool)

        # 如果检测到任何物体（掩码不为空）
        if np.any(init_mask):
            print("检测到物体，准备开始姿态跟踪。")
            return init_mask

        # 如果超时仍未检测到物体
        if time.time() - start_time > timeout_seconds:
            print("超时！未能检测到物体。")
            pipeline.stop()
            break

    if i == -1:
        raise RuntimeError("未能初始化帧计数器。物体检测失败。")

    return None


def _track_pose(
        i: int,
        pipeline: rs.pipeline,
        align: rs.align,
        fp: FoundationPose,
        cam_K: torch.Tensor,
        init_mask: np.ndarray,
        est_refine_iter: int,
        track_refine_iter: int,
        tracker_2D: Tracker_2D,
        kf: KalmanFilter6D,
        kf_mean: np.ndarray,
        kf_covariance: np.ndarray,
):
    # 获取对齐后的帧
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()

    if not color_frame or not depth_frame:
        print("获取帧失败")
        return

    # 将帧数据转换为 numpy 数组
    color = np.asanyarray(color_frame.get_data())
    depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) / 1000.0  # 深度单位转为米
    depth[(depth < 0.001) | (depth >= np.inf)] = 0  # 过滤无效深度值

    # 确保图像尺寸一致
    color = cv2.resize(color, (color.shape[1], color.shape[0]), interpolation=cv2.INTER_NEAREST)
    depth = cv2.resize(depth, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)

    if i == 0:
        # --- 第一帧：注册 (Register) ---
        mask = init_mask.astype(np.uint8) * 255
        # 使用初始掩码、RGB 和深度图进行姿态估计
        pose = fp.register(K=cam_K, rgb=color, depth=depth, ob_mask=mask, iteration=est_refine_iter)
        pose_arr = get_6d_pose_arr_from_mat(pose)
        # position = pose_arr[:3]
        # euler = pose_arr[3:]
        # pose_matrix = get_mat_from_6d_pose_arr(pose_arr)
        # print(f"第 {i} 帧 (初始姿态):")
        # print("位置 (xyz):", position)
        # print("姿态 (欧拉角):", euler)
        # print("姿态矩阵:\n", pose_matrix)

        # 如果激活了卡尔曼滤波器，用初始姿态对其进行初始化
        if kf:
            kf_mean, kf_covariance = kf.initiate(pose_arr)

        # 如果激活了 2D 跟踪器，用初始掩码对其进行初始化
        if tracker_2D:
            tracker_2D.initialize(color, init_info={"mask": init_mask})

    else:
        # --- 后续帧：跟踪 (Track) ---
        if tracker_2D:
            # 使用 2D 跟踪器获取物体在当前帧的 2D 边界框
            bbox_2d = tracker_2D.track(color)

            if not kf:
                # 如果没有卡尔曼滤波器，直接用 2D 跟踪结果调整上一帧的姿态作为预测
                fp.pose_last = adjust_pose_to_image_point(
                    ob_in_cam=fp.pose_last, K=cam_K,
                    x=bbox_2d[0] + bbox_2d[2] / 2, y=bbox_2d[1] + bbox_2d[3] / 2
                )
            else:
                # 如果有卡尔曼滤波器，进行状态更新和预测
                kf_mean, kf_covariance = kf.update(kf_mean, kf_covariance,
                                                   get_6d_pose_arr_from_mat(fp.pose_last))
                # 从 2D 跟踪结果中获取测量值
                measurement_xy = np.array(get_pose_xy_from_image_point(
                    ob_in_cam=fp.pose_last, K=cam_K,
                    x=bbox_2d[0] + bbox_2d[2] / 2, y=bbox_2d[1] + bbox_2d[3] / 2
                ))
                kf_mean, kf_covariance = kf.update_from_xy(kf_mean, kf_covariance, measurement_xy)
                # 使用滤波后的结果作为姿态预测
                fp.pose_last = torch.from_numpy(get_mat_from_6d_pose_arr(kf_mean[:6])).unsqueeze(0).to(
                    fp.pose_last.device)

        # 调用 FoundationPose 的 track_one 方法进行姿态跟踪和精炼
        pose = fp.track_one(rgb=color, depth=depth, K=cam_K, iteration=track_refine_iter)

        # 从姿态矩阵中提取数组形式的姿态
        pose_arr = get_6d_pose_arr_from_mat(pose)
        pose_matrix = get_mat_from_6d_pose_arr(pose_arr)

        # # --- 坐标系变换 ---
        # # 将相机坐标系下的物体姿态 (pose_matrix) 转换到机器人基座坐标系下
        # # T_obj2base = T_base_tool0 @ T_cam2gripper @ T_obj2cam
        # T_obj2base = T_base_tool0 @ T_cam2gripper @ pose_matrix
        # position = T_obj2base[:3, 3]
        # # 复制位置并微调 z 轴，可能用于机器人抓取规划
        # position_for_robot = position.copy()
        # position_for_robot[2] += 0.05
        # print(f"第 {i} 帧: 物体在机器人基座坐标系下的位置: {position_for_robot}")

        # 如果同时激活了 2D 跟踪和卡尔曼滤波，执行卡尔曼滤波的预测步骤
        if tracker_2D and kf:
            kf_mean, kf_covariance = kf.predict(kf_mean, kf_covariance)

    return pose_matrix


def obj_track(
        mesh_path: str,
        cam_K: torch.Tensor,
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
    主函数，执行完整的 6D 姿态跟踪流程。
    """
    # 1. 加载 3D 模型 (Mesh)
    mesh, to_origin, bbox = _prepare_3d_mesh(
        mesh_path,
        apply_scale,
        force_apply_color,
    )

    # 2. 实例化 6D 姿态估计器 (FoundationPose)
    fp = _prepare_foundationpose(mesh)

    # 3. 实例化 2D 跟踪器
    if activate_2d_tracker:  # 如果激活，默认使用 Cutie 作为 2D 跟踪器
        tracker_2D = Cutie()
    else:
        tracker_2D = Tracker_2D()  # 否则使用一个空的占位跟踪器

    # 4. 初始化卡尔曼滤波器和姿态序列
    kf = KalmanFilter6D(args.kf_measurement_noise_scale) if activate_kalman_filter else None

    # 5. 设置 RealSense 相机管道
    pipeline, align = _prepare_realsense()

    try:
        # 6. 初始物体检测循环
        init_mask = _get_mask(
            pipeline=pipeline,
            class_name=class_name,
            yolo_model=yolo_model,
            timeout_seconds=120
        )

        #################################################
        # 7. 6D 姿态跟踪主循环
        #################################################
        i: int = 0
        frame_times = []
        pose_seq = []  # 用于存储每一帧的姿态
        kf_mean: np.ndarray | None = None
        kf_covariance: np.ndarray | None = None
        while True:
            frame_start_time = time.time()

            pose = _track_pose(
                i=i,
                pipeline=pipeline,
                align=align,
                fp=fp,
                cam_K=cam_K,
                init_mask=init_mask,
                est_refine_iter=est_refine_iter,
                track_refine_iter=track_refine_iter,
                tracker_2D=tracker_2D,
                kf=kf,
                kf_mean=kf_mean,
                kf_covariance=kf_covariance,
            )

            # --- 可视化 ---
            # 将姿态应用到模型的中心，以便正确绘制边界框
            center_pose = pose @ np.linalg.inv(to_origin)
            # 在图像上绘制 3D 边界框
            vis_color = draw_posed_3d_box(cam_K, img=color, ob_in_cam=center_pose, bbox=bbox)
            # 在图像上绘制 XYZ 坐标轴
            vis_color = draw_xyz_axis(
                vis_color,
                ob_in_cam=center_pose,
                scale=0.1,  # 坐标轴的长度
                K=cam_K,
                thickness=2,
                transparency=0,
                is_input_rgb=True,
            )
            # 显示最终的可视化结果
            cv2.imshow("Pose Tracking", cv2.cvtColor(vis_color, cv2.COLOR_RGB2BGR))
            # 如果按下 'q' 键，则退出循环
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # 记录当前帧的姿态
            pose_seq.append(pose.reshape(4, 4))
            frame_end_time = time.time()
            print(f"帧处理时间: {frame_end_time - frame_start_time:.4f} 秒")
            i += 1
            frame_times.append(frame_end_time - frame_start_time)

    except KeyboardInterrupt:
        # 捕获 Ctrl+C 中断
        print("正在停止...")

    finally:
        # 确保在程序退出时停止相机并关闭窗口
        print("停止 RealSense 管道。")
        pipeline.stop()
        cv2.destroyAllWindows()

    #################################################
    # 8. 保存姿态序列
    #################################################
    if len(pose_seq) > 0:
        pose_seq_array = np.array(pose_seq)
        np.save(
            pose_output_path, pose_seq_array
        )
        print(f"姿态序列已保存到: {pose_output_path}")

    # 清理 GPU 显存
    torch.cuda.empty_cache()


if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="实时 6D 物体姿态跟踪脚本")

    # 定义命令行参数
    parser.add_argument("--mesh_path", type=str, required=True, help="要跟踪物体的 3D 模型文件路径 (.stl, .ply, etc.)")
    parser.add_argument("--pose_output_path", type=str, default="pose_sequence.npy",
                        help="保存姿态序列的 .npy 文件路径")
    # (以下可视化路径参数在当前脚本中未直接使用，但为扩展性保留)
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

    # 解析参数
    args = parser.parse_args()

    # 加载 YOLO 模型
    yolo_model_path = args.yolo_model_path
    yolo_model = YOLO(yolo_model_path)
    class_name = args.class_name

    # 调用主跟踪函数
    obj_track(args.mesh_path,
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

    # 再次清理 GPU 显存
    torch.cuda.empty_cache()
