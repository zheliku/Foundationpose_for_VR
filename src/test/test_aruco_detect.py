import sys
import time
from typing import Tuple

import numpy as np
import cv2

try:
    import pyrealsense2 as rs
except ImportError as e:
    print("请先安装 Intel RealSense Python SDK: pip install pyrealsense2")
    sys.exit(1)

# ============== 可调参数 ==============
ARUCO_DICT = cv2.aruco.DICT_4X4_50  # 你也可以改成 DICT_6X6_250 / DICT_7X7_1000 等
MARKER_LENGTH_M = 0.045  # 单个 ArUco 标记实物边长（米）, 用于姿态估计
SHOW_POSE_AXES = True  # 如果为 True 且拿到相机内参，将绘制姿态坐标轴


def build_rs_pipeline(width=1280, height=720, fps=30) -> Tuple[rs.pipeline, rs.align]:
    """初始化 RealSense 流水线（仅彩色），并返回对齐器（对齐到彩色）"""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    # 只要彩色即可；如果你还需要深度，可再 enable_stream depth 并使用 align。
    align = rs.align(rs.stream.color)
    pipeline.start(config)
    return pipeline, align


def rs_intrinsics_to_cv(camera_intrinsics: rs.intrinsics) -> Tuple[np.ndarray, np.ndarray]:
    """
    将 RealSense 的内参转换为 OpenCV 相机矩阵与畸变系数。
    RealSense 提供的 coeffs 顺序一般是 [k1, k2, p1, p2, k3, k4, k5, k6]（按模型不同）
    OpenCV 常用前5项 Brown-Conrady: k1, k2, p1, p2, k3
    """
    fx, fy = camera_intrinsics.fx, camera_intrinsics.fy
    cx, cy = camera_intrinsics.ppx, camera_intrinsics.ppy
    cam_mtx = np.array([[fx, 0, cx],
                        [0, fy, cy],
                        [0, 0, 1]], dtype=np.float32)
    coeffs = list(camera_intrinsics.coeffs)
    if len(coeffs) < 5:
        # 没有畸变或数量不足时，退化为零畸变
        dist = np.zeros((1, 5), dtype=np.float32)
    else:
        dist = np.array([coeffs[:5]], dtype=np.float32)
    return cam_mtx, dist


def main():
    # 1) 准备 ArUco 检测器（OpenCV 4.11 新式 API：ArucoDetector）
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    params = cv2.aruco.DetectorParameters()  # 4.11 新写法：直接构造，不再用 *_create()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX  # 亚像素角点精炼
    detector = cv2.aruco.ArucoDetector(dictionary, params)

    # 2) 启动 RealSense
    pipeline, align = build_rs_pipeline()
    profile = pipeline.get_active_profile()
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_stream.get_intrinsics()

    # 3) 将 RealSense 内参转换为 OpenCV 格式（供姿态估计/画坐标轴使用）
    camera_matrix, dist_coeffs = rs_intrinsics_to_cv(intr)
    has_intrinsics = camera_matrix is not None and dist_coeffs is not None

    print("按 q 退出窗口。")

    # 小工具：FPS 估计
    last_t = time.time()
    fps = 0.0

    try:
        while True:
            frames = pipeline.wait_for_frames()
            # 如果启用了深度流，这里可以 align 到彩色；只彩色时对齐不改变内容
            frames = align.process(frames)
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())

            # 4) 检测 ArUco 标记（新版 API：detector.detectMarkers）
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, rejected = detector.detectMarkers(frame)

            # 可视化检测结果
            if ids is not None and len(ids) > 0:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)

                # 5) （可选）姿态估计与绘制坐标轴
                if SHOW_POSE_AXES and has_intrinsics and MARKER_LENGTH_M > 0:
                    # 估计每个标记的 rvec / tvec（按单标记假设）
                    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                        corners, MARKER_LENGTH_M, camera_matrix, dist_coeffs
                    )
                    for rvec, tvec in zip(rvecs, tvecs):
                        # 画坐标轴（长度设为标记边长的 0.75 倍）
                        cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs,
                                          rvec, tvec, MARKER_LENGTH_M * 0.75)

            # 6) 叠加 FPS 文本
            now = time.time()
            dt = max(now - last_t, 1e-6)
            fps = 0.9 * fps + 0.1 * (1.0 / dt)  # EMA 平滑
            last_t = now
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # 显示
            cv2.imshow("RealSense ArUco (OpenCV 4.11)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
