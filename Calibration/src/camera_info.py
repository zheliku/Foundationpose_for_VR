import pyrealsense2 as rs
import numpy as np

# 配置RealSense相机
pipeline = rs.pipeline()  # pyright: ignore
config = rs.config()  # pyright: ignore

# 配置彩色流
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)  # pyright: ignore

# 启动相机
print("启动RealSense相机...")
profile = pipeline.start(config)

# 从相机获取内参
color_stream = profile.get_stream(rs.stream.color)  # pyright: ignore
intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

# 构建相机内参矩阵
camera_matrix_np = np.array(
    [[intrinsics.fx, 0, intrinsics.ppx], [0, intrinsics.fy, intrinsics.ppy], [0, 0, 1]]
)

# 获取畸变系数 (RealSense使用Brown-Conrady模型)
dist_coeffs_np = np.array(intrinsics.coeffs)

print(f"相机内参矩阵:\n{camera_matrix_np}")
print(f"畸变系数: {dist_coeffs_np}")
