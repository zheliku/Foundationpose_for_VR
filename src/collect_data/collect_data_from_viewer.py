import pyrealsense2 as rs
import numpy as np
import cv2
import os

# 创建管道
pipeline = rs.pipeline()
config = rs.config()

test_name = "20251021_141149"

# 打开 .bag 文件
config.enable_device_from_file(f'../raw_data/{test_name}.bag')

# 开始管道
pipeline.start(config)

# 创建对齐对象
align = rs.align(rs.stream.color)

# 创建输出目录
depth_output_dir = f'../raw_data/{test_name}/depth'
color_output_dir = f'../raw_data/{test_name}/rgb'
os.makedirs(depth_output_dir, exist_ok=True)
os.makedirs(color_output_dir, exist_ok=True)
max_frames = 300  # 设置最大帧数

try:
    fram_count = 0
    while fram_count < max_frames:
        # 等待新帧
        frames = pipeline.wait_for_frames()

        # 对齐帧
        aligned_frames = align.process(frames)

        # 获取对齐后的深度和颜色帧
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # 转换为 NumPy 数组
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # 将 BGR 转换为 RGB
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        # 保存图像
        cv2.imwrite(os.path.join(depth_output_dir, f'{fram_count:05d}.png'), depth_image)
        cv2.imwrite(os.path.join(color_output_dir, f'{fram_count:05d}.png'), color_image)

        fram_count += 1

        # 显示图像（可选）
        cv2.imshow('Depth Image', depth_image)
        cv2.imshow('Color Image', color_image)

        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # 停止管道
    pipeline.stop()
    cv2.destroyAllWindows()