import pyrealsense2 as rs
import numpy as np
import cv2
import os
import json
from datetime import datetime


def setup_realsense():
    """配置并启动RealSense摄像头"""
    print("RealSense")

    # 创建管道
    pipeline = rs.pipeline()
    config = rs.config()

    # 启用彩色和深度流
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # 启动管道
    profile = pipeline.start(config)

    # 获取深度传感器和深度比例
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"深度比例系数: {depth_scale}")

    # 创建对齐对象（深度对齐到彩色）
    align_to = rs.stream.color
    align = rs.align(align_to)

    return pipeline, align, depth_scale


def capture_rgbd_data(output_dir, num_frames=1000, capture_interval=0.5):
    """
    采集RGB-D数据

    参数:
        output_dir: 输出目录
        num_frames: 要采集的帧数
        capture_interval: 采集间隔（秒）
    """
    # 创建输出目录
    rgb_dir = os.path.join(output_dir, "rgb")
    depth_dir = os.path.join(output_dir, "depth")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    # 设置摄像头
    pipeline, align, depth_scale = setup_realsense()

    # 获取相机内参
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    intr = color_frame.profile.as_video_stream_profile().intrinsics
    camera_params = {
        "width": intr.width,
        "height": intr.height,
        "fx": intr.fx,
        "fy": intr.fy,
        "ppx": intr.ppx,
        "ppy": intr.ppy,
        "depth_scale": depth_scale
    }

    # 保存相机参数
    with open(os.path.join(output_dir, "camera_params.json"), "w") as f:
        json.dump(camera_params, f, indent=4)

    print(f"开始采集 {num_frames} 帧数据，间隔 {capture_interval}秒...")

    try:
        frame_count = 0
        while frame_count < num_frames:
            # 等待帧
            frames = pipeline.wait_for_frames()

            # 对齐深度帧到彩色帧
            aligned_frames = align.process(frames)

            # 获取对齐后的帧
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            # 转换为numpy数组
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # cv2.imwrite("test.png", depth_image)

            # 生成时间戳
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

            # 保存图像
            rgb_filename = os.path.join(rgb_dir, f"{frame_count:d5}.png")
            # rgb_filename = os.path.join(rgb_dir, f"frame_{timestamp}.png")
            depth_filename = os.path.join(depth_dir, f"{frame_count:d5}.png")
            # depth_filename = os.path.join(depth_dir, f"frame_{timestamp}.png")

            cv2.imwrite(rgb_filename, color_image)
            cv2.imwrite(depth_filename, depth_image)

            print(f"已采集帧 {frame_count + 1}/{num_frames}")
            frame_count += 1

            # 显示预览
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            # 水平堆叠图像
            images = np.hstack((color_image, depth_colormap))

            # 显示图像
            cv2.namedWindow('RealSense Capture', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense Captrue', images)

            # 等待间隔或按键
            key = cv2.waitKey(int(capture_interval * 1000))
            if key == ord('q'):
                break

    finally:
        # 停止管道
        pipeline.stop()
        cv2.destroyAllWindows()
        print("采集完成!")


if __name__ == "__main__":
    # 配置输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_directory = f"FoundationPose/demo_data/mydata_{timestamp}"

    # 开始采集
    capture_rgbd_data(output_directory, num_frames=300, capture_interval=0.001)
