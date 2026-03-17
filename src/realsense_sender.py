"""
RealSense RGBD 图像采集发送端

运行在连接 RealSense 相机的电脑上，采集彩色图像和深度图像并发送到服务器。

使用方法：
    uv run python realsense_sender.py

按 Ctrl+C 停止
"""

import numpy as np
import pyrealsense2 as rs

from zmq_utils import RGBDSender

# ==================== 配置 ====================
SERVER_IP = "172.24.244.81"  # 服务器 IP 地址
# SERVER_IP = "127.0.0.1"  # 服务器 IP 地址
SERVER_PORT = 5555  # 服务器接收端口
JPEG_QUALITY = 80  # JPEG 压缩质量 (1-100)
STATS_INTERVAL = 60  # 每隔多少帧打印统计信息

# RealSense 相机配置
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30  # 改为 30 FPS（与参考脚本一致，减少不必要的帧积压）
# ==============================================


def main() -> None:
    # 初始化 RealSense
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(
        rs.stream.color, CAMERA_WIDTH, CAMERA_HEIGHT, rs.format.bgr8, CAMERA_FPS
    )
    config.enable_stream(
        rs.stream.depth, CAMERA_WIDTH, CAMERA_HEIGHT, rs.format.z16, CAMERA_FPS
    )
    pipeline.start(config)

    # 创建对齐对象，将深度帧对齐到彩色帧（关键！）
    align = rs.align(rs.stream.color)

    print(
        f"[RealSense] Camera started: {CAMERA_WIDTH}x{CAMERA_HEIGHT} @ {CAMERA_FPS} FPS"
    )
    print("[RealSense] Depth alignment to color frame: ENABLED")

    # 连接到服务器（HWM=1 减少积压延迟）
    sender = RGBDSender(f"tcp://{SERVER_IP}:{SERVER_PORT}", hwm=1, bind=False)

    frame_count = 0
    try:
        while True:
            # 获取帧
            frames = pipeline.wait_for_frames()

            # 对齐深度帧到彩色帧（关键！确保深度和彩色像素坐标一致）
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            # 转换为 numpy 数组
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # 发送 RGBD 图像
            if sender.send_rgbd(color_image, depth_image, quality=JPEG_QUALITY):
                frame_count += 1
                if frame_count % STATS_INTERVAL == 0:
                    print(f"[RealSense] Sent {frame_count} RGBD frames")
    except KeyboardInterrupt:
        print("\n[RealSense] Stopping...")
    finally:
        sender.close()
        pipeline.stop()
        print(f"[RealSense] Total frames sent: {frame_count}")


if __name__ == "__main__":
    main()
