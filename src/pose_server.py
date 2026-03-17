"""
6D 位姿追踪中继服务器

接收来自 RealSense 的 RGBD 图像，进行位姿追踪后转发给 Unity。
包含两个阶段：
  - 检测阶段：SAM3 检测目标，转发原始图像
  - 追踪阶段：FoundationPose 追踪，转发位姿 + 带标记图像

运行在服务器上：
    uv run python relay_server.py

按 Ctrl+C 停止
"""

import numpy as np
from pathlib import Path

from pose_tracker_api import PoseTracker
from zmq_utils import LatencyProbe, LatencyTracker, RGBDReceiver, TrackingPublisher

# 获取脚本所在目录的绝对路径
SCRIPT_DIR = Path(__file__).parent.resolve()

# ==================== 配置 ====================
RECEIVE_PORT = 5555  # 接收 RealSense 图像的端口
PUBLISH_PORT = 5556  # 发布给 Unity 的端口
LATENCY_PORT = 5560  # 网络延迟探测端口
TRACKING_TOPIC = "tracking"  # 追踪数据主题
STATS_INTERVAL = 30  # 每隔多少帧打印统计信息

# ============= 追踪器配置（请根据实际情况修改）=============
MESH_PATH = str(
    SCRIPT_DIR / "../data/online/cube/mesh/cube.stl"
)  # 目标物体 3D 模型路径
CAM_K = [
    [609.454406738281, 0.0, 322.085693359375],
    [0.0, 608.353942871094, 242.440429687500],
    [0.0, 0.0, 1.0],
]  # 相机内参
TEXT_PROMPT = "white cube"  # SAM3 文本提示

# 可选配置
APPLY_SCALE = 1.0  # 模型缩放因子
FORCE_APPLY_COLOR = True  # 为无色模型强制着色（匹配 on_demo3）
APPLY_COLOR = [0, 159, 237]  # 强制着色时使用的颜色 (RGB)
SAM3_CONFIDENCE = 0.8  # SAM3 检测置信度阈值（匹配 on_demo3）
EST_REFINE_ITER = 5  # 初始化精炼迭代次数
TRACK_REFINE_ITER = 2  # 追踪精炼迭代次数
ACTIVATE_2D_TRACKER = True  # 是否启用 2D 跟踪器辅助
DEBUG_OUTPUT_DIR = str(SCRIPT_DIR / "../data/debug")  # 调试输出目录
# ==============================================


def main() -> None:
    # 初始化接收器和发布器
    receiver = RGBDReceiver(f"tcp://*:{RECEIVE_PORT}", hwm=2, bind=True)
    publisher = TrackingPublisher(f"tcp://*:{PUBLISH_PORT}", hwm=1, bind=True)

    # 启动网络延迟探测服务（后台运行）
    latency_probe = LatencyProbe.create_server(f"tcp://*:{LATENCY_PORT}")
    latency_probe.start()

    # 创建模型推理时间追踪器
    tracker = LatencyTracker(window_size=100)

    # 初始化位姿追踪器
    print("[Server] 正在初始化位姿追踪器...")
    pose_tracker = PoseTracker(
        mesh_path=MESH_PATH,
        cam_K=np.array(CAM_K),
        text_prompt=TEXT_PROMPT,
        apply_scale=APPLY_SCALE,
        force_apply_color=FORCE_APPLY_COLOR,
        apply_color=APPLY_COLOR,
        sam3_confidence_threshold=SAM3_CONFIDENCE,
        est_refine_iter=EST_REFINE_ITER,
        track_refine_iter=TRACK_REFINE_ITER,
        activate_2d_tracker=ACTIVATE_2D_TRACKER,
        debug_output_dir=DEBUG_OUTPUT_DIR,
    )

    print(f"[Server] Waiting for RGBD images on port {RECEIVE_PORT}...")
    print(f"[Server] Publishing tracking data on port {PUBLISH_PORT}...")
    print(f"[Server] Latency probe available on port {LATENCY_PORT}")

    frame_count = 0
    try:
        while True:
            # 接收 RGBD 图像
            result = receiver.recv_rgbd(timeout_ms=100)
            if result is None:
                continue

            color, depth = result

            # 深度转换为米
            depth_m = depth.astype(np.float64) / 1000.0
            depth_m[(depth_m < 0.001) | (depth_m >= np.inf)] = 0

            # === 位姿追踪推理 ===
            with tracker.track_model():
                tracking_result = pose_tracker.process_frame(color, depth_m)
            # ====================

            # 转发给 Unity
            if publisher.publish_tracking(
                TRACKING_TOPIC,
                phase=tracking_result.phase.value,
                color=tracking_result.color,
                pose_matrix=tracking_result.pose_matrix,
                quality=80,
            ):
                frame_count += 1
                if frame_count % STATS_INTERVAL == 0:
                    stats = receiver.get_stats()
                    model_stats = tracker.model_stats.get_stats()
                    phase_str = "TRACKING" if pose_tracker.is_tracking else "DETECTING"
                    print(
                        f"[Server] Frames: {frame_count} | "
                        f"Phase: {phase_str} | "
                        f"FPS: {stats['fps']:.1f} | "
                        f"Interval: {stats['interval_avg_ms']:.1f}ms (±{stats['interval_std_ms']:.1f}) | "
                        f"Model: {model_stats['avg']:.1f}ms"
                    )

    except KeyboardInterrupt:
        print("\n[Server] Stopping...")
        stats = receiver.get_stats()
        model_stats = tracker.model_stats.get_stats()
        print("\n=== Final Statistics ===")
        print(f"Total Frames: {int(stats['frame_count'])}")
        print(f"Average FPS: {stats['fps']:.1f}")
        print(f"Avg Frame Interval: {stats['interval_avg_ms']:.1f}ms")
        print(f"Avg Model Time: {model_stats['avg']:.1f}ms")
    finally:
        latency_probe.stop()
        receiver.close()
        publisher.close()


if __name__ == "__main__":
    main()
