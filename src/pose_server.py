"""
6D 位姿追踪中继服务器

接收来自 RealSense 的 RGBD 图像，进行位姿追踪后转发给 Unity。
包含两个阶段：
  - 检测阶段：SAM3 检测目标，转发原始图像
  - 追踪阶段：FoundationPose 追踪，转发位姿 + 带标记图像

运行在服务器上：
    uv run python relay_server.py

按 Ctrl+C 停止

架构定位：
- 当前主链路核心中枢（接收 RGBD -> FoundationPose -> 发布 tracking）。
- 也是后续“Quest 双目 + 深度估计”接入 FoundationPose 的目标承载点。

分层职责：
- communicate 层：PayloadReceiver / PayloadSender 负责网络收发。
- payload 层：RGBDDecoder / TrackingEncoder 负责协议编解码。
- 业务层：PoseTracker 负责检测与追踪推理。
"""

import numpy as np
import time
from pathlib import Path

from pose_tracker_api import PoseTracker
from zmq_utils import (
    LatencyProbe,
    LatencyStats,
    LatencyTracker,
    PayloadReceiver,
    PayloadSender,
    RGBDDecoder,
    TrackingDecoder,
    TrackingEncoder,
)

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
    """服务主循环：收包解码 -> 推理 -> 编码发布 -> 打印统计。"""
    # 初始化接收器和发布器
    receiver = PayloadReceiver(f"tcp://*:{RECEIVE_PORT}", hwm=2, bind=True)
    rgbd_decoder = RGBDDecoder()
    publisher = PayloadSender(
        f"tcp://*:{PUBLISH_PORT}", hwm=1, bind=True, send_topic=True
    )
    tracking_encoder = TrackingEncoder()
    tracking_decoder = TrackingDecoder()

    # 帧率统计
    start_time = time.perf_counter()
    last_recv_time = 0.0
    interval_stats = LatencyStats(window_size=100)

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
            parts = receiver.recv_payload(timeout_ms=100)
            if parts is None:
                continue

            result = rgbd_decoder.decode(parts)
            if result is None:
                continue

            now = time.perf_counter()
            if last_recv_time > 0:
                interval_stats.record((now - last_recv_time) * 1000.0)
            last_recv_time = now

            color, depth = result

            # 深度转换为米
            depth_m = depth.astype(np.float64) / 1000.0
            depth_m[(depth_m < 0.001) | (depth_m >= np.inf)] = 0

            # === 位姿追踪推理 ===
            with tracker.track_model():
                tracking_result = pose_tracker.process_frame(color, depth_m)
            # ====================

            payload = tracking_encoder.encode(
                phase=tracking_result.phase.value,
                color=tracking_result.color,
                pose_matrix=tracking_result.pose_matrix,
                quality=80,
            )

            # 转发给 Unity
            if payload is None:
                continue

            if tracking_decoder.decode(payload) is None:
                continue

            if publisher.send_payload(payload, topic=TRACKING_TOPIC):
                frame_count += 1
                if frame_count % STATS_INTERVAL == 0:
                    elapsed = max(now - start_time, 1e-6)
                    fps = frame_count / elapsed
                    stats = interval_stats.get_stats()
                    model_stats = tracker.model_stats.get_stats()
                    phase_str = "TRACKING" if pose_tracker.is_tracking else "DETECTING"
                    print(
                        f"[Server] Frames: {frame_count} | "
                        f"Phase: {phase_str} | "
                        f"FPS: {fps:.1f} | "
                        f"Interval: {stats['avg']:.1f}ms (±{stats['std']:.1f}) | "
                        f"Model: {model_stats['avg']:.1f}ms"
                    )

    except KeyboardInterrupt:
        print("\n[Server] Stopping...")
        elapsed = max(time.perf_counter() - start_time, 1e-6)
        fps = frame_count / elapsed
        stats = interval_stats.get_stats()
        model_stats = tracker.model_stats.get_stats()
        print("\n=== Final Statistics ===")
        print(f"Total Frames: {frame_count}")
        print(f"Average FPS: {fps:.1f}")
        print(f"Avg Frame Interval: {stats['avg']:.1f}ms")
        print(f"Avg Model Time: {model_stats['avg']:.1f}ms")
    finally:
        latency_probe.stop()
        receiver.close()
        publisher.close()


if __name__ == "__main__":
    main()
