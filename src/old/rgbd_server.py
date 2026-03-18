"""
RGBD 图像中继服务器

接收来自 RealSense 的 RGBD 图像，直接转发给 Unity。
统计 FPS、帧间隔和模型推理时间。
同时提供网络延迟探测服务（RTT 测量）。

运行在服务器上：
    uv run python relay_server.py

按 Ctrl+C 停止
"""

from zmq_utils import LatencyProbe, LatencyTracker, RGBDPublisher, RGBDReceiver

# ==================== 配置 ====================
RECEIVE_PORT = 5555  # 接收 RealSense 图像的端口
PUBLISH_PORT = 5556  # 发布给 Unity 的端口
LATENCY_PORT = 5560  # 网络延迟探测端口
RGBD_TOPIC = "rgbd"  # RGBD 主题
STATS_INTERVAL = 60  # 每隔多少帧打印统计信息
# ==============================================


def main() -> None:
    # 初始化接收器和发布器
    receiver = RGBDReceiver(f"tcp://*:{RECEIVE_PORT}", hwm=2, bind=True)
    publisher = RGBDPublisher(f"tcp://*:{PUBLISH_PORT}", hwm=1, bind=True)

    # 启动网络延迟探测服务（后台运行）
    latency_probe = LatencyProbe.create_server(f"tcp://*:{LATENCY_PORT}")
    latency_probe.start()

    # 创建模型推理时间追踪器
    tracker = LatencyTracker(window_size=100)

    print(f"[Server] Waiting for RGBD images on port {RECEIVE_PORT}...")
    print(f"[Server] Publishing to Unity on port {PUBLISH_PORT}...")
    print(f"[Server] Latency probe available on port {LATENCY_PORT}")

    frame_count = 0
    try:
        while True:
            # 接收 RGBD 图像
            result = receiver.recv_rgbd(timeout_ms=100)
            if result is None:
                continue

            color, depth = result

            # === 模型推理区域（未来添加大模型处理）===
            # with tracker.track_model():
            #     output = model.predict(color)
            # ==========================================

            # 转发给 Unity
            if publisher.publish_rgbd(RGBD_TOPIC, color, depth, quality=80):
                frame_count += 1
                if frame_count % STATS_INTERVAL == 0:
                    stats = receiver.get_stats()
                    model_stats = tracker.model_stats.get_stats()
                    print(
                        f"[Server] Frames: {frame_count} | "
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
