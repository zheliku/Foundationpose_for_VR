"""
Quest3 双目图像接收与显示

接收 Unity(NetMQ Push) 发送的左右相机 JPEG 数据，
使用 pyzmq 接收后通过 OpenCV 实时显示。

消息格式：multipart [left_jpg, right_jpg]

运行：
    uv run python quest_stereo_receiver.py

在当前系统中的角色：
- 这是 Quest 双目链路的“下游验证节点”。
- 用于确认 Unity 发送端的双目 payload 已正确传输和解码。
- 当前阶段仅做图像传输与显示，不涉及深度估计与 FoundationPose 推理。
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from zmq_utils import PayloadReceiver, StereoJpegDecoder


# ==================== 配置 ====================
LISTEN_PORT = 5557
STATS_INTERVAL = 60
WINDOW_NAME = "Quest3 Stereo (Left | Right)"
SHOW_WINDOW = True


# ==============================================
def main() -> None:
    """主循环：收包 -> 双目解码 -> 拼接显示 -> 统计。"""
    receiver = PayloadReceiver(f"tcp://*:{LISTEN_PORT}", hwm=1, bind=True)
    decoder = StereoJpegDecoder()

    print(f"[StereoReceiver] Listening on tcp://*:{LISTEN_PORT}")
    print("[StereoReceiver] Press 'q' or ESC to exit")

    frame_count = 0
    dropped_by_drain = 0
    start_time = time.perf_counter()
    last_stats_time = start_time
    last_stats_frame_count = 0
    last_stats_dropped_by_drain = 0

    decode_time_acc = 0.0
    compose_time_acc = 0.0
    display_time_acc = 0.0
    has_logged_payload_format = False

    try:
        while True:
            parts = receiver.recv_payload(timeout_ms=100)
            if parts is None:
                continue

            if not has_logged_payload_format:
                has_logged_payload_format = True
                mode = "PackedSingleJpeg" if len(parts) == 1 else "DualJpeg"
                print(f"[StereoReceiver] PayloadParts={len(parts)}, DecodeMode={mode}")

            dropped_by_drain += getattr(receiver, "last_drain_count", 0)

            decode_start = time.perf_counter()
            parsed = decoder.decode(parts)
            decode_time_acc += time.perf_counter() - decode_start
            if parsed is None:
                continue

            left_image, right_image = parsed

            compose_start = time.perf_counter()
            if left_image.shape[0] != right_image.shape[0]:
                target_height = min(left_image.shape[0], right_image.shape[0])
                left_image = cv2.resize(
                    left_image,
                    (
                        int(left_image.shape[1] * target_height / left_image.shape[0]),
                        target_height,
                    ),
                    interpolation=cv2.INTER_LINEAR,
                )
                right_image = cv2.resize(
                    right_image,
                    (
                        int(
                            right_image.shape[1] * target_height / right_image.shape[0]
                        ),
                        target_height,
                    ),
                    interpolation=cv2.INTER_LINEAR,
                )
            compose_time_acc += time.perf_counter() - compose_start

            stereo_view = np.hstack((left_image, right_image))
            display_start = time.perf_counter()
            if SHOW_WINDOW:
                cv2.imshow(WINDOW_NAME, stereo_view)
            display_time_acc += time.perf_counter() - display_start

            frame_count += 1

            if frame_count % STATS_INTERVAL == 0:
                now = time.perf_counter()
                total_elapsed = now - start_time
                interval_elapsed = now - last_stats_time

                total_fps = frame_count / total_elapsed if total_elapsed > 0 else 0.0

                interval_frames = frame_count - last_stats_frame_count
                interval_processed_fps = (
                    interval_frames / interval_elapsed if interval_elapsed > 0 else 0.0
                )

                interval_drain_drop = dropped_by_drain - last_stats_dropped_by_drain
                interval_ingress = interval_frames + interval_drain_drop
                interval_ingress_fps = (
                    interval_ingress / interval_elapsed if interval_elapsed > 0 else 0.0
                )
                drain_drop_rate = (
                    interval_drain_drop / interval_ingress
                    if interval_ingress > 0
                    else 0.0
                )

                avg_decode_ms = (
                    decode_time_acc / interval_frames * 1000
                    if interval_frames > 0
                    else 0.0
                )
                avg_compose_ms = (
                    compose_time_acc / interval_frames * 1000
                    if interval_frames > 0
                    else 0.0
                )
                avg_display_ms = (
                    display_time_acc / interval_frames * 1000
                    if interval_frames > 0
                    else 0.0
                )

                print(
                    "[StereoReceiver] "
                    f"Frames={frame_count}, TotalFPS={total_fps:.1f}, "
                    f"Interval={interval_elapsed:.3f}s, "
                    f"ProcFPS={interval_processed_fps:.1f}, IngressFPS={interval_ingress_fps:.1f}, "
                    f"DrainDrop={interval_drain_drop}, DrainDropRate={drain_drop_rate:.1%}, "
                    f"Decode={avg_decode_ms:.2f}ms, Compose={avg_compose_ms:.2f}ms, Display={avg_display_ms:.2f}ms"
                )

                last_stats_time = now
                last_stats_frame_count = frame_count
                last_stats_dropped_by_drain = dropped_by_drain
                decode_time_acc = 0.0
                compose_time_acc = 0.0
                display_time_acc = 0.0

            if SHOW_WINDOW:
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break

    except KeyboardInterrupt:
        print("\n[StereoReceiver] Stopping...")
    finally:
        elapsed = time.perf_counter() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0.0
        print(
            f"[StereoReceiver] Total Frames={frame_count}, Avg FPS={fps:.1f}, "
            f"DrainDropped={dropped_by_drain}"
        )

        receiver.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
