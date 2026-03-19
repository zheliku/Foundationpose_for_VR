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


# ==============================================
def main() -> None:
    """主循环：收包 -> 双目解码 -> 拼接显示 -> 统计。"""
    receiver = PayloadReceiver(f"tcp://*:{LISTEN_PORT}", hwm=1, bind=True)
    decoder = StereoJpegDecoder()

    print(f"[StereoReceiver] Listening on tcp://*:{LISTEN_PORT}")
    print("[StereoReceiver] Press 'q' or ESC to exit")

    frame_count = 0
    start_time = time.perf_counter()

    try:
        while True:
            parts = receiver.recv_payload(timeout_ms=100)
            if parts is None:
                continue

            parsed = decoder.decode(parts)
            if parsed is None:
                continue

            left_image, right_image = parsed

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

            stereo_view = np.hstack((left_image, right_image))
            cv2.imshow(WINDOW_NAME, stereo_view)

            frame_count += 1
            if frame_count % STATS_INTERVAL == 0:
                elapsed = time.perf_counter() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0.0
                print(f"[StereoReceiver] Frames={frame_count}, FPS={fps:.1f}")

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

    except KeyboardInterrupt:
        print("\n[StereoReceiver] Stopping...")
    finally:
        elapsed = time.perf_counter() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0.0
        print(f"[StereoReceiver] Total Frames={frame_count}, Avg FPS={fps:.1f}")

        receiver.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
