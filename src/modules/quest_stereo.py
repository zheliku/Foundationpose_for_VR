"""
Quest 双目网络输入 API（模块化版）

设计目标：
1. 接受常规网络参数输入（监听地址、端口、超时、HWM）。
2. 通过统一方法提供双目图像输出（stereo）。
3. 封装传输与解码细节，保持与 realsense.py 相近的使用体验。

说明：
- 输入来自 Unity 的 QuestStereoEncoder + PayloadSender。
- 协议兼容：
  - Dual: [left_jpg, right_jpg]
  - Packed: [packed_stereo_jpg]
    - Packed+Meta: [packed_stereo_jpg, metadata_json]
    - Dual+Meta: [left_jpg, right_jpg, metadata_json]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    # 常规包导入路径。
    from zmq_utils import PayloadReceiver, StereoDecoder
    from zmq_utils.payload.message.stereo import QuestStereoMsg
except ModuleNotFoundError:
    # 兼容直接运行 src/modules/quest_stereo.py 的场景。
    src_root = Path(__file__).resolve().parents[1]
    if str(src_root) not in sys.path:
        sys.path.append(str(src_root))
    from zmq_utils import PayloadReceiver, StereoDecoder
    from zmq_utils.payload.message.stereo import QuestStereoMsg


class QuestStereoCamera:
    """
    Quest 双目网络输入最小可用 API。

    使用方式（示例）：
        cam = QuestStereoCamera(listen_port=5557)
        cam.start()

        frame = cam.get_stereo_frames()
        if frame is not None:
            left = frame.left
            right = frame.right

        cam.stop()

    约定：
    - `start()` 后才可调用取帧方法。
    - `get_stereo_frames()` 返回 None 表示超时或解码失败。
    - `stop()` 可重复调用（幂等）。
    """

    # 输入配置。
    listen_host: str = "*"  # 监听地址。
    listen_port: int = 5557  # 监听端口。
    hwm: int = 1  # ZMQ 高水位。
    timeout_ms: int = 100  # 默认接收超时（毫秒）。
    endpoint: str = ""  # 完整 ZMQ endpoint。

    # 运行时对象。
    receiver: PayloadReceiver | None = None  # 负载接收器。
    decoder: StereoDecoder  # 双目 JPEG 解码器（__init__ 中创建）。

    # 运行状态标志。
    _started: bool = False  # 是否已经启动接收。
    _has_logged_payload_format: bool = False  # 是否已打印过负载格式日志。

    # 统计计数器。
    _received_count: int = 0  # 收到 payload 次数。
    _decoded_count: int = 0  # 成功解码次数。
    _decode_fail_count: int = 0  # 解码失败次数。
    _drained_count: int = 0  # 队列清空累计数量。
    _decode_time_acc_ms: float = 0.0  # 累计解码耗时（毫秒）。
    _sender_meta_count: int = 0  # 含发送端元数据的帧数。
    _sender_no_meta_count: int = 0  # 不含元数据的帧数。
    _sender_gap_count: int = 0  # 发送端帧号累计缺口。
    _last_sender_frame_id: int | None = None  # 上一帧发送端帧号。
    _last_sender_mono_ms: float | None = None  # 上一帧发送端单调时钟。
    _sender_fps_ema: float = 0.0  # 发送端帧率估计（EMA）。
    _sender_delta_min_ms: float | None = None  # raw_delta 的历史最小值（偏移基线）。
    _sender_delta_raw_ema_ms: float = 0.0  # raw_delta 的平滑值。
    _sender_delay_est_ema_ms: float = 0.0  # 估计延迟的平滑值。

    def __init__(
        self,
        listen_host: str = "*",
        listen_port: int = 5557,
        hwm: int = 1,
        timeout_ms: int = 100,
    ) -> None:
        """
        初始化 Quest 双目接收器。

        参数：
        - listen_host: 监听地址。
        - listen_port: 监听端口。
        - hwm: ZMQ 高水位。
        - timeout_ms: 默认接收超时（毫秒）。

        初始化流程：
        1. 保存网络参数。
        2. 生成 endpoint。
        3. 创建双目 JPEG 解码器。
        """
        self.listen_host = str(listen_host)
        self.listen_port = int(listen_port)
        self.hwm = int(hwm)
        self.timeout_ms = int(timeout_ms)

        self.endpoint = f"tcp://{self.listen_host}:{self.listen_port}"

        self.decoder = StereoDecoder()

    def start(self) -> None:
        """启动网络接收器。"""
        if self._started:
            return

        self.receiver = PayloadReceiver(
            self.endpoint,
            hwm=self.hwm,
            bind=True,
        )
        self.receiver.start_latest_buffer(poll_timeout_ms=10)

        self._started = True
        print(f"[QuestStereoCamera] Listening on {self.endpoint}")

    def stop(self) -> None:
        """停止接收并释放资源。"""
        if not self._started:
            return

        self._started = False

        if self.receiver is not None:
            self.receiver.close()
            self.receiver = None

    def get_stereo_frames(
        self,
        timeout_ms: int | None = None,
    ) -> QuestStereoMsg | None:
        """
        获取最新一组 Quest 双目图像。

        返回：
        - QuestStereoMsg: 成功解码
        - None: 超时、空包或解码失败
        """
        if not self._started or self.receiver is None:
            raise RuntimeError("QuestStereoCamera 尚未启动，请先调用 start()。")

        wait_ms = self.timeout_ms if timeout_ms is None else int(timeout_ms)
        popped = self.receiver.pop_latest_payload(timeout_ms=wait_ms)
        if popped is None:
            return None

        parts, drained = popped
        self._received_count += 1
        self._drained_count += int(drained)

        if not self._has_logged_payload_format:
            self._has_logged_payload_format = True
            mode = self._infer_payload_mode(parts)
            print(
                "[QuestStereoCamera] " f"PayloadParts={len(parts)}, DecodeMode={mode}"
            )

        decode_start = time.perf_counter()
        # 将 multipart 解码成完整 QuestStereoMsg。
        message = self.decoder.decode(parts)
        self._decode_time_acc_ms += (time.perf_counter() - decode_start) * 1000.0

        if message is None:
            self._decode_fail_count += 1
            return None
        self._decoded_count += 1

        # 本机接收时刻（单调时钟，毫秒）。
        local_rx_ms = time.perf_counter() * 1000.0
        sender_mono_ms = message.sender_mono_ms
        sender_frame_id = message.frame_id
        sender_delta_raw_ms: float | None = None
        sender_delay_est_ms: float | None = None

        if message.has_metadata:
            self._sender_meta_count += 1

            if sender_frame_id is not None:
                prev_sender_frame_id = self._last_sender_frame_id
                if (
                    self._last_sender_frame_id is not None
                    and sender_frame_id > self._last_sender_frame_id + 1
                ):
                    self._sender_gap_count += (
                        sender_frame_id - self._last_sender_frame_id - 1
                    )
                self._last_sender_frame_id = sender_frame_id
            else:
                prev_sender_frame_id = None

            if sender_mono_ms is not None:
                # raw_delta 同时包含网络延迟和两机时钟偏移（偏移可能很大）。
                sender_delta_raw_ms = local_rx_ms - sender_mono_ms

                alpha = 0.15
                self._sender_delta_raw_ema_ms = (
                    sender_delta_raw_ms
                    if self._sender_delta_raw_ema_ms <= 0.0
                    else self._sender_delta_raw_ema_ms * (1.0 - alpha)
                    + sender_delta_raw_ms * alpha
                )

                # 基线法：用历史最小 raw_delta 作为“时钟偏移+最小链路时延”近似基线。
                # estimated = raw_delta - min_delta，可在时钟不同步条件下稳定观察“额外排队延迟”。
                if self._sender_delta_min_ms is None:
                    self._sender_delta_min_ms = sender_delta_raw_ms
                else:
                    self._sender_delta_min_ms = min(
                        self._sender_delta_min_ms, sender_delta_raw_ms
                    )

                sender_delay_est_ms = max(
                    sender_delta_raw_ms - self._sender_delta_min_ms, 0.0
                )
                self._sender_delay_est_ema_ms = (
                    sender_delay_est_ms
                    if self._sender_delay_est_ema_ms <= 0.0
                    else self._sender_delay_est_ema_ms * (1.0 - alpha)
                    + sender_delay_est_ms * alpha
                )

                # 依据发送端元数据估计“实际发送帧率”，用于诊断 frame_id 跳变是否异常。
                if (
                    prev_sender_frame_id is not None
                    and sender_frame_id is not None
                    and self._last_sender_mono_ms is not None
                ):
                    frame_delta = sender_frame_id - prev_sender_frame_id
                    mono_delta_ms = sender_mono_ms - self._last_sender_mono_ms
                    if frame_delta > 0 and mono_delta_ms > 1e-6:
                        sender_fps_inst = frame_delta * 1000.0 / mono_delta_ms
                        self._sender_fps_ema = (
                            sender_fps_inst
                            if self._sender_fps_ema <= 0.0
                            else self._sender_fps_ema * (1.0 - alpha)
                            + sender_fps_inst * alpha
                        )

                self._last_sender_mono_ms = sender_mono_ms
        else:
            self._sender_no_meta_count += 1

        # 将链路诊断字段回填到消息对象，供下游模块直接消费。
        message.timestamp_ms = local_rx_ms
        message.sender_delta_raw_ms = sender_delta_raw_ms
        message.sender_delay_est_ms = sender_delay_est_ms
        return message

    @staticmethod
    def _infer_payload_mode(parts: list[bytes]) -> str:
        # 通过消息结构反推当前传输模式，便于首次日志诊断。
        message = QuestStereoMsg.from_parts(parts)
        if message is None:
            return f"Unknown({len(parts)} parts)"

        if message.is_packed:
            return (
                "PackedSingleJpeg+Meta" if message.has_metadata else "PackedSingleJpeg"
            )

        return "DualJpeg+Meta" if message.has_metadata else "DualJpeg"

    def get_stats(self) -> dict[str, float | int]:
        """返回当前累计统计。"""
        avg_decode_ms = (
            self._decode_time_acc_ms / self._received_count
            if self._received_count > 0
            else 0.0
        )

        return {
            "received": self._received_count,
            "decoded": self._decoded_count,
            "decode_failed": self._decode_fail_count,
            "drained": self._drained_count,
            "avg_decode_ms": avg_decode_ms,
            "sender_meta": self._sender_meta_count,
            "sender_no_meta": self._sender_no_meta_count,
            "sender_gap": self._sender_gap_count,
            "sender_fps": self._sender_fps_ema,
            "sender_min_delta_ms": self._sender_delta_min_ms or 0.0,
            "sender_raw_delta_ms": self._sender_delta_raw_ema_ms,
            "sender_est_delay_ms": self._sender_delay_est_ema_ms,
        }

    def __enter__(self) -> "QuestStereoCamera":
        """上下文管理：进入 with 时自动启动。"""
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """上下文管理：离开 with 时自动停止。"""
        self.stop()


if __name__ == "__main__":
    """
    OpenCV 可视化示例：
    1) 启动 Quest 双目接收器。
    2) 循环读取最新双目图像。
    3) 显示左右拼接图，并打印基础统计。
    4) 按 q 或 ESC 退出并释放资源。
    """

    LISTEN_PORT = 5557
    STATS_INTERVAL = 60
    WINDOW_NAME = "Quest Stereo (Left | Right)"

    camera = QuestStereoCamera(listen_port=LISTEN_PORT, hwm=1, timeout_ms=100)

    frame_count = 0
    start_time = time.perf_counter()
    last_stats_time = start_time
    last_stats_frame_count = 0

    try:
        camera.start()
        print("窗口已打开，按 q 或 ESC 退出。")
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

        while True:
            frame = camera.get_stereo_frames()
            if frame is None:
                continue

            left = frame.left
            right = frame.right
            if left is None or right is None:
                continue

            if left.shape[0] != right.shape[0]:
                target_height = min(left.shape[0], right.shape[0])
                left = cv2.resize(
                    left,
                    (
                        int(left.shape[1] * target_height / left.shape[0]),
                        target_height,
                    ),
                    interpolation=cv2.INTER_LINEAR,
                )
                right = cv2.resize(
                    right,
                    (
                        int(right.shape[1] * target_height / right.shape[0]),
                        target_height,
                    ),
                    interpolation=cv2.INTER_LINEAR,
                )

            stereo_view = np.hstack((left, right))
            cv2.imshow(WINDOW_NAME, stereo_view)

            frame_count += 1
            if frame_count % STATS_INTERVAL == 0:
                now = time.perf_counter()
                total_elapsed = max(now - start_time, 1e-6)
                interval_elapsed = max(now - last_stats_time, 1e-6)

                total_fps = frame_count / total_elapsed
                interval_frames = frame_count - last_stats_frame_count
                interval_fps = interval_frames / interval_elapsed

                stats = camera.get_stats()
                print(
                    "[QuestStereoCamera] "
                    f"Frames={frame_count}, TotalFPS={total_fps:.1f}, "
                    f"IntervalFPS={interval_fps:.1f}, "
                    f"Received={stats['received']}, Decoded={stats['decoded']}, "
                    f"DecodeFailed={stats['decode_failed']}, Drained={stats['drained']}, "
                    f"AvgDecode={float(stats['avg_decode_ms']):.2f}ms"
                )

                last_stats_time = now
                last_stats_frame_count = frame_count

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
    finally:
        camera.stop()
        cv2.destroyAllWindows()
