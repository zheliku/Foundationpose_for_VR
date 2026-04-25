"""
Quest 多 Topic 网络输入 API

设计目标：
1. 接受网络参数输入（监听地址、端口、超时、HWM）。
2. 通过 SUB 模式同时接收双目图像（quest_stereo）和相机信息（quest_camera_info）。
3. 封装传输与解码细节，对外提供简洁的取帧/取标定接口。

Topic 协议：
- quest_stereo: 双目 JPEG 图像帧（高频）。
- quest_camera_info: 相机静态标定信息（低频，通常不变）。
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from zmq_utils import CameraInfoDecoder, PayloadReceiver, StereoDecoder
    from zmq_utils.payload.message.quest_camera_info_msg import QuestCameraInfoMsg
    from zmq_utils.payload.message.quest_stereo_msg import QuestStereoMsg
except ModuleNotFoundError:
    src_root = Path(__file__).resolve().parents[1]
    if str(src_root) not in sys.path:
        sys.path.append(str(src_root))
    from zmq_utils import CameraInfoDecoder, PayloadReceiver, StereoDecoder
    from zmq_utils.payload.message.quest_camera_info_msg import QuestCameraInfoMsg
    from zmq_utils.payload.message.quest_stereo_msg import QuestStereoMsg


# 默认 topic 名称。
TOPIC_STEREO = "quest_stereo"
TOPIC_CAMERA_INFO = "quest_camera_info"


@dataclass
class QuestStereoCalibration:
    """Quest 双目标定信息（由网络 camera_info 消息构造）。"""

    left_fx: float
    left_fy: float
    left_cx: float
    left_cy: float
    baseline_m: float
    calib_width: int
    calib_height: int

    @classmethod
    def from_camera_info_msg(cls, msg: QuestCameraInfoMsg) -> QuestStereoCalibration:
        """从网络传输的 QuestCameraInfoMsg 构造标定对象。"""
        # calib_width/height 使用 activeArraySize 的宽高。
        width = int(msg.active_right) - int(msg.active_left)
        height = int(msg.active_bottom) - int(msg.active_top)
        if width <= 0 or height <= 0:
            width = msg.sensor_width
            height = msg.sensor_height
        return cls(
            left_fx=msg.left_fx,
            left_fy=msg.left_fy,
            left_cx=msg.left_cx,
            left_cy=msg.left_cy,
            baseline_m=msg.baseline_m,
            calib_width=width,
            calib_height=height,
        )

    def _compute_center_crop_mapping(
        self, width: int, height: int
    ) -> tuple[float, float, float, float]:
        """计算从标定坐标系到运行分辨率的中心裁剪+缩放映射。"""
        src_w = float(max(self.calib_width, 1))
        src_h = float(max(self.calib_height, 1))
        dst_w = float(max(width, 1))
        dst_h = float(max(height, 1))

        src_aspect = src_w / src_h
        dst_aspect = dst_w / dst_h

        crop_x, crop_y, crop_w, crop_h = 0.0, 0.0, src_w, src_h

        if abs(src_aspect - dst_aspect) > 1e-6:
            if src_aspect > dst_aspect:
                crop_w = src_h * dst_aspect
                crop_x = (src_w - crop_w) * 0.5
            else:
                crop_h = src_w / dst_aspect
                crop_y = (src_h - crop_h) * 0.5

        sx = dst_w / max(crop_w, 1e-6)
        sy = dst_h / max(crop_h, 1e-6)
        return crop_x, crop_y, sx, sy

    def scaled_k(
        self, width: int, height: int, assume_center_crop: bool = True
    ) -> np.ndarray:
        """把标定内参映射到运行分辨率下的内参矩阵 K。"""
        if assume_center_crop:
            crop_x, crop_y, sx, sy = self._compute_center_crop_mapping(width, height)
            cx = (self.left_cx - crop_x) * sx
            cy = (self.left_cy - crop_y) * sy
        else:
            sx = width / max(self.calib_width, 1)
            sy = height / max(self.calib_height, 1)
            cx = self.left_cx * sx
            cy = self.left_cy * sy

        return np.array(
            [
                [self.left_fx * sx, 0.0, cx],
                [0.0, self.left_fy * sy, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )


class QuestReceiver:
    """
    Quest 多 Topic 网络接收器。

    使用方式（示例）：
        recv = QuestReceiver(listen_port=5557)
        recv.start()

        # 获取双目帧。
        frame = recv.get_stereo_frames()
        if frame is not None:
            left = frame.left

        # 获取相机信息。
        info = recv.get_camera_info()

        recv.stop()
    """

    # 输入配置。
    listen_host: str = "*"
    listen_port: int = 5557
    hwm: int = 20
    timeout_ms: int = 100
    endpoint: str = ""

    # 订阅 topic 列表。
    topics: list[str]

    # 运行时对象。
    receiver: PayloadReceiver | None = None
    stereo_decoder: StereoDecoder
    camera_info_decoder: CameraInfoDecoder

    # 缓存：按 topic 存储最新消息。
    _latest_stereo: QuestStereoMsg | None = None
    _latest_camera_info: QuestCameraInfoMsg | None = None
    _camera_info_version: int = 0

    # 运行状态。
    _started: bool = False

    # 统计计数器。
    _received_count: int = 0
    _decoded_count: int = 0
    _decode_fail_count: int = 0
    _decode_time_acc_ms: float = 0.0
    _sender_gap_count: int = 0
    _last_sender_frame_id: int | None = None
    _last_sender_mono_ms: float | None = None
    _sender_fps_ema: float = 0.0
    _sender_delta_min_ms: float | None = None
    _sender_delta_raw_ema_ms: float = 0.0
    _sender_delay_est_ema_ms: float = 0.0

    def __init__(
        self,
        listen_host: str = "*",
        listen_port: int = 5557,
        hwm: int = 20,
        timeout_ms: int = 100,
        topics: list[str] | None = None,
    ) -> None:
        self.listen_host = str(listen_host)
        self.listen_port = int(listen_port)
        # HWM 不能太小：Quest stereo 帧约 45KB@37fps，如果 SUB 队列只能缓 1 条，
        # 主循环一次 imshow/解码期间积压的帧会被 ZMQ 底层丢弃，可能导致整条消息流断流。
        self.hwm = int(hwm)
        self.timeout_ms = int(timeout_ms)
        self.endpoint = f"tcp://{self.listen_host}:{self.listen_port}"
        self.topics = topics if topics is not None else [TOPIC_STEREO, TOPIC_CAMERA_INFO]

        self.stereo_decoder = StereoDecoder()
        self.camera_info_decoder = CameraInfoDecoder()
        self._latest_stereo = None
        self._latest_camera_info = None
        self._camera_info_version = 0

    def start(self) -> None:
        """启动网络接收器。"""
        if self._started:
            return

        self.receiver = PayloadReceiver(
            self.endpoint,
            hwm=self.hwm,
            bind=True,
            topics=self.topics,
        )
        self._started = True
        print(f"[QuestReceiver] Listening on {self.endpoint}, topics={self.topics}")

    def stop(self) -> None:
        """停止接收并释放资源。"""
        if not self._started:
            return
        self._started = False
        if self.receiver is not None:
            self.receiver.close()
            self.receiver = None

    def poll_all(self, timeout_ms: int | None = None) -> None:
        """轮询所有 topic，按 topic 分别更新内部缓存。

        使用 recv_all_latest_by_topic 按 topic 分别 drain，
        确保每个 topic 都能获取到最新消息，不会因跨 topic drain 而丢失。
        """
        if not self._started or self.receiver is None:
            raise RuntimeError("QuestReceiver 尚未启动，请先调用 start()。")

        wait_ms = self.timeout_ms if timeout_ms is None else int(timeout_ms)
        latest_by_topic = self.receiver.recv_all_latest_by_topic(timeout_ms=wait_ms)
        if latest_by_topic is None:
            return

        # 按 topic 分发到对应解码器。
        for topic, payload in latest_by_topic.items():
            self._received_count += 1
            if topic == TOPIC_STEREO:
                self._decode_stereo(payload)
            elif topic == TOPIC_CAMERA_INFO:
                self._decode_camera_info(payload)

    def get_stereo_frames(self, timeout_ms: int | None = None) -> QuestStereoMsg | None:
        """获取最新一组 Quest 双目图像。"""
        self.poll_all(timeout_ms)
        return self._latest_stereo

    def get_camera_info(self) -> QuestCameraInfoMsg | None:
        """获取最新相机信息（可能为 None，未收到过时）。"""
        return self._latest_camera_info

    def get_camera_info_version(self) -> int:
        """Return the latest camera_info message version."""
        return self._camera_info_version

    def get_calibration(self) -> QuestStereoCalibration | None:
        """从最新相机信息构造标定对象。未收到时返回 None。"""
        if self._latest_camera_info is None:
            return None
        return QuestStereoCalibration.from_camera_info_msg(self._latest_camera_info)

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
            "avg_decode_ms": avg_decode_ms,
            "sender_gap": self._sender_gap_count,
            "sender_fps": self._sender_fps_ema,
            "sender_min_delta_ms": self._sender_delta_min_ms or 0.0,
            "sender_raw_delta_ms": self._sender_delta_raw_ema_ms,
            "sender_est_delay_ms": self._sender_delay_est_ema_ms,
        }

    def _decode_stereo(self, payload: bytes) -> None:
        """解码双目帧并更新缓存。"""
        decode_start = time.perf_counter()
        message = self.stereo_decoder.decode(payload)
        self._decode_time_acc_ms += (time.perf_counter() - decode_start) * 1000.0

        if message is None:
            self._decode_fail_count += 1
            return
        self._decoded_count += 1

        # 链路诊断。
        local_rx_ms = time.perf_counter() * 1000.0
        self._update_sender_diagnostics(message, local_rx_ms)

        # 回填接收端时间戳和诊断字段。
        message.timestamp_ms = local_rx_ms
        message.sender_delta_raw_ms = (
            local_rx_ms - message.sender_mono_ms if message.sender_mono_ms else None
        )
        message.sender_delay_est_ms = (
            max(message.sender_delta_raw_ms - (self._sender_delta_min_ms or 0.0), 0.0)
            if message.sender_delta_raw_ms is not None
            else None
        )

        self._latest_stereo = message

    def _decode_camera_info(self, payload: bytes) -> None:
        """解码相机信息并更新缓存。"""
        decode_start = time.perf_counter()
        message = self.camera_info_decoder.decode(payload)
        self._decode_time_acc_ms += (time.perf_counter() - decode_start) * 1000.0

        if message is None:
            self._decode_fail_count += 1
            return
        self._decoded_count += 1
        self._latest_camera_info = message
        self._camera_info_version += 1
        logging.debug(
            "[QuestReceiver] camera_info updated: fx=%.1f baseline=%.4fm",
            message.left_fx,
            message.baseline_m,
        )

    def _update_sender_diagnostics(
        self, message: QuestStereoMsg, local_rx_ms: float
    ) -> None:
        """更新发送端链路诊断统计。"""
        sender_mono_ms = message.sender_mono_ms
        sender_frame_id = message.frame_id
        alpha = 0.15
        previous_frame_id = self._last_sender_frame_id
        previous_mono_ms = self._last_sender_mono_ms

        if sender_frame_id is not None:
            if (
                previous_frame_id is not None
                and sender_frame_id > previous_frame_id + 1
            ):
                self._sender_gap_count += sender_frame_id - previous_frame_id - 1
            self._last_sender_frame_id = sender_frame_id

        if sender_mono_ms is not None:
            sender_delta_raw_ms = local_rx_ms - sender_mono_ms

            self._sender_delta_raw_ema_ms = (
                sender_delta_raw_ms
                if self._sender_delta_raw_ema_ms <= 0.0
                else self._sender_delta_raw_ema_ms * (1.0 - alpha)
                + sender_delta_raw_ms * alpha
            )

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

            # 估计发送端帧率。
            if (
                previous_frame_id is not None
                and sender_frame_id is not None
                and previous_mono_ms is not None
            ):
                frame_delta = sender_frame_id - previous_frame_id
                mono_delta_ms = sender_mono_ms - previous_mono_ms
                if frame_delta > 0 and mono_delta_ms > 1e-6:
                    sender_fps_inst = frame_delta * 1000.0 / mono_delta_ms
                    self._sender_fps_ema = (
                        sender_fps_inst
                        if self._sender_fps_ema <= 0.0
                        else self._sender_fps_ema * (1.0 - alpha)
                        + sender_fps_inst * alpha
                    )

            self._last_sender_mono_ms = sender_mono_ms

    def __enter__(self) -> "QuestReceiver":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()


if __name__ == "__main__":
    """
    OpenCV 可视化示例：
    1) 启动 Quest 多 Topic 接收器。
    2) 循环轮询双目图像和相机信息。
    3) 显示左右拼接图，并打印统计与标定信息。
    4) 按 q 或 ESC 退出。
    """

    LISTEN_PORT = 5557
    STATS_INTERVAL = 60
    WINDOW_NAME = "Quest Stereo (Left | Right)"

    receiver = QuestReceiver(listen_port=LISTEN_PORT, timeout_ms=50)

    frame_count = 0
    start_time = time.perf_counter()
    last_stats_time = start_time
    last_stats_frame_count = 0
    camera_info_printed = False
    last_warn_time = start_time
    placeholder = np.zeros((240, 640, 3), dtype=np.uint8)
    cv2.putText(
        placeholder,
        "Waiting for Quest frames...",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    try:
        receiver.start()
        print("窗口已打开，按 q 或 ESC 退出。")
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(WINDOW_NAME, placeholder)
        cv2.waitKey(1)

        while True:
            # 先处理窗口事件，避免没数据时假死。
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

            stereo = receiver.get_stereo_frames()

            # 没收到数据时，每 3 秒打印一次诊断，并保持占位图刷新。
            if stereo is None or stereo.left is None or stereo.right is None:
                now = time.perf_counter()
                if now - last_warn_time >= 3.0:
                    stats = receiver.get_stats()
                    print(
                        f"[QuestReceiver] 等待数据中... "
                        f"Received={stats['received']}, Decoded={stats['decoded']}, "
                        f"DecodeFailed={stats['decode_failed']}. "
                        f"请确认 Unity 端已运行且 Sender 的 topic=quest_stereo 生效。"
                    )
                    last_warn_time = now
                cv2.imshow(WINDOW_NAME, placeholder)
                continue

            left = stereo.left
            right = stereo.right

            # 首次收到 camera_info 时打印标定信息。
            if not camera_info_printed:
                info = receiver.get_camera_info()
                if info is not None:
                    calib = receiver.get_calibration()
                    print(
                        f"[QuestReceiver] CameraInfo: "
                        f"fx={info.left_fx:.1f} fy={info.left_fy:.1f} "
                        f"cx={info.left_cx:.1f} cy={info.left_cy:.1f} "
                        f"baseline={info.baseline_m:.6f}m "
                        f"sensor={info.sensor_width}x{info.sensor_height} "
                        f"current={info.current_width}x{info.current_height}"
                    )
                    if calib is not None:
                        print(
                            f"[QuestReceiver] Calibration: "
                            f"calib={calib.calib_width}x{calib.calib_height} "
                            f"baseline={calib.baseline_m:.6f}m"
                        )
                    camera_info_printed = True

            if left.shape[0] != right.shape[0]:
                target_height = min(left.shape[0], right.shape[0])
                left = cv2.resize(
                    left,
                    (int(left.shape[1] * target_height / left.shape[0]), target_height),
                    interpolation=cv2.INTER_LINEAR,
                )
                right = cv2.resize(
                    right,
                    (int(right.shape[1] * target_height / right.shape[0]), target_height),
                    interpolation=cv2.INTER_LINEAR,
                )

            stereo_view = np.hstack((left, right))
            cv2.imshow(WINDOW_NAME, stereo_view)

            frame_count += 1
            if frame_count % STATS_INTERVAL == 0:
                now = time.perf_counter()
                interval_elapsed = max(now - last_stats_time, 1e-6)
                interval_fps = (frame_count - last_stats_frame_count) / interval_elapsed

                stats = receiver.get_stats()
                print(
                    f"[QuestReceiver] "
                    f"Frames={frame_count}, FPS={interval_fps:.1f}, "
                    f"Received={stats['received']}, Decoded={stats['decoded']}, "
                    f"DecodeFailed={stats['decode_failed']}, "
                    f"AvgDecode={float(stats['avg_decode_ms']):.2f}ms"
                )

                last_stats_time = now
                last_stats_frame_count = frame_count
    finally:
        receiver.stop()
        cv2.destroyAllWindows()
