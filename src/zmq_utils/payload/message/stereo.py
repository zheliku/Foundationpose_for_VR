from __future__ import annotations

from dataclasses import dataclass

import msgpack
import numpy as np
from numpy.typing import NDArray


@dataclass
class QuestStereoMsg:
    """Quest 双目单帧消息：包含图像字节、元数据与解码后的图像。"""

    packed_image: bytes | None = None  # 左右拼接后的单张编码图像字节。

    frame_id: int | None = None  # 发送端帧号。
    sender_mono_ms: float | None = None  # 发送端单调时钟（毫秒）。
    unity_frame: int | None = None  # Unity 发送时的 frameCount。

    left: NDArray[np.uint8] | None = None  # 解码后的左目 BGR 图像。
    right: NDArray[np.uint8] | None = None  # 解码后的右目 BGR 图像。
    timestamp_ms: float | None = None  # 接收端本地时间戳（毫秒）。
    sender_delta_raw_ms: float | None = None  # 原始跨端时钟差（本地 - 发送端）。
    sender_delay_est_ms: float | None = None  # 基于最小基线估算的额外排队延迟。

    def serialize(self) -> bytes | None:
        """将消息序列化为 MessagePack 负载。"""
        if self.packed_image is None:
            return None

        payload = {
            "image_jpeg": bytes(self.packed_image),
            "frame_id": int(self.frame_id or 0),
            "sender_mono_ms": float(self.sender_mono_ms or 0.0),
            "unity_frame": int(self.unity_frame or 0),
        }
        return msgpack.packb(payload, use_bin_type=True)

    @classmethod
    def deserialize(cls, payload: bytes) -> QuestStereoMsg | None:
        """从 MessagePack 负载反序列化消息对象。"""
        try:
            data = msgpack.unpackb(payload, raw=False, strict_map_key=False)
            if not isinstance(data, dict):
                return None

            image_raw = data.get("image_jpeg")
            if not isinstance(image_raw, (bytes, bytearray)):
                return None

            return cls(
                packed_image=bytes(image_raw),
                frame_id=int(data.get("frame_id", 0)),
                sender_mono_ms=float(data.get("sender_mono_ms", 0.0)),
                unity_frame=int(data.get("unity_frame", 0)),
            )
        except (TypeError, ValueError):
            return None
