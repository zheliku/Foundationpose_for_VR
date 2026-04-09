from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class QuestStereoMsg:
    """Quest 双目完整消息：包含原始发送字节、元数据与解码后的图像。"""

    left_image: bytes | None = None  # 左目编码图像字节（Dual 模式）。
    right_image: bytes | None = None  # 右目编码图像字节（Dual 模式）。
    packed_image: bytes | None = None  # 拼接编码图像字节（Packed 模式）。

    frame_id: int | None = None  # 发送端帧号。
    sender_mono_ms: float | None = None  # 发送端单调时钟（毫秒）。
    unity_frame: int | None = None  # Unity 发送时的 frameCount。

    left: NDArray[np.uint8] | None = None  # 解码后的左目 BGR 图像。
    right: NDArray[np.uint8] | None = None  # 解码后的右目 BGR 图像。
    timestamp_ms: float | None = None  # 接收端本地时间戳（毫秒）。
    sender_delta_raw_ms: float | None = None  # 原始跨端时钟差（本地 - 发送端）。
    sender_delay_est_ms: float | None = None  # 基于最小基线估算的额外排队延迟。

    @property
    def is_packed(self) -> bool:
        """是否为 Packed 模式消息。"""
        return self.packed_image is not None

    @property
    def has_metadata(self) -> bool:
        """是否包含可用发送端元数据。"""
        return self.frame_id is not None and self.sender_mono_ms is not None

    def to_parts(self, include_metadata: bool = True) -> list[bytes] | None:
        """将消息转换为发送用 multipart 数据。"""
        if self.packed_image is not None:
            parts = [self.packed_image]
        elif self.left_image is not None and self.right_image is not None:
            parts = [self.left_image, self.right_image]
        else:
            return None

        if include_metadata and self.has_metadata:
            parts.append(self.to_metadata_json_bytes())

        return parts

    def to_metadata_json_bytes(self) -> bytes:
        """将元数据编码为 UTF-8 JSON 字节。"""
        payload = {
            "frame_id": int(self.frame_id or 0),
            "sender_mono_ms": float(self.sender_mono_ms or 0.0),
            "unity_frame": int(self.unity_frame or 0),
        }
        return json.dumps(payload, ensure_ascii=False).encode("utf-8")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> QuestStereoMsg | None:
        """从元数据字典构建消息对象（不含图像字节）。"""
        if "frame_id" not in value or "sender_mono_ms" not in value:
            return None

        try:
            unity_frame_raw = value.get("unity_frame")
            unity_frame = int(unity_frame_raw) if unity_frame_raw is not None else None
            return cls(
                frame_id=int(value["frame_id"]),
                sender_mono_ms=float(value["sender_mono_ms"]),
                unity_frame=unity_frame,
            )
        except (TypeError, ValueError):
            return None

    @classmethod
    def from_json_bytes(cls, payload: bytes) -> QuestStereoMsg | None:
        """从 UTF-8 JSON 元数据字节构建消息对象。"""
        try:
            text = payload.decode("utf-8")
            data = json.loads(text)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None

        if not isinstance(data, dict):
            return None

        return cls.from_dict(data)

    @classmethod
    def from_parts(cls, parts: list[bytes]) -> QuestStereoMsg | None:
        """从完整 multipart 数据构建消息对象。"""
        if not parts:
            return None

        metadata = cls.from_json_bytes(parts[-1])
        image_parts = parts[:-1] if metadata is not None else parts

        if len(image_parts) == 1:
            message = cls(packed_image=bytes(image_parts[0]))
        elif len(image_parts) == 2:
            message = cls(
                left_image=bytes(image_parts[0]),
                right_image=bytes(image_parts[1]),
            )
        else:
            return None

        if metadata is not None:
            message.frame_id = metadata.frame_id
            message.sender_mono_ms = metadata.sender_mono_ms
            message.unity_frame = metadata.unity_frame

        return message
