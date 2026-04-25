from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import msgpack


@dataclass(frozen=True)
class RGBDMsg:
    """RGBD 传输消息：单条消息同时包含彩色图与深度图字节。

    注意：这里保存的是“压缩后的图像字节”，不是 numpy 原始数组。
    这样可以显著减少 MessagePack 负载体积，并让接收端按需用 OpenCV 解码。
    """

    color_image: bytes  # 彩色图编码字节（通常为 JPG，有损压缩）。
    depth_image: bytes  # 深度图编码字节（通常为 PNG，无损压缩）。
    timestamp_ms: float = 0.0  # 可选时间戳（毫秒），用于调试延迟或回放对齐。

    def serialize(self) -> bytes:
        """序列化为 MessagePack 字节。"""
        # 字段名需与 Unity 侧 RGBDMsg.cs 保持一致。
        payload = {
            "color_image": bytes(self.color_image),
            "depth_image": bytes(self.depth_image),
            "timestamp_ms": float(self.timestamp_ms),
        }
        return cast(bytes, msgpack.packb(payload, use_bin_type=True))

    @classmethod
    def deserialize(cls, payload: bytes) -> RGBDMsg | None:
        """从 MessagePack 字节反序列化。"""
        try:
            data = msgpack.unpackb(payload, raw=False, strict_map_key=False)
            if not isinstance(data, dict):
                return None

            # MessagePack 的 bin 字段在 Python 中应还原为 bytes/bytearray。
            color_raw = data.get("color_image")
            depth_raw = data.get("depth_image")
            if not isinstance(color_raw, (bytes, bytearray)):
                return None
            if not isinstance(depth_raw, (bytes, bytearray)):
                return None

            return cls(
                color_image=bytes(color_raw),
                depth_image=bytes(depth_raw),
                timestamp_ms=float(data.get("timestamp_ms", 0.0)),
            )
        except (TypeError, ValueError):
            return None
