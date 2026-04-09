from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RGBDMsg:
    """RGBD 传输消息：单条消息同时包含彩色图与深度图字节。"""

    color_image: bytes  # 彩色图编码字节（通常为 JPG）。
    depth_image: bytes  # 深度图编码字节（通常为 PNG）。

    def to_parts(self) -> list[bytes]:
        """转换为 multipart 两段字节数据。"""
        return [self.color_image, self.depth_image]

    @classmethod
    def from_parts(cls, parts: list[bytes]) -> RGBDMsg | None:
        """从 multipart 两段字节数据构建 RGBD 消息。"""
        if len(parts) != 2:
            return None

        color = parts[0]
        depth = parts[1]
        if not isinstance(color, (bytes, bytearray)) or not isinstance(
            depth, (bytes, bytearray)
        ):
            return None

        return cls(bytes(color), bytes(depth))
