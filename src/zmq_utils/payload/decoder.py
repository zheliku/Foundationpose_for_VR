from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray


class PayloadDecoder(ABC):
    """通用 payload 解码器接口"""

    @abstractmethod
    def decode(self, parts: list[bytes]) -> object | None:
        pass


class StereoJpegDecoder(PayloadDecoder):
    """解析 [left_jpg, right_jpg] 为左右 BGR 图像"""

    def decode(
        self, parts: list[bytes]
    ) -> tuple[NDArray[np.uint8], NDArray[np.uint8]] | None:
        if len(parts) != 2:
            return None

        left = cv2.imdecode(np.frombuffer(parts[0], np.uint8), cv2.IMREAD_COLOR)
        right = cv2.imdecode(np.frombuffer(parts[1], np.uint8), cv2.IMREAD_COLOR)

        if left is None or right is None:
            return None

        return cast(NDArray[np.uint8], left), cast(NDArray[np.uint8], right)


class RGBDDecoder(PayloadDecoder):
    """解析 [color_jpg, depth_png] 为 (color, depth)"""

    def decode(
        self, parts: list[bytes]
    ) -> tuple[NDArray[np.uint8], NDArray[np.uint16]] | None:
        if len(parts) != 2:
            return None

        color = cv2.imdecode(np.frombuffer(parts[0], np.uint8), cv2.IMREAD_COLOR)
        depth = cv2.imdecode(np.frombuffer(parts[1], np.uint8), cv2.IMREAD_UNCHANGED)

        if color is None or depth is None:
            return None

        return cast(NDArray[np.uint8], color), depth.astype(np.uint16)


class TrackingDecoder(PayloadDecoder):
    """解析 [phase_byte, color_jpg, pose_json]"""

    def decode(
        self, parts: list[bytes]
    ) -> tuple[int, NDArray[np.uint8], NDArray[np.float64] | None] | None:
        if len(parts) != 3:
            return None

        phase = int(parts[0][0]) if len(parts[0]) > 0 else 0
        color = cv2.imdecode(np.frombuffer(parts[1], np.uint8), cv2.IMREAD_COLOR)
        if color is None:
            return None

        pose_json = parts[2].decode("utf-8") if len(parts[2]) > 0 else ""
        pose_matrix: NDArray[np.float64] | None = None
        if pose_json:
            try:
                pose_matrix = np.array(
                    json.loads(pose_json)["matrix"], dtype=np.float64
                )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                return None

        return phase, cast(NDArray[np.uint8], color), pose_matrix


class Utf8TextDecoder(PayloadDecoder):
    """解析第一帧为 UTF-8 文本"""

    def decode(self, parts: list[bytes]) -> str | None:
        if len(parts) == 0:
            return None
        try:
            return parts[0].decode("utf-8")
        except UnicodeDecodeError:
            return None


class IntDecoder(PayloadDecoder):
    """解析第一帧为 int"""

    def decode(self, parts: list[bytes]) -> int | None:
        if len(parts) == 0:
            return None
        try:
            return int(parts[0].decode("utf-8").strip())
        except (UnicodeDecodeError, ValueError):
            return None
