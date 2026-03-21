from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray


"""
业务解码层：把 list[bytes] payload 还原为业务对象。

通信分层约定：
- 本文件只关心协议解析，不关心 socket 收发。
- socket 收发由 zmq_utils.communicate.receiver.PayloadReceiver 负责。
"""


class PayloadDecoder(ABC):
    """通用 payload 解码器接口。"""

    @abstractmethod
    def decode(self, parts: list[bytes]) -> object | None:
        """将 bytes[] 解析为目标对象；失败返回 None。"""
        pass


class StereoJpegDecoder(PayloadDecoder):
    """解析 [left_jpg, right_jpg] 或 [packed_stereo_jpg] 为左右 BGR 图像。"""

    def decode(
        self, parts: list[bytes]
    ) -> tuple[NDArray[np.uint8], NDArray[np.uint8]] | None:
        if len(parts) == 1:
            packed = cv2.imdecode(np.frombuffer(parts[0], np.uint8), cv2.IMREAD_COLOR)
            if packed is None:
                return None

            height, width = packed.shape[:2]
            if width < 2:
                return None

            mid = width // 2
            left = packed[:, :mid]
            right = packed[:, mid:]
            return cast(NDArray[np.uint8], left), cast(NDArray[np.uint8], right)

        if len(parts) != 2:
            return None

        left = cv2.imdecode(np.frombuffer(parts[0], np.uint8), cv2.IMREAD_COLOR)
        right = cv2.imdecode(np.frombuffer(parts[1], np.uint8), cv2.IMREAD_COLOR)

        if left is None or right is None:
            return None

        return cast(NDArray[np.uint8], left), cast(NDArray[np.uint8], right)


class RGBDDecoder(PayloadDecoder):
    """解析 [color_jpg, depth_png] 为 (color, depth)。"""

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
    """解析 [phase_byte, color_jpg, pose_json]。"""

    def decode(
        self, parts: list[bytes]
    ) -> tuple[int, NDArray[np.uint8], NDArray[np.float64] | None] | None:
        """解码 tracking payload。

        返回：
        - (phase, color, pose_matrix)
        - 其中 pose_matrix 可为 None（无位姿）。
        """
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
    """解析第一帧为 UTF-8 文本。"""

    def decode(self, parts: list[bytes]) -> str | None:
        if len(parts) == 0:
            return None
        try:
            return parts[0].decode("utf-8")
        except UnicodeDecodeError:
            return None
