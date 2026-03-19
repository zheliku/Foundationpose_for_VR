"""
ZMQ 工具包 - 通用 Payload 接收与解析

目标：
- 提供可复用的通用 multipart 接收器
- 提供可复用的数据解析器（双目 JPEG、文本、整数）
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import cast

import cv2
import numpy as np
import zmq
from numpy.typing import NDArray

from .base import PubNode, PullNode, PushNode


class MultipartSender(PushNode):
    """通用 multipart 发送器（PUSH）"""

    def send_payload(self, parts: list[bytes]) -> bool:
        return self.send_multipart(parts)


class TopicMultipartPublisher(PubNode):
    """通用 topic + multipart 发布器（PUB）"""

    def publish_payload(self, topic: str, parts: list[bytes]) -> bool:
        if self.socket is None:
            return False
        try:
            self.socket.send_multipart(
                [topic.encode("utf-8"), *parts], flags=zmq.NOBLOCK
            )
            return True
        except zmq.Again:
            return False


class MultipartReceiver(PullNode):
    """通用 multipart 接收器（PULL）"""

    def recv_payload(self, timeout_ms: int = 0) -> list[bytes] | None:
        return self.recv_multipart_latest(timeout_ms)


class PayloadParser(ABC):
    """通用 payload 解析器接口"""

    @abstractmethod
    def parse(self, parts: list[bytes]) -> object | None:
        pass


class PayloadEncoder(ABC):
    """通用 payload 编码器接口"""

    @abstractmethod
    def encode(self, *args: object, **kwargs: object) -> list[bytes] | None:
        pass


class StereoJpegParser(PayloadParser):
    """解析 [left_jpg, right_jpg] 为左右 BGR 图像"""

    def parse(
        self, parts: list[bytes]
    ) -> tuple[NDArray[np.uint8], NDArray[np.uint8]] | None:
        if len(parts) != 2:
            return None

        left = cv2.imdecode(np.frombuffer(parts[0], np.uint8), cv2.IMREAD_COLOR)
        right = cv2.imdecode(np.frombuffer(parts[1], np.uint8), cv2.IMREAD_COLOR)

        if left is None or right is None:
            return None

        return cast(NDArray[np.uint8], left), cast(NDArray[np.uint8], right)


class RGBDPayloadEncoder:
    """编码 (color, depth) 为 [color_jpg, depth_png]"""

    def encode(
        self,
        color: NDArray[np.uint8],
        depth: NDArray[np.uint16],
        quality: int = 80,
    ) -> list[bytes] | None:
        success, color_buf = cv2.imencode(
            ".jpg", color, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        )
        if not success:
            return None

        success, depth_buf = cv2.imencode(".png", depth)
        if not success:
            return None

        return [color_buf.tobytes(), depth_buf.tobytes()]


class RGBDPayloadParser(PayloadParser):
    """解析 [color_jpg, depth_png] 为 (color, depth)"""

    def parse(
        self, parts: list[bytes]
    ) -> tuple[NDArray[np.uint8], NDArray[np.uint16]] | None:
        if len(parts) != 2:
            return None

        color = cv2.imdecode(np.frombuffer(parts[0], np.uint8), cv2.IMREAD_COLOR)
        depth = cv2.imdecode(np.frombuffer(parts[1], np.uint8), cv2.IMREAD_UNCHANGED)

        if color is None or depth is None:
            return None

        return cast(NDArray[np.uint8], color), depth.astype(np.uint16)


class TrackingPayloadEncoder:
    """编码追踪结果为 [phase_byte, color_jpg, pose_json]"""

    def encode(
        self,
        phase: int,
        color: NDArray[np.uint8],
        pose_matrix: NDArray[np.float64] | None = None,
        quality: int = 80,
    ) -> list[bytes] | None:
        success, color_buf = cv2.imencode(
            ".jpg", color, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        )
        if not success:
            return None

        pose_json = (
            json.dumps({"matrix": pose_matrix.tolist()})
            if pose_matrix is not None
            else ""
        )

        return [bytes([phase]), color_buf.tobytes(), pose_json.encode("utf-8")]


class TrackingPayloadParser(PayloadParser):
    """解析 [phase_byte, color_jpg, pose_json]"""

    def parse(
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


class Utf8TextParser(PayloadParser):
    """解析第一帧为 UTF-8 文本"""

    def parse(self, parts: list[bytes]) -> str | None:
        if len(parts) == 0:
            return None
        try:
            return parts[0].decode("utf-8")
        except UnicodeDecodeError:
            return None


class IntParser(PayloadParser):
    """解析第一帧为 int"""

    def parse(self, parts: list[bytes]) -> int | None:
        if len(parts) == 0:
            return None
        try:
            return int(parts[0].decode("utf-8").strip())
        except (UnicodeDecodeError, ValueError):
            return None
