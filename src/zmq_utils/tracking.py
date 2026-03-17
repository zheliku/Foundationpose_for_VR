"""
ZMQ 工具包 - 追踪数据传输

提供追踪结果（位姿 + 带标记图像）的发布/订阅功能。

数据协议:
    - Frame 1: topic (UTF-8 string)
    - Frame 2: phase (1 byte, 0=detecting, 1=tracking)
    - Frame 3: color (JPEG bytes)
    - Frame 4: pose (UTF-8 JSON string, 空字符串表示无位姿)
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import zmq
from numpy.typing import NDArray

from .base import PubNode, SubNode


class TrackingPublisher(PubNode):
    """追踪结果发布节点"""

    def publish_tracking(
        self,
        topic: str,
        phase: int,
        color: NDArray[np.uint8],
        pose_matrix: NDArray[np.float64] | None = None,
        quality: int = 80,
    ) -> bool:
        """
        发布追踪数据

        参数
        ----
        topic : str
            主题名称
        phase : int
            追踪阶段 (0=detecting, 1=tracking)
        color : NDArray[np.uint8]
            BGR 图像
        pose_matrix : NDArray[np.float64] | None
            4x4 位姿矩阵，仅追踪阶段有效
        quality : int
            JPEG 压缩质量

        返回
        ----
        bool
            是否发送成功
        """
        if self.socket is None:
            return False

        # 编码 color 为 JPEG
        success, color_buf = cv2.imencode(
            ".jpg", color, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        )
        if not success:
            return False

        # 编码 pose 为 JSON
        if pose_matrix is not None:
            pose_json = json.dumps({"matrix": pose_matrix.tolist()})
        else:
            pose_json = ""

        # 发送多帧消息（使用 NOBLOCK 避免积压）
        try:
            self.socket.send_multipart(
                [
                    topic.encode("utf-8"),
                    bytes([phase]),
                    color_buf.tobytes(),
                    pose_json.encode("utf-8"),
                ],
                flags=zmq.NOBLOCK,
            )
            return True
        except zmq.Again:
            # 网络拥塞时丢弃帧，保证实时性
            return False


class TrackingSubscriber(SubNode):
    """追踪结果订阅节点"""

    def _create_socket(self) -> zmq.Socket[bytes]:
        """重写以禁用 CONFLATE（因为是多帧消息）"""
        sock = self.ctx.socket(zmq.SUB)
        sock.setsockopt_string(zmq.SUBSCRIBE, self.topic)
        return sock

    def recv_tracking(
        self, timeout_ms: int = 10
    ) -> tuple[int, NDArray[np.uint8], NDArray[np.float64] | None] | None:
        """
        接收追踪数据

        返回
        ----
        tuple[int, NDArray, NDArray | None] | None
            (phase, color, pose_matrix) 或 None
        """
        if not self.socket:
            return None

        try:
            if not self.socket.poll(timeout=timeout_ms):
                return None

            parts = self.socket.recv_multipart()

            # 清空积压
            while int(self.socket.get(zmq.EVENTS)) & zmq.POLLIN:
                parts = self.socket.recv_multipart()

            if len(parts) != 4:  # topic, phase, color, pose
                return None

            phase = parts[1][0]
            color = cv2.imdecode(np.frombuffer(parts[2], np.uint8), cv2.IMREAD_COLOR)

            pose_json = parts[3].decode("utf-8")
            if pose_json:
                pose_data = json.loads(pose_json)
                pose_matrix = np.array(pose_data["matrix"], dtype=np.float64)
            else:
                pose_matrix = None

            if color is None:
                return None

            return phase, color, pose_matrix

        except zmq.ZMQError:
            return None
