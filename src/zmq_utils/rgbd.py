"""
ZMQ 工具包 - RGBD 图像传输

提供 RGBD (彩色+深度) 图像发送/接收功能。
"""

from __future__ import annotations

import time
from typing import cast

import cv2
import numpy as np
import zmq
from numpy.typing import NDArray

from .base import PubNode, PullNode, PushNode, SubNode
from .timing import LatencyStats


class RGBDSender(PushNode):
    """RGBD 图像发送节点（同时发送 color + depth）"""

    def send_rgbd(
        self,
        color: NDArray[np.uint8] | None,
        depth: NDArray[np.uint16] | None,
        quality: int = 80,
    ) -> bool:
        """发送 RGBD 图像对

        Args:
            color: BGR 彩色图像
            depth: 深度图像（uint16，单位通常是 mm）
            quality: JPEG 压缩质量
        """
        if color is None or depth is None:
            return False
        # 编码 color 为 JPEG
        success, color_buf = cv2.imencode(
            ".jpg", color, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        )
        if not success:
            return False
        # depth 使用 PNG 无损压缩（保留精度）
        success, depth_buf = cv2.imencode(".png", depth)
        if not success:
            return False
        return self.send_multipart([color_buf.tobytes(), depth_buf.tobytes()])


class RGBDReceiver(PullNode):
    """RGBD 图像接收节点（带本地 FPS 统计）"""

    def __init__(
        self,
        endpoint: str,
        hwm: int = 1,
        bind: bool = True,
        stats_window: int = 100,
    ) -> None:
        super().__init__(endpoint, hwm, bind)
        self.interval_stats = LatencyStats(window_size=stats_window)
        self._last_recv_time: float = 0.0
        self._frame_count: int = 0
        self._start_time: float = 0.0

    def recv_rgbd(
        self, timeout_ms: int = 0
    ) -> tuple[NDArray[np.uint8], NDArray[np.uint16]] | None:
        """接收 RGBD 图像对

        Returns:
            (color, depth) 或 None
        """
        parts = self.recv_multipart_latest(timeout_ms)
        if not parts or len(parts) != 2:
            return None

        # 更新统计
        self._update_stats()

        color_raw = cv2.imdecode(np.frombuffer(parts[0], np.uint8), cv2.IMREAD_COLOR)
        depth_raw = cv2.imdecode(
            np.frombuffer(parts[1], np.uint8), cv2.IMREAD_UNCHANGED
        )
        if color_raw is None or depth_raw is None:
            return None
        return cast(NDArray[np.uint8], color_raw), depth_raw.astype(np.uint16)

    def _update_stats(self) -> None:
        """更新内部统计"""
        now = time.perf_counter()
        if self._start_time == 0.0:
            self._start_time = now
        if self._last_recv_time > 0:
            interval_ms = (now - self._last_recv_time) * 1000
            self.interval_stats.record(interval_ms)
        self._last_recv_time = now
        self._frame_count += 1

    def get_fps(self) -> float:
        """获取实时 FPS（基于滑动窗口平均帧间隔）"""
        avg_interval = self.interval_stats.get_stats()["avg"]
        if avg_interval <= 0:
            return 0.0
        return 1000.0 / avg_interval  # interval 是 ms，转为 FPS

    def get_stats(self) -> dict[str, float]:
        """获取统计信息"""
        interval = self.interval_stats.get_stats()
        return {
            "fps": self.get_fps(),
            "frame_count": self._frame_count,
            "interval_avg_ms": interval["avg"],
            "interval_std_ms": interval["std"],
        }


class RGBDPublisher(PubNode):
    """RGBD 图像发布节点（服务器向客户端广播 RGBD）"""

    def publish_rgbd(
        self,
        topic: str,
        color: NDArray[np.uint8] | None,
        depth: NDArray[np.uint16] | None,
        quality: int = 80,
    ) -> bool:
        """发布 RGBD 图像对

        Args:
            topic: 主题名称
            color: BGR 彩色图像
            depth: 深度图像（uint16，单位通常是 mm）
            quality: JPEG 压缩质量
        """
        if color is None or depth is None or self.socket is None:
            return False
        # 编码 color 为 JPEG
        success, color_buf = cv2.imencode(
            ".jpg", color, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        )
        if not success:
            return False
        # depth 使用 PNG 无损压缩
        success, depth_buf = cv2.imencode(".png", depth)
        if not success:
            return False
        self.socket.send_multipart(
            [topic.encode("utf-8"), color_buf.tobytes(), depth_buf.tobytes()]
        )
        return True


class RGBDSubscriber(SubNode):
    """RGBD 图像订阅节点（客户端接收 RGBD）"""

    def _create_socket(self) -> zmq.Socket[bytes]:
        """重写以禁用 CONFLATE（因为 RGBD 是多帧消息）"""
        sock = self.ctx.socket(zmq.SUB)
        # 注意：不设置 CONFLATE，因为多帧消息不能使用 CONFLATE
        sock.setsockopt_string(zmq.SUBSCRIBE, self.topic)
        return sock

    def recv_rgbd(
        self, timeout_ms: int = 10
    ) -> tuple[NDArray[np.uint8], NDArray[np.uint16]] | None:
        """接收 RGBD 图像对

        Returns:
            (color, depth) 或 None
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

            if len(parts) != 3:  # topic, color, depth
                return None
            color_raw = cv2.imdecode(
                np.frombuffer(parts[1], np.uint8), cv2.IMREAD_COLOR
            )
            depth_raw = cv2.imdecode(
                np.frombuffer(parts[2], np.uint8), cv2.IMREAD_UNCHANGED
            )
            if color_raw is None or depth_raw is None:
                return None
            return cast(NDArray[np.uint8], color_raw), depth_raw.astype(np.uint16)
        except zmq.ZMQError:
            return None
