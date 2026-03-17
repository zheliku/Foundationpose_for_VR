"""
ZMQ 工具包 - 图像传输

提供图像发送/接收功能，支持 Push/Pull 和 Pub/Sub 模式。
"""

from __future__ import annotations

import time
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from .base import PubNode, PullNode, PushNode, SubNode
from .timing import LatencyStats


class ImageSender(PushNode):
    """图像发送节点（JPEG 压缩）"""

    def send_image(self, frame: NDArray[np.uint8] | None, quality: int = 80) -> bool:
        """发送单张图像"""
        if frame is None:
            return False
        success, buffer = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        )
        if success:
            return self.send_raw(buffer.tobytes())
        return False


class ImageReceiver(PullNode):
    """图像接收节点（带本地 FPS 统计）"""

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

    def recv_image(self, timeout_ms: int = 0) -> NDArray[np.uint8] | None:
        """接收单张图像"""
        data = self.recv_raw_latest(timeout_ms)
        if data:
            # 更新统计
            self._update_stats()

            nparr = np.frombuffer(data, np.uint8)
            result = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return cast(NDArray[np.uint8], result) if result is not None else None
        return None

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
        """获取当前 FPS"""
        if self._start_time == 0.0:
            return 0.0
        elapsed = time.perf_counter() - self._start_time
        return self._frame_count / elapsed if elapsed > 0 else 0.0

    def get_stats(self) -> dict[str, float]:
        """获取统计信息"""
        interval = self.interval_stats.get_stats()
        return {
            "fps": self.get_fps(),
            "frame_count": self._frame_count,
            "interval_avg_ms": interval["avg"],
            "interval_std_ms": interval["std"],
        }


class ImagePublisher(PubNode):
    """图像发布节点（服务器向客户端广播图像）"""

    def publish_image(
        self, topic: str, frame: NDArray[np.uint8] | None, quality: int = 80
    ) -> bool:
        """发布图像

        Args:
            topic: 主题名称
            frame: BGR 彩色图像
            quality: JPEG 压缩质量
        """
        if frame is None or self.socket is None:
            return False
        success, buffer = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        )
        if success:
            self.socket.send_multipart([topic.encode("utf-8"), buffer.tobytes()])
            return True
        return False


class ImageSubscriber(SubNode):
    """图像订阅节点（客户端接收图像）"""

    def recv_image(self, timeout_ms: int = 10) -> NDArray[np.uint8] | None:
        """接收图像"""
        data = self.recv_raw_latest(timeout_ms)
        if data:
            nparr = np.frombuffer(data, np.uint8)
            result = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return cast(NDArray[np.uint8], result) if result is not None else None
        return None
