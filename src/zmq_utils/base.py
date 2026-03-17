"""
ZMQ 工具包 - 基础节点类

提供 ZMQ 通信的抽象基类和基础传输模式。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import zmq


class BaseNode(ABC):
    """ZMQ 节点基类，处理 socket 生命周期管理"""

    ctx: zmq.Context[zmq.Socket[bytes]]
    socket: zmq.Socket[bytes] | None
    endpoint: str
    hwm: int
    is_bind: bool

    def __init__(self, endpoint: str, hwm: int = 1, bind: bool = False) -> None:
        """
        Args:
            endpoint: ZMQ 地址，如 "tcp://127.0.0.1:5555"
            hwm: High Water Mark，缓冲区消息上限
            bind: True=作为服务端 bind，False=作为客户端 connect
        """
        self.ctx = zmq.Context.instance()
        self.socket = None
        self.endpoint = endpoint
        self.hwm = hwm
        self.is_bind = bind
        self._setup_socket()

    @abstractmethod
    def _create_socket(self) -> zmq.Socket[bytes]:
        """创建并配置 socket，子类实现"""
        pass

    def _setup_socket(self) -> None:
        """统一处理 bind/connect 逻辑"""
        self.socket = self._create_socket()
        self.socket.set_hwm(self.hwm)
        if self.is_bind:
            self.socket.bind(self.endpoint)
            print(f"[{self.__class__.__name__}] Bound to {self.endpoint}")
        else:
            self.socket.connect(self.endpoint)
            print(f"[{self.__class__.__name__}] Connected to {self.endpoint}")

    def close(self) -> None:
        if self.socket:
            self.socket.close()
            self.socket = None
            print(f"[ZMQ] Node closed: {self.endpoint}")

    def __enter__(self) -> "BaseNode":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


class PushNode(BaseNode):
    """单向发送节点，用于高吞吐数据流（如视频流）"""

    def __init__(self, endpoint: str, hwm: int = 1, bind: bool = False) -> None:
        super().__init__(endpoint, hwm, bind)

    def _create_socket(self) -> zmq.Socket[bytes]:
        return self.ctx.socket(zmq.PUSH)

    def send_raw(self, data: bytes) -> bool:
        """发送原始字节数据，网络拥塞时丢弃

        Returns:
            True 发送成功，False 被丢弃（网络拥塞）
        """
        if not self.socket:
            return False
        try:
            self.socket.send(data, flags=zmq.NOBLOCK)
            return True
        except zmq.Again:
            return False  # 网络拥塞时丢弃，保证实时性

    def send_multipart(self, parts: list[bytes]) -> bool:
        """发送多帧消息（用于 RGBD 等多数据组合）"""
        if not self.socket:
            return False
        try:
            self.socket.send_multipart(parts, flags=zmq.NOBLOCK)
            return True
        except zmq.Again:
            return False


class PullNode(BaseNode):
    """单向接收节点，支持只取最新帧 (Conflate)"""

    def __init__(self, endpoint: str, hwm: int = 1, bind: bool = True) -> None:
        super().__init__(endpoint, hwm, bind)

    def _create_socket(self) -> zmq.Socket[bytes]:
        sock = self.ctx.socket(zmq.PULL)
        return sock

    def recv_raw_latest(self, timeout_ms: int = 0) -> bytes | None:
        """读取缓冲区直到最后一帧 (Conflate 逻辑)

        Args:
            timeout_ms: 等待超时，0=非阻塞，-1=永久阻塞
        """
        if not self.socket:
            return None
        try:
            if timeout_ms != -1 and not self.socket.poll(timeout=timeout_ms):
                return None
            msg = self.socket.recv()
            # 持续读取直到没有更多消息，只保留最后一个
            while int(self.socket.get(zmq.EVENTS)) & zmq.POLLIN:
                msg = self.socket.recv()
            return msg
        except zmq.ZMQError:
            return None

    def recv_multipart_latest(self, timeout_ms: int = 0) -> list[bytes] | None:
        """读取多帧消息的最新一组"""
        if not self.socket:
            return None
        try:
            if timeout_ms != -1 and not self.socket.poll(timeout=timeout_ms):
                return None
            msg = self.socket.recv_multipart()
            while int(self.socket.get(zmq.EVENTS)) & zmq.POLLIN:
                msg = self.socket.recv_multipart()
            return msg
        except zmq.ZMQError:
            return None


class PubNode(BaseNode):
    """发布节点，用于广播状态信息"""

    def __init__(self, endpoint: str, hwm: int = 1, bind: bool = True) -> None:
        super().__init__(endpoint, hwm, bind)

    def _create_socket(self) -> zmq.Socket[bytes]:
        return self.ctx.socket(zmq.PUB)

    def publish_raw(self, topic: str, data: bytes) -> None:
        """发布带 topic 的消息"""
        if self.socket:
            self.socket.send_multipart([topic.encode("utf-8"), data])


class SubNode(BaseNode):
    """订阅节点，用于订阅状态信息"""

    topic: str

    def __init__(
        self, endpoint: str, topic: str = "", hwm: int = 1, bind: bool = False
    ) -> None:
        self.topic = topic
        super().__init__(endpoint, hwm, bind)

    def _create_socket(self) -> zmq.Socket[bytes]:
        sock = self.ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.CONFLATE, 1)  # 只保留最新消息
        sock.setsockopt_string(zmq.SUBSCRIBE, self.topic)
        return sock

    def recv_raw_latest(self, timeout_ms: int = 10) -> bytes | None:
        """接收最新消息内容（自动丢弃旧消息）"""
        if not self.socket:
            return None
        try:
            if not self.socket.poll(timeout=timeout_ms):
                return None
            _topic = self.socket.recv()  # 读 topic（丢弃）
            return self.socket.recv()  # 读 content
        except zmq.ZMQError:
            return None
