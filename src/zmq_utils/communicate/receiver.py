from __future__ import annotations

from typing import Any

import zmq


"""
通用传输接收层（不包含业务语义）。

职责：
1) 管理 ZMQ socket 生命周期（创建、监听、关闭）。
2) 提供统一接收接口（single payload）。
3) 支持两种模式：
   - use_topic=False -> PULL（不含 topic）
   - use_topic=True  -> SUB（首帧是 topic）

说明：
- recv_*_latest 会主动清空积压，仅保留最新消息，适合实时流场景。
"""


class PayloadReceiver:
    """通用 payload 接收器，支持可选 topic 前缀（PULL/SUB）。"""

    # ZMQ 上下文与底层 socket。
    ctx: zmq.Context[zmq.Socket[bytes]]
    socket: zmq.Socket[bytes] | None

    # 连接配置。
    endpoint: str  # 完整连接地址（如 tcp://*:5557）。
    hwm: int  # 高水位，控制接收侧积压上限。
    is_bind: bool  # True=bind，False=connect。

    # 协议与订阅配置。
    use_topic: bool  # 是否启用 topic 前缀模式（SUB）。
    topic: str  # use_topic=True 时订阅的 topic。
    conflate: bool  # 是否启用 CONFLATE（仅保留最新消息）。

    def __init__(
        self,
        endpoint: str,
        hwm: int = 1,
        bind: bool = True,
        use_topic: bool = False,
        topic: str = "",
        conflate: bool = False,
    ) -> None:
        # 仅做必要初始化：保存配置并创建 socket。
        self.ctx = zmq.Context.instance()
        self.socket = None
        self.endpoint = endpoint
        self.hwm = hwm
        self.is_bind = bind
        self.use_topic = use_topic
        self.topic = topic
        self.conflate = bool(conflate)
        self._setup_socket()

    def _setup_socket(self) -> None:
        """创建 socket 并执行 bind/connect。"""
        socket_type = zmq.SUB if self.use_topic else zmq.PULL
        self.socket = self.ctx.socket(socket_type)
        self.socket.set_hwm(self.hwm)

        if self.conflate:
            self.socket.setsockopt(zmq.CONFLATE, 1)

        if self.use_topic:
            self.socket.setsockopt_string(zmq.SUBSCRIBE, self.topic)

        if self.is_bind:
            self.socket.bind(self.endpoint)
            print(f"[{self.__class__.__name__}] Bound to {self.endpoint}")
        else:
            self.socket.connect(self.endpoint)
            print(f"[{self.__class__.__name__}] Connected to {self.endpoint}")

    def _recv_frame_once(self, *, nonblock: bool) -> bytes | None:
        """读取一条单帧 payload。"""
        if self.socket is None:
            return None

        flags = zmq.NOBLOCK if nonblock else 0

        if self.use_topic:
            msg = self.socket.recv_multipart(flags=flags)
            # 单帧 topic 协议固定为 [topic, payload]。
            if len(msg) != 2:
                return None
            return bytes(msg[1])

        return bytes(self.socket.recv(flags=flags))

    def recv_frame_latest(self, timeout_ms: int = 0) -> bytes | None:
        """读取并返回最新一条单帧 payload。

        行为：
        - timeout_ms=-1 时阻塞等待一条消息。
        - 其余场景先 poll，再读取一条并非阻塞 drain 到队尾。
        - 仅返回最后一条消息，旧消息被丢弃。
        """
        if self.socket is None:
            return None

        try:
            if self.conflate:
                if timeout_ms != -1 and not self.socket.poll(
                    timeout=max(int(timeout_ms), 0)
                ):
                    return None

                msg = self._recv_frame_once(nonblock=False)
                return msg

            if timeout_ms == -1:
                msg = self._recv_frame_once(nonblock=False)
            else:
                if not self.socket.poll(timeout=max(int(timeout_ms), 0)):
                    return None
                try:
                    msg = self._recv_frame_once(nonblock=True)
                except zmq.Again:
                    return None

            if msg is None:
                return None

            # 非 conflate 场景下主动 drain 队列，仅保留最新消息。
            while True:
                try:
                    next_msg = self._recv_frame_once(nonblock=True)
                    if next_msg is None:
                        continue
                    msg = next_msg
                except zmq.Again:
                    break
            return msg
        except zmq.ZMQError:
            return None

    def close(self) -> None:
        """关闭 socket。"""
        if self.socket is not None:
            self.socket.close()
            self.socket = None
            print(f"[ZMQ] Node closed: {self.endpoint}")

    def __enter__(self) -> "PayloadReceiver":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()
