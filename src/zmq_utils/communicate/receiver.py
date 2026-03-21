from __future__ import annotations

import json
from typing import Any

import zmq


"""
通用传输接收层（不包含业务语义）。

职责：
1) 管理 ZMQ socket 生命周期（创建、监听、关闭）。
2) 提供统一接收接口（packet/payload/raw/text/json）。
3) 支持两种模式：
   - use_topic=False -> PULL（不含 topic）
   - use_topic=True  -> SUB（首帧是 topic）

说明：
- recv_*_latest 会主动清空积压，仅保留最新消息，适合实时流场景。
"""


class PayloadReceiver:
    """通用 payload 接收器，支持可选 topic 前缀（PULL/SUB）。"""

    ctx: zmq.Context[zmq.Socket[bytes]]
    socket: zmq.Socket[bytes] | None
    endpoint: str
    hwm: int
    is_bind: bool

    def __init__(
        self,
        endpoint: str,
        hwm: int = 1,
        bind: bool = True,
        use_topic: bool = False,
        topic: str = "",
    ) -> None:
        self.ctx = zmq.Context.instance()
        self.socket = None
        self.endpoint = endpoint
        self.hwm = hwm
        self.is_bind = bind
        self.use_topic = use_topic
        self.topic = topic
        self.last_drain_count = 0
        self._setup_socket()

    """创建 socket 并执行 bind/connect。"""

    def _setup_socket(self) -> None:
        socket_type = zmq.SUB if self.use_topic else zmq.PULL
        self.socket = self.ctx.socket(socket_type)
        self.socket.set_hwm(self.hwm)

        if self.use_topic:
            self.socket.setsockopt(zmq.CONFLATE, 1)
            self.socket.setsockopt_string(zmq.SUBSCRIBE, self.topic)

        if self.is_bind:
            self.socket.bind(self.endpoint)
            print(f"[{self.__class__.__name__}] Bound to {self.endpoint}")
        else:
            self.socket.connect(self.endpoint)
            print(f"[{self.__class__.__name__}] Connected to {self.endpoint}")

    """读取并返回“最新一条消息”（multipart）。

    行为：
    - 先 poll 等待可读。
    - 读取一条后继续 drain，直到队列为空。
    - 只返回最后一条消息。
    """

    def _recv_message_latest(self, timeout_ms: int = 0) -> list[bytes] | None:
        if self.socket is None:
            return None

        try:
            if timeout_ms != -1 and not self.socket.poll(timeout=timeout_ms):
                self.last_drain_count = 0
                return None

            msg = self.socket.recv_multipart()
            drained = 0
            while int(self.socket.get(zmq.EVENTS)) & zmq.POLLIN:
                msg = self.socket.recv_multipart()
                drained += 1
            self.last_drain_count = drained
            return msg
        except zmq.ZMQError:
            self.last_drain_count = 0
            return None

    """接收完整 packet。

    返回：
    - use_topic=True  -> (topic, payload_parts)
    - use_topic=False -> ("", payload_parts)
    """

    def recv_packet(self, timeout_ms: int = 0) -> tuple[str, list[bytes]] | None:
        msg = self._recv_message_latest(timeout_ms=timeout_ms)
        if msg is None:
            return None

        if self.use_topic:
            if len(msg) == 0:
                return None
            topic = msg[0].decode("utf-8", errors="ignore")
            return topic, list(msg[1:])

        return "", list(msg)

    """仅返回 payload 部分（不含 topic）。"""

    def recv_payload(self, timeout_ms: int = 0) -> list[bytes] | None:
        packet = self.recv_packet(timeout_ms=timeout_ms)
        if packet is None:
            return None
        _, parts = packet
        return parts

    """返回 payload 第一帧（原始字节）。"""

    def recv_raw_latest(self, timeout_ms: int = 0) -> bytes | None:
        parts = self.recv_payload(timeout_ms=timeout_ms)
        if not parts:
            return None
        return parts[0]

    """返回 payload 第一帧（UTF-8 文本）。"""

    def recv_text_latest(
        self, timeout_ms: int = 0, encoding: str = "utf-8"
    ) -> str | None:
        data = self.recv_raw_latest(timeout_ms=timeout_ms)
        if data is None:
            return None
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            return None

    """返回 payload 第一帧（JSON 对象）。"""

    def recv_json_latest(self, timeout_ms: int = 0) -> Any | None:
        text = self.recv_text_latest(timeout_ms=timeout_ms)
        if text is None:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    """关闭 socket。"""

    def close(self) -> None:
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
