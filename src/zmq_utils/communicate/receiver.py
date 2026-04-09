from __future__ import annotations

import threading
from typing import Any

import zmq


"""
通用传输接收层（不包含业务语义）。

职责：
1) 管理 ZMQ socket 生命周期（创建、监听、关闭）。
2) 提供统一接收接口（packet/payload）。
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
        self._latest_cond = threading.Condition()
        self._latest_active = False
        self._latest_thread: threading.Thread | None = None
        self._latest_poll_timeout_ms = 10
        self._latest_packet: tuple[str, list[bytes]] | None = None
        self._latest_packet_drained = 0
        self._setup_socket()

    def _latest_loop(self) -> None:
        """后台线程：持续拉取最新 packet 并覆盖缓存。"""
        while self._latest_active:
            packet = self.recv_packet_latest(timeout_ms=self._latest_poll_timeout_ms)
            if packet is None:
                continue

            drained = int(self.last_drain_count)
            with self._latest_cond:
                self._latest_packet = packet
                self._latest_packet_drained = drained
                self._latest_cond.notify_all()

    def start_latest_buffer(self, poll_timeout_ms: int = 10) -> None:
        """启动后台 latest 缓冲循环。"""
        if self._latest_active:
            return

        self._latest_poll_timeout_ms = max(int(poll_timeout_ms), 1)
        with self._latest_cond:
            self._latest_packet = None
            self._latest_packet_drained = 0

        self._latest_active = True
        self._latest_thread = threading.Thread(
            target=self._latest_loop,
            name="PayloadReceiverLatestLoop",
            daemon=True,
        )
        self._latest_thread.start()

    def stop_latest_buffer(self) -> None:
        """停止后台 latest 缓冲循环。"""
        if not self._latest_active:
            return

        self._latest_active = False
        with self._latest_cond:
            self._latest_cond.notify_all()

        if self._latest_thread is not None:
            self._latest_thread.join(timeout=0.5)
            self._latest_thread = None

        with self._latest_cond:
            self._latest_packet = None
            self._latest_packet_drained = 0

    def pop_latest_packet(
        self, timeout_ms: int = 0
    ) -> tuple[str, list[bytes], int] | None:
        """从后台缓冲中弹出最新 packet。

        返回值：(topic, payload_parts, drained_count)
        - 当未开启后台缓冲时，退化为单次 recv_packet_latest 调用。
        """
        if not self._latest_active:
            packet = self.recv_packet_latest(timeout_ms=timeout_ms)
            if packet is None:
                return None
            return packet[0], packet[1], int(self.last_drain_count)

        timeout_sec: float | None
        if timeout_ms < 0:
            timeout_sec = None
        else:
            timeout_sec = max(int(timeout_ms), 0) / 1000.0

        with self._latest_cond:
            if self._latest_packet is None:
                self._latest_cond.wait(timeout=timeout_sec)

            packet = self._latest_packet
            drained = self._latest_packet_drained
            self._latest_packet = None
            self._latest_packet_drained = 0

        if packet is None:
            return None
        return packet[0], packet[1], drained

    def pop_latest_payload(self, timeout_ms: int = 0) -> tuple[list[bytes], int] | None:
        """从后台缓冲中弹出最新 payload。

        返回值：(payload_parts, drained_count)
        """
        packet = self.pop_latest_packet(timeout_ms=timeout_ms)
        if packet is None:
            return None

        _, parts, drained = packet
        return parts, drained

    def _setup_socket(self) -> None:
        """创建 socket 并执行 bind/connect。"""
        socket_type = zmq.SUB if self.use_topic else zmq.PULL
        self.socket = self.ctx.socket(socket_type)
        self.socket.set_hwm(self.hwm)

        if self.use_topic:
            self.socket.setsockopt_string(zmq.SUBSCRIBE, self.topic)

        if self.is_bind:
            self.socket.bind(self.endpoint)
            print(f"[{self.__class__.__name__}] Bound to {self.endpoint}")
        else:
            self.socket.connect(self.endpoint)
            print(f"[{self.__class__.__name__}] Connected to {self.endpoint}")

    def _recv_message_latest(self, timeout_ms: int = 0) -> list[bytes] | None:
        """读取并返回最新一条 multipart 消息。

        行为：
        - timeout_ms=-1 时阻塞等待一条消息。
        - 其余场景先 poll，再读取一条并非阻塞 drain 到队尾。
        - 仅返回最后一条消息，旧消息被丢弃。
        """
        if self.socket is None:
            return None

        try:
            if timeout_ms == -1:
                msg = self.socket.recv_multipart()
            else:
                if not self.socket.poll(timeout=max(int(timeout_ms), 0)):
                    self.last_drain_count = 0
                    return None
                try:
                    msg = self.socket.recv_multipart(flags=zmq.NOBLOCK)
                except zmq.Again:
                    self.last_drain_count = 0
                    return None

            drained = 0
            while True:
                try:
                    msg = self.socket.recv_multipart(flags=zmq.NOBLOCK)
                    drained += 1
                except zmq.Again:
                    break

            self.last_drain_count = drained
            return msg
        except zmq.ZMQError:
            self.last_drain_count = 0
            return None

    def recv_packet_latest(self, timeout_ms: int = 0) -> tuple[str, list[bytes]] | None:
        """接收最新完整 packet。

        返回：
        - use_topic=True  -> (topic, payload_parts)
        - use_topic=False -> ("", payload_parts)
        """
        msg = self._recv_message_latest(timeout_ms=timeout_ms)
        if msg is None:
            return None

        if self.use_topic:
            if len(msg) == 0:
                self.last_drain_count = 0
                return None
            topic = msg[0].decode("utf-8", errors="ignore")
            return topic, list(msg[1:])

        return "", list(msg)

    def recv_payload_latest(self, timeout_ms: int = 0) -> list[bytes] | None:
        """仅返回最新 payload（不含 topic）。"""
        packet = self.recv_packet_latest(timeout_ms=timeout_ms)
        if packet is None:
            return None
        _, parts = packet
        return parts

    def close(self) -> None:
        """关闭 socket。"""
        self.stop_latest_buffer()
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
