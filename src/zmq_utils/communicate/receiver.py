"""通用传输接收层（不包含业务语义）。

职责：
1) 管理 ZMQ socket 生命周期（创建、监听、关闭）。
2) 提供统一接收接口（single payload + topic）。
3) 统一使用 SUB 模式，首帧是 topic。

说明：
- recv_all_latest_by_topic 按 topic 分别 drain，适合多 topic 实时流场景。
- recv_frame_latest 不区分 topic drain，仅适合单 topic 场景。
"""

from __future__ import annotations

from typing import Any

import zmq


class PayloadReceiver:
    """通用 payload 接收器（SUB 模式，支持多 topic 订阅）。

    参数说明：
    - endpoint: ZMQ 地址。
    - hwm: 高水位，控制接收侧积压上限。
    - bind: True=bind，False=connect。
    - topics: 订阅的 topic 列表（字符串或字符串列表）。
    """

    # ZMQ 上下文与底层 socket。
    ctx: zmq.Context[zmq.Socket[bytes]]
    socket: zmq.Socket[bytes] | None

    # 连接配置。
    endpoint: str
    hwm: int
    is_bind: bool

    # 订阅配置。
    topics: list[str]  # 订阅的 topic 列表。

    def __init__(
        self,
        endpoint: str,
        hwm: int = 20,
        bind: bool = True,
        topics: list[str] | str | None = None,
    ) -> None:
        self.ctx = zmq.Context.instance()
        self.socket = None
        self.endpoint = endpoint
        self.hwm = hwm
        self.is_bind = bind

        # 统一转为 list[str]。
        if topics is None:
            self.topics = [""]
        elif isinstance(topics, str):
            self.topics = [topics]
        else:
            self.topics = list(topics)

        self._setup_socket()

    def _setup_socket(self) -> None:
        """创建 SUB socket 并执行 bind/connect + 订阅所有 topics。"""
        self.socket = self.ctx.socket(zmq.SUB)
        self.socket.set_hwm(self.hwm)

        # 订阅所有配置的 topics。
        for topic in self.topics:
            self.socket.setsockopt_string(zmq.SUBSCRIBE, topic)

        if self.is_bind:
            self.socket.bind(self.endpoint)
            print(f"[{self.__class__.__name__}] Bound to {self.endpoint}")
        else:
            self.socket.connect(self.endpoint)
            print(f"[{self.__class__.__name__}] Connected to {self.endpoint}")

    def _recv_frame_once(self, *, nonblock: bool) -> tuple[str, bytes] | None:
        """读取一条 SUB 消息，返回 (topic, payload)。"""
        if self.socket is None:
            return None

        flags = zmq.NOBLOCK if nonblock else 0

        try:
            msg = self.socket.recv_multipart(flags=flags)
            # SUB 协议固定为 [topic, payload]。
            if len(msg) != 2:
                return None
            return msg[0].decode("utf-8", errors="replace"), bytes(msg[1])
        except zmq.ZMQError:
            return None

    def recv_all_latest_by_topic(
        self, timeout_ms: int = 0
    ) -> dict[str, bytes] | None:
        """按 topic 分别 drain，返回每个 topic 的最新 payload。

        行为：
        - timeout_ms=-1 时阻塞等待第一条消息。
        - 其余场景先 poll，再读取一条并非阻塞 drain 队列中所有剩余消息。
        - 每个 topic 仅保留最新一条，旧消息被丢弃。
        - 返回 None 表示超时无消息；否则返回 {topic: payload} 字典。
        """
        if self.socket is None:
            return None

        try:
            # 第一步：等待第一条消息。
            if timeout_ms == -1:
                first = self._recv_frame_once(nonblock=False)
            else:
                if not self.socket.poll(timeout=max(int(timeout_ms), 0)):
                    return None
                first = self._recv_frame_once(nonblock=True)

            if first is None:
                return None

            # 第二步：非阻塞 drain 队列，按 topic 分别保留最新消息。
            latest: dict[str, bytes] = {first[0]: first[1]}
            while True:
                result = self._recv_frame_once(nonblock=True)
                if result is None:
                    break
                latest[result[0]] = result[1]

            return latest
        except zmq.ZMQError:
            return None

    def recv_frame_latest(
        self, timeout_ms: int = 0
    ) -> tuple[str, bytes] | None:
        """读取并返回最新一条消息，格式为 (topic, payload)。

        注意：此方法不区分 topic drain，多 topic 场景下会丢失非最后 topic 的消息。
        多 topic 场景请使用 recv_all_latest_by_topic。

        行为：
        - timeout_ms=-1 时阻塞等待一条消息。
        - 其余场景先 poll，再读取一条并非阻塞 drain 到队尾。
        - 仅返回最后一条消息，旧消息被丢弃。
        """
        if self.socket is None:
            return None

        try:
            if timeout_ms == -1:
                result = self._recv_frame_once(nonblock=False)
            else:
                if not self.socket.poll(timeout=max(int(timeout_ms), 0)):
                    return None
                result = self._recv_frame_once(nonblock=True)

            if result is None:
                return None

            # 主动 drain 队列，仅保留最新消息。
            while True:
                newer = self._recv_frame_once(nonblock=True)
                if newer is None:
                    break
                result = newer

            return result
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
