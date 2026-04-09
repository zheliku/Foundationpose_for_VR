from __future__ import annotations

import json
from typing import Any

import zmq


"""
通用传输发送层（不包含业务语义）。

职责：
1) 管理 ZMQ socket 生命周期（创建、连接、关闭）。
2) 提供统一发送接口（payload/multipart/raw/text/json）。
3) 支持两种模式：
     - send_topic=False -> PUSH（点对点/管道式）
     - send_topic=True  -> PUB（发布订阅式，首帧为 topic）

不负责：
- 图像/深度/位姿等业务编码。
    业务编码由 zmq_utils.payload.encoder 中的 Encoder 负责。
"""


class PayloadSender:
    """通用 payload 发送器，支持可选 topic 前缀（PUSH/PUB）。

    参数说明：
    - endpoint: ZMQ 地址（如 tcp://127.0.0.1:5555）。
    - hwm: 高水位，控制积压上限。
    - bind: True 表示服务端 bind；False 表示客户端 connect。
    - send_topic: True 时使用 PUB 并发送 topic；False 使用 PUSH。
    - default_topic: send_topic=True 且 send_payload 未传 topic 时使用。
    """

    ctx: zmq.Context[zmq.Socket[bytes]]
    socket: zmq.Socket[bytes] | None
    endpoint: str
    hwm: int
    is_bind: bool

    def __init__(
        self,
        endpoint: str,
        hwm: int = 1,
        bind: bool = False,
        send_topic: bool = False,
        default_topic: str | None = None,
    ) -> None:
        self.ctx = zmq.Context.instance()
        self.socket = None
        self.endpoint = endpoint
        self.hwm = hwm
        self.is_bind = bind
        self.send_topic = send_topic
        self.default_topic = default_topic
        self._setup_socket()

    """创建 socket 并执行 bind/connect。"""

    def _setup_socket(self) -> None:
        socket_type = zmq.PUB if self.send_topic else zmq.PUSH
        self.socket = self.ctx.socket(socket_type)
        self.socket.set_hwm(self.hwm)
        if self.is_bind:
            self.socket.bind(self.endpoint)
            print(f"[{self.__class__.__name__}] Bound to {self.endpoint}")
        else:
            self.socket.connect(self.endpoint)
            print(f"[{self.__class__.__name__}] Connected to {self.endpoint}")

    """发送业务 payload（list[bytes]）。

    当 send_topic=True 时，发送格式为：
        [topic_utf8, *parts]
    否则发送格式为：
        [*parts]

    返回：
    - True: 成功发送
    - False: 发送失败（常见于 NOBLOCK 下拥塞）
    """

    def send_payload(self, parts: list[bytes], topic: str | None = None) -> bool:
        if self.socket is None:
            return False

        if not parts:
            return False

        try:
            if self.send_topic:
                actual_topic = topic if topic is not None else self.default_topic
                if not actual_topic:
                    return False
                self.socket.send_multipart(
                    [actual_topic.encode("utf-8"), *parts], flags=zmq.NOBLOCK
                )
            else:
                self.socket.send_multipart(parts, flags=zmq.NOBLOCK)
            return True
        except zmq.Again:
            return False

    """send_payload 的同义别名，便于表达“发送多帧消息”。"""

    def send_multipart(self, parts: list[bytes], topic: str | None = None) -> bool:
        return self.send_payload(parts, topic=topic)

    """关闭 socket。"""

    def close(self) -> None:
        if self.socket is not None:
            self.socket.close()
            self.socket = None
            print(f"[ZMQ] Node closed: {self.endpoint}")

    def __enter__(self) -> "PayloadSender":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()
