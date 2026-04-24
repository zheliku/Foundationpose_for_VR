"""通用传输发送层（不包含业务语义）。

职责：
1) 管理 ZMQ socket 生命周期（创建、连接、关闭）。
2) 提供统一发送接口（single payload + topic）。
3) 统一使用 PUB 模式，发送格式为 multipart [topic, payload]。

不负责：
- 图像/深度/位姿等业务编码。
  业务编码由 zmq_utils.payload.encoder 中的 Encoder 负责。
"""

from __future__ import annotations

from typing import Any

import zmq


class PayloadSender:
    """通用 payload 发送器（PUB 模式，必须指定 topic）。

    参数说明：
    - endpoint: ZMQ 地址（如 tcp://127.0.0.1:5555）。
    - hwm: 高水位，控制积压上限。
    - bind: True 表示服务端 bind；False 表示客户端 connect。
    """

    # ZMQ 上下文与底层 socket。
    ctx: zmq.Context[zmq.Socket[bytes]]
    socket: zmq.Socket[bytes] | None

    # 连接配置。
    endpoint: str  # 完整连接地址。
    hwm: int  # 高水位。
    is_bind: bool  # True=bind，False=connect。

    def __init__(
        self,
        endpoint: str,
        hwm: int = 1,
        bind: bool = False,
    ) -> None:
        self.ctx = zmq.Context.instance()
        self.socket = None
        self.endpoint = endpoint
        self.hwm = hwm
        self.is_bind = bind
        self._setup_socket()

    def _setup_socket(self) -> None:
        """创建 PUB socket 并执行 bind/connect。"""
        self.socket = self.ctx.socket(zmq.PUB)
        self.socket.set_hwm(self.hwm)
        if self.is_bind:
            self.socket.bind(self.endpoint)
            print(f"[{self.__class__.__name__}] Bound to {self.endpoint}")
        else:
            self.socket.connect(self.endpoint)
            print(f"[{self.__class__.__name__}] Connected to {self.endpoint}")

    def send_payload(self, payload: bytes, topic: str) -> bool:
        """发送业务 payload（单帧 bytes），必须指定 topic。

        发送格式：multipart [topic_utf8, payload]。

        返回：
        - True: 成功发送
        - False: 发送失败（常见于 NOBLOCK 下拥塞）
        """
        if self.socket is None:
            return False

        if payload is None or len(payload) == 0:
            return False

        if not topic:
            return False

        try:
            self.socket.send_multipart(
                [topic.encode("utf-8"), bytes(payload)], flags=zmq.NOBLOCK
            )
            return True
        except zmq.Again:
            return False

    def close(self) -> None:
        """关闭 socket。"""
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
