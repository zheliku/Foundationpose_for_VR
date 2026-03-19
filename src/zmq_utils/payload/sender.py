from __future__ import annotations

import zmq

from ..base import PubNode, PushNode


class MultipartSender(PushNode):
    """通用 multipart 发送器（PUSH）"""

    def send_payload(self, parts: list[bytes]) -> bool:
        return self.send_multipart(parts)


class TopicPayloadSender(PubNode):
    """通用 topic + payload 发送器（PUB）"""

    def send_payload(self, topic: str, parts: list[bytes]) -> bool:
        if self.socket is None:
            return False
        try:
            self.socket.send_multipart(
                [topic.encode("utf-8"), *parts], flags=zmq.NOBLOCK
            )
            return True
        except zmq.Again:
            return False
