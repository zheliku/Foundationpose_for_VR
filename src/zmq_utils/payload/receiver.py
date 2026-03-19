from __future__ import annotations

from ..base import PullNode


class MultipartReceiver(PullNode):
    """通用 multipart 接收器（PULL）"""

    def recv_payload(self, timeout_ms: int = 0) -> list[bytes] | None:
        return self.recv_multipart_latest(timeout_ms)
