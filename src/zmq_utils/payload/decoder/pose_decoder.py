from __future__ import annotations

from .base_decoder import PayloadDecoder
from ..message.pose import PoseMsg


class PoseDecoder(PayloadDecoder):
    """Decode one-part pose server JSON payload."""

    def decode(self, parts: list[bytes]) -> dict[str, object] | None:
        if len(parts) < 1:
            return None

        message = PoseMsg.from_json_bytes(parts[0])
        if message is None:
            return None

        return message.to_dict()
