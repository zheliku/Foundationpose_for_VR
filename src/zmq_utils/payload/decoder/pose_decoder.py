from __future__ import annotations

import json

from .base_decoder import PayloadDecoder


class PoseDecoder(PayloadDecoder):
    """Decode one-part pose server JSON payload."""

    def decode(self, parts: list[bytes]) -> dict[str, object] | None:
        if len(parts) < 1:
            return None

        try:
            text = parts[0].decode("utf-8")
            obj = json.loads(text)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None

        if not isinstance(obj, dict):
            return None

        return obj
