from __future__ import annotations

from .base_decoder import PayloadDecoder


class Utf8TextDecoder(PayloadDecoder):
    """Decode first part as UTF-8 text."""

    def decode(self, parts: list[bytes]) -> str | None:
        if len(parts) == 0:
            return None
        try:
            return parts[0].decode("utf-8")
        except UnicodeDecodeError:
            return None
