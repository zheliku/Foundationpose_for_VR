from __future__ import annotations

from abc import ABC, abstractmethod


class PayloadDecoder(ABC):
    """Base payload decoder interface."""

    @abstractmethod
    def decode(self, parts: list[bytes]) -> object | None:
        raise NotImplementedError
