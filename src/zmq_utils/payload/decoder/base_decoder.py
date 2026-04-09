from __future__ import annotations

from abc import ABC, abstractmethod


class PayloadDecoder(ABC):
    """Base payload decoder interface."""

    @abstractmethod
    def decode(self, payload: bytes) -> object | None:
        raise NotImplementedError
