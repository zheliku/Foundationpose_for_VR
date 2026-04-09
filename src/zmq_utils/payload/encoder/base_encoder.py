from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseEncoder(ABC):
    """Base payload encoder interface."""

    @abstractmethod
    def encode(self, *args: Any, **kwargs: Any) -> bytes | None:
        raise NotImplementedError
