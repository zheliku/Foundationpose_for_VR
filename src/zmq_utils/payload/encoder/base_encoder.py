from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseEncoder(ABC):
    """业务 payload 编码器抽象接口。

    约定：
    - 子类把业务对象编码为单帧 bytes；
    - 传输层只关心 bytes 和 topic，不关心具体业务字段；
    - 返回 None 表示当前输入不足或编码失败，发送端应跳过该帧。
    """

    @abstractmethod
    def encode(self, *args: Any, **kwargs: Any) -> bytes | None:
        """编码业务数据为单帧 payload；参数格式由子类自行定义。"""
        raise NotImplementedError
