from __future__ import annotations

from abc import ABC, abstractmethod


class PayloadDecoder(ABC):
    """业务 payload 解码器抽象接口。

    约定：
    - 输入永远是网络层收到的单帧 bytes；
    - 输出由具体业务解码器决定，例如 QuestStereoMsg、PoseMsg 或 RGBD 元组；
    - 返回 None 表示 payload 非法或解码失败，上层应直接丢弃该帧。
    """

    @abstractmethod
    def decode(self, payload: bytes) -> object | None:
        """解码单帧 payload；具体协议由子类实现。"""
        raise NotImplementedError
