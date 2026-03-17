"""
ZMQ 工具包 - JSON/Pose 数据传输

提供 JSON 格式数据（如位姿信息）的发布/订阅功能。
"""

from __future__ import annotations

import json
from typing import Any

from .base import PubNode, SubNode


class PosePublisher(PubNode):
    """位姿/JSON 数据发布节点"""

    def publish_pose(self, topic: str, data_dict: dict[str, Any]) -> None:
        """发布位姿数据（JSON 格式）"""
        json_str = json.dumps(data_dict)
        self.publish_raw(topic, json_str.encode("utf-8"))


class PoseSubscriber(SubNode):
    """位姿/JSON 数据订阅节点"""

    def recv_pose(self, timeout_ms: int = 10) -> dict[str, Any] | None:
        """接收位姿数据"""
        data = self.recv_raw_latest(timeout_ms)
        if data:
            return json.loads(data.decode("utf-8"))
        return None
