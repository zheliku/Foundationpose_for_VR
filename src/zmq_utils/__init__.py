"""
ZMQ 工具包

提供基于 ZeroMQ 的 payload 收发与编解码工具。

模块结构：
- communicate: 通用 payload 收发 (PayloadSender, PayloadReceiver)
- payload: 通用 payload 编解码

使用示例：
    from zmq_utils import PayloadSender, PayloadReceiver, RGBDEncoder
"""

# 通用 payload 收发
from .communicate import PayloadReceiver, PayloadSender

# 通用 payload 编解码
from .payload import (
    BaseEncoder,
    PayloadDecoder,
    PoseServerDecoder,
    PoseServerEncoder,
    RGBDDecoder,
    RGBDEncoder,
    StereoJpegDecoder,
    TrackingDecoder,
    TrackingEncoder,
    Utf8TextDecoder,
)


__all__ = [
    # Payload 收发
    "PayloadSender",
    "PayloadReceiver",
    # Payload 编解码
    "BaseEncoder",
    "PayloadDecoder",
    "StereoJpegDecoder",
    "RGBDEncoder",
    "RGBDDecoder",
    "TrackingEncoder",
    "TrackingDecoder",
    "Utf8TextDecoder",
    "PoseServerEncoder",
    "PoseServerDecoder",
]
