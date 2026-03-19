"""
ZMQ 工具包

提供基于 ZeroMQ 的 payload 收发、编解码与延迟测量工具。

模块结构：
- communicate: 通用 payload 收发 (PayloadSender, PayloadReceiver)
- payload: 通用 payload 编解码
- timing: 计时工具 (LatencyStats, Timer, LatencyTracker)
- latency: 网络延迟测量 (LatencyProbe)

使用示例：
    from zmq_utils import PayloadSender, PayloadReceiver, RGBDEncoder, LatencyProbe
"""

# 通用 payload 收发
from .communicate import PayloadReceiver, PayloadSender

# 通用 payload 编解码
from .payload import (
    PayloadDecoder,
    RGBDDecoder,
    RGBDEncoder,
    StereoJpegDecoder,
    TrackingDecoder,
    TrackingEncoder,
    Utf8TextDecoder,
)

# 计时工具
from .timing import LatencyStats, LatencyTracker, Timer

# 网络延迟测量
from .latency import LatencyProbe, measure_network_latency


__all__ = [
    # Payload 收发
    "PayloadSender",
    "PayloadReceiver",
    # Payload 编解码
    "PayloadDecoder",
    "StereoJpegDecoder",
    "RGBDEncoder",
    "RGBDDecoder",
    "TrackingEncoder",
    "TrackingDecoder",
    "Utf8TextDecoder",
    # 计时
    "LatencyStats",
    "Timer",
    "LatencyTracker",
    # 网络延迟
    "LatencyProbe",
    "measure_network_latency",
]
