"""
ZMQ 工具包 - 网络延迟测量

提供基于 RTT (Round-Trip Time) 的精确网络延迟测量。
这种方法不需要时钟同步，因为测量完全在同一台机器上完成。
"""

from __future__ import annotations

import time
import threading
from typing import Callable

import zmq

from .timing import LatencyStats


class LatencyProbe:
    """网络延迟探测器

    使用 Ping-Pong 方式测量 RTT，然后除以 2 得到单程延迟估计。
    这种方法完全不需要时钟同步！

    用法示例 - 服务器端：
        probe_server = LatencyProbe.create_server("tcp://*:5560")
        probe_server.start()  # 在后台响应 ping

    用法示例 - 客户端：
        probe_client = LatencyProbe.create_client("tcp://server_ip:5560")
        rtt, one_way = probe_client.ping()
        print(f"RTT: {rtt:.2f}ms, 单程延迟: {one_way:.2f}ms")
    """

    def __init__(self, endpoint: str, is_server: bool = False) -> None:
        """
        Args:
            endpoint: ZMQ 地址
            is_server: True=服务端(响应ping), False=客户端(发送ping)
        """
        self.endpoint = endpoint
        self.is_server = is_server
        self.ctx = zmq.Context.instance()

        if is_server:
            self.socket = self.ctx.socket(zmq.REP)
            self.socket.bind(endpoint)
        else:
            self.socket = self.ctx.socket(zmq.REQ)
            self.socket.connect(endpoint)

        self._running = False
        self._thread: threading.Thread | None = None
        self.stats = LatencyStats(window_size=100)

    @classmethod
    def create_server(cls, endpoint: str) -> "LatencyProbe":
        """创建服务器端探测器"""
        return cls(endpoint, is_server=True)

    @classmethod
    def create_client(cls, endpoint: str) -> "LatencyProbe":
        """创建客户端探测器"""
        return cls(endpoint, is_server=False)

    def ping(self, timeout_ms: int = 1000) -> tuple[float, float] | None:
        """发送 ping 并返回 RTT 和估计的单程延迟（毫秒）

        Returns:
            (rtt_ms, one_way_ms) 或 None（超时）
        """
        if self.is_server:
            raise RuntimeError("服务端不能发送 ping，请使用客户端")

        self.socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, timeout_ms)

        try:
            start = time.perf_counter_ns()
            self.socket.send(b"ping")
            self.socket.recv()  # 等待 pong
            rtt_ns = time.perf_counter_ns() - start
            rtt_ms = rtt_ns / 1_000_000
            one_way_ms = rtt_ms / 2
            self.stats.record(one_way_ms)
            return rtt_ms, one_way_ms
        except zmq.Again:
            return None  # 超时

    def ping_continuous(
        self,
        count: int = 10,
        interval_ms: int = 100,
        callback: Callable[[int, float, float], None] | None = None,
    ) -> dict[str, float]:
        """连续发送多次 ping 并返回统计结果

        Args:
            count: ping 次数
            interval_ms: 每次 ping 之间的间隔
            callback: 每次 ping 后的回调函数 (index, rtt, one_way)

        Returns:
            统计结果字典 (avg, min, max, std, count)
        """
        for i in range(count):
            result = self.ping()
            if result and callback:
                callback(i, result[0], result[1])
            time.sleep(interval_ms / 1000)
        return self.stats.get_stats()

    def start(self) -> None:
        """启动服务端后台响应线程"""
        if not self.is_server:
            raise RuntimeError("只有服务端才能启动后台响应")
        self._running = True
        self._thread = threading.Thread(target=self._serve_loop, daemon=True)
        self._thread.start()
        print(f"[LatencyProbe] Server started on {self.endpoint}")

    def _serve_loop(self) -> None:
        """服务端响应循环"""
        self.socket.setsockopt(zmq.RCVTIMEO, 100)
        while self._running:
            try:
                msg = self.socket.recv()
                self.socket.send(msg)  # 立即回复
            except zmq.Again:
                continue  # 超时，继续
            except zmq.ZMQError:
                break

    def stop(self) -> None:
        """停止服务"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        self.socket.close()

    def close(self) -> None:
        """关闭连接"""
        self.stop()

    def __enter__(self) -> "LatencyProbe":
        return self

    def __exit__(self, *args) -> None:
        self.close()


def measure_network_latency(
    server_ip: str, port: int = 5560, ping_count: int = 20
) -> dict[str, float]:
    """便捷函数：测量到服务器的网络延迟

    注意：需要服务器端运行 LatencyProbe.create_server()

    Args:
        server_ip: 服务器 IP 地址
        port: 探测端口
        ping_count: ping 次数

    Returns:
        延迟统计 (avg, min, max, std 单位毫秒)
    """
    with LatencyProbe.create_client(f"tcp://{server_ip}:{port}") as probe:
        print(f"[LatencyProbe] Measuring latency to {server_ip}:{port}...")
        stats = probe.ping_continuous(
            count=ping_count,
            callback=lambda i, rtt, ow: print(
                f"  Ping {i+1}/{ping_count}: RTT={rtt:.2f}ms, OneWay={ow:.2f}ms"
            ),
        )
        print(f"[LatencyProbe] Average one-way latency: {stats['avg']:.2f}ms")
        return stats
