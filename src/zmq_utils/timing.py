"""
ZMQ 工具包 - 计时工具

提供延迟统计和计时器功能。
"""

from __future__ import annotations

import time
from collections import deque
from statistics import mean, stdev
from typing import Any


class LatencyStats:
    """延迟统计收集器，支持滚动窗口统计

    用法示例：
        stats = LatencyStats(window_size=100)
        stats.record(latency_ms)
        print(stats.get_stats())
    """

    def __init__(self, window_size: int = 100) -> None:
        """
        Args:
            window_size: 滚动窗口大小，保留最近 N 次测量
        """
        self._latencies: deque[float] = deque(maxlen=window_size)
        self._total_count: int = 0

    def record(self, latency_ms: float) -> None:
        """记录一次延迟测量（毫秒）"""
        self._latencies.append(latency_ms)
        self._total_count += 1

    def get_stats(self) -> dict[str, float]:
        """获取统计数据

        Returns:
            包含 avg, min, max, std, count 的字典
        """
        if not self._latencies:
            return {"avg": 0.0, "min": 0.0, "max": 0.0, "std": 0.0, "count": 0}
        lat_list = list(self._latencies)
        return {
            "avg": mean(lat_list),
            "min": min(lat_list),
            "max": max(lat_list),
            "std": stdev(lat_list) if len(lat_list) > 1 else 0.0,
            "count": self._total_count,
        }

    def get_avg(self) -> float:
        """快速获取平均延迟"""
        return mean(self._latencies) if self._latencies else 0.0

    def reset(self) -> None:
        """重置统计数据"""
        self._latencies.clear()
        self._total_count = 0

    def __repr__(self) -> str:
        stats = self.get_stats()
        return f"LatencyStats(avg={stats['avg']:.2f}ms, count={stats['count']})"


class Timer:
    """简单计时器，用于测量代码段执行时间

    用法示例：
        timer = Timer()
        timer.start()
        # ... 执行代码 ...
        elapsed = timer.stop()  # 返回毫秒

        # 或使用 with 语法
        with Timer() as t:
            # ... 执行代码 ...
        print(t.elapsed_ms)
    """

    def __init__(self) -> None:
        self._start_ns: int = 0
        self._elapsed_ms: float = 0.0

    def start(self) -> "Timer":
        """开始计时"""
        self._start_ns = time.perf_counter_ns()
        return self

    def stop(self) -> float:
        """停止计时并返回耗时（毫秒）"""
        self._elapsed_ms = (time.perf_counter_ns() - self._start_ns) / 1_000_000
        return self._elapsed_ms

    @property
    def elapsed_ms(self) -> float:
        """获取上次测量的耗时（毫秒）"""
        return self._elapsed_ms

    def __enter__(self) -> "Timer":
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop()


class LatencyTracker:
    """延迟追踪器，用于统计模型推理时间

    用法示例：
        tracker = LatencyTracker()

        # 运行模型时记录推理时间
        with tracker.track_model():
            result = model.predict(image)

        # 获取统计
        print(tracker.get_summary())
    """

    def __init__(self, window_size: int = 100) -> None:
        self.model_stats = LatencyStats(window_size)

    def record_model(self, latency_ms: float) -> None:
        """记录模型推理时间"""
        self.model_stats.record(latency_ms)

    def track_model(self) -> Timer:
        """返回计时器用于 with 语法追踪模型推理时间

        用法：
            with tracker.track_model() as t:
                result = model.predict(image)
            # 自动记录到 model_stats
        """
        return _ModelTimerContext(self)

    def get_summary(self) -> dict[str, float]:
        """获取统计摘要"""
        return self.model_stats.get_stats()

    def print_summary(self) -> None:
        """打印统计摘要"""
        model = self.model_stats.get_stats()
        print(f"[Model] avg: {model['avg']:.2f}ms, count: {int(model['count'])}")

    def reset(self) -> None:
        """重置统计"""
        self.model_stats.reset()


class _ModelTimerContext(Timer):
    """内部类：用于自动记录模型推理时间"""

    def __init__(self, tracker: LatencyTracker) -> None:
        super().__init__()
        self._tracker = tracker

    def __exit__(self, *args: Any) -> None:
        super().__exit__(*args)
        self._tracker.record_model(self.elapsed_ms)
