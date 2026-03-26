from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray

try:
    import pyrealsense2 as rs
except ImportError as exc:
    raise SystemExit("未找到 pyrealsense2，请先安装 RealSense Python SDK。") from exc

rs_any = cast(Any, rs)


@dataclass(slots=True)
class RGBDFrame:
    color_bgr: NDArray[np.uint8]
    depth_m: NDArray[np.float64]
    timestamp_s: float


class RGBDSource(Protocol):
    def start(self) -> None: ...

    def read(self) -> RGBDFrame | None: ...

    def stop(self) -> None: ...


@dataclass(slots=True)
class RealSenseConfig:
    width: int = 640
    height: int = 480
    fps: int = 30
    align_to_color: bool = True
    min_depth_m: float = 0.1
    max_depth_m: float = 3.0


class RealSenseRGBDSource:
    def __init__(self, config: RealSenseConfig) -> None:
        self.config = config
        self._pipeline = rs_any.pipeline()
        self._profile = None
        self._align = None
        self._depth_scale = 0.001
        self.cam_k = np.eye(3, dtype=np.float64)

    def start(self) -> None:
        rs_config = rs_any.config()
        rs_config.enable_stream(
            rs_any.stream.color,
            self.config.width,
            self.config.height,
            rs_any.format.bgr8,
            self.config.fps,
        )
        rs_config.enable_stream(
            rs_any.stream.depth,
            self.config.width,
            self.config.height,
            rs_any.format.z16,
            self.config.fps,
        )
        self._profile = self._pipeline.start(rs_config)

        if self.config.align_to_color:
            self._align = rs_any.align(rs_any.stream.color)

        depth_sensor = self._profile.get_device().first_depth_sensor()
        self._depth_scale = float(depth_sensor.get_depth_scale())
        color_profile = self._profile.get_stream(
            rs_any.stream.color
        ).as_video_stream_profile()
        intr = color_profile.get_intrinsics()
        self.cam_k = np.array(
            [[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def read(self) -> RGBDFrame | None:
        frames = self._pipeline.wait_for_frames()
        if self._align is not None:
            frames = self._align.process(frames)

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            return None

        color = np.asanyarray(color_frame.get_data())
        depth_m = (
            np.asanyarray(depth_frame.get_data()).astype(np.float64) * self._depth_scale
        )
        invalid = (depth_m < self.config.min_depth_m) | (
            depth_m > self.config.max_depth_m
        )
        depth_m[invalid] = 0.0

        return RGBDFrame(
            color_bgr=color,
            depth_m=depth_m,
            timestamp_s=time.perf_counter(),
        )

    def stop(self) -> None:
        self._pipeline.stop()
