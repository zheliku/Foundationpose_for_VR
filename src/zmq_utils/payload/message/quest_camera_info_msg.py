"""Quest 相机静态信息传输消息。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import msgpack


@dataclass(frozen=True)
class QuestCameraInfoMsg:
    """Quest 相机静态信息消息（单次传输即包含左右目全部标定参数）。

    字段来源：PassthroughCameraAccess.Intrinsics + 额外传感器信息。
    该消息由 Unity 端低频发送，Python 端缓存后即可长期使用。
    """

    # 设备支持状态。
    is_supported: bool  # PassthroughCameraAccess.IsSupported。

    # 左目内参。
    left_fx: float  # 左目焦距 fx（像素）。
    left_fy: float  # 左目焦距 fy（像素）。
    left_cx: float  # 左目主点 cx（像素）。
    left_cy: float  # 左目主点 cy（像素）。

    # 右目内参。
    right_fx: float  # 右目焦距 fx（像素）。
    right_fy: float  # 右目焦距 fy（像素）。
    right_cx: float  # 右目主点 cx（像素）。
    right_cy: float  # 右目主点 cy（像素）。

    # 畸变系数（Quest 通常为空）。
    left_distortion: tuple[float, ...]  # 左目畸变系数。
    right_distortion: tuple[float, ...]  # 右目畸变系数。

    # 双目基线。
    baseline_m: float  # 左右目镜头中心距离（米）。

    # 传感器分辨率（Intrinsics.SensorResolution / activeArraySize）。
    sensor_width: int  # 传感器有效宽度（像素）。
    sensor_height: int  # 传感器有效高度（像素）。

    # 有效阵列区域（activeArraySize）。
    active_left: int  # 有效区域左边界。
    active_top: int  # 有效区域上边界。
    active_right: int  # 有效区域右边界。
    active_bottom: int  # 有效区域下边界。

    # 请求分辨率（RequestedResolution，即 Unity 侧配置的分辨率）。
    left_requested_width: int  # 左目请求宽度（像素）。
    left_requested_height: int  # 左目请求高度（像素）。
    right_requested_width: int  # 右目请求宽度（像素）。
    right_requested_height: int  # 右目请求高度（像素）。

    # 当前运行分辨率（CurrentResolution，实际输出分辨率）。
    current_width: int  # 当前图像宽度（像素）。
    current_height: int  # 当前图像高度（像素）。

    # 帧率。
    max_framerate: int  # 最大帧率。

    # 左目镜头偏移（Intrinsics.LensOffset，相对于 IMU/Gyroscope）。
    left_lens_offset_px: float  # 位置 x（米）。
    left_lens_offset_py: float  # 位置 y（米）。
    left_lens_offset_pz: float  # 位置 z（米）。
    left_lens_offset_qx: float  # 旋转 x。
    left_lens_offset_qy: float  # 旋转 y。
    left_lens_offset_qz: float  # 旋转 z。
    left_lens_offset_qw: float  # 旋转 w。

    # 右目镜头偏移。
    right_lens_offset_px: float  # 位置 x（米）。
    right_lens_offset_py: float  # 位置 y（米）。
    right_lens_offset_pz: float  # 位置 z（米）。
    right_lens_offset_qx: float  # 旋转 x。
    right_lens_offset_qy: float  # 旋转 y。
    right_lens_offset_qz: float  # 旋转 z。
    right_lens_offset_qw: float  # 旋转 w。

    # 发送端时间戳。
    sender_mono_ms: float  # 发送端单调时钟（毫秒）。

    def serialize(self) -> bytes:
        """序列化为 MessagePack 字节。"""
        payload = {
            "is_supported": bool(self.is_supported),
            "left_fx": float(self.left_fx),
            "left_fy": float(self.left_fy),
            "left_cx": float(self.left_cx),
            "left_cy": float(self.left_cy),
            "right_fx": float(self.right_fx),
            "right_fy": float(self.right_fy),
            "right_cx": float(self.right_cx),
            "right_cy": float(self.right_cy),
            "left_distortion": list(self.left_distortion),
            "right_distortion": list(self.right_distortion),
            "baseline_m": float(self.baseline_m),
            "sensor_width": int(self.sensor_width),
            "sensor_height": int(self.sensor_height),
            "active_left": int(self.active_left),
            "active_top": int(self.active_top),
            "active_right": int(self.active_right),
            "active_bottom": int(self.active_bottom),
            "left_requested_width": int(self.left_requested_width),
            "left_requested_height": int(self.left_requested_height),
            "right_requested_width": int(self.right_requested_width),
            "right_requested_height": int(self.right_requested_height),
            "current_width": int(self.current_width),
            "current_height": int(self.current_height),
            "max_framerate": int(self.max_framerate),
            "left_lens_offset_px": float(self.left_lens_offset_px),
            "left_lens_offset_py": float(self.left_lens_offset_py),
            "left_lens_offset_pz": float(self.left_lens_offset_pz),
            "left_lens_offset_qx": float(self.left_lens_offset_qx),
            "left_lens_offset_qy": float(self.left_lens_offset_qy),
            "left_lens_offset_qz": float(self.left_lens_offset_qz),
            "left_lens_offset_qw": float(self.left_lens_offset_qw),
            "right_lens_offset_px": float(self.right_lens_offset_px),
            "right_lens_offset_py": float(self.right_lens_offset_py),
            "right_lens_offset_pz": float(self.right_lens_offset_pz),
            "right_lens_offset_qx": float(self.right_lens_offset_qx),
            "right_lens_offset_qy": float(self.right_lens_offset_qy),
            "right_lens_offset_qz": float(self.right_lens_offset_qz),
            "right_lens_offset_qw": float(self.right_lens_offset_qw),
            "sender_mono_ms": float(self.sender_mono_ms),
        }
        return cast(bytes, msgpack.packb(payload, use_bin_type=True))

    @classmethod
    def deserialize(cls, payload: bytes) -> QuestCameraInfoMsg | None:
        """从 MessagePack 字节反序列化。"""
        try:
            data = msgpack.unpackb(payload, raw=False, strict_map_key=False)
            if not isinstance(data, dict):
                return None

            left_dist_raw = data.get("left_distortion", [])
            right_dist_raw = data.get("right_distortion", [])
            if not isinstance(left_dist_raw, (list, tuple)):
                left_dist_raw = []
            if not isinstance(right_dist_raw, (list, tuple)):
                right_dist_raw = []

            return cls(
                is_supported=bool(data.get("is_supported", True)),
                left_fx=float(data.get("left_fx", 0.0)),
                left_fy=float(data.get("left_fy", 0.0)),
                left_cx=float(data.get("left_cx", 0.0)),
                left_cy=float(data.get("left_cy", 0.0)),
                right_fx=float(data.get("right_fx", 0.0)),
                right_fy=float(data.get("right_fy", 0.0)),
                right_cx=float(data.get("right_cx", 0.0)),
                right_cy=float(data.get("right_cy", 0.0)),
                left_distortion=tuple(float(v) for v in left_dist_raw),
                right_distortion=tuple(float(v) for v in right_dist_raw),
                baseline_m=float(data.get("baseline_m", 0.0)),
                sensor_width=int(data.get("sensor_width", 0)),
                sensor_height=int(data.get("sensor_height", 0)),
                active_left=int(data.get("active_left", 0)),
                active_top=int(data.get("active_top", 0)),
                active_right=int(data.get("active_right", 0)),
                active_bottom=int(data.get("active_bottom", 0)),
                left_requested_width=int(data.get("left_requested_width", 0)),
                left_requested_height=int(data.get("left_requested_height", 0)),
                right_requested_width=int(data.get("right_requested_width", 0)),
                right_requested_height=int(data.get("right_requested_height", 0)),
                current_width=int(data.get("current_width", 0)),
                current_height=int(data.get("current_height", 0)),
                max_framerate=int(data.get("max_framerate", 0)),
                left_lens_offset_px=float(data.get("left_lens_offset_px", 0.0)),
                left_lens_offset_py=float(data.get("left_lens_offset_py", 0.0)),
                left_lens_offset_pz=float(data.get("left_lens_offset_pz", 0.0)),
                left_lens_offset_qx=float(data.get("left_lens_offset_qx", 0.0)),
                left_lens_offset_qy=float(data.get("left_lens_offset_qy", 0.0)),
                left_lens_offset_qz=float(data.get("left_lens_offset_qz", 0.0)),
                left_lens_offset_qw=float(data.get("left_lens_offset_qw", 1.0)),
                right_lens_offset_px=float(data.get("right_lens_offset_px", 0.0)),
                right_lens_offset_py=float(data.get("right_lens_offset_py", 0.0)),
                right_lens_offset_pz=float(data.get("right_lens_offset_pz", 0.0)),
                right_lens_offset_qx=float(data.get("right_lens_offset_qx", 0.0)),
                right_lens_offset_qy=float(data.get("right_lens_offset_qy", 0.0)),
                right_lens_offset_qz=float(data.get("right_lens_offset_qz", 0.0)),
                right_lens_offset_qw=float(data.get("right_lens_offset_qw", 1.0)),
                sender_mono_ms=float(data.get("sender_mono_ms", 0.0)),
            )
        except (TypeError, ValueError):
            return None
