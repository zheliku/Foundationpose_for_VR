from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PoseMsg:
    """Pose 传输消息：用于 pose 编码器与解码器之间统一交换。"""

    timestamp_ms: float  # 该条消息时间戳（毫秒）。
    stage: int  # Pipeline 阶段编号。
    phase: str  # 阶段名称（例如 REGISTER / TRACK）。
    det_count: int  # 检测框数量。
    depth_valid_ratio: float  # 深度有效像素占比（0~1）。
    fps: float  # 实时帧率估计值。
    has_pose: bool  # 是否存在有效位姿。
    pose_matrix: list[list[float]] | None  # 4x4 位姿矩阵（嵌套数组表示）。
    yolo_ms: float  # YOLO 阶段耗时（毫秒）。
    depth_ms: float  # 深度估计阶段耗时（毫秒）。
    cutie_ms: float  # Cutie 跟踪阶段耗时（毫秒）。
    pose_ms: float  # Pose 求解阶段耗时（毫秒）。

    @classmethod
    def from_runtime(
        cls,
        *,
        timestamp_ms: float,
        stage: int,
        phase: str,
        det_count: int,
        depth_valid_ratio: float,
        fps: float,
        timing_ms: dict[str, Any] | None,
        pose_4x4: np.ndarray | None,
    ) -> PoseMsg:
        """从运行时输入构建 Pose 消息对象。"""
        timing = timing_ms or {}
        pose_matrix: list[list[float]] | None = None
        if pose_4x4 is not None:
            pose_matrix = np.asarray(pose_4x4, dtype=np.float64).reshape(4, 4).tolist()

        return cls(
            timestamp_ms=float(timestamp_ms),
            stage=int(stage),
            phase=str(phase),
            det_count=int(det_count),
            depth_valid_ratio=float(depth_valid_ratio),
            fps=float(fps),
            has_pose=pose_matrix is not None,
            pose_matrix=pose_matrix,
            yolo_ms=float(timing.get("yolo", 0.0)),
            depth_ms=float(timing.get("depth", 0.0)),
            cutie_ms=float(timing.get("cutie", 0.0)),
            pose_ms=float(timing.get("pose", 0.0)),
        )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PoseMsg | None:
        """从字典反序列化为 Pose 消息对象。"""
        try:
            pose_matrix_raw = value.get("pose_matrix")
            pose_matrix: list[list[float]] | None = None
            if pose_matrix_raw is not None:
                pose_matrix = [
                    [float(item) for item in row] for row in pose_matrix_raw  # type: ignore[arg-type]
                ]
            else:
                flat = value.get("pose_matrix_flat")
                if flat is not None:
                    flat_values = [float(item) for item in flat]  # type: ignore[arg-type]
                    if len(flat_values) == 16:
                        pose_matrix = [
                            flat_values[0:4],
                            flat_values[4:8],
                            flat_values[8:12],
                            flat_values[12:16],
                        ]

            timing = value.get("timing_ms")
            if isinstance(timing, dict):
                yolo_ms = float(timing.get("yolo", value.get("yolo_ms", 0.0)))
                depth_ms = float(timing.get("depth", value.get("depth_ms", 0.0)))
                cutie_ms = float(timing.get("cutie", value.get("cutie_ms", 0.0)))
                pose_ms = float(timing.get("pose", value.get("pose_ms", 0.0)))
            else:
                yolo_ms = float(value.get("yolo_ms", 0.0))
                depth_ms = float(value.get("depth_ms", 0.0))
                cutie_ms = float(value.get("cutie_ms", 0.0))
                pose_ms = float(value.get("pose_ms", 0.0))

            return cls(
                timestamp_ms=float(value.get("timestamp_ms", 0.0)),
                stage=int(value.get("stage", 0)),
                phase=str(value.get("phase", "")),
                det_count=int(value.get("det_count", 0)),
                depth_valid_ratio=float(value.get("depth_valid_ratio", 0.0)),
                fps=float(value.get("fps", 0.0)),
                has_pose=bool(value.get("has_pose", False)),
                pose_matrix=pose_matrix,
                yolo_ms=yolo_ms,
                depth_ms=depth_ms,
                cutie_ms=cutie_ms,
                pose_ms=pose_ms,
            )
        except (TypeError, ValueError):
            return None

    @classmethod
    def from_json_bytes(cls, payload: bytes) -> PoseMsg | None:
        """从 UTF-8 JSON 字节反序列化为 Pose 消息对象。"""
        try:
            text = payload.decode("utf-8")
            data = json.loads(text)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None

        if not isinstance(data, dict):
            return None

        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        """转换为字典，供 JSON 编码或调试输出。"""
        pose_matrix_flat: list[float] | None = None
        if self.pose_matrix is not None:
            pose_matrix_flat = [float(item) for row in self.pose_matrix for item in row]

        return {
            "timestamp_ms": float(self.timestamp_ms),
            "stage": int(self.stage),
            "phase": str(self.phase),
            "det_count": int(self.det_count),
            "depth_valid_ratio": float(self.depth_valid_ratio),
            "fps": float(self.fps),
            "has_pose": bool(self.has_pose),
            "pose_matrix": self.pose_matrix,
            "pose_matrix_flat": pose_matrix_flat,
            "yolo_ms": float(self.yolo_ms),
            "depth_ms": float(self.depth_ms),
            "cutie_ms": float(self.cutie_ms),
            "pose_ms": float(self.pose_ms),
            "timing_ms": {
                "yolo": float(self.yolo_ms),
                "depth": float(self.depth_ms),
                "cutie": float(self.cutie_ms),
                "pose": float(self.pose_ms),
            },
        }

    def to_json_bytes(self) -> bytes:
        """编码为 UTF-8 JSON 字节。"""
        return json.dumps(self.to_dict(), ensure_ascii=False).encode("utf-8")
