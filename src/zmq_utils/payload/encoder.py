# pyright: reportIncompatibleMethodOverride=false
from __future__ import annotations

import json

import cv2
import numpy as np
from numpy.typing import NDArray


"""
业务编码层：把业务对象编码为 list[bytes] payload。

通信分层约定：
- 本文件只负责“对象 -> bytes[]”的业务协议编码。
- 实际发送由 zmq_utils.communicate.sender.PayloadSender 负责。

当前协议：
1) RGBD: [color_jpg, depth_png]
2) Tracking: [phase_byte, color_jpg, pose_json]
"""


class RGBDEncoder:
    """编码 (color, depth) 为 [color_jpg, depth_png]。"""

    def encode(
        self,
        color: NDArray[np.uint8],
        depth: NDArray[np.uint16],
        quality: int = 80,
        *args: object,
        **kwargs: object,
    ) -> list[bytes] | None:
        """编码 RGBD 数据。

        参数：
        - color: BGR 彩色图（uint8）。
        - depth: 深度图（uint16），保留精度。
        - quality: JPEG 质量，仅作用于 color。

        返回：
        - [color_jpg, depth_png] 或 None（编码失败）。
        """
        success, color_buf = cv2.imencode(
            ".jpg", color, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        )
        if not success:
            return None

        success, depth_buf = cv2.imencode(".png", depth)
        if not success:
            return None

        return [color_buf.tobytes(), depth_buf.tobytes()]


class TrackingEncoder: 
    """编码追踪结果为 [phase_byte, color_jpg, pose_json]。"""

    def encode(
        self,
        phase: int,
        color: NDArray[np.uint8],
        pose_matrix: NDArray[np.float64] | None = None,
        quality: int = 80,
        *args: object,
        **kwargs: object,
    ) -> list[bytes] | None:
        """编码 Tracking 数据。

        phase 语义：
        - 0: detecting
        - 1: tracking

        pose_matrix 为 None 时，pose_json 发送空字符串，表示当前无有效位姿。
        """
        success, color_buf = cv2.imencode(
            ".jpg", color, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        )
        if not success:
            return None

        pose_json = (
            json.dumps({"matrix": pose_matrix.tolist()})
            if pose_matrix is not None
            else ""
        )

        return [bytes([phase]), color_buf.tobytes(), pose_json.encode("utf-8")]
