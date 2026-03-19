# pyright: reportIncompatibleMethodOverride=false
from __future__ import annotations

import json

import cv2
import numpy as np
from numpy.typing import NDArray


class RGBDPayloadEncoder:
    """编码 (color, depth) 为 [color_jpg, depth_png]"""

    def encode_payload(
        self,
        color: NDArray[np.uint8],
        depth: NDArray[np.uint16],
        quality: int = 80,
    ) -> list[bytes] | None:
        success, color_buf = cv2.imencode(
            ".jpg", color, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        )
        if not success:
            return None

        success, depth_buf = cv2.imencode(".png", depth)
        if not success:
            return None

        return [color_buf.tobytes(), depth_buf.tobytes()]


class TrackingPayloadEncoder:
    """编码追踪结果为 [phase_byte, color_jpg, pose_json]"""

    def encode_payload(
        self,
        phase: int,
        color: NDArray[np.uint8],
        pose_matrix: NDArray[np.float64] | None = None,
        quality: int = 80,
    ) -> list[bytes] | None:
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
