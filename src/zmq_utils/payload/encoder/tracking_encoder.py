from __future__ import annotations

import json

import cv2
import numpy as np
from numpy.typing import NDArray

from .base_encoder import BaseEncoder


class TrackingEncoder(BaseEncoder):
    """Encode tracking payload to [phase_byte, color_jpg, pose_json]."""

    def encode(
        self,
        phase: int,
        color: NDArray[np.uint8],
        pose_matrix: NDArray[np.float64] | None = None,
        quality: int = 80,
        *args: object,
        **kwargs: object,
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
