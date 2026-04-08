from __future__ import annotations

import json
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from .base_decoder import PayloadDecoder


class TrackingDecoder(PayloadDecoder):
    """Decode [phase_byte, color_jpg, pose_json]."""

    def decode(
        self, parts: list[bytes]
    ) -> tuple[int, NDArray[np.uint8], NDArray[np.float64] | None] | None:
        if len(parts) != 3:
            return None

        phase = int(parts[0][0]) if len(parts[0]) > 0 else 0
        color = cv2.imdecode(np.frombuffer(parts[1], np.uint8), cv2.IMREAD_COLOR)
        if color is None:
            return None

        pose_json = parts[2].decode("utf-8") if len(parts[2]) > 0 else ""
        pose_matrix: NDArray[np.float64] | None = None
        if pose_json:
            try:
                pose_matrix = np.array(
                    json.loads(pose_json)["matrix"], dtype=np.float64
                )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                return None

        return phase, cast(NDArray[np.uint8], color), pose_matrix
