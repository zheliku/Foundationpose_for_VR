from __future__ import annotations

from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from .base_decoder import PayloadDecoder


class RGBDDecoder(PayloadDecoder):
    """Decode [color_jpg, depth_png] to (color, depth)."""

    def decode(
        self, parts: list[bytes]
    ) -> tuple[NDArray[np.uint8], NDArray[np.uint16]] | None:
        if len(parts) != 2:
            return None

        color = cv2.imdecode(np.frombuffer(parts[0], np.uint8), cv2.IMREAD_COLOR)
        depth = cv2.imdecode(np.frombuffer(parts[1], np.uint8), cv2.IMREAD_UNCHANGED)

        if color is None or depth is None:
            return None

        return cast(NDArray[np.uint8], color), depth.astype(np.uint16)
