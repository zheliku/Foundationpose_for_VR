from __future__ import annotations

from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from .base_decoder import PayloadDecoder


class StereoDecoder(PayloadDecoder):
    """Decode [left_jpg, right_jpg] or [packed_stereo_jpg] to stereo BGR."""

    def decode(
        self, parts: list[bytes]
    ) -> tuple[NDArray[np.uint8], NDArray[np.uint8]] | None:
        if len(parts) == 1:
            packed = cv2.imdecode(np.frombuffer(parts[0], np.uint8), cv2.IMREAD_COLOR)
            if packed is None:
                return None

            _, width = packed.shape[:2]
            if width < 2:
                return None

            mid = width // 2
            left = packed[:, :mid]
            right = packed[:, mid:]
            return cast(NDArray[np.uint8], left), cast(NDArray[np.uint8], right)

        if len(parts) != 2:
            return None

        left = cv2.imdecode(np.frombuffer(parts[0], np.uint8), cv2.IMREAD_COLOR)
        right = cv2.imdecode(np.frombuffer(parts[1], np.uint8), cv2.IMREAD_COLOR)

        if left is None or right is None:
            return None

        return cast(NDArray[np.uint8], left), cast(NDArray[np.uint8], right)
