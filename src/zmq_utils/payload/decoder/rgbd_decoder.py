from __future__ import annotations

from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from .base_decoder import PayloadDecoder
from ..message.rgbd import RGBDMsg


class RGBDDecoder(PayloadDecoder):
    """Decode single RGBD payload bytes to (color, depth)."""

    def decode(
        self, payload: bytes
    ) -> tuple[NDArray[np.uint8], NDArray[np.uint16]] | None:
        message = RGBDMsg.deserialize(payload)
        if message is None:
            return None

        color = cv2.imdecode(
            np.frombuffer(message.color_image, np.uint8), cv2.IMREAD_COLOR
        )
        depth = cv2.imdecode(
            np.frombuffer(message.depth_image, np.uint8), cv2.IMREAD_UNCHANGED
        )

        if color is None or depth is None:
            return None

        return cast(NDArray[np.uint8], color), depth.astype(np.uint16)
