from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from .base_encoder import BaseEncoder
from ..message.rgbd import RGBDMsg


class RGBDEncoder(BaseEncoder):
    """Encode (color, depth) to [color_jpg, depth_png]."""

    def encode(
        self,
        color: NDArray[np.uint8],
        depth: NDArray[np.uint16],
        quality: int = 80,
        *args: object,
        **kwargs: object,
    ) -> list[bytes] | None:
        success, color_buf = cv2.imencode(
            ".jpg", color, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        )
        if not success:
            return None

        success, depth_buf = cv2.imencode(".png", depth)
        if not success:
            return None

        payload = RGBDMsg(
            color_image=color_buf.tobytes(),
            depth_image=depth_buf.tobytes(),
        )
        return payload.to_parts()
