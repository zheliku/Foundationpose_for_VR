from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image as PILImage

from modules.realsense_rgbd import RGBDFrame
from modules.yoloe26 import MaskResult


@dataclass(slots=True)
class Sam3Config:
    project_dir: Path
    confidence_threshold: float = 0.8


class Sam3Segmenter:
    def __init__(self, config: Sam3Config) -> None:
        self.config = config
        sam3_path = config.project_dir / "sam3"
        if str(config.project_dir) not in sys.path:
            sys.path.append(str(config.project_dir))
        if str(sam3_path) not in sys.path:
            sys.path.append(str(sam3_path))

        import importlib

        sam3_image_processor = importlib.import_module(
            "sam3.model.sam3_image_processor"
        )
        sam3_model_builder = importlib.import_module("sam3.model_builder")

        Sam3Processor = sam3_image_processor.Sam3Processor
        build_sam3_image_model = sam3_model_builder.build_sam3_image_model

        checkpoint_path = str(sam3_path / "assets/sam3_ckpt/sam3.pt")
        model = build_sam3_image_model(
            checkpoint_path=checkpoint_path,
            load_from_HF=False,
        )
        self.processor: Any = Sam3Processor(
            model,
            confidence_threshold=config.confidence_threshold,
        )

    def segment(self, frame: RGBDFrame, prompt: str) -> MaskResult:
        color_rgb = cv2.cvtColor(frame.color_bgr, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(color_rgb)
        inference_state = self.processor.set_image(pil_image)
        output = self.processor.set_text_prompt(state=inference_state, prompt=prompt)

        masks, scores = output["masks"], output["scores"]
        if len(masks) == 0:
            return MaskResult(mask_u8=None, score=None, label=prompt)

        best_idx = int(scores.argmax().item())
        best_mask = masks[best_idx]
        best_score = float(scores[best_idx].item())
        if best_score < float(self.config.confidence_threshold):
            return MaskResult(mask_u8=None, score=best_score, label=prompt)

        if isinstance(best_mask, torch.Tensor):
            mask_np = best_mask.detach().cpu().numpy()
        else:
            mask_np = np.asarray(best_mask)

        if mask_np.ndim == 3:
            mask_np = mask_np.squeeze(0)
        mask_u8 = ((mask_np > 0.5).astype(np.uint8) * 255).astype(np.uint8)
        return MaskResult(mask_u8=mask_u8, score=best_score, label=prompt)
