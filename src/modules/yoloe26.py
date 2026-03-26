from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
import torch
from ultralytics import YOLOE
from numpy.typing import NDArray

from modules.realsense_rgbd import RGBDFrame


@dataclass(slots=True)
class MaskResult:
    mask_u8: NDArray[np.uint8] | None
    score: float | None
    label: str | None


class Segmenter(Protocol):
    def segment(self, frame: RGBDFrame) -> MaskResult: ...


@dataclass(slots=True)
class Yoloe26Config:
    model_path: Path
    prompt: list[str]
    mobileclip2_ts_path: Path | None = None
    conf: float = 0.15
    imgsz: int = 640
    max_det: int = 1
    mask_threshold: float = 0.5
    use_half: bool = False
    min_mask_area_ratio: float = 0.001
    morphology_kernel: int = 3


class Yoloe26Segmenter:
    def __init__(self, config: Yoloe26Config) -> None:
        self.config = config
        if not config.model_path.exists():
            raise FileNotFoundError(f"YOLOE 权重不存在: {config.model_path}")

        self._ensure_mobileclip2_asset()

        self.device: str | int = 0 if torch.cuda.is_available() else "cpu"
        self.model = YOLOE(str(config.model_path))
        self.model.set_classes(config.prompt)
        self.model.fuse()

    def _ensure_mobileclip2_asset(self) -> None:
        candidate_paths: list[Path] = []
        if self.config.mobileclip2_ts_path is not None:
            candidate_paths.append(self.config.mobileclip2_ts_path)

        project_dir = Path(__file__).resolve().parents[2]
        candidate_paths.extend(
            [
                project_dir / "mobileclip2_b.ts",
                Path.cwd() / "mobileclip2_b.ts",
            ]
        )

        asset_path = next((p for p in candidate_paths if p.exists()), None)
        if asset_path is None:
            return

        from ultralytics.utils import SETTINGS

        weights_dir = Path(SETTINGS["weights_dir"])
        weights_dir.mkdir(parents=True, exist_ok=True)
        target = weights_dir / "mobileclip2_b.ts"
        if not target.exists():
            shutil.copy2(asset_path, target)

    def segment(self, frame: RGBDFrame) -> MaskResult:
        target_hw = (int(frame.color_bgr.shape[0]), int(frame.color_bgr.shape[1]))
        result = self.model.predict(
            source=frame.color_bgr,
            conf=self.config.conf,
            imgsz=self.config.imgsz,
            max_det=self.config.max_det,
            device=self.device,
            half=self.config.use_half,
            save=False,
            verbose=False,
        )[0]

        if (
            result.masks is None
            or result.masks.data is None
            or len(result.masks.data) == 0
        ):
            return MaskResult(mask_u8=None, score=None, label=None)

        best_idx = self._select_best_instance_index(result)
        aligned_mask = self._build_aligned_mask(result, best_idx, target_hw)
        refined_mask = self._refine_mask(aligned_mask, target_hw)
        if refined_mask is None:
            return MaskResult(mask_u8=None, score=None, label=None)

        score: float | None = None
        label: str | None = None
        if (
            result.boxes is not None
            and result.boxes.conf is not None
            and len(result.boxes.conf) > 0
        ):
            score = float(result.boxes.conf[best_idx].item())
            if result.boxes.cls is not None and len(result.boxes.cls) > best_idx:
                cls_idx = int(result.boxes.cls[best_idx].item())
                label = (
                    self.model.names.get(cls_idx)
                    if isinstance(self.model.names, dict)
                    else str(cls_idx)
                )

        return MaskResult(mask_u8=refined_mask, score=score, label=label)

    def _select_best_instance_index(self, result) -> int:
        if (
            result.boxes is not None
            and result.boxes.conf is not None
            and len(result.boxes.conf) > 0
        ):
            return int(torch.argmax(result.boxes.conf).item())

        masks = result.masks.data.detach().cpu().numpy()
        areas = (
            (masks >= self.config.mask_threshold)
            .reshape(masks.shape[0], -1)
            .sum(axis=1)
        )
        return int(np.argmax(areas))

    def _build_aligned_mask(
        self, result, best_idx: int, target_hw: tuple[int, int]
    ) -> np.ndarray:
        h, w = target_hw

        if (
            getattr(result.masks, "xy", None) is not None
            and len(result.masks.xy) > best_idx
        ):
            mask = np.zeros((h, w), dtype=np.uint8)
            polygon = np.round(result.masks.xy[best_idx]).astype(np.int32)
            if polygon.size >= 6:
                cv2.fillPoly(mask, [polygon], (255,))
                return mask

        mask_data = result.masks.data[best_idx].detach().cpu().numpy()
        binary = (mask_data >= self.config.mask_threshold).astype(np.uint8) * 255
        if binary.shape != (h, w):
            binary = cv2.resize(binary, (w, h), interpolation=cv2.INTER_NEAREST)
        return binary

    def _refine_mask(
        self, mask_u8: np.ndarray, target_hw: tuple[int, int]
    ) -> np.ndarray | None:
        h, w = target_hw
        if mask_u8.shape != (h, w):
            mask_u8 = cv2.resize(mask_u8, (w, h), interpolation=cv2.INTER_NEAREST)

        binary = (mask_u8 > 0).astype(np.uint8)
        kernel_size = max(1, int(self.config.morphology_kernel))
        if kernel_size > 1:
            kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        if num_labels <= 1:
            return None

        component_areas = stats[1:, cv2.CC_STAT_AREA]
        best_label = int(np.argmax(component_areas)) + 1
        largest = (labels == best_label).astype(np.uint8)

        area_ratio = float(largest.sum()) / float(h * w)
        if area_ratio < float(self.config.min_mask_area_ratio):
            return None

        return largest.astype(np.uint8) * 255
