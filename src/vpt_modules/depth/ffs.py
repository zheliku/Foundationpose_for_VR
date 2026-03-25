from __future__ import annotations

import importlib
import importlib.util
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import torch

from vpt_modules.types import DepthResult


def _load_module_from_path(module_name: str, module_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass(slots=True)
class FFSConfig:
    repo_dir: Path
    model_path: Path
    device: str = "cuda"
    valid_iters: int = 8
    max_disp: int = 192
    scale: float = 1.0
    optimize_build_volume: str = "normal"
    input_mode: Literal["rgb", "gray"] = "gray"


class FastFoundationStereoDepth:
    def __init__(self, config: FFSConfig) -> None:
        self.config = config
        if not config.repo_dir.exists():
            raise FileNotFoundError(
                f"Fast-FoundationStereo 目录不存在: {config.repo_dir}"
            )
        if not config.model_path.exists():
            raise FileNotFoundError(f"FFS 权重不存在: {config.model_path}")

        if str(config.repo_dir) not in sys.path:
            sys.path.insert(0, str(config.repo_dir))

        ffs_utils = _load_module_from_path(
            "ffs_utils_local", config.repo_dir / "Utils.py"
        )
        self._amp_dtype: torch.dtype = ffs_utils.AMP_DTYPE
        self._input_padder = importlib.import_module("core.utils.utils").InputPadder

        device = config.device
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)

        old_utils = sys.modules.get("Utils")
        sys.modules["Utils"] = ffs_utils
        try:
            model = torch.load(
                str(config.model_path), map_location="cpu", weights_only=False
            )
        finally:
            if old_utils is None:
                sys.modules.pop("Utils", None)
            else:
                sys.modules["Utils"] = old_utils

        model.args.valid_iters = int(config.valid_iters)
        model.args.max_disp = int(config.max_disp)
        self.model = model.to(self.device).eval()

    def estimate(
        self, left_bgr: np.ndarray, right_bgr: np.ndarray, fx: float, baseline_m: float
    ) -> DepthResult:
        t0 = time.perf_counter()

        if self.config.input_mode == "rgb":
            left_rgb = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)
            right_rgb = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB)
        else:
            left_gray = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)
            left_rgb = np.repeat(left_gray[..., None], 3, axis=2)
            right_rgb = np.repeat(right_gray[..., None], 3, axis=2)

        if self.config.scale != 1.0:
            left_rgb = cv2.resize(
                left_rgb, dsize=None, fx=self.config.scale, fy=self.config.scale
            )
            right_rgb = cv2.resize(
                right_rgb, dsize=(left_rgb.shape[1], left_rgb.shape[0])
            )

        left_t = (
            torch.as_tensor(left_rgb, device=self.device)
            .float()[None]
            .permute(0, 3, 1, 2)
        )
        right_t = (
            torch.as_tensor(right_rgb, device=self.device)
            .float()[None]
            .permute(0, 3, 1, 2)
        )
        padder = self._input_padder(left_t.shape, divis_by=32, force_square=False)
        left_t, right_t = padder.pad(left_t, right_t)

        with torch.inference_mode():
            with torch.cuda.amp.autocast(
                enabled=(self.device.type == "cuda"), dtype=self._amp_dtype
            ):
                disp = self.model.forward(
                    left_t,
                    right_t,
                    iters=self.config.valid_iters,
                    test_mode=True,
                    optimize_build_volume=self.config.optimize_build_volume,
                )

        disp_np = (
            padder.unpad(disp.float()).squeeze(0).squeeze(0).detach().cpu().numpy()
        )
        disp_np = np.clip(disp_np, 1e-6, None)
        fx_scaled = float(fx) * float(self.config.scale)
        depth = (fx_scaled * float(baseline_m)) / disp_np
        if self.config.scale != 1.0:
            depth = cv2.resize(depth, dsize=(left_bgr.shape[1], left_bgr.shape[0]))

        depth = depth.astype(np.float64)
        valid_ratio = float((depth > 0).mean())
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return DepthResult(
            depth_m=depth,
            valid_ratio=valid_ratio,
            meta={"elapsed_ms": elapsed_ms},
        )
