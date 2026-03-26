"""
Quest 双目 -> Fast-FoundationStereo -> FoundationPose 测试脚本。

目标：
- 复用 Unity 已有双目传输协议（DualJpeg / PackedSingleJpeg）。
- 实时估计深度并输入 PoseTracker，输出 cube 的 6D 位姿。
- 作为联调脚本，不包含服务器转发逻辑。

运行示例：
    pixi run python src/quest_stereo_pose_test.py
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import cv2
import numpy as np
import torch
from numpy.typing import NDArray

from zmq_utils import PayloadReceiver, StereoJpegDecoder

if TYPE_CHECKING:
    from pose_tracker_api import PoseTracker


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
FFS_DIR = PROJECT_DIR / "Fast-FoundationStereo"

if str(FFS_DIR) not in sys.path:
    sys.path.insert(0, str(FFS_DIR))

AMP_DTYPE: torch.dtype
InputPadder: Any


def _load_module_from_path(module_name: str, module_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


try:
    ffs_utils = _load_module_from_path("ffs_utils", FFS_DIR / "Utils.py")
    AMP_DTYPE = ffs_utils.AMP_DTYPE
    InputPadder = importlib.import_module("core.utils.utils").InputPadder
except (ModuleNotFoundError, AttributeError, RuntimeError) as exc:
    raise RuntimeError(
        "无法导入 Fast-FoundationStereo 模块，请确认目录和依赖完整。"
    ) from exc


@dataclass
class StereoCalibration:
    left_fx: float
    left_fy: float
    left_cx: float
    left_cy: float
    right_fx: float
    right_fy: float
    right_cx: float
    right_cy: float
    baseline_m: float
    calib_width: int
    calib_height: int
    left_rotation_xyzw: NDArray[np.float64]
    right_rotation_xyzw: NDArray[np.float64]
    left_translation_xyz: NDArray[np.float64]
    right_translation_xyz: NDArray[np.float64]

    def _compute_center_crop_mapping(
        self, width: int, height: int
    ) -> tuple[float, float, float, float]:
        src_w = float(max(self.calib_width, 1))
        src_h = float(max(self.calib_height, 1))
        dst_w = float(max(width, 1))
        dst_h = float(max(height, 1))

        src_aspect = src_w / src_h
        dst_aspect = dst_w / dst_h

        crop_x = 0.0
        crop_y = 0.0
        crop_w = src_w
        crop_h = src_h

        if abs(src_aspect - dst_aspect) > 1e-6:
            if src_aspect > dst_aspect:
                crop_w = src_h * dst_aspect
                crop_x = (src_w - crop_w) * 0.5
            else:
                crop_h = src_w / dst_aspect
                crop_y = (src_h - crop_h) * 0.5

        sx = dst_w / max(crop_w, 1e-6)
        sy = dst_h / max(crop_h, 1e-6)
        return crop_x, crop_y, sx, sy

    def scaled_k(
        self, width: int, height: int, assume_center_crop: bool = True
    ) -> NDArray[np.float64]:
        if assume_center_crop:
            crop_x, crop_y, sx, sy = self._compute_center_crop_mapping(width, height)
            cx = (self.left_cx - crop_x) * sx
            cy = (self.left_cy - crop_y) * sy
        else:
            sx = width / max(self.calib_width, 1)
            sy = height / max(self.calib_height, 1)
            cx = self.left_cx * sx
            cy = self.left_cy * sy
        return np.array(
            [
                [self.left_fx * sx, 0.0, cx],
                [0.0, self.left_fy * sy, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def scaled_right_k(
        self, width: int, height: int, assume_center_crop: bool = True
    ) -> NDArray[np.float64]:
        if assume_center_crop:
            crop_x, crop_y, sx, sy = self._compute_center_crop_mapping(width, height)
            cx = (self.right_cx - crop_x) * sx
            cy = (self.right_cy - crop_y) * sy
        else:
            sx = width / max(self.calib_width, 1)
            sy = height / max(self.calib_height, 1)
            cx = self.right_cx * sx
            cy = self.right_cy * sy
        return np.array(
            [
                [self.right_fx * sx, 0.0, cx],
                [0.0, self.right_fy * sy, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )


def quat_xyzw_to_rotmat(q: NDArray[np.float64]) -> NDArray[np.float64]:
    x, y, z, w = [float(v) for v in q]
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s

    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float64,
    )


def rotmat_to_quat_wxyz(rot: NDArray[np.float64]) -> NDArray[np.float64]:
    m = rot
    trace = float(m[0, 0] + m[1, 1] + m[2, 2])
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s

    q = np.array([w, x, y, z], dtype=np.float64)
    q_norm = np.linalg.norm(q)
    if q_norm > 1e-12:
        q = q / q_norm
    return q


def quat_wxyz_to_rotmat(q: NDArray[np.float64]) -> NDArray[np.float64]:
    w, x, y, z = [float(v) for v in q]
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float64,
    )


def slerp_quat_wxyz(
    qa: NDArray[np.float64], qb: NDArray[np.float64], t: float
) -> NDArray[np.float64]:
    qa_n = qa / max(np.linalg.norm(qa), 1e-12)
    qb_n = qb / max(np.linalg.norm(qb), 1e-12)
    dot = float(np.dot(qa_n, qb_n))

    if dot < 0.0:
        qb_n = -qb_n
        dot = -dot

    if dot > 0.9995:
        q = qa_n + t * (qb_n - qa_n)
        return q / max(np.linalg.norm(q), 1e-12)

    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    theta = theta_0 * t
    q2 = qb_n - qa_n * dot
    q2 /= max(np.linalg.norm(q2), 1e-12)
    return qa_n * np.cos(theta) + q2 * np.sin(theta)


def rotation_geodesic_deg(r1: NDArray[np.float64], r2: NDArray[np.float64]) -> float:
    rel = r1.T @ r2
    cos_theta = np.clip((np.trace(rel) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def generate_cube_symmetry_tfs() -> NDArray[np.float64]:
    mats: list[NDArray[np.float64]] = []
    basis = np.eye(3, dtype=np.float64)
    import itertools

    for perm in itertools.permutations([0, 1, 2]):
        permuted = basis[:, perm]
        for signs in itertools.product([-1.0, 1.0], repeat=3):
            r = permuted @ np.diag(signs)
            if np.linalg.det(r) > 0.9:
                exists = False
                for m in mats:
                    if np.allclose(m, r, atol=1e-6):
                        exists = True
                        break
                if not exists:
                    mats.append(r)

    tfs: list[NDArray[np.float64]] = []
    for r in mats:
        tf = np.eye(4, dtype=np.float64)
        tf[:3, :3] = r
        tfs.append(tf)
    return np.stack(tfs, axis=0)


class PoseStabilizer:
    def __init__(
        self,
        translation_alpha: float,
        rotation_alpha: float,
        symmetry_tfs: NDArray[np.float64] | None,
    ) -> None:
        self.translation_alpha = float(np.clip(translation_alpha, 0.0, 1.0))
        self.rotation_alpha = float(np.clip(rotation_alpha, 0.0, 1.0))
        self.symmetry_tfs = symmetry_tfs
        self.prev_pose: NDArray[np.float64] | None = None

    def reset(self) -> None:
        self.prev_pose = None

    def _select_symmetric_candidate(
        self, pose: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        if self.prev_pose is None or self.symmetry_tfs is None:
            return pose

        prev_r = self.prev_pose[:3, :3]
        prev_t = self.prev_pose[:3, 3]
        best_pose = pose
        best_score = float("inf")
        for sym in self.symmetry_tfs:
            cand = pose @ sym
            cand_r = cand[:3, :3]
            cand_t = cand[:3, 3]
            rot_deg = rotation_geodesic_deg(prev_r, cand_r)
            trans = float(np.linalg.norm(cand_t - prev_t))
            score = rot_deg + trans * 100.0
            if score < best_score:
                best_score = score
                best_pose = cand
        return best_pose

    def stabilize(self, pose: NDArray[np.float64]) -> NDArray[np.float64]:
        pose = self._select_symmetric_candidate(pose)
        if self.prev_pose is None:
            self.prev_pose = pose.copy()
            return pose

        prev = self.prev_pose
        out = np.eye(4, dtype=np.float64)

        prev_t = prev[:3, 3]
        cur_t = pose[:3, 3]
        out[:3, 3] = (
            self.translation_alpha * cur_t + (1.0 - self.translation_alpha) * prev_t
        )

        prev_q = rotmat_to_quat_wxyz(prev[:3, :3])
        cur_q = rotmat_to_quat_wxyz(pose[:3, :3])
        out_q = slerp_quat_wxyz(prev_q, cur_q, self.rotation_alpha)
        out[:3, :3] = quat_wxyz_to_rotmat(out_q)

        self.prev_pose = out.copy()
        return out


class DepthPostFilter:
    def __init__(
        self,
        temporal_alpha: float,
        median_ksize: int,
        bilateral_d: int,
        bilateral_sigma_color: float,
        bilateral_sigma_space: float,
        keep_prev_invalid: bool,
        adaptive_temporal: bool,
        motion_threshold: float,
        fast_motion_alpha: float,
        blend_max_rel_change: float,
    ) -> None:
        self.temporal_alpha = float(np.clip(temporal_alpha, 0.0, 1.0))
        self.median_ksize = int(median_ksize)
        self.bilateral_d = int(bilateral_d)
        self.bilateral_sigma_color = float(bilateral_sigma_color)
        self.bilateral_sigma_space = float(bilateral_sigma_space)
        self.keep_prev_invalid = bool(keep_prev_invalid)
        self.adaptive_temporal = bool(adaptive_temporal)
        self.motion_threshold = float(max(motion_threshold, 0.0))
        self.fast_motion_alpha = float(np.clip(fast_motion_alpha, 0.0, 1.0))
        self.blend_max_rel_change = float(max(blend_max_rel_change, 0.0))
        self.prev_depth: NDArray[np.float32] | None = None
        self.prev_luma: NDArray[np.uint8] | None = None
        self.last_motion_score = 0.0
        self.last_used_alpha = self.temporal_alpha

    def apply(
        self,
        depth: NDArray[np.float32],
        min_depth: float,
        max_depth: float,
        reference_bgr: NDArray[np.uint8] | None = None,
    ) -> NDArray[np.float32]:
        out = np.asarray(depth, dtype=np.float32)

        if self.median_ksize >= 3 and self.median_ksize % 2 == 1:
            out = cv2.medianBlur(out, self.median_ksize)

        if self.bilateral_d > 0:
            out = cv2.bilateralFilter(
                out,
                d=self.bilateral_d,
                sigmaColor=self.bilateral_sigma_color,
                sigmaSpace=self.bilateral_sigma_space,
            )

        out[(out < min_depth) | (out > max_depth) | (~np.isfinite(out))] = 0.0

        used_alpha = self.temporal_alpha
        if (
            self.adaptive_temporal
            and reference_bgr is not None
            and self.prev_luma is not None
            and self.prev_luma.shape == reference_bgr.shape[:2]
        ):
            cur_luma = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2GRAY)
            motion_score = float(
                np.mean(
                    np.abs(
                        cur_luma.astype(np.float32) - self.prev_luma.astype(np.float32)
                    )
                )
            )
            self.last_motion_score = motion_score
            if motion_score > self.motion_threshold:
                used_alpha = min(used_alpha, self.fast_motion_alpha)

        if used_alpha > 0.0 and self.prev_depth is not None:
            if self.prev_depth.shape == out.shape:
                valid = out > 0
                blend_mask = valid.copy()
                if self.blend_max_rel_change > 0.0:
                    denom = np.maximum(self.prev_depth, 1e-3)
                    rel_change = np.abs(out - self.prev_depth) / denom
                    blend_mask &= rel_change <= self.blend_max_rel_change
                out_blend = out.copy()
                out_blend[blend_mask] = (
                    used_alpha * out[blend_mask]
                    + (1.0 - used_alpha) * self.prev_depth[blend_mask]
                )
                if self.keep_prev_invalid:
                    out_blend[~valid] = self.prev_depth[~valid]
                out = out_blend

        if reference_bgr is not None:
            self.prev_luma = cast(
                NDArray[np.uint8], cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2GRAY)
            )

        self.last_used_alpha = used_alpha
        out = np.asarray(out, dtype=np.float32)
        self.prev_depth = cast(NDArray[np.float32], out.copy())
        return cast(NDArray[np.float32], out)


class StereoRectifier:
    def __init__(
        self, calib: StereoCalibration, assume_center_crop: bool = True
    ) -> None:
        self.calib = calib
        self.assume_center_crop = bool(assume_center_crop)
        self.maps_ready = False
        self.left_map1: NDArray[np.float32] | None = None
        self.left_map2: NDArray[np.float32] | None = None
        self.right_map1: NDArray[np.float32] | None = None
        self.right_map2: NDArray[np.float32] | None = None
        self.rectified_left_k: NDArray[np.float64] | None = None

    def prepare(self, width: int, height: int) -> None:
        if self.maps_ready:
            return

        k_left = self.calib.scaled_k(
            width, height, assume_center_crop=self.assume_center_crop
        )
        k_right = self.calib.scaled_right_k(
            width, height, assume_center_crop=self.assume_center_crop
        )
        d_left = np.zeros((5, 1), dtype=np.float64)
        d_right = np.zeros((5, 1), dtype=np.float64)

        r_w_l = quat_xyzw_to_rotmat(self.calib.left_rotation_xyzw)
        r_w_r = quat_xyzw_to_rotmat(self.calib.right_rotation_xyzw)
        t_w_l = self.calib.left_translation_xyz.reshape(3, 1)
        t_w_r = self.calib.right_translation_xyz.reshape(3, 1)

        r_l_to_r = r_w_r.T @ r_w_l
        t_l_to_r = r_w_r.T @ (t_w_l - t_w_r)

        r1, r2, p1, _p2, _q, _roi1, _roi2 = cv2.stereoRectify(
            k_left,
            d_left,
            k_right,
            d_right,
            (width, height),
            r_l_to_r,
            t_l_to_r,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0.0,
        )

        left_map1, left_map2 = cv2.initUndistortRectifyMap(
            k_left,
            d_left,
            r1,
            p1,
            (width, height),
            cv2.CV_32FC1,
        )
        right_map1, right_map2 = cv2.initUndistortRectifyMap(
            k_right,
            d_right,
            r2,
            _p2,
            (width, height),
            cv2.CV_32FC1,
        )

        self.left_map1 = cast(NDArray[np.float32], left_map1)
        self.left_map2 = cast(NDArray[np.float32], left_map2)
        self.right_map1 = cast(NDArray[np.float32], right_map1)
        self.right_map2 = cast(NDArray[np.float32], right_map2)

        self.rectified_left_k = p1[:3, :3].astype(np.float64)
        self.maps_ready = True

    def rectify(
        self, left_bgr: NDArray[np.uint8], right_bgr: NDArray[np.uint8]
    ) -> tuple[NDArray[np.uint8], NDArray[np.uint8], NDArray[np.float64]]:
        h, w = left_bgr.shape[:2]
        if not self.maps_ready:
            self.prepare(width=w, height=h)

        if (
            self.left_map1 is None
            or self.left_map2 is None
            or self.right_map1 is None
            or self.right_map2 is None
            or self.rectified_left_k is None
        ):
            raise RuntimeError("立体校正映射未初始化")

        left_rect = cv2.remap(
            left_bgr,
            self.left_map1,
            self.left_map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        right_rect = cv2.remap(
            right_bgr,
            self.right_map1,
            self.right_map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )

        return (
            np.asarray(left_rect, dtype=np.uint8),
            np.asarray(right_rect, dtype=np.uint8),
            self.rectified_left_k,
        )


class FastStereoDepthEstimator:
    def __init__(
        self,
        model_path: Path,
        device: str,
        valid_iters: int,
        max_disp: int,
        scale: float,
        optimize_build_volume: str,
    ) -> None:
        if device == "cuda" and not torch.cuda.is_available():
            logging.warning("CUDA 不可用，自动切换到 CPU。")
            device = "cpu"

        self.device = torch.device(device)
        self.scale = float(scale)
        self.valid_iters = int(valid_iters)
        self.optimize_build_volume = optimize_build_volume

        old_utils_module = sys.modules.get("Utils")
        sys.modules["Utils"] = ffs_utils
        try:
            self.model = torch.load(
                str(model_path), map_location="cpu", weights_only=False
            )
        finally:
            if old_utils_module is None:
                sys.modules.pop("Utils", None)
            else:
                sys.modules["Utils"] = old_utils_module

        self.model.args.valid_iters = self.valid_iters
        self.model.args.max_disp = int(max_disp)
        self.model = self.model.to(self.device).eval()

        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

    def warmup(
        self,
        width: int,
        height: int,
        rounds: int = 2,
        input_mode: Literal["rgb", "gray"] = "rgb",
    ) -> None:
        if rounds <= 0:
            return
        dummy_left = np.zeros((height, width, 3), dtype=np.uint8)
        dummy_right = np.zeros((height, width, 3), dtype=np.uint8)
        for _ in range(rounds):
            self.predict_depth(
                dummy_left,
                dummy_right,
                fx=1.0,
                baseline_m=1.0,
                input_mode=input_mode,
            )

    def predict_depth(
        self,
        left_bgr: NDArray[np.uint8],
        right_bgr: NDArray[np.uint8],
        fx: float,
        baseline_m: float,
        input_mode: Literal["rgb", "gray"] = "rgb",
    ) -> tuple[NDArray[np.float32], dict[str, float]]:
        t0 = time.perf_counter()
        if input_mode == "rgb":
            left_rgb = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)
            right_rgb = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB)
        else:
            left_gray = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)
            left_rgb = np.repeat(left_gray[..., None], 3, axis=2)
            right_rgb = np.repeat(right_gray[..., None], 3, axis=2)

        if self.scale != 1.0:
            left_rgb = cv2.resize(
                left_rgb,
                dsize=None,
                fx=self.scale,
                fy=self.scale,
                interpolation=cv2.INTER_LINEAR,
            )
            right_rgb = cv2.resize(
                right_rgb,
                dsize=(left_rgb.shape[1], left_rgb.shape[0]),
                interpolation=cv2.INTER_LINEAR,
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

        padder = InputPadder(left_t.shape, divis_by=32, force_square=False)
        left_t, right_t = padder.pad(left_t, right_t)
        t1 = time.perf_counter()

        with torch.inference_mode():
            with torch.cuda.amp.autocast(
                enabled=(self.device.type == "cuda"), dtype=AMP_DTYPE
            ):
                disp = self.model.forward(
                    left_t,
                    right_t,
                    iters=self.valid_iters,
                    test_mode=True,
                    optimize_build_volume=self.optimize_build_volume,
                )

        t2 = time.perf_counter()
        disp_np = (
            padder.unpad(disp.float()).squeeze(0).squeeze(0).detach().cpu().numpy()
        )
        disp_np = np.clip(disp_np, 1e-6, None)

        fx_scaled = fx * self.scale
        depth = (fx_scaled * baseline_m) / disp_np
        if self.scale != 1.0:
            depth = cv2.resize(
                depth,
                dsize=(left_bgr.shape[1], left_bgr.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        depth = depth.astype(np.float32)
        depth[~np.isfinite(depth)] = 0

        t3 = time.perf_counter()
        timing = {
            "prep_ms": (t1 - t0) * 1000.0,
            "forward_ms": (t2 - t1) * 1000.0,
            "post_ms": (t3 - t2) * 1000.0,
        }
        return depth, timing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quest 双目图像接收 + Fast-FoundationStereo 深度估计 + FoundationPose 测试"
    )
    parser.add_argument("--listen_port", type=int, default=5557)
    parser.add_argument("--hwm", type=int, default=1)
    parser.add_argument(
        "--model_path",
        type=Path,
        default=FFS_DIR / "weights/20-30-48/model_best_bp2_serialize.pth",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--valid_iters", type=int, default=3)
    parser.add_argument("--max_disp", type=int, default=192)
    parser.add_argument("--stereo_scale", type=float, default=1.0)
    parser.add_argument("--process_width", type=int, default=640)
    parser.add_argument("--process_height", type=int, default=480)
    parser.add_argument(
        "--stereo_input_mode", type=str, default="gray", choices=["rgb", "gray"]
    )
    parser.add_argument(
        "--optimize_build_volume",
        type=str,
        default="pytorch1",
        choices=["pytorch1", "triton"],
    )
    parser.add_argument("--min_depth", type=float, default=0.1)
    parser.add_argument("--max_depth", type=float, default=3.0)
    parser.add_argument("--depth_temporal_alpha", type=float, default=0.0)
    parser.add_argument("--depth_keep_prev_invalid", type=int, default=0)
    parser.add_argument("--depth_adaptive_temporal", type=int, default=1)
    parser.add_argument("--depth_motion_threshold", type=float, default=5.5)
    parser.add_argument("--depth_fast_motion_alpha", type=float, default=0.03)
    parser.add_argument("--depth_blend_max_rel_change", type=float, default=0.25)
    parser.add_argument("--depth_median_ksize", type=int, default=0)
    parser.add_argument("--depth_bilateral_d", type=int, default=0)
    parser.add_argument("--depth_bilateral_sigma_color", type=float, default=0.05)
    parser.add_argument("--depth_bilateral_sigma_space", type=float, default=3.0)
    parser.add_argument("--stabilize_pose", type=int, default=1)
    parser.add_argument("--pose_translation_alpha", type=float, default=0.35)
    parser.add_argument("--pose_rotation_alpha", type=float, default=0.25)
    parser.add_argument(
        "--symmetry_mode", type=str, default="cube", choices=["none", "cube"]
    )
    parser.add_argument("--enable_rectify", type=int, default=0)
    parser.add_argument("--show_window", type=int, default=1)
    parser.add_argument("--stats_interval", type=int, default=30)
    parser.add_argument("--warmup_rounds", type=int, default=2)
    parser.add_argument("--warmup_width", type=int, default=640)
    parser.add_argument("--warmup_height", type=int, default=480)
    parser.add_argument(
        "--calib_assume_center_crop",
        type=int,
        default=1,
        help="当标定分辨率与输入分辨率宽高比不一致时，按中心裁剪后再缩放内参。1=启用(推荐)，0=仅线性缩放。",
    )
    parser.add_argument(
        "--calib_dir",
        type=Path,
        default=PROJECT_DIR / "docs/20260322_070544",
    )

    parser.add_argument(
        "--mesh_path",
        type=Path,
        default=PROJECT_DIR / "data/online/cube/mesh/cube.stl",
    )
    parser.add_argument("--text_prompt", type=str, default="white cube")
    parser.add_argument("--sam3_confidence", type=float, default=0.8)
    parser.add_argument("--est_refine_iter", type=int, default=5)
    parser.add_argument("--track_refine_iter", type=int, default=2)
    parser.add_argument("--activate_2d_tracker", type=int, default=1)
    parser.add_argument("--debug_output_dir", type=Path, default=None)
    parser.add_argument("--pose_log_interval", type=int, default=5)
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_quest_calibration(calib_dir: Path) -> StereoCalibration:
    left = _read_json(calib_dir / "left_camera_characteristics.json")
    right = _read_json(calib_dir / "right_camera_characteristics.json")

    left_intr = left["intrinsics"]
    right_intr = right["intrinsics"]
    left_t = np.array(left["pose"]["translation"], dtype=np.float64)
    right_t = np.array(right["pose"]["translation"], dtype=np.float64)
    left_q = np.array(left["pose"]["rotation"], dtype=np.float64)
    right_q = np.array(right["pose"]["rotation"], dtype=np.float64)
    baseline_m = float(np.linalg.norm(right_t - left_t))

    sensor = left.get("sensor", {})
    active = sensor.get("activeArraySize", {})
    width = int(active.get("right", 1280) - active.get("left", 0))
    height = int(active.get("bottom", 1280) - active.get("top", 0))

    return StereoCalibration(
        left_fx=float(left_intr["fx"]),
        left_fy=float(left_intr["fy"]),
        left_cx=float(left_intr["cx"]),
        left_cy=float(left_intr["cy"]),
        right_fx=float(right_intr["fx"]),
        right_fy=float(right_intr["fy"]),
        right_cx=float(right_intr["cx"]),
        right_cy=float(right_intr["cy"]),
        baseline_m=baseline_m,
        calib_width=width,
        calib_height=height,
        left_rotation_xyzw=left_q,
        right_rotation_xyzw=right_q,
        left_translation_xyz=left_t,
        right_translation_xyz=right_t,
    )


def colorize_depth(
    depth: NDArray[np.float32], min_depth: float, max_depth: float
) -> NDArray[np.uint8]:
    norm = ((depth - min_depth) / max(max_depth - min_depth, 1e-6)).clip(0.0, 1.0)
    vis = cv2.applyColorMap((norm * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
    invalid = (depth <= min_depth) | (depth >= max_depth) | (~np.isfinite(depth))
    if invalid.any():
        vis[invalid] = 0
    return cast(NDArray[np.uint8], vis)


def draw_text(img: NDArray[np.uint8], text: str, x: int, y: int) -> None:
    cv2.putText(
        img,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not args.model_path.exists():
        raise FileNotFoundError(f"Fast-FoundationStereo 模型不存在: {args.model_path}")
    if not args.mesh_path.exists():
        raise FileNotFoundError(f"Mesh 不存在: {args.mesh_path}")
    if not args.calib_dir.exists():
        raise FileNotFoundError(f"标定目录不存在: {args.calib_dir}")

    calib = load_quest_calibration(args.calib_dir)
    logging.info(
        "[Config] left_fx=%.3f left_fy=%.3f left_cx=%.3f left_cy=%.3f baseline=%.6fm calib=%dx%d",
        calib.left_fx,
        calib.left_fy,
        calib.left_cx,
        calib.left_cy,
        calib.baseline_m,
        calib.calib_width,
        calib.calib_height,
    )

    estimator = FastStereoDepthEstimator(
        model_path=args.model_path,
        device=args.device,
        valid_iters=args.valid_iters,
        max_disp=args.max_disp,
        scale=args.stereo_scale,
        optimize_build_volume=args.optimize_build_volume,
    )

    logging.info(
        "[Pipeline] Warming up depth model first (%d rounds, %dx%d)...",
        args.warmup_rounds,
        args.warmup_width,
        args.warmup_height,
    )
    estimator.warmup(
        width=args.warmup_width,
        height=args.warmup_height,
        rounds=args.warmup_rounds,
        input_mode=cast(Literal["rgb", "gray"], args.stereo_input_mode),
    )
    logging.info("[Pipeline] Depth warmup done")

    foundationpose_path = PROJECT_DIR / "FoundationPose"
    if str(foundationpose_path) not in sys.path:
        sys.path.insert(0, str(foundationpose_path))
    sys.modules.pop("Utils", None)

    from pose_tracker_api import PoseTracker

    symmetry_tfs: NDArray[np.float64] | None = None
    if args.symmetry_mode == "cube":
        symmetry_tfs = generate_cube_symmetry_tfs()

    rectifier = (
        StereoRectifier(calib, assume_center_crop=bool(args.calib_assume_center_crop))
        if bool(args.enable_rectify)
        else None
    )
    depth_filter = DepthPostFilter(
        temporal_alpha=args.depth_temporal_alpha,
        median_ksize=args.depth_median_ksize,
        bilateral_d=args.depth_bilateral_d,
        bilateral_sigma_color=args.depth_bilateral_sigma_color,
        bilateral_sigma_space=args.depth_bilateral_sigma_space,
        keep_prev_invalid=bool(args.depth_keep_prev_invalid),
        adaptive_temporal=bool(args.depth_adaptive_temporal),
        motion_threshold=args.depth_motion_threshold,
        fast_motion_alpha=args.depth_fast_motion_alpha,
        blend_max_rel_change=args.depth_blend_max_rel_change,
    )
    pose_stabilizer = (
        PoseStabilizer(
            translation_alpha=args.pose_translation_alpha,
            rotation_alpha=args.pose_rotation_alpha,
            symmetry_tfs=symmetry_tfs,
        )
        if bool(args.stabilize_pose)
        else None
    )

    receiver = PayloadReceiver(
        f"tcp://*:{args.listen_port}", hwm=args.hwm, bind=True, use_topic=False
    )
    decoder = StereoJpegDecoder()

    pose_tracker: PoseTracker | None = None
    window_name = "Quest Stereo -> Depth -> FoundationPose"

    frame_count = 0
    recv_count = 0
    start_time = time.perf_counter()
    last_stats_time = start_time
    infer_acc_ms = 0.0
    tracker_acc_ms = 0.0
    compose_acc_ms = 0.0
    drain_total = 0
    overlay_phase = "DETECTING"
    overlay_depth_timing = {"prep_ms": 0.0, "forward_ms": 0.0, "post_ms": 0.0}
    overlay_depth_valid_ratio = 0.0
    decode_mode = "Unknown"
    has_warned_packed_mode = False
    has_warned_aspect_stretch = False
    has_warned_calib_aspect_mismatch = False
    has_logged_k_mapping = False

    logging.info(
        "[Pipeline] Listening on tcp://*:%d, press 'q' or ESC to exit",
        args.listen_port,
    )

    try:
        while True:
            parts = receiver.recv_payload(timeout_ms=100)
            if parts is None:
                continue
            recv_count += 1

            drain_total += getattr(receiver, "last_drain_count", 0)
            decode_mode = "PackedSingleJpeg" if len(parts) == 1 else "DualJpeg"
            if decode_mode == "PackedSingleJpeg" and not has_warned_packed_mode:
                has_warned_packed_mode = True
                logging.warning(
                    "[Quality] 当前为 PackedSingleJpeg 模式，立体匹配精度通常低于 DualJpeg。建议在 Unity 将 packStereoIntoSingleJpeg 关闭，并提高 JPEG 质量或切 PNG。"
                )
            parsed = decoder.decode(parts)
            if parsed is None:
                continue

            left_bgr, right_bgr = parsed
            left_bgr = np.asarray(left_bgr, dtype=np.uint8)
            right_bgr = np.asarray(right_bgr, dtype=np.uint8)
            if left_bgr.shape[:2] != right_bgr.shape[:2]:
                target_h = min(left_bgr.shape[0], right_bgr.shape[0])
                target_w = min(left_bgr.shape[1], right_bgr.shape[1])
                left_bgr = cv2.resize(left_bgr, (target_w, target_h))
                right_bgr = cv2.resize(right_bgr, (target_w, target_h))

            h, w = left_bgr.shape[:2]
            if (
                not has_warned_calib_aspect_mismatch
                and calib.calib_width > 0
                and calib.calib_height > 0
                and w > 0
                and h > 0
            ):
                calib_ratio = float(calib.calib_width) / float(calib.calib_height)
                frame_ratio = float(w) / float(h)
                if abs(calib_ratio - frame_ratio) > 1e-3:
                    has_warned_calib_aspect_mismatch = True
                    logging.warning(
                        "[CalibAspect] 标定分辨率=%dx%d(%.3f), 输入帧=%dx%d(%.3f)。若上游存在裁剪而非等比缩放，按当前方式缩放 K 可能造成 3D 框形变。",
                        calib.calib_width,
                        calib.calib_height,
                        calib_ratio,
                        w,
                        h,
                        frame_ratio,
                    )
            cam_k = calib.scaled_k(
                width=w,
                height=h,
                assume_center_crop=bool(args.calib_assume_center_crop),
            )
            if not has_logged_k_mapping:
                has_logged_k_mapping = True
                map_mode = (
                    "center-crop+scale"
                    if bool(args.calib_assume_center_crop)
                    else "linear-scale-only"
                )
                logging.info(
                    "[KMap] mode=%s K=[fx=%.2f fy=%.2f cx=%.2f cy=%.2f] frame=%dx%d calib=%dx%d",
                    map_mode,
                    float(cam_k[0, 0]),
                    float(cam_k[1, 1]),
                    float(cam_k[0, 2]),
                    float(cam_k[1, 2]),
                    w,
                    h,
                    calib.calib_width,
                    calib.calib_height,
                )

            compose_t0 = time.perf_counter()
            if rectifier is not None:
                left_bgr, right_bgr, cam_k = rectifier.rectify(
                    cast(NDArray[np.uint8], left_bgr),
                    cast(NDArray[np.uint8], right_bgr),
                )

            src_h, src_w = left_bgr.shape[:2]
            target_w = max(int(args.process_width), 0)
            target_h = max(int(args.process_height), 0)
            if (
                not has_warned_aspect_stretch
                and target_w > 0
                and target_h > 0
                and src_w > 0
                and src_h > 0
            ):
                src_ratio = float(src_w) / float(src_h)
                dst_ratio = float(target_w) / float(target_h)
                if abs(src_ratio - dst_ratio) > 1e-3:
                    has_warned_aspect_stretch = True
                    logging.warning(
                        "[Aspect] 处理尺寸会改变宽高比: src=%dx%d(%.3f), target=%dx%d(%.3f)。如不希望拉伸，请将 --process_width/--process_height 设为与输入同宽高比（例如 640x480）。",
                        src_w,
                        src_h,
                        src_ratio,
                        target_w,
                        target_h,
                        dst_ratio,
                    )
            if (
                target_w > 0
                and target_h > 0
                and (src_w != target_w or src_h != target_h)
            ):
                interpolation = (
                    cv2.INTER_AREA
                    if target_w < src_w or target_h < src_h
                    else cv2.INTER_LINEAR
                )
                left_bgr = cv2.resize(
                    left_bgr, (target_w, target_h), interpolation=interpolation
                )
                right_bgr = cv2.resize(
                    right_bgr, (target_w, target_h), interpolation=interpolation
                )
                sx = target_w / max(src_w, 1)
                sy = target_h / max(src_h, 1)
                cam_k = cam_k.copy()
                cam_k[0, 0] *= sx
                cam_k[0, 2] *= sx
                cam_k[1, 1] *= sy
                cam_k[1, 2] *= sy
            compose_acc_ms += (time.perf_counter() - compose_t0) * 1000.0

            if pose_tracker is None:
                pose_tracker = PoseTracker(
                    mesh_path=str(args.mesh_path),
                    cam_K=cam_k,
                    text_prompt=args.text_prompt,
                    symmetry_tfs=symmetry_tfs,
                    sam3_confidence_threshold=args.sam3_confidence,
                    est_refine_iter=args.est_refine_iter,
                    track_refine_iter=args.track_refine_iter,
                    activate_2d_tracker=bool(args.activate_2d_tracker),
                    debug_output_dir=(
                        str(args.debug_output_dir) if args.debug_output_dir else None
                    ),
                )
                logging.info("[Pipeline] PoseTracker initialized")

            infer_t0 = time.perf_counter()
            depth_m, depth_timing = estimator.predict_depth(
                cast(NDArray[np.uint8], left_bgr),
                cast(NDArray[np.uint8], right_bgr),
                fx=cam_k[0, 0],
                baseline_m=calib.baseline_m,
                input_mode=cast(Literal["rgb", "gray"], args.stereo_input_mode),
            )
            depth_m[(depth_m < args.min_depth) | (depth_m > args.max_depth)] = 0
            depth_m = depth_filter.apply(
                depth_m,
                args.min_depth,
                args.max_depth,
                reference_bgr=cast(NDArray[np.uint8], left_bgr),
            )
            depth_valid_mask = depth_m > 0
            overlay_depth_valid_ratio = float(depth_valid_mask.mean())
            overlay_depth_timing = depth_timing
            infer_acc_ms += (time.perf_counter() - infer_t0) * 1000.0

            tracker_t0 = time.perf_counter()
            tracking_result = pose_tracker.process_frame(
                color=cast(NDArray[np.uint8], left_bgr),
                depth=depth_m.astype(np.float64),
            )

            if tracking_result.phase != PoseTracker.Phase.TRACKING and pose_stabilizer:
                pose_stabilizer.reset()

            if (
                pose_stabilizer is not None
                and tracking_result.phase == PoseTracker.Phase.TRACKING
                and tracking_result.pose_matrix is not None
            ):
                stable_pose = pose_stabilizer.stabilize(tracking_result.pose_matrix)
                tracking_result.pose_matrix = stable_pose
                tracking_result.color = pose_tracker._draw_visualization(
                    cast(NDArray[np.uint8], left_bgr), stable_pose
                )
            tracker_acc_ms += (time.perf_counter() - tracker_t0) * 1000.0

            frame_count += 1
            overlay_phase = (
                "TRACKING"
                if tracking_result.phase == PoseTracker.Phase.TRACKING
                else "DETECTING"
            )

            if (
                tracking_result.phase == PoseTracker.Phase.TRACKING
                and tracking_result.pose_matrix is not None
                and frame_count % max(args.pose_log_interval, 1) == 0
            ):
                t = tracking_result.pose_matrix[:3, 3]
                logging.info(
                    "[Pose] t=(%.4f, %.4f, %.4f)m",
                    float(t[0]),
                    float(t[1]),
                    float(t[2]),
                )

            if args.show_window:
                depth_vis = colorize_depth(depth_m, args.min_depth, args.max_depth)
                pose_vis = tracking_result.color
                top = np.hstack((left_bgr, right_bgr))
                bottom = np.hstack((depth_vis, pose_vis))
                canvas = np.asarray(np.vstack((top, bottom)), dtype=np.uint8)

                now = time.perf_counter()
                elapsed = max(now - start_time, 1e-6)
                total_fps = frame_count / elapsed
                ingress_fps = recv_count / elapsed

                draw_text(
                    canvas,
                    f"Phase: {overlay_phase} | Decode: {decode_mode} | Rectify: {bool(args.enable_rectify)}",
                    12,
                    28,
                )
                draw_text(
                    canvas,
                    f"Depth prep/forward/post: {overlay_depth_timing['prep_ms']:.1f}/{overlay_depth_timing['forward_ms']:.1f}/{overlay_depth_timing['post_ms']:.1f}ms",
                    12,
                    54,
                )
                draw_text(
                    canvas,
                    f"ProcFPS: {total_fps:.1f} | IngressFPS: {ingress_fps:.1f} | DrainDrop: {drain_total}",
                    12,
                    80,
                )
                draw_text(
                    canvas,
                    f"DepthValid: {overlay_depth_valid_ratio:.1%} | Proc: {left_bgr.shape[1]}x{left_bgr.shape[0]} | PoseStab: {bool(args.stabilize_pose)}",
                    12,
                    106,
                )
                draw_text(
                    canvas,
                    f"GhostCtrl motion={depth_filter.last_motion_score:.1f} alpha={depth_filter.last_used_alpha:.2f}",
                    12,
                    132,
                )

                cv2.imshow(window_name, canvas)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

            if frame_count % max(args.stats_interval, 1) == 0:
                now = time.perf_counter()
                elapsed = now - start_time
                interval = now - last_stats_time
                fps = frame_count / max(elapsed, 1e-6)
                proc_fps = args.stats_interval / max(interval, 1e-6)
                avg_infer = infer_acc_ms / max(args.stats_interval, 1)
                avg_tracker = tracker_acc_ms / max(args.stats_interval, 1)
                avg_compose = compose_acc_ms / max(args.stats_interval, 1)
                ingress_fps = recv_count / max(elapsed, 1e-6)
                phase = "TRACKING" if pose_tracker.is_tracking else "DETECTING"
                logging.info(
                    "[Stats] recv=%d frames=%d phase=%s mode=%s rectify=%s total_fps=%.1f ingress_fps=%.1f proc_fps=%.1f infer=%.1fms tracker=%.1fms compose=%.1fms depth_valid=%.1f%% drain=%d",
                    recv_count,
                    frame_count,
                    phase,
                    decode_mode,
                    bool(args.enable_rectify),
                    fps,
                    ingress_fps,
                    proc_fps,
                    avg_infer,
                    avg_tracker,
                    avg_compose,
                    overlay_depth_valid_ratio * 100.0,
                    drain_total,
                )
                last_stats_time = now
                infer_acc_ms = 0.0
                tracker_acc_ms = 0.0
                compose_acc_ms = 0.0

    except KeyboardInterrupt:
        logging.info("\n[Pipeline] Interrupted by user")
    finally:
        elapsed = time.perf_counter() - start_time
        fps = frame_count / max(elapsed, 1e-6)
        logging.info(
            "[Pipeline] Exit. frames=%d avg_fps=%.1f drain=%d",
            frame_count,
            fps,
            drain_total,
        )
        receiver.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
