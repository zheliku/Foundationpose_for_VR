"""
RealSense 左右图 -> Fast-FoundationStereo 深度 -> FoundationPose 本地验证脚本。

用途：
- 评估 FFS 深度输入下的 pose 稳定性与实时性。
- 与 `realsense_native_depth_pose_test.py` 做 A/B，判断 pose 质量劣化是否来自 FFS 深度。

运行示例：
    pixi run python src/realsense_stereo_ffs_pose_test.py
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import logging
import sys
import time
from pathlib import Path
from typing import Any, Literal, cast

import cv2
import numpy as np
import torch

try:
    import pyrealsense2 as rs
except ImportError as exc:
    raise SystemExit("未找到 pyrealsense2，请先安装 RealSense Python SDK。") from exc

rs_any = cast(Any, rs)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
FFS_DIR = PROJECT_DIR / "Fast-FoundationStereo"

if str(FFS_DIR) not in sys.path:
    sys.path.insert(0, str(FFS_DIR))


def _load_module_from_path(module_name: str, module_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ffs_utils = _load_module_from_path("ffs_utils_local", FFS_DIR / "Utils.py")
AMP_DTYPE: torch.dtype = ffs_utils.AMP_DTYPE
InputPadder = importlib.import_module("core.utils.utils").InputPadder


def generate_cube_symmetry_tfs() -> np.ndarray:
    import itertools

    mats: list[np.ndarray] = []
    basis = np.eye(3, dtype=np.float64)
    for perm in itertools.permutations([0, 1, 2]):
        permuted = basis[:, perm]
        for signs in itertools.product([-1.0, 1.0], repeat=3):
            r = permuted @ np.diag(signs)
            if np.linalg.det(r) > 0.9:
                if not any(np.allclose(m, r, atol=1e-6) for m in mats):
                    mats.append(r)
    tfs = []
    for r in mats:
        tf = np.eye(4, dtype=np.float64)
        tf[:3, :3] = r
        tfs.append(tf)
    return np.stack(tfs, axis=0)


def rotmat_to_quat_wxyz(rot: np.ndarray) -> np.ndarray:
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
    return q / max(np.linalg.norm(q), 1e-12)


def quat_wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
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


def slerp_quat(qa: np.ndarray, qb: np.ndarray, t: float) -> np.ndarray:
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


class PoseStabilizer:
    def __init__(self, translation_alpha: float, rotation_alpha: float) -> None:
        self.translation_alpha = float(np.clip(translation_alpha, 0.0, 1.0))
        self.rotation_alpha = float(np.clip(rotation_alpha, 0.0, 1.0))
        self.prev_pose: np.ndarray | None = None

    def reset(self) -> None:
        self.prev_pose = None

    def stabilize(self, pose: np.ndarray) -> np.ndarray:
        if self.prev_pose is None:
            self.prev_pose = pose.copy()
            return pose

        out = np.eye(4, dtype=np.float64)
        out[:3, 3] = (
            self.translation_alpha * pose[:3, 3]
            + (1.0 - self.translation_alpha) * self.prev_pose[:3, 3]
        )
        prev_q = rotmat_to_quat_wxyz(self.prev_pose[:3, :3])
        cur_q = rotmat_to_quat_wxyz(pose[:3, :3])
        out_q = slerp_quat(prev_q, cur_q, self.rotation_alpha)
        out[:3, :3] = quat_wxyz_to_rotmat(out_q)
        self.prev_pose = out.copy()
        return out


class DepthPostFilter:
    def __init__(
        self,
        temporal_alpha: float,
        adaptive_temporal: bool,
        motion_threshold: float,
        fast_motion_alpha: float,
        blend_max_rel_change: float,
    ) -> None:
        self.temporal_alpha = float(np.clip(temporal_alpha, 0.0, 1.0))
        self.adaptive_temporal = bool(adaptive_temporal)
        self.motion_threshold = float(max(motion_threshold, 0.0))
        self.fast_motion_alpha = float(np.clip(fast_motion_alpha, 0.0, 1.0))
        self.blend_max_rel_change = float(max(blend_max_rel_change, 0.0))
        self.prev_depth: np.ndarray | None = None
        self.prev_luma: np.ndarray | None = None
        self.last_motion_score = 0.0
        self.last_used_alpha = self.temporal_alpha

    def apply(
        self,
        depth: np.ndarray,
        min_depth: float,
        max_depth: float,
        reference_bgr: np.ndarray | None = None,
    ) -> np.ndarray:
        out = np.asarray(depth, dtype=np.float32)
        out[(out < min_depth) | (out > max_depth) | (~np.isfinite(out))] = 0

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

        if (
            used_alpha > 0
            and self.prev_depth is not None
            and self.prev_depth.shape == out.shape
        ):
            valid = out > 0
            blend_mask = valid.copy()
            if self.blend_max_rel_change > 0.0:
                denom = np.maximum(self.prev_depth, 1e-3)
                rel_change = np.abs(out - self.prev_depth) / denom
                blend_mask &= rel_change <= self.blend_max_rel_change

            out_blend = out.copy()
            out_blend[blend_mask] = (
                used_alpha * out[blend_mask]
                + (1 - used_alpha) * self.prev_depth[blend_mask]
            )
            out = out_blend

        if reference_bgr is not None:
            self.prev_luma = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2GRAY)

        self.last_used_alpha = used_alpha
        self.prev_depth = out.copy()
        return out


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

    def warmup(self, width: int, height: int, rounds: int) -> None:
        if rounds <= 0:
            return
        dummy_left = np.zeros((height, width, 3), dtype=np.uint8)
        dummy_right = np.zeros((height, width, 3), dtype=np.uint8)
        for _ in range(rounds):
            self.predict_depth(
                dummy_left, dummy_right, fx=1.0, baseline_m=1.0, input_mode="gray"
            )

    def predict_depth(
        self,
        left_bgr: np.ndarray,
        right_bgr: np.ndarray,
        fx: float,
        baseline_m: float,
        input_mode: Literal["rgb", "gray"] = "gray",
    ) -> tuple[np.ndarray, dict[str, float]]:
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
        depth = np.asarray(depth, dtype=np.float32)
        depth[~np.isfinite(depth)] = 0
        t3 = time.perf_counter()

        return depth, {
            "prep_ms": (t1 - t0) * 1000.0,
            "forward_ms": (t2 - t1) * 1000.0,
            "post_ms": (t3 - t2) * 1000.0,
        }


def colorize_depth(depth: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
    norm = ((depth - min_depth) / max(max_depth - min_depth, 1e-6)).clip(0.0, 1.0)
    vis = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    invalid = (depth <= min_depth) | (depth >= max_depth) | (~np.isfinite(depth))
    if invalid.any():
        vis[invalid] = 0
    return vis


def draw_text(img: np.ndarray, text: str, x: int, y: int) -> None:
    cv2.putText(
        img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2, cv2.LINE_AA
    )
    cv2.putText(
        img,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RealSense Stereo+FFS -> FoundationPose 验证"
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--valid_iters", type=int, default=3)
    parser.add_argument("--max_disp", type=int, default=192)
    parser.add_argument("--stereo_scale", type=float, default=1.0)
    parser.add_argument(
        "--stereo_input_mode", type=str, default="gray", choices=["gray", "rgb"]
    )
    parser.add_argument(
        "--optimize_build_volume",
        type=str,
        default="pytorch1",
        choices=["pytorch1", "triton"],
    )
    parser.add_argument("--min_depth", type=float, default=0.1)
    parser.add_argument("--max_depth", type=float, default=3.0)
    parser.add_argument("--depth_temporal_alpha", type=float, default=0.25)
    parser.add_argument("--depth_adaptive_temporal", type=int, default=1)
    parser.add_argument("--depth_motion_threshold", type=float, default=6.0)
    parser.add_argument("--depth_fast_motion_alpha", type=float, default=0.05)
    parser.add_argument("--depth_blend_max_rel_change", type=float, default=0.35)
    parser.add_argument("--warmup_rounds", type=int, default=2)
    parser.add_argument("--show_window", type=int, default=1)
    parser.add_argument("--stats_interval", type=int, default=30)

    parser.add_argument(
        "--model_path",
        type=Path,
        default=FFS_DIR / "weights/20-30-48/model_best_bp2_serialize.pth",
    )
    parser.add_argument(
        "--mesh_path", type=Path, default=PROJECT_DIR / "data/online/cube/mesh/cube.stl"
    )
    parser.add_argument("--text_prompt", type=str, default="white cube")
    parser.add_argument("--sam3_confidence", type=float, default=0.8)
    parser.add_argument("--est_refine_iter", type=int, default=5)
    parser.add_argument("--track_refine_iter", type=int, default=2)
    parser.add_argument("--activate_2d_tracker", type=int, default=1)
    parser.add_argument(
        "--symmetry_mode", type=str, default="cube", choices=["none", "cube"]
    )
    parser.add_argument("--stabilize_pose", type=int, default=1)
    parser.add_argument("--pose_translation_alpha", type=float, default=0.35)
    parser.add_argument("--pose_rotation_alpha", type=float, default=0.25)
    parser.add_argument("--pose_log_interval", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not args.model_path.exists():
        raise FileNotFoundError(f"FFS 模型不存在: {args.model_path}")
    if not args.mesh_path.exists():
        raise FileNotFoundError(f"Mesh 不存在: {args.mesh_path}")

    estimator = FastStereoDepthEstimator(
        model_path=args.model_path,
        device=args.device,
        valid_iters=args.valid_iters,
        max_disp=args.max_disp,
        scale=args.stereo_scale,
        optimize_build_volume=args.optimize_build_volume,
    )
    logging.info("[Pipeline] Warmup FFS (%d rounds)...", args.warmup_rounds)
    estimator.warmup(args.width, args.height, args.warmup_rounds)
    logging.info("[Pipeline] FFS warmup done")

    foundationpose_path = PROJECT_DIR / "FoundationPose"
    if str(foundationpose_path) not in sys.path:
        sys.path.insert(0, str(foundationpose_path))
    sys.modules.pop("Utils", None)
    from pose_tracker_api import PoseTracker

    pipeline = rs_any.pipeline()
    config = rs_any.config()
    config.enable_stream(
        rs_any.stream.infrared, 1, args.width, args.height, rs_any.format.y8, args.fps
    )
    config.enable_stream(
        rs_any.stream.infrared, 2, args.width, args.height, rs_any.format.y8, args.fps
    )
    profile = pipeline.start(config)

    left_profile = profile.get_stream(
        rs_any.stream.infrared, 1
    ).as_video_stream_profile()
    right_profile = profile.get_stream(
        rs_any.stream.infrared, 2
    ).as_video_stream_profile()
    intr = left_profile.get_intrinsics()
    extr = left_profile.get_extrinsics_to(right_profile)
    baseline = abs(float(extr.translation[0]))
    if baseline <= 0:
        raise RuntimeError("无法获取有效 baseline")

    cam_k = np.array(
        [[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    symmetry_tfs = (
        generate_cube_symmetry_tfs() if args.symmetry_mode == "cube" else None
    )

    tracker = PoseTracker(
        mesh_path=str(args.mesh_path),
        cam_K=cam_k,
        text_prompt=args.text_prompt,
        symmetry_tfs=symmetry_tfs,
        sam3_confidence_threshold=args.sam3_confidence,
        est_refine_iter=args.est_refine_iter,
        track_refine_iter=args.track_refine_iter,
        activate_2d_tracker=bool(args.activate_2d_tracker),
    )
    pose_stabilizer = (
        PoseStabilizer(args.pose_translation_alpha, args.pose_rotation_alpha)
        if bool(args.stabilize_pose)
        else None
    )
    depth_filter = DepthPostFilter(
        temporal_alpha=args.depth_temporal_alpha,
        adaptive_temporal=bool(args.depth_adaptive_temporal),
        motion_threshold=args.depth_motion_threshold,
        fast_motion_alpha=args.depth_fast_motion_alpha,
        blend_max_rel_change=args.depth_blend_max_rel_change,
    )

    window_name = "RealSense Stereo+FFS -> FoundationPose"
    frame_count = 0
    start_time = time.perf_counter()
    last_stats_time = start_time
    capture_acc = 0.0
    infer_acc = 0.0
    pose_acc = 0.0
    overlay_timing = {"prep_ms": 0.0, "forward_ms": 0.0, "post_ms": 0.0}

    logging.info(
        "[Config] size=%dx%d@%d fx=%.3f baseline=%.6fm mode=%s",
        args.width,
        args.height,
        args.fps,
        intr.fx,
        baseline,
        args.stereo_input_mode,
    )

    try:
        while True:
            t0 = time.perf_counter()
            frames = pipeline.wait_for_frames()
            left_frame = frames.get_infrared_frame(1)
            right_frame = frames.get_infrared_frame(2)
            if not left_frame or not right_frame:
                continue
            left_gray = np.asanyarray(left_frame.get_data())
            right_gray = np.asanyarray(right_frame.get_data())
            left_bgr = cv2.cvtColor(left_gray, cv2.COLOR_GRAY2BGR)
            right_bgr = cv2.cvtColor(right_gray, cv2.COLOR_GRAY2BGR)
            capture_acc += (time.perf_counter() - t0) * 1000.0

            t1 = time.perf_counter()
            depth_m, timing = estimator.predict_depth(
                cast(np.ndarray, left_bgr),
                cast(np.ndarray, right_bgr),
                fx=float(intr.fx),
                baseline_m=baseline,
                input_mode=cast(Literal["rgb", "gray"], args.stereo_input_mode),
            )
            depth_m = depth_filter.apply(
                depth_m,
                args.min_depth,
                args.max_depth,
                reference_bgr=cast(np.ndarray, left_bgr),
            )
            overlay_timing = timing
            infer_acc += (time.perf_counter() - t1) * 1000.0

            t2 = time.perf_counter()
            result = tracker.process_frame(
                cast(np.ndarray, left_bgr), depth_m.astype(np.float64)
            )
            if (
                result.phase == PoseTracker.Phase.TRACKING
                and result.pose_matrix is not None
                and pose_stabilizer is not None
            ):
                stable_pose = pose_stabilizer.stabilize(result.pose_matrix)
                result.pose_matrix = stable_pose
                result.color = tracker._draw_visualization(
                    cast(np.ndarray, left_bgr), stable_pose
                )
            elif (
                result.phase != PoseTracker.Phase.TRACKING
                and pose_stabilizer is not None
            ):
                pose_stabilizer.reset()
            pose_acc += (time.perf_counter() - t2) * 1000.0

            frame_count += 1
            valid_ratio = float((depth_m > 0).mean())

            if (
                result.phase == PoseTracker.Phase.TRACKING
                and result.pose_matrix is not None
                and frame_count % max(args.pose_log_interval, 1) == 0
            ):
                t = result.pose_matrix[:3, 3]
                logging.info(
                    "[Pose] t=(%.4f, %.4f, %.4f)m",
                    float(t[0]),
                    float(t[1]),
                    float(t[2]),
                )

            if bool(args.show_window):
                depth_vis = colorize_depth(depth_m, args.min_depth, args.max_depth)
                top = np.hstack((left_bgr, right_bgr))
                bottom = np.hstack((depth_vis, result.color))
                canvas = np.vstack((top, bottom))
                elapsed = max(time.perf_counter() - start_time, 1e-6)
                fps = frame_count / elapsed
                phase = (
                    "TRACKING"
                    if result.phase == PoseTracker.Phase.TRACKING
                    else "DETECTING"
                )
                draw_text(
                    canvas,
                    f"Phase: {phase} | FPS: {fps:.1f} | DepthValid: {valid_ratio:.1%}",
                    12,
                    28,
                )
                draw_text(
                    canvas,
                    f"FFS prep/fw/post: {overlay_timing['prep_ms']:.1f}/{overlay_timing['forward_ms']:.1f}/{overlay_timing['post_ms']:.1f}ms",
                    12,
                    54,
                )
                draw_text(
                    canvas,
                    f"Capture: {capture_acc/max(frame_count,1):.1f}ms | Infer: {infer_acc/max(frame_count,1):.1f}ms | Pose: {pose_acc/max(frame_count,1):.1f}ms",
                    12,
                    80,
                )
                draw_text(
                    canvas,
                    f"GhostCtrl motion={depth_filter.last_motion_score:.1f} alpha={depth_filter.last_used_alpha:.2f}",
                    12,
                    106,
                )
                cv2.imshow(window_name, canvas)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

            if frame_count % max(args.stats_interval, 1) == 0:
                now = time.perf_counter()
                elapsed = now - start_time
                interval = now - last_stats_time
                total_fps = frame_count / max(elapsed, 1e-6)
                proc_fps = args.stats_interval / max(interval, 1e-6)
                phase = "TRACKING" if tracker.is_tracking else "DETECTING"
                logging.info(
                    "[Stats] frames=%d phase=%s total_fps=%.1f proc_fps=%.1f capture=%.1fms infer=%.1fms pose=%.1fms depth_valid=%.1f%%",
                    frame_count,
                    phase,
                    total_fps,
                    proc_fps,
                    capture_acc / max(args.stats_interval, 1),
                    infer_acc / max(args.stats_interval, 1),
                    pose_acc / max(args.stats_interval, 1),
                    valid_ratio * 100.0,
                )
                last_stats_time = now
                capture_acc = 0.0
                infer_acc = 0.0
                pose_acc = 0.0

    except KeyboardInterrupt:
        logging.info("\n[Pipeline] Interrupted by user")
    finally:
        elapsed = max(time.perf_counter() - start_time, 1e-6)
        logging.info(
            "[Pipeline] Exit frames=%d avg_fps=%.1f", frame_count, frame_count / elapsed
        )
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
