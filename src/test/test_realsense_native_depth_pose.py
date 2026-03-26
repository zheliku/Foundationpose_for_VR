"""
RealSense 原生深度 -> FoundationPose 本地验证脚本。

用途：
- 作为基线链路，验证使用 RealSense 原生 depth 时的 pose 质量与稳定性。
- 与 `realsense_stereo_ffs_pose_test.py` 做 A/B 对比，定位问题是否来自 FFS 深度。

运行示例：
    pixi run python src/realsense_native_depth_pose_test.py
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError as exc:
    raise SystemExit("未找到 pyrealsense2，请先安装 RealSense Python SDK。") from exc

rs_any = cast(Any, rs)

from pose_tracker_api import PoseTracker


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent


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


class PoseStabilizer:
    def __init__(
        self,
        translation_alpha: float,
        rotation_alpha: float,
    ) -> None:
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
        prev_t = self.prev_pose[:3, 3]
        cur_t = pose[:3, 3]
        out[:3, 3] = (
            self.translation_alpha * cur_t + (1.0 - self.translation_alpha) * prev_t
        )

        prev_q = rotmat_to_quat_wxyz(self.prev_pose[:3, :3])
        cur_q = rotmat_to_quat_wxyz(pose[:3, :3])
        out_q = slerp_quat(prev_q, cur_q, self.rotation_alpha)
        out[:3, :3] = quat_wxyz_to_rotmat(out_q)

        self.prev_pose = out.copy()
        return out


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
        description="RealSense 原生深度 -> FoundationPose 验证"
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--min_depth", type=float, default=0.1)
    parser.add_argument("--max_depth", type=float, default=3.0)
    parser.add_argument("--stats_interval", type=int, default=30)
    parser.add_argument("--show_window", type=int, default=1)
    parser.add_argument("--align_to_color", type=int, default=1)

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

    if not args.mesh_path.exists():
        raise FileNotFoundError(f"Mesh 不存在: {args.mesh_path}")

    pipeline = rs_any.pipeline()
    config = rs_any.config()
    config.enable_stream(
        rs_any.stream.color, args.width, args.height, rs_any.format.bgr8, args.fps
    )
    config.enable_stream(
        rs_any.stream.depth, args.width, args.height, rs_any.format.z16, args.fps
    )
    profile = pipeline.start(config)

    align = rs_any.align(rs_any.stream.color) if bool(args.align_to_color) else None
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = float(depth_sensor.get_depth_scale())
    color_profile = profile.get_stream(rs_any.stream.color).as_video_stream_profile()
    intr = color_profile.get_intrinsics()
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

    stabilizer = (
        PoseStabilizer(args.pose_translation_alpha, args.pose_rotation_alpha)
        if bool(args.stabilize_pose)
        else None
    )

    window_name = "RealSense NativeDepth -> FoundationPose"
    frame_count = 0
    start_time = time.perf_counter()
    last_stats_time = start_time
    capture_acc = 0.0
    pose_acc = 0.0

    logging.info(
        "[Config] size=%dx%d@%d depth_scale=%.6f align_to_color=%s symmetry=%s",
        args.width,
        args.height,
        args.fps,
        depth_scale,
        bool(args.align_to_color),
        args.symmetry_mode,
    )

    try:
        while True:
            t0 = time.perf_counter()
            frames = pipeline.wait_for_frames()
            if align is not None:
                frames = align.process(frames)

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color = np.asanyarray(color_frame.get_data())
            depth_m = (
                np.asanyarray(depth_frame.get_data()).astype(np.float64) * depth_scale
            )
            depth_m[(depth_m < args.min_depth) | (depth_m > args.max_depth)] = 0.0
            capture_acc += (time.perf_counter() - t0) * 1000.0

            t1 = time.perf_counter()
            result = tracker.process_frame(
                cast(np.ndarray, color), cast(np.ndarray, depth_m)
            )
            pose_acc += (time.perf_counter() - t1) * 1000.0

            if (
                result.phase == PoseTracker.Phase.TRACKING
                and result.pose_matrix is not None
                and stabilizer is not None
            ):
                stable_pose = stabilizer.stabilize(result.pose_matrix)
                result.pose_matrix = stable_pose
                result.color = tracker._draw_visualization(
                    cast(np.ndarray, color), stable_pose
                )
            elif result.phase != PoseTracker.Phase.TRACKING and stabilizer is not None:
                stabilizer.reset()

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
                pose_vis = result.color
                depth_vis = colorize_depth(
                    depth_m.astype(np.float32), args.min_depth, args.max_depth
                )
                canvas = np.hstack((pose_vis, depth_vis))
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
                    f"Capture+Align: {capture_acc/max(frame_count,1):.1f}ms | Pose: {pose_acc/max(frame_count,1):.1f}ms",
                    12,
                    54,
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
                    "[Stats] frames=%d phase=%s total_fps=%.1f proc_fps=%.1f capture=%.1fms pose=%.1fms depth_valid=%.1f%%",
                    frame_count,
                    phase,
                    total_fps,
                    proc_fps,
                    capture_acc / max(args.stats_interval, 1),
                    pose_acc / max(args.stats_interval, 1),
                    valid_ratio * 100.0,
                )
                last_stats_time = now
                capture_acc = 0.0
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
