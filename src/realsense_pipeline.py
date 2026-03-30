"""
RealSense 主流程：
1) 读取左右双目图像。
2) YOLOE26 实时分割 cube mask。
3) Fast-FoundationStereo 估计同帧深度。
4) 将同帧 RGB + Depth + Mask 输入 FoundationPose 实时估计位姿并绘制。

说明：
- 本脚本只保留一个主流程入口，不再提供独立测试函数。
- 模块测试放在主流程里做：按 1/2/3/4 可切换流程阶段进行检查。

按键：
- 1: 只看 RealSense 双目输入
- 2: 看 RealSense + YOLO mask
- 3: 看 RealSense + YOLO + FFS 深度
- 4: 全流程（含 FoundationPose）
- r: 重置位姿状态，重新检测并注册
- q 或 ESC: 退出
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

from modules import (
    FastFoundationStereoConfig,
    FastFoundationStereoRealtime,
    FoundationPoseConfig,
    FoundationPoseEstimator,
    RealSenseCamera,
    Yoloe26Config,
    Yoloe26Masker,
)

rs_any = cast(Any, rs)

THIS_FILE = Path(__file__).resolve()
SRC_DIR = THIS_FILE.parent
PROJECT_DIR = SRC_DIR.parent


def draw_text(img: np.ndarray, text: str, x: int, y: int) -> None:
    """统一文本绘制样式。"""
    cv2.putText(
        img,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (15, 15, 15),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )


def colorize_depth(depth: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
    """把米制深度渲染为伪彩色，便于观察。"""
    depth_f = np.asarray(depth, dtype=np.float32)
    norm = ((depth_f - min_depth) / max(max_depth - min_depth, 1e-6)).clip(0.0, 1.0)
    vis = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    invalid = (depth_f <= min_depth) | (depth_f >= max_depth) | (~np.isfinite(depth_f))
    if invalid.any():
        vis[invalid] = 0
    return vis


def to_bgr(gray_or_bgr: np.ndarray) -> np.ndarray:
    """统一为 BGR 三通道。"""
    if gray_or_bgr.ndim == 2:
        return cv2.cvtColor(gray_or_bgr, cv2.COLOR_GRAY2BGR)
    return gray_or_bgr[..., :3]


def generate_cube_symmetry_tfs() -> np.ndarray:
    """生成立方体旋转对称集合，用于 FoundationPose 的对称约束。"""
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

    out: list[np.ndarray] = []
    for r in mats:
        tf = np.eye(4, dtype=np.float64)
        tf[:3, :3] = r
        out.append(tf)
    return np.stack(out, axis=0)


def get_left_intrinsics_and_baseline(
    cam: RealSenseCamera,
) -> tuple[np.ndarray, float, float]:
    """读取左目 K、基线 baseline、左目 fx。"""
    if cam.pipeline is None:
        raise RuntimeError("RealSense pipeline 不可用，请先 start()。")

    profile = cam.pipeline.get_active_profile()
    left_stream = profile.get_stream(
        rs_any.stream.infrared, 1
    ).as_video_stream_profile()
    right_stream = profile.get_stream(
        rs_any.stream.infrared, 2
    ).as_video_stream_profile()

    intr = left_stream.get_intrinsics()
    extr = left_stream.get_extrinsics_to(right_stream)

    baseline = abs(float(extr.translation[0]))
    if baseline <= 0:
        raise RuntimeError("无法读取有效 baseline。")

    cam_k = np.array(
        [[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return cam_k, baseline, float(intr.fx)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RealSense 主流程位姿估计")

    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--serial_number", type=str, default=None)

    parser.add_argument(
        "--yolo_model_path",
        type=Path,
        default=PROJECT_DIR / "checkpoints" / "yoloe-26l-seg.pt",
    )
    parser.add_argument(
        "--mobileclip2_path",
        type=Path,
        default=PROJECT_DIR / "mobileclip2_b.ts",
    )
    parser.add_argument("--yolo_prompt", type=str, default="white cube")
    parser.add_argument("--yolo_conf", type=float, default=0.15)
    parser.add_argument("--yolo_imgsz", type=int, default=640)
    parser.add_argument("--yolo_max_det", type=int, default=2)
    parser.add_argument("--yolo_mask_threshold", type=float, default=0.5)

    parser.add_argument(
        "--ffs_model_path",
        type=Path,
        default=PROJECT_DIR
        / "Fast-FoundationStereo"
        / "weights"
        / "20-30-48"
        / "model_best_bp2_serialize.pth",
    )
    parser.add_argument("--ffs_device", type=str, default="cuda")
    parser.add_argument("--ffs_scale", type=float, default=1.0)
    parser.add_argument("--ffs_valid_iters", type=int, default=4)
    parser.add_argument("--ffs_max_disp", type=int, default=192)
    parser.add_argument(
        "--ffs_optimize_build_volume",
        type=str,
        default="triton",
        choices=["triton", "pytorch1"],
    )
    parser.add_argument("--min_depth", type=float, default=0.1)
    parser.add_argument("--max_depth", type=float, default=3.0)

    parser.add_argument(
        "--mesh_path",
        type=Path,
        default=PROJECT_DIR / "data" / "online" / "cube" / "mesh" / "cube.stl",
    )
    parser.add_argument("--est_refine_iter", type=int, default=5)
    parser.add_argument("--track_refine_iter", type=int, default=2)
    parser.add_argument(
        "--symmetry_mode",
        type=str,
        default="cube",
        choices=["none", "cube"],
    )

    parser.add_argument("--stats_interval", type=int, default=30)
    return parser.parse_args()


def validate_paths(args: argparse.Namespace) -> None:
    """检查关键模型文件是否存在。"""
    for p in [
        args.yolo_model_path,
        args.mobileclip2_path,
        args.ffs_model_path,
        args.mesh_path,
    ]:
        if not Path(p).exists():
            raise FileNotFoundError(f"必要文件不存在: {p}")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    validate_paths(args)

    camera = RealSenseCamera(
        width=int(args.width),
        height=int(args.height),
        fps=int(args.fps),
        serial_number=args.serial_number,
    )

    yolo = Yoloe26Masker(
        Yoloe26Config(
            model_path=str(args.yolo_model_path),
            conf=float(args.yolo_conf),
            imgsz=int(args.yolo_imgsz),
            max_det=int(args.yolo_max_det),
            mask_threshold=float(args.yolo_mask_threshold),
            use_half=False,
            device=None,
            mobileclip2_path=str(args.mobileclip2_path),
        ),
        init_prompt=args.yolo_prompt,
    )

    ffs = FastFoundationStereoRealtime(
        FastFoundationStereoConfig(
            model_dir=str(args.ffs_model_path),
            device=str(args.ffs_device),
            scale=float(args.ffs_scale),
            valid_iters=int(args.ffs_valid_iters),
            max_disp=int(args.ffs_max_disp),
            optimize_build_volume=str(args.ffs_optimize_build_volume),
        )
    )

    camera.start()
    cam_k, baseline, fx = get_left_intrinsics_and_baseline(camera)

    symmetry_tfs = (
        generate_cube_symmetry_tfs() if args.symmetry_mode == "cube" else None
    )
    pose_estimator = FoundationPoseEstimator(
        FoundationPoseConfig(
            mesh_path=str(args.mesh_path),
            cam_k=cam_k,
            est_refine_iter=int(args.est_refine_iter),
            track_refine_iter=int(args.track_refine_iter),
            symmetry_tfs=symmetry_tfs,
            debug=0,
            debug_dir=None,
        )
    )

    # stage 用于在同一个主流程中测试各阶段，不拆分独立方法。
    # 1: RealSense, 2: +YOLO, 3: +FFS, 4: +FoundationPose
    stage = 4
    has_pose = False

    frame_count = 0
    start_t = time.perf_counter()
    stats_t = start_t
    yolo_acc = 0.0
    ffs_acc = 0.0
    pose_acc = 0.0

    cv2.namedWindow("Pipeline", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Mask", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Depth", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Stereo", cv2.WINDOW_AUTOSIZE)

    try:
        logging.info("按 1/2/3/4 切阶段，按 r 重置，按 q/ESC 退出")

        while True:
            stereo = camera.get_stereo_frames()
            left_bgr = to_bgr(stereo.left)
            right_bgr = to_bgr(stereo.right)

            # 默认占位，确保任何阶段下都有可视化输出。
            yolo_ms = 0.0
            ffs_ms = 0.0
            pose_ms = 0.0
            det_count = 0
            mask_bw = np.zeros(left_bgr.shape[:2], dtype=np.uint8)
            depth_m = np.zeros(left_bgr.shape[:2], dtype=np.float32)
            vis = left_bgr.copy()
            phase = "STAGE1_CAMERA"

            if stage >= 2:
                t0 = time.perf_counter()
                yolo_res = yolo.infer(left_bgr)
                yolo_ms = (time.perf_counter() - t0) * 1000.0
                yolo_acc += yolo_ms

                det_count = yolo_res.det_count
                mask_bw = yolo_res.mask_bw
                vis = yolo_res.overlay.copy()
                phase = "STAGE2_YOLO"

            if stage >= 3:
                t1 = time.perf_counter()
                depth_m = ffs.predict_depth(
                    left_image=left_bgr,
                    right_image=right_bgr,
                    fx=float(fx),
                    baseline=float(baseline),
                )
                depth_m = np.asarray(depth_m, dtype=np.float32)
                depth_m[
                    (depth_m < float(args.min_depth))
                    | (depth_m > float(args.max_depth))
                ] = 0
                ffs_ms = (time.perf_counter() - t1) * 1000.0
                ffs_acc += ffs_ms
                phase = "STAGE3_FFS"

            if stage >= 4:
                t2 = time.perf_counter()

                # register 使用“同帧 left + 同帧 depth + 同帧 mask”。
                if not has_pose:
                    if det_count > 0 and np.count_nonzero(mask_bw) > 0:
                        pose = pose_estimator.register(
                            rgb=left_bgr,
                            depth=depth_m.astype(np.float64),
                            mask=mask_bw,
                        )
                        vis = pose_estimator.visualize_pose(left_bgr, pose)
                        has_pose = True
                        phase = "REGISTER"
                    else:
                        phase = "WAIT_DETECT"
                else:
                    pose = pose_estimator.track(
                        rgb=left_bgr,
                        depth=depth_m.astype(np.float64),
                    )
                    vis = pose_estimator.visualize_pose(left_bgr, pose)
                    phase = "TRACK"

                pose_ms = (time.perf_counter() - t2) * 1000.0
                pose_acc += pose_ms

            frame_count += 1
            elapsed = max(time.perf_counter() - start_t, 1e-6)
            fps = frame_count / elapsed
            depth_valid = float((depth_m > 0).mean()) if stage >= 3 else 0.0

            depth_vis = colorize_depth(
                depth_m, float(args.min_depth), float(args.max_depth)
            )
            stereo_vis = np.hstack((left_bgr, right_bgr))

            draw_text(
                vis,
                f"{phase} | fps={fps:.1f} | stage={stage} | det={det_count}",
                12,
                28,
            )
            draw_text(
                vis,
                f"yolo={yolo_ms:.1f}ms ffs={ffs_ms:.1f}ms pose={pose_ms:.1f}ms depth_valid={depth_valid:.1%}",
                12,
                54,
            )
            draw_text(vis, "key: 1/2/3/4 stage  r reset  q quit", 12, 80)
            draw_text(stereo_vis, f"timestamp={stereo.timestamp_ms:.1f}ms", 12, 28)

            cv2.imshow("Pipeline", vis)
            cv2.imshow("Mask", mask_bw)
            cv2.imshow("Depth", depth_vis)
            cv2.imshow("Stereo", stereo_vis)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key in (ord("1"), ord("2"), ord("3"), ord("4")):
                stage = int(chr(key))
                # 切换阶段时重置 pose，避免阶段切换造成状态污染。
                has_pose = False
                pose_estimator.reset()
                logging.info("[pipeline] switch stage -> %d", stage)
            if key == ord("r"):
                has_pose = False
                pose_estimator.reset()
                logging.info("[pipeline] reset -> 等待重新检测")

            if frame_count % max(int(args.stats_interval), 1) == 0:
                now = time.perf_counter()
                interval = max(now - stats_t, 1e-6)
                proc_fps = int(args.stats_interval) / interval
                logging.info(
                    "[stats] frames=%d stage=%d phase=%s total_fps=%.1f proc_fps=%.1f avg(yolo/ffs/pose)=%.1f/%.1f/%.1fms",
                    frame_count,
                    stage,
                    phase,
                    fps,
                    proc_fps,
                    yolo_acc / max(int(args.stats_interval), 1),
                    ffs_acc / max(int(args.stats_interval), 1),
                    pose_acc / max(int(args.stats_interval), 1),
                )
                stats_t = now
                yolo_acc = 0.0
                ffs_acc = 0.0
                pose_acc = 0.0

    except KeyboardInterrupt:
        logging.info("\n[pipeline] 用户中断")
    finally:
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
