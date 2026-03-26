from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from modules.cutie import CutieConfig, CutieTracker2D
from modules.fast_foundationstereo import FFSConfig, FastFoundationStereoDepth
from modules.foundationpose import FoundationPoseConfig, FoundationPoseEstimator
from modules.realsense_rgbd import RGBDFrame
from modules.yoloe26 import Yoloe26Config, Yoloe26Segmenter
from zmq_utils import PayloadReceiver, StereoJpegDecoder


@dataclass(slots=True)
class StereoCalibration:
    left_fx: float
    left_fy: float
    left_cx: float
    left_cy: float
    baseline_m: float
    calib_width: int
    calib_height: int

    def scaled_k(self, width: int, height: int) -> np.ndarray:
        sx = width / max(self.calib_width, 1)
        sy = height / max(self.calib_height, 1)
        return np.array(
            [
                [self.left_fx * sx, 0.0, self.left_cx * sx],
                [0.0, self.left_fy * sy, self.left_cy * sy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )


@dataclass(slots=True)
class QuestStereoPoseConfig:
    listen_port: int
    project_dir: Path
    calib_dir: Path
    mesh_path: Path
    yoloe_model: Path
    mobileclip2_ts_path: Path | None
    yoloe_prompt: str
    ffs_model_path: Path
    process_width: int = 640
    process_height: int = 480
    max_frames: int = 0
    show_window: bool = True
    enable_cutie: bool = True
    preload_foundationpose: bool = False
    stats_interval: int = 30
    min_depth_m: float = 0.1
    max_depth_m: float = 3.0


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_quest_calibration(calib_dir: Path) -> StereoCalibration:
    left = _read_json(calib_dir / "left_camera_characteristics.json")
    right = _read_json(calib_dir / "right_camera_characteristics.json")

    left_intr = left["intrinsics"]
    left_t = np.array(left["pose"]["translation"], dtype=np.float64)
    right_t = np.array(right["pose"]["translation"], dtype=np.float64)
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
        baseline_m=baseline_m,
        calib_width=width,
        calib_height=height,
    )


def colorize_depth(depth: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
    norm = ((depth - min_depth) / max(max_depth - min_depth, 1e-6)).clip(0.0, 1.0)
    vis = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    invalid = (depth <= min_depth) | (depth >= max_depth) | (~np.isfinite(depth))
    if invalid.any():
        vis[invalid] = 0
    return vis


class QuestStereoPoseRunner:
    def __init__(self, config: QuestStereoPoseConfig) -> None:
        self.config = config

        self.calib = load_quest_calibration(config.calib_dir)
        logging.info("[QuestPipeline] Loading models...")
        self.segmenter = Yoloe26Segmenter(
            Yoloe26Config(
                model_path=config.yoloe_model,
                prompt=[config.yoloe_prompt],
                mobileclip2_ts_path=config.mobileclip2_ts_path,
                conf=0.15,
                imgsz=640,
                max_det=1,
                mask_threshold=0.5,
            )
        )
        self.depth_estimator = FastFoundationStereoDepth(
            FFSConfig(
                repo_dir=config.project_dir / "Fast-FoundationStereo",
                model_path=config.ffs_model_path,
                device="cuda",
                valid_iters=3,
                max_disp=192,
                scale=1.0,
                optimize_build_volume="pytorch1",
                input_mode="gray",
            )
        )
        self.pose_estimator: FoundationPoseEstimator | None = None
        if bool(config.preload_foundationpose):
            init_width = (
                int(config.process_width)
                if int(config.process_width) > 0
                else int(self.calib.calib_width)
            )
            init_height = (
                int(config.process_height)
                if int(config.process_height) > 0
                else int(self.calib.calib_height)
            )
            init_cam_k = self.calib.scaled_k(width=init_width, height=init_height)
            logging.info(
                "[QuestPipeline] Loading FoundationPose (init K: %dx%d)...",
                init_width,
                init_height,
            )
            self.pose_estimator = self._build_pose_estimator(init_cam_k)
            logging.info("[QuestPipeline] FoundationPose loaded")
        else:
            logging.info("[QuestPipeline] FoundationPose lazy init: ON")
        logging.info("[QuestPipeline] Models loaded")

        self.receiver = PayloadReceiver(
            endpoint=f"tcp://*:{config.listen_port}",
            bind=True,
            use_topic=False,
            hwm=1,
        )
        self.decoder = StereoJpegDecoder()

        self.cutie_tracker: CutieTracker2D | None = None
        self.initialized = False
        self.frame_count = 0

    def _build_pose_estimator(self, cam_k: np.ndarray) -> FoundationPoseEstimator:
        return FoundationPoseEstimator(
            FoundationPoseConfig(
                project_dir=self.config.project_dir,
                mesh_path=self.config.mesh_path,
                cam_k=cam_k,
                est_refine_iter=5,
                track_refine_iter=2,
            )
        )

    def _try_init_cutie(self, color_bgr: np.ndarray, mask_u8: np.ndarray) -> None:
        if not self.config.enable_cutie:
            return
        try:
            self.cutie_tracker = CutieTracker2D(CutieConfig())
            self.cutie_tracker.initialize(color_bgr, mask_u8)
        except Exception as exc:
            logging.warning("[Cutie] init failed, fallback to FP only: %s", exc)
            self.cutie_tracker = None

    def run(self) -> None:
        logging.info("[QuestPipeline] Listening on tcp://*:%d", self.config.listen_port)
        start_t = time.perf_counter()
        window_name = "Quest Stereo YOLOE+FFS+FoundationPose+Cutie"

        while True:
            parts = self.receiver.recv_payload(timeout_ms=100)
            if parts is None:
                continue

            parsed = self.decoder.decode(parts)
            if parsed is None:
                continue

            left_bgr, right_bgr = parsed
            left_bgr = np.asarray(left_bgr, dtype=np.uint8)
            right_bgr = np.asarray(right_bgr, dtype=np.uint8)

            if left_bgr.shape[:2] != right_bgr.shape[:2]:
                h = min(left_bgr.shape[0], right_bgr.shape[0])
                w = min(left_bgr.shape[1], right_bgr.shape[1])
                left_bgr = cv2.resize(left_bgr, (w, h))
                right_bgr = cv2.resize(right_bgr, (w, h))

            if self.config.process_width > 0 and self.config.process_height > 0:
                left_bgr = cv2.resize(
                    left_bgr, (self.config.process_width, self.config.process_height)
                )
                right_bgr = cv2.resize(
                    right_bgr, (self.config.process_width, self.config.process_height)
                )

            h, w = left_bgr.shape[:2]
            cam_k = self.calib.scaled_k(width=w, height=h)
            if self.pose_estimator is None:
                logging.info("[QuestPipeline] Lazy-loading FoundationPose...")
                self.pose_estimator = self._build_pose_estimator(cam_k)
                logging.info("[QuestPipeline] FoundationPose loaded")
            else:
                self.pose_estimator.set_camera_k(cam_k)

            depth_result = self.depth_estimator.estimate(
                left_bgr=left_bgr,
                right_bgr=right_bgr,
                fx=float(cam_k[0, 0]),
                baseline_m=self.calib.baseline_m,
            )
            depth_m = np.asarray(depth_result.depth_m, dtype=np.float64)
            depth_m[
                (depth_m < self.config.min_depth_m)
                | (depth_m > self.config.max_depth_m)
            ] = 0.0

            frame = RGBDFrame(
                color_bgr=np.asarray(left_bgr, dtype=np.uint8),
                depth_m=depth_m,
                timestamp_s=time.perf_counter(),
            )

            if not self.initialized:
                mask_result = self.segmenter.segment(frame)
                if mask_result.mask_u8 is not None:
                    pose_result = self.pose_estimator.initialize(
                        frame, mask_result.mask_u8
                    )
                    self.initialized = pose_result.pose_4x4 is not None
                    if self.initialized:
                        self._try_init_cutie(left_bgr, mask_result.mask_u8)
                else:
                    pose_result = None
            else:
                if self.cutie_tracker is not None and self.cutie_tracker.is_initialized:
                    bbox = self.cutie_tracker.track(left_bgr)
                    center = self.cutie_tracker.bbox_center_xy(bbox)
                    if center is not None:
                        self.pose_estimator.apply_tracking_hint(center[0], center[1])

                pose_result = self.pose_estimator.track(frame)
                if pose_result.pose_4x4 is None:
                    self.initialized = False

            self.frame_count += 1
            elapsed = max(time.perf_counter() - start_t, 1e-6)
            fps = self.frame_count / elapsed
            phase = "TRACKING" if self.initialized else "DETECTING"

            if self.frame_count % max(self.config.stats_interval, 1) == 0:
                logging.info(
                    "[QuestPipeline] frames=%d phase=%s fps=%.1f depth_valid=%.1f%%",
                    self.frame_count,
                    phase,
                    fps,
                    float((depth_m > 0).mean() * 100.0),
                )
                if pose_result is not None and pose_result.pose_4x4 is not None:
                    t = pose_result.pose_4x4[:3, 3]
                    logging.info(
                        "[Pose] t=(%.4f, %.4f, %.4f)m",
                        float(t[0]),
                        float(t[1]),
                        float(t[2]),
                    )

            if self.config.show_window:
                pose_vis = left_bgr if pose_result is None else pose_result.vis_bgr
                depth_vis = colorize_depth(
                    depth_m.astype(np.float32),
                    self.config.min_depth_m,
                    self.config.max_depth_m,
                )
                canvas = np.hstack((pose_vis, depth_vis))
                cv2.putText(
                    canvas,
                    f"Phase: {phase} | FPS: {fps:.1f}",
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 80),
                    2,
                )
                cv2.putText(
                    canvas,
                    "Pipeline: YOLOE + FFS + FoundationPose + Cutie",
                    (12, 56),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 80),
                    2,
                )
                cv2.imshow(window_name, canvas)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

            if (
                self.config.max_frames > 0
                and self.frame_count >= self.config.max_frames
            ):
                break

        self.close()

    def close(self) -> None:
        self.receiver.close()
        cv2.destroyAllWindows()


def run_quest_stereo_pose(config: QuestStereoPoseConfig) -> None:
    runner = QuestStereoPoseRunner(config)
    try:
        runner.run()
    finally:
        runner.close()


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Quest stereo + YOLOE + Fast-FoundationStereo + FoundationPose + Cutie 实时位姿测试"
    )
    parser.add_argument("--listen_port", type=int, default=5557)
    parser.add_argument(
        "--calib_dir",
        type=Path,
        default=project_dir / "docs/20260322_070544",
    )
    parser.add_argument(
        "--mesh_path",
        type=Path,
        default=project_dir / "data/online/cube/mesh/cube.stl",
    )
    parser.add_argument(
        "--yoloe_model",
        type=Path,
        default=project_dir / "checkpoints/yoloe-26l-seg.pt",
    )
    parser.add_argument(
        "--mobileclip2_ts",
        type=Path,
        default=project_dir / "mobileclip2_b.ts",
        help="mobileclip2_b.ts 的本地路径；用于离线或 GitHub 限流场景。",
    )
    parser.add_argument(
        "--ffs_model_path",
        type=Path,
        default=project_dir
        / "Fast-FoundationStereo/weights/20-30-48/model_best_bp2_serialize.pth",
    )
    parser.add_argument("--prompt", type=str, default="white block")
    parser.add_argument("--process_width", type=int, default=640)
    parser.add_argument("--process_height", type=int, default=480)
    parser.add_argument("--show_window", type=int, default=1)
    parser.add_argument("--enable_cutie", type=int, default=1)
    parser.add_argument(
        "--preload_foundationpose",
        type=int,
        default=1,
        help="1=启动时预加载 FoundationPose（更快首帧，但某些环境可能触发原生库退出）",
    )
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--stats_interval", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    project_dir = Path(__file__).resolve().parent.parent

    config = QuestStereoPoseConfig(
        listen_port=args.listen_port,
        project_dir=project_dir,
        calib_dir=args.calib_dir,
        mesh_path=args.mesh_path,
        yoloe_model=args.yoloe_model,
        mobileclip2_ts_path=args.mobileclip2_ts,
        yoloe_prompt=args.prompt,
        ffs_model_path=args.ffs_model_path,
        process_width=args.process_width,
        process_height=args.process_height,
        max_frames=args.max_frames,
        show_window=bool(args.show_window),
        enable_cutie=bool(args.enable_cutie),
        preload_foundationpose=bool(args.preload_foundationpose),
        stats_interval=args.stats_interval,
    )
    logging.info("[QuestPipeline] Starting...")
    logging.info(
        "[QuestPipeline] Expect stereo stream on tcp://*:%d", config.listen_port
    )
    run_quest_stereo_pose(config)


if __name__ == "__main__":
    main()
