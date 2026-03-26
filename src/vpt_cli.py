from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import cv2
import numpy as np

from modules.foundationpose import FoundationPoseConfig, FoundationPoseEstimator
from modules.realsense_rgbd import RGBDFrame
from modules.yoloe26 import Yoloe26Config, Yoloe26Segmenter
from modules.realsense_rgbd import RealSenseConfig, RealSenseRGBDSource


class PipelineStatus(Enum):
    DETECTING = "detecting"
    TRACKING = "tracking"
    LOST = "lost"


@dataclass(slots=True)
class PipelineResult:
    status: PipelineStatus
    pose: np.ndarray | None
    vis_bgr: np.ndarray
    mask_u8: np.ndarray | None
    debug: dict[str, str]


class _FirstFrameMaskPipeline:
    def __init__(
        self, segmenter: Yoloe26Segmenter, pose_estimator: FoundationPoseEstimator
    ) -> None:
        self.segmenter = segmenter
        self.pose_estimator = pose_estimator
        self._initialized = False

    def process(self, frame: RGBDFrame) -> PipelineResult:
        if not self._initialized:
            mask_result = self.segmenter.segment(frame)
            if mask_result.mask_u8 is None:
                return PipelineResult(
                    status=PipelineStatus.DETECTING,
                    pose=None,
                    vis_bgr=frame.color_bgr,
                    mask_u8=None,
                    debug={"stage": "segment"},
                )

            pose_result = self.pose_estimator.initialize(frame, mask_result.mask_u8)
            self._initialized = pose_result.pose_4x4 is not None
            return PipelineResult(
                status=(
                    PipelineStatus.TRACKING
                    if self._initialized
                    else PipelineStatus.LOST
                ),
                pose=pose_result.pose_4x4,
                vis_bgr=pose_result.vis_bgr,
                mask_u8=mask_result.mask_u8,
                debug={"stage": "init"},
            )

        pose_result = self.pose_estimator.track(frame)
        if pose_result.pose_4x4 is None:
            self._initialized = False
            status = PipelineStatus.LOST
        else:
            status = PipelineStatus.TRACKING

        return PipelineResult(
            status=status,
            pose=pose_result.pose_4x4,
            vis_bgr=pose_result.vis_bgr,
            mask_u8=None,
            debug={"stage": "track"},
        )


def _parse_cam_k(text: str) -> np.ndarray:
    values = [float(v.strip()) for v in text.split(",") if v.strip()]
    if len(values) != 9:
        raise ValueError("--cam_k 需要 9 个逗号分隔数字")
    return np.array(values, dtype=np.float64).reshape(3, 3)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VPT 模块化 CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_cmd = subparsers.add_parser(
        "run-rgbd-yoloe26-fp",
        help="RealSense RGBD + YOLOE26 首帧掩码 + FoundationPose",
    )
    run_cmd.add_argument(
        "--project_dir", type=Path, default=Path(__file__).resolve().parent.parent
    )
    run_cmd.add_argument("--mesh_path", type=Path, required=True)
    run_cmd.add_argument("--yoloe_model", type=Path, required=True)
    run_cmd.add_argument("--prompt", type=str, default="white block")
    run_cmd.add_argument("--cam_k", type=str, default="")
    run_cmd.add_argument("--width", type=int, default=640)
    run_cmd.add_argument("--height", type=int, default=480)
    run_cmd.add_argument("--fps", type=int, default=30)
    run_cmd.add_argument("--min_depth_m", type=float, default=0.1)
    run_cmd.add_argument("--max_depth_m", type=float, default=3.0)
    run_cmd.add_argument("--sam_conf", type=float, default=0.3)
    run_cmd.add_argument("--imgsz", type=int, default=640)
    run_cmd.add_argument("--max_det", type=int, default=1)
    run_cmd.add_argument("--mask_threshold", type=float, default=0.5)
    run_cmd.add_argument("--est_refine_iter", type=int, default=10)
    run_cmd.add_argument("--track_refine_iter", type=int, default=5)
    run_cmd.add_argument("--show_mask", type=int, default=1)
    return parser


def _run_rgbd_yoloe26_fp(args: argparse.Namespace) -> None:
    rs_source = RealSenseRGBDSource(
        RealSenseConfig(
            width=args.width,
            height=args.height,
            fps=args.fps,
            min_depth_m=args.min_depth_m,
            max_depth_m=args.max_depth_m,
            align_to_color=True,
        )
    )
    rs_source.start()

    cam_k = _parse_cam_k(args.cam_k) if args.cam_k else rs_source.cam_k
    segmenter = Yoloe26Segmenter(
        Yoloe26Config(
            model_path=args.yoloe_model,
            prompt=[args.prompt],
            conf=args.sam_conf,
            imgsz=args.imgsz,
            max_det=args.max_det,
            mask_threshold=args.mask_threshold,
        )
    )
    pose_estimator = FoundationPoseEstimator(
        FoundationPoseConfig(
            project_dir=args.project_dir,
            mesh_path=args.mesh_path,
            cam_k=cam_k,
            est_refine_iter=args.est_refine_iter,
            track_refine_iter=args.track_refine_iter,
        )
    )
    pipeline = _FirstFrameMaskPipeline(
        segmenter=segmenter, pose_estimator=pose_estimator
    )

    window_main = "VPT Pipeline - YOLOE26 First Mask + FoundationPose"
    window_mask = "VPT Pipeline - First Mask"
    fps_ema = 0.0
    alpha = 0.1

    try:
        while True:
            t0 = time.perf_counter()
            frame = rs_source.read()
            if frame is None:
                continue

            result = pipeline.process(frame)

            loop_s = max(time.perf_counter() - t0, 1e-6)
            inst_fps = 1.0 / loop_s
            fps_ema = (
                inst_fps if fps_ema == 0.0 else (1 - alpha) * fps_ema + alpha * inst_fps
            )

            vis = result.vis_bgr.copy()
            cv2.putText(
                vis,
                f"Status: {result.status.value}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 80),
                2,
            )
            cv2.putText(
                vis,
                f"FPS: {fps_ema:.1f}",
                (10, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 80),
                2,
            )
            cv2.putText(
                vis,
                "Key: q/ESC",
                (10, 79),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 80),
                2,
            )

            cv2.imshow(window_main, vis)
            if int(args.show_mask) == 1 and result.mask_u8 is not None:
                cv2.imshow(window_mask, result.mask_u8)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        rs_source.stop()
        cv2.destroyAllWindows()


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.command == "run-rgbd-yoloe26-fp":
        _run_rgbd_yoloe26_fp(args)
        return
    raise ValueError(f"未知命令: {args.command}")


if __name__ == "__main__":
    main()
