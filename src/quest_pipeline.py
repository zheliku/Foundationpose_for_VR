"""
Quest 主流程：
1) 从网络接收左右双目图像。
2) YOLOE26 实时分割 cube mask。
3) Fast-FoundationStereo 估计同帧深度。
4) 将同帧 RGB + Depth + Mask 输入 FoundationPose 实时估计位姿并绘制。
5) 可选接入 Cutie 2D 跟踪，用 bbox 中心对 FoundationPose 的 pose_last 做先验校正。

说明：
- 本脚本与 realsense_pipeline.py 保持同一主流程结构。
- 仅保留 Quest 必要差异：网络接收与标定 K 映射。
- 删除额外稳定化分支（rectify/深度时域滤波/位姿平滑），先保证流程与基线一致。

按键：
- 1: 只看 Quest 双目输入
- 2: 看 Quest + YOLO mask
- 3: 看 Quest + YOLO + FFS 深度
- 4: 全流程（含 FoundationPose）
- r: 重置位姿状态，重新检测并注册
- q 或 ESC: 退出
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from modules import (
    FastFoundationStereoConfig,
    FastFoundationStereoRealtime,
    FoundationPoseConfig,
    FoundationPoseEstimator,
    QuestStereoCamera,
    QuestStereoFrame,
    Yoloe26Config,
    Yoloe26Masker,
)
from modules.cutie import CutieConfig, CutieTracker

THIS_FILE = Path(__file__).resolve()
SRC_DIR = THIS_FILE.parent
PROJECT_DIR = SRC_DIR.parent


@dataclass
class StereoCalibration:
    left_fx: float
    left_fy: float
    left_cx: float
    left_cy: float
    baseline_m: float
    calib_width: int
    calib_height: int

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
    ) -> np.ndarray:
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

    def _perm_parity(perm: tuple[int, int, int]) -> int:
        inv = 0
        for i in range(3):
            for j in range(i + 1, 3):
                if perm[i] > perm[j]:
                    inv += 1
        return 1 if (inv % 2 == 0) else -1

    for perm in itertools.permutations((0, 1, 2)):
        parity = _perm_parity(perm)
        for sx, sy, sz in itertools.product((-1.0, 1.0), repeat=3):
            if parity * sx * sy * sz < 0:
                continue
            r = np.zeros((3, 3), dtype=np.float64)
            r[0, perm[0]] = sx
            r[1, perm[1]] = sy
            r[2, perm[2]] = sz
            mats.append(r)

    out: list[np.ndarray] = []
    for r in mats:
        tf = np.eye(4, dtype=np.float64)
        tf[:3, :3] = r
        out.append(tf)
    return np.stack(out, axis=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quest 主流程位姿估计")

    parser.add_argument("--listen_host", type=str, default="*")
    parser.add_argument("--listen_port", type=int, default=5557)
    parser.add_argument("--recv_hwm", type=int, default=1)
    parser.add_argument("--recv_timeout_ms", type=int, default=100)

    parser.add_argument(
        "--calib_dir",
        type=Path,
        default=PROJECT_DIR / "docs" / "20260322_070544",
    )
    parser.add_argument("--calib_assume_center_crop", type=int, default=1)

    parser.add_argument("--process_width", type=int, default=640)
    parser.add_argument("--process_height", type=int, default=480)

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

    parser.add_argument("--activate_2d_tracker", type=int, default=1)
    parser.add_argument("--cutie_erosion_size", type=int, default=5)
    return parser.parse_args()


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


def preprocess_stereo_pair(
    left_raw: np.ndarray,
    right_raw: np.ndarray,
    target_width: int,
    target_height: int,
) -> tuple[np.ndarray, np.ndarray]:
    left_bgr = to_bgr(left_raw)
    right_bgr = to_bgr(right_raw)

    if left_bgr.shape[:2] != right_bgr.shape[:2]:
        out_h = min(left_bgr.shape[0], right_bgr.shape[0])
        out_w = min(left_bgr.shape[1], right_bgr.shape[1])
        left_bgr = cv2.resize(left_bgr, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        right_bgr = cv2.resize(
            right_bgr, (out_w, out_h), interpolation=cv2.INTER_LINEAR
        )

    if target_width > 0 and target_height > 0:
        h, w = left_bgr.shape[:2]
        if w != target_width or h != target_height:
            interpolation = (
                cv2.INTER_AREA
                if (target_width < w or target_height < h)
                else cv2.INTER_LINEAR
            )
            left_bgr = cv2.resize(
                left_bgr, (target_width, target_height), interpolation=interpolation
            )
            right_bgr = cv2.resize(
                right_bgr, (target_width, target_height), interpolation=interpolation
            )

    return left_bgr, right_bgr


def wait_first_frame(
    camera: QuestStereoCamera,
    timeout_ms: int,
) -> QuestStereoFrame:
    start = time.perf_counter()
    timeout_s = max(float(timeout_ms) / 1000.0, 0.1)
    while (time.perf_counter() - start) < timeout_s:
        frame = camera.get_stereo_frames(timeout_ms=100)
        if frame is not None:
            return frame
    raise RuntimeError("等待 Quest 首帧超时，请检查 Unity 发送端是否已启动。")


def validate_paths(args: argparse.Namespace) -> None:
    """检查关键模型文件是否存在。"""
    for p in [
        args.yolo_model_path,
        args.mobileclip2_path,
        args.ffs_model_path,
        args.mesh_path,
        args.calib_dir / "left_camera_characteristics.json",
        args.calib_dir / "right_camera_characteristics.json",
    ]:
        if not Path(p).exists():
            raise FileNotFoundError(f"必要文件不存在: {p}")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    validate_paths(args)

    calib = load_quest_calibration(args.calib_dir)
    logging.info(
        "[QuestCalib] fx=%.3f fy=%.3f cx=%.3f cy=%.3f baseline=%.6fm calib=%dx%d",
        calib.left_fx,
        calib.left_fy,
        calib.left_cx,
        calib.left_cy,
        calib.baseline_m,
        calib.calib_width,
        calib.calib_height,
    )

    camera = QuestStereoCamera(
        listen_host=str(args.listen_host),
        listen_port=int(args.listen_port),
        hwm=int(args.recv_hwm),
        timeout_ms=int(args.recv_timeout_ms),
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

    symmetry_tfs = (
        generate_cube_symmetry_tfs() if args.symmetry_mode == "cube" else None
    )

    use_2d_tracker = bool(args.activate_2d_tracker)
    cutie_tracker = (
        CutieTracker(
            CutieConfig(seg_threshold=0.1, erosion_size=int(args.cutie_erosion_size))
        )
        if use_2d_tracker
        else None
    )
    cutie_initialized = False

    camera.start()
    first = wait_first_frame(
        camera, timeout_ms=max(int(args.recv_timeout_ms) * 50, 5000)
    )
    left0, right0 = preprocess_stereo_pair(
        first.left,
        first.right,
        target_width=max(int(args.process_width), 0),
        target_height=max(int(args.process_height), 0),
    )

    h0, w0 = left0.shape[:2]
    cam_k = calib.scaled_k(
        width=w0,
        height=h0,
        assume_center_crop=bool(args.calib_assume_center_crop),
    )
    fx = float(cam_k[0, 0])
    logging.info(
        "[KMap] mode=%s fx=%.2f fy=%.2f cx=%.2f cy=%.2f frame=%dx%d",
        (
            "center-crop+scale"
            if bool(args.calib_assume_center_crop)
            else "linear-scale-only"
        ),
        float(cam_k[0, 0]),
        float(cam_k[1, 1]),
        float(cam_k[0, 2]),
        float(cam_k[1, 2]),
        w0,
        h0,
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
    # 1: Quest, 2: +YOLO, 3: +FFS, 4: +FoundationPose
    stage = 4
    has_pose = False

    frame_count = 0
    start_t = time.perf_counter()
    stats_t = start_t
    yolo_acc = 0.0
    ffs_acc = 0.0
    cutie_acc = 0.0
    pose_acc = 0.0

    cv2.namedWindow("Pipeline", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Mask", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Depth", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Stereo", cv2.WINDOW_AUTOSIZE)

    pending: QuestStereoFrame | None = first

    try:
        logging.info("按 1/2/3/4 切阶段，按 r 重置，按 q/ESC 退出")

        while True:
            stereo = pending if pending is not None else camera.get_stereo_frames()
            pending = None
            if stereo is None:
                continue

            left_bgr, right_bgr = preprocess_stereo_pair(
                stereo.left,
                stereo.right,
                target_width=w0,
                target_height=h0,
            )

            # 默认占位，确保任何阶段下都有可视化输出。
            yolo_ms = 0.0
            ffs_ms = 0.0
            pose_ms = 0.0
            cutie_ms = 0.0
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
                    fx=fx,
                    baseline=float(calib.baseline_m),
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

                cutie_bbox = [-1, -1, 0, 0]
                cutie_mask: np.ndarray | None = None

                if not has_pose:
                    if det_count > 0 and np.count_nonzero(mask_bw) > 0:
                        pose = pose_estimator.register(
                            rgb=left_bgr,
                            depth=depth_m.astype(np.float64),
                            mask=mask_bw,
                        )
                        vis = pose_estimator.visualize_pose(left_bgr, pose)
                        has_pose = True
                        if cutie_tracker is not None:
                            ct0 = time.perf_counter()
                            try:
                                _ = cutie_tracker.initialize(
                                    left_bgr, init_mask=mask_bw
                                )
                                cutie_initialized = True
                            except Exception as exc:
                                cutie_initialized = False
                                logging.warning("[cutie] 初始化失败: %s", exc)
                            cutie_ms = (time.perf_counter() - ct0) * 1000.0
                            cutie_acc += cutie_ms
                        phase = "REGISTER"
                    else:
                        phase = "WAIT_DETECT"
                else:
                    if cutie_tracker is not None and cutie_initialized:
                        ct0 = time.perf_counter()
                        try:
                            cutie_res = cutie_tracker.track(left_bgr)
                            cutie_bbox = cutie_res.bbox_xywh
                            cutie_mask = (cutie_res.mask > 0).astype(np.uint8) * 255

                            x, y, bw, bh = cutie_bbox
                            if bw > 0 and bh > 0:
                                cx = float(x + bw / 2.0)
                                cy = float(y + bh / 2.0)
                                pose_estimator.adjust_pose_to_image_point(cx, cy)
                            elif det_count > 0 and np.count_nonzero(mask_bw) > 0:
                                _ = cutie_tracker.initialize(
                                    left_bgr, init_mask=mask_bw
                                )
                                cutie_initialized = True
                        except Exception as exc:
                            logging.warning("[cutie] 跟踪失败: %s", exc)
                            cutie_initialized = False
                        cutie_ms = (time.perf_counter() - ct0) * 1000.0
                        cutie_acc += cutie_ms

                    pose = pose_estimator.track(
                        rgb=left_bgr,
                        depth=depth_m.astype(np.float64),
                    )
                    vis = pose_estimator.visualize_pose(left_bgr, pose)

                    if cutie_bbox[2] > 0 and cutie_bbox[3] > 0:
                        x, y, bw, bh = cutie_bbox
                        cv2.rectangle(
                            vis,
                            (int(x), int(y)),
                            (int(x + bw), int(y + bh)),
                            (0, 255, 255),
                            2,
                        )
                        if cutie_mask is not None:
                            mask_bw = cutie_mask
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
                f"yolo={yolo_ms:.1f}ms ffs={ffs_ms:.1f}ms cutie={cutie_ms:.1f}ms pose={pose_ms:.1f}ms depth_valid={depth_valid:.1%}",
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
                has_pose = False
                pose_estimator.reset()
                cutie_initialized = False
                logging.info("[pipeline] switch stage -> %d", stage)
            if key == ord("r"):
                has_pose = False
                pose_estimator.reset()
                cutie_initialized = False
                logging.info("[pipeline] reset -> 等待重新检测")

            if frame_count % max(int(args.stats_interval), 1) == 0:
                now = time.perf_counter()
                interval = max(now - stats_t, 1e-6)
                proc_fps = int(args.stats_interval) / interval
                q_stats = camera.get_stats()
                logging.info(
                    "[stats] frames=%d stage=%d phase=%s total_fps=%.1f proc_fps=%.1f avg(yolo/ffs/cutie/pose)=%.1f/%.1f/%.1f/%.1fms depth_valid=%.1f%% recv=%s decode_fail=%s drained=%s",
                    frame_count,
                    stage,
                    phase,
                    fps,
                    proc_fps,
                    yolo_acc / max(int(args.stats_interval), 1),
                    ffs_acc / max(int(args.stats_interval), 1),
                    cutie_acc / max(int(args.stats_interval), 1),
                    pose_acc / max(int(args.stats_interval), 1),
                    depth_valid * 100.0,
                    q_stats.get("received", 0),
                    q_stats.get("decode_failed", 0),
                    q_stats.get("drained", 0),
                )
                stats_t = now
                yolo_acc = 0.0
                ffs_acc = 0.0
                cutie_acc = 0.0
                pose_acc = 0.0

    except KeyboardInterrupt:
        logging.info("\n[pipeline] 用户中断")
    finally:
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
