"""Quest 结构化 Pipeline：对外提供位姿 API，并在 main 中演示可视化。"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# 允许直接以脚本方式运行：python src/pipeline/quest_pipeline.py
if __package__ is None or __package__ == "":
    SRC_DIR = Path(__file__).resolve().parents[1]
    if str(SRC_DIR) not in sys.path:
        sys.path.append(str(SRC_DIR))

from modules import (  # noqa: E402
    FastFoundationStereoConfig,
    FastFoundationStereoRealtime,
    FoundationPoseConfig,
    FoundationPoseEstimator,
    QuestStereoCamera,
    QuestStereoFrame,
    Yoloe26Config,
    Yoloe26Masker,
)
from modules.cutie import CutieConfig, CutieTracker  # noqa: E402

THIS_FILE = Path(__file__).resolve()
SRC_DIR = THIS_FILE.parent.parent
PROJECT_DIR = SRC_DIR.parent


# =========================
# 数据结构定义（输入/输出）
# =========================


@dataclass
class StereoCalibration:
    """Quest 双目标定信息。"""

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
        """计算从标定坐标系到运行分辨率的中心裁剪+缩放映射。"""
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
        """把标定内参映射到运行分辨率下的内参矩阵 K。"""
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


@dataclass
class PipelineStepTiming:
    """单帧分阶段耗时（毫秒）。"""

    yolo_ms: float = 0.0
    depth_ms: float = 0.0
    cutie_ms: float = 0.0
    pose_ms: float = 0.0


@dataclass
class PipelineDebugData:
    """调试可视化数据，供 main 示例使用。"""

    vis_bgr: np.ndarray
    mask_bw: np.ndarray
    depth_vis_bgr: np.ndarray
    stereo_vis_bgr: np.ndarray


@dataclass
class PosePipelineOutput:
    """Pipeline API 输出：面向外部传输和上层业务。"""

    timestamp_ms: float
    stage: int
    phase: str
    det_count: int
    depth_valid_ratio: float
    fps: float
    pose_4x4: np.ndarray | None
    timing: PipelineStepTiming
    debug: PipelineDebugData | None = None


# =========================
# 公共工具函数
# =========================


def _to_bgr(gray_or_bgr: np.ndarray) -> np.ndarray:
    """把输入图像统一成 BGR 三通道。"""
    if gray_or_bgr.ndim == 2:
        return cv2.cvtColor(gray_or_bgr, cv2.COLOR_GRAY2BGR)
    return gray_or_bgr[..., :3]


def _draw_text(img: np.ndarray, text: str, x: int, y: int) -> None:
    """统一文本绘制样式，便于在高亮和暗部都可读。"""
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


def _colorize_depth(
    depth_m: np.ndarray, min_depth: float, max_depth: float
) -> np.ndarray:
    """把米制深度转为伪彩色，便于人工观察。"""
    depth_f = np.asarray(depth_m, dtype=np.float32)
    denom = max(float(max_depth) - float(min_depth), 1e-6)
    norm = ((depth_f - float(min_depth)) / denom).clip(0.0, 1.0)
    vis = cv2.applyColorMap((norm * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)

    invalid = (depth_f <= float(min_depth)) | (depth_f >= float(max_depth))
    invalid = invalid | (~np.isfinite(depth_f))
    if invalid.any():
        vis[invalid] = 0
    return vis


def _generate_cube_symmetry_tfs() -> np.ndarray:
    """生成立方体旋转对称群（24 个），用于 FoundationPose 对称约束。"""
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


def _bool_flag(flag: int | bool) -> bool:
    """统一处理 int/bool 风格开关参数。"""
    return bool(int(flag)) if isinstance(flag, (int, np.integer)) else bool(flag)


def _path_str(path: Any) -> str:
    """把 Path/其他路径对象统一转成字符串。"""
    return str(path)


# =========================
# Quest Pipeline 实现
# =========================


class QuestStereoPosePipeline:
    """
    Quest 位姿 Pipeline（结构化独立实现）。

    说明：
    1. `start()`：启动网络接收、读取标定并初始化模型。
    2. `run()`：处理一帧并返回 `PosePipelineOutput`。
    3. `stop()`：释放接收器资源。

    API 输入：
    - Quest 网络双目图（内部读取）。
    - 标定参数、模型参数（构建时注入）。

    API 输出：
    - `PosePipelineOutput`，核心是 `pose_4x4`（可直接用于网络回传）。
    """

    def __init__(
        self,
        args: argparse.Namespace,
        camera: QuestStereoCamera,
        yolo: Yoloe26Masker,
        ffs: FastFoundationStereoRealtime,
        cutie_tracker: CutieTracker | None,
    ) -> None:
        self.args = args
        self.camera = camera
        self.yolo = yolo
        self.ffs = ffs
        self.cutie_tracker = cutie_tracker

        # FoundationPose 在拿到映射后的 K 后初始化。
        self.pose_estimator: FoundationPoseEstimator | None = None

        # 标定与运行分辨率状态。
        self.calib: StereoCalibration | None = None
        self.cam_k: np.ndarray | None = None
        self.fx = 0.0
        self.frame_w = 0
        self.frame_h = 0

        # 启动阶段保存首帧，避免首帧丢失。
        self._pending_frame: QuestStereoFrame | None = None

        # 对称约束预先缓存，避免循环内重复计算。
        self.symmetry_tfs = (
            _generate_cube_symmetry_tfs() if args.symmetry_mode == "cube" else None
        )

        # 深度阈值与统计配置。
        self.min_depth = float(args.min_depth)
        self.max_depth = float(args.max_depth)
        self.stats_interval = max(int(args.stats_interval), 1)

        # 运行阶段：
        # 1=仅输入、2=+YOLO、3=+深度、4=+位姿。
        self.stage = 4

        # 运行状态位。
        self._started = False
        self._has_pose = False
        self._cutie_initialized = False

        # 统计累加器。
        self._frame_count = 0
        self._start_t = 0.0
        self._stats_t = 0.0
        self._yolo_acc = 0.0
        self._depth_acc = 0.0
        self._cutie_acc = 0.0
        self._pose_acc = 0.0

    @staticmethod
    def _read_json(path: Path) -> dict:
        """读取 UTF-8 JSON。"""
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _load_calibration(self, calib_dir: Path) -> StereoCalibration:
        """加载 Quest 左右相机标定并计算 baseline。"""
        left = self._read_json(calib_dir / "left_camera_characteristics.json")
        right = self._read_json(calib_dir / "right_camera_characteristics.json")

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

    @staticmethod
    def _preprocess_stereo_pair(
        left_raw: np.ndarray,
        right_raw: np.ndarray,
        target_width: int,
        target_height: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """把双目图统一到同分辨率与 BGR 格式。"""
        left_bgr = _to_bgr(left_raw)
        right_bgr = _to_bgr(right_raw)

        # 若左右尺寸不一致，先取共同最小尺寸对齐。
        if left_bgr.shape[:2] != right_bgr.shape[:2]:
            out_h = min(left_bgr.shape[0], right_bgr.shape[0])
            out_w = min(left_bgr.shape[1], right_bgr.shape[1])
            left_bgr = cv2.resize(
                left_bgr, (out_w, out_h), interpolation=cv2.INTER_LINEAR
            )
            right_bgr = cv2.resize(
                right_bgr, (out_w, out_h), interpolation=cv2.INTER_LINEAR
            )

        # 再根据目标处理分辨率进行缩放。
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
                    right_bgr,
                    (target_width, target_height),
                    interpolation=interpolation,
                )

        return left_bgr, right_bgr

    def _wait_first_frame(self, timeout_ms: int) -> QuestStereoFrame:
        """等待 Quest 首帧，避免在未收到数据时继续初始化。"""
        start = time.perf_counter()
        timeout_s = max(float(timeout_ms) / 1000.0, 0.1)

        while (time.perf_counter() - start) < timeout_s:
            frame = self.camera.get_stereo_frames(timeout_ms=100)
            if frame is not None:
                return frame

        raise RuntimeError("等待 Quest 首帧超时，请检查 Unity 发送端是否已启动。")

    def start(self) -> None:
        """启动 Pipeline：加载标定、启动接收、初始化 FoundationPose。"""
        if self._started:
            return

        # 先读取标定文件，确保参数完整再启动流程。
        self.calib = self._load_calibration(Path(self.args.calib_dir))
        logging.info(
            "[QuestCalib] fx=%.3f fy=%.3f cx=%.3f cy=%.3f baseline=%.6fm calib=%dx%d",
            self.calib.left_fx,
            self.calib.left_fy,
            self.calib.left_cx,
            self.calib.left_cy,
            self.calib.baseline_m,
            self.calib.calib_width,
            self.calib.calib_height,
        )

        # 启动网络接收器。
        self.camera.start()

        # 读取首帧并完成分辨率/内参映射。
        first = self._wait_first_frame(
            timeout_ms=max(int(self.args.recv_timeout_ms) * 50, 5000)
        )
        left0, right0 = self._preprocess_stereo_pair(
            first.left,
            first.right,
            target_width=max(int(self.args.process_width), 0),
            target_height=max(int(self.args.process_height), 0),
        )

        self.frame_h, self.frame_w = left0.shape[:2]
        self.cam_k = self.calib.scaled_k(
            width=self.frame_w,
            height=self.frame_h,
            assume_center_crop=_bool_flag(self.args.calib_assume_center_crop),
        )
        self.fx = float(self.cam_k[0, 0])

        logging.info(
            "[KMap] mode=%s fx=%.2f fy=%.2f cx=%.2f cy=%.2f frame=%dx%d",
            (
                "center-crop+scale"
                if _bool_flag(self.args.calib_assume_center_crop)
                else "linear-scale-only"
            ),
            float(self.cam_k[0, 0]),
            float(self.cam_k[1, 1]),
            float(self.cam_k[0, 2]),
            float(self.cam_k[1, 2]),
            self.frame_w,
            self.frame_h,
        )

        # 初始化 FoundationPose 估计器。
        cfg = FoundationPoseConfig(
            mesh_path=_path_str(self.args.mesh_path),
            cam_k=self.cam_k,
            est_refine_iter=int(self.args.est_refine_iter),
            track_refine_iter=int(self.args.track_refine_iter),
            symmetry_tfs=self.symmetry_tfs,
            debug=0,
            debug_dir=None,
        )
        self.pose_estimator = FoundationPoseEstimator(cfg)

        # 保存首帧供 run() 首次处理。
        self._pending_frame = first

        # 重置运行统计。
        self._started = True
        self._has_pose = False
        self._cutie_initialized = False
        self._frame_count = 0
        self._start_t = time.perf_counter()
        self._stats_t = self._start_t
        self._yolo_acc = 0.0
        self._depth_acc = 0.0
        self._cutie_acc = 0.0
        self._pose_acc = 0.0

    def stop(self) -> None:
        """停止 Pipeline：关闭接收器并清理状态。"""
        if not self._started:
            return
        self.camera.stop()
        self._started = False

    def set_stage(self, stage: int) -> None:
        """切换执行阶段，并重置跟踪状态避免状态污染。"""
        stage_int = int(stage)
        if stage_int < 1 or stage_int > 4:
            raise ValueError(f"stage 必须在 1..4 之间，当前值: {stage_int}")
        self.stage = stage_int
        self.reset_tracking_state()

    def reset_tracking_state(self) -> None:
        """仅重置位姿跟踪状态，不重启接收器。"""
        self._has_pose = False
        self._cutie_initialized = False
        if self.pose_estimator is not None:
            self.pose_estimator.reset()

    def _maybe_log_stats(self, output: PosePipelineOutput) -> None:
        """按固定间隔打印统计信息，便于线上观察性能。"""
        if self._frame_count % self.stats_interval != 0:
            return

        now = time.perf_counter()
        interval = max(now - self._stats_t, 1e-6)
        proc_fps = self.stats_interval / interval

        q_stats = self.camera.get_stats()

        logging.info(
            "[stats] frames=%d stage=%d phase=%s total_fps=%.1f proc_fps=%.1f "
            "avg(yolo/depth/cutie/pose)=%.1f/%.1f/%.1f/%.1fms depth_valid=%.1f%% "
            "recv=%s decode_fail=%s drained=%s",
            self._frame_count,
            self.stage,
            output.phase,
            output.fps,
            proc_fps,
            self._yolo_acc / self.stats_interval,
            self._depth_acc / self.stats_interval,
            self._cutie_acc / self.stats_interval,
            self._pose_acc / self.stats_interval,
            output.depth_valid_ratio * 100.0,
            q_stats.get("received", 0),
            q_stats.get("decode_failed", 0),
            q_stats.get("drained", 0),
        )

        self._stats_t = now
        self._yolo_acc = 0.0
        self._depth_acc = 0.0
        self._cutie_acc = 0.0
        self._pose_acc = 0.0

    def run(self, return_debug: bool = False) -> PosePipelineOutput | None:
        """
        执行一帧 Pipeline，并返回位姿结果。

        输入：
        - 内部输入源：Quest 网络双目帧（自动接收）。
        - 内部参数：阶段开关、模型配置、标定参数。

        输出：
        - `PosePipelineOutput`：其中 `pose_4x4` 为核心输出。
        - 若当前未收到帧，则返回 None（调用方可继续下一轮轮询）。
        """
        if not self._started:
            raise RuntimeError("Pipeline 尚未启动，请先调用 start()。")
        if self.pose_estimator is None:
            raise RuntimeError("pose_estimator 尚未初始化。")
        if self.calib is None:
            raise RuntimeError("标定信息尚未初始化。")

        # 读取一帧；优先消费 start() 保存的首帧。
        if self._pending_frame is not None:
            stereo = self._pending_frame
            self._pending_frame = None
        else:
            stereo = self.camera.get_stereo_frames()

        # Quest 接收可能超时，此时返回 None，让上层继续轮询。
        if stereo is None:
            return None

        # 统一双目图格式与分辨率。
        left_bgr, right_bgr = self._preprocess_stereo_pair(
            stereo.left,
            stereo.right,
            target_width=self.frame_w,
            target_height=self.frame_h,
        )

        # 默认占位数据，确保每个阶段都可以安全返回输出结构。
        timing = PipelineStepTiming()
        det_count = 0
        phase = "STAGE1_CAMERA"
        pose_4x4: np.ndarray | None = None

        mask_bw = np.zeros(left_bgr.shape[:2], dtype=np.uint8)
        depth_m = np.zeros(left_bgr.shape[:2], dtype=np.float32)
        vis_bgr = left_bgr.copy()

        # 阶段2：YOLO 分割。
        if self.stage >= 2:
            t0 = time.perf_counter()
            yolo_result = self.yolo.infer(left_bgr)
            timing.yolo_ms = (time.perf_counter() - t0) * 1000.0
            self._yolo_acc += timing.yolo_ms

            det_count = yolo_result.det_count
            mask_bw = yolo_result.mask_bw
            vis_bgr = yolo_result.overlay.copy()
            phase = "STAGE2_YOLO"

        # 阶段3：FFS 深度。
        if self.stage >= 3:
            t1 = time.perf_counter()
            depth_m = self.ffs.predict_depth(
                left_image=left_bgr,
                right_image=right_bgr,
                fx=self.fx,
                baseline=float(self.calib.baseline_m),
            )
            timing.depth_ms = (time.perf_counter() - t1) * 1000.0
            self._depth_acc += timing.depth_ms

            depth_m = np.asarray(depth_m, dtype=np.float32)
            invalid = (depth_m < self.min_depth) | (depth_m > self.max_depth)
            depth_m[invalid] = 0.0
            phase = "STAGE3_FFS"

        # 阶段4：FoundationPose 注册/跟踪。
        if self.stage >= 4:
            t2 = time.perf_counter()

            cutie_bbox = [-1, -1, 0, 0]
            cutie_mask: np.ndarray | None = None

            # 首次成功检测后注册位姿。
            if not self._has_pose:
                has_valid_mask = det_count > 0 and np.count_nonzero(mask_bw) > 0
                if has_valid_mask:
                    pose_4x4 = self.pose_estimator.register(
                        rgb=left_bgr,
                        depth=depth_m.astype(np.float64),
                        mask=mask_bw,
                    )
                    pose_4x4 = np.asarray(pose_4x4, dtype=np.float64).reshape(4, 4)
                    self._has_pose = True
                    vis_bgr = self.pose_estimator.visualize_pose(left_bgr, pose_4x4)

                    # 可选：用同帧 mask 初始化 Cutie。
                    if self.cutie_tracker is not None:
                        ct0 = time.perf_counter()
                        try:
                            _ = self.cutie_tracker.initialize(
                                left_bgr, init_mask=mask_bw
                            )
                            self._cutie_initialized = True
                        except Exception as exc:  # pragma: no cover
                            self._cutie_initialized = False
                            logging.warning("[cutie] 初始化失败: %s", exc)
                        timing.cutie_ms += (time.perf_counter() - ct0) * 1000.0
                    phase = "REGISTER"
                else:
                    phase = "WAIT_DETECT"

            # 已注册后进入跟踪。
            else:
                if self.cutie_tracker is not None and self._cutie_initialized:
                    ct0 = time.perf_counter()
                    try:
                        cutie_result = self.cutie_tracker.track(left_bgr)
                        cutie_bbox = cutie_result.bbox_xywh
                        cutie_mask = (cutie_result.mask > 0).astype(np.uint8) * 255

                        x, y, bw, bh = cutie_bbox
                        if bw > 0 and bh > 0:
                            cx = float(x + bw / 2.0)
                            cy = float(y + bh / 2.0)
                            self.pose_estimator.adjust_pose_to_image_point(cx, cy)
                        elif det_count > 0 and np.count_nonzero(mask_bw) > 0:
                            _ = self.cutie_tracker.initialize(
                                left_bgr, init_mask=mask_bw
                            )
                            self._cutie_initialized = True
                    except Exception as exc:  # pragma: no cover
                        logging.warning("[cutie] 跟踪失败: %s", exc)
                        self._cutie_initialized = False
                    timing.cutie_ms += (time.perf_counter() - ct0) * 1000.0

                pose_4x4 = self.pose_estimator.track(
                    rgb=left_bgr,
                    depth=depth_m.astype(np.float64),
                )
                pose_4x4 = np.asarray(pose_4x4, dtype=np.float64).reshape(4, 4)
                vis_bgr = self.pose_estimator.visualize_pose(left_bgr, pose_4x4)

                if cutie_bbox[2] > 0 and cutie_bbox[3] > 0:
                    x, y, bw, bh = cutie_bbox
                    cv2.rectangle(
                        vis_bgr,
                        (int(x), int(y)),
                        (int(x + bw), int(y + bh)),
                        (0, 255, 255),
                        2,
                    )
                    if cutie_mask is not None:
                        mask_bw = cutie_mask

                phase = "TRACK"

            timing.pose_ms = (time.perf_counter() - t2) * 1000.0
            self._pose_acc += timing.pose_ms
            self._cutie_acc += timing.cutie_ms

        # 更新帧统计。
        self._frame_count += 1
        elapsed = max(time.perf_counter() - self._start_t, 1e-6)
        fps = self._frame_count / elapsed
        depth_valid_ratio = float((depth_m > 0).mean()) if self.stage >= 3 else 0.0

        # 如需调试图像，则在 API 返回结构中附带，不在 run 内显示。
        debug_data: PipelineDebugData | None = None
        if return_debug:
            depth_vis_bgr = _colorize_depth(depth_m, self.min_depth, self.max_depth)
            stereo_vis_bgr = np.hstack((left_bgr, right_bgr))

            _draw_text(
                vis_bgr,
                f"{phase} | fps={fps:.1f} | stage={self.stage} | det={det_count}",
                12,
                28,
            )
            _draw_text(
                vis_bgr,
                (
                    f"yolo={timing.yolo_ms:.1f}ms depth={timing.depth_ms:.1f}ms "
                    f"cutie={timing.cutie_ms:.1f}ms pose={timing.pose_ms:.1f}ms "
                    f"depth_valid={depth_valid_ratio:.1%}"
                ),
                12,
                54,
            )
            _draw_text(stereo_vis_bgr, f"timestamp={stereo.timestamp_ms:.1f}ms", 12, 28)

            debug_data = PipelineDebugData(
                vis_bgr=vis_bgr,
                mask_bw=mask_bw,
                depth_vis_bgr=depth_vis_bgr,
                stereo_vis_bgr=stereo_vis_bgr,
            )

        output = PosePipelineOutput(
            timestamp_ms=float(stereo.timestamp_ms),
            stage=self.stage,
            phase=phase,
            det_count=det_count,
            depth_valid_ratio=depth_valid_ratio,
            fps=fps,
            pose_4x4=pose_4x4,
            timing=timing,
            debug=debug_data,
        )

        self._maybe_log_stats(output)
        return output


# =========================
# 参数与构建函数
# =========================


def build_arg_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="Quest 位姿 Pipeline（结构化 API 版）")

    # Quest 网络输入参数。
    parser.add_argument("--listen_host", type=str, default="*")
    parser.add_argument("--listen_port", type=int, default=5557)
    parser.add_argument("--recv_hwm", type=int, default=1)
    parser.add_argument("--recv_timeout_ms", type=int, default=100)

    # 标定与处理分辨率参数。
    parser.add_argument(
        "--calib_dir",
        type=Path,
        default=PROJECT_DIR / "docs" / "20260322_070544",
    )
    parser.add_argument("--calib_assume_center_crop", type=int, default=1)
    parser.add_argument("--process_width", type=int, default=640)
    parser.add_argument("--process_height", type=int, default=480)

    # YOLO 参数。
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

    # FFS 参数。
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

    # FoundationPose 参数。
    parser.add_argument(
        "--mesh_path",
        type=Path,
        default=PROJECT_DIR / "data" / "online" / "cube" / "mesh" / "cube.stl",
    )
    parser.add_argument("--est_refine_iter", type=int, default=5)
    parser.add_argument("--track_refine_iter", type=int, default=2)
    parser.add_argument(
        "--symmetry_mode", type=str, default="cube", choices=["none", "cube"]
    )

    # 统计与 2D tracker 参数。
    parser.add_argument("--stats_interval", type=int, default=30)
    parser.add_argument("--activate_2d_tracker", type=int, default=1)
    parser.add_argument("--cutie_erosion_size", type=int, default=5)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""
    return build_arg_parser().parse_args(argv)


def validate_paths(args: argparse.Namespace) -> None:
    """检查关键路径是否存在，避免运行中途失败。"""
    required_paths = [
        args.yolo_model_path,
        args.mobileclip2_path,
        args.ffs_model_path,
        args.mesh_path,
        args.calib_dir / "left_camera_characteristics.json",
        args.calib_dir / "right_camera_characteristics.json",
    ]
    for path in required_paths:
        if not Path(path).exists():
            raise FileNotFoundError(f"必要文件不存在: {path}")


def build_quest_pipeline(args: argparse.Namespace) -> QuestStereoPosePipeline:
    """构建 Quest Pipeline 对象（API 工厂函数）。"""
    validate_paths(args)

    camera = QuestStereoCamera(
        listen_host=str(args.listen_host),
        listen_port=int(args.listen_port),
        hwm=int(args.recv_hwm),
        timeout_ms=int(args.recv_timeout_ms),
    )

    yolo = Yoloe26Masker(
        Yoloe26Config(
            model_path=_path_str(args.yolo_model_path),
            conf=float(args.yolo_conf),
            imgsz=int(args.yolo_imgsz),
            max_det=int(args.yolo_max_det),
            mask_threshold=float(args.yolo_mask_threshold),
            use_half=False,
            device=None,
            mobileclip2_path=_path_str(args.mobileclip2_path),
        ),
        init_prompt=args.yolo_prompt,
    )

    ffs = FastFoundationStereoRealtime(
        FastFoundationStereoConfig(
            model_dir=_path_str(args.ffs_model_path),
            device=str(args.ffs_device),
            scale=float(args.ffs_scale),
            valid_iters=int(args.ffs_valid_iters),
            max_disp=int(args.ffs_max_disp),
            optimize_build_volume=str(args.ffs_optimize_build_volume),
        )
    )

    use_2d_tracker = _bool_flag(args.activate_2d_tracker)
    cutie_tracker = (
        CutieTracker(
            CutieConfig(seg_threshold=0.1, erosion_size=int(args.cutie_erosion_size))
        )
        if use_2d_tracker
        else None
    )

    return QuestStereoPosePipeline(
        args=args,
        camera=camera,
        yolo=yolo,
        ffs=ffs,
        cutie_tracker=cutie_tracker,
    )


def run_quest_pipeline(args: argparse.Namespace) -> None:
    """示例运行函数：循环调用 API，并在这里展示图像。"""
    pipeline = build_quest_pipeline(args)
    pipeline.start()

    cv2.namedWindow("Quest Pipeline", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Quest Mask", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Quest Depth", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Quest Stereo", cv2.WINDOW_AUTOSIZE)

    try:
        logging.info("按 1/2/3/4 切阶段，按 r 重置，按 q/ESC 退出")

        while True:
            # 这里通过 API 获取当前帧位姿结果。
            output = pipeline.run(return_debug=True)
            if output is None:
                continue

            # main 负责展示，不把显示逻辑放进 API run()。
            if output.debug is not None:
                cv2.imshow("Quest Pipeline", output.debug.vis_bgr)
                cv2.imshow("Quest Mask", output.debug.mask_bw)
                cv2.imshow("Quest Depth", output.debug.depth_vis_bgr)
                cv2.imshow("Quest Stereo", output.debug.stereo_vis_bgr)

            # 若拿到位姿，可在这里进行传输（示例仅日志显示）。
            if output.pose_4x4 is not None:
                t = output.pose_4x4[:3, 3]
                logging.debug(
                    "[pose] phase=%s xyz=(%.4f, %.4f, %.4f)",
                    output.phase,
                    float(t[0]),
                    float(t[1]),
                    float(t[2]),
                )

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key in (ord("1"), ord("2"), ord("3"), ord("4")):
                pipeline.set_stage(int(chr(key)))
                logging.info("[pipeline] switch stage -> %d", pipeline.stage)
            if key == ord("r"):
                pipeline.reset_tracking_state()
                logging.info("[pipeline] reset -> 等待重新检测")

    except KeyboardInterrupt:
        logging.info("\n[pipeline] 用户中断")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


def main(argv: list[str] | None = None) -> None:
    """脚本入口：构建并运行 main 示例。"""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_quest_pipeline(args)


if __name__ == "__main__":
    main()
