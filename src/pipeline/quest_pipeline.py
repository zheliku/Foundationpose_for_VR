"""Quest 结构化 Pipeline：对外提供位姿 API，并在 main 中演示可视化。"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# 允许直接以脚本方式运行：python src/pipeline/quest_pipeline.py
if __package__ is None or __package__ == "":
    SRC_DIR = Path(__file__).resolve().parents[1]
    if str(SRC_DIR) not in sys.path:
        sys.path.append(str(SRC_DIR))

from modules import (  # noqa: E402
    FastFoundationStereoRealtime,
    FoundationPoseEstimator,
    QuestStereoCamera,
    QuestStereoCalibration,
    QuestStereoMsg,
    Yoloe26Masker,
)
from modules.cutie import CutieTracker  # noqa: E402

THIS_FILE = Path(__file__).resolve()
SRC_DIR = THIS_FILE.parent.parent
PROJECT_DIR = SRC_DIR.parent


# =========================
# 数据结构定义（输入/输出）
# =========================


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
    frame_id: int | None
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


def _draw_hud(
    img: np.ndarray,
    lines: str | list[str],
    x: int = 12,
    y: int = 28,
    line_gap: int = 24,
) -> None:
    """统一绘制 HUD 文本并按图像宽度自适应换行。"""
    max_chars = max((img.shape[1] - x - 12) // 9, 12)
    wrapped: list[str] = []

    line_list = [lines] if isinstance(lines, str) else lines

    for line in line_list:
        if len(line) <= max_chars:
            wrapped.append(line)
            continue

        words = line.split(" ")
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    wrapped.append(current)
                current = word
        if current:
            wrapped.append(current)

    for idx, line in enumerate(wrapped):
        yy = y + idx * line_gap
        cv2.putText(
            img,
            line,
            (x, yy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (15, 15, 15),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            line,
            (x, yy),
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


# =========================
# Quest Pipeline 实现
# =========================


class QuestStereoPosePipeline:
    """
    Quest 位姿 Pipeline（结构化独立实现）。

    说明：
    1. `start()`：仅启动网络接收并重置运行状态。
    2. `run()`：处理一帧并返回 `PosePipelineOutput`。
    3. `stop()`：释放接收器资源。

    API 输入：
    - Quest 网络双目图（内部读取）。
    - 标定参数、模型参数（构建时注入）。

    API 输出：
    - `PosePipelineOutput`，核心是 `pose_4x4`（可直接用于网络回传）。
    """

    # 依赖注入。
    args: argparse.Namespace  # 命令行/配置参数集合。
    camera: QuestStereoCamera  # Quest 双目输入源。
    yolo: Yoloe26Masker  # 2D 分割模块。
    ffs: FastFoundationStereoRealtime  # 双目深度模块。
    cutie_tracker: CutieTracker | None  # 可选 2D 跟踪模块。

    # 运行期对象与标定状态。
    pose_estimator: FoundationPoseEstimator | None = None  # FoundationPose 估计器。
    calib: QuestStereoCalibration | None = None  # Quest 标定参数。
    cam_k: np.ndarray | None = None  # 映射到运行分辨率后的相机内参 K。
    fx: float = 0.0  # 当前帧使用的焦距 fx。
    frame_w: int = 0  # 运行分辨率宽度。
    frame_h: int = 0  # 运行分辨率高度。
    # 可配置运行参数。
    symmetry_tfs: np.ndarray | None = None  # 对称变换集合。
    min_depth: float = 0.1  # 最小有效深度（米）。
    max_depth: float = 3.0  # 最大有效深度（米）。
    stats_interval: int = 30  # 统计日志输出间隔（帧）。

    # 流程状态标志。
    stage: int = 4  # 当前处理阶段（1..4）。
    _started: bool = False  # Pipeline 是否已启动。
    _has_pose: bool = False  # 是否已完成首次注册并进入跟踪。
    _cutie_initialized: bool = False  # Cutie 是否已初始化。

    # 性能统计累加器。
    _frame_count: int = 0  # 已处理帧数。
    _start_t: float = 0.0  # 整体统计起始时间。
    _stats_t: float = 0.0  # 上次统计打印时间。
    _last_frame_t: float = 0.0  # 上一帧完成时间（用于实时 FPS）。
    _fps_rt: float = 0.0  # 平滑后的实时 FPS。
    _yolo_acc: float = 0.0  # 累计 YOLO 耗时（毫秒）。
    _depth_acc: float = 0.0  # 累计深度估计耗时（毫秒）。
    _cutie_acc: float = 0.0  # 累计 Cutie 耗时（毫秒）。
    _pose_acc: float = 0.0  # 累计位姿估计耗时（毫秒）。

    def __init__(
        self,
        args: argparse.Namespace,
        camera: QuestStereoCamera,
        yolo: Yoloe26Masker,
        ffs: FastFoundationStereoRealtime,
        cutie_tracker: CutieTracker | None,
    ) -> None:
        """
        初始化 Quest 位姿 Pipeline。

        参数：
        - args: 命令行与运行配置。
        - camera: Quest 双目输入模块。
        - yolo: 2D 分割模块。
        - ffs: 双目深度模块。
        - cutie_tracker: 可选 2D 跟踪模块。

        初始化流程：
        1. 绑定外部依赖对象。
        2. 读取对称性与深度阈值配置。
        3. 读取标定并计算运行内参 K。
        4. 创建 FoundationPose 估计器。
        """
        self.args = args
        self.camera = camera
        self.yolo = yolo
        self.ffs = ffs
        self.cutie_tracker = cutie_tracker

        # 对称约束预先缓存，避免循环内重复计算。
        self.symmetry_tfs = (
            _generate_cube_symmetry_tfs() if args.symmetry_mode == "cube" else None
        )

        # 深度阈值与统计配置。
        self.min_depth = float(args.min_depth)
        self.max_depth = float(args.max_depth)
        self.stats_interval = max(int(args.stats_interval), 1)

        # 标定文件在构建阶段读入一次，后续不再变化。
        self.calib = self.camera.get_stereo_calibration(Path(self.args.calib_dir))
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

        # 相机参数固定，因此在 __init__ 一次性完成 K 与 PoseEstimator 初始化。
        self.frame_w = max(int(self.args.process_width), 0)
        self.frame_h = max(int(self.args.process_height), 0)
        if self.frame_w <= 0 or self.frame_h <= 0:
            self.frame_w = int(self.calib.calib_width)
            self.frame_h = int(self.calib.calib_height)

        self.cam_k = self.calib.scaled_k(
            width=self.frame_w,
            height=self.frame_h,
            assume_center_crop=bool(self.args.calib_assume_center_crop),
        )
        self.fx = float(self.cam_k[0, 0])

        logging.info(
            "[KMap] mode=%s fx=%.2f fy=%.2f cx=%.2f cy=%.2f frame=%dx%d",
            (
                "center-crop+scale"
                if bool(self.args.calib_assume_center_crop)
                else "linear-scale-only"
            ),
            float(self.cam_k[0, 0]),
            float(self.cam_k[1, 1]),
            float(self.cam_k[0, 2]),
            float(self.cam_k[1, 2]),
            self.frame_w,
            self.frame_h,
        )

        self.pose_estimator = FoundationPoseEstimator(
            mesh_path=str(self.args.mesh_path),
            cam_k=self.cam_k,
            est_refine_iter=int(self.args.est_refine_iter),
            track_refine_iter=int(self.args.track_refine_iter),
            symmetry_tfs=self.symmetry_tfs,
            debug=0,
            debug_dir=None,
        )

    @staticmethod
    def _preprocess_stereo_pair(
        left_raw: np.ndarray,
        right_raw: np.ndarray,
        target_width: int,
        target_height: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """把双目图统一到同分辨率与 BGR 格式。"""
        # 直接在方法内完成灰度转 BGR，避免额外包装函数。
        left_bgr = (
            cv2.cvtColor(left_raw, cv2.COLOR_GRAY2BGR)
            if left_raw.ndim == 2
            else left_raw[..., :3]
        )
        right_bgr = (
            cv2.cvtColor(right_raw, cv2.COLOR_GRAY2BGR)
            if right_raw.ndim == 2
            else right_raw[..., :3]
        )

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

    def start(self) -> None:
        """启动 Pipeline：仅启动接收器并重置运行状态。"""
        if self._started:
            return

        # 启动网络接收器。
        self.camera.start()

        # 重置运行统计。
        self._started = True
        self._has_pose = False
        self._cutie_initialized = False
        if self.pose_estimator is not None:
            self.pose_estimator.reset()
        self._frame_count = 0
        self._start_t = time.perf_counter()
        self._stats_t = self._start_t
        self._last_frame_t = 0.0
        self._fps_rt = 0.0
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

    def _log_stats_if_due(self, output: PosePipelineOutput) -> None:
        """按固定间隔打印统计信息，便于线上观察性能。"""
        if self._frame_count % self.stats_interval != 0:
            return

        now = time.perf_counter()
        interval = max(now - self._stats_t, 1e-6)
        window_fps = self.stats_interval / interval

        q_stats = self.camera.get_stats()
        sender_est_ms = float(q_stats.get("sender_est_delay_ms", 0.0) or 0.0)
        sender_raw_ms = float(q_stats.get("sender_raw_delta_ms", 0.0) or 0.0)
        sender_gap = int(q_stats.get("sender_gap", 0) or 0)
        sender_fps = float(q_stats.get("sender_fps", 0.0) or 0.0)

        logging.info(
            "[stats] frames=%d stage=%d phase=%s rt_fps=%.1f window_fps=%.1f "
            "avg(yolo/depth/cutie/pose)=%.1f/%.1f/%.1f/%.1fms depth_valid=%.1f%% "
            "recv=%s decode_fail=%s sender_fps=%.1f sender_est=%.1fms sender_raw=%.1fms sender_gap=%s",
            self._frame_count,
            self.stage,
            output.phase,
            output.fps,
            window_fps,
            self._yolo_acc / self.stats_interval,
            self._depth_acc / self.stats_interval,
            self._cutie_acc / self.stats_interval,
            self._pose_acc / self.stats_interval,
            output.depth_valid_ratio * 100.0,
            q_stats.get("received", 0),
            q_stats.get("decode_failed", 0),
            sender_fps,
            sender_est_ms,
            sender_raw_ms,
            sender_gap,
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

        # 读取一帧网络双目图。
        stereo = self.camera.get_stereo_frames()

        # Quest 接收可能超时，此时返回 None，让上层继续轮询。
        if stereo is None:
            return None

        # StereoDecoder 理论上会填充左右图和接收时间，这里做保护以满足静态类型检查。
        if stereo.left is None or stereo.right is None or stereo.timestamp_ms is None:
            return None

        # K 在 __init__ 已固定，这里直接按目标分辨率处理双目图。
        left_bgr, right_bgr = self._preprocess_stereo_pair(
            stereo.left,
            stereo.right,
            target_width=self.frame_w,
            target_height=self.frame_h,
        )

        stereo_timestamp_ms = float(stereo.timestamp_ms)

        if self.pose_estimator is None:
            raise RuntimeError("pose_estimator 尚未初始化。")
        if self.calib is None:
            raise RuntimeError("标定信息尚未初始化。")

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
        now = time.perf_counter()
        self._frame_count += 1
        if self._last_frame_t > 0.0:
            dt = max(now - self._last_frame_t, 1e-6)
            inst_fps = 1.0 / dt
            self._fps_rt = (
                inst_fps
                if self._fps_rt <= 0.0
                else (self._fps_rt * 0.85 + inst_fps * 0.15)
            )
        fps = self._fps_rt if self._fps_rt > 0.0 else 0.0
        self._last_frame_t = now
        depth_valid_ratio = float((depth_m > 0).mean()) if self.stage >= 3 else 0.0

        # 如需调试图像，则在 API 返回结构中附带，不在 run 内显示。
        debug_data: PipelineDebugData | None = None
        if return_debug:
            depth_vis_bgr = _colorize_depth(depth_m, self.min_depth, self.max_depth)
            stereo_vis_bgr = np.hstack((left_bgr, right_bgr))

            # 首行固定展示 fps，其他信息按短行显示，避免窗口文本溢出。
            _draw_hud(
                vis_bgr,
                [
                    f"fps={fps:.1f} | stage={self.stage} | phase={phase}",
                    f"det={det_count} | depth_valid={depth_valid_ratio:.1%}",
                    f"yolo={timing.yolo_ms:.1f}ms | depth={timing.depth_ms:.1f}ms",
                    f"cutie={timing.cutie_ms:.1f}ms | pose={timing.pose_ms:.1f}ms",
                ],
            )
            _draw_hud(stereo_vis_bgr, f"timestamp={stereo_timestamp_ms:.1f}ms")

            debug_data = PipelineDebugData(
                vis_bgr=vis_bgr,
                mask_bw=mask_bw,
                depth_vis_bgr=depth_vis_bgr,
                stereo_vis_bgr=stereo_vis_bgr,
            )

        output = PosePipelineOutput(
            timestamp_ms=stereo_timestamp_ms,
            frame_id=stereo.frame_id,
            stage=self.stage,
            phase=phase,
            det_count=det_count,
            depth_valid_ratio=depth_valid_ratio,
            fps=fps,
            pose_4x4=pose_4x4,
            timing=timing,
            debug=debug_data,
        )

        self._log_stats_if_due(output)
        return output


# =========================
# 参数与构建函数
# =========================


def build_arg_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="Quest 位姿 Pipeline（结构化 API 版）")

    # Quest 网络输入参数。
    parser.add_argument(
        "--listen_host",
        type=str,
        default="*",
        help="Quest 双目接收端监听地址。'*' 表示监听所有网卡地址。",
    )
    parser.add_argument(
        "--listen_port",
        type=int,
        default=5557,
        help="Quest 双目接收端口（需与 Unity 发送端端口一致）。",
    )
    parser.add_argument(
        "--recv_hwm",
        type=int,
        default=1,
        help="接收端高水位（High Water Mark），值小可降低延迟但更容易丢旧帧。",
    )
    parser.add_argument(
        "--recv_timeout_ms",
        type=int,
        default=100,
        help="接收超时时间（毫秒）。超时后当前轮询返回空帧并继续下一轮。",
    )

    # 标定与处理分辨率参数。
    parser.add_argument(
        "--calib_dir",
        type=Path,
        default=PROJECT_DIR / "Calibration" / "20260322_070544",
        help="Quest 双目标定目录，需包含 left_camera_characteristics.json 与 right_camera_characteristics.json。",
    )
    parser.add_argument(
        "--calib_assume_center_crop",
        type=int,
        default=1,
        help="是否按“中心裁剪+缩放”映射标定内参到运行分辨率（1=是，0=仅线性缩放）。",
    )
    parser.add_argument(
        "--process_width",
        type=int,
        default=640,
        help="算法处理分辨率宽度（像素）。",
    )
    parser.add_argument(
        "--process_height",
        type=int,
        default=480,
        help="算法处理分辨率高度（像素）。",
    )

    # YOLO 参数。
    parser.add_argument(
        "--yolo_model_path",
        type=Path,
        default=PROJECT_DIR / "checkpoints" / "yoloe-26l-seg.pt",
        help="YOLOE 分割模型权重路径。",
    )
    parser.add_argument(
        "--mobileclip2_path",
        type=Path,
        default=PROJECT_DIR / "mobileclip2_b.ts",
        help="YOLOE 文本编码器（mobileclip2）权重路径。",
    )
    parser.add_argument(
        "--yolo_prompt",
        type=str,
        default="white cube",
        help="YOLOE 文本提示词，用于指定目标类别（例如 white cube）。",
    )
    parser.add_argument(
        "--yolo_conf",
        type=float,
        default=0.15,
        help="YOLO 检测置信度阈值，越大越严格。",
    )
    parser.add_argument(
        "--yolo_imgsz",
        type=int,
        default=640,
        help="YOLO 推理输入尺寸。",
    )
    parser.add_argument(
        "--yolo_max_det",
        type=int,
        default=2,
        help="YOLO 每帧最大保留检测数量。",
    )
    parser.add_argument(
        "--yolo_mask_threshold",
        type=float,
        default=0.5,
        help="YOLO 分割 mask 的二值化阈值。",
    )

    # FFS 参数。
    parser.add_argument(
        "--ffs_model_path",
        type=Path,
        default=PROJECT_DIR
        / "Fast-FoundationStereo"
        / "weights"
        / "23-36-37"
        / "model_best_bp2_serialize.pth",
        help="Fast-FoundationStereo 权重路径。",
    )
    parser.add_argument(
        "--ffs_device",
        type=str,
        default="cuda",
        help="FFS 推理设备（cuda 或 cpu）。",
    )
    parser.add_argument(
        "--ffs_scale",
        type=float,
        default=1.0,
        help="FFS 推理缩放系数，<1 可提速但会牺牲精度。",
    )
    parser.add_argument(
        "--ffs_valid_iters",
        type=int,
        default=4,
        help="FFS 网络迭代次数，越大通常越稳但更慢。",
    )
    parser.add_argument(
        "--ffs_max_disp",
        type=int,
        default=192,
        help="FFS 最大视差范围（像素）。",
    )
    parser.add_argument(
        "--ffs_optimize_build_volume",
        type=str,
        default="triton",
        choices=["triton", "pytorch1"],
        help="FFS 体构建优化后端：triton 或 pytorch1。",
    )
    parser.add_argument(
        "--ffs_seed",
        type=int,
        default=-1,
        help="FFS 随机种子；<0 为速度优先模式，>=0 为确定性模式。",
    )
    parser.add_argument(
        "--ffs_cudnn_benchmark",
        type=int,
        default=1,
        choices=[0, 1],
        help="FFS 是否开启 cudnn.benchmark（1=开启，0=关闭）。",
    )
    parser.add_argument(
        "--ffs_use_trt",
        type=int,
        default=1,
        choices=[0, 1],
        help="是否优先使用 TensorRT 路径（1=启用，0=关闭）。",
    )
    parser.add_argument(
        "--ffs_trt_precision",
        type=str,
        default="fp16",
        choices=["fp16", "fp32"],
        help="TRT engine 精度标签。",
    )
    parser.add_argument(
        "--ffs_trt_strict",
        type=int,
        default=0,
        choices=[0, 1],
        help="TRT 依赖/资源缺失时是否直接报错（1=严格模式）。",
    )
    parser.add_argument(
        "--ffs_trt_tag",
        type=str,
        default="",
        help="TRT artifact tag；为空时按输入尺寸与参数自动拼接。",
    )
    parser.add_argument(
        "--ffs_trt_platform_tag",
        type=str,
        default="",
        help="TRT 平台标签（如 win/linux）；为空时自动识别。",
    )
    parser.add_argument(
        "--ffs_trt_feature_engine_path",
        type=str,
        default="",
        help="TRT feature engine 绝对路径；为空时按 tag 自动匹配。",
    )
    parser.add_argument(
        "--ffs_trt_post_engine_path",
        type=str,
        default="",
        help="TRT post engine 绝对路径；为空时按 tag 自动匹配。",
    )
    parser.add_argument(
        "--min_depth",
        type=float,
        default=0.1,
        help="有效深度下限（米），低于此值会被置零。",
    )
    parser.add_argument(
        "--max_depth",
        type=float,
        default=3.0,
        help="有效深度上限（米），高于此值会被置零。",
    )

    # FoundationPose 参数。
    parser.add_argument(
        "--mesh_path",
        type=Path,
        default=PROJECT_DIR / "data" / "online" / "cube" / "mesh" / "cube.stl",
        help="FoundationPose 使用的目标物体网格模型路径。",
    )
    parser.add_argument(
        "--est_refine_iter",
        type=int,
        default=5,
        help="FoundationPose 首次注册阶段迭代次数。",
    )
    parser.add_argument(
        "--track_refine_iter",
        type=int,
        default=2,
        help="FoundationPose 连续跟踪阶段迭代次数。",
    )
    parser.add_argument(
        "--symmetry_mode",
        type=str,
        default="cube",
        choices=["none", "cube"],
        help="对称约束模式：none 关闭，cube 使用立方体 24 对称群。",
    )

    # 统计与 2D tracker 参数。
    parser.add_argument(
        "--stats_interval",
        type=int,
        default=30,
        help="统计日志输出间隔（按帧数计）。",
    )
    parser.add_argument(
        "--activate_2d_tracker",
        type=int,
        default=1,
        help="是否启用 Cutie 2D 跟踪（1=启用，0=关闭）。",
    )
    parser.add_argument(
        "--cutie_erosion_size",
        type=int,
        default=5,
        help="Cutie mask 腐蚀核大小（像素），用于稳定 bbox 边界。",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""
    return build_arg_parser().parse_args(argv)


def build_quest_pipeline(args: argparse.Namespace) -> QuestStereoPosePipeline:
    """构建 Quest Pipeline 对象（API 工厂函数）。"""
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

    camera = QuestStereoCamera(
        listen_host=str(args.listen_host),
        listen_port=int(args.listen_port),
        hwm=int(args.recv_hwm),
        timeout_ms=int(args.recv_timeout_ms),
    )

    yolo = Yoloe26Masker(
        model_path=str(args.yolo_model_path),
        init_prompt=args.yolo_prompt,
        conf=float(args.yolo_conf),
        imgsz=int(args.yolo_imgsz),
        max_det=int(args.yolo_max_det),
        mask_threshold=float(args.yolo_mask_threshold),
        use_half=False,
        device=None,
        mobileclip2_path=str(args.mobileclip2_path),
    )

    ffs = FastFoundationStereoRealtime(
        model_dir=str(args.ffs_model_path),
        device=str(args.ffs_device),
        scale=float(args.ffs_scale),
        valid_iters=int(args.ffs_valid_iters),
        max_disp=int(args.ffs_max_disp),
        optimize_build_volume=str(args.ffs_optimize_build_volume),
        seed=int(args.ffs_seed),
        cudnn_benchmark=bool(args.ffs_cudnn_benchmark),
        use_trt=bool(args.ffs_use_trt),
        trt_precision=str(args.ffs_trt_precision),
        trt_strict=bool(args.ffs_trt_strict),
        trt_tag=str(args.ffs_trt_tag),
        trt_platform_tag=str(args.ffs_trt_platform_tag),
        trt_feature_engine_path=str(args.ffs_trt_feature_engine_path),
        trt_post_engine_path=str(args.ffs_trt_post_engine_path),
    )

    use_2d_tracker = bool(args.activate_2d_tracker)
    cutie_tracker = (
        CutieTracker(seg_threshold=0.1, erosion_size=int(args.cutie_erosion_size))
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
