"""
Fast-FoundationStereo 实时推理（精简核心版）

目标：
1. 只保留核心功能：实时双目深度估计。
2. 简单显示：预测深度 vs RealSense 硬件原生深度。
3. 与 realsense.py 直接配合使用，不引入额外复杂功能。
"""

from __future__ import annotations

import importlib
import logging
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np


def _draw_label(img: np.ndarray, text: str, x: int, y: int) -> None:
    cv2.putText(
        img,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (10, 10, 10),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )


def _colorize_depth(
    depth: np.ndarray,
    min_depth: float,
    max_depth: float,
    invalid_mask: np.ndarray | None = None,
) -> np.ndarray:
    denom = max(float(max_depth) - float(min_depth), 1e-6)
    norm = ((depth - float(min_depth)) / denom).clip(0.0, 1.0)
    vis = cv2.applyColorMap((norm * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
    if invalid_mask is not None and invalid_mask.any():
        vis[invalid_mask] = 0
    return vis


def _colorize_error(
    error_map: np.ndarray,
    error_max: float,
    invalid_mask: np.ndarray | None = None,
) -> np.ndarray:
    norm = (error_map / max(float(error_max), 1e-6)).clip(0.0, 1.0)
    gray = (norm * 255.0).astype(np.uint8)
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if invalid_mask is not None and invalid_mask.any():
        vis[invalid_mask] = 0
    return vis


class _PyTorchStereoBackend:
    """PyTorch 推理后端，负责模型加载、体构建后端检查与前向推理。"""

    def __init__(self, host: "FastFoundationStereoRealtime") -> None:
        self.host = host

    def prepare_optimize_build_volume(self) -> None:
        """检查 Triton 可用性，避免误判已启用加速。"""
        if self.host.optimize_build_volume != "triton":
            return
        try:
            importlib.import_module("triton")
        except Exception:
            logging.warning("未检测到 triton，optimize_build_volume 回退为 pytorch1。")
            self.host.optimize_build_volume = "pytorch1"

    def load_model(self) -> None:
        """按需加载 PyTorch 模型。"""
        if self.host.model is not None:
            self.host.runtime_backend = "pytorch"
            return
        if self.host.model_pth_path is None:
            raise FileNotFoundError("未找到可用的 pth 文件，无法回退到 PyTorch 推理。")

        self.host.model = self.host.torch.load(
            str(self.host.model_pth_path),
            map_location="cpu",
            weights_only=False,
        )
        self.host.model.args.valid_iters = int(self.host.valid_iters)
        self.host.model.args.max_disp = int(self.host.max_disp)
        self.host.model = self.host.model.to(self.host.device).eval()
        self.host.runtime_backend = "pytorch"

    def predict_disparity(self, left_t: Any, right_t: Any) -> Any:
        """执行 PyTorch 路径的视差推理。"""
        self.load_model()

        padder = self.host.InputPadder(left_t.shape, divis_by=32, force_square=False)
        left_t, right_t = padder.pad(left_t, right_t)

        if self.host.device.type == "cuda":
            if hasattr(self.host.torch, "autocast"):
                autocast_ctx = self.host.torch.autocast(
                    "cuda",
                    enabled=True,
                    dtype=self.host.AMP_DTYPE,
                )
            else:
                autocast_ctx = self.host.torch.cuda.amp.autocast(
                    enabled=True,
                    dtype=self.host.AMP_DTYPE,
                )
        else:
            autocast_ctx = nullcontext()

        with self.host.torch.inference_mode():
            with autocast_ctx:
                disp = self.host.model.forward(
                    left_t,
                    right_t,
                    iters=int(self.host.valid_iters),
                    test_mode=True,
                    optimize_build_volume=str(self.host.optimize_build_volume),
                )
        if self.host.device.type == "cuda":
            self.host.torch.cuda.synchronize()
        return padder.unpad(disp.float())


class _TrtStereoBackend:
    """TRT 推理后端，负责 engine 匹配、runner 创建与前向推理。"""

    def __init__(self, host: "FastFoundationStereoRealtime") -> None:
        self.host = host

    @staticmethod
    def _platform_tag() -> str:
        if sys.platform.startswith("win"):
            return "win"
        if sys.platform.startswith("linux"):
            return "linux"
        if sys.platform == "darwin":
            return "mac"
        return "unknown"

    @staticmethod
    def _artifact_tag(height: int, width: int, valid_iters: int, max_disp: int) -> str:
        return f"h{height}-w{width}-it{valid_iters}-md{max_disp}"

    @staticmethod
    def _first_existing_path(candidates: list[Path]) -> Path | None:
        for path in candidates:
            if path.is_file():
                return path
        return None

    def ensure_runner(self, infer_h: int, infer_w: int) -> None:
        """按输入分辨率与参数自动匹配并加载 TRT engine。"""
        if self.host.trt_runner is not None and self.host.trt_input_hw == (
            infer_h,
            infer_w,
        ):
            return

        if self.host.TrtRunner is None or self.host.OmegaConf is None:
            self.host._fallback_to_pytorch("TRT 依赖未准备完成")
            return

        model_root = self.host.model_root_dir
        if model_root is None:
            self.host._fallback_to_pytorch("模型目录不存在")
            return

        tag = self.host.trt_tag.strip() or self._artifact_tag(
            infer_h,
            infer_w,
            self.host.valid_iters,
            self.host.max_disp,
        )
        platform_tag = self.host.trt_platform_tag.strip() or self._platform_tag()

        def resolve_engine_path(
            explicit_path: str,
            runner_name: str,
        ) -> Path | None:
            if explicit_path:
                return Path(explicit_path).resolve()
            return self._first_existing_path(
                [
                    model_root
                    / f"{runner_name}-{tag}.{platform_tag}.{self.host.trt_precision}.engine",
                    model_root / f"{runner_name}-{tag}.{platform_tag}.engine",
                    model_root / f"{runner_name}-{tag}.engine",
                ]
            )

        feature_engine_path = resolve_engine_path(
            self.host.trt_feature_engine_path,
            "feature_runner",
        )
        post_engine_path = resolve_engine_path(
            self.host.trt_post_engine_path,
            "post_runner",
        )

        if feature_engine_path is None or post_engine_path is None:
            self.host._fallback_to_pytorch(
                f"未找到匹配的 engine，tag={tag}, platform={platform_tag}, precision={self.host.trt_precision}"
            )
            return

        try:
            # 不依赖导出时 YAML，直接用运行参数构造 TRT 运行配置。
            cfg_data = {
                "max_disp": int(self.host.max_disp),
                "valid_iters": int(self.host.valid_iters),
                "normalize": True,
                "cv_group": 8,
            }

            cfg = self.host.OmegaConf.create(cfg_data)
            self.host.trt_runner = self.host.TrtRunner(
                cfg,
                str(feature_engine_path),
                str(post_engine_path),
            )
            self.host.trt_input_hw = (infer_h, infer_w)
            self.host.runtime_backend = "trt"
            logging.info(
                "TRT runner ready: tag=%s, size=%dx%d, feature=%s, post=%s",
                tag,
                infer_h,
                infer_w,
                feature_engine_path.name,
                post_engine_path.name,
            )
        except Exception as exc:
            self.host._fallback_to_pytorch("创建 TRT runner 失败", exc)

    def predict_disparity(self, left_t: Any, right_t: Any) -> Any:
        """执行 TRT 路径的视差推理。"""
        infer_h = int(left_t.shape[2])
        infer_w = int(left_t.shape[3])
        self.ensure_runner(infer_h, infer_w)

        # 可能在 ensure_runner 里触发了回退。
        if self.host.trt_runner is None:
            return self.host._pt_backend.predict_disparity(left_t, right_t)

        with self.host.torch.inference_mode():
            disp = self.host.trt_runner.forward(left_t, right_t)
        if self.host.device.type == "cuda":
            self.host.torch.cuda.synchronize()
        return disp.float()


class FastFoundationStereoRealtime:
    """实时双目深度估计器。

    架构说明：
    1. `_PyTorchStereoBackend` 负责 PyTorch 模型推理。
    2. `_TrtStereoBackend` 负责 TRT engine 匹配与推理。
    3. 当前类只做参数管理、后端路由、失败回退与统一输出。
    """

    def __init__(
        self,
        model_dir: str,
        device: str = "cuda",
        scale: float = 1.0,
        valid_iters: int = 4,
        max_disp: int = 192,
        optimize_build_volume: str = "triton",
        seed: int = -1,
        cudnn_benchmark: bool = True,
        use_trt: bool = True,
        trt_precision: str = "fp16",
        trt_strict: bool = False,
        trt_tag: str = "",
        trt_platform_tag: str = "",
        trt_feature_engine_path: str = "",
        trt_post_engine_path: str = "",
    ) -> None:
        """
        初始化 Fast-FoundationStereo 推理器。

        参数：
        - model_dir: 模型权重路径。
        - device: 推理设备，支持 cuda/cpu。
        - scale: 推理缩放比例。
        - valid_iters: 网络迭代次数。
        - max_disp: 最大视差。
        - optimize_build_volume: 体构建优化模式。
        - use_trt: 是否优先启用 TRT 推理。
        - trt_precision: TRT 精度标签（用于匹配 engine 命名）。
        - trt_strict: TRT 资源缺失时是否直接报错。

        初始化流程：
        1. 保存推理参数。
        2. 配置工程路径并动态导入依赖。
        3. 设置日志、随机种子与设备。
        4. 默认走 TRT，失败时按策略回退 PyTorch。
        """
        self.scale, self.valid_iters, self.max_disp = (
            float(scale),
            int(valid_iters),
            int(max_disp),
        )
        self.optimize_build_volume = str(optimize_build_volume)
        self.seed, self.cudnn_benchmark = int(seed), bool(cudnn_benchmark)
        self.use_trt = bool(use_trt)
        self.trt_precision = str(trt_precision).lower()
        self.trt_strict = bool(trt_strict)
        self.trt_tag = str(trt_tag)
        self.trt_platform_tag = str(trt_platform_tag)
        self.trt_feature_engine_path = str(trt_feature_engine_path)
        self.trt_post_engine_path = str(trt_post_engine_path)
        self.runtime_backend = "pytorch"
        self.trt_runner: Any = None
        self.trt_input_hw: tuple[int, int] | None = None
        self.model: Any = None

        # 定位 Fast-FoundationStereo 项目根目录并加入模块搜索路径。
        project_root = Path(__file__).resolve().parents[2]
        self.ffs_root = project_root / "Fast-FoundationStereo"
        if str(self.ffs_root) not in sys.path:
            sys.path.append(str(self.ffs_root))

        # 动态导入，避免路径型工程下的静态分析误报。
        core_utils = importlib.import_module("core.utils.utils")
        utils_mod = importlib.import_module("Utils")
        import torch

        self.torch = torch
        self.InputPadder = core_utils.InputPadder
        self.AMP_DTYPE = utils_mod.AMP_DTYPE
        self.set_logging_format = utils_mod.set_logging_format
        self.set_seed = utils_mod.set_seed

        # 解析模型目录：
        # - 支持传 pth 文件路径；
        # - 支持传权重目录；
        # - 支持仅传目录名（自动到 Fast-FoundationStereo/weights 下查找）。
        self.model_root_dir, self.model_pth_path = self._resolve_model_paths(model_dir)
        self.model_dir = str(self.model_pth_path or self.model_root_dir)

        # 后端实例：
        # - _PyTorchStereoBackend: 纯 PyTorch 推理路径
        # - _TrtStereoBackend: TRT engine 选择与推理路径
        self._pt_backend = _PyTorchStereoBackend(self)
        self._trt_backend = _TrtStereoBackend(self)

        self.set_logging_format(level=logging.INFO)
        self.torch.autograd.set_grad_enabled(False)

        # 指定 CUDA 但不可用时自动回退 CPU。
        runtime_device = str(device)
        if runtime_device == "cuda" and not self.torch.cuda.is_available():
            logging.warning("CUDA 不可用，自动回退到 CPU。")
            runtime_device = "cpu"
        self.device = self.torch.device(runtime_device)

        # TRT 依赖 CUDA；若当前设备不是 CUDA，则直接切回 PyTorch。
        if self.use_trt and self.device.type != "cuda":
            if self.trt_strict:
                raise RuntimeError("TRT 仅支持 CUDA 设备，但当前 device 不是 cuda。")
            logging.warning(
                "当前 device=%s，TRT 不可用，自动回退到 PyTorch。", self.device
            )
            self.use_trt = False

        # 速度优先与确定性模式可切换。
        if self.seed >= 0:
            self.set_seed(self.seed)
            logging.info("使用确定性模式: seed=%d", self.seed)
        else:
            self.torch.backends.cudnn.deterministic = False
            if self.device.type == "cuda":
                self.torch.backends.cudnn.benchmark = bool(self.cudnn_benchmark)
            logging.info(
                "使用速度优先模式: cudnn.benchmark=%s",
                str(self.torch.backends.cudnn.benchmark),
            )

        # TRT 运行时依赖按需导入，避免在纯 PyTorch 路径引入无关依赖。
        if self.use_trt:
            try:
                core_mod = importlib.import_module("core.foundation_stereo")
                omegaconf_mod = importlib.import_module("omegaconf")
                self.TrtRunner = core_mod.TrtRunner
                self.OmegaConf = omegaconf_mod.OmegaConf
            except Exception as exc:
                if self.trt_strict:
                    raise RuntimeError("TRT 依赖导入失败。") from exc
                logging.warning("TRT 依赖导入失败，自动回退 PyTorch。错误: %s", exc)
                self.use_trt = False

        # PyTorch 路径下使用 Triton 体构建时，提前检查可用性。
        if not self.use_trt:
            self._pt_backend.prepare_optimize_build_volume()
            self._pt_backend.load_model()
        else:
            self.runtime_backend = "trt"
            logging.info("TRT 模式已启用，首次推理将按参数自动匹配 engine 文件。")

        if hasattr(self.torch, "set_float32_matmul_precision"):
            self.torch.set_float32_matmul_precision("high")

        if self.device.type == "cuda":
            self.torch.backends.cuda.matmul.allow_tf32 = True
            self.torch.backends.cudnn.allow_tf32 = True

    def _resolve_model_paths(self, model_dir: str) -> tuple[Path, Path | None]:
        raw = Path(str(model_dir))
        candidates: list[Path] = []

        if raw.is_absolute():
            candidates.append(raw)
        else:
            candidates.append((Path.cwd() / raw).resolve())
            if self.ffs_root is not None:
                candidates.append((self.ffs_root / "weights" / raw).resolve())

        target: Path | None = None
        for cand in candidates:
            if cand.exists():
                target = cand
                break

        if target is None:
            checked = ", ".join(str(c) for c in candidates)
            raise FileNotFoundError(f"未找到模型路径: {model_dir}，已检查: {checked}")

        if target.is_file():
            if target.suffix.lower() not in {".pth", ".pt"}:
                raise FileNotFoundError(f"文件存在但不是 pth/pt: {target}")
            return target.parent, target

        # 目录模式：优先约定文件名，其次回退到目录内第一个 pth。
        pth = target / "model_best_bp2_serialize.pth"
        if pth.is_file():
            return target, pth

        pth_candidates = sorted(target.glob("*.pth"))
        return target, (pth_candidates[0] if pth_candidates else None)

    def _predict_disparity(self, left_t: Any, right_t: Any) -> Any:
        """统一的后端路由入口。

        设计说明：
        1. 外部只感知 `predict_depth`，不需要知道具体使用 TRT 还是 PyTorch。
        2. 默认优先 TRT；若 TRT 不可用（缺文件/初始化失败/非 CUDA），
           会在 `_fallback_to_pytorch` 中切换到 PyTorch。
        3. 该函数只负责“选路由”，具体推理细节由后端类各自处理。
        """
        if self.use_trt:
            return self._trt_backend.predict_disparity(left_t, right_t)
        return self._pt_backend.predict_disparity(left_t, right_t)

    def _fallback_to_pytorch(self, reason: str, exc: Exception | None = None) -> None:
        # 回退策略：
        # - `trt_strict=True`：直接抛错，强制暴露 TRT 问题；
        # - `trt_strict=False`：记录告警并切换到 PyTorch，保证链路可运行。
        message = f"TRT 不可用: {reason}"
        if self.trt_strict:
            if exc is not None:
                raise RuntimeError(message) from exc
            raise RuntimeError(message)

        if exc is None:
            logging.warning("%s，自动回退 PyTorch。", message)
        else:
            logging.warning("%s，自动回退 PyTorch。错误: %s", message, exc)

        self.use_trt = False
        self.trt_runner = None
        self.trt_input_hw = None
        if self._pt_backend is not None:
            self._pt_backend.prepare_optimize_build_volume()
            self._pt_backend.load_model()

    def predict_depth(
        self,
        left_image: np.ndarray,
        right_image: np.ndarray,
        fx: float,
        baseline: float,
        return_timing: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, float]]:
        """
        单帧预测深度（米）。

        输入：
        - left_image/right_image: 可为灰度(H,W)或彩色(H,W,C)。
        - fx: 左相机焦距（像素）。
        - baseline: 双目基线（米）。

        输出：
        - depth_meter: 深度图（米，shape=(H,W)）
        """
        t0 = time.perf_counter()

        # 统一输入到 3 通道：
        # - 灰度图复制成 3 通道
        # - 彩色图保留前 3 通道
        left = left_image
        right = right_image
        if left.ndim == 2:
            left = np.repeat(left[..., None], 3, axis=2)
        else:
            left = left[..., :3]
        if right.ndim == 2:
            right = np.repeat(right[..., None], 3, axis=2)
        else:
            right = right[..., :3]

        # 缩放用于加速，右图跟随左图尺寸。
        if self.scale != 1.0:
            left = cv2.resize(
                left,
                dsize=None,
                fx=float(self.scale),
                fy=float(self.scale),
                interpolation=cv2.INTER_LINEAR,
            )
            right = cv2.resize(
                right,
                dsize=(left.shape[1], left.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        # Numpy -> Tensor，转 NCHW。
        left_t = (
            self.torch.as_tensor(left).to(self.device).float()[None].permute(0, 3, 1, 2)
        )
        right_t = (
            self.torch.as_tensor(right)
            .to(self.device)
            .float()[None]
            .permute(0, 3, 1, 2)
        )
        t1 = time.perf_counter()

        # 前向推理：
        # - 默认尝试 TRT；
        # - TRT 不可用时自动回退到 PyTorch；
        # - 对上层接口保持同一返回格式（disparity tensor）。
        t_forward_begin = time.perf_counter()
        disp = self._predict_disparity(left_t, right_t)
        t2 = time.perf_counter()

        # 恢复尺寸并转到 numpy。
        disp = disp.squeeze(0).squeeze(0).detach().cpu().numpy()
        disp = np.clip(disp, 1e-6, None)

        # 视差转深度：depth = fx * baseline / disp。
        fx_scaled = float(fx) * float(self.scale)
        depth_meter = (fx_scaled * float(baseline)) / disp

        # 若做过缩放，恢复到原图大小，便于和硬件深度对齐比较。
        if self.scale != 1.0:
            depth_meter = cv2.resize(
                depth_meter,
                dsize=(left_image.shape[1], left_image.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        depth_meter[~np.isfinite(depth_meter)] = 0
        t3 = time.perf_counter()

        depth_meter = depth_meter.astype(np.float32)
        if not return_timing:
            return depth_meter

        timing = {
            "prep_ms": (t1 - t0) * 1000.0,
            "forward_ms": (t2 - t_forward_begin) * 1000.0,
            "post_ms": (t3 - t2) * 1000.0,
            "infer_ms": (t3 - t0) * 1000.0,
        }
        return depth_meter, timing


if __name__ == "__main__":
    """
    最小实时示例：
    1. 读取 RealSense 双目与深度。
    2. 运行 Fast-FoundationStereo 预测深度。
    3. 显示预测深度与硬件深度对比（两个窗口）。
    """

    # ===== 直接配置参数（按你的要求，不使用 argparse） =====
    model_dir = str(
        Path(__file__).resolve().parents[2]
        / "Fast-FoundationStereo"
        / "weights"
        / "23-36-37"
    )
    width = 640
    height = 480
    fps = 30
    device = "cuda"
    scale = 1.0
    valid_iters = 4
    max_disp = 192
    optimize_build_volume = "triton"
    use_trt = False
    trt_precision = "fp16"
    trt_strict = False
    trt_tag = ""  # 为空时根据输入尺寸 + valid_iters + max_disp 自动匹配。
    seed = -1  # <0 速度优先；>=0 确定性模式。
    cudnn_benchmark = True
    warmup_frames = 15
    compare_min_depth = 0.1
    compare_max_depth = 10.0
    error_max = 1.0
    vis_min_depth = 0.1
    vis_max_depth = 10.0
    # ================================================

    # 导入并启动 RealSense 相机。
    from realsense import RealSenseCamera

    camera = RealSenseCamera(width=width, height=height, fps=fps)
    camera.start()

    # 初始化双目推理器。
    estimator = FastFoundationStereoRealtime(
        model_dir=model_dir,
        device=device,
        scale=scale,
        valid_iters=valid_iters,
        max_disp=max_disp,
        optimize_build_volume=optimize_build_volume,
        use_trt=use_trt,
        trt_precision=trt_precision,
        trt_strict=trt_strict,
        trt_tag=trt_tag,
        seed=seed,
        cudnn_benchmark=cudnn_benchmark,
    )

    # 在循环开始前一次性读取标定参数，避免循环内判断逻辑。
    calib = camera.get_stereo_calibration()
    fx = float(calib.fx)
    baseline = float(calib.baseline_m)
    depth_scale = float(calib.depth_scale)

    logging.info(
        "推理参数: model=%s, backend=%s, optimize_build_volume=%s, valid_iters=%d, scale=%.2f",
        model_dir,
        estimator.runtime_backend,
        estimator.optimize_build_volume,
        valid_iters,
        scale,
    )

    try:
        window_name = "FoundationStereo vs RealSense"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        print("窗口已打开，按 q 或 ESC 退出。")

        frame_idx = 0
        ema_loop_fps = 0.0
        ema_infer_fps = 0.0

        while True:
            loop_t0 = time.perf_counter()

            # 取双目图（用于模型输入）与对齐 RGBD（用于硬件深度对比）。
            stereo = camera.get_stereo_frames()
            rgbd = camera.get_aligned_rgbd_frames()
            capture_ms = (time.perf_counter() - loop_t0) * 1000.0

            left_gray = stereo.left
            right_gray = stereo.right

            pred_depth_meter, infer_timing = cast(
                tuple[np.ndarray, dict[str, float]],
                estimator.predict_depth(
                    left_image=left_gray,
                    right_image=right_gray,
                    fx=fx,
                    baseline=baseline,
                    return_timing=True,
                ),
            )

            # 把硬件原始 z16 深度转成米。
            native_depth_meter = rgbd.depth.astype(np.float32) * depth_scale

            valid_mask = (
                (pred_depth_meter > compare_min_depth)
                & (pred_depth_meter < compare_max_depth)
                & (native_depth_meter > compare_min_depth)
                & (native_depth_meter < compare_max_depth)
            )

            if valid_mask.any():
                diff = pred_depth_meter - native_depth_meter
                mae = float(np.mean(np.abs(diff[valid_mask])))
                rmse = float(np.sqrt(np.mean((diff[valid_mask]) ** 2)))
            else:
                diff = pred_depth_meter - native_depth_meter
                mae = float("nan")
                rmse = float("nan")

            # 把米制深度转成伪彩色显示。
            pred_invalid = (pred_depth_meter <= vis_min_depth) | (
                pred_depth_meter >= vis_max_depth
            )
            native_invalid = (native_depth_meter <= vis_min_depth) | (
                native_depth_meter >= vis_max_depth
            )
            error_map = np.abs(diff)
            err_invalid = ~valid_mask

            left_vis = cv2.cvtColor(left_gray, cv2.COLOR_GRAY2BGR)
            pred_vis = _colorize_depth(
                pred_depth_meter,
                vis_min_depth,
                vis_max_depth,
                pred_invalid,
            )
            native_vis = _colorize_depth(
                native_depth_meter,
                vis_min_depth,
                vis_max_depth,
                native_invalid,
            )
            err_vis = _colorize_error(error_map, error_max, err_invalid)

            top = np.concatenate([left_vis, pred_vis], axis=1)
            bottom = np.concatenate([native_vis, err_vis], axis=1)
            canvas = np.concatenate([top, bottom], axis=0)

            h, w = left_vis.shape[:2]
            _draw_label(canvas, "Left IR", 12, 28)
            _draw_label(canvas, "FoundationStereo Depth (m)", w + 12, 28)
            _draw_label(canvas, "RealSense Native Depth (m)", 12, h + 28)
            _draw_label(canvas, "Abs Error (m)", w + 12, h + 28)

            loop_ms = (time.perf_counter() - loop_t0) * 1000.0
            loop_fps = 1000.0 / max(loop_ms, 1e-6)
            infer_fps = 1000.0 / max(infer_timing["infer_ms"], 1e-6)

            if frame_idx >= warmup_frames:
                if ema_loop_fps == 0.0:
                    ema_loop_fps = loop_fps
                    ema_infer_fps = infer_fps
                else:
                    ema_loop_fps = 0.9 * ema_loop_fps + 0.1 * loop_fps
                    ema_infer_fps = 0.9 * ema_infer_fps + 0.1 * infer_fps

            stats = (
                f"FPS(loop): {loop_fps:.1f} | FPS(loop_ema): {ema_loop_fps:.1f} | "
                f"FPS(infer): {infer_fps:.1f} | FPS(infer_ema): {ema_infer_fps:.1f} | "
                f"MAE: {mae:.3f}m | RMSE: {rmse:.3f}m"
            )
            timing_stats = (
                f"capture:{capture_ms:.1f}ms | prep:{infer_timing['prep_ms']:.1f}ms | "
                f"forward:{infer_timing['forward_ms']:.1f}ms | post:{infer_timing['post_ms']:.1f}ms | "
                f"infer_total:{infer_timing['infer_ms']:.1f}ms"
            )
            _draw_label(canvas, stats, 12, canvas.shape[0] - 42)
            _draw_label(canvas, timing_stats, 12, canvas.shape[0] - 14)
            if frame_idx < warmup_frames:
                _draw_label(canvas, f"Warmup: {frame_idx + 1}/{warmup_frames}", 12, 54)

            cv2.imshow(window_name, canvas)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

            frame_idx += 1
    finally:
        camera.stop()
        cv2.destroyAllWindows()
