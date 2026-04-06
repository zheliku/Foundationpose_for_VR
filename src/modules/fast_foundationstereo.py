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


class FastFoundationStereoRealtime:
    """实时双目深度估计器。"""

    # 输入配置。
    model_dir: str = ""  # 模型权重路径。
    scale: float = 1.0  # 推理缩放比例。
    valid_iters: int = 4  # 网络迭代次数。
    max_disp: int = 192  # 最大视差。
    optimize_build_volume: str = "triton"  # 体构建优化后端。

    # 路径与依赖对象。
    ffs_root: Path | None = None  # Fast-FoundationStereo 根目录。
    torch: Any = None  # torch 模块引用。
    InputPadder: Any = None  # 网络输入 padding 工具。
    AMP_DTYPE: Any = None  # 混合精度 dtype。
    set_logging_format: Any = None  # 日志配置函数。
    set_seed: Any = None  # 随机种子函数。

    # 运行时状态对象。
    device: Any = None  # 当前推理设备。
    model: Any = None  # 已加载的立体深度模型。

    def __init__(
        self,
        model_dir: str,
        device: str = "cuda",
        scale: float = 1.0,
        valid_iters: int = 4,
        max_disp: int = 192,
        optimize_build_volume: str = "triton",
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

        初始化流程：
        1. 保存推理参数。
        2. 配置工程路径并动态导入依赖。
        3. 设置日志、随机种子与设备。
        4. 加载模型并写入关键推理参数。
        """
        self.model_dir = str(model_dir)
        self.scale = float(scale)
        self.valid_iters = int(valid_iters)
        self.max_disp = int(max_disp)
        self.optimize_build_volume = str(optimize_build_volume)

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

        self.set_logging_format(level=logging.INFO)
        self.set_seed(0)
        self.torch.autograd.set_grad_enabled(False)

        # 指定 CUDA 但不可用时自动回退 CPU。
        runtime_device = str(device)
        if runtime_device == "cuda" and not self.torch.cuda.is_available():
            logging.warning("CUDA 不可用，自动回退到 CPU。")
            runtime_device = "cpu"
        self.device = self.torch.device(runtime_device)

        # 加载模型并写入关键推理参数。
        self.model = self.torch.load(
            self.model_dir,
            map_location="cpu",
            weights_only=False,
        )
        self.model.args.valid_iters = int(self.valid_iters)
        self.model.args.max_disp = int(self.max_disp)
        self.model = self.model.to(self.device).eval()

        if hasattr(self.torch, "set_float32_matmul_precision"):
            self.torch.set_float32_matmul_precision("high")

    def predict_depth(
        self,
        left_image: np.ndarray,
        right_image: np.ndarray,
        fx: float,
        baseline: float,
    ) -> np.ndarray:
        """
        单帧预测深度（米）。

        输入：
        - left_image/right_image: 可为灰度(H,W)或彩色(H,W,C)。
        - fx: 左相机焦距（像素）。
        - baseline: 双目基线（米）。

        输出：
        - depth_meter: 深度图（米，shape=(H,W)）
        """
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

        # 按网络约束进行 padding。
        padder = self.InputPadder(left_t.shape, divis_by=32, force_square=False)
        left_t, right_t = padder.pad(left_t, right_t)

        # 前向推理。
        if self.device.type == "cuda":
            # 优先使用新接口 torch.autocast；旧版本再回退到 cuda.amp.autocast。
            if hasattr(self.torch, "autocast"):
                autocast_ctx = self.torch.autocast(
                    "cuda",
                    enabled=True,
                    dtype=self.AMP_DTYPE,
                )
            else:
                autocast_ctx = self.torch.cuda.amp.autocast(
                    enabled=True,
                    dtype=self.AMP_DTYPE,
                )
        else:
            autocast_ctx = nullcontext()
        with self.torch.inference_mode():
            with autocast_ctx:
                disp = self.model.forward(
                    left_t,
                    right_t,
                    iters=int(self.valid_iters),
                    test_mode=True,
                    optimize_build_volume=str(self.optimize_build_volume),
                )

        # 恢复尺寸并转到 numpy。
        disp = padder.unpad(disp.float()).squeeze(0).squeeze(0).detach().cpu().numpy()
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
        return depth_meter.astype(np.float32)


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
        / "20-30-48"
        / "model_best_bp2_serialize.pth"
    )
    width = 640
    height = 480
    fps = 30
    device = "cuda"
    scale = 1.0
    valid_iters = 4
    max_disp = 192
    optimize_build_volume = "triton"
    vis_min_depth = 0.1
    vis_max_depth = 8.0
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
    )

    # 在循环开始前一次性读取标定参数，避免循环内判断逻辑。
    if camera.pipeline is None:
        raise RuntimeError("RealSense pipeline 不可用。")

    profile = camera.pipeline.get_active_profile()
    import pyrealsense2 as rs

    rs_any = cast(Any, rs)
    left_video = profile.get_stream(rs_any.stream.infrared, 1).as_video_stream_profile()
    right_video = profile.get_stream(
        rs_any.stream.infrared, 2
    ).as_video_stream_profile()

    intr = left_video.get_intrinsics()
    extr = left_video.get_extrinsics_to(right_video)
    depth_sensor = profile.get_device().first_depth_sensor()

    fx = float(intr.fx)
    baseline = abs(float(extr.translation[0]))
    depth_scale = float(depth_sensor.get_depth_scale())

    if baseline <= 0:
        raise RuntimeError("RealSense baseline 无效，无法进行深度反算。")

    try:
        cv2.namedWindow("Stereo IR", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("Pred Depth (FoundationStereo)", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("Native Depth (RealSense)", cv2.WINDOW_AUTOSIZE)
        print("窗口已打开，按 q 或 ESC 退出。")

        while True:
            # 取双目图（用于模型输入）与对齐 RGBD（用于硬件深度对比）。
            stereo = camera.get_stereo_frames()
            rgbd = camera.get_aligned_rgbd_frames()

            # 简单显示左右双目图（拼接后单窗口）。
            stereo_vis = np.hstack([stereo.left, stereo.right])
            cv2.imshow("Stereo IR", stereo_vis)

            # 运行实时深度估计。
            infer_t0 = time.perf_counter()
            pred_depth_meter = estimator.predict_depth(
                left_image=stereo.left,
                right_image=stereo.right,
                fx=fx,
                baseline=baseline,
            )
            infer_ms = (time.perf_counter() - infer_t0) * 1000.0

            # 把硬件原始 z16 深度转成米。
            native_depth_meter = rgbd.depth.astype(np.float32) * depth_scale

            # 把米制深度转成伪彩色显示。
            pred_norm = (
                (pred_depth_meter - vis_min_depth)
                / max(vis_max_depth - vis_min_depth, 1e-6)
            ).clip(0.0, 1.0)
            native_norm = (
                (native_depth_meter - vis_min_depth)
                / max(vis_max_depth - vis_min_depth, 1e-6)
            ).clip(0.0, 1.0)

            pred_u8 = (pred_norm * 255.0).astype(np.uint8)
            native_u8 = (native_norm * 255.0).astype(np.uint8)

            pred_vis = cv2.applyColorMap(pred_u8, cv2.COLORMAP_TURBO)
            native_vis = cv2.applyColorMap(native_u8, cv2.COLORMAP_TURBO)

            # 显示简单状态文本。
            cv2.putText(
                pred_vis,
                f"infer: {infer_ms:.1f} ms",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Pred Depth (FoundationStereo)", pred_vis)
            cv2.imshow("Native Depth (RealSense)", native_vis)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        camera.stop()
        cv2.destroyAllWindows()
