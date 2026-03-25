from __future__ import annotations

# 这个脚本是“最简实时版”示例：
# 1) 不使用命令行参数，所有配置都写在顶部
# 2) 从 RealSense 相机实时读取彩色图像
# 3) 用 YOLOE 做检测/分割
# 4) 在画面上叠加关键信息（FPS、推理耗时、检测数量、设备名）

from pathlib import Path
import time
from typing import Any, cast

import cv2
import numpy as np
import pyrealsense2 as rs
import torch
from ultralytics import YOLOE

rs_any = cast(Any, rs)

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent

# ===== 直接改这里 =====
MODEL_PATH = PROJECT_DIR / "checkpoints/yoloe-26l-seg.pt"  # 模型权重
PROMPT = ["white block"]  # 文本提示词
CONF = 0.15  # 置信度阈值
IMGSZ = 640  # 推理尺寸
MAX_DET = 2  # 最多目标数
MASK_THRESHOLD = 0.5  # 分割 mask 二值化阈值
USE_HALF = False  # YOLOE-seg 建议 False

# RealSense 相机参数（彩色流）
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# 可视化开关
SHOW_MASK_WINDOW = True  # 是否单独显示黑白 mask 窗口
WINDOW_NAME_OVERLAY = "YOLOE RealSense Overlay"
WINDOW_NAME_MASK = "YOLOE RealSense Mask BW"


def choose_device() -> str | int:
    """有 GPU 就用 GPU(0)，否则用 CPU。"""
    return 0 if torch.cuda.is_available() else "cpu"


def build_merged_binary_mask(result, threshold: float) -> np.ndarray | None:
    """把分割结果合并成一张黑白 mask（白=目标，黑=背景）。"""
    if result.masks is None or result.masks.data is None or len(result.masks.data) == 0:
        return None

    # result.masks.data 形状通常是 [N, H, W]，N 是检测到的实例数量
    masks = result.masks.data.detach().cpu().numpy()
    binary_masks = (masks >= threshold).astype(np.uint8) * 255

    # 多个目标的 mask 取逐像素最大值，合成一张总 mask
    return np.max(binary_masks, axis=0)


def put_info_text(
    image: np.ndarray,
    lines: list[str],
    x: int = 10,
    y: int = 25,
    line_height: int = 24,
) -> None:
    """在图像左上角逐行写信息文本。"""
    for i, line in enumerate(lines):
        yy = y + i * line_height

        # 先画黑色描边，提升可读性
        cv2.putText(
            image,
            line,
            (x, yy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )

        # 再画亮色正文
        cv2.putText(
            image,
            line,
            (x, yy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (50, 255, 50),
            1,
            cv2.LINE_AA,
        )


def main() -> None:
    # ---------- 1) 检查模型 ----------
    base_dir = Path(__file__).resolve().parent
    model_path = (base_dir / MODEL_PATH).resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    # ---------- 2) 初始化 YOLOE ----------
    device = choose_device()
    model = YOLOE(str(model_path))
    model.set_classes(PROMPT)
    model.fuse()

    # ---------- 3) 初始化 RealSense ----------
    rs_pipeline = rs_any.pipeline()
    rs_config = rs_any.config()

    # 只开彩色流，最简够用
    rs_config.enable_stream(
        rs_any.stream.color,
        CAMERA_WIDTH,
        CAMERA_HEIGHT,
        rs_any.format.bgr8,
        CAMERA_FPS,
    )

    profile = rs_pipeline.start(rs_config)
    device_name = profile.get_device().get_info(rs_any.camera_info.name)

    # ---------- 4) 主循环（实时推理） ----------
    # 使用指数滑动平均，让 FPS 显示更稳定
    fps_ema = 0.0
    alpha = 0.10  # 越大越灵敏，越小越平滑

    try:
        while True:
            loop_start = time.perf_counter()

            # 4.1 取一帧 RealSense 彩色图像
            frames = rs_pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if color_frame is None:
                continue

            frame_bgr = np.asanyarray(color_frame.get_data())

            # 4.2 做一次模型推理，并记录推理耗时
            infer_start = time.perf_counter()
            result = model.predict(
                source=frame_bgr,
                conf=CONF,
                imgsz=IMGSZ,
                max_det=MAX_DET,
                device=device,
                half=USE_HALF,
                save=False,
                verbose=False,
            )[0]
            infer_ms = (time.perf_counter() - infer_start) * 1000.0

            # 4.3 渲染叠加图（框、类别、mask）
            overlay = result.plot()

            # 检测数量（如果没有框，记为 0）
            det_count = 0
            if result.boxes is not None and result.boxes.data is not None:
                det_count = len(result.boxes.data)

            # 4.4 计算当前 FPS（按整轮循环时间）
            loop_sec = max(time.perf_counter() - loop_start, 1e-6)
            inst_fps = 1.0 / loop_sec
            fps_ema = (
                inst_fps if fps_ema == 0.0 else (1 - alpha) * fps_ema + alpha * inst_fps
            )

            # 4.5 叠加文本信息
            info_lines = [
                f"Device: {device_name}",
                f"Compute: {'GPU:0' if device == 0 else 'CPU'}",
                f"FPS: {fps_ema:.1f}",
                f"Infer: {infer_ms:.1f} ms",
                f"Detections: {det_count}",
                f"Frame: {frame_bgr.shape[1]}x{frame_bgr.shape[0]}",
                "Key: q / ESC to quit",
            ]
            put_info_text(overlay, info_lines)

            cv2.imshow(WINDOW_NAME_OVERLAY, overlay)

            # 4.6 可选显示黑白 mask
            if SHOW_MASK_WINDOW:
                merged_mask = build_merged_binary_mask(result, MASK_THRESHOLD)
                if merged_mask is None:
                    merged_mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
                cv2.imshow(WINDOW_NAME_MASK, merged_mask)

            # 4.7 按键退出
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

    finally:
        # ---------- 5) 资源释放 ----------
        rs_pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
