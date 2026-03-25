from __future__ import annotations

# 这是一个“最简单”的单图示例：
# 1) 不使用命令行参数
# 2) 参数全部写在代码顶部
# 3) 只保留核心：检测 + 分割 + 黑白mask

from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLOE

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent

# ===== 直接改这里 =====
IMAGE_PATH = Path(__file__).parent / "img/00000.png"  # 输入图片
MODEL_PATH = PROJECT_DIR / "checkpoints/yoloe-26l-seg.pt"  # 模型权重
OUTPUT_DIR = Path(__file__).parent / "results/single_image_basic"  # 输出目录

PROMPT = ["white block"]  # 文本提示词
CONF = 0.15  # 置信度阈值
IMGSZ = 640  # 推理尺寸
MAX_DET = 2  # 最多目标数
MASK_THRESHOLD = 0.5  # mask 二值化阈值
SAVE_OVERLAY = True  # 是否保存叠加图
SAVE_MASK = True  # 是否保存黑白mask
USE_HALF = False  # YOLOE-seg 建议 False


def choose_device() -> str | int:
    """有 GPU 就用 GPU(0)，否则用 CPU。"""
    return 0 if torch.cuda.is_available() else "cpu"


def build_merged_binary_mask(result, threshold: float) -> np.ndarray | None:
    """把分割结果转成一张黑白 mask（白=目标，黑=背景）。"""
    if result.masks is None or result.masks.data is None or len(result.masks.data) == 0:
        return None
    masks = result.masks.data.detach().cpu().numpy()
    binary_masks = (masks >= threshold).astype(np.uint8) * 255
    return np.max(binary_masks, axis=0)


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"模型文件不存在: {MODEL_PATH}")
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"图片文件不存在: {IMAGE_PATH}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = choose_device()
    model = YOLOE(str(MODEL_PATH))
    model.set_classes(PROMPT)
    model.fuse()

    result = model.predict(
        source=IMAGE_PATH,
        conf=CONF,
        imgsz=IMGSZ,
        max_det=MAX_DET,
        device=device,
        half=USE_HALF,
        save=False,
        verbose=False,
    )[0]

    overlay = result.plot()
    merged_mask = build_merged_binary_mask(result, MASK_THRESHOLD)

    if SAVE_OVERLAY:
        cv2.imwrite(str(OUTPUT_DIR / "overlay.png"), overlay)
    cv2.imshow("YOLOE Overlay", overlay)

    if merged_mask is not None:
        if SAVE_MASK:
            cv2.imwrite(str(OUTPUT_DIR / "mask_bw.png"), merged_mask)
        cv2.imshow("YOLOE Mask BW", merged_mask)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
