"""
Cutie 2D Tracker API（模块化版）

设计目标：
1. 输入明确：当前图像 + 初始化信息（mask 或 bbox）。
2. 输出明确：2D 跟踪框 bbox 与当前 mask。
3. 与 foundationpose.py 解耦：只做 2D 跟踪，不包含 6D 逻辑。
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
import torch
from torchvision.transforms.functional import to_tensor


@dataclass
class CutieTrackResult:
    """单帧跟踪输出。"""

    # 跟踪框 [x, y, w, h]。
    bbox_xywh: list[int]

    # 当前帧目标 mask（二值，0/1）。
    mask: np.ndarray


class Tracker2D:
    """2D tracker 抽象接口（空实现占位）。"""

    def initialize(
        self,
        frame: np.ndarray,
        init_mask: np.ndarray | None = None,
        init_bbox: list[int] | None = None,
    ) -> CutieTrackResult:
        return CutieTrackResult(
            bbox_xywh=[-1, -1, 0, 0], mask=np.zeros(frame.shape[:2], dtype=np.uint8)
        )

    def track(self, frame: np.ndarray) -> CutieTrackResult:
        return CutieTrackResult(
            bbox_xywh=[-1, -1, 0, 0], mask=np.zeros(frame.shape[:2], dtype=np.uint8)
        )


class CutieTracker(Tracker2D):
    """Cutie 的简洁封装。"""

    # 输入配置。
    seg_threshold: float = 0.1  # 分割阈值。
    erosion_size: int = 5  # mask 腐蚀核大小（像素）。

    # 运行时对象。
    device: str = "cpu"  # 推理设备标识（cuda/cpu）。
    model: Any = None  # Cutie 模型对象。
    processor: Any = None  # Cutie 推理处理器（InferenceCore）。

    def __init__(
        self,
        seg_threshold: float = 0.1,
        erosion_size: int = 5,
    ) -> None:
        """
        初始化 Cutie 跟踪器。

        参数：
        - seg_threshold: 分割阈值。
        - erosion_size: mask 后处理腐蚀核大小。

        初始化流程：
        1. 保存基础参数。
        2. 选择运行设备。
        3. 配置 Cutie 包导入路径。
        4. 创建模型与 InferenceCore 处理器。
        """
        super().__init__()
        self.seg_threshold = float(seg_threshold)
        self.erosion_size = int(erosion_size)

        # 自动选择设备。
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 简洁做法：显式从项目目录 Cutie/cutie 导入。
        # 这样不会再和当前文件名 cutie.py 发生同名冲突。
        project_root = Path(__file__).resolve().parents[2]
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        # 兼容 Cutie 内部大量 `from cutie.xxx import ...` 的绝对导入写法：
        # 先导入真正包 Cutie.cutie，再注册为顶级别名 cutie。
        cutie_pkg = importlib.import_module("Cutie.cutie")
        sys.modules["cutie"] = cutie_pkg

        # 延迟导入 Cutie，避免无关脚本因依赖问题失败。
        from cutie.inference.inference_core import InferenceCore
        from cutie.utils.get_default_model import get_default_model

        self.model = get_default_model()
        if hasattr(self.model, "to"):
            self.model = self.model.to(self.device)

        self.processor = InferenceCore(self.model, cfg=self.model.cfg)
        self.processor.max_internal_size = -1

    @staticmethod
    def _ensure_rgb(frame: np.ndarray) -> np.ndarray:
        """统一输入图像为 RGB 3 通道格式。"""
        if frame.ndim == 2:
            frame = np.repeat(frame[..., None], 3, axis=2)
        elif frame.ndim == 3:
            frame = frame[..., :3]
        else:
            raise ValueError("frame 维度不正确，应为 (H,W) 或 (H,W,C)。")
        return frame

    @staticmethod
    def _mask_from_bbox(h: int, w: int, bbox_xywh: list[int]) -> np.ndarray:
        """由 bbox 生成初始化 mask。"""
        x, y, bw, bh = [int(v) for v in bbox_xywh]
        mask = np.zeros((h, w), dtype=np.uint8)

        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(w, x + max(0, bw))
        y1 = min(h, y + max(0, bh))
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 1
        return mask

    def _bbox_from_mask(self, mask: np.ndarray) -> list[int]:
        """由 mask 提取 bbox。"""
        kernel = np.ones((int(self.erosion_size), int(self.erosion_size)), np.uint8)
        mask_eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)

        rows = np.any(mask_eroded, axis=1)
        cols = np.any(mask_eroded, axis=0)
        if np.any(rows) and np.any(cols):
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]
            return [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]
        return [-1, -1, 0, 0]

    def initialize(
        self,
        frame: np.ndarray,
        init_mask: np.ndarray | None = None,
        init_bbox: list[int] | None = None,
    ) -> CutieTrackResult:
        """
        初始化跟踪器。

        输入：
        - frame: 初始帧图像。
        - init_mask: 初始目标 mask（二值）。
        - init_bbox: 初始目标框 [x,y,w,h]。

        说明：
        - init_mask 与 init_bbox 至少提供一个。
        - 若两者都提供，优先使用 init_mask。
        """
        frame = self._ensure_rgb(frame)

        if init_mask is None and init_bbox is None:
            raise ValueError("initialize 需要 init_mask 或 init_bbox 至少一个。")

        if init_mask is None:
            init_mask = self._mask_from_bbox(
                frame.shape[0], frame.shape[1], init_bbox or [-1, -1, 0, 0]
            )
        else:
            init_mask = (init_mask > 0).astype(np.uint8)

        with torch.no_grad():
            frame_t = to_tensor(frame).to(self.device).float()
            mask_t = torch.from_numpy(init_mask).to(self.device)

            # Cutie 用非零实例 id 表示目标，0 是背景。
            objects = np.unique(init_mask)
            objects = objects[objects != 0].tolist()

            prob = self.processor.step(frame_t, mask_t, objects=objects)
            out_mask_t = self.processor.output_prob_to_mask(prob)
            out_mask = out_mask_t.detach().cpu().numpy().astype(np.uint8)

        bbox_xywh = self._bbox_from_mask(out_mask)
        torch.cuda.empty_cache()
        return CutieTrackResult(bbox_xywh=bbox_xywh, mask=out_mask)

    def track(self, frame: np.ndarray) -> CutieTrackResult:
        """
        跟踪后续帧。

        输入：
        - frame: 当前帧。

        输出：
        - bbox_xywh: 跟踪框。
        - mask: 当前目标 mask。
        """
        frame = self._ensure_rgb(frame)

        with torch.no_grad():
            frame_t = to_tensor(frame).to(self.device).float()
            prob = self.processor.step(frame_t)
            out_mask_t = self.processor.output_prob_to_mask(prob)
            out_mask = out_mask_t.detach().cpu().numpy().astype(np.uint8)

        bbox_xywh = self._bbox_from_mask(out_mask)
        torch.cuda.empty_cache()
        return CutieTrackResult(bbox_xywh=bbox_xywh, mask=out_mask)

    def reset(self) -> None:
        """清理 Cutie 时序 memory，避免重新 register 后旧目标状态污染新跟踪。"""
        from cutie.inference.inference_core import InferenceCore

        self.processor = InferenceCore(self.model, cfg=self.model.cfg)
        self.processor.max_internal_size = -1


if __name__ == "__main__":
    """
    最小实时示例：
    1) 用 RealSense 彩色流作为输入。
    2) 用固定初始框初始化 Cutie。
    3) 实时输出 mask 与 bbox。
    """

    from realsense import RealSenseCamera

    # 直接配置（不使用 argparse）。
    width, height, fps = 640, 480, 30
    init_bbox = [220, 140, 180, 180]  # [x,y,w,h]，请按你的目标手动调整

    camera = RealSenseCamera(width=width, height=height, fps=fps)
    camera.start()

    tracker = CutieTracker(seg_threshold=0.1, erosion_size=5)

    cv2.namedWindow("Cutie Frame", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Cutie Mask", cv2.WINDOW_AUTOSIZE)

    try:
        # 用第一帧做初始化。
        first_rgbd = camera.get_aligned_rgbd_frames()
        init_res = tracker.initialize(first_rgbd.color_bgr, init_bbox=init_bbox)

        print("Cutie tracker initialized, press q/ESC to quit.")

        while True:
            rgbd = camera.get_aligned_rgbd_frames()
            res = tracker.track(rgbd.color_bgr)

            vis = rgbd.color_bgr.copy()
            x, y, w, h = res.bbox_xywh
            if w > 0 and h > 0:
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 255), 2)

            # mask 显示为黑白图（白=目标）。
            mask_vis = (res.mask > 0).astype(np.uint8) * 255

            cv2.imshow("Cutie Frame", vis)
            cv2.imshow("Cutie Mask", mask_vis)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        camera.stop()
        cv2.destroyAllWindows()
