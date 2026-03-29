"""
YOLOE-26 实时掩码 API（精简工程版）

功能目标：
1. 实时接收图像。
2. 根据提示文本（prompt）执行检测/分割。
3. 输出目标黑白掩码（mask）。

输入（每帧）：
- image_bgr: numpy 图像，支持灰度(H,W)或彩色(H,W,C)。
- prompt: str 或 list[str]，例如 "white block"。

输出（每帧）：
- overlay: 叠加检测/分割可视化图。
- mask_bw: 黑白掩码（白=目标，黑=背景），uint8。
- det_count: 检测数量。
- infer_ms: 单帧推理耗时。

说明：
- 参考了 test_yoloe.py.py 与 test_yoloe_realsense.py 的核心流程。
- 保留关键可配置项，不做功能缩减。
"""

from __future__ import annotations

import time
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import torch
from ultralytics import YOLOE


@dataclass
class Yoloe26Config:
    """YOLOE-26 运行配置。"""

    # 模型权重路径。
    model_path: str

    # 推理阈值与尺寸。
    conf: float = 0.15
    imgsz: int = 640
    max_det: int = 2

    # mask 二值化阈值。
    mask_threshold: float = 0.5

    # 是否启用半精度。YOLOE-seg 一般建议 False。
    use_half: bool = False

    # 指定设备：0 表示 GPU0，"cpu" 表示 CPU。
    # 若设为 None，将自动选择（有 CUDA 则 0，否则 cpu）。
    device: str | int | None = None

    # 可选：本地 mobileclip2_b.ts 路径。
    # 作用：避免 Ultralytics 在 set_classes() 时联网下载文本编码器。
    mobileclip2_path: str | None = None


@dataclass
class Yoloe26Result:
    """单帧输出结果。"""

    # 叠加可视化图。
    overlay: np.ndarray

    # 合并后的黑白掩码（白=目标，黑=背景）。
    mask_bw: np.ndarray

    # 检测框数量。
    det_count: int

    # 推理耗时（毫秒）。
    infer_ms: float

    # 当前生效的提示词。
    prompt: list[str]


class Yoloe26Masker:
    """YOLOE-26 实时掩码生成器。"""

    def __init__(self, config: Yoloe26Config, init_prompt: str | list[str]) -> None:
        self.cfg = config

        # 校验模型路径，提前失败更易定位问题。
        model_path = Path(self.cfg.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        # 自动选择设备：有 CUDA 则 GPU0，否则 CPU。
        if self.cfg.device is None:
            self.device: str | int = 0 if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.cfg.device

        # 在 set_classes 前配置本地 mobileclip2 权重路径。
        self._configure_mobileclip2_path()

        # 加载模型并设置初始类别提示。
        self.model = YOLOE(str(model_path))
        self._prompt: list[str] = []
        self.set_prompt(init_prompt)

        # 融合层可提升推理速度。
        self.model.fuse()

    def _configure_mobileclip2_path(self) -> None:
        """
        配置 mobileclip2_b.ts 的本地路径。

        Ultralytics 在 YOLOE 文本提示阶段会通过固定文件名 `mobileclip2_b.ts`
        去当前权重目录/下载源查找文本编码器。这里提前把本地文件挂到可查找位置，
        避免 GitHub 403 或离线环境失败。
        """
        if not self.cfg.mobileclip2_path:
            return

        src = Path(self.cfg.mobileclip2_path).expanduser().resolve()
        if not src.exists():
            raise FileNotFoundError(f"mobileclip2 文件不存在: {src}")

        from ultralytics.utils import SETTINGS

        # 让 attempt_download_asset 优先在此目录查找。
        SETTINGS["weights_dir"] = str(src.parent)

        # 若文件名不是标准名，则复制一份标准名供 YOLOE 固定字符串查找。
        std_name = src.parent / "mobileclip2_b.ts"
        if src.name != "mobileclip2_b.ts" and not std_name.exists():
            shutil.copy2(src, std_name)

    @staticmethod
    def _normalize_prompt(prompt: str | list[str]) -> list[str]:
        """统一提示词格式为 list[str]，并去除空字符串。"""
        if isinstance(prompt, str):
            items = [prompt]
        else:
            items = list(prompt)

        items = [x.strip() for x in items if x.strip()]
        if not items:
            raise ValueError("prompt 不能为空，请提供至少一个有效提示词。")
        return items

    @staticmethod
    def _ensure_bgr_u8(image: np.ndarray) -> np.ndarray:
        """把输入图像标准化为 BGR uint8 三通道。"""
        if image.ndim == 2:
            # 灰度图转三通道，便于统一送入模型。
            image = np.repeat(image[..., None], 3, axis=2)
        elif image.ndim == 3:
            image = image[..., :3]
        else:
            raise ValueError("image 维度不正确，应为 (H,W) 或 (H,W,C)。")

        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        return image

    def set_prompt(self, prompt: str | list[str]) -> None:
        """
        更新提示词。

        仅当提示词确实变化时才调用 set_classes，避免不必要开销。
        """
        prompt_list = self._normalize_prompt(prompt)
        if prompt_list != self._prompt:
            self.model.set_classes(prompt_list)
            self._prompt = prompt_list

    def infer(
        self, image_bgr: np.ndarray, prompt: str | list[str] | None = None
    ) -> Yoloe26Result:
        """
        单帧推理：图像 + 提示词 -> 叠加图 + 黑白掩码。

        参数：
        - image_bgr: 输入图像（灰度或彩色均可）。
        - prompt: 可选，传入时会覆盖当前提示词。
        """
        if prompt is not None:
            self.set_prompt(prompt)

        frame = self._ensure_bgr_u8(image_bgr)

        t0 = time.perf_counter()
        result = self.model.predict(
            source=frame,
            conf=float(self.cfg.conf),
            imgsz=int(self.cfg.imgsz),
            max_det=int(self.cfg.max_det),
            device=self.device,
            half=bool(self.cfg.use_half),
            save=False,
            verbose=False,
        )[0]
        infer_ms = (time.perf_counter() - t0) * 1000.0

        # 渲染叠加图。
        overlay = result.plot()

        # 统计检测数量。
        det_count = 0
        if result.boxes is not None and result.boxes.data is not None:
            det_count = int(len(result.boxes.data))

        # 构建黑白掩码。
        if (
            result.masks is None
            or result.masks.data is None
            or len(result.masks.data) == 0
        ):
            mask_bw = np.zeros(frame.shape[:2], dtype=np.uint8)
        else:
            # 不同版本下 masks.data 可能是 torch.Tensor 或 numpy.ndarray，这里统一兼容。
            masks_data = cast(Any, result.masks.data)
            if hasattr(masks_data, "detach"):
                masks = masks_data.detach().cpu().numpy()
            else:
                masks = np.asarray(masks_data)
            binary_masks = (masks >= float(self.cfg.mask_threshold)).astype(
                np.uint8
            ) * 255
            mask_bw = np.max(binary_masks, axis=0)

        return Yoloe26Result(
            overlay=overlay,
            mask_bw=mask_bw,
            det_count=det_count,
            infer_ms=infer_ms,
            prompt=list(self._prompt),
        )


if __name__ == "__main__":
    """
    最小实时示例：
    1) 用 realsense.py 读取彩色图像。
    2) 每帧调用 YOLOE-26 生成掩码。
    3) 实时显示 overlay 与黑白 mask。

    退出：按 q 或 ESC。
    """

    project_dir = Path(__file__).resolve().parents[2]

    # ===== 直接改这里（不使用 argparse） =====
    MODEL_PATH = str(project_dir / "checkpoints" / "yoloe-26l-seg.pt")
    PROMPT: str | list[str] = ["white block"]
    CONF = 0.15
    IMGSZ = 640
    MAX_DET = 2
    MASK_THRESHOLD = 0.5
    USE_HALF = False
    MOBILECLIP2_PATH = str(project_dir / "mobileclip2_b.ts")

    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    CAMERA_FPS = 30
    # =====================================

    from realsense import RealSenseCamera

    camera = RealSenseCamera(width=CAMERA_WIDTH, height=CAMERA_HEIGHT, fps=CAMERA_FPS)
    camera.start()

    masker = Yoloe26Masker(
        Yoloe26Config(
            model_path=MODEL_PATH,
            conf=CONF,
            imgsz=IMGSZ,
            max_det=MAX_DET,
            mask_threshold=MASK_THRESHOLD,
            use_half=USE_HALF,
            device=None,
            mobileclip2_path=MOBILECLIP2_PATH,
        ),
        init_prompt=PROMPT,
    )

    cv2.namedWindow("YOLOE26 Overlay", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("YOLOE26 Mask BW", cv2.WINDOW_AUTOSIZE)

    try:
        print("窗口已打开，按 q 或 ESC 退出。")
        while True:
            # 这里使用 RGBD 接口拿 color 帧，满足“实时接收图像”需求。
            rgbd = camera.get_aligned_rgbd_frames()

            out = masker.infer(rgbd.color_bgr)

            # 在 overlay 上叠加简洁信息，便于实时观察。
            cv2.putText(
                out.overlay,
                f"infer: {out.infer_ms:.1f} ms | det: {out.det_count} | prompt: {', '.join(out.prompt)}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("YOLOE26 Overlay", out.overlay)
            cv2.imshow("YOLOE26 Mask BW", out.mask_bw)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break
    finally:
        camera.stop()
        cv2.destroyAllWindows()
