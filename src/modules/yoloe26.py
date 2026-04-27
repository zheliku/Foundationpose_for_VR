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

    # 被选中用于下游 FoundationPose 的 mask 下标；无检测时为 -1。
    selected_index: int = -1

    # 被选中 mask 的面积占整幅图比例。
    mask_area_ratio: float = 0.0


class Yoloe26Masker:
    """YOLOE-26 实时掩码生成器。"""

    # 输入配置。
    model_path: str = ""  # YOLOE 权重路径。
    conf: float = 0.15  # 置信度阈值。
    imgsz: int = 640  # 推理输入尺寸。
    max_det: int = 2  # 最大检测数量。
    mask_threshold: float = 0.5  # mask 二值化阈值。
    use_half: bool = False  # 是否使用半精度。
    device: str | int | None = None  # 推理设备。
    mobileclip2_path: str | None = None  # 文本编码器权重路径。

    # 运行时对象与状态。
    model: YOLOE  # 已加载的 YOLOE 模型（__init__ 中创建）。
    _prompt: list[str]  # 当前生效提示词（__init__ 中初始化）。

    def __init__(
        self,
        model_path: str,
        init_prompt: str | list[str],
        conf: float = 0.15,
        imgsz: int = 640,
        max_det: int = 2,
        mask_threshold: float = 0.5,
        use_half: bool = False,
        device: str | int | None = None,
        mobileclip2_path: str | None = None,
    ) -> None:
        """
        初始化 YOLOE 掩码生成器。

        参数：
        - model_path: YOLOE 权重路径。
        - init_prompt: 初始提示词。
        - conf/imgsz/max_det: 推理阈值与输入尺寸配置。
        - mask_threshold: mask 二值化阈值。
        - use_half: 是否使用半精度。
        - device: 指定推理设备；为空时自动选择。
        - mobileclip2_path: 文本编码器权重路径。

        初始化流程：
        1. 保存配置并检查权重文件。
        2. 自动选择运行设备。
        3. 配置 mobileclip2 本地路径。
        4. 加载模型并设置初始提示词。
        """
        self.model_path = str(model_path)
        self.conf = float(conf)
        self.imgsz = int(imgsz)
        self.max_det = int(max_det)
        self.mask_threshold = float(mask_threshold)
        self.use_half = bool(use_half)
        self.device = device
        self.mobileclip2_path = mobileclip2_path

        # 校验模型路径，提前失败更易定位问题。
        model_path_obj = Path(self.model_path)
        if not model_path_obj.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path_obj}")

        # 自动选择设备：有 CUDA 则 GPU0，否则 CPU。
        if self.device is None:
            self.device = 0 if torch.cuda.is_available() else "cpu"

        # 在 set_classes 前配置本地 mobileclip2 权重路径。
        self._configure_mobileclip2_path()

        # 加载模型并设置初始类别提示。
        self.model = YOLOE(str(self.model_path))
        self._prompt = []
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
        if not self.mobileclip2_path:
            return

        src = Path(self.mobileclip2_path).expanduser().resolve()
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
            conf=float(self.conf),
            imgsz=int(self.imgsz),
            max_det=int(self.max_det),
            device=self.device,
            half=bool(self.use_half),
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

        selected_index = -1

        # 构建黑白掩码。注意：下游 FoundationPose 需要的是单个目标 mask，
        # 不能把多个检测直接 union，否则误检会污染初始注册。
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
            binary_masks = (masks >= float(self.mask_threshold)).astype(np.uint8) * 255

            scores = np.ones((binary_masks.shape[0],), dtype=np.float32)
            if result.boxes is not None and getattr(result.boxes, "conf", None) is not None:
                conf = result.boxes.conf
                if hasattr(conf, "detach"):
                    scores = conf.detach().cpu().numpy().astype(np.float32)
                else:
                    scores = np.asarray(conf, dtype=np.float32)

            areas = binary_masks.reshape(binary_masks.shape[0], -1).sum(axis=1)
            valid = areas > 0
            if np.any(valid):
                score = scores[: binary_masks.shape[0]].copy()
                score[~valid] = -1.0
                selected_index = int(np.argmax(score))
                mask_bw = binary_masks[selected_index]
            else:
                mask_bw = np.zeros(frame.shape[:2], dtype=np.uint8)

        if mask_bw.shape[:2] != frame.shape[:2]:
            mask_bw = cv2.resize(
                mask_bw,
                (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        mask_area_ratio = float(np.count_nonzero(mask_bw)) / float(mask_bw.size)

        return Yoloe26Result(
            overlay=overlay,
            mask_bw=mask_bw,
            det_count=det_count,
            infer_ms=infer_ms,
            prompt=list(self._prompt),
            selected_index=selected_index,
            mask_area_ratio=mask_area_ratio,
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
        model_path=MODEL_PATH,
        init_prompt=PROMPT,
        conf=CONF,
        imgsz=IMGSZ,
        max_det=MAX_DET,
        mask_threshold=MASK_THRESHOLD,
        use_half=USE_HALF,
        device=None,
        mobileclip2_path=MOBILECLIP2_PATH,
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
