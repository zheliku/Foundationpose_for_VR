"""
RealSense 相机基础 API（新项目重构版）

设计目标：
1. 接受常规相机参数输入（宽、高、帧率、序列号等）。
2. 通过两个独立方法提供两种输出：
   - 左右双目图像（stereo）
   - 对齐后的 RGBD 图像（depth 对齐到 color）
3. 实现尽量简单朴素，便于在工程内复用与维护。

说明：
- 本文件不依赖旧项目脚本逻辑，可独立使用。
- 默认基于 Intel RealSense（pyrealsense2）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import cv2
import numpy as np

try:
    # 延迟失败策略：
    # - 如果环境中没有安装 pyrealsense2，导入阶段不会导致整个工程其他模块崩溃；
    # - 在真正实例化 RealSenseAPI 时再给出清晰错误。
    import pyrealsense2 as _rs
except Exception:  # pragma: no cover - 该分支取决于本机环境是否安装 RealSense SDK
    _rs = None


# pyrealsense2 的类型提示较弱，这里统一转为 Any，减少静态检查噪声。
rs = cast(Any, _rs)


@dataclass
class StereoFrame:
    """左右双目图像容器。"""

    # 左目图像（通常来自 infrared 1），格式为 uint8 灰度图。
    left: np.ndarray
    # 右目图像（通常来自 infrared 2），格式为 uint8 灰度图。
    right: np.ndarray
    # 时间戳，单位毫秒（由 RealSense 帧对象提供）。
    timestamp_ms: float


@dataclass
class RGBDFrame:
    """对齐后的 RGBD 图像容器。"""

    # 彩色图像，BGR 排列，格式为 uint8。
    color_bgr: np.ndarray
    # 深度图像为 RealSense 原始 z16 值（不是米制深度），格式为 uint16。
    depth: np.ndarray
    # 时间戳，单位毫秒（以对齐后的 color 帧时间戳为准）。
    timestamp_ms: float


class RealSenseCamera:
    """
    RealSense 相机最小可用 API。

    使用方式（示例）：
        cam = RealSenseCamera(width=640, height=480, fps=30)
        cam.start()

        stereo = cam.get_stereo_frames()
        rgbd = cam.get_aligned_rgbd_frames()

        cam.stop()

    约定：
    - `start()` 后才可调用取帧方法。
    - `stop()` 可重复调用（幂等），方便在异常处理中安全清理。
    """

    # 输入配置。
    width: int = 640  # 采集宽度。
    height: int = 480  # 采集高度。
    fps: int = 30  # 采集帧率。
    serial_number: str | None = None  # 可选设备序列号。

    # 运行时对象。
    pipeline: Any = None  # RealSense pipeline 对象。
    config: Any = None  # RealSense 配置对象。
    _align_to_color: Any = None  # depth->color 对齐器。

    # 运行状态标记。
    _started: bool = False  # 是否已启动数据流。

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        serial_number: str | None = None,
    ) -> None:
        """
        初始化 RealSense 相机对象（不立即启动）。

        参数：
        - width: 采集宽度。
        - height: 采集高度。
        - fps: 采集帧率。
        - serial_number: 可选设备序列号。

        初始化流程：
        1. 检查 pyrealsense2 环境是否可用。
        2. 保存采集参数。
        3. 运行时对象保持默认占位，等待 start() 创建。
        """
        # 环境检查放在实例化阶段执行，错误更可读。
        if rs is None:
            raise RuntimeError(
                "未检测到 pyrealsense2，请先安装 Intel RealSense SDK 与 Python 绑定。"
            )

        # 保存用户输入的基础采集参数。
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.serial_number = serial_number

    def start(self) -> None:
        """
        启动 RealSense 数据流。

        这里一次性开启四路流：
        1) color：用于 RGB 图像
        2) depth：用于深度图
        3) infrared(1)：左目
        4) infrared(2)：右目

        这样可同时满足 stereo 与 RGBD 两种调用需求。
        """
        # 已启动时直接返回，保持接口幂等。
        if self._started:
            return

        # 构建管线与配置对象。
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # 多相机场景可通过序列号绑定设备，防止选错相机。
        if self.serial_number:
            self.config.enable_device(self.serial_number)

        # 开启彩色流（BGR8）。
        self.config.enable_stream(
            rs.stream.color,
            self.width,
            self.height,
            rs.format.bgr8,
            self.fps,
        )
        # 开启深度流（z16）。
        self.config.enable_stream(
            rs.stream.depth,
            self.width,
            self.height,
            rs.format.z16,
            self.fps,
        )
        # 开启左目红外流（infrared 1）。
        self.config.enable_stream(
            rs.stream.infrared,
            1,
            self.width,
            self.height,
            rs.format.y8,
            self.fps,
        )
        # 开启右目红外流（infrared 2）。
        self.config.enable_stream(
            rs.stream.infrared,
            2,
            self.width,
            self.height,
            rs.format.y8,
            self.fps,
        )

        # 真正启动硬件采集。
        self.pipeline.start(self.config)

        # 创建对齐器：将 depth 对齐到 color 坐标系。
        self._align_to_color = rs.align(rs.stream.color)

        # 更新运行状态。
        self._started = True

    def stop(self) -> None:
        """
        停止采集并释放资源。

        该方法可重复调用，适合写在 finally 里保证稳定清理。
        """
        # 未启动时无需处理，直接返回。
        if not self._started:
            return

        # 先尝试停止管线。
        if self.pipeline is not None:
            self.pipeline.stop()

        # 清空运行时对象，避免误用旧引用。
        self.pipeline = None
        self.config = None
        self._align_to_color = None
        self._started = False

    def get_stereo_frames(self) -> StereoFrame:
        """
        获取一组左右双目图像。

        返回：
        - StereoFrame.left：左目灰度图
        - StereoFrame.right：右目灰度图

        注意：
        - 该方法基于 infrared(1/2) 流，不做额外图像增强。
        - 如需时序稳定，可在上层自行做队列或时间戳同步策略。
        """
        # 防御式检查：未启动时禁止取帧。
        if not self._started or self.pipeline is None:
            raise RuntimeError("RealSenseCamera 尚未启动，请先调用 start()。")

        # 阻塞等待下一组可用帧。
        frames = self.pipeline.wait_for_frames()

        # 提取左右红外帧。
        left_frame = frames.get_infrared_frame(1)
        right_frame = frames.get_infrared_frame(2)

        # 防御式校验，避免返回空数据。
        if not left_frame or not right_frame:
            raise RuntimeError("未获取到有效的双目红外帧。")

        # 转为 numpy 数组，便于后续算法直接使用。
        left = np.asanyarray(left_frame.get_data())
        right = np.asanyarray(right_frame.get_data())

        # 使用左目时间戳作为本次 stereo 帧时间。
        timestamp_ms = float(left_frame.get_timestamp())

        return StereoFrame(left=left, right=right, timestamp_ms=timestamp_ms)

    def get_aligned_rgbd_frames(self) -> RGBDFrame:
        """
        获取一组对齐后的 RGBD 图像。

        处理流程：
        1) 取原始 frameset（color + depth）
        2) 使用 align(depth->color) 完成像素坐标对齐
        3) 输出对齐后的 color 与 depth

        返回：
        - RGBDFrame.color_bgr：BGR 彩色图
        - RGBDFrame.depth：对齐后的深度图
        """
        # 防御式检查：确保已启动且对齐器可用。
        if not self._started or self.pipeline is None or self._align_to_color is None:
            raise RuntimeError("RealSenseCamera 尚未启动，请先调用 start()。")

        # 阻塞等待下一组原始帧。
        frames = self.pipeline.wait_for_frames()

        # 执行对齐：把深度图映射到彩色图坐标系。
        aligned_frames = self._align_to_color.process(frames)

        # 提取对齐后的彩色帧与深度帧。
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        # 防御式校验，确保输出完整。
        if not color_frame or not depth_frame:
            raise RuntimeError("未获取到有效的对齐 RGBD 帧。")

        # 转为 numpy 数组供上游算法处理。
        color_bgr = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data())

        # 以 color 帧时间戳作为本次 RGBD 时间。
        timestamp_ms = float(color_frame.get_timestamp())

        return RGBDFrame(color_bgr=color_bgr, depth=depth, timestamp_ms=timestamp_ms)

    def __enter__(self) -> "RealSenseCamera":
        """上下文管理：进入 with 时自动启动。"""
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """上下文管理：离开 with 时自动停止。"""
        self.stop()


if __name__ == "__main__":
    """
    OpenCV 可视化示例：
    1) 创建并启动相机
    2) 循环读取双目图像与对齐 RGBD
    3) 使用 OpenCV 实时显示图像
    4) 按 q 或 ESC 退出并释放资源
    """

    # 示例参数使用最常见的 640x480@30，可按需修改。
    camera = RealSenseCamera(width=640, height=480, fps=30)

    try:
        # 显式启动相机流。
        camera.start()

        print("窗口已打开，按 q 或 ESC 退出。")

        # 只创建一次窗口，避免在循环中重复创建。
        cv2.namedWindow("RealSense Left IR", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("RealSense Right IR", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("RealSense Color", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("RealSense Depth (Aligned)", cv2.WINDOW_AUTOSIZE)

        while True:
            # 获取一组双目图像（左/右灰度图）。
            stereo = camera.get_stereo_frames()

            # 获取一组深度对齐到彩色坐标系的 RGBD 数据。
            rgbd = camera.get_aligned_rgbd_frames()

            # 左右灰度图直接显示。
            cv2.imshow("RealSense Left IR", stereo.left)
            cv2.imshow("RealSense Right IR", stereo.right)

            # 彩色图直接显示（BGR）。
            cv2.imshow("RealSense Color", rgbd.color_bgr)

            # 深度图是 uint16 原始 z16，显示时沿用官方示例的 0.03 缩放系数。
            # 注意：这里只是显示友好的伪彩色，不代表真实米制深度。
            depth_u8 = cv2.convertScaleAbs(rgbd.depth, alpha=0.03)
            depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)
            cv2.imshow("RealSense Depth (Aligned)", depth_color)

            # waitKey(1) 既用于刷新窗口，也用于读取按键。
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
    finally:
        # 无论中途是否异常，都确保释放相机资源。
        camera.stop()
        # 关闭所有 OpenCV 窗口，避免残留空窗。
        cv2.destroyAllWindows()
