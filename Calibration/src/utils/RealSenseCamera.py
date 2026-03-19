"""
RealSense相机管理模块

该模块封装了Intel RealSense相机的初始化、配置、图像获取等功能。
支持以下功能：
- 相机的启动和停止
- 实时图像获取
- 相机内参提取
- 异常处理

Author: ChArUcoDetect Team
Date: 2025-12-10
"""

import pyrealsense2.pyrealsense2 as rs
import numpy as np
import utils.log as log

# 配置日志
logger = log.setup_logger(__file__)


class CameraError(Exception):
    """相机相关异常的基类"""

    pass


class CameraInitError(CameraError):
    """相机初始化失败异常"""

    pass


class CameraNotStartedError(CameraError):
    """相机未启动异常"""

    pass


class FrameAcquisitionError(CameraError):
    """图像获取失败异常"""

    pass


class RealSenseCamera:
    """
    RealSense相机管理类

        该类封装了Intel RealSense相机的所有操作，提供简洁的API接口。
        采用上下文管理器模式，确保资源正确释放。

        Attributes:
            width: 图像宽度，单位：像素
            height: 图像高度，单位：像素
            fps: 帧率，单位：帧/秒
    """

    # 默认配置常量
    DEFAULT_WIDTH = 1280
    DEFAULT_HEIGHT = 720
    DEFAULT_FPS = 30

    # 相机内参矩阵的维度
    CAMERA_MATRIX_SHAPE = (3, 3)
    # Brown-Conrady畸变模型的系数数量
    DISTORTION_COEFFS_COUNT = 5

    def __init__(
        self,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        fps: int = DEFAULT_FPS,
    ):
        """
        初始化RealSense相机

        Args:
            width: 图像宽度，单位：像素，默认为1280
            height: 图像高度，单位：像素，默认为720
            fps: 帧率，单位：帧/秒，默认为30

        Raises:
            ValueError: 如果参数不合法
        """
        # 参数验证
        if width <= 0 or height <= 0:
            raise ValueError(f"图像尺寸必须为正数: width={width}, height={height}")
        if fps <= 0:
            raise ValueError(f"帧率必须为正数: fps={fps}")

        # 保存配置参数
        self.width = width
        self.height = height
        self.fps = fps

        # 相机相关对象（延迟初始化）
        self._pipeline: rs.pipeline | None = None
        self._config: rs.config | None = None
        self._is_started = False

        logger.info(f"RealSense相机已初始化: {width}x{height} @ {fps}fps")

    def _initialize_pipeline(self):
        """
        初始化相机管道和配置

        该方法在start()中被调用，实现延迟初始化，降低耦合度。

        Raises:
            CameraInitError: 如果初始化失败
        """
        try:
            pipeline = rs.pipeline()
            config = rs.config()

            # 配置彩色流
            config.enable_stream(
                rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps
            )

            self._pipeline = pipeline
            self._config = config
            logger.debug("相机管道和配置初始化成功")
        except Exception as e:
            raise CameraInitError(f"相机初始化失败: {e}") from e

    def start(self) -> tuple[np.ndarray, np.ndarray]:
        """
        启动相机并获取相机内参

        该方法会：
        1. 初始化相机管道
        2. 启动数据流
        3. 提取相机内参和畸变系数

        Returns:
            tuple[np.ndarray, np.ndarray]:
                - camera_matrix: 相机内参矩阵 (3x3)
                - dist_coeffs: 畸变系数数组 (5,)

        Raises:
            CameraInitError: 如果相机启动失败
            RuntimeError: 如果相机已经启动

        Note:
            相机内参从相机硬件自动获取，无需手动标定。
            RealSense相机使用Brown-Conrady畸变模型。
        """
        if self._is_started:
            raise RuntimeError("相机已经启动，请勿重复启动")

        try:
            # 初始化管道
            self._initialize_pipeline()

            if self._pipeline is None or self._config is None:
                raise CameraInitError("相机管道初始化失败")

            logger.info("正在启动RealSense相机...")

            # 启动管道
            profile = self._pipeline.start(self._config)
            self._is_started = True

            # 提取相机内参
            camera_matrix, dist_coeffs = self._extract_camera_intrinsics(profile)

            logger.info("相机已启动")
            logger.info(f"相机内参矩阵:\n{camera_matrix}")
            logger.info(f"畸变系数: {dist_coeffs}")
            logger.info("按 'q' 键退出")

            return camera_matrix, dist_coeffs

        except CameraInitError:
            raise
        except Exception as e:
            raise CameraInitError(f"相机启动失败: {e}") from e

    def _extract_camera_intrinsics(self, profile) -> tuple[np.ndarray, np.ndarray]:
        """
        从相机配置文件中提取内参

        Args:
            profile: RealSense相机配置文件对象

        Returns:
            tuple[np.ndarray, np.ndarray]:
                - camera_matrix: 3x3相机内参矩阵
                - dist_coeffs: 畸变系数数组

        Raises:
            CameraInitError: 如果无法获取内参
        """
        try:
            # 获取彩色流的内参
            color_stream = profile.get_stream(rs.stream.color)
            intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

            # 构建相机内参矩阵 (OpenCV格式)
            # 矩阵形式：
            # [[fx,  0, cx],
            #  [ 0, fy, cy],
            #  [ 0,  0,  1]]
            camera_matrix = np.array(
                [
                    [intrinsics.fx, 0, intrinsics.ppx],
                    [0, intrinsics.fy, intrinsics.ppy],
                    [0, 0, 1],
                ],
                dtype=np.float64,
            )

            # 获取畸变系数 (Brown-Conrady模型)
            # 系数顺序：[k1, k2, p1, p2, k3]
            dist_coeffs = np.array(intrinsics.coeffs, dtype=np.float64)

            return camera_matrix, dist_coeffs

        except Exception as e:
            raise CameraInitError(f"无法获取相机内参: {e}") from e

    def get_frame(self, timeout_ms: int = 5000) -> np.ndarray:
        """
        获取一帧图像

        Args:
            timeout_ms: 超时时间，单位：毫秒，默认为5000ms

        Returns:
            np.ndarray: BGR格式的图像数据 (HxWx3)

        Raises:
            CameraNotStartedError: 如果相机未启动
            FrameAcquisitionError: 如果获取图像失败

        Note:
            该方法会阻塞等待新帧到达，直到超时。
        """
        if not self._is_started:
            raise CameraNotStartedError("相机未启动，请先调用start()方法")
        if self._pipeline is None:
            raise CameraNotStartedError("相机管道未初始化，请先调用start()方法")

        try:
            # 等待并获取新帧
            frames = self._pipeline.wait_for_frames(timeout_ms)
            color_frame = frames.get_color_frame()

            if not color_frame:
                raise FrameAcquisitionError("未能获取彩色帧")

            # 转换为numpy数组
            frame_data = np.asanyarray(color_frame.get_data())
            return frame_data

        except RuntimeError as e:
            raise FrameAcquisitionError(f"获取图像帧失败: {e}") from e

    def stop(self):
        """
        停止相机

        释放相机资源，关闭数据流。该方法是幂等的，多次调用不会产生错误。

        Note:
            建议使用try-finally或上下文管理器确保该方法被调用。
        """
        if self._is_started and self._pipeline is not None:
            try:
                self._pipeline.stop()
                self._is_started = False
                logger.info("相机已关闭")
            except Exception as e:
                logger.error(f"关闭相机时发生错误: {e}")
        else:
            logger.debug("相机已经关闭或未启动")

    def is_started(self) -> bool:
        """
        检查相机是否已启动

        Returns:
            bool: 如果相机已启动返回true，否则返回false
        """
        return self._is_started

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口，自动关闭相机"""
        self.stop()
        return False

    def __del__(self):
        """析构函数，确保资源释放"""
        self.stop()
