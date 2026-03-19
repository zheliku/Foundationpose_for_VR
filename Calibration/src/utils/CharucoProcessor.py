"""
ChArUco标定板处理模块

该模块提供ChArUco标定板的检测、位姿估计和可视化功能。
支持以下功能：
- ChArUco角点和ArUco标记的检测
- 标定板位姿估计（旋转和平移）
- 检测结果可视化
- 位姿信息显示

Author: ChArUcoDetect Team
Date: 2025-12-10
"""

import cv2
import numpy as np
import utils.log as log

# 配置日志
logger = log.setup_logger(__file__)


class CharucoProcessor:
    """
    ChArUco标定板处理器类

    该类封装了ChArUco标定板的检测、位姿估计和可视化功能。
    使用OpenCV的ArUco模块进行检测，支持多种检测参数配置。

    Attributes:
        detector: OpenCV的ChArUco检测器对象

    Constants:
        MIN_CORNERS_FOR_POSE: 位姿估计所需的最少角点数
    """

    # 位姿估计所需的最少角点数
    MIN_CORNERS_FOR_POSE = 6

    def __init__(
        self,
        charuco_board: cv2.aruco.CharucoBoard,
        charuco_params: cv2.aruco.CharucoParameters,
        detector_params: cv2.aruco.DetectorParameters,
        refine_parameters: cv2.aruco.RefineParameters,
    ):
        """
        初始化ChArUco检测器

        Args:
            charuco_board: ChArUco标定板对象，定义板的几何结构
            charuco_params: ChArUco检测参数，包含相机内参等
            detector_params: ArUco检测器参数，控制标记检测行为
            refine_parameters: 标记精化参数，用于提高检济精度

        Note:
            所有参数都应该通过setting模块的Config类创建，
            以保证配置的一致性和可维护性。
        """
        # 创建ChArUco检测器
        self.detector = cv2.aruco.CharucoDetector(
            charuco_board, charuco_params, detector_params, refine_parameters
        )
        logger.debug("CharucoProcessor初始化完成")

    @property
    def charuco_board(self) -> cv2.aruco.CharucoBoard:
        """
        获取ChArUco标定板对象

        Returns:
            cv2.aruco.CharucoBoard: ChArUco标定板对象
        """
        return self.detector.getBoard()

    @property
    def charuco_params(self) -> cv2.aruco.CharucoParameters:
        """
        获取ChArUco检测参数

        Returns:
            cv2.aruco.CharucoParameters: ChArUco检测参数对象
        """
        return self.detector.getCharucoParameters()

    @property
    def detector_params(self) -> cv2.aruco.DetectorParameters:
        """
        获取ArUco检测器参数

        Returns:
            cv2.aruco.DetectorParameters: ArUco检测器参数对象
        """
        return self.detector.getDetectorParameters()

    @property
    def refine_parameters(self) -> cv2.aruco.RefineParameters:
        """
        获取标记精化参数

        Returns:
            cv2.aruco.RefineParameters: 标记精化参数对象
        """
        return self.detector.getRefineParameters()

    def detect_board(self, gray_image: np.ndarray):
        """
        检测ChArUco标定板

        该方法会同时检测ArUco标记和ChArUco角点。
        ArUco标记是印刷在标定板上的二维码，
        ChArUco角点是白色和黑色方格的交点。

        Args:
            gray_image: 灰度图像 (HxW)，单通道8位图像

        Returns:
            tuple[
                np.ndarray | None,  # charuco_corners: ChArUco角点坐标 (Nx1x2)
                np.ndarray | None,  # charuco_ids: ChArUco角点ID (Nx1)
                tuple[np.ndarray, ...] | list[np.ndarray] | None,  # marker_corners: ArUco标记角点坐标
                np.ndarray | None,  # marker_ids: ArUco标记ID (Nx1)
            ]

        Note:
            - 如果检测失败，相应的返回值为None
            - 角点坐标采用像素坐标系，原点在图像左上角
        """
        return self.detector.detectBoard(gray_image)

    def estimate_pose(
        self, charuco_corners: np.ndarray | None, charuco_ids: np.ndarray | None
    ) -> tuple[bool, np.ndarray | None, np.ndarray | None]:
        """
        估计ChArUco标定板的位姿

        使用PnP (Perspective-n-Point)算法从2D-3D对应关系估计相机相对于
        标定板的位姿。需要至少6个角点才能进行可靠的位姿估计。

        Args:
            charuco_corners: ChArUco角点坐标 (Nx1x2)，像素坐标
            charuco_ids: ChArUco角点ID (Nx1)

        Returns:
            tuple[bool, np.ndarray | None, np.ndarray | None]:
                - success: 是否成功估计位姿
                - rvec: 旋转向量 (3x1)，Rodrigues格式
                - tvec: 平移向量 (3x1)，单位：米

        Note:
            - 旋转向量可以通过cv2.Rodrigues()转换为旋转矩阵
            - 平移向量表示标定板原点在相机坐标系下的位置
            - 角点数不足时返回(False, None, None)
        """
        # 检查角点数量是否足够
        if (
            charuco_corners is None
            or charuco_ids is None
            or len(charuco_ids) < self.MIN_CORNERS_FOR_POSE
        ):
            if charuco_ids is not None:
                logger.debug(
                    f"角点数不足，无法估计位姿："
                    f"需要{self.MIN_CORNERS_FOR_POSE}个，当前{len(charuco_ids)}个"
                )
            return False, None, None

        # 获取棋盘的3D点坐标和对应的2D图像点
        obj_points, img_points = self.charuco_board.matchImagePoints(
            charuco_corners,  # pyright: ignore[reportArgumentType, reportCallIssue]
            charuco_ids,  # pyright: ignore[reportArgumentType, reportCallIssue]
        )

        # 使用solvePnP估计位姿
        success, rvec, tvec = cv2.solvePnP(
            obj_points,
            img_points,
            self.charuco_params.cameraMatrix,
            self.charuco_params.distCoeffs,
        )

        if success:
            logger.debug(f"位姿估计成功，使用{len(charuco_ids)}个角点")

        return success, rvec, tvec

    @classmethod
    def draw_detection_results(
        cls,
        image: np.ndarray,
        charuco_corners: np.ndarray | None,
        charuco_ids: np.ndarray | None,
        marker_corners: tuple[np.ndarray, ...] | list[np.ndarray] | None,
        marker_ids: np.ndarray | None,
    ) -> np.ndarray:
        """
        在图像上绘制检测结果

        该方法会在原图像上绘制：
        - ChArUco角点：红色圆圈和ID编号
        - ArUco标记：绿色边框和ID编号

        Args:
            image: 输入图像 (HxWx3)，BGR格式
            charuco_corners: ChArUco角点坐标 (Nx1x2)
            charuco_ids: ChArUco角点ID (Nx1)
            marker_corners: ArUco标记角点坐标 (Nx4x2)
            marker_ids: ArUco标记ID (Nx1)

        Returns:
            np.ndarray: 绘制了检测结果的图像

        Note:
            - 该方法会直接修改输入图像
            - 如果某项检测结果为None，则跳过绘制
        """
        # 绘制ChArUco角点
        if charuco_ids is not None and charuco_corners is not None:
            cv2.aruco.drawDetectedCornersCharuco(
                image,
                charuco_corners,
                charuco_ids,  # pyright: ignore[reportArgumentType, reportCallIssue]
            )
            logger.debug(f"绘制了{len(charuco_ids)}个ChArUco角点")

        # 绘制ArUco标记
        if marker_ids is not None and marker_corners is not None:
            cv2.aruco.drawDetectedMarkers(
                image,
                marker_corners,
                marker_ids,  # pyright: ignore[reportArgumentType, reportCallIssue]
            )
            logger.debug(f"绘制了{len(marker_ids)}个ArUco标记")

        return image

    def draw_pose(
        self, image: np.ndarray, rvec: np.ndarray, tvec: np.ndarray
    ) -> np.ndarray:
        """
        在图像上绘制坐标轴

        绘制以标定板原点为中心的3D坐标系：
        - X轴：红色
        - Y轴：绿色
        - Z轴：蓝色（指向相机）

        Args:
            image: 输入图像 (HxWx3)，BGR格式
            rvec: 旋转向量 (3x1)
            tvec: 平移向量 (3x1)

        Returns:
            np.ndarray: 绘制了坐标轴的图像

        Note:
            - 坐标轴长度为棋盘格边长的3倍
            - 该方法会直接修改输入图像
        """
        # 计算坐标轴长度（棋盘格边长的3倍）
        axis_length = self.charuco_board.getSquareLength() * 3

        # 绘制坐标轴（红色=X轴, 绿色=Y轴, 蓝色=Z轴）
        cv2.drawFrameAxes(
            image,
            self.charuco_params.cameraMatrix,
            self.charuco_params.distCoeffs,
            rvec,
            tvec,
            axis_length,
            3,  # 线条粗细
        )

        logger.debug("已绘制坐标轴")
        return image

    @classmethod
    def draw_info_text(
        cls,
        image: np.ndarray,
        tvec: np.ndarray,
        rvec: np.ndarray,
        charuco_ids: np.ndarray | None,
    ) -> np.ndarray:
        """
        在图像上绘制位姿信息文本

        显示以下信息：
        - 标定板到相机的距离（米）
        - 检测到的角点数量
        - 平移向量 (x, y, z)
        - 旋转向量 (rx, ry, rz)

        Args:
            image: 输入图像 (HxWx3)，BGR格式
            tvec: 平移向量 (3x1)，单位：米
            rvec: 旋转向量 (3x1)
            charuco_ids: ChArUco角点ID (Nx1)

        Returns:
            np.ndarray: 绘制了信息文本的图像

        Note:
            - 文本显示在图像左上角
            - 使用绿色字体
            - 该方法会直接修改输入图像
        """
        # 计算距离（平移向量的模）
        distance = np.linalg.norm(tvec)

        # 准备文本信息
        info_texts = [
            f"Distance: {distance:.3f}m",
            f"Corners: {len(charuco_ids) if charuco_ids is not None else 0}",
            f"tvec: [{tvec[0][0]:.3f}, {tvec[1][0]:.3f}, {tvec[2][0]:.3f}]",
            f"rvec: [{rvec[0][0]:.3f}, {rvec[1][0]:.3f}, {rvec[2][0]:.3f}]",
        ]

        # 绘制文本
        for i, text in enumerate(info_texts):
            cv2.putText(
                image,
                text,
                (10, 30 + i * 30),  # 位置：左上角，每行间30像素
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,  # 字体大小
                (0, 255, 0),  # 颜色：绿色 (BGR)
                2,  # 线条粗细
            )

        logger.debug("已绘制位姿信息文本")
        return image
