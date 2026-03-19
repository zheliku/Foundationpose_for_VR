"""
ChArUco标定板检测主程序

该模块提供两种检测模式：
1. 实时相机模式：使用RealSense相机进行实时检测
2. 静态图片模式：对保存的图片进行检测

主要功能：
- 检测ChArUco角点和ArUco标记
- 估计标定板位姿（旋转和平移向量）
- 可视化检测结果和坐标系
- 显示位姿信息（距离、角点数等）

Date: 2025-12-10
"""

import datetime
import json

import cv2
import cv2.aruco
import numpy as np
from pathlib import Path

import setting
from utils import CharucoProcessor, RealSenseCamera, log

# 配置日志
# logging.basicConfig(level=logging.INFO)
logger = log.setup_logger(__file__)


def process_image_detection(
    image: np.ndarray,
    processor: CharucoProcessor,
):
    """
    处理单幅图像的ChArUco检测

    该函数封装了完整的检测流程：
    1. 灰度化图像
    2. 检测标定板
    3. 估计位姿
    4. 绘制结果

    Args:
        image: 输入图像 (HxWx3)，BGR格式
        processor: ChArUco处理器对象

    Returns:
        np.ndarray: 绘制了检测结果的图像

    Note:
        - 检测信息会通过logger输出
        - 位姿估计需要至少6个角点
    """
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 检测ChArUco标定板
    charuco_corners, charuco_ids, marker_corners, marker_ids = processor.detect_board(
        gray
    )

    # 打印检测结果
    corner_count = len(charuco_ids) if charuco_ids is not None else 0
    marker_count = len(marker_ids) if marker_ids is not None else 0

    logger.debug(f"检测结果: {corner_count}个ChArUco角点, {marker_count}个ArUco标记")

    # 绘制检测结果
    image = processor.draw_detection_results(
        image, charuco_corners, charuco_ids, marker_corners, marker_ids  # type: ignore
    )

    # 估计位姿
    success, rvec, tvec = processor.estimate_pose(charuco_corners, charuco_ids)

    if success:
        # 位姿估计成功
        logger.debug("=== 棋盘位姿信息 ===")
        logger.debug(
            f"旋转向量 (rvec): {rvec.flatten()}" # type: ignore
        )
        logger.debug(
            f"平移向量 (tvec): {tvec.flatten()}" # type: ignore
        )

        # 计算并显示距离
        distance = np.linalg.norm(tvec)  # type: ignore
        logger.debug(f"距离相机: {distance:.3f} 米")

        # 绘制坐标轴和位姿信息
        image = processor.draw_pose(image, rvec, tvec)  # type: ignore
        image = processor.draw_info_text(image, tvec, rvec, charuco_ids)  # type: ignore
    else:
        # 位姿估计失败
        logger.debug("位姿估计失败")
        logger.debug(
            f"检测到的角点数量不足（需要至少{processor.MIN_CORNERS_FOR_POSE}个，"
            f"当前: {corner_count}）"
        )
        pass

    return (
        image,
        success,
        rvec,
        tvec,
        (charuco_corners, charuco_ids, marker_corners, marker_ids),
    )


def run_realtime_detection(processor: CharucoProcessor):
    """
    运行实时检测模式（RealSense相机）

    该函数会：
    1. 启动RealSense相机
    2. 循环获取并处理图像帧
    3. 实时显示检测结果
    4. 响应用户按键退出

    Args:
        processor: ChArUco处理器对象

    Note:
        - 按 'q' 键退出程序
        - 使用上下文管理器确保相机正确关闭
    """
    # 使用上下文管理器确保资源正确释放
    with RealSenseCamera() as camera:
        try:
            # 启动相机并获取内参
            camera.start()

            logger.info("开始实时检测，按 'q' 键退出")

            while True:
                try:
                    # 获取一帧图像
                    image = camera.get_frame()

                    # 处理图像（检测、位姿估计、绘制）
                    image, success, rvec, tvec, corners_and_ids = (
                        process_image_detection(image, processor)
                    )

                    # 显示图像
                    cv2.imshow("RealSense ChArUco Detection", image)

                    # 按下空格保存 rvec、tvec
                    key = cv2.waitKey(1) & 0xFF

                    if key == ord(" "):
                        logger.info("保存 rvec、tvec")
                        save_image_detection(
                            setting.output_path, image, tvec, rvec, corners_and_ids
                        )

                    # 按'q'键退出
                    if key == ord("q"):
                        logger.info("用户请求退出")
                        break

                except KeyboardInterrupt:
                    logger.info("接收到中断信号")
                    break
                except Exception as e:
                    logger.error(f"处理图像时发生错误: {e}")
                    continue

        finally:
            cv2.destroyAllWindows()
            logger.info("实时检测结束")


def save_image_detection(
    output_path: Path, img: np.ndarray, tvec=None, rvec=None, corners_and_ids=None
):
    json_path = (
        setting.output_path
        / f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}" / "realsense_calibration.json"
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)

    charuco_corners, charuco_ids, marker_corners, marker_ids = corners_and_ids  # type: ignore

    charucos = (
        [
            {"id": int(charuco_id[0]), "corner": list(charuco_corner[0].tolist())}
            for charuco_corner, charuco_id in zip(charuco_corners, charuco_ids)
        ]
        if charuco_corners is not None and charuco_ids is not None
        else []
    )

    markers = (
        [
            {
                "ids": int(marker_id[0]),
                "corners": list(marker_corner[0].tolist()),
            }
            for marker_corner, marker_id in zip(marker_corners, marker_ids)
        ]
        if marker_corners
        else []
    )

    json.dump(
        {
            "tvec": [float(tvec[i][0]) for i in range(3)] if tvec is not None else None,
            "rvec": [float(rvec[i][0]) for i in range(3)] if rvec is not None else None,
            "charuco": charucos,
            "markers": markers,
        },
        json_path.open("w"),
    )
    cv2.imwrite(str(json_path.with_suffix(".png")), img)


def run_static_image_detection(
    img_paths: list[Path],
    processor: CharucoProcessor,
):
    """
    运行静态图片检测模式

    该函数会：
    1. 加载指定的图片
    2. 对每张图片进行检测
    3. 显示检测结果
    4. 等待用户按键

    Args:
        img_paths: 图片路径列表
        processor: ChArUco处理器对象

    Note:
        - 所有图片处理完后需要按任意键关闭窗口
        - 支持多张图片同时显示
    """
    if not img_paths:
        logger.warning("未指定图片路径")
        return

    logger.info(f"开始处理{len(img_paths)}张图片")

    for img_path in img_paths:
        try:
            # 读取图像
            image = cv2.imread(str(img_path))

            if image is None:
                logger.error(f"无法读取图片: {img_path}")
                continue

            logger.info(f"处理图片: {img_path.name}")

            # 处理图像（检测、位姿估计、绘制）
            image, success, rvec, tvec = process_image_detection(image, processor) # type: ignore

            # 显示结果
            window_name = f"Static Image {img_path.name} ChArUco Detection"
            cv2.imshow(window_name, image)

        except Exception as e:
            logger.error(f"处理图片{img_path}时发生错误: {e}")
            continue

    logger.info("所有图片处理完毕，按任意键关闭窗口")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    """
    主程序入口

    初始化配置并根据模式选择启动相应的检测方法。
    """
    logger.info("=" * 50)
    logger.info("ChArUco标定板检测系统")
    logger.info("=" * 50)

    # 创建ChArUco检测器配置
    logger.info("初始化配置...")
    charuco_board = setting.charuco_board()
    charuco_params = setting.charuco_parameters()
    detector_params = cv2.aruco.DetectorParameters()
    refine_params = cv2.aruco.RefineParameters()

    # 创建ChArUco处理器
    processor = CharucoProcessor(
        charuco_board,
        charuco_params,
        detector_params,
        refine_params,
    )
    logger.info("配置初始化完成")

    # 选择检测模式
    USE_REALSENSE = False  # True=实时相机, False=静态图片
    USE_REALSENSE = True

    if USE_REALSENSE:
        logger.info("启动实时相机检测模式")
        run_realtime_detection(processor)
    else:
        logger.info("启动静态图片检测模式")

        # 配置图片路径
        img_path = Path(__file__).parent.parent / "image" / "frame_0.png"

        if not img_path.exists():
            logger.error(f"图片不存在: {img_path}")
            return

        # 可以处理多张图片
        img_paths = [img_path]

        run_static_image_detection(img_paths, processor)

    logger.info("程序结束")


if __name__ == "__main__":
    main()
