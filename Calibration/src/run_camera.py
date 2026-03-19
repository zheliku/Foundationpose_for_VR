"""
相机运行和图像采集脚本

该脚本用于运行RealSense相机并采集图像数据。

使用说明：
- 按 Enter 键保存当前帧
- 按 'q' 键退出程序

Author: ChArUcoDetect Team
Date: 2025-12-10
"""

import cv2
from pathlib import Path
import logging
from utils.RealSenseCamera import RealSenseCamera

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ====================
# 配置参数
# ====================

# 图像保存目录
SAVE_DIR = Path(__file__).parent.parent / "image"

# 窗口名称
WINDOW_NAME = "RealSense Camera Preview"

# 按键定义
KEY_ENTER = 13  # Enter键的ASCII码
KEY_QUIT = ord('q')  # 'q'键

def setup_save_directory(save_dir: Path) -> Path:
    """
    设置图像保存目录
    
    Args:
        save_dir: 保存目录路径
    
    Returns:
        Path: 创建后的目录路径
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"图像保存目录: {save_dir}")
    return save_dir


def run_camera_preview():
    """
    运行相机预览和图像采集
    
    该函数会：
    1. 初始化相机
    2. 显示实时预览
    3. 响应用户按键
    4. 保存图像或退出
    
    按键说明：
        - Enter: 保存当前帧
        - 'q': 退出程序
    """
    # 设置保存目录
    save_path = setup_save_directory(SAVE_DIR)
    
    # 初始化帧计数器
    frame_count = 0
    
    # 使用上下文管理器确保相机正确关闭
    with RealSenseCamera() as camera:
        try:
            # 启动相机
            camera.start()
            
            logger.info("\n" + "=" * 50)
            logger.info("操作说明：")
            logger.info("  - 按 Enter 键保存当前帧")
            logger.info("  - 按 'q' 键退出程序")
            logger.info("=" * 50 + "\n")
            
            while True:
                try:
                    # 获取一帧图像
                    frame = camera.get_frame()
                    
                    # 显示帧
                    cv2.imshow(WINDOW_NAME, frame)
                    
                    # 获取按键
                    key = cv2.waitKey(1) & 0xFF
                    
                    # 处理按键事件
                    if key == KEY_ENTER:
                        # 保存图片
                        filename = f"frame_{frame_count}.png"
                        file_path = save_path / filename
                        cv2.imwrite(str(file_path), frame)
                        frame_count += 1
                        logger.info(f"✅ 已保存第 {frame_count} 张图片: {filename}")
                    
                    elif key == KEY_QUIT:
                        # 退出程序
                        logger.info("用户请求退出")
                        break
                    
                except KeyboardInterrupt:
                    logger.info("接收到中断信号")
                    break
                except Exception as e:
                    logger.error(f"处理图像帧时发生错误: {e}")
                    continue
            
        finally:
            cv2.destroyAllWindows()
            logger.info(f"\n总计保存 {frame_count} 张图片")
            logger.info("相机关闭")


def main():
    """
    主程序入口
    """
    logger.info("=" * 50)
    logger.info("RealSense相机图像采集工具")
    logger.info("=" * 50)
    
    try:
        run_camera_preview()
        logger.info("✅ 程序正常结束")
    except Exception as e:
        logger.error(f"❌ 程序异常: {e}")
        raise


if __name__ == "__main__":
    main()