"""
ChArUco标定板生成脚本

该脚本用于生成ChArUco标定板图像，用于打印和相机标定。

使用步骤：
1. 运行脚本生成标定板图像
2. 按 1:1 比例打印图像
3. 测量实际尺寸确保准确性
4. 将标定板贴在平面上使用

Author: ChArUcoDetect Team
Date: 2025-12-10
"""

import cv2
from pathlib import Path
import logging
import setting

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ====================
# 配置参数
# ====================

# 图像尺寸 (宽, 高) - 单位：像素
IMAGE_SIZE = (1000, 1400)

# 边距大小 - 单位：像素
# 增加边距可以避免打印时裁切到标记
MARGINS = 10

# 边框位数
# 用于在标定板周围添加额外的黑色边框
BORDER_BITS = 1


def generate_charuco_board_image() -> Path:
    """
    生成ChArUco标定板图像
    
    该函数会：
    1. 创建保存目录
    2. 生成标定板图像
    3. 显示图像预览
    4. 保存图像到文件
    
    Returns:
        Path: 保存的图像路径
    
    Note:
        - 图像保存在项目根目录的 charuco 文件夹中
        - 文件名：charuco_board.png
    """
    # 设置保存路径
    save_path = Path(__file__).parent.parent / "charuco" / "charuco_board.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"保存路径: {save_path}")

    # 生成标定板图像
    logger.info("正在生成ChArUco标定板图像...")
    logger.info(f"图像尺寸: {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]} 像素")
    logger.info(f"边距: {MARGINS} 像素")
    logger.info(f"边框位数: {BORDER_BITS}")
    
    img = setting.charuco_board().generateImage(
        IMAGE_SIZE, 
        marginSize=MARGINS, 
        borderBits=BORDER_BITS
    )

    # 显示图像预览
    logger.info("显示图像预览，按任意键继续...")
    cv2.imshow("ChArUco Board Preview", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 保存图片
    cv2.imwrite(str(save_path), img)
    logger.info(f"✅ 标定板图片已生成: {save_path}")
    
    return save_path


def print_usage_instructions():
    """
    打印使用说明
    
    提示用户如何正确使用生成的标定板。
    """
    logger.info("\n" + "=" * 60)
    logger.info("⚠️  重要提示")
    logger.info("=" * 60)
    logger.info("• 请按 1:1 比例打印，不要缩放！")
    logger.info("• 打印后请测量实际尺寸确保准确性")
    logger.info("• 将标定板贴在平面上，避免弯曲")
    logger.info("• 保持标定板表面清洁，避免反光")
    logger.info("=" * 60 + "\n")


def main():
    """
    主程序入口
    """
    logger.info("=" * 60)
    logger.info("ChArUco标定板生成器")
    logger.info("=" * 60)
    
    try:
        # 生成标定板
        save_path = generate_charuco_board_image()
        
        # 打印使用说明
        print_usage_instructions()
        
        logger.info("✅ 生成完成！")
        
    except Exception as e:
        logger.error(f"❌ 生成失败: {e}")
        raise


if __name__ == "__main__":
    main()
