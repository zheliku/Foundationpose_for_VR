import matplotlib

matplotlib.use("Agg")  # 使用非交互式后端，避免Qt版本冲突
import matplotlib.pyplot as plt

import torch
import os

#################################### For Image ####################################
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import draw_box_on_image, normalize_bbox, plot_results

# 使用本地下载的模型检查点
checkpoint_path = os.path.join(os.path.dirname(__file__), "assets/sam3_ckpt/sam3.pt")

# Load the model (使用本地路径，禁止从HuggingFace下载)
model = build_sam3_image_model(checkpoint_path=checkpoint_path, load_from_HF=False)
processor = Sam3Processor(model)
# Load an image
image = Image.open(
    "/home/zheliku/projects/Foundationpose_for_VR/data/offline/cube/rgb/00004.png"
)
inference_state = processor.set_image(image)
# Prompt the model with text
output = processor.set_text_prompt(state=inference_state, prompt="white cube")

# Get the masks, bounding boxes, and scores
masks, boxes, scores = output["masks"], output["boxes"], output["scores"]

print(f"Found {len(masks)} object(s)")
print(f"Scores: {scores}")
print(f"Masks shape: {masks.shape}")  # 查看mask的形状

# ==================== 保存 Mask 图像 ====================
import numpy as np
from PIL import Image as PILImage

# 选择置信度最高的mask (或者你可以根据需要选择)
if len(masks) > 0:
    # 获取最高分数的索引
    best_idx = scores.argmax().item()
    best_mask = masks[best_idx]  # shape: (H, W) 或 (1, H, W)
    best_score = scores[best_idx].item()

    print(f"\nBest mask index: {best_idx}, score: {best_score:.4f}")
    print(f"Best mask shape: {best_mask.shape}")

    # 将mask转换为numpy数组
    if isinstance(best_mask, torch.Tensor):
        mask_np = best_mask.cpu().numpy()
    else:
        mask_np = np.array(best_mask)

    # 如果mask是3D的 (1, H, W)，去掉第一个维度
    if len(mask_np.shape) == 3:
        mask_np = mask_np.squeeze(0)

    # 转换为二值mask图像 (0 或 255)
    mask_binary = (mask_np > 0.5).astype(np.uint8) * 255

    # 保存mask图像
    mask_image = PILImage.fromarray(mask_binary, mode="L")  # 'L' 表示灰度图
    mask_save_path = "/home/zheliku/projects/Foundationpose_for_VR/sam3/output_mask.png"
    mask_image.save(mask_save_path)
    print(f"Best mask saved to: {mask_save_path}")

    # 如果你想保存所有检测到的mask，可以合并它们
    # 或者分别保存每个mask
    combined_mask = np.zeros_like(mask_np, dtype=np.uint8)
    for i, m in enumerate(masks):
        if isinstance(m, torch.Tensor):
            m_np = m.cpu().numpy()
        else:
            m_np = np.array(m)
        if len(m_np.shape) == 3:
            m_np = m_np.squeeze(0)
        combined_mask = np.maximum(combined_mask, (m_np > 0.5).astype(np.uint8))

    combined_mask_image = PILImage.fromarray(combined_mask * 255, mode="L")
    combined_save_path = (
        "/home/zheliku/projects/Foundationpose_for_VR/sam3/output_mask_combined.png"
    )
    combined_mask_image.save(combined_save_path)
    print(f"Combined mask saved to: {combined_save_path}")

# ==================== 可视化结果 ====================
plot_results(image, inference_state)

# 保存可视化结果图片
plt.savefig(
    "/home/zheliku/projects/Foundationpose_for_VR/sam3/output_result.png",
    dpi=150,
    bbox_inches="tight",
)
print(
    "Visualization saved to: /home/zheliku/projects/Foundationpose_for_VR/sam3/output_result.png"
)
