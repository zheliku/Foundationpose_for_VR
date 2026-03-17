# SAM3 使用笔记

**项目地址**：https://github.com/meta-segmentation/sam3 (基于 Meta 的 SAM 系列)

## 1. SAM3 与 SAM2 的主要区别

| 特性 | SAM2 | SAM3 |
|------|------|------|
| 输入提示 | 点击点/框 (Point/Box) | **文本提示 (Text Prompt)** |
| 交互方式 | 手动指定前景/背景点 | 自然语言描述目标 |
| 使用场景 | 精确分割已知位置的物体 | 根据语义描述自动定位分割 |
| 模型架构 | SAM2ImagePredictor | Sam3Processor |

**SAM3 的核心优势**：无需手动点击，直接用文本描述（如 "white cube"、"red ball"）即可自动检测和分割目标。

---

## 2. 环境搭建

### 2.1 依赖安装

```toml
# pixi.toml 中添加以下依赖
[pypi-dependencies]
torch = { version = ">=2.8.0", index = "https://download.pytorch.org/whl/cu129" }
torchvision = { version = ">=0.23.0", index = "https://download.pytorch.org/whl/cu129" }
pillow = ">=12.0.0, <13"
matplotlib = ">=3.10.7, <4"
opencv-python = ">=4.11.0.86, <5"
```

### 2.2 模型权重下载

将 SAM3 模型权重放置到以下路径：
```
sam3/assets/sam3_ckpt/sam3.pt
```

---

## 3. 基本使用示例

### 3.1 导入必要模块

```python
from PIL import Image
import torch
import numpy as np
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
```

### 3.2 加载模型

```python
# 使用本地模型检查点
checkpoint_path = "sam3/assets/sam3_ckpt/sam3.pt"

# 构建模型（禁止从 HuggingFace 下载）
model = build_sam3_image_model(checkpoint_path=checkpoint_path, load_from_HF=False)

# 创建处理器（可设置置信度阈值）
processor = Sam3Processor(model, confidence_threshold=0.5)
```

**Sam3Processor 参数说明**：
- `model`: 加载的 SAM3 模型
- `resolution`: 输入分辨率，默认 1008
- `device`: 计算设备，默认 "cuda"
- `confidence_threshold`: 检测置信度阈值（0-1），越高越严格

### 3.3 加载图像

```python
# 读取图像（支持 PIL.Image 或 numpy 数组）
image = Image.open("path/to/image.png")

# 设置图像到处理器
inference_state = processor.set_image(image)
```

### 3.4 文本提示检测

```python
# 使用文本提示进行检测（这是 SAM3 的核心功能！）
output = processor.set_text_prompt(
    state=inference_state, 
    prompt="white cube"  # 描述目标物体的文本
)

# 获取检测结果
masks = output["masks"]   # 分割掩码，形状: (N, H, W)
boxes = output["boxes"]   # 边界框，形状: (N, 4)，格式: [x0, y0, x1, y1]
scores = output["scores"] # 置信度分数，形状: (N,)

print(f"检测到 {len(masks)} 个物体")
print(f"置信度分数: {scores}")
```

---

## 4. 完整示例代码

```python
"""
SAM3 图像分割示例
演示如何使用 SAM3 基于文本提示进行自动目标检测与分割
"""

import matplotlib
matplotlib.use("Agg")  # 非交互式后端
import matplotlib.pyplot as plt

import torch
import numpy as np
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import plot_results

# ==================== 1. 加载模型 ====================
checkpoint_path = "sam3/assets/sam3_ckpt/sam3.pt"
model = build_sam3_image_model(checkpoint_path=checkpoint_path, load_from_HF=False)
processor = Sam3Processor(model, confidence_threshold=0.5)

# ==================== 2. 加载图像 ====================
image = Image.open("path/to/your/image.png")
inference_state = processor.set_image(image)

# ==================== 3. 文本提示检测 ====================
output = processor.set_text_prompt(state=inference_state, prompt="white cube")
masks, boxes, scores = output["masks"], output["boxes"], output["scores"]

print(f"检测到 {len(masks)} 个物体")
print(f"置信度分数: {scores}")

# ==================== 4. 获取最佳 Mask ====================
if len(masks) > 0:
    # 选择置信度最高的 mask
    best_idx = scores.argmax().item()
    best_mask = masks[best_idx]
    best_score = scores[best_idx].item()
    
    print(f"最佳 mask 索引: {best_idx}, 分数: {best_score:.4f}")
    
    # 转换为 numpy 数组
    if isinstance(best_mask, torch.Tensor):
        mask_np = best_mask.cpu().numpy()
    else:
        mask_np = np.array(best_mask)
    
    # 如果是 3D 的 (1, H, W)，去掉第一个维度
    if len(mask_np.shape) == 3:
        mask_np = mask_np.squeeze(0)
    
    # 转换为二值图像
    mask_binary = (mask_np > 0.5).astype(np.uint8) * 255
    
    # 保存 mask
    from PIL import Image as PILImage
    mask_image = PILImage.fromarray(mask_binary, mode="L")
    mask_image.save("output_mask.png")
    print("Mask 已保存到: output_mask.png")

# ==================== 5. 可视化结果 ====================
plot_results(image, inference_state)
plt.savefig("output_result.png", dpi=150, bbox_inches="tight")
print("可视化结果已保存到: output_result.png")
```

---

## 5. 高级功能

### 5.1 调整置信度阈值

```python
# 方法1：在创建处理器时设置
processor = Sam3Processor(model, confidence_threshold=0.7)

# 方法2：动态调整阈值
processor.set_confidence_threshold(0.8, state=inference_state)
```

**阈值建议**：
- `0.5`: 较宽松，可能检测到不相关物体
- `0.6-0.7`: 推荐范围，平衡检测率和准确性
- `0.8+`: 严格模式，只有高置信度的检测才会通过

### 5.2 添加几何提示（辅助框）

```python
# 添加一个辅助边界框提示（格式：[center_x, center_y, width, height]，归一化到 0-1）
output = processor.add_geometric_prompt(
    box=[0.5, 0.5, 0.3, 0.3],  # 中心点 (0.5, 0.5)，宽高各 0.3
    label=True,  # True=正样本框，False=负样本框
    state=inference_state
)
```

### 5.3 重置所有提示

```python
# 清除所有文本和几何提示
processor.reset_all_prompts(state=inference_state)
```

---

## 6. SAM2 vs SAM3 代码对比

### SAM2 方式（点击提示）

```python
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))
predictor.set_image(image)

# 需要手动指定点坐标
input_point = np.array([[300, 275]])  # 手动点击位置
input_label = np.array([1])           # 1=前景

masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)
```

### SAM3 方式（文本提示）

```python
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

model = build_sam3_image_model(checkpoint_path=checkpoint_path)
processor = Sam3Processor(model)
inference_state = processor.set_image(image)

# 直接用文本描述，无需手动点击！
output = processor.set_text_prompt(
    state=inference_state, 
    prompt="white cube"  # 文本描述
)
masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
```

---

## 7. 常见问题

### Q1: cv2.imshow() 调用后程序崩溃（段错误）

**原因**：SAM3 内部使用的 `decord` 库在模块加载时与 OpenCV 冲突。

**解决方案**：修改 `sam3/sam3/train/data/sam3_image_dataset.py`，将 `decord` 导入改为延迟加载：

```python
# 移除模块级导入
# from decord import cpu, VideoReader  # 删除这行

# 在 _load_images() 方法中使用延迟导入
if ".mp4" in path and path[-4:] == ".mp4":
    try:
        from decord import cpu, VideoReader
    except ImportError:
        raise ImportError("decord is required for video loading")
    # ... 使用 decord
```

### Q2: 检测结果不准确

**解决方案**：
1. 调整置信度阈值：`--sam3_confidence_threshold 0.8`
2. 使用更精确的文本描述，如 "small white plastic cube" 而不是 "cube"
3. 确保目标物体在画面中清晰可见

### Q3: 模型加载慢

第一次加载时需要初始化 CUDA 和编译内核，后续使用会快很多。可以考虑预加载模型并复用。

---

## 8. 输出结果说明

| 输出 | 类型 | 形状 | 说明 |
|------|------|------|------|
| `masks` | Tensor | (N, H, W) | N 个检测到的分割掩码 |
| `boxes` | Tensor | (N, 4) | 边界框，格式 [x0, y0, x1, y1] |
| `scores` | Tensor | (N,) | 每个检测的置信度分数 |

---

**作者**：基于 Meta SAM 系列  
**更新日期**：2026-02-04
