# Foundationpose_for_VR 模块化 API / CLI 规划（V2）

本文档对应全新模块化实现：`src/vpt_modules/*` 与 `src/vpt_cli.py`。

目标：

- 不依赖旧测试脚本作为入口。
- 每个模型模块独立输入/输出契约。
- 支持“一行代码调用”与“一条 CLI 命令运行”。

---

## 1. 模块目录

```text
src/
  vpt_modules/
    types.py               # 统一数据结构
    contracts.py           # 统一协议接口（Protocol）
    pipeline.py            # 流水线编排（首帧分割，后续追踪）
    sensors/
      realsense_rgbd.py    # RealSense RGBD 数据源
    segmenters/
      yoloe26.py           # YOLOE26 分割模块
    depth/
      ffs.py               # Fast-FoundationStereo 深度模块
    pose/
      foundationpose_estimator.py # FoundationPose 模块
  vpt_cli.py               # CLI 入口
```

---

## 2. 统一输入输出契约

核心数据结构定义在 `vpt_modules/types.py`：

- `RGBDFrame(color_bgr, depth_m, timestamp_s)`
- `MaskResult(mask_u8, score, label)`
- `DepthResult(depth_m, valid_ratio, meta)`
- `PoseResult(pose_4x4, vis_bgr)`
- `PipelineResult(status, pose, vis_bgr, mask_u8, debug)`

流水线状态：

- `detecting`：还未拿到有效首帧掩码。
- `tracking`：已初始化 FoundationPose 并持续追踪。
- `lost`：追踪失败，可按策略重置并重新检测。

---

## 3. 各模块 API（可一行调用）

## 3.1 RealSense RGBD

文件：`vpt_modules/sensors/realsense_rgbd.py`

一行调用示例：

```python
frame = RealSenseRGBDSource(RealSenseConfig()).read()
```

完整建议：

```python
source = RealSenseRGBDSource(RealSenseConfig(width=640, height=480, fps=30))
source.start()
frame = source.read()  # RGBDFrame | None
source.stop()
```

---

## 3.2 YOLOE26 分割（首帧掩码）

文件：`vpt_modules/segmenters/yoloe26.py`

一行调用示例：

```python
mask_result = Yoloe26Segmenter(cfg).segment(frame)
```

输入：`RGBDFrame`（用其中 `color_bgr`）  
输出：`MaskResult(mask_u8, score, label)`

---

## 3.3 Fast-FoundationStereo 深度

文件：`vpt_modules/depth/ffs.py`

一行调用示例：

```python
depth_result = FastFoundationStereoDepth(cfg).estimate(left_bgr, right_bgr, fx, baseline_m)
```

输入：左右图 + `fx` + baseline  
输出：`DepthResult(depth_m, valid_ratio, meta)`

说明：该模块已按导入冲突风险做隔离处理（`Utils` 同名冲突）。

---

## 3.4 FoundationPose 位姿

文件：`vpt_modules/pose/foundationpose_estimator.py`

一行调用示例：

```python
init_pose = FoundationPoseEstimator(cfg).initialize(frame, mask_u8)
```

后续追踪：

```python
track_pose = estimator.track(frame)
```

---

## 3.5 首帧掩码流水线（你现在要的实验）

文件：`vpt_modules/pipeline.py`

策略：

- 第一次成功分割时：`YOLOE26 -> mask -> FoundationPose.initialize`
- 后续帧：`FoundationPose.track`

一行调用示例：

```python
result = FirstFrameMaskPipeline(segmenter, pose_estimator).process(frame)
```

---

## 4. CLI 设计（类 CLI，一条命令跑）

入口：`src/vpt_cli.py`

命令：

```bash
pixi run python src/vpt_cli.py run-rgbd-yoloe26-fp \
  --mesh_path data/online/cube/mesh/cube.stl \
  --yoloe_model checkpoints/yoloe-26l-seg.pt \
  --prompt "white block"
```

说明：

- 这条命令对应你的 A/B 需求：`RealSense RGBD + YOLOE26 首帧掩码 + FoundationPose`。
- 不走旧测试脚本入口。

---

## 5. 推荐扩展（下一阶段）

建议再补三个子命令（接口已可扩展）：

1. `run-stereo-ffs-fp`  
   `Stereo -> FFS depth -> FoundationPose`

2. `bench-mask-init`  
   对比 `SAM3 vs YOLOE26` 首帧初始化耗时与成功率。

3. `record-rgbd-session`  
   统一录制 `color/depth/cam_k/timestamp`，便于离线回放与可复现实验。

---

## 6. 使用边界

- V2 模块是新入口，不会替换你当前线上脚本。
- 你可以并行保留旧链路，逐步迁移。
- 当你确认 YOLOE26 首帧质量达标，再把老入口切换到 V2 即可。
