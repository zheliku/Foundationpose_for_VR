# AI 会话交接记录（2026-03-30）

本文档汇总 2026-03-30 当天在 `src/modules` 下完成的模块化改造、报错定位与修复进度，供后续 AI 直接接手。

---

## 1. 当日目标

1. 将实时链路核心能力模块化：
   - RealSense 采集 API
   - Fast-FoundationStereo 深度估计 API
   - YOLOE-26 掩码 API
   - FoundationPose 位姿估计 API
   - Cutie 2D 跟踪 API
2. 保持接口清晰、实现简洁、便于后续组合：
   - `realsense -> yoloe26 -> fast-foundationstereo -> foundationpose`
   - 可选 `cutie` 对 `foundationpose` 做 2D 引导优化

---

## 2. 新增/重写模块

## 2.1 RealSense 模块

- 文件：`src/modules/realsense.py`
- 核心类：`RealSenseCamera`
- 核心能力：
  - `start()/stop()`
  - `get_stereo_frames()` 返回左右 IR
  - `get_aligned_rgbd_frames()` 返回对齐 RGBD
- 说明：
  - 按用户要求，不新增额外 API（保持双方法调用模式）。
  - 深度显示示例采用官方风格 `alpha=0.03` 伪彩。
  - `pipeline/config` 已改为公有字段，外部可访问。

## 2.2 Fast-FoundationStereo 模块

- 文件：`src/modules/fast-foundationstereo.py`
- 核心类：`FastFoundationStereoRealtime`
- 核心能力：
  - 实时输入左右图 + `fx/baseline`
  - 输出米制深度图
- 本日精简点：
  - 删除额外统计/可视化辅助函数（只保留核心）。
  - `main` 取消 argparse，改为顶部直配参数。
  - `predict_depth` 仅返回深度（耗时在外部统计）。
  - `autocast` 改为新接口优先（兼容旧版回退）。

## 2.3 YOLOE 模块

- 文件：`src/modules/yoloe26.py`
- 核心类：`Yoloe26Masker`
- 核心能力：
  - 输入图像 + prompt
  - 输出 overlay + 黑白 mask + 检测数 + 推理耗时
- 关键修复：
  - 增加 `mobileclip2_path` 配置项。
  - 启动时将本地 `mobileclip2_b.ts` 注册到 Ultralytics 可检索路径。
  - 修复 `SETTINGS["weights_dir"]` 类型错误（必须是 `str`）。

## 2.4 FoundationPose 模块

- 文件：`src/modules/foundationpose.py`
- 核心类：`FoundationPoseEstimator`
- 核心能力：
  - `register(rgb, depth, mask)`
  - `track(rgb, depth)`
  - `estimate(...)`
  - `adjust_pose_to_image_point(x, y)`（接收外部2D引导）
  - `visualize_pose(...)`
- 架构约束：
  - 不依赖 `cutie.py`，保持解耦。

## 2.5 Cutie 模块

- 文件：`src/modules/cutie.py`
- 核心类：`CutieTracker`
- 核心能力：
  - `initialize(frame, init_mask/init_bbox)`
  - `track(frame)`
  - 输出 `bbox + mask`
- 架构约束：
  - 不依赖 `foundationpose.py`，保持解耦。

---

## 3. 当日关键报错与修复

## 3.1 `foundationpose.py` 导入失败

- 现象：`ModuleNotFoundError`（`FoundationPose.datareader`）
- 原因：脚本直跑与包导入层级不一致。
- 修复：
  - 导入改为“包导入优先 + 直导回退”双路径兼容。

## 3.2 `foundationpose.py` 初始 mask 为空

- 现象：`reader.get_mask(0)` 返回 `None`。
- 原因：离线目录没有 `masks/00000.png`，实际是 `0_mask.png`。
- 修复：
  - 增加回退读取 `data/offline/cube/0_mask.png`。

## 3.3 `foundationpose.py` dtype 冲突（反复出现）

- 现象：
  - `RuntimeError: expected mat1 and mat2 to have the same dtype, but got: float != double`
- 报错位置：`FoundationPose/Utils.py` 的 `K @ pts.T`。
- 结论：
  - FoundationPose 内部几何流程里 `pts` 走到了 double，`K` 若是 float32 会冲突。
- 已做修复：
  - `cam_k` 在封装层改为 `float64`。
  - mesh 处理后再强制一次 `vertices/normals -> float32`（保守处理）。
- 状态：
  - 已提交修复，需再次实机复验是否彻底消除。

## 3.4 `cutie.py` 导入冲突（反复出现）

- 现象：
  - `No module named 'cutie.inference'; 'cutie' is not a package`
- 根因：
  - 当前脚本名 `cutie.py` 与第三方包 `cutie` 同名，发生遮蔽。
  - 且 Cutie 内部使用 `from cutie...` 绝对导入链。
- 已做修复：
  - 先导入 `Cutie.cutie`，再桥接注册 `sys.modules['cutie'] = cutie_pkg`。
  - 之后统一用 `from cutie...` 导入，满足其内部绝对导入。
- 状态：
  - 修复已落地，需再次实机复验。

---

## 4. 当前模块状态（交接快照）

1. `realsense.py`：可用，接口稳定。
2. `fast-foundationstereo.py`：已精简为核心链路，可用。
3. `yoloe26.py`：已支持本地 `mobileclip2_b.ts`，路径可配置。
4. `foundationpose.py`：功能完整，主要待确认 dtype 冲突是否完全消除。
5. `cutie.py`：已做同名冲突桥接，待最终运行确认。

---

## 5. 下次 AI 建议执行顺序

1. 先验证 `foundationpose.py` 是否还报 `float != double`：
   - 命令：`pixi run python .\src\modules\foundationpose.py`
2. 再验证 `cutie.py` 是否导入通过：
   - 命令：`pixi run python .\src\modules\cutie.py`
3. 若两者都通过，开始做组合脚本（编排层，不改模块边界）：
   - `realsense + yoloe26 + fast-foundationstereo + foundationpose (+cutie)`

---

## 6. 已知非致命告警（可先忽略）

1. `QWindowsContext: OleInitialize() failed ... 0x80010106`
   - Qt/COM 线程模型警告，常见于 OpenCV Qt 后端，不一定致命。
2. `TORCH_CUDA_ARCH_LIST is not set`
   - 编译范围提示，非当前功能阻塞。
3. `torch.set_default_tensor_type() is deprecated`
   - 上游库内部警告，暂不影响当前功能验证。

---

## 7. 本日新增/重点文件清单

- `src/modules/realsense.py`
- `src/modules/fast-foundationstereo.py`
- `src/modules/yoloe26.py`
- `src/modules/foundationpose.py`
- `src/modules/cutie.py`

---

## 8. 备注

- 用户偏好：
  1. 配置尽量直观（示例中可接受直配参数）。
  2. 模块解耦（2D tracker 与 6D pose 分离）。
  3. 先保证链路跑通，再逐步做优雅化与性能优化。
