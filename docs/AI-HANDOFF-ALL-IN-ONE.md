# Foundationpose_for_VR 项目总交接文档（唯一入口）

更新时间：2026-04-09

本文件是项目唯一长期维护的 AI 接手文档。历史会话文档已融合并清理，后续请只更新本文件。

## 1. 项目目标

- 目标：实现 VR 场景中的实时 6D 位姿估计与可视化。
- 现阶段主线：双目图像输入 -> 2D 分割 -> 双目深度 -> 6D 位姿（FoundationPose）。
- 核心原则：链路可运行优先、模块解耦优先、可观测性优先。

## 2. 当前可运行主链路

### 2.1 Quest 链路

- 入口脚本：src/pipeline/quest_pipeline.py
- 数据流：QuestStereoCamera -> Yoloe26Masker -> FastFoundationStereoRealtime -> FoundationPoseEstimator
- 可选增强：CutieTracker 用于 2D 跟踪引导

### 2.2 RealSense 链路

- 入口脚本：src/pipeline/realsense_pipeline.py
- 数据流：RealSenseCamera -> Yoloe26Masker -> FastFoundationStereoRealtime -> FoundationPoseEstimator
- 可选增强：CutieTracker 用于 2D 跟踪引导

### 2.3 分阶段调试（两条链路一致）

- stage 1：仅输入
- stage 2：输入 + YOLO 分割
- stage 3：输入 + YOLO + 深度
- stage 4：全链路（含 FoundationPose）

## 3. 代码结构（当前有效）

### 3.1 核心模块目录

- src/modules/realsense.py
- src/modules/quest_stereo.py
- src/modules/yoloe26.py
- src/modules/fast_foundationstereo.py
- src/modules/foundationpose.py
- src/modules/cutie.py

### 3.2 编排层目录

- src/pipeline/realsense_pipeline.py
- src/pipeline/quest_pipeline.py

### 3.3 已清理的旧入口/旧链路（不要恢复）

- src/pose_server.py
- src/pose_tracker_api.py
- src/vpt_cli.py
- src/VOT.py
- src/zmq_utils/timing.py
- src/zmq_utils/latency.py

## 4. 运行方式（常用）

在 Foundationpose_for_VR 目录执行：

- Quest：pixi run python .\src\pipeline\quest_pipeline.py
- RealSense：pixi run python .\src\pipeline\realsense_pipeline.py

窗口热键（main 示例）：

- 1/2/3/4：切换阶段
- r：重置跟踪状态
- q 或 ESC：退出

## 5. 调试信息与统计口径（当前定义）

### 5.1 HUD

- HUD 使用统一函数 \_draw_hud（已合并原先 \_draw_text 与 \_draw_hud_lines）。
- 文本支持按窗口宽度自动换行，避免遮挡。
- 首行 fps 为实时帧率（不是累计均值）。

### 5.2 FPS 定义

- 实时 fps：基于相邻帧间隔 1/dt 计算，并用 EMA 平滑。
- 统计日志字段：
  - rt_fps：实时平滑 fps
  - window_fps：统计窗口 fps（按 stats_interval 计算）

## 6. 关键工程约束

- 不要把网络地址写死在代码里，配置应可外部输入并可持久化。
- 数据提供方与发送方保持解耦，便于静态图测试与模块复用。
- API 改动优先保持最小闭环，避免跨模块大改。
- 可观测性必须保留：阶段、检测数、深度有效率、耗时分项、fps。

## 7. 历史问题与已落地修复（融合摘要）

- 处理分辨率与标定分辨率不一致会导致位姿系统性误差，已引入中心裁剪+缩放 K 映射策略。
- FoundationPose 导入符号与 Utils 冲突问题已做隔离处理。
- debug_dir 空值崩溃、模块同名遮蔽、导出符号不稳定等问题均已修复。
- Packed 单图策略存在质量/稳定性风险，调试基线优先 Dual 思路。

## 8. 工作日志（按日期）

### 8.1 今日工作总结（2026-04-06）

#### 8.1.1 代码清理与架构收敛

- 清理了历史旧入口与过期测试依赖，主入口收敛到 pipeline 目录。
- 统一改为“模块 API + pipeline 编排”模式，减少分散脚本。

#### 8.1.2 API 风格统一

- 移除了 Config 类中间层，改为直接参数调用。
- 统一了 modules 与 pipeline 的类成员声明方式：
  - 类体显式成员清单
  - 成员分组
  - 中文注释
- **init** 文档补齐并精简，默认值尽量下沉到类体。

#### 8.1.3 初始化与执行策略修正

- start() 职责收敛为启动与状态重置，不再做重初始化杂项。
- Quest 链路保留在 **init** 完成 K 与 PoseEstimator 初始化。
- RealSense 链路保留 run() 首帧懒初始化路径。

#### 8.1.4 调试显示修复

- 修复了 Quest/RealSense 窗口 HUD 文本溢出问题。
- 将 fps 从累计均值改为实时帧率显示。
- 合并 HUD 绘制方法，减少重复逻辑并统一样式。

### 8.2 今日工作总结（2026-04-09）

本次工作主要围绕「网络协议收敛 + MessagePack 全链路统一 + Windows/pixi 环境稳定化 + ONNX 导出回归验证」。

#### 8.2.1 协议与通信层重构（Python）

- 统一为单帧 payload 模式，移除多段 parts/multipart 兼容路径。
- `PayloadSender`/`PayloadReceiver` 改为 single-payload 主接口：
  - sender 统一发送 `bytes`；
  - receiver 统一接收 `bytes`；
  - Quest 接收侧启用 conflate/latest 语义，保留“只消费最新帧”的实时策略。
- 删除旧统计与旧模式诊断路径：
  - 移除 drained、`_sender_meta_count`、`_sender_no_meta_count` 等旧口径；
  - 移除 `_infer_payload_mode` 及其调用。

#### 8.2.2 序列化统一为 MessagePack（Python）

- 以下消息模型已改为 MessagePack：
  - `src/zmq_utils/payload/message/pose.py`
  - `src/zmq_utils/payload/message/rgbd.py`
  - `src/zmq_utils/payload/message/stereo.py`
- `PoseMsg` API 已收口为仅保留 `serialize/deserialize`，不再保留 JSON 路径。
- 编解码层同步改造：
  - `src/zmq_utils/payload/encoder/base_encoder.py`
  - `src/zmq_utils/payload/decoder/base_decoder.py`
  - `src/zmq_utils/payload/encoder/pose_encoder.py`
  - `src/zmq_utils/payload/decoder/pose_decoder.py`
  - `src/zmq_utils/payload/encoder/rgbd_encoder.py`
  - `src/zmq_utils/payload/decoder/rgbd_decoder.py`
  - `src/zmq_utils/payload/decoder/stereo_decoder.py`

#### 8.2.3 Unity 侧同步（C#）

- Unity 网络层已同步到 MessagePack：
  - `Assets/Scripts/Net/Payload/Message/PoseMsg.cs`
  - `Assets/Scripts/Net/Payload/Message/RGBDMsg.cs`
  - `Assets/Scripts/Net/Payload/Message/QuestStereoMsg.cs`
- Unity 编解码器与收发器同步更新：
  - `Assets/Scripts/Net/Communicate/PayloadSender.cs`
  - `Assets/Scripts/Net/Communicate/PayloadReceiver.cs`
  - `Assets/Scripts/Net/Payload/Encoder/*.cs`
  - `Assets/Scripts/Net/Payload/Decoder/*.cs`
- 删除旧 `TryDeserialize` 路径，反序列化入口收敛为 `Deserialize`。

#### 8.2.4 Windows + pixi 环境排障与收敛

- 目标：移除临时 pip 安装痕迹，恢复“全部由 pixi 管理”。
- 处理过程：
  - 清理并重建 `.pixi/envs/default`；
  - 处理 Windows 文件锁问题（VS Code Black Formatter 的 Python LSP 占用环境文件）；
  - 依赖声明确认：`pixi.toml` 现包含 `msgpack`、`onnx`、`pillow`（均由 pixi 解析与安装）。
- 为解决 Windows 下 `torchvision -> PIL.Image` 导入链不稳定问题：
  - 在 `Fast-FoundationStereo/core/foundation_stereo.py` 顶部预加载 Pillow。

#### 8.2.5 ONNX 导出回归验证（已通过）

- 验证命令（示例）：
  - `pixi run python scripts/make_onnx.py --model_dir weights/20-30-48/model_best_bp2_serialize.pth --save_path output/`
- 产物已生成：
  - `Fast-FoundationStereo/output/feature_runner.onnx`
  - `Fast-FoundationStereo/output/post_runner.onnx`
  - `Fast-FoundationStereo/output/onnx.yaml`
- 备注：脚本当前对 `save_path` 参数较敏感，建议使用目录形式并带尾部斜杠（如 `output/`）。

#### 8.2.6 运行状态与观察

- `src/pose_server.py` 在当前环境可持续运行并输出统计日志。
- 统计字段中 `sender_raw` 数值很大属于跨进程/跨设备时钟基准差异，不应直接解释为真实网络延迟；优先关注 `sender_est` 与趋势变化。

#### 8.2.7 当前改动规模（Foundationpose_for_VR 子仓）

- 变更文件：19 个。
- 变更量：457 insertions / 612 deletions。
- 主体为协议收敛与旧兼容路径删除，属于“行为统一 + 技术债清理”型改造。

## 9. 后续 AI 接手建议

建议按以下顺序开始：

1. 先确认链路可运行，再做调参。
2. 用 stage 2/3/4 逐段定位问题，不要一上来全链路盲调。
3. 先用固定基线场景验证：
   - 掩码质量
   - 深度有效率
   - 位姿稳定性
4. 任何协议或显示字段变更，都同步更新本文件第 5 节。
5. 若改动网络协议，必须 Python 与 Unity 同步改，避免单边兼容逻辑回流。
6. 若需要重建 pixi 环境，先确认无 Python LSP/formatter 进程占用 `.pixi/envs/default`。
7. ONNX 导出回归建议固定检查三项：

- `feature_runner.onnx` 是否生成
- `post_runner.onnx` 是否生成
- `onnx.yaml` 是否生成

## 10. 文档维护规则

- 仅维护本文件，避免再次产生多份历史交接文档。
- 每次完成较大改动后，只需更新：
  - 第 5 节（统计/调试口径）
  - 第 8 节（当日总结）
  - 第 9 节（下一步建议）
