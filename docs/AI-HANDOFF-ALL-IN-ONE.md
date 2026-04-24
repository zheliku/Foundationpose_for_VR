# Foundationpose_for_VR 项目总交接文档（唯一入口）

更新时间：2026-04-24

本文件是项目唯一长期维护的 AI 接手文档。历史会话文档已融合并清理，后续请只更新本文件。

## 1. 项目目标

- 目标：实现 VR 场景中的实时 6D 位姿估计与可视化。
- 现阶段主线：双目图像输入 -> 2D 分割 -> 双目深度 -> 6D 位姿（FoundationPose）。
- 核心原则：链路可运行优先、模块解耦优先、可观测性优先。

## 2. 当前可运行主链路

### 2.1 Quest 链路

- 入口脚本：src/pipeline/quest_pipeline.py
- 数据流：QuestReceiver（双 topic SUB: quest_stereo + quest_camera_info） -> Yoloe26Masker -> FastFoundationStereoRealtime -> FoundationPoseEstimator
- 可选增强：CutieTracker 用于 2D 跟踪引导
- 标定来源：默认从网络 camera_info 消息获取（camera_source=network），也可从本地缓存/标定目录加载（camera_source=local）

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
- src/modules/quest_io.py
- src/modules/yoloe26.py
- src/modules/fast_foundationstereo.py
- src/modules/foundationpose.py
- src/modules/cutie.py

### 3.2 编排层目录

- src/pipeline/realsense_pipeline.py
- src/pipeline/quest_pipeline.py

### 3.3 已清理的旧入口/旧链路（不要恢复）

- src/pose_tracker_api.py
- src/vpt_cli.py
- src/VOT.py
- src/zmq_utils/timing.py
- src/zmq_utils/latency.py
- src/modules/quest_stereo.py（已重命名为 quest_io.py）
- src/modules/quest_receiver.py（已重命名为 quest_io.py）
- Assets/Scripts/Net/Payload/Encoder/StaticStereoEncoder.cs（旧测试用编码器，已删除）

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

### 5.3 Quest 双 Topic 通信口径（当前定义）

- Unity 侧 PayloadSender 统一使用 PUB 模式，支持多个 SenderEntry（encoder + topic + targetFps）。
- Unity 侧 PayloadReceiver 统一使用 SUB 模式，支持多个 ReceiverEntry（topic + decoder），后台线程 drain 时按 topic 分别缓存最新帧。
- Python 侧 PayloadSender 统一 PUB 模式，send_payload 必须指定 topic。
- Python 侧 PayloadReceiver 统一 SUB 模式：
  - recv_all_latest_by_topic：按 topic 分别 drain，返回 {topic: payload} 字典，适合多 topic 场景。
  - recv_frame_latest：不区分 topic drain，仅适合单 topic 场景。
- Topic 名称约定：
  - quest_stereo：双目图像帧（高频，约 10fps）。
  - quest_camera_info：相机静态标定信息（低频，约 1fps，内容通常不变）。
  - pose：位姿结果（服务端发布，Unity 接收）。

### 5.4 Camera Info 缓存策略（当前定义）

- pose_server 每次接收到 quest_camera_info 消息后：
  - 与本地 camera_info_latest.json 比较。
  - 若内容不同 → 备份旧版为 camera_info_<timestamp>.json，保存新版。
  - 若内容相同 → 仅更新 _received_at 时间戳。
- Pipeline 启动时：
  - camera_source=local → 优先从缓存目录/标定目录加载，无缓存时等待网络。
  - camera_source=network → 仅等待网络 camera_info 消息。
- QuestStereoCalibration 构造方式：
  - QuestCameraInfoMsg.from_camera_info_msg()：从网络消息构造。
  - QuestStereoCalibration.from_local_json()：从本地 JSON 构造（旧工作流兼容）。

### 5.5 Fast-FoundationStereo TRT 产物与命名口径（当前定义）

- TRT/ONNX 命名已统一为参数标签，不再使用旧兼容命名：
  - tag 规则：`h{height}-w{width}-it{valid_iters}-md{max_disp}`
  - ONNX：`feature_runner-{tag}.onnx`、`post_runner-{tag}.onnx`
  - Engine：`feature_runner-{tag}.{platform}.{precision}.engine`、`post_runner-{tag}.{platform}.{precision}.engine`
- 兼容策略已收敛：
  - 不再生成 legacy alias；
  - 不再回退 legacy 文件名；
  - 运行时仅按新命名匹配。
- 配置口径已收敛：
  - 不再导出 `onnx.yaml`；
  - 运行时 TRT 配置由代码按当前参数构造（不依赖 YAML 元数据文件）。

### 5.6 相机封装边界口径（当前定义）

- RealSense：
  - `pyrealsense2` 仅允许在 `src/modules/realsense.py` 内部使用；
  - 调用方（modules/pipeline）统一通过 `RealSenseCamera.get_stereo_calibration()` 获取 `fx/fy/cx/cy/baseline/depth_scale`；
  - 不应在调用方直接访问 `camera.pipeline`。
- Quest：
  - 标定读取职责已下沉到 `src/modules/quest_io.py`；
  - 调用方统一通过 `QuestReceiver.get_calibration()` 获取标定对象；
  - `quest_pipeline.py` 不再维护本地 `_load_calibration`。

### 5.7 FFS 性能计时口径（当前定义）

- `FastFoundationStereoRealtime.predict_depth(return_timing=True)` 中：
  - `prep_ms`：输入预处理（numpy->tensor、缩放、维度转换等）；
  - `forward_ms`：网络前向推理；
  - `post_ms`：后处理（回 CPU、视差转深度、尺寸恢复等）；
  - `infer_ms`：`prep + forward + post` 总耗时。
- 因此，`infer_ms` 不是“纯模型 forward 时间”，与官方只统计 forward 的 profiling 脚本口径不同。

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

### 8.3 今日工作总结（2026-04-10）

本次工作主要围绕「Fast-FoundationStereo TRT 工程化收敛 + 命名规范统一 + 运行时结构简化」。

#### 8.3.1 TRT 构建链路收敛（pixi 一键化）

- `pixi.toml` 任务已整理为从 pth -> onnx -> engine 的一键链路（`pixi run build`）。
- 构建范围覆盖 `weights` 下 3 个模型目录，并按多参数组批量导出（当前基线包含 640x480、valid_iters=4/8）。
- 产物落地到各自权重目录，便于运行时按目录 + tag 精确选择。

#### 8.3.2 导出与构建脚本规范化

- `Fast-FoundationStereo/scripts/make_onnx.py` 与 `Fast-FoundationStereo/scripts/build_trt_engine.py` 已重构为清晰的 argparse 流程。
- 输出命名全部改为 tag 化；已移除 legacy alias 及旧命名兼容分支。
- `make_onnx.py` 已去除 `onnx.yaml` 生成逻辑，目录更干净。

#### 8.3.3 TRT 运行时口径收敛（Python）

- `src/modules/fast_foundationstereo.py` 已默认 TRT 优先，失败时按策略回退 PyTorch（`trt_strict` 可控）。
- 运行时仅匹配新命名产物，不再使用旧文件名回退。
- TRT 配置不再依赖 YAML 文件，改为运行时按参数构造，避免元数据文件分散。

#### 8.3.4 FFS 模块代码结构重构与精简

- `FastFoundationStereoRealtime` 已改为“统一调度 + 双后端类”结构：
  - `_PyTorchStereoBackend`：PyTorch 模型加载与推理；
  - `_TrtStereoBackend`：engine 匹配、runner 初始化与 TRT 推理。
- 清理了若干一跳封装与重复辅助逻辑（如 trivial sync 封装），并将 TRT 专属工具函数内聚到 TRT 后端类。
- 在不改变功能的前提下缩减行数并提升逻辑关联性与可读性。

#### 8.3.5 TRT demo 脚本同步

- `Fast-FoundationStereo/scripts/run_demo_tensorrt.py` 已同步到“无 YAML”模式。
- demo 通过参数直接构造 tag 并定位 engine，行为与主线口径一致。

### 8.4 今日工作总结（2026-04-24）

本次工作主要围绕「Quest 双 Topic 通信重构 + Sender/Receiver 多 topic 架构统一 + Camera Info 缓存策略 + 旧兼容清理」。

#### 8.4.1 新增 QuestCameraInfoMsg 消息（Python + C#）

- Python: src/zmq_utils/payload/message/quest_camera_info_msg.py
- C#: Assets/Scripts/Net/Payload/Message/QuestCameraInfoMsg.cs
- 包含左右目内参、畸变、基线、传感器分辨率、有效阵列区域、当前分辨率、帧率、镜头偏移位置/旋转等所有静态信息。

#### 8.4.2 新增编码器/解码器

- C# QuestCameraInfoEncoder: Assets/Scripts/Net/Payload/Encoder/QuestCameraInfoEncoder.cs
  - 带摘要缓存，仅当相机信息变化时重新编码。
- Python CameraInfoDecoder: src/zmq_utils/payload/decoder/camera_info_decoder.py

#### 8.4.3 PayloadSender/PayloadReceiver 多 Topic 重构（C# + Python）

- C# PayloadSender：改为 List<SenderEntry> 配置，每个 Entry 独立编码器+topic+帧率，统一 PUB 模式。
- C# PayloadReceiver：改为 List<ReceiverEntry> 配置，每个 Entry 独立 topic+解码器，统一 SUB 模式，按 topic 路由分发。
- Python PayloadSender：移除 PUSH 模式和 default_topic，统一 PUB，topic 必填。
- Python PayloadReceiver：移除 PULL 模式，统一 SUB，支持多 topics 订阅，recv_frame_latest 返回 (topic, payload) 元组。

#### 8.4.4 quest_stereo.py → quest_receiver.py 重命名与重构

- QuestStereoCamera → QuestReceiver，支持双 topic 接收。
- 新增 poll_all() 方法：轮询所有 topic 并更新缓存。
- 新增 get_camera_info() / get_calibration() 接口。
- QuestStereoCalibration 新增 from_camera_info_msg() 和 from_local_json() 类方法。
- 移除本地文件读取的 get_stereo_calibration(calib_dir) 旧接口。

#### 8.4.5 quest_pipeline.py 懒初始化改造

- 标定初始化改为懒初始化：camera_source=network 时等待网络 camera_info，camera_source=local 时优先本地。
- 新增 _init_from_calibration / _try_init_from_network 内部方法。
- 新增 --camera_source 和 --camera_cache_dir 命令行参数。

#### 8.4.6 pose_server.py 双 Topic 处理 + 缓存

- 运行时监控 camera_info 变化并自动保存到本地缓存目录。
- 内容变化时备份旧版，保存新版。
- PayloadSender.send_payload 调用已改为必须传入 topic 参数。

#### 8.4.7 旧兼容代码清理

- QuestStereoMsg：移除 packed_image_jpeg_legacy 字段和旧协议反序列化分支。
- StereoDecoder：移除旧协议解码分支。
- PayloadSender.cs：移除 PUSH 模式和 SenderUseTopicLegacyPrefKey 旧键。
- PayloadReceiver.cs：移除 PULL 模式和 RawPayloadEvent。
- 删除 StaticStereoEncoder.cs（旧测试用编码器，无引用）。
- 删除 quest_stereo.py（已重命名为 quest_receiver.py）。

### 8.5 今日工作总结补充（2026-04-10）

本次补充主要围绕「相机封装边界修正 + Pipeline 命名与职责收敛 + FFS 参数对齐 + 性能口径澄清」。

#### 8.4.1 RealSense 封装边界修正

- 在 `src/modules/realsense.py` 新增 `StereoCalibration` 与 `get_stereo_calibration()`。
- `src/modules/fast_foundationstereo.py` 的 main 示例已改为通过 `RealSenseCamera` 读取标定，移除直接 `pyrealsense2` 与 `camera.pipeline` 访问。
- `src/pipeline/realsense_pipeline.py` 同步改为通过 `get_stereo_calibration()` 构造 K，不再在 pipeline 层触达 RealSense SDK 细节。

#### 8.4.2 Quest 标定职责下沉与 API 统一

- 在 `src/modules/quest_stereo.py` 新增 `QuestStereoCalibration` 与 `QuestStereoCamera.get_stereo_calibration(calib_dir)`，含目录级缓存。
- `src/pipeline/quest_pipeline.py` 已删除本地 `_load_calibration`，统一改为调用相机模块提供的标定接口。
- 模块导出已更新，`src/modules/__init__.py` 可直接导入 `QuestStereoCalibration`。

#### 8.4.3 Pipeline 命名收敛

- `realsense_pipeline.py` 与 `quest_pipeline.py` 中统计函数名由 `_maybe_log_stats` 统一改为 `_log_stats_if_due`，语义更明确。

#### 8.4.4 Quest Pipeline FFS 参数与新版模块对齐

- `src/pipeline/quest_pipeline.py` 的 FFS 默认权重切到 `20-30-48`。
- 新增并透传以下参数到 `FastFoundationStereoRealtime`：
  - `--ffs_seed`
  - `--ffs_cudnn_benchmark`
  - `--ffs_use_trt`
  - `--ffs_trt_precision`
  - `--ffs_trt_strict`
  - `--ffs_trt_tag`
  - `--ffs_trt_platform_tag`
  - `--ffs_trt_feature_engine_path`
  - `--ffs_trt_post_engine_path`

#### 8.4.5 FFS 速度差异排查结论（重要）

- 对 `20-30-48 + valid_iters=4` 的慢速现象进行了专项排查。
- 关键结论：
  - 当前业务脚本显示的 `infer_ms` 为总耗时（prep+forward+post），不是纯 forward；
  - 文档表格是特定 profiling 条件下的基准值，不能直接与业务全链路 HUD 值等价比较；
  - 在当前本机环境使用官方 `scripts/profile_speed.py` 复测，同配置平均约 `39.1ms`（warmup 后），与业务实测量级一致。

### 8.6 今日工作总结（2026-04-24 补充）

本次补充主要围绕「修复双 Topic drain bug + 完善消息字段 + 新增编解码器 + 模块重命名」。

#### 8.6.1 修复 PayloadReceiver 跨 Topic Drain Bug

- Python PayloadReceiver 新增 `recv_all_latest_by_topic` 方法：按 topic 分别 drain，返回每个 topic 的最新 payload 字典。
- 原 `recv_frame_latest` 不区分 topic drain，标记为单 topic 场景专用。
- C# PayloadReceiver 的 ReceiveLoop 修复：drain 时将每条消息按 topic 分别存入 `_latestByTopic`，而非只保留最后一条。
- 根因：多 topic 场景下，`recv_frame_latest` 的跨 topic drain 会丢弃非最后 topic 的消息，导致 stereo 帧无法正常接收。

#### 8.6.2 QuestCameraInfoMsg 字段补充

- Python/C# 两侧新增字段：
  - `is_supported`（bool）：PassthroughCameraAccess.IsSupported。
  - `left_requested_width/height`、`right_requested_width/height`（int）：RequestedResolution。
- C# QuestCameraInfoEncoder 的 BuildMessage 和 ComputeDigest 同步更新。

#### 8.6.3 新增编解码器

- Python CameraInfoEncoder：src/zmq_utils/payload/encoder/camera_info_encoder.py
- C# CameraInfoDecoder：Assets/Scripts/Net/Payload/Decoder/CameraInfoDecoder.cs
  - 继承 BaseDecoder，触发 OnCameraInfoReceived 事件。

#### 8.6.4 quest_receiver.py → quest_io.py 重命名

- 文件重命名为 quest_io.py，类名 QuestReceiver 保持不变。
- poll_all 方法改用 `recv_all_latest_by_topic` 替代 `recv_frame_latest`，修复双 topic 消息丢失问题。
- modules/__init__.py 导出路径同步更新。

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
7. Quest 链路测试前，确认 Unity 场景中 PayloadSender 的 SenderEntry 列表已配置 quest_stereo 和 quest_camera_info 两个 topic。
8. camera_source=local 时确认 --calib_dir 或 --camera_cache_dir 指向有效目录。
9. 新增 topic 时，Python 侧 PayloadReceiver 的 topics 参数、QuestReceiver 的 topics 列表、以及 Unity 侧 SenderEntry/ReceiverEntry 需同步更新。
10. ONNX 导出回归建议固定检查两项：
    - `feature_runner.onnx` 是否生成
    - `post_runner.onnx` 是否生成
11. TRT 构建回归建议固定检查四项：
    - engine 文件是否带完整 tag（含 size/iters/max_disp）
    - 文件名是否包含平台标识（win/linux）
    - 文件名是否包含精度标识（fp16/fp32）
    - 运行时是否仅依赖新命名（无 legacy fallback）

## 10. 文档维护规则

- 仅维护本文件，避免再次产生多份历史交接文档。
- 每次完成较大改动后，只需更新：
  - 第 5 节（统计/调试口径）
  - 第 8 节（当日总结）
  - 第 9 节（下一步建议）
