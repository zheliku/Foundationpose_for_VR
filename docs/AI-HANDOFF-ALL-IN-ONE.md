# Foundationpose_for_VR AI 总交接文档（All In One）

更新时间：2026-04-03

本文档用于后续 AI 一次性熟悉项目背景、架构演进、历史会话与当前可运行链路。

## 0. 截至今日（2026-04-03）的最新工作总结

### 0.1 当前可运行主链路
- 主脚本已统一为 src/realsense_pipeline.py（单入口）。
- 运行链路：RealSense Stereo -> YOLOE26(mask) -> Fast-FoundationStereo(depth) -> FoundationPose(pose)。
- 在同一主流程内支持阶段切换调试（按键 1/2/3/4）、重置（r）、退出（q/ESC）。

### 0.2 模块与导入结构调整
- 已将 src/modules/fast-foundationstereo.py 重命名为 src/modules/fast_foundationstereo.py，避免连字符导入问题。
- 新增 src/modules/__init__.py，支持标准包导入。

### 0.3 FoundationPose 关键兼容修复
- 修复 dr 属性在 FoundationPose.estimater 中不稳定导出的问题（增加 Utils 回退解析）。
- 修复 trimesh_add_pure_colored_texture / draw_posed_3d_box / draw_xyz_axis 等符号导出不稳定问题（统一符号解析）。
- 修复 debug_dir=None 导致 os.makedirs 崩溃的问题（自动回落到有效目录）。
- 修复 FoundationPose 与 Fast-FoundationStereo 的同名 Utils 模块冲突（导入隔离，避免 compute_mesh_diameter 丢失）。

### 0.4 追踪稳定性增强（你当前重点）
- 已在 src/realsense_pipeline.py 接入 CutieTracker：
  - register 成功后，用同帧 mask 初始化 Cutie。
  - track 阶段每帧先跑 Cutie，使用 bbox 中心修正 FoundationPose 的 pose_last，再执行 track。
  - 当 Cutie 丢失但 YOLO mask 有效时，自动重初始化 Cutie。
- 新增统计项 cutie 耗时，便于观测性能代价。

### 0.5 已知现象与建议
- TORCH_CUDA_ARCH_LIST、cl.exe 编码告警通常不阻断主流程。
- FoundationPose 原生日志较多，建议后续做日志分级/降噪。
- 快速运动下仍建议继续加入自动重注册阈值策略（面积突变、深度有效率突降等）。

## 1. 合并来源清单
- AI-HANDOFF-PROJECT-INTRO.md
- AI-HANDOFF-PROMPT-TEMPLATE.md
- AI-HANDOFF-SESSION-2026-03-21.md
- AI-HANDOFF-SESSION-2026-03-22.md
- AI-HANDOFF-SESSION-2026-03-30.md
- MODULAR-API-CLI-PLAN.md
- system-architecture-2026-03-20.md
- UnityRuntimeInspector.md

> 下方为原始文档按文件顺序全量归并内容。

---

## 原始文档：AI-HANDOFF-PROJECT-INTRO.md

# VR Pose Tracking 项目 AI 对接总览

本文档用于让新接入的 AI 在最短时间内理解本项目现状、架构、协议、关键脚本和下一步工作。

---

## 1. 项目一句话

这是一个 **VR 位姿追踪系统**：

- 使用图像输入（历史为 RealSense RGBD，目标为 Quest 双目）
- 在服务器执行 FoundationPose 推理
- 输出位姿与可视化图像回传到 VR 端（Unity/Quest）

---

## 2. 当前阶段（非常重要）

## 2.1 已经跑通的链路（主链路）

`RealSense -> Python 服务器 -> Unity`

- RealSense 发送 RGBD（color+depth）到服务器。
- 服务器进行检测/追踪并输出 tracking payload。
- Unity 侧接收并解码，驱动显示与物体位姿。

## 2.2 正在迁移的目标链路

`Quest 双目 -> Python 服务器 -> Quest`

- 目标：Quest 发送左右图，服务器生成深度并输入 FoundationPose，再回传 pose。
- 当前落地：**仅完成双目两张静态图传输与简单显示**（用于链路验证）。
- 未完成：双目深度估计 + FoundationPose 全流程闭环。

---

## 3. 系统总体架构

项目采用“传输层 / 协议层 / 业务层”解耦：

1. 传输层（Transport）

- Sender: 只发 bytes[]
- Receiver: 只收 bytes[]
- 可选 topic（PUB/SUB）或无 topic（PUSH/PULL）

2. 协议层（Payload）

- Encoder: 业务对象 -> bytes[]
- Decoder: bytes[] -> 业务对象

3. 业务层（Tracking / Viewer）

- FoundationPose 推理
- Unity 显示与 pose 消费

---

## 4. 文字架构图

```text
[现有可运行链路]

RealSense Sender (Python)
  └─ RGBDEncoder -> PayloadSender(PUSH)
       tcp://server:5555
           ↓
Pose Server (Python)
  ├─ PayloadReceiver(PULL)
  ├─ RGBDDecoder
  ├─ PoseTracker (SAM3 + FoundationPose)
  ├─ TrackingEncoder
  └─ PayloadSender(PUB, topic=tracking)
       tcp://*:5556
           ↓
Unity (Receiver + Decoder)
  ├─ PayloadReceiver(SUB, topic=tracking)
  └─ TrackingDecoder
      ├─ OnImageReceived -> ImageViewer
      └─ OnPoseReceived  -> CubeFollow


[目标迁移链路（进行中）]

Quest (Unity)
  └─ QuestStereoEncoder / StaticStereoEncoder
      -> PayloadSender
           ↓
Server (Python, 待补全)
  ├─ PayloadReceiver
  ├─ StereoJpegDecoder
  ├─ Depth Estimation (双目 -> 深度)
  ├─ FoundationPose (color + depth)
  ├─ TrackingEncoder
  └─ PayloadSender
           ↓
Quest (Unity)
  └─ PayloadReceiver -> TrackingDecoder -> 可视化/姿态驱动
```

---

## 5. 通信协议定义（当前）

## 5.1 Stereo payload

- `parts[0]`: left_jpg
- `parts[1]`: right_jpg

## 5.2 RGBD payload

- `parts[0]`: color_jpg
- `parts[1]`: depth_png

## 5.3 Tracking payload

- `parts[0]`: phase_byte（0=detecting, 1=tracking）
- `parts[1]`: color_jpg
- `parts[2]`: pose_json（空字符串表示无位姿）

---

## 6. 关键目录与脚本索引

## 6.1 Unity 侧（显示与网络组件）

- `Assets/Scripts/Net/Communicate/Receiver/PayloadReceiver.cs`
  - Unity 通用接收器，线程收包、主线程派发 `OnPayloadReceived`。
- `Assets/Scripts/Net/Communicate/Sender/PayloadSender.cs`
  - Unity 通用发送器，固定帧率发送、IP 持久化、丢帧统计。
- `Assets/Scripts/Net/Payload/Encoder/`
  - `EncoderBase.cs`
  - `QuestStereoEncoder.cs`
  - `StaticStereoEncoder.cs`
  - `IntegerEncoder.cs`
- `Assets/Scripts/Net/Payload/Decoder/`
  - `RGBDDecoder.cs`
  - `TrackingDecoder.cs`
- `Assets/Scripts/Net/ReceiveData.cs`
  - `RawData`、`PoseData` 及事件类型定义。
- `Assets/Scripts/Net/ImageViewer.cs`
  - 图像显示与统计。
- `Assets/Scripts/CubeFollow.cs`
  - 消费 pose 更新物体变换。

## 6.2 Python 侧（推理与传输）

- `src/realsense_sender.py`
  - RealSense 采集并发送 RGBD。
- `src/pose_server.py`
  - 接收 RGBD -> FoundationPose 推理 -> 发布 tracking。
- `src/quest_stereo_receiver.py`
  - Quest 双目接收显示（当前用于验证双目链路）。
- `src/pose_tracker_api.py`
  - PoseTracker 封装（SAM3 + FoundationPose）。
- `src/zmq_utils/communicate/`
  - `sender.py` / `receiver.py`（统一传输层）。
- `src/zmq_utils/payload/`
  - `encoder.py` / `decoder.py`（统一协议层）。
- `src/zmq_utils/timing.py`, `src/zmq_utils/latency.py`
  - 性能统计与网络延迟探测。

---

## 7. 运行入口与端口约定

## 7.1 主要脚本

- 发送端（旧主链路）：`realsense_sender.py`
- 服务器：`pose_server.py`
- 双目验证接收端：`quest_stereo_receiver.py`

## 7.2 默认端口

- `5555`: RGBD 输入到服务器
- `5556`: tracking 输出到 Unity
- `5557`: Quest 双目测试链路（静态图接收）
- `5560`: latency probe

---

## 8. 最新架构决策（与历史版本区别）

1. Unity 接收器统一为单一 `PayloadReceiver`，不再维护多种 Receiver 继承类。
2. 解码逻辑统一放在 Decoder 中，Receiver 不做业务解析。
3. Python 传输统一为 `PayloadSender` / `PayloadReceiver`，topic/non-topic 通过参数选择。
4. Encoder/Decoder 命名统一，去掉旧的 Payload 冗余命名。
5. Scene 绑定改为可视化事件驱动（Inspector 可直接拖拽 `OnPayloadReceived`）。

---

## 9. 已知状态与风险

1. Quest 双目全链路（含深度估计）尚未完全接通。
2. `realsense_sender.py` / `pose_server.py` 在你的终端记录中近期退出码为 1，建议下次联调优先检查：
   - 设备可用性（RealSense/Quest 输入是否可读）
   - 模型与依赖路径
   - 端口占用与 topic 一致性
3. Python 某些类型检查提示可能来自语言服务器缓存或旧符号残留，运行逻辑未必受影响。

---

## 10. 给新 AI 的建议工作顺序

1. 先读本文件，再读：
   - `docs/system-architecture-2026-03-20.md`
   - `src/pose_server.py`
   - `Assets/Scripts/Net/Communicate/Receiver/PayloadReceiver.cs`
2. 明确当前任务是“旧链路维护”还是“新链路迁移”。
3. 若做新链路迁移，优先打通：
   - Stereo 输入稳定接收
   - 深度估计模块输出对齐到 FoundationPose 输入格式
   - Tracking 回传与 Quest 侧可视化闭环
4. 所有协议改动必须同步更新：
   - Python encoder/decoder
   - Unity decoder
   - 场景事件绑定

---

## 11. 项目目标（中期）

从“RealSense 驱动的服务器追踪”平滑迁移为“Quest 双目驱动的端到端实时追踪”，并保持：

- 可替换的数据源（静态图 / 相机流）
- 可扩展的 payload 协议
- 可观察的延迟与稳定性指标

这份文档即为后续 AI 接手时的统一入口。

---

## 12. 最近会话记录（按日期）

- `docs/AI-HANDOFF-SESSION-2026-03-21.md`
  - 记录了 2026-03-21 当天关于 Quest 双目 FPS 排查、发送端优化、统计口径修正、协议兼容（Dual/Packed）与稳定性回退决策。

建议后续 AI 阅读顺序：

1. 本文档（项目总览）
2. `docs/AI-HANDOFF-SESSION-2026-03-21.md`（最近上下文）
3. `docs/system-architecture-2026-03-20.md`（架构基线）


---

## 原始文档：AI-HANDOFF-PROMPT-TEMPLATE.md

# AI 对接 Prompt 模板（可直接复制）

> 用法：先把本模板发给新的 AI，再附上你的具体任务。

---

## 1) 超短模板（快速问题/小改动）

```text
你正在接手一个 Unity + Python 的 VR 位姿追踪项目。
先阅读：
1) docs/AI-HANDOFF-PROJECT-INTRO.md
2) docs/system-architecture-2026-03-20.md
3) docs/AI-HANDOFF-SESSION-2026-03-21.md
4) docs/AI-HANDOFF-SESSION-2026-03-22.md

并在开始前先用 3-5 条要点复述你读到的“当前状态 + 今日风险点”，再开始改动。

工作要求：
- 不要重构无关模块，优先最小改动。
- 通信层必须保持 Sender/Receiver 与 Encoder/Decoder 分层。
- Unity 侧 Receiver 只收原始 payload，Decoder 负责业务解析。
- 改动后请给出：变更文件清单、原因、风险、下一步建议。

当前任务：
【在这里写你的需求】
```

---

## 2) 完整模板（复杂功能/联调/架构改造）

```text
你是当前项目的接手 AI，请先完整理解项目文档与当前状态。

【必须先读】
- docs/AI-HANDOFF-PROJECT-INTRO.md
- docs/system-architecture-2026-03-20.md
- docs/AI-HANDOFF-SESSION-2026-03-21.md
- docs/AI-HANDOFF-SESSION-2026-03-22.md

【项目背景】
- 这是一个 Unity + Python 的 VR 位姿追踪系统。
- 已跑通旧链路：RealSense -> 服务器(FoundationPose) -> Unity。
- 正在迁移新链路：Quest 双目 -> 服务器(深度估计+FoundationPose) -> Quest。
- 当前新链路已可运行并可观测，但仍在做质量/稳定性优化（含内参映射与传输模式 A/B）。

【截至 2026-03-22 的已知重点】
1. 处理分辨率若设成 1:1 会导致画面拉伸，需保持与输入一致的宽高比（如 640x480）。
2. Quest 路径中标定分辨率与输入分辨率可能不一致，内参映射需考虑“中心裁剪+缩放”。
3. PackedSingleJpeg 质量风险高于 Dual 模式；联调时优先 Dual 做基线。
4. 目标是“先保证可验证与可回滚”，再做性能优化。

【关键架构约束】
1. 传输层/协议层解耦：
   - Sender/Receiver 只管传输
   - Encoder/Decoder 只管业务编解码
2. Unity 侧：
   - 单一 PayloadReceiver
   - Decoder 处理协议
   - 优先 Inspector 可视化事件绑定
3. Python 侧：
   - 统一 PayloadSender/PayloadReceiver
   - 可选 topic（PUB/SUB）或无 topic（PUSH/PULL）
4. 不做无关重构；必要时先保证链路可观测、可回滚。

【协议约束】
- Stereo: [left_jpg, right_jpg]
- RGBD: [color_jpg, depth_png]
- Tracking: [phase_byte, color_jpg, pose_json]

【请按以下格式输出】
A. 你对当前状态的理解（5-10条）
B. 你的实施计划（分步骤）
C. 实际改动（文件级）
D. 验证结果（运行/日志/错误）
E. 风险与待办
F. 阻塞项与需要我补充的信息（如有）

【当前任务】
【在这里写你的需求】
```

---

## 3) 常用任务补充片段（按需追加到模板末尾）

## A. 联调优先

```text
优先保证端到端可运行。若出现报错，先做最小修复让链路恢复，再讨论重构。
```

## B. 文档优先

```text
请在改动代码的同时同步更新 docs，确保后续 AI 能接手。
```

## C. 仅排查不改架构

```text
本次只允许修 bug，不允许调整架构边界和命名体系。
```

## D. 允许中等重构

```text
允许在不破坏现有协议的前提下做中等重构，但必须给出回滚点与迁移说明。
```

---

## 4) 你自己可直接用的一句话版本

```text
请先阅读 docs/AI-HANDOFF-PROJECT-INTRO.md、docs/system-architecture-2026-03-20.md、docs/AI-HANDOFF-SESSION-2026-03-21.md、docs/AI-HANDOFF-SESSION-2026-03-22.md，再按“最小改动、分层不破坏、可验证可回滚”的原则完成以下任务：【你的任务】
```

---

## 5) 今日推荐：一键接手 Prompt（直接复制）

```text
你正在接手 VR-Pose-Tracking（Unity + Python）项目，请严格按以下步骤工作：

【先读文档】
1) docs/AI-HANDOFF-PROJECT-INTRO.md
2) docs/system-architecture-2026-03-20.md
3) docs/AI-HANDOFF-SESSION-2026-03-21.md
4) docs/AI-HANDOFF-SESSION-2026-03-22.md

【开始前先输出】
- 用 5-10 条总结你理解的当前状态
- 列出你将执行的最小改动计划

【硬约束】
- 不重构无关模块；优先最小改动
- 通信层保持 Sender/Receiver 与 Encoder/Decoder 分层
- Unity Receiver 只收 payload，Decoder 做协议解析
- 先做可验证、可回滚的改动，再做性能优化

【当前已知风险点】
- 处理分辨率宽高比不一致会导致画面拉伸
- Quest 内参映射需考虑中心裁剪+缩放，不可仅线性缩放
- 联调优先 Dual 模式，PackedSingleJpeg 仅作对照

【输出格式】
A. 当前理解
B. 实施计划
C. 变更文件
D. 验证结果
E. 风险/待办
F. 阻塞与所需补充信息

当前任务：
【在这里写你的需求】
```


---

## 原始文档：AI-HANDOFF-SESSION-2026-03-21.md

# AI 会话交接记录（2026-03-21）

本文档记录 2026-03-21 当天围绕 Quest 双目链路做的排查、改动、结论与待办，供后续 AI 直接续做。

---

## 1. 当日目标

1. 排查 `quest_stereo_receiver.py` 显示 FPS 只有 27~36 的原因。
2. 区分“配置问题 / 统计问题 / 发送性能瓶颈”。
3. 在不破坏分层（Sender/Receiver 与 Encoder/Decoder）的前提下做可回滚优化。

---

## 2. 核心结论（已验证）

1. Python 端 FPS 统计公式无明显错误；`60 帧打印间隔约 1.67s` 对应约 `35.9 FPS`，和日志一致。
2. 接收端存在“只保留最新帧”逻辑（drain），因此旧统计会低估入口吞吐；已补充 `IngressFPS` 与 `DrainDrop` 诊断口径。
3. 发送端原先“处理耗时 + 固定等待”会形成降速效应，已改为按帧预算等待余量。
4. Quest 链路的主要瓶颈在发送侧采集/编码（`Blit + ReadPixels + JPEG`），而不是 Python 显示。
5. `PassthroughCameraAccess` 在真机更可能提供 `RenderTexture`，60 是 `MaxFramerate` 上限，不保证实际稳定 60。

---

## 3. 当日主要代码改动

## 3.1 Python 接收与统计增强

### 文件

- `src/quest_stereo_receiver.py`
- `src/zmq_utils/communicate/receiver.py`
- `src/zmq_utils/payload/decoder.py`

### 改动

1. `receiver.py`
   - 新增 `last_drain_count`，每次收包记录 drain 丢弃数量。
2. `quest_stereo_receiver.py`
   - 新增统计维度：`TotalFPS` / `ProcFPS` / `IngressFPS` / `DrainDrop` / `DrainDropRate`。
   - 新增阶段耗时统计：`Decode` / `Compose` / `Display`（ms）。
   - 新增 `Interval=...s`，避免“肉眼估计 1s”误判。
   - 新增 `SHOW_WINDOW` 开关，支持无窗口压测。
   - 新增启动日志：`PayloadParts` 与 `DecodeMode`（DualJpeg 或 PackedSingleJpeg）。
3. `decoder.py`
   - `StereoJpegDecoder` 兼容两种输入：
     - 传统 `[left_jpg, right_jpg]`
     - 单图拼接 `[packed_stereo_jpg]`

---

## 3.2 Unity 发送与编码侧调整

### 文件

- `Assets/Scripts/Net/Communicate/Sender/PayloadSender.cs`
- `Assets/Scripts/Net/Payload/Encoder/QuestStereoEncoder.cs`

### 改动

1. `PayloadSender.cs`
   - 发送循环改为按预算等待（避免处理耗时与固定等待叠加）。
   - 新增日志：`ActualFPS` / `DropRate` / `Encode` / `NetSend` / `Interval`。
2. `QuestStereoEncoder.cs`
   - 新增 `outputScale`（编码输出缩放；`1.0` 表示不降分辨率）。
   - 去除 `ReadPixels` 后不必要的 `Apply(false,false)`。
   - 新增可选 `packStereoIntoSingleJpeg` 路径（把双目拼成一张 JPEG）。
   - 新增运行时类型日志：`LeftType/RightType`，并按类型做分支优化：
     - `Texture2D` 且 `outputScale=1.0` 时尝试直接 `EncodeToJPG`
     - 其他情况走 `RenderTexture` 读回路径
   - 修复 packed 路径先拷贝源纹理再拼接的逻辑。

---

## 4. 重要回滚/稳定性决策

1. `packStereoIntoSingleJpeg` 在某些运行中导致“画面不动”风险，默认值已调整为 `false`（优先稳定）。
2. Python 端保持 dual/packed 双协议兼容，便于后续继续压测 packed 方案。
3. 如后续要继续提速，建议在稳定基线（DualJpeg）上做 A/B，不要直接替换默认链路。

---

## 5. 当前状态快照（交接时）

1. 画面可动性：以 `DualJpeg` 模式为主（更稳）。
2. 典型吞吐：约 35~40 FPS（受 Quest 采集与编码链路上限影响）。
3. 统计可观测性：已具备入口/处理/丢帧/阶段耗时多口径日志。

---

## 6. 后续 AI 建议优先级

1. **先验证真实瓶颈位置**
   - 读取 Unity 日志中的 `Encode` 与 `NetSend`。
   - 若 `Encode` 明显高于 `NetSend`，优先优化编码路径。
2. **保持 640x480 输入前提下提速**
   - `outputScale` 固定 `1.0`。
   - 优先尝试：异步读回（`AsyncGPUReadback`）+ 双缓冲。
3. **packed 单图策略二期**
   - 仅在独立分支压测，确认稳定再考虑默认开启。
4. **链路级替代方案（中期）**
   - 若继续追求 60，评估更低开销编码/传输路径（例如硬件编码、压缩格式替代、共享内存/原生插件链路）。

---

## 7. 风险与注意事项

1. `Library/PackageCache` 下的改动属于临时热修，包刷新后可能丢失（若存在此类改动，需转正式包管理方案）。
2. Quest 相机 `MaxFramerate` 是上限而非保证值；环境光、系统负载、渲染压力都会影响实际帧率。
3. 任何协议变更都要同步 Python Decoder 与 Unity Encoder/Decoder，避免“能收包但画面静止/解析错位”。

---

## 8. 建议的下一条接手 Prompt

请先阅读：

1. `docs/AI-HANDOFF-PROJECT-INTRO.md`
2. `docs/AI-HANDOFF-SESSION-2026-03-21.md`
3. `docs/system-architecture-2026-03-20.md`

目标：在保持 `640x480` 输入和现有协议兼容的前提下，把 Quest 双目链路从 35~40 FPS 稳定提升到更高，并给出可回滚实现与对比日志。

---

## 9. 构建异常长期修复（新增）

### 现象

- Quest 运行后再次打包，可能报错：`ArgumentException: An item with the same key has already been added. Key: Assembly-CSharp`。
- 调用栈指向 Voice 包内 `Conduit` 的 `AssemblyWalker`（`Library/PackageCache/com.meta.xr.sdk.voice@.../AssemblyWalker.cs`）。

### 根因

- 项目使用了聚合包 `com.meta.xr.sdk.all`，会自动拉入 `com.meta.xr.sdk.voice`。
- 即便项目未使用语音功能，Voice 的 Editor 构建回调仍会触发 Conduit 扫描并在重复 key 时崩溃。

### 已落地修复

- 修改 `Packages/manifest.json`：移除 `com.meta.xr.sdk.all`。
- 改为显式声明所需 XR 子包（保留原有功能组合，但排除 `com.meta.xr.sdk.voice`）：
  - `com.meta.xr.sdk.core`
  - `com.meta.xr.sdk.audio`
  - `com.meta.xr.sdk.haptics`
  - `com.meta.xr.mrutilitykit`
  - `com.meta.xr.sdk.platform`
  - `com.meta.xr.sdk.interaction`
  - `com.meta.xr.sdk.interaction.ovr`

### 说明

- 这是“包依赖层”的长期修复，优于每次手工删 `PackageCache` 的临时方案。
- 该改动可随 Unity Package 解析持续生效，不依赖本地缓存热修。


---

## 原始文档：AI-HANDOFF-SESSION-2026-03-22.md

# AI 会话交接记录（2026-03-22）

本文档记录 2026-03-22 当天围绕 Quest 双目 → Fast-FoundationStereo → FoundationPose 的排查、修复与结论，供下次 AI 直接续做。

---

## 1. 当日目标

1. 排查 Quest 链路中画面比例异常与 3D 框“稳定压扁”问题。
2. 解释 Quest 与 RealSense 在同流程下表现差异，并定位可改进环节。
3. 增强 Unity/Python 两端可观测性，便于后续定量调参。

---

## 2. 核心结论（本日）

1. **正方形显示问题已确认是脚本处理尺寸导致**：`process_width/process_height` 曾为 `640x640`，会把 `640x480` 拉伸为 1:1。
2. **“3D 框总有一面被压一截”属于系统性误差，不是随机跟踪抖动**：Quest 路径中内参与分辨率映射需考虑“裁剪+缩放”，不能只做线性缩放。
3. **Quest 与 RealSense 不是等价输入条件**：Quest 当前多为 RGB 透传+编码传输，RealSense 为深度传感链路，压缩、同步和光学路径差异会放大到 pose 质量。
4. **PackedSingleJpeg 模式已被明确标注风险**：该模式虽省传输开销，但对立体匹配质量更敏感，建议优先用 Dual 模式做基线。

---

## 3. 当日主要代码改动

## 3.1 Python 主链路（Quest 测试脚本）

### 文件

- `src/quest_stereo_pose_test.py`

### 改动

1. 默认处理分辨率改为 `640x480`（避免默认 1:1 拉伸）。
2. 增加一次性告警：
   - `[Aspect]`：处理目标宽高比与输入不一致时提示（防止无意拉伸）。
   - `[CalibAspect]`：标定分辨率与输入分辨率宽高比不一致时提示（提示潜在内参映射风险）。
3. 增加内参映射可控参数：
   - `--calib_assume_center_crop`（默认 `1`，推荐）
   - 含义：当标定与输入宽高比不一致时，按“中心裁剪后再缩放”映射 `K`。
4. 新增 `[KMap]` 日志，打印本次运行实际生效的 `fx/fy/cx/cy` 与映射模式。

---

## 3.2 Unity 发送端（本日关联确认）

### 文件

- `Assets/Scripts/Net/Payload/Encoder/QuestStereoEncoder.cs`

### 当日关注点（与已实现能力）

1. 编码链路已支持 `JPEG/PNG` 选项，默认偏高质量。
2. 已有纹理类型与编码统计日志，便于判断 `GetTexture()` 实际类型与编码耗时。
3. 可识别是否走 `PackedSingleJpeg`，并配合 Python 端日志进行协议一致性校验。

---

## 4. 本日验证与现象对照

1. 运行界面已可直接显示：
   - `Decode: PackedSingleJpeg/DualJpeg`
   - `Proc: WxH`
   - `Depth prep/forward/post`
   - `DrainDrop`
2. `--help` 已确认包含新参数 `--calib_assume_center_crop`。
3. 静态检查：`quest_stereo_pose_test.py` 无新增语法错误。

---

## 5. 当前状态快照（交接时）

1. Quest 链路仍可运行，但质量与稳定性调优尚在进行中。
2. 已具备“比例/内参映射异常”诊断日志，不再是黑盒调参。
3. `pose_tracker_api.py` 在本轮结束时被回退为用户当前版本（不依赖本次修改即可继续）。

---

## 6. 下一步建议（优先级）

1. **先做 A/B 验证（高优先）**
   - A: `--calib_assume_center_crop 1`
   - B: `--calib_assume_center_crop 0`
   - 对比指标：3D 框贴合度、稳定性、`[KMap]` 与 `[CalibAspect]` 日志。
2. **协议基线固定**
   - 先固定 `Dual`（非 Packed）做基线，减少压缩形变干扰。
3. **发送端质量对照**
   - 在 Unity 侧对比 `JPEG(95/100)` 与 `PNG`，记录 payload 与 FPS。
4. **如仍存在固定方向形变**
   - 需要进一步核对 Quest 取图是否存在非中心裁剪或额外旋转/镜像路径。

---

## 7. 建议下次 AI 使用的接手 Prompt

请先阅读：

1. `docs/AI-HANDOFF-PROJECT-INTRO.md`
2. `docs/AI-HANDOFF-SESSION-2026-03-21.md`
3. `docs/AI-HANDOFF-SESSION-2026-03-22.md`
4. `docs/system-architecture-2026-03-20.md`

目标：在 Quest 双目链路中，固定 Dual 高质量传输，完成 `calib_assume_center_crop` 的 A/B 对比并给出定量结论（贴合度/FPS/稳定性），必要时继续修正内参映射与上游裁剪假设。

---

## 8. 备注

- 工作区中出现的 `.utmp`、`compile_commands`、`build.ninja` 等文件属于 Unity/NDK 构建产物，不属于本次算法逻辑改动范围。
- 用户偏好：网络配置不写死 IP（通过 UI 输入并持久化）；架构偏好数据提供方与发送方解耦。


---

## 原始文档：AI-HANDOFF-SESSION-2026-03-30.md

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


---

## 原始文档：MODULAR-API-CLI-PLAN.md

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


---

## 原始文档：system-architecture-2026-03-20.md

# VR Pose Tracking 系统改造记录与架构说明（2026-03-20）

## 1. 今日改动总结

今天完成了以下关键改造：

1. **Unity 接收链路统一**
   - 统一为单一 `PayloadReceiver`（不再拆分多种 Receiver 继承）。
   - `PayloadReceiver` 只负责收包，不做业务解码。
   - RGBD 与 Tracking 解析均下沉到独立 Decoder（`RGBDDecoder` / `TrackingDecoder`）。

2. **Unity 发送链路命名统一**
   - 编码器统一去掉 Payload 前缀：
     - `EncoderBase`
     - `QuestStereoEncoder`
     - `StaticStereoEncoder`
     - `IntegerEncoder`
   - `PayloadSender` 对编码器的依赖统一到 `EncoderBase`。

3. **Unity 事件模型统一**
   - Receiver/Sender 的 IP 变化事件统一为 `UnityEvent` 风格，方便 Inspector 可视化绑定。
   - Decoder 改为支持 Inspector 直接绑定 `OnPayloadReceived`（你要求“可视化优先”后已切换）。

4. **Python 发送/接收统一**
   - 统一为 `PayloadSender` / `PayloadReceiver` 两个核心通信类。
   - 支持可选 topic（PUB/SUB）或无 topic（PUSH/PULL）模式。
   - 支持 multipart/raw/text/json 的对称发送接收接口。

5. **旧通信模块清理**
   - 删除已无主流程引用的旧模块：`base.py`, `image.py`, `pose.py`, `rgbd.py`, `tracking.py`。
   - 精简 `zmq_utils/__init__.py` 和 `zmq_utils/payload/__init__.py` 的导出，避免历史 API 误导。

---

## 2. 架构演进：从旧链路到新链路

## 2.1 旧链路（已存在并可运行）

`RealSense -> 服务器 -> Unity`

- RealSense 端采集彩色+深度图，编码后发给服务器。
- 服务器执行 FoundationPose 推理，输出位姿与可视化图像。
- Unity 收到 tracking payload，解码后驱动物体跟随。

## 2.2 新目标链路（正在迁移）

`Quest 双目 -> 服务器 -> Quest`

- Quest 端发送左右眼图像到服务器。
- 服务器侧进行深度估计（由双目生成深度）后，输入 FoundationPose。
- 再将 tracking 结果返回给 Quest（图像/位姿）。

## 2.3 当前落地状态（截至 2026-03-20）

- 已完成：Quest 双目 **静态两张图** 的编码、传输、接收与简单显示验证。
- 未完全接通：
  - 双目 -> 深度估计模块 -> FoundationPose 推理全链路尚未全部串起。
  - Quest 侧基于返回 pose 的完整闭环展示仍在后续接入。

---

## 3. 当前系统通信设计（核心思想）

通信层采用四段式分层：

1. **Sender（传输发送）**
   - 只负责把 `list[bytes]` 发出去。
   - 不关心业务含义（图像/深度/pose）。

2. **Encoder（业务编码）**
   - 把业务对象（图像、深度、位姿等）编码成 `payload parts`。
   - 例如：RGBD 编码为 `[color_jpg, depth_png]`。

3. **Receiver（传输接收）**
   - 只负责收 `payload parts`，可选 topic。
   - 不直接进行业务解析。

4. **Decoder（业务解码）**
   - 把原始 `payload parts` 解析回业务对象并派发事件。
   - 例如：Tracking 解码为 phase + color + poseMatrix。

这套设计的目标是：

- 通信协议与业务处理解耦。
- Unity / Python 两端职责对齐，便于互相替换和调试。
- 支持静态图测试、回放测试与未来扩展新 payload 类型。

---

## 4. 协议定义（当前使用）

## 4.1 Stereo（Quest 双目）

- `parts[0]`: left_jpg
- `parts[1]`: right_jpg

## 4.2 RGBD

- `parts[0]`: color_jpg
- `parts[1]`: depth_png

## 4.3 Tracking

- `parts[0]`: phase_byte（0=detecting, 1=tracking）
- `parts[1]`: color_jpg
- `parts[2]`: pose_json（空字符串表示无 pose）

---

## 5. 文字架构图（当前状态 + 目标状态）

```text
[当前可运行主链路（旧链路）]

RealSense Sender (Python)
  └─ RGBDEncoder -> PayloadSender(PUSH)
       tcp://server:5555
           ↓
Pose Server (Python)
  ├─ PayloadReceiver(PULL)
  ├─ RGBDDecoder
  ├─ FoundationPose / PoseTracker
  ├─ TrackingEncoder
  └─ PayloadSender(PUB with topic=tracking)
       tcp://*:5556
           ↓
Unity PayloadReceiver(SUB, topic=tracking)
  └─ TrackingDecoder
      ├─ OnImageReceived -> ImageViewer
      └─ OnPoseReceived  -> CubeFollow


[目标迁移链路（进行中）]

Quest (Unity)
  └─ QuestStereoEncoder / StaticStereoEncoder
      -> PayloadSender(PUSH/PUB)
           ↓
Stereo/Depth Server (Python, 规划/在建)
  ├─ PayloadReceiver
  ├─ StereoJpegDecoder
  ├─ Depth Estimation (双目算深度)
  ├─ FoundationPose (输入 color + depth)
  ├─ TrackingEncoder
  └─ PayloadSender
           ↓
Quest (Unity)
  └─ PayloadReceiver -> TrackingDecoder
      ├─ 可视化
      └─ 姿态驱动（对象/相机）
```

---

## 6. 关键脚本职责映射（当前）

## Unity

- `Assets/Scripts/Net/Communicate/Sender/PayloadSender.cs`
  - 通用发送器，管理连接、发送频率、丢帧统计、IP 持久化。
- `Assets/Scripts/Net/Communicate/Receiver/PayloadReceiver.cs`
  - 通用接收器，线程收包，主线程派发 `OnPayloadReceived`。
- `Assets/Scripts/Net/Payload/Encoder/*`
  - 各类业务编码器（QuestStereo/StaticStereo/Integer）。
- `Assets/Scripts/Net/Payload/Decoder/*`
  - 各类业务解码器（RGBD/Tracking）。
- `Assets/Scripts/Net/ReceiveData.cs`
  - RawData/PoseData 数据结构与 UnityEvent 定义。
- `Assets/Scripts/Net/ImageViewer.cs`
  - 图像渲染与 FPS/延迟统计显示。
- `Assets/Scripts/CubeFollow.cs`
  - 使用 pose 驱动物体。

## Python

- `src/realsense_sender.py`
  - RealSense 采集与 RGBD 发送。
- `src/pose_server.py`
  - 接收 RGBD、调用 PoseTracker、发送 tracking。
- `src/quest_stereo_receiver.py`
  - 接收双目图像并做基础显示验证。
- `src/zmq_utils/communicate/*`
  - 统一 PayloadSender/PayloadReceiver 通信层。
- `src/zmq_utils/payload/*`
  - 统一 Encoder/Decoder 协议层。
- `src/pose_tracker_api.py`
  - FoundationPose + SAM3 封装 API。

---

## 7. 后续建议（按优先级）

1. 打通 `Quest 双目 -> 深度估计 -> FoundationPose -> Quest 回传` 一条完整链路。
2. 为每种 payload 增加显式 `protocol version` 字段，降低未来协议升级风险。
3. 增加端到端联调脚本（启动顺序、端口探活、topic 自检、样例帧校验）。
4. 增加异常观测：发送端 drop rate、接收端 decode fail rate、推理耗时分位数。

---

## 8. 一句话结论

你现在已经完成了通信架构的“统一与解耦”基础工程；系统从“设备耦合链路”转向“可替换的协议化链路”，下一步只需把双目算深度模块接上，即可进入 Quest 端位姿闭环阶段。


---

## 原始文档：UnityRuntimeInspector.md


Skip to content
yasirkula
UnityRuntimeInspector
Repository navigation
Code
Issues
6
 (6)
Pull requests
3
 (3)
Agents
Discussions
Security
Insights
Owner avatar
UnityRuntimeInspector
Public
yasirkula/UnityRuntimeInspector
Go to file
t
Name		
yasirkula
yasirkula
Simplified Package Manager installation instruction
f8755d6
 · 
10 months ago
.github
Simplified Package Manager installation instruction
10 months ago
Plugins
- Updated Unity version to 2021.3.41f1
last year
LICENSE.txt
Added Package Manager support
6 years ago
LICENSE.txt.meta
Added Package Manager support
6 years ago
Plugins.meta
Added Package Manager support
6 years ago
package.json
- Updated Unity version to 2021.3.41f1
last year
package.json.meta
Added Package Manager support
6 years ago
Repository files navigation
README
Contributing
MIT license
Runtime Inspector & Hierarchy for Unity 3D
screenshot

Available on Asset Store: https://assetstore.unity.com/packages/tools/gui/runtime-inspector-hierarchy-111349
可在 Asset Store 获取： https://assetstore.unity.com/packages/tools/gui/runtime-inspector-hierarchy-111349

Forum Thread: https://forum.unity.com/threads/runtime-inspector-and-hierarchy-open-source.501220/
论坛帖子： https://forum.unity.com/threads/runtime-inspector-and-hierarchy-open-source.501220/

Discord: https://discord.gg/UJJt549AaV
Discord： https://discord.gg/UJJt549AaV

GitHub Sponsors ☕  GitHub 赞助商☕

A. ABOUT  A. 关于
This is a simple yet powerful runtime Inspector and Hierarchy solution for Unity 3D that should work on pretty much any platform that Unity supports, including mobile platforms.
这是一个简单而强大的 Unity 3D 运行时检查器和层级解决方案，几乎可以在 Unity 支持的任何平台上运行，包括移动平台。

B. LICENSE  B. 许可证
Runtime Inspector & Hierarchy is licensed under the MIT License (Asset Store version is governed by the Asset Store EULA). Please note that this asset uses an external asset which is licensed under the BSD 3-Clause License.
Runtime Inspector & Hierarchy 采用 MIT 许可证授权（ Asset Store 版本受 Asset Store EULA 约束）。请注意，此资源使用了外部资源，该外部资源采用 BSD 3-Clause 许可证授权。

C. INSTALLATION  C. 安装
There are 5 ways to install this plugin:
安装此插件有 5 种方法：

import RuntimeInspector.unitypackage via Assets-Import Package
通过 Assets-Import Package 导入 RuntimeInspector.unitypackage
clone/download this repository and move the Plugins folder to your Unity project's Assets folder
克隆/ 下载此存储库，并将 Plugins 文件夹移动到 Unity 项目的 Assets 文件夹中。
import it from Asset Store
从 Asset Store 导入
(via Package Manager) click the + button and install the package from the following git URL:
（通过软件包管理器） 点击“+”按钮，然后从以下 git URL 安装软件包：
https://github.com/yasirkula/UnityRuntimeInspector.git
(via OpenUPM) after installing openupm-cli, run the following command:
（通过 OpenUPM ） 安装 openupm-cli 后，运行以下命令：
openupm add com.yasirkula.runtimeinspector
FAQ  常问问题
New Input System isn't supported on Unity 2019.2.5 or earlier
Unity 2019.2.5 或更早版本不支持新的输入系统。
Add ENABLE_INPUT_SYSTEM compiler directive to Player Settings/Scripting Define Symbols (these symbols are platform specific, so if you change the active platform later, you'll have to add the compiler directive again).
将 ENABLE_INPUT_SYSTEM 编译器指令添加到玩家设置/脚本定义符号中 （这些符号是平台特定的，因此如果您稍后更改活动平台，则必须再次添加编译器指令）。

"Unity.InputSystem" assembly can't be resolved on Unity 2018.4 or earlier
在 Unity 2018.4 或更早版本中无法解析“Unity.InputSystem”程序集。
Remove Unity.InputSystem assembly from RuntimeInspector.Runtime Assembly Definition File's Assembly Definition References list.
从 RuntimeInspector.Runtime 程序集定义文件的程序集定义引用列表中移除 Unity.InputSystem 程序集。

D. HOW TO  D. 如何
To use the hierarchy in your scene, drag&drop the RuntimeHierarchy prefab to your canvas
要在场景中使用层级结构，请将 RuntimeHierarchy 预制件拖放到画布上。
To use the inspector in your scene, drag&drop the RuntimeInspector prefab to your canvas
要在场景中使用检查器，请将 RuntimeInspector 预制件拖放到画布上。
You can connect the inspector to the hierarchy so that whenever the selection in the hierarchy changes, inspector inspects the newly selected object. To do this, assign the inspector to the Connected Inspector property of the hierarchy.
您可以将检查器连接到层级结构，这样，每当层级结构中的选择发生变化时，检查器都会检查新选择的对象。为此，请将检查器分配给层级结构的 “已连接检查器” 属性。

You can also connect the hierarchy to the inspector so that whenever an object reference in the inspector is highlighted, the selection in hierarchy is updated. To do this, assign the hierarchy to the Connected Hierarchy property of the inspector.
您还可以将层级结构连接到检查器，这样，每当检查器中的对象引用被选中时，层级结构中的选择也会随之更新。为此，请将层级结构分配给检查器的 “已连接层级结构” 属性。

Note that these connections are one-directional, meaning that assigning the inspector to the hierarchy will not automatically assign the hierarchy to the inspector or vice versa. Also note that the inspector and the hierarchy are not singletons and therefore, you can have several instances of them in your scene at a time with different configurations.
请注意，这些连接是单向的 ，这意味着将检查器分配给层级结构不会自动将层级结构分配给检查器，反之亦然。另请注意，检查器和层级结构都不是单例，因此，您可以在场景中同时拥有多个具有不同配置的检查器和层级结构实例。

E. FEATURES  E. 特点
Both panels are heavily optimized in terms of GC in order not to cause any unnecessary allocations. By default, both the inspector and the hierarchy are refreshed 4 times a second to reflect any changes to their user interface almost immediately. Each refresh of the inspector generates some garbage for GC since most of the time, the inspected object has variables of value types. These variables are boxed when accessed via reflection and this boxing creates some unavoidable garbage. However, this process can be greatly optimized by increasing the Refresh Interval of the inspector and/or the hierarchy
这两个面板都针对垃圾回收进行了深度优化，以避免不必要的内存分配。默认情况下，检查器和层级视图每秒刷新 4 次，以便几乎立即反映用户界面的任何更改。每次刷新检查器都会产生一些垃圾需要进行垃圾回收，因为大多数情况下，被检查的对象都包含值类型的变量。这些变量在通过反射访问时会被装箱 ，而这种装箱操作会产生一些不可避免的垃圾。但是，可以通过增加检查器和/或层级视图的刷新间隔来显著优化这个过程。
Includes a built-in color picker and a reference picker:
内置颜色选择器和参考选择器：
screenshot

Visual appearance of the inspector and the hierarchy can be tweaked by changing their Skin. There are two premade skins included in the Skins directory: LightSkin and DarkSkin. You can create your own skins using the Assets-Create-yasirkula-RuntimeInspector-UI Skin context menu
可以通过更改皮肤来调整检查器及其层级的视觉外观。Skins 目录中包含两个预制皮肤： LightSkin 和 DarkSkin 。您可以使用 Assets-Create-yasirkula-RuntimeInspector-UI Skin 上下文菜单创建自己的皮肤。
screenshot

The hierarchy supports multi-selection:
该层级结构支持多选：
screenshot

E.1. INSPECTOR  E.1. 检查员
screenshot

RuntimeInspector works similar to the editor Inspector. It can expose commonly used Unity types out-of-the-box, as well as custom classes and structs that are marked with System.Serializable attribute. 1-dimensional arrays and generic Lists are also supported.
RuntimeInspector 的工作方式与编辑器 Inspector 类似。它可以开箱即用地公开常用的 Unity 类型，以及使用 System.Serializable 属性标记的自定义类和结构体。此外，它还支持一维数组和泛型列表。

Refresh Interval: as the name suggests, this is the refresh interval of the inspector. At each refresh, values of all the exposed fields and properties are refreshed. This generates some garbage for boxed value types (unavoidable) and thus, increasing this value even slightly should help with GC a lot
刷新间隔 ：顾名思义，这是检查器的刷新间隔。每次刷新时，所有公开字段和属性的值都会被刷新。这会为装箱值类型生成一些垃圾数据（这是不可避免的），因此，即使稍微增加此值也能显著改善垃圾回收。
Expose Fields: determines which fields of the inspected object should be exposed: None, Serializable Only or All
公开字段 ：确定应公开被检查对象的哪些字段： 无 、 仅可序列化字段或全部
Expose Properties: determines which properties of the inspected object should be exposed
公开属性 ：确定应公开被检查对象的哪些属性。
Array Indices Start At One: when enabled, exposed arrays and lists start their indices at 1 instead of 0 (just a visual change)
数组索引从 1 开始 ：启用后，公开的数组和列表的索引将从 1 开始而不是 0（只是视觉上的变化）。
Use Title Case Naming: when enabled, variable names are displayed in title case format (e.g. m_myVariable becomes My Variable)
使用首字母大写命名 ：启用后，变量名将以首字母大写格式显示（例如， m_myVariable 变为 My Variable ）。
Show Add Component Button: when enabled, Add Component button will appear while inspecting a GameObject
显示“添加组件”按钮 ：启用后，在检查游戏对象时将显示 “添加组件” 按钮。
Show Remove Component Button: when enabled, Remove Component button will appear under inspected components
显示“移除组件”按钮 ：启用后， “移除组件” 按钮将显示在已检查组件下方。
Show Inspect Reference Button: when enabled, ObjectReferenceFields will show an arrow next to the selected Object reference. When that arrow is clicked, inspector will automatically inspect that Object
显示“检查引用”按钮 ：启用后， “对象引用字段 ”将在选定的对象引用旁边显示一个箭头。单击该箭头，检查器将自动检查该对象。
Show Tooltips: when enabled, hovering over a variable's name for a while will show a tooltip displaying the variable's name. Can be useful for variables whose names are partially obscured
显示工具提示 ：启用后，将鼠标悬停在变量名称上片刻，即可显示包含变量名称的工具提示。这对于名称部分被遮挡的变量非常有用。
Tooltip Delay: determines how long the cursor should remain static over a variable's name before the tooltip appears. Has no effect if Show Tooltips is disabled
工具提示延迟 ：决定光标在变量名上停留多长时间后才会显示工具提示。如果 “显示工具提示” 已禁用，则此设置无效。
Nest Limit: imagine exposing a linked list. This variable defines how many nodes you can expose in the inspector starting from the initial node until the inspector stops exposing any further nodes
嵌套限制 ：想象一下显示一个链表。此变量定义了从初始节点开始，检查器可以显示多少个节点，直到检查器停止显示任何后续节点为止。
Inspected Object Header Visibility: if the inspected object has a collapsible header, determines that header's visibility
被检查对象头部可见性 ：如果被检查对象具有可折叠的头部，则确定该头部的可见性
Pool Capacity: the UI elements are pooled to avoid unnecessary Instantiate and Destroy calls. This value defines the pool capacity for each of the UI elements individually. On standalone platforms, you can increase this value for better performance
池容量 ：UI 元素被放入池中，以避免不必要的实例化和销毁调用。此值定义了每个 UI 元素的池容量。在独立平台上，您可以增加此值以提高性能。
Settings: an array of settings for the inspector. A new settings asset can be created using the Assets-Create-yasirkula-RuntimeInspector-Settings context menu. A setting asset stores 4 different things:
设置 ：检查器设置的数组。可以使用 “资源”-“创建”-“yasirkula”-“运行时检查器”-“设置” 上下文菜单创建新的设置资源。设置资源存储 4 种不同的内容：
Standard Drawers and Reference Drawers: a drawer is a prefab used to expose a single variable in the inspector. For variables that extend UnityEngine.Object, a reference drawer is created and for other variables, a standard drawer is created
标准抽屉和引用抽屉 ：抽屉是一种预制件，用于在检视面板中显示单个变量。对于继承自 UnityEngine.Object 的变量，会创建一个引用抽屉；对于其他变量，会创建一个标准抽屉。
While searching for a suitable drawer for a variable, the corresponding drawers list is traversed from bottom to top until a drawer that supports that variable type is found. If such a drawer is not found, that variable is not exposed
在查找适合变量的抽屉时，会从下到上遍历相应的抽屉列表，直到找到支持该变量类型的抽屉为止。如果找不到这样的抽屉，则不会公开该变量。
Hidden Variables: allows you to hide some variables from the inspector for a given type and all the types that extend/implement it. You can enter asterisk character (*) to hide all the variables for that type
隐藏变量 ：允许您对给定类型及其所有扩展/实现类型隐藏某些变量。您可以使用星号 (*) 隐藏该类型的所有变量。
Exposed Variables: allows you to expose (counter) some hidden variables. A variable goes through a number of filters before it is exposed:
公开变量 ：允许您公开（显示）一些隐藏变量。变量在公开之前会经过一系列过滤器：
Its Type must be serializable
它的类型必须是可序列化的。
It must not have a System.Obsolete, System.NonSerialized or HideInInspector attribute
它不能具有 System.Obsolete 、 System.NonSerialized 或 HideInInspector 属性。
If it is in Exposed Variables, it is exposed
如果它位于 “公开变量” 中，则表示它已公开
It must not be in Hidden Variables
它不能位于隐藏变量中。
it must pass the Expose Fields and Expose Properties filters
它必须通过 “公开字段” 和 “公开属性” 筛选器。
So, to expose only a specific set of variables for a given type, you can hide all of its variables by entering an asterisk to its Hidden Variables and then entering the set of exposed variables to its Exposed Variables
因此，要仅公开给定类型的一组特定变量，您可以通过在其 “隐藏变量” 中输入星号来隐藏其所有变量，然后在其 “公开变量” 中输入要公开的变量集。
While changing the inspector's settings, you are advised not to touch InternalSettings; instead create a separate Settings asset and add it to the Settings array of the inspector. Otherwise, when InternalSettings is changed in an update, your settings might get overridden.
修改检查器设置时，建议不要直接修改 InternalSettings ；而是创建一个单独的 Settings 资源，并将其添加到检查器的 Settings 数组中。否则，当 InternalSettings 在更新过程中发生更改时，您的设置可能会被覆盖。

E.2. HIERARCHY  E.2. 层级结构
screenshot

RuntimeHierarchy simply exposes the objects in your scenes to the user interface. In addition to exposing the currently active Unity scenes in the hierarchy, you can also expose a specific set of objects under what is called a pseudo-scene in the hierarchy. Pseudo-scenes can help you categorize the objects in your scene. Adding/removing objects to/from pseudo-scenes is only possible via the scripting API and helper components.
RuntimeHierarchy 的作用是将场景中的对象暴露给用户界面。除了在层级视图中显示当前活动的 Unity 场景之外，你还可以将一组特定的对象（称为伪场景） 暴露在层级视图中。伪场景可以帮助你对场景中的对象进行分类。向伪场景添加/从伪场景中移除对象只能通过脚本 API 和辅助组件来实现。

Refresh Interval: the refresh interval of the hierarchy. At each refresh, the destroyed objects are removed from the hierarchy while newly created objects are added to the hierarchy. Sibling indices of the objects are also synced with the Unity Hierarchy at each refresh
刷新间隔 ：层级结构的刷新间隔。每次刷新时，已销毁的对象会从层级结构中移除，而新创建的对象则会添加到层级结构中。每次刷新时，对象的兄弟索引也会与 Unity 层级结构同步。
Object Names Refresh Interval: accessing GameObject.name property generates garbage. Therefore, names of objects in the hierarchy are not synced at each Refresh Interval but rather at each Object Names Refresh Interval to help avoid excessive garbage
对象名称刷新间隔 ：访问 GameObject.name 属性会产生垃圾数据。因此，层级结构中的对象名称并非在每个刷新间隔同步，而是在每个对象名称刷新间隔同步， 以避免产生过多垃圾数据。
Search Refresh Interval: the refresh interval for the search results. At each refresh, each GameObject's name is checked to see if it matches the searched term, so this process will generate some garbage
搜索刷新间隔 ：搜索结果的刷新间隔。每次刷新时，系统都会检查每个游戏对象的名称是否与搜索词匹配，因此此过程会产生一些垃圾数据。
Allow Multi Selection: when disabled, only a single Transform can be selected in the hierarchy
允许多选 ：禁用此选项后，层次结构中只能选择一个变换。
Expose Unity Scenes: when disabled, Unity scenes are not exposed in the hierarchy. This is useful when you want to use the hierarchy solely for pseudo-scenes
公开 Unity 场景 ：禁用此选项后，Unity 场景将不会在层级视图中显示。如果您只想将层级视图用于伪场景，这将非常有用。
Exposed Unity Scenes Subset: specifies the scenes that are exposed in the hierarchy by their name. When empty, all scenes are exposed
公开的 Unity 场景子集 ：指定在层级结构中按名称公开的场景。如果为空，则公开所有场景。
Expose Dont Destroy On Load Scene: when enabled, DontDesroyOnLoad objects will be exposed in the hierarchy
公开加载时不销毁场景 ：启用后， DontDesroyOnLoad 对象将在层级视图中公开。
Pseudo Scenes Order: the order of the pseudo-scenes from top to bottom in the hierarchy. Note that entering a pseudo-scene here does not automatically create it when the application starts. Pseudo-scenes can be created via the scripting API only
伪场景顺序 ：伪场景在层级结构中从上到下的顺序。请注意，在此处输入伪场景并不会在应用程序启动时自动创建它。伪场景只能通过脚本 API 创建。
Pointer Long Press Action: determines what will happen when an object is clicked and then held for a while:
指针长按操作 ：决定点击并按住某个对象一段时间后会发生什么：
None: nothing ¯\_(ツ)_/¯
无 ：什么都没有 ¯\_(ツ)_/¯
Create Dragged Reference Item: creates a dragged reference item that can be dropped onto a reference drawer in the inspector to assign the held object(s) to that variable (similar to Unity's drag&drop reference assignment)
创建拖拽引用项 ：创建一个可拖拽的引用项 ，可以将其拖放到检视面板中的引用抽屉中，以将所持有的对象分配给该变量（类似于 Unity 的拖放引用分配）。
Show Multi Selection Toggles: displays multi-selection toggles in front of each object. This is mostly useful on mobile devices where CTRL and Shift keys aren't present. Has no effect if Allow Multi Selection is disabled
显示多选切换按钮 ：在每个对象前面显示多选切换按钮。这在移动设备上尤其有用，因为移动设备上通常没有 Ctrl 和 Shift 键。如果 “允许多选” 已禁用，则此功能无效。
Show Multi Selection Toggles Then Create Dragged Reference Item: if multi-selection toggles aren't visible, displays them. Otherwise, creates a dragged reference item
显示多选切换按钮，然后创建拖动参考项 ：如果多选切换按钮不可见，则显示它们。否则，创建一个拖动参考项。
Pointer Long Press Duration: determines how long an object should be held until the Pointer Long Press Action is executed
指针长按持续时间 ：决定了在执行指针长按操作之前需要按住对象多长时间。
Double Click Threshold: when an object in the hierarchy is double clicked, OnItemDoubleClicked event is raised (see SCRIPTING API). This value determines the maximum allowed delay between two clicks to register a double click
双击阈值 ：当层级结构中的对象被双击时，会触发 OnItemDoubleClicked 事件（参见脚本 API ）。此值决定了两次点击之间允许的最大延迟，以判定是否触发双击事件。
Can Reorganize Items: when enabled, dropping a dragged reference item that holds Transform(s) onto an object in the hierarchy will change the dragged Transform(s)' parents (similar to parenting in Unity's Hierarchy)
可以重新组织项目 ：启用后，将包含变换的拖动引用项放到层级结构中的对象上，将更改拖动变换的父级（类似于 Unity 层级结构中的父级关系）。
Can Drop Dragged Parent On Child: when enabled, a dragged reference item can be dropped onto one of its child objects. In this case, the child object will be unparented and then the dragged reference item will become a child of it. Has no effect if Can Reorganize Items is disabled
可将拖动的父对象放置到子对象上 ：启用此功能后，可以将拖动的引用项放置到其子对象上。此时，子对象将失去父级关系，拖动的引用项将成为该子对象的子对象。如果 “可重新组织项目” 已禁用，则此功能无效。
Can Drop Dragged Objects To Pseudo Scenes: when enabled, dropping a dragged reference item onto a pseudo-scene or above/below a root object in the pseudo-scene will automatically add it to that pseudo-scene. Has no effect if Can Reorganize Items is disabled
可将拖动对象放置到伪场景中 ：启用此功能后，将拖动的参考项放置到伪场景中，或放置到伪场景中根对象的上方/下方，将自动将其添加到该伪场景中。如果 “可重新组织项目” 已禁用，则此功能无效。
Show Tooltips: when enabled, hovering over an object for a while will show a tooltip displaying the object's name. Can be useful for objects with very long names
显示工具提示 ：启用后，将鼠标悬停在对象上一段时间，将显示一个工具提示，其中包含对象的名称。这对于名称很长的对象非常有用。
Tooltip Delay: determines how long the cursor should remain static over an object before the tooltip appears. Has no effect if Show Tooltips is disabled
工具提示延迟 ：决定光标在对象上停留多长时间后才会显示工具提示。如果 “显示工具提示” 已禁用，则此设置无效。
Show Horizontal Scrollbar: when enabled, a horizontal scrollbar will be displayed if the names displayed in the hierarchy don't fit the available space. Note that only the visible items' width values are used to determine the size of the scrollable area
显示水平滚动条 ：启用后，如果层级结构中显示的名称超出可用空间，则会显示水平滚动条。请注意，滚动区域的大小仅根据可见项的宽度值来确定。
Sync Selection With Editor Hierarchy: simply synchronizes the selected object between the Unity Hierarchy and this RuntimeHierarchy
将选择与编辑器层级同步 ：简单地将 Unity 层级和此运行时层级之间的选定对象同步。
Additional settings for Can Reorganize Items can be found at the RuntimeHierarchy/ScrollView/Viewport object:
可以在 RuntimeHierarchy/ScrollView/Viewport 对象中找到 “可重新组织项目” 的其他设置：

screenshot

Sibling Index Modification Area: when a dragged reference item is dropped near the top or bottom edges of a Transform in hierarchy, it will be inserted above or belove the target Transform. This value determines the size of the area near the top and bottom edges
同级索引修改区域 ：当拖动的引用项放置在层级结构中变换的顶部或底部边缘附近时，它将被插入到目标变换的上方或下方。此值决定了顶部和底部边缘附近区域的大小。
Scrollable Area: while hovering the cursor near the top or bottom edges of the scroll view with a dragged reference item, scroll view will automatically be scrolled to show contents in that direction. This value determines the size of the area near the top and bottom edges of the scroll view
可滚动区域 ：当拖动参考项并将光标悬停在滚动视图的顶部或底部边缘附近时，滚动视图将自动滚动以显示该方向的内容。此值决定了滚动视图顶部和底部边缘附近区域的大小。
Scroll Speed: determines how fast the scroll view will be scrolled while hovering the cursor over Scrollable Area
滚动速度 ：决定当光标悬停在可滚动区域上时，滚动视图的滚动速度。
F. SCRIPTING API  F. 脚本 API
Values of the variables that are mentioned in E.1 and E.2 sections can be tweaked at runtime via their corresponding properties. Any changes to these properties will be reflected to UI immediately. Here, you will find some interesting things that you can do with the inspector and the hierarchy via scripting:
E.1 和 E.2 节中提到的变量值可以在运行时通过其对应的属性进行调整。对这些属性的任何更改都会立即反映在用户界面上。在这里，您会发现一些可以通过脚本使用检查器和层级结构实现的有趣功能：

You can change the inspected object in the inspector using the following functions:
您可以使用以下函数在检查器中更改被检查的对象：
public void Inspect( object obj );
public void StopInspect();
You can access the currently inspected object via the InspectedObject property of the inspector
您可以通过检查器的 InspectedObject 属性访问当前正在检查的对象。
You can change the selected object in the hierarchy using the following functions:
您可以使用以下功能更改层次结构中选定的对象：
// SelectOptions is an enum flag meaning that it can take multiple values with | (OR) operator. These values are:
// - Additive: new selection will be appended to the current selection instead of replacing it
// - FocusOnSelection: scroll view will be snapped to the selected object(s)
// - ForceRevealSelection: normally, when selection changes, the new selection will be fully explored in the hierarchy (i.e. all of the parents of the selection will be
//   expanded to reveal the selection). This doesn't automatically happen if selection doesn't change. When this flag is set, however, the selected objects will be fully
//   revealed/explored even if the selection doesn't change
public bool Select( Transform selection, SelectOptions selectOptions = SelectOptions.None ); // Selects the specified Transform. Returns true when the selection is changed successfully
public bool Select( IList<Transform> selection, SelectOptions selectOptions = SelectOptions.None ); // Selects the specified Transform(s)

public void Deselect(); // Deselects all Transforms
public void Deselect( Transform deselection ); // Deselects only the specified Transform
public void Deselect( IList<Transform> deselection ); // Deselects only the specified Transform(s)

public bool IsSelected( Transform transform ); // Returns true if the selection includes the Transform
You can access the currently selected object(s) in the hierarchy via the CurrentSelection property
您可以通过 CurrentSelection 属性访问层次结构中当前选定的对象。
Hierarchy's multi-selection toggles can be enabled manually via the MultiSelectionToggleSelectionMode property
可以通过 MultiSelectionToggleSelectionMode 属性手动启用层级结构的多选切换功能。
You can call the Refresh() function on the inspector and/or the hierarchy to refresh them manually
您可以手动调用检查器和/或层级视图上的 Refresh() 函数来刷新它们。
You can lock the inspector and/or the hierarchy via the IsLocked property
您可以通过 IsLocked 属性锁定检查器和/或层级结构。
You can register to the OnSelectionChanged event of the hierarchy to get notified when the selection has changed
您可以注册层级结构的 OnSelectionChanged 事件，以便在选择发生变化时收到通知。
You can register to the OnInspectedObjectChanging delegate of the inspector to get notified when the inspected object is about to change and, if you prefer, change the inspected object altogether. For example, if you want to inspect only objects that have a Renderer component attached, you can use the following function:
您可以注册到检查器的 OnInspectedObjectChanging 代理，以便在被检查对象即将更改时收到通知，如果您愿意，还可以直接更改被检查对象。例如，如果您只想检查附加了 Renderer 组件的对象，可以使用以下函数：
private object OnlyInspectObjectsWithRenderer( object previousInspectedObject, object newInspectedObject )
{
	GameObject go = newInspectedObject as GameObject;
	if( go != null && go.GetComponent<Renderer>() != null )
		return newInspectedObject;

	// Don't inspect objects without a Renderer component
	return null;
}
You can register to the ComponentFilter delegate of the inspector to filter the list of visible components of a GameObject in the inspector (e.g. hide some components)
您可以注册到检查器 ComponentFilter 委托，以过滤检查器中游戏对象的可见组件列表（例如，隐藏某些组件）。
runtimeInspector.ComponentFilter = ( GameObject gameObject, List<Component> components ) =>
{
    // Simply remove the undesired Components from the 'components' list
};
You can register to the GameObjectFilter delegate of the hierarchy to hide some objects from the hierarchy (or, you can add those objects to RuntimeInspectorUtils.IgnoredTransformsInHierarchy and they will be hidden from all hierarchies; just make sure to remove them from this HashSet before they are destroyed)
您可以向层级结构的 GameObjectFilter 代理注册，以隐藏层级结构中的某些对象（或者，您可以将这些对象添加到 RuntimeInspectorUtils.IgnoredTransformsInHierarchy 中，它们将从所有层级结构中隐藏；只需确保在销毁它们之前从该 HashSet 中移除它们）。
runtimeHierarchy.GameObjectFilter = ( Transform obj ) =>
{
    if( obj.CompareTag( "Main Camera" ) )
        return false; // Hide Main Camera from hierarchy
 
    return true;
};
You can register to the OnItemDoubleClicked event of the hierarchy to get notified when an object in the hierarchy is double clicked
您可以注册层级结构的 OnItemDoubleClicked 事件，以便在层级结构中的对象被双击时收到通知。
You can add RuntimeInspectorButton attribute to your functions to expose them as buttons in the inspector. These buttons appear when an object of that type is inspected. This attribute takes 3 parameters:
您可以为函数添加 RuntimeInspectorButton 特性，使其在检查器中显示为按钮。当检查该类型的对象时，这些按钮就会出现。此特性接受 3 个参数：
string label: the text that will appear on the button
字符串标签 ：按钮上将显示的文本
bool isInitializer: if set to true and the function returns an object that is assignable to the type that the function was defined in, the resulting value of the function will be assigned back to the inspected object. In other words, this function can be used to initialize null objects or change the variables of structs
bool isInitializer ：如果设置为 true，且函数返回的对象可以赋值给定义该函数的类型，则函数返回的值将被赋值给被检查的对象。换句话说，此函数可用于初始化空对象或更改结构体的变量。
ButtonVisibility visibility: determines when the button can be visible. Buttons with ButtonVisibility.InitializedObjects can appear only when the inspected object is not null whereas buttons with ButtonVisibility.UninitializedObjects can appear only when the inspected object is null. You can use ButtonVisibility.InitializedObjects | ButtonVisibility.UninitializedObjects to always show the button in the inspector
ButtonVisibility 属性决定按钮何时可见。值为 ButtonVisibility.InitializedObjects 的按钮仅在被检查对象不为空时显示，值为 ButtonVisibility.UninitializedObjects 的按钮仅在被检查对象为空时显示。您可以使用 ButtonVisibility.InitializedObjects | ButtonVisibility.UninitializedObjects 使按钮始终显示在检查器中。
Although you can't add RuntimeInspectorButton attribute to Unity's built-in functions, you can show buttons under built-in Unity types via extension methods. You must write all such extension methods in a single static class, mark the methods with RuntimeInspectorButton attribute and then introduce these functions to the RuntimeInspector as follows: RuntimeInspectorUtils.ExposedExtensionMethodsHolder = typeof( TheScriptThatContainsTheExtensionsMethods );
虽然不能将 RuntimeInspectorButton 特性添加到 Unity 的内置函数中，但可以通过扩展方法在 Unity 内置类型下显示按钮。您必须将所有此类扩展方法编写在一个静态类中，使用 RuntimeInspectorButton 特性标记这些方法，然后按如下方式将这些函数引入 RuntimeInspector： RuntimeInspectorUtils.ExposedExtensionMethodsHolder = typeof( TheScriptThatContainsTheExtensionsMethods );
F.1. PSEUDO-SCENES  F.1. 伪场景
You can use the following functions to add object(s) to pseudo-scenes in the hierarchy:
您可以使用以下函数向层级结构中的伪场景添加对象：

public void AddToPseudoScene( string scene, Transform transform );
public void AddToPseudoScene( string scene, IEnumerable<Transform> transforms );
These functions will create the relevant pseudo-scenes automatically if they do not exist.
如果相关的伪场景不存在，这些函数将自动创建它们。

You can use the following functions to remove object(s) from pseudo-scenes in the hierarchy:
您可以使用以下函数从层级结构中的伪场景中移除对象：

public void RemoveFromPseudoScene( string scene, Transform transform, bool deleteSceneIfEmpty );
public void RemoveFromPseudoScene( string scene, IEnumerable<Transform> transforms, bool deleteSceneIfEmpty );
You can use the following functions to create or delete a pseudo-scene manually:
您可以使用以下函数手动创建或删除伪场景：

public void CreatePseudoScene( string scene, Transform rootTransform = null );
public void DeletePseudoScene( string scene );
public void DeleteAllPseudoScenes();
The optional rootTransform parameter of CreatePseudoScene acts similar to PseudoSceneSourceTransform with the following differences:
CreatePseudoScene 的可选参数 rootTransform 与 PseudoSceneSourceTransform 的作用类似，但有以下区别：

Doesn't require adding a component to the source Transform
无需向源转换添加组件
When a Transform is dragged & dropped onto the pseudo-scene, its parent will actually be changed to rootTransform
当一个变换对象被拖放到伪场景中时，它的父对象实际上会更改为根变换对象。
During search, selected Transform's displayed path will stop at rootTransform (i.e. won't include its parents)
搜索过程中，所选变换的显示路径将止于根变换 （即不会包含其父变换）。
F.1.1. PseudoSceneSourceTransform
F.1.1. 伪场景源变换
This helper component allows you to add an object's children to a pseudo-scene in the hierarchy. When a child is added to or removed from the object, this component refreshes the pseudo-scene automatically. If HideOnDisable is enabled, the object's children are removed from the pseudo-scene when the object is disabled.
此辅助组件允许您将对象的子对象添加到层级结构中的伪场景。当子对象被添加或移除时，此组件会自动刷新伪场景。如果启用了 “禁用时隐藏” 功能，则当对象被禁用时，其子对象也会从伪场景中移除。

F.2. COLOR PICKER  F.2. 颜色选择器
You can access the built-in color picker via ColorPicker.Instance and then present it with the following function:
您可以通过 ColorPicker.Instance 访问内置颜色选择器，然后使用以下函数显示它：

public void Show( ColorWheelControl.OnColorChangedDelegate onColorChanged, ColorWheelControl.OnColorChangedDelegate onColorConfirmed, Color initialColor, Canvas referenceCanvas );
onColorChanged: invoked regularly as the user changes the color. ColorWheelControl.OnColorChangedDelegate takes a Color32 parameter
onColorChanged ：当用户更改颜色时定期调用。 ColorWheelControl.OnColorChangedDelegate 接受一个 Color32 参数。
onColorConfirmed: invoked when user submits the color via OK button
onColorConfirmed ：当用户通过 “确定” 按钮提交颜色时调用。
initialColor: the initial value of the color picker
initialColor ：颜色选择器的初始值
referenceCanvas: if assigned, the reference canvas' properties will be copied to the color picker canvas
referenceCanvas ：如果指定，则参考画布的属性将复制到颜色选择器画布。
You can change the color picker's visual appearance by assigning a UISkin to its Skin property.
您可以通过将 UISkin 分配给颜色选择器的 Skin 属性来更改其视觉外观。

F.3. OBJECT REFERENCE PICKER
F.3. 对象引用选择器
You can access the built-in object reference picker via ObjectReferencePicker.Instance and then present it with the following function:
您可以通过 ObjectReferencePicker.Instance 访问内置的对象引用选择器，然后使用以下函数显示它：

public void Show( ReferenceCallback onReferenceChanged, ReferenceCallback onSelectionConfirmed, NameGetter referenceNameGetter, NameGetter referenceDisplayNameGetter, object[] references, object initialReference, bool includeNullReference, string title, Canvas referenceCanvas );
onReferenceChanged: invoked when the user selects a reference from the list. ReferenceCallback takes an object parameter
onReferenceChanged ：当用户从列表中选择一个引用时调用。ReferenceCallback 接受一个对象参数 ReferenceCallback
onSelectionConfirmed: invoked when user submits the selected reference via OK button
onSelectionConfirmed ：当用户通过 “确定” 按钮提交所选参考文献时调用。
referenceNameGetter: NameGetter takes an object parameter and returns that object's name as string. The passed function will be used to sort the references list and compare the references' names with the search string
referenceNameGetter ： NameGetter 接受一个对象参数，并返回该对象的名称字符串。传入的函数将用于对引用列表进行排序，并将引用名称与搜索字符串进行比较。
referenceDisplayNameGetter: the passed function will be used to get display names for the references. Usually, the same function is passed to this parameter and the referenceNameGetter parameter
referenceDisplayNameGetter ：传入的函数将用于获取引用的显示名称。通常，此参数和 referenceNameGetter 参数会传入同一个函数。
references: array of references to pick from
参考文献 ：要从中选取的参考文献数组
initialReference: initially selected reference
initialReference ：初始选择的引用
includeNullReference: is set to true, a null reference option will be added to the top of the references list
includeNullReference ：如果设置为 true ，则会在引用列表顶部添加一个空引用选项。
title: title of the object reference picker
标题 ：对象引用选择器的标题
referenceCanvas: if assigned, the reference canvas' properties will be copied to the object reference picker canvas
referenceCanvas ：如果已赋值，则引用画布的属性将被复制到对象引用选择器画布。
You can change the object reference picker's visual appearance by assigning a UISkin to its Skin property.
您可以通过将 UISkin 分配给对象的 Skin 属性来更改对象引用选择器的视觉外观。

F.4. DRAGGED REFERENCE ITEMS
F.4. 拖拽的参考项目
In section E.2, it is mentioned that you can drag&drop objects from the hierarchy to the variables in the inspector to assign these objects to those variables. However, you are not limited with just hierarchy. There are two helper components that you can use to create dragged reference items for other objects:
在 E.2 节中提到，您可以将对象从层级结构拖放到检查器中的变量上，从而将这些对象分配给相应的变量。但是，您并不局限于层级结构。您可以使用两个辅助组件来创建其他对象的拖拽引用项：

DraggedReferenceSourceCamera: when attached to a camera, casts a ray to your scene at each mouse click and creates a dragged reference item if you hold on an object for a while. You can register to the ProcessRaycastHit delegate of this component to filter the objects than can create a dragged reference item. For example, if you want only objects with tag NPC to be able to create a dragged reference item, you can use the following function:
DraggedReferenceSourceCamera ：当附加到摄像机时，每次鼠标点击都会向场景投射一条射线，如果按住某个对象一段时间，则会创建一个拖动参考项。您可以注册此组件的 ProcessRaycastHit 代理来筛选可以创建拖动参考项的对象。例如，如果您只想让带有 NPC 标签的对象能够创建拖动参考项，可以使用以下函数：
private Object CreateDraggedReferenceItemForNPCsOnly( RaycastHit hit )
{
	if( hit.collider.gameObject.CompareTag( "NPC" ) )
		return hit.collider.gameObject;

	// Non-NPC objects can't create dragged reference items
	return null;
}
DraggedReferenceSourceUI: when assigned to a UI element, that element can create a dragged reference item for its References object(s) after it is clicked and held for a while
DraggedReferenceSourceUI ：当分配给一个 UI 元素时，该元素在被点击并按住一段时间后，可以为其 References 对象创建一个拖动引用项。
You can also use your own scripts to create dragged reference items by calling the following functions in the RuntimeInspectorUtils class:
您还可以通过调用 RuntimeInspectorUtils 类中的以下函数，使用自己的脚本创建拖动的参考项：

public static DraggedReferenceItem CreateDraggedReferenceItem( Object reference, PointerEventData draggingPointer, UISkin skin = null );
public static DraggedReferenceItem CreateDraggedReferenceItem( Object[] references, PointerEventData draggingPointer, UISkin skin = null, Canvas referenceCanvas = null );
G. CUSTOM DRAWERS (EDITORS)
G. 自定义抽屉（编辑）
NOTE: if you just want to hide some fields/properties from the RuntimeInspector, simply use Settings asset's Hidden Variables list (mentioned in section E.1).
注意： 如果您只想从 RuntimeInspector 中隐藏某些字段/属性，只需使用 “设置” 资源的 “隐藏变量” 列表（在 E.1 节中提到）。

You can introduce your own custom drawers to RuntimeInspector. These drawers will then be used to draw inspected objects' properties in RuntimeInspector. If no custom drawer is specified for a type, built-in ObjectField will be used to draw all properties of that type. There are 2 ways to create custom drawers:
您可以向 RuntimeInspector 添加自定义抽屉。这些抽屉将用于在 RuntimeInspector 中绘制被检查对象的属性。如果未为某个类型指定自定义抽屉，则将使用内置的 ObjectField 来绘制该类型的所有属性。创建自定义抽屉有两种方法：

Creating a drawer prefab and adding it to the Settings asset mentioned in section E.1. Each drawer extends from InspectorField base class. There is also an ExpandableInspectorField abstract class that allows you to create an expandable/collapsable drawer like arrays. Lastly, extending ObjectReferenceField class allows you to create drawers that can be assigned values via the reference picker or via drag&drop
创建抽屉预制件并将其添加到 E.1 节中提到的设置资源中。每个抽屉都继承自 InspectorField 基类。此外，还有一个 ExpandableInspectorField 抽象类，允许您创建类似数组的可展开/可折叠抽屉。最后，继承 ObjectReferenceField 类允许您创建可以通过引用选择器或拖放操作赋值的抽屉。
This option provides the most flexibility because you'll be able to customize the drawer prefab as you wish. The downside is, you'll have to create a prefab asset and manually add it to RuntimeInspector's Settings asset. All built-in drawers use this method; they can be as simple as BoolField and TransformField, or as complex as BoundsField, GameObjectField and ArrayField
此选项提供了最大的灵活性，因为您可以根据需要自定义抽屉预制件。缺点是，您必须创建一个预制件资源，并手动将其添加到 RuntimeInspector 的设置资源中。所有内置抽屉都使用此方法；它们可以像 BoolField 和 TransformField 一样简单，也可以像 BoundsField 、 GameObjectField 和 ArrayField 一样复杂。
Extending IRuntimeInspectorCustomEditor interface and decorating the class/struct with RuntimeInspectorCustomEditor attribute
扩展 IRuntimeInspectorCustomEditor 接口，并使用 RuntimeInspectorCustomEditor 特性修饰类/结构体。
This option is simpler because you won't have to create a prefab asset for the drawer. Created custom drawer will internally be used by ObjectField to populate its sub-drawers. This option should be sufficient for most use-cases. But imagine that you want to create a custom drawer for Matrix4x4 where the cells are displayed in a 4x4 grid. In this case, you must use the first option because you'll need a custom prefab with 16 InputFields organized in a 4x4 grid for it. But if you can represent the custom drawer you have in mind by using a combination of built-in drawers, then this second option should suffice
此选项更简单，因为您无需为抽屉创建预制件资源。ObjectField 内部会使用创建的自定义抽屉来填充其子抽屉。此选项足以满足大多数使用场景。但假设您想为 Matrix4x4 创建一个自定义抽屉，其中单元格以 4x4 网格形式显示。在这种情况下，您必须使用第一个选项，因为您需要一个包含 16 个以 4x4 网格排列的 InputField 的自定义预制件。但是，如果您可以通过组合使用内置抽屉来表示您设想的自定义抽屉，那么第二个选项就足够了。
G.1. InspectorField  G.1. 检查字段
To have a standardized visual appearance across all the drawers, there are some common variables for each drawer:
为了使所有抽屉的外观保持一致，每个抽屉都采用了一些通用变量：

Layout Element: is used to set the height of the drawer. A standard height is set by the currently active Inspector skin's Line Height property. This value is multiplied by the virtual HeightMultiplier property of the drawer. For ExpandableInspectorField's of unknown height, this variable should be left unassigned
布局元素 ：用于设置抽屉的高度。标准高度由当前激活的检查器皮肤的 “行高” 属性设置。该值乘以抽屉的虚拟 “高度乘数” 属性。对于高度未知的 ExpandableInspectorField，此变量应保持未赋值状态。
Variable Name Text: the Text object that displays the name of the exposed variable
变量名称文本 ：显示已公开变量名称的文本对象
Variable Name Mask: to understand this one, you may have to examine a simple drawer like BoolField. An Image is drawn on top of the Variable Name Text in order to mask its visible area in an efficient way. And this mask is assigned to this variable
变量名掩码 ：要理解这一点，您可能需要查看一个简单的绘图器，例如 BoolField。在变量名文本上方绘制一个图像 ，以有效地遮盖其可见区域。并且此掩码被分配给该变量。
Each drawer has access to the following properties:
每个抽屉都可以访问以下属性：

object Value: the most recent value of the variable that this drawer is bound to. It is refreshed at each refresh interval of the inspector. Changing this property will also change the bound object
对象值 ：此抽屉绑定的变量的最新值。它会在检查器每次刷新间隔时刷新。更改此属性也会更改绑定的对象。
RuntimeInspector Inspector: the RuntimeInspector that currently uses this drawer
RuntimeInspector 检查器 ：当前使用此抽屉的 RuntimeInspector
UISkin Skin: the skin that is assigned to this drawer
UISkin 皮肤 ：分配给此抽屉的皮肤
Type BoundVariableType: the type of the bound object
BoundVariableType 类型 ：绑定对象的类型
int Depth: the depth that this drawer is drawn at. As Depth increases, a padding should be applied to the contents of this drawer from left (in OnDepthChanged function)
int Depth ：此抽屉的绘制深度。随着 Depth 的增加，应在此抽屉的内容左侧添加内边距（在 OnDepthChanged 函数中）。
string Name: the name of the bound variable. When set, the variable name is converted to title case format if Use Title Case Naming is enabled in the inspector
字符串名称 ：绑定变量的名称。如果在检查器中启用了 “使用首字母大写命名” ，则设置此参数后，变量名称将转换为首字母大写格式。
string NameRaw: When set, the variable name is used as is without being converted to title case format
string NameRaw ：设置后，变量名将按原样使用，而不转换为首字母大写格式。
float HeightMultiplier: affects the height of the drawer
float HeightMultiplier ：影响抽屉的高度
There are some special functions on drawers that are invoked on certain circumstances:
抽屉程序有一些特殊功能，会在特定情况下调用：

void Initialize(): should be used instead of Awake/Start to initialize the drawer
void Initialize() ：应该使用该方法代替 Awake / Start 来初始化抽屉。
bool SupportsType( Type type ): returns whether or not this drawer can expose (supports) a certain type in the inspector
bool SupportsType(类型 type) : 返回此抽屉是否可以在检查器中显示（支持）特定类型
bool CanBindTo( Type type, MemberInfo variable ): returns whether or not this drawer can expose the provided variable. This function is called only if SupportsType returns true. This function is useful for drawers that can expose only variables with specific attribute(s) (e.g. NumberRangeField queries RangeAttribute). Please note that the variable parameter can be null. By default, this function returns true
bool CanBindTo(Type type, MemberInfo variable) : 返回此抽屉是否可以公开提供的变量 。仅当 SupportsType 返回 true 时才会调用此函数。此函数适用于只能公开具有特定属性的变量的抽屉（例如， NumberRangeField 查询 RangeAttribute）。请注意， variable 参数可以为 null 。默认情况下，此函数返回 true。
void OnBound( MemberInfo variable ): called when the drawer is bound to a variable via reflection. Please note that the variable parameter can be null
void OnBound( MemberInfo variable ) ：当抽屉通过反射绑定到变量时调用。请注意， 变量参数可以为空。
void OnUnbound(): called when the drawer is unbound from the variable that it was bound to
void OnUnbound() ：当抽屉从其绑定的变量中解除绑定时调用
void OnInspectorChanged(): called when the Inspector property of the drawer is changed
void OnInspectorChanged() ：当抽屉的 Inspector 属性更改时调用
void OnSkinChanged(): called when the Skin property of the drawer is changed. Your custom drawers must adjust their UI elements' visual appearance here to comply with the assigned skin's standards
void OnSkinChanged() ：当抽屉的 Skin 属性发生更改时调用。您的自定义抽屉必须在此处调整其 UI 元素的视觉外观，以符合所分配皮肤的标准。
void OnDepthChanged(): called when the Depth property of the drawer is changed. Here, your custom drawers must add a padding to their content from left to comply with the nesting standard. This function is also called when the Skin changes
void OnDepthChanged() ：当抽屉的 Depth 属性发生变化时调用。此时，您的自定义抽屉必须为其内容添加左侧内边距，以符合嵌套标准。当皮肤发生变化时，也会调用此函数。
void Refresh(): called when the value of the bound object is refreshed. Drawers must refresh the values of their UI elements here. Invoked by RuntimeInspector at every Refresh Interval seconds
void Refresh() ：当绑定对象的值刷新时调用。抽屉必须在此处刷新其 UI 元素的值。RuntimeInspector 每隔 Refresh Interval 秒调用一次。
G.2. ExpandableInspectorField
G.2. 可扩展检查器字段
Custom drawers that extend ExpandableInspectorField have access to the following properties:
继承自 ExpandableInspectorField 的自定义抽屉可以访问以下属性：

bool IsExpanded: returns whether the drawer is expanded or collapsed. When set to true, the drawer is expanded and its contents are drawn under it
bool IsExpanded ：返回抽屉是展开还是折叠状态。设置为 true 时，抽屉展开，其内容显示在抽屉下方。
HeaderVisibility HeaderVisibility: sets the visibility of this drawer's header: Collapsible, AlwaysVisible or Hidden. By default, this value is set to Collapsible
HeaderVisibility ：设置此抽屉标题的可见性： 可折叠 、 始终可见或隐藏 。默认值为可折叠。
int Length: the number of elements that this drawer aims to draw. If its value does not match the number of child drawers that this drawer has, the contents of the drawer are regenerated
int Length ：此抽屉要绘制的元素数量。如果其值与此抽屉拥有的子抽屉数量不匹配，则重新生成抽屉的内容。
ExpandableInspectorField has the following special functions:
ExpandableInspectorField 具有以下特殊功能：

void GenerateElements(): the sub-drawers of this drawer must be generated here
void GenerateElements() ：必须在此处生成此抽屉的子抽屉。
void ClearElements(): the sub-drawers of this drawer must be cleared here
void ClearElements() ：必须在此处清除此抽屉的子抽屉。
Sub-drawers of an ExpandableInspectorField should be stored in the protected List<InspectorField> elements variable as ExpandableInspectorField uses this list to compare the number of sub-drawers with the Length property. When Refresh() is called, sub-drawers in this list are refreshed automatically and when ClearElements() is called, sub-drawers in this list are cleared automatically.
ExpandableInspectorField 的子抽屉应存储在 protected List<InspectorField> elements 变量中，因为 ExpandableInspectorField 使用此列表将子抽屉的数量与 Length 属性进行比较。 调用 Refresh() 时，此列表中的子抽屉会自动刷新；调用 ClearElements() 时，此列表中的子抽屉会自动清除。

You can create sub-drawers using the RuntimeInspector.CreateDrawerForType( Type type, Transform drawerParent, int depth, bool drawObjectsAsFields = true ) function. If no drawer is found that can expose this type, the function returns null. Here, for ExpandableInspectorFields, the drawerParent parameter should be set as the drawArea variable of the ExpandableInspectorField. If the drawObjectsAsFields parameter is set to true and if the type extends UnityEngine.Object, Reference Drawers are searched for a drawer that supports this type. Otherwise Standard Drawers are searched.
您可以使用 RuntimeInspector.CreateDrawerForType( Type type, Transform drawerParent, int depth, bool drawObjectsAsFields = true ) 函数创建子绘制器。如果找不到可以公开此类型的绘制器，该函数将返回 null 。对于 ExpandableInspectorFields， drawerParent 参数应设置为 ExpandableInspectorField 的 drawArea 变量。如果 drawObjectsAsFields 参数设置为 true，并且该类型继承自 UnityEngine.Object ，则会在引用绘制器中搜索支持此类型的绘制器。否则，将搜索标准绘制器 。

After creating sub-drawers, ExpandableInspectorFields must bind their sub-drawers to their corresponding variables manually. This is done via the following BindTo functions of the InspectorField class:
创建子抽屉后， ExpandableInspectorField 必须手动将其子抽屉绑定到相应的变量。这可以通过 InspectorField 类的以下 BindTo 函数完成：

BindTo( InspectorField parent, MemberInfo variable, string variableName = null ): binds the object to a MemberInfo (it can be received via reflection). Here, parent parameter should be set to this ExpandableInspectorField. If variableName is set to null, its value is fetched directly from the MemberInfo parameter
BindTo( InspectorField parent, MemberInfo variable, string variableName = null ) ：将对象绑定到 MemberInfo （可通过反射获取）。此处， parent 参数应设置为此 ExpandableInspectorField 。如果 variableName 设置为 null，则其值直接从 MemberInfo 参数中获取。
BindTo( Type variableType, string variableName, Getter getter, Setter setter, MemberInfo variable = null ): this one allows you to define your own getter and setter functions for this sub-drawer. For example, ArrayField uses this function because there is no direct MemberInfo to access an element of an array. With this method, you can use custom functions instead of MemberInfos to get/set the values of the bound objects (ArrayField uses Array.GetValue for its elements' getter function and Array.SetValue for its elements' setter function)
BindTo( Type variableType, string variableName, Getter getter, Setter setter, MemberInfo variable = null ) ：此选项允许您为该子抽屉定义自定义的 getter 和 setter 函数。例如， ArrayField 使用此函数，因为没有直接的 MemberInfo 可以访问数组元素。通过此方法，您可以使用自定义函数而不是 MemberInfo 来获取/设置绑定对象的值（ArrayField 使用 Array.GetValue 作为其元素的 getter 函数，使用 Array.SetValue 作为其元素的 setter 函数）。
There are also some helper functions in ExpandableInspectorField to easily create sub-drawers without having to call CreateDrawerForType or BindTo manually:
ExpandableInspectorField 中还有一些辅助函数，可以轻松创建子抽屉，而无需手动调用 CreateDrawerForType 或 BindTo ：

InspectorField CreateDrawerForComponent( Component component, string variableName = null ): creates a Standard Drawer for a component
InspectorField CreateDrawerForComponent( Component component, string variableName = null ) ：为组件创建一个标准抽屉。
InspectorField CreateDrawerForVariable( MemberInfo variable, string variableName = null ): creates a drawer for the variable that the MemberInfo stores. This variable must be declared inside inspected object's class/struct or one of its base classes
InspectorField CreateDrawerForVariable( MemberInfo variable, string variableName = null ) ：为 MemberInfo 存储的变量创建一个抽屉。此变量必须在被检查对象的类/结构体或其基类之一中声明。
InspectorField CreateDrawer( Type variableType, string variableName, Getter getter, Setter setter, bool drawObjectsAsFields = true ): similar to the BindTo function with the Getter and Setter parameters, allows you to use custom functions to get and set the value of the object that the sub-drawer is bound to
InspectorField CreateDrawer( Type variableType, string variableName, Getter getter, Setter setter, bool drawObjectsAsFields = true ) ：类似于带有 Getter 和 Setter 参数的 BindTo 函数，允许您使用自定义函数来获取和设置子抽屉绑定对象的值。
G.3. ObjectReferenceField
Drawers that extend ObjectReferenceField class have access to the void OnReferenceChanged( Object reference ) function that is called when the reference assigned to that drawer is changed.

G.4. Helper Classes
PointerEventListener: this is a simple helper component that invokes PointerDown event when its UI GameObject is pressed, PointerUp event when it is released and PointerClick event when it is clicked

BoundInputField: most of the built-in drawers use this component for their input fields. This helper component allows you to validate the input as it is entered and also get notified when the input is submitted. It has the following properties and functions:

string DefaultEmptyValue: the default value that the input field will have when its input is empty. For example, NumberField sets this value to "0"
string Text: a property to refresh the current value of the input field. If the input field is currently focused and being edited, then this property will not change its text immediately but store the value in a variable so that it can be used when the input field is no longer focused. Also, setting this property will not invoke the OnValueChanged event
UISkin Skin: the skin that this input field uses. When set, input field will adjust its UI accordingly
OnValueChangedDelegate OnValueChanged: called while the value of input field is being edited (called at each change to the input). The OnValueChangedDelegate has the following signature: bool OnValueChangedDelegate( BoundInputField source, string input ). A function that is registered to this event should parse the input and return true if the input is valid, false otherwise
OnValueChangedDelegate OnValueSubmitted: called when user finishes editing the value of input field. Similar to OnValueChanged, a function that is registered to this event should parse the input and return true only if the input is valid
bool CacheTextOnValueChange: determines what will happen when user stops editing the input field while its contents are invalid (i.e. its background has turned red). If this variable is set to true, input field's text will revert to the latest value that returned true for OnValueChanged. Otherwise, the text will revert to the value input field had when it was focused
G.5. RuntimeInspectorCustomEditor Attribute
To create drawers without having to create a prefab for it, you can declara a class/struct that extends IRuntimeInspectorCustomEditor and has one or more RuntimeInspectorCustomEditor attributes.

RuntimeInspectorCustomEditor attribute has the following properties:

Type inspectedType: the type this custom drawer supports (can expose)
bool editorForChildClasses: if set to true, types derived from inspectedType can also be drawn with this drawer. By default, this value is false
IRuntimeInspectorCustomEditor has the following functions:

void GenerateElements( ObjectField parent ): called by built-in ObjectField's GenerateElements function. Sub-drawers should be added to ObjectField in this function
void Refresh(): called by ObjectField's Refresh function
void Cleanup(): called by ObjectField's ClearElements function. If the drawer has created some disposable resources, they must be disposed here. No need to destroy the created sub-drawers here because it is handled by ObjectField automatically, as explained in ExpandableInspectorField section
Inside GenerateElements function, you can call parent parameter's CreateDrawerForComponent, CreateDrawerForVariable and CreateDrawer functions to create sub-drawers. In addition to these, you can also call the following helper functions of ObjectField:

void CreateDrawersForVariables( params string[] variables ): creates drawers for the specified variables of the inspected object. If no specific variables are provided, drawers will be created for all exposed variables of the inspected object
void CreateDrawersForVariablesExcluding( params string[] variablesToExclude ): creates drawers for all exposed variables of the inspected object except the variables specified in variablesToExclude list. If no variables are excluded, drawers will be created for all exposed variables of the inspected object
Here are some example custom drawers:

screenshot

// Custom drawer for Collider type and the types that derive from it
[RuntimeInspectorCustomEditor( typeof( Collider ), true )]
public class ColliderEditor : IRuntimeInspectorCustomEditor
{
	public void GenerateElements( ObjectField parent )
	{
		// Exposes only "enabled" and "isTrigger" properties of Colliders
		// Note that we could achieve the same thing by modifying the "Hidden Variables" and "Exposed Variables" lists of RuntimeInspector's Settings asset
		parent.CreateDrawersForVariables( "enabled", "isTrigger" );
	}

	public void Refresh() { }
	public void Cleanup() { }
}
screenshot

// Custom drawer for MeshRenderer type (but not the types that derive from it)
[RuntimeInspectorCustomEditor( typeof( MeshRenderer ), false )]
public class MeshRendererEditor : IRuntimeInspectorCustomEditor
{
	public void GenerateElements( ObjectField parent )
	{
		// Get the MeshRenderer object we are inspecting
		MeshRenderer renderer = (MeshRenderer) parent.Value;

		// Instead of exposing the MeshRenderer's properties, expose its sharedMaterial's properties
		ExpandableInspectorField materialField = (ExpandableInspectorField) parent.CreateDrawer( typeof( Material ), "", () => renderer.sharedMaterial, ( value ) => renderer.sharedMaterial = (Material) value, false );

		// The drawer for materials is, by default, an ExpandableInspectorField. We don't want to draw its collapsible header in this example
		materialField.HeaderVisibility = RuntimeInspector.HeaderVisibility.Hidden;
	}

	public void Refresh() { }
	public void Cleanup() { }
}
screenshot

// Custom drawer for Camera type (but not the types that derive from it)
[RuntimeInspectorCustomEditor( typeof( Camera ), false )]
public class CameraEditor : IRuntimeInspectorCustomEditor
{
	// Some of the sub-drawers that are created inside GenerateElements
	private BoolField isOrthographicField;
	private NumberField orthographicSizeField, fieldOfViewField;

	public void GenerateElements( ObjectField parent )
	{
		// Create sub-drawers for the Camera's "orthographic", "orthographicSize" and "fieldOfView" properties and store them in variables
		isOrthographicField = (BoolField) parent.CreateDrawerForVariable( typeof( Camera ).GetProperty( "orthographic", BindingFlags.Public | BindingFlags.Instance ), "Is Orthographic" );
		orthographicSizeField = (NumberField) parent.CreateDrawerForVariable( typeof( Camera ).GetProperty( "orthographicSize", BindingFlags.Public | BindingFlags.Instance ) );
		fieldOfViewField = (NumberField) parent.CreateDrawerForVariable( typeof( Camera ).GetProperty( "fieldOfView", BindingFlags.Public | BindingFlags.Instance ) );

		// Add additional indentation for "orthographicSize" and "fieldOfView" sub-drawers
		orthographicSizeField.Depth++;
		fieldOfViewField.Depth++;

		// Create sub-drawers for the rest of the exposed properties of the Camera
		parent.CreateDrawersForVariablesExcluding( "orthographic", "orthographicSize", "fieldOfView" );
	}

	public void Refresh()
	{
		// Check if Camera is currently using orthographic projection
		bool isOrthographicCamera = (bool) isOrthographicField.Value;

		// Show either "orthographicSize" sub-drawer or "fieldOfView" sub-drawer depending on camera's current projection type
		// (Here, we're first checking if the sub-drawer is already active/inactive via 'activeSelf' for optimization purposes because GameObject.SetActive
		// causes considerable GC allocations and unfortunately doesn't automatically check if GameObject is already active/inactive, at least on some Unity versions)
		if( orthographicSizeField.gameObject.activeSelf != isOrthographicCamera )
			orthographicSizeField.gameObject.SetActive( isOrthographicCamera );
		if( fieldOfViewField.gameObject.activeSelf == isOrthographicCamera )
			fieldOfViewField.gameObject.SetActive( !isOrthographicCamera );
	}

	public void Cleanup() { }
}
About
Runtime Inspector and Hierarchy solution for Unity for debugging and runtime editing purposes

Resources
 Readme
License
 MIT license
Contributing
 Contributing
 Activity
Stars
 2.1k stars
Watchers
 41 watching
Forks
 148 forks
Report repository
Releases 20
v1.7.4
Latest
on Mar 29, 2025
+ 19 releases
Sponsor this project
@yasirkula
yasirkula
https://yasirkula.itch.io/unity3d/donate
Learn more about GitHub Sponsors
Packages
No packages published
Contributors
2
@yasirkula
yasirkula
@i-xt
i-xt
Languages
C#
98.1%
 
ShaderLab
1.9%
Footer
© 2026 GitHub, Inc.
Footer navigation
Terms
Privacy
Security
Status
Community
Docs
Contact
Manage cookies
Do not share my personal information

