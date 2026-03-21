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
