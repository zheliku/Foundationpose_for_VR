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
