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
