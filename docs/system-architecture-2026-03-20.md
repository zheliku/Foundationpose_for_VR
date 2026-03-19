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
