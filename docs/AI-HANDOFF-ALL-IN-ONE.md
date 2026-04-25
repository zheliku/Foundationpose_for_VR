# Foundationpose_for_VR 项目总交接文档（唯一入口）

更新时间：2026-04-26

本文件是项目唯一长期维护的 AI 接手文档。历史会话记录已重新核对并压缩，后续请优先更新本文件，避免再产生多份互相冲突的交接文档。

## 1. 项目目标与当前主线

### 1.1 总目标

实现 VR/Quest 场景中的实时 6D 位姿估计与 Unity 可视化：

1. Unity/Quest 采集左右 Passthrough 相机图像与相机静态信息。
2. Python 服务端接收双目图和相机信息。
3. Python 侧执行 2D 分割、双目深度估计和 FoundationPose 6D 位姿估计。
4. Python 将位姿结果通过 ZMQ PUB topic 回传 Unity。
5. Unity 解码位姿，按发送帧号对齐当时相机/参考节点姿态后，把物体放回世界坐标。

### 1.2 当前主线链路

当前重点不是旧的单脚本 demo，而是结构化链路：

- Quest 主链路：Unity 多 topic 发送 → Python `pose_server.py` → Quest Pipeline → 位姿 topic 回传 Unity。
- RealSense 调试链路：RealSense 本机双目 → RealSense Pipeline → OpenCV 本地窗口调试。
- 深度主方案：Fast-FoundationStereo，默认优先 TensorRT，缺少 engine 时可按参数回退 PyTorch。
- 位姿主方案：FoundationPose register + track。
- 2D 主方案：YOLOE-26 语义分割，Cutie 可选用于后续帧 2D 跟踪辅助。

### 1.3 当前工程原则

- 链路可运行优先：协议变更必须 Python/Unity 同步。
- 模块边界清晰：相机、网络、编码、Pipeline、位姿应用互相解耦。
- 可观测性必须保留：阶段、phase、检测数、深度有效率、耗时分项、FPS、收发统计。
- 不恢复旧兼容分支：旧入口、旧 topic/PUSH/PULL、旧 packed 图像协议、旧 TRT legacy 命名均不应回流。

## 2. 当前可运行入口

### 2.1 Python Quest 位姿服务（推荐主入口）

运行目录：`Foundationpose_for_VR`

```powershell
pixi run python .\src\pose_server.py
```

职责：

- 接收 Unity 发来的 `quest_stereo` 与 `quest_camera_info` 两个 topic。
- 构建并运行 Quest Pipeline。
- 缓存 `camera_info_latest.json`。
- 发布 `pose` topic 给 Unity。
- 可选显示本地 OpenCV 调试窗口。

常用参数：

```powershell
pixi run python .\src\pose_server.py --run_stage 4 --camera_source network --local_debug 1
```

### 2.2 Python Quest Pipeline 本地调试入口

```powershell
pixi run python .\src\pipeline\quest_pipeline.py
```

说明：

- 只跑 Quest 输入 → YOLO → FFS → FoundationPose 的 pipeline 示例。
- 不负责将 pose 回传 Unity；回传请使用 `pose_server.py`。

### 2.3 Python RealSense Pipeline 本地调试入口

```powershell
pixi run python .\src\pipeline\realsense_pipeline.py
```

说明：

- 用 RealSense 左右红外作为双目输入。
- 适合在无 Quest/Unity 联调时验证 YOLO、FFS、FoundationPose 基础能力。

### 2.4 pixi 任务入口

`pixi.toml` 当前有效任务：

- `pixi run build`：构建 FoundationPose C++ 扩展、导出 ONNX、构建 TRT engine。
- `pixi run demo-yoloe`：运行 RealSense pipeline 测试 YOLOE/主链路。
- `pixi run demo-pipeline`：当前已指向 `src/pipeline/quest_pipeline.py`。

注意：旧 `src/quest_stereo_pose_pipeline.py` 不存在，不能再作为任务入口。

## 3. 分阶段调试口径

Quest 与 RealSense Pipeline 统一使用 4 个阶段：

1. stage 1：仅输入图像。
2. stage 2：输入 + YOLOE 2D 分割。
3. stage 3：输入 + YOLOE + Fast-FoundationStereo 深度。
4. stage 4：完整链路，包含 FoundationPose register/track。

窗口热键：

- `1/2/3/4`：切换阶段。
- `r`：重置跟踪状态，下一次有效 mask 会重新 register。
- `q` 或 `ESC`：退出。

## 4. 当前代码结构

### 4.1 Python 核心模块

- `src/modules/realsense.py`
  - 封装 `pyrealsense2`。
  - 对外提供 `RealSenseCamera.get_stereo_frames()`、`get_aligned_rgbd_frames()`、`get_stereo_calibration()`。
  - 调用方不应直接访问 RealSense SDK 或 `camera.pipeline`。

- `src/modules/quest_io.py`
  - Quest 多 topic 接收模块。
  - 对外提供 `QuestReceiver.get_stereo_frames()`、`get_camera_info()`、`get_calibration()`。
  - 内部使用 `recv_all_latest_by_topic()`，避免多 topic drain 丢消息。

- `src/modules/yoloe26.py`
  - YOLOE-26 分割封装。
  - 输入 BGR/灰度图，输出 overlay、二值 mask、检测数和耗时。

- `src/modules/fast_foundationstereo.py`
  - Fast-FoundationStereo 实时深度封装。
  - 内部有 `_PyTorchStereoBackend` 与 `_TrtStereoBackend` 双后端。
  - 默认可优先 TRT；TRT 不可用时由 `trt_strict` 控制是否回退 PyTorch。

- `src/modules/foundationpose.py`
  - FoundationPose 封装。
  - 提供 `register()`、`track()`、`visualize_pose()`、`adjust_pose_to_image_point()`。
  - 已隔离 `FoundationPose.Utils` 与 Fast-FoundationStereo `Utils` 同名导入冲突。

- `src/modules/cutie.py`
  - Cutie 2D tracker 封装。
  - 可用 YOLO mask 初始化，后续输出 bbox/mask 辅助 FoundationPose 跟踪。

### 4.2 Python 编排层

- `src/pipeline/quest_pipeline.py`
  - Quest 输入完整算法链路。
  - `camera_source=network` 默认会先尝试读取 `Calibration/cache/camera_info_latest.json` 预初始化 K/FoundationPose，随后收到网络 camera_info 时校验并按需刷新。
  - `--preload_camera_cache 0` 可关闭该快速启动策略，恢复为严格等待网络 camera_info。
  - `camera_source=local` 时仅优先读 `Calibration/cache/camera_info_latest.json`；失败后仍等待网络 camera_info。

- `src/pipeline/realsense_pipeline.py`
  - RealSense 输入完整算法链路。
  - 首帧运行时懒读取 RealSense 标定并构建 FoundationPoseEstimator。

- `src/pose_server.py`
  - 当前 Quest 端到端服务主入口。
  - 负责 pose 发布、camera_info 缓存、延迟统计、本地 debug 窗口和键盘控制。

### 4.3 Python 网络与协议层

- `src/zmq_utils/communicate/sender.py`
  - `PayloadSender`，统一 PUB 模式。
  - `send_payload(payload, topic)` 必须显式指定 topic。

- `src/zmq_utils/communicate/receiver.py`
  - `PayloadReceiver`，统一 SUB 模式。
  - 所有接收场景统一使用 `recv_all_latest_by_topic()`，按 topic 分别保留最新 payload。

- `src/zmq_utils/payload/message/*.py`
  - MessagePack 消息定义：`PoseMsg`、`QuestStereoMsg`、`QuestCameraInfoMsg`。

- `src/zmq_utils/payload/encoder/*.py` / `decoder/*.py`
  - 业务对象和 MessagePack payload 之间的转换层。

### 4.4 Unity 侧脚本

- `Assets/Scripts/Net/Communicate/PayloadSender.cs`
  - 多 `SenderEntry` PUB 发送器。
  - 每个 Entry 独立绑定 `encoder + topic + targetFps`。
  - Quest 发送端通常配置：
    - `quest_stereo`：绑定 `QuestStereoEncoder`，高频。
    - `quest_camera_info`：绑定 `QuestCameraInfoEncoder`，低频。

- `Assets/Scripts/Net/Communicate/PayloadReceiver.cs`
  - 多 `ReceiverEntry` SUB 接收器。
  - 后台线程按 topic 分别缓存最新 payload，主线程分发到对应 decoder。
  - Unity 接收 pose 时通常订阅 `pose` topic 并绑定 `PoseDecoder`。

- `Assets/Scripts/Net/Payload/Encoder/QuestStereoEncoder.cs`
  - 读取左右 PassthroughCameraAccess 纹理。
  - 左右图分别 JPEG 编码，封装为 `QuestStereoMsg`。
  - 编码成功后触发 `OnFrameEncoded(frame_id)`，供 `PoseFollow` 缓存发送时参考姿态。

- `Assets/Scripts/Net/Payload/Encoder/QuestCameraInfoEncoder.cs`
  - 读取左右相机内参、分辨率、镜头偏移、基线等静态信息。
  - 每次编码都会刷新 `sender_mono_ms`，发送频率由 `PayloadSender` 的 `targetFps` 控制。

- `Assets/Scripts/Net/Payload/Decoder/PoseDecoder.cs`
  - 解码 Python 回传的 `PoseMsg`。
  - 默认执行 OpenCV 相机坐标到 Unity 坐标的转换。
  - 有效位姿通过 `OnPoseReceived(Pose, frame_id)` 事件派发。

- `Assets/Scripts/PoseFollow.cs`
  - 消费 `PoseDecoder` 事件。
  - 通过 `frame_id` 查找 `QuestStereoEncoder.OnFrameEncoded` 时缓存的 sourceTarget 姿态。
  - 将 Python 回传的相机系位姿转换到 Unity 世界坐标并应用到当前 Transform。

- `Assets/Scripts/PoseSmoother.cs`
  - `PoseFollow` 的指数平滑插件。
  - 平滑逻辑按 Unity `Update()` 频率运行，不被网络回包帧率限制。

- `Assets/Scripts/PcaApiInfoDumper.cs`
  - PassthroughCameraAccess API 信息导出工具。
  - 用于排查左右相机分辨率、内参、镜头偏移和纹理状态。

- `Assets/Scripts/CameraViewerManager.cs`
  - 本地 UI 显示左右 Passthrough 相机纹理，便于 Quest 端可视化联调。

## 5. 当前网络协议与 topic 约定

### 5.1 总体约定

Python 与 Unity 当前统一为：

- ZMQ PUB/SUB。
- multipart `[topic, payload]`。
- payload 为单帧 MessagePack bytes。
- 不再使用 PUSH/PULL。
- 不再使用 multipart 业务分片。
- 不再保留旧 JSON pose 路径。

### 5.2 Topic 名称

- `quest_stereo`
  - Unity → Python。
  - 高频双目 JPEG 图像。
  - 消息类型：`QuestStereoMsg`。

- `quest_camera_info`
  - Unity → Python。
  - 低频相机静态信息。
  - 消息类型：`QuestCameraInfoMsg`。

- `pose`
  - Python → Unity。
  - 位姿与调试状态。
  - 消息类型：`PoseMsg`。

### 5.3 多 topic drain 关键点

多 topic 场景严禁用“不区分 topic 的 drain”作为主循环消费方式，否则可能只保留最后一个 topic，导致另一个 topic 被丢弃。

当前修复策略：

- Python：`PayloadReceiver.recv_all_latest_by_topic()` 按 topic 保存最新 payload。
- QuestReceiver：`poll_all()` 使用 `recv_all_latest_by_topic()`。
- Unity：`PayloadReceiver.ReceiveLoop()` drain 时每条消息按 topic 写入 `_latestByTopic`。

### 5.4 HWM 经验值

Quest stereo 帧较大，且 `pose_server` 启动阶段会初始化 TRT/FoundationPose/Warp，可能持续数秒。

当前建议：

- Python Quest 接收端 `recv_hwm` 默认 20。
- Unity stereo 发送端不要把发送 HWM 设得过大，避免排队延迟。
- pose 发布端可用较低 HWM（如 1），保证 Unity 侧只消费最新 pose。

## 6. 标定与相机信息缓存

### 6.1 Quest camera_info 字段

`QuestCameraInfoMsg` 当前包含：

- `is_supported`。
- 左右目 `fx/fy/cx/cy`。
- 左右畸变数组（Quest 当前通常为空）。
- `baseline_m`。
- `sensor_width/sensor_height`。
- `active_left/top/right/bottom`。
- 左右 `RequestedResolution`。
- `current_width/current_height`。
- `max_framerate`。
- 左右 `LensOffset` 的 position 与 quaternion。
- `sender_mono_ms`。

### 6.2 Python 标定构造方式

- 网络与缓存统一使用 `QuestStereoCalibration.from_camera_info_msg()`。
- Pipeline 通过 `QuestReceiver.get_calibration()` 获取网络标定，通过 `camera_info_latest.json` 预加载缓存标定。

### 6.3 缓存策略

`pose_server.py` 每次发现新的 camera_info digest 后：

1. 将消息转为 JSON dict。
2. 写入 `Calibration/cache/camera_info_latest.json`。
3. 若与旧 latest 核心内容不同，先把旧文件备份为 `camera_info_<timestamp>.json`。
4. 若内容不变，仅更新 `_received_at`。

Pipeline 启动策略：

- `camera_source=network` + `preload_camera_cache=1`（默认）：
  - 启动时先读取 `Calibration/cache/camera_info_latest.json`。
  - 若缓存存在，则立即初始化 K 与 FoundationPoseEstimator。
  - 此时只要收到 `quest_stereo` 就可以开始估计 pose，不必阻塞等待本次会话的 `quest_camera_info`。
  - 后续收到网络 `quest_camera_info` 后，会与当前标定签名比较；若不同且 `network_calib_update=1`，则刷新 K/PoseEstimator 并重置跟踪状态。
- `camera_source=network` + `preload_camera_cache=0`：严格等待网络 `quest_camera_info` 后再初始化。
- `camera_source=local`：优先读缓存 latest；若失败，再等待网络 camera_info。

### 6.4 K 映射策略

Quest 标定可能来自 sensor/active array 分辨率，而算法处理分辨率通常为 640x480。

当前 `QuestStereoCalibration.scaled_k()` 支持：

- `calib_assume_center_crop=1`：中心裁剪 + 缩放映射（默认）。
- `calib_assume_center_crop=0`：仅线性缩放。

注意：`quest_pipeline._preprocess_stereo_pair()` 只缩放实际接收图像，不会把 640x480 图像扩回 active array 再二次裁剪。

## 7. 深度与 TensorRT 口径

### 7.1 Fast-FoundationStereo 后端

`FastFoundationStereoRealtime` 当前结构：

- `_PyTorchStereoBackend`：加载 `.pth` 并执行 PyTorch 推理。
- `_TrtStereoBackend`：按输入尺寸、迭代次数、最大视差和平台标签匹配 TRT engine。
- `FastFoundationStereoRealtime.predict_depth()`：统一入口，输出米制深度图。

### 7.2 TRT/ONNX 命名规则

当前使用参数化 tag，不再生成/回退 legacy 名称。

- tag：`h{height}-w{width}-it{valid_iters}-md{max_disp}`。
- ONNX：`feature_runner-{tag}.onnx`、`post_runner-{tag}.onnx`。
- Engine：`feature_runner-{tag}.{platform}.{precision}.engine`、`post_runner-{tag}.{platform}.{precision}.engine`。

运行时匹配顺序：

1. 显式传入的 engine path。
2. `{runner}-{tag}.{platform}.{precision}.engine`。
3. `{runner}-{tag}.{platform}.engine`。
4. `{runner}-{tag}.engine`。

### 7.3 TRT 配置口径

- 不再依赖 `onnx.yaml`。
- 运行时直接由参数构造 `OmegaConf`。
- `trt_strict=1` 时 TRT 不可用直接报错。
- `trt_strict=0` 时缺 engine 或初始化失败会回退 PyTorch。

### 7.4 FFS 耗时字段

`predict_depth(return_timing=True)` 返回：

- `prep_ms`：numpy → tensor、缩放、维度转换等预处理。
- `forward_ms`：模型/engine 前向推理。
- `post_ms`：回 CPU、视差转深度、恢复尺寸等后处理。
- `infer_ms`：`prep + forward + post` 总耗时。

因此业务 HUD/日志中的 `infer_ms` 或 pipeline `depth_ms` 不是纯 forward 时间，不能直接与官方 profile 表格等价比较。

## 8. 位姿回传与 Unity 应用口径

### 8.1 PoseMsg 字段

Python `PoseEncoder` 输出 MessagePack 字段：

- `timestamp_ms`
- `frame_id`
- `stage`
- `phase`
- `det_count`
- `depth_valid_ratio`
- `fps`
- `has_pose`
- `pose_matrix_flat`（4x4 行优先展平，16 个数）
- `yolo_ms/depth_ms/cutie_ms/pose_ms`

无有效位姿时：

- `has_pose=false`
- `pose_matrix_flat=null`
- 若 `send_when_no_pose=1`，Python 仍会发送状态包，Unity `PoseDecoder` 会忽略无 pose 包。

### 8.2 坐标转换

Python FoundationPose 输出是 OpenCV 相机坐标口径：

- x 向右。
- y 向下。
- z 向前。

Unity `PoseDecoder` 默认 `convertFromOpenCvCamera=true`，会转换为 Unity 常用口径：

- x 向右。
- y 向上。
- z 向前。

### 8.3 按发送帧对齐 Unity 世界位姿

当前 Unity 不再简单把 Python 返回 pose 直接设置到物体，而是：

1. `QuestStereoEncoder` 每次编码 stereo 成功后递增 `frame_id` 并触发 `OnFrameEncoded(frame_id)`。
2. `PoseFollow.HandleFrameEncoded(frame_id)` 缓存该发送帧时 `sourceTarget` 的世界位姿。
3. Python 回包带回同一个 `frame_id`。
4. `PoseFollow.FollowTarget(pose, frame_id)` 查找对应发送帧的 `sourceTargetPose`。
5. 将相机/参考系下的局部 pose 转换到 Unity 世界坐标。
6. `Update()` 每帧应用最新目标 pose，可选经过 `PoseSmoother` 平滑。

如果 Unity 日志出现“未命中发送帧缓存”：

- 检查 `QuestStereoEncoder.OnFrameEncoded` 是否绑定或被 `PoseFollow` 自动找到。
- 检查 `sourceTarget` 是否为空。
- 检查 `sourceTargetCacheSize` 是否太小。
- 检查 Python 回包 `frame_id` 是否被正确传递。

## 9. 调试与统计口径

### 9.1 Pipeline HUD

Quest/RealSense Pipeline HUD 显示：

- `fps`：实时 EMA FPS。
- `stage` 与 `phase`。
- `det`。
- `depth_valid`。
- `yolo/depth/cutie/pose` 分阶段耗时。

HUD 文本会根据窗口宽度换行，避免长文本溢出。

### 9.2 Pipeline stats 日志

Pipeline 统计日志包含：

- `rt_fps`：基于相邻输出帧间隔计算并 EMA 平滑。
- `window_fps`：按 `stats_interval` 统计窗口计算。
- `avg(yolo/depth/cutie/pose)`。
- `depth_valid`。

Quest 额外包含：

- `recv`
- `decode_fail`
- `sender_fps`
- `sender_est`
- `sender_raw`
- `sender_gap`

注意：`sender_raw` 是跨进程/跨设备单调时钟差，不能直接解释为真实网络延迟；优先看 `sender_est` 与趋势。

### 9.3 pose_server 延迟日志

`pose_server.py` 额外统计：

- `quest_rx->unity_tx`：从 Quest 帧接收时间戳到 Python 发出 pose 的估计总耗时。
- `run`：一次 pipeline.run 总耗时。
- `wait`：粗略等待/取帧耗时。
- `proc`：算法耗时合计。
- `send`：ZMQ 发送耗时。
- `pose_ratio`：输出有效 pose 的比例。
- `drop`：pose 发布失败比例。

## 10. 环境与依赖口径

### 10.1 Python 环境

项目使用 pixi 管理，核心文件：`Foundationpose_for_VR/pixi.toml`。

当前依赖特点：

- Python 3.12。
- CUDA 12.8。
- PyTorch 2.7.x cu128。
- TensorRT cu12。
- pyrealsense2。
- ultralytics/YOLOE。
- msgpack、onnx、pillow。
- Cutie 以本地 editable path 引入。

### 10.2 Windows 注意事项

若重建 `.pixi/envs/default` 失败：

- 先确认 VS Code Python LSP、Black Formatter、残留 Python 进程没有占用环境文件。
- 关闭相关进程后再重建。

### 10.3 FoundationPose C++ 扩展

`pixi run build` 中 `_build-fp` 会构建 `FoundationPose/mycpp`。

若 FoundationPose 导入报 C++ 扩展缺失，应先检查该构建是否成功。

## 11. 已清理/不要恢复的旧内容

不要恢复以下旧入口或旧协议路径：

- `src/pose_tracker_api.py`
- `src/vpt_cli.py`
- `src/VOT.py`
- `src/zmq_utils/timing.py`
- `src/zmq_utils/latency.py`
- `src/modules/quest_stereo.py`
- `src/modules/quest_receiver.py`
- `src/quest_stereo_pose_pipeline.py`
- Unity 旧 `StaticStereoEncoder.cs`
- ZMQ PUSH/PULL 模式
- Python `PayloadSender` 的 default topic
- 旧 packed_image_jpeg_legacy 单图协议
- Pose JSON 传输路径
- TRT legacy alias / legacy fallback 文件名
- `onnx.yaml` 运行时依赖

## 12. 最近核对后修正的重点

本次重新阅读代码后，确认并修正以下陈旧点：

1. Quest 接收模块当前是 `quest_io.py`，不是 `quest_stereo.py` 或 `quest_receiver.py`。
2. Quest 主服务入口应优先使用 `src/pose_server.py`。
3. `pixi.toml` 中 `demo-pipeline` 已修正为 `src/pipeline/quest_pipeline.py`。
4. Unity 已有 `PoseFollow + PoseSmoother` 的回包位姿应用链路，文档已补充该口径。
5. 多 topic drain bug 已在 Python 与 Unity 两侧通过“按 topic 缓存最新帧”修复。
6. Quest Pipeline 当前默认 FFS 权重路径在代码中是 `23-36-37/model_best_bp2_serialize.pth`；RealSense Pipeline 默认是 `20-30-48/model_best_bp2_serialize.pth`。
7. 项目代码中若干仍偏英文或过短的注释/文档字符串已补充为更详细中文说明。
8. 新增 Quest 快速启动策略：`camera_source=network` 默认通过 `preload_camera_cache=1` 读取本地 `camera_info_latest.json` 预初始化 K/FoundationPose，收到 `quest_stereo` 后即可开始估计；网络 `quest_camera_info` 后续用于校验与刷新。
9. 新增 `network_calib_update` 参数：默认收到不同网络标定后刷新 K/PoseEstimator 并重置跟踪；可设为 0 禁用自动刷新。

## 13. 后续 AI 接手建议

建议按以下顺序接手：

1. 先确认 `Calibration/cache/camera_info_latest.json` 已有一次有效缓存；有缓存时 `pose_server.py` 默认会先初始化 FoundationPose，后续收到 stereo 即可估计。
2. 再跑 `pose_server.py`，确认 Unity → Python 的 `quest_stereo` 能收到；`quest_camera_info` 仍应持续发送，用于校验/刷新缓存。
3. 若想严格验证网络 camera_info，可加 `--preload_camera_cache 0` 强制等待本次会话的网络标定。
4. 若不希望运行中因网络标定变化重建 PoseEstimator，可加 `--network_calib_update 0`。
5. 用 stage 1/2/3/4 逐段定位，不要直接在 stage 4 盲调。
6. stage 2 优先看 mask 是否稳定、目标 prompt 是否正确。
7. stage 3 优先看 depth_valid 和深度范围，不要先调 FoundationPose。
8. stage 4 再看 register 是否成功、track 是否稳定。
9. 若 Unity 中物体位置明显错位，优先检查：
   - `PoseDecoder.convertFromOpenCvCamera`
   - `PoseFollow.sourceTarget`
   - `frame_id` 缓存命中
   - Quest K 映射方式
10. 若 stereo 收不到但 camera_info 能收到，优先检查：
    - Unity `PayloadSender` 是否有 `quest_stereo` Entry
    - `QuestStereoEncoder` 左右相机是否 `IsPlaying`
    - `recv_hwm` 是否太小
11. 若 camera_info 收不到，优先检查：
    - Unity `quest_camera_info` topic 是否配置
    - `QuestCameraInfoEncoder` 左右相机引用是否有效
12. 若 TRT 不生效，先确认 engine 文件名是否带完整 tag、平台和精度标签。
13. 若协议字段变更，必须同步修改：
    - Python message/encoder/decoder
    - Unity Msg/Encoder/Decoder
    - 本文档第 5、6、8 节

## 14. 文档维护规则

- 只维护本文件作为长期交接入口。
- 每次完成较大改动后，至少更新：
  - 第 2 节：入口是否变化。
  - 第 4 节：模块职责是否变化。
  - 第 5/6/8/9 节：协议、标定、位姿和统计口径是否变化。
  - 第 12 节：最近修正重点。
  - 第 13 节：下一步接手建议。
