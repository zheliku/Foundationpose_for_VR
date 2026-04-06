# Foundationpose_for_VR 项目总交接文档（唯一入口）

更新时间：2026-04-06

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

## 8. 今日工作总结（2026-04-06）

### 8.1 代码清理与架构收敛

- 清理了历史旧入口与过期测试依赖，主入口收敛到 pipeline 目录。
- 统一改为“模块 API + pipeline 编排”模式，减少分散脚本。

### 8.2 API 风格统一

- 移除了 Config 类中间层，改为直接参数调用。
- 统一了 modules 与 pipeline 的类成员声明方式：
  - 类体显式成员清单
  - 成员分组
  - 中文注释
- **init** 文档补齐并精简，默认值尽量下沉到类体。

### 8.3 初始化与执行策略修正

- start() 职责收敛为启动与状态重置，不再做重初始化杂项。
- Quest 链路保留在 **init** 完成 K 与 PoseEstimator 初始化。
- RealSense 链路保留 run() 首帧懒初始化路径。

### 8.4 调试显示修复

- 修复了 Quest/RealSense 窗口 HUD 文本溢出问题。
- 将 fps 从累计均值改为实时帧率显示。
- 合并 HUD 绘制方法，减少重复逻辑并统一样式。

## 9. 后续 AI 接手建议

建议按以下顺序开始：

1. 先确认链路可运行，再做调参。
2. 用 stage 2/3/4 逐段定位问题，不要一上来全链路盲调。
3. 先用固定基线场景验证：
   - 掩码质量
   - 深度有效率
   - 位姿稳定性
4. 任何协议或显示字段变更，都同步更新本文件第 5 节。

## 10. 文档维护规则

- 仅维护本文件，避免再次产生多份历史交接文档。
- 每次完成较大改动后，只需更新：
  - 第 5 节（统计/调试口径）
  - 第 8 节（当日总结）
  - 第 9 节（下一步建议）
