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
