"""Quest 位姿服务：接收 Unity 发来的双目图像与相机信息，并回传位姿结果。"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# 允许直接脚本运行：python src/pose_server.py
if __package__ is None or __package__ == "":
    SRC_DIR = Path(__file__).resolve().parent
    if str(SRC_DIR) not in sys.path:
        sys.path.append(str(SRC_DIR))

from pipeline.quest_pipeline import (  # noqa: E402
    build_arg_parser as build_pipeline_arg_parser,
    build_quest_pipeline,
)
from zmq_utils import PayloadSender, PoseEncoder  # noqa: E402
from zmq_utils.payload.message.quest_camera_info_msg import QuestCameraInfoMsg  # noqa: E402


CAMERA_INFO_VOLATILE_KEYS = frozenset({"_received_at", "sender_mono_ms"})


# =========================
# Camera Info 缓存管理
# =========================


def _camera_info_to_dict(msg: QuestCameraInfoMsg) -> dict:
    """将 QuestCameraInfoMsg 转为可序列化的 dict（用于 JSON 持久化）。"""
    import msgpack as _msgpack

    # 利用 serialize -> unpackb 获得 flat dict。
    raw = _msgpack.unpackb(msg.serialize(), raw=False, strict_map_key=False)
    return dict(raw)


def _camera_info_core_dict(info: dict) -> dict:
    """Return camera_info fields that describe calibration, not send/receive time."""
    return {k: v for k, v in info.items() if k not in CAMERA_INFO_VOLATILE_KEYS}


def _save_camera_info(
    msg: QuestCameraInfoMsg,
    cache_dir: Path,
) -> None:
    """将 camera_info 保存为本地 JSON（带时间戳）。

    策略：
    - 若最新版与当前版不同 -> 备份旧版，保存新版。
    - 若相同 -> 仅更新接收时间戳。
    - 每次接收都保存最新版到 camera_info_latest.json。
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    latest_path = cache_dir / "camera_info_latest.json"
    current_dict = _camera_info_to_dict(msg)
    current_dict["_received_at"] = datetime.now().isoformat()

    # 比较与最新版是否相同。
    if latest_path.is_file():
        try:
            with latest_path.open("r", encoding="utf-8") as f:
                existing = json.load(f)

            # 移除元数据后比较核心字段。
            existing_core = _camera_info_core_dict(existing)
            current_core = _camera_info_core_dict(current_dict)

            if existing_core != current_core:
                # 内容不同 -> 备份旧版。
                ts = existing.get("_received_at", "unknown")
                safe_ts = ts.replace(":", "-").replace(".", "-")
                backup_path = cache_dir / f"camera_info_{safe_ts}.json"
                shutil.copy2(str(latest_path), str(backup_path))
                logging.info("[camera_info] 内容变化，旧版已备份: %s", backup_path.name)
        except Exception as exc:
            logging.warning("[camera_info] 读取/比较旧版失败: %s", exc)

    # 保存最新版。
    with latest_path.open("w", encoding="utf-8") as f:
        json.dump(current_dict, f, indent=2, ensure_ascii=False)


# =========================
# 参数与工具函数
# =========================


def build_arg_parser() -> argparse.ArgumentParser:
    """在 Quest Pipeline 参数基础上，扩展 pose_server 的命令行参数。"""
    parser = build_pipeline_arg_parser()

    parser.description = "Quest 位姿服务（接收双目+相机信息，向 Unity 发布位姿）"
    parser.add_argument(
        "--run_stage",
        type=int,
        default=4,
        help="Pipeline 执行阶段（1~4）。默认 4。",
    )
    parser.add_argument(
        "--pose_pub_host",
        type=str,
        default="*",
        help="位姿发布端绑定地址。",
    )
    parser.add_argument(
        "--pose_pub_port",
        type=int,
        default=5556,
        help="位姿发布端口（ZMQ PUB）。",
    )
    parser.add_argument(
        "--pose_topic",
        type=str,
        default="pose",
        help="位姿消息的 topic 名称。",
    )
    parser.add_argument(
        "--pose_pub_hwm",
        type=int,
        default=1,
        help="发布端高水位。",
    )
    parser.add_argument(
        "--send_when_no_pose",
        type=int,
        default=1,
        help="无位姿时是否仍发送状态包（1=发送，0=跳过）。",
    )
    parser.add_argument(
        "--pub_log_interval",
        type=int,
        default=60,
        help="每处理多少帧打印一次发布统计日志。",
    )
    parser.add_argument(
        "--local_debug",
        type=int,
        default=1,
        help="是否开启本地 OpenCV 调试窗口。",
    )
    parser.add_argument(
        "--show_depth_window",
        type=int,
        default=1,
        help="local_debug=1 时是否显示深度窗口。",
    )
    parser.add_argument(
        "--show_stereo_window",
        type=int,
        default=1,
        help="local_debug=1 时是否显示双目窗口。",
    )
    parser.add_argument(
        "--latency_ema_alpha",
        type=float,
        default=0.15,
        help="延迟平滑系数 EMA alpha。",
    )
    parser.add_argument(
        "--enable_keyboard_control",
        type=int,
        default=1,
        help="是否启用键盘控制。",
    )
    parser.add_argument(
        "--reset_interval_sec",
        type=float,
        default=0.0,
        help="自动重置跟踪的周期（秒）。<=0 关闭。",
    )

    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析 pose_server 命令行参数。"""
    return build_arg_parser().parse_args(argv)


def _ema(prev: float, value: float, alpha: float) -> float:
    """计算指数滑动平均。"""
    if prev <= 0.0:
        return value
    return prev * (1.0 - alpha) + value * alpha


def _draw_text_block(
    image: np.ndarray,
    lines: list[str],
    x: int = 10,
    y: int = 24,
    gap: int = 22,
    anchor: str = "top-left",
    panel_alpha: float = 0.55,
) -> None:
    """在调试图上绘制带半透明底板的文本块。"""
    if not lines:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    padding = 8

    sizes = [cv2.getTextSize(line, font, scale, thickness)[0] for line in lines]
    text_w = max((w for (w, _) in sizes), default=0)
    text_h = max((h for (_, h) in sizes), default=18)

    if anchor == "bottom-left":
        y = image.shape[0] - padding - (len(lines) - 1) * gap

    top = max(y - text_h - padding, 0)
    bottom = min(y + (len(lines) - 1) * gap + padding, image.shape[0] - 1)
    left = max(x - padding, 0)
    right = min(x + text_w + padding, image.shape[1] - 1)

    overlay = image.copy()
    cv2.rectangle(overlay, (left, top), (right, bottom), (0, 0, 0), -1)
    cv2.addWeighted(overlay, panel_alpha, image, 1.0 - panel_alpha, 0, image)

    for i, line in enumerate(lines):
        yy = y + i * gap
        cv2.putText(
            image, line, (x, yy), font, scale, (15, 15, 15), 2, cv2.LINE_AA
        )
        cv2.putText(
            image, line, (x, yy), font, scale, (245, 245, 245), 1, cv2.LINE_AA
        )


# =========================
# 主循环
# =========================


def run_pose_server(args: argparse.Namespace) -> None:
    """运行端到端位姿服务主循环。"""
    endpoint = f"tcp://{args.pose_pub_host}:{int(args.pose_pub_port)}"
    topic = str(args.pose_topic)
    log_interval = max(int(args.pub_log_interval), 1)
    send_when_no_pose = bool(int(args.send_when_no_pose))
    local_debug = bool(int(args.local_debug))
    show_depth_window = bool(int(args.show_depth_window))
    show_stereo_window = bool(int(args.show_stereo_window))
    enable_keyboard_control = bool(int(args.enable_keyboard_control))
    latency_alpha = float(np.clip(args.latency_ema_alpha, 0.01, 1.0))
    reset_interval_sec = max(float(args.reset_interval_sec), 0.0)
    camera_cache_dir = Path(args.camera_cache_dir)

    # 构建并配置 Quest 位姿 Pipeline。
    pipeline = build_quest_pipeline(args)
    pipeline.set_stage(int(args.run_stage))

    # 创建位姿发布端。
    sender = PayloadSender(
        endpoint=endpoint,
        hwm=max(int(args.pose_pub_hwm), 1),
        bind=True,
    )
    encoder = PoseEncoder()

    frame_count = 0
    sent_count = 0
    dropped_count = 0
    pose_count = 0
    reset_count = 0
    start_t = time.perf_counter()
    last_reset_t = start_t

    # camera_info 监控。
    last_saved_camera_info_version = 0

    # 延迟统计（毫秒）。
    run_ms_ema = 0.0
    proc_ms_ema = 0.0
    wait_ms_ema = 0.0
    send_ms_ema = 0.0
    e2e_ms_ema = 0.0

    if local_debug:
        cv2.namedWindow("PoseServer Debug", cv2.WINDOW_AUTOSIZE)
        if show_depth_window:
            cv2.namedWindow("PoseServer Depth", cv2.WINDOW_AUTOSIZE)
        if show_stereo_window:
            cv2.namedWindow("PoseServer Stereo", cv2.WINDOW_AUTOSIZE)

    # 窗口占位图：等待首帧期间避免 OpenCV 窗口假死。
    # 若已用本地 camera_info 缓存完成预初始化，则只需要等待 stereo 图像即可开始估计。
    calib_ready_at_start = bool(getattr(pipeline, "_calib_initialized", False))
    waiting_text = (
        "Waiting for Quest stereo..."
        if calib_ready_at_start
        else "Waiting for Quest camera_info & stereo..."
    )
    waiting_placeholder = np.zeros((240, 640, 3), dtype=np.uint8)
    cv2.putText(
        waiting_placeholder,
        waiting_text,
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    pipeline.start()
    logging.info(
        "[pose_server] started recv=tcp://%s:%d pub=%s topic=%s stage=%d camera_source=%s calib_ready=%s preload_cache=%s",
        args.listen_host,
        int(args.listen_port),
        endpoint,
        topic,
        int(args.run_stage),
        args.camera_source,
        "YES" if calib_ready_at_start else "NO",
        int(getattr(args, "preload_camera_cache", 1)),
    )

    try:
        last_wait_log_t = time.perf_counter()
        while True:
            loop_t0 = time.perf_counter()
            output = pipeline.run(return_debug=local_debug)
            if output is None:
                # 关键：无数据时也要刷窗口事件，否则 OpenCV 窗口会被系统判为“未响应”。
                if local_debug:
                    cv2.imshow("PoseServer Debug", waiting_placeholder)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break
                # 每 3 秒打印一次等待诊断，方便定位是缺标定还是缺 stereo。
                now_wait = time.perf_counter()
                if now_wait - last_wait_log_t >= 3.0:
                    cam_info = pipeline.camera.get_camera_info()
                    stereo_peek = pipeline.camera._latest_stereo  # 仅诊断用
                    recv_stats = pipeline.camera.get_stats()
                    logging.info(
                        "[pose_server] waiting... calib_ready=%s camera_info=%s stereo=%s recv=%s decoded=%s",
                        "YES" if getattr(pipeline, "_calib_initialized", False) else "NO",
                        "OK" if cam_info is not None else "None",
                        "OK" if stereo_peek is not None else "None",
                        recv_stats.get("received", 0),
                        recv_stats.get("decoded", 0),
                    )
                    last_wait_log_t = now_wait
                continue
            after_run_t = time.perf_counter()

            # 检查是否有新的 camera_info 并保存。
            camera_info = pipeline.camera.get_camera_info()
            if camera_info is not None:
                info_version = pipeline.camera.get_camera_info_version()
                if info_version != last_saved_camera_info_version:
                    _save_camera_info(camera_info, camera_cache_dir)
                    last_saved_camera_info_version = info_version

            # 可选自动重置。
            now_t = time.perf_counter()
            if (
                reset_interval_sec > 1e-6
                and (now_t - last_reset_t) >= reset_interval_sec
            ):
                pipeline.reset_tracking_state()
                reset_count += 1
                last_reset_t = now_t
                logging.info(
                    "[pose_server] auto reset tracking (interval=%.2fs)",
                    reset_interval_sec,
                )

            frame_count += 1
            if output.pose_4x4 is not None:
                pose_count += 1

            if output.pose_4x4 is None and not send_when_no_pose:
                continue

            payload = encoder.encode(
                timestamp_ms=output.timestamp_ms,
                frame_id=int(output.frame_id or 0),
                stage=output.stage,
                phase=output.phase,
                det_count=output.det_count,
                depth_valid_ratio=output.depth_valid_ratio,
                fps=output.fps,
                timing_ms={
                    "yolo": output.timing.yolo_ms,
                    "depth": output.timing.depth_ms,
                    "cutie": output.timing.cutie_ms,
                    "pose": output.timing.pose_ms,
                },
                pose_4x4=output.pose_4x4,
            )
            if payload is None:
                dropped_count += 1
                continue

            send_t0 = time.perf_counter()
            sent = sender.send_payload(payload, topic=topic)
            send_ms = (time.perf_counter() - send_t0) * 1000.0

            # 统计分段耗时。
            run_ms = (after_run_t - loop_t0) * 1000.0
            proc_ms = (
                float(output.timing.yolo_ms)
                + float(output.timing.depth_ms)
                + float(output.timing.cutie_ms)
                + float(output.timing.pose_ms)
            )
            wait_ms = max(run_ms - proc_ms, 0.0)
            total_ms = max(
                time.perf_counter() * 1000.0 - float(output.timestamp_ms), 0.0
            )

            run_ms_ema = _ema(run_ms_ema, run_ms, latency_alpha)
            proc_ms_ema = _ema(proc_ms_ema, proc_ms, latency_alpha)
            wait_ms_ema = _ema(wait_ms_ema, wait_ms, latency_alpha)
            send_ms_ema = _ema(send_ms_ema, send_ms, latency_alpha)
            e2e_ms_ema = _ema(e2e_ms_ema, total_ms, latency_alpha)

            if sent:
                sent_count += 1
            else:
                dropped_count += 1

            # 本地调试显示与键盘交互。
            if local_debug and output.debug is not None:
                debug_vis = output.debug.vis_bgr.copy()
                _draw_text_block(
                    debug_vis,
                    [
                        f"total(ms) quest_rx->unity_tx={e2e_ms_ema:.1f}",
                        f"split(ms) run={run_ms_ema:.1f} wait={wait_ms_ema:.1f} proc={proc_ms_ema:.1f} send={send_ms_ema:.2f}",
                        f"queue topic={topic} sent={sent_count} drop={dropped_count} reset={reset_count}",
                    ],
                    anchor="bottom-left",
                )
                cv2.imshow("PoseServer Debug", debug_vis)

                if show_depth_window:
                    cv2.imshow("PoseServer Depth", output.debug.depth_vis_bgr)
                if show_stereo_window:
                    cv2.imshow("PoseServer Stereo", output.debug.stereo_vis_bgr)

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if enable_keyboard_control and key == ord("r"):
                    pipeline.reset_tracking_state()
                    reset_count += 1
                    last_reset_t = time.perf_counter()
                    logging.info("[pose_server] manual reset tracking")
                if enable_keyboard_control and key in (
                    ord("1"),
                    ord("2"),
                    ord("3"),
                    ord("4"),
                ):
                    pipeline.set_stage(int(chr(key)))
                    reset_count += 1
                    last_reset_t = time.perf_counter()
                    logging.info("[pose_server] switch stage -> %d", pipeline.stage)

            if frame_count % log_interval == 0:
                elapsed = max(time.perf_counter() - start_t, 1e-6)
                pub_fps = sent_count / elapsed
                pose_ratio = pose_count / max(frame_count, 1)
                drop_ratio = dropped_count / max(sent_count + dropped_count, 1)
                logging.info(
                    "[pose_server] frames=%d sent=%d dropped=%d pub_fps=%.1f pose_ratio=%.1f%% drop=%.1f%% phase=%s "
                    "lat(ms):quest_rx->unity_tx=%.1f run=%.1f wait=%.1f proc=%.1f send=%.2f reset=%d",
                    frame_count,
                    sent_count,
                    dropped_count,
                    pub_fps,
                    pose_ratio * 100.0,
                    drop_ratio * 100.0,
                    output.phase,
                    e2e_ms_ema,
                    run_ms_ema,
                    wait_ms_ema,
                    proc_ms_ema,
                    send_ms_ema,
                    reset_count,
                )
    except KeyboardInterrupt:
        logging.info("\n[pose_server] interrupted by user")
    finally:
        pipeline.stop()
        sender.close()
        if local_debug:
            cv2.destroyAllWindows()


def main(argv: list[str] | None = None) -> None:
    """脚本入口。"""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_pose_server(args)


if __name__ == "__main__":
    main()
