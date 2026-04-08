"""Quest 位姿服务：接收 Unity 发来的双目图像，并回传位姿结果。"""

from __future__ import annotations

import argparse
import logging
import sys
import time
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


def build_arg_parser() -> argparse.ArgumentParser:
    """在 Quest Pipeline 参数基础上，扩展 pose_server 的命令行参数。"""
    parser = build_pipeline_arg_parser()

    parser.description = "Quest 位姿服务（接收双目，向 Unity 发布位姿）"
    parser.add_argument(
        "--run_stage",
        type=int,
        default=4,
        help="服务启动后设置 Pipeline 执行阶段（1~4）。默认 4，表示完整流程：相机+检测+深度+位姿。",
    )
    parser.add_argument(
        "--pose_pub_host",
        type=str,
        default="*",
        help="位姿发布端绑定地址。'*' 表示监听所有网卡，127.0.0.1 表示仅本机可访问。",
    )
    parser.add_argument(
        "--pose_pub_port",
        type=int,
        default=5556,
        help="位姿发布端口（ZMQ PUB）。Unity 端 Receiver 需连接到相同端口。",
    )
    parser.add_argument(
        "--pose_topic",
        type=str,
        default="payload",
        help="发布消息的 topic 名称。Unity 端若启用 topic 过滤，需与此值保持一致。",
    )
    parser.add_argument(
        "--pose_pub_hwm",
        type=int,
        default=1,
        help="发布端高水位（High Water Mark）。值越小越实时，值越大越不易丢包但会增加排队延迟。",
    )
    parser.add_argument(
        "--send_when_no_pose",
        type=int,
        default=1,
        help="当当前帧无位姿时是否仍发送状态包（1=发送，0=跳过）。",
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
        help="是否开启本地 OpenCV 调试窗口（1=开启，0=关闭）。",
    )
    parser.add_argument(
        "--show_depth_window",
        type=int,
        default=1,
        help="在 local_debug=1 时，是否显示深度可视化窗口。",
    )
    parser.add_argument(
        "--show_stereo_window",
        type=int,
        default=1,
        help="在 local_debug=1 时，是否显示双目拼接窗口。",
    )
    parser.add_argument(
        "--latency_ema_alpha",
        type=float,
        default=0.15,
        help="延迟平滑系数 EMA alpha（0~1）。越大越灵敏，越小越平滑。",
    )
    parser.add_argument(
        "--enable_keyboard_control",
        type=int,
        default=1,
        help="是否启用键盘控制（1=启用，支持 q/r/1~4；0=禁用）。",
    )
    parser.add_argument(
        "--reset_interval_sec",
        type=float,
        default=0.0,
        help="自动重置跟踪的周期（秒）。<=0 表示关闭自动重置。",
    )

    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析 pose_server 命令行参数。"""
    return build_arg_parser().parse_args(argv)


def _ema(prev: float, value: float, alpha: float) -> float:
    """计算指数滑动平均，用于平滑抖动较大的实时指标。"""
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
    """在调试图上绘制带半透明底板的文本块，提升可读性。"""
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
            image,
            line,
            (x, yy),
            font,
            scale,
            (15, 15, 15),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            line,
            (x, yy),
            font,
            scale,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )


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

    # 构建并配置 Quest 位姿 Pipeline。
    pipeline = build_quest_pipeline(args)
    pipeline.set_stage(int(args.run_stage))

    # 创建发布端：将位姿结果通过 ZMQ PUB 广播给 Unity。
    sender = PayloadSender(
        endpoint=endpoint,
        hwm=max(int(args.pose_pub_hwm), 1),
        bind=True,
        send_topic=True,
        default_topic=topic,
    )
    encoder = PoseEncoder()

    frame_count = 0
    sent_count = 0
    dropped_count = 0
    pose_count = 0
    reset_count = 0
    start_t = time.perf_counter()
    last_reset_t = start_t

    # 延迟统计（毫秒）：run=单帧总耗时，proc=算法耗时和，wait=run-proc，send=发送耗时，e2e=接收->发送完成。
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

    # 先启动 Pipeline 再进入循环，确保网络接收和模型状态就绪。
    pipeline.start()
    logging.info(
        "[pose_server] started recv=tcp://%s:%d pub=%s topic=%s stage=%d",
        args.listen_host,
        int(args.listen_port),
        endpoint,
        topic,
        int(args.run_stage),
    )

    try:
        while True:
            # 1) 跑一帧完整 Pipeline，拿到结构化输出（含位姿和调试信息）。
            loop_t0 = time.perf_counter()
            output = pipeline.run(return_debug=local_debug)
            if output is None:
                continue
            after_run_t = time.perf_counter()

            # 2) 可选自动重置：长时间运行时周期性清空跟踪状态，避免状态漂移。
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

            # 配置为“无位姿不发包”时，直接跳过本帧发送。
            if output.pose_4x4 is None and not send_when_no_pose:
                continue

            # 3) 编码协议负载：统一字段结构，便于 Unity 端稳定解码。
            packet_parts = encoder.encode(
                timestamp_ms=output.timestamp_ms,
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
            send_t0 = time.perf_counter()
            sent = sender.send_multipart(packet_parts)
            send_ms = (time.perf_counter() - send_t0) * 1000.0

            # 4) 统计分段耗时：
            # - run: pipeline.run 总耗时
            # - proc: 算法阶段耗时和（yolo/depth/cutie/pose）
            # - wait: run - proc，近似表示取帧等待/队列等待等非算法开销
            # - send: ZMQ 发送耗时
            run_ms = (after_run_t - loop_t0) * 1000.0
            proc_ms = (
                float(output.timing.yolo_ms)
                + float(output.timing.depth_ms)
                + float(output.timing.cutie_ms)
                + float(output.timing.pose_ms)
            )
            wait_ms = max(run_ms - proc_ms, 0.0)
            # 当前可准确测得的总延迟：Python 收到 Quest 帧到 Python 发送完成（不含 Unity 端接收/渲染）。
            total_ms = max(
                time.perf_counter() * 1000.0 - float(output.timestamp_ms), 0.0
            )

            # 5) 用 EMA 平滑实时指标，避免单帧抖动影响观测。
            run_ms_ema = _ema(run_ms_ema, run_ms, latency_alpha)
            proc_ms_ema = _ema(proc_ms_ema, proc_ms, latency_alpha)
            wait_ms_ema = _ema(wait_ms_ema, wait_ms, latency_alpha)
            send_ms_ema = _ema(send_ms_ema, send_ms, latency_alpha)
            e2e_ms_ema = _ema(e2e_ms_ema, total_ms, latency_alpha)

            if sent:
                sent_count += 1
            else:
                dropped_count += 1

            # 6) 本地调试显示与键盘交互（q 退出，r 重置，1~4 切阶段）。
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

                    # 7) 周期日志：便于离线排查吞吐、丢包与延迟趋势。
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
