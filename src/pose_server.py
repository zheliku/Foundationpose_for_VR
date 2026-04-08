"""Quest pose server: receive stereo from Unity and publish pose results back."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Allow running directly: python src/pose_server.py
if __package__ is None or __package__ == "":
    SRC_DIR = Path(__file__).resolve().parent
    if str(SRC_DIR) not in sys.path:
        sys.path.append(str(SRC_DIR))

from pipeline.quest_pipeline import (  # noqa: E402
    build_arg_parser as build_pipeline_arg_parser,
    build_quest_pipeline,
)
from zmq_utils import PayloadSender, PoseServerEncoder  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the pose server CLI parser on top of quest pipeline args."""
    parser = build_pipeline_arg_parser()

    parser.description = "Quest pose server (receive stereo, publish pose to Unity)"
    parser.add_argument("--run_stage", type=int, default=4)
    parser.add_argument("--pose_pub_host", type=str, default="*")
    parser.add_argument("--pose_pub_port", type=int, default=5556)
    parser.add_argument("--pose_topic", type=str, default="payload")
    parser.add_argument("--pose_pub_hwm", type=int, default=1)
    parser.add_argument("--send_when_no_pose", type=int, default=1)
    parser.add_argument("--pub_log_interval", type=int, default=60)

    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse pose server arguments."""
    return build_arg_parser().parse_args(argv)


def run_pose_server(args: argparse.Namespace) -> None:
    """Run the end-to-end pose server loop."""
    endpoint = f"tcp://{args.pose_pub_host}:{int(args.pose_pub_port)}"
    topic = str(args.pose_topic)
    log_interval = max(int(args.pub_log_interval), 1)
    send_when_no_pose = bool(int(args.send_when_no_pose))

    pipeline = build_quest_pipeline(args)
    pipeline.set_stage(int(args.run_stage))

    sender = PayloadSender(
        endpoint=endpoint,
        hwm=max(int(args.pose_pub_hwm), 1),
        bind=True,
        send_topic=True,
        default_topic=topic,
    )
    encoder = PoseServerEncoder()

    frame_count = 0
    sent_count = 0
    dropped_count = 0
    pose_count = 0
    start_t = time.perf_counter()

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
            output = pipeline.run(return_debug=False)
            if output is None:
                continue

            frame_count += 1
            if output.pose_4x4 is not None:
                pose_count += 1

            if output.pose_4x4 is None and not send_when_no_pose:
                continue

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
            sent = sender.send_multipart(packet_parts)
            if sent:
                sent_count += 1
            else:
                dropped_count += 1

            if frame_count % log_interval == 0:
                elapsed = max(time.perf_counter() - start_t, 1e-6)
                pub_fps = sent_count / elapsed
                pose_ratio = pose_count / max(frame_count, 1)
                drop_ratio = dropped_count / max(sent_count + dropped_count, 1)
                logging.info(
                    "[pose_server] frames=%d sent=%d dropped=%d pub_fps=%.1f pose_ratio=%.1f%% drop=%.1f%% phase=%s",
                    frame_count,
                    sent_count,
                    dropped_count,
                    pub_fps,
                    pose_ratio * 100.0,
                    drop_ratio * 100.0,
                    output.phase,
                )
    except KeyboardInterrupt:
        logging.info("\n[pose_server] interrupted by user")
    finally:
        pipeline.stop()
        sender.close()


def main(argv: list[str] | None = None) -> None:
    """Entrypoint."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_pose_server(args)


if __name__ == "__main__":
    main()
