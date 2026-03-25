import argparse
import logging
import os
import sys
import time

import cv2
import numpy as np
import torch

try:
    import pyrealsense2 as rs
except ImportError as exc:
    raise SystemExit(
        "未找到 pyrealsense2，请先安装 RealSense Python SDK 后再运行。"
    ) from exc

code_dir = os.path.dirname(os.path.realpath(__file__))
project_dir = os.path.dirname(code_dir)
sys.path.append(project_dir)

from Utils import AMP_DTYPE, set_logging_format, set_seed
from core.utils.utils import InputPadder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        default=f"{project_dir}/weights/20-30-48/model_best_bp2_serialize.pth",
        type=str,
    )
    parser.add_argument("--width", default=640, type=int)
    parser.add_argument("--height", default=480, type=int)
    parser.add_argument("--fps", default=60, type=int)
    parser.add_argument("--scale", default=1.0, type=float)
    parser.add_argument("--valid_iters", default=4, type=int)
    parser.add_argument("--max_disp", default=192, type=int)
    parser.add_argument("--min_depth", default=0.1, type=float)
    parser.add_argument("--max_depth", default=10.0, type=float)
    parser.add_argument("--vis_max_depth", default=10.0, type=float)
    parser.add_argument("--error_max", default=1.0, type=float)
    parser.add_argument("--sample_points", default=6, type=int)
    parser.add_argument(
        "--optimize_build_volume", default="triton", choices=["pytorch1", "triton"]
    )
    parser.add_argument("--seed", default=-1, type=int)
    parser.add_argument("--cudnn_benchmark", default=1, type=int, choices=[0, 1])
    parser.add_argument(
        "--window_name", default="FoundationStereo vs RealSense", type=str
    )
    parser.add_argument("--device", default="cuda", type=str)
    return parser.parse_args()


def colorize_depth(depth, min_depth, max_depth, invalid_mask=None):
    denom = max(max_depth - min_depth, 1e-6)
    norm = ((depth - min_depth) / denom).clip(0.0, 1.0)
    vis = cv2.applyColorMap((norm * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
    if invalid_mask is not None and invalid_mask.any():
        vis[invalid_mask] = 0
    return vis


def colorize_error(error_map, error_max, invalid_mask=None):
    norm = (error_map / max(error_max, 1e-6)).clip(0.0, 1.0)
    gray = (norm * 255.0).astype(np.uint8)
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if invalid_mask is not None and invalid_mask.any():
        vis[invalid_mask] = 0
    return vis


def draw_label(img, text, x, y):
    cv2.putText(
        img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 10, 10), 3, cv2.LINE_AA
    )
    cv2.putText(
        img,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )


def sample_points_from_valid_mask(valid_mask, sample_points):
    if sample_points <= 0:
        return []
    ys, xs = np.where(valid_mask)
    if len(xs) == 0:
        return []

    num = min(sample_points, len(xs))
    pick_ids = np.linspace(0, len(xs) - 1, num=num, dtype=int)
    picked = []
    used = set()
    for idx in pick_ids:
        p = (int(xs[idx]), int(ys[idx]))
        if p not in used:
            picked.append(p)
            used.add(p)
    return picked


def draw_sample_markers(left_vis, pred_vis, native_vis, err_vis, points, diff_map):
    for i, (x, y) in enumerate(points):
        color = (0, 255, 255)
        cv2.circle(left_vis, (x, y), 4, color, -1, cv2.LINE_AA)
        cv2.circle(pred_vis, (x, y), 4, color, -1, cv2.LINE_AA)
        cv2.circle(native_vis, (x, y), 4, color, -1, cv2.LINE_AA)
        cv2.circle(err_vis, (x, y), 4, color, -1, cv2.LINE_AA)
        label = f"{i+1}:{abs(float(diff_map[y, x])):.2f}m"
        text_x = min(x + 6, err_vis.shape[1] - 140)
        text_y = min(y + 16, err_vis.shape[0] - 8)
        draw_label(err_vis, label, text_x, text_y)


def setup_realsense(width, height, fps):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.infrared, 1, width, height, rs.format.y8, fps)
    config.enable_stream(rs.stream.infrared, 2, width, height, rs.format.y8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    profile = pipeline.start(config)

    left_profile = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
    right_profile = profile.get_stream(rs.stream.infrared, 2).as_video_stream_profile()
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = float(depth_sensor.get_depth_scale())

    intr = left_profile.get_intrinsics()
    fx = float(intr.fx)
    extr = left_profile.get_extrinsics_to(right_profile)
    baseline = abs(float(extr.translation[0]))
    if baseline <= 0:
        raise RuntimeError("无法从 RealSense 获取有效基线（baseline）。")

    return pipeline, fx, baseline, depth_scale


def load_model(model_dir, valid_iters, max_disp, device):
    model = torch.load(model_dir, map_location="cpu", weights_only=False)
    model.args.valid_iters = valid_iters
    model.args.max_disp = max_disp
    model = model.to(device).eval()
    return model


def predict_depth(model, left_gray, right_gray, fx, baseline, args, device):
    t0 = time.perf_counter()
    left_rgb = np.repeat(left_gray[..., None], 3, axis=2)
    right_rgb = np.repeat(right_gray[..., None], 3, axis=2)
    if args.scale != 1.0:
        left_rgb = cv2.resize(
            left_rgb,
            dsize=None,
            fx=args.scale,
            fy=args.scale,
            interpolation=cv2.INTER_LINEAR,
        )
        right_rgb = cv2.resize(
            right_rgb,
            dsize=(left_rgb.shape[1], left_rgb.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    left_tensor = torch.as_tensor(left_rgb).to(device).float()[None].permute(0, 3, 1, 2)
    right_tensor = (
        torch.as_tensor(right_rgb).to(device).float()[None].permute(0, 3, 1, 2)
    )
    padder = InputPadder(left_tensor.shape, divis_by=32, force_square=False)
    left_tensor, right_tensor = padder.pad(left_tensor, right_tensor)
    t1 = time.perf_counter()

    with torch.inference_mode():
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=AMP_DTYPE):
            disp = model.forward(
                left_tensor,
                right_tensor,
                iters=args.valid_iters,
                test_mode=True,
                optimize_build_volume=args.optimize_build_volume,
            )
    t2 = time.perf_counter()
    disp = padder.unpad(disp.float()).squeeze(0).squeeze(0).detach().cpu().numpy()
    disp = np.clip(disp, 1e-6, None)

    fx_scaled = fx * args.scale
    pred_depth = (fx_scaled * baseline) / disp
    if args.scale != 1.0:
        pred_depth = cv2.resize(
            pred_depth,
            dsize=(left_gray.shape[1], left_gray.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    pred_depth[~np.isfinite(pred_depth)] = 0
    t3 = time.perf_counter()

    timing = {
        "prep_ms": (t1 - t0) * 1000.0,
        "forward_ms": (t2 - t1) * 1000.0,
        "post_ms": (t3 - t2) * 1000.0,
    }
    return pred_depth, timing


def main():
    args = parse_args()
    set_logging_format(level=logging.INFO)
    torch.autograd.set_grad_enabled(False)

    if args.device == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA 不可用，自动切换到 CPU。")
        args.device = "cpu"
    device = torch.device(args.device)

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    if args.seed >= 0:
        set_seed(args.seed)
        logging.info(f"使用确定性模式: seed={args.seed}")
    else:
        torch.backends.cudnn.deterministic = False
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = bool(args.cudnn_benchmark)
        logging.info(
            f"使用速度优先模式: cudnn.benchmark={torch.backends.cudnn.benchmark}"
        )

    logging.info(f"加载模型: {args.model_dir}")
    model = load_model(args.model_dir, args.valid_iters, args.max_disp, device)

    logging.info("启动 RealSense 流...")
    pipeline, fx, baseline, depth_scale = setup_realsense(
        args.width, args.height, args.fps
    )
    logging.info(
        f"fx={fx:.3f}, baseline={baseline:.6f}m, depth_scale={depth_scale:.6f}m"
    )

    ema_fps = 0.0
    try:
        while True:
            loop_start = time.perf_counter()
            frames = pipeline.wait_for_frames()
            left_frame = frames.get_infrared_frame(1)
            right_frame = frames.get_infrared_frame(2)
            depth_frame = frames.get_depth_frame()
            if not left_frame or not right_frame or not depth_frame:
                continue

            left_gray = np.asanyarray(left_frame.get_data())
            right_gray = np.asanyarray(right_frame.get_data())
            native_depth = (
                np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_scale
            )
            capture_time = time.perf_counter() - loop_start

            infer_start = time.perf_counter()
            pred_depth, infer_timing = predict_depth(
                model, left_gray, right_gray, fx, baseline, args, device
            )
            infer_time = time.perf_counter() - infer_start

            valid_mask = (
                (pred_depth > args.min_depth)
                & (pred_depth < args.max_depth)
                & (native_depth > args.min_depth)
                & (native_depth < args.max_depth)
            )
            if valid_mask.any():
                diff = pred_depth - native_depth
                mae = float(np.mean(np.abs(diff[valid_mask])))
                rmse = float(np.sqrt(np.mean((diff[valid_mask]) ** 2)))
            else:
                mae = float("nan")
                rmse = float("nan")

            pred_invalid = (pred_depth <= args.min_depth) | (
                pred_depth >= args.vis_max_depth
            )
            native_invalid = (native_depth <= args.min_depth) | (
                native_depth >= args.vis_max_depth
            )
            err = np.abs(pred_depth - native_depth)
            err_invalid = ~valid_mask

            left_vis = cv2.cvtColor(left_gray, cv2.COLOR_GRAY2BGR)
            pred_vis = colorize_depth(
                pred_depth, args.min_depth, args.vis_max_depth, pred_invalid
            )
            native_vis = colorize_depth(
                native_depth, args.min_depth, args.vis_max_depth, native_invalid
            )
            err_vis = colorize_error(err, args.error_max, err_invalid)

            sample_points = sample_points_from_valid_mask(
                valid_mask, args.sample_points
            )
            if sample_points:
                draw_sample_markers(
                    left_vis,
                    pred_vis,
                    native_vis,
                    err_vis,
                    sample_points,
                    pred_depth - native_depth,
                )

            top = np.concatenate([left_vis, pred_vis], axis=1)
            bottom = np.concatenate([native_vis, err_vis], axis=1)
            canvas = np.concatenate([top, bottom], axis=0)

            h, w = left_vis.shape[:2]
            draw_label(canvas, "Left IR", 12, 28)
            draw_label(canvas, "FoundationStereo Depth (m)", w + 12, 28)
            draw_label(canvas, "RealSense Native Depth (m)", 12, h + 28)
            draw_label(canvas, "Abs Error (m)", w + 12, h + 28)
            draw_label(canvas, "Error gray: brighter = larger error", w + 12, h + 54)

            total_time = time.perf_counter() - loop_start
            inst_fps = 1.0 / max(total_time, 1e-6)
            infer_fps = 1.0 / max(infer_time, 1e-6)
            ema_fps = inst_fps if ema_fps == 0 else (0.9 * ema_fps + 0.1 * inst_fps)

            stats = f"FPS(loop): {inst_fps:.1f} | FPS(EMA): {ema_fps:.1f} | FPS(infer): {infer_fps:.1f} | MAE: {mae:.3f}m | RMSE: {rmse:.3f}m"
            draw_label(canvas, stats, 12, canvas.shape[0] - 14)
            timing_stats = (
                f"capture:{capture_time*1000:.1f}ms | prep:{infer_timing['prep_ms']:.1f}ms | "
                f"forward:{infer_timing['forward_ms']:.1f}ms | post:{infer_timing['post_ms']:.1f}ms"
            )
            draw_label(canvas, timing_stats, 12, canvas.shape[0] - 42)

            cv2.imshow(args.window_name, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
