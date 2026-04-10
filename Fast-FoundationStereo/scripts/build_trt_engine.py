from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def _platform_tag() -> str:
    if os.name == "nt":
        return "win"
    if sys.platform.startswith("linux"):
        return "linux"
    if sys.platform == "darwin":
        return "mac"
    return "unknown"


def build_artifact_tag(height: int, width: int, valid_iters: int, max_disp: int) -> str:
    """构建参数标签，与 make_onnx.py 保持一致。"""
    return f"h{height}-w{width}-it{valid_iters}-md{max_disp}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build TensorRT engines from exported ONNX models."
    )
    code_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "--onnx_dir",
        type=str,
        default=str(code_dir.parent / "output"),
        help="ONNX 所在目录。",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="导出时使用的固定输入高度（用于自动生成标签）。",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="导出时使用的固定输入宽度（用于自动生成标签）。",
    )
    parser.add_argument(
        "--valid_iters",
        type=int,
        default=4,
        help="导出时固化的迭代次数（用于自动生成标签）。",
    )
    parser.add_argument(
        "--max_disp",
        type=int,
        default=192,
        help="导出时固化的 max_disp（用于自动生成标签）。",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="可选自定义标签；为空则按参数自动生成。",
    )
    parser.add_argument(
        "--feature_onnx", type=str, default="",
        help="特征提取模型的 ONNX 文件名；为空则按标签自动生成。",
    )
    parser.add_argument(
        "--post_onnx", type=str, default="",
        help="后处理模型的 ONNX 文件名；为空则按标签自动生成。",
    )
    parser.add_argument(
        "--feature_engine", type=str, default="",
        help="特征提取模型的 engine 输出文件名；为空则按标签自动生成。",
    )
    parser.add_argument(
        "--post_engine", type=str, default="",
        help="后处理模型的 engine 输出文件名；为空则按标签自动生成。",
    )
    parser.add_argument(
        "--platform_tag",
        type=str,
        default="",
        help="平台标签（默认自动识别 win/linux/mac）。",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp16", "fp32"],
        default="fp16",
        help="engine 目标精度。",
    )
    parser.add_argument(
        "--workspace_gb", type=float, default=4.0, help="TensorRT 工作空间大小（GB）。"
    )
    parser.add_argument("--verbose", action="store_true", help="启用 TensorRT 详细日志输出。")
    return parser.parse_args()


def _build_one_engine(trt, logger, onnx_path, engine_path, workspace_gb, enable_fp16):
    t0 = time.perf_counter()
    print(f"[TRT] parsing: {onnx_path}")
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        model_bytes = f.read()
    if not parser.parse(model_bytes):
        errors = [parser.get_error(i) for i in range(parser.num_errors)]
        details = "\n".join([str(e) for e in errors])
        raise RuntimeError(f"Failed to parse ONNX file: {onnx_path}\n{details}")
    print(f"[TRT] layers: {network.num_layers}")

    config = builder.create_builder_config()
    workspace_size = int(workspace_gb * (1 << 30))
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)

    if enable_fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("[TRT] fp16 enabled")
        else:
            print(
                "[WARN] FP16 requested but platform_has_fast_fp16 is False. Building with FP32."
            )

    print("[TRT] building engine...")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError(f"TensorRT build failed for {onnx_path}")

    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    print(f"[TRT] saved: {engine_path} ({time.perf_counter() - t0:.2f}s)")


if __name__ == "__main__":
    t_total = time.perf_counter()
    args = parse_args()

    try:
        import tensorrt as trt
    except ImportError as exc:
        raise SystemExit(
            "Cannot import tensorrt. Please install TensorRT Python package first (via pixi)."
        ) from exc

    onnx_dir = Path(args.onnx_dir).resolve()
    if not onnx_dir.is_dir():
        raise FileNotFoundError(f"ONNX 目录不存在: {onnx_dir}")

    platform_tag = args.platform_tag.strip() or _platform_tag()
    tag = args.tag.strip() or build_artifact_tag(
        args.height,
        args.width,
        args.valid_iters,
        args.max_disp,
    )

    feature_onnx_name = args.feature_onnx or f"feature_runner-{tag}.onnx"
    post_onnx_name = args.post_onnx or f"post_runner-{tag}.onnx"

    feature_onnx_path = onnx_dir / feature_onnx_name
    post_onnx_path = onnx_dir / post_onnx_name

    if not feature_onnx_path.is_file():
        raise FileNotFoundError(f"Cannot find ONNX file: {feature_onnx_path}")
    if not post_onnx_path.is_file():
        raise FileNotFoundError(f"Cannot find ONNX file: {post_onnx_path}")

    feature_engine_name = args.feature_engine or (
        f"feature_runner-{tag}.{platform_tag}.{args.precision}.engine"
    )
    post_engine_name = args.post_engine or (
        f"post_runner-{tag}.{platform_tag}.{args.precision}.engine"
    )
    feature_engine_path = onnx_dir / feature_engine_name
    post_engine_path = onnx_dir / post_engine_name

    print("[TRT] === Build Start ===")
    print(f"[TRT] tag: {tag}")
    print(f"[TRT] platform_tag: {platform_tag}")
    print(f"[TRT] onnx_dir: {onnx_dir}")
    print(f"[TRT] feature_onnx: {feature_onnx_path}")
    print(f"[TRT] post_onnx: {post_onnx_path}")
    print(f"[TRT] feature_engine: {feature_engine_path}")
    print(f"[TRT] post_engine: {post_engine_path}")
    print(f"[TRT] workspace_gb={args.workspace_gb}, precision={args.precision}")

    logger_level = trt.Logger.VERBOSE if args.verbose else trt.Logger.WARNING
    logger = trt.Logger(logger_level)

    print("[TRT] building feature engine")
    _build_one_engine(
        trt,
        logger,
        str(feature_onnx_path),
        str(feature_engine_path),
        args.workspace_gb,
        args.precision == "fp16",
    )

    print("[TRT] building post engine")
    _build_one_engine(
        trt,
        logger,
        str(post_onnx_path),
        str(post_engine_path),
        args.workspace_gb,
        args.precision == "fp16",
    )

    print(f"[TRT] done in {time.perf_counter() - t_total:.2f}s")
