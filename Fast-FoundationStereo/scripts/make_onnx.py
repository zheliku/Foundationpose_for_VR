from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch

os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

CODE_DIR = Path(__file__).resolve().parent
sys.path.append(str(CODE_DIR.parent))

from core.foundation_stereo import (
    TrtFeatureRunner,
    TrtPostRunner,
    build_gwc_volume_triton,
)


def build_artifact_tag(height: int, width: int, valid_iters: int, max_disp: int) -> str:
    """构建用于文件命名的参数标签。

    说明：
    - TRT engine 会固化部分关键参数（分辨率、迭代次数、max_disp）。
    - 因此导出文件名必须携带这些参数，运行时才能稳定匹配。
    """
    return f"h{height}-w{width}-it{valid_iters}-md{max_disp}"


def build_onnx_names(tag: str) -> tuple[str, str]:
    """返回 feature/post ONNX 文件名。"""
    return (
        f"feature_runner-{tag}.onnx",
        f"post_runner-{tag}.onnx",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Fast-FoundationStereo .pth checkpoint to ONNX artifacts."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=str(CODE_DIR.parent / "weights" / "model_best_bp2_serialize.pth"),
        help="pth 权重文件路径。",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="",
        help="输出目录。默认与 pth 同目录。",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="导出时固定输入高度（建议与实时输入一致）。",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="导出时固定输入宽度（建议与实时输入一致）。",
    )
    parser.add_argument(
        "--valid_iters",
        type=int,
        default=4,
        help="GRU refinement 迭代次数。",
    )
    parser.add_argument(
        "--max_disp",
        type=int,
        default=192,
        help="最大视差，影响体构建范围与速度。",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="可选自定义标签；为空则自动按参数生成。",
    )
    return parser.parse_args()


def main() -> None:
    total_t0 = time.perf_counter()
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("导出 ONNX 需要可用 CUDA 环境。")

    model_path = Path(args.model_dir).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"未找到模型文件: {model_path}")

    save_dir = Path(args.save_path).resolve() if args.save_path else model_path.parent
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.height % 32 != 0 or args.width % 32 != 0:
        raise ValueError("height 和 width 必须可被 32 整除。")

    tag = args.tag.strip() or build_artifact_tag(
        args.height,
        args.width,
        args.valid_iters,
        args.max_disp,
    )
    feature_onnx_name, post_onnx_name = build_onnx_names(tag)
    feature_onnx_path = save_dir / feature_onnx_name
    post_onnx_path = save_dir / post_onnx_name

    print("[ONNX] === Export Start ===")
    print(f"[ONNX] model: {model_path}")
    print(f"[ONNX] save_dir: {save_dir}")
    print(f"[ONNX] tag: {tag}")
    print(f"[ONNX] input_size: {args.height}x{args.width}")
    print(f"[ONNX] valid_iters={args.valid_iters}, max_disp={args.max_disp}")

    torch.autograd.set_grad_enabled(False)

    # 关键步骤：先加载 pth 并把导出相关参数写回 model.args。
    # 这样导出的 post_runner 图里会固化对应迭代次数与几何范围。
    t_load = time.perf_counter()
    model = torch.load(str(model_path), map_location="cpu", weights_only=False)
    model.args.max_disp = int(args.max_disp)
    model.args.valid_iters = int(args.valid_iters)
    model.cuda().eval()
    print(f"[ONNX] model loaded in {time.perf_counter() - t_load:.2f}s")

    feature_runner = TrtFeatureRunner(model).cuda().eval()
    post_runner = TrtPostRunner(model).cuda().eval()

    left_img = (
        torch.randn(1, 3, args.height, args.width, device="cuda", dtype=torch.float32)
        * 255.0
    )
    right_img = (
        torch.randn(1, 3, args.height, args.width, device="cuda", dtype=torch.float32)
        * 255.0
    )

    print("[ONNX] exporting feature runner...")
    t_feature = time.perf_counter()
    torch.onnx.export(
        feature_runner,
        (left_img, right_img),
        str(feature_onnx_path),
        opset_version=17,
        input_names=["left", "right"],
        output_names=[
            "features_left_04",
            "features_left_08",
            "features_left_16",
            "features_left_32",
            "features_right_04",
            "stem_2x",
        ],
        do_constant_folding=True,
    )
    print(f"[ONNX] saved: {feature_onnx_path} ({time.perf_counter() - t_feature:.2f}s)")

    (
        features_left_04,
        features_left_08,
        features_left_16,
        features_left_32,
        features_right_04,
        stem_2x,
    ) = feature_runner(left_img, right_img)

    gwc_volume = build_gwc_volume_triton(
        features_left_04.half(),
        features_right_04.half(),
        int(args.max_disp) // 4,
        model.cv_group,
    )

    print("[ONNX] exporting post runner...")
    t_post = time.perf_counter()
    torch.onnx.export(
        post_runner,
        (
            features_left_04,
            features_left_08,
            features_left_16,
            features_left_32,
            features_right_04,
            stem_2x,
            gwc_volume,
        ),
        str(post_onnx_path),
        opset_version=17,
        input_names=[
            "features_left_04",
            "features_left_08",
            "features_left_16",
            "features_left_32",
            "features_right_04",
            "stem_2x",
            "gwc_volume",
        ],
        output_names=["disp"],
        do_constant_folding=True,
    )
    print(f"[ONNX] saved: {post_onnx_path} ({time.perf_counter() - t_post:.2f}s)")

    print(f"[ONNX] done in {time.perf_counter() - total_t0:.2f}s")


if __name__ == "__main__":
    main()
