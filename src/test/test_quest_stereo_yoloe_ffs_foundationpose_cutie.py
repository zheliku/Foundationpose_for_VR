from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from quest_stereo_pose_pipeline import (
    QuestStereoPoseConfig,
    run_quest_stereo_pose,
)


# 手动联调脚本（非单元测试）：
# pixi run python src/test/test_quest_stereo_yoloe_ffs_foundationpose_cutie.py
# 或 pixi run python src/quest_stereo_pose_pipeline.py


def main() -> None:
    project_dir = Path(__file__).resolve().parent.parent.parent
    config = QuestStereoPoseConfig(
        listen_port=5557,
        project_dir=project_dir,
        calib_dir=project_dir / "docs/20260322_070544",
        mesh_path=project_dir / "data/online/cube/mesh/cube.stl",
        yoloe_model=project_dir / "checkpoints/yoloe-26l-seg.pt",
        mobileclip2_ts_path=project_dir / "mobileclip2_b.ts",
        yoloe_prompt="white block",
        ffs_model_path=project_dir
        / "Fast-FoundationStereo/weights/20-30-48/model_best_bp2_serialize.pth",
        process_width=640,
        process_height=480,
        show_window=True,
        enable_cutie=True,
        max_frames=0,
        stats_interval=30,
    )
    run_quest_stereo_pose(config)


if __name__ == "__main__":
    main()
