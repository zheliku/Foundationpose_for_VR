from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import trimesh

from vpt_modules.types import PoseResult, RGBDFrame


def _append_import_paths(project_dir: Path) -> None:
    fp_path = project_dir / "FoundationPose"
    for p in [project_dir, fp_path]:
        ps = str(p)
        if ps not in sys.path:
            sys.path.append(ps)


@dataclass(slots=True)
class FoundationPoseConfig:
    project_dir: Path
    mesh_path: Path
    cam_k: np.ndarray
    symmetry_tfs: np.ndarray | None = None
    est_refine_iter: int = 10
    track_refine_iter: int = 5
    apply_scale: float = 1.0
    force_apply_color: bool = True
    apply_color: tuple[int, int, int] = (0, 159, 237)


class FoundationPoseEstimator:
    def __init__(self, config: FoundationPoseConfig) -> None:
        self.config = config
        if not config.mesh_path.exists():
            raise FileNotFoundError(f"mesh 不存在: {config.mesh_path}")

        _append_import_paths(config.project_dir)

        from FoundationPose.estimater import (
            FoundationPose,
            PoseRefinePredictor,
            ScorePredictor,
            dr,
            draw_posed_3d_box,
            draw_xyz_axis,
            trimesh_add_pure_colored_texture,
        )

        self._draw_posed_3d_box = draw_posed_3d_box
        self._draw_xyz_axis = draw_xyz_axis
        self._cam_k = config.cam_k.astype(np.float64)

        mesh = trimesh.load(str(config.mesh_path))
        mesh.apply_scale(config.apply_scale)
        if config.force_apply_color:
            mesh = trimesh_add_pure_colored_texture(
                mesh,
                color=np.array(config.apply_color, dtype=np.uint8),
                resolution=10,
            )

        self._to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        self._bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

        scorer = ScorePredictor()
        refiner = PoseRefinePredictor()
        glctx = dr.RasterizeCudaContext()

        self._fp = FoundationPose(
            model_pts=mesh.vertices,
            model_normals=mesh.vertex_normals,
            symmetry_tfs=config.symmetry_tfs,
            mesh=mesh,
            scorer=scorer,
            refiner=refiner,
            glctx=glctx,
            debug_dir=str(config.project_dir / "FoundationPose" / "debug"),
        )
        self._pose: np.ndarray | None = None

    def initialize(self, frame: RGBDFrame, mask_u8: np.ndarray) -> PoseResult:
        self._validate_frame_shapes(frame)

        rgb_h, rgb_w = frame.color_bgr.shape[:2]
        if mask_u8.shape != (rgb_h, rgb_w):
            mask_u8 = cv2.resize(
                mask_u8, (rgb_w, rgb_h), interpolation=cv2.INTER_NEAREST
            )

        mask_binary = (mask_u8 > 0).astype(np.uint8) * 255
        mask_area_ratio = float((mask_binary > 0).sum()) / float(rgb_h * rgb_w)
        if mask_area_ratio <= 0.0:
            return PoseResult(pose_4x4=None, vis_bgr=frame.color_bgr)

        self._pose = self._fp.register(
            K=self._cam_k,
            rgb=frame.color_bgr,
            depth=frame.depth_m,
            ob_mask=mask_binary,
            iteration=int(self.config.est_refine_iter),
        )
        vis = self._draw(frame.color_bgr, self._pose)
        return PoseResult(pose_4x4=self._pose.copy(), vis_bgr=vis)

    def track(self, frame: RGBDFrame) -> PoseResult:
        if self._pose is None:
            return PoseResult(pose_4x4=None, vis_bgr=frame.color_bgr)
        self._pose = self._fp.track_one(
            rgb=frame.color_bgr,
            depth=frame.depth_m,
            K=self._cam_k,
            iteration=int(self.config.track_refine_iter),
        )
        vis = self._draw(frame.color_bgr, self._pose)
        return PoseResult(pose_4x4=self._pose.copy(), vis_bgr=vis)

    def _draw(self, color_bgr: np.ndarray, pose_4x4: np.ndarray) -> np.ndarray:
        vis = color_bgr.copy()
        vis = self._draw_posed_3d_box(
            self._cam_k, vis, ob_in_cam=pose_4x4, bbox=self._bbox
        )
        vis = self._draw_xyz_axis(
            vis,
            ob_in_cam=pose_4x4,
            scale=0.1,
            K=self._cam_k,
            thickness=3,
            transparency=0,
            is_input_rgb=True,
        )
        return vis

    def _validate_frame_shapes(self, frame: RGBDFrame) -> None:
        if frame.color_bgr.ndim != 3 or frame.color_bgr.shape[2] != 3:
            raise ValueError(f"color_bgr 形状非法: {frame.color_bgr.shape}")

        if frame.depth_m.ndim != 2:
            raise ValueError(f"depth_m 形状非法: {frame.depth_m.shape}")

        if frame.color_bgr.shape[:2] != frame.depth_m.shape[:2]:
            raise ValueError(
                f"RGB/Depth 尺寸不一致: color={frame.color_bgr.shape[:2]}, depth={frame.depth_m.shape[:2]}"
            )
