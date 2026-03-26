from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import cv2
import numpy as np
import trimesh
from numpy.typing import NDArray

from modules.realsense_rgbd import RGBDFrame
from utils.pose_tool import adjust_pose_to_image_point


@dataclass(slots=True)
class PoseResult:
    pose_4x4: NDArray[np.float64] | None
    vis_bgr: NDArray[np.uint8]


class PoseEstimator(Protocol):
    def initialize(self, frame: RGBDFrame, mask_u8) -> PoseResult: ...

    def track(self, frame: RGBDFrame) -> PoseResult: ...


def _append_import_paths(project_dir: Path) -> None:
    fp_path = project_dir / "FoundationPose"
    preferred = [str(fp_path), str(project_dir)]
    for p in preferred:
        while p in sys.path:
            sys.path.remove(p)
    for p in reversed(preferred):
        sys.path.insert(0, p)


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
        sys.modules.pop("Utils", None)

        from FoundationPose.estimater import (
            FoundationPose,
            PoseRefinePredictor,
            ScorePredictor,
        )
        from FoundationPose.Utils import (
            draw_posed_3d_box,
            draw_xyz_axis,
            trimesh_add_pure_colored_texture,
        )

        try:
            from FoundationPose.Utils import dr as fp_dr
        except Exception:
            import nvdiffrast.torch as fp_dr

        self._draw_posed_3d_box = draw_posed_3d_box
        self._draw_xyz_axis = draw_xyz_axis
        self._cam_k = config.cam_k.astype(np.float64)

        mesh = cast(trimesh.Trimesh, trimesh.load(str(config.mesh_path)))
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
        glctx = None

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

    @property
    def pose(self) -> np.ndarray | None:
        if self._pose is None:
            return None
        return self._pose.copy()

    def set_camera_k(self, cam_k: np.ndarray) -> None:
        self._cam_k = cam_k.astype(np.float64)

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

    def apply_tracking_hint(self, x: float, y: float) -> None:
        if self._pose is None:
            return
        if self._fp.pose_last is None:
            return
        self._fp.pose_last = adjust_pose_to_image_point(
            ob_in_cam=self._fp.pose_last,
            K=self._cam_k,
            x=float(x),
            y=float(y),
        )

    def draw_pose(self, color_bgr: np.ndarray, pose_4x4: np.ndarray) -> np.ndarray:
        return self._draw(color_bgr, pose_4x4)

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
