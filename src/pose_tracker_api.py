"""
6D 位姿追踪器 API

提供封装好的 PoseTracker 类，供 relay_server.py 调用。
支持两阶段输出：
  - 检测阶段 (DETECTING): SAM3 检测目标，返回原始 RGB
  - 追踪阶段 (TRACKING): FoundationPose 追踪，返回 6D 位姿 + 带标记的 RGB

使用方法：
    from src.pose_tracker_api import PoseTracker

    tracker = PoseTracker(mesh_path, cam_K, text_prompt)
    result = tracker.process_frame(color, depth)
    if result.phase == PoseTracker.Phase.TRACKING:
        print(result.pose_matrix)
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh
from numpy.typing import NDArray

# 添加必要的路径
src_path = Path(__file__).parent.parent.resolve()
foundationpose_path = src_path / "FoundationPose"
sam3_path = src_path / "sam3"

for path in [src_path, foundationpose_path, sam3_path]:
    if str(path) not in sys.path:
        sys.path.append(str(path))

from hydra.core.global_hydra import GlobalHydra

from FoundationPose.estimater import (
    FoundationPose,
    PoseRefinePredictor,
    ScorePredictor,
    dr,
    draw_posed_3d_box,
    draw_xyz_axis,
    trimesh_add_pure_colored_texture,
)
from PIL import Image as PILImage
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model
from src.VOT import Cutie, Tracker_2D
from src.utils.pose_tool import adjust_pose_to_image_point


class PoseTracker:
    """6D 位姿追踪器 API"""

    class Phase(Enum):
        """追踪阶段"""

        DETECTING = 0  # SAM3 检测阶段
        TRACKING = 1  # FoundationPose 追踪阶段

    @dataclass
    class Result:
        """追踪结果"""

        phase: "PoseTracker.Phase"
        color: NDArray[np.uint8]  # 原始/带标记的 RGB 图像 (BGR)
        pose_matrix: NDArray[np.float64] | None  # 4x4 位姿矩阵 (仅追踪阶段)

    def __init__(
        self,
        mesh_path: str,
        cam_K: NDArray[np.float64],
        text_prompt: str,
        apply_scale: float = 1.0,
        force_apply_color: bool = True,  # 匹配 on_demo3 默认值
        apply_color: list[int] | None = None,
        sam3_confidence_threshold: float = 0.8,  # 匹配 on_demo3 默认值
        est_refine_iter: int = 10,
        track_refine_iter: int = 5,
        activate_2d_tracker: bool = True,
        debug_output_dir: str | None = None,  # 调试输出目录
    ) -> None:
        """
        初始化位姿追踪器

        参数
        ----
        mesh_path : str
            目标物体的 3D 模型文件路径 (.stl, .ply, .obj 等)
        cam_K : NDArray
            3x3 相机内参矩阵
        text_prompt : str
            SAM3 文本提示，描述要检测的目标物体 (如 'white cube')
        apply_scale : float
            模型缩放因子 (默认 1.0)
        force_apply_color : bool
            是否为无色模型强制应用颜色
        apply_color : list[int]
            RGB 颜色值 [r, g, b]
        sam3_confidence_threshold : float
            SAM3 检测置信度阈值 (0-1)
        est_refine_iter : int
            初始化阶段精炼迭代次数
        track_refine_iter : int
            追踪阶段精炼迭代次数
        activate_2d_tracker : bool
            是否启用 2D 跟踪器辅助
        """
        self.cam_K = cam_K
        self.text_prompt = text_prompt
        self.sam3_confidence_threshold = sam3_confidence_threshold
        self.est_refine_iter = est_refine_iter
        self.track_refine_iter = track_refine_iter
        self.activate_2d_tracker = activate_2d_tracker
        self.debug_output_dir = Path(debug_output_dir) if debug_output_dir else None

        if apply_color is None:
            apply_color = [0, 159, 237]

        # 当前阶段
        self._phase = self.Phase.DETECTING
        self._frame_index = 0

        # 检测帧数据（用于确保检测帧与注册帧一致）
        self._detection_color: NDArray[np.uint8] | None = None
        self._detection_depth: NDArray[np.float64] | None = None
        self._detection_mask: NDArray[np.uint8] | None = None

        # 加载 3D 网格模型
        print(f"[PoseTracker] 正在加载 3D 模型: {mesh_path}")
        self.mesh, self.to_origin, self.bbox = self._prepare_mesh(
            mesh_path, apply_scale, force_apply_color, apply_color
        )

        # 初始化 FoundationPose
        print("[PoseTracker] 正在初始化 FoundationPose...")
        self.fp = self._prepare_foundationpose(self.mesh)

        # 初始化 2D 跟踪器
        if self.activate_2d_tracker:
            if GlobalHydra.instance().is_initialized():
                GlobalHydra.instance().clear()
            self.tracker_2d: Tracker_2D = Cutie()
        else:
            self.tracker_2d = Tracker_2D()

        # 初始化 SAM3
        print("[PoseTracker] 正在加载 SAM3 模型...")
        self.sam3_processor = self._prepare_sam3()

        print("[PoseTracker] 初始化完成！")

    def _prepare_mesh(
        self,
        mesh_path: str,
        apply_scale: float,
        force_apply_color: bool,
        apply_color: list[int],
    ) -> tuple[trimesh.Trimesh, NDArray[np.float64], NDArray[np.float64]]:
        """加载并预处理 3D 网格"""
        mesh = trimesh.load(mesh_path)
        mesh.apply_scale(apply_scale)

        if force_apply_color:
            mesh = trimesh_add_pure_colored_texture(
                mesh, color=np.array(apply_color), resolution=10
            )

        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

        return mesh, to_origin, bbox

    def _prepare_foundationpose(self, mesh: trimesh.Trimesh) -> FoundationPose:
        """初始化 FoundationPose 估计器"""
        scorer = ScorePredictor()
        refiner = PoseRefinePredictor()
        glctx = dr.RasterizeCudaContext()

        return FoundationPose(
            model_pts=mesh.vertices,
            model_normals=mesh.vertex_normals,
            mesh=mesh,
            scorer=scorer,
            refiner=refiner,
            glctx=glctx,
            debug_dir="FoundationPose/debug/",
        )

    def _prepare_sam3(self) -> Sam3Processor:
        """初始化 SAM3 处理器"""
        checkpoint_path = str(sam3_path / "assets/sam3_ckpt/sam3.pt")
        model = build_sam3_image_model(
            checkpoint_path=checkpoint_path, load_from_HF=False
        )
        return Sam3Processor(model, confidence_threshold=self.sam3_confidence_threshold)

    def process_frame(
        self, color: NDArray[np.uint8], depth: NDArray[np.float64]
    ) -> Result:
        """
        处理单帧 RGBD 数据

        参数
        ----
        color : NDArray[np.uint8]
            BGR 彩色图像
        depth : NDArray[np.float64]
            深度图像 (单位：米)

        返回
        ----
        Result
            包含阶段、图像和位姿的结果
        """
        if self._phase == self.Phase.DETECTING:
            return self._process_detecting(color, depth)
        else:
            return self._process_tracking(color, depth)

    def _process_detecting(
        self, color: NDArray[np.uint8], depth: NDArray[np.float64]
    ) -> Result:
        """检测阶段：使用 SAM3 检测目标"""
        # BGR -> RGB
        color_rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(color_rgb)

        # SAM3 检测
        inference_state = self.sam3_processor.set_image(pil_image)
        output = self.sam3_processor.set_text_prompt(
            state=inference_state, prompt=self.text_prompt
        )

        masks, boxes, scores = output["masks"], output["boxes"], output["scores"]

        if len(masks) > 0:
            best_idx = scores.argmax().item()
            best_mask = masks[best_idx]
            best_score = scores[best_idx].item()

            if best_score >= self.sam3_confidence_threshold:
                # 检测成功！转换 mask 并切换到追踪阶段
                if isinstance(best_mask, torch.Tensor):
                    mask_np = best_mask.cpu().numpy()
                else:
                    mask_np = np.array(best_mask)

                if len(mask_np.shape) == 3:
                    mask_np = mask_np.squeeze(0)

                init_mask = (mask_np > 0.5).astype(np.uint8) * 255

                # 保存检测帧数据（确保检测帧与注册帧一致）
                self._detection_color = color.copy()
                self._detection_depth = depth.copy()
                self._detection_mask = init_mask

                # 初始化追踪（使用同一帧的 RGBD + mask）
                self._initialize_tracking(color, depth, init_mask)

                # 返回第一帧追踪结果（使用检测帧数据）
                return self._get_first_tracking_result()

        # 未检测到，返回原始图像
        return self.Result(phase=self.Phase.DETECTING, color=color, pose_matrix=None)

    def _initialize_tracking(
        self,
        color: NDArray[np.uint8],
        depth: NDArray[np.float64],
        init_mask: NDArray[np.uint8],
    ) -> None:
        """初始化追踪状态"""
        print("[PoseTracker] 检测到目标！开始追踪...")

        # 保存调试数据
        self._save_debug_data(color, depth, init_mask)

        # FoundationPose 注册初始位姿
        self._pose = self.fp.register(
            K=self.cam_K,
            rgb=color,
            depth=depth,
            ob_mask=init_mask,
            iteration=self.est_refine_iter,
        )

        # 初始化 2D 跟踪器
        self.tracker_2d.initialize(color, init_info={"mask": init_mask})

        # 切换阶段
        self._phase = self.Phase.TRACKING
        self._frame_index = 0

    def _save_debug_data(
        self,
        color: NDArray[np.uint8],
        depth: NDArray[np.float64],
        mask: NDArray[np.uint8],
    ) -> None:
        """保存调试数据到指定目录"""
        if self.debug_output_dir is None:
            return

        self.debug_output_dir.mkdir(parents=True, exist_ok=True)

        # 保存彩色图像
        cv2.imwrite(str(self.debug_output_dir / "detection_color.png"), color)

        # 保存深度图像（转换为可视化格式）
        depth_vis = (depth * 1000).astype(np.uint16)  # 转回 mm
        cv2.imwrite(str(self.debug_output_dir / "detection_depth.png"), depth_vis)

        # 保存 mask
        cv2.imwrite(str(self.debug_output_dir / "detection_mask.png"), mask)

        print(f"[PoseTracker] 调试数据已保存到: {self.debug_output_dir}")

    def _get_first_tracking_result(self) -> Result:
        """获取首帧追踪结果（使用检测帧数据）"""
        # 获取位姿矩阵
        pose_matrix = self._pose.reshape(4, 4)

        # 使用检测帧绘制可视化
        vis_color = self._draw_visualization(self._detection_color, pose_matrix)

        self._frame_index += 1

        return self.Result(
            phase=self.Phase.TRACKING,
            color=vis_color,
            pose_matrix=pose_matrix,
        )

    def _process_tracking(
        self, color: NDArray[np.uint8], depth: NDArray[np.float64]
    ) -> Result:
        """追踪阶段：使用 FoundationPose 追踪位姿"""
        if self._frame_index > 0:
            # 2D 跟踪器辅助
            if self.activate_2d_tracker:
                bbox_2d = self.tracker_2d.track(color)
                # 用 2D 框中心修正位姿先验
                self.fp.pose_last = adjust_pose_to_image_point(
                    ob_in_cam=self.fp.pose_last,
                    K=self.cam_K,
                    x=bbox_2d[0] + bbox_2d[2] / 2,
                    y=bbox_2d[1] + bbox_2d[3] / 2,
                )

            # FoundationPose 追踪
            self._pose = self.fp.track_one(
                rgb=color,
                depth=depth,
                K=self.cam_K,
                iteration=self.track_refine_iter,
            )

        self._frame_index += 1

        # 获取位姿矩阵
        pose_matrix = self._pose.reshape(4, 4)

        # 绘制可视化
        vis_color = self._draw_visualization(color, pose_matrix)

        return self.Result(
            phase=self.Phase.TRACKING,
            color=vis_color,
            pose_matrix=pose_matrix,
        )

    def _draw_visualization(
        self, color: NDArray[np.uint8], pose: NDArray[np.float64]
    ) -> NDArray[np.uint8]:
        """在图像上绘制 3D 包围盒和坐标轴"""
        center_pose = pose @ np.linalg.inv(self.to_origin)

        vis = draw_posed_3d_box(
            self.cam_K, img=color, ob_in_cam=center_pose, bbox=self.bbox
        )
        vis = draw_xyz_axis(
            vis,
            ob_in_cam=center_pose,
            scale=0.1,
            K=self.cam_K,
            thickness=3,
            transparency=0,
            is_input_rgb=True,
        )
        return vis

    def reset(self) -> None:
        """重置追踪器，回到检测阶段"""
        self._phase = self.Phase.DETECTING
        self._frame_index = 0
        print("[PoseTracker] 已重置，返回检测阶段")

    @property
    def phase(self) -> Phase:
        """当前追踪阶段"""
        return self._phase

    @property
    def is_tracking(self) -> bool:
        """是否处于追踪阶段"""
        return self._phase == self.Phase.TRACKING
