"""
FoundationPose 6D 位姿估计 API（模块化版）

设计目标：
1. 输入明确：cam_k、rgb、depth、obj(mesh)、mask。
2. 输出明确：目标物体 4x4 位姿矩阵 pose。
3. 保持简洁：仅封装核心 register/track 能力。
4. 与 2D tracker 解耦：不直接依赖 cutie.py，只接收可选的 2D 引导点。
"""

from __future__ import annotations

import importlib
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import trimesh


@dataclass
class FoundationPoseConfig:
    """FoundationPose 初始化配置。"""

    # 目标物体网格模型路径（.obj/.ply/.stl 等）。
    mesh_path: str

    # 相机内参矩阵，shape=(3,3)。
    cam_k: np.ndarray

    # 初始阶段 refine 迭代次数（register）。
    est_refine_iter: int = 10

    # 跟踪阶段 refine 迭代次数（track）。
    track_refine_iter: int = 5

    # 模型缩放因子（例如厘米模型可设 0.01 转米）。
    apply_scale: float = 1.0

    # 对无纹理/无颜色模型强制上色。
    force_apply_color: bool = False

    # 强制上色 RGB。
    apply_color: list[int] | None = None

    # 对称变换（可选），shape=(N,4,4)。
    symmetry_tfs: np.ndarray | None = None

    # 调试开关与调试目录。
    debug: int = 0
    debug_dir: str | None = None


class FoundationPoseEstimator:
    """FoundationPose 估计器：支持 register + track。"""

    def __init__(self, config: FoundationPoseConfig) -> None:
        self.cfg = config

        # 标准化并缓存相机内参，后续每帧直接复用。
        # 这里使用 float64：FoundationPose 内部部分几何流程会把 mesh 顶点处理为 double，
        # 若 K 为 float32，注册阶段在矩阵乘法处会触发 float/double dtype 冲突。
        self.cam_k = np.asarray(self.cfg.cam_k, dtype=np.float64).reshape(3, 3)

        # 默认颜色配置。
        if self.cfg.apply_color is None:
            self.cfg.apply_color = [0, 159, 237]

        # 补充项目路径，确保可导入 FoundationPose 包。
        self.project_root = Path(__file__).resolve().parents[2]
        self.foundationpose_root = self.project_root / "FoundationPose"
        if str(self.project_root) not in sys.path:
            sys.path.append(str(self.project_root))
        if str(self.foundationpose_root) not in sys.path:
            sys.path.append(str(self.foundationpose_root))

        # 动态导入 FoundationPose 模块，避免静态路径问题。
        # 兼容两种运行方式：
        # 1) 作为包导入 FoundationPose.estimater
        # 2) 直接把 FoundationPose 目录加入 sys.path 后导入 estimater
        try:
            est_mod = importlib.import_module("FoundationPose.estimater")
        except ModuleNotFoundError:
            est_mod = importlib.import_module("estimater")

        self.ScorePredictor = est_mod.ScorePredictor
        self.PoseRefinePredictor = est_mod.PoseRefinePredictor
        self.dr = est_mod.dr
        self.FoundationPose = est_mod.FoundationPose
        self.trimesh_add_pure_colored_texture = est_mod.trimesh_add_pure_colored_texture
        self.draw_posed_3d_box = est_mod.draw_posed_3d_box
        self.draw_xyz_axis = est_mod.draw_xyz_axis

        # 加载并预处理 mesh。
        loaded_mesh = trimesh.load(self.cfg.mesh_path)
        if isinstance(loaded_mesh, trimesh.Scene):
            loaded_mesh = loaded_mesh.dump(concatenate=True)
        # trimesh 在静态类型上较宽泛，这里转 Any 以便后续直接访问 vertices 等属性。
        self.mesh = cast(Any, loaded_mesh)

        self.mesh.apply_scale(float(self.cfg.apply_scale))

        if bool(self.cfg.force_apply_color):
            self.mesh = self.trimesh_add_pure_colored_texture(
                self.mesh,
                color=np.array(self.cfg.apply_color),
                resolution=10,
            )

        # 计算包围盒与中心修正矩阵，供可视化使用。
        self.to_origin, extents = trimesh.bounds.oriented_bounds(self.mesh)
        self.bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

        # 初始化 FoundationPose 网络与渲染上下文。
        scorer = self.ScorePredictor()
        refiner = self.PoseRefinePredictor()
        glctx = self.dr.RasterizeCudaContext()

        self.estimator = self.FoundationPose(
            model_pts=self.mesh.vertices,
            model_normals=self.mesh.vertex_normals,
            symmetry_tfs=self.cfg.symmetry_tfs,
            mesh=self.mesh,
            scorer=scorer,
            refiner=refiner,
            glctx=glctx,
            debug_dir=self.cfg.debug_dir,
            debug=int(self.cfg.debug),
        )

        # 是否已完成初次注册。
        self._initialized = False

        logging.info("FoundationPose estimator initialization done")

    @staticmethod
    def _get_pose_xy_from_image_point(
        ob_in_cam: np.ndarray,
        cam_k: np.ndarray,
        x: float,
        y: float,
    ) -> tuple[float, float]:
        """
        根据给定图像点 (x,y) 反推相机坐标下新的 (tx,ty)。

        说明：
        - 保持当前 tz 不变。
        - 仅做平移修正，不改旋转。
        """
        t = ob_in_cam[:3, 3]
        fx = float(cam_k[0, 0])
        fy = float(cam_k[1, 1])
        cx = float(cam_k[0, 2])
        cy = float(cam_k[1, 2])
        tz = float(t[2])

        tx = (float(x) - cx) * tz / fx
        ty = (float(y) - cy) * tz / fy
        return tx, ty

    def adjust_pose_to_image_point(self, x: float, y: float) -> None:
        """
        用 2D 点约束修正上一帧位姿（供外部2D tracker辅助）。

        使用方式：
        - 先确保已经 register/track 过（存在 pose_last）
        - 再调用本方法
        - 最后调用 track() 完成本帧优化
        """
        # 若还未初始化，或 pose_last 不存在，则直接返回。
        if not hasattr(self.estimator, "pose_last"):
            return

        pose_last = self.estimator.pose_last

        # FoundationPose 内部常见是 torch tensor，这里统一转 numpy 处理后再转回。
        if hasattr(pose_last, "detach"):
            pose_np = pose_last.detach().cpu().numpy()
            is_tensor = True
            device = pose_last.device
            dtype = pose_last.dtype
        else:
            pose_np = np.asarray(pose_last)
            is_tensor = False
            device = None
            dtype = None

        # 支持 [4,4] 或 [1,4,4]。
        if pose_np.ndim == 3:
            mat = pose_np[0]
        else:
            mat = pose_np

        tx, ty = self._get_pose_xy_from_image_point(mat, self.cam_k, x, y)
        mat_new = mat.copy()
        mat_new[0, 3] = tx
        mat_new[1, 3] = ty

        if is_tensor:
            import torch

            out = torch.from_numpy(mat_new).to(device=device, dtype=dtype).unsqueeze(0)
            self.estimator.pose_last = out
        else:
            self.estimator.pose_last = mat_new

    def register(
        self, rgb: np.ndarray, depth: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """
        初始注册（第一帧）。

        输入：
        - rgb: 彩色图。
        - depth: 深度图（米制，float）。
        - mask: 目标 mask（bool/0-1/0-255 均可）。

        输出：
        - pose: 4x4 位姿矩阵（object in camera）。
        """
        # 统一 mask 到 uint8，兼容 FoundationPose register 输入。
        mask_u8 = (mask > 0).astype(np.uint8) * 255

        pose = self.estimator.register(
            K=self.cam_k,
            rgb=rgb,
            depth=depth,
            ob_mask=mask_u8,
            iteration=int(self.cfg.est_refine_iter),
        )

        self._initialized = True
        return np.asarray(pose).reshape(4, 4)

    def track(self, rgb: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """
        跟踪更新（后续帧）。

        输入：
        - rgb: 彩色图。
        - depth: 深度图（米制，float）。

        输出：
        - pose: 4x4 位姿矩阵（object in camera）。
        """
        if not self._initialized:
            raise RuntimeError("尚未初始化，请先调用 register()。")

        pose = self.estimator.track_one(
            rgb=rgb,
            depth=depth,
            K=self.cam_k,
            iteration=int(self.cfg.track_refine_iter),
        )
        return np.asarray(pose).reshape(4, 4)

    def estimate(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        init_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        统一入口：未初始化走 register，已初始化走 track。

        输入：
        - rgb/depth: 当前帧 RGBD。
        - init_mask: 仅第一帧需要。

        输出：
        - pose: 4x4 位姿矩阵。
        """
        if not self._initialized:
            if init_mask is None:
                raise ValueError("首次 estimate 需要提供 init_mask。")
            return self.register(rgb, depth, init_mask)
        return self.track(rgb, depth)

    def visualize_pose(
        self,
        rgb: np.ndarray,
        pose: np.ndarray,
        axis_scale: float = 0.1,
        thickness: int = 3,
    ) -> np.ndarray:
        """
        在彩色图上绘制 3D 包围盒与坐标轴。

        输出：
        - vis: 可视化结果图。
        """
        center_pose = pose @ np.linalg.inv(self.to_origin)
        vis = self.draw_posed_3d_box(
            self.cam_k,
            img=rgb,
            ob_in_cam=center_pose,
            bbox=self.bbox,
        )
        vis = self.draw_xyz_axis(
            vis,
            ob_in_cam=center_pose,
            scale=float(axis_scale),
            K=self.cam_k,
            thickness=int(thickness),
            transparency=0,
            is_input_rgb=True,
        )
        return vis

    def reset(self) -> None:
        """重置内部状态，使下次调用重新走 register。"""
        self._initialized = False


if __name__ == "__main__":
    """
    最小示例（参考 FoundationPose/run_demo.py）：
    1) 从 demo_data 读取 RGBD 与首帧 mask。
    2) 第一帧 register，后续帧 track。
    3) 实时显示位姿可视化。
    """

    root = Path(__file__).resolve().parents[2]
    foundationpose_root = root / "FoundationPose"
    demo_dir = root / "data" / "offline" / "cube"
    init_mask_path = demo_dir / "0_mask.png"

    # 这里直接复用 FoundationPose 官方数据读取器，减少样例噪声。
    # 先加入项目根目录（支持 FoundationPose.datareader 包路径），
    # 再加入 FoundationPose 目录（支持 datareader 直导路径）。
    if str(root) not in sys.path:
        sys.path.append(str(root))
    if str(foundationpose_root) not in sys.path:
        sys.path.append(str(foundationpose_root))

    try:
        datareader_mod = importlib.import_module("FoundationPose.datareader")
    except ModuleNotFoundError:
        datareader_mod = importlib.import_module("datareader")
    reader = datareader_mod.YcbineoatReader(
        video_dir=str(demo_dir), shorter_side=None, zfar=np.inf
    )

    cfg = FoundationPoseConfig(
        mesh_path=str(demo_dir / "mesh" / "textured_simple.obj"),
        cam_k=np.asarray(reader.K),
        est_refine_iter=5,
        track_refine_iter=2,
        apply_scale=1.0,
        force_apply_color=False,
        debug=0,
        debug_dir=str(foundationpose_root / "debug"),
    )

    fp = FoundationPoseEstimator(cfg)

    print("FoundationPose demo running, press q/ESC to quit.")

    for i in range(len(reader.color_files)):
        rgb = reader.get_color(i)
        depth = reader.get_depth(i)

        if i == 0:
            init_mask_img = cv2.imread(
                str(init_mask_path), cv2.IMREAD_GRAYSCALE
            )
            if init_mask_img is None:
                raise RuntimeError(f"初始 mask 读取失败: {init_mask_path}")
            init_mask = init_mask_img.astype(bool)

            pose = fp.register(rgb=rgb, depth=depth, mask=init_mask)
        else:
            pose = fp.track(rgb=rgb, depth=depth)

        vis = fp.visualize_pose(rgb, pose)
        cv2.imshow("FoundationPose", vis[..., ::-1])

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    cv2.destroyAllWindows()
