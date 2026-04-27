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
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import trimesh


class FoundationPoseEstimator:
    """FoundationPose 估计器：支持 register + track。"""

    # 输入配置。
    mesh_path: str = ""  # 目标 mesh 路径。
    est_refine_iter: int = 10  # register 阶段迭代次数。
    track_refine_iter: int = 5  # track 阶段迭代次数。
    apply_scale: float = 1.0  # mesh 缩放比例。
    force_apply_color: bool = False  # 是否强制给 mesh 上纯色。
    apply_color: list[int] | None = None  # 纯色 RGB（可选）。
    symmetry_tfs: np.ndarray | None = None  # 对称变换集合（可选）。
    debug: int = 0  # 调试等级。
    debug_dir: str | None = None  # 调试输出目录。
    cam_k: np.ndarray  # 相机内参 K（__init__ 中归一化后固定）。

    # 路径与动态导入符号。
    foundationpose_root: Path | None = None  # FoundationPose 根目录。
    ScorePredictor: Any = None  # 评分网络类。
    PoseRefinePredictor: Any = None  # 位姿精修网络类。
    dr: Any = None  # 渲染上下文模块。
    FoundationPose: Any = None  # FoundationPose 主类。
    trimesh_add_pure_colored_texture: Any = None  # mesh 纯色贴图函数。
    draw_posed_3d_box: Any = None  # 3D 包围盒绘制函数。
    draw_xyz_axis: Any = None  # 坐标轴绘制函数。

    # 运行时对象与状态。
    mesh: Any = None  # 预处理后的 mesh。
    to_origin: np.ndarray  # mesh 到中心坐标的变换（__init__ 中计算）。
    bbox: np.ndarray  # 目标包围盒（__init__ 中计算）。
    estimator: Any  # FoundationPose 实例（__init__ 中创建）。
    _initialized: bool = False  # 是否完成首帧注册。

    def __init__(
        self,
        mesh_path: str,
        cam_k: np.ndarray,
        est_refine_iter: int = 10,
        track_refine_iter: int = 5,
        apply_scale: float = 1.0,
        force_apply_color: bool = False,
        apply_color: list[int] | None = None,
        symmetry_tfs: np.ndarray | None = None,
        debug: int = 0,
        debug_dir: str | None = None,
    ) -> None:
        """
        初始化 FoundationPose 估计器。

        参数：
        - mesh_path: 目标 mesh 路径。
        - cam_k: 相机内参矩阵。
        - est_refine_iter: register 阶段迭代次数。
        - track_refine_iter: track 阶段迭代次数。
        - apply_scale: mesh 缩放比例。
        - force_apply_color/apply_color: 纯色贴图配置。
        - symmetry_tfs: 对称变换集合。
        - debug/debug_dir: 调试级别与输出目录。

        初始化流程：
        1. 保存配置并标准化相机内参。
        2. 配置工程路径并动态导入 FoundationPose 符号。
        3. 加载并预处理 mesh。
        4. 构建 FoundationPose 推理实例。
        """
        self.mesh_path = str(mesh_path)
        self.est_refine_iter = int(est_refine_iter)
        self.track_refine_iter = int(track_refine_iter)
        self.apply_scale = float(apply_scale)
        self.force_apply_color = bool(force_apply_color)
        self.apply_color = apply_color
        self.symmetry_tfs = symmetry_tfs
        self.debug = int(debug)
        self.debug_dir = debug_dir

        # 标准化并缓存相机内参，后续每帧直接复用。
        # 这里使用 float64：FoundationPose 内部部分几何流程会把 mesh 顶点处理为 double，
        # 若 K 为 float32，注册阶段在矩阵乘法处会触发 float/double dtype 冲突。
        self.cam_k = np.asarray(cam_k, dtype=np.float64).reshape(3, 3)

        # 默认颜色配置。
        if self.apply_color is None:
            self.apply_color = [0, 159, 237]

        # 补充项目路径，确保可导入 FoundationPose 包。
        project_root = Path(__file__).resolve().parents[2]
        self.foundationpose_root = project_root / "FoundationPose"
        if str(project_root) not in sys.path:
            sys.path.append(str(project_root))
        if str(self.foundationpose_root) not in sys.path:
            sys.path.append(str(self.foundationpose_root))

        # 动态导入 FoundationPose；并临时绑定 Utils，避免与 FFS 的同名模块冲突。
        try:
            utils_mod = importlib.import_module("FoundationPose.Utils")
        except ModuleNotFoundError:
            utils_mod = importlib.import_module("Utils")

        old_utils_module = sys.modules.get("Utils")
        sys.modules["Utils"] = utils_mod
        try:
            try:
                est_mod = importlib.import_module("FoundationPose.estimater")
            except ModuleNotFoundError:
                est_mod = importlib.import_module("estimater")
        finally:
            if old_utils_module is None:
                sys.modules.pop("Utils", None)
            else:
                sys.modules["Utils"] = old_utils_module

        # 不同版本导出位置不一致：优先用 estimater，其次回退到 Utils。
        def _resolve_symbol(name: str) -> Any:
            if hasattr(est_mod, name):
                return getattr(est_mod, name)
            if hasattr(utils_mod, name):
                return getattr(utils_mod, name)
            raise RuntimeError(f"FoundationPose 符号缺失: {name}")

        self.ScorePredictor = _resolve_symbol("ScorePredictor")
        self.PoseRefinePredictor = _resolve_symbol("PoseRefinePredictor")
        self.dr = _resolve_symbol("dr")
        self.FoundationPose = _resolve_symbol("FoundationPose")
        self.trimesh_add_pure_colored_texture = _resolve_symbol(
            "trimesh_add_pure_colored_texture"
        )
        self.draw_posed_3d_box = _resolve_symbol("draw_posed_3d_box")
        self.draw_xyz_axis = _resolve_symbol("draw_xyz_axis")

        # 加载并预处理 mesh。
        loaded_mesh = trimesh.load(self.mesh_path)
        if isinstance(loaded_mesh, trimesh.Scene):
            loaded_mesh = loaded_mesh.dump(concatenate=True)
        # trimesh 在静态类型上较宽泛，这里转 Any 以便后续直接访问 vertices 等属性。
        self.mesh = cast(Any, loaded_mesh)

        self.mesh.apply_scale(float(self.apply_scale))

        if bool(self.force_apply_color):
            self.mesh = self.trimesh_add_pure_colored_texture(
                self.mesh,
                color=np.array(self.apply_color),
                resolution=10,
            )

        # 计算包围盒与中心修正矩阵，供可视化使用。
        self.to_origin, extents = trimesh.bounds.oriented_bounds(self.mesh)
        self.bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

        # FoundationPose 内部会对 debug_dir 调用 os.makedirs。
        # 因此这里必须保证传入的是有效字符串路径，不能为 None。
        effective_debug_dir = self.debug_dir
        if effective_debug_dir is None or str(effective_debug_dir).strip() == "":
            effective_debug_dir = str(self.foundationpose_root / "debug" / "api")

        # 初始化 FoundationPose 网络与渲染上下文。
        scorer = self.ScorePredictor()
        refiner = self.PoseRefinePredictor()
        glctx = self.dr.RasterizeCudaContext()

        self.estimator = self.FoundationPose(
            model_pts=self.mesh.vertices,
            model_normals=self.mesh.vertex_normals,
            symmetry_tfs=self.symmetry_tfs,
            mesh=self.mesh,
            scorer=scorer,
            refiner=refiner,
            glctx=glctx,
            debug_dir=effective_debug_dir,
            debug=int(self.debug),
        )

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
            iteration=int(self.est_refine_iter),
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
            iteration=int(self.track_refine_iter),
        )
        return np.asarray(pose).reshape(4, 4)

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
        if hasattr(self.estimator, "pose_last"):
            try:
                delattr(self.estimator, "pose_last")
            except Exception:
                self.estimator.pose_last = None


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

    fp = FoundationPoseEstimator(
        mesh_path=str(demo_dir / "mesh" / "textured_simple.obj"),
        cam_k=np.asarray(reader.K),
        est_refine_iter=5,
        track_refine_iter=2,
        apply_scale=1.0,
        force_apply_color=False,
        debug=0,
        debug_dir=str(foundationpose_root / "debug"),
    )

    print("FoundationPose demo running, press q/ESC to quit.")

    for i in range(len(reader.color_files)):
        rgb = reader.get_color(i)
        depth = reader.get_depth(i)

        if i == 0:
            init_mask_img = cv2.imread(str(init_mask_path), cv2.IMREAD_GRAYSCALE)
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
