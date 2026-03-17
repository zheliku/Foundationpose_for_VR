import numpy as np
import torch
from scipy.spatial.transform import Rotation
import pyrealsense2 as rs
from typing import Tuple


def adjust_pose_to_image_point(
        ob_in_cam: torch.Tensor,
        K: np.ndarray,
        x: float = -1.,
        y: float = -1.,
) -> torch.Tensor:
    """
    调整 6D 姿态的平移部分，使其在相机中的投影中心与给定的 2D 图像坐标 (x, y) 对齐。
    这用于根据 2D 跟踪器的结果来校正 3D 姿态的预测位置。

    :param ob_in_cam: 原始的 6D 姿态矩阵 [4,4] 或 [B,4,4] 的 torch.Tensor。
    :param K: 相机内参矩阵 [3,3] 的 torch.Tensor。
    :param x, y: 目标在图像上的 2D 坐标。
    :return: 调整后的新 6D 姿态矩阵。
    """
    device = ob_in_cam.device
    dtype = ob_in_cam.dtype

    is_batched = ob_in_cam.ndim == 3
    if not is_batched:
        ob_in_cam = ob_in_cam.unsqueeze(0)  # 增加一个批次维度 [1, 4, 4]

    B = ob_in_cam.shape[0]
    ob_in_cam_new = torch.eye(4, device=device, dtype=dtype).repeat(B, 1, 1)

    for i in range(B):
        R = ob_in_cam[i, :3, :3]
        t = ob_in_cam[i, :3, 3]

        # 根据 2D 图像点反算出新的相机坐标系下的 tx, ty
        tx, ty = get_pose_xy_from_image_point(ob_in_cam[i], K, x, y)
        # 创建新的平移向量，保持 z 不变
        t_new = torch.tensor([tx, ty, t[2]], device=device, dtype=dtype)

        # 构建新的姿态矩阵
        ob_in_cam_new[i, :3, :3] = R
        ob_in_cam_new[i, :3, 3] = t_new

    return ob_in_cam_new if is_batched else ob_in_cam_new[0]


def get_pose_xy_from_image_point(
        ob_in_cam: torch.Tensor,
        K: np.ndarray,
        x: float = -1.,
        y: float = -1.,
) -> tuple:
    """
    根据给定的图像坐标 (x, y) 和当前的深度 (tz)，反向计算出在相机坐标系中对应的 (tx, ty)。
    这是 `adjust_pose_to_image_point` 的核心计算部分。

    :param ob_in_cam: 4x4 的姿态矩阵。
    :param K: 3x3 的相机内参矩阵。
    :param x, y: 图像上的坐标。
    :return: (tx, ty) 元组，即在相机坐标系下的新 x, y 坐标。
    """
    is_batched = ob_in_cam.ndim == 3
    if is_batched:
        ob_in_cam_new = ob_in_cam[0].cpu()
    else:
        ob_in_cam_new = ob_in_cam.cpu()

    if x == -1. or y == -1.:
        return x, y

    t = ob_in_cam_new[:3, 3]

    # 从内参矩阵中获取焦距和主点坐标
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    tz = t[2]  # 保持深度 tz 不变

    # 根据相机投影公式反向求解 tx 和 ty
    tx = (x - cx) * tz / fx
    ty = (y - cy) * tz / fy

    return tx, ty


def project_3d_to_2d(point_3d_homogeneous, K, ob_in_cam):
    """
    (此函数在当前脚本中未使用)
    将一个 3D 点投影到 2D 图像平面。

    :param point_3d_homogeneous: 齐次坐标表示的 3D 点。
    :param K: 相机内参矩阵。
    :param ob_in_cam: 物体在相机坐标系下的姿态矩阵。
    :return: 投影后的 2D 像素坐标 (u, v)。
    """
    # 将点变换到相机坐标系
    point_cam = ob_in_cam @ point_3d_homogeneous

    # 透视除法，得到归一化图像坐标
    x = point_cam[0] / point_cam[2]
    y = point_cam[1] / point_cam[2]

    # 应用相机内参，得到像素坐标
    u = K[0, 0] * x + K[0, 2]
    v = K[1, 1] * y + K[1, 2]

    return (int(u), int(v))


def get_mat_from_6d_pose_arr(pose_arr):
    """
    将一个 6 维姿态数组 (xyz + 欧拉角) 转换为 4x4 的齐次变换矩阵。

    :param pose_arr: 包含 3 个平移和 3 个欧拉角 (xyz顺序) 的 numpy 数组。
    :return: 4x4 的齐次变换矩阵。
    """
    # 提取平移向量 (xyz)
    xyz = pose_arr[:3]

    # 提取欧拉角
    euler_angles = pose_arr[3:]

    # 从欧拉角生成旋转矩阵
    rotation = Rotation.from_euler('xyz', euler_angles, degrees=False)
    rotation_matrix = rotation.as_matrix()

    # 构建 4x4 的齐次变换矩阵
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = xyz

    return transformation_matrix


def get_6d_pose_arr_from_mat(pose):
    """
    将 4x4 的齐次变换矩阵转换为 6 维姿态数组 (xyz + 欧拉角)。

    :param pose: 4x4 的姿态矩阵 (可以是 torch.Tensor 或 numpy.ndarray)。
    :return: 包含 3 个平移和 3 个欧拉角的 numpy 数组。
    """
    if torch.is_tensor(pose):
        is_batched = pose.ndim == 3
        if is_batched:
            pose_np = pose[0].cpu().numpy()
        else:
            pose_np = pose.cpu().numpy()
    else:
        pose_np = pose

    # 提取平移向量
    xyz = pose_np[:3, 3]
    # 提取旋转矩阵
    rotation_matrix = pose_np[:3, :3]
    # 将旋转矩阵转换为欧拉角
    euler_angles = Rotation.from_matrix(rotation_matrix).as_euler('xyz', degrees=False)
    return np.r_[xyz, euler_angles]


def rs_intrinsics_to_cv(camera_intrinsics: rs.intrinsics) -> Tuple[np.ndarray, np.ndarray]:
    """
    将 RealSense 的内参转换为 OpenCV 相机矩阵与畸变系数。
    RealSense 提供的 coeffs 顺序一般是 [k1, k2, p1, p2, k3, k4, k5, k6]（按模型不同）
    OpenCV 常用前5项 Brown-Conrady: k1, k2, p1, p2, k3
    """
    fx, fy = camera_intrinsics.fx, camera_intrinsics.fy
    cx, cy = camera_intrinsics.ppx, camera_intrinsics.ppy
    cam_mtx = np.array([[fx, 0, cx],
                        [0, fy, cy],
                        [0, 0, 1]], dtype=np.float32)
    coeffs = list(camera_intrinsics.coeffs)
    if len(coeffs) < 5:
        # 没有畸变或数量不足时，退化为零畸变
        dist = np.zeros((1, 5), dtype=np.float32)
    else:
        dist = np.array([coeffs[:5]], dtype=np.float32)
    return cam_mtx, dist