import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R


def get_transform_matrix(rvec, tvec):
    """将 rvec, tvec 转换为 4x4 齐次变换矩阵"""
    R_mat, _ = cv2.Rodrigues(np.array(rvec))
    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = np.array(tvec).flatten()
    return T


def calculate_relative_transform(T_board_src, T_board_tgt):
    """计算从 Source 到 Target 的变换: T_src_tgt = T_board_tgt * inv(T_board_src)"""
    T_src_inv = np.linalg.inv(T_board_src)
    T_src_tgt = np.dot(T_board_tgt, T_src_inv)
    return T_src_tgt


def matrix_to_rvec_tvec(T):
    """将 4x4 矩阵转回 rvec, tvec"""
    rvec, _ = cv2.Rodrigues(T[:3, :3])
    tvec = T[:3, 3]
    return rvec.flatten(), tvec.flatten()


# --- 输入数据 ---
# Set 1
rs_r1 = [-0.297774, -0.071522, -0.032113]
rs_t1 = [-0.040250, -0.034150, 0.550051]
q3l_r1 = [-0.318553, -0.088436, -0.027616]
q3l_t1 = [-0.049956, -0.129225, 0.509224]
q3r_r1 = [-0.315115, -0.058828, -0.027530]
q3r_t1 = [-0.102096, -0.130532, 0.510911]

# Set 2
rs_r2 = [-0.093023, 0.200177, 0.015920]
rs_t2 = [0.063406, -0.030088, 0.318489]
q3l_r2 = [-0.063013, 0.172031, 0.018283]
q3l_t2 = [0.057890, -0.125904, 0.282339]
q3r_r2 = [-0.058372, 0.205429, 0.017665]
q3r_t2 = [0.001840, -0.127078, 0.281757]


# --- 计算 ---
def process_pair(name, rs_r, rs_t, q3_r, q3_t):
    T_board_rs = get_transform_matrix(rs_r, rs_t)
    T_board_q3 = get_transform_matrix(q3_r, q3_t)
    T_rs_q3 = calculate_relative_transform(T_board_rs, T_board_q3)
    return T_rs_q3


# RealSense -> Quest 3 Left
T_rs_q3l_1 = process_pair("Set1_L", rs_r1, rs_t1, q3l_r1, q3l_t1)
T_rs_q3l_2 = process_pair("Set2_L", rs_r2, rs_t2, q3l_r2, q3l_t2)

# RealSense -> Quest 3 Right
T_rs_q3r_1 = process_pair("Set1_R", rs_r1, rs_t1, q3r_r1, q3r_t1)
T_rs_q3r_2 = process_pair("Set2_R", rs_r2, rs_t2, q3r_r2, q3r_t2)


# --- 结果分析与平均 ---
def analyze_results(T1, T2, label):
    # 平均平移
    t1 = T1[:3, 3]
    t2 = T2[:3, 3]
    t_avg = (t1 + t2) / 2
    t_diff_mm = np.linalg.norm(t1 - t2) * 1000

    # 平均旋转 (使用四元数插值或简单的平均 rvec，这里只要差异小，取平均矩阵转rvec即可)
    # 计算角度差
    R1 = T1[:3, :3]
    R2 = T2[:3, :3]
    R_diff = np.dot(R1, R2.T)
    angle_diff_rad = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1.0, 1.0))
    angle_diff_deg = np.degrees(angle_diff_rad)

    # 最终结果 (取Set1和Set2的几何平均，近似为直接平均tvec和rvec)
    # 更严谨的做法是平均四元数，但为此脚本简洁性，我们取 Set 2 (近距离通常更准) 或 平均值
    # 这里我们输出平均值
    rvec_avg, tvec_avg = matrix_to_rvec_tvec((T1 + T2) / 2)  # 线性近似

    print(f"--- {label} 分析 ---")
    print(f"Set 1 tvec: {t1}")
    print(f"Set 2 tvec: {t2}")
    print(f"平移误差 (Diff): {t_diff_mm:.2f} mm")
    print(f"旋转误差 (Diff): {angle_diff_deg:.3f} degrees")
    print(f"建议最终 tvec: {tvec_avg.tolist()}")
    print(f"建议最终 rvec: {rvec_avg.tolist()}")
    return rvec_avg, tvec_avg


print("CALCULATION RESULTS:\n")
final_l_rvec, final_l_tvec = analyze_results(
    T_rs_q3l_1, T_rs_q3l_2, "RealSense -> Q3 Left"
)
print("\n")
final_r_rvec, final_r_tvec = analyze_results(
    T_rs_q3r_1, T_rs_q3r_2, "RealSense -> Q3 Right"
)
