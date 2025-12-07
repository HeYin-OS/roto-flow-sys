from typing import List, Any
import cv2
import numba

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from utils.yaml_reader import YamlUtil
from utils.edge_snapping_utils import (
    create_gaussian_kernel,
    create_fdog_kernel,
    rgb_np_to_gray_tensor,
    pack_jagged_list_to_array,
    slice_candidate_group_by_index,
)


def compute_all_candidates(images_rgb_nhwc_uint8: np.ndarray):
    """
    提取所有候选边缘点
    
    修改说明：
    - 放宽局部最大值条件：从严格大于改为大于等于，允许相等的情况
    - 添加小的容差（0.01），允许接近局部最大值的点
    - 这样可以获得更多的候选点
    """
    out = []
    for i_frame in tqdm(range(images_rgb_nhwc_uint8.shape[0]), desc="Computing candidate points", unit=" image(s)"):
        image_gray_hw_uint8 = cv2.cvtColor(images_rgb_nhwc_uint8[i_frame], cv2.COLOR_RGB2GRAY)

        # gradient magnitude
        gx = cv2.Sobel(image_gray_hw_uint8, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_DEFAULT)
        gy = cv2.Sobel(image_gray_hw_uint8, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_DEFAULT)
        mag = cv2.magnitude(gx, gy)

        # normalization
        mag_norm = cv2.normalize(mag, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        # neighbor max in 3*3 window
        k = np.ones((3, 3), np.uint8)
        k[1, 1] = 0
        nbr_max = cv2.dilate(mag_norm, k)

        # local maximum - 放宽条件以获得更多候选点
        # 原条件: mag_norm > nbr_max
        # 新条件: mag_norm >= nbr_max - tolerance，允许接近局部最大值的点
        tolerance = 0.01  # 小的容差，允许接近局部最大值的点
        local_max = (mag_norm >= nbr_max - tolerance) & (mag_norm >= float(EdgeSnappingConfig.theta))

        # salient_img = local_max.astype(np.uint8) * 255
        # cv2.imwrite(f"debug/salient_points_on_frame_{i}.jpg", salient_img)

        # all candidate points on this frame
        ys, xs = np.nonzero(local_max)
        out.append(np.stack([xs, ys], axis=1).astype(np.float32))
    return out


class EdgeSnappingConfig:
    theta = None
    alpha = None
    beta = None
    beta_shift = None
    sigma_c = None
    sigma_s = None
    sigma_m = None
    rho = None
    X_MAX = None
    Y_MAX = None
    r_s = None
    candidate_num = None
    sampling_num = None
    average_weight_threshold = None
    lambda_shape = None  # 形状约束项的权重
    lambda_topology = None  # 拓扑顺序项的权重

    fdog_kernel = None
    gaussian_kernel = None

    isConfigInit: bool = False

    @staticmethod
    def load(config_yaml_path='config/snapping_init.yaml'):
        if EdgeSnappingConfig.isConfigInit:
            return

        settings = YamlUtil.read(config_yaml_path)
        s = settings['snapping']

        EdgeSnappingConfig.theta = s['theta']
        EdgeSnappingConfig.alpha = s['alpha']
        EdgeSnappingConfig.beta = s['beta']
        EdgeSnappingConfig.beta_shift = s['beta_shift']
        EdgeSnappingConfig.sigma_c = s['sigma_c']
        EdgeSnappingConfig.sigma_s = s['sigma_s']
        EdgeSnappingConfig.sigma_m = s['sigma_m']
        EdgeSnappingConfig.rho = s['rho']
        EdgeSnappingConfig.X_MAX = s['x']
        EdgeSnappingConfig.Y_MAX = s['y']
        EdgeSnappingConfig.r_s = s['r_s']
        EdgeSnappingConfig.candidate_num = s['candidate_num']
        EdgeSnappingConfig.sampling_num = s['sampling_num']
        EdgeSnappingConfig.average_weight_threshold = s['average_weight_threshold']
        EdgeSnappingConfig.lambda_shape = s.get('lambda_shape', 0.05)  # 默认值0.05
        EdgeSnappingConfig.lambda_topology = s.get('lambda_topology', 0.03)  # 默认值0.03

        EdgeSnappingConfig.isConfigInit = True

        EdgeSnappingConfig.fdog_kernel = create_fdog_kernel(
            2 * EdgeSnappingConfig.X_MAX + 1,
            EdgeSnappingConfig.sigma_c,
            EdgeSnappingConfig.sigma_s,
            EdgeSnappingConfig.rho,
            1
        )
        EdgeSnappingConfig.gaussian_kernel = create_gaussian_kernel(
            2 * EdgeSnappingConfig.Y_MAX + 1,
            EdgeSnappingConfig.sigma_m,
            0
        )


def local_snapping(stroke: np.ndarray,
                   image_rgb_hwc: np.ndarray,
                   points_stroke_candidate: List[np.ndarray]):
    """
    局部优化步骤：将用户笔画吸附到图像边缘特征上。
    
    实现论文 Section 3.1 的算法：
    1. 构建链状图 G = (V, E)，其中 V = U_0^N Q_i（Q_0 为虚拟起始点，通过将第一组能量设为0实现）
    2. 对每条边 e = (q_{i,k}, q_{i+1,k'}) 计算权重 w_e（Equation 3）
    3. 使用动态规划找最短路径（对应论文中的 s'_1）
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.backends.cudnn.benchmark = True

    # convert np gray image [H, W] to tensor [B=1, C=1, H, W]
    H = image_rgb_hwc.shape[0]
    W = image_rgb_hwc.shape[1]
    image_tensor_gray_chw = rgb_np_to_gray_tensor(device, image_rgb_hwc)

    # convert jagged array to flatten array with index pointer page
    points_stroke_candidate = candidate_point_sets_defense(points_stroke_candidate, stroke)

    points_stroke_candidate_flatten, flatten_index_ptr = pack_jagged_list_to_array(points_stroke_candidate)
    n_candidate_points = flatten_index_ptr[-1]

    # order to xy and get point number of stroke
    stroke_len = stroke.shape[0]
    stroke = stroke.astype(np.float32)

    # ready for dp
    # energy -> accumulated energy for each candidate point
    # prev -> the best previous candidate point idx
    energy = np.full(n_candidate_points, np.inf, dtype=np.float32)
    prev = np.full(n_candidate_points, -1, dtype=np.int32)

    # 虚拟起始点 Q_0 的处理：将第一组候选点的能量设为0（等价于添加虚拟起始点）
    # 论文中 V = U_0^N Q_i，其中 Q_0 = {q_{0,0}} 是虚拟点
    energy[flatten_index_ptr[0]: flatten_index_ptr[1]] = 0.0

    for i_group in range(stroke_len - 1):
        # get a candidate group of i, i+1 from flatten slice
        Q_i, Q_j = slice_candidate_group_by_index(points_stroke_candidate_flatten, flatten_index_ptr, i_group)

        # print(f"Qi_xy: {Qi_xy.shape[0]} * Qj_xy: {Qj_xy.shape[0]} = {Qi_xy.shape[0] * Qj_xy.shape[0]}")

        # weights between each two points in two groups
        p_i, p_j = stroke[i_group], stroke[i_group + 1]
        
        # 获取前一组候选点（用于计算形状约束和拓扑顺序）
        prev_q_i = None
        if i_group > 0:
            prev_start, prev_end = flatten_index_ptr[i_group - 1], flatten_index_ptr[i_group]
            prev_q_i = points_stroke_candidate_flatten[prev_start:prev_end]
        
        weights = compute_weights(H, W,
                                  p_i, p_j,
                                  Q_i, Q_j,
                                  image_tensor_gray_chw,
                                  device,
                                  prev_q_i=prev_q_i)  # shape: [K_i, K_j]

        # print(f"weights.shape: {weights.shape} ")

        dp_energy_iteration(i_group, flatten_index_ptr, energy, prev, weights)

    return pick_best_path(last_start=flatten_index_ptr[-2],
                          last_end=flatten_index_ptr[-1],
                          energy=energy,
                          prev=prev,
                          candidates_flatten=points_stroke_candidate_flatten,
                          stroke=stroke,
                          average_weight_standard=EdgeSnappingConfig.average_weight_threshold,
                          average_distance_standard=EdgeSnappingConfig.r_s / 4.0)


def candidate_point_sets_defense(points_candidate: list[np.ndarray], stroke: np.ndarray) -> list[np.ndarray]:
    fixed_candidates = []
    for t, cand in enumerate(points_candidate):
        if cand is None or (isinstance(cand, np.ndarray) and cand.size == 0):
            p = stroke[t].astype(np.float32)
            fixed_candidates.append(np.array([[p[0], p[1]]], dtype=np.float32))
        else:
            fixed_candidates.append(cand)
    return fixed_candidates


def pick_best_path(last_start,
                   last_end,
                   energy: np.ndarray,
                   prev: np.ndarray,
                   candidates_flatten: np.ndarray,
                   stroke: np.ndarray,
                   average_weight_standard,
                   average_distance_standard):
    # TODO: add checks of two conditions based on the paper
    stroke_len = stroke.shape[0]

    while True:
        # last point index of underlying candidate stroke
        best_idx = np.argmin(energy[last_start:last_end]) + last_start
        curr_idx = best_idx

        avg_en = energy[best_idx] / (stroke_len - 1)

        # use backtrack to find all points of candidate stroke
        best_path_indices = []
        while best_idx != -1:
            best_path_indices.insert(0, best_idx)
            best_idx = prev[best_idx]

        # use slice to get the candidate stroke
        candidate_stroke_xy = candidates_flatten[best_path_indices]

        avg_dist = np.linalg.norm(candidate_stroke_xy - stroke, axis=1).mean()

        # print(f"avg en = {avg_en} (std: <{average_weight_standard}), avg dist = {avg_dist} (std: >{average_distance_standard})")

        if avg_dist < average_distance_standard:
            energy[curr_idx] = np.inf
            if np.isinf(energy[last_start:last_end]).all():
                return candidate_stroke_xy
            continue

        if avg_en > average_weight_standard:
            return candidate_stroke_xy

        return candidate_stroke_xy


@numba.njit
def dp_energy_iteration(i: int, flatten_index_ptr: np.ndarray,
                        energy: np.ndarray,
                        prev: np.ndarray,
                        weights: np.ndarray):
    start_i, end_i = flatten_index_ptr[i], flatten_index_ptr[i + 1]
    start_j, end_j = flatten_index_ptr[i + 1], flatten_index_ptr[i + 2]

    for idx_j in range(end_j - start_j):
        best_prev = -1
        best_energy = np.inf

        for idx_i in range(end_i - start_i):
            bi_energy = energy[start_i + idx_i] + weights[idx_i, idx_j]

            if bi_energy < best_energy:
                best_prev = start_i + idx_i
                best_energy = bi_energy

        energy[start_j + idx_j] = best_energy
        prev[start_j + idx_j] = best_prev


def compute_weights(H: int, W: int,
                    p_i: np.ndarray, p_j: np.ndarray,
                    Q_i: np.ndarray, Q_j: np.ndarray,
                    image_gray_chw: Tensor,
                    device,
                    prev_q_i: np.ndarray = None):
    """
    计算边权重 w_e（论文 Equation 3 + 形状约束 + 拓扑顺序）。
    
    对于每条边 e = (q_{i,k}, q_{i+1,k'})：
    - 计算中点 m = (q_{i,k} + q_{i+1,k'}) / 2
    - 计算方向 v = normalize(q_{i+1,k'} - q_{i,k})
    - 计算垂直方向 u = rotate_90(v)
    - 应用 FDoG 滤波器 H(m,v)（论文 Equation 1）：先沿 u 方向（x）做 DoG，再沿 v 方向（y）做高斯平滑
    - 转换为 H̃(m,v)（论文 Equation 2）
    - 计算 deform term：||(p_{i+1} - p_i) - (q_{i+1} - q_i)||^2 / r_s^2
    - 计算 shape constraint term：形状约束项（距离和角度）
    - 计算 topology term：拓扑顺序项（避免交叉和方向一致性）
    - 最终权重：w_e = deform_term + α * H̃(m,v) + λ_shape * shape_term + λ_topology * topology_term
    
    Args:
        prev_q_i: 前一个点的候选位置（用于计算形状约束和拓扑顺序），形状为(K_{i-1}, 2)
    """
    theta_flatten_gpu = compute_affine_theta_vectorized(Q_i,
                                                        Q_j,
                                                        H, W).to(device)  # shape: [K_{i} * K_{i+1}, 2, 3]

    grid_gpu = torch.nn.functional.affine_grid(theta_flatten_gpu,
                                               size=[theta_flatten_gpu.shape[0],
                                                     1,
                                                     2 * EdgeSnappingConfig.Y_MAX + 1,
                                                     2 * EdgeSnappingConfig.X_MAX + 1],
                                               align_corners=False).to(device)  # shape: [K_{i} * K_{i+1}, 2*Y+1, 2*X+1, 2]

    image_affined = (torch.nn.functional.grid_sample(image_gray_chw.expand(grid_gpu.shape[0], -1, -1, -1),
                                                     grid_gpu,
                                                     mode='bilinear',
                                                     padding_mode='zeros',
                                                     align_corners=False)
                     .squeeze(1)
                     .reshape(Q_i.shape[0],
                              Q_j.shape[0],
                              2 * EdgeSnappingConfig.Y_MAX + 1,
                              2 * EdgeSnappingConfig.X_MAX + 1)
                     .cpu().numpy())  # change to nparray such that do tensor dot afterward

    # FDoG 滤波器实现（论文 Equation 1）：
    # H(m,v) = ∫_{-Y}^{Y} G_σm(y) ∫_{-X}^{X} I(l(x,y)) f(x) dx dy
    # 其中 f(x) = G_σc(x) - ρG_σs(x) 是沿 u 方向（x）的 DoG
    # 积分顺序：先对 x（u方向，最后一维）做 DoG，再对 y（v方向，倒数第二维）做高斯平滑
    res_dot_on_x = np.tensordot(image_affined,
                                EdgeSnappingConfig.fdog_kernel.squeeze(),
                                axes=([-1], [0]))  # 沿 u 方向（x）做 DoG

    res_dot_on_x_y = np.tensordot(res_dot_on_x,
                                  EdgeSnappingConfig.gaussian_kernel.squeeze(),
                                  axes=([-1], [0])).squeeze()  # 沿 v 方向（y）做高斯平滑

    # 转换为 H̃(m,v)（论文 Equation 2）：
    # H̃(m,v) = 1 + tanh(H(m,v)) if H(m,v) < 0, else 1
    tilde_H_response = np.where(res_dot_on_x_y < 0, 1.0 + np.tanh(res_dot_on_x_y), 1.0)

    # print(f"temp.shape = {res_dot_on_x.shape}, res_dot_on_x_y.shape = {res_dot_on_x_y.shape}, tilde_H.shape = {tilde_H_response.shape}")

    # deform term (based on paper)
    r_s_square = float(EdgeSnappingConfig.r_s) ** 2

    p_diff = (p_j - p_i).astype(np.float32)
    q_diff = Q_j.astype(np.float32)[None, :, :] - Q_i.astype(np.float32)[:, None, :]
    diff = p_diff.reshape(1, 1, 2) - q_diff
    deform_term = np.sum(diff * diff, axis=-1) / r_s_square

    # shift term (not on paper)
    shift_i = np.sum((Q_i.astype(np.float32) - p_i[None, :]) ** 2, axis=-1)  # [K_i]
    shift_j = np.sum((Q_j.astype(np.float32) - p_j[None, :]) ** 2, axis=-1)  # [K_{i+1}]
    shift_term = (shift_i[:, None] + shift_j[None, :]) / (2.0 * r_s_square)  # [K_i, K_{i+1}]

    # 基础权重
    weights = deform_term + EdgeSnappingConfig.alpha * tilde_H_response
    
    # 形状约束项（Shape Constraint Term）
    # 约束相邻点之间的距离和角度，保持笔画的形状一致性
    if EdgeSnappingConfig.lambda_shape > 0:
        shape_term = compute_shape_constraint_term(
            p_i, p_j, Q_i, Q_j, prev_q_i, r_s_square
        )
        weights = weights + EdgeSnappingConfig.lambda_shape * shape_term
    
    # 拓扑顺序项（Topology Term）
    # 约束点的顺序，避免交叉和方向反转
    if EdgeSnappingConfig.lambda_topology > 0:
        topology_term = compute_topology_term(
            p_i, p_j, Q_i, Q_j, prev_q_i, r_s_square
        )
        weights = weights + EdgeSnappingConfig.lambda_topology * topology_term

    # print(f"p_diff.shape = {p_diff.shape}")
    # print(f"q_diff.shape = {q_diff.shape}")
    # print(f"diff.shape = {diff.shape}")
    # print(f"square_norm.shape = {square_norm.shape}")
    # print(f"weights.shape = {weights.shape}")

    return weights




def compute_shape_constraint_term(p_i: np.ndarray, p_j: np.ndarray,
                                  Q_i: np.ndarray, Q_j: np.ndarray,
                                  prev_q_i: np.ndarray = None,
                                  r_s_square: float = 400.0) -> np.ndarray:
    """
    计算形状约束项（Shape Constraint Term）
    
    约束相邻点之间的距离和角度，保持笔画的形状一致性
    
    Args:
        p_i: 原始笔画点i，形状为(2,)
        p_j: 原始笔画点j，形状为(2,)
        Q_i: 候选点组i，形状为(K_i, 2)
        Q_j: 候选点组j，形状为(K_j, 2)
        prev_q_i: 前一个点的候选位置（可选），形状为(K_{i-1}, 2)
        r_s_square: r_s的平方，用于归一化
    
    Returns:
        形状约束项，形状为(K_i, K_j)
    """
    # 计算原始边的长度
    p_diff = p_j - p_i  # (2,)
    p_length = np.linalg.norm(p_diff)
    if p_length < 1e-6:
        p_length = 1e-6
    
    # 计算候选边的长度
    q_diff = Q_j[None, :, :] - Q_i[:, None, :]  # (K_i, K_j, 2)
    q_length = np.linalg.norm(q_diff, axis=-1)  # (K_i, K_j)
    
    # 距离约束：保持相邻点之间的距离接近原始距离
    # 惩罚与原始距离差异较大的情况
    length_diff = np.abs(q_length - p_length) / p_length
    length_term = length_diff * length_diff  # (K_i, K_j)
    
    # 角度约束：如果有前一个点，约束角度变化
    angle_term = np.zeros((Q_i.shape[0], Q_j.shape[0]), dtype=np.float32)
    
    if prev_q_i is not None and len(prev_q_i) > 0:
        # 使用前一组候选点的平均位置作为参考（简化处理）
        prev_q_mean = np.mean(prev_q_i, axis=0)  # (2,)
        
        # 计算前一个边到当前边的角度变化
        # 前一个边：prev_q_mean -> Q_i
        prev_edge = Q_i - prev_q_mean[None, :]  # (K_i, 2)
        prev_edge_length = np.linalg.norm(prev_edge, axis=-1, keepdims=True)  # (K_i, 1)
        prev_edge_normalized = prev_edge / (prev_edge_length + 1e-6)  # (K_i, 2)
        
        # 当前边的方向
        curr_edge_normalized = q_diff / (q_length[:, :, None] + 1e-6)  # (K_i, K_j, 2)
        
        # 计算角度变化（使用点积）
        cos_angle = np.sum(prev_edge_normalized[:, None, :] * curr_edge_normalized, axis=-1)  # (K_i, K_j)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        # 角度变化惩罚（角度变化越大，惩罚越大）
        angle_diff = 1.0 - cos_angle  # 0表示角度相同，2表示完全相反
        angle_term = angle_diff * angle_diff
    
    # 综合形状约束项
    shape_term = length_term + 0.5 * angle_term
    
    return shape_term


def compute_topology_term(p_i: np.ndarray, p_j: np.ndarray,
                         Q_i: np.ndarray, Q_j: np.ndarray,
                         prev_q_i: np.ndarray = None,
                         r_s_square: float = 400.0) -> np.ndarray:
    """
    计算拓扑顺序项（Topology Term）
    
    约束点的顺序，避免交叉和方向反转
    
    Args:
        p_i: 原始笔画点i，形状为(2,)
        p_j: 原始笔画点j，形状为(2,)
        Q_i: 候选点组i，形状为(K_i, 2)
        Q_j: 候选点组j，形状为(K_j, 2)
        prev_q_i: 前一个点的候选位置（可选），形状为(K_{i-1}, 2)
        r_s_square: r_s的平方，用于归一化
    
    Returns:
        拓扑顺序项，形状为(K_i, K_j)
    """
    topology_term = np.zeros((Q_i.shape[0], Q_j.shape[0]), dtype=np.float32)
    
    # 计算原始边的方向
    p_diff = p_j - p_i  # (2,)
    p_length = np.linalg.norm(p_diff)
    if p_length < 1e-6:
        return topology_term
    p_direction = p_diff / p_length  # (2,)
    
    # 计算候选边的方向
    q_diff = Q_j[None, :, :] - Q_i[:, None, :]  # (K_i, K_j, 2)
    q_length = np.linalg.norm(q_diff, axis=-1, keepdims=True)  # (K_i, K_j, 1)
    q_direction = q_diff / (q_length + 1e-6)  # (K_i, K_j, 2)
    
    # 方向一致性约束：候选边的方向应该与原始边的方向一致
    # 使用点积来衡量方向一致性
    direction_consistency = np.sum(p_direction[None, None, :] * q_direction, axis=-1)  # (K_i, K_j)
    # 方向一致性项：方向越不一致，惩罚越大
    direction_term = (1.0 - direction_consistency) * (1.0 - direction_consistency)  # (K_i, K_j)
    
    # 交叉检测：如果有前一个点，检测边是否交叉
    cross_term = np.zeros((Q_i.shape[0], Q_j.shape[0]), dtype=np.float32)
    
    if prev_q_i is not None and len(prev_q_i) > 0:
        # 使用前一组候选点的平均位置作为参考
        prev_q_mean = np.mean(prev_q_i, axis=0)  # (2,)
        
        # 前一个边：prev_q_mean -> Q_i
        prev_edge = Q_i - prev_q_mean[None, :]  # (K_i, 2)
        prev_edge_length = np.linalg.norm(prev_edge, axis=-1, keepdims=True)  # (K_i, 1)
        prev_edge_norm = prev_edge / (prev_edge_length + 1e-6)  # (K_i, 2)
        
        # 扩展 prev_edge_norm 以匹配 q_direction 的形状 (K_i, K_j, 2)
        prev_edge_norm_expanded = prev_edge_norm[:, None, :]  # (K_i, 1, 2)
        
        # 计算转向角度（叉积的符号表示转向方向）
        # 如果转向角度过大（接近180度），说明可能交叉
        cross_product = (prev_edge_norm_expanded[:, :, 0:1] * q_direction[:, :, 1:2] - 
                       prev_edge_norm_expanded[:, :, 1:2] * q_direction[:, :, 0:1])  # (K_i, K_j, 1)
        cross_product = cross_product.squeeze(-1)  # (K_i, K_j)
        
        # 转向角度过大时给予惩罚
        # 使用转向角度的绝对值
        turn_angle = np.abs(cross_product)
        cross_term = np.where(turn_angle > 0.8, turn_angle * turn_angle, 0.0)  # 阈值0.8约等于90度
    
    # 顺序约束：确保点按顺序排列（Q_j应该在Q_i的前进方向上）
    # 计算Q_j相对于Q_i的位置是否在前进方向上
    q_relative_dir = q_direction  # 已经归一化
    
    # 检查是否在前进方向上（点积应该为正）
    forward_check = np.sum(p_direction[None, None, :] * q_relative_dir, axis=-1)  # (K_i, K_j)
    # 如果不在前进方向（点积为负），给予惩罚
    order_term = np.where(forward_check < 0, forward_check * forward_check, 0.0)  # (K_i, K_j)
    
    # 综合拓扑顺序项
    topology_term = direction_term + 0.3 * cross_term + 0.2 * order_term
    
    return topology_term


def compute_affine_theta_vectorized(Q_i: np.ndarray,
                                    Q_j: np.ndarray,
                                    H: int, W: int,
                                    eps: np.float32 = 1e-6) -> Tensor:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Q_i = torch.from_numpy(Q_i.copy().astype(np.float32)).to(device)
    Q_j = torch.from_numpy(Q_j.copy().astype(np.float32)).to(device)

    # print(f"new Qi: {Qi.shape}")
    # print(f"new Qj: {Qj.shape}")

    len_i = Q_i.shape[0]
    len_j = Q_j.shape[0]

    m = 0.5 * (Q_i[:, None, :] + Q_j[None, :, :])  # [len_i, len_j, 2]
    d = (Q_j[None, :, :] - Q_i[:, None, :])  # [len_i, len_j, 2]
    L = torch.linalg.norm(d, dim=-1, keepdim=True)  # [len_i, len_j, 1]
    v = d / L.clamp(min=eps)  # [len_i, len_j, 2]

    u = torch.empty_like(v)  # [len_i, len_j, 2]
    u[..., 0] = -v[..., 1]
    u[..., 1] = v[..., 0]

    # print(f"m.shape: {m.shape}")
    # print(f"d.shape: {d.shape}")
    # print(f"L.shape: {L.shape}")
    # print(f"u.shape: {u.shape}")
    # print(f"v.shape: {v.shape}")

    X = float(EdgeSnappingConfig.X_MAX)
    Y = float(EdgeSnappingConfig.Y_MAX)

    # in sample grid, (x_norm. y_norm) means the center of a pixel
    sx = 2.0 / W
    sy = 2.0 / H
    bx = 1.0 / W - 1.0
    by = 1.0 / H - 1.0

    # image coordinates to NDC
    #                       [a00, a01, t0]
    # target affine matrix: [a10, a11, t1] in NDC (for pytorch grid sampling)

    # print(type(sy))
    # print(sy)

    a00 = sx * (X * u[..., 0])
    a01 = sx * (Y * v[..., 0])
    a10 = sy * (X * u[..., 1])
    a11 = sy * (Y * v[..., 1])
    t0 = sx * m[..., 0] + bx
    t1 = sy * m[..., 1] + by

    # print(f"a00.shape: {a00.shape}")
    # print(f"a01.shape: {a01.shape}")
    # print(f"a10.shape: {a10.shape}")
    # print(f"a11.shape: {a11.shape}")
    # print(f"t0.shape: {t0.shape}")
    # print(f"t1.shape: {t1.shape}")

    # build affine matrix - theta
    theta = torch.stack([
        torch.stack([a00, a01, t0], dim=-1),
        torch.stack([a10, a11, t1], dim=-1)
    ], dim=-2)  # [len_i, len_j, 2, 3]

    theta_flat = theta.reshape(len_i * len_j, 2, 3).contiguous()  # [len_i * len_j, 2, 3]

    # print(f"theta.shape: {theta.shape}")
    # print(f"theta_flat.shape: {theta_flat.shape}")

    return theta_flat
