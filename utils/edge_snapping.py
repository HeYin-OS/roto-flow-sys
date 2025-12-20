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
    lambda_deform = None  # 形变项的权重（用于增强形变控制）
    lambda_velocity = None  # 速度项的权重（防止过大的位移）
    lambda_length = None  # 长度约束项的权重（控制整体polyline长度的变化）

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
        EdgeSnappingConfig.lambda_shape = s.get('lambda_shape', 0.1)  # 默认值0.1
        EdgeSnappingConfig.lambda_topology = s.get('lambda_topology', 0.15)  # 默认值0.15
        EdgeSnappingConfig.lambda_deform = s.get('lambda_deform', 1.0)  # 默认值1.0
        EdgeSnappingConfig.lambda_velocity = s.get('lambda_velocity', 0.5)  # 默认值0.5（强化后的默认值）
        EdgeSnappingConfig.lambda_length = s.get('lambda_length', 0.2)  # 默认值0.2

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
                   points_stroke_candidate: List[np.ndarray],
                   previous_snapped_stroke: np.ndarray = None,
                   original_stroke_length: float = None):
    """
    局部优化步骤：将用户笔画吸附到图像边缘特征上。
    
    实现论文 Section 3.1 的算法：
    1. 构建链状图 G = (V, E)，其中 V = U_0^N Q_i（Q_0 为虚拟起始点，通过将第一组能量设为0实现）
    2. 对每条边 e = (q_{i,k}, q_{i+1,k'}) 计算权重 w_e（Equation 3）
    3. 使用动态规划找最短路径（对应论文中的 s'_1）
    
    Args:
        stroke: 当前帧的原始笔画，形状为(N, 2)
        image_rgb_hwc: 当前帧的RGB图像，形状为(H, W, 3)
        points_stroke_candidate: 候选点列表
        previous_snapped_stroke: 前一帧的吸附结果，形状为(N, 2)，用于计算速度约束
        original_stroke_length: 原始stroke的长度（用于长度约束，如果为None则使用当前stroke的长度）
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.backends.cudnn.benchmark = True

    # 关键修复：如果传入的stroke点数已经超过限制，直接降采样
    # 防止点数累积导致性能下降
    MAX_STROKE_POINTS = 200
    if len(stroke) > MAX_STROKE_POINTS:
        indices = np.linspace(0, len(stroke) - 1, MAX_STROKE_POINTS, dtype=np.int32)
        stroke = stroke[indices]
        if len(points_stroke_candidate) > MAX_STROKE_POINTS:
            # 同时调整候选点列表的长度
            points_stroke_candidate = points_stroke_candidate[:MAX_STROKE_POINTS]
    
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
    
    # 计算原始stroke的总长度（用于长度约束）
    # 如果提供了original_stroke_length，优先使用它（用于fitted传播，保持与前一帧的长度一致）
    if original_stroke_length is not None and original_stroke_length > 1e-6:
        stroke_total_length = original_stroke_length
    else:
        # 使用向量化计算，避免循环
        stroke_diffs = stroke[1:] - stroke[:-1]
        stroke_total_length = np.sum(np.linalg.norm(stroke_diffs, axis=1))
    
    # 计算前一帧的总长度（用于长度约束）
    prev_stroke_total_length = None
    if previous_snapped_stroke is not None and len(previous_snapped_stroke) > 1:
        # 使用向量化计算，避免循环
        prev_diffs = previous_snapped_stroke[1:] - previous_snapped_stroke[:-1]
        prev_stroke_total_length = np.sum(np.linalg.norm(prev_diffs, axis=1))

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
        
        # 获取前一帧对应点的位置（用于速度约束）
        prev_p_i = None
        prev_p_j = None
        if previous_snapped_stroke is not None and i_group < len(previous_snapped_stroke) and i_group + 1 < len(previous_snapped_stroke):
            prev_p_i = previous_snapped_stroke[i_group]
            prev_p_j = previous_snapped_stroke[i_group + 1]
        
        weights = compute_weights(H, W,
                                  p_i, p_j,
                                  Q_i, Q_j,
                                  image_tensor_gray_chw,
                                  device,
                                  prev_q_i=prev_q_i,
                                  prev_p_i=prev_p_i,
                                  prev_p_j=prev_p_j,
                                  stroke_total_length=stroke_total_length,
                                  prev_stroke_total_length=prev_stroke_total_length)  # shape: [K_i, K_j]

        dp_energy_iteration(i_group, flatten_index_ptr, energy, prev, weights)
        
        # 关键优化：减少同步频率，但确保在关键位置同步
        # 每2个点对同步一次，平衡性能和队列累积
        if torch.cuda.is_available():
            if (i_group + 1) % 2 == 0:  # 每2个点对同步一次
                torch.cuda.synchronize()  # 确保GPU操作完成
                torch.cuda.empty_cache()  # 清理GPU缓存
                # 每10个点对重置一次GPU内存统计
                if (i_group + 1) % 10 == 0:
                    torch.cuda.reset_peak_memory_stats()  # 重置峰值内存统计

    snapped_stroke = pick_best_path(last_start=flatten_index_ptr[-2],
                                    last_end=flatten_index_ptr[-1],
                                    energy=energy,
                                    prev=prev,
                                    candidates_flatten=points_stroke_candidate_flatten,
                                    stroke=stroke,
                                    average_weight_standard=EdgeSnappingConfig.average_weight_threshold,
                                    average_distance_standard=EdgeSnappingConfig.r_s / 4.0)
    
    # 后处理：如果长度变化超过阈值，进行归一化（已禁用，仅保留核心算法）
    # if original_stroke_length is not None and original_stroke_length > 1e-6:
    #     # 使用向量化计算，避免循环
    #     snapped_diffs = snapped_stroke[1:] - snapped_stroke[:-1]
    #     snapped_length = np.sum(np.linalg.norm(snapped_diffs, axis=1))
    #     
    #     # 如果长度变化超过3%，进行归一化
    #     if snapped_length > 1e-6 and abs(snapped_length - original_stroke_length) / original_stroke_length > 0.03:
    #         scale_factor = original_stroke_length / snapped_length
    #         # 以第一个点为基准进行缩放（向量化）
    #         center = snapped_stroke[0]
    #         snapped_stroke = center + (snapped_stroke - center) * scale_factor
    
    # 检查并修复拓扑结构
    snapped_stroke = check_and_fix_topology(snapped_stroke, stroke)
    
    # 显式释放GPU内存 - 更彻底的清理
    del image_tensor_gray_chw
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # 确保所有GPU操作完成
        torch.cuda.empty_cache()  # 清理GPU缓存
        # 强制同步，确保内存真正释放
        torch.cuda.ipc_collect()  # 清理进程间通信缓存（如果存在）
    
    return snapped_stroke


def check_and_fix_topology(snapped_stroke: np.ndarray, original_stroke: np.ndarray) -> np.ndarray:
    """
    检查吸附后笔画的拓扑结构，并修复拓扑问题（交叉、方向反转等）
    
    Args:
        snapped_stroke: 吸附后的笔画，形状为(N, 2)
        original_stroke: 原始笔画，形状为(N, 2)，用于参考
    
    Returns:
        修复后的笔画，形状为(N, 2)
    """
    if len(snapped_stroke) < 3:
        # 点数太少，无法检查拓扑
        return snapped_stroke
    
    # 检查是否有交叉
    has_crossing = detect_crossings(snapped_stroke)
    
    # 检查方向一致性
    direction_issues = detect_direction_issues(snapped_stroke, original_stroke)
    
    # 如果检测到拓扑问题，进行修复
    if has_crossing or direction_issues:
        fixed_stroke = fix_topology_issues(snapped_stroke, original_stroke, has_crossing, direction_issues)
        return fixed_stroke
    
    return snapped_stroke


def detect_crossings(stroke: np.ndarray) -> bool:
    """
    检测笔画中是否有边交叉
    
    优化：只检查相邻的边和局部范围内的边，避免O(n²)复杂度
    
    Args:
        stroke: 笔画点数组，形状为(N, 2)
    
    Returns:
        是否存在交叉
    """
    n = len(stroke)
    if n < 4:
        return False
    
    # 优化：限制检查范围，只检查相邻的边和局部范围内的边
    # 这样可以避免O(n²)复杂度，同时仍能检测到大部分交叉问题
    CHECK_RANGE = min(10, n // 4)  # 最多检查前后10条边，或总边数的1/4
    
    # 检查相邻的边是否交叉
    for i in range(n - 3):
        p1 = stroke[i]
        p2 = stroke[i + 1]
        
        # 只检查局部范围内的边，而不是所有边
        start_j = max(i + 2, 0)
        end_j = min(i + CHECK_RANGE + 2, n - 1)
        
        for j in range(start_j, end_j):
            if j >= n - 1:
                break
            p3 = stroke[j]
            p4 = stroke[j + 1]
            
            # 使用叉积检测线段是否相交
            if segments_intersect(p1, p2, p3, p4):
                return True
    
    return False


def segments_intersect(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> bool:
    """
    检测两条线段是否相交（使用叉积方法）
    
    Args:
        p1, p2: 第一条线段的端点
        p3, p4: 第二条线段的端点
    
    Returns:
        是否相交
    """
    def ccw(A, B, C):
        """判断三点是否逆时针排列"""
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
    
    # 两条线段相交当且仅当：
    # (p1, p2, p3) 和 (p1, p2, p4) 的方向不同，且
    # (p3, p4, p1) 和 (p3, p4, p2) 的方向不同
    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

def detect_direction_issues(snapped_stroke: np.ndarray, original_stroke: np.ndarray) -> bool:
    """
    检测方向问题（点是否按正确顺序排列）
    
    改进：更敏感地检测局部失序问题
    
    Args:
        snapped_stroke: 吸附后的笔画
        original_stroke: 原始笔画
    
    Returns:
        是否存在方向问题
    """
    if len(snapped_stroke) < 2 or len(original_stroke) < 2:
        return False
    
    # 计算原始笔画的总方向
    original_dir = original_stroke[-1] - original_stroke[0]
    original_length = np.linalg.norm(original_dir)
    
    if original_length < 1e-6:
        return False
    
    original_dir_normalized = original_dir / original_length
    
    # 计算吸附后笔画的总方向
    snapped_dir = snapped_stroke[-1] - snapped_stroke[0]
    snapped_length = np.linalg.norm(snapped_dir)
    
    if snapped_length < 1e-6:
        return False
    
    snapped_dir_normalized = snapped_dir / snapped_length
    
    # 检查方向是否一致（点积应该接近1）
    direction_consistency = np.dot(original_dir_normalized, snapped_dir_normalized)
    
    # 如果方向一致性小于0.7（更严格的阈值），认为有方向问题
    if direction_consistency < 0.7:
        return True
    
    # 检查局部方向是否频繁反转（改进：更敏感）
    reversal_count = 0
    severe_reversal_count = 0
    for i in range(len(snapped_stroke) - 2):
        edge1 = snapped_stroke[i + 1] - snapped_stroke[i]
        edge2 = snapped_stroke[i + 2] - snapped_stroke[i + 1]
        
        if np.linalg.norm(edge1) > 1e-6 and np.linalg.norm(edge2) > 1e-6:
            edge1_norm = edge1 / np.linalg.norm(edge1)
            edge2_norm = edge2 / np.linalg.norm(edge2)
            dot_product = np.dot(edge1_norm, edge2_norm)
            
            # 如果相邻边的方向相反（点积为负），计数
            if dot_product < -0.2:  # 降低阈值，更敏感
                reversal_count += 1
            if dot_product < -0.5:  # 严重反转
                severe_reversal_count += 1
    
    # 如果严重反转次数超过总边数的10%，认为有方向问题
    if severe_reversal_count > (len(snapped_stroke) - 2) * 0.1:
        return True
    
    # 如果反转次数超过总边数的15%（降低阈值），认为有方向问题
    if reversal_count > (len(snapped_stroke) - 2) * 0.15:
        return True
    
    # 检查局部顺序：检测是否有"回退"的点
    # 如果某个点相对于前一个点在错误的方向上，说明顺序有问题
    if len(snapped_stroke) >= 3:
        backward_count = 0
        for i in range(1, len(snapped_stroke) - 1):
            # 计算从i-1到i+1的方向
            forward_dir = snapped_stroke[i + 1] - snapped_stroke[i - 1]
            forward_dir_norm = forward_dir / (np.linalg.norm(forward_dir) + 1e-6)
            
            # 计算从i-1到i的方向
            to_i_dir = snapped_stroke[i] - snapped_stroke[i - 1]
            to_i_dir_norm = to_i_dir / (np.linalg.norm(to_i_dir) + 1e-6)
            
            # 如果点i在错误的方向上（点积为负），说明顺序有问题
            if np.dot(forward_dir_norm, to_i_dir_norm) < 0.3:
                backward_count += 1
        
        # 如果回退点超过10%，认为有方向问题
        if backward_count > (len(snapped_stroke) - 2) * 0.1:
            return True
    
    return False


def fix_topology_issues(snapped_stroke: np.ndarray, original_stroke: np.ndarray, 
                       has_crossing: bool, has_direction_issue: bool) -> np.ndarray:
    """
    修复拓扑问题
    
    Args:
        snapped_stroke: 吸附后的笔画
        original_stroke: 原始笔画
        has_crossing: 是否有交叉
        has_direction_issue: 是否有方向问题
    
    Returns:
        修复后的笔画
    """
    fixed_stroke = snapped_stroke.copy()
    
    # 如果有方向问题，尝试反转部分点
    if has_direction_issue:
        # 计算每个点相对于原始笔画的位置
        # 如果整体方向相反，反转整个笔画
        original_dir = original_stroke[-1] - original_stroke[0]
        snapped_dir = fixed_stroke[-1] - fixed_stroke[0]
        
        if np.dot(original_dir, snapped_dir) < 0:
            # 方向相反，反转整个笔画
            fixed_stroke = fixed_stroke[::-1]
        else:
            # 整体方向正确，但可能有局部失序，进行局部修复
            fixed_stroke = fix_local_disorder(fixed_stroke, original_stroke)
    
    # 如果有交叉，尝试平滑处理
    if has_crossing:
        # 使用简单的平滑来减少交叉
        # 对每个点，计算其与相邻点的平均位置
        smoothed = fixed_stroke.copy()
        for i in range(1, len(fixed_stroke) - 1):
            # 使用加权平均：当前点权重0.6，相邻点各0.2
            smoothed[i] = 0.6 * fixed_stroke[i] + 0.2 * fixed_stroke[i - 1] + 0.2 * fixed_stroke[i + 1]
        fixed_stroke = smoothed
    
    # 确保点之间的距离不会太短或太长
    fixed_stroke = enforce_minimum_distance(fixed_stroke, original_stroke)
    
    return fixed_stroke


def fix_local_disorder(stroke: np.ndarray, original_stroke: np.ndarray) -> np.ndarray:
    """
    修复局部失序问题
    
    检测并修复局部点的顺序问题，例如某个点相对于前一个点在错误的方向上
    
    Args:
        stroke: 需要修复的笔画
        original_stroke: 原始笔画（用于参考）
    
    Returns:
        修复后的笔画
    """
    if len(stroke) < 3:
        return stroke
    
    fixed_stroke = stroke.copy()
    
    # 计算原始笔画的方向
    original_dir = original_stroke[-1] - original_stroke[0]
    original_dir_norm = original_dir / (np.linalg.norm(original_dir) + 1e-6)
    
    # 检测并修复局部失序
    for i in range(1, len(fixed_stroke) - 1):
        # 计算从i-1到i+1的方向（期望方向）
        expected_dir = fixed_stroke[i + 1] - fixed_stroke[i - 1]
        expected_dir_norm = expected_dir / (np.linalg.norm(expected_dir) + 1e-6)
        
        # 计算从i-1到i的方向（实际方向）
        actual_dir = fixed_stroke[i] - fixed_stroke[i - 1]
        actual_dir_norm = actual_dir / (np.linalg.norm(actual_dir) + 1e-6)
        
        # 如果实际方向与期望方向差异很大（点积小于0.5），说明点i可能位置不对
        if np.dot(expected_dir_norm, actual_dir_norm) < 0.5:
            # 尝试将点i移动到更合理的位置
            # 使用线性插值：点i应该在i-1和i+1之间的中间位置
            fixed_stroke[i] = 0.5 * fixed_stroke[i - 1] + 0.5 * fixed_stroke[i + 1]
    
    return fixed_stroke


def enforce_minimum_distance(stroke: np.ndarray, original_stroke: np.ndarray) -> np.ndarray:
    """
    确保点之间的距离合理（不会太短或太长）
    
    Args:
        stroke: 笔画点数组
        original_stroke: 原始笔画（用于参考距离）
    
    Returns:
        调整后的笔画
    """
    if len(stroke) < 2:
        return stroke
    
    # 关键修复：使用固定的最大点数限制，防止点数无限增长
    # 如果original_stroke已经很大，使用固定限制；否则使用2倍限制
    original_len = len(original_stroke)
    if original_len > 100:
        # 如果原始stroke已经很大，使用固定限制（200点）
        MAX_STROKE_POINTS = 200
    else:
        # 如果原始stroke较小，使用2倍限制
        MAX_STROKE_POINTS = min(original_len * 2, 200)  # 最多200点
    
    # 如果当前stroke已经超过限制，直接降采样返回
    if len(stroke) > MAX_STROKE_POINTS:
        indices = np.linspace(0, len(stroke) - 1, MAX_STROKE_POINTS, dtype=np.int32)
        return stroke[indices]
    
    # 计算原始笔画的平均边长度
    original_lengths = []
    for i in range(len(original_stroke) - 1):
        length = np.linalg.norm(original_stroke[i + 1] - original_stroke[i])
        if length > 1e-6:
            original_lengths.append(length)
    
    if len(original_lengths) == 0:
        return stroke
    
    avg_length = np.mean(original_lengths)
    min_length = avg_length * 0.3  # 最小距离为平均距离的30%
    max_length = avg_length * 3.0  # 最大距离为平均距离的3倍
    
    fixed_stroke = stroke.copy()
    
    # 检查并调整过短或过长的边
    i = 0
    while i < len(fixed_stroke) - 1:
        # 如果点数已经超过限制，停止插入新点，只进行合并操作
        if len(fixed_stroke) >= MAX_STROKE_POINTS:
            # 只进行合并操作，不再插入新点
            edge = fixed_stroke[i + 1] - fixed_stroke[i]
            length = np.linalg.norm(edge)
            if length < min_length and length > 1e-6:
                if i + 1 < len(fixed_stroke) - 1:
                    fixed_stroke = np.delete(fixed_stroke, i + 1, axis=0)
                    continue
            i += 1
            continue
        
        edge = fixed_stroke[i + 1] - fixed_stroke[i]
        length = np.linalg.norm(edge)
        
        if length < min_length and length > 1e-6:
            # 边太短，移除中间的点（如果存在）或调整位置
            if i + 1 < len(fixed_stroke) - 1:
                # 合并到下一个点
                fixed_stroke = np.delete(fixed_stroke, i + 1, axis=0)
                continue
        elif length > max_length:
            # 边太长，在中间插入点（但不超过最大点数限制）
            num_insert = int(length / avg_length)
            if num_insert > 1:
                # 限制插入的点数，确保不超过最大点数
                max_insert = MAX_STROKE_POINTS - len(fixed_stroke)
                num_insert = min(num_insert, max_insert)
                if num_insert > 0:
                    new_points = []
                    for j in range(1, num_insert + 1):
                        t = j / (num_insert + 1)
                        new_point = fixed_stroke[i] * (1 - t) + fixed_stroke[i + 1] * t
                        new_points.append(new_point)
                    if new_points:
                        fixed_stroke = np.insert(fixed_stroke, i + 1, new_points, axis=0)
                        i += len(new_points)
        
        i += 1
    
    # 如果点数仍然超过限制，进行降采样
    if len(fixed_stroke) > MAX_STROKE_POINTS:
        # 均匀采样，保持原始点数
        indices = np.linspace(0, len(fixed_stroke) - 1, MAX_STROKE_POINTS, dtype=np.int32)
        fixed_stroke = fixed_stroke[indices]
    
    return fixed_stroke


def candidate_point_sets_defense(points_candidate: list[np.ndarray], stroke: np.ndarray) -> list[np.ndarray]:
    fixed_candidates = []
    stroke_len = stroke.shape[0]
    
    # 确保返回的列表长度与 stroke 的长度一致
    for t in range(stroke_len):
        if t < len(points_candidate):
            cand = points_candidate[t]
            if cand is None or (isinstance(cand, np.ndarray) and cand.size == 0):
                p = stroke[t].astype(np.float32)
                fixed_candidates.append(np.array([[p[0], p[1]]], dtype=np.float32))
            else:
                fixed_candidates.append(cand)
        else:
            # 如果 points_candidate 长度不足，用 stroke 点填充
            p = stroke[t].astype(np.float32)
            fixed_candidates.append(np.array([[p[0], p[1]]], dtype=np.float32))
    
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
    
    # 性能优化：限制最大迭代次数，防止无限循环
    max_iterations = 10  # 最多迭代10次
    iteration_count = 0

    while True:
        iteration_count += 1
        if iteration_count > max_iterations:
            # 如果迭代次数过多，直接返回当前最佳路径
            best_idx = np.argmin(energy[last_start:last_end]) + last_start
            best_path_indices = []
            temp_idx = best_idx
            while temp_idx != -1:
                best_path_indices.append(temp_idx)
                temp_idx = prev[temp_idx]
            best_path_indices.reverse()
            return candidates_flatten[best_path_indices]
        # last point index of underlying candidate stroke
        best_idx = np.argmin(energy[last_start:last_end]) + last_start
        curr_idx = best_idx

        avg_en = energy[best_idx] / (stroke_len - 1)

        # use backtrack to find all points of candidate stroke
        # 使用append然后reverse，避免insert(0, ...)的O(n)复杂度
        best_path_indices = []
        temp_idx = best_idx
        while temp_idx != -1:
            best_path_indices.append(temp_idx)
            temp_idx = prev[temp_idx]
        best_path_indices.reverse()  # O(n)操作，但只执行一次

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
                    prev_q_i: np.ndarray = None,
                    prev_p_i: np.ndarray = None,
                    prev_p_j: np.ndarray = None,
                    stroke_total_length: float = None,
                    prev_stroke_total_length: float = None):
    """
    计算边权重 w_e（论文 Equation 3 + 形状约束 + 拓扑顺序 + 速度约束 + 长度约束）。
    
    对于每条边 e = (q_{i,k}, q_{i+1,k'})：
    - 计算中点 m = (q_{i,k} + q_{i+1,k'}) / 2
    - 计算方向 v = normalize(q_{i+1,k'} - q_{i,k})
    - 计算垂直方向 u = rotate_90(v)
    - 应用 FDoG 滤波器 H(m,v)（论文 Equation 1）：先沿 u 方向（x）做 DoG，再沿 v 方向（y）做高斯平滑
    - 转换为 H̃(m,v)（论文 Equation 2）
    - 计算 deform term：||(p_{i+1} - p_i) - (q_{i+1} - q_i)||^2 / r_s^2
    - 计算 shape constraint term：形状约束项（距离和角度）
    - 计算 topology term：拓扑顺序项（避免交叉和方向一致性）
    - 计算 velocity term：速度约束项（防止过大的位移，强化版）
    - 计算 length term：长度约束项（控制整体polyline长度的变化）
    - 最终权重：w_e = deform_term + α * H̃(m,v) + λ_shape * shape_term + λ_topology * topology_term + λ_velocity * velocity_term + λ_length * length_term
    
    Args:
        prev_q_i: 前一个点的候选位置（用于计算形状约束和拓扑顺序），形状为(K_{i-1}, 2)
        prev_p_i: 前一帧对应点i的位置（用于计算速度约束），形状为(2,)
        prev_p_j: 前一帧对应点j的位置（用于计算速度约束），形状为(2,)
        stroke_total_length: 当前帧原始stroke的总长度
        prev_stroke_total_length: 前一帧stroke的总长度
    """
    # 使用torch.no_grad()避免梯度计算开销
    with torch.no_grad():
        # 注意：compute_affine_theta_vectorized已经返回GPU tensor，不需要再次.to(device)
        theta_flatten_gpu = compute_affine_theta_vectorized(Q_i,
                                                            Q_j,
                                                            H, W)  # shape: [K_{i} * K_{i+1}, 2, 3]
        # 确保tensor是contiguous的，避免后续操作的开销
        if not theta_flatten_gpu.is_contiguous():
            theta_flatten_gpu = theta_flatten_gpu.contiguous()
        # affine_grid操作本身就在GPU上，不需要.to(device)
        grid_gpu = torch.nn.functional.affine_grid(theta_flatten_gpu,
                                                   size=[theta_flatten_gpu.shape[0],
                                                         1,
                                                         2 * EdgeSnappingConfig.Y_MAX + 1,
                                                         2 * EdgeSnappingConfig.X_MAX + 1],
                                                   align_corners=False)  # shape: [K_{i} * K_{i+1}, 2*Y+1, 2*X+1, 2]
        # 确保grid是contiguous的
        if not grid_gpu.is_contiguous():
            grid_gpu = grid_gpu.contiguous()

        # 使用expand而不是repeat，expand是view操作，不复制数据，更高效
        # 但需要确保tensor是contiguous的
        batch_size = grid_gpu.shape[0]
        
        # 性能优化：如果batch_size太大（超过1000），可能影响性能
        # 这通常发生在候选点数量很多的情况下
        if batch_size > 1000:
            # 可以考虑进一步限制候选点数量，但这里先继续处理
            pass
        
        # 确保image_gray_chw是contiguous的
        if not image_gray_chw.is_contiguous():
            image_gray_chw = image_gray_chw.contiguous()
        
        # 关键优化：使用repeat而不是expand，因为expand在某些情况下可能不会真正共享内存
        # 虽然repeat会复制数据，但可以避免GPU内存碎片化问题
        # 对于batch_size很大的情况，可以考虑分批处理
        if batch_size > 500:
            # 如果batch_size太大，分批处理以避免GPU内存问题
            # 但这需要修改grid_sample的调用方式，暂时先继续使用expand
            image_expanded = image_gray_chw.expand(batch_size, -1, -1, -1)
        else:
            image_expanded = image_gray_chw.expand(batch_size, -1, -1, -1)
        
        # 执行grid_sample（这是最耗时的GPU操作）
        image_affined_gpu = torch.nn.functional.grid_sample(image_expanded,
                                                           grid_gpu,
                                                           mode='bilinear',
                                                           padding_mode='zeros',
                                                           align_corners=False)
        
        # 优化：减少同步频率，只在必要时同步
        # 立即释放GPU tensor，避免累积
        del theta_flatten_gpu, grid_gpu, image_expanded
        
        # 优化：.cpu()操作会隐式同步GPU，但为了确保操作完成，在CPU传输前同步一次
        # 注意：频繁同步会导致性能下降，所以只在必要时同步
        if torch.cuda.is_available():
            # 只在batch_size较大时同步，避免小操作的开销
            if batch_size > 200:
                torch.cuda.synchronize()  # 确保GPU操作完成
        
        # 将结果移到CPU并转换为numpy（这会触发GPU到CPU的传输，隐式同步）
        image_affined = (image_affined_gpu
                         .squeeze(1)
                         .reshape(Q_i.shape[0],
                                  Q_j.shape[0],
                                  2 * EdgeSnappingConfig.Y_MAX + 1,
                                  2 * EdgeSnappingConfig.X_MAX + 1)
                         .cpu().numpy())
        
        # 删除GPU tensor
        del image_affined_gpu

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
    # 形变项：惩罚候选边与原始边的差异
    r_s_square = float(EdgeSnappingConfig.r_s) ** 2

    p_diff = (p_j - p_i).astype(np.float32)
    q_diff = Q_j.astype(np.float32)[None, :, :] - Q_i.astype(np.float32)[:, None, :]
    diff = p_diff.reshape(1, 1, 2) - q_diff
    deform_term = np.sum(diff * diff, axis=-1) / r_s_square

    # shift term (not on paper) - 额外的位置偏移惩罚
    shift_i = np.sum((Q_i.astype(np.float32) - p_i[None, :]) ** 2, axis=-1)  # [K_i]
    shift_j = np.sum((Q_j.astype(np.float32) - p_j[None, :]) ** 2, axis=-1)  # [K_{i+1}]
    shift_term = (shift_i[:, None] + shift_j[None, :]) / (2.0 * r_s_square)  # [K_i, K_{i+1}]

    # 基础权重：使用 lambda_deform 增强形变控制
    # 如果形变过大，通过增加 deform_term 的权重来惩罚
    deform_weight = EdgeSnappingConfig.lambda_deform if EdgeSnappingConfig.lambda_deform is not None else 1.0
    weights = deform_weight * deform_term + EdgeSnappingConfig.alpha * tilde_H_response
    
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

    # 速度约束项（Velocity Term，强化版）
    # 防止相邻帧之间过大的位移，保持时间连续性
    if EdgeSnappingConfig.lambda_velocity > 0 and prev_p_i is not None and prev_p_j is not None:
        velocity_term = compute_velocity_term(
            prev_p_i, prev_p_j, Q_i, Q_j, r_s_square
        )
        weights = weights + EdgeSnappingConfig.lambda_velocity * velocity_term

    # 长度约束项（Length Term）
    # 控制整体polyline长度的变化，保持长度稳定性
    if EdgeSnappingConfig.lambda_length > 0 and stroke_total_length is not None and stroke_total_length > 1e-6:
        length_term = compute_length_term(
            p_i, p_j, Q_i, Q_j, stroke_total_length, prev_stroke_total_length, r_s_square
        )
        weights = weights + EdgeSnappingConfig.lambda_length * length_term

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
    
    约束相邻点之间的角度，保持笔画的形状一致性
    
    注意：长度约束已移至 compute_length_term，避免重复计算
    
    Args:
        p_i: 原始笔画点i，形状为(2,)
        p_j: 原始笔画点j，形状为(2,)
        Q_i: 候选点组i，形状为(K_i, 2)
        Q_j: 候选点组j，形状为(K_j, 2)
        prev_q_i: 前一个点的候选位置（可选），形状为(K_{i-1}, 2)
        r_s_square: r_s的平方，用于归一化（当前未使用，保留以兼容接口）
    
    Returns:
        形状约束项（角度约束），形状为(K_i, K_j)
    """
    # 计算候选边的方向（用于角度约束）
    q_diff = Q_j[None, :, :] - Q_i[:, None, :]  # (K_i, K_j, 2)
    q_length = np.linalg.norm(q_diff, axis=-1, keepdims=True)  # (K_i, K_j, 1)
    
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
        curr_edge_normalized = q_diff / (q_length + 1e-6)  # (K_i, K_j, 2)
        
        # 计算角度变化（使用点积）
        cos_angle = np.sum(prev_edge_normalized[:, None, :] * curr_edge_normalized, axis=-1)  # (K_i, K_j)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        # 角度变化惩罚（角度变化越大，惩罚越大）
        angle_diff = 1.0 - cos_angle  # 0表示角度相同，2表示完全相反
        angle_term = angle_diff * angle_diff
    
    # 形状约束项：只包含角度约束（长度约束已移至 compute_length_term）
    shape_term = angle_term
    
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
    # 增强拓扑约束的权重，确保拓扑逻辑得到保持
    # 方向一致性是最重要的，交叉检测和顺序约束也很重要
    topology_term = 1.0 * direction_term + 0.5 * cross_term + 0.4 * order_term
    
    return topology_term


def compute_velocity_term(prev_p_i: np.ndarray, prev_p_j: np.ndarray,
                          Q_i: np.ndarray, Q_j: np.ndarray,
                          r_s_square: float = 400.0) -> np.ndarray:
    """
    计算速度约束项（Velocity Term，强化版）
    
    防止相邻帧之间过大的位移，保持时间连续性。
    使用分段惩罚函数：对小位移使用线性惩罚，对大位移使用更强的平方惩罚。
    
    Args:
        prev_p_i: 前一帧对应点i的位置，形状为(2,)
        prev_p_j: 前一帧对应点j的位置，形状为(2,)
        Q_i: 候选点组i，形状为(K_i, 2)
        Q_j: 候选点组j，形状为(K_j, 2)
        r_s_square: r_s的平方，用于归一化
    
    Returns:
        速度约束项，形状为(K_i, K_j)
    """
    # 计算当前帧候选点与前一帧对应点的位移
    # 对于点i的位移
    displacement_i = Q_i - prev_p_i[None, :]  # (K_i, 2)
    displacement_i_sq = np.sum(displacement_i * displacement_i, axis=-1)  # (K_i,)
    displacement_i_norm = np.sqrt(displacement_i_sq)  # (K_i,)
    
    # 对于点j的位移
    displacement_j = Q_j - prev_p_j[None, :]  # (K_j, 2)
    displacement_j_sq = np.sum(displacement_j * displacement_j, axis=-1)  # (K_j,)
    displacement_j_norm = np.sqrt(displacement_j_sq)  # (K_j,)
    
    # 强化版速度项：使用分段惩罚函数
    # 阈值：r_s（搜索半径）作为分界点
    r_s = np.sqrt(r_s_square)
    threshold = r_s * 0.3  # 降低阈值，使速度约束更敏感，抑制摆动
    
    # 对于点i：小位移线性惩罚，大位移平方惩罚
    velocity_i = np.where(
        displacement_i_norm <= threshold,
        displacement_i_sq / r_s_square,  # 线性惩罚（归一化）
        (displacement_i_norm / r_s) ** 2  # 平方惩罚（对大位移更强）
    )
    
    # 对于点j：小位移线性惩罚，大位移平方惩罚
    velocity_j = np.where(
        displacement_j_norm <= threshold,
        displacement_j_sq / r_s_square,  # 线性惩罚（归一化）
        (displacement_j_norm / r_s) ** 2  # 平方惩罚（对大位移更强）
    )
    
    # 综合速度项：使用平均值，并添加额外的交叉项惩罚
    # 如果两个点都位移很大，给予额外惩罚
    velocity_term = (velocity_i[:, None] + velocity_j[None, :]) / 2.0
    
    # 添加交叉项：如果两个点的位移都超过阈值，给予额外惩罚
    both_large = (displacement_i_norm[:, None] > threshold) & (displacement_j_norm[None, :] > threshold)
    cross_penalty = np.where(both_large, 1.0, 0.0)  # 增强交叉惩罚，抑制摆动
    velocity_term = velocity_term + cross_penalty
    
    return velocity_term


def compute_length_term(p_i: np.ndarray, p_j: np.ndarray,
                        Q_i: np.ndarray, Q_j: np.ndarray,
                        stroke_total_length: float,
                        prev_stroke_total_length: float = None,
                        r_s_square: float = 400.0) -> np.ndarray:
    """
    计算长度约束项（Length Term）
    
    控制整体polyline长度的变化，保持长度稳定性。
    惩罚导致总长度变化过大的候选边。
    
    Args:
        p_i: 原始笔画点i，形状为(2,)
        p_j: 原始笔画点j，形状为(2,)
        Q_i: 候选点组i，形状为(K_i, 2)
        Q_j: 候选点组j，形状为(K_j, 2)
        stroke_total_length: 当前帧原始stroke的总长度
        prev_stroke_total_length: 前一帧stroke的总长度（可选）
        r_s_square: r_s的平方，用于归一化
    
    Returns:
        长度约束项，形状为(K_i, K_j)
    """
    # 计算原始边的长度
    p_edge_length = np.linalg.norm(p_j - p_i)
    if p_edge_length < 1e-6:
        p_edge_length = 1e-6
    
    # 计算候选边的长度
    q_edge_length = np.linalg.norm(Q_j[None, :, :] - Q_i[:, None, :], axis=-1)  # (K_i, K_j)
    
    # 计算该边对总长度变化的贡献
    # 如果该边变长/变短，会导致总长度变化
    edge_length_diff = np.abs(q_edge_length - p_edge_length) / p_edge_length  # (K_i, K_j)
    
    # 基础长度约束：惩罚与原始边长度差异较大的情况
    # 使用平方惩罚，但对小差异更宽容
    length_term = edge_length_diff * edge_length_diff  # (K_i, K_j)
    
    # 如果有原始长度信息（第一帧的原始长度），添加全局长度约束
    # 这是最重要的约束：当前帧的结果应该几乎和最开始的输入一样长
    if stroke_total_length is not None and stroke_total_length > 1e-6:
        # 使用第一帧的原始stroke长度作为唯一参考
        # 计算如果选择该边，估计的总长度变化（相对于第一帧原始长度）
        # 注意：这里假设其他边保持不变，虽然不完美，但可以给出局部估计
        estimated_total_length = stroke_total_length - p_edge_length + q_edge_length
        total_length_ratio_to_original = estimated_total_length / stroke_total_length
        
        # 严格惩罚任何偏离原始长度的变化
        # 使用更严格的阈值（3%），对任何变化都给予强惩罚
        # 使用指数惩罚函数，使长度约束更敏感
        length_deviation = np.abs(total_length_ratio_to_original - 1.0)
        length_change_penalty = np.where(
            length_deviation > 0.03,
            (total_length_ratio_to_original - 1.0) ** 2 * 20.0,  # 对超过3%的变化给予极强惩罚
            (total_length_ratio_to_original - 1.0) ** 2 * 10.0   # 对3%内的变化也给予强惩罚
        )
        
        # 添加额外的长度偏差惩罚（即使很小也惩罚）
        length_term = length_term + length_change_penalty + length_deviation * 2.0
    
    return length_term


def compute_affine_theta_vectorized(Q_i: np.ndarray,
                                    Q_j: np.ndarray,
                                    H: int, W: int,
                                    eps: np.float32 = 1e-6) -> Tensor:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 优化：确保输入是float32，避免类型转换开销
    if Q_i.dtype != np.float32:
        Q_i = Q_i.astype(np.float32)
    if Q_j.dtype != np.float32:
        Q_j = Q_j.astype(np.float32)
    
    # 避免不必要的copy，直接转换到GPU
    # 使用pin_memory=False避免额外的CPU-GPU传输开销
    Q_i = torch.from_numpy(Q_i).to(device, non_blocking=False)  # 改为False，确保立即传输
    Q_j = torch.from_numpy(Q_j).to(device, non_blocking=False)  # 改为False，确保立即传输

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
