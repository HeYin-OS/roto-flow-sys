"""
Mask孔洞问题分析工具
分析从60帧开始出现孔洞的原因
"""

import numpy as np
import cv2
from pathlib import Path

def analyze_mask_holes(mask_binary, mask_raw, threshold, current_box, img_h, img_w):
    """
    分析mask中的孔洞问题
    
    Args:
        mask_binary: 二值化后的mask
        mask_raw: 原始SAM2输出的mask（浮点数0-1）
        threshold: 使用的阈值
        current_box: 边界框
        img_h, img_w: 图像尺寸
    """
    analysis = {}
    
    # 1. 分析原始mask的置信度分布
    mask_mean = mask_raw.mean()
    mask_std = mask_raw.std()
    mask_min = mask_raw.min()
    mask_max = mask_raw.max()
    
    # 2. 分析边界框边缘区域的置信度
    box_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    x1, y1, x2, y2 = map(int, current_box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_w, x2), min(img_h, y2)
    box_mask[y1:y2, x1:x2] = 1
    
    from scipy.ndimage import distance_transform_edt
    dist_to_edge = distance_transform_edt(box_mask)
    
    # 边缘区域（距离边缘<30像素）
    edge_mask = (dist_to_edge < 30) & (box_mask > 0)
    edge_values = mask_raw[edge_mask] if edge_mask.any() else np.array([])
    
    # 中心区域（距离边缘>=30像素）
    center_mask = (dist_to_edge >= 30) & (box_mask > 0)
    center_values = mask_raw[center_mask] if center_mask.any() else np.array([])
    
    analysis['edge_mean'] = edge_values.mean() if len(edge_values) > 0 else 0
    analysis['edge_std'] = edge_values.std() if len(edge_values) > 0 else 0
    analysis['edge_min'] = edge_values.min() if len(edge_values) > 0 else 0
    analysis['edge_max'] = edge_values.max() if len(edge_values) > 0 else 0
    analysis['edge_below_threshold'] = (edge_values < threshold).sum() if len(edge_values) > 0 else 0
    analysis['edge_total'] = len(edge_values)
    
    analysis['center_mean'] = center_values.mean() if len(center_values) > 0 else 0
    analysis['center_std'] = center_values.std() if len(center_values) > 0 else 0
    
    # 3. 分析二值化后的mask
    mask_area = mask_binary.sum()
    analysis['binary_mask_area'] = mask_area
    
    # 4. 分析孔洞
    # 找到所有连通域
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_binary, connectivity=8
    )
    analysis['num_components'] = num_labels - 1  # 减去背景
    
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        analysis['largest_component_area'] = areas.max()
        analysis['component_areas'] = areas.tolist()
    
    # 5. 分析边缘区域的mask丢失
    # 检查边缘区域有多少像素在二值化后丢失了
    edge_lost = edge_mask & (mask_binary == 0)
    edge_lost_count = edge_lost.sum()
    analysis['edge_lost_pixels'] = edge_lost_count
    analysis['edge_lost_ratio'] = edge_lost_count / len(edge_values) if len(edge_values) > 0 else 0
    
    # 6. 分析边缘区域丢失像素的原始置信度
    if edge_lost_count > 0:
        lost_values = mask_raw[edge_lost]
        analysis['lost_pixels_mean_confidence'] = lost_values.mean()
        analysis['lost_pixels_min_confidence'] = lost_values.min()
        analysis['lost_pixels_max_confidence'] = lost_values.max()
        analysis['lost_pixels_below_threshold'] = (lost_values < threshold).sum()
        analysis['lost_pixels_above_threshold'] = (lost_values >= threshold).sum()
    
    # 7. 分析是否需要更低的阈值
    edge_threshold = threshold * 0.75
    edge_should_keep = (edge_values >= edge_threshold).sum() if len(edge_values) > 0 else 0
    edge_actually_kept = (edge_mask & (mask_binary > 0)).sum()
    analysis['edge_should_keep'] = edge_should_keep
    analysis['edge_actually_kept'] = edge_actually_kept
    analysis['edge_kept_ratio'] = edge_actually_kept / edge_should_keep if edge_should_keep > 0 else 0
    
    return analysis

def print_analysis(frame_idx, analysis, threshold, edge_threshold):
    """打印分析结果"""
    print(f"\n{'='*60}")
    print(f"帧 {frame_idx} - Mask孔洞分析")
    print(f"{'='*60}")
    
    print(f"\n【原始Mask置信度】")
    print(f"  全局平均: {analysis.get('edge_mean', 0):.4f}")
    print(f"  边缘区域平均: {analysis.get('edge_mean', 0):.4f}")
    print(f"  中心区域平均: {analysis.get('center_mean', 0):.4f}")
    print(f"  边缘区域置信度范围: [{analysis.get('edge_min', 0):.4f}, {analysis.get('edge_max', 0):.4f}]")
    
    print(f"\n【阈值分析】")
    print(f"  使用阈值: {threshold:.4f}")
    print(f"  边缘区域阈值: {edge_threshold:.4f}")
    print(f"  边缘区域低于阈值的像素: {analysis.get('edge_below_threshold', 0)} / {analysis.get('edge_total', 0)}")
    
    print(f"\n【二值化后Mask】")
    print(f"  Mask面积: {analysis.get('binary_mask_area', 0)} 像素")
    print(f"  连通域数量: {analysis.get('num_components', 0)}")
    if 'component_areas' in analysis:
        print(f"  各连通域面积: {analysis['component_areas']}")
    
    print(f"\n【边缘区域丢失分析】")
    print(f"  边缘区域丢失像素数: {analysis.get('edge_lost_pixels', 0)}")
    print(f"  边缘区域丢失比例: {analysis.get('edge_lost_ratio', 0):.2%}")
    if 'lost_pixels_mean_confidence' in analysis:
        print(f"  丢失像素的平均置信度: {analysis['lost_pixels_mean_confidence']:.4f}")
        print(f"  丢失像素的置信度范围: [{analysis['lost_pixels_min_confidence']:.4f}, {analysis['lost_pixels_max_confidence']:.4f}]")
        print(f"  丢失像素中低于阈值的: {analysis.get('lost_pixels_below_threshold', 0)}")
        print(f"  丢失像素中高于阈值的: {analysis.get('lost_pixels_above_threshold', 0)} ⚠️")
    
    print(f"\n【阈值效果评估】")
    print(f"  边缘区域应该保留: {analysis.get('edge_should_keep', 0)} 像素")
    print(f"  边缘区域实际保留: {analysis.get('edge_actually_kept', 0)} 像素")
    print(f"  保留比例: {analysis.get('edge_kept_ratio', 0):.2%}")
    
    # 诊断问题
    print(f"\n【问题诊断】")
    if analysis.get('edge_lost_ratio', 0) > 0.3:
        print(f"  ⚠️  边缘区域丢失超过30%，说明阈值可能过高")
    if analysis.get('lost_pixels_above_threshold', 0) > 0:
        print(f"  ⚠️  有 {analysis['lost_pixels_above_threshold']} 个像素置信度高于阈值但仍被丢弃，可能是局部自适应阈值逻辑问题")
    if analysis.get('num_components', 0) > 1:
        print(f"  ⚠️  有 {analysis['num_components']} 个连通域，可能目标被分割")
    if analysis.get('edge_mean', 0) < 0.3:
        print(f"  ⚠️  边缘区域平均置信度很低 ({analysis['edge_mean']:.4f})，SAM2在边缘区域识别困难")
    
    print(f"{'='*60}\n")
