"""
绘制4种策略的边界贴合误差（Boundary Adherence Error, BAE）折线图
"""
from pathlib import Path
from typing import List
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from tqdm import tqdm

# ========== 配置 ==========
TARGET_NAME = "bear"  # 修改这里来指定不同的target

# 颜色定义（与 test_stroke.py 保持一致，RGB格式需要转换为matplotlib的[0,1]范围）
COLOR_FLOW_SNAPPED = np.array([255, 100, 0]) / 255.0  # Orange Red - 纯光流+纯吸附
COLOR_DILATED_FLOW_SNAPPING = np.array([255, 50, 150]) / 255.0  # Bright Pink - 膨胀光流+纯吸附
COLOR_FLOW_FILTERED_SNAPPING = np.array([0, 255, 200]) / 255.0  # Bright Cyan - 纯光流+过滤吸附
COLOR_FITTED = np.array([50, 200, 50]) / 255.0  # Fresh Green - 膨胀光流+过滤吸附


def load_stroke_npy_files(target_name: str, strategy_name: str) -> List[np.ndarray]:
    """加载指定策略的所有笔画数据"""
    base = Path(__file__).resolve().parent.parent
    strategy_dir = base / "results" / target_name / "strokes" / strategy_name
    
    if not strategy_dir.exists():
        raise RuntimeError(f"Strategy directory does not exist: {strategy_dir}")
    
    # 查找所有npy文件
    npy_files = sorted(strategy_dir.glob("stroke_*.npy"))
    
    if not npy_files:
        raise RuntimeError(f"No stroke files found in {strategy_dir}")
    
    strokes = []
    for npy_path in npy_files:
        stroke = np.load(str(npy_path))
        strokes.append(stroke)
    
    return strokes


def load_mask_cache(target_name: str) -> np.ndarray:
    """加载mask缓存数据"""
    base = Path(__file__).resolve().parent.parent
    cache_path = base / "caches" / "mask" / f"{target_name}.pt"
    
    if not cache_path.exists():
        raise RuntimeError(f"Mask cache file does not exist: {cache_path}")
    
    masks = torch.load(str(cache_path))
    if isinstance(masks, torch.Tensor):
        masks = masks.numpy()
    
    return masks


def extract_boundary_from_mask(mask: np.ndarray) -> np.ndarray:
    """
    通过腐蚀1个像素提取mask的边界轮廓
    
    Args:
        mask: 二值mask，形状为 (H, W)，值为0或1
    
    Returns:
        boundary_points: 边界点坐标，形状为 (N, 2)，格式为 [x, y]
    """
    # 确保mask是uint8类型
    if mask.dtype != np.uint8:
        mask_uint8 = (mask * 255).astype(np.uint8)
    else:
        mask_uint8 = mask
    
    # 腐蚀1个像素
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    eroded = cv2.erode(mask_uint8, kernel, iterations=1)
    
    # 边界 = 原始mask - 腐蚀后的mask
    boundary = mask_uint8 - eroded
    
    # 提取边界点的坐标
    y_coords, x_coords = np.where(boundary > 0)
    boundary_points = np.stack([x_coords, y_coords], axis=1).astype(np.float32)
    
    return boundary_points


def build_boundary_kdtrees(masks: np.ndarray) -> List[KDTree]:
    """
    为每一帧的mask构建边界KDTree
    
    Args:
        masks: mask数组，形状为 (N, H, W)
    
    Returns:
        kdtrees: KDTree列表，每个KDTree对应一帧的边界点
    """
    n_frames = masks.shape[0]
    kdtrees = []
    
    for i in tqdm(range(n_frames), desc="Building boundary KDTrees", unit=" frame(s)"):
        mask = masks[i]
        boundary_points = extract_boundary_from_mask(mask)
        
        if len(boundary_points) == 0:
            # 如果没有边界点，使用一个虚拟点
            boundary_points = np.array([[0.0, 0.0]])
        
        kdtree = KDTree(boundary_points)
        kdtrees.append(kdtree)
    
    return kdtrees


def compute_boundary_adherence_error(stroke: np.ndarray, boundary_kdtree: KDTree) -> float:
    """
    计算单个笔画的边界贴合误差（BAE）
    
    Args:
        stroke: 笔画点序列，形状为 (n, 2)，格式为 [x, y]
        boundary_kdtree: 边界点的KDTree
    
    Returns:
        bae: 边界贴合误差（平均距离平方）
    """
    if len(stroke) == 0:
        return 0.0
    
    # 查询每个笔画点到最近边界点的距离
    distances, _ = boundary_kdtree.query(stroke)
    
    # 计算距离平方的平均值
    squared_distances = distances ** 2
    bae = np.mean(squared_distances)
    
    return bae


def compute_bae_for_sequence(strokes: List[np.ndarray], boundary_kdtrees: List[KDTree]) -> np.ndarray:
    """
    计算笔画序列中每一帧的边界贴合误差
    
    Args:
        strokes: 笔画列表，每个元素形状为 (n, 2)
        boundary_kdtrees: 边界KDTree列表
    
    Returns:
        baes: 每帧的边界贴合误差，形状为 (n_frames,)
    """
    n_frames = len(strokes)
    baes = []
    
    for i in tqdm(range(n_frames), desc="Computing BAE", unit=" frame(s)"):
        stroke = strokes[i]
        kdtree = boundary_kdtrees[i]
        
        if stroke is None or len(stroke) == 0:
            baes.append(0.0)
        else:
            bae = compute_boundary_adherence_error(stroke, kdtree)
            baes.append(bae)
    
    return np.array(baes)


def plot_boundary_adherence_error(target_name: str):
    """绘制4种策略的边界贴合误差折线图"""
    
    print("=" * 60)
    print("边界贴合误差分析 (Boundary Adherence Error)")
    print("=" * 60)
    print(f"Target: {target_name}\n")
    
    # 加载mask并构建边界KDTree
    print("Loading masks and building boundary KDTrees...")
    masks = load_mask_cache(target_name)
    print(f"Loaded masks: {masks.shape}")
    boundary_kdtrees = build_boundary_kdtrees(masks)
    print(f"Built {len(boundary_kdtrees)} boundary KDTrees\n")
    
    # 定义4种策略
    strategies = [
        ("pure_flow_and_pure_snapping", "w/o dilation, w/o filtering", COLOR_FLOW_SNAPPED),
        ("dilated_flow_and_pure_snapping", "w/ dilation, w/o filtering", COLOR_DILATED_FLOW_SNAPPING),
        ("pure_flow_and_filtered_snapping", "w/o dilation, w/ filtering", COLOR_FLOW_FILTERED_SNAPPING),
        ("fitted", "w/ dilation, w/ filtering", COLOR_FITTED),
    ]
    
    # 创建图表
    plt.figure(figsize=(12, 6))
    
    # 为每种策略计算并绘制BAE
    for strategy_name, legend_label, color in strategies:
        print(f"\nProcessing strategy: {strategy_name}")
        
        # 加载笔画数据
        strokes = load_stroke_npy_files(target_name, strategy_name)
        print(f"  Loaded {len(strokes)} frames")
        
        # 计算每帧的边界贴合误差
        baes = compute_bae_for_sequence(strokes, boundary_kdtrees)
        print(f"  Computed BAE for {len(baes)} frames")
        print(f"  BAE range: [{baes.min():.4f}, {baes.max():.4f}]")
        print(f"  Mean BAE: {baes.mean():.4f}")
        
        # 绘制折线
        frame_indices = np.arange(len(strokes))
        plt.plot(frame_indices, baes, color=color, label=legend_label, 
                linewidth=2, marker='o', markersize=3)
    
    # 设置图表属性
    plt.xlabel('Frame Index', fontsize=12)
    plt.ylabel('Boundary Adherence Error (pixels²)', fontsize=12)
    plt.title(TARGET_NAME, fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图表
    base = Path(__file__).resolve().parent.parent
    output_dir = base / "results" / target_name / "strokes"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "boundary_adherence_error.png"
    plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
    print(f"\n图表已保存到: {output_path}")
    
    # 显示图表
    plt.show()


def main():
    """主函数"""
    plot_boundary_adherence_error(TARGET_NAME)


if __name__ == '__main__':
    main()
