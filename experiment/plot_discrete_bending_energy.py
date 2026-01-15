"""
绘制4种策略的离散弯曲能量（Discrete Bending Energy）折线图
"""
from pathlib import Path
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# ========== 配置 ==========
TARGET_NAME = "blackswan"  # 修改这里来指定不同的target

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


def compute_bending_angles(stroke: np.ndarray) -> np.ndarray:
    """
    计算笔画中每个顶点的转角 θᵢ
    
    根据公式：
    vᵢ = pᵢ₊₁ - pᵢ
    cos(θᵢ) = (vᵢ₋₁ · vᵢ) / (||vᵢ₋₁|| ||vᵢ||)
    θᵢ = arccos(clamp((vᵢ₋₁ · vᵢ) / (||vᵢ₋₁|| ||vᵢ||), -1, 1))
    
    Args:
        stroke: 笔画点序列，形状为 (n, 2)
    
    Returns:
        angles: 转角数组，形状为 (n-2,)，对应顶点 p₁, p₂, ..., pₙ₋₂
    """
    n = len(stroke)
    if n < 3:
        return np.array([])
    
    # 计算线段向量 vᵢ = pᵢ₊₁ - pᵢ
    vectors = np.diff(stroke, axis=0)  # 形状: (n-1, 2)
    
    # 计算每个线段的长度
    lengths = np.linalg.norm(vectors, axis=1)  # 形状: (n-1,)
    
    angles = []
    
    # 对于每个内部顶点 i (从1到n-2)
    for i in range(1, n - 1):
        v_prev = vectors[i - 1]  # vᵢ₋₁
        v_curr = vectors[i]      # vᵢ
        
        len_prev = lengths[i - 1]
        len_curr = lengths[i]
        
        # 避免除以零
        if len_prev < 1e-10 or len_curr < 1e-10:
            angles.append(0.0)
            continue
        
        # 计算点积
        dot_product = np.dot(v_prev, v_curr)
        
        # 计算 cos(θ)
        cos_theta = dot_product / (len_prev * len_curr)
        
        # Clamp到 [-1, 1] 范围内，避免数值误差
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        
        # 计算转角
        theta = np.arccos(cos_theta)
        
        angles.append(theta)
    
    return np.array(angles)


def compute_bending_energy(stroke: np.ndarray) -> float:
    """
    计算单个笔画的离散弯曲能量
    
    根据公式：
    E_bending = 1/(n-2) * Σ(θᵢ²) for i=1 to n-2
    
    Args:
        stroke: 笔画点序列，形状为 (n, 2)
    
    Returns:
        bending_energy: 弯曲能量值
    """
    angles = compute_bending_angles(stroke)
    
    if len(angles) == 0:
        return 0.0
    
    # 计算转角平方和的平均值
    squared_angles = angles ** 2
    bending_energy = np.mean(squared_angles)
    
    return bending_energy


def compute_bending_energies_for_sequence(strokes: List[np.ndarray]) -> np.ndarray:
    """
    计算笔画序列中每一帧的弯曲能量
    
    Args:
        strokes: 笔画列表，每个元素形状为 (n, 2)
    
    Returns:
        energies: 每帧的弯曲能量，形状为 (n_frames,)
    """
    energies = []
    
    for stroke in tqdm(strokes, desc="Computing bending energies", unit=" frame(s)"):
        if stroke is None or len(stroke) < 3:
            energies.append(0.0)
        else:
            energy = compute_bending_energy(stroke)
            energies.append(energy)
    
    return np.array(energies)


def plot_bending_energy(target_name: str):
    """绘制4种策略的离散弯曲能量折线图"""
    
    print("=" * 60)
    print("离散弯曲能量分析 (Discrete Bending Energy)")
    print("=" * 60)
    print(f"Target: {target_name}\n")
    
    # 定义4种策略
    strategies = [
        ("pure_flow_and_pure_snapping", "w/o dilation, w/o filtering", COLOR_FLOW_SNAPPED),
        ("dilated_flow_and_pure_snapping", "w/ dilation, w/o filtering", COLOR_DILATED_FLOW_SNAPPING),
        ("pure_flow_and_filtered_snapping", "w/o dilation, w/ filtering", COLOR_FLOW_FILTERED_SNAPPING),
        ("fitted", "w/ dilation, w/ filtering", COLOR_FITTED),
    ]
    
    # 创建图表
    plt.figure(figsize=(12, 6))
    
    # 为每种策略计算并绘制弯曲能量
    for strategy_name, legend_label, color in strategies:
        print(f"\nProcessing strategy: {strategy_name}")
        
        # 加载笔画数据
        strokes = load_stroke_npy_files(target_name, strategy_name)
        print(f"  Loaded {len(strokes)} frames")
        
        # 计算每帧的弯曲能量
        bending_energies = compute_bending_energies_for_sequence(strokes)
        print(f"  Computed bending energies for {len(bending_energies)} frames")
        print(f"  Bending energy range: [{bending_energies.min():.6f}, {bending_energies.max():.6f}]")
        print(f"  Mean bending energy: {bending_energies.mean():.6f}")
        
        # 绘制折线
        frame_indices = np.arange(len(strokes))
        plt.plot(frame_indices, bending_energies, color=color, label=legend_label, 
                linewidth=2, marker='o', markersize=3)
    
    # 设置图表属性
    plt.xlabel('Frame Index', fontsize=12)
    plt.ylabel('Discrete Bending Energy (rad²)', fontsize=12)
    plt.title(TARGET_NAME, fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图表
    base = Path(__file__).resolve().parent.parent
    output_dir = base / "results" / target_name / "strokes"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "discrete_bending_energy.png"
    plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
    print(f"\n图表已保存到: {output_path}")
    
    # 显示图表
    plt.show()


def main():
    """主函数"""
    plot_bending_energy(TARGET_NAME)


if __name__ == '__main__':
    main()
