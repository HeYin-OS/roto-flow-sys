"""
Mask腐蚀和光流处理测试脚本
1. 读取mask并进行形态学腐蚀（2-4像素）
2. 用腐蚀的mask截取光流
3. 对光流进行膨胀操作（约5像素）
"""
import sys
from pathlib import Path

# 添加项目根目录到Python路径
_file_path = Path(__file__).resolve()
_project_root = _file_path.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import cv2
import numpy as np
import torch
from torchvision.utils import flow_to_image
from tqdm import tqdm

from utils.path_utils import build_project_paths, get_target_name


def erode_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    对mask进行形态学腐蚀
    
    Args:
        mask: 二值mask，形状为(H, W)，值为0或1
        kernel_size: 腐蚀核大小（像素宽度）
    
    Returns:
        腐蚀后的mask
    """
    # 确保mask是uint8类型
    if mask.dtype != np.uint8:
        mask_uint8 = (mask * 255).astype(np.uint8)
    else:
        mask_uint8 = mask
    
    # 创建圆形结构元素
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size * 2 + 1, kernel_size * 2 + 1))
    
    # 执行腐蚀操作
    eroded = cv2.erode(mask_uint8, kernel, iterations=1)
    
    # 转换回0-1范围
    return (eroded / 255.0).astype(np.float32)


def dilate_flow(flow_masked: np.ndarray, mask_eroded: np.ndarray, original_flow: np.ndarray, dilation_radius: int) -> np.ndarray:
    """
    对光流进行膨胀操作
    
    逻辑：
    1. 先用mask截取光流（flow_masked）
    2. 膨胀截取的光流（在mask区域内传播光流向量）
    3. 空白的部分使用原先的光流填充
    
    Args:
        flow_masked: 被mask截取后的光流，形状为(H, W, 2)，mask外为0
        mask_eroded: 腐蚀后的二值mask，形状为(H, W)，值为0或1
        original_flow: 原始光流，形状为(H, W, 2)，用于填充空白部分
        dilation_radius: 膨胀半径（像素宽度）
    
    Returns:
        膨胀后的光流（空白部分使用原始光流值）
    """
    H, W = flow_masked.shape[:2]
    
    # 确保mask是二值的
    mask_binary = (mask_eroded > 0.5).astype(np.uint8)
    
    if mask_binary.sum() == 0:
        # 如果mask为空，直接返回原始光流
        return original_flow.copy()
    
    # 创建膨胀核
    kernel_size = dilation_radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # 对mask进行膨胀，找到需要填充的区域（膨胀区域）
    mask_dilated = cv2.dilate(mask_binary, kernel, iterations=1)
    fill_region = (mask_dilated > 0) & (mask_binary == 0)
    
    # 初始化结果：先对截取的光流进行膨胀
    dilated_flow = flow_masked.copy()
    
    # 创建inpaint mask（需要填充的区域：mask外但在膨胀区域内）
    inpaint_mask = fill_region.astype(np.uint8) * 255
    
    # 对每个通道分别进行膨胀填充
    for c in range(2):  # dx和dy两个通道
        flow_channel = flow_masked[:, :, c]
        
        # 计算有效值的范围（只在mask内）
        valid_mask = mask_binary > 0
        if valid_mask.sum() > 0:
            flow_min = flow_channel[valid_mask].min()
            flow_max = flow_channel[valid_mask].max()
            
            if flow_max > flow_min:
                # 归一化到0-255以便inpaint
                flow_normalized = ((flow_channel - flow_min) / (flow_max - flow_min) * 255).astype(np.uint8)
                
                # 使用inpaint填充（只填充fill_region）
                flow_filled = cv2.inpaint(flow_normalized, inpaint_mask, dilation_radius, cv2.INPAINT_TELEA)
                
                # 恢复原始范围
                flow_filled_float = flow_filled.astype(np.float32) / 255.0 * (flow_max - flow_min) + flow_min
                
                # 在填充区域使用膨胀后的值
                dilated_flow[:, :, c] = np.where(fill_region, flow_filled_float, flow_channel)
    
    # 找到空白部分（值为0或接近0的区域，且不在mask内）
    # 检查每个像素的magnitude
    flow_magnitude = np.sqrt(dilated_flow[:, :, 0]**2 + dilated_flow[:, :, 1]**2)
    blank_region = (flow_magnitude < 1e-6) & (mask_binary == 0)
    
    # 在空白部分使用原始光流填充
    if blank_region.any():
        blank_3d = np.stack([blank_region, blank_region], axis=-1)
        dilated_flow = np.where(blank_3d, original_flow, dilated_flow)
    
    return dilated_flow


def flow_to_rgb(flow: np.ndarray) -> np.ndarray:
    """
    将光流转换为RGB可视化图像（使用PyTorch的flow_to_image函数）
    与compute_RAFT.py中的格式保持一致
    
    Args:
        flow: 光流数组，形状为(H, W, 2)，值为[dx, dy]
    
    Returns:
        RGB图像，形状为(H, W, 3)，值范围0-255
    """
    # 将numpy数组转换为torch tensor
    # flow格式: (H, W, 2) -> 需要转换为 (2, H, W)
    flow_tensor = torch.from_numpy(flow).float()
    flow_tensor = flow_tensor.permute(2, 0, 1)  # (2, H, W)
    
    # 添加batch维度: (1, 2, H, W)
    flow_tensor = flow_tensor.unsqueeze(0)
    
    # 使用torchvision的flow_to_image函数
    # 返回形状: (1, 3, H, W)，值范围0-255
    rgb_tensor = flow_to_image(flow_tensor)
    
    # 转换为numpy并调整维度: (1, 3, H, W) -> (1, H, W, 3) -> (H, W, 3)
    rgb_tensor = rgb_tensor.permute(0, 2, 3, 1)  # (1, H, W, 3)
    rgb = rgb_tensor.squeeze(0).cpu().numpy()  # (H, W, 3)
    
    return rgb.astype(np.uint8)


def main():
    """主函数"""
    # 获取项目路径
    base, config_dir, cache_dir, frame_dir, result_dir, _ = build_project_paths()
    target_name = get_target_name(frame_dir)
    
    print("=" * 60)
    print("Mask腐蚀和光流处理工具")
    print("=" * 60)
    print(f"目标名称: {target_name}")
    print(f"缓存目录: {cache_dir}")
    print(f"结果目录: {result_dir}")
    print("=" * 60)
    
    # 1. 读取mask
    mask_cache_path = cache_dir / "mask" / f"{target_name}.pt"
    if not mask_cache_path.exists():
        raise FileNotFoundError(f"Mask缓存文件不存在: {mask_cache_path}")
    
    print(f"\n📦 读取mask: {mask_cache_path}")
    masks = torch.load(str(mask_cache_path))
    if isinstance(masks, torch.Tensor):
        masks = masks.numpy()
    
    print(f"✅ Mask形状: {masks.shape}, 数据类型: {masks.dtype}")
    
    # 2. 读取光流
    flow_cache_path = cache_dir / "flow" / f"{target_name}.pt"
    if not flow_cache_path.exists():
        raise FileNotFoundError(f"光流缓存文件不存在: {flow_cache_path}")
    
    print(f"\n📦 读取光流: {flow_cache_path}")
    flows = torch.load(str(flow_cache_path))
    if isinstance(flows, torch.Tensor):
        flows = flows.numpy()
    
    print(f"✅ 光流形状: {flows.shape}, 数据类型: {flows.dtype}")
    
    # 检查维度匹配（光流是N-1帧，mask是N帧）
    n_frames_mask = masks.shape[0]
    n_frames_flow = flows.shape[0]
    
    if n_frames_mask != n_frames_flow + 1:
        print(f"⚠️  警告: Mask帧数({n_frames_mask})与光流帧数({n_frames_flow}+1)不匹配")
        print(f"   将使用前{min(n_frames_mask, n_frames_flow + 1)}帧")
    
    # 使用较小的帧数
    n_frames = min(n_frames_mask - 1, n_frames_flow)
    
    # 3. 对mask进行腐蚀（2-4像素）
    erode_sizes = [2, 3, 4]  # 尝试不同的腐蚀大小
    erode_size = 3  # 默认使用3像素
    
    print(f"\n🔧 对mask进行形态学腐蚀（{erode_size}像素）...")
    masks_eroded = []
    
    for i in tqdm(range(n_frames_mask), desc="腐蚀mask", unit="帧"):
        mask = masks[i]
        mask_eroded = erode_mask(mask, erode_size)
        masks_eroded.append(mask_eroded)
    
    masks_eroded = np.array(masks_eroded)
    print(f"✅ 腐蚀完成，形状: {masks_eroded.shape}")
    
    # 保存mask-erode缓存
    mask_erode_cache_path = cache_dir / "mask-erode" / f"{target_name}.pt"
    mask_erode_cache_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n💾 保存mask-erode缓存到: {mask_erode_cache_path}")
    torch.save(torch.from_numpy(masks_eroded), str(mask_erode_cache_path))
    
    # 4. 保存腐蚀的mask可视化
    mask_erode_output_dir = result_dir / "mask-erode" / target_name
    mask_erode_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🎨 保存腐蚀的mask可视化到: {mask_erode_output_dir}")
    
    # 获取帧路径用于命名
    frame_paths = sorted([
        p for p in frame_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".png")
    ])
    
    for i in tqdm(range(len(masks_eroded)), desc="保存腐蚀mask", unit="帧"):
        mask_eroded = masks_eroded[i]
        mask_binary = (mask_eroded * 255).astype(np.uint8)
        
        # 使用帧文件名
        if i < len(frame_paths):
            frame_name = frame_paths[i].stem
            output_path = mask_erode_output_dir / f"{frame_name}_eroded.png"
        else:
            output_path = mask_erode_output_dir / f"{i:05d}_eroded.png"
        
        cv2.imwrite(str(output_path), mask_binary)
    
    # 5. 用腐蚀的mask截取光流
    print(f"\n✂️  用腐蚀的mask截取光流...")
    flows_masked = []
    
    for i in tqdm(range(n_frames), desc="截取光流", unit="帧"):
        flow = flows[i]  # 形状: (H, W, 2)
        # 使用第i帧的腐蚀mask（光流是从frame i到frame i+1）
        mask_eroded = masks_eroded[i]
        
        # 将mask扩展到光流的通道维度
        mask_3d = np.stack([mask_eroded, mask_eroded], axis=-1)  # (H, W, 2)
        
        # 应用mask（在mask外的地方设为0）
        flow_masked = flow * mask_3d
        flows_masked.append(flow_masked)
    
    flows_masked = np.array(flows_masked)
    print(f"✅ 光流截取完成，形状: {flows_masked.shape}")
    
    # 6. 对光流进行膨胀（约5像素）
    dilation_radius = 3
    print(f"\n🔧 对光流进行膨胀操作（{dilation_radius}像素）...")
    flows_dilated = []
    
    for i in tqdm(range(n_frames), desc="膨胀光流", unit="帧"):
        flow_masked = flows_masked[i]
        mask_eroded = masks_eroded[i]
        original_flow = flows[i]  # 原始光流，用于在mask外的地方
        
        # 对光流进行膨胀（依据腐蚀过的mask）
        flow_dilated = dilate_flow(flow_masked, mask_eroded, original_flow, dilation_radius)
        
        flows_dilated.append(flow_dilated)
    
    flows_dilated = np.array(flows_dilated)
    print(f"✅ 光流膨胀完成，形状: {flows_dilated.shape}")
    
    # 7. 保存光流可视化
    flow_dilate_output_dir = result_dir / "flow-dilate" / target_name
    flow_dilate_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🎨 保存光流可视化到: {flow_dilate_output_dir}")
    
    for i in tqdm(range(n_frames), desc="保存光流", unit="帧"):
        flow_dilated = flows_dilated[i]
        
        # 转换为RGB可视化
        flow_rgb = flow_to_rgb(flow_dilated)
        
        # 使用帧文件名
        if i < len(frame_paths):
            frame_name = frame_paths[i].stem
            output_path = flow_dilate_output_dir / f"{frame_name}_flow_dilated.png"
        else:
            output_path = flow_dilate_output_dir / f"{i:05d}_flow_dilated.png"
        
        # 直接保存RGB格式（与compute_RAFT.py保持一致）
        cv2.imwrite(str(output_path), flow_rgb)
    
    # 8. 保存处理后的光流数据（可选）
    flow_dilate_cache_path = cache_dir / "flow-dilate" / f"{target_name}.pt"
    flow_dilate_cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 保存处理后的光流到: {flow_dilate_cache_path}")
    torch.save(torch.from_numpy(flows_dilated), str(flow_dilate_cache_path))
    
    print("\n" + "=" * 60)
    print("✅ 处理完成！")
    print("=" * 60)
    print(f"腐蚀的mask可视化: {mask_erode_output_dir}")
    print(f"腐蚀的mask缓存: {mask_erode_cache_path}")
    print(f"处理后的光流可视化: {flow_dilate_output_dir}")
    print(f"处理后的光流缓存: {flow_dilate_cache_path}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
