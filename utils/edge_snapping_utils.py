"""工具函数模块：用于边缘吸附算法的辅助函数

本模块包含以下类别的工具函数：
- 图像处理：DPI获取、单位转换、RGB转灰度张量
- 核函数生成：高斯核、fDoG核
- 数组处理：锯齿数组打包、候选点切片
"""
from typing import List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch import Tensor


def get_dpi(img_url: str) -> Tuple[int, int]:
    """获取图像的DPI（每英寸点数）信息
    
    Args:
        img_url: 图像文件路径
        
    Returns:
        包含(x_dpi, y_dpi)的元组，如果图像没有DPI信息则返回默认值(96, 96)
    """
    img = Image.open(img_url)
    dpi = img.info.get('dpi', (96, 96))
    return dpi


def mm_to_pixels(mm: float, img_url: str) -> float:
    """将毫米单位转换为像素单位
    
    Args:
        mm: 毫米值
        img_url: 图像文件路径（用于获取DPI信息）
        
    Returns:
        对应的像素长度
    """
    inches = mm / 25.4
    dpi = get_dpi(img_url)
    pixel_length = dpi[0] * inches
    return pixel_length


def create_gaussian_kernel(size: int, sigma: float, direction: int):
    """创建高斯核
    
    Args:
        size: 核的大小
        sigma: 高斯核的标准差
        direction: 方向，0表示垂直方向，1表示水平方向
        
    Returns:
        高斯核数组，如果direction无效则返回None
    """
    kernel = cv2.getGaussianKernel(size, sigma, cv2.CV_32F)
    if direction == 0:
        return kernel
    elif direction == 1:
        return kernel.T
    return None


def create_fdog_kernel(size: int, sigma_c: float, sigma_s: float, rho: float, direction: int):
    """创建fDoG（频域差分高斯）核
    
    fDoG核是两个不同标准差的高斯核的差分，用于边缘检测。
    
    Args:
        size: 核的大小
        sigma_c: 中心高斯核的标准差
        sigma_s: 周围高斯核的标准差
        rho: 差分权重系数
        direction: 方向，0表示垂直方向，1表示水平方向
        
    Returns:
        fDoG核数组
    """
    kernel1 = create_gaussian_kernel(size, sigma_c, direction)
    kernel2 = create_gaussian_kernel(size, sigma_s, direction)
    dog_kernel = kernel1 - rho * kernel2
    return dog_kernel


def rgb_np_to_gray_tensor(device: torch.device, image_rgb_hwc: np.ndarray) -> Tensor:
    """将RGB格式的numpy数组转换为灰度张量
    
    Args:
        device: PyTorch设备（CPU或CUDA）
        image_rgb_hwc: RGB格式的图像数组，形状为[H, W, C]
        
    Returns:
        灰度张量，形状为[1, 1, H, W]，值域为[0, 1]
    """
    image_gray_hw = cv2.cvtColor(image_rgb_hwc.astype(np.float32), cv2.COLOR_RGB2GRAY) / 255.0
    image_tensor_gray_gpu = (
        torch.from_numpy(image_gray_hw)
        .unsqueeze(0).unsqueeze(0)
        .to(device, non_blocking=True)
        .contiguous()
    )  # shape: [1, 1, H, W]
    return image_tensor_gray_gpu


def pack_jagged_list_to_array(points_stroke_candidate: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """将锯齿数组（jagged array）打包为连续数组，并生成索引指针
    
    将多个不同长度的候选点数组打包成一个连续的二维数组，同时生成索引指针
    用于快速访问每个组的起始和结束位置。
    
    Args:
        points_stroke_candidate: 候选点列表，每个元素是一个形状为[N, 2]的numpy数组
        
    Returns:
        (points_flatten, index_ptr) 元组：
        - points_flatten: 打包后的候选点数组，形状为[总点数, 2]
        - index_ptr: 索引指针数组，index_ptr[i+1] - index_ptr[i] 表示第i组的点数
    """
    # 总笔画点数
    n_stroke_points = len(points_stroke_candidate)

    # 索引范围指针，index_ptr[i] 表示第i组的起始索引
    # index_ptr[i+1] - index_ptr[i] 表示第i组的总点数
    index_ptr = np.zeros(n_stroke_points + 1, dtype=np.int32)
    for i in range(n_stroke_points):
        index_ptr[i + 1] = index_ptr[i] + (0 if points_stroke_candidate[i] is None else len(points_stroke_candidate[i]))
    n_total_candidates = index_ptr[-1]

    # 扁平化候选点数组
    points_flatten = np.empty((n_total_candidates, 2), dtype=np.float32)
    i_current = 0
    for i in range(n_stroke_points):
        points = points_stroke_candidate[i]
        if points is None or len(points) == 0:
            continue
        # 确保是float格式
        points_float = points.astype(np.float32)
        # 构建扁平数组
        points_flatten[i_current: i_current + len(points), :] = points_float[:, :]
        i_current += len(points)

    return points_flatten, index_ptr


def slice_candidate_group_by_index(candidates_flatten_xy: np.ndarray, 
                                   flatten_index_ptr: np.ndarray, 
                                   i: int) -> Tuple[np.ndarray, np.ndarray]:
    """根据索引从扁平数组中切片出两个候选点组
    
    Args:
        candidates_flatten_xy: 扁平化的候选点数组
        flatten_index_ptr: 索引指针数组
        i: 组索引，将返回第i组和第i+1组
        
    Returns:
        (Qi_xy, Qj_xy) 元组，分别表示第i组和第i+1组的候选点
    """
    Ui = slice(flatten_index_ptr[i], flatten_index_ptr[i + 1])
    Uj = slice(flatten_index_ptr[i + 1], flatten_index_ptr[i + 2])
    Qi_xy = candidates_flatten_xy[Ui]
    Qj_xy = candidates_flatten_xy[Uj]
    return Qi_xy, Qj_xy

