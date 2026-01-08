from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List
import gc
import shutil

import cv2
import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from utils.edge_snapping import compute_all_candidates, EdgeSnappingConfig, local_snapping
from utils.kd_tree import BatchKDTree
from utils.yaml_reader import YamlUtil
from utils.gif_writer import save_fixed_length_gif_from_bgr
from utils.path_utils import build_project_paths as build_project_paths_from_utils, get_target_name
from torchvision.utils import flow_to_image

COLOR_ORIGIN = (255, 255, 0)  # Vivid Orange
COLOR_FLOW = (200, 130, 255)  # Soft Lavender
COLOR_SNAPPING = (0, 150, 255)  # Tech Blue
COLOR_FLOW_SNAPPED = (255, 100, 0)  # Orange Red
COLOR_FITTED = (50, 200, 50)  # Fresh Green
COLOR_DILATED_FLOW_SNAPPING = (255, 50, 150)  # Bright Pink - 膨胀光流+纯吸附
COLOR_FLOW_FILTERED_SNAPPING = (0, 255, 200)  # Bright Cyan - 纯光流+过滤吸附
THICKNESS = 2


@dataclass(frozen=True)
class ProjectPaths:
    """Absolute paths derived from the project root, shared across the module."""

    base: Path
    config: Path
    cache: Path
    stroke: Path
    debug: Path
    result: Path


@dataclass(frozen=True)
class StrokeEnvironment:
    """Static environment information loaded from configuration files."""

    paths: ProjectPaths
    frame_dir: Path
    target_name: str

    @property
    def stroke_dir(self) -> Path:
        """Directory used to cache stroke *.npy files for the current target."""

        return self.paths.stroke / self.target_name

    @property
    def flow_cache_file(self) -> Path:
        """Path to the optical-flow cache tensor for the current target."""

        return self.paths.cache / "flow" / f"{self.target_name}.pt"

    @property
    def mask_cache_file(self) -> Path:
        """Path to the original mask cache tensor for the current target."""

        return self.paths.cache / "mask" / f"{self.target_name}.pt"

    @property
    def intermediates_dir(self) -> Path:
        """Base directory for all intermediate results."""
        return self.paths.result / self.target_name / "intermediates"

    @property
    def final_dir(self) -> Path:
        """Base directory for final results."""
        return self.paths.result / self.target_name / "final"

    # === 光流相关中间结果 ===
    @property
    def flow_original_dir(self) -> Path:
        """原始光流可视化（未经任何处理）"""
        return self.intermediates_dir / "flow-original"

    @property
    def flow_crop_by_mask_dir(self) -> Path:
        """被mask剪裁后的光流"""
        return self.intermediates_dir / "flow-crop-by-mask"

    @property
    def flow_dilate_only_dir(self) -> Path:
        """仅膨胀处理的光流（不含原始光流）"""
        return self.intermediates_dir / "flow-dilate-only"

    @property
    def flow_dilate_final_dir(self) -> Path:
        """最终合成的光流结果（膨胀+原始）"""
        return self.intermediates_dir / "flow-dilate-final"

    # === Mask相关中间结果 ===
    @property
    def mask_original_dir(self) -> Path:
        """原始mask可视化"""
        return self.intermediates_dir / "mask-original"

    @property
    def mask_erode_dir(self) -> Path:
        """腐蚀后的mask（用于光流截取）"""
        return self.intermediates_dir / "mask-erode"

    @property
    def mask_dilate_dir(self) -> Path:
        """膨胀后的mask（用于过滤候选点）"""
        return self.intermediates_dir / "mask-dilate"

    # === 特征点相关中间结果 ===
    @property
    def salient_dir(self) -> Path:
        """全部候选特征点"""
        return self.intermediates_dir / "salient"

    @property
    def salient_filtered_dir(self) -> Path:
        """过滤后的候选特征点"""
        return self.intermediates_dir / "salient-filtered"


@dataclass
class ViewerState:
    """Mutable UI state controlled by keyboard shortcuts."""

    current_frame: int = 0
    current_stroke_index: int = 2  # 默认使用3号stroke（索引从0开始，3号对应索引2）
    show_origin: bool = False
    show_snapping: bool = False  # z键：纯吸附结果
    show_flow: bool = False  # x键：纯光流结果
    show_flow_snapped: bool = False  # c键：纯光流+纯吸附结果
    show_dilated_flow_snapping: bool = False  # v键：膨胀光流+纯吸附结果
    show_flow_filtered_snapping: bool = False  # b键：纯光流+过滤吸附结果
    show_fitted: bool = True  # n键：fitted结果（默认开启）


@dataclass
class StrokeBuffers:
    """Frame-aligned buffers containing different stroke propagation results."""

    flow: List[np.ndarray | None] = field(default_factory=list)
    snapping: List[np.ndarray | None] = field(default_factory=list)
    flow_snapped: List[np.ndarray | None] = field(default_factory=list)
    dilated_flow_snapping: List[np.ndarray | None] = field(default_factory=list)
    flow_filtered_snapping: List[np.ndarray | None] = field(default_factory=list)
    fitted: List[np.ndarray | None] = field(default_factory=list)

    def reset(self, n_frame: int) -> None:
        """Pre-allocate buffers with None placeholders for `n_frame` frames."""

        self.flow = [None] * n_frame
        self.snapping = [None] * n_frame
        self.flow_snapped = [None] * n_frame
        self.dilated_flow_snapping = [None] * n_frame
        self.flow_filtered_snapping = [None] * n_frame
        self.fitted = [None] * n_frame


@dataclass
class StrokeData:
    """In-memory data required to propagate strokes over all frames."""

    images_rgb: np.ndarray
    flow_nhw2: np.ndarray  # 处理后的光流（mask腐蚀+光流膨胀，用于fitted策略）
    flow_nhw2_unfiltered: np.ndarray  # 原始光流（从/cache/flow/读取，用于flow、flow_snapped、snapping策略）
    kd_tree: BatchKDTree  # 使用过滤后的候选点（mask腐蚀+膨胀过滤）
    kd_tree_unfiltered: BatchKDTree = None  # 使用未过滤的候选点（用于纯光流和纯吸附）
    original_stroke_length: float = None  # 第一帧原始stroke的长度（用于长度约束）


@dataclass
class RuntimeContext:
    """Aggregated structure bundling environment, data, buffers, and state."""

    env: StrokeEnvironment
    data: StrokeData
    strokes_library: List[np.ndarray]
    buffers: StrokeBuffers
    viewer: ViewerState


class KeyHandlerRegistry:
    """Simple registry used to map a keyboard key to its handler callback."""

    def __init__(self) -> None:
        self._handlers: Dict[int, Callable[[], bool]] = {}

    def register(self, key_char: str) -> Callable[[Callable[[], bool]], Callable[[], bool]]:
        """Decorator that associates a handler with a single-character key."""

        def decorator(func: Callable[[], bool]) -> Callable[[], bool]:
            self._handlers[ord(key_char)] = func
            return func

        return decorator

    def dispatch(self, key_code: int) -> bool:
        """Execute the handler corresponding to `key_code`, if any."""

        handler = self._handlers.get(key_code)
        return handler() if handler else False


def build_project_paths() -> ProjectPaths:
    """Construct an immutable `ProjectPaths` instance anchored at repo root."""

    base, config_dir, cache_dir, frame_dir, result_dir, stroke_dir = build_project_paths_from_utils()
    return ProjectPaths(
        base=base,
        config=config_dir,
        cache=cache_dir,
        stroke=stroke_dir,
        debug=base / "debug",  # debug目录独立于result_dir
        result=result_dir,  # results目录
    )


def load_environment() -> StrokeEnvironment:
    """Load configuration metadata and resolve frame directory for the target."""

    # 使用path_utils中的函数获取所有路径信息
    base, config_dir, cache_dir, frame_dir, result_dir, stroke_dir = build_project_paths_from_utils()
    target_name = get_target_name(frame_dir)

    # 构建ProjectPaths实例
    paths = ProjectPaths(
        base=base,
        config=config_dir,
        cache=cache_dir,
        stroke=stroke_dir,
        debug=base / "debug",  # debug目录独立于result_dir
        result=result_dir,  # results目录
    )

    return StrokeEnvironment(
        paths=paths,
        frame_dir=frame_dir,
        target_name=target_name,
    )


def get_frame_image_paths(env: StrokeEnvironment) -> List[Path]:
    """List and sort available frame paths for the current target."""

    return sorted(
        path for path in env.frame_dir.iterdir()
        if path.suffix.lower() in (".jpg", ".png")
    )


def read_strokes(env: StrokeEnvironment) -> List[np.ndarray]:
    """Load cached stroke numpy arrays, if present, in ascending order."""

    env.stroke_dir.mkdir(parents=True, exist_ok=True)
    strokes: List[np.ndarray] = []
    for i in range(1, 100):
        path = env.stroke_dir / f"stroke_{i:02d}.npy"
        if not path.exists():
            break
        stroke = np.load(str(path)).astype(np.float32)
        strokes.append(stroke)
        print(f"Loaded stroke from:  {path}, shape: {stroke.shape}, dtype: {stroke.dtype}")
    if strokes:
        print(f"Total strokes loaded: {len(strokes)}")
        for idx, stroke in enumerate(strokes):
            print(f"  Stroke {idx + 1}: shape={stroke.shape}, dtype={stroke.dtype}, num_points={len(stroke)}")
    return strokes


def read_images_batch(paths: List[Path], flag: Any) -> np.ndarray:
    """Read `paths` sequentially into a stacked numpy array."""

    out = []
    for i_path in tqdm(range(len(paths)), desc="Reading images:", unit=" image(s)"):
        img = cv2.imread(str(paths[i_path]), flag)
        out.append(img)
    return np.stack(out)


def read_optical_flow_cache(env: StrokeEnvironment) -> Tensor:
    """Load the precomputed optical-flow tensor for the configured target."""

    cache_path = env.flow_cache_file
    if cache_path.exists():
        return torch.load(str(cache_path))
    raise ValueError(f"Optical flow cache file does not exist: {cache_path}")


def read_mask_cache(env: StrokeEnvironment) -> np.ndarray:
    """Load the original mask cache tensor for the configured target."""

    cache_path = env.mask_cache_file
    if cache_path.exists():
        masks = torch.load(str(cache_path))
        if isinstance(masks, torch.Tensor):
            masks = masks.numpy()
        return masks
    raise ValueError(f"Mask cache file does not exist: {cache_path}")


def erode_mask(mask: np.ndarray, thickness: int) -> np.ndarray:
    """
    对mask进行形态学腐蚀
    
    Args:
        mask: 二值mask，形状为(H, W)，值为0或1
        thickness: 腐蚀厚度（像素），表示从边界向内侵蚀的像素数
    
    Returns:
        腐蚀后的mask
    
    Note:
        - thickness=1: 边界向内侵蚀约1像素（核大小3x3）
        - thickness=3: 边界向内侵蚀约3像素（核大小7x7）
        - thickness=5: 边界向内侵蚀约5像素（核大小11x11）
    """
    if thickness <= 0:
        return mask.copy()
    
    # 确保mask是uint8类型
    if mask.dtype != np.uint8:
        mask_uint8 = (mask * 255).astype(np.uint8)
    else:
        mask_uint8 = mask

    # 创建圆形结构元素
    # thickness表示腐蚀厚度，核大小 = 2*thickness + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thickness * 2 + 1, thickness * 2 + 1))

    # 执行腐蚀操作
    eroded = cv2.erode(mask_uint8, kernel, iterations=1)

    # 转换回0-1范围
    return (eroded / 255.0).astype(np.float32)


def dilate_mask(mask: np.ndarray, thickness: int) -> np.ndarray:
    """
    对mask进行形态学膨胀
    
    Args:
        mask: 二值mask，形状为(H, W)，值为0或1
        thickness: 膨胀厚度（像素），表示从边界向外扩展的像素数
    
    Returns:
        膨胀后的mask
    
    Note:
        - thickness=1: 边界向外扩展约1像素（核大小3x3）
        - thickness=3: 边界向外扩展约3像素（核大小7x7）
        - thickness=5: 边界向外扩展约5像素（核大小11x11）
    """
    if thickness <= 0:
        return mask.copy()
    
    # 确保mask是uint8类型
    if mask.dtype != np.uint8:
        mask_uint8 = (mask * 255).astype(np.uint8)
    else:
        mask_uint8 = mask

    # 创建圆形结构元素
    # thickness表示膨胀厚度，核大小 = 2*thickness + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thickness * 2 + 1, thickness * 2 + 1))

    # 执行膨胀操作
    dilated = cv2.dilate(mask_uint8, kernel, iterations=1)

    # 转换回0-1范围
    return (dilated / 255.0).astype(np.float32)


def filter_points_by_mask(points: np.ndarray, mask: np.ndarray, keep_inside: bool = False) -> np.ndarray:
    """
    使用mask过滤点
    
    Args:
        points: 点坐标数组，形状为(N, 2)，格式为[x, y]，值为float32
        mask: 二值mask，形状为(H, W)，值为0或1
        keep_inside: 如果为True，保留mask内部的点（mask值为1的点）；如果为False，保留mask外部的点（mask值为0的点）
    
    Returns:
        过滤后的点坐标数组，形状为(M, 2)，M <= N
    """
    if len(points) == 0:
        return points

    H, W = mask.shape

    # 将点坐标转换为整数索引
    x_coords = points[:, 0].astype(np.int32)
    y_coords = points[:, 1].astype(np.int32)

    # 边界检查
    valid_indices = (x_coords >= 0) & (x_coords < W) & (y_coords >= 0) & (y_coords < H)

    if not valid_indices.any():
        return np.array([], dtype=np.float32).reshape(0, 2)

    # 获取mask值（物体内部为1，外部为0）
    mask_values = np.zeros(len(points), dtype=np.float32)
    mask_values[valid_indices] = mask[y_coords[valid_indices], x_coords[valid_indices]]

    # 根据keep_inside参数决定保留哪些点
    if keep_inside:
        # 保留mask内部的点（mask值为1的点）
        filtered_indices = mask_values > 0.5
    else:
        # 保留mask外部的点（mask值为0的点）
        filtered_indices = mask_values < 0.5

    return points[filtered_indices]


def generate_salient_images(env: StrokeEnvironment, points_all_candidates: List[np.ndarray], height: int, width: int) -> None:
    """Dump unfiltered salient edge candidate maps for debugging or offline inspection."""

    work_dir = env.salient_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    
    # 清空目录中的文件（不删除目录）
    if work_dir.exists():
        for file_path in work_dir.iterdir():
            if file_path.is_file():
                file_path.unlink()
    
    for i_img in tqdm(range(len(points_all_candidates)), desc="Generating unfiltered salient point images:", unit=" image(s)"):
        canvas = np.zeros((height, width), np.uint8)
        canvas[points_all_candidates[i_img][:, 1].astype(np.int32), points_all_candidates[i_img][:, 0].astype(np.int32)] = 255
        file_path = work_dir / f"{i_img:03d}.png"
        cv2.imwrite(str(file_path), canvas)


def generate_salient_filtered_images(env: StrokeEnvironment, points_all_candidates: List[np.ndarray], height: int, width: int) -> None:
    """Dump filtered salient edge candidate maps for debugging or offline inspection."""

    work_dir = env.salient_filtered_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    
    # 清空目录中的文件（不删除目录）
    if work_dir.exists():
        for file_path in work_dir.iterdir():
            if file_path.is_file():
                file_path.unlink()
    
    for i_img in tqdm(range(len(points_all_candidates)), desc="Generating filtered salient point images:", unit=" image(s)"):
        canvas = np.zeros((height, width), np.uint8)
        canvas[points_all_candidates[i_img][:, 1].astype(np.int32), points_all_candidates[i_img][:, 0].astype(np.int32)] = 255
        file_path = work_dir / f"{i_img:03d}.png"
        cv2.imwrite(str(file_path), canvas)


def flow_batch_to_bgr_images(flows: np.ndarray) -> np.ndarray:
    """
    批量将光流转换为BGR可视化图像（使用torchvision的flow_to_image函数）
    这是统一的光流可视化标准函数，确保颜色空间正确
    
    Args:
        flows: 光流数组，形状为(N, H, W, 2)，值为[dx, dy]
    
    Returns:
        BGR图像数组，形状为(N, H, W, 3)，值范围0-255，uint8类型
    """
    # 转换为torch tensor: (N, H, W, 2) -> (N, 2, H, W)
    flow_tensor = torch.from_numpy(flows).float()
    flow_tensor = flow_tensor.permute(0, 3, 1, 2)  # (N, 2, H, W)
    
    # 使用torchvision的flow_to_image函数
    # 输入: (N, 2, H, W) - 光流的 x, y 分量
    # 输出: (N, 3, H, W) - RGB图像，dtype=uint8，值范围0-255
    rgb_tensor = flow_to_image(flow_tensor)
    
    # 转换为numpy: (N, 3, H, W) -> (N, H, W, 3)
    # flow_to_image 已经返回 uint8 类型，范围 0-255
    rgb_images = rgb_tensor.permute(0, 2, 3, 1).cpu().numpy()  # (N, H, W, 3)
    
    # 批量转换RGB为BGR（OpenCV格式）
    bgr_images = np.empty_like(rgb_images)
    for i in range(rgb_images.shape[0]):
        bgr_images[i] = cv2.cvtColor(rgb_images[i], cv2.COLOR_RGB2BGR)
    
    return bgr_images


def save_flow_images_to_dir(flow_images_bgr: np.ndarray, work_dir: Path, desc: str) -> None:
    """
    统一的光流图像保存函数
    
    Args:
        flow_images_bgr: BGR格式的光流图像，形状为(N, H, W, 3)，uint8类型
        work_dir: 保存目录
        desc: 进度条描述
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    
    # 清空目录中的文件
    if work_dir.exists():
        for file_path in work_dir.iterdir():
            if file_path.is_file():
                file_path.unlink()
    
    n_frames = flow_images_bgr.shape[0]
    
    for i in tqdm(range(n_frames), desc=desc, unit=" image(s)"):
        file_path = work_dir / f"{i:05d}.png"
        cv2.imwrite(str(file_path), flow_images_bgr[i])


def generate_flow_original_images(env: StrokeEnvironment, flows: np.ndarray) -> None:
    """
    生成原始光流的可视化图像（未经任何处理）
    
    Args:
        env: 环境配置
        flows: 原始光流数组，形状为(N, H, W, 2)
    """
    # 批量转换光流为BGR图像
    flow_images_bgr = flow_batch_to_bgr_images(flows)
    
    # 保存到目录
    save_flow_images_to_dir(flow_images_bgr, env.flow_original_dir, "Generating original flow images:")


def generate_mask_original_images(env: StrokeEnvironment, masks: np.ndarray) -> None:
    """
    生成原始mask的可视化图像
    
    Args:
        env: 环境配置
        masks: 原始mask数组，形状为(N, H, W)，值为0-1
    """
    work_dir = env.mask_original_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    
    # 清空目录中的文件
    if work_dir.exists():
        for file_path in work_dir.iterdir():
            if file_path.is_file():
                file_path.unlink()
    
    n_frames = masks.shape[0]
    
    for i in tqdm(range(n_frames), desc="Generating original mask images:", unit=" image(s)"):
        mask = masks[i]  # 形状: (H, W)
        
        # 转换为uint8 (0-255)
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # 创建彩色版本（绿色）用于更好的可视化
        mask_colored = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        mask_colored[:, :, 1] = mask_uint8  # 绿色通道
        
        # 保存灰度图像
        file_path = work_dir / f"{i:05d}.png"
        cv2.imwrite(str(file_path), mask_uint8)
        
        # 保存彩色图像
        file_path_colored = work_dir / f"{i:05d}_colored.png"
        cv2.imwrite(str(file_path_colored), mask_colored)


def generate_mask_erode_images(env: StrokeEnvironment, masks_eroded: np.ndarray, masks_original: np.ndarray = None) -> None:
    """
    生成腐蚀后mask的可视化图像（包含对比图和差异图）
    
    Args:
        env: 环境配置
        masks_eroded: 腐蚀后的mask数组，形状为(N, H, W)，值为0-1
        masks_original: 原始mask数组，形状为(N, H, W)，值为0-1（用于生成对比图）
    """
    work_dir = env.mask_erode_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    
    # 清空目录中的文件
    if work_dir.exists():
        for file_path in work_dir.iterdir():
            if file_path.is_file():
                file_path.unlink()
    
    n_frames = masks_eroded.shape[0]
    
    for i in tqdm(range(n_frames), desc="Generating eroded mask images:", unit=" image(s)"):
        mask = masks_eroded[i]  # 形状: (H, W)
        
        # 转换为uint8 (0-255)
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # 保存灰度图像
        file_path = work_dir / f"{i:05d}.png"
        cv2.imwrite(str(file_path), mask_uint8)
        
        # 如果提供了原始mask，生成对比图和差异图
        if masks_original is not None and i < masks_original.shape[0]:
            mask_orig = masks_original[i]
            mask_orig_uint8 = (mask_orig * 255).astype(np.uint8)
            
            # 创建对比图（原始=红色，腐蚀后=绿色，重叠=黄色）
            comparison = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            comparison[:, :, 2] = mask_orig_uint8  # 红色：原始mask
            comparison[:, :, 1] = mask_uint8  # 绿色：腐蚀后mask
            # 重叠部分会显示为黄色（红+绿）
            
            # 保存对比图
            file_path_comp = work_dir / f"{i:05d}_comparison.png"
            cv2.imwrite(str(file_path_comp), comparison)
            
            # 创建差异图（显示被腐蚀掉的区域）
            diff = mask_orig_uint8 - mask_uint8  # 被腐蚀掉的部分
            diff_colored = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            diff_colored[:, :, 0] = diff  # 蓝色：被腐蚀掉的边界
            diff_colored[:, :, 1] = mask_uint8  # 绿色：腐蚀后的mask
            
            # 保存差异图
            file_path_diff = work_dir / f"{i:05d}_diff.png"
            cv2.imwrite(str(file_path_diff), diff_colored)


def generate_mask_dilate_images(env: StrokeEnvironment, masks_dilated: np.ndarray, masks_original: np.ndarray = None) -> None:
    """
    生成膨胀后mask的可视化图像（包含对比图和差异图）
    
    Args:
        env: 环境配置
        masks_dilated: 膨胀后的mask数组，形状为(N, H, W)，值为0-1
        masks_original: 原始mask数组，形状为(N, H, W)，值为0-1（用于生成对比图）
    """
    work_dir = env.mask_dilate_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    
    # 清空目录中的文件
    if work_dir.exists():
        for file_path in work_dir.iterdir():
            if file_path.is_file():
                file_path.unlink()
    
    n_frames = masks_dilated.shape[0]
    
    for i in tqdm(range(n_frames), desc="Generating dilated mask images:", unit=" image(s)"):
        mask = masks_dilated[i]  # 形状: (H, W)
        
        # 转换为uint8 (0-255)
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # 保存灰度图像
        file_path = work_dir / f"{i:05d}.png"
        cv2.imwrite(str(file_path), mask_uint8)
        
        # 如果提供了原始mask，生成对比图和差异图
        if masks_original is not None and i < masks_original.shape[0]:
            mask_orig = masks_original[i]
            mask_orig_uint8 = (mask_orig * 255).astype(np.uint8)
            
            # 创建对比图（原始=红色，膨胀后=绿色，重叠=黄色）
            comparison = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            comparison[:, :, 2] = mask_orig_uint8  # 红色：原始mask
            comparison[:, :, 1] = mask_uint8  # 绿色：膨胀后mask
            # 重叠部分会显示为黄色（红+绿）
            
            # 保存对比图
            file_path_comp = work_dir / f"{i:05d}_comparison.png"
            cv2.imwrite(str(file_path_comp), comparison)
            
            # 创建差异图（显示膨胀增加的区域）
            diff = mask_uint8 - mask_orig_uint8  # 膨胀增加的部分
            diff_colored = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            diff_colored[:, :, 0] = diff  # 蓝色：膨胀增加的边界
            diff_colored[:, :, 2] = mask_orig_uint8  # 红色：原始mask
            
            # 保存差异图
            file_path_diff = work_dir / f"{i:05d}_diff.png"
            cv2.imwrite(str(file_path_diff), diff_colored)


def generate_mask_triple_comparison(env: StrokeEnvironment, masks_original: np.ndarray, 
                                   masks_eroded: np.ndarray, masks_dilated: np.ndarray) -> None:
    """
    生成三合一对比图：原始、腐蚀、膨胀同时显示
    
    Args:
        env: 环境配置
        masks_original: 原始mask数组，形状为(N, H, W)
        masks_eroded: 腐蚀后mask数组，形状为(N, H, W)
        masks_dilated: 膨胀后mask数组，形状为(N, H, W)
    """
    work_dir = env.intermediates_dir / "mask-comparison"
    work_dir.mkdir(parents=True, exist_ok=True)
    
    # 清空目录中的文件
    if work_dir.exists():
        for file_path in work_dir.iterdir():
            if file_path.is_file():
                file_path.unlink()
    
    n_frames = min(masks_original.shape[0], masks_eroded.shape[0], masks_dilated.shape[0])
    
    for i in tqdm(range(n_frames), desc="Generating mask comparison images:", unit=" image(s)"):
        mask_orig = masks_original[i]
        mask_erode = masks_eroded[i]
        mask_dilate = masks_dilated[i]
        
        H, W = mask_orig.shape
        
        # 创建三合一彩色对比图
        # 蓝色：腐蚀后的区域（最小）
        # 红色：原始mask区域（中等）
        # 绿色：膨胀后的区域（最大）
        comparison = np.zeros((H, W, 3), dtype=np.uint8)
        
        # 蓝色：腐蚀后mask
        comparison[:, :, 0] = (mask_erode * 255).astype(np.uint8)
        
        # 红色：原始mask
        comparison[:, :, 2] = (mask_orig * 255).astype(np.uint8)
        
        # 绿色：膨胀后mask
        comparison[:, :, 1] = (mask_dilate * 255).astype(np.uint8)
        
        # 保存对比图
        file_path = work_dir / f"{i:05d}_triple.png"
        cv2.imwrite(str(file_path), comparison)
        
        # 创建边界对比图（只显示边界差异）
        edge_comparison = np.zeros((H, W, 3), dtype=np.uint8)
        
        # 红色：被腐蚀掉的边界（原始 - 腐蚀）
        eroded_boundary = ((mask_orig - mask_erode) * 255).astype(np.uint8)
        edge_comparison[:, :, 2] = eroded_boundary
        
        # 绿色：膨胀增加的边界（膨胀 - 原始）
        dilated_boundary = ((mask_dilate - mask_orig) * 255).astype(np.uint8)
        edge_comparison[:, :, 1] = dilated_boundary
        
        # 蓝色：原始mask边界
        orig_uint8 = (mask_orig * 255).astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        orig_edge = cv2.morphologyEx(orig_uint8, cv2.MORPH_GRADIENT, kernel)
        edge_comparison[:, :, 0] = orig_edge
        
        # 保存边界对比图
        file_path_edge = work_dir / f"{i:05d}_edges.png"
        cv2.imwrite(str(file_path_edge), edge_comparison)


def generate_flow_crop_by_mask_images(env: StrokeEnvironment, flows: np.ndarray, masks_eroded: np.ndarray) -> np.ndarray:
    """
    生成被mask剪裁后的光流可视化图像
    
    Args:
        env: 环境配置
        flows: 原始光流数组，形状为(N, H, W, 2)
        masks_eroded: 腐蚀后的mask数组，形状为(N, H, W)
    
    Returns:
        flow_cropped: 裁剪后的光流数组，形状为(N, H, W, 2)
    """
    # 光流帧数是N-1帧，mask帧数是N帧
    n_frames = min(flows.shape[0], masks_eroded.shape[0])
    
    # 创建裁剪后的光流数组
    flow_cropped = np.zeros_like(flows[:n_frames])
    
    for i in range(n_frames):
        # 将mask扩展到光流的通道维度
        mask_3d = np.stack([masks_eroded[i], masks_eroded[i]], axis=-1)  # (H, W, 2)
        
        # 应用mask（在mask外的地方设为0）
        flow_cropped[i] = flows[i] * mask_3d
    
    # 批量转换光流为BGR图像并保存
    flow_images_bgr = flow_batch_to_bgr_images(flow_cropped)
    save_flow_images_to_dir(flow_images_bgr, env.flow_crop_by_mask_dir, "Generating flow cropped by mask images:")
    
    return flow_cropped


def dilate_flow_binary(flow: np.ndarray, thickness: int) -> np.ndarray:
    """
    将光流按照二值对待进行膨胀
    
    Args:
        flow: 光流数组，形状为(H, W, 2)，值为[dx, dy]
        thickness: 膨胀厚度（像素），表示从有效光流区域向外扩展的像素数
    
    Returns:
        膨胀后的光流，形状为(H, W, 2)
    
    Note:
        - thickness=1: 向外扩展约1像素（核大小3x3）
        - thickness=3: 向外扩展约3像素（核大小7x7）
        - thickness=5: 向外扩展约5像素（核大小11x11）
    """
    if thickness <= 0:
        return flow
    
    H, W = flow.shape[:2]
    
    # 创建二值mask：有光流的地方为1，无光流的地方为0
    flow_magnitude = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
    binary_mask = (flow_magnitude > 1e-6).astype(np.uint8)
    
    # 创建圆形结构元素
    # thickness表示膨胀厚度，核大小 = 2*thickness + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thickness * 2 + 1, thickness * 2 + 1))
    
    # 对二值mask进行膨胀
    dilated_mask = cv2.dilate(binary_mask, kernel, iterations=1)
    
    # 对于新膨胀出的区域，使用最近邻插值填充光流值
    # 首先找到新膨胀出的区域
    new_region = (dilated_mask > 0) & (binary_mask == 0)
    
    # 如果没有新区域，直接返回原光流
    if not new_region.any():
        return flow
    
    # 对于新区域的每个点，使用最近邻的有效光流值
    # 使用距离变换找到最近的有效光流点
    from scipy.ndimage import distance_transform_edt
    
    # 创建输出光流
    flow_dilated = flow.copy()
    
    # 对x和y分量分别处理
    for channel in range(2):
        # 获取有效光流的mask
        valid_mask = binary_mask > 0
        
        # 使用距离变换找到最近的有效点
        indices = distance_transform_edt(~valid_mask, return_distances=False, return_indices=True)
        
        # 对新区域使用最近邻插值
        flow_dilated[new_region, channel] = flow[indices[0][new_region], indices[1][new_region], channel]
    
    return flow_dilated


def generate_flow_dilate_only_images(env: StrokeEnvironment, flows_cropped: np.ndarray, thickness: int) -> np.ndarray:
    """
    对裁剪后的光流进行膨胀，并生成可视化图像（仅显示膨胀部分）
    
    Args:
        env: 环境配置
        flows_cropped: 裁剪后的光流数组，形状为(N, H, W, 2)
        thickness: 膨胀厚度（像素）
    
    Returns:
        flow_dilated: 膨胀后的光流数组，形状为(N, H, W, 2)
    """
    n_frames = flows_cropped.shape[0]
    flow_dilated = np.zeros_like(flows_cropped)
    
    # 对每帧光流进行膨胀
    for i in tqdm(range(n_frames), desc=f"Dilating flow ({thickness}px thickness):", unit=" frame(s)"):
        flow_dilated[i] = dilate_flow_binary(flows_cropped[i], thickness)
    
    # 批量转换光流为BGR图像并保存
    flow_images_bgr = flow_batch_to_bgr_images(flow_dilated)
    save_flow_images_to_dir(flow_images_bgr, env.flow_dilate_only_dir, "Saving dilated flow images:")
    
    return flow_dilated


def generate_flow_dilate_final_images(env: StrokeEnvironment, flows_original: np.ndarray, flows_dilated: np.ndarray) -> np.ndarray:
    """
    将膨胀后的光流覆盖回原始光流，并生成可视化图像（最终合成结果）
    
    Args:
        env: 环境配置
        flows_original: 原始光流数组，形状为(N, H, W, 2)
        flows_dilated: 膨胀后的光流数组，形状为(N, H, W, 2)
    
    Returns:
        flow_final: 最终的光流数组，形状为(N, H, W, 2)
    """
    n_frames = min(flows_original.shape[0], flows_dilated.shape[0])
    flow_final = flows_original[:n_frames].copy()
    
    # 合并膨胀光流到原始光流
    for i in range(n_frames):
        # 创建mask：膨胀光流有值的地方
        flow_dil_magnitude = np.sqrt(flows_dilated[i, :, :, 0]**2 + flows_dilated[i, :, :, 1]**2)
        has_dilated_flow = flow_dil_magnitude > 1e-6
        
        # 将膨胀光流覆盖回原始光流（只在有值的地方覆盖）
        mask_3d = np.stack([has_dilated_flow, has_dilated_flow], axis=-1)
        flow_final[i] = np.where(mask_3d, flows_dilated[i], flows_original[i])
    
    # 批量转换光流为BGR图像并保存
    flow_images_bgr = flow_batch_to_bgr_images(flow_final)
    save_flow_images_to_dir(flow_images_bgr, env.flow_dilate_final_dir, "Generating final dilated flow images:")
    
    return flow_final


# def generate_salient_stroke_images(env: StrokeEnvironment, points_stroke_candidates: List[np.ndarray], height: int, width: int, i_frame: int) -> None:
#     """Dump stroke-wise salient candidates for a particular frame."""
#
#     canvas = np.zeros((height, width), np.uint8)
#     for i_group in range(len(points_stroke_candidates)):
#         canvas[points_stroke_candidates[i_group][:, 1], points_stroke_candidates[i_group][:, 0]] = 255
#     work_dir = env.salient_stroke_dir
#     work_dir.mkdir(parents=True, exist_ok=True)
#     file_path = work_dir / f"{i_frame:03d}.jpg"
#     cv2.imwrite(str(file_path), canvas)


def rgb_to_bgr(color: tuple) -> tuple:
    """Convert an RGB tuple/list to BGR order."""
    return color[::-1]


def generate_prediction_stroke_on_0(buffers: StrokeBuffers, data: StrokeData, stroke_0: np.ndarray) -> None:
    """Snap the reference stroke onto frame-0 edges prior to propagation."""

    # 保存原始拓扑修复设置
    original_topology_fix = EdgeSnappingConfig.enable_topology_fix

    # 生成fitted结果（使用过滤后的候选点，启用拓扑修复）
    EdgeSnappingConfig.enable_topology_fix = True
    points_stroke_candidate = data.kd_tree.query_batch(
        0,
        stroke_0,
        EdgeSnappingConfig.r_s,
    )
    stroke_0_snapped = local_snapping(
        stroke_0,
        data.images_rgb[0],
        points_stroke_candidate,
        previous_snapped_stroke=None,  # 第一帧没有前一帧
        original_stroke_length=data.original_stroke_length,  # 传递原始长度用于长度约束
    )
    buffers.fitted[0] = stroke_0_snapped.astype(np.float32)

    # 生成snapping结果（使用未过滤的候选点，禁用拓扑修复）
    EdgeSnappingConfig.enable_topology_fix = False
    points_stroke_candidate_unfiltered = data.kd_tree_unfiltered.query_batch(
        0,
        stroke_0,
        EdgeSnappingConfig.r_s,
    )
    stroke_0_snapping = local_snapping(
        stroke_0,
        data.images_rgb[0],
        points_stroke_candidate_unfiltered,  # 使用未过滤的候选点
        previous_snapped_stroke=None,  # 第一帧没有前一帧
        original_stroke_length=data.original_stroke_length,  # 传递原始长度用于长度约束
    )
    buffers.snapping[0] = stroke_0_snapping.astype(np.float32)

    # 生成flow结果（第0帧直接复制原始测试笔画，不进行吸附）
    buffers.flow[0] = stroke_0.astype(np.float32)

    # 生成flow_snapped结果（第0帧使用未过滤点集的吸附结果，禁用拓扑修复）
    buffers.flow_snapped[0] = stroke_0_snapping.astype(np.float32)

    # 生成dilated_flow_snapping结果（第0帧使用未过滤点集的吸附结果，禁用拓扑修复）
    buffers.dilated_flow_snapping[0] = stroke_0_snapping.astype(np.float32)

    # 生成flow_filtered_snapping结果（第0帧使用过滤后候选点的吸附结果，禁用拓扑修复）
    stroke_0_filtered_snapping = local_snapping(
        stroke_0,
        data.images_rgb[0],
        points_stroke_candidate,  # 使用过滤后的候选点
        previous_snapped_stroke=None,
        original_stroke_length=data.original_stroke_length,
    )
    buffers.flow_filtered_snapping[0] = stroke_0_filtered_snapping.astype(np.float32)

    # 恢复原始拓扑修复设置
    EdgeSnappingConfig.enable_topology_fix = original_topology_fix
    
    # 输出第0帧所有预测结果的形状
    print("\n" + "=" * 60)
    print("Frame 0 预测结果形状:")
    print("=" * 60)
    print(f"  Original stroke shape: {stroke_0.shape}, dtype: {stroke_0.dtype}")
    print(f"  Pure snapping shape: {buffers.snapping[0].shape}, dtype: {buffers.snapping[0].dtype}")
    print(f"  Pure flow shape: {buffers.flow[0].shape}, dtype: {buffers.flow[0].dtype}")
    print(f"  Pure flow + pure snapping shape: {buffers.flow_snapped[0].shape}, dtype: {buffers.flow_snapped[0].dtype}")
    print(f"  Dilated flow + pure snapping shape: {buffers.dilated_flow_snapping[0].shape}, dtype: {buffers.dilated_flow_snapping[0].dtype}")
    print(f"  Pure flow + filtered snapping shape: {buffers.flow_filtered_snapping[0].shape}, dtype: {buffers.flow_filtered_snapping[0].dtype}")
    print(f"  Fitted shape: {buffers.fitted[0].shape}, dtype: {buffers.fitted[0].dtype}")
    print("=" * 60 + "\n")


def generate_snapping_prediction(
    i_frame: int,
    i: int,
    stroke_copied: np.ndarray,
    buffers: StrokeBuffers,
    data: StrokeData,
) -> np.ndarray:
    """
    生成当前帧的snapping预测结果。
    使用自身的数据进行传播，如果是第一帧则使用第0帧的结果。
    
    Args:
        i_frame: 当前帧索引
        i: 循环索引（用于判断是否是第一帧）
        stroke_copied: 从前一帧复制的fitted stroke（仅用于查询候选点）
        buffers: 存储预测结果的缓冲区
        data: 包含图像和KD树的数据
    
    Returns:
        当前帧的snapping预测结果
    """
    # 使用自身的数据进行传播
    # 如果是第一帧（i_frame=1），使用第0帧的snapping结果
    previous_snapping = buffers.snapping[i_frame - 1]
    if previous_snapping is None:
        raise RuntimeError(f"Missing snapping stroke for frame {i_frame - 1}")

    # 使用未过滤的候选点进行纯吸附传播
    # 使用前一帧的snapping位置查询候选点
    points_stroke_candidate_unfiltered = data.kd_tree_unfiltered.query_batch(
        i_frame,
        previous_snapping,
        EdgeSnappingConfig.r_s,
    )

    # 获取前一帧的snapping结果用于速度约束
    previous_snapped_stroke = buffers.snapping[i_frame - 1] if i > 0 else None
    
    # 禁用拓扑修复
    original_topology_fix = EdgeSnappingConfig.enable_topology_fix
    EdgeSnappingConfig.enable_topology_fix = False
    
    stroke_snapping = local_snapping(
        previous_snapping,
        data.images_rgb[i_frame],
        points_stroke_candidate_unfiltered,  # 使用未过滤的候选点
        previous_snapped_stroke=previous_snapped_stroke,
        original_stroke_length=data.original_stroke_length,  # 传递原始长度用于长度约束
    )
    
    # 恢复拓扑修复设置
    EdgeSnappingConfig.enable_topology_fix = original_topology_fix
    
    return stroke_snapping


def generate_flow_prediction(
    i_frame: int,
    i: int,
    stroke_copied: np.ndarray,
    buffers: StrokeBuffers,
    data: StrokeData,
    H: int,
    W: int,
) -> np.ndarray:
    """
    生成当前帧的flow预测结果。
    使用自身的数据进行传播，如果是第一帧则使用第0帧的结果。
    
    Args:
        i_frame: 当前帧索引
        i: 循环索引（用于判断是否是第一帧）
        stroke_copied: 从前一帧复制的fitted stroke（未使用，保留以保持接口一致）
        buffers: 存储预测结果的缓冲区
        data: 包含光流数据的数据
        H: 图像高度
        W: 图像宽度
    
    Returns:
        当前帧的flow预测结果
    """
    # 使用自身的数据进行传播
    # 如果是第一帧（i_frame=1），使用第0帧的flow结果
    previous_flow = buffers.flow[i_frame - 1]
    if previous_flow is None:
        raise RuntimeError(f"Missing flow stroke for frame {i_frame - 1}")

    # 光流传播：从 frame i-1 到 frame i
    # flow_nhw2_unfiltered[i_frame - 1] 存储的是从 frame i-1 到 frame i 的未膨胀光流
    # 格式：[H, W, 2]，其中 [:, :, 0] 是 x 方向，[:, :, 1] 是 y 方向
    x, y = previous_flow[:, 0], previous_flow[:, 1]

    # 边界检查：确保索引在有效范围内
    x_clipped = np.clip(x.astype(np.int32), 0, W - 1)
    y_clipped = np.clip(y.astype(np.int32), 0, H - 1)
    flow_vectors = data.flow_nhw2_unfiltered[i_frame - 1, y_clipped, x_clipped]  # [N, 2] 格式：[dx, dy]
    stroke_flow = previous_flow + flow_vectors
    
    return stroke_flow


def generate_flow_snapped_prediction(
    i_frame: int,
    i: int,
    stroke_copied: np.ndarray,
    buffers: StrokeBuffers,
    data: StrokeData,
    H: int,
    W: int,
) -> np.ndarray:
    """
    生成当前帧的flow_snapped预测结果。
    先进行光流传播，再进行吸附（使用未过滤的候选点）。
    使用自身的数据进行传播，如果是第一帧则使用第0帧的结果。
    
    Args:
        i_frame: 当前帧索引
        i: 循环索引（用于判断是否是第一帧）
        stroke_copied: 从前一帧复制的fitted stroke（未使用，保留以保持接口一致）
        buffers: 存储预测结果的缓冲区
        data: 包含图像、光流和KD树的数据
        H: 图像高度
        W: 图像宽度
    
    Returns:
        当前帧的flow_snapped预测结果
    """
    # 使用自身的数据进行传播
    # 如果是第一帧（i_frame=1），使用第0帧的flow_snapped结果
    previous_flow_snapped = buffers.flow_snapped[i_frame - 1]
    if previous_flow_snapped is None:
        raise RuntimeError(f"Missing flow_snapped stroke for frame {i_frame - 1}")

    # 第一步：光流传播（使用未膨胀的光流）
    # flow_nhw2_unfiltered[i_frame - 1] 存储的是从 frame i-1 到 frame i 的未膨胀光流
    x, y = previous_flow_snapped[:, 0], previous_flow_snapped[:, 1]

    # 边界检查：确保索引在有效范围内
    x_clipped = np.clip(x.astype(np.int32), 0, W - 1)
    y_clipped = np.clip(y.astype(np.int32), 0, H - 1)
    flow_vectors = data.flow_nhw2_unfiltered[i_frame - 1, y_clipped, x_clipped]  # [N, 2] 格式：[dx, dy]
    stroke_after_flow = previous_flow_snapped + flow_vectors

    # 第二步：使用未过滤的候选点进行吸附
    points_stroke_candidate_unfiltered = data.kd_tree_unfiltered.query_batch(
        i_frame,
        stroke_after_flow,
        EdgeSnappingConfig.r_s,
    )

    # 获取前一帧的flow_snapped结果用于速度约束
    previous_snapped_stroke = buffers.flow_snapped[i_frame - 1] if i > 0 else None
    
    # 禁用拓扑修复
    original_topology_fix = EdgeSnappingConfig.enable_topology_fix
    EdgeSnappingConfig.enable_topology_fix = False
    
    stroke_flow_snapped = local_snapping(
        stroke_after_flow,
        data.images_rgb[i_frame],
        points_stroke_candidate_unfiltered,  # 使用未过滤的候选点
        previous_snapped_stroke=previous_snapped_stroke,
        original_stroke_length=data.original_stroke_length,  # 传递原始长度用于长度约束
    )
    
    # 恢复拓扑修复设置
    EdgeSnappingConfig.enable_topology_fix = original_topology_fix
    
    return stroke_flow_snapped


def generate_dilated_flow_snapping_prediction(
    i_frame: int,
    i: int,
    stroke_copied: np.ndarray,
    buffers: StrokeBuffers,
    data: StrokeData,
    H: int,
    W: int,
) -> np.ndarray:
    """
    生成当前帧的dilated_flow_snapping预测结果。
    先用处理后的光流（膨胀光流）传播，再用未过滤候选点吸附。
    
    Args:
        i_frame: 当前帧索引
        i: 循环索引（用于判断是否是第一帧）
        stroke_copied: 从前一帧复制的stroke（未使用，保留以保持接口一致）
        buffers: 存储预测结果的缓冲区
        data: 包含图像、光流和KD树的数据
        H: 图像高度
        W: 图像宽度
    
    Returns:
        当前帧的dilated_flow_snapping预测结果
    """
    # 使用自身的数据进行传播
    previous_dilated_flow_snapping = buffers.dilated_flow_snapping[i_frame - 1]
    if previous_dilated_flow_snapping is None:
        raise RuntimeError(f"Missing dilated_flow_snapping stroke for frame {i_frame - 1}")

    # 第一步：光流传播（使用处理后的光流：膨胀光流）
    x, y = previous_dilated_flow_snapping[:, 0], previous_dilated_flow_snapping[:, 1]

    # 边界检查：确保索引在有效范围内
    x_clipped = np.clip(x.astype(np.int32), 0, W - 1)
    y_clipped = np.clip(y.astype(np.int32), 0, H - 1)
    flow_vectors = data.flow_nhw2[i_frame - 1, y_clipped, x_clipped]  # 使用膨胀光流
    stroke_after_flow = previous_dilated_flow_snapping + flow_vectors

    # 第二步：使用未过滤的候选点进行吸附（禁用拓扑修复）
    points_stroke_candidate_unfiltered = data.kd_tree_unfiltered.query_batch(
        i_frame,
        stroke_after_flow,
        EdgeSnappingConfig.r_s,
    )

    # 获取前一帧的结果用于速度约束
    previous_snapped_stroke = buffers.dilated_flow_snapping[i_frame - 1] if i > 0 else None
    
    # 禁用拓扑修复
    original_topology_fix = EdgeSnappingConfig.enable_topology_fix
    EdgeSnappingConfig.enable_topology_fix = False
    
    stroke_dilated_flow_snapping = local_snapping(
        stroke_after_flow,
        data.images_rgb[i_frame],
        points_stroke_candidate_unfiltered,  # 使用未过滤的候选点
        previous_snapped_stroke=previous_snapped_stroke,
        original_stroke_length=data.original_stroke_length,
    )
    
    # 恢复拓扑修复设置
    EdgeSnappingConfig.enable_topology_fix = original_topology_fix
    
    return stroke_dilated_flow_snapping


def generate_flow_filtered_snapping_prediction(
    i_frame: int,
    i: int,
    stroke_copied: np.ndarray,
    buffers: StrokeBuffers,
    data: StrokeData,
    H: int,
    W: int,
) -> np.ndarray:
    """
    生成当前帧的flow_filtered_snapping预测结果。
    先用原始光流传播，再用过滤后候选点吸附。
    
    Args:
        i_frame: 当前帧索引
        i: 循环索引（用于判断是否是第一帧）
        stroke_copied: 从前一帧复制的stroke（未使用，保留以保持接口一致）
        buffers: 存储预测结果的缓冲区
        data: 包含图像、光流和KD树的数据
        H: 图像高度
        W: 图像宽度
    
    Returns:
        当前帧的flow_filtered_snapping预测结果
    """
    # 使用自身的数据进行传播
    previous_flow_filtered_snapping = buffers.flow_filtered_snapping[i_frame - 1]
    if previous_flow_filtered_snapping is None:
        raise RuntimeError(f"Missing flow_filtered_snapping stroke for frame {i_frame - 1}")

    # 第一步：光流传播（使用原始光流）
    x, y = previous_flow_filtered_snapping[:, 0], previous_flow_filtered_snapping[:, 1]

    # 边界检查：确保索引在有效范围内
    x_clipped = np.clip(x.astype(np.int32), 0, W - 1)
    y_clipped = np.clip(y.astype(np.int32), 0, H - 1)
    flow_vectors = data.flow_nhw2_unfiltered[i_frame - 1, y_clipped, x_clipped]  # 使用原始光流
    stroke_after_flow = previous_flow_filtered_snapping + flow_vectors

    # 第二步：使用过滤后的候选点进行吸附（禁用拓扑修复）
    points_stroke_candidate_filtered = data.kd_tree.query_batch(
        i_frame,
        stroke_after_flow,
        EdgeSnappingConfig.r_s,
    )

    # 获取前一帧的结果用于速度约束
    previous_snapped_stroke = buffers.flow_filtered_snapping[i_frame - 1] if i > 0 else None
    
    # 禁用拓扑修复
    original_topology_fix = EdgeSnappingConfig.enable_topology_fix
    EdgeSnappingConfig.enable_topology_fix = False
    
    stroke_flow_filtered_snapping = local_snapping(
        stroke_after_flow,
        data.images_rgb[i_frame],
        points_stroke_candidate_filtered,  # 使用过滤后的候选点
        previous_snapped_stroke=previous_snapped_stroke,
        original_stroke_length=data.original_stroke_length,
    )
    
    # 恢复拓扑修复设置
    EdgeSnappingConfig.enable_topology_fix = original_topology_fix
    
    return stroke_flow_filtered_snapping


def generate_fitted_prediction(
    i_frame: int,
    i: int,
    stroke_copied: np.ndarray,
    buffers: StrokeBuffers,
    data: StrokeData,
    H: int,
    W: int,
) -> np.ndarray:
    """
    生成当前帧的fitted预测结果。
    
    Args:
        i_frame: 当前帧索引
        i: 循环索引（用于判断是否是第一帧）
        stroke_copied: 从前一帧复制的fitted stroke
        buffers: 存储预测结果的缓冲区
        data: 包含图像、光流和KD树的数据
        H: 图像高度
        W: 图像宽度
    
    Returns:
        当前帧的fitted预测结果
    """
    # fitted 传播：使用当前帧的 fitted stroke 位置采样光流
    x_fit, y_fit = stroke_copied[:, 0], stroke_copied[:, 1]
    x_fit_clipped = np.clip(x_fit.astype(np.int32), 0, W - 1)
    y_fit_clipped = np.clip(y_fit.astype(np.int32), 0, H - 1)
    flow_vectors_fit = data.flow_nhw2[i_frame - 1, y_fit_clipped, x_fit_clipped]  # [N, 2]
    stroke_fitted = stroke_copied + flow_vectors_fit

    # 使用过滤后的候选点进行fitted传播
    points_stroke_candidate_fitted = data.kd_tree.query_batch(
        i_frame,
        stroke_fitted,
        EdgeSnappingConfig.r_s,
    )
    # 获取前一帧的fitted结果用于速度约束
    previous_fitted_stroke = None
    if i > 0 and buffers.fitted[i_frame - 1] is not None:
        previous_fitted_stroke = buffers.fitted[i_frame - 1]
    
    stroke_fitted = local_snapping(
        stroke_fitted,
        data.images_rgb[i_frame],
        points_stroke_candidate_fitted,  # 使用过滤后的候选点
        previous_snapped_stroke=previous_fitted_stroke,
        original_stroke_length=data.original_stroke_length,  # 传递原始长度用于长度约束
    )
    
    return stroke_fitted


def generate_prediction_strokes_subsequent(buffers: StrokeBuffers, data: StrokeData) -> None:
    """Iteratively propagate strokes across frames using flow & snapping."""

    # 预先计算H和W，避免每次循环重复计算
    H, W = data.images_rgb.shape[1], data.images_rgb.shape[2]
    n_frames = data.images_rgb.shape[0]

    # 使用mininterval减少tqdm更新频率，提高性能
    for i in tqdm(range(n_frames - 1), desc="Generating prediction strokes on subsequent frames:", unit=" batch", mininterval=0.5):
        i_frame = i + 1

        stroke_copied = buffers.fitted[i_frame - 1]
        if stroke_copied is None:
            raise RuntimeError(f"Missing fitted stroke for frame {i_frame - 1}")

        # 生成snapping预测结果
        stroke_snapping = generate_snapping_prediction(
            i_frame, i, stroke_copied, buffers, data
        )
        buffers.snapping[i_frame] = stroke_snapping

        # 在每次local_snapping调用后立即清理GPU缓存，防止累积
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            # 每10帧重置一次GPU内存统计，可能有助于减少内存碎片化
            if i_frame % 10 == 0:
                torch.cuda.reset_peak_memory_stats()

        # 生成flow预测结果
        stroke_flow = generate_flow_prediction(
            i_frame, i, stroke_copied, buffers, data, H, W
        )
        buffers.flow[i_frame] = stroke_flow

        # 生成flow_snapped预测结果
        stroke_flow_snapped = generate_flow_snapped_prediction(
            i_frame, i, stroke_copied, buffers, data, H, W
        )
        buffers.flow_snapped[i_frame] = stroke_flow_snapped

        # 在每次local_snapping调用后立即清理GPU缓存，防止累积
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            # 每10帧重置一次GPU内存统计，可能有助于减少内存碎片化
            if i_frame % 10 == 0:
                torch.cuda.reset_peak_memory_stats()

        # 生成dilated_flow_snapping预测结果
        stroke_dilated_flow_snapping = generate_dilated_flow_snapping_prediction(
            i_frame, i, stroke_copied, buffers, data, H, W
        )
        buffers.dilated_flow_snapping[i_frame] = stroke_dilated_flow_snapping

        # 在每次local_snapping调用后立即清理GPU缓存，防止累积
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            # 每10帧重置一次GPU内存统计，可能有助于减少内存碎片化
            if i_frame % 10 == 0:
                torch.cuda.reset_peak_memory_stats()

        # 生成flow_filtered_snapping预测结果
        stroke_flow_filtered_snapping = generate_flow_filtered_snapping_prediction(
            i_frame, i, stroke_copied, buffers, data, H, W
        )
        buffers.flow_filtered_snapping[i_frame] = stroke_flow_filtered_snapping

        # 在每次local_snapping调用后立即清理GPU缓存，防止累积
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            # 每10帧重置一次GPU内存统计，可能有助于减少内存碎片化
            if i_frame % 10 == 0:
                torch.cuda.reset_peak_memory_stats()

        # 生成fitted预测结果
        stroke_fitted = generate_fitted_prediction(
            i_frame, i, stroke_copied, buffers, data, H, W
        )
        buffers.fitted[i_frame] = stroke_fitted

        # 在每次local_snapping调用后立即清理GPU缓存，防止累积
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            # 每10帧重置一次GPU内存统计，可能有助于减少内存碎片化
            if i_frame % 10 == 0:
                torch.cuda.reset_peak_memory_stats()

        # 定期清理Python对象和GPU内存，防止累积
        if i % 5 == 0:  # 每5帧清理一次（提高清理频率）
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # 同时清理GPU缓存


def propagate_strokes_with_snapping_flow(data: StrokeData, buffers: StrokeBuffers, stroke_initial: np.ndarray) -> None:
    """Full propagation pipeline for a single initial stroke polyline."""

    n_frame = data.images_rgb.shape[0]
    buffers.reset(n_frame)
    # 注意：第0帧的flow、snapping、flow_snapped和fitted会在generate_prediction_stroke_on_0中初始化

    # 输出输入笔画数据的形状
    print("\n" + "=" * 60)
    print("输入笔画数据信息:")
    print("=" * 60)
    print(f"  Stroke initial shape: {stroke_initial.shape}, dtype: {stroke_initial.dtype}")
    print(f"  Number of points: {len(stroke_initial)}")
    print(f"  Total frames: {n_frame}")
    print("=" * 60 + "\n")

    # 计算第一帧原始stroke的长度（用于长度约束）
    if data.original_stroke_length is None:
        original_length = 0.0
        for i in range(len(stroke_initial) - 1):
            original_length += np.linalg.norm(stroke_initial[i + 1] - stroke_initial[i])
        # 更新data中的原始长度
        data.original_stroke_length = original_length if original_length > 1e-6 else None

    generate_prediction_stroke_on_0(buffers, data, stroke_initial)
    generate_prediction_strokes_subsequent(buffers, data)
    
    # 输出所有帧的预测结果形状统计
    print("\n" + "=" * 60)
    print("所有帧预测结果形状统计:")
    print("=" * 60)
    
    # 统计每种预测结果的形状
    snapping_shapes = [buffers.snapping[i].shape if buffers.snapping[i] is not None else None for i in range(n_frame)]
    flow_shapes = [buffers.flow[i].shape if buffers.flow[i] is not None else None for i in range(n_frame)]
    flow_snapped_shapes = [buffers.flow_snapped[i].shape if buffers.flow_snapped[i] is not None else None for i in range(n_frame)]
    fitted_shapes = [buffers.fitted[i].shape if buffers.fitted[i] is not None else None for i in range(n_frame)]
    
    # 找出所有不同的形状
    unique_snapping = set(s for s in snapping_shapes if s is not None)
    unique_flow = set(s for s in flow_shapes if s is not None)
    unique_flow_snapped = set(s for s in flow_snapped_shapes if s is not None)
    unique_fitted = set(s for s in fitted_shapes if s is not None)
    
    print(f"  Snapping shapes: {unique_snapping}")
    print(f"  Flow shapes: {unique_flow}")
    print(f"  Flow_snapped shapes: {unique_flow_snapped}")
    print(f"  Fitted shapes: {unique_fitted}")
    
    # 统计每帧的点数
    print("\n各帧点数统计 (前5帧和后5帧):")
    for i in range(min(5, n_frame)):
        if buffers.snapping[i] is not None:
            print(f"  Frame {i:03d}: snapping={len(buffers.snapping[i])}, flow={len(buffers.flow[i]) if buffers.flow[i] is not None else 0}, "
                  f"flow_snapped={len(buffers.flow_snapped[i]) if buffers.flow_snapped[i] is not None else 0}, "
                  f"fitted={len(buffers.fitted[i]) if buffers.fitted[i] is not None else 0}")
    if n_frame > 10:
        print("  ...")
        for i in range(max(5, n_frame - 5), n_frame):
            if buffers.snapping[i] is not None:
                print(f"  Frame {i:03d}: snapping={len(buffers.snapping[i])}, flow={len(buffers.flow[i]) if buffers.flow[i] is not None else 0}, "
                      f"flow_snapped={len(buffers.flow_snapped[i]) if buffers.flow_snapped[i] is not None else 0}, "
                      f"fitted={len(buffers.fitted[i]) if buffers.fitted[i] is not None else 0}")
    
    print("=" * 60 + "\n")


def draw_curves(canvas: np.ndarray, context: RuntimeContext) -> None:
    """Render all requested stroke overlays onto `canvas`."""

    state = context.viewer
    buffers = context.buffers

    stroke_origin = context.strokes_library[state.current_stroke_index]

    if state.show_origin:
        cv2.polylines(canvas, [stroke_origin.astype(np.int32)], False, rgb_to_bgr(COLOR_ORIGIN), THICKNESS, lineType=cv2.LINE_AA)

    stroke_flow = buffers.flow[state.current_frame]
    if stroke_flow is not None and state.show_flow:
        cv2.polylines(canvas, [stroke_flow.astype(np.int32)], False, rgb_to_bgr(COLOR_FLOW), THICKNESS, lineType=cv2.LINE_AA)

    stroke_snapping = buffers.snapping[state.current_frame]
    if stroke_snapping is not None and state.show_snapping:
        cv2.polylines(canvas, [stroke_snapping.astype(np.int32)], False, rgb_to_bgr(COLOR_SNAPPING), THICKNESS, lineType=cv2.LINE_AA)

    stroke_flow_snapped = buffers.flow_snapped[state.current_frame]
    if stroke_flow_snapped is not None and state.show_flow_snapped:
        cv2.polylines(canvas, [stroke_flow_snapped.astype(np.int32)], False, rgb_to_bgr(COLOR_FLOW_SNAPPED), THICKNESS, lineType=cv2.LINE_AA)

    stroke_dilated_flow_snapping = buffers.dilated_flow_snapping[state.current_frame]
    if stroke_dilated_flow_snapping is not None and state.show_dilated_flow_snapping:
        cv2.polylines(canvas, [stroke_dilated_flow_snapping.astype(np.int32)], False, rgb_to_bgr(COLOR_DILATED_FLOW_SNAPPING), THICKNESS, lineType=cv2.LINE_AA)

    stroke_flow_filtered_snapping = buffers.flow_filtered_snapping[state.current_frame]
    if stroke_flow_filtered_snapping is not None and state.show_flow_filtered_snapping:
        cv2.polylines(canvas, [stroke_flow_filtered_snapping.astype(np.int32)], False, rgb_to_bgr(COLOR_FLOW_FILTERED_SNAPPING), THICKNESS, lineType=cv2.LINE_AA)

    stroke_fitted = buffers.fitted[state.current_frame]
    if stroke_fitted is not None and state.show_fitted:
        cv2.polylines(canvas, [stroke_fitted.astype(np.int32)], False, rgb_to_bgr(COLOR_FITTED), THICKNESS, lineType=cv2.LINE_AA)


def propagate_current_stroke(context: RuntimeContext) -> None:
    """Re-run propagation for the stroke chosen by the viewer."""

    stroke = context.strokes_library[context.viewer.current_stroke_index]
    propagate_strokes_with_snapping_flow(context.data, context.buffers, stroke)
    export_stroke_gifs(context)


def export_stroke_gifs(context: RuntimeContext) -> None:
    """Export stroke propagation results to GIFs and PNG images grouped by stroke categories."""

    env = context.env
    data = context.data
    buffers = context.buffers
    n_frame = data.images_rgb.shape[0]
    target_dir = env.final_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    # 生成 input.png（第 0 帧的原始 stroke）
    H, W = data.images_rgb.shape[1], data.images_rgb.shape[2]
    crop_ratio = EdgeSnappingConfig.export_crop_ratio if EdgeSnappingConfig.export_crop_ratio is not None else 1.0
    
    # 计算裁剪区域（从中心开始）
    crop_h = int(H * crop_ratio)
    crop_w = int(W * crop_ratio)
    start_y = (H - crop_h) // 2
    start_x = (W - crop_w) // 2
    end_y = start_y + crop_h
    end_x = start_x + crop_w
    
    # 生成第 0 帧的原始 stroke 图像
    background_input = cv2.cvtColor(data.images_rgb[0], cv2.COLOR_RGB2BGR)
    stroke_origin = context.strokes_library[context.viewer.current_stroke_index]
    cv2.polylines(
        background_input,
        [stroke_origin.astype(np.int32)],
        False,
        rgb_to_bgr(COLOR_ORIGIN),
        THICKNESS,
        lineType=cv2.LINE_AA,
    )
    
    # 保存不剪裁版本
    input_path_full = target_dir / "input.png"
    cv2.imwrite(str(input_path_full), background_input)
    
    # 保存剪裁版本
    input_image_cropped = background_input[start_y:end_y, start_x:end_x]
    input_path_cropped = target_dir / "input_cropped.png"
    cv2.imwrite(str(input_path_cropped), input_image_cropped)

    stroke_categories = (
        ("fitted", buffers.fitted, COLOR_FITTED),
        ("pure_flow", buffers.flow, COLOR_FLOW),
        ("pure_snapping", buffers.snapping, COLOR_SNAPPING),
        ("pure_flow_and_pure_snapping", buffers.flow_snapped, COLOR_FLOW_SNAPPED),
        ("dilated_flow_and_pure_snapping", buffers.dilated_flow_snapping, COLOR_DILATED_FLOW_SNAPPING),
        ("pure_flow_and_filtered_snapping", buffers.flow_filtered_snapping, COLOR_FLOW_FILTERED_SNAPPING),
    )

    for name, strokes, color_rgb in stroke_categories:
        frames_bgr: List[np.ndarray] = []
        
        # 创建该策略的PNG图片目录
        png_dir = target_dir / name
        png_dir.mkdir(parents=True, exist_ok=True)
        
        # 清空目录中的文件（不删除目录）
        if png_dir.exists():
            for file_path in png_dir.iterdir():
                if file_path.is_file():
                    file_path.unlink()

        for idx in tqdm(range(n_frame), desc=f"Building {name} GIF and PNGs", unit=" frame(s)"):
            background = cv2.cvtColor(data.images_rgb[idx], cv2.COLOR_RGB2BGR)
            stroke_data = strokes[idx]
            if stroke_data is not None:
                cv2.polylines(
                    background,
                    [stroke_data.astype(np.int32)],
                    False,
                    rgb_to_bgr(color_rgb),
                    THICKNESS,
                    lineType=cv2.LINE_AA,
                )
            frames_bgr.append(background)
            
            # 保存PNG图片，命名格式：[策略名_5位0填充的帧号].png
            # 从中心裁剪的区域
            png_image = background[start_y:end_y, start_x:end_x]
            png_filename = f"{name}_{idx:05d}.png"
            png_path = png_dir / png_filename
            cv2.imwrite(str(png_path), png_image)

        # GIF 输出改为不剪裁（使用完整的 frames_bgr）
        # 传入足够大的 crop_size 来禁用裁剪（传入大于图像尺寸的值，函数内部会限制为图像尺寸）
        out_path = target_dir / f"{name}.gif"
        reference_curve = None
        if name == "fitted" and strokes[0] is not None:
            reference_curve = strokes[0]
        save_fixed_length_gif_from_bgr(
            frames_bgr,
            out_path=str(out_path),
            fps=12.0,
            loop=0,
            optimize=True,
            reference_curve=reference_curve,
            crop_size=(H * 2, W * 2),  # 传入足够大的值以确保不裁剪（函数内部会限制为图像尺寸）
        )


def build_runtime_context() -> RuntimeContext:
    """Collect all runtime dependencies and execute initial propagation."""

    env = load_environment()
    print(f"tracing target: {env.target_name}")
    print(f"frame images folder: {env.frame_dir}")

    EdgeSnappingConfig.load(str(env.paths.config / "snapping_init.yaml"))

    strokes_library = read_strokes(env)
    if not strokes_library:
        raise RuntimeError(f"No stroke files found in {env.stroke_dir}")

    frame_image_paths = get_frame_image_paths(env)
    if not frame_image_paths:
        raise RuntimeError(f"No frame images found in {env.frame_dir}")

    images_rgb_nhwc_uint8 = read_images_batch(frame_image_paths, cv2.IMREAD_COLOR_RGB)

    # 只读取一次原始光流缓存
    flow_tensor = read_optical_flow_cache(env)
    flow_nhw2_original = flow_tensor.numpy()
    print(f"Loaded optical flow cache: {flow_nhw2_original.shape}, {flow_nhw2_original.dtype}")

    points_all_candidates = compute_all_candidates(images_rgb_nhwc_uint8)

    # 保存未过滤的候选点（用于纯光流和纯吸附）
    # 使用列表推导式深拷贝每个numpy数组
    points_all_candidates_unfiltered = [points.copy() for points in points_all_candidates]

    # 初始化处理后的光流（默认使用原始光流）
    flow_nhw2_for_fitted = flow_nhw2_original

    # 使用原始mask进行双重过滤：
    # 1. 腐蚀3个像素，过滤掉物体内部的点
    # 2. 膨胀3个像素，过滤掉物体外部的点
    # 最终保留的点：在膨胀mask内部，但不在腐蚀mask内部的点（边界区域）
    try:
        masks_original = read_mask_cache(env)
        print(f"Loaded original mask cache: {masks_original.shape}, {masks_original.dtype}")
        # print("使用原始mask进行双重过滤（腐蚀3像素过滤内部点，膨胀3像素过滤外部点）...")

        erode_size = EdgeSnappingConfig.erode_size
        dilate_size = EdgeSnappingConfig.dilate_size
        flow_erode_size = EdgeSnappingConfig.flow_erode_size
        flow_dilate_size = EdgeSnappingConfig.flow_dilate_size

        # 输出参数信息（帮助理解实际效果）
        print(f"\n📊 Mask和光流处理参数:")
        print(f"  erode_size (候选点过滤): {erode_size}px 厚度 (核: {erode_size * 2 + 1}x{erode_size * 2 + 1})")
        print(f"  dilate_size (候选点过滤): {dilate_size}px 厚度 (核: {dilate_size * 2 + 1}x{dilate_size * 2 + 1})")
        print(f"  flow_erode_size (光流截取): {flow_erode_size}px 厚度 (核: {flow_erode_size * 2 + 1}x{flow_erode_size * 2 + 1})")
        print(f"  flow_dilate_size (光流膨胀): {flow_dilate_size}px 厚度 (核: {flow_dilate_size * 2 + 1}x{flow_dilate_size * 2 + 1})\n")

        # 收集所有mask用于可视化
        masks_eroded_for_flow_list = []
        masks_dilated_for_filter_list = []
        points_all_candidates_filtered = []
        
        for i in tqdm(range(len(points_all_candidates)), desc="Processing Masks and Filtering Points", unit=" frame(s)"):
            if i < masks_original.shape[0]:
                mask_original = masks_original[i]

                # 对原始mask进行腐蚀（用于过滤内部候选点，使用erode_size）
                mask_eroded = erode_mask(mask_original, erode_size)
                
                # 对原始mask进行腐蚀（用于截取光流，使用flow_erode_size）
                mask_eroded_for_flow = erode_mask(mask_original, flow_erode_size)
                masks_eroded_for_flow_list.append(mask_eroded_for_flow)

                # 对原始mask进行膨胀（用于过滤外部点）
                mask_dilated = dilate_mask(mask_original, dilate_size)
                masks_dilated_for_filter_list.append(mask_dilated)

                # 第一步：使用腐蚀mask过滤掉物体内部的点（保留外部点）
                points_after_erode = filter_points_by_mask(points_all_candidates[i], mask_eroded, keep_inside=False)

                # 第二步：使用膨胀mask过滤掉物体外部的点（保留内部点）
                points_filtered = filter_points_by_mask(points_after_erode, mask_dilated, keep_inside=True)

                points_all_candidates_filtered.append(points_filtered)
            else:
                # 如果mask帧数不够，使用原始点
                points_all_candidates_filtered.append(points_all_candidates[i])

        # === 生成所有中间结果可视化 ===
        print("\n" + "=" * 60)
        print("生成中间结果可视化...")
        print("=" * 60)
        
        # Mask可视化
        generate_mask_original_images(env, masks_original)
        if masks_eroded_for_flow_list:
            masks_eroded_for_flow_array = np.array(masks_eroded_for_flow_list)
            generate_mask_erode_images(env, masks_eroded_for_flow_array, masks_original)
        if masks_dilated_for_filter_list:
            masks_dilated_for_filter_array = np.array(masks_dilated_for_filter_list)
            generate_mask_dilate_images(env, masks_dilated_for_filter_array, masks_original)
        
        # 生成三合一对比图
        if masks_eroded_for_flow_list and masks_dilated_for_filter_list:
            generate_mask_triple_comparison(env, masks_original, 
                                          masks_eroded_for_flow_array, 
                                          masks_dilated_for_filter_array)
        
        # 特征点可视化
        generate_salient_images(env, points_all_candidates, images_rgb_nhwc_uint8.shape[1], images_rgb_nhwc_uint8.shape[2])
        generate_salient_filtered_images(env, points_all_candidates_filtered, images_rgb_nhwc_uint8.shape[1], images_rgb_nhwc_uint8.shape[2])
        
        # 光流处理和可视化
        generate_flow_original_images(env, flow_nhw2_original)
        
        if masks_eroded_for_flow_list:
            # 1. 生成裁剪后的光流可视化（flow-crop-by-mask）
            flow_cropped = generate_flow_crop_by_mask_images(env, flow_nhw2_original, masks_eroded_for_flow_array)
            
            # 2. 对裁剪后的光流进行膨胀（flow-dilate-only）
            flow_dilated = generate_flow_dilate_only_images(env, flow_cropped, flow_dilate_size)
            
            # 3. 将膨胀后的光流覆盖回原始光流（flow-dilate-final）
            flow_nhw2_for_fitted = generate_flow_dilate_final_images(env, flow_nhw2_original, flow_dilated)
            
            print(f"\n✅ 光流处理完成: erode_size={flow_erode_size}, dilate_size={flow_dilate_size}")
        
        print("=" * 60 + "\n")
        
        points_all_candidates = points_all_candidates_filtered
        # print("✅ 点过滤完成")
    except ValueError as e:
        print(f"⚠️  警告: {e}")
        print("   将使用未过滤的点")

    kd_tree_groups = BatchKDTree(points_all_candidates)
    # 为未过滤的候选点创建 kd_tree（用于纯光流和纯吸附）
    kd_tree_groups_unfiltered = BatchKDTree(points_all_candidates_unfiltered)

    data = StrokeData(
        images_rgb=images_rgb_nhwc_uint8,
        flow_nhw2=flow_nhw2_for_fitted,  # 处理后的光流（腐蚀+膨胀，用于fitted）
        flow_nhw2_unfiltered=flow_nhw2_original,  # 原始光流（用于flow、flow_snapped、snapping）
        kd_tree=kd_tree_groups,
        kd_tree_unfiltered=kd_tree_groups_unfiltered,
    )

    buffers = StrokeBuffers()
    viewer = ViewerState()

    context = RuntimeContext(
        env=env,
        data=data,
        strokes_library=strokes_library,
        buffers=buffers,
        viewer=viewer,
    )

    propagate_current_stroke(context)
    return context


def main():
    """Entry point used when launching the module as a script."""

    context = build_runtime_context()
    registry = KeyHandlerRegistry()
    viewer = context.viewer
    data = context.data

    @registry.register('a')
    def handle_prev_frame() -> bool:
        if viewer.current_frame > 0:
            viewer.current_frame -= 1
        return False

    @registry.register('d')
    def handle_next_frame() -> bool:
        if viewer.current_frame < data.images_rgb.shape[0] - 1:
            viewer.current_frame += 1
        return False

    @registry.register('z')
    def toggle_snapping_visibility() -> bool:
        """z键：切换纯吸附结果（pure_snapping）的显示"""
        viewer.show_snapping = not viewer.show_snapping
        return False

    @registry.register('x')
    def toggle_flow_visibility() -> bool:
        """x键：切换纯光流结果（pure_flow）的显示"""
        viewer.show_flow = not viewer.show_flow
        return False

    @registry.register('c')
    def toggle_flow_snapped_visibility() -> bool:
        """c键：切换纯光流+纯吸附结果（pure_flow_and_pure_snapping）的显示"""
        viewer.show_flow_snapped = not viewer.show_flow_snapped
        return False

    @registry.register('v')
    def toggle_dilated_flow_snapping_visibility() -> bool:
        """v键：切换膨胀光流+纯吸附结果（dilated_flow_and_pure_snapping）的显示"""
        viewer.show_dilated_flow_snapping = not viewer.show_dilated_flow_snapping
        return False

    @registry.register('b')
    def toggle_flow_filtered_snapping_visibility() -> bool:
        """b键：切换纯光流+过滤吸附结果（pure_flow_and_filtered_snapping）的显示"""
        viewer.show_flow_filtered_snapping = not viewer.show_flow_filtered_snapping
        return False

    @registry.register('n')
    def toggle_fitted_visibility() -> bool:
        """n键：切换fitted结果的显示"""
        viewer.show_fitted = not viewer.show_fitted
        return False

    def create_switch_test_stroke_handler(index: int) -> Callable[[], bool]:
        def handler() -> bool:
            if index >= len(context.strokes_library):
                return False
            if index != viewer.current_stroke_index:
                viewer.current_stroke_index = index
                propagate_current_stroke(context)
            return False

        return handler

    for idx, key_char in enumerate(('1', '2', '3')):
        registry.register(key_char)(create_switch_test_stroke_handler(idx))

    @registry.register('q')
    def handle_quit() -> bool:
        return True

    while True:
        # Poll for keyboard events via OpenCV and dispatch to registered handlers.
        key = cv2.waitKey(1) & 0xFF
        if registry.dispatch(key):
            break

        canvas = cv2.cvtColor(context.data.images_rgb[viewer.current_frame], cv2.COLOR_RGB2BGR)
        draw_curves(canvas, context)

        # 在左上角显示当前帧数
        frame_text = f"Frame: {viewer.current_frame}/{data.images_rgb.shape[0] - 1}"
        cv2.putText(
            canvas,
            frame_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(context.env.target_name, canvas)


if __name__ == '__main__':
    main()
