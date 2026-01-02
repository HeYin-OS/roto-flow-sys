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

COLOR_ORIGIN = (255, 255, 0)  # Vivid Orange
COLOR_FLOW = (200, 130, 255)  # Soft Lavender
COLOR_SNAPPING = (0, 150, 255)  # Tech Blue
COLOR_FLOW_SNAPPED = (255, 100, 0)  # Orange Red
COLOR_FITTED = (50, 200, 50)  # Fresh Green
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
    def cache_file(self) -> Path:
        """Path to the optical-flow cache tensor for the current target (dilated flow, used for fitted)."""

        return self.paths.cache / "flow-dilate" / f"{self.target_name}.pt"

    @property
    def flow_cache_file_unfiltered(self) -> Path:
        """Path to the unfiltered optical-flow cache tensor for the current target (used for flow and flow_snapped)."""

        return self.paths.cache / "flow" / f"{self.target_name}.pt"

    @property
    def mask_cache_file(self) -> Path:
        """Path to the original mask cache tensor for the current target."""

        return self.paths.cache / "mask" / f"{self.target_name}.pt"

    @property
    def salient_dir(self) -> Path:
        """Directory used to dump candidate salient points for debugging."""

        return self.paths.result / "salient" / self.target_name

    # @property
    # def salient_stroke_dir(self) -> Path:
    #     """Directory used to dump salient stroke groups for debugging."""
    #
    #     return self.paths.result / "salient" / self.target_name


@dataclass
class ViewerState:
    """Mutable UI state controlled by keyboard shortcuts."""

    current_frame: int = 0
    current_stroke_index: int = 2  # 默认使用3号stroke（索引从0开始，3号对应索引2）
    show_origin: bool = False
    show_snapping: bool = False  # z键：纯吸附结果
    show_flow: bool = False  # x键：纯光流结果
    show_flow_snapped: bool = False  # c键：光流吸附结果
    show_fitted: bool = True  # v键：fitted结果


@dataclass
class StrokeBuffers:
    """Frame-aligned buffers containing different stroke propagation results."""

    flow: List[np.ndarray | None] = field(default_factory=list)
    snapping: List[np.ndarray | None] = field(default_factory=list)
    flow_snapped: List[np.ndarray | None] = field(default_factory=list)
    fitted: List[np.ndarray | None] = field(default_factory=list)

    def reset(self, n_frame: int) -> None:
        """Pre-allocate buffers with None placeholders for `n_frame` frames."""

        self.flow = [None] * n_frame
        self.snapping = [None] * n_frame
        self.flow_snapped = [None] * n_frame
        self.fitted = [None] * n_frame


@dataclass
class StrokeData:
    """In-memory data required to propagate strokes over all frames."""

    images_rgb: np.ndarray
    flow_nhw2: np.ndarray  # 膨胀后的光流（用于fitted）
    flow_nhw2_unfiltered: np.ndarray  # 未膨胀的光流（用于flow和flow_snapped）
    kd_tree: BatchKDTree  # 使用过滤后的候选点
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
    """Load the precomputed optical-flow tensor for the configured target (dilated flow)."""

    cache_path = env.cache_file
    if cache_path.exists():
        return torch.load(str(cache_path))
    raise ValueError(f"Optical flow cache file does not exist: {cache_path}")


def read_optical_flow_cache_unfiltered(env: StrokeEnvironment) -> Tensor:
    """Load the unfiltered optical-flow tensor for the configured target (unfiltered flow)."""

    cache_path = env.flow_cache_file_unfiltered
    if cache_path.exists():
        return torch.load(str(cache_path))
    raise ValueError(f"Unfiltered optical flow cache file does not exist: {cache_path}")


def read_mask_cache(env: StrokeEnvironment) -> np.ndarray:
    """Load the original mask cache tensor for the configured target."""

    cache_path = env.mask_cache_file
    if cache_path.exists():
        masks = torch.load(str(cache_path))
        if isinstance(masks, torch.Tensor):
            masks = masks.numpy()
        return masks
    raise ValueError(f"Mask cache file does not exist: {cache_path}")


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


def dilate_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    对mask进行形态学膨胀
    
    Args:
        mask: 二值mask，形状为(H, W)，值为0或1
        kernel_size: 膨胀核大小（像素宽度）
    
    Returns:
        膨胀后的mask
    """
    # 确保mask是uint8类型
    if mask.dtype != np.uint8:
        mask_uint8 = (mask * 255).astype(np.uint8)
    else:
        mask_uint8 = mask

    # 创建圆形结构元素
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size * 2 + 1, kernel_size * 2 + 1))

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
    """Dump salient edge candidate maps for debugging or offline inspection."""

    work_dir = env.salient_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    
    # 清空目录中的文件（不删除目录）
    if work_dir.exists():
        for file_path in work_dir.iterdir():
            if file_path.is_file():
                file_path.unlink()
    
    for i_img in tqdm(range(len(points_all_candidates)), desc="Generating salient point images:", unit=" image(s)"):
        canvas = np.zeros((height, width), np.uint8)
        canvas[points_all_candidates[i_img][:, 1].astype(np.int32), points_all_candidates[i_img][:, 0].astype(np.int32)] = 255
        file_path = work_dir / f"{i_img:03d}.png"
        cv2.imwrite(str(file_path), canvas)


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

    # 生成fitted结果（使用过滤后的候选点）
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

    # 生成snapping结果（使用未过滤的候选点）
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

    # 生成flow结果（第0帧使用未过滤点集吸附）
    buffers.flow[0] = stroke_0_snapping.astype(np.float32)

    # 生成flow_snapped结果（第0帧使用未过滤点集的吸附结果）
    buffers.flow_snapped[0] = stroke_0_snapping.astype(np.float32)
    
    # 输出第0帧四种预测结果的形状
    print("\n" + "=" * 60)
    print("Frame 0 预测结果形状:")
    print("=" * 60)
    print(f"  Original stroke shape: {stroke_0.shape}, dtype: {stroke_0.dtype}")
    print(f"  Snapping shape: {buffers.snapping[0].shape}, dtype: {buffers.snapping[0].dtype}")
    print(f"  Flow shape: {buffers.flow[0].shape}, dtype: {buffers.flow[0].dtype}")
    print(f"  Flow_snapped shape: {buffers.flow_snapped[0].shape}, dtype: {buffers.flow_snapped[0].dtype}")
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
    
    stroke_snapping = local_snapping(
        previous_snapping,
        data.images_rgb[i_frame],
        points_stroke_candidate_unfiltered,  # 使用未过滤的候选点
        previous_snapped_stroke=previous_snapped_stroke,
        original_stroke_length=data.original_stroke_length,  # 传递原始长度用于长度约束
    )
    
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
    
    stroke_flow_snapped = local_snapping(
        stroke_after_flow,
        data.images_rgb[i_frame],
        points_stroke_candidate_unfiltered,  # 使用未过滤的候选点
        previous_snapped_stroke=previous_snapped_stroke,
        original_stroke_length=data.original_stroke_length,  # 传递原始长度用于长度约束
    )
    
    return stroke_flow_snapped


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

    stroke_fitted = buffers.fitted[state.current_frame]
    if stroke_fitted is not None and state.show_fitted:
        cv2.polylines(canvas, [stroke_fitted.astype(np.int32)], False, rgb_to_bgr(COLOR_FITTED), THICKNESS, lineType=cv2.LINE_AA)


def propagate_current_stroke(context: RuntimeContext) -> None:
    """Re-run propagation for the stroke chosen by the viewer."""

    stroke = context.strokes_library[context.viewer.current_stroke_index]
    propagate_strokes_with_snapping_flow(context.data, context.buffers, stroke)
    export_stroke_gifs(context)


def export_stroke_gifs(context: RuntimeContext) -> None:
    """Export stroke propagation results to GIFs and JPG images grouped by stroke categories."""

    env = context.env
    data = context.data
    buffers = context.buffers
    n_frame = data.images_rgb.shape[0]
    target_dir = env.paths.result / "final" / env.target_name
    target_dir.mkdir(parents=True, exist_ok=True)

    stroke_categories = (
        ("fitted", buffers.fitted, COLOR_FITTED),
        ("flow", buffers.flow, COLOR_FLOW),
        ("snapping", buffers.snapping, COLOR_SNAPPING),
        ("flow_snapped", buffers.flow_snapped, COLOR_FLOW_SNAPPED),
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
            png_filename = f"{name}_{idx:05d}.png"
            png_path = png_dir / png_filename
            cv2.imwrite(str(png_path), background)

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

    flow_tensor = read_optical_flow_cache(env)
    flow_nhw2_float32 = flow_tensor.numpy()
    print(f"Loaded optical flow cache (dilated): {flow_nhw2_float32.shape}, {flow_nhw2_float32.dtype}")

    flow_tensor_unfiltered = read_optical_flow_cache_unfiltered(env)
    flow_nhw2_unfiltered_float32 = flow_tensor_unfiltered.numpy()
    print(f"Loaded optical flow cache (unfiltered): {flow_nhw2_unfiltered_float32.shape}, {flow_nhw2_unfiltered_float32.dtype}")

    points_all_candidates = compute_all_candidates(images_rgb_nhwc_uint8)

    # 保存未过滤的候选点（用于纯光流和纯吸附）
    # 使用列表推导式深拷贝每个numpy数组
    points_all_candidates_unfiltered = [points.copy() for points in points_all_candidates]

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

        points_all_candidates_filtered = []
        for i in tqdm(range(len(points_all_candidates)), desc="Filtering Salient Points", unit=" frame(s)"):
            if i < masks_original.shape[0]:
                mask_original = masks_original[i]

                # 对原始mask进行腐蚀（用于过滤内部点）
                mask_eroded = erode_mask(mask_original, erode_size)

                # 对原始mask进行膨胀（用于过滤外部点）
                mask_dilated = dilate_mask(mask_original, dilate_size)

                # 第一步：使用腐蚀mask过滤掉物体内部的点（保留外部点）
                points_after_erode = filter_points_by_mask(points_all_candidates[i], mask_eroded, keep_inside=False)

                # 第二步：使用膨胀mask过滤掉物体外部的点（保留内部点）
                points_filtered = filter_points_by_mask(points_after_erode, mask_dilated, keep_inside=True)

                points_all_candidates_filtered.append(points_filtered)
                # print(f"  帧 {i}: {len(points_all_candidates[i])} -> {len(points_after_erode)} (腐蚀后) -> {len(points_filtered)} (最终) 个点")
            else:
                # 如果mask帧数不够，使用原始点
                points_all_candidates_filtered.append(points_all_candidates[i])

        points_all_candidates = points_all_candidates_filtered
        # print("✅ 点过滤完成")
    except ValueError as e:
        print(f"⚠️  警告: {e}")
        print("   将使用未过滤的点")

    generate_salient_images(env, points_all_candidates, images_rgb_nhwc_uint8.shape[1], images_rgb_nhwc_uint8.shape[2])
    kd_tree_groups = BatchKDTree(points_all_candidates)
    # 为未过滤的候选点创建 kd_tree（用于纯光流和纯吸附）
    kd_tree_groups_unfiltered = BatchKDTree(points_all_candidates_unfiltered)

    data = StrokeData(
        images_rgb=images_rgb_nhwc_uint8,
        flow_nhw2=flow_nhw2_float32,  # 膨胀后的光流（用于fitted）
        flow_nhw2_unfiltered=flow_nhw2_unfiltered_float32,  # 未膨胀的光流（用于flow和flow_snapped）
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
        """z键：切换纯吸附结果（snapping）的显示"""
        viewer.show_snapping = not viewer.show_snapping
        return False

    @registry.register('x')
    def toggle_flow_visibility() -> bool:
        """x键：切换纯光流结果（flow）的显示"""
        viewer.show_flow = not viewer.show_flow
        return False

    @registry.register('c')
    def toggle_flow_snapped_visibility() -> bool:
        """c键：切换光流吸附结果（flow_snapped）的显示"""
        viewer.show_flow_snapped = not viewer.show_flow_snapped
        return False

    @registry.register('v')
    def toggle_fitted_visibility() -> bool:
        """v键：切换fitted结果的显示"""
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
