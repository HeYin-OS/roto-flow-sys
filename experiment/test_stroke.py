from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List

import cv2
import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from utils.edge_snapping import compute_all_candidates, EdgeSnappingConfig, local_snapping
from utils.kd_tree import BatchKDTree
from utils.yaml_reader import YamlUtil
from utils.gif_writer import save_fixed_length_gif_from_bgr

COLOR_ORIGIN = (255, 255, 0)  # Vivid Orange
COLOR_FLOW = (200, 130, 255)  # Soft Lavender
COLOR_SNAPPING = (0, 150, 255)  # Tech Blue
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
        """Path to the optical-flow cache tensor for the current target."""

        return self.paths.cache / f"{self.target_name}.pt"

    @property
    def salient_dir(self) -> Path:
        """Directory used to dump candidate salient points for debugging."""

        return self.paths.debug / "salient" / self.target_name

    @property
    def salient_stroke_dir(self) -> Path:
        """Directory used to dump salient stroke groups for debugging."""

        return self.paths.debug / "salient_stroke" / self.target_name


@dataclass
class ViewerState:
    """Mutable UI state controlled by keyboard shortcuts."""

    current_frame: int = 0
    current_stroke_index: int = 0
    show_origin: bool = True
    show_flow: bool = False
    show_snapping: bool = False
    show_fitted: bool = True


@dataclass
class StrokeBuffers:
    """Frame-aligned buffers containing different stroke propagation results."""

    flow: List[np.ndarray | None] = field(default_factory=list)
    snapping: List[np.ndarray | None] = field(default_factory=list)
    fitted: List[np.ndarray | None] = field(default_factory=list)

    def reset(self, n_frame: int) -> None:
        """Pre-allocate buffers with None placeholders for `n_frame` frames."""

        self.flow = [None] * n_frame
        self.snapping = [None] * n_frame
        self.fitted = [None] * n_frame


@dataclass
class StrokeData:
    """In-memory data required to propagate strokes over all frames."""

    images_rgb: np.ndarray
    flow_nhw2: np.ndarray
    kd_tree: BatchKDTree


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

    base = Path(__file__).resolve().parent.parent
    return ProjectPaths(
        base=base,
        config=base / "config",
        cache=base / "caches",
        stroke=base / "stroke",
        debug=base / "debug",
    )


def load_environment() -> StrokeEnvironment:
    """Load configuration metadata and resolve frame directory for the target."""

    paths = build_project_paths()
    config_path = paths.config / "test_video_init.yaml"
    config_data = YamlUtil.read(str(config_path))
    frame_dir_raw = Path(config_data['video']['url_head'])
    frame_dir = (paths.base / frame_dir_raw).resolve() if not frame_dir_raw.is_absolute() else frame_dir_raw.resolve()
    return StrokeEnvironment(
        paths=paths,
        frame_dir=frame_dir,
        target_name=frame_dir.name,
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
        strokes.append(np.load(str(path)).astype(np.float32))
        print(f"Loaded stroke from:  {path}")
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

    cache_path = env.cache_file
    if cache_path.exists():
        return torch.load(str(cache_path))
    raise ValueError(f"Optical flow cache file does not exist: {cache_path}")


def generate_salient_images(env: StrokeEnvironment, points_all_candidates: List[np.ndarray], height: int, width: int) -> None:
    """Dump salient edge candidate maps for debugging or offline inspection."""

    for i_img in tqdm(range(len(points_all_candidates)), desc="Generating salient point images:", unit=" image(s)"):
        canvas = np.zeros((height, width), np.uint8)
        canvas[points_all_candidates[i_img][:, 1].astype(np.int32), points_all_candidates[i_img][:, 0].astype(np.int32)] = 255
        work_dir = env.salient_dir
        work_dir.mkdir(parents=True, exist_ok=True)
        file_path = work_dir / f"{i_img:03d}.jpg"
        cv2.imwrite(str(file_path), canvas)


def generate_salient_stroke_images(env: StrokeEnvironment, points_stroke_candidates: List[np.ndarray], height: int, width: int, i_frame: int) -> None:
    """Dump stroke-wise salient candidates for a particular frame."""

    canvas = np.zeros((height, width), np.uint8)
    for i_group in range(len(points_stroke_candidates)):
        canvas[points_stroke_candidates[i_group][:, 1], points_stroke_candidates[i_group][:, 0]] = 255
    work_dir = env.salient_stroke_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    file_path = work_dir / f"{i_frame:03d}.jpg"
    cv2.imwrite(str(file_path), canvas)


def rgb_to_bgr(color: tuple) -> tuple:
    """Convert an RGB tuple/list to BGR order."""
    return color[::-1]


def generate_prediction_stroke_on_0(buffers: StrokeBuffers, data: StrokeData, stroke_0: np.ndarray) -> None:
    """Snap the reference stroke onto frame-0 edges prior to propagation."""

    points_stroke_candidate = data.kd_tree.query_batch(
        0,
        stroke_0,
        EdgeSnappingConfig.r_s,
    )
    stroke_0_snapped = local_snapping(
        stroke_0,
        data.images_rgb[0],
        points_stroke_candidate,
    )
    buffers.fitted[0] = stroke_0_snapped.astype(np.float32)


def generate_prediction_strokes_subsequent(buffers: StrokeBuffers, data: StrokeData) -> None:
    """Iteratively propagate strokes across frames using flow & snapping."""

    for i in tqdm(range(data.images_rgb.shape[0] - 1), desc="Generating prediction strokes on subsequent frames:", unit=" batch"):
        i_frame = i + 1

        stroke_copied = buffers.fitted[i_frame - 1]
        if stroke_copied is None:
            raise RuntimeError(f"Missing fitted stroke for frame {i_frame - 1}")

        points_stroke_candidate = data.kd_tree.query_batch(
            i_frame,
            stroke_copied,
            EdgeSnappingConfig.r_s,
        )

        if i == 0 or buffers.snapping[i_frame - 1] is None:
            previous_snapping = stroke_copied
        else:
            previous_snapping = buffers.snapping[i_frame - 1]
        stroke_snapping = local_snapping(
            previous_snapping,
            data.images_rgb[i_frame],
            points_stroke_candidate,
        )
        buffers.snapping[i_frame] = stroke_snapping

        # 光流传播：从 frame i-1 到 frame i
        # flow_nhw2[i_frame - 1] 存储的是从 frame i-1 到 frame i 的光流
        # 格式：[H, W, 2]，其中 [:, :, 0] 是 x 方向，[:, :, 1] 是 y 方向
        H, W = data.images_rgb.shape[1], data.images_rgb.shape[2]
        
        if i == 0 or buffers.flow[i_frame - 1] is None:
            x, y = stroke_copied[:, 0], stroke_copied[:, 1]
            previous_flow = stroke_copied
        else:
            previous_flow = buffers.flow[i_frame - 1]
            x, y = previous_flow[:, 0], previous_flow[:, 1]
        
        # 边界检查：确保索引在有效范围内
        x_clipped = np.clip(x.astype(np.int32), 0, W - 1)
        y_clipped = np.clip(y.astype(np.int32), 0, H - 1)
        flow_vectors = data.flow_nhw2[i_frame - 1, y_clipped, x_clipped]  # [N, 2] 格式：[dx, dy]
        stroke_flow = previous_flow + flow_vectors
        buffers.flow[i_frame] = stroke_flow

        # fitted 传播：使用当前帧的 fitted stroke 位置采样光流
        x_fit, y_fit = stroke_copied[:, 0], stroke_copied[:, 1]
        x_fit_clipped = np.clip(x_fit.astype(np.int32), 0, W - 1)
        y_fit_clipped = np.clip(y_fit.astype(np.int32), 0, H - 1)
        flow_vectors_fit = data.flow_nhw2[i_frame - 1, y_fit_clipped, x_fit_clipped]  # [N, 2]
        stroke_fitted = stroke_copied + flow_vectors_fit
        stroke_fitted = local_snapping(
            stroke_fitted,
            data.images_rgb[i_frame],
            points_stroke_candidate,
        )
        buffers.fitted[i_frame] = stroke_fitted


def propagate_strokes_with_snapping_flow(data: StrokeData, buffers: StrokeBuffers, stroke_initial: np.ndarray) -> None:
    """Full propagation pipeline for a single initial stroke polyline."""

    n_frame = data.images_rgb.shape[0]
    buffers.reset(n_frame)
    buffers.flow[0] = None
    buffers.snapping[0] = None
    buffers.fitted[0] = None
    generate_prediction_stroke_on_0(buffers, data, stroke_initial)
    generate_prediction_strokes_subsequent(buffers, data)


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

    stroke_fitted = buffers.fitted[state.current_frame]
    if stroke_fitted is not None and state.show_fitted:
        cv2.polylines(canvas, [stroke_fitted.astype(np.int32)], False, rgb_to_bgr(COLOR_FITTED), THICKNESS, lineType=cv2.LINE_AA)


def propagate_current_stroke(context: RuntimeContext) -> None:
    """Re-run propagation for the stroke chosen by the viewer."""

    stroke = context.strokes_library[context.viewer.current_stroke_index]
    propagate_strokes_with_snapping_flow(context.data, context.buffers, stroke)
    export_stroke_gifs(context)


def export_stroke_gifs(context: RuntimeContext) -> None:
    """Export stroke propagation results to GIFs grouped by stroke categories."""

    env = context.env
    data = context.data
    buffers = context.buffers
    n_frame = data.images_rgb.shape[0]
    target_dir = env.paths.debug / "results" / env.target_name
    target_dir.mkdir(parents=True, exist_ok=True)

    stroke_categories = (
        ("fitted", buffers.fitted, COLOR_FITTED),
        ("flow", buffers.flow, COLOR_FLOW),
        ("snapping", buffers.snapping, COLOR_SNAPPING),
    )

    for name, strokes, color_rgb in stroke_categories:
        frames_bgr: List[np.ndarray] = []

        for idx in tqdm(range(n_frame), desc=f"Building {name} GIF", unit=" frame(s)"):
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
    print(f"Loaded optical flow cache: {flow_nhw2_float32.shape}, {flow_nhw2_float32.dtype}")

    points_all_candidates = compute_all_candidates(images_rgb_nhwc_uint8)
    generate_salient_images(env, points_all_candidates, images_rgb_nhwc_uint8.shape[1], images_rgb_nhwc_uint8.shape[2])
    kd_tree_groups = BatchKDTree(points_all_candidates)

    data = StrokeData(
        images_rgb=images_rgb_nhwc_uint8,
        flow_nhw2=flow_nhw2_float32,
        kd_tree=kd_tree_groups,
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
    def toggle_origin_visibility() -> bool:
        viewer.show_origin = not viewer.show_origin
        return False

    @registry.register('x')
    def toggle_flow_visibility() -> bool:
        viewer.show_flow = not viewer.show_flow
        return False

    @registry.register('c')
    def toggle_snapping_visibility() -> bool:
        viewer.show_snapping = not viewer.show_snapping
        return False

    @registry.register('v')
    def toggle_fitted_visibility() -> bool:
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

        cv2.imshow(context.env.target_name, canvas)


if __name__ == '__main__':
    main()
