from pathlib import Path
from typing import Any, Callable, Dict, List

import cv2
import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from utils.edge_snapping import compute_all_candidates, EdgeSnappingConfig, local_snapping
from utils.kd_tree import BatchKDTree
from utils.raft_predictor import RAFTPredictor
from utils.yaml_reader import YamlUtil


BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"
CACHE_DIR = BASE_DIR / "caches"
STROKE_DIR = BASE_DIR / "stroke"
DEBUG_DIR = BASE_DIR / "debug"

_config_path = CONFIG_DIR / "test_video_init.yaml"
path_head_raw = YamlUtil.read(str(_config_path))['video']['url_head']
path_head = Path(path_head_raw)
if not path_head.is_absolute():
    path_head = (BASE_DIR / path_head).resolve()
else:
    path_head = path_head.resolve()

target_name = path_head.name
stroke_save_folder_path = STROKE_DIR / target_name

print(f"tracing target: {target_name}")
print(f"frame images folder: {path_head}")


def get_frame_image_paths():
    paths = sorted(
        path for path in path_head.iterdir()
        if path.suffix.lower() in (".jpg", ".png")
    )
    return paths


def read_strokes() -> List[np.ndarray]:
    stroke_save_folder_path.mkdir(parents=True, exist_ok=True)
    out = []
    for i in range(1, 100):
        path = stroke_save_folder_path / f"stroke_{i:02d}.npy"
        if path.exists():
            out.append(np.load(str(path)).astype(np.float32))
            print(f"Loaded stroke from:  {path}")
        else:
            # print(f"{path} does not exist")
            break
    return out


def read_images_batch(paths: List[Path], flag: Any):
    out = []
    for i_path in tqdm(range(len(paths)), desc="Reading images:", unit=" image(s)"):
        img = cv2.imread(str(paths[i_path]), flag)
        out.append(img)
    return np.stack(out)


def read_optical_flow_cache() -> Tensor | None:
    cache_path = CACHE_DIR / f"{target_name}.pt"
    if cache_path.exists():
        return torch.load(str(cache_path))
    else:
        raise ValueError(f"Optical flow cache file does not exist: {cache_path}")


def generate_salient_images(points_all_candidates, height, width):
    for i_img in tqdm(range(len(points_all_candidates)), desc="Generating salient point images:", unit=" image(s)"):
        canvas = np.zeros((height, width), np.uint8)
        canvas[points_all_candidates[i_img][:, 1].astype(np.int32), points_all_candidates[i_img][:, 0].astype(np.int32)] = 255
        work_dir = DEBUG_DIR / "salient" / target_name
        work_dir.mkdir(parents=True, exist_ok=True)
        file_path = work_dir / f"{i_img:03d}.jpg"
        cv2.imwrite(str(file_path), canvas)


def generate_salient_stroke_images(points_stroke_candidates, height, width, i_frame):
    canvas = np.zeros((height, width), np.uint8)
    for i_group in range(len(points_stroke_candidates)):
        canvas[points_stroke_candidates[i_group][:, 1], points_stroke_candidates[i_group][:, 0]] = 255
    work_dir = DEBUG_DIR / "salient_stroke" / target_name
    work_dir.mkdir(parents=True, exist_ok=True)
    file_path = work_dir / f"{i_frame:03d}.jpg"
    cv2.imwrite(str(file_path), canvas)


def generate_prediction_stroke_on_0(stroke_0: np.ndarray,
                                    images_rgb_nhwc_uint8: np.ndarray,
                                    kd_tree_groups: BatchKDTree):
    """Generate prediction stroke on frame 0 using edge snapping"""
    global strokes_fitted, flag_current_frame
    points_stroke_candidate = kd_tree_groups.query_batch(0,
                                                         stroke_0,
                                                         EdgeSnappingConfig.r_s)
    stroke_0_snapped = local_snapping(stroke_0,
                                      images_rgb_nhwc_uint8[0],
                                      points_stroke_candidate)
    strokes_fitted[0] = stroke_0_snapped.astype(np.float32)


def generate_prediction_strokes_subsequent(images_rgb_nhwc: np.ndarray,
                                           kd_tree_groups: BatchKDTree,
                                           flow_nhw2: np.ndarray):
    """Generate prediction strokes from frame 1 (0 as start) to frame n-1 using edge snapping and optical flow"""
    global strokes_flow, strokes_snapping, strokes_fitted

    for i in tqdm(range(images_rgb_nhwc.shape[0] - 1), desc="Generating prediction strokes on subsequent frames:", unit=" batch"):
        i_frame = i + 1

        stroke_copied = strokes_fitted[i_frame - 1]
        points_stroke_candidate = kd_tree_groups.query_batch(i_frame,
                                                             stroke_copied,
                                                             EdgeSnappingConfig.r_s)

        # pure edge snapping strokes
        stroke_snapping = None
        if i == 0:
            stroke_snapping = local_snapping(stroke_copied,
                                             images_rgb_nhwc[i_frame],
                                             points_stroke_candidate)
        else:
            stroke_snapping = local_snapping(strokes_snapping[i_frame - 1],
                                             images_rgb_nhwc[i_frame],
                                             points_stroke_candidate)
        strokes_snapping[i_frame] = stroke_snapping

        # pure optical flow strokes
        stroke_flow = None
        if i == 0:
            x, y = stroke_copied[:, 0], stroke_copied[:, 1]
            stroke_flow = stroke_copied + flow_nhw2[i_frame - 1, y.astype(np.int32), x.astype(np.int32)]
        else:
            x, y = strokes_flow[i_frame - 1][:, 0], strokes_flow[i_frame - 1][:, 1]
            stroke_flow = strokes_flow[i_frame - 1] + flow_nhw2[i_frame - 1, y.astype(np.int32), x.astype(np.int32)]
        strokes_flow[i_frame] = stroke_flow

        # real propagated strokes
        x, y = stroke_copied[:, 0], stroke_copied[:, 1]
        stroke_fitted = stroke_copied + flow_nhw2[i_frame - 1, y.astype(np.int32), x.astype(np.int32)]
        stroke_fitted = local_snapping(stroke_fitted,
                                       images_rgb_nhwc[i_frame],
                                       points_stroke_candidate)
        strokes_fitted[i_frame] = stroke_fitted


def rgb_to_bgr(color: tuple):
    """Convert an RGB tuple/list to BGR order."""
    return color[::-1]


# all are xy-order
strokes_flow: List = []
strokes_snapping: List = []
strokes_fitted: List = []

flag_current_frame: int = 0
flag_current_test_stroke: int = 0

is_visible_origin: bool = True
is_visible_flow: bool = False
is_visible_snapping: bool = False
is_visible_fitted: bool = True

color_origin = (255, 255, 0)  # Vivid Orange
color_flow = (200, 130, 255)  # Soft Lavender
color_snapping = (0, 150, 255)  # Tech Blue
color_fitted = (50, 200, 50)  # Fresh Green

thickness = 2

# Global references for handlers
images_rgb_nhwc_uint8_global: np.ndarray | None = None
flow_nhw2_float32_global: np.ndarray | None = None
kd_tree_groups_global: BatchKDTree | None = None
strokes_test_global: List[np.ndarray] = []
n_frame_global: int = 0


def draw_curves(canvas: np.ndarray, stroke_origin: np.ndarray):
    global color_origin, color_flow, color_snapping, color_fitted
    global thickness, flag_current_frame
    global strokes_flow, strokes_snapping, strokes_fitted
    global is_visible_origin, is_visible_flow, is_visible_snapping, is_visible_fitted

    # print(stroke_origin)
    # print(f"stroke origin shape: {stroke_origin.shape}, dtype:{stroke_origin.dtype}")

    # the input original stroke
    if is_visible_origin:
        cv2.polylines(canvas, [stroke_origin.astype(np.int32)], False, rgb_to_bgr(color_origin), thickness, lineType=cv2.LINE_AA)

    # pure optical flow stroke
    if strokes_flow[flag_current_frame] is not None and is_visible_flow:
        cv2.polylines(canvas, [strokes_flow[flag_current_frame].astype(np.int32)], False, rgb_to_bgr(color_flow), thickness, lineType=cv2.LINE_AA)

    # pure snapped stroke
    if strokes_snapping[flag_current_frame] is not None and is_visible_snapping:
        cv2.polylines(canvas, [strokes_snapping[flag_current_frame].astype(np.int32)], False, rgb_to_bgr(color_snapping), thickness, lineType=cv2.LINE_AA)

    # fitted stroke
    if strokes_fitted[flag_current_frame] is not None and is_visible_fitted:
        cv2.polylines(canvas, [strokes_fitted[flag_current_frame].astype(np.int32)], False, rgb_to_bgr(color_fitted), thickness, lineType=cv2.LINE_AA)


def init_stroke_system(n_frame: int):
    global strokes_flow, strokes_snapping, strokes_fitted

    strokes_flow = []
    strokes_snapping = []
    strokes_fitted = []

    for i_frame in range(n_frame):
        strokes_flow.append(None)
        strokes_snapping.append(None)
        strokes_fitted.append(None)


def propagate_strokes_with_snapping_flow(flow_nhw2_float32: np.ndarray,
                                         images_rgb_nhwc_uint8: np.ndarray,
                                         kd_tree_groups: BatchKDTree,
                                         n_frame: int,
                                         strokes_0: np.ndarray):
    init_stroke_system(n_frame)
    generate_prediction_stroke_on_0(strokes_0,
                                    images_rgb_nhwc_uint8,
                                    kd_tree_groups)
    generate_prediction_strokes_subsequent(images_rgb_nhwc_uint8,
                                           kd_tree_groups,
                                           flow_nhw2_float32)


def main():
    global images_rgb_nhwc_uint8_global
    global flow_nhw2_float32_global
    global kd_tree_groups_global
    global strokes_test_global
    global n_frame_global
    global flag_current_frame
    global flag_current_test_stroke
    global is_visible_origin
    global is_visible_flow
    global is_visible_snapping
    global is_visible_fitted
    EdgeSnappingConfig.load(str(CONFIG_DIR / "snapping_init.yaml"))
    strokes_test = read_strokes()

    frame_image_paths = get_frame_image_paths()

    # [N, H, W, C] (RGB), val w.r.t. [0, 255]
    images_rgb_nhwc_uint8 = read_images_batch(frame_image_paths, cv2.IMREAD_COLOR_RGB)

    # [N-1, H, W, 2] float
    flow_nhw2_float32 = read_optical_flow_cache().numpy()
    print(f"Loaded optical flow cache: {flow_nhw2_float32.shape}, {flow_nhw2_float32.dtype}")

    # stroke prediction workflow
    points_all_candidates = compute_all_candidates(images_rgb_nhwc_uint8)
    generate_salient_images(points_all_candidates, images_rgb_nhwc_uint8.shape[1], images_rgb_nhwc_uint8.shape[2])
    kd_tree_groups = BatchKDTree(points_all_candidates)
    n_frame = images_rgb_nhwc_uint8.shape[0]
    propagate_strokes_with_snapping_flow(flow_nhw2_float32,
                                         images_rgb_nhwc_uint8,
                                         kd_tree_groups,
                                         n_frame,
                                         strokes_test[flag_current_test_stroke])

    images_rgb_nhwc_uint8_global = images_rgb_nhwc_uint8
    flow_nhw2_float32_global = flow_nhw2_float32
    kd_tree_groups_global = kd_tree_groups
    strokes_test_global = strokes_test
    n_frame_global = n_frame

    def register_key_handler(key_char: str) -> Callable[[Callable[[], bool]], Callable[[], bool]]:
        def decorator(func: Callable[[], bool]) -> Callable[[], bool]:
            key_handlers[ord(key_char)] = func
            return func
        return decorator

    key_handlers: Dict[int, Callable[[], bool]] = {}

    @register_key_handler('a')
    def handle_prev_frame() -> bool:
        global flag_current_frame
        if flag_current_frame > 0:
            flag_current_frame -= 1
        return False

    @register_key_handler('d')
    def handle_next_frame() -> bool:
        global flag_current_frame
        if flag_current_frame < n_frame_global - 1:
            flag_current_frame += 1
        return False

    @register_key_handler('z')
    def toggle_origin_visibility() -> bool:
        global is_visible_origin
        is_visible_origin = not is_visible_origin
        return False

    @register_key_handler('x')
    def toggle_flow_visibility() -> bool:
        global is_visible_flow
        is_visible_flow = not is_visible_flow
        return False

    @register_key_handler('c')
    def toggle_snapping_visibility() -> bool:
        global is_visible_snapping
        is_visible_snapping = not is_visible_snapping
        return False

    @register_key_handler('v')
    def toggle_fitted_visibility() -> bool:
        global is_visible_fitted
        is_visible_fitted = not is_visible_fitted
        return False

    def create_switch_test_stroke_handler(index: int) -> Callable[[], bool]:
        def handler() -> bool:
            global flag_current_test_stroke
            if index >= len(strokes_test_global):
                return False
            if index != flag_current_test_stroke:
                flag_current_test_stroke = index
                propagate_strokes_with_snapping_flow(
                    flow_nhw2_float32_global,
                    images_rgb_nhwc_uint8_global,
                    kd_tree_groups_global,
                    n_frame_global,
                    strokes_test_global[flag_current_test_stroke]
                )
            return False
        return handler

    for idx, key_char in enumerate(('1', '2', '3')):
        register_key_handler(key_char)(create_switch_test_stroke_handler(idx))

    @register_key_handler('q')
    def handle_quit() -> bool:
        return True

    # use registered handlers to process key events
    while True:
        # acquire input key
        key = cv2.waitKey(1) & 0xFF
        handler = key_handlers.get(key)
        if handler and handler():
            break

        canvas = cv2.cvtColor(images_rgb_nhwc_uint8_global[flag_current_frame], cv2.COLOR_RGB2BGR)
        draw_curves(canvas, strokes_test_global[flag_current_test_stroke])

        cv2.imshow(target_name, canvas)


if __name__ == '__main__':
    main()
