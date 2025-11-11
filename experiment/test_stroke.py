import os
from typing import List, Any

import cv2
import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from utils.edge_snapping import compute_all_candidates, EdgeSnappingConfig, local_snapping
from utils.kd_tree import BatchKDTree
from utils.raft_predictor import RAFTPredictor
from utils.yaml_reader import YamlUtil

path_head = "." + YamlUtil.read("../config/test_video_init.yaml")['video']['url_head']
target_name = path_head.split('/')[-1]
stroke_save_folder_path = "../stroke/" + target_name + "/"

print(f"tracing target: {target_name}")
print(f"frame images folder: {path_head}")


def get_frame_image_paths():
    paths = sorted(
        os.path.join(path_head, file_name)
        for file_name in os.listdir(path_head)
        if file_name.endswith((".jpg", ".png"))
    )
    return paths


def read_strokes() -> List[np.ndarray]:
    stroke_path_head = stroke_save_folder_path + "stroke_"
    out = []
    for i in range(1, 100):
        path = stroke_path_head + f"{i:02d}" + ".npy"
        if os.path.exists(path):
            out.append(np.load(path).astype(np.float32))
            print(f"Loaded stroke from:  {path}")
        else:
            # print(f"{path} does not exist")
            break
    return out


def read_images_batch(paths: List[str], flag: Any):
    out = []
    for i_path in tqdm(range(len(paths)), desc="Reading images:", unit=" image(s)"):
        img = cv2.imread(paths[i_path], flag)
        out.append(img)
    return np.stack(out)


def read_optical_flow_cache() -> Tensor | None:
    cache_path = "../caches/" + target_name + ".pt"
    if os.path.exists(cache_path):
        return torch.load(cache_path)
    else:
        ValueError("Optical flow cache file does not exist!")
        return None


def generate_salient_images(points_all_candidates, height, width):
    for i_img in tqdm(range(len(points_all_candidates)), desc="Generating salient point images:", unit=" image(s)"):
        canvas = np.zeros((height, width), np.uint8)
        canvas[points_all_candidates[i_img][:, 1], points_all_candidates[i_img][:, 0]] = 255

        work_dir = "../debug/salient/" + target_name + "/"

        if not os.path.exists(work_dir):
            os.makedirs(work_dir)

        file_path = work_dir + f"{i_img:03d}.jpg"

        cv2.imwrite(file_path, canvas)


def generate_salient_stroke_images(points_stroke_candidates, height, width, i_frame):
    canvas = np.zeros((height, width), np.uint8)
    for i_group in range(len(points_stroke_candidates)):
        canvas[points_stroke_candidates[i_group][:, 1], points_stroke_candidates[i_group][:, 0]] = 255

    work_dir = "../debug/salient_stroke/" + target_name + "/"

    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    file_path = work_dir + f"{i_frame:03d}.jpg"

    cv2.imwrite(file_path, canvas)


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
    global flag_current_frame, is_visible_origin, is_visible_flow, is_visible_snapping, is_visible_fitted
    EdgeSnappingConfig.load("../config/snapping_init.yaml")
    strokes_test = read_strokes()

    frame_image_paths = get_frame_image_paths()

    # [N, H, W, C] (RGB), val w.r.t. [0, 255]
    images_rgb_nhwc_uint8 = read_images_batch(frame_image_paths, cv2.IMREAD_COLOR_RGB)

    # [N-1, H, W, 2] float
    flow_nhw2_float32 = read_optical_flow_cache().numpy()
    print(f"Loaded optical flow cache: {flow_nhw2_float32.shape}, {flow_nhw2_float32.dtype}")

    # stroke prediction workflow
    points_all_candidates = compute_all_candidates(images_rgb_nhwc_uint8)
    kd_tree_groups = BatchKDTree(points_all_candidates)
    n_frame = images_rgb_nhwc_uint8.shape[0]
    propagate_strokes_with_snapping_flow(flow_nhw2_float32,
                                         images_rgb_nhwc_uint8,
                                         kd_tree_groups,
                                         n_frame,
                                         strokes_test[flag_current_test_stroke])

    # use A D to switch frames
    # z x c d to switch stroke visibility
    # 1 2 3 to switch tested stroke index
    while True:
        # acquire input key
        key = cv2.waitKey(1) & 0xFF
        if key == ord('a'):
            if flag_current_frame > 0:
                flag_current_frame = flag_current_frame - 1
        elif key == ord('d'):
            if flag_current_frame < n_frame - 1:
                flag_current_frame = flag_current_frame + 1
        elif key == ord('z'):
            is_visible_origin = not is_visible_origin
        elif key == ord('x'):
            is_visible_flow = not is_visible_flow
        elif key == ord('c'):
            is_visible_snapping = not is_visible_snapping
        elif key == ord('v'):
            is_visible_fitted = not is_visible_fitted
        elif key == ord('q'):
            break

        canvas = cv2.cvtColor(images_rgb_nhwc_uint8[flag_current_frame], cv2.COLOR_RGB2BGR)
        draw_curves(canvas, strokes_test[flag_current_test_stroke])

        cv2.imshow(target_name, canvas)


if __name__ == '__main__':
    main()
