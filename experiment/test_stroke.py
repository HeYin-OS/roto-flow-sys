import os
from typing import List, Any

import cv2
import numpy as np
import torch
from torch import candidate
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
            out.append(np.load(path))
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


# all are xy-order
strokes_flow = [None]
strokes_snapping = [None]
strokes_fitted = []

color_origin = (255, 250, 205)  # lemon chiffon
color_flow = (255, 182, 193)  # light pink
color_snapping = (251, 250, 129)  # light blue
color_fitted = (144, 238, 144)  # light green


def generate_prediction_strokes(stroke_0: np.ndarray,
                                images_rgb_nhwc: np.ndarray,
                                points_all_image_candidates: List[np.ndarray],
                                flow_nhw2: np.ndarray):
    # clear prediction history
    global strokes_flow, strokes_snapping, strokes_fitted

    strokes_flow = [None]
    strokes_snapping = [None]
    strokes_fitted = []

    # fit the curve on frame 0
    stroke_0_fitted = local_snapping(stroke_0, images_rgb_nhwc, points_all_image_candidates)
    strokes_fitted.append(stroke_0_fitted)

    # kd-tree

    kd_tree_all_candidates = BatchKDTree(points_all_image_candidates)


def draw_curves(canvas: np.ndarray):
    pass


def main():
    EdgeSnappingConfig.load("../config/snapping_init.yaml")

    strokes_test = read_strokes()

    frame_image_paths = get_frame_image_paths()

    # [N, H, W, C] (RGB), val w.r.t. [0, 1]
    images_rgb_nhwc = read_images_batch(frame_image_paths, cv2.IMREAD_COLOR_RGB)

    # [N-1, H, W, 2] equal to [num of consecutive two frames, height, width, x and y]
    flow_nhw2 = read_optical_flow_cache().numpy()
    print(f"Loaded optical flow cache: {flow_nhw2.shape}, {flow_nhw2.dtype}")

    points_all_candidates = compute_all_candidates(images_rgb_nhwc)

    # TODO: interactive viewing system by cv2
    n_frame = images_rgb_nhwc.shape[0]
    H = images_rgb_nhwc.shape[1]
    W = images_rgb_nhwc.shape[2]

    flag_current_frame: int = 0
    flag_current_test_stroke: int = 0

    # generate_prediction_strokes()

    kd_tree_groups = BatchKDTree(points_all_candidates)

    points_stroke_candidate = kd_tree_groups.query_batch(0, strokes_test[0], EdgeSnappingConfig.r_s)

    stroke_snapped = local_snapping(strokes_test[0], images_rgb_nhwc[0], points_stroke_candidate)

    # use A D to switch frames
    # while True:
    #     # acquire input key
    #     key = cv2.waitKey(1) & 0xFF
    #     if key == ord('a'):
    #         if flag_current_frame > 0:
    #             flag_current_frame = flag_current_frame - 1
    #     if key == ord('d'):
    #         if flag_current_frame < n_frame - 1:
    #             flag_current_frame = flag_current_frame + 1
    #     if key == ord('q'):
    #         break
    #
    #     canvas = cv2.cvtColor(images_rgb_nhwc[flag_current_frame], cv2.COLOR_RGB2BGR)
    #
    #     cv2.imshow(target_name, canvas)

    # TODO: flow + candidates + stroke -> prediction curve


if __name__ == '__main__':
    main()
