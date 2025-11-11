import os
from typing import List

import cv2
import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from torchvision.utils import flow_to_image

from utils.raft_predictor import RAFTPredictor
from utils.yaml_reader import YamlUtil

path_head = "." + YamlUtil.read("../config/test_video_init.yaml")['video']['url_head']
target_name = path_head.split('/')[-1]
stroke_save_folder_path = "../stroke/" + target_name + "/"

print(f"frame images folder: {path_head}")


def get_frame_image_paths():
    out = sorted(
        os.path.join(path_head, file_name)
        for file_name in os.listdir(path_head)
        if file_name.endswith((".jpg", ".png"))
    )
    return out


def read_image_for_RAFT(paths: List[str]):
    out = []
    for path in paths:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        out.append(np.transpose(img, (2, 0, 1)))

    out_stacked = np.stack(out).astype(np.float32) / 255.0

    return out_stacked


def compute_optical_flow_vector_fields(images_np: np.ndarray):
    # if it has cache, load the cache
    cache_path = "../caches/" + target_name + ".pt"

    if os.path.exists(cache_path):
        print(f"Already exists at: {cache_path}")
        return torch.load(cache_path)

    # compute the vector fields
    images_tensor = torch.from_numpy(images_np)

    frames_1_batch = images_tensor[:-1]
    frames_2_batch = images_tensor[1:]

    flow_list = []

    predictor = RAFTPredictor()

    for i in tqdm(range(frames_1_batch.shape[0]), desc="Handling frame batch:", unit="batch"):
        flow = predictor.compute_optical_flow_single(frames_1_batch[i], frames_2_batch[i])
        flow_list.append(flow)

    results = torch.stack(flow_list, dim=0)

    torch.save(results, cache_path)

    print(f"Saved results: {cache_path}")

    return results


def write_rgb_flow_images(flow: Tensor):
    flow_imgs = flow_to_image(flow.permute(0, 3, 1, 2))
    flow_imgs_np = flow_imgs.permute(0, 2, 3, 1).cpu().numpy()

    work_dir = "../debug/flow/" + target_name + "/"

    os.makedirs(work_dir)

    for i in tqdm(range(flow_imgs_np.shape[0]), desc="Writing RGB flow images:", unit="image"):
        path = work_dir + f"frame_{i:2d}_to_{i + 1:2d}.jpg"
        cv2.imwrite(path, flow_imgs_np[i])


if __name__ == "__main__":
    # [N], all frame image paths with dict order
    frame_image_paths = get_frame_image_paths()

    print(f"frame number: {len(frame_image_paths)}")

    # [N, C, H, W], data w.r.t. [0, 1]
    images_ready = read_image_for_RAFT(frame_image_paths)

    print(f"images info: {images_ready[0].shape}")

    # [N-1, H, W, 2]
    flow = compute_optical_flow_vector_fields(images_ready)

    print(f"flow info: {flow.shape}")

    # [N-1, C, H, W]
    write_rgb_flow_images(flow)
