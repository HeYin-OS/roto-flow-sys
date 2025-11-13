from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from torchvision.utils import flow_to_image

from utils.raft_predictor import RAFTPredictor
from utils.yaml_reader import YamlUtil


def build_project_paths() -> tuple[Path, Path, Path, Path, Path]:
    """按照 test_stroke.py 的逻辑解析项目路径。"""

    base = Path(__file__).resolve().parent.parent
    config_dir = base / "config"
    cache_dir = base / "caches"
    frame_dir_raw = YamlUtil.read(str(config_dir / "test_video_init.yaml"))['video']['url_head']

    frame_dir = (base / frame_dir_raw).resolve() if not Path(frame_dir_raw).is_absolute() else Path(frame_dir_raw).resolve()
    debug_dir = base / "debug"
    stroke_dir = base / "stroke"

    return base, config_dir, cache_dir, frame_dir, debug_dir, stroke_dir


BASE_DIR, CONFIG_DIR, CACHE_DIR, FRAME_DIR, DEBUG_DIR, STROKE_DIR = build_project_paths()
TARGET_NAME = FRAME_DIR.name

print(f"frame images folder: {FRAME_DIR}")


def get_frame_image_paths() -> List[Path]:
    """获取当前目标的帧图像路径列表。"""
    return sorted(
        path for path in FRAME_DIR.iterdir()
        if path.suffix.lower() in (".jpg", ".png")
    )


def read_image_for_RAFT(paths: List[Path]) -> np.ndarray:
    """读取并归一化用于 RAFT 的图像序列。"""
    out = []
    for path in paths:
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        out.append(np.transpose(img, (2, 0, 1)))

    out_stacked = np.stack(out).astype(np.float32) / 255.0

    return out_stacked


def compute_optical_flow_vector_fields(images_np: np.ndarray):
    """
    计算相邻帧之间的光流场。
    
    返回格式：[N-1, H, W, 2]，其中：
    - results[i] 是从 frame i 到 frame i+1 的光流
    - results[i, y, x, 0] 是 x 方向（水平）的位移
    - results[i, y, x, 1] 是 y 方向（垂直）的位移
    
    语义：results[i, y, x] 表示从 frame i 的位置 (x, y) 到 frame i+1 的位移向量 [dx, dy]
    """
    # if it has cache, load the cache
    cache_path = CACHE_DIR / f"{TARGET_NAME}.pt"

    if cache_path.exists():
        print(f"Already exists at: {cache_path}")
        return torch.load(str(cache_path))

    # compute the vector fields
    images_tensor = torch.from_numpy(images_np)

    frames_1_batch = images_tensor[:-1]  # frame i
    frames_2_batch = images_tensor[1:]    # frame i+1

    flow_list = []

    predictor = RAFTPredictor()

    for i in tqdm(range(frames_1_batch.shape[0]), desc="Handling frame batch:", unit="batch"):
        # 计算从 frame i 到 frame i+1 的光流
        flow = predictor.compute_optical_flow_single(frames_1_batch[i], frames_2_batch[i])
        flow_list.append(flow)

    results = torch.stack(flow_list, dim=0)  # [N-1, H, W, 2]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(results, str(cache_path))

    print(f"Saved results: {cache_path}")

    return results


def write_rgb_flow_images(flow: Tensor):
    flow_imgs = flow_to_image(flow.permute(0, 3, 1, 2))
    flow_imgs_np = flow_imgs.permute(0, 2, 3, 1).cpu().numpy()

    work_dir = DEBUG_DIR / "flow" / TARGET_NAME
    work_dir.mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(flow_imgs_np.shape[0]), desc="Writing RGB flow images:", unit="image"):
        path = work_dir / f"frame_{i:02d}_to_{i + 1:02d}.jpg"
        cv2.imwrite(str(path), flow_imgs_np[i])


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
