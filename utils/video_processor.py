import os
from typing import List

import PySide6
import cv2
import numpy as np
import torch
from PySide6.QtGui import QPixmap, QImage
from torch import Tensor
from tqdm import tqdm

from utils.edge_snapping import compute_all_candidates, EdgeSnappingConfig
from utils.kd_tree import BatchKDTree
from utils.raft_predictor import RAFTPredictor
from utils.yaml_reader import YamlUtil


class Video:
    def __init__(self, yaml_url):
        self.url_head: str = YamlUtil.read(yaml_url)['video']['url_head']
        self.format: str = YamlUtil.read(yaml_url)['video']['format']

        # init snapping config
        EdgeSnappingConfig.load('config/snapping_init.yaml')

        if self.format == "jpg" or self.format == "png":
            print(f"Reading video frames from designated URL: {self.url_head}/*****.{self.format}...")
        else:
            print(f"Reading video frames from designated URL: {self.url_head}.{self.format}...")

        self.frame_paths = self.makeFrameFileUrls()

        self.tensor_format = self.loadFrameSequenceTensor()
        print(f"✓ Pre-loaded frames as tensor, shape: {self.tensor_format.shape}, dtype: {self.tensor_format.dtype}")

        self.qPixmap_format = self.loadFrameSequenceQPixmap()
        print(f"✓ Pre-loaded frames as QPixmap, size: {len(self.qPixmap_format)}")

        self.optical_flow_cache = self.makeOpticalFlowCache()
        print(f"✓ Pre-computed Optical Flow Cache, shape: {len(self.optical_flow_cache.shape)}")

        candidate_on_each_frame = compute_all_candidates(self.tensor_format)
        self.candidate_kd_trees = BatchKDTree(candidate_on_each_frame)
        print(f"✓ Cached candidate points on all frames, shape: {len(candidate_on_each_frame)} of {type(candidate_on_each_frame[0])}")

        self.frame_num: int = self.tensor_format.shape[0]
        self.channel: int = self.tensor_format.shape[1]
        self.height: int = self.tensor_format.shape[2]
        self.width: int = self.tensor_format.shape[3]
        print(f"Video Info: Width {self.width}, Height {self.height}, {self.channel} Channels, {self.frame_num} Frames")

    def makeFrameFileUrls(self) -> List[str]:
        if self.format == "jpg" or self.format == "png":
            image_files = sorted(
                os.path.join(self.url_head, file_name)
                for file_name in os.listdir(self.url_head)
                if file_name.endswith((".jpg", ".png"))
            )
            return image_files
        elif self.format == "mp4":
            return [f"{self.url_head}.{self.format}"]

        return []

    def loadFrameSequenceTensor(self) -> Tensor | None:
        if self.format == "jpg" or self.format == "png":
            frames = []
            for file_name in self.frame_paths:
                frame = cv2.imread(file_name, cv2.IMREAD_COLOR)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            frames_np = np.stack(frames, axis=0)
            return torch.from_numpy(frames_np).permute(0, 3, 1, 2).to(dtype=torch.float32)
        elif self.format == "mp4":
            frames = []
            cap = cv2.VideoCapture(self.frame_paths[0])
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            cap.release()
            frames_np = np.stack(frames, axis=0)
            return torch.from_numpy(frames_np).permute(0, 3, 1, 2).to(dtype=torch.float32)
        return None

    def loadFrameSequenceQPixmap(self) -> List[PySide6.QtGui.QPixmap] | None:
        if self.format == "jpg" or self.format == "png":
            frames = []
            for file_name in self.frame_paths:
                pixmap = QPixmap(file_name)
                frames.append(pixmap)
            return frames
        elif self.format == "mp4":
            frames = []
            cap = cv2.VideoCapture(self.frame_paths[0])
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                h, w, ch = frame.shape
                bytes_per_line = ch * w
                q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

                pixmap = QPixmap.fromImage(q_img)
                frames.append(pixmap)
            cap.release()
            return frames
        return None

    def makeOpticalFlowCache(self):
        strs = self.url_head.split("/")
        cache_path = "./caches/" + strs[-1] + ".pt"

        if os.path.exists(cache_path):
            print(f"✓ Loaded optical flow cache from: {cache_path}")
            return torch.load(cache_path)

        frames1 = self.tensor_format[:-1]
        frames2 = self.tensor_format[1:]

        flow_list = []

        predictor = RAFTPredictor()
        for i in tqdm(range(frames1.shape[0]), desc="Handling frame batch:", unit="batch"):
            flow = predictor.compute_optical_flow_single(frames1[i], frames2[i])
            flow_cpu = flow.detach().to("cpu", non_blocking=True)
            del flow
            flow_list.append(flow_cpu)

        result = torch.stack(flow_list, dim=0).squeeze(1)

        torch.save(result, cache_path)
        print(f"✓ Saved optical flow cache to: {cache_path}")

        return result

    def getFrameImagePath(self):
        return f"{self.url_head}{0:05d}.{self.format}"
