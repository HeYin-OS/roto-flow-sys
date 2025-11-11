import numpy as np
import cv2
import torch
import torch.nn.functional as F
import os
import torchvision
from torch import Tensor
from torchvision.transforms import v2

REPO_DIR = "../dinov3"

dinov3_vitb16 = torch.hub.load(REPO_DIR, 'dinov3_vitb16', source='local', weights="../dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth")


def make_transform(resize_size: int = 256):
    to_tensor = v2.ToImage()
    resize = v2.Resize((resize_size, resize_size), antialias=True)
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return v2.Compose([to_tensor, resize, to_float, normalize])


def read_video(path: str, ext: str, frame_num: int = 0, start_file_name: str = "00000") -> Tensor | None:
    if ext == "mp4":
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        video_np = np.stack(frames, 0)
        video_tensor = torch.from_numpy(video_np).permute(0, 3, 1, 2).float()
        return video_tensor

    elif ext == "jpg":
        out = []
        start_idx = int(start_file_name)
        for i in range(frame_num):
            frame_name = f"{start_idx + i:05d}.{ext}"
            img_path = os.path.join(path, frame_name)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img_tensor = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()
            out.append(img_tensor)
        return torch.stack(out, 0)
    return None


def main():
    cap_tensor = read_video("../test/images/bear/", ext="jpg", frame_num=82)
    print(cap_tensor.shape)
    print(dinov3_vitb16)


if __name__ == "__main__":
    main()
