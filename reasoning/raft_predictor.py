import gc
from typing import Tuple, Literal

import torch
from torchvision.models.optical_flow import Raft_Large_Weights, Raft_Small_Weights, raft_large, raft_small


class RAFTPredictor:
    def __init__(self, model_size: Literal["large", "small"] = "large"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        weights = None
        selected_model = None
        if model_size == "large":
            weights = Raft_Large_Weights.DEFAULT
            selected_model = raft_large(weights=weights).to(self.device)
        elif model_size == "small":
            weights = Raft_Small_Weights.DEFAULT
            selected_model = raft_small(weights=weights).to(self.device)

        self.model = selected_model.eval()

        self.transforms = weights.transforms()

    def make_divisible_by_8(self, size: Tuple[int, int]) -> Tuple[int, int]:
        width, height = size
        new_width = ((width + 7) // 8) * 8
        new_height = ((height + 7) // 8) * 8
        return new_width, new_height

    def compute_optical_flow_single(self, frame1: torch.Tensor, frame2: torch.Tensor) -> torch.Tensor:
        """
        estimate the optical flow from frame1 and frame2
        
        返回格式：[H, W, 2]，其中：
        - flow[:, :, 0] 是 x 方向（水平）的位移
        - flow[:, :, 1] 是 y 方向（垂直）的位移
        
        语义：flow[y, x] 表示从 frame1 的位置 (x, y) 到 frame2 的位移向量 [dx, dy]
        """
        with torch.no_grad():
            list_of_flow = self.model(frame1.unsqueeze(0).to(self.device), frame2.unsqueeze(0).to(self.device))
        return list_of_flow[-1].squeeze(0).permute(1, 2, 0).cpu()


