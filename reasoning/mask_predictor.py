import sys
import shutil
import tempfile
import argparse
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_fill_holes
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# HuggingFace & SAM2 imports
from transformers import AutoProcessor, AutoModelForCausalLM
from sam2.build_sam import build_sam2_video_predictor

# 项目内部引用
_file_path = Path(__file__).resolve()
_project_root = _file_path.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
from utils.path_utils import build_project_paths, get_target_name

class FlorenceBoxExtractor:
    """基于Florence-2的提示框提取类 (保持不变)"""
    def __init__(self, device: Optional[str] = None):
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"✅ 初始化 Florence-2 ({self.device})")
        self.model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)
    
    def extract_boxes(self, image: Image.Image, text_prompt: str) -> List[np.ndarray]:
        task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
        inputs = self.processor(text=task_prompt + text_prompt, images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            generated_ids = self.model.generate(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], max_new_tokens=1024, num_beams=3)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        results = self.processor.post_process_generation(generated_text, task=task_prompt, image_size=(image.width, image.height))
        return results[task_prompt]['bboxes']

class MaskExtractor:
    """基于SAM2的视频序列Mask提取类 (精简版)"""
    
    def __init__(self, sam2_config_path=None, sam2_checkpoint_path=None, device=None):
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        base, _, _, _, _, _ = build_project_paths()
        
        # 路径自动判定
        if sam2_config_path is None:
            # 优先使用 Base+ 配置
            config_path = base / "sam2_git" / "sam2" / "configs" / "sam2.1" / "sam2.1_hiera_b+.yaml"
            ckpt_path = base / "sam2_git" / "checkpoints" / "sam2.1_hiera_base_plus.pt"
            if not config_path.exists(): # Fallback to Large
                config_path = base / "sam2_git" / "sam2" / "configs" / "sam2.1" / "sam2.1_hiera_l.yaml"
                ckpt_path = base / "sam2_git" / "checkpoints" / "sam2.1_hiera_large.pt"
            sam2_config_path = str(config_path)
            sam2_checkpoint_path = str(ckpt_path)

        print(f"✅ 初始化 SAM2 Video Predictor")
        print(f"   Config: {Path(sam2_config_path).name}")
        
        self.video_predictor = build_sam2_video_predictor(
            sam2_config_path, sam2_checkpoint_path, device=self.device, apply_postprocessing=True
        )

    def extract_masks_from_video(
        self,
        frame_paths: List[Path],
        initial_box: np.ndarray,
        target_name: str,
        save_visualization: bool = True,
        use_cache: bool = True,
        debug_frames: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        核心逻辑：视频Mask提取 + 强力后处理
        """
        base, _, cache_dir, _, result_dir, _ = build_project_paths()
        cache_path = cache_dir / "mask" / f"{target_name}.pt"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 1. 缓存检查
        if use_cache and cache_path.exists():
            print(f"📦 加载缓存: {cache_path}")
            masks = torch.load(str(cache_path))
            if isinstance(masks, torch.Tensor): masks = masks.numpy()
            if save_visualization: self._save_visualizations(masks, frame_paths, target_name, result_dir)
            # 缓存模式下的 Debug
            if debug_frames: self._generate_debug_images(masks, frame_paths, initial_box, debug_frames, result_dir / "mask" / target_name / "debug")
            return masks

        # 2. 准备临时视频目录 (SAM2 Video API Requirement)
        temp_dir = tempfile.mkdtemp(prefix="sam2_video_")
        try:
            for i, fp in enumerate(tqdm(frame_paths, desc="准备视频帧")):
                Image.open(fp).convert("RGB").save(Path(temp_dir) / f"{i:05d}.jpg", quality=95)

            # 3. 初始化推理状态
            inference_state = self.video_predictor.init_state(video_path=temp_dir)
            
            # 4. 首帧提示 (Box)
            box_tensor = torch.tensor(
                [[initial_box[0], initial_box[1]], [initial_box[2], initial_box[3]]],
                dtype=torch.float32, device=self.device
            )
            # 使用 Box 提示 (frame 0)
            self.video_predictor.add_new_points_or_box(
                inference_state=inference_state, frame_idx=0, obj_id=1, box=box_tensor
            )
            
            # 5. 视频传播与后处理 (关键步骤)
            print("🚀 开始视频传播与后处理...")
            masks = []
            
            # 预定义形态学核
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)) # 强力闭合核

            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                for _, _, pred_masks in tqdm(self.video_predictor.propagate_in_video(inference_state), total=len(frame_paths)):
                    if len(pred_masks) > 0:
                        # 获取 Logits (未Sigmoid的值)
                        logits = pred_masks[0][0].cpu().numpy()
                        
                        # --- 修复孔洞的核心逻辑 ---
                        # A. 极低阈值: 包含所有潜在像素 (-1.0 ≈ 27% prob)
                        mask_bin = (logits > -1.0).astype(np.uint8)
                        
                        # B. 开运算: 去除低阈值引入的微小噪点
                        mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel_open)
                        
                        # C. 闭运算: 强力连接断裂的内部区域 (如皮毛)
                        mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kernel_close)
                        
                        # D. 填充孔洞: 填补任何剩余的封闭空洞
                        mask_bin = binary_fill_holes(mask_bin).astype(np.float32)
                        
                        masks.append(mask_bin)
                    else:
                        masks.append(np.zeros_like(masks[-1]) if masks else np.zeros((1024, 1024), dtype=np.float32))

            masks = np.array(masks)

        finally:
            if Path(temp_dir).exists(): shutil.rmtree(temp_dir)

        # 6. 保存与输出
        if use_cache: torch.save(torch.from_numpy(masks), str(cache_path))
        if save_visualization: self._save_visualizations(masks, frame_paths, target_name, result_dir)
        if debug_frames: self._generate_debug_images(masks, frame_paths, initial_box, debug_frames, result_dir / "mask" / target_name / "debug")
        
        return masks

    def _save_visualizations(self, masks, frame_paths, target_name, result_dir):
        out_dir = result_dir / "mask" / target_name
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"🎨 保存可视化 -> {out_dir}")
        for m, fp in zip(masks, frame_paths):
            cv2.imwrite(str(out_dir / f"{fp.stem}_mask.png"), (m * 255).astype(np.uint8))

    def _generate_debug_images(self, masks, frame_paths, box, debug_frames, debug_dir):
        """生成Debug图 (直接使用计算好的Mask)"""
        debug_dir.mkdir(parents=True, exist_ok=True)
        print(f"🐛 生成 Debug 图像 -> {debug_dir}")
        
        for idx in debug_frames:
            if idx >= len(masks): continue
            
            img = cv2.cvtColor(cv2.imread(str(frame_paths[idx])), cv2.COLOR_BGR2RGB)
            mask = masks[idx]
            
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            
            # 左图：原图 + Box (如果是第一帧)
            ax[0].imshow(img)
            if idx == 0:
                ax[0].add_patch(Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], 
                                        linewidth=2, edgecolor='red', facecolor='none'))
            ax[0].set_title(f"Frame {idx} Original")
            
            # 右图：Mask 覆盖
            ax[1].imshow(img)
            ax[1].imshow(mask, alpha=0.5, cmap='spring') # 半透明覆盖
            ax[1].set_title("Processed Mask Overlay")
            
            for a in ax: a.axis('off')
            plt.tight_layout()
            plt.savefig(debug_dir / f"debug_{idx:04d}.png")
            plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_prompt", type=str, default="bear")
    parser.add_argument("--target_name", type=str, default=None)
    parser.add_argument("--no_cache", action="store_true")
    args = parser.parse_args()

    # 路径设置
    base, _, _, frame_dir, _, _ = build_project_paths()
    target_name = args.target_name or get_target_name(frame_dir)
    frame_paths = sorted([p for p in frame_dir.iterdir() if p.suffix.lower() in (".jpg", ".png")])
    if not frame_paths: raise ValueError("无图片")

    # 1. Florence-2 提取 Box
    print(f"🎯 目标: {args.text_prompt}")
    box_extractor = FlorenceBoxExtractor()
    boxes = box_extractor.extract_boxes(Image.open(frame_paths[0]).convert("RGB"), args.text_prompt)
    if not boxes: raise ValueError("未找到目标Box")
    
    # 扩大 Box (Padding) - 应对物体移动
    ref_box = np.array(boxes[0], dtype=np.float32)
    w, h = ref_box[2] - ref_box[0], ref_box[3] - ref_box[1]
    pad_x, pad_y = max(w * 0.3, 40), max(h * 0.3, 40)
    ref_box[0] = max(0, ref_box[0] - pad_x)
    ref_box[1] = max(0, ref_box[1] - pad_y)
    ref_box[2] += pad_x
    ref_box[3] += pad_y
    
    # 2. SAM2 视频分割
    extractor = MaskExtractor()
    extractor.extract_masks_from_video(
        frame_paths=frame_paths,
        initial_box=ref_box,
        target_name=target_name,
        use_cache=not args.no_cache,
        debug_frames=list(range(60, 81)) # 默认检查容易出问题的帧段
    )
    print("✅ 全部完成")

if __name__ == "__main__":
    main()