import sys
import shutil
import tempfile
import argparse
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
from PIL import Image
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
    """基于Florence-2的提示框提取类"""
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
    """基于SAM2的视频序列Mask提取类 (Anchor & Grow 防溢出版)"""
    
    def __init__(self, sam2_config_path=None, sam2_checkpoint_path=None, device=None):
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        base, _, _, _, _, _ = build_project_paths()
        
        if sam2_config_path is None:
            config_path = base / "sam2_git" / "sam2" / "configs" / "sam2.1" / "sam2.1_hiera_b+.yaml"
            ckpt_path = base / "sam2_git" / "checkpoints" / "sam2.1_hiera_base_plus.pt"
            if not config_path.exists():
                config_path = base / "sam2_git" / "sam2" / "configs" / "sam2.1" / "sam2.1_hiera_l.yaml"
                ckpt_path = base / "sam2_git" / "checkpoints" / "sam2.1_hiera_large.pt"
            sam2_config_path = str(config_path)
            sam2_checkpoint_path = str(ckpt_path)

        print(f"✅ 初始化 SAM2 Video Predictor")
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
        
        base, _, cache_dir, _, result_dir, _ = build_project_paths()
        cache_path = cache_dir / "mask" / f"{target_name}.pt"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 强制重新推理
        if cache_path.exists():
            print(f"🗑️  删除旧缓存: {cache_path}")
            cache_path.unlink()

        temp_dir = tempfile.mkdtemp(prefix="sam2_video_")
        try:
            for i, fp in enumerate(tqdm(frame_paths, desc="准备视频帧")):
                Image.open(fp).convert("RGB").save(Path(temp_dir) / f"{i:05d}.jpg", quality=95)

            inference_state = self.video_predictor.init_state(video_path=temp_dir)
            
            # --- Prompt Logic (仅使用 Box，移除 Point) ---
            # 原因：对于不规则物体，中心点可能误中背景，导致模型学习错误的纹理特征
            box_tensor = torch.tensor(
                [[initial_box[0], initial_box[1]], [initial_box[2], initial_box[3]]],
                dtype=torch.float32, device=self.device
            )
            # 即使不使用点提示，也需要传递空的 points 和 labels（在正确设备上）
            # 因为 SAM2 内部会尝试连接 box_coords 和 points
            empty_points = torch.zeros(1, 0, 2, dtype=torch.float32, device=self.device)
            empty_labels = torch.zeros(1, 0, dtype=torch.int32, device=self.device)
            self.video_predictor.add_new_points_or_box(
                inference_state=inference_state, 
                frame_idx=0, 
                obj_id=1, 
                box=box_tensor,
                points=empty_points,
                labels=empty_labels
            )
            
            print("🚀 开始视频传播 (Anchor & Grow 策略)...")
            masks = []
            
            # 预定义 Kernel
            # 开运算：用于切断物体与背景之间细微的粘连
            kernel_cut = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)) 
            # 闭运算：用于连接物体内部的断裂
            kernel_heal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                for _, _, pred_masks in tqdm(self.video_predictor.propagate_in_video(inference_state), total=len(frame_paths), desc="推理进度"):
                    if len(pred_masks) > 0:
                        logits = pred_masks[0][0].cpu().numpy()
                        
                        # --- 核心策略：Anchor (高信度) + Candidate (低信度) ---
                        
                        # 1. 生成 Core Mask (高阈值 > 0.0)
                        # 这是绝对可信的区域，但可能有洞
                        mask_core = (logits > 0.0).astype(np.uint8)
                        
                        # 2. 生成 Candidate Mask (低阈值 > -1.5)
                        # 这包含完整皮毛，但也包含粘连的背景
                        mask_candidate = (logits > -1.5).astype(np.uint8)
                        
                        # 3. 切断粘连 (关键步骤)
                        # 对 Candidate 做一次开运算，把那些细细的连接线（连接到背景石头的）切断
                        mask_candidate = cv2.morphologyEx(mask_candidate, cv2.MORPH_OPEN, kernel_cut)
                        
                        # 4. 条件生长 (Reconstruction)
                        # 只有与 Core Mask 有交集的 Candidate 区域才被保留
                        # 这就像是：只保留那些"以此为核心生长出来"的陆地
                        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_candidate, connectivity=8)
                        
                        final_mask_bin = np.zeros_like(mask_candidate)
                        
                        # 遍历所有连通块 (从1开始，0是背景)
                        for i in range(1, num_labels):
                            # 创建当前块的掩码
                            component_mask = (labels == i)
                            
                            # 检查该块是否包含 Core Mask 的像素
                            # 如果这个低阈值块里 包含了 高阈值的像素，说明它是物体的一部分
                            if np.logical_and(component_mask, mask_core).any():
                                final_mask_bin[component_mask] = 1
                        
                        # 5. 最终愈合
                        # 此时背景已经去掉了，可以大胆地进行闭运算和填孔
                        final_mask_bin = cv2.morphologyEx(final_mask_bin, cv2.MORPH_CLOSE, kernel_heal)
                        
                        # 填孔
                        contours, _ = cv2.findContours(final_mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if contours:
                            # 再次保险：只取最大的那个（防止有些背景噪点正好也包含了高信度像素，虽然概率很低）
                            max_cnt = max(contours, key=cv2.contourArea)
                            final_output = np.zeros_like(final_mask_bin)
                            cv2.drawContours(final_output, [max_cnt], -1, 1, thickness=cv2.FILLED)
                        else:
                            final_output = final_mask_bin

                        masks.append(final_output.astype(np.float32))
                    else:
                        masks.append(np.zeros((1024, 1024), dtype=np.float32))

            masks = np.array(masks)

        finally:
            if Path(temp_dir).exists(): shutil.rmtree(temp_dir)

        if use_cache: 
            print(f"💾 更新缓存: {cache_path}")
            torch.save(torch.from_numpy(masks), str(cache_path))
            
        if save_visualization: self._save_visualizations(masks, frame_paths, target_name, result_dir)
        if debug_frames: self._generate_debug_images(masks, frame_paths, initial_box, debug_frames, result_dir / "mask" / target_name / "debug")
        
        return masks

    def _save_visualizations(self, masks, frame_paths, target_name, result_dir):
        out_dir = result_dir / "mask" / target_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for m, fp in zip(masks, frame_paths):
            cv2.imwrite(str(out_dir / f"{fp.stem}_mask.png"), (m * 255).astype(np.uint8))

    def _generate_debug_images(self, masks, frame_paths, box, debug_frames, debug_dir):
        debug_dir.mkdir(parents=True, exist_ok=True)
        print(f"🐛 生成 Debug 图像 -> {debug_dir}")
        for idx in debug_frames:
            if idx >= len(masks): continue
            img = cv2.cvtColor(cv2.imread(str(frame_paths[idx])), cv2.COLOR_BGR2RGB)
            mask = masks[idx]
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            ax[0].imshow(img)
            if idx == 0:
                ax[0].add_patch(Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], linewidth=2, edgecolor='red', facecolor='none'))
            ax[0].set_title(f"Frame {idx}")
            ax[1].imshow(img)
            ax[1].imshow(mask, alpha=0.6, cmap='spring') 
            ax[1].set_title("Result")
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

    base, _, _, frame_dir, _, _ = build_project_paths()
    target_name = args.target_name or get_target_name(frame_dir)
    frame_paths = sorted([p for p in frame_dir.iterdir() if p.suffix.lower() in (".jpg", ".png")])
    if not frame_paths: raise ValueError("无图片")

    print(f"🎯 目标: {args.text_prompt}")
    box_extractor = FlorenceBoxExtractor()
    boxes = box_extractor.extract_boxes(Image.open(frame_paths[0]).convert("RGB"), args.text_prompt)
    if not boxes: raise ValueError("未找到目标Box")
    
    # --- 修正点：收紧 Box Padding ---
    # 之前是 0.2 (20%)，对于容易与背景混淆的物体，太大的框会包含太多背景干扰
    # 现在改为 0.05 (5%)，只留一点点余量
    ref_box = np.array(boxes[0], dtype=np.float32)
    w, h = ref_box[2] - ref_box[0], ref_box[3] - ref_box[1]
    pad_x, pad_y = max(w * 0.05, 5), max(h * 0.05, 5) # 极小的 Padding
    ref_box[0] = max(0, ref_box[0] - pad_x)
    ref_box[1] = max(0, ref_box[1] - pad_y)
    ref_box[2] += pad_x
    ref_box[3] += pad_y
    
    extractor = MaskExtractor()
    extractor.extract_masks_from_video(
        frame_paths=frame_paths,
        initial_box=ref_box,
        target_name=target_name,
        use_cache=not args.no_cache, # 这里保留逻辑，但在函数内部我们强制删除了旧缓存
        debug_frames=list(range(0, 10)) + list(range(60, 70)) # 检查开头和中间
    )
    print("✅ 全部完成")

if __name__ == "__main__":
    main()