"""
基于Florence-2和SAM2的mask提取模块
提供提示框提取和视频序列mask提取功能
"""
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

# 添加项目根目录到Python路径，以便导入utils模块
_file_path = Path(__file__).resolve()
_project_root = _file_path.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from tqdm import tqdm

from utils.path_utils import build_project_paths, get_target_name


class FlorenceBoxExtractor:
    """基于Florence-2的提示框提取类"""
    
    def __init__(self, device: Optional[str] = None):
        """
        初始化Florence-2模型
        
        Args:
            device: 计算设备，默认为cuda（如果可用）或cpu
        """
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"初始化Florence-2，使用设备: {self.device}")
        
        # 加载Florence-2模型
        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Florence-2-base", 
            trust_remote_code=True
        ).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(
            "microsoft/Florence-2-base", 
            trust_remote_code=True
        )
        print("✅ Florence-2模型加载完成")
    
    def extract_boxes(self, image: Image.Image, text_prompt: str) -> List[np.ndarray]:
        """
        从图像中提取文本提示对应的边界框
        
        Args:
            image: PIL Image对象，RGB格式
            text_prompt: 文本提示，例如"bear"
        
        Returns:
            边界框列表，每个框为[x1, y1, x2, y2]格式的numpy数组
        """
        task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
        inputs = self.processor(
            text=task_prompt + text_prompt, 
            images=image, 
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3
            )
        
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        results = self.processor.post_process_generation(
            generated_text, 
            task=task_prompt, 
            image_size=(image.width, image.height)
        )
        
        bboxes = results[task_prompt]['bboxes']
        return bboxes


class MaskExtractor:
    """基于SAM2的视频序列mask提取类"""
    
    def __init__(
        self,
        sam2_config_path: Optional[str] = None,
        sam2_checkpoint_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        初始化SAM2模型
        
        Args:
            sam2_config_path: SAM2配置文件路径，如果为None则使用默认路径
            sam2_checkpoint_path: SAM2检查点路径，如果为None则使用默认路径
            device: 计算设备，默认为cuda（如果可用）或cpu
        """
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # 获取项目根目录
        base, _, _, _, _, _ = build_project_paths()
        
        # 设置默认路径
        if sam2_config_path is None:
            sam2_config_path = str(base / "sam2_git" / "sam2" / "configs" / "sam2.1" / "sam2.1_hiera_l.yaml")
        if sam2_checkpoint_path is None:
            sam2_checkpoint_path = str(base / "sam2_git" / "checkpoints" / "sam2.1_hiera_large.pt")
        
        print(f"初始化SAM2，使用设备: {self.device}")
        print(f"配置文件: {sam2_config_path}")
        print(f"检查点: {sam2_checkpoint_path}")
        
        # 加载SAM2模型
        self.sam2_model = build_sam2(sam2_config_path, sam2_checkpoint_path, device=self.device)
        self.predictor = SAM2ImagePredictor(self.sam2_model)
        print("✅ SAM2模型加载完成")
    
    def extract_mask_from_box(
        self, 
        image: np.ndarray, 
        box: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        从单张图像和边界框提取mask
        
        Args:
            image: 图像数组，RGB格式，形状为(H, W, 3)
            box: 边界框，格式为[x1, y1, x2, y2]
        
        Returns:
            (mask, score) 元组，mask为二值mask，score为置信度分数
        """
        self.predictor.set_image(image)
        input_box = np.array([box])
        
        masks, scores, _ = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box,
            multimask_output=False
        )
        
        return masks[0], scores[0]
    
    def extract_masks_from_video(
        self,
        frame_paths: List[Path],
        boxes: List[np.ndarray],
        target_name: str,
        save_visualization: bool = True,
        use_cache: bool = True
    ) -> np.ndarray:
        """
        从视频序列中提取mask
        
        Args:
            frame_paths: 帧图像路径列表
            boxes: 每帧对应的边界框列表，如果为None则使用第一帧的框
            target_name: 目标名称，用于保存结果
            save_visualization: 是否保存可视化结果
            use_cache: 是否使用缓存
        
        Returns:
            mask列表，每个mask为二值图像数组
        """
        # 获取路径
        base, _, cache_dir, _, result_dir, _ = build_project_paths()
        
        # 缓存路径
        cache_path = cache_dir / "mask" / f"{target_name}.pt"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 检查缓存
        if use_cache and cache_path.exists():
            print(f"📦 从缓存加载mask: {cache_path}")
            masks = torch.load(str(cache_path))
            if isinstance(masks, torch.Tensor):
                masks = masks.numpy()
            print(f"✅ 加载了 {len(masks)} 个mask")
            
            # 如果启用可视化，仍然保存可视化结果
            if save_visualization:
                self._save_visualizations(masks, frame_paths, target_name, result_dir)
            
            return masks
        
        # 读取第一帧获取参考框（如果boxes为空，使用第一帧的框）
        if len(boxes) == 0:
            raise ValueError("必须提供至少一个边界框")
        
        # 如果只有一个框，所有帧使用同一个框；否则每帧使用对应的框
        use_single_box = len(boxes) == 1
        reference_box = boxes[0] if use_single_box else None
        
        masks = []
        n_frames = len(frame_paths)
        
        print(f"开始提取 {n_frames} 帧的mask...")
        
        for i, frame_path in enumerate(tqdm(frame_paths, desc="提取mask", unit="帧")):
            # 读取图像
            image = cv2.imread(str(frame_path))
            if image is None:
                raise ValueError(f"无法读取图像: {frame_path}")
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 选择当前帧的框
            if use_single_box:
                current_box = reference_box
            elif len(boxes) > i:
                current_box = boxes[i]
            else:
                # 如果框的数量少于帧数，使用最后一个框
                current_box = boxes[-1]
            
            if current_box is None:
                raise ValueError(f"第 {i} 帧没有对应的边界框")
            
            # 提取mask
            mask, score = self.extract_mask_from_box(image_rgb, current_box)
            # 确保mask是二值的（0或1）
            mask_binary = (mask > 0.5).astype(np.float32)
            masks.append(mask_binary)
        
        # 转换为numpy数组并保存缓存
        masks_array = np.array(masks)
        print(f"💾 保存mask缓存到: {cache_path}")
        torch.save(torch.from_numpy(masks_array), str(cache_path))
        
        # 保存可视化结果
        if save_visualization:
            self._save_visualizations(masks_array, frame_paths, target_name, result_dir)
        
        return masks_array
    
    def _save_visualizations(
        self,
        masks: np.ndarray,
        frame_paths: List[Path],
        target_name: str,
        result_dir: Path
    ):
        """
        保存mask的可视化结果（二值图像）
        
        Args:
            masks: mask数组，形状为(N, H, W)，值为0或1
            frame_paths: 帧路径列表
            target_name: 目标名称
            result_dir: 结果目录
        """
        output_dir = result_dir / "mask" / target_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🎨 保存可视化结果到: {output_dir}")
        
        # 确保masks是numpy数组
        if not isinstance(masks, np.ndarray):
            masks = np.array(masks)
        
        for i, (mask, frame_path) in enumerate(tqdm(
            zip(masks, frame_paths), 
            total=len(masks), 
            desc="保存可视化", 
            unit="帧"
        )):
            # 确保mask是二值的，然后转换为0-255
            if mask.max() <= 1.0:
                mask_binary = (mask * 255).astype(np.uint8)
            else:
                mask_binary = (mask > 0.5).astype(np.uint8) * 255
            
            # 使用原始帧文件名（不带扩展名）+ _mask.png
            frame_name = frame_path.stem
            output_path = output_dir / f"{frame_name}_mask.png"
            
            cv2.imwrite(str(output_path), mask_binary)


def extract_masks_from_text_prompt(
    text_prompt: str,
    target_name: Optional[str] = None,
    sam2_config_path: Optional[str] = None,
    sam2_checkpoint_path: Optional[str] = None,
    save_visualization: bool = True,
    use_cache: bool = True
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    从文本提示提取视频序列的mask（便捷函数）
    
    Args:
        text_prompt: 文本提示，例如"bear"
        target_name: 目标名称，如果为None则从配置中获取
        sam2_config_path: SAM2配置文件路径
        sam2_checkpoint_path: SAM2检查点路径
        save_visualization: 是否保存可视化结果
        use_cache: 是否使用缓存
    
    Returns:
        (masks, boxes) 元组，masks为mask数组（形状为(N, H, W)），boxes为边界框列表
    """
    # 获取路径
    base, _, _, frame_dir, _, _ = build_project_paths()
    
    if target_name is None:
        target_name = get_target_name(frame_dir)
    
    # 获取帧路径
    frame_paths = sorted([
        p for p in frame_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".png")
    ])
    
    if len(frame_paths) == 0:
        raise ValueError(f"在 {frame_dir} 中未找到图像文件")
    
    print(f"找到 {len(frame_paths)} 帧图像")
    
    # 初始化提取器
    box_extractor = FlorenceBoxExtractor()
    mask_extractor = MaskExtractor(
        sam2_config_path=sam2_config_path,
        sam2_checkpoint_path=sam2_checkpoint_path
    )
    
    # 从第一帧提取边界框
    print(f"从第一帧提取 '{text_prompt}' 的边界框...")
    first_image = Image.open(frame_paths[0]).convert("RGB")
    boxes = box_extractor.extract_boxes(first_image, text_prompt)
    
    if len(boxes) == 0:
        raise ValueError(f"未找到 '{text_prompt}' 的边界框")
    
    print(f"✅ 找到 {len(boxes)} 个边界框，使用第一个框")
    # 使用第一个框
    reference_box = boxes[0]
    
    # 提取所有帧的mask
    masks = mask_extractor.extract_masks_from_video(
        frame_paths=frame_paths,
        boxes=[reference_box] * len(frame_paths),  # 所有帧使用同一个框
        target_name=target_name,
        save_visualization=save_visualization,
        use_cache=use_cache
    )
    
    return masks, boxes


if __name__ == "__main__":
    """
    主函数：从配置中读取图片序列位置，提取mask并生成可视化图像
    """
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="基于Florence-2和SAM2的mask提取工具")
    parser.add_argument(
        "--text_prompt",
        type=str,
        default="bear",
        help="文本提示，例如 'bear', 'person' 等（默认: bear）"
    )
    parser.add_argument(
        "--target_name",
        type=str,
        default=None,
        help="目标名称，如果为None则从配置中自动获取（默认: None）"
    )
    parser.add_argument(
        "--no_cache",
        action="store_true",
        help="不使用缓存，强制重新计算mask"
    )
    parser.add_argument(
        "--no_visualization",
        action="store_true",
        help="不保存可视化图像"
    )
    parser.add_argument(
        "--sam2_config",
        type=str,
        default=None,
        help="SAM2配置文件路径（默认: 使用项目默认路径）"
    )
    parser.add_argument(
        "--sam2_checkpoint",
        type=str,
        default=None,
        help="SAM2检查点路径（默认: 使用项目默认路径）"
    )
    
    args = parser.parse_args()
    
    # 获取项目路径
    base, config_dir, cache_dir, frame_dir, result_dir, _ = build_project_paths()
    target_name = args.target_name if args.target_name else get_target_name(frame_dir)
    
    print("=" * 60)
    print("Mask提取工具 - 基于Florence-2和SAM2")
    print("=" * 60)
    print(f"项目根目录: {base}")
    print(f"配置目录: {config_dir}")
    print(f"缓存目录: {cache_dir}")
    print(f"帧图像目录: {frame_dir}")
    print(f"结果目录: {result_dir}")
    print(f"目标名称: {target_name}")
    print(f"文本提示: {args.text_prompt}")
    print("=" * 60)
    
    # 检查帧目录是否存在
    if not frame_dir.exists():
        raise FileNotFoundError(f"帧图像目录不存在: {frame_dir}")
    
    # 检查是否有图像文件
    frame_paths = sorted([
        p for p in frame_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".png")
    ])
    
    if len(frame_paths) == 0:
        raise ValueError(f"在 {frame_dir} 中未找到图像文件（.jpg 或 .png）")
    
    print(f"\n找到 {len(frame_paths)} 帧图像")
    print(f"第一帧: {frame_paths[0].name}")
    print(f"最后一帧: {frame_paths[-1].name}")
    
    try:
        # 提取mask
        print(f"\n开始提取mask...")
        masks, boxes = extract_masks_from_text_prompt(
            text_prompt=args.text_prompt,
            target_name=target_name,
            sam2_config_path=args.sam2_config,
            sam2_checkpoint_path=args.sam2_checkpoint,
            save_visualization=not args.no_visualization,
            use_cache=not args.no_cache
        )
        
        print("\n" + "=" * 60)
        print("✅ Mask提取完成！")
        print("=" * 60)
        print(f"提取的mask数量: {len(masks)}")
        print(f"Mask形状: {masks.shape}")
        print(f"找到的边界框数量: {len(boxes)}")
        if len(boxes) > 0:
            print(f"第一个边界框: {boxes[0]}")
        
        # 显示保存路径
        cache_path = cache_dir / "mask" / f"{target_name}.pt"
        print(f"\n缓存文件: {cache_path}")
        if cache_path.exists():
            print(f"  ✅ 已保存")
        else:
            print(f"  ⚠️  未找到（可能未启用缓存）")
        
        if not args.no_visualization:
            vis_dir = result_dir / "mask" / target_name
            print(f"\n可视化结果目录: {vis_dir}")
            if vis_dir.exists():
                vis_files = list(vis_dir.glob("*_mask.png"))
                print(f"  ✅ 已保存 {len(vis_files)} 个可视化图像")
            else:
                print(f"  ⚠️  目录不存在")
        
        print("\n" + "=" * 60)
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
