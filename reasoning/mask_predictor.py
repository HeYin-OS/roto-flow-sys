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
from scipy.ndimage import binary_fill_holes, distance_transform_edt
from transformers import AutoProcessor, AutoModelForCausalLM
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

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
        # 尝试使用base_plus模型（在性能和准确性之间更好的平衡）
        # 如果不存在，回退到large模型
        if sam2_config_path is None:
            base_plus_config = str(base / "sam2_git" / "sam2" / "configs" / "sam2.1" / "sam2.1_hiera_b+.yaml")
            base_large_config = str(base / "sam2_git" / "sam2" / "configs" / "sam2.1" / "sam2.1_hiera_l.yaml")
            if Path(base_plus_config).exists():
                sam2_config_path = base_plus_config
                print("📌 使用SAM2.1 Base Plus配置（更好的性能/准确性平衡）")
            else:
                sam2_config_path = base_large_config
                print("📌 使用SAM2.1 Large配置")
        
        if sam2_checkpoint_path is None:
            base_plus_checkpoint = str(base / "sam2_git" / "checkpoints" / "sam2.1_hiera_base_plus.pt")
            base_large_checkpoint = str(base / "sam2_git" / "checkpoints" / "sam2.1_hiera_large.pt")
            if Path(base_plus_checkpoint).exists():
                sam2_checkpoint_path = base_plus_checkpoint
            else:
                sam2_checkpoint_path = base_large_checkpoint
        
        print(f"初始化SAM2，使用设备: {self.device}")
        print(f"配置文件: {sam2_config_path}")
        print(f"检查点: {sam2_checkpoint_path}")
        
        # 验证配置文件和检查点版本是否匹配
        config_version = "2.1" if "sam2.1" in sam2_config_path else "2.0"
        checkpoint_version = "2.1" if "sam2.1" in sam2_checkpoint_path or "2.1" in str(sam2_checkpoint_path) else "2.0"
        
        if config_version != checkpoint_version:
            print(f"⚠️  警告: 配置文件版本({config_version})与检查点版本({checkpoint_version})不匹配！")
            print(f"   这可能导致模型加载错误或性能下降。")
            print(f"   建议使用匹配的版本：sam2.1配置应使用sam2.1检查点")
        else:
            print(f"✅ 版本匹配: 配置文件({config_version})与检查点({checkpoint_version})版本一致")
        
        # 加载SAM2模型
        # build_sam2函数会根据配置文件自动加载对应的模型架构
        # SAM2.1和SAM2使用相同的build_sam2函数，区别在于配置文件和检查点
        self.sam2_model = build_sam2(sam2_config_path, sam2_checkpoint_path, device=self.device)
        self.predictor = SAM2ImagePredictor(self.sam2_model)
        
        # 显示实际加载的模型信息
        model_name = self.sam2_model.__class__.__name__
        print(f"✅ SAM2模型加载完成")
        print(f"   模型类型: {model_name}")
        print(f"   使用版本: {config_version} (根据配置文件)")
        
        # 保存设备信息供后续使用
        self.device = self.sam2_model.device if hasattr(self.sam2_model, 'device') else device
    
    def extract_mask_from_box(
        self, 
        image: np.ndarray, 
        box: np.ndarray,
        reset_predictor: bool = True,
        use_multimask: bool = True,
        use_point_prompt: bool = True,
        high_precision_mode: bool = True,  # 默认启用高精度模式
        ensemble_passes: int = 2  # 默认2次集成推理
    ) -> Tuple[np.ndarray, float]:
        """
        从单张图像和边界框提取mask（每帧独立处理）
        
        Args:
            image: 图像数组，RGB格式，形状为(H, W, 3)
            box: 边界框，格式为[x1, y1, x2, y2]
            reset_predictor: 是否重置predictor（确保每帧独立），默认True
            use_multimask: 是否使用多mask输出，选择最好的，默认True
            use_point_prompt: 是否使用点提示（边界框中心点），默认True
            high_precision_mode: 高精度模式，使用更多计算提升精度，默认False
            ensemble_passes: 集成推理次数，多次推理并融合结果，默认1
        
        Returns:
            (mask, score) 元组，mask为二值mask，score为置信度分数
        """
        # 高精度模式：使用更高分辨率的图像
        original_image = image
        scale_factor = 1.0
        
        # 确保box是numpy数组（修复类型问题）
        if not isinstance(box, np.ndarray):
            box = np.array(box, dtype=np.float32)
        else:
            box = box.astype(np.float32)
        
        if high_precision_mode:
            # 将图像放大1.5倍（提升细节）
            h, w = image.shape[:2]
            new_h, new_w = int(h * 1.5), int(w * 1.5)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            # 相应调整边界框（numpy数组可以直接乘以标量）
            box = box * 1.5
            scale_factor = 1.5
        
        # 每帧独立：重置predictor状态，确保不依赖前帧
        if reset_predictor:
            self.predictor.set_image(image)
        else:
            # 如果图像尺寸或内容变化，仍需要重置
            self.predictor.set_image(image)
        
        input_box = np.array([box])
        
        # 改进：同时使用边界框和多个点提示（点提示通常更准确）
        # 高精度模式：使用更多的点提示
        point_coords = None
        point_labels = None
        if use_point_prompt:
            center_x = (box[0] + box[2]) / 2.0
            center_y = (box[1] + box[3]) / 2.0
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            
            if high_precision_mode:
                # 高精度模式：使用9个点（中心 + 8个方向）
                margin = 0.15
                x1_inner = x1 + w * margin
                y1_inner = y1 + h * margin
                x2_inner = x2 - w * margin
                y2_inner = y2 - h * margin
                x_mid = (x1 + x2) / 2.0
                y_mid = (y1 + y2) / 2.0
                
                point_coords = np.array([
                    [center_x, center_y],      # 中心
                    [x1_inner, y1_inner],      # 左上
                    [x_mid, y1_inner],         # 上中
                    [x2_inner, y1_inner],      # 右上
                    [x1_inner, y_mid],         # 左中
                    [x2_inner, y_mid],         # 右中
                    [x1_inner, y2_inner],      # 左下
                    [x_mid, y2_inner],         # 下中
                    [x2_inner, y2_inner],      # 右下
                ], dtype=np.float32)
                point_labels = np.array([1] * 9, dtype=np.int32)
            else:
                # 标准模式：5个点
                margin = 0.1
                x1_inner = x1 + w * margin
                y1_inner = y1 + h * margin
                x2_inner = x2 - w * margin
                y2_inner = y2 - h * margin
                
                point_coords = np.array([
                    [center_x, center_y],  # 中心点
                    [x1_inner, y1_inner],  # 左上
                    [x2_inner, y1_inner],  # 右上
                    [x1_inner, y2_inner],  # 左下
                    [x2_inner, y2_inner],  # 右下
                ], dtype=np.float32)
                point_labels = np.array([1, 1, 1, 1, 1], dtype=np.int32)
        
        # 集成推理：多次推理并融合结果（高精度模式）
        all_masks = []
        all_scores = []
        
        for pass_idx in range(ensemble_passes):
            if pass_idx > 0 and len(all_masks) > 0:
                # 后续推理：使用前一次的结果作为mask提示（迭代优化）
                prev_mask = all_masks[-1]
                # SAM2的mask_input需要是numpy数组，形状为(H, W)，值在[0, 1]范围
                # 注意：SAM2内部会将mask下采样到256x256，所以这里直接使用原始尺寸即可
                mask_input = prev_mask.astype(np.float32)
                # 确保值在[0, 1]范围
                mask_input = np.clip(mask_input, 0.0, 1.0)
            else:
                mask_input = None
            
            if use_multimask:
                # 使用多mask输出
                masks, scores, _ = self.predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=input_box,
                    multimask_output=True,  # 输出3个mask
                    mask_input=mask_input  # 使用前一次的mask作为提示（numpy数组）
                )
                
                # 选择策略：综合考虑置信度和完整性
                best_mask = None
                best_score = 0.0
                best_combined_score = -1.0
                
                for m, s in zip(masks, scores):
                    area = m.sum()
                    if s > 0.2:  # 至少置信度>0.2
                        max_possible_area = image.shape[0] * image.shape[1] * 0.5
                        normalized_area = min(1.0, area / max_possible_area) if max_possible_area > 0 else 0
                        combined_score = s * 0.7 + normalized_area * 0.3
                        
                        if combined_score > best_combined_score:
                            best_combined_score = combined_score
                            best_mask = m
                            best_score = s
                
                if best_mask is None:
                    # 如果所有mask置信度都很低，选择面积最大的
                    areas = [m.sum() for m in masks]
                    best_idx = np.argmax(areas)
                    best_mask = masks[best_idx]
                    best_score = scores[best_idx]
                
                all_masks.append(best_mask)
                all_scores.append(best_score)
            else:
                masks, scores, _ = self.predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=input_box,
                    multimask_output=False,
                    mask_input=mask_input
                )
                all_masks.append(masks[0])
                all_scores.append(scores[0])
        
        # 融合多个推理结果
        if ensemble_passes > 1:
            # 方法：取所有mask的平均值（软融合），然后二值化
            combined_mask = np.mean(all_masks, axis=0)
            combined_score = np.mean(all_scores)
            # 对融合后的mask进行二值化（阈值0.5）
            final_mask = (combined_mask > 0.5).astype(np.float32)
        else:
            final_mask = all_masks[0]
            combined_score = all_scores[0]
        
        # 如果使用了高精度模式，需要将mask缩放回原始尺寸
        if high_precision_mode and scale_factor > 1.0:
            original_h, original_w = original_image.shape[:2]
            final_mask = cv2.resize(
                final_mask.astype(np.float32),
                (original_w, original_h),
                interpolation=cv2.INTER_LINEAR
            )
            # 重新二值化（因为resize可能产生中间值）
            final_mask = (final_mask > 0.5).astype(np.float32)
        
        return final_mask, combined_score
    
    def extract_masks_from_video(
        self,
        frame_paths: List[Path],
        boxes: List[np.ndarray],
        target_name: str,
        save_visualization: bool = True,
        use_cache: bool = True,
        debug_frames: Optional[List[int]] = None,
        update_box_interval: Optional[int] = None,
        box_extractor: Optional['FlorenceBoxExtractor'] = None,
        text_prompt: Optional[str] = None,
        high_precision_mode: bool = True,  # 默认启用高精度模式
        ensemble_passes: int = 2  # 默认2次集成推理
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
        
        # Debug模式：如果指定了debug_frames，保存这些帧的可视化
        if debug_frames is None:
            debug_frames = []
        debug_dir = None
        if len(debug_frames) > 0:
            debug_dir = result_dir / "mask" / target_name / "debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            print(f"🐛 Debug模式: 将保存帧 {debug_frames} 的调试可视化到 {debug_dir}")
        
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
            
            # 即使使用缓存，也生成debug图像（需要重新读取图像和边界框）
            if debug_dir is not None and len(debug_frames) > 0:
                print(f"🐛 从缓存加载，但仍生成debug图像...")
                # 需要重新提取边界框信息（从调用者传入的boxes）
                if len(boxes) > 0:
                    # 确保boxes是numpy数组
                    boxes = [np.array(b, dtype=np.float32) if not isinstance(b, np.ndarray) else b.astype(np.float32) for b in boxes]
                    reference_box = boxes[0] if len(boxes) == 1 else None
                    
                    for i in debug_frames:
                        if i < len(frame_paths):
                            try:
                                # 读取图像
                                image = cv2.imread(str(frame_paths[i]))
                                if image is None:
                                    continue
                                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                                
                                # 获取当前帧的边界框（确保是numpy数组）
                                if reference_box is not None:
                                    current_box = reference_box.copy()
                                elif len(boxes) > i:
                                    current_box = boxes[i].copy()
                                else:
                                    current_box = boxes[-1].copy() if len(boxes) > 0 else None
                                
                                if current_box is None:
                                    continue
                                
                                # 重新提取mask用于debug（使用高精度模式）
                                mask_raw, _ = self.extract_mask_from_box(
                                    image_rgb, current_box, 
                                    use_multimask=True,
                                    use_point_prompt=True,
                                    high_precision_mode=high_precision_mode,  # 默认True
                                    ensemble_passes=ensemble_passes  # 默认2
                                )
                                
                                # 计算阈值
                                mask_mean = mask_raw.mean()
                                if mask_mean < 0.6:
                                    threshold = max(0.35, mask_mean - 0.15)
                                else:
                                    threshold = 0.5
                                
                                # 使用缓存中的mask作为二值化结果
                                mask_binary = masks[i] if i < len(masks) else (mask_raw > threshold).astype(np.uint8)
                                
                                # 保存debug可视化
                                self._save_debug_visualization(
                                    image_rgb, current_box, mask_raw, mask_binary, threshold, i, debug_dir
                                )
                            except Exception as e:
                                print(f"  ⚠️  生成帧 {i} 的debug图像时出错: {e}")
            
            return masks
        
        # 读取第一帧获取参考框（如果boxes为空，使用第一帧的框）
        if len(boxes) == 0:
            raise ValueError("必须提供至少一个边界框")
        
        # 确保boxes是numpy数组列表
        boxes = [np.array(b, dtype=np.float32) if not isinstance(b, np.ndarray) else b.astype(np.float32) for b in boxes]
        
        # 每帧独立处理：不使用共享的reference_box
        # 如果只有一个框，作为初始参考；否则每帧使用对应的框
        use_single_box = len(boxes) == 1
        initial_box = boxes[0].copy() if use_single_box else None
        
        # 每帧独立提取边界框设置
        independent_box_extraction = (
            box_extractor is not None and 
            text_prompt is not None
        )
        if independent_box_extraction:
            print(f"🔄 启用每帧独立边界框提取: 每帧从当前帧提取边界框")
        
        masks = []
        n_frames = len(frame_paths)
        
        # Debug模式：如果指定了debug_frames，保存这些帧的可视化
        if debug_frames is None:
            debug_frames = []
        debug_dir = None
        if len(debug_frames) > 0:
            base, _, _, _, result_dir, _ = build_project_paths()
            debug_dir = result_dir / "mask" / target_name / "debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            print(f"🐛 Debug模式: 将保存帧 {debug_frames} 的调试可视化到 {debug_dir}")
        
        print(f"开始提取 {n_frames} 帧的mask（每帧独立处理）...")
        
        for i, frame_path in enumerate(tqdm(frame_paths, desc="提取mask", unit="帧")):
            # 读取图像
            image = cv2.imread(str(frame_path))
            if image is None:
                raise ValueError(f"无法读取图像: {frame_path}")
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_h, img_w = image_rgb.shape[:2]
            
            # 每帧独立：从当前帧提取边界框
            if independent_box_extraction:
                try:
                    current_image_pil = Image.open(frame_path).convert("RGB")
                    current_boxes = box_extractor.extract_boxes(current_image_pil, text_prompt)
                    if len(current_boxes) > 0:
                        # 确保boxes是numpy数组
                        current_boxes = [np.array(b, dtype=np.float32) if not isinstance(b, np.ndarray) else b.astype(np.float32) for b in current_boxes]
                        
                        # 使用当前帧提取的边界框
                        # 改进：如果提取到多个框，选择面积最大的（更可能包含完整目标）
                        if len(current_boxes) > 1:
                            areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in current_boxes]
                            best_box_idx = np.argmax(areas)
                            current_box = current_boxes[best_box_idx].copy()
                            if i < 5 or i % 20 == 0:
                                print(f"  📦 帧 {i}: 从 {len(current_boxes)} 个框中选择面积最大的")
                        else:
                            current_box = current_boxes[0].copy()
                        
                        # 使用当前帧的图像尺寸扩大边界框
                        box_width = current_box[2] - current_box[0]
                        box_height = current_box[3] - current_box[1]
                        # 增加padding到30%，确保包含完整目标
                        padding_x = max(box_width * 0.30, 40)  # 增加到30%，至少40像素
                        padding_y = max(box_height * 0.30, 40)
                        current_box[0] = max(0, current_box[0] - padding_x)
                        current_box[1] = max(0, current_box[1] - padding_y)
                        current_box[2] = min(img_w, current_box[2] + padding_x)
                        current_box[3] = min(img_h, current_box[3] + padding_y)
                        if i < 5 or i % 20 == 0:  # 只打印前5帧和每20帧
                            print(f"  📦 帧 {i}: 从当前帧提取边界框 -> {current_box}")
                    else:
                        # 如果提取失败，使用初始框（仅作为fallback）
                        if initial_box is not None:
                            current_box = np.array(initial_box, dtype=np.float32) if not isinstance(initial_box, np.ndarray) else initial_box.astype(np.float32).copy()
                            # 根据当前帧尺寸调整
                            current_box[2] = min(img_w, current_box[2])
                            current_box[3] = min(img_h, current_box[3])
                            print(f"  ⚠️  帧 {i}: 边界框提取失败，使用初始框（已调整到当前帧尺寸）")
                        else:
                            raise ValueError(f"帧 {i}: 无法提取边界框且无初始框")
                except Exception as e:
                    # 如果提取失败，使用初始框（仅作为fallback）
                    if initial_box is not None:
                        current_box = np.array(initial_box, dtype=np.float32) if not isinstance(initial_box, np.ndarray) else initial_box.astype(np.float32).copy()
                        current_box[2] = min(img_w, current_box[2])
                        current_box[3] = min(img_h, current_box[3])
                        print(f"  ⚠️  帧 {i}: 边界框提取出错，使用初始框（已调整）: {e}")
                    else:
                        raise ValueError(f"帧 {i}: 边界框提取失败: {e}")
            else:
                # 如果不启用独立提取，使用传入的boxes
                if use_single_box:
                    if initial_box is not None:
                        current_box = np.array(initial_box, dtype=np.float32) if not isinstance(initial_box, np.ndarray) else initial_box.astype(np.float32).copy()
                        # 根据当前帧尺寸调整
                        current_box[2] = min(img_w, current_box[2])
                        current_box[3] = min(img_h, current_box[3])
                    else:
                        current_box = None
                elif len(boxes) > i:
                    current_box = np.array(boxes[i], dtype=np.float32) if not isinstance(boxes[i], np.ndarray) else boxes[i].astype(np.float32).copy()
                    current_box[2] = min(img_w, current_box[2])
                    current_box[3] = min(img_h, current_box[3])
                else:
                    if len(boxes) > 0:
                        current_box = np.array(boxes[-1], dtype=np.float32) if not isinstance(boxes[-1], np.ndarray) else boxes[-1].astype(np.float32).copy()
                        current_box[2] = min(img_w, current_box[2])
                        current_box[3] = min(img_h, current_box[3])
                    else:
                        current_box = None
            
            if current_box is None:
                raise ValueError(f"第 {i} 帧没有对应的边界框")
            
            # 提取mask（使用高精度模式：多mask输出 + 9个点提示 + 集成推理）
            mask, score = self.extract_mask_from_box(
                image_rgb, current_box, 
                use_multimask=True,
                use_point_prompt=True,
                high_precision_mode=high_precision_mode,  # 高精度模式（默认True）
                ensemble_passes=ensemble_passes  # 集成推理次数（默认2）
            )
            
            # 激进策略：使用非常低的阈值，优先保留所有可能的区域
            # 自适应阈值：根据mask的置信度分布调整阈值
            mask_mean = mask.mean()
            mask_min = mask.min()
            mask_max = mask.max()
            mask_std = mask.std()
            
            # 更激进的阈值策略：大幅降低阈值
            if mask_mean < 0.6:
                # 置信度较低时，使用非常低的阈值
                threshold = max(0.2, mask_mean - 0.2)  # 最低0.2，比之前更低
            else:
                threshold = 0.4  # 即使置信度高，也使用较低的阈值（从0.5降到0.4）
            
            # 改进：使用局部自适应阈值（边缘区域使用更低阈值）
            # 创建边界框mask
            box_mask = np.zeros((img_h, img_w), dtype=np.uint8)
            x1, y1, x2, y2 = map(int, current_box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w, x2), min(img_h, y2)
            box_mask[y1:y2, x1:x2] = 1
            
            # 计算到边界框边缘的距离（用于边缘区域检测）
            dist_to_edge = distance_transform_edt(box_mask)
            
            # 激进策略：边缘区域使用极低的阈值
            # 在边界框边缘区域（距离边缘<50像素，扩大范围）使用更低的阈值
            edge_mask_region = (dist_to_edge < 50) & (box_mask > 0)  # 扩大到50像素
            if edge_mask_region.any():
                edge_mean_confidence = mask[edge_mask_region].mean()
                # 边缘区域使用极低的阈值
                if edge_mean_confidence < 0.3:
                    edge_threshold = max(0.15, edge_mean_confidence * 0.6)  # 非常激进
                else:
                    edge_threshold = max(0.2, threshold * 0.6)  # 边缘区域降低40%
            else:
                edge_threshold = threshold * 0.7
            
            # 使用更激进的策略：边缘区域几乎保留所有非零值
            mask_binary = np.where(
                (mask > threshold) | ((dist_to_edge < 50) & (mask > edge_threshold)),
                1, 0
            ).astype(np.uint8)
            
            # Debug输出：特别关注60帧后的变化
            if i >= 60 and i <= 80:  # 60-80帧范围
                mask_area_before = mask_binary.sum()
                # 检查边界框是否在图像范围内
                box_in_bounds = (
                    0 <= current_box[0] < img_w and
                    0 <= current_box[1] < img_h and
                    0 < current_box[2] <= img_w and
                    0 < current_box[3] <= img_h
                )
                # 计算边界框覆盖的图像比例
                box_area = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
                box_coverage = box_area / (img_w * img_h)
                
                # 计算边缘区域的mask统计
                edge_mask_region = (dist_to_edge < 50) & (box_mask > 0)
                edge_mask_values = mask[edge_mask_region] if edge_mask_region.any() else np.array([])
                edge_mean = edge_mask_values.mean() if len(edge_mask_values) > 0 else 0
                
                print(f"\n[帧 {i}] Debug信息 (激进策略):")
                print(f"  边界框: [{current_box[0]:.1f}, {current_box[1]:.1f}, {current_box[2]:.1f}, {current_box[3]:.1f}]")
                print(f"  边界框是否在图像内: {box_in_bounds}")
                print(f"  边界框覆盖图像比例: {box_coverage:.2%}")
                print(f"  SAM2置信度分数: {score:.4f}")
                print(f"  Mask置信度统计: mean={mask_mean:.4f}, min={mask_min:.4f}, max={mask_max:.4f}, std={mask_std:.4f}")
                print(f"  边缘区域(50px内)平均置信度: {edge_mean:.4f}")
                print(f"  使用阈值: {threshold:.4f} (边缘区域: {edge_threshold:.4f})")
                print(f"  二值化前mask面积: {mask_area_before} 像素")
            
            # 激进策略：多次膨胀+闭运算，优先填充所有空洞
            # 第一步：先进行膨胀，连接断开的区域
            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask_binary = cv2.dilate(mask_binary, kernel_dilate, iterations=1)
            
            # 第二步：多次闭运算，逐步填充空洞
            kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel_small)
            
            kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel_medium)
            
            kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel_large)
            
            # 第三步：使用更大的kernel进行闭运算（针对大空洞）
            kernel_very_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel_very_large)
            
            # 第四步：再次膨胀边缘区域，确保边缘完整
            edge_expand_mask = (dist_to_edge < 50) & (box_mask > 0)
            if edge_expand_mask.any():
                edge_mask_binary = mask_binary.copy()
                edge_mask_binary[~edge_expand_mask] = 0
                kernel_edge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                edge_mask_expanded = cv2.dilate(edge_mask_binary, kernel_edge, iterations=2)
                mask_binary = np.maximum(mask_binary, edge_mask_expanded)
            
            # 连通域分析：选择最大连通域，过滤小的碎片区域
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                mask_binary, connectivity=8
            )
            if num_labels > 1:
                # 找到最大连通域（排除背景，索引0）
                areas = stats[1:, cv2.CC_STAT_AREA]  # 跳过背景（索引0）
                largest_label = np.argmax(areas) + 1  # +1因为跳过了背景
                mask_binary = (labels == largest_label).astype(np.uint8)
            
            # 孔洞填充：填充mask内部的孔洞
            mask_binary = binary_fill_holes(mask_binary).astype(np.float32)
            
            # Debug输出：形态学操作后的统计
            if i >= 60 and i <= 80:
                mask_area_after = mask_binary.sum()
                # 计算空洞数量（连通域数量-1，减去背景）
                num_labels_final, _, stats_final, _ = cv2.connectedComponentsWithStats(
                    (mask_binary > 0.5).astype(np.uint8), connectivity=8
                )
                num_holes = num_labels_final - 1  # 减去背景
                
                # 计算mask在边界框内的分布
                box_mask = np.zeros((img_h, img_w), dtype=np.uint8)
                x1, y1, x2, y2 = map(int, current_box)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img_w, x2), min(img_h, y2)
                box_mask[y1:y2, x1:x2] = 1
                
                mask_in_box = (mask_binary > 0.5) & (box_mask > 0)
                mask_out_box = (mask_binary > 0.5) & (box_mask == 0)
                area_in_box = mask_in_box.sum()
                area_out_box = mask_out_box.sum()
                
                print(f"  形态学操作后mask面积: {mask_area_after} 像素 (变化: {mask_area_after - mask_area_before:+d})")
                print(f"  连通域数量: {num_labels_final} (空洞数: {num_holes})")
                print(f"  Mask在边界框内面积: {area_in_box} 像素")
                print(f"  Mask在边界框外面积: {area_out_box} 像素")
                if mask_area_after > 0:
                    print(f"  边界框外占比: {area_out_box / mask_area_after:.2%}")
                
                # 检查是否有明显空洞（面积突然下降）
                # 检查mask质量（不依赖前帧，每帧独立评估）
                if mask_area_after < 1000:  # 如果mask面积太小，可能是提取失败
                    print(f"  ⚠️  Mask面积过小: {mask_area_after} 像素（可能提取失败）")
                
                print()
            
            # 保存debug可视化（如果启用了debug模式）
            if debug_dir is not None and i in debug_frames:
                self._save_debug_visualization(
                    image_rgb, current_box, mask, mask_binary, threshold, i, debug_dir
                )
            
            masks.append(mask_binary)
        
        # 转换为numpy数组并保存缓存
        masks_array = np.array(masks)
        print(f"\n💾 保存mask缓存到: {cache_path}")
        torch.save(torch.from_numpy(masks_array), str(cache_path))
        print(f"✅ 已保存 {len(masks_array)} 个mask到缓存")
        
        # 保存可视化结果
        if save_visualization:
            vis_dir = result_dir / "mask" / target_name
            print(f"🎨 保存可视化结果到: {vis_dir}")
            self._save_visualizations(masks_array, frame_paths, target_name, result_dir)
            print(f"✅ 已保存 {len(masks_array)} 个可视化图像")
        
        return masks_array
    
    def _save_debug_visualization(
        self,
        image: np.ndarray,
        box: np.ndarray,
        mask_raw: np.ndarray,
        mask_binary: np.ndarray,
        threshold: float,
        frame_idx: int,
        debug_dir: Path
    ):
        """
        保存debug可视化图像，显示边界框、原始mask、二值化mask的叠加
        
        Args:
            image: 原始图像，RGB格式
            box: 边界框 [x1, y1, x2, y2]
            mask_raw: 原始mask（浮点数，0-1）
            mask_binary: 二值化后的mask
            threshold: 使用的阈值
            frame_idx: 帧索引
            debug_dir: 保存目录
        """
        # 设置matplotlib支持中文（修复字体警告）
        try:
            # 尝试使用系统中文字体
            import platform
            if platform.system() == 'Windows':
                plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
            elif platform.system() == 'Darwin':  # macOS
                plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'STHeiti']
            else:  # Linux
                plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC']
            plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        except:
            # 如果设置失败，使用英文标签避免警告
            pass
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 16))
            
            # 1. 原始图像 + 边界框
            ax = axes[0, 0]
            ax.imshow(image)
            rect = Rectangle(
                (box[0], box[1]), box[2] - box[0], box[3] - box[1],
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)
            ax.set_title(f'Frame {frame_idx}: Original Image + Bounding Box\nBox: [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]')
            ax.axis('off')
            
            # 2. 原始mask（热力图）
            ax = axes[0, 1]
            im = ax.imshow(mask_raw, cmap='hot', vmin=0, vmax=1)
            rect = Rectangle(
                (box[0], box[1]), box[2] - box[0], box[3] - box[1],
                linewidth=2, edgecolor='cyan', facecolor='none'
            )
            ax.add_patch(rect)
            ax.set_title(f'Raw Mask (threshold={threshold:.3f})\nmean={mask_raw.mean():.3f}, min={mask_raw.min():.3f}, max={mask_raw.max():.3f}')
            plt.colorbar(im, ax=ax)
            ax.axis('off')
            
            # 3. 二值化mask
            ax = axes[1, 0]
            ax.imshow(mask_binary, cmap='gray', vmin=0, vmax=1)
            rect = Rectangle(
                (box[0], box[1]), box[2] - box[0], box[3] - box[1],
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)
            mask_area = mask_binary.sum()
            ax.set_title(f'Binary Mask\nArea: {mask_area} pixels')
            ax.axis('off')
            
            # 4. 叠加显示：图像 + mask轮廓 + 边界框
            ax = axes[1, 1]
            ax.imshow(image)
            # 显示mask轮廓
            mask_contour = (mask_binary > 0.5).astype(np.uint8)
            contours, _ = cv2.findContours(mask_contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            ax.contour(mask_contour, levels=[0.5], colors='lime', linewidths=2)
            rect = Rectangle(
                (box[0], box[1]), box[2] - box[0], box[3] - box[1],
                linewidth=2, edgecolor='red', facecolor='none', linestyle='--'
            )
            ax.add_patch(rect)
            ax.set_title(f'Overlay: Image + Mask Contour + Bounding Box')
            ax.axis('off')
            
            plt.tight_layout()
            debug_path = debug_dir / f"frame_{frame_idx:04d}_debug.png"
            plt.savefig(debug_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  💾 已保存debug可视化: {debug_path}")
        except Exception as e:
            print(f"  ⚠️  保存debug可视化时出错: {e}")
            import traceback
            traceback.print_exc()
            try:
                plt.close()
            except:
                pass
    
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
    use_cache: bool = True,
    high_precision_mode: bool = True,  # 默认启用高精度模式
    ensemble_passes: int = 2  # 默认2次集成推理
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
        high_precision_mode: 高精度模式，使用更多计算提升精度（图像1.5倍分辨率，9个点提示），默认False
        ensemble_passes: 集成推理次数，多次推理并融合结果，默认1（建议2-3次）
    
    Returns:
        (masks, boxes) 元组，masks为mask数组（形状为(N, H, W)），boxes为边界框列表
    """
    # 获取路径
    base, _, cache_dir, frame_dir, result_dir, _ = build_project_paths()
    
    if target_name is None:
        target_name = get_target_name(frame_dir)
    
    # 显示训练对象
    print("=" * 60)
    print(f"🎯 训练对象: {text_prompt}")
    print(f"📁 目标名称: {target_name}")
    print("=" * 60)
    
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
    # 确保boxes是numpy数组
    boxes = [np.array(b, dtype=np.float32) if not isinstance(b, np.ndarray) else b.astype(np.float32) for b in boxes]
    
    # 使用第一个框，并扩大边界框以确保包含完整目标
    reference_box = boxes[0].copy()
    
    # 获取第一帧图像尺寸
    img_width, img_height = first_image.size
    
    # 扩大边界框30%，确保后续帧即使目标移动也能包含完整目标
    # 针对60帧后的空洞问题，增加padding以覆盖目标可能移动的范围
    box_width = reference_box[2] - reference_box[0]
    box_height = reference_box[3] - reference_box[1]
    padding_x = box_width * 0.30  # 增加到30% padding
    padding_y = box_height * 0.30
    # 同时确保至少有一定的像素padding
    padding_x = max(padding_x, 40)  # 至少40像素
    padding_y = max(padding_y, 40)
    reference_box[0] = max(0, reference_box[0] - padding_x)
    reference_box[1] = max(0, reference_box[1] - padding_y)
    reference_box[2] = min(img_width, reference_box[2] + padding_x)
    reference_box[3] = min(img_height, reference_box[3] + padding_y)
    print(f"📦 扩大边界框: 原始={boxes[0]}, 扩大后={reference_box} (padding: {padding_x:.1f}, {padding_y:.1f})")
    
    # 提取所有帧的mask（每帧独立处理）
    # Debug模式：自动检测60-80帧
    debug_frames = list(range(60, 81))  # 60-80帧用于debug
    
    # 每帧独立：从每帧提取边界框（不使用共享的reference_box）
    # 高精度模式（已内置，默认启用）
    print(f"\n🚀 高精度模式（已内置）:")
    print(f"   ✅ 图像分辨率提升1.5倍")
    print(f"   ✅ 使用9个点提示（标准模式5个）")
    print(f"   ✅ 集成推理: {ensemble_passes}次推理并融合结果")
    print()
    
    masks = mask_extractor.extract_masks_from_video(
        frame_paths=frame_paths,
        boxes=[reference_box] * len(frame_paths),  # 仅作为fallback，实际每帧会重新提取
        target_name=target_name,
        save_visualization=save_visualization,
        use_cache=use_cache,
        debug_frames=debug_frames,
        update_box_interval=None,  # 不使用间隔更新，改为每帧独立提取
        box_extractor=box_extractor,  # 传入边界框提取器
        text_prompt=text_prompt,  # 传入文本提示
        high_precision_mode=high_precision_mode,  # 高精度模式
        ensemble_passes=ensemble_passes  # 集成推理次数
    )
    
    # 显示最终保存的目录
    print("\n" + "=" * 60)
    print("📂 最终保存目录:")
    cache_path = cache_dir / "mask" / f"{target_name}.pt"
    print(f"  💾 缓存文件: {cache_path}")
    if save_visualization:
        vis_dir = result_dir / "mask" / target_name
        print(f"  🎨 可视化结果: {vis_dir}")
    print("=" * 60)
    
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
    parser.add_argument(
        "--high_precision",
        action="store_true",
        default=True,  # 默认启用
        help="启用高精度模式：图像分辨率提升1.5倍，使用9个点提示（已内置，默认启用）"
    )
    parser.add_argument(
        "--ensemble_passes",
        type=int,
        default=2,  # 默认2次
        help="集成推理次数，多次推理并融合结果（默认: 2，建议2-3次以获得更高精度）"
    )
    parser.add_argument(
        "--disable_high_precision",
        action="store_true",
        help="禁用高精度模式（如果不需要最高精度，可以禁用以提升速度）"
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
        # 高精度模式已内置（默认启用），可通过命令行参数禁用
        high_precision = not args.disable_high_precision if hasattr(args, 'disable_high_precision') else True
        ensemble_count = args.ensemble_passes if hasattr(args, 'ensemble_passes') else 2
        
        masks, boxes = extract_masks_from_text_prompt(
            text_prompt=args.text_prompt,
            target_name=target_name,
            sam2_config_path=args.sam2_config,
            sam2_checkpoint_path=args.sam2_checkpoint,
            save_visualization=not args.no_visualization,
            use_cache=not args.no_cache,
            high_precision_mode=high_precision,
            ensemble_passes=ensemble_count
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
