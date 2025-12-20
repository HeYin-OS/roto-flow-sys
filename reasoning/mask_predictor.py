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
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import tempfile
import shutil

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
        
        # 保存配置路径，用于后续创建video_predictor
        self.sam2_config_path = sam2_config_path
        self.sam2_checkpoint_path = sam2_checkpoint_path
        
        # 加载SAM2模型（用于单帧预测）
        # build_sam2函数会根据配置文件自动加载对应的模型架构
        # SAM2.1和SAM2使用相同的build_sam2函数，区别在于配置文件和检查点
        self.sam2_model = build_sam2(sam2_config_path, sam2_checkpoint_path, device=self.device)
        self.predictor = SAM2ImagePredictor(self.sam2_model)
        
        # 视频预测器（延迟加载，只在需要时创建）
        self.video_predictor = None
        
        # 显示实际加载的模型信息
        model_name = self.sam2_model.__class__.__name__
        print(f"✅ SAM2模型加载完成")
        print(f"   模型类型: {model_name}")
        print(f"   使用版本: {config_version} (根据配置文件)")
        
        # 保存设备信息供后续使用
        self.device = self.sam2_model.device if hasattr(self.sam2_model, 'device') else device
    
    def _get_video_predictor(self):
        """延迟加载视频预测器"""
        if self.video_predictor is None:
            print("🎬 初始化SAM2视频预测器...")
            self.video_predictor = build_sam2_video_predictor(
                self.sam2_config_path,
                self.sam2_checkpoint_path,
                device=self.device,
                apply_postprocessing=True
            )
            print("✅ SAM2视频预测器初始化完成")
        return self.video_predictor
    
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
            high_precision_mode: 高精度模式，影响集成推理次数，默认True
            ensemble_passes: 集成推理次数，多次推理并融合结果，默认2
        
        Returns:
            (mask, score) 元组，mask为二值mask，score为置信度分数
        """
        # 确保box是numpy数组（修复类型问题）
        if not isinstance(box, np.ndarray):
            box = np.array(box, dtype=np.float32)
        else:
            box = box.astype(np.float32)
        
        # 每帧独立：重置predictor状态，确保不依赖前帧
        if reset_predictor:
            self.predictor.set_image(image)
        else:
            # 如果图像尺寸或内容变化，仍需要重置
            self.predictor.set_image(image)
        
        input_box = np.array([box])
        
        # 使用边界框 + 单个中心点提示（避免角点落在背景中产生误导）
        point_coords = None
        point_labels = None
        if use_point_prompt:
            # 只使用边界框的中心点，避免角点落在背景中
            center_x = (box[0] + box[2]) / 2.0
            center_y = (box[1] + box[3]) / 2.0
            point_coords = np.array([[center_x, center_y]], dtype=np.float32)
            point_labels = np.array([1], dtype=np.int32)
        
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
            # 对融合后的mask进行二值化（降低阈值到-1.0以包含更多"可能是物体"的像素）
            # SAM2输出的是logits，0.0代表50%概率，-1.0代表约27%概率，提高召回率
            final_mask = (combined_mask > -1.0).astype(np.float32)
        else:
            # 单次推理也需要应用阈值（SAM2输出的是logits）
            final_mask = (all_masks[0] > -1.0).astype(np.float32)
            combined_score = all_scores[0]
        
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
                                
                                # 使用统一的低阈值（-1.0）以提高召回率，避免孔洞
                                threshold = -1.0
                                
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
        
        # 使用第一帧的边界框作为初始prompt
        initial_box = boxes[0].copy()
        
        # 读取第一帧图像以获取尺寸
        first_image = cv2.imread(str(frame_paths[0]))
        if first_image is None:
            raise ValueError(f"无法读取第一帧图像: {frame_paths[0]}")
        first_image_rgb = cv2.cvtColor(first_image, cv2.COLOR_BGR2RGB)
        img_h, img_w = first_image_rgb.shape[:2]
        
        # 扩大边界框（与之前的逻辑一致）
        box_width = initial_box[2] - initial_box[0]
        box_height = initial_box[3] - initial_box[1]
        padding_x = max(box_width * 0.25, 30)
        padding_y = max(box_height * 0.25, 30)
        initial_box[0] = max(0, initial_box[0] - padding_x)
        initial_box[1] = max(0, initial_box[1] - padding_y)
        initial_box[2] = min(img_w, initial_box[2] + padding_x)
        initial_box[3] = min(img_h, initial_box[3] + padding_y)
        
        print(f"🎬 使用SAM2视频预测器（方案一：治本之策）")
        print(f"   初始边界框: {initial_box}")
        print(f"   图像尺寸: {img_w}x{img_h}")
        
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
        
        # 创建临时目录用于视频初始化
        temp_dir = None
        try:
            temp_dir = tempfile.mkdtemp(prefix="sam2_video_")
            print(f"📁 创建临时目录: {temp_dir}")
            
            # 将所有图像文件复制或链接到临时目录（格式：00000.jpg, 00001.jpg, ...）
            for i, frame_path in enumerate(tqdm(frame_paths, desc="准备视频帧", unit="帧")):
                # 读取图像并转换为JPEG格式
                img = Image.open(frame_path).convert("RGB")
                temp_frame_path = Path(temp_dir) / f"{i:05d}.jpg"
                img.save(temp_frame_path, "JPEG", quality=95)
            
            # 初始化视频预测器
            video_predictor = self._get_video_predictor()
            
            # 初始化视频状态
            print(f"🎬 初始化视频状态（{n_frames}帧）...")
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                inference_state = video_predictor.init_state(
                    video_path=temp_dir,
                    offload_video_to_cpu=False,
                    offload_state_to_cpu=False,
                    async_loading_frames=False
                )
            
            # 在第一帧（frame_idx=0）添加边界框prompt
            print(f"📦 在第一帧添加边界框prompt...")
            obj_id = 0  # 对象ID
            frame_idx = 0  # 第一帧
            
            # 边界框格式：[x1, y1, x2, y2] -> [[x1, y1], [x2, y2]]
            box_tensor = torch.tensor(
                [[initial_box[0], initial_box[1]], [initial_box[2], initial_box[3]]],
                dtype=torch.float32,
                device=self.device
            )
            
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                video_predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    box=box_tensor,
                    clear_old_points=True,
                    normalize_coords=True
                )
            
            # 传播到整个视频
            print(f"🚀 开始传播mask到整个视频...")
            masks = []
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                for frame_idx, obj_ids, pred_masks in tqdm(
                    video_predictor.propagate_in_video(inference_state),
                    desc="传播mask",
                    total=n_frames,
                    unit="帧"
                ):
                    # pred_masks 形状: [num_objects, 1, H, W]
                    # 我们只关心第一个对象（obj_id=0）
                    if len(pred_masks) > 0:
                        mask = pred_masks[0][0].cpu().numpy()  # [H, W]
                        masks.append(mask)
                    else:
                        # 如果没有预测到mask，创建一个全零mask
                        video_h = inference_state["video_height"]
                        video_w = inference_state["video_width"]
                        masks.append(np.zeros((video_h, video_w), dtype=np.float32))
            
            # 确保masks数量与帧数一致
            if len(masks) < n_frames:
                # 如果某些帧没有mask，用最后一帧的mask填充
                last_mask = masks[-1] if len(masks) > 0 else np.zeros((img_h, img_w), dtype=np.float32)
                while len(masks) < n_frames:
                    masks.append(last_mask.copy())
            elif len(masks) > n_frames:
                masks = masks[:n_frames]
            
            # 后处理：应用形态学操作和孔洞填充
            print(f"🔧 应用后处理...")
            processed_masks = []
            for i, mask in enumerate(tqdm(masks, desc="后处理mask", unit="帧")):
                # 二值化：降低阈值到-1.0以包含更多"可能是物体"的像素（提高召回率）
                # SAM2输出的是logits，0.0代表50%概率，-1.0代表约27%概率
                mask_binary = (mask > -1.0).astype(np.uint8)
                
                # 形态学操作：先开运算去噪（去除小的背景噪点）
                kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_OPEN, kernel_small)
                
                # 形态学操作：强力闭运算填充孔洞（连接断裂的皮毛纹理）
                kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
                mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel_large)
                
                # 孔洞填充
                mask_binary = binary_fill_holes(mask_binary).astype(np.float32)
                
                processed_masks.append(mask_binary)
            
            masks = processed_masks
            
            print(f"✅ 完成视频mask提取，共 {len(masks)} 帧")
            
        finally:
            # 清理临时目录
            if temp_dir is not None and Path(temp_dir).exists():
                try:
                    shutil.rmtree(temp_dir)
                    print(f"🗑️  已清理临时目录: {temp_dir}")
                except Exception as e:
                    print(f"⚠️  清理临时目录失败: {e}")
        
        # 保存缓存
        if use_cache:
            print(f"💾 保存mask缓存到: {cache_path}")
            torch.save(torch.from_numpy(np.array(masks)), str(cache_path))
            print(f"✅ 已保存 {len(masks)} 个mask到缓存")
        
        # 保存可视化结果
        if save_visualization:
            vis_dir = result_dir / "mask" / target_name
            print(f"🎨 保存可视化结果到: {vis_dir}")
            self._save_visualizations(masks, frame_paths, target_name, result_dir)
            print(f"✅ 已保存 {len(masks)} 个可视化图像")
        
        # 生成debug图像（如果需要）
        if debug_dir is not None and len(debug_frames) > 0:
            print(f"🐛 生成debug图像...")
            for i in debug_frames:
                if i < len(frame_paths) and i < len(masks):
                    try:
                        # 读取图像
                        image = cv2.imread(str(frame_paths[i]))
                        if image is None:
                            continue
                        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        
                        # 使用提取的mask
                        mask_binary = masks[i]
                        
                        # 保存debug可视化（使用-1.0阈值标记，实际mask已处理后处理）
                        self._save_debug_visualization(
                            image_rgb, initial_box, mask_binary, mask_binary, -1.0, i, debug_dir
                        )
                    except Exception as e:
                        print(f"  ⚠️  生成帧 {i} 的debug图像时出错: {e}")
        
        return np.array(masks)
    
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
        high_precision_mode: 高精度模式（保持兼容性，影响集成推理），默认True
        ensemble_passes: 集成推理次数，多次推理并融合结果，默认2（建议2-3次）
    
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
    
    masks = mask_extractor.extract_masks_from_video(
        frame_paths=frame_paths,
        boxes=[reference_box],  # 使用扩大后的边界框
        target_name=target_name,
        save_visualization=save_visualization,
        use_cache=use_cache,
        debug_frames=debug_frames,
        high_precision_mode=high_precision_mode,
        ensemble_passes=ensemble_passes
    )
    
    print("\n" + "=" * 60)
    print("📂 最终保存目录:")
    cache_path = cache_dir / "mask" / f"{target_name}.pt"
    print(f"  💾 缓存文件: {cache_path}")
    if save_visualization:
        vis_dir = result_dir / "mask" / target_name
        print(f"  🎨 可视化结果: {vis_dir}")
    print("=" * 60)
    
    return masks, [reference_box]


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
        "--ensemble_passes",
        type=int,
        default=2,  # 默认2次
        help="集成推理次数，多次推理并融合结果（默认: 2，建议2-3次以获得更高精度）"
    )
    
    args = parser.parse_args()
    
    # 调用提取函数
    ensemble_count = args.ensemble_passes if hasattr(args, 'ensemble_passes') else 2
    masks, boxes = extract_masks_from_text_prompt(
        text_prompt=args.text_prompt,
        target_name=args.target_name,
        sam2_config_path=args.sam2_config,
        sam2_checkpoint_path=args.sam2_checkpoint,
        save_visualization=not args.no_visualization,
        use_cache=not args.no_cache,
        high_precision_mode=True,  # 保持兼容性，但不再使用Resize
        ensemble_passes=ensemble_count
    )
    
    print(f"\n✅ 完成！共提取 {len(masks)} 个mask")
