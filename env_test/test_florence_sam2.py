import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoProcessor, AutoModelForCausalLM
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# --- 配置 ---
IMAGE_PATH = "E:/Jetbrain_series_project/Pycharm/roto-3/targets/images/bear/00000.jpg"      # 请放一张有熊的图片在根目录
TEXT_PROMPT = "bear"
SAM2_CHECKPOINT = "E:/Jetbrain_series_project/Pycharm/roto-3/sam2_git/checkpoints/sam2.1_hiera_large.pt"
SAM2_CONFIG = "E:/Jetbrain_series_project/Pycharm/roto-3/sam2_git/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

def main():
    # 1. 加载 Florence-2 (自动下载，无需手动配置)
    print("Loading Florence-2...")
    florence_model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True).to(device).eval()
    florence_processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)

    # 2. 加载 SAM 2
    print("Loading SAM 2...")
    sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    # 3. 读取图片
    if not os.path.exists(IMAGE_PATH):
        print("❌ 图片不存在，请放一张 bear.jpg 测试")
        return
    
    image = Image.open(IMAGE_PATH).convert("RGB")
    
    # 4. Florence-2 推理: Text -> Box
    print(f"Detecting '{TEXT_PROMPT}'...")
    task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
    inputs = florence_processor(text=task_prompt + TEXT_PROMPT, images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = florence_model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3
        )
    
    generated_text = florence_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    results = florence_processor.post_process_generation(generated_text, task=task_prompt, image_size=(image.width, image.height))
    
    # 获取 Bbox [x1, y1, x2, y2]
    bboxes = results[task_prompt]['bboxes']
    print(f"✅ Found {len(bboxes)} boxes.")
    
    if len(bboxes) == 0:
        return

    # 5. SAM 2 推理: Box -> Mask
    print("Segmenting with SAM 2...")
    sam2_predictor.set_image(np.array(image))
    input_boxes = np.array(bboxes)
    
    masks, scores, _ = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False
    )

    # 6. 可视化保存
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_mask(masks[0], plt.gca())
    show_box(input_boxes[0], plt.gca())
    plt.axis('off')
    plt.savefig("result_florence.png")
    print("🎉 Success! Result saved to 'result_florence.png'")

def show_mask(mask, ax):
    color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax):
    x0, y0, x1, y1 = box
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor='green', facecolor='none', lw=2))

if __name__ == "__main__":
    main()