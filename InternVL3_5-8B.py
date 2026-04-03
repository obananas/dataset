import os
import json
import argparse
import torch
import numpy as np
import torchvision.transforms as T
from tqdm import tqdm
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from torchvision.transforms.functional import InterpolationMode

# --- 官方图像预处理逻辑 ---
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(tile) for tile in images]
    return torch.stack(pixel_values)

# --- 主推理逻辑 ---
def run_inference(data_file, img_root, output_file, model_path):
    print(f"Loading InternVL3.5 from {model_path}...")
    
    # 1. 加载模型
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map="auto"
    ).eval()
    
    # 2. 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

    # 3. 读取数据
    with open(data_file, 'r', encoding='utf-8') as f:
        lines = [json.loads(line.strip()) for line in f if line.strip()]

    print(f"Total samples: {len(lines)}")
    
    # 4. 配置生成参数 (根据官方建议设置并消除警告)
    generation_config = dict(
        max_new_tokens=512, 
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    # 5. 循环处理
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for item in tqdm(lines, desc="Processing"):
            try:
                img_path = os.path.join(img_root, item.get("image_path", ""))
                question = item.get("question", "")
                
                if not os.path.exists(img_path):
                    continue

                # 处理图片
                pixel_values = load_image(img_path, max_num=12).to(torch.bfloat16).cuda()
                
                # InternVL 标准指令格式：一定要带 <image>\n
                full_question = f'<image>\n{question}'
                
                with torch.no_grad():
                    response = model.chat(
                        tokenizer=tokenizer,
                        pixel_values=pixel_values,
                        question=full_question,
                        generation_config=generation_config
                    )

                # 保存结果
                result = item.copy()
                result["pred_answer"] = response
                f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                f_out.flush()

            except Exception as e:
                print(f"\nError processing {item}: {e}")
                continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--img_root", type=str, default="./")
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()
    
    run_inference(args.data_file, args.img_root, args.output_file, args.model_path)