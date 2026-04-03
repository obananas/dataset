import os
import json
import argparse
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

# InternVL 专用的图像预处理配置
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

def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # 计算目标比例
    target_ratios = set()
    for i in range(1, max_num + 1):
        for j in range(1, max_num + 1):
            if i * j <= max_num:
                target_ratios.add((i, j))
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # 裁剪与缩放
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height), Image.BICUBIC)
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size), Image.BICUBIC)
        processed_images.append(thumbnail_img)
    return processed_images

def run_inference(data_file, img_root, output_file, model_path):
    print(f"Loading InternVL3 model from {model_path}...")
    
    # 1. 加载模型与 Tokenizer
    # InternVL3 通常使用 AutoModel 直接加载，并开启 trust_remote_code
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True
    ).eval().cuda()
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    
    # 2. 准备数据预处理
    transform = build_transform(input_size=448)

    # 3. 读取数据
    with open(data_file, 'r', encoding='utf-8') as f:
        lines = [json.loads(line.strip()) for line in f if line.strip()]

    print(f"Total samples: {len(lines)}")
    
    # 4. 循环推理
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for item in tqdm(lines, desc="InternVL3 Inferencing"):
            try:
                image_filename = item.get("image_path", "")
                question = item.get("question", "")
                
                if not image_filename or not question:
                    continue

                full_image_path = os.path.join(img_root, image_filename)
                if not os.path.exists(full_image_path):
                    print(f"Warning: Image not found at {full_image_path}")
                    continue

                # 5. 图像加载与动态切片预处理
                pixel_values = []
                image = Image.open(full_image_path).convert('RGB')
                curr_images = dynamic_preprocess(image, image_size=448, use_thumbnail=True, max_num=6)
                for img in curr_images:
                    pixel_values.append(transform(img))
                pixel_values = torch.stack(pixel_values).to(torch.bfloat16).cuda()

                # 6. 构建 Prompt 格式
                # InternVL 默认格式: <image>\nQuestion
                prompt = f"<image>\n{question}"
                
                # 7. 生成
                generation_config = dict(max_new_tokens=64, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id)
                
                with torch.no_grad():
                    response = model.chat(
                        tokenizer,
                        pixel_values,
                        prompt,
                        generation_config
                    )

                # 8. 保存结果
                result_item = item.copy()
                result_item["pred_answer"] = response
                
                f_out.write(json.dumps(result_item, ensure_ascii=False) + "\n")
                f_out.flush()

            except Exception as e:
                print(f"\nError processing sample: {e}")
                continue

    print(f"Inference done. Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, required=True, help="Path to input jsonl")
    parser.add_argument("--img_root", type=str, default="./", help="Root directory for images")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save output jsonl")
    parser.add_argument("--model_path", type=str, required=True, help="Local path or HF ID")
    args = parser.parse_args()
    
    run_inference(args.data_file, args.img_root, args.output_file, args.model_path)