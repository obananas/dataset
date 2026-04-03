import os
import json
import argparse
import torch
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, HunYuanVLForConditionalGeneration

def clean_repeated_substrings(text):
    """清理生成文本中可能出现的异常重复子串 (HunyuanOCR 官方推荐)"""
    n = len(text)
    if n < 8000:
        return text
    for length in range(2, n // 10 + 1):
        candidate = text[-length:] 
        count = 0
        i = n - length
        
        while i >= 0 and text[i:i + length] == candidate:
            count += 1
            i -= length

        if count >= 10:
            return text[:n - length * (count - 1)]  
    return text

def run_inference(data_file, img_root, output_file, model_path):
    print(f"Loading HunyuanOCR model from {model_path}...")
    
    # 1. 加载 Processor (HunyuanOCR 建议 use_fast=False)
    processor = AutoProcessor.from_pretrained(model_path, use_fast=False)
    
    # 2. 加载模型
    model = HunYuanVLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager", # 匹配官方示例
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    
    # 3. 读取数据
    with open(data_file, 'r', encoding='utf-8') as f:
        lines = [json.loads(line.strip()) for line in f if line.strip()]

    print(f"Total samples: {len(lines)}")
    
    # 4. 循环推理
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for item in tqdm(lines, desc="HunyuanOCR Inferencing"):
            try:
                image_filename = item.get("image_path", "")
                question = item.get("question", "检测并识别图片中的文字，将文本坐标格式化输出。")
                
                if not image_filename:
                    continue

                full_image_path = os.path.join(img_root, image_filename)
                if not os.path.exists(full_image_path):
                    print(f"\nWarning: Image not found at {full_image_path}")
                    continue

                # 5. 打开图片
                image_input = Image.open(full_image_path).convert("RGB")

                # 6. 构建 Chat 格式
                messages = [
                    {"role": "system", "content": ""},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": full_image_path},
                            {"type": "text", "text": question},
                        ],
                    }
                ]

                # 7. 准备输入
                prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(
                    text=[prompt],
                    images=image_input,
                    padding=True,
                    return_tensors="pt",
                ).to(model.device)

                # 8. 生成
                with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs, 
                        max_new_tokens=16384, 
                        do_sample=False
                    )
                
                # 9. 剪裁并解码
                input_ids = inputs.input_ids if "input_ids" in inputs else inputs.inputs
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)
                ]
                
                raw_output = processor.batch_decode(
                    generated_ids_trimmed, 
                    skip_special_tokens=True, 
                    clean_up_tokenization_spaces=False
                )[0]

                # 10. 后处理清洗
                final_output = clean_repeated_substrings(raw_output)

                # 11. 实时保存结果
                result_item = item.copy()
                result_item["pred_answer"] = final_output
                
                f_out.write(json.dumps(result_item, ensure_ascii=False) + "\n")
                f_out.flush()

            except Exception as e:
                print(f"\nError processing sample {image_filename}: {e}")
                continue

    print(f"Inference done. Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, required=True, help="Path to input jsonl")
    parser.add_argument("--img_root", type=str, default="./", help="Root directory for images")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save output jsonl")
    parser.add_argument("--model_path", type=str, default="tencent/HunyuanOCR", help="Local path or HF ID")
    args = parser.parse_args()
    
    run_inference(args.data_file, args.img_root, args.output_file, args.model_path)