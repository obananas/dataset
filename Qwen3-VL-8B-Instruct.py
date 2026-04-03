import os
import json
import argparse
import torch
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

def run_inference(data_file, img_root, output_file, model_path):
    print(f"Loading Qwen3-VL model from {model_path}...")
    
    # 1. 加载模型
    # Qwen3-VL 建议使用 bfloat16 以获得最佳精度与速度平衡
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", # 强烈建议开启，显著降低长文本显存占用
        device_map="auto",
        trust_remote_code=True
    )
    
    # 2. 加载 Processor (包含 Tokenizer 和图像处理逻辑)
    processor = AutoProcessor.from_pretrained(model_path)
    
    # 3. 读取数据
    with open(data_file, 'r', encoding='utf-8') as f:
        lines = [json.loads(line.strip()) for line in f if line.strip()]

    print(f"Total samples: {len(lines)}")
    
    # 4. 循环推理
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for item in tqdm(lines, desc="Qwen3-VL Inferencing"):
            try:
                image_filename = item.get("image_path", "")
                question = item.get("question", "")
                
                if not image_filename or not question:
                    continue

                full_image_path = os.path.join(img_root, image_filename)
                if not os.path.exists(full_image_path):
                    print(f"Warning: Image not found at {full_image_path}")
                    continue

                # 5. 构建标准 Chat 格式
                # Qwen3-VL 能够原生处理本地路径字符串
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": full_image_path},
                            {"type": "text", "text": question},
                        ],
                    }
                ]

                # 6. 准备输入
                # Qwen3 的 Processor 自动处理图像缩放与 patch 划分
                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt"
                ).to(model.device)

                # 7. 生成
                with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs, 
                        max_new_tokens=512,
                        do_sample=False # 设为 False 保证预测结果的一致性
                    )
                
                # 8. 剪裁输入部分并解码
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, 
                    skip_special_tokens=True, 
                    clean_up_tokenization_spaces=False
                )[0]

                # 9. 实时保存结果
                result_item = item.copy()
                result_item["pred_answer"] = output_text
                
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