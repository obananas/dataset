import os
import json
import argparse
import torch
from tqdm import tqdm
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

def run_inference(data_file, img_root, output_file, model_path):
    print(f"Loading Kimi-VL model from {model_path}...")
    
    # 1. 加载模型
    # 按照 Kimi-VL 官方建议，使用 bfloat16 和 flash_attention_2 提升效率
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        trust_remote_code=True
    )
    
    # 2. 加载 Processor
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    # 3. 读取数据
    with open(data_file, 'r', encoding='utf-8') as f:
        lines = [json.loads(line.strip()) for line in f if line.strip()]

    print(f"Total samples: {len(lines)}")
    
    # 4. 循环推理
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for item in tqdm(lines, desc="Kimi-VL Inferencing"):
            try:
                image_filename = item.get("image_path", "")
                question = item.get("question", "")
                
                if not image_filename or not question:
                    continue

                full_image_path = os.path.join(img_root, image_filename)
                if not os.path.exists(full_image_path):
                    print(f"Warning: Image not found at {full_image_path}")
                    continue

                # 5. 打开图像并构建 Chat 模板
                # 注意：Kimi-VL 的 processor.apply_chat_template 此时处理的是文本占位符
                raw_image = Image.open(full_image_path).convert("RGB")
                
                messages = [
                    {
                        "role": "user", 
                        "content": [
                            {"type": "image", "image": full_image_path}, # 模板中通常传入路径占位
                            {"type": "text", "text": question}
                        ]
                    }
                ]
                
                # 6. 准备输入
                # 先应用聊天模板获取 prompt 文本
                prompt_text = processor.apply_chat_template(
                    messages, 
                    add_generation_prompt=True, 
                    tokenize=False # 这里先获取字符串
                )
                
                # 再通过 processor 将图像和文本编码为 tensors
                inputs = processor(
                    images=raw_image, 
                    text=prompt_text, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True
                ).to(model.device)

                # 7. 生成
                with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs, 
                        max_new_tokens=512,
                        do_sample=False # 保证结果一致性
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
    parser = argparse.ArgumentParser(description="Kimi-VL Batch Inference Script")
    parser.add_argument("--data_file", type=str, required=True, help="Path to input jsonl")
    parser.add_argument("--img_root", type=str, default="./", help="Root directory for images")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save output jsonl")
    parser.add_argument("--model_path", type=str, required=True, help="Local path or HF ID (moonshotai/Kimi-VL-A3B-Instruct)")
    args = parser.parse_args()
    
    run_inference(args.data_file, args.img_root, args.output_file, args.model_path)