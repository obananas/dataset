import os
import json
import argparse
import torch
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

def run_inference(data_file, img_root, output_file, model_path):
    print(f"Loading model from {model_path}...")
    
    # 1. 加载模型
    # Qwen2.5-VL 推荐使用 bfloat16，并开启 flash_attention_2 (如果显卡支持)
    # 如果显存不足，可以尝试 device_map="auto" 或者加载量化版本
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    
    # 2. 加载 Processor
    processor = AutoProcessor.from_pretrained(model_path)
    
    # 3. 读取数据
    with open(data_file, 'r', encoding='utf-8') as f:
        lines = [json.loads(line.strip()) for line in f if line.strip()]

    print(f"Total samples: {len(lines)}")
    
    results = []

    # 4. 循环推理
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for item in tqdm(lines, desc="Inferencing"):
            try:
                # 获取图片路径和问题
                # 假设 JSONL 里的键是 "image" 和 "question"
                image_filename = item.get("image_path", "")
                question = item.get("question", "")
                
                if not image_filename or not question:
                    print(f"Skipping invalid item: {item}")
                    continue

                full_image_path = os.path.join(img_root, image_filename)

                # 构建 Qwen-VL 要求的 message 格式
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": full_image_path,
                            },
                            {"type": "text", "text": question},
                        ],
                    }
                ]

                # 准备推理输入
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                
                image_inputs, video_inputs = process_vision_info(messages)
                
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                
                # 将输入移动到模型所在的设备
                inputs = inputs.to(model.device)

                # 生成
                # max_new_tokens 可以根据需要调整
                generated_ids = model.generate(**inputs, max_new_tokens=512)
                
                # 截取生成的这部分 token（去掉输入的 token）
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]

                # 将结果存回 item 或创建新字典
                result_item = item.copy()
                result_item["pred_answer"] = output_text
                
                # 实时写入文件（防止中断丢失）
                f_out.write(json.dumps(result_item, ensure_ascii=False) + "\n")
                f_out.flush()

            except Exception as e:
                print(f"\nError processing item {item}: {e}")
                continue

    print(f"Inference done. Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, required=True, help="Path to input jsonl file")
    parser.add_argument("--img_root", type=str, required=False, default="./", help="Root directory for images")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save output jsonl")
    parser.add_argument("--model_path", type=str, required=True, help="Local path or HuggingFace ID of the model")
    args = parser.parse_args()
    
    run_inference(args.data_file, args.img_root, args.output_file, args.model_path)