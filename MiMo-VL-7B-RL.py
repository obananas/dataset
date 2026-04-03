import os
import json
import argparse
import torch
from tqdm import tqdm
from PIL import Image
# 核心修改：引入正确的加载类
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

def run_inference(data_file, img_root, output_file, model_path):
    print(f"Loading MiMo-VL model (Qwen2.5-VL based) from {model_path}...")
    
    # 1. 使用正确的类加载模型
    # MiMo-VL-7B-RL 基于 Qwen2.5-VL，必须使用这个类才能识别其 Config
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16, # 或者使用 dtype=torch.bfloat16 消除警告
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    
    # 2. 加载 Processor
    processor = AutoProcessor.from_pretrained(model_path)
    
    with open(data_file, 'r', encoding='utf-8') as f:
        lines = [json.loads(line.strip()) for line in f if line.strip()]

    print(f"Total samples: {len(lines)}")
    
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for item in tqdm(lines, desc="Inferencing"):
            try:
                image_filename = item.get("image_path", "")
                question = item.get("question", "")
                
                if not image_filename or not question:
                    continue

                full_image_path = os.path.join(img_root, image_filename)

                # MiMo-VL 继承了 Qwen2.5-VL 的消息格式
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": full_image_path},
                            {"type": "text", "text": question+"/no_think"},
                        ],
                    }
                ]

                # 准备输入
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
                ).to(model.device)

                # 生成
                generated_ids = model.generate(**inputs, max_new_tokens=512)
                
                # 剪切 input 部分并解码
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]

                result_item = item.copy()
                result_item["pred_answer"] = output_text
                
                f_out.write(json.dumps(result_item, ensure_ascii=False) + "\n")
                f_out.flush()

            except Exception as e:
                print(f"\nError: {e}")
                continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--img_root", type=str, default="./")
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()
    
    run_inference(args.data_file, args.img_root, args.output_file, args.model_path)