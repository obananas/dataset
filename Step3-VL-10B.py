import os
import sys
import json
import argparse
import torch
from tqdm import tqdm
from PIL import Image

def run_inference(data_file, img_root, output_file, model_path):
    # --- 核心修复：将模型路径添加到系统路径 ---
    # 这样 modeling_step.py 里的 import 语句就能找到本地的 vision_encoder.py
    if model_path not in sys.path:
        sys.path.insert(0, model_path)
    
    # 获取模型文件夹绝对路径并切换到该目录（有些模型依赖当前目录读取配置）
    abs_model_path = os.path.abspath(model_path)
    if abs_model_path not in sys.path:
        sys.path.insert(0, abs_model_path)
    # ----------------------------------------

    print(f"Loading Step3-VL-10B from {model_path}...")
    
    from transformers import AutoProcessor, AutoModelForCausalLM

    key_mapping = {
        "^vision_model": "model.vision_model",
        r"^model(?!\.(language_model|vision_model))": "model.language_model",
        "vit_large_projector": "model.vit_large_projector",
    }

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        key_mapping=key_mapping
    ).eval()

    # 读取数据
    with open(data_file, 'r', encoding='utf-8') as f:
        lines = [json.loads(line.strip()) for line in f if line.strip()]

    print(f"Total samples: {len(lines)}")
    
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for item in tqdm(lines, desc="Step3-VL Inferencing"):
            try:
                image_filename = item.get("image_path", "")
                question = item.get("question", "") + "  Skip thinking process and output the answer directly."
                
                full_image_path = os.path.join(img_root, image_filename)
                if not os.path.exists(full_image_path):
                    continue

                image = Image.open(full_image_path).convert("RGB")

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": question}
                        ]
                    },
                ]

                inputs = processor.apply_chat_template(
                    messages, 
                    add_generation_prompt=True, 
                    tokenize=True,
                    return_dict=True, 
                    return_tensors="pt"
                ).to(model.device)

                with torch.no_grad():
                    generate_ids = model.generate(
                        **inputs, 
                        max_new_tokens=512, 
                        do_sample=False,
                        pad_token_id=processor.tokenizer.eos_token_id,
                        eos_token_id=processor.tokenizer.eos_token_id
                    )
                
                input_len = inputs["input_ids"].shape[-1]
                output_text = processor.decode(
                    generate_ids[0, input_len:], 
                    skip_special_tokens=True
                )

                result_item = item.copy()
                result_item["pred_answer"] = output_text.strip()
                
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