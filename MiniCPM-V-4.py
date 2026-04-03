import os
import json
import argparse
import torch
from tqdm import tqdm
from PIL import Image
from transformers import AutoModel, AutoTokenizer

def run_inference(data_file, img_root, output_file, model_path):
    print(f"Loading MiniCPM-V model from {model_path}...")
    
    # 1. 加载模型
    # MiniCPM-V 推荐使用 bf16，如果显卡不支持 flash_attention_2，可以使用 sdpa
    model = AutoModel.from_pretrained(
        model_path, 
        trust_remote_code=True,
        attn_implementation='sdpa', 
        torch_dtype=torch.bfloat16
    )
    model = model.eval().cuda()
    
    # 2. 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # 3. 读取数据
    with open(data_file, 'r', encoding='utf-8') as f:
        lines = [json.loads(line.strip()) for line in f if line.strip()]

    print(f"Total samples: {len(lines)}")
    
    # 4. 循环推理
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for item in tqdm(lines, desc="MiniCPM-V Inferencing"):
            try:
                image_filename = item.get("image_path", "")
                question = item.get("question", "")
                
                if not image_filename or not question:
                    continue

                full_image_path = os.path.join(img_root, image_filename)
                if not os.path.exists(full_image_path):
                    print(f"Image not found: {full_image_path}")
                    continue
                
                # 加载图片并转换为 RGB
                image = Image.open(full_image_path).convert('RGB')

                # 5. 构建 MiniCPM-V 要求的 msgs 格式
                # 注意：MiniCPM-V 的 content 列表中可以直接放入 Image 对象
                msgs = [{'role': 'user', 'content': [image, question]}]

                # 6. 使用 model.chat 进行推理
                # MiniCPM-V 内部会自动处理图片 resize 和 prompt 拼接
                with torch.no_grad():
                    answer = model.chat(
                        image=None,  # 如果 image 已在 msgs 中，这里可以传 None
                        msgs=msgs,
                        tokenizer=tokenizer,
                        sampling=False,   # sampling=False 相当于 greedy search，结果更稳定
                        temperature=0.7   # 只有 sampling=True 时才生效
                    )

                # 7. 保存结果
                result_item = item.copy()
                result_item["pred_answer"] = answer
                
                f_out.write(json.dumps(result_item, ensure_ascii=False) + "\n")
                f_out.flush()

            except Exception as e:
                print(f"\nError processing {item.get('id', 'unknown')}: {e}")
                continue

    print(f"Inference done. Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, required=True, help="Path to input jsonl file")
    parser.add_argument("--img_root", type=str, default="./", help="Root directory for images")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save output jsonl")
    parser.add_argument("--model_path", type=str, required=True, help="Path to MiniCPM-V model")
    args = parser.parse_args()
    
    run_inference(args.data_file, args.img_root, args.output_file, args.model_path)