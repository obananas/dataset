import os
import json
import argparse
import torch
from tqdm import tqdm
from transformers import Qwen3VLMoeForConditionalGeneration, AutoProcessor

def run_inference(data_file, img_root, output_file, model_path):
    print(f"Loading Qwen3-VL MoE model from {model_path}...")
    
    # 1. 加载 MoE 模型
    # A3B 架构代表 Active 3B，虽然总参数量大，但推理时激活参数少，速度快
    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", 
        device_map="auto",
        trust_remote_code=True
    )
    
    # 2. 加载 Processor
    processor = AutoProcessor.from_pretrained(model_path)
    
    # 3. 读取数据
    with open(data_file, 'r', encoding='utf-8') as f:
        lines = [json.loads(line.strip()) for line in f if line.strip()]

    print(f"Total samples: {len(lines)}")
    
    # 4. 循环推理
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for item in tqdm(lines, desc="Qwen3-MoE Inferencing"):
            try:
                image_filename = item.get("image_path", "")
                question = item.get("question", "")
                
                if not image_filename or not question:
                    continue

                full_image_path = os.path.join(img_root, image_filename)
                if not os.path.exists(full_image_path):
                    continue

                # 5. 构建消息格式
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
                # Qwen3-VL 系列支持自动处理多图和视频
                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt"
                ).to(model.device) # 确保输入在模型所在的 GPU 上

                # 7. 生成
                with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs, 
                        max_new_tokens=512,
                        do_sample=False
                    )
                
                # 8. 剪裁并解码
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, 
                    skip_special_tokens=True, 
                    clean_up_tokenization_spaces=False
                )[0]

                # 9. 实时保存
                result_item = item.copy()
                result_item["pred_answer"] = output_text
                
                f_out.write(json.dumps(result_item, ensure_ascii=False) + "\n")
                f_out.flush()

            except Exception as e:
                print(f"\nError: {e}")
                continue

    print(f"Inference done. Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--img_root", type=str, default="./")
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()
    
    run_inference(args.data_file, args.img_root, args.output_file, args.model_path)