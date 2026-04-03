import os
import json
import argparse
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images

def run_inference(data_file, img_root, output_file, model_path):
    print(f"Loading DeepSeek-VL model from {model_path}...")
    
    # 1. 加载 Processor 和 Tokenizer
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    # 2. 加载模型 (官方推荐使用 AutoModelForCausalLM + trust_remote_code)
    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path, 
        trust_remote_code=True
    )
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()
    
    # 3. 读取待处理数据
    with open(data_file, 'r', encoding='utf-8') as f:
        lines = [json.loads(line.strip()) for line in f if line.strip()]

    print(f"Total samples to process: {len(lines)}")
    
    # 4. 循环推理
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for item in tqdm(lines, desc="DeepSeek-VL Inferencing"):
            try:
                image_filename = item.get("image_path", "")
                question = item.get("question", "")
                
                if not image_filename or not question:
                    continue

                # 拼接完整图片路径
                full_image_path = os.path.join(img_root, image_filename)
                if not os.path.exists(full_image_path):
                    print(f"Image not found: {full_image_path}")
                    continue

                # 5. 构建官方要求的 Conversation 格式
                conversation = [
                    {
                        "role": "User",
                        "content": f"<image_placeholder>{question}",
                        "images": [full_image_path],
                    },
                    {"role": "Assistant", "content": ""},
                ]

                # 6. 准备输入
                pil_images = load_pil_images(conversation)
                prepare_inputs = vl_chat_processor(
                    conversations=conversation,
                    images=pil_images,
                    force_batchify=True
                ).to(vl_gpt.device)

                # 7. 提取视觉特征并融合
                inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

                # 8. 生成回答 (注意：调用 vl_gpt.language_model)
                with torch.no_grad():
                    outputs = vl_gpt.language_model.generate(
                        inputs_embeds=inputs_embeds,
                        attention_mask=prepare_inputs.attention_mask,
                        pad_token_id=tokenizer.eos_token_id,
                        bos_token_id=tokenizer.bos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        max_new_tokens=512,
                        do_sample=False,
                        use_cache=True
                    )

                # 9. 解码结果
                answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)

                # 保存
                result_item = item.copy()
                result_item["pred_answer"] = answer
                
                f_out.write(json.dumps(result_item, ensure_ascii=False) + "\n")
                f_out.flush()

            except Exception as e:
                print(f"\nError processing sample {item.get('id', 'unknown')}: {e}")
                continue

    print(f"Done! Results saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, required=True, help="输入 JSONL 文件路径")
    parser.add_argument("--img_root", type=str, default="./", help="图片根目录")
    parser.add_argument("--output_file", type=str, required=True, help="结果输出 JSONL 路径")
    parser.add_argument("--model_path", type=str, required=True, help="模型权重路径")
    args = parser.parse_args()
    
    run_inference(args.data_file, args.img_root, args.output_file, args.model_path)