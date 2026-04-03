import argparse
import torch
from PIL import Image
import json
import os
from tqdm import tqdm
import sys

import warnings

# 过滤掉特定的元参数复制警告
warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")
# 导入原版 LLaVA 相关的组件
llava_project_root = "" 
if llava_project_root not in sys.path:
    sys.path.insert(0, llava_project_root)

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

def load_model(model_path):
    print(f"正在加载原版 LLaVA 模型: {model_path} ...")
    disable_torch_init()
    
    model_name = get_model_name_from_path(model_path)
    # load_pretrained_model 会自动处理 device_map
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=model_name,
        device_map="auto",
        torch_dtype="float16"
    )
    return model, tokenizer, image_processor, model_name

def process_batch(batch_items, image_root, model, tokenizer, image_processor, model_name):
    """
    处理一个 Batch 的数据。
    注意：原版 LLaVA 对 Batch 推理的支持不如 HF 完善，通常建议单条或小 Batch 手动循环。
    这里为了性能，采用循环推理，但保持了 Batch 接口一致性。
    """
    results = []

    # 确定对话模板
    # llava-v1.5/1.6 通常使用 "llava_v1" 或 "vicuna_v1"
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v1"

    for item in batch_items:
        img_rel_path = item['image_path']
        question = item['question']
        img_path = os.path.join(image_root, img_rel_path)
        
        try:
            image = Image.open(img_path).convert("RGB")
            # 处理图片 (原版 LLaVA 1.6 会处理多尺度)
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()
        except Exception as e:
            print(f"\nWarning: 加载图片失败 {img_path}, 错误: {e}")
            continue

        # 构建 Prompt
        conv = conv_templates[conv_mode].copy()
        # 拼接图片占位符
        qs = question
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # Tokenize 文本和图片占位符
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        # 生成
        stop_str = conv.sep if conv.sep_style in [SeparatorStyle.SINGLE, SeparatorStyle.MPT] else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                max_new_tokens=200,
                use_cache=True,
                stopping_criteria=[stopping_criteria]
            )

        # 解码
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        
        # 清理生成的结尾停止符
        if output_text.endswith(stop_str):
            output_text = output_text[:-len(stop_str)].strip()

        results.append({
            "id": item['id'],
            "question": item['question'],
            "gt_answer": item['gt_answer'],
            "pred_answer": output_text,
            "image_path": item['image_path'],
            "category": item.get('category', '')
        })
        
    return results

def run_inference(data_path, image_root, output_path, model_path, batch_size=1):
    # 加载模型
    model, tokenizer, image_processor, model_name = load_model(model_path)
    
    # 读取数据
    with open(data_path, 'r', encoding='utf-8') as f:
        all_lines = [json.loads(line) for line in f.readlines()]
    
    print(f"开始推理，共 {len(all_lines)} 条数据...")

    with open(output_path, 'w', encoding='utf-8') as f_out:
        # 原版代码通常逐条推理稳定性更高，此处按 batch_size 逻辑循环
        for i in tqdm(range(0, len(all_lines), batch_size)):
            batch_items = all_lines[i : i + batch_size]
            batch_results = process_batch(batch_items, image_root, model, tokenizer, image_processor, model_name)
            
            for res in batch_results:
                f_out.write(json.dumps(res, ensure_ascii=False) + "\n")
            f_out.flush()
    
    print(f"推理完成，结果已保存至: {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--img_root", type=str, required=False, default="./")
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=False, default=1)
    args = parser.parse_args()
    run_inference(args.data_file, args.img_root, args.output_file, args.model_path, batch_size=args.batch_size)
