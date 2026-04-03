import json
import string
import collections
import os
from difflib import SequenceMatcher
import re

class OCREvaluator:
    def __init__(self, input_file):
        self.input_file = input_file

    def normalize_text(self, s):
        """
        【升级版】标准化文本:
        1. 兼容性替换 (& -> and)
        2. 转小写
        3. 移除特定前缀 (如 "the answer is")
        4. 数字单词转数字 (one -> 1)
        5. 移除冠词 (a, an, the)
        6. 移除标点
        7. 移除多余空白
        """
        if s is None:
            return ""
        s = str(s).lower().strip()

        # --- 步骤 A: 移除常见模型废话前缀 ---
        # 很多模型喜欢输出 "the answer is 5"，而 GT 只有 "5"
        prefixes = ["the answer is ", "answer:", "predicted:", "result:"]
        for p in prefixes:
            if s.startswith(p):
                s = s[len(p):].strip()

        # --- 步骤 B: 数字单词转数字 (Word-to-Digit) ---
        # 这是一个手动映射表，解决 "two" != "2" 的问题
        num_map = {
            'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
            'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
            'ten': '10'
        }
        # 使用正则把单词替换为数字 (匹配独立的单词)
        for word, digit in num_map.items():
            # \b 表示单词边界，防止把 "zone" 里的 "one" 替换掉
            s = re.sub(r'\b' + word + r'\b', digit, s)

        # --- 步骤 C: 替换特殊符号和单位 ---
        s = s.replace('&', 'and')
        s = s.replace('percent', '') # 因为下面我们会把 % 删掉，所以这里要把 percent 也删掉保持一致
        s = s.replace('dollars', '') # 同理，GT可能是 $10 (去标点后是10)，Pred是 10 dollars
        s = s.replace('<think>\n</think>\n\n', '')

        # --- 步骤 D: 移除冠词 (Stop words) ---
        # 解决 "the apple" != "apple"
        s = re.sub(r'\b(a|an|the)\b', '', s)

        # --- 步骤 E: 移除标点符号 (原逻辑) ---
        translator = str.maketrans('', '', string.punctuation)
        s = s.translate(translator)
        
        # --- 步骤 F: 规范化空格 (原逻辑) ---
        s = " ".join(s.split())
        
        return s

    def _levenshtein(self, seq1, seq2):
        """计算编辑距离的核心算法 (DP实现)"""
        size_x = len(seq1) + 1
        size_y = len(seq2) + 1
        matrix = [[0 for _ in range(size_y)] for _ in range(size_x)]
        for x in range(size_x): matrix[x][0] = x
        for y in range(size_y): matrix[0][y] = y

        for x in range(1, size_x):
            for y in range(1, size_y):
                if seq1[x-1] == seq2[y-1]:
                    matrix[x][y] = min(
                        matrix[x-1][y] + 1,
                        matrix[x-1][y-1],
                        matrix[x][y-1] + 1
                    )
                else:
                    matrix[x][y] = min(
                        matrix[x-1][y] + 1,
                        matrix[x-1][y-1] + 1,
                        matrix[x][y-1] + 1
                    )
        return matrix[size_x-1][size_y-1]

    def calculate_metrics(self, gt, pred):
        """计算单条数据的所有指标"""
        # 1. Accuracy (Exact Match)
        acc = 1.0 if gt == pred else 0.0
        
        # 【新增】Soft Accuracy (包含匹配)
        # 逻辑：如果 GT 包含在 Pred 中，或者 Pred 包含在 GT 中
        # 举例：GT="white", Pred="white color" -> Soft Acc = 1.0
        soft_acc = 0.0
        if gt and pred: # 防止空字符串
             if gt in pred or pred in gt:
                 soft_acc = 1.0
        elif gt == pred: # 都是空字符串
             soft_acc = 1.0

        # 2. Similarity (Sequence Matcher)
        sim = SequenceMatcher(None, gt, pred).ratio()
        
        # 3. Edit Distance (Levenshtein)
        edit_dist = self._levenshtein(gt, pred)
        
        # 4. CER (Character Error Rate) = EditDist / len(GT)
        # 防止除以0
        cer = edit_dist / len(gt) if len(gt) > 0 else (0.0 if len(pred) == 0 else 1.0)
        
        # 5. WER (Word Error Rate)
        gt_words = gt.split()
        pred_words = pred.split()
        word_edit_dist = self._levenshtein(gt_words, pred_words)
        wer = word_edit_dist / len(gt_words) if len(gt_words) > 0 else (0.0 if len(pred_words) == 0 else 1.0)
        
        # 6. F1 Score (Token level)
        common = collections.Counter(gt_words) & collections.Counter(pred_words)
        num_same = sum(common.values())
        if len(gt_words) == 0 or len(pred_words) == 0:
            f1 = 1.0 if gt_words == pred_words else 0.0
        else:
            precision = 1.0 * num_same / len(pred_words)
            recall = 1.0 * num_same / len(gt_words)
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            
        return {
            "acc": acc,
            "soft_acc": soft_acc,
            "sim": sim,
            "dist": edit_dist,
            "cer": cer,
            "wer": wer,
            "f1": f1
        }

    def process_dataset(self, data_list):
        """主处理逻辑：分组并计算"""
        groups = collections.defaultdict(list)
        
        cat_mapping = {
            'Books': 'Scene-Text', 'Business': 'Scene-Text', 'Commons': 'Scene-Text',
            'Markers': 'Scene-Text', 'Metal': 'Scene-Text', 'Quotes': 'Scene-Text', 'Signposts': 'Scene-Text',
            'Charts': 'Document', 'Documents': 'Document', 'Invoices': 'Document'
        }

        print(f"正在处理 {len(data_list)} 条数据...")
        
        for item in data_list:
            # 获取原始数据
            raw_gt = item.get('gt_answer', '')
            raw_pred = item.get('pred_answer', '')
            img_path = item.get('image_path', '')
            category = item.get('category', 'Unknown')
            
            # 标准化
            norm_gt = self.normalize_text(raw_gt)
            norm_pred = self.normalize_text(raw_pred)
            
            # 计算指标
            metrics = self.calculate_metrics(norm_gt, norm_pred)
            
            # 存入不同分组
            # 1. 总体
            groups['Overall'].append(metrics)
            
            # 2. 按是否编辑 (Edit vs Original)
            # 假设文件名包含 "_edit" 为编辑过的图
            if '_edit' in str(img_path):
                groups['Type: Edited'].append(metrics)
            else:
                groups['Type: Original'].append(metrics)
                

            # 3. 按类别 (Category)
            groups[f'Cat: {category}'].append(metrics)

            big_cat = cat_mapping.get(category, "Others")
            groups[f'Big_Cat: {big_cat}'].append(metrics)
        return groups

    def print_report(self, groups):
        """打印格式化报表"""
        # 定义表头
        headers = ["Dataset Group", "Count", "Acc %", "Soft_Acc %", "F1 %", "Sim %", "CER ↓", "WER ↓"]
        row_fmt = "{:<20} | {:<7} | {:<10} | {:<10} | {:<7} | {:<7} | {:<7} | {:<7}"
        
        print("\n" + "="*94)
        print(f"📊 FactVQA BENCHMARK EVALUATION REPORT on  {self.input_file}")
        print("="*94)
        print(row_fmt.format(*headers))
        print("-" * 94)
        
        # 排序：Overall -> Type -> Category
        def sort_rank(key):
            if key == 'Overall': return 0
            if key.startswith('Grouping'): return 1
            if key.startswith('Type'): return 2
            return 3

        sorted_keys = sorted(groups.keys(), key=lambda x: (sort_rank(x), x))
        
        for key in sorted_keys:
            metrics_list = groups[key]
            count = len(metrics_list)
            if count == 0: continue
            
            # 计算平均值
            avg_acc = sum(m['acc'] for m in metrics_list) / count * 100
            avg_soft_acc = sum(m['soft_acc'] for m in metrics_list) / count * 100
            avg_f1 = sum(m['f1'] for m in metrics_list) / count * 100
            avg_sim = sum(m['sim'] for m in metrics_list) / count * 100
            avg_cer = sum(m['cer'] for m in metrics_list) / count
            avg_wer = sum(m['wer'] for m in metrics_list) / count
            
            print(row_fmt.format(
                key, 
                count, 
                f"{avg_acc:.1f}", 
                f"{avg_soft_acc:.1f}", 
                f"{avg_f1:.1f}", 
                f"{avg_sim:.1f}", 
                f"{avg_cer:.2f}", 
                f"{avg_wer:.2f}"
            ))
        print("-" * 94)
        print("说明: Acc/F1/Sim 越高越好 (↑). CER/WER 越低越好 (↓).")

def load_jsonl(file_path):
    """读取 jsonl 文件"""
    data = []
    if not os.path.exists(file_path):
        print(f"❌ 错误: 找不到文件 {file_path}")
        return []
        
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        print(f"⚠️ 警告: 第 {line_number} 行不是有效的 JSON，已跳过。")
        print(f"✅ 成功加载 {len(data)} 条数据，来自 {file_path}")
    except Exception as e:
        print(f"❌ 读取文件出错: {e}")
    return data

# ================= 执行逻辑 =================
if __name__ == "__main__":
    # 配置你的文件名
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    args = parser.parse_args()
    
    # 1. 检查是否存在文件，不存在则创建一个示例文件演示
    if not os.path.exists(args.input_file):
        print(f"文件 {args.input_file} 不存在，正在创建示例文件...")
        sample_data = [
            {"id": 1, "question": "Example Q1", "gt_answer": "No", "pred_answer": "No", "image_path": "./Books/img_001.jpg", "category": "Books"},
            {"id": 2, "question": "Example Q2", "gt_answer": "2022", "pred_answer": "2008", "image_path": "./Business/img_001.jpg", "category": "Business"}
        ]
        with open(args.input_file, 'w', encoding='utf-8') as f:
            for item in sample_data:
                f.write(json.dumps(item) + "\n")
    
    # 2. 加载数据
    dataset = load_jsonl(args.input_file)

    if dataset:
        # 3. 运行评测
        evaluator = OCREvaluator(args.input_file)
        results = evaluator.process_dataset(dataset)
        
        # 4. 打印报表
        evaluator.print_report(results)