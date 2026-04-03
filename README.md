# FactVQA

## Repository Structure

├── images/                 # Test image directory

├── images_archive_part_*.zip  # Compressed image archives

├── FactVQA.jsonl           # Dataset annotations (question-answer-image mapping)

├── *.py                    # Inference scripts for various models

├── results_*.jsonl         # Inference results from evaluated models

└── README.md

## Supported Models
This repository provides ready-to-use inference scripts for:
- HunyuanOCR
- InternVL3-8B / InternVL3_5-8B
- Kimi-VL-A3B-Instruct
- MiMo-VL-7B-RL
- MiniCPM-V-4
- Qwen2.5-VL / Qwen3-VL-8B / Qwen3-VL-30B-A3B
- Step3-VL-10B
- deepseek-vl-7b-chat
- LLaVA-1.5 / LLaVA-1.6

Evaluation results from closed-source models including Claude-4-Sonnet and Gemini-2.5-Pro are also included.
