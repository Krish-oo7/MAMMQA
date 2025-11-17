# Rethinking Information Synthesis in Multimodal Question Answering A Multi-Agent Perspective 
**Paper:** [*Rethinking Information Synthesis in Multimodal Question Answering: A Multi-Agent Perspective*](https://arxiv.org/abs/2505.20816) 📚📊🔍\
**Authors:** Tejas Anvekar*, Krishna Singh Rajput*, Chitta Baral, Vivek Gupta  
Arizona State University  

---

## Overview

This repository contains the official implementation of **MAMMQA**, a **prompt-driven, multi-agent system** for multimodal question answering (MMQA).

![MAMMQA Framework](https://github.com/Krish-oo7/MAMMQA/blob/main/Assets/Architecture.png)

The framework decomposes reasoning across **three interpretable agents**:

1. **Modality Expert Agents** — extract insights from text, tables, or images.  
2. **Cross-Modality Synthesis Agents** — integrate these insights to form cross-modal reasoning chains.  
3. **Aggregator Agent** — synthesizes multiple agent outputs into a final, evidence-grounded answer.

Unlike traditional monolithic or fine-tuned MMQA models, MAMMQA is **zero-shot**, modular, and **LLM-agnostic**, compatible with both **OpenAI GPT-4o**, **Gemini 1.5-Flash**, and **Qwen2.5-VL** models.

---
## ⚙️ Setup and Installation
1. Prerequisites
Ensure you have **Python 3.8+** installed.

2. Dependencies
Install the required Python packages using pip:
```Bash
pip install pandas openai tqdm python-dotenv
```
3. API Key Configuration
The agents rely on the `openai` library to interface with various Large Language Models (LLMs) (e.g., GPT-4o-mini, Qwen, Gemini).

Create a file named `.env` in the root directory of the repository.

Add your API key for the chosen model (e.g., OpenAI or DashScope) to the file. The `My Agents.py` file uses environment variables like `DASHSCOPE_API_KEY` or `OPENAI_API_KEY`.

Example `.env` content:
```
OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
# Or
# DASHSCOPE_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

## Dataset Preparation
Our experiments use the MULTIMODALQA and MANYMODALQA datasets.Create a root data directory, the datasets and structure the files to match the paths expected by the Dataloader.py and run_mt_script.py scripts.

## How to Run the Agent
The main execution script is Run MT Script.py. It uses multithreading for efficient evaluation and allows configuration via command-line arguments. Running MULTIMODALQARun the evaluation on the MultiModalQA benchmark: \
```python
python "run_mt_script.py" \
    --dataset_type "multimqa" \
    --dev_file "./data/MultiModalQA/endgame_dev_filtered_data.json" \
    --tables_file "./data/MultiModalQA/MMQA_tables.jsonl" \
    --texts_file "./data/MultiModalQA/MMQA_texts.jsonl" \
    --images_file "./data/MultiModalQA/MMQA_images.jsonl" \
    --images_base_url "./data/MultiModalQA/final_dataset_images" \
    --model "gpt-4o-mini" \
    --results_csv "multimqa_results.csv" \
    --num_iterations 100 \
    --num_threads 16
```
    
Running MANYMODALQA
Run the evaluation on the ManyModalQA benchmark:
```python
python "run_mt_script.py" \
    --dataset_type "manymqa" \
    --dev_file "./data/ManyModalQA/ManyModalQAData/official_aaai_split_dev_data.json" \
    --tables_file "./data/MultiModalQA/MMQA_tables.jsonl" \
    --texts_file "./data/MultiModalQA/MMQA_texts.jsonl" \
    --images_file "./data/MultiModalQA/MMQA_images.jsonl" \
    --images_base_url "./data/ManyModalQA/ManyModalImages" \
    --model "gpt-4o-mini" \
    --results_csv "manymqa_results.csv" \
    --num_iterations 100 \
    --num_threads 16
```

## 📖 Cite us:  
```bibtex
@misc{rajput2025rethinkinginformationsynthesismultimodal,
      title={Rethinking Information Synthesis in Multimodal Question Answering A Multi-Agent Perspective}, 
      author={Krishna Singh Rajput and Tejas Anvekar and Chitta Baral and Vivek Gupta},
      year={2025},
      eprint={2505.20816},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.20816}, 
}
```
