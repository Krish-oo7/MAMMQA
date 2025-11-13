# MAMMQA: Multi-Agent Framework for Multimodal Question Answering  
**Paper:** *Rethinking Information Synthesis in Multimodal Question Answering: A Multi-Agent Perspective*  
**Authors:** Krishna Singh Rajput, Tejas Anvekar, Chitta Baral, Vivek Gupta  
Arizona State University  

---

## Overview

This repository contains the official implementation of **MAMMQA**, a **prompt-driven, multi-agent system** for multimodal question answering (MMQA).  
The framework decomposes reasoning across **three interpretable agents**:

1. **Modality Expert Agents** — extract insights from text, tables, or images.  
2. **Cross-Modality Synthesis Agents** — integrate these insights to form cross-modal reasoning chains.  
3. **Aggregator Agent** — synthesizes multiple agent outputs into a final, evidence-grounded answer.

Unlike traditional monolithic or fine-tuned MMQA models, MAMMQA is **zero-shot**, modular, and **LLM-agnostic**, compatible with both **OpenAI GPT-4o**, **Gemini 1.5-Flash**, and **Qwen2.5-VL** models.

---

## Dataset Preparation
Our experiments use the MULTIMODALQA and MANYMODALQA datasets.Create a root data directory, e.g., data/.Download the datasets and structure the files to match the paths expected by the Dataloader.py and Run MT Script.py scripts.The expected directory structure should be:data/
├── MultiModalQA/
│   ├── endgame_dev_filtered_data.json
│   ├── MMQA_tables.jsonl
│   ├── MMQA_texts.jsonl
│   ├── MMQA_images.jsonl
│   └── final_dataset_images/  # Directory containing all image files
└── ManyModalQA/
    └── ManyModalImages/     # Directory containing all image files for ManyModalQA

## How to Run the Agent
The main execution script is Run MT Script.py. It uses multithreading for efficient evaluation and allows configuration via command-line arguments.Running MULTIMODALQARun the evaluation on the 

MultiModalQA benchmark:Bash
python "Run MT Script.py" \
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
    
Running MANYMODALQA
Run the evaluation on the ManyModalQA benchmark.Bash
python "Run MT Script.py" \
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
