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



