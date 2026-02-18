# Understanding and Harnessing Sparsity in Unified Multimodal Models

---

[![Task](https://img.shields.io/badge/Task-Unified%20Multimodal-blue)](#)
[![Focus](https://img.shields.io/badge/Focus-Sparse%20Activation-green)](#)
[![Python](https://img.shields.io/badge/Python-3.10+-brightgreen)](#)

This repository contains the code and experiments for **Efficient Unified Multimodal Modeling (Efficient-UG)**, a study on redundancy and dynamic sparsity in unified models that jointly support multimodal **understanding** and **generation**.

<p align="center">
  <img src="efficient_ug.svg" alt="Efficient-UG overview" width="72%">
</p>

## ⚡ TL;DR
- Unified multimodal models show strong task-dependent redundancy across understanding and generation paths.
- Generation modules are more compression-sensitive than understanding modules.
- Sparse expert activation (MoE-style adaptation) recovers generation quality with lower active parameters.
- The resulting BAGEL variant keeps competitive performance while activating roughly half of parameters.

## 🔍 Overview
Unified multimodal models promise one architecture for reasoning and content generation, but this unification introduces non-uniform compute demand across tasks and samples.

Efficient-UG analyzes these inefficiencies through training-free probing and sparse adaptation, covering:
1. **Depth Pruning** (layer dropping),
2. **Width Reduction** (neuron partitioning),
3. **Expert Partitioning** for sparse MoE preparation.

## 📰 News
- Feb 2026: README reorganized with a cleaner research-repo layout and command flow.

## ✨ Why This Repo
This codebase unifies and adapts model components from:
- [BAGEL](https://github.com/ByteDance-Seed/Bagel)
- [Ming-Omni](https://github.com/inclusionAI/Ming/tree/main)
- [Qwen-Image](https://github.com/QwenLM/Qwen-Image)

Key adapted entry files:
- `modeling/bagel/bagel.py`
- `Ming/modeling_bailingmm.py`
- `diffusers/pipelines/qwenimage/modeling_qwen2_5_vl.py`

These adaptations provide consistent layer/dimension interfaces for systematic pruning and sparse computation studies.

## 📦 Installation
```bash
conda create -n efficient_ug python=3.10 -y
conda activate efficient_ug

pip install -r requirements.txt
```

## 🚀 Quick Start
### 1) Depth Pruning Evaluation
Understanding:
```bash
bash eval/vlm/evaluate_ld.sh
```

Generation:
```bash
bash scripts/eval/bagel/run_geneval_ld.sh
bash scripts/eval/ming/run_geneval_ld.sh
bash scripts/eval/qwen/run_geneval_ld.sh
```

### 2) Width Reduction Evaluation
Understanding:
```bash
bash eval/vlm/evaluate_wr.sh
```

Generation:
```bash
bash scripts/eval/bagel/run_geneval_wr.sh
bash scripts/eval/ming/run_geneval_wr.sh
bash scripts/eval/qwen/run_geneval_wr.sh
```

### 3) Neuron Partitioning Example
```bash
python neuron_partition.py
```

### 4) Dense-to-Sparse Expert Conversion
Use:
- `dense2sparse.ipynb`

This notebook demonstrates converting dense generation modules into sparse expert-style structures for adaptive activation.

## 🧠 Core Methods
1. **Depth Pruning via Layer Dropping**
- Reduces inference depth while preserving multimodal understanding quality as much as possible.

2. **Width Reduction via Neuron Partitioning**
- Identifies and prunes less active neurons for task-specific compactness.

3. **Expert Partitioning for MoE Preparation**
- Splits generation modules into experts for sparse activation and later expert-based adaptation.

## 🗂️ Repository Layout
```text
SparseUnifiedModel/
├── modeling/                      # Core model definitions (BAGEL, Ming-Omni, Qwen-Image)
│   └── bagel/
├── Ming/                          # Ming-Omni related modeling files
│   └── modeling_bailingmm.py
├── diffusers/                     # Adapted Qwen-Image modules and pipelines
│   └── pipelines/qwenimage/
├── data/                          # Data preprocessing utilities
├── eval/                          # Evaluation for understanding and generation
│   └── vlm/
├── scripts/                       # Shell launchers for different models/tasks
│   └── eval/
├── utils/                         # Shared utility functions
├── dense2sparse.ipynb             # Dense-to-sparse MoE preparation demo
├── neuron_partition.py            # Neuron importance analysis and partitioning
├── inference.ipynb                # Inference/pruning walkthrough
├── inferencer.py                  # Unified inference interface
├── efficient_ug.svg               # Project figure
├── prompts.txt                    # Example prompts
├── requirements.txt
└── README.md
```

## 📄 Citation
If you use this repository in your research, please cite the associated paper when available.

## 📬 Contact
- `shwai.he@bytedance.com`
- `sheny@bytedance.com`
