<div align="center">

# Understanding and Harnessing Sparsity in Unified Multimodal Models

<p align="center">
  <a href="https://arxiv.org/abs/2512.02351"><img src="https://img.shields.io/badge/arXiv-2512.02351-b31b1b.svg?style=for-the-badge" alt="arXiv"></a>
  <a href="https://shwai-he.github.io/SparseUnifiedModel/"><img src="https://img.shields.io/badge/Project-Page-0d6b5d.svg?style=for-the-badge&logo=google-chrome&logoColor=white" alt="Project Page"></a>
  <a href="https://huggingface.co/LLM-Drop"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Checkpoints-FFD21E.svg?style=for-the-badge" alt="Hugging Face"></a>
  <a href="https://github.com/Shwai-He/SparseUnifiedModel"><img src="https://img.shields.io/badge/Task-Unified%20Multimodal-4F46E5.svg?style=for-the-badge" alt="Task"></a>
  <a href="https://github.com/Shwai-He/SparseUnifiedModel"><img src="https://img.shields.io/badge/Focus-Sparse%20Activation-10B981.svg?style=for-the-badge" alt="Focus"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10+-3776AB.svg?style=for-the-badge&logo=python&logoColor=white" alt="Python"></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-blue.svg?style=for-the-badge" alt="License"></a>
</p>

<p align="center">
  <strong><a href="https://shwai-he.github.io/">Shwai He</a><sup>1,2</sup></strong> &nbsp;&bull;&nbsp;
  <strong><a href="https://v3alab.github.io/author/chaorui-deng/">Chaorui Deng</a><sup>1</sup></strong> &nbsp;&bull;&nbsp;
  <strong><a href="https://www.ang-li.com/">Ang Li</a><sup>2</sup></strong> &nbsp;&bull;&nbsp;
  <strong><a href="https://shenyann.github.io/">Shen Yan</a><sup>1,&dagger;</sup></strong>
</p>

<p align="center">
  <sup>1</sup><strong>ByteDance Seed</strong> &nbsp;&nbsp;|&nbsp;&nbsp; <sup>2</sup><strong>University of Maryland, College Park</strong><br>
  <sup>&dagger;</sup><em>Corresponding author / Project Lead</em>
</p>

</div>

---

## 🧭 Navigation

- [📌 News & Highlights](#-news--highlights)
- [🔍 Core Insights & Key Findings](#-core-insights--key-findings)
- [⚙️ Method Overview & Sparsity Probing](#️-method-overview--sparsity-probing)
- [📊 Multimodal Benchmark Results](#-multimodal-benchmark-results)
- [🤗 MoE Adaptation Checkpoints](#-moe-adaptation-checkpoints)
- [📦 Installation](#-installation)
- [🧩 Supported Architectures](#-supported-architectures)
- [🚀 Quickstart & Inference](#-quickstart--inference)
- [🔬 Probing & Evaluation Workflows](#-probing--evaluation-workflows)
- [📂 Code Structure](#-code-structure)
- [📑 Citation & Contact](#-citation--contact)

---

## 📌 News & Highlights

- **[2025.12]** 📄 Paper released on arXiv: [arXiv:2512.02351](https://arxiv.org/abs/2512.02351)!
- **[2025.12]** 🌐 Interactive Project Page live at [shwai-he.github.io/SparseUnifiedModel](https://shwai-he.github.io/SparseUnifiedModel/) with an interactive multimodal sparsity visualizer!
- **[2025.12]** 🤗 Released MoE adaptation checkpoints for BAGEL (`BAGEL-MoE-7B-GEN-16to8` & `BAGEL-MoE-7B-GEN-32to16`) on [Hugging Face](https://huggingface.co/LLM-Drop).
- **[2025.12]** 🚀 Full codebase released supporting training-free depth/width probing and sparse MoE conversion for **BAGEL**, **Ming-Omni**, and **Qwen-Image**.

---

## 🔍 Core Insights & Key Findings

Unified multimodal models integrate **understanding** (e.g., visual question answering, reasoning, cross-modal retrieval) and **generation** (e.g., text-to-image synthesis, visual editing) into a single, cohesive architecture. However, unifying these distinct modalities introduces critical inference bottlenecks, compute imbalances, and parameter redundancies.

We conduct a systematic, training-free probing investigation across depth and width dimensions:

```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                                SYSTEMIC COMPRESSION ASYMMETRY                            │
├─────────────────────────────────────────────┬────────────────────────────────────────────┤
│         🧠 Understanding Components         │          🎨 Generation Components          │
├─────────────────────────────────────────────┼────────────────────────────────────────────┤
│ • High compressibility in generation tasks  │ • High sensitivity to compression          │
│ • 50%+ depth/width pruned with ~0% drop     │ • Severe quality collapse if pruned static │
│ • Serves as coarse high-level semantic prior│ • Requires high-precision continuous token │
│ • Tolerates aggressive layer dropping       │ • Best handled via dynamic sparse MoE      │
└─────────────────────────────────────────────┴────────────────────────────────────────────┘
```

1. **Understanding components are heavily compressible during generation:**
   - In generation tasks (e.g., text-to-image synthesis), understanding layers primarily extract coarse conditioning representations.
   - Up to **50% of understanding layers/neurons can be dropped** with negligible loss in image alignment and visual quality (GenEval drops < 1.2%).
2. **Generation components are exceptionally fragile:**
   - In contrast, generation layers model complex, fine-grained pixel distributions. Moderate pruning leads to rapid artifacts, semantic distortion, and image quality collapse.
3. **Dynamic Sparsity Motivates Sparse MoE Adaptation:**
   - Inspection of neuron activation distributions reveals sample-dependent, input-specific activation patterns across generation layers.
   - Rather than static pruning, we convert dense MLP layers into **Mixture-of-Experts (MoE)** and sparsely activate them (e.g., top-8 out of 16 experts, or top-16 out of 32 experts).
   - **Result:** Halves active generation FLOPs while fully preserving dense generation quality!

<p align="center">
  <img src="docs/static/images/efficient_ug.svg" alt="Two-stage Efficiency Optimization" width="85%">
  <br>
  <em>Figure 1: Two-stage optimization pipeline: (1) Training-free component probing across depth and width, followed by (2) Sparse MoE adaptation to recover generation fidelity with 50% active parameters.</em>
</p>

---

## ⚙️ Method Overview & Sparsity Probing

Our framework provides a unified pipeline for probing and exploiting multimodal sparsity:

```
  Unified Multimodal Model
            │
            ├───► 1. Depth Probing (Layer Dropping via Cosine Similarity / Output Impact)
            │         └── Identify redundant transformer layers across tasks
            │
            ├───► 2. Width Probing (Neuron Activation Calibration & Partitioning)
            │         └── Measure activation frequency across calibration datasets
            │
            └───► 3. MoE Adaptation (Dense MLP ──► Sparse Expert Routing)
                      └── Partition weight matrices into N experts and route Top-K dynamically
```

### Probing Dimensions

| Dimension | Mechanism | Metric / Criteria | Target Components |
|---|---|---|---|
| **Depth Pruning** | Layer Dropping | Activation Cosine Similarity $\cos(\mathbf{h}_l, \mathbf{h}_{l-1})$ & Feature Distance | Attention blocks, FFN blocks, whole layers |
| **Width Reduction** | Neuron Pruning | Empirical Activation Frequency $\|a_i\|_1$ on calibration prompts | Intermediate MLP dimensions / FFN channels |
| **MoE Adaptation** | Expert Slicing + Router | Dynamic top-$k$ routing over partitioned weight shards | Generation MLP modules (16to8, 32to16) |

---

## 📊 Multimodal Benchmark Results

### 1. Visual Generation Benchmarks (GenEval, DPG-Bench, ImageReward)

Evaluated on **BAGEL-7B** and its compressed / MoE variants:

| Method | Active Params / Ratio | GenEval Overall ↑ | GenEval Single Obj ↑ | GenEval Two Obj ↑ | GenEval Color ↑ | GenEval Position ↑ | DPG-Bench ↑ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Dense Full Model** | 7.0B (100%) | **0.652** | 0.981 | 0.742 | 0.812 | 0.461 | 82.4 |
| Layer Drop (Und-50%) | 5.2B (74%) | 0.648 | 0.978 | 0.739 | 0.809 | 0.457 | 81.9 |
| Width Reduction (Und-50%) | 5.2B (74%) | 0.645 | 0.976 | 0.735 | 0.805 | 0.453 | 81.7 |
| Width Reduction (Gen-50%) | 5.2B (74%) | 0.518 | 0.892 | 0.584 | 0.671 | 0.312 | 68.3 |
| **BAGEL-MoE (16 → 8)** | **3.8B (54%)** | **0.649** | **0.980** | **0.740** | **0.810** | **0.459** | **82.1** |
| **BAGEL-MoE (32 → 16)** | **3.8B (54%)** | **0.651** | **0.981** | **0.741** | **0.811** | **0.460** | **82.3** |

> **Key Takeaway:** Static generation pruning drops GenEval from **0.652 → 0.518**, while **BAGEL-MoE recovers it to 0.651** with only 54% active compute!

---

### 2. Visual Understanding Benchmarks (MME, MMBench, POPE, TextVQA)

| Model Variant | Active Params | MME Total ↑ | MMBench ↑ | POPE (F1) ↑ | TextVQA ↑ | SEED-Bench ↑ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **BAGEL Dense Baseline** | 7.0B | **1942.3** | **78.4** | **88.6** | **68.2** | **72.5** |
| Und-Layer Drop (20% Pruned) | 5.9B | 1918.5 | 77.2 | 88.1 | 67.4 | 71.8 |
| Und-Layer Drop (40% Pruned) | 4.9B | 1856.1 | 74.8 | 86.9 | 65.1 | 69.4 |
| Und-Width Reduction (20%) | 5.9B | 1925.7 | 77.6 | 88.3 | 67.8 | 72.0 |
| Und-Width Reduction (40%) | 4.9B | 1868.2 | 75.3 | 87.2 | 65.8 | 70.1 |

---

## 🤗 MoE Adaptation Checkpoints

Pretrained sparse MoE checkpoints are hosted on Hugging Face:

| Checkpoint Name | Base Architecture | Total Experts | Active Experts | Sparsity Ratio | Hugging Face Repository |
|---|---|:---:|:---:|:---:|---|
| `BAGEL-MoE-7B-GEN-16to8` | BAGEL-7B | 16 | 8 | 50% Active | [🤗 LLM-Drop/BAGEL-MoE-7B-GEN-16to8](https://huggingface.co/LLM-Drop/BAGEL-MoE-7B-GEN-16to8) |
| `BAGEL-MoE-7B-GEN-32to16` | BAGEL-7B | 32 | 16 | 50% Active | [🤗 LLM-Drop/BAGEL-MoE-7B-GEN-32to16](https://huggingface.co/LLM-Drop/BAGEL-MoE-7B-GEN-32to16) |

---

## 📦 Installation

### 1. Create and Activate Environment

```bash
# Create conda environment
conda create -n sparse_um python=3.10 -y
conda activate sparse_um

# Install PyTorch (CUDA 12.1+ recommended)
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

# Install requirements
pip install -r requirements.txt

# Install FlashAttention-2 (optional but recommended for speed)
pip install flash-attn --no-build-isolation
```

---

## 🧩 Supported Architectures

We provide unified compression hooks via `modeling/compression_mixin.py` across:

| Architecture | Paradigm | Key Components | Implementation Directory |
|---|---|---|---|
| **BAGEL** | Decoder-Only Unified LLM | Qwen2 LLM backbone + SigLIP + Diffusion VAE | `modeling/bagel/` |
| **Ming-Omni** | MoE Multimodal LLM | Sparse MoE routing + Cross-modal projector | `modeling/ming/` |
| **Qwen-Image** | Encoder + Diffusion Decoder | Qwen2.5-VL text encoder + Continuous Diffusion | `modeling/qwen/` & `modeling/diffusers/` |

---

## 🚀 Quickstart & Inference

### 1. Interleaved Generation & Understanding with BAGEL

```python
import torch
from inferencer import InterleaveInferencer
from modeling.bagel import Bagel, BagelConfig
from modeling.autoencoder import load_ae
from modeling.qwen2 import Qwen2Tokenizer

# 1. Load model and tokenizer
model_path = "LLM-Drop/BAGEL-MoE-7B-GEN-16to8"
tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
model = Bagel.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
vae_model = load_ae("vae_path")

inferencer = InterleaveInferencer(
    model=model,
    vae_model=vae_model,
    tokenizer=tokenizer,
    vae_transform=None,
    vit_transform=None,
    new_token_ids={}
)

# 2. Perform text-to-image synthesis
context = inferencer.init_gen_context()
inferencer.update_context_text("A cinematic shot of a red vintage sports car driving through a misty neon-lit forest at dusk.", context)
images = inferencer.generate_image(context, cfg_text_scale=4.0)
images[0].save("output.png")
```

---

## 🔬 Probing & Evaluation Workflows

### 1. Depth Pruning Probing (Layer Dropping)

Run Layer Dropping evaluation on GenEval for BAGEL, Ming-Omni, or Qwen:

```bash
# Generation task (GenEval benchmark)
bash scripts/eval/bagel/run_geneval_ld.sh
bash scripts/eval/ming/run_geneval_ld.sh
bash scripts/eval/qwen/run_geneval_ld.sh

# Understanding task (MME benchmark)
bash scripts/eval/bagel/run_vlm_ld.sh
bash scripts/eval/ming/run_vlm_ld.sh
```

### 2. Width Reduction Probing (Neuron Pruning)

Profile and prune intermediate neuron activations based on calibration samples:

```bash
# 1. Calculate neuron importance scores and partition masks
python scripts/neuron_partition.py   --model_path /path/to/bagel   --calibration_task understanding   --calibration_samples 64

# 2. Run width reduction evaluation
bash scripts/eval/bagel/run_geneval_wr.sh
bash scripts/eval/bagel/run_vlm_wr.sh
```

### 3. Converting Dense Models to Sparse MoE

Follow our interactive step-by-step Jupyter Notebook:

```bash
jupyter notebook notebooks/dense2sparse.ipynb
```

Or convert programmatically by specifying expert partitions:
```python
# Partition dense intermediate MLP weights into 16 or 32 sparse expert slices
from utils.moe_utils import convert_dense_to_sparse_moe

sparse_model = convert_dense_to_sparse_moe(
    dense_model=model,
    num_experts=16,
    active_experts=8,
    target_modules=["mlp.down_proj", "mlp.gate_up_proj"]
)
```

---

## 📂 Code Structure

```
SparseUnifiedModel/
├── modeling/                   # Core multimodal model definitions
│   ├── bagel/                  # BAGEL unified multimodal LLM
│   ├── ming/                   # Ming-Omni MoE multimodal architecture
│   ├── qwen/                   # Qwen-Image text encoder & vision stack
│   ├── qwen2/                  # Base Qwen2 language model components
│   ├── siglip/                 # SigLIP vision transformer encoder
│   ├── diffusers/              # Diffusion pipelines and schedulers
│   └── compression_mixin.py    # Layer dropping and pruning instrumentation
│
├── eval/                       # Comprehensive evaluation suites
│   ├── gen/                    # Generation benchmarks (GenEval, DPG-Bench)
│   │   ├── gen_images.py       # Width reduction image generation runner
│   │   ├── gen_images_ld.py    # Layer drop image generation runner
│   │   ├── compress_utils.py   # Shared pruning utilities and hooks
│   │   └── geneval/            # GenEval scoring and prompt datasets
│   └── vlm/                    # Multimodal understanding benchmarks
│       └── eval/               # MME, MMBench, POPE, MMMU, MathVista, etc.
│
├── scripts/                    # Execution and evaluation scripts
│   ├── eval/                   # Benchmark runner bash scripts
│   │   ├── bagel/              # BAGEL evaluation scripts (LD, WR, Baseline)
│   │   ├── ming/               # Ming-Omni evaluation scripts
│   │   └── qwen/               # Qwen-Image evaluation scripts
│   ├── inferencer.py           # Interleaved inference engine
│   └── neuron_partition.py     # Neuron importance profiling and partitioner
│
├── tools/                      # Utilities and deployment integrations
│   ├── ming_sdk/               # Ming SDK client
│   ├── vllm/                   # vLLM integration patches
│   └── gradio_demo.py          # Interactive web UI demo
│
├── notebooks/                  # Interactive exploration and walkthroughs
│   ├── dense2sparse.ipynb      # Step-by-step Dense -> Sparse MoE conversion
│   ├── inference_bagel.ipynb   # BAGEL interactive inference demo
│   ├── inference_qwen.ipynb    # Qwen-Image interactive inference demo
│   └── inference_ming.ipynb    # Ming-Omni interactive inference demo
│
├── docs/                       # Project website & visual assets
│   ├── index.html              # Modern academic project webpage
│   └── static/images/          # Figures, architecture charts, and plots
├── requirements.txt            # Python dependencies
└── LICENSE                     # Apache-2.0 License
```

---

## 📑 Citation & Contact

If you find this work or codebase helpful in your research, please cite our paper:

```bibtex
@misc{he2025understandingharnessingsparsityunified,
  title={Understanding and Harnessing Sparsity in Unified Multimodal Models},
  author={Shwai He and Chaorui Deng and Ang Li and Shen Yan},
  year={2025},
  eprint={2512.02351},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2512.02351},
}
```

### Contact & Inquiries
- **Shwai He**: `shwai.he@bytedance.com`
- **Shen Yan**: `sheny@bytedance.com`

---

<p align="center">
  Released under the <a href="./LICENSE">Apache 2.0 License</a>.
</p>
