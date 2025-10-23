# Understanding Slimness and Sparsity in Unified Multimodal Models: An Empirical Study

This repository contains the code and experiments for our work on **Efficient Unified Multimodal Modeling**, which studies redundancy and dynamic sparsity in unified models that jointly perform multimodal understanding and generation. The project analyzes how compression and adaptive computation can improve scalability and efficiency in unified multimodal architectures.

---

## 🔍 Overview

Unified multimodal models aim to integrate understanding (e.g., reasoning, classification) and generation (e.g., text-to-image synthesis, captioning) within a single architecture.
While this unification brings the promise of general-purpose multimodal intelligence, it also introduces inference inefficiencies due to task-specific activation, compute imbalance, and input variability.
Despite the recent progress, a systematic understanding of where and how these inefficiencies arise across different components remains limited.

This project, Efficient-UG, conducts a comprehensive analysis of unified multimodal models using training-free pruning as a probing methodology, covering both depth pruning and width reduction.
Our study finds that:

- The understanding components—though crucial for reasoning—can be substantially compressed in generation tasks without severe degradation.

- The generation components, however, are highly sensitive to compression, with performance dropping sharply even under moderate pruning.

- To address this imbalance, we introduce Mixture-of-Experts (MoE) Adaptation, inspired by dynamic neuron activation patterns across samples.
This approach partitions the generation module into multiple experts and activates them sparsely to recover performance.
Through expert-frozen tuning and fully trainable adaptation, we show that sparse activation restores generation quality while maintaining efficiency.

As a result, our BAGEL model achieves comparable performance to the full model while activating only about half of its parameters, offering new insights into efficient unified multimodal modeling.

![Diagram of Efficient UG](efficient_ug.svg)

---

## 🧩 Modeling Files

This repository integrates and adapts modeling files from [**BAGEL**](https://github.com/ByteDance-Seed/Bagel), [**Ming-Omni**](https://github.com/inclusionAI/Ming/tree/main), and [**Qwen-Image**](https://github.com/QwenLM/Qwen-Image) for unified multimodal experimentation.  
Each model retains its original implementation style, while we introduce targeted modifications to ensure **compatibility**, **efficiency**, and enable **depth pruning** and **width reduction** within a unified compression framework.

The corresponding modified files are listed below:

- **BAGEL** → `modeling/bagel/bagel.py`  
- **Ming-Omni** → `Ming/modeling_bailingmm.py`  
- **Qwen-Image** → `diffusers/pipelines/qwenimage/modeling_qwen2_5_vl.py`  

These adaptations provide consistent layer and dimension interfaces across heterogeneous architectures, allowing fine-grained control of model components during pruning and compression analysis.


---





## 📂 Code Structure

