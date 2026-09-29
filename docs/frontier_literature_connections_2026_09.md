# 👁️ SparseUnifiedModel & VLM-Compression: 每日前沿文献关联与多模态统一稀疏/流加速落地库 (2026-09)

**Document ID:** `SPARSEUMM-LIT-202609` | **Last Updated:** `2026-09-29` | **Target Path:** `docs/frontier_literature_connections_2026_09.md` | **Total Routed Papers:** `24`

> [!IMPORTANT]
> **🔗 跨仓库文献引用链闭环 (Cross-Repository Reference Chain Closure)**
> 本文件由每日 AI 前沿论文精读流水线自动路由生成，专门收录与我们 **TMLR 2026 / ICML 2025 / ECCV 2026 多模态代表作 (*Sparsity for Unified Multimodal Models*, `Shwai-He/SparseUnifiedModel` & `Shwai-He/VLM-Compression`)** 直接关联的跨模态视觉/文本 Token 剪枝（`CoverPruner`, `SFPruner`, `CLSE`, `ASL`, `AnchorPrune`, `LearnPruner`, `VLA-Pruner`, `RT-VLA`）、多模态 KV 缓存压缩（`LightKV`, `RotateK`, `MixKV`, `VestigeKV`）以及多模态流匹配/单步生成加速（`DEE-VLA`, `Flow3D-OPD`, `IMLE-VLA`, `SnapFlow`, `Flow-OPD`, `Self-OPD`, `MoE-FM`）最新 arXiv 论文笔记。
> 每一篇收录文献均包含：**核心痛点、底层数学公式、ASCII 架构图、关键实测指标**，以及**与 `SparseUnifiedModel` 仓库具体代码模块和我们已发表代表作（Our Works）的双向锚定**。

---

## 🌟 1. 核心关联文献与本仓库模块映射速查表 (Executive Reference-to-Module Matrix)

| 收录日期 | 论文标题与 arXiv 链接 | 关键实测收益 / 核心结论 | 锚定本仓库代码模块与文档路径 (`Target Module`) | 原始精读归档 |
| :---: | :--- | :--- | :--- | :---: |
| `2026-09-29` | [**✂️ CoverPruner & SFPruner**](https://arxiv.org/abs/2609.03158) (`arXiv:2609.03158`) | 在 LLaVA-NeXT、Qwen2.5-VL 与 InternVL-2.5 等高分辨率多模态模型上，当剪除 **80%–88.9% 视觉 Token**（仅保留 64–128 个 Token）时，`CoverPruner` 与... | `sparse_umm/token_pruning.py` (k-Medoids Facility Location Coverage & Barycentric Surrogate Folding) | [2026-09-29](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-29_ai_paper_notes.md) |
| `2026-09-29` | [**⚡ VestigeKV**](https://arxiv.org/abs/2609.03949) (`arXiv:2609.03949`) | 在基于 MLA 架构的长上下文大模型上（128K–256K 上下文长度），`VestigeKV` 无需任何重新训练或旁路预测器，在仅加载 **15%–20% KV 潜向量**的稀疏注意力预算下，在 RULER、LongBench... | `sparse_umm/kv_compression.py` (Zero-Overhead Vestigial Branch Sparse-Attention KV Indexing) | [2026-09-29](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-29_ai_paper_notes.md) |
| `2026-09-29` | [**🦾 DEE-VLA**](https://arxiv.org/abs/2609.29382) (`arXiv:2609.29382`) | 在 LIBERO（Spatial / Object / Goal / Long）与真机双臂灵巧操作任务上，`DEE-VLA` 在成功率与全深度 10-NFE 基线持平（甚至因减少自由空间过拟合而提升 **+0.8%**）的同时，平... | `sparse_umm/flow_generation.py` (Decoupled Early-Exit Compute Allocation across VLM & Flow Experts) | [2026-09-29](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-29_ai_paper_notes.md) |
| `2026-09-29` | [**🌍 WM2VLA & InternW0-Δ**](https://arxiv.org/abs/2609.24682) (`arXiv:2609.24682`) | 在 RoboTwin、CALVIN、SIMPLER 及多项真机长程操作基准上，`WM2VLA` 与 `InternW0-Δ` 相较于无世界模型先验的同规模 VLA 策略，平均任务成功率大幅跃升 **+11.5%–+16.8%**（... | `sparse_umm/` (`Shwai-He/SparseUnifiedModel` & `Shwai-He/VLM-Compression`) | [2026-09-29](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-29_ai_paper_notes.md) |
| `2026-09-29` | [**🧬 Failure-RSI & Flow3D-OPD**](https://arxiv.org/abs/2606.31270) (`arXiv:2606.31270`) | **`Failure-RSI`**：在 OSWorld 与多模态计算机操作基准上，仅利用推理期失败轨迹自动合成工具与控制补丁，无需微调底层大模型权重即可将任务成功率相对提升 **+24.6%**，且合成的代码补丁具备跨任务泛化性。 | `sparse_umm/flow_generation.py` (Multi-Teacher On-Policy Flow Trajectory Distillation) | [2026-09-29](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-29_ai_paper_notes.md) |
| `2026-09-28` | [**✂️ CLSE**](https://arxiv.org/abs/2606.24165) (`arXiv:2606.24165`) | 在 LLaVA-NeXT-7B/13B 与 Qwen2.5-VL-7B 上，免训练剪除 **75% 视觉 Token** 时，在 DocVQA、ChartQA、TextVQA 与 Video-MME 等 10 项多模态基准上平均保... | `sparse_umm/token_pruning.py` (Cross-Layer Spectral Entropy Evolution & Subspace Orthogonal Deduplication) | [2026-09-28](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-28_ai_paper_notes.md) |
| `2026-09-28` | [**✂️ ASL**](https://arxiv.org/abs/2601.07667) (`arXiv:2601.07667`) | 在 Llama-3.1-8B/70B 与 Qwen2.5-14B 上，针对 RULER、InfiniteBench 与 Needle-in-a-Haystack（128K 上下文）评测表明：在相同的... | `sparse_umm/token_pruning.py` (Marginal Information Gain Adaptive Layer Selection) | [2026-09-28](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-28_ai_paper_notes.md) |
| `2026-09-28` | [**🦾 IMLE-VLA**](https://arxiv.org/abs/2609.04369) (`arXiv:2609.04369`) | 在 LIBERO、SimplerEnv 与真实双臂灵巧操作任务上，IMLE-VLA 以 **1-NFE 单步前向** 将动作生成吞吐量与控制频率提升 **6.4×–8.5×**，同时任务成功率不仅远超单步回归基线（+14.2%）... | `sparse_umm/flow_generation.py` (Single-Step IMLE Mode-Covering Action/Visual Generation) | [2026-09-28](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-28_ai_paper_notes.md) |
| `2026-09-27` | [**OBCache**](https://arxiv.org/abs/2510.07651) (`arXiv:2510.07651`) | **即插即用全面提升主流基线**：在 **Llama-3.1-8B-Instruct**、**Qwen-2.5-7B/14B-Instruct** 与 **Mistral-7B** 上，将 OBCache 的... | `sparse_umm/` (`Shwai-He/SparseUnifiedModel` & `Shwai-He/VLM-Compression`) | [2026-09-27](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-27_ai_paper_notes.md) |
| `2026-09-27` | [**Code as Worlds**](https://arxiv.org/abs/2608.27549) (`arXiv:2608.27549`) | **定量物理推理与反事实预测大幅领先**：在涵盖刚体碰撞、流体倾倒、多摆耦合及遮挡轨迹预测的物理推理基准（PhysBench、CLEVRER、ComPhy）上，**Code as Worlds** 将开源与闭源顶级 VLM 的定量... | `sparse_umm/` (`Shwai-He/SparseUnifiedModel` & `Shwai-He/VLM-Compression`) | [2026-09-27](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-27_ai_paper_notes.md) |
| `2026-09-26` | [**🤖 VLA-Pruner**](https://arxiv.org/abs/2511.16449) (`arXiv:2511.16449`) | 在 OpenVLA 与主流机器人操控基准（LIBERO-Spatial / Object / Goal / Long）上，剔除 **50%–75% 视觉 Token** 仍保持与全量 Token 持平的任务成功率，端到端控制频率显... | `sparse_umm/token_pruning.py` (Temporal Motion Prompt + Action-Aware Token Pruning) | [2026-09-26](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-26_ai_paper_notes.md) |
| `2026-09-25` | [**Fully Looped Transformer**](https://arxiv.org/abs/2605.18797) (`arXiv:2605.18797`) | 在完全不增加任何额外参数（0 Extra Parameters）的条件下，Fully Looped Transformer 在 $K=8, 12$ 步循环预训练中完全消除了传统 Looped Transformer 的梯度尖峰（G... | `sparse_umm/` (`Shwai-He/SparseUnifiedModel` & `Shwai-He/VLM-Compression`) | [2026-09-25](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-25_ai_paper_notes.md) |
| `2026-09-24` | [**LearnPruner**](https://arxiv.org/abs/2604.23950) (`arXiv:2604.23950`) | 在 **LLaVA-1.5/NeXT** 与 **Qwen2-VL** 上，LearnPruner 仅保留 **11.1%–16.7% 视觉 Token**，FLOPs 降低 **68%**，在 10 项多模态基准上的平均精度达到... | `sparse_umm/token_pruning.py` (Two-Stage Visual Deduplication + Differentiable Mask) | [2026-09-24](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-24_ai_paper_notes.md) |
| `2026-09-24` | [**MixKV**](https://arxiv.org/abs/2510.20707) (`arXiv:2510.20707`) | 在 **MileBench**、**Video-MME** 与多图长上下文评测中，MixKV 在 **10% 极限缓存预算**下比 SnapKV 与 PyramidKV 平均提升 **`+5.3%`**。 | `sparse_umm/kv_compression.py` (Modality-Adaptive Importance × Diversity KV Eviction) | [2026-09-24](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-24_ai_paper_notes.md) |
| `2026-09-24` | [**AEWM**](https://arxiv.org/abs/2609.28416) (`arXiv:2609.28416`) | 在 **VisualWebArena**、**OSWorld** 与长程具身任务上，AEWM 将不可逆错误操作率降低 **52%**，端到端任务成功率比无状态编辑的 Tree-of-Thoughts 高出... | `sparse_umm/` (`Shwai-He/SparseUnifiedModel` & `Shwai-He/VLM-Compression`) | [2026-09-24](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-24_ai_paper_notes.md) |
| `2026-09-23` | [**RT-VLA**](https://arxiv.org/abs/2606.14010) (`arXiv:2606.14010`) | 在机器人操作基准上，RT-VLA 将纯视觉模式下的编码与推理耗时降低 **44.8x**，端到端帧率突破 **60 Hz**，同时保留了 7B 教师模型 **96% 以上** 的任务成功率。 | `sparse_umm/` (`Shwai-He/SparseUnifiedModel` & `Shwai-He/VLM-Compression`) | [2026-09-23](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-23_ai_paper_notes.md) |
| `2026-09-23` | [**HiMoE-VLA**](https://arxiv.org/abs/2512.05693) (`arXiv:2512.05693`) | 在跨 50+ 任务的 Open-X Embodiment 与仿真套件上，HiMoE-VLA 比同激活参数量的稠密 VLA 与单层 MoE-VLA 平均成功率提升 **`+8.7%`**。 | `sparse_umm/` (`Shwai-He/SparseUnifiedModel` & `Shwai-He/VLM-Compression`) | [2026-09-23](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-23_ai_paper_notes.md) |
| `2026-09-22` | [**SnapFlow**](https://arxiv.org/abs/2604.05656) (`arXiv:2604.05656`) | 在 **LIBERO**（Spatial / Object / Goal / Long）与真实机械臂双臂操作基准上，SnapFlow 将动作专家推理步数从 10 NFE 压缩至 **1 NFE**，动作生成阶段延迟降低... | `sparse_umm/` (`Shwai-He/SparseUnifiedModel` & `Shwai-He/VLM-Compression`) | [2026-09-22](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-22_ai_paper_notes.md) |
| `2026-09-22` | [**LightKV**](https://arxiv.org/abs/2605.00789) (`arXiv:2605.00789`) | 在 **LLaVA-1.6-34B** 与 **InternVL-2** 上将视觉 KV 缓存直接压缩 **50%–75%**，在 TextVQA、DocVQA 与计数基准上实现 **99.4%** 的原始性能保持率。 | `sparse_umm/kv_compression.py` (Text-Guided Bipartite Soft-Merging of Visual KV) | [2026-09-22](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-22_ai_paper_notes.md) |
| `2026-09-21` | [**RotateK**](https://arxiv.org/abs/2605.19218) (`arXiv:2605.19218`) | 在 **LLaVA-NeXT**、**Qwen2-VL-7B** 与 **InternVL-2** 上，RotateK 剪除 **50%–60% 的 Key 通道**而无需微调，且与视觉 Token 剪枝（如 FastV / VL... | `sparse_umm/kv_compression.py` (RoPE-Compatible Block-Orthogonal Key Channel Truncation) | [2026-09-21](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-21_ai_paper_notes.md) |
| `2026-09-21` | [**Self-OPD**](https://arxiv.org/abs/2608.26872) (`arXiv:2608.26872`) | 在不加载任何外部教师的情况下，Self-OPD 将 4 步流匹配模型的生成与控制成功率提升 **`+14.2%`**，甚至超越了 50 步原始基准模型。 | `sparse_umm/flow_generation.py` (On-Policy Trajectory Distillation for Multimodal Flow) | [2026-09-21](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-21_ai_paper_notes.md) |
| `2026-09-21` | [**MoE-FM**](https://arxiv.org/abs/2604.15009) (`arXiv:2604.15009`) | 在潜空间语言生成与多模态推理中，MoE-FM 在仅使用 **2–4 步 NFE** 时即可达到单稠密流模型 16–32 步的生成质量，推理延迟降低 **3.8x**。 | `sparse_umm/flow_generation.py` (Mixture-of-Flows Piecewise Straight Velocity Fields) | [2026-09-21](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-21_ai_paper_notes.md) |
| `2026-09-20` | [**Flow-OPD**](https://arxiv.org/abs/2605.08063) (`arXiv:2605.08063`) | 在 2 步与 4 步流匹配生成基准上，Flow-OPD 将 FID 与条件指令遵循得分相比离线轨迹蒸馏（Reflow / Progressive Distillation）提升 **`18%–27%`**。 | `sparse_umm/flow_generation.py` (On-Policy Trajectory Distillation for Multimodal Flow) | [2026-09-20](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-20_ai_paper_notes.md) |
| `2026-09-18` | [**✂️ AnchorPrune**](https://arxiv.org/abs/2609.08842) (`arXiv:2609.08842`) | **评估模型**：Qwen2-VL-7B/72B、LLaVA-NeXT-34B； | `sparse_umm/token_pruning.py` (Cross-Modal Geometric Anchor Fidelity Pruning) | [2026-09-18](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-18_ai_paper_notes.md) |

---

## 📐 2. 逐篇论文深度机制解构、数学公式与本仓库落地指南 (Per-Paper Deep-Dive Cards)

### 2.1 [2026-09-29] ✂️ *CoverPruner & SFPruner: Who Speaks for the Pruned? Visual Token Pruning as Coverage Optimization & Single-Forward Ridge Leverage*
> 🏷️ **核心关键词**：Visual Token Pruning · Representational Coverage Maximization (RCM) · Ridge Leverage Score · High-Resolution MLLMs  
> 🔗 **arXiv 链接**：[`arXiv:2609.03158`](https://arxiv.org/abs/2609.03158) (`CoverPruner`) & [`arXiv:2607.23046`](https://arxiv.org/abs/2607.23046) (`SFPruner`)

```
  高分辨率视觉 Token 集合 V = {v_1..v_N}
       ├──► [ CoverPruner (2609.03158) ] : 语义显著性 q_i × 设施选址最大覆盖 max_{j ∈ S} sim(v_i, v_j) ──► 确保每个被剪 Token 有高相似“代言人”
       └──► [ SFPruner    (2607.23046) ] : 语义引导核矩阵岭杠杆分数 τ_i(λ) + 单次方向性排序掩码    ──► 零迭代 O(N d^2) 单次前向去冗余保留子集 S*
```

#### 🎯 背景与痛点 (Problem Statement)
现有视觉语言大模型（VLM / MLLM）的视觉 Token 剪枝算法大多遵循“独立显著性排序（Top- $k$ Salience Ranking）”范式——即根据跨模态注意力得分或范数选出前 $K$ 个最高分的视觉 Token。然而，高分辨率图像中的高显著性 Token 往往高度聚集在少数显著前景物体内部（形成严重的**空间与语义聚团同质化**），导致被保留的 $K$ 个 Token 彼此高度冗余，而大面积次显著但包含关键空间关系、文字细节或环境上下文的视觉区域却“无人代言（No One Speaks for the Pruned）”而被整体抹除；与此同时，传统的子模函数或 $k$ -Medoids 多样性子集选择需要 $O(K N^2)$ 串行贪心迭代，在 GPU 上引入不可接受的推理延迟。

#### 💡 核心方法与数学公式 (Core Methodology & Formulation)
* **表征覆盖最大化目标（Representational Coverage Maximization, `CoverPruner`）**：
  设视觉 Token 表征集合为 $\mathcal{V} = \lbrace v _ 1, \dots, v _ N \rbrace \subset \mathbb{R}^d$ ，跨模态指令相关性先验权重为 $w _ i \ge 0$ 。`CoverPruner` 不再最大化保留子集 $\mathcal{S} \subset \mathcal{V}$ （ $\lvert \mathcal{S} \rvert = K$ ）的孤立得分之和，而是最大化**全体原始视觉 Token 在保留子集 $\mathcal{S}$ 上的加权最近邻表征覆盖度（Weighted Facility Location Coverage）**：

$$
\mathcal{S}^{\star} = \arg\max _ {\mathcal{S} \subset \mathcal{V}, \lvert \mathcal{S} \rvert = K} \sum _ {i=1}^{N} w _ i \cdot \max _ {j \in \mathcal{S}} \left( \frac{\langle v _ i, v _ j \rangle}{\lVert v _ i \rVert _ 2 \lVert v _ j \rVert _ 2 + \epsilon} \right)
$$

* **单次前向语义引导岭杠杆打分（Single-Forward Ridge Leverage Score, `SFPruner`）**：
  为彻底消除组合优化的串行迭代开销，`SFPruner` 将语义显著性对角矩阵 $W = \mathrm{diag}(w _ 1, \dots, w _ N)^{1/2}$ 融入特征矩阵 $\tilde{V} = W V \in \mathbb{R}^{N \times d}$ ，通过单次前向闭式计算每个视觉 Token 在主成分子空间中的**统计岭杠杆分数（Ridge Leverage Score）** $\tau _ i(\lambda)$ ，并结合排序方向掩码（Ranking-Based Directional Masking）一步抑制高共线性冗余邻居：

$$
\tau _ i(\lambda) = \tilde{v} _ i^{\top} \left( \tilde{V}^{\top} \tilde{V} + \lambda I _ d \right)^{-1} \tilde{v} _ i, \quad \tilde{\mathcal{S}} _ {\text{SF}}(i) = \tau _ i(\lambda) \cdot \left( 1 - \max _ {j : w _ j > w _ i} \cos(\tilde{v} _ i, \tilde{v} _ j) \right)
$$

#### 📊 关键实验与结论 (Key Results & Conclusions)
* 在 LLaVA-NeXT、Qwen2.5-VL 与 InternVL-2.5 等高分辨率多模态模型上，当剪除 **80%–88.9% 视觉 Token**（仅保留 64–128 个 Token）时，`CoverPruner` 与 `SFPruner` 在细粒度 OCR（DocVQA、TextVQA）与空间关系推理（MMBench、GQA）上比 FastV、SparseVLM 与 PruMerge 平均提升 **+3.4%–+5.2%**，且算子本身在 GPU 上仅耗时 **< 0.8 ms**（零迭代并行矩阵求逆）。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发 (Connection to Our Works)
* **锚定我们的代表作**：直接印证并补充了我们在 ***Understanding and Harnessing Sparsity for Unified Multimodal Models***（`TMLR 2026`, `SparseUnifiedModel`）、***ModelLesion (`EXP-E14` Token 信息密度 + 中心化分岔召回)*** 以及 ***Efficient Ads (`HisTrim`)*** 中的多模态与长序列去冗余思想。
* **落地到 `VLADrop`、`axon_v2`、`ModelLesion` 与 `efficient_ads`**：在 `VLADrop` (`models/pi0.5/`, `models/openvla-oft/`) 与 `ModelLesion` (`token_info_density_compressor.py`) 中，可直接引入 $d \times d$ 协方差矩阵求逆的**单次前向岭杠杆分数 $\tau _ i(\lambda)$ **，用闭式二阶子空间独立性度量替代启发式余弦去重。

---

> [!TIP]
> **🎯 `SparseUnifiedModel` 仓库代码级落地点 (`Target Module`)**：`sparse_umm/token_pruning.py` (k-Medoids Facility Location Coverage & Barycentric Surrogate Folding)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-29_ai_paper_notes.md`


---

### 2.2 [2026-09-29] ⚡ *VestigeKV: The NoPE-MLA KV Cache Carries Its Own Sparse-Attention Signal in a Vestigial Branch*
> 🏷️ **核心关键词**：Multi-Head Latent Attention (MLA) · NoPE (No Positional Encoding) · Sparse Attention · Training-Free KV Cache Eviction  
> 🔗 **arXiv 链接**：[`arXiv:2609.03949`](https://arxiv.org/abs/2609.03949)

```
  NoPE-MLA 潜向量 c_t^KV ──┬──► 主内容分支 (Content Latent) ──► 吸收进 W_UK 参与语义矩阵乘法
                           └──► 残余解耦分支 k_t^vest (原 RoPE 解耦通道在 NoPE 化后的低维残余信号)
                                      │
                                      ▼ (天然编码与 Query 无关的全局 Token 显著性范数 ||k_t^vest||_2)
                           [ 零额外索引开销 O(1) 稀疏 KV 检索与预算驱逐 ]
```

#### 🎯 背景与痛点 (Problem Statement)
多头潜在注意力（Multi-Head Latent Attention, MLA，如 DeepSeek-V2/V3/R1 所采用）通过将键值联合压缩为低维潜向量 $c _ t^{KV} \in \mathbb{R}^{d _ c}$ 并辅以解耦旋转位置分支 $k _ t^{R} \in \mathbb{R}^{d _ R}$ ，大幅降低了 KV 缓存体积。近期前沿架构进一步发现在深层或特定长上下文头中去除显式位置编码（NoPE-MLA）可显著提升长度外推能力。然而，在长序列推理中对 MLA 实施动态稀疏注意力（Sparse Attention）或 KV 驱逐时，由于 $c _ t^{KV}$ 被紧耦合在矩阵吸收（Matrix Absorption）中，传统方法必须额外维护一套旁路哈希索引或展开高维 Key 向量计算注意力分数，破坏了 MLA 的计算与访存紧凑性。

#### 💡 核心方法与数学公式 (Core Methodology & Formulation)
* **残余解耦分支的内生稀疏显著性发现（Vestigial Branch Saliency）**：
  `VestigeKV` 对 NoPE-MLA 架构进行了深入的谱几何剖析，发现原本为解耦位置编码设计的低维分支 $k _ t^{\text{vest}} = W _ {KR} h _ t \in \mathbb{R}^{d _ R}$ （其中 $d _ R \ll d _ c$ ，通常仅 32 或 64 维）在去除 RoPE 旋转约束后，并未退化为无用噪声，而是自发演化为一个**与具体查询向量无关（Query-Independent）的全局注意力先验门控通道**！
* **零开销残余范数打分与混合预算保留准则**：
  设第 $l$ 层历史上下文位置 $t \in \lbrace 1, \dots, S \rbrace$ 的低维残余分支向量为 $k _ t^{\text{vest},(l)}$ 。`VestigeKV` 证明历史 Token $t$ 在后续解码步被关注的期望注意力上界由其残余分支的二阶能量范数与轻量级局部内积共同主导：

$$
\mathcal{I} _ {\text{Vestige}}^{(l)}(t) = \lVert k _ t^{\text{vest},(l)} \rVert _ 2^2 + \eta \cdot \frac{\left\lvert \langle q _ {\text{cur}}^{\text{vest},(l)}, k _ t^{\text{vest},(l)} \rangle \right\rvert}{\lVert q _ {\text{cur}}^{\text{vest},(l)} \rVert _ 2 + \epsilon}
$$

  在固定 KV 预算 $B$ 下，仅需在极低维（ $d _ R = 64$ ）残余分支上按 $\mathcal{I} _ {\text{Vestige}}^{(l)}(t)$ 筛选 Top- $B$ 索引集 $\mathcal{T} _ B$ ，随后仅从 HBM 中Gather 对应的主潜向量 $\lbrace c _ t^{KV} \rbrace _ {t \in \mathcal{T} _ B}$ 参与 MLA 矩阵吸收计算。

#### 📊 关键实验与结论 (Key Results & Conclusions)
* 在基于 MLA 架构的长上下文大模型上（128K–256K 上下文长度），`VestigeKV` 无需任何重新训练或旁路预测器，在仅加载 **15%–20% KV 潜向量**的稀疏注意力预算下，在 RULER、LongBench 与大海捞针基准上达到 **99.4%** 的全量 MLA 精度，解码阶段 HBM 带宽占用再降 **4.8×**。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发 (Connection to Our Works)
* **锚定我们的代表作**：与我们在 ***Transformer-Geometry***（`EMNLP 2026`, `arXiv:2609.15975`）中关于注意力子空间正交解耦、***Pruning-on-Representations***（`ICML 2026`）中的隐空间范数相变、以及 ***Efficient Ads (`HisTrim`)*** 和 ***Awesome-LLMs-Pruning*** 的 KV Cache 压缩体系高度契合。
* **落地到 `efficient_ads`、`TraceCraft` 与 `Awesome-LLMs-Pruning`**：在 `TraceCraft` (`tracecraft/spectral_kv.py`) 与 `efficient_ads` (`histrim/kv_compressor.py`) 中，对于采用低秩联合 KV 压缩的推荐/智能体骨干网络，可直接利用低维解耦旁路通道的二阶范数 $\lVert k _ t^{\text{vest}} \rVert _ 2^2$ 作为 O(1) 零开销的静态 KV 驱逐优先级指标。

---

## 🔥 板块二：全球前沿热点精选 (Trending Frontier)

> [!TIP]
> **🎯 `SparseUnifiedModel` 仓库代码级落地点 (`Target Module`)**：`sparse_umm/kv_compression.py` (Zero-Overhead Vestigial Branch Sparse-Attention KV Indexing)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-29_ai_paper_notes.md`


---

### 2.3 [2026-09-29] 🦾 *DEE-VLA: Decoupled Early Exits for Task-Dependent Compute Allocation in Flow-Matching VLAs*
> 🏷️ **核心关键词**：Vision-Language-Action (VLA) · Flow Matching · Decoupled Early Exits · Dynamic Compute Allocation  
> 🔗 **arXiv 链接**：[`arXiv:2609.29382`](https://arxiv.org/abs/2609.29382)

```
  多模态观测 (I_t, l) ──► [ VLM 感知主干 (层 1..L_vlm) ] ──► 动态早退层 l_vlm* (自由空间移动早退，精细对准深层)
                                                                  │ (跨模块KV桥接投影)
                                                                  ▼
                     [ Flow-Matching Action Expert (层 1..L_act, 积分步 k=1..K) ]
                                                                  ├──► 动作网络深度早退 l_act*(k)
                                                                  └──► 速度场曲率收敛早退步 K* ──► 实时机器人控制动作 a_t
```

#### 🎯 背景与痛点 (Problem Statement)
现有的视觉-语言-动作（VLA）基础模型（如 $\pi _ 0$ 、 $\pi _ {0.5}$ 、GR00T）通常由百亿级参数的多模态 VLM 主干与数亿参数的流匹配（Flow-Matching）动作专家（Action Expert）组成。以往的早退（Early Exit）或层剪枝方法往往将 VLM 主干深度与动作专家深度**强行绑定（Coupled Depth Scaling）**，或者对一整段轨迹的所有时间步施加相同的静态深度预算。然而，机器人操纵任务具有显著的**时空异质性（Spatio-Temporal Heterogeneity）**：在粗粒度场景理解已完成的抓取接近阶段，VLM 主干仅需浅层表征即可维持语义定位，而进入毫米级插孔（Peg-in-Hole）接触瞬间，VLM 无需重算深层语义但 Action Expert 却需要更深的网络层数与更多的流匹配 ODE 修正步来解析高频接触动力学。

#### 💡 核心方法与数学公式 (Core Methodology & Formulation)
* **三轴解耦动态计算空间（Tri-Axis Decoupled Compute Space）**：
  `DEE-VLA` 将单次控制周期的推理算力分解为三个可独立调节的正交自由度：（1）VLM 主干退出层 $l _ {\text{vlm}} \in \lbrace 1, \dots, L _ {\text{vlm}} \rbrace$ ；（2）第 $k$ 个流匹配步中 Action Expert 的退出层 $l _ {\text{act}}^{(k)} \in \lbrace 1, \dots, L _ {\text{act}} \rbrace$ ；（3）流匹配歐拉积分的总终止步数 $K^{\star} \in \lbrace 1, \dots, K _ {\max} \rbrace$ 。
* **跨层隐状态余弦稳定性与速度场曲率早退门控**：
  在 VLM 主干内部，当相邻两层视觉-语言融合隐状态的余弦相似度超过语义收敛阈值 $\tau _ {\text{vlm}}$ 时提前退出并通过轻量级层对齐投影器生成 KV 缓存；在流匹配 Action Expert 内部，同时监测跨层速度预测残差与跨 ODE 步的**流场直线性曲率（Flow Straightness Curvature）**：

$$
\mathcal{E} _ {\text{depth}}^{(k)}(l) = \frac{\lVert v _ {\theta}^{(l)}(x _ {t _ k}, t _ k) - v _ {\theta}^{(l-1)}(x _ {t _ k}, t _ k) \rVert _ 2}{\lVert v _ {\theta}^{(l-1)}(x _ {t _ k}, t _ k) \rVert _ 2 + \epsilon} \le \delta _ {\text{act}}, \quad \mathcal{E} _ {\text{flow}}(k) = \lVert v _ {\theta}^{\star}(x _ {t _ k}, t _ k) - v _ {\theta}^{\star}(x _ {t _ {k-1}}, t _ {k-1}) \rVert _ 2 \le \delta _ {\text{ode}}
$$

  一旦 $\mathcal{E} _ {\text{flow}}(k) \le \delta _ {\text{ode}}$ （表明当前局部流场已呈直线匀速轨迹），立即跳过剩余 ODE 积分步并利用一阶欧拉外推直接输出终端动作块 $\hat{x} _ 1 = x _ {t _ k} + (1 - t _ k) v _ {\theta}^{\star}(x _ {t _ k}, t _ k)$ 。

#### 📊 关键实验与结论 (Key Results & Conclusions)
* 在 LIBERO（Spatial / Object / Goal / Long）与真机双臂灵巧操作任务上，`DEE-VLA` 在成功率与全深度 10-NFE 基线持平（甚至因减少自由空间过拟合而提升 **+0.8%**）的同时，平均削减了 **54.2% 的总 FLOPs** 与 **49.6% 的端到端控制延迟**，自动展现出“自由空间巡航浅层少步、接触操作阶段深层精细积分”的涌现算力分配规律。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发 (Connection to Our Works)
* **锚定我们的代表作**：与我们在 `axon_v2` / `VLADrop` 中构建的 **Tri-Orthogonal Depth-Width-Step Compression (`G19`)**、**Once-for-All Switchable Multi-Gear Loops** 以及 **Looped VLA 动态停止准则 $K(s _ t, t, m)$ ** 完全同源！
* **落地到 `axon_v2` 与 `VLADrop` (`VLM-Compression`)**：可在 `axon/layers/looped_ode.py` 与 `VLADrop/models/pi0.5/` 中直接融合 `DEE-VLA` 的双判据 $\left( \mathcal{E} _ {\text{depth}}^{(k)}(l), \mathcal{E} _ {\text{flow}}(k) \right)$ ，将 VLM 编码器深度门控与 Action Expert 的 ODE 步曲率门控彻底解耦。

---

> [!TIP]
> **🎯 `SparseUnifiedModel` 仓库代码级落地点 (`Target Module`)**：`sparse_umm/flow_generation.py` (Decoupled Early-Exit Compute Allocation across VLM & Flow Experts)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-29_ai_paper_notes.md`


---

### 2.4 [2026-09-29] 🌍 *WM2VLA & InternW0-Δ: Think Like a World Model, Act Like a VLA — Distilling World-Model Representations & Causal Imprint into Compact Robot Policies*
> 🏷️ **核心关键词**：World Action Model (WAM) · World-Model Representation Distillation · Causal Imprint · Rollout-Free Real-Time Control  
> 🔗 **arXiv 链接**：[`arXiv:2609.24682`](https://arxiv.org/abs/2609.24682) (`WM2VLA`) & [`arXiv:2609.31394`](https://arxiv.org/abs/2609.31394) (`InternW0-Δ`)

```
  [ 训练期: 20,000+ 小时跨本体物理交互视频与动作流 ]
       ├──► 高容量视频世界模型教师 (预测未来帧潜表征 z_{t+1:t+H}^{WM})
       │            │
       │            ▼ (Causal Imprint 因果烙印 & 跨层未来动力学表征蒸馏 L_WM-Align)
       └──► 紧凑型 VLA 策略学生 (中间层隐状态 h_t^{VLA} ──► 流匹配动作头 a_{t:t+H})
                    │
  [ 推理期: 完全剥离未来视频生成分支 (Zero Video Rollout)，以纯紧凑 VLA 实现 < 20ms 闭环物理常识控制 ]
```

#### 🎯 背景与痛点 (Problem Statement)
将视频物理世界模型（World Models）引入机器人具身控制面临一个尖锐的**“预测质量—控制延迟悖论”**：一方面，在大规模人类与机器人操作视频上预训练的世界模型蕴含了极其丰富的三维物理接触、刚体碰撞与遮挡预测先验；另一方面，若在测试推理期让世界模型显式滚动生成未来视频帧（Video Rollout）再交由逆动力学模型解码动作，单次决策延迟高达 **500–2,000 ms**，根本无法满足 20–50Hz 的实时闭环控制需求。如何让轻量级 VLA 策略在不执行任何推理期视频生成的条件下，依然具备世界模型的前瞻物理推演能力？

#### 💡 核心方法与数学公式 (Core Methodology & Formulation)
* **跨模态因果烙印与未来动力学隐空间对齐（Causal Imprint & Representation Distillation）**：
  `WM2VLA`（*Think Like a World Model, Act Like a VLA*）与上海 AI Lab 的 `InternW0-Δ`（基于 20,000+ 小时开源具身数据训练的 Mixture-of-Transformers 世界动作模型）提出了高度一致的核心解耦范式：在训练阶段，冻结或联合演化高容量世界模型专家，提取其对未来 $H$ 步状态转移的预测潜表征 $Z _ {t+1:t+H}^{\text{WM}} \in \mathbb{R}^{H \times d _ w}$ ；同时在紧凑型 VLA 策略网络的第 $l^{\star}$ 层引入轻量级预测投影头 $P _ {\phi}$ ，强迫当前观测下的策略隐状态 $h _ t^{\text{VLA},(l^{\star})}$ 直接烙印（Imprint）未来物理演化流形：

$$
\mathcal{L} _ {\text{total}} = \underbrace{\mathbb{E} _ {t, \tau, x _ 0} \left\lVert v _ {\theta}\left( x _ {\tau}, \tau \mid h _ t^{\text{VLA}} \right) - \left( a _ {t:t+H} - x _ 0 \right) \right\rVert _ 2^2} _ {\text{Conditional Flow-Matching Action Loss}} + \lambda _ {\text{WM}} \cdot \underbrace{\left( 1 - \frac{\left\langle P _ {\phi}\left( h _ t^{\text{VLA},(l^{\star})} \right), \mathrm{sg}\left( Z _ {t+1:t+H}^{\text{WM}} \right) \right\rangle}{\left\lVert P _ {\phi}\left( h _ t^{\text{VLA},(l^{\star})} \right) \right\rVert _ F \left\lVert \mathrm{sg}\left( Z _ {t+1:t+H}^{\text{WM}} \right) \right\rVert _ F + \epsilon} \right)} _ {\text{Rollout-Free World-Model Causal Imprint Loss}}
$$

* **非对称 Mixture-of-Transformers (MoT) 与推理期零滚动剥离**：
  在推理阶段，投影头 $P _ {\phi}$ 与全部视频世界模型参数被物理卸载（Zero-Overhead Pruning at Inference），紧凑型 VLA 凭借已被因果烙印重塑的中间层几何表征 $h _ t^{\text{VLA},(l^{\star})}$ ，单次前向即可生成符合长程物理可行性的动作块。

#### 📊 关键实验与结论 (Key Results & Conclusions)
* 在 RoboTwin、CALVIN、SIMPLER 及多项真机长程操作基准上，`WM2VLA` 与 `InternW0-Δ` 相较于无世界模型先验的同规模 VLA 策略，平均任务成功率大幅跃升 **+11.5%–+16.8%**（尤其在可变形物体与推挡接触任务上优势显著），同时推理延迟较显式视频生成世界模型降低 **18×–35×**。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发 (Connection to Our Works)
* **锚定我们的代表作**：直接赋能我们的 **`axon_v2` (`axon/data_rsi/world_verifier.py` & `SnapFlow`)**、**`VLADrop` (`models/gigabrain-0/`, `models/pi0.5/`)** 以及 **`mera` (`mera/flow_matching_merge.py`)**。
* **落地到 `axon_v2` 与 `VLADrop`**：在训练经过 `VLADrop` 层剪枝与 `SnapFlow` 1-NFE 蒸馏的紧凑 VLA 学生模型时，除了匹配教师的动作速度场，可在保留的桥接层上附加一项 $\mathcal{L} _ {\text{WM-Imprint}}$ 未来状态余弦对齐损失，在不增加任何端侧推理算力的前提下注入物理世界模型先验！

---

> [!TIP]
> **🎯 `SparseUnifiedModel` 仓库代码级落地点 (`Target Module`)**：`sparse_umm/` (`Shwai-He/SparseUnifiedModel` & `Shwai-He/VLM-Compression`)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-29_ai_paper_notes.md`


---

### 2.5 [2026-09-29] 🧬 *Failure-RSI & Flow3D-OPD: Inference-Time Failure-Driven Agent Patching & Multi-Teacher On-Policy Flow Distillation*
> 🏷️ **核心关键词**：Inference-Time Self-Improvement · Failure-Driven Code Patching · Multi-Teacher On-Policy Distillation (OPD) · Flow-Matching DiT  
> 🔗 **arXiv 链接**：[`arXiv:2606.31270`](https://arxiv.org/abs/2606.31270) (`Failure-RSI`, ECCV 2026) & [`arXiv:2609.07137`](https://arxiv.org/abs/2609.07137) (`Flow3D-OPD`)

```
  [ 轨迹一: Failure-RSI 失败驱动推理期智能体自进化 (2606.31270) ]
  Agent 执行失败轨迹 τ_fail ──► 跨模态反事实根因定位 (Root-Cause Diagnosis) ──► 合成工具/动作护栏代码补丁 ΔC ──► 沙箱回归验证后热更新脚手架 C_{t+1}

  [ 轨迹二: Flow3D-OPD 多教师在线流匹配蒸馏 (2609.07137) ]
  少步学生自生成轨迹 x_t^{stu} ──► 查询 M 个专长教师速度场 {v_m^{tea}(x_t^{stu}, t)} ──► 置信度加权速度场融合 ──► 消除单教师盲区与暴露偏差
```

#### 🎯 背景与痛点 (Problem Statement)
1. **智能体自进化中的“幸存者偏差陷阱”**：绝大多数智能体自我改进（Self-Improvement / Rejection Sampling）框架仅收集并强化**成功轨迹（Success-Only Trajectories）**，而将占比高达 60%–80% 的失败轨迹直接丢弃。这导致智能体只能在其已知能力圈内反复强化，无法修复因系统环境变化、API 边缘异常或 UI 布局漂移引发的确定性失败模式。
2. **单教师在线流匹配蒸馏的专长盲区**：在流匹配（Flow-Matching）扩散 Transformer 的在线策略蒸馏（On-Policy Distillation, OPD）中，单一教师模型往往难以在所有几何拓扑或动作模态上同时保持最优，若在学生自生成轨迹上盲目向劣质教师分支对齐，会导致局部几何失真。

#### 💡 核心方法与数学公式 (Core Methodology & Formulation)
* **失败驱动推理期脚手架补丁合成（Failure-Driven Inference-Time Code Patching, `Failure-RSI`）**：
  给定失败执行轨迹 $\tau _ {\text{fail}} = \left( s _ 0, a _ 0, s _ 1, \dots, s _ T \right)$ ，诊断器首先识别出首个偏离预期状态转移的**关键分岔步（Critical Divergence Step）** $t^{\star} = \arg\max _ t \mathcal{D} _ {\text{sem}}\left( s _ {t+1}, \hat{s} _ {t+1}^{\text{exp}} \right)$ ，并生成最小可执行脚手架代码补丁 $\Delta \mathcal{H}$ （如前置状态校验器、异常恢复重试器或坐标校准转换函数），仅当补丁在历史成功集 $\mathcal{D} _ {\text{pass}}$ 上零退化且修复 $\tau _ {\text{fail}}$ 时予以合并：

$$
\mathcal{H} _ {k+1} = \mathcal{H} _ k \oplus \Delta \mathcal{H}^{\star}, \quad \text{where} \quad \Delta \mathcal{H}^{\star} = \arg\max _ {\Delta \mathcal{H}} \mathbb{I}\left\lbrace \mathrm{Eval}(\mathcal{H} _ k \oplus \Delta \mathcal{H}, \tau _ {\text{fail}}) = 1 \right\rbrace \cdot \mathbb{I}\left\lbrace \mathrm{Regress}(\mathcal{H} _ k \oplus \Delta \mathcal{H}, \mathcal{D} _ {\text{pass}}) = 0 \right\rbrace
$$

* **多教师在线策略速度场蒸馏（Multi-Teacher On-Policy Flow Distillation, `Flow3D-OPD`）**：
  在少步学生模型 $v _ {\theta}$ 沿自身积分轨迹采样的在线状态 $x _ t^{\text{stu}} \sim p _ {\theta}(x _ t)$ 上，同时查询 $M$ 个异构专长流匹配教师 $\lbrace u _ {\psi _ m} \rbrace _ {m=1}^{M}$ ，并按各教师在当前状态邻域的能量匹配置信度 $\alpha _ m(x _ t^{\text{stu}}, t)$ 动态融合目标速度场：

$$
\mathcal{L} _ {\text{MT-OPD}}(\theta) = \mathbb{E} _ {t, x _ t^{\text{stu}} \sim p _ {\theta}} \left\lVert v _ {\theta}\left( x _ t^{\text{stu}}, t \right) - \sum _ {m=1}^{M} \alpha _ m\left( x _ t^{\text{stu}}, t \right) \cdot u _ {\psi _ m}\left( x _ t^{\text{stu}}, t \right) \right\rVert _ 2^2, \quad \sum _ {m=1}^{M} \alpha _ m = 1
$$

#### 📊 关键实验与结论 (Key Results & Conclusions)
* **`Failure-RSI`**：在 OSWorld 与多模态计算机操作基准上，仅利用推理期失败轨迹自动合成工具与控制补丁，无需微调底层大模型权重即可将任务成功率相对提升 **+24.6%**，且合成的代码补丁具备跨任务泛化性。
* **`Flow3D-OPD`**：在流匹配 Diffusion Transformer 少步（1–4 NFE）蒸馏中，多教师在线轨迹对齐比单教师离线蒸馏在几何保真度与分布覆盖度指标上提升 **+8.3%**。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发 (Connection to Our Works)
* **锚定我们的代表作**：
  1. **`TraceCraft`、`Better-Peer-Review` 与 `stock_prediction`**：`Failure-RSI` 的“关键分岔步诊断 + 零退化回归代码补丁合并（ $\mathcal{H} _ k \oplus \Delta \mathcal{H}^{\star}$ ）”可直接强化 `TraceCraft` (`tracecraft/autoresearch_loop.py`)、`Better-Peer-Review` (`rsi_bpr_eval/mutable_operator.py`) 与 `stock_prediction` (`rsi_campaign/mutable_operator.py`) 的算子自进化循环！
  2. **`mera` 与 `axon_v2`**：`Flow3D-OPD` 的多教师在线速度场加权融合公式可直接落地到 `mera` (`mera/flow_matching_merge.py`) 的多专家流匹配模型融合以及 `axon_v2` (`axon/distillation/on_policy_flow.py`) 的多技能 VLA 联合蒸馏中。

> [!TIP]
> **🎯 `SparseUnifiedModel` 仓库代码级落地点 (`Target Module`)**：`sparse_umm/flow_generation.py` (Multi-Teacher On-Policy Flow Trajectory Distillation)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-29_ai_paper_notes.md`


---

### 2.6 [2026-09-28] ✂️ *CLSE: Spectral Evolution-Guided Token Pruning in Multimodal Large Language Models*
> 🏷️ **核心关键词**：Multimodal Token Pruning · Cross-Layer Spectral Evolution · Discrete Cosine Transform (DCT) · Training-Free Compression  
> 🔗 **arXiv 链接**：[`arXiv:2606.24165`](https://arxiv.org/abs/2606.24165) (ECCV 2026)

```
  层 l-1 视觉隐状态 H^(l-1) ──► [ 通道维 DCT 频域投影 Φ ] ──► 频谱能量分布 P^(l-1)(ω) ┐
                                                                                      ├──► [ 跨层谱演化散度 D_CLSE(i) ] ──► Top-K 语义活跃视觉 Token 保留
  层 l   视觉隐状态 H^(l)   ──► [ 通道维 DCT 频域投影 Φ ] ──► 频谱能量分布 P^(l)(ω)   ┘
```

#### 🎯 背景与痛点 (Problem Statement)
现有多模态大模型（MLLM / VLM）免训练视觉 Token 剪枝方法（如 FastV、SparseVLM）大多依赖**单层静态注意力分数**（如第 $l$ 层文本对视觉 Token 的注意力权重 $A _ {t, v}^{(l)}$ ）。然而，由于 RoPE 旋转位置编码的远程衰减与视觉 Sink Token 现象，单层注意力极易受到空间位置偏置（Position Bias）误导——许多在当前层注意力得分较高但跨层表征几乎停止更新的“静态背景/锚点冗余 Token”被错误保留，而真正正在经历高频语义整合的关键局部视觉 Token 却被过早裁剪。

#### 💡 核心方法与数学公式 (Core Methodology & Formulation)
* **跨层频域映射与归一化能量谱**：
  设第 $l$ 层第 $i$ 个视觉 Token 的隐状态向量为 $h _ i^{(l)} \in \mathbb{R}^d$ 。CLSE 首先通过正交离散余弦变换（DCT）基矩阵 $\Phi \in \mathbb{R}^{d \times d}$ 将特征变化量 $\Delta h _ i^{(l)} = h _ i^{(l)} - h _ i^{(l-1)}$ 投影至频域，计算频谱系数 $c _ i^{(l)} = \Phi h _ i^{(l)}$ ，并构造归一化频域能量分布：

$$
p _ {i, k}^{(l)} = \frac{\left( c _ {i, k}^{(l)} \right)^2}{\sum _ {m=1}^{d} \left( c _ {i, m}^{(l)} \right)^2 + \epsilon}, \quad k \in \lbrace 1, \dots, d \rbrace
$$

* **跨层谱演化散度（Cross-Layer Spectral Evolution Score）**：
  原文发现：真正参与跨模态语义推理的视觉 Token 在穿越浅层到中层 Transformer 时，其能量会从高频局部纹理分量向低频全局语义分量发生剧烈的**谱重分布（Spectral Redistribution）**；而背景冗余 Token 的频谱分布则保持停滞。因此定义第 $i$ 个视觉 Token 在第 $l$ 层的跨层谱演化显著性打分为对称 Jensen-Shannon 谱演化散度与残差平行演化幅度的乘积：

$$
\mathcal{S} _ {\text{CLSE}}^{(l)}(i) = \mathrm{JSD}\left( p _ i^{(l)} \parallel p _ i^{(l-1)} \right) \cdot \frac{\lVert h _ i^{(l)} - h _ i^{(l-1)} \rVert _ 2}{\lVert h _ i^{(l-1)} \rVert _ 2 + \epsilon}
$$

* **谱演化引导渐进剪枝**：
  在预设的剪枝过渡层 $l \in \mathcal{L} _ {\text{prune}}$ ，仅保留 $\mathcal{S} _ {\text{CLSE}}^{(l)}(i)$ 排名前 $K _ l$ 的语义活跃 Token，被裁剪的背景 Token 按频谱相似度加权合并至最近邻保留 Token 中以守恒低频能量。

#### 📊 关键实验与结论 (Key Results & Conclusions)
* 在 LLaVA-NeXT-7B/13B 与 Qwen2.5-VL-7B 上，免训练剪除 **75% 视觉 Token** 时，在 DocVQA、ChartQA、TextVQA 与 Video-MME 等 10 项多模态基准上平均保留了 **99.1%** 的原始全量精度，显著超越单层注意力剪枝基线（+3.8%），预填充（Prefill）FLOPs 降低 **68%**。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发 (Connection to Our Works)
* **锚定我们的代表作**：直接呼应我们在 ***Understanding and Harnessing Sparsity for Unified Multimodal Models***（`TMLR 2026`, `SparseUnifiedModel`）、***Demystifying When Pruning Works via Representation Hierarchies***（`ICML 2026`, `Pruning-on-Representations`）以及 ***Transformer-Geometry***（`EMNLP 2026`, `arXiv:2609.15975`）中提出的跨层表征几何演化理论。
* **落地到 `VLADrop` (`VLM-Compression`) 与 `efficient_ads` (`HisTrim`)**：在我们的 `VLADrop` 具身视觉编码器与 `HisTrim` 多阶段分层序列裁剪中，可将单层注意力打分升级为 **跨层平行/正交残差演化率 + 频域谱重分布散度 $\mathcal{S} _ {\text{CLSE}}^{(l)}$ **，用零额外参数的逐层残差差分替代易受位置偏置干扰的静态注意力权重。

---

> [!TIP]
> **🎯 `SparseUnifiedModel` 仓库代码级落地点 (`Target Module`)**：`sparse_umm/token_pruning.py` (Cross-Layer Spectral Entropy Evolution & Subspace Orthogonal Deduplication)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-28_ai_paper_notes.md`


---

### 2.7 [2026-09-28] ✂️ *ASL: Adaptive Layer Selection for Layer-Wise Token Pruning in LLM Inference*
> 🏷️ **核心关键词**：Layer-Wise Token Pruning · Adaptive Layer Selection · Attention Variance · Long-Context LLM Inference  
> 🔗 **arXiv 链接**：[`arXiv:2601.07667`](https://arxiv.org/abs/2601.07667) (ACL 2026 Findings)

```
  输入长序列 X ──► 逐层前向传播 l=1..L ──► 实时监测注意力熵变与表征漂移率 η_l
                                                    │
                        ┌───────────────────────────┴───────────────────────────┐
                        ▼ (η_l 跌破相变阈值 τ: 语义路由已收敛)                     ▼ (η_l > τ: 仍在剧烈跨位置交互)
          [ 触发 ASL 单次 Token 剪枝 (One-Shot Selection) ]                [ 保持全长序列继续前向传播 ]
```

#### 🎯 背景与痛点 (Problem Statement)
现有的长上下文逐层 Token 剪枝方法（如 PyramidInfer、LazyLLM）通常采用**跨样本固定的剪枝层配置**（例如硬编码在第 4、8、16 层按固定比例裁剪 Token）。然而，不同复杂度与不同上下文长度的输入样本，其跨位置信息汇聚的完成深度截然不同：简单检索任务在第 6 层已完成关键信息聚焦，而多跳推理任务直到第 18 层仍在跨段落聚合线索。静态固定剪枝层要么在困难样本上过早剪断推理链，要么在简单样本上浪费大量冗余计算。

#### 💡 核心方法与数学公式 (Core Methodology & Formulation)
* **跨层注意力方差与路由收敛度度量**：
  设第 $l$ 层查询窗口对上下文 Token 的平均注意力分布为 $\bar{\alpha}^{(l)} \in \Delta^{N-1}$ 。ASL 提出用注意力分布的**二阶方差锐度（Attention Variance Sharpness）**与相邻层注意力分布的 **余弦收敛度** 联合度量当前层是否已完成信息路由聚焦：

$$
\mathcal{C} _ l = \mathrm{Var}\left( \bar{\alpha}^{(l)} \right) \cdot \frac{\left\langle \bar{\alpha}^{(l)}, \bar{\alpha}^{(l-1)} \right\rangle}{\lVert \bar{\alpha}^{(l)} \rVert _ 2 \lVert \bar{\alpha}^{(l-1)} \rVert _ 2 + \epsilon}
$$

* **自适应剪枝层触发准则（Adaptive Layer Selection）**：
  当第 $l$ 层的聚焦收敛指数 $\mathcal{C} _ l$ 首次超过样本自适应阈值 $\tau _ {\text{ASL}}$ 且层间相对增幅趋于平缓（即 $\lvert \mathcal{C} _ l - \mathcal{C} _ {l-1} \rvert \le \delta$ ）时，ASL 判定该样本在层 $l^{\star}$ 已越过“信息收集—语义提纯相变点”，随即在层 $l^{\star}$ 触发 **One-Shot Token Selection**，一次性保留核心上下文子集 $\mathcal{I} _ {\text{keep}}$ ：

$$
l^{\star}(x) = \min \left\lbrace l \in \lbrace l _ {\min}, \dots, L \rbrace \middle| \mathcal{C} _ l(x) \ge \tau _ {\text{ASL}} \land \lvert \mathcal{C} _ l(x) - \mathcal{C} _ {l-1}(x) \rvert \le \delta \right\rbrace
$$

#### 📊 关键实验与结论 (Key Results & Conclusions)
* 在 Llama-3.1-8B/70B 与 Qwen2.5-14B 上，针对 RULER、InfiniteBench 与 Needle-in-a-Haystack（128K 上下文）评测表明：在相同的 **2.4× 端到端推理加速比**下，ASL 比固定层级剪枝基线在多跳问答与长程聚合任务上平均提升 **+4.6 分**，彻底消除了静态早剪导致的“大海捞针丢失”现象。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发 (Connection to Our Works)
* **锚定我们的代表作**：与我们在 ***Uncovering the Redundancy in Transformers via Layer Dropping***（`TMLR 2025`, `LLM-Drop`）、***Router-Tuning: A Simple and Effective Approach for Enabling Dynamic-Depth in Transformers***（`EMNLP 2025`, `Router-Tuning-Mixture-of-Depths`）以及 ***Demystifying When Pruning Works via Representation Hierarchies***（`ICML 2026`, `Pruning-on-Representations`）中揭示的“语义表征相变层（Phase-Transition Layer）”高度吻合。
* **落地到 `LLM-Drop`、`ModelLesion` 与 `efficient_ads`**：可将 ASL 的样本级在线收敛准则 $\mathcal{C} _ l(x)$ 引入 `efficient_ads` 的 `HisTrim` 多阶段裁剪触发器以及 `LLM-Drop` 的动态跳过门控中，实现**按样本难度自适应推迟或提前剪枝触发层 $l^{\star}(x)$ **。

---

> [!TIP]
> **🎯 `SparseUnifiedModel` 仓库代码级落地点 (`Target Module`)**：`sparse_umm/token_pruning.py` (Marginal Information Gain Adaptive Layer Selection)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-28_ai_paper_notes.md`


---

### 2.8 [2026-09-28] 🦾 *IMLE-VLA: Fast Single-Step Action Generation for Vision-Language-Action Policies*
> 🏷️ **核心关键词**：Vision-Language-Action (VLA) · Implicit Maximum Likelihood Estimation (cIMLE) · Single-Step 1-NFE Generation · Robotic Control  
> 🔗 **arXiv 链接**：[`arXiv:2609.04369`](https://arxiv.org/abs/2609.04369)

```
  多模态观测 o_t + 潜噪声集 {z_1..z_M} ──► [ 单步条件生成器 G_θ(o_t, z_m) ] ──► 候选动作块集合 {â_1..â_M}
                                                                                        │
  真实专家演示动作块 a_t ───────────────► [ cIMLE 最近邻覆盖损失: min_m ||a_t - â_m||_2^2 ] ◄─┘
  (推理期: 仅需采样单一 z ~ N(0,I)，1-NFE 单次前向直接输出平滑连续动作块 â_t = G_θ(o_t, z))
```

#### 🎯 背景与痛点 (Problem Statement)
当前主流视觉—语言—动作（VLA）基座模型（如 $\pi _ 0$ 、 $\pi _ {0.5}$ 、GR00T）普遍采用流匹配（Flow Matching）或扩散模型作为动作头（Action Head），虽然能拟合多峰（Multi-modal）人类演示轨迹，但在实时推理时必须执行 5–10 步串行 ODE 数值积分，导致控制频率受限且端侧延迟高昂。若直接用均方误差（MSE）回归训练单步生成器，则会把多条合法绕障轨迹平均到障碍物中心，引发致命的**模式坍缩（Mode Collapse）**；而对抗生成或步数蒸馏则常常面临训练不稳定与高阶曲率失真。

#### 💡 核心方法与数学公式 (Core Methodology & Formulation)
* **条件隐式最大似然单步生成目标（Conditional IMLE Objective）**：
  IMLE-VLA 彻底抛弃了“对每个噪声都强制拉向真实样本（会导致多峰平均）”或“反向 KL 模式寻优（会导致模式丢弃）”的传统思路，转而在训练期对每个专家演示动作块 $a _ t \in \mathbb{R}^{H \times d _ a}$ （观测条件为 $o _ t$ ）采样 $M$ 个潜噪声向量 $\lbrace z _ m \rbrace _ {m=1}^M \sim \mathcal{N}(0, I)$ ，通过单步动作网络 $G _ \theta(o _ t, z _ m)$ 并行生成 $M$ 个候选动作块，并**仅优化距离真实演示动作 $a _ t$ 最近的那一个候选样本**：

$$
\mathcal{L} _ {\text{cIMLE}}(\theta) = \mathbb{E} _ {(o _ t, a _ t) \sim \mathcal{D}} \left[ \mathbb{E} _ {z _ {1:M} \sim \mathcal{N}(0, I)} \left[ \min _ {m \in \lbrace 1, \dots, M \rbrace} \lVert G _ \theta(o _ t, z _ m) - a _ t \rVert _ 2^2 \right] \right]
$$

* **几何意义与零模式坍缩保证**：
  因为损失函数要求“每一个真实训练样本 $a _ t$ 都至少被一个生成样本 $G _ \theta(o _ t, z _ {m^{\star}})$ 贴近覆盖”，生成器被显式驱动去完整覆盖专家多峰动作流形的全部模式分支；而在实际机器人部署推理时，只需采样单个噪声 $z \sim \mathcal{N}(0, I)$ 执行 **1-NFE 单次前向传播** $\hat{a} _ t = G _ \theta(o _ t, z)$ ，即可直接输出高保真动作序列。

#### 📊 关键实验与结论 (Key Results & Conclusions)
* 在 LIBERO、SimplerEnv 与真实双臂灵巧操作任务上，IMLE-VLA 以 **1-NFE 单步前向** 将动作生成吞吐量与控制频率提升 **6.4×–8.5×**，同时任务成功率不仅远超单步回归基线（+14.2%），更追平甚至略超 10 步流匹配（10-NFE Flow Matching）策略。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发 (Connection to Our Works)
* **锚定我们的活跃研究线**：直接赋能我们的 **`axon_v2` / `axon` (`vla-distillation` — SnapFlow 1-NFE & MeanFlow/IMM 动作头蒸馏)** 与 **`VLADrop` (`VLM-Compression`)**。
* **落地到 `axon_v2` 的 SnapFlow 1-NFE 训练器**：在 `axon_v2` 的 Stage-2 1-NFE 动作头蒸馏中，单步全视野弦速度匹配（Chord Velocity Matching）在遇到高度对称的双侧避障演示时偶有轨迹折中倾向；将 **cIMLE 最近邻多噪声覆盖项 $\min _ {m} \lVert G _ \theta(o _ t, z _ m) - a _ t \rVert _ 2^2$ ** 作为辅助正则项融入 `SnapFlow` 1-NFE 损失，可在零推理开销下彻底根除 1-NFE 蒸馏的对称模式折中问题！

---

> [!TIP]
> **🎯 `SparseUnifiedModel` 仓库代码级落地点 (`Target Module`)**：`sparse_umm/flow_generation.py` (Single-Step IMLE Mode-Covering Action/Visual Generation)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-28_ai_paper_notes.md`


---

### 2.9 [2026-09-27] OBCache: Optimal Brain KV Cache Pruning for Efficient Long-Context LLM Inference

* **论文信息**：Yuzhe Gu, Xiyu Liang, Jiaojiao Zhao, Enmao Diao (`arXiv:2510.07651`, **ICML 2026**)
* **核心关键词**：KV Cache Eviction、Optimal Brain Damage (OBD)、Second-Order Taylor Perturbation、Output-Aware Saliency、Joint KV Pruning

#### 📐 架构与核心算法流程图 (ASCII Blueprint)

```text
+-----------------------------------------------------------------------------------+
|         OBCache: Optimal Brain Damage (OBD) Layer-Wise KV Cache Pruning           |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  Prefill / Decoding Step: Queries Q \in R^{S_q x d_k}, Cached K, V \in R^{S_k x d}|
|                                        |                                          |
|                                        v                                          |
|  +-----------------------------------------------------------------------------+  |
|  | 1. Attention Output Perturbation Objective (层输出二阶泰勒扰动建模)         |  |
|  |    Target: Minimize || O - \tilde{O}(\mathcal{M}) ||_F^2 where O = A V       |  |
|  |    Instead of heuristic \sum_i A_{i,j}, expand \Delta O w.r.t. masked K_j,V_j|  |
|  +-----------------------------------------------------------------------------+  |
|                                        |                                          |
|           +----------------------------+----------------------------+             |
|           v                            v                            v             |
|  +-----------------+          +-----------------+          +-------------------+  |
|  | Isolated Value  |          |  Isolated Key   |          | Joint KV Saliency |  |
|  | Score \Omega_j^V|          |  Score \Omega_j^K|         | Score \Omega_j^{KV}| |
|  | ||A_{:,j}||_2^2 |          | Softmax Jacobian|          | Exact Rank-1      |  |
|  | * ||V_j||_2^2   |          | Coupling Term   |          | Softmax Renorm    |  |
|  +-----------------+          +-----------------+          +-------------------+  |
|                                        |                                          |
|                                        v                                          |
|  +-----------------------------------------------------------------------------+  |
|  | 2. Plug-and-Play Eviction Gate (即插即用淘汰门控: 兼容 SnapKV / PyramidKV)  |  |
|  |    Evict tokens with minimal \Omega_j^{KV} -> Retain top-B KV budget        |  |
|  +-----------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------+
```

#### 🎯 背景与痛点 (Background & Pain Points)
* **启发式注意力权重累加的理论缺陷**：主流长上下文 KV 缓存淘汰算法（如 H2O、SnapKV、PyramidKV）均使用累积注意力分数 $s _ j = \sum _ {i} A _ {i,j}$ 作为 Token $j$ 的重要性指标。然而，注意力层真正传递给后续残差流的是加权输出矩阵 $O = A V \in \mathbb{R}^{S _ q \times d _ v}$ ：
  1. **忽略 Value 向量范数与方向抵消**：若某个历史 Token $j$ 的注意力权重 $A _ {i,j}$ 较高，但其对应的 Value 向量范数 $\Vert V _ j\Vert _ 2 \approx 0$ ，或者其 $V _ j$ 与当前上下文均值方向完全重合，驱逐它对注意力输出 $O$ 的实际影响极小；反之，注意力权重中等但 $\Vert V _ j\Vert _ 2$ 极大且承载正交关键信息的 Token 被驱逐后会造成严重的输出畸变。
  2. **忽略 Softmax 分母重归一化效应（Denominator Renormalization）**：驱逐第 $j$ 个 Key 相当于将注意力得分 $Z _ {i,j} \to -\infty$ ，这不仅移除了 $A _ {i,j} V _ j$ ，还会通过 Softmax 分母缩放将其余所有保留 Token 的注意力权重放大 $\frac{1}{1 - A _ {i,j}}$ 倍。

#### 💡 核心方法与数学公式 (Core Methodology & Math)
1. **基于 Optimal Brain Damage (OBD) 的二阶输出扰动构建**：
   设某注意力头在查询窗口 $Q \in \mathbb{R}^{S _ q \times d _ k}$ 下的注意力概率矩阵为 $A = \text{Softmax}\left(\frac{Q K^\top}{\sqrt{d _ k}}\right) \in \mathbb{R}^{S _ q \times S _ k}$ ，输出为 $O = A V \in \mathbb{R}^{S _ q \times d _ v}$ 。定义驱逐准则为最小化层输出矩阵的 Frobenius 范数平方误差 $\mathcal{E} = \frac{1}{2} \Vert O - \tilde{O} \Vert _ F^2$ 。
2. **单 Value、单 Key 与联合 KV 对的闭式显著性公式（Closed-Form Saliency Scores）**：
   * **孤立 Value 剪枝显著性（Isolated Value Saliency $\Omega _ j^V$ ）**：
     当将第 $j$ 个 Token 的 Value 向量置零（ $V _ j \leftarrow 0$ ）时， $\mathcal{E}$ 对 $V _ j$ 的海森矩阵（Hessian）为 $\mathbf{H} _ {V _ j} = \frac{\partial^2 \mathcal{E}}{\partial V _ j \partial V _ j^\top} = \left(\sum _ {i=1}^{S _ q} A _ {i,j}^2\right) I _ {d _ v}$ 。根据二阶泰勒展开，孤立 Value 显著性得分为：

$$
\Omega _ j^V = \frac{1}{2} V _ j^\top \mathbf{H} _ {V _ j} V _ j = \frac{1}{2} \Vert A _ {:, j} \Vert _ 2^2 \cdot \Vert V _ j \Vert _ 2^2
$$

注意此处注意力权重是**平方和 $\Vert A _ {:,j}\Vert _ 2^2$ **（二阶能量）而非启发式的线性求和 $\Vert A _ {:,j}\Vert _ 1$ ，且显式乘上了 Value 范数平方 $\Vert V _ j\Vert _ 2^2$ ！
   * **联合 KV 剪枝与 Softmax 重归一化修正（Joint KV Saliency $\Omega _ j^{KV}$ ）**：
     当真正从缓存中移除第 $j$ 个 KV 对（即令未归一化 logit $Z _ {i,j} \to -\infty$ ）时，剩余 Token $k \neq j$ 的注意力权重精确变为 $\tilde{A} _ {i,k} = \frac{A _ {i,k}}{1 - A _ {i,j}}$ 。因此，移除第 $j$ 个 KV 对在第 $i$ 个查询位置引起的**精确输出残差**为：

$$
\Delta O _ i^{(-j)} = O _ i - \tilde{O} _ i^{(-j)} = O _ i - \frac{O _ i - A _ {i,j} V _ j}{1 - A _ {i,j}} = \frac{A _ {i,j}}{1 - A _ {i,j}} \big( V _ j - O _ i \big)
$$

对该精确残差在所有查询位置 $i \in \lbrace1, \dots, S _ q\rbrace$ 上求二阶能量，即得到极其优雅的**联合 KV 闭式显著性得分**：

$$
\Omega _ j^{KV} = \frac{1}{2} \sum _ {i=1}^{S _ q} \left( \frac{A _ {i,j}}{1 - A _ {i,j}} \right)^2 \big\Vert V _ j - O _ i \big\Vert _ 2^2
$$

#### 📊 关键实验与结论 (Key Experiments & Takeaways)
* **即插即用全面提升主流基线**：在 **Llama-3.1-8B-Instruct**、**Qwen-2.5-7B/14B-Instruct** 与 **Mistral-7B** 上，将 OBCache 的 $\Omega _ j^{KV}$ 闭式打分直接替换 H2O、SnapKV 与 PyramidKV 的启发式打分（零额外超参），在 **LongBench**（16 个长文本任务）与 **RULER**（128K 极限大海捞针与多跳追踪）上，在仅保留 **5%–10% KV 缓存预算**下将平均准确率提升 **`+2.8%` 至 `+6.4%`**。
* **计算开销近乎为零**： $\Vert V _ j - O _ i\Vert _ 2^2 = \Vert V _ j\Vert _ 2^2 - 2 \langle V _ j, O _ i \rangle + \Vert O _ i\Vert _ 2^2$ 可直接复用 FlashAttention 已经算出的输出向量 $O _ i$ ，无需显式物化完整的 $S _ q \times S _ k$ 矩阵，Prefill 延迟增加小于 `1.2%`。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发
1. **对我们 `vla-dtr` & `Efficient Ads / HisTrim` 中 `Exclude-Self Value-Space Perpendicular KV Pruning` 的精确二阶理论证明！**
   * 请仔细对比 OBCache 的核心公式 $\Omega _ j^{KV} = \frac{1}{2}\sum _ i \left(\frac{A _ {i,j}}{1 - A _ {i,j}}\right)^2 \Vert V _ j - O _ i\Vert _ 2^2$ 与我们在 `vla-dtr`（定律 5）和 `ads-rsi` 中独立提出的 **`Exclude-Self Value-Space Perpendicular VLM KV Pruning`**：
     * 其中的因子 $\frac{A _ {i,j}}{1 - A _ {i,j}}$ 正是**排除自身注意力权重后的重归一化系数（Exclude-Self Renormalization）**！
     * 其中的 $\Vert V _ j - O _ i\Vert _ 2^2$ 度量的正是第 $j$ 个 Token 的 Value 向量相对于当前聚合输出均值 $O _ i$ 的**偏离能量（即正交/非共线奇异度）**！如果 $V _ j \approx O _ i$ （即该 Token 的 Value 与上下文均值完全共线/冗余），即便 $A _ {i,j}$ 再大， $\Vert V _ j - O _ i\Vert _ 2^2 \approx 0$ ，驱逐它也完全不改变注意力输出！
2. **落地融合方案（Perp-OBCache）**：
   * 在我们的论文撰写与代码实现中，可以直接引用 ICML 2026 的 OBCache 作为二阶泰勒理论背书，并指出我们进一步将 $\Vert V _ j - O _ i\Vert _ 2^2$ 投影到了输出投影矩阵 $W _ O$ 之后的残差切空间 $\Vert(V _ j - O _ i) W _ O P _ \perp(h _ i)\Vert _ 2^2$ ，从而构成了比 OBCache 更进一层的**流形正交切空间二阶最优脑缓存剪枝（Manifold-Orthogonal OBCache）**。

---

## 🔥 板块二：全球前沿热点精选 (Trending Frontier)

> **赛道锚点**：前沿研发智能体递归自我改进（Agent Harness RSI）、抗过拟合正则化进化、可执行代码物理世界模型（Code as Worlds）。

---

> [!TIP]
> **🎯 `SparseUnifiedModel` 仓库代码级落地点 (`Target Module`)**：`sparse_umm/` (`Shwai-He/SparseUnifiedModel` & `Shwai-He/VLM-Compression`)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-27_ai_paper_notes.md`


---

### 2.10 [2026-09-27] Code as Worlds: Agentic Discovery of Executable World Representations for Physical Reasoning

* **论文信息**：Hanyang Wang, Yimo Cai, Weiliang Chen et al. (`arXiv:2608.27549`, 2026-08, 清华大学 & 智源研究院 BAAI)
* **核心关键词**：Code as Worlds、Executable World Models、Abductive Physical Reasoning、Render-and-Compare Loop、VLM Physical Supervision

#### 📐 架构与核心算法流程图 (ASCII Blueprint)

```text
+-----------------------------------------------------------------------------------+
|       Code as Worlds: Agentic Abductive Discovery of Executable World Models      |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  Observed Video Frames I_{1:T} ---> [1. VLM Abductive Proposer \pi_\theta]        |
|                                                    |                              |
|                             Synthesize Executable Physics Script C^{(k)}          |
|                             (Rigid/Fluid Params, Gravity, Friction, Initial Vel)  |
|                                                    v                              |
|  +-----------------------------------------------------------------------------+  |
|  | 2. Deterministic Physics Engine & Differentiable/Symbolic Renderer          |  |
|  |    Execute C^{(k)} ---> Simulated Trajectories \hat{S}_{1:T}^{(k)} & Frames |  |
|  +-----------------------------------------------------------------------------+  |
|                                                    |                              |
|                                                    v                              |
|  +-----------------------------------------------------------------------------+  |
|  | 3. Spatio-Temporal Discrepancy Feedback (时空残差诊断与代码迭代修正)        |  |
|  |    \Delta_{1:T}^{(k)} = Compare(I_{1:T}, \hat{I}_{1:T}^{(k)})               |  |
|  |    Refine C^{(k+1)} <- \pi_\theta(C^{(k)}, \Delta_{1:T}^{(k)}) until < \epsilon|
|  +-----------------------------------------------------------------------------+  |
|                                                    |                              |
|                                                    v                              |
|       Verified Executable World C^* ---> Counterfactual Rollout & VLM Training    |
+-----------------------------------------------------------------------------------+
```

#### 🎯 背景与痛点 (Background & Pain Points)
* **像素级生成世界模型的“物理幻觉”与定量推理失能**：以 Sora、Wan2.1 或潜空间扩散模型为代表的隐式视频世界模型（Implicit Pixel/Latent World Models）虽然能生成视觉逼真的视频，但缺乏精确的牛顿力学、动量守恒与碰撞几何约束，无法回答诸如“若将斜面摩擦系数减半，滑块将在第几秒撞击挡板？”等定量反事实物理推理问题。
* **纯文本思维链（CoT）无法闭环验证连续动力学**：多模态大模型（VLMs）在面对真实视频时，仅凭自然语言 CoT 极易在估算初速度、质量比与弹性恢复系数时产生累积误差，且缺乏与视觉观测对齐的闭环验证手段。

#### 💡 核心方法与数学公式 (Core Methodology & Math)
1. **作为可执行程序的世界表示（World-as-Executable-Program）**：
   将观测到的物理场景视频 $I _ {1:T}$ 背后的隐状态世界建模为一段参数化的可执行物理仿真程序 $C = (\mathcal{O}, \Theta _ {\text{phys}}, f _ {\text{dyn}})$ ，其中 $\mathcal{O}$ 为几何实体集合， $\Theta _ {\text{phys}} = \lbrace m _ i, \mu _ i, e _ i, \mathbf{v} _ {i,0}\rbrace$ 为连续物理参数（质量、摩擦系数、恢复系数、初速度）， $f _ {\text{dyn}}$ 为确定性物理求解器（如 Box2D / MuJoCo / Blender Python API）。
2. **溯因推理智能体发现闭环（Abductive Discovery Loop）**：
   寻找最能解释观测视频 $I _ {1:T}$ 的可执行代码世界 $C^\star$ 被形式化为最大后验（MAP）逆问题：

$$
C^\star = \arg\max _ {C \in \mathcal{C}} \log P(I _ {1:T} \mid \text{Render}(\text{Sim}(C))) + \log P _ {\text{prior}}(C)
$$

   智能体通过 $K$ 步迭代完成溯因搜索：在第 $k$ 步，执行当前代码假设 $C^{(k)}$ 获得仿真轨迹 $\hat{\mathbf{x}} _ {1:T}^{(k)} = \text{Sim}(C^{(k)})$ ，并与从真实视频提取的目标追踪轨迹 $\mathbf{x} _ {1:T}^{\text{obs}}$ 计算时空运动学残差（Kinematic Discrepancy）：

$$
\mathcal{L} _ {\text{kin}}(C^{(k)}) = \sum _ {t=1}^T \Big( \Vert \hat{\mathbf{x}} _ t^{(k)} - \mathbf{x} _ t^{\text{obs}} \Vert _ 2^2 + \lambda _ v \Vert \hat{\mathbf{v}} _ t^{(k)} - \mathbf{v} _ t^{\text{obs}} \Vert _ 2^2 \Big)
$$

   智能体将结构化残差诊断报告（例如：“仿真物体在第 1.2s 碰撞后反弹高度偏低 18%，表明恢复系数 $e$ 被低估”）反馈给代码生成策略 $\pi _ \theta(C^{(k+1)} \mid C^{(k)}, \nabla \mathcal{L} _ {\text{kin}})$ ，实现符号结构与连续物理参数的联合修正。

#### 📊 关键实验与结论 (Key Experiments & Takeaways)
* **定量物理推理与反事实预测大幅领先**：在涵盖刚体碰撞、流体倾倒、多摆耦合及遮挡轨迹预测的物理推理基准（PhysBench、CLEVRER、ComPhy）上，**Code as Worlds** 将开源与闭源顶级 VLM 的定量物理问答准确率从 `46.2%` 大幅提升至 **`78.9%`**（`+32.7%`）。
* **可扩展合成数据飞轮**：利用智能体自主发现并验证通过的可执行代码世界 $C^\star$ ，可通过扰动代码中的物理参数自动合成数十万条具备 100% 精确物理真值的反事实推理轨迹，用于蒸馏训练轻量级 VLM，使其单次前向推理能力显著跃升。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发
* **对我们 `Physical AI / VLA Loop & Distillation`（`vla-distillation` & `vla-loop`）的合成数据飞轮启发**：
  * 我们在 `vla-distillation`（定律 G27 v3：Quad-Pillar Data-RSI Co-Design）与 `vla-dtr`（定律 G21-G23：Bi-Modal Online Data-RSI Curriculum）中核心强调了利用高保真专家轨迹与困难状态重采样（DAgger / Rollout）来消除 1-NFE / 3-NFE 蒸馏的分布偏移。结合 **Code as Worlds** 的思想，我们可以让智能体从失败的机器人操作视频中逆向合成可执行 MuJoCo/LIBERO 场景配置代码，在接触边界（Contact-Rich Boundary）附近自动进行参数微扰重采样，为 Looped VLA 提供零人工标注成本的边界物理对抗课程！

---

> [!TIP]
> **🎯 `SparseUnifiedModel` 仓库代码级落地点 (`Target Module`)**：`sparse_umm/` (`Shwai-He/SparseUnifiedModel` & `Shwai-He/VLM-Compression`)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-27_ai_paper_notes.md`


---

### 2.11 [2026-09-26] 🤖 *VLA-Pruner: Temporal-Aware Dual-Level Visual Token Pruning for Efficient Vision-Language-Action Inference*
> **聚焦领域**：Vision-Language-Action (VLA) · Embodied AI · Visual Token Pruning · Temporal Consistency  
> **arXiv**：[`arXiv:2511.16449`](https://arxiv.org/abs/2511.16449)

```
  连续控制帧视觉流 ──► [ 层级一 (Prefill): 跨模态指令-视觉语义重要度评估 ]
                                           │
                                           ▼
                       [ 层级二 (Decode): 时域指数平滑动作相关性追踪 S_t = λS_{t-1} + (1-λ)A_t ]
                                           │
                                           ▼
                       [ Combine-then-Filter 联合剪枝: 避免浅层误删关键操控锚点 ]
```

#### 🎯 背景与痛点剖析 (Problem Statement)
* **“语义显著性”与“动作控制必要性”的错位（Semantic-Action Gap）**：在机械臂精细操控任务（如 LIBERO）中，单帧静态视觉编码器认为显著的背景物体，未必是当前动作步（Action Chunk）夹爪需要接触的目标；反之，若在浅层仅凭静态视觉注意力盲目丢弃大量 Patch Token，会导致深层 Action Expert 丢失空间几何锚点，引发轨迹剧烈抖动。

#### 💡 核心方法与数学实现 (Mathematical Formulations)
1. **双层重要度融合准则 (Combine-then-Filter Dual-Level Criterion)**：
   - 同时提取语言指令在 Prefill 阶段对第 $i$ 个视觉 Token 的语义关注度 $I _ {\text{sem}}^{(i)}$ ，以及解码器生成动作 Token 时的交叉注意力得分 $I _ {\text{act}, t}^{(i)}$ ；
2. **跨时间步动作相关性平滑 (Temporal Action Smoothing)**：
   - 利用连续控制帧之间的时间连续性，引入历史动作注意力动量缓存：

$$
\tilde{I} _ {\text{act}, t}^{(i)} = \lambda \tilde{I} _ {\text{act}, t-1}^{(i)} + (1 - \lambda) I _ {\text{act}, t}^{(i)}
$$

   - 仅保留综合得分 $S _ t^{(i)} = I _ {\text{sem}}^{(i)} \cdot \tilde{I} _ {\text{act}, t}^{(i)}$ 最高的视觉 Token 子集。

#### 📊 关键实验与结论 (Experiments & Findings)
* 在 OpenVLA 与主流机器人操控基准（LIBERO-Spatial / Object / Goal / Long）上，剔除 **50%–75% 视觉 Token** 仍保持与全量 Token 持平的任务成功率，端到端控制频率显著提升。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发 (Relevance & Synergy with Our Works)
* **🎯 锚定代表作与在研主线**：
  * [Active Line: *Physical AI (`VLADrop` / `DTR` / `HiSTrim` Exclude-Self Value-Space Perp KV256)*]
  * [Paper #8: *Understanding and Harnessing Sparsity for Unified Multimodal Models* (TMLR 2026)]
  * [Paper #9: *Uncovering the Redundancy in Transformers via Layer Dropping* (TMLR 2025)]
* **🔬 机理对比与技术演进**：
  * 我们在 W38 周记（9/15–9/17）中深刻总结了两条核心定律：（1）**Layer 0（纯 ID Embedding、尚未经过上下文交互）绝不能直接做激进 Token Drop**，必须在表征充分上下文化之后再按浅层保守、深层激进的曲线压缩；（2）**VLA 的鲁棒性来源于三个时间尺度的“伤口愈合（Wound Healing）”纠错通道**（步内注意力、步间去噪、episode 内周期性视觉重锚）；
  * `VLA-Pruner` 的时域平滑动量 $\tilde{I} _ {\text{act}, t}$ 恰恰显式利用了我们指出的第三层“episode 内时域连续重锚”特性！
* **💡 下一阶段研究（Next Research Directions）落地启发**：
  * 在 `Physical AI` (MLSys) 论文中，可将 `VLA-Pruner` 纳入 Related Work 与对比讨论，突出我们 **全栈四维协同压缩（数据 DTR + Token `HiSTrim` + 层 `VLADrop/Loop` + 步数 `SnapFlow` 单步蒸馏）** 相比单一视觉 Token 剪枝在真实硬件延迟（Batch=1 访存带宽瓶颈）上的系统级代差优势。

---

## 🔥 板块二：全球前沿热点精选 (Trending Frontier)

---

> [!TIP]
> **🎯 `SparseUnifiedModel` 仓库代码级落地点 (`Target Module`)**：`sparse_umm/token_pruning.py` (Temporal Motion Prompt + Action-Aware Token Pruning)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-26_ai_paper_notes.md`


---

### 2.12 [2026-09-25] Fully Looped Transformer: Stabilizing Looped Models via Attention Injection and Residual Scaling

* **论文信息**：`arXiv:2605.18797` (2026-05)
* **核心关键词**：Fully Looped Transformer、Attention Injection、Anchor KV Grounding、Gradient Oscillation Prevention

#### 📐 架构与核心算法流程图 (ASCII Blueprint)

```text
+-----------------------------------------------------------------------------------+
|       Fully Looped Transformer with Parameter-Free Initial Attention Injection    |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  Initial Pass (k=0): Input Embedding H^{(0)} ---> Compute Anchor (K^{(0)}, V^{(0)})|
|                                        |                                          |
|                                        v                                          |
|  Loop Iteration k = 1 .. K:                                                       |
|  +-----------------------------------------------------------------------------+  |
|  | 1. Anchor-Injected Multi-Head Attention (零参数初始锚点键值注入)            |  |
|  |    \tilde{K}^{(k)} = (1 - \lambda_k) K^{(k)} + \lambda_k K^{(0)}            |  |
|  |    \tilde{V}^{(k)} = (1 - \lambda_k) V^{(k)} + \lambda_k V^{(0)}            |  |
|  |    Prevents representation drift & provides direct gradient highway to k=0  |  |
|  +-----------------------------------------------------------------------------+  |
|                                        |                                          |
|                                        v                                          |
|  +-----------------------------------------------------------------------------+  |
|  | 2. Unit-Sphere / Variance-Preserving Residual Update                        |  |
|  |    H^{(k+1)} = \text{Norm}\big( H^{(k)} + \frac{1}{\sqrt{K}} f_\theta(H^{(k)}, \tilde{K}^{(k)}, \tilde{V}^{(k)}) \big)|
|  +-----------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------+
```

#### 🎯 背景与痛点 (Background & Pain Points)
* **深层循环中的“初始锚点遗忘”与反向传播雅可比谱半径失控**：当一个循环 Transformer 连续迭代 $K \ge 8$ 步时，第 $k$ 步的隐状态 $H^{(k)}$ 经过反复的非线性自注意力和 FFN 变换后，逐渐丢失了原始输入 Token 的精细词法锚点信息；同时在反向传播（BPTT）中，共享权重连乘 $\prod _ {k=1}^K \big(I + \frac{\partial f _ \theta}{\partial H^{(k)}}\big)$ 极易引发梯度震荡或消失。

#### 💡 核心方法与数学公式 (Core Methodology & Math)
1. **零参数初始注意力注入（Parameter-Free Attention Injection）**：
   缓存首轮（ $k=0$ ）计算得到的初始键值张量 $\left(K^{(0)}, V^{(0)}\right)$ 。在后续任意第 $k \in \lbrace1, \dots, K\rbrace$ 次循环中，通过凸组合或拼接将初始锚点注入当前步的注意力键值中：

$$
O^{(k)} = \text{Softmax}\left( \frac{Q^{(k)} \big( (1-\lambda) K^{(k)} + \lambda K^{(0)} \big)^\top}{\sqrt{d _ k}} \right) \Big( (1-\lambda) V^{(k)} + \lambda V^{(0)} \Big)
$$

   这一设计在计算图上为每一个循环步 $k$ 建立了一条直通初始表征 $\left(K^{(0)}, V^{(0)}\right)$ 的**一阶梯度短路高速通道（Direct Gradient Highway）**：

$$
\frac{\partial \mathcal{L}}{\partial H^{(0)}} = \frac{\partial \mathcal{L}}{\partial H^{(K)}} \prod _ {k=1}^K J _ k + \lambda \sum _ {k=1}^K \frac{\partial \mathcal{L}}{\partial O^{(k)}} \frac{\partial O^{(k)}}{\partial (K^{(0)}, V^{(0)})} \frac{\partial (K^{(0)}, V^{(0)})}{\partial H^{(0)}}
$$

   从而彻底消除了高循环步数下的梯度消失与震荡！

#### 📊 关键实验与结论 (Key Experiments & Takeaways)
* 在完全不增加任何额外参数（0 Extra Parameters）的条件下，Fully Looped Transformer 在 $K=8, 12$ 步循环预训练中完全消除了传统 Looped Transformer 的梯度尖峰（Gradient Spikes），验证集困惑度（PPL）降低 **`1.45`**，下游推理基准提升 **`+4.9%`**。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发
* **直接印证我们 `vla-loop` 定律（Lightweight Dropped-Span VLM Cross-KV Grounding）！**
  * 我们在 `vla-loop` 中发现，当动作专家循环迭代 $K=3,4$ 步时，若每一步都强绑回初始锚点 VLM Prefix KV（即此处的 $\left(K^{(0)}, V^{(0)}\right)$ ），即可完美阻止循环轨迹漂移！该论文的梯度短路公式为我们 `vla-loop` 的 Cross-KV Grounding 提供了极其漂亮的反向传播雅可比谱稳定性证明。

---

> [!TIP]
> **🎯 `SparseUnifiedModel` 仓库代码级落地点 (`Target Module`)**：`sparse_umm/` (`Shwai-He/SparseUnifiedModel` & `Shwai-He/VLM-Compression`)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-25_ai_paper_notes.md`


---

### 2.13 [2026-09-24] LearnPruner: Two-Stage Differentiable Visual Token Pruning for Large Vision-Language Models

* **论文信息**：`arXiv:2604.23950` (2026-04)
* **核心关键词**：Two-Stage Visual Token Pruning、Differentiable Gumbel/Sigmoid Masking、Shallow Deduplication & Deep Grounding

#### 📐 架构与核心算法流程图 (ASCII Blueprint)

```text
+-----------------------------------------------------------------------------------+
|       LearnPruner: Two-Stage Differentiable Visual Token Pruning for LVLMs        |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  Visual Patch Tokens V^{(0)} (N_v = 576)                                          |
|          |                                                                        |
|          v                                                                        |
|  +-----------------------------------------------------------------------------+  |
|  | Stage 1 (Shallow Layer l_1): Vision-Intrinsic Redundancy Pruning            |  |
|  |    Removes background & spatially homogeneous patches BEFORE cross-modal    |  |
|  |    stabilizes -> Retains N_1 tokens                                         |  |
|  +-----------------------------------------------------------------------------+  |
|          |                                                                        |
|          v                                                                        |
|  +-----------------------------------------------------------------------------+  |
|  | Stage 2 (Mid Layer l_2): Instruction-Grounded Cross-Modal Pruning           |  |
|  |    Prunes task-irrelevant objects using stabilized text-to-vision attention |  |
|  |    Differentiable Soft-to-Hard Attention Bias: A_{i,j} + \log m_j(\tau)     |  |
|  +-----------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------+
```

#### 🎯 背景与痛点 (Background & Pain Points)
* **单阶段过早剪枝的“跨模态盲视”与过晚剪枝的“算力浪费”**：若在极浅层（如第 2 层）就仅凭文本指令去剪除大量视觉 Token，此时文本与视觉表征尚未完成跨模态对齐，极易误删目标物体；而若等到第 16 层才剪枝，前 16 层已经消耗了超过 50% 的全量视觉 FLOPs。

#### 💡 核心方法与数学公式 (Core Methodology & Math)
1. **浅层视觉内生去重 + 中层指令对齐聚焦的两阶段架构**：
   在浅层 $l _ 1$ ，仅基于视觉自注意力与空间局部方差剔除纯背景冗余块（保留率 $\rho _ 1 \approx 50$ %）；在中层 $l _ 2$ ，利用已对齐的跨模态交互特征进一步筛选与指令强相关的核心块（保留率 $\rho _ 2 \approx 15$ %）。
2. **注意力对数掩码软硬退火（Differentiable Log-Mask Annealing）**：
   训练期将连续重要性得分 $s _ j \in (0, 1)$ 通过温度 $\tau$ 转化为软掩码 $m _ j(\tau) = \sigma\big((s _ j - \theta _ {\text{thr}})/\tau\big)$ ，并以对数偏置注入注意力矩阵：

$$
\tilde{A} _ {i, j} = \frac{m _ j(\tau) \exp(q _ i^\top k _ j / \sqrt{d _ k})}{\sum _ {r} m _ r(\tau) \exp(q _ i^\top k _ r / \sqrt{d _ k})}
$$

   随着 $\tau \to 0^+$ ， $m _ j(\tau) \to \lbrace0, 1\rbrace$ ，训练期软注意力平滑收敛至推理期的物理硬剔除，实现零训练-推理鸿沟。

#### 📊 关键实验与结论 (Key Experiments & Takeaways)
* 在 **LLaVA-1.5/NeXT** 与 **Qwen2-VL** 上，LearnPruner 仅保留 **11.1%–16.7% 视觉 Token**，FLOPs 降低 **68%**，在 10 项多模态基准上的平均精度达到全 Token 模型的 **99.6%**。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发
* **与我们 *Sparsity for Unified Multimodal Models* (TMLR 2026) & *Demystifying When Pruning Works via Representation Hierarchies* (ICML 2026) 的完美契合**：验证了根据表征层级演化阶段（浅层模态内去重 vs. 中层跨模态语义聚焦）分阶段设置不同剪枝准则的必要性。

---

> [!TIP]
> **🎯 `SparseUnifiedModel` 仓库代码级落地点 (`Target Module`)**：`sparse_umm/token_pruning.py` (Two-Stage Visual Deduplication + Differentiable Mask)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-24_ai_paper_notes.md`


---

### 2.14 [2026-09-24] MixKV: Balancing Importance and Diversity for Modality-Specific KV Cache Compression

* **论文信息**：`arXiv:2510.20707` (2025/2026)
* **核心关键词**：Importance-Diversity Trade-off、Modality-Specific KV Compression、Cosine Repulsion Selection

#### 📐 架构与核心算法流程图 (ASCII Blueprint)

```text
+-----------------------------------------------------------------------------------+
|       MixKV: Balancing Importance and Diversity in Multimodal KV Compression      |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  Visual KV Cache (High Spatial Redundancy) vs. Text KV Cache (High Info Density)  |
|                                        |                                          |
|                                        v                                          |
|  +-----------------------------------------------------------------------------+  |
|  | Modality-Adaptive Submodular Selection Objective                            |  |
|  |    \max_{S: |S|=B} \sum_{i \in S} \text{Imp}(i) - \lambda_{\text{mod}} \sum_{i,j \in S} \cos(K_i, K_j)|
|  |    * Vision modality: High \lambda_{\text{vis}} avoids picking 50 tokens    |  |
|  |      from the same salient foreground patch                                 |  |
|  +-----------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------+
```

#### 🎯 背景与痛点 (Background & Pain Points)
* **纯重要性排序在视觉模态上的“局部高光扎堆陷阱”**：在多模态长上下文中，视觉特征具有极强的空间局部相关性。若仅按注意力得分 Top- $B$ 挑选视觉 KV，预算内的 $B$ 个槽位会被画面中心最显著物体的几十个高度相似的相邻图像块占满，而画面边缘的关键次要物体则被完全清空。

#### 💡 核心方法与数学公式 (Core Methodology & Math)
1. **模态自适应重要性-多样性边际增益准则（Modality-Adaptive Marginal Gain）**：
   在贪心或分块并行选择保留集 $S$ 时，第 $j$ 个候选 Token 的综合得分为其注意力重要性减去其与已选集合在 Key/Value 空间的最大余弦冗余度：

$$
\Phi(j \mid S) = s _ {\text{imp}}(j) - \lambda _ m \cdot \max _ {i \in S} \left( \frac{\langle K _ j, K _ i \rangle}{\Vert K _ j\Vert _ 2 \Vert K _ i\Vert _ 2} \right)
$$

   其中视觉模态的排斥权重 $\lambda _ {\text{vis}} > \lambda _ {\text{text}}$ ，根据各层模态内平均余弦相似度自动校准。

#### 📊 关键实验与结论 (Key Experiments & Takeaways)
* 在 **MileBench**、**Video-MME** 与多图长上下文评测中，MixKV 在 **10% 极限缓存预算**下比 SnapKV 与 PyramidKV 平均提升 **`+5.3%`**。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发
* **与我们 `vla-dtr` 和 `Efficient Ads / HisTrim` 的正交子空间选择完全一致**：在多视角机器人相机或长用户历史序列中，通过 Gram-Schmidt 正交投影排斥共线项，正是最大化子空间体积（Determinantal Point Process）的快速实现。

---

## 🔥 板块二：全球前沿热点精选 (Trending Frontier)

---

> [!TIP]
> **🎯 `SparseUnifiedModel` 仓库代码级落地点 (`Target Module`)**：`sparse_umm/kv_compression.py` (Modality-Adaptive Importance × Diversity KV Eviction)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-24_ai_paper_notes.md`


---

### 2.15 [2026-09-24] AEWM: Agent-Editing World Model with Inference-Time Action Judge and State Revision

* **论文信息**：`arXiv:2609.28416` (2026-09)
* **核心关键词**：Agent-Editing World Model、Inference-Time State Revision、Action Judge、Latent Trajectory Correction

#### 📐 架构与核心算法流程图 (ASCII Blueprint)

```text
+-----------------------------------------------------------------------------------+
|       AEWM: Agent-Editing World Model (Action Judge & Inference State Revision)   |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  Agent State z_t ---> Propose Candidate Action a_t ---> World Model Predicts \hat{z}_{t+1}|
|                                                                |                  |
|                                                                v                  |
|  +-----------------------------------------------------------------------------+  |
|  | 1. Action Judge J_\phi(z_t, a_t, \hat{z}_{t+1})                             |  |
|  |    Detects dead-ends, safety violations, or sub-goal regression BEFORE exec |  |
|  +-----------------------------------------------------------------------------+  |
|                                                                |                  |
|                                             If Judge Score < \tau_{\text{pass}}   |
|                                                                v                  |
|  +-----------------------------------------------------------------------------+  |
|  | 2. Active State Revision Operator \mathcal{E}_\psi(z_t, \hat{z}_{t+1})      |  |
|  |    Edits internal memory/belief state z_t -> z_t^{\text{revised}} to prune  |  |
|  |    corrupted assumptions and resample clean action a_t^*                    |  |
|  +-----------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------+
```

#### 🎯 背景与痛点 (Background & Pain Points)
* **仅重采样动作无法清除已污染的内部记忆状态**：在长程 Web 操作或代码修复任务中，当智能体的内部信念/上下文记忆 $z _ t$ 已经混入了错误的假设时，单纯利用世界模型拒绝当前动作并从同一状态 $z _ t$ 重新采样，依然会反复生成同类的错误动作。

#### 💡 核心方法与数学公式 (Core Methodology & Math)
1. **基于反事实进度判别的内部状态编辑算子（Counterfactual State Revision）**：
   当世界模型预测下一状态 $\hat{z} _ {t+1} = f _ {\text{WM}}(z _ t, a _ t)$ 未能通过动作评判器 $J _ \phi(z _ t, a _ t, \hat{z} _ {t+1}) < \tau$ 时，触发状态编辑器 $\mathcal{E} _ \psi$ 直接在信念状态/工作记忆上施加反事实修正增量：

$$
z _ t^{\text{rev}} = z _ t + \mathcal{E} _ \psi\big( z _ t, a _ t, \hat{z} _ {t+1}, \nabla _ {z _ t} J _ \phi(z _ t, a _ t, \hat{z} _ {t+1}) \big)
$$

   随后基于修正后的干净状态 $z _ t^{\text{rev}}$ 重新生成可执行动作 $a _ t^\star \sim \pi _ \theta(\cdot \mid z _ t^{\text{rev}})$ 。

#### 📊 关键实验与结论 (Key Experiments & Takeaways)
* 在 **VisualWebArena**、**OSWorld** 与长程具身任务上，AEWM 将不可逆错误操作率降低 **52%**，端到端任务成功率比无状态编辑的 Tree-of-Thoughts 高出 **`+10.8%`**。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发
* **对我们 `vla-loop` 动态循环早停与修正（Bridge-Readout Dynamic Halting）的启发**：在循环迭代中若检测到预测轨迹能量异常，可通过低秩正交校正算子直接修正潜状态而非盲目增加循环次数。

---

> [!TIP]
> **🎯 `SparseUnifiedModel` 仓库代码级落地点 (`Target Module`)**：`sparse_umm/` (`Shwai-He/SparseUnifiedModel` & `Shwai-He/VLM-Compression`)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-24_ai_paper_notes.md`


---

### 2.16 [2026-09-23] RT-VLA: Real-Time Vision-Language-Action Models via Knowledge Distillation

* **论文信息**：`arXiv:2606.14010` (2026-06)
* **核心关键词**：Real-Time VLA、Cross-Architecture Knowledge Distillation、Visual-Action Feature Alignment

#### 📐 架构与核心算法流程图 (ASCII Blueprint)

```text
+-----------------------------------------------------------------------------------+
|       RT-VLA: Real-Time Vision-Language-Action Model via Knowledge Distillation   |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  Heavy Teacher VLA (7B VLM + Multi-Step Action Head)                              |
|        |                                      |                                   |
|        | Intermediate Visual-Language         | Continuous Action Trajectory      |
|        | Relational Affinity Matrix           | Distribution Supervision          |
|        v                                      v                                   |
|  Compact Student RT-VLA (Sub-1B Vision-Language Backbone + Lightweight Head)      |
|  ===> 44.8x Faster Vision-Mode Inference & High-Frequency Closed-Loop Control     |
+-----------------------------------------------------------------------------------+
```

#### 🎯 背景与痛点 (Background & Pain Points)
* **7B+ 视觉语言骨干限制了边缘端机器人的板载部署**：主流通用 VLA（如 OpenVLA、 $\pi _ 0$ ）依赖 3B–7B 的 VLM 主干处理每帧高分辨率图像，在车载或机载边缘 GPU 上单帧推理高达数百毫秒。

#### 💡 核心方法与数学公式 (Core Methodology & Math)
1. **关系亲和矩阵蒸馏 + 动作分布联合对齐**：
   由于教师与学生主干隐藏维度不同（ $d _ T \neq d _ S$ ），RT-VLA 不做刚性逐元素回归，而是对齐归一化特征余弦关系矩阵 $G^{(T)} = \tilde{H} _ T \tilde{H} _ T^\top \in \mathbb{R}^{N \times N}$ 与动作输出：

$$
\mathcal{L} _ {\text{RT-VLA}} = \big\Vert \pi _ S(x) - \pi _ T(x) \big\Vert _ 1 + \lambda _ {\text{rel}} \left\lVert \frac{H _ S H _ S^\top}{\Vert H _ S H _ S^\top\Vert _ F} - \frac{H _ T H _ T^\top}{\Vert H _ T H _ T^\top\Vert _ F} \right\rVert _ F^2
$$

#### 📊 关键实验与结论 (Key Experiments & Takeaways)
* 在机器人操作基准上，RT-VLA 将纯视觉模式下的编码与推理耗时降低 **44.8x**，端到端帧率突破 **60 Hz**，同时保留了 7B 教师模型 **96% 以上** 的任务成功率。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发
* **为我们 `vla-distillation` 提供了极佳的跨尺度关系蒸馏损失项**：可将 $\big\Vert \tilde{H} _ S \tilde{H} _ S^\top - \tilde{H} _ T \tilde{H} _ T^\top \big\Vert _ F^2$ 结合进我们的宽度+深度联合压缩（Tri-Orthogonal G19）中，免除维度对齐投影矩阵的参数开销。

---

## 🔥 板块二：全球前沿热点精选 (Trending Frontier)

---

> [!TIP]
> **🎯 `SparseUnifiedModel` 仓库代码级落地点 (`Target Module`)**：`sparse_umm/` (`Shwai-He/SparseUnifiedModel` & `Shwai-He/VLM-Compression`)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-23_ai_paper_notes.md`


---

### 2.17 [2026-09-23] HiMoE-VLA: Hierarchical Mixture-of-Experts for Generalist Vision-Language-Action Policies

* **论文信息**：`arXiv:2512.05693` (2025/2026)
* **核心关键词**：Hierarchical MoE、Generalist VLA Policy、Task-Skill Decoupled Routing、Gradient Conflict Mitigation

#### 📐 架构与核心算法流程图 (ASCII Blueprint)

```text
+-----------------------------------------------------------------------------------+
|       HiMoE-VLA: Hierarchical Mixture-of-Experts for Generalist VLA Policies      |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  Language Goal + Visual State ---> [Level-1: Task/Embodiment Router G_{\text{task}}]|
|                                           |                                       |
|                   Selects Domain Expert Group \mathcal{G}_m                       |
|                                           v                                       |
|         Proprioception + Local Patch ---> [Level-2: Skill Primitive Router G_{\text{skill}}]|
|                                           |                                       |
|                   Activates Fine-Grained Motor Primitives (Reach / Grasp / Place) |
+-----------------------------------------------------------------------------------+
```

#### 🎯 背景与痛点 (Background & Pain Points)
* **异构本体与多任务联合训练中的“扁平路由混淆”**：在跨机械臂本体、跨数十种操作任务的通用 VLA 训练中，单层扁平 MoE 路由器容易按表层视觉背景而非底层运动学技能聚类，导致不同任务间出现严重的负迁移。

#### 💡 核心方法与数学公式 (Core Methodology & Math)
1. **双层语义-运动解耦条件路由（Bi-Level Semantic-Kinematic Conditional Routing）**：
   高层路由器 $G _ {\text{task}}(c _ {\text{lang}}, I _ {\text{global}})$ 根据语言指令与全局视觉场景选择任务簇 $m \in \lbrace1, \dots, M\rbrace$ ，低层路由器 $G _ {\text{skill}}^{(m)}(s _ {\text{prop}}, I _ {\text{wrist}})$ 根据本体关节状态与腕部相机高频特征在簇内选择动作基元专家 $e \in \mathcal{E} _ m$ ：

$$
P(e \mid x) = \sum _ {m=1}^M G _ {\text{task}}(m \mid c _ {\text{lang}}, I _ {\text{global}}) \cdot G _ {\text{skill}}^{(m)}(e \mid s _ {\text{prop}}, I _ {\text{wrist}}) \cdot \mathbb{I}(e \in \mathcal{E} _ m)
$$

#### 📊 关键实验与结论 (Key Experiments & Takeaways)
* 在跨 50+ 任务的 Open-X Embodiment 与仿真套件上，HiMoE-VLA 比同激活参数量的稠密 VLA 与单层 MoE-VLA 平均成功率提升 **`+8.7%`**。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发
* **与我们 `ads-rsi` 中的 GemTagger 分层路由与 *Router-Tuning* (EMNLP 2025) 高度契合**：将高层任务上下文路由与底层高频状态路由树状解耦，可大幅提升细粒度专家的专业化纯度。

---

> [!TIP]
> **🎯 `SparseUnifiedModel` 仓库代码级落地点 (`Target Module`)**：`sparse_umm/` (`Shwai-He/SparseUnifiedModel` & `Shwai-He/VLM-Compression`)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-23_ai_paper_notes.md`


---

### 2.18 [2026-09-22] SnapFlow: One-Step Action Generation for Flow-Matching VLAs via Progressive Self-Distillation

* **论文信息**：`arXiv:2604.05656` (2026-04)
* **核心关键词**：Flow-Matching VLA、1-NFE Action Generation、Progressive Self-Distillation、Chord Velocity Matching

#### 📐 架构与核心算法流程图 (ASCII Blueprint)

```text
+-----------------------------------------------------------------------------------+
|      SnapFlow: 1-NFE Action Generation for Flow-Matching VLAs via Self-Distill    |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  VLM Prefix KV Cache (Visual + Language) + Action Noise A_0 ~ N(0, I)             |
|                                        |                                          |
|                                        v                                          |
|  +-----------------------------------------------------------------------------+  |
|  | 1. Two-Step Euler Teacher Chord Construction (两步欧拉教师割线目标构造)     |  |
|  |    a_{t + \Delta t} = a_t + \Delta t \cdot v_{\theta^-}(a_t, t, \Delta t)   |  |
|  |    a_{t + 2\Delta t} = a_{t+\Delta t} + \Delta t \cdot v_{\theta^-}(a_{t+\Delta t}, t+\Delta t, \Delta t)|
|  |    Target Chord Velocity: \bar{u}_{\text{chord}} = \frac{a_{t+2\Delta t} - a_t}{2\Delta t}|
|  +-----------------------------------------------------------------------------+  |
|                                        |                                          |
|                                        v                                          |
|  +-----------------------------------------------------------------------------+  |
|  | 2. Progressive Halving Schedule: N = 16 -> 8 -> 4 -> 2 -> 1 NFE             |  |
|  |    Student predicts single-step jump: \hat{A}_1 = A_0 + v_\theta(A_0, 0, 1) |  |
|  +-----------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------+
```

#### 🎯 背景与痛点 (Background & Pain Points)
* **多步 ODE 动作去噪拖慢具身实时控制频率**：以 $\pi _ 0$ 、 $\pi _ {0.5}$ 与 GR00T 为代表的现代视觉语言动作模型（VLAs）普遍采用条件流匹配（Conditional Flow Matching）动作专家，在推理时需对动作块（Action Chunk $A \in \mathbb{R}^{H \times d _ a}$ ）执行 $N=10$ 步欧拉积分。尽管动作专家本身参数量较小（如 300M），但 10 次串行交叉注意力与 FFN 前向传播占用了超过 65% 的端到端推理延迟。

#### 💡 核心方法与数学公式 (Core Methodology & Math)
1. **步长条件化割线速度场自蒸馏（Step-Conditioned Chord Velocity Self-Distillation）**：
   扩展动作专家网络输入为 $\left(a _ t, t, \delta\right)$ ，其中 $t \in [0, 1)$ 为当前流时刻， $\delta \in \lbrace2^{-k}\rbrace$ 为目标积分跨度（Step Size）。当跨度从 $\delta$ 倍增至 $2\delta$ 时，利用指数移动平均（EMA）目标网络 $\theta^-$ 执行两次半步积分生成割线目标速度（Chord Velocity）：

$$
\tilde{a} _ {t+\delta} = a _ t + \delta \cdot v _ {\theta^-}(a _ t, t, \delta \mid C _ {\text{VLM}})
$$

$$
u _ {\text{chord}}(a _ t, t, 2\delta) = \frac{1}{2} v _ {\theta^-}(a _ t, t, \delta \mid C _ {\text{VLM}}) + \frac{1}{2} v _ {\theta^-}(\tilde{a} _ {t+\delta}, t+\delta, \delta \mid C _ {\text{VLM}})
$$

   最小化单步跨度预测与双步合成割线之间的 Huber/L2 损失：

$$
\mathcal{L} _ {\text{SnapFlow}}(\theta) = \mathbb{E} _ {t, \delta, a _ 0} \Big[ \big\Vert v _ \theta(a _ t, t, 2\delta \mid C _ {\text{VLM}}) - \text{sg}\big(u _ {\text{chord}}(a _ t, t, 2\delta)\big) \big\Vert _ 2^2 \Big]
$$

2. **推理期零迭代一步生成（1-NFE Inference）**：
   当 $\delta = 1, t = 0$ 时，只需单次前向传播即可直接输出完整动作序列 $\hat{a} _ 1 = a _ 0 + v _ \theta(a _ 0, 0, 1 \mid C _ {\text{VLM}})$ 。

#### 📊 关键实验与结论 (Key Experiments & Takeaways)
* 在 **LIBERO**（Spatial / Object / Goal / Long）与真实机械臂双臂操作基准上，SnapFlow 将动作专家推理步数从 10 NFE 压缩至 **1 NFE**，动作生成阶段延迟降低 **8.4x**，端到端控制频率提升 **2.6x**，同时保持了原始 10 步模型 **98.5%** 以上的成功率。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发
* **正是我们 `vla-distillation` 技能库的核心基石之一（定律 G16–G21 & G27 v3）！**
  * 我们在 `vla-distillation` 中已经系统证明：单纯的 SnapFlow 1-NFE 在高曲率接触任务（如 LIBERO-10 长程插拔）中若不配合 **Perp-Directional Decomposition（G20 正交-平行速度场解耦）**、**MeanFlow + IMM + SFP 联合目标（G21）** 以及 **Stage-2 Cumulative Rank-128 Weight Folding（G27 v3）**，会出现约 1.5%–2.5% 的末端精度折损；将 SnapFlow 的渐进弦长目标与我们的四支柱 Data-RSI 协同设计结合，即可在零推理分支开销下实现超越 10-NFE 教师的无损 1-NFE/3-NFE 闭环控制。

---

> [!TIP]
> **🎯 `SparseUnifiedModel` 仓库代码级落地点 (`Target Module`)**：`sparse_umm/` (`Shwai-He/SparseUnifiedModel` & `Shwai-He/VLM-Compression`)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-22_ai_paper_notes.md`


---

### 2.19 [2026-09-22] LightKV: Make Your LVLM KV Cache More Lightweight

* **论文信息**：`arXiv:2605.00789` (2026-05)
* **核心关键词**：LVLM KV Cache Compression、Cross-Modality Message Passing、Prompt-Guided Visual Aggregation

#### 📐 架构与核心算法流程图 (ASCII Blueprint)

```text
+-----------------------------------------------------------------------------------+
|            LightKV: Prompt-Guided Cross-Modality Visual KV Aggregation            |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  Visual Tokens V_{1:N_v} + Text Instruction Tokens T_{1:N_t}                      |
|                                        |                                          |
|                                        v                                          |
|  +-----------------------------------------------------------------------------+  |
|  | 1. Cross-Modality Message Passing Score (文本指令引导的视觉重要性传递)      |  |
|  |    s_i = \frac{1}{N_t} \sum_{j \in \text{Text}} A_{j \to i}^{\text{cross}}  |  |
|  |    Partition V into Anchor Set \mathcal{A} (Top-K) & Redundant Set \mathcal{R}|
|  +-----------------------------------------------------------------------------+  |
|                                        |                                          |
|                                        v                                          |
|  +-----------------------------------------------------------------------------+  |
|  | 2. Soft Bipartite KV Aggregation (二分图软聚合而非硬丢弃)                   |  |
|  |    \tilde{K}_a = K_a + \sum_{r \in \mathcal{R}} W_{a,r} K_r,                |  |
|  |    \tilde{V}_a = V_a + \sum_{r \in \mathcal{R}} W_{a,r} V_r                 |  |
|  +-----------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------+
```

#### 🎯 背景与痛点 (Background & Pain Points)
* **硬丢弃（Hard Eviction）导致的背景空间上下文丢失**：高分辨率多模态模型（LVLM）单张图产生 576–2,304 个视觉 Token。直接硬丢弃低注意力视觉 Token 会抹除背景空间相对位置与全局计数信息（例如数物体个数任务）。

#### 💡 核心方法与数学公式 (Core Methodology & Math)
1. **指令引导的二分图 KV 软合并（Prompt-Guided Bipartite KV Merging）**：
   利用文本指令 Token 对视觉 Token 的跨模态注意力选出锚点集合 $\mathcal{A}$ 与待合并集合 $\mathcal{R}$ 。对于每个被淘汰的视觉 Token $r \in \mathcal{R}$ ，计算其与锚点 $a \in \mathcal{A}$ 在 Key 空间的余弦相似度分布 $W _ {a,r} = \text{Softmax} _ a(\beta \cos(K _ a, K _ r))$ ，并执行注意力守恒的加权合并：

$$
\tilde{K} _ a = \frac{\alpha _ a K _ a + \sum _ {r \in \mathcal{R}} \alpha _ r W _ {a,r} K _ r}{\alpha _ a + \sum _ {r \in \mathcal{R}} \alpha _ r W _ {a,r}}, \qquad \tilde{V} _ a = \frac{\alpha _ a V _ a + \sum _ {r \in \mathcal{R}} \alpha _ r W _ {a,r} V _ r}{\alpha _ a + \sum _ {r \in \mathcal{R}} \alpha _ r W _ {a,r}}
$$

#### 📊 关键实验与结论 (Key Experiments & Takeaways)
* 在 **LLaVA-1.6-34B** 与 **InternVL-2** 上将视觉 KV 缓存直接压缩 **50%–75%**，在 TextVQA、DocVQA 与计数基准上实现 **99.4%** 的原始性能保持率。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发
* **与我们 `vla-dtr` 及 *Sparsity for Unified Multimodal Models* (TMLR 2026) 的结合**：在合并非核心视觉 Token 时，仅合并与锚点平行的背景分量，而将正交运动边缘特征显式保留为独立锚点。

---

## 🔥 板块二：全球前沿热点精选 (Trending Frontier)

---

> [!TIP]
> **🎯 `SparseUnifiedModel` 仓库代码级落地点 (`Target Module`)**：`sparse_umm/kv_compression.py` (Text-Guided Bipartite Soft-Merging of Visual KV)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-22_ai_paper_notes.md`


---

### 2.20 [2026-09-21] RotateK: Rotation-Aligned Key Channel Pruning for Vision-Language Models

* **论文信息**：`arXiv:2605.19218` (2026-05)
* **核心关键词**：Key Channel Pruning、Orthogonal Rotation Alignment、Vision-Language Models (VLMs)、Head-Dimension Compression

#### 📐 架构与核心算法流程图 (ASCII Blueprint)

```text
+-----------------------------------------------------------------------------------+
|       RotateK: Rotation-Aligned Key Channel Pruning for Vision-Language Models    |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  Attention Score Invariance under Orthogonal Rotation R \in O(d_k):               |
|    Q K^\top = (Q R)(K R)^\top   where R^\top R = I_{d_k}                          |
|                                        |                                          |
|                                        v                                          |
|  +-----------------------------------------------------------------------------+  |
|  | 1. Cross-Modal Key-Query Co-Energy SVD (跨模态查询-键联合能量奇异值对齐)    |  |
|  |    Compute covariance C_K = \mathbb{E}[K_{\text{vis}}^\top K_{\text{vis}}]  |  |
|  |    Eigendecompose C_K = R \Lambda R^\top ---> Fold R into W_Q, W_K offline  |  |
|  +-----------------------------------------------------------------------------+  |
|                                        |                                          |
|                                        v                                          |
|  +-----------------------------------------------------------------------------+  |
|  | 2. Tail Channel Truncation (尾部低能量通道截断: 兼容 RoPE 2x2 块旋转)       |  |
|  |    Retain top-r channels (r = 0.4 d_k) -> 60% Key Cache & GEMM Reduction    |  |
|  +-----------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------+
```

#### 🎯 背景与痛点 (Background & Pain Points)
* **原始坐标轴下的通道能量弥散**：在多模态大模型（VLM）中，除序列长度方向（Token 维度）冗余外，注意力头内部的特征维度 $d _ k$ （如 $d _ k=128$ ）在视觉特征空间中实际上具有极低的本征秩。然而，在原始训练得到的正交基下，信号能量均匀弥散在全部 128 个通道上，直接按坐标轴剪除任何通道都会造成较大的内积误差 $\Vert Q K^\top - \tilde{Q} \tilde{K}^\top\Vert _ F$ 。

#### 💡 核心方法与数学公式 (Core Methodology & Math)
1. **RoPE 兼容的分块正交旋转能量集中（RoPE-Compatible Block-Orthogonal Rotation）**：
   由于旋转位置编码（RoPE）以二维子平面 $\left(2i, 2i+1\right)$ 为单位作用： $R _ \Theta(m) = \text{diag}(R _ {\theta _ 1}^{(m)}, \dots, R _ {\theta _ {d _ k/2}}^{(m)})$ ，为保持与 RoPE 的可交换性，RotateK 将 $d _ k/2$ 个二维频率对按预期内积能量贡献 $\mathcal{E} _ i = \mathbb{E}\big[ \Vert q _ {[2i:2i+1]} \Vert _ 2^2 \cdot \Vert k _ {[2i:2i+1]} \Vert _ 2^2 \big]$ 进行重排，并在每个同频子空间内执行正交主轴对齐 $U _ i \in O(2)$ ：

$$
\tilde{W} _ Q = W _ Q U _ {\text{rot}}, \qquad \tilde{W} _ K = W _ K U _ {\text{rot}}
$$

2. **误差上界最小化通道截断**：
   保留能量最高的前 $r$ 个通道子块，此时注意力 logit 截断误差满足紧上界：

$$
\mathbb{E}\big[ | q^\top k - \tilde{q} _ {1:r}^\top \tilde{k} _ {1:r} |^2 \big] \le \sum _ {i = r/2 + 1}^{d _ k/2} \lambda _ i(C _ Q) \lambda _ i(C _ K)
$$

#### 📊 关键实验与结论 (Key Experiments & Takeaways)
* 在 **LLaVA-NeXT**、**Qwen2-VL-7B** 与 **InternVL-2** 上，RotateK 剪除 **50%–60% 的 Key 通道**而无需微调，且与视觉 Token 剪枝（如 FastV / VLA-Pruner）**100% 正交兼容**，联合实现 **4.2x** 注意力加速且 VQA 精度损失 `<0.5%`。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发
* **与我们 `MerA` SVD 初始化及 *Sparsity for Unified Multimodal Models* (TMLR 2026) 的正交协同**：
  * RotateK 在特征通道维度 $d _ k$ 上的正交旋转浓缩与我们在 Token 维度 $N _ {\text{vis}}$ 上的剪枝构成了完整的二维矩阵联合低秩逼近（Row + Column Dual Sparsity），可直接嵌入 `vla-distillation` 的视觉前缀压缩器中。

---

> [!TIP]
> **🎯 `SparseUnifiedModel` 仓库代码级落地点 (`Target Module`)**：`sparse_umm/kv_compression.py` (RoPE-Compatible Block-Orthogonal Key Channel Truncation)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-21_ai_paper_notes.md`


---

### 2.21 [2026-09-21] Self-OPD: On-Policy Distillation for Flow Matching Models without Teacher

* **论文信息**：`arXiv:2608.26872` (2026-08)
* **核心关键词**：Teacher-Free Flow Distillation、Stochastic SDE Branching、All-Branch Pull-Push Objective

#### 📐 架构与核心算法流程图 (ASCII Blueprint)

```text
+-----------------------------------------------------------------------------------+
|       Self-OPD: Teacher-Free On-Policy Distillation for Flow Matching             |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  Student State z_t ---> Branch into K Stochastic SDE Rollouts + 1 Deterministic   |
|                                        |                                          |
|                                        v                                          |
|  +-----------------------------------------------------------------------------+  |
|  | Self-Verifier / Reward Scorer ranks terminal states {z_1^{(1)}, ..., z_1^{(K)}}|
|  +-----------------------------------------------------------------------------+  |
|                                        |                                          |
|                                        v                                          |
|  +-----------------------------------------------------------------------------+  |
|  | All-Branch Pull-Push Velocity Objective                                     |  |
|  |    Pull v_\theta(z_t, t) toward high-reward branches z_1^+                  |  |
|  |    Push v_\theta(z_t, t) away from low-reward branches z_1^-                |  |
|  +-----------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------+
```

#### 🎯 背景与痛点 (Background & Pain Points)
* **少步流匹配蒸馏对超大教师模型的依赖及教师能力上限锁死**：传统流匹配蒸馏必须先训练并常驻一个庞大的多步教师模型，且学生模型的性能永远无法超越教师。

#### 💡 核心方法与数学公式 (Core Methodology & Math)
1. **全分支拉推速度场目标（All-Branch Pull-Push Velocity Objective）**：
   在中间时刻 $t$ ，从当前状态 $z _ t$ 分叉出 $K$ 条带随机扩散项的探索分支 $\lbrace z _ 1^{(k)}\rbrace _ {k=1}^K$ ，根据终端奖励 $r(z _ 1^{(k)})$ 计算归一化优势权重 $A _ k$ ，直接构造自引导目标速度向量：

$$
v _ {\text{target}}(z _ t, t) = \sum _ {k=1}^K \text{Softmax}(\beta A) _ k \frac{z _ 1^{(k)} - z _ t}{1 - t} - \lambda _ {\text{push}} \sum _ {j: A _ j < 0} |A _ j| \frac{z _ 1^{(j)} - z _ t}{1 - t}
$$

#### 📊 关键实验与结论 (Key Experiments & Takeaways)
* 在不加载任何外部教师的情况下，Self-OPD 将 4 步流匹配模型的生成与控制成功率提升 **`+14.2%`**，甚至超越了 50 步原始基准模型。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发
* **对我们 `vla-distillation` 与 `vla-loop` 的启发**：在 VLA 少步动作生成中，可利用物理仿真器的成功/碰撞反馈作为终端奖励 $r(z _ 1^{(k)})$ ，通过 Self-OPD 的全分支拉推速度场目标让 1-NFE / 3-NFE 学生策略超越 10-NFE 模仿学习教师！

---

> [!TIP]
> **🎯 `SparseUnifiedModel` 仓库代码级落地点 (`Target Module`)**：`sparse_umm/flow_generation.py` (On-Policy Trajectory Distillation for Multimodal Flow)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-21_ai_paper_notes.md`


---

### 2.22 [2026-09-21] MoE-FM: Towards Faster Language Model Inference Using Mixture-of-Experts Flow Matching

* **论文信息**：`arXiv:2604.15009` (2026-04)
* **核心关键词**：Mixture-of-Experts Flow Matching、Piecewise-Linear Vector Fields、Latent Flow Language Models

#### 📐 架构与核心算法流程图 (ASCII Blueprint)

```text
+-----------------------------------------------------------------------------------+
|          MoE-FM: Mixture-of-Experts Flow Matching for Fast Inference              |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  Latent State z_t at Time t ---> Time- & State-Conditioned Router G(z_t, t)       |
|                                        |                                          |
|            +---------------------------+---------------------------+              |
|            v                           v                           v              |
|  [Expert Field v_1(z_t,t)]   [Expert Field v_2(z_t,t)]   [Expert Field v_E(z_t,t)]|
|  (Local Straight Transport)  (Local Straight Transport)  (Local Straight Transport)|
|            +---------------------------+---------------------------+              |
|                                        |                                          |
|                                        v                                          |
|            Composite Velocity v(z_t, t) = \sum_{e \in Top-k} g_e(z_t,t) v_e(z_t,t)|
+-----------------------------------------------------------------------------------+
```

#### 🎯 背景与痛点 (Background & Pain Points)
* **全局单一速度场拟合多峰分布时的轨迹弯曲（Trajectory Curvature）**：当使用单个稠密网络拟合高度多模态的语言或动作分布时，不同模式的流线在中间时刻发生交叉，迫使平均速度场严重弯曲，从而需要数十步 ODE 积分才能避免离散化截断误差。

#### 💡 核心方法与数学公式 (Core Methodology & Math)
1. **分片局部直线化的专家向量场分解**：
   将全局速度场 $v(z _ t, t)$ 分解为 $E$ 个局部专家速度场的稀疏组合，并加入专家内轨迹曲率惩罚以促使每个专家负责的局部区域保持直线传输：

$$
\mathcal{L} _ {\text{MoE-FM}} = \mathbb{E} _ {t, z _ 0, z _ 1} \left[ \left\lVert \sum _ {e \in \text{Top-}k} g _ e(z _ t, t) v _ e(z _ t, t) - (z _ 1 - z _ 0) \right\rVert _ 2^2 + \mu \sum _ {e \in \text{Top-}k} g _ e(z _ t, t) \big\Vert \partial _ t v _ e(z _ t, t) \big\Vert _ 2^2 \right]
$$

#### 📊 关键实验与结论 (Key Experiments & Takeaways)
* 在潜空间语言生成与多模态推理中，MoE-FM 在仅使用 **2–4 步 NFE** 时即可达到单稠密流模型 16–32 步的生成质量，推理延迟降低 **3.8x**。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发
* **对我们 `vla-distillation` 多模态动作块（Action Chunk）生成的启发**：机器人操作往往存在“从左侧绕行”或“从右侧抓取”的多峰分叉模式，引入时间与状态联合门控的轻量级 LoRA 专家速度场可有效消除多峰平均导致的直线穿越障碍物问题。

---

> [!TIP]
> **🎯 `SparseUnifiedModel` 仓库代码级落地点 (`Target Module`)**：`sparse_umm/flow_generation.py` (Mixture-of-Flows Piecewise Straight Velocity Fields)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-21_ai_paper_notes.md`


---

### 2.23 [2026-09-20] Flow-OPD: On-Policy Distillation for Flow Matching Models

* **论文信息**：`arXiv:2605.08063` (2026-05)
* **核心关键词**：Flow Matching、On-Policy Distillation、Velocity Field Alignment、Exposure Bias Mitigation

#### 📐 架构与核心算法流程图 (ASCII Blueprint)

```text
+-----------------------------------------------------------------------------------+
|            Flow-OPD: On-Policy Distillation for Flow Matching Models              |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  Noise z_0 ~ N(0,I) ---> Rollout Few-Step Student Trajectory:                     |
|                          \tilde{z}_{t_{k+1}} = \tilde{z}_{t_k} + \Delta t \cdot v_\theta(\tilde{z}_{t_k}, t_k)|
|                                        |                                          |
|                                        v                                          |
|  +-----------------------------------------------------------------------------+  |
|  | 1. On-Policy Teacher Velocity Query (在学生真实轨迹状态上查询教师速度场)    |  |
|  |    Query Frozen Multi-Step Teacher v_{\text{teacher}}(\tilde{z}_{t_k}, t_k) |  |
|  |    Corrects off-manifold drift encountered only during student rollout      |  |
|  +-----------------------------------------------------------------------------+  |
|                                        |                                          |
|                                        v                                          |
|  +-----------------------------------------------------------------------------+  |
|  | 2. Two-Stage Domain-Specialized Teacher Cultivation & Student Orchestration |  |
|  +-----------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------+
```

#### 🎯 背景与痛点 (Background & Pain Points)
* **离线流匹配蒸馏的“轨迹偏离暴露偏差（Off-Manifold Exposure Bias）”**：在将 50 步流匹配（Flow Matching）模型蒸馏为 1–4 步极速学生模型时，传统离线蒸馏仅在教师生成的理想直线插值轨迹 $z _ t = (1-t)z _ 0 + t z _ 1$ 上监督学生。然而在实际少步推理时，学生模型第 1 步的微小离散化误差就会使其落入教师从未示范过的流形外区域（Off-Manifold State），导致后续步骤误差滚雪球式发散。

#### 💡 核心方法与数学公式 (Core Methodology & Math)
1. **学生在线轨迹上的速度场拉回目标（On-Policy Velocity Pull-Back）**：
   令学生少步求解器从高斯噪声 $z _ 0 \sim \mathcal{N}(0, I)$ 出发自回归生成在线状态序列 $\lbrace\tilde{z} _ {t _ k}\rbrace _ {k=0}^{K-1}$ 。在学生真实到达的状态 $\tilde{z} _ {t _ k}$ 处调用教师速度场 $u _ \phi(\tilde{z} _ {t _ k}, t _ k)$ 计算拉回目标：

$$
\mathcal{L} _ {\text{Flow-OPD}}(\theta) = \mathbb{E} _ {z _ 0, k} \Big[ w(t _ k) \big\Vert v _ \theta(\text{sg}(\tilde{z} _ {t _ k}), t _ k) - u _ \phi(\text{sg}(\tilde{z} _ {t _ k}), t _ k) \big\Vert _ 2^2 \Big]
$$

   其中 $\text{sg}(\cdot)$ 表示停止梯度算子，确保学生学会从自身产生的离散化偏移状态中主动修正回真实数据流形。

#### 📊 关键实验与结论 (Key Experiments & Takeaways)
* 在 2 步与 4 步流匹配生成基准上，Flow-OPD 将 FID 与条件指令遵循得分相比离线轨迹蒸馏（Reflow / Progressive Distillation）提升 **`18%–27%`**。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发
* **直接印证我们 `vla-distillation` 定律 G16（MerA-VelLoRA × Closed-Loop DAgger）与 G27 v3**：
  * Flow-OPD 在ODE轨迹内部的状态级 On-Policy Velocity Pull-Back 与我们在 `vla-distillation` 中提出的闭环 DAgger 状态重采样互为“步内（Intra-Chunk）”与“步间（Inter-Chunk）”对偶！将两者结合即可同时消除少步 ODE 离散化漂移与环境交互累积误差。

---

> [!TIP]
> **🎯 `SparseUnifiedModel` 仓库代码级落地点 (`Target Module`)**：`sparse_umm/flow_generation.py` (On-Policy Trajectory Distillation for Multimodal Flow)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-20_ai_paper_notes.md`


---

### 2.24 [2026-09-18] ✂️ *AnchorPrune: Geometry-Preserving Representation Hierarchy Compression for Multimodal Large Language Models*
> **聚焦领域**：Multimodal Sparsity · Representation Hierarchies · Layer Dropping · Geometric Manifolds  
> **arXiv**：[`arXiv:2609.08842`](https://arxiv.org/abs/2609.08842)

```
  多模态隐状态流形 ──► [ 1. 局部几何锚点提取 (Anchor SVD) ] ──► 计算流形重构失真率 D_l
                                     │                                      │
                                     ▼                                      ▼
                      [ 2. 层级表征阶梯贡献判定 ]             [ 3. 联合压缩: 40% 层丢弃 + 50% Token 稀疏 ]
                      判为冗余饱和层 ──► 予以跳过               零微调保留 99.2% MMBench 精度
```

#### 🎯 背景与痛点 (Problem Statement)
多模态大模型在深层网络中存在极高比例的视觉表征冗余。现有的 Token 剪枝与 Layer Dropping 往往割裂进行：若先剪 Token 再丢层，会导致跨模态语义对齐发生断崖式崩塌；若仅做静态层丢弃，浅层大量的背景无用 Token 依然占据巨大的显存与 Attention 算力。

#### 💡 核心方法与原文底层数学实现 (Mathematical Formulations)
1. **多模态局部几何锚点矩阵 (Multimodal Geometric Anchors)**：
   - 在第 $l$ 层提取多模态激活流形 $\mathcal{M} _ l$ 上的代表性锚点子集 $\mathcal{A} _ l = \lbrace a _ 1, a _ 2, \dots, a _ K\rbrace \subset \mathbb{R}^{d}$ ；
   - 求解局部切空间的主成分基底，定义层级几何表征流形失真度指标 $\mathcal{D} _ l$ ：

$$
\mathcal{D} _ l \triangleq \frac{1}{K} \sum _ {k=1}^K \left\lVert a _ k - \Pi _ {\mathcal{A} _ {l-1}}(a _ k) \right\rVert _ 2^2
$$

   - 当 $\mathcal{D} _ l < \tau _ {\text{layer}}$ 时，判定该层为表征阶梯中的平坦饱和层，可安全丢弃。
2. **锚点引导的动态 Token 稀疏过滤 (Anchor-Guided Token Sparsification)**：
   - 仅保留与核心几何锚点内积相似度大于动态阈值的 Token，在浅层过滤掉 50% 以上的无用背景 Patch，同时维持深层关键语义边界。

#### 📊 关键实验与结论 (Experiments & Findings)
* **评估模型**：Qwen2-VL-7B/72B、LLaVA-NeXT-34B；
* **压缩指标**：联合跳过 **40% Transformer 层** 并剔除 **50% 视觉 Token**，无需微调，在 MME、MMBench、ChartQA 上平均精度损失仅 **0.8%**，端到端推理提速 **2.7 倍**，显存峰值降低 **62%**。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发 (Relevance & Synergy with Our Works)
* **🎯 锚定代表作**：
  * [Paper #15: *Demystifying When Pruning Works via Representation Hierarchies* (ICML 2026)]
  * [Paper #8: *Understanding and Harnessing Sparsity for Unified Multimodal Models* (TMLR 2026)]
  * [Paper #9: *Uncovering the Redundancy in Transformers via Layer Dropping* (TMLR 2025)]
* **🔬 机理对比与技术演进**：
  * 我们在 *ICML 26* 与 *TMLR 25* 中奠定了从“表征层级阶梯（Representation Hierarchies）”解释剪枝机理的理论基石；
  * *AnchorPrune* 将我们的层级冗余理论推进到了“层丢弃（Layer Dropping）与 Token 动态稀疏（Token Sparsity）的二维联合优化”，提供了具体的几何锚点判据；
* **💡 下一阶段研究（Next Research Directions）落地启发**：
  * 可直接将锚点流形失真度 $\mathcal{D} _ l$ 集成至我们的多模态轻量化评估脚本中，作为我们后续多模态稀疏化大模型训练的正则化损失函数。

---

> [!TIP]
> **🎯 `SparseUnifiedModel` 仓库代码级落地点 (`Target Module`)**：`sparse_umm/token_pruning.py` (Cross-Modal Geometric Anchor Fidelity Pruning)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-18_ai_paper_notes.md`


---
