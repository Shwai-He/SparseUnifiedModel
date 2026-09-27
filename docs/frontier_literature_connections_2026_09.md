# 👁️ SparseUnifiedModel & VLM-Compression: 每日前沿文献关联与多模态统一稀疏/流加速落地库 (2026-09)

**Document ID:** `SPARSEUMM-LIT-202609` | **Last Updated:** `2026-09-27` | **Target Path:** `docs/frontier_literature_connections_2026_09.md` | **Total Routed Papers:** `16`

> [!IMPORTANT]
> **🔗 跨仓库文献引用链闭环 (Cross-Repository Reference Chain Closure)**
> 本文件由每日 AI 前沿论文精读流水线自动路由生成，专门收录与我们 **TMLR 2026 / ICML 2025 / ECCV 2026 多模态代表作 (*Sparsity for Unified Multimodal Models*, `Shwai-He/SparseUnifiedModel` & `Shwai-He/VLM-Compression`)** 直接关联的跨模态视觉/文本 Token 剪枝（`AnchorPrune`, `LearnPruner`, `VLA-Pruner`, `RT-VLA`）、多模态 KV 缓存压缩（`LightKV`, `RotateK`, `MixKV`）以及多模态流匹配生成加速（`SnapFlow`, `Flow-OPD`, `Self-OPD`, `MoE-FM`）最新 arXiv 论文笔记。
> 每一篇收录文献均包含：**核心痛点、底层数学公式、ASCII 架构图、关键实测指标**，以及**与 `SparseUnifiedModel` 仓库具体代码模块和我们已发表代表作（Our Works）的双向锚定**。

---

## 🌟 1. 核心关联文献与本仓库模块映射速查表 (Executive Reference-to-Module Matrix)

| 收录日期 | 论文标题与 arXiv 链接 | 关键实测收益 / 核心结论 | 锚定本仓库代码模块与文档路径 (`Target Module`) | 原始精读归档 |
| :---: | :--- | :--- | :--- | :---: |
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

### 2.1 [2026-09-27] OBCache: Optimal Brain KV Cache Pruning for Efficient Long-Context LLM Inference

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

### 2.2 [2026-09-27] Code as Worlds: Agentic Discovery of Executable World Representations for Physical Reasoning

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

### 2.3 [2026-09-26] 🤖 *VLA-Pruner: Temporal-Aware Dual-Level Visual Token Pruning for Efficient Vision-Language-Action Inference*
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

### 2.4 [2026-09-25] Fully Looped Transformer: Stabilizing Looped Models via Attention Injection and Residual Scaling

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

### 2.5 [2026-09-24] LearnPruner: Two-Stage Differentiable Visual Token Pruning for Large Vision-Language Models

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

### 2.6 [2026-09-24] MixKV: Balancing Importance and Diversity for Modality-Specific KV Cache Compression

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

### 2.7 [2026-09-24] AEWM: Agent-Editing World Model with Inference-Time Action Judge and State Revision

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

### 2.8 [2026-09-23] RT-VLA: Real-Time Vision-Language-Action Models via Knowledge Distillation

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

### 2.9 [2026-09-23] HiMoE-VLA: Hierarchical Mixture-of-Experts for Generalist Vision-Language-Action Policies

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

### 2.10 [2026-09-22] SnapFlow: One-Step Action Generation for Flow-Matching VLAs via Progressive Self-Distillation

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

### 2.11 [2026-09-22] LightKV: Make Your LVLM KV Cache More Lightweight

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

### 2.12 [2026-09-21] RotateK: Rotation-Aligned Key Channel Pruning for Vision-Language Models

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

### 2.13 [2026-09-21] Self-OPD: On-Policy Distillation for Flow Matching Models without Teacher

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

### 2.14 [2026-09-21] MoE-FM: Towards Faster Language Model Inference Using Mixture-of-Experts Flow Matching

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

### 2.15 [2026-09-20] Flow-OPD: On-Policy Distillation for Flow Matching Models

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

### 2.16 [2026-09-18] ✂️ *AnchorPrune: Geometry-Preserving Representation Hierarchy Compression for Multimodal Large Language Models*
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
