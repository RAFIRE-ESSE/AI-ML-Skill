---
name: ai-ml-engineering
description: Comprehensive AI & Machine Learning Engineering skill providing exhaustive reference guides, architectures, loss functions, metrics, optimization techniques, tabular DL, LLMs/GenAI fine-tuning, computer vision, RAG, time series, recsys, causal inference, and distributed training strategies.
---

# AI & Machine Learning Engineering Skill

This skill provides a complete, production-ready engineering reference for Machine Learning (ML), Deep Learning (DL), Large Language Models (LLMs), Computer Vision (CV), Time Series, Recommendation Systems, Causal Inference, and Generative AI.

---

## MANDATORY AGENT EXECUTION PROTOCOL

When executing any project, feature, or coding task under this skill, the AI agent MUST strictly follow this 4-step workflow:

### Step 1 — Mandatory Local Reference Review
Before writing any code or proposing solutions, the agent MUST review all relevant local skill reference files located inside this skill's `references/` directory:
1. **[Machine Learning Complete Reference](file:///home/devil/Downloads/AI&ML-Skill/.agents/skills/ai-ml-engineering/references/Machine_Learning_Complete_Reference.md)** — Exhaustive guide for preprocessing, scaling, encoders, classical models, time series, RecSys, causal inference, XAI, and metrics.
2. **[Deep Learning Complete Reference](file:///home/devil/Downloads/AI&ML-Skill/.agents/skills/ai-ml-engineering/references/Deep_Learning_Complete_Reference.md)** — Exhaustive guide for activations, loss functions, optimizers, layers, LLM PEFT/alignment, vision backbones, vector DBs, GNNs, and quantization.
3. **[Landmark Papers Master Index](file:///home/devil/Downloads/AI&ML-Skill/.agents/skills/ai-ml-engineering/references/papers/README.md)** — Index of downloaded local paper PDFs (`references/papers/pdf/`) and detailed technical breakdown notes (`references/papers/notes/`).
4. **[ML 1,000 Papers Catalog](file:///home/devil/Downloads/AI&ML-Skill/.agents/skills/ai-ml-engineering/references/papers/ML_1000_Landmark_Papers.md)** & **[DL 1,000 Papers Catalog](file:///home/devil/Downloads/AI&ML-Skill/.agents/skills/ai-ml-engineering/references/papers/DL_1000_Landmark_Papers.md)**.

### Step 2 — Web & SOTA Search
The agent MUST perform a targeted web and academic search (arXiv, GitHub, PapersWithCode) to find the latest, rare, highly efficient, or state-of-the-art algorithms, functions, papers, or techniques relevant to the specific problem domain.

### Step 3 — Structured Implementation Planning
The agent MUST create a detailed implementation plan (`implementation_plan.md`) that:
- Selects the optimal model architecture, loss function, optimizer, and preprocessing pipeline based on local reference manuals and web research.
- Formulates exact mathematical equations for loss functions and activation gates.
- Cites seminal research papers (Author, Year, arXiv ID) for every chosen component.

### Step 4 — Continuous Skill Self-Updating
Whenever any new function, optimizer, activation, loss function, architecture, or research paper is discovered, implemented, or introduced during a project, the agent MUST automatically append/add it to the corresponding skill reference manuals (`references/Machine_Learning_Complete_Reference.md`, `references/Deep_Learning_Complete_Reference.md`, `references/papers/README.md`, or paper catalogs) inside this skill directory for future reuse.

---

## Core Technical Reference Map

- **Classical Machine Learning**: Preprocessing, Encoders, Tree Boosters (XGBoost, LightGBM, CatBoost), Imbalanced Learning (SMOTE/ADASYN), XAI (SHAP, LIME, ALE, DiCE).
- **Deep Learning & Neural Architectures**: Activations (GELU, SwiGLU, Mish), Losses (Focal, Dice, Tversky, InfoNCE, ArcFace), Optimizers (AdamW, Lion, Sophia, SAM), Layers (RoPE, ALiBi, GQA, MoE, Mamba).
- **Generative AI & LLMs**: PEFT (LoRA, QLoRA, DoRA, Prefix-Tuning), Alignment (DPO, KTO, ORPO, RLHF), Serving (vLLM PagedAttention, Continuous Batching, TensorRT-LLM).
- **Computer Vision & Multimodal**: ResNet, ConvNeXt, ViT, Swin, YOLO v1-v10, DETR, SAM-2, Latent Diffusion (Stable Diffusion, ControlNet, LCM), CLIP, DINOv2.
- **Time Series & Sequential**: ARIMA, Prophet, Greykite, STL, DTW, MinT Reconciliation, Sktime, Tsfresh.
- **RecSys & Causal**: SVD++, Implicit ALS, FM/FFM, BPR, Two-Tower Retrieval, PSM, DiD, Causal Forests (`EconML`), CUPED, `DoWhy`.
- **Quantization & Distributed DL**: GGUF, GPTQ, AWQ, BitsAndBytes NF4, FSDP, DeepSpeed ZeRO-1/2/3, Tensor/Pipeline Parallelism.
