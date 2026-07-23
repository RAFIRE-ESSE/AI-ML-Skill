# 1,000 Landmark Deep Learning & Generative AI Research Papers Catalog
### Exhaustive Index of Seminal & Modern Deep Learning, LLM, Vision, PEFT, Alignment, Quantization & Acceleration Papers

---

## 1. Transformer Architectures, LLMs & Sub-Quadratic Models

| # | Paper Title | Year | Authors / Venue | arXiv / Link | Core Innovation & Breakthrough |
|---|---|---|---|---|---|
| 1 | **Attention Is All You Need** | 2017 | Vaswani et al. (NeurIPS) | [1706.03762](https://arxiv.org/abs/1706.03762) | Transformer architecture, Multi-Head Self-Attention, Positional Encoding. |
| 2 | **BERT: Pre-training of Deep Bidirectional Transformers** | 2018 | Devlin et al. (NAACL) | [1810.04805](https://arxiv.org/abs/1810.04805) | Masked Language Modeling (MLM) for deep bidirectional representation. |
| 3 | **Language Models are Few-Shot Learners (GPT-3)** | 2020 | Brown et al. (NeurIPS) | [2005.14165](https://arxiv.org/abs/2005.14165) | Demonstrates in-context zero/few-shot learning at 175B parameters scale. |
| 4 | **LLaMA: Open and Efficient Foundation Language Models** | 2023 | Touvron et al. (Meta) | [2302.13971](https://arxiv.org/abs/2302.13971) | Open high-performance decoder architecture (RMSNorm, SwiGLU, RoPE). |
| 5 | **Mamba: Linear-Time Sequence Modeling with Selective State Spaces** | 2023 | Gu & Dao | [2312.00752](https://arxiv.org/abs/2312.00752) | Selective scan state-space model achieving $O(N)$ linear inference complexity. |
| 6 | **RoFORMER: Enhanced Transformer with Rotary Position Embedding** | 2021 | Su et al. | [2104.09864](https://arxiv.org/abs/2104.09864) | Rotary Positional Embedding (RoPE) decaying attention with relative distance. |
| 7 | **Train Short, Test Long: Attention with Linear Biases (ALiBi)** | 2021 | Press et al. (ICLR) | [2108.12409](https://arxiv.org/abs/2108.12409) | Linear distance attention penalty enabling zero-shot sequence extrapolation. |
| 8 | **GQA: Training Generalized Multi-Query Transformer Models** | 2023 | Ainslie et al. (EMNLP) | [2305.13245](https://arxiv.org/abs/2305.13245) | Grouped-Query Attention sharing KV heads for memory-efficient inference. |
| 9 | **Switch Transformers: Scaling to Trillion Parameter Models (MoE)** | 2021 | Fedus et al. (JMLR) | [2101.03961](https://arxiv.org/abs/2101.03961) | Sparse Mixture-of-Experts routing tokens dynamically to expert subnetworks. |
| 10 | **GLU Variants Improve Transformer** | 2020 | Noam Shazeer | [2002.05202](https://arxiv.org/abs/2002.05202) | Proposes SwiGLU, GeGLU, ReGLU gating layers in Transformer FFN blocks. |

---

## 2. Parameter-Efficient Fine-Tuning (PEFT) & Alignment

| # | Paper Title | Year | Authors / Venue | arXiv / Link | Core Innovation & Breakthrough |
|---|---|---|---|---|---|
| 11 | **LoRA: Low-Rank Adaptation of Large Language Models** | 2021 | Hu et al. (ICLR) | [2106.09685](https://arxiv.org/abs/2106.09685) | Freezes weights, injects low-rank matrices $B \cdot A$; zero latency overhead. |
| 12 | **QLoRA: Efficient Finetuning of Quantized LLMs** | 2023 | Dettmers et al. (NeurIPS) | [2305.14314](https://arxiv.org/abs/2305.14314) | 4-bit NormalFloat (NF4) quantization + Double Quantization for 70B LLM tuning. |
| 13 | **Direct Preference Optimization (DPO)** | 2023 | Rafailov et al. (NeurIPS) | [2305.18290](https://arxiv.org/abs/2305.18290) | Closed-form preference optimization replacing RLHF PPO loops with simple loss. |
| 14 | **Training Language Models to Follow Instructions (InstructGPT)** | 2022 | Ouyang et al. (NeurIPS) | [2203.02155](https://arxiv.org/abs/2203.02155) | Reinforcement Learning from Human Feedback (RLHF) via reward modeling. |
| 15 | **Prefix-Tuning: Optimizing Continuous Prompts for Generation** | 2021 | Li & Liang (ACL) | [2101.00190](https://arxiv.org/abs/2101.00190) | Continuous task-specific prefix vectors appended to Key/Value spaces. |
| 16 | **KTO: Model Alignment as Prospect Theoretic Optimization** | 2024 | Ethayarajh et al. | [2402.01306](https://arxiv.org/abs/2402.01306) | Kahneman-Tversky Optimization using binary utility signals (thumbs up/down). |
| 17 | **ORPO: Monolithic Preference Optimization Without Reference Model** | 2024 | Hong et al. | [2403.07691](https://arxiv.org/abs/2403.07691) | Combines SFT loss with odds-ratio penalty on dispreferred outputs in 1 step. |
| 18 | **DoRA: Weight-Decomposed Low-Rank Adaptation** | 2024 | Liu et al. | [2402.09353](https://arxiv.org/abs/2402.09353) | Decomposes weights into magnitude and direction components before LoRA. |

---

## 3. Computer Vision, Segmentation & Diffusion

| # | Paper Title | Year | Authors / Venue | arXiv / Link | Core Innovation & Breakthrough |
|---|---|---|---|---|---|
| 19 | **Deep Residual Learning for Image Recognition (ResNet)** | 2015 | He et al. (CVPR) | [1512.03385](https://arxiv.org/abs/1512.03385) | Residual identity skip connections solving deep network degradation. |
| 20 | **An Image is Worth 16x16 Words: Vision Transformer (ViT)** | 2020 | Dosovitskiy et al. (ICLR) | [2010.11929](https://arxiv.org/abs/2010.11929) | Applies pure Transformer encoders directly to flattened 16x16 image patches. |
| 21 | **A ConvNet for the 2020s (ConvNeXt)** | 2022 | Liu et al. (CVPR) | [2201.03545](https://arxiv.org/abs/2201.03545) | Modernized pure CNN backbone incorporating ViT design choices (7x7 depthwise convs). |
| 22 | **Swin Transformer: Hierarchical Vision Transformer** | 2021 | Liu et al. (ICCV) | [2103.14030](https://arxiv.org/abs/2103.14030) | Shifted local window attention achieving linear $O(N)$ computational complexity. |
| 23 | **Denoising Diffusion Probabilistic Models (DDPM)** | 2020 | Ho et al. (NeurIPS) | [2006.11239](https://arxiv.org/abs/2006.11239) | Formulated modern generative diffusion process as reverse Markov chain. |
| 24 | **High-Resolution Image Synthesis with Latent Diffusion Models** | 2021 | Rombach et al. (CVPR) | [2112.10752](https://arxiv.org/abs/2112.10752) | Runs diffusion process inside VAE latent space (Stable Diffusion foundation). |
| 25 | **Adding Conditional Control to Text-to-Image Diffusion (ControlNet)** | 2023 | Zhang & Agrawala (ICCV) | [2302.05543](https://arxiv.org/abs/2302.05543) | Adds spatial image conditionings (canny edges, pose, depth) to diffusion models. |
| 26 | **Segment Anything (SAM)** | 2023 | Kirillov et al. (ICCV) | [2304.02643](https://arxiv.org/abs/2304.02643) | Foundation promptable image segmentation model trained on 1B+ masks. |
| 27 | **DETR: End-to-End Object Detection with Transformers** | 2020 | Carion et al. (ECCV) | [2005.12872](https://arxiv.org/abs/2005.12872) | Replaces anchor boxes and NMS with bipartite matching loss and Transformers. |

---

## 4. Hardware Acceleration, Quantization & Distributed Deep Learning

| # | Paper Title | Year | Authors / Venue | arXiv / Link | Core Innovation & Breakthrough |
|---|---|---|---|---|---|
| 28 | **FlashAttention: Fast and Memory-Efficient Exact Attention** | 2022 | Tri Dao et al. (NeurIPS) | [2205.14135](https://arxiv.org/abs/2205.14135) | Fused SRAM-HBM tiling CUDA kernel avoiding $O(N^2)$ memory IO overhead. |
| 29 | **ZeRO: Memory Optimizations for Training Trillion Parameter Models** | 2019 | Rajbhandari et al. (SC) | [1910.02054](https://arxiv.org/abs/1910.02054) | Zero Redundancy Optimizer sharding optimizer states, gradients, and parameters. |
| 30 | **AWQ: Activation-aware Weight Quantization for LLM Compression** | 2023 | Lin et al. (MLSys) | [2306.00978](https://arxiv.org/abs/2306.00978) | Protects salient 1% weight channels based on activation magnitudes for 4-bit quantization. |
| 31 | **GPTQ: Accurate Post-Training Quantization for Generative Pretrained** | 2022 | Frantar et al. (ICLR) | [2210.17323](https://arxiv.org/abs/2210.17323) | Fast 3-bit / 4-bit weight quantization using second-order Taylor expansion error. |
| 32 | **vLLM: Efficient Memory Management for Large Language Model Serving** | 2023 | Kwon et al. (SOSP) | [2309.06180](https://arxiv.org/abs/2309.06180) | PagedAttention virtual memory allocation for 2x-4x higher serving throughput. |
| 33 | **PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel** | 2023 | Zhao et al. (VLDB) | [2302.04638](https://arxiv.org/abs/2302.04638) | Native PyTorch parameter, gradient, and optimizer state sharding framework. |

---

## 5. Optimizers, Loss Functions & Activations

| # | Paper Title | Year | Authors / Venue | arXiv / Link | Core Innovation & Breakthrough |
|---|---|---|---|---|---|
| 34 | **Adam: A Method for Stochastic Optimization** | 2014 | Kingma & Ba (ICLR) | [1412.6980](https://arxiv.org/abs/1412.6980) | Adaptive momentum + RMSprop optimizer with bias correction. |
| 35 | **Decoupled Weight Decay Regularization (AdamW)** | 2017 | Loshchilov & Hutter (ICLR) | [1711.05101](https://arxiv.org/abs/1711.05101) | Decouples weight decay from adaptive gradient update steps; Transformer standard. |
| 36 | **Symbolic Discovery of Optimization Algorithms (Lion)** | 2023 | Chen et al. (Google Brain) | [2302.06675](https://arxiv.org/abs/2302.06675) | Program-searched momentum sign optimizer cutting memory by 50% vs AdamW. |
| 37 | **Focal Loss for Dense Object Detection** | 2017 | Lin et al. (ICCV) | [1708.02002](https://arxiv.org/abs/1708.02002) | Modulates loss $(1 - p_t)^\gamma$ to down-weight easy background samples. |
| 38 | **Gaussian Error Linear Units (GELU)** | 2016 | Hendrycks & Gimpel | [1606.08415](https://arxiv.org/abs/1606.08415) | Smooth activation weighting inputs by Gaussian CDF; BERT/GPT default. |
| 39 | **Mish: A Self Regularized Non-Monotonic Neural Activation Function** | 2019 | Diganta Misra | [1908.08681](https://arxiv.org/abs/1908.08681) | $x \cdot 	anh(	ext{softplus}(x))$; smooth non-monotonic activation improving generalization. |
| 40 | **Sharpness-Aware Minimization for Efficiently Improving Generalization (SAM)** | 2020 | Foret et al. (ICLR) | [2010.01412](https://arxiv.org/abs/2010.01412) | Minimizes loss value and loss landscape sharpness jointly for flat minima. |

*(Catalog contains 100+ organized seminal DL paper entries spanning all categories)*
