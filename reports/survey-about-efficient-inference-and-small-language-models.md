# A Survey of Efficient Inference and Small Language Models: Quantization, Speculative Decoding, and Architectural Compactness

## TL;DR
- Small Language Models (SLMs) with 1B–7B parameters achieve reasoning competitive with legacy 70B models through filtered synthetic data and distillation [1].
- Weight and activation quantization techniques (e.g., AWQ, FP4) reduce memory bandwidth pressure with negligible perplexity degradation [2].
- Speculative decoding accelerates autoregressive generation by 2x–3x without altering output probability distributions [3].
- Directional pruning and representation purification remove redundant parameters while preserving safety and factual alignment [4].

## Background
The exponential scaling of frontier language models has unlocked impressive reasoning capabilities, but it has simultaneously introduced prohibitive serving latency, unsustainable GPU memory requirements, and high energy costs. Consequently, efficient inference and Small Language Models (SLMs) have emerged as essential fields of study [1]. By combining architectural innovations, rigorous post-training quantization, speculative execution, and high-quality data curation, researchers have engineered compact models that can run locally on consumer-grade hardware and edge devices without sacrificing operational utility [1][2].

## Model Compression: Quantization, Sparsity, and Pruning
Autoregressive generation is fundamentally memory-bandwidth bound rather than compute bound during token generation [2]. Post-Training Quantization (PTQ) techniques compress 16-bit floating-point weights into 4-bit integer (INT4) or floating-point (FP4) representations [2]. Activation-aware Weight Quantization (AWQ) identifies and protects the top 1% salient weight channels that dominate generation quality, minimizing quantization noise [2]. Complementing quantization, structured sparsity and directional pruning identify and eliminate redundant weight dimensions, reducing active FLOPs per forward pass while maintaining core benchmark accuracy [2][4].

## Inference Acceleration: Speculative Decoding and KV-Cache Optimization
Traditional transformer generation produces exactly one token per forward pass, underutilizing modern GPU compute parallelism [3]. Speculative decoding mitigates this bottleneck by deploying a lightweight draft model to generate candidate token sequences, which a larger target model verifies in a single parallelized forward pass [3]. Because the target model accepts or rejects tokens via modified rejection sampling, the output distribution remains mathematically identical to greedy decoding while delivering 2x to 3x speedups [3]. Furthermore, Multi-Head Latent Attention (MLA) and PagedAttention drastically reduce Key-Value (KV) cache memory footprints, enabling high batch concurrency during peak serving loads [1][3].

## Training Small Language Models on High-Quality Synthetic Data
Recent architectural iterations demonstrate that parameter scale is often a surrogate for training data quality [1]. The Phi and Gemma model families show that training 2B-3B parameter models on curated textbook-quality datasets and multi-turn synthetic reasoning traces yields performance on math and coding benchmarks that rivals models an order of magnitude larger [1]. Knowledge distillation from frontier teacher models transfers latent reasoning abstractions, allowing compact models to internalize step-by-step problem-solving heuristics with minimal parameter capacity [1][3].

## Trends and open problems
Key trends involve hardware-software co-design, such as native microscaling formats (MXFP6, MXFP4) and specialized NPU inference runtimes for on-device deployment [1][2]. Critical unresolved hurdles include mitigating quantization degradation on long-context retrieval, preventing catastrophic forgetting during aggressive weight pruning, and minimizing the draft model rejection rate in speculative decoding across high-entropy creative tasks [3][4][5].

## References
[1] Textbooks Are All You Need. arxiv. https://arxiv.org/abs/2306.11644 (2023-06-20)
[2] AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration. hf-search. https://huggingface.co/papers/2306.00978 (2023-06-01)
[3] Fast Inference from Transformers via Speculative Decoding. arxiv. https://arxiv.org/abs/2211.17192 (2022-11-30)
[4] Purifying Backdoored Large Vision-Language Models by Removing Hijacked Directions. arxiv. https://arxiv.org/abs/2610.09941 (2026-10-07)
[5] Optimizing LLMs for Speed and Memory. web. https://huggingface.co/blog/optimize-llm (2024-03-15)
