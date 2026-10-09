# A Survey of World Models: Foundations, Dynamics Learning, and Embodied Agent Planning

## TL;DR
- World models form internalized predictive representations of environment transition dynamics that allow agents to simulate potential futures without physical interaction [1].
- Latent recurrent architectures such as Recurrent State-Space Models (RSSM) and transformer-based autoregressive visual token predictors decouple state estimation from observation dimensionality [2].
- Modern scalable video-conditioned world models integrate RingAttention to extend context lengths across million-frame physical simulations [3].
- Action-faithful robot world models trained with counterfactual trajectories significantly reduce policy hallucination in continuous control robotics [4].

## Background
The conceptual lineage of world models traces back to cognitive science and foundational reinforcement learning architectures by Ha and Schmidhuber [1]. Rather than operating directly upon raw, high-dimensional perceptual streams such as pixel observations, a world model decomposes autonomous decision-making into an internal sensory encoder, a latent predictive transition dynamic model, and a decision policy. By projecting complex spatial data into compact latent spaces, world models empower autonomous agents to simulate rollouts purely inside latent imagination, drastically improving sample efficiency [2]. Over the past three years, this paradigm has expanded from low-dimensional discrete game environments into large-scale robotic manipulation, autonomous driving simulators, and foundation model physics engines [3][4].

## Latent State Dynamics and Recurrent Predictive Architectures
Early implementations relied on variational autoencoders coupled with recurrent neural networks (RNNs) and Mixture Density Networks to predict future latent states and reward distributions [1]. This foundational design evolved into Recurrent State-Space Models (RSSM), famously operationalized across the Dreamer family of algorithms, which segregate deterministic latent pathways from stochastic transition variables [2]. Recent comparative analyses demonstrate that RSSM achieves high sample efficiency in continuous control tasks by maintaining temporal consistency while bounding variance across extended imaginary rollouts [2]. However, recurrent latent dynamics frequently suffer from compounding error accumulation when rollout horizons extend past several dozen steps, motivating the shift toward attention-driven sequence prediction [3].

## Transformer and Diffusion-Based Sequence World Models
To resolve the horizon limits of recurrent formulations, subsequent architectures reframed environment dynamics as sequential generative modeling over discrete visual tokens. Transformers operating over Vector-Quantized (VQ) representations predict next tokens autoregressively across space and time [3]. Systems scaling to million-length contexts leverage RingAttention and masked sequence packing, allowing long-horizon physical coherence across extended video sequences [3]. In parallel, latent diffusion formulations treat future state prediction as iterative denoising conditioned on historical frames and executed actions [4]. Empirical comparisons show that while transformer models excel at discrete token predictability and fast inference, diffusion models deliver superior visual fidelity and spatial detail at the expense of higher inference computational overhead [3][4].

## Action Conditioning and Counterfactual Policy Optimization
A persistent vulnerability of video prediction models applied as world simulators is action unfaithfulness, where simulated futures fail to respect the control commands issued by the agent. Recent breakthroughs incorporate counterfactual post-training to enforce strict action compliance across cross-embodiment robotic datasets [4]. By generating paired synthetic trajectories where actions are counterfactually modified, the dynamics network learns robust causal invariance rather than merely copying visual correlations [4]. Furthermore, benchmark evaluations indicate that policies optimized entirely in latent imagination achieve comparable or superior task success rates compared to real-world supervised fine-tuning, while circumventing physical wear and safety risks [1][2][4].

## Trends and open problems
Key frontiers in world models center on cross-embodiment generalization and physical plausibility guarantees. While existing models capture visual kinetics, true 3D intuitive physics—including mass distribution, friction, and fluid mechanics—remains difficult to ground without explicit geometric inductive biases [3][4]. Moreover, evaluating world models remains debated: pixel-level reconstruction metrics such as FVD and PSNR correlate weakly with downstream planning success, prompting the development of goal-reachability benchmarks and latent prediction divergence metrics [2][4].

## References
[1] World Models. arxiv. https://arxiv.org/abs/1803.10122 (2018-05-09)
[2] Understanding World or Predicting Future? A Comprehensive Survey of World Models. web. https://dl.acm.org/doi/10.1145/3746449 (2025-09-09)
[3] World Model on Million-Length Video And Language With RingAttention. hf-search. https://huggingface.co/papers/2402.08268 (2024-02-13)
[4] DreamTrue: Action-Faithful Robot World Model with Counterfactual Post-Training. arxiv. https://arxiv.org/abs/2610.12468 (2026-10-08)
