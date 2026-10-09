# A Survey of Video and Multimodal Generation: Diffusion Transformers, Spatiotemporal Tokenization, and Physical World Modeling

## TL;DR
- Diffusion Transformers (DiT) operating over continuous 3D latent spaces have become the dominant architecture for high-definition video synthesis [1].
- Causal spatiotemporal autoencoders compress visual signals into unified latent representations while maintaining strict temporal causality [2].
- Incorporating explicit world modeling objectives into multimodal foundation models grounds generated scenes in physical dynamics and kinematic consistency [3].
- Scaling laws demonstrate predictable reductions in Fréchet Video Distance (FVD) as training compute and patch resolution increase [4].

## Background
Video generation represents one of the most computationally intensive frontiers in generative artificial intelligence. Moving beyond static 2D image synthesis requires models to understand complex 3D camera geometry, continuous temporal consistency, dynamic object interactions, and intuitive physics [1]. While early efforts adapted generative adversarial networks (GANs) and autoregressive token predictors, modern breakthroughs are powered by Diffusion Transformers (DiTs) trained across billions of internet-scale video clips [1][2]. As generative quality approaches photorealism, research focus has pivoted toward physical world grounding, camera trajectory controllability, and multimodal cross-conditioning [3][4].

## Diffusion Transformers vs. U-Net Architectures
For several years, convolutional 2D/3D U-Net backbones with temporal attention layers dominated latent diffusion models [1]. However, U-Nets impose rigid structural inductive biases that scale poorly as model parameters exceed several billion weights [1][2]. The adoption of Diffusion Transformers (DiT) revolutionized video generation by replacing convolutional blocks with standard transformer architectures operating over spatiotemporal latent patches [1]. Empirical comparisons highlight that DiT backbones demonstrate superior compute efficiency, clean scaling curves, and higher visual fidelity, efficiently parallelizing attention across long temporal sequences [1][2].

## Spatiotemporal Tokenization and Causal Latent Representations
Generating high-resolution video directly in pixel space is computationally prohibitive. Consequently, models rely on neural video tokenizers—such as 3D Variational Autoencoders (3D-VAEs) or causal video autoencoders—to compress raw frames along both spatial dimensions (e.g., 8x downsampling) and temporal dimensions (e.g., 4x temporal compression) [2]. Implementing causal temporal convolutions ensures that future frames do not leak into the compression of past frames, which is vital for streaming video synthesis and interactive generation [2]. Comparative benchmarks show that causal tokenizers substantially reduce temporal flicker while preserving high-frequency motion details [2][4].

## World Modeling Integration and Physical Grounding
A foundational limitation of purely statistical video generators is their propensity to produce visually convincing yet physically impossible dynamics—such as morphing objects, vanishing entities, or broken collision laws [3]. Recent architectures bridge this divide by weaving visual world modeling inductive biases into multimodal foundation models [3]. By training models to jointly predict future latent states and optical flow fields conditioned on geometric actions or camera pose vectors, systems achieve markedly superior physical consistency and camera trajectory adherence [3][4].

## Trends and open problems
Contemporary developments highlight unified multimodal foundation models capable of interleaved text, audio, image, and video generation within a shared representation space [1][3]. Persistent research bottlenecks include extensive inference compute costs (requiring dozens of diffusion sampling steps), the absence of reliable automated metrics to capture human perceptual realism beyond FVD, and the challenge of simulating long-horizon interactive environments spanning multiple minutes [2][4].

## References
[1] Scalable Diffusion Models with Transformers. arxiv. https://arxiv.org/abs/2212.09748 (2022-12-19)
[2] Sora: A Review on Background, Technology, Limitations, and Opportunities of Large Vision Models. hf-search. https://huggingface.co/papers/2402.17177 (2024-02-27)
[3] WOVEN: Weaving Visual World Modeling into Multimodal LLMs. arxiv. https://arxiv.org/abs/2610.12417 (2026-10-08)
[4] Diffusion Models for Video Generation. web. https://lilianweng.github.io/posts/2024-04-12-diffusion-video/ (2024-04-12)
