# A Survey of Reinforcement Learning for LLM Reasoning: Methods, Search, and Verifier Guidance

## TL;DR
- Reinforcement learning from verifiable rewards has superseded pure supervised instruction tuning for complex reasoning and mathematical derivation [1].
- Process-supervised reward models (PRMs) provide granular step-by-step feedback that significantly outperforms terminal outcome-supervised reward models (ORMs) [2].
- Test-time compute scaling via tree search and Monte Carlo Rollouts enables smaller language models to surpass much larger static models in multi-hop deduction [3].
- Dynamic compute allocation via cross-turn estimation mitigates overthinking overhead by deciding adaptively when agents should invoke deep reasoning traces [4].

## Background
Large language models historically derived reasoning capabilities from autoregressive next-token prediction and Supervised Fine-Tuning (SFT) over curated chain-of-thought demonstrations [1]. However, SFT inherently suffers from exposure bias and distributional shift: once a model deviates from the demonstration trajectory, errors cascade uncontrollably. Reinforcement Learning (RL), particularly when guided by verifiable ground-truth rewards or dense process verifiers, transforms reasoning from passive token mimicry into active exploration and credit assignment [1][2]. Recent models leverage rule-based reward signals and tree search to unlock self-correction, systematic verification, and test-time compute scaling across mathematical theorem proving and competitive programming [3][4].

## Outcome vs. Process Reward Formulations and Credit Assignment
The core dichotomy in RL for reasoning lies between Outcome-supervised Reward Models (ORMs) and Process-supervised Reward Models (PRMs) [1][2]. ORMs assign a scalar reward solely at the conclusion of a multi-step solution. While easy to automate via unit tests or final answer parsing, ORMs suffer from sparse rewards and credit assignment ambiguity, often reinforcing incorrect intermediate logic that happens to arrive at the correct final answer [1]. Conversely, PRMs evaluate the mathematical validity of every individual reasoning step [2]. Experimental findings demonstrate that training RL policies with step-level process supervision reduces logical hallucinations by over 40% compared to terminal outcome rewards, enabling the agent to prune erroneous exploratory branches early in training [2].

## Policy Optimization Algorithms: PPO, DPO, and Rule-Based Search
Policy optimization in reasoning domains has progressed through several algorithmic paradigms. Proximal Policy Optimization (PPO) was initially standard for alignment, but its multi-model overhead (actor, critic, reference, reward models) creates severe memory bottlenecks when training long chain-of-thought traces [1]. Direct Preference Optimization (DPO) and its reasoning-specialized variants bypass the critic by optimizing policy log-ratios directly over paired positive and negative rationales [3]. More recently, reinforcement learning from verifiable environments leverages rejection sampling and rule-based search without a learned critic, optimizing the policy directly against deterministic execution feedback such as compiler status or symbolic math engines [3][4].

## Test-Time Scaling and Adaptive Inference Budgets
A foundational shift in post-training reasoning is the trading of inference-time compute for parameter scale [3]. Instead of relying on single greedy generations, systems execute Best-of-N sampling, beam search, or Monte Carlo Tree Search (MCTS) guided by trained value networks or verifiers [3]. Empirical analyses reveal that allocating exponential test-time compute reliably improves task accuracy along log-linear scaling curves [3]. However, fixed extended reasoning chains introduce substantial computational latency and redundant token generation on trivial queries [4]. Adaptive reasoning frameworks address this inefficiency through cross-turn estimation, dynamically predicting problem difficulty and determining whether to invoke extended reasoning or provide immediate answers [4].

## Trends and open problems
Current open frontiers focus on extending RL reasoning beyond formal, verifiable domains like mathematics and coding into unstructured domains such as medicine, law, and creative problem solving [1][3]. In open-ended domains, establishing reliable ground-truth verifiers without introducing reward hacking remains unresolved [2]. Furthermore, preventing mode collapse during prolonged multi-turn RL training and mitigating catastrophic forgetting of general conversational fluency are critical ongoing challenges [1][4][5].

## References
[1] DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arxiv. https://arxiv.org/abs/2402.03300 (2024-02-05)
[2] Let's Verify Step by Step. arxiv. https://arxiv.org/abs/2305.20050 (2023-05-31)
[3] Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters. hf-search. https://huggingface.co/papers/2408.03314 (2024-08-06)
[4] When Should Agents Think? Adaptive Reasoning via Cross-Turn Estimation. arxiv. https://arxiv.org/abs/2610.12061 (2026-10-08)
[5] Accelerating Reinforcement Learning for LLM Reasoning with Modern Compute. web. https://developer.nvidia.com/blog/reinforcement-learning-for-llm-reasoning/ (2025-04-12)
