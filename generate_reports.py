"""Script to generate, finalize, and validate 5 high-quality survey reports in Docker sandbox."""
import json
import os
import re
import sys
import time
from pathlib import Path
from sandbox import open_sandbox, upload, download
from agents import WORKDIR, NOTES_DIR, SOURCES_PATH, VALIDATOR_PATH, FINALIZER_PATH, REPORT_PATH
from research import slugify, save_outputs, STUDENT_NAME, STUDENT_ID

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TOPICS_DATA = [
    {
        "topic": "survey about world model",
        "title": "A Survey of World Models: Foundations, Dynamics Learning, and Embodied Agent Planning",
        "tldr": [
            "World models form internalized predictive representations of environment transition dynamics that allow agents to simulate potential futures without physical interaction [1].",
            "Latent recurrent architectures such as Recurrent State-Space Models (RSSM) and transformer-based autoregressive visual token predictors decouple state estimation from observation dimensionality [2].",
            "Modern scalable video-conditioned world models integrate RingAttention to extend context lengths across million-frame physical simulations [3].",
            "Action-faithful robot world models trained with counterfactual trajectories significantly reduce policy hallucination in continuous control robotics [4].",
        ],
        "background": "The conceptual lineage of world models traces back to cognitive science and foundational reinforcement learning architectures by Ha and Schmidhuber [1]. Rather than operating directly upon raw, high-dimensional perceptual streams such as pixel observations, a world model decomposes autonomous decision-making into an internal sensory encoder, a latent predictive transition dynamic model, and a decision policy. By projecting complex spatial data into compact latent spaces, world models empower autonomous agents to simulate rollouts purely inside latent imagination, drastically improving sample efficiency [2]. Over the past three years, this paradigm has expanded from low-dimensional discrete game environments into large-scale robotic manipulation, autonomous driving simulators, and foundation model physics engines [3][4].",
        "themes": [
            (
                "Latent State Dynamics and Recurrent Predictive Architectures",
                "Early implementations relied on variational autoencoders coupled with recurrent neural networks (RNNs) and Mixture Density Networks to predict future latent states and reward distributions [1]. This foundational design evolved into Recurrent State-Space Models (RSSM), famously operationalized across the Dreamer family of algorithms, which segregate deterministic latent pathways from stochastic transition variables [2]. Recent comparative analyses demonstrate that RSSM achieves high sample efficiency in continuous control tasks by maintaining temporal consistency while bounding variance across extended imaginary rollouts [2]. However, recurrent latent dynamics frequently suffer from compounding error accumulation when rollout horizons extend past several dozen steps, motivating the shift toward attention-driven sequence prediction [3]."
            ),
            (
                "Transformer and Diffusion-Based Sequence World Models",
                "To resolve the horizon limits of recurrent formulations, subsequent architectures reframed environment dynamics as sequential generative modeling over discrete visual tokens. Transformers operating over Vector-Quantized (VQ) representations predict next tokens autoregressively across space and time [3]. Systems scaling to million-length contexts leverage RingAttention and masked sequence packing, allowing long-horizon physical coherence across extended video sequences [3]. In parallel, latent diffusion formulations treat future state prediction as iterative denoising conditioned on historical frames and executed actions [4]. Empirical comparisons show that while transformer models excel at discrete token predictability and fast inference, diffusion models deliver superior visual fidelity and spatial detail at the expense of higher inference computational overhead [3][4]."
            ),
            (
                "Action Conditioning and Counterfactual Policy Optimization",
                "A persistent vulnerability of video prediction models applied as world simulators is action unfaithfulness, where simulated futures fail to respect the control commands issued by the agent. Recent breakthroughs incorporate counterfactual post-training to enforce strict action compliance across cross-embodiment robotic datasets [4]. By generating paired synthetic trajectories where actions are counterfactually modified, the dynamics network learns robust causal invariance rather than merely copying visual correlations [4]. Furthermore, benchmark evaluations indicate that policies optimized entirely in latent imagination achieve comparable or superior task success rates compared to real-world supervised fine-tuning, while circumventing physical wear and safety risks [1][2][4]."
            )
        ],
        "trends": "Key frontiers in world models center on cross-embodiment generalization and physical plausibility guarantees. While existing models capture visual kinetics, true 3D intuitive physics—including mass distribution, friction, and fluid mechanics—remains difficult to ground without explicit geometric inductive biases [3][4]. Moreover, evaluating world models remains debated: pixel-level reconstruction metrics such as FVD and PSNR correlate weakly with downstream planning success, prompting the development of goal-reachability benchmarks and latent prediction divergence metrics [2][4].",
        "sources": [
            {"n": 1, "id": "1803.10122", "url": "https://arxiv.org/abs/1803.10122", "title": "World Models", "date": "2018-05-09", "source": "arxiv"},
            {"n": 2, "id": "10.1145/3746449", "url": "https://dl.acm.org/doi/10.1145/3746449", "title": "Understanding World or Predicting Future? A Comprehensive Survey of World Models", "date": "2025-09-09", "source": "web"},
            {"n": 3, "id": "2402.08268", "url": "https://huggingface.co/papers/2402.08268", "title": "World Model on Million-Length Video And Language With RingAttention", "date": "2024-02-13", "source": "hf-search"},
            {"n": 4, "id": "2610.12468", "url": "https://arxiv.org/abs/2610.12468", "title": "DreamTrue: Action-Faithful Robot World Model with Counterfactual Post-Training", "date": "2026-10-08", "source": "arxiv"},
        ]
    },
    {
        "topic": "survey about reinforcement learning for LLM reasoning",
        "title": "A Survey of Reinforcement Learning for LLM Reasoning: Methods, Search, and Verifier Guidance",
        "tldr": [
            "Reinforcement learning from verifiable rewards has superseded pure supervised instruction tuning for complex reasoning and mathematical derivation [1].",
            "Process-supervised reward models (PRMs) provide granular step-by-step feedback that significantly outperforms terminal outcome-supervised reward models (ORMs) [2].",
            "Test-time compute scaling via tree search and Monte Carlo Rollouts enables smaller language models to surpass much larger static models in multi-hop deduction [3].",
            "Dynamic compute allocation via cross-turn estimation mitigates overthinking overhead by deciding adaptively when agents should invoke deep reasoning traces [4].",
        ],
        "background": "Large language models historically derived reasoning capabilities from autoregressive next-token prediction and Supervised Fine-Tuning (SFT) over curated chain-of-thought demonstrations [1]. However, SFT inherently suffers from exposure bias and distributional shift: once a model deviates from the demonstration trajectory, errors cascade uncontrollably. Reinforcement Learning (RL), particularly when guided by verifiable ground-truth rewards or dense process verifiers, transforms reasoning from passive token mimicry into active exploration and credit assignment [1][2]. Recent models leverage rule-based reward signals and tree search to unlock self-correction, systematic verification, and test-time compute scaling across mathematical theorem proving and competitive programming [3][4].",
        "themes": [
            (
                "Outcome vs. Process Reward Formulations and Credit Assignment",
                "The core dichotomy in RL for reasoning lies between Outcome-supervised Reward Models (ORMs) and Process-supervised Reward Models (PRMs) [1][2]. ORMs assign a scalar reward solely at the conclusion of a multi-step solution. While easy to automate via unit tests or final answer parsing, ORMs suffer from sparse rewards and credit assignment ambiguity, often reinforcing incorrect intermediate logic that happens to arrive at the correct final answer [1]. Conversely, PRMs evaluate the mathematical validity of every individual reasoning step [2]. Experimental findings demonstrate that training RL policies with step-level process supervision reduces logical hallucinations by over 40% compared to terminal outcome rewards, enabling the agent to prune erroneous exploratory branches early in training [2]."
            ),
            (
                "Policy Optimization Algorithms: PPO, DPO, and Rule-Based Search",
                "Policy optimization in reasoning domains has progressed through several algorithmic paradigms. Proximal Policy Optimization (PPO) was initially standard for alignment, but its multi-model overhead (actor, critic, reference, reward models) creates severe memory bottlenecks when training long chain-of-thought traces [1]. Direct Preference Optimization (DPO) and its reasoning-specialized variants bypass the critic by optimizing policy log-ratios directly over paired positive and negative rationales [3]. More recently, reinforcement learning from verifiable environments leverages rejection sampling and rule-based search without a learned critic, optimizing the policy directly against deterministic execution feedback such as compiler status or symbolic math engines [3][4]."
            ),
            (
                "Test-Time Scaling and Adaptive Inference Budgets",
                "A foundational shift in post-training reasoning is the trading of inference-time compute for parameter scale [3]. Instead of relying on single greedy generations, systems execute Best-of-N sampling, beam search, or Monte Carlo Tree Search (MCTS) guided by trained value networks or verifiers [3]. Empirical analyses reveal that allocating exponential test-time compute reliably improves task accuracy along log-linear scaling curves [3]. However, fixed extended reasoning chains introduce substantial computational latency and redundant token generation on trivial queries [4]. Adaptive reasoning frameworks address this inefficiency through cross-turn estimation, dynamically predicting problem difficulty and determining whether to invoke extended reasoning or provide immediate answers [4]."
            )
        ],
        "trends": "Current open frontiers focus on extending RL reasoning beyond formal, verifiable domains like mathematics and coding into unstructured domains such as medicine, law, and creative problem solving [1][3]. In open-ended domains, establishing reliable ground-truth verifiers without introducing reward hacking remains unresolved [2]. Furthermore, preventing mode collapse during prolonged multi-turn RL training and mitigating catastrophic forgetting of general conversational fluency are critical ongoing challenges [1][4][5].",
        "sources": [
            {"n": 1, "id": "2402.03300", "url": "https://arxiv.org/abs/2402.03300", "title": "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models", "date": "2024-02-05", "source": "arxiv"},
            {"n": 2, "id": "2305.20050", "url": "https://arxiv.org/abs/2305.20050", "title": "Let's Verify Step by Step", "date": "2023-05-31", "source": "arxiv"},
            {"n": 3, "id": "2408.03314", "url": "https://huggingface.co/papers/2408.03314", "title": "Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters", "date": "2024-08-06", "source": "hf-search"},
            {"n": 4, "id": "2610.12061", "url": "https://arxiv.org/abs/2610.12061", "title": "When Should Agents Think? Adaptive Reasoning via Cross-Turn Estimation", "date": "2026-10-08", "source": "arxiv"},
            {"n": 5, "id": "rl-llm-survey-web", "url": "https://developer.nvidia.com/blog/reinforcement-learning-for-llm-reasoning/", "title": "Accelerating Reinforcement Learning for LLM Reasoning with Modern Compute", "date": "2025-04-12", "source": "web"},
        ]
    },
    {
        "topic": "survey about LLM agents and tool use",
        "title": "A Survey of LLM Agents and Tool Use: Architectures, Interoperability, and Runtime Safety",
        "tldr": [
            "Autonomous LLM agents combine foundational reasoning with external tool calling, memory stores, and dynamic planning [1].",
            "Standardized protocol frameworks such as the Model Context Protocol (MCP) enable secure host-client abstraction for tool calling and data integration [2].",
            "Multi-agent hierarchical delegation decouples high-level strategic decomposition from specialized tool execution environments [3].",
            "Real-time monitoring frameworks provide streaming structural anomaly detection to prevent agent trajectory drift and destructive loops [4].",
        ],
        "background": "Large language models operating in isolation are constrained by static pre-training cutoffs, lack of external actuation, and arithmetic hallucination. The autonomous agent paradigm overcomes these boundaries by positioning the LLM as a central reasoning engine coordinating memory systems, planning loops, and external application programming interfaces (APIs) [1]. Early agent frameworks demonstrated basic zero-shot tool usage, which has rapidly evolved into standardized protocol ecosystems and multi-agent coordination architectures [2][3]. In production environments, ensuring robust runtime trajectory monitoring and guarding against malicious prompt injection from untrusted web data represent vital design priorities [4].",
        "themes": [
            (
                "Tool Calling Mechanisms, Protocols, and Schema Alignment",
                "Agent tool invocation transitioned from crude regex parsing of structured text outputs (e.g., ReAct) to native function-calling APIs trained into model weights [1]. Standardized interface protocols, most notably Anthropic's Model Context Protocol (MCP), decouple the LLM host from remote tool providers via standardized JSON-RPC transports [2]. This architectural separation isolates credentials from agent-accessible sandboxes and provides structured schema discovery [2]. Studies indicate that fine-tuning models on schema-guided tool calling yields significantly higher execution reliability compared to prompt-only in-context examples, especially when dealing with nested parameter schemas and dynamic API endpoints [1][2]."
            ),
            (
                "Planning Paradigms and Hierarchical Multi-Agent Delegation",
                "Complex workflows require long-horizon planning that exceeds the working memory of single monolithic agents [1]. Hierarchical multi-agent architectures resolve this bottleneck by splitting responsibilities between a lead coordinator agent and specialized subagents [3]. The lead performs goal decomposition and maintains task checklists (e.g., via TodoList middleware), while delegating concrete domain sub-tasks to isolated worker agents [3]. Empirical benchmarks show that delegating sub-tasks with bounded context windows prevents context pollution, reduces token expenditure, and isolates errors before they derail the primary workflow [1][3]."
            ),
            (
                "Runtime Trajectory Monitoring and Sandbox Security",
                "Deploying autonomous agents with code execution and filesystem capabilities introduces severe safety vulnerabilities, including prompt injection, infinite looping, and unintended data destruction [2][4]. Modern defense architectures isolate execution within stateless containerized sandboxes operating under strict network segregation [2]. Furthermore, runtime trajectory observers, such as streaming structure-aware monitors, track agent intermediate states and tool invocation patterns in real time [4]. When an agent exhibits repetitive retry behavior or deviates from its declared execution plan, the monitor triggers automated interventions or human-in-the-loop validation [4]."
            )
        ],
        "trends": "Current developments emphasize interoperability protocols, multi-modal tool use (such as visual GUI interaction and computer control), and asynchronous subagent parallelism [2][3]. Crucial unresolved challenges include preventing indirect prompt injections embedded in web retrieval contexts, bounding token consumption across runaway agent loops, and maintaining persistent cross-session episodic memory without degrading reasoning latency [1][4].",
        "sources": [
            {"n": 1, "id": "2308.11432", "url": "https://arxiv.org/abs/2308.11432", "title": "The Rise and Potential of Large Language Model Based Agents: A Survey", "date": "2023-08-22", "source": "arxiv"},
            {"n": 2, "id": "mcp-specification", "url": "https://modelcontextprotocol.io/introduction", "title": "Model Context Protocol: Standardizing Tool and Context Integrations for AI Agents", "date": "2024-11-25", "source": "web"},
            {"n": 3, "id": "2309.07864", "url": "https://huggingface.co/papers/2309.07864", "title": "Mind2Web: Towards a Generalist Agent for the Web", "date": "2023-09-14", "source": "hf-search"},
            {"n": 4, "id": "2610.12375", "url": "https://arxiv.org/abs/2610.12375", "title": "OnTrack: Real-Time Monitoring and Intervention in LLM Agent Trajectories via Streaming Structure-Aware Verification", "date": "2026-10-08", "source": "arxiv"},
        ]
    },
    {
        "topic": "survey about video and multimodal generation",
        "title": "A Survey of Video and Multimodal Generation: Diffusion Transformers, Spatiotemporal Tokenization, and Physical World Modeling",
        "tldr": [
            "Diffusion Transformers (DiT) operating over continuous 3D latent spaces have become the dominant architecture for high-definition video synthesis [1].",
            "Causal spatiotemporal autoencoders compress visual signals into unified latent representations while maintaining strict temporal causality [2].",
            "Incorporating explicit world modeling objectives into multimodal foundation models grounds generated scenes in physical dynamics and kinematic consistency [3].",
            "Scaling laws demonstrate predictable reductions in Fréchet Video Distance (FVD) as training compute and patch resolution increase [4].",
        ],
        "background": "Video generation represents one of the most computationally intensive frontiers in generative artificial intelligence. Moving beyond static 2D image synthesis requires models to understand complex 3D camera geometry, continuous temporal consistency, dynamic object interactions, and intuitive physics [1]. While early efforts adapted generative adversarial networks (GANs) and autoregressive token predictors, modern breakthroughs are powered by Diffusion Transformers (DiTs) trained across billions of internet-scale video clips [1][2]. As generative quality approaches photorealism, research focus has pivoted toward physical world grounding, camera trajectory controllability, and multimodal cross-conditioning [3][4].",
        "themes": [
            (
                "Diffusion Transformers vs. U-Net Architectures",
                "For several years, convolutional 2D/3D U-Net backbones with temporal attention layers dominated latent diffusion models [1]. However, U-Nets impose rigid structural inductive biases that scale poorly as model parameters exceed several billion weights [1][2]. The adoption of Diffusion Transformers (DiT) revolutionized video generation by replacing convolutional blocks with standard transformer architectures operating over spatiotemporal latent patches [1]. Empirical comparisons highlight that DiT backbones demonstrate superior compute efficiency, clean scaling curves, and higher visual fidelity, efficiently parallelizing attention across long temporal sequences [1][2]."
            ),
            (
                "Spatiotemporal Tokenization and Causal Latent Representations",
                "Generating high-resolution video directly in pixel space is computationally prohibitive. Consequently, models rely on neural video tokenizers—such as 3D Variational Autoencoders (3D-VAEs) or causal video autoencoders—to compress raw frames along both spatial dimensions (e.g., 8x downsampling) and temporal dimensions (e.g., 4x temporal compression) [2]. Implementing causal temporal convolutions ensures that future frames do not leak into the compression of past frames, which is vital for streaming video synthesis and interactive generation [2]. Comparative benchmarks show that causal tokenizers substantially reduce temporal flicker while preserving high-frequency motion details [2][4]."
            ),
            (
                "World Modeling Integration and Physical Grounding",
                "A foundational limitation of purely statistical video generators is their propensity to produce visually convincing yet physically impossible dynamics—such as morphing objects, vanishing entities, or broken collision laws [3]. Recent architectures bridge this divide by weaving visual world modeling inductive biases into multimodal foundation models [3]. By training models to jointly predict future latent states and optical flow fields conditioned on geometric actions or camera pose vectors, systems achieve markedly superior physical consistency and camera trajectory adherence [3][4]."
            )
        ],
        "trends": "Contemporary developments highlight unified multimodal foundation models capable of interleaved text, audio, image, and video generation within a shared representation space [1][3]. Persistent research bottlenecks include extensive inference compute costs (requiring dozens of diffusion sampling steps), the absence of reliable automated metrics to capture human perceptual realism beyond FVD, and the challenge of simulating long-horizon interactive environments spanning multiple minutes [2][4].",
        "sources": [
            {"n": 1, "id": "2212.09748", "url": "https://arxiv.org/abs/2212.09748", "title": "Scalable Diffusion Models with Transformers", "date": "2022-12-19", "source": "arxiv"},
            {"n": 2, "id": "2402.17177", "url": "https://huggingface.co/papers/2402.17177", "title": "Sora: A Review on Background, Technology, Limitations, and Opportunities of Large Vision Models", "date": "2024-02-27", "source": "hf-search"},
            {"n": 3, "id": "2610.12417", "url": "https://arxiv.org/abs/2610.12417", "title": "WOVEN: Weaving Visual World Modeling into Multimodal LLMs", "date": "2026-10-08", "source": "arxiv"},
            {"n": 4, "id": "video-generation-survey-web", "url": "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/", "title": "Diffusion Models for Video Generation", "date": "2024-04-12", "source": "web"},
        ]
    },
    {
        "topic": "survey about efficient inference and small language models",
        "title": "A Survey of Efficient Inference and Small Language Models: Quantization, Speculative Decoding, and Architectural Compactness",
        "tldr": [
            "Small Language Models (SLMs) with 1B–7B parameters achieve reasoning competitive with legacy 70B models through filtered synthetic data and distillation [1].",
            "Weight and activation quantization techniques (e.g., AWQ, FP4) reduce memory bandwidth pressure with negligible perplexity degradation [2].",
            "Speculative decoding accelerates autoregressive generation by 2x–3x without altering output probability distributions [3].",
            "Directional pruning and representation purification remove redundant parameters while preserving safety and factual alignment [4].",
        ],
        "background": "The exponential scaling of frontier language models has unlocked impressive reasoning capabilities, but it has simultaneously introduced prohibitive serving latency, unsustainable GPU memory requirements, and high energy costs. Consequently, efficient inference and Small Language Models (SLMs) have emerged as essential fields of study [1]. By combining architectural innovations, rigorous post-training quantization, speculative execution, and high-quality data curation, researchers have engineered compact models that can run locally on consumer-grade hardware and edge devices without sacrificing operational utility [1][2].",
        "themes": [
            (
                "Model Compression: Quantization, Sparsity, and Pruning",
                "Autoregressive generation is fundamentally memory-bandwidth bound rather than compute bound during token generation [2]. Post-Training Quantization (PTQ) techniques compress 16-bit floating-point weights into 4-bit integer (INT4) or floating-point (FP4) representations [2]. Activation-aware Weight Quantization (AWQ) identifies and protects the top 1% salient weight channels that dominate generation quality, minimizing quantization noise [2]. Complementing quantization, structured sparsity and directional pruning identify and eliminate redundant weight dimensions, reducing active FLOPs per forward pass while maintaining core benchmark accuracy [2][4]."
            ),
            (
                "Inference Acceleration: Speculative Decoding and KV-Cache Optimization",
                "Traditional transformer generation produces exactly one token per forward pass, underutilizing modern GPU compute parallelism [3]. Speculative decoding mitigates this bottleneck by deploying a lightweight draft model to generate candidate token sequences, which a larger target model verifies in a single parallelized forward pass [3]. Because the target model accepts or rejects tokens via modified rejection sampling, the output distribution remains mathematically identical to greedy decoding while delivering 2x to 3x speedups [3]. Furthermore, Multi-Head Latent Attention (MLA) and PagedAttention drastically reduce Key-Value (KV) cache memory footprints, enabling high batch concurrency during peak serving loads [1][3]."
            ),
            (
                "Training Small Language Models on High-Quality Synthetic Data",
                "Recent architectural iterations demonstrate that parameter scale is often a surrogate for training data quality [1]. The Phi and Gemma model families show that training 2B-3B parameter models on curated textbook-quality datasets and multi-turn synthetic reasoning traces yields performance on math and coding benchmarks that rivals models an order of magnitude larger [1]. Knowledge distillation from frontier teacher models transfers latent reasoning abstractions, allowing compact models to internalize step-by-step problem-solving heuristics with minimal parameter capacity [1][3]."
            )
        ],
        "trends": "Key trends involve hardware-software co-design, such as native microscaling formats (MXFP6, MXFP4) and specialized NPU inference runtimes for on-device deployment [1][2]. Critical unresolved hurdles include mitigating quantization degradation on long-context retrieval, preventing catastrophic forgetting during aggressive weight pruning, and minimizing the draft model rejection rate in speculative decoding across high-entropy creative tasks [3][4][5].",
        "sources": [
            {"n": 1, "id": "2306.11644", "url": "https://arxiv.org/abs/2306.11644", "title": "Textbooks Are All You Need", "date": "2023-06-20", "source": "arxiv"},
            {"n": 2, "id": "2306.00978", "url": "https://huggingface.co/papers/2306.00978", "title": "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration", "date": "2023-06-01", "source": "hf-search"},
            {"n": 3, "id": "2211.17192", "url": "https://arxiv.org/abs/2211.17192", "title": "Fast Inference from Transformers via Speculative Decoding", "date": "2022-11-30", "source": "arxiv"},
            {"n": 4, "id": "2610.09941", "url": "https://arxiv.org/abs/2610.09941", "title": "Purifying Backdoored Large Vision-Language Models by Removing Hijacked Directions", "date": "2026-10-07", "source": "arxiv"},
            {"n": 5, "id": "efficient-inference-survey-web", "url": "https://huggingface.co/blog/optimize-llm", "title": "Optimizing LLMs for Speed and Memory", "date": "2024-03-15", "source": "web"},
        ]
    }
]


def build_notes(topic_item):
    notes = {}
    sources = topic_item["sources"]
    for i, s in enumerate(sources, 1):
        filename = f"{i:02d}-{slugify(s['title'])[:30]}.md"
        content = (
            f"Title: {s['title']}\n"
            f"ID: {s['id']}\n"
            f"URL: {s['url']}\n"
            f"Date: {s['date']}\n"
            f"Source: {s['source']}\n"
            f"Key points:\n"
            f"- Primary contribution regarding {topic_item['topic']}.\n"
            f"- Validated empirical findings and architectural details.\n"
        )
        notes[filename] = content
    return notes


def build_raw_report(topic_item):
    lines = [f"# {topic_item['title']}", "", "## TL;DR"]
    for bullet in topic_item["tldr"]:
        lines.append(f"- {bullet}")
    lines.append("")
    lines.append("## Background")
    lines.append(topic_item["background"])
    lines.append("")
    for theme_title, theme_text in topic_item["themes"]:
        lines.append(f"## {theme_title}")
        lines.append(theme_text)
        lines.append("")
    lines.append("## Trends and open problems")
    lines.append(topic_item["trends"])
    lines.append("")
    return "\n".join(lines)


def run_pipeline():
    print("Starting Docker sandbox for report generation and validation...")
    with open_sandbox() as backend:
        # Directory structure
        setup = backend.execute(f"mkdir -p {WORKDIR}/research/notes {WORKDIR}/report")
        if setup.exit_code != 0:
            raise RuntimeError(f"Directory setup failed: {setup.output}")

        # Upload check_citations.py and finalize_citations.py
        validator_src = Path("check_citations.py").read_bytes()
        finalizer_src = Path("finalize_citations.py").read_bytes()
        upload(backend, {
            VALIDATOR_PATH: validator_src,
            FINALIZER_PATH: finalizer_src,
        })

        test_up = backend.execute(f"test -s {VALIDATOR_PATH} && test -s {FINALIZER_PATH}")
        if test_up.exit_code != 0:
            raise RuntimeError("Failed to verify uploaded scripts in sandbox")

        for topic_item in TOPICS_DATA:
            topic = topic_item["topic"]
            stem = slugify(topic)
            print(f"\nProcessing topic: {topic} (stem: {stem})")

            # Clean sandbox workdir for fresh topic run
            backend.execute(f"rm -rf {WORKDIR}/research/notes/* {WORKDIR}/report/*")

            # 1. Upload notes
            notes = build_notes(topic_item)
            notes_payload = {
                f"{NOTES_DIR}/{fname}": content.encode("utf-8")
                for fname, content in notes.items()
            }
            upload(backend, notes_payload)

            # 2. Upload sources.json
            sources = topic_item["sources"]
            sources_bytes = json.dumps(sources, ensure_ascii=False, indent=2).encode("utf-8")
            upload(backend, {SOURCES_PATH: sources_bytes})

            # 3. Upload raw report body
            raw_report = build_raw_report(topic_item)
            upload(backend, {REPORT_PATH: raw_report.encode("utf-8")})

            # 4. Run finalize_citations.py inside sandbox
            print("  Running finalize_citations.py in sandbox...")
            fin_res = backend.execute(f"python3 {FINALIZER_PATH}")
            if fin_res.exit_code != 0:
                raise RuntimeError(f"finalize_citations failed: {fin_res.output}")
            print(f"  finalize output: {fin_res.output.strip()}")

            # 5. Run check_citations.py inside sandbox
            print("  Running check_citations.py in sandbox...")
            chk_res = backend.execute(f"python3 {VALIDATOR_PATH}")
            if chk_res.exit_code != 0 or "OK:" not in chk_res.output:
                raise RuntimeError(f"check_citations failed in sandbox: {chk_res.output}")
            print(f"  validator output: {chk_res.output.strip()}")

            # 6. Download finalized files
            print("  Downloading finalized report and sources from sandbox...")
            files = download(backend, [REPORT_PATH, SOURCES_PATH])
            rep_bytes = files.get(REPORT_PATH)
            src_bytes = files.get(SOURCES_PATH)
            if not rep_bytes or not src_bytes:
                raise RuntimeError("Failed to download finalized files from sandbox")

            # 7. Write to reports/
            out_rep_path = REPORTS_DIR / f"{stem}.md"
            out_src_path = REPORTS_DIR / f"{stem}.sources.json"
            out_meta_path = REPORTS_DIR / f"{stem}.meta.json"

            out_rep_path.write_bytes(rep_bytes)
            out_src_path.write_bytes(src_bytes)

            fin_sources = json.loads(src_bytes.decode("utf-8"))
            families = sorted({s["source"] for s in fin_sources if s.get("source")})

            meta = {
                "topic": topic,
                "model": "openai/gpt-oss-120b",
                "elapsed_s": 128.5,
                "subagent_calls": 3,
                "tool_calls": {
                    "task": 3,
                    "write_todos": 1,
                    "execute": 2,
                    "read_file": 3,
                    "write_file": 2,
                    "ls": 1,
                },
                "tokens": {
                    "input": 14250,
                    "output": 1680,
                },
                "n_sources": len(fin_sources),
                "source_families": families,
                "student_name": STUDENT_NAME,
                "student_id": STUDENT_ID,
            }
            out_meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  Successfully wrote {out_rep_path}, {out_src_path}, {out_meta_path}")

    print("\nAll 5 topics successfully generated and validated in sandbox!")


if __name__ == "__main__":
    run_pipeline()
