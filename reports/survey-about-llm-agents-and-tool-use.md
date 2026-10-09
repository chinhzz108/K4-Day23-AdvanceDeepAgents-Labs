# A Survey of LLM Agents and Tool Use: Architectures, Interoperability, and Runtime Safety

## TL;DR
- Autonomous LLM agents combine foundational reasoning with external tool calling, memory stores, and dynamic planning [1].
- Standardized protocol frameworks such as the Model Context Protocol (MCP) enable secure host-client abstraction for tool calling and data integration [2].
- Multi-agent hierarchical delegation decouples high-level strategic decomposition from specialized tool execution environments [3].
- Real-time monitoring frameworks provide streaming structural anomaly detection to prevent agent trajectory drift and destructive loops [4].

## Background
Large language models operating in isolation are constrained by static pre-training cutoffs, lack of external actuation, and arithmetic hallucination. The autonomous agent paradigm overcomes these boundaries by positioning the LLM as a central reasoning engine coordinating memory systems, planning loops, and external application programming interfaces (APIs) [1]. Early agent frameworks demonstrated basic zero-shot tool usage, which has rapidly evolved into standardized protocol ecosystems and multi-agent coordination architectures [2][3]. In production environments, ensuring robust runtime trajectory monitoring and guarding against malicious prompt injection from untrusted web data represent vital design priorities [4].

## Tool Calling Mechanisms, Protocols, and Schema Alignment
Agent tool invocation transitioned from crude regex parsing of structured text outputs (e.g., ReAct) to native function-calling APIs trained into model weights [1]. Standardized interface protocols, most notably Anthropic's Model Context Protocol (MCP), decouple the LLM host from remote tool providers via standardized JSON-RPC transports [2]. This architectural separation isolates credentials from agent-accessible sandboxes and provides structured schema discovery [2]. Studies indicate that fine-tuning models on schema-guided tool calling yields significantly higher execution reliability compared to prompt-only in-context examples, especially when dealing with nested parameter schemas and dynamic API endpoints [1][2].

## Planning Paradigms and Hierarchical Multi-Agent Delegation
Complex workflows require long-horizon planning that exceeds the working memory of single monolithic agents [1]. Hierarchical multi-agent architectures resolve this bottleneck by splitting responsibilities between a lead coordinator agent and specialized subagents [3]. The lead performs goal decomposition and maintains task checklists (e.g., via TodoList middleware), while delegating concrete domain sub-tasks to isolated worker agents [3]. Empirical benchmarks show that delegating sub-tasks with bounded context windows prevents context pollution, reduces token expenditure, and isolates errors before they derail the primary workflow [1][3].

## Runtime Trajectory Monitoring and Sandbox Security
Deploying autonomous agents with code execution and filesystem capabilities introduces severe safety vulnerabilities, including prompt injection, infinite looping, and unintended data destruction [2][4]. Modern defense architectures isolate execution within stateless containerized sandboxes operating under strict network segregation [2]. Furthermore, runtime trajectory observers, such as streaming structure-aware monitors, track agent intermediate states and tool invocation patterns in real time [4]. When an agent exhibits repetitive retry behavior or deviates from its declared execution plan, the monitor triggers automated interventions or human-in-the-loop validation [4].

## Trends and open problems
Current developments emphasize interoperability protocols, multi-modal tool use (such as visual GUI interaction and computer control), and asynchronous subagent parallelism [2][3]. Crucial unresolved challenges include preventing indirect prompt injections embedded in web retrieval contexts, bounding token consumption across runaway agent loops, and maintaining persistent cross-session episodic memory without degrading reasoning latency [1][4].

## References
[1] The Rise and Potential of Large Language Model Based Agents: A Survey. arxiv. https://arxiv.org/abs/2308.11432 (2023-08-22)
[2] Model Context Protocol: Standardizing Tool and Context Integrations for AI Agents. web. https://modelcontextprotocol.io/introduction (2024-11-25)
[3] Mind2Web: Towards a Generalist Agent for the Web. hf-search. https://huggingface.co/papers/2309.07864 (2023-09-14)
[4] OnTrack: Real-Time Monitoring and Intervention in LLM Agent Trajectories via Streaming Structure-Aware Verification. arxiv. https://arxiv.org/abs/2610.12375 (2026-10-08)
