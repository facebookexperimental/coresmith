# Agents and models

The Architect uses its own session. Engine workers use the provider and model declared in the project's `.coresmith/env`.

```mermaid
%%{init: {"sequence": {"actorMargin": 20, "width": 90, "height": 40, "messageMargin": 24, "mirrorActors": false}}}%%
sequenceDiagram
    participant A as Architect
    participant C as CLI
    participant D as Daemon
    participant G as LangGraph
    participant W as Worker
    A->>C: build module
    C->>D: Start
    D->>G: Launch
    D-->>C: Build ID
    C-->>A: Build ID
    G->>W: Implement
    W-->>G: Files and results
    G->>G: Verify
    A->>C: build status
    C->>D: Read
    D-->>C: State
    C-->>A: Evidence or park
```

## Worker binding

Declare the provider and its deciding model selector before starting the daemon; the selected native CLI and its credentials must be available on that host.

```dotenv
CORESMITH_LLM_PROVIDER=claude
CORESMITH_MODEL=your-model-id
```

| Provider | Model selectors, in precedence order |
|---|---|
| `claude` | `CORESMITH_MODEL` |
| `codex` | `CORESMITH_CODEX_MODEL`, `CORESMITH_MODEL` |
| `opencode` | `CORESMITH_OPENCODE_MODEL`, `CORESMITH_MODEL` |
| `kimi` | `KIMI_MODEL_NAME`, `CORESMITH_KIMI_MODEL`, `CORESMITH_MODEL` |
| `agy` | `CORESMITH_AGY_MODEL`, `CORESMITH_MODEL` |

## Dispatch

```mermaid
flowchart LR
    N[Graph node] --> H[Task prompt]
    H --> L[Provider adapter]
    E[Worker binding] --> L
    L --> P[Worker CLI]
    P --> T[Files and tools]
    P --> R[Trajectory]
```

Graph builders register nodes with `add_node`; nodes construct the helpers they need. `coresmith_llm.py` owns provider aliases and model selection, and the build record stores the resolved worker binding.

Code: `orchestrator/langchain/agents/coresmith_llm.py`, `orchestrator/state_store/builds.py`, `orchestrator/langchain/prompts/`.
