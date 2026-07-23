# Pipeline overview

coresmith is a **daemon**, not a CLI. The CLI (`bin/coresmith`) auto-discovers a long-running FastAPI process per project root and shells to its HTTP endpoints. The daemon owns the LangGraph runtimes and the SQLite checkpoints; the CLI just observes and resumes them.

## End-to-end flow

```mermaid
flowchart TB
    subgraph CLI/MCP
      C[bin/coresmith CLI]
      M[mcp_server.py - stdio MCP]
    end
    subgraph Daemon[coresmithd - FastAPI]
      L[graph_lifecycle.py]
      A[Architecture graph]
      P[Pipeline graph]
      B[Backend graph]
      T[Tapeout graph]
    end
    subgraph Disk[(project_root)]
      RQ[inputs/requirements.md]
      ARCH[arch/*.md, *.json]
      RTL[rtl/*.v]
      TB[tb/cocotb/*.py]
      SYN[syn/output/*]
      SUB[openframe_submission/]
      CK[.coresmith/*.db checkpoints]
    end
    C -- HTTP --> L
    M -- stdio --> L
    L --> A --> ARCH
    L --> P --> RTL
    P --> TB
    P --> SYN
    L --> B --> SYN
    L --> T --> SUB
    A & P & B & T -.checkpoint.-> CK
```

The pipeline is **LLM-driven, not deterministic**. Every code-gen and every diagnose step calls an LLM (Claude CLI or Codex CLI). The LLM has shell, file-edit, and search tools and writes RTL, testbenches, and physical-design TCL directly to disk.

## Why three graphs and not one

Each graph has its own SQLite checkpoint and its own thread ID. Splitting the run lets you:

- **Pause** at logical boundaries: architecture sign-off (OK2DEV) is a separate gate from pipeline completion.
- **Re-enter** a downstream graph without re-running the upstream one (e.g. retry backend on the same frontend RTL).
- **Bound the blast radius** of corruption: a parked frontend checkpoint cannot disturb the architecture checkpoint.

Each graph is built by a `build_*_graph()` function in `orchestrator/langgraph/`, compiled with an `AsyncSqliteSaver` checkpointer, and driven by `GraphLifecycle` in `orchestrator/graph_lifecycle.py`.

## The disk-first contract

Graph state holds **routing markers and small dicts**, never the large artifacts. RTL, testbenches, GDS, DEF, log files — all live on disk under the project root. State holds:

- **File paths** (e.g. `rtl_path`, `tb_path`, `routed_def_path`).
- **Routing flags** (e.g. `lint_clean`, `sim_passed`, `synth_success`).
- **Structured results** (small JSON dicts from EDA tool parsers / LLM auditors).
- **Accumulators** (e.g. `completed_blocks` with reducer `operator.add`).

Outer agents and humans inspect the on-disk files; the graph just remembers where they are. Re-runs reuse on-disk content when its mtime is newer than `pipeline_run_start` (see `_file_is_fresh` in `orchestrator/langgraph/pipeline_graph.py:172`).

## The interrupt-and-resume contract

Every graph parks on `interrupt(payload)` rather than auto-approving anything important. The outer agent (Claude on cron, a human, or another script) reads the payload and resumes the graph with one of the `supported_actions` listed in the payload.

| Interrupt | Common payload type | Typical resume actions |
|---|---|---|
| Architecture PRD questions | `prd_questions` | `continue`, `abort` |
| Architecture block diagram | `architecture_review_needed` | `continue`, `feedback`, `abort` |
| Constraint structural violations | `architecture_review_needed` | `retry`, `accept`, `feedback`, `abort` |
| Final review (OK2DEV) | `final_review` | `accept`, `feedback`, `abort` |
| Per-block RTL/TB failure | `human_intervention_needed` | `retry`, `fix_rtl`, `fix_tb`, `add_constraint`, `skip`, `abort` |
| Tier integration review | `uarch_integration_review` | `approve`, `revise`, `abort` |
| Integration DV failure | `integration_dv_failure` | `retry`, `fix_rtl`, `fix_tb`, `abort` |
| Validation DV failure | `validation_dv_failure` | `retry`, `fix_rtl`, `fix_tb`, `abort` |
| Backend EDA failure | `human_intervention_needed` (graph=backend) | `retry`, `skip`, `abort` |
| Tapeout failure | `tapeout_intervention_needed` | `retry`, `fix_pnr`, `skip`, `abort` |

Full payload schemas and resume semantics are in the [Interrupts catalog](../reference/interrupts.md).

## What runs on what model

- `DEFAULT_MODEL = "opus-4.8"` for load-bearing steps (integration lead, integration review, contract audit, backend EDA).
- `BLOCK_MODEL = "sonnet-4.6"` for per-block agents (uArch, RTL, TB, debug).
- Set `CORESMITH_MODEL` and `CORESMITH_BLOCK_MODEL` to override.
- Set `CORESMITH_LLM_PROVIDER=codex` to swap Claude CLI for Codex CLI (gpt-5.6) — same prompts, different backend.

See [LLM abstraction](../agents/llm-abstraction.md) for details.
