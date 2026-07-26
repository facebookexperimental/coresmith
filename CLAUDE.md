# CLAUDE.md

Guidance for AI agents working on this repo. Keep this file under ~200 lines.

## What coresmith is

AI-orchestrated ASIC pipeline. Three LangGraph state machines run in sequence:

1. **Architecture** (`orchestrator/architecture/`, `orchestrator/langgraph/architecture_graph.py`) — PRD → block diagram → SAD/FRD/ERS → constraint check → final-review gate.
2. **Frontend pipeline** (`orchestrator/langgraph/pipeline_graph.py`) — per-block loop: generate uArch spec → generate RTL → Verilator lint → generate cocotb TB → simulate → synthesize → diagnose/retry on failure. After each tier: uArch integration review. After all tiers: integration check (chip_top.v generation + lint) → integration DV → validation DV.
3. **Backend** (`orchestrator/langgraph/backend_graph.py`) — PnR, DRC, LVS, signoff. Driven via MCP, not headless.

The pipeline is **LLM-driven**, not deterministic. Every code-gen + diagnose step calls an LLM (Claude or Codex). The LLM has shell/file-edit tools and writes RTL/TB directly to disk.

## Architecture: coresmithd daemon + outer agent

coresmith is a **daemon**, not a CLI. There is no `run_pipeline.py` auto-approver any more. The flow is:

1. **`orchestrator/daemon/server.py`** — FastAPI service, one process per `CORESMITH_PROJECT_ROOT`. Wraps the LangGraph via `orchestrator/graph_lifecycle.py`. Writes `<project_root>/.coresmith/daemon.json` ({port, pid}) on startup so clients can discover it. Endpoints: `POST /run/start`, `GET /run/state`, `POST /run/resume`, `POST /run/pause`, `GET /healthz`.
2. **`bin/coresmith`** — CLI client. Auto-discovers the daemon via daemon.json and shells to its HTTP endpoints. Subcommands: `daemon start|stop|status`, `run start|pause`, `state`, `resume`, `logs`.
3. **Outer agent (Claude on cron)** — the **autochecker**. The pipeline parks on every interrupt — it never auto-approves. A scheduled Claude invocation runs `coresmith state`, inspects the pending interrupt's payload, decides, and runs `coresmith resume --action ...`. If the daemon is dead or the run is stuck, the agent restarts it.

```
┌────────────────┐     HTTP      ┌─────────────────────┐     LangGraph
│  bin/coresmith │ ◀──────────▶ │    coresmithd       │ ◀──────────────▶ pipeline_graph
│  CLI           │              │  (FastAPI daemon)   │                  + SQLite checkpoint
└────────────────┘              └─────────────────────┘
        ▲                                ▲
        │ exec                           │ daemon.json (port, pid)
┌───────┴──────────────────────────┐    │
│ Claude on cron (10-min cadence)  │────┘
│  - reads escalation state         │
│  - decides + resumes              │
│  - fixes/restarts on failure      │
└───────────────────────────────────┘
```

## How to run

```bash
# Pick / create the run directory
RUN_DIR=/home/ubuntu/coresmith-runs/<task>-<flavor>-$(date +%Y%m%d-%H%M%S)
mkdir -p $RUN_DIR/inputs && cp <spec>.md $RUN_DIR/inputs/requirements.md

# Export the project root + LLM provider env vars (Codex shown; Claude also works)
export CORESMITH_PROJECT_ROOT=$RUN_DIR
export CORESMITH_LLM_PROVIDER=codex
export CORESMITH_CODEX_MODEL=gpt-5.6
export CORESMITH_MODEL=gpt-5.6
export CORESMITH_BLOCK_MODEL=gpt-5.6
export CORESMITH_CODEX_SANDBOX=danger-full-access
export CORESMITH_SKIP_SYNTH=1            # unless Sky130 PDK is local
export CORESMITH_ENABLE_MEMORY_MAP=0
export CORESMITH_ENABLE_CLOCK_TREE=0
export CORESMITH_ENABLE_REGISTER_SPEC=0

# Start the daemon
bin/coresmith daemon start --project-root $RUN_DIR

# (1) Architecture phase — generates PRD, ERS, FRD, block_diagram.json
#     and the initial arch/uarch_specs/<block>.md files from a
#     natural-language requirements doc. Parks at PRD review, block
#     diagram review, constraint check, etc.; outer agent resumes
#     each via `coresmith resume`.
bin/coresmith architecture start --project-root $RUN_DIR \
    --requirements $RUN_DIR/inputs/requirements.md

# (2) Frontend pipeline — runs against the architecture artifacts from (1)
#     OR against a hand-written examples/<design>/blocks.yaml if you are
#     intentionally skipping the architecture phase. Skipping is supported
#     but the daemon will print a warning at /run/start because
#     `integration_review` then can't verify cross-block data_width and
#     `validation_dv` soft-aborts on missing ERS. Set
#     CORESMITH_SKIP_ARCH_WARN=1 to silence the warning for rapid
#     iteration / batch evaluation runs where you only care about the
#     per-block frontend loop.
bin/coresmith run start --project-root $RUN_DIR \
    --blocks-file /home/ubuntu/coresmith/examples/<design>/blocks.yaml

# Inspect / drive
bin/coresmith state --project-root $RUN_DIR
bin/coresmith resume --project-root $RUN_DIR --action approve
bin/coresmith logs  --project-root $RUN_DIR -n 100
```

`model_reasoning_effort = "high"` is set globally in `~/.codex/config.toml`. `/home/ubuntu` is already trusted there, so subdirs need no extra entry.

**Always use an isolated run dir** — never start the daemon against the repo root. Convention:

```
/home/ubuntu/coresmith-runs/<task>-<flavor>-<YYYYMMDD-HHMMSS>/
  inputs/requirements.md     # copy of the spec
  blocks.yaml                # design-specific blocks override (optional)
  .coresmith/                # daemon.json, checkpoints, events, escalations, traces
  rtl/  tb/  arch/           # generated collateral
```

## What "done" means (DELIVERABLE vs milestones) — read before declaring success

**The deliverable is a VERIFIED `chip_top` RTL: `integration_check` accepted → `integration_dv` PASS → `validation_dv` PASS (→ backend GDS with clean DRC/LVS + closed timing).** Everything before that is a MILESTONE, not the deliverable:

- The **Amaranth model** (the composition / `uarch_integration_gate` byte-exact pass, `derate_ledger measured=99`) proves the *design intent*. It is **NOT the chip.** A byte-exact model can sit atop RTL that is wrong — the model and the RTL are separate transcriptions (the RTL is lowered/verified separately).
- **`pipeline_done=True`** and the log line **`PIPELINE COMPLETE: N/N blocks passed`** mean the **PER-BLOCK FRONTEND** is complete (every block passed its *own* DV). They do **NOT** mean the chip is integrated or verified. The run can be `pipeline_done=True` AND `status=interrupted` (parked at `integration_check`/`integration_dv`/`validation_dv`) at the same time — that is NOT done.
- After per-block completion the run enters **integration**: `integration_check` (assembles `chip_top`, parks for `accept` — lint-clean ≠ functionally-correct) → `integration_dv` (chip_top vs golden) → `validation_dv` (ERS/KPI). The outer agent MUST drive each of these; they do not auto-run.

**Before reporting a design "closed"/"done"/"working", confirm `integration_dv` AND `validation_dv` actually PASSED (not just per-block / not just the model gate), and ideally simulate the real `chip_top` RTL on representative content.** Do not infer the chip from `pipeline_done`, a `PIPELINE COMPLETE` string, or a byte-exact model.

## Outer-agent decision contract

The pipeline raises interrupts at: `uarch_spec_review`, `uarch_integration_review`, per-block retry/skip choices, `integration_check` accept, `integration_dv` / `validation_dv` failures. **There is no auto-approve.** Every interrupt parks the graph; only an outer-agent `coresmith resume` advances it.

When `coresmith state` shows `pending_interrupt_count > 0`:

1. Read the `interrupts[0].payload` blob. Fields you care about: `type`, `block_name`, `previous_error`, `supported_actions`, `tier`, `issues_found`.
2. Pick an action from the payload's `supported_actions` (or the defaults below).
3. Run `coresmith resume --action <action> [--feedback "..."] [--rtl-fix-description "..."] [--block-actions '{"blk": "retry"}']`.

| Interrupt `type`            | Common decisions                                                                  |
|-----------------------------|-----------------------------------------------------------------------------------|
| `uarch_spec_review`         | `approve` (almost always), `revise` with `--feedback "..."`                       |
| `uarch_integration_review`  | `approve` (default; integration reviewer always edits something), `revise` + `--block-actions` |
| `pipeline_incomplete`       | `abort` (graph gate failed)                                                       |
| per-block ask_human         | `retry` (attempt < max), `skip` (exhausted), `fix_rtl`/`fix_tb` (you patched disk) |
| `integration_dv` / `validation_dv` | `approve` if the run is acceptable, `revise` to re-debug, `fix_rtl`/`fix_tb` if you patched, `abort` |

`fix_rtl` / `fix_tb` mean **you already edited the file on disk** — the pipeline re-runs the failing stage trusting your edit, it does *not* call an LLM.

## Common pitfalls & how to handle them

- **Don't `rm -rf .coresmith/`** — rename it: `.coresmith.cleared-<reason>-<ts>/`, `.coresmith.failed-<reason>-<ts>/`, `.coresmith.aborted-<reason>-<ts>/`. The repo root is littered with archives because they're valuable for forensics.
- **Architectural decisions during integration review**: the integration-review LLM agent edits uArch specs on **every run**, so `issues_fixed > 0` is the steady state. Default behavior now honors `action: approve` despite `issues_fixed > 0`; set `CORESMITH_STRICT_INTEGRATION_REVIEW=1` to restore the old auto-`revise` on stale RTL. Sending `revise` without `block_actions` is a no-op that strands the pipeline at `status=done` with `next_nodes=[]`; use `restart_block(from_node='generate_rtl')` per affected block + then `approve`. (The HTTP daemon currently exposes start/state/resume/pause; for `restart_block` use the MCP server.)
- **Integration Lead JSON-vs-disk mismatch**: the agent (Codex) sometimes writes the full chip_top via its file-edit tool and then returns `{"verilog": "\`include \"<output_path>\""}`. The integration_lead.py now detects this and prefers the on-disk file; if neither source has a real module declaration it raises so retry triggers.
- **Validation DV TB module name**: cocotb's `MODULE` is `Path(tb_path).stem`. `run_integration_simulation` preserves the original stem on copy so `test_<design>_validation.py` resolves. If you see `No module named test_<design>_validation` at 0 ns, the copy logic regressed.
- **Pipeline parked at `status=done, completed_count<total_blocks, next_nodes=[]`**: graph fell into a terminal state with work still pending. Almost always the integration_review revise-loop or a node that ran and exited without advancing. Stop the daemon, archive `.coresmith.failed-<ts>/`, and relaunch.
- **Generated RTL/TB look stale**: the pipeline reuses existing files when their mtime is newer than the spec mtime. If you regenerate a uArch spec but the RTL doesn't refresh, the spec mtime is older than the RTL. Use `mcp.restart_block(block_name, from_node='generate_rtl')`.
- **Daemon does not come up**: check `<project_root>/.coresmith/daemon.log` — port conflict, missing PDK, or preflight failure are the usual culprits.

## Where things live

- `orchestrator/langgraph/` — graphs (architecture, pipeline, backend) and helpers (`integration_helpers.py`, `pipeline_helpers.py`).
- `orchestrator/graph_lifecycle.py` — start / resume / pause / checkpoint plumbing shared by the MCP server and the FastAPI daemon.
- `orchestrator/daemon/server.py` — the **coresmithd** FastAPI daemon. One per project_root.
- `bin/coresmith` — the CLI client. Auto-discovers the daemon by reading `daemon.json`.
- `orchestrator/mcp_server.py` — the MCP entry point. Same graph plumbing, but the transport is stdio MCP for Claude CLI / Cursor. Key tools: `start_architecture`, `start_pipeline`, `resume_*`, `restart_block`, `restart_node`, `get_*_state`.
- `orchestrator/langchain/agents/` — LLM-agent wrappers (RTL gen, TB gen, integration lead, integration review, debug, etc.).
- `orchestrator/langchain/prompts/` — system prompts. Edits here change LLM behavior across runs.
- `orchestrator/architecture/specialists/` — per-architecture-step modules (PRD, block diagram, SAD, FRD, ERS, constraint check, final review).
- `examples/<design>/{requirements.md,blocks.yaml,...}` — design specs the pipeline consumes. Goldens (Python reference implementations) live next to them.
- `tb/` is **in `.gitignore`** — generated cocotb TBs, RD harnesses, integration TBs. Design-specific test harnesses do not belong in the coresmith repo; keep them in a downstream design-collateral tree instead.
- `rtl/` — generated Verilog. **Do not hand-edit and commit** — these are pipeline outputs. The repo root only commits canonical reference RTL for examples; per-run RTL lives in the run dir.
- `tests/` — fast unit + integration tests. `pytest -m "not live_llm and not requires_nix and not e2e"` runs everything that doesn't need an LLM or EDA toolchain.

## Conventions for AI edits

- **Generic-only in coresmith**: every fix here must work for any codec/design. Design-specific harnesses or scripts (e.g. RD wrappers tied to a particular top-level port set) belong in a downstream design-collateral tree, not in coresmith `scripts/` or `tb/rd/`.
- **Pipeline behavior changes are gated** by env vars when they change observable outputs. Existing example: `CORESMITH_STRICT_INTEGRATION_REVIEW=1` restores pre-fix behavior. Follow this pattern.
- **Don't commit generated artifacts**. `.coresmith/`, `sim_build/`, and `tb/` are ignored for a reason. If you find yourself running `git add -f` in any of those, stop and put the file somewhere else.
- **Don't touch other in-flight uncommitted changes**. The repo regularly has 5-10 modified files from prior in-progress work. Use `git add -- <specific files>` to stage only what you authored.
- **Tests must cover both branches** when you gate behavior on an env var. `orchestrator/tests/test_pipeline_graph.py::TestRouteAfterIntegrationReview` is the template: one test with the env var unset, one with it set.
- **Don't kill running `codex`/`claude` processes** — there is often a parallel manual run going. Check `ps -ef | grep -E "codex|claude"` and inspect cwd via `/proc/<pid>/cwd` if uncertain.

## Useful commands

```bash
# Daemon health for a run
bin/coresmith daemon status --project-root <run-dir>

# Inspect generated pipeline traces (OTel spans in SQLite)
make traces

# Fast tests (no LLM, no EDA)
pytest orchestrator/tests/ -v -m "not live_llm and not requires_nix and not e2e"

# Find the active run dir
ls -dt /home/ubuntu/coresmith-runs/*/ | head -3
```

## When in doubt

- `bin/coresmith state` is the source of truth for the run; `daemon.log` for the daemon itself; `pipeline_events.jsonl` for the LangGraph node-by-node timeline.
- Read `<project_root>/.coresmith/contract_audit/*.json` — the contract audit is the LLM's structured diagnosis of why integration/validation DV failed. It includes `first_divergence` (specific signal trace from VCD), `affected_blocks`, `recommended_action`, `suggested_fix`.
- The contract audit is reliably more precise than its `confidence` score suggests. If it says `local_fix_possible: true` and points at specific RTL lines, the fix is usually small.
