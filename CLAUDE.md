# CLAUDE.md

Guidance for AI agents working on this repo. Keep this file under ~200 lines.

## What socmate is

AI-orchestrated ASIC pipeline. Three LangGraph state machines run in sequence:

1. **Architecture** (`orchestrator/architecture/`, `orchestrator/langgraph/architecture_graph.py`) — PRD → block diagram → SAD/FRD/ERS → constraint check → final-review gate.
2. **Frontend pipeline** (`orchestrator/langgraph/pipeline_graph.py`) — per-block loop: generate uArch spec → generate RTL → Verilator lint → generate cocotb TB → simulate → synthesize → diagnose/retry on failure. After each tier: uArch integration review. After all tiers: integration check (chip_top.v generation + lint) → integration DV → validation DV.
3. **Backend** (`orchestrator/langgraph/backend_graph.py`) — PnR, DRC, LVS, signoff. Driven via MCP, not headless.

The pipeline is **LLM-driven**, not deterministic. Every code-gen + diagnose step calls an LLM (Claude or Codex). The LLM has shell/file-edit tools and writes RTL/TB directly to disk.

## How to run

Two entry points:

1. **`run_pipeline.py`** — the canonical CLI. Pure frontend pipeline, no per-design hardcoding. Reads `orchestrator/config.yaml` (or `SOCMATE_BLOCKS_FILE` override) for the block list. Auto-approves uArch reviews and retries blocks up to `MAX_ATTEMPTS`. Does not pause for outer-agent decisions; interrupts are handled inline.

   ```bash
   make pipeline                         # the default config
   SOCMATE_BLOCKS_FILE=examples/<design>/blocks.yaml make pipeline
   ```

2. **`scripts/manual_graph_segment.py`** — one graph segment per invocation, exits with the current state. The outer agent (you) inspects, decides, then calls the script again with the next resume action. Use this when you want explicit control of every approve/feedback/abort decision and you don't want to write a runner.

   ```bash
   venv/bin/python scripts/manual_graph_segment.py \
       --project-root <run-dir>/project \
       start-architecture --requirements <path>
   # inspect state, then:
   venv/bin/python scripts/manual_graph_segment.py \
       --project-root <run-dir>/project \
       resume-architecture --action accept|feedback|continue --feedback "<text>"
   ```

If you need a long-running headless driver, write it for your specific design — don't put per-design KPI answers, freeform-question heuristics, or codec/transformer/whatever auto-defaults in this repo. Those belong with the design (e.g. in PPABench collateral). The orchestrator stays generic.

**Always use an isolated run dir** for non-trivial work, never `make pipeline` in the repo root. Convention:

```
/home/ubuntu/socmate-runs/<task>-<flavor>-<YYYYMMDD-HHMMSS>/
  inputs/requirements.md     # copy of the spec
  project/                   # SOCMATE_PROJECT_ROOT
    .socmate/                # pipeline state, checkpoints, escalations, traces
    rtl/ tb/ arch/           # generated collateral
  launch.sh                  # cd /home/ubuntu/socmate && exports + run_pipeline.py (or manual_graph_segment.py loop)
  run.log
```

Required env for Codex backend (the project's actually-used default; the README still shows Claude):
```bash
export SOCMATE_PROJECT_ROOT=<run-dir>/project
export SOCMATE_LLM_PROVIDER=codex
export SOCMATE_CODEX_MODEL=gpt-5.5
export SOCMATE_MODEL=gpt-5.5
export SOCMATE_BLOCK_MODEL=gpt-5.5
export SOCMATE_CODEX_SANDBOX=danger-full-access
export SOCMATE_SKIP_SYNTH=1               # unless Sky130 PDK is local
export SOCMATE_ENABLE_MEMORY_MAP=0
export SOCMATE_ENABLE_CLOCK_TREE=0
export SOCMATE_ENABLE_REGISTER_SPEC=0
```

`model_reasoning_effort = "high"` is set globally in `~/.codex/config.toml`. Trust the run dir in `~/.codex/config.toml` (`[projects."<dir>"] trust_level = "trusted"`) before launching, otherwise Codex prompts.

## Outer-agent escalation contract

The pipeline itself only handles per-block retries inline. Higher-level interrupts (architecture review, uArch integration review, integration_dv / validation_dv failures) raise out of the graph and are answered by whatever driver is wrapping the LangGraph state machine. If you write such a driver, the contract is: write the interrupt payload as JSON to `<run-dir>/.socmate/escalations/<kind>.json`, poll for `<kind>.decision.json`, then call `mcp.resume_*` with that decision. As the outer agent you:

1. Read `<kind>.json` (interrupt payload + contract audit + reference files).
2. Decide `action` (`accept | feedback | continue | abort` for architecture; `approve | retry | skip | abort | fix_rtl | fix_tb | revise` for pipeline).
3. Write `<kind>.decision.json` with `action`, `rationale`, plus `feedback` / `rtl_fix_description` / `block_actions` as applicable.

`fix_rtl` and `fix_tb` mean **you already edited the RTL/TB on disk** — the pipeline trusts and re-runs the failing stage. The pipeline does **not** auto-call an LLM to fix on those actions.

## Common pitfalls & how to handle them

- **Don't `rm -rf .socmate/`** — rename it: `.socmate.cleared-<reason>-<ts>/`, `.socmate.failed-<reason>-<ts>/`, `.socmate.aborted-<reason>-<ts>/`. The repo root is littered with archives because they're valuable for forensics.
- **Architectural decisions during integration review**: the integration-review LLM agent edits uArch specs on **every run**, so `issues_fixed > 0` is the steady state. Default behavior now honors `action: approve` despite `issues_fixed > 0`; set `SOCMATE_STRICT_INTEGRATION_REVIEW=1` to restore the old auto-`revise` on stale RTL. Sending `revise` without `block_actions` is a no-op that strands the pipeline at `status=done` with `next_nodes=[]`; use `restart_block(from_node='generate_rtl')` per affected block + then `approve`.
- **Integration Lead JSON-vs-disk mismatch**: the agent (Codex) sometimes writes the full chip_top via its file-edit tool and then returns `{"verilog": "\`include \"<output_path>\""}`. The integration_lead.py now detects this and prefers the on-disk file; if neither source has a real module declaration it raises so retry triggers.
- **Validation DV TB module name**: cocotb's `MODULE` is `Path(tb_path).stem`. `run_integration_simulation` preserves the original stem on copy so `test_<design>_validation.py` resolves. If you see `No module named test_<design>_validation` at 0 ns, the copy logic regressed.
- **Pipeline exit code 1 with `pipeline_done=false`, `completed_count<total_blocks`, `next_nodes=[]`**: graph fell into a terminal state with work still pending. Almost always the integration_review revise-loop or a node that ran and exited without advancing. Clear `pipeline_checkpoint.db*` and relaunch with `--skip-architecture`.
- **`make pipeline` exits early**: see `run_pipeline.py` — `[PIPELINE GATE FAILED]` means a per-block fail, while a normal `PIPELINE COMPLETE` at `12/13` blocks is the integration-review-stuck symptom from above.
- **Generated RTL/TB look stale**: the pipeline reuses existing files when their mtime is newer than the spec mtime. If you regenerate a uArch spec but the RTL doesn't refresh, the spec mtime is older than the RTL. Use `mcp.restart_block(block_name, from_node='generate_rtl')`.

## Where things live

- `orchestrator/langgraph/` — graphs (architecture, pipeline, backend) and helpers (`integration_helpers.py`, `pipeline_helpers.py`).
- `orchestrator/langchain/agents/` — LLM-agent wrappers (RTL gen, TB gen, integration lead, integration review, debug, etc.).
- `orchestrator/langchain/prompts/` — system prompts. Edits here change LLM behavior across runs.
- `orchestrator/architecture/specialists/` — per-architecture-step modules (PRD, block diagram, SAD, FRD, ERS, constraint check, final review).
- `orchestrator/mcp_server.py` — the MCP entry point. Long file; key tools: `start_architecture`, `start_pipeline`, `resume_*`, `restart_block`, `restart_node`, `get_*_state`.
- `run_pipeline.py` (repo root) — generic frontend CLI runner. Auto-approves uArch reviews and retries blocks; suited for unattended runs of well-debugged designs.
- `scripts/manual_graph_segment.py` — one-segment-at-a-time runner for outer-agent control. Used when you want to inspect state and write each resume decision yourself.
- `examples/<design>/{requirements.md,blocks.yaml,...}` — design specs the pipeline consumes. Goldens (Python reference implementations) live next to them.
- `tb/` is **in `.gitignore`** — generated cocotb TBs, RD harnesses, integration TBs. Design-specific test harnesses do not belong in the socmate repo; put them in ppabench under `evaluation/sample_collateral/<run>/rd_harness/` instead.
- `rtl/` — generated Verilog. **Do not hand-edit and commit** — these are pipeline outputs. The repo root only commits canonical reference RTL for examples; per-run RTL lives in the run dir.
- `tests/` — fast unit + integration tests. `pytest -m "not live_llm and not requires_nix and not e2e"` runs everything that doesn't need an LLM or EDA toolchain.

## Conventions for AI edits

- **Generic-only in socmate**: every fix here must work for any codec/design. Codec-specific harnesses or scripts (e.g. RD wrappers tied to a particular top-level port set) go in **ppabench `evaluation/sample_collateral/.../rd_harness/`**, not in socmate `scripts/` or `tb/rd/`.
- **Pipeline behavior changes are gated** by env vars when they change observable outputs. Existing example: `SOCMATE_STRICT_INTEGRATION_REVIEW=1` restores pre-fix behavior. Follow this pattern.
- **Don't commit generated artifacts**. `.socmate/`, `sim_build/`, and `tb/` are ignored for a reason. If you find yourself running `git add -f` in any of those, stop and put the file somewhere else.
- **Don't touch other in-flight uncommitted changes**. The repo regularly has 5-10 modified files from prior in-progress work. Use `git add -- <specific files>` to stage only what you authored.
- **Tests must cover both branches** when you gate behavior on an env var. `orchestrator/tests/test_pipeline_graph.py::TestRouteAfterIntegrationReview` is the template: one test with the env var unset, one with it set.
- **Don't kill running `codex`/`claude` processes** — there is often a parallel manual run going. Check `ps -ef | grep -E "codex|claude"` and inspect cwd via `/proc/<pid>/cwd` if uncertain.

## Useful commands

```bash
# Inspect generated pipeline traces (OTel spans in SQLite)
make traces

# Fast tests (no LLM, no EDA)
pytest orchestrator/tests/ -v -m "not live_llm and not requires_nix and not e2e"

# Generic RD eval for any codec design (encoder-side PSNR from m_axis_recon_check)
# -- the harness itself is design-specific; live in ppabench

# Find the active run dir
ls -dt /home/ubuntu/socmate-runs/*/ | head -3

# Watch a run's escalations queue (you, as outer agent, write decision JSON here)
ls -la <run-dir>/project/.socmate/escalations/
```

## When in doubt

- Read `<run-dir>/project/.socmate/pipeline_events.jsonl` for the last 50 graph_node_enter / graph_node_exit events.
- Read `<run-dir>/project/.socmate/contract_audit/*.json` — the contract audit is the LLM's structured diagnosis of why integration/validation DV failed. It includes `first_divergence` (specific signal trace from VCD), `affected_blocks`, `recommended_action`, `suggested_fix`.
- The contract audit is reliably more precise than its `confidence` score suggests. If it says `local_fix_possible: true` and points at specific RTL lines, the fix is usually small.
