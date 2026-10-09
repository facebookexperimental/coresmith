# AGENTS.md

Guidance for coding agents (Claude Code, Codex, ...) driving or editing CoreSmith.
Full docs: the [CoreSmith book](docs/gitbook/README.md)
and [docs/CLI_TACTICS.md](docs/CLI_TACTICS.md) (every verb, refusal code, env var).

## Your role: the Architect

You are the Architect: you turn the human's intent into design collateral and
advance the chip **only** through `coresmith`. The engine never prompts, resumes
or decides for you, and nothing is auto-approved. Every park waits for your answer.

- `.coresmith/project.sqlite` is the chip state (requirements, contracts, pins,
  stages, builds, checks, parks). Prose, RTL, testbenches and SystemC models are
  files you write and `register`.
- Every verb takes `--json` and `--project-root` (or `$CORESMITH_PROJECT_ROOT`).
  A refusal writes nothing and prints a stable code plus `blocked_by`. Fix
  what it names and run the verb again; don't look for a bypass flag.
- `coresmith <verb> --help` and `coresmith schema <kind>` show the exact flags and
  document shapes. Check them before guessing.

## Setup

```bash
export CORESMITH_PROJECT_ROOT=/path/to/run    # one isolated dir per chip, never the repo root
mkdir -p $CORESMITH_PROJECT_ROOT/.coresmith
cat >> $CORESMITH_PROJECT_ROOT/.coresmith/env <<'EOF'
CORESMITH_LLM_PROVIDER=claude                 # claude | codex | opencode | kimi | agy
CORESMITH_MODEL=<model-id>                    # worker model (codex: CORESMITH_CODEX_MODEL)
EOF
coresmith daemon start
```

`.coresmith/env` is the worker binding the engine's own LLM workers use. Its
values override the shell environment, and the daemon re-reads it before each
launch. A missing binding blocks builds with `WORKER_UNBOUND`.

## The flow: a stage machine

`coresmith stage status` lists what blocks the next stage (and any `regressed`
stages). `coresmith stage next` advances once nothing blocks. Run both often.

| Stage | Your work (verbs) | Exit evidence |
|---|---|---|
| requirements | `prd add`, `frd add` (PERF/TIME need `--metric` and `--min`/`--max`), `frd verifier`, or bulk `register prd\|frd\|ers` | Registered requirements with must-haves and acceptance |
| decomposition | write + `register block_diagram`, `register sad` | Modules and system architecture |
| interfaces | `contract add` (block↔block edges), `pin add` (chip I/O), `fabric init` (bus), `register abi`, then `vip generate`, `shell assemble` | Every block port is on exactly one edge or pin; shell elaborates; edges lock on exit |
| uarch | `register uarch --block <b> <md>`, `target bind <b> --file targets/<b>.json`, SystemC `model/<b>_model.cpp` + `model build`, `model eval` | Each module ready: spec, HDL target, built model, declared model checks passing |
| blocks | `build targets <b>` (dry-run readiness), `build module <b>`, `build status`, `build resume` | Every module has a current, completed recorded build |
| integration | `run start` (reuses current builds; assembles `chip_top`; integration DV) | Integration DV passes on the current composition |
| acceptance | answer `run` parks, `resume` | Validation DV (ERS) passes |
| backend | `backend start [--full]`, `backend state`, `backend resume` | DRC/LVS/timing signoff `ppa_ok` |

What the main verbs are for:

- **`build module <b>`** is the only way a block gets done. It records its inputs
  (spec, contracts, target, model, FRD targets, acceptance TB, worker binding,
  tool/PDK/clock), runs the module graph (RTL → lint → assertions → DV +
  coverage → synth + timing → FRD targets), and publishes only on measured
  receipts. `--seed-rtl <file>` builds your own RTL without regenerating it.
  Changing any input makes the pass stale, so start a new build.
- **`verify rtl|synth|chip`, `tool run_*`, `block-status`, `block-done`** are
  diagnostics on files already on disk. Use them to iterate quickly. They never
  satisfy a stage. Neither does a hand-written `check add`.
- **`model author --block <b>` / `harness author`** explicitly ask an engine
  worker to author a SystemC model or eval harness. Checks such as `model build`
  and `model eval` never author anything.
- **`ruling add`** records a binding policy that every worker prompt receives.
  **`question add|answer`** keeps open decisions attached to items.
- **`status`, `item show`, `frd show`, `build show|compare|lineage`,
  `actions`** are read-only views of state, evidence and history.
  `frd show <id>` is more reliable than `frd list --unverified`.
- **`run revise-blocks` / `run restart-node`** re-enter specific blocks or nodes
  during a run without `--force`-restarting the whole run.

## Answering parks

Read the park's `type`, `previous_error` and `supported_actions`, then answer once
with a supported action and a `--rationale`:

| Lifecycle | Inspect | Answer |
|---|---|---|
| Module build | `build status --build-id ID` | `build resume --build-id ID --action A` |
| Frontend run | `state --json` / `interrupts --pending` | `resume --action A [--feedback ..]` |
| Backend | `backend state` | `backend resume --action A` |

`fix_rtl` / `fix_tb` mean *you already edited the file on disk*; the stage re-runs
on your edit without an LLM call. `target_unmeasured` means a tool or environment
problem, not a design failure: fix the tool, then `retry`. Don't approve blindly.
Decide each gate on its evidence, and read `.coresmith/contract_audit/*.json` when
integration or validation DV fails (it gives the first divergence and the
affected blocks).

## What "done" means

The deliverable is a **verified `chip_top`**: integration DV **and** validation DV
pass on the current composition (then backend signoff for GDS). None of these
mean the chip is done: `pipeline_done`, "N/N blocks passed", a passing SystemC
model, or a lint-clean top. Check `stage status`, and simulate the real
`chip_top` on representative inputs before you claim success.

## Pitfalls seen in real runs

- **Ports**: a block port is either a contract edge **or** a pin, never both
  (`PIN_DOUBLE_DRIVEN`). One pin maps to one whole port. To split a bundle,
  declare separate ports in the block diagram. `clk`/`rst_n` are
  `pin add --kind clock|reset` without `--from`.
- **`static` edges** (IRQs, straps) need `--field name:W`, otherwise they produce
  no ports. Locked edges change only with `--unlock --reason`.
- **Bounded FRD items** (PERF/TIME/area/power) pass only on a *measured* value. The
  bound cocotb test must call `orchestrator.harness.measure.record("PERF-001",
  value, unit=..., test=...)`, which appends to `$CORESMITH_MEASUREMENTS`.
  Area and power use `frd verifier --kind eda --entry area_um2|power_mw`.
- **Rendered views** (`arch/prd_spec.md`, `arch/frd_spec.md`, `.coresmith/*_spec.json`)
  come from the DB. Don't hand-edit them: change the row, then
  `state write` / `frd render`.
- **Never `rm -rf .coresmith/`**. Rename it to `.coresmith.<reason>-<ts>/`, because
  it holds the forensics. Start a new project root instead of resetting state.
- Daemon not up: read `.coresmith/daemon.log`. Timeline: `logs`,
  `.coresmith/pipeline_events.jsonl`. Worker calls: `.coresmith/llm_calls.jsonl`.

## Editing the engine

- Map: `bin/coresmith` + `orchestrator/harness/cli*.py` (CLI),
  `orchestrator/daemon/server.py` (daemon), `orchestrator/langgraph/` (graphs),
  `orchestrator/state_store/` (DB: stages, builds, targets),
  `orchestrator/module_build.py`, `orchestrator/langchain/{agents,prompts}/` (workers),
  `orchestrator/architecture/specialists/`, `docs/gitbook/` (the book).
- Fixes must be **generic**: no design, benchmark or exercise names in engine
  code or prompts. Design-specific harnesses live in the run dir.
- Gate any change in observable behavior behind a `CORESMITH_*` env var, and test
  both branches.
- Don't commit generated output (`.coresmith/`, `sim_build/`, `tb/`, per-run `rtl/`).
  Stage only the files you wrote. The tree often has unrelated in-flight edits.
- Don't kill `claude`/`codex` processes you didn't start. Parallel runs are common.
- Fast tests: `pytest orchestrator/tests/ -m "not live_llm and not requires_nix and not e2e"`.
