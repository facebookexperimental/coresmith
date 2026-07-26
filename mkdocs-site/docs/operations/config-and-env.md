# Configuration and environment variables

coresmith is configured via two layers: `orchestrator/config.yaml` (project-wide defaults) and environment variables (per-run overrides).

## `orchestrator/config.yaml`

Top-level keys and what they do:

### `project`

Project metadata used by prompts and validators.

- `name` — project name.
- `target_pdk` — `sky130`.
- `target_clock_mhz` — default 50.
- `rtl_language` — `verilog-2005`.
- `synthesis_tool` — `yosys`.
- `process_constraints` — list of Sky130 restrictions (no tri-state, no async resets, no latches) embedded into RTL prompts.

### `backend`

EDA tool configuration.

- `openroad_binary`, `magic_binary`, `netgen_binary`, `yosys_binary`, `klayout_binary` — wrapper scripts (often Nix-wrapped) used by `backend_helpers.run_*` functions.
- `pdk_variant` — `sky130A`.
- `std_cell_lib` — `sky130_fd_sc_hd`.
- `rcx_rules` — parasitic extraction rules.
- `target_utilization` — 45 %.
- `target_density` — 0.6.
- `pnr_timeout_s`, `drc_timeout_s`, `lvs_timeout_s` — per-tool wall-clock budgets.

### `tapeout`

OpenFrame shuttle dimensions and timeouts.

- `target` — `openframe`.
- `die_width_um` = 3520, `die_height_um` = 5188.
- `core_margin_um` = 100.
- `io_pads` = 44.
- `wrapper_pnr_timeout_s`, `wrapper_drc_timeout_s`, `precheck_timeout_s`.

### `temporal`

Optional Temporal workflow configuration. Used only when multi-machine orchestration is desired; single-machine runs ignore this section.

### `langchain`

LLM agent defaults.

- `model` — `opus-4.8`.
- `temperature` — `0.1`.
- `max_tokens` — `16000`.
- `agents` — per-agent prompts + tool exposure (kept here for backward compatibility; the actual prompts now live in `orchestrator/langchain/prompts/`).

### `telemetry`

- `enabled` — `true`.
- `trace_db` — path to OTel SQLite (default `.coresmith/traces.db`).

### `blocks`

Optional. If absent, blocks come from `blocks.yaml` (in the run dir) or `.coresmith/block_specs.json` (written by the architecture phase).

### `architecture`

- `max_block_attempts` — per-block retry limit.
- `pdk_config` — PDK YAML path.
- `benchmark_timeout_s`, `benchmark_cache_db` — for synthesis benchmarks.
- `state_dir` — `.coresmith`.
- `revision_triggers` — failure conditions that should re-run the architecture phase.

## Environment variables

### Project root & paths

| Var | Purpose |
|---|---|
| `CORESMITH_PROJECT_ROOT` | The run directory. Defaults to the repo root (don't!). |
| `CORESMITH_CONFIG_PATH` | `orchestrator/config.yaml` location. |
| `CORESMITH_BLOCKS_FILE` | Override `blocks.yaml` path. |
| `CORESMITH_LOG_DIR` | Step-log output dir; default `.coresmith/step_logs`. |
| `CORESMITH_SOURCE_ROOT` | Golden-model source directory scanned by the block diagram specialist. |
| `PDK_ROOT` | Propagated to Magic subprocess env. |

### LLM provider & model

| Var | Effect |
|---|---|
| `CORESMITH_LLM_PROVIDER` | `claude` (default) or `codex` |
| `CORESMITH_MODEL` | Override `DEFAULT_MODEL` (default `opus-4.8`) |
| `CORESMITH_BLOCK_MODEL` | Override `BLOCK_MODEL` (default `sonnet-4.6`) |
| `CORESMITH_CODEX_MODEL` | Override Codex default (`gpt-5.6`) |
| `CORESMITH_CODEX_SANDBOX` | Codex sandbox level (`danger-full-access` for unrestricted) |
| `CORESMITH_TIMEOUT_MULTIPLIER` | Scales every per-agent timeout. |

### Per-agent timeouts (seconds)

| Var | Default | Agent |
|---|---|---|
| `CORESMITH_UARCH_TIMEOUT` | 2700 | `UarchSpecGenerator` |
| `CORESMITH_RTL_TIMEOUT` | 1800 | `RTLGeneratorAgent` |
| `CORESMITH_TB_TIMEOUT` | 1800 | `TestbenchGeneratorAgent` |
| `CORESMITH_INTEGRATION_LEAD_TIMEOUT` | 2700 | `IntegrationLeadAgent` |
| `CORESMITH_INTEGRATION_REVIEW_TIMEOUT` | 2700 | `IntegrationReviewAgent` |
| `CORESMITH_INTEGRATION_TB_TIMEOUT` | 2700 | `IntegrationTestbenchGenerator` |
| `CORESMITH_VALIDATION_DV_TIMEOUT` | 2700 | `ValidationDVGenerator` |
| `CORESMITH_TIMING_CLOSURE_TIMEOUT` | 2700 | `TimingClosureAgent` |

### Architecture-stage toggles

By default the optional stages are **off**. Both ENABLE and legacy SKIP forms are recognized; SKIP wins.

| Var | Effect |
|---|---|
| `CORESMITH_ENABLE_MEMORY_MAP` | `1` runs the memory map specialist |
| `CORESMITH_ENABLE_CLOCK_TREE` | `1` runs the clock tree specialist |
| `CORESMITH_ENABLE_REGISTER_SPEC` | `1` runs the register spec specialist |
| `CORESMITH_SKIP_MEMORY_MAP` | Legacy override — forces off |
| `CORESMITH_SKIP_CLOCK_TREE` | Legacy override |
| `CORESMITH_SKIP_REGISTER_SPEC` | Legacy override |

### Pipeline behavior toggles

| Var | Effect |
|---|---|
| `CORESMITH_SKIP_SYNTH` | `1` skips Yosys + PDK preflight (frontend-only runs) |
| `CORESMITH_STRICT_INTEGRATION_REVIEW` | `1` auto-revises when `issues_fixed > 0` (old behavior) |
| `CORESMITH_ALLOW_SKIP_INTEGRATION_DV` | `1` adds `skip` to the integration DV failure interrupt |
| `CORESMITH_ALLOW_SKIP_VALIDATION_DV` | `1` adds `skip` to the validation DV failure interrupt |

### Backend tool overrides

| Var | Effect |
|---|---|
| `CORESMITH_BACKEND_OPENROAD` | OpenROAD binary path |
| `CORESMITH_BACKEND_MAGIC` | Magic binary path |
| `CORESMITH_BACKEND_NETGEN` | Netgen binary path |
| `CORESMITH_BACKEND_KLAYOUT` | KLayout binary path |

## Recommended setup

The CLAUDE.md "How to run" block is the canonical incantation:

```bash
RUN_DIR=/home/ubuntu/coresmith-runs/<task>-<flavor>-$(date +%Y%m%d-%H%M%S)
mkdir -p $RUN_DIR/inputs && cp <spec>.md $RUN_DIR/inputs/requirements.md

export CORESMITH_PROJECT_ROOT=$RUN_DIR
export CORESMITH_LLM_PROVIDER=codex             # or claude
export CORESMITH_CODEX_MODEL=gpt-5.6
export CORESMITH_MODEL=gpt-5.6
export CORESMITH_BLOCK_MODEL=gpt-5.6
export CORESMITH_CODEX_SANDBOX=danger-full-access
export CORESMITH_SKIP_SYNTH=1                   # unless Sky130 PDK is local
export CORESMITH_ENABLE_MEMORY_MAP=0
export CORESMITH_ENABLE_CLOCK_TREE=0
export CORESMITH_ENABLE_REGISTER_SPEC=0

bin/coresmith daemon start --project-root $RUN_DIR
bin/coresmith run start --project-root $RUN_DIR \
    --blocks-file /home/ubuntu/coresmith/examples/<design>/blocks.yaml
```

## Preflight check

Before any real run:

```bash
make preflight
```

Validates that the Sky130 PDK is present at the expected paths, Yosys ≥ 0.40 and Verilator ≥ 5.0 are on PATH, and the LLM provider CLI is callable. Outputs structured JSON listing exactly what is missing.

`CORESMITH_SKIP_SYNTH=1` skips the PDK and Yosys checks but still requires Verilator + cocotb.
