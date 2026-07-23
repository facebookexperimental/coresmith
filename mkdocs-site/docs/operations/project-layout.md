# Project layout

A coresmith run is rooted at an isolated **project root**. The daemon, the graphs, the agents, and the outer agent all operate against the same root. CLAUDE.md says it loudly: **always use an isolated run dir** — never start the daemon against the repo root.

## Convention

```
/home/ubuntu/coresmith-runs/<task>-<flavor>-<YYYYMMDD-HHMMSS>/
├── inputs/
│   └── requirements.md                # the spec the architecture phase reads
├── blocks.yaml                        # optional override of orchestrator/config.yaml blocks
├── .coresmith/                        # daemon-managed state
│   ├── daemon.json                    # {port, pid} for CLI discovery
│   ├── daemon.log                     # daemon stdout/stderr
│   ├── architecture_checkpoint.db     # LangGraph SQLite (architecture)
│   ├── pipeline_checkpoint.db         # LangGraph SQLite (frontend pipeline)
│   ├── backend_checkpoint.db          # LangGraph SQLite (backend)
│   ├── tapeout_checkpoint.db          # LangGraph SQLite (tapeout)
│   ├── pipeline_events.jsonl          # graph node enter/exit timeline
│   ├── llm_calls.jsonl                # every LLM call with prompts, tokens, cost
│   ├── traces.db                      # OpenTelemetry spans
│   ├── prd_spec.json                  # JSON copies of architecture outputs
│   ├── block_diagram.json
│   ├── memory_map.json
│   ├── clock_tree.json
│   ├── register_spec.json
│   ├── ers_spec.json
│   ├── block_specs.json               # frontend pipeline consumes this
│   ├── block_diagram_viz.json         # ReactFlow visualization
│   ├── blocks/<block>/
│   │   ├── constraints.json           # accumulated rules from DebugAgent
│   │   ├── diagnosis.json             # latest debug agent output
│   │   ├── previous_error.txt         # last failure log
│   │   ├── attempt_history.json       # all retry attempts
│   │   └── best_result.json           # cached "best" sim result (for regression guard)
│   ├── step_logs/<block>/
│   │   └── <phase>_attempt_N.log      # raw output of each lint/sim/synth attempt
│   └── contract_audit/
│       ├── integration_dv_contract_audit.json
│       └── validation_dv_contract_audit.json
├── arch/
│   ├── prd_spec.md
│   ├── sad_spec.md
│   ├── frd_spec.md
│   ├── block_diagram.md
│   ├── memory_map.md
│   ├── clock_tree.md
│   ├── register_spec.md
│   ├── ers_spec.md
│   ├── DV_RULES.md                    # optional, project-wide DV conventions
│   └── uarch_specs/
│       └── <block>.md                 # per-block microarchitecture
├── rtl/
│   ├── <block>/
│   │   └── <block>.v
│   └── integration/
│       └── <design>_top.v             # chip_top.v after IntegrationLeadAgent
├── tb/
│   ├── cocotb/
│   │   └── test_<block>.py
│   └── integration/
│       ├── test_<design>.py           # integration DV
│       └── test_<design>_validation.py # validation DV
├── sim_build/<block>/
│   ├── dump.vcd
│   └── wavekit_audit.json
├── syn/output/<design>/
│   ├── <design>_netlist.v             # post-Yosys flat netlist
│   └── <design>.sdc                   # synthesis-side constraints
├── chip_finish/                       # dashboard + 3D viewer
│   ├── 3d.html
│   ├── <block>_layout.png
│   └── dashboard.html
├── openframe_submission/              # tapeout output
│   ├── openframe_project_wrapper.v
│   ├── user_defines.v
│   └── …
└── .pdk/                              # optional local PDK copy (Sky130A/B)
```

## What's in `.gitignore`

The coresmith repo's `.gitignore` excludes:

- `.coresmith/`
- `sim_build/`
- `tb/cocotb/`, `tb/integration/`, `tb/rd/` (generated testbenches)
- `rtl/` *for per-run designs* — the repo only commits canonical reference RTL for the `examples/` designs

So when you run against an isolated project root, none of the generated artifacts should be in git, but the run dir is fully self-contained.

## Backing up vs cleaning up

**Don't `rm -rf .coresmith/`.** The repo root is intentionally littered with `.coresmith.cleared-<reason>-<ts>/` and `.coresmith.failed-<reason>-<ts>/` archives because they're valuable for forensics — diagnosis JSON, contract audits, step logs, and the LLM call log can explain *why* a run failed days after the fact.

Convention: when starting fresh, *rename* the directory:

```bash
mv .coresmith .coresmith.cleared-video_codec-broken-$(date +%Y%m%d-%H%M%S)
```

## What lives where

| Question | Where to look |
|---|---|
| What did the architecture phase produce? | `arch/*.md` (human-readable), `.coresmith/*.json` (structured) |
| Why did block X fail? | `.coresmith/blocks/X/diagnosis.json`, then `step_logs/X/*` |
| What did the LLM see and produce? | `.coresmith/llm_calls.jsonl` |
| What was the graph doing when it parked? | `coresmith state` (live), then `.coresmith/pipeline_events.jsonl` |
| Why did integration DV fail? | `.coresmith/contract_audit/integration_dv_contract_audit.json` |
| What was the final design? | `arch/ers_spec.md`, `rtl/integration/<design>_top.v`, `chip_finish/dashboard.html` |
| Are the EDA tools happy? | `syn/output/<design>/<design>_netlist.v` (frontend), backend graph artifacts under `runs/` (created by backend nodes) |

## Reproducibility

A finished run dir is reproducible: another developer with the same PDK, same EDA toolchain versions, and the same `inputs/requirements.md` should be able to point a fresh daemon at a *fresh* run dir and get an equivalent design (modulo LLM nondeterminism).

For exact replays, use the `llm_calls.jsonl` to inspect what was actually said — though replay is not officially supported, the prompts and responses are all there.
