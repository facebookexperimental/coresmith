# State, Artifacts, And Observability

Coresmith is designed so a controller can stop, inspect, edit, and resume without
losing context.

## Checkpoints

Each graph has a SQLite checkpoint:

| Graph | Checkpoint |
| --- | --- |
| Architecture | `.coresmith/architecture_checkpoint.db` |
| Frontend pipeline | `.coresmith/pipeline_checkpoint.db` |
| Backend | `.coresmith/backend_checkpoint.db` |
| Tapeout | `.coresmith/tapeout_checkpoint.db` |

The shared `GraphLifecycle` wrapper owns the graph instance, checkpointer,
background task, thread id, status, and last error message. It also recovers
parked interrupts after process restarts and closes orphaned graph events.

## Event Log

Every graph transition writes line-delimited JSON to:

```text
.coresmith/pipeline_events.jsonl
```

Events include graph/node names, enter/exit markers, block names, attempts,
success flags, log paths, and failure summaries. This file is the fastest way
to follow a live run:

```bash
tail -F .coresmith/pipeline_events.jsonl | jq -c .
```

## Traces

OpenTelemetry spans are written to:

```text
.coresmith/traces.db
```

Use:

```bash
make traces
```

or query directly with SQLite to find slow nodes, repeated retries, and failure
diagnostics.

## Per-Block State

Each block gets transient state under:

```text
.coresmith/blocks/<block>/
```

Key files:

| File | Purpose |
| --- | --- |
| `constraints.json` | Constraints accumulated from diagnosis or human/outer-agent input |
| `diagnosis.json` | Latest structured diagnosis |
| `attempt_history.json` | Recent failure categories and attempts |
| `previous_error.txt` | Tail of the latest tool or generation error |
| `best_result.json` | Known-good simulation result used to avoid regressing RTL |

## Step Logs

EDA subprocess logs are written under:

```text
.coresmith/step_logs/<block>/<step>_attempt<N>.log
```

Interrupt payloads include `step_log_paths`; use those paths before deciding to
retry. The payload is a pointer, not the full evidence.

## Generated Design Artifacts

Common outputs:

| Path | Contents |
| --- | --- |
| `arch/uarch_specs/<block>.md` | Per-block microarchitecture spec |
| `rtl/<block>/<block>.v` | Generated block RTL |
| `tb/cocotb/test_<block>.py` | Generated block testbench |
| `tb/cocotb/<block>_model.py` | Golden model import wrapper, when applicable |
| `sim_build/<block>/` | Verilator/cocotb simulation output |
| `syn/output/<block>/` | Block synthesis output |
| `rtl/integration/` | Integrated top-level RTL |
| `.coresmith/contract_audit/` | Top-level DV failure audit context and result |

## Operational Rule

Do not treat a retry action as a fix. On an interrupt, read the referenced
artifacts, determine the root cause, edit the right file or add a precise
constraint, then resume with the action that describes what changed.

