# Pipeline Overview

Socmate is implemented as a set of checkpointed LangGraph state machines. The
graphs share a project root and write their durable state under `.coresmith/`.
The same graphs can be driven by the CLI runner, `coresmithd`, or the MCP server.

## Graphs

| Graph | Builder | Purpose | Checkpoint |
| --- | --- | --- | --- |
| Architecture | `build_architecture_graph` | Requirements to PRD/SAD/FRD/ERS, block diagram, and block specs | `.coresmith/architecture_checkpoint.db` |
| Frontend pipeline | `build_pipeline_graph` | uArch, RTL, lint, testbench, simulation, synthesis, integration, DV | `.coresmith/pipeline_checkpoint.db` |
| Backend | `build_backend_graph` | Flat top synthesis, PnR, DRC, LVS, timing, wrapper, precheck | `.coresmith/backend_checkpoint.db` |
| Tapeout | `build_tapeout_graph` | OpenFrame wrapper PnR/DRC/LVS and MPW precheck | `.coresmith/tapeout_checkpoint.db` |

## End-To-End Flow

```text
requirements or block registry
  |
  v
Architecture graph
  PRD -> SAD -> FRD -> block diagram -> optional memory map/clock/registers
  -> constraint check -> ERS -> OK2DEV
  |
  v
Frontend pipeline graph
  tiered blocks in parallel:
    uArch spec -> RTL + lint -> testbench + sim -> synthesis
  tier integration review
  all-block gate
  integration top RTL -> integration DV -> validation DV
  |
  v
Backend graph
  flat synthesis -> PnR -> DRC -> LVS -> timing -> wrapper -> precheck
  |
  v
Tapeout graph
  OpenFrame wrapper -> wrapper signoff -> submission directory
```

## Disk-First Execution

The pipeline intentionally keeps large artifacts out of graph state:

- Specs are written under `arch/`.
- RTL is written under `rtl/`.
- Testbenches are written under `tb/cocotb/`.
- Tool logs are written under `.coresmith/step_logs/`.
- Block-local diagnosis state is written under `.coresmith/blocks/<block>/`.
- Graph progress is checkpointed in SQLite.

This matters operationally: when a graph interrupts, the outer controller should
read the referenced files, make edits on disk when needed, and resume the graph
with the action that matches the edit.

## Controllers

The controller is transport, not pipeline logic:

- `run_pipeline.py` is a legacy headless frontend runner. It auto-approves some
  interrupts and retries until limits are hit.
- `coresmithd` is an HTTP daemon for one project root. It keeps the process
  alive and exposes run start/state/resume/pause endpoints.
- MCP exposes architecture, frontend, backend, tapeout, inspection, and restart
  tools for Claude Code or another MCP client.

Use `coresmithd` when an external service or script should own the control
loop. Use MCP when an interactive agent should inspect files, diagnose issues,
and resume the graphs directly.

