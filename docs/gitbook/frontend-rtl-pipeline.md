# Frontend RTL Pipeline

The frontend graph is the main Socmate implementation pipeline. It is defined
in `orchestrator/langgraph/pipeline_graph.py` and launched with
`build_pipeline_graph`.

## Block Loading

The block queue is resolved in this order:

1. An explicit `blocks_file`.
2. Architecture-generated `.coresmith/block_specs.json`.
3. The `blocks:` section in `orchestrator/config.yaml`.

Blocks are sorted by their `tier`. Blocks in the same tier run in parallel.
Tier `N + 1` does not start until every block in tier `N` reaches the tier
review point.

## Per-Block Lifecycle

Each block runs through a subgraph:

```text
init_block
  -> generate_uarch_spec
  -> review_uarch_spec
  -> generate_rtl
  -> generate_testbench
  -> synthesize
  -> block_done
```

Failure routes go through:

```text
diagnose -> decide -> retry_rtl | retry_tb | ask_human | block_done
```

### `init_block`

Initializes per-block state, creates a golden model import wrapper when
`python_source` is provided, and resets:

- `.coresmith/blocks/<block>/constraints.json`
- `.coresmith/blocks/<block>/diagnosis.json`
- `.coresmith/blocks/<block>/attempt_history.json`
- `.coresmith/blocks/<block>/previous_error.txt`

### `generate_uarch_spec`

Generates or revises `arch/uarch_specs/<block>.md`. The agent reads block
description, golden model code when available, architecture documents, and any
constraints on disk.

### `review_uarch_spec`

Per-block uArch review is auto-approved. Cross-block coherence is handled by a
tier-level integration review after the blocks complete.

### `generate_rtl`

Generates Verilog at the block's `rtl_target`, then runs Verilator lint. Lint is
part of this node. If lint fails, the pipeline runs up to two local LLM fix
attempts before escalating to diagnosis.

If a previous attempt passed simulation, the node can preserve that RTL and
force testbench regeneration instead of overwriting a known-good candidate.

### `generate_testbench`

Generates or reuses the cocotb testbench, then runs simulation. Simulation is
part of this node. When failures look like testbench framework errors, the node
attempts local testbench fixes before escalating. Assertion failures and
behavioral mismatches are sent to diagnosis because they may be RTL bugs.

### `synthesize`

Runs Yosys for the block at the requested target clock. Synthesis failures get
up to two local RTL fix attempts. If `CORESMITH_SKIP_SYNTH=1`, synthesis is
treated as a no-op success so RTL and simulation can run on hosts without the
Sky130 PDK.

### `diagnose` And `decide`

The diagnosis step classifies the failure and writes structured context to
`.coresmith/blocks/<block>/diagnosis.json`. Fast paths handle known
infrastructure and testbench failures without calling the heavier debug agent.

The deterministic decision rules then choose:

- `retry_tb` for testbench bugs
- `retry_rtl` for lint, simulation, synthesis, and most infrastructure failures
- `ask_human` for low confidence, repeated infrastructure failure, or explicit
  human-needed diagnosis
- `escalate` when the same category repeats or attempts are exhausted

### `ask_human`

This node interrupts the graph. The interrupt payload includes the block name,
phase, diagnosis, recent errors, supported actions, RTL/testbench/uArch paths,
step log paths, and outer-agent guidance.

Supported frontend actions are:

| Action | Meaning |
| --- | --- |
| `retry` | Re-enter the graph without claiming a disk fix |
| `fix_rtl` | Resume after the controller edited RTL or top-level RTL |
| `fix_tb` | Resume after the controller edited a generated testbench |
| `add_constraint` | Add a precise constraint and retry generation |
| `skip` | Mark the current failure as skipped |
| `abort` | Stop the graph |

## Tier Review

After a tier finishes, the integration review agent checks uArch specs for
cross-block interface coherence and emits one chip-level interrupt:

```text
integration_review -> approve | revise | abort
```

Approve advances to the next tier. Revise reruns the current tier from the
uArch stage. Abort stops the pipeline.

## Completion Gate

The pipeline does not proceed to integration unless every expected block passed
simulation and synthesis. If any block failed or never completed,
`pipeline_complete` interrupts with `pipeline_incomplete` and includes failed
block names, missing block names, and step log paths.

## Integration And Top-Level DV

After all blocks pass:

1. `integration_check` loads architecture connections, parses block ports,
   generates integrated top-level RTL under `rtl/integration/`, and lints it.
2. `integration_dv` generates and runs a top-level smoke/integration cocotb
   testbench.
3. `validation_dv` generates and runs ERS/KPI validation tests.

Top-level DV failures run a contract audit before interrupting. The audit helps
separate testbench bugs, top-level wiring bugs, block RTL bugs, and true
requirements-contract failures.

