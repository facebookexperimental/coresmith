# Architecture Phase

The architecture phase converts product intent into the documents and block
registry that the frontend pipeline consumes. It is implemented in
`orchestrator/langgraph/architecture_graph.py`.

## Inputs

The graph starts from:

- high-level requirements text
- `target_clock_mhz`
- PDK configuration, usually `orchestrator/pdk/configs/sky130.yaml`
- existing architecture state, if present under `.coresmith/`

When the architecture graph completes, frontend block definitions normally come
from `.coresmith/block_specs.json`. The frontend can also skip this phase and
load blocks from `orchestrator/config.yaml` or an explicit YAML file.

## Document Stack

| Artifact | Meaning | Typical file |
| --- | --- | --- |
| PRD | What functionality is required | `.coresmith/prd_spec.json`, `arch/prd_spec.md` |
| SAD | System architecture and rationale | `arch/sad_spec.md` |
| FRD | Functional and performance requirements | `arch/frd_spec.md` |
| Block diagram | Block decomposition and connections | `.coresmith/block_diagram.json`, docs under `arch/` |
| ERS | Engineering requirements passed to implementation and DV | `.coresmith/ers_spec.json`, `arch/ers_spec.md` |

## Graph Path

```text
Gather Requirements
  -> System Architecture
  -> Functional Requirements
  -> Block Diagram
  -> Memory Map
  -> Clock Tree
  -> Register Spec
  -> Constraint Check
  -> Finalize Architecture
  -> Create Documentation
  -> Final Review
  -> Architecture Complete
```

Some stages can loop:

- PRD sizing questions interrupt for answers before the PRD is finalized.
- Block diagram ambiguity can interrupt for clarification.
- Structural constraint violations interrupt instead of being silently fixed.
- Constraint iterations can return to the block diagram.
- Final review is the OK2DEV gate. Revisions loop back before frontend starts.

## Output Contract For Frontend

The frontend pipeline expects each block spec to include at least:

```yaml
blocks:
  block_name:
    tier: 1
    rtl_target: "rtl/block_name/block_name.v"
    testbench: "tb/cocotb/test_block_name.py"
    description: |
      Behavioral and interface contract for this block.
```

Optional block fields include `python_source` for a golden model and any
additional fields used by prompts or local project conventions.

