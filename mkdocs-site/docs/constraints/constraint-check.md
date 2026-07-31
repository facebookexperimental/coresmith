# Constraint checking

The constraint checker is the bridge between the architecture phase and the pipeline phase. It runs after all the per-specialist nodes (block diagram, memory map, clock tree, register spec) have produced their artifacts, and it decides whether the design is internally consistent enough to hand off to the frontend pipeline.

Source: [`orchestrator/architecture/constraints.py`](https://github.com/facebookexperimental/coresmith/blob/main/orchestrator/architecture/constraints.py) (~1190 lines).

## Where it runs in the graph

```mermaid
flowchart LR
    BD[Block Diagram] --> MM[Memory Map]
    MM --> CT[Clock Tree]
    CT --> RS[Register Spec]
    RS --> ERS[Engineering Requirements]
    ERS --> CC[Constraint Check]
    CC -->|PASS - all_pass=True| FIN[Finalize Architecture]
    CC -->|STRUCTURAL violations| ECS[Escalate Constraints ⏸]
    CC -->|DOC_FIX violations| DF[Doc Fix]
    DF --> ERS
    CC -->|AUTO_FIX violations| CI[Increment Round]
    CI -->|round ≤ max_rounds| BD
    CI -->|exhausted| EEX[Escalate Exhausted ⏸]
```

!!! note "The ERS is emitted *before* the gate"
    `arch/ers_spec.md` used to be written by `create_documentation_node`,
    which runs **after** `Finalize Architecture` — i.e. after the only
    automated gate in the architecture phase. Any ERS-vs-anything check was
    therefore structurally impossible: at Constraint Check time
    `state["ers_spec"]` was still the seeded `None` and the file did not exist
    on disk. The **Engineering Requirements** node now emits it one step
    earlier, and `create_documentation_node` *reuses* that result rather than
    regenerating (so the ERS that ships is the one the gate approved). Doc Fix
    routes through the node too, so a repaired FRD refreshes the ERS before
    constraints are re-checked. `CORESMITH_ERS_BEFORE_CONSTRAINTS=0` restores
    the old ordering.

Entry point: `constraint_check_node:1137` in `architecture_graph.py`, which calls `check_constraints(...)` in `constraints.py`.

## What it checks

Twelve constraint categories. Some are deterministic Python checks; the rest are flagged by an LLM that reads the same artifacts plus rule text in the prompt.

### Deterministic checks

| # | Check | What it verifies |
|---|---|---|
| 10 | **GPIO pad budget** (shuttle-only) | `_count_block_io_pads()` counts pads claimed across blocks. Compared against `_get_shuttle_limits().usable_io_pads`. Structural/error if overflow; warning at >90% utilization. (`constraints.py:124-157`) |
| 11 | **Shuttle area fit** (shuttle-only) | Estimated die area ≤ shuttle user area. Assumes ~200K gates/mm² at 60% utilization. Structural/error if exceeded. (`constraints.py:159-211`) |
| 12 | **Derived geometry** | Cross-checks block/source dimensions, columns/rows, coordinate ranges, transaction counts, field widths. Catches codec-style transposition bugs (e.g. width-derived columns and height-derived rows swapped). (`constraints.py:566-687`) |

### LLM-driven checks

The LLM gets the rule text embedded in the prompt and produces structured violations.

| # | Check | What it verifies |
|---|---|---|
| 1 | **Peripheral count** | Nibble-decode bus allows ~8–15 peripherals max. |
| 2 | **Memory overlap** | No two address regions overlap. |
| 3 | **SRAM budget** | Total SRAM usage ≤ ERS `sram_budget_kb`. |
| 4 | **Clock domain crossings** | If multiple clock domains, every cross-domain connection needs an explicit CDC module. |
| 5 | **Gate budget** | Total estimated gates ≤ ERS `max_gate_count` (default 2M). |
| 7 | **Register spec ↔ memory map** | Register spec blocks must match memory map peripherals. |
| 8 | **Per-block CDC membership** | Every block belongs to exactly one clock domain. |
| 9 | **Data width consistency** | Source width = dest width on every connection, or an explicit adapter block exists. |

### Hybrid

| # | Check | Note |
|---|---|---|
| (deterministic) | **Performance / cycle budget** | Cycle-count claims vs clock × FPS × dimensions. |
| 6 | **Tier assignment reasonableness** | LLM judgment call — e.g. an FFT block shouldn't be tier 1. |

### Cross-artifact consistency

The architecture phase emits several artifacts describing **one** design:
`block_diagram.json` (topology + per-edge `semantic_contract`),
`interface_contracts.json` (frozen bit layout, `rate_description`,
`representations.state_semantics[*].rule`), the FRD, and the ERS. Every other
entry in the catalog is *intra*-artifact (block diagram vs itself, memory map
vs itself) or artifact-vs-uArch-spec — so when one artifact was re-issued and
the others were not, nothing at architecture time noticed. The disagreements
surfaced later, one stale statement at a time, in a different downstream gate,
each costing a full re-spec.

`cross_artifact_consistency` closes that. Like
`inter_block_payload_protocol_coherence`, it has two halves sharing one
`check` id:

| Half | Where | What it catches |
|---|---|---|
| **Deterministic** | `orchestrator/architecture/cross_artifact.py` | A *named* numeric quantity two artifacts state differently — clock rate, line rate, phase width, capacity. Normalizes units before comparing (`9 KiB` == `73728 bits`). |
| **LLM subagent** | catalog entry `cross_artifact_consistency` | Semantic / scheduling contradictions: "rise-scheduled here, fall-scheduled there", "always accepted here, backpressured there". Reported as **candidates**, with both cited locations split out for a human. |

The deterministic half **never guesses**. It judges a value only when the
document writes the quantity's name immediately next to the number
(`tHIGH>=80 ns`, `SCK <=6.25 MHz`, `12.5 MHz QSPI clock`). Anything else — an
approximate value (`~30 ns`), a range, a list member, an ambiguous unit
(`KB`: 1000 or 1024?), or a value with no adjacent name — is skipped and
logged as a note. Binding numbers to identifiers merely *nearby* was tried
first and measured against 58 real architecture runs: it flagged 48 of them,
because one clause routinely mentions the interface clock, the core clock and
a latency together. Adjacency-only naming flags 5 findings across the same 58
runs.

The catalog entry has to claim ownership of the class **explicitly**, because
the shared subagent system prompt says *"Do not invent constraints. If you see
something else wrong, ignore it; another subagent owns it."* — which would
otherwise suppress exactly this finding.

Both halves emit `category="structural"`, so a contradiction routes to
`Escalate Constraints` and blocks `Finalize Architecture` until the operator
resolves it or explicitly `accept`s a documented supersession. Nothing
auto-edits an artifact. `CORESMITH_CROSS_ARTIFACT_GATE=0` disables the gate.

## How a check actually runs

`check_constraints(...)` (`constraints.py:831`):

1. Run the deterministic checks first; collect their violations.
2. Build the LLM user message: concatenate the requirements, block diagram JSON, memory map, clock tree, register spec, benchmark data, and any project markdown documents (`arch/prd_spec.md`, `arch/sad_spec.md`, etc.).
3. Build the LLM system prompt from `prompts/constraint_check.md`, with template variables `{bus_rules}` (lines 949-965) and `{additional_rules}` (lines 977-1004).
4. Call `ClaudeLLM(model=DEFAULT_MODEL, timeout=600).call(system=..., prompt=...)`.
5. Parse the LLM response (`_parse_response`) as structured JSON.
6. Merge LLM violations with deterministic ones — drop LLM violations that duplicate a `check` already produced deterministically.
7. Return a `constraint_result` dict.

## The violation shape

```python
{
    "violation": str,                        # human-readable description
    "category": "structural" | "auto_fixable",
    "check": str,                            # short rule key (e.g. "gpio_pad_budget")
    "severity": "error" | "warning",
}
```

The `category` is what drives routing:

- `structural` — the architecture is incompatible with itself; an LLM revision is unlikely to fix it without architect input. Routes to `Escalate Constraints`.
- `auto_fixable` — the LLM can probably fix this if it sees the violation as feedback on the next round.

The `severity` is informational; warnings don't block the gate.

## The output dict

```python
constraint_result = {
    "violations": list[Violation],
    "all_pass": bool,           # True iff violations is empty
    "has_structural": bool,     # True if any v.category == "structural"
    "category": list[str],      # all categories seen (e.g. ["structural", "auto_fixable"])
}
```

## Routing on the result

`route_after_constraints:2040`:

| Verdict | Next node | Outer-agent involvement |
|---|---|---|
| `PASS` (`all_pass`) | `Finalize Architecture` | none |
| `STRUCTURAL` (`has_structural`) | `Escalate Constraints` interrupt | architect picks `retry`, `accept`, `feedback`, or `abort` |
| `AUTO_FIX` | `Constraint Iteration` | none — graph re-enters `Block Diagram` with violations as feedback |

The auto-fix loop is bounded: `route_after_increment:2142` enforces both a per-round budget (`round <= max_rounds`) and a lifetime budget (`total_rounds <= max_rounds * 3`). Hitting either triggers `Escalate Exhausted`, which always involves the architect.

## Data sources

The checker cross-references everything written by upstream specialists:

| Source | Field consulted |
|---|---|
| `block_diagram` dict | blocks, interfaces, connections, estimated_gates |
| `memory_map` dict | peripherals, address allocations |
| `clock_tree` dict | domains, crossings |
| `register_spec` dict | register_blocks |
| `ers_spec` dict | area_budget (max_die_area_mm2, max_gate_count), dataflow (bus_protocol, sram_budget_kb), speed_and_feeds (target_clock_mhz), technology |
| Project filesystem | `arch/prd_spec.md`, `sad_spec.md`, `frd_spec.md`, `block_diagram.md`, `ers_spec.md`, `arch/uarch_specs/*.md` |

The deterministic checks pull constants from `_get_shuttle_limits()` (lines 34–60) and `_extract_sram_budget_kb()` (lines 1138–1176).

## Graceful failure

If the LLM call itself fails (timeout, parse error, circuit breaker), `check_constraints` returns the deterministic violations plus a warning-level violation recording the LLM error (`constraints.py:1097-1109`). The pipeline still gets a usable result rather than crashing the architecture graph.

## Constraints are not architect questions

A *violation* is not the same as a *question for the architect*. The constraint check just classifies what's broken. Whether that needs human input depends on:

- `category` — `structural` violations always escalate.
- `round` — even `auto_fixable` violations eventually escalate when the auto-fix budget is exhausted.

The actual escalation mechanism is documented in [Architect escalation](architect-escalation.md).
