# Contracts between agents

coresmith's agents are deliberately *not* talking to each other in real time — there's no message bus, no shared context, no AutoGen-style dialogue. Instead, the rules between agents are enforced by **the graph topology**, **the disk artifacts they share**, and **a small number of audit/review agents** that check downstream consumers' assumptions against upstream producers' outputs.

This page maps out where each contract lives.

## The disk is the message bus

Agents communicate by writing files. The producer writes; the consumer reads. The graph node sequence guarantees the producer ran first.

| Producer agent | File it writes | Consumer agent(s) |
|---|---|---|
| `gather_requirements_node` / PRD | `arch/prd_spec.md`, `.coresmith/prd_spec.json` | every downstream architecture node, integration testbench generator |
| `system_architecture_node` / SAD | `arch/sad_spec.md` | block diagram, ERS doc |
| `functional_requirements_node` / FRD | `arch/frd_spec.md` | uArch spec generator, ERS doc |
| `block_diagram_node` | `arch/block_diagram.md`, `.coresmith/block_diagram.json` | uArch spec gen, integration lead, integration review, integration TB, validation DV |
| `create_documentation_node` / ERS | `arch/ers_spec.md`, `.coresmith/ers_spec.json` | RTL generator, TB generator, uArch spec gen, validation DV |
| `UarchSpecGenerator` | `arch/uarch_specs/<block>.md` | RTL generator, TB generator, debug agent, integration review |
| `RTLGeneratorAgent` | `rtl/<block>/<block>.v` | TB generator, integration check, lint, synth, contract audit |
| `TestbenchGeneratorAgent` | `tb/cocotb/test_<block>.py` | sim, debug agent, contract audit |
| `IntegrationLeadAgent` | top-level `chip_top.v` | integration DV, validation DV, backend graph |
| `DebugAgent` | `.coresmith/blocks/<block>/diagnosis.json`, `.coresmith/blocks/<block>/constraints.json` | next round of RTL/TB generator |
| `ContractAuditAgent` | `.coresmith/contract_audit/<stage>_contract_audit.json` | outer agent / human; included in interrupt payload |
| Backend tools (via `BackendEDAAgent`) | `syn/output/`, `routed.def`, GDS, SPICE | next backend stage, tapeout |

A consumer agent's prompt always tells it *where* to read. The graph doesn't pass blobs; it passes paths.

## Two enforcement agents

Two agents exist *specifically* to enforce contracts between other agents:

### IntegrationReviewAgent — enforces inter-block interfaces

After each tier finishes generating uArch specs and RTL, the graph runs `IntegrationReviewAgent.review(...)`. Its job is to cross-reference:

- Architecture connections (`block_diagram.json`)
- Every uArch spec in the current tier (`arch/uarch_specs/<name>.md` — specifically Section 9 with the Verilog port stubs)
- PRD-defined bus protocol & data width

It looks for **width mismatches, direction mismatches, protocol mismatches, reset polarity mismatches**, and **edits the uArch specs on disk to fix them**. It returns `{summary, issues_found, issues_fixed}` and parks the graph on `uarch_integration_review` for the outer agent's approval.

Because the agent always finds something to clean up, `issues_fixed > 0` is the *expected* steady state. The default outer-agent decision is `approve`. The env var `CORESMITH_STRICT_INTEGRATION_REVIEW=1` restores the old behavior where `issues_fixed > 0` forces a `revise` and re-runs the tier's RTL.

### ContractAuditAgent — enforces top-level DV contracts

When integration DV or validation DV fails, `ContractAuditAgent` runs and emits a structured diagnosis JSON. The most important fields:

- `category` — `UARCH_INTERFACE_CONTRACT_ERROR`, `UARCH_SPEC_ERROR`, `ARCHITECTURE_ERROR` (architectural), `RTL_BUG`, `TESTBENCH_BUG`, `TIMING`, `CONTRACT_FAILURE` (local).
- `contract_failure` — `True` whenever the category indicates an interface or architecture issue.
- `first_divergence` — concrete: signal names from VCD, golden vs RTL observation, log refs.
- `affected_blocks` — which blocks need changes.
- `recommended_action` — `fix_rtl`, `fix_tb`, `revise_uarch`, `ask_human`.
- `local_fix_possible` — whether a small edit will resolve it.

Normalization in `contract_audit_agent.py:137-167` forces `recommended_action="revise_uarch"` whenever `category` indicates a contract / spec / architecture issue. This routes the outer agent toward re-running the architecture phase (or restarting blocks from `generate_uarch_spec`) rather than patching RTL that was implementing a broken spec.

## Implicit contracts enforced by node sequencing

| Contract | Enforced where |
|---|---|
| PRD must exist before SAD/FRD/block diagram | Architecture graph edges (`Gather Requirements` → ...). |
| Block diagram must exist before specialist stages | Direct edges from `Block Diagram` to `Memory Map`/`Clock Tree`/`Register Spec`. |
| Constraints must pass before finalize | `route_after_constraints` (`architecture_graph.py:2040`) routes only `PASS` or `ACCEPT` to `Finalize`. |
| OK2DEV must be granted before frontend pipeline starts | The architecture graph and pipeline graph are separate. The pipeline reads `.coresmith/block_specs.json`, which is only written by `finalize_node`. |
| All blocks must pass before chip_top.v generation | `pipeline_complete_node:2049` gates on `all(b["success"] for b in completed_blocks)`. |
| `chip_top.v` must lint clean before integration DV | `route_after_integration` (`pipeline_graph.py:2524`) routes only `SUCCESS` to `integration_dv`. |
| Integration DV must pass before validation DV | `route_after_integration_dv:3063` routes only `PASSED` to `validation_dv`. |
| Backend has a passing flat netlist before PnR | `route_after_flat_synth:496` only proceeds on `SUCCESS`. |
| Tapeout MPW precheck must pass before submission | `route_after_precheck` routes only `PASS` to `tapeout_complete`. |

## Constraint propagation via `.coresmith/blocks/<block>/constraints.json`

When `DebugAgent` identifies a recurring failure pattern, it appends a structured rule to the block's `constraints.json`:

```json
{
  "constraints": [
    {
      "rule": "always synchronize ready to enable downstream backpressure",
      "rationale": "Sim failure at attempt 2: ready was registered, causing 1-cycle drop"
    }
  ]
}
```

The RTL and TB generator prompts read this file at the top of every run. This is how a failure on attempt 2 prevents a re-occurrence on attempt 3, even though the LLM call is fresh.

The outer agent can also write constraints via the `add_constraint` resume action on a `human_intervention_needed` interrupt.

## Reset / interface conventions

The shared conventions every agent assumes:

- Verilog-2005, synchronous reset, no async resets.
- AXI-Stream interfaces use the FSM pattern from `prompts/rtl_generator.md` (valid self-cancellation is explicitly forbidden — a known LLM failure mode).
- Sky130 process rules: no tri-state, no latches, no clock gating.
- Reset polarity must match what the uArch spec declares (integration review enforces).
- Data widths must match between source and dest, or an adapter block must exist (constraint check enforces).
- Clock domain crossings must use a CDC module (constraint check enforces).
