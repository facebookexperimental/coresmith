# Prompt registry

All LLM behavior in coresmith is controlled by the prompt markdown files in [`orchestrator/langchain/prompts/`](https://github.com/facebookexperimental/coresmith/tree/main/orchestrator/langchain/prompts). Editing one of these files changes the corresponding agent's behavior across every run.

## Architecture phase

| File | Agent / specialist | Purpose |
|---|---|---|
| `prd_spec.md` | `specialists/prd_spec.py` | Senior SoC systems engineer drafting PRD sizing questions then synthesizing the final PRD. |
| `sad_spec.md` | `specialists/sad_spec.py` | System architecture document — high-level design decisions and rationale. |
| `frd_spec.md` | `specialists/frd_spec.py` | Functional requirements with quantitative acceptance criteria. |
| `ers_doc.md` | `specialists/ers_doc.py` | Engineering requirements specification synthesizing PRD/SAD/FRD/diagram into per-block requirements. |
| `block_diagram.md` | `specialists/block_diagram.py` | Block-level architect — designs blocks, tiers, interfaces, semantic contracts, connections. |
| `memory_map.md` | `specialists/memory_map.py` | Peripheral / SRAM / CSR memory layout. |
| `clock_tree.md` | `specialists/clock_tree.py` | Clock domains, crossings, reset spec. |
| `register_spec.md` | `specialists/register_spec.py` | Per-block CSR definitions. |
| `constraint_check.md` | `architecture/constraints.py` | Cross-cutting constraint reviewer (bus rules, contracts, bandwidth, tensor shapes). |

## Frontend pipeline

| File | Agent | Purpose |
|---|---|---|
| `uarch_spec_generator.md` | `UarchSpecGenerator` | VLSI micro-architect writing implementation specs from golden models. |
| `rtl_generator.md` | `RTLGeneratorAgent` | Verilog-2005 RTL author with Sky130 process rules and AXI-Stream FSM patterns. |
| `testbench_generator.md` | `TestbenchGeneratorAgent` | cocotb test author with golden model binding. |
| `integration_review.md` | `IntegrationReviewAgent` | Chip integration auditor — edits uArch specs to fix interface mismatches. |
| `integration_lead.md` | `IntegrationLeadAgent` | Top-level chip_top.v author. |
| `integration_testbench.md` | `IntegrationTestbenchGenerator` | System-level smoke/integration TB. |
| `validation_dv.md` | `ValidationDVGenerator` | KPI / ERS-driven validation TB with `REQUIREMENT_COVERAGE`. |
| `debug_agent.md` | `DebugAgent` | Failure diagnosis with structured JSON output. |
| `contract_audit.md` | `ContractAuditAgent` | DV-failure root-cause analysis (first divergence, affected blocks, recommended action). |
| `lint_fixer.md` | (RTL local loop) | Yosys lint error fixer. |
| `synth_fixer.md` | (synth local loop) | Synthesis error fixer. |
| `timing_closure.md` | `TimingClosureAgent` | Pipeline / restructure / constraint strategies. |
| `decide.md` | (outer agent / autochecker) | Decision framework — what action to pick when resuming an interrupt. |

## Backend & tapeout

| File | Step | Purpose |
|---|---|---|
| `backend_synthesis.md` / `backend_synth_llm.md` | Yosys synth | Adapt + interpret Yosys flow. |
| `backend_pnr.md` / `backend_pnr_llm.md` | OpenROAD PnR | Adapt + interpret PnR TCL. |
| `backend_drc.md` / `backend_drc_llm.md` | Magic DRC | Adapt + interpret DRC report. |
| `backend_lvs.md` / `backend_lvs_llm.md` | Netgen LVS | Adapt + interpret LVS output. |
| `backend_timing_signoff.md` | Post-route STA | Interpret slack / WNS / sign-off. Supports `CONDITIONAL_PASS`. |
| `backend_wrapper_llm.md` | OpenFrame wrapper | Generate wrapper RTL + submission tree. |
| `backend_mpw_precheck.md` | MPW precheck | Interpret precheck output. |
| `sdc_generator.md` | SDC files | Synopsys Design Constraints. |
| `tapeout_wrapper_synth.md` | Wrapper synth | Yosys synth on the wrapper. |
| `tapeout_wrapper_pnr.md` | Wrapper PnR | OpenROAD on the fixed OpenFrame die. |
| `tapeout_wrapper_drc.md` | Wrapper DRC | Magic DRC at wrapper level. |
| `tapeout_wrapper_lvs.md` | Wrapper LVS | Netgen LVS — mismatch tolerated. |
| `tapeout_diagnosis.md` | Failure triage | Decide `auto_retry`, `continue`, or `escalate`. |
| `tapeout_complete.md` | Sign-off | Validate DRC/LVS/precheck and write PRD-compliance assessment. |

## Dashboards & viewers

| File | Purpose |
|---|---|
| `block_diagram.md` (used twice) | Visualization annotation. |
| `dashboard_doc.md` / `dashboard.html.j2` | Status dashboard generation. |
| `chip_finish_dashboard.md` / `chip_finish_template.html` | Final chip status report. |
| `3d_viewer.html.j2` | Three.js GDS viewer template. |

## Editing prompts

These files are part of the canonical pipeline behavior. Conventions when editing:

1. **Test both branches** of any conditional behavior. The `test_pipeline_graph.py::TestRouteAfterIntegrationReview` test class is the template for env-var-gated behavior.
2. **Don't bake design-specific examples into a generic prompt.** Codec-specific guidance belongs in `the evaluation harness/a downstream design-collateral tree`, not in the prompt for *all* RTL generation.
3. **Token cost matters.** The orchestrator pays the cache write per turn. Prompts under 5K tokens get reliably cached; bigger ones thrash.
