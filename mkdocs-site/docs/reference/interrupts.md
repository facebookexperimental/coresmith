# Interrupts catalog

Every interrupt in every coresmith graph, with its trigger, payload, supported actions, and resume behavior. Use this page when you (or your outer agent) sees `pending_interrupt_count > 0` and needs to decide what to send.

## Architecture graph

### `prd_questions`

- **Node:** `escalate_prd_node:1614`
- **Trigger:** Phase 1 of `Gather Requirements` produced sizing questions.
- **Payload fields:** `type`, `phase`, `round`, `max_rounds`, `supported_actions`, `questions[]`
- **Supported actions:** `continue`, `abort`
- **Resume effect:**
    - `continue` (with answers in `feedback` as JSON) → re-enter `Gather Requirements` for phase 2.
    - `abort` → `Abort`.

### `architecture_review_needed` (block diagram)

- **Node:** `escalate_diagram_node:1714`
- **Trigger:** `block_diagram_node` returned `questions` (LLM-flagged ambiguities) or returned no blocks.
- **Payload fields:** `type`, `phase`, `questions[]`, `block_diagram_summary`, `supported_actions`
- **Supported actions:** `continue`, `feedback`, `abort`
- **Resume effect:**
    - `continue` → `Memory Map`.
    - `feedback` (with text) → re-enter `Block Diagram` with `human_feedback` set.
    - `abort` → `Abort`.

### `architecture_review_needed` (constraints structural)

- **Node:** `escalate_constraints_node:1767`
- **Trigger:** `constraint_result.has_structural == True`.
- **Payload fields:** `type`, `phase`, `round`, `max_rounds`, `violations[]`, `violations_history[]`, `block_diagram_summary`, `supported_actions`
- **Supported actions:** `retry`, `accept`, `feedback`, `abort`
- **Resume effect:**
    - `retry` → `Block Diagram` (round preserved; LLM sees the violations as feedback).
    - `accept` → `Finalize Architecture`.
    - `feedback` (with text) → `Block Diagram` with `human_feedback`.
    - `abort` → `Abort`.

### `architecture_review_needed` (constraints exhausted)

- **Node:** `escalate_exhausted_node:1835`
- **Trigger:** `round > max_rounds` *or* `total_rounds > max_rounds*3`.
- **Payload fields:** same as constraints structural.
- **Supported actions:** `retry`, `accept`, `feedback`, `abort`
- **Resume effect:**
    - `retry`/`feedback` → `Block Diagram` with `round` reset to 1 (but `total_rounds` preserved — the hard ceiling stands).
    - `accept` → `Finalize Architecture`.
    - `abort` → `Abort`.

### `final_review` (OK2DEV gate)

- **Node:** `escalate_final_review_node:1483`
- **Trigger:** After `Create Documentation` completes.
- **Payload fields:** `type`, `phase`, `block_specs_path`, `block_diagram_doc_validation_errors[]`, `prd_summary`, `sad_summary`, `frd_summary`, `supported_actions`
- **Supported actions:** `accept`, `feedback`, `abort`
- **Resume effect:**
    - `accept` → `Architecture Complete` (success, hands off to pipeline).
    - `feedback` (with text) → `_feedback_revision_target(text)` picks the earliest spec mentioned: PRD/SAD/FRD/Block Diagram.
    - `abort` → `Abort`.

## Frontend pipeline graph

### `uarch_integration_review`

- **Node:** `integration_review_node:1891`
- **Trigger:** Every tier, after parallel block subgraphs finish.
- **Payload fields:** `type`, `tier`, `block_names[]`, `issues_found`, `issues_fixed`, `summary`, `supported_actions`
- **Supported actions:** `approve`, `revise`, `abort`
- **Resume effect:**
    - `approve` (default) → `advance_tier`. Even when `issues_fixed > 0`.
    - `revise` (with `block_actions`) → `init_tier`. **Must** include `block_actions` like `{"blk": "retry"}` for the blocks whose RTL is stale relative to the freshly edited spec.
    - `abort` → END.
- **Note:** Setting `CORESMITH_STRICT_INTEGRATION_REVIEW=1` makes `issues_fixed > 0` auto-route to `revise`. Default behavior honors explicit `approve`.

### `pipeline_incomplete`

- **Node:** `pipeline_complete_node:2113`
- **Trigger:** Not all blocks have `success=True` after all tiers finished.
- **Payload fields:** `type`, `failed_blocks[]`, `supported_actions`
- **Supported actions:** `retry`, `abort`
- **Resume effect:** `retry` re-enters the tier the failed block is in; `abort` ends the run.
- **Operational note:** Almost always indicates a stuck state. The standard fix is to `restart_block(... from_node='generate_rtl')` per affected block, then `approve` past `uarch_integration_review`.

### `human_intervention_needed` (per block)

- **Node:** `ask_human_node:1438` (in the block subgraph)
- **Trigger:** `decide_node` chose `ask_human` because `attempt > max_attempts` or `debug_action == "ask_human"`.
- **Payload fields:** `type`, `block_name`, `attempt`, `phase`, `diagnosis`, `suggested_fix`, `confidence`, `step_log_paths{}`, `rtl_path`, `testbench_path`, `uarch_spec_path`, `supported_actions`, `outer_agent_guidance`, `ers_summary`, RTL snippet
- **Supported actions:** `retry`, `fix_rtl`, `fix_tb`, `add_constraint`, `skip`, `abort`
- **Resume effect:**
    - `retry` → `generate_rtl` (LLM regenerates).
    - `fix_rtl` → `generate_rtl` step but skipping the LLM (re-runs lint against on-disk file).
    - `fix_tb` → `generate_testbench` step (re-runs sim against on-disk file).
    - `add_constraint` (with `feedback`) → constraint appended to `.coresmith/blocks/<name>/constraints.json`; `retry` semantics.
    - `skip` → `block_done` with failure.
    - `abort` → `block_done` and pipeline aborted.

### `integration_failure`

- **Node:** `integration_check_node:2482`
- **Trigger:** chip_top.v lint failed or port mismatch list non-empty.
- **Payload fields:** `type`, `mismatches[]`, `lint_errors[]`, `integration_top_path`, `supported_actions`
- **Supported actions:** `retry`, `fix_rtl`, `skip`, `abort`
- **Resume effect:**
    - `retry` → re-enter `integration_check` (LLM regenerates chip_top).
    - `fix_rtl` → re-enter `integration_check` but trust the on-disk file.
    - `skip` → END (pipeline aborted).
    - `abort` → END.

### `integration_dv_failure`

- **Node:** `integration_dv_node:2763`
- **Trigger:** Integration smoke TB failed cocotb.
- **Payload fields:** `type`, `contract_audit{}`, `tb_path`, `integration_top_path`, `step_log_paths{}`, `supported_actions`
- **Supported actions:** `retry`, `fix_rtl`, `fix_tb`, `abort` (+ `skip` if `CORESMITH_ALLOW_SKIP_INTEGRATION_DV=1`)
- **Resume effect:**
    - `retry` → re-enter `integration_dv` (regenerates TB).
    - `fix_rtl`/`fix_tb` → re-run cocotb against on-disk file.
    - `skip` (env-gated) → advance to `validation_dv`.
    - `abort` → END.
- **Note:** The `contract_audit` field is the ContractAuditAgent output; use it to decide between `fix_rtl` and `fix_tb`.

### `validation_dv_failure`

- **Node:** `validation_dv_node:3370`
- **Trigger:** Validation TB (ERS-driven, KPI tests) failed.
- **Payload fields:** same as integration DV plus `ers_context`.
- **Supported actions:** `retry`, `fix_rtl`, `fix_tb`, `abort` (+ `skip` if `CORESMITH_ALLOW_SKIP_VALIDATION_DV=1`)
- **Resume effect:** same shape as integration DV failure.

## Backend graph

### `human_intervention_needed` (graph=backend)

- **Node:** `ask_human_node:1378` (backend)
- **Trigger:** `decide_node` after diagnose escalated.
- **Payload fields:** `type`, `graph`, `block_name`, `attempt`, `max_attempts`, `phase`, `error`, `diagnosis`, `category`, `confidence`, `suggested_fix`, `attempt_history[]`, `category_counts`, `routed_def_path`, `gds_path`, `step_log_paths{}`, `supported_actions`
- **Supported actions:** `retry`, `skip`, `abort`
- **Resume effect:**
    - `retry` → `increment_attempt` → re-enter the failing phase (PnR / DRC / LVS / timing).
    - `skip` → `advance_block` (mark failure, continue).
    - `abort` → `backend_complete`.

## Tapeout graph

### `tapeout_intervention_needed`

- **Node:** `ask_human_node:737` (tapeout)
- **Trigger:** `diagnose_tapeout` returned `escalate` or a wrapper step failed.
- **Payload fields:** `type`, `graph`, `phase`, `error`, `attempt`, `diagnosis`, `wrapper_drc_result`, `wrapper_lvs_result`, `precheck_result`, `submission_dir`, `supported_actions`
- **Supported actions:** `retry`, `fix_pnr`, `skip`, `abort`
- **Resume effect:**
    - `retry` → re-enter `generate_wrapper` (regenerate everything).
    - `fix_pnr` → re-enter `wrapper_pnr` (re-run just PnR with the architect's edits).
    - `skip` → `tapeout_complete` (accept as-is).
    - `abort` → `tapeout_complete` (terminal).

## Universal resume conventions

| Action | Meaning across all graphs |
|---|---|
| `retry` | LLM regenerates; same attempt budget consumed. |
| `fix_rtl` / `fix_tb` / `fix_pnr` | Outer agent already edited the file on disk; re-run the failing stage **without an LLM call**. |
| `add_constraint` | Outer agent records a rule in `.coresmith/blocks/<name>/constraints.json`; next LLM regeneration will see it. |
| `skip` | Give up on this thing; move forward (may require env opt-in for DV stages). |
| `abort` | Terminate the whole graph. |
| `accept` | Architect chooses to proceed despite violations. |
| `approve` / `revise` | Used only by `uarch_integration_review`. |
| `continue` | Architect answered questions; proceed. |
| `feedback` | Architect supplies written text; LLM re-runs with it in context. |

## Reading the payload from outside

Programmatic access:

- HTTP daemon: `GET /run/state` or `GET /architecture/state` returns `interrupts[0].payload`.
- MCP: `get_pipeline_state` / `get_architecture_state` / `get_backend_state` / `get_tapeout_state`.
- CLI: `coresmith state` (pipeline) or `coresmith architecture state`.

The payload is the full source of truth for the decision — `step_log_paths`, `rtl_path`, `tb_path`, and contract audit JSON paths point to everything you might need to inspect on disk before resuming.
