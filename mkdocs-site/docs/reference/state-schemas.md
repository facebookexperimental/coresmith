# State schemas

All four coresmith graphs define their state as a `TypedDict`. This page collects every state field across all four graphs in one place. Use it to answer "where does field X live and what writes it?" without grepping.

## `ArchGraphState` — architecture graph

Defined at `orchestrator/langgraph/architecture_graph.py:109-163`.

| Field | Type | Reducer | Set / written by | Read by |
|---|---|---|---|---|
| `project_root` | `str` | — | START | every node |
| `requirements` | `str` | — | START | every architecture node |
| `pdk_summary` | `str` | — | START | PRD prompt |
| `target_clock_mhz` | `float` | — | START | PRD, block diagram, clock tree |
| `pdk_config` | `dict` | — | START | constraint check |
| `max_rounds` | `int` | — | START | route_after_increment |
| `round` | `int` | — | increment_round_node | route_after_increment |
| `total_rounds` | `int` | — | increment_round_node | route_after_increment |
| `phase` | `str` | — | every node | observability |
| `prd_spec` | `dict?` | — | gather_requirements (phase 2) | SAD, FRD, ERS, block diagram |
| `prd_questions` | `list?` | — | gather_requirements (phase 1) | route_after_prd, Escalate PRD |
| `prd_answers` | `dict?` | — | outer agent on resume | gather_requirements (phase 2) |
| `sad_spec` | `dict?` | — | system_architecture_node | FRD, ERS |
| `frd_spec` | `dict?` | — | functional_requirements_node | block diagram, uArch, ERS |
| `ers_spec` | `dict?` | — | create_documentation_node | downstream pipeline |
| `block_diagram` | `dict?` | — | block_diagram_node | every specialist, constraints |
| `memory_map` | `dict?` | — | memory_map_node | constraints |
| `clock_tree` | `dict?` | — | clock_tree_node | constraints |
| `register_spec` | `dict?` | — | register_spec_node | constraints |
| `benchmark_data` | `dict?` | — | external (MCP `run_benchmark`) | constraints |
| `constraint_result` | `dict?` | — | constraint_check_node | route_after_constraints |
| `violations_history` | `list[dict]` | `operator.add` | constraint_check_node | escalation payloads |
| `questions` | `list[dict]` | `operator.add` | block_diagram_node | Escalate Diagram |
| `human_response_history` | `list[dict]` | `operator.add` | every escalation node | escalation payloads |
| `human_response` | `dict?` | — | resume | router functions |
| `human_feedback` | `str` | — | resume | block_diagram_node |
| `block_diagram_doc` | `dict?` | — | create_documentation_node | dashboards |
| `block_diagram_doc_validation_errors` | `list[str]` | — | create_documentation_node | final review |
| `block_specs_path` | `str` | — | finalize_node | frontend pipeline |
| `block_diagram_doc_path` | `str` | — | create_documentation_node | dashboards |
| `success` | `bool` | — | mark_success_node, abort_node | terminal |
| `error` | `str` | — | abort_node | terminal |

## `BlockState` — per-block subgraph (frontend pipeline)

Defined at `orchestrator/langgraph/pipeline_graph.py:202-261`.

| Field | Type | Reducer | Notes |
|---|---|---|---|
| `project_root` | `str` | — | Inherited via Send |
| `target_clock_mhz` | `float` | — | Inherited |
| `max_attempts` | `int` | — | Retry budget |
| `pipeline_run_start` | `float` | — | Epoch; used by `_file_is_fresh` |
| `current_block` | `dict` | — | The block dict from `block_specs.json` |
| `attempt` | `int` | — | 1-indexed; incremented at `decide_node:1280` |
| `phase` | `str` | — | `init`/`uarch`/`rtl`/`lint`/`tb`/`sim`/`synth` |
| `uarch_approved` | `bool` | — | Auto-set true |
| `lint_clean` | `bool` | — | Verilator pass |
| `sim_passed` | `bool` | — | cocotb pass |
| `synth_success` | `bool` | — | Yosys pass |
| `synth_gate_count` | `int` | — | From Yosys |
| `rtl_path` | `str` | — | Path to generated RTL |
| `tb_path` | `str` | — | Path to generated TB |
| `step_log_paths` | `dict` | `_last` | `{step: log_path}` |
| `debug_action` | `str` | — | `retry_rtl`/`retry_tb`/`ask_human`/`escalate` |
| `human_response` | `dict?` | — | Outer-agent action |
| `force_regen_tb` | `bool` | — | Skip TB reuse |
| `completed_blocks` | `list[dict]` | `operator.add` | Reducer; appends back to orchestrator |

## `OrchestratorState` — frontend pipeline (tier-level)

Defined at `orchestrator/langgraph/pipeline_graph.py:268-312`. Most fields use `Annotated[..., _last]` reducers because the parallel block subgraphs all write to the same state.

| Field | Type | Reducer | Notes |
|---|---|---|---|
| `project_root` | `str` | `_last` | — |
| `target_clock_mhz` | `float` | `_last` | — |
| `max_attempts` | `int` | `_last` | — |
| `block_queue` | `list[dict]` | `_last` | All blocks |
| `pipeline_run_start` | `float` | `_last` | mtime gate |
| `tier_list` | `list[int]` | — | Sorted unique tiers |
| `current_tier_index` | `int` | — | Index into `tier_list` |
| `completed_blocks` | `list[dict]` | `operator.add` | Reducer over all parallel branches |
| `integration_review_action` | `str?` | — | `approve`/`revise`/`abort` |
| `integration_result` | `dict?` | — | chip_top.v + lint result |
| `integration_dv_result` | `dict?` | — | integration smoke result |
| `validation_dv_result` | `dict?` | — | KPI validation result |
| `contract_audit_result` | `dict?` | — | ContractAuditAgent JSON |
| `pipeline_done` | `bool` | — | Set when validation DV passes |
| `pipeline_aborted` | `bool` | — | Set on human abort |

## `BackendState` — backend graph

Defined at `orchestrator/langgraph/backend_graph.py:72-146`. Annotation `_last` on artifact paths protects against parallel writes.

| Field | Type | Reducer | Notes |
|---|---|---|---|
| `project_root` | `str` | — | — |
| `target_clock_mhz` | `float` | — | — |
| `max_attempts` | `int` | — | — |
| `block_queue` | `list[dict]` | — | Legacy; ignored by Backend Lead |
| `frontend_blocks` | `list[dict]` | — | From pipeline |
| `architecture_connections` | `list[dict]` | — | Reference |
| `design_name` | `str` | — | Top module |
| `block_rtl_paths` | `dict[str,str]` | — | `{name: path}` |
| `glue_blocks` | `list[dict]` | — | Unused in current flow |
| `integration_top_path` | `str` | `_last` | `rtl/integration/{design}_top.v` |
| `flat_netlist_path` | `str` | `_last` | Yosys output |
| `flat_sdc_path` | `str` | `_last` | SDC |
| `synth_gate_count` | `int` | — | From Yosys |
| `synth_area_um2` | `float` | — | Estimated |
| `current_block` | `dict` | — | `{"name": design_name}` (synthetic) |
| `current_block_index` | `int` | — | Always 0 in flat mode |
| `attempt` | `int` | — | 1-indexed |
| `phase` | `str` | — | `init`/`synth`/`pnr`/`drc`/`lvs`/`signoff`/`wrapper`/`precheck` |
| `constraints` | `list[dict]` | — | Timing/design rules |
| `attempt_history` | `list[dict]` | — | `[{attempt, phase, error, category}]` |
| `previous_error` | `str` | — | LLM-readable context |
| `floorplan_result` | `dict?` | — | `{success, design_area_um2, ...}` |
| `place_result` / `cts_result` / `route_result` | `dict?` | — | per-stage |
| `timing_result` | `dict?` | — | `{met, wns_ns, ..., sign_off}` |
| `power_result` | `dict?` | — | `{total_power_mw, ...}` |
| `drc_result` | `dict?` | — | `{clean, violation_count}` |
| `lvs_result` | `dict?` | — | `{match, device_delta, net_delta, llm_analysis}` |
| `debug_result` | `dict?` | — | `{category, diagnosis, needs_human, suggested_fix, next_action}` |
| `precheck_result` | `dict?` | — | `{pass, checks, errors, warnings}` |
| `routed_def_path` | `str` | `_last` | OpenROAD output |
| `pnr_verilog_path` | `str` | `_last` | Post-PnR Verilog |
| `pwr_verilog_path` | `str` | `_last` | Power-pin-augmented Verilog |
| `spef_path` | `str` | `_last` | Parasitic extraction |
| `gds_path` | `str` | `_last` | From Magic |
| `spice_path` | `str` | `_last` | From Magic |
| `wrapper_rtl_path` | `str` | `_last` | `openframe_project_wrapper.v` |
| `wrapper_result` | `dict?` | — | `{success, wrapper_path, gpio_used, ...}` |
| `submission_dir` | `str` | `_last` | `openframe_submission/` |
| `completed_blocks` | `list[dict]` | `operator.add` | Reducer |
| `step_log_paths` | `dict` | `_last` | Step logs |
| `human_response` | `dict?` | — | Outer agent action |
| `backend_done` | `bool` | — | Terminal flag |
| `viewer_3d_path` | `str` | — | `chip_finish/3d.html` |
| `layout_2d_png_path` | `str` | — | Per-block PNG |
| `final_report_path` | `str` | — | `chip_finish/dashboard.html` |

## `TapeoutState` — tapeout graph

Defined at `orchestrator/langgraph/tapeout_graph.py:65-108`.

| Field | Type | Reducer | Notes |
|---|---|---|---|
| `project_root` | `str` | — | — |
| `target_clock_mhz` | `float` | — | — |
| `blocks` | `list[dict]` | — | Frontend blocks for reference |
| `completed_backend_blocks` | `list[dict]` | — | Carries GDS/DEF paths |
| `gpio_mapping` | `dict?` | — | Explicit pad assignments (auto if None) |
| `phase` | `str` | — | `init`/`wrapper`/`pnr`/`drc`/`lvs`/`precheck`/`done` |
| `attempt` | `int` | — | Retry counter |
| `max_attempts` | `int` | — | Default 3 |
| `previous_error` | `str` | — | LLM-readable context |
| `wrapper_result` | `dict?` | — | `{success, wrapper_path, gpio_used, ...}` |
| `wrapper_pnr_result` | `dict?` | — | `{success, routed_def_path, ...}` |
| `wrapper_drc_result` | `dict?` | — | `{clean, violation_count}` |
| `wrapper_lvs_result` | `dict?` | — | `{match, device_delta, net_delta}` |
| `precheck_result` | `dict?` | — | `{pass, checks, errors, warnings}` |
| `submission_result` | `dict?` | — | `{submission_dir, files_copied}` |
| `diagnosis_result` | `dict?` | — | LLM triage |
| `wrapper_rtl_path` / `wrapper_netlist_path` / `wrapper_routed_def` / `wrapper_gds_path` / `wrapper_spice_path` | `str` | `_last` | Artifact paths |
| `submission_dir` | `str` | `_last` | `openframe_submission/` |
| `step_log_paths` | `dict` | — | Step logs |
| `human_response` | `dict?` | — | `{action: retry/fix_pnr/skip/abort}` |
| `tapeout_done` | `bool` | — | Terminal flag |
