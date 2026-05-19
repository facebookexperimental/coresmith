# Backend And Tapeout

The backend and tapeout graphs run after the frontend pipeline has generated and
verified RTL and synthesis artifacts.

## Backend Graph

The backend graph is implemented in `orchestrator/langgraph/backend_graph.py`.
It operates on the integrated flat design, not as a fully independent per-block
frontend rerun.

```text
init_design
  -> flat_top_synthesis
  -> run_pnr
  -> drc
  -> lvs
  -> timing_signoff
  -> generate_wrapper
  -> mpw_precheck
  -> backend_complete
  -> generate_3d_view
  -> final_report
```

The backend start gate checks that all blocks have required RTL and synthesis
artifacts. If any `rtl/<block>/...v` or `syn/output/<block>/<block>_netlist.v`
artifact is missing, backend start fails before resetting the backend
checkpoint.

## Backend Artifacts

Common backend outputs are written under:

```text
syn/output/<design>/
syn/output/<design>/pnr/
openframe_submission/
```

Important paths carried in backend state include:

- `integration_top_path`
- `flat_netlist_path`
- `flat_sdc_path`
- `routed_def_path`
- `pnr_verilog_path`
- `pwr_verilog_path`
- `spef_path`
- `gds_path`
- `spice_path`
- `wrapper_rtl_path`
- `submission_dir`
- `final_report_path`

## Backend Failure Handling

EDA steps are LLM-assisted. Each step receives a baseline script or prompt,
design context, prior failures, and constraints. Failures route through backend
diagnosis and either retry with adjusted constraints, interrupt for review, skip
the failing block or phase, or abort.

MCP exposes direct helper tools such as `run_backend_step(step="pnr" | "drc" |
"lvs", ...)` for focused iteration. Those helpers are operational shortcuts,
not replacements for the full graph gate.

## Tapeout Graph

The tapeout graph is implemented in `orchestrator/langgraph/tapeout_graph.py`.
It builds an OpenFrame submission from passing backend results.

```text
generate_wrapper
  -> synthesize_wrapper
  -> wrapper_pnr
  -> wrapper_drc
  -> wrapper_lvs
  -> mpw_precheck
  -> tapeout_complete
```

Tapeout requires at least one passing backend block. It can reuse backend wrapper
artifacts when available. Optional GPIO mapping can be provided as JSON;
otherwise wrapper generation attempts to assign GPIOs automatically.

## Tapeout Outputs

The tapeout graph reports:

- wrapper RTL
- wrapper netlist
- routed DEF
- wrapper GDS
- wrapper SPICE
- submission directory
- MPW precheck result

The OpenFrame submission directory is usually `openframe_submission/`.

