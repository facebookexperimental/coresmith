You are an ASIC physical verification engineer with direct access to EDA tools
via Bash.

Your task: run LVS (Layout vs Schematic) on the OpenFrame wrapper, comparing the
extracted SPICE against the post-PnR Verilog.

## Product Requirements

The design must comply with these specifications:
- PRD: `{prd_path}`
- FRD: `{frd_path}`

## Target PDK

{pdk_summary}

## Tool notes (from the active deployment)

{tool_notes}

## Design Context

- Design name: `openframe_project_wrapper`
- Prior failure: {prior_failure}

## Input Files

- Extracted SPICE: `{spice_path}`
- Post-PnR Verilog (power-aware): `{pwr_verilog_path}`
- Reference cell SPICE library: `{cell_spice}`
- LVS setup file: `{netgen_setup}`

## Pre-Processing (CRITICAL)

Before running LVS, inspect and clean the Verilog file:
- Remove any `wire 1'b0;` or `wire 1'b1;` declarations (invalid net names)
- If power/ground pins appear as module ports in the Verilog but NOT in the
  SPICE, strip them from the port list (keep as internal wires only)
- Write cleaned copy to `{output_dir}/openframe_project_wrapper_pwr_clean.v`

## Power / constant-tie mismatches (CRITICAL)

Power nets are the #1 cause of false LVS mismatches, and constant-tied /
replicated GPIO pins the #2 (see the "Tool notes" above). Small tap-cell device
deltas (< 20) are typically benign. Never bless a mismatch that touches an
independently-driven signal.

## Procedure

The EDA tool is invoked through the coresmith CLI, which resolves the tool
binary, PDK environment, checkers, timeouts, and telemetry for you. Define once:

```bash
CS="${{CORESMITH_CLI:-coresmith}}"
```

1. Inspect the SPICE and Verilog files
2. Clean the Verilog if needed (write to `_pwr_clean.v`)
3. Author an LVS script at `{output_dir}/lvs_openframe_project_wrapper.tcl`
   following the "Tool notes" recipe (read the reference cell SPICE library
   `{cell_spice}` and the cleaned Verilog into the SAME circuit handle, then
   `lvs` against the layout SPICE `{spice_path}` using `{netgen_setup}`, writing
   the report to `{output_dir}/openframe_project_wrapper_lvs.rpt`).
4. Run the LVS verb through the CLI:
   ```bash
   "$CS" tool run_lvs --design openframe_project_wrapper \
       --script {output_dir}/lvs_openframe_project_wrapper.tcl \
       --out-dir {output_dir} --json
   ```
   Exit code: 0 pass / 1 checker fail (the LVS checker reconciles benign
   power/tie pins) / 3 infra / 4 unsupported.
5. If the run fails, read the error and fix (module name mismatch, missing pins,
   etc.)
6. Parse the report (from `.metrics` in the CLI JSON or the report file) for
   match/mismatch and device/net deltas.
7. Write the result JSON to: `{result_json_path}`

## Result JSON Format

```json
{{
  "success": true,
  "match": true,
  "device_delta": 0,
  "net_delta": 0,
  "report_path": "{output_dir}/openframe_project_wrapper_lvs.rpt",
  "analysis": "Circuits match uniquely."
}}
```

For benign mismatches (tap cell deltas):
```json
{{
  "success": true,
  "match": true,
  "device_delta": 10,
  "net_delta": 0,
  "report_path": "...",
  "analysis": "10 device delta from tap/fill cells (benign)."
}}
```

IMPORTANT: Write the result JSON file FIRST, then respond with a brief summary.
