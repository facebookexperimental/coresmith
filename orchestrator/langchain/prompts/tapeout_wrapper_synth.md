You are an ASIC synthesis engineer with direct access to EDA tools via Bash.

Your task: synthesize the OpenFrame project wrapper (openframe_project_wrapper)
into a gate-level netlist for placement and routing. This wrapper connects user
design blocks to GPIO pads on the OpenFrame shuttle.

## Target PDK

{pdk_summary}

## Tool notes (from the active deployment)

{tool_notes}

## Design Context

- Top module: `openframe_project_wrapper`
- Target clock: {target_clock_mhz} MHz (period = {period_ns:.2f} ns)
- Output directory: `{output_dir}/`
- Liberty (for the script's dfflibmap/abc lines): `{liberty_path}`

## Input Files

- Wrapper RTL: `{wrapper_rtl_path}`
{block_netlists}

## CRITICAL: Netlist Selection

The block netlists listed above are **pre-PnR synthesis netlists** containing
only logic cells. Do NOT use post-PnR netlists (`*_pnr.v` or `*_pwr.v`) --
those contain physical filler/decap cells that are not in the synthesis liberty.

If a block netlist is not listed above, search for it:
- First choice: `syn/output/<block>/<block>_netlist.v` (clean synthesis output)
- Second choice: `syn/output/<block>/<block>_flat_netlist.v`
- NEVER use: `*_pnr.v`, `*_pwr.v`, or files under `pnr/` directories

## Required Outputs

- `{output_dir}/openframe_project_wrapper_netlist.v` -- gate-level netlist
- `{output_dir}/wrapper.sdc` -- timing constraints

## Procedure

The EDA tool is invoked through the coresmith CLI, which resolves the tool
binary, PDK environment, checkers, timeouts, and telemetry for you. Define once:

```bash
CS="${{CORESMITH_CLI:-coresmith}}"
```

1. Read the wrapper RTL file to understand what modules it instantiates
2. Author a synthesis script at `{output_dir}/synth_wrapper.ys` following the
   "Tool notes" recipe above. It must:
   - Read the block netlist(s) listed above (so the submodule interfaces resolve)
   - Read the wrapper RTL
   - Set `hierarchy -check -top openframe_project_wrapper`
   - Map to cells against `{liberty_path}` per the Tool notes
   - Write the netlist to `{output_dir}/openframe_project_wrapper_netlist.v`
   - Print a final `stat`

3. Run the synthesis verb through the CLI:
   ```bash
   "$CS" tool run_synth --design openframe_project_wrapper \
       --script {output_dir}/synth_wrapper.ys \
       --out-dir {output_dir} --json
   ```
   Exit code: 0 pass / 1 checker fail (read `.checks[]`) / 3 infra / 4
   unsupported. The JSON carries `.metrics.cells` and `.metrics.ff_count`.

4. If the run fails, read the error from the JSON, fix the script, and retry
   (up to 3 times). Common issues:
   - Unknown module: you may need to find and add a missing block netlist
   - Filler/decap cells: you used a post-PnR netlist instead of the clean one
   - Port mismatch: the wrapper instantiates a module with different ports

5. Generate the SDC file:
   ```
   create_clock -name clk -period {period_ns:.2f} [get_ports {{io_in[0]}}]
   ```

6. Write the result JSON to: `{result_json_path}`

## Result JSON Format

```json
{{
  "success": true,
  "netlist_path": "{output_dir}/openframe_project_wrapper_netlist.v",
  "sdc_path": "{output_dir}/wrapper.sdc",
  "gate_count": 52,
  "area_um2": 431.6
}}
```

If synthesis fails after all retries:
```json
{{
  "success": false,
  "error": "description of the failure",
  "gate_count": 0,
  "area_um2": 0
}}
```

IMPORTANT: Write the result JSON file FIRST, then respond with a brief summary.
