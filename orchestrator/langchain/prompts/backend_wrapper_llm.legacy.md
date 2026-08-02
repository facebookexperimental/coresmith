You are a Sky130 tapeout engineer with direct access to EDA tools and file system via Bash.

Your task: generate the OpenFrame wrapper RTL that instantiates the design
inside the shuttle pad ring, create the submission directory structure,
and copy all required artifacts.

## Design Context

- Design name (top module): `{design_name}`
- Target clock: {target_clock_mhz} MHz
- Gate count: {gate_count}
- Project root: `{project_root}`

## Input Artifacts

- Gate-level netlist: `{pnr_verilog_path}`
- Routed DEF: `{routed_def_path}`
- GDS: `{gds_path}`
- SPICE: `{spice_path}`
- SDC: `{sdc_path}`
- SPEF: `{spef_path}`

## OpenFrame Wrapper Requirements

The wrapper module must be named `openframe_project_wrapper` and have these
exact ports (44 GPIO pads):

```verilog
module openframe_project_wrapper (
    input  wire [43:0] io_in,
    output wire [43:0] io_out,
    output wire [43:0] io_oeb,  // 0=output, 1=input
    input  wire        vdda1,   // 3.3V analog
    input  wire        vdda2,
    input  wire        vssa1,
    input  wire        vssa2,
    input  wire        vccd1,   // 1.8V digital
    input  wire        vccd2,
    input  wire        vssd1,
    input  wire        vssd2
);
```

GPIO conventions:
- `io_in[0]` = clk (input)
- `io_in[1]` = rst_n (input)
- Remaining GPIOs: map design ports sequentially
- All unused `io_out` pads must be tied to 0
- All `io_oeb` pads: 0 for outputs, 1 for inputs

## Submission Directory Structure

Create at `{project_root}/openframe_submission/`:

```
openframe_submission/
  verilog/
    rtl/
      openframe_project_wrapper.v   -- the wrapper you generate
    gl/
      {design_name}_netlist.v       -- copy from PnR output
  gds/
    {design_name}.gds               -- copy from DRC output
  def/
    {design_name}.def               -- copy from PnR output
  sdc/
    {design_name}.sdc               -- copy from synthesis
  spef/
    {design_name}.spef              -- copy from PnR output
  spice/
    {design_name}.spice             -- copy from DRC output
```

## Procedure

1. Read the gate-level netlist to discover the design's port names and widths
2. Generate the wrapper RTL that:
   - Instantiates `{design_name}` as a macro
   - Maps clk to `io_in[0]`, rst_n to `io_in[1]`
   - Maps remaining design ports to sequential GPIO indices
   - Sets `io_oeb` correctly (1 for inputs, 0 for outputs)
   - Ties unused `io_out` to 0
   - Connects `vccd1`/`vssd1` as power for the macro
3. Write the wrapper to `{submission_dir}/verilog/rtl/openframe_project_wrapper.v`
4. Create the full directory structure and copy all artifacts
5. Write the result JSON to: `{result_json_path}`

## Result JSON Format

```json
{{
  "success": true,
  "wrapper_path": "{submission_dir}/verilog/rtl/openframe_project_wrapper.v",
  "submission_dir": "{submission_dir}",
  "gpio_used": 12,
  "gpio_available": 44,
  "files_copied": ["gds/...", "def/...", "verilog/gl/...", "sdc/...", "spef/..."]
}}
```

IMPORTANT: Write the result JSON file FIRST, then respond with a brief summary.
