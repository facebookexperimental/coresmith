# Usage

The pipeline can be driven by `coresmithd` or by MCP. Both use the same
LangGraph checkpoints and on-disk artifacts.

## Before Starting

Run preflight from the project root:

```bash
make preflight
```

For a small smoke test:

```bash
make demo
```

For a custom block registry, provide YAML with a `blocks:` section:

```yaml
blocks:
  adder8:
    tier: 1
    rtl_target: "rtl/adder8/adder8.v"
    testbench: "tb/cocotb/test_adder8.py"
    description: |
      8-bit unsigned adder...
```

## Option A: `coresmithd`

`coresmithd` is a FastAPI daemon for one project root. It exposes the frontend
pipeline as HTTP endpoints and writes a discovery file at:

```text
.coresmith/daemon.json
```

Start it:

```bash
export CORESMITH_PROJECT_ROOT=/path/to/project
venv/bin/python -m orchestrator.daemon.server
```

The daemon binds to a free `127.0.0.1` port unless `--port` is provided.

### HTTP Flow

Read the daemon port:

```bash
PORT=$(jq -r .port .coresmith/daemon.json)
```

Start a run:

```bash
curl -sS -X POST "http://127.0.0.1:${PORT}/run/start" \
  -H "content-type: application/json" \
  -d '{
    "max_attempts": 5,
    "target_clock_mhz": 50.0,
    "blocks_file": "examples/adder8/blocks.yaml"
  }' | jq .
```

Poll state:

```bash
curl -sS "http://127.0.0.1:${PORT}/run/state" | jq .
```

Pause:

```bash
curl -sS -X POST "http://127.0.0.1:${PORT}/run/pause" | jq .
```

Resume after an interrupt:

```bash
curl -sS -X POST "http://127.0.0.1:${PORT}/run/resume" \
  -H "content-type: application/json" \
  -d '{
    "action": "fix_rtl",
    "rtl_fix_description": "Corrected output width in rtl/adder8/adder8.v",
    "rationale": "Verilator lint reported a width mismatch on sum."
  }' | jq .
```

Supported daemon resume actions mirror the frontend graph actions:

- `approve`
- `retry`
- `fix_rtl`
- `fix_tb`
- `add_constraint`
- `skip`
- `abort`

The current daemon implementation covers the frontend run lifecycle. Backend
and tapeout control are currently exposed through MCP.

## Option B: MCP

Start the MCP server:

```bash
make mcp
```

An MCP client can then call the pipeline tools.

### Architecture

Use the architecture tools when starting from requirements:

```text
start_architecture(requirements="...", target_clock_mhz=50.0)
get_architecture_state()
resume_architecture(action="answer" | "approve" | "revise" | "abort", ...)
```

When architecture reaches OK2DEV, it writes `.coresmith/block_specs.json` for
the frontend pipeline.

### Frontend

Start the frontend pipeline:

```text
start_pipeline(max_attempts=5, target_clock_mhz=50.0)
```

or with an explicit registry:

```text
start_pipeline(
  max_attempts=5,
  target_clock_mhz=50.0,
  blocks_file="examples/adder8/blocks.yaml"
)
```

Monitor:

```text
get_pipeline_state()
get_pipeline_events(limit=100)
```

Resume:

```text
resume_pipeline(action="approve")
resume_pipeline(action="fix_rtl", rtl_fix_description="...")
resume_pipeline(action="fix_tb", rtl_fix_description="...")
resume_pipeline(action="add_constraint", constraint="...")
resume_pipeline(action="retry")
resume_pipeline(action="abort")
```

For multiple parallel block interrupts, pass per-block actions as JSON:

```text
resume_pipeline(
  action="retry",
  block_actions='{"alu": "fix_rtl", "decoder": "fix_tb"}'
)
```

Pause and restart helpers:

```text
pause_pipeline()
restart_node(node_name="generate_rtl")
restart_block(block_name="alu", from_node="generate_uarch_spec")
```

### Backend

After the frontend state reports `next_action: "start_backend"`, call:

```text
start_backend(max_attempts=3, target_clock_mhz=50.0)
get_backend_state()
resume_backend(action="retry" | "skip" | "abort")
pause_backend()
```

For focused backend iteration:

```text
run_backend_step(step="pnr", block_name="chip_top")
run_backend_step(step="drc", block_name="chip_top")
run_backend_step(step="lvs", block_name="chip_top")
```

### Tapeout

After backend passes:

```text
start_tapeout(target_clock_mhz=50.0, max_attempts=2)
get_tapeout_state()
resume_tapeout(action="retry" | "fix_pnr" | "skip" | "abort")
```

Optional GPIO mapping is passed as a JSON string to `start_tapeout`.

## Headless CLI

The legacy CLI runner is still useful for CI or demos:

```bash
python run_pipeline.py
```

It auto-approves uArch and integration review interrupts, retries failures until
`MAX_ATTEMPTS`, and stops if the all-block completion gate fails. Use
`coresmithd` or MCP for controlled diagnosis and explicit resume decisions.

