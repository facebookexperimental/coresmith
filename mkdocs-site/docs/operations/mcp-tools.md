# MCP tools

The MCP server (`orchestrator/mcp_server.py`) exposes the same `GraphLifecycle` plumbing as the HTTP daemon, but over stdio MCP — the protocol used by Claude CLI, Cursor, and other LLM IDE clients. The backend and tapeout graphs are only reachable through MCP today; the HTTP daemon currently exposes only architecture and pipeline.

## When to use what

| Surface | Best for |
|---|---|
| HTTP daemon | Outer agents on cron, shell scripts, CI |
| MCP server | Interactive LLM-driven sessions, Cursor / Claude CLI workflows |
| Both | Operate against the same checkpoints — they're transport, not state |

## Tools

### Architecture

| Tool | Arguments | Effect |
|---|---|---|
| `start_architecture` | `requirements`, `target_clock_mhz`, `pdk_config_path`, `max_rounds` | Launch architecture graph |
| `get_architecture_state` | — | Poll status + interrupts |
| `resume_architecture` | `action`, `feedback`, `rationale` | Respond to interrupt |
| `pause_architecture` | — | Cancel task |
| `restart_architecture_node` | `from_node`, `feedback` | Fork from checkpoint to re-run a specific node |

### PDK & benchmarking

| Tool | Arguments | Effect |
|---|---|---|
| `run_benchmark` | block-specific args | Run synthesis benchmark on a block (returns gate count, timing) |
| `characterize_pdk` | `pdk_config_path` | Load / describe PDK |

### Pipeline

| Tool | Arguments | Effect |
|---|---|---|
| `get_pipeline_status` | — | Short-form status poll |
| `get_pipeline_events` | `start_line`, `end_line` | Stream events from `pipeline_events.jsonl` |
| `start_pipeline` | `blocks_file`, `max_attempts`, `target_clock_mhz` | Launch pipeline |
| `get_pipeline_state` | — | Full state snapshot |
| `resume_pipeline` | `action`, `feedback`, `rtl_fix_description`, `block_actions`, `rationale` | Respond to interrupt |
| `pause_pipeline` | — | Cancel task |
| `restart_node` | `block_name`, `from_node` | Fork a single block to re-run from a node |
| `skip_block` | `block_name` | Mark block as skipped |
| `restart_block` | `block_name`, `from_node`, `feedback` | Restart a block from a specific node (more flexible than `restart_node`) |
| `run_step` | `block_name`, `step_name`, `override_params` | Run a single step manually (debug) |

### Backend

| Tool | Arguments | Effect |
|---|---|---|
| `start_backend` | — | Launch backend graph |
| `get_backend_state` | — | Poll |
| `resume_backend` | `action`, `feedback`, `rationale` | Respond |
| `pause_backend` | — | Cancel |
| `skip_backend_block` | `block_name` | Skip |
| `run_backend_step` | `block_name`, `step_name` | Run a single backend step |

### Tapeout

| Tool | Arguments | Effect |
|---|---|---|
| `start_tapeout` | — | Launch tapeout graph |
| `get_tapeout_state` | — | Poll |
| `resume_tapeout` | `action`, `feedback`, `rationale` | Respond |

### Project utilities

| Tool | Arguments | Effect |
|---|---|---|
| `reset_project` | — | Wipe `.coresmith/`, start fresh |
| `get_project_info` | — | Project summary (PDK, artifacts present, ERS, block_specs) |
| `mark_block_passed` | `block_name`, `success` | Manually set a block's pass/fail state |
| `get_graph_structure` | — | Dump graph node/edge structure + prompts |
| `get_node_prompt` | `node_id` | Retrieve the system prompt for a node |

## Useful patterns

### Re-running a node after an outer-agent edit

```text
1. Outer agent inspects diagnosis.json, decides the RTL needs a manual fix.
2. Outer agent edits rtl/<block>/<block>.v on disk.
3. Outer agent calls resume_pipeline(action="fix_rtl") to re-run the failing stage
   without an LLM call.
```

### Re-doing an architecture decision after the fact

```text
restart_architecture_node(from_node="Block Diagram",
                         feedback="Use a wider data bus to absorb the burst from CSI ingress.")
```

This forks from the most recent checkpoint where Block Diagram was the next node and re-enters with the new feedback.

### Forcing a block to re-author RTL after spec drift

```text
restart_block(block_name="entropy_enc", from_node="generate_rtl")
```

Useful when the integration review edited the uArch spec but the block's RTL still reflects the old spec.

## When MCP and HTTP can collide

Both surfaces operate on the same SQLite checkpoint. If you have both an HTTP daemon and an MCP session pointing at the same project root, both can issue start/resume calls. The lock inside `GraphLifecycle.safe_start` / `safe_resume` serializes the actual graph invocations, but you can still end up with confusing interleavings (e.g. MCP resumes an interrupt that the HTTP daemon was about to resume with a different action).

Convention: one driver per project root. Pick HTTP for automation, MCP for hands-on, but don't mix.
