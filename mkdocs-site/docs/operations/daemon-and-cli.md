# Daemon and CLI

coresmith runs as a long-lived FastAPI process per project root. The CLI is a thin client that auto-discovers the daemon and shells to its HTTP endpoints.

## The daemon — `coresmithd`

Source: [`orchestrator/daemon/server.py`](https://github.com/facebookexperimental/coresmith/blob/main/orchestrator/daemon/server.py).

One process per `CORESMITH_PROJECT_ROOT`. On startup it:

1. Loads two `GraphLifecycle` instances — one for the architecture graph, one for the pipeline.
2. Writes `<project_root>/.coresmith/daemon.json` with `{port, pid}` so clients can discover it.
3. Logs to `<project_root>/.coresmith/daemon.log`.
4. Recovers any orphaned checkpoints (parked from a prior crash) and surfaces them as `status=interrupted`.

### HTTP routes

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/healthz` | — | `{ok, project_root, status, pid}` |
| POST | `/run/start` | `{max_attempts, target_clock_mhz, blocks_file}` | `{started, block_count, status}` |
| GET | `/run/state` | — | `_shape_state(snap)` (see below) |
| POST | `/run/resume` | `{action, feedback, rtl_fix_description, block_actions, rationale}` | `{resumed, interrupts, action}` |
| POST | `/run/pause` | — | `{paused, reason}` |
| POST | `/architecture/start` | `{requirements, requirements_file, target_clock_mhz, pdk_config_path, max_rounds}` | `{started, status, requirements_length, target_clock_mhz, pdk_summary}` |
| GET | `/architecture/state` | — | `_shape_arch_state(snap)` |
| POST | `/architecture/resume` | `{action, feedback, rationale}` | `{resumed, action}` |
| POST | `/architecture/pause` | — | `{paused, reason}` |

The backend and tapeout graphs are reachable via the **MCP server**, not the HTTP daemon. See [MCP tools](mcp-tools.md).

### Pipeline state shape (`/run/state`)

```python
{
    "status": "idle" | "running" | "interrupted" | "done" | "paused" | "error",
    "thread_id": str,
    "project_root": str,
    "completed_count": int,
    "completed_blocks": list[{"name", "success", "attempts", "gate_count"}],
    "total_blocks": int,
    "remaining_count": int,
    "pipeline_done": bool,
    "next_nodes": list[str],          # what the graph would run next if resumed
    "interrupts": list[{"id", "payload"}],
    "pending_interrupt_count": int,
    "interrupt_type": str,            # first interrupt's payload.type for quick triage
}
```

### Architecture state shape (`/architecture/state`)

```python
{
    "status": ...,                    # same as pipeline
    "thread_id": str,
    "project_root": str,
    "phase": str,                     # prd, sad, frd, block_diagram, ..., documentation
    "round": int,
    "max_rounds": int,
    "has_prd": bool, "has_sad": bool, "has_frd": bool, "has_ers": bool,
    "has_block_diagram": bool,
    "block_specs_path": str,
    "next_nodes": list[str],
    "interrupts": list[{"id", "payload"}],
    "pending_interrupt_count": int,
    "interrupt_type": str,
    "error": str,
    "success": bool,
}
```

## The CLI — `bin/coresmith`

Source: [`bin/coresmith`](https://github.com/facebookexperimental/coresmith/blob/main/bin/coresmith).

Auto-discovers the daemon by reading `<project_root>/.coresmith/daemon.json`. Every command takes `--project-root` (defaulting to `$CORESMITH_PROJECT_ROOT` or the current directory).

### Daemon lifecycle

| Subcommand | Effect |
|---|---|
| `daemon start [--port]` | Find a free port, fork the daemon, write daemon.json, log to `.coresmith/daemon.log`. |
| `daemon stop` | SIGTERM the daemon process. |
| `daemon status` | Print daemon.json + liveness check. |

### Pipeline runs

| Subcommand | Effect |
|---|---|
| `run start [--blocks-file] [--max-attempts] [--target-clock-mhz]` | `POST /run/start` |
| `run pause` | `POST /run/pause` |
| `state` | `GET /run/state` |
| `resume --action <act> [--feedback] [--rtl-fix-description] [--block-actions] [--rationale]` | `POST /run/resume` |
| `logs [-n / --tail]` | Tail `.coresmith/pipeline_events.jsonl` (default 50 lines). |

Pipeline `--action` values: `approve`, `retry`, `skip`, `abort`, `feedback`, `fix_rtl`, `fix_tb`, `revise`.

`--block-actions` is JSON of the form `'{"blk1": "retry", "blk2": "skip"}'`. Used after a `uarch_integration_review` revise to tell the orchestrator which blocks need re-running.

### Architecture runs

| Subcommand | Effect |
|---|---|
| `architecture start [--requirements-file] [--requirements] [--target-clock-mhz] [--pdk-config-path] [--max-rounds]` | `POST /architecture/start` |
| `architecture state` | `GET /architecture/state` |
| `architecture resume --action <act> [--feedback] [--rationale]` | `POST /architecture/resume` |
| `architecture pause` | `POST /architecture/pause` |

Architecture `--action` values: `continue`, `retry`, `accept`, `feedback`, `abort`.

## Checkpointing

Each graph has its own SQLite checkpoint at:

- `<project_root>/.coresmith/architecture_checkpoint.db`
- `<project_root>/.coresmith/pipeline_checkpoint.db`

`GraphLifecycle.ensure_graph()` (`graph_lifecycle.py:104`) sets:

- `journal_mode=WAL`
- `synchronous=FULL`
- `busy_timeout=5000`

On startup, `_close_orphaned_events()` (`graph_lifecycle.py:69`) scans `pipeline_events.jsonl` and writes synthetic `graph_node_exit` events for any `graph_node_enter` that has no matching exit — this happens when the daemon was killed mid-node.

## Lifecycle wrapper

`GraphLifecycle` (`graph_lifecycle.py:23-223`) wraps each graph with:

| Method | Effect |
|---|---|
| `ensure_graph()` | Lazily build graph + AsyncSqliteSaver. Detect parked checkpoints. |
| `reset_for_new_run()` | Wipe checkpoint DB + WAL/SHM/journal. Reset status to `idle`. |
| `safe_start(initial_input, config)` | `asyncio.create_task(run_task(...))` under a lock. |
| `safe_resume(resume_input, config)` | Same as start but with a `Command(resume=...)`. |
| `run_task(...)` | The actual background task. Catches `GraphInterrupt` (→ `interrupted`), `CircuitBreakerOpen` (→ `error`), `asyncio.CancelledError` (→ `paused`), `Exception` (→ `error`). Truncates `error_message` to 10K chars. |
| `cleanup()` | Close async SQLite context manager. |

The lock guarantees only one start/resume per graph at a time, so two HTTP requests racing each other can't both kick off a new task.

## Restarting the daemon

The daemon is stateless apart from the checkpoint. `daemon stop` followed by `daemon start` re-attaches to the existing checkpoint files; the graph picks up wherever it parked. Orphaned event logs are reconciled on startup.

If the daemon itself is broken (e.g. port conflict, bad permissions), look at `.coresmith/daemon.log` — startup errors are written there before any HTTP response is possible.
