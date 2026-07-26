# LangGraph patterns used in coresmith

Every coresmith graph is built with `langgraph.graph.StateGraph` and follows the same patterns. Knowing them makes any individual graph reference half as much reading.

## State as a `TypedDict` with reducers

Each graph defines a `State` `TypedDict`. Most fields are *overwritten* on each node return, but some fields are *accumulated* across nodes using `typing.Annotated` reducers.

```python
class OrchestratorState(TypedDict):
    project_root: Annotated[str, _last]                     # latest wins (fan-in safe)
    completed_blocks: Annotated[list[dict], operator.add]   # appended across all branches
    integration_review_action: Optional[str]                # overwritten
```

Common reducers in coresmith:

- `operator.add` — append items from every parallel branch (used for `completed_blocks`, `violations_history`, `human_response_history`).
- `_last` — keep the value from whichever branch returned last (used for paths/IDs that need fan-in safety: `step_log_paths`, `routed_def_path`).
- *(no reducer)* — last write wins; not safe under `Send` fan-out.

You will see `Annotated[..., _last]` heavily on fields that get written during parallel block subgraphs running via `Send(...)`.

## Nodes, edges, and conditional routing

A node is a `def node_name(state) -> dict` (or `async def`). Whatever the function returns is *merged into state* using the reducers defined on each field.

Edges are added with:

```python
builder.add_edge("source_node", "target_node")
builder.add_conditional_edges(
    "source_node",
    route_function,           # returns one of the keys below
    {"PASS": "finalize", "FAIL": "diagnose", "ASK": "ask_human"},
)
```

The router is a plain Python function (`def route_function(state) -> str`) that reads state and returns a string key. By convention, the router string keys in coresmith are written in UPPER_SNAKE_CASE.

You'll see one router per branching node. Their names are stable across the codebase: `route_after_<node>`, `route_decision`, `route_after_human`, `route_after_increment`.

## Per-block subgraphs and `Send(...)` fan-out

The frontend pipeline graph compiles a **block subgraph** once, then fans it out per block in a tier using LangGraph's `Send(...)` primitive (`orchestrator/langgraph/pipeline_graph.py:1735`). Each `Send` invocation passes a *new* state dict scoped to one block.

```python
def fan_out_tier(state):
    tier_blocks = [b for b in state["block_queue"] if b["tier"] == current_tier]
    return [Send("process_block", block_state_for(b, state)) for b in tier_blocks]
```

This is what makes per-tier parallelism work: blocks within the same tier run their full uArch→RTL→TB→synth loop concurrently, and their results are merged back via `operator.add` accumulators.

## Interrupts and resume

LangGraph's `interrupt(payload)` parks the graph mid-node and persists the payload in the checkpoint. When the caller invokes `graph.ainvoke({Command(resume=value)}, config)`, the interrupted node receives `value` as the return of `interrupt(...)` and execution continues.

coresmith wraps this in `GraphLifecycle.safe_resume(...)` (`orchestrator/graph_lifecycle.py:217`), which:

1. Collects pending interrupt IDs from the checkpoint.
2. Builds a `{interrupt_id: resume_value}` map.
3. Issues the resume as a new async task tied to the same thread ID.

Every interrupt in coresmith carries a `supported_actions` field so the outer agent doesn't have to guess what's legal. See [Interrupts catalog](../reference/interrupts.md).

## Checkpointing and recovery

Each graph has its own `AsyncSqliteSaver` rooted at `<project_root>/.coresmith/<name>_checkpoint.db` (e.g. `architecture_checkpoint.db`, `pipeline_checkpoint.db`). Pragmas set by `GraphLifecycle.ensure_graph()` (`orchestrator/graph_lifecycle.py:104`):

- `journal_mode=WAL`
- `synchronous=FULL`
- `busy_timeout=5000`

On daemon startup, `GraphLifecycle.ensure_graph` detects a parked checkpoint and surfaces it as `status=interrupted`. `_close_orphaned_events()` (`orchestrator/graph_lifecycle.py:69`) walks `pipeline_events.jsonl` and writes a synthetic `graph_node_exit` for any `graph_node_enter` that the daemon never closed (i.e. the daemon was killed mid-node).

## Telemetry surfaces

Every LLM call writes to `<project_root>/.coresmith/llm_calls.jsonl` with full prompts, response, token counts, and cost (`orchestrator/langchain/agents/coresmith_llm.py:225`).

Every graph node enter/exit and every interrupt is logged to `pipeline_events.jsonl` and emitted as an OpenTelemetry span. Spans are persisted in `<project_root>/.coresmith/traces.db`; `make traces` opens the dashboard.

These three files (`llm_calls.jsonl`, `pipeline_events.jsonl`, `traces.db`) are the source of truth for *what happened*; the graph state is the source of truth for *what's next*.
