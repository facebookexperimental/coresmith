# LLM abstraction (`ClaudeLLM`)

All agents reach the LLM through a single class: `ClaudeLLM` in [`orchestrator/langchain/agents/coresmith_llm.py`](https://github.com/facebookexperimental/coresmith/blob/main/orchestrator/langchain/agents/coresmith_llm.py) (~1260 lines). It abstracts over two CLI-driven providers and is the only place that knows about subprocess management, telemetry, timeouts, and tool exposure.

## Providers

| Provider | Selected when | Underlying binary | Streaming protocol |
|---|---|---|---|
| Claude CLI | default (or `CORESMITH_LLM_PROVIDER=claude`) | `claude -p` | `--output-format stream-json` |
| Codex CLI | `CORESMITH_LLM_PROVIDER=codex` | `codex exec --json` | JSON Lines |

Both expose the standard Anthropic / OpenAI tools (shell, file read/write/edit, glob, grep, web fetch/search). The graphs do not care which one is in use; the same prompt produces the same shape of output through either.

## Models

Default tiers (`coresmith_llm.py:405-434`):

| Tier | Default | Env override |
|---|---|---|
| `DEFAULT_MODEL` | `opus-4.8` | `CORESMITH_MODEL` |
| `BLOCK_MODEL` | `sonnet-4.6` | `CORESMITH_BLOCK_MODEL` |
| Codex model | `gpt-5.6` | `CORESMITH_CODEX_MODEL` |

The choice between `DEFAULT_MODEL` and `BLOCK_MODEL` is made *per agent* in its `__init__`. Integration / review / audit agents always pick `DEFAULT_MODEL`; per-block RTL/TB/uArch/debug agents pick `BLOCK_MODEL`.

Short names (`opus-4.8`, `sonnet-4.6`, `haiku-4.5`) are mapped to full Claude model IDs via `_CLI_MODEL_MAP`. For Codex, `_CODEX_MODEL_MAP` does the equivalent.

## Constructor

```python
ClaudeLLM(
    model: str = DEFAULT_MODEL,
    claude_path: str = "",
    codex_path: str = "",
    timeout: int = 1200,
    max_turns: int = 50,
    disable_tools: bool = False,
)
```

- `timeout` — wall-clock budget for the whole conversation. Most agents override this with `CORESMITH_<NAME>_TIMEOUT`.
- `max_turns` — how many tool-using turns before forcing a final answer.
- `disable_tools` — turns off `Bash`/`Write`/`Edit` for agents that should emit text only (e.g. `BackendEDAAgent`).

## Entry point

```python
async def call(
    self,
    system: str = "",
    prompt: str = "",
    run_name: str = "",
) -> str:
```

Returns the LLM's final response text. The system prompt is the agent's role; the prompt is the per-call task plus any disk paths the LLM needs to read.

## Tools available to the LLM

With `disable_tools=False` (the default):

- `Bash` — full shell.
- `Read`, `Write`, `Edit`, `Glob`, `Grep` — filesystem.
- `WebFetch`, `WebSearch` — outbound HTTP.
- `Task`, `EnterPlanMode` — Claude CLI built-ins.

The LLM can invoke any of these mid-response. Agents that need controlled output (e.g. structured JSON only, with no side effects) instantiate `ClaudeLLM(..., disable_tools=True)`.

## Circuit breaker

`coresmith_llm.py:45-108` implements a process-wide circuit breaker. After 3 consecutive LLM call failures (timeout, non-zero exit, parse error) the breaker opens and subsequent calls raise `CircuitBreakerOpen` immediately. It auto-resets after 60 seconds of no calls.

`GraphLifecycle.run_task` catches `CircuitBreakerOpen` and parks the graph with `status=error` rather than thrashing.

## Stall detection

A separate watchdog (`coresmith_llm.py:1136`) kills the CLI subprocess if it produces no stdout/stderr lines for `_STALL_THRESHOLD_S` seconds (default 1200, scaled by `CORESMITH_TIMEOUT_MULTIPLIER`). This catches hung Claude / Codex processes that are otherwise waiting on a permission prompt forever.

## Telemetry

Every call writes a JSONL row to `<project_root>/.coresmith/llm_calls.jsonl` (`coresmith_llm.py:225`):

```json
{
  "ts": "...",
  "run_name": "rtl_gen:foo",
  "model": "opus-4.8",
  "provider": "claude_cli",
  "duration_s": 47.2,
  "input_tokens": 12000,
  "output_tokens": 4500,
  "cache_read_tokens": 9000,
  "cache_creation_tokens": 3000,
  "cost_usd": 0.18,
  "num_turns": 6,
  "system_prompt": "...",
  "user_prompt": "...",
  "response": "..."
}
```

Each call also emits an OpenTelemetry span. Spans are persisted in `<project_root>/.coresmith/traces.db`. Run `make traces` to open the local dashboard.

## Overriding behavior

| Env var | Effect |
|---|---|
| `CORESMITH_LLM_PROVIDER` | `claude` (default) or `codex` |
| `CORESMITH_MODEL` | Override `DEFAULT_MODEL` |
| `CORESMITH_BLOCK_MODEL` | Override `BLOCK_MODEL` |
| `CORESMITH_CODEX_MODEL` | Override Codex default |
| `CORESMITH_CODEX_SANDBOX` | Passed through to Codex (`danger-full-access` for unrestricted) |
| `CORESMITH_TIMEOUT_MULTIPLIER` | Scales every agent's timeout (useful for slow models). |
| Per-agent timeouts | See [Config & env vars](../operations/config-and-env.md) for the full table. |
