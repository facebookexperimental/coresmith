# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
ClaudeLLM -- Plain Python LLM client backed by an agent CLI.

Provides a simple ``call(system, prompt) -> str`` interface that all agents
(RTLGenerator, TestbenchGenerator, DebugAgent, TimingClosureAgent, etc.) use.

Uses the ``claude`` binary (Claude Code CLI) by default. Set
``CORESMITH_LLM_PROVIDER=codex`` to use ``codex exec`` instead.

Telemetry
---------
Every LLM call is logged to ``.coresmith/llm_calls.jsonl`` with full prompt
and response content, enabling prompt engineering iteration.  Calls are
also wrapped in OpenTelemetry spans with ``input.value`` and ``output.value``
attributes for the webview trace viewer.
"""

from __future__ import annotations

import asyncio
import contextvars
import json as _json
import logging
import os
import queue
import subprocess
import shutil
import tempfile
import threading
import time as _time_mod
from pathlib import Path

from orchestrator._timeouts import scaled

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fix #9 -- Circuit breaker for systemic LLM failures
# ---------------------------------------------------------------------------

class CircuitBreakerOpen(Exception):
    """Raised when the LLM circuit breaker is open (too many consecutive failures)."""


class _CircuitBreaker:
    """Simple circuit breaker that opens after *threshold* consecutive failures.

    Auto-resets after *reset_after_s* seconds of inactivity, so transient
    outages don't require manual intervention.
    """

    def __init__(self, threshold: int = 3, reset_after_s: float = 60.0) -> None:
        self.threshold = threshold
        self.reset_after_s = reset_after_s
        self.consecutive_failures = 0
        self.last_failure_time = 0.0
        self.is_open = False

    def check(self) -> None:
        """Raise ``CircuitBreakerOpen`` if the breaker is open."""
        if self.is_open:
            now = _time_mod.monotonic()
            if now - self.last_failure_time > self.reset_after_s:
                # Auto-reset after cooldown
                self.is_open = False
                self.consecutive_failures = 0
                logger.info("LLM circuit breaker auto-reset after %.0fs cooldown", self.reset_after_s)
                return
            raise CircuitBreakerOpen(
                f"LLM circuit breaker open: {self.consecutive_failures} consecutive "
                f"failures. Check API key / connectivity. "
                f"Auto-resets in {self.reset_after_s - (now - self.last_failure_time):.0f}s."
            )

    def record_success(self) -> None:
        self.consecutive_failures = 0
        self.is_open = False

    def record_failure(self) -> None:
        self.consecutive_failures += 1
        self.last_failure_time = _time_mod.monotonic()
        if self.consecutive_failures >= self.threshold:
            self.is_open = True
            logger.error(
                "LLM circuit breaker OPEN after %d consecutive failures",
                self.consecutive_failures,
            )


# Per-graph circuit breaker registry keyed by graph name (via contextvars)
_llm_breakers: dict[str, _CircuitBreaker] = {}
_llm_breakers_lock = threading.Lock()

_breaker_context: contextvars.ContextVar[str] = contextvars.ContextVar(
    "breaker_context", default=""
)


def _get_breaker(key: str = "") -> _CircuitBreaker:
    with _llm_breakers_lock:
        if key not in _llm_breakers:
            _llm_breakers[key] = _CircuitBreaker(threshold=3, reset_after_s=60.0)
        return _llm_breakers[key]


# ---------------------------------------------------------------------------
# Fix #11 -- Active subprocess registry for external kill capability
# ---------------------------------------------------------------------------

_active_processes_lock = threading.Lock()
_active_processes: dict[int, subprocess.Popen] = {}  # thread-id -> Popen


def _register_process(proc: subprocess.Popen) -> None:
    """Register a running CLI subprocess so it can be killed externally."""
    with _active_processes_lock:
        _active_processes[threading.get_ident()] = proc


def _unregister_process() -> None:
    """Remove the current thread's subprocess from the registry."""
    with _active_processes_lock:
        _active_processes.pop(threading.get_ident(), None)


def kill_active_cli_processes() -> int:
    """Kill all active Claude CLI subprocesses.

    Called by the MCP server's pause_* handlers to terminate hung CLI
    processes that ``asyncio.Task.cancel()`` cannot reach (because the
    blocking ``Popen.communicate()`` runs in a thread executor).

    Returns the number of processes killed.
    """
    killed = 0
    with _active_processes_lock:
        for tid, proc in list(_active_processes.items()):
            try:
                if proc.poll() is None:
                    proc.kill()
                    killed += 1
                    logger.warning("Killed stuck Claude CLI process pid=%d (thread %d)", proc.pid, tid)
            except Exception:
                pass
        _active_processes.clear()
    return killed


# ---------------------------------------------------------------------------
# LLM call telemetry -- JSONL + OpenTelemetry
# ---------------------------------------------------------------------------

_LLM_LOG_RELPATH = ".coresmith/llm_calls.jsonl"
_TRUNCATE_ATTR = 32_000  # OTel attribute max (span attrs); JSONL is untruncated


def _default_project_root() -> str:
    return str(Path(__file__).resolve().parent.parent.parent)


def _llm_log_root() -> str:
    return (
        os.environ.get("CORESMITH_LLM_LOG_ROOT", "").strip()
        or os.environ.get("CORESMITH_TELEMETRY_ROOT", "").strip()
        or os.environ.get("CORESMITH_PROJECT_ROOT", "").strip()
        or _default_project_root()
    )


def _get_llm_tracer():
    """Lazy import to avoid circular deps at module load time."""
    try:
        from opentelemetry import trace
        return trace.get_tracer("coresmith.llm")
    except Exception:
        return None


def _log_llm_call(
    *,
    model: str,
    provider: str,
    system_prompt: str,
    user_prompt: str,
    response: str,
    duration_s: float,
    timeout: int,
    error: str = "",
    timed_out: bool = False,
    usage: dict | None = None,
    start_ts_ns: int | None = None,
) -> None:
    """Write an LLM call record to the JSONL log and an OTel span.

    The JSONL log at ``.coresmith/llm_calls.jsonl`` contains the FULL
    prompt and response (never truncated).  OTel span attributes are
    truncated to ~32K chars to stay within exporter limits.
    """
    ts = _time_mod.time()
    usage = usage or {}
    record = {
        "ts": ts,
        "iso": _time_mod.strftime("%Y-%m-%dT%H:%M:%S", _time_mod.localtime(ts)),
        "model": model,
        "provider": provider,
        "system_prompt_len": len(system_prompt),
        "user_prompt_len": len(user_prompt),
        "response_len": len(response),
        "duration_s": round(duration_s, 2),
        "timeout": timeout,
        "timed_out": timed_out,
        "error": error,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "response": response,
        "usage": usage,
    }

    # Write JSONL (full, untruncated)
    project_root = _llm_log_root()
    log_path = Path(project_root) / _LLM_LOG_RELPATH
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(_json.dumps(record, default=str) + "\n")
    except Exception:
        logger.exception("Failed to write LLM call log")

    # Write OTel span (truncated attributes for exporter safety)
    tracer = _get_llm_tracer()
    if tracer is not None:
        try:
            from opentelemetry import trace
            end_ns = _time_mod.time_ns()
            start_ns = start_ts_ns or (end_ns - int(duration_s * 1_000_000_000))
            attrs = {
                "llm.model_name": model,
                "llm.provider": provider,
                "llm.timeout_s": timeout,
                "llm.duration_s": round(duration_s, 2),
                "llm.timed_out": timed_out,
                "llm.system_prompt_len": len(system_prompt),
                "llm.user_prompt_len": len(user_prompt),
                "llm.response_len": len(response),
                "input.value": (system_prompt + "\n---\n" + user_prompt)[:_TRUNCATE_ATTR],
                "input.mime_type": "text/plain",
                "output.value": response[:_TRUNCATE_ATTR],
                "output.mime_type": "text/plain",
            }
            # Real token / cost telemetry from CLI stream-json `result` event.
            # `input_tokens` here excludes cached input -- add cache_read for
            # an apples-to-apples "tokens delivered to model" sum.
            for k_src, k_dst in (
                ("input_tokens", "llm.input_tokens"),
                ("output_tokens", "llm.output_tokens"),
                ("cache_read_input_tokens", "llm.cache_read_tokens"),
                ("cache_creation_input_tokens", "llm.cache_creation_tokens"),
                ("total_cost_usd", "llm.cost_usd"),
                ("num_turns", "llm.num_turns"),
            ):
                if k_src in usage and usage[k_src] is not None:
                    attrs[k_dst] = usage[k_src]
            span = tracer.start_span(
                f"LLM {model} ({provider})",
                attributes=attrs,
                start_time=start_ns,
            )
            if error:
                span.set_attribute("error", error[:1000])
                span.set_status(trace.StatusCode.ERROR, error[:200])
            span.end(end_time=end_ns)
        except Exception:
            pass  # telemetry must never break the LLM call

# ---------------------------------------------------------------------------
# Stream-JSON output parsing
# ---------------------------------------------------------------------------

def _parse_stream_json(stdout: str) -> tuple[str, dict]:
    """Parse Claude CLI ``--output-format stream-json`` output.

    Each line is one JSON event.  The terminating ``result`` event
    contains the canonical final text plus a ``usage`` block with token
    counts and ``total_cost_usd`` (subscription users see the equivalent
    cost the API would have charged).  If the process was killed
    mid-stream, we fall back to concatenating ``text`` content from
    every ``assistant`` event so the caller still gets *some* response
    text for diagnosis.

    Returns ``(final_text, usage_dict)``.  ``usage_dict`` may be empty
    if no ``result`` event was emitted (timeout / stall / crash).
    """
    final_text = ""
    usage: dict = {}
    cost_usd: float | None = None
    num_turns: int | None = None
    fallback_chunks: list[str] = []
    for raw in stdout.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = _json.loads(raw)
        except _json.JSONDecodeError:
            continue
        ev_type = obj.get("type")
        if ev_type == "result":
            final_text = obj.get("result", "") or final_text
            usage = obj.get("usage") or {}
            cost_usd = obj.get("total_cost_usd")
            num_turns = obj.get("num_turns")
        elif ev_type == "assistant":
            msg = obj.get("message", {}) or {}
            for block in msg.get("content", []) or []:
                if isinstance(block, dict) and block.get("type") == "text":
                    fallback_chunks.append(block.get("text", "") or "")
    if not final_text and fallback_chunks:
        final_text = "".join(fallback_chunks)
    out_usage = dict(usage) if usage else {}
    if cost_usd is not None:
        out_usage["total_cost_usd"] = cost_usd
    if num_turns is not None:
        out_usage["num_turns"] = num_turns
    return final_text, out_usage


def _parse_codex_json(stdout: str) -> tuple[str, dict]:
    """Parse Codex CLI ``exec --json`` output.

    Codex emits one JSON event per line.  The final answer is carried by
    ``item.completed`` events whose item is an ``agent_message``; usage
    arrives on ``turn.completed``.  If multiple agent messages are present,
    the last one is the final response.
    """
    final_text = ""
    usage: dict = {}
    for raw in stdout.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = _json.loads(raw)
        except _json.JSONDecodeError:
            continue
        ev_type = obj.get("type")
        if ev_type == "item.completed":
            item = obj.get("item", {}) or {}
            if item.get("type") == "agent_message":
                final_text = item.get("text", "") or final_text
        elif ev_type == "turn.completed":
            usage = obj.get("usage") or usage
    return final_text, usage



def _parse_kimi_acp_json(stdout: str) -> tuple[str, dict]:
    """Parse a captured Kimi Code ACP JSON-RPC transcript.

    Kimi streams assistant text through ``session/update`` notifications and
    returns token usage on the terminating ``session/prompt`` response. Text
    chunks are concatenated in arrival order. Unknown notifications and
    malformed lines are deliberately ignored so an upstream additive schema
    change does not break a running pipeline.
    """
    chunks: list[str] = []
    usage: dict = {}
    for raw in stdout.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = _json.loads(raw)
        except _json.JSONDecodeError:
            continue

        if obj.get("method") == "session/update":
            update = (obj.get("params") or {}).get("update") or {}
            if update.get("sessionUpdate") != "agent_message_chunk":
                continue
            content = update.get("content") or {}
            if content.get("type") == "text":
                chunks.append(content.get("text", "") or "")
            continue

        result = obj.get("result") or {}
        raw_usage = result.get("usage") or {}
        if raw_usage:
            usage = {
                "input_tokens": raw_usage.get("inputTokens", 0),
                "output_tokens": raw_usage.get("outputTokens", 0),
                "total_tokens": raw_usage.get("totalTokens", 0),
            }
            if raw_usage.get("cachedReadTokens") is not None:
                usage["cache_read_input_tokens"] = raw_usage["cachedReadTokens"]
            if raw_usage.get("cachedWriteTokens") is not None:
                usage["cache_creation_input_tokens"] = raw_usage["cachedWriteTokens"]
            if raw_usage.get("thoughtTokens") is not None:
                usage["reasoning_output_tokens"] = raw_usage["thoughtTokens"]

    return "".join(chunks), usage


def _log_codex_turns(stdout: str, project_root: str, pid: int, wall_start: float) -> int:
    """Append every Codex CLI turn to ``.coresmith/codex_turns.jsonl``.

    Codex emits one JSON event per line on stdout when invoked with
    ``--json``. We persist them verbatim (plus a synthetic ``pid``/``ts``
    header) so the trajectory viewer can show the agent's actual
    reasoning + tool calls instead of just the final response. Each line
    stays a self-contained JSON object so callers can ``jq`` over the
    file directly.

    Returns the number of events written. Failures are swallowed so this
    never breaks a live run.
    """
    try:
        log = Path(project_root) / ".coresmith" / "codex_turns.jsonl"
        log.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        with log.open("a", encoding="utf-8") as f:
            for raw in stdout.splitlines():
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    obj = _json.loads(raw)
                except _json.JSONDecodeError:
                    continue
                # Wrap each codex event with a small header so the webview
                # can correlate by pid (-> llm_start's pid -> run_name).
                rec = {
                    "ts": _time_mod.time(),
                    "wall_start": wall_start,
                    "pid": pid,
                    "event": obj,
                }
                f.write(_json.dumps(rec, default=str))
                f.write("\n")
                n += 1
        return n
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Model name mapping: short names -> Claude CLI model IDs
# ---------------------------------------------------------------------------

_CLI_MODEL_MAP = {
    "opus-4.7":  "claude-opus-4-7",
    "opus-4.6":  "claude-opus-4-7",          # legacy alias -> current Opus
    "sonnet-4.6": "claude-sonnet-4-6",
    "sonnet-4.5": "claude-sonnet-4-6",       # legacy alias -> current Sonnet
    "haiku-4.5": "claude-haiku-4-5-20251001",
    "haiku-3.5": "claude-haiku-4-5-20251001", # legacy alias -> current Haiku
}

_CODEX_MODEL_MAP = {
    # Preserve existing CoreSmith model tiers when switching providers.
    "opus-4.7": "gpt-5.5",
    "opus-4.6": "gpt-5.5",
    "sonnet-4.6": "gpt-5.4-mini",
    "sonnet-4.5": "gpt-5.4-mini",
    "haiku-4.5": "gpt-5.4-mini",
    "haiku-3.5": "gpt-5.4-mini",
}

_KIMI_MODEL_MAP = {
    # Kimi Code catalog aliases (not raw API model IDs).
    "opus-4.7": "kimi-code/k3",
    "opus-4.6": "kimi-code/k3",
    "sonnet-4.6": "kimi-code/kimi-for-coding",
    "sonnet-4.5": "kimi-code/kimi-for-coding",
    "haiku-4.5": "kimi-code/kimi-for-coding",
    "haiku-3.5": "kimi-code/kimi-for-coding",
}


# Default model used by every agent unless overridden. Set the CORESMITH_MODEL
# environment variable (to either a short name above or a full Claude CLI
# model ID) to override at runtime without code changes -- useful when the
# default version is unavailable on a fresh CLI install.
DEFAULT_MODEL = "opus-4.7"
DEFAULT_CODEX_MODEL = "gpt-5.5"
DEFAULT_KIMI_MODEL = "kimi-code/k3"

# Cheaper model for per-block agents (uarch, rtl, testbench, diagnose, lint
# fix, tb fix).  Integration and review agents still call DEFAULT_MODEL.
# Override with CORESMITH_BLOCK_MODEL env var.
BLOCK_MODEL = "sonnet-4.6"


def _resolve_model(model: str, provider: str = "claude_cli") -> str:
    """Map short model name to the selected CLI model ID.

    Honours the ``CORESMITH_MODEL`` environment variable as a runtime
    override: if set, it wins over whatever the caller passed in. Empty
    or unset model strings fall back to ``DEFAULT_MODEL``.
    """
    if provider == "codex_cli":
        env_override = (
            os.environ.get("CORESMITH_CODEX_MODEL", "").strip()
            or os.environ.get("CORESMITH_MODEL", "").strip()
        )
        if env_override:
            return _CODEX_MODEL_MAP.get(env_override, env_override)
        if not model:
            return DEFAULT_CODEX_MODEL
        return _CODEX_MODEL_MAP.get(model, model)


    if provider == "kimi_cli":
        # The KIMI_MODEL_* provider is exposed under a reserved runtime alias.
        if os.environ.get("KIMI_MODEL_NAME", "").strip():
            return "__kimi_env_model__"

        env_override = (
            os.environ.get("CORESMITH_KIMI_MODEL", "").strip()
            or os.environ.get("CORESMITH_MODEL", "").strip()
        )
        if env_override:
            return _KIMI_MODEL_MAP.get(env_override, env_override)
        if not model:
            return DEFAULT_KIMI_MODEL
        return _KIMI_MODEL_MAP.get(model, model)

    env_override = os.environ.get("CORESMITH_MODEL", "").strip()
    if env_override:
        model = env_override
    elif not model:
        model = DEFAULT_MODEL
    return _CLI_MODEL_MAP.get(model, model)


def block_model() -> str:
    """Return the model to use for per-block agents.

    Defaults to ``BLOCK_MODEL`` (Sonnet) but can be overridden with the
    ``CORESMITH_BLOCK_MODEL`` env var.  Used by uarch/rtl/testbench/diagnose
    agents so the bulk of a run goes through Sonnet, with Opus reserved
    for the chip-level integration step.
    """
    return os.environ.get("CORESMITH_BLOCK_MODEL", "").strip() or BLOCK_MODEL


def _detect_provider() -> str:
    """Detect which LLM provider to use.

    Defaults to Claude CLI.  Set ``CORESMITH_LLM_PROVIDER=codex`` (or
    ``codex_cli``) to route calls through ``codex exec``.
    """
    provider = os.environ.get("CORESMITH_LLM_PROVIDER", "").strip().lower()
    if provider in {"codex", "codex_cli"}:
        return "codex_cli"
    if provider in {"kimi", "kimi_cli"}:
        return "kimi_cli"
    if provider in {"claude", "claude_cli", ""}:
        return "claude_cli"
    raise ValueError(
        "Unsupported CORESMITH_LLM_PROVIDER={!r}. Use 'claude', 'codex', or 'kimi'.".format(provider)
    )
    return "claude_cli"


# ---------------------------------------------------------------------------
# Claude CLI helpers
# ---------------------------------------------------------------------------

def _find_claude_binary() -> str:
    """Locate the Claude CLI binary.

    Searches (in order):
      1. CLAUDE_CLI_PATH environment variable
      2. ``claude`` on $PATH  (``shutil.which``)
      3. Common install locations (~/.local/bin, ~/.npm/bin, /usr/local/bin)

    Raises ``FileNotFoundError`` if nothing is found.
    """
    env_path = os.environ.get("CLAUDE_CLI_PATH", "")
    if env_path and os.path.isfile(env_path) and os.access(env_path, os.X_OK):
        return env_path

    which_path = shutil.which("claude")
    if which_path:
        return which_path

    candidates = [
        os.path.expanduser("~/.local/bin/claude"),
        os.path.expanduser("~/.npm/bin/claude"),
        os.path.expanduser("~/.claude/local/claude"),
    ]
    for path in candidates:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path

    raise FileNotFoundError(
        "Claude CLI not found. Install it with: npm install -g @anthropic-ai/claude-code\n"
        "Or set CLAUDE_CLI_PATH to the binary location."
    )


def _find_codex_binary() -> str:
    """Locate the Codex CLI binary."""
    env_path = os.environ.get("CODEX_CLI_PATH", "")
    if env_path and os.path.isfile(env_path) and os.access(env_path, os.X_OK):
        return env_path

    which_path = shutil.which("codex")
    if which_path:
        return which_path

    candidates = [
        os.path.expanduser("~/.local/bin/codex"),
        os.path.expanduser("~/.npm/bin/codex"),
    ]
    for path in candidates:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path

    raise FileNotFoundError(
        "Codex CLI not found. Install Codex CLI or set CODEX_CLI_PATH to the binary location."
    )



def _find_kimi_binary() -> str:
    """Locate the current Kimi Code CLI binary."""
    env_path = os.environ.get("KIMI_CLI_PATH", "")
    if env_path and os.path.isfile(env_path) and os.access(env_path, os.X_OK):
        return env_path

    which_path = shutil.which("kimi")
    if which_path:
        return which_path

    candidates = [
        os.path.expanduser("~/.local/bin/kimi"),
        os.path.expanduser("~/.npm-global/bin/kimi"),
        os.path.expanduser("~/.npm/bin/kimi"),
    ]
    for path in candidates:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path

    raise FileNotFoundError(
        "Kimi Code CLI not found. Install it with: "
        "npm install -g @moonshot-ai/kimi-code\n"
        "Or set KIMI_CLI_PATH to the binary location."
    )


# ---------------------------------------------------------------------------
# ClaudeLLM -- plain Python class (no LangChain)
# ---------------------------------------------------------------------------

class ClaudeLLM:
    """LLM client backed by the Claude Code CLI.

    Simple interface: ``text = await llm.call(system="...", prompt="...")``

    Shells out to ``claude -p`` for each invocation.  Includes a circuit
    breaker that opens after 3 consecutive failures and auto-resets
    after 60 seconds.

    Usage::

        llm = ClaudeLLM(model=DEFAULT_MODEL, timeout=180)
        text = await llm.call(system="You are ...", prompt="Generate ...")

    The ``model`` argument may be left empty to fall back to ``DEFAULT_MODEL``,
    and the ``CORESMITH_MODEL`` env var overrides both at call time.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        claude_path: str = "",
        kimi_path: str = "",
        codex_path: str = "",
        timeout: int = 1200,
        max_turns: int = 50,
        disable_tools: bool = False,
    ) -> None:
        self.model = model or DEFAULT_MODEL
        self.claude_path = claude_path
        self.kimi_path = kimi_path
        self.codex_path = codex_path
        self.timeout = timeout
        self.max_turns = max_turns
        self.disable_tools = disable_tools

        self._provider = _detect_provider()
        logger.info("ClaudeLLM using %s provider.", self._provider)
        if self._provider == "claude_cli" and not self.claude_path:
            self.claude_path = os.environ.get("CLAUDE_CLI_PATH", "")
            if not self.claude_path:
                # Let FileNotFoundError propagate -- we'd rather crash
                # `__init__` clearly than leave self.claude_path empty
                # and Popen with cmd[0]="" later, which surfaces as the
                # opaque ``PermissionError: [Errno 13] Permission denied: ''``.
                self.claude_path = _find_claude_binary()
                logger.info(f"Found Claude CLI at: {self.claude_path}")
        elif self._provider == "codex_cli" and not self.codex_path:
            self.codex_path = os.environ.get("CODEX_CLI_PATH", "")
            if not self.codex_path:
                self.codex_path = _find_codex_binary()
                logger.info("Found Codex CLI at: %s", self.codex_path)
        elif self._provider == "kimi_cli" and not self.kimi_path:
            self.kimi_path = _find_kimi_binary()
            logger.info("Found Kimi Code CLI at: %s", self.kimi_path)

    async def call(
        self,
        system: str = "",
        prompt: str = "",
        run_name: str = "",
    ) -> str:
        """Call the Claude CLI and return the response text.

        Args:
            system: System prompt text.
            prompt: User/human prompt text.
            run_name: Label for telemetry events (replaces LangChain config.run_name).

        Returns:
            Response text from the LLM.
        """
        _get_breaker(_breaker_context.get("")).check()

        # Write llm_start event
        project_root = _llm_log_root()
        self._write_llm_event(project_root, "llm_start", {
            "model": _resolve_model(self.model, self._provider),
            "provider": self._provider,
            "run_name": run_name,
            "prompt_chars": len(prompt),
            "system_chars": len(system),
        })

        try:
            loop = asyncio.get_running_loop()
            text = await loop.run_in_executor(
                None, self._generate_via_cli, system, prompt,
            )
            _get_breaker(_breaker_context.get("")).record_success()

            # Write llm_end event
            self._write_llm_event(project_root, "llm_end", {
                "model": _resolve_model(self.model, self._provider),
                "provider": self._provider,
                "run_name": run_name,
                "output_chars": len(text),
            })

            return text
        except CircuitBreakerOpen:
            raise
        except Exception as e:
            _get_breaker(_breaker_context.get("")).record_failure()

            # Write llm_error event
            self._write_llm_event(project_root, "llm_error", {
                "model": _resolve_model(self.model, self._provider),
                "provider": self._provider,
                "run_name": run_name,
                "error": str(e)[:500],
            })

            raise

    # ------------------------------------------------------------------
    # Claude CLI path
    # ------------------------------------------------------------------

    # Stall detection: if no new stdout/stderr output for this many seconds
    # the process is likely hung on a permission prompt or similar.
    # Set to 1200s (20 min) for slower specialist calls (clock tree, memory map, etc.)
    # Scaled by CORESMITH_TIMEOUT_MULTIPLIER (default 1.0) so slow local models
    # like Qwen 3.6 27B (~25 tok/s on a single RTX PRO 6000) can finish
    # their 25-30K-token reasoning + write trajectories without being killed.
    _STALL_THRESHOLD_S: int = 1200  # scaled by scaled() at use sites; see orchestrator._timeouts
    # How often to poll the subprocess and emit heartbeat events.
    _POLL_INTERVAL_S: float = 2.0
    # Heartbeat events are written every N poll cycles (to avoid log spam).
    _HEARTBEAT_EVERY_N: int = 15  # ~30s at 2s poll

    def _generate_via_cli(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        if self._provider == "codex_cli":
            return self._generate_via_codex_cli(system_prompt, user_prompt)

        if self._provider == "kimi_cli":
            return self._generate_via_kimi_cli(system_prompt, user_prompt)
        return self._generate_via_claude_cli(system_prompt, user_prompt)

    def _generate_via_claude_cli(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Call the Claude CLI (``claude -p``) synchronously.

        Uses ``Popen`` with a polling watchdog instead of blocking
        ``subprocess.run()``.  This enables:
          - **Stall detection**: kills the process if no output arrives for
            ``_STALL_THRESHOLD_S`` seconds (catches permission prompts).
          - **Liveness heartbeats**: writes periodic events to the pipeline
            event log so the outer agent can see progress.
          - **Process registry**: the ``Popen`` handle is registered in
            ``_active_processes`` so ``kill_active_cli_processes()`` can
            force-terminate it from ``pause_pipeline()``.
          - **Partial output capture**: on timeout or stall, any partial
            stdout/stderr is included in the error message for diagnosis.

        System messages are passed via ``--system-prompt``.
        Human/AI messages are concatenated and piped to stdin.
        """
        resolved_model = _resolve_model(self.model, self._provider)

        cmd: list[str] = [
            self.claude_path,
            "-p",                                # print mode (non-interactive)
            "--output-format", "stream-json",    # JSONL with per-event usage + cost
            "--verbose",                         # required by CLI for stream-json under --print
            "--model", resolved_model,
            "--max-turns", str(self.max_turns),
            "--permission-mode", "auto",  # headless: auto-approve tool use, no prompts
        ]

        if self.disable_tools:
            cmd.extend([
                "--disallowedTools",
                "Bash,Read,Write,Edit,Glob,Grep,WebFetch,WebSearch,"
                "Task,NotebookEdit,EnterPlanMode",
            ])

        if system_prompt:
            cmd.extend(["--system-prompt", system_prompt])

        logger.debug(
            f"Claude CLI invocation: model={resolved_model}, "
            f"prompt_len={len(user_prompt)}, system_len={len(system_prompt)}"
        )
        logger.info(f"Claude CLI cmd (first 200): {' '.join(cmd)[:200]}")

        project_root = _llm_log_root()

        t0 = _time_mod.monotonic()
        span_start_ns = _time_mod.time_ns()
        max_retries = 3

        # ``_generate_via_cli`` is itself dispatched via
        # ``loop.run_in_executor`` in ``call()``, so this whole function
        # already runs off the asyncio event loop.  Each LangGraph
        # ``Send()`` fan-out lands in its own executor thread, which is
        # what makes per-tier block fan-out actually concurrent.
        usage: dict = {}
        for attempt in range(max_retries):
            try:
                output, stderr_text, returncode, elapsed, timed_out, stalled, usage = (
                    self._run_cli_with_watchdog(
                        cmd, user_prompt, project_root, resolved_model, t0,
                    )
                )
            except FileNotFoundError:
                elapsed = _time_mod.monotonic() - t0
                logger.error("Claude CLI binary not found")
                error_msg = "claude CLI binary not found"
                output = (
                    "[ClaudeLLM error: claude CLI binary not found. "
                    "Install: npm install -g @anthropic-ai/claude-code]"
                )
                _log_llm_call(
                    model=resolved_model,
                    provider="claude_cli",
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    response=output,
                    duration_s=elapsed,
                    timeout=self.timeout,
                    error=error_msg,
                    start_ts_ns=span_start_ns,
                )
                return output

            # --- Handle timeout / stall with full diagnostic output ---
            if timed_out or stalled:
                reason = "stalled" if stalled else "timed out"
                error_msg = f"claude CLI {reason} after {elapsed:.0f}s"
                if stderr_text:
                    error_msg += f" | stderr: {stderr_text[:300]}"
                if output:
                    error_msg += f" | partial stdout: {output[:300]}"
                full_output = f"[ClaudeLLM error: {error_msg}]"
                logger.error("Claude CLI %s: %s", reason, error_msg)
                _log_llm_call(
                    model=resolved_model,
                    provider="claude_cli",
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    response=full_output,
                    duration_s=elapsed,
                    timeout=self.timeout,
                    error=error_msg,
                    timed_out=True,
                    usage=usage,
                    start_ts_ns=span_start_ns,
                )
                self._write_llm_event(project_root, f"llm_{reason}", {
                    "model": resolved_model,
                    "elapsed_s": round(elapsed, 1),
                    "partial_stdout_len": len(output),
                    "stderr": stderr_text[:500],
                    "partial_stdout": output[:500],
                })
                return full_output

            # --- Normal completion ---
            logger.debug(
                f"Claude CLI attempt={attempt+1}, retcode={returncode}, "
                f"stdout_len={len(output)}, first100={output[:100]!r}"
            )

            if output.startswith("Error: Reached max turns") and attempt < max_retries - 1:
                wait = 5 * (attempt + 1)
                logger.warning(
                    f"Claude CLI returned '{output}', retrying in {wait}s "
                    f"(attempt {attempt+1}/{max_retries})"
                )
                _time_mod.sleep(wait)
                t0 = _time_mod.monotonic()
                span_start_ns = _time_mod.time_ns()
                continue
            break

        elapsed = _time_mod.monotonic() - t0

        logger.info(
            f"Claude CLI output: retcode={returncode}, "
            f"stdout_len={len(output)}, first100={output[:100]!r}"
        )
        if returncode != 0:
            logger.warning(
                f"Claude CLI exited with code {returncode}: {stderr_text[:500]}"
            )

        if not output:
            error_msg = (
                f"[ClaudeLLM error: claude CLI returned empty response. "
                f"exit_code={returncode}, stderr: {stderr_text[:500]}]"
            )
            output = error_msg
            logger.error(f"LLM empty response: {error_msg}")
            self._write_llm_event(project_root, "llm_empty_response", {
                "model": resolved_model,
                "provider": "claude_cli",
                "exit_code": returncode,
                "stderr": stderr_text[:500],
                "error": error_msg[:300],
            })

        _log_llm_call(
            model=resolved_model,
            provider="claude_cli",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response=output,
            duration_s=elapsed,
            timeout=self.timeout,
            usage=usage,
            start_ts_ns=span_start_ns,
        )

        return output

    def _generate_via_codex_cli(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Call the Codex CLI (``codex exec``) synchronously."""
        resolved_model = _resolve_model(self.model, self._provider)

        workdir = (
            os.environ.get("CORESMITH_CODEX_WORKDIR", "").strip()
            or os.environ.get("CORESMITH_PROJECT_ROOT", "").strip()
            or _default_project_root()
        )
        isolate_workdir = os.environ.get("CORESMITH_CODEX_ISOLATE_WORKDIR", "1").strip().lower()
        if isolate_workdir not in {"0", "false", "no", "off"}:
            Path(workdir).mkdir(parents=True, exist_ok=True)
            workdir = tempfile.mkdtemp(prefix="codex-call-", dir=workdir)
        log_root = _llm_log_root()
        sandbox = os.environ.get("CORESMITH_CODEX_SANDBOX", "workspace-write").strip()

        cmd: list[str] = [
            self.codex_path,
            "exec",
            "--json",
            "--dangerously-bypass-approvals-and-sandbox",
            "--sandbox", sandbox,
            "--skip-git-repo-check",
            "-C", workdir,
            "-m", resolved_model,
            "-",
        ]

        combined_prompt = self._build_codex_prompt(system_prompt, user_prompt)

        logger.debug(
            "Codex CLI invocation: model=%s, prompt_len=%d, system_len=%d",
            resolved_model,
            len(user_prompt),
            len(system_prompt),
        )
        logger.info("Codex CLI cmd (first 200): %s", " ".join(cmd)[:200])

        t0 = _time_mod.monotonic()
        span_start_ns = _time_mod.time_ns()
        try:
            output, stderr_text, returncode, elapsed, timed_out, stalled, usage = (
                    self._run_cli_with_watchdog(
                        cmd, combined_prompt, log_root, resolved_model, t0,
                    )
            )
        except FileNotFoundError:
            elapsed = _time_mod.monotonic() - t0
            error_msg = "codex CLI binary not found"
            output = (
                "[ClaudeLLM error: codex CLI binary not found. "
                "Install Codex CLI or set CODEX_CLI_PATH]"
            )
            _log_llm_call(
                model=resolved_model,
                provider="codex_cli",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response=output,
                duration_s=elapsed,
                timeout=self.timeout,
                error=error_msg,
                start_ts_ns=span_start_ns,
            )
            return output

        if timed_out or stalled:
            reason = "stalled" if stalled else "timed out"
            error_msg = f"codex CLI {reason} after {elapsed:.0f}s"
            if stderr_text:
                error_msg += f" | stderr: {stderr_text[:300]}"
            if output:
                error_msg += f" | partial stdout: {output[:300]}"
            full_output = f"[ClaudeLLM error: {error_msg}]"
            logger.error("Codex CLI %s: %s", reason, error_msg)
            _log_llm_call(
                model=resolved_model,
                provider="codex_cli",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response=full_output,
                duration_s=elapsed,
                timeout=self.timeout,
                error=error_msg,
                timed_out=True,
                usage=usage,
                start_ts_ns=span_start_ns,
            )
            self._write_llm_event(log_root, f"llm_{reason}", {
                "model": resolved_model,
                "provider": "codex_cli",
                "elapsed_s": round(elapsed, 1),
                "partial_stdout_len": len(output),
                "stderr": stderr_text[:500],
                "partial_stdout": output[:500],
            })
            return full_output

        elapsed = _time_mod.monotonic() - t0
        logger.info(
            "Codex CLI output: retcode=%s, stdout_len=%d, first100=%r",
            returncode,
            len(output),
            output[:100],
        )
        if returncode != 0:
            logger.warning("Codex CLI exited with code %s: %s", returncode, stderr_text[:500])

        if not output:
            error_msg = (
                f"[ClaudeLLM error: codex CLI returned empty response. "
                f"exit_code={returncode}, stderr: {stderr_text[:500]}]"
            )
            output = error_msg
            logger.error("LLM empty response: %s", error_msg)
            self._write_llm_event(log_root, "llm_empty_response", {
                "model": resolved_model,
                "provider": "codex_cli",
                "exit_code": returncode,
                "stderr": stderr_text[:500],
                "error": error_msg[:300],
            })

        _log_llm_call(
            model=resolved_model,
            provider="codex_cli",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response=output,
            duration_s=elapsed,
            timeout=self.timeout,
            usage=usage,
            start_ts_ns=span_start_ns,
        )

        return output
    def _generate_via_kimi_cli(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Call Kimi Code through its stdin/stdout ACP server.

        ACP keeps large prompts off the process argument list and provides a
        stable JSON-RPC stream for assistant text, usage, and tool permission
        requests. Authentication remains owned by the installed ``kimi`` CLI.
        """
        resolved_model = _resolve_model(self.model, self._provider)
        combined_prompt = self._build_kimi_prompt(system_prompt, user_prompt)
        project_root = _llm_log_root()
        workdir = (
            os.environ.get("CORESMITH_KIMI_WORKDIR", "").strip()
            or os.environ.get("CORESMITH_PROJECT_ROOT", "").strip()
            or _default_project_root()
        )
        Path(workdir).mkdir(parents=True, exist_ok=True)

        t0 = _time_mod.monotonic()
        span_start_ns = _time_mod.time_ns()
        try:
            output, stderr_text, returncode, elapsed, timed_out, stalled, usage = (
                self._run_kimi_acp_with_watchdog(
                    combined_prompt, workdir, project_root, resolved_model, t0,
                )
            )
        except FileNotFoundError:
            elapsed = _time_mod.monotonic() - t0
            error_msg = "kimi CLI binary not found"
            output = (
                "[ClaudeLLM error: Kimi Code CLI binary not found. Install: "
                "npm install -g @moonshot-ai/kimi-code]"
            )
            _log_llm_call(
                model=resolved_model,
                provider="kimi_cli",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response=output,
                duration_s=elapsed,
                timeout=self.timeout,
                error=error_msg,
                start_ts_ns=span_start_ns,
            )
            return output

        if timed_out or stalled:
            reason = "stalled" if stalled else "timed out"
            error_msg = f"kimi CLI {reason} after {elapsed:.0f}s"
            if stderr_text:
                error_msg += f" | stderr: {stderr_text[:300]}"
            if output:
                error_msg += f" | partial stdout: {output[:300]}"
            output = f"[ClaudeLLM error: {error_msg}]"
            _log_llm_call(
                model=resolved_model,
                provider="kimi_cli",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response=output,
                duration_s=elapsed,
                timeout=self.timeout,
                error=error_msg,
                timed_out=True,
                usage=usage,
                start_ts_ns=span_start_ns,
            )
            return output

        if not output:
            detail = stderr_text[:500] or f"ACP exited with code {returncode}"
            output = f"[ClaudeLLM error: kimi CLI returned empty response. {detail}]"

        _log_llm_call(
            model=resolved_model,
            provider="kimi_cli",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response=output,
            duration_s=elapsed,
            timeout=self.timeout,
            error=stderr_text if returncode != 0 else "",
            usage=usage,
            start_ts_ns=span_start_ns,
        )
        return output

    @staticmethod
    def _build_kimi_prompt(system_prompt: str, user_prompt: str) -> str:
        """Build a single ACP text block while preserving prompt roles."""
        if not system_prompt:
            return user_prompt
        return (
            "<system>\n"
            f"{system_prompt}\n"
            "</system>\n\n"
            "<user>\n"
            f"{user_prompt}\n"
            "</user>\n"
        )

    def _run_kimi_acp_with_watchdog(
        self,
        prompt: str,
        workdir: str,
        project_root: str,
        resolved_model: str,
        t0: float,
    ) -> tuple[str, str, int, float, bool, bool, dict]:
        """Run one Kimi ACP session and capture its JSON-RPC transcript."""
        process = subprocess.Popen(
            [self.kimi_path, "acp"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=workdir,
            bufsize=1,
        )
        _register_process(process)
        self._write_llm_event(project_root, "llm_call_start", {
            "model": resolved_model,
            "provider": "kimi_cli",
            "timeout_s": self.timeout,
            "prompt_len": len(prompt),
            "pid": process.pid,
        })

        messages: queue.Queue[str] = queue.Queue()
        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []
        last_activity = _time_mod.monotonic()
        timed_out = False
        stalled = False
        complete = False
        protocol_error = ""
        session_id = ""
        prompt_request_id = 4

        def _read_stdout() -> None:
            nonlocal last_activity
            try:
                for line in process.stdout:
                    stdout_chunks.append(line)
                    messages.put(line)
                    last_activity = _time_mod.monotonic()
            except (ValueError, OSError):
                pass

        def _read_stderr() -> None:
            nonlocal last_activity
            try:
                for line in process.stderr:
                    stderr_chunks.append(line)
                    last_activity = _time_mod.monotonic()
            except (ValueError, OSError):
                pass

        def _send(payload: dict) -> None:
            raw = _json.dumps(payload, separators=(",", ":")) + "\n"
            process.stdin.write(raw)
            process.stdin.flush()

        def _request(request_id: int, method: str, params: dict) -> None:
            _send({
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params,
            })

        t_out = threading.Thread(target=_read_stdout, daemon=True)
        t_err = threading.Thread(target=_read_stderr, daemon=True)
        t_out.start()
        t_err.start()

        try:
            _request(1, "initialize", {
                "protocolVersion": 1,
                "clientCapabilities": {},
                "clientInfo": {"name": "coresmith", "version": "0.1"},
            })
            poll_count = 0
            while not complete:
                elapsed = _time_mod.monotonic() - t0
                if elapsed > self.timeout:
                    timed_out = True
                    break
                if _time_mod.monotonic() - last_activity > scaled(self._STALL_THRESHOLD_S):
                    stalled = True
                    break
                if process.poll() is not None and messages.empty():
                    protocol_error = protocol_error or "Kimi ACP process exited before completing the prompt"
                    break

                try:
                    raw = messages.get(timeout=self._POLL_INTERVAL_S)
                except queue.Empty:
                    poll_count += 1
                    if poll_count % self._HEARTBEAT_EVERY_N == 0:
                        self._write_llm_event(project_root, "llm_call_heartbeat", {
                            "model": resolved_model,
                            "provider": "kimi_cli",
                            "elapsed_s": round(elapsed, 1),
                            "stdout_bytes": sum(len(c) for c in stdout_chunks),
                            "stderr_bytes": sum(len(c) for c in stderr_chunks),
                            "pid": process.pid,
                        })
                    continue

                try:
                    message = _json.loads(raw)
                except _json.JSONDecodeError:
                    continue

                # ACP servers may ask the client to approve a tool call. CoreSmith
                # is intentionally non-interactive, so choose a deterministic
                # one-shot answer rather than hanging on a terminal prompt.
                if message.get("method") == "session/request_permission":
                    params = message.get("params") or {}
                    options = params.get("options") or []
                    wanted = (
                        ("reject_once", "reject_always")
                        if self.disable_tools
                        else ("allow_once", "allow_always")
                    )
                    selected = next(
                        (o for kind in wanted for o in options if o.get("kind") == kind),
                        None,
                    )
                    outcome = (
                        {"outcome": "selected", "optionId": selected["optionId"]}
                        if selected
                        else {"outcome": "cancelled"}
                    )
                    _send({"jsonrpc": "2.0", "id": message.get("id"), "result": {"outcome": outcome}})
                    continue

                response_id = message.get("id")
                if response_id not in {1, 2, 3, prompt_request_id}:
                    continue
                if message.get("error"):
                    error = message["error"]
                    protocol_error = str(error.get("message") or error)
                    break

                if response_id == 1:
                    _request(2, "session/new", {"cwd": workdir, "mcpServers": []})
                elif response_id == 2:
                    session_id = (message.get("result") or {}).get("sessionId", "")
                    if not session_id:
                        protocol_error = "Kimi ACP session/new returned no sessionId"
                        break
                    _request(3, "session/set_config_option", {
                        "sessionId": session_id,
                        "configId": "model",
                        "value": resolved_model,
                    })
                elif response_id == 3:
                    effective_prompt = prompt
                    if self.disable_tools:
                        effective_prompt = (
                            "Do not use tools or modify files. Answer using only the supplied context.\n\n"
                            + prompt
                        )
                    _request(prompt_request_id, "session/prompt", {
                        "sessionId": session_id,
                        "prompt": [{"type": "text", "text": effective_prompt}],
                    })
                elif response_id == prompt_request_id:
                    complete = True
        except (BrokenPipeError, OSError) as exc:
            protocol_error = f"Kimi ACP transport error: {exc}"
        finally:
            if (timed_out or stalled) and session_id and process.poll() is None:
                try:
                    _request(5, "session/cancel", {"sessionId": session_id})
                except (BrokenPipeError, OSError):
                    pass
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            t_out.join(timeout=5)
            t_err.join(timeout=5)
            _unregister_process()

        elapsed = _time_mod.monotonic() - t0
        stdout_text = "".join(stdout_chunks).strip()
        stderr_text = "".join(stderr_chunks).strip()
        if protocol_error:
            stderr_text = f"{protocol_error}\n{stderr_text}".strip()
        response_text, usage = _parse_kimi_acp_json(stdout_text)
        return (
            response_text,
            stderr_text,
            0 if complete else (
                process.returncode if process.returncode is not None else -1
            ),
            elapsed,
            timed_out,
            stalled,
            usage,
        )


    @staticmethod
    def _build_codex_prompt(system_prompt: str, user_prompt: str) -> str:
        if not system_prompt:
            return user_prompt
        return (
            "<system>\n"
            f"{system_prompt}\n"
            "</system>\n\n"
            "<user>\n"
            f"{user_prompt}\n"
            "</user>\n"
        )

    # ------------------------------------------------------------------
    # Popen + watchdog internals
    # ------------------------------------------------------------------

    # How often (in poll cycles) to update the live streaming file.
    _STREAM_UPDATE_EVERY_N: int = 1  # every poll (~2s)

    def _run_cli_with_watchdog(
        self,
        cmd: list[str],
        user_prompt: str,
        project_root: str,
        resolved_model: str,
        t0: float,
    ) -> tuple[str, str, int, float, bool, bool, dict]:
        """Run the CLI via ``Popen`` with stall detection and heartbeats.

        Returns ``(response_text, stderr, returncode, elapsed, timed_out,
        stalled, usage)`` where ``response_text`` is the final model
        response extracted from the stream-json ``result`` event (or a
        concatenation of partial assistant-text chunks if the stream was
        cut short), and ``usage`` is the per-call token/cost dict (may be
        empty on failure).
        Raises ``FileNotFoundError`` if the binary is missing.
        """
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        _register_process(process)

        self._write_llm_event(project_root, "llm_call_start", {
            "model": resolved_model,
            "timeout_s": self.timeout,
            "prompt_len": len(user_prompt),
            "pid": process.pid,
        })

        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []
        last_activity = _time_mod.monotonic()
        timed_out = False
        stalled = False

        def _read_stream(stream, chunks: list[str]) -> None:
            """Read lines from a stream, updating last_activity timestamp."""
            nonlocal last_activity
            try:
                for line in stream:
                    chunks.append(line)
                    last_activity = _time_mod.monotonic()
            except (ValueError, OSError):
                pass  # stream closed

        # Write prompt to stdin and close it immediately
        try:
            process.stdin.write(user_prompt)
            process.stdin.close()
        except (BrokenPipeError, OSError):
            pass

        # Start reader threads for stdout and stderr
        t_out = threading.Thread(target=_read_stream, args=(process.stdout, stdout_chunks), daemon=True)
        t_err = threading.Thread(target=_read_stream, args=(process.stderr, stderr_chunks), daemon=True)
        t_out.start()
        t_err.start()

        # Set up live streaming trajectory file for realtime webview updates
        live_dir = Path(project_root) / ".coresmith" / "live_streams"
        live_dir.mkdir(parents=True, exist_ok=True)
        stream_path = live_dir / f"{process.pid}.json"
        wall_start = _time_mod.time()  # wall clock for event correlation

        # Write initial stream file immediately so the webview can detect
        # a streaming call before the first poll cycle completes.
        try:
            init_data = _json.dumps({
                "pid": process.pid,
                "model": resolved_model,
                "started_ts": wall_start,
                "elapsed_s": 0,
                "partial_stdout": "",
                "stdout_bytes": 0,
                "stderr_bytes": 0,
            }, default=str)
            tmp = stream_path.with_suffix(".tmp")
            tmp.write_text(init_data, encoding="utf-8")
            tmp.rename(stream_path)
        except Exception:
            pass

        try:
            poll_count = 0
            while process.poll() is None:
                _time_mod.sleep(self._POLL_INTERVAL_S)
                poll_count += 1
                elapsed = _time_mod.monotonic() - t0

                # Hard timeout
                if elapsed > self.timeout:
                    logger.error(
                        "Claude CLI hard timeout after %.0fs, killing pid=%d",
                        elapsed, process.pid,
                    )
                    process.kill()
                    timed_out = True
                    break

                # Stall detection (no output activity)
                stall_time = _time_mod.monotonic() - last_activity
                if stall_time > scaled(self._STALL_THRESHOLD_S):
                    partial = "".join(stderr_chunks + stdout_chunks)
                    logger.error(
                        "Claude CLI stalled for %.0fs with no output (pid=%d). "
                        "Likely hung on interactive prompt. "
                        "Partial output: %s",
                        stall_time, process.pid, partial[:500],
                    )
                    process.kill()
                    stalled = True
                    break

                # Update live streaming trajectory file.
                # Always update elapsed_s so the webview shows a
                # progressing timer even while the model is still
                # processing the prompt (no output yet).
                if poll_count % self._STREAM_UPDATE_EVERY_N == 0:
                    len(stdout_chunks)
                    try:
                        chunks_snap = list(stdout_chunks)
                        stream_data = _json.dumps({
                            "pid": process.pid,
                            "model": resolved_model,
                            "started_ts": wall_start,
                            "elapsed_s": round(elapsed, 1),
                            "partial_stdout": "".join(chunks_snap),
                            "stdout_bytes": sum(len(c) for c in chunks_snap),
                            "stderr_bytes": sum(len(c) for c in stderr_chunks),
                        }, default=str)
                        tmp = stream_path.with_suffix(".tmp")
                        tmp.write_text(stream_data, encoding="utf-8")
                        tmp.rename(stream_path)
                    except Exception:
                        pass

                # Periodic heartbeat event
                if poll_count % self._HEARTBEAT_EVERY_N == 0:
                    self._write_llm_event(project_root, "llm_call_heartbeat", {
                        "model": resolved_model,
                        "elapsed_s": round(elapsed, 1),
                        "stdout_bytes": sum(len(c) for c in stdout_chunks),
                        "stderr_bytes": sum(len(c) for c in stderr_chunks),
                        "pid": process.pid,
                    })
        finally:
            # Wait for reader threads to finish draining pipes
            t_out.join(timeout=5)
            t_err.join(timeout=5)
            _unregister_process()
            # Mark streaming trajectory file as done (with final output)
            # instead of deleting it immediately.  This avoids a data gap
            # between file deletion and llm_calls.jsonl write -- the
            # webview serve.py will clean up stale done files.
            try:
                final_elapsed = _time_mod.monotonic() - t0
                done_data = _json.dumps({
                    "pid": process.pid,
                    "model": resolved_model,
                    "started_ts": wall_start,
                    "elapsed_s": round(final_elapsed, 1),
                    "partial_stdout": "".join(stdout_chunks),
                    "stdout_bytes": sum(len(c) for c in stdout_chunks),
                    "stderr_bytes": sum(len(c) for c in stderr_chunks),
                    "done": True,
                    "done_ts": _time_mod.time(),
                }, default=str)
                tmp = stream_path.with_suffix(".tmp")
                tmp.write_text(done_data, encoding="utf-8")
                tmp.rename(stream_path)
            except Exception:
                try:
                    stream_path.unlink(missing_ok=True)
                except Exception:
                    pass

        elapsed = _time_mod.monotonic() - t0
        stdout_text = "".join(stdout_chunks).strip()
        stderr_text = "".join(stderr_chunks).strip()
        returncode = process.returncode if process.returncode is not None else -1

        # Parse provider JSON output. On stall/timeout we fall back to
        # whatever assistant text leaked through.
        if self._provider == "codex_cli":
            response_text, usage = _parse_codex_json(stdout_text)
            # Persist every codex turn (reasoning, tool calls, tool
            # results, agent messages) so the trajectory viewer can show
            # the agent's actual decision-making, not just the final
            # answer. Keyed by pid so the webview can correlate to the
            # llm_start event that carries the human-readable run_name.
            try:
                _log_codex_turns(
                    stdout_text,
                    project_root,
                    process.pid if process.pid else 0,
                    wall_start,
                )
            except Exception:
                pass
        else:
            response_text, usage = _parse_stream_json(stdout_text)
        if not response_text:
            # Stream-json parsing produced nothing useful — surface raw
            # stdout (may be empty / a CLI error string) so downstream
            # error messages still have something to print.
            response_text = stdout_text

        return response_text, stderr_text, returncode, elapsed, timed_out, stalled, usage

    @staticmethod
    def _write_llm_event(project_root: str, event_type: str, data: dict) -> None:
        """Write a non-fatal event to the pipeline event log."""
        try:
            from orchestrator.langgraph.event_stream import write_graph_event
            write_graph_event(project_root, "LLM", event_type, data)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Backward compatibility aliases
# ---------------------------------------------------------------------------
ClaudeChatModel = ClaudeLLM
CursorChatModel = ClaudeLLM
