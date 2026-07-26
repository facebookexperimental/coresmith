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
import re as _re
import shutil
import signal
import subprocess
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


def _killpg_safe(pgid: int, sig: int) -> None:
    """os.killpg swallowing the benign 'group already gone' errors."""
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def _reap_process_group(
    process: subprocess.Popen, pgid: int, grace_s: float = 10.0
) -> None:
    """Terminate the child's process group so no grandchild survives the call.

    The CLI is launched with ``start_new_session=True``, so the child is its
    own session/group leader and ``pgid == child.pid`` at spawn. A grandchild
    the CLI spawned (e.g. a sim process) shares that pgid and inherits our
    stdout/stderr write-end; if it lives on after the CLI's final response, the
    reader threads block on the still-open pipe until the hard-timeout deadline
    (the observed ~45-min post-response exit stall).

    We SIGTERM the group (releasing grandchildren gracefully), reap the direct
    child within ``grace_s``, then SIGKILL any group survivors. Capturing
    ``pgid`` at spawn (rather than re-deriving via ``os.getpgid`` after the
    child may already be reaped) avoids the pid-reuse race.
    """
    # Graceful: let the whole group wind down. If only the (already-exited)
    # leader remains, this is a no-op ESRCH.
    _killpg_safe(pgid, signal.SIGTERM)
    # Make sure the direct child is reaped (poll() may already have done this;
    # wait() then returns immediately with the cached returncode).
    try:
        process.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except Exception:
            pass
        try:
            process.wait(timeout=2)
        except Exception:
            pass
    except Exception:
        pass
    # Hard-kill any grandchild that ignored SIGTERM and is still holding pipes.
    _killpg_safe(pgid, signal.SIGKILL)


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


def reap_active_cli_processes(grace_s: float = 10.0) -> int:
    """Reap the whole process GROUP of every active CLI subprocess.

    Unlike ``kill_active_cli_processes`` -- a shallow ``proc.kill()`` on the
    direct child only -- this runs the SAME process-group reap the watchdog
    finally-path uses (``_reap_process_group``): SIGTERM the child's group,
    grace-wait the direct child, then SIGKILL any group survivors. That is what
    a *pause* needs: an in-flight ``codex``/``claude`` call spawns tool/sim
    grandchildren that share the child's process group; a bare child kill
    orphans them and they keep running (observed live -- an orphaned codex kept
    burning tokens after a run pause). Reaping the group stops the whole tree.

    Each ``Popen`` was launched with ``start_new_session=True`` so it is its own
    session/group leader (``pgid == pid`` at spawn); we use that captured pid as
    the group id (never re-derive via ``os.getpgid`` -- pid-reuse race). The
    watchdog thread owning each child is blocked in the reader loop; the group
    reap makes its ``process.poll()`` return, so its own finally-path reap is a
    harmless idempotent no-op (ESRCH) and ``_unregister_process`` a no-op pop.

    Returns the number of live process groups reaped. Best-effort: a failure on
    one child never blocks reaping the rest or the pause itself.
    """
    # Snapshot + clear under the lock, then reap OUTSIDE it: the grace-wait can
    # take up to ``grace_s`` and must not hold the registry lock (that would
    # block the owning watchdog thread's _unregister_process / other callers).
    with _active_processes_lock:
        procs = list(_active_processes.items())
        _active_processes.clear()
    reaped = 0
    for tid, proc in procs:
        try:
            if proc.poll() is None:
                logger.warning(
                    "Reaping in-flight CLI process group pid=%d (thread %d) on pause",
                    proc.pid, tid,
                )
                _reap_process_group(proc, proc.pid, grace_s=grace_s)
                reaped += 1
        except Exception:
            logger.debug("reap of pid=%s failed", getattr(proc, "pid", "?"), exc_info=True)
    return reaped


# ---------------------------------------------------------------------------
# LLM call telemetry -- JSONL + OpenTelemetry
# ---------------------------------------------------------------------------

_LLM_LOG_RELPATH = ".coresmith/llm_calls.jsonl"
_TRUNCATE_ATTR = 32_000  # OTel attribute max (span attrs); JSONL is untruncated


# ---------------------------------------------------------------------------
# Call-site attribution (Package C: record/replay)
# ---------------------------------------------------------------------------
# The llm_calls.jsonl corpus historically had no call-site identifier -- it was
# joinable to a node/block/run only by timestamp. We stamp each record with the
# ``run_name`` the caller passed to ``call()``, a process-global monotonic
# ``call_index`` (so record/replay can key on ``(run_name, call_index)`` which
# survives prompt edits), and the current ``graph`` (from the breaker context).
#
# ``_call_site_context`` is a ContextVar so concurrent block fan-out tasks each
# see their own attribution. ``call()`` sets it, then propagates it into the
# ``run_in_executor`` worker via ``contextvars.copy_context().run`` (executor
# threads do NOT inherit contextvars automatically).
_call_site_context: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "llm_call_site", default=None
)

_call_index_lock = threading.Lock()
_call_index_counter = 0


def _next_call_index() -> int:
    """Process-global monotonic LLM call counter (thread-safe)."""
    global _call_index_counter
    with _call_index_lock:
        _call_index_counter += 1
        return _call_index_counter


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
    # Call-site attribution (empty when logged outside call() -- e.g. direct
    # _generate_via_cli unit tests). No signature change at the 5+ call sites:
    # the fields are pulled from the ContextVar call() set before dispatch.
    site = _call_site_context.get(None) or {}
    record = {
        "ts": ts,
        "iso": _time_mod.strftime("%Y-%m-%dT%H:%M:%S", _time_mod.localtime(ts)),
        "model": model,
        "provider": provider,
        "run_name": site.get("run_name", ""),
        "call_index": site.get("call_index"),
        "graph": site.get("graph", ""),
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
                "llm.run_name": site.get("run_name", ""),
                "llm.graph": site.get("graph", ""),
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
            if site.get("call_index") is not None:
                attrs["llm.call_index"] = site.get("call_index")
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

    The session/thread id is emitted as a ``thread.started`` event; we
    surface it in the returned ``usage`` dict as ``session_id`` so callers
    can later ``codex exec resume <session_id> "<prompt>"`` to continue the
    same conversation (fixes retry thrashing where a rebuild has no memory
    of prior attempts).
    """
    final_text = ""
    usage: dict = {}
    session_id: str = ""
    for raw in stdout.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = _json.loads(raw)
        except _json.JSONDecodeError:
            continue
        ev_type = obj.get("type")
        if ev_type == "thread.started":
            # codex 0.130.0 emits {"type":"thread.started","thread_id":"..."}.
            # Accept a few key spellings defensively.
            sid = (
                obj.get("thread_id")
                or obj.get("session_id")
                or obj.get("id")
                or ""
            )
            if sid:
                session_id = str(sid)
        elif ev_type == "item.completed":
            item = obj.get("item", {}) or {}
            if item.get("type") == "agent_message":
                final_text = item.get("text", "") or final_text
        elif ev_type == "turn.completed":
            usage = obj.get("usage") or usage
    if session_id:
        # usage may be the raw turn.completed dict; copy so we don't mutate
        # a shared object, and stamp the session id onto it.
        usage = dict(usage) if usage else {}
        usage["session_id"] = session_id
    return final_text, usage


def _parse_opencode_json(stdout: str) -> tuple[str, dict]:
    """Parse OpenCode ``run --format json`` NDJSON events."""
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
        ev_type = obj.get("type")
        if ev_type == "text":
            part = obj.get("part") or {}
            if part.get("type") == "text":
                chunks.append(part.get("text", "") or "")
        elif ev_type == "step_finish":
            tokens = (obj.get("part") or {}).get("tokens") or {}
            usage = {
                "input_tokens": tokens.get("input", 0),
                "output_tokens": tokens.get("output", 0),
                "total_tokens": tokens.get("total", 0),
                "cache_read_input_tokens": (tokens.get("cache") or {}).get("read", 0),
                "cache_creation_input_tokens": (tokens.get("cache") or {}).get("write", 0),
                "reasoning_output_tokens": tokens.get("reasoning", 0),
                "total_cost_usd": (obj.get("part") or {}).get("cost", 0),
            }
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


def _log_opencode_turns(stdout: str, project_root: str, pid: int, wall_start: float) -> int:
    """Append OpenCode NDJSON events to ``.coresmith/opencode_turns.jsonl``.

    ``opencode run --thinking --format json`` emits exposed reasoning, text,
    tool, and step events. Persist the complete valid-JSON event stream so a
    run can be audited or replayed without mixing reasoning into the final
    response returned to agents. Malformed lines and all logging failures are
    deliberately non-fatal.
    """
    try:
        log = Path(project_root) / ".coresmith" / "opencode_turns.jsonl"
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
    "opus-5":    "opus",                       # latest Claude Opus 5
    "sonnet-5":  "sonnet",                     # latest Claude Sonnet 5
    "opus-4.8":  "claude-opus-4-8",          # pinned previous Opus
    "opus-4.7":  "claude-opus-4-8",          # legacy alias -> current Opus
    "opus-4.6":  "claude-opus-4-8",          # legacy alias -> current Opus
    "sonnet-4.6": "claude-sonnet-4-6",
    "sonnet-4.5": "claude-sonnet-4-6",       # legacy alias -> current Sonnet
    "haiku-4.5": "claude-haiku-4-5-20251001",
    "haiku-3.5": "claude-haiku-4-5-20251001", # legacy alias -> current Haiku
}

_CODEX_MODEL_MAP = {
    # Preserve existing CoreSmith model tiers when switching providers.
    "opus-5": "gpt-5.6-sol",
    "sonnet-5": "gpt-5.6-terra",
    "opus-4.8": "gpt-5.6",
    "opus-4.7": "gpt-5.6",                    # legacy alias -> current Codex tier
    "opus-4.6": "gpt-5.6",                    # legacy alias -> current Codex tier
    "sonnet-4.6": "gpt-5.4-mini",
    "sonnet-4.5": "gpt-5.4-mini",
    "haiku-4.5": "gpt-5.4-mini",
    "haiku-3.5": "gpt-5.4-mini",
}

# agy (Google Antigravity) CLI. The reasoning-effort tier is BAKED INTO the
# model id -- ``agy models`` lists "Gemini 3.1 Pro (High)" / "(Low)" etc. --
# so "high effort" is selected by choosing the "(High)" variant, not a flag.
# Maps the CoreSmith opus/sonnet/haiku tiers onto Gemini tiers.
_AGY_MODEL_MAP = {
    "opus-5": "Gemini 3.1 Pro (High)",
    "sonnet-5": "Gemini 3.1 Pro (High)",
    "opus-4.8": "Gemini 3.1 Pro (High)",
    "opus-4.7": "Gemini 3.1 Pro (High)",
    "opus-4.6": "Gemini 3.1 Pro (High)",
    "sonnet-4.6": "Gemini 3.1 Pro (High)",
    "sonnet-4.5": "Gemini 3.1 Pro (High)",
    "haiku-4.5": "Gemini 3.5 Flash (High)",
    "haiku-3.5": "Gemini 3.5 Flash (High)",
}

_OPENCODE_MODEL_MAP = {
    # OpenRouter's hosted Kimi K3 for every CoreSmith tier.
    "opus-5": "openrouter/moonshotai/kimi-k3",
    "sonnet-5": "openrouter/moonshotai/kimi-k3",
    "opus-4.8": "openrouter/moonshotai/kimi-k3",
    "opus-4.7": "openrouter/moonshotai/kimi-k3",
    "opus-4.6": "openrouter/moonshotai/kimi-k3",
    "sonnet-4.6": "openrouter/moonshotai/kimi-k3",
    "sonnet-4.5": "openrouter/moonshotai/kimi-k3",
    "haiku-4.5": "openrouter/moonshotai/kimi-k3",
    "haiku-3.5": "openrouter/moonshotai/kimi-k3",
}


# Default model used by every agent unless overridden. Set the CORESMITH_MODEL
# environment variable (to either a short name above or a full Claude CLI
# model ID) to override at runtime without code changes -- useful when the
# default version is unavailable on a fresh CLI install.
DEFAULT_MODEL = "opus-5"
DEFAULT_CODEX_MODEL = "gpt-5.6-sol"
DEFAULT_AGY_MODEL = "Gemini 3.1 Pro (High)"

DEFAULT_OPENCODE_MODEL = "openrouter/moonshotai/kimi-k3"
# Cheaper model for per-block agents (uarch, rtl, testbench, diagnose, lint
# fix, tb fix).  Integration and review agents still call DEFAULT_MODEL.
# Override with CORESMITH_BLOCK_MODEL env var.
BLOCK_MODEL = "sonnet-5"


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

    if provider == "agy_cli":
        env_override = (
            os.environ.get("CORESMITH_AGY_MODEL", "").strip()
            or os.environ.get("CORESMITH_MODEL", "").strip()
        )
        if env_override:
            return _AGY_MODEL_MAP.get(env_override, env_override)
        if not model:
            return DEFAULT_AGY_MODEL
        return _AGY_MODEL_MAP.get(model, model)
    if provider == "opencode_cli":
        env_override = (
            os.environ.get("CORESMITH_OPENCODE_MODEL", "").strip()
            or os.environ.get("CORESMITH_MODEL", "").strip()
        )
        if env_override:
            return _OPENCODE_MODEL_MAP.get(env_override, env_override)
        if not model:
            return DEFAULT_OPENCODE_MODEL
        return _OPENCODE_MODEL_MAP.get(model, model)

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


def arch_reasoning_effort() -> str:
    """Reasoning-effort tier for the architecture/spec stages (PRD, SAD, FRD,
    uarch). These few calls set the frozen artifacts every downstream block
    inherits -- a decomposition or feasibility mistake here costs whole triage
    rounds -- so they default to ``xhigh`` while the bulk per-block work stays
    at the global default (``high``). Override with
    ``CORESMITH_CODEX_REASONING_EFFORT_ARCH``. Codex-only: the claude path
    keeps ``CORESMITH_CLAUDE_EFFORT`` and agy bakes effort into the model name.
    """
    return (os.environ.get("CORESMITH_CODEX_REASONING_EFFORT_ARCH", "").strip()
            or "xhigh")


# ---------------------------------------------------------------------------
# Testing-provider registry (Package C)
# ---------------------------------------------------------------------------
# ``fault``/``replay`` route calls to in-process backends under
# ``orchestrator.testing``. Production NEVER imports that package: _detect_provider
# only compares strings, and _get_testing_backend does the import lazily, so a
# non-test run has zero coupling to the test harness.
_TESTING_PROVIDER_MODULES: dict[str, str] = {
    "fault": "orchestrator.testing.fault_provider",
    "replay": "orchestrator.testing.replay_provider",
}
_TESTING_PROVIDERS: frozenset[str] = frozenset(_TESTING_PROVIDER_MODULES)


def _get_testing_backend(provider: str):
    """Lazily import + return the testing backend for ``provider``.

    Raises a clear error if the ``orchestrator.testing`` package is absent
    (it only ships with the test suite). Never imported on a production path.
    """
    import importlib

    mod_name = _TESTING_PROVIDER_MODULES.get(provider)
    if not mod_name:
        raise ValueError(f"Unknown testing provider {provider!r}")
    try:
        mod = importlib.import_module(mod_name)
    except ImportError as exc:  # pragma: no cover - only in a stripped install
        raise RuntimeError(
            f"CORESMITH_LLM_PROVIDER={provider!r} needs the test-only "
            f"module {mod_name!r}, which is not importable: {exc}"
        ) from exc
    return mod.get_backend()


def _detect_provider() -> str:
    """Detect which LLM provider to use.

    Defaults to Claude CLI.  Set ``CORESMITH_LLM_PROVIDER=codex`` (or
    ``codex_cli``) to route calls through ``codex exec``. The test-only
    providers ``fault``/``replay`` route to ``orchestrator.testing`` backends.
    """
    provider = os.environ.get("CORESMITH_LLM_PROVIDER", "").strip().lower()
    if provider in {"codex", "codex_cli"}:
        return "codex_cli"
    if provider in {"opencode", "opencode_cli", "openrouter"}:
        return "opencode_cli"
    if provider in {"claude", "claude_cli", ""}:
        return "claude_cli"
    if provider in {"agy", "gemini", "antigravity", "agy_cli"}:
        return "agy_cli"
    if provider in _TESTING_PROVIDERS:
        return provider
    raise ValueError(
        "Unsupported CORESMITH_LLM_PROVIDER={!r}. Use 'claude', 'codex', "
        "'opencode', 'agy', or a testing provider ({}).".format(
            provider, ", ".join(sorted(_TESTING_PROVIDERS))
        )
    )


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


def _find_agy_binary() -> str:
    """Locate the agy (Google Antigravity) CLI binary."""
    env_path = os.environ.get("AGY_CLI_PATH", "")
    if env_path and os.path.isfile(env_path) and os.access(env_path, os.X_OK):
        return env_path

    which_path = shutil.which("agy")
    if which_path:
        return which_path

    candidates = [
        os.path.expanduser("~/.local/bin/agy"),
        os.path.expanduser("~/.npm-global/bin/agy"),
    ]
    for path in candidates:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path

    raise FileNotFoundError(
        "agy CLI not found. Install Antigravity CLI or set AGY_CLI_PATH to the binary location."
    )


# ---------------------------------------------------------------------------
# Codex ``exec resume`` capability probe (A-Fix 6)
# ---------------------------------------------------------------------------
# Newer codex CLIs reject flags on ``exec resume`` that they accept on plain
# ``exec`` (measured: ``error: unexpected argument '--sandbox'`` -> exit 2 ->
# every session-resume falls back to a fresh 81 KB cold call). We probe
# ``codex exec resume --help`` once per binary path, extract the flags it
# advertises, and drop any resume-argv flag the CLI doesn't know about. Probe
# failure -> ``None`` -> the legacy (unfiltered) argv, so old CLIs still work.

_RESUME_FLAGS_CACHE: dict[str, frozenset[str] | None] = {}
_RESUME_FLAGS_CACHE_LOCK = threading.Lock()
_RESUME_PROBE_TIMEOUT_S = 10


def _codex_resume_supported_flags(codex_path: str) -> frozenset[str] | None:
    """Return the flag tokens ``codex exec resume`` advertises, or ``None``.

    ``None`` means "could not determine" (binary missing, probe timed out, or
    help had no recognizable flags) -> callers should use the legacy argv
    unchanged. Cached per binary path in ``_RESUME_FLAGS_CACHE``.
    """
    if not codex_path:
        return None
    with _RESUME_FLAGS_CACHE_LOCK:
        if codex_path in _RESUME_FLAGS_CACHE:
            return _RESUME_FLAGS_CACHE[codex_path]

    flags: frozenset[str] | None
    try:
        proc = subprocess.run(
            [codex_path, "exec", "resume", "--help"],
            capture_output=True,
            text=True,
            timeout=_RESUME_PROBE_TIMEOUT_S,
        )
        help_text = (proc.stdout or "") + "\n" + (proc.stderr or "")
        # Match both long (--flag) and short (-C) option tokens.
        found = set(_re.findall(r"(?<![\w-])(--[A-Za-z][\w-]*|-[A-Za-z])(?![\w-])", help_text))
        flags = frozenset(found) if found else None
    except Exception:
        # Missing binary / timeout / any error -> "unknown" -> legacy argv.
        flags = None

    with _RESUME_FLAGS_CACHE_LOCK:
        _RESUME_FLAGS_CACHE[codex_path] = flags
    return flags


def _invalidate_resume_flags(codex_path: str) -> None:
    """Drop the cached probe result so the next call re-probes ``codex_path``."""
    with _RESUME_FLAGS_CACHE_LOCK:
        _RESUME_FLAGS_CACHE.pop(codex_path, None)


def _find_opencode_binary() -> str:
    """Locate the official OpenCode CLI binary."""
    env_path = os.environ.get("OPENCODE_CLI_PATH", "")
    if env_path and os.path.isfile(env_path) and os.access(env_path, os.X_OK):
        return env_path

    which_path = shutil.which("opencode")
    if which_path:
        return which_path

    candidates = [
        os.path.expanduser("~/.opencode/bin/opencode"),
        os.path.expanduser("~/.local/bin/opencode"),
        os.path.expanduser("~/.npm-global/bin/opencode"),
        os.path.expanduser("~/.npm/bin/opencode"),
    ]
    for path in candidates:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path

    raise FileNotFoundError(
        "OpenCode CLI not found. Install it with: npm install -g opencode-ai\n"
        "Or set OPENCODE_CLI_PATH to the binary location."
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
        codex_path: str = "",
        agy_path: str = "",
        opencode_path: str = "",
        timeout: int = 1200,
        max_turns: int = 50,
        disable_tools: bool = False,
        reasoning_effort: str = "",
    ) -> None:
        self.model = model or DEFAULT_MODEL
        self.claude_path = claude_path
        self.codex_path = codex_path
        self.agy_path = agy_path
        self.opencode_path = opencode_path
        self.timeout = timeout
        self.max_turns = max_turns
        self.disable_tools = disable_tools
        # Per-instance codex reasoning-effort override (e.g. the architecture
        # specialists pass arch_reasoning_effort() -> "xhigh"). Empty -> the
        # CORESMITH_CODEX_REASONING_EFFORT env var -> "high". Codex-only; the
        # claude/agy paths have their own effort mechanisms.
        self.reasoning_effort = (reasoning_effort or "").strip()
        # Session id captured from the last codex ``thread.started`` event
        # (empty for claude / when no id was seen). Callers read this after a
        # ``call()`` to thread it into a later resume.
        self.last_session_id: str = ""

        self._provider = _detect_provider()
        logger.info("ClaudeLLM using %s provider.", self._provider)
        if self._provider in _TESTING_PROVIDERS:
            # Testing backends (fault/replay) run in-process -- no CLI binary,
            # so importing agents in CI does not require the Claude/Codex CLI.
            pass
        elif self._provider == "claude_cli" and not self.claude_path:
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
                logger.info(f"Found Codex CLI at: {self.codex_path}")
        elif self._provider == "agy_cli" and not self.agy_path:
            self.agy_path = os.environ.get("AGY_CLI_PATH", "")
            if not self.agy_path:
                self.agy_path = _find_agy_binary()
                logger.info(f"Found agy CLI at: {self.agy_path}")
        elif self._provider == "opencode_cli" and not self.opencode_path:
            self.opencode_path = _find_opencode_binary()
            logger.info("Found OpenCode CLI at: %s", self.opencode_path)

    async def call(
        self,
        system: str = "",
        prompt: str = "",
        run_name: str = "",
        resume_session_id: str | None = None,
    ) -> str:
        """Call the Claude CLI and return the response text.

        Args:
            system: System prompt text.
            prompt: User/human prompt text.
            run_name: Label for telemetry events (replaces LangChain config.run_name).
            resume_session_id: When set AND ``CORESMITH_CODEX_RESUME`` is truthy
                and the provider is codex, resume the prior codex session
                (``codex exec resume <id> ...``) instead of starting a fresh
                ``codex exec`` -- so a retried agent continues its earlier
                conversation. Default-OFF: unless the env flag is set this
                argument is ignored and existing behavior is unchanged. If the
                resume fails (e.g. session gone) the codex path falls back to a
                fresh exec.

        Returns:
            Response text from the LLM.

        The codex session id captured from the ``thread.started`` event (when
        provider is codex) is stashed on ``self.last_session_id`` after the call
        so callers can thread it into a later resume.
        """
        _get_breaker(_breaker_context.get("")).check()

        # Reset the per-call captured session id (codex only).
        self.last_session_id = ""

        # Stamp call-site attribution into the ContextVar so _log_llm_call can
        # join each record to (run_name, call_index, graph) without a signature
        # change at its 5+ call sites. Concurrent block fan-out tasks each have
        # their own context copy, so these don't collide across blocks.
        _call_site_context.set({
            "run_name": run_name,
            "call_index": _next_call_index(),
            "graph": _breaker_context.get(""),
        })

        # Write llm_start event
        project_root = _llm_log_root()
        self._write_llm_event(project_root, "llm_start", {
            "model": _resolve_model(self.model, self._provider),
            "provider": self._provider,
            "run_name": run_name,
            "prompt_chars": len(prompt),
            "system_chars": len(system),
        })

        t_start = _time_mod.monotonic()
        try:
            loop = asyncio.get_running_loop()
            # Executor threads do NOT inherit contextvars; copy the current
            # context (which carries _call_site_context) into the worker so the
            # synchronous _log_llm_call inside _generate_via_cli sees it.
            ctx = contextvars.copy_context()
            text = await loop.run_in_executor(
                None,
                lambda: ctx.run(
                    self._generate_via_cli, system, prompt, resume_session_id,
                ),
            )
            _get_breaker(_breaker_context.get("")).record_success()

            # Write llm_end event
            self._write_llm_event(project_root, "llm_end", {
                "model": _resolve_model(self.model, self._provider),
                "provider": self._provider,
                "run_name": run_name,
                "output_chars": len(text),
                "session_id": self.last_session_id,
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

            # Also record the FAILED call in llm_calls.jsonl. Historically only
            # completed calls were logged, so raised failures (provider
            # exceptions, timeouts that surface as exceptions) never made it into
            # the replay corpus. Best-effort: telemetry must never mask the real
            # error being re-raised.
            try:
                _log_llm_call(
                    model=_resolve_model(self.model, self._provider),
                    provider=self._provider,
                    system_prompt=system,
                    user_prompt=prompt,
                    response="",
                    duration_s=_time_mod.monotonic() - t_start,
                    timeout=self.timeout,
                    error=str(e)[:2000],
                )
            except Exception:
                logger.debug("failed-call llm log write failed", exc_info=True)

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
    # After the child's response is complete / returncode obtained, how long to
    # let its process group wind down before SIGKILL. Bounds the post-response
    # reap so a lingering grandchild (a sim the CLI spawned) that holds our
    # stdout/stderr pipe open can't stall the exit until the hard-timeout.
    _REAP_GRACE_S: float = 10.0

    def _generate_via_cli(
        self,
        system_prompt: str,
        user_prompt: str,
        resume_session_id: str | None = None,
    ) -> str:
        if self._provider in _TESTING_PROVIDERS:
            # Preserves the opencode_patch 3-arg seam: the testing backend
            # receives the LLM instance so it can inspect self.model / self.timeout.
            return _get_testing_backend(self._provider).generate(
                self, system_prompt, user_prompt, resume_session_id,
            )

        if self._provider == "opencode_cli":
            return self._generate_via_opencode_cli(system_prompt, user_prompt)

        if self._provider == "codex_cli":
            return self._generate_via_codex_cli(
                system_prompt, user_prompt, resume_session_id,
            )

        if self._provider == "agy_cli":
            return self._generate_via_agy_cli(
                system_prompt, user_prompt, resume_session_id,
            )

        # resume is a codex-only feature; claude path ignores it.
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

        # Reasoning-effort control (Claude Code ``--effort <level>``). Default
        # OFF -- existing behavior is byte-identical unless CORESMITH_CLAUDE_EFFORT
        # is set (e.g. ``high`` to match a codex ``model_reasoning_effort=high``
        # or an agy "(High)" model tier in a cross-provider A/B).
        _claude_effort = os.environ.get("CORESMITH_CLAUDE_EFFORT", "").strip()
        if _claude_effort:
            cmd.extend(["--effort", _claude_effort])

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

    @staticmethod
    def _codex_resume_enabled() -> bool:
        """Whether codex SESSION RESUME is enabled via ``CORESMITH_CODEX_RESUME``.

        Default-OFF globally: existing behavior is unchanged unless the flag is
        explicitly truthy (``1``/``true``/``yes``/``on``). The microarch
        experiment runner sets it default-ON for its own process.
        """
        val = os.environ.get("CORESMITH_CODEX_RESUME", "").strip().lower()
        return val in {"1", "true", "yes", "on"}

    @staticmethod
    def _build_codex_cmd(
        codex_path: str,
        resolved_model: str,
        workdir: str,
        sandbox: str,
        resume_session_id: str | None = None,
        supported_flags: frozenset[str] | None = None,
        reasoning_effort: str = "",
    ) -> list[str]:
        """Construct the ``codex exec [resume <id>]`` argv (testable, no I/O).

        ``reasoning_effort`` overrides the model_reasoning_effort tier for this
        call (the architecture specialists pass "xhigh"); empty falls back to
        the ``CORESMITH_CODEX_REASONING_EFFORT`` env var, then "high".

        When ``resume_session_id`` is set AND ``CORESMITH_CODEX_RESUME`` is
        truthy, the command becomes ``codex exec resume <id> [flags] -`` so the
        agent continues its prior conversation; otherwise it is a plain
        ``codex exec [flags] -``. The prompt is still piped on stdin via the
        trailing ``-``.

        ``supported_flags`` is the capability-probe result for the resume
        subcommand (from ``_codex_resume_supported_flags``). On a resume argv,
        any flag NOT in ``supported_flags`` is dropped (fixes ``error:
        unexpected argument '--sandbox'`` on newer CLIs). ``None`` means
        "unknown" -> no filtering (legacy argv). The FRESH (non-resume) argv is
        NEVER filtered, so it stays byte-identical to prior behavior.
        """
        head: list[str] = [codex_path, "exec"]
        is_resume = bool(resume_session_id) and ClaudeLLM._codex_resume_enabled()
        if is_resume:
            head += ["resume", resume_session_id]

        # (flag, value-or-None) in the original argv order.
        tail_spec: list[tuple[str, str | None]] = [
            ("--json", None),
            ("--dangerously-bypass-approvals-and-sandbox", None),
            ("--sandbox", sandbox),
            ("--skip-git-repo-check", None),
            ("-C", workdir),
            ("-c", "model_reasoning_effort=" + (
                (reasoning_effort or "").strip()
                or os.environ.get("CORESMITH_CODEX_REASONING_EFFORT", "high").strip()
                or "high"
            )),
            ("-m", resolved_model),
        ]

        tail: list[str] = []
        for flag, value in tail_spec:
            if (
                is_resume
                and supported_flags is not None
                and flag not in supported_flags
            ):
                continue  # CLI's `exec resume` doesn't accept this flag -> drop.
            tail.append(flag)
            if value is not None:
                tail.append(value)
        tail.append("-")
        return head + tail

    # Flag tokens the codex ``exec`` argv carries (values + trailing "-"
    # excluded); used to report which flags the resume capability probe dropped.
    _CODEX_EXEC_FLAG_TOKENS: frozenset[str] = frozenset({
        "--json",
        "--dangerously-bypass-approvals-and-sandbox",
        "--sandbox",
        "--skip-git-repo-check",
        "-C",
        "-c",
        "-m",
    })

    @staticmethod
    def _resume_flags_dropped(supported_flags: frozenset[str]) -> set[str]:
        """Flags the fresh argv carries but the resume subcommand won't accept."""
        return {
            f for f in ClaudeLLM._CODEX_EXEC_FLAG_TOKENS
            if f not in supported_flags
        }

    @staticmethod
    def _codex_launch_cwd(cmd: list[str], workdir: str) -> str | None:
        """Working dir to launch the codex ``Popen`` in.

        When ``-C``/``--cd`` is present in ``cmd`` codex switches to
        ``workdir`` itself, so the launch cwd is irrelevant (return ``None``
        -> preserve prior behavior). When the flag filter dropped ``-C`` on a
        resume argv, codex would otherwise inherit the CALLER's cwd and write
        relative paths there; return ``workdir`` so ``Popen(cwd=...)`` restores
        the isolation the ``-C`` flag used to provide.
        """
        if "-C" in cmd or "--cd" in cmd:
            return None
        return workdir

    def _generate_via_codex_cli(
        self,
        system_prompt: str,
        user_prompt: str,
        resume_session_id: str | None = None,
    ) -> str:
        """Call the Codex CLI (``codex exec``) synchronously.

        When ``resume_session_id`` is set AND ``CORESMITH_CODEX_RESUME`` is
        truthy, the first attempt resumes the prior session; if that fails
        (non-zero exit / empty output -- e.g. the session file is gone) we log a
        warning and fall back to a fresh ``codex exec`` so a missing session
        never hard-fails the run.
        """
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

        resuming = bool(resume_session_id) and self._codex_resume_enabled()
        # A-Fix 6: probe which flags `codex exec resume` accepts and drop the
        # rest, so a newer CLI doesn't reject the resume with exit 2 (which
        # historically forced a fresh 81 KB cold call every time).
        supported_flags = (
            _codex_resume_supported_flags(self.codex_path) if resuming else None
        )
        cmd: list[str] = self._build_codex_cmd(
            self.codex_path, resolved_model, workdir, sandbox,
            resume_session_id, supported_flags=supported_flags,
            reasoning_effort=self.reasoning_effort,
        )
        if resuming:
            logger.info(
                "Codex CLI RESUMING session %s (CORESMITH_CODEX_RESUME on).",
                resume_session_id,
            )
            if supported_flags is not None:
                dropped = self._resume_flags_dropped(supported_flags)
                if dropped:
                    logger.warning(
                        "Codex CLI resume: dropping unsupported flags %s", sorted(dropped),
                    )
                    self._write_llm_event(log_root, "llm_resume_flags_dropped", {
                        "model": resolved_model,
                        "provider": "codex_cli",
                        "resume_session_id": resume_session_id,
                        "dropped_flags": sorted(dropped),
                        "supported_flags": sorted(supported_flags),
                    })

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
                        cwd=self._codex_launch_cwd(cmd, workdir),
                    )
            )
            # RESUME FALLBACK: a resume that fails (session gone -> non-zero
            # exit / empty output, not a timeout/stall) must never hard-fail.
            if (
                resuming
                and not timed_out
                and not stalled
                and (returncode != 0 or not (output or "").strip())
            ):
                # A-Fix 6: if the resume was rejected for an unexpected argument,
                # our cached probe is stale (or the CLI changed under us). Drop
                # the cache, re-probe once, rebuild the resume argv, and retry the
                # RESUME (not a fresh exec) so we still keep the session cache.
                if "unexpected argument" in (stderr_text or "").lower():
                    _invalidate_resume_flags(self.codex_path)
                    supported_flags = _codex_resume_supported_flags(self.codex_path)
                    self._write_llm_event(log_root, "llm_resume_flags_dropped", {
                        "model": resolved_model,
                        "provider": "codex_cli",
                        "resume_session_id": resume_session_id,
                        "reason": "unexpected_argument",
                        "stderr": (stderr_text or "")[:300],
                        "supported_flags": (
                            sorted(supported_flags) if supported_flags is not None else None
                        ),
                    })
                    reprobe_cmd = self._build_codex_cmd(
                        self.codex_path, resolved_model, workdir, sandbox,
                        resume_session_id, supported_flags=supported_flags,
                        reasoning_effort=self.reasoning_effort,
                    )
                    t0 = _time_mod.monotonic()
                    span_start_ns = _time_mod.time_ns()
                    output, stderr_text, returncode, elapsed, timed_out, stalled, usage = (
                            self._run_cli_with_watchdog(
                                reprobe_cmd, combined_prompt, log_root, resolved_model, t0,
                                cwd=self._codex_launch_cwd(reprobe_cmd, workdir),
                            )
                    )

            # If the resume (and any re-probe retry) still failed, rebuild a
            # FRESH exec command and re-run once.
            if (
                resuming
                and not timed_out
                and not stalled
                and (returncode != 0 or not (output or "").strip())
            ):
                logger.warning(
                    "Codex CLI resume of session %s failed "
                    "(rc=%s, out_len=%d, stderr=%r); falling back to a fresh exec.",
                    resume_session_id, returncode, len(output or ""),
                    (stderr_text or "")[:200],
                )
                self._write_llm_event(log_root, "llm_resume_fallback", {
                    "model": resolved_model,
                    "provider": "codex_cli",
                    "resume_session_id": resume_session_id,
                    "exit_code": returncode,
                    "stderr": (stderr_text or "")[:500],
                })
                fresh_cmd = self._build_codex_cmd(
                    self.codex_path, resolved_model, workdir, sandbox, None,
                    reasoning_effort=self.reasoning_effort,
                )
                t0 = _time_mod.monotonic()
                span_start_ns = _time_mod.time_ns()
                output, stderr_text, returncode, elapsed, timed_out, stalled, usage = (
                        self._run_cli_with_watchdog(
                            fresh_cmd, combined_prompt, log_root, resolved_model, t0,
                            cwd=self._codex_launch_cwd(fresh_cmd, workdir),
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

        # Surface the codex session id (from the thread.started event, stashed
        # into usage by _parse_codex_json) so a later retry can resume it.
        try:
            sid = (usage or {}).get("session_id")
            if sid:
                self.last_session_id = str(sid)
        except Exception:  # noqa: BLE001 - never break the call over telemetry
            pass

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

    def _generate_via_opencode_cli(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Call OpenCode with Kimi through its built-in OpenRouter provider."""
        resolved_model = _resolve_model(self.model, self._provider)
        workdir = (
            os.environ.get("CORESMITH_OPENCODE_WORKDIR", "").strip()
            or os.environ.get("CORESMITH_PROJECT_ROOT", "").strip()
            or _default_project_root()
        )
        Path(workdir).mkdir(parents=True, exist_ok=True)
        project_root = _llm_log_root()
        combined_prompt = self._build_codex_prompt(system_prompt, user_prompt)

        cmd = [
            self.opencode_path,
            "--pure",
            "run",
            "--format", "json",
            "--thinking",
            "--model", resolved_model,
            "--dir", workdir,
            "--auto",
        ]

        process_env = os.environ.copy()
        if self.disable_tools:
            try:
                inline_config = _json.loads(
                    process_env.get("OPENCODE_CONFIG_CONTENT", "{}") or "{}"
                )
            except _json.JSONDecodeError as exc:
                raise ValueError(
                    "OPENCODE_CONFIG_CONTENT must be valid JSON when disable_tools=True"
                ) from exc
            if not isinstance(inline_config, dict):
                raise ValueError("OPENCODE_CONFIG_CONTENT must contain a JSON object")
            inline_config["permission"] = "deny"
            process_env["OPENCODE_CONFIG_CONTENT"] = _json.dumps(inline_config)

        logger.info(
            "OpenCode invocation: model=%s prompt_len=%d system_len=%d",
            resolved_model,
            len(user_prompt),
            len(system_prompt),
        )
        t0 = _time_mod.monotonic()
        span_start_ns = _time_mod.time_ns()
        try:
            output, stderr_text, returncode, elapsed, timed_out, stalled, usage = (
                self._run_cli_with_watchdog(
                    cmd,
                    combined_prompt,
                    project_root,
                    resolved_model,
                    t0,
                    process_env=process_env,
                )
            )
        except FileNotFoundError:
            elapsed = _time_mod.monotonic() - t0
            error_msg = "OpenCode CLI binary not found"
            output = (
                "[ClaudeLLM error: OpenCode CLI binary not found. "
                "Install: npm install -g opencode-ai]"
            )
            _log_llm_call(
                model=resolved_model,
                provider="opencode_cli",
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
            error_msg = f"OpenCode CLI {reason} after {elapsed:.0f}s"
            if stderr_text:
                error_msg += f" | stderr: {stderr_text[:300]}"
            output = f"[ClaudeLLM error: {error_msg}]"
        elif returncode != 0:
            error_msg = (
                f"OpenCode CLI exited with code {returncode}: "
                f"{stderr_text[:500] or output[:500]}"
            )
            output = f"[ClaudeLLM error: {error_msg}]"
        elif not output:
            error_msg = f"OpenCode CLI returned empty response: {stderr_text[:500]}"
            output = f"[ClaudeLLM error: {error_msg}]"
        else:
            error_msg = ""

        _log_llm_call(
            model=resolved_model,
            provider="opencode_cli",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response=output,
            duration_s=elapsed,
            timeout=self.timeout,
            error=error_msg,
            timed_out=timed_out or stalled,
            usage=usage,
            start_ts_ns=span_start_ns,
        )
        return output

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
    # agy (Antigravity) CLI path
    # ------------------------------------------------------------------

    @staticmethod
    def _agy_resume_enabled() -> bool:
        """Whether agy conversation RESUME is enabled via ``CORESMITH_AGY_RESUME``.

        Default-OFF. ``agy --print`` is a one-shot that does NOT echo its
        conversation id on stdout, so there is no reliable per-call id to thread
        into ``--conversation <id>`` on a later retry (picking the newest
        ``conversations/*.db`` by mtime is racy under concurrent block fan-out).
        We therefore run STATELESS by default -- which matches codex's own
        default (its resume is likewise gated OFF via CORESMITH_CODEX_RESUME).
        Set CORESMITH_AGY_RESUME=1 only if a caller threads an id captured
        out-of-band.
        """
        val = os.environ.get("CORESMITH_AGY_RESUME", "").strip().lower()
        return val in {"1", "true", "yes", "on"}

    @staticmethod
    def _build_agy_cmd(
        agy_path: str,
        resolved_model: str,
        workdir: str,
        timeout_s: int,
        resume_session_id: str | None = None,
    ) -> list[str]:
        """Construct the ``agy --print`` argv (testable, no I/O).

        The reasoning-effort tier is encoded in ``resolved_model`` (e.g.
        "Gemini 3.1 Pro (High)"), so there is no separate effort flag.
        ``--dangerously-skip-permissions`` auto-approves tool use (headless, no
        prompt hangs). ``--add-dir <workdir>`` grants the agent read/write to
        the project root so tool-using agents (e.g. the Integration Lead) can
        edit files, mirroring codex's ``-C <workdir>`` + workspace-write.
        ``--print-timeout`` is pinned to our own watchdog timeout so agy does
        not self-abort at its 5-minute default mid-way through a long
        high-effort turn. The prompt itself is piped on stdin (avoids ARG_MAX
        on the 80 KB+ prompts CoreSmith sends).
        """
        cmd: list[str] = [
            agy_path,
            "--print",
            "--dangerously-skip-permissions",
            "--model", resolved_model,
        ]
        if workdir:
            cmd.extend(["--add-dir", workdir])
        if resume_session_id and ClaudeLLM._agy_resume_enabled():
            cmd.extend(["--conversation", resume_session_id])
        cmd.extend(["--print-timeout", f"{int(timeout_s)}s"])
        return cmd

    def _generate_via_agy_cli(
        self,
        system_prompt: str,
        user_prompt: str,
        resume_session_id: str | None = None,
    ) -> str:
        """Call the agy (Antigravity) CLI (``agy --print``) synchronously.

        Runs stateless one-shot: the system+user prompt is combined into a
        single stdin payload (agy has no separate ``--system-prompt`` flag) and
        the plain stdout text is captured as the response (agy ``--print``
        emits the final answer as clean text, not a JSON event stream).
        Timeouts / stalls / empty responses are handled and logged exactly like
        the codex/claude paths, so agy calls land in ``llm_calls.jsonl`` with
        the same ``run_name`` call-site attribution.
        """
        resolved_model = _resolve_model(self.model, self._provider)

        workdir = (
            os.environ.get("CORESMITH_AGY_WORKDIR", "").strip()
            or os.environ.get("CORESMITH_PROJECT_ROOT", "").strip()
            or _default_project_root()
        )
        log_root = _llm_log_root()

        cmd: list[str] = self._build_agy_cmd(
            self.agy_path, resolved_model, workdir, self.timeout, resume_session_id,
        )
        # agy has no --system-prompt; fold system+user into one stdin payload
        # using the same <system>/<user> envelope the codex path uses.
        combined_prompt = self._build_codex_prompt(system_prompt, user_prompt)

        logger.debug(
            "agy CLI invocation: model=%s, prompt_len=%d, system_len=%d",
            resolved_model, len(user_prompt), len(system_prompt),
        )
        logger.info("agy CLI cmd (first 200): %s", " ".join(cmd)[:200])

        t0 = _time_mod.monotonic()
        span_start_ns = _time_mod.time_ns()
        try:
            output, stderr_text, returncode, elapsed, timed_out, stalled, usage = (
                self._run_cli_with_watchdog(
                    cmd, combined_prompt, log_root, resolved_model, t0,
                    cwd=workdir,
                )
            )
        except FileNotFoundError:
            elapsed = _time_mod.monotonic() - t0
            error_msg = "agy CLI binary not found"
            output = (
                "[ClaudeLLM error: agy CLI binary not found. "
                "Install Antigravity CLI or set AGY_CLI_PATH]"
            )
            _log_llm_call(
                model=resolved_model, provider="agy_cli",
                system_prompt=system_prompt, user_prompt=user_prompt,
                response=output, duration_s=elapsed, timeout=self.timeout,
                error=error_msg, start_ts_ns=span_start_ns,
            )
            return output

        if timed_out or stalled:
            reason = "stalled" if stalled else "timed out"
            error_msg = f"agy CLI {reason} after {elapsed:.0f}s"
            if stderr_text:
                error_msg += f" | stderr: {stderr_text[:300]}"
            if output:
                error_msg += f" | partial stdout: {output[:300]}"
            full_output = f"[ClaudeLLM error: {error_msg}]"
            logger.error("agy CLI %s: %s", reason, error_msg)
            _log_llm_call(
                model=resolved_model, provider="agy_cli",
                system_prompt=system_prompt, user_prompt=user_prompt,
                response=full_output, duration_s=elapsed, timeout=self.timeout,
                error=error_msg, timed_out=True, usage=usage,
                start_ts_ns=span_start_ns,
            )
            self._write_llm_event(log_root, f"llm_{reason}", {
                "model": resolved_model, "provider": "agy_cli",
                "elapsed_s": round(elapsed, 1),
                "partial_stdout_len": len(output),
                "stderr": stderr_text[:500], "partial_stdout": output[:500],
            })
            return full_output

        elapsed = _time_mod.monotonic() - t0
        logger.info(
            "agy CLI output: retcode=%s, stdout_len=%d, first100=%r",
            returncode, len(output), output[:100],
        )
        if returncode != 0:
            logger.warning("agy CLI exited with code %s: %s", returncode, stderr_text[:500])

        if not output:
            error_msg = (
                f"[ClaudeLLM error: agy CLI returned empty response. "
                f"exit_code={returncode}, stderr: {stderr_text[:500]}]"
            )
            output = error_msg
            logger.error("LLM empty response: %s", error_msg)
            self._write_llm_event(log_root, "llm_empty_response", {
                "model": resolved_model, "provider": "agy_cli",
                "exit_code": returncode, "stderr": stderr_text[:500],
                "error": error_msg[:300],
            })

        _log_llm_call(
            model=resolved_model, provider="agy_cli",
            system_prompt=system_prompt, user_prompt=user_prompt,
            response=output, duration_s=elapsed, timeout=self.timeout,
            usage=usage, start_ts_ns=span_start_ns,
        )
        return output

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
        cwd: str | None = None,
        process_env: dict[str, str] | None = None,
    ) -> tuple[str, str, int, float, bool, bool, dict]:
        """Run the CLI via ``Popen`` with stall detection and heartbeats.

        Returns ``(response_text, stderr, returncode, elapsed, timed_out,
        stalled, usage)`` where ``response_text`` is the final model
        response extracted from the stream-json ``result`` event (or a
        concatenation of partial assistant-text chunks if the stream was
        cut short), and ``usage`` is the per-call token/cost dict (may be
        empty on failure).

        ``cwd`` sets the child process working directory. It is used by the
        codex resume path when the ``-C``/``--cd`` flag was dropped by the
        capability filter (newer CLIs reject ``-C`` on ``exec resume``): the
        intended isolated workdir is passed here instead so relative-path
        writes in the resumed turn still land in the codex-call-* dir rather
        than the caller's cwd. ``None`` preserves the prior launch behavior.
        Raises ``FileNotFoundError`` if the binary is missing.
        """
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd,
            env=process_env,
            # Own session/group so we can reap the ENTIRE tree (the CLI plus any
            # sim/tool grandchildren it spawns) at the end -- see
            # _reap_process_group. Isolating the group also means killpg can't
            # signal the daemon/pytest process itself.
            start_new_session=True,
        )
        _register_process(process)
        # Capture the child's process-group id now, while it is guaranteed
        # alive and the group leader (pgid == pid). Re-deriving it later via
        # os.getpgid() would race a pid-reuse after poll() reaps the child.
        child_pgid = process.pid

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
            # The response is complete (loop exited on child exit, timeout, or
            # stall). Reap the child's process group so any lingering grandchild
            # (e.g. a sim the CLI spawned that inherited our stdout/stderr) is
            # killed and the pipe write-end closes -- otherwise the reader
            # threads below block on the still-open pipe until the hard-timeout
            # deadline (observed ~45-min post-response exit stall).
            _reap_process_group(process, child_pgid, grace_s=self._REAP_GRACE_S)
            # Backstop: close our read-ends so any reader still blocked wakes
            # with EOF even if a survivor somehow escaped the group kill.
            for _stream in (process.stdout, process.stderr):
                try:
                    _stream.close()
                except Exception:
                    pass
            # Reader threads should now see EOF promptly; short join suffices.
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
        elif self._provider == "opencode_cli":
            response_text, usage = _parse_opencode_json(stdout_text)
            # OpenCode stores sessions internally, but CoreSmith also keeps a
            # project-local raw trajectory so exposed reasoning remains with
            # the run artifacts and can be correlated by pid/run_name.
            try:
                _log_opencode_turns(
                    stdout_text,
                    project_root,
                    process.pid if process.pid else 0,
                    wall_start,
                )
            except Exception:
                pass
        elif self._provider == "agy_cli":
            # agy --print emits the final answer as plain text (no JSON event
            # stream); the whole stdout IS the response. No token/cost usage is
            # surfaced on stdout, so usage stays empty for agy calls.
            response_text, usage = stdout_text, {}
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
