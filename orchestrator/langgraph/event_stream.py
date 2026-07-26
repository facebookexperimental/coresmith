# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Pipeline observability via JSONL event logging.

Provides:
- write_graph_event(): standalone writer for graph-level events (node enter/exit,
  routing decisions).
- read_events(): reads back the JSONL log with optional timestamp filtering.
- format_event_summary(): formats events into human-readable terminal output.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

logger = logging.getLogger(__name__)

# Module-level lock protecting all JSONL file writes.  Both the
# ClaudeLLM.call() event writes and write_graph_event() append to the
# same file.  When pipeline blocks run in parallel via asyncio.to_thread(),
# concurrent writes can interleave and corrupt JSONL lines.  A
# threading lock (not asyncio.Lock) is needed because the callers
# are synchronous functions.
_write_lock = threading.Lock()

# ---------------------------------------------------------------------------
# ANSI colour constants
# ---------------------------------------------------------------------------
DIM = "\033[2m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

_LOG_RELPATH = ".coresmith/pipeline_events.jsonl"
_VALID_EXIT_STATUS = {"ok", "error", "skipped", "interrupted"}


# ---------------------------------------------------------------------------
# Exit hook registry -- async callbacks fired on graph_node_exit events
# ---------------------------------------------------------------------------

ExitHookFn = Callable[[str, str, dict], Awaitable[None]]

_exit_hooks: list[ExitHookFn] = []


def register_exit_hook(fn: ExitHookFn) -> None:
    """Register an async callback that fires on every ``graph_node_exit`` event.

    The callback signature is ``async fn(project_root, node_name, record)``.
    Hooks run as fire-and-forget ``asyncio.Task`` instances so they never
    block the pipeline.
    """
    _exit_hooks.append(fn)


# ---------------------------------------------------------------------------
# Standalone graph-level event writer
# ---------------------------------------------------------------------------

def write_graph_event(
    project_root: str,
    node_name: str,
    event_type: str,
    data: dict | None = None,
) -> None:
    """Write a graph-level event (node enter/exit, routing decisions) to the
    pipeline events JSONL log.

    After writing, any registered exit hooks are dispatched as background
    async tasks when the event type is ``graph_node_exit``.

    Args:
        project_root: Root directory of the project.
        node_name: Name of the graph node emitting the event.
        event_type: Event type string (e.g. ``graph_node_enter``).
        data: Arbitrary extra fields to include in the log record.
    """
    log_path = Path(project_root) / _LOG_RELPATH
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ts = time.time()
    payload = dict(data or {})
    if event_type == "graph_node_exit":
        payload["status"] = _derive_exit_status(payload)

    record = {
        "ts": ts,
        "iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(ts)),
        "event": event_type,
        "node": node_name,
        **payload,
    }
    line = json.dumps(record, default=str) + "\n"
    with _write_lock, open(log_path, "a", encoding="utf-8") as fh:
        fh.write(line)

    # Dispatch exit hooks as fire-and-forget async tasks
    if event_type == "graph_node_exit" and _exit_hooks:
        try:
            loop = asyncio.get_running_loop()
            for hook in _exit_hooks:
                loop.create_task(_safe_hook(hook, project_root, node_name, record))
        except RuntimeError:
            pass  # No event loop running (e.g. sync test context)


async def _safe_hook(
    hook: ExitHookFn, project_root: str, node_name: str, record: dict
) -> None:
    """Wrapper that catches and logs exceptions from exit hooks."""
    try:
        await hook(project_root, node_name, record)
    except Exception:
        logger.exception("Exit hook %s failed for node %s", hook.__name__, node_name)


def _derive_exit_status(data: dict) -> str:
    """Return a uniform status for a graph-node exit event.

    Historically most callers wrote ad hoc fields such as ``passed``,
    ``success``, ``clean``, or ``error``.  Keeping this normalization in the
    writer makes existing nodes observable without forcing every graph to be
    migrated in one large patch.
    """
    explicit = str(data.get("status", "")).strip().lower()
    aliases = {
        "true": "ok",
        "pass": "ok",
        "passed": "ok",
        "success": "ok",
        "fail": "error",
        "failed": "error",
        "failure": "error",
        "abort": "interrupted",
        "aborted": "interrupted",
    }
    explicit = aliases.get(explicit, explicit)
    if explicit in _VALID_EXIT_STATUS:
        return explicit

    if data.get("pipeline_aborted") or data.get("interrupted"):
        return "interrupted"

    if data.get("skipped") is True or data.get("skip") is True:
        return "skipped"
    result = str(data.get("result", "")).strip().lower()
    if result in {"skip", "skipped"}:
        return "skipped"

    for key in (
        "passed",
        "success",
        "clean",
        "ok",
        "synth_ok",
        "integration_ok",
        "validation_ok",
    ):
        if key in data and data.get(key) is False:
            return "error"

    if data.get("error") or data.get("exception") or data.get("failure"):
        return "error"

    return "ok"


# ---------------------------------------------------------------------------
# Event reader
# ---------------------------------------------------------------------------

def read_events(
    project_root: str,
    after_ts: float = 0,
) -> list[dict]:
    """Read pipeline events from the JSONL log.

    Args:
        project_root: Root directory of the project.
        after_ts: Only return events with ``ts`` strictly greater than this
            value.  Defaults to ``0`` (return all events).

    Returns:
        List of event dicts in chronological order.
    """
    log_path = Path(project_root) / _LOG_RELPATH
    if not log_path.exists():
        return []
    events: list[dict] = []
    with open(log_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("ts", 0) > after_ts:
                events.append(record)
    return events


# ---------------------------------------------------------------------------
# Failure aggregation helper
# ---------------------------------------------------------------------------

def aggregate_failure_categories(events: list[dict]) -> dict:
    """Aggregate failure categories from pipeline events.

    Scans ``graph_node_exit`` events for failure indicators (``category``,
    failed ``clean``/``passed``/``success`` flags) and returns a summary
    dict useful for trend analysis.

    Args:
        events: List of event dicts (from :func:`read_events`, pre-filtered
            to frontend events if desired).

    Returns:
        Dict with keys:
        - ``failure_categories``: ``{category: count}``
        - ``per_block_failures``: ``{block_name: [categories]}``
        - ``systematic_patterns``: categories affecting 3+ distinct blocks
        - ``total_failures``: total failure event count
        - ``avg_retries``: average attempts per block (from ``block_start``)
    """
    failure_categories: dict[str, int] = {}
    per_block_failures: dict[str, list[str]] = {}
    # Track which blocks have each category for systematic pattern detection
    category_blocks: dict[str, set[str]] = {}
    total_attempts: dict[str, int] = {}

    for e in events:
        block = e.get("block", "")
        etype = e.get("event", "")

        if etype == "block_start" and block:
            total_attempts[block] = e.get("attempt", 1)

        if etype == "graph_node_exit":
            cat = e.get("category", "")
            if cat and block:
                failure_categories[cat] = failure_categories.get(cat, 0) + 1
                per_block_failures.setdefault(block, []).append(cat)
                category_blocks.setdefault(cat, set()).add(block)

    # Systematic patterns: categories affecting 3+ distinct blocks
    systematic_patterns = [
        cat for cat, blocks in category_blocks.items()
        if len(blocks) >= 3
    ]

    total_failures = sum(failure_categories.values())
    avg_retries = (
        sum(total_attempts.values()) / len(total_attempts)
        if total_attempts else 0.0
    )

    return {
        "failure_categories": failure_categories,
        "per_block_failures": per_block_failures,
        "systematic_patterns": systematic_patterns,
        "total_failures": total_failures,
        "avg_retries": round(avg_retries, 2),
    }


# ---------------------------------------------------------------------------
# Human-readable formatter
# ---------------------------------------------------------------------------

def format_event_summary(events: list[dict]) -> str:
    """Format a list of pipeline events into a human-readable terminal summary.

    Handles event types: ``graph_node_enter``, ``graph_node_exit``,
    ``graph_route``, ``chat_model_start``, ``llm_end``, ``llm_error``,
    ``block_start``, ``block_phase``.

    Args:
        events: List of event dicts as returned by :func:`read_events`.

    Returns:
        Multi-line string with ANSI-coloured output.
    """
    lines: list[str] = []
    for ev in events:
        ts_str = ev.get("iso", "")
        etype = ev.get("event", "")

        if etype == "graph_node_enter":
            node = ev.get("node", "?")
            extra = ""
            # Show answer status for Gather Requirements node
            if node == "Gather Requirements" and "has_human_response" in ev:
                has_hr = ev.get("has_human_response", False)
                has_ans = ev.get("has_answers_key", False)
                if has_hr and has_ans:
                    extra = f"  {GREEN}(with answers){RESET}"
                elif has_hr:
                    extra = f"  {RED}(human_response present but NO answers key!){RESET}"
                else:
                    extra = f"  {DIM}(initial run, no answers yet){RESET}"
            lines.append(f"{DIM}{ts_str}{RESET}  {CYAN}>> {node}{RESET}{extra}")

        elif etype == "graph_node_exit":
            node = ev.get("node", "?")
            elapsed = ev.get("elapsed_s")
            suffix = f"  {DIM}({elapsed:.2f}s){RESET}" if elapsed is not None else ""
            # Show answer info for Escalate PRD exit
            if node == "Escalate PRD" and "has_answers" in ev:
                has_ans = ev.get("has_answers", False)
                count = ev.get("answer_count", 0)
                if has_ans:
                    suffix += f"  {GREEN}(received {count} answers){RESET}"
                else:
                    suffix += f"  {RED}(NO ANSWERS in response!){RESET}"

            # Show pass/fail status for tool stages
            if "clean" in ev:
                clean = ev.get("clean", False)
                suffix += f"  {GREEN}CLEAN{RESET}" if clean else f"  {RED}ERRORS{RESET}"
            if "passed" in ev:
                passed = ev.get("passed", False)
                suffix += f"  {GREEN}PASSED{RESET}" if passed else f"  {RED}FAILED{RESET}"
            if ev.get("success") is not None and node == "Synthesize":
                ok = ev.get("success", False)
                gc = ev.get("gate_count", 0)
                area = ev.get("chip_area_um2", 0.0)
                if ok:
                    area_str = f", {area:,.1f} µm²" if area else ""
                    suffix += f"  {GREEN}{gc} gates{area_str}{RESET}"
                else:
                    suffix += f"  {RED}FAILED{RESET}"
            if ev.get("has_llm_error"):
                suffix += f"  {RED}LLM ERROR{RESET}"
            if ev.get("category"):
                cat = ev.get("category", "")
                conf = ev.get("confidence", "")
                suffix += f"  {YELLOW}[{cat}]{RESET}"
                if conf:
                    suffix += f" {DIM}conf={conf}{RESET}"

            lines.append(f"{DIM}{ts_str}{RESET}  {GREEN}<< {node}{RESET}{suffix}")

            # Show tool stdout for failed stages (truncated)
            tool_out = ev.get("tool_stdout", "")
            if tool_out and (ev.get("clean") is False or ev.get("passed") is False
                            or ev.get("success") is False):
                # Show last 3 lines of tool output
                out_lines = [line.strip() for line in tool_out.strip().split("\n") if line.strip()]
                for ol in out_lines[-3:]:
                    lines.append(f"{DIM}{ts_str}{RESET}    {RED}{ol[:120]}{RESET}")

            # Show error preview for RTL generation failures
            error_preview = ev.get("error_preview", "")
            if error_preview and (ev.get("has_error") or ev.get("has_llm_error")):
                lines.append(f"{DIM}{ts_str}{RESET}    {RED}{error_preview[:150]}{RESET}")

            # Show diagnosis preview
            diag_preview = ev.get("diagnosis_preview", "")
            if diag_preview:
                lines.append(f"{DIM}{ts_str}{RESET}    {YELLOW}Diagnosis: {diag_preview[:150]}{RESET}")

        elif etype == "graph_route":
            src = ev.get("from", "?")
            dst = ev.get("to", "?")
            lines.append(
                f"{DIM}{ts_str}{RESET}  {YELLOW}-> {src} => {dst}{RESET}"
            )

        elif etype == "chat_model_start":
            model = ev.get("model", "?")
            run_name = ev.get("run_name", "")
            msg_count = ev.get("message_count", 0)
            total_chars = ev.get("total_chars", 0)
            label = f"  {DIM}{run_name}{RESET}" if run_name else ""
            lines.append(
                f"{DIM}{ts_str}{RESET}  {CYAN}[Chat] {model}{RESET}{label}"
                f"  {DIM}({msg_count} msgs, {total_chars} chars){RESET}"
            )

        elif etype == "llm_end":
            elapsed = ev.get("elapsed_s")
            output_chars = ev.get("output_chars", 0)
            elapsed_str = f"{elapsed:.2f}s" if elapsed is not None else "?"
            lines.append(
                f"{DIM}{ts_str}{RESET}  {GREEN}[LLM] done{RESET}"
                f"  {DIM}{elapsed_str}, {output_chars} chars out{RESET}"
            )

        elif etype == "llm_error":
            elapsed = ev.get("elapsed_s")
            error = ev.get("error", "")[:120]
            elapsed_str = f"{elapsed:.2f}s" if elapsed is not None else "?"
            lines.append(
                f"{DIM}{ts_str}{RESET}  {RED}[LLM] ERROR{RESET}"
                f"  {DIM}{elapsed_str}{RESET}  {RED}{error}{RESET}"
            )

        elif etype == "llm_empty_response":
            model = ev.get("model", "?")
            stderr = ev.get("stderr", "")[:120]
            error = ev.get("error", "")[:150]
            lines.append(
                f"{DIM}{ts_str}{RESET}  {RED}[LLM] EMPTY RESPONSE{RESET}"
                f"  {RED}model={model}{RESET}"
            )
            if stderr:
                lines.append(
                    f"{DIM}{ts_str}{RESET}    {RED}stderr: {stderr}{RESET}"
                )
            if error:
                lines.append(
                    f"{DIM}{ts_str}{RESET}    {RED}{error}{RESET}"
                )

        elif etype == "block_start":
            block = ev.get("block", ev.get("block_name", "?"))
            attempt = ev.get("attempt", "?")
            lines.append(
                f"{DIM}{ts_str}{RESET}  {CYAN}=== Block: {block} (attempt {attempt}) ==={RESET}"
            )

        elif etype == "block_phase":
            phase = ev.get("phase", "?")
            block = ev.get("block_name", "")
            suffix = f"  {DIM}[{block}]{RESET}" if block else ""
            lines.append(
                f"{DIM}{ts_str}{RESET}  {YELLOW}--- {phase} ---{RESET}{suffix}"
            )

        elif etype == "escalation_response":
            node = ev.get("node", "?")
            action = ev.get("action", "?")
            has_answers = ev.get("has_answers", False)
            answer_keys = ev.get("answer_keys", [])
            status = f"{GREEN}answers={len(answer_keys)}{RESET}" if has_answers else f"{RED}NO ANSWERS{RESET}"
            lines.append(
                f"{DIM}{ts_str}{RESET}  {CYAN}[ESCALATION]{RESET} "
                f"Received response from outer agent: action={action}, {status}"
            )
            if has_answers and answer_keys:
                keys_str = ", ".join(answer_keys[:5])
                if len(answer_keys) > 5:
                    keys_str += f", ... +{len(answer_keys) - 5} more"
                lines.append(
                    f"{DIM}{ts_str}{RESET}    {DIM}answer keys: {keys_str}{RESET}"
                )

        elif etype == "escalation_warning":
            warning = ev.get("warning", "")
            hint = ev.get("hint", "")
            lines.append(
                f"{DIM}{ts_str}{RESET}  {RED}[WARNING]{RESET} {warning}"
            )
            if hint:
                lines.append(
                    f"{DIM}{ts_str}{RESET}    {YELLOW}HINT: {hint}{RESET}"
                )

        else:
            # Unknown event type -- render generically
            lines.append(f"{DIM}{ts_str}  [{etype}] {ev}{RESET}")

    return "\n".join(lines)
