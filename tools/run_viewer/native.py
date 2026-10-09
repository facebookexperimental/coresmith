"""Native provider trajectories: Claude Code session files and stream-json
transcripts, Codex rollouts and ``exec --json`` streams.

Records are data, never instructions. Content is exported by allowlist:
visible text, tool inputs, tool results, attachments of known kinds and small
lifecycle events. Hidden reasoning, signatures and encrypted payloads become
length markers (:mod:`privacy`). Long text is never cut here; the UI previews
it with an explicit full-detail control.

Deduplication uses stable identities only:

* Claude records by ``uuid`` (the native file, the invocation stream and an
  engine live-stream capture carry the same uuid for the same record);
* Codex response items by item id (``AgentMessage`` = response ``message``,
  ``Reasoning`` = response ``reasoning``); user messages by text within one
  rollout;
* a Codex ``exec --json`` item (invocation transcript, engine turn log, live
  stream) is reconciled with the rollout item it copies by its output/text
  digest, in order; an item with no rollout counterpart is kept and flagged
  ``exec-stream only``. Codex exec item ids (``item_0``...) restart per call
  and are never used across calls.
"""
from __future__ import annotations

import collections
import hashlib
import json
import re
import shlex
from pathlib import Path

from .privacy import Ledger, reasoning_marker, redact, scrub
from .sources import Sources, iso, parse_iso, read_jsonl, ref

NOISE_ATTACHMENTS = frozenset({"total_tokens_reminder", "deferred_tools_record", "deferred_tools_delta", "environment",
                               "silent_turn_reminder", "agent_listing_delta", "skill_listing", "remote_session_change",
                               "session_context", "auto_mode", "model", "date", "thinking_drop"})
OMITTED_ATTACHMENTS = frozenset({"credential_org"})          # never exported, only counted
SKIP_CLAUDE_TYPES = frozenset({"queue-operation", "last-prompt", "atis-latch", "file-history-snapshot"})
SHELL_TOOLS = frozenset({"Bash", "Monitor"})   # Claude tools whose input is a shell command


def _h(text) -> str:
    if not isinstance(text, str):
        text = json.dumps(text, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode("utf-8", "surrogateescape")).hexdigest()[:20]


def _json_text(value) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=1, ensure_ascii=False, default=str)


class Agent:
    """One trajectory (Architect, native sub-agent, engine helper)."""

    def __init__(self, arm: str, aid: str, *, kind: str, label: str, provider: str):
        self.arm, self.id, self.kind, self.label, self.provider = arm, aid, kind, label, provider
        self.model_requested = None
        self.models = collections.Counter()
        self.session_id = None
        self.agent_key = None
        self.parent = None              # {agent, via, tool_use_id, label, basis}
        self.children: list[str] = []
        self.sources: list[str] = []
        self.turns: list[dict] = []
        self.notes: list[str] = []
        self.meta: dict = {}
        self.usage_snapshots: list[dict] = []
        self.tasks: dict[str, dict] = {}        # Claude background task id -> lifecycle
        self.engine_call = None                 # helper: the llm_calls record summary
        self._uuid: dict[str, list[dict]] = {}
        self._key: dict[str, list[dict]] = {}
        self._msg_usage: dict[str, dict] = {}
        self._order = 0

    def add(self, **t) -> dict:
        self._order += 1
        t["flags"] = list(t.get("flags") or [])        # never share list objects between turns
        t["refs"] = list(t.get("refs") or [])
        t["meta"] = dict(t.get("meta") or {})
        t["_o"] = self._order
        if "text" in t:
            t["text"] = t["text"] if isinstance(t["text"], str) else _json_text(t["text"])
            t["len"] = len(t["text"])
        self.turns.append(t)
        return t

    def note(self, text: str) -> None:
        if text not in self.notes:
            self.notes.append(text)

    def seen_again(self, turns: list[dict], r: dict) -> None:
        for t in turns:
            if r not in t["refs"]:
                t["refs"].append(r)


# ====================================================================== Claude
def _claude_blocks_text(content, ledger: Ledger | None = None) -> str:
    content = scrub(content, ledger)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if not isinstance(b, dict):
                parts.append(str(b))
            elif b.get("type") == "text":
                parts.append(str(b.get("text", "")))
            elif b.get("type") == "image":
                src = b.get("source") or {}
                parts.append(f"[image block not exported: {len(str(src.get('data', '')))} base64 characters]")
            elif b.get("type") == "tool_reference":
                parts.append(f"[tool reference: {b.get('tool_name')}]")
            else:
                parts.append(_json_text(b))
        return "\n".join(parts)
    return _json_text(content)


_BASH_META = ("interrupted", "backgroundTaskId", "returnCodeInterpretation", "timedOutAfterMs", "noOutputExpected",
              "isImage", "persistedOutputPath", "persistedOutputSize", "taskId", "timeoutMs", "persistent")
_AGENT_META = ("agentId", "status", "resolvedModel", "isAsync", "description", "outputFile", "totalDurationMs",
               "totalTokens", "totalToolUseCount", "task_id", "task_type")


def claude_record(agent: Agent, rec: dict, r: dict, ledger: Ledger, *, flags=()) -> list[dict]:
    """Turns for one Claude ``user`` / ``assistant`` / ``attachment`` /
    ``system`` record (native file or stream). Returns new turns; a record
    whose uuid was already imported only gains a reference."""
    uuid = rec.get("uuid")
    if uuid and uuid in agent._uuid:
        agent.seen_again(agent._uuid[uuid], r)
        return []
    rtype = rec.get("type")
    ts = parse_iso(rec.get("timestamp"))
    base = {"ts": ts, "uuid": uuid, "refs": [r], "flags": list(flags)}
    if rec.get("isSidechain"):
        base["flags"].append("sidechain")
    out: list[dict] = []
    msg = rec.get("message") if isinstance(rec.get("message"), dict) else {}
    content = msg.get("content")
    if rtype == "assistant":
        mid = msg.get("id")
        model = msg.get("model")
        if model and model != "<synthetic>":
            agent.models[str(model)] += 1
        usage = msg.get("usage")
        first = bool(mid) and mid not in agent._msg_usage
        if first and isinstance(usage, dict):
            agent._msg_usage[mid] = {k: usage.get(k) for k in ("input_tokens", "output_tokens",
                                                               "cache_read_input_tokens", "cache_creation_input_tokens")}
        meta = {"msg_id": mid, "model": model, "stop_reason": msg.get("stop_reason")}
        if rec.get("error") or rec.get("isApiErrorMessage"):
            base["flags"].append("api_error")
            meta["error"] = redact(str(rec.get("error") or ""), ledger)
        blocks = content if isinstance(content, list) else [{"type": "text", "text": content or ""}]
        for b in blocks:
            if not isinstance(b, dict):
                continue
            bt = b.get("type")
            if bt == "text":
                ledger.public_text(agent.id, b.get("text") or "")
                out.append(agent.add(kind="assistant", role="assistant", text=redact(b.get("text") or "", ledger),
                                     meta=dict(meta), **base))
            elif bt in ("thinking", "redacted_thinking"):
                raw = b.get("thinking") if bt == "thinking" else b.get("data")
                out.append(agent.add(kind="thinking", role="assistant",
                                     text=reasoning_marker(ledger, "claude_" + bt, raw, b.get("signature"),
                                                           scope=agent.id if bt == "thinking" else None),
                                     meta={"msg_id": mid, "block": bt}, **base))
            elif bt == "tool_use":
                inp = b.get("input")
                tmeta = dict(meta)
                if isinstance(inp, dict):
                    for k in ("description", "run_in_background", "timeout", "subagent_type", "model", "to",
                              "timeout_ms", "persistent"):
                        if k in inp:
                            tmeta[k] = inp[k]
                if b.get("name") in SHELL_TOOLS and isinstance(inp, dict) and isinstance(inp.get("command"), str):
                    text = redact(inp["command"], ledger)      # the command itself; other inputs are in meta
                else:
                    text = _json_text(scrub(inp, ledger))
                out.append(agent.add(kind="tool_use", role="assistant", name=b.get("name"), tid=b.get("id"),
                                     text=text, meta=tmeta, **base))
            else:
                out.append(agent.add(kind="event", role="assistant", name=bt, text=_json_text(scrub(b, ledger)),
                                     meta=dict(meta), **base))
        if out and first and isinstance(usage, dict):
            out[0]["meta"]["usage"] = agent._msg_usage[mid]
    elif rtype == "user":
        origin = (rec.get("origin") or {}).get("kind") if isinstance(rec.get("origin"), dict) else None
        meta = {}
        if origin:
            meta["origin"] = origin
        if rec.get("isCompactSummary"):
            base["flags"].append("compact_summary")
        if rec.get("isMeta"):
            base["flags"].append("meta")
        tur = rec.get("toolUseResult") if isinstance(rec.get("toolUseResult"), dict) else None
        if isinstance(content, str):
            out.append(agent.add(kind="user", role="user", text=redact(content, ledger), meta=meta, **base))
        elif isinstance(content, list):
            for b in content:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "tool_result":
                    rmeta = dict(meta)
                    if tur:
                        for k in _BASH_META + _AGENT_META:
                            if k in tur and not isinstance(tur[k], (dict, list)):
                                rmeta[k] = tur[k]
                    out.append(agent.add(kind="tool_result", role="tool", tid=b.get("tool_use_id"),
                                         err=bool(b.get("is_error")),
                                         text=redact(_claude_blocks_text(b.get("content"), ledger), ledger),
                                         meta=rmeta, **base))
                elif b.get("type") == "text":
                    out.append(agent.add(kind="user", role="user", text=redact(b.get("text") or "", ledger),
                                         meta=dict(meta), **base))
                else:
                    out.append(agent.add(kind="user", role="user", text=redact(_claude_blocks_text([b], ledger), ledger),
                                         meta=dict(meta), **base))
    elif rtype == "attachment":
        att = rec.get("attachment") if isinstance(rec.get("attachment"), dict) else {}
        atype = str(att.get("type") or "")
        if atype in OMITTED_ATTACHMENTS:
            ledger.counts["omitted_attachment:" + atype] += 1
            return []
        rendered = rec.get("rendered")
        if rendered:
            text = "\n".join(str(x.get("content", "")) if isinstance(x, dict) else str(x) for x in rendered)
        elif atype == "queued_command":
            text = _claude_blocks_text(att.get("prompt"), ledger)
        else:
            text = _json_text(scrub({k: v for k, v in att.items() if k != "type"}, ledger))
        if atype in NOISE_ATTACHMENTS:
            base["flags"].append("noise")
        out.append(agent.add(kind="attachment", role="system", name=atype, text=redact(text, ledger), **base))
    elif rtype == "system":
        sub = rec.get("subtype") or ""
        payload = {k: v for k, v in rec.items() if k not in ("type", "uuid", "parentUuid", "timestamp", "sessionId",
                                                            "session_id", "cwd", "version", "gitBranch", "userType",
                                                            "isSidechain", "entrypoint", "slug")}
        if sub == "thinking_tokens":
            return []
        out.append(agent.add(kind="event", role="system", name=sub or "system", text=_json_text(scrub(payload, ledger)),
                             **base))
    else:
        return []
    if uuid:
        agent._uuid.setdefault(uuid, []).extend(out)
    return out


def parse_claude_session(agent: Agent, path: Path, sources: Sources, ledger: Ledger, role: str) -> str:
    """A native Claude Code session file (main, sub-agent or engine helper)."""
    sid = sources.add(path, role, agent.arm)
    agent.sources.append(sid)
    for line, rec, raw in read_jsonl(path):
        sources.stat(sid, "records")
        if rec is None:
            sources.stat(sid, "unparseable")
            continue
        rtype = rec.get("type")
        if rec.get("sessionId") and not agent.session_id:
            agent.session_id = rec.get("sessionId")
        if rec.get("agentId") and not agent.agent_key:
            agent.agent_key = rec.get("agentId")
        if rtype in SKIP_CLAUDE_TYPES:
            sources.skip(sid, rtype)
            continue
        if rtype == "cost-state":
            snap = {k: rec.get(k) for k in ("totalCostUSD", "totalAPIDuration", "totalToolDuration", "totalDuration",
                                             "startTime", "hasUnknownModelCost", "totalLinesAdded", "totalLinesRemoved")}
            snap["modelUsage"] = scrub(rec.get("modelUsage") or {}, ledger)
            snap["src"] = ref(sid, line)
            snap["scope"] = "native session cumulative counter (never summed)"
            agent.usage_snapshots.append(snap)
            sources.skip(sid, "cost-state (usage snapshot)")
            continue
        if rtype == "system" and rec.get("subtype") == "thinking_tokens":
            sources.skip(sid, "system/thinking_tokens")
            continue
        uuid = rec.get("uuid")
        before = len(agent.turns)
        if uuid and uuid in agent._uuid:
            agent.seen_again(agent._uuid[uuid], ref(sid, line))
            sources.stat(sid, "duplicates")
            continue
        new = claude_record(agent, rec, ref(sid, line), ledger)
        if new:
            sources.stat(sid, "kept", len(new))
        elif len(agent.turns) == before:
            sources.skip(sid, f"{rtype}" + (f"/{rec.get('subtype')}" if rec.get("subtype") else "")
                         + (" (" + str((rec.get("attachment") or {}).get("type")) + ")" if rtype == "attachment" else ""))
    return sid


_TASK_SUBTYPES = ("task_started", "task_notification", "task_updated", "task_progress", "background_tasks_changed")


def parse_claude_stream(agent: Agent, path: Path, sources: Sources, ledger: Ledger, role: str, *,
                        route=None, invocation: dict | None = None) -> str:
    """A ``claude --print --output-format stream-json`` transcript (Architect
    invocation or an engine live-stream capture given as ``(line, event)``
    pairs via ``path`` being a list). ``route(parent_tool_use_id)`` returns
    the sub-agent owning a forwarded record."""
    if isinstance(path, tuple):
        lines_iter = path[1]
        sid = path[0]
    else:
        sid = sources.add(path, role, agent.arm)
        lines_iter = ((ln, rec) for ln, rec, _ in read_jsonl(path))
    agent.sources.append(sid)
    last_ts = None
    thinking_buf: dict = {}         # (parent_tool_use_id, message id, block index) -> streamed thinking text
    stream_msg: dict = {}           # parent_tool_use_id -> message id of the streaming message
    for line, rec in lines_iter:
        if not isinstance(path, tuple):
            sources.stat(sid, "records")
        if rec is None:
            sources.stat(sid, "unparseable")
            continue
        rtype, sub = rec.get("type"), rec.get("subtype")
        r = ref(sid, line)
        if rtype == "stream_event":
            ev = rec.get("event") or {}
            d = ev.get("delta") or {}
            if ev.get("type") == "message_start":
                stream_msg[rec.get("parent_tool_use_id")] = (ev.get("message") or {}).get("id")
            if d.get("type") == "thinking_delta":
                ptu_ = rec.get("parent_tool_use_id")
                key = (ptu_, stream_msg.get(ptu_), ev.get("index"))
                thinking_buf[key] = thinking_buf.get(key, "") + (d.get("thinking") or "")
            elif d.get("type") == "signature_delta":
                ledger.private("stream_signature_delta", d.get("signature") or "")
            sources.skip(sid, "stream_event (partial deltas; the complete record is imported)")
            continue
        if rtype in ("user", "assistant"):
            target = agent
            ptu = rec.get("parent_tool_use_id")
            if ptu and route is not None:
                target = route(ptu) or agent
                if target is agent:
                    sources.skip(sid, "forwarded record of an unknown sub-agent (kept on the parent)")
            uuid = rec.get("uuid")
            if uuid and uuid in target._uuid:
                target.seen_again(target._uuid[uuid], r)
                sources.stat(sid, "duplicates")
                if rec.get("timestamp"):
                    last_ts = parse_iso(rec.get("timestamp"))
                continue
            new = claude_record(target, rec, r, ledger, flags=("forwarded",) if target is not agent else ())
            sources.stat(sid, "kept", len(new))
            if rec.get("timestamp"):
                last_ts = parse_iso(rec.get("timestamp"))
            continue
        if rtype == "system" and sub == "init":
            if rec.get("session_id") and not agent.session_id:
                agent.session_id = rec.get("session_id")
            if rec.get("model"):
                agent.meta.setdefault("init_model", rec.get("model"))
            summary = {k: rec.get(k) for k in ("model", "claude_code_version", "permissionMode", "cwd", "output_style")}
            summary["tools"] = len(rec.get("tools") or [])
            summary["mcp_servers"] = len(rec.get("mcp_servers") or [])
            agent.add(kind="event", role="system", name="init", text=_json_text(summary), ts=last_ts, refs=[r],
                      flags=["ts_estimated"])
            sources.stat(sid, "kept")
            if invocation is not None:
                invocation["init"] = summary
            continue
        if rtype == "result":
            summary = {k: rec.get(k) for k in ("subtype", "is_error", "num_turns", "duration_ms", "duration_api_ms",
                                               "terminal_reason", "api_error_status", "session_id", "total_cost_usd")}
            summary["result"] = redact(str(rec.get("result") or ""), ledger)
            summary["cost_scope"] = "cumulative over the native session lineage (not an increment)"
            usage = scrub(rec.get("modelUsage") or {}, ledger)
            agent.add(kind="event", role="system", name="result", text=_json_text({**summary, "modelUsage": usage}),
                      ts=last_ts, refs=[r], flags=["ts_estimated"], meta={"is_error": rec.get("is_error"),
                                                                         "subtype": rec.get("subtype")})
            agent.usage_snapshots.append({"src": r, "total_cost_usd": rec.get("total_cost_usd"),
                                          "modelUsage": usage, "scope": summary["cost_scope"],
                                          "num_turns": rec.get("num_turns")})
            sources.stat(sid, "kept")
            if invocation is not None:
                invocation.setdefault("results", []).append({"src": r, **{k: summary[k] for k in summary if k != "result"},
                                                             "result_len": len(summary["result"])})
            continue
        if rtype == "system" and sub in _TASK_SUBTYPES:
            tid = rec.get("task_id")
            if tid:
                task = agent.tasks.setdefault(tid, {"task_id": tid, "events": []})
                for k in ("tool_use_id", "description", "task_type", "subagent_type", "is_backgrounded", "status",
                          "output_file", "summary"):
                    if rec.get(k) not in (None, ""):
                        task[k] = rec.get(k)
                if sub == "task_updated":
                    patch = rec.get("patch") or {}
                    if patch.get("status"):
                        task["status"] = patch["status"]
                    if patch.get("end_time"):
                        task["end_ts"] = float(patch["end_time"]) / 1000.0
                task["events"].append({"subtype": sub, "ts_est": last_ts, "src": r})
            if sub in ("task_progress",):
                sources.skip(sid, "system/task_progress (summarised on the task)")
                continue
            if sub == "task_started" and rec.get("prompt"):
                payload = {k: rec.get(k) for k in ("task_id", "tool_use_id", "description", "task_type", "subagent_type",
                                                   "is_backgrounded", "spawn_depth")}
                payload["prompt"] = redact(rec.get("prompt"), ledger)
            else:
                payload = {k: v for k, v in rec.items() if k not in ("type", "subtype", "uuid", "session_id")}
            agent.add(kind="event", role="system", name=sub, text=_json_text(scrub(payload, ledger)), ts=last_ts,
                      refs=[r], tid=rec.get("tool_use_id"), flags=["ts_estimated"] + (["noise"] if sub in (
                          "background_tasks_changed", "task_updated") else []))
            sources.stat(sid, "kept")
            continue
        if rtype == "system" and sub in ("thinking_tokens",):
            sources.skip(sid, "system/thinking_tokens")
            continue
        if rtype in ("tool_progress",):
            sources.skip(sid, "tool_progress (heartbeat)")
            continue
        if rtype == "system" and sub == "status":
            sources.skip(sid, "system/status (request lifecycle)")
            continue
        if rtype == "rate_limit_event":
            info = rec.get("rate_limit_info") or {}
            safe = {k: info.get(k) for k in ("status", "rateLimitType", "resetsAt", "isUsingOverage")}
            agent.add(kind="event", role="system", name="rate_limit_event", text=_json_text(safe), ts=last_ts,
                      refs=[r], flags=["ts_estimated", "noise"])
            sources.stat(sid, "kept")
            continue
        if rtype == "system":
            payload = {k: v for k, v in rec.items() if k not in ("type", "uuid", "session_id")}
            if sub == "commands_changed":
                payload = {"commands": len(rec.get("commands") or [])}
            agent.add(kind="event", role="system", name=sub or "system", text=_json_text(scrub(payload, ledger)),
                      ts=last_ts, refs=[r], flags=["ts_estimated"] + (["noise"] if sub == "commands_changed" else []))
            sources.stat(sid, "kept")
            continue
        sources.skip(sid, f"{rtype}/{sub}" if sub else str(rtype))
    for (ptu_, _mid, _idx), text in thinking_buf.items():
        owner = (route(ptu_) if (ptu_ and route is not None) else None) or agent
        ledger.private_text("stream_thinking_delta", text, owner.id)
    return sid


# ====================================================================== Codex
def _codex_text(content, ledger: Ledger) -> str:
    if isinstance(content, str):
        return redact(content, ledger)
    parts = []
    for c in content or []:
        if not isinstance(c, dict):
            parts.append(str(c))
        elif c.get("type") in ("input_text", "output_text", "text", "Text"):
            parts.append(redact(str(c.get("text", "")), ledger))
        elif c.get("type") == "encrypted_content" or "encrypted_content" in c:
            n = ledger.private("codex_encrypted_part", c.get("encrypted_content") or "")
            parts.append(f"[encrypted payload not exported: {n} characters]")
        elif c.get("type") in ("input_image", "image"):
            parts.append("[image not exported]")
        else:
            parts.append(_json_text(scrub(c, ledger)))
    return "\n".join(parts)


def _cmd_text(cmd) -> str:
    if isinstance(cmd, list):
        if len(cmd) == 3 and cmd[1] in ("-lc", "-c") and str(cmd[0]).endswith("sh"):
            return str(cmd[2])
        return " ".join(shlex.quote(str(x)) for x in cmd)
    return str(cmd or "")


def _exec_cmd_text(cmd: str) -> str:
    """``/bin/bash -lc '...'`` from an exec stream, as the inner script."""
    try:
        toks = shlex.split(cmd)
    except ValueError:
        return cmd
    if len(toks) == 3 and toks[1] in ("-lc", "-c") and toks[0].endswith("sh"):
        return toks[2]
    return cmd


def _duration_s(d) -> float | None:
    if isinstance(d, dict) and ("secs" in d or "nanos" in d):
        return float(d.get("secs") or 0) + float(d.get("nanos") or 0) / 1e9
    if isinstance(d, (int, float)):
        return float(d)
    return None


def _file_changes_text(changes, ledger: Ledger) -> tuple[str, list[str]]:
    paths = []
    parts = []
    if isinstance(changes, dict):
        items = changes.items()
    elif isinstance(changes, list):
        items = [(c.get("path"), c) for c in changes if isinstance(c, dict)]
    else:
        items = []
    for path, ch in items:
        paths.append(str(path))
        if isinstance(ch, dict):
            kind = ch.get("type") or ch.get("kind")
            body = ch.get("unified_diff") or ch.get("content") or ch.get("diff") or ""
            mv = ch.get("move_path")
            parts.append(f"--- {kind}: {path}" + (f" -> {mv}" if mv else "") + "\n" + redact(str(body), ledger))
        else:
            parts.append(f"--- {path}")
    return "\n\n".join(parts), sorted(paths)


def parse_codex_rollout(agent: Agent, path: Path, sources: Sources, ledger: Ledger, role: str) -> str:
    sid = sources.add(path, role, agent.arm)
    agent.sources.append(sid)
    turn_scope = "initial"
    for line, rec, raw in read_jsonl(path):
        sources.stat(sid, "records")
        if rec is None:
            sources.stat(sid, "unparseable")
            continue
        r = ref(sid, line)
        ts = parse_iso(rec.get("timestamp"))
        rtype = rec.get("type")
        pl = rec.get("payload") if isinstance(rec.get("payload"), dict) else {}
        pt = pl.get("type")
        if (rtype == "event_msg" and pt == "task_started") or rtype == "turn_context":
            turn_scope = pl.get("turn_id") or pl.get("root_turn_id") or turn_scope
        user_scope = pl.get("turn_id") or turn_scope
        base = {"ts": ts, "refs": [r]}

        def keyed(key, **t):
            if key and key in agent._key:
                agent.seen_again(agent._key[key], r)
                sources.stat(sid, "duplicates")
                return None
            turn = agent.add(**t, **base)
            if key:
                agent._key.setdefault(key, []).append(turn)
                turn["key"] = key
            sources.stat(sid, "kept")
            return turn

        if rtype == "session_meta":
            src = pl.get("source") if isinstance(pl.get("source"), dict) else {}
            spawn = ((src.get("subagent") or {}).get("thread_spawn") or {}) if isinstance(src, dict) else {}
            meta = {"thread_id": pl.get("id"), "session_id": pl.get("session_id"),
                    "forked_from_id": pl.get("forked_from_id"),
                    "parent_thread_id": pl.get("parent_thread_id") or spawn.get("parent_thread_id"),
                    "agent_path": pl.get("agent_path") or spawn.get("agent_path"),
                    "agent_nickname": pl.get("agent_nickname") or spawn.get("agent_nickname"),
                    "depth": spawn.get("depth"), "thread_source": pl.get("thread_source"),
                    "originator": pl.get("originator"), "cli_version": pl.get("cli_version"), "cwd": pl.get("cwd"),
                    "source": pl.get("source") if isinstance(pl.get("source"), str) else "subagent",
                    "started": pl.get("timestamp"), "history_mode": pl.get("history_mode")}
            agent.meta.update({k: v for k, v in meta.items() if v is not None})
            if pl.get("id") and not agent.session_id:
                agent.session_id = pl.get("id")
            keyed(f"meta:{pl.get('id')}", kind="event", role="system", name="session_meta",
                  text=_json_text({k: v for k, v in meta.items() if v is not None}))
            if pl.get("base_instructions"):
                bi = pl.get("base_instructions")
                bi = bi.get("text") if isinstance(bi, dict) else bi
                keyed(f"bi:{_h(bi)}", kind="system", role="system", name="base_instructions",
                      text=redact(_json_text(bi), ledger), flags=["noise"])
        elif rtype == "turn_context":
            if pl.get("model"):
                agent.models[str(pl.get("model"))] += 1
            summary = {k: pl.get(k) for k in ("turn_id", "cwd", "approval_policy", "model", "effort", "summary")}
            summary["sandbox_policy"] = (pl.get("sandbox_policy") or {}).get("type")
            keyed(None, kind="event", role="system", name="turn_context", text=_json_text(summary), flags=["noise"])
        elif rtype == "world_state":
            st = pl.get("state") if isinstance(pl.get("state"), dict) else {}
            keyed(None, kind="event", role="system", name="world_state",
                  text=_json_text({"full": pl.get("full"), "keys": sorted(str(k) for k in st)}), flags=["noise"])
        elif rtype == "token_usage_record":
            agent.usage_snapshots.append({"ts": ts, "src": r, "response_id": pl.get("response_id"),
                                          "turn_id": pl.get("turn_id"), "usage": pl.get("usage"),
                                          "thread_token_usage": pl.get("thread_token_usage"),
                                          "kind": "token_usage_record"})
            sources.skip(sid, "token_usage_record (usage snapshot)")
        elif rtype == "compacted":
            hist = pl.get("replacement_history") or []
            for it in hist if isinstance(hist, list) else []:
                if isinstance(it, dict) and it.get("encrypted_content"):
                    ledger.private("codex_compacted_encrypted", it.get("encrypted_content"))
            keyed(None, kind="event", role="system", name="compacted",
                  text=_json_text({"window_number": pl.get("window_number"), "window_id": pl.get("window_id"),
                                   "previous_window_id": pl.get("previous_window_id"),
                                   "replacement_history_items": len(hist) if isinstance(hist, list) else None,
                                   "message": redact(str(pl.get("message") or ""), ledger)}))
        elif rtype == "inter_agent_communication_metadata":
            sources.skip(sid, "inter_agent_communication_metadata")
        elif rtype == "response_item":
            if pt == "message":
                role_ = pl.get("role") or "unknown"
                if role_ == "assistant" and pl.get("channel") == "analysis":
                    keyed(f"rs:{pl.get('id')}" if pl.get("id") else None,
                          kind="thinking", role="assistant",
                          text=reasoning_marker(ledger, "codex_analysis_message", pl.get("content") or ""))
                    continue
                text = _codex_text(pl.get("content"), ledger)
                kind = {"assistant": "assistant", "user": "user"}.get(role_, "system")
                key = f"msg:{pl.get('id')}" if role_ == "assistant" else (f"utext:{user_scope}:{_h(text)}" if role_ == "user"
                                                                         else f"dev:{pl.get('id') or _h(text)}")
                keyed(key, kind=kind, role=role_, text=text, flags=["noise"] if role_ == "developer" else [])
            elif pt in ("reasoning", "agent_reasoning", "agent_reasoning_raw_content", "reasoning_summary"):
                key = f"rs:{pl.get('id')}" if pl.get("id") else None
                if key and key in agent._key:
                    ledger.private("codex_reasoning_copy", pl.get("encrypted_content") or "")
                    agent.seen_again(agent._key[key], r)
                    sources.stat(sid, "duplicates")
                    continue
                raw = pl.get("summary") or pl.get("content") or pl.get("text")
                if isinstance(raw, list):
                    raw = "\n".join(str(x.get("text", "")) if isinstance(x, dict) else str(x) for x in raw)
                keyed(key, kind="thinking", role="assistant",
                      text=reasoning_marker(ledger, "codex_reasoning", raw or "", pl.get("encrypted_content")))
            elif pt in ("function_call", "custom_tool_call", "local_shell_call"):
                args = pl.get("arguments") if "arguments" in pl else (pl.get("input") or pl.get("action"))
                if isinstance(args, str) and pt == "function_call":
                    try:
                        args = json.loads(args)
                    except ValueError:
                        pass
                name = pl.get("name") or pt
                if pl.get("namespace"):
                    name = f"{pl.get('namespace')}.{name}"
                keyed(f"call:{pl.get('call_id') or pl.get('id')}", kind="tool_use", role="assistant", name=name,
                      tid=pl.get("call_id") or pl.get("id"), text=redact(_json_text(scrub(args, ledger)), ledger),
                      meta={"status": pl.get("status")})
            elif pt in ("function_call_output", "custom_tool_call_output", "local_shell_call_output"):
                outv = pl.get("output")
                text = _codex_text(outv, ledger) if isinstance(outv, list) else redact(_json_text(outv), ledger)
                keyed(f"out:{pl.get('call_id') or pl.get('id')}", kind="tool_result", role="tool",
                      tid=pl.get("call_id") or pl.get("id"), text=text)
            elif pt == "agent_message":
                keyed(f"amsg:{pl.get('id')}", kind="user", role="agent", name="inter-agent message",
                      text=_codex_text(pl.get("content"), ledger),
                      meta={"author": pl.get("author"), "recipient": pl.get("recipient")})
            else:
                keyed(None, kind="event", role="system", name=f"response_item/{pt}",
                      text=_json_text({"keys": sorted(pl.keys())}), flags=["unrecognised"])
                sources.skip(sid, f"unrecognised response_item/{pt} (keys only)")
        elif rtype == "event_msg":
            if pt == "item_completed":
                it = pl.get("item") if isinstance(pl.get("item"), dict) else {}
                itype = it.get("type")
                done = pl.get("completed_at_ms")
                if done:
                    base["ts"] = float(done) / 1000.0
                if itype == "CommandExecution":
                    dur = _duration_s(it.get("duration"))
                    cmd = _cmd_text(it.get("command"))
                    try:
                        exit_code = int(it.get("exit_code")) if it.get("exit_code") not in (None, "") else None
                    except (TypeError, ValueError):
                        exit_code = None
                    meta = {"cwd": it.get("cwd"), "duration_s": dur, "status": it.get("status"),
                            "process_id": it.get("process_id"), "source": it.get("source"),
                            "start_ts": (base["ts"] - dur) if (base["ts"] and dur is not None) else None,
                            "out_digest": _h(it.get("aggregated_output") or ""), "exit_code": exit_code}
                    t1 = keyed(f"cmd:{it.get('id')}", kind="tool_use", role="assistant", name="exec_command",
                               tid=it.get("id"), text=redact(cmd, ledger), meta=meta)
                    if t1 is not None:
                        t2 = agent.add(kind="tool_result", role="tool", tid=it.get("id"), exit=exit_code,
                                       err=exit_code not in (0, None), text=redact(it.get("aggregated_output") or "", ledger),
                                       meta={"out_digest": meta["out_digest"]}, **base)
                        agent._key[f"cmd:{it.get('id')}"].append(t2)
                elif itype == "FileChange":
                    text, paths = _file_changes_text(it.get("changes"), ledger)
                    keyed(f"fc:{it.get('id')}", kind="tool_use", role="assistant", name="file_change", tid=it.get("id"),
                          text=text, meta={"paths": paths, "status": it.get("status")})
                elif itype == "AgentMessage":
                    if it.get("channel") == "analysis" or it.get("phase") == "analysis":
                        keyed(f"rs:{it.get('id')}", kind="thinking", role="assistant",
                              text=reasoning_marker(ledger, "codex_analysis_item", it.get("content") or ""))
                        continue
                    keyed(f"msg:{it.get('id')}", kind="assistant", role="assistant",
                          text=_codex_text(it.get("content"), ledger), meta={"phase": it.get("phase")})
                elif itype == "UserMessage":
                    text = _codex_text(it.get("content"), ledger)
                    keyed(f"utext:{user_scope}:{_h(text)}", kind="user", role="user", text=text)
                elif itype == "Reasoning":
                    key = f"rs:{it.get('id')}"
                    raw = it.get("raw_content") or it.get("summary_text")
                    if isinstance(raw, list):
                        raw = "\n".join(str(x.get("text", x)) if isinstance(x, dict) else str(x) for x in raw)
                    if key in agent._key:
                        ledger.private("codex_reasoning_item", raw or "")
                        agent.seen_again(agent._key[key], r)
                        sources.stat(sid, "duplicates")
                        continue
                    keyed(key, kind="thinking", role="assistant", text=reasoning_marker(ledger, "codex_reasoning_item", raw or ""))
                elif itype == "SubAgentActivity":
                    keyed(f"sa:{it.get('id')}:{it.get('kind')}", kind="event", role="system", name="sub-agent activity",
                          tid=it.get("id"), text=_json_text({k: it.get(k) for k in ("kind", "agent_thread_id", "agent_path")}),
                          meta={"child_thread": it.get("agent_thread_id"), "activity": it.get("kind")})
                elif itype == "ContextCompaction":
                    keyed(f"cc:{it.get('id')}", kind="event", role="system", name="context compaction",
                          text=_json_text({"started_at_ms": pl.get("started_at_ms"), "completed_at_ms": done}))
                else:
                    safe = scrub({k: v for k, v in it.items() if k != "type"}, ledger)
                    keyed(f"item:{it.get('id')}", kind="event", role="system", name=str(itype), text=_json_text(safe),
                          flags=["noise"] if itype == "Extension" else [])
            elif pt == "token_count":
                info = pl.get("info") or {}
                agent.usage_snapshots.append({"ts": ts, "src": r, "kind": "token_count",
                                              "total_token_usage": info.get("total_token_usage"),
                                              "last_token_usage": info.get("last_token_usage")})
                sources.skip(sid, "event_msg/token_count (usage snapshot)")
            elif pt in ("task_started", "task_complete", "turn_aborted", "error", "thread_settings_applied",
                        "item_started", "item_updated", "exec_command_begin", "exec_command_end", "background_event"):
                safe = scrub({k: v for k, v in pl.items() if k not in ("type", "item", "thread_settings")}, ledger)
                if pt == "task_complete" and isinstance(safe.get("last_agent_message"), str):
                    safe["last_agent_message"] = redact(safe["last_agent_message"], ledger)
                keyed(None, kind="event", role="system", name=pt, text=_json_text(safe),
                      flags=["noise"] if pt in ("thread_settings_applied", "item_started", "item_updated") else [])
            else:
                keyed(None, kind="event", role="system", name=f"event_msg/{pt}",
                      text=_json_text({"keys": sorted(pl.keys())}), flags=["unrecognised"])
                sources.skip(sid, f"unrecognised event_msg/{pt} (keys only)")
        else:
            keyed(None, kind="event", role="system", name=f"{rtype}", text=_json_text({"keys": sorted(pl.keys())}),
                  flags=["unrecognised"])
            sources.skip(sid, f"unrecognised {rtype} (keys only)")
    return sid


def _exec_match_key(item: dict) -> str | None:
    t = item.get("type")
    if t == "command_execution":
        try:
            ec = int(item.get("exit_code")) if item.get("exit_code") is not None else None
        except (TypeError, ValueError):
            ec = None
        return f"cmd|{_h(item.get('aggregated_output') or '')}|{ec}"
    if t == "agent_message":
        return f"msg|{_h(item.get('text') or '')}"
    if t == "file_change":
        return "fc|" + "|".join(sorted(str(c.get("path")) for c in item.get("changes") or [] if isinstance(c, dict)))
    return None


def _rollout_pool(agent: Agent) -> dict[str, collections.deque]:
    pool: dict[str, collections.deque] = collections.defaultdict(collections.deque)
    for t in agent.turns:
        if t.get("_exec_only"):
            continue
        if t["kind"] == "tool_use" and t.get("name") == "exec_command":
            pool[f"cmd|{t['meta'].get('out_digest')}|{t['meta'].get('exit_code')}"].append(t)
        elif t["kind"] == "assistant":
            pool[f"msg|{_h(t['text'])}"].append(t)
        elif t["kind"] == "tool_use" and t.get("name") == "file_change":
            pool["fc|" + "|".join(t["meta"].get("paths") or [])].append(t)
    return pool


def reconcile_codex_exec(agent: Agent, events, sid: str, sources: Sources, ledger: Ledger, *,
                         call_scope: str, prior: dict | None = None) -> dict:
    """Fold a Codex ``exec --json`` stream (``[(line, ts, event)]``) into an
    agent already holding its rollout. Returns ``{event_hash: [turns]}`` so a
    later copy of the same stream (live capture) can be recognised by
    identical payload within the same call scope."""
    pool = _rollout_pool(agent)
    seen: dict[str, list[dict]] = {}
    started: dict[str, tuple] = {}
    prev_ts = None
    turn_no = 0
    thread = None
    stats = collections.Counter()
    for line, ts, ev in events:
        r = ref(sid, line)
        et = ev.get("type")
        h = _h({"scope": call_scope, "event": ev})
        if prior is not None and h in prior:
            agent.seen_again(prior[h], r)
            stats["copy_of_turn_log"] += 1
            continue
        if et == "thread.started":
            thread = ev.get("thread_id")
            if thread and not agent.session_id:
                agent.session_id = thread
            stats["thread.started"] += 1
            continue
        if et == "turn.started":
            turn_no += 1
            stats["turn.started"] += 1
            continue
        if et in ("item.started", "item.updated"):
            item = ev.get("item") or {}
            started[f"{call_scope}|t{turn_no}|{item.get('id')}"] = (ts, line)
            stats["item progress (merged into the completed item)"] += 1
            continue
        if et == "item.completed":
            item = ev.get("item") or {}
            ikey = f"{call_scope}|t{turn_no}|{item.get('id')}"
            mk = _exec_match_key(item)
            if mk and pool.get(mk):
                t = pool[mk].popleft()
                agent.seen_again([t], r)
                seen[h] = [t]
                stats["matched to rollout item"] += 1
                prev_ts = t.get("ts") or prev_ts
                continue
            t_ts = ts if ts is not None else prev_ts
            flags = ["exec_stream_only"] + ([] if ts is not None else ["ts_estimated"])
            itype = item.get("type")
            new = []
            if itype == "command_execution":
                try:
                    ec = int(item.get("exit_code")) if item.get("exit_code") is not None else None
                except (TypeError, ValueError):
                    ec = None
                st = started.get(ikey)
                meta = {"status": item.get("status"), "start_ts": st[0] if st else None, "exit_code": ec,
                        "out_digest": _h(item.get("aggregated_output") or "")}
                new.append(agent.add(kind="tool_use", role="assistant", name="exec_command", tid=ikey,
                                     text=redact(_exec_cmd_text(item.get("command") or ""), ledger), ts=t_ts, refs=[r],
                                     flags=list(flags), meta=meta, _exec_only=True))
                new.append(agent.add(kind="tool_result", role="tool", tid=ikey, exit=ec, err=ec not in (0, None),
                                     text=redact(item.get("aggregated_output") or "", ledger), ts=t_ts, refs=[r],
                                     flags=list(flags), _exec_only=True))
            elif itype == "agent_message":
                new.append(agent.add(kind="assistant", role="assistant", text=redact(item.get("text") or "", ledger),
                                     ts=t_ts, refs=[r], flags=list(flags), _exec_only=True))
            elif itype == "reasoning":
                new.append(agent.add(kind="thinking", role="assistant",
                                     text=reasoning_marker(ledger, "codex_exec_reasoning", item.get("text") or ""),
                                     ts=t_ts, refs=[r], flags=list(flags), _exec_only=True))
            elif itype == "file_change":
                text = "\n".join(f"--- {c.get('kind')}: {c.get('path')}" for c in item.get("changes") or []
                                 if isinstance(c, dict))
                new.append(agent.add(kind="tool_use", role="assistant", name="file_change", text=text, ts=t_ts,
                                     refs=[r], flags=list(flags), _exec_only=True,
                                     meta={"paths": sorted(str(c.get("path")) for c in item.get("changes") or []
                                                           if isinstance(c, dict))}))
            else:
                safe = scrub({k: v for k, v in item.items() if k not in ("type",)}, ledger)
                new.append(agent.add(kind="event", role="system", name=str(itype), text=_json_text(safe), ts=t_ts,
                                     refs=[r], flags=list(flags), _exec_only=True))
            seen[h] = new
            stats["exec-stream only item"] += 1
            continue
        if et in ("turn.completed", "turn.failed", "error"):
            payload = scrub({k: v for k, v in ev.items() if k != "type"}, ledger)
            t = agent.add(kind="event", role="system", name=et, text=_json_text(payload), ts=ts if ts is not None else prev_ts,
                          refs=[r], flags=["exec_stream_only"] + ([] if ts is not None else ["ts_estimated"]),
                          _exec_only=True)
            seen[h] = [t]
            stats[et] += 1
            continue
        stats[f"other:{et}"] += 1
    for k in ("item progress (merged into the completed item)", "thread.started", "turn.started"):
        if stats.get(k):
            sources.skip(sid, k, stats[k])
    sources.stat(sid, "kept", stats.get("exec-stream only item", 0))
    sources.stat(sid, "duplicates", stats.get("matched to rollout item", 0) + stats.get("copy_of_turn_log", 0))
    agent.meta.setdefault("exec_reconcile", {})[sid] = dict(stats)
    return seen


# ====================================================================== finishing
def finish(agent: Agent) -> None:
    """Order turns by time (records without a timestamp keep their place
    after the previous one), number them and compute statistics."""
    last = None
    for t in sorted(agent.turns, key=lambda t: t["_o"]):
        if t.get("ts") is None:
            t["_k"] = (last if last is not None else 0.0)
            if "ts_estimated" not in t["flags"] and last is not None:
                t["flags"].append("ts_estimated")
        else:
            t["_k"] = t["ts"]
            last = t["ts"]
    agent.turns.sort(key=lambda t: (t["_k"], t["_o"]))
    for i, t in enumerate(agent.turns, 1):
        t["seq"] = i
        t["iso"] = iso(t.get("ts")) if t.get("ts") is not None else None
        t.pop("_o", None)
        t.pop("_k", None)
        t.pop("_exec_only", None)
    kinds = collections.Counter(t["kind"] for t in agent.turns)
    tools = collections.Counter(t.get("name") for t in agent.turns if t["kind"] == "tool_use")
    stamped = [t["ts"] for t in agent.turns if t.get("ts") is not None and "ts_estimated" not in t["flags"]]
    tokens = collections.Counter()
    for u in agent._msg_usage.values():
        for k, v in u.items():
            if isinstance(v, (int, float)):
                tokens[k] += v
    agent.stats = {"turns": len(agent.turns), "by_kind": dict(kinds), "tools": dict(tools),
                   "first_ts": min(stamped) if stamped else None, "last_ts": max(stamped) if stamped else None,
                   "duplicates_referenced": sum(max(0, len(t["refs"]) - 1) for t in agent.turns),
                   "reasoning_markers": kinds.get("thinking", 0),
                   "unique_message_ids": len(agent._msg_usage),
                   "tokens_by_unique_message_id": dict(tokens) if agent._msg_usage else None,
                   "models": dict(agent.models)}


def to_index(agent: Agent) -> dict:
    return {"id": agent.id, "arm": agent.arm, "kind": agent.kind, "label": agent.label, "provider": agent.provider,
            "model_requested": agent.model_requested, "models": dict(agent.models), "session_id": agent.session_id,
            "agent_key": agent.agent_key, "parent": agent.parent, "children": agent.children, "sources": agent.sources,
            "notes": agent.notes, "meta": agent.meta, "stats": agent.stats, "engine_call": agent.engine_call,
            "tasks": list(agent.tasks.values()), "usage": usage_summary(agent)}


def usage_summary(agent: Agent) -> dict:
    """Usage as the records report it: the latest cumulative snapshot (never
    a sum of snapshots) and, for Claude, the sum over unique message ids."""
    out: dict = {}
    claude_cost = [s for s in agent.usage_snapshots if "totalCostUSD" in s or "total_cost_usd" in s]
    if claude_cost:
        last = claude_cost[-1]
        out["latest_cumulative_snapshot"] = {k: last.get(k) for k in ("totalCostUSD", "total_cost_usd", "modelUsage",
                                                                      "scope", "src", "num_turns")}
        out["snapshots"] = len(claude_cost)
    tc = [s for s in agent.usage_snapshots if s.get("kind") == "token_count" and s.get("total_token_usage")]
    if tc:
        out["codex_thread_total_latest"] = {"total_token_usage": tc[-1]["total_token_usage"], "src": tc[-1]["src"],
                                            "scope": "Codex thread cumulative counter (latest; never summed)"}
        out["snapshots"] = len(tc)
    tur = [s for s in agent.usage_snapshots if s.get("kind") == "token_usage_record"]
    if tur:
        by_resp = {}
        for s in tur:
            if s.get("response_id") and isinstance(s.get("usage"), dict):
                by_resp[s["response_id"]] = s["usage"]
        tot = collections.Counter()
        for u in by_resp.values():
            for k, v in u.items():
                if isinstance(v, (int, float)):
                    tot[k] += v
        out["codex_sum_unique_responses"] = {"responses": len(by_resp), "usage": dict(tot),
                                             "scope": "sum of per-response usage over unique response ids"}
    if agent.stats.get("tokens_by_unique_message_id"):
        out["claude_sum_unique_messages"] = {"messages": agent.stats["unique_message_ids"],
                                             "usage": agent.stats["tokens_by_unique_message_id"],
                                             "scope": "sum of per-message usage over unique message ids observed"}
    return out


_TASK_ID_RE = re.compile(r"<task-id>([^<]+)</task-id>")
_STATUS_RE = re.compile(r"<status>([^<]+)</status>")


def notification_tasks(agent: Agent) -> None:
    """Background task completions announced as user ``task-notification``
    messages in a native session (no stream): ``<task-id>`` -> timestamp."""
    for t in agent.turns:
        if t["kind"] == "user" and (t.get("meta") or {}).get("origin") == "task-notification":
            for m in _TASK_ID_RE.finditer(t.get("text") or ""):
                task = agent.tasks.setdefault(m.group(1), {"task_id": m.group(1), "events": []})
                st = _STATUS_RE.search(t.get("text") or "")
                task.setdefault("notified", []).append({"ts": t.get("ts"), "seq": t.get("seq"),
                                                        "status": st.group(1) if st else None})
