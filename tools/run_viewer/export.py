"""Export a study snapshot to the static run viewer.

    python3 -m tools.run_viewer.export --snapshot SNAP --out VIEWER [--arm NAME ...] [--analysis FILE.md]

``SNAP`` is a study-shaped directory (``arms/<arm>/{config.json, status.json,
invocations/, native-sessions/, work/.coresmith/}``, optional
``manifest.json`` and ``READY.json``). ``VIEWER`` receives ``index.html``,
the UI assets and ``data/`` shards loaded on demand with ``<script>`` tags,
so the viewer works from ``file://`` and from any static HTTP server.
Nothing in ``SNAP`` is modified.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import shutil
import sys
import time
from pathlib import Path

from . import cli_calls, engine, lineage, native
from .privacy import ENCRYPTED_RE, Ledger, mask_env_argv, redact, scrub
from .sources import Sources, iso, parse_iso, read_jsonl

UI_DIR = Path(__file__).resolve().parent / "ui"
PAGE_TURNS = 200
PAGE_BYTES = 1_500_000
SCHEMA = 1


# ------------------------------------------------------------------ helpers
def _load_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, ValueError):
        return default


def _text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


class Writer:
    """JSONP shards: ``RV.load(key, value)`` per file, so a ``<script>`` tag
    can load them from ``file://``."""

    def __init__(self, out: Path):
        self.out = out
        self.files = 0
        self.bytes = 0

    def write(self, rel: str, key: str, value) -> str:
        p = self.out / "data" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        body = json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
        body = body.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
        blob = f"RV.load({json.dumps(key)},{body});\n"
        p.write_text(blob, encoding="utf-8")
        self.files += 1
        self.bytes += len(blob.encode("utf-8"))
        return "data/" + rel


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)[:80]


# ------------------------------------------------------------------ one arm
def export_arm(arm: str, arm_dir: Path, sources: Sources, ledger: Ledger, writer: Writer) -> dict:
    cs = arm_dir / "work" / ".coresmith"
    t0 = time.time()
    config = scrub(_load_json(arm_dir / "config.json", {}) or {}, ledger)
    status = scrub(_load_json(arm_dir / "status.json", {}) or {}, ledger)
    for name, role in (("config.json", "arm configuration"), ("status.json", "arm driver status"),
                       ("interventions.jsonl", "coordinator interventions"), ("initial-prompt.txt", "initial prompt")):
        if (arm_dir / name).is_file():
            sources.add(arm_dir / name, role, arm)
    interventions = []
    if (arm_dir / "interventions.jsonl").is_file():
        for line, rec, _ in read_jsonl(arm_dir / "interventions.jsonl"):
            if rec is not None:
                interventions.append(scrub(rec, ledger))
    provider = (config.get("provider") or status.get("provider") or "").lower()
    invocations = []
    for inv in sorted((arm_dir / "invocations").glob("*")) if (arm_dir / "invocations").is_dir() else []:
        if not inv.is_dir():
            continue
        rec = {"name": inv.name, "dir": sources.rel(inv)}
        for f in ("command.json", "status.json", "result.json", "response.txt", "prompt.txt", "stderr.log"):
            p = inv / f
            if p.is_file():
                sid = sources.add(p, f"Architect invocation {inv.name} {f}", arm)
                if f == "command.json":
                    rec["command"] = mask_env_argv(_load_json(p, []) or [])
                elif f.endswith(".json"):
                    rec[f[:-5]] = scrub(_load_json(p, {}) or {}, ledger)
                else:
                    rec[f.replace(".", "_")] = redact(_text(p) or "", ledger)
                rec.setdefault("sources", {})[f] = sid
        if (inv / "transcript.jsonl").is_file():
            rec["transcript"] = inv / "transcript.jsonl"
        invocations.append(rec)

    project = engine.load_project(cs / "project.sqlite", sources, arm, ledger)
    checkpoints = {g: engine.load_checkpoints(cs / f"{g}_checkpoint.db", g, sources, arm)
                   for g in ("build", "pipeline", "backend", "architecture")}
    ev = engine.load_events(cs, sources, arm, ledger)
    events = ev["events"]
    epochs = engine.daemon_epochs(events, cs)
    calls = engine.load_llm_calls(cs / "llm_calls.jsonl", sources, arm, ledger)
    live = engine.load_live_streams(cs / "live_streams", sources, arm)
    turnlog = engine.load_codex_turns(cs / "codex_turns.jsonl", sources, arm)
    steps = engine.load_step_logs(cs / "step_logs", sources, arm)
    for extra in ("daemon.json", "daemon.log", "STATE.md", "OPERATOR_RULINGS.md"):
        if (cs / extra).is_file():
            sources.add(cs / extra, f"engine {extra}", arm)

    agents: dict[str, native.Agent] = {}
    used_native: set[str] = set()
    ns_root = arm_dir / "native-sessions"

    # ---------------------------------------------------------- Architect
    session_ids = []
    for inv in invocations:
        sid_ = (inv.get("status") or {}).get("session_id") or status.get("session_id")
        if sid_ and sid_ not in session_ids:
            session_ids.append(sid_)
    if not session_ids and status.get("session_id"):
        session_ids.append(status["session_id"])
    arch_by_session = {}
    for n, sess in enumerate(session_ids, 1):
        aid = "architect" if n == 1 else f"architect-{n}"
        a = native.Agent(arm, aid, kind="architect", label="Architect" + ("" if n == 1 else f" (session {n})"),
                         provider=provider)
        a.model_requested = config.get("model")
        a.session_id = sess
        a.meta["effort"] = config.get("effort")
        agents[aid] = a
        arch_by_session[sess] = a

    claude_files = sorted(ns_root.glob(".claude/projects/*/*.jsonl")) if ns_root.is_dir() else []
    codex_files = sorted(ns_root.glob(".codex/sessions/**/*.jsonl")) if ns_root.is_dir() else []
    codex_meta = {}
    for p in codex_files:
        for _, rec, _ in read_jsonl(p):
            if rec and rec.get("type") == "session_meta":
                pl = rec.get("payload") or {}
                src = pl.get("source") if isinstance(pl.get("source"), dict) else {}
                spawn = ((src.get("subagent") or {}).get("thread_spawn") or {}) if isinstance(src, dict) else {}
                codex_meta[str(p)] = {"id": pl.get("id"), "cwd": pl.get("cwd"),
                                      "parent": pl.get("parent_thread_id") or spawn.get("parent_thread_id"),
                                      "path": pl.get("agent_path") or spawn.get("agent_path"),
                                      "nick": pl.get("agent_nickname") or spawn.get("agent_nickname"),
                                      "ts": parse_iso(pl.get("timestamp"))}
            break
    codex_by_id = {m["id"]: Path(p) for p, m in codex_meta.items() if m.get("id")}
    claude_by_id = {p.stem: p for p in claude_files}

    for sess, a in arch_by_session.items():
        if provider == "claude":
            main = claude_by_id.get(sess)
            if main is not None:
                native.parse_claude_session(a, main, sources, ledger, "native Claude session (Architect)")
                used_native.add(str(main))
            # sub-agents: an Agent tool result (of the Architect or of another
            # sub-agent) names the agent id it started
            sub_dir = (main.parent / sess / "subagents") if main is not None else None
            subs = []
            if sub_dir is not None and sub_dir.is_dir():
                for sp in sorted(sub_dir.glob("agent-*.jsonl")):
                    key = sp.stem[len("agent-"):]
                    sa = native.Agent(arm, f"sub-{key}", kind="native-subagent", label=f"sub-agent {key}",
                                      provider="claude")
                    native.parse_claude_session(sa, sp, sources, ledger, "native Claude sub-agent session")
                    used_native.add(str(sp))
                    sa.agent_key = sa.agent_key or key
                    subs.append((key, sa))
            spawner = {}            # agent key -> (parent agent, tool_use id)
            for owner in [a] + [sa for _, sa in subs]:
                for t in owner.turns:
                    if t["kind"] == "tool_result" and t["meta"].get("agentId"):
                        spawner.setdefault(str(t["meta"]["agentId"]), (owner, t["tid"]))
            # a sub-agent started by another sub-agent: its Agent tool result
            # does not carry the id, the invocation stream's task_started does
            uses = {t["tid"]: owner for owner in [a] + [sa for _, sa in subs] for t in owner.turns
                    if t["kind"] == "tool_use" and t.get("name") in ("Agent", "Task")}
            for inv in invocations:
                if not inv.get("transcript"):
                    continue
                for _ln, rec, _raw in read_jsonl(inv["transcript"]):
                    if rec and rec.get("type") == "system" and rec.get("subtype") == "task_started" \
                            and rec.get("task_id") and rec.get("tool_use_id") in uses:
                        spawner.setdefault(str(rec["task_id"]), (uses[rec["tool_use_id"]], rec["tool_use_id"]))
            tid_to_agent = {tid: key for key, (_o, tid) in spawner.items()}
            for key, sa in subs:
                if key in spawner:
                    owner, tid = spawner[key]
                    st = next((t for t in owner.turns if t["kind"] == "tool_use" and t.get("tid") == tid), None)
                    sa.label = f"sub-agent: {(st or {}).get('meta', {}).get('description') or key}"
                    sa.model_requested = (st or {}).get("meta", {}).get("model")
                    sa.parent = {"agent": owner.id, "tool_use_id": tid, "label": "exact",
                                 "basis": "the parent's Agent tool result returned this agentId"}
                    if st:
                        st["child"] = sa.id
                    owner.children.append(sa.id)
                else:
                    sa.parent = {"agent": a.id, "label": "weak",
                                 "basis": "sub-agent file stored under the Architect session directory; no Agent "
                                          "tool result names it"}
                    a.children.append(sa.id)
                agents[sa.id] = sa
            route_map = {tid: agents.get(f"sub-{ak}") for tid, ak in tid_to_agent.items()}
            for inv in invocations:
                if inv.get("transcript") and ((inv.get("status") or {}).get("session_id") or sess) == sess:
                    native.parse_claude_stream(a, inv["transcript"], sources, ledger,
                                               f"Architect invocation {inv['name']} stream-json transcript",
                                               route=lambda tid: route_map.get(tid), invocation=inv)
                    inv["agent"] = a.id
        elif provider == "codex":
            main = codex_by_id.get(sess)
            if main is not None:
                native.parse_codex_rollout(a, main, sources, ledger, "native Codex rollout (Architect)")
                used_native.add(str(main))
            for inv in invocations:
                if inv.get("transcript") and ((inv.get("status") or {}).get("session_id") or sess) == sess:
                    sid_ = sources.add(inv["transcript"], f"Architect invocation {inv['name']} codex exec --json transcript", arm)
                    a.sources.append(sid_)
                    evs = []
                    for line, rec, _ in read_jsonl(inv["transcript"]):
                        sources.stat(sid_, "records")
                        if rec is None:
                            sources.stat(sid_, "unparseable")
                            continue
                        evs.append((line, None, rec))
                    native.reconcile_codex_exec(a, evs, sid_, sources, ledger, call_scope=f"invocation:{inv['name']}")
                    inv["agent"] = a.id
            # spawned threads, recursively
            frontier = [sess]
            while frontier:
                parent_id = frontier.pop(0)
                parent_agent = next((x for x in agents.values() if x.session_id == parent_id), None)
                for p, m in sorted(codex_meta.items(), key=lambda kv: kv[1].get("ts") or 0):
                    if m.get("parent") != parent_id or p in used_native:
                        continue
                    sa = native.Agent(arm, f"sub-{_slug((m.get('path') or '').strip('/').split('/')[-1] or m['id'][-12:])}",
                                      kind="native-subagent",
                                      label=f"sub-agent {m.get('path') or m['id']} ({m.get('nick') or 'unnamed'})",
                                      provider="codex")
                    if sa.id in agents:
                        sa.id = f"{sa.id}-{m['id'][-6:]}"
                    native.parse_codex_rollout(sa, Path(p), sources, ledger, "native Codex rollout (spawned thread)")
                    used_native.add(p)
                    spawn = None
                    if parent_agent is not None:
                        spawn = next((t for t in parent_agent.turns if t["kind"] == "event" and
                                      t["meta"].get("child_thread") == m["id"]), None)
                        call = next((t for t in parent_agent.turns if t["kind"] == "tool_use" and spawn is not None
                                     and t.get("tid") == spawn.get("tid")), None)
                        if call is not None:
                            call["child"] = sa.id
                        parent_agent.children.append(sa.id)
                    sa.parent = {"agent": parent_agent.id if parent_agent else None, "label": "exact",
                                 "tool_use_id": spawn.get("tid") if spawn else None,
                                 "basis": "rollout session_meta parent_thread_id"
                                          + (" and the parent's sub-agent activity item" if spawn else "")}
                    agents[sa.id] = sa
                    frontier.append(m["id"])

    # ---------------------------------------------------------- engine helpers
    start_events = [e for e in events if e["event"] == "llm_call_start"]
    beats = collections.Counter(e["pid"] for e in events if e["event"] == "llm_call_heartbeat")
    llm_lifecycle = [e for e in events if e["node"] == "LLM" and e["event"] in ("llm_start", "llm_end", "llm_error",
                                                                               "llm_nonzero_exit", "llm_timed out")]
    claimed_starts: set[str] = set()
    helper_records = []
    for c in calls:
        ep = engine.epoch_for(epochs, c["start_ts"] if c["start_ts"] is not None else c["ts"])
        c["epoch"] = ep["epoch"] if ep else None
        cands = [e for e in start_events if e["id"] not in claimed_starts
                 and (e["fields"] or {}).get("call_index") == c["call_index"]
                 and c["start_ts"] is not None and c["start_ts"] - 10 <= (e["ts"] or 0) <= (c["ts"] or 0)]
        c["start_event"] = cands[0]["id"] if len(cands) == 1 else None
        c["child_pid"] = cands[0]["pid"] if len(cands) == 1 else None
        if len(cands) == 1:
            claimed_starts.add(cands[0]["id"])
        helper_records.append(("call", c, cands[0] if len(cands) == 1 else None))
    for e in start_events:
        if e["id"] in claimed_starts:
            continue
        ep = engine.epoch_for(epochs, e["ts"])
        # the daemon's llm_start (same process, written just before the
        # child starts) names the call; the call has no llm_calls record yet
        st = [x for x in llm_lifecycle if x["event"] == "llm_start" and x["ts"] is not None and e["ts"] is not None
              and e["ts"] - 5 <= x["ts"] <= e["ts"] + 1 and (ep is None or x["pid"] == ep["pid"])]
        fs = (st[-1]["fields"] or {}) if len(st) == 1 else {}
        helper_records.append(("unfinished", {"call_index": (e["fields"] or {}).get("call_index"),
                                              "start_ts": e["ts"], "ts": None, "epoch": ep["epoch"] if ep else None,
                                              "child_pid": e["pid"], "provider": fs.get("provider"),
                                              "model": (e["fields"] or {}).get("model"), "run_name": fs.get("run_name") or "",
                                              "run_name_basis": "the daemon's llm_start event just before the child "
                                                                "started" if fs else None,
                                              "start_event": e["id"], "graph": ""}, e))
    used_live: set[int] = set()
    used_turnlog: set[tuple] = set()
    for kind, c, sev in helper_records:
        aid = f"h{c.get('epoch') or 0}.{int(c['call_index']):02d}" if c.get("call_index") is not None else f"h-line{c.get('line')}"
        if aid in agents:
            aid = f"{aid}-{c.get('line') or 'x'}"
        node, blk = lineage.run_name_parts(c.get("run_name") or "")
        label = (f"helper e{c.get('epoch')}#{c.get('call_index')}: {c.get('run_name') or '(run name unknown)'}"
                 + ("  [unfinished at snapshot]" if kind == "unfinished" else ""))
        h = native.Agent(arm, aid, kind="engine-helper", label=label,
                         provider={"claude_cli": "claude", "codex_cli": "codex"}.get(c.get("provider"), provider))
        h.model_requested = c.get("model")
        pid = c.get("child_pid")
        lv = live.get(pid) if pid is not None else None
        if lv is not None and sev is not None and abs((lv["meta"].get("started_ts") or 0) - (sev["ts"] or 0)) > 30:
            lv = None
        session = (c.get("usage") or {}).get("session_id") or (lv or {}).get("session_id")
        tl_key = None
        for key, g in turnlog.items():
            if key in used_turnlog:
                continue
            hd = g["header"]
            if pid is not None and hd.get("pid") == pid and hd.get("call_index") == c.get("call_index"):
                tl_key = key
                break
        if session is None and tl_key is not None:
            session = turnlog[tl_key]["thread_id"]
        h.session_id = session
        h.engine_call = {"kind": kind, "line": c.get("line"), "src": c.get("src"), "call_index": c.get("call_index"),
                         "epoch": c.get("epoch"), "run_name": c.get("run_name"), "node": node, "block": blk,
                         "graph": c.get("graph"), "provider": c.get("provider"), "model": c.get("model"),
                         "start_ts": c.get("start_ts"), "ts": c.get("ts"), "duration_s": c.get("duration_s"),
                         "timeout": c.get("timeout"), "timed_out": c.get("timed_out"), "error": c.get("error"),
                         "usage": c.get("usage"), "lens": c.get("lens"), "child_pid": pid,
                         "start_event": c.get("start_event"), "heartbeats": beats.get(pid, 0) if pid else 0,
                         "joins": []}
        j = h.engine_call["joins"]
        if c.get("start_event"):
            j.append({"to": "llm_call_start event", "id": c["start_event"], "label": "strong",
                      "basis": "unique matching call_index and time window; the completed call log has no process id"})
        # native session
        if session:
            p = claude_by_id.get(session) if h.provider == "claude" else codex_by_id.get(session)
            if p is not None and str(p) not in used_native:
                if h.provider == "claude":
                    native.parse_claude_session(h, p, sources, ledger, "native Claude session (engine helper)")
                else:
                    native.parse_codex_rollout(h, p, sources, ledger, "native Codex rollout (engine helper)")
                used_native.add(str(p))
                j.append({"to": "native session", "id": str(sources.rel(p)), "label": "exact",
                          "basis": "provider session/thread id equals the call's session id"})
            elif p is None:
                h.note("The provider session file for this call is not in the snapshot.")
        prior = None
        call_scope = f"pid{pid}:{(turnlog.get(tl_key) or {}).get('header', {}).get('wall_start')}" if tl_key else f"pid{pid}"
        if tl_key is not None:
            g = turnlog[tl_key]
            used_turnlog.add(tl_key)
            if g["sid"] not in h.sources:
                h.sources.append(g["sid"])
            prior = native.reconcile_codex_exec(h, g["lines"], g["sid"], sources, ledger, call_scope=call_scope)
            j.append({"to": "engine turn log", "id": g["sid"], "label": "exact",
                      "basis": "turn-log header pid and call_index equal the call's"})
        if lv is not None:
            used_live.add(pid)
            j.append({"to": "live stream capture", "id": lv["sid"], "label": "exact",
                      "basis": "live_streams/<pid>.json of the call's child pid, started with the call"})
            if h.provider == "claude":
                native.parse_claude_stream(h, (lv["sid"], lv["lines"]), sources, ledger, "live stream")
            else:
                if lv["sid"] not in h.sources:
                    h.sources.append(lv["sid"])
                native.reconcile_codex_exec(h, [(i, None, e) for i, e in lv["lines"]], lv["sid"], sources, ledger,
                                            call_scope=call_scope, prior=prior)
        # the engine's own prompt/response record
        if kind == "call":
            first_user = next((t for t in h.turns if t["kind"] == "user"), None)
            h.add(kind="system", role="system", name="system_prompt (llm_calls)", text=c["system_prompt"],
                  ts=c["start_ts"], refs=[c["src"]], flags=["ts_estimated"])
            if first_user is not None and first_user["text"].strip() == (c["user_prompt"] or "").strip():
                first_user["refs"].append(c["src"])
            else:
                h.add(kind="user", role="user", name="user_prompt (llm_calls)", text=c["user_prompt"],
                      ts=c["start_ts"], refs=[c["src"]], flags=["ts_estimated"])
            last_asst = next((t for t in reversed(h.turns) if t["kind"] == "assistant"), None)
            if last_asst is not None and last_asst["text"].strip() == (c["response"] or "").strip():
                last_asst["refs"].append(c["src"])
            else:
                h.add(kind="assistant", role="assistant", name="response (llm_calls)", text=c["response"],
                      ts=c["ts"], refs=[c["src"]], meta={"error": c.get("error"), "timed_out": c.get("timed_out")})
        else:
            h.note("No llm_calls record: the call had not finished (or its record was lost) when the snapshot was taken.")
        # lifecycle events of the daemon for this run name
        lo = c.get("start_ts")
        hi = c.get("ts")
        for e in llm_lifecycle:
            rn = (e["fields"] or {}).get("run_name")
            if lo is None or e["ts"] is None:
                continue
            if e["ts"] < lo - 5 or (hi is not None and e["ts"] > hi + 5):
                continue
            if rn is not None and rn != c.get("run_name"):
                continue
            if e["event"] == "llm_end" and (e["fields"] or {}).get("session_id") and session and \
                    e["fields"]["session_id"] != session:
                continue
            h.engine_call.setdefault("lifecycle_events", []).append(e["id"])
        agents[aid] = h

    # live streams / turn logs never claimed by a call, and unlinked native sessions
    for pid, lv in live.items():
        if pid in used_live:
            continue
        aid = f"stream-{pid}"
        h = native.Agent(arm, aid, kind="engine-helper", label=f"unlinked live stream pid {pid}", provider=provider)
        h.session_id = lv["session_id"]
        h.note("No helper call record or llm_call_start event names this child pid.")
        if provider == "claude":
            native.parse_claude_stream(h, (lv["sid"], lv["lines"]), sources, ledger, "live stream")
        else:
            h.sources.append(lv["sid"])
            native.reconcile_codex_exec(h, [(i, None, e) for i, e in lv["lines"]], lv["sid"], sources, ledger,
                                        call_scope=f"pid{pid}")
        agents[aid] = h
    for key, g in turnlog.items():
        if key in used_turnlog:
            continue
        aid = f"turnlog-{key[0]}"
        h = native.Agent(arm, aid, kind="engine-helper", label=f"unlinked turn log pid {key[0]} call_index "
                         f"{g['header'].get('call_index')}", provider="codex")
        h.session_id = g["thread_id"]
        h.sources.append(g["sid"])
        native.reconcile_codex_exec(h, g["lines"], g["sid"], sources, ledger, call_scope=f"pid{key[0]}:{key[1]}")
        agents[aid] = h
    for p in claude_files + codex_files:
        if str(p) in used_native:
            continue
        sid_ = p.stem if p.suffix == ".jsonl" else p.name
        aid = f"native-{_slug(sid_[-16:])}"
        o = native.Agent(arm, aid, kind="native-unlinked", label=f"native session {sid_} (not linked)",
                         provider="claude" if ".claude" in str(p) else "codex")
        if ".claude" in str(p):
            native.parse_claude_session(o, p, sources, ledger, "native Claude session (unlinked)")
        else:
            native.parse_codex_rollout(o, p, sources, ledger, "native Codex rollout (unlinked)")
        o.note("No Architect, sub-agent or helper record links to this session.")
        agents[aid] = o

    for a in agents.values():
        native.finish(a)
        native.notification_tasks(a)

    # ---------------------------------------------------------- shell calls + CLI audit
    tool_inputs = []
    for a in agents.values():
        for t in a.turns:
            if t["kind"] != "tool_use":
                continue
            wpath = None
            text = t.get("text") or ""
            if t.get("name") in ("Write",):
                try:
                    inp = json.loads(text)
                    wpath, text = inp.get("file_path"), inp.get("content") or ""
                except ValueError:
                    pass
            elif t.get("name") == "file_change" and len(t["meta"].get("paths") or []) == 1:
                wpath = t["meta"]["paths"][0]
            tool_inputs.append((f"{a.id}:{t['seq']}", t.get("name"), text, wpath))
    wrappers = cli_calls.find_wrappers(tool_inputs)
    shell = []
    for a in agents.values():
        results = {}
        for t in a.turns:
            if t["kind"] == "tool_result" and t.get("tid"):
                results.setdefault(t["tid"], t)
        for t in a.turns:
            if t["kind"] != "tool_use" or not (t.get("name") in native.SHELL_TOOLS or t.get("name") == "exec_command"):
                continue
            res = results.get(t.get("tid"))
            m = t.get("meta") or {}
            rm = (res or {}).get("meta") or {}
            if t.get("name") == "exec_command":
                start = m.get("start_ts") if m.get("start_ts") is not None else t.get("ts")
                end = t.get("ts") if m.get("start_ts") is not None or res else None
                if "exec_stream_only" in t["flags"] and m.get("start_ts") is None:
                    start, end = t.get("ts"), t.get("ts")
                exit_code = m.get("exit_code")
                end_basis = "rollout completion time (start = completion - duration)" if m.get("start_ts") is not None \
                    else "exec stream item time"
                background = False
            else:
                start = t.get("ts")
                end = res.get("ts") if res else None
                end_basis = "tool result record time"
                bg_id = rm.get("backgroundTaskId") or rm.get("taskId") if (m.get("run_in_background") or
                                                                           t.get("name") == "Monitor") else None
                background = bool(bg_id)
                if background:
                    task = a.tasks.get(str(bg_id)) or {}
                    end = task.get("end_ts") or next((n["ts"] for n in task.get("notified") or [] if n.get("ts")), None)
                    end_basis = "background task end (stream task_updated / task-notification)" if end else \
                        "background task end not observed"
                exit_code = None
                if res is not None and not background:
                    mm = re.search(r"Exit code (\d+)", res.get("text") or "")
                    exit_code = int(mm.group(1)) if mm else (1 if res.get("err") else 0)
            parsed = cli_calls.parse_invocations(t.get("text") or "", wrappers)
            shell.append({"id": f"{a.id}:{t['seq']}", "agent": a.id, "seq": t["seq"],
                          "result_seq": res["seq"] if res else None, "tool": t.get("name"), "start_ts": start,
                          "end_ts": end, "end_basis": end_basis, "exit_code": exit_code,
                          "is_error": bool(res and res.get("err")), "background": background,
                          "interrupted": bool(rm.get("interrupted")), "open_window_s": 6 * 3600 if background and not end else 0,
                          "command": t.get("text") or "", "output": (res or {}).get("text") or "",
                          "invocations": parsed["invocations"], "loop": parsed["loop"], "scripted": parsed["scripted"],
                          "heredoc": parsed["heredoc"], "loop_expanded": parsed.get("loop_expanded"),
                          "mentions_coresmith": parsed.get("mentions_coresmith"), "flags": t["flags"]})
    stats = cli_calls.match(project["actions"], shell)
    first_native = min((t["ts"] for a in agents.values() for t in a.turns if t.get("ts") is not None), default=None)
    if first_native is not None:
        for a_ in project["actions"]:
            if not a_.get("native") and a_.get("ts") is not None and a_["ts"] < first_native:
                a_["context_note"] = (f"recorded before the first captured native record ({iso(first_native)}): run by "
                                      "the study setup, outside every captured trajectory")

    # ---------------------------------------------------------- lineage
    helper_index = [{"id": a.id, "engine_call": a.engine_call} for a in agents.values() if a.engine_call]
    lin = lineage.build_lineage(arm, project, checkpoints, events, epochs, helper_index, steps)
    runs = lineage.graph_runs(checkpoints, events, project, lin["builds"], lin["parks"])
    for h in helper_index:
        agents[h["id"]].engine_call["builds"] = h.get("builds") or []
    for r in runs:
        r["actions"] = []
        verb = {"pipeline": "run", "backend": "backend", "architecture": "architecture"}.get(r["graph"])
        for a_ in project["actions"]:
            if a_["argv"][:1] == [verb] and r["first_ts"] is not None and r["first_ts"] - 120 <= a_["ts"] <= (r["last_ts"] or 0) + 600:
                lab = "exact" if (a_.get("run_id") and a_.get("run_id") in r["run_ids"]) else "weak"
                r["actions"].append({"action": a_["id"], "label": lab,
                                     "basis": "audit run_id equals the thread's run id" if lab == "exact"
                                     else f"`{verb}` verb audited within the thread's checkpoint span"})
                a_.setdefault("graph_runs", []).append({"run": r["id"], "label": lab})
    for h in helper_index:
        call = h["engine_call"]
        if not h.get("builds") and call.get("graph") in ("pipeline", "backend"):
            for r in runs:
                if r["graph"] == call["graph"] and r["first_ts"] and call.get("start_ts") and \
                        r["first_ts"] - 5 <= call["start_ts"] <= (r["last_ts"] or 0) + 900:
                    agents[h["id"]].engine_call.setdefault("graph_runs", []).append(
                        {"run": r["id"], "label": "strong", "basis": "the call's graph equals the thread's graph and it "
                                                                     "ran inside the thread's checkpoint span"})
                    r.setdefault("helpers", []).append(h["id"])

    # ---------------------------------------------------------- write shards
    # every fingerprint of this arm is known now: withhold copied fragments
    # of signatures / encrypted payloads wherever an agent printed them
    for a in agents.values():
        for t in a.turns:
            if t.get("text"):
                new = ledger.scrub_private_runs(t["text"])
                if new is not t["text"]:
                    t["text"] = new
                    t["len"] = len(new)
                    if "private_fragment_withheld" not in t["flags"]:
                        t["flags"].append("private_fragment_withheld")

    def _scrub_strings(v):
        if isinstance(v, str):
            return ledger.scrub_private_runs(v)
        if isinstance(v, list):
            return [_scrub_strings(x) for x in v]
        if isinstance(v, dict):
            return {k: _scrub_strings(x) for k, x in v.items()}
        return v
    for e in events:
        e["fields"] = _scrub_strings(e["fields"])
    agent_index = []
    for a in agents.values():
        rec = native.to_index(a)
        pages = []
        page, size = [], 0
        for t in a.turns:
            page.append(t)
            size += len(t.get("text") or "") + 200
            if len(page) >= PAGE_TURNS or size >= PAGE_BYTES:
                pages.append(page)
                page, size = [], 0
        if page:
            pages.append(page)
        rec["pages"] = []
        for k, pg in enumerate(pages):
            rel = writer.write(f"{arm}/agents/{_slug(a.id)}/p{k}.js", f"turns:{arm}:{a.id}:{k}", pg)
            rec["pages"].append({"file": rel, "from": pg[0]["seq"], "to": pg[-1]["seq"],
                                 "first_ts": next((t["ts"] for t in pg if t.get("ts")), None),
                                 "last_ts": next((t["ts"] for t in reversed(pg) if t.get("ts")), None),
                                 "bytes": sum(len(t.get("text") or "") for t in pg)})
        rec["shell_calls"] = sum(1 for s in shell if s["agent"] == a.id)
        agent_index.append(rec)
    agent_index.sort(key=lambda r: ({"architect": 0, "native-subagent": 1, "engine-helper": 2}.get(r["kind"], 3),
                                    (r.get("engine_call") or {}).get("start_ts") or (r["stats"] or {}).get("first_ts") or 0))
    writer.write(f"{arm}/events.js", f"events:{arm}", events)
    ck_all = []
    for g, ck in checkpoints.items():
        for thread, nss in (ck.get("threads") or {}).items():
            for nsn, cps in nss.items():
                ck_all.extend(cps)
    writer.write(f"{arm}/checkpoints.js", f"checkpoints:{arm}", ck_all)
    step_index = []
    for s in steps:
        text = ledger.scrub_private_runs(redact(_text(s["path"]) or "", ledger))
        rel = writer.write(f"{arm}/steps/{s['id']}.js", f"step:{arm}:{s['id']}", {"text": text})
        step_index.append({k: v for k, v in s.items() if k != "path"} | {"file": rel, "chars": len(text)})
    proj_tables = {k: v for k, v in project.items() if isinstance(v, list) and k not in ("actions", "builds")}
    writer.write(f"{arm}/project.js", f"project:{arm}", proj_tables)
    helper_calls = []
    for a in agent_index:
        if a.get("engine_call"):
            helper_calls.append({"agent": a["id"], **a["engine_call"]})
    shell_index = []
    for s in shell:
        if not s["invocations"] and not s["audit"]:
            continue
        shell_index.append({k: v for k, v in s.items() if k not in ("output", "command")} |
                           {"command_preview": s["command"][:400], "command_len": len(s["command"]),
                            "output_len": len(s["output"])})
    missing = [{"call": s["id"], "argv": m["argv"], "code": m["code"], "why": m["why"], "ts": s["start_ts"],
                "agent": s["agent"]} for s in shell for m in s.get("missing") or []]
    unaudited_counts = collections.Counter(m["code"] for m in missing)
    # usage roll-up (never summing cumulative snapshots)
    helper_cost = [c["usage"].get("total_cost_usd") for c in calls if isinstance((c.get("usage") or {}).get("total_cost_usd"), (int, float))]
    helper_unknown = sum(1 for c in calls if not isinstance((c.get("usage") or {}).get("total_cost_usd"), (int, float)))
    out = {
        "arm": arm, "config": config, "status": status, "interventions": interventions,
        "invocations": [{k: v for k, v in inv.items() if k != "transcript"} for inv in invocations],
        "provider": provider,
        "counts": {"actions": len(project["actions"]), "builds": len(project["builds"]), "events": len(events),
                   "event_files": len(ev["files"]), "event_copies_deduplicated": sum(len(e["copies"]) for e in events),
                   "within_file_repeats_kept": ev["within_file_repeats"], "helper_calls": len(calls),
                   "helpers_unfinished": sum(1 for k, _, _ in helper_records if k == "unfinished"),
                   "agents": len(agent_index), "turns": sum(len(a.turns) for a in agents.values()),
                   "shell_calls": len(shell), "shell_calls_invoking_coresmith": sum(1 for s in shell if s["invocations"]),
                   "observed_invocations_missing_from_audit": len(missing), "step_logs": len(steps),
                   "checkpoints": {g: ck.get("checkpoints") for g, ck in checkpoints.items()},
                   "project_tables": project.get("counts"), "interrupts": len(project["interrupts"])},
        "cli_join": stats, "missing_by_code": dict(unaudited_counts), "cli_wrappers": wrappers,
        "actions": project["actions"], "shell": shell_index, "missing": missing,
        "builds": lin["builds"], "parks": lin["parks"], "graph_runs": runs, "agents": agent_index,
        "helper_calls": helper_calls, "steps": step_index, "epochs": epochs, "event_files": ev["files"],
        "stages": project["stages"], "results": project["results"],
        "evidence": {"dv": project["dv_results"], "coverage": project["coverage_results"], "ppa": project["ppa_history"]},
        "usage": {"architect": [a.get("usage") for a in agent_index if a["kind"] == "architect"],
                  "helper_cost_usd_sum_over_calls": round(sum(helper_cost), 4) if helper_cost else None,
                  "helper_calls_without_cost": helper_unknown,
                  "helper_cost_scope": "sum of llm_calls usage.total_cost_usd over distinct finished calls; calls "
                                       "without a cost are unknown, not zero"},
        "export_seconds": round(time.time() - t0, 1),
    }
    return out


# ------------------------------------------------------------------ top level
def export(snapshot: Path, out: Path, arms: list[str] | None = None, analysis: Path | None = None) -> dict:
    snapshot = Path(snapshot).resolve()
    out = Path(out).resolve()
    if out == snapshot or snapshot in out.parents or out in snapshot.parents:
        raise SystemExit("snapshot and viewer output directories must not contain one another")
    manifest = _load_json(snapshot / "manifest.json", {}) or {}
    ready = _load_json(snapshot / "READY.json", None)
    sources = Sources(snapshot, manifest)
    ledger = Ledger()
    if (out / "data").exists():
        shutil.rmtree(out / "data")
    out.mkdir(parents=True, exist_ok=True)
    writer = Writer(out)
    arm_dirs = sorted(p for p in (snapshot / "arms").glob("*") if p.is_dir())
    if arms:
        arm_dirs = [p for p in arm_dirs if p.name in arms]
    arms_out = {}
    for d in arm_dirs:
        print(f"[run_viewer] exporting {d.name} ...", file=sys.stderr, flush=True)
        arms_out[d.name] = export_arm(d.name, d, sources, ledger, writer)
    analysis_text = None
    if analysis and Path(analysis).is_file():
        analysis_text = Path(analysis).read_text(encoding="utf-8")
        writer.write("analysis.js", "analysis", {"markdown": analysis_text, "source": str(analysis)})
    index = {
        "schema": SCHEMA, "generated_at": iso(time.time()), "snapshot": {
            "path": str(snapshot), "ready": ready, "manifest": {k: manifest.get(k) for k in (
                "started_at", "finished_at", "study", "errors", "consistency", "scope", "bytes") if k in manifest},
            "manifest_files": len(manifest.get("files") or []), "unread_manifest_files": None},
        "arms": arms_out, "sources": None, "privacy": None, "has_analysis": analysis_text is not None,
    }
    index["snapshot"]["unread_manifest_files"] = sources.unregistered_manifest_files()
    index["sources"] = sources.to_json()
    # privacy audit of everything written so far + the index itself
    ledger.resolve()
    idx_blob = json.dumps(index, ensure_ascii=False, default=str)
    leaks, enc_hits, scanned = [], 0, 0
    for p in sorted((out / "data").rglob("*.js")):
        blob = p.read_text(encoding="utf-8")
        scanned += 1
        enc_hits += len(ENCRYPTED_RE.findall(blob))
        for f in ledger.leaks(blob):
            leaks.append({"file": str(p.relative_to(out)), "fingerprint_len": len(f)})
    enc_hits += len(ENCRYPTED_RE.findall(idx_blob))
    leaks += [{"file": "data/index.js", "fingerprint_len": len(f)} for f in ledger.leaks(idx_blob)]
    index["privacy"] = {"withheld": dict(ledger.counts), "fingerprints_checked": ledger.fingerprint_count,
                        "files_scanned": scanned + 1, "leaks": leaks, "encrypted_pattern_hits": enc_hits,
                        "ok": not leaks and enc_hits == 0}
    writer.write("index.js", "index", index)
    for f in ("index.html", "app.js", "app.css"):
        shutil.copyfile(UI_DIR / f, out / f)
    summary = {"out": str(out), "files": writer.files, "bytes": writer.bytes, "privacy_ok": index["privacy"]["ok"],
               "arms": {k: v["counts"] for k, v in arms_out.items()}}
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--snapshot", required=True, type=Path, help="study-shaped snapshot directory (read only)")
    ap.add_argument("--out", required=True, type=Path, help="viewer output directory (its data/ is replaced)")
    ap.add_argument("--arm", action="append", help="export only this arm (repeatable)")
    ap.add_argument("--analysis", type=Path, help="markdown analysis shown in the Analysis view")
    args = ap.parse_args(argv)
    summary = export(args.snapshot, args.out, args.arm, args.analysis)
    print(json.dumps(summary, indent=1))
    return 0 if summary["privacy_ok"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
