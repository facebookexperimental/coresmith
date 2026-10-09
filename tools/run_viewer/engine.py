"""Read-only loaders for one arm's engine state (``work/.coresmith``).

SQLite files are opened ``mode=ro`` (``immutable=1`` as a fallback for a copy
without sidecars). Every row and line keeps a source reference. Nothing here
joins records; :mod:`lineage` and :mod:`cli_calls` do that with labels.
"""
from __future__ import annotations

import collections
import hashlib
import json
import re
import sqlite3
from pathlib import Path

from . import msgpack_lite
from .privacy import Ledger, redact, scrub
from .sources import Sources, iso, parse_iso, read_jsonl, ref


def open_ro(path: Path) -> sqlite3.Connection | None:
    """A snapshot database is a static online backup: ``immutable=1`` reads
    it without locks and without creating ``-wal`` / ``-shm`` sidecars (a
    ``mode=ro`` open of a WAL-mode file creates them). A database that still
    has a non-empty ``-wal`` next to it is opened ``mode=ro`` so committed
    WAL frames are not ignored."""
    wal = Path(str(path) + "-wal")
    order = (f"file:{path}?immutable=1", f"file:{path}?mode=ro")
    if wal.is_file() and wal.stat().st_size > 0:
        order = order[::-1]
    for uri in order:
        try:
            con = sqlite3.connect(uri, uri=True, timeout=2.0)
            con.row_factory = sqlite3.Row
            con.execute("select 1 from sqlite_master limit 1").fetchall()
            return con
        except sqlite3.Error:
            continue
    return None


def _uj(text, default=None):
    if text is None or text == "":
        return default
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return default


def _tables(con) -> set[str]:
    return {r[0] for r in con.execute("select name from sqlite_master where type='table'")}


# ------------------------------------------------------------------ project.sqlite
_JSON_COLS = ("argv_json", "inputs_json", "worker_json", "seed_json", "result_json", "payload_json",
              "resolution_json", "extra_json", "meta_json", "value_json", "blocked_by_json", "spec_json",
              "diagnosis_json", "real_blocks_json", "stub_blocks_json", "wiring_errors_json", "elab_errors_json",
              "boundary_json", "args_json", "conflict_json", "stolen_from_json")
_ROW_TABLES = ("actions", "builds", "dv_results", "coverage_results", "ppa_history", "interrupts", "decisions",
               "stages", "results", "items", "checks", "rulings", "ruling_uses", "constraints", "attempts",
               "diagnoses", "questions", "leases", "run_flags", "integration_snapshots", "artifacts", "models",
               "blocks", "verifiers", "item_links", "settings")


def _row(table: str, r: sqlite3.Row, sid: str, ledger: Ledger) -> dict:
    d = dict(r)
    key = d.get("id", d.get("name", d.get("kind", d.get("block"))))
    for col in list(d):
        if col in _JSON_COLS:
            d[col[:-5]] = _uj(d.pop(col), None)
    # engine rows are not model records: only encrypted/credential-like
    # strings are redacted; a public ``reasoning`` rationale is kept.
    d = scrub(d, ledger, drop=frozenset())
    d["src"] = ref(sid, key=f"{table}#{key}")
    return d


def load_project(db_path: Path, sources: Sources, arm: str, ledger: Ledger) -> dict:
    out: dict = {t: [] for t in _ROW_TABLES}
    out["counts"] = {}
    if not db_path.is_file():
        out["missing"] = True
        return out
    sid = sources.add(db_path, "engine project database (SQLite online backup)", arm)
    con = open_ro(db_path)
    if con is None:
        sources.note(sid, "could not be opened read-only")
        out["missing"] = True
        return out
    have = _tables(con)
    for t in have:
        try:
            out["counts"][t] = int(con.execute(f'select count(*) from "{t}"').fetchone()[0])
        except sqlite3.Error:
            out["counts"][t] = None
    order = {"actions": "id", "builds": "requested_at, id", "dv_results": "id", "coverage_results": "id",
             "ppa_history": "id", "interrupts": "ts, id", "decisions": "ts, id", "stages": "ordinal",
             "checks": "id", "items": "id", "item_links": "from_id, to_id, rel"}
    for t in _ROW_TABLES:
        if t not in have:
            continue
        rows = con.execute(f'select * from "{t}" order by {order.get(t, "rowid")}').fetchall()
        out[t] = [_row(t, r, sid, ledger) for r in rows]
        sources.stat(sid, "records", len(rows))
        sources.stat(sid, "kept", len(rows))
    con.close()
    for a in out["actions"]:
        a["argv"] = [str(x) for x in (a.get("argv") or [])]
        a["iso"] = iso(a.get("ts"))
    out["source"] = sid
    return out


# ------------------------------------------------------------------ checkpoints
def load_checkpoints(db_path: Path, graph: str, sources: Sources, arm: str) -> dict:
    """``{graph, source, threads: {thread: {ns: [checkpoint...]}}, writes_total}``.
    A checkpoint lists the nodes whose consumed channel versions changed
    since its parent (``ran``) and the channels of its pending writes
    (names and value sizes only)."""
    out = {"graph": graph, "path": None, "source": None, "threads": {}, "checkpoints": 0, "writes": 0}
    if not db_path.is_file():
        return out
    sid = sources.add(db_path, f"LangGraph {graph} checkpoint database (SQLite online backup)", arm)
    out["source"] = sid
    con = open_ro(db_path)
    if con is None or "checkpoints" not in _tables(con):
        sources.note(sid, "no checkpoints table")
        return out
    writes = collections.defaultdict(list)
    if "writes" in _tables(con):
        for r in con.execute("select thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, length(value) n "
                             "from writes order by thread_id, checkpoint_ns, checkpoint_id, task_id, idx"):
            writes[(r["thread_id"], r["checkpoint_ns"], r["checkpoint_id"])].append(
                {"task": r["task_id"], "channel": r["channel"], "bytes": r["n"]})
            out["writes"] += 1
    rows = con.execute("select thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, checkpoint, metadata "
                       "from checkpoints order by thread_id, checkpoint_ns, checkpoint_id").fetchall()
    con.close()
    by_id = {}
    for r in rows:
        meta = {}
        try:
            meta = json.loads(r["metadata"]) if r["metadata"] else {}
        except (TypeError, ValueError):
            try:
                meta = msgpack_lite.unpack(r["metadata"])
            except Exception:  # noqa: BLE001 - metadata is optional evidence
                meta = {}
        head = msgpack_lite.checkpoint_header(r["checkpoint"]) if r["checkpoint"] else {"ts": None, "seen": {}}
        cp = {"id": r["checkpoint_id"], "parent": r["parent_checkpoint_id"], "thread": r["thread_id"],
              "ns": r["checkpoint_ns"] or "", "ts": parse_iso(head.get("ts")), "iso": head.get("ts"),
              "step": meta.get("step") if isinstance(meta, dict) else None,
              "source_kind": meta.get("source") if isinstance(meta, dict) else None,
              "parents": meta.get("parents") if isinstance(meta, dict) else None,
              "writes": writes.get((r["thread_id"], r["checkpoint_ns"], r["checkpoint_id"]), []),
              "decode_error": head.get("error"), "_seen": head.get("seen") or {},
              "src": ref(sid, key=f"checkpoints#{r['thread_id']}|{r['checkpoint_ns']}|{r['checkpoint_id']}")}
        by_id[(cp["thread"], cp["ns"], cp["id"])] = cp
        out["threads"].setdefault(cp["thread"], {}).setdefault(cp["ns"], []).append(cp)
        out["checkpoints"] += 1
    for thread, nss in out["threads"].items():
        for ns, cps in nss.items():
            cps.sort(key=lambda c: (c["ts"] or 0, c["id"]))
            for c in cps:
                parent = by_id.get((thread, ns, c["parent"])) if c["parent"] else None
                prev = parent["_seen"] if parent else {}
                c["ran"] = sorted(n for n, d in c["_seen"].items() if prev.get(n) != d)
                c["interrupt_writes"] = sum(1 for w in c["writes"] if w["channel"] == "__interrupt__")
                c["resume_writes"] = sum(1 for w in c["writes"] if w["channel"] == "__resume__")
    for cps in (c for nss in out["threads"].values() for c in nss.values()):
        for c in cps:
            c.pop("_seen", None)
    sources.stat(sid, "records", out["checkpoints"])
    sources.stat(sid, "kept", out["checkpoints"])
    return out


# ------------------------------------------------------------------ pipeline events
_EVENT_FILE_RE = re.compile(r"^pipeline_events(?:\.(?P<stamp>[0-9]{8}-[0-9]{6}(?:-\d+)?))?\.jsonl$")
_EVENT_CORE = ("ts", "iso", "pid", "event", "node", "block", "build_id")


def event_files(coresmith_dir: Path) -> list[Path]:
    """The active log and every rotated log, oldest rotation first, active last."""
    files = [p for p in coresmith_dir.glob("pipeline_events*.jsonl") if _EVENT_FILE_RE.match(p.name)]
    return sorted(files, key=lambda p: (p.name == "pipeline_events.jsonl", p.name))


def _canon(rec: dict) -> str:
    return hashlib.sha256(json.dumps(rec, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def load_events(coresmith_dir: Path, sources: Sources, arm: str, ledger: Ledger) -> dict:
    """Every graph event of every pipeline_events*.jsonl. Identity is the
    whole record: an identical record in another file is a copy (kept once,
    every copy referenced); identical lines inside one file are kept as
    separate events and counted."""
    events: list[dict] = []
    seen: dict[str, list[dict]] = {}
    files = []
    within_file_repeats = 0
    for path in event_files(coresmith_dir):
        active = path.name == "pipeline_events.jsonl"
        sid = sources.add(path, "graph event log (" + ("active" if active else "rotated at run start") + ")", arm)
        file_hashes = collections.Counter()
        n = 0
        first = last = None
        for line, rec, raw in read_jsonl(path):
            sources.stat(sid, "records")
            if rec is None:
                sources.stat(sid, "unparseable")
                continue
            n += 1
            h = _canon(rec)
            r = ref(sid, line)
            occurrence = file_hashes[h]
            file_hashes[h] += 1
            if occurrence:
                within_file_repeats += 1
            if occurrence < len(seen.get(h, [])):
                seen[h][occurrence]["copies"].append(r)
                sources.stat(sid, "duplicates")
                continue
            ts = rec.get("ts") if isinstance(rec.get("ts"), (int, float)) else None
            first = ts if first is None else min(first, ts or first)
            last = ts if last is None else max(last, ts or last)
            fields = {k: v for k, v in rec.items() if k not in _EVENT_CORE}
            ev = {"id": f"{sid}:{line}", "ts": ts, "iso": iso(ts), "pid": rec.get("pid"),
                  "event": str(rec.get("event") or ""), "node": str(rec.get("node") or ""),
                  "block": rec.get("block") or rec.get("block_name") or None, "build_id": rec.get("build_id") or None,
                  "fields": scrub(fields, ledger, drop=frozenset({"signature", "encrypted_content"})),
                  "src": r, "copies": [], "active_file": active}
            seen.setdefault(h, []).append(ev)
            events.append(ev)
            sources.stat(sid, "kept")
        files.append({"source": sid, "path": sources.items[sid]["path"], "active": active, "events": n,
                      "first_ts": first, "last_ts": last})
    events.sort(key=lambda e: (e["ts"] if e["ts"] is not None else float("inf"), e["id"]))
    for i, e in enumerate(events):
        e["seq"] = i
    return {"events": events, "files": files, "within_file_repeats": within_file_repeats}


# ------------------------------------------------------------------ daemon epochs
_GRAPH_EVENT_PREFIXES = ("graph_node_", "llm_start", "llm_end", "llm_error", "build_", "gate_result", "spec_",
                         "seed_rtl_used", "builds_reused", "acceptance_dv", "interrupt", "advisory")


def daemon_epochs(events: list[dict], coresmith_dir: Path) -> list[dict]:
    """Daemon processes identified by the writer pid of graph events (a CLI
    process writes only isolated ``stage_done`` / ``block_published``
    events; ``llm_call_start`` / heartbeats carry the child CLI pid). Each
    epoch: ``{epoch, pid, first_ts, last_ts, start_ts}``; ``start_ts`` comes
    from daemon.json for the live daemon when it matches."""
    by_pid: dict = {}
    for e in events:
        if e["event"] in ("llm_call_start", "llm_call_heartbeat", "llm_nonzero_exit", "llm_timed out"):
            continue
        if not e["event"].startswith(_GRAPH_EVENT_PREFIXES) or e["ts"] is None:
            continue
        d = by_pid.setdefault(e["pid"], {"pid": e["pid"], "first_ts": e["ts"], "last_ts": e["ts"], "events": 0})
        d["first_ts"] = min(d["first_ts"], e["ts"])
        d["last_ts"] = max(d["last_ts"], e["ts"])
        d["events"] += 1
    epochs = sorted((d for d in by_pid.values() if d["events"] >= 3), key=lambda d: d["first_ts"])
    info = {}
    try:
        info = json.loads((coresmith_dir / "daemon.json").read_text())
    except (OSError, ValueError):
        info = {}
    log_pids = []
    try:
        log_pids = [int(m) for m in re.findall(r"Started server process \[(\d+)\]",
                                              (coresmith_dir / "daemon.log").read_text(errors="replace"))]
    except OSError:
        pass
    for i, d in enumerate(epochs, 1):
        d["epoch"] = i
        d["start_ts"] = info.get("started_at") if info.get("pid") == d["pid"] else None
        d["in_daemon_log"] = d["pid"] in log_pids if log_pids else None
    return epochs


def epoch_for(epochs: list[dict], ts: float | None) -> dict | None:
    """The daemon epoch whose lifetime contains ``ts`` (epochs are ordered;
    an epoch lasts until the next one starts)."""
    if ts is None or not epochs:
        return None
    cur = None
    for d in epochs:
        start = d.get("start_ts") or d["first_ts"]
        if ts + 5 >= start:
            cur = d
    return cur


# ------------------------------------------------------------------ helper calls
def load_llm_calls(path: Path, sources: Sources, arm: str, ledger: Ledger) -> list[dict]:
    out = []
    if not path.is_file():
        return out
    sid = sources.add(path, "engine helper call log (llm_calls.jsonl; one record per finished call)", arm)
    for line, rec, raw in read_jsonl(path):
        sources.stat(sid, "records")
        if rec is None:
            sources.stat(sid, "unparseable")
            continue
        ts = rec.get("ts") if isinstance(rec.get("ts"), (int, float)) else None
        dur = rec.get("duration_s") if isinstance(rec.get("duration_s"), (int, float)) else None
        out.append({"line": line, "src": ref(sid, line), "ts": ts, "iso": iso(ts),
                    "start_ts": (ts - dur) if (ts is not None and dur is not None) else None,
                    "call_index": rec.get("call_index"), "run_name": rec.get("run_name") or "",
                    "graph": rec.get("graph") or "", "provider": rec.get("provider"), "model": rec.get("model"),
                    "duration_s": dur, "timeout": rec.get("timeout"), "timed_out": bool(rec.get("timed_out")),
                    "error": redact(rec.get("error") or "", ledger),
                    "usage": scrub(rec.get("usage") or {}, ledger),
                    "system_prompt": redact(rec.get("system_prompt") or "", ledger),
                    "user_prompt": redact(rec.get("user_prompt") or "", ledger),
                    "response": redact(rec.get("response") or "", ledger),
                    "lens": {k: rec.get(k) for k in ("system_prompt_len", "user_prompt_len", "response_len")}})
        sources.stat(sid, "kept")
    return out


def load_live_streams(d: Path, sources: Sources, arm: str) -> dict[int, dict]:
    """``{pid: {path, sid, meta, lines: [(i, event)], session_id}}``; the
    events are parsed by :mod:`native` (they are provider records)."""
    out = {}
    if not d.is_dir():
        return out
    for p in sorted(d.glob("*.json")):
        try:
            pid = int(p.stem)
        except ValueError:
            continue
        sid = sources.add(p, "engine live stream capture (provider stdout of one helper call)", arm)
        try:
            blob = json.loads(p.read_text(encoding="utf-8", errors="replace"))
        except ValueError:
            sources.stat(sid, "unparseable")
            continue
        lines = []
        session_id = None
        for i, ln in enumerate((blob.get("partial_stdout") or "").splitlines(), 1):
            if not ln.strip():
                continue
            sources.stat(sid, "records")
            try:
                ev = json.loads(ln)
            except ValueError:
                sources.stat(sid, "unparseable")
                continue
            if not isinstance(ev, dict):
                continue
            if ev.get("type") == "system" and ev.get("subtype") == "init" and ev.get("session_id"):
                session_id = session_id or ev.get("session_id")
            if ev.get("type") == "thread.started" and ev.get("thread_id"):
                session_id = session_id or ev.get("thread_id")
            lines.append((i, ev))
        meta = {k: blob.get(k) for k in ("pid", "model", "started_ts", "elapsed_s", "stdout_bytes", "stderr_bytes",
                                         "done", "done_ts")}
        out[pid] = {"path": p, "sid": sid, "meta": meta, "lines": lines, "session_id": session_id}
    return out


def load_codex_turns(path: Path, sources: Sources, arm: str) -> dict[tuple, dict]:
    """``codex_turns.jsonl`` grouped by call identity ``(pid, wall_start)``:
    ``{header, sid, lines: [(line, ts, event)], thread_id}``."""
    out: dict[tuple, dict] = {}
    if not path.is_file():
        return out
    sid = sources.add(path, "engine provider turn log (codex_turns.jsonl; identity header + raw provider event)", arm)
    for line, rec, raw in read_jsonl(path):
        sources.stat(sid, "records")
        if rec is None:
            sources.stat(sid, "unparseable")
            continue
        key = (rec.get("pid"), rec.get("wall_start"))
        g = out.setdefault(key, {"header": {k: rec.get(k) for k in ("pid", "wall_start", "call_index", "run_name",
                                                                   "process_scope")},
                                 "sid": sid, "lines": [], "thread_id": None})
        ev = rec.get("event") if isinstance(rec.get("event"), dict) else None
        if ev is None:
            sources.skip(sid, "no event")
            continue
        if ev.get("type") == "thread.started":
            g["thread_id"] = g["thread_id"] or ev.get("thread_id")
        g["lines"].append((line, rec.get("ts"), ev))
    return out


# ------------------------------------------------------------------ step logs
def load_step_logs(d: Path, sources: Sources, arm: str) -> list[dict]:
    """Tool step logs with their own header (the snapshot copy's mtime is the
    collection time, so only the header ``Timestamp`` dates a log)."""
    out = []
    if not d.is_dir():
        return out
    for p in sorted(d.rglob("*")):
        if not p.is_file():
            continue
        sid = sources.add(p, "engine tool step log", arm)
        head = {}
        try:
            with open(p, "r", encoding="utf-8", errors="replace") as fh:
                first = [next(fh, "") for _ in range(12)]
        except OSError:
            first = []
        for ln in first:
            ln = ln.rstrip("\n")
            if ln.startswith("=== ") and ln.endswith(" LOG ==="):
                head["kind"] = ln[4:-8].strip().lower()
            for key in ("Timestamp", "Block", "Attempt", "Command", "Return code"):
                if ln.startswith(key + ":"):
                    head[key.lower().replace(" ", "_")] = ln[len(key) + 1:].strip()
        stem = p.name[:-4] if p.name.endswith(".log") else p.name
        rnd = None
        m = re.search(r"\.round(\d+)$", stem)
        if m:
            rnd = int(m.group(1))
            stem = stem[:m.start()]
        step, _, att = stem.rpartition("_attempt")
        ts = None
        if head.get("timestamp"):
            ts = parse_iso(head["timestamp"] + ("Z" if "+" not in head["timestamp"] and "Z" not in head["timestamp"] else ""))
        out.append({"id": sid, "block": p.parent.name, "name": p.name, "step": step or stem,
                    "attempt": int(att) if att.isdigit() else None, "archived_round": rnd, "bytes": p.stat().st_size,
                    "ts": ts, "iso": iso(ts), "header": head, "path": p, "src": ref(sid)})
        sources.stat(sid, "records")
        sources.stat(sid, "kept")
    return out
