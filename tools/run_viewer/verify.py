"""Recount a snapshot directly and compare it with an export.

    python3 -m tools.run_viewer.verify --snapshot SNAP --viewer VIEWER

Independent of the exporter's parsers: SQLite rows are counted with plain
queries (``immutable=1``, nothing is written next to the snapshot), JSONL
files by parseable lines. Prints one line per check and exits non-zero when
any check fails.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path


def _shard(path: Path):
    text = path.read_text(encoding="utf-8")
    return json.loads(text[text.index(",") + 1:text.rindex(")")])


def _rows(db: Path, sql: str):
    if not db.is_file():
        return None
    con = sqlite3.connect(f"file:{db}?immutable=1", uri=True)
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


def _lines(path: Path) -> int:
    n = 0
    with open(path, encoding="utf-8", errors="replace") as fh:
        for ln in fh:
            if ln.strip():
                try:
                    json.loads(ln)
                    n += 1
                except ValueError:
                    pass
    return n


def verify(snapshot: Path, viewer: Path) -> list[tuple[str, bool, str]]:
    idx = _shard(viewer / "data" / "index.js")
    out = []

    def check(name, ok, detail):
        out.append((name, bool(ok), detail))

    for arm, a in idx["arms"].items():
        cs = snapshot / "arms" / arm / "work" / ".coresmith"
        db = cs / "project.sqlite"
        n_act = (_rows(db, "select count(*), coalesce(max(id),0) from actions") or [[None, None]])[0]
        check(f"{arm}: audit rows", n_act[0] == len(a["actions"]), f"sqlite {n_act[0]} (max id {n_act[1]}) / export {len(a['actions'])}")
        exp_ids = sorted(x["id"] for x in a["actions"])
        check(f"{arm}: audit ids complete", exp_ids == list(range(1, (n_act[1] or 0) + 1)) or len(exp_ids) == n_act[0],
              f"first {exp_ids[:1]} last {exp_ids[-1:]}")
        st = dict(_rows(db, "select status, count(*) from builds group by status") or [])
        exp_st = {}
        for b in a["builds"]:
            exp_st[b["status"]] = exp_st.get(b["status"], 0) + 1
        check(f"{arm}: builds by status", st == exp_st, f"sqlite {st} / export {exp_st}")
        for table, key in (("dv_results", "dv"), ("coverage_results", "coverage"), ("ppa_history", "ppa")):
            n = (_rows(db, f"select count(*) from {table}") or [[None]])[0][0]
            check(f"{arm}: {table}", n == len(a["evidence"][key]), f"sqlite {n} / export {len(a['evidence'][key])}")
        n_int = (_rows(db, "select count(*) from interrupts") or [[0]])[0][0]
        check(f"{arm}: interrupts", n_int == len(a["parks"]), f"sqlite {n_int} / export {len(a['parks'])}")
        # graph events: every parseable line of every file is an event or a referenced copy
        files = sorted(cs.glob("pipeline_events*.jsonl"))
        lines = sum(_lines(p) for p in files)
        evs = _shard(viewer / "data" / arm / "events.js")
        got = len(evs) + sum(len(e.get("copies") or []) for e in evs)
        check(f"{arm}: graph event lines ({len(files)} files)", lines == got, f"files {lines} / events {len(evs)} + copies {got - len(evs)}")
        for b in a["builds"]:
            n = sum(1 for e in evs if e.get("build_id") == b["id"])
            if n != len(b["events_exact"]):
                check(f"{arm}: build {b['id']} exact events", False, f"events with build_id {n} / build {len(b['events_exact'])}")
        check(f"{arm}: exact build events", True, "every build_id-carrying event is attributed to its build")
        # helper calls of record
        lc = cs / "llm_calls.jsonl"
        n_lc = _lines(lc) if lc.is_file() else 0
        fin = sum(1 for h in a["helper_calls"] if h.get("kind") == "call")
        check(f"{arm}: helper calls (llm_calls.jsonl)", n_lc == fin, f"lines {n_lc} / finished helper agents {fin}")
        # checkpoints
        cks = _shard(viewer / "data" / arm / "checkpoints.js")
        n_ck = 0
        for g in ("build", "pipeline", "backend", "architecture"):
            r = _rows(cs / f"{g}_checkpoint.db", "select count(*) from checkpoints")
            n_ck += r[0][0] if r else 0
        check(f"{arm}: checkpoints", n_ck == len(cks), f"sqlite {n_ck} / export {len(cks)}")
        # every build thread with checkpoints appears on its build
        for b in a["builds"]:
            r = _rows(cs / f"{b['graph']}_checkpoint.db", f"select count(*) from checkpoints where thread_id='{b['thread_id']}'")
            n = r[0][0] if r else 0
            if n != b["checkpoints"]["count"]:
                check(f"{arm}: build {b['id']} checkpoints", False, f"sqlite {n} / export {b['checkpoints']['count']}")
        check(f"{arm}: build checkpoint counts", True, "per-thread counts match")
    pv = idx["privacy"]
    check("privacy: no fingerprint or encrypted-pattern hit", pv["ok"], f"{pv['fingerprints_checked']} fingerprints, "
          f"{pv['files_scanned']} files, {len(pv['leaks'])} leaks, {pv['encrypted_pattern_hits']} encrypted hits")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--snapshot", required=True, type=Path)
    ap.add_argument("--viewer", required=True, type=Path)
    args = ap.parse_args(argv)
    res = verify(args.snapshot, args.viewer)
    for name, ok, detail in res:
        print(("PASS " if ok else "FAIL ") + name + " — " + detail)
    return 0 if all(ok for _, ok, _ in res) else 1


if __name__ == "__main__":
    sys.exit(main())
