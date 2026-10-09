"""CLI -> build / graph thread -> node events -> helper calls -> parks ->
evidence, with a label on every relationship.

Labels (shown on the provenance screen):

* ``exact``  -- an equal stable identifier: ``build_id`` field, thread id,
  interrupt id, session/thread id, ``call_index`` inside one daemon process,
  a build id string in an audited argv;
* ``strong`` -- unique candidate on content + time (a module named in a
  helper's run name and the helper ran inside that module's only build
  window; a block-scoped event of the same daemon process between a build's
  Init Block and Block Done events);
* ``weak``   -- time window only, or the closest of several candidates;
* ``unlinked``.

The engine's own lineage predicate (``state_store/builds.py``) is
re-evaluated twice on the snapshot rows: as the engine reads it (active event
file only, exact checkpoint namespace membership) and as this viewer reads it
(every event file, the recorded namespace matched to its checkpointed
ancestor on ``|`` boundaries). Both readings are shown; nothing is rewritten.
"""
from __future__ import annotations

import collections
import math
import re

from .sources import iso

BUILD_ID_RE = re.compile(r"\bb-[A-Za-z0-9_]+-\d{8}T\d{6}-[0-9a-f]{6}\b")
INTERRUPT_ID_RE = re.compile(r"\bint-[0-9a-f]{16}\b")
_RUN_NAME_BLOCK = re.compile(r"\[([^\]]+)\]")
_RETRY = re.compile(r"\s*-\s*Retry #\d+\s*$")
RUN_NAME_NODE = {"Generate Verilog": "Generate RTL", "Analyze Failure": "Diagnose Failure",
                 "Integration Testbench": "Integration DV", "Validation DV": "Validation DV",
                 "Flat Top Synthesis": "Flat Top Synthesis"}


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def run_name_parts(run_name: str) -> tuple[str, str | None]:
    """``("Generate Testbench", "Sdram Ctrl")`` from a helper run name."""
    m = _RUN_NAME_BLOCK.search(run_name or "")
    block = m.group(1) if m else None
    base = _RETRY.sub("", _RUN_NAME_BLOCK.sub("", run_name or "")).strip()
    return RUN_NAME_NODE.get(base, base), block


def _finite(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


# ------------------------------------------------------------------ namespaces
def ns_segments(ns: str | None) -> list[str]:
    return [s for s in (ns or "").split("|") if s]


def ns_scope(recorded: str | None, checkpointed: set[str]) -> dict:
    """The recorded completion namespace against the namespaces that have
    checkpoints: exact membership (the engine's predicate) and the longest
    checkpointed ancestor, compared segment by segment (``a:1|b:2`` has
    ancestor ``a:1``; ``a:12`` is not an ancestor of ``a:1``)."""
    segs = ns_segments(recorded)
    exact = (recorded or "") in checkpointed if recorded else False
    ancestor = None
    for k in range(len(segs), 0, -1):
        cand = "|".join(segs[:k])
        if cand in checkpointed:
            ancestor = cand
            break
    return {"recorded": recorded, "exact_member": exact, "checkpointed_ancestor": ancestor,
            "ancestor_depth": (len(ns_segments(ancestor)) if ancestor else 0), "recorded_depth": len(segs),
            "checkpointed": sorted(checkpointed)}


# ------------------------------------------------------------------ evidence
def verify_evidence(rows: dict, attempt: int, timing_required: bool) -> list[str]:
    """``builds.verify_build_evidence`` re-derived from snapshot rows."""
    problems = []
    dv = [r for r in rows["dv"] if r.get("scope") == "rtl"]
    if not dv:
        problems.append("no dv_results row for the build")
    else:
        last = dv[-1]
        if int(last.get("attempt") or 0) != attempt:
            problems.append(f"the latest dv_results row is attempt {last.get('attempt')}, not {attempt}")
        elif last.get("source") != "gate":
            problems.append("the latest dv_results row is not the gate's own verdict")
        elif last.get("passed") != 1 or last.get("skipped"):
            problems.append("the latest dv_results row is not a non-skipped pass")
    cov = rows["coverage"]
    if not cov:
        problems.append("no coverage_results row for the build")
    else:
        last = cov[-1]
        unc = last.get("uncovered")
        if isinstance(unc, str):
            import json
            try:
                unc = json.loads(unc)
            except ValueError:
                unc = {}
        unc = unc or {}
        if last.get("attempt") is None or int(last.get("attempt") or 0) != attempt:
            problems.append(f"the latest coverage row is not attempt {attempt}'s")
        elif unc.get("applicable") is False or last.get("pct") is None:
            problems.append("line coverage was not measured for this attempt")
        else:
            if not _finite(last.get("pct")):
                problems.append("coverage row has no finite percentage")
            if not _finite(unc.get("floor")):
                problems.append("coverage row has no declared floor")
            if unc.get("passed") is not True:
                problems.append("coverage closure is not a recorded pass")
    ppa = [r for r in rows["ppa"] if r.get("probe") == "synth"]
    if not ppa:
        problems.append("no synthesis ppa_history row for the build")
    else:
        last = ppa[-1]
        if int(last.get("attempt") or 0) != attempt:
            problems.append(f"the latest synthesis PPA row is attempt {last.get('attempt')}, not {attempt}")
        else:
            if not _finite(last.get("cells")):
                problems.append("no finite cell count")
            if last.get("ppa_ok") != 1:
                problems.append("no positive PPA verdict (ppa_ok)")
            if timing_required and not _finite(last.get("wns_ns")):
                problems.append("timing required but no finite WNS")
    return problems


# ------------------------------------------------------------------ main join
def build_lineage(arm: str, project: dict, checkpoints: dict, events: list[dict], epochs: list[dict],
                  helpers: list[dict], step_logs: list[dict]) -> dict:
    builds = project["builds"]
    actions = project["actions"]
    ev_by_build = collections.defaultdict(list)
    for e in events:
        if e.get("build_id"):
            ev_by_build[e["build_id"]].append(e)
    results = {(r.get("block"), r.get("kind")): r for r in project["results"]}
    evidence = {"dv": project["dv_results"], "coverage": project["coverage_results"], "ppa": project["ppa_history"]}
    by_build_rows = collections.defaultdict(lambda: {"dv": [], "coverage": [], "ppa": []})
    for k, rows in evidence.items():
        for r in rows:
            if r.get("build_id"):
                by_build_rows[r["build_id"]][k].append(r)
    out_builds = []
    block_events = [e for e in events if e.get("block")]
    for b in builds:
        bid = b["id"]
        res = b.get("result") or {}
        inputs = b.get("inputs") or {}
        exact = sorted(ev_by_build.get(bid, []), key=lambda e: e["seq"])
        init = next((e for e in exact if e["node"] == "Init Block" and e["event"] == "graph_node_enter"), None)
        done = next((e for e in reversed(exact) if e["node"] == "Block Done"), None)
        term = next((e for e in reversed(exact) if e["event"] in ("build_terminal", "build_parked")), None)
        lo = init["ts"] if init else b.get("started_at")
        hi = done["ts"] if done else (b.get("finished_at") or (term["ts"] if term else None))
        daemon_pid = init["pid"] if init else None
        # the next Init Block of the same module bounds an open window
        if hi is None and lo is not None:
            later = [e["ts"] for e in block_events if e["node"] == "Init Block" and e["block"] == b["module"]
                     and e["ts"] and e["ts"] > lo + 1 and e.get("build_id") != bid]
            hi = min(later) if later else None
        inferred = []
        if lo is not None:
            for e in block_events:
                if e.get("build_id") or e["block"] != b["module"] or e["ts"] is None:
                    continue
                if e["ts"] < lo - 0.5 or (hi is not None and e["ts"] > hi + 0.5):
                    continue
                if daemon_pid is not None and e["pid"] != daemon_pid:
                    continue
                inferred.append(e)
        # LLM lifecycle events name the block in run_name only
        llm_ev = []
        if lo is not None:
            for e in events:
                if e["node"] != "LLM" or e["event"] not in ("llm_start", "llm_end", "llm_error") or e["ts"] is None:
                    continue
                _, blk = run_name_parts((e.get("fields") or {}).get("run_name") or "")
                if blk and norm(blk) == norm(b["module"]) and e["ts"] >= lo - 0.5 and (hi is None or e["ts"] <= hi + 0.5):
                    if daemon_pid is None or e["pid"] == daemon_pid:
                        llm_ev.append(e)
        all_ev = sorted(exact + inferred + llm_ev, key=lambda e: e["seq"])
        segments = node_segments(all_ev)
        # checkpoints
        graph_ck = checkpoints.get(b.get("graph") or "") or {}
        thread_nss = (graph_ck.get("threads") or {}).get(b.get("thread_id") or "", {})
        ck_count = sum(len(v) for v in thread_nss.values())
        nss = set(thread_nss.keys())
        scope = ns_scope(res.get("checkpoint_ns"), nss) if res else None
        dispatch_scope = ns_scope(b.get("checkpoint_ns"), nss) if b.get("checkpoint_ns") else None
        ck_list = []
        for ns, cps in thread_nss.items():
            for c in cps:
                ck_list.append(c)
        ck_list.sort(key=lambda c: (c.get("ts") or 0, c["id"]))
        if b.get("graph") == "pipeline" and scope and scope["checkpointed_ancestor"]:
            anc = scope["checkpointed_ancestor"]
            ck_list = [c for c in ck_list if c["ns"] == anc or c["ns"].startswith(anc + "|")]
        rows = by_build_rows.get(bid, {"dv": [], "coverage": [], "ppa": []})
        # lineage readings
        active_nodes = {e["node"] for e in exact if e["active_file"]}
        all_nodes = {e["node"] for e in exact}

        def reading(nodes: set, ns_ok_fn) -> dict:
            reasons = []
            if b["status"] != "completed":
                reasons.append(f"build is {b['status']}")
            if not res.get("thread_id"):
                reasons.append("the completion names no graph thread")
            elif res.get("thread_id") != b.get("thread_id"):
                reasons.append("the completion ran on another thread than the recorded one")
            if ck_count <= 0:
                reasons.append(f"no checkpoints for thread {b.get('thread_id')!r} in the {b.get('graph')} graph")
            elif b.get("graph") == "pipeline" and not ns_ok_fn():
                reasons.append("the completion's checkpoint namespace is not recorded in the pipeline checkpoint")
            if "Init Block" not in nodes or "Block Done" not in nodes:
                reasons.append("graph events do not carry the build id through init and completion")
            if b["status"] == "completed":
                reasons.extend("evidence: " + p for p in verify_evidence(rows, int(res.get("attempt") or 0),
                                                                         bool(res.get("timing_required"))))
            return {"ok": not reasons, "reasons": reasons}

        engine_reading = reading(active_nodes, lambda: bool(scope and scope["exact_member"]))
        viewer_reading = reading(all_nodes, lambda: bool(scope and (scope["exact_member"] or scope["checkpointed_ancestor"])))
        best = (results.get((b["module"], "best")) or {}).get("value") or {}
        sup = (results.get((b["module"], "best_superseded")) or {}).get("value") or {}
        rec = {
            "id": bid, "module": b["module"], "entry": b["entry"], "graph": b["graph"], "thread_id": b["thread_id"],
            "status": b["status"], "run_id": b.get("run_id"), "requested_at": b.get("requested_at"),
            "started_at": b.get("started_at"), "finished_at": b.get("finished_at"), "error": b.get("error"),
            "dispatch_ns": b.get("checkpoint_ns"), "inputs": inputs, "worker": b.get("worker"), "seed": b.get("seed"),
            "result": res, "src": b["src"],
            "events_exact": [e["id"] for e in exact],
            "events_inferred": [e["id"] for e in inferred],
            "events_llm": [e["id"] for e in llm_ev],
            "events_in_rotated_files": sum(1 for e in exact if not e["active_file"]),
            "event_window": {"from": lo, "to": hi, "daemon_pid": daemon_pid,
                             "basis": "Init Block / Block Done events carrying this build id" if init else
                                      "build row started_at / finished_at (no init event found)"},
            "segments": segments,
            "checkpoints": {"graph": b.get("graph"), "thread": b.get("thread_id"), "count": ck_count,
                            "namespaces": {ns: len(cps) for ns, cps in thread_nss.items()},
                            "list": [c["id"] for c in ck_list]},
            "completion_ns": scope, "dispatch_ns_scope": dispatch_scope,
            "lineage": {"engine": engine_reading, "viewer": viewer_reading,
                        "active_file_nodes": sorted(active_nodes), "all_file_nodes": sorted(all_nodes)},
            "evidence": {k: [r["id"] for r in v] for k, v in rows.items()},
            "published": best.get("build_id") == bid, "superseded": sup.get("build_id") == bid,
            "attempts_seen": sorted({e["fields"].get("attempt") for e in all_ev if isinstance(e["fields"].get("attempt"), int)}),
            "actions": [], "helpers": [], "interrupts": [], "step_logs": [],
        }
        out_builds.append(rec)
    by_id = {b["id"]: b for b in out_builds}
    # ---------------- actions -> builds
    for a in actions:
        a.setdefault("builds", [])
        text = " ".join(a["argv"])
        for m in set(BUILD_ID_RE.findall(text)):
            if m in by_id:
                a["builds"].append({"build": m, "label": "exact", "basis": "build id in the audited argv"})
                by_id[m]["actions"].append({"action": a["id"], "label": "exact", "basis": "build id in the audited argv"})
        argv = a["argv"]
        pos = [t for t in argv if not t.startswith("-")]
        if len(pos) >= 3 and pos[0] == "build" and pos[1] == "module":
            mod = pos[2]
            cands = [b for b in out_builds if b["module"] == mod and b["requested_at"] is not None
                     and a["ts"] - 30 <= b["requested_at"] <= a["ts"] + 1]
            if len(cands) == 1:
                b = cands[0]
                basis = "`build module` of this module; the build row was requested within 30 s before the audit row"
                a["builds"].append({"build": b["id"], "label": "strong", "basis": basis, "dispatch": True})
                b["actions"].append({"action": a["id"], "label": "strong", "basis": basis, "dispatch": True})
    # ---------------- helpers -> builds / graph runs
    for h in helpers:
        call = h.get("engine_call") or {}
        st, en = call.get("start_ts"), call.get("ts")
        node, blk = run_name_parts(call.get("run_name") or "")
        h["builds"] = []
        if st is None:
            continue
        cands = []
        for b in out_builds:
            lo = b["event_window"]["from"] or b.get("started_at")
            hi = b["event_window"]["to"] or b.get("finished_at")
            if lo is None:
                continue
            if blk and norm(blk) != norm(b["module"]):
                continue
            if call.get("graph") and b["graph"] != call.get("graph"):
                continue
            if st >= lo - 2 and (hi is None or (en or st) <= hi + 5):
                cands.append(b)
        if len(cands) == 1:
            b = cands[0]
            basis = (f"helper run name names {blk!r}; the call ran inside the only {b['graph']}-graph window of that module"
                     if blk else "the call ran inside this build's window")
            label = "strong" if blk else "weak"
            seg = next((s for s in b["segments"] if norm(s["node"]) == norm(node) and s["enter_ts"] is not None
                        and s["enter_ts"] - 2 <= st and (s["exit_ts"] is None or (en or st) <= s["exit_ts"] + 5)), None)
            h["builds"].append({"build": b["id"], "label": label, "basis": basis,
                                "segment": seg["i"] if seg else None})
            b["helpers"].append({"agent": h["id"], "label": label, "basis": basis, "segment": seg["i"] if seg else None,
                                 "node": node})
            if seg is not None:
                seg.setdefault("helpers", []).append(h["id"])
        elif len(cands) > 1:
            h["builds"] = [{"build": b["id"], "label": "ambiguous", "basis": "several overlapping build windows"}
                           for b in cands]
    # ---------------- interrupts / decisions / resumes
    decisions = {d.get("interrupt_id"): d for d in project["decisions"] if d.get("interrupt_id")}
    parks = []
    for it in project["interrupts"]:
        p = {"id": it["id"], "graph": it["graph"], "node": it["node"], "block": it.get("block") or
             (it.get("payload") or {}).get("block_name"), "kind": it["kind"], "status": it["status"],
             "ts": it.get("ts"), "resolved_ts": it.get("resolved_ts"), "consumed_ts": it.get("consumed_ts"),
             "resolved_by": it.get("resolved_by"), "run_id": it.get("run_id"), "payload": it.get("payload"),
             "resolution": it.get("resolution"), "lg_interrupt_id": it.get("lg_interrupt_id"), "src": it["src"],
             "decision": None, "actions": [], "build": None}
        d = decisions.get(it["id"])
        if d:
            p["decision"] = {"id": d["id"], "action": d["action"], "reasoning": d.get("reasoning"), "actor": d.get("actor"),
                             "ts": d.get("ts"), "label": "exact", "basis": "decisions.interrupt_id", "src": d["src"]}
        if it["graph"] == "build" and p["block"]:
            cands = [b for b in out_builds if b["module"] == p["block"] and b["graph"] == "build"
                     and (b["started_at"] or 0) - 1 <= (p["ts"] or 0) <= ((b["finished_at"] or float("inf")) + 1)]
            if len(cands) == 1:
                p["build"] = {"build": cands[0]["id"], "label": "strong",
                              "basis": "build-graph park of this module inside the build's lifetime"}
                cands[0]["interrupts"].append({"interrupt": it["id"], "label": "strong"})
        for a in actions:
            text = " ".join(a["argv"])
            if it["id"] in text:
                p["actions"].append({"action": a["id"], "label": "exact", "basis": "interrupt id in the audited argv"})
            elif p.get("resolved_ts") and a["argv"] and (a["argv"][0] in ("resume",) or a["argv"][:2] in (
                    ["build", "resume"], ["backend", "resume"], ["run", "resume"])) and abs(a["ts"] - p["resolved_ts"]) <= 10:
                p["actions"].append({"action": a["id"], "label": "strong" if abs(a["ts"] - p["resolved_ts"]) <= 3 else "weak",
                                     "basis": "resume verb audited within seconds of the resolution"})
        for x in p["actions"]:
            for a in actions:
                if a["id"] == x["action"]:
                    a.setdefault("interrupts", []).append({"interrupt": it["id"], "label": x["label"]})
        parks.append(p)
    # ---------------- step logs -> builds
    for s in step_logs:
        s["builds"] = []
        if s.get("ts") is None:
            continue
        cands = [b for b in out_builds if norm(b["module"]) == norm(s["block"]) and b["started_at"]
                 and b["started_at"] - 2 <= s["ts"] <= (b["finished_at"] or float("inf")) + 2]
        if len(cands) == 1:
            s["builds"].append({"build": cands[0]["id"], "label": "strong",
                                "basis": "step log header timestamp inside the module's build lifetime"})
            cands[0]["step_logs"].append(s["id"])
    return {"builds": out_builds, "parks": parks}


def node_segments(events: list[dict]) -> list[dict]:
    """Ordered node iterations from enter/exit pairs (per block and node);
    an exit with no open enter is an instantaneous segment; other events
    attach to the innermost open segment of their block."""
    segs: list[dict] = []
    open_: dict[tuple, dict] = {}
    for e in events:
        if e["node"] == "LLM":
            continue
        key = (e.get("block"), e["node"])
        if e["event"] == "graph_node_enter":
            s = {"i": len(segs), "node": e["node"], "block": e.get("block"), "enter_ts": e["ts"], "exit_ts": None,
                 "enter": e["id"], "exit": None, "status": "open", "attempt": e["fields"].get("attempt"),
                 "events": [e["id"]]}
            if key in open_:
                open_[key]["status"] = "re-entered"
            open_[key] = s
            segs.append(s)
        elif e["event"] == "graph_node_exit":
            s = open_.pop(key, None)
            if s is None:
                s = {"i": len(segs), "node": e["node"], "block": e.get("block"), "enter_ts": None, "exit_ts": e["ts"],
                     "enter": None, "exit": e["id"], "status": None, "attempt": e["fields"].get("attempt"),
                     "events": [], "instant": True}
                segs.append(s)
            s["exit_ts"] = e["ts"]
            s["exit"] = e["id"]
            s["events"].append(e["id"])
            s["status"] = e["fields"].get("status") or "exited"
            if s.get("attempt") is None:
                s["attempt"] = e["fields"].get("attempt")
            s["duration_s"] = (s["exit_ts"] - s["enter_ts"]) if s["enter_ts"] is not None else None
        else:
            target = open_.get(key) or next((x for k, x in reversed(list(open_.items())) if k[0] == e.get("block")), None)
            if target is None:
                segs.append({"i": len(segs), "node": e["node"], "block": e.get("block"), "enter_ts": e["ts"],
                             "exit_ts": e["ts"], "enter": None, "exit": None, "status": e["event"],
                             "attempt": e["fields"].get("attempt"), "events": [e["id"]], "instant": True})
            else:
                target["events"].append(e["id"])
    for s in segs:
        s["enter_iso"] = iso(s["enter_ts"])
        s["exit_iso"] = iso(s["exit_ts"])
    return segs


def graph_runs(checkpoints: dict, events: list[dict], project: dict, builds: list[dict], parks: list[dict]) -> list[dict]:
    """Non-build graph threads (pipeline, backend, architecture): their
    checkpoint namespaces, the time span their checkpoints cover, the run ids
    seen, the events written in that span by nodes outside any block, the
    builds dispatched on the thread and their parks."""
    runs = []
    for graph, ck in checkpoints.items():
        if graph == "build":
            continue
        for thread, nss in (ck.get("threads") or {}).items():
            cps = [c for v in nss.values() for c in v]
            tss = [c["ts"] for c in cps if c.get("ts")]
            lo, hi = (min(tss), max(tss)) if tss else (None, None)
            evs = [e for e in events if lo is not None and e["ts"] is not None and lo - 2 <= e["ts"] <= hi + 600
                   and not e.get("block") and e["node"] not in ("LLM", "Stage", "Build", "daemon")]
            root = sorted(nss.get("", []), key=lambda c: (c.get("ts") or 0, c["id"]))
            runs.append({"id": f"{graph}:{thread}", "graph": graph, "thread": thread, "source": ck.get("source"),
                         "checkpoints": len(cps), "namespaces": {ns: len(v) for ns, v in nss.items()},
                         "first_ts": lo, "last_ts": hi,
                         "root_steps": [{"id": c["id"], "ts": c.get("ts"), "step": c.get("step"), "ran": c.get("ran"),
                                         "source_kind": c.get("source_kind"), "interrupt_writes": c.get("interrupt_writes"),
                                         "resume_writes": c.get("resume_writes")} for c in root],
                         "events": [e["id"] for e in evs],
                         "events_basis": "events of nodes outside any block, written from the thread's first checkpoint "
                                         "to 10 minutes after its last (weak: time window)",
                         "builds": [b["id"] for b in builds if b["graph"] == graph and b["thread_id"] == thread],
                         "parks": [p["id"] for p in parks if p["graph"] == graph],
                         "run_ids": sorted({b.get("run_id") for b in builds if b["graph"] == graph and b.get("run_id")} |
                                           {p.get("run_id") for p in parks if p["graph"] == graph and p.get("run_id")})})
    return runs
