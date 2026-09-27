# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""``coresmith verify ...`` + read-only state queries.

``register_subcommands(sub)`` wires the harness subcommands onto the existing
``bin/coresmith`` argparse tree. Uniform exit codes across every subcommand::

    0  pass
    1  fail
    2  usage / unknown block
    3  infra / timeout
    4  skip / cannot-judge

``--json`` is accepted on all subcommands (machine-readable output).

CRITICAL: this module MUST import without importing ``orchestrator.langgraph``
(``pipeline_helpers.PROJECT_ROOT`` freezes at import). Every heavy import is
therefore deferred into the handler bodies -- keep it that way (there is a unit
test asserting langgraph is not imported by importing this module).
"""

from __future__ import annotations

import json
import time
import sys

# Exit codes (single source of truth).
EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_USAGE = 2
EXIT_INFRA = 3
EXIT_SKIP = 4


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def _emit(args, payload: dict, human: str) -> None:
    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(human)


def _bootstrap(args):
    from orchestrator.harness.env import bootstrap_project_root
    return bootstrap_project_root(getattr(args, "project_root", None))


# ---------------------------------------------------------------------------
# Read-only queries (direct disk reads; scoreboard when present, else fallback)
# ---------------------------------------------------------------------------
def cmd_dv_status(args) -> int:
    try:
        root = _bootstrap(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_USAGE
    from orchestrator.state_store.store import Scoreboard

    block = getattr(args, "block", None)
    sb = Scoreboard(root)

    from orchestrator.state_store.project_db import open_project
    _pdb = open_project(root)
    rows = sb.latest_dv(block=block) if sb.exists() else []
    if rows:
        for r in rows:
            try:
                best = [x for x in _pdb.results(str(r.get("block"))) if x["kind"] == "best"]
                r["stale"] = bool(best) and (best[0]["ts"] or 0) > (r.get("ts") or 0)
            except Exception:  # noqa: BLE001
                r["stale"] = False
        payload = {"source": "scoreboard", "block": block, "rows": rows}
        human_lines = [
            f"{r.get('block'):<22} {r.get('scope'):<11} "
            f"{'PASS' if r.get('passed') else ('SKIP' if r.get('skipped') else 'FAIL')} "
            f"src={r.get('source')} attempt={r.get('attempt')} "
            f"tests={r.get('tests_passed')}/{r.get('tests_total')}"
            + ("  [stale]" if r.get("stale") else "")
            for r in rows
        ] or ["(no dv rows recorded)"]
        _emit(args, payload, "\n".join(human_lines))
        return EXIT_PASS

    # Fallback: the best results recorded in the project database
    rows = []
    try:
        for item in _pdb.results(block or None):
            # ``best`` = published pass (sim AND synth AND timing);
            # ``dv_best`` = DV pass only, not yet done.
            if item["kind"] not in ("best", "dv_best"):
                continue
            data = item["value"]
            rows.append({
                "block": item["block"], "scope": "rtl", "source": "results",
                "kind": item["kind"], "done": item["kind"] == "best",
                "passed": bool(data.get("sim_passed")),
                "attempt": data.get("attempt"),
                "tests_passed": data.get("tests_passed"),
                "tests_total": data.get("tests_total"),
            })
    except Exception:  # noqa: BLE001
        rows = []
    payload = {"source": "results", "block": block, "rows": rows}
    human = "\n".join(
        f"{r['block']:<22} rtl        "
        f"{'PASS' if r['passed'] else 'FAIL'} "
        f"({'done: sim+synth+timing' if r.get('done') else 'dv pass only'}) "
        f"tests={r.get('tests_passed')}/{r.get('tests_total')}"
        for r in rows
    ) or "(no DV rows or best results recorded)"
    _emit(args, payload, human)
    return EXIT_PASS


def cmd_ppa(args) -> int:
    try:
        root = _bootstrap(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_USAGE
    from orchestrator.state_store.store import Scoreboard

    block = args.block
    sb = Scoreboard(root)
    history = bool(getattr(args, "history", False))

    if sb.exists() and (sb.latest_ppa(block) or sb.ppa_rows(block)):
        rows = sb.ppa_rows(block) if history else [sb.latest_ppa(block)]
        rows = [r for r in rows if r]
        payload = {"source": "scoreboard", "block": block, "rows": rows}
        human = "\n".join(
            f"[{r.get('probe')}] ff={r.get('ff')} cells={r.get('cells')} "
            f"mem_bits={r.get('mem_bits')} area={r.get('area_um2')} "
            f"wns={r.get('wns_ns')} elaborated={r.get('elaborated')} "
            f"ppa_ok={r.get('ppa_ok')} (attempt={r.get('attempt')}, {r.get('source')})"
            for r in rows
        ) or "(no ppa rows)"
        _emit(args, payload, human)
        return EXIT_PASS

    # Disk fallback: parse syn/output/<b>/<b>_report.txt for FF count.
    report = root / "syn" / "output" / block / f"{block}_report.txt"
    if report.exists():
        try:
            from orchestrator.langgraph.ppa_check import count_flops_from_stat
            text = report.read_text(errors="replace")
            ff = count_flops_from_stat(text)
            payload = {
                "source": "disk", "block": block, "ff": ff,
                "report_path": str(report),
            }
            _emit(args, payload, f"{block}: ff={ff} (from {report.name})")
            return EXIT_PASS
        except Exception as exc:  # noqa: BLE001
            print(f"could not parse {report}: {exc}", file=sys.stderr)
            return EXIT_INFRA

    payload = {"source": "none", "block": block, "reason": "no ppa data"}
    _emit(args, payload, f"{block}: no PPA data (no scoreboard row, no synth report)")
    return EXIT_SKIP


def cmd_coverage(args) -> int:
    try:
        root = _bootstrap(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_USAGE
    from orchestrator.state_store.store import Scoreboard

    block = args.block
    sb = Scoreboard(root)
    row = sb.coverage_latest(block) if sb.exists() else None
    if not row:
        payload = {"source": "none", "block": block, "reason": "no coverage recorded"}
        _emit(
            args, payload,
            f"{block}: no coverage recorded "
            "(re-run `coresmith verify rtl <block> --coverage`)",
        )
        return EXIT_SKIP

    uncovered = []
    try:
        uncovered = json.loads(row.get("uncovered") or "[]")
    except Exception:  # noqa: BLE001
        uncovered = []
    payload = {
        "source": "scoreboard", "block": block,
        "points_total": row.get("points_total"),
        "points_hit": row.get("points_hit"),
        "pct": row.get("pct"),
    }
    if getattr(args, "uncovered", False):
        payload["uncovered"] = uncovered
    human = (
        f"{block}: {row.get('points_hit')}/{row.get('points_total')} "
        f"points ({row.get('pct')}%)"
    )
    if getattr(args, "uncovered", False):
        human += "\nUncovered:\n" + "\n".join(
            f"  {u.get('file')}:{u.get('line')}  {u.get('text')}"
            for u in uncovered[:200]
        )
    _emit(args, payload, human)
    return EXIT_PASS


def cmd_contracts(args) -> int:
    try:
        root = _bootstrap(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_USAGE
    block = args.block
    try:
        from orchestrator.langchain.agents.contract_lookup import load_block_contracts
        view = load_block_contracts(str(root), block)
    except Exception as exc:  # noqa: BLE001
        print(f"contract lookup failed: {exc}", file=sys.stderr)
        return EXIT_INFRA
    edges = view.get("edges") or []
    payload = {"block": block, "defaults": view.get("defaults") or {}, "edges": edges}
    human_lines = []
    if view.get("defaults"):
        human_lines.append(f"defaults: {view['defaults']}")
    for e in edges:
        human_lines.append(
            f"[{e.get('role')}] {e.get('producer_block')} -> "
            f"{e.get('consumer_block')} ({e.get('signal') or e.get('name') or '?'})"
        )
    _emit(args, payload, "\n".join(human_lines) or f"{block}: no contract edges")
    return EXIT_PASS


def _run(handler):
    """Wrap an int-returning handler so the CLI exits with its code."""
    def _f(args):
        raise SystemExit(handler(args))
    return _f


def _add_project_root(p) -> None:
    p.add_argument("--project-root", help="overrides $CORESMITH_PROJECT_ROOT")


def _add_json(p) -> None:
    p.add_argument("--json", action="store_true", help="machine-readable output")


def _state_db(args):
    from orchestrator.state_store.project_db import open_project
    return open_project(_bootstrap(args))



# ---------------------------------------------------------------------------
# Architect sitting (step 1): register / item / link / check / question / stage / status.
# The architect can only advance a run through these; they are the old graph
# nodes as deterministic tools.

def _problems_lines(problems):
    return [f"  [{q.get('severity','error')[:4]}] {q.get('code')} {q.get('where')}: {q.get('text')}" for q in problems]


def cmd_register(args) -> int:
    """``coresmith register <kind> <path> [--block b]`` -- parse, validate, record an artifact."""
    from orchestrator.harness.tools.register import register
    db = _state_db(args)
    res = register(db, db.root, args.kind, args.path, block=getattr(args, "block", "") or "", actor="cli")
    lines = [f"register {args.kind}: {'OK' if res.get('ok') else 'REFUSED'}"]
    if res.get("ok"):
        lines.append(f"  artifact {res['artifact']['kind']} v{res['artifact']['version']} sha {res['artifact']['sha']}; "
                     f"items {res.get('items')}; links {res.get('links')}")
    lines += _problems_lines(res.get("problems") or [])
    _emit(args, res, "\n".join(lines))
    return EXIT_PASS if res.get("ok") else EXIT_FAIL


def cmd_item(args) -> int:
    db = _state_db(args)
    verb = getattr(args, "verb", "list")
    if verb == "show":
        it = db.item(args.id)
        if not it:
            _emit(args, {"error": "no such item"}, f"no item {args.id}")
            return EXIT_USAGE
        payload = {"item": it, "links_out": db.links(from_id=args.id), "links_in": db.links(to_id=args.id),
                   "checks": db.checks(args.id)}
        lines = [f"{it['id']} [{it['kind']}] {it['status']} prio={it['priority'] or '-'}", f"  {it['text'][:300]}",
                 f"  acceptance: {it['acceptance'][:200] or '-'}", f"  model check: {it['model_check'][:200] or '-'}"]
        lines += [f"  -> {l['rel']} {l['to_id']}" for l in payload["links_out"]]
        lines += [f"  <- {l['rel']} {l['from_id']}" for l in payload["links_in"]]
        lines += [f"  check {c['kind']}: {c['status']} {c['evidence'][:120]}" for c in payload["checks"]]
        _emit(args, payload, "\n".join(lines))
        return EXIT_PASS
    if verb == "status":
        db.set_item_status(args.id, args.status)
        _emit(args, {"id": args.id, "status": args.status}, f"{args.id} -> {args.status}")
        return EXIT_PASS
    rows = db.items(kind=getattr(args, "kind", None) or None, artifact=getattr(args, "artifact", None) or None,
                    status=getattr(args, "status", None) or None, must_have=bool(getattr(args, "must", False)))
    lines = [f"{r['id']:<18} {r['kind']:<5} {r['status']:<12} {r['priority'] or '-':<12} {r['text'][:80]}" for r in rows] or ["(no items)"]
    _emit(args, {"items": rows}, "\n".join(lines))
    return EXIT_PASS


def cmd_link(args) -> int:
    db = _state_db(args)
    try:
        db.link_items(args.from_id, args.to_id, args.rel, source="cli")
    except ValueError as exc:
        _emit(args, {"error": str(exc)}, f"error: {exc}")
        return EXIT_USAGE
    _emit(args, {"from": args.from_id, "to": args.to_id, "rel": args.rel}, f"{args.from_id} -{args.rel}-> {args.to_id}")
    return EXIT_PASS


def cmd_check(args) -> int:
    db = _state_db(args)
    verb = getattr(args, "verb", "list")
    if verb == "add":
        try:
            cid = db.add_check(args.item, args.kind, args.status, evidence=args.evidence or "", sha=args.sha or "",
                               run_id=args.run_id or "", actor=args.actor or "cli")
        except ValueError as exc:
            _emit(args, {"error": str(exc)}, f"error: {exc}")
            return EXIT_USAGE
        _emit(args, {"id": cid}, f"check #{cid}: {args.item} {args.kind} {args.status}")
        return EXIT_PASS
    rows = db.checks(getattr(args, "item", None) or None, kind=getattr(args, "kind", None) or None,
                     latest=bool(getattr(args, "latest", False)))
    lines = [f"{c['item_id']:<18} {c['kind']:<14} {c['status']:<13} {c['sha'] or '-':<10} {(c['evidence'] or '')[:80]}" for c in rows] or ["(no checks)"]
    _emit(args, {"checks": rows}, "\n".join(lines))
    return EXIT_PASS


def cmd_question(args) -> int:
    db = _state_db(args)
    verb = getattr(args, "verb", "list")
    if verb == "add":
        qid = db.add_question(args.text, item_id=args.item or "", must_answer=not args.optional, asked_by=args.by or "architect")
        _emit(args, {"id": qid}, f"question Q{qid} recorded ({'must answer' if not args.optional else 'optional'})")
        return EXIT_PASS
    if verb == "answer":
        rid = None
        if args.ruling:
            rid = db.add_ruling("global", args.ruling, rationale=f"answers Q{args.id}", source=args.by or "human",
                                question_ref=f"question:{args.id}")
            db.export_rulings_view()
        ok = db.answer_question(int(args.id), args.ruling or args.answer or "", ruling_id=rid)
        _emit(args, {"id": int(args.id), "answered": ok, "ruling_id": rid},
              f"Q{args.id} {'answered' if ok else 'not open'}" + (f" by ruling R{rid}" if rid else ""))
        return EXIT_PASS if ok else EXIT_USAGE
    rows = db.questions(open_only=not getattr(args, "all", False))
    lines = [f"Q{q['id']:<4} {q['status']:<9} {'MUST' if q['must_answer'] else 'opt ':<5} {q['item_id'] or '-':<12} {q['text'][:90]}" for q in rows] or ["(no open questions)"]
    _emit(args, {"questions": rows}, "\n".join(lines))
    return EXIT_PASS


def cmd_stage(args) -> int:
    from orchestrator.state_store import stages as st
    db = _state_db(args)
    verb = getattr(args, "verb", "status")
    if verb == "next":
        res = st.advance(db, db.root, actor="cli")
        if res["advanced"]:
            lines = [f"stage {res['done']} DONE -> now {res['stage']}"]
        else:
            lines = [f"stage {res['stage']} BLOCKED:"]
        for b in res["blocked_by"]:
            lines.append(f"  {b['code']}: {b['text']}" + (f" [{b['count']}] {', '.join(b['ids'][:12])}" if b['ids'] else ""))
        _emit(args, res, "\n".join(lines))
        return EXIT_PASS if res["advanced"] else EXIT_FAIL
    res = st.status(db, db.root)
    lines = [f"stage {res['stage']} ({res['index'] + 1}/{len(res['stages'])}); done: {', '.join(res['done']) or '-'}",
             "can advance" if res["can_advance"] else "blocked by:"]
    for b in res["blocked_by"]:
        lines.append(f"  {b['code']}: {b['text']}" + (f" [{b['count']}] {', '.join(b['ids'][:12])}" if b['ids'] else ""))
    _emit(args, res, "\n".join(lines))
    return EXIT_PASS


def cmd_status(args) -> int:
    """One screen: stage, artifacts, item coverage, open questions, blockers."""
    from orchestrator.state_store import stages as st
    db = _state_db(args)
    s = st.status(db, db.root)
    arts = db.artifacts()
    items = db.items()
    by_status = {}
    for i in items:
        by_status[i["status"]] = by_status.get(i["status"], 0) + 1
    qs = db.questions(open_only=True)
    payload = {"stage": s, "artifacts": arts, "items": {"total": len(items), "by_status": by_status},
               "open_questions": len(qs), "blocks": [b["name"] for b in db.block_specs()]}
    lines = [f"stage: {s['stage']}  (done: {', '.join(s['done']) or '-'})",
             "artifacts: " + (", ".join(f"{a['kind']}@v{a['version']}" for a in arts) or "-"),
             f"items: {len(items)} {by_status}", f"open questions: {len(qs)}  blocks: {len(payload['blocks'])}",
             "blockers:" if s["blocked_by"] else "ready: coresmith stage next"]
    for b in s["blocked_by"]:
        lines.append(f"  {b['code']}: {b['text']}" + (f" [{b['count']}] {', '.join(b['ids'][:12])}" if b['ids'] else ""))
    _emit(args, payload, "\n".join(lines))
    return EXIT_PASS


def cmd_model(args) -> int:
    """``coresmith model init|build|run|eval|register --arch`` -- the executable SAD as tools."""
    from orchestrator.harness.tools import model as mt
    db = _state_db(args)
    verb = getattr(args, "verb", "build")
    if not getattr(args, "arch", False):
        from orchestrator.harness.tools import integrate as it
        if verb == "refine":
            blocks = [getattr(args, "block", "")] if getattr(args, "block", "") else None
            res = it.model_refine(db, db.root, blocks=blocks)
            fe = res.get("frd_eval") or {}
            lines = [f"model refine: {'OK' if res['ok'] else 'NOT PASSING'} build={res.get('build_ok')} smoke={res.get('smoke_ok')} "
                     f"frd_eval gate_ok={fe.get('gate_ok')}" + (f" missing={res.get('missing_models')}" if res.get("missing_models") else "")]
            if fe.get("summary"):
                lines.append(f"  {fe['summary'].get('counts')} failed={fe['summary'].get('failed', [])[:8]}")
            if not res.get("build_ok") and res.get("build_log"):
                lines.append(res["build_log"][-800:])
            _emit(args, res, "\n".join(lines))
            return EXIT_PASS if res["ok"] else EXIT_FAIL
        if verb == "eval":
            res = it.model_eval(db, db.root)
            s = res.get("summary") or {}
            lines = [f"FRD evaluation on the SoC model: {'PASS' if res.get('gate_ok') else 'NOT PASSING'}" + (f" ({res['error']})" if res.get("error") else "")]
            if s:
                lines.append(f"  {s['counts']} failed={s['failed'][:10]} unanswered_must={s['unanswered_must'][:10]}")
            _emit(args, res, "\n".join(lines))
            return EXIT_PASS if res.get("gate_ok") else EXIT_FAIL
        _emit(args, {"error": f"model {verb} needs --arch"}, f"error: model {verb} needs --arch (refine/eval work on the SoC model)")
        return EXIT_USAGE
    if verb == "init":
        res = mt.arch_init(db.root)
        _emit(args, res, f"{res['path']} {'created' if res.get('created') else 'exists'}" + (f"\n  {res['hint']}" if res.get("hint") else ""))
        return EXIT_PASS
    if verb == "build":
        res = mt.arch_build(db.root)
        lines = [f"arch model build: {'OK' if res['ok'] else 'FAILED'}"] + ([res["error"]] if res.get("error") else []) \
            + [f"  {q}" for q in res.get("problems") or []] + ([res["log"][-1500:]] if not res["ok"] and res.get("log") else [])
        _emit(args, res, "\n".join(lines))
        return EXIT_PASS if res["ok"] else (EXIT_INFRA if res.get("tool_error") else EXIT_FAIL)
    if verb == "run":
        res = mt.arch_run(db.root, ns=int(getattr(args, "ns", 10000) or 10000))
        st = res.get("stats") or {}
        lines = [f"arch model run: {'OK' if res['ok'] else 'FAILED'}"]
        if st:
            lines.append(f"  total_cycles={st.get('total_cycles')} dynamic_mw={st.get('dynamic_mw', 0):.3f} decerr={st.get('decerr')}")
            for l in st.get("links") or []:
                lines.append(f"  link {l['from']}->{l['to']}: {l['bytes']} B, {l['bytes_per_cycle']:.3f} B/cyc, util {l['utilization']:.2f}, max_out {l['max_outstanding']}")
        elif res.get("log"):
            lines.append(res["log"][-800:])
        _emit(args, res, "\n".join(lines))
        return EXIT_PASS if res["ok"] else EXIT_FAIL
    if verb == "eval":
        res = mt.arch_eval(db, db.root, timeout_s=getattr(args, "timeout", None), repairs=getattr(args, "repairs", None))
        s = res.get("summary") or {}
        lines = [f"arch FRD evaluation: {'PASS' if res.get('gate_ok') else 'NOT PASSING'}" + (f" ({res['error']})" if res.get("error") else "")]
        if s:
            lines.append(f"  {s['counts']} failed={s['failed'][:10]} unanswered_must={s['unanswered_must'][:10]}")
        if res.get("report"):
            lines.append(f"  report: {res['report']}")
        _emit(args, res, "\n".join(lines))
        return EXIT_PASS if res.get("gate_ok") else EXIT_FAIL
    if verb == "register":
        res = mt.arch_register(db, db.root)
        _emit(args, res, f"arch_model registered v{res['artifact']['version']}" if res.get("ok") else f"error: {res.get('error')}")
        return EXIT_PASS if res.get("ok") else EXIT_FAIL
    return EXIT_USAGE


def cmd_fabric(args) -> int:
    """``coresmith fabric derive`` -- FabricSpec from the measured link table."""
    from orchestrator.harness.tools import model as mt
    db = _state_db(args)
    res = mt.fabric_derive(db, db.root, name=getattr(args, "name", None) or None,
                           headroom=float(getattr(args, "headroom", 2.0) or 2.0), write=not getattr(args, "dry_run", False))
    if not res.get("ok"):
        _emit(args, res, "fabric derive: FAILED " + str(res.get("error") or res.get("problems")))
        return EXIT_FAIL
    f = res["fabric"]
    lines = [f"fabric '{f['name']}': {len(f['masters'])} masters x {len(f['slaves'])} slaves, data {f['data_width']} b, "
             f"outstanding {f['max_outstanding']} (busiest link {f['derived_from']['busiest_link_bytes_per_cycle']:.3f} B/cyc, headroom {f['derived_from']['headroom']})"]
    lines += [f"  master {m['name']}" for m in f["masters"]]
    lines += [f"  slave {s['name']} {s['protocol']} @ {s['base']} +{s['size']}" for s in f["slaves"]]
    if res.get("path"):
        lines.append(f"  written: {res['path']}")
    _emit(args, res, "\n".join(lines))
    return EXIT_PASS


def cmd_architect(args) -> int:
    """``coresmith architect start|status|stop`` -- the one long-lived architect session."""
    from orchestrator.architect import ArchitectSession
    root = _bootstrap(args)
    verb = getattr(args, "verb", "status")
    sess = ArchitectSession(root, model=getattr(args, "model", None) or None,
                            max_turns=int(getattr(args, "max_turns", 300) or 300),
                            max_sittings=int(getattr(args, "max_sittings", 12) or 12))
    if verb == "start":
        stop = sess.dir / "STOP"
        if stop.exists():
            stop.unlink()
        st = sess.run()
        _emit(args, st, f"architect {st.get('state')} at stage {st.get('stage')} after {st.get('sittings')} sitting(s)"
              + (f" (${float(st.get('cost_usd') or 0):.2f})" if st.get("cost_usd") else "")
              + (f": {st.get('stop_reason')}" if st.get("stop_reason") else ""))
        return EXIT_PASS if st.get("state") == "done" else EXIT_FAIL
    if verb == "stop":
        (sess.dir / "STOP").write_text(str(time.time()))
        _emit(args, {"stop_requested": True}, "stop requested (takes effect between sittings)")
        return EXIT_PASS
    st = sess.status()
    _emit(args, st, json.dumps(st, indent=2, default=str))
    return EXIT_PASS


def cmd_block_status(args) -> int:
    from orchestrator.harness.tools.block import block_status
    db = _state_db(args)
    st = block_status(db, db.root, args.block)
    lines = [f"{st['block']}: {'DONE' if st['done'] else 'pending'} tier={st['tier']} cluster={st['cluster'] or '-'}"
             + (" (primitive)" if st["primitive"] else ""),
             f"  rtl {st['rtl_path']} {'ok' if st['rtl_exists'] else 'MISSING'}; tb {st['tb_path']} {'ok' if st['tb_exists'] else 'MISSING'}",
             f"  edges {len(st['edges'])}, vips {len(st['vips'])}, owns {', '.join(st['owned_items']) or '-'}, attempts {st['attempts']}"]
    if st.get("best"):
        lines.append(f"  best: wns {st['best'].get('wns_ns')} gates {st['best'].get('gate_count')} attempt {st['best'].get('attempt')}")
    _emit(args, st, "\n".join(lines))
    return EXIT_PASS


def cmd_block_done(args) -> int:
    from orchestrator.harness.tools.block import block_done
    db = _state_db(args)
    from orchestrator.langgraph.pipeline_helpers import resolve_run_clock_mhz
    res = block_done(db, db.root, args.block,
                     target_clock_mhz=resolve_run_clock_mhz(getattr(args, "target_clock_mhz", None), db.root),
                     seed=getattr(args, "seed", None), actor=getattr(args, "actor", "") or "cluster")
    lines = [f"block-done {args.block}: {'PUBLISHED' if res['ok'] else 'REFUSED'}" + (f" -- {res.get('reason')}" if res.get("reason") else "")]
    for k, v in (res.get("stages") or {}).items():
        lines.append(f"  {k:<12} {'ok' if v.get('ok') else ('tool_error' if v.get('tool_error') else 'FAIL')} "
                     + " ".join(f"{kk}={vv}" for kk, vv in v.items() if kk not in ("ok", "details", "log") and vv not in (None, "", [], {}))[:200])
    _emit(args, res, "\n".join(lines))
    if res["ok"]:
        return EXIT_PASS
    return EXIT_INFRA if res.get("tool_error") else EXIT_FAIL


def cmd_schema(args) -> int:
    """``coresmith schema <kind>`` -- the document shape ``register`` expects."""
    from orchestrator.harness.tools.schema import SCHEMAS, schema
    kind = getattr(args, "kind", "") or ""
    if not kind:
        _emit(args, {"kinds": sorted(SCHEMAS)}, "kinds: " + ", ".join(sorted(SCHEMAS)) + "\n(coresmith schema <kind>)")
        return EXIT_PASS
    _emit(args, {"kind": kind, "schema": schema(kind)}, schema(kind))
    return EXIT_PASS if kind in SCHEMAS else EXIT_USAGE


def cmd_vip(args) -> int:
    from orchestrator.harness.tools import integrate as it
    db = _state_db(args)
    res = it.vip_generate(db, db.root)
    lines = [f"vip generate: {'OK' if res.get('ok') else 'FAILED'} {res.get('vips', 0)} VIP(s), {res.get('contract_slices', 0)} contract slice(s)"
             + (f" -- {res['error']}" if res.get("error") else "")]
    lines += [f"  {k}: {v}" for k, v in (res.get("errors") or {}).items()][:12]
    _emit(args, res, "\n".join(lines))
    return EXIT_PASS if res.get("ok") else EXIT_FAIL


def cmd_shell(args) -> int:
    from orchestrator.harness.tools import integrate as it
    db = _state_db(args)
    res = it.shell_assemble(db, db.root, tier=getattr(args, "tier", None), all_real=bool(getattr(args, "all_real", False)))
    if res.get("error"):
        _emit(args, res, f"shell assemble: {res['error']}")
        return EXIT_FAIL
    lines = [f"shell assemble: {'OK' if res['ok'] else 'NOT CLEAN'} top={res['top']} real={len(res['real'])} stubs={len(res['stubs'])} "
             f"wires={res['wires']} boundary_ports={res['boundary_ports']} elaborated={res['elaborated']}"]
    lines += [f"  wiring: {w}" for w in res.get("wiring_errors") or []][:10]
    lines += [f"  elab: {e}" for e in res.get("elab_errors") or []][:10]
    if res.get("hint"):
        lines.append("  " + res["hint"])
    _emit(args, res, "\n".join(lines))
    return EXIT_PASS if res["ok"] else EXIT_FAIL

def cmd_blocks(args) -> int:
    """The block queue from the project database."""
    db = _state_db(args)
    rows = db.blocks()
    _emit(args, {"blocks": rows}, "\n".join(
        f"{b.get('name'):<28} tier={b.get('tier')} rtl={b.get('rtl_target', '')}" for b in rows
    ) or "(no blocks recorded; run the architecture phase or import block specs)")
    return EXIT_PASS


def cmd_block(args) -> int:
    """One block: registry entry, contracts, constraints, attempts, results."""
    db = _state_db(args)
    b = db.block(args.block)
    if b is None:
        print(f"unknown block: {args.block}", file=sys.stderr)
        return EXIT_USAGE
    payload = {"block": b, "contract_edges": db.contract_edges_for_block(args.block),
               "constraints": db.constraints(args.block),
               "attempts": db.attempt_history(args.block),
               "diagnosis": db.diagnosis(args.block),
               "results": db.results(args.block)}
    human = (f"{args.block}: tier={b.get('tier')} rtl={b.get('rtl_target', '')} "
             f"edges={len(payload['contract_edges'])} constraints={len(payload['constraints'])} "
             f"attempts={len(payload['attempts'])} results={[r['kind'] for r in payload['results']]}")
    _emit(args, payload, human)
    return EXIT_PASS


def cmd_attempts(args) -> int:
    db = _state_db(args)
    rows = db.attempt_history(args.block)
    _emit(args, {"block": args.block, "attempts": rows}, "\n".join(
        f"attempt {r.get('attempt')}: {r.get('category')} -- {str(r.get('error', ''))[:100]}" for r in rows
    ) or "(no attempts recorded)")
    return EXIT_PASS


def cmd_constraints(args) -> int:
    db = _state_db(args)
    if getattr(args, "add", None):
        db.add_constraint(args.block, args.add, source="human")
    rows = db.constraints(args.block)
    _emit(args, {"block": args.block, "constraints": rows}, "\n".join(
        f"[{r.get('source')}] {r.get('rule')}" for r in rows) or "(no constraints)")
    return EXIT_PASS


def cmd_results(args) -> int:
    db = _state_db(args)
    rows = db.results(getattr(args, "block", None) or None)
    _emit(args, {"results": rows}, "\n".join(
        f"{r['block']:<28} {r['kind']:<14} {json.dumps(r['value'])[:100]}" for r in rows
    ) or "(no results recorded)")
    return EXIT_PASS


def cmd_settings(args) -> int:
    db = _state_db(args)
    if getattr(args, "set", None):
        name, _, value = args.set.partition("=")
        db.set_setting(name.strip(), value)
    _emit(args, {"settings": db.settings()}, "\n".join(
        f"{k}={v}" for k, v in db.settings().items()) or "(no settings)")
    return EXIT_PASS


def cmd_leases(args) -> int:
    """List the run's process leases; ``--steal NAME --reason R`` drops one."""
    db = _state_db(args)
    stolen = None
    if getattr(args, "steal", None):
        if not getattr(args, "reason", None):
            _emit(args, {"error": "--steal requires --reason"}, "--steal requires --reason")
            return EXIT_USAGE
        stolen = db.steal_lease(args.steal, args.reason)
    rows = db.leases()
    payload = {"leases": rows, "stolen": stolen}
    lines = [
        f"{r['name']:<28} pid={r['holder_pid']}@{r['holder_host']} "
        f"{'EXPIRED' if r['expired'] else 'live'} "
        f"expires_in={r['expires_ts'] - __import__('time').time():+.0f}s "
        f"meta={ {k: v for k, v in r['meta'].items() if k != 'ttl_s'} }"
        for r in rows
    ] or ["(no leases)"]
    if stolen:
        lines.insert(0, f"stole {stolen['name']} from pid {stolen['holder_pid']} ({args.reason})")
    _emit(args, payload, "\n".join(lines))
    return EXIT_PASS


def cmd_interrupts(args) -> int:
    """List parked interrupts; ``--resolve ID --action A [--feedback F]`` queues
    an answer (works with the daemon down; applied on its next tick)."""
    db = _state_db(args)
    resolved = None
    if getattr(args, "resolve", None):
        if not getattr(args, "action", None):
            _emit(args, {"error": "--resolve requires --action"}, "--resolve requires --action")
            return EXIT_USAGE
        resolution = {"action": args.action, "feedback": args.feedback or "",
                      "rationale": args.rationale or "", "block_actions": {}}
        resolved = db.resolve_interrupt(args.resolve, resolution, resolved_by="cli")
        if not resolved:
            _emit(args, {"error": f"no pending interrupt {args.resolve!r}"},
                  f"no pending interrupt {args.resolve!r}")
            return EXIT_USAGE
    rows = db.interrupts(status="pending" if getattr(args, "pending", False) else None)
    lines = [
        f"{r['id']:<22} {r['status']:<9} {r['graph']:<12} {r['node'][:28]:<28} "
        f"block={r['block'] or '-'} kind={r['kind']} "
        f"actions={r['payload'].get('supported_actions', [])}"
        for r in rows
    ] or ["(no interrupts recorded)"]
    if resolved:
        lines.insert(0, f"queued action={args.action!r} for {args.resolve}")
    _emit(args, {"interrupts": rows, "resolved": args.resolve if resolved else None},
          "\n".join(lines))
    return EXIT_PASS


def cmd_ruling(args) -> int:
    """``coresmith ruling add|list|revoke`` -- the operator rulings channel (C2)."""
    db = _state_db(args)
    from orchestrator.state_store.rulings import apply_ruling_to_interrupts
    verb = getattr(args, "verb", "list")
    if verb == "add":
        try:
            rid = db.add_ruling(args.scope, args.text, rationale=args.rationale or "",
                                source=args.source or "human",
                                question_ref=args.question_ref or None,
                                supersedes_id=args.supersedes)
        except ValueError as exc:
            _emit(args, {"error": str(exc)}, f"error: {exc}")
            return EXIT_USAGE
        ruling = db.ruling(rid)
        resolved = apply_ruling_to_interrupts(db, ruling)
        db.export_rulings_view()
        payload = {"id": rid, "ruling": ruling, "resolved_interrupts": resolved,
                   "conflicts": ruling["conflicts"]}
        lines = [f"ruling R{rid} [{ruling['scope']}] recorded"]
        if resolved:
            lines.append(f"resolved {len(resolved)} pending interrupt(s): {resolved}")
        if ruling["conflicts"]:
            lines.append(f"WARNING conflicts with requirements.md: {ruling['conflicts']}")
        _emit(args, payload, "\n".join(lines))
        return EXIT_PASS
    if verb == "revoke":
        ok = db.revoke_ruling(args.id, args.reason or "")
        db.export_rulings_view()
        _emit(args, {"revoked": ok, "id": args.id},
              f"R{args.id} {'revoked' if ok else 'not found or already revoked'}")
        return EXIT_PASS if ok else EXIT_USAGE
    rows = db.rulings(scope=getattr(args, "scope", None) or None,
                      active_only=not getattr(args, "all", False))
    lines = [f"R{r['id']:<4} {r['scope']:<24} {r['source']:<10} "
             f"{'REVOKED ' if r.get('revoked_ts') else ''}{r['text'][:90]}"
             + (f"  (q: {r['question_ref']})" if r.get('question_ref') else "")
             for r in rows] or ["(no rulings)"]
    _emit(args, {"rulings": rows}, "\n".join(lines))
    return EXIT_PASS


def _register_state(sub) -> None:
    """Read (and a few write) commands over the project database."""
    rp = sub.add_parser("ruling", help="operator rulings: add | list | revoke")
    rsub = rp.add_subparsers(dest="verb")
    ra = rsub.add_parser("add", help="record a binding ruling")
    _add_project_root(ra)
    _add_json(ra)
    ra.add_argument("--scope", required=True, help="global | arch | block:<name> | edge:<edge_id>")
    ra.add_argument("--text", required=True)
    ra.add_argument("--rationale", default="")
    ra.add_argument("--source", default="human", choices=["human", "chip_lead", "coordinator"])
    ra.add_argument("--question-ref", dest="question_ref", default=None,
                    help="interrupt:<id> | prd:<qid> | prd:* | block:<name>:<kind>")
    ra.add_argument("--supersedes", type=int, default=None)
    ra.set_defaults(func=_run(cmd_ruling), verb="add")
    rl = rsub.add_parser("list", help="list rulings")
    _add_project_root(rl)
    _add_json(rl)
    rl.add_argument("--scope", default=None)
    rl.add_argument("--all", action="store_true", help="include revoked/superseded")
    rl.set_defaults(func=_run(cmd_ruling), verb="list")
    rr = rsub.add_parser("revoke", help="revoke a ruling")
    _add_project_root(rr)
    _add_json(rr)
    rr.add_argument("id", type=int)
    rr.add_argument("--reason", default="")
    rr.set_defaults(func=_run(cmd_ruling), verb="revoke")
    it = sub.add_parser("interrupts", help="parked interrupts (project database)")
    _add_project_root(it)
    _add_json(it)
    it.add_argument("--pending", action="store_true", help="only status=pending")
    it.add_argument("--resolve", default=None, metavar="ID", help="queue an answer for ID")
    it.add_argument("--action", default=None, help="the answer's action (with --resolve)")
    it.add_argument("--feedback", default=None)
    it.add_argument("--rationale", default=None)
    it.set_defaults(func=_run(cmd_interrupts))

    # --- architect sitting tools -------------------------------------------
    rg = sub.add_parser("register", help="register an artifact (prd|sad|frd|ers|block_diagram|contracts|abi|uarch|arch_model|harness)")
    rg.add_argument("kind"); rg.add_argument("path"); rg.add_argument("--block", default="")
    _add_project_root(rg); _add_json(rg); rg.set_defaults(func=_run(cmd_register))
    ip = sub.add_parser("item", help="ontology items: list | show | status")
    isub = ip.add_subparsers(dest="verb")
    il = isub.add_parser("list"); il.add_argument("--kind"); il.add_argument("--artifact"); il.add_argument("--status"); il.add_argument("--must", action="store_true")
    _add_project_root(il); _add_json(il); il.set_defaults(func=_run(cmd_item), verb="list")
    ish = isub.add_parser("show"); ish.add_argument("id"); _add_project_root(ish); _add_json(ish); ish.set_defaults(func=_run(cmd_item), verb="show")
    ist = isub.add_parser("status"); ist.add_argument("id"); ist.add_argument("status"); _add_project_root(ist); _add_json(ist); ist.set_defaults(func=_run(cmd_item), verb="status")
    lk = sub.add_parser("link", help="link two items: <from> <to> <rel> (derives_from|owned_by|verified_by|cites|covers)")
    lk.add_argument("from_id"); lk.add_argument("to_id"); lk.add_argument("rel"); _add_project_root(lk); _add_json(lk); lk.set_defaults(func=_run(cmd_link))
    ck = sub.add_parser("check", help="tool verdicts about items: add | list")
    csub = ck.add_subparsers(dest="verb")
    ca = csub.add_parser("add"); ca.add_argument("item"); ca.add_argument("kind"); ca.add_argument("status")
    ca.add_argument("--evidence", default=""); ca.add_argument("--sha", default=""); ca.add_argument("--run-id", dest="run_id", default=""); ca.add_argument("--actor", default="")
    _add_project_root(ca); _add_json(ca); ca.set_defaults(func=_run(cmd_check), verb="add")
    cl = csub.add_parser("list"); cl.add_argument("--item"); cl.add_argument("--kind"); cl.add_argument("--latest", action="store_true")
    _add_project_root(cl); _add_json(cl); cl.set_defaults(func=_run(cmd_check), verb="list")
    qp = sub.add_parser("question", help="item-scoped questions: add | answer | list")
    qsub = qp.add_subparsers(dest="verb")
    qa = qsub.add_parser("add"); qa.add_argument("text"); qa.add_argument("--item", default=""); qa.add_argument("--optional", action="store_true"); qa.add_argument("--by", default="")
    _add_project_root(qa); _add_json(qa); qa.set_defaults(func=_run(cmd_question), verb="add")
    qn = qsub.add_parser("answer"); qn.add_argument("id"); qn.add_argument("--ruling", default=""); qn.add_argument("--answer", default=""); qn.add_argument("--by", default="")
    _add_project_root(qn); _add_json(qn); qn.set_defaults(func=_run(cmd_question), verb="answer")
    ql = qsub.add_parser("list"); ql.add_argument("--all", action="store_true"); _add_project_root(ql); _add_json(ql); ql.set_defaults(func=_run(cmd_question), verb="list")
    sp = sub.add_parser("stage", help="the run's state machine: status | next")
    ssub = sp.add_subparsers(dest="verb")
    ss = ssub.add_parser("status"); _add_project_root(ss); _add_json(ss); ss.set_defaults(func=_run(cmd_stage), verb="status")
    sn = ssub.add_parser("next"); _add_project_root(sn); _add_json(sn); sn.set_defaults(func=_run(cmd_stage), verb="next")
    stp = sub.add_parser("status", help="one-screen run status (stage, artifacts, items, questions, blockers)")
    _add_project_root(stp); _add_json(stp); stp.set_defaults(func=_run(cmd_status))
    mp = sub.add_parser("model", help="executable models: init | build | run | eval | register (--arch)")
    msub = mp.add_subparsers(dest="verb")
    for v, h in (("init", "write a template model/arch/arch_model.json"), ("build", "generate + compile"),
                 ("run", "run the smoke scenario; writes stats.json"), ("eval", "FRD evaluation (agent-authored harness; --arch or the SoC model)"),
                 ("register", "register the arch_model artifact"),
                 ("refine", "per-block SystemC models + assembly + build + smoke + FRD eval (--block b re-authors one)")):
        mv = msub.add_parser(v, help=h); mv.add_argument("--arch", action="store_true")
        if v == "refine":
            mv.add_argument("--block", default=""); mv.add_argument("--all", action="store_true")
        if v == "run":
            mv.add_argument("--ns", type=int, default=10000)
        if v == "eval":
            mv.add_argument("--timeout", type=int); mv.add_argument("--repairs", type=int)
        _add_project_root(mv); _add_json(mv); mv.set_defaults(func=_run(cmd_model), verb=v)
    fp = sub.add_parser("fabric", help="fabric tools: derive (FabricSpec from the arch model's link table)")
    fsub = fp.add_subparsers(dest="verb")
    fd = fsub.add_parser("derive"); fd.add_argument("--name"); fd.add_argument("--headroom", type=float, default=2.0)
    fd.add_argument("--dry-run", dest="dry_run", action="store_true")
    _add_project_root(fd); _add_json(fd); fd.set_defaults(func=_run(cmd_fabric), verb="derive")
    ap = sub.add_parser("architect", help="the architect sitting: start | status | stop")
    asb = ap.add_subparsers(dest="verb")
    a1 = asb.add_parser("start", help="run sittings until the run enters the blocks stage (foreground)")
    a1.add_argument("--model"); a1.add_argument("--max-turns", dest="max_turns", type=int, default=300)
    a1.add_argument("--max-sittings", dest="max_sittings", type=int, default=12)
    _add_project_root(a1); _add_json(a1); a1.set_defaults(func=_run(cmd_architect), verb="start")
    a2 = asb.add_parser("status"); _add_project_root(a2); _add_json(a2); a2.set_defaults(func=_run(cmd_architect), verb="status")
    a3 = asb.add_parser("stop"); _add_project_root(a3); _add_json(a3); a3.set_defaults(func=_run(cmd_architect), verb="stop")
    bs = sub.add_parser("block-status", help="one block: paths, edges, VIPs, owned items, published pass")
    bs.add_argument("block"); _add_project_root(bs); _add_json(bs); bs.set_defaults(func=_run(cmd_block_status))
    bd = sub.add_parser("block-done", help="the block gate: conformance -> DV -> synth -> timing; publishes best on a pass")
    bd.add_argument("block"); bd.add_argument("--target-clock-mhz", dest="target_clock_mhz", type=float, default=None)
    bd.add_argument("--seed", type=int); bd.add_argument("--actor", default="")
    _add_project_root(bd); _add_json(bd); bd.set_defaults(func=_run(cmd_block_done))
    sc = sub.add_parser("schema", help="the document shape `register <kind>` expects (prd|sad|frd|ers|block_diagram|contracts|abi|uarch|arch_model)")
    sc.add_argument("kind", nargs="?", default=""); _add_json(sc); sc.set_defaults(func=_run(cmd_schema))
    vp = sub.add_parser("vip", help="interface VIPs: generate (from the registered contracts)")
    vsub = vp.add_subparsers(dest="verb")
    vg = vsub.add_parser("generate"); _add_project_root(vg); _add_json(vg); vg.set_defaults(func=_run(cmd_vip), verb="generate")
    shp = sub.add_parser("shell", help="chip shell: assemble (real RTL for published blocks, stubs for the rest; elaborate)")
    shsub = shp.add_subparsers(dest="verb")
    sha = shsub.add_parser("assemble"); sha.add_argument("--tier"); sha.add_argument("--all-real", dest="all_real", action="store_true")
    _add_project_root(sha); _add_json(sha); sha.set_defaults(func=_run(cmd_shell), verb="assemble")
    ls = sub.add_parser("leases", help="process leases held in the project database")
    _add_project_root(ls)
    _add_json(ls)
    ls.add_argument("--steal", default=None, metavar="NAME", help="drop a lease by name")
    ls.add_argument("--reason", default=None, help="why the lease is being stolen")
    ls.set_defaults(func=_run(cmd_leases))
    for name, handler, help_ in (
        ("blocks", cmd_blocks, "the block queue (project database)"),
        ("results", cmd_results, "recorded gate results per block"),
        ("settings", cmd_settings, "run settings (engine SHA, contract version, ...)"),
    ):
        p = sub.add_parser(name, help=help_)
        _add_project_root(p)
        _add_json(p)
        if name == "results":
            p.add_argument("block", nargs="?", default="")
        if name == "settings":
            p.add_argument("--set", default=None, help="NAME=VALUE")
        p.set_defaults(func=_run(handler))
    for name, handler, help_ in (
        ("block", cmd_block, "everything recorded for one block"),
        ("attempts", cmd_attempts, "a block's attempt history"),
        ("constraints", cmd_constraints, "a block's constraint ledger (--add RULE appends)"),
    ):
        p = sub.add_parser(name, help=help_)
        _add_project_root(p)
        _add_json(p)
        p.add_argument("block")
        if name == "constraints":
            p.add_argument("--add", default=None)
        p.set_defaults(func=_run(handler))


def register_subcommands(sub) -> None:
    """Register harness subcommands on the ``bin/coresmith`` subparser action."""
    _register_verify(sub)
    _register_state(sub)
    _register_queries(sub)
    _register_tool(sub)


def _register_queries(sub) -> None:
    # dv-status [block]
    ds = sub.add_parser("dv-status", help="DV scoreboard (per-block pass/fail)")
    _add_project_root(ds)
    _add_json(ds)
    ds.add_argument("block", nargs="?", help="restrict to one block")
    ds.set_defaults(func=_run(cmd_dv_status))

    # complexity [block] -- decomposition checker


    # ppa <block> [--history]
    pp = sub.add_parser("ppa", help="PPA (FF/area/cells) for a block")
    _add_project_root(pp)
    _add_json(pp)
    pp.add_argument("block")
    pp.add_argument("--history", action="store_true", help="all rows, not just latest")
    pp.set_defaults(func=_run(cmd_ppa))

    # coverage <block> [--uncovered]
    cv = sub.add_parser("coverage", help="coverage summary for a block")
    _add_project_root(cv)
    _add_json(cv)
    cv.add_argument("block")
    cv.add_argument("--uncovered", action="store_true", help="list uncovered points")
    cv.set_defaults(func=_run(cmd_coverage))

    # contracts <block>
    ct = sub.add_parser("contracts", help="interface contracts for a block")
    _add_project_root(ct)
    _add_json(ct)
    ct.add_argument("block")
    ct.set_defaults(func=_run(cmd_contracts))



def _register_verify(sub) -> None:
    """Verify subcommands (implemented in the harness ``verify`` module)."""
    # Deferred: verify handlers live in orchestrator.harness.cli_verify to keep
    # this module langgraph-free at import. Wired in the harness+verify commit.
    try:
        from orchestrator.harness import cli_verify
    except Exception:  # noqa: BLE001
        return
    cli_verify.register_verify(sub, _run, _add_project_root, _add_json)


def _register_tool(sub) -> None:
    """``coresmith tool <verb>`` + ``coresmith pdk info`` (deployment EDA verbs).

    Deferred import (like ``_register_verify``): cli_tool stays langgraph-free at
    import; the registry/deployment imports it triggers are inside its handlers.
    """
    try:
        from orchestrator.harness import cli_tool
    except Exception:  # noqa: BLE001
        return
    cli_tool.register_tool(sub, _run, _add_project_root, _add_json)
    cli_tool.register_pdk(sub, _run, _add_project_root, _add_json)
