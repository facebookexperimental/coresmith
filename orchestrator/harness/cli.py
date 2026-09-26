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
