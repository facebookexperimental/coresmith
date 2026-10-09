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

import contextlib
import json
import sys
import time

# Exit codes (single source of truth).
EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_USAGE = 2
EXIT_INFRA = 3
EXIT_SKIP = 4


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
# First line of the last human-readable output (the ``actions`` log summary).
_LAST_SUMMARY = ""


def _emit(args, payload: dict, human: str) -> None:
    global _LAST_SUMMARY
    _LAST_SUMMARY = (str(human or "").splitlines() or [""])[0][:200]
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


def _record_action(args, handler, rc) -> None:
    """Best-effort: log this invocation into the project DB ``actions`` table.

    Never raises; skipped for ``coresmith actions`` itself and when there is no
    project root or no project database yet (a verb must not create state).
    """
    try:
        argv = _cli_argv(args)
        if argv is None:
            argv = [str(a) for a in sys.argv[1:]]
        if handler is cmd_actions or (argv and argv[0] == "actions"):
            return
        from orchestrator.harness.env import bootstrap_project_root
        root = bootstrap_project_root(getattr(args, "project_root", None))
        from orchestrator.state_store.project_db import DB_NAME, ProjectDB
        if not (root / ".coresmith" / DB_NAME).is_file():
            return
        db = ProjectDB(root)
        db.ensure_schema()
        db.record_action(argv, rc if isinstance(rc, int) or rc is None else EXIT_FAIL, summary=_LAST_SUMMARY,
                         actor=_role() or _actor_env() or "cli")
    except Exception:  # noqa: BLE001 - the audit log must never break a verb
        pass


# ---------------------------------------------------------------------------
# Role-scoped CLI: CORESMITH_ROLE=architect|worker|watchdog (unset: no restriction)
ROLE_ENV = "CORESMITH_ROLE"
_ALL = None   # every sub-verb
# verb -> allowed sub-verbs (``_ALL`` = any; a set may contain None = the bare verb)
ROLE_ALLOW: dict[str, dict] = {
    # the watchdog keeps the run alive: lifecycle (daemon, run start/pause),
    # observation (stage status, question list, state check) -- never design authority
    "watchdog": {"daemon": _ALL, "state": _ALL, "interrupts": _ALL, "leases": _ALL, "actions": _ALL,
                 "resume": _ALL, "logs": _ALL, "status": _ALL, "run": {"start", "pause"},
                 "stage": {"status"}, "question": {"list"}},
    "worker": {"tool": _ALL, "target": _ALL, "pdk": _ALL, "block-status": _ALL, "block-done": _ALL, "verify": _ALL,
               "frd": {None, "verifier", "verifiers", "show", "list"}, "check": {"add"}, "question": {"add"},
               "constraints": _ALL, "blocks": _ALL, "results": _ALL, "contracts": _ALL, "dv-status": _ALL,
               "ppa": _ALL, "coverage": _ALL, "status": _ALL, "actions": _ALL},
}
# architect: the Architect (the coding agent calling the CLI) is unrestricted; the
# role is identity metadata for the actions log (``actions.actor``), never a limit
# (an empty deny-set allows every verb, including `daemon` and `run start`).
ROLE_DENY: dict[str, dict] = {"architect": {}}
ROLES = ("architect", "worker", "watchdog")
_VALUED_TOP_OPTS = ("--project-root",)


def _role() -> str:
    import os
    return (os.environ.get(ROLE_ENV) or "").strip().lower()


def _actor_env() -> str:
    """``CORESMITH_ACTOR``: who is running the CLI when no role says so (a
    cluster worker session sets ``worker:<cluster>``)."""
    import os
    return (os.environ.get("CORESMITH_ACTOR") or "").strip()


def role_verb(argv) -> tuple[str | None, str | None]:
    """The first two positional tokens of a ``coresmith`` argv (verb, sub-verb)."""
    pos, skip = [], False
    for tok in [str(a) for a in (argv or [])]:
        if skip:
            skip = False
            continue
        if tok.startswith("-"):
            skip = tok in _VALUED_TOP_OPTS
            continue
        pos.append(tok)
        if len(pos) == 2:
            break
    return (pos[0] if pos else None), (pos[1] if len(pos) > 1 else None)


def role_allows(role: str | None, argv) -> bool:
    """Whether ``role`` may run the ``coresmith`` command ``argv`` (argv
    without the program name). An empty role allows everything; an unknown
    role allows nothing; an argv without a verb (``--help``) is allowed."""
    role = (role or "").strip().lower()
    if not role:
        return True
    verb, sub = role_verb(argv)
    if verb is None:
        return True
    if role in ROLE_DENY:
        deny = ROLE_DENY[role]
        if verb not in deny:
            return True
        return deny[verb] is not _ALL and sub not in deny[verb]
    allow = ROLE_ALLOW.get(role)
    if allow is None or verb not in allow:
        return False
    subs = allow[verb]
    return subs is _ALL or sub in subs


def role_refusal(role: str, argv) -> str:
    """``ROLE_FORBIDDEN <role> may not run <verb>`` for a refused argv."""
    verb, sub = role_verb(argv)
    rule = (ROLE_DENY.get(role) or ROLE_ALLOW.get(role) or {}).get(verb, _ALL)
    what = f"{verb} {sub}" if sub and rule is not _ALL else (verb or "")
    suffix = "" if role in ROLES else f" (unknown role; {ROLE_ENV} must be one of {', '.join(ROLES)})"
    return f"ROLE_FORBIDDEN {role} may not run {what}{suffix}"


def record_role_refusal(argv, role: str, message: str) -> None:
    """Best-effort: log a ``ROLE_FORBIDDEN`` refusal made before dispatch
    (``bin/coresmith``) into the ``actions`` table with rc 2 and the role as
    actor. Never raises; needs an existing project database."""
    try:
        argv = [str(a) for a in (argv or [])]
        root = None
        for i, tok in enumerate(argv):
            if tok == "--project-root" and i + 1 < len(argv):
                root = argv[i + 1]
            elif tok.startswith("--project-root="):
                root = tok.split("=", 1)[1]
        from orchestrator.harness.env import bootstrap_project_root
        root_p = bootstrap_project_root(root)
        from orchestrator.state_store.project_db import DB_NAME, ProjectDB
        if not (root_p / ".coresmith" / DB_NAME).is_file():
            return
        db = ProjectDB(root_p)
        db.ensure_schema()
        db.record_action(argv, EXIT_USAGE, summary=str(message)[:200], actor=role or "cli")
    except Exception:  # noqa: BLE001 - the audit log must never break the refusal
        pass


# ``bin/coresmith`` lifecycle verbs (served by the daemon over HTTP, not by
# this module) that still belong in the ``actions`` log.
LOGGED_DAEMON_VERBS = ("daemon", "run", "resume", "backend")


def record_cli_action(argv, rc, summary: str = "", *, create: bool = False) -> None:
    """Best-effort: log one ``bin/coresmith`` daemon-client verb (``daemon``,
    ``run``, ``resume``, ``backend``) into the ``actions`` table with its rc and
    the role / ``CORESMITH_ACTOR`` as actor. Needs the project's
    ``.coresmith/`` dir; the database is created only when ``create``. Never
    raises."""
    try:
        argv = [str(a) for a in (argv or [])]
        root = None
        for i, tok in enumerate(argv):
            if tok == "--project-root" and i + 1 < len(argv):
                root = argv[i + 1]
            elif tok.startswith("--project-root="):
                root = tok.split("=", 1)[1]
        from orchestrator.harness.env import bootstrap_project_root
        root_p = bootstrap_project_root(root)
        from orchestrator.state_store.project_db import DB_NAME, ProjectDB
        if not (root_p / ".coresmith" / DB_NAME).is_file() and not (create and (root_p / ".coresmith").is_dir()):
            return
        db = ProjectDB(root_p)
        db.ensure_schema()
        code = rc if isinstance(rc, int) else (0 if rc is None else EXIT_FAIL)
        db.record_action(argv, code, summary=str(summary or "")[:200], actor=_role() or _actor_env() or "cli")
    except Exception:  # noqa: BLE001 - the audit log must never break a verb
        pass


def _cli_argv(args):
    """The argv to role-check: tests pass ``args._argv``; the real CLI is
    ``sys.argv``; any other in-process call is not a CLI invocation."""
    a = getattr(args, "_argv", None)
    if a is not None:
        return [str(x) for x in a]
    from pathlib import Path
    if Path(sys.argv[0] if sys.argv else "").name == "coresmith":
        return [str(x) for x in sys.argv[1:]]
    return None


def _run(handler):
    """Wrap an int-returning handler so the CLI exits with its code (and the
    invocation lands in the project DB ``actions`` log). Refuses a verb the
    ``CORESMITH_ROLE`` may not run (``ROLE_FORBIDDEN``, exit 2)."""
    def _f(args):
        global _LAST_SUMMARY
        _LAST_SUMMARY = ""
        role = _role()
        argv = _cli_argv(args) if role else None
        if role and argv is not None and not role_allows(role, argv):
            msg = role_refusal(role, argv)
            print(msg, file=sys.stderr)
            _LAST_SUMMARY = msg
            _record_action(args, handler, EXIT_USAGE)
            raise SystemExit(EXIT_USAGE)
        try:
            rc = handler(args)
        except SystemExit as exc:
            _record_action(args, handler, exc.code if isinstance(exc.code, int) else EXIT_FAIL)
            raise
        except BaseException as exc:
            if not _LAST_SUMMARY:
                _LAST_SUMMARY = f"{type(exc).__name__}: {exc}"[:200]
            _record_action(args, handler, EXIT_FAIL)
            raise
        _record_action(args, handler, rc)
        raise SystemExit(rc)
    return _f


def _add_project_root(p) -> None:
    p.add_argument("--project-root", help="overrides $CORESMITH_PROJECT_ROOT")


def _add_json(p) -> None:
    p.add_argument("--json", action="store_true", help="machine-readable output")


def _state_db(args):
    from orchestrator.state_store.project_db import open_project
    return open_project(_bootstrap(args))



# ---------------------------------------------------------------------------
# The state machine: register / item / link / check / question / stage / status.
# The Architect (the coding agent calling the CLI) advances a run only through
# these; they are the old graph nodes as deterministic tools.

def _problems_lines(problems):
    return [f"  [{q.get('severity','error')[:4]}] {q.get('code')} {q.get('where')}: {q.get('text')}" for q in problems]


def cmd_register(args) -> int:
    """``coresmith register <kind> <path> [--block b]`` -- parse, validate, record an artifact."""
    from orchestrator.harness.tools.register import register
    db = _state_db(args)
    res = register(db, db.root, args.kind, args.path, block=getattr(args, "block", "") or "", actor="cli",
                   unlock=bool(getattr(args, "unlock", False)), reason=getattr(args, "reason", "") or "")
    lines = [f"register {args.kind}: {'OK' if res.get('ok') else 'REFUSED'}"
             + (f" [unlocked: {args.reason}]" if getattr(args, "unlock", False) and res.get("ok") else "")]
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
        lines += [f"  -> {lk['rel']} {lk['to_id']}" for lk in payload["links_out"]]
        lines += [f"  <- {lk['rel']} {lk['from_id']}" for lk in payload["links_in"]]
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


def _unknown_block_refs(db, *ids) -> list[str]:
    """``block:<name>`` references to blocks the registry does not have (only
    checked once blocks are registered)."""
    refs = [str(i) for i in ids if str(i).startswith("block:")]
    if not refs:
        return []
    try:
        names = {b["name"] for b in db.blocks()}
    except Exception:  # noqa: BLE001
        names = set()
    return [r for r in refs if names and r.split(":", 1)[1] not in names]


def cmd_link(args) -> int:
    db = _state_db(args)
    bad = _unknown_block_refs(db, args.from_id, args.to_id)
    if bad:
        prob = {"code": "LINK_UNKNOWN_BLOCK", "where": ", ".join(bad), "severity": "error",
                "text": "not a registered block (coresmith blocks)"}
        _emit(args, {"ok": False, "problems": [prob]}, "\n".join(["link: REFUSED"] + _problems_lines([prob])))
        return EXIT_USAGE
    try:
        db.link_items(args.from_id, args.to_id, args.rel, source="cli")
    except ValueError as exc:
        _emit(args, {"error": str(exc)}, f"error: {exc}")
        return EXIT_USAGE
    _emit(args, {"from": args.from_id, "to": args.to_id, "rel": args.rel}, f"{args.from_id} -{args.rel}-> {args.to_id}")
    return EXIT_PASS


def cmd_unlink(args) -> int:
    """``coresmith unlink <from> <to> <rel>`` -- delete one link (``<to>`` may end in ``*``)."""
    from orchestrator.state_store.ontology import LINK_RELS
    db = _state_db(args)
    if args.rel not in LINK_RELS:
        _emit(args, {"error": f"bad link rel {args.rel!r}"}, f"error: bad link rel {args.rel!r}: {LINK_RELS}")
        return EXIT_USAGE
    n = db.unlink_items(args.from_id, args.to_id, args.rel)
    _emit(args, {"from": args.from_id, "to": args.to_id, "rel": args.rel, "removed": n},
          f"{args.from_id} -{args.rel}-> {args.to_id}: {n} link(s) removed")
    return EXIT_PASS if n else EXIT_USAGE


def cmd_check(args) -> int:
    db = _state_db(args)
    verb = getattr(args, "verb", "list")
    if verb == "add":
        value = getattr(args, "value", None)
        if not args.status and value is None:
            _emit(args, {"error": "give a status or --value"}, "error: give a status (pass|fail|...) or --value N")
            return EXIT_USAGE
        from orchestrator.state_store.ontology import CheckStatusConflict
        try:
            cid = db.add_check(args.item, args.kind, args.status or None, evidence=args.evidence or "", sha=args.sha or "",
                               run_id=args.run_id or "", actor=args.actor or _role() or "cli", value=value)
        except CheckStatusConflict as exc:
            prob = {"code": "CHECK_STATUS_CONFLICT", "where": args.item, "severity": "error",
                    "text": f"value {exc.value:g} derives {exc.derived!r} from the item's bounds; the explicit "
                            f"status {exc.given!r} contradicts it (omit the status: the bounds decide)"}
            _emit(args, {"ok": False, "problems": [prob]}, "\n".join(["check add: REFUSED"] + _problems_lines([prob])))
            return EXIT_USAGE
        except ValueError as exc:
            _emit(args, {"error": str(exc)}, f"error: {exc}")
            return EXIT_USAGE
        row = next((c for c in reversed(db.checks(args.item, kind=args.kind)) if c["id"] == cid), {})
        it = db.item(args.item) or {}
        payload = {"id": cid, "status": row.get("status"), "value": value,
                   "bounds": {"metric": it.get("metric"), "min": it.get("bound_min"), "max": it.get("bound_max"),
                              "unit": it.get("unit")}}
        human = f"check #{cid}: {args.item} {args.kind} {row.get('status')}"
        if value is not None:
            lo = "-" if it.get("bound_min") is None else f"{it['bound_min']:g}"
            hi = "-" if it.get("bound_max") is None else f"{it['bound_max']:g}"
            human += f" (value {value:g} {it.get('unit') or ''}; {it.get('metric') or 'bounds'} min {lo} max {hi})"
        _emit(args, payload, human)
        return EXIT_PASS
    rows = db.checks(getattr(args, "item", None) or None, kind=getattr(args, "kind", None) or None,
                     latest=bool(getattr(args, "latest", False)))
    lines = [f"{c['item_id']:<18} {c['kind']:<14} {c['status']:<13} {c['sha'] or '-':<10} {(c['evidence'] or '')[:80]}" for c in rows] or ["(no checks)"]
    _emit(args, {"checks": rows}, "\n".join(lines))
    return EXIT_PASS


def _num_id(s, prefix: str) -> int | None:
    """``Q3`` / ``q3`` / ``3`` -> 3 (``prefix`` is the letter the listings print)."""
    t = str(s or "").strip()
    if t[:1].upper() == prefix.upper():
        t = t[1:]
    return int(t) if t.isdigit() else None


def _bad_id(args, what: str, raw) -> int:
    _emit(args, {"error": f"bad {what} id {raw!r}"}, f"error: bad {what} id {raw!r} (e.g. {what[0].upper()}3 or 3)")
    return EXIT_USAGE


def cmd_question(args) -> int:
    db = _state_db(args)
    verb = getattr(args, "verb", "list")
    if verb == "add":
        qid = db.add_question(args.text, item_id=args.item or "", must_answer=not args.optional,
                              asked_by=args.by or _actor_env() or _role() or "architect")
        _emit(args, {"id": qid}, f"question Q{qid} recorded ({'must answer' if not args.optional else 'optional'})")
        return EXIT_PASS
    if verb == "show":
        qid = _num_id(args.id, "Q")
        if qid is None:
            return _bad_id(args, "question", args.id)
        q = db.question(qid)
        if not q:
            _emit(args, {"error": "no such question"}, f"no question Q{qid}")
            return EXIT_USAGE
        _emit(args, {"question": q}, f"Q{q['id']} {q['status']} {'MUST' if q['must_answer'] else 'opt'} "
                                     f"item={q['item_id'] or '-'} by={q['asked_by'] or '-'}\n  {q['text']}"
              + (f"\n  answer: {q['answer']}" if q.get("answer") else "")
              + (f" (ruling R{q['ruling_id']})" if q.get("ruling_id") else ""))
        return EXIT_PASS
    if verb == "answer":
        qid = _num_id(args.id, "Q")
        if qid is None:
            return _bad_id(args, "question", args.id)
        # the question row first: a ruling is written only for a question that
        # was actually answered (no orphan rulings on a bad id / closed question)
        ok = db.answer_question(qid, args.ruling or args.answer or "")
        rid = None
        if ok and args.ruling:
            rid = db.add_ruling("global", args.ruling, rationale=f"answers Q{qid}", source=args.by or "human",
                                question_ref=f"question:{qid}")
            db.set_question_ruling(qid, rid)
            db.export_rulings_view()
        _emit(args, {"id": qid, "answered": ok, "ruling_id": rid},
              f"Q{qid} {'answered' if ok else 'not open'}" + (f" by ruling R{rid}" if rid else ""))
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
        lines += _advisory_lines(res)
        _emit(args, res, "\n".join(lines))
        return EXIT_PASS if res["advanced"] else EXIT_FAIL
    res = st.status(db, db.root)
    lines = [f"stage {res['stage']} ({res['index'] + 1}/{len(res['stages'])}); done: {', '.join(res['done']) or '-'}",
             "can advance" if res["can_advance"] else "blocked by:"]
    for b in res["blocked_by"]:
        lines.append(f"  {b['code']}: {b['text']}" + (f" [{b['count']}] {', '.join(b['ids'][:12])}" if b['ids'] else ""))
    lines += _advisory_lines(res)
    _emit(args, res, "\n".join(lines))
    return EXIT_PASS


def _advisory_lines(res: dict) -> list[str]:
    """Advisories (e.g. ``MODEL_ONLY_FAILED``) printed after the blockers; they never affect ``can_advance``."""
    return [f"  advisory {a['code']}: {a['text']}" + (f" [{a['count']}] {', '.join(a['ids'][:12])}" if a.get("ids") else "")
            for a in res.get("advisories") or []]


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
    lines += _advisory_lines(s)
    _emit(args, payload, "\n".join(lines))
    return EXIT_PASS


@contextlib.contextmanager
def _engine_logs_to_stderr():
    """Engine progress lines (the graph helpers print them to stdout) go to
    stderr for the duration of a tool call, so a ``--json`` stdout stays one
    machine-readable document."""
    with contextlib.redirect_stdout(sys.stderr):
        yield


def _model_exit(res: dict) -> int:
    """The model/harness exit contract: 0 pass; 1 build/run/check failure;
    2 missing or invalid input (``required`` names the path); 3 provider or
    toolchain failure (``tool_error`` / ``provider_error``)."""
    from orchestrator.harness.tools.integrate import INPUT_ERRORS
    if res.get("error") in INPUT_ERRORS:
        return EXIT_USAGE
    if res.get("tool_error") or res.get("provider_error") or res.get("provider_errors"):
        return EXIT_INFRA
    return EXIT_PASS if res.get("ok") else EXIT_FAIL


def _eval_lines(title: str, res: dict) -> list[str]:
    ok = res.get("gate_ok")
    head = f"{title}: " + ("PASS" if ok else ("NOT PASSING" if ok is False else "NOT EVALUATED"))
    if res.get("error"):
        head += f" ({res['error']})"
    lines = [head]
    if res.get("required"):
        lines.append(f"  required: {res['required']}")
    if res.get("hint"):
        lines.append(f"  {res['hint']}")
    s = res.get("summary") or {}
    if s:
        lines.append(f"  {s['counts']} failed={s['failed'][:10]} unanswered_must={s['unanswered_must'][:10]}"
                     + (f" no_value={s['no_value'][:10]} (bounded items need a measured value)" if s.get("no_value") else ""))
    if res.get("report"):
        lines.append(f"  report: {res['report']}")
    for k in ("build_log", "run_log"):
        if res.get(k):
            lines.append(res[k][-800:])
    return lines


def _read_diagnostics(args) -> str | None:
    """The ``--diagnostics <file>`` text for an authoring verb ('' when not
    given; None after printing the error when the file cannot be read)."""
    path = getattr(args, "diagnostics", "") or ""
    if not path:
        return ""
    from pathlib import Path
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")[-12000:]
    except OSError as exc:
        _emit(args, {"ok": False, "error": "DIAGNOSTICS_UNREADABLE", "required": path},
              f"error: --diagnostics {path}: {exc}")
        return None


def cmd_model(args) -> int:
    """``coresmith model init|build|run|eval|register [--arch]`` and ``coresmith
    model author --block <b>``: executable models as tools. ``build``, ``run``
    and ``eval`` are checks of the files that exist and never construct an
    authoring agent; ``author`` is the explicit authoring verb (one call per
    NAMED block). Exit: 0 pass; 1 build/run/check failure; 2 missing or
    invalid input (``required`` names the path); 3 provider/toolchain failure."""
    from orchestrator.harness.tools import integrate as it
    from orchestrator.harness.tools import model as mt
    db = _state_db(args)
    verb = getattr(args, "verb", "build")
    if verb == "refine":
        res = {"ok": False, "error": "MODEL_REFINE_REMOVED",
               "hint": ("`model refine` (implicit authoring of every missing model behind a check) was removed. "
                        "Use `coresmith model build` (assemble + compile + smoke what exists; lists missing models), "
                        "`coresmith model author --block <b>` (explicit authoring of the named blocks), "
                        "`coresmith model eval` (pure FRD check of the supplied harness) and "
                        "`coresmith harness author` (explicit harness authoring).")}
        _emit(args, res, f"error: {res['error']}: {res['hint']}")
        return EXIT_USAGE
    if verb == "author":
        blocks = [b for b in (getattr(args, "block", None) or []) if b]
        if not blocks:
            _emit(args, {"ok": False, "error": "NO_BLOCKS", "required": "--block <name> (repeatable)"},
                  "error: model author needs --block <name> (repeatable); only the named blocks are authored")
            return EXIT_USAGE
        diag = _read_diagnostics(args)
        if diag is None:
            return EXIT_USAGE
        with _engine_logs_to_stderr():
            res = it.model_author(db, db.root, blocks, diagnostics=diag)
        lines = [f"model author: {'OK' if res.get('ok') else 'FAILED'}" + (f" ({res['error']})" if res.get("error") else "")]
        for b, r in (res.get("blocks") or {}).items():
            lines.append(f"  {b}: {'written' if r.get('written') else 'NOT written'}"
                         + (f" -- provider error: {r['provider_error'][:200]}" if r.get("provider_error") else "")
                         + (f" -- {r['notes'][:160]}" if r.get("notes") else ""))
        for k in ("unknown", "primitive", "hint", "required"):
            if res.get(k):
                lines.append(f"  {k}: {res[k]}")
        _emit(args, res, "\n".join(lines))
        return _model_exit(res)
    if not getattr(args, "arch", False):
        if verb == "build":
            with _engine_logs_to_stderr():
                res = it.model_build(db, db.root)
            lines = [f"model build: {'OK' if res.get('ok') else 'NOT PASSING'} build={res.get('build_ok')} "
                     f"smoke={res.get('smoke_ok')}"
                     + (f" missing={res['missing_models']}" if res.get("missing_models") else "")
                     + (f" ({res['parked_reason']})" if res.get("parked_reason") else "")
                     + (f" ({res['error']})" if res.get("error") else "")]
            for k in ("required", "hint"):
                if res.get(k):
                    lines.append(f"  {k}: {res[k]}")
            if not res.get("build_ok") and res.get("build_log"):
                lines.append(res["build_log"][-800:])
            _emit(args, res, "\n".join(lines))
            if res.get("missing_models") or res.get("error") == "MODEL_PATH_UNSUPPORTED":
                return EXIT_USAGE          # missing/unsupported input: the implementations named in `required`
            return _model_exit(res)
        if verb == "eval":
            with _engine_logs_to_stderr():
                res = it.model_eval(db, db.root)
            _emit(args, res, "\n".join(_eval_lines("FRD evaluation on the SoC model", res)))
            return _model_exit(res)
        _emit(args, {"error": f"model {verb} needs --arch"},
              f"error: model {verb} needs --arch (build/eval without --arch work on the SoC model; author needs --block)")
        return EXIT_USAGE
    if verb == "init":
        res = mt.arch_init(db.root)
        _emit(args, res, f"{res['path']} {'created' if res.get('created') else 'exists'}" + (f"\n  {res['hint']}" if res.get("hint") else ""))
        return EXIT_PASS
    if verb == "build":
        res = mt.arch_build(db.root)
        lines = [f"arch model build: {'OK' if res['ok'] else 'FAILED'}" + (f" ({res['error']})" if res.get("error") else "")] \
            + ([f"  required: {res['required']}"] if res.get("required") else []) \
            + ([f"  {res['hint']}"] if res.get("hint") else []) \
            + [f"  {q}" for q in res.get("problems") or []] + ([res["log"][-1500:]] if not res["ok"] and res.get("log") else [])
        _emit(args, res, "\n".join(lines))
        return _model_exit(res)
    if verb == "run":
        res = mt.arch_run(db.root, ns=int(getattr(args, "ns", 10000) or 10000))
        st = res.get("stats") or {}
        lines = [f"arch model run: {'OK' if res['ok'] else 'FAILED'}" + (f" ({res['error']})" if res.get("error") else "")]
        for k in ("required", "hint"):
            if res.get(k):
                lines.append(f"  {k}: {res[k]}")
        if st:
            lines.append(f"  total_cycles={st.get('total_cycles')} dynamic_mw={st.get('dynamic_mw', 0):.3f} decerr={st.get('decerr')}")
            for lk in st.get("links") or []:
                lines.append(f"  link {lk['from']}->{lk['to']}: {lk['bytes']} B, {lk['bytes_per_cycle']:.3f} B/cyc, util {lk['utilization']:.2f}, max_out {lk['max_outstanding']}")
        elif res.get("log"):
            lines.append(res["log"][-800:])
        _emit(args, res, "\n".join(lines))
        return _model_exit(res)
    if verb == "eval":
        with _engine_logs_to_stderr():
            res = mt.arch_eval(db, db.root, timeout_s=getattr(args, "timeout", None))
        _emit(args, res, "\n".join(_eval_lines("arch FRD evaluation", res)))
        return _model_exit(res)
    if verb == "register":
        res = mt.arch_register(db, db.root)
        _emit(args, res, f"arch_model registered v{res['artifact']['version']}" if res.get("ok")
              else f"error: {res.get('error')}" + (f" required: {res['required']}" if res.get("required") else ""))
        return _model_exit(res)
    return EXIT_USAGE


def cmd_harness(args) -> int:
    """``coresmith harness author [--arch] [--diagnostics <log>]``: one explicit
    FRD-harness author call for the SoC model (or the executable architecture
    model). Same exit contract as ``model``."""
    from orchestrator.harness.tools import integrate as it
    db = _state_db(args)
    if getattr(args, "verb", "author") != "author":
        return EXIT_USAGE
    diag = _read_diagnostics(args)
    if diag is None:
        return EXIT_USAGE
    arch = bool(getattr(args, "arch", False))
    with _engine_logs_to_stderr():
        res = it.harness_author(db, db.root, arch=arch, diagnostics=diag)
    lines = [f"harness author{' --arch' if arch else ''}: {'OK' if res.get('ok') else 'FAILED'}"
             + (f" ({res['error']})" if res.get("error") else "")]
    for k in ("required", "hint"):
        if res.get(k):
            lines.append(f"  {k}: {res[k]}")
    if res.get("provider_error"):
        lines.append(f"  provider error: {res['provider_error'][:300]}")
    if res.get("sources"):
        lines.append(f"  sources: {res['sources']}")
    _emit(args, res, "\n".join(lines))
    return _model_exit(res)


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
    if res.get("item_checks") or res.get("unverified_items") or res.get("deferred_items"):
        lines.append("  items verified: " + (", ".join(f"{c['item']}[{c['entry']}={c['status']}"
                                                     + (f" value={c['value']:g}" if c.get("value") is not None else "") + "]"
                                                     for c in res.get("item_checks") or []) or "-"))
        if res.get("unverified_items"):
            lines.append(f"  items with NO verifier (no check recorded): {', '.join(res['unverified_items'])} "
                         "-- coresmith frd verifier <id> --kind cocotb --path <tb> --entry <test>")
        if res.get("deferred_items"):
            lines.append(f"  items verified elsewhere (not by this block's cocotb run): {', '.join(res['deferred_items'])}")
        if res.get("unmeasured_items"):
            lines.append(f"  bounded items with NO measurement (tool_error): {', '.join(res['unmeasured_items'])} "
                         "-- the TB must call orchestrator.harness.measure.record(<item>, <value>, unit=, test=)")
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


def _interrupt_id(db, raw: str) -> str:
    """The interrupt id ``raw`` names: exact, a leading ``Q``/``R`` + digits
    stripped (the form other listings print), or a unique prefix of a pending id."""
    t = str(raw or "").strip()
    if t[:1] in ("Q", "R", "q", "r") and t[1:].isdigit():
        t = t[1:]
    try:
        rows = db.interrupts(status="pending")
    except Exception:  # noqa: BLE001
        return t
    ids = [r["id"] for r in rows]
    if t in ids:
        return t
    hits = [i for i in ids if i.startswith(t) or i == f"int-{t}" or i.startswith(f"int-{t}")]
    return hits[0] if len(hits) == 1 else t


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
        args.resolve = _interrupt_id(db, args.resolve)
        resolved = db.resolve_interrupt(args.resolve, resolution, resolved_by=_role() or "cli")
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
        rid = _num_id(args.id, "R")
        if rid is None:
            return _bad_id(args, "ruling", args.id)
        ok = db.revoke_ruling(rid, args.reason or "")
        db.export_rulings_view()
        _emit(args, {"revoked": ok, "id": rid},
              f"R{rid} {'revoked' if ok else 'not found or already revoked'}")
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
    rr.add_argument("id", help="R3 or 3")
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

    # --- Architect tools -------------------------------------------
    rg = sub.add_parser("register", help="register an artifact (prd|sad|frd|ers|block_diagram|contracts|abi|uarch|arch_model|harness)")
    rg.add_argument("kind")
    rg.add_argument("path")
    rg.add_argument("--block", default="")
    rg.add_argument("--unlock", action="store_true", help="contracts: allow changing locked edges (needs --reason)")
    rg.add_argument("--reason", default="", help="why locked edges change (recorded in the actions log)")
    _add_project_root(rg)
    _add_json(rg)
    rg.set_defaults(func=_run(cmd_register))
    ip = sub.add_parser("item", help="ontology items: list | show | status")
    isub = ip.add_subparsers(dest="verb")
    il = isub.add_parser("list")
    il.add_argument("--kind")
    il.add_argument("--artifact")
    il.add_argument("--status")
    il.add_argument("--must", action="store_true")
    _add_project_root(il)
    _add_json(il)
    il.set_defaults(func=_run(cmd_item), verb="list")
    ish = isub.add_parser("show")
    ish.add_argument("id")
    _add_project_root(ish)
    _add_json(ish)
    ish.set_defaults(func=_run(cmd_item), verb="show")
    ist = isub.add_parser("status")
    ist.add_argument("id")
    ist.add_argument("status")
    _add_project_root(ist)
    _add_json(ist)
    ist.set_defaults(func=_run(cmd_item), verb="status")
    lk = sub.add_parser("link", help="link two items: <from> <to> <rel> (derives_from|owned_by|verified_by|cites|covers)")
    lk.add_argument("from_id")
    lk.add_argument("to_id")
    lk.add_argument("rel")
    _add_project_root(lk)
    _add_json(lk)
    lk.set_defaults(func=_run(cmd_link))
    ul = sub.add_parser("unlink", help="remove a link: <from> <to> <rel> (<to> may end in * to match a prefix)")
    ul.add_argument("from_id")
    ul.add_argument("to_id")
    ul.add_argument("rel")
    _add_project_root(ul)
    _add_json(ul)
    ul.set_defaults(func=_run(cmd_unlink))
    ck = sub.add_parser("check", help="tool verdicts about items: add | list")
    csub = ck.add_subparsers(dest="verb")
    ca = csub.add_parser("add")
    ca.add_argument("item")
    ca.add_argument("kind")
    ca.add_argument("status", nargs="?", default=None, help="pass|fail|not_testable|skipped|tool_error "
                    "(optional with --value: derived from the item's bounds)")
    ca.add_argument("--value", type=float, default=None, help="measured value; status derives from the item's bounds")
    ca.add_argument("--evidence", default="")
    ca.add_argument("--sha", default="")
    ca.add_argument("--run-id", dest="run_id", default="")
    ca.add_argument("--actor", default="")
    _add_project_root(ca)
    _add_json(ca)
    ca.set_defaults(func=_run(cmd_check), verb="add")
    cl = csub.add_parser("list")
    cl.add_argument("--item")
    cl.add_argument("--kind")
    cl.add_argument("--latest", action="store_true")
    _add_project_root(cl)
    _add_json(cl)
    cl.set_defaults(func=_run(cmd_check), verb="list")
    qp = sub.add_parser("question", help="item-scoped questions: add | answer | list")
    qsub = qp.add_subparsers(dest="verb")
    qa = qsub.add_parser("add")
    qa.add_argument("text")
    qa.add_argument("--item", default="")
    qa.add_argument("--optional", action="store_true")
    qa.add_argument("--by", default="")
    _add_project_root(qa)
    _add_json(qa)
    qa.set_defaults(func=_run(cmd_question), verb="add")
    qn = qsub.add_parser("answer")
    qn.add_argument("id", help="Q3 or 3")
    qn.add_argument("--ruling", default="")
    qn.add_argument("--answer", default="")
    qn.add_argument("--by", default="")
    _add_project_root(qn)
    _add_json(qn)
    qn.set_defaults(func=_run(cmd_question), verb="answer")
    qs = qsub.add_parser("show")
    qs.add_argument("id", help="Q3 or 3")
    _add_project_root(qs)
    _add_json(qs)
    qs.set_defaults(func=_run(cmd_question), verb="show")
    ql = qsub.add_parser("list")
    ql.add_argument("--all", action="store_true")
    _add_project_root(ql)
    _add_json(ql)
    ql.set_defaults(func=_run(cmd_question), verb="list")
    sp = sub.add_parser("stage", help="the run's state machine: status | next")
    ssub = sp.add_subparsers(dest="verb")
    ss = ssub.add_parser("status")
    _add_project_root(ss)
    _add_json(ss)
    ss.set_defaults(func=_run(cmd_stage), verb="status")
    sn = ssub.add_parser("next")
    _add_project_root(sn)
    _add_json(sn)
    sn.set_defaults(func=_run(cmd_stage), verb="next")
    stp = sub.add_parser("status", help="one-screen run status (stage, artifacts, items, questions, blockers)")
    _add_project_root(stp)
    _add_json(stp)
    stp.set_defaults(func=_run(cmd_status))
    mp = sub.add_parser("model", help="executable reference models: init | build | run | eval | register "
                                      "[--arch]; author --block <b> (explicit authoring). Checks never author.")
    msub = mp.add_subparsers(dest="verb")
    for v, h in (("init", "write a template model/arch/arch_model.json (--arch)"),
                 ("build", "--arch: generate + compile the architecture model; without: assemble + compile + smoke "
                           "the SoC model from the implementations that exist (missing ones are listed, never authored)"),
                 ("run", "run the architecture model's smoke scenario; writes stats.json (--arch)"),
                 ("eval", "FRD evaluation with the SUPPLIED harness (--arch or the SoC model): a pure check; "
                          "a missing harness exits 2 with the required path, never authors or repairs"),
                 ("register", "register the arch_model artifact (--arch)"),
                 ("author", "explicit SystemC model authoring for the NAMED blocks only: --block <b> (repeatable) "
                            "[--diagnostics <build log>]; exit 3 when the provider cannot run. No --arch: the "
                            "architecture model is generated from arch_model.json, never authored"),
                 ("refine", "removed (exit 2): use model build / model author / model eval / harness author")):
        mv = msub.add_parser(v, help=h)
        if v != "author":
            # ``model author`` has no --arch form: argparse rejects it (exit 2)
            # instead of accepting a flag that would silently do nothing.
            mv.add_argument("--arch", action="store_true")
        if v == "author":
            mv.add_argument("--block", action="append", default=[], help="block to author (repeatable)")
            mv.add_argument("--diagnostics", default="", help="path to a build log: makes this a repair request")
        if v == "refine":
            mv.add_argument("--block", default="")
            mv.add_argument("--all", action="store_true")
        if v == "run":
            mv.add_argument("--ns", type=int, default=10000)
        if v == "eval":
            mv.add_argument("--timeout", type=int)
        _add_project_root(mv)
        _add_json(mv)
        mv.set_defaults(func=_run(cmd_model), verb=v)
    hp = sub.add_parser("harness", help="the FRD evaluation harness: author [--arch] (explicit authoring)")
    hsub = hp.add_subparsers(dest="verb")
    ha = hsub.add_parser("author", help="one explicit harness-author call for the SoC model (or --arch: the "
                                        "architecture model) [--diagnostics <build log>]; exit 3 when the provider cannot run")
    ha.add_argument("--arch", action="store_true")
    ha.add_argument("--diagnostics", default="", help="path to a build/run log: makes this a repair request")
    _add_project_root(ha)
    _add_json(ha)
    ha.set_defaults(func=_run(cmd_harness), verb="author")
    bs = sub.add_parser("block-status", help="one block: paths, edges, VIPs, owned items, published pass")
    bs.add_argument("block")
    _add_project_root(bs)
    _add_json(bs)
    bs.set_defaults(func=_run(cmd_block_status))
    bd = sub.add_parser("block-done", help="the block gate: conformance -> DV -> synth -> timing; publishes best on a pass")
    bd.add_argument("block")
    bd.add_argument("--target-clock-mhz", dest="target_clock_mhz", type=float, default=None)
    bd.add_argument("--seed", type=int)
    bd.add_argument("--actor", default="")
    _add_project_root(bd)
    _add_json(bd)
    bd.set_defaults(func=_run(cmd_block_done))
    sc = sub.add_parser("schema", help="the document shape `register <kind>` expects (prd|sad|frd|ers|block_diagram|contracts|abi|uarch|arch_model)")
    sc.add_argument("kind", nargs="?", default="")
    _add_json(sc)
    sc.set_defaults(func=_run(cmd_schema))
    vp = sub.add_parser("vip", help="interface VIPs: generate (from the registered contracts)")
    vsub = vp.add_subparsers(dest="verb")
    vg = vsub.add_parser("generate")
    _add_project_root(vg)
    _add_json(vg)
    vg.set_defaults(func=_run(cmd_vip), verb="generate")
    shp = sub.add_parser("shell", help="chip shell: assemble (real RTL for published blocks, stubs for the rest; elaborate)")
    shsub = shp.add_subparsers(dest="verb")
    sha = shsub.add_parser("assemble")
    sha.add_argument("--tier")
    sha.add_argument("--all-real", dest="all_real", action="store_true")
    _add_project_root(sha)
    _add_json(sha)
    sha.set_defaults(func=_run(cmd_shell), verb="assemble")
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


def cmd_actions(args) -> int:
    """``coresmith actions [--limit N | --all] [--since ID] [--script]`` -- the CLI audit log."""
    db = _state_db(args)
    since = getattr(args, "since", None)
    limit = None if getattr(args, "all", False) else int(getattr(args, "limit", 200) or 200)
    rows = db.actions(limit=limit, since_id=since)
    total = db.actions_count(since_id=since)
    omitted = max(0, total - len(rows))
    if getattr(args, "script", False):
        import shlex
        # refused verbs (rc != 0) wrote nothing; they stay in the script, marked, and no `set -e`
        body = [shlex.join(["coresmith", *r["argv"]]) + (f"  # rc={r['rc']}" if r["rc"] not in (0, None) else "")
                for r in rows if (r["argv"] or [""])[0] != "actions"]
        print("\n".join(["#!/bin/sh", "# replay of the coresmith actions log", *body]))
        return EXIT_PASS
    lines = [f"#{r['id']} {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(r['ts']))} {r['actor'] or '-'} "
             f"rc={r['rc'] if r['rc'] is not None else '-'} {' '.join(r['argv'])}" for r in rows]
    if omitted:
        lines.append(f"({omitted} earlier row(s) omitted by --limit {limit}; --all prints every row)"
                     if since is None else f"({omitted} later row(s) omitted by --limit {limit}; --all prints every row)")
    _emit(args, {"actions": rows, "total": total, "limit": limit, "omitted": omitted},
          "\n".join(lines) or "no recorded actions")
    return EXIT_PASS


def _register_actions(sub) -> None:
    ac = sub.add_parser("actions", help="audit log of CLI verbs run against this project")
    _add_project_root(ac)
    _add_json(ac)
    ac.add_argument("--limit", type=int, default=200, help="at most N rows (the most recent; default 200)")
    ac.add_argument("--all", action="store_true", help="every row (no limit)")
    ac.add_argument("--since", type=int, default=None, metavar="ID", help="only rows after action ID")
    ac.add_argument("--script", action="store_true",
                    help="print the log as a replayable shell script (one `coresmith <argv>` per line)")
    ac.set_defaults(func=_run(cmd_actions))


def cmd_state_write(args) -> int:
    from orchestrator.harness.cli_frd import cmd_state_write as _w
    return _w(args)


def cmd_state_check(args) -> int:
    from orchestrator.harness.cli_state import cmd_state_check as _c
    return _c(args)


def _register_state_write(sub) -> None:
    """``coresmith state write [--force]``. ``bin/coresmith`` already owns a
    ``state`` verb (the daemon's run state); augment that parser instead of
    registering a second one, and dispatch on the optional ``write``."""
    sp = sub.choices.get("state") if hasattr(sub, "choices") else None
    orig = getattr(sp, "_defaults", {}).get("func") if sp is not None else None
    if sp is None:
        sp = sub.add_parser("state", help="coresmith state write: render the DB-sourced views")
        _add_project_root(sp)
    sp.add_argument("verb", nargs="?", choices=["write", "check"], default=None,
                    help="write: export registry views + render DB-sourced PRD/FRD documents; "
                         "check: audit the database rows against the files on disk")
    sp.add_argument("--force", action="store_true", help="state write: also overwrite file-sourced documents")
    if not any("--json" in a.option_strings for a in sp._actions):
        _add_json(sp)
    writer = _run(cmd_state_write)
    checker = _run(cmd_state_check)

    def _dispatch(args):
        if getattr(args, "verb", None) == "write":
            return writer(args)
        if getattr(args, "verb", None) == "check":
            return checker(args)
        if orig is not None:
            return orig(args)
        print("usage: coresmith state write [--force] | coresmith state check", file=sys.stderr)
        raise SystemExit(EXIT_USAGE)
    sp.set_defaults(func=_dispatch)


def _register_line_item_verbs(sub) -> None:
    """FRD / contract / fabric / pin line-item verbs (their own modules, deferred)."""
    import importlib
    for mod in ("cli_frd", "cli_contract", "cli_fabric", "cli_pins"):
        try:
            m = importlib.import_module(f"orchestrator.harness.{mod}")
            m.register(sub, _run, _add_project_root, _add_json)
        except Exception as exc:  # noqa: BLE001 - one broken module must not drop every harness verb
            print(f"coresmith: {mod} verbs unavailable: {exc}", file=sys.stderr)


def register_subcommands(sub) -> None:
    """Register harness subcommands on the ``bin/coresmith`` subparser action."""
    _register_verify(sub)
    _register_state(sub)
    _register_queries(sub)
    _register_tool(sub)
    _register_line_item_verbs(sub)
    try:
        _register_state_write(sub)
    except Exception as exc:  # noqa: BLE001 - never drop the other harness verbs
        print(f"coresmith: state write unavailable: {exc}", file=sys.stderr)
    _register_actions(sub)


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
    from orchestrator.harness.targets import register_cli
    register_cli(sub, _run, _add_project_root, _add_json)
