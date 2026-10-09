# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""``coresmith prd ...`` / ``coresmith frd ...`` / ``coresmith state write``:
requirement line items authored straight into the project database.

Every write verb flips its artifact to DB-sourced (``db:prd`` / ``db:frd``)
and immediately re-renders the on-disk view (``arch/frd_spec.md``;
``arch/prd_spec.md`` + ``.coresmith/prd_spec.json``) so the tools that read
files (``frd_eval``, ``model eval``) always see the database's content.

Wired from ``orchestrator.harness.cli.register_subcommands`` via
``register``. Like ``cli.py``, this module must import without importing
``orchestrator.langgraph`` -- every heavy import is deferred into handlers.
"""

from __future__ import annotations

import json
from pathlib import Path

EXIT_PASS, EXIT_FAIL, EXIT_USAGE = 0, 1, 2

BOUNDED_KINDS = ("PERF", "TIME")
PRD_KINDS = ("FR", "KPI", "CON")


def _cli():
    from orchestrator.harness import cli
    return cli


def _db(args):
    return _cli()._state_db(args)


def _problem(code: str, where: str, text: str, severity: str = "error") -> dict:
    return {"code": code, "where": where, "text": text, "severity": severity}


def _refuse(args, verb: str, problems: list[dict], rc: int = EXIT_USAGE) -> int:
    cli = _cli()
    cli._emit(args, {"ok": False, "problems": problems},
              "\n".join([f"{verb}: REFUSED"] + cli._problems_lines(problems)))
    return rc


def _split_ids(s: str | None) -> list[str]:
    return [t.strip() for t in (s or "").split(",") if t.strip()]


_UNSET = "unset"


def _bound(v):
    """``--min``/``--max`` value: a float, None for ``none``/``-`` (clear), or
    ``_UNSET`` when the option was not given."""
    if v is None:
        return _UNSET
    if str(v).strip().lower() in ("none", "-", ""):
        return None
    return float(str(v).replace(",", ""))


def _flip_and_render(db, kind: str) -> list[str]:
    """Mark ``kind`` DB-sourced and re-render its on-disk view."""
    from orchestrator.harness.tools import render
    db.ensure_db_artifact(kind, registered_by="cli")
    db.register_artifact(kind, f"db:{kind}", sha="db", registered_by="cli")
    paths = [render.write_frd(db, db.root)] if kind == "frd" else render.write_prd(db, db.root)
    return [str(p) for p in paths]


def _rel(db, p: str) -> str:
    try:
        return str(Path(p).relative_to(db.root))
    except ValueError:
        return p


def _wrote(db, paths) -> str:
    return "wrote " + ", ".join(_rel(db, p) for p in paths)


def advisories(db, item: dict) -> list[dict]:
    """What the stage machine will still demand of one FRD item (never a refusal)."""
    from orchestrator.state_store.ontology import item_must_have
    iid, out = item["id"], []
    if not item.get("acceptance"):
        out.append(_problem("FRD_NO_ACCEPTANCE", iid, "no acceptance criteria (--acceptance)", "warning"))
    if not item.get("priority"):
        out.append(_problem("FRD_NO_PRIORITY", iid, "no priority (--priority)", "warning"))
    if item_must_have(item):
        if item.get("kind") not in ("PHYS", "MPW") and not db.links(from_id=iid, rel="owned_by"):
            out.append(_problem("REQ_UNOWNED", iid, "must-have no block owns (--owner <block>)", "warning"))
        if not db.verifiers(item_id=iid):
            out.append(_problem("FRD_UNVERIFIED", iid, "no verifier (coresmith frd verifier <id> --kind ...)", "warning"))
    return out


# ---------------------------------------------------------------------------
# prd

def cmd_prd(args) -> int:
    from orchestrator.state_store.ontology import is_item_id
    cli = _cli()
    db = _db(args)
    verb = getattr(args, "verb", None) or "list"
    if verb == "list":
        rows = db.items(artifact="prd", kind=(getattr(args, "kind", None) or "").upper() or None)
        lines = [f"{r['id']:<16} {r['kind']:<4} {r['status']:<10} {r['priority'] or '-':<12} {r['text'][:80]}"
                 for r in rows] or ["(no PRD items)"]
        cli._emit(args, {"items": rows}, "\n".join(lines))
        return EXIT_PASS
    iid = (args.id or "").strip()
    if verb == "retire":
        if not db.retire_item(iid):
            return _refuse(args, "prd retire", [_problem("NO_ITEM", iid, "no such item")])
        paths = _flip_and_render(db, "prd")
        cli._emit(args, {"id": iid, "retired": True, "written": paths}, f"{iid} retired; {_wrote(db, paths)}")
        return EXIT_PASS
    if not is_item_id(iid):
        return _refuse(args, f"prd {verb}", [_problem("PRD_BAD_ID", iid,
                                                      "id must look like FR-<AREA>-<n> / KPI-<AREA>-<n> / CON-<n>")])
    if verb == "add":
        prev = db.item(iid)
        if prev and prev.get("status") != "retired":
            return _refuse(args, "prd add", [_problem("PRD_EXISTS", iid, f"already exists; use coresmith prd edit {iid}")])
        kind = (args.kind or iid.split("-")[0]).upper()
        if kind not in PRD_KINDS:
            return _refuse(args, "prd add", [_problem("PRD_BAD_KIND", iid, f"kind {kind!r} not one of {PRD_KINDS} (--kind)")])
        from orchestrator.harness.tools.render import PRD_SECTIONS
        prio = args.priority or ("must_have" if kind in ("KPI", "CON") else "")
        item = db.upsert_item("prd", {"id": iid, "kind": kind, "section": PRD_SECTIONS[kind], "text": args.text,
                                      "priority": prio, "acceptance": args.acceptance or "", "status": "open"})
    else:  # edit
        if not db.item(iid):
            return _refuse(args, "prd edit", [_problem("NO_ITEM", iid, "no such item")])
        fields = {k: v for k, v in (("text", args.text), ("priority", args.priority), ("acceptance", args.acceptance))
                  if v is not None}
        item = db.edit_item(iid, **fields) if fields else db.item(iid)
    paths = _flip_and_render(db, "prd")
    cli._emit(args, {"item": item, "written": paths},
              f"prd {verb} {iid} [{item['kind']}] prio={item['priority'] or '-'}; {_wrote(db, paths)}")
    return EXIT_PASS


# ---------------------------------------------------------------------------
# frd

def _fmt_bounds(r: dict) -> str:
    if not r.get("metric"):
        return ""
    lo = "" if r.get("bound_min") is None else f"{r['bound_min']:g}"
    hi = "" if r.get("bound_max") is None else f"{r['bound_max']:g}"
    return f"{r['metric']}[{lo}..{hi}]{r.get('unit') or ''}"


def _fmt_check(c: dict) -> str:
    return f"{c['kind']}={c['status']}" + (f"({c['value']:g})" if c.get("value") is not None else "")


def _item_line(db, r: dict, checks: list[dict] | None = None) -> str:
    nv = len(db.verifiers(item_id=r["id"]))
    ck = " ".join(_fmt_check(c) for c in (checks if checks is not None else db.latest_checks_by_kind(r["id"])))
    return (f"{r['id']:<16} {r['kind']:<5} {r['status']:<12} {r['priority'] or '-':<12} v={nv} "
            f"{_fmt_bounds(r)}  {r['text'][:70]}" + (f"  [{ck}]" if ck else ""))


def _frd_write(args, db, verb: str) -> int:
    """``frd add`` / ``frd edit``."""
    from orchestrator.harness.tools import extract
    from orchestrator.harness.tools.render import frd_section
    from orchestrator.state_store.ontology import is_item_id
    cli = _cli()
    iid = (args.id or "").strip()
    if not is_item_id(iid):
        return _refuse(args, f"frd {verb}", [_problem("FRD_BAD_ID", iid, "id must look like PERF-001 / INV-003 / IFACE-NNN")])
    kind = iid.split("-")[0].upper()
    prev = db.item(iid)
    if verb == "add" and prev and prev.get("status") != "retired":
        return _refuse(args, "frd add", [_problem("FRD_EXISTS", iid, f"already exists; use coresmith frd edit {iid}")])
    if verb == "edit" and not prev:
        return _refuse(args, "frd edit", [_problem("NO_ITEM", iid, "no such item; use coresmith frd add")])
    try:
        lo, hi = _bound(args.min), _bound(args.max)
    except ValueError as exc:
        return _refuse(args, f"frd {verb}", [_problem("FRD_BAD_BOUND", iid, f"--min/--max must be numbers: {exc}")])
    base = prev if verb == "edit" else {}
    eff_lo = base.get("bound_min") if lo == _UNSET else lo
    eff_hi = base.get("bound_max") if hi == _UNSET else hi
    eff_metric = base.get("metric") if args.metric is None else (args.metric or None)
    problems = []
    if kind in BOUNDED_KINDS and eff_lo is None and eff_hi is None:
        problems.append(_problem("FRD_NO_BOUNDS", iid, f"a {kind} item must be measurable: give --metric and --min and/or --max"))
    if (eff_lo is not None or eff_hi is not None) and not eff_metric:
        problems.append(_problem("FRD_NO_METRIC", iid, "a bound needs --metric (the measured quantity, e.g. fps)"))
    if eff_lo is not None and eff_hi is not None and eff_lo > eff_hi:
        problems.append(_problem("FRD_BAD_BOUNDS", iid, f"--min {eff_lo} > --max {eff_hi}"))
    derives = _split_ids(args.derives_from)
    bad = [d for d in derives if not is_item_id(d)]
    if bad:
        problems.append(_problem("FRD_BAD_REF", ",".join(bad), "--derives-from ids must be item ids"))
    if problems:
        return _refuse(args, f"frd {verb}", problems)

    if verb == "add":
        db.upsert_item("frd", {
            "id": iid, "kind": kind, "section": args.section or frd_section(kind), "text": args.text,
            "priority": args.priority or "must_have", "acceptance": args.acceptance or "",
            "model_check": args.model_check or "", "metric": eff_metric, "bound_min": eff_lo, "bound_max": eff_hi,
            "unit": args.unit, "status": "open"})
    else:
        fields = {k: v for k, v in (("text", args.text), ("priority", args.priority), ("acceptance", args.acceptance),
                                    ("model_check", args.model_check), ("unit", args.unit), ("section", args.section),
                                    ("metric", args.metric)) if v is not None}
        if lo != _UNSET:
            fields["bound_min"] = lo
        if hi != _UNSET:
            fields["bound_max"] = hi
        if fields:
            db.edit_item(iid, **fields)
    item = db.item(iid)
    refs = list(derives)
    if verb == "add" or args.text is not None or args.acceptance is not None:
        refs += [r for r in extract.references(item["text"] + " " + (item.get("acceptance") or ""), exclude=iid)
                 if r not in refs]
    linked = []
    for ref in refs:
        if ref != iid:
            db.link_items(iid, ref, "derives_from", source="cli")
            linked.append(ref)
    owner = (args.owner or "").strip()
    if owner:
        if verb == "edit":  # --owner replaces: an item has one owning block
            db.unlink_items(iid, "block:*", "owned_by")
        db.link_items(iid, f"block:{owner}", "owned_by", source="cli")
    notes = []
    known = {i["id"] for i in db.items()}
    notes += [_problem("FRD_REF_UNKNOWN", ref, "derives_from target is not a registered item (yet)", "warning")
              for ref in linked if ref not in known]
    if owner:
        try:
            blocks = {b["name"] for b in db.block_specs()}
        except Exception:  # noqa: BLE001
            blocks = set()
        if blocks and owner not in blocks:
            notes.append(_problem("FRD_OWNER_UNKNOWN", owner, "not a block of the registered block diagram", "warning"))
    adv = advisories(db, item)
    paths = _flip_and_render(db, "frd")
    lines = [f"frd {verb} {iid}: OK [{kind}] prio={item['priority'] or '-'} {_fmt_bounds(item)}".rstrip()]
    if linked:
        lines.append(f"  derives_from: {', '.join(linked)}")
    if owner:
        lines.append(f"  owned_by: block:{owner}")
    if notes or adv:
        lines.append("  still required / advisory:")
        lines += cli._problems_lines(notes + adv)
    lines.append(f"  {_wrote(db, paths)}")
    cli._emit(args, {"ok": True, "item": item, "derives_from": linked, "owner": owner or None,
                     "advisories": notes + adv, "written": paths}, "\n".join(lines))
    return EXIT_PASS


def _frd_show(args, db) -> int:
    cli = _cli()
    it = db.item(args.id)
    if not it:
        return _refuse(args, "frd show", [_problem("NO_ITEM", args.id, "no such item")])
    from orchestrator.state_store.ontology import check_rank
    latest = db.latest_checks_by_kind(args.id)
    top = max((check_rank(c["kind"]) for c in latest), default=None)
    payload = {"item": it, "links_out": db.links(from_id=args.id), "links_in": db.links(to_id=args.id),
               "verifiers": db.verifiers(item_id=args.id), "checks": latest,
               "status_from": [c["kind"] for c in latest if check_rank(c["kind"]) == top],
               "advisories": advisories(db, it) if it.get("status") != "retired" else []}
    lines = [f"{it['id']} [{it['kind']}] {it['status']} prio={it['priority'] or '-'} section={it.get('section') or '-'}",
             f"  {it['text'][:300]}",
             f"  acceptance: {(it.get('acceptance') or '-')[:200]}",
             f"  model check: {(it.get('model_check') or '-')[:200]}",
             f"  metric: {it.get('metric') or '-'} min={it.get('bound_min')} max={it.get('bound_max')} "
             f"unit={it.get('unit') or '-'}"]
    lines += [f"  -> {lk['rel']} {lk['to_id']}" for lk in payload["links_out"]]
    lines += [f"  <- {lk['rel']} {lk['from_id']}" for lk in payload["links_in"]]
    lines += [f"  verifier #{v['id']} {v['kind']} {v['path'] or '-'}::{v['entry'] or '-'} block={v['block'] or '-'}"
              for v in payload["verifiers"]]
    lines += [f"  check {c['kind']} (rank {check_rank(c['kind'])}): {c['status']}"
              + (f" value={c['value']:g}" if c.get("value") is not None else "")
              + (" <- decides the status" if c["kind"] in payload["status_from"] else "")
              + f" {(c['evidence'] or '')[:100]}" for c in payload["checks"]]
    lines += cli._problems_lines(payload["advisories"])
    cli._emit(args, payload, "\n".join(lines))
    return EXIT_PASS


def _frd_verifier(args, db) -> int:
    from orchestrator.state_store.ontology import VERIFIER_KINDS
    cli = _cli()
    iid = args.id
    it = db.item(iid)
    if not it or it.get("status") == "retired":
        return _refuse(args, "frd verifier", [_problem("NO_ITEM", iid, "no such item")])
    if args.kind not in VERIFIER_KINDS:
        return _refuse(args, "frd verifier", [_problem("FRD_BAD_VERIFIER_KIND", args.kind, f"one of {VERIFIER_KINDS}")])
    if args.kind == "cocotb" and not (args.path and args.entry):
        return _refuse(args, "frd verifier", [_problem("FRD_VERIFIER_INCOMPLETE", iid,
                                                       "a cocotb verifier needs --path <tb.py> and --entry <cocotb test name>")])
    if args.kind == "chip" and not args.entry:
        return _refuse(args, "frd verifier", [_problem("FRD_VERIFIER_INCOMPLETE", iid,
                                                       "a chip verifier needs --entry <cocotb test name of the "
                                                       "integration/validation testbench> (--path optional)")])
    vargs = None
    if args.args:
        try:
            vargs = json.loads(args.args)
        except ValueError as exc:
            return _refuse(args, "frd verifier", [_problem("FRD_BAD_ARGS", iid, f"--args is not JSON: {exc}")])
    if args.kind == "eda":
        from orchestrator.state_store.module_targets import EDA_MEASURES, unit_factor
        if args.entry not in EDA_MEASURES:
            return _refuse(args, "frd verifier", [_problem(
                "FRD_VERIFIER_INCOMPLETE", iid, f"an eda verifier needs --entry {' | '.join(sorted(EDA_MEASURES))} "
                "(measured by the engine on the module build's synthesized netlist)")])
        if it.get("bound_min") is None and it.get("bound_max") is None:
            return _refuse(args, "frd verifier", [_problem("FRD_NO_BOUNDS", iid, "an eda measurement needs a bound "
                                                           "to judge: coresmith frd edit --metric/--min/--max")])
        if unit_factor(args.entry, it.get("unit")) is None:
            return _refuse(args, "frd verifier", [_problem(
                "FRD_BAD_UNIT", iid, f"{args.entry} is measured in {EDA_MEASURES[args.entry]['base_unit']}; set "
                f"--unit to one of {sorted(EDA_MEASURES[args.entry]['units'])} (frd edit {iid} --unit ...)")])
        if vargs is not None and not isinstance(vargs, dict):
            return _refuse(args, "frd verifier", [_problem("FRD_BAD_ARGS", iid, "--args must be a JSON object")])
    block = args.block or ""
    if not block:
        owners = [lk["to_id"].split(":", 1)[1] for lk in db.links(from_id=iid, rel="owned_by")
                  if lk["to_id"].startswith("block:")]
        if len(owners) == 1:
            block = owners[0]
    same = [v for v in db.verifiers(item_id=iid) if v["kind"] == args.kind and (v["path"] or "") == (args.path or "")
            and (v["entry"] or "") == (args.entry or "")]
    if same and same[0]["block"] and same[0]["block"] != block and not getattr(args, "replace", False):
        v0 = same[0]
        return _refuse(args, "frd verifier", [_problem(
            "VERIFIER_REBOUND", f"#{v0['id']}",
            f"{iid} {args.path or '-'}::{args.entry or '-'} is already bound to block {v0['block']!r}; "
            f"--replace to rebind it to {block or '(none)'!r}")])
    notes = []
    if block:
        owners = {lk["to_id"].split(":", 1)[1] for lk in db.links(from_id=iid, rel="owned_by")
                  if lk["to_id"].startswith("block:")}
        if owners and block not in owners:
            notes.append(_problem("VERIFIER_BLOCK_NOT_OWNER", block,
                                  f"{iid} is owned by {', '.join(sorted(owners))}, not {block}", "warning"))
    vid = db.add_verifier(iid, args.kind, path=args.path or "", entry=args.entry or "", args=vargs, block=block)
    paths = _flip_and_render(db, "frd")
    lines = [f"verifier #{vid}: {iid} {args.kind} {args.path or '-'}" + (f"::{args.entry}" if args.entry else "")
             + f" block={block or '-'}" + (" (rebound)" if same and same[0]["block"] != block else "")]
    cli._emit(args, {"id": vid, "verifier": db.verifier(vid), "written": paths, "advisories": notes},
              "\n".join(lines + cli._problems_lines(notes)))
    return EXIT_PASS


def cmd_frd(args) -> int:
    cli = _cli()
    db = _db(args)
    verb = getattr(args, "verb", None) or "list"
    if verb in ("add", "edit"):
        return _frd_write(args, db, verb)
    if verb == "show":
        return _frd_show(args, db)
    if verb == "verifier":
        return _frd_verifier(args, db)
    if verb == "retire":
        if not db.retire_item(args.id):
            return _refuse(args, "frd retire", [_problem("NO_ITEM", args.id, "no such item")])
        paths = _flip_and_render(db, "frd")
        cli._emit(args, {"id": args.id, "retired": True, "written": paths}, f"{args.id} retired; {_wrote(db, paths)}")
        return EXIT_PASS
    if verb == "list":
        from orchestrator.state_store.ontology import item_must_have
        rows = db.items(artifact="frd", kind=(getattr(args, "kind", None) or "").upper() or None,
                        must_have=bool(getattr(args, "must", False)))
        if getattr(args, "unverified", False):
            rows = [r for r in rows if item_must_have(r) and not db.verifiers(item_id=r["id"])]
        for r in rows:
            r["checks"] = [{k: c[k] for k in ("kind", "status", "value", "ts")} for c in db.latest_checks_by_kind(r["id"])]
        cli._emit(args, {"items": rows}, "\n".join(_item_line(db, r, r["checks"]) for r in rows) or "(no FRD items)")
        return EXIT_PASS
    if verb == "verifiers":
        rows = db.verifiers(item_id=args.id or None, block=args.block or None, kind=args.kind or None)
        lines = [f"#{v['id']:<4} {v['item_id']:<16} {v['kind']:<8} {v['block'] or '-':<12} {v['path'] or '-'}"
                 + (f"::{v['entry']}" if v["entry"] else "") for v in rows] or ["(no verifiers)"]
        cli._emit(args, {"verifiers": rows}, "\n".join(lines))
        return EXIT_PASS
    if verb == "verifier-rm":
        v = db.verifier(args.vid)
        if not v or not db.remove_verifier(args.vid):
            return _refuse(args, "frd verifier-rm", [_problem("NO_VERIFIER", str(args.vid), "no such verifier")])
        paths = _flip_and_render(db, "frd")
        cli._emit(args, {"removed": args.vid, "item_id": v["item_id"], "written": paths},
                  f"verifier #{args.vid} of {v['item_id']} removed; {_wrote(db, paths)}")
        return EXIT_PASS
    if verb == "render":
        from orchestrator.harness.tools import render
        art = db.artifact("frd")
        if not args.out and art and not render.db_sourced(db, "frd") and not args.force:
            return _refuse(args, "frd render", [_problem(
                "FRD_FILE_SOURCED", art["path"],
                "the FRD is registered from a file; --out elsewhere, or --force to overwrite arch/frd_spec.md")])
        p = render.write_frd(db, db.root, args.out)
        cli._emit(args, {"written": [str(p)]}, str(p))
        return EXIT_PASS
    return _refuse(args, "frd", [_problem("BAD_VERB", str(verb), "unknown verb")])


# ---------------------------------------------------------------------------
# state write

def cmd_state_write(args) -> int:
    """``coresmith state write [--force]``: export the registry views and render
    the DB-sourced requirement documents (file-sourced ones only with --force)."""
    from orchestrator.harness.tools import render
    cli = _cli()
    db = _db(args)
    db.export_views()
    res = render.write_views(db, db.root, force=bool(getattr(args, "force", False)))
    try:
        from orchestrator.harness.top_module import declared_top
        top = declared_top(db.root) or "chip_top"
    except Exception:  # noqa: BLE001
        top = "chip_top"
    res["written"] += [str(p) for p in render.write_pin_views(db, db.root, top=top)]
    lines = ["state write: exported registry views"]
    lines += [f"  wrote {_rel(db, p)}" for p in res["written"]]
    lines += [f"  skipped {s['kind']}: {s['reason']}" for s in res["skipped"]]
    cli._emit(args, res, "\n".join(lines))
    return EXIT_PASS


# ---------------------------------------------------------------------------
# argparse

def _frd_item_opts(p, *, add: bool) -> None:
    p.add_argument("--priority", default=None, help="must_have (default on add) | should_have | nice_to_have")
    p.add_argument("--acceptance", default=None, help="measurable pass/fail criterion")
    p.add_argument("--model-check", dest="model_check", default=None,
                   help="how the architecture model observes it, or 'not model-testable -- <reason>'")
    p.add_argument("--metric", default=None, help="measured quantity (required with a bound), e.g. fps")
    p.add_argument("--min", default=None, help="lower bound (number; 'none' clears on edit)")
    p.add_argument("--max", default=None, help="upper bound (number; 'none' clears on edit)")
    p.add_argument("--unit", default=None)
    p.add_argument("--owner", default=None, help="owning block (item -owned_by-> block:<owner>)")
    p.add_argument("--derives-from", dest="derives_from", default=None, help="comma-separated PRD/FRD ids")
    p.add_argument("--section", default=None, help="FRD section heading (default by id prefix)")
    if not add:
        p.add_argument("--text", default=None, help="new requirement text")


def register(sub, run, add_project_root, add_json) -> None:
    """Add this module's subcommands to the ``bin/coresmith`` subparser action."""
    def leaf(parent, name, verb, handler, help_=None):
        p = parent.add_parser(name, help=help_)
        add_project_root(p)
        add_json(p)
        p.set_defaults(func=run(handler), verb=verb)
        return p

    pp = sub.add_parser("prd", help="PRD line items in the project database: add | edit | retire | list")
    pp.set_defaults(func=run(cmd_prd), verb="list")
    psub = pp.add_subparsers(dest="verb")
    pa = leaf(psub, "add", "add", cmd_prd, "add a PRD item (FR-*, KPI-*, CON-*)")
    pa.add_argument("text")
    pa.add_argument("--id", required=True)
    pa.add_argument("--kind", choices=PRD_KINDS, default=None, help="default: the id prefix")
    pa.add_argument("--priority", default=None, help="must_have | should_have (KPI/CON default must_have)")
    pa.add_argument("--acceptance", default=None, help="KPI: '<threshold> -- <test method>'")
    pe = leaf(psub, "edit", "edit", cmd_prd, "edit a PRD item")
    pe.add_argument("id")
    pe.add_argument("--text", default=None)
    pe.add_argument("--priority", default=None)
    pe.add_argument("--acceptance", default=None)
    pr = leaf(psub, "retire", "retire", cmd_prd, "retire a PRD item")
    pr.add_argument("id")
    pl = leaf(psub, "list", "list", cmd_prd, "list PRD items")
    pl.add_argument("--kind", default=None)

    fp = sub.add_parser("frd", help="FRD line items in the project database: add | edit | retire | list | show | "
                                    "verifier | verifiers | verifier-rm | render")
    fp.set_defaults(func=run(cmd_frd), verb="list")
    fsub = fp.add_subparsers(dest="verb")
    fa = leaf(fsub, "add", "add", cmd_frd, "add an FRD item (PERF/TIME items need bounds)")
    fa.add_argument("text")
    fa.add_argument("--id", required=True)
    _frd_item_opts(fa, add=True)
    fe = leaf(fsub, "edit", "edit", cmd_frd, "edit an FRD item (links are additive)")
    fe.add_argument("id")
    _frd_item_opts(fe, add=False)
    fr = leaf(fsub, "retire", "retire", cmd_frd, "retire an FRD item")
    fr.add_argument("id")
    fl = leaf(fsub, "list", "list", cmd_frd, "list FRD items")
    fl.add_argument("--must", action="store_true", help="must-have items only")
    fl.add_argument("--kind", default=None, help="PERF | IFACE | INV | TIME | ...")
    fl.add_argument("--unverified", action="store_true", help="must-have items with no verifier")
    fs = leaf(fsub, "show", "show", cmd_frd, "one item: bounds, links, verifiers, latest checks")
    fs.add_argument("id")
    fv = leaf(fsub, "verifier", "verifier", cmd_frd, "attach a verifier to an item")
    fv.add_argument("id")
    fv.add_argument("--kind", required=True,
                    help="cocotb | eda | python | systemc | judge | manual | chip. A module build measures its "
                         "targets through cocotb (an acceptance test that calls harness.measure.record) and eda "
                         "(the engine's synthesis/STA) verifiers")
    fv.add_argument("--path", default=None, help="e.g. tb/cocotb/test_fft64.py (cocotb: required; the module's "
                                                 "acceptance testbench)")
    fv.add_argument("--entry", default=None, help="cocotb: the test name (required); eda: area_um2 | power_mw")
    fv.add_argument("--block", default=None, help="default: the item's owner when it has exactly one")
    fv.add_argument("--args", default=None,
                    help="JSON arguments, e.g. a workload label {\"workload\":\"dhrystone\"}; eda power_mw: "
                         "{\"activity\":0.1,\"duty\":0.5} (primary-input toggles per clock, propagated by OpenSTA); "
                         "{\"scope\":\"integration\"} defers the item to chip-level verification")
    fv.add_argument("--replace", action="store_true",
                    help="rebind an existing (item, path, entry) verifier to a different --block")
    fvs = leaf(fsub, "verifiers", "verifiers", cmd_frd, "list verifiers")
    fvs.add_argument("id", nargs="?", default=None)
    fvs.add_argument("--block", default=None)
    fvs.add_argument("--kind", default=None)
    fvr = leaf(fsub, "verifier-rm", "verifier-rm", cmd_frd, "remove a verifier by id")
    fvr.add_argument("vid", type=int)
    fre = leaf(fsub, "render", "render", cmd_frd, "write the FRD markdown from the database")
    fre.add_argument("--out", default=None, help="default arch/frd_spec.md")
    fre.add_argument("--force", action="store_true", help="overwrite a file-sourced arch/frd_spec.md")
