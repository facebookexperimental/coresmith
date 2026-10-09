# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""``coresmith contract ...``: interface-contract edges as line items.

    contract add <pb>.<pp> <cb>.<cp> --protocol <family> [--width N] [--edge-id ID]
                 [--bus-param k=v ...] [--field name:width[:msb:lsb] ...]
                 [--sideband name:width ...] [--timing k=v ...] [--policy TEXT]
                 [--semantic TEXT] [--spec JSON] [--unlock --reason TEXT]
    contract set <edge_id> <dotted.key> <value> [--unlock --reason TEXT]
    contract rm <edge_id> [--unlock --reason TEXT]
    contract lock [--block b] | contract unlock [--block b] --reason TEXT
    contract show <edge_id> | contract list [--block b] [--locked]

Every write assembles the would-be contracts document and runs the validator
``register contracts`` runs (``tools.validate.validate_contracts``, against the
registered block diagram when there is one). An ``error`` problem this change
introduces -- or one that names the edge being written -- refuses the write
and nothing is written; problems the document already had are reported, not
fatal, so a contract set can be built one edge at a time. Locked edges (locked
when the ``interfaces`` stage completes) change only with ``--unlock --reason``,
which licenses that one write: the edge stays locked (``contract unlock`` is
the only way to leave an edge open). ``contract add`` also refuses an unknown
protocol family (``CT_BAD_PROTOCOL``) and, once blocks are registered, an
endpoint block that is not one of them (``CT_UNKNOWN_BLOCK``); an edge between
blocks the diagram does not connect is a warning (``CT_OFF_DIAGRAM``). Every
successful write marks the ``contracts`` artifact DB-sourced (``db:contracts``).

Wired from ``orchestrator.harness.cli.register_subcommands`` via ``register``.
Like ``cli.py``, this module must import without importing ``orchestrator.langgraph``.
"""

from __future__ import annotations

import json
import sys
from typing import Any

_FAIL, _USAGE = 1, 2
# diagram-coverage problems: a pre-existing gap is not the written edge's fault
_COVERAGE = ("CT_MISSING_EDGE", "CT_MISSING_CONTRACT", "CT_FABRIC_PORT_UNREACHED")


def _cli():
    from orchestrator.harness import cli
    return cli


def _literal(v: str) -> Any:
    from orchestrator.state_store.project_db import _coerce_literal
    return _coerce_literal(v)


def _kv(items, what: str) -> dict:
    from orchestrator.state_store.project_db import _set_dotted
    out: dict = {}
    for it in items or []:
        k, sep, v = str(it).partition("=")
        if not sep or not k.strip():
            raise ValueError(f"{what} expects k=v, got {it!r}")
        _set_dotted(out, k.strip(), _literal(v))
    return out


def _endpoint(s: str) -> tuple[str, str]:
    b, sep, p = str(s).partition(".")
    if not sep or not b or not p:
        raise ValueError(f"endpoint must be <block>.<port>, got {s!r}")
    return b, p


def _fields(items) -> list[dict]:
    """``name:width`` packs LSB-first after the previous field; ``name:width:msb:lsb`` is explicit."""
    out, offset = [], 0
    for it in items or []:
        parts = str(it).split(":")
        if len(parts) not in (2, 4) or not parts[0]:
            raise ValueError(f"--field expects name:width[:msb:lsb], got {it!r}")
        w = int(parts[1])
        if len(parts) == 4:
            msb, lsb = int(parts[2]), int(parts[3])
        else:
            lsb, msb = offset, offset + w - 1
        offset = max(offset, msb + 1)
        out.append({"name": parts[0], "width": w, "msb": msb, "lsb": lsb})
    return out


def _sidebands(items) -> list[dict]:
    out = []
    for it in items or []:
        name, sep, w = str(it).partition(":")
        if not sep or not name:
            raise ValueError(f"--sideband expects name:width, got {it!r}")
        out.append({"name": name, "width": int(w)})
    return out


def _normalize(edge: dict) -> list[str]:
    """Complete the structured timing object (in place) when the edge carries
    one or its family needs one (req_resp latency); returns the normaliser's notes."""
    from orchestrator.architecture.specialists.contract_timing import REQ_RESP, normalize_timing
    fam = str(edge.get("handshake_protocol") or "").strip().lower()
    if isinstance(edge.get("timing"), dict) or fam in REQ_RESP:
        _changed, notes = normalize_timing(edge)
        return list(notes)
    return []


def _key(p: dict) -> tuple:
    return (p.get("code"), p.get("where"), p.get("text"))


def _check(db, edge_id: str, new_edge: dict | None) -> tuple[list[dict], list[dict]]:
    """(fatal, reported) problems of replacing/adding (``new_edge``) or removing
    (``None``) ``edge_id`` in the current contracts document."""
    from orchestrator.harness.tools import validate
    doc = db.contracts() or {}
    before = list(doc.get("contracts") or [])
    after = [e for e in before if str(e.get("edge_id") or "") != edge_id]
    if new_edge is not None:
        idx = next((i for i, e in enumerate(before) if str(e.get("edge_id") or "") == edge_id), None)
        after.insert(len(after) if idx is None else idx, new_edge)
    diagram = db.block_diagram() or None
    probs = validate.validate_contracts({**doc, "contracts": after}, diagram) if after else []
    if before:
        old = {_key(p) for p in validate.validate_contracts({**doc, "contracts": before}, diagram)}
    else:   # the first edge: the diagram connections it leaves uncovered were uncovered already
        old = {_key(p) for p in probs if p.get("code") in _COVERAGE}
    fatal = [p for p in probs if (p.get("severity") or "error") == "error"
             and (_key(p) not in old or edge_id in str(p.get("where") or ""))]
    return fatal, [p for p in probs if p not in fatal]


def _lines(problems, tag: str = "") -> list[str]:
    return [f"  [{(p.get('severity') or 'error')[:4]}{tag}] {p.get('code')} {p.get('where')}: {p.get('text')}"
            for p in problems]


def _unlock_ok(args) -> bool:
    if getattr(args, "unlock", False) and not (getattr(args, "reason", "") or "").strip():
        print("--unlock requires --reason TEXT (it is recorded in the actions log)", file=sys.stderr)
        return False
    return True


def _locked(args, exc, verb: str) -> int:
    problem = {"code": "CT_LOCKED", "where": ", ".join(exc.edge_ids),
               "text": f"locked edges would change; coresmith contract {verb} ... --unlock --reason TEXT",
               "severity": "error"}
    _cli()._emit(args, {"ok": False, "edge_ids": exc.edge_ids, "problems": [problem]},
                 "\n".join([f"contract {verb}: REFUSED (CT_LOCKED {', '.join(exc.edge_ids)})", *_lines([problem])]))
    return _FAIL


def _families() -> tuple[str, ...]:
    from orchestrator.architecture.specialists.contract_timing import (
        ALWAYS_ACCEPTED,
        REQ_RESP,
        STATIC,
        STREAMING,
    )
    return tuple(STREAMING) + tuple(REQ_RESP) + tuple(ALWAYS_ACCEPTED) + tuple(STATIC)


def _base_block(name: str) -> str:
    """``core[1]`` -> ``core`` (an instance of a multi-instance block)."""
    return str(name).split("[", 1)[0]


def _edge_checks(db, edge: dict) -> tuple[list[dict], list[dict]]:
    """(fatal, warnings) of one new edge beyond the document validator: the
    protocol family, the endpoint blocks, and whether the diagram connects them."""
    from orchestrator.harness.tools.validate import _port_of
    fatal, warn = [], []
    eid = str(edge.get("edge_id") or "")
    fam = str(edge.get("handshake_protocol") or "").strip().lower()
    if fam not in _families():
        fatal.append({"code": "CT_BAD_PROTOCOL", "where": eid, "severity": "error",
                      "text": f"unknown protocol family {fam!r}; one of {', '.join(_families())}"})
    pb, cb = str(edge.get("producer_block") or ""), str(edge.get("consumer_block") or "")
    try:
        names = {b["name"] for b in db.blocks()}
    except Exception:  # noqa: BLE001
        names = set()
    if names:
        unknown = [b for b in (pb, cb) if _base_block(b) not in names]
        if unknown:
            text = f"not a registered block: {', '.join(unknown)} (coresmith blocks)"
            if any(_looks_like_top(db, b) for b in unknown):
                text += ("; chip I/O is declared with `coresmith pin add <name> --dir in|out|inout "
                         "--from <block>.<port>`, not a contract edge")
            fatal.append({"code": "CT_UNKNOWN_BLOCK", "where": eid, "severity": "error", "text": text})
    conns = list((db.block_diagram() or {}).get("connections") or [])
    if conns:
        pairs = {(_base_block(_port_of(c, "producer")[0] or ""), _base_block(_port_of(c, "consumer")[0] or ""))
                 for c in conns}
        if (_base_block(pb), _base_block(cb)) not in pairs and (_base_block(cb), _base_block(pb)) not in pairs:
            warn.append({"code": "CT_OFF_DIAGRAM", "where": eid, "severity": "warning",
                         "text": f"the block diagram has no connection between {pb} and {cb} (add the connection; "
                                 "chip I/O is `coresmith pin add`, not an edge)"})
    return fatal, warn


_TOP_NAMES = ("chip", "chip_top", "top", "soc_top", "soc", "pads", "io", "chip_io")


def _looks_like_top(db, name: str) -> bool:
    """Whether an unknown endpoint block is really the chip boundary."""
    n = _base_block(name).lower()
    if n in _TOP_NAMES:
        return True
    try:
        from orchestrator.harness.top_module import declared_top
        top = (declared_top(db.root) or "").lower()
    except Exception:  # noqa: BLE001
        top = ""
    return bool(top) and n == top


def _mark_db_sourced(db) -> None:
    """The contracts artifact now comes from the rows (like the frd verbs)."""
    db.ensure_db_artifact("contracts", registered_by="cli")
    db.register_artifact("contracts", "db:contracts", sha="db", registered_by="cli")


def _write(args, db, verb: str, edge_id: str, new_edge: dict | None, writer, notes=(), pre=()) -> int:
    """Validate the would-be document, then run ``writer`` (the one DB write).
    ``pre``: (fatal, warnings) from :func:`_edge_checks`."""
    from orchestrator.state_store.project_db import ContractLockedError
    cli = _cli()
    pre_fatal, pre_warn = (list(pre[0]), list(pre[1])) if pre else ([], [])
    fatal, reported = _check(db, edge_id, new_edge)
    fatal = pre_fatal + fatal
    warn = pre_warn + [{"code": "CT_TIMING_NORMALIZED", "where": edge_id, "text": n, "severity": "warning"}
                       for n in notes]
    if fatal:
        cli._emit(args, {"ok": False, "edge_id": edge_id, "problems": fatal + reported + warn},
                  "\n".join([f"contract {verb} {edge_id}: REFUSED ({', '.join(sorted({p['code'] for p in fatal}))})",
                             *_lines(fatal), *_lines(reported, ",pre"), *_lines(warn)]))
        return _FAIL
    try:
        res = dict(writer() or {})
    except ContractLockedError as exc:
        return _locked(args, exc, verb)
    _mark_db_sourced(db)
    res.update({"ok": True, "edge_id": edge_id, "problems": reported + warn})
    head = f"contract {verb} {edge_id}: OK"
    if "version" in res:
        head += f" v{res['version']}{' (unchanged)' if res.get('changed') is False else ''}"
    if getattr(args, "unlock", False):
        head += f" [unlocked for this write: {args.reason.strip()}]"
        res["reason"] = args.reason.strip()
    cli._emit(args, res, "\n".join([head, *_lines(reported, ",pre"), *_lines(warn)]))
    return 0


# --------------------------------------------------------------------- verbs
def cmd_contract_add(args) -> int:
    if not _unlock_ok(args):
        return _USAGE
    try:
        pb, pp = _endpoint(args.producer)
        cb, cp = _endpoint(args.consumer)
        edge: dict = json.loads(args.spec) if args.spec else {}
        if not isinstance(edge, dict):
            raise ValueError("--spec must be a JSON object")
        edge.update({"producer_block": pb, "producer_port": pp, "consumer_block": cb, "consumer_port": cp,
                     "handshake_protocol": args.protocol})
        edge["edge_id"] = args.edge_id or edge.get("edge_id") or f"{pb}__{pp}__to__{cb}__{cp}"
        fields = _fields(args.field)
        if fields:
            edge["fields"] = fields
        if args.width is not None:
            edge["data_width_bits"] = int(args.width)
        elif fields and edge.get("data_width_bits") in (None, ""):
            edge["data_width_bits"] = sum(f["width"] for f in fields)
        if args.bus_param:
            edge["bus_params"] = {**(edge.get("bus_params") or {}), **_kv(args.bus_param, "--bus-param")}
        if args.sideband:
            edge["sideband_signals"] = _sidebands(args.sideband)
        if args.timing:
            t = edge.get("timing") if isinstance(edge.get("timing"), dict) else {}
            edge["timing"] = {**t, **_kv(args.timing, "--timing")}
        if args.policy is not None:
            edge["flow_control_policy"] = _literal(args.policy) if args.policy.lstrip().startswith("{") else args.policy
        if args.semantic is not None:
            edge["semantic_contract"] = args.semantic
    except (ValueError, TypeError) as exc:
        print(f"contract add: {exc}", file=sys.stderr)
        return _USAGE
    notes = _normalize(edge)
    db = _cli()._state_db(args)
    return _write(args, db, "add", edge["edge_id"], edge,
                  lambda: db.upsert_contract_edge(edge, unlock=bool(args.unlock)), notes, pre=_edge_checks(db, edge))


def cmd_contract_set(args) -> int:
    from orchestrator.state_store.project_db import _set_dotted
    if not _unlock_ok(args):
        return _USAGE
    db = _cli()._state_db(args)
    row = db.contract_edge(args.edge_id)
    if row is None:
        print(f"contract set: no edge {args.edge_id!r}", file=sys.stderr)
        return _USAGE
    key = args.key.strip()
    if key == "edge_id":
        print("contract set: edge_id cannot be changed (contract rm + contract add)", file=sys.stderr)
        return _USAGE
    plain = json.loads(json.dumps(row["spec"]))
    try:
        _set_dotted(plain, key, _literal(args.value))
    except ValueError as exc:
        print(f"contract set: {exc}", file=sys.stderr)
        return _USAGE
    edge = json.loads(json.dumps(plain))
    notes = _normalize(edge)
    unlock = bool(args.unlock)
    if edge == plain:
        def writer():
            return db.set_contract_field(args.edge_id, key, args.value, unlock=unlock)
    else:   # the timing normaliser completed the object: write the normalised edge in one step
        def writer():
            return db.upsert_contract_edge(edge, unlock=unlock)
    return _write(args, db, "set", args.edge_id, edge, writer, notes)


def cmd_contract_rm(args) -> int:
    if not _unlock_ok(args):
        return _USAGE
    db = _cli()._state_db(args)
    if db.contract_edge(args.edge_id) is None:
        print(f"contract rm: no edge {args.edge_id!r}", file=sys.stderr)
        return _USAGE
    if db.contract_edge(args.edge_id)["locked"] and not args.unlock:
        from orchestrator.state_store.project_db import ContractLockedError
        return _locked(args, ContractLockedError([args.edge_id]), "rm")
    doc = db.contracts() or {}
    rest = {**doc, "contracts": [e for e in doc.get("contracts") or [] if str(e.get("edge_id") or "") != args.edge_id]}
    return _write(args, db, "rm", args.edge_id, None,
                  lambda: {"contracts_version": db.import_contracts(rest, unlock=bool(args.unlock))})


def cmd_contract_lock(args) -> int:
    cli = _cli()
    unlock = args.verb == "unlock"
    reason = (getattr(args, "reason", "") or "").strip()
    if unlock and not reason:
        print("contract unlock requires --reason TEXT", file=sys.stderr)
        return _USAGE
    db = cli._state_db(args)
    n = db.lock_contracts(args.block or None, locked=not unlock)
    scope = f"block {args.block}" if args.block else "all edges"
    head = f"contract {args.verb}: {n} edge(s) ({scope})" + (f" [reason: {reason}]" if unlock else "")
    cli._emit(args, {"ok": True, "verb": args.verb, "block": args.block, "edges": n, "reason": reason}, head)
    return 0


def cmd_contract_show(args) -> int:
    cli = _cli()
    row = cli._state_db(args).contract_edge(args.edge_id)
    if row is None:
        print(f"contract show: no edge {args.edge_id!r}", file=sys.stderr)
        return _USAGE
    cli._emit(args, row, f"{row['edge_id']} v{row['version']}{' LOCKED' if row['locked'] else ''}\n"
                         + json.dumps(row["spec"], indent=2, default=str))
    return 0


def cmd_contract_list(args) -> int:
    cli = _cli()
    rows = cli._state_db(args).contract_rows()
    if args.block:
        rows = [r for r in rows if args.block in (r["spec"].get("producer_block"), r["spec"].get("consumer_block"))]
    if args.locked:
        rows = [r for r in rows if r["locked"]]
    out = [{"edge_id": r["edge_id"], "family": r["spec"].get("handshake_protocol"),
            "width": r["spec"].get("data_width_bits"), "version": r["version"], "locked": r["locked"]} for r in rows]
    lines = [f"{r['edge_id']}  {r['family'] or '?'}  w={r['width'] if r['width'] is not None else '?'}  "
             f"v{r['version']}  {'LOCKED' if r['locked'] else 'open'}" for r in out]
    cli._emit(args, {"contracts": out}, "\n".join(lines) or "no contract edges")
    return 0


def _unlock_args(p) -> None:
    p.add_argument("--unlock", action="store_true", help="allow changing a locked edge (needs --reason)")
    p.add_argument("--reason", default="", help="why the locked edge changes (recorded in the actions log)")


def register(sub, run, add_project_root, add_json) -> None:
    """Add this module's subcommands to the ``bin/coresmith`` subparser action."""
    cp = sub.add_parser("contract", help="interface-contract edges: add | set | rm | lock | unlock | show | list")
    csub = cp.add_subparsers(dest="verb")

    def _p(name, handler, help_):
        p = csub.add_parser(name, help=help_)
        add_project_root(p)
        add_json(p)
        p.set_defaults(func=run(handler), verb=name)
        return p

    a = _p("add", cmd_contract_add, "add or replace one edge (validated before it is written)")
    a.add_argument("producer", help="<producer_block>.<port>")
    a.add_argument("consumer", help="<consumer_block>.<port>")
    a.add_argument("--protocol", required=True,
                   help="axi4 | axi_lite | apb | axi_stream | srdy_drdy | req_resp | valid_only | mem_write | static")
    a.add_argument("--width", type=int, default=None, help="data_width_bits (default: sum of --field widths)")
    a.add_argument("--edge-id", dest="edge_id", default=None, help="default <pb>__<pp>__to__<cb>__<cp>")
    a.add_argument("--bus-param", dest="bus_param", action="append", default=[], metavar="K=V")
    a.add_argument("--field", action="append", default=[], metavar="NAME:WIDTH[:MSB:LSB]")
    a.add_argument("--sideband", action="append", default=[], metavar="NAME:WIDTH")
    a.add_argument("--timing", action="append", default=[], metavar="K=V",
                   help="timing key (dotted allowed, e.g. burst.max_beats=16)")
    a.add_argument("--policy", default=None, help="flow_control_policy (text or JSON object)")
    a.add_argument("--semantic", default=None, help="semantic_contract text")
    a.add_argument("--spec", default=None, help="JSON edge used as the base; flags override")
    _unlock_args(a)
    s = _p("set", cmd_contract_set, "set one (dotted) field of an edge")
    s.add_argument("edge_id")
    s.add_argument("key", help="dotted key, e.g. timing.valid_to_ready_max_stall")
    s.add_argument("value", help="JSON literal (8, true, null, {...}) or a string")
    _unlock_args(s)
    r = _p("rm", cmd_contract_rm, "remove one edge")
    r.add_argument("edge_id")
    _unlock_args(r)
    lk = _p("lock", cmd_contract_lock, "lock every edge (or a block's edges)")
    lk.add_argument("--block", default=None)
    ul = _p("unlock", cmd_contract_lock, "unlock every edge (or a block's edges)")
    ul.add_argument("--block", default=None)
    ul.add_argument("--reason", required=True)
    sh = _p("show", cmd_contract_show, "one edge: spec, version, lock")
    sh.add_argument("edge_id")
    ls = _p("list", cmd_contract_list, "one line per edge: edge_id, family, width, version, lock")
    ls.add_argument("--block", default=None)
    ls.add_argument("--locked", action="store_true", help="only locked edges")
