# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""``coresmith fabric ...`` spec verbs: the registered FabricSpec rows.

Line-item CLI verbs over the project database (``orchestrator.state_store``):
``init``, ``master add|rm``, ``slave add|rm``, ``set``, ``show``, ``params``,
``rm`` and ``derive``. The database row is the single source of truth: every
write loads the current row, applies one change, coerces the values through
``orchestrator.fabric.params`` and stores it with ``ProjectDB.set_fabric_spec``
(which validates, versions, and re-exports the read-only
``.coresmith/fabric_spec.json`` view). ``block_diagram()`` / ``block_specs()``
render the row into the fabric's primitive block.

Refusals carry stable codes: ``FABRIC_INVALID`` (the spec would not validate or a
parameter value is illegal; nothing written), ``FABRIC_NOT_FOUND``,
``FABRIC_AMBIGUOUS`` (several fabrics, no ``--name``), ``FABRIC_EXISTS``,
``FABRIC_PORT_NOT_FOUND``, ``FABRIC_USAGE`` (exit 2), ``FABRIC_DERIVE_FAILED``;
the ``show`` warning ``FABRIC_NO_PRIMITIVE_BLOCK``.

``FabricSpec.validate()`` refuses a fabric without masters or slaves, and the
database refuses any spec that does not validate, so ``init`` takes the first
master and slave inline (``--master`` / ``--slave``) and the row is valid from
its first version; removing the last master/slave is refused the same way.

Wired from ``orchestrator.harness.cli.register_subcommands`` via ``register``.
Like ``cli.py``, this module must import without importing ``orchestrator.langgraph``.
"""

from __future__ import annotations

import dataclasses
import difflib
import functools
import json


# ----------------------------------------------------------------- helpers
def _cli():
    from orchestrator.harness import cli
    return cli


class _Refusal(Exception):
    def __init__(self, code: str, problems: list[str], rc: int | None = None):
        super().__init__("; ".join(problems))
        self.code, self.problems, self.rc = code, list(problems), rc


def _refuse(args, exc: _Refusal) -> int:
    c = _cli()
    payload = {"ok": False, "code": exc.code, "problems": exc.problems}
    human = "\n".join([f"{exc.code}: fabric change refused, nothing written"] + [f"  - {p}" for p in exc.problems])
    c._emit(args, payload, human)
    return exc.rc if exc.rc is not None else c.EXIT_FAIL


def _row(db, name: str | None) -> dict:
    """The fabric row to edit: ``name``, or the only one."""
    rows = db.fabric_specs()
    if name:
        row = next((r for r in rows if r["name"] == name), None)
        if row is None:
            have = f" (have: {', '.join(r['name'] for r in rows)})" if rows else " (coresmith fabric init)"
            raise _Refusal("FABRIC_NOT_FOUND", [f"no fabric {name!r} registered{have}"])
        return row
    if not rows:
        raise _Refusal("FABRIC_NOT_FOUND", ["no fabric registered (coresmith fabric init --name ...)"])
    if len(rows) > 1:
        raise _Refusal("FABRIC_AMBIGUOUS", [f"{len(rows)} fabrics registered "
                                            f"({', '.join(r['name'] for r in rows)}); pass --name"])
    return rows[0]


def _spec_of(row_spec: dict, name: str):
    from orchestrator.fabric.spec import FabricSpec
    return FabricSpec.from_json({**row_spec, "name": name})


def _to_row(fs, extra_from: dict | None = None) -> dict:
    """FabricSpec -> the JSON stored in the row, keeping non-field keys of the
    previous row (e.g. ``derived_from`` provenance)."""
    from orchestrator.fabric.spec import FabricSpec
    fields = {f.name for f in dataclasses.fields(FabricSpec)}
    extras = {k: v for k, v in (extra_from or {}).items() if k not in fields}
    return {**extras, **fs.to_json()}


def _kv(items) -> list[tuple[str, str]]:
    out = []
    for it in items or []:
        if "=" not in it:
            raise _Refusal("FABRIC_USAGE", [f"--param {it!r}: expected key=value"], rc=_cli().EXIT_USAGE)
        k, v = it.split("=", 1)
        out.append((k.strip(), v.strip()))
    return out


def _coerce(role: str, protocol: str | None, param: str, raw):
    from orchestrator.fabric import params as fp
    try:
        return fp.coerce(role, protocol, param, raw)
    except ValueError as exc:
        raise _Refusal("FABRIC_INVALID", [str(exc)]) from None


def _check_protocol(role: str, protocol: str) -> None:
    from orchestrator.fabric import params as fp
    try:
        fp.allowed(role, protocol)
    except ValueError as exc:
        raise _Refusal("FABRIC_INVALID", [str(exc)]) from None


def _apply_params(obj, role: str, protocol: str | None, pairs) -> None:
    for k, v in pairs:
        setattr(obj, k, _coerce(role, protocol, k, v))


def _port(fs, role: str, name: str):
    ports = fs.masters if role == "master" else fs.slaves
    p = next((x for x in ports if x.name == name), None)
    if p is None:
        have = ", ".join(x.name for x in ports) or "none"
        raise _Refusal("FABRIC_PORT_NOT_FOUND", [f"no {role} {name!r} in fabric {fs.name!r} (have: {have})"])
    return p


def _store(args, db, name: str, fs, prev_spec: dict | None, what: str) -> int:
    """Validate and write ``fs`` as the row ``name``; emit the outcome."""
    c = _cli()
    problems = fs.validate()
    if problems:
        raise _Refusal("FABRIC_INVALID", problems)
    res = db.set_fabric_spec(name, _to_row(fs, prev_spec))
    if not res.get("ok"):
        raise _Refusal("FABRIC_INVALID", res.get("problems") or ["fabric spec refused"])
    view = db.path.parent / "fabric_spec.json"
    payload = {**res, "action": what, "spec": db.fabric_spec(name)["spec"], "view": str(view)}
    tag = "" if res.get("changed") else " (unchanged)"
    human = (f"fabric '{name}' v{res['version']}{tag}: {what} "
             f"({len(fs.masters)} master(s) x {len(fs.slaves)} slave(s), data {fs.data_width} b)")
    c._emit(args, payload, human)
    return c.EXIT_PASS


def _verb(handler):
    """``handler(args, db)`` -> a verb callable as ``f(args)`` (the CLI; opens the
    project database) or ``f(args, db)``; a ``_Refusal`` becomes a coded exit."""
    @functools.wraps(handler)
    def _f(args, db=None) -> int:
        try:
            return handler(args, db if db is not None else _cli()._state_db(args))
        except _Refusal as exc:
            return _refuse(args, exc)
    return _f


# ----------------------------------------------------------------- verbs
def _parse_slave_inline(text: str):
    from orchestrator.fabric.spec import FabricSlave
    parts = [p.strip() for p in text.split(":")]
    if len(parts) != 4:
        raise _Refusal("FABRIC_USAGE", [f"--slave {text!r}: expected name:protocol:base:size "
                                        "(e.g. ram:axi4:0x0:0x2000)"], rc=_cli().EXIT_USAGE)
    nm, proto, base, size = parts
    _check_protocol("slave", proto)
    return FabricSlave(name=nm, protocol=proto, base=_coerce("slave", proto, "base", base),
                       size=_coerce("slave", proto, "size", size))


@_verb
def cmd_fabric_init(args, db) -> int:
    """``coresmith fabric init --name N --master M --slave n:proto:base:size`` -- a new fabric row."""
    from orchestrator.fabric.spec import FabricMaster, FabricSpec
    name = args.name
    prev = db.fabric_spec(name)
    if prev is not None and not getattr(args, "replace", False):
        raise _Refusal("FABRIC_EXISTS", [f"fabric {name!r} already registered (v{prev['version']}); "
                                         "edit it with fabric set / master / slave, or pass --replace"])
    if not args.master or not args.slave:
        raise _Refusal("FABRIC_USAGE", ["a fabric needs at least one --master NAME and one "
                                        "--slave name:protocol:base:size (FabricSpec.validate refuses an "
                                        "empty fabric)"], rc=_cli().EXIT_USAGE)
    fs = FabricSpec(name=name)
    fs.data_width = _coerce("fabric", None, "data_width", args.data_width)
    fs.addr_width = _coerce("fabric", None, "addr_width", args.addr_width)
    _apply_params(fs, "fabric", None, _kv(args.param))
    fs.masters = [FabricMaster(name=m) for m in args.master]
    fs.slaves = [_parse_slave_inline(s) for s in args.slave]
    return _store(args, db, name, fs, None, "initialised" if prev is None else "replaced")


@_verb
def cmd_fabric_master_add(args, db) -> int:
    """``coresmith fabric master add NAME [--protocol axi4] [--param k=v]``."""
    from orchestrator.fabric.spec import FabricMaster
    row = _row(db, args.name)
    fs = _spec_of(row["spec"], row["name"])
    _check_protocol("master", args.protocol)
    m = FabricMaster(name=args.port, protocol=args.protocol)
    _apply_params(m, "master", args.protocol, _kv(args.param))
    fs.masters.append(m)
    return _store(args, db, row["name"], fs, row["spec"], f"master {args.port} added")


@_verb
def cmd_fabric_slave_add(args, db) -> int:
    """``coresmith fabric slave add NAME --protocol P --base 0x.. --size 0x.. [--param k=v]``."""
    from orchestrator.fabric.spec import FabricSlave
    row = _row(db, args.name)
    fs = _spec_of(row["spec"], row["name"])
    _check_protocol("slave", args.protocol)
    s = FabricSlave(name=args.port, protocol=args.protocol,
                    base=_coerce("slave", args.protocol, "base", args.base),
                    size=_coerce("slave", args.protocol, "size", args.size))
    _apply_params(s, "slave", args.protocol, _kv(args.param))
    fs.slaves.append(s)
    return _store(args, db, row["name"], fs, row["spec"], f"slave {args.port} added")


def _port_rm(args, db, role: str) -> int:
    row = _row(db, args.name)
    fs = _spec_of(row["spec"], row["name"])
    p = _port(fs, role, args.port)
    (fs.masters if role == "master" else fs.slaves).remove(p)
    return _store(args, db, row["name"], fs, row["spec"], f"{role} {args.port} removed")


@_verb
def cmd_fabric_master_rm(args, db) -> int:
    """``coresmith fabric master rm NAME``."""
    return _port_rm(args, db, "master")


@_verb
def cmd_fabric_slave_rm(args, db) -> int:
    """``coresmith fabric slave rm NAME``."""
    return _port_rm(args, db, "slave")


@_verb
def cmd_fabric_set(args, db) -> int:
    """``coresmith fabric set PARAM VALUE`` / ``set master|slave.PORT.PARAM VALUE``."""
    row = _row(db, args.name)
    fs = _spec_of(row["spec"], row["name"])
    key = args.param
    parts = key.split(".")
    if len(parts) == 1:
        setattr(fs, key, _coerce("fabric", None, key, args.value))
    elif len(parts) == 3 and parts[0] in ("master", "slave"):
        role, port, param = parts
        p = _port(fs, role, port)
        setattr(p, param, _coerce(role, p.protocol, param, args.value))
    else:
        raise _Refusal("FABRIC_USAGE", [f"{key!r}: expected PARAM (fabric-wide) or master|slave.PORT.PARAM"],
                       rc=_cli().EXIT_USAGE)
    return _store(args, db, row["name"], fs, row["spec"], f"{key} = {args.value}")


def _rendering_blocks(db, name: str) -> list[str]:
    try:
        doc = db.block_diagram() or {}
    except Exception:  # noqa: BLE001 - show must still work without a diagram
        return []
    return [str(b.get("name")) for b in doc.get("blocks") or []
            if str(b.get("kind") or "").lower() == "primitive"
            and isinstance(b.get("fabric"), dict) and b["fabric"].get("name") == name]


def _show_lines(row: dict, fs, blocks: list[str]) -> list[str]:
    from orchestrator.fabric import params as fp
    lines = [f"fabric '{row['name']}' v{row['version']}  module {fs.module_name}"]
    lines += [f"  {k:<18} {getattr(fs, k)}" for k in fp.allowed("fabric")]
    lines.append(f"  masters ({len(fs.masters)}):")
    lines += [f"    {m.name:<16} {m.protocol:<8} id_width={m.id_width} max_outstanding={m.max_outstanding}"
              for m in fs.masters]
    lines.append(f"  slaves ({len(fs.slaves)}):")
    for s in sorted(fs.slaves, key=lambda s: s.base):
        extra = "".join(f" {k}={getattr(s, k)}" for k in ("data_width", "max_outstanding")
                        if getattr(s, k) is not None)
        lines.append(f"    {s.name:<16} {s.protocol:<8} {s.base:#010x}-{s.base + s.size - 1:#010x} "
                     f"size={s.size:#x}{extra}")
    if blocks:
        lines.append(f"  rendered by primitive block(s): {', '.join(blocks)}")
    else:
        lines.append("  warning FABRIC_NO_PRIMITIVE_BLOCK: no primitive block in the block diagram renders "
                     f"this fabric (declare a block named {row['name']!r} with kind primitive, "
                     "primitive cs_fabric)")
    return lines


@_verb
def cmd_fabric_show(args, db) -> int:
    """``coresmith fabric show [--name]`` -- ports, params, version, rendering block."""
    c = _cli()
    rows = [_row(db, args.name)] if args.name else db.fabric_specs()
    if not rows:
        raise _Refusal("FABRIC_NOT_FOUND", ["no fabric registered (coresmith fabric init --name ...)"])
    out, lines = [], []
    for row in rows:
        fs = _spec_of(row["spec"], row["name"])
        blocks = _rendering_blocks(db, row["name"])
        out.append({"name": row["name"], "version": row["version"], "spec": row["spec"],
                    "problems": fs.validate(), "blocks": blocks,
                    "warnings": [] if blocks else ["FABRIC_NO_PRIMITIVE_BLOCK"]})
        lines += _show_lines(row, fs, blocks)
    c._emit(args, {"ok": True, "fabrics": out}, "\n".join(lines))
    return c.EXIT_PASS


def cmd_fabric_params(args) -> int:
    """``coresmith fabric params [fabric|master|slave] [protocol]`` -- the parameter table
    (no project database needed)."""
    from orchestrator.fabric import params as fp
    c = _cli()
    role, proto = getattr(args, "role", None), getattr(args, "protocol", None)
    try:
        if role and proto:
            fp.allowed(role, proto)
        rows = fp.rows(role, proto)
    except ValueError as exc:
        return _refuse(args, _Refusal("FABRIC_USAGE", [str(exc)], rc=c.EXIT_USAGE))
    c._emit(args, {"ok": True, "params": rows}, fp.describe(role, proto))
    return c.EXIT_PASS


@_verb
def cmd_fabric_rm(args, db) -> int:
    """``coresmith fabric rm --name N`` -- delete a fabric row."""
    c = _cli()
    row = _row(db, args.name)
    db.delete_fabric_spec(row["name"])
    c._emit(args, {"ok": True, "deleted": row["name"], "version": row["version"]},
            f"fabric '{row['name']}' (v{row['version']}) deleted")
    return c.EXIT_PASS


@_verb
def cmd_fabric_derive(args, db) -> int:
    """``coresmith fabric derive [--name] [--headroom 2.0] [--dry-run]`` -- FabricSpec
    from the arch model's measured link table, written to the fabric row."""
    from orchestrator.harness.tools import model as mt
    c = _cli()
    dry = bool(getattr(args, "dry_run", False))
    res = mt.fabric_derive(db, db.root, name=getattr(args, "name", None) or None,
                           headroom=float(getattr(args, "headroom", 2.0) or 2.0), write=not dry)
    if not res.get("ok"):
        if res.get("problems"):
            raise _Refusal("FABRIC_INVALID", list(res["problems"]))
        c._emit(args, {**res, "code": "FABRIC_DERIVE_FAILED"},
                "FABRIC_DERIVE_FAILED: " + str(res.get("error") or "fabric derive failed"))
        return c.EXIT_FAIL
    f = res["fabric"]
    df = f.get("derived_from") or {}
    lines = [f"fabric '{f['name']}': {len(f['masters'])} masters x {len(f['slaves'])} slaves, "
             f"data {f['data_width']} b, outstanding {f['max_outstanding']} (busiest link "
             f"{float(df.get('busiest_link_bytes_per_cycle') or 0):.3f} B/cyc, headroom {df.get('headroom')})"]
    if dry:
        cur = db.fabric_spec(f["name"])
        old = json.dumps(cur["spec"], indent=2, sort_keys=True).splitlines() if cur else ["(none)"]
        new = json.dumps(f, indent=2, sort_keys=True).splitlines()
        label = f"fabric {f['name']} v{cur['version']}" if cur else f"fabric {f['name']} (none)"
        diff = list(difflib.unified_diff(old, new, fromfile=label, tofile="derived", lineterm=""))
        lines.append("dry run, nothing written:")
        lines += diff or ["(no change)"]
        c._emit(args, {**res, "dry_run": True, "diff": "\n".join(diff)}, "\n".join(lines))
        return c.EXIT_PASS
    lines += [f"  master {m['name']}" for m in f["masters"]]
    lines += [f"  slave {s['name']} {s['protocol']} @ {s['base']} +{s['size']}" for s in f["slaves"]]
    lines.append(f"  fabric row v{res.get('version')}{'' if res.get('changed') else ' (unchanged)'}; "
                 f"view {res['path']}")
    c._emit(args, res, "\n".join(lines))
    return c.EXIT_PASS


# ----------------------------------------------------------------- parsers
def register(sub, run, add_project_root, add_json) -> None:
    """Add this module's subcommands to the ``bin/coresmith`` subparser action."""
    top = sub.add_parser("fabric", help="the registered fabric spec: init | master | slave | set | show | "
                                        "params | rm | derive")
    fsub = top.add_subparsers(dest="verb")

    def _leaf(p, handler, name_required=False):
        p.add_argument("--name", required=name_required, help="fabric name (default: the only registered fabric)")
        add_project_root(p)
        add_json(p)
        p.set_defaults(func=run(handler))

    p = fsub.add_parser("init", help="register a new fabric with its first master(s) and slave(s)")
    p.add_argument("--data-width", dest="data_width", default="32")
    p.add_argument("--addr-width", dest="addr_width", default="32")
    p.add_argument("--master", action="append", default=[], metavar="NAME", help="an axi4 master (repeatable)")
    p.add_argument("--slave", action="append", default=[], metavar="NAME:PROTOCOL:BASE:SIZE",
                   help="a slave window, e.g. ram:axi4:0x0:0x2000 (repeatable)")
    p.add_argument("--param", action="append", default=[], metavar="K=V", help="fabric-wide parameter (repeatable)")
    p.add_argument("--replace", action="store_true", help="overwrite an existing fabric of this name")
    _leaf(p, cmd_fabric_init, name_required=True)

    for role in ("master", "slave"):
        rsub = fsub.add_parser(role, help=f"add / remove a {role} port").add_subparsers(dest="port_verb")
        a = rsub.add_parser("add", help=f"add a {role} port")
        a.add_argument("port", metavar="NAME")
        if role == "master":
            a.add_argument("--protocol", default="axi4")
        else:
            a.add_argument("--protocol", required=True, help="axi4 | axi_lite | apb")
            a.add_argument("--base", required=True, help="window base, 0x..")
            a.add_argument("--size", required=True, help="window size in bytes (power of two), 0x..")
        a.add_argument("--param", action="append", default=[], metavar="K=V", help="port parameter (repeatable)")
        _leaf(a, cmd_fabric_master_add if role == "master" else cmd_fabric_slave_add)
        r = rsub.add_parser("rm", help=f"remove a {role} port")
        r.add_argument("port", metavar="NAME")
        _leaf(r, cmd_fabric_master_rm if role == "master" else cmd_fabric_slave_rm)

    s = fsub.add_parser("set", help="set a fabric-wide PARAM, or master|slave.PORT.PARAM")
    s.add_argument("param")
    s.add_argument("value")
    _leaf(s, cmd_fabric_set)

    _leaf(fsub.add_parser("show", help="ports, parameters, version and the rendering primitive block"),
          cmd_fabric_show)

    pp = fsub.add_parser("params", help="the parameter table: [fabric|master|slave] [protocol]")
    pp.add_argument("role", nargs="?", choices=("fabric", "master", "slave"))
    pp.add_argument("protocol", nargs="?")
    add_project_root(pp)
    add_json(pp)
    pp.set_defaults(func=run(cmd_fabric_params))

    _leaf(fsub.add_parser("rm", help="delete a registered fabric"), cmd_fabric_rm, name_required=True)

    d = fsub.add_parser("derive", help="FabricSpec from the arch model's measured link table -> the fabric row")
    d.add_argument("--headroom", type=float, default=2.0)
    d.add_argument("--dry-run", dest="dry_run", action="store_true",
                   help="print a diff against the current row; write nothing")
    _leaf(d, cmd_fabric_derive)
