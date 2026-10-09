# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""``coresmith pin ...``: the chip pins as line items (the ``pins`` table).

    pin add <name> --dir in|out|inout [--width N] [--from <block>.<port>]
                   [--kind signal|clock|reset|power] [--bus io --msb N --lsb N [--oe <sig>]]
                   [--unlock --reason TEXT]
    pin set <name> <field> <value> [--unlock --reason TEXT]   # dir width from block port bus msb lsb oe kind
    pin rm <name> [--unlock --reason TEXT]
    pin lock | pin unlock --reason TEXT
    pin list [--json]

A block port either connects to another block (a contract edge) or leaves
the chip (a pin). ``shell assemble`` makes the pin set the top's boundary;
a port on no edge and no pin is refused there (``SHELL_UNDECLARED_PORT``).
Refusals (nothing written, exit 1): ``PIN_EXISTS``, ``PIN_LOCKED``,
``PIN_UNKNOWN_BLOCK``, ``PIN_UNKNOWN_PORT``, ``PIN_DOUBLE_DRIVEN``,
``PIN_NO_SOURCE``, ``PIN_BAD_*`` (see ``state_store/pins.py``). Pins lock with
the contract edges when ``interfaces`` completes; ``--unlock --reason``
licenses one write and the pin stays locked (``pin unlock`` is the only way
to leave the set open). ``state write`` renders ``arch/pinout.md`` (and the
Caravel ``prd["pin_map"]`` when a pin carries ``--bus``).

Wired from ``orchestrator.harness.cli.register_subcommands`` via ``register``.
Must import without importing ``orchestrator.langgraph``.
"""
from __future__ import annotations

import sys

_FAIL, _USAGE = 1, 2


def _cli():
    from orchestrator.harness import cli
    return cli


def _unlock_ok(args) -> bool:
    if getattr(args, "unlock", False) and not (getattr(args, "reason", "") or "").strip():
        print("--unlock requires --reason TEXT (it is recorded in the actions log)", file=sys.stderr)
        return False
    return True


def _refuse(args, verb: str, name: str, exc) -> int:
    probs = [{"where": p.get("where", name), **p} for p in getattr(exc, "problems", [])] or \
        [{"code": getattr(exc, "code", "PIN_ERROR"), "where": name, "text": str(exc), "severity": "error"}]
    codes = sorted({p["code"] for p in probs})
    _cli()._emit(args, {"ok": False, "pin": name, "problems": probs},
                 "\n".join([f"pin {verb} {name}: REFUSED ({', '.join(codes)})",
                            *[f"  [erro] {p['code']} {p['where']}: {p['text']}" for p in probs]]))
    return _FAIL


def _line(p: dict) -> str:
    frm = f"{p['block']}.{p['port']}" if p.get("block") else "-"
    w = f"[{p['width'] - 1}:0]" if int(p.get("width") or 1) > 1 else ""
    bus = f"  {p['bus']}[{p['msb']}:{p['lsb']}]" + (f" oe={p['oe']}" if p.get("oe") else "") if p.get("bus") else ""
    return (f"{p['name']}{w}  {p['dir']:<5} {p['kind']:<6} <- {frm}{bus}  v{p['version']}"
            f"  {'LOCKED' if p['locked'] else 'open'}")


def _ok(args, verb: str, name: str, res: dict) -> int:
    head = f"pin {verb} {name}: OK" + (f" v{res['version']}" if "version" in res else "") \
        + (" (unchanged)" if res.get("changed") is False else "")
    if getattr(args, "unlock", False):
        head += f" [unlocked for this write: {args.reason.strip()}]"
        res["reason"] = args.reason.strip()
    _cli()._emit(args, {"ok": True, **res}, head)
    return 0


def _from(s: str | None) -> tuple[str | None, str | None]:
    if not s:
        return None, None
    b, sep, p = str(s).partition(".")
    if not sep or not b or not p:
        raise ValueError(f"--from must be <block>.<port>, got {s!r}")
    return b, p


def cmd_pin_add(args) -> int:
    from orchestrator.state_store.pins import PinError
    if not _unlock_ok(args):
        return _USAGE
    try:
        block, port = _from(args.from_)
    except ValueError as exc:
        print(f"pin add: {exc}", file=sys.stderr)
        return _USAGE
    kind = args.kind or "signal"
    pin = {"name": args.name, "dir": args.dir, "width": args.width, "block": block, "port": port,
           "bus": args.bus, "msb": args.msb, "lsb": args.lsb, "oe": args.oe, "kind": kind}
    db = _cli()._state_db(args)
    try:
        res = db.pin_add(pin, unlock=bool(args.unlock))
    except PinError as exc:
        return _refuse(args, "add", args.name, exc)
    return _ok(args, "add", args.name, res)


def cmd_pin_set(args) -> int:
    from orchestrator.state_store.pins import PinError
    if not _unlock_ok(args):
        return _USAGE
    db = _cli()._state_db(args)
    try:
        res = db.pin_set(args.name, args.field, args.value, unlock=bool(args.unlock))
    except PinError as exc:
        if exc.code in ("PIN_NOT_FOUND", "PIN_BAD_FIELD"):
            print(f"pin set: {exc}", file=sys.stderr)
            return _USAGE
        return _refuse(args, "set", args.name, exc)
    return _ok(args, "set", args.name, res)


def cmd_pin_rm(args) -> int:
    from orchestrator.state_store.pins import PinError
    if not _unlock_ok(args):
        return _USAGE
    db = _cli()._state_db(args)
    try:
        db.pin_rm(args.name, unlock=bool(args.unlock))
    except PinError as exc:
        if exc.code == "PIN_NOT_FOUND":
            print(f"pin rm: {exc}", file=sys.stderr)
            return _USAGE
        return _refuse(args, "rm", args.name, exc)
    return _ok(args, "rm", args.name, {"name": args.name, "removed": True})


def cmd_pin_lock(args) -> int:
    cli = _cli()
    unlock = args.verb == "unlock"
    reason = (getattr(args, "reason", "") or "").strip()
    if unlock and not reason:
        print("pin unlock requires --reason TEXT", file=sys.stderr)
        return _USAGE
    n = cli._state_db(args).lock_pins(locked=not unlock)
    cli._emit(args, {"ok": True, "verb": args.verb, "pins": n, "reason": reason},
              f"pin {args.verb}: {n} pin(s)" + (f" [reason: {reason}]" if unlock else ""))
    return 0


def cmd_pin_list(args) -> int:
    cli = _cli()
    rows = cli._state_db(args).pins()
    cli._emit(args, {"pins": rows}, "\n".join(_line(p) for p in rows)
              or "no pins (coresmith pin add <name> --dir in|out|inout --from <block>.<port>)")
    return 0


def _unlock_args(p) -> None:
    p.add_argument("--unlock", action="store_true", help="allow changing a locked pin (needs --reason)")
    p.add_argument("--reason", default="", help="why the locked pin changes (recorded in the actions log)")


def register(sub, run, add_project_root, add_json) -> None:
    """Add ``coresmith pin ...`` to the ``bin/coresmith`` subparser action."""
    pp = sub.add_parser("pin", help="chip pins (the declared top boundary): add | set | rm | lock | unlock | list")
    pp.set_defaults(func=run(cmd_pin_list), verb="list")
    psub = pp.add_subparsers(dest="verb")

    def _p(name, handler, help_):
        p = psub.add_parser(name, help=help_)
        add_project_root(p)
        add_json(p)
        p.set_defaults(func=run(handler), verb=name)
        return p

    a = _p("add", cmd_pin_add, "declare one chip pin (validated before it is written)")
    a.add_argument("name", help="the top-level port name")
    a.add_argument("--dir", required=True, choices=("in", "out", "inout"))
    a.add_argument("--width", type=int, default=1)
    a.add_argument("--from", dest="from_", default=None, metavar="BLOCK.PORT",
                   help="the block port the pin exposes (required for signal pins)")
    a.add_argument("--kind", default=None, choices=("signal", "clock", "reset", "power"),
                   help="default signal; clock/reset without --from are the shell's clk/rst nets")
    a.add_argument("--bus", default=None, help="pad bus the pin maps onto (Caravel: io)")
    a.add_argument("--msb", type=int, default=None)
    a.add_argument("--lsb", type=int, default=None)
    a.add_argument("--oe", default=None, help="active-high output-enable signal (out/inout on a pad bus)")
    _unlock_args(a)
    s = _p("set", cmd_pin_set, "set one field: dir width from block port bus msb lsb oe kind")
    s.add_argument("name")
    s.add_argument("field")
    s.add_argument("value", help="'none' clears an optional field; from takes <block>.<port>")
    _unlock_args(s)
    r = _p("rm", cmd_pin_rm, "remove one pin")
    r.add_argument("name")
    _unlock_args(r)
    _p("lock", cmd_pin_lock, "lock every pin")
    ul = _p("unlock", cmd_pin_lock, "unlock every pin")
    ul.add_argument("--reason", required=True)
    _p("list", cmd_pin_list, "the pinout: one line per pin")
