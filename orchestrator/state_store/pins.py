# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Chip pins as rows (``pins`` table): the declared chip boundary.

A block port either connects to another block (a contract edge) or leaves
the chip (a pin). The shell assembler used to INFER the boundary from the
ports no edge covers, which a stub-only shell cannot do (stubs carry only
contract ports), so a contract-only design elaborated with zero boundary
ports. A pin names the top-level port, its direction and width, and the
block port it exposes (``--from <block>.<port>``); ``kind`` clock/reset pins
without a block are the shell's clk/rst nets, fanned to every block.

Rules (validated by :func:`pin_problems`, refused with stable codes):

    PIN_EXISTS          add of a pin name that exists (use ``pin set``)
    PIN_LOCKED          the pin (or, for add, the pin set) is locked; repeat
                        with ``--unlock --reason`` -- the pin stays locked
    PIN_UNKNOWN_BLOCK   ``--from`` names a block that is not registered
    PIN_UNKNOWN_PORT    the block has ``interfaces`` rows and none is the port
    PIN_DOUBLE_DRIVEN   the block port is on a contract edge, or another pin
                        already exposes it
    PIN_BAD_*           malformed name / dir / kind / width / bus mapping
    PIN_NO_SOURCE       a ``signal`` pin without ``--from``

Pins lock together with the contract edges when ``interfaces`` completes
(``CORESMITH_CONTRACT_AUTOLOCK``). Langgraph-free.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any

PIN_DIRS = ("in", "out", "inout")
PIN_KINDS = ("signal", "clock", "reset", "power")
# pin set fields: the editable columns (``from`` sets block and port together)
PIN_FIELDS = ("dir", "width", "block", "port", "from", "bus", "msb", "lsb", "oe", "kind")
PINS_SCHEMA = """
CREATE TABLE IF NOT EXISTS pins (
    name TEXT PRIMARY KEY,           -- chip-level signal name (becomes the top port)
    dir TEXT NOT NULL,               -- in | out | inout
    width INTEGER NOT NULL DEFAULT 1,
    block TEXT, port TEXT,           -- the block port this pin exposes (NULL for clk/rst tied by the shell)
    bus TEXT, msb INTEGER, lsb INTEGER, oe TEXT,   -- Caravel/pad-bus mapping (optional)
    kind TEXT NOT NULL DEFAULT 'signal',           -- signal | clock | reset | power
    version INTEGER NOT NULL DEFAULT 1,
    locked INTEGER NOT NULL DEFAULT 0,
    ts REAL NOT NULL
);
"""
PINS_VIEW = "pins.json"
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")
_COLS = ("name", "dir", "width", "block", "port", "bus", "msb", "lsb", "oe", "kind", "version", "locked", "ts")


class PinError(ValueError):
    """A pin write refused with a stable ``code`` (nothing was written)."""

    def __init__(self, code: str, message: str, problems: list[dict] | None = None):
        self.code = code
        self.problems = problems or [{"code": code, "text": message, "severity": "error"}]
        super().__init__(f"{code}: {message}")


def _problem(code: str, where: str, text: str, severity: str = "error") -> dict:
    return {"code": code, "where": where, "text": text, "severity": severity}


def _base_block(name: str) -> str:
    return str(name or "").split("[", 1)[0]


def _none(v: Any) -> Any:
    if isinstance(v, str) and v.strip().lower() in ("", "none", "null", "-"):
        return None
    return v


def normalize_pin(pin: dict) -> dict:
    """Coerce a pin dict's types (width/msb/lsb ints, empty strings -> None)."""
    out = {k: _none(pin.get(k)) for k in ("name", "dir", "block", "port", "bus", "oe", "kind")}
    out["name"] = str(out["name"] or "").strip()
    out["dir"] = str(out["dir"] or "").strip().lower()
    out["kind"] = str(out["kind"] or "signal").strip().lower()
    for k in ("width", "msb", "lsb"):
        v = _none(pin.get(k))
        try:
            out[k] = int(v) if v is not None else None
        except (TypeError, ValueError):
            out[k] = v   # reported by pin_problems
    if out["width"] is None:
        out["width"] = 1
    return out


def pin_problems(db, pin: dict, *, replacing: str | None = None) -> list[dict]:
    """Every error of writing ``pin`` (a normalized dict) into the current
    project state. ``replacing``: the name of the pin being edited (its own
    row does not count as a conflict)."""
    name = pin["name"]
    out: list[dict] = []
    if not _IDENT.match(name or ""):
        out.append(_problem("PIN_BAD_NAME", name or "?", "a pin name must be a Verilog identifier"))
    if pin["dir"] not in PIN_DIRS:
        out.append(_problem("PIN_BAD_DIR", name, f"dir must be one of {', '.join(PIN_DIRS)}, got {pin['dir']!r}"))
    if pin["kind"] not in PIN_KINDS:
        out.append(_problem("PIN_BAD_KIND", name, f"kind must be one of {', '.join(PIN_KINDS)}, got {pin['kind']!r}"))
    if not isinstance(pin["width"], int) or pin["width"] < 1:
        out.append(_problem("PIN_BAD_WIDTH", name, f"width must be an integer >= 1, got {pin['width']!r}"))
    if pin.get("bus"):
        msb, lsb = pin.get("msb"), pin.get("lsb")
        if not isinstance(msb, int) or not isinstance(lsb, int) or msb < 0 or lsb < 0:
            out.append(_problem("PIN_BAD_BUS", name, "--bus needs integer --msb and --lsb"))
        elif isinstance(pin["width"], int) and abs(msb - lsb) + 1 != pin["width"]:
            out.append(_problem("PIN_BAD_BUS", name, f"bus bits [{msb}:{lsb}] are {abs(msb - lsb) + 1} wide, "
                                                     f"the pin is {pin['width']}"))
    elif pin.get("msb") is not None or pin.get("lsb") is not None or pin.get("oe"):
        out.append(_problem("PIN_BAD_BUS", name, "--msb/--lsb/--oe map the pin onto a pad bus: give --bus too"))
    if pin.get("oe") and pin["dir"] == "in":
        out.append(_problem("PIN_BAD_BUS", name, "--oe drives an output enable; an 'in' pin has none"))
    block, port = pin.get("block"), pin.get("port")
    if bool(block) != bool(port):
        out.append(_problem("PIN_BAD_FROM", name, "--from needs <block>.<port>"))
    elif block:
        names = set(db.block_names())
        if _base_block(block) not in names:
            out.append(_problem("PIN_UNKNOWN_BLOCK", name, f"{block} is not a registered block (coresmith blocks)"
                                                           + (f"; blocks: {', '.join(sorted(names))}" if names else "")))
        else:
            ifaces = db.interface_names(_base_block(block))
            if ifaces and port not in ifaces:
                out.append(_problem("PIN_UNKNOWN_PORT", name, f"{block} has no port {port!r} "
                                                              f"(its ports: {', '.join(ifaces)})"))
            for e in db.contract_rows():
                s = e["spec"]
                if (s.get("producer_block"), s.get("producer_port")) == (block, port) \
                        or (s.get("consumer_block"), s.get("consumer_port")) == (block, port):
                    out.append(_problem("PIN_DOUBLE_DRIVEN", name, f"{block}.{port} is on contract edge "
                                                                   f"{e['edge_id']}; a port is an edge OR a pin"))
                    break
            other = next((p for p in db.pins() if p["name"] not in (name, replacing)
                          and (p.get("block"), p.get("port")) == (block, port)), None)
            if other:
                out.append(_problem("PIN_DOUBLE_DRIVEN", name, f"{block}.{port} is already pin {other['name']}"))
    elif pin["kind"] == "signal":
        out.append(_problem("PIN_NO_SOURCE", name, "a signal pin exposes a block port: --from <block>.<port> "
                                                   "(clock/reset pins without --from are the shell's clk/rst)"))
    return out


class PinMixin:
    """``pins`` rows on :class:`~orchestrator.state_store.project_db.ProjectDB`."""

    @staticmethod
    def _pin_row(r) -> dict:
        d = {k: r[k] for k in _COLS}
        d["locked"] = bool(d["locked"])
        return d

    def pins(self) -> list[dict]:
        with self._conn() as db:
            try:
                rows = db.execute("SELECT * FROM pins ORDER BY rowid").fetchall()
            except Exception:  # noqa: BLE001 - a database older than the table
                return []
        return [self._pin_row(r) for r in rows]

    def pin(self, name: str) -> dict | None:
        return next((p for p in self.pins() if p["name"] == name), None)

    def interface_names(self, block: str) -> list[str]:
        """The block's port names from the ``interfaces`` rows (diagram order)."""
        with self._conn() as db:
            rows = db.execute("SELECT name, spec_json FROM interfaces WHERE block=? ORDER BY ordinal",
                              (block,)).fetchall()
        out = []
        for r in rows:
            n = r["name"]
            if not n:
                try:
                    spec = json.loads(r["spec_json"] or "null")
                except ValueError:
                    spec = None
                n = spec.get("name") if isinstance(spec, dict) else (spec if isinstance(spec, str) else None)
            if n and n not in out:
                out.append(str(n))
        return out

    def _pins_locked(self) -> bool:
        return any(p["locked"] for p in self.pins())

    def _pin_write(self, pin: dict, *, version: int, locked: bool) -> None:
        with self._tx() as db:
            db.execute(
                "INSERT OR REPLACE INTO pins(name, dir, width, block, port, bus, msb, lsb, oe, kind, version, locked, ts) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (pin["name"], pin["dir"], int(pin["width"]), pin.get("block"), pin.get("port"), pin.get("bus"),
                 pin.get("msb"), pin.get("lsb"), pin.get("oe"), pin["kind"], int(version), int(locked), time.time()))

    def pin_add(self, pin: dict, *, unlock: bool = False) -> dict:
        """Add one pin. Raises :class:`PinError` (nothing written) on
        ``PIN_EXISTS``, ``PIN_LOCKED`` (the pin set is locked and not
        ``unlock``) or any :func:`pin_problems` error. A pin added to a
        locked set is locked (the unlock licenses this write only)."""
        pin = normalize_pin(pin)
        if self.pin(pin["name"]) is not None:
            raise PinError("PIN_EXISTS", f"pin {pin['name']} exists (coresmith pin set {pin['name']} <field> <value>)")
        set_locked = self._pins_locked()
        if set_locked and not unlock:
            raise PinError("PIN_LOCKED", "the pin set is locked; repeat with --unlock --reason TEXT")
        probs = pin_problems(self, pin)
        if probs:
            raise PinError(probs[0]["code"], probs[0]["text"], probs)
        self._pin_write(pin, version=1, locked=set_locked)
        self.ensure_db_artifact("pins", registered_by="cli")
        self.export_views()
        return {**self.pin(pin["name"]), "changed": True}

    def pin_set(self, name: str, field: str, value: Any, *, unlock: bool = False) -> dict:
        """Set one field (``PIN_FIELDS``; ``from`` = ``<block>.<port>``). The
        version bumps on a change; a locked pin refuses unless ``unlock`` and
        stays locked."""
        cur = self.pin(name)
        if cur is None:
            raise PinError("PIN_NOT_FOUND", f"no pin {name!r} (coresmith pin list)")
        field = str(field).strip().lower()
        if field not in PIN_FIELDS:
            raise PinError("PIN_BAD_FIELD", f"field must be one of {', '.join(PIN_FIELDS)}, got {field!r}")
        new = {k: cur[k] for k in ("name", "dir", "width", "block", "port", "bus", "msb", "lsb", "oe", "kind")}
        if field == "from":
            v = _none(value)
            if v is None:
                new["block"] = new["port"] = None
            else:
                b, sep, p = str(v).partition(".")
                if not sep or not b or not p:
                    raise PinError("PIN_BAD_FROM", f"from must be <block>.<port>, got {value!r}")
                new["block"], new["port"] = b, p
        else:
            new[field] = value
        new = normalize_pin(new)
        changed = any(new[k] != cur[k] for k in new)
        if not changed:
            return {**cur, "changed": False}
        if cur["locked"] and not unlock:
            raise PinError("PIN_LOCKED", f"pin {name} is locked; repeat with --unlock --reason TEXT")
        probs = pin_problems(self, new, replacing=name)
        if probs:
            raise PinError(probs[0]["code"], probs[0]["text"], probs)
        self._pin_write(new, version=int(cur["version"]) + 1, locked=bool(cur["locked"]))
        self.ensure_db_artifact("pins", registered_by="cli")
        self.export_views()
        return {**self.pin(name), "changed": True}

    def pin_rm(self, name: str, *, unlock: bool = False) -> bool:
        cur = self.pin(name)
        if cur is None:
            raise PinError("PIN_NOT_FOUND", f"no pin {name!r} (coresmith pin list)")
        if cur["locked"] and not unlock:
            raise PinError("PIN_LOCKED", f"pin {name} is locked; repeat with --unlock --reason TEXT")
        with self._tx() as db:
            db.execute("DELETE FROM pins WHERE name=?", (name,))
        self.export_views()
        return True

    def lock_pins(self, locked: bool = True) -> int:
        """Lock (or unlock) every pin; returns the row count."""
        with self._tx() as db:
            try:
                return db.execute("UPDATE pins SET locked=?", (int(locked),)).rowcount
            except Exception:  # noqa: BLE001 - a database older than the table
                return 0
