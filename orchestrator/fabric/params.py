# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""The fabric parameter system: which knobs each fabric / port kind takes,
their types, defaults and legal ranges, and string -> value coercion for the
``coresmith fabric`` CLI verbs.

Names and defaults come from the FabricSpec / FabricMaster / FabricSlave
dataclass fields (so a new field with a default shows up here with its
default); types, choices, ranges and help text are annotated by hand below.
The authoritative whole-spec check stays ``FabricSpec.validate()``.
"""
from __future__ import annotations

import dataclasses

from orchestrator.fabric.spec import (
    LATENCY_MODES,
    MASTER_PROTOCOLS,
    SLAVE_PROTOCOLS,
    FabricMaster,
    FabricSlave,
    FabricSpec,
)

HEX = "hex"  # type tag: an int written/parsed as 0x...

# Hand annotations: param -> (type, choices, min, max, help).
_FABRIC = {
    "data_width": (int, [32, 64, 128], None, None, "crossbar data width in bits (every slave port shares it)"),
    "addr_width": (int, [32, 40, 48, 56, 64], None, None, "address width in bits"),
    "user_width": (int, None, 1, 64, "AXI user-signal width"),
    "ordering": (str, ["per_id", "none"], None, None,
                 "per_id: AXI per-ID ordering; none: an axi_cut per axi4 slave port"),
    "err_slave": (bool, None, None, None, "unmapped addresses answer DECERR"),
    "max_outstanding": (int, None, 1, 64,
                        "axi_xbar MaxMstTrans and default converter AxiMax{Write,Read}Txns"),
    "latency_mode": (str, list(LATENCY_MODES), None, None, "axi_xbar LatencyMode"),
    "slave_cut": (bool, None, None, None, "an axi_cut on every slave port (both converter sides)"),
    "pipeline_stages": (int, None, 0, 4, "axi_xbar Cfg.PipelineStages (multicut stages inside the crossbar)"),
    "unique_ids": (bool, None, None, None, "axi_xbar Cfg.UniqueIds (masters keep one slave per in-flight ID)"),
    "shared_apb_bridge": (bool, None, None, None, "route every APB slave through one shared AXI->APB bridge"),
}
_MASTER_AXI4 = {
    "id_width": (int, None, 1, 16, "AXI ID width (all masters share one)"),
    "max_outstanding": (int, None, 1, 64, "transactions per direction the fabric tracks for this master"),
}
_SLAVE_WINDOW = {
    "base": (HEX, None, 0, None, "window base address (aligned to size)"),
    "size": (HEX, None, 1, None, "window size in bytes (power of two)"),
}
_SLAVE_AXI4 = {
    **_SLAVE_WINDOW,
    "data_width": (int, [32, 64, 128], None, None,
                   "port data width (must equal the fabric's; width conversion is not generated yet)"),
    "max_outstanding": (int, None, 1, 64,
                        "axi_lite/apb converter ports only (axi4 ports use the fabric's max_outstanding)"),
}


def _defaults(cls) -> dict:
    out = {}
    for f in dataclasses.fields(cls):
        if f.default is not dataclasses.MISSING:
            out[f.name] = f.default
        elif f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
            out[f.name] = f.default_factory()  # type: ignore[misc]
    return out


def _leaf(cls, annotations: dict) -> dict:
    dflt = _defaults(cls)
    missing = set(annotations) - set(dflt)
    if missing:
        raise RuntimeError(f"fabric params annotate unknown {cls.__name__} fields: {sorted(missing)}")
    return {k: {"type": t, "default": dflt[k], "choices": ch, "min": lo, "max": hi, "help": h}
            for k, (t, ch, lo, hi, h) in annotations.items()}


PARAM_SCHEMA: dict = {
    "fabric": _leaf(FabricSpec, _FABRIC),
    "master": {"axi4": _leaf(FabricMaster, _MASTER_AXI4)},
    "slave": {
        "axi4": _leaf(FabricSlave, {k: v for k, v in _SLAVE_AXI4.items() if k != "max_outstanding"}),
        "axi_lite": _leaf(FabricSlave, {**_SLAVE_WINDOW, "max_outstanding": _SLAVE_AXI4["max_outstanding"]}),
        "apb": _leaf(FabricSlave, {**_SLAVE_WINDOW, "max_outstanding": _SLAVE_AXI4["max_outstanding"]}),
    },
}
assert set(PARAM_SCHEMA["master"]) == set(MASTER_PROTOCOLS)
assert set(PARAM_SCHEMA["slave"]) == set(SLAVE_PROTOCOLS)

ROLES = ("fabric", "master", "slave")


def allowed(role: str, protocol: str | None = None) -> dict:
    """The parameter table for ``role`` (fabric | master | slave) and, for
    ports, ``protocol``. Raises ValueError on an unknown role/protocol."""
    if role == "fabric":
        return PARAM_SCHEMA["fabric"]
    if role not in ("master", "slave"):
        raise ValueError(f"unknown role {role!r}; one of {ROLES}")
    table = PARAM_SCHEMA[role]
    if protocol not in table:
        raise ValueError(f"unknown {role} protocol {protocol!r}; one of {tuple(table)}")
    return table[protocol]


def _where(role: str, protocol: str | None) -> str:
    return "fabric" if role == "fabric" else f"{role}/{protocol}"


def coerce(role: str, protocol: str | None, param: str, raw):
    """Parse ``raw`` (a CLI string) into the typed value of ``param``, checking
    choices and range. Raises ValueError with an actionable message."""
    table = allowed(role, protocol)
    if param not in table:
        raise ValueError(f"unknown parameter {param} for {_where(role, protocol)}; "
                         f"allowed: {', '.join(table)}")
    ent = table[param]
    t = ent["type"]
    s = str(raw).strip()
    try:
        if t is bool:
            low = s.lower()
            if low in ("true", "1", "yes", "on"):
                val = True
            elif low in ("false", "0", "no", "off"):
                val = False
            else:
                raise ValueError(s)
        elif t == HEX:
            val = int(s, 16) if s.lower().startswith("0x") else int(s, 10)
        elif t is int:
            val = int(s, 0)
        else:
            val = s
    except ValueError:
        kind = {bool: "true/false/1/0", HEX: "an integer (0x.. or decimal)", int: "an integer"}.get(t, "a string")
        raise ValueError(f"{_where(role, protocol)}.{param}: {raw!r} is not {kind}") from None
    if ent["choices"] is not None and val not in ent["choices"]:
        raise ValueError(f"{_where(role, protocol)}.{param}: {val!r} not in {ent['choices']}")
    if ent["min"] is not None and val < ent["min"]:
        raise ValueError(f"{_where(role, protocol)}.{param}: {val} below minimum {ent['min']}")
    if ent["max"] is not None and val > ent["max"]:
        raise ValueError(f"{_where(role, protocol)}.{param}: {val} above maximum {ent['max']}")
    if param == "size" and t == HEX and (val <= 0 or val & (val - 1)):
        raise ValueError(f"{_where(role, protocol)}.size: {val:#x} must be a power of two")
    return val


def _fmt_default(ent: dict) -> str:
    d = ent["default"]
    if ent["type"] == HEX and isinstance(d, int):
        return f"{d:#x}"
    return "-" if d is None else str(d).lower() if isinstance(d, bool) else str(d)


def _fmt_range(ent: dict) -> str:
    if ent["choices"] is not None:
        return "|".join(str(c) for c in ent["choices"])
    if ent["type"] is bool:
        return "true|false"
    lo, hi = ent["min"], ent["max"]
    if lo is None and hi is None:
        return ""
    return f"{'' if lo is None else lo}..{'' if hi is None else hi}"


def _type_name(t) -> str:
    return t if isinstance(t, str) else t.__name__


def rows(role: str | None = None, protocol: str | None = None) -> list[dict]:
    """Flat parameter rows (role, protocol, param, type, default, range, help)."""
    out = []
    roles = [role] if role else list(ROLES)
    for r in roles:
        if r == "fabric":
            groups = [(None, allowed("fabric"))]
        else:
            protos = [protocol] if protocol else list(PARAM_SCHEMA.get(r, {}))
            groups = [(p, allowed(r, p)) for p in protos]
        for p, table in groups:
            for k, ent in table.items():
                out.append({"role": r, "protocol": p or "", "param": k, "type": _type_name(ent["type"]),
                            "default": _fmt_default(ent), "range": _fmt_range(ent), "help": ent["help"]})
    return out


def describe(role: str | None = None, protocol: str | None = None) -> str:
    """A text table of the parameters (``coresmith fabric params``)."""
    rs = rows(role, protocol)
    head = ("role", "protocol", "param", "type", "default", "range", "help")
    cols = [[h] + [str(r[h]) for r in rs] for h in head]
    widths = [max(len(c) for c in col) for col in cols[:-1]]
    lines = []
    for i in range(len(rs) + 1):
        cells = [cols[j][i].ljust(widths[j]) for j in range(len(widths))] + [cols[-1][i]]
        lines.append("  ".join(cells).rstrip())
    return "\n".join(lines)
