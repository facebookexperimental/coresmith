# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""The structured ``timing`` object of an interface contract (A2).

Cycle-level interface timing used to live only in prose ("response EXACTLY
1 cycle after request (IFACE-013)"), so each block's testbench modelled its
neighbour from its own reading and the two never met before chip
integration. ``timing`` makes the rule a checkable field:

    "timing": {
      "req_to_rsp_cycles": {"min": 1, "max": 1, "exact": 1} | null,  # req_resp
      "valid_to_ready_max_stall": <int> | null,   # streaming: null = unbounded
      "ordering": "in_order" | "out_of_order_tagged" | "n/a",
      "burst": {"last_signal": "tlast" | null, "max_beats": <int> | null},
      "reset_idle_cycles": <int>,                  # valids low after reset
      "valid_hold_until_ready": <bool>             # AXI rule for streaming
    }

``normalize_timing`` fills family defaults and strips fields a family
cannot have (deterministic, no LLM); ``timing_violations`` reports what the
specialist must supply (``req_resp`` needs an explicit latency). The VIP
generator and the assertion stage read the same object.
"""
from __future__ import annotations

import os
from typing import Any

STREAMING = ("axi_stream", "srdy_drdy", "axi4", "axi_lite", "apb")  # bus families: valid/ready per channel
REQ_RESP = ("req_resp",)
ALWAYS_ACCEPTED = ("mem_write", "valid_only")
STATIC = ("static",)
ORDERINGS = ("in_order", "out_of_order_tagged", "n/a")


def timing_gate_enabled() -> bool:
    """A ``req_resp`` edge without an explicit latency is a structural
    violation (CORESMITH_CONTRACT_TIMING_GATE, default on)."""
    return (os.environ.get("CORESMITH_CONTRACT_TIMING_GATE", "1") or "1").strip().lower() \
        not in {"0", "false", "no", "off", ""}


def _int_or_none(v: Any) -> int | None:
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _last_signal(contract: dict) -> str | None:
    for s in contract.get("sideband_signals") or []:
        name = str((s or {}).get("name", "") if isinstance(s, dict) else s).lower()
        if name.endswith("last"):
            return name
    return None


def default_timing(family: str, contract: dict | None = None) -> dict:
    fam = (family or "").strip().lower()
    t: dict[str, Any] = {
        "req_to_rsp_cycles": None,
        "valid_to_ready_max_stall": None,
        "ordering": "n/a",
        "burst": {"last_signal": None, "max_beats": None},
        "reset_idle_cycles": 1,
        "valid_hold_until_ready": False,
    }
    if fam in STREAMING:
        t["ordering"] = "in_order"
        t["valid_hold_until_ready"] = True
        t["burst"]["last_signal"] = _last_signal(contract or {})
    elif fam in REQ_RESP:
        t["ordering"] = "in_order"
    elif fam in STATIC:
        t["reset_idle_cycles"] = 0
    return t


def normalize_timing(contract: dict) -> tuple[bool, list[str]]:
    """Fill defaults and enforce family shape in place. Returns
    ``(changed, notes)``; never raises on malformed input."""
    fam = str(contract.get("handshake_protocol") or "").strip().lower()
    label = str(contract.get("edge_id") or "?")
    notes: list[str] = []
    raw = contract.get("timing")
    base = default_timing(fam, contract)
    given = dict(raw) if isinstance(raw, dict) else {}
    t = {**base, **{k: v for k, v in given.items() if k in base}}
    # Normalise sub-objects
    r = t.get("req_to_rsp_cycles")
    if isinstance(r, (int, float, str)) and str(r).strip() != "":
        n = _int_or_none(r)
        r = {"min": n, "max": n, "exact": n} if n is not None else None
    if isinstance(r, dict):
        exact = _int_or_none(r.get("exact"))
        mn = _int_or_none(r.get("min"))
        mx = _int_or_none(r.get("max"))
        if exact is not None:
            mn, mx = exact, exact
        if mn is None and mx is not None:
            mn = 0
        r = {"min": mn, "max": mx, "exact": exact}
        if mn is not None and mx is not None and mx < mn:
            notes.append(f"{label}: timing.req_to_rsp_cycles max<min; max raised to min")
            r["max"] = mn
    else:
        r = None
    t["req_to_rsp_cycles"] = r
    b = t.get("burst") if isinstance(t.get("burst"), dict) else {}
    t["burst"] = {"last_signal": (str(b.get("last_signal")) if b.get("last_signal") else
                                  base["burst"]["last_signal"]),
                  "max_beats": _int_or_none(b.get("max_beats"))}
    t["valid_to_ready_max_stall"] = _int_or_none(t.get("valid_to_ready_max_stall"))
    ric = _int_or_none(t.get("reset_idle_cycles"))
    t["reset_idle_cycles"] = base["reset_idle_cycles"] if ric is None else max(0, ric)
    t["valid_hold_until_ready"] = bool(t.get("valid_hold_until_ready"))
    if t.get("ordering") not in ORDERINGS:
        t["ordering"] = base["ordering"]
    # Family shape
    if fam in ALWAYS_ACCEPTED or fam in STATIC:
        if t["valid_to_ready_max_stall"] is not None or t["valid_hold_until_ready"] \
                or t["req_to_rsp_cycles"] is not None:
            notes.append(f"{label}: timing stall/latency fields dropped (family {fam} "
                         "is always-accepted / untimed)")
        t["valid_to_ready_max_stall"] = None
        t["valid_hold_until_ready"] = False
        t["req_to_rsp_cycles"] = None
        if fam in STATIC:
            t["ordering"] = "n/a"
            t["burst"] = {"last_signal": None, "max_beats": None}
    elif fam in STREAMING:
        if t["req_to_rsp_cycles"] is not None:
            notes.append(f"{label}: timing.req_to_rsp_cycles dropped (streaming family)")
            t["req_to_rsp_cycles"] = None
    elif fam in REQ_RESP:
        if t["valid_to_ready_max_stall"] is not None:
            notes.append(f"{label}: timing.valid_to_ready_max_stall dropped (req_resp has no ready)")
            t["valid_to_ready_max_stall"] = None
        t["valid_hold_until_ready"] = False
    changed = (t != raw)
    contract["timing"] = t
    return changed, notes


def timing_violations(contract: dict) -> list[dict]:
    """Structural violations the specialist must fix (empty when the gate is off)."""
    if not timing_gate_enabled():
        return []
    fam = str(contract.get("handshake_protocol") or "").strip().lower()
    cid = str(contract.get("edge_id") or "?")
    out: list[dict] = []
    t = contract.get("timing") if isinstance(contract.get("timing"), dict) else {}
    if fam in REQ_RESP:
        r = t.get("req_to_rsp_cycles")
        if not isinstance(r, dict) or r.get("min") is None:
            out.append({"edge": cid, "type": "missing_timing",
                        "violation": (f"{cid}: req_resp edge has no timing.req_to_rsp_cycles "
                                      "(min/max/exact cycles from request accept to response "
                                      "valid). State the latency both blocks are built to; "
                                      "'bounded' is max, 'exactly N' is exact.")})
    return out


def timing_summary(contract: dict) -> str:
    """One line for prompts and reports."""
    t = contract.get("timing") if isinstance(contract.get("timing"), dict) else None
    if not t:
        return "timing: unspecified"
    parts = []
    r = t.get("req_to_rsp_cycles")
    if isinstance(r, dict) and r.get("min") is not None:
        if r.get("exact") is not None:
            parts.append(f"response exactly {r['exact']} cycle(s) after accept")
        else:
            parts.append(f"response {r['min']}..{r['max'] if r.get('max') is not None else 'inf'} cycles after accept")
    if t.get("valid_to_ready_max_stall") is not None:
        parts.append(f"ready within {t['valid_to_ready_max_stall']} cycle(s) of valid")
    if t.get("valid_hold_until_ready"):
        parts.append("valid holds until ready")
    if t.get("ordering") and t["ordering"] != "n/a":
        parts.append(t["ordering"])
    b = t.get("burst") or {}
    if b.get("last_signal"):
        parts.append(f"bursts end with {b['last_signal']}" + (f" (<= {b['max_beats']} beats)" if b.get("max_beats") else ""))
    parts.append(f"valids idle {t.get('reset_idle_cycles', 1)} cycle(s) after reset")
    return "timing: " + "; ".join(parts)
