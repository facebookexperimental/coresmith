# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Fabric Resolution (B1): make the SoC bus a generated primitive.

Deterministic pass over an accepted block diagram:

* a block declared ``kind: primitive`` with a ``fabric`` spec is validated,
  normalised (tier 0, rtl_target/testbench paths, golden exemption) and its
  edges are typed ``axi4`` / ``axi_lite`` / ``apb`` with the fabric-side
  prefixes ``s_<master>`` / ``m_<slave>``;
* a block that only *looks* like interconnect (arbiter / crossbar /
  interconnect / bus bridge / address decoder in its name or description) or
  a target with >= 2 initiators over memory-style edges is reported as a
  candidate: the node proposes a FabricSpec skeleton (masters and slaves
  inferred from the edges, addresses to be filled) and asks
  (``fabric_ambiguous``) instead of guessing an address map.

Nothing here calls an LLM.
"""
from __future__ import annotations

import re
from typing import Any

from orchestrator.fabric.spec import FabricSpec

_INTERCONNECT_TOKENS = {"arbiter", "crossbar", "xbar", "interconnect", "fabric", "busmatrix",
                        "decoder", "bridge"}
_INTERCONNECT_PAIRS = {("bus", "matrix"), ("bus", "bridge"), ("bus", "fabric"), ("address", "decoder"),
                       ("apb", "decoder"), ("apb", "bridge"), ("axi", "mux"), ("axi", "demux"),
                       ("axi", "bridge"), ("axi", "interconnect"), ("axi", "arbiter")}


def smells_like_interconnect(text: str) -> bool:
    """Token match over identifiers AND prose (``axi_arbiter`` counts: an
    underscore is not a word boundary for ``\\b``)."""
    toks = [t for t in re.split(r"[^a-z0-9]+", (text or "").lower()) if t]
    if any(t in _INTERCONNECT_TOKENS for t in toks):
        return True
    return any((a, b) in _INTERCONNECT_PAIRS for a, b in zip(toks, toks[1:]))
MEMORY_STYLE = {"req_resp", "mem_write", "axi4", "axi_lite", "apb"}


def _edges(doc: dict) -> list[dict]:
    return [c for c in (doc.get("connections") or []) if isinstance(c, dict)]


def _blocks(doc: dict) -> list[dict]:
    return [b for b in (doc.get("blocks") or []) if isinstance(b, dict) and b.get("name")]


def declared_fabrics(doc: dict) -> list[dict]:
    return [b for b in _blocks(doc) if str(b.get("kind") or "").lower() == "primitive"
            and isinstance(b.get("fabric"), dict)]


def candidate_blocks(doc: dict) -> list[dict]:
    """Interconnect-shaped blocks without a fabric spec."""
    out = []
    for b in _blocks(doc):
        if str(b.get("kind") or "").lower() == "primitive":
            continue
        if smells_like_interconnect(f"{b.get('name', '')} {b.get('description', '')}"):
            out.append(b)
    return out


def shared_targets(doc: dict) -> dict[str, list[str]]:
    """target -> initiators for targets reached by >= 2 blocks over memory-style edges."""
    by: dict[str, set[str]] = {}
    for c in _edges(doc):
        fam = str(c.get("handshake_protocol") or "").lower()
        if fam in MEMORY_STYLE and c.get("from") and c.get("to"):
            by.setdefault(str(c["to"]), set()).add(str(c["from"]))
    return {t: sorted(s) for t, s in by.items() if len(s) >= 2}


def propose_spec(doc: dict, block: dict, data_width: int = 32) -> dict:
    """A FabricSpec skeleton for an interconnect-shaped block: masters = the
    blocks with memory-style edges INTO it, slaves = the blocks it drives."""
    name = str(block["name"])
    masters, slaves = [], []
    for c in _edges(doc):
        fam = str(c.get("handshake_protocol") or "").lower()
        if c.get("to") == name and c.get("from") and fam not in ("static", "valid_only"):
            masters.append({"name": str(c["from"]), "protocol": "axi4", "id_width": 4, "max_outstanding": 4})
        if c.get("from") == name and c.get("to") and fam not in ("static", "valid_only"):
            proto = "apb" if fam == "apb" else "axi_lite" if fam in ("axi_lite", "req_resp") else "axi4"
            slaves.append({"name": str(c["to"]), "protocol": proto, "base": None, "size": None})
    masters = sorted({m["name"]: m for m in masters}.values(), key=lambda m: m["name"])
    slaves = sorted({s["name"]: s for s in slaves}.values(), key=lambda s: s["name"])
    return {"name": re.sub(r"[^a-z0-9_]", "_", name.lower()), "data_width": data_width,
            "addr_width": 32, "masters": masters, "slaves": slaves, "ordering": "per_id",
            "err_slave": True}


def _normalize_fabric_block(block: dict, spec: FabricSpec) -> dict:
    b = dict(block)
    b["kind"] = "primitive"
    b["primitive"] = "cs_fabric"
    b["tier"] = 0
    b.setdefault("subsystem", "interconnect")
    b["fabric"] = spec.to_json()
    sub = b.get("subsystem") or "interconnect"
    b["rtl_target"] = b.get("rtl_target") or f"rtl/{sub}/{spec.module_name}.v"
    b["testbench"] = b.get("testbench") or f"tb/cocotb/test_{spec.module_name}.py"
    b["golden_exempt"] = True
    b.setdefault("no_golden_reason",
                 "generated fabric primitive over pulp-platform axi; verified by its generated testbench")
    b.setdefault("python_source", "")
    b["description"] = b.get("description") or (
        f"SoC fabric: {len(spec.masters)} master(s) x {len(spec.slaves)} slave(s)")
    return b


def _retype_edges(doc: dict, block: dict, spec: FabricSpec) -> tuple[list[dict], list[str]]:
    """Edges into/out of the fabric get bus families and fabric-side prefixes."""
    name = str(block["name"])
    masters = {m.name for m in spec.masters}
    slaves = {s.name: s for s in spec.slaves}
    out, notes = [], []
    for c in _edges(doc):
        c = dict(c)
        if c.get("to") == name:
            src = str(c.get("from") or "")
            if src in masters:
                c["handshake_protocol"] = "axi4"
                c["to_port"] = f"s_{src}"
                c.setdefault("from_port", "m_axi")
                c["bus_name"] = spec.name
                c["data_width"] = spec.data_width
            else:
                notes.append(f"edge {src}->{name}: {src!r} is not a declared fabric master")
        elif c.get("from") == name:
            dst = str(c.get("to") or "")
            if dst in slaves:
                c["handshake_protocol"] = slaves[dst].protocol
                c["from_port"] = f"m_{dst}"
                c.setdefault("to_port", "s_apb" if slaves[dst].protocol == "apb" else "s_axi")
                c["bus_name"] = spec.name
                c["data_width"] = spec.data_width
            else:
                notes.append(f"edge {name}->{dst}: {dst!r} is not a declared fabric slave")
        out.append(c)
    return out, notes


def resolve(doc: dict) -> dict[str, Any]:
    """Resolve fabrics in a block diagram.

    Returns ``{"diagram", "fabrics": [names], "errors": [...], "notes": [...],
    "ambiguous": [{"block", "proposed_spec", "reason"}]}``. ``diagram`` is the
    rewritten document (unchanged when nothing applies)."""
    doc = {**doc, "blocks": [dict(b) for b in _blocks(doc)], "connections": [dict(c) for c in _edges(doc)]}
    errors, notes, fabrics, ambiguous = [], [], [], []
    for b in declared_fabrics(doc):
        spec = FabricSpec.from_json(b["fabric"])
        errs = spec.validate()
        if errs:
            errors.append({"block": b["name"], "errors": errs})
            continue
        nb = _normalize_fabric_block(b, spec)
        doc["blocks"] = [nb if x.get("name") == b["name"] else x for x in doc["blocks"]]
        doc["connections"], n = _retype_edges(doc, nb, spec)
        notes.extend(n)
        fabrics.append(str(b["name"]))
    declared = set(fabrics) | {b["name"] for b in declared_fabrics(doc)}
    for b in candidate_blocks(doc):
        if b["name"] in declared:
            continue
        # a candidate wired to a declared fabric is a legitimate custom block
        touches = any((c.get("from") == b["name"] and c.get("to") in declared)
                      or (c.get("to") == b["name"] and c.get("from") in declared) for c in _edges(doc))
        if touches:
            continue
        ambiguous.append({"block": b["name"], "reason": "interconnect-shaped block without a fabric spec",
                          "proposed_spec": propose_spec(doc, b)})
    if not declared:
        for target, inits in shared_targets(doc).items():
            if any(a["block"] == target for a in ambiguous):
                continue
            ambiguous.append({"block": target, "reason": f"{len(inits)} initiators share this target: {inits}",
                              "proposed_spec": {"name": "soc", "data_width": 32, "addr_width": 32,
                                                "masters": [{"name": i, "protocol": "axi4", "id_width": 4,
                                                             "max_outstanding": 4} for i in inits],
                                                "slaves": [{"name": target, "protocol": "axi4",
                                                            "base": None, "size": None}],
                                                "ordering": "per_id", "err_slave": True}})
    return {"diagram": doc, "fabrics": fabrics, "errors": errors, "notes": notes, "ambiguous": ambiguous}
