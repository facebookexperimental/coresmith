# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Deterministic validators the architect cannot talk past. Each returns a
list of problem dicts ``{"code", "where", "text", "severity"}`` (empty = ok).
The codes are stable so rulings/waivers can name them."""
from __future__ import annotations

import re
from collections import Counter


def _p(code: str, where: str, text: str, severity: str = "error") -> dict:
    return {"code": code, "where": where, "text": text, "severity": severity}


def _port_of(conn: dict, side: str) -> tuple[str, str]:
    """(block, port) for producer/consumer whichever key spelling the diagram
    used (``producer_block``/``producer_port`` or the block-diagram prompt's
    ``from``/``to``/``interface``, where ``interface`` names the edge once)."""
    b = conn.get(f"{side}_block") or conn.get(side) or conn.get("from" if side == "producer" else "to") or ""
    p = conn.get(f"{side}_port") or conn.get(f"{side}_interface") or conn.get("interface") or ""
    return str(b), str(p)


def _width_of(conn: dict):
    for wk in ("data_width_bits", "data_width", "width", "width_bits"):
        v = conn.get(wk)
        if v not in (None, ""):
            return v
    return None


def _instances(b: dict) -> int:
    try:
        return int(b.get("instances") or b.get("instance_count") or 1)
    except (TypeError, ValueError):
        return 1


def _conn_matches(conn: dict, contract: dict) -> bool:
    pb, pp = _port_of(conn, "producer")
    cb, cp = _port_of(conn, "consumer")
    if (str(contract.get("producer_block")), str(contract.get("consumer_block"))) != (pb, cb):
        return False
    if conn.get("producer_port") or conn.get("consumer_port"):
        return str(contract.get("producer_port")) == pp and str(contract.get("consumer_port")) == cp
    iface = str(conn.get("interface") or "")
    if not iface:
        return True
    toks = [t.strip() for t in re.split(r"[/,]| or ", iface) if t.strip()]
    hay = " ".join(str(contract.get(k) or "") for k in ("producer_port", "consumer_port", "edge_id"))
    return any(t.split(" ")[0] in hay for t in toks)


def validate_block_diagram(doc: dict) -> list[dict]:
    """Every block named once; instance multiplicity explicit; every connection
    joins two declared blocks and names an instance when the block has more
    than one; no reversed duplicate of a req/resp pair; width fields numeric."""
    problems: list[dict] = []
    blocks = list(doc.get("blocks") or [])
    names = [str(b.get("name") or "") for b in blocks]
    for n, c in Counter(names).items():
        if c > 1:
            problems.append(_p("BD_DUP_BLOCK", n, f"block '{n}' declared {c} times"))
    if not names:
        problems.append(_p("BD_EMPTY", "blocks", "no blocks"))
    multi = {str(b["name"]): _instances(b) for b in blocks if b.get("name")}
    for b in blocks:
        for k in ("interfaces", "interface"):
            v = b.get(k)
            if isinstance(v, dict):
                for port, spec in v.items():
                    if isinstance(spec, str) and re.search(r"\bone per instance\b|\bper instance\b|/ s_\w+\b", spec) and multi.get(str(b.get("name")), 1) < 2:
                        problems.append(_p("BD_FOLDED_INSTANCES", f"{b.get('name')}.{port}",
                                           f"port '{port}' describes multiple instances ('{spec[:60]}') but the block declares "
                                           "no 'instances' count; set instances=N and one connection per instance"))
    conns = list(doc.get("connections") or [])
    seen_pairs: dict[tuple, dict] = {}
    for i, c in enumerate(conns):
        pb, pp = _port_of(c, "producer")
        cb, cp = _port_of(c, "consumer")
        where = c.get("edge_id") or c.get("name") or f"connections[{i}]"
        for side, b in (("producer", pb), ("consumer", cb)):
            if b and b not in multi:
                problems.append(_p("BD_UNKNOWN_BLOCK", where, f"{side} block '{b}' is not in blocks[]"))
            if b in multi and multi[b] > 1 and not (c.get(f"{side}_instance") is not None or re.search(r"\[\d+\]$", str(c.get(f"{side}_block", "")))):
                problems.append(_p("BD_INSTANCE_UNSPECIFIED", where,
                                   f"block '{b}' has {multi[b]} instances; the connection must name the instance "
                                   f"({side}_instance or '{b}[k]')"))
        if not pb or not cb:
            problems.append(_p("BD_CONN_INCOMPLETE", where, "connection lacks producer/consumer block"))
        key = (pb, pp, cb, cp)
        rkey = (cb, cp, pb, pp)
        if key in seen_pairs and pp:
            problems.append(_p("BD_DUP_CONN", where, f"duplicate connection {pb}.{pp} -> {cb}.{cp}"))
        elif rkey in seen_pairs and str(c.get("handshake_protocol") or c.get("protocol") or "").lower() in ("req_resp", "request_response", ""):
            problems.append(_p("BD_REVERSED_DUP", where,
                               f"{cb}.{cp} -> {pb}.{pp} already exists: a req/resp channel is ONE edge (the response "
                               "travels on it), not two reversed edges"))
        seen_pairs[key] = c
        w = _width_of(c)
        if w is not None:
            try:
                int(w)
            except (TypeError, ValueError):
                problems.append(_p("BD_WIDTH_NAN", where, f"width={w!r} is not an integer"))
    return problems


def validate_contracts(doc: dict, diagram: dict | None = None) -> list[dict]:
    """Structural checks (the specialist's own validator), the timing object
    on every edge that needs one, no phantom payload fields on bus edges,
    every diagram connection covered, and diagram/contract widths agree."""
    problems: list[dict] = []
    contracts = list(doc.get("contracts") or [])
    if not contracts:
        return [_p("CT_EMPTY", "contracts", "no contracts")]
    try:
        from orchestrator.architecture.specialists.interface_definition import _validate_contracts
        expected = []
        for c in (diagram or {}).get("connections") or []:
            pb, pp = _port_of(c, "producer")
            cb, cp = _port_of(c, "consumer")
            expected.append({"producer_block": pb, "producer_port": pp, "consumer_block": cb, "consumer_port": cp, **c})
        _res, notes = _validate_contracts({"contracts": contracts}, expected)
        for v in _res.get("contract_violations") or []:
            problems.append(_p("CT_" + str(v.get("type") or "STRUCTURAL").upper(), str(v.get("edge") or ""), str(v.get("violation") or "")))
    except Exception as exc:  # noqa: BLE001 -- the specialist validator is best-effort here
        problems.append(_p("CT_VALIDATOR_ERROR", "contracts", f"structural validator raised: {exc}", "warning"))
    try:
        from orchestrator.architecture.specialists.contract_timing import timing_violations
        for c in contracts:
            for t in timing_violations(c) or []:
                problems.append(_p("CT_TIMING", str(c.get("edge_id") or ""), str(t)))
    except Exception:  # noqa: BLE001
        pass
    bus = ("axi4", "axi_lite", "apb")
    ids = Counter(str(c.get("edge_id") or "") for c in contracts)
    seen_pairs: set[tuple] = set()
    for c in contracts:
        eid = str(c.get("edge_id") or "")
        if ids[eid] > 1:
            problems.append(_p("CT_DUP_EDGE", eid, "duplicate edge_id"))
        proto = str(c.get("handshake_protocol") or "").lower()
        if proto in bus:
            names = {str(f.get("name") if isinstance(f, dict) else f) for f in c.get("fields") or []}
            if names & {"data", "payload"}:
                problems.append(_p("CT_BUS_PHANTOM_FIELD", eid,
                                   f"{proto} edge lists a generic payload field {sorted(names & {'data', 'payload'})}; "
                                   "the AMBA channel set is the port set", "warning"))
        key = (c.get("producer_block"), c.get("producer_port"), c.get("consumer_block"), c.get("consumer_port"))
        rkey = (key[2], key[3], key[0], key[1])
        if rkey in seen_pairs and proto == "req_resp":
            problems.append(_p("CT_REVERSED_DUP", eid, "reversed duplicate of a req_resp edge"))
        seen_pairs.add(key)
    if diagram:
        conns = list(diagram.get("connections") or [])
        pair_conns = Counter((_port_of(k, "producer")[0], _port_of(k, "consumer")[0]) for k in conns)
        pair_contracts: dict[tuple, list] = {}
        for c in contracts:
            pair_contracts.setdefault((str(c.get("producer_block")), str(c.get("consumer_block"))), []).append(c)
        for conn in conns:
            pb, pp = _port_of(conn, "producer")
            cb, cp = _port_of(conn, "consumer")
            matches = [c for c in contracts if _conn_matches(conn, c)]
            if not matches:
                cands = pair_contracts.get((pb, cb)) or []
                if cands and len(cands) == pair_conns[(pb, cb)]:
                    # same number of edges between the pair: the names drifted, the edge exists
                    problems.append(_p("CT_EDGE_NAME_DRIFT", f"{pb}.{pp}->{cb}.{cp}",
                                       f"diagram interface '{pp}' is not spelled in the contract port names "
                                       f"({', '.join(str(c.get('edge_id')) for c in cands[:3])}); rename one side", "warning"))
                    matches = cands
                else:
                    problems.append(_p("CT_MISSING_EDGE", f"{pb}.{pp}->{cb}.{cp}", "diagram connection has no contract"))
                    continue
            c = matches[0]
            dw = _width_of(conn)
            cw = c.get("data_width_bits")
            if dw not in (None, "") and cw not in (None, "") and int(dw) != int(cw):
                problems.append(_p("CT_WIDTH_MISMATCH", str(c.get("edge_id") or ""),
                                   f"diagram says {dw} bits, contract says {cw}"))
        # every declared multi-instance / fabric master port must be reached
        for b in diagram.get("blocks") or []:
            fab = b.get("fabric") if isinstance(b.get("fabric"), dict) else None
            if fab:
                reached = {str(c.get("consumer_port")) for c in contracts if c.get("consumer_block") == b.get("name")}
                for m in fab.get("masters") or []:
                    mp = f"s_{m.get('name')}" if isinstance(m, dict) else f"s_{m}"
                    if mp not in reached:
                        problems.append(_p("CT_FABRIC_PORT_UNREACHED", f"{b.get('name')}.{mp}",
                                           "fabric master port has no contract edge (an initiator is missing or folded)"))
    for edge in contracts:
        policy = edge.get("flow_control_policy")
        if policy is not None and not isinstance(policy, dict):
            problems.append(_p("CT_POLICY_TYPE", str(edge.get("edge_id", "")),
                               "flow_control_policy must be an object"))
    return problems


def validate_uarch_spec(markdown: str, block: str) -> list[dict]:
    problems = []
    req = ["## 2", "## 3", "### 4a", "## 5", "### 6a", "## 9"]
    for h in req:
        if h not in (markdown or ""):
            problems.append(_p("UA_SECTION_MISSING", f"{block}:{h}", f"section '{h}' missing"))
    if "MEETS" in (markdown or "") and not re.search(r"PERF-\d{3}", markdown or ""):
        problems.append(_p("UA_MEETS_UNCITED", block, "claims MEETS without citing a PERF-nnn id", "warning"))
    return problems
