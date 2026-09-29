# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Assertion stage (A3): the invariants a block's spec promises must exist as
assertions in its RTL (or as a VIP bind), and no comment may claim an
assertion that is not there.

In the SoC benchmark the coherence controller's spec cited "sim assertions"
for its N=1 snoop contract and the AXI arbiter's RTL said "checked by
assertion" -- and 17.6k lines of RTL contained no assertion at all. This
stage runs between RTL generation and testbench generation:

1. ``extract_invariants`` builds the checklist: every ``INV-*`` id in the
   uArch spec's §4a plus a ``TIM-<edge>-<rule>`` per timing rule of the
   block's contract edges;
2. ``find_assertions`` locates real assertions in the RTL (SVA
   ``assert property``, immediate ``assert(``, ``$error``-guarded checks)
   and the ``// INV: <id>`` tags next to them;
3. ``phantom_assertion_claims`` lints comments that claim an assertion with
   none nearby;
4. ``evaluate`` reports missing ids (TIM ids are satisfied by the edge's VIP
   SVA bind when it exists) and phantom claims.

CORESMITH_ASSERTION_STAGE = 1 (gate, default) | advisory (record only) | 0.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from orchestrator.langgraph.contract_conformance import strip_preprocessor

INV_ID_RE = re.compile(r"\bINV-[A-Z0-9][A-Z0-9_\-]*\b")
TAG_RE = re.compile(r"//\s*INV:\s*([A-Za-z0-9_\-]+(?:\s*,\s*[A-Za-z0-9_\-]+)*)")
ASSERT_RE = re.compile(
    r"(\bassert\s+property\b|\bassert\s*\(|\bassume\s+property\b|\bcover\s+property\b|"
    r"\$error\s*\(|\$fatal\s*\(|\bERROR_CHECK\b)")
PHANTOM_RE = re.compile(
    r"//.*\b(checked|verified|guarded|covered|enforced|caught)\s+(by|via|with)\s+(an?\s+|the\s+)?"
    r"(sva\s+|sim\s+|simulation\s+|runtime\s+)?assert(ion|ions|ed)?\b"
    r"|//.*\bassert(ed|ion)?\s+(elsewhere|in\s+(the\s+)?(tb|testbench|dv|simulation))\b",
    re.IGNORECASE)
_SECTION_4A = re.compile(r"^#{2,4}\s*4a\.?\b", re.IGNORECASE)
_SECTION_NEXT = re.compile(r"^#{2,3}\s*\d+[a-z]?\.?\s")
_WINDOW = 3


def mode() -> str:
    v = (os.environ.get("CORESMITH_ASSERTION_STAGE", "1") or "1").strip().lower()
    if v in {"0", "false", "no", "off", ""}:
        return "off"
    if v in {"advisory", "advise", "warn", "report"}:
        return "advisory"
    return "gate"


# --------------------------------------------------------------------------
# 1. checklist
# --------------------------------------------------------------------------

def _section_4a(spec_text: str) -> str:
    lines = spec_text.splitlines()
    out, inside = [], False
    for ln in lines:
        if _SECTION_4A.match(ln.strip()):
            inside = True
            continue
        if inside and _SECTION_NEXT.match(ln.strip()) and not _SECTION_4A.match(ln.strip()):
            break
        if inside:
            out.append(ln)
    return "\n".join(out)


def spec_invariants(spec_text: str) -> list[dict]:
    """``[{id, text}]`` -- one entry per ``INV-*`` id in §4a (first id of a
    table row / bullet is the entry; ids in parentheses are aliases)."""
    seen: set[str] = set()
    out: list[dict] = []
    for ln in _section_4a(spec_text).splitlines():
        ids = INV_ID_RE.findall(ln)
        if not ids:
            continue
        primary = ids[0]
        if primary in seen:
            continue
        seen.add(primary)
        text = re.sub(r"\s+", " ", ln.strip().strip("|")).strip()
        out.append({"id": primary, "aliases": [i for i in ids[1:] if i != primary],
                    "text": text[:300], "source": "uarch:4a"})
    return out


def contract_invariants(project_root, block: str) -> list[dict]:
    """``TIM-<edge>-<rule>`` entries from the block's contract timing."""
    try:
        from orchestrator.state_store.project_db import open_project
        edges = open_project(project_root).contract_edges_for_block(block)
    except Exception:  # noqa: BLE001
        edges = []
    from orchestrator.architecture.specialists.contract_timing import normalize_timing
    out: list[dict] = []
    for e in edges:
        if not isinstance(e, dict):
            continue
        e = json.loads(json.dumps(e, default=str))
        normalize_timing(e)
        eid = str(e.get("edge_id") or "?")
        fam = str(e.get("handshake_protocol") or "")
        t = e.get("timing") or {}
        role = "producer" if e.get("producer_block") == block else "consumer"
        lat = t.get("req_to_rsp_cycles") or {}
        if fam == "req_resp" and lat.get("min") is not None:
            rule = (f"exact {lat['exact']}" if lat.get("exact") is not None
                    else f"[{lat.get('min')}:{lat.get('max')}]")
            out.append({"id": f"TIM-{eid}-latency", "edge_id": eid, "role": role,
                        "text": f"response {rule} cycle(s) after request accept", "source": "contract"})
        if fam in ("axi_stream", "srdy_drdy") and t.get("valid_hold_until_ready"):
            out.append({"id": f"TIM-{eid}-hold", "edge_id": eid, "role": role,
                        "text": "valid and payload hold until ready", "source": "contract"})
        if t.get("valid_to_ready_max_stall") is not None:
            out.append({"id": f"TIM-{eid}-stall", "edge_id": eid, "role": role,
                        "text": f"ready within {t['valid_to_ready_max_stall']} cycles of valid",
                        "source": "contract"})
        if int(t.get("reset_idle_cycles") or 0) >= 1 and fam != "static":
            out.append({"id": f"TIM-{eid}-reset_idle", "edge_id": eid, "role": role,
                        "text": f"valids low {t['reset_idle_cycles']} cycle(s) after reset",
                        "source": "contract"})
    return out


def extract_invariants(project_root, block: str) -> list[dict]:
    spec = Path(project_root) / "arch" / "uarch_specs" / f"{block}.md"
    try:
        text = spec.read_text(encoding="utf-8", errors="replace")
    except OSError:
        text = ""
    return spec_invariants(text) + contract_invariants(project_root, block)


# --------------------------------------------------------------------------
# 2. assertions in the RTL
# --------------------------------------------------------------------------

def find_assertions(rtl_text: str) -> dict:
    """``{"assertions": [line_no...], "tags": {id: line_no}, "covered": {id}}``.

    The RTL is read as simulation sees it (no defines), so an assertion under
    `` `ifndef SYNTHESIS `` counts. A tag covers an id when an assertion sits
    within ``_WINDOW`` lines after (or on) the tag line.
    """
    text = strip_preprocessor(rtl_text or "", defines=())
    lines = text.splitlines()
    assertions = [i + 1 for i, ln in enumerate(lines) if ASSERT_RE.search(ln)]
    tags: dict[str, int] = {}
    for i, ln in enumerate(lines):
        m = TAG_RE.search(ln)
        if m:
            for tid in re.split(r"\s*,\s*", m.group(1)):
                tags.setdefault(tid.strip(), i + 1)
    aset = set(assertions)
    covered = {tid for tid, ln in tags.items()
               if any((ln + k) in aset for k in range(0, _WINDOW + 1))}
    return {"assertions": assertions, "tags": tags, "covered": covered}


# --------------------------------------------------------------------------
# 3. phantom claims
# --------------------------------------------------------------------------

def phantom_assertion_claims(rtl_text: str) -> list[dict]:
    """Comments that claim an assertion with none within ``_WINDOW`` lines."""
    lines = (rtl_text or "").splitlines()
    aset = {i + 1 for i, ln in enumerate(lines) if ASSERT_RE.search(ln)}
    out = []
    for i, ln in enumerate(lines):
        if PHANTOM_RE.search(ln) and not any((i + 1 + k) in aset for k in range(-_WINDOW, _WINDOW + 1)):
            out.append({"line": i + 1, "text": ln.strip()[:160]})
    return out


# --------------------------------------------------------------------------
# 4. verdict
# --------------------------------------------------------------------------

def _vip_sva_covers(project_root, block: str, inv: dict) -> bool:
    """A TIM-* id is covered when the edge's VIP SVA bind for this block's side
    exists and states that rule."""
    try:
        from orchestrator.langgraph.vip_lib.codegen import sva_bind_enabled, vips_for_block
        if not sva_bind_enabled():
            return False
        for r in vips_for_block(project_root, block):
            if r["edge_id"] != inv.get("edge_id") or not r.get("sva"):
                continue
            sv = Path(r["sva"]).read_text(errors="replace") if Path(r["sva"]).exists() else ""
            rule = inv["id"].rsplit("-", 1)[-1]
            return {"latency": "a_latency", "hold": "a_hold", "stall": "a_stall",
                    "reset_idle": "a_reset_idle"}.get(rule, "") in sv
    except Exception:  # noqa: BLE001
        return False
    return False


def evaluate(project_root, block: str, rtl_path) -> dict:
    """The stage verdict for one block; also written to
    ``.coresmith/blocks/<block>/assertions_checklist.json``."""
    try:
        rtl = Path(rtl_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        rtl = ""
    checklist = extract_invariants(project_root, block)
    found = find_assertions(rtl)
    phantom = phantom_assertion_claims(rtl)
    covered, missing = [], []
    for inv in checklist:
        ids = [inv["id"]] + list(inv.get("aliases") or [])
        if any(i in found["covered"] for i in ids):
            covered.append({**inv, "by": "rtl"})
        elif inv["source"] == "contract" and _vip_sva_covers(project_root, block, inv):
            covered.append({**inv, "by": "vip_sva"})
        else:
            missing.append(inv)
    m = mode()
    ok = (m == "off") or (m == "advisory") or (not missing and not phantom)
    report = {"block": block, "mode": m, "ok": ok, "assertion_count": len(found["assertions"]),
              "checklist": checklist, "covered": covered, "missing": missing,
              "phantom_claims": phantom, "tags": found["tags"]}
    try:
        bdir = Path(project_root) / ".coresmith" / "blocks" / block
        bdir.mkdir(parents=True, exist_ok=True)
        (bdir / "assertions_checklist.json").write_text(json.dumps(report, indent=2, default=str))
    except OSError:
        pass
    return report


def feedback_text(report: dict) -> str:
    lines = ["ASSERTION STAGE: the RTL does not carry the assertions its spec promises."]
    if report.get("missing"):
        lines.append("Missing (add `// INV: <id>` directly above a real assertion under "
                     "`ifndef SYNTHESIS`, e.g. `always @(posedge clk) if (rst_n && !(<cond>)) "
                     "$error(\"<id>\");` or `assert property (@(posedge clk) disable iff (!rst_n) <expr>);`):")
        for inv in report["missing"]:
            lines.append(f"  - {inv['id']}: {inv.get('text', '')}")
    if report.get("phantom_claims"):
        lines.append("Phantom claims (a comment says an assertion exists but none is within "
                     f"{_WINDOW} lines -- add the assertion or delete the claim):")
        for p in report["phantom_claims"]:
            lines.append(f"  - line {p['line']}: {p['text']}")
    lines.append(f"Assertions found in the RTL: {report.get('assertion_count', 0)}.")
    return "\n".join(lines)
