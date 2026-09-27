# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Item extractors: turn the architect's documents into ontology items.

Shapes come from the artifacts the engine already writes (``prd_spec.json``,
``ers_spec.json``, ``arch/frd_spec.md``, ``arch/uarch_specs/<b>.md``); an
id-in-string convention (``"FR-CPU-2: ..."``) is accepted but the id must be
there -- a requirement without an id cannot be linked, checked or waived.
"""
from __future__ import annotations

import json
import re

from orchestrator.state_store.ontology import is_item_id

_LEAD_ID = re.compile(r"^\s*`?([A-Z][A-Z0-9]*(?:-[A-Za-z0-9_]+)*-\d+[a-z]?)`?\s*(?:\[([^\]]*)\])?\s*[:\-–]\s*(.*)$", re.S)
_REF_RE = re.compile(r"\b([A-Z][A-Z0-9]*(?:-[A-Za-z0-9_]+)*-\d{1,3}[a-z]?)\b")


def split_lead_id(text: str) -> tuple[str, str, str]:
    """``"FR-CPU-2 [HARD, R2.1]: body"`` -> (id, tags, body); ("", "", text) if none."""
    m = _LEAD_ID.match(text or "")
    if not m or not is_item_id(m.group(1)):
        return "", "", text or ""
    return m.group(1), m.group(2) or "", m.group(3).strip()


def references(text: str, *, exclude: str = "") -> list[str]:
    out = []
    for r in _REF_RE.findall(text or ""):
        if r != exclude and is_item_id(r) and r not in out:
            out.append(r)
    return out


def _priority_from_tags(tags: str, body: str) -> str:
    t = (tags or "").upper()
    if "HARD" in t or "MUST" in t:
        return "must_have"
    if "GOAL" in t or "SHOULD" in t or "SELF" in t:
        return "should_have"
    return ""


def extract_prd_items(doc: dict) -> tuple[list[dict], list[str]]:
    """PRD JSON (``{"prd": {...}}`` or the inner dict). Items: FR-* (functional
    requirements), KPI-* (validation_kpis), CON-n (constraints), Q-n
    (open_items -> also questions). Returns (items, problems)."""
    prd = doc.get("prd", doc) if isinstance(doc, dict) else {}
    items, problems = [], []
    for i, fr in enumerate(prd.get("functional_requirements") or [], 1):
        if isinstance(fr, dict):
            iid, body = str(fr.get("id") or ""), str(fr.get("requirement") or fr.get("text") or "")
            tags, acc = str(fr.get("priority") or ""), str(fr.get("acceptance") or "")
        else:
            iid, tags, body = split_lead_id(str(fr))
            acc = ""
        if not iid:
            problems.append(f"functional_requirements[{i}] has no id (expected 'FR-<AREA>-<n>: ...')")
            continue
        items.append({"id": iid, "kind": "FR", "section": "functional_requirements", "text": body,
                      "priority": _priority_from_tags(tags, body) or (tags if isinstance(fr, dict) else ""),
                      "acceptance": acc, "extra": {"tags": tags, "refs": references(body, exclude=iid)}})
    for i, k in enumerate(prd.get("validation_kpis") or [], 1):
        if not isinstance(k, dict) or not k.get("id"):
            problems.append(f"validation_kpis[{i}] has no id")
            continue
        items.append({"id": str(k["id"]), "kind": "KPI", "section": "validation_kpis",
                      "text": str(k.get("metric") or ""), "priority": "must_have",
                      "acceptance": f"{k.get('threshold', '')} -- {k.get('test_method', '')}".strip(" -"),
                      "extra": {"source": k.get("source", "")}})
    for i, c in enumerate(prd.get("constraints") or [], 1):
        text = c if isinstance(c, str) else json.dumps(c)
        iid, tags, body = split_lead_id(text)
        items.append({"id": iid or f"CON-{i}", "kind": "CON", "section": "constraints", "text": body or text,
                      "priority": "must_have", "acceptance": ""})
    for i, q in enumerate(prd.get("open_items") or [], 1):
        text = q if isinstance(q, str) else json.dumps(q)
        iid, _t, body = split_lead_id(text)
        items.append({"id": iid or f"Q-{i}", "kind": "Q", "section": "open_items", "text": body or text,
                      "priority": "must_have", "acceptance": "", "status": "open"})
    return items, problems


def extract_frd_items(markdown: str) -> tuple[list[dict], list[str]]:
    from orchestrator.systemc_model.frd_eval import extract_requirements
    reqs = extract_requirements(markdown)
    items, problems = [], []
    for r in reqs:
        text = r["requirement"]
        if not r.get("acceptance"):
            problems.append(f"{r['id']}: no acceptance criteria")
        items.append({"id": r["id"], "kind": r["id"].split("-")[0], "section": r.get("section", ""), "text": text,
                      "priority": r.get("priority") or _priority_from_tags("", text), "acceptance": r.get("acceptance", ""),
                      "model_check": r.get("model_check", ""), "extra": {"refs": references(text + " " + r.get("acceptance", ""), exclude=r["id"])}})
    if not items:
        problems.append("no '**ID**: XXX-NNN' requirement blocks found")
    return items, problems


def extract_ers_items(doc: dict) -> tuple[list[dict], list[tuple[str, str, str]], list[str]]:
    """ERS JSON. Items: FR-* (functional_requirements), INV-* (system_invariants),
    VAL-* (validation_dv_requirements, with ``covers`` -> links), ERS-<block>-n
    (per_block_requirements), C-ERS-n (open_items). Returns (items, links, problems)."""
    ers = doc.get("ers", doc) if isinstance(doc, dict) else {}
    items, links, problems = [], [], []
    for i, fr in enumerate(ers.get("functional_requirements") or [], 1):
        text = fr if isinstance(fr, str) else str(fr.get("requirement") or fr.get("text") or fr.get("id") or "")
        iid, tags, body = split_lead_id(text) if isinstance(fr, str) else (str(fr.get("id") or ""), str(fr.get("priority") or ""), text)
        if not iid:
            problems.append(f"functional_requirements[{i}] has no id")
            continue
        items.append({"id": iid, "kind": "FR", "section": "functional_requirements", "text": body,
                      "priority": _priority_from_tags(tags, body), "acceptance": "",
                      "extra": {"tags": tags, "refs": references(body + " " + tags, exclude=iid)}})
        for ref in references(tags):
            links.append((iid, ref, "derives_from"))
    for inv in ers.get("system_invariants") or []:
        if not isinstance(inv, dict) or not inv.get("id"):
            problems.append("system_invariants entry without id")
            continue
        items.append({"id": str(inv["id"]), "kind": "INV", "section": "system_invariants",
                      "text": str(inv.get("description") or ""), "priority": "must_have",
                      "acceptance": str(inv.get("verification") or inv.get("golden_reference") or ""),
                      "extra": {"affected_blocks": inv.get("affected_blocks") or []}})
        for b in inv.get("affected_blocks") or []:
            links.append((str(inv["id"]), f"block:{b}", "owned_by"))
    for v in ers.get("validation_dv_requirements") or []:
        if not isinstance(v, dict) or not v.get("id"):
            problems.append("validation_dv_requirements entry without id")
            continue
        items.append({"id": str(v["id"]), "kind": "VAL", "section": "validation_dv_requirements",
                      "text": str(v.get("requirement") or ""), "priority": _priority_from_tags("", str(v.get("requirement") or "")) or "must_have",
                      "acceptance": f"{v.get('measurable_kpi', '')}: {v.get('threshold', '')} -- {v.get('test_method', '')}",
                      "extra": {"covers": v.get("covers") or []}})
        for c in v.get("covers") or []:
            if is_item_id(str(c)):
                links.append((str(v["id"]), str(c), "covers"))
    for pb in ers.get("per_block_requirements") or []:
        if not isinstance(pb, dict):
            continue
        block = str(pb.get("block") or pb.get("block_name") or pb.get("name") or "")
        for j, r in enumerate(pb.get("requirements") or [], 1):
            text = r if isinstance(r, str) else str(r.get("requirement") or r.get("text") or "")
            iid, tags, body = split_lead_id(text)
            iid = iid or f"ERS-{block}-{j}"
            items.append({"id": iid, "kind": "ERS", "section": f"per_block:{block}", "text": body or text,
                          "priority": _priority_from_tags(tags, body), "acceptance": "",
                          "extra": {"block": block, "refs": references(text, exclude=iid)}})
            links.append((iid, f"block:{block}", "owned_by"))
            for ref in references(text, exclude=iid):
                links.append((iid, ref, "derives_from"))
    for i, o in enumerate(ers.get("open_items") or [], 1):
        text = o if isinstance(o, str) else json.dumps(o)
        iid, _t, body = split_lead_id(text)
        items.append({"id": iid or f"C-ERS-{i}", "kind": "C", "section": "open_items", "text": body or text,
                      "priority": "should_have", "acceptance": ""})
    return items, links, problems


_UARCH_INV = re.compile(r"\b(INV-[A-Z0-9_]+-\d{3})\b")
_UARCH_CITES = re.compile(r"\b((?:PERF|INV|TIME|IFACE|VAL|FR)-[A-Za-z0-9_]*-?\d{1,3})\b")


def extract_uarch_items(markdown: str, block: str) -> tuple[list[dict], list[tuple[str, str, str]]]:
    """A block's uArch spec: its INV-<BLK>-nnn invariants become items owned by
    the block; every FRD id it cites (§6.1 PERF cross-refs, §4a) is a ``cites``
    link -- that is how "MEETS PERF-004" becomes a queryable claim."""
    items, links = [], []
    seen = set()
    for m in _UARCH_INV.finditer(markdown or ""):
        iid = m.group(1)
        if iid in seen:
            continue
        seen.add(iid)
        line_start = markdown.rfind("\n", 0, m.start()) + 1
        line_end = markdown.find("\n", m.end())
        line = markdown[line_start:line_end if line_end > 0 else None].strip(" -*|")
        items.append({"id": iid, "kind": "INV", "section": f"uarch:{block}", "text": line[:500],
                      "priority": "must_have", "acceptance": "", "extra": {"block": block}})
        links.append((iid, f"block:{block}", "owned_by"))
    for m in _UARCH_CITES.finditer(markdown or ""):
        ref = m.group(1)
        if is_item_id(ref) and not ref.startswith("INV-" + block.upper()) and (f"uarch:{block}", ref) not in seen:
            seen.add((f"uarch:{block}", ref))
            links.append((f"block:{block}", ref, "cites"))
    return items, links
