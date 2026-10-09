# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""``coresmith register <kind> <path>``: parse, validate, record.

The architect writes documents; only this function turns them into ontology
rows the state machine can reason about. It never edits the document. A
document that cannot be registered (no ids, structural violations) is
reported with stable problem codes and NOT recorded, so the DB only ever
holds artifacts that passed.
"""
from __future__ import annotations

import json
from pathlib import Path

from orchestrator.harness.tools import extract, validate
from orchestrator.state_store.ontology import file_sha, is_item_id

KINDS = ("prd", "sad", "frd", "ers", "block_diagram", "contracts", "abi", "uarch", "arch_model", "harness")


def _load(path: Path):
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() == ".json":
        return json.loads(text), text
    return None, text


def register(db, project_root, kind: str, path: str, *, block: str = "", actor: str = "cli",
             allow_warnings: bool = True, unlock: bool = False, reason: str = "") -> dict:
    """Returns ``{"ok", "kind", "artifact", "items", "links", "problems"}``.
    ``ok`` is False when any problem has severity ``error``; nothing is
    recorded in that case. ``contracts``: a document that would change a
    locked edge is refused (``CT_LOCKED``) unless ``unlock`` (the CLI demands
    a ``reason`` with it)."""
    pr = Path(project_root)
    p = Path(path)
    if not p.is_absolute():
        p = pr / p
    if not p.exists():
        return {"ok": False, "kind": kind, "problems": [{"code": "NO_FILE", "where": str(path), "text": "file not found", "severity": "error"}]}
    if kind not in KINDS:
        return {"ok": False, "kind": kind, "problems": [{"code": "BAD_KIND", "where": kind, "text": f"kind must be one of {KINDS}", "severity": "error"}]}
    doc, text = _load(p)
    sha = file_sha(p)
    items: list[dict] = []
    links: list[tuple[str, str, str]] = []
    problems: list[dict] = []
    art_kind = kind
    if kind == "prd":
        if doc is None:
            problems.append({"code": "PRD_NOT_JSON", "where": str(p), "text": "register the structured .coresmith/prd_spec.json", "severity": "error"})
        else:
            items, probs = extract.extract_prd_items(doc)
            problems += [{"code": "PRD_ITEM", "where": "prd", "text": t, "severity": "error"} for t in probs]
    elif kind == "frd":
        items, probs = extract.extract_frd_items(text)
        problems += [{"code": "FRD_ITEM", "where": "frd", "text": t, "severity": "error" if "no '**ID**" in t else "warning"} for t in probs]
        for it in items:
            for ref in it.get("extra", {}).get("refs") or []:
                links.append((it["id"], ref, "derives_from"))
    elif kind == "ers":
        if doc is None:
            problems.append({"code": "ERS_NOT_JSON", "where": str(p), "text": "register .coresmith/ers_spec.json", "severity": "error"})
        else:
            items, links, probs = extract.extract_ers_items(doc)
            problems += [{"code": "ERS_ITEM", "where": "ers", "text": t, "severity": "warning"} for t in probs]
    elif kind == "block_diagram":
        if doc is None:
            problems.append({"code": "BD_NOT_JSON", "where": str(p), "text": "register the block_diagram.json", "severity": "error"})
        else:
            problems += validate.validate_block_diagram(doc)
            for inv in doc.get("system_invariants") or []:
                iid = str((inv.get("id") if isinstance(inv, dict) else inv) or "")
                if iid and not is_item_id(iid):
                    problems.append({"code": "BD_INV_NAMESPACE", "where": iid,
                                     "text": "diagram invariant ids must be FRD ids (INV-nnn); cite, do not invent (SI-* is a third namespace)",
                                     "severity": "warning"})
    elif kind == "contracts":
        if doc is None:
            problems.append({"code": "CT_NOT_JSON", "where": str(p), "text": "register interface_contracts.json", "severity": "error"})
        else:
            diagram = None
            bd = pr / ".coresmith" / "block_diagram.json"
            if bd.exists():
                try:
                    diagram = json.loads(bd.read_text())
                except ValueError:
                    diagram = None
            problems += validate.validate_contracts(doc, diagram)
    elif kind == "uarch":
        if not block:
            problems.append({"code": "UA_NO_BLOCK", "where": str(p), "text": "--block <name> required", "severity": "error"})
        else:
            art_kind = f"uarch:{block}"
            problems += validate.validate_uarch_spec(text, block)
            items, links = extract.extract_uarch_items(text, block)
    elif kind == "abi":
        if len(text) < 200:
            problems.append({"code": "ABI_THIN", "where": str(p), "text": "the HW/SW ABI must carry the memory map, register maps and the GPU ISA", "severity": "error"})
    # sad / arch_model / harness: registered by content, no items
    errors = [q for q in problems if q.get("severity") == "error"]
    if errors:
        return {"ok": False, "kind": kind, "artifact": art_kind, "items": len(items), "problems": problems}
    if kind == "contracts" and doc is not None:
        from orchestrator.state_store.project_db import ContractLockedError
        if unlock and not (reason or "").strip():
            return {"ok": False, "kind": kind, "artifact": art_kind, "items": 0, "problems": [
                {"code": "CT_UNLOCK_NO_REASON", "where": "--unlock", "severity": "error",
                 "text": "--unlock requires --reason TEXT (recorded in the actions log)"}]}
        try:
            db.import_contracts(doc, unlock=unlock)
        except ContractLockedError as exc:
            return {"ok": False, "kind": kind, "artifact": art_kind, "items": 0, "problems": [
                {"code": "CT_LOCKED", "where": ", ".join(exc.edge_ids), "severity": "error",
                 "text": "locked edges would change; coresmith register contracts --unlock --reason ..."}]}
        except Exception as exc:  # noqa: BLE001
            problems.append({"code": "CT_IMPORT", "where": "db", "text": f"contracts import failed: {exc}", "severity": "warning"})
    art = db.register_artifact(art_kind, str(p.relative_to(pr)) if p.is_relative_to(pr) else str(p), sha=sha,
                               meta={"block": block} if block else {}, registered_by=actor)
    counts = db.upsert_items(art_kind, items, artifact_sha=sha) if items or kind in ("prd", "frd", "ers") else {"items": 0}
    elsewhere = counts.get("owned_elsewhere") or []
    if elsewhere:
        problems.append({"code": "ITEM_OWNED_ELSEWHERE", "where": ", ".join(e["id"] for e in elsewhere),
                         "severity": "warning",
                         "text": "ids owned by another artifact were NOT moved or rewritten (their links from this "
                                 "document were still added): " + ", ".join(f"{e['id']}@{e['artifact']}" for e in elsewhere)})
    n_links = 0
    for a, b, rel in links:
        try:
            db.link_items(a, b, rel, source=art_kind)
            n_links += 1
        except ValueError:
            pass
    if kind == "prd":
        for it in items:
            if it["kind"] == "Q":
                if not any(q["text"] == it["text"] for q in db.questions(open_only=False)):
                    db.add_question(it["text"], item_id=it["id"], must_answer=True, asked_by="prd")
    if kind == "block_diagram" and doc is not None:
        try:
            db.import_block_diagram(doc)
        except Exception as exc:  # noqa: BLE001
            problems.append({"code": "BD_IMPORT", "where": "db", "text": f"queue import failed: {exc}", "severity": "warning"})
        for b in doc.get("blocks") or []:
            for owned in (b.get("owns") or b.get("requirements") or []):
                if is_item_id(str(owned)):
                    db.link_items(str(owned), f"block:{b.get('name')}", "owned_by", source="block_diagram")
    if kind == "ers" and doc is not None:
        view = export_ers_view(pr, p, doc)
        if view:
            problems.append({"code": "ERS_VIEW", "where": str(view.relative_to(pr)), "severity": "info",
                             "text": "wrote the ERS view validation_dv reads"})
    return {"ok": True, "kind": kind, "artifact": art, "items": counts, "links": n_links, "problems": problems}


ERS_VIEW = Path(".coresmith") / "ers_spec.json"


def export_ers_view(project_root, source: Path, doc: dict) -> Path | None:
    """Write ``.coresmith/ers_spec.json`` (the file ``validation_dv`` reads)
    from a registered ERS document; None when ``source`` already is that file."""
    pr = Path(project_root)
    target = pr / ERS_VIEW
    try:
        if Path(source).resolve() == target.resolve():
            return None
    except OSError:
        pass
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    tmp.replace(target)
    return target
