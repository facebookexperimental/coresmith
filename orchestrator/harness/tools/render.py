# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Render DB-authored requirements back to the documents the engine reads.

``coresmith frd add|edit|...`` write straight into ``project.sqlite``; the
FRD evaluation (``systemc_model.frd_eval``) and ``model eval`` still read
``arch/frd_spec.md`` and the PRD tools read ``.coresmith/prd_spec.json``.
These renderers produce exactly the shapes the extractors parse
(``extract.extract_frd_items`` / ``extract.extract_prd_items``), so a rendered
file re-registers to the same ids, bounds, priorities, acceptance, model
checks and derives_from references.
"""
from __future__ import annotations

import json
from pathlib import Path

from orchestrator.harness.tools.extract import RENDER_MARKER, split_trailing_tag

HEADER = f"<!-- {RENDER_MARKER}; edit with coresmith frd add/edit -->"
PRD_HEADER = f"<!-- {RENDER_MARKER}; edit with coresmith prd add/edit -->"
PINOUT_HEADER = f"<!-- {RENDER_MARKER}; edit with coresmith pin add/set/rm -->"
PINOUT_VIEW = "arch/pinout.md"

FRD_SECTIONS = {
    "PERF": "Performance Requirements",
    "IFACE": "Interface Requirements",
    "INV": "Semantic Invariants",
    "TIME": "Timing Requirements",
    "PHYS": "Physical Design Requirements",
    "TEST": "Test Requirements",
}
DEFAULT_SECTION = "Requirements"

PRD_SECTIONS = {"FR": "functional_requirements", "KPI": "validation_kpis", "CON": "constraints", "Q": "open_items"}


def frd_section(kind: str) -> str:
    return FRD_SECTIONS.get((kind or "").upper(), DEFAULT_SECTION)


def priority_tag(priority: str) -> str:
    p = (priority or "").lower()
    if "must" in p or p in ("hard", "p0", "required"):
        return "HARD"
    if "should" in p or p == "goal":
        return "SHOULD"
    return p.upper()


def _num(v) -> str:
    if v is None:
        return "-"
    f = float(v)
    return str(int(f)) if f.is_integer() and abs(f) < 1e15 else repr(f)


def _one_line(s) -> str:
    return " ".join(str(s or "").split())


def metric_line(item: dict) -> str:
    """``fps; min 55; max -; unit fps`` (``**Metric**`` field value)."""
    return (f"{_one_line(item.get('metric'))}; min {_num(item.get('bound_min'))}; "
            f"max {_num(item.get('bound_max'))}; unit {_one_line(item.get('unit')) or '-'}")


def _derives(links: list[dict], iid: str) -> list[str]:
    out = []
    for lk in links:
        if lk.get("from_id") == iid and lk.get("rel") == "derives_from" and lk["to_id"] not in out:
            out.append(lk["to_id"])
    return out


def render_frd(items: list[dict], links: list[dict]) -> str:
    """Markdown ``extract_requirements`` parses back to the same items."""
    live = [i for i in items if i.get("status") != "retired"]
    order = list(dict.fromkeys(FRD_SECTIONS.values()))
    groups: dict[str, list[dict]] = {}
    for it in sorted(live, key=lambda i: i["id"]):
        sec = _one_line(it.get("section")) or frd_section(it.get("kind") or it["id"].split("-")[0])
        groups.setdefault(sec, []).append(it)
    names = [s for s in order if s in groups] + [s for s in groups if s not in order]
    out = [HEADER, "", "# Functional Requirements Document", ""]
    for sec in names:
        out += [f"## {sec}", ""]
        for n, it in enumerate(groups[sec], 1):
            body, old_tags = split_trailing_tag(_one_line(it.get("text")))
            tags = []
            tag = priority_tag(it.get("priority") or "")
            for t in ([tag] if tag else []) + old_tags + _derives(links, it["id"]):
                if t and t not in tags:
                    tags.append(t)
            req = body + (f" [{', '.join(tags)}]" if tags else "")
            out.append(f"{n}. **ID**: {it['id']}")
            out.append(f"   - **Requirement**: {req}")
            for label, key in (("Acceptance criteria", "acceptance"), ("Priority", "priority"),
                               ("Model check", "model_check")):
                val = _one_line(it.get(key))
                if val:
                    out.append(f"   - **{label}**: {val}")
            if it.get("metric"):
                out.append(f"   - **Metric**: {metric_line(it)}")
            out.append("")
    return "\n".join(out).rstrip() + "\n"


def render_prd(items: list[dict], *, title: str = "", base: dict | None = None) -> tuple[str, dict]:
    """(markdown, ``prd_spec.json`` dict) for the PRD items. ``base`` is the
    existing inner ``prd`` dict (its title and other keys are kept)."""
    prd = dict(base or {})
    prd["title"] = title or prd.get("title") or "PRD"
    frs, kpis, cons, opens = [], [], [], []
    for it in sorted((i for i in items if i.get("status") != "retired"), key=lambda i: i["id"]):
        kind = (it.get("kind") or it["id"].split("-")[0]).upper()
        text = _one_line(it.get("text"))
        if kind == "KPI":
            thr, _, method = (it.get("acceptance") or "").partition(" -- ")
            kpis.append({"id": it["id"], "metric": text, "threshold": thr.strip(), "test_method": method.strip()})
        elif kind == "CON":
            cons.append(f"{it['id']}: {text}")
        elif kind == "Q":
            opens.append(f"{it['id']}: {text}")
        else:
            tag = priority_tag(it.get("priority") or "")
            if it.get("acceptance") or (it.get("priority") or "") not in ("", "must_have", "should_have"):
                frs.append({"id": it["id"], "requirement": text, "priority": it.get("priority") or "",
                            "acceptance": _one_line(it.get("acceptance"))})
            else:
                frs.append(f"{it['id']}{f' [{tag}]' if tag else ''}: {text}")
    prd.update({"functional_requirements": frs, "validation_kpis": kpis, "constraints": cons, "open_items": opens})
    md = [PRD_HEADER, "", f"# {prd['title']}", "", "## Functional requirements", ""]
    for fr in frs:
        md.append(f"- **{fr['id']}** [{priority_tag(fr['priority']) or '-'}]: {fr['requirement']} "
                  f"(acceptance: {fr['acceptance']})" if isinstance(fr, dict) else f"- {fr}")
    md += ["", "## Validation KPIs", ""]
    md += [f"- **{k['id']}**: {k['metric']} -- threshold {k['threshold'] or '-'}; test {k['test_method'] or '-'}" for k in kpis]
    md += ["", "## Constraints", ""] + [f"- {c}" for c in cons]
    if opens:
        md += ["", "## Open items", ""] + [f"- {q}" for q in opens]
    return "\n".join(md).rstrip() + "\n", {"prd": prd}


# ---------------------------------------------------------------------------
# Writers (the DB -> disk views)

def db_sourced(db, kind: str) -> bool:
    art = db.artifact(kind)
    return bool(art and str(art.get("path") or "").startswith("db:"))


def write_frd(db, root, out: str | None = None) -> Path:
    root = Path(root)
    p = Path(out) if out else Path("arch/frd_spec.md")
    p = p if p.is_absolute() else root / p
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render_frd(db.items(artifact="frd"), db.links(rel="derives_from")), encoding="utf-8")
    return p


def write_prd(db, root) -> list[Path]:
    root = Path(root)
    jp = root / ".coresmith" / "prd_spec.json"
    base = {}
    if jp.exists():
        try:
            doc = json.loads(jp.read_text())
            base = doc.get("prd", doc) if isinstance(doc, dict) else {}
        except ValueError:
            base = {}
    md, doc = render_prd(db.items(artifact="prd"), base=base if isinstance(base, dict) else {})
    mp = root / "arch" / "prd_spec.md"
    mp.parent.mkdir(parents=True, exist_ok=True)
    jp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text(md, encoding="utf-8")
    jp.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return [mp, jp]


def write_views(db, root, *, force: bool = False, kinds: tuple[str, ...] = ("frd", "prd")) -> dict:
    """Render every DB-sourced requirements artifact (or every registered one
    with ``force``). Returns ``{"written": [paths], "skipped": [{kind, reason}]}``."""
    written, skipped = [], []
    for kind in kinds:
        art = db.artifact(kind)
        if not art:
            skipped.append({"kind": kind, "reason": "not registered"})
            continue
        if not db_sourced(db, kind) and not force:
            skipped.append({"kind": kind, "reason": f"file-sourced ({art.get('path')}); --force to overwrite"})
            continue
        paths = [write_frd(db, root)] if kind == "frd" else write_prd(db, root)
        written += [str(p) for p in paths]
    return {"written": written, "skipped": skipped}


# ---------------------------------------------------------------------------
# Pins: arch/pinout.md and the Caravel prd["pin_map"]

def render_pinout(pins: list[dict], *, top: str = "") -> str:
    """The pinout table (markdown) of the ``pins`` rows."""
    L = [PINOUT_HEADER, "", f"# Pinout{f' of {top}' if top else ''}", "",
         f"{len(pins)} pin(s); every block port is a contract edge or one of these pins "
         "(`coresmith shell assemble` refuses anything else).", "",
         "| pin | dir | width | kind | block port | pad bus | oe | lock |", "|---|---|---|---|---|---|---|---|"]
    for p in pins:
        frm = f"`{p['block']}.{p['port']}`" if p.get("block") else "(shell clk/rst net)" \
            if p.get("kind") in ("clock", "reset") else "-"
        bus = f"{p['bus']}[{p['msb']}:{p['lsb']}]" if p.get("bus") else "-"
        L.append(f"| `{p['name']}` | {p['dir']} | {p['width']} | {p['kind']} | {frm} | {bus} | "
                 f"{p.get('oe') or '-'} | {'locked' if p.get('locked') else 'open'} |")
    return "\n".join(L) + "\n"


def pin_map_doc(pins: list[dict]) -> dict | None:
    """``prd["pin_map"]`` (``architecture.pin_map`` schema) from the pins that
    carry a pad ``bus``; None when none does. An ``inout`` pin is two entries
    (``<name>_in`` / ``<name>_out``) over the same bits."""
    entries = []
    top_bit = -1
    for p in pins:
        if not p.get("bus") or p.get("msb") is None or p.get("lsb") is None:
            continue
        msb, lsb = int(p["msb"]), int(p["lsb"])
        top_bit = max(top_bit, msb, lsb)
        if p["dir"] == "in":
            entries.append({"signal": p["name"], "dir": "in", "msb": msb, "lsb": lsb})
        elif p["dir"] == "out":
            entries.append({"signal": p["name"], "dir": "out", "msb": msb, "lsb": lsb,
                            **({"oe": p["oe"]} if p.get("oe") else {})})
        else:
            entries.append({"signal": f"{p['name']}_in", "dir": "in", "msb": msb, "lsb": lsb})
            entries.append({"signal": f"{p['name']}_out", "dir": "out", "msb": msb, "lsb": lsb,
                            **({"oe": p["oe"]} if p.get("oe") else {})})
    if not entries:
        return None
    return {"bus_width": max(38, top_bit + 1), "entries": entries, "source": "pins"}


def write_pin_views(db, root, *, top: str = "") -> list[Path]:
    """``arch/pinout.md`` and, when a pin carries ``--bus``, ``prd["pin_map"]``
    in ``.coresmith/prd_spec.json`` (a pin map this writer put there is
    removed again when no pin carries a bus). Nothing when there are no pins."""
    root = Path(root)
    pins = db.pins() if hasattr(db, "pins") else []
    if not pins:
        return []
    mp = root / PINOUT_VIEW
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text(render_pinout(pins, top=top), encoding="utf-8")
    written = [mp]
    jp = root / ".coresmith" / "prd_spec.json"
    doc: dict = {}
    if jp.exists():
        try:
            doc = json.loads(jp.read_text())
        except ValueError:
            doc = {}
    if not isinstance(doc, dict):
        doc = {}
    prd = doc.get("prd") if isinstance(doc.get("prd"), dict) else None
    pm = pin_map_doc(pins)
    if pm is not None:
        if prd is None:
            prd = doc.setdefault("prd", {})
        prd["pin_map"] = pm
    elif prd is not None and isinstance(prd.get("pin_map"), dict) and prd["pin_map"].get("source") == "pins":
        prd.pop("pin_map")
    else:
        return written
    jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    written.append(jp)
    return written
