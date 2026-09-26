# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Operator rulings: a sanctioned, additive channel for run-time policy (C2).

The chip lead was told to read ``inputs/OPERATOR_RULINGS.md`` -- but nothing
read it, and ADDING it mid-run is oracle tampering (``trust.py`` hashes
everything under ``inputs/``). In the SoC benchmark 20 ABI questions sat
unanswered for that reason.

A ruling is a row: scope (``global`` | ``arch`` | ``block:<name>`` |
``edge:<edge_id>``), text, rationale, source, optional ``question_ref``
(``interrupt:<id>`` | ``prd:<qid>`` | ``block:<name>:<kind>``) and a
supersedes chain. Rulings never touch ``inputs/`` or the frozen specs:

* every consumer prompt gets the same deterministic
  "## Operator rulings (binding)" section (``render_rulings_section``), and
  each use is ledgered in ``ruling_uses``;
* block-scoped rulings also materialize as persistent ``constraints`` rows
  (source ``operator_ruling``);
* a ruling with a ``question_ref`` resolves the matching pending interrupt
  (``apply_ruling_to_interrupts``) so the parked branch auto-resumes;
* ``.coresmith/OPERATOR_RULINGS.md`` is a read-only VIEW outside ``inputs/``;
* a soft validator flags (never blocks) a ruling whose numbers contradict
  ``inputs/requirements.md``.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

CONSTRAINT_SOURCE = "operator_ruling"
_SCOPE_RE = re.compile(r"^(global|arch|block:[A-Za-z0-9_\-]+|edge:[A-Za-z0-9_\-:.]+)$")
_NUM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(MHz|GHz|kHz|ns|ps|us|ms|KiB|MiB|GiB|KB|MB|GB|bits?|bytes?|B|fps|%|cycles?|mm2|um2|MB/s|GB/s)\b",
                     re.IGNORECASE)


def _row(r) -> dict:
    d = dict(r)
    d["conflicts"] = json.loads(d.pop("conflict_json") or "[]")
    return d


def validate_scope(scope: str) -> str:
    scope = (scope or "").strip()
    if not _SCOPE_RE.match(scope):
        raise ValueError(f"bad ruling scope {scope!r}: global | arch | block:<name> | edge:<edge_id>")
    return scope


def _requirement_quantities(project_root: Path) -> dict[str, set[str]]:
    """unit -> values mentioned in inputs/requirements.md (lowercased unit)."""
    req = project_root / "inputs" / "requirements.md"
    out: dict[str, set[str]] = {}
    try:
        text = req.read_text(encoding="utf-8")
    except OSError:
        return out
    for value, unit in _NUM_RE.findall(text):
        out.setdefault(unit.lower(), set()).add(value)
    return out


def check_conflicts(text: str, project_root: str | Path) -> list[dict]:
    """Quantities in ``text`` whose unit appears in requirements.md with a
    DIFFERENT value. Advisory: the ruling is recorded either way."""
    known = _requirement_quantities(Path(project_root))
    out = []
    for value, unit in _NUM_RE.findall(text or ""):
        vals = known.get(unit.lower())
        if vals and value not in vals:
            out.append({"unit": unit, "ruling_value": value,
                        "requirement_values": sorted(vals)})
    return out


class RulingMixin:
    """Ruling primitives mixed into ``ProjectDB``."""

    def add_ruling(self, scope: str, text: str, *, rationale: str = "",
                   source: str = "human", question_ref: str | None = None,
                   supersedes_id: int | None = None) -> int:
        scope = validate_scope(scope)
        text = (text or "").strip()
        if not text:
            raise ValueError("a ruling needs text")
        conflicts = check_conflicts(text, self.root)
        with self._tx() as db:
            cur = db.execute(
                "INSERT INTO rulings(scope, question_ref, text, rationale, source, ts, "
                "supersedes_id, conflict_json) VALUES (?,?,?,?,?,?,?,?)",
                (scope, question_ref, text, rationale or "", source, time.time(),
                 supersedes_id, json.dumps(conflicts)))
            rid = int(cur.lastrowid)
        if scope.startswith("block:"):
            block = scope.split(":", 1)[1]
            self.add_constraint(block, text, source=CONSTRAINT_SOURCE,
                                ruling_id=rid, rationale=rationale or "")
        return rid

    def revoke_ruling(self, ruling_id: int, reason: str) -> bool:
        with self._tx() as db:
            cur = db.execute("UPDATE rulings SET revoked_ts=?, revoked_reason=? WHERE id=? "
                             "AND revoked_ts IS NULL", (time.time(), reason, ruling_id))
            ok = cur.rowcount == 1
            if ok:
                db.execute("DELETE FROM constraints WHERE source=? AND "
                           "json_extract(extra_json, '$.ruling_id')=?",
                           (CONSTRAINT_SOURCE, ruling_id))
        return ok

    def ruling(self, ruling_id: int) -> dict | None:
        with self._conn() as db:
            row = db.execute("SELECT * FROM rulings WHERE id=?", (ruling_id,)).fetchone()
        return _row(row) if row else None

    def rulings(self, *, scope: str | None = None, active_only: bool = True) -> list[dict]:
        sql, args = "SELECT * FROM rulings", []
        conds = []
        if scope:
            conds.append("scope=?")
            args.append(scope)
        if active_only:
            conds.append("revoked_ts IS NULL")
        if conds:
            sql += " WHERE " + " AND ".join(conds)
        with self._conn() as db:
            rows = [_row(r) for r in db.execute(sql + " ORDER BY id", tuple(args)).fetchall()]
            if active_only:
                # A supersede is permanent: revoking the replacement leaves NO
                # ruling, it does not resurrect the one it replaced.
                superseded = {r["supersedes_id"] for r in db.execute(
                    "SELECT supersedes_id FROM rulings WHERE supersedes_id IS NOT NULL")}
                rows = [r for r in rows if r["id"] not in superseded]
        return rows

    def rulings_for(self, *, block: str | None = None, edge_ids=(), arch: bool = False) -> list[dict]:
        """Active rulings applicable to a consumer, most specific last."""
        scopes = ["global"] + (["arch"] if arch else [])
        if block:
            scopes.append(f"block:{block}")
        scopes += [f"edge:{e}" for e in (edge_ids or ())]
        rank = {s: i for i, s in enumerate(scopes)}
        out = [r for r in self.rulings() if r["scope"] in rank]
        out.sort(key=lambda r: (rank[r["scope"]], r["id"]))
        return out

    def record_ruling_uses(self, rulings: list[dict], *, consumer: str, block: str | None = None,
                           node: str | None = None, attempt: int | None = None) -> None:
        if not rulings:
            return
        rid = self.run_id()
        with self._tx() as db:
            for r in rulings:
                db.execute("INSERT INTO ruling_uses(ruling_id, consumer, block, node, attempt, "
                           "run_id, ts) VALUES (?,?,?,?,?,?,?)",
                           (r["id"], consumer, block or "", node or "", attempt, rid, time.time()))

    def ruling_uses(self, ruling_id: int | None = None) -> list[dict]:
        with self._conn() as db:
            if ruling_id is None:
                rows = db.execute("SELECT * FROM ruling_uses ORDER BY id").fetchall()
            else:
                rows = db.execute("SELECT * FROM ruling_uses WHERE ruling_id=? ORDER BY id",
                                  (ruling_id,)).fetchall()
        return [dict(r) for r in rows]

    def export_rulings_view(self) -> Path:
        """``.coresmith/OPERATOR_RULINGS.md``: the read-only view every prompt
        and the chip lead are pointed at (never under ``inputs/``)."""
        target = self.path.parent / "OPERATOR_RULINGS.md"
        active = self.rulings()
        revoked = [r for r in self.rulings(active_only=False) if r.get("revoked_ts")]
        lines = ["# Operator rulings (binding)", "",
                 "Read-only view of the `rulings` table. Add with "
                 "`coresmith ruling add --scope S --text T`.", ""]
        by_scope: dict[str, list[dict]] = {}
        for r in active:
            by_scope.setdefault(r["scope"], []).append(r)
        for scope in sorted(by_scope, key=lambda s: (s not in ("global", "arch"), s)):
            lines.append(f"## {scope}")
            for r in by_scope[scope]:
                lines.append(f"- [R{r['id']}] {r['text']}" +
                             (f" — _{r['rationale']}_" if r.get("rationale") else "") +
                             (f" (question: {r['question_ref']})" if r.get("question_ref") else "") +
                             (f" ⚠ conflicts: {r['conflicts']}" if r.get("conflicts") else ""))
            lines.append("")
        if not active:
            lines += ["(none)", ""]
        if revoked:
            lines.append("## Revoked")
            for r in revoked:
                lines.append(f"- [R{r['id']}] {r['text']} — revoked: {r.get('revoked_reason', '')}")
            lines.append("")
        self._write_text_view(target, "\n".join(lines))
        return target


def render_rulings_section(db, *, consumer: str, block: str | None = None,
                           edge_ids=(), arch: bool = False, node: str | None = None,
                           attempt: int | None = None) -> str:
    """The binding section appended to a consumer prompt; ledgers each use.
    Returns "" when nothing applies (so prompts are byte-identical to before)."""
    try:
        rulings = db.rulings_for(block=block, edge_ids=edge_ids, arch=arch)
    except Exception:  # noqa: BLE001 - never break a prompt over bookkeeping
        return ""
    if not rulings:
        return ""
    lines = ["", "## Operator rulings (binding)",
             "These override the chip-lead's defaults. They never override a "
             "frozen requirement; if one seems to, say so instead of complying.", ""]
    for r in rulings:
        line = f"- [R{r['id']}, {r['scope']}] {r['text']}"
        if r.get("rationale"):
            line += f" — {r['rationale']}"
        lines.append(line)
    lines.append("")
    try:
        db.record_ruling_uses(rulings, consumer=consumer, block=block, node=node, attempt=attempt)
    except Exception:  # noqa: BLE001
        pass
    return "\n".join(lines)


def rulings_section(project_root: str | Path | None, **kw) -> str:
    """``render_rulings_section`` for callers that only have a project root."""
    pr = str(project_root or "").strip()
    if not pr or pr == ".":
        return ""
    try:
        from orchestrator.state_store.project_db import open_project
        db = open_project(pr)
    except Exception:  # noqa: BLE001
        return ""
    return render_rulings_section(db, **kw)


def rulings_section_env(**kw) -> str:
    """``rulings_section`` against ``$CORESMITH_PROJECT_ROOT``."""
    import os
    return rulings_section(os.environ.get("CORESMITH_PROJECT_ROOT", ""), **kw)


def apply_ruling_to_interrupts(db, ruling: dict) -> list[str]:
    """Resolve pending interrupts a ruling answers. Returns the ids resolved.

    ``interrupt:<id>`` resolves that row (block parks get ``add_constraint``,
    others ``continue`` with the text as feedback). ``prd:<qid>`` fills one
    answer of a pending ``prd_questions`` park; the park resolves once every
    question id it lists has an answer. ``prd:*`` answers every remaining
    question with the ruling text. ``block:<name>:<kind>`` resolves the pending
    park of that block and kind with ``add_constraint``.
    """
    ref = str(ruling.get("question_ref") or "").strip()
    text = ruling["text"]
    by = f"ruling:{ruling['id']}"
    if not ref:
        return []
    resolved: list[str] = []
    pending = db.interrupts(status="pending")
    if ref.startswith("interrupt:"):
        iid = ref.split(":", 1)[1]
        row = db.interrupt(iid)
        if row and row["status"] == "pending":
            res = _resolution_for(row, text, ruling)
            if db.resolve_interrupt(iid, res, resolved_by=by):
                resolved.append(iid)
        return resolved
    if ref.startswith("prd:"):
        qid = ref.split(":", 1)[1]
        for row in pending:
            if row["kind"] != "prd_questions":
                continue
            qids = _question_ids(row["payload"])
            partial = dict((row.get("resolution") or {}).get("answers") or {}) \
                if row.get("resolution") else _partial_answers(db, row["id"])
            if qid == "*":
                for q in qids:
                    partial.setdefault(q, text)
            elif qid in qids:
                partial[qid] = text
            else:
                continue
            _store_partial(db, row["id"], partial)
            if qids and all(q in partial for q in qids):
                if db.resolve_interrupt(row["id"], {"action": "continue", "answers": partial,
                                                    "feedback": "", "rationale": by},
                                        resolved_by=by):
                    resolved.append(row["id"])
        return resolved
    if ref.startswith("block:"):
        parts = ref.split(":")
        block = parts[1] if len(parts) > 1 else ""
        kind = parts[2] if len(parts) > 2 else ""
        for row in pending:
            if row["block"] == block and (not kind or row["kind"] == kind):
                if db.resolve_interrupt(row["id"], _resolution_for(row, text, ruling), resolved_by=by):
                    resolved.append(row["id"])
        return resolved
    return resolved


def _resolution_for(row: dict, text: str, ruling: dict) -> dict:
    supported = list((row.get("payload") or {}).get("supported_actions") or [])
    if "add_constraint" in supported:
        return {"action": "add_constraint", "constraint": text, "feedback": text,
                "rationale": f"ruling:{ruling['id']}", "block_actions": {}}
    action = "continue" if ("continue" in supported or not supported) else supported[0]
    return {"action": action, "feedback": text, "rationale": f"ruling:{ruling['id']}",
            "block_actions": {}}


def _question_ids(payload: dict) -> list[str]:
    qs = payload.get("questions") or payload.get("question_ids") or []
    out = []
    for q in qs:
        if isinstance(q, dict):
            qid = q.get("id") or q.get("question_id") or q.get("key")
            if qid:
                out.append(str(qid))
        elif q:
            out.append(str(q))
    return out


def _partial_key(iid: str) -> str:
    return f"prd_partial:{iid}"


def _partial_answers(db, iid: str) -> dict:
    v = db.get_flag(_partial_key(iid), {})
    return dict(v) if isinstance(v, dict) else {}


def _store_partial(db, iid: str, answers: dict) -> None:
    db.set_flag(_partial_key(iid), answers)
