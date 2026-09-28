# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""The project ontology (architect sitting, step 1).

Requirements used to live in prose: PRD functional requirements were strings
with an id prefix, the FRD was markdown only, the ERS JSON had ids nobody
queried, and coverage was a sentence a model wrote. This module makes every
registered thing a row:

* ``artifacts``  -- a document/model/harness the architect registered
  (``kind`` prd | sad | frd | ers | block_diagram | contracts | abi |
  uarch:<block> | arch_model | model:<block> | harness), content-hashed;
* ``items``      -- the identified items inside them (``PERF-001``, ``INV-003``,
  ``FR-CPU-2``, ``KPI-ISA-1``, ``VAL-007``, ``ERS-rv_exec-3`` ...) with
  priority / acceptance / model-check text and a lifecycle ``status``;
* ``item_links`` -- ``derives_from`` (FRD -> PRD, ERS -> FRD), ``owned_by``
  (item -> block), ``verified_by`` (item -> check kind), ``cites``;
* ``checks``     -- every tool verdict about an item (model_eval, contract
  validation, elaboration, sim, synth, timing, acceptance), bound to the
  content sha it was computed on;
* ``questions``  -- item-scoped open questions (must_answer blocks the stage
  machine until a ruling answers them);
* ``stages``     -- the deterministic state machine's ledger.

Everything here is langgraph-free and usable from the CLI, the daemon and the
graph nodes alike.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

ONTOLOGY_SCHEMA = """
CREATE TABLE IF NOT EXISTS artifacts (
    kind TEXT PRIMARY KEY,
    path TEXT NOT NULL,
    sha TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    meta_json TEXT,
    registered_by TEXT,
    ts REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS items (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    artifact TEXT NOT NULL,
    section TEXT,
    text TEXT NOT NULL,
    priority TEXT,
    acceptance TEXT,
    model_check TEXT,
    status TEXT NOT NULL DEFAULT 'open',
    extra_json TEXT,
    artifact_sha TEXT,
    ts REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS items_kind ON items(kind);
CREATE INDEX IF NOT EXISTS items_artifact ON items(artifact);
CREATE TABLE IF NOT EXISTS item_links (
    from_id TEXT NOT NULL,
    to_id TEXT NOT NULL,
    rel TEXT NOT NULL,
    source TEXT,
    ts REAL NOT NULL,
    PRIMARY KEY (from_id, to_id, rel)
);
CREATE TABLE IF NOT EXISTS checks (
    id INTEGER PRIMARY KEY,
    item_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    evidence TEXT,
    sha TEXT,
    run_id TEXT,
    actor TEXT,
    ts REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS checks_item ON checks(item_id);
CREATE TABLE IF NOT EXISTS questions (
    id INTEGER PRIMARY KEY,
    item_id TEXT,
    text TEXT NOT NULL,
    must_answer INTEGER NOT NULL DEFAULT 1,
    status TEXT NOT NULL DEFAULT 'open',
    ruling_id INTEGER,
    answer TEXT,
    asked_by TEXT,
    ts REAL NOT NULL,
    answered_ts REAL
);
CREATE TABLE IF NOT EXISTS stages (
    name TEXT PRIMARY KEY,
    ordinal INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    entered_ts REAL,
    done_ts REAL,
    blocked_by_json TEXT,
    ts REAL NOT NULL
);
"""

ITEM_STATUSES = ("open", "verified", "failed", "waived", "not_testable")
CHECK_STATUSES = ("pass", "fail", "not_testable", "skipped", "tool_error")
LINK_RELS = ("derives_from", "owned_by", "verified_by", "cites", "covers")

_ID_RE = re.compile(r"^[A-Z][A-Z0-9]*(?:-[A-Za-z0-9_]+)*-\d+[a-z]?$")


def is_item_id(s: str) -> bool:
    return bool(_ID_RE.match((s or "").strip()))


def file_sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def _row(r) -> dict:
    d = dict(r)
    for k in ("meta_json", "extra_json", "blocked_by_json"):
        if k in d:
            d[k[:-5]] = json.loads(d.pop(k) or "null")
    return d


class OntologyMixin:
    """Mounted on ``ProjectDB`` (needs ``_tx`` / ``_conn``)."""

    # -- artifacts ---------------------------------------------------------
    def register_artifact(self, kind: str, path: str, *, sha: str = "", meta: dict | None = None,
                          registered_by: str = "") -> dict:
        sha = sha or file_sha(Path(self.path).parent.parent / path if not Path(path).is_absolute() else path)
        with self._tx() as db:
            prev = db.execute("SELECT version, sha FROM artifacts WHERE kind=?", (kind,)).fetchone()
            version = 1 if prev is None else (prev["version"] + (0 if prev["sha"] == sha else 1))
            db.execute("INSERT INTO artifacts(kind, path, sha, version, meta_json, registered_by, ts) "
                       "VALUES (?,?,?,?,?,?,?) ON CONFLICT(kind) DO UPDATE SET path=excluded.path, "
                       "sha=excluded.sha, version=excluded.version, meta_json=excluded.meta_json, "
                       "registered_by=excluded.registered_by, ts=excluded.ts",
                       (kind, str(path), sha, version, json.dumps(meta or {}), registered_by, time.time()))
        return self.artifact(kind)

    def artifact(self, kind: str) -> dict | None:
        with self._conn() as db:
            r = db.execute("SELECT * FROM artifacts WHERE kind=?", (kind,)).fetchone()
        return _row(r) if r else None

    def artifacts(self) -> list[dict]:
        with self._conn() as db:
            return [_row(r) for r in db.execute("SELECT * FROM artifacts ORDER BY ts").fetchall()]

    # -- items -------------------------------------------------------------
    def upsert_items(self, artifact: str, items: list[dict], *, artifact_sha: str = "") -> dict:
        """Replace the item set of ``artifact`` (ids that vanished are kept with
        status ``retired``; existing ids keep their status unless the text
        changed, which re-opens them). Malformed ids are skipped and returned
        in ``rejected_ids``. Returns counts."""
        now = time.time()
        seen, added, changed, rejected = set(), 0, 0, []
        with self._tx() as db:
            for it in items:
                iid = str(it["id"]).strip()
                if not is_item_id(iid):
                    rejected.append(iid)
                    continue
                seen.add(iid)
                prev = db.execute("SELECT text, status FROM items WHERE id=?", (iid,)).fetchone()
                status = it.get("status") or (prev["status"] if prev and prev["text"] == it.get("text", "") else "open")
                if prev is None:
                    added += 1
                elif prev["text"] != it.get("text", ""):
                    changed += 1
                db.execute(
                    "INSERT INTO items(id, kind, artifact, section, text, priority, acceptance, model_check, status, "
                    "extra_json, artifact_sha, ts) VALUES (?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET "
                    "kind=excluded.kind, artifact=excluded.artifact, section=excluded.section, text=excluded.text, "
                    "priority=excluded.priority, acceptance=excluded.acceptance, model_check=excluded.model_check, "
                    "status=excluded.status, extra_json=excluded.extra_json, artifact_sha=excluded.artifact_sha, ts=excluded.ts",
                    (iid, it.get("kind") or iid.split("-")[0], artifact, it.get("section") or "", it.get("text") or "",
                     (it.get("priority") or "").lower(), it.get("acceptance") or "", it.get("model_check") or "",
                     status, json.dumps(it.get("extra") or {}), artifact_sha, now))
            if seen:
                db.execute(f"UPDATE items SET status='retired' WHERE artifact=? AND id NOT IN ({','.join('?' * len(seen))})",
                           (artifact, *seen))
            else:
                db.execute("UPDATE items SET status='retired' WHERE artifact=?", (artifact,))
        return {"items": len(seen), "added": added, "changed": changed, "rejected_ids": rejected}

    def items(self, *, kind: str | None = None, artifact: str | None = None, status: str | None = None,
              must_have: bool = False) -> list[dict]:
        q, args = "SELECT * FROM items WHERE 1=1", []
        if kind:
            q += " AND kind=?"
            args.append(kind)
        if artifact:
            q += " AND artifact=?"
            args.append(artifact)
        if status:
            q += " AND status=?"
            args.append(status)
        else:
            q += " AND status!='retired'"
        with self._conn() as db:
            rows = [_row(r) for r in db.execute(q + " ORDER BY artifact, id", args).fetchall()]
        return [r for r in rows if not must_have or item_must_have(r)]

    def item(self, item_id: str) -> dict | None:
        with self._conn() as db:
            r = db.execute("SELECT * FROM items WHERE id=?", (item_id,)).fetchone()
        return _row(r) if r else None

    def set_item_status(self, item_id: str, status: str) -> None:
        if status not in ITEM_STATUSES + ("retired",):
            raise ValueError(f"bad item status {status!r}")
        with self._tx() as db:
            db.execute("UPDATE items SET status=? WHERE id=?", (status, item_id))

    # -- links -------------------------------------------------------------
    def link_items(self, from_id: str, to_id: str, rel: str, *, source: str = "") -> None:
        if rel not in LINK_RELS:
            raise ValueError(f"bad link rel {rel!r}: {LINK_RELS}")
        with self._tx() as db:
            db.execute("INSERT OR REPLACE INTO item_links(from_id, to_id, rel, source, ts) VALUES (?,?,?,?,?)",
                       (from_id, to_id, rel, source, time.time()))

    def links(self, *, from_id: str | None = None, to_id: str | None = None, rel: str | None = None) -> list[dict]:
        q, args = "SELECT * FROM item_links WHERE 1=1", []
        for col, v in (("from_id", from_id), ("to_id", to_id), ("rel", rel)):
            if v:
                q += f" AND {col}=?"
                args.append(v)
        with self._conn() as db:
            return [dict(r) for r in db.execute(q + " ORDER BY ts", args).fetchall()]

    # -- checks ------------------------------------------------------------
    def add_check(self, item_id: str, kind: str, status: str, *, evidence: str = "", sha: str = "",
                  run_id: str = "", actor: str = "") -> int:
        if status not in CHECK_STATUSES:
            raise ValueError(f"bad check status {status!r}: {CHECK_STATUSES}")
        with self._tx() as db:
            cur = db.execute("INSERT INTO checks(item_id, kind, status, evidence, sha, run_id, actor, ts) "
                             "VALUES (?,?,?,?,?,?,?,?)", (item_id, kind, status, evidence, sha, run_id, actor, time.time()))
            new = {"pass": "verified", "fail": "failed", "not_testable": "not_testable"}.get(status)
            if new:
                db.execute("UPDATE items SET status=? WHERE id=? AND status!='waived'", (new, item_id))
            return int(cur.lastrowid)

    def checks(self, item_id: str | None = None, *, kind: str | None = None, latest: bool = False) -> list[dict]:
        q, args = "SELECT * FROM checks WHERE 1=1", []
        if item_id:
            q += " AND item_id=?"
            args.append(item_id)
        if kind:
            q += " AND kind=?"
            args.append(kind)
        with self._conn() as db:
            rows = [dict(r) for r in db.execute(q + " ORDER BY ts", args).fetchall()]
        if latest:
            last: dict[tuple, dict] = {}
            for r in rows:
                last[(r["item_id"], r["kind"])] = r
            rows = list(last.values())
        return rows

    def latest_check(self, item_id: str, kind: str | None = None) -> dict | None:
        rows = self.checks(item_id, kind=kind)
        return rows[-1] if rows else None

    # -- questions ---------------------------------------------------------
    def add_question(self, text: str, *, item_id: str = "", must_answer: bool = True, asked_by: str = "") -> int:
        with self._tx() as db:
            cur = db.execute("INSERT INTO questions(item_id, text, must_answer, status, asked_by, ts) VALUES (?,?,?,?,?,?)",
                             (item_id or None, text, int(must_answer), "open", asked_by, time.time()))
            return int(cur.lastrowid)

    def answer_question(self, qid: int, answer: str, *, ruling_id: int | None = None) -> bool:
        with self._tx() as db:
            cur = db.execute("UPDATE questions SET status='answered', answer=?, ruling_id=?, answered_ts=? "
                             "WHERE id=? AND status='open'", (answer, ruling_id, time.time(), qid))
            return cur.rowcount > 0

    def questions(self, *, open_only: bool = True, must_answer: bool | None = None) -> list[dict]:
        q, args = "SELECT * FROM questions WHERE 1=1", []
        if open_only:
            q += " AND status='open'"
        if must_answer is not None:
            q += " AND must_answer=?"
            args.append(int(must_answer))
        with self._conn() as db:
            return [dict(r) for r in db.execute(q + " ORDER BY id", args).fetchall()]

    # -- stages ------------------------------------------------------------
    def stage_rows(self) -> list[dict]:
        with self._conn() as db:
            return [_row(r) for r in db.execute("SELECT * FROM stages ORDER BY ordinal").fetchall()]

    def stage_set(self, name: str, ordinal: int, status: str, *, blocked_by: list | None = None) -> None:
        now = time.time()
        with self._tx() as db:
            prev = db.execute("SELECT entered_ts FROM stages WHERE name=?", (name,)).fetchone()
            entered = (prev["entered_ts"] if prev else None) or (now if status in ("active", "done") else None)
            db.execute("INSERT INTO stages(name, ordinal, status, entered_ts, done_ts, blocked_by_json, ts) "
                       "VALUES (?,?,?,?,?,?,?) ON CONFLICT(name) DO UPDATE SET status=excluded.status, "
                       "entered_ts=excluded.entered_ts, done_ts=excluded.done_ts, blocked_by_json=excluded.blocked_by_json, "
                       "ts=excluded.ts",
                       (name, ordinal, status, entered, now if status == "done" else None,
                        json.dumps(blocked_by or []), now))


def item_must_have(item: dict) -> bool:
    p = (item.get("priority") or "").lower()
    txt = item.get("text") or ""
    return "must" in p or p in ("hard", "p0", "required") or "[HARD" in txt
