# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""The project ontology (the Architect's step 1).

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
* ``stages``     -- the deterministic state machine's ledger;
* ``verifiers``  -- how an item is verified (a cocotb test, a python check, a
  SystemC model probe, an LLM judge, a manual sign-off), per item/block;
* ``actions``    -- the audit log of every CLI verb that went through
  ``orchestrator.harness.cli._run`` (argv, exit code, one-line summary).

Items may carry a measurable ``metric`` with ``bound_min`` / ``bound_max`` /
``unit``; a check that records a numeric ``value`` then derives its own
pass/fail from those bounds.

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
    ts REAL NOT NULL,
    metric TEXT,
    bound_min REAL,
    bound_max REAL,
    unit TEXT
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
    ts REAL NOT NULL,
    value REAL
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
CREATE TABLE IF NOT EXISTS verifiers (
    id INTEGER PRIMARY KEY,
    item_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    path TEXT,
    entry TEXT,
    args_json TEXT,
    block TEXT,
    ts REAL NOT NULL,
    UNIQUE(item_id, kind, path, entry)
);
CREATE INDEX IF NOT EXISTS verifiers_item ON verifiers(item_id);
CREATE TABLE IF NOT EXISTS actions (
    id INTEGER PRIMARY KEY,
    ts REAL NOT NULL,
    actor TEXT,
    argv_json TEXT NOT NULL,
    rc INTEGER,
    summary TEXT,
    run_id TEXT
);
"""

# Columns added after the first release of the tables above; ``ensure_schema``
# ALTERs them onto databases created before they existed.
ONTOLOGY_ADDED_COLUMNS = (
    ("items", "metric", "TEXT"),
    ("items", "bound_min", "REAL"),
    ("items", "bound_max", "REAL"),
    ("items", "unit", "TEXT"),
    ("checks", "value", "REAL"),
)

ITEM_STATUSES = ("open", "verified", "failed", "waived", "not_testable")
CHECK_STATUSES = ("pass", "fail", "not_testable", "skipped", "tool_error")
LINK_RELS = ("derives_from", "owned_by", "verified_by", "cites", "covers")
# ``chip``: a test of the chip-level integration / validation testbench
# (stamped as ``integration_dv`` / ``validation_dv`` after the chip simulation).
# ``eda``: a quantity the engine's tools measure on a module build's
# synthesized netlist (``entry`` = ``area_um2`` | ``power_mw``; see
# ``state_store.module_targets``).
VERIFIER_KINDS = ("cocotb", "python", "systemc", "judge", "manual", "chip", "eda")
# Authority of a check kind for an item's derived status: the latest check of
# the highest-ranked kind present decides (a model pass never overrides an RTL
# fail; an RTL pass after an RTL fail is a pass). Unknown kinds rank 2 (RTL-level).
CHECK_KIND_RANK = {"model_eval": 1, "block_dv": 2, "integration_dv": 3, "validation_dv": 4, "acceptance": 4,
                   "signoff": 5}
DEFAULT_CHECK_RANK = 2
_STATUS_OF_CHECK = {"pass": "verified", "fail": "failed", "not_testable": "not_testable"}


class CheckStatusConflict(ValueError):
    """An explicit check status contradicts the status derived from the value and the item's bounds."""

    def __init__(self, item_id: str, given: str, derived: str, value: float):
        self.item_id, self.given, self.derived, self.value = item_id, given, derived, value
        super().__init__(f"CHECK_STATUS_CONFLICT: {item_id} value {value:g} derives {derived!r} from the item's "
                         f"bounds; explicit status {given!r} contradicts it")


def check_rank(kind: str) -> int:
    return CHECK_KIND_RANK.get(str(kind or ""), DEFAULT_CHECK_RANK)


def derive_status_from_bounds(item: dict | None, value: float) -> str | None:
    """``pass``/``fail`` from the item's [bound_min, bound_max] (a missing side is
    open); None when the item has no bounds."""
    lo = item.get("bound_min") if item else None
    hi = item.get("bound_max") if item else None
    if lo is None and hi is None:
        return None
    return "pass" if (lo is None or value >= lo) and (hi is None or value <= hi) else "fail"


def derived_item_status(checks: list[dict]) -> str | None:
    """The item status the checks imply (None: no checks). The latest check of
    the highest-ranked kind decides; a skipped/tool_error verdict at that rank
    leaves the item ``open`` (not judged at the authoritative level)."""
    if not checks:
        return None
    top = max(check_rank(c["kind"]) for c in checks)
    latest = max((c for c in checks if check_rank(c["kind"]) == top), key=lambda c: (c["ts"], c["id"]))
    return _STATUS_OF_CHECK.get(latest["status"], "open")


ITEM_EDIT_FIELDS = ("text", "priority", "acceptance", "model_check", "metric", "bound_min", "bound_max",
                    "unit", "section")

_ID_RE = re.compile(r"^[A-Z][A-Z0-9]*(?:-[A-Za-z0-9_]+)*-\d+[a-z]?$")


def is_item_id(s: str) -> bool:
    return bool(_ID_RE.match((s or "").strip()))


def file_sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def _opt_float(v) -> float | None:
    if v is None or v == "":
        return None
    return float(v)


def _opt_str(v) -> str | None:
    if v is None:
        return None
    return str(v)


_ITEM_UPSERT_SQL = (
    "INSERT INTO items(id, kind, artifact, section, text, priority, acceptance, model_check, status, "
    "extra_json, artifact_sha, ts, metric, bound_min, bound_max, unit) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
    "ON CONFLICT(id) DO UPDATE SET "
    "kind=excluded.kind, artifact=excluded.artifact, section=excluded.section, text=excluded.text, "
    "priority=excluded.priority, acceptance=excluded.acceptance, model_check=excluded.model_check, "
    "status=excluded.status, extra_json=excluded.extra_json, artifact_sha=excluded.artifact_sha, ts=excluded.ts, "
    "metric=COALESCE(excluded.metric, items.metric), bound_min=COALESCE(excluded.bound_min, items.bound_min), "
    "bound_max=COALESCE(excluded.bound_max, items.bound_max), unit=COALESCE(excluded.unit, items.unit)")


def _upsert_item_row(db, artifact: str, it: dict, artifact_sha: str, now: float) -> str:
    """Insert/update one item inside an open transaction; returns ``added`` |
    ``changed`` | ``same``. A None metric/bound/unit keeps the stored value."""
    iid = str(it["id"]).strip()
    prev = db.execute("SELECT text, status FROM items WHERE id=?", (iid,)).fetchone()
    text = it.get("text") or ""
    status = it.get("status") or (prev["status"] if prev and prev["text"] == text else "open")
    db.execute(_ITEM_UPSERT_SQL,
               (iid, it.get("kind") or iid.split("-")[0], artifact, it.get("section") or "", text,
                (it.get("priority") or "").lower(), it.get("acceptance") or "", it.get("model_check") or "",
                status, json.dumps(it.get("extra") or {}), artifact_sha, now,
                _opt_str(it.get("metric")), _opt_float(it.get("bound_min")), _opt_float(it.get("bound_max")),
                _opt_str(it.get("unit"))))
    if prev is None:
        return "added"
    return "changed" if prev["text"] != text else "same"


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
        in ``rejected_ids``; an id a *different* artifact already owns (live)
        is neither moved nor rewritten and is returned in ``owned_elsewhere``.
        Returns counts."""
        now = time.time()
        seen, added, changed, rejected, elsewhere = set(), 0, 0, [], []
        with self._tx() as db:
            for it in items:
                iid = str(it["id"]).strip()
                if not is_item_id(iid):
                    rejected.append(iid)
                    continue
                prev = db.execute("SELECT artifact, status FROM items WHERE id=?", (iid,)).fetchone()
                if prev is not None and prev["artifact"] != artifact and prev["status"] != "retired":
                    # another artifact owns this id (e.g. an FRD INV-001 cited by the
                    # ERS): never move or rewrite it -- the caller still adds links
                    if iid not in elsewhere:
                        elsewhere.append(iid)
                    continue
                seen.add(iid)
                what = _upsert_item_row(db, artifact, {**it, "id": iid}, artifact_sha, now)
                added += what == "added"
                changed += what == "changed"
            if seen:
                db.execute(f"UPDATE items SET status='retired' WHERE artifact=? AND id NOT IN ({','.join('?' * len(seen))})",
                           (artifact, *seen))
            else:
                db.execute("UPDATE items SET status='retired' WHERE artifact=?", (artifact,))
        return {"items": len(seen), "added": added, "changed": changed, "rejected_ids": rejected,
                "owned_elsewhere": [{"id": i, "artifact": self.item(i)["artifact"]} for i in elsewhere]}

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

    def upsert_item(self, artifact: str, item: dict) -> dict:
        """Add or update ONE item of ``artifact`` (siblings are left alone,
        unlike :meth:`upsert_items`). Malformed id -> ValueError."""
        iid = str(item.get("id") or "").strip()
        if not is_item_id(iid):
            raise ValueError(f"malformed item id {iid!r}")
        with self._tx() as db:
            _upsert_item_row(db, artifact, {**item, "id": iid}, str(item.get("artifact_sha") or ""), time.time())
        return self.item(iid)

    def edit_item(self, item_id: str, **fields) -> dict:
        """Change individual fields of an item. A text change re-opens the item
        (unless it is waived). Unknown field -> ValueError; missing item -> KeyError."""
        bad = [k for k in fields if k not in ITEM_EDIT_FIELDS]
        if bad:
            raise ValueError(f"unknown item field(s) {bad}: {ITEM_EDIT_FIELDS}")
        with self._tx() as db:
            prev = db.execute("SELECT text, status FROM items WHERE id=?", (item_id,)).fetchone()
            if prev is None:
                raise KeyError(item_id)
            sets, args = [], []
            for k, v in fields.items():
                if k in ("bound_min", "bound_max"):
                    v = _opt_float(v)
                elif k == "priority":
                    v = (v or "").lower()
                elif k in ("metric", "unit"):
                    v = _opt_str(v)
                else:
                    v = "" if v is None else str(v)
                sets.append(f"{k}=?")
                args.append(v)
            if "text" in fields and (fields["text"] or "") != prev["text"] and prev["status"] != "waived":
                sets.append("status='open'")
            if sets:
                sets.append("ts=?")
                args.append(time.time())
                db.execute(f"UPDATE items SET {', '.join(sets)} WHERE id=?", (*args, item_id))
        return self.item(item_id)

    def retire_item(self, item_id: str) -> bool:
        with self._tx() as db:
            cur = db.execute("UPDATE items SET status='retired' WHERE id=?", (item_id,))
            return cur.rowcount > 0

    def ensure_db_artifact(self, kind: str, *, registered_by: str = "cli") -> dict:
        """An artifact row for items authored straight into the database
        (path ``db:<kind>``); an existing row is returned unchanged."""
        with self._tx() as db:
            db.execute("INSERT OR IGNORE INTO artifacts(kind, path, sha, version, meta_json, registered_by, ts) "
                       "VALUES (?,?,?,?,?,?,?)", (kind, f"db:{kind}", "db", 1, "{}", registered_by, time.time()))
        return self.artifact(kind)

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

    def unlink_items(self, from_id: str, to_id: str | None = None, rel: str | None = None) -> int:
        """Delete links out of ``from_id`` (optionally only to ``to_id`` / of
        ``rel``); a ``to_id`` ending in ``*`` matches by prefix. Returns the count."""
        q, args = "DELETE FROM item_links WHERE from_id=?", [from_id]
        if to_id and to_id.endswith("*"):
            q += " AND substr(to_id, 1, ?)=?"
            args += [len(to_id) - 1, to_id[:-1]]
        elif to_id:
            q += " AND to_id=?"
            args.append(to_id)
        if rel:
            q += " AND rel=?"
            args.append(rel)
        with self._tx() as db:
            return db.execute(q, args).rowcount

    def links(self, *, from_id: str | None = None, to_id: str | None = None, rel: str | None = None) -> list[dict]:
        q, args = "SELECT * FROM item_links WHERE 1=1", []
        for col, v in (("from_id", from_id), ("to_id", to_id), ("rel", rel)):
            if v:
                q += f" AND {col}=?"
                args.append(v)
        with self._conn() as db:
            return [dict(r) for r in db.execute(q + " ORDER BY ts", args).fetchall()]

    # -- checks ------------------------------------------------------------
    def add_check(self, item_id: str, kind: str, status: str | None = None, *, evidence: str = "", sha: str = "",
                  run_id: str = "", actor: str = "", value: float | None = None) -> int:
        """Record a verdict. With a numeric ``value`` and an item with bounds
        the status is derived (pass iff within [bound_min, bound_max], a
        missing bound is open): an explicit status that contradicts it raises
        :class:`CheckStatusConflict` and nothing is written. A value on an
        item without bounds needs an explicit status. The item's status is then
        recomputed from all its checks (:func:`derived_item_status`)."""
        if status is not None and not str(status).strip():
            status = None
        if value is not None:
            value = float(value)
            derived = derive_status_from_bounds(self.item(item_id), value)
            if derived is None and not status:
                raise ValueError(f"item {item_id} has no bounds; give a status")
            if derived is not None:
                if status and status != derived:
                    raise CheckStatusConflict(item_id, status, derived, value)
                status = derived
        if status not in CHECK_STATUSES:
            raise ValueError(f"bad check status {status!r}: {CHECK_STATUSES}")
        with self._tx() as db:
            cur = db.execute("INSERT INTO checks(item_id, kind, status, evidence, sha, run_id, actor, ts, value) "
                             "VALUES (?,?,?,?,?,?,?,?,?)",
                             (item_id, kind, status, evidence, sha, run_id, actor, time.time(), value))
            self._recompute_item_status(db, item_id)
            return int(cur.lastrowid)

    @staticmethod
    def _recompute_item_status(db, item_id: str) -> str | None:
        """Re-derive one item's status from its checks inside an open
        transaction (waived/retired items are left alone)."""
        rows = [dict(r) for r in db.execute("SELECT id, kind, status, ts FROM checks WHERE item_id=?", (item_id,))]
        new = derived_item_status(rows)
        if new:
            db.execute("UPDATE items SET status=? WHERE id=? AND status NOT IN ('waived', 'retired')", (new, item_id))
        return new

    def recompute_item_status(self, item_id: str) -> str | None:
        with self._tx() as db:
            return self._recompute_item_status(db, item_id)

    def latest_checks_by_kind(self, item_id: str) -> list[dict]:
        """The latest check of every kind recorded for ``item_id``, highest rank first."""
        rows = self.checks(item_id, latest=True)
        return sorted(rows, key=lambda c: (-check_rank(c["kind"]), c["kind"]))

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

    # -- verifiers ---------------------------------------------------------
    def add_verifier(self, item_id: str, kind: str, *, path: str = "", entry: str = "", args: dict | list | None = None,
                     block: str = "") -> int:
        """Attach a verifier to an item (upsert on item/kind/path/entry) and
        link ``item -verified_by-> verifier:<id>``. Returns the verifier id."""
        if kind not in VERIFIER_KINDS:
            raise ValueError(f"bad verifier kind {kind!r}: {VERIFIER_KINDS}")
        now = time.time()
        with self._tx() as db:
            db.execute("INSERT INTO verifiers(item_id, kind, path, entry, args_json, block, ts) VALUES (?,?,?,?,?,?,?) "
                       "ON CONFLICT(item_id, kind, path, entry) DO UPDATE SET args_json=excluded.args_json, "
                       "block=excluded.block, ts=excluded.ts",
                       (item_id, kind, path or "", entry or "", json.dumps(args if args is not None else {}),
                        block or "", now))
            vid = int(db.execute("SELECT id FROM verifiers WHERE item_id=? AND kind=? AND path=? AND entry=?",
                                 (item_id, kind, path or "", entry or "")).fetchone()["id"])
            db.execute("INSERT OR REPLACE INTO item_links(from_id, to_id, rel, source, ts) VALUES (?,?,?,?,?)",
                       (item_id, f"verifier:{vid}", "verified_by", "cli", now))
        return vid

    @staticmethod
    def _verifier_row(r) -> dict:
        d = dict(r)
        d["args"] = json.loads(d.pop("args_json") or "null")
        return d

    def verifiers(self, *, item_id: str | None = None, block: str | None = None, kind: str | None = None) -> list[dict]:
        q, args = "SELECT * FROM verifiers WHERE 1=1", []
        for col, v in (("item_id", item_id), ("block", block), ("kind", kind)):
            if v:
                q += f" AND {col}=?"
                args.append(v)
        with self._conn() as db:
            return [self._verifier_row(r) for r in db.execute(q + " ORDER BY id", args).fetchall()]

    def verifier(self, vid: int) -> dict | None:
        with self._conn() as db:
            r = db.execute("SELECT * FROM verifiers WHERE id=?", (int(vid),)).fetchone()
        return self._verifier_row(r) if r else None

    def remove_verifier(self, vid: int) -> bool:
        with self._tx() as db:
            cur = db.execute("DELETE FROM verifiers WHERE id=?", (int(vid),))
            db.execute("DELETE FROM item_links WHERE to_id=? AND rel='verified_by'", (f"verifier:{int(vid)}",))
            return cur.rowcount > 0

    # -- actions (CLI audit log) ------------------------------------------
    def record_action(self, argv: list[str], rc: int | None, *, summary: str = "", actor: str = "cli") -> int:
        run_id = ""
        getter = getattr(self, "get_setting", None)
        if getter is not None:
            run_id = getter("run_id", "") or ""
        with self._tx() as db:
            cur = db.execute("INSERT INTO actions(ts, actor, argv_json, rc, summary, run_id) VALUES (?,?,?,?,?,?)",
                             (time.time(), actor, json.dumps([str(a) for a in argv]),
                              None if rc is None else int(rc), summary or "", run_id))
            return int(cur.lastrowid)

    def actions_count(self, *, since_id: int | None = None) -> int:
        with self._conn() as db:
            if since_id is not None:
                return int(db.execute("SELECT COUNT(*) FROM actions WHERE id>?", (int(since_id),)).fetchone()[0])
            return int(db.execute("SELECT COUNT(*) FROM actions").fetchone()[0])

    def actions(self, *, limit: int | None = 200, since_id: int | None = None) -> list[dict]:
        """Recorded CLI actions, ascending by id: the ``limit`` most recent, or
        the first ``limit`` after ``since_id`` (``limit`` None: all of them)."""
        if limit is None:
            limit = -1   # SQLite: no limit
        with self._conn() as db:
            if since_id is not None:
                rows = db.execute("SELECT * FROM actions WHERE id>? ORDER BY id LIMIT ?",
                                  (int(since_id), int(limit))).fetchall()
            else:
                rows = db.execute("SELECT * FROM (SELECT * FROM actions ORDER BY id DESC LIMIT ?) ORDER BY id",
                                  (int(limit),)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["argv"] = json.loads(d.pop("argv_json") or "[]")
            out.append(d)
        return out

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

    def question(self, qid: int) -> dict | None:
        with self._conn() as db:
            r = db.execute("SELECT * FROM questions WHERE id=?", (int(qid),)).fetchone()
        return dict(r) if r else None

    def set_question_ruling(self, qid: int, ruling_id: int) -> None:
        with self._tx() as db:
            db.execute("UPDATE questions SET ruling_id=? WHERE id=?", (int(ruling_id), int(qid)))

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
