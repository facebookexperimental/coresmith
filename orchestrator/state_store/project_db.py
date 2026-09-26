# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""One SQLite project state for the engine: ``<project_root>/.coresmith/project.sqlite``.

This is the canonical home of everything that used to live in JSON files under
``.coresmith/``: the block registry (block diagram, block specs, block queue),
the interface contracts, per-block lifecycle state (attempt history, diagnoses,
constraints, best result and the other gate results), and the run settings.
The DV / PPA / coverage scoreboard tables live in the same database.

Design rules
------------
* SQLite in WAL mode, ``busy_timeout`` 5 s. Writes raise on failure: a lost
  state write is a bug, not something to swallow.
* The API returns the same dict shapes the JSON documents had, so callers
  that used to ``json.load`` a file call one function instead. Each block row
  keeps the original block dict as ``extra_json`` next to the typed columns so
  the round trip is faithful even for keys this schema does not model.
* Immutability of the architecture registry is a *policy of the pipeline*
  (the architecture phase writes it once); there are no checksum sidecars.
  Contract edges carry an integer ``version`` that amendments bump; a block
  records the version its uArch spec was written against, which replaces the
  old ``uarch_spec_contract_sha1`` sidecars and ``contract_stale`` markers.
* Read-only JSON *views* are regenerated from the database on every write, at
  the paths the JSON state used to live (``.coresmith/block_diagram.json``,
  ``block_specs.json``, ``interface_contracts.json`` and
  ``blocks/<b>/{constraints,diagnosis,attempt_history,best_result}.json``).
  Agents and tools keep reading them; nothing may write them (mode 0444), and
  LLM-authored drafts of the architecture documents go to ``.coresmith/drafts/``
  before the engine imports them.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from orchestrator.state_store.store import _SCHEMA as _SCOREBOARD_SCHEMA
from orchestrator.state_store.interrupts import InterruptMixin
from orchestrator.state_store.leases import LeaseMixin
from orchestrator.state_store.rulings import RulingMixin

DB_NAME = "project.sqlite"
# Result kinds that carry a per-block pass and are exported as block views.
#   dv_best -- the block's DV pass (sim green), written the moment cocotb passes;
#   best    -- the block's PUBLISHED pass: sim AND synth AND timing met (written
#              by block_done only). Consumers that mean "the block is done" must
#              read ``best``; consumers that mean "DV passed" read ``dv_best``.
RESULT_VIEW_KINDS = ("best", "dv_best")


DRAFTS_DIR = "drafts"
BLOCK_VIEW_KINDS = ("constraints", "diagnosis", "attempt_history", "best_result")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS settings (
    name TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at REAL
);
CREATE TABLE IF NOT EXISTS doc_meta (
    doc TEXT NOT NULL,          -- block_diagram | interface_contracts
    key TEXT NOT NULL,
    value_json TEXT,
    PRIMARY KEY (doc, key)
);
CREATE TABLE IF NOT EXISTS blocks (
    name TEXT PRIMARY KEY,
    ordinal INTEGER NOT NULL DEFAULT 0,
    tier TEXT,
    subsystem TEXT,
    description TEXT,
    python_source TEXT,
    rtl_target TEXT,
    testbench TEXT,
    estimated_gates INTEGER,
    flip_flop_budget INTEGER,
    area_budget_um2 REAL,
    extra_json TEXT,
    in_queue INTEGER NOT NULL DEFAULT 1,
    spec_contract_version INTEGER
);
CREATE TABLE IF NOT EXISTS interfaces (
    id INTEGER PRIMARY KEY,
    block TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    name TEXT,
    spec_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_interfaces_block ON interfaces(block);
CREATE TABLE IF NOT EXISTS connections (
    id INTEGER PRIMARY KEY,
    ordinal INTEGER NOT NULL,
    from_block TEXT,
    to_block TEXT,
    from_port TEXT,
    to_port TEXT,
    interface TEXT,
    data_width INTEGER,
    bus_name TEXT,
    handshake_protocol TEXT,
    extra_json TEXT
);
CREATE TABLE IF NOT EXISTS contracts (
    edge_id TEXT PRIMARY KEY,
    ordinal INTEGER NOT NULL,
    producer_block TEXT,
    producer_port TEXT,
    consumer_block TEXT,
    consumer_port TEXT,
    handshake_protocol TEXT,
    data_width_bits INTEGER,
    spec_json TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS attempts (
    id INTEGER PRIMARY KEY,
    block TEXT NOT NULL,
    round INTEGER NOT NULL DEFAULT 1,
    attempt INTEGER,
    category TEXT,
    error TEXT,
    extra_json TEXT,
    ts REAL
);
CREATE INDEX IF NOT EXISTS idx_attempts_block ON attempts(block);
CREATE TABLE IF NOT EXISTS diagnoses (
    id INTEGER PRIMARY KEY,
    block TEXT NOT NULL,
    round INTEGER NOT NULL DEFAULT 1,
    attempt INTEGER,
    category TEXT,
    confidence REAL,
    diagnosis_json TEXT NOT NULL,
    ts REAL
);
CREATE INDEX IF NOT EXISTS idx_diagnoses_block ON diagnoses(block);
CREATE TABLE IF NOT EXISTS constraints (
    id INTEGER PRIMARY KEY,
    block TEXT NOT NULL,
    rule TEXT NOT NULL,
    source TEXT,
    attempt INTEGER,
    extra_json TEXT,
    ts REAL
);
CREATE INDEX IF NOT EXISTS idx_constraints_block ON constraints(block);
CREATE TABLE IF NOT EXISTS results (
    block TEXT NOT NULL,
    kind TEXT NOT NULL,          -- best | dv_best | integration | gate_sim | conformance | dv_summary | throughput | ...
    value_json TEXT NOT NULL,
    report_path TEXT,
    ts REAL,
    PRIMARY KEY (block, kind)
);
-- Run state (C1): process locks and per-run facts that used to live in
-- flocks, PID files and in-memory sets. See state_store/leases.py.
CREATE TABLE IF NOT EXISTS leases (
    name TEXT PRIMARY KEY,
    holder_pid INTEGER NOT NULL,
    holder_host TEXT NOT NULL,
    token TEXT NOT NULL,
    acquired_ts REAL NOT NULL,
    expires_ts REAL NOT NULL,
    meta_json TEXT,
    stolen_from_json TEXT
);
CREATE TABLE IF NOT EXISTS run_flags (
    name TEXT NOT NULL,
    run_id TEXT NOT NULL DEFAULT '',
    value_json TEXT,
    ts REAL NOT NULL,
    PRIMARY KEY (name, run_id)
);
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY,
    interrupt_id TEXT,
    interrupt_type TEXT,
    block TEXT,
    action TEXT NOT NULL,
    reasoning TEXT,
    decision_index INTEGER NOT NULL,
    run_id TEXT NOT NULL DEFAULT '',
    ts REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_decisions_run ON decisions(run_id);
CREATE TABLE IF NOT EXISTS interrupts (
    id TEXT PRIMARY KEY,             -- coresmith interrupt id (in the payload)
    lg_interrupt_id TEXT,            -- LangGraph Interrupt.id once the daemon has seen it
    graph TEXT NOT NULL,             -- architecture | pipeline | backend
    branch TEXT,
    node TEXT NOT NULL,
    block TEXT,
    kind TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',   -- pending | resolved | consumed | abandoned
    resolution_json TEXT,
    resolved_by TEXT,
    run_id TEXT NOT NULL DEFAULT '',
    ts REAL NOT NULL,
    resolved_ts REAL,
    consumed_ts REAL
);
CREATE INDEX IF NOT EXISTS idx_interrupts_status ON interrupts(status, graph);
-- Operator rulings (C2): additive run-time policy, never under inputs/.
CREATE TABLE IF NOT EXISTS rulings (
    id INTEGER PRIMARY KEY,
    scope TEXT NOT NULL,             -- global | arch | block:<name> | edge:<edge_id>
    question_ref TEXT,               -- interrupt:<id> | prd:<qid> | block:<name>:<kind>
    text TEXT NOT NULL,
    rationale TEXT,
    source TEXT NOT NULL DEFAULT 'human',   -- human | chip_lead | coordinator
    ts REAL NOT NULL,
    supersedes_id INTEGER,
    revoked_ts REAL,
    revoked_reason TEXT,
    conflict_json TEXT
);
CREATE TABLE IF NOT EXISTS ruling_uses (
    id INTEGER PRIMARY KEY,
    ruling_id INTEGER NOT NULL,
    consumer TEXT NOT NULL,
    block TEXT,
    node TEXT,
    attempt INTEGER,
    run_id TEXT NOT NULL DEFAULT '',
    ts REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ruling_uses_ruling ON ruling_uses(ruling_id);
"""

_BLOCK_COLUMNS = ("tier", "subsystem", "description", "python_source", "rtl_target",
                  "testbench", "estimated_gates", "flip_flop_budget", "area_budget_um2")
_SPEC_KEYS = ("name", "tier", "python_source", "rtl_target", "testbench", "description")
_CONN_COLUMNS = ("from", "to", "from_port", "to_port", "interface", "data_width",
                 "bus_name", "handshake_protocol")
_CONTRACT_COLUMNS = ("producer_block", "producer_port", "consumer_block", "consumer_port",
                     "handshake_protocol", "data_width_bits")
_LEGACY_FILES = ("block_diagram.json", "block_specs.json", "interface_contracts.json")


def _j(value: Any) -> str:
    return json.dumps(value, default=str, sort_keys=False)


def _uj(text: str | None, default: Any):
    if not text:
        return default
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return default


def _int(v: Any) -> int | None:
    try:
        return int(v) if v is not None and v != "" else None
    except (TypeError, ValueError):
        return None


def _float(v: Any) -> float | None:
    try:
        return float(v) if v is not None and v != "" else None
    except (TypeError, ValueError):
        return None


class ProjectDB(LeaseMixin, InterruptMixin, RulingMixin):
    """The project database. Construct with :func:`open_project` in most code."""

    def __init__(self, project_root: str | Path):
        self.root = Path(project_root).resolve()
        self.path = self.root / ".coresmith" / DB_NAME

    # ------------------------------------------------------------------ core
    def ensure_schema(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as db:
            db.executescript(_SCHEMA)
            db.executescript(_SCOREBOARD_SCHEMA)
            for table in ("attempts", "diagnoses"):
                cols = {r[1] for r in db.execute(f"PRAGMA table_info({table})")}
                if "round" not in cols:
                    db.execute(f"ALTER TABLE {table} ADD COLUMN round INTEGER NOT NULL DEFAULT 1")
            cols = {r[1] for r in db.execute("PRAGMA table_info(ppa_history)")}
            if cols and "tns_ns" not in cols:
                db.execute("ALTER TABLE ppa_history ADD COLUMN tns_ns REAL")

    @contextmanager
    def _conn(self):
        db = sqlite3.connect(str(self.path), timeout=5.0, isolation_level=None)
        try:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA busy_timeout=5000")
            db.row_factory = sqlite3.Row
            yield db
        finally:
            db.close()

    @contextmanager
    def _tx(self):
        with self._conn() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                yield db
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise

    def exists(self) -> bool:
        return self.path.is_file()

    # -------------------------------------------------------------- settings
    def get_setting(self, name: str, default: str | None = None) -> str | None:
        with self._conn() as db:
            row = db.execute("SELECT value FROM settings WHERE name=?", (name,)).fetchone()
        return row["value"] if row else default

    def set_setting(self, name: str, value: str) -> None:
        with self._tx() as db:
            db.execute(
                "INSERT INTO settings(name, value, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(name) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (name, str(value), time.time()),
            )

    def settings(self) -> dict[str, str]:
        with self._conn() as db:
            return {r["name"]: r["value"] for r in db.execute("SELECT name, value FROM settings ORDER BY name")}

    # --------------------------------------------------------- block diagram
    def import_block_diagram(self, doc: dict) -> None:
        """Replace the block registry with the architecture phase's block diagram."""
        blocks = list(doc.get("blocks") or [])
        connections = list(doc.get("connections") or [])
        with self._tx() as db:
            db.execute("DELETE FROM interfaces")
            db.execute("DELETE FROM connections")
            db.execute("DELETE FROM doc_meta WHERE doc='block_diagram'")
            seen: set[str] = set()
            for i, b in enumerate(blocks):
                name = str(b.get("name") or "").strip()
                if not name:
                    continue
                seen.add(name)
                cols = {c: b.get(c) for c in _BLOCK_COLUMNS}
                db.execute(
                    "INSERT INTO blocks(name, ordinal, tier, subsystem, description, python_source, "
                    "rtl_target, testbench, estimated_gates, flip_flop_budget, area_budget_um2, extra_json) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(name) DO UPDATE SET "
                    "ordinal=excluded.ordinal, tier=excluded.tier, subsystem=excluded.subsystem, "
                    "description=excluded.description, python_source=excluded.python_source, "
                    "rtl_target=excluded.rtl_target, testbench=excluded.testbench, "
                    "estimated_gates=excluded.estimated_gates, flip_flop_budget=excluded.flip_flop_budget, "
                    "area_budget_um2=excluded.area_budget_um2, extra_json=excluded.extra_json",
                    (name, i, _s(cols["tier"]), _s(cols["subsystem"]), _s(cols["description"]),
                     _s(cols["python_source"]), _s(cols["rtl_target"]), _s(cols["testbench"]),
                     _int(cols["estimated_gates"]), _int(cols["flip_flop_budget"]),
                     _float(cols["area_budget_um2"]), _j(b)),
                )
                for k, iface in enumerate(b.get("interfaces") or []):
                    iname = iface.get("name") if isinstance(iface, dict) else str(iface)
                    db.execute(
                        "INSERT INTO interfaces(block, ordinal, name, spec_json) VALUES (?, ?, ?, ?)",
                        (name, k, _s(iname), _j(iface)),
                    )
            # blocks that vanished from the diagram leave the queue but keep history
            db.execute(
                f"UPDATE blocks SET in_queue=0 WHERE name NOT IN ({','.join('?' * len(seen)) or 'NULL'})",
                tuple(seen),
            )
            for i, c in enumerate(connections):
                db.execute(
                    "INSERT INTO connections(ordinal, from_block, to_block, from_port, to_port, interface, "
                    "data_width, bus_name, handshake_protocol, extra_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (i, _s(c.get("from")), _s(c.get("to")), _s(c.get("from_port")), _s(c.get("to_port")),
                     _s(c.get("interface")), _int(c.get("data_width")), _s(c.get("bus_name")),
                     _s(c.get("handshake_protocol")), _j(c)),
                )
            for key, value in doc.items():
                if key in ("blocks", "connections"):
                    continue
                db.execute("INSERT INTO doc_meta(doc, key, value_json) VALUES ('block_diagram', ?, ?)",
                           (key, _j(value)))
        self.export_views()

    def block_diagram(self) -> dict:
        """The block diagram document ({} when the architecture phase never ran)."""
        with self._conn() as db:
            brows = db.execute("SELECT * FROM blocks WHERE in_queue=1 ORDER BY ordinal, name").fetchall()
            crows = db.execute("SELECT * FROM connections ORDER BY ordinal").fetchall()
            meta = db.execute("SELECT key, value_json FROM doc_meta WHERE doc='block_diagram'").fetchall()
        if not brows and not meta:
            return {}
        doc: dict[str, Any] = {"blocks": [self._block_dict(r) for r in brows],
                               "connections": [_uj(r["extra_json"], {}) for r in crows]}
        for r in meta:
            doc[r["key"]] = _uj(r["value_json"], None)
        return doc

    @staticmethod
    def _block_dict(row: sqlite3.Row) -> dict:
        base = _uj(row["extra_json"], {}) or {}
        base["name"] = row["name"]
        for c in _BLOCK_COLUMNS:
            v = row[c]
            if v is not None:
                base[c] = v
        return base

    def blocks(self) -> list[dict]:
        """All queued blocks as full dicts, in diagram order."""
        with self._conn() as db:
            rows = db.execute("SELECT * FROM blocks WHERE in_queue=1 ORDER BY ordinal, name").fetchall()
        return [self._block_dict(r) for r in rows]

    def block(self, name: str) -> dict | None:
        with self._conn() as db:
            row = db.execute("SELECT * FROM blocks WHERE name=?", (name,)).fetchone()
        return self._block_dict(row) if row else None

    def block_names(self) -> list[str]:
        with self._conn() as db:
            return [r["name"] for r in db.execute(
                "SELECT name FROM blocks WHERE in_queue=1 ORDER BY ordinal, name")]

    # ------------------------------------------------------------ block specs
    def import_block_specs(self, specs: list[dict]) -> None:
        """Set the RTL pipeline's block queue (the ``block_specs.json`` handoff)."""
        with self._tx() as db:
            names = []
            for i, spec in enumerate(specs):
                name = str(spec.get("name") or "").strip()
                if not name:
                    continue
                names.append(name)
                existing = db.execute("SELECT extra_json FROM blocks WHERE name=?", (name,)).fetchone()
                extra = _uj(existing["extra_json"], {}) if existing else {}
                extra.update({k: v for k, v in spec.items() if k not in _SPEC_KEYS})
                db.execute(
                    "INSERT INTO blocks(name, ordinal, tier, description, python_source, rtl_target, "
                    "testbench, extra_json, in_queue) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1) "
                    "ON CONFLICT(name) DO UPDATE SET ordinal=excluded.ordinal, tier=excluded.tier, "
                    "description=COALESCE(excluded.description, blocks.description), "
                    "python_source=excluded.python_source, rtl_target=excluded.rtl_target, "
                    "testbench=excluded.testbench, extra_json=excluded.extra_json, in_queue=1",
                    (name, i, _s(spec.get("tier")), _s(spec.get("description")),
                     _s(spec.get("python_source")), _s(spec.get("rtl_target")),
                     _s(spec.get("testbench")), _j(extra)),
                )
            db.execute(
                f"UPDATE blocks SET in_queue=0 WHERE name NOT IN ({','.join('?' * len(names)) or 'NULL'})",
                tuple(names),
            )
        self.export_views()

    def block_specs(self) -> list[dict]:
        """The block queue in the ``block_specs.json`` shape (empty when unset)."""
        with self._conn() as db:
            rows = db.execute("SELECT * FROM blocks WHERE in_queue=1 ORDER BY ordinal, name").fetchall()
        out = []
        for r in rows:
            d = {k: (r[k] if k != "name" else r["name"]) for k in _SPEC_KEYS}
            for k in ("python_source", "rtl_target", "testbench", "description"):
                if d[k] is None:
                    d[k] = ""
            if d["tier"] is None:
                d["tier"] = "1"
            out.append(d)
        return out

    # -------------------------------------------------------------- contracts
    def import_contracts(self, doc: dict) -> int:
        """Replace the interface contracts; returns the new contracts version.

        Edges whose content is unchanged keep their version; changed or new edges
        get the new version, so per-block staleness is exact.
        """
        edges = list(doc.get("contracts") or [])
        with self._tx() as db:
            current = {r["edge_id"]: (r["spec_json"], r["version"]) for r in db.execute(
                "SELECT edge_id, spec_json, version FROM contracts")}
            new_version = max([v for _, v in current.values()] + [0]) + 1
            db.execute("DELETE FROM contracts")
            db.execute("DELETE FROM doc_meta WHERE doc='interface_contracts'")
            changed = 0
            for i, e in enumerate(edges):
                edge_id = str(e.get("edge_id") or f"edge_{i}")
                spec = _j(e)
                prev = current.get(edge_id)
                version = prev[1] if prev and prev[0] == spec else new_version
                if version == new_version:
                    changed += 1
                db.execute(
                    "INSERT INTO contracts(edge_id, ordinal, producer_block, producer_port, consumer_block, "
                    "consumer_port, handshake_protocol, data_width_bits, spec_json, version) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (edge_id, i, _s(e.get("producer_block")), _s(e.get("producer_port")),
                     _s(e.get("consumer_block")), _s(e.get("consumer_port")),
                     _s(e.get("handshake_protocol")), _int(e.get("data_width_bits")), spec, version),
                )
            for key, value in doc.items():
                if key == "contracts":
                    continue
                db.execute("INSERT INTO doc_meta(doc, key, value_json) VALUES ('interface_contracts', ?, ?)",
                           (key, _j(value)))
            effective = new_version if changed else max([v for _, v in current.values()] + [0])
            db.execute(
                "INSERT INTO settings(name, value, updated_at) VALUES ('contracts_version', ?, ?) "
                "ON CONFLICT(name) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (str(effective), time.time()),
            )
        self.export_views()
        return effective

    def contracts(self) -> dict:
        """The interface contracts document ({} when none were defined)."""
        with self._conn() as db:
            rows = db.execute("SELECT spec_json FROM contracts ORDER BY ordinal").fetchall()
            meta = db.execute("SELECT key, value_json FROM doc_meta WHERE doc='interface_contracts'").fetchall()
        if not rows and not meta:
            return {}
        doc: dict[str, Any] = {}
        for r in meta:
            doc[r["key"]] = _uj(r["value_json"], None)
        doc["contracts"] = [_uj(r["spec_json"], {}) for r in rows]
        return doc

    def contract_edges_for_block(self, name: str) -> list[dict]:
        with self._conn() as db:
            rows = db.execute(
                "SELECT spec_json FROM contracts WHERE producer_block=? OR consumer_block=? ORDER BY ordinal",
                (name, name)).fetchall()
        return [_uj(r["spec_json"], {}) for r in rows]

    def contracts_version(self) -> int:
        return int(self.get_setting("contracts_version", "0") or 0)

    def block_contract_version(self, name: str) -> int:
        """Highest version among the edges this block produces or consumes (0 = none)."""
        with self._conn() as db:
            row = db.execute(
                "SELECT MAX(version) AS v FROM contracts WHERE producer_block=? OR consumer_block=?",
                (name, name)).fetchone()
        return int(row["v"] or 0)

    def stamp_block_spec(self, name: str) -> int:
        """Record that ``name``'s uArch spec was written against the live contracts."""
        v = self.block_contract_version(name)
        with self._tx() as db:
            db.execute("UPDATE blocks SET spec_contract_version=? WHERE name=?", (v, name))
        return v

    def stale_spec_blocks(self, names: list[str] | None = None) -> list[str]:
        """Blocks whose recorded spec contract version is behind their live edges."""
        with self._conn() as db:
            rows = db.execute("SELECT name, spec_contract_version FROM blocks WHERE in_queue=1").fetchall()
        out = []
        for r in rows:
            if names is not None and r["name"] not in names:
                continue
            stamped = r["spec_contract_version"]
            if stamped is None:
                continue  # never stamped: not a staleness question
            if self.block_contract_version(r["name"]) > int(stamped):
                out.append(r["name"])
        return out

    # ------------------------------------------------------ per-block state
    # A "round" is one lifecycle of the block subgraph (each init_block entry
    # starts a new one). Attempt history and diagnoses are kept for every round;
    # the pipeline reads the current round, reviewers can read them all.
    def current_round(self, block: str) -> int:
        return int(self.get_setting(f"round:{block}", "1") or 1)

    def begin_round(self, block: str) -> int:
        """Start a new lifecycle round for ``block`` and return its number."""
        with self._tx() as db:
            row = db.execute("SELECT value FROM settings WHERE name=?", (f"round:{block}",)).fetchone()
            recorded = max(
                int(db.execute("SELECT COALESCE(MAX(round), 0) FROM attempts WHERE block=?", (block,)).fetchone()[0]),
                int(db.execute("SELECT COALESCE(MAX(round), 0) FROM diagnoses WHERE block=?", (block,)).fetchone()[0]),
            )
            # A block's first lifecycle is round 1; anything recorded before a
            # round was started counts as round 1 and the next one becomes 2.
            new = max(int(row["value"]) if row else 0, recorded) + 1
            db.execute(
                "INSERT INTO settings(name, value, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(name) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (f"round:{block}", str(new), time.time()),
            )
        self.export_block_views(block)
        return new

    def attempt_history(self, block: str, all_rounds: bool = False) -> list[dict]:
        with self._conn() as db:
            if all_rounds:
                rows = db.execute("SELECT * FROM attempts WHERE block=? ORDER BY id", (block,)).fetchall()
            else:
                rows = db.execute("SELECT * FROM attempts WHERE block=? AND round=? ORDER BY id",
                                  (block, self.current_round(block))).fetchall()
        out = []
        for r in rows:
            d = _uj(r["extra_json"], {}) or {}
            d.update({"attempt": r["attempt"], "error": r["error"], "category": r["category"]})
            if all_rounds:
                d["round"] = r["round"]
            out.append({k: v for k, v in d.items() if v is not None})
        return out

    def record_attempt(self, block: str, entry: dict) -> None:
        extra = {k: v for k, v in entry.items() if k not in ("attempt", "error", "category")}
        with self._tx() as db:
            db.execute(
                "INSERT INTO attempts(block, round, attempt, category, error, extra_json, ts) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (block, self.current_round(block), _int(entry.get("attempt")), _s(entry.get("category")),
                 _s(entry.get("error")), _j(extra) if extra else None, time.time()),
            )
        self.export_block_views(block)

    def clear_attempts(self, block: str) -> None:
        with self._tx() as db:
            db.execute("DELETE FROM attempts WHERE block=?", (block,))
        self.export_block_views(block)

    def diagnosis(self, block: str, all_rounds: bool = False) -> dict | None:
        """The latest diagnosis of the current round (None at a fresh round)."""
        with self._conn() as db:
            if all_rounds:
                row = db.execute("SELECT diagnosis_json FROM diagnoses WHERE block=? ORDER BY id DESC LIMIT 1",
                                 (block,)).fetchone()
            else:
                row = db.execute(
                    "SELECT diagnosis_json FROM diagnoses WHERE block=? AND round=? ORDER BY id DESC LIMIT 1",
                    (block, self.current_round(block))).fetchone()
        return _uj(row["diagnosis_json"], None) if row else None

    def diagnoses(self, block: str) -> list[dict]:
        """Every diagnosis ever recorded for ``block`` (all rounds, oldest first)."""
        with self._conn() as db:
            rows = db.execute("SELECT round, attempt, ts, diagnosis_json FROM diagnoses WHERE block=? ORDER BY id",
                              (block,)).fetchall()
        return [{"round": r["round"], "attempt": r["attempt"], "ts": r["ts"],
                 **(_uj(r["diagnosis_json"], {}) or {})} for r in rows]

    def set_diagnosis(self, block: str, diagnosis: dict, attempt: int | None = None) -> None:
        with self._tx() as db:
            db.execute(
                "INSERT INTO diagnoses(block, round, attempt, category, confidence, diagnosis_json, ts) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (block, self.current_round(block), attempt, _s(diagnosis.get("category")),
                 _float(diagnosis.get("confidence")), _j(diagnosis), time.time()),
            )
        self.export_block_views(block)

    def clear_diagnosis(self, block: str) -> None:
        with self._tx() as db:
            db.execute("DELETE FROM diagnoses WHERE block=?", (block,))
        self.export_block_views(block)

    def constraints(self, block: str) -> list[dict]:
        with self._conn() as db:
            rows = db.execute("SELECT * FROM constraints WHERE block=? ORDER BY id", (block,)).fetchall()
        out = []
        for r in rows:
            d = _uj(r["extra_json"], {}) or {}
            d.update({"rule": r["rule"], "source": r["source"], "attempt": r["attempt"]})
            out.append({k: v for k, v in d.items() if v is not None})
        return out

    def add_constraint(self, block: str, rule: str, source: str = "", attempt: int | None = None,
                       **extra: Any) -> None:
        with self._tx() as db:
            db.execute(
                "INSERT INTO constraints(block, rule, source, attempt, extra_json, ts) VALUES (?, ?, ?, ?, ?, ?)",
                (block, str(rule), _s(source), attempt, _j(extra) if extra else None, time.time()),
            )
        self.export_block_views(block)

    def set_constraints(self, block: str, items: list[dict]) -> None:
        """Replace the block's constraint list (order preserved)."""
        with self._tx() as db:
            db.execute("DELETE FROM constraints WHERE block=?", (block,))
            for c in items:
                if not isinstance(c, dict) or not c.get("rule"):
                    continue
                extra = {k: v for k, v in c.items() if k not in ("rule", "source", "attempt")}
                db.execute(
                    "INSERT INTO constraints(block, rule, source, attempt, extra_json, ts) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (block, str(c["rule"]), _s(c.get("source")), _int(c.get("attempt")),
                     _j(extra) if extra else None, time.time()),
                )
        self.export_block_views(block)

    def prune_constraints(self, block: str, keep_sources: tuple[str, ...]) -> int:
        """Drop per-lifecycle constraints, keeping the regeneration-proof sources."""
        with self._tx() as db:
            cur = db.execute(
                f"DELETE FROM constraints WHERE block=? AND (source IS NULL OR source NOT IN "
                f"({','.join('?' * len(keep_sources)) or 'NULL'}))",
                (block, *keep_sources),
            )
            dropped = cur.rowcount
        self.export_block_views(block)
        return dropped

    def result(self, block: str, kind: str) -> dict | None:
        with self._conn() as db:
            row = db.execute("SELECT value_json FROM results WHERE block=? AND kind=?",
                             (block, kind)).fetchone()
        return _uj(row["value_json"], None) if row else None

    def set_result(self, block: str, kind: str, value: dict, report_path: str | None = None) -> None:
        if kind in RESULT_VIEW_KINDS and "spec_sha256" not in value:
            import hashlib
            spec = self.root / "arch/uarch_specs" / f"{block}.md"
            if spec.is_file():
                value = {**value, "spec_sha256": hashlib.sha256(spec.read_bytes()).hexdigest()}
        with self._tx() as db:
            db.execute(
                "INSERT INTO results(block, kind, value_json, report_path, ts) VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(block, kind) DO UPDATE SET value_json=excluded.value_json, "
                "report_path=excluded.report_path, ts=excluded.ts",
                (block, kind, _j(value), report_path, time.time()),
            )
        if kind in RESULT_VIEW_KINDS:
            self.export_block_views(block)

    def update_result(self, block: str, kind: str, **fields: Any) -> dict:
        with self._conn() as db:
            row = db.execute("SELECT value_json, report_path FROM results WHERE block=? AND kind=?",
                             (block, kind)).fetchone()
        cur = (_uj(row["value_json"], {}) if row else {}) or {}
        cur.update(fields)
        self.set_result(block, kind, cur, report_path=row["report_path"] if row else None)
        return cur

    def clear_result(self, block: str, kind: str) -> None:
        with self._tx() as db:
            db.execute("DELETE FROM results WHERE block=? AND kind=?", (block, kind))
        if kind in RESULT_VIEW_KINDS:
            self.export_block_views(block)

    def invalidate_results_for_specs(self, spec_hashes: dict[str, str]) -> list[str]:
        """Archive and clear pass results for different (or unrecorded) spec bytes.

        Both the DV-only pass (``dv_best``) and the published block pass
        (``best``) are earned against one reviewed spec; a changed spec voids
        both.
        """
        invalidated = []
        with self._tx() as db:
            for block, digest in spec_hashes.items():
                hit = False
                for kind in RESULT_VIEW_KINDS:
                    row = db.execute("SELECT value_json FROM results WHERE block=? AND kind=?",
                                     (block, kind)).fetchone()
                    best = _uj(row["value_json"], {}) if row else None
                    if best is None or best.get("spec_sha256") == digest:
                        continue
                    archived = {"previous_best": best, "previous_kind": kind,
                                "adopted_spec_sha256": digest,
                                "reason": "reviewed spec changed; verification required"}
                    db.execute("INSERT OR REPLACE INTO results(block,kind,value_json,ts) VALUES(?,?,?,?)",
                               (block, "spec_invalidated", _j(archived), time.time()))
                    db.execute("DELETE FROM results WHERE block=? AND kind=?", (block, kind))
                    hit = True
                if hit:
                    invalidated.append(block)
        for block in invalidated:
            self.export_block_views(block)
        return invalidated

    def results(self, block: str | None = None) -> list[dict]:
        with self._conn() as db:
            if block:
                rows = db.execute("SELECT * FROM results WHERE block=? ORDER BY kind", (block,)).fetchall()
            else:
                rows = db.execute("SELECT * FROM results ORDER BY block, kind").fetchall()
        return [{"block": r["block"], "kind": r["kind"], "value": _uj(r["value_json"], {}),
                 "report_path": r["report_path"], "ts": r["ts"]} for r in rows]

    # ------------------------------------------------------------- views
    @staticmethod
    def _write_view(target: Path, value) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
        os.chmod(tmp, 0o444)
        os.replace(tmp, target)

    @staticmethod
    def _write_text_view(target: Path, text: str) -> None:
        """A read-only non-JSON view (jsonl, markdown), written atomically."""
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.chmod(tmp, 0o444)
        os.replace(tmp, target)

    def export_decisions_view(self) -> Path:
        """``.coresmith/chip_lead/decisions.jsonl`` regenerated from ``decisions``.

        The jsonl used to BE the ledger (guarded by a flock); it is now a
        read-only view for tools that tail it.
        """
        target = self.path.parent / "chip_lead" / "decisions.jsonl"
        lines = [json.dumps({
            "interrupt_type": d.get("interrupt_type", ""),
            "block_name": d.get("block", ""),
            "action": d.get("action", ""),
            "reasoning": d.get("reasoning", ""),
            "decision_index": d.get("decision_index"),
            "interrupt_id": d.get("interrupt_id", ""),
            "ts": d.get("ts"),
        }, default=str) for d in self.decisions()]
        self._write_text_view(target, ("\n".join(lines) + "\n") if lines else "")
        return target

    def begin_run(self, run_id: str | None = None) -> str:
        """Mint (or adopt) the run id every run-scoped table is keyed by."""
        import uuid
        rid = run_id or f"run-{int(time.time())}-{uuid.uuid4().hex[:8]}"
        self.set_setting("run_id", rid)
        return rid

    # WP-75: the registry views are read-only by convention, but the chip lead
    # (and operators) edit them with file tools -- the prompts tell them to fix
    # `.coresmith/interface_contracts.json` on disk. Re-exporting the database
    # over such an edit silently restored 24 illegal port names on a live run.
    # An edited view is therefore IMPORTED into the database before the views
    # are regenerated, so the on-disk edit becomes canonical instead of lost.
    def _adopt_external_view_edit(self, target: Path, importer, kind) -> bool:
        if not target.exists():
            return False
        try:
            data = target.read_bytes()
        except OSError:
            return False
        recorded = self.get_setting(f"view_sha:{target.name}")
        if not recorded:
            return False  # never exported by this database: nothing to compare
        current = hashlib.sha256(data).hexdigest()
        if current == recorded:
            return False
        try:
            parsed = json.loads(data.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return False  # unparsable edit: the database wins, the file is regenerated
        if not isinstance(parsed, kind):
            return False
        importer(parsed)
        self.set_setting(f"view_adopted:{target.name}", current)
        return True

    def _export_view(self, target: Path, value) -> None:
        self._write_view(target, value)
        self.set_setting(f"view_sha:{target.name}",
                         hashlib.sha256(target.read_bytes()).hexdigest())

    def export_views(self) -> None:
        """Regenerate the read-only registry views from the database, after
        adopting any on-disk edit made to them since the last export (WP-75)."""
        if getattr(self, "_exporting_views", False):
            return
        self._exporting_views = True
        try:
            cdir = self.path.parent
            self._adopt_external_view_edit(cdir / "block_diagram.json",
                                           self.import_block_diagram, dict)
            self._adopt_external_view_edit(cdir / "block_specs.json",
                                           self.import_block_specs, list)
            self._adopt_external_view_edit(cdir / "interface_contracts.json",
                                           self.import_contracts, dict)
            bd = self.block_diagram()
            if bd:
                self._export_view(cdir / "block_diagram.json", bd)
            specs = self.block_specs()
            if specs:
                self._export_view(cdir / "block_specs.json", specs)
            contracts = self.contracts()
            if contracts:
                self._export_view(cdir / "interface_contracts.json", contracts)
            try:
                self.export_rulings_view()
            except OSError:
                pass
        finally:
            self._exporting_views = False
        note = cdir / "STATE.md"
        if not note.exists():
            note.write_text(
                "The project state is project.sqlite. The JSON files next to it and under "
                "blocks/<block>/ are read-only views regenerated from the database on every "
                "write; use the CLI (`coresmith blocks|block|contracts|attempts|constraints|"
                "results|settings`) or the pipeline to change state. LLM-authored drafts of "
                "the architecture documents are written to drafts/ and imported.\n",
                encoding="utf-8")

    def export_block_views(self, block: str) -> None:
        """Regenerate ``blocks/<block>/*.json`` read-only views."""
        bdir = self.path.parent / "blocks" / block
        self._write_view(bdir / "constraints.json", self.constraints(block))
        self._write_view(bdir / "diagnosis.json", self.diagnosis(block) or {})
        self._write_view(bdir / "attempt_history.json", self.attempt_history(block))
        self._write_view(bdir / "attempt_history_all_rounds.json", self.attempt_history(block, all_rounds=True))
        self._write_view(bdir / "diagnoses_all_rounds.json", self.diagnoses(block))
        for kind, fname in (("best", "best_result.json"), ("dv_best", "dv_best_result.json")):
            val = self.result(block, kind)
            target = bdir / fname
            if val is None:
                target.unlink(missing_ok=True)
            else:
                self._write_view(target, val)

    def drafts_dir(self) -> Path:
        d = self.path.parent / DRAFTS_DIR
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ---------------------------------------------------------- migration
    def import_legacy_files(self) -> list[str]:
        """Import the pre-SQLite JSON state of an existing run directory once.

        Returns the list of imported artifacts. Idempotent: guarded by the
        ``legacy_import_done`` setting.
        """
        if self.get_setting("legacy_import_done") == "1":
            return []
        cdir = self.path.parent
        imported: list[str] = []
        # Read every legacy document before importing: the views regenerated by
        # an import land on these same paths.
        docs = {}
        for name in ("block_diagram.json", "block_specs.json", "block_queue.json",
                     "interface_contracts.json"):
            f = cdir / name
            if f.is_file() and f.stat().st_size > 0:
                docs[name] = _uj(f.read_text(encoding="utf-8"), None)
        if isinstance(docs.get("block_diagram.json"), dict):
            self.import_block_diagram(docs["block_diagram.json"])
            imported.append("block_diagram.json")
        if isinstance(docs.get("block_specs.json"), list):
            self.import_block_specs(docs["block_specs.json"])
            imported.append("block_specs.json")
        elif isinstance(docs.get("block_queue.json"), list):
            self.import_block_specs(docs["block_queue.json"])
            imported.append("block_queue.json")
        if isinstance(docs.get("interface_contracts.json"), dict):
            self.import_contracts(docs["interface_contracts.json"])
            imported.append("interface_contracts.json")
        blocks_dir = cdir / "blocks"
        if blocks_dir.is_dir():
            kinds = (("best", "best_result.json"), ("dv_best", "dv_best_result.json"),
                     ("coverage", "coverage.json"),
                     ("throughput", "throughput.json"), ("dv_summary", "dv_summary.json"),
                     ("ppa", "ppa_report.json"), ("provenance", "provenance.json"))
            for bdir in sorted(p for p in blocks_dir.iterdir() if p.is_dir()):
                name = bdir.name
                # Read every legacy file first: the first import regenerates the
                # read-only views on these same paths.
                def _load(fname):
                    f = bdir / fname
                    return _uj(f.read_text(encoding="utf-8"), None) if f.is_file() else None
                hist = _load("attempt_history.json")
                diag = _load("diagnosis.json")
                cons = _load("constraints.json")
                results = {kind: _load(fname) for kind, fname in kinds}
                if isinstance(hist, list):
                    for entry in hist:
                        if isinstance(entry, dict):
                            self.record_attempt(name, entry)
                    imported.append(f"blocks/{name}/attempt_history.json")
                if isinstance(diag, dict) and diag:
                    self.set_diagnosis(name, diag)
                    imported.append(f"blocks/{name}/diagnosis.json")
                if isinstance(cons, list):
                    self.set_constraints(name, cons)
                    imported.append(f"blocks/{name}/constraints.json")
                for kind, fname in kinds:
                    val = results.get(kind)
                    if isinstance(val, dict):
                        self.set_result(name, kind, val, report_path=str(bdir / fname))
                        imported.append(f"blocks/{name}/{fname}")
                # A pre-gate run recorded its sim-pass as ``best`` before synth
                # and timing ran. Under the done-result gate that record is a
                # DV pass only: keep it as ``dv_best`` and let ``best`` mean
                # "sim AND synth AND timing" from here on.
                legacy_best = results.get("best")
                if (isinstance(legacy_best, dict) and not legacy_best.get("done")
                        and not isinstance(results.get("dv_best"), dict)):
                    self.set_result(name, "dv_best", legacy_best,
                                    report_path=str(bdir / "best_result.json"))
                    self.clear_result(name, "best")
                    imported.append(f"blocks/{name}/best_result.json->dv_best")
        self.set_setting("legacy_import_done", "1")
        self.export_views()
        return imported


def _s(v: Any) -> str | None:
    if v is None:
        return None
    return str(v)


def open_project(project_root: str | Path) -> ProjectDB:
    """Open (creating if needed) the project database and import legacy JSON once."""
    db = ProjectDB(project_root)
    db.ensure_schema()
    db.import_legacy_files()
    return db
