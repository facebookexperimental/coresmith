# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""``Scoreboard`` -- the DV / PPA / coverage result tables of the project database.

Design constraints (from the Package B plan):

* SQLite in WAL mode; writer ``busy_timeout=5000``, reader ``mode=ro`` +
  ``busy_timeout=3000`` (mirrors ``telemetry/reader.py``).
* Three tables: ``dv_results``, ``ppa_history``, ``coverage_results``.
* **Every write is best-effort** (wrapped in try/except) -- a scoreboard
  failure must NEVER fail a pipeline node. Reads never raise either; they
  return ``[]`` / ``None`` on any error (missing db, locked, corrupt).

The scoreboard is a *record* of runs, not the source of truth: the CLI
read-only queries fall back to on-disk artifacts (``best_result.json``,
``syn/output/<b>/<b>_report.txt``) when the db is absent.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS dv_results (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    ts             REAL,
    block          TEXT,
    scope          TEXT,   -- model | rtl | synth | chip | chip_model | validation
    source         TEXT,   -- gate | agent
    attempt        INTEGER,
    passed         INTEGER,
    skipped        INTEGER,
    seed           INTEGER,
    tests_passed   INTEGER,
    tests_total    INTEGER,
    tests_failed   INTEGER,
    first_divergence TEXT,  -- JSON
    detail         TEXT,
    log_path       TEXT,
    duration_s     REAL
);
CREATE INDEX IF NOT EXISTS idx_dv_block_scope ON dv_results(block, scope);

CREATE TABLE IF NOT EXISTS ppa_history (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    ts             REAL,
    block          TEXT,
    attempt        INTEGER,
    source         TEXT,   -- gate | agent
    probe          TEXT,   -- generic | cellcount | synth | ...
    cells          INTEGER,
    ff             INTEGER,
    mem_bits       INTEGER,
    area_um2       REAL,
    wns_ns         REAL,
    tns_ns         REAL,
    elaborated     INTEGER,
    budget_ff      INTEGER,
    budget_area_um2 REAL,
    ppa_ok         INTEGER,
    reasons        TEXT,   -- JSON
    report_path    TEXT
);
CREATE INDEX IF NOT EXISTS idx_ppa_block ON ppa_history(block);

CREATE TABLE IF NOT EXISTS coverage_results (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    ts             REAL,
    block          TEXT,
    scope          TEXT,
    points_total   INTEGER,
    points_hit     INTEGER,
    pct            REAL,
    uncovered      TEXT,   -- JSON
    dat_path       TEXT,
    annotated_dir  TEXT
);
CREATE INDEX IF NOT EXISTS idx_cov_block ON coverage_results(block);
"""


# Columns added after the first release: every measurement row carries the
# run and the recorded build it was earned in (``build_id`` links it to the
# ``builds`` table, see state_store/builds.py), and a PPA row carries the
# context without which a number is not comparable: the flow stage, the
# tool, the PDK, the clock, the workload and the basis of its power figure
# (``measured`` | ``estimated`` | ``unavailable``; NULL power with a basis of
# ``unavailable`` is "not measured", never zero).
SCOREBOARD_ADDED_COLUMNS = (
    ("ppa_history", "tns_ns", "REAL"),
    ("dv_results", "run_id", "TEXT NOT NULL DEFAULT ''"),
    ("dv_results", "build_id", "TEXT NOT NULL DEFAULT ''"),
    ("coverage_results", "run_id", "TEXT NOT NULL DEFAULT ''"),
    ("coverage_results", "build_id", "TEXT NOT NULL DEFAULT ''"),
    ("ppa_history", "run_id", "TEXT NOT NULL DEFAULT ''"),
    ("ppa_history", "build_id", "TEXT NOT NULL DEFAULT ''"),
    ("ppa_history", "stage", "TEXT"),
    ("ppa_history", "tool", "TEXT"),
    ("ppa_history", "pdk", "TEXT"),
    ("ppa_history", "clock_mhz", "REAL"),
    ("ppa_history", "workload", "TEXT"),
    ("ppa_history", "power_mw", "REAL"),
    ("ppa_history", "power_basis", "TEXT"),
    # a chip-level verdict names the composition it measured (every module's
    # current build + the integrated top's bytes, builds.composition_identity)
    ("dv_results", "composition_sha", "TEXT"),
    ("ppa_history", "composition_sha", "TEXT"),
    # a coverage row belongs to one attempt of one source, like a DV row
    ("coverage_results", "attempt", "INTEGER"),
    ("coverage_results", "source", "TEXT"),
)
POWER_BASES = ("measured", "estimated", "unavailable")


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns introduced after a database was created (idempotent, and
    safe when two connections migrate at once: the second ``ALTER`` of the
    same column is a duplicate-column error, not a lost column)."""
    for table, column, decl in SCOREBOARD_ADDED_COLUMNS:
        cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
        if cols and column not in cols:
            try:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
            except sqlite3.OperationalError as exc:
                if "duplicate column" not in str(exc).lower():
                    raise


def _b(x: Any) -> int | None:
    """Coerce a tri-state (None keeps NULL) boolean to an int for storage."""
    if x is None:
        return None
    return int(bool(x))


def _json(x: Any) -> str | None:
    if x is None:
        return None
    try:
        return json.dumps(x, default=str)
    except Exception:  # noqa: BLE001
        return None


class Scoreboard:
    """Best-effort SQLite scoreboard over ``<project_root>/.coresmith/project.sqlite``."""

    def __init__(self, project_root: str | Path):
        self.project_root = Path(project_root)
        # Shares the canonical project database (see project_db.py).
        self.db_path = self.project_root / ".coresmith" / "project.sqlite"

    # ------------------------------------------------------------------
    # Connections
    # ------------------------------------------------------------------
    def _writer_conn(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path), isolation_level=None, timeout=5.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.row_factory = sqlite3.Row
        return conn

    def _reader_conn(self) -> sqlite3.Connection | None:
        """Read-only connection, or ``None`` if the db does not exist / can't open."""
        if not self.db_path.exists():
            return None
        try:
            conn = sqlite3.connect(
                f"file:{self.db_path}?mode=ro", uri=True, timeout=3.0,
            )
            conn.execute("PRAGMA busy_timeout=3000")
            conn.row_factory = sqlite3.Row
            return conn
        except Exception:  # noqa: BLE001
            return None

    def exists(self) -> bool:
        return self.db_path.exists()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------
    def ensure_schema(self) -> bool:
        """Create the tables if absent. Best-effort -> returns success bool."""
        try:
            conn = self._writer_conn()
            try:
                conn.executescript(_SCHEMA)
                _migrate(conn)
            finally:
                conn.close()
            return True
        except Exception:  # noqa: BLE001
            return False

    # ------------------------------------------------------------------
    # Writes (all best-effort: swallow every exception)
    # ------------------------------------------------------------------
    def record_dv(
        self,
        *,
        block: str,
        scope: str,
        source: str = "gate",
        attempt: int = 0,
        passed: bool = False,
        skipped: bool = False,
        seed: int | None = None,
        tests_passed: int | None = None,
        tests_total: int | None = None,
        tests_failed: int | None = None,
        first_divergence: Any = None,
        detail: str = "",
        log_path: str = "",
        duration_s: float | None = None,
        run_id: str = "",
        build_id: str = "",
        composition_sha: str | None = None,
    ) -> bool:
        try:
            conn = self._writer_conn()
            try:
                conn.executescript(_SCHEMA)
                _migrate(conn)
                conn.execute(
                    "INSERT INTO dv_results (ts, block, scope, source, attempt, "
                    "passed, skipped, seed, tests_passed, tests_total, "
                    "tests_failed, first_divergence, detail, log_path, duration_s, "
                    "run_id, build_id, composition_sha) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        time.time(), block, scope, source, int(attempt or 0),
                        _b(passed), _b(skipped),
                        (int(seed) if seed is not None else None),
                        tests_passed, tests_total, tests_failed,
                        _json(first_divergence), detail or "", log_path or "",
                        duration_s, run_id or "", build_id or "", composition_sha,
                    ),
                )
            finally:
                conn.close()
            return True
        except Exception:  # noqa: BLE001
            return False

    def record_ppa(
        self,
        *,
        block: str,
        attempt: int = 0,
        source: str = "gate",
        probe: str = "",
        cells: int | None = None,
        ff: int | None = None,
        mem_bits: int | None = None,
        area_um2: float | None = None,
        wns_ns: float | None = None,
        tns_ns: float | None = None,
        elaborated: bool | None = None,
        budget_ff: int | None = None,
        budget_area_um2: float | None = None,
        ppa_ok: bool | None = None,
        reasons: Any = None,
        report_path: str = "",
        run_id: str = "",
        build_id: str = "",
        stage: str | None = None,
        tool: str | None = None,
        pdk: str | None = None,
        clock_mhz: float | None = None,
        workload: str | None = None,
        power_mw: float | None = None,
        power_basis: str | None = None,
        composition_sha: str | None = None,
    ) -> bool:
        """One PPA row. ``power_mw`` is NULL unless a figure exists; its
        ``power_basis`` says what kind of figure it is (``measured`` |
        ``estimated`` | ``unavailable``). A row never carries a power figure
        without a basis, and a basis of ``unavailable`` never carries a figure:
        an absent measurement stays NULL with its reason in ``reasons``."""
        if power_basis is not None and power_basis not in POWER_BASES:
            return False
        if power_mw is not None and power_basis in (None, "unavailable"):
            return False
        if power_mw is None and power_basis in ("measured", "estimated"):
            return False
        try:
            conn = self._writer_conn()
            try:
                conn.executescript(_SCHEMA)
                _migrate(conn)
                conn.execute(
                    "INSERT INTO ppa_history (ts, block, attempt, source, probe, "
                    "cells, ff, mem_bits, area_um2, wns_ns, tns_ns, elaborated, budget_ff, "
                    "budget_area_um2, ppa_ok, reasons, report_path, run_id, build_id, "
                    "stage, tool, pdk, clock_mhz, workload, power_mw, power_basis, composition_sha) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        time.time(), block, int(attempt or 0), source, probe or "",
                        cells, ff, mem_bits, area_um2, wns_ns, tns_ns, _b(elaborated),
                        budget_ff, budget_area_um2, _b(ppa_ok),
                        _json(reasons), report_path or "", run_id or "", build_id or "",
                        stage, tool, pdk, clock_mhz, workload, power_mw, power_basis, composition_sha,
                    ),
                )
            finally:
                conn.close()
            return True
        except Exception:  # noqa: BLE001
            return False

    def record_coverage(
        self,
        *,
        block: str,
        scope: str = "rtl",
        points_total: int | None = None,
        points_hit: int | None = None,
        pct: float | None = None,
        uncovered: Any = None,
        dat_path: str = "",
        annotated_dir: str = "",
        run_id: str = "",
        build_id: str = "",
        attempt: int | None = None,
        source: str = "gate",
    ) -> bool:
        try:
            conn = self._writer_conn()
            try:
                conn.executescript(_SCHEMA)
                _migrate(conn)
                conn.execute(
                    "INSERT INTO coverage_results (ts, block, scope, points_total, "
                    "points_hit, pct, uncovered, dat_path, annotated_dir, run_id, build_id, attempt, source) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        time.time(), block, scope, points_total, points_hit, pct,
                        _json(uncovered), dat_path or "", annotated_dir or "",
                        run_id or "", build_id or "",
                        (int(attempt) if attempt is not None else None), source or "gate",
                    ),
                )
            finally:
                conn.close()
            return True
        except Exception:  # noqa: BLE001
            return False

    # ------------------------------------------------------------------
    # Build-scoped reads (the evidence a recorded build is judged by)
    # ------------------------------------------------------------------
    def rows_for_build(self, build_id: str) -> dict[str, list[dict]]:
        """Every measurement row recorded under ``build_id``:
        ``{dv: [...], ppa: [...], coverage: [...]}`` (oldest first)."""
        out: dict[str, list[dict]] = {"dv": [], "ppa": [], "coverage": []}
        if not build_id:
            return out
        conn = self._reader_conn()
        if conn is None:
            return out
        try:
            for key, table in (("dv", "dv_results"), ("ppa", "ppa_history"), ("coverage", "coverage_results")):
                try:
                    out[key] = self._rows(conn.execute(
                        f"SELECT * FROM {table} WHERE build_id = ? ORDER BY id", (build_id,)))
                except Exception:  # noqa: BLE001 - an old database without the column
                    out[key] = []
            return out
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Reads (never raise: return [] / None on any failure)
    # ------------------------------------------------------------------
    @staticmethod
    def _rows(cur) -> list[dict]:
        return [dict(r) for r in cur.fetchall()]

    def latest_dv(
        self, block: str | None = None, scope: str | None = None,
    ) -> list[dict]:
        """Latest row per (block, scope). Filter by block/scope when given."""
        conn = self._reader_conn()
        if conn is None:
            return []
        try:
            where = []
            params: list[Any] = []
            if block:
                where.append("block = ?")
                params.append(block)
            if scope:
                where.append("scope = ?")
                params.append(scope)
            clause = ("WHERE " + " AND ".join(where)) if where else ""
            sql = (
                "SELECT * FROM dv_results WHERE id IN "
                f"(SELECT MAX(id) FROM dv_results {clause} GROUP BY block, scope) "
                "ORDER BY block, scope"
            )
            return self._rows(conn.execute(sql, params))
        except Exception:  # noqa: BLE001
            return []
        finally:
            conn.close()

    def dv_rows(self, block: str | None = None) -> list[dict]:
        conn = self._reader_conn()
        if conn is None:
            return []
        try:
            if block:
                cur = conn.execute(
                    "SELECT * FROM dv_results WHERE block = ? ORDER BY ts", (block,),
                )
            else:
                cur = conn.execute("SELECT * FROM dv_results ORDER BY ts")
            return self._rows(cur)
        except Exception:  # noqa: BLE001
            return []
        finally:
            conn.close()

    def latest_ppa(self, block: str) -> dict | None:
        conn = self._reader_conn()
        if conn is None:
            return None
        try:
            cur = conn.execute(
                "SELECT * FROM ppa_history WHERE block = ? ORDER BY id DESC LIMIT 1",
                (block,),
            )
            row = cur.fetchone()
            return dict(row) if row else None
        except Exception:  # noqa: BLE001
            return None
        finally:
            conn.close()

    def ppa_rows(self, block: str) -> list[dict]:
        conn = self._reader_conn()
        if conn is None:
            return []
        try:
            return self._rows(conn.execute(
                "SELECT * FROM ppa_history WHERE block = ? ORDER BY ts", (block,),
            ))
        except Exception:  # noqa: BLE001
            return []
        finally:
            conn.close()

    def coverage_latest(self, block: str) -> dict | None:
        conn = self._reader_conn()
        if conn is None:
            return None
        try:
            cur = conn.execute(
                "SELECT * FROM coverage_results WHERE block = ? "
                "ORDER BY id DESC LIMIT 1",
                (block,),
            )
            row = cur.fetchone()
            return dict(row) if row else None
        except Exception:  # noqa: BLE001
            return None
        finally:
            conn.close()

    def coverage_rows(self, block: str) -> list[dict]:
        conn = self._reader_conn()
        if conn is None:
            return []
        try:
            return self._rows(conn.execute(
                "SELECT * FROM coverage_results WHERE block = ? ORDER BY ts",
                (block,),
            ))
        except Exception:  # noqa: BLE001
            return []
        finally:
            conn.close()
