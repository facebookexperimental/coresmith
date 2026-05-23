# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Benchmark results cache backed by SQLite.

Stores synthesis benchmark results keyed on (component, params, pdk, clock_mhz)
so that the same benchmark isn't re-synthesized across multiple conversations.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

DEFAULT_DB_PATH = ".coresmith/benchmark_cache.db"


def _params_hash(params: dict) -> str:
    """Compute a stable hash of benchmark parameters."""
    canonical = json.dumps(params, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


class BenchmarkCache:
    """SQLite-backed cache for benchmark synthesis results."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._ensure_table()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _ensure_table(self) -> None:
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS benchmarks (
                component   TEXT NOT NULL,
                params_hash TEXT NOT NULL,
                pdk_name    TEXT NOT NULL,
                clock_mhz   REAL NOT NULL,
                params_json TEXT NOT NULL,
                result_json TEXT NOT NULL,
                created_at  TEXT DEFAULT (datetime('now')),
                PRIMARY KEY (component, params_hash, pdk_name, clock_mhz)
            )
        """)
        conn.commit()

    def get(
        self,
        component: str,
        params: dict,
        pdk_name: str,
        clock_mhz: float,
    ) -> dict[str, Any] | None:
        """Retrieve a cached benchmark result.

        Returns None on cache miss.
        """
        ph = _params_hash(params)
        conn = self._get_conn()
        row = conn.execute(
            """
            SELECT result_json FROM benchmarks
            WHERE component = ? AND params_hash = ? AND pdk_name = ? AND clock_mhz = ?
            """,
            (component, ph, pdk_name, clock_mhz),
        ).fetchone()

        if row is None:
            return None

        result = json.loads(row["result_json"])
        result["cached"] = True
        return result

    def store(
        self,
        component: str,
        params: dict,
        pdk_name: str,
        clock_mhz: float,
        result: dict[str, Any],
    ) -> None:
        """Store a benchmark result in the cache."""
        ph = _params_hash(params)
        conn = self._get_conn()
        conn.execute(
            """
            INSERT OR REPLACE INTO benchmarks
                (component, params_hash, pdk_name, clock_mhz, params_json, result_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (component, ph, pdk_name, clock_mhz,
             json.dumps(params, sort_keys=True),
             json.dumps(result, default=str)),
        )
        conn.commit()

    def clear(self) -> None:
        """Clear all cached results."""
        conn = self._get_conn()
        conn.execute("DELETE FROM benchmarks")
        conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
