# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Leases: process locks that live in the project database (C1).

Run state used to sit in flocks, PID files and in-memory sets outside
``project.sqlite`` (the chip-lead ledger lock, the integration-sim flock,
daemon.json, ``_consumed_interrupt_ids``). A crash left stale files behind and
a restart forgot the in-memory state. A lease is a named row with a holder
pid/host, an opaque token and an expiry:

* ``acquire`` succeeds when the row is free, expired, or held by a dead pid on
  this host (the displaced holder is recorded in ``stolen_from_json``);
* ``renew``/``release`` require the token, so a slow old holder can never
  release a replacement's lease (the daemon.json race of 2026-05-19);
* cross-host holders are only ever displaced by expiry -- the pid check is
  meaningless off-host and SQLite/WAL is single-host anyway.

``db_lease`` is the blocking context manager most call sites want: it polls
for the lease, heartbeats at ``ttl/3`` from a daemon thread, and releases on
exit. All writes run in ``BEGIN IMMEDIATE`` transactions, which is what makes
the check-then-set atomic across processes.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import threading
import time
import uuid
from typing import Any

from orchestrator.state_store.procutil import hostname, pid_is_alive


class LeaseUnavailable(RuntimeError):
    """Raised by ``db_lease`` when the lease could not be acquired in ``wait_s``."""


def _row(r) -> dict:
    d = dict(r)
    d["meta"] = json.loads(d.pop("meta_json") or "{}")
    d["stolen_from"] = json.loads(d.pop("stolen_from_json") or "null")
    d["expired"] = float(d.get("expires_ts") or 0) < time.time()
    return d


class LeaseMixin:
    """Lease primitives mixed into ``ProjectDB`` (needs ``_tx``/``_conn``)."""

    def acquire_lease(self, name: str, ttl_s: float, *, meta: dict | None = None,
                      steal_if_expired: bool = True, pid: int | None = None) -> str | None:
        """Take ``name`` for ``ttl_s`` seconds; returns the token or ``None``."""
        import os
        now = time.time()
        my_pid = int(pid if pid is not None else os.getpid())
        my_host = hostname()
        token = uuid.uuid4().hex
        with self._tx() as db:
            row = db.execute("SELECT * FROM leases WHERE name=?", (name,)).fetchone()
            stolen = None
            if row is not None:
                # A live lease is never re-issued, not even to its own pid: two
                # threads of one process must serialize like two processes do,
                # and re-issuing would silently invalidate the first token.
                expired = float(row["expires_ts"]) < now
                dead = (row["holder_host"] == my_host and not pid_is_alive(row["holder_pid"]))
                if not (steal_if_expired and (expired or dead)):
                    return None
                stolen = {**dict(row), "reason": "expired" if expired else "holder_dead",
                          "stolen_ts": now}
                stolen.pop("stolen_from_json", None)
            db.execute(
                "INSERT INTO leases(name, holder_pid, holder_host, token, acquired_ts, "
                "expires_ts, meta_json, stolen_from_json) VALUES (?,?,?,?,?,?,?,?) "
                "ON CONFLICT(name) DO UPDATE SET holder_pid=excluded.holder_pid, "
                "holder_host=excluded.holder_host, token=excluded.token, "
                "acquired_ts=excluded.acquired_ts, expires_ts=excluded.expires_ts, "
                "meta_json=excluded.meta_json, stolen_from_json=excluded.stolen_from_json",
                (name, my_pid, my_host, token, now, now + float(ttl_s),
                 json.dumps({**(meta or {}), "ttl_s": float(ttl_s)}, default=str),
                 json.dumps(stolen, default=str) if stolen else None),
            )
        return token

    def renew_lease(self, name: str, token: str, ttl_s: float) -> bool:
        with self._tx() as db:
            cur = db.execute("UPDATE leases SET expires_ts=? WHERE name=? AND token=?",
                             (time.time() + float(ttl_s), name, token))
            return cur.rowcount == 1

    def release_lease(self, name: str, token: str) -> bool:
        with self._tx() as db:
            cur = db.execute("DELETE FROM leases WHERE name=? AND token=?", (name, token))
            return cur.rowcount == 1

    def steal_lease(self, name: str, reason: str) -> dict | None:
        """Operator override: drop ``name`` whatever its state; returns the old row."""
        with self._tx() as db:
            row = db.execute("SELECT * FROM leases WHERE name=?", (name,)).fetchone()
            if row is None:
                return None
            db.execute("DELETE FROM leases WHERE name=?", (name,))
        old = _row(row)
        old["stolen_reason"] = reason
        return old

    def lease(self, name: str) -> dict | None:
        with self._conn() as db:
            row = db.execute("SELECT * FROM leases WHERE name=?", (name,)).fetchone()
        return _row(row) if row else None

    def leases(self) -> list[dict]:
        with self._conn() as db:
            rows = db.execute("SELECT * FROM leases ORDER BY name").fetchall()
        return [_row(r) for r in rows]

    # ------------------------------------------------------------ run flags
    def run_id(self) -> str:
        return self.get_setting("run_id", "") or ""

    def set_flag(self, name: str, value: Any, *, run_id: str | None = None) -> None:
        rid = self.run_id() if run_id is None else run_id
        with self._tx() as db:
            db.execute(
                "INSERT INTO run_flags(name, run_id, value_json, ts) VALUES (?,?,?,?) "
                "ON CONFLICT(name, run_id) DO UPDATE SET value_json=excluded.value_json, "
                "ts=excluded.ts", (name, rid, json.dumps(value, default=str), time.time()))

    def get_flag(self, name: str, default: Any = None, *, run_id: str | None = None) -> Any:
        rid = self.run_id() if run_id is None else run_id
        with self._conn() as db:
            row = db.execute("SELECT value_json FROM run_flags WHERE name=? AND run_id=?",
                             (name, rid)).fetchone()
        if row is None:
            return default
        try:
            return json.loads(row["value_json"])
        except (TypeError, ValueError):
            return default

    def clear_flag(self, name: str, *, run_id: str | None = None) -> None:
        rid = self.run_id() if run_id is None else run_id
        with self._tx() as db:
            db.execute("DELETE FROM run_flags WHERE name=? AND run_id=?", (name, rid))

    # ------------------------------------------------------------ decisions
    def add_decision(self, *, action: str, interrupt_type: str = "", block: str = "",
                     reasoning: str = "", interrupt_id: str = "", actor: str = "",
                     run_id: str | None = None) -> int:
        """Append a decision (who answered a park: ``actor``); returns its
        1-based index in this run."""
        rid = self.run_id() if run_id is None else run_id
        with self._tx() as db:
            n = db.execute("SELECT COUNT(*) FROM decisions WHERE run_id=?", (rid,)).fetchone()[0]
            db.execute(
                "INSERT INTO decisions(interrupt_id, interrupt_type, block, action, reasoning, "
                "decision_index, run_id, ts, actor) VALUES (?,?,?,?,?,?,?,?,?)",
                (interrupt_id, interrupt_type, block, action, reasoning, int(n) + 1, rid,
                 time.time(), actor or ""))
        return int(n) + 1

    def decision_count(self, *, run_id: str | None = None, actor: str | None = None) -> int:
        rid = self.run_id() if run_id is None else run_id
        sql, args = "SELECT COUNT(*) FROM decisions WHERE run_id=?", [rid]
        if actor is not None:
            sql += " AND actor=?"
            args.append(actor)
        with self._conn() as db:
            return int(db.execute(sql, tuple(args)).fetchone()[0])

    def decisions(self, *, run_id: str | None = None, last: int | None = None) -> list[dict]:
        rid = self.run_id() if run_id is None else run_id
        with self._conn() as db:
            rows = db.execute("SELECT * FROM decisions WHERE run_id=? ORDER BY id",
                              (rid,)).fetchall()
        out = [dict(r) for r in rows]
        return out[-last:] if last else out


@contextlib.contextmanager
def db_lease(db, name: str, ttl_s: float = 60.0, *, wait_s: float = 0.0,
             poll_s: float = 0.25, heartbeat: bool = True, meta: dict | None = None):
    """Hold lease ``name`` for the block; blocks up to ``wait_s`` to get it.

    Raises ``LeaseUnavailable`` when the lease is still held (by a live holder)
    after ``wait_s``. Yields the token.
    """
    deadline = time.time() + max(0.0, float(wait_s))
    token = db.acquire_lease(name, ttl_s, meta=meta)
    while token is None and time.time() < deadline:
        time.sleep(poll_s)
        token = db.acquire_lease(name, ttl_s, meta=meta)
    if token is None:
        holder = db.lease(name) or {}
        raise LeaseUnavailable(
            f"lease {name!r} held by pid {holder.get('holder_pid')}@"
            f"{holder.get('holder_host')} until {holder.get('expires_ts')}")
    stop = threading.Event()
    hb: threading.Thread | None = None
    if heartbeat:
        def _beat():
            while not stop.wait(max(0.5, float(ttl_s) / 3.0)):
                try:
                    if not db.renew_lease(name, token, ttl_s):
                        return  # stolen or released: stop beating
                except Exception:  # noqa: BLE001 - a failed beat lets it expire
                    return
        hb = threading.Thread(target=_beat, name=f"lease-{name}", daemon=True)
        hb.start()
    try:
        yield token
    finally:
        stop.set()
        if hb is not None:
            hb.join(timeout=2.0)
        with contextlib.suppress(Exception):
            db.release_lease(name, token)


@contextlib.asynccontextmanager
async def adb_lease(db, name: str, ttl_s: float = 60.0, *, wait_s: float = 0.0,
                    poll_s: float = 0.25, heartbeat: bool = True, meta: dict | None = None):
    """``db_lease`` for coroutines: the acquire wait never blocks the loop."""
    deadline = time.time() + max(0.0, float(wait_s))
    token = await asyncio.to_thread(db.acquire_lease, name, ttl_s, meta=meta)
    while token is None and time.time() < deadline:
        await asyncio.sleep(poll_s)
        token = await asyncio.to_thread(db.acquire_lease, name, ttl_s, meta=meta)
    if token is None:
        holder = db.lease(name) or {}
        raise LeaseUnavailable(
            f"lease {name!r} held by pid {holder.get('holder_pid')}@"
            f"{holder.get('holder_host')} until {holder.get('expires_ts')}")
    stop = threading.Event()
    hb: threading.Thread | None = None
    if heartbeat:
        def _beat():
            while not stop.wait(max(0.5, float(ttl_s) / 3.0)):
                try:
                    if not db.renew_lease(name, token, ttl_s):
                        return
                except Exception:  # noqa: BLE001
                    return
        hb = threading.Thread(target=_beat, name=f"lease-{name}", daemon=True)
        hb.start()
    try:
        yield token
    finally:
        stop.set()
        if hb is not None:
            hb.join(timeout=2.0)
        with contextlib.suppress(Exception):
            db.release_lease(name, token)
