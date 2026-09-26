# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Parked interrupts as first-class rows in the project database (C1-3).

"Parked" used to be derivable only from LangGraph's checkpoint, and a parked
branch could not be answered while its ``Send()`` siblings were still running
(``/run/resume`` returned 409). Every park now:

1. mints a deterministic ``interrupt_id`` (so the node's re-execution on
   resume finds the same row instead of minting a new one),
2. inserts a ``pending`` row with the payload, and
3. optionally waits ``CORESMITH_INTERRUPT_WAIT_S`` for a resolution to land in
   the table before raising ``interrupt()`` at all -- a ruling or a targeted
   resume answers the branch while the siblings keep running.

The daemon binds LangGraph's own ``Interrupt.id`` to the row when it sees it
(``bind_lg_id``) and, at the next superstep boundary, resumes ONLY the
branches whose rows are ``resolved`` (``resolved_for``). This module stays
langgraph-free: the graph code calls it around ``interrupt()``.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any

_STABLE_KEYS = ("type", "block_name", "block", "attempt", "phase", "stage", "node",
                "question_ids", "round")


def wait_seconds() -> float:
    """How long a park waits in the DB before raising ``interrupt()``
    (CORESMITH_INTERRUPT_WAIT_S, default 0 = raise immediately)."""
    try:
        return max(0.0, float(os.environ.get("CORESMITH_INTERRUPT_WAIT_S", "0") or 0))
    except ValueError:
        return 0.0


def interrupt_id_for(payload: dict, *, graph: str, node: str, run_id: str) -> str:
    """A deterministic id: the same park re-executed after a resume maps to
    the same row. Payload keys that vary per park (attempt, block, type, ...)
    are part of the key; free text is not."""
    key = {k: payload.get(k) for k in _STABLE_KEYS if k in payload}
    key.update(graph=graph, node=node, run_id=run_id)
    return "int-" + hashlib.sha1(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _row(r) -> dict:
    d = dict(r)
    d["payload"] = json.loads(d.pop("payload_json") or "{}")
    d["resolution"] = json.loads(d.pop("resolution_json") or "null")
    return d


class InterruptMixin:
    """Interrupt-table primitives mixed into ``ProjectDB``."""

    def park_interrupt(self, payload: dict, *, graph: str, node: str,
                       block: str | None = None, kind: str | None = None,
                       branch: str | None = None) -> tuple[str, dict]:
        """Record a park; idempotent on the deterministic id. Returns
        ``(interrupt_id, payload_with_id)``."""
        rid = self.run_id()
        # Always derived from the stable keys: a payload copied from another
        # park (a new attempt, another block) gets its own row, and the same
        # park re-executed after a resume lands on the same one.
        iid = interrupt_id_for(payload, graph=graph, node=node, run_id=rid)
        # Mutate in place: callers (and tests) hold the payload object and
        # expect the parked value to BE it.
        payload["interrupt_id"] = iid
        out = payload
        with self._tx() as db:
            row = db.execute("SELECT status FROM interrupts WHERE id=?", (iid,)).fetchone()
            if row is None:
                db.execute(
                    "INSERT INTO interrupts(id, graph, branch, node, block, kind, payload_json, "
                    "status, run_id, ts) VALUES (?,?,?,?,?,?,?,'pending',?,?)",
                    (iid, graph, branch, node,
                     block or payload.get("block_name") or payload.get("block") or "",
                     kind or str(payload.get("type") or ""),
                     json.dumps(payload, default=str), rid, time.time()))
            elif row["status"] == "consumed":
                # The same park, raised again after its answer was used: a
                # fresh pending row (re-open) so a new answer can be queued.
                db.execute("UPDATE interrupts SET status='pending', resolution_json=NULL, "
                           "resolved_by=NULL, resolved_ts=NULL, consumed_ts=NULL, ts=?, "
                           "payload_json=? WHERE id=?",
                           (time.time(), json.dumps(payload, default=str), iid))
        return iid, out

    def bind_lg_id(self, interrupt_id: str, lg_interrupt_id: str) -> None:
        with self._tx() as db:
            db.execute("UPDATE interrupts SET lg_interrupt_id=? WHERE id=? AND "
                       "(lg_interrupt_id IS NULL OR lg_interrupt_id<>?)",
                       (lg_interrupt_id, interrupt_id, lg_interrupt_id))

    def resolve_interrupt(self, interrupt_id: str, resolution: dict, *,
                          resolved_by: str = "human") -> bool:
        """Queue an answer for a pending park. False when no pending row."""
        with self._tx() as db:
            cur = db.execute(
                "UPDATE interrupts SET status='resolved', resolution_json=?, resolved_by=?, "
                "resolved_ts=? WHERE id=? AND status='pending'",
                (json.dumps(resolution, default=str), resolved_by, time.time(), interrupt_id))
            return cur.rowcount == 1

    def consume_interrupt(self, interrupt_id: str) -> dict | None:
        """Mark a resolved row consumed and return its resolution."""
        with self._tx() as db:
            row = db.execute("SELECT * FROM interrupts WHERE id=? AND status='resolved'",
                             (interrupt_id,)).fetchone()
            if row is None:
                return None
            db.execute("UPDATE interrupts SET status='consumed', consumed_ts=? WHERE id=?",
                       (time.time(), interrupt_id))
        return _row(row)["resolution"]

    def consume_lg_interrupts(self, lg_ids: list[str]) -> list[str]:
        """Mark the rows behind these LangGraph ids consumed (a plain resume
        answered them). Returns the coresmith ids touched."""
        if not lg_ids:
            return []
        with self._tx() as db:
            q = ",".join("?" * len(lg_ids))
            rows = db.execute(f"SELECT id FROM interrupts WHERE lg_interrupt_id IN ({q}) "
                              "AND status IN ('pending','resolved')", tuple(lg_ids)).fetchall()
            ids = [r["id"] for r in rows]
            if ids:
                q2 = ",".join("?" * len(ids))
                db.execute(f"UPDATE interrupts SET status='consumed', consumed_ts=? "
                           f"WHERE id IN ({q2})", (time.time(), *ids))
        return ids

    def abandon_interrupt(self, interrupt_id: str) -> bool:
        with self._tx() as db:
            cur = db.execute("UPDATE interrupts SET status='abandoned' WHERE id=? AND "
                             "status IN ('pending','resolved')", (interrupt_id,))
            return cur.rowcount == 1

    def interrupt(self, interrupt_id: str) -> dict | None:
        with self._conn() as db:
            row = db.execute("SELECT * FROM interrupts WHERE id=?", (interrupt_id,)).fetchone()
        return _row(row) if row else None

    def interrupts(self, *, status: str | None = None, graph: str | None = None,
                   run_id: str | None = None) -> list[dict]:
        rid = self.run_id() if run_id is None else run_id
        sql, args = "SELECT * FROM interrupts WHERE run_id=?", [rid]
        if status:
            sql += " AND status=?"
            args.append(status)
        if graph:
            sql += " AND graph=?"
            args.append(graph)
        with self._conn() as db:
            rows = db.execute(sql + " ORDER BY ts", tuple(args)).fetchall()
        return [_row(r) for r in rows]

    def resolved_for(self, lg_ids: list[str]) -> dict[str, dict]:
        """``{lg_interrupt_id: resolution}`` for resolved rows among ``lg_ids``."""
        if not lg_ids:
            return {}
        q = ",".join("?" * len(lg_ids))
        with self._conn() as db:
            rows = db.execute(f"SELECT * FROM interrupts WHERE status='resolved' AND "
                              f"lg_interrupt_id IN ({q})", tuple(lg_ids)).fetchall()
        return {r["lg_interrupt_id"]: _row(r)["resolution"] for r in rows}

    def wait_for_resolution(self, interrupt_id: str, wait_s: float,
                            poll_s: float = 2.0) -> dict | None:
        """Poll up to ``wait_s`` for an answer; consumes and returns it."""
        deadline = time.time() + max(0.0, wait_s)
        while True:
            res = self.consume_interrupt(interrupt_id)
            if res is not None:
                return res
            if time.time() >= deadline:
                return None
            time.sleep(min(poll_s, max(0.05, deadline - time.time())))


def park_and_wait(db, payload: dict, *, graph: str, node: str, block: str | None = None,
                  kind: str | None = None) -> tuple[dict, Any]:
    """Record the park, optionally wait for a queued answer.

    Returns ``(payload_with_id, resolution_or_None)``. The caller raises
    ``interrupt(payload_with_id)`` when the resolution is ``None``.
    """
    iid, out = db.park_interrupt(payload, graph=graph, node=node, block=block, kind=kind)
    wait = wait_seconds()
    res = db.wait_for_resolution(iid, wait) if wait > 0 else db.consume_interrupt(iid)
    return out, res
