# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Audit F9: /run/state block counters are attempt-scoped, not append-only.

completed_blocks accumulates one entry per completion EVENT across resumes /
re-validation passes -- the reference codec decoder reported completed_count=84 for 21
blocks and remaining_count=-63. The shaped state must count unique blocks
(latest event wins) and floor remaining at zero, while preserving the raw
event count for forensics.
"""
from __future__ import annotations

from types import SimpleNamespace

from orchestrator.daemon.server import _shape_state


def _snap(completed, queue):
    return SimpleNamespace(
        values={"completed_blocks": completed, "block_queue": queue},
        tasks=[],
        next=[],
    )


def test_completed_count_dedupes_resume_events():
    completed = [
        {"name": "a", "success": False, "attempts": 1},
        {"name": "b", "success": True, "attempts": 1},
        {"name": "a", "success": True, "attempts": 2},
        {"name": "a", "success": True, "attempts": 3},
        {"name": "b", "success": True, "attempts": 2},
    ]
    queue = [{"name": "a"}, {"name": "b"}, {"name": "c"}]
    out = _shape_state(_snap(completed, queue))
    assert out["completed_count"] == 2
    assert out["completion_events"] == 5          # raw ledger preserved
    assert out["remaining_count"] == 1
    rows = {r["name"]: r for r in out["completed_blocks"]}
    assert len(rows) == 2
    # the LATEST completion event wins (final attempt outcome)
    assert rows["a"]["success"] is True and rows["a"]["attempts"] == 3


def test_remaining_count_never_negative():
    completed = [{"name": n, "success": True, "attempts": k}
                 for k in range(1, 5) for n in ("a", "b")]
    out = _shape_state(_snap(completed, [{"name": "a"}, {"name": "b"}]))
    assert out["completed_count"] == 2
    assert out["completion_events"] == 8
    assert out["remaining_count"] == 0


def test_single_pass_run_unchanged():
    completed = [{"name": "a", "success": True, "attempts": 1}]
    out = _shape_state(_snap(completed, [{"name": "a"}, {"name": "b"}]))
    assert out["completed_count"] == 1
    assert out["completion_events"] == 1
    assert out["remaining_count"] == 1


def _snap_with_interrupt(completed, queue, block_name):
    """Snapshot with one PARKED interrupt for ``block_name``."""
    intr = SimpleNamespace(id="i0", value={"type": "ask_human",
                                           "block_name": block_name})
    task = SimpleNamespace(interrupts=[intr])
    snap = _snap(completed, queue)
    return SimpleNamespace(values=snap.values, next=snap.next, tasks=[task])


def test_live_interrupt_for_a_previously_completed_block_is_not_hidden():
    """Regression: a re-entered block's LIVE interrupt was silently dropped.

    ``completed_blocks`` is append-only across attempts, so a block that
    completed once and was later re-entered (revise_interface, fix_rtl, a
    re-spec) stays in the set forever. The old filter treated any interrupt
    whose block was in that set as stale and removed it, so /run/state
    reported ``pending_interrupt_count: 0`` and ``interrupts: []`` while the
    graph was parked on it -- and the next POST /run/resume returned
    ``{"resumed": true, "interrupts": 1}``.
    """
    snap = _snap_with_interrupt(
        completed=[{"name": "blk_a", "success": True}],
        queue=["blk_a", "blk_b"],
        block_name="blk_a",           # same block, now parked again
    )
    out = _shape_state(snap)
    assert out["pending_interrupt_count"] == 1, "live interrupt was hidden"
    assert len(out["interrupts"]) == 1
    assert out["interrupts"][0]["stale_suspected"] is True
    assert out["suspected_stale_interrupt_count"] == 1
    assert out["live_interrupt_count"] == 0


def test_interrupt_for_an_uncompleted_block_is_not_flagged_stale():
    snap = _snap_with_interrupt(
        completed=[{"name": "blk_a", "success": True}],
        queue=["blk_a", "blk_b"],
        block_name="blk_b",
    )
    out = _shape_state(snap)
    assert out["pending_interrupt_count"] == 1
    assert out["interrupts"][0]["stale_suspected"] is False
    assert out["live_interrupt_count"] == 1
    assert out["suspected_stale_interrupt_count"] == 0
