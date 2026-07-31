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

import time
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


def _snap_with_interrupt(completed, queue, block_name, iid="i0"):
    """Snapshot with one PARKED interrupt for ``block_name``."""
    intr = SimpleNamespace(id=iid, value={"type": "ask_human",
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


# --------------------------------------------------------------------------- #
# Staleness is keyed to WHEN THE INTERRUPT WAS RAISED, not to bare membership
# in the append-only completed_blocks (and not to time since the last resume).
# --------------------------------------------------------------------------- #

def test_a_just_raised_interrupt_on_a_completed_block_reads_LIVE(monkeypatch):
    """THE BUG: in a two-pass run every block completes pass 1, so every pass-2
    interrupt was born ``stale_suspected: true`` / ``live_interrupt_count: 0``
    -- and an automated consumer concludes there is nothing to act on. Observed
    on a parked hands-off raster run whose ``contract_conformance_unrepairable``
    interrupt was entirely live."""
    import orchestrator.daemon.server as srv

    monkeypatch.setattr(srv, "_interrupt_first_seen", {}, raising=False)
    snap = _snap_with_interrupt(
        completed=[{"name": "blk_a", "success": True, "completed_at": 1000.0}],
        queue=["blk_a", "blk_b"],
        block_name="blk_a",
        iid="fresh",
    )
    out = _shape_state(snap)          # first sight => raised now >> 1000.0
    row = out["interrupts"][0]
    assert row["stale_suspected"] is False
    assert row["stale_basis"] == "raised_after_block_completed"
    assert out["live_interrupt_count"] == 1
    assert out["suspected_stale_interrupt_count"] == 0


def test_an_interrupt_the_graph_moved_PAST_is_still_flagged_stale(monkeypatch):
    """The leftover the label exists for: the interrupt was raised, answered,
    and the block completed AFTERWARDS -- so the checkpoint's copy is stale."""
    import orchestrator.daemon.server as srv

    monkeypatch.setattr(srv, "_interrupt_first_seen", {"old": 10.0},
                        raising=False)
    snap = _snap_with_interrupt(
        completed=[{"name": "blk_a", "success": True,
                    "completed_at": time.time()}],
        queue=["blk_a", "blk_b"],
        block_name="blk_a",
        iid="old",
    )
    out = _shape_state(snap)
    row = out["interrupts"][0]
    assert row["stale_suspected"] is True
    assert row["stale_basis"] == "raised_before_block_completed"
    assert out["suspected_stale_interrupt_count"] == 1
    assert out["pending_interrupt_count"] == 1     # surfaced, never hidden


def test_an_unstamped_completion_is_no_evidence_and_reads_LIVE(monkeypatch):
    """Checkpoints written before ``completed_at`` existed carry no completion
    time. Absence of evidence is not evidence of absence -- this file's whole
    thesis -- so the interrupt is reported LIVE, with the basis stated."""
    import orchestrator.daemon.server as srv

    monkeypatch.setattr(srv, "_interrupt_first_seen", {"legacy": 10.0},
                        raising=False)
    snap = _snap_with_interrupt(
        completed=[{"name": "blk_a", "success": True}],   # no completed_at
        queue=["blk_a", "blk_b"],
        block_name="blk_a",
        iid="legacy",
    )
    out = _shape_state(snap)
    row = out["interrupts"][0]
    assert row["stale_suspected"] is False
    assert row["stale_basis"] == "completion_unstamped"
    assert out["live_interrupt_count"] == 1


def test_first_seen_is_remembered_across_polls_and_pruned_when_gone(monkeypatch):
    """The raise time must not drift forward on every poll (that is what made
    the age unusable), and an id that stops being pending is forgotten."""
    import orchestrator.daemon.server as srv

    monkeypatch.setattr(srv, "_interrupt_first_seen", {}, raising=False)
    srv._note_interrupts_seen({"a"}, now=100.0)
    srv._note_interrupts_seen({"a"}, now=500.0)
    assert srv._interrupt_raised_ts("a") == 100.0
    srv._note_interrupts_seen({"b"}, now=900.0)
    assert "a" not in srv._interrupt_first_seen
    assert srv._interrupt_raised_ts("b") == 900.0
