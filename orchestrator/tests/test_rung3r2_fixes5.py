# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""[rung3r2-fixes-5] Three operational mechanisms proven live in the rung-3
video_codec feasibility rerun (run2):

  1. OPERATOR_SPEC_PIN -- generate_uarch_spec_node skips REGEN when the operator
     pinned the on-disk uarch spec (the review/mem-price gate still prices it).
  2. run-pause reaps the in-flight LLM CLI child's whole process group
     (reap_active_cli_processes) so an orphaned codex stops burning tokens.
  3. Fresh-session escalation -- after two consecutive identical mem-price
     manifests the NEXT regen drops the sticky codex session resume + prepends a
     mandatory-directives preamble; convergent rounds keep resuming.

Fully hermetic: no live LLM/EDA. ``generate_uarch_spec`` is monkeypatched, the
process registry is exercised with a fake ``Popen`` + monkeypatched
``os.killpg``, and the mem-price gate runs on disk with a cold characterizer.
"""
from __future__ import annotations

import asyncio
import json
import signal
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.langgraph import mem_price as mp
from orchestrator.langgraph import pipeline_graph as pg


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _spec(root, block, text):
    d = Path(root) / "arch" / "uarch_specs"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{block}.md").write_text(text)


def _blkdir(root, block):
    d = Path(root) / ".coresmith" / "blocks" / block
    d.mkdir(parents=True, exist_ok=True)
    return d


def _run_node(state):
    return asyncio.run(pg.generate_uarch_spec_node(state))


def _capture_events(monkeypatch):
    events: list = []
    monkeypatch.setattr(
        pg, "write_graph_event",
        lambda pr, node, etype, payload=None: events.append((node, etype, payload)),
    )
    return events


def _mock_generate(monkeypatch, session_id="new-sid"):
    m = AsyncMock(return_value={
        "spec_text": "SPEC", "spec_summary": {}, "block_name": "b",
        "session_id": session_id,
    })
    monkeypatch.setattr(pg, "generate_uarch_spec", m)
    return m


# ===========================================================================
# Item 1 -- OPERATOR_SPEC_PIN
# ===========================================================================

class TestOperatorSpecPin:
    def _state(self, root, block="entropy", human_response=None):
        return {
            "current_block": {"name": block, "rtl_target": f"rtl/{block}.v",
                              "python_source": "", "description": ""},
            "project_root": str(root),
            "human_response": human_response or {},
        }

    def test_pin_skips_regen_and_emits_event_with_rationale(self, tmp_path, monkeypatch):
        events = _capture_events(monkeypatch)
        gen = _mock_generate(monkeypatch)
        _spec(tmp_path, "entropy", "# MEM x: 8x256 ports=1rw impl=fpmem justification=y\n")
        pin = _blkdir(tmp_path, "entropy") / "OPERATOR_SPEC_PIN"
        pin.write_text("operator pin: spec settled, do not regenerate")

        out = _run_node(self._state(tmp_path))

        gen.assert_not_called()  # regeneration SKIPPED
        assert out == {"uarch_approved": False, "phase": "uarch"}
        pinned = [e for e in events if e[1] == "spec_pinned"]
        assert len(pinned) == 1
        assert pinned[0][2]["rationale"] == "operator pin: spec settled, do not regenerate"
        assert pinned[0][2]["block"] == "entropy"

    def test_no_pin_regenerates(self, tmp_path, monkeypatch):
        _capture_events(monkeypatch)
        gen = _mock_generate(monkeypatch)
        _spec(tmp_path, "entropy", "# MEM x: 8x256 ports=1rw impl=fpmem justification=y\n")
        _run_node(self._state(tmp_path))
        gen.assert_called_once()  # today's behavior: regen

    def test_ignore_env_regenerates_pinned(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CORESMITH_IGNORE_SPEC_PINS", "1")
        _capture_events(monkeypatch)
        gen = _mock_generate(monkeypatch)
        _spec(tmp_path, "entropy", "# MEM x: 8x256 ports=1rw impl=fpmem justification=y\n")
        (_blkdir(tmp_path, "entropy") / "OPERATOR_SPEC_PIN").write_text("pin")
        _run_node(self._state(tmp_path))
        gen.assert_called_once()  # escape hatch: regen despite the pin

    def test_pin_without_spec_on_disk_regenerates(self, tmp_path, monkeypatch):
        # A pin with no spec.md to protect must NOT skip (nothing to reuse).
        _capture_events(monkeypatch)
        gen = _mock_generate(monkeypatch)
        (_blkdir(tmp_path, "entropy") / "OPERATOR_SPEC_PIN").write_text("pin")
        _run_node(self._state(tmp_path))
        gen.assert_called_once()

    def test_gate_still_prices_and_can_fail_pinned_spec(self, tmp_path, monkeypatch):
        # The pin only prevents REGEN; the mem-price gate (review path) reads the
        # spec off disk regardless, so a pinned OVER-BUDGET spec STILL fails.
        monkeypatch.setattr(mp, "characterizer_warm", lambda pdk=None: False)
        _spec(tmp_path, "recon",
              "area_budget_um2 = 250000\n"
              "# MEM r: 8x235520 ports=1rw1r impl=sram justification=frame\n")
        (_blkdir(tmp_path, "recon") / "OPERATOR_SPEC_PIN").write_text("pin")
        r = pg._mem_price_gate_verdict(str(tmp_path), "recon")
        assert r is not None and r["action"] == "revise"  # gate not masked by the pin


# ===========================================================================
# Item 2 -- run-pause reaps the in-flight CLI child's process group
# ===========================================================================

from orchestrator.langchain.agents.coresmith_llm import (  # noqa: E402
    _active_processes,
    _active_processes_lock,
    _register_process,
    reap_active_cli_processes,
)


class TestPauseReap:
    def setup_method(self):
        with _active_processes_lock:
            _active_processes.clear()

    def teardown_method(self):
        with _active_processes_lock:
            _active_processes.clear()

    def test_registered_group_reaped_on_pause(self, monkeypatch):
        killpg_calls: list = []
        monkeypatch.setattr(
            "orchestrator.langchain.agents.coresmith_llm.os.killpg",
            lambda pgid, sig: killpg_calls.append((pgid, sig)),
        )
        proc = MagicMock(spec=subprocess.Popen)
        proc.poll.return_value = None  # still running
        proc.pid = 4242
        proc.wait.return_value = 0
        _register_process(proc)

        reaped = reap_active_cli_processes(grace_s=0.01)

        assert reaped == 1
        # SIGTERM the group, then SIGKILL any survivor -- same reap the watchdog uses.
        assert (4242, signal.SIGTERM) in killpg_calls
        assert (4242, signal.SIGKILL) in killpg_calls
        proc.wait.assert_called()  # direct child reaped within grace
        with _active_processes_lock:
            assert len(_active_processes) == 0  # registry drained

    def test_already_exited_not_reaped(self, monkeypatch):
        killpg_calls: list = []
        monkeypatch.setattr(
            "orchestrator.langchain.agents.coresmith_llm.os.killpg",
            lambda pgid, sig: killpg_calls.append((pgid, sig)),
        )
        proc = MagicMock(spec=subprocess.Popen)
        proc.poll.return_value = 0  # already exited
        proc.pid = 5
        _register_process(proc)
        assert reap_active_cli_processes() == 0
        assert killpg_calls == []  # nothing to reap

    def test_empty_registry_is_noop(self):
        # The resume-after-pause path (no in-flight call registered) is unaffected.
        assert reap_active_cli_processes() == 0


# ===========================================================================
# Item 3 -- fresh-session escalation for sticky respecs
# ===========================================================================

_OVER = ("area_budget_um2 = 250000\n"
         "# MEM m: 8x40000 ports=1rw impl=sram justification=x\n")


class TestFreshSessionGate:
    """_mem_price_gate_verdict: two identical rounds escalate (fresh session),
    then a still-identical round defers."""

    def _pg(self, monkeypatch):
        monkeypatch.setattr(mp, "characterizer_warm", lambda pdk=None: False)
        return pg

    def _ledger(self, root, block):
        return json.loads((Path(root) / ".coresmith" / "blocks" / block
                           / "mem_price.json").read_text())

    def test_two_identical_rounds_escalate_then_defer(self, tmp_path, monkeypatch):
        p = self._pg(monkeypatch)
        monkeypatch.setenv("CORESMITH_MEM_PRICE_MAX_REVISE", "3")
        _spec(tmp_path, "ent", _OVER)
        # round 1: over budget -> ordinary revise (no fresh escalation yet)
        r1 = p._mem_price_gate_verdict(str(tmp_path), "ent")
        assert r1["action"] == "revise" and not r1.get("fresh_session")
        # round 2 (identical spec): FRESH-SESSION escalation instead of defer
        r2 = p._mem_price_gate_verdict(str(tmp_path), "ent")
        assert r2["action"] == "revise" and r2["fresh_session"] is True
        blk = Path(tmp_path) / ".coresmith" / "blocks" / "ent"
        assert (blk / "mem_price_fresh_escalate").exists()   # one-shot node signal
        assert (blk / "mem_price_fresh_escalated").exists()  # persistent "done once"
        # round 3 (STILL identical): fresh session did not help -> defer
        r3 = p._mem_price_gate_verdict(str(tmp_path), "ent")
        assert r3 is None
        led = self._ledger(tmp_path, "ent")
        assert led["deferred"] is True and led["over_budget"] is True
        assert "fresh session" in led["deferred_reason"]

    def test_convergent_rounds_do_not_escalate(self, tmp_path, monkeypatch):
        p = self._pg(monkeypatch)
        monkeypatch.setenv("CORESMITH_MEM_PRICE_MAX_REVISE", "3")
        _spec(tmp_path, "irdc", _OVER)
        r1 = p._mem_price_gate_verdict(str(tmp_path), "irdc")
        assert r1["action"] == "revise" and not r1.get("fresh_session")
        # a DIFFERENT (worse) manifest -> not identical -> ordinary revise, no escalation
        _spec(tmp_path, "irdc", _OVER +
              "# MEM n: 8x40000 ports=1rw impl=sram justification=y\n")
        r2 = p._mem_price_gate_verdict(str(tmp_path), "irdc")
        assert r2["action"] == "revise" and not r2.get("fresh_session")
        blk = Path(tmp_path) / ".coresmith" / "blocks" / "irdc"
        assert not (blk / "mem_price_fresh_escalate").exists()


class TestFreshSessionNodeResume:
    """generate_uarch_spec_node: the fresh escalation drops resume + adds the
    preamble; convergent rounds resume the stored session id."""

    def _state(self, root, feedback="SHRINK the recon window"):
        return {
            "current_block": {"name": "ent", "rtl_target": "rtl/ent.v",
                              "python_source": "", "description": ""},
            "project_root": str(root),
            "human_response": {"action": "revise", "feedback": feedback},
        }

    def test_fresh_escalation_drops_resume_and_adds_preamble(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CORESMITH_CODEX_RESUME", "1")
        _capture_events(monkeypatch)
        gen = _mock_generate(monkeypatch, session_id="fresh-sid")
        _spec(tmp_path, "ent", "prev spec body\n")
        blk = _blkdir(tmp_path, "ent")
        (blk / "codex_session_id").write_text("stale-sid")   # would resume, but...
        (blk / "mem_price_fresh_escalate").write_text("2")   # ...escalation drops it

        _run_node(self._state(tmp_path))

        kw = gen.call_args.kwargs
        assert kw["resume_session_id"] is None  # FRESH session (resume dropped)
        assert "this is a fresh start" in kw["feedback"]
        assert "MANDATORY" in kw["feedback"]
        assert "SHRINK the recon window" in kw["feedback"]  # accumulated feedback kept
        assert "2 times" in kw["feedback"]                  # N surfaced
        assert not (blk / "mem_price_fresh_escalate").exists()  # one-shot consumed
        assert (blk / "codex_session_id").read_text() == "fresh-sid"  # new session stored

    def test_convergent_round_resumes_stored_session(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CORESMITH_CODEX_RESUME", "1")
        _capture_events(monkeypatch)
        gen = _mock_generate(monkeypatch, session_id="next-sid")
        _spec(tmp_path, "ent", "prev spec body\n")
        (_blkdir(tmp_path, "ent") / "codex_session_id").write_text("live-sid")
        # no escalation marker -> convergent round
        _run_node(self._state(tmp_path))
        kw = gen.call_args.kwargs
        assert kw["resume_session_id"] == "live-sid"   # keeps resuming
        assert "this is a fresh start" not in kw["feedback"]

    def test_resume_off_by_default_passes_no_session(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CORESMITH_CODEX_RESUME", raising=False)
        _capture_events(monkeypatch)
        gen = _mock_generate(monkeypatch)
        _spec(tmp_path, "ent", "prev spec body\n")
        (_blkdir(tmp_path, "ent") / "codex_session_id").write_text("live-sid")
        _run_node(self._state(tmp_path))
        # default-OFF: byte-identical to today -- no resume id threaded
        assert gen.call_args.kwargs["resume_session_id"] is None
