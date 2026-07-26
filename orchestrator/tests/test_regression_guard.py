# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Regression-guard fix (Bug 1) + budget sizing rule (Bug 3).

Bug 1: when a block already passed sim, re-entering it (e.g. an
integration-review restart) used to FORCE testbench regeneration, which
produced a worse TB that re-failed -> infinite regen/fail loop that wedged
whole runs. The fix REUSES the passing TB by default; the old behavior is
opt-in via CORESMITH_FORCE_TB_REGEN=1.

Bug 3: the uArch storage-budget prompt must forbid `sram_budget = 0` for a
block holding any storage structure >= 2 Kbit (the deadlock case the PPA
gate flags but the fixer can't repair).
"""
from __future__ import annotations

import json

import pytest

from orchestrator.langgraph.pipeline_graph import generate_rtl_node


def _passed_block_state(tmp_path):
    """A BlockState re-entering a block that previously PASSED sim."""
    block = {
        "name": "fifo_blk",
        "tier": 1,
        "rtl_target": "rtl/fifo_blk.v",
        "testbench": "tb/cocotb/test_fifo_blk.py",
        "description": "block that already passed sim",
    }
    # On-disk: the RTL exists + best_result.json marks sim_passed.
    rtl = tmp_path / block["rtl_target"]
    rtl.parent.mkdir(parents=True, exist_ok=True)
    rtl.write_text("module fifo_blk(); endmodule\n")
    bdir = tmp_path / ".coresmith" / "blocks" / "fifo_blk"
    bdir.mkdir(parents=True, exist_ok=True)
    (bdir / "best_result.json").write_text(json.dumps(
        {"sim_passed": True, "attempt": 1, "tests_passed": 7, "tests_total": 7}
    ))
    return {
        "project_root": str(tmp_path),
        "target_clock_mhz": 50.0,
        "max_attempts": 3,
        "current_block": block,
        "attempt": 2,                 # re-entry -> guard fires
        "phase": "init",
        "lint_clean": False,
        "force_regen_tb": False,
        "step_log_paths": {},
    }


class TestRegressionGuardReusesTB:
    @pytest.mark.asyncio
    async def test_default_reuses_tb_not_force_regen(self, tmp_path, monkeypatch):
        # Default: a re-entered passing block REUSES its TB (no regen loop).
        monkeypatch.delenv("CORESMITH_FORCE_TB_REGEN", raising=False)
        out = await generate_rtl_node(_passed_block_state(tmp_path))
        assert out["force_regen_tb"] is False
        assert out["lint_clean"] is True
        assert out["rtl_path"].endswith("rtl/fifo_blk.v")

    @pytest.mark.asyncio
    async def test_env_opt_in_restores_force_regen(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CORESMITH_FORCE_TB_REGEN", "1")
        out = await generate_rtl_node(_passed_block_state(tmp_path))
        assert out["force_regen_tb"] is True


class _RegenSentinel(Exception):
    """Raised by a stubbed generate_rtl to prove the node REGENERATED rather
    than taking the skip-reuse early return."""


class TestPpaRetryBypass:
    """A-Fix 4: a block that passed sim but FAILED the PPA gate (ppa_ok is
    False) must REGENERATE, not reuse the PPA-failing RTL (deadlock)."""

    @pytest.mark.asyncio
    async def test_ppa_false_regenerates_and_backs_up(self, tmp_path, monkeypatch):
        import orchestrator.langgraph.pipeline_graph as pg

        async def _boom_gen(*a, **k):
            raise _RegenSentinel()

        monkeypatch.setattr(pg, "generate_rtl", _boom_gen)
        st = _passed_block_state(tmp_path)
        st["ppa_ok"] = False  # live PPA verdict: failed
        with pytest.raises(_RegenSentinel):
            await pg.generate_rtl_node(st)
        # The bypass backs up the passing RTL BEFORE regenerating.
        backup = (tmp_path / ".coresmith" / "blocks" / "fifo_blk"
                  / "rtl_backup_attempt2.v")
        assert backup.exists()
        best = json.loads(
            (tmp_path / ".coresmith" / "blocks" / "fifo_blk"
             / "best_result.json").read_text()
        )
        assert best["sim_passed"] is True  # stays True (keyed on live ppa_ok)
        assert best.get("ppa_retry_attempt") == 2

    @pytest.mark.asyncio
    async def test_ppa_none_still_reuses_no_regen(self, tmp_path, monkeypatch):
        import orchestrator.langgraph.pipeline_graph as pg

        async def _boom_gen(*a, **k):
            raise _RegenSentinel()

        monkeypatch.setattr(pg, "generate_rtl", _boom_gen)
        monkeypatch.delenv("CORESMITH_FORCE_TB_REGEN", raising=False)
        st = _passed_block_state(tmp_path)  # no ppa_ok -> None -> not a PPA retry
        out = await pg.generate_rtl_node(st)  # must NOT raise (reuse path)
        assert out["lint_clean"] is True
        assert out["rtl_path"].endswith("rtl/fifo_blk.v")
        # no backup created on the reuse path
        assert not (tmp_path / ".coresmith" / "blocks" / "fifo_blk"
                    / "rtl_backup_attempt2.v").exists()

    def test_bypass_helper_invalidates_sim_cache(self, tmp_path):

        import orchestrator.langgraph.pipeline_graph as pg

        block_name = "fifo_blk"
        rtl = tmp_path / "rtl" / f"{block_name}.v"
        rtl.parent.mkdir(parents=True, exist_ok=True)
        rtl.write_text("module fifo_blk(); endmodule\n")
        bdir = tmp_path / ".coresmith" / "blocks" / block_name
        bdir.mkdir(parents=True, exist_ok=True)
        (bdir / "best_result.json").write_text(json.dumps({"sim_passed": True}))
        sim_dir = tmp_path / "sim_build" / block_name
        sim_dir.mkdir(parents=True, exist_ok=True)
        (sim_dir / "stale.o").write_text("x")

        state = {"project_root": str(tmp_path)}
        pg._ppa_retry_bypass(state, block_name, rtl, 3)

        assert (bdir / "rtl_backup_attempt3.v").exists()
        assert not sim_dir.exists()  # sim cache invalidated
        best = json.loads((bdir / "best_result.json").read_text())
        assert best["ppa_retry_attempt"] == 3
        assert best.get("ppa_bypass_backup", "").endswith("rtl_backup_attempt3.v")


class TestDeterministicGateRetryBypass:
    """Finding 1 (pipeline-campaign-3): a block that PASSED sim but was routed
    BACK to generate_rtl by a DETERMINISTIC gate (stage/storage/ifdef lint) must
    REGENERATE -- reusing the same sim-passing RTL re-fails the same gate forever
    (the skip_regen livelock, seen twice live in Phase B). Mirrors the PPA cases.
    """

    def _mark(self, tmp_path, marker):
        bdir = tmp_path / ".coresmith" / "blocks" / "fifo_blk"
        bdir.mkdir(parents=True, exist_ok=True)
        (bdir / "previous_error.txt").write_text(marker)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("marker", [
        "UNSYNTHESIZABLE -- pre-synth storage lint (yosys NOT run):\n"
        "WIDE FLAT PACKED STORAGE WITH DYNAMIC PART-SELECT in fifo_blk ...",
        "UNSYNTHESIZABLE COMBINATIONAL CLOUD -- deterministic stage-realization "
        "lint (pre-yosys) in fifo_blk.",
        "SPLIT-BRAIN CONDITIONAL-COMPILATION RTL in fifo_blk ...",
    ])
    async def test_deterministic_marker_regenerates_and_backs_up(
        self, tmp_path, monkeypatch, marker,
    ):
        import orchestrator.langgraph.pipeline_graph as pg

        async def _boom_gen(*a, **k):
            raise _RegenSentinel()

        monkeypatch.setattr(pg, "generate_rtl", _boom_gen)
        st = _passed_block_state(tmp_path)
        self._mark(tmp_path, marker)
        with pytest.raises(_RegenSentinel):
            await pg.generate_rtl_node(st)
        bdir = tmp_path / ".coresmith" / "blocks" / "fifo_blk"
        assert (bdir / "rtl_backup_attempt2.v").exists()
        best = json.loads((bdir / "best_result.json").read_text())
        assert best["sim_passed"] is True             # stays True
        assert best.get("lint_retry_attempt") == 2    # lint (not ppa) bypass

    @pytest.mark.asyncio
    async def test_non_deterministic_error_still_reuses(self, tmp_path, monkeypatch):
        # A generic sim/TB failure (no deterministic-gate marker) still REUSES
        # the passing RTL -- the bypass is scoped to the livelocking gate class.
        import orchestrator.langgraph.pipeline_graph as pg

        async def _boom_gen(*a, **k):
            raise _RegenSentinel()

        monkeypatch.setattr(pg, "generate_rtl", _boom_gen)
        monkeypatch.delenv("CORESMITH_FORCE_TB_REGEN", raising=False)
        st = _passed_block_state(tmp_path)
        self._mark(tmp_path, "cocotb assertion at 1200ns: expected 0xAB got 0xCD")
        out = await pg.generate_rtl_node(st)  # must NOT raise (reuse path)
        assert out["lint_clean"] is True
        assert not (tmp_path / ".coresmith" / "blocks" / "fifo_blk"
                    / "rtl_backup_attempt2.v").exists()

    def test_deterministic_gate_retry_helper(self, tmp_path):
        import orchestrator.langgraph.pipeline_graph as pg

        bdir = tmp_path / "blk"
        bdir.mkdir()
        assert pg._deterministic_gate_retry(bdir) is False  # no file
        for marker in pg._DETERMINISTIC_GATE_MARKERS:
            (bdir / "previous_error.txt").write_text("noise\n" + marker + "\nmore")
            assert pg._deterministic_gate_retry(bdir) is True
        (bdir / "previous_error.txt").write_text("a plain simulation mismatch")
        assert pg._deterministic_gate_retry(bdir) is False


class TestBudgetSizingRule:
    def test_uarch_prompt_forbids_sram_budget_zero_on_big_storage(self):
        from pathlib import Path

        import orchestrator.langchain.agents.uarch_spec_generator as u
        prompt = u.SYSTEM_PROMPT
        # The mandatory computed sizing rule must be present.
        assert "2048 bits" in prompt or "2 Kbit" in prompt
        assert "SPEC ERROR" in prompt
        assert "depth" in prompt and "width" in prompt
        # sanity: file on disk matches the loaded prompt
        md = (Path(u.__file__).resolve().parent.parent / "prompts"
              / "uarch_spec_generator.md").read_text()
        assert "SPEC ERROR" in md
