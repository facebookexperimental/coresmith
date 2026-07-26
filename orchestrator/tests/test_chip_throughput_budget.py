# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Audit F8: a requirements-declared chip cycle budget must be RESOLVED and
reported even when no chip-window measurement artifact exists -- the reference codec
encoder declared "budget **500000 cyc/frame**" yet chip_throughput.json
carried budget_source="none"."""
from __future__ import annotations

from orchestrator.langgraph.throughput_gate import (
    chip_throughput_budget,
    evaluate_chip_throughput,
)


def _mk_reqs(root, text):
    d = root / "inputs"
    d.mkdir(parents=True, exist_ok=True)
    (d / "requirements.md").write_text(text)


class TestRequirementsBudgetResolution:
    def test_bold_cyc_per_frame_budget_resolves(self, tmp_path):
        _mk_reqs(tmp_path, "- Encode cycles/frame measured externally, "
                           "reported not gating; budget **500000 cyc/frame** "
                           "(generous; >30 fps at 50 MHz).\n")
        budget, source, note = chip_throughput_budget(str(tmp_path))
        assert budget == 500000.0
        assert source.startswith("requirements")
        assert "frame" in source

    def test_comma_separated_cycles_per_op(self, tmp_path):
        _mk_reqs(tmp_path, "throughput budget 1,024 cycles/op steady state\n")
        budget, source, _ = chip_throughput_budget(str(tmp_path))
        assert budget == 1024.0
        assert "op" in source

    def test_no_declared_budget_stays_none(self, tmp_path):
        _mk_reqs(tmp_path, "no perf numbers here\n")
        budget, source, _ = chip_throughput_budget(str(tmp_path))
        assert budget is None and source == "none"

    def test_state_budget_still_wins(self, tmp_path):
        _mk_reqs(tmp_path, "budget **500000 cyc/frame**\n")
        budget, source, _ = chip_throughput_budget(
            str(tmp_path), state={"chip_cyc_per_op_budget": 777})
        assert budget == 777.0 and source == "state"


class TestUnmeasuredBudgetSurfaced:
    def test_missing_artifact_still_records_declared_budget(self, tmp_path):
        _mk_reqs(tmp_path, "budget **500000 cyc/frame**\n")
        sim = tmp_path / "sim_build"
        sim.mkdir()
        rec = evaluate_chip_throughput(str(tmp_path), str(sim))
        assert rec["applicable"] is False and rec["passed"] is None
        assert rec["budget_cyc_per_op"] == 500000.0
        assert rec["budget_source"].startswith("requirements")
        assert "UNMEASURED" in rec["reason"]

    def test_missing_artifact_no_budget_unchanged(self, tmp_path):
        sim = tmp_path / "sim_build"
        sim.mkdir()
        rec = evaluate_chip_throughput(str(tmp_path), str(sim))
        assert rec["budget_cyc_per_op"] is None
        assert rec["budget_source"] == "none"
