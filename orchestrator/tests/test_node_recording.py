# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Item-10 (B3) node integration: scoreboard recording + oracle-tamper gate."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from orchestrator.state_store.store import Scoreboard
from orchestrator.state_store.trust import write_oracle_manifest


class TestRecordHelpers:
    def test_record_dv_row_writes(self, tmp_path):
        (tmp_path / ".coresmith").mkdir()
        from orchestrator.langgraph import pipeline_graph as pg
        pg._record_dv_row(str(tmp_path), block="adder", scope="rtl",
                          source="gate", passed=True, tests_passed=4,
                          tests_total=4)
        row = Scoreboard(tmp_path).latest_dv(block="adder", scope="rtl")[0]
        assert row["source"] == "gate" and row["passed"] == 1

    def test_record_ppa_row_writes(self, tmp_path):
        (tmp_path / ".coresmith").mkdir()
        from orchestrator.langgraph import pipeline_graph as pg
        pg._record_ppa_row(str(tmp_path), block="adder", source="gate",
                          probe="synth", ff=42, ppa_ok=True)
        row = Scoreboard(tmp_path).latest_ppa("adder")
        assert row["ff"] == 42 and row["ppa_ok"] == 1

    def test_record_ppa_row_persists_wns(self, tmp_path):
        # pdk-fixes-1: the pre-layout STA WNS now reaches the ppa_history
        # wns_ns column (it was always NULL before timing was obtainable).
        (tmp_path / ".coresmith").mkdir()
        from orchestrator.langgraph import pipeline_graph as pg
        pg._record_ppa_row(str(tmp_path), block="mem_blk", source="gate",
                          probe="synth", ff=100, wns_ns=-3.56, ppa_ok=False)
        row = Scoreboard(tmp_path).latest_ppa("mem_blk")
        assert row["wns_ns"] == pytest.approx(-3.56)

    def test_record_helpers_never_raise_on_bad_root(self, tmp_path):
        # .coresmith is a FILE -> scoreboard write fails silently.
        (tmp_path / ".coresmith").write_text("nope")
        from orchestrator.langgraph import pipeline_graph as pg
        pg._record_dv_row(str(tmp_path), block="x", scope="rtl", passed=True)
        pg._record_ppa_row(str(tmp_path), block="x", probe="synth")


@pytest.mark.asyncio
class TestGenerateTestbenchRecording:
    def _state(self, tmp_path, block_name="test_block"):
        block_dir = tmp_path / ".coresmith" / "blocks" / block_name
        block_dir.mkdir(parents=True)
        (tmp_path / "test.v").write_text(f"module {block_name}(); endmodule\n")
        (tmp_path / "test_tb.py").write_text("import cocotb\n")
        return {
            "current_block": {"name": block_name, "testbench": "test_tb.py"},
            "rtl_path": str(tmp_path / "test.v"),
            "tb_path": str(tmp_path / "test_tb.py"),
            "attempt": 1,
            "project_root": str(tmp_path),
            "pipeline_run_start": 0,
            "step_log_paths": {},
            "preserve_testbench": True,
            "force_regen_tb": False,
        }

    async def test_records_gate_dv_row_on_pass(self, tmp_path):
        from orchestrator.langgraph.pipeline_graph import generate_testbench_node
        state = self._state(tmp_path)
        mock = {"passed": True, "log": "PASS", "tests_passed": 6,
                "tests_total": 6, "tests_failed": 0, "log_path": "/tmp/sim.log"}
        with patch("orchestrator.langgraph.pipeline_graph.run_simulation",
                   return_value=mock):
            out = await generate_testbench_node(state)
        assert out["sim_passed"] is True
        row = Scoreboard(tmp_path).latest_dv(block="test_block", scope="rtl")[0]
        assert row["source"] == "gate"
        assert row["passed"] == 1
        assert row["tests_total"] == 6

    async def test_oracle_tamper_fails_closed(self, tmp_path, monkeypatch):
        # Seed + snapshot an oracle, then tamper it -> node must flip to failed.
        (tmp_path / "inputs").mkdir()
        (tmp_path / "inputs" / "golden.py").write_text("def g(x):\n    return x\n")
        write_oracle_manifest(tmp_path)
        (tmp_path / "inputs" / "golden.py").write_text("def g(x):\n    return 0\n")

        from orchestrator.langgraph.pipeline_graph import generate_testbench_node
        state = self._state(tmp_path)
        mock = {"passed": True, "log": "PASS", "tests_passed": 6,
                "tests_total": 6, "tests_failed": 0, "log_path": "/tmp/sim.log"}
        with patch("orchestrator.langgraph.pipeline_graph.run_simulation",
                   return_value=mock):
            out = await generate_testbench_node(state)
        assert out["sim_passed"] is False
        row = Scoreboard(tmp_path).latest_dv(block="test_block", scope="rtl")[0]
        assert row["passed"] == 0

    async def test_no_manifest_does_not_block(self, tmp_path):
        # No oracle manifest -> non-blocking, a clean sim still passes.
        from orchestrator.langgraph.pipeline_graph import generate_testbench_node
        state = self._state(tmp_path)
        mock = {"passed": True, "log": "PASS", "tests_passed": 1,
                "tests_total": 1, "tests_failed": 0, "log_path": "/tmp/sim.log"}
        with patch("orchestrator.langgraph.pipeline_graph.run_simulation",
                   return_value=mock):
            out = await generate_testbench_node(state)
        assert out["sim_passed"] is True
