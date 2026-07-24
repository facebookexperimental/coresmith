# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the rung-2 mcu3 live-run fixes [rung2-fixes-1].

Defect 1 -- uArch integration gate must record SKIPPED-HONEST (not passed) for a
            goldenless / requirements-only run.
Defect 2 -- the PDK-free synthesizability probes must run + GATE + record real
            metrics under CORESMITH_SKIP_SYNTH; yosys-absent -> tooling_missing.
Defect 3 -- escalation-retry routes doc-sourced violations to Doc Fix (budget
            reset), and a single constraint pass surfaces ALL violations.
Defect 4 -- the SAD/FRD prompts forbid invented DFT/scan pins (transcription).

No live LLM / EDA required (real yosys is used for defect 2 where present; the
tooling-missing branch monkeypatches shutil.which). Compatible with
`-m "not live_llm and not requires_nix and not e2e"`.
"""
from __future__ import annotations

from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Defect 1 -- goldenless uArch integration gate is SKIPPED-HONEST, not passed
# ---------------------------------------------------------------------------

class TestGateStatusClassifier:
    """describe_gate_status distinguishes a real PASS from a no-op SKIP."""

    def test_goldenless_empty_project_not_applicable(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CORESMITH_BLOCK_GOLDENS", "1")
        monkeypatch.delenv("CORESMITH_SOURCE_ROOT", raising=False)
        from orchestrator.architecture import model_integration as mi

        status = mi.describe_gate_status(str(tmp_path))
        assert status["applicable"] is False
        assert "no golden reference" in status["reason"]
        assert status["block_models_present"] is False
        assert status["reference_resolvable"] is False

    def test_full_toy_project_is_applicable(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CORESMITH_BLOCK_GOLDENS", "1")
        monkeypatch.delenv("CORESMITH_SOURCE_ROOT", raising=False)
        from orchestrator.architecture import model_integration as mi

        models = tmp_path / "arch" / "block_models"
        models.mkdir(parents=True)
        (models / "mul2.py").write_text("# block model\n")
        (models / "_chip_model.py").write_text("def simulate(s):\n    return s\n")
        (tmp_path / "inputs").mkdir()
        (tmp_path / "inputs" / "toy_golden.py").write_text("def run(s):\n    return s\n")

        status = mi.describe_gate_status(str(tmp_path))
        assert status["applicable"] is True
        assert status["block_models_present"] is True
        assert status["chip_model_present"] is True
        assert status["reference_resolvable"] is True

    def test_flag_off_not_applicable(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CORESMITH_BLOCK_GOLDENS", raising=False)
        monkeypatch.setenv("CORESMITH_PROFILE", "legacy")
        from orchestrator.architecture import model_integration as mi

        status = mi.describe_gate_status(str(tmp_path))
        assert status["applicable"] is False
        assert status["block_goldens_enabled"] is False


class TestUarchGateSkipVsPass:
    @pytest.mark.asyncio
    async def test_goldenless_run_records_skipped_not_passed(
        self, tmp_path, monkeypatch
    ):
        """The mcu3 scenario: no golden, no block models -> the µarch gate must
        record SKIPPED-HONEST, NOT report passed=True (was a 4ms vacuous pass)."""
        monkeypatch.setenv("CORESMITH_BLOCK_GOLDENS", "1")
        monkeypatch.delenv("CORESMITH_SOURCE_ROOT", raising=False)
        from orchestrator.langgraph import pipeline_graph as pg

        async def _noop(_pr):
            return None

        monkeypatch.setattr(pg, "_maybe_generate_chip_model", _noop)

        state = {"project_root": str(tmp_path)}
        out = await pg.uarch_integration_gate_node(state)
        mir = out["model_integration_result"]
        assert mir.get("skipped") is True
        assert mir.get("passed") is not True          # NOT a pass
        assert "no golden reference" in mir.get("reason", "")

        # Routing treats skipped like pass (non-blocking) -> begin_rtl_pass.
        assert pg.route_after_uarch_gate({**state, **out}) == "begin_rtl_pass"

        # Scoreboard records a skipped chip_model DV row (not passed).
        from orchestrator.state_store.store import Scoreboard
        rows = Scoreboard(str(tmp_path)).latest_dv(
            block="chip_model", scope="chip_model")
        assert rows, "expected a skipped chip_model scoreboard row"
        assert rows[0]["skipped"] == 1
        assert rows[0]["passed"] == 0

    @pytest.mark.asyncio
    async def test_golden_run_still_passes(self, tmp_path, monkeypatch):
        """A design whose gate actually compared models must PASS unchanged."""
        monkeypatch.setenv("CORESMITH_BLOCK_GOLDENS", "1")
        from orchestrator.architecture import model_integration as mi
        from orchestrator.langgraph import pipeline_graph as pg

        async def _noop(_pr):
            return None

        monkeypatch.setattr(pg, "_maybe_generate_chip_model", _noop)
        monkeypatch.setattr(mi, "run_model_integration_gate", lambda pr: [])
        monkeypatch.setattr(
            mi, "describe_gate_status",
            lambda pr: {"applicable": True, "reason": "ok",
                        "block_goldens_enabled": True,
                        "block_models_present": True,
                        "chip_model_present": True, "reference_resolvable": True},
        )

        state = {"project_root": str(tmp_path)}
        out = await pg.uarch_integration_gate_node(state)
        mir = out["model_integration_result"]
        assert mir.get("passed") is True
        assert mir.get("skipped") is not True
        assert pg.route_after_uarch_gate({**state, **out}) == "begin_rtl_pass"

    @pytest.mark.asyncio
    async def test_model_integration_node_goldenless_skips(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("CORESMITH_BLOCK_GOLDENS", "1")
        monkeypatch.delenv("CORESMITH_SOURCE_ROOT", raising=False)
        from orchestrator.langgraph import pipeline_graph as pg

        async def _noop(_pr):
            return None

        monkeypatch.setattr(pg, "_maybe_generate_chip_model", _noop)
        state = {"project_root": str(tmp_path)}
        out = await pg.model_integration_node(state)
        mir = out["model_integration_result"]
        assert mir.get("skipped") is True
        assert mir.get("passed") is not True
        # Non-blocking: proceeds to integration_dv exactly like a pass.
        assert pg.route_after_model_integration({**state, **out}) == "integration_dv"


# ---------------------------------------------------------------------------
# Defect 2 -- PDK-free probes run + GATE + record real metrics under SKIP_SYNTH
# ---------------------------------------------------------------------------

_ADDER_RTL = """\
module blk (
    input  wire        clk,
    input  wire [7:0]  a,
    input  wire [7:0]  b,
    output reg  [8:0]  y
);
    always @(posedge clk) y <= a + b;
endmodule
"""


def _has_yosys() -> bool:
    import shutil
    return shutil.which("yosys") is not None


@pytest.mark.skipif(not _has_yosys(), reason="yosys required")
class TestPpaGatePdkFreeUnderSkipSynth:
    def _rtl(self, tmp_path) -> str:
        p = tmp_path / "blk.v"
        p.write_text(_ADDER_RTL)
        return str(p)

    def test_probes_run_and_record_metrics_when_gate_flag_off(
        self, tmp_path, monkeypatch
    ):
        # CORESMITH_PPA_GATE off, but require_gate_flag=False -> PDK-free probes
        # still run and surface REAL metrics (the SKIP_SYNTH path).
        monkeypatch.delenv("CORESMITH_PPA_GATE", raising=False)
        monkeypatch.setenv("CORESMITH_PROFILE", "legacy")
        from orchestrator.langgraph.pipeline_graph import _evaluate_ppa_gate

        ppa_ok, reasons, meta = _evaluate_ppa_gate(
            str(tmp_path), "blk", self._rtl(tmp_path), None,
            require_gate_flag=False,
        )
        # With no budget nothing is JUDGED (ppa_ok None), but the probes RAN:
        assert meta.get("cells") is not None and meta["cells"] > 0
        assert meta.get("elaborated") is True
        assert "tooling_missing" not in meta

    def test_gate_flag_off_default_still_short_circuits(self, tmp_path, monkeypatch):
        # Default require_gate_flag=True keeps the legacy behavior: PPA gate off
        # -> (None, [], {}) with NO probes (byte-identical to before).
        monkeypatch.delenv("CORESMITH_PPA_GATE", raising=False)
        monkeypatch.setenv("CORESMITH_PROFILE", "legacy")
        from orchestrator.langgraph.pipeline_graph import _evaluate_ppa_gate

        res = _evaluate_ppa_gate(str(tmp_path), "blk", self._rtl(tmp_path), None)
        assert res == (None, [], {})

    def test_over_ceiling_fails_under_skip_synth(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CORESMITH_MAX_CELLS", "1")   # any real block exceeds 1
        monkeypatch.setenv("CORESMITH_PROFILE", "legacy")
        from orchestrator.langgraph.pipeline_graph import _evaluate_ppa_gate

        ppa_ok, reasons, meta = _evaluate_ppa_gate(
            str(tmp_path), "blk", self._rtl(tmp_path), None,
            require_gate_flag=False,
        )
        assert ppa_ok is False
        assert any("cell count" in r for r in reasons)

    @pytest.mark.asyncio
    async def test_synthesize_node_skip_synth_records_and_gates(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("CORESMITH_SKIP_SYNTH", "1")
        monkeypatch.setenv("CORESMITH_MAX_CELLS", "1")   # force over-ceiling
        monkeypatch.setenv("CORESMITH_PROFILE", "legacy")
        from orchestrator.langgraph import pipeline_graph as pg

        rtl = self._rtl(tmp_path)
        state = {
            "current_block": {"name": "blk"},
            "attempt": 1,
            "project_root": str(tmp_path),
            "rtl_path": rtl,
            "pipeline_phase": "rtl",
            "step_log_paths": {},
        }
        out = await pg.synthesize_node(state)
        assert out["synth_success"] is True        # SKIP_SYNTH: synth itself skipped
        assert out["ppa_ok"] is False              # but the PDK-free probe GATED

        # Real metrics recorded in ppa_history (was all-NULL).
        from orchestrator.state_store.store import Scoreboard
        row = Scoreboard(str(tmp_path)).latest_ppa("blk")
        assert row is not None
        assert row["probe"] == "skip_synth"
        assert row["cells"] is not None and row["cells"] > 0

        # And route_after_synth actually GATES on it under SKIP_SYNTH.
        assert pg.route_after_synth({**state, **out}) == "diagnose"

    def test_no_yosys_is_tooling_missing_park_path(self, tmp_path, monkeypatch):
        # yosys absent -> the PDK-free probes cannot run -> tooling_missing; under
        # strict that parks (fixes-4 path), never a silent pass.
        monkeypatch.setenv("CORESMITH_PROFILE", "strict")
        monkeypatch.delenv("CORESMITH_GATE_FAIL_OPEN", raising=False)
        from orchestrator.langgraph import ppa_check as pc
        from orchestrator.langgraph.pipeline_graph import (
            _evaluate_ppa_gate,
            _ppa_should_park_tooling_missing,
        )

        monkeypatch.setattr(pc.shutil, "which", lambda _b: None)
        ppa_ok, reasons, meta = _evaluate_ppa_gate(
            str(tmp_path), "blk", self._rtl(tmp_path), None,
            require_gate_flag=False,
        )
        assert ppa_ok is None
        assert meta.get("tooling_missing") is True
        assert _ppa_should_park_tooling_missing(str(tmp_path), ppa_ok, meta) is True


# ---------------------------------------------------------------------------
# Defect 3(a) -- escalation retry routes doc-sourced violations to Doc Fix
# ---------------------------------------------------------------------------

def _doc_violation():
    return {
        "violation": "SAD pinout summary says 18 signal bits; port table has 26.",
        "category": "auto_fixable",
        "check": "derived_arithmetic_consistency",
        "severity": "error",
        "source_doc": "sad",
        "suggested_fix": "Recompute the summary from the emitted port table.",
    }


class TestEscalationRetryDocFix:
    def test_constraint_escalation_retry_with_doc_violation_goes_to_doc_fix(self):
        from orchestrator.langgraph.architecture_graph import (
            route_after_constraint_escalation,
        )
        state = {
            "human_response": {"action": "retry"},
            "constraint_result": {"violations": [_doc_violation()]},
            "doc_fix_attempts": 0,
        }
        assert route_after_constraint_escalation(state) == "Doc Fix"

    def test_exhausted_escalation_feedback_with_doc_violation_goes_to_doc_fix(self):
        from orchestrator.langgraph.architecture_graph import (
            route_after_exhausted_escalation,
        )
        state = {
            "human_response": {"action": "feedback", "feedback": "fix the summary"},
            "constraint_result": {"violations": [_doc_violation()]},
            "doc_fix_attempts": 0,
        }
        assert route_after_exhausted_escalation(state) == "Doc Fix"

    def test_escalation_retry_without_doc_violation_still_block_diagram(self):
        from orchestrator.langgraph.architecture_graph import (
            route_after_constraint_escalation,
        )
        # A structural (non-doc) violation retry still goes to Block Diagram.
        state = {
            "human_response": {"action": "retry"},
            "constraint_result": {"violations": [
                {"category": "structural", "violation": "peripheral mismatch"},
            ]},
        }
        assert route_after_constraint_escalation(state) == "Block Diagram"

    def test_escalation_retry_over_budget_falls_through_to_block_diagram(self):
        from orchestrator.langgraph.architecture_graph import (
            _DOC_FIX_MAX_ATTEMPTS,
            route_after_constraint_escalation,
        )
        state = {
            "human_response": {"action": "retry"},
            "constraint_result": {"violations": [_doc_violation()]},
            "doc_fix_attempts": _DOC_FIX_MAX_ATTEMPTS,
        }
        assert route_after_constraint_escalation(state) == "Block Diagram"

    @pytest.mark.asyncio
    async def test_constraint_escalation_node_resets_doc_fix_budget(
        self, tmp_path, monkeypatch
    ):
        import orchestrator.langgraph.architecture_graph as ag

        monkeypatch.setattr(ag, "interrupt", lambda payload: {"action": "retry"})
        state = {
            "round": 3, "max_rounds": 3,
            "constraint_result": {"violations": [_doc_violation()]},
            "memory_map": {}, "violations_history": [],
            "project_root": str(tmp_path),
            "doc_fix_attempts": 2,
        }
        update = await ag.escalate_constraints_node(state)
        assert update["doc_fix_attempts"] == 0   # fresh budget on operator retry

    @pytest.mark.asyncio
    async def test_exhausted_escalation_node_resets_doc_fix_budget(
        self, tmp_path, monkeypatch
    ):
        import orchestrator.langgraph.architecture_graph as ag

        monkeypatch.setattr(
            ag, "interrupt",
            lambda payload: {"action": "feedback",
                             "feedback": "regenerate the SAD summary"},
        )
        state = {
            "round": 4, "max_rounds": 3,
            "constraint_result": {"violations": [_doc_violation()]},
            "violations_history": [],
            "project_root": str(tmp_path),
            "doc_fix_attempts": 2,
        }
        update = await ag.escalate_exhausted_node(state)
        assert update["doc_fix_attempts"] == 0
        assert update.get("human_feedback") == "regenerate the SAD summary"


# ---------------------------------------------------------------------------
# Defect 3(b) -- a single constraint pass surfaces ALL violations
# ---------------------------------------------------------------------------

class TestConstraintMultiViolation:
    def test_extract_entries_from_violations_array(self):
        from orchestrator.architecture.constraints import _extract_violation_entries
        parsed = {
            "pass": False,
            "violations": [
                {"violation_text": "cols wrong", "source_doc": "sad"},
                {"violation_text": "rows wrong", "source_doc": "frd"},
            ],
        }
        entries = _extract_violation_entries(parsed)
        assert len(entries) == 2

    def test_extract_entries_legacy_flat_shape(self):
        from orchestrator.architecture.constraints import _extract_violation_entries
        parsed = {
            "pass": False,
            "violation_text": "single breach", "source_doc": "sad",
            "evidence": "field", "suggested_fix": "fix it",
        }
        entries = _extract_violation_entries(parsed)
        assert len(entries) == 1
        assert entries[0]["violation_text"] == "single breach"

    @pytest.mark.asyncio
    async def test_subagent_surfaces_all_violations(self, monkeypatch):
        import json
        from unittest.mock import AsyncMock

        import orchestrator.langchain.agents.coresmith_llm as llm_mod
        from orchestrator.architecture import constraints

        response = json.dumps({
            "pass": False,
            "violations": [
                {"violation_text": "columns 45 should be 26",
                 "evidence": "SAD says 45", "suggested_fix": "use 26",
                 "source_doc": "sad"},
                {"violation_text": "cycles_per_txn 10 should be 12",
                 "evidence": "FRD says 10", "suggested_fix": "use 12",
                 "source_doc": "frd"},
                {"violation_text": "coord_max 511 should be 255",
                 "evidence": "SAD says 511", "suggested_fix": "use 255",
                 "source_doc": "sad"},
            ],
        })

        class _FakeLLM:
            def __init__(self, *a, **k):
                pass
            call = AsyncMock(return_value=response)

        monkeypatch.setattr(llm_mod, "ClaudeLLM", _FakeLLM)
        constraint = {
            "id": "derived_arithmetic_consistency",
            "category": "auto_fixable", "severity": "error",
            "description": "arithmetic consistency",
        }
        violations = await constraints._run_constraint_subagent(
            constraint, "artifact bundle", 30)
        assert len(violations) == 3
        assert all(v["check"] == "derived_arithmetic_consistency" for v in violations)
        assert {v["source_doc"] for v in violations} == {"sad", "frd"}

    @pytest.mark.asyncio
    async def test_subagent_legacy_single_violation_still_works(self, monkeypatch):
        import json
        from unittest.mock import AsyncMock

        import orchestrator.langchain.agents.coresmith_llm as llm_mod
        from orchestrator.architecture import constraints

        response = json.dumps({
            "pass": False,
            "violation_text": "gate count exceeds budget",
            "evidence": "3M > 2M", "suggested_fix": "split",
            "source_doc": "block_diagram",
        })

        class _FakeLLM:
            def __init__(self, *a, **k):
                pass
            call = AsyncMock(return_value=response)

        monkeypatch.setattr(llm_mod, "ClaudeLLM", _FakeLLM)
        constraint = {
            "id": "gate_budget", "category": "auto_fixable",
            "severity": "error", "description": "gate budget",
        }
        violations = await constraints._run_constraint_subagent(
            constraint, "bundle", 30)
        assert len(violations) == 1
        assert "gate count" in violations[0]["violation"]


# ---------------------------------------------------------------------------
# Defect 4 -- SAD/FRD prompts forbid invented DFT/scan pins
# ---------------------------------------------------------------------------

class TestSadFrdPinDiscipline:
    @staticmethod
    def _norm(text: str) -> str:
        # Collapse line-wrapping so multi-line phrases match regardless of where
        # the prompt happens to wrap.
        return " ".join(text.split())

    def test_sad_prompt_forbids_invented_dft_pins(self):
        from orchestrator.architecture.specialists import sad_spec
        prompt = self._norm(sad_spec.SYSTEM_PROMPT)
        assert "PIN DISCIPLINE" in prompt
        assert "TRANSCRIPTION of the PRD interface contract" in prompt
        assert "scan_en" in prompt
        assert "pin-count arithmetic" in prompt
        assert "RECOMPUTED" in prompt

    def test_sad_prompt_no_longer_mandates_a_scan_pin(self):
        # The old unconditional "Include at least one test/debug pin (JTAG or
        # scan_en)" line is gone -- DFT pins are now PRD-gated.
        from orchestrator.architecture.specialists import sad_spec
        prompt = self._norm(sad_spec.SYSTEM_PROMPT)
        assert "Include at least one test/debug pin" not in prompt
        assert "ONLY when the PRD interface contract explicitly lists them" in prompt

    def test_frd_prompt_pins_transcription_rule(self):
        from orchestrator.architecture.specialists import frd_spec
        prompt = self._norm(frd_spec.SYSTEM_PROMPT)
        assert "PIN/PORT DISCIPLINE" in prompt
        assert "TRANSCRIBE the PRD interface contract" in prompt
        assert "scan_en" in prompt
