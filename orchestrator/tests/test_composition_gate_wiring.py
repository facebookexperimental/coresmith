# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the model-integration node graph wiring + env-flag gating (v2).

No LLM, no EDA. Verifies:
- the model_integration node is in the orchestrator graph and integration_check
  routes through it;
- route_after_model_integration flag-off + parked-action routing;
- model_integration_node is a pure no-op (returns {}) when the flag is off;
- generate_uarch_spec writes NO block models when the flag is unset and DOES
  attempt block-model generation (into arch/block_models/) when set (mocked
  agent).
"""

from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import MemorySaver

from orchestrator.langgraph import pipeline_graph
from orchestrator.langgraph.pipeline_graph import (
    build_pipeline_graph,
    model_integration_node,
    route_after_model_integration,
    route_after_integration,
    route_after_uarch_gate,
)


class TestModelIntegrationWiring:
    def test_node_present(self):
        graph = build_pipeline_graph(checkpointer=MemorySaver())
        assert "model_integration" in graph.get_graph().nodes.keys()

    def test_integration_routes_to_model_integration(self):
        # Clean integration result -> model_integration (was integration_dv).
        assert route_after_integration({"integration_result": {"error_count": 0}}) \
            == "model_integration"

    def test_integration_accepted_by_user_routes_to_model_integration(self):
        assert route_after_integration(
            {"integration_result": {"accepted_by_user": True, "error_count": 3}}
        ) == "model_integration"

    def test_integration_still_ends_on_error(self):
        from langgraph.graph import END
        # An uncorrected integration error must still terminate (NOT run the
        # model-integration gate) -- behavior preserved from before the feature.
        assert route_after_integration(
            {"integration_result": {"error_count": 2}}
        ) == END


class TestRouteAfterModelIntegration:
    def test_pass_goes_to_integration_dv(self):
        assert route_after_model_integration(
            {"model_integration_result": {"passed": True}}
        ) == "integration_dv"

    def test_no_result_goes_to_integration_dv(self):
        # Flag-off path sets no result -> default through to integration_dv.
        assert route_after_model_integration({}) == "integration_dv"

    def test_retry_reruns_gate(self):
        assert route_after_model_integration(
            {"model_integration_result": {"passed": False, "action_taken": "retry"}}
        ) == "model_integration"

    def test_revise_uarch_reruns_gate(self):
        assert route_after_model_integration(
            {"model_integration_result":
                {"passed": False, "action_taken": "revise_uarch"}}
        ) == "model_integration"

    def test_abort_ends(self):
        from langgraph.graph import END
        assert route_after_model_integration(
            {"model_integration_result": {"aborted": True, "action_taken": "abort"}}
        ) == END


class TestModelIntegrationNodeFlagOff:
    @pytest.mark.asyncio
    async def test_node_is_noop_when_flag_off(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CORESMITH_BLOCK_GOLDENS", raising=False)
        result = await model_integration_node({"project_root": str(tmp_path)})
        # Pure no-op: empty dict, no state mutation, identical to pre-feature.
        assert result == {}

    @pytest.mark.asyncio
    async def test_node_skips_honestly_when_flag_on_and_no_models(
        self, tmp_path, monkeypatch
    ):
        # rung2 defect 1: flag on but no block models / no reference => the gate
        # is NOT applicable. It must record SKIPPED-HONEST, NOT vacuously report
        # passed=True (the mcu3 4ms vacuous-pass hole). Still non-blocking.
        monkeypatch.setenv("CORESMITH_BLOCK_GOLDENS", "1")
        monkeypatch.delenv("CORESMITH_SOURCE_ROOT", raising=False)
        result = await model_integration_node({"project_root": str(tmp_path)})
        mir = result.get("model_integration_result", {})
        assert mir.get("skipped") is True
        assert mir.get("passed") is not True
        assert "no golden reference" in mir.get("reason", "")


# A minimal valid MyHDL block model the mocked generator writes to disk.
_FAKE_BLOCK_MODEL = '''\
from myhdl import block, Signal, intbv, always_seq

W = 8

@block
def blk(clk, rst, din, dvld_in, dout, dvld_out):
    @always_seq(clk.posedge, reset=rst)
    def logic():
        dvld_out.next = dvld_in
        if dvld_in:
            dout.next = din
    return logic
'''


class TestUarchSpecBlockModelGating:
    """generate_uarch_spec must write NO block models when the flag is off,
    and attempt block-model generation (into arch/block_models/) when on."""

    @pytest.mark.asyncio
    async def test_no_block_model_when_flag_off(self, tmp_path, monkeypatch):
        import orchestrator.langgraph.pipeline_helpers as ph

        monkeypatch.delenv("CORESMITH_BLOCK_GOLDENS", raising=False)
        monkeypatch.setattr(ph, "PROJECT_ROOT", tmp_path)

        class _FakeAgent:
            def __init__(self, *a, **k):
                pass

            async def generate(self, **kwargs):
                return {
                    "spec_text": "# spec\n## Interfaces\nmodule x",
                    "block_name": kwargs["block_name"],
                    "spec_summary": {},
                }

        import orchestrator.langchain.agents.uarch_spec_generator as usg
        monkeypatch.setattr(usg, "UarchSpecGenerator", _FakeAgent)

        block = {"name": "blk", "description": "d", "python_source": ""}
        await ph.generate_uarch_spec(block)

        bm_dir = tmp_path / "arch" / "block_models"
        assert not bm_dir.exists() or not list(bm_dir.glob("*.py")), \
            "no block models should be written when the flag is off"

    @pytest.mark.asyncio
    async def test_block_model_attempted_when_flag_on(self, tmp_path, monkeypatch):
        import orchestrator.langgraph.pipeline_helpers as ph

        monkeypatch.setenv("CORESMITH_BLOCK_GOLDENS", "1")
        monkeypatch.setattr(ph, "PROJECT_ROOT", tmp_path)

        # Reference implementation so resolve_reference_implementation succeeds.
        (tmp_path / "inputs").mkdir(parents=True, exist_ok=True)
        (tmp_path / "inputs" / "d_golden.py").write_text(
            "def run(x): return x\n", encoding="utf-8"
        )

        class _FakeUarch:
            def __init__(self, *a, **k):
                pass

            async def generate(self, **kwargs):
                return {
                    "spec_text": "# spec\n## Interfaces\nmodule x",
                    "block_name": kwargs["block_name"],
                    "spec_summary": {},
                }

        import orchestrator.langchain.agents.uarch_spec_generator as usg
        monkeypatch.setattr(usg, "UarchSpecGenerator", _FakeUarch)

        called = {}

        class _FakeModel:
            def __init__(self, *a, **k):
                pass

            async def generate(self, **kwargs):
                called["block"] = kwargs["block_name"]
                out = kwargs["output_path"]
                from pathlib import Path as _P
                _P(out).parent.mkdir(parents=True, exist_ok=True)
                _P(out).write_text(_FAKE_BLOCK_MODEL, encoding="utf-8")
                return {"path": out}

        import orchestrator.langchain.agents.block_golden_generator as bgg
        monkeypatch.setattr(bgg, "BlockGoldenGenerator", _FakeModel)

        block = {"name": "blk", "description": "d", "python_source": ""}
        await ph.generate_uarch_spec(block)

        assert called.get("block") == "blk", "block model generator must be invoked"
        assert (tmp_path / "arch" / "block_models" / "blk.py").exists()


# ---------------------------------------------------------------------------
# CORESMITH_DETERMINISTIC_BFM advisory bypass of the model-level composition
# gate (feat/composition-advisory).
#
# Rationale: the composition _chip_model.py pin driver is LLM-authored and
# stimulus/DUT-fragile (it can mis-decode the IN-window stimulus and produce
# corrupt/all-zero composed output DESPITE byte-correct per-block models). When
# CORESMITH_DETERMINISTIC_BFM=1 the deterministic integration DV on the real
# chip_top RTL is the AUTHORITATIVE contract check, so a model-level mismatch
# must be ADVISORY (log + proceed), NOT a hard block / full re-spec.
#
# Two-branch env gate (repo convention): flag unset = old BLOCKING behavior
# (interrupt/park -> re-spec); flag set = advisory-proceed (no interrupt).
# ---------------------------------------------------------------------------

# One diverging violation -> the gate would normally PARK an interrupt.
_FAKE_VIOLATIONS = [{
    "first_divergence_block": "raster",
    "expected": "0102",
    "observed": "0000",
    "gap_class": "block_math",
}]


def _install_failing_gate(monkeypatch):
    """Force the composition gate to return violations (would normally block).

    Stubs the LLM chip-model generation and the deterministic MyHDL verify, and
    records whether ``interrupt()`` (the blocking park) was reached.
    """
    import orchestrator.architecture.model_integration as _mi

    async def _noop_gen(pr):  # _maybe_generate_chip_model stub
        return None

    def _fake_gate(pr, *a, **k):
        return list(_FAKE_VIOLATIONS)

    calls = {"interrupt": 0}

    def _fake_interrupt(payload):
        calls["interrupt"] += 1
        return {"action": "abort"}

    monkeypatch.setattr(pipeline_graph, "_maybe_generate_chip_model", _noop_gen)
    monkeypatch.setattr(_mi, "run_model_integration_gate", _fake_gate)
    monkeypatch.setattr(pipeline_graph, "interrupt", _fake_interrupt)
    # Gate must be "on" (two-pass composition path) so the node is not a no-op.
    monkeypatch.setenv("CORESMITH_BLOCK_GOLDENS", "1")
    return calls


class TestCompositionAdvisoryRouting:
    """Pure routing: an advisory-bypass result is non-blocking (proceeds)."""

    def test_single_pass_advisory_bypass_goes_to_integration_dv(self):
        assert route_after_model_integration(
            {"model_integration_result":
                {"passed": False, "advisory_bypass": True}}
        ) == "integration_dv"

    def test_two_pass_advisory_bypass_goes_to_begin_rtl_pass(self):
        assert route_after_uarch_gate(
            {"model_integration_result":
                {"passed": False, "advisory_bypass": True}}
        ) == "begin_rtl_pass"

    def test_flag_off_blocking_result_still_re_specs(self):
        # Without advisory_bypass, a parked block_math failure still re-specs
        # (single-pass reruns the gate; two-pass routes to init_tier). Proves the
        # advisory routing is additive and the blocking path is unchanged.
        assert route_after_model_integration(
            {"model_integration_result":
                {"passed": False, "action_taken": "revise_uarch"}}
        ) == "model_integration"
        assert route_after_uarch_gate(
            {"model_integration_result": {
                "passed": False, "gap_class": "block_math",
                "action_taken": "revise_uarch"}}
        ) == "init_tier"


class TestModelIntegrationAdvisoryBypass:
    """Single-pass model_integration_node: flag OFF blocks, flag ON is advisory."""

    @pytest.mark.asyncio
    async def test_flag_off_still_blocks_and_parks(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CORESMITH_DETERMINISTIC_BFM", raising=False)
        calls = _install_failing_gate(monkeypatch)
        result = await model_integration_node({"project_root": str(tmp_path)})
        mir = result.get("model_integration_result", {})
        # Old behavior: the gate PARKED (interrupt reached) and did not proceed.
        assert calls["interrupt"] == 1
        assert mir.get("advisory_bypass") is not True
        assert mir.get("passed") is not True

    @pytest.mark.asyncio
    async def test_flag_on_is_advisory_and_proceeds(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CORESMITH_DETERMINISTIC_BFM", "1")
        calls = _install_failing_gate(monkeypatch)
        result = await model_integration_node({"project_root": str(tmp_path)})
        mir = result.get("model_integration_result", {})
        # New behavior: NO interrupt (non-blocking), advisory bypass recorded,
        # and routing proceeds to integration_dv.
        assert calls["interrupt"] == 0
        assert mir.get("advisory_bypass") is True
        assert mir.get("passed") is not True
        assert mir.get("first_divergence_block") == "raster"
        assert route_after_model_integration(result) == "integration_dv"


class TestUarchGateAdvisoryBypass:
    """Two-pass uarch_integration_gate_node: flag OFF blocks, flag ON is advisory."""

    @pytest.mark.asyncio
    async def test_flag_off_still_blocks_and_parks(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CORESMITH_DETERMINISTIC_BFM", raising=False)
        calls = _install_failing_gate(monkeypatch)
        result = await pipeline_graph.uarch_integration_gate_node(
            {"project_root": str(tmp_path)}
        )
        mir = result.get("model_integration_result", {})
        assert calls["interrupt"] == 1
        assert mir.get("advisory_bypass") is not True
        assert mir.get("passed") is not True

    @pytest.mark.asyncio
    async def test_flag_on_is_advisory_and_proceeds(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CORESMITH_DETERMINISTIC_BFM", "1")
        calls = _install_failing_gate(monkeypatch)
        result = await pipeline_graph.uarch_integration_gate_node(
            {"project_root": str(tmp_path)}
        )
        mir = result.get("model_integration_result", {})
        # New behavior: no interrupt, advisory bypass recorded, routing proceeds
        # to the RTL pass (does NOT re-spec-loop back to init_tier).
        assert calls["interrupt"] == 0
        assert mir.get("advisory_bypass") is True
        assert mir.get("passed") is not True
        assert mir.get("first_divergence_block") == "raster"
        assert route_after_uarch_gate(result) == "begin_rtl_pass"
