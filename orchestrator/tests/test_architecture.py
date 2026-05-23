# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for the architecture stage tooling.

Covers: PDKConfig, ArchitectureState, constraint checker, stub specialists,
benchmark cache, benchmark runner (template rendering), and Temporal handoff.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)


@pytest.fixture
def tmp_project(tmp_path):
    """Create a temporary project directory with .coresmith/ state dir."""
    coresmith_dir = tmp_path / ".coresmith"
    coresmith_dir.mkdir()
    return str(tmp_path)


@pytest.fixture
def sample_block_diagram():
    """A realistic block diagram for testing."""
    return {
        "blocks": [
            {
                "name": "scrambler",
                "description": "PRBS energy dispersal",
                "tier": 1,
                "python_source": "PyDVB/dvb/Scrambler.py",
                "rtl_target": "rtl/dvbt/scrambler.v",
                "testbench": "tb/cocotb/test_scrambler.py",
                "interfaces": {"input": {"width": 1}, "output": {"width": 1}},
                "estimated_gates": 500,
            },
            {
                "name": "conv_encoder",
                "description": "K=7 rate-1/2 convolutional encoder",
                "tier": 1,
                "python_source": "PyDVB/dvb/Convolutional.py",
                "rtl_target": "rtl/dvbt/conv_encoder.v",
                "testbench": "tb/cocotb/test_conv_encoder.py",
                "interfaces": {"input": {"width": 1}, "output": {"width": 2}},
                "estimated_gates": 300,
            },
            {
                "name": "fft_engine",
                "description": "2048/8192-point FFT/IFFT",
                "tier": 3,
                "python_source": "PyDVB/dvb/OFDM.py",
                "rtl_target": "rtl/dvbt/fft_engine.v",
                "testbench": "tb/cocotb/test_fft_engine.py",
                "interfaces": {"input": {"width": 4}, "output": {"width": 4}},
                "estimated_gates": 500000,
            },
        ],
        "connections": [
            {"from": "scrambler", "to": "conv_encoder", "interface": "axis", "data_width": 1},
            {"from": "conv_encoder", "to": "fft_engine", "interface": "axis", "data_width": 2},
        ],
    }


@pytest.fixture
def sky130_yaml_path():
    """Path to the sky130 PDK config YAML."""
    return str(Path(_PROJECT_ROOT) / "orchestrator" / "pdk" / "configs" / "sky130.yaml")


# ---------------------------------------------------------------------------
# PDKConfig tests
# ---------------------------------------------------------------------------


class TestPDKConfig:
    def test_from_yaml(self, sky130_yaml_path):
        from orchestrator.pdk import PDKConfig

        pdk = PDKConfig.from_yaml(sky130_yaml_path, pdk_root="/tmp/fake_pdk")

        assert pdk.name == "sky130"
        assert pdk.process_nm == 130
        assert pdk.supply_voltage == 1.8
        assert pdk.std_cell_library == "sky130_fd_sc_hd"
        assert pdk.site_name == "unithd"
        assert pdk.default_corner == "tt_025C_1v80"
        assert "tt_025C_1v80" in pdk.corners

    def test_liberty_path_resolution(self, sky130_yaml_path):
        from orchestrator.pdk import PDKConfig

        pdk = PDKConfig.from_yaml(sky130_yaml_path, pdk_root="/opt/pdk")
        lib_path = pdk.liberty_path()

        assert lib_path.startswith("/opt/pdk/")
        assert "sky130_fd_sc_hd__tt_025C_1v80.lib" in lib_path

    def test_liberty_path_invalid_corner(self, sky130_yaml_path):
        from orchestrator.pdk import PDKConfig

        pdk = PDKConfig.from_yaml(sky130_yaml_path, pdk_root="/tmp")

        with pytest.raises(KeyError, match="nonexistent"):
            pdk.liberty_path("nonexistent")

    def test_to_summary(self, sky130_yaml_path):
        from orchestrator.pdk import PDKConfig

        pdk = PDKConfig.from_yaml(sky130_yaml_path, pdk_root="/tmp")
        summary = pdk.to_summary()

        assert "sky130" in summary
        assert "130nm" in summary
        assert "1.8V" in summary

    def test_serialization_roundtrip(self, sky130_yaml_path):
        from orchestrator.pdk import PDKConfig

        pdk = PDKConfig.from_yaml(sky130_yaml_path, pdk_root="/tmp/pdk")
        d = pdk.to_dict()
        pdk2 = PDKConfig.from_dict(d)

        assert pdk2.name == pdk.name
        assert pdk2.process_nm == pdk.process_nm
        assert pdk2.supply_voltage == pdk.supply_voltage
        assert pdk2.default_corner == pdk.default_corner
        assert len(pdk2.corners) == len(pdk.corners)


# ---------------------------------------------------------------------------
# ArchitectureState tests
# ---------------------------------------------------------------------------


class TestArchitectureState:
    def test_create_default(self):
        from orchestrator.architecture.state import ArchitectureState

        state = ArchitectureState()
        assert state.requirements == ""
        assert state.block_diagram == {}
        assert state.block_specs == []

    def test_json_roundtrip(self, tmp_project):
        from orchestrator.architecture.state import (
            ArchitectureState,
            load_state,
            save_state,
        )

        state = ArchitectureState(
            requirements="DVB-T transceiver",
            target_clock_mhz=50.0,
            block_diagram={"blocks": [{"name": "scrambler"}]},
        )
        save_state(state, tmp_project)

        loaded = load_state(tmp_project)
        assert loaded.requirements == "DVB-T transceiver"
        assert loaded.target_clock_mhz == 50.0
        assert loaded.block_diagram["blocks"][0]["name"] == "scrambler"

    def test_load_nonexistent_returns_default(self, tmp_project):
        from orchestrator.architecture.state import load_state

        state = load_state(tmp_project)
        assert state.requirements == ""

    def test_question_lifecycle(self):
        from orchestrator.architecture.state import (
            ArchitectureQuestion,
            ArchitectureState,
        )

        state = ArchitectureState()

        q = ArchitectureQuestion(
            agent="block_diagram",
            question="What data rate?",
            context="Needed for FFT sizing",
            priority="blocking",
        )
        assert q.id  # auto-generated
        assert q.timestamp  # auto-generated
        assert not q.is_answered()

        state.add_question(q)
        assert len(state.pending_questions) == 1
        assert state.has_blocking_questions()

        # Answer it
        state.answer_question(q.id, "100 Mbps")
        assert len(state.pending_questions) == 0
        assert len(state.answered_questions) == 1
        assert state.answered_questions[0]["answer"] == "100 Mbps"
        assert not state.has_blocking_questions()


# ---------------------------------------------------------------------------
# Constraint checker tests
# ---------------------------------------------------------------------------


class TestConstraintChecker:
    """Constraint checker is now a per-constraint subagent dispatcher.
    Tests mock ClaudeLLM so the subagent fan-out happens with deterministic
    per-call responses."""

    def _make_subagent_mock(self, per_check_response):
        """Build an AsyncMock for ClaudeLLM.call that returns the configured
        response per constraint id. Maps from constraint id (looked up via
        run_name) to a {pass, violation_text, ...} dict."""
        from unittest.mock import AsyncMock
        import json as _json

        async def _fake_call(*args, **kwargs):
            run_name = kwargs.get("run_name", "")
            check_id = run_name.split(":", 1)[1] if ":" in run_name else ""
            resp = per_check_response.get(check_id, {"pass": True, "evidence": "not applicable"})
            return _json.dumps(resp)

        return AsyncMock(side_effect=_fake_call)

    @pytest.mark.asyncio
    async def test_all_subagents_pass_returns_no_violations(self, sample_block_diagram):
        from unittest.mock import patch
        from orchestrator.architecture.constraints import check_constraints

        # Every subagent returns PASS
        mock_call = self._make_subagent_mock(per_check_response={})
        with patch("orchestrator.langchain.agents.coresmith_llm.ClaudeLLM") as MockLLM:
            MockLLM.return_value.call = mock_call
            violations = await check_constraints(
                block_diagram=sample_block_diagram,
                memory_map={"result": {"peripherals": [], "sram": {}}},
                clock_tree={},
                register_spec={},
            )
        assert violations == []

    @pytest.mark.asyncio
    async def test_one_subagent_failure_surfaces_as_violation(self, sample_block_diagram):
        from unittest.mock import patch
        from orchestrator.architecture.constraints import check_constraints

        # gate_budget subagent fails; others pass
        mock_call = self._make_subagent_mock(per_check_response={
            "gate_budget": {
                "pass": False,
                "violation_text": "Total gate count 3,000,000 exceeds budget 2,000,000.",
                "evidence": "block_diagram.blocks[0].estimated_gates=3000000",
                "suggested_fix": "Split huge_block into smaller blocks.",
            },
        })
        diagram = {
            "blocks": [{"name": "huge_block", "estimated_gates": 3_000_000, "tier": 3}],
            "connections": [],
        }
        with patch("orchestrator.langchain.agents.coresmith_llm.ClaudeLLM") as MockLLM:
            MockLLM.return_value.call = mock_call
            violations = await check_constraints(
                block_diagram=diagram,
                memory_map={},
                clock_tree={},
                register_spec={},
            )
        assert len(violations) == 1
        v = violations[0]
        assert v["check"] == "gate_budget"
        assert v["category"] == "auto_fixable"
        assert v["severity"] == "error"
        assert "3,000,000" in v["violation"]
        assert "estimated_gates" in v["evidence"]
        assert "Split huge_block" in v["suggested_fix"]

    @pytest.mark.asyncio
    async def test_applies_predicate_skips_subagent(self, sample_block_diagram):
        """Constraints with applies() returning False shouldn't call the LLM
        at all. CDC presence requires multiple clock domains; with an empty
        clock_tree it must be skipped."""
        from unittest.mock import patch
        from orchestrator.architecture.constraints import check_constraints

        called_ids: list[str] = []

        async def _spy(*args, **kwargs):
            run_name = kwargs.get("run_name", "")
            check_id = run_name.split(":", 1)[1] if ":" in run_name else ""
            called_ids.append(check_id)
            import json as _json
            return _json.dumps({"pass": True, "evidence": "ok"})

        from unittest.mock import AsyncMock
        with patch("orchestrator.langchain.agents.coresmith_llm.ClaudeLLM") as MockLLM:
            MockLLM.return_value.call = AsyncMock(side_effect=_spy)
            await check_constraints(
                block_diagram=sample_block_diagram,
                memory_map={},
                clock_tree={},   # no domains
                register_spec={},
            )

        # CDC subagent must NOT have been invoked
        assert "clock_domain_crossings" not in called_ids
        # block_connectivity always applies (>= 2 blocks in fixture)
        assert "block_connectivity" in called_ids

    @pytest.mark.asyncio
    async def test_subagent_invalid_json_is_treated_as_failure(self, sample_block_diagram):
        """If a subagent returns unparseable output, the check is reported as
        a warning-severity violation rather than silently passing."""
        from unittest.mock import AsyncMock, patch
        from orchestrator.architecture.constraints import check_constraints

        with patch("orchestrator.langchain.agents.coresmith_llm.ClaudeLLM") as MockLLM:
            MockLLM.return_value.call = AsyncMock(return_value="I dunno, looks fine to me")
            violations = await check_constraints(
                block_diagram=sample_block_diagram,
                memory_map={},
                clock_tree={},
                register_spec={},
            )

        # Every applicable subagent will produce a violation (parse failure)
        assert len(violations) > 0
        for v in violations:
            assert "Subagent response was not valid JSON" in v["violation"] or \
                   v["check"].endswith("_subagent_error") or v["category"] == "structural"

    @pytest.mark.asyncio
    async def test_subagent_exception_is_warning(self, sample_block_diagram):
        """A subagent that raises (timeout, API error) becomes a warning-severity
        violation tagged with `_subagent_error`."""
        from unittest.mock import AsyncMock, patch
        from orchestrator.architecture.constraints import check_constraints

        with patch("orchestrator.langchain.agents.coresmith_llm.ClaudeLLM") as MockLLM:
            MockLLM.return_value.call = AsyncMock(side_effect=RuntimeError("LLM timeout"))
            violations = await check_constraints(
                block_diagram=sample_block_diagram,
                memory_map={},
                clock_tree={},
                register_spec={},
            )

        assert any(
            v["check"].endswith("_subagent_error") and v["severity"] == "warning"
            for v in violations
        )

    def test_shuttle_checks_skipped_by_default(self, sample_block_diagram):
        """The shuttle GPIO / area checks are opt-in; default off for soft IP."""
        from orchestrator.architecture.constraints import _shuttle_constraints_enabled

        assert _shuttle_constraints_enabled() is False

    def test_inter_block_payload_protocol_coherence_registered(self):
        """The bit-ledger + handshake-protocol + flow-control subagent must
        be present, must always apply when connections exist, and must
        prompt for all three dimensions."""
        from orchestrator.architecture.constraints import _CONSTRAINT_CATALOG

        match = [c for c in _CONSTRAINT_CATALOG if c["id"] == "inter_block_payload_protocol_coherence"]
        assert len(match) == 1
        c = match[0]
        assert c["category"] == "structural"
        assert c["severity"] == "error"
        # The applies() predicate triggers whenever there's at least one connection.
        assert c["applies"]({"block_diagram": {"connections": [{"from": "a", "to": "b"}]}}) is True
        assert c["applies"]({"block_diagram": {"connections": []}}) is False
        # The description must explicitly cover all three dimensions the
        # user asked about (bit ledger, protocol family, flow control).
        desc = c["description"].lower()
        assert "bit" in desc and "ledger" in desc, "must mention bit ledger"
        assert "axi-stream" in desc, "must mention AXI-Stream as a protocol family"
        assert "ocp" in desc or "avalon" in desc or "wishbone" in desc, "must enumerate non-AXI families"
        assert "bubble" in desc or "fifo" in desc, "must cover flow-control bubbles/buffering"
        assert "backpressure" in desc, "must call out backpressure"
        assert "rate" in desc or "burst" in desc, "must mention rate/burst matching"

    @pytest.mark.asyncio
    async def test_inter_block_coherence_detects_bit_ledger_disagreement(self, sample_block_diagram):
        """When the subagent reports a bit-ledger disagreement between
        producer and consumer of a connection, the violation flows through
        with the right check id."""
        from unittest.mock import patch
        from orchestrator.architecture.constraints import check_constraints

        mock_call = self._make_subagent_mock(per_check_response={
            "inter_block_payload_protocol_coherence": {
                "pass": False,
                "violation_text": (
                    "raster_block_assembler.m_axis_mb_lookup_tdata packs "
                    "[6:0]=mb_x but recon_neighbor_store.s_axis_mb_lookup_tdata "
                    "decodes [20:14]=mb_x; same 21-bit width, reversed bit order."
                ),
                "evidence": (
                    "arch/uarch_specs/raster_block_assembler.md ledger differs "
                    "from arch/uarch_specs/recon_neighbor_store.md ledger"
                ),
                "suggested_fix": (
                    "Define a single mb_meta21 type in the SAD; regenerate "
                    "recon_neighbor_store to consume it."
                ),
            },
        })
        with patch("orchestrator.langchain.agents.coresmith_llm.ClaudeLLM") as MockLLM:
            MockLLM.return_value.call = mock_call
            violations = await check_constraints(
                block_diagram=sample_block_diagram,
                memory_map={},
                clock_tree={},
                register_spec={},
            )

        match = [v for v in violations if v["check"] == "inter_block_payload_protocol_coherence"]
        assert len(match) == 1
        v = match[0]
        assert v["severity"] == "error"
        assert v["category"] == "structural"
        assert "mb_x" in v["violation"]
        assert "mb_meta21" in v["suggested_fix"]

    def test_anti_pattern_warnings_no_longer_in_corpus(self):
        """Smoke test: the regex extractor was removed, so requirements-doc
        anti-pattern phrases no longer feed a claim-extraction stage at all.
        The subagent receives the requirements text verbatim and decides
        per its system prompt how to treat anti-pattern lines."""
        from orchestrator.architecture import constraints as cmod

        for forbidden in ("_extract_dimension_facts", "_check_derived_constraints",
                          "_check_performance_constraints", "_artifact_text",
                          "_select_derived_geometry_pair"):
            assert not hasattr(cmod, forbidden), (
                f"Removed regex helper '{forbidden}' reappeared in constraints.py"
            )


# ---------------------------------------------------------------------------
# Stub specialist tests
# ---------------------------------------------------------------------------


class TestStubSpecialists:
    @pytest.mark.asyncio
    async def test_memory_map_simple_design(self, sample_block_diagram):
        """With <= 3 blocks and no bus infra, analyze_memory_map returns a
        simplified no-op result (simple-design escape hatch)."""
        from orchestrator.architecture.specialists.memory_map import analyze_memory_map

        result = await analyze_memory_map(sample_block_diagram)

        assert result["questions"] == []
        mm = result["result"]
        assert mm["peripherals"] == []
        assert mm["peripheral_count"] == 0
        assert mm["sram"] is None

    @pytest.mark.asyncio
    async def test_clock_tree_via_llm(self, sample_block_diagram):
        """Clock tree now uses LLM; verify structure with mocked response."""
        from unittest.mock import AsyncMock, patch

        llm_response = json.dumps({
            "domains": [{"name": "clk_sys", "frequency_mhz": 100.0, "source": "PLL"}],
            "crossings": [],
            "reset_spec": {"strategy": "synchronous", "domains": ["clk_sys"]},
            "num_domains": 1,
            "cdc_required": False,
        })

        with patch(
            "orchestrator.langchain.agents.coresmith_llm.ClaudeLLM"
        ) as MockLLM:
            MockLLM.return_value.call = AsyncMock(return_value=llm_response)
            from orchestrator.architecture.specialists.clock_tree import analyze_clock_tree
            result = await analyze_clock_tree(sample_block_diagram, target_clock_mhz=100.0)

        ct = result["result"]
        assert len(ct["domains"]) == 1
        assert ct["domains"][0]["frequency_mhz"] == 100.0
        assert ct["cdc_required"] is False

    @pytest.mark.asyncio
    async def test_register_spec_via_llm(self, sample_block_diagram):
        """Register spec now uses LLM; verify structure with mocked response."""
        from unittest.mock import AsyncMock, patch

        llm_response = json.dumps({
            "total_blocks": 4,
            "blocks": [
                {"name": "scrambler", "num_config": 8, "num_status": 8,
                 "registers": []},
                {"name": "conv_encoder", "num_config": 8, "num_status": 8,
                 "registers": []},
                {"name": "fft_engine", "num_config": 8, "num_status": 8,
                 "registers": []},
                {"name": "top_csr", "num_config": 8, "num_status": 8,
                 "registers": []},
            ],
        })

        with patch(
            "orchestrator.langchain.agents.coresmith_llm.ClaudeLLM"
        ) as MockLLM:
            MockLLM.return_value.call = AsyncMock(return_value=llm_response)
            from orchestrator.architecture.specialists.register_spec import (
                analyze_register_spec,
            )
            result = await analyze_register_spec(sample_block_diagram)

        rs = result["result"]
        assert rs["total_blocks"] == 4
        assert any(b["name"] == "scrambler" for b in rs["blocks"])
        assert any(b["name"] == "top_csr" for b in rs["blocks"])

        scrambler_block = next(b for b in rs["blocks"] if b["name"] == "scrambler")
        assert scrambler_block["num_config"] == 8
        assert scrambler_block["num_status"] == 8


# ---------------------------------------------------------------------------
# Benchmark cache tests
# ---------------------------------------------------------------------------


class TestBenchmarkCache:
    def test_store_and_retrieve(self, tmp_project):
        from orchestrator.architecture.benchmarks.cache import BenchmarkCache

        db_path = os.path.join(tmp_project, ".coresmith", "benchmark_cache.db")
        cache = BenchmarkCache(db_path)

        result = {"gate_count": 847, "area_um2": 12340}
        cache.store("multiplier", {"width": 16}, "sky130", 50.0, result)

        cached = cache.get("multiplier", {"width": 16}, "sky130", 50.0)
        assert cached is not None
        assert cached["gate_count"] == 847
        assert cached["cached"] is True

        cache.close()

    def test_cache_miss(self, tmp_project):
        from orchestrator.architecture.benchmarks.cache import BenchmarkCache

        db_path = os.path.join(tmp_project, ".coresmith", "benchmark_cache.db")
        cache = BenchmarkCache(db_path)

        cached = cache.get("multiplier", {"width": 16}, "sky130", 50.0)
        assert cached is None

        cache.close()

    def test_different_params_different_keys(self, tmp_project):
        from orchestrator.architecture.benchmarks.cache import BenchmarkCache

        db_path = os.path.join(tmp_project, ".coresmith", "benchmark_cache.db")
        cache = BenchmarkCache(db_path)

        cache.store("multiplier", {"width": 16}, "sky130", 50.0, {"gate_count": 847})
        cache.store("multiplier", {"width": 32}, "sky130", 50.0, {"gate_count": 3412})

        c16 = cache.get("multiplier", {"width": 16}, "sky130", 50.0)
        c32 = cache.get("multiplier", {"width": 32}, "sky130", 50.0)
        assert c16["gate_count"] == 847
        assert c32["gate_count"] == 3412

        cache.close()

    def test_clear(self, tmp_project):
        from orchestrator.architecture.benchmarks.cache import BenchmarkCache

        db_path = os.path.join(tmp_project, ".coresmith", "benchmark_cache.db")
        cache = BenchmarkCache(db_path)

        cache.store("multiplier", {"width": 16}, "sky130", 50.0, {"gate_count": 847})
        cache.clear()

        assert cache.get("multiplier", {"width": 16}, "sky130", 50.0) is None
        cache.close()


# ---------------------------------------------------------------------------
# Benchmark template rendering tests
# ---------------------------------------------------------------------------


class TestBenchmarkTemplates:
    def test_multiplier_template(self):
        from orchestrator.architecture.benchmarks.runner import _render_template

        verilog = _render_template("multiplier", {"width": 16})
        assert "module benchmark_multiplier" in verilog
        assert "[15:0]" in verilog  # width-1 = 15
        assert "[31:0]" in verilog  # 2*width-1 = 31

    def test_fifo_template(self):
        from orchestrator.architecture.benchmarks.runner import _render_template

        verilog = _render_template("fifo", {"width": 8, "depth": 64})
        assert "module benchmark_fifo" in verilog
        assert "[7:0]" in verilog

    def test_sram_array_template(self):
        from orchestrator.architecture.benchmarks.runner import _render_template

        verilog = _render_template("sram_array", {"width": 8, "depth": 4096})
        assert "module benchmark_sram_array" in verilog

    def test_fft_butterfly_template(self):
        from orchestrator.architecture.benchmarks.runner import _render_template

        verilog = _render_template("fft_butterfly", {"width": 16, "radix": 2})
        assert "module benchmark_fft_butterfly" in verilog
        assert "signed" in verilog  # FFT uses signed arithmetic

    def test_counter_template(self):
        from orchestrator.architecture.benchmarks.runner import _render_template

        verilog = _render_template("counter", {"width": 32})
        assert "module benchmark_counter" in verilog
        assert "[31:0]" in verilog


# ---------------------------------------------------------------------------
# Block specs JSON roundtrip test
# ---------------------------------------------------------------------------


class TestBlockSpecsRoundtrip:
    def test_block_specs_json_roundtrip(self, tmp_project, sample_block_diagram):
        """Verify that finalize -> block_specs.json roundtrip works."""
        from orchestrator.architecture.state import ArchitectureState

        ArchitectureState(
            requirements="test",
            block_diagram=sample_block_diagram,
        )

        # Simulate finalize_architecture
        block_specs = []
        for block in sample_block_diagram["blocks"]:
            spec = {
                "name": block["name"],
                "tier": block["tier"],
                "python_source": block["python_source"],
                "rtl_target": block["rtl_target"],
                "testbench": block["testbench"],
                "description": block["description"],
            }
            block_specs.append(spec)

        specs_path = Path(tmp_project) / ".coresmith" / "block_specs.json"
        specs_path.write_text(json.dumps(block_specs, indent=2))

        # Verify JSON roundtrip
        loaded = json.loads(specs_path.read_text())
        assert len(loaded) == 3
        assert loaded[0]["name"] == "scrambler"
        assert loaded[2]["name"] == "fft_engine"
        assert loaded[2]["tier"] == 3


# ---------------------------------------------------------------------------
# Integration: end-to-end state flow
# ---------------------------------------------------------------------------


class TestEndToEndStateFlow:
    @pytest.mark.asyncio
    async def test_full_architecture_flow(self, tmp_project, sample_block_diagram):
        """Test the complete state flow: init -> block diagram -> memory map ->
        clock -> registers -> constraints -> finalize.

        All specialists now use LLMs; mock them to keep this as a unit test.
        """
        from unittest.mock import AsyncMock, patch
        from orchestrator.architecture.state import ArchitectureState, load_state, save_state
        from orchestrator.architecture.specialists.memory_map import analyze_memory_map
        from orchestrator.architecture.specialists.clock_tree import analyze_clock_tree
        from orchestrator.architecture.specialists.register_spec import analyze_register_spec
        from orchestrator.architecture.constraints import check_constraints

        ct_response = json.dumps({
            "domains": [{"name": "clk_sys", "frequency_mhz": 50.0, "source": "PLL"}],
            "crossings": [], "num_domains": 1, "cdc_required": False,
            "reset_spec": {"strategy": "synchronous", "domains": ["clk_sys"]},
        })
        rs_response = json.dumps({
            "total_blocks": 4,
            "blocks": [
                {"name": "scrambler", "num_config": 8, "num_status": 8, "registers": []},
                {"name": "conv_encoder", "num_config": 8, "num_status": 8, "registers": []},
                {"name": "fft_engine", "num_config": 8, "num_status": 8, "registers": []},
                {"name": "top_csr", "num_config": 8, "num_status": 8, "registers": []},
            ],
        })
        cc_response = json.dumps({"violations": [], "reasoning": "All checks pass."})

        state = ArchitectureState(
            requirements="DVB-T transceiver",
            target_clock_mhz=50.0,
        )
        state.block_diagram = sample_block_diagram
        save_state(state, tmp_project)

        mm = await analyze_memory_map(sample_block_diagram)
        state.memory_map = mm
        save_state(state, tmp_project)

        with patch("orchestrator.langchain.agents.coresmith_llm.ClaudeLLM") as MockLLM:
            MockLLM.return_value.call = AsyncMock(return_value=ct_response)
            ct = await analyze_clock_tree(sample_block_diagram, 50.0)
        state.clock_tree = ct
        save_state(state, tmp_project)

        with patch("orchestrator.langchain.agents.coresmith_llm.ClaudeLLM") as MockLLM:
            MockLLM.return_value.call = AsyncMock(return_value=rs_response)
            rs = await analyze_register_spec(sample_block_diagram)
        state.register_spec = rs
        save_state(state, tmp_project)

        with patch("orchestrator.langchain.agents.coresmith_llm.ClaudeLLM") as MockLLM:
            MockLLM.return_value.call = AsyncMock(return_value=cc_response)
            violations = await check_constraints(
                block_diagram=sample_block_diagram,
                memory_map=mm,
                clock_tree=ct,
                register_spec=rs,
            )
        assert violations == []

        block_specs = []
        for block in sample_block_diagram["blocks"]:
            block_specs.append({
                "name": block["name"],
                "tier": block["tier"],
                "python_source": block["python_source"],
                "rtl_target": block["rtl_target"],
                "testbench": block["testbench"],
                "description": block["description"],
            })
        state.block_specs = block_specs
        save_state(state, tmp_project)

        final = load_state(tmp_project)
        assert final.requirements == "DVB-T transceiver"
        assert len(final.block_specs) == 3
