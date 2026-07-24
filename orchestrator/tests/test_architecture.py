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
        import json as _json
        from unittest.mock import AsyncMock

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

        from orchestrator.architecture.constraints import check_constraints
        from orchestrator.architecture.specialists.clock_tree import analyze_clock_tree
        from orchestrator.architecture.specialists.memory_map import analyze_memory_map
        from orchestrator.architecture.specialists.register_spec import analyze_register_spec
        from orchestrator.architecture.state import ArchitectureState, load_state, save_state

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
        cc_response = json.dumps({
            "pass": True,
            "violation_text": "",
            "evidence": "All checks pass.",
            "suggested_fix": "",
        })

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


# ---------------------------------------------------------------------------
# Interface Definition specialist (Stage B)
# ---------------------------------------------------------------------------


class TestInterfaceDefinition:
    """The Interface Definition specialist freezes per-edge bit-level
    contracts before per-block uArch specs are generated. These tests
    mock the LLM and exercise the structural validation + I/O path."""

    @pytest.mark.asyncio
    async def test_no_op_when_no_connections(self, tmp_project):
        from orchestrator.architecture.specialists.interface_definition import (
            analyze_interface_definition,
        )

        result = await analyze_interface_definition(
            block_diagram={"blocks": [{"name": "solo"}], "connections": []},
            project_root=tmp_project,
        )
        assert result["result"]["contracts"] == []
        assert "Single-block" in result["result"]["design_summary"]
        # No file written for a no-op design.
        from pathlib import Path
        assert not (Path(tmp_project) / ".coresmith" / "interface_contracts.json").exists()

    @pytest.mark.asyncio
    async def test_specialist_persists_contracts_to_disk(self, tmp_project):
        """The specialist should write interface_contracts.json with the
        canonical schema. We mock the LLM to return a known good response."""
        from pathlib import Path
        from unittest.mock import AsyncMock, patch

        from orchestrator.architecture.specialists.interface_definition import (
            analyze_interface_definition,
        )

        contracts_payload = {
            "design_summary": "Two-block AXI-Stream pipeline.",
            "default_packing_convention": "msb_first_by_field_list",
            "default_endianness_rationale": "Matches the video codec byte serialization.",
            "contracts": [
                {
                    "edge_id": "a__m_axis_data__to__b__s_axis_data",
                    "producer_block": "a",
                    "producer_port": "m_axis_data",
                    "consumer_block": "b",
                    "consumer_port": "s_axis_data",
                    "handshake_protocol": "axi_stream",
                    "data_width_bits": 16,
                    "sideband_signals": [{"name": "tlast", "purpose": "end-of-frame"}],
                    "fields": [
                        {"name": "high", "msb": 15, "lsb": 8, "width": 8,
                         "signed": False, "encoding": "binary"},
                        {"name": "low", "msb": 7, "lsb": 0, "width": 8,
                         "signed": False, "encoding": "binary"},
                    ],
                    "packing_convention": "msb_first_by_field_list",
                    "bootstrap_policy": {"required": False, "policy_type": "none"},
                },
            ],
            "open_questions": [],
        }
        import json as _json
        target = Path(tmp_project) / ".coresmith" / "interface_contracts.json"

        async def _fake_call(*args, **kwargs):
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(_json.dumps(contracts_payload, indent=2))
            return f"Wrote contracts to {target}"

        with patch("orchestrator.langchain.agents.coresmith_llm.ClaudeLLM") as MockLLM:
            MockLLM.return_value.call = AsyncMock(side_effect=_fake_call)
            result = await analyze_interface_definition(
                block_diagram={
                    "blocks": [{"name": "a"}, {"name": "b"}],
                    "connections": [{"from": "a", "to": "b"}],
                },
                project_root=tmp_project,
            )

        assert target.exists()
        ir = result["result"]
        assert len(ir["contracts"]) == 1
        assert ir["contracts"][0]["data_width_bits"] == 16

    @staticmethod
    def _completion_mislabel_bundle(target):
        """A block diagram + LLM contract payload where the specialist wrongly
        froze request_response + feedback_cycle=true on a valid_only strobe, a
        static bundle, and a mem_write port (it inferred a response because a
        separate completion event exists), a legitimately request_response
        req_resp read, and a genuine srdy_drdy stream carrying elastic_fifo."""
        import json as _json

        diagram = {
            "blocks": [{"name": n} for n in ("ctrl", "core", "store", "sa", "sb")],
            "connections": [
                {"from": "ctrl", "to": "core", "handshake_protocol": "valid_only"},
                {"from": "core", "to": "ctrl", "handshake_protocol": "static"},
                {"from": "core", "to": "store", "handshake_protocol": "mem_write"},
                {"from": "ctrl", "to": "store", "handshake_protocol": "req_resp"},
                {"from": "sa", "to": "sb", "handshake_protocol": "srdy_drdy"},
            ],
        }

        def _c(prod, cons, fam, fc):
            return {
                "producer_block": prod, "consumer_block": cons,
                "handshake_protocol": fam, "data_width_bits": 8,
                "fields": [{"name": "d", "width": 8, "msb": 7, "lsb": 0}],
                "flow_control_policy": fc,
            }

        rr = {"semantics": "request_response", "feedback_cycle": True,
              "consumer_can_stall": True, "min_buffer_depth_beats": 4}
        payload = {
            "design_summary": "generic completion-event mislabel demo",
            "default_packing_convention": "msb_first_by_field_list",
            "contracts": [
                _c("ctrl", "core", "valid_only", dict(rr)),
                _c("core", "ctrl", "static", dict(rr)),
                _c("core", "store", "mem_write", dict(rr)),
                _c("ctrl", "store", "req_resp",
                   {"semantics": "request_response", "feedback_cycle": False}),
                _c("sa", "sb", "srdy_drdy",
                   {"semantics": "elastic_fifo", "min_buffer_depth_beats": 4,
                    "feedback_cycle": True, "consumer_can_stall": True}),
            ],
            "open_questions": [],
        }

        async def _fake_call(*args, **kwargs):
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(_json.dumps(payload, indent=2))
            return f"wrote {target}"

        return diagram, _fake_call

    @pytest.mark.asyncio
    async def test_neutralized_policy_persisted_to_disk(self, tmp_project, monkeypatch):
        """The neutralized (free_running) flow_control_policy for the
        no-backpressure families must be written BACK to interface_contracts.json
        -- the coherence gate and per-block spec generators read the DISK file,
        so an in-memory-only fix does not stick."""
        import json as _json
        from pathlib import Path
        from unittest.mock import AsyncMock, patch

        from orchestrator.architecture.constraints import (
            _check_interface_family_coherence,
        )
        from orchestrator.architecture.specialists.interface_definition import (
            analyze_interface_definition,
        )

        monkeypatch.delenv("CORESMITH_INTERFACE_FAMILY_PROPAGATION", raising=False)
        monkeypatch.delenv("CORESMITH_INTERFACE_FAMILY_GATE", raising=False)
        target = Path(tmp_project) / ".coresmith" / "interface_contracts.json"
        diagram, fake_call = self._completion_mislabel_bundle(target)

        with patch("orchestrator.langchain.agents.coresmith_llm.ClaudeLLM") as MockLLM:
            MockLLM.return_value.call = AsyncMock(side_effect=fake_call)
            await analyze_interface_definition(
                block_diagram=diagram, project_root=tmp_project,
            )

        # Re-read the DISK artifact -- this is what downstream actually consumes.
        on_disk = _json.loads(target.read_text())
        by_fam = {c["handshake_protocol"]: c for c in on_disk["contracts"]}

        for fam in ("valid_only", "static", "mem_write"):
            fc = by_fam[fam].get("flow_control_policy") or {}
            assert fc["semantics"] == "free_running", (fam, fc)
            assert fc.get("feedback_cycle") is False, (fam, fc)
            assert int(fc.get("min_buffer_depth_beats") or 0) == 0, (fam, fc)
            assert not fc.get("consumer_can_stall"), (fam, fc)
            assert not fc.get("producer_can_stall"), (fam, fc)

        # req_resp + the genuine streaming edge are UNTOUCHED on disk.
        assert (by_fam["req_resp"]["flow_control_policy"]["semantics"]
                == "request_response")
        sfc = by_fam["srdy_drdy"]["flow_control_policy"]
        assert sfc["semantics"] == "elastic_fifo"
        assert int(sfc.get("min_buffer_depth_beats") or 0) == 4

        # The honest coherence gate (which reloads from disk) now passes clean.
        assert _check_interface_family_coherence(diagram, on_disk) == []

    @pytest.mark.asyncio
    async def test_neutralized_policy_writeback_gate_off(self, tmp_project, monkeypatch):
        """Gate OFF -> the raw LLM policy is left on disk verbatim (pre-fix
        behavior); the honest gate still catches it."""
        import json as _json
        from pathlib import Path
        from unittest.mock import AsyncMock, patch

        from orchestrator.architecture.constraints import (
            _check_interface_family_coherence,
        )
        from orchestrator.architecture.specialists.interface_definition import (
            analyze_interface_definition,
        )

        monkeypatch.setenv("CORESMITH_INTERFACE_FAMILY_PROPAGATION", "0")
        monkeypatch.delenv("CORESMITH_INTERFACE_FAMILY_GATE", raising=False)
        target = Path(tmp_project) / ".coresmith" / "interface_contracts.json"
        diagram, fake_call = self._completion_mislabel_bundle(target)

        with patch("orchestrator.langchain.agents.coresmith_llm.ClaudeLLM") as MockLLM:
            MockLLM.return_value.call = AsyncMock(side_effect=fake_call)
            await analyze_interface_definition(
                block_diagram=diagram, project_root=tmp_project,
            )

        on_disk = _json.loads(target.read_text())
        by_fam = {c["handshake_protocol"]: c for c in on_disk["contracts"]}
        # Untouched: still request_response on the no-backpressure families.
        for fam in ("valid_only", "static", "mem_write"):
            assert (by_fam[fam]["flow_control_policy"]["semantics"]
                    == "request_response")
        # And the honest gate still catches them (fix is the generator, not the
        # gate; the gate default-ON is orthogonal to the propagation gate).
        assert _check_interface_family_coherence(diagram, on_disk)

    @pytest.mark.asyncio
    async def test_validator_flags_field_width_sum_mismatch(self, tmp_project):
        """The structural validator should note when declared
        data_width_bits doesn't equal the sum of field widths."""
        from pathlib import Path
        from unittest.mock import AsyncMock, patch

        from orchestrator.architecture.specialists.interface_definition import (
            analyze_interface_definition,
        )

        bad_payload = {
            "design_summary": "drift demo",
            "default_packing_convention": "msb_first_by_field_list",
            "contracts": [
                {
                    "edge_id": "a__m__to__b__s",
                    "producer_block": "a",
                    "producer_port": "m_axis_x",
                    "consumer_block": "b",
                    "consumer_port": "s_axis_x",
                    "handshake_protocol": "axi_stream",
                    "data_width_bits": 32,  # declared
                    "fields": [
                        {"name": "f1", "msb": 15, "lsb": 0, "width": 16},
                    ],  # actual sum = 16
                    "packing_convention": "msb_first_by_field_list",
                },
            ],
            "open_questions": [],
        }
        import json as _json
        target = Path(tmp_project) / ".coresmith" / "interface_contracts.json"

        async def _fake_call(*args, **kwargs):
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(_json.dumps(bad_payload))
            return f"wrote {target}"

        with patch("orchestrator.langchain.agents.coresmith_llm.ClaudeLLM") as MockLLM:
            MockLLM.return_value.call = AsyncMock(side_effect=_fake_call)
            result = await analyze_interface_definition(
                block_diagram={
                    "blocks": [{"name": "a"}, {"name": "b"}],
                    "connections": [{"from": "a", "to": "b"}],
                },
                project_root=tmp_project,
            )

        notes = result["result"].get("validation_notes", [])
        assert any("declared data_width_bits=32" in n for n in notes), (
            f"expected width-sum drift note, got: {notes}"
        )

    @pytest.mark.asyncio
    async def test_validator_flags_missing_contract_for_declared_edge(self, tmp_project):
        """Every directed edge in block_diagram must be represented by
        exactly one contract entry. Validator should flag missing ones."""
        from pathlib import Path
        from unittest.mock import AsyncMock, patch

        from orchestrator.architecture.specialists.interface_definition import (
            analyze_interface_definition,
        )

        payload = {
            "design_summary": "partial coverage",
            "default_packing_convention": "msb_first_by_field_list",
            "contracts": [
                {
                    "edge_id": "a__m__to__b__s",
                    "producer_block": "a",
                    "consumer_block": "b",
                    "data_width_bits": 8,
                    "fields": [{"name": "byte", "msb": 7, "lsb": 0, "width": 8}],
                },
                # NOTE: missing the b->c contract
            ],
            "open_questions": [],
        }
        import json as _json
        target = Path(tmp_project) / ".coresmith" / "interface_contracts.json"

        async def _fake_call(*args, **kwargs):
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(_json.dumps(payload))
            return f"wrote {target}"

        with patch("orchestrator.langchain.agents.coresmith_llm.ClaudeLLM") as MockLLM:
            MockLLM.return_value.call = AsyncMock(side_effect=_fake_call)
            result = await analyze_interface_definition(
                block_diagram={
                    "blocks": [{"name": "a"}, {"name": "b"}, {"name": "c"}],
                    "connections": [
                        {"from": "a", "to": "b"},
                        {"from": "b", "to": "c"},
                    ],
                },
                project_root=tmp_project,
            )

        notes = result["result"].get("validation_notes", [])
        assert any("b -> c" in n for n in notes), (
            f"expected missing-contract note for b->c, got: {notes}"
        )

    @pytest.mark.asyncio
    async def test_validator_flags_freerunning_on_cycle_edge(self, tmp_project):
        """An edge that is part of a directed cycle must NOT have
        flow_control_policy.semantics == free_running (or skid)."""
        from pathlib import Path
        from unittest.mock import AsyncMock, patch

        from orchestrator.architecture.specialists.interface_definition import (
            analyze_interface_definition,
        )

        # a -> b -> a cycle: both edges are on a cycle.
        payload = {
            "design_summary": "tight feedback loop",
            "default_packing_convention": "msb_first_by_field_list",
            "contracts": [
                {
                    "edge_id": "a__m__to__b__s",
                    "producer_block": "a",
                    "consumer_block": "b",
                    "data_width_bits": 8,
                    "fields": [{"name": "x", "msb": 7, "lsb": 0, "width": 8}],
                    "flow_control_policy": {
                        "semantics": "free_running",
                        "feedback_cycle": False,
                        "rationale": "wrong: producer cannot stall on this loop",
                    },
                },
                {
                    "edge_id": "b__m__to__a__s",
                    "producer_block": "b",
                    "consumer_block": "a",
                    "data_width_bits": 8,
                    "fields": [{"name": "y", "msb": 7, "lsb": 0, "width": 8}],
                    "flow_control_policy": {
                        "semantics": "elastic_fifo",
                        "min_buffer_depth_beats": 4,
                        "feedback_cycle": True,
                        "rationale": "absorbs backpressure",
                    },
                },
            ],
            "open_questions": [],
        }
        import json as _json
        target = Path(tmp_project) / ".coresmith" / "interface_contracts.json"

        async def _fake_call(*args, **kwargs):
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(_json.dumps(payload))
            return f"wrote {target}"

        with patch("orchestrator.langchain.agents.coresmith_llm.ClaudeLLM") as MockLLM:
            MockLLM.return_value.call = AsyncMock(side_effect=_fake_call)
            result = await analyze_interface_definition(
                block_diagram={
                    "blocks": [{"name": "a"}, {"name": "b"}],
                    "connections": [
                        {"from": "a", "to": "b"},
                        {"from": "b", "to": "a"},
                    ],
                },
                project_root=tmp_project,
            )

        notes = result["result"].get("validation_notes", [])
        # The a->b edge should be flagged for free_running on a cycle.
        assert any("free_running" in n for n in notes), (
            f"expected free_running-on-cycle violation, got: {notes}"
        )
        # And feedback_cycle should also be flagged (set to False on cycle).
        assert any("feedback_cycle" in n for n in notes), (
            f"expected feedback_cycle mismatch note, got: {notes}"
        )

    @pytest.mark.asyncio
    async def test_validator_flags_shallow_elastic_fifo(self, tmp_project):
        """elastic_fifo with min_buffer_depth_beats < 2 should be flagged."""
        from pathlib import Path
        from unittest.mock import AsyncMock, patch

        from orchestrator.architecture.specialists.interface_definition import (
            analyze_interface_definition,
        )

        payload = {
            "design_summary": "shallow elastic",
            "default_packing_convention": "msb_first_by_field_list",
            "contracts": [
                {
                    "edge_id": "a__m__to__b__s",
                    "producer_block": "a",
                    "consumer_block": "b",
                    "data_width_bits": 8,
                    "fields": [{"name": "x", "msb": 7, "lsb": 0, "width": 8}],
                    "flow_control_policy": {
                        "semantics": "elastic_fifo",
                        "min_buffer_depth_beats": 1,  # too shallow
                        "feedback_cycle": False,
                        "rationale": "shallow",
                    },
                },
            ],
            "open_questions": [],
        }
        import json as _json
        target = Path(tmp_project) / ".coresmith" / "interface_contracts.json"

        async def _fake_call(*args, **kwargs):
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(_json.dumps(payload))
            return f"wrote {target}"

        with patch("orchestrator.langchain.agents.coresmith_llm.ClaudeLLM") as MockLLM:
            MockLLM.return_value.call = AsyncMock(side_effect=_fake_call)
            result = await analyze_interface_definition(
                block_diagram={
                    "blocks": [{"name": "a"}, {"name": "b"}],
                    "connections": [{"from": "a", "to": "b"}],
                },
                project_root=tmp_project,
            )

        notes = result["result"].get("validation_notes", [])
        assert any("elastic_fifo" in n and "min_buffer_depth" in n for n in notes), (
            f"expected shallow-elastic-fifo note, got: {notes}"
        )

    def test_edges_in_cycles_helper(self):
        """The Tarjan-based cycle detector returns exactly the edges in
        any directed cycle of the block diagram."""
        from orchestrator.architecture.specialists.interface_definition import (
            _edges_in_cycles,
        )

        # a -> b -> c -> a (3-cycle), plus d->e (tree edge), self-loop on f.
        adj = {
            "a": ["b"],
            "b": ["c"],
            "c": ["a"],
            "d": ["e"],
            "f": ["f"],
        }
        cycle_edges = _edges_in_cycles(adj)
        assert ("a", "b") in cycle_edges
        assert ("b", "c") in cycle_edges
        assert ("c", "a") in cycle_edges
        assert ("d", "e") not in cycle_edges
        assert ("f", "f") in cycle_edges


# ---------------------------------------------------------------------------
# cross_spec_contract_adherence subagent (Stage C)
# ---------------------------------------------------------------------------


class TestCrossSpecContractAdherence:
    """The new constraint subagent verifies per-block uArch specs match
    the canonical interface_contracts. It only applies when both
    interface_contracts and the per-block specs exist."""

    def _stub_call(self, per_check_response):
        """Same pattern as TestConstraintChecker._make_subagent_mock."""
        import json as _json
        from unittest.mock import AsyncMock

        async def _fake_call(*args, **kwargs):
            run_name = kwargs.get("run_name", "")
            check_id = run_name.split(":", 1)[1] if ":" in run_name else ""
            resp = per_check_response.get(
                check_id, {"pass": True, "evidence": "n/a"}
            )
            return _json.dumps(resp)

        return AsyncMock(side_effect=_fake_call)

    def test_applies_predicate_requires_contracts(self):
        from orchestrator.architecture.constraints import _CONSTRAINT_CATALOG
        c = next(
            x for x in _CONSTRAINT_CATALOG
            if x["id"] == "cross_spec_contract_adherence"
        )
        # No contracts => skip.
        assert c["applies"]({"interface_contracts": {}}) is False
        assert c["applies"]({"interface_contracts": {"contracts": []}}) is False
        # With contracts => apply (the subagent itself decides whether to
        # check per-block specs or no-op).
        assert c["applies"](
            {"interface_contracts": {"contracts": [{"edge_id": "x"}]}}
        ) is True

    @pytest.mark.asyncio
    async def test_subagent_fires_when_contracts_present(self, tmp_project):
        """When interface_contracts.json exists with non-empty contracts,
        the cross_spec_contract_adherence subagent must be invoked."""
        from unittest.mock import patch

        from orchestrator.architecture.constraints import check_constraints

        contracts = {
            "contracts": [
                {"edge_id": "x__m__to__y__s", "producer_block": "x",
                 "consumer_block": "y", "data_width_bits": 8,
                 "fields": [{"name": "b", "msb": 7, "lsb": 0, "width": 8}]},
            ],
        }
        mock_call = self._stub_call(per_check_response={
            "cross_spec_contract_adherence": {
                "pass": False,
                "violation_text": "producer port y's tdata layout differs from contract",
                "evidence": "arch/uarch_specs/x.md: m_axis_data[7:0]=b vs contract msb=7 lsb=0 OK; "
                            "arch/uarch_specs/y.md: s_axis_data[3:0]=b mismatch",
                "suggested_fix": "Edit arch/uarch_specs/y.md to specify tdata[7:0]=b",
            },
        })
        with patch("orchestrator.langchain.agents.coresmith_llm.ClaudeLLM") as MockLLM:
            MockLLM.return_value.call = mock_call
            violations = await check_constraints(
                block_diagram={
                    "blocks": [{"name": "x"}, {"name": "y"}],
                    "connections": [{"from": "x", "to": "y"}],
                },
                memory_map={},
                clock_tree={},
                register_spec={},
                project_root=tmp_project,
                interface_contracts=contracts,
            )
        cross_v = [v for v in violations if v["check"] == "cross_spec_contract_adherence"]
        assert len(cross_v) == 1
        assert "producer port y" in cross_v[0]["violation"]
        assert cross_v[0]["severity"] == "error"

    @pytest.mark.asyncio
    async def test_subagent_skipped_when_no_contracts(self, tmp_project):
        from unittest.mock import patch

        from orchestrator.architecture.constraints import check_constraints

        called: list[str] = []

        async def _spy(*args, **kwargs):
            run_name = kwargs.get("run_name", "")
            check_id = run_name.split(":", 1)[1] if ":" in run_name else ""
            called.append(check_id)
            import json as _json
            return _json.dumps({"pass": True, "evidence": "ok"})

        from unittest.mock import AsyncMock
        with patch("orchestrator.langchain.agents.coresmith_llm.ClaudeLLM") as MockLLM:
            MockLLM.return_value.call = AsyncMock(side_effect=_spy)
            await check_constraints(
                block_diagram={
                    "blocks": [{"name": "a"}, {"name": "b"}],
                    "connections": [{"from": "a", "to": "b"}],
                },
                memory_map={},
                clock_tree={},
                register_spec={},
                project_root=tmp_project,
                # No interface_contracts and no file on disk
            )
        assert "cross_spec_contract_adherence" not in called


# ---------------------------------------------------------------------------
# cross_spec_fifo_depth_adherence subagent (PR #52)
# ---------------------------------------------------------------------------


class TestCrossSpecFifoDepthAdherence:
    """Catches per-block RTL whose FIFO `localparam DEPTH` is below the
    contract's `flow_control_policy.min_buffer_depth_beats`. The v9 codec
    deadlock at 3.3% of frame input was exactly this class of bug."""

    def test_constraint_registered(self):
        from orchestrator.architecture.constraints import _CONSTRAINT_CATALOG
        match = [
            c for c in _CONSTRAINT_CATALOG
            if c["id"] == "cross_spec_fifo_depth_adherence"
        ]
        assert len(match) == 1
        assert match[0]["severity"] == "error"

    def test_applies_only_when_elastic_fifo_edges_exist(self):
        from orchestrator.architecture.constraints import _CONSTRAINT_CATALOG
        c = next(
            x for x in _CONSTRAINT_CATALOG
            if x["id"] == "cross_spec_fifo_depth_adherence"
        )
        # No contracts at all -> skip
        assert c["applies"]({"interface_contracts": {}}) is False
        # Contracts but no elastic_fifo semantics -> skip
        assert c["applies"]({
            "interface_contracts": {
                "contracts": [
                    {"edge_id": "a", "flow_control_policy": {"semantics": "skid"}},
                    {"edge_id": "b", "flow_control_policy": {"semantics": "free_running"}},
                ],
            }
        }) is False
        # At least one elastic_fifo -> apply
        assert c["applies"]({
            "interface_contracts": {
                "contracts": [
                    {"edge_id": "a", "flow_control_policy": {"semantics": "skid"}},
                    {"edge_id": "b", "flow_control_policy": {"semantics": "elastic_fifo"}},
                ],
            }
        }) is True

    def test_artifact_bundle_includes_rtl_heads(self, tmp_project):
        """The audit must see the per-block RTL heads in the bundle so it
        can grep for `localparam DEPTH = N` declarations."""
        from pathlib import Path

        from orchestrator.architecture.constraints import _build_artifact_bundle

        rtl_dir = Path(tmp_project) / "rtl" / "design"
        rtl_dir.mkdir(parents=True)
        sample = (
            "/* Block alpha */\n"
            "module alpha (input clk, input rst_n);\n"
            "    localparam [8:0] FIFO_DEPTH = 9'd256;\n"
            "    reg [31:0] block_fifo_mem [0:255];\n"
            "endmodule\n"
        )
        (rtl_dir / "alpha.v").write_text(sample)

        bundle = _build_artifact_bundle(
            block_diagram={"blocks": [{"name": "alpha"}]},
            memory_map={},
            clock_tree={},
            register_spec={},
            benchmark_results=None,
            pdk_config=None,
            requirements="",
            ers_spec=None,
            project_root=tmp_project,
        )
        assert "rtl/design/alpha.v" in bundle
        assert "FIFO_DEPTH = 9'd256" in bundle
