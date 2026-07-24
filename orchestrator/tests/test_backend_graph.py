# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for the backend (physical design) LangGraph execution graph.

Tests:
- Graph construction (compiles, has expected nodes)
- BackendState schema (includes Backend Lead + artifact fields)
- Routing functions (route_after_pnr, route_after_drc, route_after_lvs,
  route_after_timing, route_decision, route_after_human,
  route_after_increment, route_after_advance_lead)
- Internal nodes (init_design, backend_complete)
- Happy path (no integration top -> flat synth fails -> diagnose -> skip)
"""

from __future__ import annotations

from pathlib import Path as _Path
from unittest.mock import AsyncMock, patch

import pytest
from langgraph.checkpoint.memory import MemorySaver

from orchestrator.langgraph.backend_graph import (
    _PROMPT_DIR,
    BackendState,
    _safe_format,
    advance_block_node,
    backend_complete_node,
    build_backend_graph,
    init_design_node,
    route_after_advance_lead,
    route_after_drc,
    route_after_human,
    route_after_increment,
    route_after_lvs,
    route_after_pnr,
    route_after_precheck,
    route_after_timing,
    route_decision,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_backend_block(name: str, tier: int = 1) -> dict:
    return {
        "name": name,
        "tier": tier,
        "rtl_path": f"rtl/dvbt/{name}.v",
        "netlist_path": f"synth/{name}_netlist.v",
        "sdc_path": f"synth/{name}.sdc",
        "description": f"Backend test block {name}",
    }


def _fft16_backend_blocks():
    return [
        _make_backend_block("fft_butterfly", tier=1),
        _make_backend_block("twiddle_rom", tier=1),
        _make_backend_block("fft_controller", tier=2),
    ]


def _initial_backend_state(blocks=None) -> dict:
    if blocks is None:
        blocks = [_make_backend_block("fft_butterfly")]
    return {
        "project_root": "/tmp/test",
        "target_clock_mhz": 50.0,
        "max_attempts": 3,
        "block_queue": blocks,
        # Backend Lead fields
        "frontend_blocks": blocks,
        "architecture_connections": [],
        "design_name": "test_chip_top",
        "block_rtl_paths": {},
        "glue_blocks": [],
        "integration_top_path": "",
        "flat_netlist_path": "",
        "flat_sdc_path": "",
        "synth_gate_count": 0,
        "synth_area_um2": 0.0,
        # Legacy compat
        "current_block_index": 0,
        "current_block": {},
        "attempt": 1,
        "phase": "init",
        "constraints": [],
        "attempt_history": [],
        "previous_error": "",
        "floorplan_result": None,
        "place_result": None,
        "cts_result": None,
        "route_result": None,
        "drc_result": None,
        "lvs_result": None,
        "timing_result": None,
        "power_result": None,
        "debug_result": None,
        "completed_blocks": [],
        "human_response": None,
        "backend_done": False,
        "routed_def_path": "",
        "pnr_verilog_path": "",
        "pwr_verilog_path": "",
        "spef_path": "",
        "gds_path": "",
        "spice_path": "",
        "step_log_paths": {},
        "final_report_path": "",
    }


# ═══════════════════════════════════════════════════════════════════════════
# Graph Construction
# ═══════════════════════════════════════════════════════════════════════════

class TestGraphConstruction:
    def test_compiles_without_error(self):
        graph = build_backend_graph(checkpointer=MemorySaver())
        assert graph is not None

    def test_compiles_without_checkpointer(self):
        graph = build_backend_graph(checkpointer=None)
        assert graph is not None

    def test_has_expected_nodes(self):
        graph = build_backend_graph(checkpointer=MemorySaver())
        node_names = list(graph.get_graph().nodes.keys())
        expected = [
            "init_design", "flat_top_synthesis", "run_pnr",
            "drc", "lvs", "timing_signoff",
            "diagnose", "decide", "ask_human",
            "increment_attempt", "advance_block", "backend_complete",
            "final_report",
        ]
        for name in expected:
            assert name in node_names, f"Missing node: {name}"

    def test_no_old_stub_nodes(self):
        """Ensure the old separate floorplan/place/cts/route/power nodes are gone."""
        graph = build_backend_graph(checkpointer=MemorySaver())
        node_names = list(graph.get_graph().nodes.keys())
        removed = ["floorplan", "place", "cts", "route", "power_analysis", "init_block"]
        for name in removed:
            assert name not in node_names, f"Old stub node still present: {name}"

    def test_node_count(self):
        graph = build_backend_graph(checkpointer=MemorySaver())
        # 16 real nodes + __start__ + __end__ = 18
        # (added generate_wrapper between timing_signoff and mpw_precheck,
        # and generate_3d_view between backend_complete and final_report)
        node_names = list(graph.get_graph().nodes.keys())
        assert len(node_names) == 18


# ═══════════════════════════════════════════════════════════════════════════
# State Schema
# ═══════════════════════════════════════════════════════════════════════════

class TestBackendState:
    def test_has_required_fields(self):
        annotations = BackendState.__annotations__
        required = [
            "project_root", "target_clock_mhz", "max_attempts", "block_queue",
            "current_block_index", "current_block", "attempt", "phase",
            "constraints", "attempt_history", "previous_error",
            "floorplan_result", "place_result", "cts_result",
            "route_result", "drc_result", "lvs_result",
            "timing_result", "power_result", "debug_result",
            "completed_blocks", "human_response", "backend_done",
            "frontend_blocks", "architecture_connections", "design_name",
        ]
        for f in required:
            assert f in annotations, f"Missing field: {f}"

    def test_has_artifact_path_fields(self):
        """New artifact path fields added for real EDA tool integration."""
        annotations = BackendState.__annotations__
        artifact_fields = [
            "routed_def_path", "pnr_verilog_path", "pwr_verilog_path",
            "spef_path", "gds_path", "spice_path", "step_log_paths",
            "flat_netlist_path", "flat_sdc_path", "integration_top_path",
        ]
        for f in artifact_fields:
            assert f in annotations, f"Missing artifact field: {f}"


# ═══════════════════════════════════════════════════════════════════════════
# Routing Functions
# ═══════════════════════════════════════════════════════════════════════════

class TestRouteAfterPnR:
    def test_success_goes_to_drc(self):
        assert route_after_pnr({"route_result": {"success": True}}) == "drc"

    def test_fail_goes_to_diagnose(self):
        assert route_after_pnr({"route_result": {"success": False}}) == "diagnose"

    def test_missing_result_goes_to_diagnose(self):
        assert route_after_pnr({}) == "diagnose"


class TestRouteAfterDRC:
    def test_clean_goes_to_lvs(self):
        assert route_after_drc({"drc_result": {"clean": True}}) == "lvs"

    def test_fail_goes_to_diagnose(self):
        assert route_after_drc({"drc_result": {"clean": False}}) == "diagnose"

    def test_missing_result_goes_to_diagnose(self):
        assert route_after_drc({}) == "diagnose"


class TestRouteAfterLVS:
    def test_match_goes_to_timing(self):
        assert route_after_lvs({"lvs_result": {"match": True}}) == "timing_signoff"

    def test_fail_goes_to_diagnose(self):
        assert route_after_lvs({"lvs_result": {"match": False}}) == "diagnose"

    def test_missing_result_goes_to_diagnose(self):
        assert route_after_lvs({}) == "diagnose"


class TestRouteAfterTiming:
    def test_met_goes_to_advance(self):
        # After timing_signoff MET -> generate_wrapper (was advance_block before
        # the wrapper-generation step was inserted into the backend pipeline).
        assert route_after_timing({"timing_result": {"met": True}}) == "generate_wrapper"

    def test_violated_goes_to_diagnose(self):
        assert route_after_timing({"timing_result": {"met": False}}) == "diagnose"

    def test_missing_result_goes_to_diagnose(self):
        assert route_after_timing({}) == "diagnose"


class TestRouteAfterPrecheck:
    def test_pass_goes_to_advance(self):
        assert route_after_precheck({"precheck_result": {"pass": True}}) == "advance_block"

    def test_fail_goes_to_diagnose(self):
        assert route_after_precheck({"precheck_result": {"pass": False}}) == "diagnose"

    def test_llm_cannot_override_failed_precheck(self):
        state = {"precheck_result": {"pass": False, "llm_analysis": {"submission_ready": True}}}
        assert route_after_precheck(state) == "diagnose"


class TestRouteDecision:
    def test_retry_pnr(self):
        assert route_decision({"debug_result": {"next_action": "retry_pnr"}}) == "increment_attempt"

    def test_ask_human(self):
        assert route_decision({"debug_result": {"next_action": "ask_human"}}) == "ask_human"

    def test_escalate(self):
        assert route_decision({"debug_result": {"next_action": "escalate"}}) == "advance_block"

    def test_default_is_increment(self):
        assert route_decision({"debug_result": {"next_action": "??"}}) == "increment_attempt"

    def test_missing_action(self):
        assert route_decision({"debug_result": {}}) == "increment_attempt"


class TestRouteAfterHuman:
    def test_retry(self):
        assert route_after_human({"human_response": {"action": "retry"}}) == "increment_attempt"

    def test_skip(self):
        assert route_after_human({"human_response": {"action": "skip"}}) == "advance_block"

    def test_abort(self):
        assert route_after_human({"human_response": {"action": "abort"}}) == "backend_complete"

    def test_default(self):
        assert route_after_human({"human_response": {"action": "??"}}) == "increment_attempt"

    def test_missing_response(self):
        assert route_after_human({}) == "increment_attempt"


class TestRouteAfterIncrement:
    def test_within_limit(self):
        assert route_after_increment({"attempt": 2, "max_attempts": 3}) == "run_pnr"

    def test_at_limit(self):
        assert route_after_increment({"attempt": 3, "max_attempts": 3}) == "run_pnr"

    def test_exhausted(self):
        assert route_after_increment({"attempt": 4, "max_attempts": 3}) == "advance_block"


class TestRouteAfterAdvanceLead:
    def test_always_returns_backend_complete(self):
        state = {"current_block_index": 0, "block_queue": [{}, {}, {}]}
        assert route_after_advance_lead(state) == "backend_complete"

    def test_empty_queue(self):
        state = {"current_block_index": 0, "block_queue": []}
        assert route_after_advance_lead(state) == "backend_complete"


# ═══════════════════════════════════════════════════════════════════════════
# Internal Nodes
# ═══════════════════════════════════════════════════════════════════════════

class TestInternalNodes:
    @pytest.mark.asyncio
    async def test_init_design_sets_current_block(self):
        state = _initial_backend_state(_fft16_backend_blocks())
        result = await init_design_node(state)
        assert result["current_block"]["name"] == "test_chip_top"
        assert result["attempt"] == 1
        assert result["phase"] == "init"
        assert result["step_log_paths"] == {}

    @pytest.mark.asyncio
    async def test_init_design_no_integration_top(self):
        state = _initial_backend_state(_fft16_backend_blocks())
        result = await init_design_node(state)
        assert "No integration top-level RTL" in result.get("previous_error", "")

    @pytest.mark.asyncio
    async def test_backend_complete(self):
        state = {"completed_blocks": [{"name": "a", "success": True}], "project_root": "/tmp/test"}
        result = await backend_complete_node(state)
        assert result["backend_done"] is True

    @pytest.mark.asyncio
    async def test_advance_block_precheck_hard_fail_not_overridden_by_llm(self):
        state = {
            "project_root": "/tmp/test",
            "current_block": {"name": "top"},
            "attempt": 1,
            "drc_result": {"clean": True},
            "lvs_result": {"match": True},
            "timing_result": {"met": True},
            "precheck_result": {"pass": False, "llm_analysis": {"submission_ready": True}},
            "route_result": {"success": True},
            "step_log_paths": {},
            "constraints": [],
        }
        result = await advance_block_node(state)
        assert result["completed_blocks"][0]["success"] is False


# ═══════════════════════════════════════════════════════════════════════════
# Happy Path (full graph -- no integration top -> flat synth fails -> diagnose -> skip)
# ═══════════════════════════════════════════════════════════════════════════

_MOCK_DASHBOARD = AsyncMock(return_value="<html><body>stub</body></html>")
_DASHBOARD_PATCH = (
    "orchestrator.architecture.specialists.chip_finish_dashboard"
    ".generate_chip_finish_dashboard"
)


_DIAGNOSE_PATCH = (
    "orchestrator.architecture.specialists.tapeout_diagnosis"
    ".diagnose_tapeout_failure"
)

_MOCK_DIAGNOSE = AsyncMock(return_value={
    "category": "PNR_FAILURE",
    "diagnosis": "No integration top RTL for synthesis",
    "confidence": 0.2,
    "action": "continue",
    "suggested_fix": "Provide integration top-level RTL",
    "pnr_overrides": {},
})


class TestHappyPath:
    @pytest.mark.asyncio
    @patch(_DASHBOARD_PATCH, _MOCK_DASHBOARD)
    @patch(_DIAGNOSE_PATCH, new_callable=AsyncMock, return_value={
        "category": "PNR_FAILURE",
        "diagnosis": "No integration top RTL for synthesis",
        "confidence": 0.2,
        "action": "continue",
        "suggested_fix": "Provide integration top-level RTL",
        "pnr_overrides": {},
    })
    async def test_no_integration_top_hits_interrupt_then_skips(
        self, mock_diag, backend_graph,
    ):
        """Walk: no integration top -> flat synth fails -> diagnose -> ask_human -> skip."""
        from langgraph.types import Command

        config = {"configurable": {"thread_id": "test-backend-happy-1"}}
        state = _initial_backend_state([_make_backend_block("fft_butterfly")])

        await backend_graph.ainvoke(state, config)

        result = await backend_graph.ainvoke(
            Command(resume={"action": "skip"}), config
        )

        assert result["backend_done"] is True
        assert len(result["completed_blocks"]) == 1
        assert result["completed_blocks"][0]["name"] == "test_chip_top"
        assert result["completed_blocks"][0]["skipped"] is True


class TestEdaTimeout:
    """`_eda_timeout` gates per-stage EDA-step ceilings on env vars so
    large-design PnR/signoff runs don't get the OpenROAD child killed
    mid-route, while small designs keep the snappy default."""

    def test_default_when_unset(self, monkeypatch):
        from orchestrator.langgraph.backend_graph import _eda_timeout
        monkeypatch.delenv("CORESMITH_PNR_TIMEOUT", raising=False)
        assert _eda_timeout("CORESMITH_PNR_TIMEOUT", 1800) == 1800

    def test_override_when_set(self, monkeypatch):
        from orchestrator.langgraph.backend_graph import _eda_timeout
        monkeypatch.setenv("CORESMITH_PNR_TIMEOUT", "5400")
        assert _eda_timeout("CORESMITH_PNR_TIMEOUT", 1800) == 5400

    def test_garbage_falls_back_to_default(self, monkeypatch):
        from orchestrator.langgraph.backend_graph import _eda_timeout
        monkeypatch.setenv("CORESMITH_PNR_TIMEOUT", "not-an-int")
        assert _eda_timeout("CORESMITH_PNR_TIMEOUT", 1800) == 1800

    def test_nonpositive_falls_back_to_default(self, monkeypatch):
        from orchestrator.langgraph.backend_graph import _eda_timeout
        monkeypatch.setenv("CORESMITH_PNR_TIMEOUT", "0")
        assert _eda_timeout("CORESMITH_PNR_TIMEOUT", 1800) == 1800


def _routed_def(tmp_path, n_components):
    """A routed DEF with `n_components` placed cells (`COMPONENTS <N> ;`)."""
    p = tmp_path / "routed.def"
    comps = "\n".join(
        f"- u{i} sky130_fd_sc_hd__inv_2 + PLACED ( {i} {i} ) N ;"
        for i in range(n_components))
    p.write_text(
        f"DESIGN d ;\nCOMPONENTS {n_components} ;\n{comps}\n"
        "END COMPONENTS\nEND DESIGN\n")
    return str(p)


class TestPnrLinkedCellCount:
    """`pnr_linked_cell_count` reads the placed-instance count from a DEF."""

    def test_reads_components_count(self, tmp_path):
        from orchestrator.langgraph.backend_graph import pnr_linked_cell_count
        assert pnr_linked_cell_count(_routed_def(tmp_path, 42)) == 42

    def test_missing_def_returns_none(self):
        from orchestrator.langgraph.backend_graph import pnr_linked_cell_count
        assert pnr_linked_cell_count("/no/such/routed.def") is None


class TestCellCountShortfall:
    """`pnr_cell_count_shortfall_error` (Fix 2): a fragment linked as the top
    (linked cells far below synth gate_count) hard-fails PnR."""

    def test_fragment_far_below_synth_fails(self, tmp_path):
        from orchestrator.langgraph.backend_graph import (
            pnr_cell_count_shortfall_error,
        )
        # gross-deficit shape: 154 linked vs 4,682 synth (~0.03) -> fail.
        err = pnr_cell_count_shortfall_error(4682, _routed_def(tmp_path, 154))
        assert err is not None
        assert "Cell-count shortfall" in err and "154" in err and "4682" in err

    def test_matching_counts_pass(self, tmp_path):
        from orchestrator.langgraph.backend_graph import (
            pnr_cell_count_shortfall_error,
        )
        # PnR ADDS physical cells, so linked >= synth is normal -> no fail.
        assert pnr_cell_count_shortfall_error(
            200, _routed_def(tmp_path, 240)) is None

    def test_unknown_gate_count_never_false_fails(self, tmp_path):
        from orchestrator.langgraph.backend_graph import (
            pnr_cell_count_shortfall_error,
        )
        assert pnr_cell_count_shortfall_error(0, _routed_def(tmp_path, 3)) is None

    def test_unreadable_def_never_false_fails(self):
        from orchestrator.langgraph.backend_graph import (
            pnr_cell_count_shortfall_error,
        )
        assert pnr_cell_count_shortfall_error(4682, "/no/such.def") is None

    def test_ratio_boundary(self, tmp_path):
        from orchestrator.langgraph.backend_graph import (
            pnr_cell_count_shortfall_error,
        )
        # exactly at the ratio floor -> not below -> pass; just under -> fail.
        assert pnr_cell_count_shortfall_error(
            100, _routed_def(tmp_path, 50), min_ratio=0.5) is None
        assert pnr_cell_count_shortfall_error(
            100, _routed_def(tmp_path, 49), min_ratio=0.5) is not None


class TestCellCountGuardGate:
    """Env gate for the cell-count guard, both branches (default ON)."""

    def test_flag_both_branches(self, monkeypatch):
        from orchestrator.langgraph.sram_wrapper import (
            pnr_cellcount_guard_enabled,
        )
        monkeypatch.delenv("CORESMITH_PNR_CELLCOUNT_GUARD", raising=False)
        assert pnr_cellcount_guard_enabled() is True            # default ON
        monkeypatch.setenv("CORESMITH_PNR_CELLCOUNT_GUARD", "0")
        assert pnr_cellcount_guard_enabled() is False           # pre-fix restored

    def test_min_ratio_default_and_override(self, monkeypatch):
        from orchestrator.langgraph.sram_wrapper import pnr_cellcount_min_ratio
        monkeypatch.delenv("CORESMITH_PNR_CELLCOUNT_MIN_RATIO", raising=False)
        assert pnr_cellcount_min_ratio() == 0.5
        monkeypatch.setenv("CORESMITH_PNR_CELLCOUNT_MIN_RATIO", "0.7")
        assert pnr_cellcount_min_ratio() == 0.7
        # out-of-band overrides are clamped to (0, 1].
        monkeypatch.setenv("CORESMITH_PNR_CELLCOUNT_MIN_RATIO", "9")
        assert pnr_cellcount_min_ratio() == 1.0


# ---------------------------------------------------------------------------
# Brace-safe prompt templating (_safe_format)
# ---------------------------------------------------------------------------

# Prompts loaded by `_run_llm_eda_step` through the brace-safe formatter.
_EDA_PROMPTS = [
    "backend_synth_llm.md",
    "backend_pnr_llm.md",
    "backend_drc_llm.md",
    "backend_lvs_llm.md",
    "backend_wrapper_llm.md",
]


class TestSafeFormat:
    """Brace-safe prompt templating: a prompt may embed Verilog/tcl braces AND
    `{placeholder}` fields; only recognized placeholders are substituted, all
    other braces stay literal, and no `str.format`-style crash occurs."""

    def test_verilog_braces_preserved_with_placeholder(self):
        # This is the exact regression: an unescaped Verilog concat/replication
        # in the prompt made str.format raise KeyError and crash the EDA node.
        tmpl = (
            "Design: {design_name}\n"
            "assign io_out = {31'b0, done, qspi_o, 2'b0};\n"
            "assign io_oeb = {31'b1, {4{oe}}, 2'b1};\n"
            "tcl braces {a b c} stay literal\n"
        )
        # str.format is the thing that breaks:
        with pytest.raises((KeyError, ValueError)):
            tmpl.format(design_name="widget_top")
        out = _safe_format(tmpl, {"design_name": "widget_top"})
        assert "Design: widget_top" in out                 # placeholder filled
        assert "{31'b0, done, qspi_o, 2'b0}" in out         # concat literal
        assert "{31'b1, {4{oe}}, 2'b1}" in out              # replication literal
        assert "{a b c}" in out                             # tcl braces literal

    def test_unknown_placeholder_left_literal_not_crash(self):
        # A `{name}` whose name is not a context key is passed through unchanged
        # (missing-key safe -- never raises).
        out = _safe_format("keep {unknown} here, fill {known}", {"known": "X"})
        assert out == "keep {unknown} here, fill X"

    def test_double_brace_unescaped_like_str_format(self):
        # `{{`/`}}` unescape to single braces exactly like str.format, and an
        # escaped `{{name}}` is NOT substituted.
        tmpl = 'json {{"k": "{v}"}} and {{name}}'
        assert _safe_format(tmpl, {"v": "1", "name": "N"}) == tmpl.format(v="1", name="N")
        assert _safe_format(tmpl, {"v": "1", "name": "N"}) == 'json {"k": "1"} and {name}'

    def test_format_spec_and_conversion_applied(self):
        assert _safe_format("{p:.2f}", {"p": 12.3456}) == "12.35"
        assert _safe_format("{x!r}", {"x": "hi"}) == "'hi'"

    def test_all_backend_prompts_render_identically_to_str_format(self):
        # AUDIT guard: every backend prompt must render byte-identically under
        # _safe_format and str.format for its real (word-name) placeholder set.
        # Guarantees the swap changed no rendering, incl. the `{{ }}` JSON blocks.
        import re
        field_re = re.compile(r"\{(\w+)(?:![rsa])?(?::[^{}]*)?\}")
        for name in _EDA_PROMPTS:
            text = (_Path(_PROMPT_DIR) / name).read_text()
            ctx = {}
            for f in set(field_re.findall(text)):
                # numeric fields may carry :.Nf specs -> give them a float
                ctx[f] = 12.3456 if f.endswith(("_ns", "_mhz")) else f"<{f}>"
            assert _safe_format(text, ctx) == text.format(**ctx), name

    def test_lvs_prompt_no_longer_crashes_str_format(self):
        # Belt-and-suspenders: the LVS example braces are escaped, so even a
        # plain str.format renders (and matches _safe_format).
        text = (_Path(_PROMPT_DIR) / "backend_lvs_llm.md").read_text()
        ctx = {
            "design_name": "d", "netgen_setup": "s", "netgen_bin": "b",
            "spice_path": "sp", "pwr_verilog_path": "pv", "output_dir": "od",
            "attempt": 1, "prior_failure": "None", "constraints": "c",
            "result_json_path": "rj",
        }
        rendered = _safe_format(text, ctx)
        assert rendered == text.format(**ctx)
        # ... and the rendered example is natural single-brace Verilog.
        assert "assign io_out = {31'b0, done, qspi_o, 2'b0};" in rendered
        assert "{4{oe}}, 2'b1};" in rendered
