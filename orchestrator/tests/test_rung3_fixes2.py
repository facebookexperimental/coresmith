# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""RUNG-3 FIXES-2 -- MAX-GEOMETRY stimulus gate (engine defect from the eval).

Defect: DV never drove MAX-GEOMETRY stimulus, so a truncated index-width bug
        shipped in a "verified" chip. Fix = MAX-GEOMETRY prompt requirement +
        a domain-generic deterministic `# MAXGEO` marker gate.

The MAX-GEOMETRY gate is DOMAIN-GENERIC: dimension names are DATA. The
`TestMaxgeoGateGenericNonVideo` class is the explicit two-domain genericity
proof (a FIFO-depth / max-burst / address-range design, no video).
"""

from __future__ import annotations

import json

import pytest

from orchestrator.langgraph import pipeline_graph


# ===========================================================================
# Helpers
# ===========================================================================
def _write_ers(root, doc: dict) -> None:
    cs = root / ".coresmith"
    cs.mkdir(parents=True, exist_ok=True)
    (cs / "ers_spec.json").write_text(json.dumps({"ers": doc}), encoding="utf-8")


def _write_tb(path, body: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return str(path)


# Two-domain declarations (dimension names are DATA, not engine vocabulary).
_VIDEO_DIMS_DOC = {
    "constraints": [
        {"name": "frame_width", "max": 640},
        {"name": "frame_height", "max": 352},
    ]
}
# NON-VIDEO: exercises different name-role keys (name/parameter/dimension) and
# different extent-role keys (depth/maximum/range) -- all generic.
_NONVIDEO_DIMS_DOC = {
    "constraints": [
        {"name": "cmd_fifo", "depth": 512},
        {"parameter": "max_burst_len", "maximum": 256},
        {"dimension": "addr_range", "range": 1024},
    ]
}
_NO_DIMS_DOC = {
    "summary": "8-bit ripple-carry adder",
    "functional_requirements": [
        {"id": "FR-1", "requirement": "sum = a + b"},
    ],
    # A 2-deep handshake is BELOW the extent floor -> not a declared dimension.
    "constraints": [{"name": "handshake_fifo", "depth": 2}],
}


# ===========================================================================
# Defect 1 (a): prompt requirement present in BOTH generators.
# ===========================================================================
class TestMaxgeoPromptRequirement:
    def _prompts(self):
        from orchestrator.langchain.agents import (
            integration_testbench_generator as itg,
        )
        from orchestrator.langchain.agents import validation_dv_generator as vdg
        return {
            "integration": itg.SYSTEM_PROMPT,
            "validation": vdg.SYSTEM_PROMPT,
        }

    def test_both_prompts_require_maxgeo_marker(self):
        for name, prompt in self._prompts().items():
            assert "MAX-GEOMETRY" in prompt, name
            assert "# MAXGEO:" in prompt, name
            assert "<dim_name>=<max_value>" in prompt, name

    def test_prompts_are_domain_generic(self):
        # Video is only an EXAMPLE; the requirement must explicitly generalize
        # to non-video dimensions and forbid assuming video. Collapse whitespace
        # so line wraps in the prompt don't break the substring checks.
        for name, prompt in self._prompts().items():
            low = " ".join(prompt.lower().split())
            assert "do not assume video" in low, name
            for kind in ("fifo depth", "burst length", "address range"):
                assert kind in low, (name, kind)

    def test_prompts_allow_sparse_at_max_dims(self):
        for name, prompt in self._prompts().items():
            assert "SPARSE" in prompt, name
            assert "2^n" in prompt, name


# ===========================================================================
# Defect 1 (b): domain-generic declared-dimension extraction.
# ===========================================================================
class TestDeclaredDimensions:
    def test_video_dims_extracted(self, tmp_path):
        _write_ers(tmp_path, _VIDEO_DIMS_DOC)
        dims = pipeline_graph._declared_dimensions(str(tmp_path))
        assert dims == {"frame_width": 640, "frame_height": 352}

    def test_nonvideo_dims_extracted_generically(self, tmp_path):
        # Different name-role AND extent-role keys, no video vocabulary at all.
        _write_ers(tmp_path, _NONVIDEO_DIMS_DOC)
        dims = pipeline_graph._declared_dimensions(str(tmp_path))
        assert dims == {
            "cmd_fifo": 512, "max_burst_len": 256, "addr_range": 1024,
        }

    def test_no_dims_returns_empty(self, tmp_path):
        _write_ers(tmp_path, _NO_DIMS_DOC)
        assert pipeline_graph._declared_dimensions(str(tmp_path)) == {}

    def test_tiny_extent_below_floor_skipped(self, tmp_path):
        _write_ers(tmp_path, {"constraints": [{"name": "x", "depth": 4}]})
        # 4 < _DIM_MIN_EXTENT (8) -> not a declared dimension.
        assert pipeline_graph._declared_dimensions(str(tmp_path)) == {}

    def test_bool_and_nonint_extent_rejected(self, tmp_path):
        _write_ers(tmp_path, {"constraints": [
            {"name": "flag", "max": True},          # bool is not an int extent
            {"name": "label", "max": "n/a"},         # non-numeric string
        ]})
        assert pipeline_graph._declared_dimensions(str(tmp_path)) == {}

    def test_missing_files_no_raise(self, tmp_path):
        assert pipeline_graph._declared_dimensions(str(tmp_path)) == {}


# ===========================================================================
# Defect 1 (b): the deterministic marker gate verdict.
# ===========================================================================
class TestMaxgeoGateVerdict:
    def test_missing_marker_when_dims_declared_is_rejected(self, tmp_path):
        _write_ers(tmp_path, _VIDEO_DIMS_DOC)
        tb = _write_tb(tmp_path / "tb" / "t.py", "import cocotb\n# no marker\n")
        v = pipeline_graph._maxgeo_gate_verdict(str(tmp_path), tb)
        assert v is not None
        assert v["uncovered_dims"] == {"frame_width": 640, "frame_height": 352}

    def test_marker_covering_all_dims_passes(self, tmp_path):
        _write_ers(tmp_path, _VIDEO_DIMS_DOC)
        tb = _write_tb(
            tmp_path / "tb" / "t.py",
            "import cocotb\n# MAXGEO: frame_width=640 frame_height=352\n",
        )
        assert pipeline_graph._maxgeo_gate_verdict(str(tmp_path), tb) is None

    def test_partial_marker_is_rejected(self, tmp_path):
        _write_ers(tmp_path, _VIDEO_DIMS_DOC)
        tb = _write_tb(
            tmp_path / "tb" / "t.py",
            "import cocotb\n# MAXGEO: frame_width=640\n",   # height missing
        )
        v = pipeline_graph._maxgeo_gate_verdict(str(tmp_path), tb)
        assert v is not None
        assert v["uncovered_dims"] == {"frame_height": 352}

    def test_no_dims_declared_is_noop(self, tmp_path):
        _write_ers(tmp_path, _NO_DIMS_DOC)
        tb = _write_tb(tmp_path / "tb" / "t.py", "import cocotb\n")
        assert pipeline_graph._maxgeo_gate_verdict(str(tmp_path), tb) is None

    def test_gate_disabled_env_is_noop(self, tmp_path, monkeypatch):
        _write_ers(tmp_path, _VIDEO_DIMS_DOC)
        tb = _write_tb(tmp_path / "tb" / "t.py", "import cocotb\n# no marker\n")
        # Enabled (default) -> rejected; disabled -> no-op. Both branches.
        assert pipeline_graph._maxgeo_gate_verdict(str(tmp_path), tb) is not None
        monkeypatch.setenv("CORESMITH_MAXGEO_GATE", "0")
        assert not pipeline_graph._maxgeo_gate_enabled()
        assert pipeline_graph._maxgeo_gate_verdict(str(tmp_path), tb) is None


class TestMaxgeoGateGenericNonVideo:
    """GENERICITY EVIDENCE: the marker requirement + deterministic check fire
    IDENTICALLY on a non-video dimension class (FIFO depth / max burst / address
    range) with zero video vocabulary anywhere in the design."""

    def test_nonvideo_missing_marker_rejected(self, tmp_path):
        _write_ers(tmp_path, _NONVIDEO_DIMS_DOC)
        tb = _write_tb(tmp_path / "tb" / "t.py", "import cocotb\n")
        v = pipeline_graph._maxgeo_gate_verdict(str(tmp_path), tb)
        assert v is not None
        assert v["uncovered_dims"] == {
            "cmd_fifo": 512, "max_burst_len": 256, "addr_range": 1024,
        }

    def test_nonvideo_marker_covering_all_passes(self, tmp_path):
        _write_ers(tmp_path, _NONVIDEO_DIMS_DOC)
        tb = _write_tb(
            tmp_path / "tb" / "t.py",
            "import cocotb\n"
            "# MAXGEO: cmd_fifo=512 max_burst_len=256 addr_range=1024\n",
        )
        assert pipeline_graph._maxgeo_gate_verdict(str(tmp_path), tb) is None

    def test_nonvideo_partial_marker_rejected(self, tmp_path):
        _write_ers(tmp_path, _NONVIDEO_DIMS_DOC)
        tb = _write_tb(
            tmp_path / "tb" / "t.py",
            "import cocotb\n# MAXGEO: cmd_fifo=512 addr_range=1024\n",
        )
        v = pipeline_graph._maxgeo_gate_verdict(str(tmp_path), tb)
        assert v is not None
        assert v["uncovered_dims"] == {"max_burst_len": 256}


# ===========================================================================
# Defect 1 (b): the gate is WIRED into the DV nodes (flips a green sim).
# ===========================================================================
def _wire_integration_node(monkeypatch, tmp_path, *, tb_body):
    top = tmp_path / "rtl" / "integration" / "chip_top.v"
    top.parent.mkdir(parents=True, exist_ok=True)
    top.write_text("module chip_top(input clk); endmodule\n")
    blk = tmp_path / "rtl" / "a.v"
    blk.write_text("module a(input clk); endmodule\n")
    tb = _write_tb(tmp_path / "tb" / "chip_tb.py", tb_body)

    from orchestrator.langgraph.integration_helpers import (
        VerilogModule, VerilogPort,
    )

    monkeypatch.setattr(
        pipeline_graph, "load_architecture_connections", lambda pr: ([], "chip"))
    monkeypatch.setattr(
        pipeline_graph, "parse_verilog_ports",
        lambda p: VerilogModule(name="a", ports=[VerilogPort("clk", "input")]))
    monkeypatch.setattr(
        pipeline_graph, "run_integration_simulation",
        lambda *a, **k: {"passed": True, "log": "sim ran", "log_path": ""})
    monkeypatch.setattr(pipeline_graph, "_maybe_run_chip_equiv",
                        lambda *a, **k: None)
    monkeypatch.setattr(pipeline_graph, "write_graph_event", lambda *a, **k: None)

    async def _gen(**_kw):
        return {"testbench_path": tb, "tb_path": tb, "test_count": 3}
    monkeypatch.setattr(pipeline_graph, "generate_integration_testbench", _gen)

    async def _fake_audit(**kw):
        return {"category": "INTEGRATION_TB_BUG",
                "outer_agent_summary": "x", "audit_path": ""}
    monkeypatch.setattr(pipeline_graph, "_run_top_level_contract_audit", _fake_audit)

    state = {
        "project_root": str(tmp_path),
        "integration_result": {
            "top_rtl_path": str(top),
            "design_name": "chip",
            "block_rtl_paths": {"a": str(blk)},
        },
    }
    return state


class TestMaxgeoNodeWiring:
    @pytest.mark.asyncio
    async def test_integration_dv_flips_to_failed_when_marker_missing(
        self, tmp_path, monkeypatch
    ):
        _write_ers(tmp_path, _VIDEO_DIMS_DOC)
        captured = {}

        def fake_interrupt(payload):
            captured.update(payload)
            return {"action": "abort"}
        monkeypatch.setattr(pipeline_graph, "interrupt", fake_interrupt)

        state = _wire_integration_node(
            monkeypatch, tmp_path, tb_body="import cocotb\n# no marker\n")
        result = await pipeline_graph.integration_dv_node(state)

        dv = result["integration_dv_result"]
        assert dv["passed"] is False
        assert captured.get("type") == "integration_dv_failure"
        assert "MAX-GEOMETRY" in captured.get("sim_log", "")

    @pytest.mark.asyncio
    async def test_integration_dv_passes_when_marker_present(
        self, tmp_path, monkeypatch
    ):
        _write_ers(tmp_path, _VIDEO_DIMS_DOC)
        monkeypatch.setattr(pipeline_graph, "interrupt",
                            lambda p: {"action": "abort"})
        state = _wire_integration_node(
            monkeypatch, tmp_path,
            tb_body="import cocotb\n# MAXGEO: frame_width=640 frame_height=352\n")
        result = await pipeline_graph.integration_dv_node(state)
        assert result["integration_dv_result"]["passed"] is True

    @pytest.mark.asyncio
    async def test_integration_dv_noop_when_no_dims(self, tmp_path, monkeypatch):
        _write_ers(tmp_path, _NO_DIMS_DOC)     # no declared dims -> gate no-ops
        monkeypatch.setattr(pipeline_graph, "interrupt",
                            lambda p: {"action": "abort"})
        state = _wire_integration_node(
            monkeypatch, tmp_path, tb_body="import cocotb\n# no marker\n")
        result = await pipeline_graph.integration_dv_node(state)
        assert result["integration_dv_result"]["passed"] is True


# ===========================================================================
# Defect 1 (c): seeded chip-equiv stream sized at max geometry.
# ===========================================================================
class TestMaxgeoEquivNvectors:
    def _wire_equiv(self, monkeypatch, tmp_path, captured):
        from orchestrator.architecture import composition as _comp
        from orchestrator.langgraph import rtl_model_equiv as _rme

        (tmp_path / "arch" / _comp.BLOCK_MODELS_DIRNAME).mkdir(
            parents=True, exist_ok=True)
        (tmp_path / "arch" / _comp.BLOCK_MODELS_DIRNAME / "_chip_model.py").write_text(
            "def simulate(s):\n    return s\n")
        monkeypatch.setattr(_comp, "block_goldens_enabled", lambda: True)
        monkeypatch.setattr(_rme, "rtl_model_equiv_enabled", lambda: True)

        def _fake_equiv(*a, **k):
            captured["n_vectors"] = k.get("n_vectors")
            return {"passed": True, "skipped": False, "reason": "ok"}
        monkeypatch.setattr(_rme, "check_chip_model_equivalence", _fake_equiv)

    def test_nvectors_scales_to_max_dim(self, tmp_path, monkeypatch):
        _write_ers(tmp_path, _VIDEO_DIMS_DOC)          # max declared = 640
        captured = {}
        self._wire_equiv(monkeypatch, tmp_path, captured)
        pipeline_graph._maybe_run_chip_equiv(
            str(tmp_path), "chip", "top.v", {"a": "a.v"})
        assert captured["n_vectors"] == 640

    def test_nvectors_default_when_no_dims(self, tmp_path, monkeypatch):
        _write_ers(tmp_path, _NO_DIMS_DOC)             # no dims -> default 64
        captured = {}
        self._wire_equiv(monkeypatch, tmp_path, captured)
        pipeline_graph._maybe_run_chip_equiv(
            str(tmp_path), "chip", "top.v", {"a": "a.v"})
        assert captured["n_vectors"] == 64



if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
