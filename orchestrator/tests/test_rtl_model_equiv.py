"""Tests for the deterministic RTL<->model byte-exact equivalence checker.

These exercise verilator FOR REAL (no mocks): a tiny self-contained AXI-Stream
block is Verilated and driven by the engine harness, and its output is byte-
compared to a Python reference. The marquee test is WRONG-RTL-FAILS: a "+2"
RTL against a "+1" reference must FAIL with a divergence reason -- that is the
proof this catches what an LLM-authored TB can be told to skip.
"""

import shutil
import textwrap
from pathlib import Path

import pytest

from orchestrator.langgraph.rtl_model_equiv import (
    check_rtl_model_equivalence,
    rtl_model_equiv_enabled,
)

_no_verilator = shutil.which("verilator") is None


def requires_verilator(fn):
    """Heavy: exercises verilator for real. Carries ``requires_nix`` so it is
    EXCLUDED from the default fast run (``-m "not requires_nix"``) -- a stray
    ``pytest`` must never relaunch a Verilator build -- and skips when verilator
    is absent. Run explicitly with ``-m requires_nix`` on a box that can take it.
    """
    fn = pytest.mark.requires_nix(fn)
    fn = pytest.mark.skipif(_no_verilator, reason="verilator not on PATH")(fn)
    return fn


def _skip_if_unbuildable(res):
    """Skip when the harness couldn't build/run (e.g. an unsupported verilator
    version -- cocotb needs >=5.036). The divergence-detection logic must then
    be verified on a box with a working verilator+cocotb, not false-failed here.
    """
    r = (res.get("reason") or "").lower()
    if res.get("skipped") and any(
        k in r for k in ("cannot judge", "build", "verilator", "cocotb", "no tests")
    ):
        pytest.skip(f"verilator/cocotb cannot build here: {res.get('reason', '')[:140]}")


# ---------------------------------------------------------------------------
# Flag default-on / off
# ---------------------------------------------------------------------------

def test_flag_default_on(monkeypatch):
    monkeypatch.delenv("CORESMITH_RTL_MODEL_EQUIV", raising=False)
    assert rtl_model_equiv_enabled() is True


def test_flag_explicit_off(monkeypatch):
    for v in ("0", "false", "off", "no", ""):
        monkeypatch.setenv("CORESMITH_RTL_MODEL_EQUIV", v)
        assert rtl_model_equiv_enabled() is False


def test_flag_explicit_on(monkeypatch):
    for v in ("1", "true", "yes", "on"):
        monkeypatch.setenv("CORESMITH_RTL_MODEL_EQUIV", v)
        assert rtl_model_equiv_enabled() is True


# ---------------------------------------------------------------------------
# Verilog fixtures: an 8-bit AXI-Stream pass-through that adds a constant.
# Single s_axis in / single m_axis out, 1-cycle latency, full handshake.
# ---------------------------------------------------------------------------

def _axis_addk_rtl(module: str, k: int) -> str:
    return textwrap.dedent(f"""
    module {module} (
        input  wire        clk,
        input  wire        rst_n,
        input  wire [7:0]  s_axis_tdata,
        input  wire        s_axis_tvalid,
        output wire        s_axis_tready,
        input  wire        s_axis_tlast,
        output reg  [7:0]  m_axis_tdata,
        output reg         m_axis_tvalid,
        input  wire        m_axis_tready,
        output reg         m_axis_tlast
    );
        // Always ready to accept input when the output side is free.
        assign s_axis_tready = (~m_axis_tvalid) | m_axis_tready;

        always @(posedge clk) begin
            if (!rst_n) begin
                m_axis_tdata  <= 8'd0;
                m_axis_tvalid <= 1'b0;
                m_axis_tlast  <= 1'b0;
            end else begin
                if (s_axis_tvalid && s_axis_tready) begin
                    m_axis_tdata  <= s_axis_tdata + 8'd{k};
                    m_axis_tvalid <= 1'b1;
                    m_axis_tlast  <= s_axis_tlast;
                end else if (m_axis_tvalid && m_axis_tready) begin
                    m_axis_tvalid <= 1'b0;
                    m_axis_tlast  <= 1'b0;
                end
            end
        end
    endmodule
    """)


def _addk_model_py(k: int) -> str:
    # The "block model": a deterministic pure-Python reference. The resolver
    # picks up the generic name ``process``.
    return textwrap.dedent(f"""
    def process(stream):
        return [(int(x) + {k}) & 0xFF for x in stream]
    """)


def _running_sum_rtl(module: str) -> str:
    return textwrap.dedent(f"""
    module {module} (
        input  wire        clk,
        input  wire        rst_n,
        input  wire [7:0]  s_axis_tdata,
        input  wire        s_axis_tvalid,
        output wire        s_axis_tready,
        input  wire        s_axis_tlast,
        output reg  [7:0]  m_axis_tdata,
        output reg         m_axis_tvalid,
        input  wire        m_axis_tready,
        output reg         m_axis_tlast
    );
        reg [7:0] acc;
        assign s_axis_tready = (~m_axis_tvalid) | m_axis_tready;
        always @(posedge clk) begin
            if (!rst_n) begin
                acc <= 8'd0;
                m_axis_tdata <= 8'd0;
                m_axis_tvalid <= 1'b0;
                m_axis_tlast <= 1'b0;
            end else begin
                if (s_axis_tvalid && s_axis_tready) begin
                    acc <= acc + s_axis_tdata;
                    m_axis_tdata <= acc + s_axis_tdata;
                    m_axis_tvalid <= 1'b1;
                    m_axis_tlast <= s_axis_tlast;
                end else if (m_axis_tvalid && m_axis_tready) begin
                    m_axis_tvalid <= 1'b0;
                    m_axis_tlast <= 1'b0;
                end
            end
        end
    endmodule
    """)


def _running_sum_model_py() -> str:
    return textwrap.dedent("""
    def process(stream):
        out = []
        acc = 0
        for x in stream:
            acc = (acc + int(x)) & 0xFF
            out.append(acc)
        return out
    """)


def _write_pair(tmp_path: Path, module: str, rtl: str, model: str):
    rtl_path = tmp_path / f"{module}.v"
    rtl_path.write_text(rtl)
    model_path = tmp_path / f"{module}_model.py"
    model_path.write_text(model)
    return rtl_path, model_path


# ---------------------------------------------------------------------------
# CORRECT RTL passes
# ---------------------------------------------------------------------------

@requires_verilator
def test_correct_addk_passes(tmp_path):
    module = "rme_addone"
    rtl_path, model_path = _write_pair(
        tmp_path, module, _axis_addk_rtl(module, 1), _addk_model_py(1)
    )
    res = check_rtl_model_equivalence(
        module, str(rtl_path), str(model_path),
        project_root=tmp_path, seed=20260624, n_vectors=24, max_cycles=5000,
    )
    _skip_if_unbuildable(res)
    assert res["skipped"] is False, res
    assert res["passed"] is True, res
    assert res["checked_vectors"] == 24, res


@requires_verilator
def test_running_sum_passes(tmp_path):
    module = "rme_runsum"
    rtl_path, model_path = _write_pair(
        tmp_path, module, _running_sum_rtl(module), _running_sum_model_py()
    )
    res = check_rtl_model_equivalence(
        module, str(rtl_path), str(model_path),
        project_root=tmp_path, seed=777, n_vectors=20, max_cycles=5000,
    )
    _skip_if_unbuildable(res)
    assert res["passed"] is True, res
    assert res["skipped"] is False, res


# ---------------------------------------------------------------------------
# WRONG RTL fails -- the key proof.
# ---------------------------------------------------------------------------

@requires_verilator
def test_wrong_addk_fails_with_divergence(tmp_path):
    module = "rme_addone"
    # RTL adds 2, but the model/reference adds 1.
    rtl_path = tmp_path / f"{module}.v"
    rtl_path.write_text(_axis_addk_rtl(module, 2))
    model_path = tmp_path / f"{module}_model.py"
    model_path.write_text(_addk_model_py(1))

    res = check_rtl_model_equivalence(
        module, str(rtl_path), str(model_path),
        project_root=tmp_path, seed=20260624, n_vectors=24, max_cycles=5000,
    )
    _skip_if_unbuildable(res)
    assert res["passed"] is False, res
    assert res["skipped"] is False, res
    # Reason must name a divergence offset.
    assert "DIVERGENCE" in res["reason"] or "mismatch" in res["reason"], res
    assert "offset" in res["reason"] or "vector/offset" in res["reason"], res


# ---------------------------------------------------------------------------
# Honest SKIPs (never a false pass)
# ---------------------------------------------------------------------------

@requires_verilator
def test_non_axis_interface_skips(tmp_path):
    # A plain combinational block with no AXI-Stream ports.
    module = "rme_plain"
    rtl_path = tmp_path / f"{module}.v"
    rtl_path.write_text(textwrap.dedent(f"""
    module {module} (
        input  wire       clk,
        input  wire       rst_n,
        input  wire [7:0] a,
        input  wire [7:0] b,
        output reg  [7:0] y
    );
        always @(posedge clk) y <= a + b;
    endmodule
    """))
    model_path = tmp_path / f"{module}_model.py"
    model_path.write_text(_addk_model_py(1))

    res = check_rtl_model_equivalence(
        module, str(rtl_path), str(model_path),
        project_root=tmp_path, seed=1, n_vectors=8, max_cycles=2000,
    )
    assert res["skipped"] is True, res
    assert res["passed"] is False, res
    assert "AXI-Stream" in res["reason"], res


@requires_verilator
def test_unresolvable_reference_skips(tmp_path):
    # Valid AXIS block, but the model exposes NO pure-Python reference callable
    # (only constants) -> must SKIP, not pass.
    module = "rme_addone"
    rtl_path = tmp_path / f"{module}.v"
    rtl_path.write_text(_axis_addk_rtl(module, 1))
    model_path = tmp_path / f"{module}_model.py"
    model_path.write_text("DEPTH = 512\nADDR_W = 9\n")

    res = check_rtl_model_equivalence(
        module, str(rtl_path), str(model_path),
        project_root=tmp_path, seed=1, n_vectors=8, max_cycles=2000,
    )
    assert res["skipped"] is True, res
    assert res["passed"] is False, res
    assert "reference" in res["reason"].lower(), res
