# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Sim-build trace + fingerprint fixes (rung1).

Bug 1: the integration/validation sim's PASS verdict HARD-REQUIRES a WaveKit VCD
audit (fail-closed on a missing/empty VCD), so its Makefile must ALWAYS build with
tracing -- it must NOT depend on CORESMITH_SIM_TRACE the way per-block sims may.

Bug 2: Verilator bakes compile-time flags (VM_TRACE) into the Vtop binary, and
cocotb's make won't rebuild on a flag-only change. `apply_build_fingerprint` must
clear the stale build when the compile inputs change, and reuse it otherwise.
"""
from __future__ import annotations

import subprocess

import pytest

from orchestrator.langgraph import integration_helpers as ih
from orchestrator.langgraph.pipeline_helpers import apply_build_fingerprint


# ---------------------------------------------------------------------------
# Bug 1 -- integration sim Makefile always traces, regardless of the env flag
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("trace_env", [None, "0", "false", "1", "yes"])
def test_integration_makefile_always_has_trace_flags(tmp_path, monkeypatch, trace_env):
    monkeypatch.setattr(ih, "PROJECT_ROOT", tmp_path)
    if trace_env is None:
        monkeypatch.delenv("CORESMITH_SIM_TRACE", raising=False)
    else:
        monkeypatch.setenv("CORESMITH_SIM_TRACE", trace_env)
    # Even the old "force off" knob must not disable tracing on THIS path now.
    monkeypatch.setenv("CORESMITH_SIM_NO_TRACE", "1")

    top = tmp_path / "chip_top.v"
    top.write_text("module chip_top(input clk);\nendmodule\n")
    tb = tmp_path / "test_chip_top.py"
    tb.write_text("import cocotb\n\n@cocotb.test()\nasync def t(dut):\n    pass\n")

    def _fake_run(cmd, *a, **k):
        # The Makefile is already written by the time make is invoked.
        return subprocess.CompletedProcess(cmd, 0, "** TESTS=1 PASS=1 FAIL=0 **", "")

    monkeypatch.setattr(ih.subprocess, "run", _fake_run)
    monkeypatch.setattr(ih, "run_wavekit_vcd_audit", lambda *a, **k: {"ok": True})

    ih.run_integration_simulation("chip", str(top), {}, str(tb))

    makefile = (tmp_path / "sim_build" / "integration" / "Makefile").read_text()
    assert "WAVES = 1" in makefile
    assert "--trace --trace-structs" in makefile


def test_integration_makefile_honors_declared_design_top(tmp_path, monkeypatch):
    """A multi-wrapper file must simulate the explicitly selected graded top."""
    monkeypatch.setattr(ih, "PROJECT_ROOT", tmp_path)
    top = tmp_path / "reference_codec_dec_top.v"
    top.write_text(
        "module reference_codec_dec_top(input x); endmodule\n"
        "module user_project_wrapper(input wb_clk_i); endmodule\n"
    )
    tb = tmp_path / "test_top.py"
    tb.write_text("import cocotb\n")

    monkeypatch.setattr(
        ih.subprocess,
        "run",
        lambda cmd, *a, **k: subprocess.CompletedProcess(
            cmd, 0, "** TESTS=1 PASS=1 FAIL=0 **", ""
        ),
    )
    monkeypatch.setattr(
        ih, "run_wavekit_vcd_audit", lambda *a, **k: {"ok": True}
    )

    ih.run_integration_simulation(
        "user_project_wrapper", str(top), {}, str(tb)
    )
    makefile = (tmp_path / "sim_build" / "integration" / "Makefile").read_text()
    assert "TOPLEVEL = user_project_wrapper" in makefile


# ---------------------------------------------------------------------------
# Bug 2 -- build-input fingerprint clears a stale build on change, reuses else
# ---------------------------------------------------------------------------
def _seed_build(sim_dir):
    """Create a fake prior Verilator/cocotb build tree in sim_dir."""
    obj = sim_dir / "sim_build"
    obj.mkdir(parents=True)
    (obj / "Vtop").write_text("stale-binary")
    (sim_dir / "dump.vcd").write_text("$date stale $end")
    (sim_dir / "results.xml").write_text("<results/>")


class TestBuildFingerprint:
    def test_first_call_no_products_records_fp_no_clear(self, tmp_path):
        # No pre-existing build -> the first fingerprint is a cheap no-op wipe.
        cleared = apply_build_fingerprint(tmp_path, "MAKEFILE-A", ["/x.v"])
        assert cleared is False
        assert (tmp_path / ".build_fingerprint").exists()
        assert not (tmp_path / "sim_build").exists()

    def test_first_call_with_products_wipes(self, tmp_path):
        # rung1-fixes-4 Defect A: a MISSING fingerprint next to PRE-EXISTING
        # build products (an unfingerprinted build laid down by e.g. an agent's
        # in-context verify) is UNTRUSTED -> wipe before building, do not reuse.
        _seed_build(tmp_path)
        cleared = apply_build_fingerprint(tmp_path, "MAKEFILE-A", ["/x.v"])
        assert cleared is True
        assert not (tmp_path / "sim_build").exists()       # stale obj dir wiped
        assert not (tmp_path / "dump.vcd").exists()
        assert not (tmp_path / "results.xml").exists()
        assert (tmp_path / ".build_fingerprint").exists()  # fp now recorded

    def test_first_call_results_xml_only_wipes(self, tmp_path):
        # results.xml alone (no obj dir) still counts as a prior build.
        (tmp_path / "results.xml").write_text("<results/>")
        cleared = apply_build_fingerprint(tmp_path, "MAKEFILE-A", ["/x.v"])
        assert cleared is True
        assert not (tmp_path / "results.xml").exists()

    def test_first_call_config_inputs_are_not_products(self, tmp_path):
        # Makefile / copied TB / flock are config, NOT build products -> no wipe.
        (tmp_path / "Makefile").write_text("SIM = verilator")
        (tmp_path / "test_chip_top.py").write_text("# tb")
        (tmp_path / ".lock").write_text("")
        cleared = apply_build_fingerprint(tmp_path, "MAKEFILE-A", ["/x.v"])
        assert cleared is False
        assert (tmp_path / "Makefile").exists()            # config untouched
        assert (tmp_path / "test_chip_top.py").exists()

    def test_unchanged_inputs_reuse_build(self, tmp_path):
        apply_build_fingerprint(tmp_path, "MAKEFILE-A", ["/x.v"])
        _seed_build(tmp_path)
        cleared = apply_build_fingerprint(tmp_path, "MAKEFILE-A", ["/x.v"])
        assert cleared is False
        assert (tmp_path / "sim_build" / "Vtop").exists()
        assert (tmp_path / "dump.vcd").exists()

    def test_makefile_change_clears_build(self, tmp_path):
        apply_build_fingerprint(tmp_path, "WAVES = 0", ["/x.v"])
        _seed_build(tmp_path)
        cleared = apply_build_fingerprint(tmp_path, "WAVES = 1", ["/x.v"])
        assert cleared is True
        assert not (tmp_path / "sim_build").exists()   # stale obj dir wiped
        assert not (tmp_path / "dump.vcd").exists()    # stale VCD wiped
        assert not (tmp_path / "results.xml").exists()
        # fingerprint updated to the new inputs
        assert (tmp_path / ".build_fingerprint").read_text()

    def test_source_content_change_clears_build(self, tmp_path):
        src = tmp_path / "block.v"
        src.write_text("module a; endmodule")
        apply_build_fingerprint(tmp_path, "MAKEFILE-A", [str(src)])
        _seed_build(tmp_path)
        src.write_text("module a; wire changed; endmodule")   # RTL edited
        cleared = apply_build_fingerprint(tmp_path, "MAKEFILE-A", [str(src)])
        assert cleared is True
        assert not (tmp_path / "sim_build").exists()

    def test_reused_after_reclear(self, tmp_path):
        # A -> B (clear) -> B (reuse): the fingerprint tracks the LATEST inputs.
        apply_build_fingerprint(tmp_path, "A", [])
        apply_build_fingerprint(tmp_path, "B", [])
        _seed_build(tmp_path)
        cleared = apply_build_fingerprint(tmp_path, "B", [])
        assert cleared is False
        assert (tmp_path / "sim_build" / "Vtop").exists()


# ---------------------------------------------------------------------------
# Defect A (rung1-fixes-4) -- integration pre-run hygiene clears agent debris
# ---------------------------------------------------------------------------
def test_integration_pre_run_hygiene_clears_agent_debris(tmp_path, monkeypatch):
    """run_integration_simulation must WIPE any pre-existing build products in the
    authoritative sim dir before writing its traced Makefile, so an agent's
    stale/traceless Vtop can never be reused by the gate's cocotb make."""
    monkeypatch.setattr(ih, "PROJECT_ROOT", tmp_path)
    top = tmp_path / "chip_top.v"
    top.write_text("module chip_top(input clk);\nendmodule\n")
    tb = tmp_path / "test_chip_top.py"
    tb.write_text("import cocotb\n")

    # Seed an untrusted prior build (as an agent's in-context verify would leave).
    sim_dir = tmp_path / "sim_build" / "integration"
    _seed_build(sim_dir)
    stale_obj = sim_dir / "sim_build" / "Vtop"
    assert stale_obj.exists()

    seen = {}

    def _fake_run(cmd, *a, **k):
        # Capture whether the stale binary survived until make was invoked.
        seen["stale_obj_present_at_make"] = stale_obj.exists()
        return subprocess.CompletedProcess(cmd, 0, "** TESTS=1 PASS=1 FAIL=0 **", "")

    monkeypatch.setattr(ih.subprocess, "run", _fake_run)
    monkeypatch.setattr(ih, "run_wavekit_vcd_audit", lambda *a, **k: {"ok": True})

    ih.run_integration_simulation("chip", str(top), {}, str(tb))

    # The stale obj dir/binary + outputs must have been wiped before make ran.
    assert seen["stale_obj_present_at_make"] is False
    assert not stale_obj.exists()
    assert not (sim_dir / "dump.vcd").exists()
    assert not (sim_dir / "results.xml").exists()


def test_agent_chip_verify_uses_scratch_namespace(tmp_path, monkeypatch):
    """verify_chip(record_source="agent") must build in sim_build/agent_integration,
    never the gate-authoritative sim_build/integration."""
    import json as _json
    from orchestrator.harness import verify as V

    (tmp_path / ".coresmith").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".coresmith" / "integration_result.json").write_text(_json.dumps({
        "design_name": "codec", "top_rtl_path": str(tmp_path / "chip_top.v"),
        "block_rtl_paths": {}, "tb_path": str(tmp_path / "tb.py"),
    }))
    (tmp_path / "chip_top.v").write_text("module chip_top(); endmodule")
    (tmp_path / "tb.py").write_text("# tb")

    captured = {}

    def _fake_sim(design, top, blocks, tbp, attempt=1, sim_scope="integration"):
        captured["sim_scope"] = sim_scope
        return {"passed": True, "log": "ok", "log_path": "/i.log"}

    import orchestrator.langgraph.integration_helpers as _ih
    monkeypatch.setattr(_ih, "run_integration_simulation", _fake_sim)

    V.verify_chip(tmp_path, record_source="agent")
    assert captured["sim_scope"] == "agent_integration"
    # gate namespace untouched; agent scratch lock dir created instead
    assert (tmp_path / "sim_build" / "agent_integration").is_dir()
    assert not (tmp_path / "sim_build" / "integration").exists()

    V.verify_chip(tmp_path, record_source="gate")
    assert captured["sim_scope"] == "integration"
