# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the branch-parity smoke (rung3 split-brain backstop).

The smoke rebuilds a block under the synth-side macro world (-DSYNTHESIS ...)
and reruns the SAME seeded vectors, then compares the verdict to the default
build. Divergence = the "allowed" conditional region actually changed hardware
-> fail closed. The live cocotb double-build is exercised at the LOGIC level here
via an injected ``sim_runner`` (this box has no cocotb-grade verilator; the
mechanism + fail semantics are what these tests pin).
"""
from __future__ import annotations

import os

import pytest

from orchestrator.harness import branch_parity as bp


_BLOCK = {"name": "blk"}
_RTL_WITH_REGION = (
    "module blk(input clk, output reg [7:0] q);\n"
    "`ifdef COCOTB_SIM\n  initial $dumpvars(0, blk);\n`endif\n"
    "  always @(posedge clk) q <= 8'd1;\nendmodule\n"
)
_RTL_NO_REGION = "module blk(input clk, output reg [7:0] q);\n always @(posedge clk) q <= 8'd1;\nendmodule\n"


def _runner(base_res, synth_res, calls=None):
    """Build an injected sim_runner returning ``base_res`` for the default
    (extra_defines falsy) build and ``synth_res`` for the -DSYNTHESIS build."""
    def run(block, rtl_path, tb_path, attempt=1, extra_defines=None, sim_subdir=None):
        if calls is not None:
            calls.append({"extra_defines": extra_defines, "sim_subdir": sim_subdir,
                          "seed": os.environ.get("CORESMITH_DV_SEED_PIN")})
        return dict(synth_res) if extra_defines else dict(base_res)
    return run


# --------------------------------------------------------------------------- #
# gating
# --------------------------------------------------------------------------- #
def test_has_conditional_region():
    assert bp.has_conditional_region(_RTL_WITH_REGION) is True
    assert bp.has_conditional_region(_RTL_NO_REGION) is False


def test_gate_default_on_only_with_region(monkeypatch):
    monkeypatch.delenv("CORESMITH_BRANCH_PARITY", raising=False)
    assert bp.branch_parity_enabled(_RTL_WITH_REGION) is True
    assert bp.branch_parity_enabled(_RTL_NO_REGION) is False


def test_gate_force_off(monkeypatch):
    monkeypatch.setenv("CORESMITH_BRANCH_PARITY", "0")
    assert bp.branch_parity_enabled(_RTL_WITH_REGION) is False


def test_gate_force_on_even_without_region(monkeypatch):
    monkeypatch.setenv("CORESMITH_BRANCH_PARITY", "1")
    assert bp.branch_parity_enabled(_RTL_NO_REGION) is True


def test_no_region_does_not_run(monkeypatch):
    monkeypatch.delenv("CORESMITH_BRANCH_PARITY", raising=False)
    res = bp.check_branch_parity(_BLOCK, "x.v", "tb.py", rtl_text=_RTL_NO_REGION,
                                 sim_runner=_runner({}, {}))
    assert res.ran is False and res.ok is True


def test_gate_off_does_not_run(monkeypatch):
    monkeypatch.setenv("CORESMITH_BRANCH_PARITY", "0")
    res = bp.check_branch_parity(_BLOCK, "x.v", "tb.py", rtl_text=_RTL_WITH_REGION,
                                 sim_runner=_runner({}, {}))
    assert res.ran is False


# --------------------------------------------------------------------------- #
# verdict comparison
# --------------------------------------------------------------------------- #
_PASS = {"passed": True, "tests_passed": 5, "tests_total": 5, "tests_failed": 0}


def test_agreeing_verdicts_pass(monkeypatch):
    monkeypatch.delenv("CORESMITH_BRANCH_PARITY", raising=False)
    res = bp.check_branch_parity(_BLOCK, "x.v", "tb.py", rtl_text=_RTL_WITH_REGION,
                                 sim_runner=_runner(_PASS, _PASS))
    assert res.ran and res.ok and not res.skipped


def test_diverging_verdicts_fail_closed(monkeypatch):
    monkeypatch.delenv("CORESMITH_BRANCH_PARITY", raising=False)
    # default build PASSES the vectors, synth build FAILS them (both ran the
    # vectors -> comparable) -> split-brain -> fail closed.
    synth_fail = {"passed": False, "tests_passed": 2, "tests_total": 5, "tests_failed": 3}
    res = bp.check_branch_parity(_BLOCK, "x.v", "tb.py", rtl_text=_RTL_WITH_REGION,
                                 sim_runner=_runner(_PASS, synth_fail))
    assert res.ran and res.ok is False and not res.skipped
    msg = res.as_prev_error("blk")
    assert "DIVERGENCE" in msg and "ONE implementation" in msg


def test_different_test_counts_fail_closed(monkeypatch):
    monkeypatch.delenv("CORESMITH_BRANCH_PARITY", raising=False)
    other = {"passed": True, "tests_passed": 4, "tests_total": 5, "tests_failed": 1}
    res = bp.check_branch_parity(_BLOCK, "x.v", "tb.py", rtl_text=_RTL_WITH_REGION,
                                 sim_runner=_runner(_PASS, other))
    assert res.ran and res.ok is False


# --------------------------------------------------------------------------- #
# build-error -> skip (never a false fail)
# --------------------------------------------------------------------------- #
def test_synth_build_compile_error_skips(monkeypatch):
    monkeypatch.delenv("CORESMITH_BRANCH_PARITY", raising=False)
    compile_err = {"passed": False, "tests_passed": 0, "tests_total": 0,
                   "tests_failed": 0, "log": "%Error: Cannot find module foo"}
    res = bp.check_branch_parity(_BLOCK, "x.v", "tb.py", rtl_text=_RTL_WITH_REGION,
                                 sim_runner=_runner(_PASS, compile_err))
    assert res.ran and res.ok is True and res.skipped is True


def test_timeout_build_skips(monkeypatch):
    monkeypatch.delenv("CORESMITH_BRANCH_PARITY", raising=False)
    to = {"passed": False, "sim_timed_out": True, "log": "SIM_TIMEOUT"}
    res = bp.check_branch_parity(_BLOCK, "x.v", "tb.py", rtl_text=_RTL_WITH_REGION,
                                 sim_runner=_runner(_PASS, to))
    assert res.ran and res.ok is True and res.skipped is True


def test_runner_exception_skips(monkeypatch):
    monkeypatch.delenv("CORESMITH_BRANCH_PARITY", raising=False)
    def boom(*a, **k):
        raise RuntimeError("verilator missing")
    res = bp.check_branch_parity(_BLOCK, "x.v", "tb.py", rtl_text=_RTL_WITH_REGION,
                                 sim_runner=boom)
    assert res.ran and res.ok is True and res.skipped is True


# --------------------------------------------------------------------------- #
# build invocation contract: same pinned seed, synth defines, distinct dirs
# --------------------------------------------------------------------------- #
def test_both_builds_use_same_seed_and_synth_defines(monkeypatch):
    monkeypatch.delenv("CORESMITH_BRANCH_PARITY", raising=False)
    monkeypatch.delenv("CORESMITH_DV_SEED_PIN", raising=False)
    calls = []
    bp.check_branch_parity(_BLOCK, "x.v", "tb.py", rtl_text=_RTL_WITH_REGION,
                           sim_runner=_runner(_PASS, _PASS, calls=calls))
    assert len(calls) == 2
    seeds = {c["seed"] for c in calls}
    assert len(seeds) == 1 and next(iter(seeds))  # identical, non-empty seed
    base_call = next(c for c in calls if not c["extra_defines"])
    synth_call = next(c for c in calls if c["extra_defines"])
    assert synth_call["extra_defines"] == bp.SYNTH_PARITY_DEFINES
    assert "SYNTHESIS" in synth_call["extra_defines"]
    assert base_call["sim_subdir"] != synth_call["sim_subdir"]  # isolated builds
    # the caller's environment is restored afterward (no pin leaked)
    assert os.environ.get("CORESMITH_DV_SEED_PIN") is None


def test_caller_pinned_seed_is_honored_and_restored(monkeypatch):
    monkeypatch.delenv("CORESMITH_BRANCH_PARITY", raising=False)
    monkeypatch.setenv("CORESMITH_DV_SEED_PIN", "424242")
    calls = []
    bp.check_branch_parity(_BLOCK, "x.v", "tb.py", rtl_text=_RTL_WITH_REGION,
                           sim_runner=_runner(_PASS, _PASS, calls=calls))
    assert all(c["seed"] == "424242" for c in calls)
    assert os.environ.get("CORESMITH_DV_SEED_PIN") == "424242"  # restored


# --------------------------------------------------------------------------- #
# run_simulation seam: default path (no defines/subdir) is byte-identical
# --------------------------------------------------------------------------- #
def test_run_simulation_extra_defines_makefile_line(monkeypatch, tmp_path):
    """The new params only ADD Verilator ``-D`` Makefile lines + change the build
    dir; with both omitted the Makefile is byte-identical to before (default DV
    path stays safe). We intercept at Popen and read the on-disk Makefile so no
    verilator is invoked."""
    import orchestrator.langgraph.pipeline_helpers as ph

    monkeypatch.setattr(ph, "PROJECT_ROOT", tmp_path)
    rtl = tmp_path / "blk.v"
    rtl.write_text(_RTL_NO_REGION)
    tb = tmp_path / "test_blk.py"
    tb.write_text("# tb\n")

    monkeypatch.setattr(ph, "apply_build_fingerprint", lambda *a, **k: None)
    monkeypatch.setattr(ph, "_normalize_cocotb_timing_keywords", lambda *a, **k: None)
    monkeypatch.setattr(ph, "create_golden_model_wrapper", lambda *a, **k: None)

    captured = {}

    class _Stop(Exception):
        pass

    def fake_popen(cmd, *a, **k):
        sim_dir = cmd[2]                      # [make, "-C", sim_dir]
        captured["makefile"] = open(os.path.join(sim_dir, "Makefile")).read()
        captured["sim_dir"] = sim_dir
        raise _Stop()

    monkeypatch.setattr(ph.subprocess, "Popen", fake_popen)

    with pytest.raises(_Stop):
        ph.run_simulation(_BLOCK, str(rtl), str(tb))
    default_mk = captured["makefile"]
    default_dir = captured["sim_dir"]
    assert "-D" not in default_mk                 # no define lines by default
    assert default_dir.endswith(os.path.join("sim_build", "blk"))

    with pytest.raises(_Stop):
        ph.run_simulation(_BLOCK, str(rtl), str(tb),
                          extra_defines=["SYNTHESIS", "CORESMITH_SRAM_SYNTH"],
                          sim_subdir="blk__parity_synth")
    synth_mk = captured["makefile"]
    assert captured["sim_dir"].endswith("blk__parity_synth")   # isolated dir
    assert "EXTRA_ARGS += -DSYNTHESIS" in synth_mk
    assert "EXTRA_ARGS += -DCORESMITH_SRAM_SYNTH" in synth_mk
    # the ONLY difference is the two added define lines
    assert synth_mk.replace("EXTRA_ARGS += -DSYNTHESIS\n", "").replace(
        "EXTRA_ARGS += -DCORESMITH_SRAM_SYNTH\n", "") == default_mk
