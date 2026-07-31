# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the QSPI-slave BUS-PROTOCOL conformance DV.

The conformance DV is the fix for the RECURRING frontend protocol-completeness
gap: generated ``qspi_slave_frontend`` blocks keep shipping protocol-INCOMPLETE
code (the image codec dropped cmd 0x05 read_status -> host wait_done() times out; aes-v3 /
fft-v2 launched read data a nibble early with a short/missing dummy turnaround).
It slipped through because the run's integration DV fell back to the co-tuned LLM
BFM whenever the compute-lane oracle could not be modeled, so the QSPI-slave
contract was never exercised.

Fast tests (no EDA): the ``conformance_enabled()`` env gate, and that the
generated conformance TB is valid Python containing the mandated checks (CFG
read-back through the dummy turnaround, cmd 0x05 read_status, bad-opcode
robustness, post-START 0x05 liveness), while the default integration TB stays
byte-identical.

Meta-tests (requires_nix + Icarus): run the compute-lane-INDEPENDENT conformance
TB against the real AES chip-top and assert it ACCEPTS the conformant frontend,
REJECTS a no-0x05 twin, and REJECTS a short-dummy (nibble-early read) twin.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from orchestrator.langgraph import bfm_lib
from orchestrator.langgraph.bfm_lib import (
    QSPIContract,
    classify_chip_bus,
    conformance_enabled,
    render_conformance_tb,
    render_conformance_test_fn,
    render_integration_tb,
    write_qspi_conformance_tb,
)


# ---------------------------------------------------------------------------
# Section 1: env gate (conformance_enabled)
# ---------------------------------------------------------------------------
def test_conformance_off_by_default(monkeypatch):
    monkeypatch.delenv("CORESMITH_DETERMINISTIC_BFM", raising=False)
    monkeypatch.delenv("CORESMITH_QSPI_CONFORMANCE", raising=False)
    assert conformance_enabled() is False


def test_conformance_follows_deterministic_bfm(monkeypatch):
    # ON automatically when the deterministic BFM campaign is on...
    monkeypatch.setenv("CORESMITH_DETERMINISTIC_BFM", "1")
    monkeypatch.delenv("CORESMITH_QSPI_CONFORMANCE", raising=False)
    assert conformance_enabled() is True
    # ...unless explicitly disabled.
    monkeypatch.setenv("CORESMITH_QSPI_CONFORMANCE", "0")
    assert conformance_enabled() is False


def test_conformance_forced_on_independent_of_deterministic(monkeypatch):
    monkeypatch.delenv("CORESMITH_DETERMINISTIC_BFM", raising=False)
    monkeypatch.setenv("CORESMITH_QSPI_CONFORMANCE", "1")
    assert conformance_enabled() is True


# ---------------------------------------------------------------------------
# Section 2: codegen -- the conformance TB is valid + enforces the contract
# ---------------------------------------------------------------------------
def test_render_conformance_tb_is_valid_python_with_mandated_checks():
    c = QSPIContract()
    src = render_conformance_tb(c, "chip_top")
    compile(src, "<conformance_tb>", "exec")
    # exactly one @cocotb.test() (compute-lane independent -- no golden plan)
    assert src.count("@cocotb.test()") == 1
    assert "qspi_slave_protocol_conformance" in src
    # (A) CFG read-back through the dummy turnaround
    assert "read-back" in src and "cfg1_addr" in src
    # (B) cmd 0x05 read_status + bad-opcode robustness
    assert "read_status" in src and "send_opcode_only" in src
    # (C) post-START 0x05 liveness (the missing-0x05 catch)
    assert "liveness" in src.lower() and "status_done_bit" in src
    # the inlined BFM is DUT-blind (contract-only)
    assert "QSPIMasterBFM" in src and "CONTRACT = QSPIContract(" in src


def test_cfg_probe_patterns_are_nibble_distinct_at_every_width():
    """A CFG probe byte with two IDENTICAL nibbles cannot detect a read
    serializer that launched one nibble early -- it returns the same byte either
    way. The old first probe was 0x11223344 (low byte 0x44), which cost a
    waveform session to re-derive by hand. Every byte of both probes, at every
    contract-selectable width, must have two different nibbles."""
    from orchestrator.langgraph.bfm_lib import codegen as _cg

    for width in (1, 2, 3, 4):
        assert _cg.nibble_distinct(_cg.CFG_PROBE_A, width), (width, hex(_cg.CFG_PROBE_A))
        assert _cg.nibble_distinct(_cg.CFG_PROBE_B, width), (width, hex(_cg.CFG_PROBE_B))
    # the two probes must also differ at the narrowest width, or the second
    # write/read-back round is not a second observation at all
    assert (_cg.CFG_PROBE_A & 0xFF) != (_cg.CFG_PROBE_B & 0xFF)
    # ...and the property must survive into the emitted testbench text
    src = render_conformance_tb(QSPIContract(), "chip_top")
    assert f"0x{_cg.CFG_PROBE_A:08X}" in src
    assert f"0x{_cg.CFG_PROBE_B:08X}" in src
    assert "0x11223344" not in src


def test_conformance_test_fn_op_probe_toggle():
    c = QSPIContract()
    with_probe = render_conformance_test_fn(c, start_clock=True, with_op_probe=True)
    without = render_conformance_test_fn(c, start_clock=False, with_op_probe=False)
    assert "liveness" in with_probe.lower()
    assert "liveness" not in without.lower()
    # every generated test starts its own clock (cocotb 2.x kills the
    # first test's clock coroutine between tests; relying on persistence left
    # appended conformance tests hanging on a dead clock)
    assert "Clock(dut." in with_probe
    assert "Clock(dut." in without


def test_integration_tb_include_conformance_appends_second_test():
    c = QSPIContract()
    plan = bfm_lib.StimulusPlan(
        cfg=[(c.cfg0_addr, 1, 4)], writes=[(c.in_addr, bytes(range(16)))],
        out_addr=c.out_addr, out_len=16, expected=bytes(range(16)),
        case_name="kat",
    )
    combined = render_integration_tb(c, "chip_top", plan, include_conformance=True)
    compile(combined, "<combined_tb>", "exec")
    assert combined.count("@cocotb.test()") == 2
    assert "deterministic_qspi_dv" in combined
    assert "qspi_slave_protocol_conformance" in combined
    # the appended conformance test starts ITS OWN clock: under cocotb 2.x
    # the golden test's clock coroutine is killed between tests, so relying on
    # persistence hung the second test on a dead clock
    tail = combined.split("qspi_slave_protocol_conformance", 1)[1]
    assert "Clock(dut." in tail


def test_integration_tb_default_off_is_backward_compatible():
    c = QSPIContract()
    plan = bfm_lib.StimulusPlan(
        cfg=[(c.cfg0_addr, 1, 4)], writes=[(c.in_addr, bytes(range(16)))],
        out_addr=c.out_addr, out_len=16, expected=bytes(range(16)),
        case_name="kat",
    )
    # default (no include_conformance) is byte-identical to the pre-change output
    assert render_integration_tb(c, "chip_top", plan) == render_integration_tb(
        c, "chip_top", plan, include_conformance=False
    )
    assert render_integration_tb(c, "chip_top", plan).count("@cocotb.test()") == 1
    assert "qspi_slave_protocol_conformance" not in render_integration_tb(
        c, "chip_top", plan
    )


def test_write_qspi_conformance_tb_persists_file(tmp_path):
    c = QSPIContract()
    out = tmp_path / "tb" / "test_conf.py"
    res = write_qspi_conformance_tb(str(tmp_path), "chip_top", c, str(out))
    assert out.exists()
    compile(out.read_text(), "<written>", "exec")
    assert res["qspi_conformance"] is True and res["conformance_only"] is True
    assert res["test_count"] == 1
    assert res["contract_fingerprint"] == c.fingerprint()


def test_conformance_tb_is_deterministic_for_identical_contract():
    c = QSPIContract()
    assert render_conformance_tb(c, "chip_top") == render_conformance_tb(c, "chip_top")


# ---------------------------------------------------------------------------
# Section 3: meta-tests -- real cocotb/Icarus sim proves catch/pass behaviour
# ---------------------------------------------------------------------------
_SOLUTION_DIR = Path(
    os.environ.get(
        "CORESMITH_BFM_METATEST_SOLUTION_DIR",
        os.path.expanduser(
            "~/coresmith-runs/exp-aes-qspi-cleanmaster-20260709-021607/solution"
        ),
    )
)
_RUN_DIR = _SOLUTION_DIR.parent

requires_icarus = pytest.mark.skipif(
    shutil.which("iverilog") is None, reason="Icarus Verilog not installed"
)
requires_aes_fixture = pytest.mark.skipif(
    not (_SOLUTION_DIR / "qspi_slave_frontend.v").exists(),
    reason="AES QSPI-slave fixture RTL not present",
)


def _break_no_0x05(frontend_src: str) -> str:
    """Drop the cmd 0x05 read_status decode (the image codec class) by re-coding the
    opcode so a real 0x05 is never recognized."""
    anchor = "    localparam [7:0] CMD_READ_STATUS = 8'h05;"
    assert anchor in frontend_src, "CMD_READ_STATUS localparam anchor not found"
    return frontend_src.replace(
        anchor, "    localparam [7:0] CMD_READ_STATUS = 8'h55;", 1
    )


def _break_short_dummy(frontend_src: str) -> str:
    """Shorten the read-data bus turnaround by a nibble so read data launches a
    NIBBLE EARLY (the aes-v3 / fft-v2 ST_DUMMY class)."""
    anchor = "            if (nib_count_q >= 3'd2) begin"
    assert frontend_src.count(anchor) == 1, "short-dummy anchor not unique"
    return frontend_src.replace(
        anchor, "            if (nib_count_q >= 3'd1) begin", 1
    )


def _build_variant(dst: Path, *, mutate=None) -> str:
    dst.mkdir(parents=True, exist_ok=True)
    for v in _SOLUTION_DIR.glob("*.v"):
        shutil.copy2(v, dst / v.name)
    if mutate is not None:
        fe = dst / "qspi_slave_frontend.v"
        fe.write_text(mutate(fe.read_text()))
    # top file stem must match top module name (user_project_wrapper)
    (dst / "user_project_wrapper.v").rename(dst / "caravel_gpio_ctrl.v")
    (dst / "aes_qspi_top.v").rename(dst / "user_project_wrapper.v")
    return "user_project_wrapper"


def _run_conformance_sim(tmp_path: Path, *, mutate=None) -> bool:
    """Generate the conformance-only TB and run it via cocotb/Icarus against the
    (optionally mutated) AES chip-top. Returns True iff the conformance DV passed."""
    sandbox = tmp_path / "dut"
    top = _build_variant(sandbox, mutate=mutate)
    contract = classify_chip_bus(
        str(_RUN_DIR), (sandbox / f"{top}.v").read_text(), None
    )
    assert contract is not None, "AES chip-top must classify QSPI-slave"
    write_qspi_conformance_tb(str(_RUN_DIR), top, contract, str(sandbox / "test_conf.py"))

    sources = " ".join(sorted(p.name for p in sandbox.glob("*.v")))
    (sandbox / "Makefile").write_text(
        "SIM = icarus\nTOPLEVEL_LANG = verilog\n"
        f"VERILOG_SOURCES = {sources}\nTOPLEVEL = {top}\nMODULE = test_conf\n"
        "WAVES = 0\n"
        "include $(shell cocotb-config --makefiles)/Makefile.sim\n"
    )
    # cocotb's Makefile flow needs cocotb-config on PATH; it lives beside the
    # running interpreter regardless of how pytest was invoked.
    env = dict(os.environ)
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    proc = subprocess.run(
        ["make"], cwd=sandbox, env=env, capture_output=True, text=True, timeout=600
    )
    return proc.returncode == 0


@requires_icarus
@requires_aes_fixture
@pytest.mark.requires_nix
@pytest.mark.slow
def test_conformance_accepts_conformant_frontend(tmp_path):
    assert _run_conformance_sim(tmp_path) is True, (
        "conformance DV must ACCEPT the conformant QSPI-slave frontend"
    )


@requires_icarus
@requires_aes_fixture
@pytest.mark.requires_nix
@pytest.mark.slow
def test_conformance_rejects_missing_0x05(tmp_path):
    assert _run_conformance_sim(tmp_path, mutate=_break_no_0x05) is False, (
        "conformance DV must REJECT a frontend that dropped cmd 0x05 read_status "
        "(host wait_done() would time out at the grader)"
    )


@requires_icarus
@requires_aes_fixture
@pytest.mark.requires_nix
@pytest.mark.slow
def test_conformance_rejects_short_dummy_read(tmp_path):
    assert _run_conformance_sim(tmp_path, mutate=_break_short_dummy) is False, (
        "conformance DV must REJECT a frontend that launched read data a nibble "
        "early (short/missing dummy-byte turnaround)"
    )
