# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the deterministic, contract-faithful QSPI-slave BFM library.

Fast tests (no EDA): the classifier, stimulus/oracle plan builder, codegen
determinism, generated-TB syntactic validity, and the CORESMITH_DETERMINISTIC_BFM
two-branch env gate.

Meta-test (requires_nix + Verilator): the property the LLM BFM lacked -- the
library BFM MUST ACCEPT a conforming QSPI-slave DUT and REJECT a known
non-conforming one (the AES read-serializer "doubled-low-nibble" bug). It uses
the post-fix (conformant) AES solution RTL as the accept fixture and a
reconstructed doubled-low-nibble twin as the reject fixture. Skipped when the
fixtures / Verilator are not present.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

from orchestrator.langgraph import bfm_lib
from orchestrator.langgraph.bfm_lib import (
    QSPIContract,
    build_plan_from_run,
    classify_chip_bus,
    deterministic_bfm_enabled,
    plan_deterministic_dv,
    render_bfm_module,
    render_integration_tb,
)

# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------

_GPIO_TOP = """
module chip_top (
    input  wire        wb_clk_i,
    input  wire        wb_rst_i,
    input  wire [37:0] io_in,
    output wire [37:0] io_out,
    output wire [37:0] io_oeb
);
endmodule
"""

_NON_GPIO_TOP = """
module chip_top (
    input  wire        clk,
    input  wire        rst,
    input  wire [31:0] s_axis_tdata,
    input  wire        s_axis_tvalid,
    output wire        s_axis_tready
);
endmodule
"""

# a tiny pure reference: ciphertext = bytewise-inverted data (self-contained)
_GOLDEN = textwrap.dedent(
    """
    def run(stimulus):
        data = bytes(stimulus["data"])
        return bytes((~b) & 0xFF for b in data)
    """
)

_ACCEPT_STIM = textwrap.dedent(
    """
    cases = [
        ("case0", {"key": list(range(16)), "data": list(range(16))}),
    ]
    """
)


def _make_run(tmp_path: Path, *, qspi: bool = True, with_stim: bool = True) -> Path:
    root = tmp_path / "run"
    (root / ".coresmith").mkdir(parents=True)
    (root / "inputs").mkdir(parents=True)
    proto = "custom QSPI slave over Caravel GPIO" if qspi else "AXI-Stream"
    (root / ".coresmith" / "prd_spec.json").write_text(
        '{"prd": {"dataflow": {"bus_protocol": "%s"}}}' % proto
    )
    if with_stim:
        (root / "inputs" / "acceptance_stimulus.py").write_text(_ACCEPT_STIM)
        (root / "inputs" / "aes_golden.py").write_text(_GOLDEN)
    return root


# ---------------------------------------------------------------------------
# contract + codegen determinism
# ---------------------------------------------------------------------------

def test_contract_defaults_match_protocol():
    c = QSPIContract()
    assert (c.cmd_write, c.cmd_read, c.cmd_status) == (0x02, 0x03, 0x05)
    assert c.dummy_bytes == 1
    assert c.sample_edge == "rising"
    assert c.endianness == "little"
    assert (c.in_addr, c.out_addr) == (0x001000, 0x002000)
    assert c.fingerprint() == QSPIContract().fingerprint()


def test_bfm_generator_is_deterministic():
    c = QSPIContract()
    a = render_bfm_module(c)
    b = render_bfm_module(c)
    assert a == b, "same contract must render a byte-identical BFM"
    assert "QSPIMasterBFM" in a and "CONTRACT =" in a
    # a different contract changes the emitted CONTRACT literal
    c2 = QSPIContract(dummy_bytes=2)
    assert render_bfm_module(c2) != a


def test_render_integration_tb_is_valid_python():
    c = QSPIContract()
    plan = bfm_lib.StimulusPlan(
        cfg=[(c.cfg0_addr, 1, 4)],
        writes=[(c.in_addr, bytes(range(32)))],
        out_addr=c.out_addr,
        out_len=16,
        expected=bytes(range(16)),
        case_name="unit",
    )
    src = render_integration_tb(c, "chip_top", plan)
    compile(src, "<generated_tb>", "exec")  # must parse
    assert "QSPIMasterBFM" in src
    assert "deterministic_qspi_dv" in src
    # the DUT RTL is never referenced -- the driver is DUT-blind
    assert "chip_top.v" not in src


def test_integration_tb_emits_cycle_accounting():
    # v3 Section 2: the deterministic TB carries a codegen-written cycle counter
    # that times the op window (START committed -> DONE) and writes the measured
    # chip cyc/op to integration_throughput.json.
    c = QSPIContract()
    plan = bfm_lib.StimulusPlan(
        cfg=[(c.cfg0_addr, 1, 4)],
        writes=[(c.in_addr, bytes(range(32)))],
        out_addr=c.out_addr, out_len=16,
        expected=bytes(range(16)), case_name="unit",
    )
    src = render_integration_tb(c, "chip_top", plan)
    compile(src, "<generated_tb>", "exec")  # doubled f-string braces must parse
    assert "RisingEdge" in src
    assert "_cycle_counter" in src
    assert "integration_throughput.json" in src
    assert "measured_cyc_per_op" in src
    # op window captured across start()/wait_done()
    assert "_op_start_cyc" in src and "_op_end_cyc" in src


# ---------------------------------------------------------------------------
# classifier
# ---------------------------------------------------------------------------

def test_classifier_positive(tmp_path):
    run = _make_run(tmp_path, qspi=True)
    c = classify_chip_bus(str(run), _GPIO_TOP, connections=None)
    assert isinstance(c, QSPIContract)


def test_classifier_negative_without_gpio_boundary(tmp_path):
    run = _make_run(tmp_path, qspi=True)
    assert classify_chip_bus(str(run), _NON_GPIO_TOP, connections=None) is None


def test_classifier_negative_without_corroboration(tmp_path):
    # GPIO boundary present but neither PRD nor connections mention QSPI
    run = _make_run(tmp_path, qspi=False)
    assert classify_chip_bus(str(run), _GPIO_TOP, connections=None) is None


def test_classifier_connections_corroboration(tmp_path):
    run = _make_run(tmp_path, qspi=False)
    conns = [{"interface": "qspi_pin_sample", "from_block": "wrapper"}]
    assert classify_chip_bus(str(run), _GPIO_TOP, connections=conns) is not None


# ---------------------------------------------------------------------------
# stimulus / oracle plan
# ---------------------------------------------------------------------------

def test_build_plan_from_run(tmp_path):
    run = _make_run(tmp_path, qspi=True)
    c = QSPIContract()
    plan = build_plan_from_run(str(run), c)
    assert plan is not None
    # IN window write = key(16) + data(16)
    assert plan.writes and plan.writes[0][0] == c.in_addr
    assert len(plan.writes[0][1]) == 32
    # CFG0 = n_blocks = 1
    assert (c.cfg0_addr, 1, 4) in plan.cfg
    # oracle = bytewise-inverted data
    assert plan.expected == bytes((~b) & 0xFF for b in range(16))
    assert plan.out_len == 16


def test_build_plan_returns_none_without_reference(tmp_path):
    run = _make_run(tmp_path, qspi=True, with_stim=False)
    assert build_plan_from_run(str(run), QSPIContract()) is None


# ---------------------------------------------------------------------------
# CORESMITH_DETERMINISTIC_BFM two-branch env gate (repo convention)
# ---------------------------------------------------------------------------

def test_env_gate_off_by_default(monkeypatch):
    monkeypatch.delenv("CORESMITH_DETERMINISTIC_BFM", raising=False)
    assert deterministic_bfm_enabled() is False


def test_env_gate_on_when_set(monkeypatch):
    monkeypatch.setenv("CORESMITH_DETERMINISTIC_BFM", "1")
    assert deterministic_bfm_enabled() is True


def test_plan_deterministic_dv_qspi_when_enabled(tmp_path, monkeypatch):
    """Flag-on + QSPI-slave chip-top -> deterministic (contract-enforcing) plan."""
    monkeypatch.setenv("CORESMITH_DETERMINISTIC_BFM", "1")
    run = _make_run(tmp_path, qspi=True)
    contract, plan = plan_deterministic_dv(str(run), _GPIO_TOP, connections=None)
    assert contract is not None and plan is not None


def test_plan_deterministic_dv_falls_back_for_non_qspi(tmp_path, monkeypatch):
    """Flag-on + non-QSPI chip-top -> (None, None): caller keeps LLM BFM +
    logs the loud not-contract-enforcing advisory."""
    monkeypatch.setenv("CORESMITH_DETERMINISTIC_BFM", "1")
    run = _make_run(tmp_path, qspi=True)
    contract, plan = plan_deterministic_dv(str(run), _NON_GPIO_TOP, connections=None)
    assert contract is None and plan is None


# ---------------------------------------------------------------------------
# META-TEST: accept conformant DUT / reject non-conformant read serializer
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

requires_verilator = pytest.mark.skipif(
    shutil.which("verilator") is None, reason="Verilator not installed"
)
requires_aes_fixture = pytest.mark.skipif(
    not (_SOLUTION_DIR / "qspi_slave_frontend.v").exists(),
    reason="AES QSPI-slave fixture RTL not present",
)


def _reconstruct_doubled_low_nibble(frontend_src: str) -> str:
    """Reintroduce the documented read-serializer defect: DATA read bytes emit
    the LOW nibble in both serialization slots (highs dropped). STATUS path
    untouched so the op still completes -> OUT ciphertext mismatch."""
    anchor = (
        "                    PH_READ_HI: begin\n"
        "                        m_qspi_pin_drive_data <= {read_byte_q[7:4], 4'h0, irq_level_q};"
    )
    patched = (
        "                    PH_READ_HI: begin\n"
        "                        if (pending_read_status_q == 1'b1)\n"
        "                            m_qspi_pin_drive_data <= {read_byte_q[7:4], 4'h0, irq_level_q};\n"
        "                        else\n"
        "                            m_qspi_pin_drive_data <= {read_byte_q[3:0], 4'h0, irq_level_q};"
    )
    assert anchor in frontend_src, "read-serializer anchor not found in fixture"
    return frontend_src.replace(anchor, patched, 1)


def _build_variant(dst: Path, *, buggy: bool) -> str:
    """Assemble a sandbox whose top file stem == top module (user_project_wrapper).
    Returns the top module name."""
    dst.mkdir(parents=True, exist_ok=True)
    for v in _SOLUTION_DIR.glob("*.v"):
        shutil.copy2(v, dst / v.name)
    if buggy:
        fe = dst / "qspi_slave_frontend.v"
        fe.write_text(_reconstruct_doubled_low_nibble(fe.read_text()))
    # top module `user_project_wrapper` lives in aes_qspi_top.v; make the file
    # stem match the module name and move the caravel adapter off that name.
    (dst / "user_project_wrapper.v").rename(dst / "caravel_gpio_ctrl.v")
    (dst / "aes_qspi_top.v").rename(dst / "user_project_wrapper.v")
    return "user_project_wrapper"


def _run_det_bfm_sim(tmp_path: Path, *, buggy: bool) -> bool:
    """Generate the deterministic TB and run it via cocotb/Verilator against the
    conformant or buggy DUT. Returns True iff the DV passed."""
    sandbox = tmp_path / ("buggy" if buggy else "solution")
    top = _build_variant(sandbox, buggy=buggy)

    contract, plan = plan_deterministic_dv(
        str(_RUN_DIR), (sandbox / f"{top}.v").read_text(), connections=None
    )
    assert contract is not None and plan is not None, "AES run must classify QSPI-slave"

    tb = sandbox / "test_det_bfm.py"
    tb.write_text(render_integration_tb(contract, top, plan))

    sources = " ".join(sorted(p.name for p in sandbox.glob("*.v")))
    (sandbox / "Makefile").write_text(
        "SIM = verilator\nTOPLEVEL_LANG = verilog\n"
        f"VERILOG_SOURCES = {sources}\nTOPLEVEL = {top}\nMODULE = test_det_bfm\n"
        "WAVES = 0\nEXTRA_ARGS += --build-jobs 1\n"
        "include $(shell cocotb-config --makefiles)/Makefile.sim\n"
    )
    env = dict(os.environ)
    proc = subprocess.run(
        ["make"], cwd=sandbox, env=env, capture_output=True, text=True, timeout=600
    )
    # cocotb's make returns 0 iff every test passed, nonzero on any failure.
    return proc.returncode == 0


@requires_verilator
@requires_aes_fixture
@pytest.mark.requires_nix
@pytest.mark.slow
def test_metatest_accepts_conformant_dut(tmp_path):
    assert _run_det_bfm_sim(tmp_path, buggy=False) is True, (
        "deterministic BFM must ACCEPT the conformant QSPI-slave DUT"
    )


@requires_verilator
@requires_aes_fixture
@pytest.mark.requires_nix
@pytest.mark.slow
def test_metatest_rejects_nonconformant_read_serializer(tmp_path):
    assert _run_det_bfm_sim(tmp_path, buggy=True) is False, (
        "deterministic BFM must REJECT the doubled-low-nibble read serializer "
        "(the bug a co-tuned BFM passed)"
    )


# ---------------------------------------------------------------------------
# Section 6: passive ROM-responder BFM (DUT-mastered second bus)
# ---------------------------------------------------------------------------

def test_rom_bfm_module_is_deterministic_and_parses():
    from orchestrator.langgraph.bfm_lib import (
        QSPIRomContract, render_rom_bfm_module,
    )
    r = QSPIRomContract(pin_prefix="rom", read_cmd=0x03, addr_bytes=3)
    a = render_rom_bfm_module(r)
    b = render_rom_bfm_module(r)
    assert a == b  # deterministic for identical contracts
    assert "QSPIRomResponderBFM" in a and "ROM_CONTRACT" in a
    # a different contract yields different source
    assert render_rom_bfm_module(QSPIRomContract(read_cmd=0x0B)) != a


def test_integration_tb_wires_both_bfms_when_rom_present():
    from orchestrator.langgraph.bfm_lib import QSPIRomContract
    c = QSPIContract()
    plan = bfm_lib.StimulusPlan(
        cfg=[(c.cfg0_addr, 1, 4)], writes=[(c.in_addr, bytes(range(8)))],
        out_addr=c.out_addr, out_len=8, expected=bytes(range(8)),
        case_name="rom",
    )
    rom = QSPIRomContract(pin_prefix="rom")
    src = render_integration_tb(c, "chip_top", plan,
                                rom_contract=rom, rom_data=bytes(range(64)))
    compile(src, "<gen_tb>", "exec")  # must parse
    # BOTH BFMs present and started
    assert "QSPIMasterBFM" in src and "QSPIRomResponderBFM" in src
    assert "cocotb.start_soon(_rom.run())" in src
    assert "ROM_DATA = bytes.fromhex" in src


def test_integration_tb_omits_rom_when_absent():
    # Backward compatible: no rom_contract -> byte-identical single-BFM TB.
    c = QSPIContract()
    plan = bfm_lib.StimulusPlan(
        cfg=[(c.cfg0_addr, 1, 4)], writes=[(c.in_addr, bytes(range(8)))],
        out_addr=c.out_addr, out_len=8, expected=bytes(range(8)),
        case_name="norom",
    )
    src = render_integration_tb(c, "chip_top", plan)
    assert "QSPIRomResponderBFM" not in src
    assert "_rom.run()" not in src
