# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Deterministic BFM + integration-TB code generation.

``render_bfm_module(contract)`` emits the QSPI-slave master **driver** as text,
purely from the contract params -- the same contract in yields a byte-identical
BFM out (the property the LLM pin driver lacked). ``render_integration_tb(...)``
wraps that driver in a self-contained cocotb integration testbench that runs the
QSPI-slave *host flow* (write CFG/IN, START, poll DONE, read OUT) and asserts the
DUT output equals the LLM-supplied golden reference's expected bytes.

The BFM source is inlined into the TB (not imported) so the generated test is
hermetic: it depends only on ``cocotb``, never on the DUT RTL and never on the
coresmith package being importable inside the sim subprocess.
"""

from __future__ import annotations

import inspect

from . import qspi_contract as _contract_mod
from . import qspi_master_bfm as _bfm_mod
from . import qspi_rom_bfm as _rom_mod
from .maxgeo import BusMaxgeoCoverage, bus_maxgeo_coverage
from .qspi_contract import QSPIContract
from .qspi_rom_bfm import QSPIRomContract
from .stimulus import StimulusPlan


# ---------------------------------------------------------------------------
# CFG read-back probe patterns (bus-protocol conformance, phase A)
# ---------------------------------------------------------------------------
# EVERY BYTE of these constants must have two DIFFERENT nibbles, at every probe
# width the contract can select (``cfg1_width_bytes`` 1..4 -> the low 1..4 bytes
# are used). The old first probe was ``0x11223344``, whose low byte is 0x44:
# both nibbles identical. A read serializer launched ONE NIBBLE EARLY at the
# byte boundary still returns 0x44 for 0x44, so the probe that exists precisely
# to catch a nibble-early launch could not distinguish it from a correct read.
# Measured cost: a full waveform session to re-derive by hand what the probe was
# supposed to prove. Nibble-distinct patterns make a one-nibble slip a value
# mismatch, which is what the assertion message already claims to detect.
CFG_PROBE_A = 0x3C71965A
CFG_PROBE_B = 0xC31E4BA5


def nibble_distinct(value: int, width_bytes: int) -> bool:
    """True when every byte of ``value``'s low ``width_bytes`` has two DIFFERENT
    nibbles -- the property that makes a one-nibble-early read serializer show up
    as a value mismatch instead of an identical byte."""
    for i in range(max(1, int(width_bytes))):
        byte = (int(value) >> (8 * i)) & 0xFF
        if (byte >> 4) == (byte & 0xF):
            return False
    return True


def render_contract_literal(contract: QSPIContract) -> str:
    """A deterministic ``QSPIContract(...)`` constructor literal."""
    items = ", ".join(
        f"{k}={v!r}" for k, v in sorted(contract.to_dict().items())
    )
    return f"QSPIContract({items})"


def render_rom_contract_literal(rom: QSPIRomContract) -> str:
    """A deterministic ``QSPIRomContract(...)`` constructor literal."""
    items = ", ".join(f"{k}={v!r}" for k, v in sorted(rom.to_dict().items()))
    return f"QSPIRomContract({items})"


def render_rom_bfm_module(rom: QSPIRomContract) -> str:
    """Emit the passive ROM-responder BFM (the DUT-mastered second bus) as text.

    Inlined into the TB alongside the master BFM so the generated test is
    hermetic. Deterministic: identical for identical rom contracts.
    """
    contract_src = inspect.getsource(_rom_mod.QSPIRomContract)
    bfm_src = inspect.getsource(_rom_mod.QSPIRomResponderBFM)
    deassert_src = inspect.getsource(_rom_mod._Deassert)
    return (
        "# === AUTO-GENERATED passive ROM-responder BFM (DUT-mastered bus) ===\n"
        "# Answers the DUT's SPI-flash READs from a supplied operand buffer;\n"
        "# DUT-BLIND (samples csn/sck/io, never inspects DUT internals).\n"
        f"# ROM contract fingerprint: {rom.fingerprint()}\n"
        + contract_src + "\n\n" + deassert_src + "\n\n" + bfm_src + "\n\n"
        + f"ROM_CONTRACT = {render_rom_contract_literal(rom)}\n"
    )


def render_bfm_module(contract: QSPIContract) -> str:
    """Emit the deterministic QSPI-slave master driver as standalone source.

    Contains the (param-free) contract dataclass + BFM class bodies verbatim
    plus a ``CONTRACT`` literal built from ``contract``. Deterministic: identical
    for identical contracts.
    """
    contract_src = inspect.getsource(_contract_mod.QSPIContract)
    bfm_src = inspect.getsource(_bfm_mod.QSPIMasterBFM)
    header = (
        "# === AUTO-GENERATED deterministic QSPI-slave master BFM ===\n"
        "# DUT-BLIND: this driver is a pure function of the bus contract; it\n"
        "# NEVER inspects the DUT RTL, so it cannot co-tune to a non-conformant\n"
        f"# DUT. Contract fingerprint: {contract.fingerprint()}\n"
        "from dataclasses import asdict, dataclass\n"
        "from typing import Any\n\n"
    )
    return (
        header
        + contract_src
        + "\n\n"
        + bfm_src
        + "\n\n"
        + f"CONTRACT = {render_contract_literal(contract)}\n"
    )


def render_maxgeo_probe_block(coverage: BusMaxgeoCoverage | None) -> str:
    """Emit phase (D): drive every BUS maximum the design declares, at maximum.

    Empty string when the design declares no bus-drivable maximum -- the emitted
    testbench is then byte-identical to before this phase existed.

    What it proves, and what it does NOT: each probe drives a real dimensional
    maximum on the pins (full-width address, longest legal write burst, longest
    legal read burst, largest command opcode) and then re-checks two things a
    truncated index/length counter breaks -- the frontend returned to idle, and a
    nibble-distinct CFG1 sentinel written BEFORE the traffic is still intact.
    Content is sparse on purpose; the MAX-GEOMETRY gate's own message says sparse
    content at maximum extent is the accepted shape when a full workload is too
    slow. It does NOT check compute results: this testbench has no golden model,
    which is exactly why the compute-lane dims are reported as NOT covered.
    """
    if coverage is None:
        return ""
    probes: list[str] = []
    if coverage.address:
        probes.append(
            "    # address at FULL declared extent: a frontend whose address\n"
            "    # register is narrower aliases this into a decoded aperture.\n"
            "    await bfm.write(0x%X, bytes([0xA5]))\n"
            "    _ = await bfm.read(0x%X, 1)\n"
            "    await _still_conformant('a full-extent address transfer "
            "@0x%X')\n" % (coverage.address, coverage.address, coverage.address)
        )
    if coverage.write_bytes:
        probes.append(
            "    # longest declared IN write burst (sparse content: the point is\n"
            "    # the write pointer/length counter at maximum extent).\n"
            "    _n = %d\n"
            "    await bfm.write(c.in_addr, bytes(((_i * 7 + 0x5A) & 0xFF) "
            "for _i in range(_n)))\n"
            "    await _still_conformant('a %d-byte IN write burst')\n"
            % (coverage.write_bytes, coverage.write_bytes)
        )
    if coverage.read_bytes:
        probes.append(
            "    # longest declared OUT read burst. The returned bytes are NOT\n"
            "    # checked (no compute oracle here); a read-length counter that\n"
            "    # underflows leaves the DUT driving the lanes, which the\n"
            "    # sentinel read-back immediately after detects.\n"
            "    _rd = await bfm.read(c.out_addr, %d)\n"
            "    dut._log.info('[qspi-conformance] max-extent OUT read of %d B: "
            "head=%%s tail=%%s', _rd[:8].hex(), _rd[-8:].hex())\n"
            "    await _still_conformant('a %d-byte OUT read burst')\n"
            % (coverage.read_bytes, coverage.read_bytes, coverage.read_bytes)
        )
    if coverage.opcode:
        probes.append(
            "    # largest declared command opcode -- the command decoder at the\n"
            "    # top of its field. Must return to idle like any other opcode.\n"
            "    await bfm.send_opcode_only(0x%02X)\n"
            "    await _still_conformant('command opcode 0x%02X (declared max)')\n"
            % (coverage.opcode, coverage.opcode)
        )
    if not probes:
        return ""
    header = (
        "\n"
        "    # -- (D) MAX-EXTENT BUS probes (index/address width at maximum) ----\n"
        "    # A chip can pass every fixed-small test and still ship a counter\n"
        "    # that wraps at a 2^n boundary BELOW the declared maximum. These\n"
        "    # probes drive the BUS maxima this design declares. Compute-lane\n"
        "    # geometry is NOT covered here -- see the MAXGEO_NOT_COVERED marker\n"
        "    # at the top of this file.\n"
        "    await bfm.reset_dut(10)\n"
        "    _sent = 0x%08X & _wm          # nibble-distinct sentinel\n"
        "    await bfm.write_reg(c.cfg1_addr, _sent, _w)\n"
        "\n"
        "    async def _still_conformant(_what):\n"
        "        _s = await bfm.read_status()\n"
        "        assert (_s & busy_bit) == 0, (\n"
        "            'QSPI-slave MAX-EXTENT probe: frontend still BUSY after %%s '\n"
        "            '-- it WEDGED at maximum extent. An index/length counter '\n"
        "            'sized below the declared maximum stalls exactly here.' %% _what)\n"
        "        _rb = int.from_bytes(await bfm.read(c.cfg1_addr, _w), 'little')\n"
        "        assert _rb == _sent, (\n"
        "            'QSPI-slave MAX-EXTENT probe: the CFG1 sentinel changed from '\n"
        "            '0x%%x to 0x%%x after %%s. A truncated address/length counter '\n"
        "            'wrapped out of its aperture and into the decoded register '\n"
        "            'file.' %% (_sent, _rb, _what))\n"
        "\n" % (CFG_PROBE_A,)
    )
    return header + "\n".join(probes)


def render_conformance_test_fn(
    contract: QSPIContract,
    *,
    start_clock: bool = True,
    with_op_probe: bool = True,
    coverage: BusMaxgeoCoverage | None = None,
) -> str:
    """Emit the QSPI-slave BUS-PROTOCOL conformance ``@cocotb.test()`` as source.

    COMPUTE-LANE INDEPENDENT: the test needs no golden compute model -- it exercises
    only the standard chassis QSPI-slave contract the frontend keeps re-deriving
    and losing a command / turnaround from. Four phases:

      (A) cmd 0x02 write + cmd 0x03 read-back through the dummy-byte turnaround
          (catches a read launched a NIBBLE EARLY -- short/missing dummy -- and
          read-serializer nibble misalignment);
      (B) cmd 0x05 READ_STATUS decodes to a well-formed idle status, and the
          frontend does not wedge on a bad opcode (ERROR-on-bad-opcode is advisory);
      (C) cmd 0x05 status is LIVE after START (catches a DROPPED 0x05 read_status,
          the image codec class where the host's wait_done() times out);
      (D) every BUS maximum the design declares, driven at maximum extent
          (:func:`render_maxgeo_probe_block`) -- omitted entirely when
          ``coverage`` is None or the design declares no bus-drivable maximum.

    ``start_clock`` is retained for signature compatibility but NO LONGER
    changes the emitted code: every generated ``@cocotb.test()`` starts its own
    clock coroutine. The old False path relied on the FIRST test's clock
    persisting across tests in the same sim -- true under cocotb 1.x, false
    under cocotb 2.x (coroutines are killed between tests), which left every
    appended conformance test hanging on a dead clock. ``with_op_probe`` gates
    phase (C); it is redundant when a golden host-flow test in the same module
    already drove START -> wait_done(0x05).
    """
    del start_clock  # every test starts its own clock (cocotb 2.x kills coroutines)
    clk = contract.clk_name
    _probe_a = f"0x{CFG_PROBE_A:08X}"
    _probe_b = f"0x{CFG_PROBE_B:08X}"
    clock_line = (
        f'    cocotb.start_soon(Clock(dut.{clk}, c.clk_period_ns, '
        'unit="ns").start())\n'
    )
    maxgeo_block = render_maxgeo_probe_block(coverage)
    op_probe = ""
    if with_op_probe:
        op_probe = '''
    # -- (C) 0x05 status LIVENESS after START (catches a MISSING 0x05 command) --
    # Drives a minimal GENERIC op (CFG0=1, one 16-byte IN block, START) and polls
    # STATUS via cmd 0x05. It does NOT check compute-output correctness (no golden
    # model needed) -- it only asserts the 0x05 READ_STATUS path reflects a LIVE
    # status. A frontend that never decoded 0x05 returns 0x00 forever here, so the
    # host's wait_done() would time out at the grader; that failure is pulled
    # forward to generation.
    await bfm.reset_dut(10)
    await bfm.write_reg(c.cfg0_addr, 1, 4)
    await bfm.write(c.in_addr, bytes(range(16)))
    await bfm.start()
    _seen = 0
    _got_done = False
    for _ in range(4000):
        _st = await bfm.read_status()
        _seen |= _st
        if _st & (done_bit | err_bit):
            _got_done = bool(_st & done_bit)
            break
        await bfm._tick(16)
    assert _seen != 0, (
        "QSPI-slave conformance: STATUS read via cmd 0x05 is STUCK at 0x00 after "
        "START -- the frontend never asserted BUSY/DONE/ERROR on the 0x05 path. "
        "This is the MISSING-0x05 read_status class (host wait_done() times out "
        "and the whole chip fails the grader on a trivial bus gap). Decode cmd "
        "0x05 and return the {BUSY,DONE,ERROR} status bits."
    )
    dut._log.info(
        "[qspi-conformance] 0x05 status liveness OK (status-or=0x%02x, done=%s)",
        _seen, _got_done)
'''
    return f'''
@cocotb.test()
async def qspi_slave_protocol_conformance(dut):
    """DUT-blind QSPI-slave BUS PROTOCOL conformance -- COMPUTE-LANE INDEPENDENT.

    Runs whether or not the exercise-specific compute lane is modeled. Enforces
    the standard chassis QSPI-slave contract the frontend keeps re-deriving and
    losing a command / turnaround from. A frontend missing 0x05 or mistiming the
    read turnaround FAILS here, at generation -- not silently at the secret grader.
    """
    c = CONTRACT
{clock_line}    busy_bit = 1 << c.status_busy_bit
    done_bit = 1 << c.status_done_bit
    err_bit = 1 << c.status_error_bit
    bfm = QSPIMasterBFM(dut, c)
    await bfm.reset_dut(10)

    # -- (A) CFG write / read-back through the dummy-byte turnaround ----------
    # Probe the RW CFG1 scalar (a pure store, no compute coupling): write a
    # distinct-nibble pattern, read it back via cmd 0x03 through the dummy
    # turnaround. A wrong read-back means the read launched a NIBBLE EARLY
    # (short/missing dummy) or the serializer is nibble-misaligned. CFG1 is
    # used rather than CFG0 because some designs consume CFG0 (n_blocks/length)
    # in the control path; CFG1 is the neutral read/write scalar. The probe
    # width comes from the contract (default 1 byte -- universally legal on a
    # byte-addressed map; a hardcoded 4-byte access spans undecoded bytes on
    # designs whose CFG1 is a narrow register and false-fails them).
    _w = int(getattr(c, "cfg1_width_bytes", 1) or 1)
    _wm = (1 << (8 * _w)) - 1
    for _addr, _val in ((c.cfg1_addr, {_probe_a} & _wm),
                        (c.cfg1_addr, {_probe_b} & _wm)):
        await bfm.write_reg(_addr, _val, _w)
        _rb = await bfm.read(_addr, _w)
        _got = int.from_bytes(_rb, "little")
        assert _got == _val, (
            "QSPI-slave conformance: CFG1 read-back mismatch @0x%06x -- wrote "
            "0x%08x, read 0x%08x (width %d). A wrong read-back byte means the "
            "read data launched a NIBBLE EARLY (short/missing dummy-byte "
            "turnaround) or the read serializer is nibble-misaligned. Prefetch "
            "the read DURING the dummy phase and serialize MSB-nibble-first."
            % (_addr, _val, _got, _w)
        )

    # -- (B) 0x05 READ_STATUS decodes to a well-formed idle status -----------
    await bfm.reset_dut(10)
    _st = await bfm.read_status()
    assert (_st & busy_bit) == 0, (
        "QSPI-slave conformance: idle STATUS.BUSY=1 via cmd 0x05 (status=0x%02x)."
        % _st
    )
    assert (_st & done_bit) == 0, (
        "QSPI-slave conformance: idle STATUS.DONE=1 before any op via cmd 0x05 "
        "(status=0x%02x)." % _st
    )

    # bad-opcode robustness: the frontend must not wedge; per the chassis protocol
    # it SHOULD flag STATUS.ERROR on an unknown opcode (advisory -- logged).
    await bfm.send_opcode_only(0xAB)
    _st_bad = await bfm.read_status()
    assert (_st_bad & busy_bit) == 0, (
        "QSPI-slave conformance: frontend WEDGED after a bad opcode -- STATUS "
        "still BUSY via cmd 0x05 (status=0x%02x). An unknown opcode must return "
        "the frontend to idle." % _st_bad
    )
    if (_st_bad & err_bit) == 0:
        dut._log.warning(
            "[qspi-conformance] ADVISORY: bad opcode 0xAB did not raise "
            "STATUS.ERROR (status=0x%02x). The chassis protocol flags unknown "
            "opcodes via STATUS.ERROR (bit %d).", _st_bad, c.status_error_bit)
{op_probe}{maxgeo_block}'''


def render_maxgeo_markers(coverage: BusMaxgeoCoverage | None) -> str:
    """The ``# MAXGEO`` header block for a generated testbench.

    Three lines, all machine-readable:

      ``# MAXGEO: <dim>=<max> ...``       -- ONLY dims this TB drives at maximum
      ``# MAXGEO_SCOPE: ...``             -- what kind of coverage this is
      ``# MAXGEO_NOT_COVERED: <dim>=<max> ...`` -- what it does NOT reach

    The last two deliberately do NOT match the gate's ``# MAXGEO\\b`` marker
    regex (``MAXGEO_`` has no word boundary after ``MAXGEO``), so recording an
    UNCOVERED dim can never be misread as coverage. A marker is a claim; a claim
    without the traffic behind it is worse than no claim at all.
    """
    if coverage is None:
        return ""
    lines = [ln for ln in (coverage.marker_line(),) if ln]
    if not lines and not coverage.uncovered:
        return ""
    lines.append(
        "# MAXGEO_SCOPE: bus-contract-only -- this testbench is the deterministic "
        "QSPI-slave\n#   conformance DV. It has NO compute oracle, so it drives the "
        "BUS dimensions at\n#   their declared maxima and cannot drive compute-lane "
        "geometry at all."
    )
    if coverage.uncovered_line():
        lines.append(coverage.uncovered_line())
    return "\n".join(lines) + "\n"


def render_conformance_tb(
    contract: QSPIContract,
    design_name: str,
    declared_dims: dict | None = None,
) -> str:
    """Emit a standalone, hermetic QSPI-slave BUS-PROTOCOL conformance testbench.

    Used at integration DV when a golden host-flow plan cannot be derived (the
    exercise-specific compute-lane oracle is NOT modeled -- e.g. a BT.656 input
    lane). The deterministic master BFM still drives the QSPI-slave pins and the
    conformance test enforces the standard command set + timing, so a frontend
    that dropped 0x05 or mistimed the read turnaround FAILS at generation instead
    of slipping through to the secret grader on the LLM-authored BFM.

    ``declared_dims`` is the design's machine-readable dimensional maxima
    (``{name: max}``, from the caller's ``_declared_dimensions``). The bus-drivable
    subset becomes real max-extent traffic plus a ``# MAXGEO`` marker; everything
    else is recorded as NOT covered. ``None`` keeps the emitted TB byte-identical
    to before max-geometry coverage existed.
    """
    coverage = (
        bus_maxgeo_coverage(contract, declared_dims)
        if declared_dims else None
    )
    bfm_src = render_bfm_module(contract)
    fn = render_conformance_test_fn(
        contract, start_clock=True, with_op_probe=True, coverage=coverage
    )
    maxgeo_markers = render_maxgeo_markers(coverage)
    return f'''# Copyright (c) Meta Platforms, Inc. and affiliates.
# AUTO-GENERATED by orchestrator.langgraph.bfm_lib -- DO NOT HAND-EDIT.
#
# QSPI-slave BUS-PROTOCOL CONFORMANCE DV for chip-top {design_name!r}.
# COMPUTE-LANE INDEPENDENT: enforces the standard chassis QSPI-slave contract
# (cmd 0x02 write / 0x03 read + dummy turnaround / 0x05 read_status) even when
# the exercise-specific compute oracle is NOT modeled. This is the gate a frontend
# that dropped 0x05 or mistimed the read turnaround must fail at generation -- the
# the image codec gap where such a frontend slipped through on the co-tuned LLM BFM.
{maxgeo_markers}from __future__ import annotations
# The future import is REQUIRED, not style: the BFM class source is inlined
# below via inspect.getsource, and its `-> QSPIContract` method annotations
# only defer (instead of evaluating at class-body time, where the name does
# not exist yet) when this file itself has the future import. Without it the
# generated module dies at import with NameError -- which is how the
# contract-enforcing conformance DV shipped without ever having produced an
# importable testbench.
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

{bfm_src}
{fn}'''


def render_integration_tb(
    contract: QSPIContract,
    design_name: str,
    plan: StimulusPlan,
    rom_contract: QSPIRomContract | None = None,
    rom_data: bytes = b"",
    *,
    include_conformance: bool = False,
    declared_dims: dict | None = None,
) -> str:
    """Emit a hermetic deterministic cocotb integration testbench.

    The BFM is inlined; the golden reference has already been evaluated at
    generation time into ``plan.expected`` (the LLM supplies the MODEL, this
    driver supplies the pins). The single test drives the host flow and asserts
    byte-exact OUT == expected.

    When ``rom_contract`` is supplied the DUT masters a SECOND bus (it reads
    operands from an external ROM): the passive ROM-responder BFM is ALSO inlined
    and started as a background coroutine, answering the DUT's reads from
    ``rom_data`` -- so a DUT that stalls waiting for an external device is
    actually exercised. Both BFMs are then wired into the one deterministic TB.

    When ``include_conformance`` is True the compute-lane-independent QSPI-slave
    bus-protocol conformance test (:func:`render_conformance_test_fn`) is appended
    to the SAME module so one sim run enforces BOTH the golden host-flow output AND
    the raw bus contract (CFG read-back through the dummy turnaround, 0x05 status,
    bad-opcode robustness). Default False keeps the emitted TB byte-identical.

    ``declared_dims`` (with ``include_conformance``) additionally drives the
    design's BUS maxima at maximum extent and emits the matching ``# MAXGEO``
    marker. It never invents coverage for the compute lane: the golden host-flow
    case above is whatever geometry the golden reference chose, and this
    testbench does not claim otherwise.
    """
    coverage = (
        bus_maxgeo_coverage(contract, declared_dims)
        if (declared_dims and include_conformance) else None
    )
    maxgeo_markers = render_maxgeo_markers(coverage)
    bfm_src = render_bfm_module(contract)
    cfg_lits = ", ".join(f"({a}, {v}, {w})" for (a, v, w) in plan.cfg)
    write_lits = ", ".join(f"({a}, bytes.fromhex({d.hex()!r}))" for (a, d) in plan.writes)
    clk = contract.clk_name

    # Optional DUT-mastered ROM bus: inline the responder + start it.
    rom_src = ""
    rom_start = ""
    if rom_contract is not None:
        rom_src = "\n" + render_rom_bfm_module(rom_contract) + (
            f"ROM_DATA = bytes.fromhex({bytes(rom_data).hex()!r})\n")
        rom_start = (
            "    # DUT-mastered ROM bus: passive responder answers the DUT's\n"
            "    # SPI-flash reads from the operand buffer (started before the\n"
            "    # host flow so operands are available when the DUT fetches).\n"
            "    _rom = QSPIRomResponderBFM(dut, ROM_CONTRACT, ROM_DATA)\n"
            "    cocotb.start_soon(_rom.run())\n")

    tb = f'''# Copyright (c) Meta Platforms, Inc. and affiliates.
# AUTO-GENERATED by orchestrator.langgraph.bfm_lib -- DO NOT HAND-EDIT.
#
# Deterministic QSPI-slave integration DV for chip-top {design_name!r}.
# The pin driver is the library BFM (contract-faithful, DUT-blind); the LLM
# supplies only the golden MODEL, evaluated at generation time into the
# expected bytes below. A DUT that violates the QSPI-slave read/dummy/serialize
# contract de-aligns and is caught -- the property the co-tuned LLM BFM lacked.
{maxgeo_markers}from __future__ import annotations
# The future import is REQUIRED, not style: the BFM class source is inlined
# below via inspect.getsource, and its `-> QSPIContract` method annotations
# only defer (instead of evaluating at class-body time, where the name does
# not exist yet) when this file itself has the future import. Without it the
# generated module dies at import with NameError -- which is how the
# contract-enforcing conformance DV shipped without ever having produced an
# importable testbench.
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

{bfm_src}
{rom_src}
# ---- host-flow plan (deterministic; oracle = LLM golden reference) ----------
CFG_WRITES = [{cfg_lits}]          # (addr, value, width_bytes)
IN_WRITES = [{write_lits}]         # (addr, data)
OUT_ADDR = {plan.out_addr}
OUT_LEN = {plan.out_len}
EXPECTED = bytes.fromhex({plan.expected.hex()!r})
CASE_NAME = {plan.case_name!r}


@cocotb.test()
async def deterministic_qspi_dv(dut):
    """Contract-faithful QSPI-slave host-flow DV for case {plan.case_name!r}."""
    c = CONTRACT
    cocotb.start_soon(Clock(dut.{clk}, c.clk_period_ns, unit="ns").start())

    # v3 Section 2: DETERMINISTIC chip-level cycle accounting. A free-running
    # counter (written by codegen, NOT measured by an LLM) times the op window
    # from op-start-committed (START written) to DONE visible on the status pin
    # -- mirroring how a grader-style host measures cycles/op. Started before the
    # host flow so it is live when START is issued.
    _cyc = {{"n": 0}}

    async def _cycle_counter():
        while True:
            await RisingEdge(dut.{clk})
            _cyc["n"] += 1

    cocotb.start_soon(_cycle_counter())

{rom_start}    bfm = QSPIMasterBFM(dut, c)
    await bfm.reset_dut(10)

    for addr, value, width in CFG_WRITES:
        await bfm.write_reg(addr, value, width)
    for addr, data in IN_WRITES:
        await bfm.write(addr, data)

    await bfm.start()
    _op_start_cyc = _cyc["n"]          # op-start committed (START written)
    done = await bfm.wait_done()
    _op_end_cyc = _cyc["n"]            # DONE visible on the status pin

    # Record the measured chip op window (one START->DONE = one op). Written
    # BEFORE the correctness assert so the measurement lands even if OUT differs;
    # the engine's integration measured-throughput gate reads this artifact and
    # gates it against the chip budget x 1.1.
    try:
        import json as _json
        _measured = max(0, _op_end_cyc - _op_start_cyc)
        with open("integration_throughput.json", "w") as _tf:
            _json.dump({{"measured_cyc_per_op": float(_measured),
                        "n_ops": 1}}, _tf)
        dut._log.info("[det-bfm] chip op window = %d cyc (START->DONE)",
                      _measured)
    except Exception:  # noqa: BLE001 - measurement never fails the DV
        pass

    assert done, "DUT never signalled STATUS.DONE within the contract timeout"

    out = await bfm.read(OUT_ADDR, OUT_LEN)
    dut._log.info("[det-bfm] expected=%s", EXPECTED.hex())
    dut._log.info("[det-bfm] observed=%s", out.hex())
    assert out == EXPECTED, (
        "QSPI-slave contract violation: OUT != golden reference. "
        f"exp={{EXPECTED.hex()}} got={{out.hex()}}"
    )
'''
    if include_conformance:
        # Append the compute-lane-independent bus-protocol conformance test to the
        # SAME module: the golden host-flow test above already started the clock
        # and drove START -> wait_done(0x05), so the appended test reuses that
        # clock (start_clock=False) and skips the redundant op probe -- it adds
        # the CFG read-back-through-dummy and bad-opcode robustness coverage the
        # host-flow alone does not exercise.
        tb += render_conformance_test_fn(
            contract, start_clock=False, with_op_probe=False, coverage=coverage
        )
    return tb
