# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Post-synthesis GATE-LEVEL SIMULATION -- run the SYNTHESIZED NETLIST against
the same stimulus the RTL passed, and fail closed on any divergence.

WHY THIS EXISTS
---------------
Every functional gate in the pipeline (lint, cocotb DV, coverage) reads the
*RTL source*. Every PPA gate (cells, area, WNS, PnR, DRC) reads the *synthesized
netlist*. Nothing ever ran the netlist against the golden. A design can
therefore ship where the two are DIFFERENT HARDWARE -- the classic case is a
module split by a preprocessor guard into a simulation implementation and a
synthesis implementation, where the synthesis side is a stub. DV passes (it
compiled the sim side); synthesis, PPA, PnR and DRC all pass (they compiled the
stub); line coverage is ~100% (the stub's few lines all execute); and the tape
-out is a brick.

Coverage cannot catch this: a stub executes fully while computing nothing.
Branch-parity (``harness.branch_parity``) catches the *preprocessor* form only,
and only by rebuilding RTL. This gate closes the whole class -- INCLUDING
mis-synthesis, wrongly black-boxed leaves, dropped logic behind a synthesis
pragma, and an un-driven output the RTL happened to initialize -- because it
simulates the ACTUAL ARTIFACT that becomes silicon.

METHOD: cycle-accurate VECTOR REPLAY (design-agnostic)
-----------------------------------------------------
1. Re-run the block's existing RTL DV under a PINNED seed with a shallow VCD
   (``--trace-depth 1``), so the trace holds the top-level PORTS and little
   else. This is the reference run: it is the build that already passed DV
   against the golden.
2. Extract, for every top-level port, the value in effect IMMEDIATELY BEFORE
   each clock posedge. That is the exact steady state the design presents at
   every cycle boundary -- inputs the edge is about to sample, outputs that
   settled from the previous edge.
3. Verilate the GATE NETLIST + the PDK standard-cell simulation models and
   drive it from a generated C++ testbench (``--cc --exe``, no cocotb): for each
   recorded cycle, apply the recorded inputs, ``eval()``, compare EVERY output
   port against the recorded value, then pulse the clock.
4. Any divergence, a netlist that will not elaborate, a build/run failure, or a
   blank/absent verdict -> FAIL.

Because the reference outputs are the ones that matched the golden, "gate ==
reference" transitively means "gate == golden" -- with no design-specific golden
plumbing in this module. Nothing here knows a design, a domain or a file name.

SKY130 GATE-SIM UNDER VERILATOR (the footguns, and what we do about them)
------------------------------------------------------------------------
* **UDPs.** ``sky130_fd_sc_hd.v``'s sequential cells are built on Verilog-1995
  user-defined primitives (``sky130_fd_sc_hd__udp_dff$P`` ...) whose bodies live
  in ``primitives.v`` as ``table``/``endtable``. Verilator cannot compile UDP
  tables (``Unsupported: Verilog 1995 UDP Tables``) and ``--bbox-unsup`` would
  black-box every flop and latch -- i.e. silently delete all sequential
  behaviour, the worst possible fail-open. ``-DFUNCTIONAL`` alone does NOT
  remove them: the functional cell bodies still *instantiate* the UDPs.
  What ``-DFUNCTIONAL`` (with ``USE_POWER_PINS`` left undefined) does do is
  reduce the referenced set to TEN simple, power-pin-free UDPs. We therefore do
  not compile ``primitives.v`` at all and supply :func:`udp_shim_source` --
  behavioural ``module`` replacements for exactly those ten, each transcribed
  from its UDP truth table. Flops use non-blocking assignment, which removes the
  clock-edge race that makes zero-delay gate sim unreliable.
* **Power pins.** Yosys-written netlists carry no VPWR/VGND connections, so
  ``USE_POWER_PINS`` is left UNDEFINED and the cell models declare their own
  ``supply1``/``supply0``. This also keeps the power-good UDPs out of the
  referenced set.
* **Delays.** ``-DUNIT_DELAY=`` (empty) erases the ``#UNIT_DELAY`` annotations,
  and the ``FUNCTIONAL`` bodies carry no ``specify`` blocks -- so the whole cell
  library is zero-delay and ``--timing`` is NOT required. Functional gate sim is
  cycle-accurate, not timing-accurate; timing is signed off by STA, not here.
* **Hard macros.** The PDK's OpenRAM behavioural models drive
  ``#(T_HOLD) dout = 'bx;`` from a posedge block and re-drive from a negedge
  block. Without ``--timing`` both delays collapse to zero and the X-drive races
  the consumer, destroying every read; with ``VERBOSE=1`` they also ``$display``
  on every access. We therefore substitute a generated CYCLE-ACCURATE model per
  instantiated macro (:func:`macro_model_source`), reproducing the PDK model's
  cycle semantics with no delays and no tracing. Its INTERFACE -- which ports
  exist, and how wide each one is -- is read from the REAL macro's own Verilog
  (:func:`macro_interface`), never computed from registry metadata: a stand-in
  whose ports are guessed produces a confidently wrong verdict (see the comment
  above :func:`macro_interface`). ``CORESMITH_GATE_SIM_MACRO_MODEL=pdk`` opts
  back into the PDK's own file for a simulator that honours delays.

ENV GATES
---------
``CORESMITH_GATE_SIM``            default **ON**. ``0/false/no/off`` disables.
``CORESMITH_GATE_SIM_SCOPE``      ``chip_top`` (default) | ``block`` | ``any``.
                                  Which artifact the gate may judge; a subject
                                  outside the scope is ``not_run`` with a reason.
``CORESMITH_GATE_SIM_STRICT``     default off. When on, a MISSING TOOLCHAIN
                                  (no Verilator, no PDK cell models) is a FAIL
                                  instead of a non-blocking ``not_run``.
``CORESMITH_GATE_SIM_MAX_CYCLES`` default 200000. Reduced-stimulus cap: replay
                                  at most this many cycles.
``CORESMITH_GATE_SIM_TIMEOUT_S``  default 1800.
``CORESMITH_GATE_SIM_MACRO_MODEL`` ``generated`` (default) | ``pdk``.
``CORESMITH_GATE_SIM_DEBUG``      default off. TRIAGE AID: keep comparing past
                                  the first divergence and report how many there
                                  were in total. Does not change the verdict.
``CORESMITH_GATE_SIM_MAX_DIVERGENCES`` default 32. How many mismatches the debug
                                  mode records in full.

FAIL SEMANTICS (fail-closed is the whole point)
-----------------------------------------------
``pass``     gate netlist reproduced the reference cycle-for-cycle.
``fail``     divergence / netlist will not elaborate / build or run error /
             timeout / MISSING OR BLANK VERDICT / zero cycles compared / zero
             output bits compared / reference produced no output activity.
``not_run``  the gate does not apply (disabled, no netlist recorded, toolchain
             absent). Never reported as ``ok`` in strict mode. A ``not_run`` is
             ALWAYS recorded explicitly so a downstream reader cannot mistake
             absence for success.

All heavy imports are deferred; importing this module costs nothing.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

# ---------------------------------------------------------------------------
# Env gates
# ---------------------------------------------------------------------------

GATE_SIM_ENV = "CORESMITH_GATE_SIM"
_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if raw in _FALSE:
        return False
    if raw in _TRUE:
        return True
    return default


def gate_sim_enabled() -> bool:
    """Whether the post-synthesis gate-level simulation gate runs.

    **Default ON.** Nothing else in the flow simulates the artifact that becomes
    silicon: every functional gate reads RTL, every PPA gate reads a netlist.

    The gate was briefly turned default-OFF because it was MIS-SCOPED -- it ran
    per block, and a per-block netlist is an intermediate whose stimulus is that
    block's own testbench rather than real chip traffic. That is fixed by
    :func:`gate_sim_scope`, not by disabling the gate: the default scope is now
    ``chip_top``, so what is default-on is the judgement of the FLAT CHIP
    NETLIST against the integration-DV vectors. Per-block replay still exists as
    fast local feedback (``CORESMITH_GATE_SIM_SCOPE=block``) -- it is how a
    synthesis-side stub scoring 143/144 line coverage was caught.

    ``CORESMITH_GATE_SIM=0`` disables it entirely, which is recorded as
    ``disabled`` with a reason that names what is no longer being checked.
    """
    return _flag(GATE_SIM_ENV, True)


GATE_SIM_SCOPE_ENV = "CORESMITH_GATE_SIM_SCOPE"


def gate_sim_scope() -> str:
    """Which artifact this gate is allowed to judge. Default ``"chip_top"``.

    A PER-BLOCK netlist is not what becomes silicon -- the integrated
    ``chip_top`` netlist is -- and per-block replay uses that block's own
    testbench vectors rather than real chip traffic. So the gate is scoped to
    chip_top: any other subject is ``not_run`` with an explicit reason, never
    a pass and never a failure.

    ``CORESMITH_GATE_SIM_SCOPE=block`` opts back into per-block replay, which
    is genuinely useful as fast local feedback -- it is how a synthesis-side
    stub scoring 143/144 line coverage was caught. ``=any`` judges whatever it
    is handed.

    The chip-top subject is real: ``backend_graph`` runs this gate immediately
    after ``flat_top_synthesis`` -- the first point at which a flat chip netlist
    exists on disk -- and marks the subject ``is_chip_top``, so the guard admits
    it (see :func:`block_is_chip_top`).
    """
    val = os.environ.get(GATE_SIM_SCOPE_ENV, "").strip().lower()
    return val if val in {"block", "chip_top", "any"} else "chip_top"


def block_is_chip_top(block: dict) -> bool:
    """True when this subject is the integrated chip top, not a leaf block."""
    if block.get("is_chip_top") or block.get("scope") == "chip_top":
        return True
    name = str(block.get("name", ""))
    return name in {"chip_top", "integration"} or name.endswith("_chip_top")


def gate_sim_strict() -> bool:
    """When on, an absent toolchain/PDK is a FAIL rather than a non-blocking
    ``not_run``. Default off so a host with no PDK can still run the frontend
    (the same posture as ``CORESMITH_SKIP_SYNTH``), while CI/backend hosts can
    demand the gate actually ran."""
    return _flag("CORESMITH_GATE_SIM_STRICT", False)


def gate_sim_max_cycles() -> int:
    """Reduced-stimulus cap: replay at most this many recorded cycles.

    Gate sim is ~1-2 orders of magnitude slower than RTL sim, so a full-length
    stimulus is usually not affordable per block per attempt. Truncating the
    replay keeps the gate cheap; it never weakens the verdict for the cycles it
    does compare, and a stub diverges within the first handful of output beats.
    """
    try:
        return max(1, int(os.environ.get("CORESMITH_GATE_SIM_MAX_CYCLES", "") or 200000))
    except ValueError:
        return 200000


def gate_sim_timeout_s() -> int:
    try:
        return max(1, int(os.environ.get("CORESMITH_GATE_SIM_TIMEOUT_S", "") or 1800))
    except ValueError:
        return 1800


def gate_sim_macro_model_mode() -> str:
    raw = (os.environ.get("CORESMITH_GATE_SIM_MACRO_MODEL", "") or "generated").strip().lower()
    return "pdk" if raw == "pdk" else "generated"


GATE_SIM_DEBUG_ENV = "CORESMITH_GATE_SIM_DEBUG"


def gate_sim_debug() -> bool:
    """TRIAGE AID: keep comparing after the first divergence. Default off.

    The default driver stops at the first mismatch, so ``cycles_compared`` reads
    like a test length when it is really "where it stopped" -- 5151 of 200000
    recorded vectors is 2.6% of the stimulus, and nothing says whether the
    remaining 97% would have diverged once or ten thousand times. That is the
    difference between "one wrong beat" and "the memory never writes", and it is
    the first thing a human wants to know.

    This does NOT change the verdict: one divergence has always been a FAIL and
    still is. It only adds ``divergence_count`` and the first
    :func:`gate_sim_max_divergences` mismatches to ``detail``. The default
    verdict JSON is byte-for-byte unchanged, because the driver is generated
    without the debug code at all when this is off.
    """
    return _flag(GATE_SIM_DEBUG_ENV, False)


def gate_sim_max_divergences() -> int:
    """How many individual mismatches debug mode records in full (default 32).

    Bounded on purpose: a design whose memory never writes diverges on every
    output beat, and 200000 recorded mismatches is not a report, it is a
    denial of service on the reader. The COUNT is always exact.
    """
    try:
        return max(1, int(os.environ.get("CORESMITH_GATE_SIM_MAX_DIVERGENCES", "") or 32))
    except ValueError:
        return 32


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

STATUS_PASS = "pass"
STATUS_FAIL = "fail"
STATUS_NOT_RUN = "not_run"
STATUS_DISABLED = "disabled"


@dataclass
class GateSimResult:
    """Verdict of one gate-level simulation.

    ``status`` is the authoritative field. ``ok`` is a convenience that is True
    only for ``pass`` and for a non-blocking ``not_run``/``disabled``; it is
    NEVER True for ``fail``. Readers that treat a missing result as success are
    the exact bug this module exists to prevent, so ``status`` is always set and
    ``STATUS_NOT_RUN`` always carries a ``reason``.
    """

    ran: bool = False
    ok: bool = True
    status: str = STATUS_NOT_RUN
    reason: str = ""
    netlist_path: str = ""
    cycles_compared: int = 0
    output_bits_compared: int = 0
    first_divergence: dict = field(default_factory=dict)
    detail: dict = field(default_factory=dict)

    @property
    def blocking(self) -> bool:
        return self.status == STATUS_FAIL

    def as_dict(self) -> dict:
        return {
            "ran": self.ran,
            "ok": self.ok,
            "status": self.status,
            "reason": self.reason,
            "netlist_path": self.netlist_path,
            "cycles_compared": self.cycles_compared,
            "output_bits_compared": self.output_bits_compared,
            "first_divergence": self.first_divergence,
            "detail": self.detail,
        }

    def as_prev_error(self, block: str = "") -> str:
        where = f" in {block}" if block else ""
        div = self.first_divergence or {}
        lines = [
            f"GATE-LEVEL SIMULATION FAILURE{where}: the SYNTHESIZED NETLIST does "
            f"not reproduce the behaviour the RTL was verified with.",
            "",
            f"  reason           : {self.reason}",
            f"  netlist          : {self.netlist_path}",
            f"  cycles compared  : {self.cycles_compared}",
            f"  output bits cmp'd: {self.output_bits_compared}",
        ]
        if div:
            lines += [
                "",
                f"  FIRST DIVERGENCE at cycle {div.get('cycle')}",
                f"    port     : {div.get('port')}",
                f"    expected : {div.get('expected')}   (RTL reference)",
                f"    actual   : {div.get('actual')}   (gate netlist)",
            ]
        # Only present when the run was made under CORESMITH_GATE_SIM_DEBUG: the
        # default driver stops at the first mismatch, so it cannot know a total.
        total = (self.detail or {}).get("divergence_count")
        if total:
            lines.append(f"    total    : {total} diverging port-cycles "
                         f"(CORESMITH_GATE_SIM_DEBUG kept comparing)")
        lines += [
            "",
            "WHAT THIS MEANS: RTL DV, coverage and every PPA/PnR number were "
            "measured on DIFFERENT HARDWARE. The RTL you simulated and the "
            "netlist you are taping out are not the same design.",
            "",
            "TYPICAL CAUSES, most likely first:",
            "  1. The module is SPLIT by a preprocessor guard (`ifdef/`ifndef) "
            "into a simulation implementation and a synthesis implementation. "
            "Write EXACTLY ONE implementation per module; a guard may only ever "
            "wrap non-functional debug/trace/assertion code.",
            "  2. Functional logic sits behind a synthesis pragma "
            "(translate_off / synthesis off) so synthesis dropped it.",
            "  3. A leaf the design depends on was black-boxed at synthesis "
            "(missing source, missing library read) and drives nothing.",
            "  4. An output is written only from an `initial` block or relies on "
            "simulation-only initialization that synthesis does not implement.",
            "  5. Non-synthesizable constructs silently reduced by the mapper "
            "(unbounded loops, real/time arithmetic, hierarchical references).",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Netlist port parsing
# ---------------------------------------------------------------------------


@dataclass
class Port:
    name: str
    direction: str  # "input" | "output" | "inout"
    width: int = 1
    msb: int = 0
    lsb: int = 0


_MODULE_DECL_RE = re.compile(r"^[ \t]*module[ \t]+([A-Za-z_][\w$]*)", re.MULTILINE)
_INSTANCE_RE = re.compile(
    r"^[ \t]*([A-Za-z_][\w$]*)[ \t]+(?:#\s*\([^;]*?\)\s*)?[A-Za-z_\\][\w$\\.\[\]]*[ \t]*\(",
    re.MULTILINE,
)


def resolve_netlist_top(netlist_text: str) -> Optional[str]:
    """Return the netlist's top module, derived from STRUCTURE not from a name.

    The top is the one declared module that no other module instantiates. This
    is deterministic and identifier-agnostic, which is the point: a block's
    architectural name is chosen by the architecture agent and is NOT a stable
    key. Keying a signoff gate off that name made it fail on a correctly
    synthesized netlist whenever the name diverged from the mandated module
    name (a Caravel ``user_project_wrapper``, a vendor-locked top).

    Returns ``None`` when the answer is ambiguous -- zero candidates, or more
    than one -- so the caller can FAIL rather than guess. A netlist with two
    un-instantiated modules is not something to pick a winner from.
    """
    declared = set(_MODULE_DECL_RE.findall(netlist_text))
    if not declared:
        return None
    instantiated = {m for m in _INSTANCE_RE.findall(netlist_text) if m in declared}
    roots = declared - instantiated
    if len(roots) == 1:
        return roots.pop()
    return None


_MODULE_RE_TMPL = r"^[ \t]*module[ \t]+{top}[ \t]*\("
_PORT_DECL_RE = re.compile(
    r"^\s*(input|output|inout)\b\s*(?:wire|reg|logic|signed|\s)*"
    r"(?:\[\s*(\d+)\s*:\s*(\d+)\s*\]\s*)?([A-Za-z_\\][\w$]*)\s*;",
    re.MULTILINE,
)


def parse_netlist_ports(netlist_text: str, top: str) -> list[Port]:
    """Return the top module's ports, in declaration order.

    Handles the Yosys ``write_verilog`` shape (``module top(a, b, c);`` followed
    by separate ``input``/``output`` declarations). Returns ``[]`` when the top
    module is not present -- the caller treats that as a FAIL, not a skip.
    """
    m = re.search(_MODULE_RE_TMPL.format(top=re.escape(top)), netlist_text, re.MULTILINE)
    if not m:
        return []
    body = netlist_text[m.start():]
    end = re.search(r"^\s*endmodule\b", body, re.MULTILINE)
    if end:
        body = body[: end.start()]
    ports: list[Port] = []
    seen: set[str] = set()
    for d in _PORT_DECL_RE.finditer(body):
        direction, msb, lsb, name = d.group(1), d.group(2), d.group(3), d.group(4)
        if name in seen:
            continue
        seen.add(name)
        if msb is None:
            ports.append(Port(name=name, direction=direction, width=1, msb=0, lsb=0))
        else:
            hi, lo = int(msb), int(lsb)
            ports.append(
                Port(name=name, direction=direction, width=abs(hi - lo) + 1,
                     msb=hi, lsb=lo)
            )
    return ports


# ---------------------------------------------------------------------------
# Power/ground boundary nets
# ---------------------------------------------------------------------------
#
# A tri-state SIGNAL boundary genuinely defeats vector replay: nothing tells the
# driver when the pin is an input and when it is an output, so neither driving
# nor comparing it is honest. A POWER RAIL declared ``inout`` is not that. It is
# a constant, it carries no behaviour, and it is ``inout`` only because that is
# the convention every PDK/harness uses for supplies.
#
# Refusing on any ``inout`` therefore made the gate structurally unable to judge
# the one artifact it exists for: a Caravel ``user_project_wrapper``, whose ONLY
# bidirectional ports are ``vdda1 vdda2 vssa1 vssa2 vccd1 vccd2 vssd1 vssd2``.
# That is the artifact the external grader drives, so "no honest verdict is
# available" was a hole exactly where the gate is supposed to be strongest.
#
# Conservative on purpose: recognition is by EXACT name against a curated list,
# not by prefix or substring. An unrecognised inout is a SIGNAL and still vetoes
# the gate -- mis-classifying a real tri-state as a rail would tie it to a
# constant and then not compare it, i.e. fabricate a pass.
_SUPPLY_INOUTS = frozenset({
    "vdd", "vdd1v8", "vdd3v3", "vdda", "vdda1", "vdda2", "vddio", "vddd",
    "vccd", "vccd1", "vccd2", "vccd3",
    "vpwr", "vpb",          # cell-level views (VPWR / p-substrate bias)
})
_GROUND_INOUTS = frozenset({
    "vss", "vssa", "vssa1", "vssa2", "vssio", "vssd", "vssd1", "vssd2", "vssd3",
    "gnd", "vgnd", "vnb",   # cell-level views (VGND / n-well bias)
})


def power_inout_level(name: str) -> Optional[int]:
    """``1`` for a supply rail, ``0`` for a ground rail, ``None`` otherwise.

    ``None`` means "this is a signal", which is the safe answer: the caller then
    declines to render a verdict at all. Only add a name here when it is a rail
    in every context, because the consequence of a wrong entry is a pass that
    ignored a real port.
    """
    key = str(name).lstrip("\\").strip().lower()
    if key in _SUPPLY_INOUTS:
        return 1
    if key in _GROUND_INOUTS:
        return 0
    return None


def inout_is_inert(netlist_text: str, name: str) -> bool:
    """True when the netlist PROVES this inout carries no behaviour.

    Every reference must be the port's own declaration (port list, `inout`,
    `wire`) or a constant all-Z assign -- the exact form yosys emits for an
    unused mandated pad bus (`assign analog_io = 29'hzzzzzzzz;`). One reference
    that is anything else (an instance connection, a real driver, a read) means
    the port participates in the design and the honest answer stays "cannot
    judge".

    Structural on purpose. A name list ("analog_io is fine") would silently
    bless a design that actually drives its analog pads.
    """
    ref = re.compile(r"^.*\b" + re.escape(name) + r"\b.*$", re.MULTILINE)
    decl = re.compile(
        r"^\s*(?:module\b|(?:inout|input|output|wire|tri|logic)\b)")
    zassign = re.compile(
        r"^\s*assign\s+\\?" + re.escape(name) +
        r"(?:\s*\[[^\]]*\])?\s*=\s*\d*'[hbodHBOD]?[zZ_]+\s*;")
    # A same-name hierarchical pass-through (`.analog_io(analog_io),`) renames
    # nothing and drives nothing: the parent hands the mandated pad bus to the
    # leaf unchanged. Any OTHER connection shape (renamed net, expression,
    # concatenation, slice) still disqualifies.
    passthrough = re.compile(
        r"^\s*\.\\?" + re.escape(name) +
        r"\s*\(\s*\\?" + re.escape(name) + r"\s*\)\s*,?\s*$")
    for m in ref.finditer(netlist_text):
        line = m.group(0)
        if decl.match(line) or zassign.match(line) or passthrough.match(line):
            continue
        return False
    return True


def split_inouts(
    ports: list[Port], netlist_text: str = "",
) -> tuple[list[Port], list[Port], list[Port]]:
    """``(power_rails, inert, signal_inouts)`` among the bidirectional ports.

    ``inert`` are inouts the netlist proves carry no behaviour (see
    :func:`inout_is_inert`) -- mandated-but-unused pad buses like Caravel's
    ``analog_io``. They are neither driven nor compared, and excluding them
    fabricates nothing because there is nothing there.

    A rail wider than one bit is not a rail we understand, so it is reported as
    a signal rather than tied to a guessed constant.
    """
    rails: list[Port] = []
    inert: list[Port] = []
    signals: list[Port] = []
    for p in ports:
        if p.direction != "inout":
            continue
        if p.width == 1 and power_inout_level(p.name) is not None:
            rails.append(p)
        elif netlist_text and inout_is_inert(netlist_text, p.name):
            inert.append(p)
        else:
            signals.append(p)
    return rails, inert, signals


def pick_clock(ports: list[Port], hint: str = "") -> Optional[Port]:
    """Choose the clock input. Prefers an explicit hint, then the conventional
    names, then a single-bit input whose name contains ``clk``/``clock``."""
    ins = [p for p in ports if p.direction == "input" and p.width == 1]
    if hint:
        for p in ins:
            if p.name == hint:
                return p
    for want in ("clk", "clock", "i_clk", "clk_i", "sys_clk"):
        for p in ins:
            if p.name == want:
                return p
    for p in ins:
        low = p.name.lower()
        if "clk" in low or "clock" in low:
            return p
    return None


# ---------------------------------------------------------------------------
# VCD -> per-cycle port vectors
# ---------------------------------------------------------------------------


@dataclass
class Vectors:
    """Recorded pre-posedge steady state of every top-level port."""

    cycles: int = 0
    inputs: list[str] = field(default_factory=list)   # input port names, ordered
    outputs: list[str] = field(default_factory=list)  # output port names, ordered
    widths: dict = field(default_factory=dict)        # name -> width
    rows: list = field(default_factory=list)          # per cycle: (in_vals, out_vals)
    output_activity: int = 0                          # distinct output tuples seen


_VCD_VAR_RE = re.compile(
    r"^\$var\s+\S+\s+(\d+)\s+(\S+)\s+([^\s\[]+)(?:\s*\[[^\]]*\])?\s*\$end", re.MULTILINE
)


def extract_vectors(
    vcd_path: str | Path,
    ports: list[Port],
    clock: str,
    max_cycles: int,
) -> Vectors:
    """Sample every top-level port immediately BEFORE each clock posedge.

    Only top-scope variables are considered, so a shallow trace
    (``--trace-depth 1``) keeps this linear and small. Values are kept as
    binary strings (MSB first, no ``b`` prefix) of exactly the port width.
    Unknown/high-Z bits are preserved as ``x``/``z`` so the driver can decide
    (inputs: drive 0; outputs: not compared) -- and the count of ACTUALLY
    compared bits is reported so a run that compared nothing fails closed.
    """
    p = Path(vcd_path)
    if not p.exists() or p.stat().st_size == 0:
        return Vectors()

    by_name = {port.name: port for port in ports}
    want = set(by_name)
    text = p.read_text(errors="ignore")

    # --- header: id -> name (top scope only; first definition wins) ---------
    header_end = text.find("$enddefinitions")
    header = text[: header_end if header_end > 0 else len(text)]
    depth = 0
    id_of: dict[str, str] = {}
    width_of_id: dict[str, int] = {}
    for line in header.splitlines():
        s = line.strip()
        if s.startswith("$scope"):
            depth += 1
            continue
        if s.startswith("$upscope"):
            depth -= 1
            continue
        if not s.startswith("$var"):
            continue
        # cocotb/Verilator wrap the DUT in one scope; accept depth 1 and 2 so a
        # bare-top VCD and a wrapped one both work.
        if depth > 2:
            continue
        m = _VCD_VAR_RE.match(s)
        if not m:
            continue
        width, ident, name = int(m.group(1)), m.group(2), m.group(3)
        if name in want and name not in id_of.values():
            id_of[ident] = name
            width_of_id[ident] = width

    if clock not in id_of.values():
        return Vectors()

    ins = [pt.name for pt in ports if pt.direction == "input" and pt.name != clock]
    outs = [pt.name for pt in ports if pt.direction == "output"]
    widths = {pt.name: pt.width for pt in ports}
    vec = Vectors(inputs=ins, outputs=outs, widths=widths)

    cur: dict[str, str] = {n: "x" * widths[n] for n in widths}
    prev_clk = "x"
    rows: list = []
    seen_out: set = set()
    pending: list = []          # value changes buffered for the current stamp

    def _flush() -> bool:
        """Apply one timestamp's changes. Snapshots the PRE-EDGE state first
        when this timestamp carries a clock 0->1.

        A VCD groups every change at a time under one ``#T`` header with NO
        defined ordering inside the block, so the clock's rising edge and the
        outputs it produced appear together. Sampling as lines stream past
        would therefore capture POST-edge outputs whenever they happen to be
        listed before the clock line -- an off-by-one that makes a correct
        netlist look one cycle late. Snapshotting before applying the block is
        the only ordering-independent reading.
        """
        nonlocal prev_clk
        edge = False
        for _name, _val in pending:
            if _name == clock:
                new_clk = _val[-1] if _val else "x"
                if prev_clk == "0" and new_clk == "1":
                    edge = True
        if edge:
            in_vals = [_fit(cur[n], widths[n]) for n in ins]
            out_vals = [_fit(cur[n], widths[n]) for n in outs]
            rows.append((in_vals, out_vals))
            seen_out.add(tuple(out_vals))
        for _name, _val in pending:
            cur[_name] = _val
            if _name == clock:
                prev_clk = _val[-1] if _val else "x"
        pending.clear()
        return len(rows) >= max_cycles

    body = text[header_end:] if header_end > 0 else text
    stop = False
    for raw in body.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line[0] == "#":
            if _flush():
                stop = True
                break
            continue
        if line[0] == "$":
            continue
        if line[0] in "bB":
            sp = line.find(" ")
            if sp < 0:
                continue
            val, ident = line[1:sp], line[sp + 1:].strip()
        elif line[0] in "rR":
            continue
        else:
            val, ident = line[0], line[1:].strip()
        name = id_of.get(ident)
        if name is None:
            continue
        pending.append((name, val))
    if not stop:
        _flush()

    vec.rows = rows
    vec.cycles = len(rows)
    vec.output_activity = len(seen_out)
    return vec


def _fit(val: str, width: int) -> str:
    """Left-extend a VCD value string to ``width`` bits per VCD rules (extend
    with the MSB when it is x/z/0, else with 0)."""
    if not val:
        return "x" * width
    if len(val) >= width:
        return val[-width:]
    pad = val[0] if val[0] in "xzXZ" else "0"
    return pad * (width - len(val)) + val


def write_vector_file(vec: Vectors, path: str | Path, top: str, clock: str) -> None:
    """Serialize vectors as a self-describing text file the C++ driver reads."""
    p = Path(path)
    lines = [
        "# coresmith gate-sim vectors v1",
        f"# top {top}",
        f"# clock {clock}",
        f"# cycles {vec.cycles}",
        "# in " + " ".join(f"{n}:{vec.widths[n]}" for n in vec.inputs),
        "# out " + " ".join(f"{n}:{vec.widths[n]}" for n in vec.outputs),
    ]
    for in_vals, out_vals in vec.rows:
        lines.append(" ".join(in_vals) + " | " + " ".join(out_vals))
    p.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# PDK cell models + the UDP shim
# ---------------------------------------------------------------------------

# The ten UDPs the sky130 hd cell models reference under -DFUNCTIONAL with
# USE_POWER_PINS undefined. Verified against the shipped primitives.v.
_SHIM_UDPS = (
    "sky130_fd_sc_hd__udp_dff$P",
    "sky130_fd_sc_hd__udp_dff$PR",
    "sky130_fd_sc_hd__udp_dff$PS",
    "sky130_fd_sc_hd__udp_dff$NSR",
    "sky130_fd_sc_hd__udp_dlatch$P",
    "sky130_fd_sc_hd__udp_dlatch$lP",
    "sky130_fd_sc_hd__udp_dlatch$PR",
    "sky130_fd_sc_hd__udp_mux_2to1",
    "sky130_fd_sc_hd__udp_mux_2to1_N",
    "sky130_fd_sc_hd__udp_mux_4to2",
)


_SHIM_TEMPLATE = """
module {p}__udp_dff$P (output reg Q, input D, input CLK);
  always @(posedge CLK) Q <= D;
endmodule

module {p}__udp_dff$PR (output reg Q, input D, input CLK, input RESET);
  always @(posedge CLK or posedge RESET) if (RESET) Q <= 1'b0; else Q <= D;
endmodule

module {p}__udp_dff$PS (output reg Q, input D, input CLK, input SET);
  always @(posedge CLK or posedge SET) if (SET) Q <= 1'b1; else Q <= D;
endmodule

// SET dominates RESET; negative-edge clock (CLK_N).
module {p}__udp_dff$NSR (output reg Q, input SET, input RESET,
                         input CLK_N, input D);
  always @(negedge CLK_N or posedge SET or posedge RESET)
    if (SET) Q <= 1'b1; else if (RESET) Q <= 1'b0; else Q <= D;
endmodule

module {p}__udp_dlatch$P (output reg Q, input D, input GATE);
  always @(*) if (GATE) Q = D;
endmodule

module {p}__udp_dlatch$lP (output reg Q, input D, input GATE);
  always @(*) if (GATE) Q = D;
endmodule

module {p}__udp_dlatch$PR (output reg Q, input D, input GATE, input RESET);
  always @(*) if (RESET) Q = 1'b0; else if (GATE) Q = D;
endmodule

module {p}__udp_mux_2to1 (output X, input A0, input A1, input S);
  assign X = S ? A1 : A0;
endmodule

module {p}__udp_mux_2to1_N (output Y, input A0, input A1, input S);
  assign Y = ~(S ? A1 : A0);
endmodule

module {p}__udp_mux_4to2 (output X, input A0, input A1, input A2, input A3,
                          input S0, input S1);
  assign X = S1 ? (S0 ? A3 : A2) : (S0 ? A1 : A0);
endmodule
"""


def udp_shim_source(prefixes: tuple = ("sky130_fd_sc_hd",)) -> str:
    """Verilator-compatible behavioural replacements for the sky130 UDPs.

    Each body is a direct transcription of the corresponding ``primitive``
    truth table in the PDK's ``primitives.v``; port ORDER matches the primitive
    exactly because the cell models bind these positionally. Sequential cells
    use non-blocking assignment, which is what makes zero-delay (no
    ``--timing``, no ``UNIT_DELAY``) gate simulation race-free -- the reason we
    do not need the PDK's ``#UNIT_DELAY`` annotations.

    We ship replacements rather than compiling ``primitives.v`` because
    Verilator cannot compile Verilog-1995 UDP ``table`` bodies at all, and its
    ``--bbox-unsup`` escape hatch would black-box every flop and latch in the
    design -- turning the gate sim into a guaranteed false PASS.

    ``prefixes`` names the cell-library families to emit for (``hd``, ``hvl``,
    ...). Every family in the sky130 standard-cell set uses the same UDP names
    and semantics under its own prefix, so one template serves them all;
    emitting an unreferenced replacement is harmless.
    """
    parts = [
        "// Generated by coresmith harness.gate_sim -- Verilator-compatible",
        "// behavioural stand-ins for the sky130 UDP primitives referenced by",
        "// the cell models under -DFUNCTIONAL with USE_POWER_PINS undefined.",
        "// Port order matches primitives.v exactly (cells bind positionally).",
        "`default_nettype wire",
        "`timescale 1ns / 1ps",
    ]
    for pfx in prefixes:
        parts.append(_SHIM_TEMPLATE.format(p=pfx))
    return "\n".join(parts) + "\n"


# Standard-cell families we know how to shim, in preference order.
CELL_LIBRARIES = ("sky130_fd_sc_hd", "sky130_fd_sc_hvl", "sky130_fd_sc_ms",
                  "sky130_fd_sc_ls", "sky130_fd_sc_lp", "sky130_fd_sc_hs")


def cell_model_files(
    pdk_root: str | Path | None = None, netlist_text: str = "",
) -> tuple:
    """Locate the PDK standard-cell SIMULATION models (not the blackbox views).

    Returns ``(files, prefixes)``. When ``netlist_text`` is supplied only the
    libraries the netlist ACTUALLY references are returned -- compiling an
    unreferenced family is both wasted work and a source of spurious
    elaboration errors. Returns ``([], ())`` when the PDK is absent; the caller
    degrades to ``not_run`` (or FAIL under strict), never to a pass.
    """
    if pdk_root is None:
        try:
            from orchestrator.langgraph.pipeline_helpers import PDK_ROOT
            pdk_root = PDK_ROOT
        except Exception:
            return [], ()
    root = Path(pdk_root)
    wanted = [
        lib for lib in CELL_LIBRARIES
        if (not netlist_text) or re.search(rf"\b{re.escape(lib)}__", netlist_text)
    ]
    if not wanted:
        # A netlist that references no known family (e.g. everything optimized
        # to constants) still has to build: fall back to the primary core
        # library only, never the whole set.
        wanted = [CELL_LIBRARIES[0]]
    for variant in ("sky130A", "sky130B"):
        files, prefixes = [], []
        for lib in wanted:
            f = root / variant / "libs.ref" / lib / "verilog" / f"{lib}.v"
            if f.exists():
                files.append(str(f))
                prefixes.append(lib)
        if files:
            return files, tuple(prefixes)
    return [], ()


# ---------------------------------------------------------------------------
# Hard-macro simulation models
# ---------------------------------------------------------------------------


# The stand-in's INTERFACE is read from the real macro's own Verilog, never
# computed from registry metadata. Arithmetic on metadata got this wrong twice on
# one design, in opposite directions:
#
#   * ``sram_1rw1r_8_4096_8`` (word_size == write_size == 8) declares NO
#     ``wmask0`` port at all -- one whole-word mask bit IS the write enable, so
#     OpenRAM legitimately omits it. The old generator emitted ``wmask0``
#     unconditionally and gated writes on ``wmask0[0]``. The flat netlist
#     correctly leaves the pin unconnected, Verilator ties the floating input
#     low, EVERY write is suppressed, and every read returns zero -- a
#     confident gate-sim FAIL on a correctly synthesized netlist.
#   * ``sram_1rw1r_9_4096_8`` DOES declare ``wmask0``, as ``[1:0]``
#     (NUM_WMASKS == ceil(9/8) == 2). The old generator computed ``9 // 8 == 1``
#     and declared ``[0:0]``, so the netlist drove ``2'h3`` into a 1-bit port.
#     Harmless only because that mask happens to be a constant.
#
# Both are one defect: a stand-in whose interface is GUESSED. The same guess also
# hard-coded active-low ``csb0`` while the OpenROM mask-ROM models declare
# active-HIGH ``cs0``, which would have inverted every ROM access.
#
# So: read the ports, the widths, the select polarity, the mask slices and the
# ROM contents out of the macro. When any of that cannot be read, return "" --
# the caller reports the macro as unresolved, which is a FAIL ("refusing to
# simulate a design whose memories are undriven"). Never fall back to the guess.

_MACRO_COMMENT_RE = re.compile(r"//[^\n]*|/\*.*?\*/", re.DOTALL)
_MACRO_SUBPROG_RE = re.compile(
    r"\bfunction\b.*?\bendfunction\b|\btask\b.*?\bendtask\b", re.DOTALL)
_MACRO_PARAM_RE = re.compile(
    r"\b(?:parameter|localparam)\b\s*(?:integer|signed)?\s*(?:\[[^\]]*\]\s*)?"
    r"([A-Za-z_]\w*)\s*=\s*([^;,]+)")
_MACRO_PORT_DECL_RE = re.compile(
    r"\b(input|output|inout)\b\s+(?:(?:wire|reg|logic|signed)\s+)*"
    r"(?:\[\s*([^:\]]+?)\s*:\s*([^\]]+?)\s*\]\s*)?([A-Za-z_]\w*)\s*(?=[;,)])")
_MACRO_READMEM_RE = re.compile(r"\$readmem([bh])\s*\(\s*\"([^\"]+)\"")
# The PDK model's own write-merge function is the GROUND TRUTH for which data
# bits each mask bit covers -- including a partial final group (a 9-bit word
# with write_size 8 has mask bit 1 covering exactly ``[8:8]``).
_MACRO_MERGE_RE = re.compile(
    r"if\s*\(\s*write_mask\s*\[\s*(\d+)\s*\]\s*\)\s*"
    r"merge_write\d+\s*\[\s*(\d+)\s*:\s*(\d+)\s*\]")
_MACRO_IDENT_RE = re.compile(r"[A-Za-z_]\w*")
_MACRO_LITERAL_RE = re.compile(r"\b(\d+)'\s*([bodhBODH])\s*([0-9a-fA-FxzXZ_]+)")
_MACRO_SAFE_EXPR_RE = re.compile(r"^[0-9+\-*/%()<>\s]+$")
# Roles the generator understands. ``csb`` before ``cs`` so the active-low form
# is not mis-read as the active-high one -- getting that backwards yields a
# memory selected exactly when it should be idle.
_MACRO_ROLE_RE = re.compile(r"^(clk|csb|cs|web|wmask|addr|din|dout)(\d+)$")


def _macro_eval_int(expr: str, params: dict) -> Optional[int]:
    """Evaluate one small integer expression out of a macro's own source.

    Handles what OpenRAM/OpenROM actually write (``64``, ``1 << ADDR_WIDTH``,
    ``DATA_WIDTH-1``, sized literals). Returns ``None`` for anything else --
    an unresolvable width must propagate as "unknown", never as a default.
    """
    e = str(expr).strip().rstrip(";").strip()
    if not e:
        return None

    def _lit(m: re.Match) -> str:
        base = {"b": 2, "o": 8, "d": 10, "h": 16}[m.group(2).lower()]
        try:
            return str(int(m.group(3).replace("_", ""), base))
        except ValueError:
            return "?"

    e = _MACRO_LITERAL_RE.sub(_lit, e)

    def _ident(m: re.Match) -> str:
        v = params.get(m.group(0))
        return f"({v})" if isinstance(v, int) else "?"

    e = _MACRO_IDENT_RE.sub(_ident, e)
    if "?" in e or not _MACRO_SAFE_EXPR_RE.match(e):
        return None
    # Verilog `/` is integer division; every identifier is already substituted,
    # so what is left is digits and operators only.
    e = e.replace("/", "//")
    try:
        val = eval(e, {"__builtins__": {}}, {})  # noqa: S307 - whitelisted chars
    except Exception:  # noqa: BLE001 - any arithmetic error means "unknown"
        return None
    return int(val) if isinstance(val, int) and not isinstance(val, bool) else None


def _macro_params(text: str) -> dict:
    """``{name: int}`` for every parameter whose value we can evaluate.

    Iterated because a parameter may reference one declared later; a parameter we
    cannot evaluate is simply absent, which only matters if a port width needs it.
    """
    params: dict = {}
    pending = _MACRO_PARAM_RE.findall(text)
    for _ in range(6):
        rest = []
        for nm, expr in pending:
            val = _macro_eval_int(expr, params)
            if val is None:
                rest.append((nm, expr))
            else:
                params[nm] = val
        if not rest or len(rest) == len(pending):
            break
        pending = rest
    return params


def _macro_balanced(text: str, open_at: int) -> tuple[str, int]:
    """Text inside the parens starting at ``open_at``, and the index past them."""
    depth = 0
    for i in range(open_at, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return text[open_at + 1:i], i + 1
    return "", len(text)


def _macro_split_power_guard(header: str) -> tuple[str, str]:
    """Split a module header into (unguarded, ``USE_POWER_PINS``-guarded) text.

    The supplies live behind ```ifdef USE_POWER_PINS`` and are NOT part of the
    functional port list; treating them as ordinary ports would declare pins the
    netlist never connects.
    """
    plain: list[str] = []
    guarded: list[str] = []
    stack: list[bool] = []
    for line in header.splitlines():
        s = line.strip()
        if s.startswith("`"):
            tok = s[1:].split()
            kw = tok[0] if tok else ""
            if kw in ("ifdef", "ifndef"):
                arg = tok[1] if len(tok) > 1 else ""
                stack.append(arg == "USE_POWER_PINS" and kw == "ifdef")
            elif kw == "else" and stack:
                stack[-1] = not stack[-1]
            elif kw == "endif" and stack:
                stack.pop()
            continue
        (guarded if any(stack) else plain).append(line)
    return "\n".join(plain), "\n".join(guarded)


def _macro_header_names(region: str) -> list[str]:
    """Port names from a module header region, in declaration order.

    Serves both header styles: non-ANSI (``clk0,csb0,addr0``) and ANSI
    (``input wire [11:0] addr0``) -- in both, the LAST identifier of each
    comma-separated item is the port name.
    """
    out: list[str] = []
    depth = 0
    item: list[str] = []
    items: list[str] = []
    for ch in region:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            items.append("".join(item))
            item = []
            continue
        item.append(ch)
    items.append("".join(item))
    for it in items:
        ids = _MACRO_IDENT_RE.findall(it)
        if ids and ids[-1] not in out:
            out.append(ids[-1])
    return out


@dataclass
class MacroIface:
    """One hard macro's DECLARED interface, read from its own Verilog model.

    Everything here is a fact stated by the macro, not a derivation from a name
    or from registry metadata. ``ranges`` keeps the declared ``[msb:lsb]`` (not
    just the width) so the stand-in's declaration is textually equivalent to the
    real one.
    """

    name: str = ""
    order: list = field(default_factory=list)      # port names, decl order
    dirs: dict = field(default_factory=dict)       # name -> input/output/inout
    ranges: dict = field(default_factory=dict)     # name -> (msb, lsb) or None
    widths: dict = field(default_factory=dict)     # name -> declared width
    power_pins: list = field(default_factory=list)  # USE_POWER_PINS-guarded
    mask_slices: list = field(default_factory=list)  # [(hi, lo)] per mask bit
    contents: str = ""                             # $readmem file, absolute
    contents_radix: str = ""                       # "b" | "h"


def macro_interface(name: str, verilog_path: str | Path) -> Optional[MacroIface]:
    """Read one macro's declared interface out of its own Verilog model.

    Returns ``None`` when the model cannot be read, its module cannot be found,
    or any declared port's width cannot be resolved. ``None`` must degrade to
    "macro unresolved" -> FAIL; it must never degrade to a guessed interface.

    The port list is the module header's own, in declaration order. It is then
    CROSS-CHECKED against
    :func:`orchestrator.langgraph.macro_prebind.macro_ports` -- the same function
    the pre-synthesis binder uses to decide which pins to CONNECT. If the binder
    cannot see a port the header declares, the netlist will not connect it and
    the stand-in would be modelling a different interface than the one being
    simulated, so that disagreement returns ``None``. Note it must NOT resolve by
    dropping the port: an absent input ties low, which is how every write got
    silently suppressed in the first place.
    """
    p = Path(verilog_path)
    if not str(verilog_path) or not p.is_file():
        return None
    try:
        raw = p.read_text(errors="ignore")
    except OSError:
        return None

    try:
        from orchestrator.langgraph.macro_prebind import macro_ports
    except Exception:  # pragma: no cover - import guard
        return None
    declared = macro_ports(p)
    if not declared:
        return None

    text = _MACRO_COMMENT_RE.sub(" ", raw)
    m = re.search(r"\bmodule\s+%s\b" % re.escape(name), text)
    if not m:
        return None
    open_at = text.find("(", m.end())
    if open_at < 0:
        return None
    header, past = _macro_balanced(text, open_at)
    body = _MACRO_SUBPROG_RE.sub(" ", text[past:])
    plain_hdr, guarded_hdr = _macro_split_power_guard(header)

    iface = MacroIface(name=name)
    iface.order = _macro_header_names(plain_hdr)
    if not iface.order or any(n not in declared for n in iface.order):
        return None
    iface.power_pins = [n for n in _macro_header_names(guarded_hdr)
                        if n in declared]

    params = _macro_params(text)
    # Declarations may be in the header (ANSI) or the body (non-ANSI); scan both.
    # Subprograms are stripped from the body first so a function's own `input`
    # declarations cannot be mistaken for ports.
    for d in _MACRO_PORT_DECL_RE.finditer(plain_hdr + "\n" + body):
        direction, msb, lsb, nm = d.group(1), d.group(2), d.group(3), d.group(4)
        if nm not in iface.order or nm in iface.widths:
            continue
        if msb is None:
            iface.dirs[nm] = direction
            iface.ranges[nm] = None
            iface.widths[nm] = 1
            continue
        hi, lo = _macro_eval_int(msb, params), _macro_eval_int(lsb, params)
        if hi is None or lo is None:
            return None          # a width we cannot resolve is not a width
        iface.dirs[nm] = direction
        iface.ranges[nm] = (hi, lo)
        iface.widths[nm] = abs(hi - lo) + 1
    if any(nm not in iface.widths for nm in iface.order):
        return None              # a port with no resolvable declaration

    for bit, hi, lo in _MACRO_MERGE_RE.findall(text):
        iface.mask_slices.append((int(bit), int(hi), int(lo)))
    iface.mask_slices = [(hi, lo) for _b, hi, lo in
                         sorted(iface.mask_slices, key=lambda t: t[0])]

    rm = _MACRO_READMEM_RE.search(text)
    if rm:
        iface.contents_radix = rm.group(1)
        cand = Path(rm.group(2))
        if not cand.is_absolute():
            cand = p.parent / cand
        iface.contents = str(cand)
    return iface


def _macro_range_str(rng) -> str:
    return "" if rng is None else f"[{rng[0]}:{rng[1]}]"


def macro_model_source(name: str, ports: str = "", data_bits: int = 0,
                       words: int = 0, mask_bits: int = 0,
                       iface: Optional[MacroIface] = None) -> str:
    """Generate a cycle-accurate simulation model for one hard memory macro.

    Reproduces the OpenRAM behavioural model's CYCLE semantics -- inputs
    sampled at the port clock's posedge, read data presented for sampling at
    the following posedge, data held while the port is deselected -- with no
    ``#`` delays and no ``$display``. That removes the posedge-X / negedge-
    redrive race that makes the PDK's own model unusable in a zero-delay
    simulator, and the per-access tracing that makes it unusable at scale.

    ``iface`` (from :func:`macro_interface`) is REQUIRED and is the only source
    of the port list, the port widths, the select polarity, the write-mask
    slicing and the memory depth. ``ports``/``data_bits``/``words``/``mask_bits``
    are registry metadata used for the header comment and as the fallback
    mask granularity only -- they never decide what the interface looks like.
    See the comment above :func:`macro_interface` for the two false verdicts
    that deriving the interface arithmetically produced.

    Returns ``""`` -- "unresolved", which the caller turns into a FAIL -- when
    there is no ``iface``, when the macro declares a port this generator does not
    understand, when the geometry is inconsistent, or when a read-only memory's
    contents cannot be located. Guessing in any of those cases produces a
    confident verdict about hardware that was never simulated.
    """
    if iface is None or not iface.order:
        return ""

    # --- classify every declared port; refuse on anything unrecognised -------
    roles: dict = {}
    for nm in iface.order:
        m = _MACRO_ROLE_RE.match(nm)
        if not m:
            return ""            # an unknown pin means we do not know this macro
        role, idx = m.group(1), int(m.group(2))
        if role in roles.setdefault(idx, {}):
            return ""            # duplicate declaration
        roles[idx][role] = nm
    if not roles:
        return ""

    def _w(nm: str) -> int:
        return int(iface.widths.get(nm, 0) or 0)

    # --- geometry, entirely from the declarations ----------------------------
    data_names = [r[k] for r in roles.values() for k in ("din", "dout") if k in r]
    addr_names = [r["addr"] for r in roles.values() if "addr" in r]
    if not data_names or not addr_names:
        return ""
    if len({_w(n) for n in data_names}) != 1:
        return ""                # din/dout disagree: not a memory we understand
    if len({_w(n) for n in addr_names}) != 1:
        return ""
    dwidth = _w(data_names[0])
    awidth = _w(addr_names[0])
    if dwidth <= 0 or awidth <= 0:
        return ""
    # Matches the PDK model exactly: ``RAM_DEPTH = 1 << ADDR_WIDTH``. Sizing from
    # the registry's word count instead would leave an addressable index out of
    # range for a non-power-of-two macro (e.g. a 127-word SRAM with 7 address
    # bits).
    depth = 1 << awidth
    data_rng = _macro_range_str(iface.ranges.get(data_names[0]))
    if not data_rng:
        data_rng = "[0:0]"

    decl: list[str] = []
    for nm in iface.order:
        kw = "output reg" if iface.dirs.get(nm) == "output" else \
             f"{iface.dirs.get(nm, 'input')}  wire"
        rng = _macro_range_str(iface.ranges.get(nm))
        decl.append(f"  {kw} {rng + ' ' if rng else ''}{nm};")

    body: list[str] = []
    for idx in sorted(roles):
        r = roles[idx]
        clk, addr = r.get("clk"), r.get("addr")
        dout, din, web = r.get("dout"), r.get("din"), r.get("web")
        if not clk or not addr or not dout:
            return ""            # a port with no clock/address/data path
        # Select polarity is READ, not assumed: OpenRAM SRAMs declare active-low
        # `csb`, OpenROM mask ROMs declare active-high `cs`.
        if "csb" in r:
            sel = f"!{r['csb']}"
        elif "cs" in r:
            sel = r["cs"]
        else:
            sel = "1'b1"

        if web and din:
            wmask = r.get("wmask")
            writes: list[str] = []
            if wmask is None:
                # The macro has NO write mask: word_size == write_size, so the
                # write enable IS the mask. Gating on a pin that does not exist
                # is how every write got silently dropped.
                writes.append(f"        mem[{addr}] <= {din};")
            else:
                nmask = _w(wmask)
                slices = list(iface.mask_slices)
                if len(slices) != nmask:
                    # No usable merge function in the model: fall back to the
                    # registry's write granularity, but only if it reproduces the
                    # DECLARED mask width exactly. Otherwise refuse.
                    gran = int(mask_bits or 0)
                    if gran <= 0 or -(-dwidth // gran) != nmask:
                        return ""
                    slices = [(min((i + 1) * gran - 1, dwidth - 1), i * gran)
                              for i in range(nmask)]
                if nmask == 1 and slices[0] == (dwidth - 1, 0):
                    writes.append(f"        if ({wmask}[0]) mem[{addr}] <= {din};")
                else:
                    for i, (hi, lo) in enumerate(slices):
                        writes.append(
                            f"        if ({wmask}[{i}]) mem[{addr}][{hi}:{lo}] "
                            f"<= {din}[{hi}:{lo}];")
            body.append(
                f"  always @(posedge {clk}) begin\n"
                f"    if ({sel}) begin\n"
                f"      if (!{web}) begin\n"
                + "\n".join(writes) + "\n"
                "      end else begin\n"
                f"        {dout} <= mem[{addr}];\n"
                "      end\n"
                "    end\n"
                "  end"
            )
        elif web or din:
            return ""            # half a write port is not something to guess at
        else:
            body.append(
                f"  always @(posedge {clk}) begin\n"
                f"    if ({sel}) {dout} <= mem[{addr}];\n"
                f"  end"
            )

    # A memory with no write port anywhere is a ROM: its behaviour IS its
    # contents, so a stand-in that cannot load them would read zeros and hand
    # back a verdict about hardware it never modelled.
    writable = any("web" in r and "din" in r for r in roles.values())
    init = ""
    if iface.contents:
        if not Path(iface.contents).is_file():
            return ""
        init = (f'  initial $readmem{iface.contents_radix or "b"}'
                f'("{iface.contents}", mem, 0, {depth - 1});\n\n')
    elif not writable:
        return ""

    pin_list = list(iface.order)
    guard_hdr = guard_decl = ""
    if iface.power_pins:
        guard_hdr = ("`ifdef USE_POWER_PINS\n"
                     + "".join(f"  {n},\n" for n in iface.power_pins)
                     + "`endif\n")
        guard_decl = ("`ifdef USE_POWER_PINS\n"
                      + "".join(f"  inout {n};\n" for n in iface.power_pins)
                      + "`endif\n")

    geom = f"{ports or '?'}, {words or depth}w x {data_bits or dwidth}b"
    return (
        "// Generated by coresmith harness.gate_sim -- cycle-accurate stand-in\n"
        f"// for hard macro {name} ({geom}).\n"
        "// The PDK's own behavioural model drives `#(T_HOLD) dout = 'bx;` from\n"
        "// a posedge block and re-drives on negedge: that races to destruction\n"
        "// in a zero-delay simulator, and its VERBOSE $display floods a\n"
        "// full-length gate run. Cycle semantics below match the PDK model.\n"
        "//\n"
        "// PORT LIST AND WIDTHS ARE READ FROM THE REAL MACRO'S OWN VERILOG --\n"
        "// a port it does not declare must not appear here (an unconnected\n"
        "// input ties low, which silently suppressed every write), and a port\n"
        "// it does declare must have the same width.\n"
        "`default_nettype wire\n"
        "`timescale 1ns / 1ps\n\n"
        f"module {name} (\n"
        + guard_hdr
        + "  " + ",\n  ".join(pin_list) + "\n);\n"
        + guard_decl
        + "\n".join(decl) + "\n\n"
        f"  reg {data_rng} mem [0:{depth - 1}];\n\n"
        + init
        + "\n\n".join(body) + "\n"
        "endmodule\n"
    )


def macro_model_files(netlist_path: str | Path, work_dir: Path) -> tuple[list[str], list[str]]:
    """Emit simulation models for every hard macro the netlist instantiates.

    Returns ``(files, unresolved)``. ``unresolved`` names macros we could not
    model -- the caller must FAIL rather than simulate a design with an
    undriven memory (an unmodelled macro is a guaranteed false verdict, in
    either direction). A macro whose own Verilog cannot be read or parsed lands
    here too: the interface is never guessed.
    """
    try:
        from orchestrator.langgraph import macro_registry
    except Exception:
        return [], []
    try:
        found = macro_registry.detect_instantiated_macros(str(netlist_path))
    except Exception:
        return [], []
    if not found:
        return [], []

    mode = gate_sim_macro_model_mode()
    files: list[str] = []
    unresolved: list[str] = []
    work_dir.mkdir(parents=True, exist_ok=True)
    for info in found:
        model = getattr(info, "verilog", "") or ""
        if mode == "pdk" and model:
            src = Path(model)
            if src.exists():
                files.append(str(src))
                continue
        src_text = macro_model_source(
            info.name, info.ports, info.data_bits, info.words, info.mask_bits,
            iface=macro_interface(info.name, model),
        )
        if not src_text:
            unresolved.append(info.name)
            continue
        out = work_dir / f"{info.name}__gatesim.v"
        write_if_changed(out, src_text)
        files.append(str(out))
    return files, unresolved


# ---------------------------------------------------------------------------
# Generated C++ driver
# ---------------------------------------------------------------------------


def _cpp_words(width: int) -> int:
    return (width + 31) // 32


def write_if_changed(path: Path, text: str) -> None:
    """Write only when the content actually differs.

    Verilator's ``--build`` drives ``make``, which keys off mtimes. Rewriting
    identical generated collateral on every call would invalidate the whole
    object cache and turn a repeat gate sim (same netlist, e.g. a retry that
    only touched the testbench) from ~1 s back into a full C++ rebuild of the
    netlist -- by far the dominant cost of this gate.
    """
    try:
        if path.exists() and path.read_text() == text:
            return
    except OSError:
        pass
    path.write_text(text)


def render_driver_cpp(top: str, ports: list[Port], clock: str,
                      debug: Optional[bool] = None,
                      max_divergences: Optional[int] = None) -> str:
    """Generate the C++ testbench that replays the recorded vectors.

    Fully design-agnostic: the port list is derived from the netlist, so the
    same generator serves any design. The driver writes a JSON verdict and only
    ever reports success when it actually compared cycles AND bits.

    Power/ground ``inout`` ports are tied to their own rail every cycle and never
    compared -- see :func:`power_inout_level`. A non-power ``inout`` never reaches
    here: :func:`check_gate_sim` declines to render a verdict on it.

    ``debug`` (default: :func:`gate_sim_debug`) keeps comparing past the first
    divergence and adds ``divergence_count`` plus the first ``max_divergences``
    mismatches to the verdict. When it is off the debug code is not emitted at
    all, so the default driver and its verdict JSON are unchanged.
    """
    if debug is None:
        debug = gate_sim_debug()
    if max_divergences is None:
        max_divergences = gate_sim_max_divergences()
    ins = [p for p in ports if p.direction == "input" and p.name != clock]
    outs = [p for p in ports if p.direction == "output"]
    rails, _inert, _sig = split_inouts(ports)

    # Supplies are constants, but they are re-asserted every cycle: an `inout`
    # can be driven from inside the netlist during eval(), and a rail that
    # collapses mid-run would poison every cell it feeds.
    tie: list[str] = [
        f"    dut->{p.name} = {power_inout_level(p.name)};   // power rail, not compared"
        for p in rails
    ]

    drive: list[str] = []
    for i, p in enumerate(ins):
        if p.width <= 64:
            # Implicit narrowing on assignment: Verilator types the port as
            # CData/SData/IData/QData by width and the recorded value is
            # already width-fitted, so the assignment is exact.
            drive.append(f"    dut->{p.name} = parse_scalar(in_tok[{i}]);")
        else:
            nw = _cpp_words(p.width)
            drive.append(
                f"    {{ uint32_t w[{nw}]; parse_wide(in_tok[{i}], w, {nw});"
                f" for (int k=0;k<{nw};k++) dut->{p.name}[k] = w[k]; }}"
            )

    check: list[str] = []
    for i, p in enumerate(outs):
        if p.width <= 64:
            check.append(f"""    {{
      const std::string &e = out_tok[{i}];
      if (comparable(e)) {{
        uint64_t exp = parse_scalar(e);
        uint64_t act = (uint64_t)dut->{p.name};
        bits_compared += {p.width};
        if (exp != act) record("{p.name}", cycle, e, act);
      }}
    }}""")
        else:
            nw = _cpp_words(p.width)
            check.append(f"""    {{
      const std::string &e = out_tok[{i}];
      if (comparable(e)) {{
        uint32_t w[{nw}]; parse_wide(e, w, {nw});
        bits_compared += {p.width};
        for (int k = 0; k < {nw}; k++) {{
          if (w[k] != (uint32_t)dut->{p.name}[k]) {{
            // Report the differing 32-bit WORD so expected/actual stay
            // comparable on a port too wide for one scalar.
            char pn[128];
            snprintf(pn, sizeof(pn), "%s[word%d]", "{p.name}", k);
            record(pn, cycle, word_bits(e, k), (uint64_t)dut->{p.name}[k]);
            break;
          }}
        }}
      }}
    }}""")

    n_in, n_out = len(ins), len(outs)

    # --- opt-in debug instrumentation ---------------------------------------
    # Emitted only under CORESMITH_GATE_SIM_DEBUG so the default driver, its
    # verdict JSON and its PASS/FAIL semantics are exactly as before.
    if debug:
        dbg_globals = f"""
// CORESMITH_GATE_SIM_DEBUG: keep comparing after the first mismatch. The verdict
// is unchanged (one divergence is still a FAIL); this only distinguishes "one
// wrong beat" from "every beat is wrong", which the first-mismatch-only report
// cannot.
static const size_t DIV_LIMIT = {max_divergences};
static std::vector<std::string> div_all;
static long long div_count = 0;
"""
        dbg_record = """  div_count++;
  if (div_all.size() < DIV_LIMIT) {
    char buf[512];
    snprintf(buf, sizeof(buf),
             "{\\"cycle\\": %lld, \\"port\\": \\"%s\\", \\"expected\\": \\"%s\\", "
             "\\"actual\\": \\"%s\\"}",
             cycle, port, e.c_str(), to_bits(a, e.size()).c_str());
    div_all.push_back(buf);
  }
"""
        dbg_break = ("    // CORESMITH_GATE_SIM_DEBUG: do NOT stop at the first "
                     "divergence.")
        dbg_json = """  fprintf(jf, ", \\"divergence_count\\": %lld, \\"divergences\\": [",
          div_count);
  for (size_t i = 0; i < div_all.size(); i++)
    fprintf(jf, "%s%s", i ? ", " : "", div_all[i].c_str());
  fprintf(jf, "]");
"""
    else:
        dbg_globals = dbg_record = dbg_json = ""
        dbg_break = "    if (div_cycle >= 0) break;   // stop at the first divergence"

    return f"""// Generated by coresmith harness.gate_sim -- vector-replay driver for
// the post-synthesis gate netlist of `{top}`. No cocotb, no design knowledge:
// the port list below was parsed from the netlist itself.
#include "V{top}.h"
#include "verilated.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static bool comparable(const std::string &v) {{
  for (char c : v) if (c=='x'||c=='X'||c=='z'||c=='Z') return false;
  return !v.empty();
}}

static uint64_t parse_scalar(const std::string &v) {{
  uint64_t r = 0;
  for (char c : v) {{
    r <<= 1;
    if (c == '1') r |= 1ULL;
  }}
  return r;
}}

static void parse_wide(const std::string &v, uint32_t *w, int nw) {{
  for (int i = 0; i < nw; i++) w[i] = 0;
  int n = (int)v.size();
  for (int i = 0; i < n; i++) {{
    int bit = n - 1 - i;             // bit index from LSB
    if (v[i] == '1') w[bit >> 5] |= (1u << (bit & 31));
  }}
}}

static long long div_cycle = -1;
static std::string div_port, div_exp, div_act;
{dbg_globals}
// Render the actual value in the SAME binary form as the recorded expected
// value, so the divergence line in the report compares like with like.
static std::string to_bits(uint64_t v, size_t width) {{
  if (width == 0 || width > 64) width = 64;
  std::string s(width, '0');
  for (size_t i = 0; i < width; i++)
    if (v & (1ULL << (width - 1 - i))) s[i] = '1';
  return s;
}}

// The 32-bit word `k` (counting from the LSB) of an MSB-first bit string.
static std::string word_bits(const std::string &e, int k) {{
  long long hi = (long long)e.size() - 32LL * k;          // exclusive
  long long lo = hi - 32;
  if (hi <= 0) return std::string("0");
  if (lo < 0) lo = 0;
  return e.substr((size_t)lo, (size_t)(hi - lo));
}}

static void record(const char *port, long long cycle, const std::string &e,
                   uint64_t a) {{
{dbg_record}  if (div_cycle >= 0) return;
  div_cycle = cycle; div_port = port; div_exp = e;
  div_act = to_bits(a, e.size());
}}

int main(int argc, char **argv) {{
  if (argc < 3) {{ fprintf(stderr, "usage: sim <vectors> <verdict.json>\\n"); return 2; }}
  Verilated::commandArgs(argc, argv);

  std::ifstream vf(argv[1]);
  if (!vf) {{ fprintf(stderr, "gate-sim: cannot open vectors\\n"); return 2; }}

  V{top} *dut = new V{top};
  long long cycle = 0, compared = 0;
  long long bits_compared = 0;
  std::string line;
  std::vector<std::string> in_tok({n_in}), out_tok({n_out});

  // Power/ground boundary: bring the rails up before the first eval().
{chr(10).join(tie) or "  // (none)"}

  while (std::getline(vf, line)) {{
    if (line.empty() || line[0] == '#') continue;
    size_t bar = line.find('|');
    if (bar == std::string::npos) continue;
    {{
      std::istringstream is(line.substr(0, bar));
      for (int i = 0; i < {n_in}; i++) if (!(is >> in_tok[i])) in_tok[i] = "x";
    }}
    {{
      std::istringstream os(line.substr(bar + 1));
      for (int i = 0; i < {n_out}; i++) if (!(os >> out_tok[i])) out_tok[i] = "x";
    }}

    // Apply the recorded pre-edge inputs and let combinational logic settle.
    dut->{clock} = 0;
{chr(10).join(tie)}
{chr(10).join(drive)}
    dut->eval();

    // Compare the pre-edge steady state of every output port.
{chr(10).join(check)}
    compared++;

    // Clock edge.
    dut->{clock} = 1; dut->eval();
    dut->{clock} = 0; dut->eval();
    cycle++;
{dbg_break}
  }}

  dut->final();

  FILE *jf = fopen(argv[2], "w");
  if (!jf) return 2;
  fprintf(jf, "{{\\"cycles_compared\\": %lld, \\"output_bits_compared\\": %lld, ",
          compared, bits_compared);
  if (div_cycle >= 0) {{
    fprintf(jf, "\\"diverged\\": true, \\"first_divergence\\": {{\\"cycle\\": %lld, "
            "\\"port\\": \\"%s\\", \\"expected\\": \\"%s\\", \\"actual\\": \\"%s\\"}}",
            div_cycle, div_port.c_str(), div_exp.c_str(), div_act.c_str());
  }} else {{
    fprintf(jf, "\\"diverged\\": false, \\"first_divergence\\": {{}}");
  }}
{dbg_json}  fprintf(jf, "}}\\n");
  fclose(jf);
  delete dut;
  return div_cycle >= 0 ? 1 : 0;
}}
"""


# ---------------------------------------------------------------------------
# Driver: build + run
# ---------------------------------------------------------------------------

VERILATOR_DEFINES = ["FUNCTIONAL", "UNIT_DELAY="]
# Cell-library noise we deliberately silence -- none of it is a property of the
# design under test. LATCH/WIDTH/etc. are inherent to vendor cell models.
VERILATOR_SUPPRESS = [
    "-Wno-TIMESCALEMOD", "-Wno-LATCH", "-Wno-WIDTH", "-Wno-UNOPTFLAT",
    "-Wno-CASEINCOMPLETE", "-Wno-MULTIDRIVEN", "-Wno-PINMISSING",
    "-Wno-IMPLICIT", "-Wno-DECLFILENAME", "-Wno-STMTDLY", "-Wno-UNUSEDSIGNAL",
    "-Wno-UNDRIVEN", "-Wno-SYNCASYNCNET", "-Wno-COMBDLY", "-Wno-INITIALDLY",
    "-Wno-ASSIGNDLY", "-Wno-BLKANDNBLK", "-Wno-REALCVT", "-Wno-CMPCONST",
]


def build_and_run_gate_sim(
    top: str,
    netlist_path: str,
    sources: list[str],
    vectors_path: str,
    work_dir: Path,
    timeout_s: int,
) -> dict:
    """Verilate + compile + run. Returns a raw dict; never raises.

    ``ok`` in the returned dict means the netlist elaborated, built, ran, and
    produced a NON-EMPTY verdict with a real comparison in it.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    obj_dir = work_dir / "obj_dir"
    verdict_path = work_dir / "gate_sim_verdict.json"
    if verdict_path.exists():
        verdict_path.unlink()

    verilator = shutil.which("verilator")
    if not verilator:
        return {"ok": False, "tooling_missing": True, "stage": "verilate",
                "error": "verilator not installed"}

    cmd = [
        verilator, "--cc", "--exe", "--build",
        "-j", os.environ.get("CORESMITH_GATE_SIM_JOBS", "2"),
        "--top-module", top,
        "-Mdir", str(obj_dir),
        "-o", "gate_sim",
        # Deterministic zero-initialization so the netlist's power-up state
        # matches the (also 2-state, also zero-initialized) Verilator RTL
        # reference run. Random X-fill would manufacture divergences in the
        # pre-reset cycles that say nothing about the netlist.
        "--x-assign", "0", "--x-initial", "0",
    ]
    for d in VERILATOR_DEFINES:
        cmd.append(f"-D{d}")
    cmd += VERILATOR_SUPPRESS
    cmd += ["-Wno-fatal"]
    cmd += sources + [netlist_path, str(work_dir / "gate_sim_main.cpp")]

    try:
        build = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=timeout_s, cwd=str(work_dir))
    except subprocess.TimeoutExpired:
        return {"ok": False, "stage": "verilate",
                "error": f"verilator build exceeded {timeout_s}s"}
    except OSError as exc:
        return {"ok": False, "stage": "verilate", "error": f"verilator: {exc}"}

    (work_dir / "verilate.log").write_text(build.stdout + "\n" + build.stderr)
    exe = obj_dir / "gate_sim"
    if build.returncode != 0 or not exe.exists():
        tail = (build.stdout + "\n" + build.stderr)[-4000:]
        return {"ok": False, "stage": "verilate",
                "error": "the gate netlist did not elaborate/build",
                "log": tail}

    try:
        run = subprocess.run(
            [str(exe), vectors_path, str(verdict_path)],
            capture_output=True, text=True, timeout=timeout_s, cwd=str(work_dir),
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "stage": "run",
                "error": f"gate simulation exceeded {timeout_s}s"}
    except OSError as exc:
        return {"ok": False, "stage": "run", "error": f"gate sim exec: {exc}"}

    (work_dir / "run.log").write_text(run.stdout + "\n" + run.stderr)

    # FAIL-CLOSED: a missing or blank verdict is NOT a pass.
    if not verdict_path.exists() or verdict_path.stat().st_size == 0:
        return {"ok": False, "stage": "verdict",
                "error": "gate simulation produced no verdict file "
                         "(blank result must never read as a pass)",
                "log": (run.stdout + "\n" + run.stderr)[-2000:]}
    try:
        verdict = json.loads(verdict_path.read_text())
    except (OSError, ValueError) as exc:
        return {"ok": False, "stage": "verdict",
                "error": f"unparseable gate-sim verdict: {exc}"}
    if not isinstance(verdict, dict) or "cycles_compared" not in verdict:
        return {"ok": False, "stage": "verdict",
                "error": "gate-sim verdict missing cycles_compared"}

    verdict["ok"] = True
    verdict["stage"] = "run"
    verdict["returncode"] = run.returncode
    return verdict


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def check_gate_sim(
    block: dict,
    netlist_path: str,
    rtl_path: str,
    tb_path: str,
    attempt: int = 1,
    work_root: str | Path | None = None,
    sim_runner: Optional[Callable] = None,
    pdk_root: str | Path | None = None,
) -> GateSimResult:
    """Run the post-synthesis gate-level simulation for one subject.

    ``sim_runner`` is injected by tests; it defaults to
    ``pipeline_helpers.run_simulation`` and must accept
    ``(block, rtl_path, tb_path, attempt, extra_defines=, sim_subdir=,
    extra_args=)``.

    The two gates that decide whether a verdict is even ADMISSIBLE come first,
    before anything is read off disk: the env switch (:func:`gate_sim_enabled`)
    and the scope (:func:`gate_sim_scope`). "Off" outranks "out of scope" --
    when the gate is off, that is the whole story, and a reader must not be told
    a leaf block was skipped for scope reasons when nothing would have run.
    """
    block_name = block.get("name", "block")

    if not gate_sim_enabled():
        return GateSimResult(
            ran=False, ok=True, status=STATUS_DISABLED,
            reason=f"{GATE_SIM_ENV}=0 -- the synthesized netlist is NEVER "
                   f"simulated; a synthesis-side stub cannot be caught",
        )

    scope = gate_sim_scope()
    if scope == "chip_top" and not block_is_chip_top(block):
        return GateSimResult(
            ran=False, ok=True, status=STATUS_NOT_RUN,
            reason=(f"scope={scope}: this gate only judges the integrated "
                    f"chip_top netlist, and '{block_name or '?'}' is a "
                    f"leaf block. A per-block netlist is not the artifact that "
                    f"becomes silicon, and per-block replay drives it with that "
                    f"block's own testbench rather than real chip traffic. Set "
                    f"{GATE_SIM_SCOPE_ENV}=block for per-block replay."),
        )

    # Resolve the top from the NETLIST'S STRUCTURE, never from an identifier.
    #
    # A block's architectural name is chosen by the architecture agent and is
    # not a stable key: when an external contract fixes the module name (a
    # Caravel user_project_wrapper, a vendor-locked top), the block keeps its
    # own name while the RTL declares the mandated one. Keying this gate off
    # block_name made it FAIL CLOSED on a correctly synthesized netlist --
    # "top module not found", 0 cycles compared -- and which way it went
    # depended on whether the architecture agent happened to append a suffix.
    #
    # Precedence: an explicit block["top"] (a deliberate override) > the
    # netlist's own root module > the RTL's declared module. The structural
    # answer returns None when ambiguous rather than guessing.
    top = block.get("top")
    if not top:
        try:
            top = resolve_netlist_top(Path(netlist_path).read_text(errors="replace"))
        except OSError:
            top = None
    if not top:
        try:
            from orchestrator.langgraph.pipeline_helpers import rtl_module_name
            top = rtl_module_name(rtl_path, block_name)
        except Exception:  # noqa: BLE001 - resolver is best-effort
            top = block_name

    def _not_run(reason: str, **detail) -> GateSimResult:
        strict = gate_sim_strict()
        return GateSimResult(
            ran=False, ok=not strict,
            status=STATUS_FAIL if strict else STATUS_NOT_RUN,
            reason=reason, netlist_path=netlist_path, detail=detail,
        )

    def _fail(reason: str, **detail) -> GateSimResult:
        return GateSimResult(
            ran=True, ok=False, status=STATUS_FAIL, reason=reason,
            netlist_path=netlist_path, detail=detail,
        )

    # --- toolchain FIRST -----------------------------------------------------
    # The gate only ever renders a VERDICT on a host where it could actually
    # have run. Judging (even a missing netlist) on a host with no simulator or
    # no PDK would be an opinion the gate is not entitled to -- and it would
    # turn every PDK-free frontend host into a wall. Toolchain absence is
    # therefore ``not_run`` (or FAIL under STRICT, where the caller has declared
    # the gate must really run).
    if not shutil.which("verilator"):
        return _not_run("verilator not installed")

    # --- the netlist itself -------------------------------------------------
    # A netlist path that is not on disk is the SYNTH gate's business, not this
    # one's: there is no artifact here to render a behavioural verdict on. Kept
    # ``not_run`` (loudly) so a default host cannot be walled off by a flow
    # quirk, and escalated to FAIL under STRICT, where the caller has declared
    # the gate must really run. Everything the gate CAN see -- an empty netlist,
    # a missing top, one that will not elaborate, a blank verdict -- is a FAIL.
    if not netlist_path or not Path(netlist_path).exists():
        return _not_run("synthesis reported success but left NO NETLIST on "
                        f"disk at {netlist_path or '<unset>'} -- the gate "
                        "netlist was never simulated")
    netlist_text = Path(netlist_path).read_text(errors="ignore")
    if not netlist_text.strip():
        return _fail("the synthesized netlist is empty")

    cells, cell_prefixes = cell_model_files(pdk_root, netlist_text)
    if not cells:
        return _not_run("PDK standard-cell simulation models not found "
                        "(no <pdk>/<variant>/libs.ref/*/verilog/*.v)")

    ports = parse_netlist_ports(netlist_text, top)
    if not ports:
        return _fail(
            f"top module '{top}' not found in the synthesized netlist, or it "
            f"declares no ports -- the netlist is not the design under test"
        )
    clock = pick_clock(ports)
    if clock is None:
        return _not_run("no clock port identified on the netlist top -- "
                        "cycle-accurate replay needs one")
    rails, inert_inouts, signal_inouts = split_inouts(ports, netlist_text)
    if inert_inouts:
        # Mandated-but-unused pad buses (Caravel analog_io): the netlist proves
        # they carry no behaviour (declaration + constant-Z tie only), so they
        # are excluded from drive and comparison. Excluding nothing fabricates
        # nothing -- but record them so the verdict says what was not judged.
        pass
    if signal_inouts:
        # A bidirectional SIGNAL is neither driven nor compared by the replay, so
        # a PASS would carry a hole exactly where the gate is supposed to be
        # strongest. Report it instead of hiding it. Power/ground rails are not
        # this: they are constants, they are tied in the driver, and they are
        # excluded from the comparison -- otherwise the gate could never judge a
        # Caravel user_project_wrapper, whose only inouts are supplies, which is
        # precisely the artifact the external grader drives.
        return _not_run(
            "the netlist top has bidirectional port(s) "
            + ", ".join(p.name for p in signal_inouts)
            + " -- vector replay cannot drive or compare a tri-state boundary, "
              "so no honest verdict is available"
        )

    if work_root is None:
        try:
            from orchestrator.langgraph.pipeline_helpers import PROJECT_ROOT
            work_root = Path(PROJECT_ROOT) / "sim_build"
        except Exception:
            work_root = Path("sim_build")
    work_dir = Path(work_root) / f"{block_name}__gatesim"
    work_dir.mkdir(parents=True, exist_ok=True)

    macro_files, unresolved = macro_model_files(netlist_path, work_dir)
    if unresolved:
        return _fail(
            "the netlist instantiates hard macro(s) with no simulation model: "
            + ", ".join(unresolved)
            + " -- refusing to simulate a design whose memories are undriven"
        )

    # --- reference RTL run (pinned seed, shallow trace) ----------------------
    if sim_runner is None:
        try:
            from orchestrator.langgraph.pipeline_helpers import (
                run_simulation as sim_runner,  # type: ignore
            )
        except Exception as exc:  # pragma: no cover - import guard
            return _not_run(f"no RTL simulation runner available: {exc}")

    from orchestrator.harness.seed_provider import gate_seed

    seed = str(gate_seed())
    prev_pin = os.environ.get("CORESMITH_DV_SEED_PIN")
    os.environ["CORESMITH_DV_SEED_PIN"] = seed
    ref_subdir = f"{block_name}__gatesim_ref"
    try:
        ref = sim_runner(
            block, rtl_path, tb_path, attempt,
            extra_defines=None, sim_subdir=ref_subdir,
            extra_args=["--trace-depth 1"],
        )
    except TypeError:
        # Older runner without extra_args -- the deep trace still works, it is
        # just larger.
        try:
            ref = sim_runner(block, rtl_path, tb_path, attempt,
                             extra_defines=None, sim_subdir=ref_subdir)
        except Exception as exc:  # noqa: BLE001
            return _not_run(f"reference RTL run failed to launch: {exc}")
    except Exception as exc:  # noqa: BLE001
        return _not_run(f"reference RTL run failed to launch: {exc}")
    finally:
        if prev_pin is None:
            os.environ.pop("CORESMITH_DV_SEED_PIN", None)
        else:
            os.environ["CORESMITH_DV_SEED_PIN"] = prev_pin

    ref = ref or {}
    if not ref.get("passed"):
        # The reference did not pass, so there is no trustworthy behaviour to
        # compare against. That is an RTL/DV problem, not a netlist verdict.
        return _not_run("the reference RTL run did not pass under the pinned "
                        "gate-sim seed -- nothing trustworthy to compare against",
                        ref_log=str(ref.get("log", ""))[-1500:])

    vcd = Path(ref.get("vcd_path") or (Path(work_root) / ref_subdir / "dump.vcd"))
    if not vcd.exists() or vcd.stat().st_size == 0:
        return _fail("the reference RTL run produced no waveform, so the gate "
                     "netlist cannot be compared against it")

    vec = extract_vectors(vcd, ports, clock.name, gate_sim_max_cycles())
    if vec.cycles == 0:
        return _fail("no clock cycles were recorded from the reference RTL run "
                     "-- a gate sim with no stimulus must not read as a pass")
    if not vec.outputs:
        return _fail("the netlist top declares no output ports to compare")
    if vec.output_activity < 2:
        return _fail(
            "the reference RTL run never changed any output port across "
            f"{vec.cycles} cycles -- the comparison would be vacuous, so this "
            "cannot be scored as a gate-sim pass"
        )

    vectors_path = work_dir / "gate_sim_vectors.txt"
    write_vector_file(vec, vectors_path, top, clock.name)

    # --- build + run the gate netlist ---------------------------------------
    # Written content-stably so an unchanged netlist reuses the (expensive)
    # Verilator object cache instead of rebuilding from scratch every attempt.
    write_if_changed(work_dir / "gate_sim_main.cpp",
                     render_driver_cpp(top, ports, clock.name))
    shim = work_dir / "cell_udp_shim.v"
    write_if_changed(shim, udp_shim_source(cell_prefixes))

    raw = build_and_run_gate_sim(
        top=top,
        netlist_path=netlist_path,
        sources=[str(shim)] + macro_files + cells,
        vectors_path=str(vectors_path),
        work_dir=work_dir,
        timeout_s=gate_sim_timeout_s(),
    )

    if raw.get("tooling_missing"):
        return _not_run(raw.get("error", "toolchain missing"))
    if not raw.get("ok"):
        return _fail(raw.get("error", "gate simulation failed"),
                     stage=raw.get("stage", ""), log=raw.get("log", ""))

    cycles = int(raw.get("cycles_compared") or 0)
    bits = int(raw.get("output_bits_compared") or 0)
    if cycles <= 0:
        return _fail("gate simulation compared 0 cycles -- a run that did not "
                     "actually execute must never read as a pass")
    if bits <= 0:
        return _fail("gate simulation compared 0 output bits (every recorded "
                     "output was unknown) -- that is not a pass")

    detail = {"work_dir": str(work_dir), "recorded_cycles": vec.cycles,
              "seed": seed}
    if rails:
        detail["power_rails_tied"] = [p.name for p in rails]
    # Present only when the run was built under CORESMITH_GATE_SIM_DEBUG: the
    # default driver stops at the first mismatch and cannot know a total.
    if raw.get("divergence_count") is not None:
        detail["divergence_count"] = int(raw.get("divergence_count") or 0)
        detail["divergences"] = raw.get("divergences") or []

    if raw.get("diverged"):
        res = _fail("the gate netlist diverges from the verified RTL")
        res.cycles_compared = cycles
        res.output_bits_compared = bits
        res.first_divergence = raw.get("first_divergence") or {}
        res.detail = detail
        return res

    return GateSimResult(
        ran=True, ok=True, status=STATUS_PASS,
        reason=f"gate netlist reproduced the verified RTL for {cycles} cycles",
        netlist_path=netlist_path,
        cycles_compared=cycles,
        output_bits_compared=bits,
        detail={**detail, "macro_models": len(macro_files)},
    )


__all__ = [
    "GATE_SIM_ENV", "GATE_SIM_SCOPE_ENV", "GATE_SIM_DEBUG_ENV",
    "GateSimResult", "MacroIface", "Port", "Vectors",
    "STATUS_PASS", "STATUS_FAIL", "STATUS_NOT_RUN", "STATUS_DISABLED",
    "gate_sim_enabled", "gate_sim_scope", "block_is_chip_top",
    "gate_sim_strict", "gate_sim_max_cycles",
    "gate_sim_timeout_s", "gate_sim_macro_model_mode",
    "gate_sim_debug", "gate_sim_max_divergences",
    "resolve_netlist_top", "parse_netlist_ports", "pick_clock",
    "power_inout_level", "split_inouts",
    "extract_vectors", "write_vector_file",
    "udp_shim_source", "cell_model_files",
    "macro_interface", "macro_model_source",
    "macro_model_files", "render_driver_cpp", "build_and_run_gate_sim",
    "write_if_changed",
    "check_gate_sim",
]
