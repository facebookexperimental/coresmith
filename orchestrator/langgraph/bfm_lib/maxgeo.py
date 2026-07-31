# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Which declared dimensional maxima can the QSPI-slave bus HONESTLY drive?

The MAX-GEOMETRY DV gate (``pipeline_graph._maxgeo_gate_verdict``) requires the
integration testbench to advertise ``# MAXGEO: <dim>=<value>`` for every maximum
the design declares -- because a chip can pass every fixed-small-geometry test
and still ship an index/address width that wraps at a 2^n boundary below the
declared maximum.

The deterministic QSPI conformance testbench is COMPUTE-LANE INDEPENDENT: it has
no golden model, so it cannot evaluate any compute-lane transform at all. What it
CAN do is drive the *bus* at maximum extent -- the full-width address, the
longest legal write burst, the longest legal read burst, the largest command
opcode -- and those are real dimensional maxima with real index widths behind
them.

This module decides, from the bus contract plus the design's declared maxima,
exactly which dims fall in that set. Two callers use it, and they must agree:

  * ``codegen`` -- emits the max-extent probes and the ``# MAXGEO`` markers.
  * the gate    -- recomputes the expected set INDEPENDENTLY (from the contract
    the architecture produced, not from what the testbench claimed) and refuses
    to accept a conformance TB that skipped a maximum it could have driven.

That split is the point. If only codegen decided, the testbench would be marking
its own homework -- the failure mode this repo keeps paying for.

SINGLE SOURCE (dimension-registry alignment). Three numbers used to disagree on
a live run: the gate ENFORCED 9 dims, the testbench CONFESSED 13 uncovered, and
the design's own declared table had 18. Nothing was lying -- the three were
derived three different ways. They are now derived here, once:

  * :func:`declared_dimensional_maxima` -- THE declared table (param schema
    first, the legacy generic harvest folded in), for gate and generator alike.
  * :func:`bus_maxgeo_coverage` -- THE classification (bus-drivable vs
    compute-lane), for the generator's confession and the gate's demand alike.
  * :func:`maxgeo_demand` -- THE marker verdict: which declared maxima the
    testbench actually PROVED (name AND value), which merely collide in VALUE
    with some other dim's marker (weak evidence -- the old gate silently counted
    these as covered, which is the whole 13-vs-9 delta), and which are missing.

``proven + value_only + missing`` is always the full declared table, so the
three numbers now reconcile by construction.

**A marker is a claim about coverage that EXISTS.** Nothing here ever reports a
dim as covered unless the emitted testbench actually drives that value on the
pins, at that magnitude. Dims outside the bus -- anything in the compute lane --
come back in ``uncovered`` and stay there.

Role vocabulary note: the matching below reads BUS vocabulary (address, read/
write burst length, nibble count, command opcode). That is this module's own
protocol domain -- ``bfm_lib`` is the QSPI-slave chassis library and already
speaks in ``cmd 0x02`` / ``dummy_bytes`` / ``io0_bit``. The *gate* stays
domain-generic (it never greps for vocabulary); it delegates here precisely so
the protocol knowledge lives in the protocol layer.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from .qspi_contract import QSPIContract

# Longest burst (bytes) the conformance TB will drive on the bus. Each byte is
# two SCK cycles, i.e. 4 * sck_half_period wb-clks -- a 4096-byte read is ~1.3 ms
# of simulated time at the default contract, which Verilator/cocotb handles in
# seconds. A design declaring a burst larger than this simply does not get the
# marker (we did not drive it, so we do not claim it).
_DEFAULT_MAX_BURST_BYTES = 8192
MAX_BURST_ENV = "CORESMITH_CONFORMANCE_MAX_BURST_BYTES"

# Roles, in match priority order. First match wins, so the more specific
# patterns come first (a name carrying both "nibble" and "count" is a nibble
# count, not a burst length).
ROLE_NIBBLES = "transaction_nibbles"
ROLE_COMMAND = "command_opcode"
ROLE_ADDRESS = "bus_address"
ROLE_READ_LEN = "read_burst_bytes"
ROLE_WRITE_LEN = "write_burst_bytes"

_LEN_TOKENS = {
    "length", "len", "bytes", "byte", "size", "burst", "count", "words",
    # storage extents: a ``cmd_fifo_depth`` is a depth, not a maximum opcode
    "depth", "records", "entries", "capacity",
}
_READ_TOKENS = {"read", "rd", "out", "output", "rx"}
_WRITE_TOKENS = {"write", "wr", "in", "input", "tx"}


def _tokens(name: str) -> set[str]:
    return {t for t in re.split(r"[^A-Za-z0-9]+", str(name).lower()) if t}


def classify_dim_role(name: str) -> str:
    """Bus role of a declared dimension NAME, or ``""`` when it is not a bus dim.

    Deliberately conservative: an unrecognised name is NOT a bus dim, so the
    honest outcome (no marker, reported as uncovered) is the default.
    """
    toks = _tokens(name)
    if not toks:
        return ""
    if "nibble" in toks or "nibbles" in toks:
        return ROLE_NIBBLES
    if toks & {"opcode", "opcodes"}:
        return ROLE_COMMAND
    # ``cmd``/``command`` alone is ambiguous: ``cmd_fifo_depth`` is a FIFO depth,
    # not a maximum opcode. A length token anywhere in the name settles it.
    if toks & {"command", "commands", "cmd"} and not (toks & _LEN_TOKENS):
        return ROLE_COMMAND
    if toks & {"address", "addr", "addresses"} and not (toks & _LEN_TOKENS):
        return ROLE_ADDRESS
    if toks & _LEN_TOKENS:
        if toks & _READ_TOKENS:
            return ROLE_READ_LEN
        if toks & _WRITE_TOKENS:
            return ROLE_WRITE_LEN
    return ""


def max_burst_bytes() -> int:
    """Cap on the burst length the conformance TB will actually drive."""
    try:
        v = int((os.environ.get(MAX_BURST_ENV) or "").strip())
        return v if v > 0 else _DEFAULT_MAX_BURST_BYTES
    except (TypeError, ValueError):
        return _DEFAULT_MAX_BURST_BYTES


@dataclass(frozen=True)
class BusMaxgeoCoverage:
    """What the bus-only conformance TB drives, and what it honestly cannot.

    ``covered`` maps dim name -> the maximum VALUE actually driven on the pins.
    ``uncovered`` maps dim name -> declared maximum, for every dim this TB does
    not reach (compute-lane geometry, or a bus dim too large to drive). Both are
    reported; neither is ever silently dropped.
    """

    covered: dict[str, int] = field(default_factory=dict)
    uncovered: dict[str, int] = field(default_factory=dict)
    # concrete probe magnitudes the testbench must drive (0/absent = no probe)
    address: int = 0
    write_bytes: int = 0
    read_bytes: int = 0
    opcode: int = 0

    def marker_line(self) -> str:
        """The ``# MAXGEO: name=value ...`` marker, or "" when nothing is covered."""
        if not self.covered:
            return ""
        pairs = " ".join(f"{n}={v}" for n, v in sorted(self.covered.items()))
        return f"# MAXGEO: {pairs}"

    def uncovered_line(self) -> str:
        """A machine-readable record of what this TB does NOT cover.

        Emitted into the testbench itself so the absence is visible in the
        artifact, not only in a log line that scrolls away.

        The tag is ``MAXGEO_NOT_COVERED``, with an UNDERSCORE, and that is
        load-bearing: the gate's marker regex is ``#\\s*MAXGEO\\b``, and ``\\b``
        does match before a hyphen. A ``# MAXGEO-UNCOVERED: frame_width=64``
        line would therefore be parsed as a COVERAGE claim for frame_width --
        the exact inversion this line exists to prevent. ``MAXGEO_`` has no word
        boundary after ``MAXGEO``, so it cannot be misread.
        """
        if not self.uncovered:
            return ""
        pairs = " ".join(f"{n}={v}" for n, v in sorted(self.uncovered.items()))
        return f"# MAXGEO_NOT_COVERED: {pairs}"


def bus_maxgeo_coverage(
    contract: QSPIContract, declared_dims: dict | None
) -> BusMaxgeoCoverage:
    """Split the design's declared maxima into bus-drivable and not.

    Pure function of ``(contract, declared_dims)`` -- no disk, no DUT, no
    testbench text. Both the generator and the gate call it, so a probe the
    generator drops is a mismatch the gate sees.
    """
    dims = {str(k): int(v) for k, v in (declared_dims or {}).items()
            if _is_pos_int(v)}
    if not contract or not dims:
        return BusMaxgeoCoverage(covered={}, uncovered=dict(dims))

    addr_space = (1 << (8 * int(contract.addr_bytes))) - 1
    cap = max_burst_bytes()
    # A write burst must stay inside the IN aperture: the contract places OUT
    # above IN, so a burst that would run past ``out_addr`` is not a legal IN
    # write and we do not drive (or claim) it.
    in_window = cap
    if int(contract.out_addr) > int(contract.in_addr):
        in_window = int(contract.out_addr) - int(contract.in_addr)
    write_cap = min(cap, in_window)

    covered: dict[str, int] = {}
    uncovered: dict[str, int] = {}
    address = write_bytes = read_bytes = opcode = 0

    # Pass 1: the directly-drivable roles. Each is accepted only when the
    # contract can actually express the declared magnitude.
    nibble_dims: dict[str, int] = {}
    for name, value in sorted(dims.items()):
        role = classify_dim_role(name)
        if role == ROLE_ADDRESS and value <= addr_space:
            address = max(address, value)
            covered[name] = value
        elif role == ROLE_WRITE_LEN and value <= write_cap:
            write_bytes = max(write_bytes, value)
            covered[name] = value
        elif role == ROLE_READ_LEN and value <= cap:
            read_bytes = max(read_bytes, value)
            covered[name] = value
        elif role == ROLE_COMMAND and value <= 0xFF:
            opcode = max(opcode, value)
            covered[name] = value
        elif role == ROLE_NIBBLES:
            nibble_dims[name] = value       # resolved in pass 2
        else:
            uncovered[name] = value

    # Pass 2: a transaction-nibble-count maximum is covered only if the longest
    # burst we actually drive contains EXACTLY that many data nibbles (2 per
    # byte). Anything else would be a marker for traffic that never happened.
    longest = max(write_bytes, read_bytes)
    for name, value in nibble_dims.items():
        if longest and value == 2 * longest:
            covered[name] = value
        else:
            uncovered[name] = value

    return BusMaxgeoCoverage(
        covered=covered, uncovered=uncovered,
        address=address, write_bytes=write_bytes,
        read_bytes=read_bytes, opcode=opcode,
    )


def _is_pos_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v > 0


# ---------------------------------------------------------------------------
# THE declared-dimension registry (one derivation, two consumers)
# ---------------------------------------------------------------------------

#: Machine-readable spec artifacts the legacy generic ``{name, extent}`` harvest
#: reads. Identical to ``pipeline_graph``'s list, deliberately: this function
#: exists to REPLACE that one, not to disagree with it.
DIM_SOURCE_FILES = (
    "ers_spec.json", "prd_spec.json", "block_specs.json", "block_queue.json",
)


def declared_dimensional_maxima(project_root: str) -> dict[str, int]:
    """``{dim_name: max}`` -- the design's declared dimensional maxima.

    ONE derivation, for both the generator (what may I claim / what must I
    confess?) and the gate (what do I demand?). Two sources, merged max-wise,
    exactly as the gate's own ``_declared_dimensions`` merged them:

      * PRIMARY -- the typed ERS ``parameters`` block (``role`` in
        ``dimension``/``range`` carries an extent). Authoritative and
        deterministic; ``mode`` parameters are control-selects, not geometry,
        and are excluded.
      * FOLDED IN -- the legacy generic ``{name-role, extent-role}`` harvest
        over the ERS/PRD/block-spec JSON plus the FRD FUNC vectors, so a legacy
        prose ERS (no ``parameters`` block) behaves exactly as before.

    Returns ``{}`` when the design declares nothing dimensional (the gate then
    no-ops). NEVER raises: a dimensional registry that can throw would take a
    run down over an unreadable spec file.
    """
    dims: dict[str, int] = {}
    root = Path(project_root)
    try:
        from orchestrator.architecture import param_schema as _psch
    except Exception:  # noqa: BLE001 - no schema module -> no registry
        return dims
    try:
        for name, mx in _psch.declared_maxima(
                _psch.parameters_from_ers(project_root)).items():
            dims[str(name)] = max(int(mx), dims.get(str(name), 0))
    except Exception:  # noqa: BLE001
        pass
    try:
        for fname in DIM_SOURCE_FILES:
            p = root / ".coresmith" / fname
            if not p.exists():
                continue
            try:
                _psch._harvest_candidate_dims(
                    json.loads(p.read_text(encoding="utf-8")), dims)
            except (OSError, json.JSONDecodeError, ValueError, TypeError):
                continue
        frd = root / "arch" / "frd_spec.md"
        if frd.exists():
            try:
                from orchestrator.architecture.composition import (
                    parse_func_vectors,
                )
                _psch._harvest_candidate_dims(
                    parse_func_vectors(frd.read_text(encoding="utf-8")), dims)
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001
        return {k: int(v) for k, v in dims.items() if _is_pos_int(int(v))}
    return {k: int(v) for k, v in dims.items() if _is_pos_int(int(v))}


def schema_dimensional_maxima(project_root: str) -> dict[str, int]:
    """``{name: max}`` from the typed ERS ``parameters`` block ONLY.

    The subset of :func:`declared_dimensional_maxima` whose names are
    design-AFFIRMED parameter identifiers rather than free-form strings scraped
    out of arbitrary spec JSON. Functional max-geometry coverage is claimed
    against THIS table (see :func:`functional_maxgeo_dims`) because the claim
    rests on matching a declared parameter to a stimulus key by name, and a
    scraped name is not a declared parameter. ``{}`` for a legacy prose ERS.
    """
    try:
        from orchestrator.architecture import param_schema as _psch
        return {
            str(n): int(v) for n, v in _psch.declared_maxima(
                _psch.parameters_from_ers(project_root)).items()
            if _is_pos_int(int(v))
        }
    except Exception:  # noqa: BLE001
        return {}


@dataclass(frozen=True)
class MaxgeoDemand:
    """What a testbench's ``# MAXGEO`` marker actually PROVES about the table.

    ``proven``     -- the marker names the dim AND carries its declared maximum.
                      Real, attributable evidence.
    ``value_only`` -- the declared maximum appears in the marker, but under some
                      OTHER dim's name. Two dims of the same extent (a 4096-deep
                      framebuffer and a 4096-byte read burst) make this happen
                      constantly, and the name-agnostic gate counted it as
                      coverage -- claiming a compute-lane maximum was exercised
                      because a bus burst happened to be the same number.
    ``missing``    -- no marker evidence of any kind.

    The three partition the declared table, which is the point: the gate's demand
    (``missing``), the generator's confession (``value_only | missing``) and the
    declared table (all three) are now three views of ONE computation.
    """

    proven: dict[str, int] = field(default_factory=dict)
    value_only: dict[str, int] = field(default_factory=dict)
    missing: dict[str, int] = field(default_factory=dict)

    @property
    def unproven(self) -> dict[str, int]:
        """Everything without name-attributable evidence -- the honest confession."""
        return {**self.value_only, **self.missing}

    def describe(self) -> str:
        return (
            f"proven={self.proven} value_only(weak)={self.value_only} "
            f"missing={self.missing}"
        )


def maxgeo_demand(declared_dims: dict | None, marker: dict | None) -> MaxgeoDemand:
    """Split a declared table against a testbench's parsed ``# MAXGEO`` pairs.

    Pure function -- no disk, no testbench text, no DUT. ``marker`` is
    ``{name: value}`` as parsed from the artifact's ``# MAXGEO:`` lines.
    """
    dims = {str(k): int(v) for k, v in (declared_dims or {}).items()
            if _is_pos_int(v)}
    pairs = {str(k): int(v) for k, v in (marker or {}).items()
             if _is_pos_int(v)}
    values = set(pairs.values())
    proven: dict[str, int] = {}
    value_only: dict[str, int] = {}
    missing: dict[str, int] = {}
    for name, mx in sorted(dims.items()):
        if pairs.get(name) == mx:
            proven[name] = mx
        elif mx in values:
            value_only[name] = mx
        else:
            missing[name] = mx
    return MaxgeoDemand(proven=proven, value_only=value_only, missing=missing)


def token_match(dim_name: str, key: str) -> bool:
    """True when a declared dimension NAME and a stimulus KEY name the same axis.

    Subset in either direction, never a bare intersection: ``frame_width`` and
    ``burst_width`` share the token ``width`` yet are different axes, so an
    intersection rule would let a driven burst width claim coverage of a frame
    width. ``{width} <= {frame, width}`` (key ``width`` for dim ``frame_width``)
    and an exact match both pass; ``{burst,width}`` vs ``{frame,width}`` does not.
    """
    a, b = _tokens(dim_name), _tokens(key)
    if not a or not b:
        return False
    return a <= b or b <= a


def functional_maxgeo_dims(
    scalars: dict | None, declared_dims: dict | None
) -> dict[str, int]:
    """Declared maxima a FUNCTIONAL host-flow case DRIVES, by direct evidence.

    ``scalars`` is the integer-valued subset of the chosen acceptance case's
    stimulus dict -- the numbers the generated testbench actually writes onto the
    pins for that case. A dim is claimed only when BOTH hold:

      * some scalar key :func:`token_match`-es the dimension name, and
      * that scalar's VALUE equals the declared maximum.

    Everything else stays in the ``MAXGEO_NOT_COVERED`` confession. A marker is a
    claim about traffic that exists; a dimension we merely believe is implied by
    a big case is not driven evidence and is not claimed.
    """
    dims = {str(k): int(v) for k, v in (declared_dims or {}).items()
            if _is_pos_int(v)}
    vals = {str(k): int(v) for k, v in (scalars or {}).items()
            if _is_pos_int(v)}
    out: dict[str, int] = {}
    for name, mx in sorted(dims.items()):
        for key, val in sorted(vals.items()):
            if val == mx and token_match(name, key):
                out[name] = mx
                break
    return out
