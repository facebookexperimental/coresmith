# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Generic SRAM-wrapper convention + the deterministic gate that enforces it.

The CoreSmith memory contract (front-end):

* Any on-chip storage whose total bits exceed ``CORESMITH_SRAM_MIN_BITS``
  (default 4096) MUST be instantiated through the parametrized wrappers in
  ``rtl_lib/cs_sram.v`` (``cs_sram_1rw`` / ``cs_sram_1rw1r``) -- never written
  as a raw ``reg [W:0] name [0:D]`` array.
* The wrapper is behavioral in simulation and a real SRAM macro under synthesis
  (``CORESMITH_SRAM_SYNTH``); the macro flow (``macro_backend`` / ``openram_gen``)
  resolves each instance's WIDTH/DEPTH to a pre-built sky130 SRAM or generates
  one with OpenRAM, for both the block-level synth gate and the backend.

This module is pure/deterministic (regex over RTL text + a registry lookup) so
it is safe to unit-test and to call from the PPA gate with no EDA tools.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

# --- Aggressive SRAM-wrapper flag policy: OR of three independent triggers ----
#
# A raw ``reg [W-1:0] mem [0:D-1]`` in an ordinary (non-macro) module is flagged
# when ANY of the following hold (all strict ``>``, all env-tunable):
#
#     big  = width * depth > CORESMITH_SRAM_MIN_BITS      (total bits)
#     wide = width         > CORESMITH_SRAM_MIN_WIDTH      (word width)
#     deep = depth         > CORESMITH_SRAM_MACRO_DEPTH    (depth / read-mux fanout)
#     flag = big or wide or deep
#
# WHY THIS IS SAFE (it is not a deadlock): the gate's remedy is NOT "must be an
# SRAM macro". ``_suggest_wrapper`` offers ``cs_fpmem`` -- a REGISTERED flop-array
# primitive -- as a first-class resolution alongside ``cs_sram``. A flagged
# structure that genuinely can't be a single-port macro simply becomes a
# registered ``cs_fpmem`` (or is restructured); it is never stranded. The real
# target of the gate is a raw COMBINATIONAL-read ``reg[] mem[]`` -- an unregistered
# N:1 read mux that is slow and congesting -- and the fix is to REGISTER the read
# (cs_fpmem) or move to a macro (cs_sram), not to inflate flop count.
#
# Most flagged memories DO have a macro option: OpenRAM supports multi-port
# configs and the pre-built sky130 1rw1r macros go down to 4-deep, so nearly any
# addressed array can bind to a macro. Only a genuine all-entries-every-cycle
# (fully parallel-read) structure legitimately stays flops -- and that path is
# ``cs_fpmem``, still registered, still accepted by the gate.
#
# Aggressive on purpose: the previous bits-AND-depth gate let 8Kbit / 256-deep
# comb-read arrays slip through. The OR closes the width-only and depth-only leaks
# while ``cs_fpmem`` keeps every flagged-but-legitimate memory un-blocked.
DEFAULT_MIN_BITS = 2000      # env CORESMITH_SRAM_MIN_BITS   -- total-bits trigger
DEFAULT_MIN_WIDTH = 128      # env CORESMITH_SRAM_MIN_WIDTH  -- word-width trigger
DEFAULT_MACRO_DEPTH = 256    # env CORESMITH_SRAM_MACRO_DEPTH -- depth trigger


def min_bits() -> int:
    try:
        return max(1, int(os.environ.get("CORESMITH_SRAM_MIN_BITS", "") or DEFAULT_MIN_BITS))
    except ValueError:
        return DEFAULT_MIN_BITS


def min_width() -> int:
    try:
        return max(1, int(os.environ.get("CORESMITH_SRAM_MIN_WIDTH", "") or DEFAULT_MIN_WIDTH))
    except ValueError:
        return DEFAULT_MIN_WIDTH


def macro_depth() -> int:
    """Depth at/above which a raw addressed array must be a macro regardless of bits."""
    try:
        return max(2, int(os.environ.get("CORESMITH_SRAM_MACRO_DEPTH", "") or DEFAULT_MACRO_DEPTH))
    except ValueError:
        return DEFAULT_MACRO_DEPTH


def wrapper_lib_path() -> str:
    """Absolute path to the shared cs_sram wrapper library."""
    return str(Path(__file__).resolve().parent / "rtl_lib" / "cs_sram.v")


# A module spanning `module <name> ... endmodule`. Verilog modules don't nest,
# so a non-greedy match to the first endmodule is correct.
_MODULE_RE = re.compile(r"\bmodule\s+(\w+)\b(.*?)\bendmodule\b", re.DOTALL)

# An unpacked memory array: `reg [signed] [MSB:LSB] NAME [HI:LO];`
# (the packed range is optional -> 1-bit-wide memory). Whitespace/newlines ok.
_MEM_RE = re.compile(
    r"\breg\b\s*(?:signed\s*)?"
    r"(?:\[\s*(\d+)\s*:\s*(\d+)\s*\]\s*)?"          # optional packed [MSB:LSB]
    r"(\w+)\s*"                                       # name
    r"\[\s*(\d+)\s*:\s*(\d+)\s*\]\s*;",              # unpacked [A:B]
    re.DOTALL,
)


@dataclass
class UnwrappedMem:
    module: str
    name: str
    width: int
    depth: int

    @property
    def bits(self) -> int:
        return self.width * self.depth

    def describe(self) -> str:
        return (f"{self.module}.{self.name} "
                f"[{self.width}b x {self.depth}] = {self.bits} bits")


# A module whose name looks like an SRAM macro (our generic wrapper OR a macro's
# own behavioral sim-model, e.g. `sky130_sram_2kbyte_1rw1r_32x512_8`,
# `openram_sky130_64kbyte_...`, `sram_1rw_32_512_8_sky130`). A `reg mem[]` inside
# one of these IS the macro -- legitimately behavioral, not a violation.
_MACRO_MODULE_RE = re.compile(
    r"(^cs_sram)|(^cs_fpmem)|(^cs_rom)|(sram)|(^openram)|(^rom_)", re.IGNORECASE
)


def _is_macro_module(name: str) -> bool:
    return bool(_MACRO_MODULE_RE.search(name))


def detect_unwrapped_memories(
    rtl_text: str,
    threshold: int | None = None,
    min_words: int | None = None,
) -> list[UnwrappedMem]:
    """Return raw memory arrays that are NOT a macro/wrapper sim-model.

    A violation is storage declared in an ordinary block module (not a cs_sram
    wrapper or an SRAM macro's own behavioral model) that trips ANY of three
    independent, env-tunable thresholds (strict ``>``):

    * ``big``  -- total bits ``width * depth > min_bits()`` (default 2000)
    * ``wide`` -- word width ``width > min_width()``        (default 128)
    * ``deep`` -- depth ``depth > macro_depth()``           (default 256)

    The flag is ``big or wide or deep`` -- aggressive by design (see the module
    header). It targets raw combinational-read ``reg[] mem[]`` arrays; the remedy
    is a registered ``cs_fpmem`` OR a ``cs_sram`` macro (never a deadlock).

    ``threshold`` overrides the bits trigger only (back-compat). ``min_words`` is
    accepted but IGNORED -- the depth trigger is now driven by ``macro_depth()``.
    """
    thr = threshold if threshold is not None else min_bits()
    mw = min_width()
    mdd = macro_depth()
    out: list[UnwrappedMem] = []
    for mod_match in _MODULE_RE.finditer(rtl_text):
        mod_name = mod_match.group(1)
        if _is_macro_module(mod_name):
            continue  # cs_sram wrapper body OR a macro's own sim-model -> allowed
        body = mod_match.group(2)
        for m in _MEM_RE.finditer(body):
            msb, lsb = m.group(1), m.group(2)
            width = (int(msb) - int(lsb) + 1) if msb is not None else 1
            hi, lo = int(m.group(4)), int(m.group(5))
            depth = abs(hi - lo) + 1
            name = m.group(3)
            big = (width * depth > thr)   # large by total bits
            wide = (width > mw)           # wide word -> big read/write bus
            deep = (depth > mdd)          # deep -> N:1 read-mux fanout
            if big or wide or deep:
                out.append(UnwrappedMem(mod_name, name, width, depth))
    return out


def uses_wrapper(rtl_text: str) -> bool:
    """True if the RTL instantiates any CoreSmith memory primitive.

    Covers the macro-backed wrappers (cs_sram_1rw/1rw1r), the reference
    flop-memory primitives (cs_fpmem_1rw/1rw1r), and the mask-ROM primitive
    (cs_rom_1r) -- all live in the shared rtl_lib/cs_sram.v, so any one of them
    means that lib must be read in lint/sim/synth.
    """
    pat = r"\bcs_(?:sram_1rw(?:1r)?|fpmem_1rw(?:1r)?|rom_1r)\b"
    return bool(re.search(pat + r"\s*#?\s*\(", rtl_text)) or \
        bool(re.search(pat + r"\s+\w+\s*\(", rtl_text))


def gate_memory_wrapping(rtl_text: str, threshold: int | None = None) -> tuple[bool, list[str]]:
    """Deterministic gate verdict for the SRAM-wrapper contract.

    Returns ``(ok, reasons)``. ``ok`` is False when any storage above the
    threshold is a raw reg array rather than a cs_sram wrapper instance.
    """
    violations = detect_unwrapped_memories(rtl_text, threshold)
    if not violations:
        return True, []
    reasons = []
    for v in violations:
        reasons.append(
            f"memory {v.describe()} is a raw `reg [] mem []` array read as a bare "
            f"combinational mux -- forbidden (it becomes an unregistered N:1 read "
            f"mux: slow + congesting). Instantiate a CoreSmith memory primitive "
            f"instead: {_suggest_wrapper(v.width, v.depth)}"
        )
    return False, reasons


def _suggest_wrapper(width: int, depth: int) -> str:
    """Recommend the right memory primitive. A FLOP array is fine -- it just has
    to be the registered `cs_fpmem` template, not a raw comb-read array. Deep/large
    memories prefer the macro-backed `cs_sram` (single-cycle, dense); shallow ones
    use `cs_fpmem` (registered flop). Both are acceptable; a raw reg array is not."""
    fpmem = f"cs_fpmem_1rw #(.WIDTH({width}), .DEPTH({depth}))  (registered flop array -- a flop array is fine, just capture the read)"
    sram = f"cs_sram_1rw #(.WIDTH({width}), .DEPTH({depth}))  (SRAM macro -- single-cycle, denser)"
    if depth >= macro_depth() or width * depth >= min_bits():
        return f"{sram}; OR, if you want it in flops, {fpmem}"
    return f"{fpmem}; OR, for a denser single-cycle memory, {sram}"


# --- SRAM area accounting (so the uarch/PPA gate prices in RAM) ---------------
# sky130 1rw/1rw1r OpenRAM cells incl. periphery & routing ~= this many um^2 per
# stored bit (env-tunable). A cs_sram blackboxes to 0 flops + 0 std-cell area, so
# without this the architect pays nothing for memory; this converts wrapped bits
# into a die-area cost the area gate can budget against.
DEFAULT_UM2_PER_BIT = 1.7

# default wrapper geometry if an instance omits the params (cs_sram_1rw defaults)
_DEFAULT_W, _DEFAULT_D = 32, 512

_CSSRAM_RE = re.compile(r"\bcs_sram_1rw(?:1r)?\b")
_MODULE_DECL_RE = re.compile(r"module\s+$")
_W_RE = re.compile(r"\.WIDTH\s*\(\s*(\d+)\s*\)")
_D_RE = re.compile(r"\.DEPTH\s*\(\s*(\d+)\s*\)")


def um2_per_bit() -> float:
    try:
        return max(0.01, float(os.environ.get("CORESMITH_SRAM_UM2_PER_BIT", "") or DEFAULT_UM2_PER_BIT))
    except ValueError:
        return DEFAULT_UM2_PER_BIT


def sram_instances(rtl_text: str) -> list[tuple[int, int]]:
    """Return [(width, depth), ...] for each cs_sram wrapper INSTANTIATION.

    Window-based (not a single regex) so nested `#(.WIDTH(8),.DEPTH(N))` parens
    parse correctly; skips the `module cs_sram_*` definition itself; counts only
    instantiations that carry an explicit .WIDTH/.DEPTH param.
    """
    out: list[tuple[int, int]] = []
    for m in _CSSRAM_RE.finditer(rtl_text):
        if _MODULE_DECL_RE.search(rtl_text[max(0, m.start() - 12):m.start()]):
            continue  # the wrapper module declaration, not an instance
        end = rtl_text.find(";", m.end())
        window = rtl_text[m.end(): end if end != -1 else m.end() + 400]
        w = _W_RE.search(window)
        d = _D_RE.search(window)
        if not (w or d):
            continue
        out.append((int(w.group(1)) if w else _DEFAULT_W,
                    int(d.group(1)) if d else _DEFAULT_D))
    return out


def sram_bits(rtl_text: str) -> int:
    return sum(w * d for w, d in sram_instances(rtl_text))


# --- ROM area accounting (mask ROM via OpenRAM rom_compiler) ------------------
# Two-point calibration against sky130 OpenRAM rom_compiler output (measured
# LEF SIZE, route_supplies=ring):
#   1 KiB  (1024x8,  8,192 b)  = 175.13 x 122.55 um =  21,462 um^2
#   16 KiB (4096x32, 131,072 b) = 315.13 x 608.12 um = 191,637 um^2
# -> affine fit: area = 10,120 + 1.385 * bits (both points within 0.4%).
# A mask ROM carries a fixed decoder/periphery overhead plus a per-bit slope
# well below real small SRAM macros (the pre-built 2 KiB 1rw1r SRAM measures
# ~11.6 um^2/bit all-in). Override the slope with CORESMITH_ROM_UM2_PER_BIT
# when a better fit exists; exact generated-LEF areas remain authoritative.
DEFAULT_ROM_UM2_PER_BIT = 1.385
ROM_MACRO_OVERHEAD_UM2 = 10120.0

_CSROM_RE = re.compile(r"\bcs_rom_1r\b")


def rom_um2_per_bit() -> float:
    try:
        return max(0.001, float(
            os.environ.get("CORESMITH_ROM_UM2_PER_BIT", "") or DEFAULT_ROM_UM2_PER_BIT))
    except ValueError:
        return DEFAULT_ROM_UM2_PER_BIT


def rom_area_um2(bits: int) -> float:
    """Affine mask-ROM area model (per-macro overhead + per-bit slope)."""
    if bits <= 0:
        return 0.0
    return ROM_MACRO_OVERHEAD_UM2 + float(bits) * rom_um2_per_bit()


def rom_instances(rtl_text: str) -> list[tuple[int, int]]:
    """[(width, depth), ...] for each cs_rom_1r INSTANTIATION (mirrors
    sram_instances; skips the module definition itself)."""
    out: list[tuple[int, int]] = []
    for m in _CSROM_RE.finditer(rtl_text):
        if _MODULE_DECL_RE.search(rtl_text[max(0, m.start() - 12):m.start()]):
            continue
        end = rtl_text.find(";", m.end())
        window = rtl_text[m.end(): end if end != -1 else m.end() + 400]
        w = _W_RE.search(window)
        d = _D_RE.search(window)
        if not (w or d):
            continue
        out.append((int(w.group(1)) if w else _DEFAULT_W,
                    int(d.group(1)) if d else 1024))
    return out


def rom_bits(rtl_text: str) -> int:
    return sum(w * d for w, d in rom_instances(rtl_text))


def estimate_rom_area_um2(rtl_text: str) -> float:
    """Estimated die area (um^2) of all cs_rom_1r instances in a block, priced
    per-instance with the affine mask-ROM model (each macro pays overhead)."""
    return sum(rom_area_um2(w * d) for w, d in rom_instances(rtl_text))


def estimate_sram_area_um2(rtl_text: str, per_bit: float | None = None) -> float:
    """Estimated die area (um^2) of all cs_sram wrapper instances in a block.

    Deterministic bits*um2/bit estimate so the PPA/area gate can budget RAM even
    with no PDK present. (A PDK-accurate path could resolve each geometry via
    openram_gen and sum real LEF areas; the estimate is the floor.)
    """
    ppb = per_bit if per_bit is not None else um2_per_bit()
    return sram_bits(rtl_text) * ppb


# --- synth/backend resolution: cs_sram geometry -> concrete macro -------------

def resolve_macro(width: int, depth: int, ports: str = "1rw", registry=None):
    """Resolve a wrapper's (WIDTH, DEPTH) to a pre-built macro if one exists.

    Returns a ``MacroInfo`` for an exact pre-built match, else ``None`` (the
    caller may fall back to ``openram_gen.ensure_macro`` to compose/generate).
    Kept dependency-light so it imports cleanly with no PDK present.
    """
    try:
        from orchestrator.langgraph.openram_gen import find_exact
    except Exception:
        return None
    return find_exact(words=depth, data_bits=width, registry=registry)


# --- Part B: backend/PD synth selects the MACRO impl on the cs_sram wrappers ---
#
# The cs_sram_*/cs_mem_*/cs_rom_1r wrappers default MEM_IMPL="BEHAV" so sim/DV
# and every already-generated instantiation are byte-for-byte unchanged. Only
# the BACKEND (physical-design) synth flips them to "MACRO" -- a yosys `chparam`
# that re-derives these modules with MEM_IMPL="MACRO" BEFORE `hierarchy`, so
# their generate block selects the empty `cs_mem_macro_shell`/`cs_rom_macro_shell`
# leaf (0 storage flip-flops) instead of the behavioral flop-array body. Proven
# on yosys 0.65: a wrapped 1rw1r memory synthesizes to `cs_mem_macro_shell`
# instances with zero `$dff`/`$mem` under the directive, vs a `$mem_v2` + capture
# flops without it.
#
# cs_fpmem_* is DELIBERATELY excluded: it hard-passes MEM_IMPL("FLOP") to its
# cs_mem_* instance, and an explicit instantiation override wins over the
# chparam-changed module default, so the blessed flop tier stays flops (verified).
#
# The chparam must run BEFORE the first `hierarchy` derivation: setting the
# parameter on the base module changes the DEFAULT the un-overridden cs_sram
# instances pick up when hierarchy derives them; running it after derivation is
# a no-op on the already-baked `$paramod...` copies.
SRAM_MACRO_CHPARAM_MODULES = (
    "cs_sram_1rw",
    "cs_sram_1rw1r",
    "cs_rom_1r",
    "cs_mem_1rw",
    "cs_mem_1rw1r",
)


def backend_sram_macro_enabled() -> bool:
    """Whether the backend/PD synth path selects the SRAM MACRO impl.

    Default ON. Set ``CORESMITH_SRAM_MACRO=0`` to keep the wrappers behavioral
    in the backend synth too (the pre-fix behavior: wrapped memories stay flop
    arrays). Simulation/DV is unaffected either way -- this only governs the
    physical-design synth path.
    """
    from orchestrator.profile import ensure_applied, flag_enabled
    ensure_applied()
    return flag_enabled("CORESMITH_SRAM_MACRO", default=True)


def pnr_macro_placement_enabled() -> bool:
    """Whether the backend MATERIALIZES a bound `cs_mem_macro_shell` into its
    concrete SRAM macro in the PnR/DRC/LVS netlist (so PnR actually PLACES the
    memory) AND hard-fails a memory-absent layout.

    Default ON. Set ``CORESMITH_PNR_MACRO_PLACEMENT=0`` to restore the pre-fix
    behavior: the shell reaches PnR as an empty tie-0 stub (memory physically
    absent), and the placed-macro assertion is skipped. Only governs the
    physical-design netlist; simulation/DV are unaffected.
    """
    from orchestrator.profile import ensure_applied, flag_enabled
    ensure_applied()
    return flag_enabled("CORESMITH_PNR_MACRO_PLACEMENT", default=True)


def macro_compose_tiles_enabled() -> bool:
    """Whether a `cs_mem_macro_shell` resolved to an N-tile COMPOSITION plan is
    MATERIALIZED into its concrete tiles (bound, tiled into `<design>_macro.v`,
    every tile placed, and covered by the memory-absent assertion).

    Default ON. Set ``CORESMITH_MACRO_COMPOSE_TILES=0`` to restore the pre-fix
    behavior: a composition plan is only logged and then DROPPED -- the shell
    reaches PnR unresolved and the memory is physically absent (the false-pass
    this closes). Only governs plans (a shell that binds directly to a single
    concrete macro is unaffected either way).
    """
    from orchestrator.profile import ensure_applied, flag_enabled
    ensure_applied()
    return flag_enabled("CORESMITH_MACRO_COMPOSE_TILES", default=True)


def pnr_cellcount_guard_enabled() -> bool:
    """Whether PnR hard-fails when the linked/placed cell count is far below the
    synth ``gate_count`` (a sub-block linked as the chip top -- e.g. a 154-cell
    memory fragment where synth reported 4,682 gates for the full chip).

    Default ON. Set ``CORESMITH_PNR_CELLCOUNT_GUARD=0`` to restore the pre-fix
    behavior (no guard). Defense-in-depth backstop independent of the top-name
    heuristic; catches ANY future fragment-as-top.
    """
    from orchestrator.profile import ensure_applied, flag_enabled
    ensure_applied()
    return flag_enabled("CORESMITH_PNR_CELLCOUNT_GUARD", default=True)


def pnr_cellcount_min_ratio() -> float:
    """Minimum acceptable ratio of linked/placed cells to synth ``gate_count``
    before the cell-count guard fires. Default 0.5 (fail below ~half): PnR only
    ADDS physical cells (tap/fill/buffers) so the linked count is a lower bound
    on synth, and a gross deficit (e.g. 154/4,682 ~= 0.03) means a fragment
    was linked. Override with ``CORESMITH_PNR_CELLCOUNT_MIN_RATIO``."""
    import os
    try:
        v = float(os.environ.get("CORESMITH_PNR_CELLCOUNT_MIN_RATIO", "") or 0.5)
    except (TypeError, ValueError):
        return 0.5
    # Clamp to a sane (0, 1] band so a bad override can't disable or invert it.
    return min(max(v, 0.0), 1.0)


def backend_sram_macro_directive(*, force: bool = False) -> str:
    """The yosys `chparam` line that selects MACRO on the SRAM wrappers, or ""
    when disabled.

    Insert it in the backend synth script AFTER ``read_verilog`` (of the design
    + the ``rtl_lib/cs_sram.v`` wrapper library) and BEFORE ``hierarchy``.
    Naming a wrapper module absent from the read sources is a harmless yosys
    warning (``Selection ... did not match any module``), never an error, so the
    single static line is safe for any design.

    ``force`` bypasses the env gate (used by the Part-D memory-as-flops probe,
    which must model the MACRO-selected netlist regardless of the run flag).
    """
    if not force and not backend_sram_macro_enabled():
        return ""
    return 'chparam -set MEM_IMPL "MACRO" ' + " ".join(SRAM_MACRO_CHPARAM_MODULES)


# --- Part D: post-synth memory-as-flops hard-block gate -----------------------
#
# The backstop that turns a silently-shipped flop-array memory into an early,
# honest failure. It runs AFTER synthesis on the MACRO-selected netlist (Part B),
# where a properly wrapped memory is a `cs_mem_macro_shell` (0 flops) -- so any
# residual, above-threshold memory realized as a flop array is a real defect:
#   * a raw `reg [..] mem [..]` array the author never wrapped, or
#   * a cs_sram whose geometry the backend could not bind to a macro.
# It reuses the SRAM-macro threshold (:func:`min_bits` / :func:`macro_depth`) and
# NEVER flags the legitimate cs_fpmem FLOP tier or small sub-threshold memories.

_CSFPMEM_RE = re.compile(r"\bcs_fpmem_1rw(?:1r)?\b")


def fpmem_instances(rtl_text: str) -> list[tuple[int, int]]:
    """[(width, depth), ...] for each cs_fpmem_* INSTANTIATION (mirrors
    :func:`sram_instances`; skips the module definitions themselves)."""
    out: list[tuple[int, int]] = []
    for m in _CSFPMEM_RE.finditer(rtl_text):
        if _MODULE_DECL_RE.search(rtl_text[max(0, m.start() - 12):m.start()]):
            continue  # the wrapper module declaration, not an instance
        end = rtl_text.find(";", m.end())
        window = rtl_text[m.end(): end if end != -1 else m.end() + 400]
        w = _W_RE.search(window)
        d = _D_RE.search(window)
        if not (w or d):
            continue
        out.append((int(w.group(1)) if w else 8,
                    int(d.group(1)) if d else 16))
    return out


def mem_flop_gate_enabled() -> bool:
    """Post-synth memory-as-flops gate (default ON).

    Disable with ``CORESMITH_MEM_FLOP_GATE=0`` to restore the pre-fix behavior
    (an above-threshold flop-array memory advances silently).
    """
    from orchestrator.profile import ensure_applied, flag_enabled
    ensure_applied()
    return flag_enabled("CORESMITH_MEM_FLOP_GATE", default=True)


def _fpmem_exempt(width: int, depth: int, fpmem_geoms: list[tuple[int, int]]) -> bool:
    """A memory whose geometry matches a declared cs_fpmem instance is the
    blessed flop tier -- exempt it regardless of size."""
    return (width, depth) in set(fpmem_geoms or [])


def gate_memory_as_flops(
    memories: list[tuple[int, int]],
    *,
    fpmem_geoms: list[tuple[int, int]] | None = None,
    threshold_bits: int | None = None,
    macro_words: int | None = None,
) -> tuple[bool, list[str]]:
    """Deterministic verdict for the post-synth memory-as-flops gate.

    ``memories`` is the list of ``(width, depth)`` flop-array / inferred-memory
    geometries measured in the MACRO-selected netlist (a properly bound cs_sram
    is a shell and contributes NOTHING here). Returns ``(ok, reasons)``: ``ok``
    is False when any single memory is at/above the SRAM-macro threshold (by total
    bits, or by depth alone) yet realized as flops -- and is NOT a cs_fpmem.

    Per-memory (never summed) so many small sub-threshold arrays never add up to
    a false positive; cs_fpmem geometries and sub-threshold memories pass.
    """
    thr = threshold_bits if threshold_bits is not None else min_bits()
    md = macro_words if macro_words is not None else macro_depth()
    fpg = fpmem_geoms or []
    reasons: list[str] = []
    for width, depth in memories:
        if width <= 0 or depth <= 0:
            continue
        if _fpmem_exempt(width, depth, fpg):
            continue
        bits = width * depth
        big = (bits >= thr)      # large by total bits
        deep = (depth >= md)     # deep enough on its own (read-mux fanout)
        if not (big or deep):
            continue
        reasons.append(
            f"memory [{width}b x {depth}] = {bits:,} bits synthesized to a "
            f"flip-flop array above the SRAM-macro threshold ({thr:,} bits / "
            f"{md}-word depth) -- an unroutable flop memory + N:1 read mux. "
            f"Instantiate cs_sram_1rw/cs_sram_1rw1r (the backend binds it to a "
            f"real SRAM macro, 0 flops) or, if flops are intended below the "
            f"threshold, the registered cs_fpmem_* tier -- never a raw "
            f"`reg [..] mem [..]` array."
        )
    return (not reasons), reasons
