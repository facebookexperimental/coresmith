# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Deterministic post-synthesis PPA gate.

CoreSmith's per-block ``synthesize`` step today sets ``synth_success`` from
the Yosys return code alone -- "it mapped to Sky130 cells without erroring".
That misses the failure class that matters most for PPA: RTL that compiles
*cleanly* but is 5x too big / 50x too slow because a memory that should be an
SRAM macro synthesized to a flip-flop array (plus an N:1 read mux).

This module is the deterministic *judge* (the ``synth_fixer`` LLM stays the
*fixer*). It compares the synthesized result against the block's uArch
**storage budget** (``flip_flop_budget``), an optional area budget, and a
pre-layout WNS from OpenSTA, and flags only DIVERGENCE from the spec's
intent. A memory the uArch deliberately kept as flops (no available sky130
macro fits its ports/latency/size) carries a budget that already accounts
for those flops, so it is not flagged.

Everything here is pure + deterministic except :func:`run_pre_layout_sta`,
which shells to OpenSTA and degrades to ``None`` when the tool is absent.
The gate is wired into ``route_after_synth`` behind ``CORESMITH_PPA_GATE=1``;
default-off preserves current behavior.
"""
from __future__ import annotations

import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# A Sky130 flip-flop cell has "df" in its name (dfxtp, dfrtp, dfstp, dfbbn,
# edfxtp, sdfxtp, ...); a latch has "dl" (dlxtp) -- excluded. The generic
# (pre-libmap) forms are the $dff family ($dff, $adff, $sdff, $dffe,
# $sdffe, $dffsr).
# IGNORECASE so post-`abc -g`/techmap UPPERCASE flop cells
# ($_DFF_P_, $_SDFFE_*, $_SDFFCE_*, $_DFFSR_*) are counted under
# CORESMITH_SYNTH_GENERIC -- the generic synth path emits those, and a
# case-sensitive lowercase-only pattern silently counted 0 flops there,
# making the PPA FF-budget gate toothless under generic synth.
_FLOP_CELL_RE = re.compile(
    r"sky130_fd_sc_hd__\w*df\w*|\$\w*dff\w*", re.IGNORECASE
)
# FF-BUDGET GRANULARITY CONVENTION (rung3-fixes-1, minor 5): ``flip_flop_budget``
# is a BIT-LEVEL flip-flop count -- the number of 1-bit FFs, i.e. a 32-bit
# register counts as 32, not 1. This matches the granularity a Sky130-mapped
# ``ff_count`` reports (each mapped ``sky130_..._df*`` cell is one 1-bit flop)
# and the granularity of ``count_ff_bits_from_stat`` (sums ``$dff`` *widths* from
# a ``stat -width`` table). The memory-preserving generic probe was changed to
# emit bit-level ``logic_ff`` so the budget comparison is apples-to-apples on
# every path (real synth and SKIP_SYNTH). Keep spec wording in this unit.
_FF_BUDGET_RE = re.compile(r"flip_flop_budget[^\d\n]{0,16}(\d[\d,]*)", re.IGNORECASE)
_INT_TOKEN_RE = re.compile(r"^\d[\d,]*$")


def _cell_count(line: str) -> int | None:
    """The per-cell-type count on a Yosys ``stat`` line, format-agnostic.

    Yosys 0.65 prints ``<count>  <cellname>`` while 0.33 prints
    ``<cellname>  <count>`` -- and a cell name can itself contain digits
    (``$mem_v2``). So take the standalone *pure-integer* whitespace token,
    which is the count in both layouts and never the digits inside a name.
    """
    for tok in line.split():
        if _INT_TOKEN_RE.match(tok):
            return int(tok.replace(",", ""))
    return None


def parse_ff_budget(uarch_spec_text: str) -> int | None:
    """Pull the ``flip_flop_budget`` integer out of a uArch spec, or None.

    Tolerates the documented forms: ``flip_flop_budget ≈ 1200``,
    ``` `flip_flop_budget`: 3,400 flip-flops ```, ``flip_flop_budget = 800 FF``.

    CONVENTION: the returned budget is a BIT-LEVEL 1-bit-flop count (a 32-bit
    register = 32), matching ``count_ff_bits_from_stat`` and Sky130 ``ff_count``.
    See the ``_FF_BUDGET_RE`` comment above.
    """
    if not uarch_spec_text:
        return None
    m = _FF_BUDGET_RE.search(uarch_spec_text)
    if not m:
        return None
    return int(m.group(1).replace(",", ""))


# die/area budget INCLUDING SRAM macro area -- the dimension that prices in RAM
# (FF budget can't, since cs_sram blackboxes to 0 flops). Matches
# `area_budget_um2`, `die_area_budget_um2`, `area_budget` (um2 assumed).
#
# PARSE ROBUSTNESS (defect 2): the old regex was
# ``area_budget(?:_um2)?[^\d\n]{0,16}(\d...)`` -- "the first digit run within 16
# chars of the token". That grabs a STRAY digit near the token instead of the
# budget: a glue/wrapper spec like ``area_budget_um2` (see §2) ...`` parsed to
# **2 um2**, and the PPA gate then held a correct ~882-um2 pad-adapter to a
# 2-um2 budget (400x false over-budget). Symmetrically, when the real value sat
# just past the 16-char window -- ``area_budget_um2`: glue is expected below
# `6000 um2``` -- it matched nothing and the gate silently could not judge.
#
# The value is now anchored to a real ASSIGNMENT/COMPARISON signal: the number
# must be immediately preceded by an operator/qualifier (``= : ~ < <= below
# under approx target budget of ...``) OR immediately followed by a µm²/um2
# unit. A bare digit that is neither (a section ref, a "2 GPIO banks" count, a
# footnote) is skipped, and a unit-tagged value anywhere on the same line is
# accepted regardless of distance.
_AREA_TOKEN_RE = re.compile(r"(?:die_)?area_budget(?:_um2)?", re.IGNORECASE)
_AREA_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
# Assignment / comparison / qualifier IMMEDIATELY before the number (anchored to
# the end of the pre-number text) -- the signal that the number is the budget.
_AREA_ASSIGN_RE = re.compile(
    r"(?:[:=]|≈|~|≲|<=?|≤|>=?|≥|"
    r"\b(?:below|under|about|around|approx(?:\.|imately)?|max(?:imum)?|"
    r"at\s+most|up\s+to|of|target(?:ing|ed)?|budget(?:ed)?)\b)"
    r"\s*[`'\"(\[]?\s*$",
    re.IGNORECASE,
)
# A µm²/um2 unit IMMEDIATELY after the number -- an equally strong signal.
_AREA_UNIT_RE = re.compile(
    r"\s*`?\s*(?:um\s*\^?\s*2|µm\s*\^?\s*2|um2|µm2|µm²|sq(?:uare)?\.?\s*"
    r"(?:micron|um|µm))",
    re.IGNORECASE,
)


def parse_area_budget(uarch_spec_text: str) -> float | None:
    """Pull the per-block die-area budget (um^2, SRAM included) from a uArch spec.

    Tolerates ``area_budget_um2 = 250000``, `` `die_area_budget_um2`: 1,200,000 ``,
    ``area_budget_um2: glue is expected below `6000 um2` ``. Skips stray digits
    near the token (section refs, GPIO-bank counts) that are neither assigned nor
    unit-tagged. Returns None when no area budget is declared (gate then can't
    judge area).
    """
    if not uarch_spec_text:
        return None
    for tm in _AREA_TOKEN_RE.finditer(uarch_spec_text):
        nl = uarch_spec_text.find("\n", tm.end())
        window = uarch_spec_text[tm.end(): nl if nl != -1 else len(uarch_spec_text)]
        for nm in _AREA_NUM_RE.finditer(window):
            pre = window[: nm.start()]
            has_assign = bool(_AREA_ASSIGN_RE.search(pre))
            has_unit = bool(_AREA_UNIT_RE.match(window, nm.end()))
            if has_assign or has_unit:
                try:
                    return float(nm.group(0).replace(",", ""))
                except ValueError:
                    return None
    return None


# Structural glue / pad-adapter / wrapper blocks (a Caravel ``user_project_wrapper``,
# a pin-mux, a GPIO pad ring) are tie-offs + bit routing: their real area is a
# few hundred um2 but NONZERO. An LLM routinely writes a nonsensical sub-unit
# ``area_budget_um2 ~2`` for them, and a per-block area gate then FALSE-FLAGS the
# (correct) wrapper as hundreds-x over a 2-um2 budget. Below this floor a glue
# block's area budget is not a real constraint -- the chip-level die cap is
# authoritative. Sibling of the FF-budget ``ff_floor`` guard in evaluate_ppa().
GLUE_AREA_FLOOR_UM2 = 2000.0

_GLUE_BLOCK_NAME_RE = re.compile(
    r"user_project_wrapper|openframe\w*wrapper|pad[_-]?adapter|pad[_-]?ring|"
    r"pin[_-]?mux|gpio[_-]?(?:ctrl|mux|adapter|ring)|caravel|io[_-]?(?:mux|ring|pad|ctrl)|"
    r"(?:^|_)wrapper(?:$|_)|(?:^|_)glue(?:$|_)|chip_?top",
    re.IGNORECASE,
)
_GLUE_SPEC_RE = re.compile(
    r"structural\s+glue|pin\s+adapter|pad\s+adapter|pad\s+ring|"
    r"no\s+(?:wrapper-local\s+)?(?:storage|arithmetic|fsm|state|datapath)|"
    r"pure(?:ly)?\s+(?:structural|combinational)\s+(?:glue|wrapper|adapter|routing|"
    r"pin[_-]?mux)",
    re.IGNORECASE,
)


def is_structural_glue_block(block_name: str = "", uarch_spec_text: str = "") -> bool:
    """True when a block is a pad-adapter / pin-mux / Caravel wrapper -- pure
    structural glue whose area budget must not be held to a sub-cell value.

    Detected by block name (``user_project_wrapper``, ``*_pad_adapter``,
    ``*pin_mux*``, ...) or by a spec that self-describes as structural glue
    (``structural glue only``, ``pin adapter``, ``no ... storage, arithmetic,
    FSM``).
    """
    if block_name and _GLUE_BLOCK_NAME_RE.search(block_name):
        return True
    if uarch_spec_text and _GLUE_SPEC_RE.search(uarch_spec_text):
        return True
    return False


def floor_area_budget(
    area_budget_um2: float | None,
    block_name: str = "",
    uarch_spec_text: str = "",
) -> float | None:
    """Clamp a structural glue/wrapper/adapter block's parsed area budget up to
    :data:`GLUE_AREA_FLOOR_UM2` so a parse artifact (or a genuinely tiny declared
    value like ``~2 um2``) can't false-flag a pin-mux on area.

    A ``None`` budget (no declaration -> gate can't judge) and a non-glue block
    pass through untouched, so this only ever RELAXES the gate for glue.
    """
    if area_budget_um2 is None:
        return None
    if not is_structural_glue_block(block_name, uarch_spec_text):
        return area_budget_um2
    if area_budget_um2 < GLUE_AREA_FLOOR_UM2:
        return GLUE_AREA_FLOOR_UM2
    return area_budget_um2


def count_flops_from_stat(stat_text: str) -> int:
    """Sum flip-flop CELL COUNTS in a Yosys ``stat`` cell table.

    GRANULARITY: this returns a *cell* count. On a Sky130-mapped stat each
    ``sky130_..._df*`` cell is one 1-bit flop, so the cell count IS the
    bit-level FF count. On a *generic* (pre-techmap) stat a multi-bit ``$dff``
    is ONE cell, so this returns WORD-level counts there -- for a bit-level
    total off a generic stat use :func:`count_ff_bits_from_stat` (``stat
    -width``). The memory-preserving generic probe uses the bit-level counter
    so its ``logic_ff`` lines up with the bit-level ``flip_flop_budget``.

    Counts ONLY genuine ``stat`` cell-listing lines -- ``<count> <cellname>``
    (yosys 0.65) or ``<cellname> <count>`` (yosys 0.33): exactly two
    whitespace tokens, one a pure integer count, the other a flop cell name.

    This deliberately ignores Yosys *log* lines, which also mention ``$dff``
    but carry many tokens and stray integers -- e.g. opt_dff's
    ``Setting constant 1-bit at position 47 on $procdff$... ($dff) ...``. The
    old "regex-matches-anywhere + first-integer-token" parser summed those bit
    positions as phantom flops, reporting a true ~1,178-flop block as ~279k and
    falsely tripping the PPA gate. A real cell line is always exactly two
    tokens, so that shape is the reliable discriminator.
    """
    # HOT-PATCH (chip-lead): a full synth log can contain MULTIPLE Yosys `stat`
    # blocks (a mid-flow stat AND the final post-opt stat). Summing flop cell
    # lines across ALL of them double-counts FFs (e.g. a true 2,601-FF block was
    # reported as 5,202 = 2,601x2, falsely tripping the PPA FF-budget gate).
    # Count ONLY the final stat block: everything after the LAST stat-section
    # marker ("Printing statistics." / "=== <module> ==="). For single-block
    # inputs (e.g. the memory-preserving probe) no marker-splitting changes the
    # result, so existing behavior is preserved.
    total = 0
    for _cand in _stat_section_candidates(stat_text):
        total = 0
        for line in _cand.splitlines():
            toks = line.split()
            if len(toks) == 2:
                a, b = toks
                a_int, b_int = _INT_TOKEN_RE.match(a), _INT_TOKEN_RE.match(b)
                # exactly one pure-integer token + one (cell-name) token
                if bool(a_int) == bool(b_int):
                    continue
                num, name = (a, b) if a_int else (b, a)
            elif len(toks) == 3:
                # yosys 0.66+ liberty-aware stat: "<count> <area> <cellname>"
                num, _area, name = toks
                if not _INT_TOKEN_RE.match(num) or _INT_TOKEN_RE.match(name):
                    continue
            else:
                continue
            if _FLOP_CELL_RE.search(name):
                total += int(num.replace(",", ""))
        if total:
            break
    return total


# ``stat -width`` groups generic FF cells by port width, printing
# ``$dff_<W>`` / ``$sdffe_<W>`` / ``$adff_<W>`` / ``$dffsr_<W>`` etc. The
# trailing ``_<digits>`` on a ``$``-prefixed generic flop cell is the WIDTH.
# (Restricted to ``$``-prefixed names so a Sky130 drive-strength suffix like
# ``sky130_..._dfxtp_2`` is NOT mistaken for a width -- those cells are 1-bit.)
_FF_WIDTH_RE = re.compile(r"^\$\w*dff\w*?_(\d+)$", re.IGNORECASE)


def _stat_section_candidates(stat_text: str) -> list[str]:
    """Candidate stat-table slices, most-specific first.

    The old heuristic took the text after the LAST ``=== x ===`` marker --
    but step-log wrappers add their own ``=== STDERR ===`` / ``=== STDOUT ===``
    headers AFTER the yosys tables, so the "last section" was the (empty)
    stderr tail and every counter read 0 (armD defect #7, second mechanism).
    Yield sections from the last marker backwards, then the whole text; the
    caller uses the first candidate that actually parses to a nonzero count.
    """
    text = stat_text or ""
    markers = [m.start() for m in re.finditer(
        r"(?im)^\s*\d+(?:\.\d+)*\.\s*Printing statistics\.|^\s*===\s+\S+\s+===",
        text)]
    out = []
    for start in reversed(markers):
        out.append(text[start:])
    out.append(text)
    return out


def count_ff_bits_from_stat(stat_text: str) -> int:
    """Sum flip-flop BITS from a Yosys ``stat -width`` cell table.

    ``stat -width`` breaks generic FF cells out by width -- ``<count> $dff_<W>``
    -- so a 32-bit register (ONE ``$dff`` cell but 32 flip-flop BITS) is counted
    as ``1 x 32``. The bit total (sum of count x width) is the granularity the
    uArch ``flip_flop_budget`` is written in and the granularity a Sky130-mapped
    ``ff_count`` reports (1-bit cells), so comparing this against the budget is
    apples-to-apples. Summing plain cell COUNTS (:func:`count_flops_from_stat`)
    off a generic stat instead UNDER-reports (word-level) vs a bit-level budget
    -- the inconsistency this fixes (rung3-fixes-1, minor 5).

    A flop line with no ``_<width>`` suffix (a plain ``$dff`` from a stat WITHOUT
    ``-width``, or a non-generic 1-bit cell) counts as 1 bit each -- so this
    degrades gracefully to the word count on a stat that lacks ``-width``.
    """
    total = 0
    for _cand in _stat_section_candidates(stat_text):
        total = 0
        for line in _cand.splitlines():
            toks = line.split()
            if len(toks) == 2:
                a, b = toks
                a_int, b_int = _INT_TOKEN_RE.match(a), _INT_TOKEN_RE.match(b)
                if bool(a_int) == bool(b_int):
                    continue
                num, name = (a, b) if a_int else (b, a)
            elif len(toks) == 3:
                # yosys 0.66+ liberty-aware stat: "<count> <area> <cellname>"
                # (e.g. "2618 5.24E+04 sky130_fd_sc_hd__dfxtp_1"). The hard
                # two-token requirement skipped EVERY such line -> ff_count 0
                # against a 4,595-flop netlist (armD live, defect #7) -- the
                # FF-budget gate silently blind under the current toolchain.
                num, _area, name = toks
                if not _INT_TOKEN_RE.match(num) or _INT_TOKEN_RE.match(name):
                    continue
            else:
                continue
            if not _FLOP_CELL_RE.search(name):
                continue
            wm = _FF_WIDTH_RE.match(name)
            width = int(wm.group(1)) if wm else 1
            total += int(num.replace(",", "")) * width
        if total:
            break
    return total


_MEM_BITS_RE = re.compile(r"Number of memory bits:\s*(\d+)", re.IGNORECASE)


def parse_mem_from_stat(stat_text: str) -> dict[str, int]:
    """Pull inferred-memory counts from a Yosys ``stat`` (memory preserved).

    Counts ``$mem`` / ``$mem_v2`` cells directly rather than trusting the
    "Number of memories" summary -- post-``memory_collect`` that line reads 0
    on yosys 0.33 even with ``$mem_v2`` cells present.
    """
    text = stat_text or ""
    mb = _MEM_BITS_RE.search(text)
    count = 0
    for line in text.splitlines():
        if re.search(r"\$mem(?:_v\d+)?\b", line):
            c = _cell_count(line)
            if c is not None:
                count += c
    return {"mem_count": count, "mem_bits": int(mb.group(1)) if mb else 0}


def probe_synth_generic(
    rtl_path: str,
    top: str,
    *,
    yosys_bin: str = "yosys",
    timeout_s: int = 300,
) -> dict[str, Any] | None:
    """PDK-free, memory-PRESERVING synth probe for the PPA gate.

    Runs ``proc; opt; memory_collect; stat`` -- deliberately NOT
    ``memory_map`` -- so a correctly inferred RAM stays a ``$mem`` (counted as
    memory) instead of exploding into the flip-flops it would otherwise become.
    The gate keys off this so it never false-positives on a design that
    *correctly* used inferred SRAM. No liberty / PDK required (Yosys only).

    Returns ``None`` only when Yosys/RTL are absent (cannot judge). When Yosys
    runs, returns ``{logic_ff, mem_count, mem_bits, elaborated, reason}``;
    ``elaborated`` is False on timeout or a non-zero exit -- which is itself a
    synthesizability signal (a combinational loop or an enormous flop array
    that yosys can't elaborate in bounded time), NOT a "cannot judge".

    ``logic_ff`` is BIT-LEVEL (rung3-fixes-1, minor 5): the ``stat -width``
    breakdown is summed by :func:`count_ff_bits_from_stat` so a 32-bit register
    counts as 32 flops, matching the bit-level ``flip_flop_budget`` and Sky130
    ``ff_count``. (The old plain ``stat`` counted a multi-bit ``$dff`` as one
    word-level cell, which under-reported against a bit-level budget.)
    """
    yosys = shutil.which(yosys_bin)
    if not yosys or not rtl_path or not Path(rtl_path).exists():
        return None
    script = (
        f"read_verilog -sv {rtl_path}\n"
        f"hierarchy -top {top}\n"
        f"proc\nopt\nmemory_collect\nopt_clean\nstat -width\n"
    )
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".ys", delete=False) as fh:
            fh.write(script)
            ys = fh.name
        result = subprocess.run(
            [yosys, "-s", ys],  # NOT -q: that suppresses the stat (log-level) output
            capture_output=True, text=True, timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return {"logic_ff": None, "mem_count": 0, "mem_bits": 0,
                "elaborated": False,
                "reason": f"did not elaborate within {timeout_s}s "
                          "(combinational loop or oversized flop memory)"}
    except OSError:
        return None
    finally:
        try:
            Path(ys).unlink()
        except OSError:
            pass
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "")[-300:]
        return {"logic_ff": None, "mem_count": 0, "mem_bits": 0,
                "elaborated": False,
                "reason": f"yosys did not elaborate the design: {tail.strip()}"}
    out = result.stdout + "\n" + result.stderr
    mem = parse_mem_from_stat(out)
    return {
        # BIT-LEVEL FF count from the `stat -width` breakdown (minor 5).
        "logic_ff": count_ff_bits_from_stat(out),
        "mem_count": mem["mem_count"],
        "mem_bits": mem["mem_bits"],
        "elaborated": True,
        "reason": "",
    }


def count_cells_from_stat(stat_text: str) -> int | None:
    """Total cell count from a Yosys ``stat`` (``Number of cells: N``).

    Uses the LAST occurrence so a post-techmap stat (the materialized gate
    cloud) wins over any earlier mid-flow stat. Returns None if absent.
    """
    text = stat_text or ""
    # Format A (some yosys versions): "Number of cells:  N"
    ms = re.findall(r"Number of cells:\s*(\d[\d,]*)", text)
    if ms:
        return int(ms[-1].replace(",", ""))
    # Format B (stat breakdown table): the module-total line "<N> cells" --
    # distinct from "<N> wires" / "<N> wire bits" / "<N> public wires".
    mb = re.findall(r"(?m)^\s*(\d[\d,]*)\s+cells\s*$", text)
    if mb:
        return int(mb[-1].replace(",", ""))
    return None


def probe_synth_cellcount(
    rtl_path: str,
    top: str,
    *,
    yosys_bin: str = "yosys",
    timeout_s: int = 300,
    cwd: str | None = None,
) -> dict[str, Any] | None:
    """PDK-free generic-techmap probe that MATERIALIZES the gate cloud to count
    cells -- the synthesizability dimension the memory-PRESERVING FF probe is
    structurally blind to.

    ``probe_synth_generic`` deliberately stops at ``proc`` (no techmap), so a
    *combinational-LUT explosion* -- entropy coding VLC tables as big combinational
    LUTs, per-mode-replicated intra prediction, a wide record sliced by
    ``$func`` -- never materializes as gates and the FF-only gate can never
    fail on it. This probe runs ``proc; flatten; techmap; stat`` so the cells
    ARE materialized; a TIMEOUT or an enormous cell count is the
    un-synthesizability signal. No liberty / PDK required (generic techmap), so
    it runs even under ``CORESMITH_SKIP_SYNTH``.

    Returns ``None`` only when yosys/RTL are absent (cannot judge); otherwise
    ``{cell_count, elaborated, reason}`` with ``elaborated`` False on timeout or
    a non-zero exit (itself the un-synthesizability signal).
    """
    yosys = shutil.which(yosys_bin)
    if not yosys or not rtl_path or not Path(rtl_path).exists():
        return None
    script = (
        f"read_verilog -sv {rtl_path}\n"
        f"hierarchy -top {top}\n"
        f"proc\nflatten\nopt\ntechmap\nopt_clean\nstat\n"
    )
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".ys", delete=False) as fh:
            fh.write(script)
            ys = fh.name
        # C27: run yosys from `cwd` (the project root) when given, so RTL
        # $readmemh/$readmemb init files with project-relative paths (the
        # blessed cs_sram/cs_rom INIT_FILE convention, e.g.
        # "inputs/reference_codec_rom_images/quant_bank0.memh") resolve. Probing from a
        # temp dir made every ROM-initialized chip_top falsely
        # "not synthesizable" ("Can not open file ... for $readmemh").
        _cwd = cwd if (cwd and Path(cwd).is_dir()) else None
        result = subprocess.run(
            [yosys, "-s", ys], capture_output=True, text=True, timeout=timeout_s,
            cwd=_cwd,
        )
    except subprocess.TimeoutExpired:
        return {"cell_count": None, "elaborated": False,
                "reason": f"did not techmap within {timeout_s}s "
                          "(combinational-LUT explosion or unpipelined cloud)"}
    except OSError:
        return None
    finally:
        try:
            Path(ys).unlink()
        except OSError:
            pass
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "")[-300:]
        return {"cell_count": None, "elaborated": False,
                "reason": f"yosys techmap failed: {tail.strip()}"}
    out = result.stdout + "\n" + result.stderr
    return {"cell_count": count_cells_from_stat(out), "elaborated": True,
            "reason": ""}


def probe_synth_cellcount_multi(
    rtl_paths: list[str],
    top: str,
    *,
    yosys_bin: str = "yosys",
    timeout_s: int = 300,
    cwd: str | None = None,
) -> dict[str, Any] | None:
    """Multi-source variant of :func:`probe_synth_cellcount` for the integrated
    chip_top (top + block RTLs + the cs_mem library), already deduped by the
    caller so yosys sees no MODDUP. Same return contract.
    """
    yosys = shutil.which(yosys_bin)
    files = [p for p in (rtl_paths or []) if p and Path(p).exists()]
    if not yosys or not files:
        return None
    reads = "".join(f"read_verilog -sv {p}\n" for p in files)
    # `hierarchy -check`: an UNKNOWN child module is an ERROR, not a silent
    # blackbox. Without it yosys only warns and blackboxes the missing cell,
    # so an integrated top referencing a renamed/deleted block still
    # "elaborates" (audit F3: exactly how a drifted delivered top slips
    # through). The chip-level source set is complete by construction (deduped
    # manifest + wrapper lib), so unknowns here are genuine defects.
    script = (f"{reads}hierarchy -check -top {top}\n"
              f"proc\nflatten\nopt\ntechmap\nopt_clean\nstat\n")
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".ys", delete=False) as fh:
            fh.write(script)
            ys = fh.name
        # C27: see probe_synth_cellcount -- cwd=project_root makes project-
        # relative $readmemh INIT_FILE paths resolve during the probe.
        _cwd = cwd if (cwd and Path(cwd).is_dir()) else None
        result = subprocess.run(
            [yosys, "-s", ys], capture_output=True, text=True, timeout=timeout_s,
            cwd=_cwd,
        )
    except subprocess.TimeoutExpired:
        return {"cell_count": None, "elaborated": False,
                "reason": f"chip_top did not techmap within {timeout_s}s "
                          "(combinational-LUT explosion or unpipelined cloud)"}
    except OSError:
        return None
    finally:
        try:
            Path(ys).unlink()
        except OSError:
            pass
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "")[-300:]
        return {"cell_count": None, "elaborated": False,
                "reason": f"chip_top yosys techmap failed: {tail.strip()}"}
    out = result.stdout + "\n" + result.stderr
    return {"cell_count": count_cells_from_stat(out), "elaborated": True,
            "reason": ""}


def synth_cell_gate_enabled() -> bool:
    """Cell-explosion synthesizability guard (default ON).

    Runs the generic-techmap cell-count probe in the PPA gate so an
    un-synthesizable combinational-LUT explosion fails the block -- even under
    ``CORESMITH_SKIP_SYNTH``. Disable with ``CORESMITH_SYNTH_CELL_GATE=0``.
    """
    from orchestrator.profile import ensure_applied, flag_enabled
    ensure_applied()
    return flag_enabled("CORESMITH_SYNTH_CELL_GATE", default=True)


def max_cell_ceiling() -> int:
    """Per-block gate-level cell ceiling (sky130 generic cells).

    Generous by design -- it exists to catch a tens-of-millions-of-cells
    explosion, not to tightly constrain a real block. Override with
    ``CORESMITH_MAX_CELLS``.
    """
    try:
        return int(os.environ.get("CORESMITH_MAX_CELLS", "750000") or "750000")
    except ValueError:
        return 750000


def chip_top_min_cells() -> int:
    """Degenerate-collapse floor for the INTEGRATED chip_top.

    Elaboration + a count under the max ceiling is necessary but NOT
    sufficient: a chip_top whose block instances are pruned (their outputs
    never reach a primary I/O, or a stub/duplicate wrapper won assembly
    dedup) still elaborates and lands at a handful of cells. yosys
    ``hierarchy`` already errors on a *missing* module def; this backstops
    the *connected-but-collapsed* case. Generous by design -- a real
    multi-block integrated top is thousands of cells. Override with
    ``CORESMITH_CHIP_TOP_MIN_CELLS`` (0 disables).
    """
    try:
        return int(os.environ.get("CORESMITH_CHIP_TOP_MIN_CELLS", "64") or "64")
    except ValueError:
        return 64


def count_logic_depth_from_ltp(ltp_text: str) -> int | None:
    """Longest combinational path length (logic levels) from a Yosys ``ltp``
    dump (``Longest topological path in <m> (length=N)``). Last match wins.
    """
    ms = re.findall(r"length=(\d+)", ltp_text or "")
    if ms:
        return int(ms[-1])
    return None


_CS_MEM_REF_RE = re.compile(r"\bcs_(?:sram|fpmem|rom)\w*\b")


def _dedup_sources(rtl_path: str,
                   extra_sources: list[str] | None) -> list[str]:
    """``[rtl_path] + extra_sources`` deduped by resolved path, order kept.

    A repeated source (the cs_mem library aggregated more than once for a
    multi-memory integration wrapper) makes yosys error on module
    re-definition, killing BOTH STA sub-flows -- strictly worse than the
    blackbox the extra sources exist to prevent. First occurrence wins;
    missing files are dropped.
    """
    out: list[str] = []
    seen: set[str] = set()
    for p in [rtl_path] + list(extra_sources or []):
        if not p or not Path(p).exists():
            continue
        key = str(Path(p).resolve())
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def mem_lib_sources_for_rtl(rtl_text: str) -> list[str]:
    """Extra ``read_verilog`` sources needed to resolve engine memory-wrapper
    instances (``cs_sram*`` / ``cs_fpmem*`` / ``cs_rom*``) in a block's RTL.

    The per-block STA/depth probes synthesize the block RTL FROM SOURCE. A
    block instantiating an engine memory wrapper either fails
    ``hierarchy -check`` outright (the fan-out-aware buffered STA) or silently
    blackboxes the instance (the depth probe) when the wrapper library is not
    in the source set -- so exactly the storage-bearing blocks lose their
    buffered measurement and fall back to the pessimistic unbuffered base WNS,
    while a memory-free sibling gets the base->buffered relaxation.
    Mirrors :func:`probe_synth_cellcount_multi`, which already includes the
    library for the integrated chip_top.
    """
    if not rtl_text or not _CS_MEM_REF_RE.search(rtl_text):
        return []
    # An RTL source that already CARRIES the engine-lib modules -- either
    # inline-defined (library baked into an assembled wrapper) or pulled via
    # `include (yosys resolves the include, possibly to a DIFFERENT copy of
    # the library than wrapper_lib_path(), so path-dedup cannot catch it) --
    # must not pull the library again: yosys errors on module re-definition
    # and BOTH STA sub-flows die, which is worse than the blackbox this
    # helper exists to prevent.
    if re.search(r"\bmodule\s+cs_\w+", rtl_text):
        return []
    if re.search(r'`include\s+"[^"]*cs_(?:sram|fpmem|rom)\w*\.v"', rtl_text):
        return []
    try:
        from orchestrator.langgraph.sram_wrapper import wrapper_lib_path
        lib = wrapper_lib_path()
        return [lib] if Path(lib).exists() else []
    except Exception:  # noqa: BLE001 - best-effort; probes fall back to base
        return []


def probe_logic_depth(
    rtl_path: str,
    top: str,
    *,
    yosys_bin: str = "yosys",
    timeout_s: int = 300,
    extra_sources: list[str] | None = None,
) -> dict[str, Any] | None:
    """PDK-free combinational-DEPTH probe -- the longest register-to-register
    logic path (levels) via ``ltp -noff``.

    This ENFORCES the pipeline scheduler's intent on the actual RTL: a datapath
    collapsed into one combinational cloud (the unsynthesizable failure class)
    has an enormous register-to-register depth, whereas a properly pipelined
    design keeps each stage's depth bounded. Complements the cell-count probe (a
    deep ripple chain can be over-deep without exploding the cell count). Real
    STA covers this when a PDK is present; this proxy runs under SKIP_SYNTH
    where no STA exists.

    Returns ``None`` if yosys/RTL absent; else ``{logic_depth, elaborated,
    reason}`` (``elaborated`` False on timeout / non-zero exit).
    """
    yosys = shutil.which(yosys_bin)
    if not yosys or not rtl_path or not Path(rtl_path).exists():
        return None
    _srcs = " ".join(_dedup_sources(rtl_path, extra_sources))
    script = (
        f"read_verilog -sv {_srcs}\n"
        f"hierarchy -top {top}\n"
        f"proc\nflatten\nopt\ntechmap\nopt_clean\nltp -noff\n"
    )
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".ys", delete=False) as fh:
            fh.write(script)
            ys = fh.name
        result = subprocess.run(
            [yosys, "-s", ys], capture_output=True, text=True, timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return {"logic_depth": None, "elaborated": False,
                "reason": f"did not analyze logic depth within {timeout_s}s"}
    except OSError:
        return None
    finally:
        try:
            Path(ys).unlink()
        except OSError:
            pass
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "")[-300:]
        return {"logic_depth": None, "elaborated": False,
                "reason": f"yosys ltp failed: {tail.strip()}"}
    out = result.stdout + "\n" + result.stderr
    return {"logic_depth": count_logic_depth_from_ltp(out), "elaborated": True,
            "reason": ""}


def logic_depth_gate_enabled() -> bool:
    """Combinational-depth gate (default ON; PDK-free, runs under SKIP_SYNTH).

    Promotes the pipeline scheduler from advisory to enforcing: it checks the
    generated RTL actually registered its datapath instead of collapsing it
    into one combinational cloud. Disable with ``CORESMITH_LOGIC_DEPTH_GATE=0``.
    """
    from orchestrator.profile import ensure_applied, flag_enabled
    ensure_applied()
    return flag_enabled("CORESMITH_LOGIC_DEPTH_GATE", default=True)


def max_logic_depth() -> int:
    """Max register-to-register combinational depth (logic levels).

    Generous by design -- it catches a datapath collapsed into one comb cloud,
    not normal per-stage logic. Override with ``CORESMITH_MAX_LOGIC_DEPTH``.
    """
    try:
        return int(os.environ.get("CORESMITH_MAX_LOGIC_DEPTH", "500") or "500")
    except ValueError:
        return 500


def logic_depth_advisory_with_pdk_enabled() -> bool:
    """``CORESMITH_LOGIC_DEPTH_ADVISORY_WITH_PDK`` (default ON).

    The ltp logic-level proxy cannot DISCRIMINATE a converged staged design
    (881 levels -- carry chains / LCU bits count as levels) from an
    unsynthesizable combinational cloud (887 levels): both land far over the
    generous max. When a real PDK + STA are available the proxy becomes
    ADVISORY -- recorded, never gating, and it never short-circuits the STA --
    so the real pre-layout WNS is the timing authority. A PDK-ABSENT run keeps
    it GATING (it is the only depth signal there). Set 0/false/no/off to keep
    the proxy enforcing even when a PDK is present.
    """
    from orchestrator.profile import ensure_applied, flag_enabled
    ensure_applied()
    return flag_enabled("CORESMITH_LOGIC_DEPTH_ADVISORY_WITH_PDK", default=True)


def sta_tooling_available(synth_result: dict | None) -> bool:
    """True when a real pre-layout STA can run for this block.

    Requires the ``sta`` binary on PATH AND a ``synth_result`` that carries a
    netlist + SDC + liberty which all exist on disk -- exactly
    :func:`run_pre_layout_sta`'s non-``None`` contract, reused so the
    logic-depth advisory check and the STA it defers to agree on "a PDK is
    present". PDK-free / SKIP_SYNTH runs (no ``synth_result``) return False.
    """
    if not shutil.which("sta"):
        return False
    if not synth_result:
        return False
    for key in ("netlist_path", "sdc_path", "liberty_path"):
        p = synth_result.get(key)
        if not p or not Path(p).exists():
            return False
    return True


_SDC_PERIOD_RE = re.compile(
    r"create_clock\b[^\n]*?-period\s+(\d+(?:\.\d+)?)", re.IGNORECASE
)


def parse_sdc_period_ns(sdc_text: str) -> float | None:
    """Pull the (tightest) ``create_clock -period`` value (ns) from an SDC, or
    None. Used to give the timing gate the clock period so a negative-slack
    verdict can report how many stages a path is over and drive re-pipelining."""
    if not sdc_text:
        return None
    vals = [float(m.group(1)) for m in _SDC_PERIOD_RE.finditer(sdc_text)]
    vals = [v for v in vals if v > 0]
    return min(vals) if vals else None


def parse_sta_report(report_text: str) -> dict[str, float | None]:
    """Parse WNS/TNS (ns) from an OpenSTA ``report_wns``/``report_tns`` dump.

    Accepts BOTH the legacy ``wns <N>`` / ``tns <N>`` form and the OpenSTA 3.x
    ``wns max <N>`` / ``tns max <N>`` form. The engine used to expect only the
    legacy form, so on a modern OpenSTA (which prints the ``max`` corner token)
    the parse returned nothing and WNS/TNS silently came back ``None`` -- a
    dependency on the box's ``sta`` sed-wrapper that the engine must not rely on.
    """
    out: dict[str, float | None] = {"wns_ns": None, "tns_ns": None}
    for key, tag in (("wns_ns", "wns"), ("tns_ns", "tns")):
        # ``(?:\s+max)?`` optionally swallows the OpenSTA 3.x corner token so
        # both ``wns -3.42`` and ``wns max -3.42`` parse to the same number.
        m = re.search(
            rf"\b{tag}\b(?:\s+max)?\s+(-?\d+(?:\.\d+)?)",
            report_text or "", re.IGNORECASE,
        )
        if m:
            out[key] = float(m.group(1))
    return out


@dataclass
class PpaVerdict:
    """Outcome of the deterministic PPA gate."""

    ok: bool
    checks: list[dict[str, Any]] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    # Dimensions the gate could NOT judge (missing budget or missing
    # measurement) -- recorded per metric so a strict profile can decide
    # whether an unmeasurable PPA dimension is acceptable, instead of the
    # gate silently treating "cannot judge" as a pass. ``ok`` semantics are
    # unchanged: a missing budget still does not block here.
    unmeasured: list[dict[str, Any]] = field(default_factory=list)


def evaluate_ppa(
    *,
    actual_ff: int | None,
    ff_budget: int | None,
    ff_tolerance_pct: float = 25.0,
    ff_floor: int = 2000,
    hard_ff_ceiling: int = 50000,
    storage_ff: int | None = None,
    actual_area_um2: float | None = None,
    area_budget_um2: float | None = None,
    area_tolerance_pct: float = 20.0,
    wns_ns: float | None = None,
    wns_margin_ns: float = 0.0,
    period_ns: float | None = None,
    sta_error: str | None = None,
    budget_overridden: bool = False,
) -> PpaVerdict:
    """Compare synthesized metrics against budgets. Only flags what it can.

    ``budget_overridden``: the chip-lead explicitly accepted this block's
    storage/area cost at the uarch feasibility review (the same override the
    mem_price gate honors). The BUDGET dimensions -- area and the logic-FF
    budget -- are then recorded but DEFERRED (``deferred_by_override`` on the
    check, no ``ok`` flip), instead of re-failing a block whose cost the lead
    already signed off. The absolute FF hard ceiling and the timing dimension
    still gate: an accepted [area] blocker never waives routability or timing.

    A check is evaluated only when both its actual value and its budget are
    present; a missing budget means "cannot judge" -> that dimension passes
    (the gate never blocks on a number it doesn't have).

    ``ff_floor`` is an absolute guard: the FF check never fails below it,
    because the failure class the gate exists for -- a should-be-SRAM memory
    that became flops -- only manifests at scale (sky130's SRAM threshold is
    ~256 words, i.e. ≳2000 FF). Below the floor an over-budget block is just
    over-built logic (a code nit, negligible area), not a PPA disaster, and
    must not stall the pipeline.

    ``storage_ff`` (declared buffers / inferred memories kept as flops) is
    SEPARATED from LOGIC flops for the budget comparison: ``flip_flop_budget``
    is a LOGIC budget (bulk buffers/FIFOs are priced by the area budget, not
    the FF budget), so a legitimate line buffer / small memory must not be
    charged against it. The budget check judges ``logic_ff = actual_ff -
    storage_ff``; the absolute hard ceiling still sees TOTAL flops so a
    should-be-SRAM flop explosion is caught regardless of how it is classified.
    This is the false-flag (a legit 144-byte buffer) that got the gate disabled.
    """
    checks: list[dict[str, Any]] = []
    reasons: list[str] = []
    unmeasured: list[dict[str, Any]] = []
    ok = True

    # LOGIC vs STORAGE flip-flops. The budget is a logic-FF budget; subtract
    # declared storage so a legitimate buffer/memory is never judged against it.
    logic_ff: int | None = actual_ff
    if actual_ff is not None and storage_ff:
        logic_ff = max(0, actual_ff - int(storage_ff))

    # Flip-flop count is the gate's core dimension: record it as unmeasured
    # whenever it cannot be fully judged (no budget or no measured count).
    if not (ff_budget and actual_ff is not None):
        unmeasured.append({
            "metric": "flip_flop_count",
            "have_budget": bool(ff_budget),
            "have_actual": actual_ff is not None,
            "reason": (
                "no flip_flop_budget" if not ff_budget
                else "no measured flip_flop_count"
            ),
        })

    if ff_budget and actual_ff is not None:
        limit = ff_budget * (1.0 + ff_tolerance_pct / 100.0)
        # Judge LOGIC flops (storage separated out) against the logic budget.
        passed = logic_ff <= limit or logic_ff <= ff_floor
        checks.append({
            "metric": "flip_flop_count", "actual": logic_ff,
            "total_ff": actual_ff, "storage_ff": int(storage_ff or 0),
            "budget": ff_budget, "limit": round(limit),
            "floor": ff_floor, "passed": passed,
        })
        if not passed:
            if budget_overridden:
                # Chip-lead accepted the storage cost at feasibility review:
                # record the miss, defer the verdict (mirrors the mem_price gate).
                checks[-1]["deferred_by_override"] = True
            else:
                ok = False
                _storage_note = (
                    f" ({actual_ff:,} total - {int(storage_ff):,} storage)"
                    if storage_ff else ""
                )
                reasons.append(
                    f"LOGIC flip-flop count {logic_ff:,}{_storage_note} exceeds budget "
                    f"{ff_budget:,} by >{ff_tolerance_pct:.0f}% (limit {round(limit):,}) "
                    f"and the {ff_floor:,}-FF floor -- a storage array that should be an "
                    f"SRAM macro synthesized to flops"
                )

    # Absolute hard ceiling -- fires even with NO flip_flop_budget, and is the
    # backstop that must stop a block from advancing to chip-top synthesis. A
    # single block over this many flops is patently unroutable in sky130 and
    # almost always means a should-be-SRAM memory flopped because the RTL wrote
    # a behavioral reg-array instead of instantiating a macro (codec mb_emitter:
    # 136k FF). It does NOT override a deliberately-large budget: if the block is
    # within its (possibly huge) flip_flop_budget, the uArch owns that choice.
    _over_budget = ff_budget is None or actual_ff is None or (
        actual_ff > ff_budget * (1.0 + ff_tolerance_pct / 100.0)
    )
    if actual_ff is not None and actual_ff > hard_ff_ceiling and _over_budget:
        already = any(
            c["metric"] == "flip_flop_count" and not c["passed"]
            and not c.get("deferred_by_override") for c in checks
        )
        checks.append({
            "metric": "flip_flop_hard_ceiling", "actual": actual_ff,
            "ceiling": hard_ff_ceiling, "passed": False,
        })
        ok = False
        if not already:
            reasons.append(
                f"flip-flop count {actual_ff:,} exceeds the absolute "
                f"{hard_ff_ceiling:,}-FF hard ceiling for one block -- a memory "
                f"that should be an SRAM macro almost certainly flopped (behavioral "
                f"reg-array, no macro instance). The block must NOT advance to "
                f"chip-top synthesis; instantiate the named macro instead."
            )

    # Area is optional; only flag it as unmeasured when one of the pair was
    # supplied but not the other (a budget with no number, or a number with no
    # budget) -- when neither is present, area simply is not in scope.
    if (area_budget_um2 is not None or actual_area_um2 is not None) and not (
        area_budget_um2 and actual_area_um2 is not None
    ):
        unmeasured.append({
            "metric": "chip_area_um2",
            "have_budget": area_budget_um2 is not None,
            "have_actual": actual_area_um2 is not None,
            "reason": (
                "no area_budget_um2" if area_budget_um2 is None
                else "no measured chip_area_um2"
            ),
        })

    if area_budget_um2 and actual_area_um2 is not None:
        limit = area_budget_um2 * (1.0 + area_tolerance_pct / 100.0)
        passed = actual_area_um2 <= limit
        checks.append({
            "metric": "chip_area_um2", "actual": actual_area_um2,
            "budget": area_budget_um2, "limit": round(limit, 1), "passed": passed,
        })
        if not passed:
            if budget_overridden:
                checks[-1]["deferred_by_override"] = True
            else:
                ok = False
                reasons.append(
                    f"area {actual_area_um2:,.0f} µm² exceeds budget "
                    f"{area_budget_um2:,.0f} µm² by >{area_tolerance_pct:.0f}%"
                )

    if wns_ns is not None:
        # Grader alignment: the accelerator-chassis grader checks only cells/area/FF +
        # functional correctness and DEFERS timing to signoff (the golden
        # reference itself uses a combinational flop-array memory read that
        # would miss 50 MHz pre-layout STA). CORESMITH_PPA_TIMING_ADVISORY=1
        # makes the pre-layout WNS sub-check ADVISORY (record, do NOT block);
        # FF/area gates remain fully enforced.
        _timing_advisory = os.environ.get(
            "CORESMITH_PPA_TIMING_ADVISORY", "0").strip().lower() in (
            "1", "true", "yes", "on")
        violated = wns_ns < -abs(wns_margin_ns)
        over_ns = -wns_ns  # how many ns the reg-to-reg path is over the period
        # A GROSS violation is NEVER silently ignorable, even in advisory mode:
        # advisory only downgrades MARGINAL slack (signoff can recover a small
        # miss). A path that is over the period by more than a hard fraction
        # (or an absolute floor when the period is unknown) is a REAL,
        # re-pipeline-now defect -- exactly the AES round (~27 ns vs a 20 ns
        # clock, WNS ~= -7.31 ns) that must add a within-round stage.
        _hard_frac = _timing_hard_frac()
        _hard_floor = _timing_hard_floor_ns()
        if period_ns and period_ns > 0:
            gross = violated and over_ns > _hard_frac * period_ns
        else:
            gross = violated and over_ns > _hard_floor
        passed = (not violated) or (_timing_advisory and not gross)
        checks.append({
            "metric": "wns_ns", "actual": wns_ns,
            "limit": -abs(wns_margin_ns), "passed": passed,
            "period_ns": period_ns, "gross": bool(gross),
            "advisory": bool(_timing_advisory and violated and not gross),
        })
        if not passed:
            ok = False
            reasons.append(_wns_repipeline_reason(wns_ns, over_ns, period_ns, gross))
    elif sta_error:
        # FAIL-CLOSED (Section 3a): STA was ATTEMPTED (the block has a netlist)
        # but produced NO parseable WNS/TNS -- an unreadable netlist, a link
        # failure, a crash, a timeout. This is a MEASUREMENT FAILURE, not a
        # PDK-absent skip (that returns wns_ns=None with no sta_error and stays
        # legitimately unmeasured). The old gate skipped the timing check on a
        # None WNS and declared "within budget" -> a benchmark-integrity hole
        # (matmul_mac passed with zero timing evidence). Refuse to pass an
        # unmeasured timing dimension. CORESMITH_PPA_TIMING_ADVISORY=1 is the
        # deliberate operator opt-out (the whole timing dimension is advisory);
        # otherwise this is a gate FAILURE so the block retries / escalates.
        _timing_advisory = os.environ.get(
            "CORESMITH_PPA_TIMING_ADVISORY", "0").strip().lower() in (
            "1", "true", "yes", "on")
        checks.append({
            "metric": "wns_ns", "actual": None, "passed": bool(_timing_advisory),
            "sta_error": sta_error, "unmeasured_timing": True,
            "advisory": bool(_timing_advisory),
        })
        if not _timing_advisory:
            ok = False
            reasons.append(
                f"pre-layout STA produced no parseable timing ({sta_error}) -- "
                f"refusing to declare the block within budget on an UNMEASURED "
                f"timing dimension (fail-closed). Fix the netlist / re-run STA "
                f"(retry), or set CORESMITH_PPA_TIMING_ADVISORY=1 to accept "
                f"unmeasurable timing for this run."
            )

    return PpaVerdict(ok=ok, checks=checks, reasons=reasons, unmeasured=unmeasured)


def _timing_hard_frac() -> float:
    """Fraction of the clock period beyond which a WNS miss is a GROSS (never
    silently ignorable) violation. Override ``CORESMITH_PPA_TIMING_HARD_FRAC``."""
    try:
        return float(os.environ.get("CORESMITH_PPA_TIMING_HARD_FRAC", "0.25")
                     or "0.25")
    except ValueError:
        return 0.25


def _timing_hard_floor_ns() -> float:
    """Absolute ns over-period beyond which a WNS miss is GROSS when the period
    is unknown. Override ``CORESMITH_PPA_TIMING_HARD_NS``."""
    try:
        return float(os.environ.get("CORESMITH_PPA_TIMING_HARD_NS", "2.0")
                     or "2.0")
    except ValueError:
        return 2.0


def _wns_repipeline_reason(wns_ns: float, over_ns: float,
                           period_ns: float | None, gross: bool) -> str:
    """Actionable re-pipeline feedback for a negative-slack verdict.

    Surfaces WHICH register-to-register path is over and BY HOW MUCH, and the
    number of stages the datapath must be split into so the scheduler / uArch
    adds a within-path register -- not a bare "timing violated"."""
    if period_ns and period_ns > 0:
        path_ns = period_ns + over_ns
        stages_needed = max(2, math.ceil(path_ns / period_ns))
        return (
            f"worst negative slack {wns_ns:.2f} ns against a {period_ns:.2f} ns "
            f"period{' (GROSS -- not signoff-recoverable)' if gross else ''}: the "
            f"worst register-to-register datapath is ~{path_ns:.2f} ns, "
            f"~{100.0 * over_ns / period_ns:.0f}% over the period. RE-PIPELINE it: "
            f"insert {stages_needed - 1} register stage(s) (split into ~"
            f"{stages_needed} stages) so each stage's chained delay <= "
            f"{period_ns:.2f} ns. An op OUTSIDE the arithmetic vocabulary -- a "
            f"crypto S-box / table LUT / Galois-field multiply whose single-"
            f"instance delay exceeds the period -- is the usual cause; pipeline "
            f"WITHIN that op (register inside it), not just at its boundary."
        )
    return (
        f"worst negative slack {wns_ns:.2f} ns -- timing violated by {over_ns:.2f} "
        f"ns over the period. RE-PIPELINE the over-period register-to-register "
        f"path (add a stage; if a single op -- S-box/LUT/GF-multiply -- exceeds "
        f"the period, register WITHIN it). Often an N:1 combinational read mux on "
        f"a flop-array memory."
    )


# Characters that end a Verilog (module-type) identifier just before an
# instance parameter-override ``#( ... )``.
_IDENT_CHARS = set(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_$"
)
_STA_STRIP_INTERESTING = re.compile(r'["/#]')


def _sta_skip_ws(s: str, i: int) -> int:
    n = len(s)
    while i < n and s[i] in " \t\r\n":
        i += 1
    return i


def _sta_consume_ident(s: str, i: int) -> int:
    """Return the index just past the identifier starting at ``i`` (or ``i``)."""
    n = len(s)
    if i < n and s[i] == "\\":
        # Verilog escaped identifier: runs until the next whitespace.
        j = i + 1
        while j < n and s[j] not in " \t\r\n":
            j += 1
        return j
    j = i
    while j < n and s[j] in _IDENT_CHARS:
        j += 1
    return j


def _sta_match_paren(s: str, i: int) -> int:
    """``s[i]`` is ``(``. Return the index just past the matching ``)`` (or -1).

    Strings and ``//`` / ``/* */`` comments inside the parens are skipped so a
    stray paren in a literal (``.INIT_FILE("a)b(")``) never unbalances it.
    """
    n = len(s)
    depth = 0
    while i < n:
        c = s[i]
        if c == '"':
            i += 1
            while i < n:
                if s[i] == "\\":
                    i += 2
                    continue
                if s[i] == '"':
                    i += 1
                    break
                i += 1
            continue
        if c == "/" and i + 1 < n and s[i + 1] == "/":
            while i < n and s[i] != "\n":
                i += 1
            continue
        if c == "/" and i + 1 < n and s[i + 1] == "*":
            i += 2
            while i + 1 < n and not (s[i] == "*" and s[i + 1] == "/"):
                i += 1
            i += 2
            continue
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return -1


def strip_instance_parameters(verilog: str) -> str:
    """Drop Verilog *instance* parameter-override lists from a netlist.

    A synthesized gate-level netlist that instantiates a parameterized
    blackbox macro emits ``cs_sram_1rw #(.DEPTH(256), ...) u_mem (...)``.
    OpenSTA's ``read_verilog`` cannot parse a parameterized instantiation, so
    it fails on exactly the SRAM-bearing blocks where timing matters most. The
    macro's parameters are irrelevant to STA (it is an opaque blackbox), so the
    ``#( ... )`` between the module type and the instance name can be removed,
    leaving an STA-readable ``cs_sram_1rw u_mem (...)``.

    Only *instance overrides* are removed: the ``#( ... )`` must sit between an
    identifier (the module type) and an instance name that is itself followed
    by ``(`` (the port list). Module parameter *declarations*
    (``module m #( ... ) ( ... )`` -- ``)`` followed by ``(``) and delay
    controls (``#5`` / ``assign #(2,3) y = ...`` -- not followed by ``ident (``)
    are left untouched. Strings and comments are skipped so a ``#`` or paren
    inside them is never misparsed. Idempotent.
    """
    if "#" not in verilog:
        return verilog
    s = verilog
    n = len(s)
    out: list[str] = []
    i = 0
    while i < n:
        m = _STA_STRIP_INTERESTING.search(s, i)
        if m is None:
            out.append(s[i:])
            break
        j = m.start()
        out.append(s[i:j])
        i = j
        c = s[i]
        if c == '"':
            k = i + 1
            while k < n:
                if s[k] == "\\":
                    k += 2
                    continue
                if s[k] == '"':
                    k += 1
                    break
                k += 1
            out.append(s[i:k])
            i = k
            continue
        if c == "/" and i + 1 < n and s[i + 1] == "/":
            k = i
            while k < n and s[k] != "\n":
                k += 1
            out.append(s[i:k])
            i = k
            continue
        if c == "/" and i + 1 < n and s[i + 1] == "*":
            k = i + 2
            while k + 1 < n and not (s[k] == "*" and s[k + 1] == "/"):
                k += 1
            k = min(k + 2, n)
            out.append(s[i:k])
            i = k
            continue
        # c == '#': test for an instance parameter-override.
        p = i - 1
        while p >= 0 and s[p] in " \t\r\n":
            p -= 1
        prev_is_ident = p >= 0 and s[p] in _IDENT_CHARS
        q = _sta_skip_ws(s, i + 1)
        stripped = False
        if prev_is_ident and q < n and s[q] == "(":
            close = _sta_match_paren(s, q)
            if close != -1:
                r = _sta_skip_ws(s, close)
                ident_end = _sta_consume_ident(s, r)
                if ident_end > r:
                    t = _sta_skip_ws(s, ident_end)
                    if t < n and s[t] == "(":
                        # Instance override: drop s[i:close] (the '#( ... )').
                        i = close
                        stripped = True
        if not stripped:
            out.append("#")
            i += 1
    return "".join(out)


def strip_signed_declaration_qualifiers(verilog: str) -> str:
    """Remove ``signed`` qualifiers from structural net declarations.

    OpenSTA 3.1 rejects the ``wire signed [N:0] name;`` declarations that
    Yosys can emit in a fully technology-mapped netlist.  Signedness has no
    remaining functional meaning once arithmetic has been mapped to cells,
    so the temporary STA-only copy may safely omit those qualifiers.

    This scanner removes only a standalone ``signed`` token in a statement
    containing a ``wire``, ``reg``, ``input``, ``output``, or ``inout``
    declaration.  Comments, strings, escaped identifiers, ``$signed`` calls,
    and identifiers merely containing the word are preserved verbatim.
    Source RTL is never passed through this transformation.
    """
    if "signed" not in verilog:
        return verilog

    declaration_keywords = {"wire", "reg", "input", "output", "inout"}
    n = len(verilog)
    out: list[str] = []
    i = 0
    in_declaration = False
    while i < n:
        c = verilog[i]

        if c == '"':
            k = i + 1
            while k < n:
                if verilog[k] == "\\":
                    k += 2
                    continue
                if verilog[k] == '"':
                    k += 1
                    break
                k += 1
            out.append(verilog[i:k])
            i = k
            continue
        if c == "/" and i + 1 < n and verilog[i + 1] == "/":
            k = i + 2
            while k < n and verilog[k] != "\n":
                k += 1
            out.append(verilog[i:k])
            i = k
            continue
        if c == "/" and i + 1 < n and verilog[i + 1] == "*":
            k = i + 2
            while k + 1 < n and not (
                verilog[k] == "*" and verilog[k + 1] == "/"
            ):
                k += 1
            k = min(k + 2, n)
            out.append(verilog[i:k])
            i = k
            continue
        if c == "\\":
            # An escaped identifier runs through the next whitespace.
            k = i + 1
            while k < n and verilog[k] not in " \t\r\n":
                k += 1
            out.append(verilog[i:k])
            i = k
            continue
        if c == ";":
            in_declaration = False
            out.append(c)
            i += 1
            continue
        if c.isalpha() or c in "_$":
            k = i + 1
            while k < n and (
                verilog[k].isalnum() or verilog[k] in "_$"
            ):
                k += 1
            token = verilog[i:k]
            if token in declaration_keywords:
                in_declaration = True
            if token == "signed" and in_declaration:
                i = k
                continue
            out.append(token)
            i = k
            continue

        out.append(c)
        i += 1

    return "".join(out)


def run_pre_layout_sta(
    netlist_path: str,
    sdc_path: str,
    liberty_path: str,
    top_module: str,
    *,
    timeout_s: int = 300,
) -> dict[str, float | None] | None:
    """Pre-layout STA on the mapped netlist via OpenSTA.

    Ideal-clock, no wire RC -> optimistic, but it catches gross violations
    (a hundreds-of-ns combinational read-mux path shows huge negative slack
    even pre-layout).

    Returns:
      * ``{wns_ns, tns_ns}`` on a successful timing read;
      * ``None`` only when STA cannot even be *attempted* -- the ``sta`` binary
        is not installed or a required input (netlist / sdc / liberty) is
        missing (a global tooling/inputs gap, not a per-block measurement);
      * ``{wns_ns: None, tns_ns: None, sta_error: <reason>}`` (plus a WARNING
        log) when STA WAS attempted for a block that has a netlist but produced
        no parseable timing (unreadable netlist, link failure, crash, timeout).
        This makes measurement absence LOUD instead of a silent ``None`` -- the
        same ruler-blindness class as the frontend WNS gap: the SRAM-bearing
        blocks are precisely the ones whose netlist STA used to reject.

    The mapped netlist is first passed through :func:`strip_instance_parameters`
    and :func:`strip_signed_declaration_qualifiers` so parameterized blackbox
    instances and signed structural declarations -- both rejected by the
    supported OpenSTA parser -- become STA-readable.
    """
    sta_bin = shutil.which("sta")
    if not sta_bin:
        return None
    for p in (netlist_path, sdc_path, liberty_path):
        if not p or not Path(p).exists():
            return None

    def _fail(reason: str) -> dict[str, float | None]:
        logger.warning(
            "pre-layout STA produced no timing for %s: %s", top_module, reason
        )
        return {"wns_ns": None, "tns_ns": None, "sta_error": reason}

    try:
        raw_netlist = Path(netlist_path).read_text()
    except OSError as e:
        return _fail(f"cannot read netlist {netlist_path}: {e}")
    sta_netlist = strip_signed_declaration_qualifiers(
        strip_instance_parameters(raw_netlist)
    )

    tcl = None
    sta_nl = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", suffix="_sta.v", delete=False
        ) as nf:
            nf.write(sta_netlist)
            sta_nl = nf.name
        script = (
            f"read_liberty {liberty_path}\n"
            f"read_verilog {sta_nl}\n"
            f"link_design {top_module}\n"
            f"read_sdc {sdc_path}\n"
            f"report_wns\n"
            f"report_tns\n"
            f"exit\n"
        )
        with tempfile.NamedTemporaryFile("w", suffix=".tcl", delete=False) as fh:
            fh.write(script)
            tcl = fh.name
        result = subprocess.run(
            [sta_bin, "-no_init", "-exit", tcl],
            capture_output=True, text=True, timeout=timeout_s,
        )
        parsed = parse_sta_report(result.stdout)
        if parsed["wns_ns"] is None and parsed["tns_ns"] is None:
            tail = ((result.stderr or "") + (result.stdout or "")).strip()[-400:]
            reason = f"OpenSTA emitted no parseable WNS/TNS (rc={result.returncode})"
            if tail:
                reason += f"; output tail: {tail}"
            return _fail(reason)
        return parsed
    except subprocess.TimeoutExpired:
        return _fail(f"OpenSTA timed out after {timeout_s}s")
    except OSError as e:
        return _fail(f"OpenSTA invocation failed: {e}")
    finally:
        for _p in (tcl, sta_nl):
            if _p:
                try:
                    Path(_p).unlink()
                except OSError:
                    pass


# --------------------------------------------------------------------------- #
# Fan-out-aware (max(base, buffered)) pre-layout STA  --  engine-v31, step 1
# --------------------------------------------------------------------------- #
# WHY: :func:`run_pre_layout_sta` above measures ONE unbuffered mapped netlist.
# A Yosys-mapped netlist has NO fan-out buffering, so OpenSTA sees a single
# min-size gate driving the whole lumped pin capacitance of a high-fan-out net
# and extrapolates its delay/slew far past the cell's characterized load range
# -- tens of ns of PURE fan-out delay that a real Sky130 backend's
# set_max_fanout + repair_design pass removes. That systematically FALSE-FAILS
# high-fan-out-but-signoff-recoverable designs: the AES-v3 one-round-per-clock
# engine measured WNS -17.75 ns unbuffered here yet closes at +14.97 ns after
# standard max-fan-out buffering (and the graded golden AES closes the same
# way). This function ports the accelerator-chassis grader's fan-out-aware recipe: it
# synthesizes the RTL to Sky130 cells TWICE from source -- an unbuffered BASE
# map and a BUFFERED map (ABC ``buffer -N`` + gate sizing, the std-cell
# analogue of set_max_fanout + repair_design) -- and reports max(base, buffered)
# WNS. Fan-out buffering only ever RELAXES a path, so taking the best is
# monotonic: no design that met timing unbuffered can be false-failed, and a
# high-fan-out design gets the buffered Fmax it would actually achieve on
# silicon. Both sub-flows are placement-free and RNG-free -> deterministic
# (same input -> same WNS bit-for-bit); a buffering error falls back to BASE.
_STA_MAX_FANOUT = 16               # per-net fan-out cap for the buffered pass
_STA_DONT_USE = ("lpflow", "probe")  # standard dont_use: iso/probe cells skew timing
_STA_CELL_RE = re.compile(r'cell\s*\(\s*"([^"]+)"\s*\)\s*\{')


def _sta_dontuse_liberty(src_lib: str) -> str:
    """Return a Liberty with the lpflow/probe (dont_use) cells stripped.

    Cached in $TMPDIR; regenerated only when missing or older than the source.
    Falls back to the full library on any I/O error (never blocks measurement).
    """
    try:
        src = Path(src_lib)
        cache = Path(tempfile.gettempdir()) / "coresmith_sta_dontuse_sky130_hd.lib"
        if cache.exists() and cache.stat().st_mtime >= src.stat().st_mtime:
            return str(cache)
        txt = src.read_text()
        n = len(txt)
        out: list[str] = []
        pos = 0
        while True:
            m = _STA_CELL_RE.search(txt, pos)
            if not m:
                out.append(txt[pos:])
                break
            name = m.group(1)
            depth = 0
            j = m.end() - 1
            while j < n:
                c = txt[j]
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            end = j + 1
            if any(p in name for p in _STA_DONT_USE):
                out.append(txt[pos:m.start()])          # drop this cell block
            else:
                out.append(txt[pos:end])
            pos = end
        cache.write_text("".join(out))
        return str(cache)
    except OSError:
        return src_lib


def _maxfanout_synth_script(sources: list[str], lib: str, netlist: Path,
                            top: str, blackbox_mem: bool, buffered: bool) -> str:
    reads = " ".join(sources)
    if blackbox_mem:
        # Lift inferred memories into a blackboxed submodule (SRAM-macro model)
        # BEFORE tech mapping, then finish synthesis on the surviving logic, so
        # a byte buffer doesn't flatten into a thousand-way async read mux whose
        # unbuffered net delay is a pure synth artifact.
        mem = (
            f"synth -top {top} -flatten -run :fine\n"
            f"select -set mc t:$mem_v2\n"
            f"submod -name mem_macro @mc\n"
            f"blackbox mem_macro\n"
            f"synth -top {top} -run fine:\n"
        )
    else:
        mem = f"synth -top {top} -flatten\n"
    if buffered:
        # After logic mapping, ABC's `buffer -N` inserts a buffer tree wherever a
        # net exceeds the fan-out cap, then upsize/dnsize gate-sizes the drivers
        # -- the std-cell analogue of a Sky130 set_max_fanout + repair_design.
        abc = (f'abc -liberty {lib} -script '
               f'"+strash;dch,-f;map,-B,0.9;topo;stime,-c;'
               f'buffer,-N,{_STA_MAX_FANOUT};upsize,-c;dnsize,-c;stime,-p"\n')
    else:
        abc = f"abc -liberty {lib}\n"
    return (
        f"read_verilog -sv {reads}\n"
        f"hierarchy -check -top {top}\n"
        f"{mem}"
        f"dfflibmap -liberty {lib}\n"
        f"{abc}"
        f"setundef -zero\n"
        f"opt_clean -purge\n"
        f"write_verilog -noattr {netlist}\n"
    )


def _measure_wns_from_rtl(sources: list[str], lib: str, base_wd: Path, tag: str,
                          buffered: bool, period_ns: float, top: str,
                          clk_port: str, yosys_bin: str, sta_bin: str,
                          timeout_s: int) -> tuple[float | None, str]:
    """Synth (blackbox-mem, optionally fan-out-buffered) + OpenSTA at period.

    Returns ``(wns_ns, "")`` on success or ``(None, detail)`` on any error so a
    caller can fall back to another measurement instead of crashing the gate.
    """
    wd = base_wd / tag
    wd.mkdir(parents=True, exist_ok=True)
    netlist = wd / "netlist.v"
    yp = None
    # Blackbox-mem synth; fall back to a plain flatten for a design with no
    # inferred memory (then the blackbox submodule step has nothing to lift).
    for blackbox in (True, False):
        ys = wd / "syn.ys"
        ys.write_text(_maxfanout_synth_script(sources, lib, netlist, top,
                                              blackbox, buffered))
        try:
            yp = subprocess.run([yosys_bin, "-q", str(ys)],
                                capture_output=True, text=True, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return None, f"yosys timed out after {timeout_s}s"
        except OSError as e:
            return None, f"yosys invocation failed: {e}"
        if yp.returncode == 0 and netlist.exists():
            break
        if netlist.exists():
            netlist.unlink()
    if yp is None or not netlist.exists():
        tail = ((yp.stdout + yp.stderr)[-400:]) if yp else "yosys not run"
        return None, "yosys_fail: " + tail
    # OpenSTA's Verilog reader rejects the `signed` qualifier Yosys may emit on
    # surviving vector wires; it is meaningless for gate-level STA.
    text = netlist.read_text()
    if " signed " in text:
        netlist.write_text(re.sub(r"\bsigned\b", "", text))
    tcl = wd / "sta.tcl"
    tcl.write_text(
        f"read_liberty {lib}\n"
        f"read_verilog {netlist}\n"
        f"link_design {top}\n"
        f"create_clock -name clk -period {period_ns} [get_ports {clk_port}]\n"
        f'puts "CORESMITH_WNS [worst_slack -max]"\n'
    )
    try:
        sp = subprocess.run([sta_bin, "-no_splash", "-exit", str(tcl)],
                            capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return None, f"OpenSTA timed out after {timeout_s}s"
    except OSError as e:
        return None, f"OpenSTA invocation failed: {e}"
    out = sp.stdout + sp.stderr
    m = re.search(r"CORESMITH_WNS\s+([-0-9.eE+]+)", out)
    if m is None:
        m = re.search(r"worst slack\s*(?:-?max)?\s*([-0-9.eE+]+)", out, re.IGNORECASE)
    if m is None:
        return None, "sta_parse_fail: " + out[-400:]
    return float(m.group(1)), ""


def run_maxfanout_buffered_sta(
    rtl_path: str,
    liberty_path: str,
    top_module: str,
    period_ns: float,
    clk_port: str = "clk",
    *,
    timeout_s: int = 300,
    extra_sources: list[str] | None = None,
) -> dict[str, float | None] | None:
    """Fan-out-aware pre-layout STA: max(unbuffered BASE, fan-out-BUFFERED) WNS.

    Synthesizes ``rtl_path`` to Sky130 cells from source TWICE at ``period_ns``
    -- an unbuffered BASE map and an ABC max-fan-out-buffered + gate-sized map --
    and reports the BEST (max-slack) reg-to-reg WNS. See the module-level comment
    above for the physics. Deterministic; a buffering error falls back to BASE.

    Returns:
      * ``{base_wns_ns, buffered_wns_ns, wns_ns (=max), fmax_mhz, sta_ok,
        detail}`` when at least one measurement succeeded;
      * ``{... wns_ns: None, sta_error: <reason>}`` when STA was attempted but
        BOTH sub-flows produced no parseable timing (loud, not silent);
      * ``None`` only when the tool/inputs are absent (yosys or sta missing, or
        an unreadable RTL/liberty) -- a global tooling gap, never a per-block
        verdict, so a caller can simply fall back to the base measurement.
    """
    yosys_bin = shutil.which("yosys")
    sta_bin = shutil.which("sta")
    if not yosys_bin or not sta_bin:
        return None
    if (not rtl_path or not Path(rtl_path).exists()
            or not liberty_path or not Path(liberty_path).exists()):
        return None
    if not period_ns or period_ns <= 0:
        return None

    lib = _sta_dontuse_liberty(liberty_path)
    # Engine memory-wrapper instances (cs_sram/cs_fpmem/cs_rom) must resolve or
    # `hierarchy -check` fails BOTH sub-flows and the caller falls back to the
    # pessimistic unbuffered mapped-netlist base -- denying exactly the
    # storage-bearing blocks the buffered relaxation. Callers pass the wrapper
    # library via ``extra_sources`` (see :func:`mem_lib_sources_for_rtl`);
    # sources are deduped by resolved path so a repeated library entry cannot
    # trip yosys module re-definition.
    _srcs = _dedup_sources(rtl_path, extra_sources)
    wd = Path(tempfile.mkdtemp(prefix="coresmith_mfsta_"))
    try:
        base_wns, base_detail = _measure_wns_from_rtl(
            _srcs, lib, wd, "base", False, period_ns, top_module,
            clk_port, yosys_bin, sta_bin, timeout_s)
        # BUFFERED is the fan-out-aware relaxation; if it errors we keep BASE.
        buf_wns, buf_detail = _measure_wns_from_rtl(
            _srcs, lib, wd, "buf", True, period_ns, top_module,
            clk_port, yosys_bin, sta_bin, timeout_s)
    finally:
        try:
            shutil.rmtree(wd, ignore_errors=True)
        except OSError:
            pass

    cands = [w for w in (base_wns, buf_wns) if w is not None]
    if not cands:
        reason = f"maxfanout STA produced no timing: base[{base_detail}] buf[{buf_detail}]"
        logger.warning("fan-out-aware STA failed for %s: %s", top_module, reason)
        return {"base_wns_ns": base_wns, "buffered_wns_ns": buf_wns,
                "wns_ns": None, "fmax_mhz": None, "sta_ok": False,
                "sta_error": reason, "detail": reason}
    wns = max(cands)                      # best -- fan-out buffering is monotonic
    denom = period_ns - wns
    fmax = round(1000.0 / denom, 3) if denom > 0 else None
    return {
        "base_wns_ns": round(base_wns, 4) if base_wns is not None else None,
        "buffered_wns_ns": round(buf_wns, 4) if buf_wns is not None else None,
        "wns_ns": round(wns, 4), "fmax_mhz": fmax, "sta_ok": True, "detail": "",
    }


def sta_maxfanout_enabled() -> bool:
    """Fan-out-aware max(base,buffered) STA in the block PPA gate (default ON).

    ``CORESMITH_STA_MAXFANOUT=0`` restores the unbuffered-only pre-layout STA.
    """
    return (os.environ.get("CORESMITH_STA_MAXFANOUT", "1") or "1").strip() != "0"


def ppa_gate_enabled() -> bool:
    """True when CORESMITH_PPA_GATE is set truthy (strict profile seeds it)."""
    from orchestrator.profile import ensure_applied, flag_enabled
    ensure_applied()
    return flag_enabled("CORESMITH_PPA_GATE", default=False)


def ppa_honor_feas_override_enabled() -> bool:
    """Post-synth PPA gate honors a chip-lead ``uarch_feasibility_override``
    on the BUDGET dimensions (area + logic-FF), mirroring the mem_price gate
    (default ON). The absolute FF hard ceiling and the timing dimension still
    gate -- an accepted [area] blocker never waives routability or timing.
    ``CORESMITH_PPA_HONOR_FEAS_OVERRIDE=0`` restores budget-strict behavior.
    """
    return (os.environ.get("CORESMITH_PPA_HONOR_FEAS_OVERRIDE", "1")
            or "1").strip() != "0"


# --- Part D: post-synth memory-as-flops probe --------------------------------
#
# The memory-preserving probe (:func:`probe_synth_generic`) reads only the block
# RTL, so an engine memory WRAPPER (cs_sram_*) is an unresolved blackbox there
# (0 mem bits) and a wrapped-memory-that-flops is invisible to it. This probe
# closes that hole: it reads the design PLUS the cs_sram wrapper library and
# applies the Part-B ``chparam ... MEM_IMPL "MACRO"`` directive, so a properly
# wrapped memory becomes a `cs_mem_macro_shell` (contributing NOTHING) while a
# raw `reg [] mem []` array -- or a cs_sram whose geometry could not bind --
# stays an inferred `$mem`. It then reports each residual memory's WIDTH x DEPTH
# from the yosys JSON so :func:`sram_wrapper.gate_memory_as_flops` can flag only
# ABOVE-threshold flop memories (never cs_fpmem, never sub-threshold arrays).


def _parse_json_memories(json_text: str) -> list[tuple[int, int]]:
    """[(width, depth), ...] for every ``$mem*`` cell and inferred ``memories``
    entry in a yosys ``write_json`` dump (params are binary strings)."""
    import json as _json

    def _as_int(v) -> int:
        if isinstance(v, int):
            return v
        s = str(v)
        try:
            return int(s, 2) if set(s) <= {"0", "1"} else int(s)
        except ValueError:
            return 0

    out: list[tuple[int, int]] = []
    try:
        doc = _json.loads(json_text)
    except (ValueError, TypeError):
        return out
    for mod in (doc.get("modules") or {}).values():
        for cell in (mod.get("cells") or {}).values():
            if "mem" not in str(cell.get("type", "")).lower():
                continue
            p = cell.get("parameters", {}) or {}
            w = _as_int(p.get("WIDTH", 0))
            d = _as_int(p.get("SIZE", 0))
            if w > 0 and d > 0:
                out.append((w, d))
        for mem in (mod.get("memories") or {}).values():
            w = _as_int(mem.get("width", 0))
            d = _as_int(mem.get("size", 0))
            if w > 0 and d > 0:
                out.append((w, d))
    return out


_SHELL_TYPE_RE = re.compile(r"cs_(?:mem|rom)_macro_shell")


def _count_shell_cells(json_text: str) -> int:
    """Count instantiated macro-shell CELLS in a yosys ``write_json`` dump
    (cells whose type is a cs_mem/cs_rom macro shell) -- NOT the shell module
    definitions the wrapper library always carries."""
    import json as _json
    try:
        doc = _json.loads(json_text)
    except (ValueError, TypeError):
        return 0
    n = 0
    for mod in (doc.get("modules") or {}).values():
        for cell in (mod.get("cells") or {}).values():
            if _SHELL_TYPE_RE.search(str(cell.get("type", ""))):
                n += 1
    return n


def probe_memory_flops(
    rtl_paths: list[str],
    top: str,
    *,
    apply_macro: bool = True,
    yosys_bin: str = "yosys",
    timeout_s: int = 300,
    cwd: str | None = None,
) -> dict[str, Any] | None:
    """Memory-preserving, MACRO-aware synth probe for the memory-as-flops gate.

    Reads ``rtl_paths`` (design + the cs_sram wrapper library), applies the
    Part-B MACRO chparam BEFORE ``hierarchy`` (so wrapped memories become
    shells), then ``proc; opt; memory_collect`` -- deliberately NOT
    ``memory_map`` -- so a residual inferred memory stays a ``$mem`` we can size.

    Returns ``None`` only when yosys/RTL are absent (cannot judge). Otherwise
    ``{memories: [(w,d)], macro_shells: int, total_mem_bits, biggest_bits,
    elaborated, reason}``; ``elaborated`` is False on timeout / non-zero exit.
    """
    yosys = shutil.which(yosys_bin)
    files = [p for p in (rtl_paths or []) if p and Path(p).exists()]
    if not yosys or not files:
        return None
    reads = "".join(f"read_verilog -sv {p}\n" for p in files)
    directive = ""
    if apply_macro:
        try:
            from orchestrator.langgraph.sram_wrapper import backend_sram_macro_directive
            directive = backend_sram_macro_directive(force=True)
        except Exception:  # noqa: BLE001 - probe must never crash on import
            directive = ""
    directive_line = (directive + "\n") if directive else ""
    ys = json_path = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".ys", delete=False) as fh:
            ys = fh.name
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as jf:
            json_path = jf.name
        script = (
            f"{reads}{directive_line}hierarchy -top {top}\n"
            f"proc\nopt\nmemory_collect\nopt_clean\nstat\nwrite_json {json_path}\n"
        )
        Path(ys).write_text(script)
        _cwd = cwd if (cwd and Path(cwd).is_dir()) else None
        try:
            result = subprocess.run(
                [yosys, "-s", ys], capture_output=True, text=True,
                timeout=timeout_s, cwd=_cwd,
            )
        except subprocess.TimeoutExpired:
            return {"memories": [], "macro_shells": 0, "total_mem_bits": 0,
                    "biggest_bits": 0, "elaborated": False,
                    "reason": f"did not elaborate within {timeout_s}s "
                              "(combinational loop or oversized flop memory)"}
        except OSError:
            return None
        if result.returncode != 0:
            tail = (result.stderr or result.stdout or "")[-300:]
            return {"memories": [], "macro_shells": 0, "total_mem_bits": 0,
                    "biggest_bits": 0, "elaborated": False,
                    "reason": f"yosys did not elaborate the design: {tail.strip()}"}
        json_text = Path(json_path).read_text(errors="ignore")
    finally:
        for _p in (ys, json_path):
            try:
                if _p:
                    Path(_p).unlink()
            except OSError:
                pass
    memories = _parse_json_memories(json_text)
    # Count shell INSTANCES (cells), not the shell MODULE DEFINITIONS that the
    # cs_sram wrapper library always carries -- otherwise a BEHAV/flop netlist
    # (lib read, shell defined but never instantiated) reads as "has a shell".
    macro_shells = _count_shell_cells(json_text)
    total_bits = sum(w * d for w, d in memories)
    biggest = max((w * d for w, d in memories), default=0)
    return {
        "memories": memories,
        "macro_shells": macro_shells,
        "total_mem_bits": total_bits,
        "biggest_bits": biggest,
        "elaborated": True,
        "reason": "",
    }
