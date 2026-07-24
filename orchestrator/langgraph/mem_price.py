# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tier-2 physical-feasibility gate: machine-readable memory pricing + die rollup.

This is the deterministic backstop for the failure class where a µArch spec
declares a large storage element in PROSE (e.g. a full-dimension reconstruction
store, 8x235,520 = 1.9 Mbit), nothing prices it, per-block area budgets are
LLM-invented std-cell numbers that never account for macro area, and no die cap
exists -- so the store reaches the backend as ~230 SRAM macros / ~60 mm² of
macro area (6x a small MPW die) completely unflagged.

Two things live here, both DOMAIN-GENERIC (dimension/storage vocabulary only --
no frames/pixels/video):

1. **Memory manifest** -- one machine-readable line per storage element::

     # MEM <name>: <width>x<depth> ports=<1rw|1rw1r|2rw|...> impl=<flop|fpmem|sram> justification=<...>

   The µArch/block-golden PROMPTS mandate it; this module PARSES it. The legacy
   ``tier=<macro|registered_flop>`` / ``role=`` / ``reason=`` spelling from the
   experimental microarch flow is tolerated as a synonym.

2. **Pricing + gates** -- each declared memory is priced (real PDK area via
   ``mem_characterize.predict_mem`` when the characterizer cache is warm; an
   analytic *flop-bits* estimate otherwise -- never blocking on a missing PDK),
   and two gates fire off the priced ledger:
     * per-block: Σ(priced memory area) over the block's ``area_budget_um2``, OR
       any single memory over a hard sanity cap (``CORESMITH_MEM_SANITY_MM2``,
       default 2.0 mm²);
     * chip-level *die rollup*: Σ(per-block area) + interconnect margin over a
       resolved die-area budget (env / PRD / shuttle default).

Everything here is pure + deterministic (the only external call is
``predict_mem``, which itself degrades to ``None`` with no PDK -> analytic
fallback). Nothing raises on a missing input; a gate that cannot judge returns
``ok`` and records *why* rather than blocking.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants / env knobs
# ---------------------------------------------------------------------------

# Analytic "flop-bits" area floor: sky130 D-flop-bit incl. the load-enable mux +
# routing at typical utilization. At ~25 um^2/bit this ALSO tracks the sky130
# OpenRAM per-bit density at scale (~0.2 mm^2/KB), so it doubles as a macro-area
# floor when no PDK is present -- i.e. a 1.9 Mbit store prices at ~47 mm^2 either
# way, which is exactly the physics the gate must surface.
DEFAULT_FLOP_UM2_PER_BIT = 25.0

# Default per-memory sanity cap (mm^2). One memory bigger than this is a red flag
# regardless of the block's (possibly LLM-inflated) area budget.
DEFAULT_MEM_SANITY_MM2 = 2.0

# Above this many bits a memory is FAR outside the characterizer's sweep grid,
# where predict_mem's regressor extrapolates unreliably (observed: a 1.9 Mbit
# store priced by the model at ~2 mm^2, a ~20x under-estimate). For such large
# out-of-grid memories the analytic flop-bits value is used as a hard FLOOR under
# the model number so the gate never under-prices a huge store. Small, in-grid
# memories (below this) trust predict_mem verbatim -> no over-pricing of real
# macro LEF areas.
DEFAULT_FLOOR_MIN_BITS = 131_072  # 128 Kbit

# Interconnect / whitespace margin added on top of summed block area for the die
# rollup (routing channels, power grid, fill, block spacing).
DEFAULT_INTERCONNECT_MARGIN = 0.15
# A single-block design has NO inter-block routing, so the interconnect margin
# (which models channels BETWEEN blocks) does not apply -- otherwise a lone
# block legitimately budgeted to fill its die always busts the rollup by exactly
# the margin (mcu3 OOD run: 0.050 mm^2 block vs 0.050 mm^2 die -> 0.0575 after
# +15% -> false overflow). PDN/fill/edge-spacing are already inside the block's
# own area budget. Override via CORESMITH_SINGLE_BLOCK_MARGIN.
DEFAULT_SINGLE_BLOCK_MARGIN = 0.0

# When a shuttle is named but no explicit die budget is given, default the die
# cap to this (a ChipIgnite/Caravel user area is ~10 mm^2).
DEFAULT_SHUTTLE_DIE_MM2 = 10.0

# Std-cell gate density used to turn a block's ``estimated_gates`` into an area
# estimate when it has no explicit ``area_budget_um2`` (mirrors the shuttle
# checker's ~200K gates/mm^2 at ~60% utilization).
_GATE_DENSITY_PER_MM2 = 200_000

_FALSE_TOKENS = {"0", "false", "no", "off", ""}


def _flag_default_on(name: str) -> bool:
    """A ``CORESMITH_*`` flag that is ON unless explicitly disabled.

    Reads env directly (default "1") so it does NOT depend on the strict/legacy
    profile -- matching the other default-ON gates (ifdef lint, max-geometry).
    """
    return os.environ.get(name, "1").strip().lower() not in _FALSE_TOKENS


def mem_price_gate_enabled() -> bool:
    """CORESMITH_MEM_PRICE_GATE (default ON). 0/false/no/off to bypass."""
    return _flag_default_on("CORESMITH_MEM_PRICE_GATE")


def die_rollup_gate_enabled() -> bool:
    """CORESMITH_DIE_ROLLUP_GATE (default ON). 0/false/no/off to bypass."""
    return _flag_default_on("CORESMITH_DIE_ROLLUP_GATE")


def manifest_required() -> bool:
    """CORESMITH_MEM_MANIFEST_REQUIRED (default OFF).

    OFF (default): a storage-declaring spec with NO ``# MEM`` manifest passes
    with a LOUD warning (legacy / operator-provided prose-only specs). ON
    (strict): such a spec is REJECTED with "emit the manifest" -- the state a
    NEW pipeline-generated spec must satisfy once the prompt mandate lands.
    A spec that declares no storage at all never needs a manifest either way.
    """
    raw = os.environ.get("CORESMITH_MEM_MANIFEST_REQUIRED")
    if raw is None:
        return False
    return raw.strip().lower() not in _FALSE_TOKENS


def mem_sanity_mm2() -> float:
    """CORESMITH_MEM_SANITY_MM2 (default 2.0 mm^2) per-memory hard cap."""
    try:
        v = float(os.environ.get("CORESMITH_MEM_SANITY_MM2", "") or DEFAULT_MEM_SANITY_MM2)
        return v if v > 0 else DEFAULT_MEM_SANITY_MM2
    except ValueError:
        return DEFAULT_MEM_SANITY_MM2


def flop_um2_per_bit() -> float:
    """CORESMITH_FLOP_UM2_PER_BIT (default 25.0) analytic per-bit area floor."""
    try:
        v = float(os.environ.get("CORESMITH_FLOP_UM2_PER_BIT", "") or DEFAULT_FLOP_UM2_PER_BIT)
        return v if v > 0 else DEFAULT_FLOP_UM2_PER_BIT
    except ValueError:
        return DEFAULT_FLOP_UM2_PER_BIT


def floor_min_bits() -> int:
    """CORESMITH_MEM_FLOOR_MIN_BITS (default 131072) -- above this the analytic
    flop-bits value floors predict_mem (guards model out-of-grid under-pricing)."""
    try:
        v = int(os.environ.get("CORESMITH_MEM_FLOOR_MIN_BITS", "") or DEFAULT_FLOOR_MIN_BITS)
        return v if v > 0 else DEFAULT_FLOOR_MIN_BITS
    except ValueError:
        return DEFAULT_FLOOR_MIN_BITS


def interconnect_margin() -> float:
    """CORESMITH_INTERCONNECT_MARGIN (default 0.15)."""
    try:
        v = float(os.environ.get("CORESMITH_INTERCONNECT_MARGIN", "") or DEFAULT_INTERCONNECT_MARGIN)
        return v if v >= 0 else DEFAULT_INTERCONNECT_MARGIN
    except ValueError:
        return DEFAULT_INTERCONNECT_MARGIN


def single_block_margin() -> float:
    """CORESMITH_SINGLE_BLOCK_MARGIN (default 0.0) -- margin used when the die
    holds exactly one block (no inter-block routing)."""
    try:
        v = float(os.environ.get("CORESMITH_SINGLE_BLOCK_MARGIN", "") or DEFAULT_SINGLE_BLOCK_MARGIN)
        return v if v >= 0 else DEFAULT_SINGLE_BLOCK_MARGIN
    except ValueError:
        return DEFAULT_SINGLE_BLOCK_MARGIN


def die_budget_env_mm2() -> float | None:
    """CORESMITH_DIE_BUDGET_MM2 override, or None."""
    raw = os.environ.get("CORESMITH_DIE_BUDGET_MM2", "").strip()
    if not raw:
        return None
    try:
        v = float(raw)
        return v if v > 0 else None
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Manifest parsing
# ---------------------------------------------------------------------------

# `# MEM <name>: <W>x<D> <k=v>...`  -- name is any identifier-ish token; the
# WIDTHxDEPTH uses x / X / the unicode multiplication sign.
_MEM_LINE_RE = re.compile(
    r"^\s*#\s*MEM\s+(?P<name>[^\s:]+)\s*:\s*"
    r"(?P<w>\d+)\s*[xX×]\s*(?P<d>\d+)"
    r"(?P<rest>.*)$",
    re.MULTILINE,
)
_KV_RE = re.compile(r"(\w+)\s*=\s*([^\s].*?)(?=\s+\w+\s*=|$)")

# impl synonyms: legacy tier=macro/registered_flop -> impl=sram/fpmem.
_IMPL_SYNONYMS = {
    "macro": "sram",
    "sram": "sram",
    "registered_flop": "fpmem",
    "regflop": "fpmem",
    "fpmem": "fpmem",
    "flop": "flop",
    "flops": "flop",
    "ff": "flop",
    "reg": "flop",
    # read-only constant table -> OpenRAM rom_compiler mask ROM (cs_rom_1r)
    "rom": "rom",
    "mask_rom": "rom",
    "maskrom": "rom",
    "cs_rom": "rom",
}


@dataclass
class MemDecl:
    """One parsed ``# MEM`` manifest line."""

    name: str
    width: int
    depth: int
    ports: str = "1rw"
    impl: str = ""             # flop | fpmem | sram (normalized) | "" (unstated)
    justification: str = ""
    raw: str = ""

    @property
    def bits(self) -> int:
        return int(self.width) * int(self.depth)


def _normalize_impl(value: str) -> str:
    v = (value or "").strip().lower().split()[0] if value else ""
    v = v.strip(",;")
    return _IMPL_SYNONYMS.get(v, v)


def parse_mem_manifest(spec_text: str) -> list[MemDecl]:
    """Parse every ``# MEM`` manifest line out of a µArch/block-golden spec.

    Tolerates the ``impl=`` and legacy ``tier=`` spellings and the
    ``justification=`` / ``reason=`` / ``role=`` rationale keys. A line missing
    ports defaults to ``1rw``; a malformed line is skipped (the prompt is the
    primary defense; this parser never raises).
    """
    out: list[MemDecl] = []
    if not spec_text:
        return out
    for m in _MEM_LINE_RE.finditer(spec_text):
        try:
            width = int(m.group("w"))
            depth = int(m.group("d"))
        except (TypeError, ValueError):
            continue
        if width <= 0 or depth <= 0:
            continue
        rest = m.group("rest") or ""
        kv = {k.lower(): v.strip() for k, v in _KV_RE.findall(rest)}
        impl = _normalize_impl(kv.get("impl") or kv.get("tier") or "")
        just = kv.get("justification") or kv.get("reason") or kv.get("role") or ""
        ports = (kv.get("ports") or "1rw").strip().strip(",;")
        out.append(MemDecl(
            name=m.group("name").strip(),
            width=width, depth=depth, ports=ports or "1rw",
            impl=impl, justification=just,
            raw=m.group(0).strip(),
        ))
    return out


def manifest_signature(decls: list[MemDecl]) -> str:
    """Stable content hash of a parsed manifest (order-independent).

    Used by the revise loop to detect a byte-identical re-submission: a spec
    whose ``# MEM`` set is unchanged (same names/geometry/ports/impl) will price
    to the same total and CANNOT clear the gate, so the loop short-circuits
    straight to the bounded outcome instead of burning another LLM round.
    Sorts on the priced-relevant fields only (justification prose is ignored --
    it does not change the physics).
    """
    rows = sorted(
        f"{d.name}|{int(d.width)}x{int(d.depth)}|{d.ports}|{d.impl}"
        for d in decls
    )
    return hashlib.sha1("\n".join(rows).encode("utf-8")).hexdigest()[:16]


# The machine-readable "this spec HAS storage worth pricing" signal: a positive
# ``sram_budget`` (bits / KiB / a named macro with a count). Used ONLY to decide
# whether an ABSENT manifest is a reject (strict) vs a warn (legacy) -- never to
# price (pricing is manifest-driven). Domain-generic (no frame/line vocab).
_SRAM_BUDGET_POS_RE = re.compile(
    r"sram_budget[^\n]{0,40}?(\d[\d,]*)\s*(?:b|bit|bits|kib|kb|kbit|byte|bytes|x)\b",
    re.IGNORECASE,
)
_SRAM_MACRO_RE = re.compile(r"sram_budget[^\n]{0,80}?\bsky130_sram\w+", re.IGNORECASE)


def spec_declares_storage(spec_text: str) -> bool:
    """True if the spec machine-readably declares non-trivial on-chip storage.

    A positive ``sram_budget`` quantity or a named sky130 SRAM macro next to
    ``sram_budget``. ``sram_budget = 0`` (or absent) reads as no declared macro
    storage. This is intentionally conservative: it drives ONLY the
    absent-manifest reject-vs-warn decision, not any pricing.
    """
    if not spec_text:
        return False
    if _SRAM_MACRO_RE.search(spec_text):
        return True
    for m in _SRAM_BUDGET_POS_RE.finditer(spec_text):
        try:
            if int(m.group(1).replace(",", "")) > 0:
                return True
        except ValueError:
            continue
    return False


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------

# declared impl -> predict_mem candidate key
_CAND_KEY = {"sram": "macro", "fpmem": "registered_flop", "flop": "flop"}


@dataclass
class PricedMem:
    """A priced memory declaration + provenance of the area estimate."""

    decl: MemDecl
    area_um2: float
    # provenance of the area estimate:
    # "pdk_predict_mem" (measured row / LEF-exact / in-grid regressor) |
    # "analytic_extrapolation" (predict_mem out-of-grid analytic estimate) |
    # "analytic_sram_bits" (declared SRAM using the cs_sram/OpenRAM ruler) |
    # "analytic_rom_bits" (declared ROM using the mask-ROM affine ruler) |
    # "analytic_flop_bits" (this module's per-bit floor / cold-cache fallback)
    estimate_source: str
    recommended_impl: str = ""
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.decl.name,
            "width": self.decl.width,
            "depth": self.decl.depth,
            "bits": self.decl.bits,
            "ports": self.decl.ports,
            "declared_impl": self.decl.impl,
            "area_um2": round(self.area_um2, 2),
            "area_mm2": round(self.area_um2 / 1e6, 4),
            "estimate_source": self.estimate_source,
            "recommended_impl": self.recommended_impl,
            "justification": self.decl.justification,
            "detail": self.detail,
        }


def analytic_flop_bits_area_um2(width: int, depth: int) -> float:
    """Analytic area floor: ``width*depth`` bits at the flop-bit per-bit cost."""
    return float(width) * float(depth) * flop_um2_per_bit()


def characterizer_warm(pdk: dict | None = None) -> bool:
    """True iff the memory characterizer cache has any rows (PDK pricing usable)."""
    try:
        from orchestrator.langgraph.mem_characterize import load_table
        return bool(load_table(pdk))
    except Exception:  # noqa: BLE001 - no PDK / import hiccup -> analytic path
        return False


def price_mem_decl(decl: MemDecl, *, warm: bool | None = None,
                   pdk: dict | None = None,
                   target_mhz: float = 100.0) -> PricedMem:
    """Price one memory: real PDK area (warm cache) else analytic flop-bits.

    When the characterizer cache is warm we consult ``predict_mem`` and take the
    area of the candidate matching the DECLARED impl (falling back to the
    recommended impl's predicted area). Any hole -- cold cache, ``None`` area,
    or a raised predictor -- fails OPEN to the analytic flop-bits floor and
    records ``estimate_source="analytic_flop_bits"`` so the ledger is honest
    about which ruler was used.
    """
    if warm is None:
        warm = characterizer_warm(pdk)
    floor = analytic_flop_bits_area_um2(decl.width, decl.depth)
    apply_floor = decl.bits >= floor_min_bits()
    # A declared ROM (read-only constant table) is a mask ROM generated by the
    # OpenRAM rom_compiler -- it must NEVER be priced as SRAM bits, let alone
    # flop bits (the reference codec encoder's 21-bank codec_rom_memory priced at 5.88
    # mm^2 as SRAM; the same bits as mask ROM are ~an order of magnitude
    # smaller). The characterizer grid has no ROM candidates, so this branch
    # is authoritative until an exact generated-LEF area exists.
    if decl.impl == "rom":
        from orchestrator.langgraph.sram_wrapper import (
            rom_area_um2,
            rom_um2_per_bit,
        )
        return PricedMem(
            decl=decl,
            area_um2=rom_area_um2(decl.bits),
            estimate_source="analytic_rom_bits",
            recommended_impl="rom",
            detail=(f"OpenRAM mask ROM; affine ruler overhead + "
                    f"{rom_um2_per_bit():.3f} um^2/bit x {decl.bits:,} bits "
                    f"(replace with exact LEF area when generated)"),
        )
    if warm:
        try:
            from orchestrator.langgraph.mem_characterize import predict_mem
            pred = predict_mem(decl.width, decl.depth, decl.ports,
                               target_mhz=target_mhz, pdk=pdk)
            cands = pred.get("candidates") or {}
            key = _CAND_KEY.get(decl.impl or pred.get("recommended_impl", ""), "")
            area = None
            used_source = ""
            if key and key in cands:
                area = (cands[key] or {}).get("area_um2")
                used_source = str((cands[key] or {}).get("source", ""))
            # A declared SRAM must never inherit the registered-flop candidate
            # merely because the characterized catalog has no exact geometry,
            # nor may an out-of-grid regressor price it with a flop-read-mux
            # slope.  Such geometries are generated by OpenRAM; use the same
            # deterministic SRAM bit-density ruler as cs_sram PPA until a LEF
            # for that exact macro is available.  Exact/in-grid PDK macro areas
            # remain authoritative.
            if decl.impl == "sram" and (
                area is None or used_source == "analytic_extrapolation"
            ):
                from orchestrator.langgraph.sram_wrapper import um2_per_bit
                ppb = um2_per_bit()
                return PricedMem(
                    decl=decl,
                    area_um2=float(decl.bits) * ppb,
                    estimate_source="analytic_sram_bits",
                    recommended_impl="sram",
                    detail=(f"custom OpenRAM geometry; architecture SRAM ruler "
                            f"{ppb:.3f} um^2/bit x {decl.bits:,} bits "
                            f"(replace with exact LEF area when generated)"),
                )
            if area is None:
                area = pred.get("pred_area_um2")
                used_source = str(pred.get("estimate_source", ""))
            # Thread predict_mem's provenance through: an OUT-OF-GRID analytic
            # extrapolation is honestly labelled as such (not "pdk_predict_mem"),
            # so the ledger + reviewer see which ruler priced the memory.
            src_label = ("analytic_extrapolation"
                         if used_source == "analytic_extrapolation"
                         else "pdk_predict_mem")
            if area is not None and area > 0:
                rec = str(pred.get("recommended_impl", ""))
                if apply_floor and floor > float(area):
                    return PricedMem(
                        decl=decl, area_um2=floor,
                        estimate_source="analytic_flop_bits",
                        recommended_impl=rec,
                        detail=(f"predict_mem {float(area):,.0f} um^2 "
                                f"[{used_source or '?'}] below the analytic floor "
                                f"for a {decl.bits:,}-bit out-of-grid memory; "
                                f"using floor {flop_um2_per_bit():.1f} um^2/bit"),
                    )
                return PricedMem(
                    decl=decl, area_um2=float(area),
                    estimate_source=src_label,
                    recommended_impl=rec,
                    detail=str(pred.get("reason", "")),
                )
        except Exception:  # noqa: BLE001 - degrade to analytic, never block
            pass
    if decl.impl == "sram":
        from orchestrator.langgraph.sram_wrapper import um2_per_bit
        ppb = um2_per_bit()
        return PricedMem(
            decl=decl,
            area_um2=float(decl.bits) * ppb,
            estimate_source="analytic_sram_bits",
            recommended_impl="sram",
            detail=(f"custom OpenRAM geometry; architecture SRAM ruler "
                    f"{ppb:.3f} um^2/bit x {decl.bits:,} bits "
                    f"(replace with exact LEF area when generated)"),
        )
    return PricedMem(
        decl=decl, area_um2=floor,
        estimate_source="analytic_flop_bits",
        recommended_impl=decl.impl,
        detail=f"analytic floor {flop_um2_per_bit():.1f} um^2/bit x {decl.bits:,} bits",
    )


def price_manifest(decls: list[MemDecl], *, pdk: dict | None = None,
                   target_mhz: float = 100.0) -> list[PricedMem]:
    """Price every declaration (warm-cache probe done once for the batch)."""
    warm = characterizer_warm(pdk)
    return [price_mem_decl(d, warm=warm, pdk=pdk, target_mhz=target_mhz) for d in decls]


# ---------------------------------------------------------------------------
# Per-block verdict
# ---------------------------------------------------------------------------

@dataclass
class MemPriceVerdict:
    """Outcome of the per-block memory-price gate."""

    ok: bool
    reasons: list[str] = field(default_factory=list)
    checks: list[dict[str, Any]] = field(default_factory=list)
    total_um2: float = 0.0
    priced: list[PricedMem] = field(default_factory=list)


def _fmt_mm2(um2: float) -> str:
    return f"{um2 / 1e6:.3f} mm^2 ({um2:,.0f} um^2)"


def evaluate_mem_price(priced: list[PricedMem], *,
                       area_budget_um2: float | None = None,
                       sanity_mm2: float | None = None) -> MemPriceVerdict:
    """Gate a block's priced memory: per-memory sanity cap + Σ-vs-area-budget.

    * Any SINGLE memory over ``sanity_mm2`` (default 2.0 mm^2) is a hard FAIL,
      independent of the block's (possibly inflated) area budget -- the exact
      backstop for a 1.9 Mbit store priced at tens of mm^2.
    * Σ(priced memory area) over ``area_budget_um2`` (when declared) is a FAIL.
    Every rejection carries the price so the regen agent sees the physics.
    """
    cap_mm2 = sanity_mm2 if sanity_mm2 is not None else mem_sanity_mm2()
    cap_um2 = cap_mm2 * 1e6
    reasons: list[str] = []
    checks: list[dict[str, Any]] = []
    ok = True
    total = 0.0
    for pm in priced:
        total += pm.area_um2
        over = pm.area_um2 > cap_um2
        checks.append({
            "metric": "mem_sanity_cap", "memory": pm.decl.name,
            "actual_um2": round(pm.area_um2, 2), "cap_mm2": cap_mm2,
            "estimate_source": pm.estimate_source, "passed": not over,
        })
        if over:
            ok = False
            reasons.append(
                f"memory `{pm.decl.name}` ({pm.decl.width}x{pm.decl.depth} = "
                f"{pm.decl.bits:,} bits, impl={pm.decl.impl or '?'}) prices at "
                f"{_fmt_mm2(pm.area_um2)} [{pm.estimate_source}] -- over the "
                f"{cap_mm2:.1f} mm^2 per-memory sanity cap. A single storage "
                f"element this large is physically infeasible on a small die; "
                f"shrink the declared dimension to its true dependency window "
                f"(the line-buffer-vs-frame-store question) or bank/stream it."
            )
    if area_budget_um2 is not None and area_budget_um2 > 0:
        passed = total <= area_budget_um2
        checks.append({
            "metric": "mem_area_vs_budget", "actual_um2": round(total, 2),
            "budget_um2": area_budget_um2, "passed": passed,
        })
        if not passed:
            ok = False
            reasons.append(
                f"Σ priced memory {_fmt_mm2(total)} exceeds the block "
                f"area_budget_um2 {area_budget_um2:,.0f} um^2 "
                f"({area_budget_um2 / 1e6:.3f} mm^2). The declared storage does "
                f"not fit the block's area budget -- either the memory is "
                f"oversized (reshape to the true dependency window) or the "
                f"budget under-prices its macro area."
            )
    return MemPriceVerdict(ok=ok, reasons=reasons, checks=checks,
                           total_um2=total, priced=list(priced))


# ---------------------------------------------------------------------------
# Revise-loop convergence: trajectory + directive-rich regen feedback
# ---------------------------------------------------------------------------

# Below this Σ-area delta (um^2) two rounds are treated as "flat" (no-op change)
# rather than better/worse -- guards against float dithering.
_TRAJECTORY_EPS_UM2 = 1.0


def trajectory_label(prev_total_um2: float | None, cur_total_um2: float) -> str:
    """Classify a revise round's Σ-priced-area against the previous round.

    ``"first"`` (no prior round) | ``"worse"`` (total went UP -- the regen added
    bits/memories, the exact anti-pattern the mem-price loop must break) |
    ``"better"`` (total came down but still over budget) | ``"flat"`` (unchanged
    within epsilon). Pure; drives both the gate event and the regen directive.
    """
    if prev_total_um2 is None:
        return "first"
    d = cur_total_um2 - prev_total_um2
    if d > _TRAJECTORY_EPS_UM2:
        return "worse"
    if d < -_TRAJECTORY_EPS_UM2:
        return "better"
    return "flat"


def _pct_of_budget(area_um2: float, budget_um2: float | None) -> str:
    if budget_um2 and budget_um2 > 0:
        return f"{area_um2 / budget_um2 * 100:.0f}%"
    return "n/a"


def format_revise_directive(
    block: str,
    verdict: MemPriceVerdict,
    *,
    area_budget_um2: float | None,
    round_idx: int,
    max_revise: int,
    prev_total_um2: float | None = None,
    prev_n_memories: int | None = None,
    trajectory: str = "",
) -> str:
    """Build the directive-rich regen feedback threaded into the next spec prompt.

    Fix for the non-convergent mem-price revise loop: the prior feedback carried a
    price table but a weak "re-declare the manifest" directive that the regen
    agent misread as a granularity/completeness ask -- so it responded to a
    rejection by DECLARING MORE memories (total area went UP). This builds:

    * a compact per-memory table (name, WxD, ports, impl, priced mm^2, % of the
      block area budget) so the agent sees which entries dominate;
    * an explicit numeric target (Σ MUST come down to <= the block budget);
    * directional directives to REDUCE stored bits (shrink depths to the true
      dependency window, share buffers, stream working sets) with an anti-pattern
      guard -- splitting entries or adding manifest lines does NOT help because
      the gate sums PRICED BITS, not line count;
    * the round's TRAJECTORY vs the previous revision, so a WRONG-DIRECTION regen
      is called out with its own delta ("round N: 2.756 -> your revision: 4.498").
    """
    budget_mm2 = (area_budget_um2 or 0) / 1e6
    total_mm2 = verdict.total_um2 / 1e6
    n = len(verdict.priced)

    lines: list[str] = []
    lines.append(
        f"MEMORY BUDGET EXCEEDED -- Tier-2 physical-feasibility gate "
        f"(revise round {round_idx}/{max_revise})."
    )
    lines.append(
        "Your declared on-chip memories do not fit this block's area budget. "
        "Priced ledger (the ruler used per memory):"
    )
    lines.append("")
    lines.append(
        f"  {'memory':<22} {'WxD':<14} {'ports':<7} {'impl':<6} "
        f"{'priced mm^2':>11}  {'% budget':>8}"
    )
    lines.append("  " + "-" * 70)
    for pm in verdict.priced:
        lines.append(
            f"  {pm.decl.name[:22]:<22} "
            f"{f'{pm.decl.width}x{pm.decl.depth}'[:14]:<14} "
            f"{(pm.decl.ports or '?')[:7]:<7} "
            f"{(pm.decl.impl or '?')[:6]:<6} "
            f"{pm.area_um2 / 1e6:>11.3f}  "
            f"{_pct_of_budget(pm.area_um2, area_budget_um2):>8}"
        )
    lines.append("  " + "-" * 70)
    lines.append(
        f"  {f'TOTAL ({n} memories)':<22} {'':<14} {'':<7} {'':<6} "
        f"{total_mm2:>11.3f}  {_pct_of_budget(verdict.total_um2, area_budget_um2):>8}"
    )
    lines.append("")
    if area_budget_um2 and area_budget_um2 > 0:
        over_x = verdict.total_um2 / area_budget_um2
        lines.append(
            f"Block area budget: {budget_mm2:.3f} mm^2 ({area_budget_um2:,.0f} um^2). "
            f"Sigma priced = {total_mm2:.3f} mm^2 -- {over_x:.1f}x OVER budget."
        )

    # Trajectory: call out a WRONG-DIRECTION regen with its own delta.
    if trajectory == "worse" and prev_total_um2 is not None:
        prev_mm2 = prev_total_um2 / 1e6
        pn = f"{prev_n_memories} mems" if prev_n_memories is not None else "prior"
        lines.append(
            f"TRAJECTORY: WORSE. Your last revision INCREASED the total -- "
            f"round {round_idx - 1}: {prev_mm2:.3f} mm^2 ({pn}) -> your revision: "
            f"{total_mm2:.3f} mm^2 ({n} mems). WRONG DIRECTION: you must REDUCE "
            f"total stored bits, not add memories."
        )
    elif trajectory == "better" and prev_total_um2 is not None:
        lines.append(
            f"TRAJECTORY: improving ({prev_total_um2 / 1e6:.3f} -> {total_mm2:.3f} "
            f"mm^2) but STILL over budget -- keep reducing stored bits."
        )
    elif trajectory == "flat" and prev_total_um2 is not None:
        lines.append(
            f"TRAJECTORY: UNCHANGED total ({total_mm2:.3f} mm^2) -- the last "
            f"directive had no effect. You must actually SHRINK stored bits."
        )

    lines.append("")
    lines.append("WHAT TO DO -- reduce PRICED BITS, not line count:")
    if area_budget_um2 and area_budget_um2 > 0:
        lines.append(
            f"- Sigma priced area MUST come DOWN to <= {budget_mm2:.3f} mm^2 "
            f"(the block area budget)."
        )
    else:
        lines.append("- Sigma priced area MUST come DOWN under the block area budget.")
    lines.append(
        "- REDUCE total stored bits: shrink each depth to its TRUE dependency "
        "window (a line buffer is a few rows deep, not a whole-dimension store); "
        "SHARE buffers across stages; move working sets to STREAMING instead of "
        "buffering the whole set."
    )
    lines.append(
        "- Splitting one memory into several, or adding # MEM lines, does NOT "
        "help: the gate sums PRICED BITS across ALL memories, not the number of "
        "manifest lines. Do NOT add memories."
    )
    lines.append(
        "- Re-declare the # MEM manifest with the corrected (SMALLER) geometry."
    )
    if verdict.reasons:
        lines.append("")
        lines.append("Gate reasons:")
        for r in verdict.reasons:
            lines.append(f"- {r}")
    return "\n".join(lines)


def format_ledger(block: str, verdict: MemPriceVerdict, *,
                  area_budget_um2: float | None,
                  manifest_present: bool,
                  note: str = "",
                  over_budget: bool | None = None,
                  deferred: bool = False,
                  deferred_reason: str = "",
                  reject_rounds: int | None = None,
                  trajectory: str = "",
                  signature: str = "") -> dict[str, Any]:
    """Build the scoreboard-friendly ``mem_price.json`` ledger dict.

    ``over_budget`` (defaults to ``not verdict.ok``) is the machine-readable
    "this block busts its budget" flag; the die rollup + integration review read
    it. When the bounded revise loop gives up, the defer path re-writes the
    ledger with ``deferred=True`` + the ``deferred_reason`` / ``reject_rounds`` /
    final ``trajectory`` so the deferred excess is durable + visible downstream.
    ``signature`` pins the manifest content hash so the next round can detect a
    byte-identical re-submission.
    """
    return {
        "block": block,
        "manifest_present": manifest_present,
        "ok": verdict.ok,
        "over_budget": (not verdict.ok) if over_budget is None else bool(over_budget),
        "deferred": bool(deferred),
        "deferred_reason": deferred_reason,
        "reject_rounds": reject_rounds,
        "trajectory": trajectory,
        "manifest_signature": signature,
        "total_area_um2": round(verdict.total_um2, 2),
        "total_area_mm2": round(verdict.total_um2 / 1e6, 4),
        "area_budget_um2": area_budget_um2,
        "sanity_cap_mm2": mem_sanity_mm2(),
        "memories": [pm.to_dict() for pm in verdict.priced],
        "reasons": verdict.reasons,
        "note": note,
    }


def write_ledger(project_root: str, block: str, ledger: dict[str, Any]) -> str | None:
    """Persist the ledger to ``.coresmith/blocks/<block>/mem_price.json``."""
    try:
        d = Path(project_root) / ".coresmith" / "blocks" / block
        d.mkdir(parents=True, exist_ok=True)
        p = d / "mem_price.json"
        p.write_text(json.dumps(ledger, indent=2))
        return str(p)
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Die-area budget resolution
# ---------------------------------------------------------------------------

_SHUTTLE_TOKENS = (
    "openframe", "caravel", "chipignite", "chip ignite", "chip-ignite",
    "mpw", "efabless shuttle", "shuttle gpio", "openframe wrapper", "mpw wrapper",
)


def mentions_shuttle(*texts: str) -> bool:
    """True if any text names a known MPW shuttle (openframe/caravel/chipignite/mpw)."""
    hay = " ".join(t for t in texts if t).lower()
    return any(tok in hay for tok in _SHUTTLE_TOKENS)


def _prd_die_budget_mm2(prd: dict | None) -> float | None:
    """max_die_area_mm2 out of a PRD/ERS ``area_budget`` block, if present."""
    if not isinstance(prd, dict):
        return None
    # tolerate {prd:{area_budget:...}} / {ers:{...}} / flat {area_budget:...}
    for key in ("prd", "ers"):
        inner = prd.get(key)
        if isinstance(inner, dict):
            v = _prd_die_budget_mm2(inner)
            if v is not None:
                return v
    ab = prd.get("area_budget")
    if isinstance(ab, dict):
        for k in ("max_die_area_mm2", "die_area_mm2", "max_die_mm2"):
            v = ab.get(k)
            try:
                if v is not None and float(v) > 0:
                    return float(v)
            except (TypeError, ValueError):
                continue
    return None


def resolve_die_budget_mm2(*, prd: dict | None = None,
                           requirements: str = "",
                           ers_technology_text: str = "") -> tuple[float | None, str]:
    """Resolve the machine-readable die-area cap (mm^2) + its source.

    Priority: env ``CORESMITH_DIE_BUDGET_MM2`` > PRD/ERS ``max_die_area_mm2`` >
    a shuttle-default (10 mm^2) when a shuttle is named > None (no cap).
    Returns ``(mm2_or_None, source)``; ``source`` is one of ``env`` / ``prd`` /
    ``shuttle_default`` / ``none``.
    """
    env = die_budget_env_mm2()
    if env is not None:
        return env, "env"
    prd_v = _prd_die_budget_mm2(prd)
    if prd_v is not None:
        return prd_v, "prd"
    if mentions_shuttle(requirements, ers_technology_text):
        return DEFAULT_SHUTTLE_DIE_MM2, "shuttle_default"
    return None, "none"


# ---------------------------------------------------------------------------
# Die rollup (estimate-time + measured-time)
# ---------------------------------------------------------------------------

@dataclass
class RollupItem:
    name: str
    area_um2: float
    source: str = ""           # "area_budget" | "estimated_gates" | "measured" | ...


@dataclass
class DieRollupVerdict:
    ok: bool
    die_budget_mm2: float | None
    budget_source: str
    subtotal_um2: float
    margin: float
    total_um2: float
    items: list[RollupItem] = field(default_factory=list)
    reason: str = ""

    @property
    def has_cap(self) -> bool:
        return self.die_budget_mm2 is not None


def _rollup_table(items: list[RollupItem]) -> str:
    rows = [f"  {'block/element':<32} {'area (mm^2)':>12}  source"]
    for it in sorted(items, key=lambda x: -x.area_um2):
        rows.append(f"  {it.name[:32]:<32} {it.area_um2 / 1e6:>12.3f}  {it.source}")
    return "\n".join(rows)


def evaluate_die_rollup(items: list[RollupItem], *,
                        die_budget_mm2: float | None,
                        budget_source: str = "",
                        margin: float | None = None) -> DieRollupVerdict:
    """Sum per-item area + interconnect margin and compare to the die cap.

    No cap -> ``ok`` (never blocks) but ``has_cap`` False so the caller can LOG
    loudly that the design is un-capped. Over cap -> ``ok=False`` with an
    itemized rollup table in ``reason``.
    """
    if margin is not None:
        m = margin
    elif len([it for it in items if it.area_um2 > 0]) <= 1:
        # one (real) block on the die -> no inter-block routing to budget for.
        m = single_block_margin()
    else:
        m = interconnect_margin()
    subtotal = sum(max(0.0, it.area_um2) for it in items)
    total = subtotal * (1.0 + m)
    if die_budget_mm2 is None:
        return DieRollupVerdict(
            ok=True, die_budget_mm2=None, budget_source=budget_source or "none",
            subtotal_um2=subtotal, margin=m, total_um2=total, items=list(items),
            reason="",
        )
    cap_um2 = die_budget_mm2 * 1e6
    ok = total <= cap_um2
    reason = ""
    if not ok:
        reason = (
            f"die-area rollup {total / 1e6:.3f} mm^2 "
            f"(Σ blocks {subtotal / 1e6:.3f} mm^2 + {m * 100:.0f}% interconnect) "
            f"exceeds the {die_budget_mm2:.3f} mm^2 die budget "
            f"[{budget_source or 'die_budget'}]. The design does not fit its "
            f"target die. Itemized rollup:\n" + _rollup_table(items) +
            "\nReduce the largest contributors (usually an oversized storage "
            "element priced as macro area) or raise the die budget deliberately."
        )
    return DieRollupVerdict(
        ok=ok, die_budget_mm2=die_budget_mm2, budget_source=budget_source or "die_budget",
        subtotal_um2=subtotal, margin=m, total_um2=total, items=list(items),
        reason=reason,
    )


def block_area_estimate_um2(block: dict) -> tuple[float, str]:
    """Per-block area estimate for the arch-time rollup (no specs yet).

    Prefers a declared ``area_budget_um2`` (which INCLUDES macro area by
    convention); else derives from ``estimated_gates`` at the shuttle checker's
    gate density; else 0. Returns ``(um2, source)``.
    """
    ab = block.get("area_budget_um2")
    try:
        if ab is not None and float(ab) > 0:
            return float(ab), "area_budget_um2"
    except (TypeError, ValueError):
        pass
    g = block.get("estimated_gates")
    try:
        if g is not None and float(g) > 0:
            return float(g) / _GATE_DENSITY_PER_MM2 * 1e6, "estimated_gates"
    except (TypeError, ValueError):
        pass
    return 0.0, "unknown"


def read_ledger(project_root: str, block: str) -> dict[str, Any] | None:
    """Load a block's persisted ``mem_price.json`` ledger dict, or None."""
    try:
        p = Path(project_root) / ".coresmith" / "blocks" / block / "mem_price.json"
        if not p.exists():
            return None
        data = json.loads(p.read_text())
        return data if isinstance(data, dict) else None
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def read_block_ledger_area_um2(project_root: str, block: str) -> float | None:
    """Total priced-memory area from a block's persisted mem_price ledger, or None."""
    data = read_ledger(project_root, block)
    if not data:
        return None
    try:
        v = data.get("total_area_um2")
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def deferred_over_budget_blocks(project_root: str,
                                block_names: list[str]) -> list[dict[str, Any]]:
    """Blocks whose mem-price ledger records an over-budget DEFER, for downstream.

    The bounded revise loop accepts an over-budget spec after its cap (or on a
    byte-identical re-submission) so the pipeline never deadlocks -- but that
    excess must stay VISIBLE. This scans the persisted ledgers and returns a
    machine-readable summary (name, total/budget mm^2, over-by-x, reason, rounds)
    for every block flagged ``over_budget`` (and/or ``deferred``), so the die
    rollup and the integration review surface the deferred excess instead of
    silently shipping it. Never raises.
    """
    out: list[dict[str, Any]] = []
    for name in block_names or []:
        led = read_ledger(project_root, name)
        if not led:
            continue
        if not (led.get("over_budget") or led.get("deferred")):
            continue
        budget = led.get("area_budget_um2")
        total = led.get("total_area_um2")
        if total is None and led.get("total_area_mm2") is not None:
            try:
                total = float(led["total_area_mm2"]) * 1e6
            except (TypeError, ValueError):
                total = None
        try:
            over_x = (float(total) / float(budget)) if (budget and total) else None
        except (TypeError, ValueError, ZeroDivisionError):
            over_x = None
        out.append({
            "block": name,
            "total_area_mm2": led.get("total_area_mm2"),
            "area_budget_mm2": (float(budget) / 1e6) if budget else None,
            "over_budget_x": round(over_x, 2) if over_x is not None else None,
            "deferred": bool(led.get("deferred")),
            "deferred_reason": led.get("deferred_reason", ""),
            "reject_rounds": led.get("reject_rounds"),
            "reasons": led.get("reasons", []),
        })
    return out


def arch_rollup_items(block_diagram: dict, project_root: str = "") -> list[RollupItem]:
    """Build estimate-time rollup items from the block diagram (+ any ledgers).

    Each block contributes ``max(area_budget/gate estimate, priced memory from
    its ledger if one already exists)`` -- so a block whose declared budget
    under-prices its memory still rolls up at the priced-memory area (no
    double-count with the budget, which already includes macro area).
    """
    items: list[RollupItem] = []
    for b in (block_diagram or {}).get("blocks", []):
        if not isinstance(b, dict):
            continue
        name = str(b.get("name", "block"))
        est, src = block_area_estimate_um2(b)
        led = read_block_ledger_area_um2(project_root, name) if project_root else None
        if led is not None and led > est:
            items.append(RollupItem(name=name, area_um2=led, source="mem_ledger"))
        elif est > 0:
            items.append(RollupItem(name=name, area_um2=est, source=src))
    return items
