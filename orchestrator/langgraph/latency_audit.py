# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Latency audit -- ground the model's cycle count in per-stage arithmetic.

The block-golden Amaranth model declares a machine-readable ``STAGE_BUDGET``: for
each pipeline stage, the arithmetic that happens in ONE cycle of that stage, how
many cycles the stage runs, and how many times it is invoked. It also declares
``DECLARED_LATENCY_CYCLES`` (the top-level number, e.g. 280).

This module makes that number *checkable* instead of magic:

1. **Reconcile** -- sum(latency_cycles x iters) over stages must match the
   declared total (within a control-overhead tolerance). The 280 must add up.
2. **Price each stage** -- the chained delay of a stage's per-cycle op list,
   from the characterized PDK arithmetic model (``arith_characterize``), must
   fit one clock period. A stage whose single-cycle op chain exceeds the period
   is a combinational cloud in waiting -- caught here, at model-authoring time,
   one step BEFORE the synth gate ever runs.

It also renders the budget as a **pipeline stage map** handed to the RTL
generator: each named stage -> a registered boundary with a known per-cycle op
budget. This is the structural contract whose absence let the codec RD-search
collapse into one ``always @(*)`` cloud.

Gated by ``CORESMITH_LATENCY_AUDIT`` (default off); best-effort and fail-open --
a parse failure or an uncharacterized PDK never blocks the pipeline.
"""
from __future__ import annotations

import ast
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .arith_characterize import predict_op_delay as _predict

# op token in a STAGE_BUDGET ops list, e.g. "mul16", "add32", "sad16"
_OP_RE = re.compile(r"^([a-z]+)(\d+)$")
_KNOWN_OPS = {"add", "sub", "mul", "cmp", "mux", "shift", "sad",
              # crypto/codec primitives now priced by the delay model
              "lut", "gfmul", "xortree"}


def audit_enabled() -> bool:
    """True iff CORESMITH_LATENCY_AUDIT is set truthy (default off; strict profile seeds it)."""
    from orchestrator.profile import ensure_applied, flag_enabled
    ensure_applied()
    return flag_enabled("CORESMITH_LATENCY_AUDIT", default=False)


@dataclass
class StageAudit:
    name: str
    latency_cycles: int
    iters: int
    ops: list[str]
    rationale: str = ""
    per_cycle_delay_ns: Optional[float] = None  # None -> uncharacterized
    fits_period: Optional[bool] = None          # None -> couldn't price
    uncharacterized_ops: list[str] = field(default_factory=list)

    @property
    def contributed_cycles(self) -> int:
        return int(self.latency_cycles) * int(self.iters)


@dataclass
class AuditReport:
    stages: list[StageAudit]
    declared_latency: Optional[int]
    summed_latency: int
    period_ns: float
    reconciles: bool
    infeasible: list[str] = field(default_factory=list)   # stages whose 1-cyc chain > period
    parsed: bool = True
    note: str = ""

    @property
    def ok(self) -> bool:
        # OK = parsed, no stage over a period, and the total reconciles.
        return self.parsed and not self.infeasible and self.reconciles


def _parse_op(token: str) -> Optional[tuple[str, int]]:
    m = _OP_RE.match(str(token).strip())
    if not m:
        return None
    op, w = m.group(1), int(m.group(2))
    if op not in _KNOWN_OPS:
        return None
    return op, w


def _eval_const_int(node: ast.AST) -> Optional[int]:
    """Safely evaluate a constant-integer arithmetic AST node.

    The block-golden prompt teaches ``DECLARED_LATENCY_CYCLES = 10 + 9*5 + 9*16``
    -- an *expression*, which ``ast.literal_eval`` rejects (it only takes
    literals), so the declared total silently parsed as None and the reconcile
    guard no-op'd. This evaluates int literals + a whitelist of arithmetic
    (+, -, *, //, %, unary +/-) with NO name/call/attribute access -- never
    executes code. Returns None for anything outside that grammar.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return int(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        v = _eval_const_int(node.operand)
        if v is None:
            return None
        return v if isinstance(node.op, ast.UAdd) else -v
    if isinstance(node, ast.BinOp) and isinstance(
        node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod)
    ):
        left = _eval_const_int(node.left)
        right = _eval_const_int(node.right)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, (ast.FloorDiv, ast.Mod)):
            if right == 0:
                return None
            return left // right if isinstance(node.op, ast.FloorDiv) else left % right
    return None


def parse_stage_budget(source: str) -> tuple[list[dict], Optional[int]]:
    """Extract STAGE_BUDGET (list[dict]) and DECLARED_LATENCY_CYCLES (int).

    Uses AST + literal_eval so arbitrary code in the model never executes.
    Returns ([], None) if neither is present / parseable.
    """
    stages: list[dict] = []
    declared: Optional[int] = None
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [], None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if "STAGE_BUDGET" in targets:
            try:
                val = ast.literal_eval(node.value)
                if isinstance(val, (list, tuple)):
                    stages = [dict(s) for s in val if isinstance(s, dict)]
            except (ValueError, SyntaxError):
                pass
        if "DECLARED_LATENCY_CYCLES" in targets:
            # NOT literal_eval -- the prompt teaches an expression form
            # (10 + 9*5 + 9*16). Safe constant-arithmetic eval instead.
            v = _eval_const_int(node.value)
            if v is not None:
                declared = v
    return stages, declared


def audit_source(source: str, target_clock_mhz: float = 50.0,
                 pdk: Optional[dict] = None) -> AuditReport:
    """Parse + price + reconcile a model's STAGE_BUDGET."""
    period = 1000.0 / max(1e-6, target_clock_mhz)
    raw_stages, declared = parse_stage_budget(source)
    if not raw_stages:
        return AuditReport(
            stages=[], declared_latency=declared, summed_latency=0,
            period_ns=period, reconciles=True, parsed=False,
            note="no STAGE_BUDGET declared",
        )

    audits: list[StageAudit] = []
    infeasible: list[str] = []
    for s in raw_stages:
        name = str(s.get("name", "?"))
        lat = int(s.get("latency_cycles", 1) or 1)
        iters = int(s.get("iters", 1) or 1)
        ops = [str(o) for o in (s.get("ops") or [])]
        rationale = str(s.get("rationale", ""))

        chain = 0.0
        priced_any = False
        unchar: list[str] = []
        for tok in ops:
            parsed = _parse_op(tok)
            if parsed is None:
                unchar.append(tok)
                continue
            op, w = parsed
            d = _predict(op, w, pdk)
            if d is None:
                unchar.append(tok)
                continue
            chain += d
            priced_any = True

        per_cycle = chain if priced_any else None
        fits: Optional[bool] = None
        if per_cycle is not None:
            fits = per_cycle <= period + 1e-9
            if not fits:
                infeasible.append(name)

        audits.append(StageAudit(
            name=name, latency_cycles=lat, iters=iters, ops=ops,
            rationale=rationale, per_cycle_delay_ns=per_cycle,
            fits_period=fits, uncharacterized_ops=unchar,
        ))

    summed = sum(a.contributed_cycles for a in audits)
    if declared is None:
        reconciles = True  # nothing to reconcile against
    else:
        tol = max(2.0, 0.10 * declared)  # 10% control/handshake overhead
        reconciles = abs(summed - declared) <= tol

    return AuditReport(
        stages=audits, declared_latency=declared, summed_latency=summed,
        period_ns=period, reconciles=reconciles, infeasible=infeasible,
        parsed=True,
    )


def find_block_model(project_root: str | os.PathLike,
                     block_name: str) -> Optional[Path]:
    """Locate the Amaranth block model file for ``block_name``.

    Block models live under ``arch/block_models/``; names vary
    (``<block>.py``, ``<block>_stage.py``, ...). Match by best overlap.
    """
    bm = Path(project_root) / "arch" / "block_models"
    if not bm.is_dir():
        return None
    cands = sorted(p for p in bm.glob("*.py") if p.name != "_chip_model.py")
    # exact, then prefix, then token-overlap
    for p in cands:
        if p.stem == block_name:
            return p
    for p in cands:
        if p.stem.startswith(block_name) or block_name.startswith(p.stem):
            return p
    btoks = set(block_name.split("_"))
    best, best_score = None, 0
    for p in cands:
        score = len(btoks & set(p.stem.split("_")))
        if score > best_score:
            best, best_score = p, score
    return best if best_score >= 2 else None


def format_stage_map(report: AuditReport, title: str = "datapath") -> str:
    """Render the audited budget as a per-stage contract for the RTL prompt."""
    if not report.parsed or not report.stages:
        return ""
    lines = [
        f"# PIPELINE STAGE MAP for {title} "
        f"(period {report.period_ns:.1f} ns) -- the Amaranth model's audited "
        "per-stage budget. Realize EACH stage as a REGISTERED boundary "
        "(always @(posedge clk)); the per-cycle ops below must fit one period.",
    ]
    for s in report.stages:
        delay = (f"{s.per_cycle_delay_ns:.2f} ns/cyc"
                 if s.per_cycle_delay_ns is not None else "uncharacterized")
        flag = "" if s.fits_period in (True, None) else "  <-- EXCEEDS PERIOD: split this stage"
        ops = ", ".join(s.ops) if s.ops else "(no arithmetic)"
        lines.append(
            f"  - {s.name}: {s.latency_cycles} cyc x {s.iters} = "
            f"{s.contributed_cycles} cyc; per-cycle ops [{ops}] -> {delay}{flag}"
        )
    if report.declared_latency is not None:
        lines.append(
            f"  total: stages sum to {report.summed_latency} cyc vs declared "
            f"{report.declared_latency} cyc "
            f"({'reconciles' if report.reconciles else 'MISMATCH -- fix the budget'})."
        )
    if report.infeasible:
        lines.append(
            f"  ! {len(report.infeasible)} stage(s) exceed one period "
            f"({', '.join(report.infeasible[:6])}): these MUST be split into "
            "more registered stages or sequentialized -- do NOT emit them as a "
            "single combinational block."
        )
    return "\n".join(lines)


_CLOCK_RE = re.compile(
    r"target[ _]clock[:\s*]*\**\s*([0-9]+(?:\.[0-9]+)?)\s*MHz", re.IGNORECASE
)


def resolve_target_clock_mhz(project_root: str | os.PathLike,
                             default: float = 50.0) -> float:
    """Best-effort per-run target clock (MHz) for budget / stage-map pricing.

    The pricing was previously hardcoded to 50 MHz; on a 100 MHz run that
    under-prices (errs lenient). Resolution order: a structured constraints
    clock, then the ERS/PRD "Target clock: N MHz" line, then ``default``.
    Always returns a positive float; never raises.
    """
    try:
        root = Path(project_root)
        for cj in (root / ".coresmith" / "constraints.json",
                   root / ".coresmith" / "chip_constraints.json"):
            if cj.exists():
                import json as _json
                d = _json.loads(cj.read_text())
                for k in ("target_clock_mhz", "clock_mhz", "target_clock",
                          "clock_freq_mhz", "max_clock_mhz"):
                    v = d.get(k)
                    if isinstance(v, (int, float)) and v > 0:
                        return float(v)
        for doc in (root / "arch" / "ers_spec.md",
                    root / "arch" / "prd_spec.md"):
            if doc.exists():
                m = _CLOCK_RE.search(doc.read_text(encoding="utf-8"))
                if m:
                    val = float(m.group(1))
                    if val > 0:
                        return val
    except Exception:  # noqa: BLE001 - best-effort, never block
        pass
    return float(default)


def stage_map_fragment(project_root: str | os.PathLike, block_name: str,
                       target_clock_mhz: Optional[float] = None,
                       pdk: Optional[dict] = None) -> str:
    """Locate the block model, audit it, and render its stage map (or '').

    ``target_clock_mhz=None`` resolves the run's real clock from the run dir
    (was hardcoded 50 MHz). Best-effort: any failure returns '' so RTL
    generation is never blocked.
    """
    try:
        path = find_block_model(project_root, block_name)
        if path is None:
            return ""
        mhz = (target_clock_mhz if target_clock_mhz is not None
               else resolve_target_clock_mhz(project_root))
        report = audit_source(path.read_text(encoding="utf-8"),
                              target_clock_mhz=mhz, pdk=pdk)
        return format_stage_map(report, title=block_name)
    except Exception:  # noqa: BLE001 - never block RTL generation
        return ""


__all__ = [
    "audit_enabled", "StageAudit", "AuditReport",
    "parse_stage_budget", "audit_source", "find_block_model",
    "format_stage_map", "stage_map_fragment", "resolve_target_clock_mhz",
]
