# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Verilator coverage helpers (opt-in, default OFF).

Coverage is injected into the ``run_simulation`` Makefile builder only when
``CORESMITH_COVERAGE=1`` (per-invocation agent opt-in). These helpers annotate
the resulting ``coverage.dat`` and summarize uncovered points for
``coresmith coverage <block> --uncovered``.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional


def coverage_enabled() -> bool:
    return os.environ.get("CORESMITH_COVERAGE", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def line_cov_gate_enabled() -> bool:
    """Line-coverage floor gate on block DV (default ON).

    A REJECT-only gate: a block-DV pass whose testbench exercises less than
    ``line_cov_floor()`` percent of the block's coverage points is demoted to
    a failure with the uncovered regions as feedback (weak-TB rejector). A
    coverage number ABOVE the floor proves nothing (execution is not
    observation -- see the DV-closure masking-band analysis) and is never
    treated as evidence of correctness. ``CORESMITH_LINE_COV_GATE=0`` disables.
    """
    return os.environ.get(
        "CORESMITH_LINE_COV_GATE", "1"
    ).strip().lower() not in {"0", "false", "no", "off"}


def line_cov_floor() -> float:
    """Coverage floor percentage (``CORESMITH_LINE_COV_FLOOR``, default 70)."""
    try:
        return float(os.environ.get("CORESMITH_LINE_COV_FLOOR", "70") or 70)
    except ValueError:
        return 70.0


def line_cov_gate_verdict(sim_dir: str | Path) -> Optional[dict[str, Any]]:
    """Evaluate the line-coverage floor for one block-DV sim directory.

    Returns ``None`` when the gate is NOT APPLICABLE (gate disabled, no
    ``coverage.dat``, ``verilator_coverage`` missing, or zero coverage
    points) -- the caller must then treat DV's own verdict as final, never
    fail a block for missing tooling. Otherwise returns::

        {"passed": bool, "pct": float, "floor": float, "points_total": int,
         "points_hit": int, "uncovered_count": int, "report": str}

    ``report`` carries the uncovered-region list formatted as actionable
    feedback for the testbench-fix/regeneration loop ("you never exercised
    these lines -- add stimulus for them"), the same enrichment pattern as
    the composition gate's signature appendix. Never raises.
    """
    try:
        if not line_cov_gate_enabled():
            return None
        annotated = annotate(sim_dir)
        if annotated is None:
            return None
        s = summarize(annotated)
        pct = s.get("pct")
        if pct is None or not s.get("points_total"):
            return None
        floor = line_cov_floor()
        passed = float(pct) >= floor
        _hit = int(s.get("points_hit") or 0)
        _total = int(s["points_total"])
        lines = [
            f"LINE-COVERAGE GATE: testbench exercises {pct}% of "
            f"{s['points_total']} coverage points (floor {floor:g}%): "
            + ("PASS" if passed else "FAIL"),
        ]
        if not passed:
            lines.append(
                "TESTBENCH TOO WEAK: the DV tests passed, but they never "
                "execute the regions below. This is not an RTL bug -- "
                "STRENGTHEN THE TESTBENCH: add stimulus that drives every "
                "listed region (un-exercised modes/branches, FSM arms, "
                "wrap-around and stall/backpressure paths), keeping expected "
                "outputs derived from the golden model."
            )
            for u in s.get("uncovered", [])[:60]:
                lines.append(
                    f"  uncovered {u.get('file')}:{u.get('line')}: "
                    f"{str(u.get('text', ''))[:120]}"
                )
            more = len(s.get("uncovered", [])) - 60
            if more > 0:
                lines.append(f"  ... ({more} more uncovered points)")
        return {
            "passed": passed,
            "pct": float(pct),
            "floor": floor,
            "points_total": _total,
            "points_hit": _hit,
            "uncovered_count": _total - _hit,
            "report": "\n".join(lines)[:6000],
        }
    except Exception:  # noqa: BLE001 - the gate must never crash a DV run
        return None


def coverage_na_reason(sim_dir: str | Path) -> str:
    """Explain WHY the line-coverage gate did not apply (never raises).

    Used to PERSIST a visible, non-blank reason in the per-block report when
    ``line_cov_gate_verdict`` returns ``None`` -- so a run with no coverage is
    auditable ("no coverage.dat produced") instead of silently dropped.
    """
    try:
        if not line_cov_gate_enabled():
            return "line-coverage gate disabled (CORESMITH_LINE_COV_GATE=0)"
        if find_coverage_dat(sim_dir) is None:
            return "no coverage.dat produced by the sim"
        if not shutil.which("verilator_coverage"):
            return "verilator_coverage not on PATH"
        return "no coverage points instrumented / annotate produced no output"
    except Exception:  # noqa: BLE001
        return "coverage unavailable (evaluation error)"


def coverage_record(sim_dir: str | Path) -> dict[str, Any]:
    """ALWAYS-return persistence record for the signoff report (never raises).

    Unlike ``line_cov_gate_verdict`` (which returns ``None`` when the gate does
    not apply), this returns a dict that is either::

        {"applicable": True, "pct", "floor", "points_total", "points_hit",
         "uncovered_count", "passed"}

    or, when coverage could not be measured::

        {"applicable": False, "reason": "<why>"}

    so a run WITHOUT coverage is visible in the final report rather than being
    silently blanked. Missing tooling / no coverage.dat is ``applicable:False``
    -- it NEVER fails a block.
    """
    v = line_cov_gate_verdict(sim_dir)
    if v is None:
        return {"applicable": False, "reason": coverage_na_reason(sim_dir)}
    return {
        "applicable": True,
        "pct": v.get("pct"),
        "floor": v.get("floor"),
        "points_total": v.get("points_total"),
        "points_hit": v.get("points_hit"),
        "uncovered_count": v.get("uncovered_count"),
        "passed": v.get("passed"),
    }


def find_coverage_dat(sim_dir: str | Path) -> Optional[Path]:
    sim = Path(sim_dir)
    for name in ("coverage.dat", "logs/coverage.dat"):
        cand = sim / name
        if cand.exists():
            return cand
    hits = list(sim.rglob("coverage.dat"))
    return hits[0] if hits else None


def annotate(sim_dir: str | Path, *, timeout_s: int = 120) -> Optional[Path]:
    """Run ``verilator_coverage --annotate`` on the sim dir's coverage.dat.

    Returns the annotated-output directory, or ``None`` when the tool or the
    ``coverage.dat`` are absent (never raises).
    """
    dat = find_coverage_dat(sim_dir)
    if dat is None:
        return None
    vcov = shutil.which("verilator_coverage")
    if not vcov:
        return None
    out_dir = Path(sim_dir) / "coverage_annotated"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        # Run FROM sim_dir so the RELATIVE source paths recorded in coverage.dat
        # (e.g. ``adder.v``) resolve -- verilator_coverage --annotate reads each
        # source file to emit the hit-count-prefixed copy, and it looks for them
        # relative to its CWD. Without cwd=sim_dir the daemon's CWD is used, the
        # sources are not found, no annotated files are written, and the whole
        # coverage gate silently no-ops (verdict None) even though a real
        # coverage.dat exists. dat/out_dir are absolute so cwd only affects the
        # source lookup.
        subprocess.run(
            [vcov, "--annotate", str(out_dir), str(dat)],
            capture_output=True, text=True, timeout=timeout_s,
            cwd=str(Path(sim_dir)),
        )
    except Exception:  # noqa: BLE001
        return None
    return out_dir if any(out_dir.iterdir()) else None


# verilator_coverage --annotate prefixes each instrumented source line with a
# hit count; an UNCOVERED point is rendered with a ``%000000`` marker.
_UNCOV_RE = re.compile(r"^\s*%0+\s")
_COV_RE = re.compile(r"^\s*(%?\d+)\s")


def summarize(annotated_dir: str | Path, cap: int = 200) -> dict[str, Any]:
    """Summarize an annotated coverage tree.

    Returns ``{points_total, points_hit, pct, uncovered}`` where ``uncovered``
    is a capped list of ``{"file", "line", "text"}`` for the un-hit points.
    """
    root = Path(annotated_dir)
    total = 0
    hit = 0
    uncovered: list[dict] = []
    if not root.exists():
        return {"points_total": 0, "points_hit": 0, "pct": None, "uncovered": []}
    for f in sorted(root.rglob("*")):
        if not f.is_file():
            continue
        try:
            lines = f.read_text(errors="replace").splitlines()
        except Exception:  # noqa: BLE001
            continue
        for i, ln in enumerate(lines, 1):
            if not _COV_RE.match(ln):
                continue
            total += 1
            if _UNCOV_RE.match(ln):
                if len(uncovered) < cap:
                    uncovered.append({
                        "file": f.name, "line": i, "text": ln.strip()[:200],
                    })
            else:
                hit += 1
    pct = round(100.0 * hit / total, 2) if total else None
    return {
        "points_total": total, "points_hit": hit, "pct": pct,
        "uncovered": uncovered,
    }
