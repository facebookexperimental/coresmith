# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Deterministic signoff-scorecard aggregation for the final-report node.

This module is PURE aggregation of RECORDED facts -- no LLM, no tool calls. It
reads the run's persisted verification/PPA/coverage records (the ``Scoreboard``
sqlite db + per-block ``coverage.json`` + the run state) and emits a single
signoff scorecard as a JSON dict (``build_final_report``) and a human-readable
markdown document (``render_markdown``).

Kept out of ``pipeline_graph.py`` so it is unit-testable without the LangGraph
runtime: a test feeds a fake state + fake scoreboard and asserts the report
carries the block coverage %, Fmax, testbench names, PPA and DV verdicts.

Everything here is best-effort: a missing db / key yields a "n/a" cell, never an
exception. The report never GATES anything -- it is a record of what happened.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "coresmith.final_report/v1"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _num(x: Any) -> float | None:
    try:
        if x is None:
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def _int(x: Any) -> int | None:
    n = _num(x)
    return int(n) if n is not None else None


def fmax_mhz(wns_ns: float | None, period_ns: float | None
             ) -> float | None:
    """Achieved Fmax (MHz) from pre-layout WNS and the clock period.

    ``achieved_period = period - WNS`` (WNS<0 => slower than target); Fmax =
    1000/achieved_period. Returns ``None`` when WNS/period are unknown or the
    achieved period is non-positive (unphysical -> report as n/a).
    """
    w = _num(wns_ns)
    p = _num(period_ns)
    if w is None or p is None or p <= 0:
        return None
    achieved = p - w
    if achieved <= 0:
        return None
    return round(1000.0 / achieved, 2)


def _period_ns(target_clock_mhz: float | None) -> float | None:
    f = _num(target_clock_mhz)
    if f is None or f <= 0:
        return None
    return 1000.0 / f


def _dv_one(sb: Any, block: str, scope: str) -> dict | None:
    """Latest single dv_results row for (block, scope), or None."""
    if sb is None:
        return None
    try:
        rows = sb.latest_dv(block, scope) or []
    except Exception:  # noqa: BLE001
        return None
    return rows[0] if rows else None


def _read_json(path: Path) -> dict | None:
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return None
    return None


def _carried_forward_defects(project_root: str) -> list:
    """Read the carried-forward-defects ledger (advisory-bypass observations)."""
    try:
        p = Path(project_root) / ".coresmith" / "carried_forward_defects.json"
        if p.exists():
            data = json.loads(p.read_text())
            if isinstance(data, list):
                return data
    except Exception:  # noqa: BLE001
        pass
    return []


def _chip_throughput(project_root: str) -> dict:
    """Chip-level measured-throughput record (v3): read the persisted
    ``.coresmith/chip_throughput.json`` (from the deterministic-BFM integration
    gate) or an n/a-visible default."""
    try:
        from orchestrator.langgraph.throughput_gate import read_chip_throughput
        rec = read_chip_throughput(project_root)
        if isinstance(rec, dict):
            return rec
    except Exception:  # noqa: BLE001
        pass
    return {"scope": "chip", "applicable": False,
            "measured_cyc_per_op_chip": None,
            "reason": "no chip throughput measured"}


def _engine_provenance(project_root: str) -> dict:
    """Engine git SHA stamped at run start (+ mid-run-change flag). Section 7a.

    Prefers the run-stamped ``.coresmith/engine_sha.json`` (what the run ACTUALLY
    executed); falls back to the live engine SHA when no stamp exists.
    """
    try:
        p = Path(project_root) / ".coresmith" / "engine_sha.json"
        if p.exists():
            d = json.loads(p.read_text())
            if isinstance(d, dict):
                return d
    except Exception:  # noqa: BLE001
        pass
    try:
        from orchestrator.utils import engine_git_sha
        return {"sha": engine_git_sha(), "changed": False}
    except Exception:  # noqa: BLE001
        return {"sha": "", "changed": False}


# ---------------------------------------------------------------------------
# Per-block aggregation
# ---------------------------------------------------------------------------

def _block_coverage(project_root: str, block: str, sb: Any) -> dict:
    """Coverage fact for a block: block_dir/coverage.json first (has the full
    ``applicable``/reason shape), else the scoreboard row, else n/a-visible."""
    cov = _read_json(
        Path(project_root) / ".coresmith" / "blocks" / block / "coverage.json"
    )
    if isinstance(cov, dict) and ("applicable" in cov):
        return cov
    # Fall back to the scoreboard coverage_results row.
    row = None
    if sb is not None:
        try:
            row = sb.coverage_latest(block)
        except Exception:  # noqa: BLE001
            row = None
    if row:
        extra = {}
        try:
            extra = json.loads(row.get("uncovered") or "{}") or {}
        except Exception:  # noqa: BLE001
            extra = {}
        if not isinstance(extra, dict):
            extra = {}
        if extra.get("applicable") is False:
            return {"applicable": False,
                    "reason": extra.get("reason", "not measured")}
        pct = _num(row.get("pct"))
        total = _int(row.get("points_total"))
        hit = _int(row.get("points_hit"))
        return {
            "applicable": pct is not None and bool(total),
            "pct": pct,
            "floor": _num(extra.get("floor")),
            "points_total": total,
            "points_hit": hit,
            "uncovered_count": extra.get("uncovered_count"),
            "passed": extra.get("passed"),
        }
    return {"applicable": False, "reason": "no coverage recorded"}


def _block_throughput(project_root: str, block: str) -> dict:
    """Measured-throughput fact for a block: block_dir/throughput.json (the
    ``evaluate_block_throughput`` record) or an n/a-visible default (v3)."""
    rec = _read_json(
        Path(project_root) / ".coresmith" / "blocks" / block / "throughput.json"
    )
    if isinstance(rec, dict) and ("applicable" in rec):
        return rec
    return {"applicable": False, "reason": "no throughput recorded"}


def _block_ppa(sb: Any, block: str, period_ns: float | None) -> dict:
    row = None
    if sb is not None:
        try:
            row = sb.latest_ppa(block)
        except Exception:  # noqa: BLE001
            row = None
    if not row:
        return {"measured": False}
    wns = _num(row.get("wns_ns"))
    return {
        "measured": True,
        "cells": _int(row.get("cells")),
        "ff": _int(row.get("ff")),
        "mem_bits": _int(row.get("mem_bits")),
        "area_um2": _num(row.get("area_um2")),
        "wns_ns": wns,
        "fmax_mhz": fmax_mhz(wns, period_ns),
        "budget_ff": _int(row.get("budget_ff")),
        "budget_area_um2": _num(row.get("budget_area_um2")),
        "ppa_ok": (None if row.get("ppa_ok") is None
                   else bool(row.get("ppa_ok"))),
        "elaborated": (None if row.get("elaborated") is None
                       else bool(row.get("elaborated"))),
        "probe": row.get("probe", ""),
        "report_path": row.get("report_path", ""),
    }


def _block_testbenches(project_root: str, block: str, spec: dict,
                       dv_row: dict | None, dv_pass: bool | None
                       ) -> list[dict]:
    """Enumerate the testbenches that ran for a block, by name.

    - the block-DV cocotb TB (block spec ``testbench``) with #testcases + verdict
    - any branch-parity / smoke TB recorded in ``block_dir/dv_summary.json``
    """
    tbs: list[dict] = []
    tb_name = (spec or {}).get("testbench") or f"tb/cocotb/test_{block}.py"
    tp = _int((dv_row or {}).get("tests_passed"))
    tt = _int((dv_row or {}).get("tests_total"))
    tbs.append({
        "name": Path(tb_name).name,
        "path": tb_name,
        "kind": "block_dv",
        "tests_passed": tp,
        "tests_total": tt,
        "passed": (bool(dv_row.get("passed")) if dv_row is not None
                   else dv_pass),
    })
    # Optional extra TBs (branch-parity / smoke) if the node recorded them.
    # The primary block-DV TB is already added above from the spec, so only pull
    # the NON-block_dv entries here to avoid listing it twice.
    extra = _read_json(
        Path(project_root) / ".coresmith" / "blocks" / block / "dv_summary.json"
    )
    if isinstance(extra, dict):
        for t in extra.get("testbenches", []) or []:
            if (isinstance(t, dict) and t.get("name")
                    and t.get("kind") != "block_dv"):
                tbs.append(t)
    return tbs


def _block_hot_patches(project_root: str, block: str) -> list:
    hp = _read_json(
        Path(project_root) / ".coresmith" / "blocks" / block / "hot_patches.json"
    )
    if isinstance(hp, list):
        return hp
    if isinstance(hp, dict):
        return hp.get("patches", []) or []
    return []


def _block_names(state: dict) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for src in (state.get("block_queue") or [], state.get("completed_blocks") or []):
        for b in src:
            n = (b or {}).get("name")
            if n and n not in seen:
                seen.add(n)
                names.append(n)
    return names


def _spec_for(state: dict, block: str) -> dict:
    for b in state.get("block_queue") or []:
        if (b or {}).get("name") == block:
            return b
    return {}


def _completed_for(state: dict, block: str) -> dict:
    # last-wins: a later PASS entry overrides an earlier FAIL for the same name
    found: dict = {}
    for b in state.get("completed_blocks") or []:
        if (b or {}).get("name") == block:
            found = b
    return found


# ---------------------------------------------------------------------------
# Report build
# ---------------------------------------------------------------------------

def build_final_report(state: dict, project_root: str, *,
                       scoreboard: Any = None,
                       now: float | None = None) -> dict:
    """Aggregate the run's recorded facts into a signoff-scorecard dict.

    ``scoreboard`` is a ``Scoreboard`` instance (duck-typed: ``latest_dv``,
    ``latest_ppa``, ``coverage_latest``). When omitted, one is opened on
    ``project_root`` best-effort. ``now`` overrides the timestamp for tests.
    """
    if scoreboard is None:
        try:
            from orchestrator.state_store.store import Scoreboard
            scoreboard = Scoreboard(project_root)
        except Exception:  # noqa: BLE001
            scoreboard = None
    sb = scoreboard

    target_clock_mhz = _num(state.get("target_clock_mhz")) or 50.0
    period_ns = _period_ns(target_clock_mhz)

    blocks_out: list[dict] = []
    tb_total = 0
    cov_hit_sum = 0
    cov_total_sum = 0
    cov_pcts: list[float] = []
    fmax_vals: list[float] = []
    wns_vals: list[float] = []
    area_sum = 0.0
    area_seen = False

    for name in _block_names(state):
        spec = _spec_for(state, name)
        comp = _completed_for(state, name)
        dv_row = _dv_one(sb, name, "rtl")
        # DV verdict: scoreboard row is authoritative; else completed_blocks.
        if dv_row is not None and dv_row.get("passed") is not None:
            dv_pass = bool(dv_row.get("passed"))
        elif comp:
            dv_pass = bool(comp.get("success") or comp.get("sim_passed"))
        else:
            dv_pass = None

        cov = _block_coverage(project_root, name, sb)
        if cov.get("applicable"):
            p = _num(cov.get("pct"))
            if p is not None:
                cov_pcts.append(p)
            t = _int(cov.get("points_total"))
            h = _int(cov.get("points_hit"))
            if t:
                cov_total_sum += t
                cov_hit_sum += (h or 0)

        ppa = _block_ppa(sb, name, period_ns)
        if ppa.get("measured"):
            if ppa.get("fmax_mhz") is not None:
                fmax_vals.append(ppa["fmax_mhz"])
            if ppa.get("wns_ns") is not None:
                wns_vals.append(ppa["wns_ns"])
            if ppa.get("area_um2") is not None:
                area_sum += ppa["area_um2"]
                area_seen = True

        tbs = _block_testbenches(project_root, name, spec, dv_row, dv_pass)
        tb_total += sum(1 for t in tbs if t.get("passed") is not None
                        or t.get("ran") is not False)

        blocks_out.append({
            "name": name,
            "dv": {
                "passed": dv_pass,
                "tests_passed": _int((dv_row or {}).get("tests_passed")),
                "tests_total": _int((dv_row or {}).get("tests_total")),
                "tests_failed": _int((dv_row or {}).get("tests_failed")),
                "source": (dv_row or {}).get("source"),
                "log_path": (dv_row or {}).get("log_path", ""),
                "attempts": _int(comp.get("attempts")),
                "skipped": bool(comp.get("skipped")),
                "escalated": bool(comp.get("escalated")),
                "aborted": bool(comp.get("aborted")),
            },
            "coverage": cov,
            "throughput": _block_throughput(project_root, name),
            "testbenches": tbs,
            "ppa": ppa,
            "hot_patches": _block_hot_patches(project_root, name),
        })

    blocks_total = len(blocks_out)
    blocks_passed = sum(1 for b in blocks_out if b["dv"]["passed"] is True)

    # ---- chip level ----------------------------------------------------
    integ = state.get("integration_dv_result") or {}
    valid = state.get("validation_dv_result") or {}
    design_name = (integ.get("design_name") or valid.get("design_name")
                   or state.get("design_name") or "chip_top")

    integ_dv_row = _dv_one(sb, design_name, "chip")
    valid_dv_row = _dv_one(sb, design_name, "validation")

    def _stage(result: dict, row: dict | None, kind: str) -> dict:
        passed = result.get("passed")
        if passed is None and row is not None:
            passed = bool(row.get("passed"))
        tb = result.get("testbench_path") or result.get("testbench") or ""
        out = {
            "ran": bool(result) or row is not None,
            "passed": (None if passed is None else bool(passed)),
            "test_count": _int(result.get("test_count")
                               or (row or {}).get("tests_total")),
            "testbench": Path(tb).name if tb else "",
            "testbench_path": tb,
            "sim_log_path": result.get("sim_log_path")
            or (row or {}).get("log_path", ""),
            "action_taken": result.get("action_taken", ""),
            "skipped": bool(result.get("skipped_by_user")),
            "aborted": bool(result.get("aborted")),
        }
        if kind == "validation":
            out["requirement_count"] = _int(result.get("requirement_count"))
        return out

    integ_stage = _stage(integ, integ_dv_row, "integration")
    valid_stage = _stage(valid, valid_dv_row, "validation")

    # chip testbenches count toward the total when they ran
    for st in (integ_stage, valid_stage):
        if st["ran"] and st["testbench"]:
            tb_total += 1

    top_ppa = _block_ppa(sb, design_name, period_ns)

    cov_aggregate = (round(100.0 * cov_hit_sum / cov_total_sum, 2)
                     if cov_total_sum else None)
    cov_min = round(min(cov_pcts), 2) if cov_pcts else None
    top_fmax = round(min(fmax_vals), 2) if fmax_vals else None
    top_wns = round(min(wns_vals), 4) if wns_vals else None

    # ---- signoff verdict ----------------------------------------------
    integ_ok = integ_stage["passed"]
    valid_ok = valid_stage["passed"]
    blocks_all_pass = blocks_total > 0 and blocks_passed == blocks_total
    # Audit F1: PASS must mean the TERMINAL chip-level result, not just the DV
    # stages. A run whose validation DV passed but whose chip_top failed to
    # synthesize (or busted its die budget) previously still printed
    # "SIGNOFF SCORECARD: PASS" right under the red NOT-pipeline_done banner.
    # chip_top_synthesizable / die_budget_ok are tri-state: absent (None) on
    # runs that never reached those gates -- only an explicit False fails.
    pipeline_done = bool(state.get("pipeline_done"))
    chip_synth_ok = valid.get("chip_top_synthesizable")
    die_ok = valid.get("die_budget_ok")
    explicit_fail = (
        (blocks_total > 0 and blocks_passed < blocks_total)
        or integ_ok is False or valid_ok is False
        or chip_synth_ok is False or die_ok is False
        or bool(state.get("pipeline_aborted"))
    )
    if (blocks_all_pass and integ_ok is True and valid_ok is True
            and pipeline_done):
        status = "PASS"
        status_reason = ""
    elif explicit_fail:
        status = "FAIL"
        if blocks_total > 0 and blocks_passed < blocks_total:
            status_reason = (f"{blocks_total - blocks_passed} of "
                             f"{blocks_total} blocks did not pass")
        elif integ_ok is False:
            status_reason = "integration DV failed"
        elif valid_ok is False:
            status_reason = "validation DV failed"
        elif chip_synth_ok is False:
            status_reason = ("chip_top is NOT synthesizable: "
                             + str(valid.get("synth_fail_reason", ""))[:500])
        elif die_ok is False:
            status_reason = ("chip does NOT fit its die budget: "
                             + str(valid.get("die_rollup_reason", ""))[:500])
        else:
            status_reason = "pipeline aborted"
    else:
        status = "INCOMPLETE"
        status_reason = (
            "pipeline ended without a terminal chip-level result "
            "(pipeline_done is false)" if not pipeline_done
            else "not all signoff stages ran")

    # Section 3b: carried-forward defects an ADVISORY bypass observed but did not
    # hard-block on. Surfaced here so an advisory bypass never silently swallows
    # a quantified divergence.
    carried_defects = _carried_forward_defects(project_root)

    # Section 7a: engine provenance -- which engine build produced this run, and
    # whether the code was hot-swapped mid-run.
    engine_prov = _engine_provenance(project_root)

    # v3 Section 6: chip-level measured-throughput fact (deterministic-BFM cycle
    # accounting for the op window: START committed -> DONE). n/a when the
    # deterministic BFM did not run / not a QSPI-slave chip.
    chip_throughput = _chip_throughput(project_root)

    # Roll up the per-block throughput gate verdicts for the signoff summary.
    tput_gated = [b for b in blocks_out
                  if b.get("throughput", {}).get("applicable")]
    tput_failed = sum(1 for b in tput_gated
                      if b.get("throughput", {}).get("passed") is False)

    report = {
        "schema": SCHEMA_VERSION,
        "generated_at": _iso(now),
        "project_root": str(project_root),
        "design_name": design_name,
        "target_clock_mhz": target_clock_mhz,
        "engine_sha": engine_prov.get("sha", ""),
        "engine_sha_changed_mid_run": bool(engine_prov.get("changed")),
        "signoff": {
            "status": status,
            "status_reason": status_reason,
            "pipeline_done": pipeline_done,
            "chip_top_synthesizable": chip_synth_ok,
            "die_budget_ok": die_ok,
            "pipeline_aborted": bool(state.get("pipeline_aborted")),
            "blocks_total": blocks_total,
            "blocks_passed": blocks_passed,
            "testbenches_run": tb_total,
            "coverage_aggregate_pct": cov_aggregate,
            "coverage_min_pct": cov_min,
            "coverage_floor": _num(_floor_from_blocks(blocks_out)),
            "top_fmax_mhz": top_fmax,
            "top_wns_ns": top_wns,
            "integration_dv": _verdict_word(integ_ok, integ_stage["ran"]),
            "validation_dv": _verdict_word(valid_ok, valid_stage["ran"]),
            "carried_forward_defect_count": len(carried_defects),
            "throughput_blocks_gated": len(tput_gated),
            "throughput_blocks_failed": tput_failed,
            "chip_measured_cyc_per_op": chip_throughput.get(
                "measured_cyc_per_op_chip"),
        },
        "carried_forward_defects": carried_defects,
        "blocks": blocks_out,
        "chip": {
            "integration_dv": integ_stage,
            "validation_dv": valid_stage,
            "throughput": chip_throughput,
            "ppa": top_ppa,
            "aggregate_area_um2": (round(area_sum, 2) if area_seen else None),
            "totals": {
                "blocks_total": blocks_total,
                "blocks_passed": blocks_passed,
                "testbenches_run": tb_total,
                "coverage_aggregate_pct": cov_aggregate,
                "coverage_min_pct": cov_min,
            },
        },
    }
    return report


def _iso(now: float | None) -> str:
    t = now if now is not None else time.time()
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


def _floor_from_blocks(blocks: list[dict]) -> float | None:
    for b in blocks:
        f = b.get("coverage", {}).get("floor")
        if f is not None:
            return f
    return None


def _verdict_word(passed: bool | None, ran: bool) -> str:
    if not ran or passed is None:
        return "n/a"
    return "pass" if passed else "fail"


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def _fmt(x: Any, suffix: str = "", nd: int = 2) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, float):
        return f"{x:,.{nd}f}{suffix}"
    if isinstance(x, int):
        return f"{x:,}{suffix}"
    return f"{x}{suffix}"


def _cov_cell(cov: dict) -> str:
    if not cov.get("applicable"):
        return f"n/a ({cov.get('reason', 'not measured')})"
    pct = cov.get("pct")
    floor = cov.get("floor")
    gate = "PASS" if cov.get("passed") else "FAIL"
    return (f"{_fmt(pct, '%')} / floor {_fmt(floor, '%', 0)} [{gate}] "
            f"({_fmt(cov.get('points_hit'))}/{_fmt(cov.get('points_total'))} pts)")


def _tput_cells(tp: dict) -> tuple[str, str, str, str]:
    """(declared, measured, ratio, verdict) cells for the throughput table."""
    if not tp.get("applicable"):
        reason = tp.get("reason", "not measured")
        return ("n/a", "n/a", "n/a", f"n/a ({reason})")
    declared = _fmt(_num(tp.get("declared_cyc_per_op")))
    if tp.get("artifact_missing"):
        measured = "n/a (artifact missing)"
    else:
        measured = _fmt(_num(tp.get("measured_cyc_per_op")))
    ratio = tp.get("ratio")
    ratio_s = f"{_fmt(_num(ratio))}x" if ratio is not None else "n/a"
    verdict = "PASS" if tp.get("passed") else "FAIL"
    return (declared, measured, ratio_s, verdict)


def _dv_cell(dv: dict) -> str:
    p = dv.get("passed")
    word = "PASS" if p is True else ("FAIL" if p is False else "n/a")
    tp, tt = dv.get("tests_passed"), dv.get("tests_total")
    if tt is not None:
        return f"{word} ({_fmt(tp)}/{_fmt(tt)} tests)"
    return word


def render_markdown(report: dict) -> str:
    s = report.get("signoff", {})
    status = s.get("status", "INCOMPLETE")
    badge = {"PASS": "PASS ✅", "FAIL": "FAIL ❌"}.get(status, "INCOMPLETE ⚠️")
    lines: list[str] = []
    lines.append(f"# CoreSmith Signoff Scorecard — {report.get('design_name', 'chip')}")
    lines.append("")
    lines.append(f"## SIGNOFF: {badge}")
    lines.append("")
    if s.get("status_reason"):
        lines.append(f"**Why not PASS:** {s.get('status_reason')}")
        lines.append("")
    _synth = s.get("chip_top_synthesizable")
    _die = s.get("die_budget_ok")
    lines.append(
        f"- Terminal chip result: pipeline_done={_fmt(bool(s.get('pipeline_done')))}"
        f", chip_top synthesizable={'n/a' if _synth is None else _fmt(_synth)}"
        f", die budget ok={'n/a' if _die is None else _fmt(_die)}"
    )
    lines.append(f"- Generated: `{report.get('generated_at', '')}`")
    lines.append(f"- Project: `{report.get('project_root', '')}`")
    if report.get("engine_sha"):
        _chg = " ⚠️ CHANGED MID-RUN" if report.get("engine_sha_changed_mid_run") else ""
        lines.append(f"- Engine SHA: `{report.get('engine_sha')}`{_chg}")
    lines.append(f"- Target clock: {_fmt(report.get('target_clock_mhz'), ' MHz')}")
    lines.append(
        f"- Blocks passed: **{s.get('blocks_passed')}/{s.get('blocks_total')}**"
    )
    lines.append(f"- Testbenches run: **{s.get('testbenches_run')}**")
    lines.append(
        f"- Coverage: aggregate {_fmt(s.get('coverage_aggregate_pct'), '%')}, "
        f"min {_fmt(s.get('coverage_min_pct'), '%')} "
        f"(floor {_fmt(s.get('coverage_floor'), '%', 0)})"
    )
    lines.append(
        f"- Top Fmax: {_fmt(s.get('top_fmax_mhz'), ' MHz')} "
        f"(worst WNS {_fmt(s.get('top_wns_ns'), ' ns', 4)})"
    )
    lines.append(
        f"- Integration DV: **{s.get('integration_dv')}** · "
        f"Validation DV: **{s.get('validation_dv')}**"
    )
    _tg_gated = s.get("throughput_blocks_gated")
    if _tg_gated:
        lines.append(
            f"- Throughput: {s.get('throughput_blocks_failed', 0)}/"
            f"{_tg_gated} gated blocks over declared x1.1"
            + (f", chip {_fmt(_num(s.get('chip_measured_cyc_per_op')), ' cyc/op')}"
               if s.get("chip_measured_cyc_per_op") is not None else "")
        )
    cfd = report.get("carried_forward_defects", []) or []
    if cfd:
        lines.append(f"- Carried-forward defects: **{len(cfd)}** "
                     f"(advisory bypasses that did not hard-block)")
    lines.append("")

    # ---- carried-forward defects (advisory-bypass observations) ----
    if cfd:
        lines.append("## Carried-forward defects ⚠️")
        lines.append("")
        lines.append("Quantified divergences an ADVISORY gate observed but did "
                     "not block on. Each must be confirmed cleared by the "
                     "authoritative (integration/validation) DV.")
        lines.append("")
        for d in cfd:
            lines.append(
                f"- **{d.get('gate', '?')}** / {d.get('kind', '?')}: "
                f"{d.get('violation_count', 0)} violation(s)"
                + (f", first at `{d.get('first_divergence_block')}`"
                   if d.get('first_divergence_block') else "")
            )
            # The explanation first -- a gate/kind pair with a violation count
            # tells a reader nothing they can act on.
            if d.get("detail"):
                lines.append(f"  - {d['detail']}")
            if d.get("unmodeled") and d.get("unmodeled") != d.get("detail"):
                lines.append(f"  - unmodeled: {d['unmodeled']}")
        lines.append("")

    # ---- per-block table ----
    lines.append("## Per-block")
    lines.append("")
    lines.append(
        "| Block | DV | Coverage | Testbenches | Cells | FF | Area (µm²) | "
        "WNS (ns) | Fmax (MHz) | PPA |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for b in report.get("blocks", []):
        ppa = b.get("ppa", {})
        tb_names = ", ".join(
            f"{t.get('name')}"
            + (f" ({_fmt(t.get('tests_passed'))}/{_fmt(t.get('tests_total'))})"
               if t.get('tests_total') is not None else "")
            for t in b.get("testbenches", [])
        ) or "n/a"
        ppa_ok = ppa.get("ppa_ok")
        ppa_word = ("PASS" if ppa_ok is True else
                    "FAIL" if ppa_ok is False else "n/a")
        lines.append(
            f"| {b.get('name')} "
            f"| {_dv_cell(b.get('dv', {}))} "
            f"| {_cov_cell(b.get('coverage', {}))} "
            f"| {tb_names} "
            f"| {_fmt(ppa.get('cells'))} "
            f"| {_fmt(ppa.get('ff'))} "
            f"| {_fmt(ppa.get('area_um2'))} "
            f"| {_fmt(ppa.get('wns_ns'), '', 4)} "
            f"| {_fmt(ppa.get('fmax_mhz'))} "
            f"| {ppa_word} |"
        )
    lines.append("")

    # ---- hot patches ----
    hp_rows = [(b.get("name"), b.get("hot_patches"))
               for b in report.get("blocks", []) if b.get("hot_patches")]
    if hp_rows:
        lines.append("### Hot-patches applied")
        for name, patches in hp_rows:
            lines.append(f"- **{name}**: {len(patches)} patch(es)")
            for p in patches:
                lines.append(f"    - {p if isinstance(p, str) else json.dumps(p)}")
        lines.append("")

    # ---- throughput accountability (v3 Section 6) ----
    lines.append("## Throughput accountability")
    lines.append("")
    lines.append(
        "Delivery-time measured cycles/op vs the uArch-declared (§6.1) rate "
        "x 1.1. A block over the ceiling is an RTL performance defect (the "
        "plan->RTL drift)."
    )
    lines.append("")
    lines.append(
        "| Block | Declared cyc/op | Measured cyc/op | Ratio | Gate |")
    lines.append("|---|---|---|---|---|")
    for b in report.get("blocks", []):
        dcl, meas, ratio_s, verdict = _tput_cells(b.get("throughput", {}))
        lines.append(
            f"| {b.get('name')} | {dcl} | {meas} | {ratio_s} | {verdict} |")
    lines.append("")
    ct = report.get("chip", {}).get("throughput", {}) or {}
    _cbudget = _num(ct.get("budget_cyc_per_op"))
    _cmeas = _num(ct.get("measured_cyc_per_op_chip"))
    _cverdict = ("n/a" if ct.get("passed") is None
                 else ("PASS" if ct.get("passed") else "FAIL"))
    lines.append(
        f"- Chip op window: declared budget {_fmt(_cbudget, ' cyc/op')} "
        f"({ct.get('budget_source', 'none')}), measured "
        f"{_fmt(_cmeas, ' cyc/op')} [{_cverdict}] — grader window: "
        f"{ct.get('grader_window', 'n/a')}"
    )
    lines.append("")

    # ---- chip level ----
    chip = report.get("chip", {})
    integ = chip.get("integration_dv", {})
    valid = chip.get("validation_dv", {})
    top_ppa = chip.get("ppa", {})
    lines.append("## Chip-level")
    lines.append("")
    lines.append("| Stage | Verdict | Testbench | Tests | Notes |")
    lines.append("|---|---|---|---|---|")
    lines.append(
        f"| Integration DV | {_verdict_md(integ.get('passed'), integ.get('ran'))} "
        f"| {integ.get('testbench') or 'n/a'} "
        f"| {_fmt(integ.get('test_count'))} "
        f"| {integ.get('action_taken') or ''} |"
    )
    lines.append(
        f"| Validation DV | {_verdict_md(valid.get('passed'), valid.get('ran'))} "
        f"| {valid.get('testbench') or 'n/a'} "
        f"| {_fmt(valid.get('test_count'))} "
        f"| {_req_note(valid)} |"
    )
    lines.append("")
    lines.append(
        f"- Top-level PPA: cells {_fmt(top_ppa.get('cells'))}, "
        f"FF {_fmt(top_ppa.get('ff'))}, "
        f"area {_fmt(top_ppa.get('area_um2'), ' µm²')}, "
        f"aggregate block area {_fmt(chip.get('aggregate_area_um2'), ' µm²')}"
    )
    lines.append(
        f"- Top-level Fmax {_fmt(report.get('signoff', {}).get('top_fmax_mhz'), ' MHz')}, "
        f"WNS {_fmt(top_ppa.get('wns_ns'), ' ns', 4)}"
    )
    lines.append("")
    lines.append(
        "_Deterministic aggregation of recorded DV / coverage / PPA facts. "
        "Coverage above the floor is a weak-TB rejector only, never proof of "
        "correctness._"
    )
    lines.append("")
    return "\n".join(lines)


def _verdict_md(passed: bool | None, ran: bool | None) -> str:
    if not ran or passed is None:
        return "n/a"
    return "PASS ✅" if passed else "FAIL ❌"


def _req_note(valid: dict) -> str:
    rc = valid.get("requirement_count")
    if rc is not None:
        return f"{_fmt(rc)} requirements"
    return valid.get("action_taken") or ""


__all__ = [
    "SCHEMA_VERSION",
    "build_final_report",
    "fmax_mhz",
    "render_markdown",
]
