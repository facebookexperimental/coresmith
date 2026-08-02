# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Checker classes -- one per tool-output parser, three-state and fail-closed.

Each :class:`~orchestrator.pdk.base.Checker` here WRAPS an existing,
battle-hardened parser (``ppa_check``, ``backend_helpers``, ``drc_verdict``,
``gate_sim``) rather than re-implementing it -- the subtle LVS
benign-reconciliation and DRC macro-interior logic must not drift. A checker:

* resolves its report from ``req.inputs`` (an explicit override) or by a
  conventional filename under ``run_dir``;
* returns ``pass`` / ``fail`` / ``skip`` / ``not_run`` per
  :mod:`orchestrator.pdk.base`; a **missing** report is ``not_run`` (fail-closed
  when ``blocking``), never a false clean and never a fabricated count;
* exposes a pure ``classmethod`` (``parse_*`` / ``from_*``) over TEXT so the tool
  ``run()`` bodies -- which already hold the parsed output -- reuse the identical
  rule without touching disk twice.

All parser imports are LAZY (inside method bodies) so importing this module never
pulls in ``orchestrator.langgraph`` at load time -- ``deployments/sky130`` imports
this module, and ``backend_helpers`` imports ``sky130`` at load, so a top-level
parser import here would close that cycle.
"""

from __future__ import annotations

import re
from pathlib import Path

from orchestrator.pdk.base import Checker, CheckResult, ToolRequest

# ---------------------------------------------------------------------------
# Shared report resolution
# ---------------------------------------------------------------------------
_CHIP_AREA_RE = re.compile(r"Chip area for (?:top )?module.*?:\s*([\d.]+)")


def _resolve_report(
    req: ToolRequest | None,
    run_dir: Path | None,
    input_key: str,
    patterns: tuple[str, ...],
) -> Path | None:
    """First existing report: an explicit ``req.inputs[input_key]`` else the
    first ``run_dir`` glob match across ``patterns`` (most-specific first)."""
    if req is not None:
        override = req.inputs.get(input_key)
        if override is not None and Path(override).is_file():
            return Path(override)
    if run_dir is not None and Path(run_dir).is_dir():
        for pat in patterns:
            hits = sorted(Path(run_dir).glob(pat))
            if hits:
                return hits[0]
    return None


def _read(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return path.read_text(errors="replace")
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------
class SynthStatChecker(Checker):
    """Cell / flip-flop / area metrics from a Yosys ``stat`` report.

    Wraps :func:`ppa_check.count_cells_from_stat` (which already understands the
    Yosys 0.65 ``N cells`` box-format total AND the legacy ``Number of cells: N``
    line) and :func:`ppa_check.count_flops_from_stat`. Advisory: it reports
    metrics, it does not by itself gate synthesis (the process return-code and
    the netlist-presence check do); a missing report is ``not_run`` but
    non-blocking so it never fails an otherwise-successful synth.
    """

    name = "synth_stat"
    blocking = False

    @staticmethod
    def parse_text(stat_text: str) -> dict[str, int | float]:
        from orchestrator.langgraph.ppa_check import (
            count_cells_from_stat,
            count_flops_from_stat,
        )

        cells = count_cells_from_stat(stat_text) or 0
        ff = count_flops_from_stat(stat_text) or 0
        area = 0.0
        m = _CHIP_AREA_RE.search(stat_text or "")
        if m:
            try:
                area = float(m.group(1))
            except ValueError:
                area = 0.0
        return {"cells": int(cells), "ff_count": int(ff),
                "chip_area_um2": area}

    def check(self, req: ToolRequest, run_dir: Path) -> CheckResult:
        report = _resolve_report(
            req, run_dir, "report",
            ("*_report.txt", "*stat*.txt", "*_report.rpt", "*.rpt"))
        text = _read(report)
        if text is None:
            return CheckResult(self.name, "not_run", blocking=self.blocking,
                               details="no synthesis stat report found")
        m = self.parse_text(text)
        return CheckResult(self.name, "pass", metrics=m,
                           blocking=self.blocking,
                           details=f"{m['cells']} cells / {m['ff_count']} FF")


class LogicDepthChecker(Checker):
    """Longest combinational logic depth from a Yosys ``ltp`` dump (advisory).

    Wraps :func:`ppa_check.count_logic_depth_from_ltp`. Never blocking -- deep
    logic is a timing risk, not a hard error; it is surfaced as a metric.
    """

    name = "logic_depth"
    blocking = False

    @staticmethod
    def parse_text(ltp_text: str) -> int | None:
        from orchestrator.langgraph.ppa_check import count_logic_depth_from_ltp

        return count_logic_depth_from_ltp(ltp_text)

    def check(self, req: ToolRequest, run_dir: Path) -> CheckResult:
        report = _resolve_report(
            req, run_dir, "ltp",
            ("*ltp*.txt", "*ltp*.rpt", "*_report.txt"))
        text = _read(report)
        if text is None:
            return CheckResult(self.name, "not_run", blocking=self.blocking,
                               details="no ltp report found")
        depth = self.parse_text(text)
        if depth is None:
            return CheckResult(self.name, "not_run", blocking=self.blocking,
                               details="ltp report carried no length=N")
        return CheckResult(self.name, "pass", metrics={"logic_depth": depth},
                           blocking=self.blocking,
                           details=f"logic depth = {depth}")


# ---------------------------------------------------------------------------
# PnR / STA
# ---------------------------------------------------------------------------
class PnrReportsChecker(Checker):
    """WNS/TNS/power/area from an OpenROAD reports directory (advisory).

    Wraps :func:`backend_helpers.parse_openroad_reports` (which reads
    ``timing_wns.rpt`` / ``timing_tns.rpt`` / ``power.rpt`` / ``area.rpt`` from
    the run dir). Honest three-state: ``timing_met is None`` -- no WNS report was
    written -- is ``not_run`` (never silently "clean"); measured WNS >= 0 is
    ``pass``; a negative WNS is ``fail``. Advisory so it augments, rather than
    overrides, the process-level PnR verdict.
    """

    name = "pnr_reports"
    blocking = False

    def check(self, req: ToolRequest, run_dir: Path) -> CheckResult:
        from orchestrator.langgraph.backend_helpers import parse_openroad_reports

        if run_dir is None or not Path(run_dir).is_dir():
            return CheckResult(self.name, "not_run", blocking=self.blocking,
                               details=f"no PnR report dir: {run_dir}")
        m = parse_openroad_reports(str(run_dir))
        metrics = {k: m[k] for k in (
            "wns_ns", "tns_ns", "setup_slack_ns", "hold_slack_ns",
            "total_power_mw", "design_area_um2", "die_area_um2",
            "utilization_pct") if k in m}
        timing_met = m.get("timing_met")
        if timing_met is None:
            return CheckResult(self.name, "not_run", metrics=metrics,
                               blocking=self.blocking,
                               details="no timing WNS report -- timing unmeasured")
        status = "pass" if timing_met else "fail"
        return CheckResult(self.name, status, metrics=metrics,
                           blocking=self.blocking,
                           details=f"WNS={m.get('wns_ns')} ns")


class RouteDrcChecker(Checker):
    """Detailed-route DRC violation count from an OpenROAD ``route_drc.rpt``.

    Wraps :func:`backend_helpers.count_route_drc_violations`. Blocking: a routed
    design OpenROAD left with unresolved detailed-route DRC is not a passing PnR.
    The report is ALWAYS written by ``detailed_route`` (empty when clean), so its
    absence means routing never happened -> ``not_run`` (fail-closed), while a
    present-and-empty report reads as 0 -> ``pass``.
    """

    name = "route_drc"
    blocking = True

    def check(self, req: ToolRequest, run_dir: Path) -> CheckResult:
        from orchestrator.langgraph.backend_helpers import (
            count_route_drc_violations,
        )

        report = _resolve_report(
            req, run_dir, "route_drc",
            ("route_drc.rpt", "*route_drc*.rpt", "*route*drc*.rpt"))
        if report is None:
            return CheckResult(self.name, "not_run", blocking=self.blocking,
                               details="no route_drc.rpt -- routing did not run")
        n = count_route_drc_violations(str(report))
        status = "pass" if n == 0 else "fail"
        return CheckResult(self.name, status,
                           metrics={"route_drc_violations": n},
                           blocking=self.blocking,
                           details=f"{n} detailed-route DRC violation(s)")


class StaChecker(Checker):
    """WNS/TNS (ns) from an OpenSTA ``report_wns``/``report_tns`` dump.

    Wraps :func:`ppa_check.parse_sta_report` (accepts both the legacy ``wns N``
    and the OpenSTA 3.x ``wns max N`` forms, and scientific notation). Advisory:
    surfaces timing metrics; a negative WNS is ``fail``, an unparseable report is
    ``not_run``.
    """

    name = "sta"
    blocking = False

    @staticmethod
    def parse_text(report_text: str) -> dict[str, float | None]:
        from orchestrator.langgraph.ppa_check import parse_sta_report

        return parse_sta_report(report_text)

    def check(self, req: ToolRequest, run_dir: Path) -> CheckResult:
        report = _resolve_report(
            req, run_dir, "sta",
            ("*sta*.rpt", "*sta*.txt", "*timing*.rpt"))
        text = _read(report)
        if text is None:
            return CheckResult(self.name, "not_run", blocking=self.blocking,
                               details="no STA report found")
        m = self.parse_text(text)
        wns = m.get("wns_ns")
        metrics = {k: v for k, v in m.items() if v is not None}
        if wns is None:
            return CheckResult(self.name, "not_run", metrics=metrics,
                               blocking=self.blocking,
                               details="STA report carried no parseable WNS")
        status = "pass" if wns >= 0 else "fail"
        return CheckResult(self.name, status, metrics=metrics,
                           blocking=self.blocking, details=f"WNS={wns} ns")


# ---------------------------------------------------------------------------
# DRC / LVS
# ---------------------------------------------------------------------------
class MagicDrcChecker(Checker):
    """Magic DRC verdict via :func:`drc_verdict.classify_drc` three-state.

    Counts violation rects from the report file with
    :func:`backend_helpers.parse_drc_report` -- the report-file-aware parser that
    :func:`backend_helpers._parse_magic_drc_count` itself falls back to (it
    understands the native ``drc listall why`` rects, the ``DRC count: N`` line,
    and the Tcl-brace form, and returns ``-1`` for a missing file) -- then
    classifies with :func:`drc_verdict.classify_drc` so the honest rules are
    applied verbatim: a positive count is ``fail`` however Magic exited; "clean"
    must be EARNED (report exists, non-empty, count parsed) else ``not_run`` --
    never a false clean, never a ``-1``.
    """

    name = "magic_drc"
    blocking = True

    def check(self, req: ToolRequest, run_dir: Path) -> CheckResult:
        from orchestrator.langgraph.backend_helpers import parse_drc_report
        from orchestrator.langgraph.drc_verdict import (
            STATUS_FAIL,
            STATUS_PASS,
            classify_drc,
        )

        report = _resolve_report(
            req, run_dir, "drc_report",
            ("magic_drc.rpt", "*drc*.rpt", "*_drc.rpt"))
        # parse_drc_report returns violation_count=-1 for a missing file, which
        # classify_drc reads as "unmeasured" (not_run), never a false 0.
        parsed = parse_drc_report(str(report)) if report else \
            {"violation_count": -1}
        verdict = classify_drc(
            violation_count=parsed.get("violation_count"),
            report_path=str(report) if report else "",
            tool_ran=report is not None,
            tool="Magic",
        )
        status = {STATUS_PASS: "pass", STATUS_FAIL: "fail"}.get(
            verdict["status"], "not_run")
        vc = verdict.get("violation_count")
        return CheckResult(
            self.name, status,
            metrics={"violations": vc} if vc is not None else {},
            blocking=self.blocking, details=verdict.get("reason", ""))


class LvsMatchChecker(Checker):
    """Netgen LVS match verdict with benign-pin reconciliation.

    Reproduces the report-parsing rule from
    :func:`backend_helpers.run_netgen_lvs` (the "final result ... match
    uniquely" scan) and then CALLS -- never re-implements --
    :func:`backend_helpers.reconcile_lvs_match` so the battle-hardened
    openframe GPIO/power benign-tie reconciliation is applied identically.
    Blocking: a missing report is ``not_run``.
    """

    name = "lvs_match"
    blocking = True

    @staticmethod
    def raw_match_from_report(report_text: str) -> bool:
        """The raw netgen verdict: last 'final result' line, else a
        conservative substring scan (mirrors run_netgen_lvs)."""
        combined = report_text or ""
        final_line = ""
        for line in reversed(combined.split("\n")):
            if "final result" in line.lower():
                final_line = line.lower()
                break
        if final_line:
            return "match uniquely" in final_line
        low = combined.lower()
        return ("match" in low and "do not match" not in low
                and "failed" not in low)

    def check(self, req: ToolRequest, run_dir: Path) -> CheckResult:
        from orchestrator.langgraph.backend_helpers import reconcile_lvs_match

        report = _resolve_report(
            req, run_dir, "lvs_report",
            ("*lvs*.rpt", "lvs_*.rpt", "*_lvs.rpt"))
        report_text = _read(report)
        if report_text is None:
            return CheckResult(self.name, "not_run", blocking=self.blocking,
                               details="no LVS report found")
        raw = self.raw_match_from_report(report_text)
        recon = reconcile_lvs_match(
            raw, report_text, top_cell=(req.design if req else ""))
        match = bool(recon["lvs_match"])
        return CheckResult(
            self.name, "pass" if match else "fail",
            metrics={
                "lvs_raw_match": recon["lvs_raw_match"],
                "benign_reconciled_pins": recon["benign_reconciled_pins"],
            },
            blocking=self.blocking,
            details=recon.get("lvs_benign_analysis", "")
            or ("match uniquely" if match else "circuits do not match"))


# ---------------------------------------------------------------------------
# Lint / gate-sim
# ---------------------------------------------------------------------------
class LintChecker(Checker):
    """Verilator lint verdict: a ``%Error`` token (or a non-zero return code)
    fails the lint -- the rule from ``pipeline_helpers.lint_rtl``.
    """

    name = "lint"
    blocking = True

    @staticmethod
    def classify(stderr: str, returncode: int = 0) -> bool:
        """True iff lint is clean: no ``%Error`` and returncode 0."""
        has_errors = "%Error" in (stderr or "")
        return returncode == 0 and not has_errors

    def check(self, req: ToolRequest, run_dir: Path) -> CheckResult:
        report = _resolve_report(
            req, run_dir, "lint_log",
            ("*lint*.log", "*lint*.txt", "*.log"))
        text = _read(report)
        if text is None:
            return CheckResult(self.name, "not_run", blocking=self.blocking,
                               details="no verilator lint log found")
        clean = self.classify(text, returncode=0)
        return CheckResult(self.name, "pass" if clean else "fail",
                           blocking=self.blocking,
                           details="" if clean else text[-400:])


class GateSimVerdictChecker(Checker):
    """Gate-level sim verdict: a blank/missing/malformed verdict JSON is NOT a
    pass -- the fail-closed rule from ``harness.gate_sim`` (a verdict file that
    is absent, empty, unparseable, or missing ``cycles_compared`` is
    ``not_run``; ``mismatches == 0`` over a non-empty compare is ``pass``).
    """

    name = "gate_sim_verdict"
    blocking = True

    def check(self, req: ToolRequest, run_dir: Path) -> CheckResult:
        import json

        report = _resolve_report(
            req, run_dir, "verdict",
            ("gate_sim_verdict.json", "*verdict*.json"))
        if report is None or report.stat().st_size == 0:
            return CheckResult(self.name, "not_run", blocking=self.blocking,
                               details="gate-sim produced no (or a blank) "
                                       "verdict file")
        try:
            verdict = json.loads(report.read_text())
        except (OSError, ValueError) as exc:
            return CheckResult(self.name, "not_run", blocking=self.blocking,
                               details=f"unparseable gate-sim verdict: {exc}")
        if not isinstance(verdict, dict) or "cycles_compared" not in verdict:
            return CheckResult(self.name, "not_run", blocking=self.blocking,
                               details="gate-sim verdict missing cycles_compared")
        mism = int(verdict.get("mismatches", 0) or 0)
        cyc = int(verdict.get("cycles_compared", 0) or 0)
        metrics = {"cycles_compared": cyc, "mismatches": mism}
        status = "pass" if mism == 0 else "fail"
        return CheckResult(self.name, status, metrics=metrics,
                           blocking=self.blocking,
                           details=f"{mism} mismatch(es) over {cyc} cycles")


__all__ = [
    "SynthStatChecker",
    "LogicDepthChecker",
    "PnrReportsChecker",
    "RouteDrcChecker",
    "StaChecker",
    "MagicDrcChecker",
    "LvsMatchChecker",
    "LintChecker",
    "GateSimVerdictChecker",
]
