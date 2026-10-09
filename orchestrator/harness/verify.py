# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Engine-owned verification harness -- the functions behind ``coresmith verify``.

Each ``verify_*`` wraps an EXISTING deterministic pipeline function so an agent
can iterate against the exact check the gate applies (parity by construction).
All heavy imports are deferred into function bodies so ``harness.cli`` (which
imports this lazily) stays langgraph-free at import time.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Exit codes (kept in sync with harness.cli).
_EXIT_PASS = 0
_EXIT_FAIL = 1
_EXIT_INFRA = 3
_EXIT_SKIP = 4


@dataclass
class VerifyResult:
    """Uniform result for every ``verify_*`` call.

    ``exit_code`` maps to the CLI contract: 0 pass / 1 fail / 3 infra / 4 skip.
    """

    passed: bool
    skipped: bool = False
    infra_error: bool = False
    verdict: str = ""
    details: dict = field(default_factory=dict)
    log_path: str = ""
    duration_s: float = 0.0

    @property
    def exit_code(self) -> int:
        if self.infra_error:
            return _EXIT_INFRA
        if self.skipped:
            return _EXIT_SKIP
        return _EXIT_PASS if self.passed else _EXIT_FAIL

    def to_json(self) -> dict:
        return {
            "passed": self.passed,
            "skipped": self.skipped,
            "infra_error": self.infra_error,
            "verdict": self.verdict,
            "details": self.details,
            "log_path": self.log_path,
            "duration_s": round(self.duration_s, 3),
            "exit_code": self.exit_code,
        }

    def to_human(self) -> str:
        state = (
            "SKIP" if self.skipped
            else ("INFRA" if self.infra_error
                  else ("PASS" if self.passed else "FAIL"))
        )
        line = f"[{state}] {self.verdict}"
        if self.log_path:
            line += f"\n  log: {self.log_path}"
        return line


# ---------------------------------------------------------------------------
# Path resolution (never raise)
# ---------------------------------------------------------------------------
def _resolve_rtl_path(pr: Path, spec: dict) -> str:
    from orchestrator.harness.targets import load
    bound = load(pr, spec["name"])
    if bound:
        return bound["sources"][0]
    target = (spec.get("rtl_target") or spec.get("rtl") or "").strip()
    if target:
        p = Path(target)
        return str(p if p.is_absolute() else pr / target)
    return str(pr / "rtl" / f"{spec.get('name')}.v")


def _resolve_tb_path(pr: Path, spec: dict, override: str | None) -> str:
    if override:
        p = Path(override)
        return str(p if p.is_absolute() else pr / override)
    tb = (spec.get("testbench") or "").strip()
    if tb:
        p = Path(tb)
        return str(p if p.is_absolute() else pr / tb)
    return str(pr / "tb" / "cocotb" / f"test_{spec.get('name')}.py")


# ---------------------------------------------------------------------------
# Shared RTL<->model equivalence gate (the anti-cheat gate of record).
# Extracted so generate_testbench_node and the CLI apply the IDENTICAL check
# with the same fail-closed / harness-error-retry semantics (commits 5 + 8).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# verify_model
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# verify_chip_model
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# verify_rtl
# ---------------------------------------------------------------------------
def verify_rtl(
    pr: str | Path,
    block_spec: dict,
    *,
    attempt: int = 0,
    seed: int | None = None,
    tb_path: str | None = None,
    no_equiv: bool = False,
    lint_only: bool = False,
    coverage: bool = False,
    record_source: str = "agent",
    scoreboard: Any = None,
) -> VerifyResult:
    """lint_rtl -> run_simulation -> RTL/model equivalence, one shot.

    This is the deterministic core the gate applies (no LLM TB-fix loop -- the
    node owns that). When ``seed`` is set it is pinned via
    ``CORESMITH_DV_SEED_PIN`` for reproducibility; otherwise a fresh seed is
    used per run. Records a ``dv_results`` row via ``scoreboard`` when provided.
    """
    t0 = time.monotonic()
    root = Path(pr)
    block = block_spec.get("name")
    rtl_path = _resolve_rtl_path(root, block_spec)

    from orchestrator.harness.targets import load, revision
    bound_target = load(root, block)
    bound_revision = revision(bound_target) if bound_target else None

    def _record(res: VerifyResult, tests=(None, None, None), first_div=None) -> VerifyResult:
        if bound_revision:
            res.details["input_revision"] = bound_revision
            try:
                unchanged = revision(load(root, block)) == bound_revision
            except (ValueError, OSError):
                unchanged = False
            if not unchanged:
                res.passed = False
                res.verdict = "TARGET_CHANGED: inputs changed during verification; rerun the check"
        if scoreboard is not None:
            try:
                scoreboard.record_dv(
                    block=block, scope="rtl", source=record_source, attempt=attempt,
                    passed=res.passed, skipped=res.skipped, seed=seed,
                    tests_passed=tests[0], tests_total=tests[1], tests_failed=tests[2],
                    first_divergence=first_div, detail=res.verdict,
                    log_path=res.log_path, duration_s=res.duration_s,
                )
            except Exception:  # noqa: BLE001
                pass
        return res

    if not Path(rtl_path).exists():
        return _record(VerifyResult(
            False, verdict=f"RTL not found: {rtl_path}",
            details={"stage": "rtl", "rtl_path": rtl_path},
            duration_s=time.monotonic() - t0,
        ))

    try:
        from orchestrator.langgraph.pipeline_helpers import lint_rtl, run_simulation
    except Exception as exc:  # noqa: BLE001
        return VerifyResult(False, infra_error=True,
                            verdict=f"pipeline_helpers import failed: {exc}",
                            duration_s=time.monotonic() - t0)

    # --- lint ---
    lint = lint_rtl(rtl_path, block, attempt or 1)
    if not lint.get("clean"):
        return _record(VerifyResult(
            False, verdict="lint failed",
            details={"stage": "lint", "errors": lint.get("errors", "")[:2000]},
            log_path=lint.get("log_path", ""),
            duration_s=time.monotonic() - t0,
        ))
    if lint_only:
        return _record(VerifyResult(
            True, verdict="lint clean",
            details={"stage": "lint"}, log_path=lint.get("log_path", ""),
            duration_s=time.monotonic() - t0,
        ))

    # --- simulate ---
    tbp = _resolve_tb_path(root, block_spec, tb_path)
    if not Path(tbp).exists():
        return _record(VerifyResult(
            False, verdict=f"testbench not found: {tbp}",
            details={"stage": "sim", "tb_path": tbp},
            duration_s=time.monotonic() - t0,
        ))

    prev_seed_pin = os.environ.get("CORESMITH_DV_SEED_PIN")
    prev_cov = os.environ.get("CORESMITH_COVERAGE")
    if seed is not None:
        os.environ["CORESMITH_DV_SEED_PIN"] = str(seed)
    if coverage:
        os.environ["CORESMITH_COVERAGE"] = "1"
    try:
        sim = run_simulation(block_spec, rtl_path, tbp, attempt or 1,
                             project_root=str(pr))
    finally:
        _restore_env("CORESMITH_DV_SEED_PIN", prev_seed_pin)
        _restore_env("CORESMITH_COVERAGE", prev_cov)

    tests = (sim.get("tests_passed"), sim.get("tests_total"), sim.get("tests_failed"))
    sim_passed = bool(sim.get("passed"))
    log_path = sim.get("log_path", "")

    # Coverage (opt-in): annotate + summarize the sim's coverage.dat and record it.
    if coverage:
        try:
            from orchestrator.harness import coverage as _cov
            sim_dir = root / "sim_build" / block
            annotated = _cov.annotate(sim_dir)
            if annotated is not None:
                summary = _cov.summarize(annotated)
                if scoreboard is not None:
                    try:
                        scoreboard.record_coverage(
                            block=block, scope="rtl",
                            points_total=summary.get("points_total"),
                            points_hit=summary.get("points_hit"),
                            pct=summary.get("pct"),
                            uncovered=summary.get("uncovered"),
                            annotated_dir=str(annotated),
                        )
                    except Exception:  # noqa: BLE001
                        pass
        except Exception:  # noqa: BLE001
            pass
    if not sim_passed:
        infra = bool(sim.get("sim_timed_out"))
        return _record(VerifyResult(
            False, infra_error=infra,
            verdict=("sim TIMEOUT" if infra else "simulation failed"),
            details={"stage": "sim", "log_tail": (sim.get("log", "") or "")[-2000:]},
            log_path=log_path, duration_s=time.monotonic() - t0,
        ), tests=tests)

    return _record(VerifyResult(
        True, verdict="simulation passed",
        details={"stage": "sim", "tests": tests},
        log_path=log_path, duration_s=time.monotonic() - t0,
    ), tests=tests)


def _restore_env(key: str, prev: str | None) -> None:
    if prev is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = prev


# ---------------------------------------------------------------------------
# verify_synth
# ---------------------------------------------------------------------------
def _allow_unmeasured_synth_timing() -> bool:
    return os.environ.get("CORESMITH_VERIFY_SYNTH_ALLOW_UNMEASURED", "").strip().lower() \
        in ("1", "true", "yes")


def _measure_full_synth_timing(root: Path, block: str, rtl_path: str, res: dict) -> dict:
    """Pre-layout STA on a ``--full`` synthesis result, through the same
    ``_evaluate_ppa_gate`` block-done uses. Never raises: a tooling failure is
    reported as ``timing_error`` with no WNS."""
    try:
        from orchestrator.langgraph.pipeline_graph import _evaluate_ppa_gate
        _ok, _viol, meta = _evaluate_ppa_gate(str(root), block, rtl_path, res,
                                              require_gate_flag=False)
    except Exception as exc:  # noqa: BLE001
        return {"wns_ns": None, "tns_ns": None, "timing_measured": False,
                "timing_error": f"{type(exc).__name__}: {exc}"[:300]}
    meta = meta or {}
    wns = meta.get("wns_ns")
    return {"wns_ns": wns, "tns_ns": meta.get("tns_ns"), "timing_measured": wns is not None,
            "sta_report": meta.get("sta_report_path", "")}


def verify_synth(
    pr: str | Path,
    block_spec: dict,
    *,
    full: bool = False,
    timeout_s: int = 300,
    target_clock_mhz: float = 50.0,
    scoreboard: Any = None,
    record_source: str = "agent",
    attempt: int = 0,
) -> VerifyResult:
    """Synthesizability / PPA probe for a block.

    Default (fast): PDK-free ``probe_synth_generic`` + ``probe_synth_cellcount``
    + FF-budget compare -- mirrors the core of ``_evaluate_ppa_gate``. ``full``
    runs ``synthesize_block`` (PDK-mapped when a liberty is present).
    """
    t0 = time.monotonic()
    root = Path(pr)
    block = block_spec.get("name")
    rtl_path = _resolve_rtl_path(root, block_spec)
    from orchestrator.harness.targets import load
    if load(root, block):
        full = True
    if not Path(rtl_path).exists():
        return VerifyResult(False, verdict=f"RTL not found: {rtl_path}",
                            duration_s=time.monotonic() - t0)

    # STAGE-REALIZATION GATE (pipeline-campaign): reject a single-cycle
    # combinational cloud (a collapsed multi-stage datapath) in milliseconds,
    # BEFORE paying the yosys elaboration timeout. Same deterministic census the
    # generate_rtl acceptance path uses; parity by construction. Best-effort and
    # env-gated (CORESMITH_STAGE_LINT=0 bypasses).
    try:
        from orchestrator.langgraph.ppa_check import (
            max_cell_ceiling,
            parse_ff_budget,
            probe_synth_cellcount,
            probe_synth_generic,
        )
    except Exception as exc:  # noqa: BLE001
        return VerifyResult(False, infra_error=True,
                            verdict=f"ppa_check import failed: {exc}",
                            duration_s=time.monotonic() - t0)

    def _record(ok, ff=None, cells=None, mem_bits=None, elaborated=None,
                budget_ff=None, reasons=None, report_path="", probe="probes"):
        if scoreboard is not None:
            try:
                scoreboard.record_ppa(
                    block=block, attempt=attempt, source=record_source, probe=probe,
                    cells=cells, ff=ff, mem_bits=mem_bits, elaborated=elaborated,
                    budget_ff=budget_ff, ppa_ok=ok, reasons=reasons,
                    report_path=report_path,
                )
            except Exception:  # noqa: BLE001
                pass

    if full:
        try:
            from orchestrator.langgraph.pipeline_helpers import synthesize_block
        except Exception as exc:  # noqa: BLE001
            return VerifyResult(False, infra_error=True,
                                verdict=f"synthesize_block import failed: {exc}",
                                duration_s=time.monotonic() - t0)
        res = synthesize_block(block_spec, rtl_path, target_clock_mhz, attempt or 1, timeout_s=timeout_s)
        ok = bool(res.get("success"))
        _record(ok, ff=res.get("ff_count"), report_path=res.get("report_path", ""),
                probe="synth")
        details = {"stage": "synth", "ff": res.get("ff_count"),
                   "area_um2": res.get("chip_area_um2"),
                   "gate_count": res.get("gate_count")}
        if res.get("input_revision"):
            details["input_revision"] = res["input_revision"]
        if not ok:
            return VerifyResult(False, infra_error=bool(res.get("timed_out")),
                                verdict="synth timed out" if res.get("timed_out") else "synth FAILED", details=details,
                                log_path=res.get("log_path", ""),
                                duration_s=time.monotonic() - t0)
        # A --full synthesis is a timing verdict too (the same measurement
        # block-done publishes on): no WNS is "timing not measured" -- a tool
        # error, never a pass (it used to report PASS with wns=None when the
        # OpenSTA binary was missing).
        timing = _measure_full_synth_timing(root, block, rtl_path, res)
        details.update(timing)
        wns, tns = timing.get("wns_ns"), timing.get("tns_ns")
        try:
            neg = (wns is not None and float(wns) < 0) or (tns is not None and float(tns) < 0)
        except (TypeError, ValueError):
            neg = False
        if neg:
            return VerifyResult(False, verdict=f"synth OK; timing violated (WNS {wns} ns, TNS {tns} ns)",
                                details=details, log_path=res.get("log_path", ""),
                                duration_s=time.monotonic() - t0)
        if wns is None:
            if not _allow_unmeasured_synth_timing():
                return VerifyResult(
                    False, infra_error=True,
                    verdict=("timing not measured (no WNS: the STA did not run or crashed"
                             + (f": {timing['timing_error']}" if timing.get("timing_error") else "")
                             + ") -- check bin/sta / CORESMITH_REAL_STA; "
                             "CORESMITH_VERIFY_SYNTH_ALLOW_UNMEASURED=1 waives"),
                    details={**details, "tool_error": True},
                    log_path=res.get("log_path", ""),
                    duration_s=time.monotonic() - t0,
                )
            return VerifyResult(True, verdict="synth OK; timing NOT measured (waived: "
                                "CORESMITH_VERIFY_SYNTH_ALLOW_UNMEASURED=1)",
                                details=details, log_path=res.get("log_path", ""),
                                duration_s=time.monotonic() - t0)
        return VerifyResult(True, verdict=f"synth OK; WNS {wns} ns", details=details,
                            log_path=res.get("log_path", ""),
                            duration_s=time.monotonic() - t0)

    spec_path = root / "arch" / "uarch_specs" / f"{block}.md"
    spec_text = spec_path.read_text() if spec_path.exists() else ""
    ff_budget = parse_ff_budget(spec_text) if spec_text else None

    probe = probe_synth_generic(rtl_path, block, timeout_s=timeout_s)
    if probe is None:
        _record(None, probe="generic")
        return VerifyResult(
            False, skipped=True, verdict="yosys absent -- cannot judge PPA",
            details={"stage": "synth", "tooling_missing": True},
            duration_s=time.monotonic() - t0,
        )
    if probe.get("elaborated") is False:
        _record(False, elaborated=False, reasons=[probe.get("reason", "")],
                probe="generic")
        return VerifyResult(
            False, verdict=f"did not elaborate: {probe.get('reason', '')}",
            details={"stage": "synth", "probe": probe},
            duration_s=time.monotonic() - t0,
        )

    ff = probe.get("logic_ff")
    reasons: list[str] = []
    if ff_budget is not None and ff is not None and ff > ff_budget:
        reasons.append(f"flip-flops {ff} exceed budget {ff_budget}")

    cprobe = probe_synth_cellcount(
        rtl_path, block, timeout_s=timeout_s, cwd=str(root),
    )
    cells = None
    if cprobe is not None:
        if cprobe.get("elaborated") is False:
            _record(False, ff=ff, elaborated=False,
                    reasons=[cprobe.get("reason", "")], probe="cellcount")
            return VerifyResult(
                False, verdict=f"did not techmap: {cprobe.get('reason', '')}",
                details={"stage": "synth", "probe": cprobe},
                duration_s=time.monotonic() - t0,
            )
        cells = cprobe.get("cell_count")
        ceil = max_cell_ceiling()
        if cells is not None and cells > ceil:
            reasons.append(f"cell count {cells} exceeds ceiling {ceil}")

    ok = not reasons
    _record(ok, ff=ff, cells=cells, mem_bits=probe.get("mem_bits"),
            elaborated=True, budget_ff=ff_budget, reasons=reasons or None,
            probe="probes")
    return VerifyResult(
        ok,
        verdict=("synthesizable, within budget" if ok else "; ".join(reasons)),
        details={"stage": "synth", "ff": ff, "cells": cells,
                 "budget_ff": ff_budget, "mem_bits": probe.get("mem_bits")},
        duration_s=time.monotonic() - t0,
    )


# ---------------------------------------------------------------------------
# verify_chip
# ---------------------------------------------------------------------------
def verify_chip(
    pr: str | Path,
    *,
    tb_path: str | None = None,
    seed: int | None = None,
    stimulus: str | None = None,
    scoreboard: Any = None,
    record_source: str = "agent",
    attempt: int = 0,
) -> VerifyResult:
    """Integrated chip_top DV via run_integration_simulation.

    Inputs come from ``.coresmith/integration_result.json`` (persisted by
    integration_check). A ``sim:<scope>`` lease in the project database
    serializes concurrent chip sims (C1); a crashed sim frees it by pid death.
    """
    t0 = time.monotonic()
    root = Path(pr)
    if stimulus:
        # The chip TB is fixed by integration_result.json; there is no stimulus
        # selection on this path. Say so instead of running a DIFFERENT stimulus
        # than the caller asked for and reporting the result as theirs.
        return VerifyResult(
            False, skipped=True,
            verdict="--stimulus is not supported for chip DV "
                    "(the testbench comes from integration_result.json; "
                    "use --tb to point at a different one)",
            duration_s=time.monotonic() - t0,
        )
    ir_path = root / ".coresmith" / "integration_result.json"
    if not ir_path.exists():
        return VerifyResult(
            False, skipped=True,
            verdict="no integration_result.json (run integration_check first)",
            duration_s=time.monotonic() - t0,
        )
    try:
        ir = json.loads(ir_path.read_text())
    except Exception as exc:  # noqa: BLE001
        return VerifyResult(False, infra_error=True,
                            verdict=f"bad integration_result.json: {exc}",
                            duration_s=time.monotonic() - t0)

    design = ir.get("design_name") or ir.get("design") or root.name
    top_rtl = ir.get("top_rtl_path") or ""
    block_rtls = ir.get("block_rtl_paths") or {}
    tbp = tb_path or ir.get("tb_path") or ir.get("integration_tb_path") or ""
    if not (top_rtl and tbp):
        return VerifyResult(
            False, skipped=True,
            verdict="integration_result.json missing top_rtl_path/tb_path",
            details={"integration_result": ir},
            duration_s=time.monotonic() - t0,
        )

    try:
        from orchestrator.langgraph.integration_helpers import run_integration_simulation
    except Exception as exc:  # noqa: BLE001
        return VerifyResult(False, infra_error=True,
                            verdict=f"integration_helpers import failed: {exc}",
                            duration_s=time.monotonic() - t0)

    # Agent-invoked chip verify (record_source="agent" -- e.g. a TB-gen agent's
    # in-context check) MUST NOT build in the engine-authoritative
    # sim_build/integration dir: its pre-build would leave a stale/traceless Vtop
    # that the gate's cocotb make then reuses, emitting no dump.vcd and fail-closing
    # the mandatory WaveKit audit (2026-07-02 integration-DV failure). Route agent
    # runs to a scratch namespace so gate-side dirs stay authoritative; the gate
    # (record_source="gate") keeps sim_build/integration. The flock already
    # serializes concurrent runs within a namespace.
    sim_scope = "agent_integration" if record_source == "agent" else "integration"
    lock_dir = root / "sim_build" / sim_scope
    lock_dir.mkdir(parents=True, exist_ok=True)
    # C1: the per-namespace serialization is a lease in the project database
    # (``sim:<scope>``), not a flock on a sidecar file: a crashed sim releases
    # by pid death instead of leaving a lock nobody can inspect.
    from orchestrator.state_store.leases import LeaseUnavailable, db_lease
    from orchestrator.state_store.project_db import open_project
    _lease_db = open_project(pr)
    # Pin the seed for the sim (run_integration_simulation inherits os.environ),
    # else --seed only decorated the scoreboard row while the TB drew its own
    # seed and the reported failure did not reproduce. Unlike the block path
    # there is no mint point here, so set the minted var too, not just the pin.
    prev_seed_pin = os.environ.get("CORESMITH_DV_SEED_PIN")
    prev_seed = os.environ.get("CORESMITH_DV_SEED")
    if seed is not None:
        os.environ["CORESMITH_DV_SEED_PIN"] = str(seed)
        os.environ["CORESMITH_DV_SEED"] = str(seed)
    try:
        with db_lease(_lease_db, f"sim:{sim_scope}", ttl_s=1800, wait_s=3600,
                      meta={"design": design, "source": record_source}):
            try:
                res = run_integration_simulation(
                    design, top_rtl, block_rtls, tbp, attempt or 1, sim_scope=sim_scope,
                    project_root=str(pr),
                )
            finally:
                _restore_env("CORESMITH_DV_SEED_PIN", prev_seed_pin)
                _restore_env("CORESMITH_DV_SEED", prev_seed)
    except LeaseUnavailable as exc:
        _restore_env("CORESMITH_DV_SEED_PIN", prev_seed_pin)
        _restore_env("CORESMITH_DV_SEED", prev_seed)
        return VerifyResult(False, infra_error=True,
                            verdict=f"integration sim lease unavailable: {exc}",
                            duration_s=time.monotonic() - t0)

    passed = bool(res.get("passed"))
    if scoreboard is not None:
        try:
            scoreboard.record_dv(
                block=design, scope="chip", source=record_source, attempt=attempt,
                passed=passed, seed=seed, detail=("chip DV" if passed else "chip DV failed"),
                log_path=res.get("log_path", ""),
            )
        except Exception:  # noqa: BLE001
            pass
    return VerifyResult(
        passed,
        verdict=("chip_top DV passed" if passed else "chip_top DV failed"),
        details={"stage": "chip", "log_tail": (res.get("log", "") or "")[-2000:]},
        log_path=res.get("log_path", ""),
        duration_s=time.monotonic() - t0,
    )
