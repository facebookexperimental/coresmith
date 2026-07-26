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


def _model_wrap_path(pr: Path, block: str) -> str:
    return str(pr / "tb" / "cocotb" / f"{block}_model.py")


def _block_model_path(pr: Path, block: str) -> Path:
    return pr / "arch" / "block_models" / f"{block}.py"


# ---------------------------------------------------------------------------
# Shared RTL<->model equivalence gate (the anti-cheat gate of record).
# Extracted so generate_testbench_node and the CLI apply the IDENTICAL check
# with the same fail-closed / harness-error-retry semantics (commits 5 + 8).
# ---------------------------------------------------------------------------
def run_block_equiv_gate(
    block_name: str,
    rtl_path: str,
    project_root: str | Path,
    *,
    seed: int | None = None,
) -> dict:
    """Run the RTL-vs-model byte-exact equivalence gate for one block.

    Returns a dict::

        {"ran": bool,          # gate applicable AND executed
         "passed": bool,        # byte-exact match
         "skipped": bool,       # honest skip (non-blocking)
         "failed_closed": bool, # harness-error-persist / gate-error, fail-closed
         "reason": str,
         "checked_vectors": int,
         "prev_error_text": str | None}   # to write to previous_error.txt

    ``ran=False`` means the gate does not apply (equiv off / block-goldens off /
    no RTL) -> the caller keeps its sim verdict unchanged. Never raises.
    """
    pr = Path(project_root)
    out = {
        "ran": False, "passed": False, "skipped": False, "failed_closed": False,
        "reason": "", "checked_vectors": 0, "prev_error_text": None,
    }
    try:
        from orchestrator.architecture import composition as _composition
        from orchestrator.langgraph.gate_guard import gate_fail_open_enabled
        from orchestrator.langgraph.rtl_model_equiv import (
            check_rtl_model_equivalence as _check_equiv,
        )
        from orchestrator.langgraph.rtl_model_equiv import (
            rtl_model_equiv_enabled as _equiv_enabled,
        )
    except Exception as exc:  # noqa: BLE001
        out["reason"] = f"equiv imports unavailable: {exc}"
        return out

    if not (rtl_path and _equiv_enabled() and _composition.block_goldens_enabled()):
        return out  # gate not applicable

    from orchestrator.harness.seed_provider import gate_seed
    _seed = gate_seed(explicit=seed, use_env=True)
    model_wrap = _model_wrap_path(pr, block_name)
    out["ran"] = True
    try:
        eq = _check_equiv(block_name, rtl_path, model_wrap,
                          project_root=str(pr), seed=_seed)
        # A-Fix 2c: a HARNESS/ENV skip is not honest -> retry once at 2x, then
        # fail closed; honest skips stay non-blocking.
        if eq.get("skipped") and eq.get("harness_error"):
            eq = _check_equiv(block_name, rtl_path, model_wrap,
                              project_root=str(pr), seed=_seed, timeout_scale=2.0)
        if eq.get("skipped"):
            if eq.get("harness_error") and not gate_fail_open_enabled():
                rsn = eq.get("reason", "equivalence harness error")
                out.update(
                    failed_closed=True, reason=rsn,
                    prev_error_text=(
                        "RTL-vs-model equivalence gate could NOT run "
                        "(harness/environment error, retried once with 2x "
                        "timeout). This is NOT a pass (fail-closed):\n" + rsn
                    ),
                )
            else:
                out.update(skipped=True, reason=eq.get("reason", ""))
        elif not eq.get("passed"):
            rsn = eq.get("reason", "RTL diverged from model")
            out.update(
                passed=False, reason=rsn,
                checked_vectors=int(eq.get("checked_vectors", 0) or 0),
                prev_error_text=(
                    "RTL-vs-model equivalence gate FAILED (fix #1). The RTL "
                    "is not byte-exact to the proven Amaranth block model:\n" + rsn
                ),
            )
        else:
            out.update(
                passed=True,
                checked_vectors=int(eq.get("checked_vectors", 0) or 0),
                reason="byte-exact",
            )
    except Exception as exc:  # noqa: BLE001
        if gate_fail_open_enabled():
            out.update(ran=True, passed=True, reason=f"gate error (fail-open): {exc}")
        else:
            out.update(
                failed_closed=True, reason=f"gate errored: {exc!r}",
                prev_error_text=(
                    "RTL-vs-model equivalence gate ERRORED (fail-closed). This "
                    "is NOT a pass -- the gate harness/environment failed:\n"
                    f"{exc!r}"
                ),
            )
    return out


# ---------------------------------------------------------------------------
# verify_model
# ---------------------------------------------------------------------------
def verify_model(
    pr: str | Path, block: str, *, skip_size: bool = False,
) -> VerifyResult:
    """Elaborate + interface-check (+ size) one Amaranth block model.

    The seconds-fast deterministic check that replaces the 69-min build/verify
    death spiral: import + Amaranth-elaborate the model, confirm it kept every
    expected interface, and (unless ``skip_size``) size its datapath/memory vs
    the block area budget.
    """
    t0 = time.monotonic()
    root = Path(pr)
    try:
        from orchestrator.langgraph.microarch_exp import (
            _expected_ports_for_block,
            _factory_params,
            _read_block_diagram,
            _read_target_clock_mhz,
            _read_uarch_specs,
            _size_one_model,
            check_interface_constraint,
            elaborate_block_model,
        )
    except Exception as exc:  # noqa: BLE001
        return VerifyResult(False, infra_error=True,
                            verdict=f"microarch import failed: {exc}",
                            duration_s=time.monotonic() - t0)

    model_path = _block_model_path(root, block)
    err = elaborate_block_model(str(model_path), block)
    if err:
        return VerifyResult(
            False, verdict=f"model elaboration failed: {err}",
            details={"stage": "elaborate", "error": err, "model_path": str(model_path)},
            duration_s=time.monotonic() - t0,
        )

    expected = _expected_ports_for_block(_read_block_diagram(str(root)), block)
    params = _factory_params(str(model_path), block)
    missing = check_interface_constraint(params, expected) if params is not None else []
    if missing:
        return VerifyResult(
            False, verdict=f"model dropped interfaces: {missing}",
            details={"stage": "interface", "missing": missing, "expected": expected},
            duration_s=time.monotonic() - t0,
        )

    if skip_size:
        return VerifyResult(
            True, verdict="model elaborates + interfaces honoured (size skipped)",
            details={"stage": "elaborate+interface"},
            duration_s=time.monotonic() - t0,
        )

    target_mhz = _read_target_clock_mhz(str(root))
    spec = _read_uarch_specs(str(root), [block]).get(block, "")
    size = _size_one_model(str(model_path), block, target_mhz, spec)
    feasible = size.get("feasible", True)
    return VerifyResult(
        bool(feasible),
        verdict=("model elaborates, interfaces honoured, feasible"
                 if feasible else f"model INFEASIBLE: {size.get('detail')}"),
        details={"stage": "size", "size": size},
        duration_s=time.monotonic() - t0,
    )


# ---------------------------------------------------------------------------
# verify_chip_model
# ---------------------------------------------------------------------------
def verify_chip_model(pr: str | Path) -> VerifyResult:
    """Composed-model-vs-golden byte-exact gate (composition.run_composition_gate)."""
    t0 = time.monotonic()
    root = Path(pr)
    try:
        from orchestrator.architecture import composition as _composition
    except Exception as exc:  # noqa: BLE001
        return VerifyResult(False, infra_error=True,
                            verdict=f"composition import failed: {exc}",
                            duration_s=time.monotonic() - t0)
    if not _composition.block_goldens_enabled():
        return VerifyResult(
            False, skipped=True,
            verdict="block goldens disabled (CORESMITH_BLOCK_GOLDENS off)",
            duration_s=time.monotonic() - t0,
        )
    gate_info: dict = {}
    try:
        violations = _composition.run_composition_gate(
            str(root), result_info=gate_info) or []
    except Exception as exc:  # noqa: BLE001
        return VerifyResult(False, infra_error=True,
                            verdict=f"composition gate errored: {exc}",
                            duration_s=time.monotonic() - t0)
    # Audit F2: an empty violations list is a PASS only when the gate actually
    # CHECKED something. A no-op (reference import failure, no stimulus, no
    # chip model, ...) must surface as SKIP -- the encoder run printed
    # "composed model == golden byte-exact" right after "reference import
    # failed: No module named 'reference_codec_vectors' -- no-op".
    if not violations and (gate_info.get("skipped")
                           or not gate_info.get("checked_vectors")):
        reason = gate_info.get("reason") or "gate checked no vectors"
        return VerifyResult(
            False, skipped=True,
            verdict=f"SKIP -- {reason} (no vector checked; NOT a pass)",
            details={"gate_info": gate_info},
            duration_s=time.monotonic() - t0,
        )
    passed = not violations
    return VerifyResult(
        passed,
        verdict=("composed model == golden byte-exact" if passed
                 else f"composition gate: {len(violations)} violation(s)"),
        details={"violations": violations, "gate_info": gate_info},
        duration_s=time.monotonic() - t0,
    )


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

    def _record(res: VerifyResult, tests=(None, None, None), first_div=None) -> VerifyResult:
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
        sim = run_simulation(block_spec, rtl_path, tbp, attempt or 1)
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

    # --- equivalence gate (unless suppressed) ---
    if not no_equiv:
        eq = run_block_equiv_gate(block, rtl_path, root, seed=seed)
        if eq["ran"]:
            if eq["failed_closed"] or (not eq["passed"] and not eq["skipped"]):
                return _record(VerifyResult(
                    False, verdict=f"RTL != model: {eq['reason']}",
                    details={"stage": "equiv", "equiv": eq},
                    log_path=log_path, duration_s=time.monotonic() - t0,
                ), tests=tests, first_div={"reason": eq["reason"]})
            if eq["skipped"]:
                return _record(VerifyResult(
                    True, verdict=f"sim passed; equiv skipped ({eq['reason']})",
                    details={"stage": "equiv", "equiv": eq},
                    log_path=log_path, duration_s=time.monotonic() - t0,
                ), tests=tests)
            return _record(VerifyResult(
                True, verdict=f"sim + equiv byte-exact ({eq['checked_vectors']} vectors)",
                details={"stage": "equiv", "equiv": eq},
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
    if not Path(rtl_path).exists():
        return VerifyResult(False, verdict=f"RTL not found: {rtl_path}",
                            duration_s=time.monotonic() - t0)

    # STAGE-REALIZATION GATE (pipeline-campaign): reject a single-cycle
    # combinational cloud (a collapsed multi-stage datapath) in milliseconds,
    # BEFORE paying the yosys elaboration timeout. Same deterministic census the
    # generate_rtl acceptance path uses; parity by construction. Best-effort and
    # env-gated (CORESMITH_STAGE_LINT=0 bypasses).
    try:
        from orchestrator.langgraph.rtl_stage_lint import (
            census_rtl,
            format_stage_lint_report,
            load_stage_map,
            stage_lint_enabled,
            stage_modules_enabled,
        )
        if stage_lint_enabled():
            _sm = load_stage_map(root, block)
            _sr = census_rtl(Path(rtl_path).read_text(), stage_map=_sm,
                             enforce_stage_modules=stage_modules_enabled())
            if not _sr.ok:
                return VerifyResult(
                    False,
                    verdict=(f"unsynthesizable combinational cloud: worst "
                             f"always-block = {_sr.worst_mul:,} effective "
                             f"multipliers > cap {_sr.mul_cap}"),
                    details={"stage": "stage_lint",
                             "worst_effective_multipliers": _sr.worst_mul,
                             "mul_cap": _sr.mul_cap,
                             "mul_violations": [b.name for b in _sr.mul_violations],
                             "census": format_stage_lint_report(_sr, block=block)},
                    duration_s=time.monotonic() - t0,
                )
    except Exception:  # noqa: BLE001 - never block the probe on a lint crash
        pass

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
        res = synthesize_block(block_spec, rtl_path, target_clock_mhz, attempt or 1)
        ok = bool(res.get("success"))
        _record(ok, ff=res.get("ff_count"), report_path=res.get("report_path", ""),
                probe="synth")
        return VerifyResult(
            ok, verdict=("synth OK" if ok else "synth FAILED"),
            details={"stage": "synth", "ff": res.get("ff_count"),
                     "area_um2": res.get("chip_area_um2"),
                     "gate_count": res.get("gate_count")},
            log_path=res.get("log_path", ""),
            duration_s=time.monotonic() - t0,
        )

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
    integration_check). A flock on ``sim_build/integration/.lock`` serializes
    concurrent chip sims.
    """
    t0 = time.monotonic()
    root = Path(pr)
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

    import fcntl
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
    lock_path = lock_dir / ".lock"
    with open(lock_path, "w") as lockf:
        try:
            fcntl.flock(lockf, fcntl.LOCK_EX)
        except OSError:
            pass
        res = run_integration_simulation(
            design, top_rtl, block_rtls, tbp, attempt or 1, sim_scope=sim_scope
        )

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
