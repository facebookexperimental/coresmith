# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Quantitative fidelity budget for the composition gate (microarchitecture
restructure, step 2).

The legacy composition gate is binary: the composed model's output is either
byte-exact / accepted by a declared ``accept(expected, observed) -> bool``
predicate, or it fails. That cannot tell a 0.3 dB rounding approximation from a
21 dB catastrophe -- both are simply "not equal". The two-golden design needs a
QUANTITATIVE fidelity number with a budget, so that a model which intentionally
derates within budget passes (and is recorded), while a broken one fails, and a
large-but-acceptable derate can be flagged for chip-lead escalation.

This module is design-agnostic. A design declares:

  * a fidelity METRIC  -- ``fidelity(expected, observed) -> float`` (a score;
    higher is better unless the budget says ``direction="lower"`` for an error
    metric). For a codec this is decode + PSNR/SSIM of observed vs the reference
    output; for a filter ULP/SNR; for an MCU it is left undeclared (byte-exact).
  * a fidelity BUDGET  -- ``{floor, escalate_floor?, direction, ideal?}``.

Verdict (direction="higher"; mirrored for "lower"):

    measured >= escalate_floor  -> PASS, clean
    floor <= measured <         -> PASS, but ESCALATE (derate acceptable but
        escalate_floor               large enough to need chip-lead sign-off)
    measured <  floor           -> FAIL (below budget)

All gated behind ``CORESMITH_FIDELITY_GATE`` (default OFF); when off or when no
metric is declared, the gate keeps its existing binary behavior.
"""
from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def fidelity_gate_enabled() -> bool:
    """True when the quantitative fidelity tier is active.

    Opt-in (default off) until validated end-to-end. ``CORESMITH_FIDELITY_GATE=1``
    enables it; when on AND a fidelity metric is declared, the composition gate
    scores the composed output against the budget instead of the binary
    accept/equivalence tiers.
    """
    from orchestrator.profile import ensure_applied, flag_enabled
    ensure_applied()
    return flag_enabled("CORESMITH_FIDELITY_GATE", default=False)


def resolve_fidelity_metric(
    project_root: str,
) -> Callable[[Any, Any], float] | None:
    """Resolve a declared ``fidelity(expected, observed) -> float`` callable.

    Resolution order (first hit wins), mirroring
    :func:`composition.resolve_functional_acceptance`:

    1. ``CORESMITH_FIDELITY_METRIC`` env var -- ``path.py`` (optionally
       ``path.py:funcname``; default func name ``fidelity``).
    2. The ERS ``validation_kpis`` declaring a ``fidelity_fn`` entry.

    Returns the callable, or ``None`` when no fidelity metric is declared.
    """
    from orchestrator.architecture.composition import (
        _load_acceptance_callable,
        _load_validation_kpis,
    )

    env_path = os.environ.get("CORESMITH_FIDELITY_METRIC", "").strip()
    if env_path:
        fn = _load_acceptance_callable(env_path, project_root, default_func="fidelity")
        if fn is not None:
            return fn
        logger.warning(
            "fidelity gate: CORESMITH_FIDELITY_METRIC=%r did not resolve to a "
            "callable fidelity(expected, observed)",
            env_path,
        )

    for kpi in _load_validation_kpis(project_root):
        decl = kpi.get("fidelity_fn")
        if isinstance(decl, str) and decl.strip():
            fn = _load_acceptance_callable(
                decl.strip(), project_root, default_func="fidelity"
            )
            if fn is not None:
                return fn
            logger.warning(
                "fidelity gate: validation_kpis fidelity_fn %r did not resolve", decl
            )
    return None


def resolve_fidelity_budget(project_root: str) -> dict:
    """Resolve the fidelity budget dict.

    Keys (all optional): ``floor`` (pass/fail threshold), ``escalate_floor``
    (stricter threshold; passing ``floor`` but failing this needs chip-lead
    sign-off), ``direction`` (``"higher"`` [default] for score metrics, or
    ``"lower"`` for error metrics), ``ideal`` (the reference's own score, used to
    report ``derate_pct``), ``metric`` (display name).

    Resolution order (first hit wins):

    1. ``CORESMITH_FIDELITY_BUDGET`` env var -- inline JSON object, or a path to
       a ``.json`` file.
    2. The ERS ``validation_kpis``: a ``fidelity`` sub-dict, or flat
       ``fidelity_floor`` / ``fidelity_escalate_floor`` / ``fidelity_direction``
       / ``fidelity_ideal`` fields.

    Returns ``{}`` when no budget is declared.
    """
    from orchestrator.architecture.composition import _load_validation_kpis

    env_b = os.environ.get("CORESMITH_FIDELITY_BUDGET", "").strip()
    if env_b:
        budget = _parse_budget_blob(env_b, project_root)
        if budget:
            return budget

    for kpi in _load_validation_kpis(project_root):
        sub = kpi.get("fidelity")
        if isinstance(sub, dict) and sub:
            return _normalize_budget(sub)
        flat = {}
        if isinstance(kpi.get("fidelity_floor"), (int, float)):
            flat["floor"] = kpi["fidelity_floor"]
        if isinstance(kpi.get("fidelity_escalate_floor"), (int, float)):
            flat["escalate_floor"] = kpi["fidelity_escalate_floor"]
        if isinstance(kpi.get("fidelity_direction"), str):
            flat["direction"] = kpi["fidelity_direction"]
        if isinstance(kpi.get("fidelity_ideal"), (int, float)):
            flat["ideal"] = kpi["fidelity_ideal"]
        if flat:
            return _normalize_budget(flat)
    return {}


def _parse_budget_blob(blob: str, project_root: str | None) -> dict:
    """Parse ``CORESMITH_FIDELITY_BUDGET`` as inline JSON or a path to a .json."""
    txt = blob
    # path to a .json file?
    if not blob.lstrip().startswith("{"):
        p = Path(blob)
        if not p.is_absolute() and project_root:
            cand = Path(project_root) / p
            if cand.exists():
                p = cand
        if p.is_file():
            try:
                txt = p.read_text(encoding="utf-8")
            except OSError:
                return {}
        else:
            return {}
    try:
        data = json.loads(txt)
    except json.JSONDecodeError:
        logger.warning("fidelity gate: CORESMITH_FIDELITY_BUDGET is not valid JSON")
        return {}
    return _normalize_budget(data) if isinstance(data, dict) else {}


def _normalize_budget(b: dict) -> dict:
    out: dict = {}
    for k in ("floor", "escalate_floor", "ideal"):
        v = b.get(k)
        if isinstance(v, (int, float)):
            out[k] = float(v)
    direction = str(b.get("direction", "higher")).strip().lower()
    out["direction"] = "lower" if direction in {"lower", "less", "min"} else "higher"
    if isinstance(b.get("metric"), str):
        out["metric"] = b["metric"]
    return out


def _better_or_equal(measured: float, threshold: float, direction: str) -> bool:
    return measured >= threshold if direction == "higher" else measured <= threshold


def compute_fidelity_derate(
    project_root: str, expected: Any, observed: Any
) -> dict | None:
    """Score ``observed`` against ``expected`` and the declared budget.

    Returns a verdict dict, or ``None`` when no fidelity metric is declared (the
    caller then keeps the legacy binary tiers). The dict carries:
    ``metric``, ``measured``, ``floor``, ``escalate_floor``, ``direction``,
    ``ideal``, ``derate_pct`` (vs ``ideal`` when given), ``within_budget`` (passes
    ``floor``), ``escalate`` (passes ``floor`` but fails ``escalate_floor``).
    """
    metric = resolve_fidelity_metric(project_root)
    if metric is None:
        return None
    budget = resolve_fidelity_budget(project_root)
    try:
        measured = float(metric(expected, observed))
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "fidelity gate: metric raised %s: %s -- treating as worst case",
            type(exc).__name__, exc,
        )
        measured = float("-inf") if budget.get("direction", "higher") == "higher" else float("inf")

    direction = budget.get("direction", "higher")
    floor = budget.get("floor")
    escalate_floor = budget.get("escalate_floor")
    ideal = budget.get("ideal")

    within = True if floor is None else _better_or_equal(measured, floor, direction)
    escalate = False
    if within and escalate_floor is not None:
        escalate = not _better_or_equal(measured, escalate_floor, direction)

    derate_pct: float | None = None
    if isinstance(ideal, (int, float)) and ideal != 0:
        gap = (ideal - measured) if direction == "higher" else (measured - ideal)
        derate_pct = max(0.0, 100.0 * gap / abs(ideal))

    return {
        "metric": budget.get("metric", "fidelity"),
        "measured": measured,
        "floor": floor,
        "escalate_floor": escalate_floor,
        "direction": direction,
        "ideal": ideal,
        "derate_pct": derate_pct,
        "within_budget": bool(within),
        "escalate": bool(escalate),
    }


_LEDGER_NAME = "derate_ledger.json"


def write_derate_ledger(
    project_root: str,
    fid: dict | None,
    *,
    byte_exact: bool,
    block: str = "_integrated",
) -> None:
    """Append/refresh the integrated derate entry in ``.coresmith/derate_ledger.json``.

    Best-effort; never raises into the gate. The ledger is the Phase-2 handoff
    artifact -- it records what fidelity was traded (and whether it is within
    budget / needs escalation) so the frontend and the final report can confirm
    the shipped design's KPI against the original intent.
    """
    try:
        root = Path(project_root) / ".coresmith"
        root.mkdir(parents=True, exist_ok=True)
        p = root / _LEDGER_NAME
        doc: dict = {}
        if p.exists():
            try:
                doc = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                doc = {}
        if not isinstance(doc, dict):
            doc = {}
        entries = doc.get("entries")
        if not isinstance(entries, list):
            entries = []
        entry = {
            "block": block,
            "byte_exact": bool(byte_exact),
        }
        if fid is not None:
            entry.update(
                {
                    "metric": fid.get("metric"),
                    "measured": fid.get("measured"),
                    "floor": fid.get("floor"),
                    "escalate_floor": fid.get("escalate_floor"),
                    "direction": fid.get("direction"),
                    "derate_pct": fid.get("derate_pct"),
                    "within_budget": fid.get("within_budget"),
                    "escalate": fid.get("escalate"),
                }
            )
        # replace any prior entry for the same block (idempotent re-runs)
        entries = [e for e in entries if isinstance(e, dict) and e.get("block") != block]
        entries.append(entry)
        doc["entries"] = entries
        doc["integrated_within_budget"] = bool(
            byte_exact or (fid is not None and fid.get("within_budget"))
        )
        doc["integrated_escalate"] = bool(
            (not byte_exact) and fid is not None and fid.get("escalate")
        )
        p.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        logger.warning("fidelity gate: could not write derate ledger: %s", exc)


def read_derate_escalation(project_root: str) -> dict | None:
    """Return the integrated derate entry that needs chip-lead sign-off, else None.

    The composition gate passes functionally at a within-budget derate, but when
    that derate is above the ``escalate_floor`` (``integrated_escalate``) and has
    not yet been signed off, the gate node parks a ``derate_signoff`` interrupt
    (microarch step 3 -- derate authority + escalation). Returns None once signed
    off, so a re-run of the gate does not re-prompt.
    """
    try:
        p = Path(project_root) / ".coresmith" / _LEDGER_NAME
        if not p.is_file():
            return None
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(doc, dict) or not doc.get("integrated_escalate"):
        return None
    if doc.get("integrated_signed_off"):
        return None
    entries = doc.get("entries") if isinstance(doc.get("entries"), list) else []
    for e in entries:
        if isinstance(e, dict) and e.get("block") == "_integrated" and e.get("escalate"):
            return e
    for e in entries:  # fall back to any escalating entry
        if isinstance(e, dict) and e.get("escalate"):
            return e
    return None


def mark_derate_signed_off(project_root: str) -> None:
    """Record chip-lead approval of the integrated derate in the ledger so the
    gate does not re-prompt on a subsequent run. Best-effort."""
    try:
        p = Path(project_root) / ".coresmith" / _LEDGER_NAME
        if not p.is_file():
            return
        doc = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(doc, dict):
            doc["integrated_signed_off"] = True
            p.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        logger.warning("fidelity gate: could not mark derate signed off: %s", exc)
