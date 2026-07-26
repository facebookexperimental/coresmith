# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Measured-throughput enforcement (v3) -- close the plan->RTL cycles/op drift.

The v2 roofline (``perf_roofline``) checks throughput ONCE, pre-RTL, at plan
time: the uArch author declares a machine-readable ``perf`` block and the
roofline prices it against the PDK op delays. That catches a plan that declares
a number it already knows misses the cap -- but it says NOTHING about whether
the RTL the worker then writes actually HITS the declared number. The AES-accel
forensics proved the gap: the plan honestly declared an 11-cyc word-parallel key
schedule; the RTL worker (whose prompt had no throughput language) built a
word-serial 21-cyc REQUEST/RESPONSE FSM, and nothing re-measured. Delivered AES
graded 37 cyc/op vs 21 golden.

This module supplies the missing DELIVERY-TIME measurement + gate:

  * After a block PASSES functional DV, the block testbench emits a
    ``test_throughput_measure`` cocotb case that drives N representative ops and
    writes ``throughput_measured.json`` ({measured_cyc_per_op, n_ops}) to the
    sim dir.
  * ``evaluate_block_throughput`` parses that artifact, reads the DECLARED
    cyc/op the uArch author committed to (``perf_roofline``'s already-machine-
    parsed §6.1 ``declared_cyc_per_op`` -- the SAME parser), and FAILs the block
    when ``measured > declared x 1.1``. The failure feeds back a structured
    deficit report the diagnose loop treats like any DV failure.
  * A declared cyc/op with NO artifact fails CLOSED (the TB must emit the
    measurement) -- but a block with no declared cyc/op, or missing tooling, is
    recorded ``{applicable: False, reason}`` and NEVER crashes the run.

Gated by ``CORESMITH_MEASURED_THROUGHPUT_GATE`` (default ON). The squeeze loop
(``CORESMITH_THROUGHPUT_SQUEEZE``, default ON) is a separate, bounded
minimization pass that compares the measured number to the roofline PEAK (not
the declared number) and asks the worker to close toward it.

Task-agnostic: nothing is baked in. All numbers come from the block's own perf
model + the block's own measured artifact.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from . import perf_roofline as _pr

logger = logging.getLogger(__name__)

# A measured cyc/op may sit up to this factor above the declared number and
# still pass -- accounts for a cycle or two of legitimate framing the author
# rounded off. Above it, the RTL serialized where the plan said it would not.
THRESHOLD_RATIO = 1.1

# Artifact the block TB writes into its sim dir (see testbench_generator.md).
BLOCK_ARTIFACT = "throughput_measured.json"
# Artifact the deterministic integration TB writes (see bfm_lib/codegen.py).
CHIP_ARTIFACT = "integration_throughput.json"

_EPS = 1e-9


# --------------------------------------------------------------------------- #
# Env gates
# --------------------------------------------------------------------------- #
def measured_throughput_gate_enabled() -> bool:
    """Delivery-time measured-throughput gate on block DV (default ON).

    ``CORESMITH_MEASURED_THROUGHPUT_GATE=0`` (or false/no/off) disables it,
    restoring the v2 behavior (declared numbers are never re-measured).
    """
    return os.environ.get(
        "CORESMITH_MEASURED_THROUGHPUT_GATE", "1"
    ).strip().lower() not in {"0", "false", "no", "off", ""}


def throughput_squeeze_enabled() -> bool:
    """Post-block-DV cycle-minimization squeeze loop (default ON).

    Only ever fires on a block that already PASSED every gate but still sits
    above the roofline PEAK x 1.1 -- it asks the worker to close the remaining
    gap, bounded by ``squeeze_max_rounds``.
    """
    return os.environ.get(
        "CORESMITH_THROUGHPUT_SQUEEZE", "1"
    ).strip().lower() not in {"0", "false", "no", "off", ""}


def squeeze_max_rounds() -> int:
    """Max squeeze revision rounds per block (``CORESMITH_SQUEEZE_MAX_ROUNDS``,
    default 2). Clamped to [0, 5] so it can never run away."""
    try:
        n = int(os.environ.get("CORESMITH_SQUEEZE_MAX_ROUNDS", "2") or "2")
    except ValueError:
        n = 2
    return max(0, min(5, n))


# --------------------------------------------------------------------------- #
# Pure gate math (unit-testable, no IO)
# --------------------------------------------------------------------------- #
def threshold_for(declared_cyc_per_op: float) -> float:
    """The pass ceiling: ``declared x THRESHOLD_RATIO``."""
    return float(declared_cyc_per_op) * THRESHOLD_RATIO


def gate_math(measured: float, declared: float) -> dict[str, Any]:
    """Evaluate the measured-vs-declared gate. Pure; no IO.

    Returns ``{passed, ratio, threshold_cyc_per_op}``. ``passed`` is True iff
    ``measured <= declared x 1.1``. A non-positive declared number is treated as
    "no meaningful budget" -> passes (the caller records N/A upstream).
    """
    m = float(measured)
    d = float(declared)
    if d <= 0:
        return {"passed": True, "ratio": None, "threshold_cyc_per_op": None}
    thr = threshold_for(d)
    return {
        "passed": m <= thr + _EPS,
        "ratio": round(m / d, 4),
        "threshold_cyc_per_op": round(thr, 4),
    }


# --------------------------------------------------------------------------- #
# Declared / peak lookup  (REUSE the perf_roofline parser)
# --------------------------------------------------------------------------- #
def _perf_model(project_root: str | os.PathLike, block_name: str) -> dict | None:
    """The block's perf model dict, reusing perf_roofline's §6.1 parser.

    Prefers the already-emitted ``perf_model.json``; when absent (e.g. the v2
    roofline emit was gated off) it computes it on the fly from the uArch spec's
    ``perf`` block, so the measured gate does not depend on the roofline emit
    flag being on. Returns None when the block declares no ``perf`` block.
    """
    m = _pr.read_perf_model(project_root, block_name)
    if m:
        return m
    try:
        return _pr.perf_model_for_block(project_root, block_name)
    except Exception:  # noqa: BLE001 - best-effort, never raises
        return None


def declared_cyc_per_op(project_root: str | os.PathLike,
                        block_name: str) -> float | None:
    """The uArch author's declared §6.1 cyc/op for this block, or None."""
    m = _perf_model(project_root, block_name)
    if not m:
        return None
    v = m.get("declared_cyc_per_op")
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def roofline_peak_cyc_per_op(project_root: str | os.PathLike,
                             block_name: str) -> float | None:
    """The roofline PEAK cyc/op (the squeeze loop's target), or None."""
    m = _perf_model(project_root, block_name)
    if not m:
        return None
    v = m.get("cyc_per_op_peak")
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def binding_constraint_text(project_root: str | os.PathLike,
                            block_name: str) -> str:
    """A one-line description of the block's binding recurrence/resource."""
    m = _perf_model(project_root, block_name)
    if not m:
        return ""
    bc = m.get("binding_constraint") or {}
    detail = bc.get("detail", "")
    lever = bc.get("lever", "")
    if not detail and not lever:
        return ""
    return f"{bc.get('type', '')} {bc.get('name', '')}: {detail}; lever: {lever}"


# --------------------------------------------------------------------------- #
# Artifact parsing
# --------------------------------------------------------------------------- #
def _rtl_mtime(rtl_path: str | os.PathLike | None) -> float | None:
    """Modification time (epoch s) of the block's current RTL, or None."""
    if not rtl_path:
        return None
    try:
        p = Path(rtl_path)
        return p.stat().st_mtime if p.exists() else None
    except OSError:
        return None


def _read_artifact(sim_dir: str | os.PathLike, name: str,
                   min_mtime: float | None = None) -> dict | None:
    """Locate + parse a throughput artifact from a sim dir (never raises).

    Accepts either ``measured_cyc_per_op`` or ``cyc_per_op`` as the key so a
    slightly-off TB still measures. Returns ``{measured_cyc_per_op, n_ops}`` or
    None when absent / unparseable.

    engine-v31 step 4 -- FRESHNESS: when ``min_mtime`` is given (the block's
    current RTL mtime), an artifact written BEFORE the RTL was last modified is
    STALE (a prior DV run measured different RTL) and is IGNORED (returns None,
    same as absent). This closes the retry-path bug where a stale
    ``throughput_measured.json`` (old cyc/op) was read before the re-validate DV
    had re-measured the edited RTL -- so the gate must re-measure, not trust it.
    """
    try:
        sd = Path(sim_dir)
        cand = sd / name
        if not cand.exists():
            hits = list(sd.rglob(name))
            if not hits:
                return None
            cand = hits[0]
        if min_mtime is not None:
            try:
                art_mtime = cand.stat().st_mtime
            except OSError:
                art_mtime = None
            # +1s slop: an artifact written in the SAME whole second as the RTL
            # edit is treated as fresh (mtime granularity), only strictly-older
            # artifacts are rejected as stale.
            if art_mtime is not None and art_mtime < (min_mtime - 1.0):
                logger.info(
                    "ignoring STALE throughput artifact %s (mtime %.0f < RTL "
                    "mtime %.0f) -- re-measuring", cand, art_mtime, min_mtime)
                return None
        doc = json.loads(cand.read_text())
        if not isinstance(doc, dict):
            return None
        m = doc.get("measured_cyc_per_op")
        if m is None:
            m = doc.get("cyc_per_op")
        if m is None:
            return None
        n = doc.get("n_ops")
        return {
            "measured_cyc_per_op": float(m),
            "n_ops": int(n) if isinstance(n, (int, float)) else None,
        }
    except Exception:  # noqa: BLE001
        return None


def read_throughput_artifact(
    sim_dir: str | os.PathLike,
    rtl_path: str | os.PathLike | None = None,
) -> dict | None:
    """Parse the block ``throughput_measured.json`` artifact (or None).

    When ``rtl_path`` is given, an artifact older than the RTL is ignored as
    stale (engine-v31 step 4) so a retry re-measures instead of reading a prior
    run's number.
    """
    return _read_artifact(sim_dir, BLOCK_ARTIFACT, _rtl_mtime(rtl_path))


def read_chip_throughput_artifact(sim_dir: str | os.PathLike) -> dict | None:
    """Parse the chip ``integration_throughput.json`` artifact (or None)."""
    return _read_artifact(sim_dir, CHIP_ARTIFACT)


# --------------------------------------------------------------------------- #
# Block-level gate + record  (one dict serves gate AND persistence)
# --------------------------------------------------------------------------- #
def _na(reason: str, declared: float | None = None) -> dict:
    return {
        "gate": "measured_throughput", "scope": "block",
        "applicable": False, "passed": None,
        "measured_cyc_per_op": None, "declared_cyc_per_op": declared,
        "threshold_cyc_per_op": None, "ratio": None, "n_ops": None,
        "artifact_missing": False, "reason": reason, "report": "",
    }


def _deficit_report(block_name: str, measured: float, declared: float,
                    threshold: float, ratio: float | None,
                    binding: str) -> str:
    lines = [
        f"MEASURED-THROUGHPUT GATE: block '{block_name}' runs "
        f"{measured:g} cyc/op MEASURED in DV, but its uArch spec (§6.1) "
        f"DECLARED {declared:g} cyc/op -- over the {threshold:g} ceiling "
        f"(declared x {THRESHOLD_RATIO}"
        + (f", ratio {ratio:g}x" if ratio is not None else "") + "). FAIL.",
        "This is an RTL performance defect, NOT a testbench weakness: the plan "
        "committed to a rate the RTL did not implement (the classic plan->RTL "
        "drift -- a schedule declared word-parallel / II=1 was serialized "
        "through a shared resource, or a compile-time-static sequence was "
        "wrapped in a per-iteration REQUEST/RESPONSE handshake).",
        "FIX THE RTL to hit the declared rate:",
        "  - Do NOT handshake a compile-time-enumerable sequence (fixed round "
        "order, fixed tap/coefficient sweep): pre-stage locally and drive it "
        "from a counter/FSM (skills/srdy_drdy.md).",
        "  - A schedule the spec declared word-parallel / II=1 MUST be built "
        "that way -- do not serialize it through one shared datapath the spec "
        "does not share.",
    ]
    if binding:
        lines.append(f"  - Binding constraint from perf_model.json: {binding}.")
    return "\n".join(lines)


def evaluate_block_throughput(project_root: str | os.PathLike, block_name: str,
                              sim_dir: str | os.PathLike,
                              rtl_path: str | os.PathLike | None = None) -> dict:
    """Measured-throughput gate + persistence record for one block-DV sim.

    Returns a single dict that serves BOTH the gate (``applicable`` +
    ``passed``) and the signoff report (persisted verbatim to
    ``blocks/<b>/throughput.json``). Shape:

      applicable=False, passed=None  -> N/A (gate off / no declared cyc/op).
                                        Never demotes a block.
      applicable=True,  passed=False -> the block is too slow (measured over
                                        the ceiling) OR the artifact is missing
                                        (``artifact_missing``). Demotes the DV.
      applicable=True,  passed=True  -> measured is within the ceiling.

    Never raises: any plumbing error becomes an ``applicable:False`` record.
    """
    try:
        if not measured_throughput_gate_enabled():
            return _na("measured-throughput gate disabled "
                       "(CORESMITH_MEASURED_THROUGHPUT_GATE=0)")
        declared = declared_cyc_per_op(project_root, block_name)
        if declared is None or declared <= 0:
            return _na("block declares no §6.1 cyc/op (no machine-readable "
                       "`perf` block) -- nothing to measure against")
        art = read_throughput_artifact(sim_dir, rtl_path)
        threshold = round(threshold_for(declared), 4)
        if art is None:
            # Declared a rate but the TB emitted no measurement: fail CLOSED so
            # the TB is regenerated with the required test_throughput_measure
            # case (mirrors the coverage gate's fail-closed-on-weak-TB path).
            report = (
                f"MEASURED-THROUGHPUT GATE: block '{block_name}' declares "
                f"{declared:g} cyc/op (§6.1) but its testbench produced no "
                f"'{BLOCK_ARTIFACT}' throughput artifact. The block TB MUST "
                "include a `test_throughput_measure` cocotb case that drives N "
                "representative ops through the declared interface, counts clk "
                "edges per op in steady state (excluding the first-op fill), and "
                f"writes {{\"measured_cyc_per_op\": <float>, \"n_ops\": <int>}} to "
                f"'{BLOCK_ARTIFACT}' in the sim cwd. Regenerate the TB to add it."
            )
            return {
                "gate": "measured_throughput", "scope": "block",
                "applicable": True, "passed": False,
                "measured_cyc_per_op": None, "declared_cyc_per_op": declared,
                "threshold_cyc_per_op": threshold, "ratio": None, "n_ops": None,
                "artifact_missing": True,
                "reason": "no throughput artifact (TB missing "
                          "test_throughput_measure)",
                "report": report,
            }
        measured = art["measured_cyc_per_op"]
        gm = gate_math(measured, declared)
        passed = bool(gm["passed"])
        rec = {
            "gate": "measured_throughput", "scope": "block",
            "applicable": True, "passed": passed,
            "measured_cyc_per_op": round(measured, 4),
            "declared_cyc_per_op": declared,
            "threshold_cyc_per_op": threshold,
            "ratio": gm["ratio"], "n_ops": art.get("n_ops"),
            "artifact_missing": False, "reason": "", "report": "",
        }
        if not passed:
            rec["reason"] = (f"measured {measured:g} > declared {declared:g} "
                             f"x {THRESHOLD_RATIO} = {threshold:g}")
            rec["report"] = _deficit_report(
                block_name, measured, declared, threshold, gm["ratio"],
                binding_constraint_text(project_root, block_name),
            )
        return rec
    except Exception as exc:  # noqa: BLE001 - the gate must never crash a DV run
        return _na(f"throughput gate evaluation error: {exc}")


def squeeze_needed(project_root: str | os.PathLike, block_name: str,
                   measured_cyc_per_op: float | None) -> dict | None:
    """Should the squeeze loop fire for this (passing) block?

    Fires only when a measured number exists AND it sits above the roofline
    PEAK x 1.1. Returns ``{peak, threshold, measured, ratio, binding}`` when a
    squeeze is warranted, else None (already at/below peak x 1.1, or no peak).
    """
    if measured_cyc_per_op is None:
        return None
    peak = roofline_peak_cyc_per_op(project_root, block_name)
    if peak is None or peak <= 0:
        return None
    thr = round(peak * THRESHOLD_RATIO, 4)
    m = float(measured_cyc_per_op)
    if m <= thr + _EPS:
        return None
    return {
        "peak_cyc_per_op": peak,
        "threshold_cyc_per_op": thr,
        "measured_cyc_per_op": round(m, 4),
        "ratio": round(m / peak, 4),
        "binding": binding_constraint_text(project_root, block_name),
    }


def format_squeeze_request(block_name: str, need: dict) -> str:
    """The revision request handed to the RTL worker for a squeeze round.

    Names the measured number, the roofline peak (the target), and the binding
    constraint/lever from perf_model.json. Objective: reduce cycles toward the
    peak WITHOUT regressing function/area/Fmax.
    """
    lines = [
        f"THROUGHPUT SQUEEZE (round revision): block '{block_name}' is "
        "FUNCTIONALLY CORRECT and passed every gate, but it runs "
        f"{need.get('measured_cyc_per_op')} cyc/op -- above the roofline PEAK "
        f"of {need.get('peak_cyc_per_op')} cyc/op (target ceiling "
        f"{need.get('threshold_cyc_per_op')} = peak x {THRESHOLD_RATIO}).",
        "OBJECTIVE: reduce cycles/op toward the peak. Keep the design "
        "BYTE-EXACT to the golden model (an equivalence gate re-checks it) and "
        "do NOT regress area or Fmax.",
        "How to close the gap (pick what the binding constraint indicates):",
        "  - Widen a serial loop to K>=2 PIPELINED lanes (registered, II=1) "
        "where the spec's schedule is parallel -- do not add a combinational "
        "unroll.",
        "  - Remove per-iteration REQUEST/RESPONSE handshakes around a "
        "compile-time-static sequence: pre-stage locally, drive from a "
        "counter/FSM.",
        "  - Remove throwaway bridge/wait states between registered handshakes; "
        "accept a command on the final cycle of its write; drive DONE/status "
        "pins combinationally from the status register (decomposition-tax "
        "rules).",
    ]
    if need.get("binding"):
        lines.append(f"  - Binding constraint (from perf_model.json): "
                     f"{need['binding']}.")
    lines.append("Make a TARGETED edit to the existing RTL; do not rewrite from "
                 "scratch unless the schedule is structurally wrong.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Chip-level (integration) measured throughput
# --------------------------------------------------------------------------- #
def chip_throughput_budget(project_root: str | os.PathLike,
                           state: dict | None = None
                           ) -> tuple[float | None, str, str]:
    """Resolve a chip-level cyc/op budget for the integration gate (best-effort).

    Sources, in priority order (deterministic, never raises):
      1. an explicit ``chip_cyc_per_op_budget`` in run state,
      2. a persisted ``.coresmith/chip_perf_model.json`` with a
         ``declared_cyc_per_op`` / ``cyc_per_op_peak`` field,
      3. None -- no chip-level budget is resolvable, so the integration gate
         MEASURES + records but does not gate (fail-open N/A).

    Returns ``(budget_or_None, source, note)``.
    """
    try:
        if state:
            b = state.get("chip_cyc_per_op_budget")
            if isinstance(b, (int, float)) and b > 0:
                return float(b), "state", "chip cyc/op budget from run state"
        p = Path(project_root) / ".coresmith" / "chip_perf_model.json"
        if p.exists():
            doc = json.loads(p.read_text())
            if isinstance(doc, dict):
                for k in ("declared_cyc_per_op", "cyc_per_op_peak",
                          "perf_req_cyc_per_op"):
                    v = doc.get(k)
                    if isinstance(v, (int, float)) and v > 0:
                        return float(v), f"chip_perf_model.{k}", str(
                            doc.get("notes", ""))
        # 3. A budget declared directly in the requirements doc (audit F8):
        #    the reference codec encoder spec says "budget **500000 cyc/frame**" yet the
        #    chip record carried budget_source="none" -- the declared number
        #    must at least be RESOLVED and reported, even when no chip-window
        #    measurement exists to gate it against.
        import re as _re
        for doc_path in (Path(project_root) / "inputs" / "requirements.md",
                         Path(project_root) / "requirements.md"):
            if not doc_path.exists():
                continue
            try:
                text = doc_path.read_text(errors="ignore")
            except OSError:
                continue
            m = _re.search(
                r"budget\s*[*_`]*\s*([\d][\d,_]*)\s*[*_`]*\s*"
                r"cyc(?:les)?\s*/\s*(frame|op|block|sample|packet)",
                text, _re.IGNORECASE)
            if m:
                val = float(m.group(1).replace(",", "").replace("_", ""))
                unit = m.group(2).lower()
                if val > 0:
                    return val, f"requirements (cyc/{unit})", (
                        f"declared budget {val:g} cycles per {unit} in "
                        f"{doc_path.name}")
    except Exception:  # noqa: BLE001
        pass
    return None, "none", "no chip-level throughput budget declared"


def evaluate_chip_throughput(project_root: str | os.PathLike,
                             sim_dir: str | os.PathLike,
                             state: dict | None = None) -> dict:
    """Chip-level measured-throughput record from an integration-DV sim dir.

    Reads the deterministic-BFM cycle-accounting artifact, resolves a chip
    budget (best-effort), and gates ``measured <= budget x 1.1`` WHEN a budget
    exists. Always records ``measured_cyc_per_op_chip``. Never raises.
    """
    base = {
        "gate": "measured_throughput", "scope": "chip",
        "applicable": False, "passed": None,
        "measured_cyc_per_op_chip": None, "n_ops": None,
        "budget_cyc_per_op": None, "threshold_cyc_per_op": None,
        "ratio": None, "budget_source": "none",
        "reason": "", "report": "",
        "grader_window": "op-start-committed -> DONE visible on status pin",
    }
    try:
        budget, source, note = chip_throughput_budget(project_root, state)
        art = read_chip_throughput_artifact(sim_dir)
        if art is None:
            # Audit F8: still RESOLVE + record a declared budget so the report
            # shows "declared N cyc/frame, UNMEASURED" (a measurement gap)
            # instead of "budget none" -- the declared requirement must not
            # vanish just because the chip-window artifact is missing.
            base["reason"] = ("no integration throughput artifact "
                              "(deterministic BFM off / not QSPI-slave)")
            if budget is not None:
                base["budget_cyc_per_op"] = budget
                base["budget_source"] = source
                base["reason"] += (
                    f"; declared budget {budget:g} cyc/op ({source}) is "
                    "UNMEASURED -- a measurement gap, not a pass")
            return base
        measured = art["measured_cyc_per_op"]
        base["measured_cyc_per_op_chip"] = round(measured, 4)
        base["n_ops"] = art.get("n_ops")
        base["budget_source"] = source
        if budget is None:
            base["reason"] = f"measured only ({note}); no chip budget to gate"
            return base
        gm = gate_math(measured, budget)
        base["applicable"] = True
        base["passed"] = bool(gm["passed"])
        base["budget_cyc_per_op"] = budget
        base["threshold_cyc_per_op"] = gm["threshold_cyc_per_op"]
        base["ratio"] = gm["ratio"]
        if not base["passed"]:
            base["reason"] = (
                f"chip measured {measured:g} cyc/op > budget {budget:g} "
                f"x {THRESHOLD_RATIO} = {gm['threshold_cyc_per_op']:g} "
                f"({source})")
            base["report"] = (
                "INTEGRATION MEASURED-THROUGHPUT GATE: the chip-level op window "
                f"(START committed -> DONE) took {measured:g} cyc/op, over the "
                f"{gm['threshold_cyc_per_op']:g} ceiling. A block serialized "
                "where the chip budget assumed it would not.")
        return base
    except Exception as exc:  # noqa: BLE001
        base["reason"] = f"chip throughput evaluation error: {exc}"
        return base


def persist_chip_throughput(project_root: str | os.PathLike,
                            record: dict) -> None:
    """Write the chip throughput record to ``.coresmith/chip_throughput.json``.

    The final-report node reads this git-visible artifact. Never raises.
    """
    try:
        p = Path(project_root) / ".coresmith" / "chip_throughput.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(record, indent=2))
    except Exception:  # noqa: BLE001
        pass


def read_chip_throughput(project_root: str | os.PathLike) -> dict | None:
    """Read the persisted chip throughput record (or None)."""
    try:
        p = Path(project_root) / ".coresmith" / "chip_throughput.json"
        if p.exists():
            return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        pass
    return None


__all__ = [
    "BLOCK_ARTIFACT",
    "CHIP_ARTIFACT",
    "THRESHOLD_RATIO",
    "binding_constraint_text",
    "chip_throughput_budget",
    "declared_cyc_per_op",
    "evaluate_block_throughput",
    "evaluate_chip_throughput",
    "format_squeeze_request",
    "gate_math",
    "measured_throughput_gate_enabled",
    "persist_chip_throughput",
    "read_chip_throughput",
    "read_chip_throughput_artifact",
    "read_throughput_artifact",
    "roofline_peak_cyc_per_op",
    "squeeze_max_rounds",
    "squeeze_needed",
    "threshold_for",
    "throughput_squeeze_enabled",
]
