# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
LangGraph StateGraph for the full ASIC pipeline.

Two-level architecture:
  1. **Block Subgraph** (``BlockState``) -- self-contained lifecycle for a
     single block: uarch spec -> RTL (with lint) -> testbench (with sim) ->
     synthesize, with a diagnose/retry loop and human escalation.
  2. **Orchestrator Graph** (``OrchestratorState``) -- iterates through
     tiers and uses ``Send()`` to fan out all blocks within each tier
     for parallel execution.

Block lifecycle (simplified)::

    init -> uarch_spec -> review -> generate_rtl (lint built-in)
         -> generate_testbench (sim + local TB fix loop)
         -> synthesize -> block_done
                 |
              diagnose -> decide -> generate_rtl (direct retry)

Key design decisions:
  - Lint is folded into generate_rtl: run Verilator lint after RTL
    generation, with a local LLM fix loop before escalating.
  - Simulate is folded into generate_testbench: run cocotb sim after
    TB generation, with a local LLM fix loop for testbench bugs.
    Only escalates to diagnose for serious RTL bugs.
  - decide routes directly to generate_rtl (no intermediate
    increment_attempt node).

Tier N+1 does not start until every block in tier N completes.  Interrupts
in any block pause the entire graph (natural LangGraph behaviour).

Within a tier, blocks run in parallel: ``fan_out_tier`` emits one
``Send("process_block", ...)`` per block and LangGraph schedules every
async branch concurrently via ``asyncio.gather``.  Each per-block
``ClaudeLLM.call`` then dispatches the blocking CLI subprocess into the
default thread executor (``loop.run_in_executor`` in ``call``), so two
concurrent blocks do not serialise on the GIL or on a single Popen --
verified empirically: 3 parallel CLI calls finish in 1× wall-time.

Usage::

    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    async with AsyncSqliteSaver.from_conn_string(".coresmith/pipeline_checkpoint.db") as cp:
        graph = build_pipeline_graph(checkpointer=cp)
        result = await graph.ainvoke(initial_state, config)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import operator
import os
import re
import time as _time
from pathlib import Path
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send, interrupt
from opentelemetry import trace

from orchestrator.langgraph.event_stream import write_graph_event
from orchestrator.langgraph.integration_helpers import (
    discover_block_rtl,
    generate_integration_testbench,
    generate_validation_testbench,
    lint_top_level,
    load_architecture_connections,
    module_for_block,
    parse_verilog_ports,
    run_integration_simulation,
)
from orchestrator.langgraph.pipeline_helpers import (
    CYAN,
    GREEN,
    PROJECT_ROOT,
    RED,
    YELLOW,
    create_golden_model_wrapper,
    diagnose_failure,
    fix_lint_errors,
    fix_synth_errors,
    fix_testbench_errors,
    generate_rtl,
    generate_testbench,
    generate_uarch_spec,
    lint_rtl,
    log,
    run_simulation,
    synthesize_block,
)
from orchestrator.utils import smart_truncate

_tracer = trace.get_tracer("coresmith.langgraph.pipeline_graph")

# Maximum local LLM fix attempts before escalating to diagnose.
# Each agent node (lint, synthesize) tries to self-heal up to this
# many times before giving up and routing to the diagnose lead.
MAX_LOCAL_RETRIES = 2


def _normalize_constraint(text: str) -> str:
    """Normalize constraint text for dedup comparison.

    Fix #13: Lowercases, strips punctuation, collapses whitespace so
    semantically identical constraints worded differently are deduplicated.
    """
    import re as _re
    text = text.lower().strip()
    text = _re.sub(r"\s+", " ", text)
    text = _re.sub(r"[^\w\s]", "", text)
    return text


def _normalize_ws(text: str) -> str:
    """Collapse all whitespace to single spaces."""
    import re as _re
    return _re.sub(r"\s+", " ", text.strip())


def _fuzzy_replace(
    spec: str, original: str, replacement: str
) -> tuple[str, str]:
    """Replace *original* in *spec* with *replacement* using progressively
    looser matching.

    Fix #12: Handles LLM whitespace variations and minor paraphrasing.

    Returns:
        ``(new_spec, method)`` where *method* is ``"exact"``, ``"whitespace"``,
        ``"fuzzy"`` or ``""`` (no match found).
    """
    # 1. Exact match
    if original in spec:
        return spec.replace(original, replacement, 1), "exact"

    # 2. Whitespace-normalised match via sliding window
    norm_orig = _normalize_ws(original)
    lines = spec.split("\n")
    orig_line_count = original.count("\n") + 1
    for i in range(len(lines) - orig_line_count + 1):
        window = "\n".join(lines[i : i + orig_line_count])
        if _normalize_ws(window) == norm_orig:
            return spec.replace(window, replacement, 1), "whitespace"

    # 3. difflib fuzzy match (ratio > 0.85)
    import difflib
    best_ratio = 0.0
    best_start = -1
    best_end = -1
    for window_size in range(orig_line_count - 1, orig_line_count + 2):
        if window_size < 1 or window_size > len(lines):
            continue
        for i in range(len(lines) - window_size + 1):
            window = "\n".join(lines[i : i + window_size])
            ratio = difflib.SequenceMatcher(None, original, window).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_start = i
                best_end = i + window_size
    if best_ratio > 0.85 and best_start >= 0:
        old_window = "\n".join(lines[best_start:best_end])
        return spec.replace(old_window, replacement, 1), "fuzzy"

    return spec, ""


def _file_is_fresh(path: Path, state: dict) -> bool:
    """Check if *path* was written during the current pipeline run.

    Fix #11: prevents reuse of stale RTL/TB files from previous runs.
    Returns True if the file's mtime is newer than the pipeline start time.
    """
    try:
        run_start = state.get("pipeline_run_start", 0.0)
        if not run_start:
            return True  # no start time recorded -> assume fresh (backwards compat)
        return path.stat().st_mtime >= run_start
    except OSError:
        return False


def _last(a, b):
    """Reducer that keeps the latest value.

    Used for config keys (``project_root``, ``target_clock_mhz``, etc.)
    that are shared between the orchestrator and block subgraph states.
    Without a reducer, parallel ``Send()`` branches would conflict when
    merging their (identical) config values back into the parent state.
    """
    return b


# ---------------------------------------------------------------------------
# State -- Block Subgraph
# ---------------------------------------------------------------------------

class BlockState(TypedDict):
    """Per-block state for the block lifecycle subgraph.

    DISK-FIRST ARCHITECTURE: Graph state carries ONLY routing metadata.
    All content (RTL, testbenches, specs, constraints, diagnosis, error
    logs) lives on disk.  Specialist agents read/write files directly
    via tool use (claude CLI with Read/Write/Edit tools enabled).

    Per-block transient state on disk:
      .coresmith/blocks/<block>/constraints.json    -- accumulated constraints
      .coresmith/blocks/<block>/diagnosis.json      -- latest debug diagnosis
      .coresmith/blocks/<block>/attempt_history.json -- attempt history
      .coresmith/blocks/<block>/previous_error.txt  -- latest error context

    Existing artifact locations (unchanged):
      arch/uarch_specs/<block>.md              -- uArch spec
      rtl/<rtl_target>                         -- generated RTL
      tb/cocotb/test_<block>.py                -- testbench
      .coresmith/step_logs/<block>/*.log            -- EDA tool logs
    """

    # Config (injected via Send from orchestrator) ──────────────────────────
    project_root: str
    target_clock_mhz: float
    max_attempts: int
    pipeline_run_start: float

    # The block being processed ─────────────────────────────────────────────
    current_block: dict

    # Lifecycle tracking ────────────────────────────────────────────────────
    attempt: int
    phase: str  # "init" | "uarch" | "rtl" | "lint" | "tb" | "sim" | "synth"

    # Two-pass restructure: which orchestrator pass this block is running in.
    # "uarch" = pass 1 (spec + Amaranth block model only, no RTL); "rtl" = pass 2
    # (RTL gen + DV + synth). Threaded in from the orchestrator via Send.
    # Only meaningful when CORESMITH_BLOCK_GOLDENS is on; "rtl" (the default)
    # preserves single-pass behaviour when the flag is off.
    pipeline_phase: str
    # True only in pass 2 of the two-pass flow (after the µarch gate). Lets
    # route_after_init distinguish flag-off "rtl" (re-spec each block) from
    # pass-2 "rtl" (reuse the pass-1 spec/model, skip straight to RTL).
    uarch_pass_done: bool

    # Routing-only flags (no content -- agents read/write disk directly) ────
    uarch_approved: bool
    lint_clean: bool
    sim_passed: bool
    synth_success: bool
    synth_gate_count: int
    ppa_ok: bool | None        # deterministic PPA gate verdict (None = not run)
    ppa_reasons: list             # human-readable budget-divergence reasons
    # Post-synthesis GATE-LEVEL SIM verdict (harness.gate_sim). None = not run.
    # False routes the block to diagnose: the synthesized netlist does not
    # reproduce the behaviour the RTL was verified with, so DV and PPA were
    # measured on different hardware.
    gate_sim_ok: bool | None
    gate_sim_status: str          # "pass"|"fail"|"not_run"|"disabled"
    gate_sim_reason: str
    # Mem-price gate DEFER: set on the accept path when the bounded revise loop
    # gave up on an over-budget spec, so the deferred excess is carried into
    # state (die rollup + integration review also read the on-disk ledger flags).
    mem_price_deferred: dict | None
    # uArch feasibility verdict (from the spec's machine-readable
    # {feasible, blocking_issues} JSON summary). A non-empty blocking-issues list
    # means the block CANNOT be built byte-exactly with its frozen interface;
    # review_uarch_spec_node fires the `uarch_feasibility` interrupt instead of
    # letting a stub proceed to RTL. Threaded from generate_uarch_spec_node.
    uarch_blocking_issues: list | None
    uarch_feasible: bool | None

    # File paths (set by nodes, consumed by routing and downstream nodes) ───
    rtl_path: str          # path to generated Verilog file
    tb_path: str           # path to generated testbench file

    # Debug routing (set by diagnose_node after reading diagnosis.json) ─────
    debug_action: str      # "retry_rtl" | "retry_tb" | "ask_human" | "escalate" | ...

    # Step log file paths ──────────────────────────────────────────────────
    step_log_paths: Annotated[dict, _last]  # {step: log_path}

    # Testbench control flags ──────────────────────────────────────────────
    preserve_testbench: bool
    force_regen_tb: bool

    # Contract-conformance stage: {old_port: contract_port} the stage renamed
    # in this block's generated RTL (empty when it already conformed). Carried
    # in state as well as on disk so a reader of the block result can see that
    # the engine edited the design, not just that the block passed.
    conformance_renames: dict

    # Human interaction ─────────────────────────────────────────────────────
    human_response: dict | None

    # Output (reducer -- flows back to orchestrator) ────────────────────────
    completed_blocks: Annotated[list[dict], operator.add]


# ---------------------------------------------------------------------------
# State -- Orchestrator Graph
# ---------------------------------------------------------------------------

class OrchestratorState(TypedDict):
    """Top-level orchestrator state for tier-based parallel execution.

    The orchestrator iterates through tiers and fans out blocks within
    each tier via ``Send()``.  Results accumulate in ``completed_blocks``.

    Config keys shared with ``BlockState`` use the ``_last`` reducer so
    that parallel ``Send()`` branches can merge without conflict.
    """

    # Config (set once) ─────────────────────────────────────────────────────
    # Reducers on config keys prevent InvalidUpdateError when multiple
    # Send() branches write the same (unchanged) config values back.
    project_root: Annotated[str, _last]
    target_clock_mhz: Annotated[float, _last]
    max_attempts: Annotated[int, _last]
    block_queue: Annotated[list[dict], _last]
    pipeline_run_start: Annotated[float, _last]  # Fix #11: epoch time of pipeline start

    # Tier tracking ─────────────────────────────────────────────────────────
    tier_list: list[int]          # sorted unique tiers, e.g. [1, 2, 3]
    current_tier_index: int

    # Two-pass restructure (CORESMITH_BLOCK_GOLDENS only) ────────────────────
    # pipeline_phase: "uarch" (pass 1: spec+model, then µarch gate) | "rtl"
    # (pass 2: RTL+DV+synth). uarch_pass_done flips True once the µarch gate
    # passes and begin_rtl_pass resets the tier index. _last reducer so parallel
    # Send branches merging the (identical) value back never conflict.
    pipeline_phase: Annotated[str, _last]
    uarch_pass_done: Annotated[bool, _last]
    # Count of µarch-gate revise/re-spec iterations (both block_math and
    # contract gaps). Bounds the in-loop re-spec so a non-composing decomposition
    # retries a few variance draws instead of dead-ending, but cannot loop
    # forever. Cap = CORESMITH_UARCH_REVISE_MAX (default 4).
    uarch_revise_attempts: Annotated[int, _last]

    # Results (accumulated via reducer from all Send branches) ──────────────
    completed_blocks: Annotated[list[dict], operator.add]

    # Blocks a declared PRD pin map RETIRED before µarch/RTL (init_tier_node).
    # Each record is {block, reason: "retired_by_pin_map", skipped: True,
    # contract_signals, pin_map_signals, covered_signals, explanation}. They are
    # NOT failures and NOT missing: the chip top emits their routing itself, so
    # they are deliberately absent from block_queue and from the assembly.
    retired_blocks: Annotated[list[dict], _last]

    # Integration review decision (set by integration_review_node) ────────
    integration_review_action: str | None

    # Integration check results ────────────────────────────────────────────
    integration_result: dict | None  # set by integration_check node

    # Model-integration gate results ───────────────────────────────────────
    model_integration_result: dict | None  # set by model_integration node (env-gated)

    # Integration DV results ───────────────────────────────────────────────
    integration_dv_result: dict | None  # set by integration_dv node

    # Validation DV results ────────────────────────────────────────────────
    validation_dv_result: dict | None  # set by validation_dv node

    # Top-level contract audit results ─────────────────────────────────────
    contract_audit_result: dict | None  # set by integration/validation DV failure triage

    # Signoff scorecard (set by final_report_node just before END) ──────────
    final_report: Annotated[dict | None, _last]

    # Terminal ──────────────────────────────────────────────────────────────
    pipeline_done: bool
    # Per-block frontend completion (all blocks passed their own DV). NOT the
    # deliverable: pipeline_done stays False until integration_dv + validation_dv
    # (+ chip-top synthesizability) pass. (fix #5)
    frontend_complete: bool
    pipeline_aborted: bool  # set by pipeline_complete_node on abort resume
    # Recoverable incomplete-gate (completion bookkeeping): on a `retry` resume
    # at the pipeline_incomplete gate, re-validate failed/missing blocks against
    # the outer controller's on-disk RTL fixes (re-run their DV; passing blocks
    # reuse their RTL via skip-regen) and recount — instead of dead-ending and
    # forcing a full `--force` restart that discards the byte-exact composition.
    # Bounded by CORESMITH_REVALIDATE_MAX so a perpetually-failing block aborts.
    revalidate_attempts: Annotated[int, _last]
    revalidate_pending: Annotated[bool, _last]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _block_name(state: BlockState) -> str:
    block = state.get("current_block")
    if block:
        return block.get("name", "unknown")
    return "unknown"


def _pr(state: BlockState) -> str:
    return state.get("project_root", str(PROJECT_ROOT))


def _scoreboard(project_root: str):
    """Best-effort Scoreboard handle (None on any import failure).

    The scoreboard is a record, never a gate: a failure here must never fail a
    pipeline node, so callers guard every use.
    """
    try:
        from orchestrator.state_store.store import Scoreboard
        return Scoreboard(project_root)
    except Exception:  # noqa: BLE001
        return None


def _record_dv_row(project_root: str, **kw) -> None:
    sb = _scoreboard(project_root)
    if sb is not None:
        try:
            sb.record_dv(**kw)
        except Exception:  # noqa: BLE001
            pass


def _record_ppa_row(project_root: str, **kw) -> None:
    sb = _scoreboard(project_root)
    if sb is not None:
        try:
            sb.record_ppa(**kw)
        except Exception:  # noqa: BLE001
            pass


def _record_coverage_row(project_root: str, **kw) -> None:
    sb = _scoreboard(project_root)
    if sb is not None:
        try:
            sb.record_coverage(**kw)
        except Exception:  # noqa: BLE001
            pass


def _stamp_engine_sha(project_root: str) -> None:
    """Stamp the engine git SHA into run state (Section 7a) + WARN on a mid-run
    change. Records ``.coresmith/engine_sha.json`` {sha, first_seen, ...} on the
    first call; on re-entry, if the LIVE engine SHA differs from the recorded
    one, a mid-run code hot-swap happened -- log a LOUD warning (it reaches the
    daemon log) and record the change so the final report surfaces it. Never
    raises."""
    try:
        from orchestrator.utils import engine_git_sha
        live = engine_git_sha()
        p = Path(project_root) / ".coresmith" / "engine_sha.json"
        rec = {}
        if p.exists():
            try:
                rec = json.loads(p.read_text())
            except Exception:  # noqa: BLE001
                rec = {}
        import time as _t
        if not rec:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({
                "sha": live, "first_seen": _t.time(), "changed": False,
                "changes": [],
            }, indent=2))
            log(f"  [ENGINE] git SHA {live or '(unknown)'} stamped for this run",
                CYAN)
            return
        if live and rec.get("sha") and live != rec.get("sha") and not any(
                c.get("to") == live for c in rec.get("changes", [])):
            log(f"\n{'!'*60}\n  [ENGINE] WARNING: engine git SHA CHANGED MID-RUN "
                f"{rec.get('sha')} -> {live} (code hot-swap). Behavior may have "
                f"shifted under the running pipeline -- results before/after this "
                f"point are NOT from the same build.\n{'!'*60}\n", RED)
            rec["changed"] = True
            rec.setdefault("changes", []).append(
                {"from": rec.get("sha"), "to": live, "at": _t.time()})
            rec["sha"] = live
            try:
                p.write_text(json.dumps(rec, indent=2))
            except OSError:
                pass
    except Exception:  # noqa: BLE001 - provenance is best-effort, never blocks
        pass


def _carried_forward_defects_path(project_root: str) -> Path:
    return Path(project_root) / ".coresmith" / "carried_forward_defects.json"


def read_carried_forward_defects(project_root: str) -> list[dict]:
    """Read the run's carried-forward defects ledger (or [] on absence).

    These are QUANTIFIED defects a downstream ADVISORY bypass observed but did
    not hard-block on (a reproducible composition mismatch, or a gate that
    threw). Surfaced in the final report + the validation-DV context so an
    advisory bypass never SILENTLY swallows a real divergence.
    """
    try:
        p = _carried_forward_defects_path(project_root)
        if p.exists():
            data = json.loads(p.read_text())
            if isinstance(data, list):
                return data
    except Exception:  # noqa: BLE001
        pass
    return []


def record_carried_forward_defect(project_root: str, defect: dict) -> None:
    """Append a carried-forward defect to the run ledger (best-effort).

    ``defect`` should NAME the specific unmodeled thing (e.g. a DUT-mastered
    second bus), not a generic single-role label. De-dups on (gate, kind,
    unmodeled, first_divergence_block) so a re-entered node does not spam the
    ledger. Never raises -- this is a record, never a gate.

    Every entry leaves here with a ``detail``: the EXPLANATION a reader needs
    to act on it. Most recorders had built that sentence and then dropped it
    (or stored it under a key nothing rendered), so the ledger's entries read
    ``detail: None`` and the final report printed a gate/kind pair with no
    account of what happened. Callers that supply one keep it; the rest fall
    back to the most specific text the entry does carry.
    """
    try:
        if not str(defect.get("detail") or "").strip():
            defect = dict(defect)
            defect["detail"] = (str(defect.get("unmodeled") or "").strip()
                                or str(defect.get("note") or "").strip()
                                or str(defect.get("reason") or "").strip())
        existing = read_carried_forward_defects(project_root)
        key = (defect.get("gate"), defect.get("kind"),
               defect.get("unmodeled"), defect.get("first_divergence_block"))
        for d in existing:
            if (d.get("gate"), d.get("kind"), d.get("unmodeled"),
                    d.get("first_divergence_block")) == key:
                return  # already recorded
        existing.append(defect)
        p = _carried_forward_defects_path(project_root)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(existing, indent=2))
    except Exception:  # noqa: BLE001
        pass


def _advisory_composition_defect(project_root: str, gate: str,
                                 violations: list) -> dict:
    """Build a carried-forward defect from an ADVISORY composition-gate mismatch.

    Names the SPECIFIC unmodeled bus role (DUT-blind, from the chip_top port
    shape when it exists) instead of a generic single-role label, and carries a
    few concrete expected/observed examples so the divergence stays auditable.
    """
    first = (violations or [{}])[0]
    first_block = first.get("first_divergence_block", "")
    unmodeled = ""
    try:
        from orchestrator.langgraph import bfm_lib as _bfm_lib
        top_src = ""
        for cand in (Path(project_root) / "rtl" / "user_project_wrapper.v",
                     Path(project_root) / "rtl" / "chip_top.v"):
            if cand.exists():
                top_src = cand.read_text(encoding="utf-8", errors="replace")
                break
        unmodeled = _bfm_lib.describe_unmodeled_roles(project_root, top_src)
    except Exception:  # noqa: BLE001
        unmodeled = ""
    return {
        "gate": gate,
        "kind": "composition_mismatch",
        "advisory": True,
        "first_divergence_block": first_block,
        "violation_count": len(violations or []),
        "unmodeled": unmodeled,
        "examples": [{"expected": v.get("expected"), "observed": v.get("observed")}
                     for v in (violations or [])[:5]],
        "note": (
            "ADVISORY bypass (CORESMITH_DETERMINISTIC_BFM) proceeded past a "
            "REPRODUCIBLE model-composition mismatch. The deterministic "
            "integration DV is expected to catch it on the real chip_top; "
            "carried forward so it is re-checked, not silently swallowed."
        ),
    }


def _persist_block_coverage(project_root: str, block_name: str,
                            cov: dict | None) -> None:
    """Persist the per-block line-coverage fact from a block-DV run.

    ``cov`` is ``run_simulation``'s ``coverage`` sub-dict: either
    ``{applicable:True, pct, floor, points_total, points_hit, uncovered_count,
    passed}`` or ``{applicable:False, reason}``. It is written BOTH to the
    scoreboard ``coverage_results`` table (so ``coverage_latest(block)`` returns
    it, mirroring the CLI ``verify_rtl`` path) AND to ``coverage.json`` in the
    block dir (a git-visible artifact the final-report node reads even if the
    sqlite db is absent). A ``None``/blank cov still records a not-applicable row
    so a run WITHOUT coverage is never silently dropped from the report. Never
    raises -- the scoreboard is a record, never a gate.
    """
    if cov is None:
        cov = {"applicable": False, "reason": "coverage not evaluated"}
    try:
        block_dir = (Path(project_root) / ".coresmith" / "blocks" / block_name)
        block_dir.mkdir(parents=True, exist_ok=True)
        (block_dir / "coverage.json").write_text(json.dumps(cov, indent=2))
    except Exception:  # noqa: BLE001
        pass
    if cov.get("applicable"):
        _record_coverage_row(
            project_root, block=block_name, scope="rtl",
            points_total=cov.get("points_total"),
            points_hit=cov.get("points_hit"),
            pct=cov.get("pct"),
            uncovered={"floor": cov.get("floor"),
                       "uncovered_count": cov.get("uncovered_count"),
                       "passed": cov.get("passed")},
        )
    else:
        _record_coverage_row(
            project_root, block=block_name, scope="rtl",
            uncovered={"applicable": False, "reason": cov.get("reason", "")},
        )


def _persist_block_throughput(project_root: str, block_name: str,
                              rec: dict | None) -> None:
    """Persist the per-block measured-throughput fact from a block-DV run.

    ``rec`` is ``run_simulation``'s ``throughput`` sub-dict (the
    ``evaluate_block_throughput`` record): either ``{applicable:True, passed,
    measured_cyc_per_op, declared_cyc_per_op, threshold_cyc_per_op, ratio, ...}``
    or ``{applicable:False, reason}``. Written to ``throughput.json`` in the
    block dir (a git-visible artifact the final-report node reads). A ``None``
    record still writes a not-applicable row so a run without a measured rate is
    never silently dropped from the report. Never raises -- a record, not a
    gate.
    """
    if rec is None:
        rec = {"gate": "measured_throughput", "scope": "block",
               "applicable": False, "passed": None,
               "reason": "throughput not evaluated"}
    try:
        block_dir = (Path(project_root) / ".coresmith" / "blocks" / block_name)
        block_dir.mkdir(parents=True, exist_ok=True)
        (block_dir / "throughput.json").write_text(json.dumps(rec, indent=2))
    except Exception:  # noqa: BLE001
        pass


#: Post-repair contract-conformance failures for ONE block before the flow
#: stops spending regeneration attempts on it and parks instead. Two is the
#: cap because the first failure is news and the second is a pattern: the
#: feedback names the EXACT required port, so a generator that misses it twice
#: is not going to find it on attempt three.
_CONFORMANCE_MAX_FAILURES = 2


def _conformance_failures_path(project_root: str, block_name: str) -> Path:
    return (Path(project_root) / ".coresmith" / "blocks" / block_name
            / "_conformance_failures.txt")


def _record_block_conformance(project_root: str, block_name: str,
                              record: dict) -> None:
    """Persist the block's contract-conformance record (a git-visible artifact).

    Applied renames MUTATE generated RTL, so they are written down where the
    final report and a reviewer can see them -- an engine that silently edits
    the design it is grading is exactly the failure this whole stage exists to
    stop. Never raises: a record, not a gate.
    """
    try:
        bdir = Path(project_root) / ".coresmith" / "blocks" / block_name
        bdir.mkdir(parents=True, exist_ok=True)
        (bdir / "contract_conformance.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8")
    except OSError:
        pass
    renames = record.get("renames") or {}
    if not renames:
        return
    try:
        _chans = record.get("rename_channels") or {}
        record_carried_forward_defect(project_root, {
            "gate": "contract_conformance",
            "kind": "block_port_renamed",
            "advisory": True,
            "first_divergence_block": block_name,
            "violation_count": len(renames),
            "unmodeled": (
                f"block '{block_name}' declared "
                + ", ".join(f"{o} (contract wants {n}, channel "
                            f"'{_chans.get(o) or '?'}')"
                            for o, n in sorted(renames.items()))
                + " -- the engine RENAMED the generated ports to the contract "
                  "names so the design could be wired deterministically"),
            "detail": (
                "The RTL generator did not spell this block's channel signals "
                "the way the frozen interface contract declares them. The "
                "renames were unambiguous (one candidate port per declared "
                "signal) and were applied in place, with the pre-repair file "
                "kept alongside as <rtl>.pre_portrepair. The block's own "
                "simulation ran AFTER the rename."),
            "note": "",
        })
    except Exception:  # noqa: BLE001 - reporting must never block the flow
        pass


def _bump_conformance_failures(project_root: str, block_name: str) -> int:
    """Count consecutive post-repair conformance failures for one block."""
    p = _conformance_failures_path(project_root, block_name)
    n = 0
    try:
        if p.exists():
            n = int((p.read_text().strip() or "0"))
    except (OSError, ValueError):
        n = 0
    n += 1
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(str(n))
    except OSError:
        pass
    return n


def _reset_conformance_failures(project_root: str, block_name: str) -> None:
    """Drop the counter once the block conforms (or after a park)."""
    p = _conformance_failures_path(project_root, block_name)
    try:
        if p.exists():
            p.unlink()
    except OSError:
        pass


def _park_conformance_unrepairable(state: BlockState, block_name: str,
                                   record: dict, failures: int) -> None:
    """PARK when regeneration will not converge on the block's contract.

    The stage already told the generator the exact required port names, twice.
    Burning the remaining attempt budget on a third identical rediscovery buys
    nothing; an operator (or the outer agent) can fix the RTL, amend the
    contract, or accept a hand-wired top. The node re-executes on resume, so the
    stage re-checks the possibly-hand-fixed file: if it now conforms the block
    proceeds normally, and if it does not the block still fails (loudly) rather
    than reaching integration deviating.
    """
    pr = _pr(state)
    log(f"  [CONFORM] {block_name}: {failures} post-repair conformance "
        f"failures -- PARKING instead of spending more regeneration attempts",
        RED)
    write_graph_event(pr, "Contract Conformance", "interrupt", {
        "block": block_name, "consecutive_failures": failures,
        "deviations": (record.get("deviations") or [])[:16],
    })
    interrupt({
        "type": "contract_conformance_unrepairable",
        "block_name": block_name,
        "consecutive_failures": failures,
        "deviations": (record.get("deviations") or [])[:16],
        "renames_applied": record.get("renames") or {},
        "expected_ports": record.get("feedback", ""),
        "supported_actions": ["retry", "proceed"],
        "outer_agent_guidance": (
            f"'{block_name}' has now failed the deterministic "
            f"contract-conformance check {failures} times AFTER the engine "
            f"applied every unambiguous port rename it could prove. Its RTL "
            f"does not expose the ports .coresmith/interface_contracts.json "
            f"declares, so the deterministic chip assembler cannot wire it and "
            f"the design would fall back to an LLM-authored top. The exact "
            f"required names are in the payload's expected_ports (and in "
            f".coresmith/blocks/<block>/previous_error.txt). Either edit the "
            f"block RTL to use them, or amend the contract if the CONTRACT is "
            f"what is wrong -- then resume. The check re-runs on resume; it "
            f"passes the block only if the RTL actually conforms."
        ),
    })


def _guard_rtl_phase(state: BlockState, node_name: str) -> None:
    """Fail loud if an RTL-pass node runs during the uArch pass (phase "uarch").

    In the two-pass flow, pass 1 must only produce uArch specs + Amaranth block
    models; RTL/testbench/synth nodes are pass-2-only. Reaching one in phase
    "uarch" means the block-subgraph routing regressed -- raise so the bug
    surfaces immediately instead of silently building RTL against placeholders.
    """
    if state.get("pipeline_phase", "rtl") == "uarch":
        raise RuntimeError(
            f"{node_name} reached in pipeline_phase='uarch' (pass 1). Pass 1 "
            "is spec+block-model only; RTL/DV/synth are pass-2-only. The block "
            "subgraph routing (route_after_init / route_after_uarch_review) "
            "regressed."
        )


def _callbacks(state: BlockState) -> list:
    """Return an empty callback list (event writing is now internal to ClaudeLLM)."""
    return []


# ---------------------------------------------------------------------------
# Node: init_block  (block subgraph)
# ---------------------------------------------------------------------------

async def init_block_node(state: BlockState) -> dict:
    """Set up the block and reset per-block state.

    In the subgraph model, ``current_block`` is already populated by the
    orchestrator's ``Send()`` call.  This node creates the golden model
    wrapper, logs, and resets lifecycle fields.
    """
    block = state["current_block"]
    block_name = block["name"]

    with _tracer.start_as_current_span(f"Init Block [{block_name}]") as span:
        span.set_attribute("block_name", block_name)
        span.set_attribute("tier", block.get("tier", 0))

    write_graph_event(_pr(state), "Init Block", "graph_node_enter", {
        "block": block_name,
    })

    create_golden_model_wrapper(block_name, block.get("python_source", ""))

    log(f"\n{'='*60}", CYAN)
    log(f"  Block: {block_name} | Tier {block.get('tier', '?')}", CYAN)
    log(f"{'='*60}", CYAN)

    write_graph_event(_pr(state), "Init Block", "graph_node_exit", {
        "block": block_name,
    })

    # Initialize per-block disk state directory
    block_dir = Path(_pr(state)) / ".coresmith" / "blocks" / block_name
    block_dir.mkdir(parents=True, exist_ok=True)
    # Reset transient files for a fresh block lifecycle
    for fname in ("constraints.json", "diagnosis.json",
                  "attempt_history.json", "previous_error.txt"):
        fpath = block_dir / fname
        if fname.endswith(".json"):
            fpath.write_text("[]" if "history" in fname or "constraint" in fname else "{}")
        else:
            fpath.write_text("")

    return {
        "attempt": 1,
        "phase": "init",
        "uarch_approved": False,
        "lint_clean": False,
        "sim_passed": False,
        "synth_success": False,
        "synth_gate_count": 0,
        "rtl_path": "",
        "tb_path": "",
        "debug_action": "",
        "human_response": None,
        "step_log_paths": {},
    }


# ---------------------------------------------------------------------------
# Node: generate_uarch_spec
# ---------------------------------------------------------------------------

def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _spec_pins_ignored() -> bool:
    """CORESMITH_IGNORE_SPEC_PINS=1 disables OPERATOR_SPEC_PIN (regen as today).

    The pin is an explicit operator action; its ABSENCE = today's behavior, so no
    enabling flag is needed -- this is only the escape hatch to force regeneration
    of a pinned spec.
    """
    return _env_truthy("CORESMITH_IGNORE_SPEC_PINS")


def _uarch_session_resume_enabled() -> bool:
    """Whether the uarch respec loop RESUMES the block's codex session across
    revise rounds. Mirrors ClaudeLLM._codex_resume_enabled (the same
    CORESMITH_CODEX_RESUME global): default-OFF, so absent the flag the node
    threads no resume id and behavior is byte-identical to today."""
    return _env_truthy("CORESMITH_CODEX_RESUME")


def _uarch_feasibility_gate_enabled() -> bool:
    """CORESMITH_UARCH_FEASIBILITY_GATE (default ON): when a uArch spec declares
    itself INFEASIBLE with its frozen interface ({feasible:false, blocking_issues:
    [...]}), fire the `uarch_feasibility` interrupt to the chip-lead instead of
    letting a stub proceed to RTL. Set to 0 to restore the legacy pass-through."""
    return os.environ.get(
        "CORESMITH_UARCH_FEASIBILITY_GATE", "1"
    ).strip() != "0"


def _block_session_id_path(project_root: str, block_name: str) -> Path:
    return (Path(project_root) / ".coresmith" / "blocks" / block_name
            / "codex_session_id")


def _read_block_session_id(project_root: str, block_name: str) -> str:
    """The stored codex session id for a block's uarch spec (for resume), or ""."""
    try:
        p = _block_session_id_path(project_root, block_name)
        return p.read_text(encoding="utf-8").strip() if p.exists() else ""
    except OSError:
        return ""


def _write_block_session_id(project_root: str, block_name: str, sid: str) -> None:
    """Persist (or clear) a block's codex session id best-effort."""
    try:
        p = _block_session_id_path(project_root, block_name)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text((sid or "").strip())
    except OSError:
        pass


def _mem_price_fresh_escalation_pending(project_root: str, block_name: str) -> int | None:
    """One-shot fresh-session escalation signal from the mem-price gate.

    Returns the identical-round count N when the gate flagged entrenchment for
    this block's NEXT regen (so the node starts a FRESH codex session with a
    mandatory-directives preamble), else None. Non-consuming reader; the node
    calls the consuming variant."""
    try:
        p = (Path(project_root) / ".coresmith" / "blocks" / block_name
             / "mem_price_fresh_escalate")
        if not p.exists():
            return None
        raw = p.read_text(encoding="utf-8").strip()
        try:
            return int(raw)
        except ValueError:
            return 0
    except OSError:
        return None


def _consume_mem_price_fresh_escalation(project_root: str, block_name: str) -> int | None:
    """Read-and-delete the one-shot fresh-session escalation marker (so it fires
    for exactly the NEXT regen, whether that regen is gate- or operator-driven)."""
    n = _mem_price_fresh_escalation_pending(project_root, block_name)
    if n is not None:
        try:
            (Path(project_root) / ".coresmith" / "blocks" / block_name
             / "mem_price_fresh_escalate").unlink()
        except OSError:
            pass
    return n


async def generate_uarch_spec_node(state: BlockState) -> dict:
    """Generate (or revise) a microarchitecture spec for the current block.

    Disk-first: the agent reads all context from disk and writes the spec
    to arch/uarch_specs/<block>.md.  No content flows through state.
    """
    block = state["current_block"]
    block_name = block["name"]

    # [rung3r2-fixes-5] OPERATOR_SPEC_PIN: pass-1 regenerates the uarch spec
    # UNCONDITIONALLY on any tier re-entry -- there is no spec-reuse path -- so an
    # operator hand-edit (the documented escalation for repeated LLM
    # non-compliance) is silently clobbered by regen. When the operator has
    # pinned the on-disk spec, skip regeneration: the review / mem-price gate
    # still prices the pinned spec on disk downstream (nothing is masked -- a
    # pinned spec that busts the gate STILL fails; the pin only prevents REGEN,
    # not review). The pin file's content is the operator's rationale, surfaced
    # in the ``spec_pinned`` event. CORESMITH_IGNORE_SPEC_PINS=1 is the escape.
    _pin_path = (Path(_pr(state)) / ".coresmith" / "blocks" / block_name
                 / "OPERATOR_SPEC_PIN")
    _pinned_spec = Path(_pr(state)) / "arch" / "uarch_specs" / f"{block_name}.md"
    if _pin_path.exists() and _pinned_spec.exists() and not _spec_pins_ignored():
        rationale = ""
        try:
            rationale = _pin_path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            rationale = ""
        log(f"  [UARCH] {block_name}: OPERATOR_SPEC_PIN present -- using the "
            f"pinned on-disk spec, SKIPPING regeneration"
            + (f" (rationale: {rationale})" if rationale else ""), YELLOW)
        write_graph_event(_pr(state), "Generate Uarch Spec", "spec_pinned", {
            "block": block_name, "rationale": rationale,
            "spec_path": str(_pinned_spec),
        })
        # armC defect 3 [dv-hardening-7]: the early return skipped the block-
        # MODEL regen entirely, so guidance the operator pinned INTO the spec
        # (e.g. a mandatory corner-clamp section) never reached the model --
        # the pin suppressed exactly the regen it was written to steer. A
        # pinned SPEC still regenerates its model (OPERATOR_MODEL_PIN and the
        # gate-scope check inside the helper still protect the model file).
        try:
            from orchestrator.langgraph.pipeline_helpers import (
                _maybe_generate_block_golden,
            )

            await _maybe_generate_block_golden(block, callbacks=_callbacks(state))
        except Exception as _exc:  # noqa: BLE001 - best-effort; gate is the backstop
            log(f"  [UARCH] {block_name}: pinned-spec model regen raised: "
                f"{_exc}", YELLOW)
        return {"uarch_approved": False, "phase": "uarch"}

    # GATE-SCOPED REVISE [dv-hardening-7]: during a µarch-gate failure
    # iteration, a block the gate did NOT implicate keeps its on-disk spec
    # verbatim -- re-drawing it burns an LLM round for zero information and
    # the churn cascades (reviewer edits -> spec mtime bump -> model regen).
    from orchestrator.langgraph.pipeline_helpers import gate_scoped_reuse_reason

    _scope = gate_scoped_reuse_reason(_pr(state), block_name)
    if _scope and _pinned_spec.exists():
        log(f"  [UARCH] {block_name}: {_scope}", YELLOW)
        write_graph_event(_pr(state), "Generate Uarch Spec", "gate_scope_reuse", {
            "block": block_name, "reason": _scope,
        })
        return {"uarch_approved": False, "phase": "uarch"}

    write_graph_event(_pr(state), "Generate Uarch Spec", "graph_node_enter", {
        "block": block_name,
    })

    with _tracer.start_as_current_span(
        f"Generate Uarch Spec [{block_name}]"
    ) as span:
        span.set_attribute("block_name", block_name)

        # Feedback sources, in priority order:
        #  1. human_response revise feedback (per-block reviewer ask)
        #  2. gate re-spec feedback written to disk by init_tier_node on a
        #     gate-triggered re-spec (Engine Fix #5) -- the µarch gate's
        #     divergence diagnosis for this block (disk-first, not a state field)
        feedback = ""
        response = state.get("human_response") or {}
        if response.get("action") == "revise":
            feedback = response.get("feedback", "")
        gate_feedback = ""
        gate_fb_path = (Path(_pr(state)) / ".coresmith" / "blocks"
                        / block_name / "gate_feedback.txt")
        if gate_fb_path.exists():
            gate_feedback = gate_fb_path.read_text(encoding="utf-8").strip()
        if gate_feedback:
            feedback = (feedback + "\n\n" + gate_feedback).strip() if feedback \
                else gate_feedback

        # [rung3r2-fixes-5] Fresh-session escalation for sticky respecs. When the
        # mem-price gate has seen the manifest_signature unchanged for two
        # consecutive revise rounds, the sticky codex session is re-emitting its
        # prior conclusion despite the directives (proven live: 4 identical
        # rounds). The gate flags the NEXT regen (marker or human_response);
        # this regen then DROPS the session resume (fresh codex session) and
        # prepends a MANDATORY-directives preamble to the accumulated feedback --
        # whether the regen was gate- or operator-driven. Convergent rounds
        # (signature changing) keep resuming (default, gated on CORESMITH_CODEX_RESUME).
        fresh_session = bool(response.get("fresh_session"))
        escalation_n = _consume_mem_price_fresh_escalation(_pr(state), block_name)
        if escalation_n is not None:
            fresh_session = True
        if fresh_session and feedback:
            n_txt = escalation_n if escalation_n is not None else "several"
            feedback = (
                f"Previous attempts re-submitted an unchanged spec {n_txt} times; "
                f"this is a fresh start. The following directives are MANDATORY:\n\n"
                + feedback
            )

        # Resume-vs-fresh session for this regen. Convergent revise rounds resume
        # the block's prior codex session (a convergence feature); the fresh
        # escalation drops it so the entrenched model starts clean. Only acts
        # when CORESMITH_CODEX_RESUME is on -- otherwise byte-identical to today
        # (no id read, resume ignored by ClaudeLLM.call anyway).
        resume_session_id = None
        if _uarch_session_resume_enabled() and not fresh_session:
            resume_session_id = _read_block_session_id(_pr(state), block_name) or None
        if fresh_session:
            _write_block_session_id(_pr(state), block_name, "")  # drop stale session

        spec_path = Path(_pr(state)) / "arch" / "uarch_specs" / f"{block_name}.md"
        previous_spec = ""
        if feedback and spec_path.exists():
            previous_spec = spec_path.read_text()
            src = ("fresh-session re-spec" if fresh_session else
                   ("gate re-spec" if gate_feedback else "feedback"))
            log(f"  [UARCH] Revising spec for {block_name} with {src}...", YELLOW)
        else:
            log(f"  [UARCH] Generating microarchitecture spec for {block_name}...", YELLOW)

        result = await generate_uarch_spec(
            block, feedback=feedback, previous_spec=previous_spec,
            constraints=[],
            callbacks=_callbacks(state),
            resume_session_id=resume_session_id,
        )

        # Persist the codex session id this call produced (empty on non-codex /
        # no-session), so the next convergent round can resume it. Best-effort.
        if _uarch_session_resume_enabled():
            _write_block_session_id(_pr(state), block_name, result.get("session_id", ""))

        if "error" in result:
            log(f"  [UARCH] FAILED: {result['error']}", RED)
            span.set_attribute("error", result["error"])
        else:
            chars = len(result.get("spec_text", ""))
            log(f"  [UARCH] Generated spec ({chars} chars)", GREEN)
            span.set_attribute("chars", chars)

    write_graph_event(_pr(state), "Generate Uarch Spec", "graph_node_exit", {
        "block": block_name,
    })

    return {
        "uarch_approved": False,
        "phase": "uarch",
    }


# ---------------------------------------------------------------------------
# Node: review_uarch_spec  (INTERRUPT -- human-in-the-loop)
# ---------------------------------------------------------------------------

def _mem_price_max_revise() -> int:
    """CORESMITH_MEM_PRICE_MAX_REVISE (default 3): bound the spec re-spec loop."""
    try:
        return max(0, int(os.environ.get("CORESMITH_MEM_PRICE_MAX_REVISE", "3") or "3"))
    except ValueError:
        return 3


def _ers_parameters_block_present(project_root: str) -> bool:
    """True when the run's ERS carries a typed ``parameters`` block (param-
    schema-1). Presence of the block IS the new-schema / new-run signal: such a
    run flips the memory-manifest requirement to STRICT for that run (absent
    manifest = reject) WITHOUT touching the global CORESMITH_MEM_MANIFEST_REQUIRED
    default. Legacy prose-ERS runs lack the block -> warn-only, unchanged. Never
    raises."""
    try:
        from orchestrator.architecture import param_schema as _psch
        return _psch.ers_has_parameters_block(project_root)
    except Exception:  # noqa: BLE001
        return False


def _mem_price_gate_verdict(project_root: str, block_name: str) -> dict | None:
    """Tier-2 per-block memory-price gate at spec acceptance (Deliverable 1).

    Parses the spec's machine-readable ``# MEM`` manifest, prices each memory
    (real PDK area when the characterizer cache is warm, else the analytic
    flop-bits floor -- never blocking on a missing PDK), writes the priced
    ``mem_price.json`` ledger, and returns a re-spec request when the block busts
    its area budget or a single memory busts the sanity cap. Returns None to
    accept the spec. Loud-warns (and accepts) a legacy/prose-only spec that
    declares storage but no manifest unless strict: strict = the global
    CORESMITH_MEM_MANIFEST_REQUIRED opt-in OR (param-schema-1) the run's ERS
    declares a typed ``parameters`` block (new-schema run).
    """
    from orchestrator.langgraph import mem_price as _mprice
    from orchestrator.langgraph.ppa_check import floor_area_budget, parse_area_budget

    spec_path = (Path(project_root) / "arch" / "uarch_specs" / f"{block_name}.md")
    if not spec_path.exists():
        return None

    # PR#12 finding #7: an operator `override` on an [area] uarch_feasibility
    # blocker writes `uarch_feasibility_override`, which clears the LLM-reasoned
    # feasibility interrupt -- but this DETERMINISTIC mem_price gate is separate
    # and used to re-fire and re-enter the bounded revise loop anyway, so the
    # override never actually forced past the area verdict (observed: unbounded
    # mem_price auto-revise churn after an override). Honor the override here:
    # price the manifest for the ledger record, then DEFER (carry-forward
    # advisory) instead of demanding a re-spec. The chip-lead has explicitly
    # accepted the area; force past, don't loop.
    _bdir = Path(project_root) / ".coresmith" / "blocks" / block_name
    if (_bdir / "uarch_feasibility_override").exists():
        spec_text0 = spec_path.read_text(encoding="utf-8", errors="replace")
        decls0 = _mprice.parse_mem_manifest(spec_text0)
        area_budget0 = floor_area_budget(
            parse_area_budget(spec_text0), block_name, spec_text0)
        try:
            if decls0:
                priced0 = _mprice.price_manifest(decls0)
                verdict0 = _mprice.evaluate_mem_price(
                    priced0, area_budget_um2=area_budget0)
                _mprice.write_ledger(project_root, block_name,
                    _mprice.format_ledger(
                        block_name, verdict0, area_budget_um2=area_budget0,
                        manifest_present=True, over_budget=not verdict0.ok,
                        deferred=True,
                        deferred_reason="operator override (uarch_feasibility "
                                        "[area] override) -- mem_price forced "
                                        "past, carried forward"))
        except Exception:  # noqa: BLE001 - never block on the ledger record
            pass
        log(f"  [MEM-PRICE] {block_name}: uarch_feasibility_override present -- "
            f"FORCING PAST the mem_price gate (deferred, carried forward) "
            f"instead of re-entering the revise loop", YELLOW)
        write_graph_event(project_root, "Review Uarch Spec",
                          "mem_price_override_deferred", {"block": block_name})
        return None

    spec_text = spec_path.read_text(encoding="utf-8", errors="replace")
    decls = _mprice.parse_mem_manifest(spec_text)
    # Floor structural glue/wrapper/adapter blocks so a parse artifact (or a
    # tiny declared "~2 um2") can't hold a pin-mux to a sub-cell area budget.
    area_budget = floor_area_budget(parse_area_budget(spec_text), block_name, spec_text)

    # Absent manifest: reject only in strict manifest mode AND when the spec
    # machine-readably declares storage; otherwise warn loudly and accept.
    # Strict mode is EITHER the global CORESMITH_MEM_MANIFEST_REQUIRED opt-in
    # OR (param-schema-1) a new-schema run -- one whose ERS declares a typed
    # `parameters` block. Schema presence is the per-run strict signal, so a
    # new-schema run rejects a storage-declaring spec that omits its manifest
    # while legacy prose-only runs stay warn-only (global default untouched).
    strict_manifest = _mprice.manifest_required() or _ers_parameters_block_present(project_root)
    if not decls:
        if _mprice.spec_declares_storage(spec_text):
            if strict_manifest:
                led = _mprice.format_ledger(
                    block_name, _mprice.MemPriceVerdict(ok=False), area_budget_um2=area_budget,
                    manifest_present=False, note="storage declared but no # MEM manifest")
                _mprice.write_ledger(project_root, block_name, led)
                return {"action": "revise", "feedback": (
                    "MEMORY MANIFEST REQUIRED: this spec declares on-chip storage "
                    "(sram_budget) but emits no machine-readable memory manifest. "
                    "Add one `# MEM <name>: <width>x<depth> ports=<...> "
                    "impl=<flop|fpmem|sram> justification=<why the dependency "
                    "window cannot be smaller>` line PER storage element so the "
                    "physical-feasibility gate can price it.")}
            log(f"  [MEM-PRICE] {block_name}: spec declares storage but has NO "
                f"# MEM manifest -- UNPRICED (set CORESMITH_MEM_MANIFEST_REQUIRED=1 "
                f"to enforce). Accepting with warning.", YELLOW)
        _mprice.write_ledger(project_root, block_name, _mprice.format_ledger(
            block_name, _mprice.MemPriceVerdict(ok=True), area_budget_um2=area_budget,
            manifest_present=False, note="no memory manifest"))
        return None

    # Capture the PREVIOUS round's ledger (Σ-total + manifest signature) BEFORE
    # overwriting it, so the revise loop is trajectory-aware and can detect a
    # byte-identical re-submission (both used to break the non-convergent loop).
    prev_ledger = _mprice.read_ledger(project_root, block_name) or {}
    try:
        _pt = prev_ledger.get("total_area_um2")
        prev_total_um2 = float(_pt) if _pt is not None else None
    except (TypeError, ValueError):
        prev_total_um2 = None
    prev_sig = prev_ledger.get("manifest_signature") or ""
    _pm = prev_ledger.get("memories")
    prev_n = len(_pm) if isinstance(_pm, list) else None

    priced = _mprice.price_manifest(decls)
    verdict = _mprice.evaluate_mem_price(priced, area_budget_um2=area_budget)
    cur_sig = _mprice.manifest_signature(decls)
    trajectory = _mprice.trajectory_label(prev_total_um2, verdict.total_um2)
    ledger = _mprice.format_ledger(block_name, verdict, area_budget_um2=area_budget,
                                   manifest_present=True, trajectory=trajectory,
                                   signature=cur_sig)
    ledger_path = _mprice.write_ledger(project_root, block_name, ledger)
    write_graph_event(project_root, "Review Uarch Spec", "mem_price", {
        "block": block_name, "ok": verdict.ok,
        "total_area_mm2": round(verdict.total_um2 / 1e6, 4),
        "n_memories": len(priced), "trajectory": trajectory,
        "ledger": ledger_path or "",
    })
    if verdict.ok:
        log(f"  [MEM-PRICE] {block_name}: {len(priced)} memories priced at "
            f"{verdict.total_um2 / 1e6:.3f} mm^2 -- within budget", GREEN)
        return None

    # FAIL: bounded re-spec so the regen agent sees the physics; after the cap
    # (or on a byte-identical re-submission that cannot converge), DEFER with a
    # loud warning + waiver so the pipeline never deadlocks -- re-writing the
    # ledger with machine-readable over_budget/deferred flags so the deferred
    # excess is carried downstream.
    count_path = (Path(project_root) / ".coresmith" / "blocks" / block_name
                  / "mem_price_reject_count")
    try:
        count = int(count_path.read_text().strip()) if count_path.exists() else 0
    except (OSError, ValueError):
        count = 0
    max_revise = _mem_price_max_revise()
    # Byte-identical re-submission (same manifest hash AND same Σ-total): an
    # unchanged spec re-prices to the same verdict and can NEVER clear the gate.
    identical = bool(prev_sig) and prev_sig == cur_sig and trajectory == "flat"

    # [rung3r2-fixes-5] FRESH-SESSION ESCALATION. The first identical
    # re-submission means the sticky codex session (resumed across revise rounds)
    # is re-emitting its prior conclusion despite the directive-rich feedback
    # (proven live: 4 identical rounds -> entrenchment). Before deferring, force
    # ONE fresh-session regen: drop the session resume + a mandatory-directives
    # preamble (both applied by generate_uarch_spec_node when it sees the marker).
    # Only if a FRESH session ALSO fails to move the manifest do we defer. Bounded:
    # the escalation fires at most once per block (persistent `mem_price_fresh_escalated`
    # marker), then a still-identical round defers.
    escalated_before = (Path(project_root) / ".coresmith" / "blocks" / block_name
                        / "mem_price_fresh_escalated").exists()
    if identical and not escalated_before and count < max_revise:
        block_dir = Path(project_root) / ".coresmith" / "blocks" / block_name
        try:
            block_dir.mkdir(parents=True, exist_ok=True)
            (block_dir / "mem_price_fresh_escalated").write_text("1")
            # one-shot signal to the NEXT regen (gate- or operator-driven)
            (block_dir / "mem_price_fresh_escalate").write_text(str(count))
            count_path.write_text(str(count + 1))  # consume a bounded round
        except OSError:
            pass
        feedback = _mprice.format_revise_directive(
            block_name, verdict, area_budget_um2=area_budget,
            round_idx=count + 1, max_revise=max_revise,
            prev_total_um2=prev_total_um2, prev_n_memories=prev_n,
            trajectory=trajectory)
        log(f"  [MEM-PRICE] {block_name}: identical re-submission -- FRESH-SESSION "
            f"escalation (dropping the sticky codex session) instead of deferring",
            RED)
        write_graph_event(project_root, "Review Uarch Spec", "mem_price_fresh_escalate", {
            "block": block_name, "identical_rounds": count, "trajectory": trajectory,
        })
        _mprice.write_ledger(project_root, block_name, _mprice.format_ledger(
            block_name, verdict, area_budget_um2=area_budget, manifest_present=True,
            over_budget=True, trajectory=trajectory, signature=cur_sig))
        return {"action": "revise", "feedback": feedback, "fresh_session": True}

    if count >= max_revise or identical:
        reason_kind = ("identical spec re-submitted (fresh session did not move it; "
                       "unchanged manifest cannot converge)" if identical and count < max_revise
                       else f"re-spec cap ({max_revise}) reached")
        log(f"  [MEM-PRICE] {block_name}: over budget but {reason_kind} -- "
            f"DEFERRING to integration review with WARNING. "
            f"{'; '.join(verdict.reasons)}", RED)
        _mprice.write_ledger(project_root, block_name, _mprice.format_ledger(
            block_name, verdict, area_budget_um2=area_budget, manifest_present=True,
            over_budget=True, deferred=True, deferred_reason=reason_kind,
            reject_rounds=count, trajectory=trajectory, signature=cur_sig))
        write_graph_event(project_root, "Review Uarch Spec", "mem_price_defer", {
            "block": block_name, "over_budget": True, "deferred": True,
            "reason": reason_kind, "trajectory": trajectory,
            "total_area_mm2": round(verdict.total_um2 / 1e6, 4),
            "area_budget_um2": area_budget,
        })
        return None
    try:
        count_path.parent.mkdir(parents=True, exist_ok=True)
        count_path.write_text(str(count + 1))
    except OSError:
        pass
    feedback = _mprice.format_revise_directive(
        block_name, verdict, area_budget_um2=area_budget,
        round_idx=count + 1, max_revise=max_revise,
        prev_total_um2=prev_total_um2, prev_n_memories=prev_n,
        trajectory=trajectory)
    log(f"  [MEM-PRICE] {block_name} -> re-spec ({count + 1}/{max_revise}, "
        f"trajectory={trajectory}): {'; '.join(verdict.reasons)}", RED)
    return {"action": "revise", "feedback": feedback}


async def review_uarch_spec_node(state: BlockState) -> dict:
    """Auto-approve the uArch spec at the per-block level.

    Cross-block interface coherence is handled by the Integration Agent
    at the orchestrator level (``integration_review_node``), which runs
    after all blocks in a tier generate their specs and fires a single
    chip-level interrupt for user approval.

    Tier-2 memory-price gate (CORESMITH_MEM_PRICE_GATE, default ON): before
    accepting, price the spec's ``# MEM`` manifest and re-spec the block when a
    declared storage element is physically infeasible (over the per-memory
    sanity cap or the block area budget). Fail-OPEN: a gate error never blocks.
    """
    block = state["current_block"]
    block_name = block["name"]

    write_graph_event(_pr(state), "Review Uarch Spec", "graph_node_enter", {
        "block": block_name,
    })

    # GATE-SCOPED REVISE [dv-hardening-7]: a non-implicated block's spec was
    # reused verbatim by generate_uarch_spec_node -- it was already reviewed +
    # priced when it first passed. Re-reviewing re-EDITS it (the reviewer
    # always changes something), bumping mtimes and cascading regens of
    # correct collateral. Skip straight to approval.
    from orchestrator.langgraph.pipeline_helpers import gate_scoped_reuse_reason

    _scope = gate_scoped_reuse_reason(_pr(state), block_name)
    if _scope:
        log(f"  [UARCH-REVIEW] {block_name}: {_scope}", YELLOW)
        write_graph_event(_pr(state), "Review Uarch Spec", "gate_scope_reuse", {
            "block": block_name, "reason": _scope,
        })
        write_graph_event(_pr(state), "Review Uarch Spec", "graph_node_exit", {
            "block": block_name,
        })
        return {"human_response": {"action": "approve"},
                "uarch_approved": True,
                "mem_price_deferred": False}

    # uArch FEASIBILITY GATE (CORESMITH_UARCH_FEASIBILITY_GATE, default ON).
    # The spec's machine-readable {feasible, blocking_issues} verdict is the
    # engine's OWN diagnosis that a block cannot be built byte-exactly with its
    # frozen interface (a payload field too narrow, a port that omits data the
    # golden reads). Rather than let a stub sail to RTL (the reference codec wall), surface
    # the blockers to the chip-lead via the `uarch_feasibility` interrupt and
    # route the design back to interface revision. Reads the CANONICAL on-disk
    # spec (robust to the codex disk-first path). Fail-OPEN on a parse error --
    # absence of an explicit infeasible verdict = feasible (legacy specs).
    feasible, blocking_issues = True, []
    if _uarch_feasibility_gate_enabled():
        try:
            from orchestrator.langchain.agents.uarch_spec_generator import (
                feasibility_from_spec_text,
            )
            _spec_p = (Path(_pr(state)) / "arch" / "uarch_specs"
                       / f"{block_name}.md")
            if _spec_p.exists():
                feasible, blocking_issues = feasibility_from_spec_text(
                    _spec_p.read_text(encoding="utf-8", errors="replace"))
        except Exception as _fe:  # noqa: BLE001 - fail-open, never block on parse
            log(f"  [UARCH-FEAS] {block_name}: verdict parse skipped ({_fe})",
                YELLOW)
            feasible, blocking_issues = True, []

    _bdir = Path(_pr(state)) / ".coresmith" / "blocks" / block_name

    # C6: model/spec feasibility CONFLICT. The block-model generator declared
    # it cannot realize this block's datapath from the frozen interface
    # (model_interface_gap.txt, written at model generation) while the spec
    # verdict claims feasible -- the residual_recon_engine stub wall: a stub
    # model + stub-consistent TB/RTL passed per-block DV 6/6 and the chip
    # failed only at integration. Surface the conflict HERE as a [capability]
    # blocking issue so the existing interrupt engages the chip-lead pre-RTL.
    # A revise regenerates the model (clearing or re-asserting the marker);
    # the chip-lead override below silences it like any other blocker.
    if _uarch_feasibility_gate_enabled():
        _gap_p = _bdir / "model_interface_gap.txt"
        if _gap_p.exists():
            try:
                _gap_txt = _gap_p.read_text(encoding="utf-8").strip()
            except OSError:
                _gap_txt = "model declared an interface gap"
            _gap_issue = (
                "[capability] model/spec conflict: the block-model generator "
                "declared it cannot realize this block's datapath from the "
                f"frozen interface ({_gap_txt}) while the spec verdict claims "
                "feasible. Resolve where the missing data actually comes from "
                "(an existing contract field, a shared memory region, or a new "
                "contract field) and revise the interface -- do NOT let a stub "
                "proceed to RTL: it will pass per-block DV against its own "
                "stub-consistent TB and fail only at integration."
            )
            if _gap_issue not in blocking_issues:
                blocking_issues.append(_gap_issue)
                feasible = False

    if blocking_issues and (_bdir / "uarch_feasibility_override").exists():
        # A prior review round was overridden by the chip-lead; do not re-prompt
        # for the same (unchanged) spec on a two-pass re-entry. A genuine re-spec
        # (revise) rewrites the spec and clears intent by producing a new verdict.
        log(f"  [UARCH-FEAS] {block_name}: blockers present but chip-lead "
            f"OVERRODE earlier -- proceeding without re-prompting", YELLOW)
        blocking_issues = []
    if blocking_issues:
        try:
            import json as _json
            _bdir.mkdir(parents=True, exist_ok=True)
            (_bdir / "uarch_blocking_issues.json").write_text(
                _json.dumps({"block": block_name, "feasible": False,
                             "blocking_issues": blocking_issues}, indent=2),
                encoding="utf-8")
        except OSError:
            pass
        log(f"  [UARCH-FEAS] {block_name}: INFEASIBLE with the frozen "
            f"interface -- {len(blocking_issues)} blocking issue(s); escalating "
            f"to chip-lead (NOT emitting a stub)", RED)
        write_graph_event(_pr(state), "Review Uarch Spec",
                          "uarch_feasibility_blocked",
                          {"block": block_name,
                           "blocking_issues": blocking_issues})
        payload = {
            "type": "uarch_feasibility",
            "block_name": block_name,
            "blocking_issues": blocking_issues,
            "uarch_spec_path": str(
                Path(_pr(state)) / "arch" / "uarch_specs" / f"{block_name}.md"),
            "relative_paths": {
                "uarch_spec": f"arch/uarch_specs/{block_name}.md",
                "block_diagram": ".coresmith/block_diagram.json",
                "interface_contracts": ".coresmith/interface_contracts.json",
            },
            "supported_actions": ["revise_interface", "override", "abort"],
            "outer_agent_guidance": (
                "The microarchitecture step reports this block CANNOT be built "
                "byte-exactly against the golden within its budgets. This is the "
                "engine's OWN diagnosis, not a failure to try; do NOT instruct it "
                "to emit a stub or 'best-effort' design.\n"
                "1. Read blocking_issues -- each LEADS WITH ITS CATEGORY:\n"
                "   [interface] a frozen port/field can't carry data the golden "
                "reads -> widen/repartition the named edge in block_diagram.json "
                "/ interface_contracts.json, then resume `revise_interface`.\n"
                "   [area] storage/logic exceeds the block's area budget -> "
                "repartition, move to a shared backing memory, or raise the "
                "budget, then resume `revise_interface` (the mem-price gate also "
                "catches this later, but fix it here).\n"
                "   [timing] can't hit the cycle/throughput budget at the target "
                "clock -> add pipeline stages/lanes or relax the budget (the STA/"
                "PPA gate also catches this post-RTL).\n"
                "   [capability] fundamentally not realizable from this golden "
                "slice (fuses too many algorithms, or needs runtime-CONSTRUCTED "
                "structures like 80 on-the-fly Huffman trees). This has NO "
                "downstream backstop -- decompose into sub-blocks, or accept the "
                "block is hardware-intractable for this golden. Do NOT `override` "
                "a [capability] blocker.\n"
                "2. `revise_interface` re-specs the block after you fix the root "
                "cause. `override` ONLY for a false alarm you have verified. "
                "`abort` ends this block without RTL."
            ),
        }
        response = interrupt(payload) or {}
        action = response.get("action", "revise_interface")
        write_graph_event(_pr(state), "Review Uarch Spec",
                          "uarch_feasibility_resume",
                          {"block": block_name, "action": action})
        if action == "override":
            log(f"  [UARCH-FEAS] {block_name}: chip-lead OVERRIDE -- proceeding "
                f"despite reported blockers", YELLOW)
            try:
                _bdir.mkdir(parents=True, exist_ok=True)
                (_bdir / "uarch_feasibility_override").write_text("1")
            except OSError:
                pass
            # fall through to the normal review/approve path below
        elif action == "abort":
            return {"human_response": {"action": "skip"},
                    "uarch_approved": False,
                    "uarch_blocking_issues": blocking_issues,
                    "uarch_feasible": False}
        else:  # revise_interface (default)
            fb = ("INTERFACE REVISION REQUIRED -- the frozen interface cannot "
                  "carry what this block needs. If you have already widened the "
                  "interface, this re-spec picks it up; otherwise fix the "
                  "interface first. Blocking issues:\n- "
                  + "\n- ".join(blocking_issues))
            # An interface revision means the FROZEN inputs (contract / ERS /
            # golden slice) changed. Resuming the block's sticky codex session
            # anchors it on the pre-edit spec -- proven live on the reference codec, where a
            # plain revise re-emitted the OLD spec until a daemon restart dropped
            # the session. Force a FRESH session so the re-spec reasons from the
            # corrected design, not stale context (this is the manual daemon-
            # restart workaround, made automatic). generate_uarch_spec_node reads
            # `fresh_session` unconditionally and drops the stored session id.
            #
            # The edit is AUTHORIZED design triage, so re-baseline the oracle
            # manifest's SPEC files (ers/frd/prd) -- otherwise the tamper guard
            # fail-closes on the legitimate ERS edit during the later DV pass.
            # The golden + inputs/ stimulus are NEVER re-baselined (cheat-proof).
            try:
                from orchestrator.state_store.trust import rebaseline_oracle_specs
                _rb = rebaseline_oracle_specs(_pr(state))
                if _rb:
                    log(f"  [ORACLE] spec re-baseline after authorized "
                        f"feasibility revise: {_rb}", YELLOW)
                    write_graph_event(_pr(state), "Review Uarch Spec",
                                      "oracle_spec_rebaseline",
                                      {"block": block_name, "files": _rb})
            except Exception:  # noqa: BLE001 -- re-baseline is best-effort
                pass
            return {"human_response": {"action": "revise", "feedback": fb,
                                       "fresh_session": True},
                    "uarch_approved": False,
                    "uarch_blocking_issues": blocking_issues,
                    "uarch_feasible": False}

    from orchestrator.langgraph.mem_price import mem_price_gate_enabled
    if mem_price_gate_enabled():
        from orchestrator.langgraph.gate_guard import gate_guard
        gr = gate_guard("mem_price", _mem_price_gate_verdict, _pr(state), block_name)
        if gr.errored:
            log(f"  [MEM-PRICE] {block_name}: gate errored (fail-open, accepting): "
                f"{gr.reason}", YELLOW)
        elif isinstance(gr.value, dict) and gr.value.get("action") == "revise":
            log(f"  [UARCH] {block_name}: re-spec (memory price gate)", RED)
            write_graph_event(_pr(state), "Review Uarch Spec", "graph_node_exit", {
                "block": block_name, "action": "revise (mem price gate)",
            })
            return {"human_response": {"action": "revise",
                                       "feedback": gr.value.get("feedback", ""),
                                       "fresh_session": bool(gr.value.get("fresh_session"))},
                    "uarch_approved": False}

    # THROUGHPUT ROOFLINE (Section 1): the Fmax step prices each op combin-
    # ationally but nothing measured CYCLES-PER-OP, so a fixed-N loop on one
    # reusable datapath sails through timing while being multiples slower than
    # the pipelined design. When the spec declares a machine-readable `perf`
    # block, compute the modulo-scheduling roofline (SAME predict_op_delay the
    # Fmax step uses) and persist perf_model.json for the block. Advisory +
    # fail-open (gated CORESMITH_PERF_ROOFLINE): a declared cyc/op that MISSES
    # its FRD PERF-NNN target (or the self-imposed peak*derate budget when the
    # customer declined a hard cap) is WARNED loudly and carried in the model;
    # the µarch PPA judge enforces it as a `throughput` violation downstream.
    try:
        from orchestrator.langgraph.perf_roofline import (
            emit_perf_model,
            read_perf_model,
            roofline_enabled,
        )
        if roofline_enabled():
            emit_perf_model(_pr(state), block_name)
            _pm = read_perf_model(_pr(state), block_name)
            if _pm:
                if _pm.get("meets_throughput_req") is False:
                    log(f"  [ROOFLINE] {block_name}: declared "
                        f"{_pm.get('declared_cyc_per_op')} cyc/"
                        f"{_pm.get('op_unit','op')} MISSES "
                        f"{_pm.get('perf_req_id') or 'PERF'} target "
                        f"{_pm.get('perf_req_cyc_per_op')} "
                        f"({_pm.get('perf_req_source')}); roofline peak "
                        f"{_pm.get('cyc_per_op_peak')} -- widen to K>=2 "
                        f"pipelined lanes / break the "
                        f"{_pm.get('binding_constraint',{}).get('type','')} "
                        f"recurrence", RED)
                else:
                    log(f"  [ROOFLINE] {block_name}: peak "
                        f"{_pm.get('cyc_per_op_peak')} cyc/"
                        f"{_pm.get('op_unit','op')} @ {_pm.get('fmax_mhz')} MHz "
                        f"(II={_pm.get('II_min')}, budget "
                        f"{_pm.get('perf_req_cyc_per_op')} "
                        f"{_pm.get('perf_req_source')})", GREEN)
                write_graph_event(_pr(state), "Review Uarch Spec", "perf_roofline", {
                    "block": block_name,
                    "cyc_per_op_peak": _pm.get("cyc_per_op_peak"),
                    "perf_req_cyc_per_op": _pm.get("perf_req_cyc_per_op"),
                    "perf_req_source": _pm.get("perf_req_source"),
                    "declared_cyc_per_op": _pm.get("declared_cyc_per_op"),
                    "meets_throughput_req": _pm.get("meets_throughput_req"),
                })
    except Exception as _pe:  # noqa: BLE001 - roofline is advisory, never blocks
        log(f"  [ROOFLINE] {block_name}: skipped ({_pe})", YELLOW)

    log(f"  [UARCH] Auto-approve {block_name} "
        f"(chip-level review after tier completes)", GREEN)

    # Defer hygiene: if the mem-price gate accepted an OVER-BUDGET spec (bounded
    # revise loop exhausted / byte-identical no-op), carry the deferred excess
    # into state so it stays visible (the die rollup + integration review also
    # read the machine-readable over_budget/deferred flags off the ledger).
    mem_price_deferred = None
    try:
        from orchestrator.langgraph import mem_price as _mprice
        _led = _mprice.read_ledger(_pr(state), block_name) or {}
        if _led.get("deferred") or _led.get("over_budget"):
            _bud = _led.get("area_budget_um2")
            mem_price_deferred = {
                "block": block_name,
                "total_area_mm2": _led.get("total_area_mm2"),
                "area_budget_mm2": (float(_bud) / 1e6) if _bud else None,
                "deferred_reason": _led.get("deferred_reason", ""),
                "reject_rounds": _led.get("reject_rounds"),
            }
            log(f"  [MEM-PRICE] {block_name}: accepted OVER BUDGET "
                f"({_led.get('total_area_mm2')} mm^2) -- deferred excess carried "
                f"to integration review + die rollup", RED)
    except Exception:  # noqa: BLE001 - defer surfacing must never block approval
        mem_price_deferred = None

    write_graph_event(_pr(state), "Review Uarch Spec", "graph_node_exit", {
        "block": block_name, "action": "approve (deferred to integration review)",
        "mem_price_deferred": bool(mem_price_deferred),
    })

    return {"human_response": {"action": "approve"},
            "uarch_approved": True,
            "mem_price_deferred": mem_price_deferred,
            "uarch_feasible": feasible,
            "uarch_blocking_issues": blocking_issues or None}


# ---------------------------------------------------------------------------
# Node: generate_rtl  (with lint built-in)
# ---------------------------------------------------------------------------

# Deterministic-gate rejection markers a reused sim-passing RTL re-fails forever
# (Finding 1: the skip_regen livelock). The stage / storage / ifdef lints write
# these strings into <block>/previous_error.txt; they survive verbatim in the
# diagnosis raw-log tail when diagnose reroutes the error, so matching the file
# catches the re-entry regardless of who last wrote it.
_DETERMINISTIC_GATE_MARKERS = (
    "WIDE FLAT PACKED STORAGE WITH DYNAMIC PART-SELECT",   # storage lint report
    "pre-synth storage lint",                              # storage synth wrapper
    "SPLIT-BRAIN CONDITIONAL-COMPILATION",                 # ifdef lint report
    "deterministic stage-realization",                     # stage lint subtitle
    "UNSYNTHESIZABLE COMBINATIONAL CLOUD",                 # stage lint header
)


def _load_constraints_safe(constr_path) -> list:
    """Load a block's accumulated ``constraints.json`` -- NEVER crash on it.

    C21: a malformed constraints.json (observed on a regression sweep: a
    concatenated ``[]`` + array producing ``JSONDecodeError: Extra data``)
    must not terminate the RTL tier via an unguarded ``json.loads`` in
    block_done_node. The learned-constraints cache is best-effort context, not
    a correctness artifact -- fall back to no learned constraints (and try to
    recover the trailing valid array if the file is a concatenation) rather
    than crashing the whole pipeline. Pre-existing loader hardening, exposed by
    the sweep's extra block-failure traffic.
    """
    import json as _j
    try:
        p = Path(constr_path)
        if not p.exists():
            return []
        text = p.read_text()
        try:
            val = _j.loads(text)
            return val if isinstance(val, list) else []
        except _j.JSONDecodeError:
            # Best-effort recovery: the writer concatenated documents; take the
            # LAST valid top-level JSON array in the file if there is one.
            import re as _re
            arrays = _re.findall(r"\[.*?\]", text, _re.DOTALL)
            for chunk in reversed(arrays):
                try:
                    v = _j.loads(chunk)
                    if isinstance(v, list) and v:
                        return v
                except _j.JSONDecodeError:
                    continue
            return []
    except OSError:
        return []


def _rtl_sha1(rtl_path) -> str:
    """sha1 of the RTL file bytes ('' on any error) -- sim-pass provenance.

    dv-hardening-10: best_result.json records which RTL actually passed sim,
    so reuse/skip decisions can detect a stale pass after the RTL changed.
    """
    try:
        import hashlib

        return hashlib.sha1(Path(rtl_path).read_bytes()).hexdigest()
    except Exception:  # noqa: BLE001 - provenance is best-effort
        return ""


def _block_contract_sha1(project_root, block_name: str) -> str:
    """sha1 of THIS block's frozen interface-contract slice ('' on any error).

    C5: a sim-pass is provenance for the contract it was earned against, not
    just the RTL bytes. Delegates to the canonical helper in pipeline_helpers
    (single hashing scheme, shared with the block-model sidecar).
    """
    try:
        from orchestrator.langgraph.pipeline_helpers import block_contract_sha1

        return block_contract_sha1(project_root, block_name)
    except Exception:  # noqa: BLE001 - provenance is best-effort
        return ""


def _pass_provenance(project_root, block_name: str, rtl_path, tb_path) -> dict:
    """Provenance for a recorded sim-pass: WHICH RTL, WHICH TB, and WHICH
    interface contract it was earned with (C5, exp-reference_codec-20260713).

    The fragment_metadata_memory livelock: the recorded 5/5 pass had a MATCHING
    rtl_sha1 (the obsolete 48-bit RTL was still on disk) but belonged to the
    old 48-bit TB/contract era -- so the skip-regen fast path reused the stale
    RTL forever after the contract moved to 56 bits. Recording all three axes
    lets the reuse decision detect staleness on any of them. Keys with ''
    values are omitted (absent key -> that axis is not checked, which also
    keeps older best_result.json files honored as before).
    """
    prov = {
        "rtl_sha1": _rtl_sha1(rtl_path),
        "tb_sha1": _rtl_sha1(tb_path) if tb_path else "",
        "contract_sha1": _block_contract_sha1(project_root, block_name),
    }
    return {k: v for k, v in prov.items() if v}


def _deterministic_gate_retry(block_dir: Path) -> bool:
    """True when this regen re-entry was routed from a DETERMINISTIC gate
    rejection (stage / storage / ifdef lint).

    Such gates fail the SAME RTL every time, so the ``best_result.sim_passed``
    reuse short-circuit livelocks: it re-submits the identical rejected RTL,
    which re-fails the identical gate forever (observed twice live in Phase B).
    Mirrors the ``ppa_ok is False`` bypass, which breaks the same deadlock class
    for the PPA budget gate. Detected from the on-disk ``previous_error.txt``
    (the gate reports write their marker there).
    """
    try:
        txt = (block_dir / "previous_error.txt").read_text()
    except OSError:
        return False
    return any(m in txt for m in _DETERMINISTIC_GATE_MARKERS)


def _gate_retry_bypass(
    state: BlockState, block_name: str, rtl_path_obj: Path, attempt: int,
    *, kind: str = "ppa",
) -> None:
    """Break a deadlock where a sim-passing RTL is routed BACK to
    ``generate_rtl_node`` by a deterministic gate that reuse would re-fail
    forever.

    ``kind='ppa'`` -- the block PASSED sim but FAILED the deterministic PPA
    budget gate (``ppa_ok is False``); ``kind='lint'`` -- it was REJECTED by a
    deterministic stage/storage/ifdef lint (Finding 1). In both cases the
    regression guard's ``sim_passed`` short-circuit would reuse the SAME rejected
    RTL and re-fail the SAME gate every attempt. Before regenerating:
      1. Back up the passing RTL (a later NON-gated re-entry can still reuse a
         known-good functional version) and annotate ``best_result.json``.
      2. Invalidate the block's sim caches so the regenerated RTL re-sims fresh.
    ``best_result.json`` keeps ``sim_passed=True``; the bypass is keyed SOLELY on
    the LIVE gate verdict, so a later non-gated re-entry still reuses the RTL.
    """
    pr = _pr(state)
    block_dir = Path(pr) / ".coresmith" / "blocks" / block_name
    block_dir.mkdir(parents=True, exist_ok=True)
    backup = block_dir / f"rtl_backup_attempt{attempt}.v"
    try:
        backup.write_text(rtl_path_obj.read_text())
    except OSError:
        backup = None
    best_path = block_dir / "best_result.json"
    try:
        best = json.loads(best_path.read_text()) if best_path.exists() else {}
        if backup is not None:
            best[f"{kind}_bypass_backup"] = str(backup)
        best[f"{kind}_retry_attempt"] = attempt
        best_path.write_text(json.dumps(best))
    except (OSError, json.JSONDecodeError):
        pass
    # Invalidate sim caches (no netlist cache exists -- synth runs fresh).
    import shutil as _sh
    import tempfile as _tf
    sim_root = Path(pr) / "sim_build"
    stale = [sim_root / block_name]
    stale += list(sim_root.glob(f"rme_{block_name}*"))
    # The equivalence harness builds under /tmp/rme_<block>_* -- clean those too.
    stale += list(Path(_tf.gettempdir()).glob(f"rme_{block_name}_*"))
    for d in stale:
        try:
            if d.exists():
                _sh.rmtree(d, ignore_errors=True)
        except OSError:
            pass
    _why = ("passed sim but FAILED the PPA budget gate" if kind == "ppa" else
            "passed sim but was REJECTED by a deterministic lint gate "
            "(stage/storage/ifdef) that reusing the same RTL re-fails forever")
    log(f"  [RTL] {kind}-retry bypass: {block_name} {_why} -- regenerating "
        f"(backed up passing RTL to {backup}, invalidated sim caches)", YELLOW)
    write_graph_event(pr, "Generate RTL", f"{kind}_retry_bypass", {
        "block": block_name, "attempt": attempt,
        "backup": str(backup) if backup else "",
    })


def _ppa_retry_bypass(
    state: BlockState, block_name: str, rtl_path_obj: Path, attempt: int
) -> None:
    """A-Fix 4: break the PPA-retry deadlock (see :func:`_gate_retry_bypass`)."""
    _gate_retry_bypass(state, block_name, rtl_path_obj, attempt, kind="ppa")


def _snapshot_passing_block(
    pr: str, block_name: str, block: dict, rtl_path_obj: Path, attempt: int,
    reason: str,
) -> None:
    """PR#12 finding #4 (data-loss guard): before the regression guard
    invalidates a block's sim-pass and REGENERATES it (contract/tb/rtl-hash
    staleness), snapshot the CURRENT passing RTL **and** testbench to a
    recoverable location and record it in ``best_result.json``.

    The fft sweep casualty: a chip-lead contract edit (an authorized cs_sram
    ERS amendment) bumped user_project_wrapper's ``contract_sha1``; the guard
    invalidated its sim-pass and regenerated it into a WRONG boundary, and the
    correct passing RTL+TB were gone forever (checkpoint references RTL by
    path; the on-disk file was overwritten). A regeneration that produces a
    WORSE result must never be irrecoverable. Snapshots let a later re-entry
    (or an operator) restore the last-known-good artifact. Best-effort; never
    raises. Disable with ``CORESMITH_SNAPSHOT_BEFORE_REGEN=0``.
    """
    if os.environ.get("CORESMITH_SNAPSHOT_BEFORE_REGEN", "1").strip().lower() \
            in {"0", "false", "no", "off"}:
        return
    try:
        block_dir = Path(pr) / ".coresmith" / "blocks" / block_name
        snap = block_dir / f"passing_snapshot_attempt{attempt}"
        snap.mkdir(parents=True, exist_ok=True)
        saved = {}
        if rtl_path_obj.exists():
            dst = snap / rtl_path_obj.name
            dst.write_text(rtl_path_obj.read_text())
            saved["rtl"] = str(dst)
        tb_rel = block.get("testbench", "")
        if tb_rel:
            tb_obj = Path(pr) / tb_rel
            if tb_obj.exists():
                dst = snap / tb_obj.name
                dst.write_text(tb_obj.read_text())
                saved["tb"] = str(dst)
        best_path = block_dir / "best_result.json"
        if best_path.exists() and saved:
            best = json.loads(best_path.read_text())
            hist = best.get("passing_snapshots", [])
            hist.append({"attempt": attempt, "reason": reason, **saved})
            best["passing_snapshots"] = hist
            best_path.write_text(json.dumps(best))
        log(f"  [RTL] snapshot before regen: {block_name} passing artifacts "
            f"saved to {snap} ({reason}) -- recoverable if the regen is worse",
            YELLOW)
        write_graph_event(pr, "Generate RTL", "passing_snapshot", {
            "block": block_name, "attempt": attempt, "reason": reason,
            "dir": str(snap),
        })
    except (OSError, json.JSONDecodeError):
        pass


async def generate_rtl_node(state: BlockState) -> dict:
    """Generate RTL, then run lint with local LLM fix loop.

    Disk-first: the agent reads all context from disk (uarch spec, ERS,
    constraints, previous error, golden model) and writes the Verilog
    to disk.  After generation, runs Verilator lint and attempts local
    LLM fixes before escalating to the diagnose lead.

    Regression guard: if a previous attempt passed simulation, skip RTL
    regeneration AND reuse the passing testbench (re-validate only). Set
    CORESMITH_FORCE_TB_REGEN=1 to restore the old force-TB-regen behavior.
    """
    _guard_rtl_phase(state, "generate_rtl_node")
    block = state["current_block"]
    block_name = block["name"]
    attempt = state["attempt"]
    rtl_path_obj = Path(state["project_root"]) / block["rtl_target"]

    write_graph_event(_pr(state), "Generate RTL", "graph_node_enter", {
        "block": block_name, "attempt": attempt,
    })

    best_result_path = (
        Path(_pr(state)) / ".coresmith" / "blocks" / block_name / "best_result.json"
    )

    with _tracer.start_as_current_span(
        f"Generate RTL [{block_name}] attempt {attempt}"
    ) as span:
        span.set_attribute("block_name", block_name)
        span.set_attribute("attempt", attempt)

        # --- Regression guard ---
        # A-Fix 4: a block whose LIVE PPA verdict is False (passed sim but blew
        # its PPA budget) must REGENERATE -- reusing the passing RTL would re-fail
        # PPA forever (deadlock). Keyed on live state.ppa_ok (not best_result), so
        # a non-PPA re-entry (ppa_ok None/True) still reuses the passing RTL.
        ppa_retry = state.get("ppa_ok") is False
        if attempt > 1 and rtl_path_obj.exists() and best_result_path.exists():
            try:
                best = json.loads(best_result_path.read_text())
                # dv-hardening-10: a sim-pass is provenance for the RTL it
                # passed WITH. If the on-disk RTL hash differs from the
                # recorded one, the pass is stale -- do not reuse it (the
                # armC livelock: reuse decisions honored a sim_passed that
                # belonged to different RTL bytes).
                _rec_sha = best.get("rtl_sha1")
                if best.get("sim_passed") and _rec_sha:
                    _cur_sha = _rtl_sha1(str(rtl_path_obj))
                    if _cur_sha and _cur_sha != _rec_sha:
                        log(f"  [RTL] best_result sim-pass is for DIFFERENT "
                            f"RTL (hash mismatch) -- ignoring stale pass, "
                            f"re-validating {block_name}", YELLOW)
                        best["sim_passed"] = False
                # C5: the pass is ALSO provenance for the TB and the block's
                # frozen interface contract. fragment_metadata_memory livelock:
                # the recorded pass had a MATCHING rtl_sha1 (the obsolete
                # 48-bit RTL still on disk) but the TB/contract had moved to
                # 56 bits -- skip-regen reused the stale RTL forever. Absent
                # recorded hashes (older runs) skip these axes unchanged.
                _rec_tb = best.get("tb_sha1")
                if best.get("sim_passed") and _rec_tb:
                    _tb_obj = Path(_pr(state)) / block.get("testbench", "")
                    _cur_tb = (_rtl_sha1(str(_tb_obj))
                               if block.get("testbench") and _tb_obj.exists()
                               else "")
                    if _cur_tb and _cur_tb != _rec_tb:
                        log(f"  [RTL] best_result sim-pass was earned with a "
                            f"DIFFERENT testbench (hash mismatch) -- ignoring "
                            f"stale pass, regenerating {block_name}", YELLOW)
                        _snapshot_passing_block(
                            _pr(state), block_name, block, rtl_path_obj,
                            attempt, "tb_sha1_stale")
                        best["sim_passed"] = False
                _rec_ct = best.get("contract_sha1")
                if best.get("sim_passed") and _rec_ct:
                    _cur_ct = _block_contract_sha1(_pr(state), block_name)
                    if _cur_ct and _cur_ct != _rec_ct:
                        log(f"  [RTL] best_result sim-pass predates a revision "
                            f"of this block's interface contract -- ignoring "
                            f"stale pass, regenerating {block_name}", YELLOW)
                        _snapshot_passing_block(
                            _pr(state), block_name, block, rtl_path_obj,
                            attempt, "contract_sha1_stale")
                        best["sim_passed"] = False
                # Finding 1: was this retry routed from a DETERMINISTIC gate
                # rejection (stage/storage/ifdef lint)? Reusing the same
                # sim-passing RTL re-fails that gate forever -- the skip_regen
                # livelock. Detected from the block's previous_error.txt.
                det_gate_retry = _deterministic_gate_retry(best_result_path.parent)
                if best.get("sim_passed") and ppa_retry:
                    # PPA-retry deadlock bypass: back up the passing RTL +
                    # invalidate sim caches, then fall through to REGENERATE.
                    _ppa_retry_bypass(state, block_name, rtl_path_obj, attempt)
                elif best.get("sim_passed") and det_gate_retry:
                    # Deterministic-lint-retry deadlock bypass (same mechanism as
                    # the PPA bypass): back up + invalidate caches + regenerate.
                    _gate_retry_bypass(
                        state, block_name, rtl_path_obj, attempt, kind="lint",
                    )
                elif best.get("sim_passed"):
                    # A block that already passed sim is functionally done. On
                    # re-entry (e.g. an integration-review restart) REUSE the
                    # passing RTL+TB and just re-validate -- do NOT regenerate
                    # the testbench. Force-regenerating a passing block's TB
                    # produced a worse TB that re-failed, and since best_result
                    # stays sim_passed=True it re-triggered every restart -> an
                    # infinite regen/fail loop (observed wedging whole runs).
                    # The old force-regen behavior is recoverable, opt-in.
                    import os as _os_regen
                    force_tb = _os_regen.environ.get(
                        "CORESMITH_FORCE_TB_REGEN", ""
                    ).strip().lower() in {"1", "true", "yes", "on"}
                    log(f"  [RTL] SKIP regeneration -- attempt {best.get('attempt')} "
                        f"passed sim ({best.get('tests_passed')}/{best.get('tests_total')} tests). "
                        f"{'Forcing TB regen (opt-in)' if force_tb else 'Reusing passing TB'}.",
                        YELLOW)
                    span.set_attribute("skipped_regen", True)
                    span.set_attribute("force_regen_tb", force_tb)
                    write_graph_event(_pr(state), "Generate RTL", "graph_node_exit", {
                        "block": block_name, "attempt": attempt,
                        "action": "skip_regen (previous sim passed)"
                        + ("; force TB regen" if force_tb else "; reuse TB"),
                    })
                    return {
                        "rtl_path": str(rtl_path_obj),
                        "phase": "rtl",
                        "lint_clean": True,
                        "force_regen_tb": force_tb,
                    }
            except (json.JSONDecodeError, OSError):
                pass

        if attempt == 1 and rtl_path_obj.exists() and _file_is_fresh(rtl_path_obj, state):
            log(f"  [RTL] Using existing (fresh): {block['rtl_target']}", GREEN)
        else:
            log(f"  [RTL] Generating Verilog for {block_name}...", YELLOW)
            rtl_result = await generate_rtl(
                block, attempt,
                callbacks=_callbacks(state),
            )
            if "error" in rtl_result:
                log(f"  [RTL] FAILED: {rtl_result['error']}", RED)
                span.set_attribute("error", rtl_result["error"])

                write_graph_event(_pr(state), "Generate RTL", "graph_node_exit", {
                    "block": block_name, "attempt": attempt, "error": rtl_result["error"],
                })
                block_dir = Path(_pr(state)) / ".coresmith" / "blocks" / block_name
                block_dir.mkdir(parents=True, exist_ok=True)
                (block_dir / "previous_error.txt").write_text(
                    f"RTL generation failed: {rtl_result['error']}"
                )
                return {"rtl_path": str(rtl_path_obj), "phase": "lint", "lint_clean": False}
            else:
                log(f"  [RTL] Generated to {block['rtl_target']}", GREEN)

    # --- Lint with local fix loop ---
    rtl_path = str(rtl_path_obj)
    lint_clean = False
    lint_result = None
    existing_logs = dict(state.get("step_log_paths") or {})

    if not rtl_path_obj.exists():
        error_msg = "RTL generation failed (no file on disk)"
        log(f"  [LINT] Skipped -- {error_msg}", RED)
        block_dir = Path(_pr(state)) / ".coresmith" / "blocks" / block_name
        block_dir.mkdir(parents=True, exist_ok=True)
        (block_dir / "previous_error.txt").write_text(error_msg)
        write_graph_event(_pr(state), "Generate RTL", "graph_node_exit", {
            "block": block_name, "attempt": attempt, "lint_clean": False,
        })
        return {"rtl_path": rtl_path, "phase": "lint", "lint_clean": False,
                "step_log_paths": existing_logs}

    try:
        rtl_source = rtl_path_obj.read_text()
    except OSError:
        rtl_source = ""

    if rtl_source and not re.search(r"^\s*module\s+\w+", rtl_source, re.MULTILINE):
        corrupt_msg = "RTL file is corrupt (not valid Verilog). Needs regeneration."
        log(f"  [LINT] {corrupt_msg}", RED)
        block_dir = Path(_pr(state)) / ".coresmith" / "blocks" / block_name
        block_dir.mkdir(parents=True, exist_ok=True)
        (block_dir / "previous_error.txt").write_text(corrupt_msg)
        write_graph_event(_pr(state), "Generate RTL", "graph_node_exit", {
            "block": block_name, "attempt": attempt, "lint_clean": False,
        })
        return {"rtl_path": rtl_path, "phase": "lint", "lint_clean": False,
                "step_log_paths": existing_logs}

    with _tracer.start_as_current_span(f"Lint [{block_name}]") as lint_span:
        lint_span.set_attribute("block_name", block_name)

        for local_attempt in range(1 + MAX_LOCAL_RETRIES):
            log(f"  [LINT] Running Verilator lint"
                f"{f' (local fix #{local_attempt})' if local_attempt > 0 else ''}...",
                YELLOW)
            lint_result = await asyncio.to_thread(lint_rtl, rtl_path, block_name, attempt)

            if lint_result["clean"]:
                lint_clean = True
                log(f"  [LINT] Clean"
                    f"{f' (after {local_attempt} local fix(es))' if local_attempt > 0 else ''}",
                    GREEN)
                lint_span.set_attribute("clean", True)
                lint_span.set_attribute("local_fixes", local_attempt)
                break

            log("  [LINT] Errors found", RED)
            log(f"    {lint_result.get('errors', '')[:200]}", RED)

            if local_attempt < MAX_LOCAL_RETRIES:
                log(f"  [LINT] Attempting local LLM fix ({local_attempt + 1}/{MAX_LOCAL_RETRIES})...", YELLOW)
                write_graph_event(_pr(state), "Lint Fix", "llm_start", {
                    "block": block_name, "local_attempt": local_attempt + 1,
                })

                fixed_rtl = await fix_lint_errors(
                    block_name, rtl_path, lint_result.get("log_path", ""),
                    callbacks=_callbacks(state),
                )

                write_graph_event(_pr(state), "Lint Fix", "llm_end", {
                    "block": block_name, "local_attempt": local_attempt + 1,
                    "fix_produced": fixed_rtl is not None,
                })

                if fixed_rtl:
                    log("  [LINT] Local fix applied, re-linting...", YELLOW)
                else:
                    log("  [LINT] LLM could not produce a fix, escalating to diagnose", RED)
                    break
            else:
                log("  [LINT] Local retries exhausted, escalating to diagnose", RED)

        lint_span.set_attribute("clean", lint_clean)

    if lint_result and lint_result.get("log_path"):
        existing_logs["lint"] = lint_result["log_path"]

    if not lint_clean and lint_result:
        lint_output = lint_result.get("errors", "") or lint_result.get("warnings", "")
        block_dir = Path(_pr(state)) / ".coresmith" / "blocks" / block_name
        block_dir.mkdir(parents=True, exist_ok=True)
        (block_dir / "previous_error.txt").write_text(lint_output[-5000:])

    # FUNCTIONAL-IFDEF GATE (rung3 split-brain ban). Verilator lint compiles only
    # the ACTIVE `ifdef branch, so a two-implementation split-brain module (real
    # datapath under `ifndef SYNTHESIS, non-functional mock under `else) lints
    # AND simulates clean while every synth/backend gate builds the OTHER branch.
    # Reject it at generation time and route the SAME actionable "write ONE
    # implementation" message to regeneration. Same acceptance path + env-gate
    # convention as the pre-synth storage lint. CORESMITH_IFDEF_LINT=0 bypasses.
    if lint_clean:
        try:
            from orchestrator.langgraph.rtl_storage_lint import (
                find_functional_ifdef_regions,
                format_ifdef_lint_report,
                ifdef_lint_enabled,
            )
            if ifdef_lint_enabled():
                _is_lib = f"{os.sep}rtl_lib{os.sep}" in rtl_path
                _ir = find_functional_ifdef_regions(
                    rtl_path_obj.read_text(), is_library=_is_lib,
                )
                if not _ir.ok:
                    lint_clean = False
                    _imsg = format_ifdef_lint_report(_ir, block=block_name)
                    log(f"  [LINT] FUNCTIONAL-IFDEF GATE failed "
                        f"({len(_ir.findings)} split-brain `ifdef region(s)) -- "
                        f"one module must have ONE implementation, routing regen",
                        RED)
                    block_dir = Path(_pr(state)) / ".coresmith" / "blocks" / block_name
                    block_dir.mkdir(parents=True, exist_ok=True)
                    (block_dir / "previous_error.txt").write_text(_imsg[-5000:])
                    write_graph_event(_pr(state), "Functional Ifdef Gate",
                                      "gate_failed", {
                        "block": block_name,
                        "findings": [f.condition for f in _ir.findings],
                    })
        except Exception as _e:  # never let the gate crash the node
            log(f"  [LINT] functional-ifdef gate error (skipped): {_e}", YELLOW)

    # STAGE-REALIZATION GATE (pipeline-campaign). A spec-declared multi-stage
    # datapath collapsed into a single-cycle combinational cloud lints AND
    # simulates clean (it is functionally correct) but yosys `proc` unrolls every
    # constant-bound `for` and inlines every task/function -> the whole
    # N-candidate search elaborates in one cycle and the synth gate times out
    # (the four-generation RD-encoder wall). A deterministic arithmetic census
    # (loop-unroll + task-inline) makes the amplification visible in ms, BEFORE
    # the 600 s synth timeout, and routes an actionable module-per-stage remedy
    # to regen -- same acceptance path + env-gate convention as the storage/ifdef
    # lints. CORESMITH_STAGE_LINT=0 bypasses.
    if lint_clean:
        try:
            from orchestrator.langgraph.rtl_stage_lint import (
                census_rtl,
                census_signature,
                format_stage_lint_report,
                load_stage_map,
                stage_lint_enabled,
                stage_modules_enabled,
            )
            if stage_lint_enabled():
                _sm = load_stage_map(_pr(state), block_name)
                _sr = census_rtl(
                    rtl_path_obj.read_text(), stage_map=_sm,
                    enforce_stage_modules=stage_modules_enabled(),
                )
                if not _sr.ok:
                    lint_clean = False
                    # [Deliverable 3] trajectory-aware, fresh-session escalation:
                    # a byte-for-byte-identical census across retries means the
                    # regen re-registered outputs / renamed states without moving
                    # the arithmetic -- escalate the directive (and, after two
                    # identical rounds, demand a fresh-from-stage-map rewrite).
                    _sig = census_signature(_sr)
                    _blk_dir = Path(_pr(state)) / ".coresmith" / "blocks" / block_name
                    _blk_dir.mkdir(parents=True, exist_ok=True)
                    _sig_path = _blk_dir / "_stage_lint_signature.txt"
                    _prev_sig, _n_same = "", 0
                    if _sig_path.exists():
                        try:
                            _pv = _sig_path.read_text().split()
                            _prev_sig = _pv[0] if _pv else ""
                            _n_same = int(_pv[1]) if len(_pv) > 1 else 0
                        except Exception:  # noqa: BLE001
                            pass
                    _identical = bool(_prev_sig) and _prev_sig == _sig
                    _n_same = (_n_same + 1) if _identical else 0
                    _sig_path.write_text(f"{_sig} {_n_same}")
                    _smsg = format_stage_lint_report(
                        _sr, block=block_name,
                        trajectory=("identical" if _identical else ""),
                        fresh_session=(_n_same >= 2),
                    )
                    (_blk_dir / "previous_error.txt").write_text(_smsg[-8000:])
                    # dv-hardening-11: name the BINDING criterion. The old
                    # headline unconditionally printed the multiplier compare
                    # -- on a factor-only failure it read "32 effective
                    # multipliers > cap 64" (false on its face) and steered
                    # regens toward an already-green metric.
                    if _sr.mul_violations:
                        _why = (f"worst always-block = {_sr.worst_mul:,} "
                                f"effective multipliers > cap {_sr.mul_cap}")
                    elif _sr.factor_violations:
                        _wo = max(_sr.factor_violations,
                                  key=lambda b: b.eff_ops)
                        _why = (f"TOTAL EFFECTIVE OPS over stage-map budget; "
                                f"worst `{_wo.name}` = {_wo.eff_ops:,} eff-ops "
                                f"(multiplier census GREEN: {_sr.worst_mul} "
                                f"<= cap {_sr.mul_cap} -- do not optimize "
                                f"multipliers)")
                    else:
                        _why = "structural stage-module deficiency"
                    log(f"  [LINT] STAGE-REALIZATION GATE failed ({_why}) -- "
                        f"routing module-per-stage remedy to regen", RED)
                    write_graph_event(_pr(state), "Stage Realization Gate",
                                      "gate_failed", {
                        "block": block_name,
                        "worst_effective_multipliers": _sr.worst_mul,
                        "mul_cap": _sr.mul_cap,
                        "mul_violations": [b.name for b in _sr.mul_violations],
                        "factor_violations": [b.name for b in _sr.factor_violations],
                        "stage_module_deficient": _sr.stage_module_deficient,
                        "identical_resubmission": _identical,
                        "fresh_session_escalation": bool(_n_same >= 2),
                    })
                else:
                    # clean pass: drop any stale rejection signature so a later
                    # unrelated failure is not misread as an identical resubmission.
                    _sig_path = (Path(_pr(state)) / ".coresmith" / "blocks"
                                 / block_name / "_stage_lint_signature.txt")
                    if _sig_path.exists():
                        try:
                            _sig_path.unlink()
                        except OSError:
                            pass
        except Exception as _e:  # never let the gate crash the node
            log(f"  [LINT] stage-realization gate error (skipped): {_e}", YELLOW)

    write_graph_event(_pr(state), "Generate RTL", "graph_node_exit", {
        "block": block_name, "attempt": attempt, "lint_clean": lint_clean,
    })

    return {
        "rtl_path": rtl_path,
        "phase": "rtl" if lint_clean else "lint",
        "lint_clean": lint_clean,
        "step_log_paths": existing_logs,
    }


# ---------------------------------------------------------------------------
# Helpers: testbench bug detection
# ---------------------------------------------------------------------------

_TB_BUG_PATTERNS = [
    # Python framework / import problems
    "AttributeError", "has no attribute",
    "ModuleNotFoundError", "ImportError",
    "SyntaxError", "NameError",
    "TypeError: 'NoneType'",
    "TypeError: int() argument",
    # cocotb timing / API misuse
    "Timer(0)", "Timer( 0",
    "cocotb.result.SimFailure",
    "start_fork",                       # removed in cocotb 2.0
    "units=",                           # cocotb 2.0 wants unit= (singular)
    "unexpected keyword argument 'unit'",
    # Compile-time port/signal mismatches
    "Cannot find signal",
    "No such signal",
    "Verilator: %Error",
]


def _is_likely_testbench_bug(sim_log: str) -> bool:
    """Heuristic: returns True if sim failure looks like a TB framework bug
    (Python errors, missing signals, cocotb API misuse) rather than an RTL
    logic bug. Bare assertion failures against a Python reference model are
    NOT treated as TB bugs — they could be either a wrong reference or a
    real RTL miscompute, and the diagnose agent is far better at telling
    them apart than this string-match heuristic.
    """
    return any(p in sim_log for p in _TB_BUG_PATTERNS)


# ---------------------------------------------------------------------------
# v3 Section 4: bounded post-block-DV throughput squeeze
# ---------------------------------------------------------------------------
async def _maybe_squeeze_throughput(state, block, block_name, rtl_path, tb_path,
                                    attempt, sim_result, block_dir, span):
    """Bounded cycle-minimization squeeze for a block that passed EVERY gate.

    Fires ONLY when the block's measured cyc/op still sits above the roofline
    PEAK x 1.1. Each round asks the worker (with the measured number, the peak,
    and the binding constraint) to close the gap, then re-runs DV + measurement
    AND the byte-exact equivalence gate; the new RTL is KEPT only if it still
    passes and STRICTLY improves the measured rate, else the prior RTL is
    restored. Bounded by CORESMITH_SQUEEZE_MAX_ROUNDS (default 2); never loops on
    a block already at <= peak x 1.1; never regresses function/area/Fmax (a
    worse or failing attempt is reverted). Best-effort: any error returns the
    original result unchanged. Returns the (possibly-updated) sim_result.
    """
    import shutil
    try:
        from orchestrator.langgraph import throughput_gate as _tg
        if not _tg.throughput_squeeze_enabled():
            return sim_result
        max_rounds = _tg.squeeze_max_rounds()
        if max_rounds <= 0 or not rtl_path or not Path(rtl_path).exists():
            return sim_result
        cur = sim_result
        best_measured = ((cur or {}).get("throughput") or {}).get(
            "measured_cyc_per_op")
        need = _tg.squeeze_needed(_pr(state), block_name, best_measured)
        if need is None:
            return sim_result  # no peak / no measured / already within peak x1.1

        from orchestrator.harness.verify import run_block_equiv_gate as _run_equiv
        backup = block_dir / "rtl_pre_squeeze.v.bak"
        for rnd in range(1, max_rounds + 1):
            log(f"  [SQUEEZE] {block_name}: measured "
                f"{need['measured_cyc_per_op']} cyc/op > peak "
                f"{need['peak_cyc_per_op']} x1.1 = {need['threshold_cyc_per_op']}"
                f" -- round {rnd}/{max_rounds}", YELLOW)
            try:
                shutil.copyfile(rtl_path, backup)
            except OSError:
                return cur
            try:
                (block_dir / "previous_error.txt").write_text(
                    _tg.format_squeeze_request(block_name, need))
            except OSError:
                return cur
            write_graph_event(_pr(state), "Throughput Squeeze", "llm_start", {
                "block": block_name, "round": rnd,
                "measured": need["measured_cyc_per_op"],
                "peak": need["peak_cyc_per_op"],
            })
            rgen = await generate_rtl(block, attempt + rnd,
                                      callbacks=_callbacks(state))
            improved = False
            if not rgen.get("error"):
                new_sim = await asyncio.to_thread(
                    run_simulation, block, rtl_path, tb_path, attempt)
                new_meas = ((new_sim or {}).get("throughput") or {}).get(
                    "measured_cyc_per_op")
                ok = (bool(new_sim.get("passed")) and new_meas is not None
                      and (best_measured is None or new_meas < best_measured))
                if ok:
                    # RTL changed -> re-confirm byte-exact equivalence.
                    eqr = await asyncio.to_thread(
                        _run_equiv, block_name, rtl_path, _pr(state))
                    if eqr.get("ran") and (
                        eqr.get("failed_closed")
                        or (not eqr.get("passed") and not eqr.get("skipped"))
                    ):
                        ok = False
                        log(f"  [SQUEEZE] {block_name}: faster RTL broke "
                            "equivalence -- reverting", YELLOW)
                if ok:
                    improved = True
                    log(f"  [SQUEEZE] {block_name}: improved {best_measured} -> "
                        f"{new_meas} cyc/op (peak {need['peak_cyc_per_op']})",
                        GREEN)
                    cur = new_sim
                    best_measured = new_meas
                    span.set_attribute("throughput_squeezed", True)
                    # keep best_result.json + throughput fact in sync with the
                    # kept RTL (rtl_sha1 gates the reuse-skip logic).
                    try:
                        (block_dir / "best_result.json").write_text(json.dumps({
                            "sim_passed": True, "attempt": attempt,
                            "tests_passed": new_sim.get("tests_passed", 0),
                            "tests_total": new_sim.get("tests_total", 0),
                            "coverage": new_sim.get("coverage"),
                            "throughput": new_sim.get("throughput"),
                            # dv-hardening-10 + C5: full pass provenance
                            # (RTL + TB + contract), same as the sim node.
                            **_pass_provenance(
                                _pr(state), block_name, rtl_path, tb_path),
                        }))
                    except OSError:
                        pass
            write_graph_event(_pr(state), "Throughput Squeeze", "llm_end", {
                "block": block_name, "round": rnd, "improved": improved,
                "measured": best_measured,
            })
            if not improved:
                # a non-improving / failing / equiv-breaking attempt: restore the
                # last good RTL and stop (the worker won't do better next round).
                try:
                    shutil.copyfile(backup, rtl_path)
                except OSError:
                    pass
                break
            need = _tg.squeeze_needed(_pr(state), block_name, best_measured)
            if need is None:
                break  # reached peak x1.1 -- done
        try:
            if backup.exists():
                backup.unlink()
        except OSError:
            pass
        return cur
    except Exception as _se:  # noqa: BLE001 - squeeze is best-effort, never blocks
        log(f"  [SQUEEZE] {block_name}: skipped ({_se})", YELLOW)
        return sim_result


# ---------------------------------------------------------------------------
# Node: generate_testbench  (with simulation + local TB fix loop)
# ---------------------------------------------------------------------------

async def generate_testbench_node(state: BlockState) -> dict:
    """Generate testbench, run simulation, and fix TB locally on failure.

    After generating (or reusing) the testbench, runs cocotb simulation.
    If simulation fails and the error looks like a testbench bug (import
    error, wrong port names, timing issues), calls an LLM to fix the TB
    and re-runs -- up to MAX_LOCAL_RETRIES times.

    Only escalates to the diagnose lead for failures that appear to be
    RTL bugs (wrong computation, stuck signals, etc.).
    """
    _guard_rtl_phase(state, "generate_testbench_node")
    block = state["current_block"]
    block_name = block["name"]
    attempt = state["attempt"]
    # A blocks.yaml entry that omits `testbench` must not crash the whole
    # run -- it previously raised KeyError here and aborted every other
    # in-flight block in the tier. Default to the conventional cocotb path
    # and write it back so all downstream consumers (generate_testbench,
    # the "reuse existing" log, etc.) see a value.
    if not block.get("testbench"):
        block["testbench"] = f"tb/cocotb/test_{block_name}.py"
    tb_path_obj = Path(state["project_root"]) / block["testbench"]
    rtl_path = state.get("rtl_path", "")

    write_graph_event(_pr(state), "Generate Testbench", "graph_node_enter", {
        "block": block_name,
    })

    existing_logs = dict(state.get("step_log_paths") or {})

    # --- Guard: RTL must exist ---
    if not rtl_path or not Path(rtl_path).exists():
        log("  [TB+SIM] Skipped -- RTL file not found", RED)
        write_graph_event(_pr(state), "Generate Testbench", "graph_node_exit", {
            "block": block_name, "sim_passed": False, "reason": "no_rtl",
        })
        return {"tb_path": str(tb_path_obj), "sim_passed": False,
                "phase": "sim", "force_regen_tb": False, "step_log_paths": existing_logs}

    # --- C15: deterministic RTL-ports-vs-frozen-contract gate (pre-TB) ---
    # A stale-width RTL passed its own TB 6/6 (the TB asserted neither port
    # widths nor the contract's NORMAL path) and only died at integration.
    # This check is TB-independent and runs BEFORE testbench generation --
    # a TB cannot make a wrong-width port right, so generating one against
    # contract-contradicting RTL is pure waste. Failing here routes the
    # precise width errors into previous_error.txt so the RTL retry
    # regenerates against the frozen contract. Default-on; opt out with
    # CORESMITH_CONTRACT_PORT_GATE=0.
    if os.environ.get("CORESMITH_CONTRACT_PORT_GATE", "").strip().lower() \
            not in {"0", "false", "no", "off"}:
        _port_errors: list = []
        try:
            from orchestrator.langgraph.pipeline_helpers import (
                check_rtl_contract_ports,
            )
            _port_errors = check_rtl_contract_ports(
                _pr(state), block_name, rtl_path)
        except Exception as _pe:  # noqa: BLE001 - gate must never crash
            log(f"  [CONTRACT-PORT] {block_name}: check skipped ({_pe})",
                YELLOW)
        if _port_errors:
            for _e in _port_errors[:6]:
                log(f"  [CONTRACT-PORT] {block_name}: {_e}", RED)
            log(f"  [CONTRACT-PORT] {block_name}: RTL contradicts the "
                f"frozen contract ({len(_port_errors)} port error(s)) -- "
                f"FAILING before TB/sim", RED)
            try:
                _bd = Path(_pr(state)) / ".coresmith" / "blocks" / block_name
                _bd.mkdir(parents=True, exist_ok=True)
                (_bd / "previous_error.txt").write_text(
                    "DETERMINISTIC CONTRACT-PORT MISMATCH (no sim was run -- "
                    "a testbench cannot make a wrong-width port right). The "
                    "RTL's ports contradict the FROZEN interface contract; "
                    "regenerate the RTL against the contract widths below "
                    "(read .coresmith/interface_contracts.json, do NOT trust "
                    "any previous RTL/spec width):\n- "
                    + "\n- ".join(_port_errors), encoding="utf-8")
            except OSError:
                pass
            write_graph_event(_pr(state), "Generate Testbench",
                              "graph_node_exit", {
                                  "block": block_name, "sim_passed": False,
                                  "reason": "contract_port_mismatch",
                                  "errors": _port_errors[:8],
                              })
            return {"tb_path": str(tb_path_obj), "sim_passed": False,
                    "phase": "sim", "force_regen_tb": False,
                    "step_log_paths": existing_logs}

    # --- CONTRACT-CONFORMANCE stage: check + repair the block's PORT NAMES ---
    # Sibling of the width gate above, and for the same reason: the contract
    # already says what every channel signal is called, and a block that spells
    # one differently is unwireable. The deterministic Caravel assembler
    # resolves edges BY NAME, so a deviation makes the edge unresolvable, the
    # assembler correctly refuses, and the whole chip falls back to an
    # LLM-authored top. Measured on the first hands-off run: 8/8 blocks passed
    # every per-block gate on attempt 1, then assembly reported 10 wiring
    # hazards and the LLM fallback miswired 4 nets that lint blessed.
    #
    # The checker and the repairer already existed; nothing CALLED them. Here
    # is where they belong -- a block-time failure costs one regeneration with
    # the exact expected name, an integration-time failure has already paid for
    # every other block.
    #
    # Placed pre-TB deliberately. A repair that renames a port must be followed
    # by the block's own simulation, and running here means the sim below is
    # that re-run: the testbench is generated (or its DUT references rewritten)
    # AFTER the rename, never before it. Default-on; CORESMITH_CONTRACT_
    # CONFORMANCE_GATE=0 disables.
    _conform: dict = {}
    _conform_force_tb = False
    try:
        from orchestrator.langgraph.contract_conformance import (
            conformance_gate_enabled,
            run_conformance_stage,
        )
        _conform_on = conformance_gate_enabled()
    except Exception as _cie:  # noqa: BLE001 - gate must never crash the node
        log(f"  [CONFORM] {block_name}: stage unavailable ({_cie})", YELLOW)
        _conform_on = False
    if _conform_on:
        try:
            from orchestrator.harness.blocks import block_names as _queue_names
            _sibs = _queue_names(_pr(state))
        except Exception:  # noqa: BLE001 - siblings are best-effort
            _sibs = []
        try:
            _conform = await asyncio.to_thread(
                run_conformance_stage, _pr(state), block_name, rtl_path,
                _sibs, str(tb_path_obj),
            )
        except Exception as _ce:  # noqa: BLE001 - gate must never crash the node
            log(f"  [CONFORM] {block_name}: stage error (skipped): {_ce}",
                YELLOW)
            _conform = {}
    if _conform.get("ran"):
        _renames = _conform.get("renames") or {}
        _chans = _conform.get("rename_channels") or {}
        for _old, _new in _renames.items():
            log(f"  [CONFORM] renamed {_old} -> {_new} "
                f"(contract: channel {_chans.get(_old) or '?'})", YELLOW)
        _tbrep = _conform.get("tb") or {}
        if _tbrep.get("changed"):
            log(f"  [CONFORM] {block_name}: rewrote "
                f"{sum((_tbrep.get('applied') or {}).values())} testbench DUT "
                f"reference(s) to match "
                f"(backup at {Path(tb_path_obj).name}.pre_portrepair)", YELLOW)
        if _renames and _tbrep.get("needs_regen"):
            _conform_force_tb = True
            log(f"  [CONFORM] {block_name}: the testbench still mentions "
                f"{', '.join(_tbrep['residual'])} in a form this stage will "
                f"NOT rewrite blind (a generated TB drives getattr(dut, "
                f"<name-string>) and keys its model stimulus with the same "
                f"strings) -- REGENERATING the testbench against the repaired "
                f"RTL", YELLOW)
        _record_block_conformance(_pr(state), block_name, _conform)
        if _renames:
            log(f"  [CONFORM] {block_name}: repaired {len(_renames)} "
                f"contract deviation(s) in the generated RTL", YELLOW)
        if _conform.get("ok"):
            if not _renames:
                log(f"  [CONFORM] {block_name}: ports match the contract "
                    f"({_conform.get('checked_edges')} edge(s))", GREEN)
        else:
            _cf_n = _bump_conformance_failures(_pr(state), block_name)
            for _d in (_conform.get("deviations") or [])[:8]:
                log(f"  [CONFORM] {block_name}: {_d}", RED)
            log(f"  [CONFORM] {block_name}: RTL does NOT conform to the "
                f"interface contract after repair "
                f"({_conform.get('after_missing')} missing, failure "
                f"{_cf_n}/{_CONFORMANCE_MAX_FAILURES}) -- FAILING before "
                f"TB/sim; a deviating block must not reach integration", RED)
            try:
                _bd = Path(_pr(state)) / ".coresmith" / "blocks" / block_name
                _bd.mkdir(parents=True, exist_ok=True)
                (_bd / "previous_error.txt").write_text(
                    "DETERMINISTIC CONTRACT-CONFORMANCE FAILURE (no sim was "
                    "run). The RTL does not expose the ports the FROZEN "
                    "interface contract declares, and the deviation is not one "
                    "the engine can rename unambiguously. Regenerate the RTL "
                    "with these EXACT port names:\n\n"
                    + _conform.get("feedback", ""), encoding="utf-8")
            except OSError:
                pass
            write_graph_event(_pr(state), "Contract Conformance",
                              "gate_failed", {
                                  "block": block_name,
                                  "after_missing": _conform.get("after_missing"),
                                  "renames": _renames,
                                  "deviations": (
                                      _conform.get("deviations") or [])[:8],
                                  "consecutive_failures": _cf_n,
                              })
            if _cf_n >= _CONFORMANCE_MAX_FAILURES:
                # Cap: regeneration is not converging on the contract. PARK
                # with the exact expected names rather than burn the rest of
                # the attempt budget rediscovering the same deviation.
                _park_conformance_unrepairable(state, block_name, _conform,
                                               _cf_n)
                _reset_conformance_failures(_pr(state), block_name)
            return {"tb_path": str(tb_path_obj), "sim_passed": False,
                    "phase": "sim", "force_regen_tb": False,
                    "conformance_renames": _renames,
                    "step_log_paths": existing_logs}
        _reset_conformance_failures(_pr(state), block_name)
    elif _conform_on and _conform.get("reason"):
        log(f"  [CONFORM] {block_name}: NOT RUN -- {_conform['reason']}",
            YELLOW)

    with _tracer.start_as_current_span(
        f"Generate Testbench + Sim [{block_name}]"
    ) as span:
        span.set_attribute("block_name", block_name)

        # --- Step 1: Generate or reuse testbench ---
        # A conformance repair that renamed a port makes an EXISTING testbench
        # stale by construction, so it also forces regeneration (the reuse
        # branches below key on freshness, which a rename does not change).
        force_regen = state.get("force_regen_tb", False) or _conform_force_tb
        if not force_regen and (
            (state.get("preserve_testbench") and tb_path_obj.exists()) or
            (attempt == 1 and tb_path_obj.exists() and _file_is_fresh(tb_path_obj, state))
        ):
            log(f"  [TB] Using existing (fresh): {block['testbench']}", GREEN)
        else:
            log("  [TB] Generating cocotb testbench...", YELLOW)
            try:
                tb_result = await generate_testbench(
                    block,
                    callbacks=_callbacks(state),
                )
            except RuntimeError as exc:
                # The agent now raises if claude CLI failed to write
                # a usable testbench. Fall through to the SIM-skipped
                # path (preserves the existing retry semantics) but
                # log the actual reason instead of a misleading
                # "Generated (N tests)" / "testbench file not found"
                # mirage.
                log(f"  [TB] Generation failed: {exc}", RED)
                tb_result = {"test_count": 0}
            else:
                test_count = tb_result.get("test_count", "?")
                log(f"  [TB] Generated ({test_count} tests)", GREEN)

        tb_path = str(tb_path_obj)

        # --- Step 2: Simulate with local TB fix loop ---
        sim_passed = False
        sim_result = None
        block_dir = Path(_pr(state)) / ".coresmith" / "blocks" / block_name
        block_dir.mkdir(parents=True, exist_ok=True)

        for sim_attempt in range(1 + MAX_LOCAL_RETRIES):
            if not tb_path_obj.exists():
                log("  [SIM] Skipped -- testbench file not found", RED)
                break

            log(f"  [SIM] Running cocotb simulation"
                f"{f' (TB fix #{sim_attempt})' if sim_attempt > 0 else ''}...",
                YELLOW)
            sim_result = await asyncio.to_thread(
                run_simulation, block, rtl_path, tb_path, attempt
            )

            if sim_result["passed"]:
                sim_passed = True
                log(f"  [SIM] PASSED"
                    f"{f' (after {sim_attempt} TB fix(es))' if sim_attempt > 0 else ''}",
                    GREEN)
                span.set_attribute("passed", True)
                span.set_attribute("tb_fixes", sim_attempt)

                best_path = block_dir / "best_result.json"
                best_path.write_text(json.dumps({
                    "sim_passed": True,
                    "attempt": attempt,
                    "tests_passed": sim_result.get("tests_passed", 0),
                    "tests_total": sim_result.get("tests_total", 0),
                    # Part A: line-coverage fact travels with the passing result.
                    "coverage": sim_result.get("coverage"),
                    # v3: measured-throughput fact travels with the pass too.
                    "throughput": sim_result.get("throughput"),
                    # dv-hardening-10 + C5: provenance -- WHICH RTL, WHICH TB,
                    # and WHICH interface contract the pass was earned with.
                    # The reuse/skip logic must not honor a sim-pass after ANY
                    # of the three changed (observed livelocks: TB-only regen
                    # burned attempts against changed RTL that never re-ran;
                    # a contract widening left a stale-era pass honored forever).
                    **_pass_provenance(_pr(state), block_name, rtl_path, tb_path),
                }))
                break

            sim_log = sim_result.get("log", "")
            log("  [SIM] FAILED", RED)
            for line in sim_log.split("\n")[-5:]:
                if line.strip():
                    log(f"    {line.strip()}", RED)

            # A line-coverage-gate demotion IS a testbench weakness by
            # definition (DV passed; the TB just never exercised enough of
            # the block) -- route it to the local TB-fix loop, whose input
            # log now carries the uncovered-region list to act on. A
            # MISSING-throughput-artifact demotion (throughput_needs_tb) is
            # likewise a TB gap -- the TB lacks the required
            # test_throughput_measure case -- so it too routes to TB-fix. But a
            # measured-TOO-SLOW throughput demotion is an RTL performance defect,
            # NOT a TB bug: it does NOT set throughput_needs_tb, so it escalates
            # to diagnose (RTL fix) via the else branch below.
            is_tb_bug = _is_likely_testbench_bug(sim_log) or bool(
                sim_result.get("coverage_gate_failed")
            ) or bool(sim_result.get("throughput_needs_tb"))

            # Only run the local TB-fix loop when the heuristic actually
            # matches. Previously the orchestrator forced a TB-fix LLM call
            # on every first failure (`is_tb_bug or sim_attempt == 0`),
            # which burned ~5 minutes of compute on assertion failures that
            # were genuinely RTL bugs (or, as in mcu3, TB logic bugs that
            # required spec-level reasoning the fix-loop prompt cannot do).
            if sim_attempt < MAX_LOCAL_RETRIES and is_tb_bug:
                _tb_reason = (
                    "throughput artifact missing -- adding "
                    "test_throughput_measure to TB"
                    if sim_result.get("throughput_needs_tb")
                    else "coverage below floor -- strengthening TB"
                    if sim_result.get("coverage_gate_failed")
                    else "TB framework bug detected"
                )
                log(f"  [SIM] {_tb_reason} -- attempting "
                    f"local fix ({sim_attempt + 1}/{MAX_LOCAL_RETRIES})...", YELLOW)
                write_graph_event(_pr(state), "TB Fix", "llm_start", {
                    "block": block_name, "sim_attempt": sim_attempt + 1,
                    "is_tb_bug": is_tb_bug,
                })

                fixed = await fix_testbench_errors(
                    block_name, rtl_path, tb_path,
                    sim_result.get("log_path", ""),
                    callbacks=_callbacks(state),
                )

                write_graph_event(_pr(state), "TB Fix", "llm_end", {
                    "block": block_name, "sim_attempt": sim_attempt + 1,
                    "fix_produced": fixed is not None,
                })

                if fixed:
                    log("  [SIM] TB fix applied, re-simulating...", YELLOW)
                else:
                    log("  [SIM] LLM could not fix TB, escalating to diagnose", RED)
                    break
            else:
                # Don't pre-classify here -- the diagnose agent does that
                # well (see attempt_history.json / diagnosis.json), and a
                # wrong "Likely RTL bug" line above a real TESTBENCH_BUG
                # diagnosis is misleading.
                if is_tb_bug:
                    log("  [SIM] TB fix retries exhausted, escalating to diagnose", RED)
                else:
                    log("  [SIM] Sim failed -- escalating to diagnose for classification", RED)
                break

        span.set_attribute("sim_passed", sim_passed)

        # RTL<->model byte-exact EQUIVALENCE GATE (fix #1 / A-Fix 5 / B3).
        # Delegated to the SHARED harness gate (orchestrator.harness.verify.
        # run_block_equiv_gate) so the CLI (`coresmith verify rtl`) and this node
        # apply the IDENTICAL check -- parity by construction -- with the same
        # fail-closed + harness-error-2x-retry semantics (commits 5/8). The
        # per-block cocotb TB is LLM-authored and can be weakened; this
        # engine-run check drives generated Verilog and the Amaranth block model
        # on the same seeded vectors and asserts byte-exact. It SKIPs (never
        # false-passes) on a non-AXIS interface / no deterministic reference;
        # ran=False means the gate does not apply (equiv off / block-goldens off)
        # -> the sim verdict stands.
        if sim_passed and rtl_path:
            from orchestrator.harness.verify import run_block_equiv_gate as _run_equiv
            _eqr = await asyncio.to_thread(
                _run_equiv, block_name, rtl_path, _pr(state),
            )
            if _eqr.get("ran"):
                if _eqr.get("failed_closed") or (
                    not _eqr.get("passed") and not _eqr.get("skipped")
                ):
                    sim_passed = False
                    log(f"  [EQUIV] RTL != model -- FAIL: "
                        f"{str(_eqr.get('reason', ''))[:160]}", RED)
                    if _eqr.get("prev_error_text"):
                        try:
                            (block_dir / "previous_error.txt").write_text(
                                _eqr["prev_error_text"]
                            )
                        except OSError:
                            pass
                    span.set_attribute("equiv_passed", False)
                elif _eqr.get("skipped"):
                    log(f"  [EQUIV] skipped ({_eqr.get('reason', '')})", YELLOW)
                else:
                    log(f"  [EQUIV] RTL == model byte-exact "
                        f"({_eqr.get('checked_vectors', 0)} vectors)", GREEN)
                    span.set_attribute("equiv_passed", True)

        # BRANCH-PARITY SMOKE (rung3 split-brain backstop). When the RTL still
        # carries a conditional-compilation region that survived the functional-
        # ifdef lint (i.e. debug/assertion-only or a macro-module split), rebuild
        # the SAME block under the synth-side macro world (-DSYNTHESIS ...) and
        # rerun the SAME seeded vectors. If the two builds' verdicts diverge, the
        # "allowed" region actually changed the design's hardware -- a split-brain
        # DV alone can't see -- so fail closed. A parity build that can't compile
        # (toolchain) SKIPs, never false-fails. Env-gated CORESMITH_BRANCH_PARITY
        # (default ON only when a conditional region exists).
        parity_info: dict | None = None
        if sim_passed and rtl_path and Path(rtl_path).exists():
            try:
                from orchestrator.harness.branch_parity import check_branch_parity
                _par = await asyncio.to_thread(
                    check_branch_parity, block, rtl_path, tb_path, attempt,
                )
                if _par.ran and not _par.skipped:
                    parity_info = {
                        "name": "branch_parity_smoke", "kind": "branch_parity",
                        "ran": True, "passed": bool(_par.ok),
                    }
                    if not _par.ok:
                        sim_passed = False
                        log("  [PARITY] sim vs synth macro worlds DIVERGE -- "
                            "FAIL-CLOSED (split-brain hardware)", RED)
                        try:
                            (block_dir / "previous_error.txt").write_text(
                                _par.as_prev_error(block_name)
                            )
                        except OSError:
                            pass
                        span.set_attribute("branch_parity_passed", False)
                    else:
                        log("  [PARITY] sim == synth macro world (no split-brain)",
                            GREEN)
                        span.set_attribute("branch_parity_passed", True)
                elif _par.ran and _par.skipped:
                    parity_info = {
                        "name": "branch_parity_smoke", "kind": "branch_parity",
                        "ran": False, "skipped": True,
                        "reason": getattr(_par, "reason", ""),
                    }
                    log(f"  [PARITY] skipped ({_par.reason})", YELLOW)
            except Exception as _pe:  # never let the smoke crash the node
                log(f"  [PARITY] smoke error (skipped): {_pe}", YELLOW)

        # ORACLE INTEGRITY (B3): before accepting a pass, confirm the golden /
        # stimulus / spec that underwrites the gate was NOT edited to make the
        # RTL "match". Tampering flips the block to failed-closed.
        if sim_passed:
            try:
                from orchestrator.state_store.trust import check_oracle_manifest
                _ocheck = check_oracle_manifest(_pr(state))
            except Exception:  # noqa: BLE001
                _ocheck = {"ok": True}
            if not _ocheck.get("ok"):
                sim_passed = False
                _viol = _ocheck.get("violation") or {}
                log("  [ORACLE] tamper detected -- FAIL-CLOSED: "
                    f"{str(_viol.get('detail', ''))[:160]}", RED)
                try:
                    (block_dir / "previous_error.txt").write_text(
                        str(_viol.get("detail", "oracle tampered"))
                    )
                except OSError:
                    pass
                span.set_attribute("oracle_tamper", True)

        # v3 Section 4: bounded post-DV throughput SQUEEZE. Only when the block
        # passed EVERY gate (functional + coverage + throughput + equiv + parity
        # + oracle) but its measured cyc/op is still above the roofline PEAK x
        # 1.1 -- ask the worker to close the gap, re-verify (DV + equiv +
        # measurement), keep the better result. Bounded + fail-open.
        if sim_passed and rtl_path:
            sim_result = await _maybe_squeeze_throughput(
                state, block, block_name, rtl_path, tb_path, attempt,
                sim_result, block_dir, span,
            )

    # Write sim error for diagnose if failed
    if not sim_passed and sim_result:
        sim_log = sim_result.get("log", "")
        (block_dir / "previous_error.txt").write_text(sim_log[-5000:])

    if sim_result and sim_result.get("log_path"):
        existing_logs["simulate"] = sim_result["log_path"]

    # Don't dump multi-KB sim stdout into the event log -- log_path already
    # points to the full file on disk. Keep just enough to grep on (last
    # error line) so the JSONL stays tail-able.
    sim_log_out = sim_result.get("log", "") if sim_result else ""
    last_err = ""
    if sim_log_out and not sim_passed:
        for line in reversed(sim_log_out.splitlines()):
            if line.strip() and ("Error" in line or "FAIL" in line or "Assert" in line):
                last_err = line.strip()[:200]
                break
    write_graph_event(_pr(state), "Generate Testbench", "graph_node_exit", {
        "block": block_name,
        "sim_passed": sim_passed,
        "tb_fixes_attempted": min(sim_attempt + 1, MAX_LOCAL_RETRIES) if sim_result and not sim_passed else 0,
        "last_error": last_err,
        "log_path": sim_result.get("log_path", "") if sim_result else "",
    })

    # B3: record the authoritative (source="gate") per-block RTL DV verdict.
    _record_dv_row(
        _pr(state), block=block_name, scope="rtl", source="gate",
        attempt=attempt, passed=sim_passed,
        tests_passed=(sim_result or {}).get("tests_passed"),
        tests_total=(sim_result or {}).get("tests_total"),
        tests_failed=(sim_result or {}).get("tests_failed"),
        detail=last_err, log_path=(sim_result or {}).get("log_path", ""),
    )

    # PERSIST the per-block line-coverage fact (Part A). run_simulation's
    # line-coverage gate computed pct only to REJECT a weak TB; here we also
    # RECORD it -> scoreboard coverage_results + block_dir/coverage.json so the
    # final-report node can surface line-coverage % (or a visible "not
    # applicable: <reason>" when no coverage.dat / verilator_coverage). Recorded
    # whether or not DV passed, so a coverage-less run stays auditable.
    _persist_block_coverage(
        _pr(state), block_name, (sim_result or {}).get("coverage"),
    )

    # PERSIST the per-block measured-throughput fact (v3). run_simulation's
    # measured-throughput gate compared the TB-measured cyc/op to the uArch
    # declared §6.1 cyc/op x 1.1; here we RECORD it -> block_dir/throughput.json
    # so the final-report node can surface declared|measured|ratio|verdict (or a
    # visible "not applicable: <reason>" when the block declared no cyc/op).
    # Recorded whether or not DV passed, so a throughput-less run stays
    # auditable.
    _persist_block_throughput(
        _pr(state), block_name, (sim_result or {}).get("throughput"),
    )

    # Record the testbenches that ran for this block (names -> the final-report
    # verification-traceability list): the block-DV cocotb TB (+ #testcases +
    # verdict) plus the branch-parity smoke when it ran. Best-effort; never fails
    # the node.
    try:
        _tb_entries = [{
            "name": Path(block.get("testbench")
                         or f"tb/cocotb/test_{block_name}.py").name,
            "path": block.get("testbench", ""),
            "kind": "block_dv",
            "tests_passed": (sim_result or {}).get("tests_passed"),
            "tests_total": (sim_result or {}).get("tests_total"),
            "passed": bool(sim_passed),
        }]
        if parity_info:
            _tb_entries.append(parity_info)
        (block_dir / "dv_summary.json").write_text(json.dumps({
            "block": block_name, "sim_passed": bool(sim_passed),
            "testbenches": _tb_entries,
        }, indent=2))
    except Exception:  # noqa: BLE001
        pass

    return {
        "tb_path": tb_path,
        "sim_passed": sim_passed,
        "phase": "sim" if not sim_passed else "tb",
        "force_regen_tb": False,
        "conformance_renames": _conform.get("renames") or {},
        "step_log_paths": existing_logs,
    }


# ---------------------------------------------------------------------------
# Node: synthesize  (agent -- local LLM iteration)
# ---------------------------------------------------------------------------

def _ppa_waivers_path(project_root: str) -> Path:
    return Path(project_root) / ".coresmith" / "ppa_waivers.json"


def _ppa_tooling_waived(project_root: str) -> bool:
    """True once the operator has accepted an unmeasurable (yosys-absent) PPA
    gate for this run (A-Fix 2f) -- so we PARK at most once per run."""
    p = _ppa_waivers_path(project_root)
    if not p.exists():
        return False
    try:
        return bool(json.loads(p.read_text()).get("tooling_missing_accepted"))
    except (OSError, json.JSONDecodeError):
        return False


def _record_ppa_tooling_waiver(project_root: str, block_name: str) -> None:
    """Persist the operator's 'proceed' on an unmeasurable PPA gate."""
    p = _ppa_waivers_path(project_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    data: dict = {}
    if p.exists():
        try:
            data = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            data = {}
    data["tooling_missing_accepted"] = True
    data.setdefault("waived_at_block", block_name)
    p.write_text(json.dumps(data, indent=2))


def _ppa_should_park_tooling_missing(
    project_root: str, ppa_ok: bool | None, ppa_meta: dict | None
) -> bool:
    """A-Fix 2f decision (pure): PARK when the deterministic PPA gate could not
    run because its tooling (yosys) is absent, under the STRICT profile, and it
    has not already been waived this run. Legacy profile stays silent (never
    parks); the global CORESMITH_GATE_FAIL_OPEN escape also suppresses it. A
    gate that actually judged something (``ppa_ok`` not None) is not
    unmeasurable and never parks here.
    """
    if not (ppa_meta or {}).get("tooling_missing"):
        return False
    if ppa_ok is not None:
        return False
    from orchestrator.langgraph.gate_guard import gate_fail_open_enabled
    if gate_fail_open_enabled():
        return False
    from orchestrator.profile import ensure_applied, resolve_profile
    ensure_applied()
    if resolve_profile() != "strict":
        return False
    return not _ppa_tooling_waived(project_root)


def _park_ppa_unmeasurable(state: BlockState, block_name: str) -> None:
    """A-Fix 2f: PARK once/run on an unmeasurable (yosys-absent) PPA gate so the
    operator explicitly acknowledges it rather than the gate silently passing.

    On resume (action 'proceed') a waiver is recorded in
    ``.coresmith/ppa_waivers.json`` so the gate never re-asks this run. (The
    node re-executes on resume; the guard rechecks and this runs again, but
    ``interrupt`` then returns immediately and the waiver is written.)
    """
    pr = _pr(state)
    log(f"  [PPA] {block_name}: gate tooling (yosys) ABSENT -- parking "
        f"(strict profile) so an unmeasured PPA is acknowledged, not silently "
        f"passed", RED)
    write_graph_event(pr, "PPA Gate Unmeasurable", "interrupt", {
        "block": block_name, "reason": "tooling_missing",
    })
    interrupt({
        "type": "ppa_gate_unmeasurable",
        "block_name": block_name,
        "supported_actions": ["proceed"],
        "outer_agent_guidance": (
            "The deterministic PPA gate could NOT run because its tooling "
            "(yosys) is absent, so PPA (flip-flop/area/timing budget) is "
            "UNMEASURED for this block. Under the strict profile this parks once "
            "per run so an unmeasured PPA is not silently treated as a pass. To "
            "actually gate PPA, install yosys and re-run synthesis. To accept an "
            "unmeasured PPA for this run, resume with action='proceed' -- a "
            "waiver is recorded in .coresmith/ppa_waivers.json and the gate will "
            "not re-ask."
        ),
    })
    _record_ppa_tooling_waiver(pr, block_name)


def _evaluate_ppa_gate(
    project_root: str,
    block_name: str,
    rtl_path: str,
    synth_result: dict | None,
    *,
    require_gate_flag: bool = True,
) -> tuple[bool | None, list, dict]:
    """Deterministic PPA gate (CORESMITH_PPA_GATE=1). Returns (ppa_ok, reasons, meta).

    Keyed off a memory-PRESERVING probe so a correctly-inferred SRAM is NOT
    counted as flops (no false positives). The FF/memory check is PDK-free
    (Yosys only); area + WNS use the liberty synth / STA results when present
    (host-side). ``ppa_ok`` of None means "not evaluated / cannot judge" and
    never blocks. ``meta`` carries ``tooling_missing`` (A-Fix 2f) -- True when
    the gate's PDK-free probe could not run because yosys is absent, so the
    strict profile can PARK on an unmeasurable gate instead of silently passing.

    ``meta`` also carries the MEASURED probe metrics (rung2 defect 2): ``ff``,
    ``cells``, ``mem_bits``, ``logic_depth``, ``area_um2``, ``elaborated``,
    ``budget_ff``, ``budget_area_um2`` -- so callers can persist real numbers to
    ``ppa_history`` instead of NULLs (the SKIP_SYNTH path used to record all
    metrics as NULL even though the PDK-free probes ran).

    ``require_gate_flag`` (rung2 defect 2): when False, the PDK-free
    synthesizability probes (generic-elaborate / cell-explosion / logic-depth /
    generic-FF-budget -- none need a PDK) run and GATE even when
    ``CORESMITH_PPA_GATE`` is not seeded. The SKIP_SYNTH branch of
    ``synthesize_node`` calls with ``require_gate_flag=False`` so an
    un-synthesizable design is still caught when no real synthesis ran (the
    exact hole SKIP_SYNTH used to hide). yosys-absent still parks via
    tooling_missing.
    """
    from orchestrator.langgraph.ppa_check import (
        evaluate_ppa,
        floor_area_budget,
        mem_lib_sources_for_rtl,
        parse_area_budget,
        parse_ff_budget,
        ppa_gate_enabled,
        ppa_honor_feas_override_enabled,
        probe_synth_generic,
        run_maxfanout_buffered_sta,
        run_pre_layout_sta,
        sta_maxfanout_enabled,
    )
    if require_gate_flag and not ppa_gate_enabled():
        return None, [], {}
    if not rtl_path or not Path(rtl_path).exists():
        return None, [], {}
    spec_path = Path(project_root) / "arch" / "uarch_specs" / f"{block_name}.md"
    _spec_text = spec_path.read_text() if spec_path.exists() else ""
    ff_budget = parse_ff_budget(_spec_text) if _spec_text else None
    # Die-area budget INCLUDING SRAM macros -- the dimension that prices in RAM
    # (cs_sram blackboxes to 0 flops, so the FF budget can't see it). Estimated
    # SRAM area = wrapped bits * um2/bit, added to the synthesized std-cell area.
    # Structural glue/wrapper/adapter blocks are floored so a parse artifact (or
    # a tiny "~2 um2") can't false-flag a pin-mux on area.
    area_budget = floor_area_budget(
        parse_area_budget(_spec_text) if _spec_text else None,
        block_name, _spec_text,
    )
    try:
        _rtl_text = Path(rtl_path).read_text()
    except OSError:
        _rtl_text = ""
    _sram_area = 0.0
    try:
        from orchestrator.langgraph.sram_wrapper import estimate_sram_area_um2
        _sram_area = estimate_sram_area_um2(_rtl_text)
    except Exception:
        _sram_area = 0.0
    # Engine memory-wrapper library sources for the per-block STA/depth probes:
    # without them a cs_sram/cs_fpmem-instantiating block fails `hierarchy
    # -check` in the fan-out-aware buffered STA and is silently denied the
    # base->buffered relaxation its memory-free siblings get.
    _mem_lib_srcs = mem_lib_sources_for_rtl(_rtl_text)
    # Chip-lead uarch_feasibility override (the marker the mem_price gate
    # honors): defer the BUDGET dimensions (area + logic-FF) of this post-synth
    # gate too, instead of re-failing a storage cost the lead already accepted.
    # Hard FF ceiling + timing still gate.
    _budget_overridden = bool(
        ppa_honor_feas_override_enabled()
        and (Path(project_root) / ".coresmith" / "blocks" / block_name
             / "uarch_feasibility_override").exists()
    )

    # rung2 defect 2: seed meta with the parsed budgets so a persisted
    # ppa_history row carries them even on an early elaborate-fail flag.
    _meta: dict = {"budget_ff": ff_budget, "budget_area_um2": area_budget}
    if _budget_overridden:
        _meta["feas_override_deferred"] = True
        log(f"  [PPA] {block_name}: uarch_feasibility_override present -- "
            f"budget dimensions (area/logic-FF) DEFERRED to the die-level "
            f"rollup (hard FF ceiling + timing still gate)", YELLOW)

    def _flag(reasons: list, checks: list | None = None) -> tuple[bool, list, dict]:
        log(f"  [PPA] {block_name} -> diagnose: {'; '.join(reasons)}", RED)
        block_dir = Path(project_root) / ".coresmith" / "blocks" / block_name
        block_dir.mkdir(parents=True, exist_ok=True)
        _err = (
            "PPA gate: RTL is PPA-divergent / un-elaboratable vs the uArch "
            "storage budget:\n- " + "\n- ".join(reasons)
        )
        # A-Fix 4: give the regen prompt the exact budget-vs-actual numbers so it
        # can restructure toward the budget (instead of re-attacking blind). Also
        # drop a machine-readable ppa_report.json for the outer agent / scoreboard.
        if checks:
            _bva = "\n".join(
                f"  {c.get('metric')}: actual={c.get('actual')} "
                f"budget={c.get('budget')} limit={c.get('limit')} "
                f"passed={c.get('passed')}"
                for c in checks
            )
            _err += "\n\nBudget vs actual:\n" + _bva
            try:
                (block_dir / "ppa_report.json").write_text(
                    json.dumps({"checks": checks, "reasons": reasons}, indent=2)
                )
            except OSError:
                pass
        (block_dir / "previous_error.txt").write_text(_err)
        return False, reasons, dict(_meta)

    # HOT-PATCH (chip-lead): honor CORESMITH_SYNTH_TIMEOUT_S so the generic
    # synth probe uses the operator-set wall clock (600s here) instead of the
    # 300s default, which false-fails correct-but-slow synth on ARM (A1.Flex).
    # A timeout at 600s is then a GENUINE un-synthesizability signal, not box slowness.
    import os as _os_to
    _synth_timeout = int(_os_to.environ.get("CORESMITH_SYNTH_TIMEOUT_S", "300") or "300")
    probe = probe_synth_generic(rtl_path, block_name, timeout_s=_synth_timeout)
    # A-Fix 2f: probe_synth_generic returns None ONLY when yosys is absent (the
    # RTL already exists), so a None probe means the PDK-free gate could not run
    # -> tooling missing. The strict profile parks on this (see synthesize_node).
    if probe is None:
        _meta["tooling_missing"] = True
    else:
        # rung2 defect 2: surface the measured PDK-free metrics so the caller
        # persists real numbers (not NULLs) to ppa_history.
        _meta["ff"] = probe.get("logic_ff")
        _meta["mem_bits"] = probe.get("mem_bits")
        _meta["elaborated"] = probe.get("elaborated")
    # A probe that ran but couldn't elaborate (timeout / yosys error) is itself
    # an unsynthesizability signal -- fail it rather than "can't judge".
    if probe is not None and probe.get("elaborated") is False:
        return _flag([probe.get("reason", "design did not elaborate")])
    probe = probe or {}

    # Part D: memory-as-flops hard-block gate (default on; PDK-free). The
    # memory-PRESERVING probe above reads only the block RTL, so an engine
    # memory WRAPPER (cs_sram_*) is an unresolved blackbox there and a
    # wrapped-memory-that-flops is invisible to it. This probe reads the design
    # PLUS the wrapper lib and applies the backend MACRO selection, so a
    # properly wrapped memory becomes a shell (0 flops) while a raw
    # `reg [] mem []` array -- or a cs_sram whose geometry can't bind -- stays a
    # flop array; an ABOVE-threshold flop memory is then a hard fail with
    # actionable guidance. cs_fpmem and sub-threshold memories are never flagged.
    from orchestrator.langgraph.sram_wrapper import (
        fpmem_instances as _fpmem_insts,
    )
    from orchestrator.langgraph.sram_wrapper import (
        gate_memory_as_flops as _gate_mem_flops,
    )
    from orchestrator.langgraph.sram_wrapper import (
        mem_flop_gate_enabled as _mem_flop_gate_on,
    )
    if _mem_flop_gate_on():
        import re as _re_mf
        _has_mem = bool(probe.get("mem_bits")) or bool(
            _re_mf.search(r"\bcs_(?:sram|rom|mem)_1rw|\breg\b[^;]*\[", _rtl_text))
        if _has_mem:
            from orchestrator.langgraph.ppa_check import probe_memory_flops as _probe_mf
            mprobe = _probe_mf([rtl_path] + _mem_lib_srcs, block_name,
                               timeout_s=_synth_timeout, cwd=project_root)
            if mprobe is not None and mprobe.get("elaborated"):
                _mfok, _mfreasons = _gate_mem_flops(
                    mprobe.get("memories") or [],
                    fpmem_geoms=_fpmem_insts(_rtl_text),
                )
                if not _mfok:
                    return _flag(_mfreasons)

    # Cell-explosion synthesizability guard (default on; runs even under
    # SKIP_SYNTH -- PDK-free generic techmap). The memory-PRESERVING FF probe
    # above stops at `proc`, so a combinational-LUT explosion (entropy coding VLC tables
    # as big LUTs, per-mode-replicated intra prediction, a wide record sliced by
    # $func) never materializes as gates and the FF-only check can never fail on
    # it -- the exact class that walls the backend at synthesis. Materialize the
    # cloud with a generic techmap and fail on a techmap timeout or a cell count
    # past the ceiling.
    from orchestrator.langgraph.ppa_check import (
        max_cell_ceiling as _cell_ceiling,
    )
    from orchestrator.langgraph.ppa_check import (
        probe_synth_cellcount as _probe_cells,
    )
    from orchestrator.langgraph.ppa_check import (
        synth_cell_gate_enabled as _cell_gate_on,
    )
    if _cell_gate_on():
        cprobe = _probe_cells(rtl_path, block_name, timeout_s=_synth_timeout)
        if cprobe is not None:
            _meta["cells"] = cprobe.get("cell_count")  # rung2 defect 2
            if cprobe.get("elaborated") is False:
                return _flag([cprobe.get("reason", "did not techmap")])
            _cc = cprobe.get("cell_count")
            _ceil = _cell_ceiling()
            if _cc is not None and _cc > _ceil:
                return _flag([
                    f"gate-level cell count {_cc:,} exceeds the max-cell "
                    f"ceiling {_ceil:,} -- un-synthesizable to a tractable "
                    f"netlist (combinational-LUT explosion / unpipelined "
                    f"datapath cloud). Register the datapath into pipeline "
                    f"stages and map large tables to ROM/LUT, not flat logic."
                ])

    # Combinational-depth guard (fix #4: the pipeline scheduler made enforcing).
    # PDK-free, runs under SKIP_SYNTH. A datapath collapsed into one
    # combinational cloud (the unpipelined RD-search class) has an enormous
    # register-to-register depth; a properly scheduled pipeline keeps each stage
    # bounded. When a PDK is present the real STA/WNS check below also enforces
    # this; this proxy covers the SKIP_SYNTH case where no STA exists.
    from orchestrator.langgraph.ppa_check import (
        logic_depth_advisory_with_pdk_enabled as _depth_advisory_on,
    )
    from orchestrator.langgraph.ppa_check import (
        logic_depth_gate_enabled as _depth_gate_on,
    )
    from orchestrator.langgraph.ppa_check import (
        max_logic_depth as _max_depth,
    )
    from orchestrator.langgraph.ppa_check import (
        probe_logic_depth as _probe_depth,
    )
    from orchestrator.langgraph.ppa_check import (
        sta_tooling_available as _sta_available,
    )
    if _depth_gate_on():
        dprobe = _probe_depth(rtl_path, block_name, timeout_s=_synth_timeout,
                              extra_sources=_mem_lib_srcs)
        if dprobe is not None and dprobe.get("elaborated") is not False:
            _ld = dprobe.get("logic_depth")
            _meta["logic_depth"] = _ld  # rung2 defect 2
            _dmax = _max_depth()
            # Finding 3: when a real PDK + STA are available the depth proxy is
            # ADVISORY. The ltp level count can't tell a converged staged design
            # (881 levels) from a comb cloud (887); left gating it would
            # short-circuit the STA below -> wns_ns=NULL, then the fail-loud path
            # rejects the block for a measurement the proxy itself prevented.
            # Real pre-layout WNS (below) is the timing authority here. A
            # PDK-absent run keeps it gating -- the only depth signal it has.
            _depth_advisory = _depth_advisory_on() and _sta_available(synth_result)
            if _ld is not None and _ld > _dmax:
                if _depth_advisory:
                    _meta["logic_depth_advisory"] = True
                    _meta["logic_depth_max"] = _dmax
                    log(f"  [PPA] {block_name}: logic depth {_ld:,} > {_dmax:,} "
                        f"(ADVISORY -- PDK+STA present; recorded, NOT gating; "
                        f"real pre-layout WNS is the authority)", YELLOW)
                else:
                    return _flag([
                        f"combinational depth {_ld:,} logic levels exceeds the "
                        f"max {_dmax:,} -- the datapath is an unpipelined "
                        f"combinational cloud (won't meet timing / walls synth). "
                        f"Register it into pipeline stages per the stage map."
                    ])

    sta: dict = {}
    if synth_result:
        sta = run_pre_layout_sta(
            synth_result.get("netlist_path", ""), synth_result.get("sdc_path", ""),
            synth_result.get("liberty_path", ""), block_name,
        ) or {}
    # pdk-fixes-1: surface the pre-layout WNS so it lands in the ppa_history
    # wns_ns column (it has always been NULL) and so a LOUD sta_error (STA ran
    # for a block that has a netlist but produced no parseable timing) is
    # visible to the caller / scoreboard instead of silently vanishing.
    _meta["wns_ns"] = sta.get("wns_ns")
    if sta.get("sta_error"):
        _meta["sta_error"] = sta.get("sta_error")
    # Judge the REAL synthesized total FF when we have it. A properly
    # INSTANTIATED sky130 macro is a blackbox (~0 flops), so the real count
    # cleanly separates a macro-backed memory (low FF) from a should-be-SRAM
    # memory that flopped to a reg-array (huge FF) -- the exact case the
    # memory-PRESERVING probe's logic_ff hid (it counts the behavioral FIFO as
    # $mem and excuses it). Fall back to probe.logic_ff only under
    # CORESMITH_SKIP_SYNTH, where no real synthesis ran.
    real_ff = (synth_result or {}).get("ff_count")
    actual_ff = real_ff if real_ff is not None else probe.get("logic_ff")
    _meta["ff"] = actual_ff  # rung2 defect 2: the FF actually judged (real synth wins)
    # Total die area = synthesized std-cells + estimated SRAM macro area. The
    # SRAM term is what makes an oversized frame/output buffer (huge cs_sram)
    # fail the area gate, even though it is 0 flops -- the GDS-intractable case.
    _std_area = (synth_result or {}).get("chip_area_um2")
    actual_area = None
    if _std_area is not None or _sram_area > 0:
        actual_area = (_std_area or 0.0) + _sram_area
    _meta["area_um2"] = actual_area  # rung2 defect 2
    # STORAGE FF (declared buffers / inferred memories kept as flops) separated
    # from LOGIC FF so the FF-budget check judges only logic and never
    # false-flags a legitimate buffer (the reason the gate got disabled). The
    # memory-PRESERVING probe already isolates inferred-$mem storage as mem_bits.
    _storage_ff = probe.get("mem_bits") or 0
    _meta["storage_ff"] = _storage_ff
    # Target clock period (ns) from the block SDC, so a negative-slack verdict
    # can quantify HOW MUCH a register-to-register path is over and how many
    # stages to add (actionable re-pipeline feedback for the scheduler/uArch).
    _period_ns = None
    try:
        from orchestrator.langgraph.ppa_check import parse_sdc_period_ns
        _sdc_p = (synth_result or {}).get("sdc_path", "")
        if _sdc_p and Path(_sdc_p).exists():
            _period_ns = parse_sdc_period_ns(Path(_sdc_p).read_text())
    except Exception:  # noqa: BLE001 - period is best-effort
        _period_ns = None
    _meta["period_ns"] = _period_ns
    # engine-v31 step 1: FAN-OUT-AWARE STA. The base measurement above is the
    # UNBUFFERED mapped netlist -- it extrapolates tens of ns of pure fan-out
    # net delay on a high-fan-out net and systematically FALSE-FAILS designs a
    # real Sky130 set_max_fanout + repair_design pass would close (the AES-v3
    # one-round-per-clock engine: -17.75 ns unbuffered here vs +14.97 ns
    # buffered). Also synthesize a max-fan-out-buffered variant from RTL and
    # gate on max(base, buffered) WNS -- monotonic (buffering only relaxes), so
    # no design that met timing unbuffered can be false-failed. Deterministic;
    # falls back to the base measurement when yosys/sta/liberty are absent.
    _eff_wns = sta.get("wns_ns")
    _eff_sta_error = sta.get("sta_error")
    _liberty_p = (synth_result or {}).get("liberty_path", "")
    if (sta_maxfanout_enabled() and _liberty_p and Path(rtl_path).exists()):
        try:
            from orchestrator.langgraph.pipeline_helpers import _detect_clock_port
            _clk = _detect_clock_port(Path(rtl_path).read_text()) or "clk"
        except Exception:  # noqa: BLE001 - clock detection is best-effort
            _clk = "clk"
        _mf_period = _period_ns if (_period_ns and _period_ns > 0) else 20.0
        mf = run_maxfanout_buffered_sta(
            rtl_path, _liberty_p, block_name, _mf_period, _clk,
            timeout_s=_synth_timeout, extra_sources=_mem_lib_srcs,
        )
        if mf is not None:
            _meta["wns_ns_base_unbuffered"] = _eff_wns
            _meta["wns_ns_buffered"] = mf.get("buffered_wns_ns")
            _meta["fmax_mhz_buffered"] = mf.get("fmax_mhz")
            if mf.get("sta_ok") and mf.get("wns_ns") is not None:
                # Best of the unbuffered mapped-netlist base and the fan-out-
                # buffered RTL synth -- both measure the same reg-to-reg cone.
                _cands = [w for w in (_eff_wns, mf["wns_ns"]) if w is not None]
                _eff_wns = max(_cands) if _cands else mf["wns_ns"]
                _eff_sta_error = None  # a real measurement rescued a base None/err
                log(f"  [PPA] {block_name}: fan-out-aware STA WNS "
                    f"{mf['wns_ns']:+.2f} ns (base {mf.get('base_wns_ns')}, "
                    f"buffered {mf.get('buffered_wns_ns')}); gating on "
                    f"max(unbuffered={_meta.get('wns_ns_base_unbuffered')}, "
                    f"buffered)={_eff_wns:+.2f} ns", GREEN)
            else:
                # The fan-out-aware measurement produced NO timing (both
                # sub-flows errored, or both answered with OpenSTA's
                # no-endpoints sentinel). That used to be silent -- and silence
                # is how +1e39 ns of "slack" got compared to a budget and
                # called "within budget". Say what it saw; and when the base
                # measurement is absent too, hand the reason to the gate so the
                # timing dimension fails CLOSED as unmeasured rather than
                # skipped.
                _mf_err = str(mf.get("sta_error") or "no measurement")
                _meta["sta_maxfanout_error"] = _mf_err
                if _eff_wns is None:
                    _eff_sta_error = _eff_sta_error or _mf_err
                    log(f"  [PPA] {block_name}: fan-out-aware STA produced NO "
                        f"timing and there is no base measurement either -- "
                        f"timing is UNMEASURED: {_mf_err[:220]}", RED)
                else:
                    log(f"  [PPA] {block_name}: fan-out-aware STA produced NO "
                        f"timing ({_mf_err[:220]}) -- keeping the base "
                        f"measurement {_eff_wns:+.2f} ns", YELLOW)
    _meta["wns_ns"] = _eff_wns
    verdict = evaluate_ppa(
        actual_ff=actual_ff,
        ff_budget=ff_budget,
        storage_ff=_storage_ff,
        actual_area_um2=actual_area,
        area_budget_um2=area_budget,
        wns_ns=_eff_wns,
        period_ns=_period_ns,
        # Section 3a: STA-ran-but-no-timing must FAIL CLOSED, not pass silently.
        sta_error=_eff_sta_error,
        budget_overridden=_budget_overridden,
    )
    if not verdict.checks:
        # Nothing measurable -> cannot judge, never block. Carry tooling_missing
        # so the strict profile can park on a yosys-absent unmeasurable gate.
        return None, [], dict(_meta)
    if verdict.ok:
        log(f"  [PPA] {block_name} within budget", GREEN)
        return True, [], dict(_meta)
    return _flag(verdict.reasons, verdict.checks)


def _chip_top_synth_ok(
    project_root: str,
    design_name: str,
    top_rtl_path: str,
    block_rtl_paths: dict,
) -> tuple[bool, str]:
    """Chip-top synthesizability gate (fix #5 + #2 applied to the integrated
    top). ``pipeline_done`` requires the WHOLE chip to be synthesizable, not
    just each block -- the run-B wall was the integrated encoder, not any one
    block. Runs the PDK-free cell-explosion probe on the assembled + deduped
    chip_top sources, so it works even under ``CORESMITH_SKIP_SYNTH``.

    Returns ``(ok, reason)``. ``ok`` is True (never blocks) when the gate is
    disabled, yosys is absent, or sources are missing -- "cannot judge" never
    fails the chip.
    """
    from orchestrator.langgraph.ppa_check import (
        chip_top_min_cells as _cell_floor,
    )
    from orchestrator.langgraph.ppa_check import (
        max_cell_ceiling as _cell_ceiling,
    )
    from orchestrator.langgraph.ppa_check import (
        probe_synth_cellcount_multi as _probe_multi,
    )
    from orchestrator.langgraph.ppa_check import (
        synth_cell_gate_enabled as _cell_gate_on,
    )
    if not _cell_gate_on() or not top_rtl_path or not Path(top_rtl_path).exists():
        return True, ""
    sources = [top_rtl_path] + [p for p in (block_rtl_paths or {}).values() if p]
    try:
        _all_rtl = "\n".join(
            Path(p).read_text() for p in sources if p and Path(p).exists()
        )
        from orchestrator.langgraph.sram_wrapper import (
            uses_wrapper as _uses_wrapper,
        )
        from orchestrator.langgraph.sram_wrapper import (
            wrapper_lib_path as _wrapper_lib_path,
        )
        if _uses_wrapper(_all_rtl):
            sources.append(_wrapper_lib_path())
    except Exception:  # noqa: BLE001 - best effort
        pass
    import tempfile as _tf
    _synth_timeout = int(
        os.environ.get("CORESMITH_SYNTH_TIMEOUT_S", "300") or "300"
    )
    try:
        from orchestrator.langgraph.integration_helpers import (
            _dedup_module_sources,
            _drop_include_provided_sources,
        )
        # SYNTH-SCOPED include-provision dedup: yosys EXPANDS `include`s at
        # read time, so a listed file that another source `include`s
        # double-defines its modules -> MODDUP. Scoped HERE (not inside
        # _dedup_module_sources) because the sim assembly must keep the
        # explicit files: a legacy top may reference block modules directly
        # while a non-top source carries preprocessor-guarded `include`s the
        # sim never expands -- dropping the files there MODMISSINGs the sim.
        _synth_srcs = _drop_include_provided_sources(sources)
        _dd = Path(_tf.mkdtemp(prefix="chiptop_synth_"))
        deduped = _dedup_module_sources(_synth_srcs, _dd)
    except Exception:  # noqa: BLE001 - fall back to raw sources
        deduped = sources
    # F1 (canonical chip_top filelist): publish the deduped one-file-per-
    # module source set the gate actually synthesizes, so downstream tooling
    # (backend P&R, external graders) consumes an authoritative list instead
    # of globbing rtl/**/*.v -- the run tree can carry DUPLICATE module copies
    # (a top/ vs integration/ wrapper variant, an inline vs standalone block),
    # and a naive glob then MODDUP-collides or picks a stale/stub copy.
    try:
        _flist = Path(project_root) / ".coresmith" / "chip_top_sources.f"
        _flist.parent.mkdir(parents=True, exist_ok=True)
        _flist.write_text("\n".join(str(_p) for _p in deduped) + "\n")
    except OSError:
        pass
    # C24: yosys `hierarchy -top` needs the ACTUAL top module of the assembled
    # chip_top, NOT the DESIGN NAME. The deterministic Caravel assembly's top
    # module is `user_project_wrapper` (or openframe_project_wrapper); passing
    # design_name (e.g. a `<design>_qspi_rom_top`) made yosys fail "Module <design>
    # not found" and falsely report EVERY chip as un-synthesizable at the final
    # gate. Resolve the real top from the assembled RTL: prefer a Caravel
    # wrapper module, else the last module declared (the top is conventionally
    # last), else the parsed first module, else fall back to design_name.
    _top_name = design_name
    try:
        _top_txt = Path(top_rtl_path).read_text(errors="ignore")
        import re as _re24
        _mods = _re24.findall(r"^\s*module\s+([A-Za-z_]\w*)", _top_txt, _re24.MULTILINE)
        for _pref in ("openframe_project_wrapper", "user_project_wrapper"):
            if _pref in _mods:
                _top_name = _pref
                break
        else:
            if _mods:
                _top_name = _mods[-1]
    except OSError:
        pass
    # F3 (audit): a run may DELIVER a separate locked-ABI top at
    # rtl/chip_top.v (e.g. the ppab_dut chassis contract) that is NOT part of
    # the assembled manifest -- two near-equivalent tops that silently drift
    # apart (the reference codec encoder shipped `ppab_dut` while chip_top_sources.f
    # rooted at `reference_codec_enc_top`). When such a file exists outside the deduped
    # set and its module names are all novel, co-elaborate it, publish it in
    # the canonical filelist, and probe it as a SECOND top below so the gate
    # covers the artifact that actually gets graded. Either way record a
    # carried-forward defect naming the dual-top drift risk.
    _delivered_top = None
    try:
        import re as _re3
        _dpath = Path(project_root) / "rtl" / "chip_top.v"
        _dedup_resolved = {str(Path(_p).resolve()) for _p in deduped}
        if _dpath.exists() and str(_dpath.resolve()) not in _dedup_resolved:
            _mod_re = _re3.compile(r"^\s*module\s+([A-Za-z_]\w*)", _re3.MULTILINE)
            _d_mods = _mod_re.findall(_dpath.read_text(errors="ignore"))
            _defined: set = set()
            for _p in deduped:
                try:
                    _defined.update(
                        _mod_re.findall(Path(_p).read_text(errors="ignore")))
                except OSError:
                    continue
            _collisions = [m for m in _d_mods if m in _defined]
            if _d_mods and not _collisions:
                deduped = list(deduped) + [str(_dpath)]
                try:
                    _flist.write_text(
                        "\n".join(str(_p) for _p in deduped) + "\n")
                except (OSError, NameError):
                    pass
                _delivered_top = _d_mods[-1]
                _drift_detail = (
                    f"delivered ABI top `{_delivered_top}` ({_dpath}) is not "
                    "part of the assembled manifest -- co-elaborated + probed "
                    "as a second top; unify on ONE canonical top (a locked-ABI "
                    "wrapper around the integration module)")
            else:
                _drift_detail = (
                    f"delivered top file {_dpath} redefines manifest modules "
                    f"{_collisions[:4]} -- cannot co-elaborate, so its drift "
                    "vs the assembled manifest is UNCHECKED")
            record_carried_forward_defect(project_root, {
                "gate": "chip_top_synth",
                "kind": "canonical_top_drift",
                "unmodeled": str(_dpath),
                "detail": _drift_detail,
            })
    except Exception:  # noqa: BLE001 - detection is best-effort
        _delivered_top = None
    # C27: probe from the PROJECT ROOT so project-relative $readmemh init
    # files (cs_sram/cs_rom INIT_FILE="inputs/...") resolve; the deduped
    # source copies live in a temp dir but yosys resolves $readmemh against
    # its own cwd, not the source file's directory.
    probe = _probe_multi(
        deduped, _top_name, timeout_s=_synth_timeout, cwd=project_root,
    )
    if probe is None:
        return True, ""  # no yosys -> cannot judge, never block
    if probe.get("elaborated") is False:
        return False, probe.get("reason", "chip_top did not techmap")
    _cc = probe.get("cell_count")
    _ceil = _cell_ceiling()
    if _cc is not None and _cc > _ceil:
        return False, (
            f"chip_top gate-level cell count {_cc:,} exceeds the max-cell "
            f"ceiling {_ceil:,} -- the integrated design is un-synthesizable "
            f"to a tractable netlist"
        )
    _floor = _cell_floor()
    if _cc is not None and _floor > 0 and _cc < _floor:
        return False, (
            f"chip_top collapsed to {_cc:,} gate cells (< floor "
            f"{_floor:,}) -- it elaborated but synthesized to a near-"
            f"empty netlist, so its block instances were optimized away "
            f"(outputs never reach a primary chip I/O, or a stub/duplicate "
            f"wrapper won assembly dedup). A synthesizable-but-empty top "
            f"is not a working chip_top."
        )
    # F3: the DELIVERED ABI top (rtl/chip_top.v) must synthesize too -- it is
    # the artifact that gets graded, and it can drift from the assembled
    # manifest independently.
    if _delivered_top and _delivered_top != _top_name:
        probe2 = _probe_multi(
            deduped, _delivered_top, timeout_s=_synth_timeout,
            cwd=project_root,
        )
        if probe2 is not None:
            if probe2.get("elaborated") is False:
                return False, (
                    f"delivered ABI top `{_delivered_top}` (rtl/chip_top.v) "
                    f"did not techmap: {probe2.get('reason', '')} -- the "
                    "delivered top drifted from the assembled manifest")
            _cc2 = probe2.get("cell_count")
            if _cc2 is not None and _cc2 > _ceil:
                return False, (
                    f"delivered ABI top `{_delivered_top}` cell count "
                    f"{_cc2:,} exceeds the max-cell ceiling {_ceil:,}")
            if _cc2 is not None and _floor > 0 and _cc2 < _floor:
                return False, (
                    f"delivered ABI top `{_delivered_top}` collapsed to "
                    f"{_cc2:,} gate cells (< floor {_floor:,}) -- a "
                    "synthesizable-but-empty delivered top is not a working "
                    "chip_top")
    return True, ""


def _resolve_run_die_budget(project_root: str) -> tuple[float | None, str]:
    """Resolve the run's die-area budget (mm^2) + source for the measured rollup.

    env CORESMITH_DIE_BUDGET_MM2 > PRD ``max_die_area_mm2`` (``.coresmith/
    prd_spec.json``) > shuttle default (a shuttle named in the requirements).
    """
    from orchestrator.langgraph import mem_price as _mprice
    prd = None
    try:
        p = Path(project_root) / ".coresmith" / "prd_spec.json"
        if p.exists():
            prd = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        prd = None
    reqs = ""
    try:
        rp = Path(project_root) / "inputs" / "requirements.md"
        if rp.exists():
            reqs = rp.read_text(errors="replace")
    except OSError:
        reqs = ""
    ers_tech = ""
    try:
        ers_tech = json.dumps((prd or {}).get("technology", {}))
    except (TypeError, ValueError):
        ers_tech = ""
    return _mprice.resolve_die_budget_mm2(
        prd=prd, requirements=reqs, ers_technology_text=ers_tech)


def _container_block_names(project_root: str, block_names: list) -> set:
    """Blocks whose measured per-block synth CONTAINS other listed blocks.

    An integration-top block built as a self-contained source (its RTL --
    directly or via one level of ``include`` -- instantiates other listed
    blocks) synthesizes FLAT: its measured area already includes the leaves.
    """
    inc_re = re.compile(r'^\s*`include\s+"([^"]+)"', re.MULTILINE)
    out: set = set()
    for name in block_names:
        try:
            br = json.loads(
                (Path(project_root) / ".coresmith" / "blocks" / name
                 / "best_result.json").read_text())
            # Resolve the block's measured RTL: rtl_target when recorded;
            # otherwise glob rtl/**/<name>.v and prefer the file whose sha1
            # matches best_result.rtl_sha1 (older runs record the hash but
            # not the path -- exactly the run this fix was built for).
            cands: list = []
            if br.get("rtl_target"):
                rp0 = Path(br["rtl_target"])
                if not rp0.is_absolute():
                    rp0 = Path(project_root) / rp0
                cands.append(rp0)
            else:
                cands = sorted(
                    (Path(project_root) / "rtl").glob(f"**/{name}.v"))
            rp = None
            want_sha = str(br.get("rtl_sha1") or "")
            for c in cands:
                if not c.exists():
                    continue
                if want_sha:
                    import hashlib as _hl
                    if _hl.sha1(c.read_bytes()).hexdigest() == want_sha:
                        rp = c
                        break
                if rp is None:
                    rp = c
            text = rp.read_text(errors="ignore") if rp else ""
            for m in inc_re.finditer(text):
                ip = Path(m.group(1))
                if not ip.is_absolute():
                    ip = rp.parent / m.group(1)
                if ip.exists():
                    text += "\n" + ip.read_text(errors="ignore")
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        if not text:
            continue
        for other in block_names:
            if other == name:
                continue
            if re.search(
                rf"\b{re.escape(other)}\s+(?:#|[a-zA-Z_]\w*\s*\()", text
            ):
                out.add(name)
                break
    return out


def _subsume_container_items(items: list, containers: set) -> tuple:
    """Drop the double-count between flat container blocks and their leaves.

    Returns ``(items, note)``: when both containers and leaves carry measured
    area, keep whichever side is LARGER -- the honest design area is
    max(flat container, sum-of-leaves), never their sum -- and drop the
    other, with a note for the log/report.
    ``CORESMITH_DIE_ROLLUP_CONTAINER_DEDUP=0`` restores the legacy sum.
    """
    if (os.environ.get("CORESMITH_DIE_ROLLUP_CONTAINER_DEDUP", "1")
            or "1").strip() == "0":
        return items, ""
    cont = [i for i in items if i.name in containers]
    leaf = [i for i in items if i.name not in containers]
    if not cont or not leaf:
        return items, ""
    cont_max = max(i.area_um2 for i in cont)
    leaf_sum = sum(i.area_um2 for i in leaf)
    if leaf_sum >= cont_max:
        return leaf, (
            f"container block(s) {sorted(i.name for i in cont)} subsumed by "
            f"their leaves (flat {cont_max:,.0f} um2 <= sum-of-leaves "
            f"{leaf_sum:,.0f} um2) -- not double-counted")
    biggest = max(cont, key=lambda i: i.area_um2)
    return [biggest], (
        f"leaf blocks subsumed by flat container {biggest.name} "
        f"({cont_max:,.0f} um2 >= sum-of-leaves {leaf_sum:,.0f} um2) -- "
        f"not double-counted")


def _die_cap_excludes_sram(project_root: str) -> bool:
    """True when the run's specs declare the standard-cell area cap EXCLUDES
    blackboxed SRAM macros.

    The old detection grepped THREE literal phrases in the PRD only; a live
    run whose PRD said "blackboxed out of the scored standard-cell count" and
    whose ERS said "excluding the named SRAM macro instances exactly once"
    matched none of them, so the rollup folded 2.44 mm^2 of deferred macro
    ledger area into a 2.32 mm^2 std-cell cap and failed a delivered design.
    Detection now scans BOTH prd_spec.json and ers_spec.json with the literal
    markers plus a same-sentence proximity match of (exclud*/blackbox*) with
    (sram/macro). ``CORESMITH_DIE_ROLLUP_BROAD_EXCLUSION=0`` restores the
    legacy PRD-only literal-marker behavior.
    """
    _LITERALS = (
        "sram black boxes excluded",
        "sram footprints are zero-area black boxes",
        "memories blackboxed",
    )
    broad = (os.environ.get("CORESMITH_DIE_ROLLUP_BROAD_EXCLUSION", "1")
             or "1").strip() != "0"
    files = ("prd_spec.json", "ers_spec.json") if broad else ("prd_spec.json",)
    text = ""
    for fn in files:
        try:
            p = Path(project_root) / ".coresmith" / fn
            if p.exists():
                text += p.read_text(errors="replace").lower() + "\n"
        except OSError:
            continue
    if not text:
        return False
    if any(m in text for m in _LITERALS):
        return True
    if not broad:
        return False
    return bool(re.search(
        r"(?:exclud\w*|blackbox\w*)[^.\n]{0,80}(?:sram|macro)"
        r"|(?:sram|macro)[^.\n]{0,80}(?:exclud\w*|blackbox\w*)",
        text,
    ))


def _measured_die_rollup(project_root: str, block_names: list):
    """Post-synth die-area rollup from measured per-block PPA (Deliverable 2).

    Per-block area = the block's latest ``ppa_history.area_um2``.  When the PRD
    says its die-area cap excludes declared SRAM black boxes, that cap is
    evaluated strictly on those measured standard-cell areas; deferred SRAM
    ledgers cannot be substituted into the standard-cell cap.  Otherwise the
    legacy macro-inclusive ledger floor is retained. Returns a
    ``DieRollupVerdict`` (``has_cap`` False when no die cap resolves), or None
    when the gate is disabled.
    """
    from orchestrator.langgraph import mem_price as _mprice
    if not _mprice.die_rollup_gate_enabled():
        return None
    cap_mm2, source = _resolve_run_die_budget(project_root)
    excludes_sram = _die_cap_excludes_sram(project_root)
    sb = _scoreboard(project_root)
    items: list = []
    for name in block_names:
        area = None
        src = ""
        if sb is not None:
            try:
                row = sb.latest_ppa(name) or {}
                area = row.get("area_um2")
                if area is not None:
                    src = "ppa_history"
            except Exception:  # noqa: BLE001
                area = None
        # Defer hygiene: for an over-budget DEFERRED block the ledger's priced
        # area (which includes the oversized store's macro area) must FLOOR the
        # rollup contribution, so a smaller synthesized/absent ppa_history area
        # cannot mask the deferred excess.
        led = _mprice.read_ledger(project_root, name) or {}
        led_area = _mprice.read_block_ledger_area_um2(project_root, name)
        if (not excludes_sram) and led.get("over_budget") and led_area is not None:
            if area is None or led_area > float(area):
                area, src = led_area, "mem_ledger_deferred"
        if (not excludes_sram) and area is None and led_area is not None:
            area, src = led_area, "mem_ledger"
        if area is not None and area > 0:
            items.append(_mprice.RollupItem(name=name, area_um2=float(area), source=src))
    # Container hygiene: an integration-top block whose per-block synth is a
    # FLAT elaboration of the leaves it instantiates already CONTAINS their
    # area -- summing both double-counts the design (0.250 flat top + 0.254
    # leaves rolled to 0.504 against a 0.337 budget the 0.254 design fits).
    items, _subsume_note = _subsume_container_items(
        items, _container_block_names(project_root, block_names))
    if _subsume_note:
        log(f"  [DIE-ROLLUP] {_subsume_note}", YELLOW)
    std_rollup = _mprice.evaluate_die_rollup(
        items, die_budget_mm2=cap_mm2, budget_source=source)

    if not excludes_sram or not std_rollup.ok:
        return std_rollup

    # A grading-excluded SRAM is still physically real.  When the FRD/ERS
    # supplies a separate macro-inclusive planning cap, enforce it in addition
    # to the PRD standard-cell cap.  The FRD cap already represents the chosen
    # planning utilization, so do not add a second interconnect multiplier.
    macro_cap_mm2 = None
    try:
        _ers_text = (Path(project_root) / ".coresmith" / "ers_spec.json").read_text(
            errors="replace")
        _m = re.search(
            r"macro-inclusive(?:\s+planning)?\s+area\s*<\s*([0-9]+(?:\.[0-9]+)?)\s*mm2",
            _ers_text,
            re.IGNORECASE,
        )
        if _m:
            macro_cap_mm2 = float(_m.group(1))
    except (OSError, ValueError):
        macro_cap_mm2 = None

    macro_items = list(items)
    for name in block_names:
        led = _mprice.read_ledger(project_root, name) or {}
        for mem in led.get("memories") or []:
            if str(mem.get("declared_impl", "")).lower() != "sram":
                continue
            try:
                mem_area = float(mem.get("area_um2") or 0)
            except (TypeError, ValueError):
                mem_area = 0.0
            if mem_area > 0:
                macro_items.append(_mprice.RollupItem(
                    name=f"{name}:{mem.get('name', 'sram')}",
                    area_um2=mem_area,
                    source=str(mem.get("estimate_source") or "mem_ledger"),
                ))

    if macro_cap_mm2 is not None:
        macro_rollup = _mprice.evaluate_die_rollup(
            macro_items,
            die_budget_mm2=macro_cap_mm2,
            budget_source="frd_macro_inclusive",
            margin=0.0,
        )
        if not macro_rollup.ok:
            return macro_rollup
        std_rollup.macro_total_um2 = macro_rollup.total_um2
        std_rollup.macro_budget_mm2 = macro_cap_mm2
        std_rollup.macro_budget_ok = True

    return std_rollup


def _run_gate_sim_gate(
    state: BlockState, block: dict, block_name: str,
    synth_result: dict | None, rtl_path: str,
) -> tuple:
    """Run the post-synthesis gate-level simulation gate for one block.

    Returns ``(gate_sim_ok, status, reason)`` where ``gate_sim_ok`` is:

    * ``True``  -- the netlist reproduced the verified RTL cycle-for-cycle;
    * ``False`` -- it DIVERGED, would not elaborate, or produced a blank /
      missing / vacuous verdict (``route_after_synth`` sends this to diagnose);
    * ``None``  -- the gate did not apply (disabled, synthesis produced no
      netlist, toolchain/PDK absent). ALWAYS recorded with a reason so absence
      is visible rather than silently reading as success.

    Never raises: gate plumbing must not crash the synth node. A plumbing error
    is reported as ``not_run`` (or, under ``CORESMITH_GATE_SIM_STRICT``, as a
    failure by the harness itself).
    """
    from orchestrator.harness import gate_sim as _gs

    if not _gs.gate_sim_enabled():
        log("  [GATE-SIM] !!! GATE DISABLED (CORESMITH_GATE_SIM=0) -- the "
            "SYNTHESIZED NETLIST is never simulated; a synthesis-side stub "
            "cannot be caught", RED)
        return (None, _gs.STATUS_DISABLED, f"{_gs.GATE_SIM_ENV}=0")

    if not synth_result or not state.get("tb_path"):
        reason = ("no netlist produced by synthesis" if not synth_result
                  else "no testbench available to source reference vectors")
        log(f"  [GATE-SIM] NOT RUN -- {reason}", YELLOW)
        return (None, _gs.STATUS_NOT_RUN, reason)

    netlist_path = (synth_result or {}).get("netlist_path", "")
    log("  [GATE-SIM] Replaying verified RTL vectors through the gate "
        "netlist...", YELLOW)
    try:
        res = _gs.check_gate_sim(
            block=block,
            netlist_path=netlist_path,
            rtl_path=rtl_path,
            tb_path=state.get("tb_path", ""),
            attempt=state.get("attempt", 1),
        )
    except Exception as exc:  # noqa: BLE001 - never crash the synth node
        strict = _gs.gate_sim_strict()
        log(f"  [GATE-SIM] harness error: {exc}"
            f"{' (STRICT -> FAIL)' if strict else ' (non-blocking)'}", RED)
        return (False if strict else None, _gs.STATUS_NOT_RUN,
                f"gate-sim harness error: {exc}")

    try:
        block_dir = Path(_pr(state)) / ".coresmith" / "blocks" / block_name
        block_dir.mkdir(parents=True, exist_ok=True)
        (block_dir / "gate_sim_report.json").write_text(
            json.dumps(res.as_dict(), indent=2)
        )
    except OSError:
        pass

    write_graph_event(_pr(state), "Gate Sim", "gate_result", {
        "block": block_name, "name": "gate_level_sim", "kind": "gate_sim",
        "status": res.status, "passed": res.status == _gs.STATUS_PASS,
        "cycles_compared": res.cycles_compared,
        "reason": res.reason,
    })

    if res.status == _gs.STATUS_PASS:
        log(f"  [GATE-SIM] PASS -- netlist matched the verified RTL for "
            f"{res.cycles_compared:,} cycles "
            f"({res.output_bits_compared:,} output bits)", GREEN)
        return (True, res.status, res.reason)

    if res.status == _gs.STATUS_FAIL:
        log(f"  [GATE-SIM] FAIL -- {res.reason}", RED)
        try:
            block_dir = Path(_pr(state)) / ".coresmith" / "blocks" / block_name
            block_dir.mkdir(parents=True, exist_ok=True)
            (block_dir / "previous_error.txt").write_text(
                res.as_prev_error(block_name)
            )
        except OSError:
            pass
        return (False, res.status, res.reason)

    # not_run / disabled -- never a pass, always visible.
    log(f"  [GATE-SIM] NOT RUN -- {res.reason}", YELLOW)
    return (None, res.status, res.reason)


async def synthesize_node(state: BlockState) -> dict:
    """Run Yosys synthesis with local LLM fix loop.

    If synthesis fails, calls an LLM to fix the RTL for synthesizability
    and re-runs -- up to ``MAX_LOCAL_RETRIES`` times.  After local fixes
    are exhausted, the routing function sends failures to the diagnose
    lead for deeper analysis.
    """
    _guard_rtl_phase(state, "synthesize_node")
    block = state["current_block"]
    block_name = block["name"]

    # Honor CORESMITH_SKIP_SYNTH=1 so hosts with no Sky130 PDK can still
    # complete RTL + sim.  Treat as a no-op success.
    import os as _os
    if _os.environ.get("CORESMITH_SKIP_SYNTH") == "1":
        import sys as _sys
        _loud = (f"  [SYNTH] !!! GATE DISABLED (CORESMITH_SKIP_SYNTH=1) for "
                 f"{block_name} -- un-synthesizable RTL WILL NOT be caught")
        log(_loud, RED)
        print("=" * 78 + "\n" + _loud.strip() + "\n" + "=" * 78,
              file=_sys.stderr, flush=True)
        # rung2 defect 2: the PDK-free synthesizability probes (generic
        # elaborate / cell-explosion / logic-depth / generic-FF-budget) need no
        # PDK, so they MUST still run and GATE under SKIP_SYNTH -- otherwise
        # SKIP_SYNTH silently hides an un-synthesizable design (the exact hole
        # the synth-gate-default work closed). Pass require_gate_flag=False so
        # they run even when CORESMITH_PPA_GATE is not seeded; yosys-absent
        # falls into the tooling_missing park path below. Only area/WNS are
        # genuinely unavailable here. Real probe metrics are recorded (was NULL).
        ppa_ok, ppa_reasons, ppa_meta = _evaluate_ppa_gate(
            _pr(state), block_name, state.get("rtl_path", ""), None,
            require_gate_flag=False,
        )
        # rung3-fixes-1 (minor 4): the SKIP_SYNTH row references
        # ppa_report.json, but the PDK-free probe only writes that file on a
        # FAIL (via _evaluate_ppa_gate's _flag). On a PASS it never existed, so
        # the scoreboard row pointed at a nonexistent report. Write the
        # same-shape report from the probe metrics whenever the gate didn't
        # already drop one -- the fail path keeps its richer {checks, reasons}
        # report; the pass path now has a real file at the recorded path.
        _skip_report_path = (
            Path(_pr(state)) / ".coresmith" / "blocks" / block_name
            / "ppa_report.json"
        )
        try:
            _skip_report_path.parent.mkdir(parents=True, exist_ok=True)
            if not _skip_report_path.exists():
                _skip_report_path.write_text(json.dumps({
                    "probe": "skip_synth",
                    "ppa_ok": ppa_ok,
                    "reasons": ppa_reasons or [],
                    "checks": [],
                    "ff": ppa_meta.get("ff"),
                    "cells": ppa_meta.get("cells"),
                    "mem_bits": ppa_meta.get("mem_bits"),
                    "area_um2": ppa_meta.get("area_um2"),
                    "elaborated": ppa_meta.get("elaborated"),
                    "budget_ff": ppa_meta.get("budget_ff"),
                    "budget_area_um2": ppa_meta.get("budget_area_um2"),
                }, indent=2))
        except OSError:
            pass
        _record_ppa_row(
            _pr(state), block=block_name, attempt=state.get("attempt", 0),
            source="gate", probe="skip_synth", ppa_ok=ppa_ok,
            reasons=ppa_reasons or None,
            cells=ppa_meta.get("cells"),
            ff=ppa_meta.get("ff"),
            mem_bits=ppa_meta.get("mem_bits"),
            area_um2=ppa_meta.get("area_um2"),
            wns_ns=ppa_meta.get("wns_ns"),
            elaborated=ppa_meta.get("elaborated"),
            budget_ff=ppa_meta.get("budget_ff"),
            budget_area_um2=ppa_meta.get("budget_area_um2"),
            report_path=str(_skip_report_path),
        )
        if _ppa_should_park_tooling_missing(_pr(state), ppa_ok, ppa_meta):
            _park_ppa_unmeasurable(state, block_name)
        return {"synth_success": True, "synth_gate_count": 0,
                "ppa_ok": ppa_ok, "ppa_reasons": ppa_reasons,
                # No yosys run -> no netlist -> nothing for the gate-level sim
                # to simulate. Recorded explicitly (never blank) so the absence
                # of a gate-sim verdict is visible downstream.
                "gate_sim_ok": None, "gate_sim_status": "not_run",
                "gate_sim_reason": "CORESMITH_SKIP_SYNTH=1 -- no netlist was "
                                   "produced, so the gate netlist was never "
                                   "simulated",
                "phase": "synth"}

    rtl_path = state.get("rtl_path", "")
    if not rtl_path or not Path(rtl_path).exists():
        log("  [SYNTH] Skipped -- RTL file not found", RED)
        return {"synth_success": False, "synth_gate_count": 0, "phase": "synth"}

    write_graph_event(_pr(state), "Synthesize", "graph_node_enter", {
        "block": block_name,
    })

    result = None
    synth_ok = False
    gate_count = 0

    with _tracer.start_as_current_span(f"Synthesize [{block_name}]") as span:
        span.set_attribute("block_name", block_name)

        # PRE-SYNTH STORAGE GATE. A wide flat-packed reg sliced by a RUNTIME
        # index (directly, or inside a helper function) lowers to a barrel-
        # shifter/decoder cloud that yosys `proc` cannot elaborate -> the 600 s
        # timeout. Detect it in milliseconds BEFORE paying the timeout and hand
        # the regen an actionable report (which regs -> cs_fpmem) instead of a
        # raw log tail. The codec RD-core burned ~12 h on exactly this; the
        # detector + diagnosis already existed but were never wired.
        # CORESMITH_STORAGE_PRESYNTH_GATE=0 restores straight-to-yosys.
        storage_gate_failed = False
        if _os.environ.get("CORESMITH_STORAGE_PRESYNTH_GATE", "1").strip() != "0":
            try:
                from orchestrator.langgraph.rtl_storage_lint import (
                    find_flat_packed_dynamic_storage,
                    find_oversized_memory_arrays,
                    format_lint_report,
                    format_memory_tier_report,
                )
                _src = Path(rtl_path).read_text()
                _rpt = find_flat_packed_dynamic_storage(_src)
                # Section 5f/4a: register-tier memory over the SRAM threshold (or
                # whose single-cycle flat read mux busts the period) flattens to a
                # giant mux -- untimeable. Catch it structurally BEFORE synth, the
                # same class the flat-packed gate catches. Priced against the run's
                # target clock so an untimeable single-cycle read is caught too.
                _period = 1000.0 / max(1.0, float(state.get("target_clock_mhz", 50.0)))
                _mtr = find_oversized_memory_arrays(_src, period_ns=_period)
                if not _rpt.ok:
                    storage_gate_failed = True
                    _msg = format_lint_report(_rpt, block=block_name)
                    log(f"  [SYNTH] PRE-SYNTH STORAGE GATE failed "
                        f"({len(_rpt.findings)} flat-packed reg(s) w/ dynamic "
                        f"part-select) -- skipping yosys (would time out), "
                        f"routing actionable fix to regen", RED)
                    result = {
                        "success": False, "gate_count": 0,
                        "log": ("UNSYNTHESIZABLE -- pre-synth storage lint "
                                "(yosys NOT run):\n\n" + _msg),
                    }
                    span.set_attribute("storage_gate_failed", True)
                elif not _mtr.ok:
                    storage_gate_failed = True
                    _msg = format_memory_tier_report(_mtr, block=block_name)
                    log(f"  [SYNTH] PRE-SYNTH MEMORY-TIER GATE failed "
                        f"({len(_mtr.findings)} register-tier memor(y/ies) over "
                        f"the SRAM threshold) -- skipping yosys (untimeable flat "
                        f"read mux), routing macro-tier fix to regen", RED)
                    result = {
                        "success": False, "gate_count": 0,
                        "log": ("UNSYNTHESIZABLE -- pre-synth memory-tier lint "
                                "(yosys NOT run):\n\n" + _msg),
                    }
                    span.set_attribute("memory_tier_gate_failed", True)
            except Exception as _e:  # never let the gate crash the node
                log(f"  [SYNTH] pre-synth storage gate error: {_e}", RED)

        local_attempt = 0
        for local_attempt in range(0 if storage_gate_failed
                                   else (1 + MAX_LOCAL_RETRIES)):
            log(f"  [SYNTH] Running Yosys synthesis"
                f"{f' (local fix #{local_attempt})' if local_attempt > 0 else ''}...",
                YELLOW)
            result = await asyncio.to_thread(
                synthesize_block,
                block, rtl_path,
                target_clock_mhz=state.get("target_clock_mhz", 50.0),
                attempt=state["attempt"],
            )

            if result["success"]:
                synth_ok = True
                gate_count = result.get("gate_count", 0)
                area = result.get("chip_area_um2", 0.0)
                area_str = f", {area:,.1f} µm²" if area else ""
                log(f"  [SYNTH] SUCCESS: {gate_count:,} cells{area_str}"
                    f"{f' (after {local_attempt} local fix(es))' if local_attempt > 0 else ''}",
                    GREEN)
                span.set_attribute("success", True)
                span.set_attribute("gate_count", gate_count)
                span.set_attribute("chip_area_um2", area)
                span.set_attribute("local_fixes", local_attempt)
                break

            log("  [SYNTH] FAILED", RED)
            log(f"    {result.get('log', '')[:200]}", RED)

            if local_attempt < MAX_LOCAL_RETRIES:
                log(f"  [SYNTH] Attempting local LLM fix ({local_attempt + 1}/{MAX_LOCAL_RETRIES})...", YELLOW)
                write_graph_event(_pr(state), "Synth Fix", "llm_start", {
                    "block": block_name, "local_attempt": local_attempt + 1,
                })

                fixed_rtl = await fix_synth_errors(
                    block_name, rtl_path, result.get("log_path", ""),
                    callbacks=_callbacks(state),
                )

                write_graph_event(_pr(state), "Synth Fix", "llm_end", {
                    "block": block_name, "local_attempt": local_attempt + 1,
                    "fix_produced": fixed_rtl is not None,
                })

                if fixed_rtl:
                    log("  [SYNTH] Local fix applied, re-synthesizing...", YELLOW)
                else:
                    log("  [SYNTH] LLM could not produce a fix, escalating to diagnose", RED)
                    break
            else:
                log("  [SYNTH] Local retries exhausted, escalating to diagnose", RED)

        span.set_attribute("success", synth_ok)
        span.set_attribute("gate_count", gate_count)

    synth_log = ""
    if result:
        synth_log = result.get("log", "") or result.get("errors", "")

    if not synth_ok and result:
        block_dir = Path(_pr(state)) / ".coresmith" / "blocks" / block_name
        block_dir.mkdir(parents=True, exist_ok=True)
        (block_dir / "previous_error.txt").write_text(synth_log[-5000:])

    write_graph_event(_pr(state), "Synthesize", "graph_node_exit", {
        "block": block_name,
        "success": synth_ok,
        "gate_count": gate_count,
        "chip_area_um2": result.get("chip_area_um2", 0.0) if result else 0.0,
        "local_fixes_attempted": min(local_attempt + 1, MAX_LOCAL_RETRIES) if result and not synth_ok else 0,
        "tool_stdout": smart_truncate(synth_log, 2000, "head_tail") if synth_log else "",
        "log_path": result.get("log_path", "") if result else "",
    })

    existing_logs = dict(state.get("step_log_paths") or {})
    if result and result.get("log_path"):
        existing_logs["synthesize"] = result["log_path"]

    # Deterministic PPA gate (CORESMITH_PPA_GATE=1). A block can synthesize
    # cleanly yet blow its budget -- e.g. a memory that should be an SRAM macro
    # became a flop array. The gate (memory-preserving FF count vs the uArch
    # flip_flop_budget, + area/WNS when available) routes over-budget blocks
    # back to diagnose (see route_after_synth) so the synth_fixer restructures.
    ppa_ok, ppa_reasons, ppa_meta = (None, [], {})
    if synth_ok:
        ppa_ok, ppa_reasons, ppa_meta = _evaluate_ppa_gate(
            _pr(state), block_name, rtl_path, result,
        )
        if _ppa_should_park_tooling_missing(_pr(state), ppa_ok, ppa_meta):
            _park_ppa_unmeasurable(state, block_name)

    # --- POST-SYNTHESIS GATE-LEVEL SIMULATION (CORESMITH_GATE_SIM) ----------
    # Everything above this line was measured on two DIFFERENT artifacts: DV /
    # coverage on the RTL source, PPA on the netlist. Nothing had ever run the
    # NETLIST against the behaviour the RTL was verified with, so a module split
    # into a simulation implementation and a (stub) synthesis implementation
    # passed every gate in the pipeline. This replays the verified RTL's own
    # port vectors through the gate netlist and fails closed on divergence.
    # Only meaningful once synthesis SUCCEEDED: a failed synth has already
    # written its own actionable previous_error.txt (which this gate would
    # otherwise overwrite), and any netlist still on disk is a STALE one from a
    # previous attempt -- simulating that would produce a verdict about code
    # that is no longer the design.
    if synth_ok:
        gate_sim_ok, gate_sim_status, gate_sim_reason = _run_gate_sim_gate(
            state, block, block_name, result, rtl_path,
        )
    else:
        gate_sim_ok, gate_sim_status, gate_sim_reason = (
            None, "not_run", "synthesis failed -- no current netlist to simulate",
        )

    # B3: record the authoritative PPA verdict + measured numbers.
    # pdk-fixes-1: wns_ns is now available from the pre-layout STA (threaded
    # through ppa_meta) -- persist it so the ppa_history.wns_ns column stops
    # being unconditionally NULL when timing was actually measured.
    _record_ppa_row(
        _pr(state), block=block_name, attempt=state.get("attempt", 0),
        source="gate", probe="synth",
        ff=(result or {}).get("ff_count"),
        cells=(result or {}).get("gate_count"),
        area_um2=(result or {}).get("chip_area_um2"),
        wns_ns=ppa_meta.get("wns_ns"),
        ppa_ok=ppa_ok, reasons=ppa_reasons or None,
        report_path=(result or {}).get("report_path", ""),
    )

    return {
        "synth_success": synth_ok,
        "synth_gate_count": gate_count,
        "ppa_ok": ppa_ok,
        "ppa_reasons": ppa_reasons,
        "gate_sim_ok": gate_sim_ok,
        "gate_sim_status": gate_sim_status,
        "gate_sim_reason": gate_sim_reason,
        "phase": "synth",
        "step_log_paths": existing_logs,
    }


# ---------------------------------------------------------------------------
# Node: diagnose
# ---------------------------------------------------------------------------

def _compose_actionable_error(diag: dict, raw_log: str, max_chars: int = 5000) -> str:
    """Build an actionable ``previous_error.txt`` from a structured diagnosis.

    The regen agent (``rtl_generator``) and the diagnose node both read
    ``previous_error.txt``. Historically that file held ONLY the raw tool-log
    tail (``synth_log[-5000:]`` / ``sim_log[-5000:]``) -- so the *actionable*
    diagnosis the diagnose node produced (``suggested_fix`` + per-constraint
    ``code_snippet``) never reached the fixer, which then re-attacked the same
    wall blind (the codec RD-core burned ~12 h of synth/fix loops this way).

    This leads the file with the diagnosis (fix + specific code) and keeps a
    trimmed raw-log tail as supporting context. ``CORESMITH_ROUTE_DIAGNOSIS=0``
    restores the old raw-log-only behavior.
    """
    parts: list[str] = []
    cat = diag.get("category", "UNKNOWN")
    conf = diag.get("confidence", "")
    diagnosis = (diag.get("diagnosis") or "").strip()
    fix = (diag.get("suggested_fix") or "").strip()
    parts.append(
        f"DIAGNOSIS [{cat}"
        + (f", confidence {conf}" if conf != "" else "")
        + f"]: {diagnosis}"
    )
    if fix:
        parts.append(f"\nSUGGESTED FIX:\n{fix}")
    constraints = diag.get("constraints") or []
    if constraints:
        lines = ["\nSPECIFIC FIXES (apply these):"]
        for c in constraints:
            if isinstance(c, dict):
                desc = (c.get("description") or c.get("rule") or "").strip()
                snippet = (c.get("code_snippet") or c.get("snippet") or "").strip()
                fpath = (c.get("file") or "").strip()
                head = "- " + (desc or "fix")
                if fpath:
                    head += f"  [{fpath}]"
                lines.append(head)
                if snippet:
                    lines.append(f"    {snippet}")
            elif isinstance(c, str) and c.strip():
                lines.append(f"- {c.strip()}")
        parts.append("\n".join(lines))
    head_text = "\n".join(parts).strip()
    tail_budget = max(0, max_chars - len(head_text) - 80)
    log_tail = (raw_log or "")[-tail_budget:] if tail_budget else ""
    if log_tail:
        head_text += (
            "\n\n--- raw tool log (supporting context, truncated) ---\n" + log_tail
        )
    return head_text[:max_chars]


def _route_diagnosis_to_previous_error(block_dir, diag: dict, raw_log: str) -> bool:
    """Overwrite ``previous_error.txt`` with the actionable diagnosis.

    Returns True if it routed (a structured fix existed and routing is enabled).
    Gated by ``CORESMITH_ROUTE_DIAGNOSIS`` (default on) per the repo's
    env-gating convention so the prior raw-log behavior stays restorable.
    """
    if os.environ.get("CORESMITH_ROUTE_DIAGNOSIS", "1").strip() == "0":
        return False
    if not (diag.get("suggested_fix") or diag.get("constraints")):
        return False
    (block_dir / "previous_error.txt").write_text(
        _compose_actionable_error(diag, raw_log)
    )
    return True


def _failure_signature(error_log: str) -> str:
    """Structural fingerprint of a failure, volatile tokens normalized out.

    Two attempts that fail the SAME structural way (same error class, same
    module, same offending construct) hash to the same signature even though
    line numbers, hex addresses, paths, and counts differ. Used to detect a
    diagnose/retry loop that keeps re-hitting an identical wall (Section 5g).
    """
    import hashlib
    import re as _re
    s = (error_log or "")[-4000:]  # the tail carries the actionable error
    s = _re.sub(r"0x[0-9a-fA-F]+", "0x#", s)     # addresses / handles
    s = _re.sub(r"/[^\s:'\"]+", "/PATH", s)       # file paths
    s = _re.sub(r"\d+", "#", s)                    # line numbers, counts, times
    s = _re.sub(r"\s+", " ", s).strip().lower()
    return hashlib.sha1(s.encode("utf-8", "replace")).hexdigest()[:16]


async def diagnose_node(state: BlockState) -> dict:
    """Run DebugAgent to analyze the most recent failure."""
    block = state["current_block"]
    block_name = block["name"]
    phase = state.get("phase", "unknown")

    write_graph_event(_pr(state), "Diagnose Failure", "graph_node_enter", {
        "block": block_name, "phase": phase,
    })

    block_dir = Path(_pr(state)) / ".coresmith" / "blocks" / block_name
    block_dir.mkdir(parents=True, exist_ok=True)

    error_file = block_dir / "previous_error.txt"
    error_log = error_file.read_text() if error_file.exists() else "Unknown failure"

    # PER-BLOCK WALL BUDGET (opt-in: CORESMITH_BLOCK_WALL_BUDGET_S, 0=off). A
    # single block must not consume unbounded wall-clock re-attacking the same
    # structural wall (the codec RD-core burned ~13 h, ~9 lifecycles, on one
    # un-synthesizable block). Record first-seen on the first diagnose; once the
    # elapsed failure-time exceeds the budget, escalate to a human (ask_human)
    # instead of looping -- the diagnose node had already flagged needs_human.
    _wall_budget = float(os.environ.get("CORESMITH_BLOCK_WALL_BUDGET_S", "0") or 0)
    if _wall_budget > 0:
        import json as _jw
        import time as _time
        _seen = block_dir / "_first_seen.txt"
        if not _seen.exists():
            _seen.write_text(str(_time.time()))
        else:
            try:
                _elapsed = _time.time() - float(_seen.read_text().strip())
            except Exception:  # noqa: BLE001
                _elapsed = 0.0
            if _elapsed > _wall_budget:
                log(f"  [DIAGNOSE] BLOCK WALL BUDGET exceeded for {block_name} "
                    f"({_elapsed:.0f}s > {_wall_budget:.0f}s) -- escalating to "
                    f"human instead of another retry", RED)
                _wb = {
                    "category": "WALL_BUDGET_EXCEEDED",
                    "confidence": 1.0,
                    "diagnosis": (f"Block {block_name} exceeded its wall-clock "
                                  f"budget ({_elapsed:.0f}s > {_wall_budget:.0f}s) "
                                  f"without closing -- a structural wall "
                                  f"(un-synthesizable RTL or non-composing model). "
                                  f"Stopping the retry loop."),
                    "suggested_fix": ("Human review: simplify the block's uArch "
                                      "(smaller search/storage/datapath), or accept "
                                      "a conformant simplified fallback."),
                    "needs_human": True, "escalate": True,
                    "is_testbench_bug": False, "local_fix_possible": False,
                    "constraints": [], "affected_blocks": [block_name],
                }
                (block_dir / "diagnosis.json").write_text(_jw.dumps(_wb, indent=2))
                write_graph_event(_pr(state), "Diagnose Failure",
                                  "graph_node_exit", {
                    "block": block_name, "category": "WALL_BUDGET_EXCEEDED",
                    "confidence": 1.0, "needs_human": True,
                })
                return {"debug_action": "ask_human"}

    # Short-circuit: SIM_TIMEOUT (engine fix 2026-06-23). A pure sim timeout
    # carries no PSNR/divergence, so there is nothing for the debug LLM to
    # diagnose. Classify it as its own SIM_TIMEOUT category (NOT the generic
    # INFRASTRUCTURE_ERROR) so the router retries with run_simulation's
    # auto-extended cap instead of escalating to a human after 2 rounds.
    if "SIM_TIMEOUT:" in error_log or "Simulation exceeded" in error_log:
        log("  [DIAGNOSE] Sim timeout (no verdict) -- retry with extended cap", YELLOW)
        import json as _json
        _to_diag = {
            "category": "SIM_TIMEOUT",
            "confidence": 0.0,
            "diagnosis": "Simulation exceeded its wall-clock cap before "
                         "producing a functional verdict. Not a code bug -- "
                         "the block is slow. Next attempt uses an extended "
                         "timeout (x1.5, capped).",
            "suggested_fix": "Retry with extended sim timeout. If the block is "
                             "correct but slow, consider a smaller per-block DV "
                             "stimulus or a declared sim_timeout_s/runtime_target_s.",
            "needs_human": False,
            "is_testbench_bug": False,
            "escalate": False,
            "local_fix_possible": False,
            "constraints": [],
            "affected_blocks": [],
        }
        _ah_path = block_dir / "attempt_history.json"
        _hist = _json.loads(_ah_path.read_text()) if _ah_path.exists() else []
        _hist.append({
            "attempt": state["attempt"],
            "phase": phase,
            "error": error_log[:500],
            "category": "SIM_TIMEOUT",
        })
        _ah_path.write_text(_json.dumps(_hist, indent=2))
        (block_dir / "diagnosis.json").write_text(_json.dumps(_to_diag, indent=2))
        write_graph_event(_pr(state), "Diagnose Failure", "graph_node_exit", {
            "block": block_name, "category": "SIM_TIMEOUT",
            "confidence": 0.0, "needs_human": False,
        })
        return {"debug_action": "retry_rtl"}

    # Short-circuit: detect infrastructure failures (LLM timeout/crash)
    # and skip the debug LLM call which would likely also fail.
    _INFRA_MARKERS = ("[ClaudeLLM error:", "timed out", "exit_code=-9",
                      "circuit breaker open")
    if any(m in error_log for m in _INFRA_MARKERS):
        log("  [DIAGNOSE] Infrastructure failure detected, skipping debug LLM", YELLOW)
        infra_diag = {
            "category": "INFRASTRUCTURE_ERROR",
            "confidence": 0.0,
            "diagnosis": "LLM infrastructure failure (timeout/crash), not a code bug.",
            "suggested_fix": "Retry after backoff.",
            "needs_human": False,
            "is_testbench_bug": False,
            "escalate": False,
            "constraints": [],
            "affected_blocks": [],
        }
        import json as _json
        _ah_path = block_dir / "attempt_history.json"
        history = _json.loads(_ah_path.read_text()) if _ah_path.exists() else []
        history.append({
            "attempt": state["attempt"],
            "phase": phase,
            "error": error_log[:500],
            "category": "INFRASTRUCTURE_ERROR",
        })
        _ah_path.write_text(_json.dumps(history, indent=2))
        (block_dir / "diagnosis.json").write_text(_json.dumps(infra_diag, indent=2))
        write_graph_event(_pr(state), "Diagnose Failure", "graph_node_exit", {
            "block": block_name, "category": "INFRASTRUCTURE_ERROR",
            "confidence": 0.0, "needs_human": False,
        })
        return {"debug_action": "retry_rtl"}

    # DIAGNOSE STABILITY (Section 5g): fingerprint the STRUCTURAL failure. If the
    # SAME signature repeats on consecutive attempts, the retry loop is stuck on
    # one wall -- a third identical retry burns tokens for zero new information.
    # Force escalation instead. Transient classes (SIM_TIMEOUT, infra) already
    # returned above, so they never reach here. Gated CORESMITH_DIAGNOSE_SIG_ESCALATE
    # (default on); threshold CORESMITH_DIAGNOSE_SIG_MAX (default 3 = escalate on
    # the 3rd identical failure).
    if os.environ.get("CORESMITH_DIAGNOSE_SIG_ESCALATE", "1").strip() != "0":
        import json as _json
        _sig = _failure_signature(error_log)
        _sig_file = block_dir / "_failure_sig.json"
        _prev = {}
        if _sig_file.exists():
            try:
                _prev = _json.loads(_sig_file.read_text())
            except Exception:  # noqa: BLE001
                _prev = {}
        _run = (int(_prev.get("run", 0)) + 1) if _prev.get("sig") == _sig else 1
        try:
            _sig_file.write_text(_json.dumps({"sig": _sig, "run": _run}))
        except OSError:
            pass
        try:
            _sig_max = max(2, int(os.environ.get("CORESMITH_DIAGNOSE_SIG_MAX", "3")))
        except ValueError:
            _sig_max = 3
        if _run >= _sig_max:
            log(f"  [DIAGNOSE] SAME structural failure signature {_run}x in a row "
                f"for {block_name} ({_sig}) -- escalating (identical retries are "
                f"not converging)", RED)
            _sig_diag = {
                "category": "UARCH_SPEC_ERROR",
                "confidence": 1.0,
                "diagnosis": (
                    f"Block {block_name} has hit the IDENTICAL structural failure "
                    f"{_run} attempts in a row (signature {_sig}); the retry loop "
                    f"is not converging -- the wall is architectural, not a local "
                    f"code slip. Last error tail:\n{error_log[-800:]}"),
                "suggested_fix": (
                    "Human/architectural review: revise the uArch spec (simplify "
                    "the datapath/storage/schedule) rather than regenerating the "
                    "same RTL again."),
                "needs_human": True, "escalate": True,
                "is_testbench_bug": False, "local_fix_possible": False,
                "constraints": [], "affected_blocks": [block_name],
            }
            (block_dir / "diagnosis.json").write_text(_json.dumps(_sig_diag, indent=2))
            write_graph_event(_pr(state), "Diagnose Failure", "graph_node_exit", {
                "block": block_name, "category": "UARCH_SPEC_ERROR",
                "confidence": 1.0, "needs_human": True,
                "failure_signature": _sig, "repeat_count": _run,
            })
            return {"debug_action": "ask_human"}

    # Fast-path: detect known testbench bugs via regex to skip expensive
    # opus diagnosis call (~80-100s per invocation).
    import re as _re
    _fast_diag = None
    if phase == "sim":
        if "has no attribute" in error_log or "AttributeError" in error_log:
            _fast_diag = {
                "category": "TESTBENCH_BUG",
                "confidence": 1.0,
                "diagnosis": "Testbench references a DUT port that does not exist.",
                "suggested_fix": "Regenerate testbench with correct port names from RTL.",
                "needs_human": False,
                "is_testbench_bug": True,
                "escalate": False,
                "constraints": [],
                "affected_blocks": [],
            }
        elif "ModuleNotFoundError" in error_log or "ImportError" in error_log:
            _fast_diag = {
                "category": "TESTBENCH_BUG",
                "confidence": 1.0,
                "diagnosis": "Testbench has a missing Python import.",
                "suggested_fix": "Regenerate testbench without external dependencies.",
                "needs_human": False,
                "is_testbench_bug": True,
                "escalate": False,
                "constraints": [],
                "affected_blocks": [],
            }
        elif _re.search(r"cocotb\.result\.TestFail.*Timer\(0\)", error_log):
            _fast_diag = {
                "category": "TESTBENCH_BUG",
                "confidence": 0.95,
                "diagnosis": "Testbench uses Timer(0) causing Verilator delta-cycle race.",
                "suggested_fix": "Regenerate testbench; use FallingEdge/RisingEdge instead of Timer(0).",
                "needs_human": False,
                "is_testbench_bug": True,
                "escalate": False,
                "constraints": [],
                "affected_blocks": [],
            }
    elif phase == "lint":
        if "Module not found" in error_log or "Cannot find file" in error_log:
            _fast_diag = {
                "category": "INFRASTRUCTURE_ERROR",
                "confidence": 1.0,
                "diagnosis": "RTL file missing or module name mismatch.",
                "suggested_fix": "Regenerate RTL.",
                "needs_human": False,
                "is_testbench_bug": False,
                "escalate": False,
                "constraints": [],
                "affected_blocks": [],
            }

    if _fast_diag:
        log(f"  [DIAGNOSE] Fast-path: {_fast_diag['category']} "
            f"(skipped opus LLM call)", GREEN)
        import json as _json
        _ah_path = block_dir / "attempt_history.json"
        history = _json.loads(_ah_path.read_text()) if _ah_path.exists() else []
        history.append({
            "attempt": state["attempt"],
            "phase": phase,
            "error": error_log[:500],
            "category": _fast_diag["category"],
        })
        _ah_path.write_text(_json.dumps(history, indent=2))
        (block_dir / "diagnosis.json").write_text(_json.dumps(_fast_diag, indent=2))
        fast_action = "retry_tb" if _fast_diag.get("is_testbench_bug") else "retry_rtl"
        write_graph_event(_pr(state), "Diagnose Failure", "graph_node_exit", {
            "block": block_name, "category": _fast_diag["category"],
            "confidence": _fast_diag["confidence"], "needs_human": False,
            "fast_path": True,
        })
        return {"debug_action": fast_action}

    with _tracer.start_as_current_span(f"Diagnose [{block_name}]") as span:
        span.set_attribute("block_name", block_name)
        span.set_attribute("failed_phase", phase)

        diag = await diagnose_failure(
            block_name=block_name,
            phase=phase,
            project_root=_pr(state),
            callbacks=_callbacks(state),
        )

        category = diag.get("category", "UNKNOWN")
        span.set_attribute("category", category)
        span.set_attribute("needs_human", diag.get("needs_human", False))

    import json as _json

    (block_dir / "diagnosis.json").write_text(_json.dumps(diag, indent=2))

    # Route the structured diagnosis into previous_error.txt so the REGEN
    # (rtl_generator reads previous_error.txt, not diagnosis.json) gets the
    # actionable fix -- suggested_fix + per-constraint code snippets -- instead
    # of the raw tool-log tail it historically re-attacked blind.
    if _route_diagnosis_to_previous_error(block_dir, diag, error_log):
        log("  [DIAGNOSE] Routed actionable diagnosis -> previous_error.txt", GREEN)

    _ah_path = block_dir / "attempt_history.json"
    history = _json.loads(_ah_path.read_text()) if _ah_path.exists() else []
    history.append({
        "attempt": state["attempt"],
        "phase": phase,
        "error": error_log[:500],
        "category": category,
    })
    _ah_path.write_text(_json.dumps(history, indent=2))

    action = _route_decision(
        debug_result=diag,
        attempt_history=history,
        attempt=state["attempt"],
        max_attempts=state["max_attempts"],
        phase=phase,
    )

    write_graph_event(_pr(state), "Diagnose Failure", "graph_node_exit", {
        "block": block_name,
        "category": category,
        "confidence": diag.get("confidence", 0),
        "needs_human": diag.get("needs_human", False),
        "suggested_fix": str(diag.get("suggested_fix", ""))[:300],
        "diagnosis_preview": str(diag.get("diagnosis", ""))[:300],
    })

    return {"debug_action": action}


# ---------------------------------------------------------------------------
# Node: decide (deterministic -- no LLM call)
# ---------------------------------------------------------------------------

def _route_decision(debug_result: dict, attempt_history: list[dict],
                    attempt: int, max_attempts: int, phase: str) -> str:
    """Deterministic failure routing based on debug agent output."""
    category = debug_result.get("category", "UNKNOWN")
    confidence = debug_result.get("confidence", 0.5)
    needs_human = debug_result.get("needs_human", False)
    escalate = debug_result.get("escalate", False)

    # High-confidence machine-applicable fix -> auto-retry instead of asking
    # a human.  The LLM debug agent often hedges with needs_human=True even
    # when it has a concrete suggested_fix (e.g. the codec run's
    # transform_select fp16-static-function bug came back at confidence=0.92
    # with a precise `code_snippet`, but `needs_human=True` triggered an
    # ask_human escalation that resolved to instant retry with no new
    # context).  Skip the round-trip: the diagnosis.json is already on disk
    # and the next retry's prompt reads it.
    #   Tunable via CORESMITH_AUTO_FIX_CONFIDENCE (default 0.85).
    suggested_fix = str(debug_result.get("suggested_fix") or "").strip()
    local_fix_possible = debug_result.get("local_fix_possible", True)
    try:
        auto_fix_threshold = float(os.environ.get("CORESMITH_AUTO_FIX_CONFIDENCE", "0.85"))
    except ValueError:
        auto_fix_threshold = 0.85
    if (
        needs_human
        and confidence >= auto_fix_threshold
        and local_fix_possible
        and len(suggested_fix) >= 50
    ):
        needs_human = False  # overridden -- the retry path injects the fix

    # Count how many times each category has occurred
    category_counts: dict[str, int] = {}
    for entry in attempt_history:
        cat = entry.get("category", "UNKNOWN")
        category_counts[cat] = category_counts.get(cat, 0) + 1

    # Rule -1: SIM_TIMEOUT (engine fix 2026-06-23) -- a pure sim timeout
    # produced NO functional verdict, so the diagnose agent had nothing to
    # root-cause. run_simulation auto-extends the cap (x1.5, max 1800s) on
    # each timeout, so we RETRY (which re-runs the sim with more room) rather
    # than letting a timeout consume the INFRASTRUCTURE->ask_human or
    # same-category->escalate budget the way a real functional failure does.
    # Bounded: after the cap has been hit enough times that the timeout can no
    # longer grow (CORESMITH_SIM_TIMEOUT_MAX_RETRIES, default 4) we stop and
    # escalate so a genuinely-hung block can't loop forever.
    if category == "SIM_TIMEOUT":
        try:
            _sim_to_max = int(os.environ.get("CORESMITH_SIM_TIMEOUT_MAX_RETRIES", "4"))
        except ValueError:
            _sim_to_max = 4
        if category_counts.get("SIM_TIMEOUT", 0) >= _sim_to_max:
            return "escalate"
        return "retry_rtl"

    # Rule 0: Infrastructure errors get special handling -- escalate on 2+
    if category == "INFRASTRUCTURE_ERROR":
        if category_counts.get("INFRASTRUCTURE_ERROR", 0) >= 2:
            return "ask_human"
        return "retry_rtl"

    # Rule 1: Same category 3+ times -> stuck in a loop, escalate
    if category_counts.get(category, 0) >= 3:
        return "escalate"

    # Rule 2: Explicit escalation or human-needed flag
    if escalate:
        return "escalate"
    if needs_human:
        return "ask_human"

    # Rule 3: Out of retries
    if attempt >= max_attempts:
        return "escalate"

    # Rule 4: Low confidence -> human should look
    if confidence < 0.3:
        return "ask_human"

    # Rule 5: Testbench bug -> regenerate testbench, not RTL
    if debug_result.get("is_testbench_bug"):
        return "retry_tb"

    # Rule 6: Route based on failed phase
    if phase == "sim":
        return "retry_rtl"  # sim failure -> regenerate RTL
    if phase == "synth":
        return "retry_rtl"  # synth failure -> regenerate RTL
    if phase == "lint":
        return "retry_rtl"  # lint failure -> regenerate RTL

    # Default: retry
    return "retry_rtl"


async def decide_node(state: BlockState) -> dict:
    """Deterministic failure routing with attempt management.

    Reads debug_action from diagnose_node.  For RTL retries, increments
    the attempt counter and checks max_attempts (overriding to escalate
    if exhausted).  For TB retries, sets force_regen_tb.  Handles
    infrastructure backoff.
    """
    block = state["current_block"]
    block_name = block["name"]
    action = state.get("debug_action", "retry_rtl")

    block_title = block_name.replace("_", " ").title()

    with _tracer.start_as_current_span(f"Route Decision [{block_title}]") as span:
        span.set_attribute("block_name", block_name)
        span.set_attribute("attempt", state["attempt"])
        span.set_attribute("decision", action)

        update: dict = {}

        if action == "retry_tb":
            update["force_regen_tb"] = True

        elif action == "retry_rtl":
            # SIM_TIMEOUT retries (engine fix 2026-06-23) do NOT consume the
            # functional attempt budget: a pure timeout produced no verdict,
            # so it isn't a "real" diagnose round. run_simulation auto-extends
            # the cap each time; the SIM_TIMEOUT route rule self-bounds via
            # CORESMITH_SIM_TIMEOUT_MAX_RETRIES. Re-run the same attempt# with
            # more wall-clock so the encoder gets a real diagnosis instead of
            # exhausting attempts on repeated timeouts.
            _diag_cat = None
            try:
                _dp = Path(_pr(state)) / ".coresmith" / "blocks" / block_name / "diagnosis.json"
                if _dp.exists():
                    _diag_cat = json.loads(_dp.read_text()).get("category")
            except (json.JSONDecodeError, OSError):
                _diag_cat = None
            if _diag_cat == "SIM_TIMEOUT":
                log(f"  [RETRY] SIM_TIMEOUT -- re-running attempt {state['attempt']} "
                    f"with extended sim timeout (budget not consumed)", YELLOW)
                span.set_attribute("final_decision", action)
                write_graph_event(_pr(state), "Route Decision", "graph_node_exit", {
                    "block": block_name,
                    "decision": action,
                    "sim_timeout_retry": True,
                })
                return update

            new_attempt = state["attempt"] + 1
            if new_attempt > state["max_attempts"]:
                log(f"  [DECIDE] Retries exhausted ({state['max_attempts']} max), escalating", RED)
                action = "escalate"
                update["debug_action"] = "escalate"
            else:
                update["attempt"] = new_attempt
                log(f"  [RETRY] Attempt {new_attempt}/{state['max_attempts']}", YELLOW)

                block_dir = Path(_pr(state)) / ".coresmith" / "blocks" / block_name
                diag_path = block_dir / "diagnosis.json"
                if diag_path.exists():
                    try:
                        diag = json.loads(diag_path.read_text())
                        if diag.get("category") == "INFRASTRUCTURE_ERROR":
                            backoff_s = min(30 * (2 ** (new_attempt - 1)), 120)
                            log(f"  [RETRY] Backing off {backoff_s}s after infra failure", YELLOW)
                            await asyncio.sleep(backoff_s)
                    except (json.JSONDecodeError, OSError):
                        pass

        span.set_attribute("final_decision", action)

        write_graph_event(_pr(state), "Route Decision", "graph_node_exit", {
            "block": block_name,
            "decision": action,
            "attempt": state["attempt"],
        })

        return update


# ---------------------------------------------------------------------------
# Node: ask_human  (INTERRUPT)
# ---------------------------------------------------------------------------

async def ask_human_node(state: BlockState) -> dict:
    """Pause the graph and surface failure details to the outer agent.

    One of two nodes that call ``interrupt()`` (the other is
    ``review_uarch_spec_node``).  The outer agent (Claude Code via MCP
    tools) inspects the payload and resumes with
    ``Command(resume={"action": "...", ...})``.
    """
    block = state["current_block"]
    block_name = block["name"]
    state.get("debug_result", {})

    write_graph_event(_pr(state), "Ask Human", "graph_node_enter", {
        "block": block_name, "attempt": state["attempt"],
    })

    log(f"  [HUMAN] Intervention needed for {block_name}", YELLOW)

    import json as _json
    block_dir = Path(_pr(state)) / ".coresmith" / "blocks" / block_name
    block_dir.mkdir(parents=True, exist_ok=True)

    diag_path = block_dir / "diagnosis.json"
    diag = _json.loads(diag_path.read_text()) if diag_path.exists() else {}

    ah_path = block_dir / "attempt_history.json"
    attempt_history = _json.loads(ah_path.read_text()) if ah_path.exists() else []

    error_path = block_dir / "previous_error.txt"
    error_text = error_path.read_text() if error_path.exists() else ""

    constr_path = block_dir / "constraints.json"
    constraints = _load_constraints_safe(constr_path)

    category_counts: dict[str, int] = {}
    for entry in attempt_history:
        cat = entry.get("category", "UNKNOWN")
        category_counts[cat] = category_counts.get(cat, 0) + 1

    payload = {
        "type": "human_intervention_needed",
        "block_name": block_name,
        "attempt": state["attempt"],
        "max_attempts": state.get("max_attempts", 5),
        "phase": state.get("phase", ""),
        "error": error_text[:2000],
        "diagnosis": diag.get("diagnosis", ""),
        "category": diag.get("category", ""),
        "suggested_fix": diag.get("suggested_fix", ""),
        "confidence": diag.get("confidence", 0.5),
        "needs_human": diag.get("needs_human", False),
        "human_question": diag.get("human_question", ""),
        "attempt_history": attempt_history[-5:],
        "category_counts": category_counts,
        "constraints": constraints,
        # File paths for outer-agent diagnosis
        "rtl_path": str(
            Path(state["project_root"]) / block.get("rtl_target", "")
        ),
        "uarch_spec_path": str(
            Path(state["project_root"]) / "arch" / "uarch_specs"
            / f"{block_name}.md"
        ),
        # Step log file paths for outer-agent diagnosis
        "step_log_paths": dict(state.get("step_log_paths") or {}),
        # Testbench path
        "testbench_path": str(
            Path(state["project_root"]) / block.get("testbench", "")
        ),
        # Project-root-relative paths for all artifacts
        "relative_paths": {
            "rtl": block.get("rtl_target", ""),
            "testbench": block.get("testbench", ""),
            "uarch_spec": f"arch/uarch_specs/{block_name}.md",
            "ers": ".coresmith/ers_spec.json",
        },
        "supported_actions": [
            "retry", "fix_rtl", "fix_tb", "add_constraint", "skip", "abort",
        ],
        # Guidance for the outer agent
        "outer_agent_guidance": (
            "You are the outer-loop diagnostic agent. Do not auto-accept or "
            "blindly retry. Read the OTEL events, step logs, RTL, uarch spec, "
            "testbench, VCD/WaveKit audit, and ERS contract before choosing an "
            "action:\n"
            "1. Classify the root cause and cite concrete evidence.\n"
            "2. If the failure is infrastructure or testbench-only, fix that "
            "shared issue first, then explicitly choose retry or fix_tb.\n"
            "3. If the failure is RTL/spec behavior, edit the relevant RTL or "
            "add a precise constraint, then resume with fix_rtl or "
            "add_constraint.\n"
            "4. If the measurable ERS KPI cannot be verified or the evidence "
            "is inconclusive, escalate to a human with the missing facts.\n"
            "5. Record a rationale with every decision."
        ),
    }

    # Add ERS summary context (non-fatal if missing)
    try:
        import json as _json
        ers_path = Path(state["project_root"]) / ".coresmith" / "ers_spec.json"
        if ers_path.exists():
            ers_data = _json.loads(ers_path.read_text(encoding="utf-8"))
            ers_doc = ers_data.get("ers", {})
            ers_info = {
                "summary": ers_doc.get("summary", "")[:2000],
                "bus_protocol": ers_doc.get("dataflow", {}).get("bus_protocol", ""),
                "data_width_bits": ers_doc.get("dataflow", {}).get("data_width_bits", 0),
            }
            payload["ers_summary"] = ers_info
    except Exception:
        pass

    # Add RTL snippet (first 100 lines, non-fatal if missing)
    try:
        rtl_file = Path(state["project_root"]) / block.get("rtl_target", "")
        if rtl_file.exists():
            rtl_lines = rtl_file.read_text(encoding="utf-8").splitlines()[:100]
            payload["rtl_snippet"] = "\n".join(rtl_lines)[:3000]
    except Exception:
        pass

    response = interrupt(payload)

    write_graph_event(_pr(state), "Ask Human", "graph_node_exit", {
        "block": block_name, "action": response.get("action", "unknown"),
    })

    action = response.get("action", "abort")
    updated: dict = {"human_response": response}

    if action == "add_constraint" and response.get("constraint"):
        constraints.append({
            "rule": response["constraint"],
            "source": "human",
            "attempt": state["attempt"],
        })
        constr_path.write_text(_json.dumps(constraints, indent=2))

    if action == "fix_rtl" and response.get("description"):
        constraints.append({
            "rule": f"Outer-agent RTL fix applied: {response['description']}",
            "source": "human",
            "attempt": state["attempt"],
        })
        constr_path.write_text(_json.dumps(constraints, indent=2))

    return updated


# ---------------------------------------------------------------------------
# Node: block_done  (terminal node in the block subgraph)
# ---------------------------------------------------------------------------

async def block_done_node(state: BlockState) -> dict:
    """Record block result.  This is the terminal node of the block subgraph.

    Replaces the old ``advance_block_node`` -- no longer advances a queue
    index; instead the result flows back to the orchestrator via the
    ``completed_blocks`` reducer.
    """
    block = state["current_block"]
    block_name = block["name"]
    attempt = state["attempt"]

    sim_passed = state.get("sim_passed", False)
    synth_success = state.get("synth_success", False)
    gate_count = state.get("synth_gate_count", 0)
    phase = state.get("pipeline_phase", "rtl")

    human_resp = state.get("human_response") or {}
    is_skip = human_resp.get("action") == "skip"
    is_abort = human_resp.get("action") == "abort"
    is_escalate = state.get("debug_action") == "escalate"

    # Pass 1 of the two-pass flow (phase "uarch") only produces a uArch spec +
    # Amaranth block model -- there is NO RTL/sim/synth yet, so success is gated on
    # the spec being approved (R2), NOT on sim_passed AND synth_success. Pass 2
    # ("rtl") and the flag-off default keep the historical sim+synth gate.
    if phase == "uarch":
        all_passed = (
            state.get("uarch_approved", False)
            and not is_skip and not is_abort and not is_escalate
        )
    else:
        all_passed = (
            sim_passed and synth_success
            and not is_skip and not is_abort and not is_escalate
        )

    step_log_paths = dict(state.get("step_log_paths") or {})

    block_dir = Path(_pr(state)) / ".coresmith" / "blocks" / block_name
    constr_path = block_dir / "constraints.json"
    constraints = _load_constraints_safe(constr_path)

    # When this completion event happened. `completed_blocks` is append-only, so
    # membership alone cannot tell a LEFTOVER interrupt (the graph moved past it
    # and the block then finished) from a LIVE one (the block finished a pass
    # ago and is parked again now). The daemon compares this against when the
    # interrupt was raised; without it every pass-2 interrupt in a two-pass run
    # was labelled stale on arrival and live_interrupt_count read 0.
    completed_at = _time.time()

    if all_passed:
        result = {
            "name": block_name,
            "success": True,
            "attempts": attempt,
            "gate_count": gate_count,
            "synth_success": True,
            "constraints_learned": len(constraints),
            "step_log_paths": step_log_paths,
            "phase": phase,
            "completed_at": completed_at,
        }
        log(f"  [{block_name}] PASSED (attempt {attempt})", GREEN)
    else:
        error_path = block_dir / "previous_error.txt"
        error_text = error_path.read_text()[:500] if error_path.exists() else ""
        result = {
            "name": block_name,
            "success": False,
            "attempts": attempt,
            "error": error_text,
            "constraints_learned": len(constraints),
            "skipped": is_skip,
            "escalated": is_escalate,
            "aborted": is_abort,
            "sim_passed": sim_passed,
            "synth_success": synth_success,
            "step_log_paths": step_log_paths,
            "phase": phase,
            "completed_at": completed_at,
        }
        reason = (
            "aborted" if is_abort
            else "skipped" if is_skip
            else "escalated" if is_escalate
            else "failed"
        )
        log(f"  [{block_name}] {reason.upper()} after {attempt} attempts", RED)

    write_graph_event(_pr(state), "Block Done", "graph_node_exit", {
        "block": block_name, "success": result["success"],
    })

    return {
        "completed_blocks": [result],
    }


# ---------------------------------------------------------------------------
# Block-level routing functions
# ---------------------------------------------------------------------------

def route_after_uarch_review(state: BlockState) -> str:
    """Route after uarch spec review.

    Two-pass: in phase ``"uarch"`` (pass 1) an approved spec routes to
    ``block_done`` -- the per-block path only produces the spec + Amaranth block
    model; the chip-level µarch gate (not this block path) validates the
    decomposition. ``revise`` still re-specs; ``skip`` still ends the block.

    Phase ``"rtl"`` (and flag-off default) is unchanged: approve -> generate_rtl,
    revise -> generate_uarch_spec, skip -> block_done.
    """
    response = state.get("human_response") or {}
    action = response.get("action", "abort")
    phase = state.get("pipeline_phase", "rtl")
    if action == "revise":
        return "generate_uarch_spec"
    if action == "skip":
        return "block_done"
    # approve / default
    if phase == "uarch":
        return "block_done"
    return "generate_rtl"


route_after_uarch_review.__edge_labels__ = {
    "generate_rtl": "APPROVED",
    "generate_uarch_spec": "REVISE",
    "block_done": "SKIP",
}


def route_after_init(state: BlockState) -> str:
    """Route after init_block.

    Two-pass: in phase ``"rtl"`` (pass 2) the uArch spec + Amaranth block model
    already exist on disk from pass 1, so skip re-spec and go straight to
    ``generate_rtl`` (which reuses the on-disk spec/model). Phase ``"uarch"``
    (pass 1) and the flag-off default go to ``generate_uarch_spec`` -- identical
    to today's hard ``init_block -> generate_uarch_spec`` edge.
    """
    if state.get("pipeline_phase", "rtl") == "rtl" and state.get(
        "uarch_pass_done"
    ):
        return "generate_rtl"
    return "generate_uarch_spec"


route_after_init.__edge_labels__ = {
    "generate_uarch_spec": "SPEC",
    "generate_rtl": "RTL (pass 2)",
}


def route_after_rtl(state: BlockState) -> str:
    """Route after RTL generation + lint: CLEAN -> testbench, FAIL -> diagnose."""
    return "generate_testbench" if state.get("lint_clean") else "diagnose"


route_after_rtl.__edge_labels__ = {
    "generate_testbench": "LINT CLEAN",
    "diagnose": "LINT FAIL",
}


def route_after_tb(state: BlockState) -> str:
    """Route after testbench generation + simulation: PASS -> synthesize, FAIL -> diagnose."""
    return "synthesize" if state.get("sim_passed") else "diagnose"


route_after_tb.__edge_labels__ = {
    "synthesize": "SIM PASS",
    "diagnose": "SIM FAIL (RTL bug)",
}


def route_after_synth(state: BlockState) -> str:
    """Route after synthesis: SUCCESS -> block_done, FAIL -> diagnose.

    When the PPA gate is enabled (``CORESMITH_PPA_GATE=1``), a block that
    *compiled* but failed the deterministic PPA budget check
    (``ppa_ok is False``) is also routed to diagnose, so the synth_fixer can
    restructure it (e.g. a memory that should be an SRAM macro synthesized
    to flops). Default-off preserves the legacy "compiles == done" behavior.
    ``ppa_ok`` of None (not computed) never blocks.

    The post-synthesis GATE-LEVEL SIM gate (``CORESMITH_GATE_SIM``, default ON)
    routes ``gate_sim_ok is False`` to diagnose: the netlist that carries the
    PPA numbers does not reproduce the RTL that carries the DV pass, so the
    block is not done no matter how clean synthesis was. ``None`` (gate did not
    apply -- disabled, no netlist, no toolchain) never blocks, exactly like
    ``ppa_ok``.
    """
    if not state.get("synth_success"):
        return "diagnose"
    if state.get("gate_sim_ok") is False:
        return "diagnose"
    # rung2 defect 2: under SKIP_SYNTH the PDK-free synthesizability probes run
    # and GATE unconditionally (they need no PDK / no PPA-budget flag), so a
    # probe FAIL (ppa_ok False) must route to diagnose there too -- otherwise
    # SKIP_SYNTH would run the probes but ignore their verdict.
    import os as _os_ras

    from orchestrator.langgraph.ppa_check import ppa_gate_enabled
    _skip_synth = _os_ras.environ.get("CORESMITH_SKIP_SYNTH") == "1"
    if (ppa_gate_enabled() or _skip_synth) and state.get("ppa_ok") is False:
        return "diagnose"
    return "block_done"


route_after_synth.__edge_labels__ = {
    "block_done": "SUCCESS",
    "diagnose": "FAIL",
}


def route_decision(state: BlockState) -> str:
    """Route after decide: directly to generate_rtl, generate_testbench, etc."""
    action = state.get("debug_action", "retry_rtl")
    mapping = {
        "retry_rtl": "generate_rtl",
        "retry_tb": "generate_testbench",
        "retry_synth": "synthesize",
        "ask_human": "ask_human",
        "escalate": "block_done",
    }
    return mapping.get(action, "generate_rtl")


route_decision.__edge_labels__ = {
    "generate_rtl": "RETRY RTL",
    "generate_testbench": "RETRY TB",
    "synthesize": "RETRY SYNTH",
    "ask_human": "ASK HUMAN",
    "block_done": "ESCALATE",
}


def route_after_human(state: BlockState) -> str:
    """Route based on the human's resume action."""
    action = (state.get("human_response") or {}).get("action", "retry")
    mapping = {
        "retry": "generate_rtl",
        "fix_rtl": "generate_rtl",
        "fix_tb": "generate_testbench",
        "add_constraint": "generate_rtl",
        "skip": "block_done",
        "abort": "block_done",
    }
    return mapping.get(action, "generate_rtl")


route_after_human.__edge_labels__ = {
    "generate_rtl": "RETRY / FIX RTL",
    "generate_testbench": "FIX TB",
    "block_done": "SKIP / ABORT",
}


# ---------------------------------------------------------------------------
# Block subgraph builder
# ---------------------------------------------------------------------------

def build_block_subgraph(two_pass: bool | None = None):
    """Build the block lifecycle subgraph (uncompiled StateGraph).

    Contains the full lifecycle for a single block:
      init -> uarch spec -> review
        -> generate_rtl (with lint)
        -> generate_testbench (with sim + local TB fix)
        -> synthesize -> done

    Plus the diagnose/decide/retry failure loop, where decide routes
    directly back to generate_rtl (no intermediate increment node).

    ``two_pass`` selects the topology:
      - ``False`` (or flag off, the default) -> the historical single-pass
        graph: a HARD ``init_block -> generate_uarch_spec`` edge. Byte-identical
        node/edge set to before the two-pass restructure.
      - ``True`` (block-goldens on) -> a CONDITIONAL ``init_block`` edge
        (``route_after_init``) so pass 2 can skip re-spec and jump to
        ``generate_rtl``.
    ``None`` reads ``composition.block_goldens_enabled()``.

    Returns:
        Uncompiled ``StateGraph(BlockState)`` -- the caller compiles it
        (with or without a checkpointer) before adding it as a node.
    """
    if two_pass is None:
        from orchestrator.architecture import composition as _composition
        two_pass = _composition.block_goldens_enabled()

    graph = StateGraph(BlockState)

    # Nodes (10 -- lint, simulate, increment_attempt are folded in)
    graph.add_node("init_block", init_block_node)
    graph.add_node("generate_uarch_spec", generate_uarch_spec_node)
    graph.add_node("review_uarch_spec", review_uarch_spec_node)
    graph.add_node("generate_rtl", generate_rtl_node)
    graph.add_node("generate_testbench", generate_testbench_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("diagnose", diagnose_node)
    graph.add_node("decide", decide_node)
    graph.add_node("ask_human", ask_human_node)
    graph.add_node("block_done", block_done_node)

    # Happy path
    graph.add_edge(START, "init_block")
    if two_pass:
        # Conditional: pass 2 of the two-pass flow skips re-spec and goes
        # straight to generate_rtl; pass 1 goes to generate_uarch_spec.
        graph.add_conditional_edges("init_block", route_after_init)
    else:
        # Single-pass (flag off): historical hard edge, identical topology.
        graph.add_edge("init_block", "generate_uarch_spec")
    graph.add_edge("generate_uarch_spec", "review_uarch_spec")
    graph.add_conditional_edges("review_uarch_spec", route_after_uarch_review)
    graph.add_conditional_edges("generate_rtl", route_after_rtl)
    graph.add_conditional_edges("generate_testbench", route_after_tb)
    graph.add_conditional_edges("synthesize", route_after_synth)

    # Failure path
    graph.add_edge("diagnose", "decide")
    graph.add_conditional_edges("decide", route_decision)
    graph.add_conditional_edges("ask_human", route_after_human)

    # Terminal
    graph.add_edge("block_done", END)

    return graph


# ---------------------------------------------------------------------------
# Orchestrator nodes
# ---------------------------------------------------------------------------

def _current_phase_completed(state: OrchestratorState) -> list[dict]:
    """Completed blocks for the CURRENT pipeline_phase, deduped by name.

    The ``completed_blocks`` reducer is ``operator.add`` and NEVER resets, so
    after the two-pass flow it holds BOTH the pass-1 ``(name,"uarch")`` and the
    pass-2 ``(name,"rtl")`` results. Consumers (integration_review,
    pipeline_complete, integration_check) must look only at the current phase to
    avoid cross-pass contamination (R1).

    Filter rule: keep entries whose ``phase`` matches the current
    ``pipeline_phase``; entries with NO ``phase`` key (legacy / flag-off
    checkpoints) are always kept so single-pass behaviour is unchanged. Then
    dedup by name keeping the LAST entry (so a later pass / a retry overrides an
    earlier failure).
    """
    cur = state.get("pipeline_phase", "rtl")
    seen: dict[str, dict] = {}
    for b in state.get("completed_blocks", []):
        if not isinstance(b, dict):
            continue
        name = b.get("name")
        if not name:
            continue
        bphase = b.get("phase")
        if bphase is not None and bphase != cur:
            continue
        seen[name] = b
    return list(seen.values())


#: Defensive ceiling on consecutive partial-pin-map re-parks, mirroring
#: ``_INTEGRATION_REPARK_CAP``. In production each re-park is a real
#: ``interrupt()`` that SUSPENDS the graph, so this is never a CPU loop -- it
#: bounds an operator/outer-agent that keeps sending ``retry`` without fixing
#: the map, and stops a plain-return ``interrupt`` test double from spinning.
_PINMAP_REPARK_CAP = 20


async def _retire_pin_mapped_blocks(
    state: OrchestratorState, pr: str, block_queue: list,
) -> tuple[list, list[dict]]:
    """Drop pad-adapter blocks a declared PRD pin map already covers.

    Runs at the HEAD of the dispatch path, so a retired block is never
    microarchitected, generated, linted or gated -- the flow simply does not ask
    for a module the design does not contain. Returns ``(block_queue,
    retirement_records)``; the queue is returned unchanged when nothing is
    retired, and the records are what state / the final report carry.

    PARTIAL coverage never retires: it raises an interrupt so an operator
    resolves the contradiction (see ``pin_map_retire.plan_retirement``).
    """
    from orchestrator.architecture import pin_map_retire as _pmr

    if not _pmr.retirement_enabled():
        return block_queue, []

    already = {r.get("block") for r in (state.get("retired_blocks") or [])
               if isinstance(r, dict)}
    records: list[dict] = list(state.get("retired_blocks") or [])
    rounds = 0

    while True:
        try:
            plan = _pmr.plan_retirement(pr, block_queue)
        except Exception as _pe:  # noqa: BLE001 - never crash the dispatch path
            log(f"  [PIN-MAP] retirement check skipped ({_pe})", YELLOW)
            return block_queue, records

        if plan.retire:
            if plan.block in already:
                # Already retired on an earlier tier entry; the queue in state
                # no longer carries it, so this is the idempotent no-op path.
                return _pmr.apply_retirement(block_queue, plan), records
            log(f"\n{'='*60}", YELLOW)
            log(f"  [PIN-MAP] RETIRING block '{plan.block}' from the flow: "
                f"{plan.message}", YELLOW)
            log(f"  [PIN-MAP] it will NOT be microarchitected, generated or "
                f"gated, and it is DELIBERATELY absent from the assembled chip "
                f"(reason={_pmr.RETIRE_REASON})", YELLOW)
            log(f"{'='*60}\n", YELLOW)
            records.append(_pmr.record_retirement(pr, plan))
            already.add(plan.block)
            write_graph_event(pr, "Init Tier", "block_retired_by_pin_map", {
                "block": plan.block,
                "reason": _pmr.RETIRE_REASON,
                "covered_signals": plan.covered,
                "pin_map_signals": plan.pin_map_signals,
            })
            return _pmr.apply_retirement(block_queue, plan), records

        if not plan.park:
            if plan.reason and plan.block:
                log(f"  [PIN-MAP] '{plan.block}' NOT retired: {plan.reason}",
                    YELLOW)
            return block_queue, records

        # ---- partial coverage: a half-routed boundary, so PARK ----
        rounds += 1
        log(f"\n{'='*60}", RED)
        log(f"  [PIN-MAP] PARTIAL COVERAGE for '{plan.block}' -- refusing to "
            f"retire it AND refusing to pretend the boundary is whole", RED)
        log(f"  [PIN-MAP] {plan.message}", RED)
        log(f"{'='*60}\n", RED)
        write_graph_event(pr, "Init Tier", "pin_map_partial_coverage", {
            "block": plan.block,
            "covered_signals": plan.covered,
            "uncovered_signals": plan.uncovered,
            "round": rounds,
        })
        if rounds > _PINMAP_REPARK_CAP:
            raise RuntimeError(
                f"pin_map partially covers '{plan.block}' and the contradiction "
                f"was not resolved after {_PINMAP_REPARK_CAP} re-parks "
                f"(uncovered: {plan.uncovered}). Fix prd.pin_map or remove it.")
        response = interrupt({
            "type": "pin_map_partial_coverage",
            "block": plan.block,
            "reason": plan.reason,
            "message": plan.message,
            "contract_signals": plan.contract_signals,
            "pin_map_signals": plan.pin_map_signals,
            "covered_signals": plan.covered,
            "uncovered_signals": plan.uncovered,
            "supported_actions": ["retry", "override", "keep_block"],
            "outer_agent_guidance": (
                "The PRD's structured pin_map routes SOME of the signals this "
                "pad-adapter block is contracted to translate, and not the "
                "rest. The chip top emits routing for the mapped bits while "
                "the block would route the others -- two drivers on one pad "
                "bus, and no gate downstream can tell which was intended. "
                "Resolve it: (a) extend prd.pin_map in "
                ".coresmith/prd_spec.json to cover the uncovered signals and "
                "resume `retry`; or (b) resume `keep_block` to generate the "
                "adapter as before and leave the pin map to the assembler; or "
                "(c) resume `override` to retire the block anyway -- ONLY when "
                "you have verified the uncovered signals are genuinely routed "
                "elsewhere."
            ),
            "reference_files": {
                "prd": ".coresmith/prd_spec.json",
                "interface_contracts": ".coresmith/interface_contracts.json",
            },
        }) or {}
        action = (response.get("action") if isinstance(response, dict)
                  else "retry") or "retry"
        write_graph_event(pr, "Init Tier", "pin_map_partial_coverage_resume", {
            "block": plan.block, "action": action,
        })
        if action == "keep_block":
            log(f"  [PIN-MAP] operator KEPT '{plan.block}' in the flow despite "
                f"partial pin-map coverage -- generating it as before", YELLOW)
            return block_queue, records
        if action == "override":
            log(f"  [PIN-MAP] operator OVERRIDE: retiring '{plan.block}' with "
                f"{len(plan.uncovered)} signal(s) NOT covered by the pin map "
                f"({plan.uncovered})", YELLOW)
            plan.retire = True
            plan.message = (plan.message
                            + " -- RETIRED ANYWAY by operator override")
            records.append(_pmr.record_retirement(pr, plan))
            return _pmr.apply_retirement(block_queue, plan), records
        # retry -> re-plan against the (hopefully amended) PRD and loop


async def init_tier_node(state: OrchestratorState) -> dict:
    """Compute the tier list (once) and log the current tier."""
    pr = state.get("project_root", str(PROJECT_ROOT))

    # A declared pin map REPLACES the pad-adapter block, so the flow must not
    # ask for it. Done here -- the head of the dispatch path, before tier_list
    # and before any Send() -- so the block is skipped ahead of µarch/RTL rather
    # than generated, refused by the conformance gate and dropped at assembly.
    # Idempotent, so every tier re-entry and checkpoint resume agrees.
    block_queue = state["block_queue"]
    _retired_before = len(block_queue)
    block_queue, _retired_records = await _retire_pin_mapped_blocks(
        state, pr, block_queue)
    _queue_reduced = len(block_queue) != _retired_before

    tier_list = state.get("tier_list") or sorted(
        set(b.get("tier", 1) for b in block_queue)
    )
    current_idx = state.get("current_tier_index", 0)

    tier = tier_list[current_idx]
    tier_blocks = [b for b in block_queue if b.get("tier", 1) == tier]

    # Section 7a: stamp the engine git SHA at run start + WARN in the daemon log
    # if it changes mid-run (a hot-swap that flipped behavior under the run).
    _stamp_engine_sha(pr)

    # Engine Fix #5: thread the µarch gate's divergence diagnosis to disk for the
    # blocks it implicated, so a gate-triggered re-spec is INFORMED (otherwise
    # the Fix #4 bounded re-spec loop just redraws identical blocks). Disk-first:
    # generate_uarch_spec_node reads .coresmith/blocks/<b>/gate_feedback.txt.
    # Written per tier as each tier re-fans-out; CLEARED for unaffected blocks so
    # stale feedback never leaks into a later draw or a clean first pass (mir
    # absent/passed -> no feedback anywhere).
    mir = state.get("model_integration_result") or {}
    gate_failed = bool(mir) and not mir.get("passed", True)
    # Engine Fix #5b: when the gate precisely localized the failure (affected_
    # blocks/edge), feed only those blocks; otherwise BROADCAST to all tier
    # blocks (the first_divergence_block stub can't be trusted as a sole target).
    precise = _gate_localization_precise(mir)
    if precise:
        targets = _gate_affected_blocks(mir)
    else:
        targets = {b["name"] for b in tier_blocks}
    for b in tier_blocks:
        fb = ""
        if gate_failed and b["name"] in targets:
            fb = _gate_feedback_for_block(mir, b["name"], localized=precise)
        fbp = Path(pr) / ".coresmith" / "blocks" / b["name"] / "gate_feedback.txt"
        try:
            fbp.parent.mkdir(parents=True, exist_ok=True)
            if fb:
                fbp.write_text(fb, encoding="utf-8")
            elif fbp.exists():
                fbp.unlink()
        except OSError:
            pass

    write_graph_event(pr, "Init Tier", "graph_node_enter", {
        "tier": tier, "tier_index": current_idx,
        "block_count": len(tier_blocks),
        "block_names": [b["name"] for b in tier_blocks],
    })

    log(f"\n{'='*60}", CYAN)
    log(f"  Tier {tier}: {len(tier_blocks)} blocks "
        f"({', '.join(b['name'] for b in tier_blocks)}) | "
        f"Tier {current_idx + 1}/{len(tier_list)}", CYAN)
    log(f"{'='*60}", CYAN)

    write_graph_event(pr, "Init Tier", "graph_node_exit", {
        "tier": tier,
    })

    out = {"tier_list": tier_list}
    # Publish the reduced queue + the retirement record so EVERY downstream
    # consumer agrees the block is deliberately absent rather than missing:
    # pipeline_complete / integration_check size `expected` off block_queue,
    # discover_block_rtl + missing_from work off the same set, and the daemon's
    # total_blocks / remaining_count stop waiting on it.
    if _queue_reduced:
        out["block_queue"] = block_queue
    if _retired_records:
        out["retired_blocks"] = _retired_records
    # Seed the two-pass phase on the FIRST entry only. fan_out_tier defaults an
    # unset pipeline_phase to "rtl", so without this a block-goldens run would
    # send every block straight down the single-pass RTL path and pass 1
    # (spec+model) + the µarch integration gate would never run. begin_rtl_pass
    # is the sole writer of "rtl"; once set (pass-2 re-entry) we must not clobber
    # it back to "uarch". Flag off: add nothing -> byte-identical single-pass.
    if not state.get("pipeline_phase"):
        from orchestrator.architecture import composition as _composition
        if _composition.block_goldens_enabled():
            out["pipeline_phase"] = "uarch"
            # Engine Fix #6: stimulus<->contract consistency guard. On the FIRST
            # entry of a block-goldens run, before any block is built, check that
            # the gate stimulus is something the declared design can actually
            # accept (the oracle can process it; every stimulus config field has
            # a boundary input port). Catches the codec-class failure (arch
            # hardcoded 640x360 vs a 16x16 in-contract gate frame) in seconds
            # instead of after a multi-hour arch+pass-1 run. Never let the guard
            # itself break a run; default is warn+report, strict raises.
            try:
                from orchestrator.architecture import (
                    stimulus_contract_guard as _scg,
                )
                _viol = _scg.run_stimulus_contract_guard(pr)
            except Exception as _exc:  # noqa: BLE001
                _viol = []
                log(f"  [STIMULUS-GUARD] guard error (ignored): {_exc}", YELLOW)
            if _viol:
                import json as _json
                _errs = [v for v in _viol if v.get("severity") == "error"]
                _col = RED if _errs else YELLOW
                try:
                    (Path(pr) / ".coresmith" / _scg.REPORT_FILENAME).write_text(
                        _json.dumps(_viol, indent=2), encoding="utf-8")
                except OSError:
                    pass
                log(f"\n{'='*60}", _col)
                log(f"  STIMULUS<->CONTRACT GUARD: {len(_viol)} finding(s), "
                    f"{len(_errs)} error", _col)
                log(_scg.format_violations(_viol), _col)
                log("  (the gate stimulus may be inconsistent with the declared "
                    "design contract -- fix before this run dead-ends at the "
                    "µarch gate)", _col)
                log(f"{'='*60}\n", _col)
                if _scg.guard_strict():
                    raise RuntimeError(
                        "stimulus<->contract guard failed (strict mode):\n"
                        + _scg.format_violations(_viol))

    return out


def _gate_localization_precise(mir: dict) -> bool:
    """True when the gate genuinely localized the failure to specific block(s).

    The gate's ``first_divergence_block`` is a best-effort STUB
    (model_integration._first_divergence_block returns the first diagram block
    in declared order -- it cannot localize a wrong-bytes divergence). So a
    result carrying ONLY first_divergence_block is NOT trustworthy localization;
    only an explicit ``affected_blocks`` list or an ``affected_edge`` is. When
    localization is imprecise the re-spec feedback is broadcast to ALL blocks
    (Engine Fix #5b) rather than misdirected to the (wrong) first block.
    """
    return bool(mir.get("affected_blocks")) or bool(mir.get("affected_edge"))


def _gate_affected_blocks(mir: dict) -> set[str]:
    """Block names the µarch gate PRECISELY implicated (affected_blocks / edge).

    Excludes the unreliable ``first_divergence_block`` stub on purpose -- callers
    use :func:`_gate_localization_precise` to decide between this targeted set
    and a broadcast to all tier blocks. Engine Fix #5/#5b.
    """
    names: set[str] = set()
    for b in mir.get("affected_blocks") or []:
        if b:
            names.add(b)
    edge = mir.get("affected_edge") or {}
    for k in ("from", "to"):
        if edge.get(k):
            names.add(edge[k])
    return names


def _gate_feedback_for_block(mir: dict, block_name: str,
                             localized: bool = True) -> str:
    """Build the re-spec feedback string for one block from the gate result.

    ``localized`` False means the gate could not pin the divergence to a specific
    block, so this feedback is being broadcast to every block -- the wording asks
    each block to self-check its math against the reference at the divergence
    point. Engine Fix #5/#5b.
    """
    parts = [
        "The composed-chip integration gate FAILED -- the wired block models do "
        "not compose into a chip that matches the reference.",
        f"gap_class={mir.get('gap_class', 'block_math')}.",
    ]
    edge = mir.get("affected_edge") or {}
    if edge.get("from") or edge.get("to"):
        parts.append(f"affected interface edge: {edge.get('from','?')} -> "
                     f"{edge.get('to','?')}.")
    # Carry every suggested_fix the gate emitted (most actionable signal).
    fixes = []
    for v in (mir.get("violations") or []):
        if isinstance(v, dict) and v.get("suggested_fix"):
            fixes.append(str(v["suggested_fix"]))
    if mir.get("suggested_fix"):
        fixes.append(str(mir["suggested_fix"]))
    for f in dict.fromkeys(fixes):  # dedupe, preserve order
        parts.append(f"suggested fix: {f}")
    # Compact expected-vs-observed so the re-spec sees the behavioural gap (esp.
    # the FIRST divergence position -- the most useful clue when unlocalized).
    gap_class = mir.get("gap_class", "block_math")
    div_off = -1
    if "expected" in mir or "observed" in mir:
        exp = repr(mir.get("expected"))[:200]
        obs = repr(mir.get("observed"))[:200]
        parts.append(f"reference expected {exp}; composed chip observed {obs}.")
        try:
            from orchestrator.architecture.model_integration import (
                _is_byteseq as _ibs,
            )
            from orchestrator.architecture.model_integration import (
                first_divergence_offset as _fdo,
            )
            e, o = mir.get("expected"), mir.get("observed")
            if _ibs(e) and _ibs(o):
                div_off = _fdo(e, o)
                if div_off >= 0:
                    parts.append(
                        f"FIRST DIVERGENCE at byte offset {div_off} "
                        f"(bytes 0..{div_off-1} are byte-EXACT -> the framing/"
                        f"earlier stages composed correctly; the producer of byte "
                        f"{div_off} is the culprit).")
        except Exception:  # noqa: BLE001
            pass
    if localized:
        parts.append("Revise THIS block's microarchitecture so the composition "
                     "matches the reference; do NOT hardcode geometry/sizes that "
                     "the stimulus contract supplies at runtime.")
    elif gap_class == "block_math":
        # Targeted-restart instruction (the chip-lead localizes + restarts ONLY
        # the producing block, instead of broadcasting a full re-fan). The
        # framing composed correctly -> exactly ONE block emits wrong/short
        # content; re-fanning all blocks wastes the run (>55% of tokens once).
        # Keep the "could NOT be localized" phrasing: this text is still
        # broadcast to every block (each self-audits), while the CHIP-LEAD does
        # the precise single-block restart.
        _bisect = (
            f"A first-divergence byte offset ({div_off}) WAS found -- use it: "
            f"trace which block produces byte {div_off} through the serialization "
            f"chain and restart THAT block only. "
            if div_off >= 0 else
            "The first-divergence offset could NOT be bisected automatically -- "
            "before touching any block, do the CHEAP localization: snoop each "
            "block's output bus against the reference's per-stage output to find "
            "the single producer that diverges. If that localization ALSO fails, "
            "ESCALATE to a human interrupt (ask_human) -- do NOT broadcast a "
            "re-spec of all blocks (that wastes >55% of the run's tokens on "
            "blocks that compose correctly). ")
        parts.append(
            "The divergence could NOT be localized to a specific block "
            "automatically -- but the framing/earlier stages compose byte-exact, "
            "so this is a SINGLE-BLOCK content divergence: exactly ONE downstream "
            "block emits wrong/truncated content. CHIP-LEAD: " + _bisect +
            "restart_block(from_node='generate_uarch_spec') for THAT block ONLY "
            "-- do NOT re-fan all blocks. Each block: audit YOUR math/encoding "
            "against the reference at the divergence point and fix it if it "
            "diverges; do NOT hardcode geometry/sizes supplied at runtime.")
    else:
        parts.append("The divergence could NOT be localized to a specific block. "
                     "Audit whether YOUR block's math/encoding exactly matches the "
                     "reference at the first divergence point (compare against the "
                     "reference's per-stage behaviour); fix it if it diverges, and "
                     "do NOT hardcode geometry/sizes supplied at runtime.")
    return " ".join(parts)


def fan_out_tier(state: OrchestratorState) -> list[Send]:
    """Fan out all blocks in the current tier for parallel execution.

    Returns a list of ``Send("process_block", block_state)`` -- one per
    block.  LangGraph runs all branches concurrently and collects results
    via the ``completed_blocks`` reducer before continuing.
    """
    block_queue = state["block_queue"]
    tier_list = state["tier_list"]
    current_idx = state.get("current_tier_index", 0)
    tier = tier_list[current_idx]

    tier_blocks = [b for b in block_queue if b.get("tier", 1) == tier]

    # Two-pass phase: default "rtl" so flag-off (single-pass) behaviour is
    # unchanged. begin_rtl_pass is the sole writer of phase "rtl"; pass 1 sets
    # "uarch" once block-goldens is on.
    pipeline_phase = state.get("pipeline_phase", "rtl")
    uarch_pass_done = bool(state.get("uarch_pass_done", False))

    sends = []
    for block in tier_blocks:
        sends.append(Send("process_block", {
            "project_root": state["project_root"],
            "target_clock_mhz": state["target_clock_mhz"],
            "max_attempts": state["max_attempts"],
            "pipeline_run_start": state.get("pipeline_run_start", 0.0),
            "current_block": block,
            "attempt": 1,
            "phase": "init",
            "pipeline_phase": pipeline_phase,
            "uarch_pass_done": uarch_pass_done,
            "constraints": [],
            "attempt_history": [],
            "previous_error": "",
            "uarch_spec": None,
            "uarch_approved": False,
            "uarch_feedback": "",
            "rtl_result": None,
            "lint_result": None,
            "tb_result": None,
            "sim_result": None,
            "synth_result": None,
            "debug_result": None,
            "human_response": None,
            "completed_blocks": [],
            "step_log_paths": {},
        }))

    return sends


fan_out_tier.__edge_labels__ = {
    "process_block": "FAN OUT",
}


async def integration_review_node(state: OrchestratorState) -> dict:
    """Run the Integration Agent to check cross-block interface coherence.

    After all blocks in a tier generate their uArch specs and complete
    RTL/sim/synth, the Integration Agent reads all Section 9 stubs,
    cross-checks against the block diagram, and edits specs on disk to
    fix mismatches.  Then fires ONE chip-level interrupt for user
    approval of the full uArch.
    """
    pr = state.get("project_root", str(PROJECT_ROOT))
    state.get("completed_blocks", [])
    block_queue = state.get("block_queue", [])

    tier_list = state.get("tier_list", [])
    current_idx = state.get("current_tier_index", 0)
    tier = tier_list[current_idx] if current_idx < len(tier_list) else 1
    tier_blocks = [b for b in block_queue if b.get("tier", 1) == tier]
    block_names = [b["name"] for b in tier_blocks]

    write_graph_event(pr, "Integration Review", "graph_node_enter", {
        "tier": tier, "block_names": block_names,
    })

    if not block_names:
        write_graph_event(pr, "Integration Review", "graph_node_exit", {
            "action": "skip (no blocks)",
        })
        return {}

    # Under block-goldens the per-tier LLM integration review is REDUNDANT and
    # actively harmful, so skip it in BOTH passes (defer all cross-block checking
    # to the real gates):
    #   - pass 1 ("uarch"): the uarch_integration_gate after all tiers validates
    #     cross-block coherence on the composed Amaranth chip model (byte-exact vs
    #     the reference) before any RTL.
    #   - pass 2 ("rtl"): integration_check (chip_top assembly + lint/wiring),
    #     integration_dv (RTL == composed chip model) and validation_dv (RTL ==
    #     golden) are the authoritative RTL-level cross-block gates.
    # Beyond redundancy, the reviewer EDITS uArch specs on every run; in pass 2
    # that trips the "stale RTL after spec edit" guard below -> approve routes to
    # advance_tier but the spec-edit/re-DV churn re-parks here -> an integration-
    # review REVISE-LOOP that never reaches integration_dv. Skipping it removes
    # the loop without weakening correctness (the gates above still run). Also
    # avoids LangGraph re-running the reviewer LLM on every resume (slow / can
    # hang). Flag off -> unchanged. CORESMITH_STRICT_INTEGRATION_REVIEW=1 forces
    # the old per-tier review (and its auto-revise) back on for both passes.
    import os as _os

    from orchestrator.architecture import composition as _composition
    if (
        _composition.block_goldens_enabled()
        and _os.environ.get("CORESMITH_STRICT_INTEGRATION_REVIEW") != "1"
    ):
        _phase = state.get("pipeline_phase", "rtl")
        log("  [INTEGRATION REVIEW] block-goldens: deferring per-tier cross-block "
            f"review (phase={_phase}) to the uarch gate + integration_dv/"
            "validation_dv", GREEN)
        # Even when the per-tier LLM review is skipped, still surface any
        # mem-price DEFERs (over-budget storage) into the event stream so the
        # deferred excess stays visible (Deliverable 3).
        try:
            from orchestrator.langgraph import mem_price as _mprice
            _deferred = _mprice.deferred_over_budget_blocks(pr, block_names)
        except Exception:  # noqa: BLE001
            _deferred = []
        if _deferred:
            log("  [INTEGRATION REVIEW] mem-price DEFERRED (over-budget accepted): "
                + "; ".join(f"{d['block']} {d.get('total_area_mm2')} mm^2"
                            for d in _deferred), RED)
        write_graph_event(pr, "Integration Review", "graph_node_exit", {
            "action": f"skip (block-goldens phase={_phase}; deferred to gate + "
                      "integration_dv/validation_dv)",
            "tier": tier,
            "mem_price_deferred": _deferred,
        })
        return {"integration_review_action": "approve",
                "integration_review_failed": False}

    try:
        from orchestrator.langchain.agents.coresmith_llm import DEFAULT_MODEL
        from orchestrator.langchain.agents.integration_review_agent import (
            IntegrationReviewAgent,
        )
        agent = IntegrationReviewAgent(model=DEFAULT_MODEL, temperature=0.1)
        result = await agent.review(
            block_names=block_names,
            project_root=pr,
        )
        review_summary = result.get("summary", "No issues found.")
        issues_found = result.get("issues_found", 0)
        issues_fixed = result.get("issues_fixed", 0)
    except Exception as exc:
        review_summary = f"Integration review failed: {exc}"
        issues_found = 1
        issues_fixed = 0
        review_failed = True
    else:
        review_failed = False

    completed_by_name = {
        b.get("name"): b
        for b in _current_phase_completed(state)
        if isinstance(b, dict) and b.get("name")
    }
    failed_tier_blocks = [
        name
        for name in block_names
        if name in completed_by_name and not completed_by_name[name].get("success")
    ]
    if failed_tier_blocks:
        failure_note = (
            "Blocking tier failure: uArch integration review cannot be approved "
            "because these current-tier blocks have not passed their lifecycle: "
            f"{', '.join(failed_tier_blocks)}."
        )
        review_summary = f"{failure_note}\n\n{review_summary}"
        issues_found = int(issues_found or 0) + len(failed_tier_blocks)
        review_failed = True
    if issues_fixed:
        stale_artifact_note = (
            "Blocking uArch edits: integration review modified current-tier "
            "uArch specs after RTL/testbench artifacts were generated. The "
            "affected blocks must be regenerated from uArch before this tier "
            "can be approved; otherwise stale RTL can falsely pass against the "
            "old contract."
        )
        review_summary = f"{stale_artifact_note}\n\n{review_summary}"
        review_failed = True

    log(f"  [INTEGRATION REVIEW] {review_summary[:200]}", GREEN if issues_found == 0 else YELLOW)

    spec_paths = {
        name: str(Path(pr) / "arch" / "uarch_specs" / f"{name}.md")
        for name in block_names
    }

    # Surface any mem-price DEFERs (over-budget specs the bounded revise loop
    # accepted) so the chip-level review SEES the deferred excess rather than
    # silently shipping it (Deliverable 3).
    try:
        from orchestrator.langgraph import mem_price as _mprice
        mem_price_deferred = _mprice.deferred_over_budget_blocks(pr, block_names)
    except Exception:  # noqa: BLE001 - never block the review
        mem_price_deferred = []
    if mem_price_deferred:
        _parts = []
        for d in mem_price_deferred:
            _seg = f"{d['block']} {d.get('total_area_mm2')} mm^2"
            if d.get("over_budget_x"):
                _seg += f" ({d['over_budget_x']}x budget)"
            _parts.append(_seg)
        _defnote = "; ".join(_parts)
        review_summary = (
            f"MEM-PRICE DEFERRED (over-budget storage accepted after the bounded "
            f"revise loop): {_defnote}. These blocks bust their area budget and "
            f"were deferred here -- reduce stored bits or raise the budget "
            f"deliberately.\n\n{review_summary}"
        )

    payload = {
        "type": "uarch_integration_review",
        "tier": tier,
        "block_names": block_names,
        "spec_paths": spec_paths,
        "review_summary": review_summary,
        "issues_found": issues_found,
        "issues_fixed": issues_fixed,
        "mem_price_deferred": mem_price_deferred,
        "review_failed": review_failed,
        "supported_actions": ["approve", "revise", "abort"],
        "outer_agent_guidance": (
            "The Integration Agent has reviewed all uArch specs for "
            "cross-block interface coherence. Present this as a CHIP-LEVEL "
            "review to the user. The user approves or rejects ALL specs at "
            "once. If the Integration Agent fixed mismatches, summarize "
            "what was changed. If the user wants revisions, use "
            "restart_block(from_node='generate_uarch_spec') for affected blocks."
        ),
    }

    response = interrupt(payload)
    action = response.get("action", "abort")
    if action == "approve" and failed_tier_blocks:
        log(
            "  [INTEGRATION REVIEW] Approval rejected because current-tier "
            "blocks failed; treating as revise",
            YELLOW,
        )
        action = "revise"
    if action == "approve" and issues_fixed:
        # NOTE: The integration_review agent edits specs on every run, even
        # cosmetically, so this auto-revise creates an infinite loop:
        # revise -> restart_block -> integration_review edits again -> revise...
        # When the outer agent explicitly approves, trust that decision;
        # the integration_check
        # node at RTL level will catch any real cross-block lint/wiring
        # mismatch and surface it as a normal failure. Setting
        # CORESMITH_STRICT_INTEGRATION_REVIEW=1 restores the old auto-revise.
        import os as _os
        if _os.environ.get("CORESMITH_STRICT_INTEGRATION_REVIEW") == "1":
            log(
                "  [INTEGRATION REVIEW] Approval rejected because uArch specs "
                "were edited after block artifacts were generated; treating as revise "
                "(CORESMITH_STRICT_INTEGRATION_REVIEW=1)",
                YELLOW,
            )
            action = "revise"
        else:
            log(
                "  [INTEGRATION REVIEW] Spec edits were made; honoring explicit "
                "approve (integration_check at RTL level will catch real mismatches). "
                "Export CORESMITH_STRICT_INTEGRATION_REVIEW=1 to force revise.",
                YELLOW,
            )
    if action == "revise" and issues_found == 0 and not review_failed:
        log(
            "  [INTEGRATION REVIEW] Clean review returned revise; "
            "treating as approve",
            YELLOW,
        )
        action = "approve"

    write_graph_event(pr, "Integration Review", "graph_node_exit", {
        "action": action, "issues_found": issues_found,
        "review_failed": review_failed,
    })

    if action == "abort":
        log("  [INTEGRATION REVIEW] Aborted by user/agent", RED)
    elif action == "revise":
        log("  [INTEGRATION REVIEW] Revision requested — "
            "use restart_block to re-generate affected specs", YELLOW)

    return {
        "integration_review_action": action,
        "integration_review_failed": review_failed,
    }


async def advance_tier_node(state: OrchestratorState) -> dict:
    """Advance the tier index after all blocks in the current tier complete."""
    new_idx = state.get("current_tier_index", 0) + 1

    completed = state.get("completed_blocks", [])
    passed = sum(1 for b in completed if b.get("success"))
    total = len(completed)

    pr = state.get("project_root", str(PROJECT_ROOT))
    write_graph_event(pr, "Advance Tier", "graph_node_exit", {
        "new_tier_index": new_idx, "completed_so_far": total,
        "passed_so_far": passed,
    })

    return {"current_tier_index": new_idx}


# ---------------------------------------------------------------------------
# Orchestrator routing functions
# ---------------------------------------------------------------------------

def route_after_integration_review(state: OrchestratorState) -> str:
    """Route based on the user's integration review decision.

    approve → advance_tier (continue normally)
    abort   → END (terminate the pipeline)
    revise  → init_tier (rerun the current tier from the revised uArch specs)
    """
    action = state.get("integration_review_action", "approve")
    if action == "abort":
        return END
    if action == "revise":
        return "init_tier"
    return "advance_tier"


route_after_integration_review.__edge_labels__ = {
    "advance_tier": "APPROVED",
    "init_tier": "REVISE",
    END: "ABORT",
}


def route_next_tier(state: OrchestratorState) -> str:
    """Route after tier advancement.

    More tiers -> ``init_tier``. Tiers exhausted:
      - two-pass phase ``"uarch"`` (pass 1 done) -> ``uarch_integration_gate``
        (validate the decomposition before any RTL).
      - phase ``"rtl"`` / flag-off default -> ``pipeline_complete`` (unchanged).
    ``uarch_integration_gate`` is only ever returned when phase is ``"uarch"``,
    which only happens with block-goldens on (where that node exists).
    """
    completed = state.get("completed_blocks", [])
    if any(b.get("aborted") for b in completed):
        return "pipeline_complete"

    tier_list = state.get("tier_list", [])
    current_idx = state.get("current_tier_index", 0)
    if current_idx < len(tier_list):
        return "init_tier"
    if state.get("pipeline_phase", "rtl") == "uarch":
        return "uarch_integration_gate"
    return "pipeline_complete"


route_next_tier.__edge_labels__ = {
    "init_tier": "NEXT TIER",
    "uarch_integration_gate": "µARCH GATE",
    "pipeline_complete": "ALL DONE",
}


# ---------------------------------------------------------------------------
# Node: pipeline_complete  (orchestrator terminal)
# ---------------------------------------------------------------------------

async def pipeline_complete_node(state: OrchestratorState) -> dict:
    """Mark the pipeline as done, interrupting if any blocks failed.

    All blocks must succeed (sim + synth) before the pipeline can
    proceed to integration check and backend.  If any block failed,
    this node fires a ``pipeline_incomplete`` interrupt so the outer
    agent can diagnose each failure and restart blocks with fixes.
    """
    block_queue = state.get("block_queue", [])

    # Deduplicate completed_blocks by name (keep last entry so that
    # mark_block_passed overrides a previous failure entry), filtered to the
    # CURRENT pipeline_phase so a two-pass run's pass-1 results never inflate the
    # pass-2 passed/expected counts (R1/R3). Flag-off: phase-less entries are
    # kept, so this is identical to the prior name-only dedup.
    completed = _current_phase_completed(state)

    expected = len(block_queue) if block_queue else len(completed)
    passed = sum(1 for b in completed if b.get("success"))
    total = len(completed)

    log(f"\n{'#'*60}", CYAN)
    log(f"  FRONTEND PER-BLOCK COMPLETE: {passed}/{expected} blocks passed "
        f"(integration + DV pending -- NOT pipeline_done)", CYAN)
    log(f"{'#'*60}\n", CYAN)

    pr = state.get("project_root", str(PROJECT_ROOT))
    write_graph_event(pr, "Pipeline Complete", "graph_node_exit", {
        "passed": passed, "expected": expected, "total": total,
    })

    # --- Gate: ALL blocks must succeed before proceeding ---
    if passed < expected:
        failed_blocks = []
        for b in completed:
            if not b.get("success"):
                failed_blocks.append({
                    "name": b.get("name", "unknown"),
                    "error": b.get("error", ""),
                    "skipped": b.get("skipped", False),
                    "aborted": b.get("aborted", False),
                    "escalated": b.get("escalated", False),
                    "sim_passed": b.get("sim_passed", False),
                    "synth_success": b.get("synth_success", False),
                    "attempts": b.get("attempts", 0),
                    "step_log_paths": b.get("step_log_paths", {}),
                })

        # Also identify blocks that were expected but never completed
        completed_names = {b.get("name") for b in completed}
        missing_blocks = [
            bq.get("name", "unknown")
            for bq in block_queue
            if bq.get("name") not in completed_names
        ]

        failed_names = [fb["name"] for fb in failed_blocks]

        log(f"  [PIPELINE] {expected - passed} block(s) did not pass: "
            f"{failed_names + missing_blocks}", RED)

        payload = {
            "type": "pipeline_incomplete",
            "passed": passed,
            "expected": expected,
            "failed_blocks": failed_blocks,
            "missing_blocks": missing_blocks,
            "message": (
                f"All blocks must succeed before backend can begin. "
                f"{passed}/{expected} blocks passed. "
                f"Failed: {failed_names}. "
                f"Missing: {missing_blocks}. "
                f"Diagnose each failure (read sim logs, compare RTL against "
                f"testbench expectations, check for timing mismatches) and "
                f"restart blocks with fixes."
            ),
            "supported_actions": ["retry", "abort"],
            "outer_agent_guidance": (
                "As the outer-loop diagnostic agent, you MUST:\n"
                "1. Read step_log_paths for each failed block\n"
                "2. Read the RTL and testbench for each failed block\n"
                "3. Diagnose the root cause of each failure\n"
                "4. Restart each failed block with corrective constraints or RTL fixes\n"
                "5. Do NOT proceed to backend until all blocks pass\n"
                "6. Do NOT use run_step() to bypass this gate -- it does not "
                "register results in the pipeline checkpoint"
            ),
        }

        write_graph_event(pr, "Pipeline Incomplete", "pipeline_gate", {
            "passed": passed, "expected": expected,
            "failed_blocks": failed_names,
            "missing_blocks": missing_blocks,
        })

        resume = interrupt(payload)

        action = resume.get("action") if isinstance(resume, dict) else "abort"
        rv_attempts = int(state.get("revalidate_attempts", 0) or 0)

        # Recoverable incomplete-gate (completion bookkeeping): on `retry`, the
        # outer controller has fixed the failed block(s)' RTL on disk. Re-run the
        # rtl-phase tiers: each previously-passing block reuses its RTL via the
        # skip-regen fast path, and each failed block re-validates against the
        # edited (fresh-mtime) file through lint/sim/synth. block_done appends
        # fresh results; _current_phase_completed's last-wins dedup lets the new
        # PASS override the stale FAIL, so the recount can reach expected and
        # advance to integration — no `--force` restart that would discard the
        # byte-exact composition + already-passing blocks. Bounded so a block
        # that truly cannot pass aborts instead of looping forever.
        if action == "retry" and _revalidate_enabled() and rv_attempts < _revalidate_max():
            log(
                f"  [PIPELINE] Incomplete gate ({passed}/{expected}). "
                f"Re-validating failed/missing blocks against on-disk RTL "
                f"(pass {rv_attempts + 1}/{_revalidate_max()}); passing blocks "
                f"reuse their RTL. Re-running rtl-phase tiers, then recounting.",
                YELLOW,
            )
            # engine-v31 step 3: a failed block whose diagnosis carries a
            # high-confidence structured uarch_patch is re-spec'd + regenerated
            # (bounded to 1) instead of just re-validating its stale RTL, which
            # would re-fail and escalate. The spec edit + gate_feedback below
            # make the re-validate pass regenerate from the revised µarch.
            _patched = _route_uarch_patch_on_retry(
                pr, failed_names + missing_blocks)
            if _patched:
                log(f"  [PIPELINE] Auto-applied uarch_patch on retry for: "
                    f"{_patched} (re-spec + regen this pass)", GREEN)
                write_graph_event(pr, "Pipeline Incomplete", "uarch_patch_on_retry", {
                    "blocks": _patched, "pass": rv_attempts + 1,
                })
            return {
                "pipeline_done": False,
                "pipeline_aborted": False,
                "revalidate_pending": True,
                "revalidate_attempts": rv_attempts + 1,
                "current_tier_index": 0,
            }

        if action == "retry":
            why = (
                "re-validation disabled (CORESMITH_REVALIDATE_INCOMPLETE=0)"
                if not _revalidate_enabled()
                else f"re-validation cap {_revalidate_max()} reached"
            )
            log(
                f"  [PIPELINE] Retry at incomplete gate with {passed}/{expected} "
                f"blocks passed, but {why}. Stopping graph so the outer "
                f"controller can relaunch; not proceeding to integration.",
                YELLOW,
            )
        else:
            log(
                f"  [PIPELINE] Aborted at gate with {passed}/{expected} "
                f"blocks passed; not proceeding to integration.",
                RED,
            )
        return {
            "pipeline_done": False,
            "pipeline_aborted": True,
            "revalidate_pending": False,
        }

    # Per-block frontend done -- NOT pipeline_done. The deliverable is a
    # verified chip_top (integration_dv + validation_dv + chip-top synth), set
    # at the end of validation_dv. Setting pipeline_done here is the leak that
    # let a parked-at-integration run report as "done". (fix #5)
    return {
        "frontend_complete": True,
        "pipeline_done": False,
        "revalidate_pending": False,
    }


# ---------------------------------------------------------------------------
# Node: integration_check  (orchestrator -- verifies cross-block wiring)
# ---------------------------------------------------------------------------

def _deterministic_integration_check_enabled() -> bool:
    """A-Fix 3(a): run the deterministic port/width compatibility checker
    alongside the Integration Lead agent. Default ON; set
    ``CORESMITH_DETERMINISTIC_INTEGRATION_CHECK=0`` to disable (the operator
    ``accept`` interrupt remains the override for any false positive)."""
    return (
        (os.environ.get("CORESMITH_DETERMINISTIC_INTEGRATION_CHECK", "1") or "1")
        != "0"
    )


def _block_rtl_complete_gate_enabled() -> bool:
    """Refuse to assemble a chip that is MISSING a block. Default ON; set
    ``CORESMITH_BLOCK_RTL_COMPLETE_GATE=0`` to restore the old
    drop-the-block-and-continue behavior (the ``override`` interrupt action
    remains the per-run escape for a block that genuinely does not belong)."""
    return (
        (os.environ.get("CORESMITH_BLOCK_RTL_COMPLETE_GATE", "1") or "1") != "0"
    )


def _deterministic_caravel_top_enabled() -> bool:
    """Defect 4: when the design carries a Caravel pad-adapter block (a block
    named ``user_project_wrapper`` or one exposing io_in/io_out/io_oeb), assemble
    the wired ``user_project_wrapper`` chip_top deterministically instead of
    asking the Integration Lead LLM (which named the top after the design and
    treated the pad adapter as a peer, so the daemon never delivered a gradeable
    wired top). Default ON; set ``CORESMITH_DETERMINISTIC_CARAVEL_TOP=0`` to
    restore the LLM integration path."""
    return (
        (os.environ.get("CORESMITH_DETERMINISTIC_CARAVEL_TOP", "1") or "1") != "0"
    )


# rung3-fixes-1 (defect 2): a defensive ceiling on the number of consecutive
# retry/fix_rtl re-parks integration_check will issue before it fails closed to
# a LOUD terminal abort. In production each re-park is a real ``interrupt()``
# that SUSPENDS the graph (so this is never a CPU loop -- it bounds an operator/
# outer-agent that keeps sending retry without resolving). It also stops a
# plain-return ``interrupt`` test double from spinning. Never a silent END.
_INTEGRATION_REPARK_CAP = 50


def _mismatch_key(m: dict) -> tuple:
    """Dedup identity for a mismatch dict: (from_block, to_block, issue_type)."""
    return (
        str(m.get("from_block", "")),
        str(m.get("to_block", "")),
        str(m.get("issue_type", "")),
    )


def _merge_mismatches(
    llm_mismatches: list, deterministic_mismatches: list,
) -> list[dict]:
    """Merge the Integration Lead agent's mismatches with the deterministic
    checker's, deduped on ``(from_block, to_block, issue_type)``.

    The deterministic width/direction/missing-port findings are authoritative
    on severity: when a deterministic error collides with an LLM entry for the
    same (from, to, type) the merged entry keeps ``severity="error"`` so it
    flows into the existing ``integration_failure`` interrupt. Deterministic-only
    findings are appended and tagged ``deterministic=True``.
    """
    merged: list[dict] = []
    index: dict[tuple, int] = {}
    for m in llm_mismatches or []:
        if not isinstance(m, dict):
            continue
        entry = dict(m)
        index[_mismatch_key(entry)] = len(merged)
        merged.append(entry)
    for m in deterministic_mismatches or []:
        if not isinstance(m, dict):
            continue
        key = _mismatch_key(m)
        if key in index:
            existing = merged[index[key]]
            existing["deterministic"] = True
            if m.get("severity") == "error":
                existing["severity"] = "error"
        else:
            entry = dict(m)
            entry["deterministic"] = True
            index[key] = len(merged)
            merged.append(entry)
    return merged


async def integration_check_node(state: OrchestratorState) -> dict:
    """Run the Integration Lead agent to check compatibility and generate top-level RTL.

    After all blocks complete, this node:
    1. Loads architecture connections (block diagram)
    2. Discovers and reads all completed block RTL sources
    3. Calls the IntegrationLeadAgent to analyze compatibility and
       generate the top-level integration module
    4. Writes the generated Verilog to disk
    5. Lints the integrated design

    If errors are found, fires an interrupt with structured mismatch data
    so the outer agent can diagnose and fix.
    """
    import asyncio

    from orchestrator.langchain.agents.integration_lead import IntegrationLeadAgent

    pr = state.get("project_root", str(PROJECT_ROOT))
    # Phase-filtered + deduped (R1): integration_check runs in the rtl phase, so
    # this picks the pass-2 results (and, flag-off, all phase-less entries).
    completed = _current_phase_completed(state)
    passed_blocks = [b for b in completed if b.get("success")]
    block_queue = state.get("block_queue", [])
    # Join the RESULT dicts to their SPECS. A block result carries
    # {name, success, attempts, ...} and NOT rtl_target, so discovery would fall
    # back to filename convention and silently drop the one block whose file
    # stem differs from its name -- the contract-locked pad adapter. The spec
    # was already in scope here and used only for len().
    from orchestrator.langgraph.integration_helpers import merge_block_specs
    passed_blocks = merge_block_specs(passed_blocks, block_queue)
    expected_blocks = len(block_queue) if block_queue else len(completed)

    write_graph_event(pr, "Integration Check", "graph_node_enter", {
        "total_blocks": len(completed),
        "passed_blocks": len(passed_blocks),
        "expected_blocks": expected_blocks,
    })

    with _tracer.start_as_current_span("Integration Check") as span:
        span.set_attribute("total_blocks", len(completed))
        span.set_attribute("passed_blocks", len(passed_blocks))
        span.set_attribute("expected_blocks", expected_blocks)

        if expected_blocks and len(passed_blocks) < expected_blocks:
            failed_names = [
                b.get("name", "unknown") for b in completed
                if not b.get("success")
            ]
            completed_names = {b.get("name") for b in completed}
            missing_names = [
                b.get("name", "unknown") for b in block_queue
                if b.get("name") not in completed_names
            ]
            log(
                "  [INTEGRATION] Refusing partial integration: "
                f"{len(passed_blocks)}/{expected_blocks} blocks passed; "
                f"failed={failed_names}, missing={missing_names}",
                RED,
            )
            result = {
                "aborted": True,
                "skipped": True,
                "reason": (
                    f"Refusing partial integration: "
                    f"{len(passed_blocks)}/{expected_blocks} blocks passed; "
                    f"failed={failed_names}, missing={missing_names}"
                ),
                "error": "partial_block_set",
                "error_count": max(1, expected_blocks - len(passed_blocks)),
                "passed_blocks": len(passed_blocks),
                "expected_blocks": expected_blocks,
                "failed_blocks": failed_names,
                "missing_blocks": missing_names,
            }
            write_graph_event(pr, "Integration Check", "graph_node_exit", result)
            return {"integration_result": result}

        # C7(b): contract-staleness preflight. A block whose uarch spec was
        # generated against an OLDER interface contract than the live one
        # carries stale widths/fields in its RTL; the Integration Lead would
        # then bridge the mismatch with adapters (observed: trunc_9_to_8
        # silently destroying the REWIND opcode bit -- the chip lints clean
        # and decodes nothing). Catch it BEFORE spending the Lead call.
        # Default-on; opt out with CORESMITH_INTEGRATION_STALENESS_GATE=0.
        # Only blocks with a recorded contract stamp are checked, so runs
        # predating the stamp are unaffected.
        _stale_gate_on = os.environ.get(
            "CORESMITH_INTEGRATION_STALENESS_GATE", ""
        ).strip().lower() not in {"0", "false", "no", "off"}
        if _stale_gate_on:
            from orchestrator.langgraph.pipeline_helpers import (
                stale_uarch_spec_blocks,
            )
            _passed_names = [b.get("name", "") for b in passed_blocks
                             if b.get("name")]
            _stale = stale_uarch_spec_blocks(pr, _passed_names)
            if _stale:
                _stale_names = [s["block"] for s in _stale]
                log(f"  [INTEGRATION] STALE uarch specs: {_stale_names} -- "
                    f"the interface contract changed after these specs were "
                    f"generated; refusing to assemble", RED)
                write_graph_event(pr, "Integration Check",
                                  "integration_staleness_blocked",
                                  {"stale_blocks": _stale_names})
                payload = {
                    "type": "integration_failure",
                    "error_kind": "stale_uarch_specs",
                    "stale_blocks": _stale,
                    "error_count": len(_stale),
                    "supported_actions": ["retry", "override", "abort"],
                    "outer_agent_guidance": (
                        "Contract-staleness preflight: these blocks' uarch "
                        "specs were generated against an OLDER interface "
                        "contract than the live one, so their RTL carries "
                        "stale widths/fields; assembling would force "
                        "truncation adapters that destroy the amended "
                        "semantics. For each stale block: invalidate its "
                        "recorded sim-pass (set sim_passed=false in "
                        ".coresmith/blocks/<b>/best_result.json) and drive a "
                        "re-spec/regen of that block against the live "
                        "contract, then resume `retry` to re-run this "
                        "preflight. `override` proceeds anyway -- ONLY for a "
                        "verified false alarm. `abort` ends integration."
                    ),
                }
                response = interrupt(payload) or {}
                _act = response.get("action", "retry")
                write_graph_event(pr, "Integration Check",
                                  "integration_staleness_resume",
                                  {"action": _act})
                if _act == "override":
                    log("  [INTEGRATION] staleness OVERRIDE by chip-lead -- "
                        "assembling despite stale specs", YELLOW)
                elif _act == "abort":
                    result = {
                        "aborted": True, "skipped": True,
                        "reason": ("stale uarch specs (aborted): "
                                   f"{_stale_names}"),
                        "error": "stale_uarch_specs",
                        "error_count": len(_stale),
                        "stale_blocks": _stale_names,
                    }
                    write_graph_event(pr, "Integration Check",
                                      "graph_node_exit", result)
                    return {"integration_result": result}
                else:  # retry (after out-of-band re-spec) -> re-check NOW
                    _stale2 = stale_uarch_spec_blocks(pr, _passed_names)
                    if _stale2:
                        _s2 = [s["block"] for s in _stale2]
                        log(f"  [INTEGRATION] staleness persists after retry "
                            f"({_s2}) -- ending integration (fail-closed, "
                            f"re-drive the stale blocks first)", RED)
                        result = {
                            "aborted": True, "skipped": True,
                            "reason": ("stale uarch specs persist after "
                                       f"retry: {_s2}"),
                            "error": "stale_uarch_specs",
                            "error_count": len(_stale2),
                            "stale_blocks": _s2,
                        }
                        write_graph_event(pr, "Integration Check",
                                          "graph_node_exit", result)
                        return {"integration_result": result}
                    log("  [INTEGRATION] staleness cleared on retry -- "
                        "proceeding to assembly", GREEN)

        connections, design_name = await asyncio.to_thread(
            load_architecture_connections, pr
        )

        if not connections and len(passed_blocks) < 1:
            log("  [INTEGRATION] No architecture connections found -- "
                "skipping integration check", YELLOW)
            write_graph_event(pr, "Integration Check", "graph_node_exit", {
                "skipped": True,
                "reason": "no_connections",
            })
            return {"integration_result": {
                "skipped": True,
                "reason": "No architecture connections found",
            }}

        log(f"  [INTEGRATION] Found {len(connections)} connections, "
            f"design: {design_name}", CYAN)
        span.set_attribute("connection_count", len(connections))

        rtl_paths = await asyncio.to_thread(
            discover_block_rtl, pr, passed_blocks
        )

        # BLOCK-RTL COMPLETENESS GATE: refuse to assemble a chip that is MISSING
        # a block. `discover_block_rtl` used to drop an unresolvable block
        # SILENTLY, which structurally DELETES it from the chip: one graded run
        # lost its locked Caravel pad adapter (whose file is
        # rtl/user_project_wrapper.v, NOT <block_name>.v, because an interface
        # contract locks the module name), so `detect_wrapper_block` returned
        # None, the DEFAULT-ON deterministic Caravel assembler never ran, the LLM
        # Lead promoted the pad block's own qspi_* ports to the chip boundary, and
        # the assembled top -- and the netlist, and the GDS -- carried NO
        # io_in/io_out/io_oeb, while integration DV reported PASS on a co-tuned
        # BFM. A block that silently stops existing is an ASSEMBLY BLOCKER, not a
        # warning: park BEFORE spending the Lead call, in the same shape as the
        # staleness preflight above. Default-on; opt out with
        # CORESMITH_BLOCK_RTL_COMPLETE_GATE=0.
        if _block_rtl_complete_gate_enabled():
            from orchestrator.langgraph.integration_helpers import (
                missing_from,
            )
            # Judge the AUTHORITATIVE dict this node will actually assemble
            # from, not a second discovery -- a gate and an assembler working
            # off two different answers is its own bug.
            _missing_rtl = await asyncio.to_thread(
                missing_from, rtl_paths, passed_blocks
            )
            # Fires only on a PARTIAL resolution: some blocks resolved and at
            # least one did not, which is the silent-deletion defect (a chip
            # assembled AROUND a missing block). When NOTHING resolved there is
            # no chip to assemble at all and the existing "No block RTL could be
            # parsed" skip below already reports that honestly.
            if rtl_paths and _missing_rtl:
                log(f"  [INTEGRATION] BLOCK RTL UNRESOLVED for {_missing_rtl} -- "
                    f"refusing to assemble a chip that is missing a block (a "
                    f"silently dropped block is how a locked Caravel pad adapter "
                    f"vanished and the chip shipped with no GPIO boundary)", RED)
                write_graph_event(pr, "Integration Check",
                                  "block_rtl_unresolved",
                                  {"missing_blocks": _missing_rtl,
                                   "resolved_blocks": sorted(rtl_paths)})
                payload = {
                    "type": "integration_failure",
                    "error_kind": "unresolved_block_rtl",
                    "missing_block_rtl": _missing_rtl,
                    "resolved_block_rtl_paths": rtl_paths,
                    "error_count": len(_missing_rtl),
                    "supported_actions": ["retry", "override", "abort"],
                    "outer_agent_guidance": (
                        "These eligible blocks have NO locatable RTL file, so "
                        "assembling now would ship a chip with those blocks "
                        "DELETED -- no error, no instance, no ports. For each "
                        "one: confirm the .v file exists on disk and that the "
                        "block's `rtl_target` in .coresmith/block_specs.json "
                        "points at it. A block whose module name is locked by an "
                        "interface contract (e.g. a Caravel "
                        "user_project_wrapper) is NOT named <block_name>.v, so "
                        "`rtl_target` is the only correct answer for it. Then "
                        "resume `retry` to re-run this preflight. `override` "
                        "assembles WITHOUT those blocks -- only for a block that "
                        "genuinely does not belong in the chip. `abort` ends "
                        "integration."
                    ),
                    "reference_files": {
                        "block_specs": ".coresmith/block_specs.json",
                    },
                }
                response = interrupt(payload) or {}
                _act = response.get("action", "retry")
                write_graph_event(pr, "Integration Check",
                                  "block_rtl_unresolved_resume",
                                  {"action": _act})
                if _act == "override":
                    log("  [INTEGRATION] unresolved block RTL OVERRIDE by "
                        f"chip-lead -- assembling WITHOUT {_missing_rtl}", YELLOW)
                elif _act == "abort":
                    result = {
                        "aborted": True, "skipped": True,
                        "reason": ("unresolved block RTL (aborted): "
                                   f"{_missing_rtl}"),
                        "error": "unresolved_block_rtl",
                        "error_count": len(_missing_rtl),
                        "missing_blocks": _missing_rtl,
                    }
                    write_graph_event(pr, "Integration Check",
                                      "graph_node_exit", result)
                    return {"integration_result": result}
                else:  # retry (after fixing rtl_target / writing the file)
                    # Re-discover: a retry means rtl_target was just fixed or
                    # the file was just written. Then judge THAT dict.
                    rtl_paths = await asyncio.to_thread(
                        discover_block_rtl, pr, passed_blocks
                    )
                    _missing_rtl2 = await asyncio.to_thread(
                        missing_from, rtl_paths, passed_blocks
                    )
                    if _missing_rtl2:
                        log("  [INTEGRATION] block RTL STILL unresolved after "
                            f"retry ({_missing_rtl2}) -- ending integration "
                            "(fail-closed; resolve the block's rtl_target "
                            "first)", RED)
                        result = {
                            "aborted": True, "skipped": True,
                            "reason": ("block RTL unresolved after retry: "
                                       f"{_missing_rtl2}"),
                            "error": "unresolved_block_rtl",
                            "error_count": len(_missing_rtl2),
                            "missing_blocks": _missing_rtl2,
                        }
                        write_graph_event(pr, "Integration Check",
                                          "graph_node_exit", result)
                        return {"integration_result": result}
                    log("  [INTEGRATION] block RTL resolved on retry "
                        f"({len(rtl_paths)} blocks) -- proceeding to assembly",
                        GREEN)

        modules = {}
        block_rtl_sources: dict[str, str] = {}
        for block_name, rtl_path in rtl_paths.items():
            # Parse the BLOCK's module, not whichever comes first in
            # the file: generated files declare internal stages ahead
            # of the block itself.
            mod = await asyncio.to_thread(
                parse_verilog_ports, rtl_path,
                module_for_block(rtl_path, block_name))
            if mod.name:
                modules[block_name] = mod
                try:
                    block_rtl_sources[block_name] = Path(rtl_path).read_text(
                        encoding="utf-8", errors="replace"
                    )
                except OSError:
                    block_rtl_sources[block_name] = ""
                log(f"  [INTEGRATION] Parsed {block_name}: "
                    f"{len(mod.ports)} ports", GREEN)
            else:
                log(f"  [INTEGRATION] Failed to parse {block_name} "
                    f"at {rtl_path}", RED)

        span.set_attribute("parsed_blocks", len(modules))

        if not modules:
            log("  [INTEGRATION] No block RTL could be parsed", RED)
            write_graph_event(pr, "Integration Check", "graph_node_exit", {
                "error": "no_rtl_parsed",
            })
            return {"integration_result": {
                "skipped": True,
                "reason": "No block RTL could be parsed",
            }}

        block_port_summaries = []
        for name, mod in sorted(modules.items()):
            block_port_summaries.append({
                "name": name,
                "port_count": len(mod.ports),
                "ports": [p.to_dict() for p in mod.ports],
            })

        prd_summary = ""
        for prd_name in ("prd_spec.json", "ers_spec.json"):
            prd_path = Path(pr) / ".coresmith" / prd_name
            if prd_path.exists():
                try:
                    prd_data = json.loads(prd_path.read_text(encoding="utf-8"))
                    doc = prd_data.get("prd", prd_data.get("ers", {}))
                    prd_summary = doc.get("summary", "")
                    if doc.get("speed_and_feeds"):
                        sf = doc["speed_and_feeds"]
                        prd_summary += (
                            f"\nTarget clock: {sf.get('target_clock_mhz', '?')} MHz"
                        )
                    if doc.get("dataflow"):
                        df = doc["dataflow"]
                        prd_summary += (
                            f"\nBus protocol: {df.get('bus_protocol', '?')}"
                            f", Data width: {df.get('data_width_bits', '?')} bits"
                        )
                except (OSError, json.JSONDecodeError, KeyError):
                    pass
                break

        rtl_dir = Path(pr) / "rtl" / "integration"
        rtl_dir.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', design_name).lower()
        if not safe_name or safe_name[0].isdigit():
            safe_name = f"top_{safe_name}"
        output_path = str(rtl_dir / f"{safe_name}.v")

        # Single-block designs: generate a passthrough wrapper that
        # instantiates the block and wires all ports to the top level.
        # This ensures the backend always has an integration top-level
        # module regardless of block count.
        if len(modules) == 1:
            solo_name, solo_mod = next(iter(modules.items()))
            top_name = f"{safe_name}_top" if not safe_name.endswith("_top") else safe_name
            lines = [f"module {top_name} ("]
            port_decls = []
            for p in solo_mod.ports:
                width_str = f"[{p.msb}:{p.lsb}] " if p.width > 1 else ""
                port_decls.append(f"    {p.direction} wire {width_str}{p.name}")
            lines.append(",\n".join(port_decls))
            lines.append(");")
            lines.append("")
            inst_conns = [f"        .{p.name}({p.name})" for p in solo_mod.ports]
            lines.append(f"    {solo_mod.name} u_{solo_name} (")
            lines.append(",\n".join(inst_conns))
            lines.append("    );")
            lines.append("")
            lines.append("endmodule")
            wrapper_src = "\n".join(lines) + "\n"
            Path(output_path).write_text(wrapper_src, encoding="utf-8")

            log(f"  [INTEGRATION] Single-block design: generated wrapper "
                f"{top_name} for {solo_name}", GREEN)

            solo_rtl_path = list(rtl_paths.values())[0]
            lint_result = await asyncio.to_thread(
                lint_top_level, output_path, [solo_rtl_path], top_name
            )
            lint_clean = lint_result.get("clean", False)
            log(f"  [INTEGRATION] Lint: {'CLEAN' if lint_clean else 'ERRORS'}",
                GREEN if lint_clean else RED)

            integration_result = {
                "design_name": design_name,
                "top_module": top_name,
                "top_rtl_path": output_path,
                "block_count": 1,
                "wire_count": len(solo_mod.ports),
                "skipped_connections": [],
                "mismatches": [],
                "error_count": 0,
                "warning_count": 0,
                "lint_clean": lint_clean,
                "lint_errors": lint_result.get("errors", ""),
                "block_rtl_paths": rtl_paths,
                "single_block_wrapper": True,
            }

            write_graph_event(pr, "Integration Check", "graph_node_exit", {
                "success": True,
                "top_module": top_name,
                "block_count": 1,
                "single_block_wrapper": True,
            })
            return {"integration_result": integration_result}

        # ---- Defect 4: deterministic Caravel user_project_wrapper assembly ----
        # If a pad-adapter / Caravel wrapper block is present, assemble the wired
        # `user_project_wrapper` chip_top deterministically (locked Caravel ports,
        # instantiates + wires every block) rather than asking the Integration
        # Lead LLM, which named the top after the design and treated the pad
        # adapter as a peer block -- so the daemon never delivered a gradeable
        # wired top and every chip-lead hand-assembled one.
        from orchestrator.langgraph.integration_helpers import (
            detect_wrapper_block,
            generate_caravel_wrapper_top,
            load_interface_contract_edges,
        )
        _wrapper_block = detect_wrapper_block(modules)
        # The PRD's structured pin map, when present, lets the top route the pads
        # itself -- so the design needs no pin-adapter block and assembly no
        # longer depends on finding one.
        from orchestrator.architecture.pin_map import load_pin_map
        _pin_map = load_pin_map(pr)
        if _pin_map is not None and not _pin_map.ok:
            for _e in _pin_map.errors:
                log(f"  [INTEGRATION] pin_map: {_e}", RED)
            _pin_map = None
        if ((_wrapper_block is not None or _pin_map is not None)
                and _deterministic_caravel_top_enabled()):
            if _pin_map is not None:
                log(f"  [INTEGRATION] pin map declared ({len(_pin_map.entries)} "
                    f"signals) -- the top routes the pads itself; assembling "
                    f"wired user_project_wrapper deterministically", CYAN)
            else:
                log(f"  [INTEGRATION] Caravel wrapper block '{_wrapper_block}' "
                    f"detected -- assembling wired user_project_wrapper "
                    f"deterministically", CYAN)
            edges = await asyncio.to_thread(load_interface_contract_edges, pr)
            asm = await asyncio.to_thread(
                generate_caravel_wrapper_top,
                modules, edges, rtl_paths, str(rtl_dir), _wrapper_block,
                _pin_map,
            )
            # FAIL-LOUD (Section 2): if the deterministic assembler found a
            # wiring hazard it cannot safely resolve -- an ambiguous normalized
            # key it would otherwise [0]-pick, or a width mismatch it would
            # short/truncate -- do NOT ship the mis-wired top. Fall through to
            # the LLM Integration-Lead, which can reason about the ambiguity.
            _wiring_errors = asm.get("wiring_errors") or []
            if _wiring_errors:
                log(f"  [INTEGRATION] Deterministic Caravel top has "
                    f"{len(_wiring_errors)} wiring hazard(s) -- falling back to "
                    f"Integration Lead:", RED)
                for _we in _wiring_errors[:8]:
                    log(f"      - {_we}", RED)
                # Persist the FULL list. Logging [:8] and storing [:16] left
                # 24 of 40 hazards existing nowhere on disk -- and this list is
                # the only artifact explaining why assembly refused, so an outer
                # agent could not obtain it by any documented path.
                _haz_path = Path(pr) / ".coresmith" / "caravel_wiring_errors.json"
                try:
                    _haz_path.write_text(json.dumps({
                        "wrapper_block": _wrapper_block,
                        "count": len(_wiring_errors),
                        "wiring_errors": _wiring_errors,
                    }, indent=2), encoding="utf-8")
                except OSError:
                    pass
                write_graph_event(pr, "Integration Check", "caravel_wiring_fallback", {
                    "wrapper_block": _wrapper_block,
                    "wiring_error_count": len(_wiring_errors),
                    "wiring_errors": _wiring_errors[:16],
                    "wiring_errors_path": str(_haz_path),
                })
            if not _wiring_errors:
                top_rtl_path = asm["rtl_path"]
                # Lint with the pad block's renamed copy swapped in (avoids a
                # duplicate `module user_project_wrapper` definition at chip level).
                _lint_paths = list(asm["lint_block_paths"].values())
                lint_result = await asyncio.to_thread(
                    lint_top_level, top_rtl_path, _lint_paths, "user_project_wrapper"
                )
                lint_clean = lint_result.get("clean", False)
                # Postcondition: every block is instantiated in the assembled top.
                # A block the assembler deliberately DROPPED is not missing.
                # With a pin map the pad adapter is replaced by routing emitted
                # in the top, so requiring its instantiation would fail the
                # postcondition on the very design that fixed the problem.
                _dropped = asm.get("dropped_adapter") or ""
                missing = [b for b in modules
                           if b != _dropped and f"u_{b} (" not in asm["verilog"]]
                log(f"  [INTEGRATION] Caravel top: {len(asm['instantiated'])} blocks "
                    f"instantiated, {asm['wire_count']} internal wires, lint "
                    f"{'CLEAN' if lint_clean else 'ERRORS'}",
                    GREEN if lint_clean and not missing else YELLOW)
                integration_result = {
                    "design_name": design_name,
                    "top_module": "user_project_wrapper",
                    "top_rtl_path": top_rtl_path,
                    "block_count": asm["block_count"],
                    "wire_count": asm["wire_count"],
                    "skipped_connections": [],
                    "mismatches": [],
                    "error_count": 0 if not missing else len(missing),
                    "warning_count": 0,
                    "lint_clean": lint_clean,
                    "lint_errors": lint_result.get("errors", ""),
                    "block_rtl_paths": asm["lint_block_paths"],
                    "caravel_wrapper_assembled": True,
                    "renamed_pad_path": asm["renamed_pad_path"],
                    "missing_instantiations": missing,
                }
                write_graph_event(pr, "Integration Check", "graph_node_exit", {
                    "success": bool(lint_clean and not missing),
                    "top_module": "user_project_wrapper",
                    "block_count": asm["block_count"],
                    "wire_count": asm["wire_count"],
                    "lint_clean": lint_clean,
                    "caravel_wrapper_assembled": True,
                })
                # C23: FAIL-CLOSED. Only return the deterministic assembly as
                # the integration result when it is actually clean. A
                # not-lint-clean or missing-instance assembly must NOT return
                # success (which ended the run status=done / pipeline_done=false
                # with no next nodes and no retry) -- fall through to the
                # Integration Lead + integration_failure interrupt, the same
                # fail-closed retry path the generic branch uses.
                if lint_clean and not missing:
                    return {"integration_result": integration_result}
                log(f"  [INTEGRATION] deterministic Caravel assembly NOT clean "
                    f"(lint_clean={lint_clean}, missing={missing}) -- "
                    f"escalating to Integration Lead / fail-closed interrupt",
                    YELLOW)
            # else: wiring hazards / not-clean assembly -> fall through to the
            # Integration Lead below (which raises the integration_failure
            # interrupt for retry).

        log("  [INTEGRATION] Calling Integration Lead agent...", YELLOW)
        agent = IntegrationLeadAgent()
        try:
            agent_result = await agent.integrate(
                design_name=design_name,
                block_rtl_sources=block_rtl_sources,
                block_port_summaries=block_port_summaries,
                connections=connections,
                prd_summary=prd_summary,
                output_path=output_path,
            )
        except Exception as e:
            log(f"  [INTEGRATION] Agent failed: {e}", RED)
            write_graph_event(pr, "Integration Check", "graph_node_exit", {
                "error": str(e), "phase": "agent_call",
            })
            return {"integration_result": {
                "skipped": True,
                "reason": f"Integration Lead agent failed: {e}",
            }}

        if agent_result.get("parse_error"):
            log("  [INTEGRATION] Agent returned unparseable response", RED)
            write_graph_event(pr, "Integration Check", "graph_node_exit", {
                "error": "parse_error",
            })
            return {"integration_result": {
                "skipped": True,
                "reason": "Integration Lead agent returned unparseable response",
                "notes": agent_result.get("notes", ""),
            }}

        mismatches = agent_result.get("mismatches", [])
        module_name = agent_result.get("module_name", design_name)
        top_rtl_path = agent_result.get("rtl_path", output_path)

        # A-Fix 3(a): run the deterministic compatibility checker over the same
        # connections + parsed modules and MERGE its findings with the agent's.
        # Deterministic width/direction/missing-port mismatches carry
        # severity="error" and dedupe against the LLM's, so they flow into the
        # existing integration_failure interrupt where `accept` stays the
        # operator override. Wrapped in gate_guard so a checker exception is
        # fail-closed (surfaced as an error mismatch), never a silent pass.
        if _deterministic_integration_check_enabled():
            from orchestrator.langgraph.gate_guard import gate_guard
            from orchestrator.langgraph.integration_helpers import (
                check_integration_compatibility,
            )

            gr = gate_guard(
                "integration_compat",
                check_integration_compatibility,
                connections,
                modules,
            )
            if gr.errored:
                log(
                    "  [INTEGRATION] Deterministic compat check ERRORED "
                    f"(fail-closed): {gr.reason}",
                    RED,
                )
                mismatches = _merge_mismatches(mismatches, [{
                    "from_block": "", "to_block": "",
                    "issue_type": "compat_check_error", "severity": "error",
                    "description": (
                        "Deterministic integration compatibility check ERRORED "
                        f"(fail-closed): {gr.reason}"
                    ),
                    "suggested_fix": (
                        "NOT a pass -- fix the compatibility-check environment, "
                        "then resume 'retry'."
                    ),
                    "details": {"error": (gr.error or "")[:1000]},
                }])
            elif not gr.skipped:
                det = [m.to_dict() for m in (gr.value or [])]
                det_errors = sum(
                    1 for m in det if m.get("severity") == "error"
                )
                log(
                    "  [INTEGRATION] Deterministic compat check: "
                    f"{len(det)} finding(s), {det_errors} error(s)",
                    YELLOW if det else GREEN,
                )
                mismatches = _merge_mismatches(mismatches, det)

        chip_top_text = ""
        if top_rtl_path and os.path.exists(top_rtl_path):
            try:
                chip_top_text = Path(top_rtl_path).read_text()
            except OSError:
                chip_top_text = ""
        if not chip_top_text:
            chip_top_text = agent_result.get("verilog", "")

        from orchestrator.langchain.agents.integration_lead import (
            assert_blocks_instantiated,
        )
        postcond = assert_blocks_instantiated(
            chip_top_text, set(block_rtl_sources.keys())
        )
        if postcond:
            log(f"  [INTEGRATION] Postcondition failed: {postcond}", RED)
            write_graph_event(pr, "Integration Check", "graph_node_exit", {
                "error": "block_instantiation_postcondition_failed",
                "missing_summary": postcond,
            })
            return {"integration_result": {
                "skipped": True,
                "reason": postcond,
                "postcondition_failed": True,
                "agent_notes": agent_result.get("notes", ""),
                "top_rtl_path": top_rtl_path,
            }}

        # Memory-primitive postcondition (fix #3): the integration LLM must
        # INSTANTIATE library memory cells (cs_mem/cs_sram/cs_fpmem), never
        # DEFINE them -- an authored empty blackbox body is what the first-wins
        # dedup locked in, giving an all-zero memory in DV. Force a retry if the
        # top redefines a memory primitive.
        from orchestrator.langchain.agents.integration_lead import (
            assert_no_memory_primitive_defined,
        )
        mem_postcond = assert_no_memory_primitive_defined(chip_top_text)
        if mem_postcond:
            log(f"  [INTEGRATION] Postcondition failed: {mem_postcond}", RED)
            write_graph_event(pr, "Integration Check", "graph_node_exit", {
                "error": "memory_primitive_defined_postcondition_failed",
                "missing_summary": mem_postcond,
            })
            return {"integration_result": {
                "skipped": True,
                "reason": mem_postcond,
                "postcondition_failed": True,
                "agent_notes": agent_result.get("notes", ""),
                "top_rtl_path": top_rtl_path,
            }}

        # FUNCTIONAL-IFDEF POSTCONDITION (rung3 split-brain ban) -- the same
        # generation-time gate the per-block RTL gets, applied to chip_top. A
        # design module with a two-implementation `ifdef split-brain is
        # forbidden; the LEGITIMATE synth-blackbox / sim-behavioral pair of a
        # macro-named memory module (the SRAM-model idiom the dedup preserves)
        # is ALLOWED (find_functional_ifdef_regions carves it out). Force a
        # retry so the Integration Lead writes ONE implementation.
        # CORESMITH_IFDEF_LINT=0 bypasses.
        try:
            from orchestrator.langgraph.rtl_storage_lint import (
                find_functional_ifdef_regions,
                format_ifdef_lint_report,
                ifdef_lint_enabled,
            )
            if ifdef_lint_enabled() and chip_top_text:
                _ir = find_functional_ifdef_regions(chip_top_text)
                if not _ir.ok:
                    _imsg = format_ifdef_lint_report(_ir, block=module_name)
                    log(f"  [INTEGRATION] Postcondition failed: split-brain "
                        f"`ifdef in chip_top ({len(_ir.findings)} region(s))", RED)
                    write_graph_event(pr, "Integration Check", "graph_node_exit", {
                        "error": "functional_ifdef_postcondition_failed",
                        "missing_summary": _imsg[:400],
                    })
                    return {"integration_result": {
                        "skipped": True,
                        "reason": _imsg,
                        "postcondition_failed": True,
                        "agent_notes": agent_result.get("notes", ""),
                        "top_rtl_path": top_rtl_path,
                    }}
        except Exception as _e:  # never let the gate crash integration
            log(f"  [INTEGRATION] functional-ifdef gate error (skipped): {_e}",
                YELLOW)

        log(f"  [INTEGRATION] Agent generated {module_name}: "
            f"{len(modules)} blocks, "
            f"{agent_result.get('wire_count', 0)} wires", GREEN)
        span.set_attribute("top_module", module_name)

        block_rtl_list = list(rtl_paths.values())
        lint_result = await asyncio.to_thread(
            lint_top_level, top_rtl_path, block_rtl_list,
            design_name
        )

        lint_clean = lint_result.get("clean", False)
        log(f"  [INTEGRATION] Lint: {'CLEAN' if lint_clean else 'ERRORS'}",
            GREEN if lint_clean else RED)
        span.set_attribute("lint_clean", lint_clean)

        errors = [m for m in mismatches if m.get("severity") == "error"]
        warnings = [m for m in mismatches if m.get("severity") == "warning"]

        integration_result = {
            "design_name": design_name,
            "top_module": module_name,
            "top_rtl_path": top_rtl_path,
            "block_count": len(modules),
            "wire_count": agent_result.get("wire_count", 0),
            "skipped_connections": agent_result.get("skipped_connections", []),
            "mismatches": mismatches,
            "error_count": len(errors),
            "warning_count": len(warnings),
            "lint_clean": lint_clean,
            "lint_errors": lint_result.get("errors", ""),
            "lint_log_path": lint_result.get("log_path", ""),
            "block_rtl_paths": rtl_paths,
            "agent_notes": agent_result.get("notes", ""),
            "parsed_modules": {
                name: {
                    "port_count": len(mod.ports),
                    "inputs": len(mod.inputs()),
                    "outputs": len(mod.outputs()),
                }
                for name, mod in modules.items()
            },
        }

        # B3: persist the assembled integration result so the harness
        # (`coresmith verify chip`) can resolve top_rtl_path + block_rtl_paths
        # after the daemon parks. Best-effort -- never fails the node.
        try:
            _ir_path = Path(pr) / ".coresmith" / "integration_result.json"
            _ir_path.parent.mkdir(parents=True, exist_ok=True)
            _ir_path.write_text(json.dumps(integration_result, indent=2, default=str))
        except Exception:  # noqa: BLE001
            pass

        has_issues = len(errors) > 0 or not lint_clean
        if has_issues:
            log("  [INTEGRATION] Issues found -- interrupting for review", YELLOW)

            payload = {
                "type": "integration_failure",
                "design_name": design_name,
                "top_rtl_path": top_rtl_path,
                "block_count": len(modules),
                "error_count": len(errors),
                "warning_count": len(warnings),
                "lint_clean": lint_clean,
                "mismatches": mismatches,
                "lint_errors": lint_result.get("errors", "")[:3000],
                "lint_log_path": lint_result.get("log_path", ""),
                "block_rtl_paths": rtl_paths,
                "skipped_connections": agent_result.get("skipped_connections", []),
                "supported_actions": (
                    # `accept` is only offered when the chip_top still
                    # lint-passes despite the architectural mismatches --
                    # the operator can then advance to DV without
                    # regenerating, because the issues are naming /
                    # design-intent drift rather than syntactic
                    # violations.
                    ["accept", "retry", "fix_rtl", "skip", "abort"]
                    if lint_clean
                    else ["retry", "fix_rtl", "skip", "abort"]
                ),
                "outer_agent_guidance": (
                    "Integration Lead agent found issues. As the outer-loop "
                    "diagnostic agent, diagnose and fix before escalating:\n"
                    "1. WIDTH_MISMATCH: Read both block RTL files. Edit the RTL "
                    "on disk, then resume_pipeline(action='fix_rtl', "
                    "rtl_fix_description='Fixed width ...')\n"
                    "2. MISSING_PORT: Edit the block RTL to add it.\n"
                    "3. DIRECTION_ERROR: Fix the port direction.\n"
                    "4. LINT_ERRORS: Read the lint log and edit "
                    f"{top_rtl_path} directly.\n"
                    "5. After fixing, resume_pipeline(action='fix_rtl').\n"
                    "6. Only escalate for architectural issues.\n"
                    "7. ACCEPT: chip_top already lint-passes and the "
                    "mismatches are acceptable for this run -- proceed "
                    "to DV without further regeneration."
                ),
                "reference_files": {
                    "top_rtl": top_rtl_path,
                    "architecture": ".coresmith/architecture_state.json",
                    "block_diagram": ".coresmith/block_diagram_viz.json",
                    "lint_log": lint_result.get("log_path", ""),
                },
            }

            response = interrupt(payload)

            action = response.get("action", "abort")
            write_graph_event(pr, "Integration Check", "graph_node_exit", {
                "action": action,
                "error_count": len(errors),
                "lint_clean": lint_clean,
            })

            if action == "skip":
                integration_result["skipped_by_user"] = True
                log("  [INTEGRATION] Skipped by user/agent", YELLOW)
            elif action == "accept":
                # User/agent acknowledges the mismatch errors but the
                # chip_top still lint-passes -- advance to DV with the
                # existing top-level Verilog.  Mark the result so the
                # router stops short-circuiting on error_count > 0.
                integration_result["accepted_by_user"] = True
                log(
                    "  [INTEGRATION] Accepted despite "
                    f"{len(errors)} error(s) (lint_clean=True); "
                    "advancing to DV",
                    YELLOW,
                )
            elif action == "abort":
                integration_result["aborted"] = True
                log("  [INTEGRATION] Aborted", RED)
            elif action in ("retry", "fix_rtl"):
                fix_desc = response.get("rtl_fix_description", "")
                log(f"  [INTEGRATION] Fix applied: {fix_desc}", GREEN)
                integration_result["fix_applied"] = fix_desc
                # rung3-fixes-1 (defect 2): retry/fix_rtl records an on-disk edit
                # but this node CANNOT re-run the compatibility check in place
                # (that needs a restart_node so the RTL is re-parsed). Returning
                # here let route_after_integration END the graph SILENTLY with
                # errors still outstanding (status=done, pipeline_done=False, no
                # park, no error_message) -- the operator got no signal and DV
                # never ran. Fail-closed: RE-PARK a final integration_failure
                # interrupt that surfaces the outstanding errors and forces an
                # explicit accept (advance to DV; only when lint-clean) or abort
                # (restart_node to re-check). NEVER a silent END.
                repark_rounds = 0
                while (
                    action in ("retry", "fix_rtl")
                    and (len(errors) > 0 or not lint_clean)
                    and repark_rounds < _INTEGRATION_REPARK_CAP
                ):
                    repark_rounds += 1
                    integration_result["repark_rounds"] = repark_rounds
                    integration_result["fix_applied"] = (
                        response.get("rtl_fix_description", "")
                        if isinstance(response, dict)
                        else integration_result.get("fix_applied", "")
                    )
                    _repark_actions = (
                        ["accept", "abort"] if lint_clean else ["abort"]
                    )
                    _repark_msg = (
                        f"Integration still reports {len(errors)} outstanding "
                        "error(s)"
                        + ("" if lint_clean else " and chip_top does not lint-clean")
                        + f" after {repark_rounds} in-place fix attempt(s); the "
                        "compatibility check cannot be re-run in this node. "
                        + (
                            "ACCEPT to advance to DV (chip_top lint-passes) or "
                            if lint_clean else ""
                        )
                        + "ABORT and restart_node('integration_check') to "
                        "re-parse and re-check the edited RTL from scratch."
                    )
                    integration_result["error_message"] = _repark_msg
                    write_graph_event(
                        pr, "Integration Check", "integration_repark", {
                            "repark_round": repark_rounds,
                            "error_count": len(errors),
                            "lint_clean": lint_clean,
                            "prior_action": action,
                        },
                    )
                    log(
                        f"  [INTEGRATION] Re-park (round {repark_rounds}): "
                        f"{len(errors)} error(s) outstanding after '{action}'; "
                        "forcing accept/abort (no silent END)",
                        YELLOW,
                    )
                    repark_payload = {
                        "type": "integration_failure",
                        "design_name": design_name,
                        "top_rtl_path": top_rtl_path,
                        "block_count": len(modules),
                        "error_count": len(errors),
                        "warning_count": len(warnings),
                        "lint_clean": lint_clean,
                        "mismatches": mismatches,
                        "repark_round": repark_rounds,
                        "error_message": _repark_msg,
                        "supported_actions": _repark_actions,
                        "outer_agent_guidance": (
                            "Re-park after retry/fix_rtl at integration_check: "
                            "the outstanding errors were NOT cleared by an "
                            "in-node re-check (there is none). Do NOT expect "
                            "another retry to advance the graph. Either ACCEPT "
                            "(only offered when chip_top lint-passes; proceeds "
                            "to DV) or ABORT and "
                            "restart_node('integration_check') so the edited "
                            "RTL is re-parsed and re-checked from scratch."
                        ),
                        "block_rtl_paths": rtl_paths,
                        "reference_files": {
                            "top_rtl": top_rtl_path,
                            "architecture": ".coresmith/architecture_state.json",
                            "lint_log": lint_result.get("log_path", ""),
                        },
                    }
                    response = interrupt(repark_payload)
                    action = (
                        response.get("action", "abort")
                        if isinstance(response, dict) else "abort"
                    )

                # Re-park resolved (or there was nothing to re-park) -- finalize
                # on the terminal action. accept advances to DV (lint-clean
                # only); abort/unknown terminates; skip is honored.
                if action == "accept" and lint_clean:
                    integration_result["accepted_by_user"] = True
                    log(
                        "  [INTEGRATION] Accepted at re-park; advancing to DV",
                        YELLOW,
                    )
                elif action == "skip":
                    integration_result["skipped_by_user"] = True
                    log("  [INTEGRATION] Skipped at re-park", YELLOW)
                elif action in ("retry", "fix_rtl"):
                    if len(errors) > 0 or not lint_clean:
                        # Re-park CAP exhausted with issues still outstanding
                        # (a driver that kept sending retry). Fail-closed to a
                        # LOUD terminal abort -- never a silent END.
                        integration_result["aborted"] = True
                        integration_result["error_message"] = (
                            f"integration_check re-park cap "
                            f"({_INTEGRATION_REPARK_CAP}) exhausted with "
                            f"{len(errors)} error(s) still outstanding; "
                            "aborting. Fix the RTL on disk then "
                            "restart_node('integration_check') to re-check."
                        )
                        write_graph_event(
                            pr, "Integration Check",
                            "integration_repark_exhausted", {
                                "repark_rounds": repark_rounds,
                                "error_count": len(errors),
                                "lint_clean": lint_clean,
                            },
                        )
                        log(
                            "  [INTEGRATION] Re-park cap exhausted -- "
                            "aborting (fail-closed)",
                            RED,
                        )
                    else:
                        # Issues cleared between iterations -- record the fix
                        # and let routing proceed.
                        integration_result["fix_applied"] = (
                            response.get("rtl_fix_description", fix_desc)
                            if isinstance(response, dict) else fix_desc
                        )
                else:  # abort or unknown -> terminal, fail-closed
                    integration_result["aborted"] = True
                    log("  [INTEGRATION] Aborted at re-park", RED)

            return {"integration_result": integration_result}

        if (
            warnings
            and not os.getenv("CORESMITH_NONBLOCKING_INTEGRATION_WARNINGS")
        ):
            log(
                f"  [INTEGRATION] {len(warnings)} warning(s) -- triaging "
                "for outer-agent review",
                YELLOW,
            )

            warning_payload = {
                "type": "integration_warning_review",
                "design_name": design_name,
                "top_rtl_path": top_rtl_path,
                "block_count": len(modules),
                "error_count": 0,
                "warning_count": len(warnings),
                "lint_clean": True,
                "warnings": warnings,
                "mismatches": mismatches,
                "block_rtl_paths": rtl_paths,
                "skipped_connections": agent_result.get(
                    "skipped_connections", []
                ),
                "supported_actions": [
                    "accept",
                    "retry",
                    "fix_rtl",
                    "abort",
                ],
                "outer_agent_guidance": (
                    "Integration Lead agent flagged warnings but no hard "
                    "errors. Architecture warnings have caused DV deadlocks "
                    "in practice (closed AXI-Stream feedback loops without "
                    "a bootstrap policy, etc.), so triage before letting "
                    "the run reach DV:\n"
                    "1. Read each warning's `description` and "
                    "`suggested_fix`.\n"
                    "2. If the warning is benign or compensated elsewhere, "
                    "resume_pipeline(action='accept').\n"
                    "3. If a block needs patching, edit it on disk, then "
                    "resume_pipeline(action='fix_rtl', "
                    "rtl_fix_description='...'). The run will END so you "
                    "can issue restart_node for the affected stage.\n"
                    "4. If the integration top should be regenerated, "
                    "resume_pipeline(action='retry'). The run will END so "
                    "you can restart_node('integration_check').\n"
                    "5. If a uArch-level revision is needed (e.g. add a "
                    "request-driven bootstrap path), "
                    "resume_pipeline(action='abort') and escalate.\n"
                    "Set CORESMITH_NONBLOCKING_INTEGRATION_WARNINGS=1 to "
                    "restore the old non-blocking behavior."
                ),
                "reference_files": {
                    "top_rtl": top_rtl_path,
                    "architecture": ".coresmith/architecture_state.json",
                    "block_diagram": ".coresmith/block_diagram_viz.json",
                },
            }

            response = interrupt(warning_payload)
            action = (
                response.get("action", "abort")
                if isinstance(response, dict)
                else "abort"
            )
            integration_result["warning_triage_action"] = action

            write_graph_event(pr, "Integration Check", "graph_node_exit", {
                "action": action,
                "warning_count": len(warnings),
                "via": "warning_triage",
            })

            if action == "accept":
                integration_result["accepted_warnings"] = True
                log(
                    "  [INTEGRATION] Warnings accepted by outer agent",
                    GREEN,
                )
            elif action in ("retry", "fix_rtl"):
                fix_desc = response.get("rtl_fix_description", "")
                integration_result["fix_applied"] = fix_desc
                integration_result["aborted"] = True
                log(
                    f"  [INTEGRATION] {action} requested "
                    f"(desc='{fix_desc}'); routing to END so outer agent "
                    "can restart_node",
                    YELLOW,
                )
            else:  # abort or unknown
                integration_result["aborted"] = True
                log(
                    "  [INTEGRATION] Aborted on warning triage", RED
                )

            return {"integration_result": integration_result}

        log(f"\n{'='*60}", GREEN)
        log("  INTEGRATION CHECK PASSED", GREEN)
        log(f"  Top module: {module_name}", GREEN)
        log(f"  {len(modules)} blocks, "
            f"{agent_result.get('wire_count', 0)} wires", GREEN)
        if warnings:
            log(f"  {len(warnings)} warnings (non-blocking)", YELLOW)
        log(f"{'='*60}\n", GREEN)

        write_graph_event(pr, "Integration Check", "graph_node_exit", {
            "success": True,
            "top_module": module_name,
            "block_count": len(modules),
            "wire_count": agent_result.get("wire_count", 0),
            "warnings": len(warnings),
        })

        return {"integration_result": integration_result}


def route_after_integration(state: OrchestratorState) -> str:
    """Route after integration check: proceed to DV or END.

    Two-pass (block-goldens on): the µarch gate already ran BEFORE RTL, so this
    routes straight to ``integration_dv`` (no post-integration_check
    model_integration node). Single-pass (flag off): unchanged -- routes to the
    ``model_integration`` node (a flag-gated no-op pass-through to DV), so the
    topology is byte-identical to before the restructure.
    """
    from orchestrator.architecture import composition as _composition
    next_node = (
        "integration_dv"
        if _composition.block_goldens_enabled()
        else "model_integration"
    )
    result = state.get("integration_result") or {}
    if result.get("aborted"):
        return END
    if result.get("skipped") or result.get("skipped_by_user"):
        return END
    if result.get("lint_clean") is False:
        return END
    # `accepted_by_user` overrides the error_count gate: the operator
    # has acknowledged the architectural mismatches are acceptable for
    # this run and chip_top still lint-passes, so DV is allowed to run.
    if result.get("accepted_by_user"):
        return next_node
    if int(result.get("error_count", 0) or 0) > 0:
        return END
    return next_node


route_after_integration.__edge_labels__ = {
    END: "DONE",
    "model_integration": "Model Integration",
    "integration_dv": "DV",
}


# ---------------------------------------------------------------------------
# Node: model_integration  (LLM model-integration + deterministic Amaranth verify)
# ---------------------------------------------------------------------------
#
# Behind CORESMITH_BLOCK_GOLDENS. When OFF (default) this node is a no-op
# pass-through to integration_dv -- byte-identical routing to before the
# feature existed. When ON it (a) calls the model-integration LLM agent to wire
# every per-block Amaranth block model into a top-level Amaranth chip model
# (arch/block_models/_chip_model.py) if one is not already present, then (b)
# runs the deterministic model-integration gate, which simulates the integrated
# chip model on a stimulus and asserts its output equals the reference
# implementation's output BIT-EXACT. On divergence it PARKS an interrupt
# (exactly like validation_dv) naming the first-divergence block; otherwise it
# passes through to integration_dv.


def _shape_of(value) -> str:
    """Describe a value's CONTAINER (type + keys + element types), NEVER its
    values -- a contract hint for the integrator, not the oracle."""
    if isinstance(value, dict):
        parts = []
        for k, v in value.items():
            parts.append(f"{k!r}: {_shape_of(v)}")
        return "a dict with keys [" + ", ".join(repr(k) for k in value) + "] " \
            "-> {" + ", ".join(parts) + "}"
    if isinstance(value, (list, tuple)):
        kind = "list" if isinstance(value, list) else "tuple"
        if not value:
            return f"an empty {kind}"
        elem = _shape_of(value[0])
        return f"a {kind}[{elem}] (len={len(value)})"
    if isinstance(value, (bytes, bytearray)):
        return f"bytes (len={len(value)})"
    return type(value).__name__


def _describe_reference_output_shape(pr: str, ref_entry_callable) -> str:
    """Run the reference on the gate's actual stimulus and describe the SHAPE of
    its return container (type/keys/element types -- never values). This tells the
    integrator the EXACT container simulate() must return so a multi-output design
    yields a dict-with-all-keys, not a flat list (gap_class=contract otherwise)."""
    if ref_entry_callable is None:
        return ""
    from orchestrator.architecture import model_integration as _mi
    # Reuse the gate's stimulus resolution (env file wins, else a small derived
    # default) so the shape is exactly what the gate will expect.
    stim, found = _mi._load_env_stimulus()
    if not found:
        stim = _mi._default_stimulus(ref_entry_callable)
    if stim is None:
        return ""
    from orchestrator.architecture import composition as _composition
    out = _composition._run_reference(ref_entry_callable, stim, reraise=False)
    if out is None:
        return ""
    return _shape_of(out)


async def _maybe_generate_chip_model(pr: str) -> None:
    """LLM-generate arch/block_models/_chip_model.py if it is missing.

    Best-effort: any failure is logged and swallowed -- the deterministic gate
    then no-ops (no _chip_model.py) or flags the divergence.
    """
    from orchestrator.architecture import composition as _composition

    root = Path(pr)
    models_dir = root / "arch" / _composition.BLOCK_MODELS_DIRNAME
    if not models_dir.is_dir() or not any(models_dir.glob("*.py")):
        return
    chip_model_path = models_dir / "_chip_model.py"
    # Reuse the composed chip model ONLY if present and not stale. A revise_uarch
    # re-generates the per-block models; blindly keeping the old _chip_model.py
    # made the gate re-compose a STALE model and emit bit-identical (wrong) output
    # on every revise, so the bounded revise loop could never converge.
    if not _chip_model_needs_regen(models_dir, chip_model_path):
        return
    if chip_model_path.exists():
        log("  [MODEL-INTEGRATION] block models changed since _chip_model.py "
            "(stale); regenerating the composed chip model so revise actually "
            "re-composes.", YELLOW)

    # Generators need the FULL golden (per-block math), not the gate's bytes-only
    # wrapper -- use the generator-specific reference.
    ref_path = _composition.resolve_generator_reference(pr)
    if not ref_path:
        log("  [MODEL-INTEGRATION] no reference implementation; skipping "
            "chip-model generation", YELLOW)
        return
    try:
        reference_impl_source = Path(ref_path).read_text(encoding="utf-8")
    except OSError as exc:
        log(f"  [MODEL-INTEGRATION] cannot read {ref_path}: {exc}", YELLOW)
        return

    import json as _json
    block_diagram: dict = {}
    bd_path = root / ".coresmith" / "block_diagram.json"
    if bd_path.exists():
        try:
            block_diagram = _json.loads(bd_path.read_text(encoding="utf-8"))
        except (OSError, _json.JSONDecodeError):
            block_diagram = {}

    interface_contracts: dict = {}
    ic_path = root / ".coresmith" / "interface_contracts.json"
    if ic_path.exists():
        try:
            interface_contracts = _json.loads(ic_path.read_text(encoding="utf-8"))
        except (OSError, _json.JSONDecodeError):
            interface_contracts = {}

    # Reference entry name for the agent (so simulate() matches its shape).
    ref_entry_name = ""
    ref_entry_callable = None
    try:
        from orchestrator.architecture.model_integration import (
            _load_reference_module,
        )
        ref_module = _load_reference_module(ref_path)
        ref_entry_callable, ref_entry_name = (
            _composition.resolve_reference_entrypoint(pr, ref_module)
        )
    except Exception:  # noqa: BLE001
        ref_entry_name = ""

    # Derive the reference's OUTPUT CONTAINER shape so the integrator returns
    # the SAME container (dict-with-all-keys vs flat list). The gate compares
    # STRUCTURALLY -- a flat list when the reference returns a multi-key dict
    # is a contract failure even when the streamed bytes are byte-exact. We
    # describe the CONTAINER ONLY (type + keys + element types), never the
    # values, so this is a contract hint, not the oracle.
    ref_output_shape = ""
    try:
        ref_output_shape = _describe_reference_output_shape(pr, ref_entry_callable)
    except Exception:  # noqa: BLE001
        ref_output_shape = ""

    try:
        from orchestrator.langchain.agents.model_integration_generator import (
            ModelIntegrationGenerator,
        )
        agent = ModelIntegrationGenerator(temperature=0.1)
        log("  [MODEL-INTEGRATION] Generating integrated Amaranth chip model "
            "(_chip_model.py)...", YELLOW)
        if ref_output_shape:
            log(f"  [MODEL-INTEGRATION] reference output shape: "
                f"{ref_output_shape}", YELLOW)
        await agent.generate(
            project_root=pr,
            block_models_dir=str(models_dir),
            block_diagram=block_diagram,
            interface_contracts=interface_contracts,
            reference_impl_source=reference_impl_source,
            reference_entry_name=ref_entry_name,
            output_path=str(chip_model_path),
            reference_output_shape=ref_output_shape,
        )
        log(f"  [MODEL-INTEGRATION] Wrote {chip_model_path}", GREEN)
    except Exception as exc:  # noqa: BLE001 - best-effort; gate is the backstop
        # A rejected chip model means the composition gate has nothing to
        # compose: it NO-OPS, and the run proceeds with its strongest
        # model-vs-golden check silently absent. On the two runs that hit this,
        # the only trace was this one line, 400 lines up a daemon log. Carry it
        # forward so the final report and the validation-DV context both say a
        # gate did not run, and why.
        log(f"  [MODEL-INTEGRATION] chip-model generation failed ({exc}); the "
            f"COMPOSITION GATE will NO-OP (no _chip_model.py to compose) -- "
            f"the run loses its model-vs-golden check", RED)
        record_carried_forward_defect(pr, {
            "gate": "model_integration",
            "kind": "chip_model_generation_failed",
            "advisory": True,
            "first_divergence_block": "",
            "violation_count": 0,
            "unmodeled": (
                "the integrated chip model (_chip_model.py) was not produced, "
                "so the composition gate no-opped: NOTHING compared the wired "
                "block models against the golden reference on this run"),
            "detail": (
                f"chip-model generation failed: {exc}. The composition gate is "
                f"the run's model-vs-golden check; with no composed model it "
                f"returns 'not applicable' and every downstream verdict rests "
                f"on per-block DV alone."),
            "note": "",
        })


async def model_integration_node(state: OrchestratorState) -> dict:
    """Model-integration node: LLM integrate -> deterministic Amaranth verify.

    Runs after the last tier + integration_check and BEFORE integration_dv.
    """
    from orchestrator.architecture import composition as _composition
    from orchestrator.architecture import model_integration as _model_integration

    pr = state.get("project_root", str(PROJECT_ROOT))

    # Flag off -> pure no-op. No event, no state mutation: identical to today.
    if not _composition.block_goldens_enabled():
        return {}

    write_graph_event(pr, "Model Integration", "graph_node_enter", {})

    with _tracer.start_as_current_span("Model Integration") as span:
        # (a) LLM model-integration: write _chip_model.py if missing.
        try:
            await _maybe_generate_chip_model(pr)
        except Exception as exc:  # noqa: BLE001
            log(f"  [MODEL-INTEGRATION] chip-model gen raised: {exc}", YELLOW)

        # (b) deterministic Amaranth-simulation verify. A-Fix 2a: a gate that
        # RAISES is NOT a pass -- fail-closed. Under CORESMITH_GATE_FAIL_OPEN it
        # is tolerated (old fail-open behavior); otherwise a synthesized
        # gate_error violation falls through to the existing interrupt.
        from orchestrator.langgraph.gate_guard import gate_error_violation, gate_guard
        _gr = await asyncio.to_thread(
            gate_guard, "model_integration",
            _model_integration.run_model_integration_gate, pr,
        )
        if _gr.errored and _gr.skipped:
            # Global fail-open escape hatch engaged: tolerate the gate error.
            log(f"  [MODEL-INTEGRATION] gate raised; FAIL-OPEN escape active, "
                f"treating as pass: {_gr.reason}", YELLOW)
            span.set_attribute("error", _gr.error)
            write_graph_event(pr, "Model Integration", "graph_node_exit", {
                "skipped": True, "reason": "gate_error_fail_open",
            })
            return {"model_integration_result": {"passed": True, "error": _gr.reason}}
        if _gr.errored:
            log(f"  [MODEL-INTEGRATION] gate ERRORED (fail-closed, NOT a pass): "
                f"{_gr.reason}", RED)
            span.set_attribute("error", _gr.error)
            violations = [gate_error_violation(_gr.reason, _gr.error)]
        else:
            violations = _gr.value or []

        span.set_attribute("violation_count", len(violations))

        if not violations:
            # rung2 defect 1: distinguish a real PASS (gate compared a composed
            # chip model against a golden) from a no-op SKIP (goldenless /
            # requirements-only run). A no-op must NOT report passed=True.
            _status = _model_integration.describe_gate_status(pr)
            if not _status["applicable"]:
                span.set_attribute("skipped", True)
                span.set_attribute("skip_reason", _status["reason"])
                log("  [MODEL-INTEGRATION] SKIPPED (not applicable, NOT a "
                    f"pass): {_status['reason']}", YELLOW)
                _record_dv_row(
                    pr, block="chip_model", scope="chip_model", source="gate",
                    passed=False, skipped=True, detail=_status["reason"],
                )
                write_graph_event(pr, "Model Integration", "graph_node_exit", {
                    "skipped": True,
                    "reason": _status["reason"],
                    "gate_status": _status,
                })
                return {"model_integration_result": {
                    "skipped": True,
                    "reason": "no golden reference; gate not applicable",
                    "gate_status": _status,
                }}
            log("  [MODEL-INTEGRATION] PASSED (simulated chip model == reference)",
                GREEN)
            write_graph_event(pr, "Model Integration", "graph_node_exit", {
                "passed": True, "violation_count": 0,
            })
            return {"model_integration_result": {"passed": True}}

        # ADVISORY BYPASS (CORESMITH_DETERMINISTIC_BFM). The composition
        # _chip_model.py pin driver is LLM-authored and stimulus/DUT-fragile (it
        # can mis-decode the IN-window stimulus -> corrupt/all-zero composed
        # output DESPITE byte-correct per-block models). When the deterministic
        # integration DV is enabled it is the AUTHORITATIVE RTL-level contract
        # check downstream on the real chip_top, so a model-level composition
        # mismatch must NOT hard-block the run (nor trigger a full re-spec of all
        # blocks). Log LOUDLY, name the mismatch, and PROCEED to integration_dv.
        # Flag off -> this branch is never taken (byte-identical to before).
        from orchestrator.langgraph import bfm_lib as _bfm_lib
        if _bfm_lib.deterministic_bfm_enabled():
            first = violations[0]
            first_block = first.get("first_divergence_block", "")
            log(f"\n{'='*60}", YELLOW)
            log("  MODEL INTEGRATION GATE MISMATCH -- ADVISORY (non-blocking)",
                YELLOW)
            log("  CORESMITH_DETERMINISTIC_BFM=1: the deterministic integration "
                "DV on the real chip_top RTL is the authoritative contract "
                "check. The LLM-authored composition _chip_model.py harness is "
                "stimulus/DUT-fragile; NOT hard-blocking and NOT re-speccing.",
                YELLOW)
            log(f"  First-divergence block: {first_block or '(unlocalized)'}",
                YELLOW)
            for v in violations[:5]:
                log(f"    - expected {v.get('expected')!r} got "
                    f"{v.get('observed')!r}", YELLOW)
            # Section 3b: do NOT silently swallow a quantified mismatch. Record
            # a carried-forward defect naming the SPECIFIC unmodeled thing so it
            # surfaces in the final report + validation-DV context and is
            # re-checked on the real chip_top rather than lost behind "advisory".
            _defect = _advisory_composition_defect(pr, "model_integration", violations)
            record_carried_forward_defect(pr, _defect)
            if _defect.get("unmodeled"):
                log(f"  UNMODELED: {_defect['unmodeled']}", YELLOW)
            log(f"  PROCEEDING to integration_dv (advisory; carried forward).\n"
                f"{'='*60}\n", YELLOW)
            write_graph_event(pr, "Model Integration", "graph_node_exit", {
                "passed": False,
                "advisory_bypass": True,
                "violation_count": len(violations),
                "first_divergence_block": first_block,
                "carried_forward_defect": True,
                "unmodeled": _defect.get("unmodeled", ""),
            })
            return {"model_integration_result": {
                "passed": False,
                "advisory_bypass": True,
                "first_divergence_block": first_block,
                "violations": violations[:20],
                "carried_forward_defect": _defect,
            }}

        first = violations[0]
        first_block = first.get("first_divergence_block", "")
        # Per-field localization (model_integration sets affected_blocks when only
        # SOME output fields diverged). Carrying it into mir makes init_tier_node
        # re-spec ONLY those blocks (targeted) instead of broadcasting to all --
        # the convergence fix for framework-HDL composition runs.
        affected_blocks: list[str] = []
        for v in violations:
            for b in v.get("affected_blocks") or []:
                if b and b not in affected_blocks:
                    affected_blocks.append(b)
        log(f"\n{'='*60}", RED)
        log("  MODEL INTEGRATION GATE FAILED", RED)
        log(f"  First-divergence block: {first_block or '(unlocalized)'}", RED)
        if affected_blocks:
            log(f"  TARGETED re-spec blocks (field-localized): {affected_blocks}",
                RED)
        else:
            log("  Localization: unlocalized -> broadcast re-spec to all blocks",
                RED)
        for v in violations[:5]:
            log(f"    - expected {v.get('expected')!r} got "
                f"{v.get('observed')!r}", RED)
        log(f"{'='*60}\n", RED)

        write_graph_event(pr, "Model Integration", "graph_node_exit", {
            "passed": False,
            "violation_count": len(violations),
            "first_divergence_block": first_block,
        })

        payload = {
            "type": "model_integration_failure",
            "first_divergence_block": first_block,
            "affected_blocks": affected_blocks,
            "violations": violations[:20],
            "suggested_fix": first.get("suggested_fix", ""),
            "supported_actions": [
                "revise_uarch",  # regenerate the named block's uArch + model
                "fix_rtl",       # outer agent patched the block model / chip model on disk
                "retry",         # re-run the gate (after an on-disk fix)
                "abort",         # stop the pipeline
            ],
            "outer_agent_guidance": (
                "The model-integration agent wired every per-block Amaranth block "
                "model into a top-level Amaranth chip model; the deterministic gate "
                "simulated it and its output diverged from the reference "
                "implementation. The first-divergence block is "
                f"'{first_block or 'unlocalized'}'. As the outer-loop agent:\n"
                "1. Inspect arch/block_models/<block>.py for the named block and "
                "compare its transcribed math to the reference implementation; "
                "also inspect arch/block_models/_chip_model.py wiring / "
                "handshake / feedback.\n"
                "2. If you fixed the block model or chip model on disk, resume "
                "with action='retry'.\n"
                "3. To regenerate the uArch spec + block model via the LLM, "
                "resume with action='revise_uarch'.\n"
                "4. Only abort for a genuine architecture-level contradiction."
            ),
        }

        response = interrupt(payload)
        action = response.get("action", "abort")

        result = {
            "passed": False,
            "violations": violations[:20],
            "first_divergence_block": first_block,
            "affected_blocks": affected_blocks,
            "action_taken": action,
        }
        if action == "abort":
            result["aborted"] = True
            log("  [MODEL-INTEGRATION] Aborted", RED)
        else:
            log(f"  [MODEL-INTEGRATION] Action: {action}", YELLOW)

        return {
            "model_integration_result": result,
            "pipeline_aborted": action == "abort",
        }


def route_after_model_integration(state: OrchestratorState) -> str:
    """Route after the model-integration node.

    Pass (or flag off) -> integration_dv. A parked failure resolved with
    'retry' re-runs the node; 'abort' ends the pipeline. 'revise_uarch' and
    'fix_rtl' both imply the operator patched disk, so re-run to confirm the fix.
    """
    result = state.get("model_integration_result") or {}
    if result.get("aborted"):
        return END
    # A SKIPPED-HONEST gate (goldenless run; rung2 defect 1) is non-blocking ->
    # proceed to integration_dv exactly like a pass. An ADVISORY BYPASS
    # (CORESMITH_DETERMINISTIC_BFM: the deterministic integration DV is the
    # authoritative contract check) is likewise non-blocking -> proceed.
    if result.get("skipped") or result.get("advisory_bypass"):
        return "integration_dv"
    action = result.get("action_taken", "")
    if action in ("retry", "revise_uarch", "fix_rtl"):
        return "model_integration"
    return "integration_dv"


route_after_model_integration.__edge_labels__ = {
    "model_integration": "Retry",
    "integration_dv": "DV",
    END: "DONE",
}


# ---------------------------------------------------------------------------
# Two-pass: uarch_integration_gate + begin_rtl_pass (CORESMITH_BLOCK_GOLDENS)
# ---------------------------------------------------------------------------
#
# The µARCH GATE runs BETWEEN the two fan-outs: after pass 1 (all blocks emit
# uArch spec + Amaranth block model) and BEFORE pass 2 (RTL+DV+synth). It relocates
# the model_integration_node logic (LLM stitch _chip_model.py + deterministic
# Amaranth verify + park-on-fail) to its shift-left position so a bad decomposition
# is caught in fast Python sims before any expensive RTL work.


def _affected_edge_for_block(pr: str, block_name: str) -> dict:
    """Best-effort: find a block-diagram edge touching ``block_name``.

    Returns ``{"from","to"}`` for the first connection that references the
    block, or ``{}``. Used to enrich a contract-gap interrupt so the outer
    agent can re-open the right interface definition.
    """
    if not block_name:
        return {}
    bd_path = Path(pr) / ".coresmith" / "block_diagram.json"
    if not bd_path.exists():
        return {}
    try:
        bd = json.loads(bd_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    for c in bd.get("connections", []) or []:
        src = c.get("from", c.get("from_block", ""))
        dst = c.get("to", c.get("to_block", ""))
        if block_name in (src, dst):
            return {"from": src, "to": dst}
    return {}


async def uarch_integration_gate_node(state: OrchestratorState) -> dict:
    """µARCH GATE (two-pass): LLM stitch chip model -> deterministic Amaranth verify.

    Runs after pass 1 (uArch specs + block models) and BEFORE pass 2 (RTL). It
    is the relocated ``model_integration_node`` logic plus section-D gap
    classification. On a clean gate it returns ``{"passed": True}`` and routing
    proceeds to ``begin_rtl_pass``. On divergence it PARKS an interrupt enriched
    with ``gap_class`` / ``affected_edge`` and (for contract gaps) a
    ``revise_contract`` action; ``pipeline_phase`` stays ``"uarch"`` so a retry
    re-enters the gate -- the flip to ``"rtl"`` only happens in begin_rtl_pass.
    """
    from orchestrator.architecture import model_integration as _model_integration

    pr = state.get("project_root", str(PROJECT_ROOT))

    write_graph_event(pr, "uArch Integration Gate", "graph_node_enter", {})

    # Composition-time honesty note for the deterministic-BFM flag. The
    # composition _chip_model.py pin driver is LLM-authored (it drives the Amaranth
    # block models over the same pins and can co-tune to their quirks). The
    # CONTRACT-ENFORCING deterministic BFM runs downstream at integration_dv on
    # the real chip_top RTL (the gate that matches the fixed external host); a
    # deterministic Amaranth-substrate composition driver is the follow-on.
    try:
        from orchestrator.langgraph import bfm_lib as _bfm_lib
        if _bfm_lib.deterministic_bfm_enabled() and _bfm_lib.arch_indicates_qspi_slave(pr):
            log(
                "  [µARCH-GATE] ADVISORY: CORESMITH_DETERMINISTIC_BFM=1 on a "
                "QSPI-slave design. The composition _chip_model.py pin driver is "
                "LLM-authored (co-tuning risk); the contract-enforcing "
                "deterministic BFM runs at integration_dv on the real chip_top.",
                YELLOW,
            )
    except Exception:  # noqa: BLE001
        pass

    with _tracer.start_as_current_span("uArch Integration Gate") as span:
        # (a) LLM model-integration: write _chip_model.py if missing.
        try:
            await _maybe_generate_chip_model(pr)
        except Exception as exc:  # noqa: BLE001
            log(f"  [µARCH-GATE] chip-model gen raised: {exc}", YELLOW)

        # (b) deterministic Amaranth-simulation verify. A-Fix 2a: fail-closed -- a
        # gate that RAISES is NOT a pass. CORESMITH_GATE_FAIL_OPEN tolerates it;
        # otherwise a synthesized gate_error violation falls through to the
        # existing failure/interrupt handling below.
        from orchestrator.langgraph.gate_guard import gate_error_violation, gate_guard
        _gr = await asyncio.to_thread(
            gate_guard, "uarch_integration",
            _model_integration.run_model_integration_gate, pr,
        )
        if _gr.errored and _gr.skipped:
            log(f"  [µARCH-GATE] gate raised; FAIL-OPEN escape active, treating "
                f"as pass: {_gr.reason}", YELLOW)
            span.set_attribute("error", _gr.error)
            write_graph_event(pr, "uArch Integration Gate", "graph_node_exit", {
                "skipped": True, "reason": "gate_error_fail_open",
            })
            return {"model_integration_result": {"passed": True, "error": _gr.reason}}
        if _gr.errored:
            log(f"  [µARCH-GATE] gate ERRORED (fail-closed, NOT a pass): "
                f"{_gr.reason}", RED)
            span.set_attribute("error", _gr.error)
            violations = [gate_error_violation(_gr.reason, _gr.error)]
        else:
            violations = _gr.value or []

        span.set_attribute("violation_count", len(violations))

        if not violations:
            # rung2 defect 1: an empty violation list is a real PASS only when
            # the gate ACTUALLY compared a composed chip model against a golden
            # reference. For a goldenless / requirements-only run (no golden,
            # no block models -> gate no-op) it must record SKIPPED-HONEST and
            # NOT report passed=True. Routing treats skipped like pass
            # (non-blocking) but the state/events/scoreboard say SKIPPED.
            _status = _model_integration.describe_gate_status(pr)
            if not _status["applicable"]:
                span.set_attribute("skipped", True)
                span.set_attribute("skip_reason", _status["reason"])
                log("  [µARCH-GATE] SKIPPED (not applicable, NOT a pass): "
                    f"{_status['reason']}", YELLOW)
                _record_dv_row(
                    pr, block="chip_model", scope="chip_model", source="gate",
                    passed=False, skipped=True, detail=_status["reason"],
                )
                write_graph_event(pr, "uArch Integration Gate",
                                  "graph_node_exit", {
                                      "skipped": True,
                                      "reason": _status["reason"],
                                      "gate_status": _status,
                                  })
                return {"model_integration_result": {
                    "skipped": True,
                    "reason": "no golden reference; gate not applicable",
                    "gate_status": _status,
                }}
            # Real PASS: clear the no-progress signature so (a) the next
            # failure is never mis-read as "no progress" against a stale
            # verdict and (b) gate_scoped_reuse_reason() stops scoping regen
            # (its "failure iteration in progress" signal is this file).
            try:
                (Path(pr) / ".coresmith" / "_last_gate_signature.txt").unlink(
                    missing_ok=True
                )
            except OSError:
                pass

            # Derate sign-off (microarch step 3): the functional/fidelity gate
            # PASSED, but if the fidelity tier recorded a within-budget derate
            # large enough to need chip-lead sign-off (above escalate_floor),
            # PARK for approval instead of silently shipping the derate. The
            # chip-lead OWNS the be-exact / derate / escalate decision.
            from orchestrator.architecture import fidelity as _fidelity
            esc = _fidelity.read_derate_escalation(pr)
            if esc is not None:
                log("  [µARCH-GATE] within-budget DERATE needs chip-lead sign-off",
                    YELLOW)
                response = interrupt({
                    "type": "derate_signoff",
                    "fidelity": esc,
                    "supported_actions": ["approve", "revise_uarch", "abort"],
                    "outer_agent_guidance": (
                        "The integrated chip model PASSED functionally but at a "
                        "DERATED fidelity that is WITHIN budget yet above the "
                        "escalate threshold (measured "
                        f"{esc.get('measured')} vs floor {esc.get('floor')} / "
                        f"escalate_floor {esc.get('escalate_floor')}, "
                        f"{esc.get('direction')}-is-better; derate "
                        f"{esc.get('derate_pct')}% vs ideal). As chip-lead you OWN "
                        "this trade: resume 'approve' to accept the derate and "
                        "proceed to the RTL pass (recorded in "
                        ".coresmith/derate_ledger.json), 'revise_uarch' to re-spec "
                        "the block(s) toward higher fidelity, or 'abort'."
                    ),
                })
                action = response.get("action", "approve")
                if action == "abort":
                    log("  [µARCH-GATE] Derate sign-off: ABORTED", RED)
                    write_graph_event(pr, "uArch Integration Gate",
                                      "graph_node_exit",
                                      {"passed": False, "derate_aborted": True})
                    return {
                        "model_integration_result": {
                            "passed": False, "aborted": True,
                            "derate_signoff": esc,
                        },
                        "pipeline_aborted": True,
                    }
                if action == "revise_uarch":
                    log("  [µARCH-GATE] Derate sign-off: chip-lead chose to re-spec "
                        "for higher fidelity", YELLOW)
                    out = {
                        "model_integration_result": {
                            "passed": False, "derate_revise": True,
                            "derate_signoff": esc,
                        },
                        "uarch_revise_attempts": (
                            int(state.get("uarch_revise_attempts", 0)) + 1
                        ),
                    }
                    if route_after_uarch_gate({**state, **out}) == "init_tier":
                        out["current_tier_index"] = 0
                    return out
                # approve -> record sign-off so we don't re-prompt, then proceed
                _fidelity.mark_derate_signed_off(pr)
                log("  [µARCH-GATE] Derate sign-off: APPROVED (within budget, "
                    "recorded)", GREEN)
            log("  [µARCH-GATE] PASSED (simulated chip model == reference)", GREEN)
            write_graph_event(pr, "uArch Integration Gate", "graph_node_exit", {
                "passed": True, "violation_count": 0,
                "derate_signed_off": esc is not None,
            })
            return {"model_integration_result": {
                "passed": True, "derate_signed_off": esc is not None}}

        # ADVISORY BYPASS (CORESMITH_DETERMINISTIC_BFM). This gate runs BEFORE
        # any RTL exists; the composition _chip_model.py pin driver is
        # LLM-authored and stimulus/DUT-fragile (it can mis-decode the IN-window
        # stimulus -> corrupt/all-zero composed output DESPITE byte-correct
        # per-block models), and every mismatch here triggers a full re-spec of
        # all blocks. When the deterministic integration DV is enabled it is the
        # AUTHORITATIVE RTL-level contract check downstream on the real chip_top,
        # so this model-level composition mismatch must NOT hard-block the run
        # (nor re-spec). Log LOUDLY, name the mismatch, and PROCEED to the RTL
        # pass. Flag off -> never taken (byte-identical to before).
        from orchestrator.langgraph import bfm_lib as _bfm_lib
        if _bfm_lib.deterministic_bfm_enabled():
            first = violations[0]
            first_block = first.get("first_divergence_block", "")
            log(f"\n{'='*60}", YELLOW)
            log("  µARCH INTEGRATION GATE MISMATCH -- ADVISORY (non-blocking)",
                YELLOW)
            log("  CORESMITH_DETERMINISTIC_BFM=1: the deterministic integration "
                "DV on the real chip_top RTL is the authoritative contract "
                "check. The LLM-authored composition _chip_model.py harness is "
                "stimulus/DUT-fragile; NOT hard-blocking and NOT re-speccing "
                "all blocks.", YELLOW)
            log(f"  First-divergence block: {first_block or '(unlocalized)'}",
                YELLOW)
            for v in violations[:5]:
                log(f"    - expected {v.get('expected')!r} got "
                    f"{v.get('observed')!r}", YELLOW)
            # Section 3b: record the quantified mismatch as a carried-forward
            # defect (specific unmodeled role named) instead of swallowing it.
            _defect = _advisory_composition_defect(pr, "uarch_integration", violations)
            record_carried_forward_defect(pr, _defect)
            if _defect.get("unmodeled"):
                log(f"  UNMODELED: {_defect['unmodeled']}", YELLOW)
            log(f"  PROCEEDING to RTL pass (advisory; carried forward).\n"
                f"{'='*60}\n", YELLOW)
            write_graph_event(pr, "uArch Integration Gate", "graph_node_exit", {
                "passed": False,
                "advisory_bypass": True,
                "violation_count": len(violations),
                "first_divergence_block": first_block,
                "carried_forward_defect": True,
                "unmodeled": _defect.get("unmodeled", ""),
            })
            return {"model_integration_result": {
                "passed": False,
                "advisory_bypass": True,
                "first_divergence_block": first_block,
                "violations": violations[:20],
                "carried_forward_defect": _defect,
            }}

        first = violations[0]
        first_block = first.get("first_divergence_block", "")
        # gap_class: contract gap wins if ANY violation is a contract gap (a
        # composition / throughput contract failure dominates per-block math).
        gap_class = "block_math"
        if any(v.get("gap_class") == "contract" for v in violations):
            gap_class = "contract"
        affected_edge = _affected_edge_for_block(pr, first_block)
        # Per-field localization: model_integration sets affected_blocks when only
        # SOME output fields diverged -> targeted re-spec (init_tier_node) instead
        # of broadcast to all blocks. The convergence fix for composition runs.
        affected_blocks: list[str] = []
        for v in violations:
            for b in v.get("affected_blocks") or []:
                if b and b not in affected_blocks:
                    affected_blocks.append(b)

        log(f"\n{'='*60}", RED)
        log("  µARCH INTEGRATION GATE FAILED", RED)
        log(f"  First-divergence block: {first_block or '(unlocalized)'}", RED)
        if affected_blocks:
            log(f"  TARGETED re-spec blocks (field-localized): {affected_blocks}",
                RED)
        else:
            log("  Localization: unlocalized -> broadcast re-spec to all blocks",
                RED)
        log(f"  gap_class: {gap_class}", RED)
        for v in violations[:5]:
            log(f"    - expected {v.get('expected')!r} got "
                f"{v.get('observed')!r}", RED)
        log(f"{'='*60}\n", RED)

        write_graph_event(pr, "uArch Integration Gate", "graph_node_exit", {
            "passed": False,
            "violation_count": len(violations),
            "first_divergence_block": first_block,
            "gap_class": gap_class,
        })

        supported_actions = [
            "revise_uarch",  # re-spec the offending block/tier (block_math)
            "fix_rtl",       # outer agent patched block/chip model on disk
            "retry",         # re-run the gate after an on-disk fix
            "abort",
        ]
        if gap_class == "contract":
            # Frozen interface contracts live in the (separate) architecture
            # graph; the frontend cannot mutate them. Offer revise_contract so
            # the outer agent re-runs the architecture interface specialist.
            supported_actions.insert(1, "revise_contract")

        # NO-PROGRESS GUARD. If this revise produced a BYTE-IDENTICAL composed
        # output to the previous attempt (same observed + gap_class), the
        # whole-chip re-fan is NOT converging -- steer the outer agent toward a
        # targeted single-block fix instead of burning more re-fans (the Opus
        # codec run re-fanned identical 50B output until it hit the quota).
        no_progress = False
        if os.environ.get("CORESMITH_GATE_NO_PROGRESS_GUARD", "1").strip() != "0":
            try:
                import hashlib
                sig = hashlib.sha1(
                    (repr(first.get("observed")) + "|" + gap_class)
                    .encode("utf-8", "replace")
                ).hexdigest()
                sig_path = Path(pr) / ".coresmith" / "_last_gate_signature.txt"
                prev = sig_path.read_text().strip() if sig_path.exists() else ""
                no_progress = bool(prev) and prev == sig
                sig_path.parent.mkdir(parents=True, exist_ok=True)
                sig_path.write_text(sig)
            except Exception:  # noqa: BLE001
                pass
        if no_progress:
            log("  [µARCH-GATE] NO PROGRESS -- composed output is BYTE-IDENTICAL "
                "to the previous attempt; re-fan is not converging", RED)

        payload = {
            "type": "model_integration_failure",
            "first_divergence_block": first_block,
            "gap_class": gap_class,
            "affected_edge": affected_edge,
            "affected_blocks": affected_blocks,
            "no_progress": no_progress,
            "violations": violations[:20],
            "suggested_fix": first.get("suggested_fix", ""),
            "supported_actions": supported_actions,
            "outer_agent_guidance": (
                "The µarch gate stitched every per-block Amaranth block model into "
                "a chip model and its simulated output diverged from the "
                "reference. gap_class="
                f"'{gap_class}'. block_math => the named block's transcribed "
                "math is wrong: resume 'revise_uarch' (re-specs the block/tier) "
                "or fix arch/block_models/<block>.py then 'retry'. contract => "
                "the per-block models are each self-consistent but their declared "
                "handshake/width contract cannot compose: this needs an "
                "architecture-level interface-definition change. Resume "
                "'revise_contract' (writes a structured request to "
                ".coresmith/interface_contract_revision_request.json and ends the "
                "frontend run so the architecture interface specialist can "
                f"re-open edge {affected_edge or '(see block_diagram.json)'})."
            ),
        }

        if no_progress:
            payload["outer_agent_guidance"] = (
                "NO PROGRESS -- the last revise produced a BYTE-IDENTICAL composed "
                "output, so re-fanning all blocks is NOT working. Do NOT "
                "revise_uarch again. Localize the SINGLE diverging block (trace the "
                "first-divergence byte offset through the serialization chain) and "
                "restart_block it, or fix arch/block_models/<block>.py directly "
                "then 'retry'. " + payload["outer_agent_guidance"]
            )

        response = interrupt(payload)
        action = response.get("action", "abort")

        result = {
            "passed": False,
            "violations": violations[:20],
            "first_divergence_block": first_block,
            "gap_class": gap_class,
            "affected_edge": affected_edge,
            "affected_blocks": affected_blocks,
            "action_taken": action,
        }
        if action == "abort":
            result["aborted"] = True
            log("  [µARCH-GATE] Aborted", RED)
        else:
            log(f"  [µARCH-GATE] Action: {action} (gap_class={gap_class})", YELLOW)

        out: dict = {
            "model_integration_result": result,
            "pipeline_aborted": action == "abort",
        }
        # Count each non-abort revise/retry so the in-loop re-spec is bounded
        # (CORESMITH_UARCH_REVISE_MAX). This lets a non-composing decomposition
        # iterate a few variance draws (block_math AND contract gaps) instead of
        # dead-ending after one, while guaranteeing termination.
        if action != "abort":
            out["uarch_revise_attempts"] = (
                int(state.get("uarch_revise_attempts", 0)) + 1
            )
        # When this gate routes back to init_tier for a block_math re-spec, the
        # tier index sits at len(tier_list) (exhausted -- that's how we reached
        # the gate). Reset it to 0 so the uarch pass re-fans-out from the first
        # tier; phase stays "uarch" (begin_rtl_pass is the sole phase flipper).
        # (The contract path resets the index in write_contract_request_node.)
        if route_after_uarch_gate({**state, **out}) == "init_tier":
            out["current_tier_index"] = 0
        return out


async def begin_rtl_pass_node(state: OrchestratorState) -> dict:
    """Transition pass 1 -> pass 2: reset tier index, flip phase to "rtl".

    The SOLE writer of ``pipeline_phase: "rtl"`` (R3). Runs only after a clean
    µarch gate. Writes ONLY index + phase + flag -- ``tier_list`` is untouched
    so ``init_tier_node``'s idempotent ``state.get("tier_list") or ...`` reuses
    the already-computed list (R8).
    """
    pr = state.get("project_root", str(PROJECT_ROOT))
    # run3-followups: the banner reflects the ACTUAL gate outcome -- a green
    # CLEAN was printed four lines after an advisory-dismissed model mismatch,
    # suppressing a true early detection for ~2.5 h of downstream work.
    from orchestrator.langgraph.pipeline_helpers import uarch_gate_banner
    _banner, _colour = uarch_gate_banner(state.get("model_integration_result"))
    log(f"\n{'='*60}", _colour)
    log(f"  {_banner}", _colour)
    log(f"{'='*60}", _colour)
    write_graph_event(pr, "Begin RTL Pass", "graph_node_exit", {
        "pipeline_phase": "rtl",
    })
    return {
        "current_tier_index": 0,
        "pipeline_phase": "rtl",
        "uarch_pass_done": True,
    }


def _uarch_revise_cap() -> int:
    """Max in-loop µarch-gate re-spec iterations before giving up (END with a
    contract-revision marker for the outer agent). CORESMITH_UARCH_REVISE_MAX,
    default 4."""
    try:
        return int(os.environ.get("CORESMITH_UARCH_REVISE_MAX", "4") or 4)
    except ValueError:
        return 4


def _regen_stale_chip_model() -> bool:
    """Regenerate the composed _chip_model.py when block models changed after it
    (so revise_uarch actually re-composes instead of re-checking a stale model).
    Default ON; CORESMITH_REGEN_STALE_CHIP_MODEL=0 restores the old always-reuse."""
    return os.environ.get(
        "CORESMITH_REGEN_STALE_CHIP_MODEL", "1"
    ).strip().lower() not in {"0", "false", "no", "off"}


def _chip_model_needs_regen(models_dir, chip_model_path) -> bool:
    """True if the composed ``_chip_model.py`` must be (re)generated: it is
    missing, OR (stale-regen enabled) any block model is newer than it. This is
    what lets ``revise_uarch`` actually re-compose after it regenerates the
    per-block models — otherwise the gate re-checks a stale chip model and emits
    bit-identical output every revise, so the bounded revise loop never
    converges. With ``CORESMITH_REGEN_STALE_CHIP_MODEL=0`` only a missing chip
    model triggers regen (old behavior)."""
    from pathlib import Path as _P
    chip_model_path = _P(chip_model_path)
    models_dir = _P(models_dir)
    if not chip_model_path.exists():
        return True
    if not _regen_stale_chip_model():
        return False
    try:
        chip_mtime = chip_model_path.stat().st_mtime
        newest_block = max(
            (p.stat().st_mtime for p in models_dir.glob("*.py")
             if p.name != "_chip_model.py"),
            default=0.0,
        )
    except OSError:
        return False
    return newest_block > chip_mtime


def _revalidate_enabled() -> bool:
    """Recoverable incomplete-gate: on a ``retry`` resume, re-validate the
    failed/missing blocks (re-run their DV against the outer controller's
    on-disk RTL fixes) and recount, instead of dead-ending the graph. Default
    ON; set ``CORESMITH_REVALIDATE_INCOMPLETE=0`` to restore the old behavior
    where ``retry`` aborts and the outer controller must relaunch."""
    return os.environ.get(
        "CORESMITH_REVALIDATE_INCOMPLETE", "1"
    ).strip().lower() not in {"0", "false", "no", "off"}


def _revalidate_max() -> int:
    """Max bounded re-validation passes at the incomplete gate before aborting,
    so a perpetually-failing block cannot loop forever. CORESMITH_REVALIDATE_MAX,
    default 2."""
    try:
        return max(0, int(os.environ.get("CORESMITH_REVALIDATE_MAX", "2") or 2))
    except ValueError:
        return 2


# --------------------------------------------------------------------------- #
# engine-v31 step 3: apply a high-confidence uArch-patch diagnosis on retry
# --------------------------------------------------------------------------- #
# On the pipeline_incomplete RETRY path the failed blocks re-validate against
# their ON-DISK RTL (no LLM regen there). When the diagnose lead's verdict is a
# structured µarch revision (``diagnosis.uarch_patch``: original->replacement
# spec edits) at high confidence, re-validating the unchanged RTL just re-fails
# and the block escalates -- the revision is never applied. This routes such a
# high-confidence uarch_patch to a BLOCK RE-SPEC + REGEN: it edits the block's
# uArch spec in place, writes the prescription to ``gate_feedback.txt`` (the
# existing re-spec channel init_tier/generate_uarch_spec reads), and drops a
# ``force_regen`` marker so the re-validate pass regenerates the RTL from the
# revised spec instead of re-checking the stale file. Bounded to ONE auto-apply
# per block (a persistent marker); a second failure escalates as before.
def _uarch_patch_on_retry_enabled() -> bool:
    """Auto-apply a high-confidence diagnose ``uarch_patch`` on the retry path
    (default ON). ``CORESMITH_UARCH_PATCH_ON_RETRY=0`` restores escalate-only."""
    return os.environ.get(
        "CORESMITH_UARCH_PATCH_ON_RETRY", "1"
    ).strip().lower() not in {"0", "false", "no", "off"}


def _uarch_patch_min_confidence() -> float:
    """Min diagnose confidence to auto-apply a uarch_patch on retry (default
    0.9). ``CORESMITH_UARCH_PATCH_CONFIDENCE``."""
    try:
        return float(os.environ.get("CORESMITH_UARCH_PATCH_CONFIDENCE", "0.9"))
    except ValueError:
        return 0.9


def apply_uarch_patch_to_spec(spec_text: str, uarch_patch: dict) -> tuple[str, int]:
    """Apply a diagnosis ``uarch_patch`` (``sections_to_replace``: list of
    ``{original, replacement}``) to a uArch-spec markdown string.

    Each section's ``original`` text is replaced with its ``replacement`` (first
    occurrence). A section whose ``original`` is not found verbatim is skipped
    (the LLM may have paraphrased) -- so the apply is best-effort and idempotent
    on already-patched text. Returns ``(new_text, n_applied)``.
    """
    if not isinstance(uarch_patch, dict):
        return spec_text, 0
    sections = uarch_patch.get("sections_to_replace") or []
    out = spec_text
    n = 0
    for sec in sections:
        if not isinstance(sec, dict):
            continue
        orig = sec.get("original")
        repl = sec.get("replacement")
        if not isinstance(orig, str) or not isinstance(repl, str) or not orig:
            continue
        if orig in out and repl not in out:
            out = out.replace(orig, repl, 1)
            n += 1
    return out, n


def _route_uarch_patch_on_retry(project_root: str, block_names: list[str]) -> list[str]:
    """For each failed/missing block, auto-apply a high-confidence diagnose
    ``uarch_patch`` (edit the spec + write gate_feedback + force regen), bounded
    to ONE application per block. Returns the list of blocks patched. Never
    raises -- a plumbing error on one block simply skips it (falls back to the
    existing escalate-only behavior)."""
    if not _uarch_patch_on_retry_enabled():
        return []
    min_conf = _uarch_patch_min_confidence()
    patched: list[str] = []
    for name in block_names:
        try:
            block_dir = Path(project_root) / ".coresmith" / "blocks" / name
            diag_path = block_dir / "diagnosis.json"
            if not diag_path.exists():
                continue
            diag = json.loads(diag_path.read_text())
            uarch_patch = diag.get("uarch_patch")
            if not isinstance(uarch_patch, dict) or not uarch_patch.get(
                    "sections_to_replace"):
                continue
            try:
                conf = float(diag.get("confidence", 0) or 0)
            except (TypeError, ValueError):
                conf = 0.0
            if conf < min_conf:
                continue
            # Bounded: one auto-application per block, then escalate as before.
            marker = block_dir / "uarch_patch_applied"
            if marker.exists():
                log(f"  [UARCH-PATCH] {name}: high-confidence uarch_patch already "
                    f"auto-applied once -- escalating (bounded)", YELLOW)
                continue
            spec_path = Path(project_root) / "arch" / "uarch_specs" / f"{name}.md"
            if not spec_path.exists():
                continue
            new_text, n_applied = apply_uarch_patch_to_spec(
                spec_path.read_text(encoding="utf-8", errors="replace"), uarch_patch)
            if n_applied <= 0:
                log(f"  [UARCH-PATCH] {name}: uarch_patch sections did not match "
                    f"the current spec verbatim -- not applied", YELLOW)
                continue
            spec_path.write_text(new_text, encoding="utf-8")
            # Route the prescription through the existing gate-feedback re-spec
            # channel so the re-validate pass RE-SPECS + REGENERATES the block.
            fb_parts = [
                "MICROARCH REVISION (auto-applied from a high-confidence "
                f"diagnose uarch_patch, confidence {conf:g}):",
                (uarch_patch.get("rationale") or "").strip(),
                (diag.get("suggested_fix") or "").strip(),
                "The uArch spec above has been revised accordingly; regenerate "
                "the RTL to implement the revised microarchitecture.",
            ]
            (block_dir / "gate_feedback.txt").write_text(
                "\n\n".join(p for p in fb_parts if p), encoding="utf-8")
            # Force regen: the RTL bytes are unchanged (we edited the SPEC), so
            # the skip-regen fast path would reuse the now-stale RTL against the
            # revised µarch. Invalidate the recorded sim-pass so the re-validate
            # pass REGENERATES the RTL from the revised spec.
            best_path = block_dir / "best_result.json"
            try:
                if best_path.exists():
                    _best = json.loads(best_path.read_text())
                    if isinstance(_best, dict):
                        _best["sim_passed"] = False
                        _best["uarch_patch_invalidated"] = True
                        best_path.write_text(json.dumps(_best, indent=2))
            except (json.JSONDecodeError, OSError):
                pass
            marker.write_text(f"confidence={conf:g}; sections_applied={n_applied}\n")
            patched.append(name)
            log(f"  [UARCH-PATCH] {name}: auto-applied uarch_patch "
                f"({n_applied} spec section(s), confidence {conf:g}) -> re-spec + "
                f"regen on this retry pass (bounded to 1)", GREEN)
        except Exception as e:  # noqa: BLE001 - never fail the gate on one block
            log(f"  [UARCH-PATCH] {name}: skipped ({e})", YELLOW)
            continue
    return patched


def _pipeline_complete_route(state: OrchestratorState) -> str:
    """Decision for the edge after ``pipeline_complete`` (returns a plain key so
    it is unit-testable; the graph maps ``"end"`` -> ``END``):
      - ``pipeline_aborted`` -> ``"end"``
      - ``revalidate_pending`` (a bounded incomplete-gate retry) -> ``"init_tier"``
        (re-run the rtl-phase tiers; failed blocks re-validate their fixed RTL,
        passers reuse via skip-regen) then back to ``pipeline_complete`` to recount
      - otherwise (clean) -> ``"integration_check"``.
    """
    if state.get("pipeline_aborted"):
        return "end"
    if state.get("revalidate_pending"):
        return "init_tier"
    return "integration_check"


def route_after_uarch_gate(state: OrchestratorState) -> str:
    """Route after the µarch gate.

    Clean gate -> ``begin_rtl_pass`` (start pass 2). A parked failure:
      - ``abort`` -> END.
      - ``contract`` gap with ``revise_contract`` / ``revise_uarch`` -> END
        (the frontend cannot mutate frozen contracts; a request marker is
        written by the writer node before END).
      - ``block_math`` gap with ``revise_uarch`` (or ``retry`` / ``fix_rtl``)
        -> ``init_tier`` to re-spec the offending block/tier in phase "uarch".
    """
    result = state.get("model_integration_result") or {}
    # A SKIPPED-HONEST gate (goldenless / requirements-only run; rung2 defect 1)
    # is non-blocking -- route it exactly like a pass into the RTL pass. An
    # ADVISORY BYPASS (CORESMITH_DETERMINISTIC_BFM: the deterministic
    # integration DV on the real chip_top is the authoritative contract check)
    # is likewise non-blocking -- proceed to the RTL pass rather than re-spec.
    if (not result or result.get("passed") or result.get("skipped")
            or result.get("advisory_bypass")):
        return "begin_rtl_pass"
    if result.get("aborted"):
        return END
    action = result.get("action_taken", "")
    gap_class = result.get("gap_class", "block_math")
    if action not in ("retry", "revise_uarch", "fix_rtl", "revise_contract"):
        # Unknown action: fail safe to END rather than silently proceeding.
        return END
    # Bound the in-loop re-spec. Over the cap, route through
    # write_contract_request so it writes the machine-readable revision marker
    # for the outer agent, then ENDs (it detects the cap and aborts). Under the
    # cap, contract gaps go through write_contract_request (marker + re-spec) and
    # block_math gaps re-spec directly via init_tier.
    cap = _uarch_revise_cap()
    attempts = int(state.get("uarch_revise_attempts", 0))
    if attempts > cap:
        return "write_contract_request"
    if gap_class == "contract":
        return "write_contract_request"
    return "init_tier"


route_after_uarch_gate.__edge_labels__ = {
    "begin_rtl_pass": "GATE CLEAN",
    "init_tier": "RE-SPEC (block_math)",
    "write_contract_request": "CONTRACT GAP",
    END: "ABORT",
}


async def write_contract_request_node(state: OrchestratorState) -> dict:
    """Write a structured interface-contract revision request.

    Section D: a ``contract`` gap means the per-block models are each
    self-consistent but their declared handshake/width contract cannot compose.
    The frontend graph cannot mutate frozen contracts (architecture is a
    separate graph), so we surface a machine-readable request for the outer
    agent to re-run the architecture interface-definition specialist.

    To avoid the historical dead-end (write marker -> END after a single gate
    failure), this node is now part of a *bounded re-spec loop*. The marker is
    always written for forensics / the outer agent. Then:
      - Under the revise cap: loop back to ``init_tier`` (reset tier index) so
        the µarch pass re-fans-out and draws a fresh decomposition; a variance
        draw may compose where the prior one couldn't.
      - At/over the cap: set ``pipeline_aborted`` so we END, handing off the
        marker to the outer agent for architecture-level interface revision.
    """
    pr = state.get("project_root", str(PROJECT_ROOT))
    result = state.get("model_integration_result") or {}
    cap = _uarch_revise_cap()
    attempts = int(state.get("uarch_revise_attempts", 0))
    exhausted = attempts > cap
    gap_class = result.get("gap_class", "contract")
    if gap_class == "contract":
        reason = "uarch_integration_gate contract gap (models cannot compose)"
    else:
        reason = (
            f"uarch_integration_gate {gap_class} gap unresolved after "
            f"{attempts} revise attempt(s)"
        )
    request = {
        "type": "interface_contract_revision_request",
        "gap_class": gap_class,
        "reason": reason,
        "first_divergence_block": result.get("first_divergence_block", ""),
        "affected_edge": result.get("affected_edge", {}),
        "violations": result.get("violations", [])[:20],
        "revise_attempts": attempts,
        "revise_cap": cap,
        "exhausted": exhausted,
        "suggested_action": (
            "Re-run the architecture interface-definition specialist for the "
            "affected edge; the per-block math is self-consistent but the "
            "declared handshake/width contract cannot compose."
        ),
    }
    try:
        out = Path(pr) / ".coresmith" / "interface_contract_revision_request.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(request, indent=2), encoding="utf-8")
        log(f"  [µARCH-GATE] Wrote contract-revision request to {out}", YELLOW)
    except OSError as exc:
        log(f"  [µARCH-GATE] Could not write contract-revision request: {exc}",
            RED)
    write_graph_event(pr, "Write Contract Request", "graph_node_exit", {
        "affected_edge": request["affected_edge"],
        "revise_attempts": attempts,
        "exhausted": exhausted,
    })
    if exhausted:
        log(f"  [µARCH-GATE] Contract gap persists after {attempts} revise "
            f"attempt(s) (cap={cap}); ending for outer-agent interface revision.",
            RED)
        return {"pipeline_aborted": True}
    log(f"  [µARCH-GATE] Contract gap: re-fanning-out µarch pass "
        f"(attempt {attempts}/{cap}) to draw a composing decomposition.",
        YELLOW)
    # Re-spec: reset the tier index so init_tier re-fans-out the uarch pass.
    # phase stays "uarch" (begin_rtl_pass is the sole phase flipper).
    return {"current_tier_index": 0}


def route_after_write_contract_request(state: OrchestratorState) -> str:
    """END once the contract gap is exhausted; otherwise re-spec via init_tier."""
    if state.get("pipeline_aborted"):
        return END
    return "init_tier"


route_after_write_contract_request.__edge_labels__ = {
    "init_tier": "RE-SPEC (contract)",
    END: "EXHAUSTED -> outer agent",
}


def _format_dv_retry_context(previous_result: dict | None) -> str:
    """Format prior top-level DV failure/audit context for retry prompts."""
    if not previous_result:
        return ""
    audit = previous_result.get("contract_audit") or {}
    if not isinstance(audit, dict) or not audit:
        bits = []
        if previous_result.get("error"):
            bits.append(f"Previous error: {previous_result.get('error')}")
        if previous_result.get("sim_log_path"):
            bits.append(f"Previous sim log: {previous_result.get('sim_log_path')}")
        return "\n".join(bits)

    first = audit.get("first_divergence") or {}
    if not isinstance(first, dict):
        first = {"summary": str(first)}
    evidence = audit.get("evidence") or []
    if isinstance(evidence, list):
        evidence_text = "\n".join(f"- {item}" for item in evidence[:8])
    else:
        evidence_text = str(evidence)

    parts = [
        f"Stage: {audit.get('stage', previous_result.get('phase', 'unknown'))}",
        f"Category: {audit.get('category', 'UNKNOWN')}",
        f"Recommended action: {audit.get('recommended_action', '')}",
        f"Contract failure: {audit.get('contract_failure', False)}",
        f"First divergence: {first.get('summary', '')}",
        f"RTL observation: {first.get('rtl_observation', '')}",
        f"Golden/expected observation: {first.get('golden_observation', '')}",
        f"Suggested fix: {audit.get('suggested_fix', '')}",
        f"Outer-agent summary: {audit.get('outer_agent_summary', '')}",
        f"Audit path: {previous_result.get('contract_audit_path', audit.get('audit_path', ''))}",
        f"Failure context path: {audit.get('context_path', '')}",
        f"Sim log path: {previous_result.get('sim_log_path', '')}",
    ]
    if evidence_text:
        parts.append("Evidence:\n" + evidence_text)
    # A retry prompt quoting a verdict about a DIFFERENT failure is worse than
    # quoting nothing: it reads as a confident diagnosis and sends the fix at
    # the wrong thing. Label it, first line, before anything it says.
    _stale = contract_audit_staleness(audit, audit.get("context_path", ""))
    if _stale:
        parts.insert(0, "!! " + _stale)
    return smart_truncate("\n".join(p for p in parts if p), 12000, "head_tail")


# ---------------------------------------------------------------------------
# Node: integration_dv  (Lead DV -- generates + runs integration testbench)
# ---------------------------------------------------------------------------

def _chip_equiv_enabled() -> bool:
    """A-Fix 5(d): after a green integration sim, re-drive chip_top + the
    composed Amaranth chip model on the same seeded stimulus and assert byte
    equivalence. Default ON (only active when block-goldens is on and a
    _chip_model.py exists). Set ``CORESMITH_CHIP_EQUIV=0`` to disable."""
    return (os.environ.get("CORESMITH_CHIP_EQUIV", "1") or "1") != "0"


def _resolve_equiv_seed() -> int:
    """Fresh seed for the chip-top equivalence stimulus (pinned via
    ``CORESMITH_DV_SEED_PIN`` for reproducible debugging)."""
    from orchestrator.harness.seed_provider import gate_seed
    return gate_seed()


def _maybe_run_chip_equiv(
    pr: str, design_name: str, top_rtl_path: str, block_rtl_paths: dict,
) -> dict | None:
    """Run the chip-top RTL-vs-model equivalence gate when applicable.

    Returns ``None`` when the gate does not apply (disabled, block-goldens off,
    no chip model, RTL-model-equiv globally off). Otherwise returns the equiv
    result dict. A harness-error skip is retried once at 2x timeout (A-Fix 2c);
    a persistent harness error is turned into a fail-closed result (so the DV is
    flipped to failed) unless ``CORESMITH_GATE_FAIL_OPEN`` is set. An honest skip
    (non-AXIS top / no verilator / non-comparable model output) is returned
    as-is and keeps the sim pass.
    """
    if not _chip_equiv_enabled():
        return None
    try:
        from orchestrator.architecture import composition as _composition
        if not _composition.block_goldens_enabled():
            return None
        chip_model = (
            Path(pr) / "arch" / _composition.BLOCK_MODELS_DIRNAME / "_chip_model.py"
        )
        if not chip_model.exists():
            return None
    except Exception:  # noqa: BLE001
        return None

    from orchestrator.langgraph.gate_guard import gate_fail_open_enabled, gate_guard
    from orchestrator.langgraph.rtl_model_equiv import (
        check_chip_model_equivalence,
        rtl_model_equiv_enabled,
    )
    if not rtl_model_equiv_enabled():
        return None

    seed = _resolve_equiv_seed()

    # rung3-fixes-2 (c): when the design declares dimensional maxima, size the
    # seeded RTL-vs-model stream long enough to reach the max index magnitude
    # (so a geometry-dependent index/address-width truncation diverges here too)
    # instead of the token default. Bounded to keep the sim tractable. No dims
    # declared -> default 64 (byte-identical to before).
    n_vectors = 64
    try:
        _dims = _declared_dimensions(pr)
        if _dims:
            n_vectors = max(64, min(_MAXGEO_EQUIV_NVEC_CAP, max(_dims.values())))
    except Exception:  # noqa: BLE001
        n_vectors = 64

    def _run(scale: float):
        return gate_guard(
            "chip_model_equiv",
            check_chip_model_equivalence,
            design_name, top_rtl_path, block_rtl_paths, str(chip_model),
            project_root=pr, seed=seed, n_vectors=n_vectors, timeout_scale=scale,
        )

    gr = _run(1.0)
    if gr.errored:
        if gate_fail_open_enabled():
            return {"passed": True, "skipped": True,
                    "reason": f"chip equiv gate errored (fail-open): {gr.reason}"}
        return {"passed": False, "skipped": False,
                "reason": f"chip equiv gate ERRORED (fail-closed): {gr.reason}"}
    eq = gr.value or {}
    # Harness-error skip -> retry once at 2x, then fail closed (A-Fix 2c).
    if eq.get("skipped") and eq.get("harness_error"):
        gr2 = _run(2.0)
        if gr2.errored:
            if gate_fail_open_enabled():
                return {"passed": True, "skipped": True,
                        "reason": f"chip equiv retry errored (fail-open): {gr2.reason}"}
            return {"passed": False, "skipped": False,
                    "reason": f"chip equiv retry ERRORED (fail-closed): {gr2.reason}"}
        eq2 = gr2.value or {}
        if eq2.get("skipped") and eq2.get("harness_error"):
            if gate_fail_open_enabled():
                return {"passed": True, "skipped": True,
                        "reason": f"chip equiv harness error (fail-open): {eq2.get('reason','')}"}
            return {"passed": False, "skipped": False,
                    "reason": ("chip equiv harness error persisted after retry "
                               f"(fail-closed): {eq2.get('reason', '')}")}
        return eq2
    return eq


# ---------------------------------------------------------------------------
# MAX-GEOMETRY DV stimulus gate (rung3-fixes-2, defect 1)
# ---------------------------------------------------------------------------
# A "verified" chip once shipped a truncated index-width bug because EVERY DV
# stimulus was a tiny fixed geometry (e.g. 16x16): the geometry-dependent
# index/address/counter widths were never exercised at the declared maximum, so
# a wrap at the 2^n boundary BELOW the max (e.g. a 7-bit column index wrapping
# at 512 on a 640-wide frame) sailed through integration + validation DV. This
# gate forces at least one MAX-GEOMETRY test case whenever the design declares
# any dimensional maximum, and requires the generated testbench to advertise it
# with a machine-checkable marker.
#
# DOMAIN-GENERIC by construction: dimension NAMES are DATA read from the
# design's own machine-readable declarations -- the engine NEVER greps for
# video/codec vocabulary (no "width"/"height"/"frame" keywords in this logic). A
# declared dimension is any object in the spec sources carrying a name-role
# string AND an integer extent-role field. The marker contract is
# ``# MAXGEO: <dim_name>=<value>`` for every declared dimension; the
# deterministic check matches on the declared max VALUE (the name is free-form
# data). Testing at the declared maximum inherently crosses every 2^n index
# boundary below it -- exactly where a truncated width wraps.

_MAXGEO_MARKER_RE = re.compile(r"#\s*MAXGEO\b(.*)", re.IGNORECASE)
_MAXGEO_PAIR_RE = re.compile(r"([A-Za-z_][\w./\-]*)\s*=\s*(\d+)")
# name-role / extent-role keys denote VALUE-SEMANTICS (identity, maximum), NOT a
# problem domain: the dimension's own name is opaque data harvested from them.
_DIM_NAME_KEYS = ("name", "parameter", "param", "dimension", "dim", "field", "id")
_DIM_EXTENT_KEYS = (
    "max", "maximum", "max_value", "maxval", "max_len", "max_length",
    "max_size", "max_depth", "max_count", "max_burst", "max_burst_len",
    "depth", "range", "capacity", "length", "size", "count",
)
_DIM_MIN_EXTENT = 8              # skip trivially-tiny dims (a 2-deep handshake)
_MAXGEO_EQUIV_NVEC_CAP = 4096    # bound the seeded chip-equiv stream length


def _maxgeo_gate_enabled() -> bool:
    """Require a MAX-GEOMETRY DV test when the design declares dimensional
    maxima. Default ON; ``CORESMITH_MAXGEO_GATE=0`` disables (both branches
    tested). Env-gate convention (like :func:`_chip_equiv_enabled`)."""
    return (os.environ.get("CORESMITH_MAXGEO_GATE", "1") or "1") != "0"


def _coerce_pos_int(v) -> int | None:
    """Positive-int coercion for extent fields (rejects bools / non-ints)."""
    try:
        if isinstance(v, bool):
            return None
        if isinstance(v, int):
            return v if v > 0 else None
        if isinstance(v, float):
            return int(v) if v > 0 and float(v).is_integer() else None
        if isinstance(v, str):
            s = v.strip().replace("_", "")
            if re.fullmatch(r"\d+", s):
                n = int(s)
                return n if n > 0 else None
    except (TypeError, ValueError):
        return None
    return None


def _collect_declared_dims(obj, out: dict) -> None:
    """Recursively harvest ``{name-role: str, extent-role: int}`` dimension
    declarations from an arbitrary JSON-ish structure. Name-agnostic."""
    if isinstance(obj, dict):
        name = None
        for nk in _DIM_NAME_KEYS:
            val = obj.get(nk)
            if isinstance(val, str) and val.strip():
                name = val.strip()
                break
        if name is not None:
            best = None
            for ek in _DIM_EXTENT_KEYS:
                if ek in obj:
                    n = _coerce_pos_int(obj.get(ek))
                    if n is not None and n >= _DIM_MIN_EXTENT:
                        best = n if best is None else max(best, n)
            if best is not None:
                out[name] = max(best, out.get(name, 0))
        for v in obj.values():
            _collect_declared_dims(v, out)
    elif isinstance(obj, list):
        for item in obj:
            _collect_declared_dims(item, out)


def _declared_dimensions(project_root: str) -> dict:
    """Machine-readable dimensional maxima the design declares, as
    ``{dim_name: max_value}``.

    Primary source (param-schema-1): the typed ERS ``parameters`` block
    (``{name, role, max, boundary_values}``) -- the authoritative, deterministic
    declaration. Every extent-bearing (dimension/range) parameter contributes
    its ``max``. FALLBACK (preserved, unchanged): the generic ``{name, max}``
    harvest over the ERS/PRD/block-spec JSON + FRD FUNC vectors, so a legacy
    prose ERS (no parameters block) behaves exactly as before -> ``{}`` -> the
    gate no-ops. Returns ``{}`` when nothing dimensional is declared. NEVER
    raises."""
    dims: dict = {}
    root = Path(project_root)
    # Primary: the typed ERS parameters block (authoritative structured source).
    try:
        from orchestrator.architecture import param_schema as _psch
        for name, mx in _psch.declared_maxima(
                _psch.parameters_from_ers(project_root)).items():
            dims[name] = max(int(mx), dims.get(name, 0))
    except Exception:  # noqa: BLE001
        pass
    try:
        for fname in ("ers_spec.json", "prd_spec.json", "block_specs.json",
                      "block_queue.json"):
            p = root / ".coresmith" / fname
            if p.exists():
                try:
                    _collect_declared_dims(
                        json.loads(p.read_text(encoding="utf-8")), dims)
                except (OSError, json.JSONDecodeError):
                    pass
        frd = root / "arch" / "frd_spec.md"
        if frd.exists():
            try:
                from orchestrator.architecture.composition import (
                    parse_func_vectors,
                )
                _collect_declared_dims(
                    parse_func_vectors(frd.read_text(encoding="utf-8")), dims)
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001
        return dims
    return dims


def _ers_parameter_table(project_root: str) -> str:
    """Verbatim rendering of the typed ERS ``parameters`` block for the DV
    testbench generators (param-schema-1). Empty string when the design
    declares no parameters (so the DV prompts stay byte-identical to before the
    schema landed). NEVER raises."""
    try:
        from orchestrator.architecture import param_schema as _psch
        return _psch.format_parameter_table(_psch.parameters_from_ers(project_root))
    except Exception:  # noqa: BLE001
        return ""


def _tb_maxgeo_pairs(tb_path: str) -> dict:
    """Parse the ``# MAXGEO: <name>=<value> ...`` marker(s) from a generated
    testbench into ``{name: value}``. Empty when no marker."""
    pairs: dict = {}
    try:
        text = Path(tb_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return pairs
    for line in text.splitlines():
        m = _MAXGEO_MARKER_RE.search(line)
        if not m:
            continue
        for name, val in _MAXGEO_PAIR_RE.findall(m.group(1)):
            try:
                pairs[name] = int(val)
            except ValueError:
                pass
    return pairs


_MAXGEO_CASE_RE = re.compile(r"#\s*MAXGEO_CASE:\s*(.+)$")


def _tb_maxgeo_case(tb_path: str) -> dict:
    """Parse the ``# MAXGEO_CASE: name=<s> cfg0=<int> in_bytes=<int>
    out_bytes=<int>`` marker the deterministic codegen emits for its
    maximum-configuration functional test. ``{}`` when absent."""
    out: dict = {}
    try:
        text = Path(tb_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return out
    for line in text.splitlines():
        m = _MAXGEO_CASE_RE.search(line)
        if not m:
            continue
        for tok in m.group(1).split():
            if "=" not in tok:
                continue
            k, _, v = tok.partition("=")
            try:
                out[k] = int(v)
            except ValueError:
                out[k] = v
    return out


def _maxgeo_conformance_scope_enabled() -> bool:
    """Scope the MAX-GEOMETRY demand for the ENGINE'S OWN compute-lane-independent
    conformance testbench. Default ON; ``CORESMITH_MAXGEO_CONFORMANCE_SCOPE=0``
    restores the unscoped demand (both branches tested)."""
    return (
        os.environ.get("CORESMITH_MAXGEO_CONFORMANCE_SCOPE", "1") or "1"
    ) != "0"


def _maxgeo_conformance_scope(
    project_root: str, tb_path: str, tb_result: dict | None,
    dims: dict, marker: dict, missing: dict,
) -> dict | None:
    """Scoped verdict for the DETERMINISTIC QSPI conformance testbench, or None.

    ``None`` means "this relaxation does not apply" -- the caller then issues the
    normal, unchanged hard failure. This is deliberately narrow:

      * It keys off ``tb_result`` flags that ONLY the engine's own bfm_lib writer
        sets (``deterministic_bfm`` + ``conformance_only``). An LLM-authored
        testbench cannot reach it, no matter what it writes into its own text --
        which is the whole reason the discriminator is the caller's record and
        not a comment in the file.
      * It applies only when the testbench ALSO declares its scope in the
        artifact (``# MAXGEO_SCOPE:``), so a hand-edited TB that dropped the
        coverage cannot inherit the relaxation.
      * It keeps TEETH: the expected bus coverage is recomputed HERE, from the
        bus contract the architecture produced, and every bus dimension that
        contract could drive must actually appear in the marker. A codegen
        regression that silently stops driving the max-length read burst fails
        the gate exactly as before.

    What it relaxes is only this: a compute-lane dimension (frame_width,
    record counts, coordinate extents, ...) cannot be driven by a testbench that has no compute
    oracle, so demanding it makes the gate a permanent brick wall for that whole
    class of run rather than a defect detector. Those dims are reported, logged
    loudly, and carried forward as a defect -- never silently dropped.
    """
    if not _maxgeo_conformance_scope_enabled():
        return None
    tbr = tb_result or {}
    if not (tbr.get("deterministic_bfm") and tbr.get("conformance_only")):
        return None
    contract_dict = tbr.get("contract") or {}
    if not contract_dict:
        return None
    try:
        text = Path(tb_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    if "# MAXGEO_SCOPE:" not in text:
        return None
    try:
        from orchestrator.langgraph import bfm_lib as _bfm
        contract = _bfm.QSPIContract.from_dict(contract_dict)
        cov = _bfm.bus_maxgeo_coverage(contract, dims)
    except Exception:  # noqa: BLE001 - an unusable contract is not a relaxation
        return None
    # TEETH: every bus dim this contract COULD drive must be marked, by name AND
    # value. Recomputed here, not read from the testbench's own claims.
    skipped = {n: v for n, v in cov.covered.items() if marker.get(n) != v}
    if skipped:
        return {
            "advisory": False,
            "reason": (
                "MAX-GEOMETRY DV GATE FAILED (deterministic QSPI conformance "
                "testbench): the bus contract can drive these declared maxima "
                "and the testbench did not mark them. This is a generator "
                "regression, not an unmodeled compute lane.\n"
                f"  bus-drivable    : {cov.covered}\n"
                f"  marker pairs    : {marker or '(no # MAXGEO marker found)'}\n"
                f"  skipped by TB   : {skipped}\n"
            ),
            "declared_dims": dims, "marker_pairs": marker,
            "uncovered_dims": missing, "bus_skipped": skipped,
        }
    uncovered = {n: v for n, v in dims.items() if n not in cov.covered}
    return {
        "advisory": True,
        "reason": (
            "MAX-GEOMETRY DV GATE SCOPED (not a clean pass): this run's "
            "integration DV is the DETERMINISTIC QSPI-slave conformance "
            "testbench, which has NO compute oracle -- the exercise's compute "
            "lane is unmodeled. It drove every BUS maximum the design declares "
            "at full extent, and it CANNOT drive compute-lane geometry at all. "
            "The gate therefore demands the bus subset and RECORDS the "
            "remainder as uncovered instead of failing a run that no testbench "
            "of this kind could ever pass.\n"
            f"  declared maxima : {dims}\n"
            f"  driven at max   : {cov.covered}\n"
            f"  NOT COVERED     : {uncovered}\n"
            "The uncovered dimensions are compute-lane geometry: a truncated "
            "index width behind them would NOT be caught by this DV. Model the "
            "compute lane (a golden host-flow plan) to close them, or set "
            "CORESMITH_MAXGEO_CONFORMANCE_SCOPE=0 to demand them anyway."
        ),
        "declared_dims": dims, "marker_pairs": marker,
        "uncovered_dims": uncovered, "bus_covered": dict(cov.covered),
    }


def _maxgeo_gate_verdict(
    project_root: str, tb_path: str, tb_result: dict | None = None
) -> dict | None:
    """``None`` -> gate disabled or no declared dims (a true no-op).
    ``{"verdict": "pass", ...}`` -> evaluated and fully covered (callers LOG
    it). ``{"advisory": True, ...}`` -> covered in scope with a loud recorded
    gap. Anything else -> violation dict: the design declares dimensional
    maxima but the testbench's ``# MAXGEO`` marker does not prove a
    max-geometry case for every declared dimension.

    NAME-AGNOSTIC: each declared dimension's max VALUE must appear as a marker
    ``key=value`` pair (the key name is free-form data). Testing at the declared
    maximum inherently crosses every 2^n index boundary below it -- exactly
    where a truncated index/address width wraps. NEVER raises (a parse hiccup is
    non-blocking; the prompt requirement is the primary defense).

    ``tb_result`` is the generator's own record for this testbench. When it
    identifies the ENGINE'S deterministic, compute-lane-independent QSPI
    conformance TB, :func:`_maxgeo_conformance_scope` may return an ADVISORY
    verdict (``advisory: True``) instead of a failure -- see that function for
    why that is a scope, not a weakening. Callers MUST treat ``advisory`` as
    "passed, with a loud recorded gap", and anything else as a failure."""
    try:
        if not _maxgeo_gate_enabled():
            return None
        # run3-followups: single-sourced declared table (byte-equality with
        # the legacy _declared_dimensions is pinned by test).
        from orchestrator.langgraph.bfm_lib import maxgeo as _maxgeo_lib
        dims = _maxgeo_lib.declared_dimensional_maxima(project_root)
        if not dims:
            return None
        marker = _tb_maxgeo_pairs(tb_path)
        # run3-followups: single-sourced demand partition (bfm_lib.maxgeo).
        # `missing` is provably identical to the old value-set computation;
        # `value_only` newly NAMES the dims whose only evidence is a value
        # collision with another marker pair, so the gate's number and the
        # TB's confession finally agree.
        _demand = _maxgeo_lib.maxgeo_demand(dims, marker)
        missing = _demand.missing
        value_only = _demand.value_only
        if not missing:
            # run3-followups: an evaluated PASS is a verdict, not a silence --
            # the caller logs it so a suppressed gate can never read as green.
            return {"verdict": "pass", "declared_dims": dims,
                    "marker_pairs": marker,
                    "value_only_dims": value_only}
        scoped = _maxgeo_conformance_scope(
            project_root, tb_path, tb_result, dims, marker, missing)
        if scoped is not None:
            return scoped
        # run3-followups: a functional MAXIMUM-CONFIGURATION case (baked by the
        # deterministic codegen, advertised via # MAXGEO_CASE) drives the max
        # config register value and the full IN/OUT payload extents end-to-end
        # against the golden -- the 2^n index/address wrap class this gate
        # exists to catch IS exercised. Remaining per-dimension attainment is
        # downgraded to a LOUD advisory gap (carried-forward defect), the same
        # treatment as the bus-scoped conformance path. The gate stays HARD
        # when no such case exists or its extents miss the declared maxima.
        case = _tb_maxgeo_case(tb_path)
        if case:
            dim_values = set(dims.values())
            attained = all(
                isinstance(case.get(k), int) and case[k] in dim_values
                for k in ("cfg0", "in_bytes", "out_bytes")
            )
            if attained:
                return {
                    "advisory": True,
                    "scope": "functional-max-case",
                    "uncovered_dims": missing,
                    "value_only_dims": value_only,
                    "declared_dims": dims,
                    "marker_pairs": marker,
                    "functional_max_case": case,
                    "reason": (
                        "MAX-GEOMETRY gate: functional max-configuration case "
                        f"{case} drives the maximum configuration and full "
                        "payload extents end-to-end against the golden "
                        "reference, exercising the 2^n index/address wrap "
                        "class. Per-dimension attainment for "
                        f"{sorted(missing)} is not individually proven -- "
                        "recorded as a loud advisory gap, not a hard failure."
                    ),
                }
        reason = (
            "MAX-GEOMETRY DV GATE FAILED: the design declares dimensional "
            "maxima but the testbench does not exercise them. A chip can pass "
            "every fixed-small-geometry test yet ship a truncated index/address/"
            "counter width that wraps at a 2^n boundary BELOW the declared "
            "maximum (the class killer). The testbench MUST include at least one "
            "MAX-GEOMETRY test case and advertise it with a "
            "`# MAXGEO: <dim_name>=<value>` marker covering EVERY declared "
            "dimension at its maximum.\n"
            f"  declared maxima : {dims}\n"
            f"  marker pairs    : {marker or '(no # MAXGEO marker found)'}\n"
            f"  uncovered dims  : {missing}\n"
            f"  value-collision-only (not individually proven): {value_only}\n"
            "Fix: regenerate/edit the testbench to drive a max-geometry case "
            "(sparse/short content at the maximum dimensions is acceptable if a "
            "full workload is too slow -- the point is to exercise the index/"
            "address widths at maximum extent) and emit the marker."
        )
        return {"reason": reason, "declared_dims": dims,
                "marker_pairs": marker, "uncovered_dims": missing,
                "value_only_dims": value_only}
    except Exception:  # noqa: BLE001
        return None


async def integration_dv_node(state: OrchestratorState) -> dict:
    """Generate and run an integration-level cocotb testbench.

    This node is the Lead DV (AI) step that:
    1. Calls the IntegrationTestbenchGenerator LLM to produce a cocotb
       testbench exercising the top-level integrated design
    2. Runs the testbench via Verilator against all block RTL
    3. On failure, fires an interrupt so the outer agent can diagnose

    Runs after integration_check passes (lint-clean top-level RTL exists).
    """
    import json as _json

    pr = state.get("project_root", str(PROJECT_ROOT))
    integration_result = state.get("integration_result") or {}

    top_rtl_path = integration_result.get("top_rtl_path", "")
    design_name = integration_result.get("design_name", "chip_top")
    block_rtl_paths = integration_result.get("block_rtl_paths", {})

    write_graph_event(pr, "Integration DV", "graph_node_enter", {
        "design_name": design_name,
        "block_count": len(block_rtl_paths),
    })

    with _tracer.start_as_current_span("Integration DV") as span:
        span.set_attribute("design_name", design_name)

        if not top_rtl_path or not Path(top_rtl_path).exists():
            log("  [INTEG-DV] Skipping -- no top-level RTL found", YELLOW)
            write_graph_event(pr, "Integration DV", "graph_node_exit", {
                "skipped": True, "reason": "no_top_rtl",
            })
            return {"integration_dv_result": {
                "skipped": True,
                "reason": "No top-level RTL available",
            }}

        if len(block_rtl_paths) < 1:
            log("  [INTEG-DV] Skipping -- no block RTL found", YELLOW)
            write_graph_event(pr, "Integration DV", "graph_node_exit", {
                "skipped": True, "reason": "no_blocks",
            })
            return {"integration_dv_result": {
                "skipped": True,
                "reason": "No block RTL files found",
            }}

        # Load connections and PRD summary for context
        connections, _ = await asyncio.to_thread(
            load_architecture_connections, pr
        )

        prd_summary = ""
        for prd_name in ("prd_spec.json", "ers_spec.json"):
            prd_path = Path(pr) / ".coresmith" / prd_name
            if prd_path.exists():
                try:
                    prd_data = _json.loads(prd_path.read_text(encoding="utf-8"))
                    doc = prd_data.get("prd", prd_data.get("ers", {}))
                    prd_summary = doc.get("summary", "")
                    if doc.get("speed_and_feeds"):
                        sf = doc["speed_and_feeds"]
                        prd_summary += (
                            f"\nTarget clock: {sf.get('target_clock_mhz', '?')} MHz"
                            f", Data width: {sf.get('input_data_rate_mbps', '?')} Mbps"
                        )
                    if doc.get("dataflow"):
                        df = doc["dataflow"]
                        prd_summary += (
                            f"\nBus protocol: {df.get('bus_protocol', '?')}"
                            f", Data width: {df.get('data_width_bits', '?')} bits"
                        )
                except (OSError, _json.JSONDecodeError, KeyError):
                    pass
                break

        # Re-parse modules so the LLM gets block port details
        modules = {}
        for block_name, rtl_path in block_rtl_paths.items():
            # Parse the BLOCK's module, not whichever comes first in
            # the file: generated files declare internal stages ahead
            # of the block itself.
            mod = await asyncio.to_thread(
                parse_verilog_ports, rtl_path,
                module_for_block(rtl_path, block_name))
            if mod.name:
                modules[block_name] = mod

        previous_dv = state.get("integration_dv_result") or {}
        previous_action = previous_dv.get("action_taken", "")
        previous_tb_path = previous_dv.get("testbench_path", "")
        reuse_existing_tb = (
            previous_action in ("fix_rtl", "fix_tb")
            and previous_tb_path
            and Path(previous_tb_path).exists()
        )

        generation_error: Exception | None = None
        if reuse_existing_tb:
            log(
                "  [INTEG-DV] Reusing existing testbench after "
                f"{previous_action}: {previous_tb_path}",
                YELLOW,
            )
            tb_result = {
                "testbench_path": previous_tb_path,
                "tb_path": previous_tb_path,
                "test_count": previous_dv.get("test_count", 0),
            }
        else:
            # 1. Generate integration testbench
            log("  [INTEG-DV] Generating integration testbench...", YELLOW)
            tb_result = None

            # DETERMINISTIC BFM (CORESMITH_DETERMINISTIC_BFM=1): when the
            # chip-top's external bus is a QSPI-slave, drive the pins with the
            # contract-faithful *library* BFM instead of an LLM-authored driver
            # that co-tunes to the DUT (a non-conformant read serializer then
            # passes CoreSmith DV but fails the real fixed host -- the AES bug).
            # The LLM still supplies the golden MODEL (used at generation time to
            # compute the expected OUT bytes); only the pin driver is made
            # deterministic + DUT-blind. Flag-off -> byte-identical to before.
            try:
                from orchestrator.langgraph import bfm_lib as _bfm_lib
                # conformance_enabled() is True whenever the deterministic BFM is
                # on (or CORESMITH_QSPI_CONFORMANCE=1). Even when a golden host-flow
                # plan cannot be derived (the compute-lane oracle is not modeled,
                # e.g. a BT.656 input lane), the QSPI-slave BUS PROTOCOL is still
                # exercised by a compute-lane-INDEPENDENT conformance DV -- closing
                # the image codec gap where a frontend missing cmd 0x05 / mistiming the
                # read turnaround slipped through on the co-tuned LLM BFM.
                if _bfm_lib.conformance_enabled():
                    _top_src = await asyncio.to_thread(
                        lambda: Path(top_rtl_path).read_text(encoding="utf-8")
                    )
                    # Classify the GRADED boundary, not merely "does this file
                    # mention io_in". The external grader drives io_in/io_out/
                    # io_oeb on `user_project_wrapper`, so an assembled top whose
                    # own pins are design-prefixed (qspi_io_in/qspi_io_out/
                    # qspi_drive_en) is NOT the graded boundary even though the
                    # chassis IS QSPI-slave -- and the boundary usually still
                    # exists, in rtl/user_project_wrapper.v. The verdict says
                    # WHERE it lives and whether THIS sim will drive it.
                    _verdict = await asyncio.to_thread(
                        _bfm_lib.classify_bus_verdict, pr, _top_src, connections,
                        design_name, top_rtl_path,
                    )
                    _contract = _verdict.contract if _verdict.contract_enforcing else None
                    if _contract is not None:
                        _plan = await asyncio.to_thread(
                            _bfm_lib.build_plan_from_run, pr, _contract
                        )
                        _out = str(
                            PROJECT_ROOT / "tb" / "integration"
                            / f"test_{design_name}.py"
                        )
                        # The design's OWN declared dimensional maxima -- the
                        # same dict the MAX-GEOMETRY gate reads. Threading it in
                        # here is what lets the deterministic TB drive the BUS
                        # maxima at full extent and mark them per design,
                        # instead of the generator inventing dimension names.
                        _dims = await asyncio.to_thread(_declared_dimensions, pr)
                        if _bfm_lib.deterministic_bfm_enabled() and _plan is not None:
                            tb_result = _bfm_lib.write_deterministic_integration_tb(
                                pr, design_name, _contract, _plan, _out,
                                include_conformance=True,
                                declared_dims=_dims,
                            )
                            log(
                                "  [INTEG-DV] DETERMINISTIC BFM ACTIVE (QSPI-slave; "
                                f"contract fp={_contract.fingerprint()}, "
                                f"case={_plan.case_name}): contract-faithful, "
                                "DUT-blind host-flow + compute-lane-independent "
                                "BUS-PROTOCOL conformance (0x02/0x03+dummy/0x05) -- "
                                "this run's integration DV is CONTRACT-ENFORCING.",
                                GREEN,
                            )
                        else:
                            # No golden host-flow plan (compute lane unmodeled) or
                            # the deterministic host-flow is off: still enforce the
                            # QSPI-slave BUS PROTOCOL with a compute-lane-independent
                            # conformance DV instead of silently using the LLM BFM.
                            tb_result = _bfm_lib.write_qspi_conformance_tb(
                                pr, design_name, _contract, _out,
                                declared_dims=_dims,
                            )
                            if tb_result.get("maxgeo_covered"):
                                log(
                                    "  [INTEG-DV] MAX-EXTENT bus coverage: "
                                    f"{tb_result['maxgeo_covered']} driven at "
                                    "maximum; NOT covered (no compute oracle): "
                                    f"{tb_result.get('maxgeo_uncovered', {})}",
                                    YELLOW,
                                )
                            _lane = (
                                "unmodeled"
                                if _plan is None
                                else "modeled (deterministic host-flow off)"
                            )
                            log(
                                "  [INTEG-DV] QSPI-SLAVE CONFORMANCE DV ACTIVE "
                                f"(compute lane {_lane}; contract fp="
                                f"{_contract.fingerprint()}): the bus protocol "
                                "(cmd 0x02 write / 0x03 read + dummy turnaround / "
                                "0x05 read_status) is CONTRACT-ENFORCING even "
                                "though the compute oracle is not modeled. A "
                                "frontend missing 0x05 or mistiming the read "
                                "turnaround FAILS HERE, not at the secret grader.",
                                GREEN,
                            )
                    elif _verdict.fails_closed:
                        # THE FAIL-OPEN THIS CLOSES. The spec says the external
                        # bus is QSPI, but the module this DV elaborates does not
                        # expose the graded pin boundary -- either it lives on
                        # another module (typically the wrapper the grader drives)
                        # or no io_in/io_out/io_oeb boundary exists anywhere
                        # (spec/RTL contradiction). Historically this logged a RED
                        # advisory and PROCEEDED on the LLM-authored, DUT-co-tuned
                        # BFM: DV was silently downgraded exactly when the design
                        # is most likely to be non-conformant, and the run still
                        # reported "INTEGRATION DV PASSED" (observed live on a graded run
                        # 20260727). It is now the run's normal tb_generation
                        # interrupt (retry/fix_rtl/fix_tb/abort), per the local
                        # honest-gate idiom.
                        _gate_on = _bfm_lib.boundary_gate_enabled()
                        write_graph_event(pr, "Integration DV", "qspi_boundary_gate", {
                            "gate": "qspi_pin_boundary",
                            "status": _verdict.status,
                            "enforced": _gate_on,
                            "simulated_top": _verdict.simulated_top,
                            "boundary_module": (
                                _verdict.boundary.module if _verdict.boundary else ""
                            ),
                            "boundary_path": (
                                _verdict.boundary.path if _verdict.boundary else ""
                            ),
                            "top_ports": list(_verdict.top_ports[:16]),
                            "reason": _verdict.reason,
                        })
                        if _gate_on:
                            log(
                                "  [INTEG-DV] QSPI PIN-BOUNDARY GATE FAILED "
                                "(fail-closed -- this is NOT a pass, and the "
                                "co-tuned LLM BFM is NOT an acceptable fallback "
                                f"here): {_verdict.reason}",
                                RED,
                            )
                            generation_error = RuntimeError(
                                "QSPI pin-boundary gate: " + _verdict.reason
                            )
                        else:
                            log(
                                "  [INTEG-DV] ADVISORY (QSPI pin-boundary gate "
                                "DISABLED by CORESMITH_QSPI_BOUNDARY_GATE=0 / "
                                "CORESMITH_GATE_FAIL_OPEN=1): keeping the "
                                "LLM-authored BFM. THIS RUN'S INTEGRATION DV IS "
                                "NOT CONTRACT-ENFORCING -- the BFM may co-tune to "
                                "the DUT and pass a non-conformant design. "
                                f"{_verdict.reason}",
                                RED,
                            )
                            # An advisory bypass must never be SILENT: carry the
                            # specific unmodeled boundary forward to the final
                            # report / validation-DV context.
                            record_carried_forward_defect(pr, {
                                "gate": "qspi_pin_boundary",
                                "kind": "dv_not_contract_enforcing",
                                "advisory": True,
                                "unmodeled": (
                                    "graded Caravel pin boundary (io_in/io_out/"
                                    "io_oeb) not driven by integration DV "
                                    f"({_verdict.status}; boundary="
                                    + (
                                        _verdict.boundary.describe()
                                        if _verdict.boundary
                                        else "absent"
                                    )
                                    + f"; simulated top='{_verdict.simulated_top}')"
                                ),
                                "first_divergence_block": "",
                                "note": _verdict.reason,
                            })
                    else:
                        log(
                            "  [INTEG-DV] ADVISORY: chip-top is not a QSPI-slave "
                            "bus; keeping the LLM-authored BFM. THIS RUN'S "
                            "INTEGRATION DV IS NOT CONTRACT-ENFORCING -- the BFM "
                            "may co-tune to the DUT and pass a non-conformant "
                            f"design. ({_verdict.reason})",
                            RED,
                        )
            except Exception as _bfm_e:  # noqa: BLE001
                # A gate that RAISES is not a pass (gate_guard / A-Fix 2): when the
                # architecture says the bus is QSPI we must not fall back to the
                # co-tuned BFM just because the classifier/codegen broke. Fail
                # closed unless the gate (or the global knob) is explicitly off.
                _bfm_fail_closed = False
                try:
                    from orchestrator.langgraph import bfm_lib as _bfm_probe
                    _bfm_fail_closed = (
                        _bfm_probe.boundary_gate_enabled()
                        and _bfm_probe.arch_indicates_qspi_slave(pr, connections)
                    )
                except Exception:  # noqa: BLE001
                    _bfm_fail_closed = False
                if _bfm_fail_closed:
                    log(
                        "  [INTEG-DV] deterministic BFM path ERRORED on a run "
                        f"whose spec declares a QSPI bus ({_bfm_e}) -- failing "
                        "closed: an errored gate is not a pass, and the co-tuned "
                        "LLM BFM would silently un-enforce the bus contract.",
                        RED,
                    )
                    generation_error = RuntimeError(
                        "QSPI pin-boundary gate could not run (an errored gate is "
                        f"not a pass): {_bfm_e}"
                    )
                else:
                    log(
                        "  [INTEG-DV] deterministic BFM path errored "
                        f"({_bfm_e}); falling back to LLM BFM",
                        YELLOW,
                    )
                tb_result = None

            # `generation_error` set above == the QSPI pin-boundary gate failed
            # closed. Falling through to the LLM generator is precisely the
            # outcome the gate forbids, so skip it and let the existing
            # tb_generation failure path fire the interrupt.
            if tb_result is None and generation_error is None:
                # Re-point DV to model-equivalence when block-goldens is on: drive
                # the RTL and the integrated Amaranth chip model with the same stimulus
                # and assert RTL == chip model. Flag-off -> chip_model_path stays ""
                # so the prompt + generated TB are byte-identical to before.
                chip_model_path = ""
                try:
                    from orchestrator.architecture import composition as _composition
                    if _composition.block_goldens_enabled():
                        _cm = (
                            Path(pr) / "arch" / _composition.BLOCK_MODELS_DIRNAME
                            / "_chip_model.py"
                        )
                        if _cm.exists():
                            chip_model_path = str(_cm)
                except Exception:  # noqa: BLE001
                    chip_model_path = ""
                try:
                    tb_result = await generate_integration_testbench(
                        design_name=design_name,
                        top_rtl_path=top_rtl_path,
                        modules=modules,
                        connections=connections,
                        block_rtl_paths=block_rtl_paths,
                        prd_summary=prd_summary,
                        prior_failure=_format_dv_retry_context(previous_dv),
                        chip_model_path=chip_model_path,
                        parameter_table=_ers_parameter_table(pr),
                    )
                except Exception as e:
                    tb_result = None
                    generation_error = e
        if not reuse_existing_tb and tb_result is None:
            if generation_error is None:
                generation_error = RuntimeError("Integration testbench generation returned no result")
            e = generation_error
            log(f"  [INTEG-DV] Testbench generation failed: {e}", RED)
            error_msg = f"Integration testbench generation failed: {e}"
            contract_audit = await _run_top_level_contract_audit(
                stage="integration_dv_generation",
                project_root=pr,
                design_name=design_name,
                top_rtl_path=top_rtl_path,
                testbench_path="",
                test_count=0,
                sim_log=error_msg,
                sim_log_path="",
                block_rtl_paths=block_rtl_paths,
                            supported_actions=["retry", "fix_rtl", "fix_tb", "abort"],
            )
            payload = {
                "type": "integration_dv_failure",
                "phase": "tb_generation",
                "design_name": design_name,
                "top_rtl_path": top_rtl_path,
                "testbench_path": "",
                "test_count": 0,
                "sim_log": error_msg,
                "sim_log_path": "",
                "block_rtl_paths": block_rtl_paths,
                "contract_audit": contract_audit,
                "contract_audit_path": contract_audit.get("audit_path", ""),
                "supported_actions": [
                    "retry",
                    "fix_rtl",
                    "fix_tb",
                    "abort",
                ],
                "outer_agent_guidance": (
                    "Integration DV could not generate a usable cocotb "
                    "testbench. As the outer-loop diagnostic agent, inspect "
                    "the top-level RTL, block port contracts, generator prompt, "
                    "and any partially written testbench. Use action='fix_tb' "
                    "when the testbench generator or prompt needs repair, "
                    "action='fix_rtl' when the top-level contract is invalid, "
                    "or action='retry' after an external fix. Do not mark the "
                    "pipeline complete until Integration DV runs.\n\n"
                    "Contract audit result: "
                    f"{contract_audit.get('category', 'UNKNOWN')} -- "
                    f"{contract_audit.get('outer_agent_summary', '')}"
                ),
                "reference_files": {
                    "top_rtl": top_rtl_path,
                    "contract_audit": contract_audit.get("audit_path", ""),
                },
            }
            # run3-followups: park in integration_dv_decision instead of a
            # tail interrupt() (see the sim-failure branch for the
            # one-cycle-late defect).
            write_graph_event(pr, "Integration DV", "graph_node_exit", {
                "error": str(e),
                "phase": "tb_generation",
                "action": "pending_decision",
            })
            return {
                "integration_dv_result": {
                    "passed": False,
                    "pending_decision": True,
                    "interrupt_payload": payload,
                    "error": error_msg,
                    "phase": "tb_generation",
                    "test_count": 0,
                    "testbench_path": "",
                    "design_name": design_name,
                    "contract_audit": contract_audit,
                    "contract_audit_path": contract_audit.get("audit_path", ""),
                },
                "pipeline_done": False,
            }

        tb_path = tb_result.get("testbench_path", "")
        test_count = tb_result.get("test_count", 0)
        log(f"  [INTEG-DV] Generated ({test_count} tests): {tb_path}", GREEN)
        span.set_attribute("test_count", test_count)

        # 2. Run integration simulation
        log("  [INTEG-DV] Running integration simulation...", YELLOW)
        sim_result = await asyncio.to_thread(
            run_integration_simulation,
            design_name, top_rtl_path, block_rtl_paths, tb_path,
        )

        passed = sim_result.get("passed", False)
        sim_log = sim_result.get("log", "")

        chip_equiv_result: dict | None = None
        if passed:
            # A-Fix 5(d): a green integration sim (esp. with a loosened or
            # operator-edited TB) is NOT sufficient. The ENGINE re-drives
            # chip_top + the composed Amaranth chip model on the same seeded
            # stimulus and asserts byte equivalence. A real divergence flips the
            # DV to failed -> the existing failure interrupt; an honest skip
            # keeps the pass; a harness error is retried then fails closed.
            chip_equiv_result = await asyncio.to_thread(
                _maybe_run_chip_equiv,
                pr, design_name, top_rtl_path, block_rtl_paths,
            )
            if (
                chip_equiv_result is not None
                and not chip_equiv_result.get("passed")
                and not chip_equiv_result.get("skipped")
            ):
                passed = False
                equiv_reason = chip_equiv_result.get("reason", "")
                sim_log = (
                    (sim_log + "\n\n") if sim_log else ""
                ) + (
                    "CHIP-TOP RTL-vs-MODEL EQUIVALENCE FAILED (the integration "
                    "sim passed but chip_top does not byte-match the composed "
                    "Amaranth chip model on a fresh seeded stimulus):\n"
                    + equiv_reason
                )
                span.set_attribute("chip_equiv_failed", True)
                log("  [INTEG-DV] chip-top equivalence FAILED -- flipping DV "
                    f"to failed: {equiv_reason}", RED)
            elif chip_equiv_result is not None and chip_equiv_result.get("passed"):
                log("  [INTEG-DV] chip-top equivalence PASSED", GREEN)
            elif chip_equiv_result is not None and chip_equiv_result.get("skipped"):
                log("  [INTEG-DV] chip-top equivalence skipped "
                    f"(non-blocking): {chip_equiv_result.get('reason','')}", YELLOW)

        # MAX-GEOMETRY gate (rung3-fixes-2): a green sim is NOT sufficient when
        # the design declares dimensional maxima but the TB never exercised
        # them -- that is how a truncated index-width bug ships in a "verified"
        # chip. Flip the DV to failed -> existing failure interrupt.
        # run3-followups: the gate runs on EVERY passing cycle, including
        # operator-reused TBs (fix_tb/fix_rtl). "Trusted as-is" silently
        # DISARMED the gate on exactly the cycles that deserve more scrutiny --
        # proven live when a clobbered 2-test TB passed DV with no MAXGEO line
        # at all. Every evaluated outcome logs a verdict; silence now means
        # only "gate disabled or no declared dims".
        if passed:
            _mg = _maxgeo_gate_verdict(pr, tb_path, tb_result)
            if reuse_existing_tb and _mg is not None:
                log("  [INTEG-DV] MAX-GEOMETRY gate: evaluating an OPERATOR-"
                    "REUSED testbench (fix_tb/fix_rtl) -- operator edits get "
                    "more scrutiny, not less.", YELLOW)
            if _mg is not None and _mg.get("verdict") == "pass":
                log("  [INTEG-DV] MAX-GEOMETRY gate PASS -- every declared "
                    f"maximum appears in the TB markers: "
                    f"{_mg.get('marker_pairs', {})}", GREEN)
                write_graph_event(pr, "Integration DV", "maxgeo_gate_pass", {
                    "gate": "maxgeo",
                    "marker_pairs": _mg.get("marker_pairs", {}),
                })
            elif _mg is not None and _mg.get("advisory"):
                # SCOPED, not silent. Either the engine's compute-lane-
                # independent conformance TB drove every BUS maximum and cannot
                # drive the compute lane, or a functional max-configuration
                # case covered the wrap class without per-dimension proof; the
                # gap is logged RED, written to the event stream, and carried
                # forward as a defect so the final report and validation DV
                # both see it.
                _mg_scope = _mg.get("scope", "bus-contract-only")
                log("  [INTEG-DV] MAX-GEOMETRY gate ADVISORY "
                    f"(scope={_mg_scope}) -- this is NOT full max-geometry "
                    f"coverage. NOT COVERED: {_mg['uncovered_dims']}", RED)
                write_graph_event(pr, "Integration DV", "maxgeo_gate_scoped", {
                    "gate": "maxgeo",
                    "scope": _mg_scope,
                    "bus_covered": _mg.get("bus_covered", {}),
                    "functional_max_case": _mg.get("functional_max_case", {}),
                    "uncovered_dims": _mg.get("uncovered_dims", {}),
                })
                record_carried_forward_defect(pr, {
                    "gate": "maxgeo",
                    "kind": "max_geometry_not_covered",
                    "advisory": True,
                    "unmodeled": (
                        "dimensional maxima never individually driven at "
                        f"maximum extent: {_mg.get('uncovered_dims', {})} "
                        f"(scope={_mg_scope})"
                    ),
                    "first_divergence_block": "",
                    "note": _mg["reason"],
                })
                sim_log = ((sim_log + "\n\n") if sim_log else "") + _mg["reason"]
            elif _mg is not None:
                passed = False
                sim_log = ((sim_log + "\n\n") if sim_log else "") + _mg["reason"]
                span.set_attribute("maxgeo_gate_failed", True)
                log("  [INTEG-DV] MAX-GEOMETRY gate FAILED -- flipping DV to "
                    f"failed: uncovered={_mg['uncovered_dims']}", RED)

        # v3 Section 2: CHIP-LEVEL measured throughput. The deterministic-BFM TB
        # wrote integration_throughput.json (op window START-committed -> DONE
        # visible on the status pin -- how a grader-style host measures). Read it,
        # gate against a resolvable chip budget x 1.1, PERSIST the record for the
        # final report, and thread the measured number into run state. Measure-
        # only (never demotes) on the LLM-BFM path (no artifact) or when no chip
        # budget is resolvable. Never crashes the node.
        chip_tput: dict | None = None
        try:
            from orchestrator.langgraph import throughput_gate as _tg
            _isim = Path(pr) / "sim_build" / "integration"
            chip_tput = _tg.evaluate_chip_throughput(pr, _isim, state=state)
            _tg.persist_chip_throughput(pr, chip_tput)
            _cm = chip_tput.get("measured_cyc_per_op_chip")
            if _cm is not None:
                log(f"  [INTEG-DV] chip throughput measured {_cm} cyc/op "
                    f"({chip_tput.get('budget_source', 'none')} budget)", CYAN)
            if (passed and chip_tput.get("applicable")
                    and chip_tput.get("passed") is False):
                passed = False
                sim_log = ((sim_log + "\n\n") if sim_log else "") + (
                    chip_tput.get("report", "") or chip_tput.get("reason", ""))
                span.set_attribute("chip_throughput_gate_failed", True)
                log("  [INTEG-DV] chip MEASURED-THROUGHPUT gate FAILED -- "
                    f"flipping DV to failed: {chip_tput.get('reason', '')}", RED)
        except Exception as _ce:  # noqa: BLE001 - never crash the DV node
            log(f"  [INTEG-DV] chip throughput eval skipped ({_ce})", YELLOW)

        if passed:
            log(f"\n{'='*60}", GREEN)
            log("  INTEGRATION DV PASSED", GREEN)
            log(f"  {test_count} tests, all passing", GREEN)
            log(f"{'='*60}\n", GREEN)
            span.set_attribute("passed", True)

            write_graph_event(pr, "Integration DV", "graph_node_exit", {
                "passed": True,
                "test_count": test_count,
                "log_path": sim_result.get("log_path", ""),
                "chip_equiv": (chip_equiv_result or {}).get("reason", ""),
            })

            _record_dv_row(
                pr, block=design_name, scope="chip", source="gate", passed=True,
                tests_passed=test_count, tests_total=test_count,
                log_path=sim_result.get("log_path", ""),
                detail="integration_dv passed",
            )
            return {
                "integration_dv_result": {
                    "passed": True,
                    "test_count": test_count,
                    "testbench_path": tb_path,
                    "sim_log_path": sim_result.get("log_path", ""),
                    "design_name": design_name,
                    "chip_equiv_result": chip_equiv_result,
                    "measured_cyc_per_op_chip": (chip_tput or {}).get(
                        "measured_cyc_per_op_chip"),
                },
                "pipeline_done": False,
            }

        # 3. Simulation failed -- interrupt for outer agent diagnosis
        log("  [INTEG-DV] FAILED", RED)
        for line in sim_log.split("\n")[-10:]:
            if line.strip():
                log(f"    {line.strip()}", RED)

        span.set_attribute("passed", False)

        contract_audit = await _run_top_level_contract_audit(
            stage="integration_dv",
            project_root=pr,
            design_name=design_name,
            top_rtl_path=top_rtl_path,
            testbench_path=tb_path,
            test_count=test_count,
            sim_log=sim_log,
            sim_log_path=sim_result.get("log_path", ""),
            block_rtl_paths=block_rtl_paths,
                    supported_actions=["retry", "fix_rtl", "fix_tb", "abort"],
        )
        _record_dv_row(
            pr, block=design_name, scope="chip", source="gate", passed=False,
            first_divergence=contract_audit.get("first_divergence"),
            detail=str(contract_audit.get("category", ""))[:200],
            log_path=sim_result.get("log_path", ""),
        )

        payload = {
            "type": "integration_dv_failure",
            "design_name": design_name,
            "top_rtl_path": top_rtl_path,
            "testbench_path": tb_path,
            "test_count": test_count,
            "sim_log": sim_log[-3000:],
            "sim_log_path": sim_result.get("log_path", ""),
            "block_rtl_paths": block_rtl_paths,
            "contract_audit": contract_audit,
            "contract_audit_path": contract_audit.get("audit_path", ""),
            "supported_actions": [
                "retry",        # regenerate testbench + re-simulate
                "fix_rtl",      # outer agent fixed RTL, re-run sim only
                "fix_tb",       # outer agent fixed testbench, re-run sim only
                "abort",        # stop the pipeline
            ],
            "outer_agent_guidance": (
                "Integration DV (top-level simulation) failed. As the outer-loop "
                "diagnostic agent, read the sim log and testbench to diagnose:\n"
                "1. TESTBENCH BUG: If the testbench has incorrect port names, "
                "wrong timing, or bad assumptions, edit the testbench at "
                f"{tb_path} and resume with action='fix_tb'.\n"
                "2. RTL WIRING BUG: If the top-level wiring is wrong (e.g., "
                "signals crossed, wrong widths), edit the top-level RTL at "
                f"{top_rtl_path} and resume with action='fix_rtl'.\n"
                "3. BLOCK BUG: If a specific block's output is wrong, this may "
                "need per-block debugging. Note which block and escalate.\n"
                "4. TIMEOUT: If the sim timed out, check for combinational "
                "loops or missing clock/reset connections.\n"
                "5. After fixing, resume_pipeline(action='fix_rtl' or 'fix_tb') "
                "to re-run integration DV.\n"
                "6. Only escalate to the user for architectural issues."
                "\n\nContract audit result: "
                f"{contract_audit.get('category', 'UNKNOWN')} -- "
                f"{contract_audit.get('outer_agent_summary', '')}"
            ),
            "reference_files": {
                "top_rtl": top_rtl_path,
                "testbench": tb_path,
                "sim_log": sim_result.get("log_path", ""),
                "contract_audit": contract_audit.get("audit_path", ""),
            },
        }

        if os.environ.get("CORESMITH_ALLOW_SKIP_INTEGRATION_DV", "").lower() in (
            "1",
            "true",
            "yes",
        ):
            payload["supported_actions"].insert(-1, "skip")

        # run3-followups: do NOT call interrupt() here. LangGraph re-executes
        # this entire node from the top on resume, so a response delivered to
        # a tail interrupt() is consumed ONE FULL CYCLE LATE -- the intervening
        # default cycle regenerates the testbench and destroys operator fix_tb
        # edits (proven live: three consecutive clobbers, then a false PASS on
        # the clobbered TB). The failure parks in integration_dv_decision,
        # whose re-execution is just the interrupt() call: the response lands
        # immediately and the TB on disk at decision time is the TB the next
        # cycle sees.
        write_graph_event(pr, "Integration DV", "graph_node_exit", {
            "action": "pending_decision",
            "passed": False,
            "test_count": test_count,
        })
        return {
            "integration_dv_result": {
                "passed": False,
                "pending_decision": True,
                "interrupt_payload": payload,
                "test_count": test_count,
                "testbench_path": tb_path,
                "sim_log_path": sim_result.get("log_path", ""),
                "design_name": design_name,
                "contract_audit": contract_audit,
                "contract_audit_path": contract_audit.get("audit_path", ""),
                "chip_equiv_result": chip_equiv_result,
            },
            "pipeline_done": False,
        }


async def integration_dv_decision_node(state: OrchestratorState) -> dict:
    """Consume the operator's decision for a parked integration-DV failure.

    Split out of ``integration_dv_node`` (run3-followups): a LangGraph resume
    re-executes the interrupted node from the top, so an ``interrupt()`` at the
    tail of the big DV node consumed its response one full default cycle late,
    regenerating the testbench over operator edits before the action landed.
    This node's body is ONLY the interrupt + response handling: re-execution is
    free, the response lands immediately, and a fix_tb reuse sees the disk
    state as of decision time."""
    pr = state["project_root"]
    dv = dict(state.get("integration_dv_result") or {})
    payload = dv.get("interrupt_payload") or {
        "type": "integration_dv_failure",
        "supported_actions": ["retry", "fix_rtl", "fix_tb", "abort"],
    }
    response = interrupt(payload) or {}
    action = response.get("action", "abort")
    test_count = dv.get("test_count", 0)
    write_graph_event(pr, "Integration DV", "graph_node_exit", {
        "action": action,
        "passed": False,
        "test_count": test_count,
    })

    dv_result = dict(dv)
    dv_result.pop("interrupt_payload", None)
    dv_result["passed"] = False
    dv_result["pending_decision"] = False
    dv_result["action_taken"] = action

    if action == "skip":
        dv_result["skipped_by_user"] = True
        log("  [INTEG-DV] Skipped by user/agent", YELLOW)
    elif action == "abort":
        dv_result["aborted"] = True
        log("  [INTEG-DV] Aborted", RED)
    elif action in ("retry", "fix_rtl", "fix_tb"):
        fix_desc = response.get("rtl_fix_description", "")
        log(
            f"  [INTEG-DV] Fix applied (action={action}): "
            f"{fix_desc or '(no description provided)'}",
            GREEN,
        )
        dv_result["fix_applied"] = fix_desc
        if action == "fix_tb":
            # A-Fix 5(c): the operator hand-edited the testbench. The chip sim
            # re-runs trusting that TB, and the MAX-GEOMETRY + chip-top
            # equivalence gates evaluate it with MORE scrutiny -- record +
            # LOUDLY flag the operator TB edit so a loosened TB can't quietly
            # pass.
            dv_result["tb_operator_edited"] = True
            log(
                "  [INTEG-DV] fix_tb: operator EDITED the testbench "
                "(tb_operator_edited=True). A green sim on an operator-edited "
                "TB is NOT sufficient -- the MAX-GEOMETRY and chip-top "
                "equivalence gates still evaluate it.",
                YELLOW,
            )

    return {
        "integration_dv_result": dv_result,
        "pipeline_done": False,
        "pipeline_aborted": action == "abort",
    }


def route_after_integration_dv_decision(state: OrchestratorState) -> str:
    """Route the operator's decision back into integration DV or terminate."""
    result = state.get("integration_dv_result") or {}
    if result.get("action_taken") in ("retry", "fix_rtl", "fix_tb"):
        return "integration_dv"
    return END


route_after_integration_dv_decision.__edge_labels__ = {
    "integration_dv": "RETRY / FIX",
    END: "DONE",
}


def _load_ers_validation_context(project_root: str) -> tuple[str, int]:
    """Load ERS context for validation DV and count likely RTL-checkable reqs."""
    ers_path = Path(project_root) / ".coresmith" / "ers_spec.json"
    if not ers_path.exists():
        return "", 0

    raw = ers_path.read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return raw, 0

    ers = data.get("ers", data)
    req_count = 0

    def _count_value(value) -> None:
        nonlocal req_count
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    req_count += 1
                elif isinstance(item, dict):
                    if item.get("requirement") or item.get("id"):
                        req_count += 1
                    _count_value(item)
        elif isinstance(value, dict):
            for nested in value.values():
                _count_value(nested)

    for key in (
        "functional_requirements",
        "per_block_requirements",
        "verification_requirements",
        "validation_dv_requirements",
        "validation_kpis",
    ):
        _count_value(ers.get(key))

    return json.dumps(data, indent=2), req_count


# ---------------------------------------------------------------------------
# Contract-audit staleness
# ---------------------------------------------------------------------------
# `.coresmith/contract_audit/<stage>_contract_audit.json` is a STAGE-derived,
# therefore STABLE, path: every integration-DV failure of the same stage writes
# the same filename. So a failed or skipped audit leaves the PREVIOUS failure's
# verdict lying in the exact place the next one is read from. Observed on the
# graded run: a 0.99-confidence TESTBENCH_BUG audit describing an already-fixed
# crash sat next to a new, different failure and was quoted as its diagnosis.
#
# A verdict is only about the failure it actually read, so it now carries that
# failure's identity: the sha256 + mtime of the failure-context JSON it audited.
# Anything reading an audit can then ask "is this about the failure in front of
# me?" instead of assuming yes because the file exists.

CONTRACT_AUDIT_STAMP_KEY = "audited_context"


def _context_fingerprint(context_path: str) -> dict:
    """Identity of a failure-context file: ``{sha256, mtime, path}``.

    Empty dict when it cannot be read -- an unknown identity must never be
    mistaken for a matching one.
    """
    try:
        p = Path(context_path)
        blob = p.read_bytes()
        return {
            "path": str(p),
            "sha256": hashlib.sha256(blob).hexdigest()[:32],
            "mtime": round(p.stat().st_mtime, 3),
        }
    except (OSError, ValueError):
        return {}


def contract_audit_staleness(audit: dict | None, context_path: str = "") -> str:
    """"" when the audit provably describes the failure context in front of us.

    Otherwise a human-readable STALE description. Two ways to be stale:
    unstamped (written before stamping existed, or by a path that skipped it),
    and stamped for a DIFFERENT failure context.

    Deliberately not a gate: a stale audit is still evidence, and deleting it
    would lose the trail. It is a LABEL, so a reader (and an outer agent acting
    on `recommended_action`) can tell "this is about your failure" from "this is
    about the last one".
    """
    if not isinstance(audit, dict) or not audit:
        return ""
    stamp = audit.get(CONTRACT_AUDIT_STAMP_KEY) or {}
    path = context_path or (stamp.get("path") if isinstance(stamp, dict) else "")
    if not isinstance(stamp, dict) or not stamp.get("sha256"):
        return ("STALE? this contract audit carries no failure-context stamp, so "
                "there is no evidence it describes the CURRENT failure rather "
                "than a previous one at the same stage. Treat its category / "
                "recommended_action as unverified.")
    if not path:
        return ""
    live = _context_fingerprint(path)
    if not live:
        return (f"STALE? the failure context this audit claims to describe "
                f"({stamp.get('path', '?')}) can no longer be read, so the "
                "verdict cannot be matched to a failure.")
    if live.get("sha256") != stamp.get("sha256"):
        return (
            "STALE CONTRACT AUDIT: this verdict was produced for a DIFFERENT "
            f"failure context (audited sha256={stamp.get('sha256')} at "
            f"mtime={stamp.get('mtime')}; the context on disk now is "
            f"sha256={live.get('sha256')} at mtime={live.get('mtime')}). The "
            "audit path is stage-derived and stable, so a previous failure's "
            "verdict sits exactly where this one is read from. Do NOT act on "
            "its category or recommended_action -- re-run the audit."
        )
    return ""


async def _run_top_level_contract_audit(
    *,
    stage: str,
    project_root: str,
    design_name: str,
    top_rtl_path: str,
    testbench_path: str,
    test_count: int,
    requirement_count: int = 0,
    sim_log: str = "",
    sim_log_path: str = "",
    block_rtl_paths: dict[str, str] | None = None,
    supported_actions: list[str] | None = None,
) -> dict:
    """Run contract audit for a top-level DV failure.

    The audit is deliberately pipeline-owned: validation/integration failures
    are first classified as TB/local RTL/top wiring/contract before the outer
    agent is interrupted.
    """
    from orchestrator.langchain.agents.contract_audit_agent import ContractAuditAgent

    root = Path(project_root)
    audit_dir = root / ".coresmith" / "contract_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    safe_stage = re.sub(r"[^a-zA-Z0-9_]+", "_", stage).strip("_") or "unknown"
    context_path = audit_dir / f"{safe_stage}_failure_context.json"
    output_path = audit_dir / f"{safe_stage}_contract_audit.json"

    context = {
        "stage": stage,
        "design_name": design_name,
        "top_rtl_path": top_rtl_path,
        "testbench_path": testbench_path,
        "test_count": test_count,
        "requirement_count": requirement_count,
        "sim_log_tail": sim_log[-12000:],
        "sim_log_path": sim_log_path,
        "block_rtl_paths": block_rtl_paths or {},
        "reference_files": {
            "ers_json": str(root / ".coresmith" / "ers_spec.json"),
            "prd_json": str(root / ".coresmith" / "prd_spec.json"),
            "block_diagram": str(root / ".coresmith" / "block_diagram.json"),
            "integration_vcd": str(root / "sim_build" / "integration" / "dump.vcd"),
            "integration_wavekit_audit": str(
                root / "sim_build" / "integration" / "wavekit_audit.json"
            ),
        },
    }
    context_path.write_text(json.dumps(context, indent=2), encoding="utf-8")
    # Identity of THIS failure, captured before the audit runs. It is stamped
    # into the verdict below so every later reader can tell whether the audit
    # in front of it is about the failure in front of it.
    context_stamp = _context_fingerprint(str(context_path))
    call_start = _time.time()

    log(f"  [CONTRACT-AUDIT] Auditing {stage} failure...", YELLOW)
    agent = ContractAuditAgent(temperature=0.1)
    result = await agent.analyze(
        stage=stage,
        project_root=project_root,
        context_path=str(context_path),
        output_path=str(output_path),
    )

    # Stamp + persist. Done here rather than inside the agent so EVERY return
    # path is stamped, including the agent's own exception fallback.
    result[CONTRACT_AUDIT_STAMP_KEY] = context_stamp
    result["audited_at"] = round(call_start, 3)
    try:
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    except OSError:
        pass
    stale = contract_audit_staleness(result, str(context_path))

    write_graph_event(project_root, "Contract Audit", "graph_node_exit", {
        "stage": stage,
        "category": result.get("category", "UNKNOWN"),
        "contract_failure": result.get("contract_failure", False),
        "recommended_action": result.get("recommended_action", ""),
        "confidence": result.get("confidence", 0),
        "audit_path": str(output_path),
        "audited_context_sha256": context_stamp.get("sha256", ""),
        "stale": bool(stale),
    })
    log(
        "  [CONTRACT-AUDIT] "
        f"{result.get('category', 'UNKNOWN')} "
        f"action={result.get('recommended_action', 'ask_human')} "
        f"confidence={result.get('confidence', 0)} "
        f"ctx={context_stamp.get('sha256', '?')[:12]}",
        RED if result.get("contract_failure") else YELLOW,
    )
    if stale:
        log(f"  [CONTRACT-AUDIT] {stale}", RED)
    # run3-followups: the audit's SEMANTIC recommendation (e.g. revise_uarch,
    # forced by the agent for spec-level categories) may not be an offerable
    # resume action for the parked interrupt -- the operator was told to take
    # an action the resume endpoint rejects. Keep the semantic recommendation,
    # and add the closest OFFERABLE action plus how to use it.
    rec = str(result.get("recommended_action", "") or "")
    if supported_actions and rec and rec not in supported_actions:
        resume_action = ("retry" if "retry" in supported_actions
                         else supported_actions[0])
        result["recommended_resume_action"] = resume_action
        result["recommended_action_note"] = (
            f"'{rec}' is the audit's semantic recommendation but is not an "
            "offerable resume action for this interrupt "
            f"(offerable: {supported_actions}). Apply the fix at its source "
            f"(spec/uarch documents), then resume with '{resume_action}' -- "
            "regeneration will read the corrected documents."
        )
        log(
            f"  [CONTRACT-AUDIT] recommended '{rec}' is not an offerable "
            f"resume action; operator path: fix at source + '{resume_action}'",
            YELLOW,
        )
    result["audit_path"] = str(output_path)
    result["context_path"] = str(context_path)
    return result


def route_after_integration_dv(state: OrchestratorState) -> str:
    """Route after smoke/integration DV into ERS/KPI validation DV."""
    result = state.get("integration_dv_result") or {}
    if result.get("passed") is True:
        return "validation_dv"
    if result.get("pending_decision"):
        return "integration_dv_decision"
    if result.get("action_taken") in ("retry", "fix_rtl", "fix_tb"):
        return "integration_dv"
    return END


route_after_integration_dv.__edge_labels__ = {
    "validation_dv": "Validation DV",
    "integration_dv_decision": "Park for decision",
    "integration_dv": "Retry",
    END: "DONE",
}


# ---------------------------------------------------------------------------
# Node: validation_dv  (Lead Validation DV -- verifies ERS + KPIs)
# ---------------------------------------------------------------------------

async def validation_dv_node(state: OrchestratorState) -> dict:
    """Generate and run an ERS/KPI validation-level cocotb testbench.

    This stage follows smoke/integration DV. It validates measurable
    application intent preserved in the ERS and records requirement coverage.
    """
    pr = state.get("project_root", str(PROJECT_ROOT))
    integration_result = state.get("integration_result") or {}

    top_rtl_path = integration_result.get("top_rtl_path", "")
    design_name = integration_result.get("design_name", "chip_top")
    block_rtl_paths = integration_result.get("block_rtl_paths", {})

    write_graph_event(pr, "Validation DV", "graph_node_enter", {
        "design_name": design_name,
        "block_count": len(block_rtl_paths),
    })

    with _tracer.start_as_current_span("Validation DV") as span:
        span.set_attribute("design_name", design_name)

        # RTL ACCEPTANCE DV [dv-hardening-16] -- RTL-stage tier 2, runs FIRST:
        # native-speed (C++ Verilator) RTL-vs-GOLDEN on the FRD's mission-scale
        # acceptance cases. Integration DV's oracle is the composed MODEL, so
        # it can never see a model-vs-golden divergence; this tier is the only
        # place "does the chip work on real content" is answered before signoff
        # (armC: 8/8 integration tests passed while real frames were 21dB).
        # Honest-skip (recorded, non-blocking) when no artifact/tooling; a
        # divergence/fidelity break FAILS validation_dv with the case evidence.
        try:
            from orchestrator.langgraph.acceptance_dv import run_acceptance_dv

            _acc = await asyncio.to_thread(
                run_acceptance_dv, pr, top_rtl_path, block_rtl_paths,
            )
            span.set_attribute("acceptance_dv_passed", bool(_acc.get("passed")))
            span.set_attribute("acceptance_dv_skipped", bool(_acc.get("skipped")))
            write_graph_event(pr, "Validation DV", "acceptance_dv", {
                "passed": _acc.get("passed"), "skipped": _acc.get("skipped"),
                "reason": _acc.get("reason"),
            })
            if _acc.get("skipped"):
                log(f"  [ACCEPTANCE-DV] SKIPPED (honest): {_acc.get('reason')}",
                    YELLOW)
            elif _acc.get("passed"):
                log(f"  [ACCEPTANCE-DV] PASSED: {_acc.get('reason')}", GREEN)
            else:
                log(f"  [ACCEPTANCE-DV] FAILED: {_acc.get('reason')}", RED)
                return {"validation_dv_result": {
                    "passed": False,
                    "error": "RTL Acceptance DV failed: "
                             + str(_acc.get("reason")),
                    "phase": "acceptance_dv",
                    "acceptance_dv": {k: v for k, v in _acc.items()
                                      if k != "cases"},
                    "violations": _acc.get("violations", []),
                }}
        except Exception as _exc:  # noqa: BLE001 - never crash the node
            log(f"  [ACCEPTANCE-DV] gate error (skipped): {_exc}", YELLOW)

        if not top_rtl_path or not Path(top_rtl_path).exists():
            msg = "No top-level RTL available for Validation DV"
            log(f"  [VALIDATION-DV] FAILED -- {msg}", RED)
            return {"validation_dv_result": {
                "passed": False,
                "error": msg,
                "phase": "preflight",
                "aborted": True,
            }}

        if len(block_rtl_paths) < 1:
            msg = "No block RTL files available for Validation DV"
            log(f"  [VALIDATION-DV] FAILED -- {msg}", RED)
            return {"validation_dv_result": {
                "passed": False,
                "error": msg,
                "phase": "preflight",
                "aborted": True,
            }}

        ers_context, requirement_count = _load_ers_validation_context(pr)
        if not ers_context:
            msg = "No ERS found; Validation DV cannot verify requirements"
            log(f"  [VALIDATION-DV] FAILED -- {msg}", RED)
            return {"validation_dv_result": {
                "passed": False,
                "error": msg,
                "phase": "missing_ers",
                "aborted": True,
            }}

        # Section 3b: surface carried-forward defects (advisory-bypass
        # observations) into the validation-DV context so this authoritative
        # RTL-level check explicitly confirms each one is cleared -- an advisory
        # composition-gate bypass must not silently escape verification.
        _cfd = read_carried_forward_defects(pr)
        if _cfd:
            _cfd_lines = [
                "\n\n## CARRIED-FORWARD DEFECTS (advisory bypass -- MUST verify)",
                "An upstream ADVISORY gate proceeded past these QUANTIFIED "
                "mismatches without hard-blocking. Validation DV is the "
                "authoritative check: exercise the RTL so each is confirmed "
                "cleared (or fail with the evidence).",
            ]
            for d in _cfd:
                _cfd_lines.append(
                    f"- {d.get('gate','?')}/{d.get('kind','?')}: "
                    f"{d.get('violation_count',0)} violation(s)"
                    + (f", first at {d.get('first_divergence_block')}"
                       if d.get('first_divergence_block') else "")
                    + (f"; {d.get('detail')}" if d.get('detail') else "")
                    + (f"; UNMODELED: {d.get('unmodeled')}"
                       if d.get('unmodeled')
                       and d.get('unmodeled') != d.get('detail') else ""))
            ers_context = ers_context + "\n".join(_cfd_lines)
            log(f"  [VALIDATION-DV] {len(_cfd)} carried-forward defect(s) added "
                f"to validation context", YELLOW)

        connections, _ = await asyncio.to_thread(load_architecture_connections, pr)

        modules = {}
        for block_name, rtl_path in block_rtl_paths.items():
            # Parse the BLOCK's module, not whichever comes first in
            # the file: generated files declare internal stages ahead
            # of the block itself.
            mod = await asyncio.to_thread(
                parse_verilog_ports, rtl_path,
                module_for_block(rtl_path, block_name))
            if mod.name:
                modules[block_name] = mod

        previous_dv = state.get("validation_dv_result") or {}
        previous_action = previous_dv.get("action_taken", "")
        previous_tb_path = previous_dv.get("testbench_path", "")
        reuse_existing_tb = (
            previous_action in ("fix_rtl", "fix_tb")
            and previous_tb_path
            and Path(previous_tb_path).exists()
        )

        generation_error: Exception | None = None
        if reuse_existing_tb:
            log(
                "  [VALIDATION-DV] Reusing existing testbench after "
                f"{previous_action}: {previous_tb_path}",
                YELLOW,
            )
            tb_result = {
                "testbench_path": previous_tb_path,
                "tb_path": previous_tb_path,
                "test_count": previous_dv.get("test_count", 0),
            }
        else:
            log("  [VALIDATION-DV] Generating ERS/KPI validation testbench...", YELLOW)
            # Re-point validation DV to reference-equivalence when block-goldens
            # is on: assert the RTL's primary output == the reference impl's
            # output for the same stimulus. Flag-off -> reference_path/entry stay
            # "" so the prompt + generated TB are byte-identical to before.
            reference_path = ""
            reference_entry = ""
            try:
                from orchestrator.architecture import composition as _composition
                if _composition.block_goldens_enabled():
                    _ref = _composition.resolve_reference_implementation(pr)
                    if _ref:
                        reference_path = _ref
                        try:
                            from orchestrator.architecture.model_integration import (
                                _load_reference_module,
                            )
                            _ref_mod = _load_reference_module(_ref)
                            _fn, reference_entry = (
                                _composition.resolve_reference_entrypoint(
                                    pr, _ref_mod
                                )
                            )
                        except Exception:  # noqa: BLE001
                            reference_entry = ""
            except Exception:  # noqa: BLE001
                reference_path = ""
                reference_entry = ""
            try:
                tb_result = await generate_validation_testbench(
                    design_name=design_name,
                    top_rtl_path=top_rtl_path,
                    modules=modules,
                    connections=connections,
                    block_rtl_paths=block_rtl_paths,
                    ers_context=smart_truncate(ers_context, 30000),
                    prior_failure=_format_dv_retry_context(previous_dv),
                    reference_path=reference_path,
                    reference_entry=reference_entry,
                    parameter_table=_ers_parameter_table(pr),
                )
            except Exception as e:
                tb_result = None
                generation_error = e
        if not reuse_existing_tb and tb_result is None:
            if generation_error is None:
                generation_error = RuntimeError("Validation testbench generation returned no result")
            e = generation_error
            log(f"  [VALIDATION-DV] Testbench generation failed: {e}", RED)
            error_msg = f"Validation testbench generation failed: {e}"
            contract_audit = await _run_top_level_contract_audit(
                stage="validation_dv_generation",
                project_root=pr,
                design_name=design_name,
                top_rtl_path=top_rtl_path,
                testbench_path="",
                test_count=0,
                requirement_count=requirement_count,
                sim_log=error_msg,
                sim_log_path="",
                block_rtl_paths=block_rtl_paths,
                            supported_actions=["retry", "fix_rtl", "fix_tb", "abort"],
            )
            payload = {
                "type": "validation_dv_failure",
                "phase": "tb_generation",
                "design_name": design_name,
                "top_rtl_path": top_rtl_path,
                "testbench_path": "",
                "test_count": 0,
                "requirement_count": requirement_count,
                "sim_log": error_msg,
                "sim_log_path": "",
                "block_rtl_paths": block_rtl_paths,
                "contract_audit": contract_audit,
                "contract_audit_path": contract_audit.get("audit_path", ""),
                "supported_actions": [
                    "retry",
                    "fix_rtl",
                    "fix_tb",
                    "abort",
                ],
                "outer_agent_guidance": (
                    "Validation DV could not generate a usable cocotb "
                    "testbench for the measurable ERS/KPI requirements. "
                    "Diagnose whether the failure is missing ERS/KPI detail, "
                    "an invalid top-level contract, or a validation testbench "
                    "generation bug. Use action='fix_tb' when the validation "
                    "testbench prompt/generator needs repair, action='fix_rtl' "
                    "when RTL/top contracts must change, or action='retry' "
                    "after applying an external fix. Do not mark the pipeline "
                    "complete until Validation DV runs and verifies every ERS "
                    "requirement.\n\nContract audit result: "
                    f"{contract_audit.get('category', 'UNKNOWN')} -- "
                    f"{contract_audit.get('outer_agent_summary', '')}"
                ),
                "reference_files": {
                    "top_rtl": top_rtl_path,
                    "ers": str(Path(pr) / ".coresmith" / "ers_spec.json"),
                    "contract_audit": contract_audit.get("audit_path", ""),
                },
            }
            # run3-followups: park in validation_dv_decision instead of a tail
            # interrupt() (see integration_dv for the one-cycle-late defect).
            write_graph_event(pr, "Validation DV", "graph_node_exit", {
                "error": str(e),
                "phase": "tb_generation",
                "action": "pending_decision",
            })
            return {
                "validation_dv_result": {
                    "passed": False,
                    "pending_decision": True,
                    "interrupt_payload": payload,
                    "error": error_msg,
                    "phase": "tb_generation",
                    "requirement_count": requirement_count,
                    "test_count": 0,
                    "testbench_path": "",
                    "design_name": design_name,
                    "contract_audit": contract_audit,
                    "contract_audit_path": contract_audit.get("audit_path", ""),
                },
                "pipeline_done": False,
            }

        tb_path = tb_result.get("testbench_path", "")
        test_count = tb_result.get("test_count", 0)
        log(f"  [VALIDATION-DV] Generated ({test_count} tests): {tb_path}", GREEN)
        span.set_attribute("test_count", test_count)
        span.set_attribute("requirement_count", requirement_count)

        log("  [VALIDATION-DV] Running validation simulation...", YELLOW)
        sim_result = await asyncio.to_thread(
            run_integration_simulation,
            design_name, top_rtl_path, block_rtl_paths, tb_path,
            sim_scope="validation",
        )

        passed = sim_result.get("passed", False)
        sim_log = sim_result.get("log", "")

        # MAX-GEOMETRY gate (rung3-fixes-2): validation DV must exercise the
        # declared dimensional maxima, not just a directed small-geometry prefix
        # -- otherwise a geometry-dependent index/address-width truncation ships
        # verified. Flip to failed -> existing failure interrupt. Operator-reused
        # TBs are trusted.
        if passed and not reuse_existing_tb:
            _mg = _maxgeo_gate_verdict(pr, tb_path)
            if _mg is not None:
                passed = False
                sim_log = ((sim_log + "\n\n") if sim_log else "") + _mg["reason"]
                span.set_attribute("maxgeo_gate_failed", True)
                log("  [VALIDATION-DV] MAX-GEOMETRY gate FAILED -- flipping DV "
                    f"to failed: uncovered={_mg['uncovered_dims']}", RED)

        if passed:
            # Chip-top synthesizability gate (fix #5 + #2): pipeline_done is
            # integration_dv AND validation_dv AND chip-top synthesizable. The
            # run-B wall was the INTEGRATED encoder (un-synthesizable), not any
            # one block, and pipeline_done was True anyway. Runs even under
            # SKIP_SYNTH (PDK-free cell-explosion probe on the assembled top).
            _synth_ok, _synth_reason = _chip_top_synth_ok(
                pr, design_name, top_rtl_path, block_rtl_paths,
            )
            if not _synth_ok:
                log(f"\n{'='*60}", RED)
                log("  VALIDATION DV PASSED, but chip_top is NOT synthesizable "
                    "-- NOT pipeline_done", RED)
                log(f"  {_synth_reason}", RED)
                log(f"{'='*60}\n", RED)
                write_graph_event(pr, "Validation DV", "graph_node_exit", {
                    "passed": True,
                    "chip_top_synthesizable": False,
                    "reason": _synth_reason,
                })
                return {
                    "validation_dv_result": {
                        "passed": True,
                        "chip_top_synthesizable": False,
                        "synth_fail_reason": _synth_reason,
                        "test_count": test_count,
                        "design_name": design_name,
                    },
                    "pipeline_done": False,
                }

            # Tier-2 MEASURED die-area rollup (Deliverable 2): pipeline_done also
            # requires the whole chip to fit its die budget. Sum measured
            # per-block PPA area (+ macro area) and compare to the resolved cap.
            # No-ops when no cap resolves; fail-open on any gate error.
            try:
                _block_names = list((block_rtl_paths or {}).keys())
                _roll = _measured_die_rollup(pr, _block_names)
            except Exception as _rexc:  # noqa: BLE001
                _roll = None
                log(f"  [DIE-ROLLUP] measured rollup errored (fail-open): {_rexc}",
                    YELLOW)
            if _roll is not None and not _roll.has_cap:
                log("  [DIE-ROLLUP] no die-area budget resolvable -- chip area is "
                    "UN-CAPPED (set CORESMITH_DIE_BUDGET_MM2 or a PRD "
                    "max_die_area_mm2 to enable the rollup)", YELLOW)
            if _roll is not None and _roll.has_cap and not _roll.ok:
                log(f"\n{'='*60}", RED)
                log("  VALIDATION DV PASSED, but the chip does NOT fit its die "
                    "budget -- NOT pipeline_done", RED)
                log(f"  {_roll.reason}", RED)
                log(f"{'='*60}\n", RED)
                write_graph_event(pr, "Validation DV", "graph_node_exit", {
                    "passed": True, "die_budget_ok": False,
                    "die_total_mm2": round(_roll.total_um2 / 1e6, 4),
                    "die_budget_mm2": _roll.die_budget_mm2,
                    "reason": _roll.reason[:2000],
                })
                return {
                    "validation_dv_result": {
                        "passed": True,
                        "die_budget_ok": False,
                        "die_rollup_reason": _roll.reason,
                        "die_total_mm2": round(_roll.total_um2 / 1e6, 4),
                        "die_budget_mm2": _roll.die_budget_mm2,
                        "test_count": test_count,
                        "design_name": design_name,
                    },
                    "pipeline_done": False,
                }

            log(f"\n{'='*60}", GREEN)
            log("  VALIDATION DV PASSED", GREEN)
            log(f"  {test_count} tests, ERS requirements covered", GREEN)
            log(f"{'='*60}\n", GREEN)
            write_graph_event(pr, "Validation DV", "graph_node_exit", {
                "passed": True,
                "test_count": test_count,
                "requirement_count": requirement_count,
                "log_path": sim_result.get("log_path", ""),
                "chip_top_synthesizable": True,
            })
            _record_dv_row(
                pr, block=design_name, scope="validation", source="gate",
                passed=True, tests_passed=test_count, tests_total=test_count,
                log_path=sim_result.get("log_path", ""),
                detail="validation_dv passed",
            )
            return {
                "validation_dv_result": {
                    "passed": True,
                    "test_count": test_count,
                    "requirement_count": requirement_count,
                    "testbench_path": tb_path,
                    "sim_log_path": sim_result.get("log_path", ""),
                    "design_name": design_name,
                    "chip_top_synthesizable": True,
                },
                "pipeline_done": True,
            }

        log("  [VALIDATION-DV] FAILED", RED)
        for line in sim_log.split("\n")[-10:]:
            if line.strip():
                log(f"    {line.strip()}", RED)

        contract_audit = await _run_top_level_contract_audit(
            stage="validation_dv",
            project_root=pr,
            design_name=design_name,
            top_rtl_path=top_rtl_path,
            testbench_path=tb_path,
            test_count=test_count,
            requirement_count=requirement_count,
            sim_log=sim_log,
            sim_log_path=sim_result.get("log_path", ""),
            block_rtl_paths=block_rtl_paths,
                    supported_actions=["retry", "fix_rtl", "fix_tb", "abort"],
        )
        _record_dv_row(
            pr, block=design_name, scope="validation", source="gate", passed=False,
            first_divergence=contract_audit.get("first_divergence"),
            detail=str(contract_audit.get("category", ""))[:200],
            log_path=sim_result.get("log_path", ""),
        )

        payload = {
            "type": "validation_dv_failure",
            "design_name": design_name,
            "top_rtl_path": top_rtl_path,
            "testbench_path": tb_path,
            "test_count": test_count,
            "requirement_count": requirement_count,
            "sim_log": sim_log[-3000:],
            "sim_log_path": sim_result.get("log_path", ""),
            "block_rtl_paths": block_rtl_paths,
            "contract_audit": contract_audit,
            "contract_audit_path": contract_audit.get("audit_path", ""),
            "supported_actions": [
                "retry",
                "fix_rtl",
                "fix_tb",
                "abort",
            ],
            "outer_agent_guidance": (
                "Validation DV failed after smoke/integration DV passed. "
                "Diagnose whether the failure is a real ERS/KPI miss, an RTL "
                "bug, or an over/under-constrained validation testbench. Fix "
                "RTL with action='fix_rtl' or fix the generated validation "
                "testbench with action='fix_tb'. Do not skip this stage unless "
                "the pipeline is explicitly configured to permit validation "
                "skips.\n\nContract audit result: "
                f"{contract_audit.get('category', 'UNKNOWN')} -- "
                f"{contract_audit.get('outer_agent_summary', '')}"
            ),
            "reference_files": {
                "top_rtl": top_rtl_path,
                "testbench": tb_path,
                "sim_log": sim_result.get("log_path", ""),
                "ers": str(Path(pr) / ".coresmith" / "ers_spec.json"),
                "contract_audit": contract_audit.get("audit_path", ""),
            },
        }

        if os.environ.get("CORESMITH_ALLOW_SKIP_VALIDATION_DV", "").lower() in (
            "1",
            "true",
            "yes",
        ):
            payload["supported_actions"].insert(-1, "skip")

        # run3-followups: park in validation_dv_decision instead of a tail
        # interrupt() (see integration_dv for the one-cycle-late defect).
        write_graph_event(pr, "Validation DV", "graph_node_exit", {
            "action": "pending_decision",
            "passed": False,
            "test_count": test_count,
            "requirement_count": requirement_count,
        })
        return {
            "validation_dv_result": {
                "passed": False,
                "pending_decision": True,
                "interrupt_payload": payload,
                "test_count": test_count,
                "requirement_count": requirement_count,
                "testbench_path": tb_path,
                "sim_log_path": sim_result.get("log_path", ""),
                "design_name": design_name,
                "contract_audit": contract_audit,
                "contract_audit_path": contract_audit.get("audit_path", ""),
            },
            "pipeline_done": False,
        }


async def validation_dv_decision_node(state: OrchestratorState) -> dict:
    """Consume the operator's decision for a parked validation-DV failure.

    Same split as ``integration_dv_decision_node`` (run3-followups): the
    interrupt lives in its own node so the resume response is consumed
    immediately instead of one regeneration cycle late."""
    pr = state["project_root"]
    dv = dict(state.get("validation_dv_result") or {})
    payload = dv.get("interrupt_payload") or {
        "type": "validation_dv_failure",
        "supported_actions": ["retry", "fix_rtl", "fix_tb", "abort"],
    }
    response = interrupt(payload) or {}
    action = response.get("action", "abort")
    write_graph_event(pr, "Validation DV", "graph_node_exit", {
        "action": action,
        "passed": False,
        "phase": dv.get("phase", "simulation"),
        "test_count": dv.get("test_count", 0),
        "requirement_count": dv.get("requirement_count", 0),
    })

    dv_result = dict(dv)
    dv_result.pop("interrupt_payload", None)
    dv_result["pending_decision"] = False
    dv_result["action_taken"] = action

    if action == "skip":
        dv_result["skipped_by_user"] = True
        log("  [VALIDATION-DV] Skipped by explicit configuration", YELLOW)
    elif action == "abort":
        dv_result["aborted"] = True
        log("  [VALIDATION-DV] Aborted", RED)
    elif action in ("retry", "fix_rtl", "fix_tb"):
        fix_desc = response.get("rtl_fix_description", "")
        log(
            f"  [VALIDATION-DV] Fix applied (action={action}): "
            f"{fix_desc or '(no description provided)'}",
            GREEN,
        )
        dv_result["fix_applied"] = fix_desc

    return {
        "validation_dv_result": dv_result,
        "pipeline_done": False,
        "pipeline_aborted": action == "abort",
    }


def route_after_validation_dv_decision(state: OrchestratorState) -> str:
    """Route the operator's decision back into validation DV or terminate."""
    result = state.get("validation_dv_result") or {}
    if result.get("action_taken") in ("retry", "fix_rtl", "fix_tb"):
        return "validation_dv"
    return END


route_after_validation_dv_decision.__edge_labels__ = {
    "validation_dv": "RETRY / FIX",
    END: "DONE",
}


def route_after_validation_dv(state: OrchestratorState) -> str:
    """Route after validation DV: terminal frontend pipeline."""
    result = state.get("validation_dv_result") or {}
    if result.get("pending_decision"):
        return "validation_dv_decision"
    if result.get("action_taken") in ("retry", "fix_rtl", "fix_tb"):
        return "validation_dv"
    return END


route_after_validation_dv.__edge_labels__ = {
    "validation_dv_decision": "Park for decision",
    "validation_dv": "Retry",
    END: "DONE",
}


# ---------------------------------------------------------------------------
# Node: final_report  (deterministic signoff scorecard -- runs before END)
# ---------------------------------------------------------------------------

async def final_report_node(state: OrchestratorState) -> dict:
    """Aggregate the run's recorded facts into a signoff scorecard.

    Runs at the terminal of the pipeline (after validation_dv / integration_dv /
    pipeline_complete, immediately before END). DETERMINISTIC: it reads the
    persisted DV verdicts (scoreboard ``dv_results``), line/FSM coverage
    (per-block ``coverage.json`` + ``coverage_results``), and PPA/Fmax
    (``ppa_history`` + pre-layout WNS vs the target clock) and writes
    ``final_report.json`` + a human-readable ``final_report.md`` (the scorecard)
    into the run root. No LLM, no gate -- purely a verification-traceability
    artifact. Never raises: a report failure must not fail the pipeline.
    """
    pr = _pr(state)
    try:
        from orchestrator.langgraph.final_report import (
            build_final_report,
            render_markdown,
        )
        report = build_final_report(dict(state), pr, scoreboard=_scoreboard(pr))
        md = render_markdown(report)
        root = Path(pr)
        root.mkdir(parents=True, exist_ok=True)
        (root / "final_report.json").write_text(json.dumps(report, indent=2))
        (root / "final_report.md").write_text(md)
        sign = report.get("signoff", {})
        log(f"\n{'='*60}", CYAN)
        log(f"  SIGNOFF SCORECARD: {sign.get('status')} -- "
            f"{sign.get('blocks_passed')}/{sign.get('blocks_total')} blocks, "
            f"{sign.get('testbenches_run')} testbenches, "
            f"cov(min) {sign.get('coverage_min_pct')}%, "
            f"Fmax {sign.get('top_fmax_mhz')} MHz", CYAN)
        log(f"  wrote {root / 'final_report.md'}", CYAN)
        log(f"{'='*60}\n", CYAN)
        write_graph_event(pr, "Final Report", "graph_node_exit", {
            "status": sign.get("status"),
            "blocks_passed": sign.get("blocks_passed"),
            "blocks_total": sign.get("blocks_total"),
            "testbenches_run": sign.get("testbenches_run"),
            "coverage_min_pct": sign.get("coverage_min_pct"),
            "top_fmax_mhz": sign.get("top_fmax_mhz"),
            "report_path": str(root / "final_report.json"),
        })
        return {"final_report": report}
    except Exception as _e:  # noqa: BLE001 - never fail the run on the report
        log(f"  [FINAL-REPORT] skipped (aggregation error): {_e}", YELLOW)
        return {}


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_block_subgraph_compiled(checkpointer=None):
    """Build and compile the block lifecycle subgraph standalone.

    Used by the graph introspection / web UI visualizer so the frontend
    graph view shows the full block lifecycle pipeline (10 nodes) rather
    than the thin orchestrator wrapper (4 nodes).
    """
    return build_block_subgraph().compile(checkpointer=checkpointer)


def build_pipeline_graph(checkpointer=None):
    """Build and compile the orchestrator pipeline graph.

    The orchestrator fans out blocks within each tier for parallel
    execution via ``Send()``.  Each block runs through the full block
    lifecycle subgraph autonomously.

    Args:
        checkpointer: LangGraph checkpointer for state persistence.
            Use ``MemorySaver`` for tests, ``AsyncSqliteSaver`` for
            production.

    Returns:
        Compiled StateGraph ready for ``ainvoke`` / ``astream``.

    Topology is gated by ``CORESMITH_BLOCK_GOLDENS``. When OFF (default) the
    historical SINGLE-PASS graph is built, byte-identical node/edge set to
    before the two-pass restructure (the ``model_integration`` no-op node still
    sits after ``integration_check``; no ``uarch_integration_gate`` /
    ``begin_rtl_pass`` / ``write_contract_request`` nodes). When ON the TWO-PASS
    graph is built: pass 1 (spec+model) -> µarch gate -> pass 2 (RTL+DV+synth).
    """
    from orchestrator.architecture import composition as _composition
    two_pass = _composition.block_goldens_enabled()

    block_subgraph = build_block_subgraph(two_pass=two_pass).compile()

    orchestrator = StateGraph(OrchestratorState)

    # Nodes (shared by both topologies)
    orchestrator.add_node("init_tier", init_tier_node)
    orchestrator.add_node("process_block", block_subgraph)
    orchestrator.add_node("integration_review", integration_review_node)
    orchestrator.add_node("advance_tier", advance_tier_node)
    orchestrator.add_node("pipeline_complete", pipeline_complete_node)
    orchestrator.add_node("integration_check", integration_check_node)
    orchestrator.add_node("integration_dv", integration_dv_node)
    # run3-followups: interrupts live in dedicated decision nodes so a resume
    # re-executes only the interrupt() call -- the big DV nodes never replay a
    # default cycle over an operator's response (the one-cycle-late defect).
    orchestrator.add_node(
        "integration_dv_decision", integration_dv_decision_node)
    orchestrator.add_node("validation_dv", validation_dv_node)
    orchestrator.add_node("validation_dv_decision", validation_dv_decision_node)
    # Deterministic signoff scorecard: the single pre-END funnel for every
    # GENUINE terminal (validation_dv done, integration_dv terminal-fail,
    # pipeline_complete abort). It does NOT sit on the interrupt()-based
    # suspend/resume exits (integration_check), which are not run completions.
    orchestrator.add_node("final_report", final_report_node)
    orchestrator.add_edge("final_report", END)

    # Edges (shared)
    orchestrator.add_edge(START, "init_tier")
    orchestrator.add_conditional_edges("init_tier", fan_out_tier)
    orchestrator.add_edge("process_block", "integration_review")
    orchestrator.add_conditional_edges("integration_review", route_after_integration_review)
    orchestrator.add_conditional_edges(
        "pipeline_complete",
        lambda s: (
            END if _pipeline_complete_route(s) == "end"
            else _pipeline_complete_route(s)
        ),
        {
            # Abort/done terminal -> scorecard -> END (was END directly).
            END: "final_report",
            "init_tier": "init_tier",
            "integration_check": "integration_check",
        },
    )
    # Map the routers' END sentinel to the final_report node (routers are
    # unchanged: they still return END; only the edge target moves).
    orchestrator.add_conditional_edges(
        "integration_dv", route_after_integration_dv,
        {
            "validation_dv": "validation_dv",
            "integration_dv": "integration_dv",
            "integration_dv_decision": "integration_dv_decision",
            END: "final_report",
        },
    )
    orchestrator.add_conditional_edges(
        "integration_dv_decision", route_after_integration_dv_decision,
        {
            "integration_dv": "integration_dv",
            END: "final_report",
        },
    )
    orchestrator.add_conditional_edges(
        "validation_dv", route_after_validation_dv,
        {
            "validation_dv": "validation_dv",
            "validation_dv_decision": "validation_dv_decision",
            END: "final_report",
        },
    )
    orchestrator.add_conditional_edges(
        "validation_dv_decision", route_after_validation_dv_decision,
        {
            "validation_dv": "validation_dv",
            END: "final_report",
        },
    )

    if not two_pass:
        # ---- SINGLE-PASS (flag off): byte-identical to before -------------
        orchestrator.add_node("model_integration", model_integration_node)
        orchestrator.add_conditional_edges("advance_tier", route_next_tier)
        orchestrator.add_conditional_edges("integration_check", route_after_integration)
        orchestrator.add_conditional_edges(
            "model_integration", route_after_model_integration
        )
    else:
        # ---- TWO-PASS (flag on): µarch gate between the two fan-outs -------
        orchestrator.add_node(
            "uarch_integration_gate", uarch_integration_gate_node
        )
        orchestrator.add_node("begin_rtl_pass", begin_rtl_pass_node)
        orchestrator.add_node(
            "write_contract_request", write_contract_request_node
        )
        # advance_tier -> {init_tier | uarch_integration_gate | pipeline_complete}
        orchestrator.add_conditional_edges(
            "advance_tier",
            route_next_tier,
            {
                "init_tier": "init_tier",
                "uarch_integration_gate": "uarch_integration_gate",
                "pipeline_complete": "pipeline_complete",
            },
        )
        # µarch gate -> {begin_rtl_pass | init_tier | write_contract_request | END}
        orchestrator.add_conditional_edges(
            "uarch_integration_gate",
            route_after_uarch_gate,
            {
                "begin_rtl_pass": "begin_rtl_pass",
                "init_tier": "init_tier",
                "write_contract_request": "write_contract_request",
                END: END,
            },
        )
        orchestrator.add_edge("begin_rtl_pass", "init_tier")
        # write_contract_request -> {init_tier (bounded re-spec) | END (exhausted)}
        orchestrator.add_conditional_edges(
            "write_contract_request",
            route_after_write_contract_request,
            {
                "init_tier": "init_tier",
                END: END,
            },
        )
        # Pass 2 post-integration_check goes straight to DV (gate already ran).
        orchestrator.add_conditional_edges(
            "integration_check",
            route_after_integration,
            {END: END, "integration_dv": "integration_dv"},
        )

    return orchestrator.compile(checkpointer=checkpointer)
