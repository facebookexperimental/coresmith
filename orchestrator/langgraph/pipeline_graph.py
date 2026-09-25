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

from langgraph.func import task as _durable_task
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

    Per-block lifecycle state (constraints, diagnoses, attempt history, best
    result) lives in the project database (.coresmith/project.sqlite), which
    regenerates read-only JSON views of it under .coresmith/blocks/<block>/.
    The latest error context stays a plain file the agents read:
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


    # Routing-only flags (no content -- agents read/write disk directly) ────
    uarch_approved: bool
    lint_clean: bool
    sim_passed: bool
    synth_success: bool
    synth_gate_count: int
    ppa_ok: bool | None        # deterministic PPA gate verdict (None = not run)
    ppa_reasons: list             # human-readable budget-divergence reasons
    timing_ok: bool | None        # measured WNS >= 0 (None = not measured)
    timing_required: bool        # missing required timing cannot complete a block
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
    reuse_spec: bool  # targeted revise / single-context uArch: implement the
                      # on-disk spec as-is unless feedback is pending for
                      # this block (then revise it per block, as before)

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
    # Checkpointed result of the model-backed review. The decision node may
    # replay after interrupt(), so it must not invoke the reviewer itself.
    integration_review_bundle: Annotated[dict | None, _last]
    # Targeted revise plan from integration_review: {block: reuse_spec}. Only
    # these blocks re-enter the tier on a revise; None = normal entry.
    integration_approved_specs: Annotated[dict | None, _last]
    revise_blocks: Annotated[dict | None, _last]

    # Integration check results ────────────────────────────────────────────
    integration_result: dict | None  # set by integration_check node

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


async def _park_conformance_unrepairable(state: BlockState, block_name: str,
                                         record: dict, failures: int) -> dict:
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
    # WP-74: the in-graph chip lead decides first (it may edit the block RTL
    # or amend .coresmith/interface_contracts.json and answer `retry`); the
    # caller re-checks conformance once and parks for a human only if the
    # block still deviates. Without a chip lead this is the old human park.
    resp = await _resolve_interrupt({
        "type": "contract_conformance_unrepairable",
        "block_name": block_name,
        "consecutive_failures": failures,
        "deviations": (record.get("deviations") or [])[:16],
        "renames_applied": record.get("renames") or {},
        "expected_ports": record.get("feedback", ""),
        "supported_actions": ["retry", "proceed", "abort"],
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
    return resp if isinstance(resp, dict) else {}


# Constraint sources that survive a fresh block lifecycle: chip-level DV
# decisions and operator rules are pinned precisely because the spec appends
# they mirror are destroyed by the per-tier re-spec.
_PERSISTENT_CONSTRAINT_SOURCES = ("chip_dv_revise", "chip_dv_fix", "human")


from orchestrator.state_store.project_db import open_project as _open_project  # noqa: E402

_PROJECT_DBS: dict = {}


def _db(project_root):
    """The project database for ``project_root`` (opened once per process)."""
    root = str(project_root)
    db = _PROJECT_DBS.get(root)
    if db is None:
        db = _open_project(root)
        _PROJECT_DBS[root] = db
    return db


def _stamp_engine_sha(project_root: str) -> None:
    """Stamp the engine git SHA into the project settings at run start and warn
    loudly on a mid-run change (a code hot-swap under the run). Never raises."""
    try:
        from orchestrator.utils import engine_git_sha
        live = engine_git_sha()
        db = _db(project_root)
        rec = db.get_setting("engine_sha") or ""
        if not rec:
            db.set_setting("engine_sha", live or "")
            db.set_setting("engine_sha_first_seen", str(_time.time()))
            log(f"  [ENGINE] git SHA {live or '(unknown)'} stamped for this run", CYAN)
            return
        if live and live != rec:
            log(f"  [ENGINE] !!! engine git SHA CHANGED mid-run: {rec} -> {live} "
                f"(a code hot-swap under this run; the final report records it)", RED)
            changes = json.loads(db.get_setting("engine_sha_changes", "[]") or "[]")
            changes.append({"from": rec, "to": live, "ts": _time.time()})
            db.set_setting("engine_sha_changes", json.dumps(changes))
            db.set_setting("engine_sha", live)
    except Exception:  # noqa: BLE001 - provenance is best-effort
        pass


def _persist_chip_fix_constraint(pr: str, action: str, fix_desc: str,
                                 response: dict, contract_audit: dict) -> None:
    """Persist a chip-level fix into the affected blocks' constraints so
    regeneration cannot silently resurrect the fixed bug."""
    blocks = (response.get("affected_blocks")
              or contract_audit.get("affected_blocks") or [])
    for name in blocks:
        try:
            _db(pr).add_constraint(
                name,
                f"Chip-level {action} applied during DV -- this behavior MUST be "
                f"preserved on any regeneration: {fix_desc}",
                source="chip_dv_fix", attempt=0)
        except Exception:  # noqa: BLE001
            continue


def _load_constraints_safe(project_root, block_name: str) -> list:
    """A block's accumulated constraints from the project database (never raises)."""
    try:
        return _db(project_root).constraints(block_name)
    except Exception:  # noqa: BLE001
        return []


def _stale_specs(project_root, block_names) -> list:
    """Blocks whose uArch spec predates a change to their contract edges."""
    try:
        return [{"block": b, "reason": "interface contract revised after the spec"}
                for b in _db(project_root).stale_spec_blocks(list(block_names))]
    except Exception:  # noqa: BLE001
        return []


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

    create_golden_model_wrapper(block_name, block.get("python_source", ""),
                                project_root=_pr(state))

    log(f"\n{'='*60}", CYAN)
    log(f"  Block: {block_name} | Tier {block.get('tier', '?')}", CYAN)
    log(f"{'='*60}", CYAN)

    write_graph_event(_pr(state), "Init Block", "graph_node_exit", {
        "block": block_name,
    })

    # Per-block working directory (logs, previous_error.txt and other agent-facing
    # artifacts). Lifecycle state lives in the project database.
    block_dir = Path(_pr(state)) / ".coresmith" / "blocks" / block_name
    block_dir.mkdir(parents=True, exist_ok=True)
    (block_dir / "previous_error.txt").write_text("")
    _bdb = _db(_pr(state))
    _round = _bdb.begin_round(block_name)
    if _round > 1:
        try:
            from orchestrator.langgraph.pipeline_helpers import archive_step_logs
            archive_step_logs(block_name, _round - 1)
        except Exception:  # noqa: BLE001 - archiving is best-effort
            pass
    # Constraints are an accumulating ledger: chip-level revise/fix pins and
    # operator rules survive a fresh lifecycle; per-lifecycle (debug-agent)
    # entries are dropped.
    _bdb.prune_constraints(block_name, _PERSISTENT_CONSTRAINT_SOURCES)

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


_CHIP_LEAD_TRIPPED = False


def _chip_lead_enabled() -> bool:
    """CORESMITH_ENABLE_CHIP_LEAD=1: interrupts resolved by the IN-GRAPH
    chip-lead agent instead of parking. Default-OFF: parks exactly as today."""
    return _env_truthy("CORESMITH_ENABLE_CHIP_LEAD")


def _chip_lead_ledger_path() -> Path:
    pr = os.environ.get("CORESMITH_PROJECT_ROOT", ".")
    return Path(pr) / ".coresmith" / "chip_lead" / "decisions.jsonl"


def _chip_lead_max_decisions() -> int:
    try:
        return int(os.environ.get("CORESMITH_CHIP_LEAD_MAX_DECISIONS", "50"))
    except ValueError:
        return 50



def _engine_checkout_guard() -> list[str]:
    """WP-25/WP-37: the engine checkout is read-only for the chip lead.

    After every chip-lead decision, look for modifications in the engine's own
    git checkout (observed: a chip lead "repaired the DV resolver" inside
    orchestrator/ and its tests). ``CORESMITH_ENGINE_READONLY``: ``0`` -> off,
    anything else -> detect. Returns the modified paths (staged, unstaged or
    untracked); the CALLER parks the run. Nothing is reverted: WP-25's
    `git checkout -- .` restored from the index and missed staged edits, and
    `git clean` could destroy legitimate operator files (review round 2). The
    real boundary is a checkout the worker cannot write; this is detection.
    Never raises.
    """
    mode = (os.environ.get("CORESMITH_ENGINE_READONLY", "1") or "1").strip().lower()
    if mode in ("0", "false", "no", "off"):
        return []
    import subprocess as _sp
    root = Path(__file__).resolve().parent.parent.parent
    try:
        r = _sp.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=normal"],
                    capture_output=True, text=True, timeout=30)
    except Exception:  # noqa: BLE001
        return []
    dirty = [ln[3:] for ln in (r.stdout or "").splitlines() if ln.strip()]
    if not dirty:
        return []
    log(f"  [CHIP-LEAD] ENGINE CHECKOUT MODIFIED ({len(dirty)} path(s)): "
        f"{dirty[:6]} -- the engine is read-only for the chip lead", RED)
    try:
        write_graph_event(os.environ.get("CORESMITH_PROJECT_ROOT", str(PROJECT_ROOT)),
                          "Chip Lead", "engine_modified", {"paths": dirty[:32], "mode": mode})
    except Exception:  # noqa: BLE001
        pass
    return dirty


def _engine_modified_payload(payload: dict, dirty: list) -> dict:
    """The parked payload for a chip-lead decision made from a modified engine
    checkout (WP-37): the decision is discarded, a human takes over."""
    parked = dict(payload or {})
    parked["engine_modified"] = list(dirty)[:32]
    parked["message"] = (
        "ENGINE CHECKOUT MODIFIED after a chip-lead decision "
        f"({len(dirty)} path(s): {list(dirty)[:6]}). The decision was discarded and "
        "the chip lead is tripped for this run. Restore the engine checkout "
        "(git status / git stash) and resume; nothing was reverted "
        "automatically.\n\n" + str(parked.get("message", "")))
    return parked


async def _resolve_interrupt(payload: dict) -> dict:
    """Park (default) or let the in-graph chip lead decide. Fail-safe: any
    chip-lead failure trips to parked interrupts for the process lifetime
    (re-armed on a fresh run start in init_tier_node)."""
    global _CHIP_LEAD_TRIPPED
    if not _chip_lead_enabled() or _CHIP_LEAD_TRIPPED:
        return interrupt(payload)

    ledger = _chip_lead_ledger_path()
    # Arm-U audit #10: two concurrently-parked blocks read the same ledger
    # length -> duplicate decision_index, budget drift. Serialize readers/
    # writers with an advisory lock on a sidecar lockfile.
    import fcntl as _fcntl
    ledger.parent.mkdir(parents=True, exist_ok=True)
    _lockf = open(ledger.parent / ".ledger.lock", "a+")
    _fcntl.flock(_lockf, _fcntl.LOCK_EX)
    try:
        prior = ([ln for ln in ledger.read_text().splitlines() if ln.strip()]
                 if ledger.exists() else [])
    finally:
        _fcntl.flock(_lockf, _fcntl.LOCK_UN)
        _lockf.close()
    if len(prior) >= _chip_lead_max_decisions():
        log(f"  [CHIP-LEAD] decision budget exhausted "
            f"({len(prior)}/{_chip_lead_max_decisions()}) -- parking", YELLOW)
        _CHIP_LEAD_TRIPPED = True
        return interrupt(payload)

    try:
        from orchestrator.langchain.agents.chip_lead_agent import ChipLeadAgent
        decision = await ChipLeadAgent().decide(
            payload=payload, prior_decisions=prior[-10:],
        )
    except Exception as exc:  # noqa: BLE001
        # Arm-F live finding: a single provider hard-timeout tripped the
        # fail-safe, and un-tripping needs a daemon restart. One fresh
        # retry before the trip absorbs one-off provider stalls; a second
        # consecutive failure still trips.
        log(f"  [CHIP-LEAD] agent failed ({exc}) -- one retry before "
            "tripping", YELLOW)
        try:
            decision = await ChipLeadAgent().decide(
                payload=payload, prior_decisions=prior[-10:],
            )
        except Exception as exc2:  # noqa: BLE001
            log(f"  [CHIP-LEAD] agent failed again ({exc2}) -- tripping "
                "to parked interrupts", RED)
            _CHIP_LEAD_TRIPPED = True
            return interrupt(payload)

    _dirty = _engine_checkout_guard()
    if _dirty:
        # WP-37: a decision made from a modified engine is invalid. Trip the
        # chip lead and park for a human; never revert automatically.
        _CHIP_LEAD_TRIPPED = True
        return interrupt(_engine_modified_payload(payload, _dirty))
    action = (decision or {}).get("action", "")
    supported = payload.get("supported_actions") or []
    if not action or (supported and action not in supported):
        # Arm-S retro: a single unsupported action ('revise' at a park that
        # only offered retry/fix_*) tripped the fail-safe and stranded the
        # run until a human restarted the daemon. Give the agent exactly one
        # corrective round with the violation spelled out before tripping.
        log(f"  [CHIP-LEAD] unsupported action {action!r} (supported: "
            f"{supported}) -- one corrective retry", YELLOW)
        try:
            retry_payload = dict(payload)
            retry_payload["action_correction"] = (
                f"Your previous answer used action={action!r}, which is NOT "
                f"in supported_actions={supported}. Answer again choosing "
                "strictly from that list."
            )
            decision = await ChipLeadAgent().decide(
                payload=retry_payload, prior_decisions=prior[-10:],
            )
        except Exception:  # noqa: BLE001
            decision = None
        action = (decision or {}).get("action", "")
        if not action or (supported and action not in supported):
            log(f"  [CHIP-LEAD] unsupported action {action!r} after "
                "correction -- tripping to parked interrupts", RED)
            _CHIP_LEAD_TRIPPED = True
            return interrupt(payload)

    ledger.parent.mkdir(parents=True, exist_ok=True)
    _lockf = open(ledger.parent / ".ledger.lock", "a+")
    _fcntl.flock(_lockf, _fcntl.LOCK_EX)
    try:
        prior = ([ln for ln in ledger.read_text().splitlines() if ln.strip()]
                 if ledger.exists() else prior)
        # Re-check the budget under the write lock: the check above happened
        # before two awaited agent calls, so N concurrently-parked branches
        # can each have passed it and overshoot the cap.
        if len(prior) >= _chip_lead_max_decisions():
            log(f"  [CHIP-LEAD] decision budget exhausted "
                f"({len(prior)}/{_chip_lead_max_decisions()}) -- parking",
                YELLOW)
            _CHIP_LEAD_TRIPPED = True
            return interrupt(payload)
        # The append and the event write stay inside the try: a raise here
        # with the flock held would block every other branch forever on the
        # (synchronous) flock syscall.
        with ledger.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "interrupt_type": payload.get("type", ""),
                "block_name": payload.get("block_name", ""),
                "action": action,
                "reasoning": decision.get("reasoning", ""),
                "ts": _time.time(),
            }) + "\n")
        write_graph_event(
            os.environ.get("CORESMITH_PROJECT_ROOT", "."), "Chip Lead",
            "chip_lead_decision",
            {"type": payload.get("type", ""), "action": action,
             "decision_index": len(prior) + 1},
        )
    finally:
        _fcntl.flock(_lockf, _fcntl.LOCK_UN)
        _lockf.close()
    log(f"  [CHIP-LEAD] {payload.get('type', '?')} -> {action} "
        f"({len(prior) + 1}/{_chip_lead_max_decisions()})", GREEN)
    return decision


def _is_content_free_revise(response: dict) -> bool:
    """True when a revise carries no substance (no block_actions, feedback,
    or reasoning) -- the reviewer-churn class that is safe to downgrade to
    approve. A chip-lead/human revise naming real findings is NOT churn."""
    return not (response.get("block_actions")
                or response.get("feedback")
                or response.get("reasoning"))


def _apply_revise_uarch(pr: str, response: dict, contract_audit: dict,
                        stage: str) -> list[str]:
    """Automate the operator playbook for a chip-level ``revise``: append the
    revision feedback to each affected block's uArch spec and drop
    best_result (the block re-enters through the targeted plan)."""
    feedback = (response.get("feedback")
                or contract_audit.get("suggested_fix") or "").strip()
    blocks = (response.get("affected_blocks")
              or contract_audit.get("affected_blocks") or [])
    applied: list[str] = []
    for name in blocks:
        spec = Path(pr) / "arch" / "uarch_specs" / f"{name}.md"
        if not spec.exists():
            continue
        try:
            with spec.open("a", encoding="utf-8") as fh:
                fh.write(
                    f"\n\n## {stage.upper()} REVISION FEEDBACK (MANDATORY)\n\n"
                    "Chip-level DV failed against this block's contract. This "
                    "section OVERRIDES any conflicting statement above:\n\n"
                    f"{feedback}\n"
                )
        except OSError:
            continue
        _db(pr).clear_result(name, "best")
        try:
            _bdir = Path(pr) / ".coresmith" / "blocks" / name
            _bdir.mkdir(parents=True, exist_ok=True)
            with (_bdir / "gate_feedback.txt").open("a", encoding="utf-8") as fh:
                fh.write(f"\n\n## {stage.upper()} REVISION (MANDATORY)\n\n{feedback}\n")
        except OSError:
            pass
        # Arm-U audit CRITICAL #2/#3: spec appends are destroyed by the
        # per-tier re-spec, so a correctly-fixed bug regressed 4h later.
        # constraints.json survives regeneration and is read by the spec/RTL/
        # TB generators -- pin the revision there too.
        try:
            _db(pr).add_constraint(
                name, f"{stage.upper()} REVISION (regeneration-proof): {feedback[:1500]}",
                source="chip_dv_revise", attempt=0)
        except Exception:  # noqa: BLE001
            pass
        applied.append(name)
    return applied


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
        return {"uarch_approved": False, "phase": "uarch"}

    # Targeted revise / single-context uArch: the spec on disk is the one to
    # implement (adopted from the integration review, or written by the
    # single-context author). Reuse it unless feedback is pending for THIS
    # block -- a chip-lead finding or a mem-price re-spec -- which revises
    # the spec per block exactly as before.
    _resp0 = state.get("human_response") or {}
    _fb_pending = (
        (Path(_pr(state)) / ".coresmith" / "blocks" / block_name
         / "gate_feedback.txt").exists()
        or (_resp0.get("action") == "revise" and bool(_resp0.get("feedback")))
    )
    if state.get("reuse_spec") and _pinned_spec.exists() and not _fb_pending:
        log(f"  [UARCH] {block_name}: implementing the on-disk spec as-is "
            "(targeted revise / single-context uArch) -- no per-block "
            "regeneration", YELLOW)
        write_graph_event(_pr(state), "Generate Uarch Spec", "spec_reused", {
            "block": block_name, "spec_path": str(_pinned_spec),
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
            # Consume ONE-SHOT prescriptions (the uarch_patch MICROARCH
            # REVISION channel has no other deleter) so they cannot steer
            # later tiers or runs. init_tier's OWN gate feedback must SURVIVE
            # this node: its presence is the 'gate implicated this block'
            # signal that gate_scoped_reuse_reason / review_uarch_spec_node
            # key on during a revise iteration, and init_tier clears it
            # itself on the next tier pass.
            if gate_feedback and not _is_own_gate_feedback(gate_fb_path):
                gate_fb_path.unlink(missing_ok=True)

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
    """CORESMITH_MEM_PRICE_MAX_REVISE (default 0 since WP-10c): the mem-price
    verdict is advisory -- priced, recorded and carried to the integration review
    summary, but it no longer re-specs a block on its own. Set >0 to restore
    the bounded auto re-spec loop."""
    try:
        return max(0, int(os.environ.get("CORESMITH_MEM_PRICE_MAX_REVISE", "0") or "0"))
    except ValueError:
        return 0


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


# Every uarch_feasibility blocking issue LEADS WITH ITS CATEGORY, e.g.
# "[area] storage exceeds the block budget" -> "area" (see the park payload's
# outer_agent_guidance, which documents the tag set to the chip-lead).
_FEAS_ISSUE_TAG_RE = re.compile(r"^\s*\[([A-Za-z_][A-Za-z0-9_-]*)\]")


def _feas_issue_categories(issues) -> list[str]:
    """Ordered, de-duplicated ``[tag]`` categories of blocking issues."""
    cats: list[str] = []
    for issue in issues or []:
        m = _FEAS_ISSUE_TAG_RE.match(str(issue))
        if m:
            tag = m.group(1).lower()
            if tag not in cats:
                cats.append(tag)
    return cats


def _feas_override_scope(project_root, block_name: str) -> dict | None:
    """SCOPE of the chip-lead's ``uarch_feasibility_override``, or None.

    The marker used to be a bare ``"1"`` whose mere EXISTENCE waived every
    budget gate for the block -- block-global, and therefore wrong: an override
    granted for an ``[interface]`` (or tooling) blocker silently forced the
    block past the deterministic mem_price gate and the post-synth budget gate
    as well, and labelled within-budget blocks "over-budget accepted". The
    marker is now JSON recording WHICH categories were actually overridden and
    against WHICH contract version, so each gate can ask whether the override
    is about IT.

    Returns the scope dict (``categories`` always present and lower-cased), or
    None when there is no marker or the override has EXPIRED -- a recorded
    ``contract_sha1`` that no longer matches this block's current contract
    means the lead accepted a different design than the one in front of us. A
    legacy ``"1"`` marker reads as an ``[area]`` override so pre-JSON runs keep
    exactly the behavior they had.
    """
    path = (Path(project_root) / ".coresmith" / "blocks" / block_name
            / "uarch_feasibility_override")
    try:
        raw = path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None
    try:
        scope = json.loads(raw)
    except ValueError:
        scope = None
    if not isinstance(scope, dict):
        return {"categories": ["area"], "legacy": True}
    scope["categories"] = [
        str(c).strip().lower()
        for c in (scope.get("categories") or []) if str(c).strip()
    ]
    ver = scope.get("contract_version")
    if ver:
        # Only a KNOWN, DIFFERENT contract version expires the override.
        current = _db(project_root).block_contract_version(block_name)
        if current and current != int(ver):
            return None
    return scope


def _feas_override_covers(project_root, block_name: str, category: str) -> bool:
    """True when a live override explicitly covers ``category`` (e.g. "area")."""
    scope = _feas_override_scope(project_root, block_name)
    return bool(scope) and category in (scope.get("categories") or [])


def _mem_price_gate_verdict(project_root: str, block_name: str) -> dict | None:
    """Tier-2 per-block memory-price gate at spec acceptance (Deliverable 1).

    Parses the spec's machine-readable ``# MEM`` manifest, prices each memory
    (real PDK area when the characterizer cache is warm, else the analytic
    flop-bits floor -- never blocking on a missing PDK), writes the priced
    ``mem_price.json`` ledger, and returns a re-spec request when the block busts
    its area budget or a single memory busts the sanity cap. Returns None to
    accept the spec. A spec that declares storage without a manifest is
    advisory under WP-11. With CORESMITH_MEM_MANIFEST_REQUIRED or a typed ERS
    parameters block, its ledger records unpriced storage, not an area excess.
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
    #
    # ONLY when the override actually covers [area], though. The marker records
    # the categories the lead waived; an [interface] / [capability] / tooling
    # override says nothing about this block's MEMORY budget, and honoring its
    # bare existence forced unrelated (and within-budget) blocks past this gate.
    _feas_scope = _feas_override_scope(project_root, block_name)
    if _feas_scope and "area" in (_feas_scope.get("categories") or []):
        _ovr_cats = ", ".join(
            f"[{c}]" for c in (_feas_scope.get("categories") or [])) or "[area]"
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
                        deferred_reason=(
                            "operator override (uarch_feasibility override "
                            f"scoped to {_ovr_cats}, covers [area]) -- "
                            "mem_price forced past, carried forward")))
        except Exception:  # noqa: BLE001 - never block on the ledger record
            pass
        log(f"  [MEM-PRICE] {block_name}: uarch_feasibility_override scoped to "
            f"{_ovr_cats} covers [area] -- FORCING PAST the mem_price gate "
            f"(deferred, carried forward) instead of re-entering the revise "
            f"loop", YELLOW)
        write_graph_event(project_root, "Review Uarch Spec",
                          "mem_price_override_deferred",
                          {"block": block_name,
                           "categories": _feas_scope.get("categories") or []})
        return None

    spec_text = spec_path.read_text(encoding="utf-8", errors="replace")
    decls = _mprice.parse_mem_manifest(spec_text)
    # Floor structural glue/wrapper/adapter blocks so a parse artifact (or a
    # tiny declared "~2 um2") can't hold a pin-mux to a sub-cell area budget.
    area_budget = floor_area_budget(parse_area_budget(spec_text), block_name, spec_text)

    # WP-11 makes an absent manifest advisory even in strict manifest mode.
    # Preserve the unpriced status for strict/new-schema runs without claiming
    # an over-budget measurement: no area has been evaluated in this branch.
    strict_manifest = _mprice.manifest_required() or _ers_parameters_block_present(project_root)
    if not decls:
        if _mprice.spec_declares_storage(spec_text):
            if strict_manifest:
                led = _mprice.format_ledger(
                    block_name, _mprice.MemPriceVerdict(ok=False), area_budget_um2=area_budget,
                    manifest_present=False, over_budget=False,
                    note="storage declared but no # MEM manifest; unpriced, budget not evaluated")
                _mprice.write_ledger(project_root, block_name, led)
                log(f"  [MEM-PRICE] {block_name}: storage declared but no # MEM "
                    "manifest -- advisory (WP-11); pricing skipped", YELLOW)
                return None
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

    if blocking_issues and _feas_override_scope(_pr(state), block_name):
        # A prior review round was overridden by the chip-lead; do not re-prompt
        # for the same (unchanged) spec on a two-pass re-entry. A genuine re-spec
        # (revise) rewrites the spec and clears intent by producing a new verdict.
        # An override recorded against a since-changed interface contract has
        # EXPIRED (_feas_override_scope returns None) -- re-prompt, because the
        # lead accepted a different design than the one in front of us.
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
        response = (await _resolve_interrupt(payload)) or {}
        action = response.get("action", "revise_interface")
        write_graph_event(_pr(state), "Review Uarch Spec",
                          "uarch_feasibility_resume",
                          {"block": block_name, "action": action})
        if action == "override":
            log(f"  [UARCH-FEAS] {block_name}: chip-lead OVERRIDE -- proceeding "
                f"despite reported blockers", YELLOW)
            try:
                _bdir.mkdir(parents=True, exist_ok=True)
                # SCOPED marker: record WHICH blocker categories were waived and
                # WHICH contract version they were waived against, so the
                # downstream budget gates can tell whether this override is
                # about them and so it expires when the contract moves under it.
                (_bdir / "uarch_feasibility_override").write_text(
                    json.dumps({
                        "gate": "uarch_feasibility",
                        "categories": _feas_issue_categories(blocking_issues),
                        "contract_version": _db(_pr(state)).block_contract_version(
                            block_name),
                        "ts": _time.time(),
                    }, indent=2),
                    encoding="utf-8")
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
        "contract_version": _db(project_root).block_contract_version(block_name) or "",
    }
    return {k: v for k, v in prov.items() if v}


async def generate_rtl_node(state: BlockState) -> dict:
    """Generate RTL, then run lint with local LLM fix loop.

    Disk-first: the agent reads all context from disk (uarch spec, ERS,
    constraints, previous error, golden model) and writes the Verilog
    to disk.  After generation, runs Verilator lint and attempts local
    LLM fixes before escalating to the diagnose lead.

    A block that reaches this node regenerates its RTL (WP-10b removed the
    sha1 regression guard and the skip-regen fast path: re-entry is targeted
    by the revise plan, so nothing reaches here that should be reused).
    """
    block = state["current_block"]
    block_name = block["name"]
    attempt = state["attempt"]
    rtl_path_obj = Path(state["project_root"]) / block["rtl_target"]

    write_graph_event(_pr(state), "Generate RTL", "graph_node_enter", {
        "block": block_name, "attempt": attempt,
    })

    with _tracer.start_as_current_span(
        f"Generate RTL [{block_name}] attempt {attempt}"
    ) as span:
        span.set_attribute("block_name", block_name)
        span.set_attribute("attempt", attempt)

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


async def generate_testbench_node(state: BlockState) -> dict:
    """Generate testbench, run simulation, and fix TB locally on failure.

    After generating (or reusing) the testbench, runs cocotb simulation.
    If simulation fails and the error looks like a testbench bug (import
    error, wrong port names, timing issues), calls an LLM to fix the TB
    and re-runs -- up to MAX_LOCAL_RETRIES times.

    Only escalates to the diagnose lead for failures that appear to be
    RTL bugs (wrong computation, stuck signals, etc.).
    """
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
            # PHASE = "conformance", NOT "sim": this gate FAILS BEFORE any
            # testbench is generated, so there is no simulation, no VCD and no
            # WaveKit audit. Labelling it "sim" sent 3/3 diagnosis agents
            # hunting for waveforms that cannot exist -- they then blamed the
            # engine for the missing artifacts.
            return {"tb_path": str(tb_path_obj), "sim_passed": False,
                    "phase": "conformance", "force_regen_tb": False,
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
        _renames: dict = {}
        _record_block_conformance(_pr(state), block_name, _conform)
        if _conform.get("ok"):
            log(f"  [CONFORM] {block_name}: ports match the contract "
                f"({_conform.get('checked_edges')} edge(s))", GREEN)
        else:
            _cf_n = _bump_conformance_failures(_pr(state), block_name)
            for _d in (_conform.get("deviations") or [])[:8]:
                log(f"  [CONFORM] {block_name}: {_d}", RED)
            log(f"  [CONFORM] {block_name}: RTL does NOT conform to the "
                f"interface contract "
                f"({_conform.get('after_missing')} missing, failure "
                f"{_cf_n}/{_CONFORMANCE_MAX_FAILURES}) -- FAILING before "
                f"TB/sim; a deviating block must not reach integration", RED)
            try:
                _bd = Path(_pr(state)) / ".coresmith" / "blocks" / block_name
                _bd.mkdir(parents=True, exist_ok=True)
                (_bd / "previous_error.txt").write_text(
                    "DETERMINISTIC CONTRACT-CONFORMANCE FAILURE (no sim was "
                    "run). The RTL does not expose the ports the FROZEN "
                    "interface contract declares. Regenerate the RTL "
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
            _lead_cleared = False
            if _cf_n >= _CONFORMANCE_MAX_FAILURES:
                # Cap: regeneration is not converging on the contract. The
                # chip lead decides (WP-74); a human park only if it cannot.
                _decision = await _park_conformance_unrepairable(
                    state, block_name, _conform, _cf_n)
                _reset_conformance_failures(_pr(state), block_name)
                _lead_action = str(_decision.get("action", ""))
                if _lead_action == "retry":
                    # Re-check once after the chip lead's edits (RTL or contract).
                    _conform = await asyncio.to_thread(
                        run_conformance_stage, _pr(state), block_name, rtl_path,
                        _sibs, str(tb_path_obj),
                    )
                    _record_block_conformance(_pr(state), block_name, _conform)
                    if _conform.get("ran") and _conform.get("ok"):
                        log(f"  [CONFORM] {block_name}: conforms after the chip "
                            f"lead's repair ({_conform.get('checked_edges')} "
                            "edge(s))", GREEN)
                        _lead_cleared = True
                    else:
                        for _d in (_conform.get("deviations") or [])[:8]:
                            log(f"  [CONFORM] {block_name}: {_d}", RED)
                        log(f"  [CONFORM] {block_name}: still deviating after the "
                            "retry -- the block fails this attempt; the next "
                            "entry re-checks (and parks again after two more "
                            "failures)", RED)
                elif _lead_action == "proceed":
                    log(f"  [CONFORM] {block_name}: chip lead chose `proceed` -- "
                        "the deviation stays recorded; integration decides "
                        "whether the chip top can be assembled", YELLOW)
                    _lead_cleared = True
                # any other answer (abort, none): the block fails this attempt
                # exactly as before; the node re-checks on its next entry.
            if not _lead_cleared:
                # PHASE = "conformance" (see the width gate above): pre-TB,
                # pre-sim, no waveform exists.
                return {"tb_path": str(tb_path_obj), "sim_passed": False,
                        "phase": "conformance", "force_regen_tb": False,
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
        if (not force_regen and state.get("preserve_testbench")
                and tb_path_obj.exists()):
            log(f"  [TB] Keeping the testbench (diagnosis: RTL-side fix): "
                f"{block['testbench']}", GREEN)
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
                run_simulation, block, rtl_path, tb_path, attempt,
                project_root=_pr(state),
            )

            if sim_result["passed"]:
                sim_passed = True
                log(f"  [SIM] PASSED"
                    f"{f' (after {sim_attempt} TB fix(es))' if sim_attempt > 0 else ''}",
                    GREEN)
                span.set_attribute("passed", True)
                span.set_attribute("tb_fixes", sim_attempt)

                _db(_pr(state)).set_result(block_name, "best", {
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
                })
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
            except Exception as exc:  # noqa: BLE001
                _ocheck = {"ok": False, "violation": {"detail": f"Oracle baseline check failed: {exc}"}}
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


    # Write sim error for diagnose if failed -- but ONLY when the sim loop
    # itself failed. The equiv / branch-parity / oracle gates above flip
    # sim_passed on a PASSING sim_result after writing their own actionable
    # report; overwriting it with the tail of a passing sim log leaves
    # diagnose with no failure evidence at all.
    if not sim_passed and sim_result and sim_result.get("passed") is not True:
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


def _ppa_tooling_waived(project_root: str, run_key: float | None = None) -> bool:
    """True once the operator has accepted an unmeasurable (yosys-absent) PPA
    gate for this run (A-Fix 2f) -- so we PARK at most once per run.

    The waiver file outlives the process, so ``run_key`` (the run's
    ``pipeline_run_start``) scopes it: a waiver recorded by an EARLIER run does
    not silently suppress the park in every future run. ``run_key=None`` keeps
    the unscoped behavior for callers with no run context."""
    p = _ppa_waivers_path(project_root)
    if not p.exists():
        return False
    try:
        data = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if not data.get("tooling_missing_accepted"):
        return False
    if run_key is None:
        return True
    return str(data.get("run", "")) == str(run_key)


def _record_ppa_tooling_waiver(project_root: str, block_name: str,
                               run_key: float | None = None) -> None:
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
    if data.get("run") != run_key:
        data["waived_at_block"] = block_name
    data.setdefault("waived_at_block", block_name)
    # Scope the waiver to THIS run (see _ppa_tooling_waived).
    data["run"] = run_key
    p.write_text(json.dumps(data, indent=2))


def _ppa_should_park_tooling_missing(
    project_root: str, ppa_ok: bool | None, ppa_meta: dict | None,
    run_key: float | None = None,
) -> bool:
    """Required timing always parks when unmeasured; retain the legacy
    strict-profile policy for an absent PDK-free probe tool.
    """
    if (ppa_meta or {}).get("timing_required") and (ppa_meta or {}).get("timing_unmeasured"):
        return True
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
    return not _ppa_tooling_waived(project_root, run_key)


def _park_ppa_unmeasurable(state: BlockState, block_name: str,
                         timing_error: str = "") -> None:
    """A-Fix 2f: PARK once/run on an unmeasurable (yosys-absent) PPA gate so the
    operator explicitly acknowledges it rather than the gate silently passing.

    On resume (action 'proceed') a waiver is recorded in
    ``.coresmith/ppa_waivers.json`` so the gate never re-asks this run. (The
    node re-executes on resume; the guard rechecks and this runs again, but
    ``interrupt`` then returns immediately and the waiver is written.)
    """
    if timing_error:
        # Resuming re-executes synthesis before reaching this interrupt again.
        # No waiver turns missing required timing into a successful block.
        block_dir = Path(_pr(state)) / ".coresmith" / "blocks" / block_name
        block_dir.mkdir(parents=True, exist_ok=True)
        (block_dir / "previous_error.txt").write_text(
            "Required timing is UNMEASURED: " + timing_error +
            "\nPreserve RTL. Repair the tool/constraints and retry synthesis.\n")
        interrupt({"type": "ppa_gate_unmeasurable", "block_name": block_name,
                   "supported_actions": ["retry_synth", "abort"],
                   "outer_agent_guidance": "Required STA is incomplete: " + timing_error +
                       ". Preserve RTL; fix the tool or constraints, then retry synthesis."})
        return
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
    _record_ppa_tooling_waiver(pr, block_name,
                               state.get("pipeline_run_start") or None)


def _evaluate_ppa_gate(
    project_root: str,
    block_name: str,
    rtl_path: str,
    synth_result: dict | None,
    *,
    require_gate_flag: bool = True,
) -> tuple[bool | None, list, dict]:
    """Measure PPA on the synthesized netlist, or probe without a PDK.

    Planning budgets are advisory. Required timing must have a finite measured
    verdict; missing timing parks synthesis without blaming or regenerating RTL.
    The returned metadata identifies the selected netlist and its measurements.
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
    # Hard FF ceiling + timing still gate. Scoped exactly like mem_price -- only
    # an [area] override waives an area/FF budget; an [interface] one does not.
    _budget_overridden = bool(
        ppa_honor_feas_override_enabled()
        and _feas_override_covers(project_root, block_name, "area")
    )

    # rung2 defect 2: seed meta with the parsed budgets so a persisted
    # ppa_history row carries them even on an early elaborate-fail flag.
    _meta: dict = {"budget_ff": ff_budget, "budget_area_um2": area_budget,
                   "timing_required": bool(synth_result)}
    if _budget_overridden:
        _meta["feas_override_deferred"] = True
        log(f"  [PPA] {block_name}: uarch_feasibility_override covers [area] -- "
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
        # WP-11: only a MEASURED timing failure is a rework reason; budget
        # divergence is advisory and must not poison a later retry's
        # previous_error.txt.
        _timing_failed = any(
            (c or {}).get("metric") == "wns_ns" and (c or {}).get("passed") is False
            for c in (checks or []))
        # Preserve a failed timing verdict even when WNS itself is absent. The
        # caller previously reconstructed timing_ok solely from wns_ns, turning
        # fail-closed "STA ran but produced no timing" into None (not measured),
        # which route_after_synth intentionally allows through.
        if _timing_failed:
            _meta["timing_verdict_failed"] = True
        (block_dir / ("previous_error.txt" if _timing_failed
                      else "ppa_advisory.txt")).write_text(_err)
        return False, reasons, dict(_meta)

    # HOT-PATCH (chip-lead): honor CORESMITH_SYNTH_TIMEOUT_S so the generic
    # synth probe uses the operator-set wall clock (600s here) instead of the
    # 300s default, which false-fails correct-but-slow synth on ARM (A1.Flex).
    # A timeout at 600s is then a GENUINE un-synthesizability signal, not box slowness.
    import os as _os_to
    _synth_timeout = int(_os_to.environ.get("CORESMITH_SYNTH_TIMEOUT_S", "300") or "300")
    probe: dict = {}
    # A mapped netlist is stronger evidence than another generic synthesis.
    # Keep the PDK-free probes only for runs without a synthesis result.
    if not synth_result:
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
    _sta_dir = Path(project_root) / "syn" / "output" / block_name
    if synth_result:
        sta = run_pre_layout_sta(
            synth_result.get("netlist_path", ""), synth_result.get("sdc_path", ""),
            synth_result.get("liberty_path", ""), block_name,
            report_path=str(_sta_dir / f"{block_name}_sta.rpt"),
        ) or {}
        _meta["sta_report_path"] = str(_sta_dir / f"{block_name}_sta.rpt")
        _meta["tns_ns"] = sta.get("tns_ns")
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
    # Compare measured pre-placement candidates. The deployment repairs mapped
    # FF loads that ABC cannot see. Keep timing, area and FFs from the same
    # selected netlist; inserted buffers are not free area.
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
            report_dir=_sta_dir, project_root=project_root,
            mapped_netlist=(synth_result or {}).get("netlist_path"),
            sdc_path=(synth_result or {}).get("sdc_path"),
        )
        if mf is not None:
            _meta["wns_ns_base_unbuffered"] = _eff_wns
            _meta["wns_ns_buffered"] = mf.get("buffered_wns_ns")
            _meta["fmax_mhz_buffered"] = mf.get("fmax_mhz")
            _meta["netlist_repair_status"] = mf.get("repair_status")
            if mf.get("detail"):
                _meta["sta_maxfanout_detail"] = mf["detail"]
                log(f"  [PPA] {block_name}: {mf['detail']}", YELLOW)
            if mf.get("sta_ok") and mf.get("wns_ns") is not None:
                if _eff_wns is None or mf["wns_ns"] > _eff_wns:
                    _eff_wns = mf["wns_ns"]
                    actual_ff = mf.get("ff_count", actual_ff)
                    # This measurement maps the full memories into the retained
                    # netlist, so do not add an estimated SRAM cost again.
                    actual_area = mf.get("chip_area_um2", actual_area)
                    _meta.update(ff=actual_ff, area_um2=actual_area,
                                 ppa_netlist_path=mf.get("netlist_path"),
                                 ppa_netlist_sha256=mf.get("netlist_sha256"),
                                 ppa_variant=mf.get("selected_variant"),
                                 cells=mf.get("cells", _meta.get("cells")),
                                 tns_ns=mf.get("tns_ns"))
                    if mf.get("report_path"):
                        _meta["sta_report_path"] = mf["report_path"]
                _eff_sta_error = None  # a real measurement rescued a base None/err
                _meta.pop("sta_error", None)
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
        # A failed repair leaves an unbuffered failing circuit, not an RTL
        # verdict. Preserve a passing baseline, otherwise retry the tool step.
        if (_meta["timing_required"] and (_eff_wns is None or _eff_wns < 0)
                and (mf is None or mf.get("repair_status") in {"failed", "unavailable"})):
            error = ((mf or {}).get("detail") or (mf or {}).get("sta_error")
                     or "Mapped-netlist repair is unavailable")
            _meta.update(timing_unmeasured=True, sta_error=error,
                         wns_ns_unbuffered=sta.get("wns_ns"), wns_ns=None)
            return None, [error], dict(_meta)
    import math
    if _meta["timing_required"] and (
            _eff_wns is None or not math.isfinite(float(_eff_wns))):
        _meta.update(wns_ns=None, timing_unmeasured=True,
                     sta_error=_eff_sta_error or "STA produced no finite slack")
        return None, [_meta["sta_error"]], dict(_meta)
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


def _resolve_probe_top(design_name: str, top_txt: str, project_root: str = "") -> str:
    """Resolve a chip probe only from the validated project manifest."""
    from orchestrator.harness.top_module import CandidateError, resolve_top
    if not project_root:
        raise CandidateError("Chip probe requires a project candidate manifest")
    return resolve_top(project_root)[0]


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
    from orchestrator.harness.top_module import CandidateError, candidate_for_inputs
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
    try:
        rec = candidate_for_inputs(project_root, top_rtl_path, block_rtl_paths)
        if rec["defines"] != "none" or rec["parameters"] != "none":
            raise CandidateError("Cell probe does not support the candidate configuration")
    except CandidateError as exc:
        return False, str(exc)
    if not _cell_gate_on():
        return True, ""
    deduped, _top_name = rec["sources"], rec["top_module"]
    _synth_timeout = int(os.environ.get("CORESMITH_SYNTH_TIMEOUT_S", "300") or "300")
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


def _container_leaf_map(project_root: str, block_names: list) -> dict:
    """Map each container block -> the set of listed blocks its RTL contains.

    Same detection as ``_container_block_names`` (whose result is this map's
    key set); the per-container membership is what lets the die rollup
    de-duplicate every container against ITS OWN leaves instead of collapsing
    all containers into one group and dropping the smaller ones.
    """
    inc_re = re.compile(r'^\s*`include\s+"([^"]+)"', re.MULTILINE)
    out: dict = {}
    for name in block_names:
        try:
            br = _db(project_root).result(name, "best")
            if not br:
                continue
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
        contained = {
            other for other in block_names
            if other != name and re.search(
                rf"\b{re.escape(other)}\s+(?:#|[a-zA-Z_]\w*\s*\()", text)
        }
        if contained:
            out[name] = contained
    return out


def _container_block_names(project_root: str, block_names: list) -> set:
    """Blocks whose measured per-block synth CONTAINS other listed blocks.

    An integration-top block built as a self-contained source (its RTL --
    directly or via one level of ``include`` -- instantiates other listed
    blocks) synthesizes FLAT: its measured area already includes the leaves.
    """
    return set(_container_leaf_map(project_root, block_names))


def _subsume_container_items(items: list, containers: set,
                             containment: dict | None = None) -> tuple:
    """Drop the double-count between flat container blocks and their leaves.

    Returns ``(items, note)``. With a ``containment`` map the resolution is
    RECURSIVE: a container's honest area is max(flat container,
    sum-of-its-direct-children), where a nested container child contributes
    its RESOLVED max -- never its flat area PLUS its own leaves, which the
    flat area already includes. Disjoint container tops resolve
    independently instead of the design being reported as only the biggest
    one. Without a ``containment`` map every container is assumed to cover
    every leaf (single group, legacy behavior).
    ``CORESMITH_DIE_ROLLUP_CONTAINER_DEDUP=0`` restores the legacy sum.
    """
    if (os.environ.get("CORESMITH_DIE_ROLLUP_CONTAINER_DEDUP", "1")
            or "1").strip() == "0":
        return items, ""
    cont = [i for i in items if i.name in containers]
    leaf = [i for i in items if i.name not in containers]
    if not cont or not leaf:
        return items, ""
    notes: list = []
    kept: list = []
    by_name = {i.name: i for i in items}
    order = {i.name: n for n, i in enumerate(items)}

    if not containment:
        # Legacy: one group -- the biggest container vs every leaf.
        top = max(cont, key=lambda i: i.area_um2)
        leaf_sum = sum(i.area_um2 for i in leaf)
        if leaf_sum >= top.area_um2:
            kept.extend(leaf)
            notes.append(
                f"container block(s) {[top.name]} subsumed by "
                f"their leaves (flat {top.area_um2:,.0f} um2 <= sum-of-leaves "
                f"{leaf_sum:,.0f} um2) -- not double-counted")
        else:
            kept.append(top)
            notes.append(
                f"leaf blocks subsumed by flat container {top.name} "
                f"({sorted(i.name for i in leaf)}: {top.area_um2:,.0f} "
                f"um2 >= sum-of-leaves {leaf_sum:,.0f} um2) -- not "
                f"double-counted")
        kept.sort(key=lambda i: order.get(i.name, 0))
        return kept, "; ".join(notes)

    def _reach(name: str) -> set:
        seen: set = set()
        stack = list(containment.get(name) or ())
        while stack:
            n = stack.pop()
            if n in seen or n == name:
                continue
            seen.add(n)
            stack.extend(containment.get(n) or ())
        return seen

    claimed: set = set()

    def _resolve(name: str, visiting: set) -> tuple:
        """(kept_items, resolved_area) for the subtree rooted at ``name``."""
        if name in visiting or name in claimed:
            return [], 0.0  # cycle, or already claimed by a larger root
        item = by_name.get(name)
        children = containment.get(name) or ()
        # Direct children only: a name also reachable through a SIBLING
        # nested container is counted inside that sibling's resolution.
        direct = [c for c in sorted(children)
                  if not any(c in _reach(c2) for c2 in children if c2 != c)]
        child_kept: list = []
        child_area = 0.0
        for c in direct:
            k, a = _resolve(c, visiting | {name})
            child_kept.extend(k)
            child_area += a
        if item is None:
            return child_kept, child_area  # unmeasured container: pass through
        if name not in containers or not child_kept:
            return [item], item.area_um2
        if child_area >= item.area_um2:
            notes.append(
                f"container block(s) {[name]} subsumed by "
                f"their leaves (flat {item.area_um2:,.0f} um2 <= sum-of-leaves "
                f"{child_area:,.0f} um2) -- not double-counted")
            return child_kept, child_area
        notes.append(
            f"leaf blocks subsumed by flat container {name} "
            f"({sorted(i.name for i in child_kept)}: {item.area_um2:,.0f} "
            f"um2 >= sum-of-leaves {child_area:,.0f} um2) -- not "
            f"double-counted")
        return [item], item.area_um2

    contained_anywhere: set = set()
    for c in cont:
        contained_anywhere |= _reach(c.name)
    roots = [c for c in cont if c.name not in contained_anywhere]
    if not roots:  # pathological containment cycle: legacy single group
        return _subsume_container_items(items, containers, None)
    for top in sorted(roots, key=lambda i: -i.area_um2):
        k, _a = _resolve(top.name, set())
        kept.extend(i for i in k if i.name not in {j.name for j in kept})
        claimed.add(top.name)
        claimed |= _reach(top.name)
    # Containers never re-surface as standalone leaves.
    claimed |= {c.name for c in cont}
    kept.extend(i for i in leaf if i.name not in claimed)
    kept.sort(key=lambda i: order.get(i.name, 0))
    return kept, "; ".join(notes)


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
    _cmap = _container_leaf_map(project_root, block_names)
    items, _subsume_note = _subsume_container_items(items, set(_cmap), _cmap)
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
        "status": res.status,
        # WP-15: not_run/disabled is neither pass nor fail (38/38 leaf
        # invocations logged passed=false while never having run).
        "passed": (res.status == _gs.STATUS_PASS
                   if res.status not in (_gs.STATUS_NOT_RUN, "disabled") else None),
        "cycles_compared": res.cycles_compared,
        "reason": res.reason,
    })

    if res.status == _gs.STATUS_PASS:
        log(f"  [GATE-SIM] PASS -- netlist matched the verified RTL for "
            f"{res.cycles_compared:,} cycles "
            f"({res.output_bits_compared:,} output bits)", GREEN)
        return (True, res.status, res.reason)

    if res.status in (_gs.STATUS_FAIL, _gs.STATUS_BOUNDED):
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
    block = state["current_block"]
    block_name = block["name"]

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
        local_attempt = 0
        for local_attempt in range(1 + MAX_LOCAL_RETRIES):
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

    # Measure the selected netlist; advisory budgets cannot skip required STA.
    ppa_ok, ppa_reasons, ppa_meta = (None, [], {})
    if synth_ok:
        ppa_ok, ppa_reasons, ppa_meta = _evaluate_ppa_gate(
            _pr(state), block_name, rtl_path, result,
        )
        if _ppa_should_park_tooling_missing(
                _pr(state), ppa_ok, ppa_meta,
                state.get("pipeline_run_start") or None):
            _park_ppa_unmeasurable(state, block_name,
                ppa_meta.get("sta_error", "") if ppa_meta.get("timing_unmeasured") else "")

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
        ff=ppa_meta.get("ff", (result or {}).get("ff_count")),
        cells=ppa_meta.get("cells", (result or {}).get("gate_count")),
        area_um2=ppa_meta.get("area_um2", (result or {}).get("chip_area_um2")),
        wns_ns=ppa_meta.get("wns_ns"),
        tns_ns=ppa_meta.get("tns_ns"),
        ppa_ok=ppa_ok, reasons=ppa_reasons or None,
        report_path=ppa_meta.get("sta_report_path", (result or {}).get("report_path", "")),
    )

    timing_ok = _timing_ok_from_ppa_meta(ppa_meta)
    return {
        "synth_success": synth_ok,
        "synth_gate_count": gate_count,
        "ppa_ok": ppa_ok,
        "ppa_reasons": ppa_reasons,
        "timing_ok": timing_ok,
        "timing_required": ppa_meta.get("timing_required", False),
        "gate_sim_ok": gate_sim_ok,
        "gate_sim_status": gate_sim_status,
        "gate_sim_reason": gate_sim_reason,
        "phase": "synth",
        "step_log_paths": existing_logs,
    }


# ---------------------------------------------------------------------------
# Node: diagnose
# ---------------------------------------------------------------------------

def _timing_ok_from_ppa_meta(ppa_meta: dict | None) -> bool | None:
    """Translate PPA timing metadata without losing fail-closed outcomes."""
    meta = ppa_meta or {}
    if meta.get("timing_verdict_failed"):
        return False
    wns = meta.get("wns_ns")
    return None if wns is None else bool(float(wns) >= 0.0)

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
                _db(_pr(state)).set_diagnosis(block_name, _wb, attempt=state["attempt"])
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
        _db(_pr(state)).record_attempt(block_name, {
            "attempt": state["attempt"],
            "error": error_log[:500],
            "category": "SIM_TIMEOUT",
        })
        _hist = _db(_pr(state)).attempt_history(block_name)
        # Bounded exactly like _route_decision's Rule -1: this short-circuit
        # returns before the router ever sees the category, so the cap has to
        # be enforced here or a genuinely-hung block retries forever (the sim
        # timeout also does not consume the attempt budget in decide_node).
        try:
            _sim_to_max = int(
                os.environ.get("CORESMITH_SIM_TIMEOUT_MAX_RETRIES", "4"))
        except ValueError:
            _sim_to_max = 4
        _to_count = sum(1 for _h in _hist
                        if _h.get("category") == "SIM_TIMEOUT")
        _to_exhausted = _to_count >= _sim_to_max
        if _to_exhausted:
            log(f"  [DIAGNOSE] SIM_TIMEOUT x{_to_count} (max {_sim_to_max}) -- "
                f"the extended cap is no longer helping, escalating", RED)
            _to_diag["escalate"] = True
            _to_diag["diagnosis"] += (
                f" Retried {_to_count} time(s) with an extended cap without "
                f"producing a verdict -- the block is hung or far too slow.")
        _db(_pr(state)).set_diagnosis(block_name, _to_diag, attempt=state["attempt"])
        write_graph_event(_pr(state), "Diagnose Failure", "graph_node_exit", {
            "block": block_name, "category": "SIM_TIMEOUT",
            "confidence": 0.0, "needs_human": False,
            "escalate": _to_exhausted,
        })
        return {"debug_action": "escalate" if _to_exhausted else "retry_rtl"}

    # Short-circuit: detect infrastructure failures (LLM timeout/crash)
    # and skip the debug LLM call which would likely also fail.
    # The markers must stay LLM-SPECIFIC: a bare "timed out" also matches
    # genuine tool/design failures that land here ("Verilator lint timed out",
    # "OpenSTA timed out after 300s" inside a fail-closed PPA reason), which
    # were then misfiled as infrastructure and never diagnosed.
    _INFRA_MARKERS = ("[ClaudeLLM error:", "claude CLI timed out",
                      "exit_code=-9", "circuit breaker open",
                      # WP-15: provider quota/outage text (codex)
                      "usage limit", "usage_limit", "rate limit",
                      "rate_limit_exceeded", "insufficient_quota")
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
        _db(_pr(state)).record_attempt(block_name, {
            "attempt": state["attempt"],
            "error": error_log[:500],
            "category": "INFRASTRUCTURE_ERROR",
        })
        history = _db(_pr(state)).attempt_history(block_name)
        _db(_pr(state)).set_diagnosis(block_name, infra_diag, attempt=state["attempt"])
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
            _db(_pr(state)).set_diagnosis(block_name, _sig_diag, attempt=state["attempt"])
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
        _db(_pr(state)).record_attempt(block_name, {
            "attempt": state["attempt"],
            "error": error_log[:500],
            "category": _fast_diag["category"],
        })
        history = _db(_pr(state)).attempt_history(block_name)
        _db(_pr(state)).set_diagnosis(block_name, _fast_diag, attempt=state["attempt"])
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

    _db(_pr(state)).set_diagnosis(block_name, diag, attempt=state["attempt"])

    # Route the structured diagnosis into previous_error.txt so the REGEN
    # (rtl_generator reads previous_error.txt, not diagnosis.json) gets the
    # actionable fix -- suggested_fix + per-constraint code snippets -- instead
    # of the raw tool-log tail it historically re-attacked blind.
    if _route_diagnosis_to_previous_error(block_dir, diag, error_log):
        log("  [DIAGNOSE] Routed actionable diagnosis -> previous_error.txt", GREEN)

    _db(_pr(state)).record_attempt(block_name, {
        "attempt": state["attempt"],
        "phase": phase,
        "error": error_log[:500],
        "category": category,
    })
    history = _db(_pr(state)).attempt_history(block_name)

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

def _infrastructure_streak(history: list[dict]) -> int:
    count = 0
    for row in reversed(history):
        category = row.get("category") or (row.get("diagnosis") or {}).get("category")
        if category != "INFRASTRUCTURE_ERROR":
            break
        count += 1
    return count


def _infrastructure_retry_cap() -> int:
    try:
        return max(1, int(os.environ.get("CORESMITH_INFRA_MAX_RETRIES", "6") or 6))
    except ValueError:
        return 6


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

    # Rule 0: Infrastructure errors get special handling. WP-16: an LLM
    # outage is not the block's fault -- retry (without consuming the
    # attempt budget, see route_decision_node) and only ask a human after
    # CORESMITH_INFRA_MAX_RETRIES consecutive failures. Asking after 2
    # meant calling the chip lead (also an LLM) during the same outage,
    # and once it answered it skipped a block over 3 'attempts'.
    if category == "INFRASTRUCTURE_ERROR":
        if _infrastructure_streak(attempt_history) >= _infrastructure_retry_cap():
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
    if phase == "conformance":
        # Deterministic pre-sim contract gate (port names / widths). The fix is
        # always in the RTL, and the exact expected names are already in
        # previous_error.txt.
        return "retry_rtl"
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
                _diag_cat = (_db(_pr(state)).diagnosis(block_name) or {}).get("category")
            except Exception:  # noqa: BLE001
                _diag_cat = None
            if _diag_cat == "INFRASTRUCTURE_ERROR":
                # WP-16: the LLM was unavailable; re-run the SAME attempt
                # after a real backoff. Budget is for design failures.
                _infra_n = 0
                try:
                    _infra_n = _infrastructure_streak(_db(_pr(state)).attempt_history(block_name) or [])
                except Exception:  # unreadable history cannot reset the retry cap
                    return {"debug_action": "ask_human"}
                if _infra_n >= _infrastructure_retry_cap():
                    return {"debug_action": "ask_human"}
                backoff_s = min(60 * (2 ** max(_infra_n - 1, 0)), 900)
                log(f"  [RETRY] INFRASTRUCTURE_ERROR -- re-running attempt "
                    f"{state['attempt']} after {backoff_s}s backoff (budget not "
                    f"consumed; consecutive infra failures: {_infra_n})", YELLOW)
                write_graph_event(_pr(state), "Route Decision", "graph_node_exit", {
                    "block": block_name, "decision": action,
                    "infra_retry": True, "backoff_s": backoff_s,
                })
                await asyncio.sleep(backoff_s)
                span.set_attribute("final_decision", action)
                return update
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

                diag = _db(_pr(state)).diagnosis(block_name) or {}
                if diag.get("category") == "INFRASTRUCTURE_ERROR":
                    backoff_s = min(30 * (2 ** (new_attempt - 1)), 120)
                    log(f"  [RETRY] Backing off {backoff_s}s after infra failure", YELLOW)
                    await asyncio.sleep(backoff_s)

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

    diag = _db(_pr(state)).diagnosis(block_name) or {}
    attempt_history = _db(_pr(state)).attempt_history(block_name)

    error_path = block_dir / "previous_error.txt"
    error_text = error_path.read_text() if error_path.exists() else ""

    constraints = _load_constraints_safe(_pr(state), block_name)

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

    response = await _resolve_interrupt(payload)

    write_graph_event(_pr(state), "Ask Human", "graph_node_exit", {
        "block": block_name, "action": response.get("action", "unknown"),
    })

    action = response.get("action", "abort")
    updated: dict = {"human_response": response}

    if action == "add_constraint" and response.get("constraint"):
        _db(_pr(state)).add_constraint(
            block_name, response["constraint"], source="human", attempt=state["attempt"])

    if action == "fix_rtl" and response.get("description"):
        _db(_pr(state)).add_constraint(
            block_name, f"Outer-agent RTL fix applied: {response['description']}",
            source="human", attempt=state["attempt"])

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
    human_resp = state.get("human_response") or {}
    is_skip = human_resp.get("action") == "skip"
    is_abort = human_resp.get("action") == "abort"
    is_escalate = state.get("debug_action") == "escalate"

    all_passed = (
        sim_passed and synth_success
        and state.get("timing_ok") is not False
        and (not state.get("timing_required") or state.get("timing_ok") is True)
        and not is_skip and not is_abort and not is_escalate
    )

    step_log_paths = dict(state.get("step_log_paths") or {})

    block_dir = Path(_pr(state)) / ".coresmith" / "blocks" / block_name
    constraints = _load_constraints_safe(_pr(state), block_name)

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
    if action == "revise":
        return "generate_uarch_spec"
    if action == "skip":
        return "block_done"
    return "generate_rtl"


route_after_uarch_review.__edge_labels__ = {
    "generate_rtl": "APPROVED",
    "generate_uarch_spec": "REVISE",
    "block_done": "SKIP",
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

    The PPA budget verdict (``ppa_ok``) is advisory since WP-10c: it is
    measured and reported but never routes a compiled, DV-passing block back
    to rework.

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
    # WP-11: MEASURED timing is a hard verdict (an external oracle); the
    # PPA budget verdict (``ppa_ok``) stays advisory (WP-10c).
    if state.get("timing_ok") is False:
        return "diagnose"
    # The unmeasurable-tool interrupt already ran; retain an incomplete result
    # if resumed without a measurement instead of sending RTL to diagnosis.
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

def build_block_subgraph():
    """Build the block lifecycle subgraph (uncompiled StateGraph).

    Contains the full lifecycle for a single block:
      init -> uarch spec -> review
        -> generate_rtl (with lint)
        -> generate_testbench (with sim + local TB fix)
        -> synthesize -> done

    Plus the diagnose/decide/retry failure loop, where decide routes
    directly back to generate_rtl (no intermediate increment node).

    Returns:
        Uncompiled ``StateGraph(BlockState)`` -- the caller compiles it
        (with or without a checkpointer) before adding it as a node.
    """
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
    """Completed blocks deduped by name, keeping the LAST entry so a retry overrides an earlier failure (``completed_blocks`` is append-only)."""
    seen: dict[str, dict] = {}
    for b in state.get("completed_blocks", []):
        if not isinstance(b, dict):
            continue
        name = b.get("name")
        if not name:
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
        response = await _resolve_interrupt({
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


# Opening sentence of every feedback string init_tier writes (see
# _gate_feedback_for_block) -- its provenance marker on disk.
_GATE_FEEDBACK_HEADER = "The composed-chip integration gate FAILED"


def _is_own_gate_feedback(path: Path) -> bool:
    """True when ``gate_feedback.txt`` was written by init_tier itself.

    Sibling nodes deliver their prescription through the SAME file (the
    uarch_patch-on-retry auto-apply writes it moments before routing back
    here), so the stale-feedback clearing below must not delete a message
    generate_uarch_spec has not read yet.
    """
    try:
        return path.read_text(
            encoding="utf-8", errors="replace").lstrip().startswith(
                _GATE_FEEDBACK_HEADER)
    except OSError:
        return False


async def _single_context_uarch_stage(
    pr: str, block_queue: list, tier_blocks: list, revise: dict | None,
) -> dict | None:
    """CORESMITH_UARCH_SINGLE_CONTEXT: one session authors every missing spec
    of the design (first entry) or revises the blocks a targeted revise asked
    to re-spec. Returns the updated revise plan (re-specced blocks flip to
    ``reuse_spec``) or None when nothing changed. A spec the session did not
    produce falls through to the per-block author as before."""
    from orchestrator.langgraph.pipeline_helpers import (
        generate_uarch_specs_single_context,
    )
    spec_dir = Path(pr) / "arch" / "uarch_specs"
    if revise:
        targets = [b for b in tier_blocks
                   if b["name"] in revise and not revise[b["name"]]]
    else:
        targets = [b for b in block_queue
                   if not (spec_dir / f"{b['name']}.md").exists()]
    if not targets:
        return None
    feedback: dict[str, str] = {}
    for b in targets:
        fb = Path(pr) / ".coresmith" / "blocks" / b["name"] / "gate_feedback.txt"
        if fb.exists():
            try:
                feedback[b["name"]] = fb.read_text(encoding="utf-8").strip()
            except OSError:
                pass
    names = [b["name"] for b in targets]
    log(f"  [UARCH] single-context uArch stage: {'revising' if revise else 'authoring'} "
        f"{len(names)} spec(s) in one session ({', '.join(names)})", YELLOW)
    write_graph_event(pr, "Generate Uarch Specs", "graph_node_enter", {
        "blocks": names, "revise": bool(revise),
    })
    try:
        result = await generate_uarch_specs_single_context(
            targets, feedback_by_block=feedback)
    except Exception as exc:  # noqa: BLE001 - fall through to per-block authors
        log(f"  [UARCH] single-context uArch stage FAILED ({exc}) -- blocks "
            "fall back to per-block spec generation", RED)
        write_graph_event(pr, "Generate Uarch Specs", "graph_node_exit", {
            "error": str(exc)[:500], "blocks": names,
        })
        return None
    written = [n for n in result.get("written", []) if n in names]
    missing = [n for n in names if n not in written]
    write_graph_event(pr, "Generate Uarch Specs", "graph_node_exit", {
        "written": written, "missing": missing,
    })
    log(f"  [UARCH] single-context uArch stage wrote {len(written)}/{len(names)} "
        f"spec(s)" + (f"; per-block fallback for {', '.join(missing)}" if missing else ""),
        GREEN if not missing else YELLOW)
    for n in written:
        fb = Path(pr) / ".coresmith" / "blocks" / n / "gate_feedback.txt"
        fb.unlink(missing_ok=True)  # consumed by the single-context revision
    if revise and written:
        return {**revise, **{n: True for n in written}}
    return None



def _retire_derived_integration_artifacts(project_root: str) -> list[str]:
    """Retire superseded assembler outputs when integration checking re-runs.

    Only files the deterministic assembler writes are moved: the assembled
    ``user_project_wrapper.v`` (when the persisted integration record says
    ``caravel_wrapper_assembled``) and ``user_project_wrapper_pads.v``. An
    LLM-authored or self-assembled top is left alone, as are all sources and
    dependencies of the current validated candidate. Returns the file names
    moved. Never raises.
    """
    import time as _time

    from orchestrator.harness.top_module import CandidateError, validated_candidate

    root = Path(project_root)
    int_dir = root / "rtl" / "integration"
    if not int_dir.is_dir():
        return []
    protected = set()
    try:
        candidate = validated_candidate(root)
        protected = {Path(p).resolve() for p in
                     [*candidate["sources"], *candidate["dependencies"]]}
    except CandidateError:
        pass
    assembled = False
    try:
        rec = json.loads((root / ".coresmith" / "integration_result.json").read_text())
        assembled = bool(rec.get("caravel_wrapper_assembled"))
    except (OSError, ValueError):
        assembled = False
    names = ["user_project_wrapper_pads.v"] + (["user_project_wrapper.v"] if assembled else [])
    moved: list[str] = []
    dest = int_dir / "_stale" / _time.strftime("%Y%m%dT%H%M%S")
    for n in names:
        f = int_dir / n
        if f.is_file() and f.resolve() not in protected:
            try:
                dest.mkdir(parents=True, exist_ok=True)
                f.rename(dest / n)
                moved.append(n)
            except OSError:
                pass
    return moved


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

    # Chip-lead trip re-arms on a FRESH run start (tier 0, nothing completed);
    # mid-run tier re-entries keep a tripped lead parked.
    global _CHIP_LEAD_TRIPPED
    if current_idx == 0 and not state.get("completed_blocks"):
        _CHIP_LEAD_TRIPPED = False

    tier = tier_list[current_idx]
    tier_blocks = [b for b in block_queue if b.get("tier", 1) == tier]

    # Targeted revise plan ({block: reuse_spec}) from integration_review or a
    # DV-failure revise. It may span tiers: skip the tiers with nothing to
    # redo; a plan naming no queued block voids itself (normal full entry).
    revise = state.get("revise_blocks") or None
    tier_idx_update = None
    plan_void = False
    if revise:
        _idx = current_idx
        while _idx < len(tier_list) and not any(
            b.get("name") in revise for b in block_queue
            if b.get("tier", 1) == tier_list[_idx]
        ):
            _idx += 1
        if _idx >= len(tier_list):
            log("  Targeted revise plan names no queued block -- normal "
                "tier entry", YELLOW)
            revise, plan_void = None, True
        elif _idx != current_idx:
            log(f"  Targeted revise: tier {tier} has nothing to redo -- "
                f"skipping to tier {tier_list[_idx]}", CYAN)
            current_idx = tier_idx_update = _idx
            tier = tier_list[current_idx]
            tier_blocks = [b for b in block_queue if b.get("tier", 1) == tier]
    # Section 7a: stamp the engine git SHA at run start + WARN in the daemon log
    # if it changes mid-run (a hot-swap that flipped behavior under the run).
    _stamp_engine_sha(pr)

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

    if revise:
        _mine = [b['name'] for b in tier_blocks if b['name'] in revise]
        log(f"  Targeted revise: re-entering {', '.join(_mine)} only", CYAN)
    revise_update = None
    if _uarch_single_context_enabled():
        revise_update = await _single_context_uarch_stage(
            pr, block_queue, tier_blocks, revise)

    write_graph_event(pr, "Init Tier", "graph_node_exit", {
        "tier": tier,
        "revise_blocks": revise_update if revise_update is not None else revise,
    })

    out = {"tier_list": tier_list}
    if revise_update is not None:
        out["revise_blocks"] = revise_update
    elif plan_void:
        out["revise_blocks"] = None
    if tier_idx_update is not None:
        out["current_tier_index"] = tier_idx_update
    # Publish the reduced queue + the retirement record so EVERY downstream
    # consumer agrees the block is deliberately absent rather than missing:
    # pipeline_complete / integration_check size `expected` off block_queue,
    # discover_block_rtl + missing_from work off the same set, and the daemon's
    # total_blocks / remaining_count stop waiting on it.
    if _queue_reduced:
        out["block_queue"] = block_queue
    if _retired_records:
        out["retired_blocks"] = _retired_records
    return out


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

    # Targeted revise: only the planned blocks re-enter; the rest keep the
    # completed result they already have (completed_blocks dedups by name).
    revise = state.get("revise_blocks") or None
    if revise:
        tier_blocks = [b for b in tier_blocks if b["name"] in revise]
    single_context = _uarch_single_context_enabled()

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
            "reuse_spec": (bool(revise[block["name"]]) if revise
                           else single_context),
        }))

    return sends


fan_out_tier.__edge_labels__ = {
    "process_block": "FAN OUT",
}


def _uarch_single_context_enabled() -> bool:
    """CORESMITH_UARCH_SINGLE_CONTEXT=1: the uArch stage runs in ONE agent
    session -- at the first tier entry one author writes every missing spec of
    the design (no per-block fan-out for the spec stage), and on a targeted
    revise one session revises the blocks the chip lead named. Blocks then
    implement the on-disk spec (``reuse_spec``); a block still re-specs on its
    own when feedback is pending for it (mem-price gate, spec-review revise).
    Default OFF = one author per block, as before."""
    return _env_truthy("CORESMITH_UARCH_SINGLE_CONTEXT")


def _revise_named_blocks(response: dict, candidates: list[str]) -> list[str]:
    """Blocks a chip-level ``revise`` names explicitly: ``block_actions``
    (dict / JSON string, any action other than approve/skip),
    ``affected_blocks`` (list), else exact block names mentioned in the
    ``feedback`` / ``reasoning`` text. Order follows ``candidates``."""
    named: set[str] = set()
    actions = response.get("block_actions")
    if isinstance(actions, str) and actions.strip():
        try:
            actions = json.loads(actions)
        except json.JSONDecodeError:
            actions = None
    if isinstance(actions, dict):
        named |= {k for k, v in actions.items()
                  if str(v or "").strip().lower() not in {"approve", "skip", "keep"}}
    elif isinstance(actions, list):
        named |= {str(a) for a in actions}
    affected = response.get("affected_blocks") or []
    if isinstance(affected, str):
        affected = [a.strip() for a in affected.split(",")]
    named |= {str(a) for a in affected}
    text = " ".join(str(response.get(k) or "") for k in ("feedback", "reasoning"))
    if text:
        named |= {c for c in candidates
                  if re.search(rf"(?<![A-Za-z0-9_]){re.escape(c)}(?![A-Za-z0-9_])", text)}
    # WP-11: a structured keep/approve/skip is authoritative over a prose mention.
    if isinstance(actions, dict):
        named -= {k for k, v in actions.items()
                  if str(v or "").strip().lower() in {"approve", "skip", "keep"}}
    return [c for c in candidates if c in named]


def _adopt_reviewed_specs(pr: str, edited_blocks, reviewed_specs):
    """Adopt all reviewed files as one fail-closed operation."""
    from orchestrator.state_store.spec_adoption import adopt_reviewed_specs
    return adopt_reviewed_specs(_db(pr), edited_blocks, reviewed_specs)


def _plan_targeted_revise(
    pr: str,
    response: dict,
    block_names: list[str],
    edited_blocks: list[str],
    reviewed_specs: dict,
    failed_blocks: list[str],
    review_summary: str,
    tier: int,
) -> dict:
    """Turn a chip-level ``revise`` into a per-block plan ``{block: reuse_spec}``.

    Scope = blocks the reviewer edited + blocks the chip lead named + blocks
    that failed their lifecycle; an unscoped revise keeps today's whole-tier
    re-entry. Reviewer edits are ADOPTED (the reviewed copy becomes the
    canonical spec, newer than the block's RTL/TB so those regenerate); named
    blocks get the chip lead's findings as ``gate_feedback.txt`` and re-spec
    from their current spec; everything in scope drops ``best_result`` so the
    RTL skip-regen fast path cannot reuse a pass measured against the old spec.
    Blocks outside the scope keep their completed result untouched.
    """
    named = _revise_named_blocks(response, block_names)
    edited = [b for b in block_names if b in set(edited_blocks)]
    scope = [b for b in block_names
             if b in set(edited) | set(named) | set(failed_blocks)]
    unscoped = not scope
    if unscoped:
        scope = list(block_names)
    feedback = (str(response.get("feedback") or "").strip()
                or str(response.get("reasoning") or "").strip()
                or review_summary.strip())
    spec_dir = Path(pr) / "arch" / "uarch_specs"
    plan: dict[str, bool] = {}
    adoption = _adopt_reviewed_specs(pr, edited, reviewed_specs)
    adopt_failed = set(edited) if not adoption.ok else set()
    for name in scope:
        canonical = spec_dir / f"{name}.md"
        # A named block (or an unscoped whole-tier revise) re-specs with the
        # chip lead's findings; an edited-only block implements the reviewed
        # spec as-is; a failed-only block retries its RTL against its spec.
        needs_respec = unscoped or name in named
        if needs_respec and feedback:
            bdir = Path(pr) / ".coresmith" / "blocks" / name
            bdir.mkdir(parents=True, exist_ok=True)
            try:
                with (bdir / "gate_feedback.txt").open("a", encoding="utf-8") as fh:
                    fh.write(
                        f"\n\n## INTEGRATION REVIEW REVISION (tier {tier}; MANDATORY)\n\n"
                        f"{feedback}\n"
                    )
            except OSError:
                pass
        plan[name] = bool(canonical.exists()) and not needs_respec and name not in adopt_failed
        _db(pr).clear_result(name, "best")
    return plan


async def integration_review_prepare_node(state: OrchestratorState) -> dict:
    """Run and seal the model-backed part before the approval interrupt."""
    import hashlib

    pr = state.get("project_root", str(PROJECT_ROOT))
    block_queue = state.get("block_queue", [])
    tier_list = state.get("tier_list", [])
    current_idx = state.get("current_tier_index", 0)
    tier = tier_list[current_idx] if current_idx < len(tier_list) else 1
    block_names = [b["name"] for b in block_queue if b.get("tier", 1) == tier]
    if not block_names or state.get("integration_approved_specs"):
        return {"integration_review_bundle": None}
    try:
        from orchestrator.langchain.agents.coresmith_llm import DEFAULT_MODEL
        from orchestrator.langchain.agents.integration_review_agent import IntegrationReviewAgent
        result = await IntegrationReviewAgent(
            model=DEFAULT_MODEL, temperature=0.1,
        ).review(block_names=block_names, project_root=pr)
        reviewed_specs = dict(result.get("reviewed_specs") or {})
        hashes = {
            name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for name, path in reviewed_specs.items()
        }
        bundle = {**result, "reviewed_specs": reviewed_specs,
                  "reviewed_spec_hashes": hashes, "review_failed": False}
    except Exception as exc:
        bundle = {
            "summary": f"Integration review failed: {exc}",
            "issues_found": 1, "issues_fixed": 0, "edited_blocks": [],
            "reviewed_specs": {}, "reviewed_spec_hashes": {},
            "review_failed": True,
        }
    return {"integration_review_bundle": bundle}


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

    pending = {k: v for k, v in (state.get("integration_approved_specs") or {}).items()
               if k in block_names}
    if pending:
        import hashlib
        try:
            verified = all(
                hashlib.sha256((Path(pr) / "arch/uarch_specs" / f"{name}.md").read_bytes()).hexdigest() == digest
                and (_db(pr).result(name, "best") or {}).get("spec_sha256") == digest
                and (_db(pr).result(name, "best") or {}).get("sim_passed") is True
                for name, digest in pending.items())
        except OSError:
            verified = False
        if verified:
            return {"integration_review_action": "approve", "integration_review_failed": False,
                    "integration_approved_specs": None,
                    "revise_blocks": {k: v for k, v in (state.get("revise_blocks") or {}).items()
                                      if k not in block_names} or None}
        response = await _resolve_interrupt({
            "type": "uarch_spec_reverification_failed", "tier": tier,
            "reason": "Approved spec hashes do not have matching successful verification results",
            "affected_blocks": list(pending), "supported_actions": ["retry", "abort"],
        })
        return {"integration_review_action": "revise" if response.get("action") == "retry" else "abort",
                "integration_review_failed": True, "integration_approved_specs": pending,
                "revise_blocks": {name: True for name in pending}}

    # The compiled graph checkpoints this bundle before entering this node.
    # The fallback keeps direct callers and older embedded graphs compatible.
    bundle = state.get("integration_review_bundle")
    try:
        if bundle is not None:
            import hashlib
            reviewed_specs = dict(bundle.get("reviewed_specs") or {})
            expected = dict(bundle.get("reviewed_spec_hashes") or {})
            actual = {
                name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                for name, path in reviewed_specs.items()
            }
            if actual != expected:
                raise ValueError("reviewed uArch spec changed after review")
            result = bundle
        else:
            from orchestrator.langchain.agents.coresmith_llm import DEFAULT_MODEL
            from orchestrator.langchain.agents.integration_review_agent import (
                IntegrationReviewAgent,
            )
            agent = IntegrationReviewAgent(model=DEFAULT_MODEL, temperature=0.1)
            result = await agent.review(block_names=block_names, project_root=pr)
        review_summary = result.get("summary", "No issues found.")
        issues_found = result.get("issues_found", 0)
        issues_fixed = result.get("issues_fixed", 0)
        edited_blocks = [b for b in (result.get("edited_blocks") or [])
                         if b in block_names]
        reviewed_specs = dict(result.get("reviewed_specs") or {})
    except Exception as exc:
        review_summary = f"Integration review failed: {exc}"
        issues_found = 1
        issues_fixed = 0
        edited_blocks, reviewed_specs = [], {}
        review_failed = True
    else:
        review_failed = bool(result.get("review_failed", False))

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
        "edited_blocks": edited_blocks,
        "mem_price_deferred": mem_price_deferred,
        "review_failed": review_failed,
        "supported_actions": ["approve", "revise", "abort"],
        "outer_agent_guidance": (
            "The Integration Agent has reviewed all uArch specs for "
            "cross-block interface coherence. Present this as a CHIP-LEVEL "
            "review to the user. The user approves or rejects ALL specs at "
            "once. If the Integration Agent fixed mismatches, summarize "
            "what was changed. A `revise` is TARGETED: only the blocks in "
            "edited_blocks (the reviewed spec is adopted as-is) plus any "
            "block you name in `affected_blocks` / `block_actions` (re-spec "
            "with your `feedback`) re-enter the tier; the other blocks keep "
            "their passing result. A revise naming nothing re-runs the tier."
        ),
    }

    response = await _resolve_interrupt(payload)
    action = response.get("action", "abort")
    if action == "approve" and failed_tier_blocks:
        log(
            "  [INTEGRATION REVIEW] Approval rejected because current-tier "
            "blocks failed; treating as revise",
            YELLOW,
        )
        action = "revise"
    if (action == "revise" and issues_found == 0 and not review_failed
            and _is_content_free_revise(response)):
        # Only downgrade CONTENT-FREE revises (the stale auto-revise churn
        # class). A chip-lead/human revise carrying reasoning, feedback, or
        # block_actions names real stale-RTL findings -- discarding one sent
        # a known-bad chip straight into integration_dv (arm-U audit).
        log(
            "  [INTEGRATION REVIEW] Clean review returned content-free "
            "revise; treating as approve",
            YELLOW,
        )
        action = "approve"

    # Keep the entries of a plan that still names blocks in LATER tiers (a
    # DV-failure revise spanning tiers); this tier's own entries are replaced.
    _carry = {k: v for k, v in (state.get("revise_blocks") or {}).items()
              if k not in block_names}
    revise_blocks: dict | None = _carry or None
    approved_specs = None
    if action == "approve":
        try:
            if review_failed:
                raise ValueError(review_summary)
            adoption = _adopt_reviewed_specs(pr, edited_blocks, reviewed_specs)
            if not adoption.ok:
                raise ValueError(adoption.error)
            revise_blocks = {**_carry, **{name: True for name in adoption.reverify}} or None
            approved_specs = {name: adoption.hashes[name] for name in adoption.reverify} or None
        except Exception as exc:
            review_failed = True
            response = await _resolve_interrupt({
                "type": "uarch_spec_adoption_failed", "tier": tier,
                "affected_blocks": edited_blocks, "reason": str(exc),
                "supported_actions": ["retry", "abort"],
            })
            action = "retry" if response.get("action") == "retry" else "abort"
            revise_blocks = _carry or None
    if action == "revise":
        try:
            revise_blocks = {**_carry, **_plan_targeted_revise(
                pr, response, block_names, edited_blocks, reviewed_specs,
                failed_tier_blocks, review_summary, tier,
            )}
        except Exception as exc:
            review_failed = True
            response = await _resolve_interrupt({"type": "uarch_spec_adoption_failed",
                "reason": str(exc), "supported_actions": ["retry", "abort"]})
            action = "retry" if response.get("action") == "retry" else "abort"
    write_graph_event(pr, "Integration Review", "graph_node_exit", {
        "action": action, "issues_found": issues_found,
        "review_failed": review_failed,
        "edited_blocks": edited_blocks,
        "revise_blocks": revise_blocks,
    })
    if action == "abort":
        log("  [INTEGRATION REVIEW] Aborted by user/agent", RED)
    elif action == "revise":
        _desc = ', '.join(
            f"{b} ({'implement reviewed spec' if reuse else 're-spec with feedback'})"
            for b, reuse in (revise_blocks or {}).items())
        log(f"  [INTEGRATION REVIEW] Targeted revise -- re-entering: {_desc}",
            YELLOW)
    return {
        "integration_review_action": action,
        "integration_review_failed": review_failed,
        "integration_approved_specs": approved_specs,
        "revise_blocks": revise_blocks,
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

    plan = state.get("revise_blocks") or None
    tier_list = list(state.get("tier_list") or [])
    queue = state.get("block_queue", [])
    # A targeted re-entry must not re-run tiers that are already done:
    # skip every following tier whose blocks all passed earlier and none of
    # which the plan names (first pass: nothing is completed, nothing skips).
    passed_names = {b.get("name") for b in _current_phase_completed(state)
                    if b.get("success")}
    while new_idx < len(tier_list):
        names = [b["name"] for b in queue if b.get("tier", 1) == tier_list[new_idx]]
        if names and all(n in passed_names for n in names) \
                and not any(n in (plan or {}) for n in names):
            log(f"  Tier {tier_list[new_idx]}: all {len(names)} blocks already "
                "passed and none are planned -- skipping", CYAN)
            new_idx += 1
            continue
        break
    keep = None
    if plan:
        later = set(tier_list[new_idx:])
        if any(b.get("name") in plan and b.get("tier", 1) in later
               for b in queue):
            keep = plan
    return {"current_tier_index": new_idx, "revise_blocks": keep}


# ---------------------------------------------------------------------------
# Orchestrator routing functions
# ---------------------------------------------------------------------------

def route_after_integration_review(state: OrchestratorState) -> str:
    """Route based on the user's integration review decision.

    approve → advance_tier (continue normally)
    abort   → END (terminate the pipeline)
    revise  → init_tier (re-enter ONLY the blocks in ``revise_blocks``; the
              reviewed specs were adopted and the chip lead's findings
              written as gate feedback by integration_review_node)
    """
    action = state.get("integration_review_action", "approve")
    if action == "retry":
        return "integration_review"
    if action == "revise":
        return "init_tier"
    if action == "abort" or state.get("integration_review_failed"):
        return END
    if state.get("integration_approved_specs"):
        return "init_tier"
    return "advance_tier"


route_after_integration_review.__edge_labels__ = {
    "integration_review": "RETRY ADOPTION",
    "advance_tier": "APPROVED",
    "init_tier": "REVISE",
    END: "ABORT",
}


def route_next_tier(state: OrchestratorState) -> str:
    """Route after advance_tier: next tier, or pipeline_complete when the tier list is exhausted or a block aborted."""
    completed = state.get("completed_blocks", [])
    if any(b.get("aborted") for b in completed):
        return "pipeline_complete"

    tier_list = state.get("tier_list", [])
    current_idx = state.get("current_tier_index", 0)
    if current_idx < len(tier_list):
        return "init_tier"
    return "pipeline_complete"


route_next_tier.__edge_labels__ = {
    "init_tier": "NEXT TIER",
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

        resume = await _resolve_interrupt(payload)

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
            # WP-10b: re-enter ONLY the failed/missing blocks through the
            # targeted plan (spec reused unless a uarch_patch re-specced it);
            # init_tier skips tiers with nothing to redo.
            _plan = {n: (n not in _patched)
                     for n in list(failed_names) + list(missing_blocks)}
            return {
                "pipeline_done": False,
                "pipeline_aborted": False,
                "revalidate_pending": True,
                "revalidate_attempts": rv_attempts + 1,
                "current_tier_index": 0,
                "revise_blocks": _plan or None,
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


async def _park_candidate_failure(pr: str, design_name: str, rtl_paths: dict,
                                  reason: str, errors: list, top_rtl_path: str,
                                  *, phase: str = "single_block", deterministic: bool = False) -> dict:
    """Park a rejected candidate without claiming a chassis assembly failed."""
    from orchestrator.harness.top_module import invalidate_candidate
    invalidate_candidate(pr)
    errors = [str(e) for e in (errors or [])][:24]
    log(f"  [INTEGRATION] {reason} -- parking", RED)
    for error in errors[:8]:
        log(f"      - {error}", RED)
    payload = {
        "type": "integration_failure", "phase": phase,
        "design_name": design_name, "top_rtl_path": top_rtl_path,
        "block_count": len(rtl_paths), "block_rtl_paths": rtl_paths,
        "error_count": max(1, len(errors)), "lint_clean": False,
        "reason": reason, "errors": errors, "deterministic": deterministic,
        "supported_actions": ["retry", "fix_rtl", "abort"],
        "outer_agent_guidance": (
            f"Candidate adoption failed ({reason}; see errors). Fix the task top "
            "declaration or the block RTL identified by the errors before retrying. "
            "A declared top must already exist in the selected RTL on the single-block path. "
            "An unchanged deterministic mismatch will recur on retry."
        ),
        "reference_files": {"task": str(Path(pr) / "inputs/task.yaml"),
                            "top_rtl": top_rtl_path},
    }
    write_graph_event(pr, "Integration Check", "candidate_adoption_failed", payload)
    resp = await _resolve_interrupt(payload)
    resp = resp if isinstance(resp, dict) else {}
    action = resp.get("action", "abort")
    result = {
        "reason": reason, "candidate_adoption_failed": True,
        "errors": errors, "lint_clean": False, "top_rtl_path": top_rtl_path,
        "action_taken": action, "deterministic": deterministic,
    }
    if action in ("retry", "fix_rtl"):
        result["retry_requested"] = True
        result["fix_applied"] = str(resp.get("rtl_fix_description", ""))
    else:
        result.update(aborted=True, skipped=True)
    write_graph_event(pr, "Integration Check", "graph_node_exit", {
        "action": action, "phase": phase,
    })
    log(f"  [INTEGRATION] {phase} park -> {action}", YELLOW if result.get("retry_requested") else RED)
    return result


async def _park_caravel_assembly_failure(pr: str, design_name: str, rtl_paths: dict,
                                         reason: str, errors: list, top_rtl_path: str) -> dict:
    """WP-45: the deterministic Caravel assembly is the ONLY way to produce the
    graded `user_project_wrapper`; when it is not clean, park instead of
    falling back to an LLM-assembled top with a different module name."""
    from orchestrator.chassis.profile import CARAVEL, declared_chassis
    if declared_chassis(pr) != CARAVEL:
        return await _park_candidate_failure(pr, design_name, rtl_paths, reason, errors,
                                             top_rtl_path, phase="candidate_adoption")
    errors = [str(e) for e in (errors or [])][:24]
    log(f"  [INTEGRATION] deterministic Caravel assembly NOT clean ({reason}) -- "
        "parking (no Integration Lead fallback for a locked Caravel boundary)", RED)
    for _e in errors[:8]:
        log(f"      - {_e}", RED)
    write_graph_event(pr, "Integration Check", "caravel_assembly_failed", {
        "reason": reason, "errors": errors, "top_rtl_path": top_rtl_path,
    })
    payload = {
        "type": "integration_failure",
        "phase": "caravel_assembly",
        "design_name": design_name,
        "top_rtl_path": top_rtl_path,
        "block_count": len(rtl_paths or {}),
        "error_count": max(1, len(errors)),
        "lint_clean": False,
        "assembly_reason": reason,
        "assembly_errors": errors,
        "block_rtl_paths": rtl_paths,
        "supported_actions": ["retry", "fix_rtl", "abort"],
        "outer_agent_guidance": (
            "The ENGINE assembles the graded `user_project_wrapper` from the "
            "blocks and the interface contract; that assembly did not come out "
            f"clean ({reason}; see assembly_errors). There is NO LLM-integrator "
            "fallback for a locked Caravel boundary: a differently named top "
            "cannot be graded. Fix the block RTL or port declarations the "
            "errors point at (fix_rtl), or retry after an engine/operator fix. "
            "abort only if the block set itself is wrong."
        ),
        "reference_files": {"top_rtl": top_rtl_path},
    }
    resp = await _resolve_interrupt(payload)
    action = (resp or {}).get("action", "abort") if isinstance(resp, dict) else "abort"
    write_graph_event(pr, "Integration Check", "graph_node_exit", {
        "action": action, "phase": "caravel_assembly",
    })
    result = {
        "reason": f"caravel assembly {reason}",
        "caravel_assembly_failed": True,
        "assembly_errors": errors,
        "top_rtl_path": top_rtl_path,
        "action_taken": action,
    }
    if action in ("retry", "fix_rtl"):
        result["retry_requested"] = True
        result["fix_applied"] = str((resp or {}).get("rtl_fix_description", ""))
        log(f"  [INTEGRATION] caravel assembly park -> {action}; re-running the "
            "integration check", YELLOW)
    else:
        result["aborted"] = True
        result["skipped"] = True
        log("  [INTEGRATION] caravel assembly park -> abort", RED)
    return result


def _integration_handoff_context(pr: str, fallback_name: str,
                                 summary: str) -> tuple[str, str]:
    """Apply the task's authoritative top and full boundary requirements."""
    from orchestrator.harness.top_module import declared_top
    design_name = declared_top(pr) or fallback_name
    root = Path(pr)
    requirements_path = next(
        (path for path in (root / "inputs" / "requirements.md",
                           root / "requirements.md") if path.is_file()),
        None,
    )
    if requirements_path is not None:
        try:
            summary += ("\n\n--- AUTHORITATIVE FULL REQUIREMENTS ---\n"
                        + requirements_path.read_text(encoding="utf-8"))
        except OSError:
            pass
    return design_name, summary


async def _prepare_integration_check(state: OrchestratorState) -> dict:
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

        # WP-31/68: retire only superseded assembly artifacts, at the point
        # integration will rebuild them. Tier re-entry and reporting must
        # preserve the current validated candidate for downstream consumers.
        retired = _retire_derived_integration_artifacts(pr)
        if retired:
            log(f"  [INTEGRATION] Retired superseded derived artifact(s): {retired}", CYAN)
            write_graph_event(pr, "Integration Check", "derived_artifacts_retired",
                              {"files": retired})

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
                response = (await _resolve_interrupt(payload)) or {}
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

        # A block whose RTL file RESOLVED but whose ports could NOT be parsed is
        # dropped from `modules` -- the same silent-deletion defect the
        # completeness gate above exists to stop, one step later (and with
        # 1-of-2 parsed it would even take the single-block wrapper path below).
        # Park in the same shape rather than assembling around it.
        _unparsed = [n for n in rtl_paths if n not in modules]
        if _unparsed and _block_rtl_complete_gate_enabled():
            log(f"  [INTEGRATION] BLOCK PORTS UNPARSEABLE for {_unparsed} -- "
                f"refusing to assemble a chip that is missing a block", RED)
            write_graph_event(pr, "Integration Check", "block_rtl_unparsed",
                              {"unparsed_blocks": _unparsed,
                               "parsed_blocks": sorted(modules)})
            response = (await _resolve_interrupt({
                "type": "integration_failure",
                "error_kind": "unparsed_block_rtl",
                "unparsed_blocks": _unparsed,
                "unparsed_block_rtl_paths": {n: rtl_paths[n] for n in _unparsed},
                "error_count": len(_unparsed),
                "supported_actions": ["override", "abort"],
                "outer_agent_guidance": (
                    "These blocks' RTL files exist but their module header "
                    "could not be parsed, so assembling now would ship a chip "
                    "with those blocks DELETED -- no instance, no ports. For "
                    "each one: check that the file declares a module matching "
                    "the block (see .coresmith/block_specs.json rtl_target) in "
                    "a form the port parser accepts. `override` assembles "
                    "WITHOUT those blocks -- only for a block that genuinely "
                    "does not belong in the chip. `abort` ends integration."
                ),
            })) or {}
            _act = response.get("action", "abort")
            write_graph_event(pr, "Integration Check",
                              "block_rtl_unparsed_resume", {"action": _act})
            if _act == "override":
                log("  [INTEGRATION] unparseable block RTL OVERRIDE -- "
                    f"assembling WITHOUT {_unparsed}", YELLOW)
                rtl_paths = {n: _rp for n, _rp in rtl_paths.items()
                             if n in modules}
            else:
                result = {
                    "aborted": True, "skipped": True,
                    "reason": ("block RTL ports unparseable: "
                               f"{_unparsed}"),
                    "error": "unparsed_block_rtl",
                    "error_count": len(_unparsed),
                    "missing_blocks": _unparsed,
                }
                write_graph_event(pr, "Integration Check",
                                  "graph_node_exit", result)
                return {"integration_result": result}

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
                    break
                except (json.JSONDecodeError, OSError, AttributeError):
                    continue

        # Summaries omit exact boundary pin names and other normative clauses.
        # Give the integration author the original requirements verbatim; the
        # declared top above and this document together define the external
        # interface rather than child-module naming conventions.
        design_name, prd_summary = _integration_handoff_context(
            pr, design_name, prd_summary)

        rtl_dir = Path(pr) / "rtl" / "integration"
        rtl_dir.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', design_name).lower()
        if not safe_name or safe_name[0].isdigit():
            safe_name = f"top_{safe_name}"
        output_path = str(rtl_dir / f"{safe_name}.v")

        # A declared single-block top must already exist in the selected RTL.
        # Only an undeclared top permits a generated passthrough wrapper.
        if len(modules) == 1:
            from orchestrator.harness.top_module import (
                declared_top,
                module_declared_in,
                write_candidate_receipt,
            )
            solo_name, solo_mod = next(iter(modules.items()))
            try:
                top_name = declared_top(pr)
            except (ValueError, OSError) as exc:
                return {"integration_result": await _park_candidate_failure(
                    pr, design_name, rtl_paths, "invalid top declaration on the single-block path",
                    [str(exc)], "")}
            single_block_wrapper = not top_name
            top_mod = solo_mod
            if top_name:
                declaring_paths = list(dict.fromkeys(
                    str(Path(path).resolve()) for path in rtl_paths.values()
                    if module_declared_in(path, top_name)))
                if len(declaring_paths) != 1:
                    return {"integration_result": await _park_candidate_failure(
                        pr, design_name, rtl_paths, "declared top mismatch on the single-block path",
                        [f"The task declares top {top_name!r}; expected exactly one block RTL file "
                         f"declaring it, found {len(declaring_paths)}. No wrapper was generated."],
                        "", deterministic=True)}
                output_path = declaring_paths[0]
                top_mod = await asyncio.to_thread(parse_verilog_ports, output_path, top_name)
                log(f"  [INTEGRATION] Single-block design: using declared top "
                    f"{top_name} from {output_path}", GREEN)
            else:
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
                Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
                log(f"  [INTEGRATION] Single-block design: generated wrapper "
                    f"{top_name} for {solo_name}", GREEN)

            lint_result = await asyncio.to_thread(
                lint_top_level, output_path, list(rtl_paths.values()), top_name,
                top_module=top_name,
                project_root=_pr(state),
            )
            lint_clean = lint_result.get("clean", False)
            log(f"  [INTEGRATION] Lint: {'CLEAN' if lint_clean else 'ERRORS'}",
                GREEN if lint_clean else RED)

            integration_result = {
                "design_name": design_name,
                "top_module": top_name,
                "top_rtl_path": output_path,
                "block_count": 1,
                "wire_count": len(top_mod.ports),
                "skipped_connections": [],
                "mismatches": [],
                "error_count": 0,
                "warning_count": 0,
                "lint_clean": lint_clean,
                "lint_errors": lint_result.get("errors", ""),
                "block_rtl_paths": rtl_paths,
                "single_block_wrapper": single_block_wrapper,
            }

            try:
                if not lint_clean:
                    raise ValueError(f"Single-block candidate did not lint cleanly: {lint_result.get('errors', '')}")
                # The block can be the elaborated root itself. Every other
                # expected block must still occur as a reachable child cell.
                expected = [name for name in rtl_paths
                            if not (name == solo_name and top_name == solo_mod.name
                                    and not single_block_wrapper)]
                write_candidate_receipt(pr, top_name, output_path, rtl_paths,
                                        expected_blocks=expected,
                                        note="single-block passthrough" if single_block_wrapper else "single-block declared top",
                                        integration_result=integration_result)
            except (ValueError, OSError) as exc:
                return {"integration_result": await _park_candidate_failure(
                    pr, design_name, rtl_paths, "candidate validation failed on the single-block path",
                    [str(exc)], output_path)}
            write_graph_event(pr, "Integration Check", "graph_node_exit", {
                "success": True, "top_module": top_name, "block_count": 1,
                "single_block_wrapper": single_block_wrapper,
            })
            return {"integration_result": integration_result}

        # ---- Defect 4: deterministic Caravel user_project_wrapper assembly ----
        # If a pad-adapter / Caravel wrapper block is present, assemble the wired
        # `user_project_wrapper` chip_top deterministically (locked Caravel ports,
        # instantiates + wires every block) rather than asking the Integration
        # Lead LLM, which named the top after the design and treated the pad
        # adapter as a peer block -- so the daemon never delivered a gradeable
        # wired top and every chip-lead hand-assembled one.
        from orchestrator.chassis.profile import CARAVEL, declared_chassis
        from orchestrator.langgraph.integration_helpers import (
            detect_wrapper_block,
            generate_caravel_wrapper_top,
            load_interface_contract_edges,
        )
        # WP-51: the wrapper block is the one named as the task's declared top.
        _chassis = declared_chassis(pr)
        _wrapper_block = detect_wrapper_block(modules, _chassis.top_module) if _chassis else None
        # WP-24: the generator may have written the wrapper block as the
        # COMPLETE graded top (pads + every core block instantiated). Re-wrapping
        # it produces wiring hazards and a nested top the QSPI pin-boundary gate
        # rejects; adopt it as the chip top instead.
        # The PRD's structured pin map, when present, lets the top route the pads
        # itself -- so the design needs no pin-adapter block and assembly no
        # longer depends on finding one.
        from orchestrator.architecture.pin_map import load_pin_map
        _pin_map = load_pin_map(pr)
        if _pin_map is not None and not _pin_map.ok:
            for _e in _pin_map.errors:
                log(f"  [INTEGRATION] pin_map: {_e}", RED)
            _pin_map = None
        if (_chassis == CARAVEL and (_wrapper_block is not None or _pin_map is not None)
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
            if _wiring_errors:
                # WP-45: fail closed -- never hand a locked boundary to the
                # Integration Lead.
                return {"integration_result": await _park_caravel_assembly_failure(
                    pr, design_name, rtl_paths, "wiring hazards",
                    list(_wiring_errors), "")}
            if not _wiring_errors:
                top_rtl_path = asm["rtl_path"]
                # Lint with the pad block's renamed copy swapped in (avoids a
                # duplicate `module user_project_wrapper` definition at chip level).
                _lint_paths = list(asm["lint_block_paths"].values())
                lint_result = await asyncio.to_thread(
                    lint_top_level, top_rtl_path, _lint_paths, "user_project_wrapper",
                    top_module="user_project_wrapper",
                    project_root=_pr(state),
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
                    from orchestrator.harness.top_module import write_candidate_receipt
                    try:   # WP-49/54: the receipt (declared-top check) BEFORE the record
                        write_candidate_receipt(pr, "user_project_wrapper", top_rtl_path,
                                                asm["lint_block_paths"], note="caravel assembly",
                                                expected_blocks=set(modules) - {_dropped},
                                                integration_result=integration_result)
                    except (ValueError, OSError) as _exc:
                        return {"integration_result": await _park_caravel_assembly_failure(
                            pr, design_name, rtl_paths, "top module mismatch",
                            [str(_exc)], top_rtl_path)}
                    return {"integration_result": integration_result}
                _errs = [ln for ln in str(lint_result.get("errors", "")).splitlines()
                         if ln.strip()][:20]
                if missing:
                    _errs.append("blocks not instantiated by the assembled wrapper: "
                                 + ", ".join(missing))
                return {"integration_result": await _park_caravel_assembly_failure(
                    pr, design_name, asm["lint_block_paths"],
                    "lint errors" if not lint_clean else "missing instantiations",
                    _errs, top_rtl_path)}
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
                load_interface_contract_edges,
                merge_contract_compatibility_connections,
            )

            # The block diagram usually names only a channel and its aggregate
            # payload width.  Prefer canonical contract edges, whose explicit
            # fields resolve the actual payload ports; falling back preserves
            # legacy projects without interface_contracts.json.
            contract_edges = await asyncio.to_thread(
                load_interface_contract_edges, pr
            )
            compatibility_connections = merge_contract_compatibility_connections(
                connections, contract_edges
            )

            gr = gate_guard(
                "integration_compat",
                check_integration_compatibility,
                compatibility_connections,
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
        _hier_sources = [top_rtl_path, *rtl_paths.values()]
        postcond = assert_blocks_instantiated(
            chip_top_text, set(block_rtl_sources.keys()), source_paths=_hier_sources,
            top_module=module_name, project_root=pr,
        )
        if postcond:
            log(f"  [INTEGRATION] Postcondition failed: {postcond}", RED)
            write_graph_event(pr, "Integration Check", "graph_node_exit", {
                "error": "block_instantiation_postcondition_failed",
                "missing_summary": postcond,
            })
            # WP-22: park for the chip lead instead of ending the run (the
            # fft256_qspi run went straight to status "done" here).
            _pc_payload = {
                "type": "integration_failure",
                "phase": "postcondition",
                "design_name": design_name,
                "top_rtl_path": top_rtl_path,
                "block_count": len(block_rtl_sources),
                "error_count": 1,
                "lint_clean": False,
                "postcondition": postcond,
                "block_rtl_paths": rtl_paths,
                "agent_notes": str(agent_result.get("notes", ""))[:2000],
                "supported_actions": ["retry", "fix_rtl", "abort"],
                "outer_agent_guidance": (
                    "The Integration Lead's chip top does not instantiate every "
                    "block (postcondition). Either the top dropped blocks (retry "
                    "re-runs the Integration Lead) or you can wire the missing "
                    "blocks into the top on disk yourself (fix_rtl). abort only if "
                    "the block set itself is wrong."
                ),
                "reference_files": {"top_rtl": top_rtl_path},
            }
            _pc_resp = await _resolve_interrupt(_pc_payload)
            _pc_action = (_pc_resp or {}).get("action", "abort") if isinstance(_pc_resp, dict) else "abort"
            write_graph_event(pr, "Integration Check", "graph_node_exit", {
                "action": _pc_action, "phase": "postcondition",
            })
            _pc_result = {
                "reason": postcond,
                "postcondition_failed": True,
                "agent_notes": agent_result.get("notes", ""),
                "top_rtl_path": top_rtl_path,
                "action_taken": _pc_action,
            }
            if _pc_action in ("retry", "fix_rtl"):
                _pc_result["retry_requested"] = True
                _pc_result["fix_applied"] = str((_pc_resp or {}).get("rtl_fix_description", ""))
                log(f"  [INTEGRATION] postcondition park -> {_pc_action}; "
                    f"re-running the integration check", YELLOW)
            else:
                _pc_result["aborted"] = True
                _pc_result["skipped"] = True
                log("  [INTEGRATION] postcondition park -> abort", RED)
            return {"integration_result": _pc_result}

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

        log(f"  [INTEGRATION] Agent generated {module_name}: "
            f"{len(modules)} blocks, "
            f"{agent_result.get('wire_count', 0)} wires", GREEN)
        span.set_attribute("top_module", module_name)

        block_rtl_list = list(rtl_paths.values())
        lint_result = await asyncio.to_thread(
            lint_top_level, top_rtl_path, block_rtl_list,
            design_name, project_root=_pr(state), top_module=module_name,
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

        return {"review_bundle": {
            "integration_result": integration_result,
            "agent_result": agent_result,
            "lint_result": lint_result,
            "artifact_hashes": _integration_artifact_hashes(
                [top_rtl_path, *rtl_paths.values(),
                 Path(pr) / "inputs/task.yaml", Path(pr) / "inputs/requirements.md",
                 Path(pr) / "requirements.md", Path(pr) / ".coresmith/block_diagram.json",
                 Path(pr) / ".coresmith/interface_contracts.json"]),
        }}

_integration_prepare_task = _durable_task(_prepare_integration_check)


def _integration_artifact_hashes(paths) -> dict[str, str | None]:
    """Bind approval to file bytes and to the absence of optional inputs."""
    return {str(path): (hashlib.sha256(Path(path).read_bytes()).hexdigest()
                        if Path(path).is_file() else None)
            for path in paths if path}


async def integration_check_node(state: OrchestratorState) -> dict:
    """Prepare once per graph invocation, then approve that immutable result.

    LangGraph restarts a node on interrupt resume. A durable task checkpoints
    assembly and lint before review, so accepting cannot rerun the author or
    retire the file the operator just reviewed. Explicit graph retries start
    a new task and deliberately regenerate/recheck the candidate.
    """
    from langgraph.config import get_config
    try:
        get_config()
    except RuntimeError:  # direct callers (including focused unit tests)
        prepared = await _prepare_integration_check(state)
    else:
        prepared = await _integration_prepare_task(state)
    if "review_bundle" not in prepared:
        return prepared
    return await _approve_integration_check(state, prepared["review_bundle"])


async def _approve_integration_check(state: OrchestratorState, bundle: dict) -> dict:
    import copy
    pr = state.get("project_root", str(PROJECT_ROOT))
    integration_result = copy.deepcopy(bundle["integration_result"])
    agent_result = bundle["agent_result"]
    lint_result = bundle["lint_result"]
    design_name = integration_result["design_name"]
    module_name = integration_result["top_module"]
    top_rtl_path = integration_result["top_rtl_path"]
    rtl_paths = integration_result["block_rtl_paths"]
    mismatches = integration_result["mismatches"]
    lint_clean = integration_result["lint_clean"]
    errors = [m for m in mismatches if m.get("severity") == "error"]
    warnings = [m for m in mismatches if m.get("severity") == "warning"]
    async def adopt_result():
        from orchestrator.harness.top_module import write_candidate_receipt
        expected = bundle.get("artifact_hashes", {})
        if _integration_artifact_hashes(expected) != expected:
            return {**integration_result, "aborted": True,
                    "error": "reviewed_artifacts_changed",
                    "reason": "Integration artifacts changed after review; restart integration_check to re-check them."}
        try:
            write_candidate_receipt(pr, module_name, top_rtl_path, rtl_paths,
                                    note="integration lead", integration_result=integration_result)
            return integration_result
        except (ValueError, OSError) as exc:
            return await _park_caravel_assembly_failure(
                pr, design_name, rtl_paths, "top module mismatch", [str(exc)], top_rtl_path)

    has_issues = len(errors) > 0 or not lint_clean
    if has_issues:
        log("  [INTEGRATION] Issues found -- interrupting for review", YELLOW)

        payload = {
            "type": "integration_failure",
            "design_name": design_name,
            "top_rtl_path": top_rtl_path,
            "block_count": integration_result["block_count"],
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
                "1. WIDTH_MISMATCH: Compare the canonical interface-contract "
                "field with both reported RTL ports and the actual chip_top "
                "connection. If the checker selected a handshake/control port "
                "for a payload field, repair the checker mapping and preserve "
                "the passing RTL. Edit RTL only when those exact payload "
                "endpoints have a real width mismatch.\n"
                "2. MISSING_PORT: Confirm the canonical contract field and "
                "actual connected port before adding or renaming RTL.\n"
                "3. DIRECTION_ERROR: Confirm the exact contract endpoint, then "
                "fix a real port-direction defect.\n"
                "4. LINT_ERRORS: Read the lint log and edit "
                f"{top_rtl_path} directly.\n"
                "5. After an RTL fix, resume_pipeline(action='fix_rtl'); after "
                "an engine/checker fix, reload the engine and resume with the "
                "review action supported by this checkpoint.\n"
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

        response = await _resolve_interrupt(payload)

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
                    "block_count": integration_result["block_count"],
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
                response = await _resolve_interrupt(repark_payload)
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

        if lint_clean and not integration_result.get("aborted") and not integration_result.get("skipped_by_user"):
            integration_result = await adopt_result()
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
            "block_count": integration_result["block_count"],
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

        response = await _resolve_interrupt(warning_payload)
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

        if lint_clean and not integration_result.get("aborted") and not integration_result.get("skipped_by_user"):
            integration_result = await adopt_result()
        return {"integration_result": integration_result}

    integration_result = await adopt_result()
    if not integration_result.get("lint_clean") or integration_result.get("aborted"):
        return {"integration_result": integration_result}
    log(f"\n{'='*60}", GREEN)
    log("  INTEGRATION CHECK PASSED", GREEN)
    log(f"  Top module: {module_name}", GREEN)
    log(f"  {integration_result['block_count']} blocks, "
        f"{agent_result.get('wire_count', 0)} wires", GREEN)
    if warnings:
        log(f"  {len(warnings)} warnings (non-blocking)", YELLOW)
    log(f"{'='*60}\n", GREEN)

    write_graph_event(pr, "Integration Check", "graph_node_exit", {
        "success": True,
        "top_module": module_name,
        "block_count": integration_result["block_count"],
        "wire_count": agent_result.get("wire_count", 0),
        "warnings": len(warnings),
    })

    return {"integration_result": integration_result}


def route_after_integration(state: OrchestratorState) -> str:
    """Route after integration check: proceed to DV or END."""
    next_node = "integration_dv"
    result = state.get("integration_result") or {}
    if result.get("aborted"):
        return END
    if result.get("retry_requested"):
        return "integration_check"   # WP-22: re-run after a postcondition park
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
    "integration_dv": "DV",
    "integration_check": "Retry",
}


# ---------------------------------------------------------------------------
# Node: model_integration  (LLM model-integration + deterministic Amaranth verify)
# ---------------------------------------------------------------------------
#
# pass-through to integration_dv -- byte-identical routing to before the
# feature existed. When ON it (a) calls the model-integration LLM agent to wire
# every per-block Amaranth block model into a top-level Amaranth chip model
# (arch/block_models/_chip_model.py) if one is not already present, then (b)
# runs the deterministic model-integration gate, which simulates the integrated
# chip model on a stimulus and asserts its output equals the reference
# implementation's output BIT-EXACT. On divergence it PARKS an interrupt
# (exactly like validation_dv) naming the first-divergence block; otherwise it
# passes through to integration_dv.


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
#
# The µARCH GATE runs BETWEEN the two fan-outs: after pass 1 (all blocks emit
# uArch spec + Amaranth block model) and BEFORE pass 2 (RTL+DV+synth). It relocates
# the model_integration_node logic (LLM stitch _chip_model.py + deterministic
# Amaranth verify + park-on-fail) to its shift-left position so a bad decomposition
# is caught in fast Python sims before any expensive RTL work.


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
            diag = _db(project_root).diagnosis(name)
            if not diag:
                continue
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
            try:
                if _db(project_root).result(name, "best"):
                    _db(project_root).update_result(
                        name, "best", sim_passed=False, uarch_patch_invalidated=True)
            except Exception:  # noqa: BLE001
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
        (only the failed/missing blocks re-enter, through the targeted plan)
        then back to ``pipeline_complete`` to recount
      - otherwise (clean) -> ``"integration_check"``.
    """
    if state.get("pipeline_aborted"):
        return "end"
    if state.get("revalidate_pending"):
        return "init_tier"
    return "integration_check"


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


# ---------------------------------------------------------------------------
# MAX-GEOMETRY DV stimulus gate (rung3-fixes-2, defect 1)
# ---------------------------------------------------------------------------
# A "verified" chip once shipped a truncated index-width bug because EVERY DV
# stimulus was a tiny fixed geometry (e.g. 16x16): the geometry-dependent
# index/address/counter widths were never exercised at the declared maximum, so
# a wrap at the 2^n boundary BELOW the max (e.g. a 7-bit column index wrapping
# at 512 on a 640-wide frame) sailed through integration + validation DV. This
# gate requires executed maximum cases only when the owner declares them in
# inputs/task.yaml. Policy dimensions and testbench markers alone describe
# scope; without owner cases they are explicitly not owner-certified.
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
    """Evaluate owner certification of the design's dimensional maxima.
    Default ON; ``CORESMITH_MAXGEO_GATE=0`` disables (both branches
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
        try:
            _collect_declared_dims(_db(project_root).block_specs(), dims)
        except Exception:  # noqa: BLE001
            pass
        for fname in ("ers_spec.json", "prd_spec.json"):
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


def _file_sha256(path: str) -> str:
    """sha256 of a file's bytes, or "" when it cannot be read (WP-40)."""
    import hashlib as _hashlib
    try:
        return _hashlib.sha256(Path(path).read_bytes()).hexdigest() if path else ""
    except OSError:
        return ""


def _tb_writer_flags(tb_result: dict | None) -> dict:
    """The engine-writer flags a reused testbench must carry forward.

    ``_maxgeo_conformance_scope`` keys off flags ONLY the engine's own bfm_lib
    writer sets, and they live in the writer's in-memory return -- so a
    fix_rtl/fix_tb resume that rebuilds tb_result from the persisted DV result
    has to restore them, or the identical TB that earned an advisory verdict
    one cycle earlier hard-fails the scope gate."""
    tbr = tb_result or {}
    flags = {k: tbr[k] for k in ("deterministic_bfm", "conformance_only",
                                 "contract")
             if tbr.get(k) is not None}
    # WP-40: the deterministic TB's content identity, taken when the engine
    # wrote it. A later cycle reuses the file only while it still hashes to
    # this; an edited copy is regenerated from the contract.
    if flags.get("deterministic_bfm"):
        sha = tbr.get("tb_sha256") or _file_sha256(
            tbr.get("testbench_path") or tbr.get("tb_path") or "")
        if sha:
            flags["tb_sha256"] = sha
    return flags


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


def _tb_maxgeo_mentions(tb_path: str, dims: dict) -> dict:
    """Cocotb cases mentioning each dimension, as scope only, never proof.

    Inspect source without importing the testbench. Comments, strings and
    identifiers inside a case can describe scope without exercising it.
    """
    import ast

    try:
        source = Path(tb_path).read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source)
    except (OSError, SyntaxError, ValueError):
        return {}
    lines = source.splitlines()
    mentions = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        decorators = [d.func if isinstance(d, ast.Call) else d for d in node.decorator_list]
        if not any((isinstance(d, ast.Attribute) and d.attr == "test")
                   or (isinstance(d, ast.Name) and d.id == "test") for d in decorators):
            continue
        body = "\n".join(lines[node.lineno - 1:node.end_lineno])
        found = [name for name in dims if re.search(r"(?<![\w./-])" + re.escape(name) + r"(?![\w./-])", body)]
        if found:
            mentions[node.name] = found
    return mentions


def _maxgeo_gate_verdict(
    project_root: str, tb_path: str, tb_result: dict | None = None,
    *, sim_result: dict | None = None,
) -> dict | None:
    """Markers describe scope. Only an executed owner-declared case proves it.

    Owner task.yaml declares ``max_geometry_cases: {case_name: {dimension: max}}``.
    ``executed_cases`` comes from the simulator's fresh successful XML test rows,
    never from the generated testbench's own metadata or comments.
    An absent declaration (or empty mapping) is non-blocking ``not_declared``;
    policy-authored dimensions cannot impose an owner certification obligation.
    """
    if not _maxgeo_gate_enabled():
        return None
    try:
        import yaml

        from orchestrator.langgraph.bfm_lib.maxgeo import declared_dimensional_maxima
        dims = declared_dimensional_maxima(project_root)
        task_path = Path(project_root) / "inputs/task.yaml"
        task = yaml.safe_load(task_path.read_text()) if task_path.exists() else None
        if task is None:
            task = {}
        if not isinstance(task, dict):
            raise ValueError("inputs/task.yaml must be a mapping")
        declared = task.get("max_geometry_cases", {})
        if not isinstance(declared, dict):
            raise ValueError("max_geometry_cases must map case names to dimension maxima")
        for name, maxima in declared.items():
            if (not isinstance(name, str) or not name.strip()
                    or not isinstance(maxima, dict) or not maxima
                    or any(not isinstance(key, str) or not key.strip()
                           or type(value) is not int or value <= 0
                           for key, value in maxima.items())):
                raise ValueError(f"Malformed max_geometry_cases entry: {name!r}")
        if not dims:
            return None
        scope = {"declared_dims": dims, "marker_pairs": _tb_maxgeo_pairs(tb_path),
                 "testbench_case_mentions": _tb_maxgeo_mentions(tb_path, dims)}
        if not declared:
            return {**scope, "verdict": "not_declared",
                    "reason": "maximum geometry not owner-certified (no max_geometry_cases declared)",
                    "uncovered_dims": dims, "executed_maximum_cases": []}
        executed = (sim_result or {}).get("executed_cases") or []
        if not isinstance(executed, list) or any(not isinstance(name, str) for name in executed):
            raise ValueError("Executed case evidence must be a list of exact case names")
        if (sim_result or {}).get("passed") is not True:
            executed = []
        covered = {}
        qualifying = []
        for name, maxima in declared.items():
            if name not in executed:
                continue
            attained = {key: value for key, value in dims.items()
                        if type(maxima.get(key)) is int and maxima[key] == value}
            if attained:
                qualifying.append(name)
                covered.update(attained)
        missing = {key: value for key, value in dims.items() if key not in covered}
        return {"verdict": "unknown" if missing else "pass",
                "reason": ("MAX-GEOMETRY unknown: no successful executed owner-declared maximum case covers "
                           f"{missing}; markers describe declared scope only") if missing else
                          "Executed owner-declared maximum cases cover every declared dimension",
                **scope,
                "uncovered_dims": missing, "executed_maximum_cases": qualifying}
    except Exception as exc:  # malformed declaration/evidence cannot disable the gate
        return {"verdict": "unknown", "reason": f"Maximum-case evidence unavailable: {exc}"}


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
                            f", Input data rate: "
                            f"{sf.get('input_data_rate_mbps', '?')} Mbps"
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
        # WP-29/WP-40: the engine's deterministic BFM testbench is derived
        # from the bus contract and DUT-blind. It is engine-owned: reused only
        # while its content still hashes to what the engine wrote, regenerated
        # (edit discarded) otherwise (observed: a chip lead rewrote its
        # sampling point and SCK period to make a failing chip pass).
        _prev_flags = previous_dv.get("tb_writer_flags") or {}
        if reuse_existing_tb and _prev_flags.get("deterministic_bfm"):
            _now_sha = _file_sha256(previous_tb_path)
            if _prev_flags.get("tb_sha256") and _now_sha == _prev_flags.get("tb_sha256"):
                log("  [INTEG-DV] previous testbench is the deterministic BFM and "
                    "is unmodified (sha256 match) -- reusing", YELLOW)
                write_graph_event(pr, "Integration DV", "deterministic_tb_reused", {
                    "after_action": previous_action, "path": previous_tb_path,
                    "tb_sha256": _now_sha,
                })
            else:
                log("  [INTEG-DV] previous deterministic BFM testbench was MODIFIED "
                    "(or carries no recorded hash) -- regenerating from the "
                    f"contract (after {previous_action}); the edit is discarded", YELLOW)
                write_graph_event(pr, "Integration DV", "deterministic_tb_modified", {
                    "after_action": previous_action, "discarded_path": previous_tb_path,
                    "recorded_sha256": _prev_flags.get("tb_sha256", ""),
                    "found_sha256": _now_sha,
                })
                reuse_existing_tb = False

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
                **(previous_dv.get("tb_writer_flags") or {}),
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
                                    "  [INTEG-DV] Declared MAX-EXTENT bus stimulus scope: "
                                    f"{tb_result['maxgeo_covered']}; "
                                    "outside scope (no compute oracle): "
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
                    tb_result = await generate_integration_testbench(
                        project_root=_pr(state),
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
                            supported_actions=["retry", "fix_rtl", "fix_tb",
                                               "revise", "abort"],
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
                    # Same lawful escape as the DV-failed park: a TB that
                    # cannot be generated is often a SPEC-level defect (the
                    # top contract the uArch specs describe is un-testable),
                    # and without this the chip-lead's only exit was abort.
                    "revise",
                    "abort",
                ],
                "outer_agent_guidance": (
                    "Integration DV could not generate a usable cocotb "
                    "testbench. As the outer-loop diagnostic agent, inspect "
                    "the top-level RTL, block port contracts, generator prompt, "
                    "and any partially written testbench. Use action='fix_tb' "
                    "when the testbench generator or prompt needs repair, "
                    "action='fix_rtl' when the top-level contract is invalid, "
                    "action='revise' when the defect is in the uArch specs "
                    "themselves (appends your feedback to the affected specs "
                    "and re-runs the tiers), "
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
            project_root=_pr(state),
        )
        if _stale_candidate(sim_result):
            log("  [INTEG-DV] candidate manifest is STALE (RTL changed after "
                "adoption) -- re-running the integration check to re-adopt "
                "before simulating", YELLOW)
            write_graph_event(pr, "Integration DV", "candidate_stale", {
                "log": str(sim_result.get("log", ""))[:300]})
            return {"integration_dv_result": _reintegrate_result("integration_dv", sim_result),
                    "pipeline_done": False}

        passed = sim_result.get("passed", False)
        sim_log = sim_result.get("log", "")

        # Maximum certification is mandatory only for owner-declared cases.
        # Policy-only dimensions remain explicitly not_declared and non-blocking.
        # run3-followups: the gate runs on EVERY passing cycle, including
        # operator-reused TBs (fix_tb/fix_rtl). "Trusted as-is" silently
        # DISARMED the gate on exactly the cycles that deserve more scrutiny --
        # proven live when a clobbered 2-test TB passed DV with no MAXGEO line
        # at all. Every evaluated outcome logs a verdict; silence now means
        # only "gate disabled or no declared dims".
        _mg = None
        if passed:
            _mg = _maxgeo_gate_verdict(pr, tb_path, tb_result, sim_result=sim_result)
            if _mg is not None:
                write_graph_event(pr, "Maximum Geometry", "maxgeo_verdict", _mg)
            if reuse_existing_tb and _mg is not None:
                log("  [INTEG-DV] MAX-GEOMETRY gate: evaluating an OPERATOR-"
                    "REUSED testbench (fix_tb/fix_rtl) -- operator edits get "
                    "more scrutiny, not less.", YELLOW)
            if _mg is not None and _mg.get("verdict") == "pass":
                log("  [INTEG-DV] MAX-GEOMETRY gate PASS -- every declared "
                    f"maximum was covered by executed owner cases: "
                    f"{_mg.get('executed_maximum_cases', [])}", GREEN)
                write_graph_event(pr, "Integration DV", "maxgeo_gate_pass", {
                    "gate": "maxgeo",
                    "executed_maximum_cases": _mg.get("executed_maximum_cases", []),
                })
            elif _mg is not None and _mg.get("verdict") == "not_declared":
                log(f"  [INTEG-DV] MAX-GEOMETRY not_declared -- {_mg['reason']}", YELLOW)
            elif _mg is not None:
                passed = False
                sim_log = ((sim_log + "\n\n") if sim_log else "") + _mg["reason"]
                span.set_attribute("maxgeo_gate_failed", True)
                log("  [INTEG-DV] MAX-GEOMETRY gate FAILED -- flipping DV to "
                    f"failed: uncovered={_mg.get('uncovered_dims', {})}", RED)

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
                # WP-11: advisory -- recorded and appended, never a DV failure.
                sim_log = ((sim_log + "\n\n") if sim_log else "") + (
                    "CHIP THROUGHPUT ADVISORY (not a DV failure):\n"
                    + (chip_tput.get("report", "") or chip_tput.get("reason", "")))
                span.set_attribute("chip_throughput_advisory", True)
                log("  [INTEG-DV] chip measured throughput below budget -- "
                    f"advisory only: {chip_tput.get('reason', '')}", YELLOW)
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
                    "max_geometry": _mg,
                    "design_name": design_name,
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
            "max_geometry": _mg,
            "block_rtl_paths": block_rtl_paths,
            "contract_audit": contract_audit,
            "contract_audit_path": contract_audit.get("audit_path", ""),
            "supported_actions": (
                # WP-29: the deterministic BFM is contract-derived and
                # DUT-blind -- there is no testbench to fix.
                ["retry", "fix_rtl", "revise", "abort"]
                if (tb_result or {}).get("deterministic_bfm")
                else [
                    "retry",        # regenerate testbench + re-simulate
                    "fix_rtl",      # outer agent fixed RTL, re-run sim only
                    "fix_tb",       # outer agent fixed testbench, re-run sim only
                    "revise",       # feedback -> affected uArch specs + tier regen
                    "abort",        # stop the pipeline
                ]
            ),
            "deterministic_bfm": bool((tb_result or {}).get("deterministic_bfm")),
            "outer_agent_guidance": (
                ("THIS TESTBENCH IS THE ENGINE'S DETERMINISTIC, CONTRACT-DERIVED, "
                 "DUT-BLIND BUS-PROTOCOL BFM (the same protocol the published "
                 "grader drives). It is engine-OWNED: fix_tb is not offered and "
                 "an edited copy is discarded (content hash). A failure here is "
                 "normally an RTL defect (fix_rtl) or a contract defect (revise). "
                 "If you believe the BFM itself is wrong, say so in `reasoning` "
                 "with a concrete counterexample (signal, cycle, expected vs "
                 "observed) and choose retry or abort; the operator owns the "
                 "BFM and versions any fix.\n\n"
                 if (tb_result or {}).get("deterministic_bfm") else "")
                + "Integration DV (top-level simulation) failed. As the outer-loop "
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
                "tb_writer_flags": _tb_writer_flags(tb_result),
                "sim_log_path": sim_result.get("log_path", ""),
                "max_geometry": _mg,
                "design_name": design_name,
                "contract_audit": contract_audit,
                "contract_audit_path": contract_audit.get("audit_path", ""),
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
        "supported_actions": ["retry", "fix_rtl", "fix_tb", "revise", "abort"],
    }
    response = (await _resolve_interrupt(payload)) or {}
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
    elif action == "revise":
        revised = _apply_revise_uarch(
            pr, response, dict(payload.get("contract_audit") or {}),
            "integration_dv")
        dv_result["revised_blocks"] = revised
        log(f"  [INTEG-DV] Revise: uArch feedback appended to "
            f"{revised or '[] (no specs matched)'}; "
            + ("re-entering only those blocks" if revised
               else "re-running every tier"),
            YELLOW)
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
        **({"current_tier_index": 0,
            "revise_blocks": ({n: False for n in revised} or None)}
           if action == "revise" else {}),
    }


def _stale_candidate(sim_result: dict) -> bool:
    """WP-76: the authoritative simulation refused a stale candidate manifest
    (someone edited the RTL after adoption -- typically a chip-lead fix_rtl).
    That is not a functional failure: the design must be re-integrated and
    re-adopted, then simulated."""
    if not isinstance(sim_result, dict) or sim_result.get("passed"):
        return False
    return (sim_result.get("kind") == "candidate_mismatch"
            and "stale" in str(sim_result.get("log", "")).lower())


def _reintegrate_result(stage: str, sim_result: dict) -> dict:
    return {
        "passed": False, "candidate_stale": True, "action_taken": "reintegrate",
        "pending_decision": False, "phase": stage,
        "reason": ("the candidate manifest is stale (RTL changed after adoption); "
                   "re-running the integration check to re-adopt the design"),
        "error": str(sim_result.get("log", ""))[:500],
    }


def route_after_integration_dv_decision(state: OrchestratorState) -> str:
    """Route the operator's decision back into integration DV or terminate."""
    result = state.get("integration_dv_result") or {}
    if result.get("action_taken") == "revise":
        return "init_tier"
    if result.get("action_taken") == "fix_rtl":
        return "integration_check"   # WP-76: edited RTL must be re-adopted first
    if result.get("action_taken") in ("retry", "fix_tb"):
        return "integration_dv"
    return END


route_after_integration_dv_decision.__edge_labels__ = {
    "integration_dv": "RETRY / FIX TB",
    "integration_check": "FIX RTL (re-adopt)",
    "init_tier": "REVISE",
    END: "DONE",
}


def _load_ers_validation_context(project_root: str) -> tuple[str, int]:
    """Load ERS context and count unique declared requirement records.

    Nested fields such as ``covers`` are traceability references, not new
    requirements. A requirement object therefore contributes one identity and
    its metadata is not recursively counted. Exact duplicate coded IDs or
    uncoded requirement strings contribute once.
    """
    ers_path = Path(project_root) / ".coresmith" / "ers_spec.json"
    if not ers_path.exists():
        return "", 0

    raw = ers_path.read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return raw, 0

    ers = data.get("ers", data)
    requirement_identities: set[str] = set()

    def _identity(text: str, *, explicit_id: bool = False) -> str:
        normalized = " ".join(text.split())
        if explicit_id:
            return f"id:{normalized.casefold()}"
        # Infer IDs only from conventional all-uppercase coded prefixes.
        # Ordinary prose may begin with a hyphenated word (for example,
        # "Single-outstanding ...") and must remain a distinct text record.
        coded = re.match(
            r"^([A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+)(?=\s*:|\s|$)",
            normalized,
        )
        if coded:
            return f"id:{coded.group(1).casefold()}"
        # Signal names can be case-sensitive, so do not case-fold prose.
        return f"text:{normalized}"

    def _count_value(value) -> None:
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    if item.strip():
                        requirement_identities.add(_identity(item))
                elif isinstance(item, dict):
                    _count_value(item)
        elif isinstance(value, dict):
            declared_id = value.get("id")
            requirement = value.get("requirement")
            if isinstance(declared_id, str) and declared_id.strip():
                requirement_identities.add(
                    _identity(declared_id, explicit_id=True)
                )
                return
            if isinstance(requirement, str) and requirement.strip():
                requirement_identities.add(_identity(requirement))
                return
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

    return json.dumps(data, indent=2), len(requirement_identities)


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


_HARNESS_FAILURE_FINGERPRINTS = (
    ("syntax error near unexpected token",
     "shell syntax error in a make recipe -- the harness/Makefile is broken; "
     "no design signal was ever evaluated"),
    ("bash: -c: line",
     "shell error while composing the sim command -- harness/Makefile bug"),
    ("No module named test_",
     "cocotb TB module failed to import at 0 ns (TB copy/stem regression)"),
    ("ModuleNotFoundError",
     "TB import failure at load time -- harness environment bug"),
    ("cocotb-config: command not found",
     "cocotb toolchain missing from PATH -- environment bug"),
    ("verilator: command not found",
     "verilator missing from PATH -- environment bug"),
)


def _harness_failure_fingerprint(sim_log: str) -> str | None:
    """Match a DV sim log against unambiguous HARNESS-class failure
    fingerprints (sim never produced a design-level verdict). Returns the
    explanation string, or None when the failure could be design-related."""
    text = sim_log or ""
    for needle, why in _HARNESS_FAILURE_FINGERPRINTS:
        if needle in text:
            return why
    return None


def _harness_audit_fastpath_enabled() -> bool:
    """CORESMITH_HARNESS_AUDIT_FASTPATH (default ON): skip the LLM contract
    audit when the sim log carries an unambiguous harness-class fingerprint.
    Runtime profile 2026-08-26: 4 of arm U's 6 validation audits (138-252s
    each) and one 900s audit timeout re-diagnosed shell errors, not RTL --
    the audit adds latency and a stale-verdict risk while the right action
    (fix the harness) is already determined. Set =0 to restore an
    unconditional audit."""
    return (os.environ.get("CORESMITH_HARNESS_AUDIT_FASTPATH", "1") or "1") != "0"


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
        },
    }
    context_path.write_text(json.dumps(context, indent=2), encoding="utf-8")
    # Identity of THIS failure, captured before the audit runs. It is stamped
    # into the verdict below so every later reader can tell whether the audit
    # in front of it is about the failure in front of it.
    context_stamp = _context_fingerprint(str(context_path))
    call_start = _time.time()

    # Per-attempt audit history (arm S/M retro: contract_audit/ kept ONE
    # file, overwritten every attempt -- operators could not see that early
    # audits said "CAVLC order" while late ones said "read rendezvous"
    # without excavating codex_turns.jsonl). Archive the previous audit
    # before this attempt writes.
    if output_path.exists():
        try:
            _stamp = int(output_path.stat().st_mtime)
            output_path.rename(
                audit_dir / f"{safe_stage}_contract_audit_{_stamp}.json")
        except OSError:
            output_path.unlink(missing_ok=True)

    # Harness-class fast-path (CORESMITH_HARNESS_AUDIT_FASTPATH, default ON):
    # an unambiguous harness failure (shell error in a make recipe, TB import
    # failure at 0 ns, missing toolchain binary) cannot be design-related --
    # the LLM audit there costs 138-900s and re-derives "fix the harness".
    _fp = _harness_failure_fingerprint(sim_log)
    if _fp and _harness_audit_fastpath_enabled():
        result = ContractAuditAgent._default_result(stage, str(context_path))
        result.update({
            "category": "DV_PROCESS_ERROR",
            "contract_failure": False,
            "local_fix_possible": True,
            "confidence": 0.9,
            "recommended_action": "fix_tb",
            "suggested_fix": _fp,
            "outer_agent_summary": (
                "HARNESS-CLASS failure (LLM audit skipped by fast-path): "
                f"{_fp}. The simulation never produced a design-level "
                f"verdict; fix the harness and re-run. Sim log: "
                f"{sim_log_path or '(inline tail in failure context)'}"
            ),
        })
        result["first_divergence"]["summary"] = _fp
        result[CONTRACT_AUDIT_STAMP_KEY] = context_stamp
        result["audited_at"] = round(call_start, 3)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        write_graph_event(project_root, "Contract Audit", "graph_node_exit", {
            "stage": stage,
            "category": "DV_PROCESS_ERROR",
            "contract_failure": False,
            "recommended_action": "fix_tb",
            "confidence": 0.9,
            "audit_path": str(output_path),
            "audit_fastpath": True,
        })
        log(f"  [CONTRACT-AUDIT] {stage}: harness-class failure -- "
            "fast-path verdict, LLM audit skipped "
            "(CORESMITH_HARNESS_AUDIT_FASTPATH=0 restores it)", YELLOW)
        result["audit_path"] = str(output_path)
        result["context_path"] = str(context_path)
        return result

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
    if result.get("candidate_stale") or result.get("action_taken") == "fix_rtl":
        return "integration_check"   # WP-76
    if result.get("pending_decision"):
        return "integration_dv_decision"
    if result.get("action_taken") in ("retry", "fix_tb"):
        return "integration_dv"
    return END


route_after_integration_dv.__edge_labels__ = {
    "validation_dv": "Validation DV",
    "integration_dv_decision": "Park for decision",
    "integration_check": "Re-adopt (RTL changed)",
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
            from orchestrator.harness.task_adapter import run_task_adapter
            from orchestrator.langgraph.acceptance_dv import run_acceptance_dv

            # WP-41: a task ADAPTER (inputs/task_adapter.py) runs the task's
            # own driver/checker on the candidate and is the acceptance
            # authority when present; the engine's native stream harness is
            # the fallback for tasks without one.
            _acc = await asyncio.to_thread(
                run_task_adapter, pr, top_rtl_path, block_rtl_paths,
            )
            if _acc is None:
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
                # WP-19: park for the chip lead exactly like a simulation
                # failure. Falling through returned a FAIL signoff with no
                # decision and no fix loop (Arm F-3 re-drive).
                _acc_cases = _acc.get("cases") or []
                _acc_lines = [
                    f"  {c.get('name')}: ok={c.get('ok')} cycles={c.get('cycles')} "
                    f"rtl_bytes={c.get('rtl_bytes')} "
                    f"{c.get('note') or c.get('criterion') or ''}"
                    for c in _acc_cases
                ]
                _acc_log = (
                    "TASK ACCEPTANCE FAILED (the task's declared oracle on the "
                    "assembled candidate):\n" + "\n".join(_acc_lines)
                    + "\nCaptured oracle artifacts: "
                    + (_acc.get("captured_dir") or str(Path(pr) / ".coresmith" / "acceptance_dv"))
                    + "\nViolations: " + json.dumps(
                        _acc.get("violations", []), default=str)[:1500]
                )
                _acc_oracle = bool(_acc.get("oracle_incomplete"))
                _acc_audit = {
                    "category": ("ACCEPTANCE_ORACLE_INCOMPLETE" if _acc_oracle
                                 else "ACCEPTANCE_DV_FAILURE"),
                    "local_fix_possible": None,
                    "recommended_action": "retry" if _acc_oracle else "fix_rtl",
                    "affected_blocks": [],
                    "outer_agent_summary": (
                        f"{sum(1 for c in _acc_cases if c.get('ok') is False)}/"
                        f"{len(_acc_cases)} acceptance case(s) fail the task's "
                        "declared oracle (per-case kind/detail above). Earlier "
                        "block-level checks passed, so the defect is in what they "
                        "did not measure end-to-end: run the task's checker on the "
                        "captured artifacts to localise it."
                    ),
                    "suggested_fix": (
                        "Run the task's checker offline on the captured "
                        "artifacts, read its error (which unit / sample / field), "
                        "map that to the responsible block, "
                        "fix the RTL (fix_rtl) or the block's spec (revise)."
                    ),
                }
                _acc_payload = {
                    "type": "validation_dv_failure",
                    "phase": "acceptance_dv",
                    "design_name": design_name,
                    "top_rtl_path": top_rtl_path,
                    "testbench_path": "",
                    "test_count": len(_acc_cases),
                    "requirement_count": 0,
                    "sim_log": _acc_log[-3000:],
                    "sim_log_path": str(Path(pr) / ".coresmith" / "acceptance_dv.json"),
                    "block_rtl_paths": block_rtl_paths,
                    "contract_audit": _acc_audit,
                    "contract_audit_path": "",
                    "acceptance_dv": {k: v for k, v in _acc.items()},
                    # WP-38: no fix_tb -- the acceptance oracle is task-owned;
                    # an oracle problem offers only retry/abort.
                    "supported_actions": (["retry", "abort"] if _acc_oracle
                                          else ["retry", "fix_rtl", "revise", "abort"]),
                    "outer_agent_guidance": (
                        ("The acceptance ORACLE did not complete (kind="
                         f"{_acc.get('kind')}): {_acc.get('reason')}. This is NOT "
                         "an RTL verdict. Do not change RTL for it; the operator "
                         "fixes the adapter / toolchain, then retry.")
                        if _acc_oracle else
                        ("The task's acceptance oracle rejected the chip. Per "
                         "case: status=1 (watchdog) means the case never "
                         "completed within its cycle budget; criterion="
                         "acceptance_predicate / task_adapter_functional means "
                         "the output was wrong; kind=budget_fail means the "
                         "output was right but over the task's cycle budget "
                         "(throughput) -- a real failure of the published "
                         "grader, fixed in the RTL's architecture, not by "
                         "changing the budget (and if an internal ERS/KPI "
                         "requirement demands the slower cadence, THAT "
                         "requirement is wrong: revise it); kind=boundary_mismatch means "
                         "the candidate's top module is not the task's graded "
                         "boundary (fix the integration, never the adapter). This "
                         "is the published grader's verdict class -- there is no "
                         "testbench to relax and fix_tb is not offered. Grade the "
                         "captured output offline, localise the block, fix it "
                         "(fix_rtl / revise), then retry.")
                    ),
                    "reference_files": {
                        "top_rtl": top_rtl_path,
                        "acceptance_dv": str(Path(pr) / ".coresmith" / "acceptance_dv.json"),
                        "captured_streams": (_acc.get("captured_dir") or str(Path(pr) / ".coresmith" / "acceptance_dv")),
                        "ers": str(Path(pr) / ".coresmith" / "ers_spec.json"),
                    },
                }
                write_graph_event(pr, "Validation DV", "graph_node_exit", {
                    "action": "pending_decision", "passed": False,
                    "phase": "acceptance_dv", "test_count": len(_acc_cases),
                })
                return {"validation_dv_result": {
                    "passed": False,
                    "pending_decision": True,
                    "interrupt_payload": _acc_payload,
                    "error": "RTL Acceptance DV failed: "
                             + str(_acc.get("reason")),
                    "phase": "acceptance_dv",
                    "test_count": len(_acc_cases),
                    "requirement_count": 0,
                    "testbench_path": "",
                    "design_name": design_name,
                    "contract_audit": _acc_audit,
                    "contract_audit_path": "",
                    "acceptance_dv": {k: v for k, v in _acc.items()
                                      if k != "cases"},
                    "violations": _acc.get("violations", []),
                }, "pipeline_done": False}
        except Exception as _exc:  # noqa: BLE001
            # WP-52: a required oracle that RAISES parks as oracle_incomplete;
            # it never "skips" into the rest of signoff.
            log(f"  [ACCEPTANCE-DV] gate raised: {_exc!r} -- parking", RED)
            _exc_acc = {
                "passed": False, "skipped": False, "oracle_incomplete": True,
                "kind": "infrastructure_error",
                "reason": f"acceptance gate raised: {_exc!r}", "cases": [],
                "violations": [{"type": "acceptance_dv_failure",
                                "criterion": "acceptance_dv_oracle_incomplete",
                                "kind": "infrastructure_error",
                                "suggested_fix": "Not an RTL verdict: the acceptance gate "
                                                 "itself failed. Fix the adapter / engine, "
                                                 "then retry."}],
            }
            _exc_audit = {
                "category": "ACCEPTANCE_ORACLE_INCOMPLETE", "local_fix_possible": None,
                "recommended_action": "retry", "affected_blocks": [],
                "outer_agent_summary": f"the acceptance gate raised: {str(_exc)[:400]}",
                "suggested_fix": "retry after the operator fixes the adapter / engine",
            }
            write_graph_event(pr, "Validation DV", "graph_node_exit", {
                "action": "pending_decision", "passed": False,
                "phase": "acceptance_dv", "error": str(_exc)[:300],
            })
            return {"validation_dv_result": {
                "passed": False, "pending_decision": True,
                "interrupt_payload": {
                    "type": "validation_dv_failure", "phase": "acceptance_dv",
                    "design_name": design_name, "top_rtl_path": top_rtl_path,
                    "testbench_path": "", "test_count": 0, "requirement_count": 0,
                    "sim_log": f"acceptance gate raised: {_exc!r}"[-3000:],
                    "sim_log_path": "", "block_rtl_paths": block_rtl_paths,
                    "contract_audit": _exc_audit, "contract_audit_path": "",
                    "acceptance_dv": _exc_acc,
                    "supported_actions": ["retry", "abort"],
                    "outer_agent_guidance": (
                        "The acceptance ORACLE raised an exception (see sim_log). "
                        "This is NOT an RTL verdict: do not change RTL for it; the "
                        "operator fixes the adapter / engine, then retry."),
                    "reference_files": {"top_rtl": top_rtl_path},
                },
                "error": f"acceptance gate raised: {_exc!r}",
                "phase": "acceptance_dv", "test_count": 0, "requirement_count": 0,
                "testbench_path": "", "design_name": design_name,
                "contract_audit": _exc_audit, "contract_audit_path": "",
                "acceptance_dv": _exc_acc, "violations": _exc_acc["violations"],
            }, "pipeline_done": False}

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
                **(previous_dv.get("tb_writer_flags") or {}),
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
                tb_result = await generate_validation_testbench(
                    project_root=_pr(state),
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
                            supported_actions=["retry", "fix_rtl", "fix_tb",
                                               "revise", "abort"],
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
                    # See the integration_dv tb_generation park: tier
                    # regeneration must be reachable from here too.
                    "revise",
                    "abort",
                ],
                "outer_agent_guidance": (
                    "Validation DV could not generate a usable cocotb "
                    "testbench for the measurable ERS/KPI requirements. "
                    "Diagnose whether the failure is missing ERS/KPI detail, "
                    "an invalid top-level contract, or a validation testbench "
                    "generation bug. Use action='fix_tb' when the validation "
                    "testbench prompt/generator needs repair, action='fix_rtl' "
                    "when RTL/top contracts must change, action='revise' when "
                    "the uArch specs themselves must change (appends your "
                    "feedback to the affected specs and re-runs the tiers), "
                    "or action='retry' "
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
                    # WP-14b: keep the canonical TB path so a chip-lead fix_tb
                    # re-admits the edited file instead of regenerating.
                    "testbench_path": (str(_canon_tb) if (_canon_tb := Path(pr) / "tb" / "validation" / "test_chip_top_validation.py").exists() else ""),
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
            sim_scope="validation", project_root=_pr(state),
        )
        if _stale_candidate(sim_result):
            log("  [VALIDATION-DV] candidate manifest is STALE (RTL changed after "
                "adoption) -- re-running the integration check to re-adopt "
                "before simulating", YELLOW)
            write_graph_event(pr, "Validation DV", "candidate_stale", {
                "log": str(sim_result.get("log", ""))[:300]})
            return {"validation_dv_result": _reintegrate_result("validation_dv", sim_result),
                    "pipeline_done": False}

        passed = sim_result.get("passed", False)
        sim_log = sim_result.get("log", "")

        # Same owner-certification contract as integration DV: not_declared is
        # scope evidence only and never flips a passing simulation to failed.
        # run3-followups: same contract as integration_dv -- the gate runs on
        # EVERY passing cycle (operator-reused TBs get MORE scrutiny) and every
        # evaluated outcome logs a verdict; a pass returns a dict, never None.
        _mg = None
        if passed:
            _mg = _maxgeo_gate_verdict(pr, tb_path, tb_result, sim_result=sim_result)
            if _mg is not None:
                write_graph_event(pr, "Maximum Geometry", "maxgeo_verdict", _mg)
            if reuse_existing_tb and _mg is not None:
                log("  [VALIDATION-DV] MAX-GEOMETRY gate: evaluating an "
                    "OPERATOR-REUSED testbench (fix_tb/fix_rtl) -- operator "
                    "edits get more scrutiny, not less.", YELLOW)
            if _mg is not None and _mg.get("verdict") == "pass":
                log("  [VALIDATION-DV] MAX-GEOMETRY gate PASS -- every "
                    f"declared maximum was covered by executed owner cases: "
                    f"{_mg.get('executed_maximum_cases', [])}", GREEN)
                write_graph_event(pr, "Validation DV", "maxgeo_gate_pass", {
                    "gate": "maxgeo",
                    "executed_maximum_cases": _mg.get("executed_maximum_cases", []),
                })
            elif _mg is not None and _mg.get("verdict") == "not_declared":
                log(f"  [VALIDATION-DV] MAX-GEOMETRY not_declared -- {_mg['reason']}", YELLOW)
            elif _mg is not None:
                passed = False
                sim_log = ((sim_log + "\n\n") if sim_log else "") + _mg["reason"]
                span.set_attribute("maxgeo_gate_failed", True)
                log("  [VALIDATION-DV] MAX-GEOMETRY gate FAILED -- flipping DV "
                    f"to failed: uncovered={_mg.get('uncovered_dims', {})}", RED)

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
                        "max_geometry": _mg,
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
                        "max_geometry": _mg,
                        "die_total_mm2": round(_roll.total_um2 / 1e6, 4),
                        "die_budget_mm2": _roll.die_budget_mm2,
                        "test_count": test_count,
                        "design_name": design_name,
                    },
                    "pipeline_done": False,
                }

            log(f"\n{'='*60}", GREEN)
            log("  VALIDATION DV PASSED", GREEN)
            log(
                f"  {test_count} tests; {requirement_count} unique ERS "
                "requirement records supplied as validation context",
                GREEN,
            )
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
                    "max_geometry": _mg,
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
            "max_geometry": _mg,
            "block_rtl_paths": block_rtl_paths,
            "contract_audit": contract_audit,
            "contract_audit_path": contract_audit.get("audit_path", ""),
            "supported_actions": [
                "retry",
                "fix_rtl",
                "fix_tb",
                "revise",
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
                "tb_writer_flags": _tb_writer_flags(tb_result),
                "sim_log_path": sim_result.get("log_path", ""),
                "max_geometry": _mg,
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
        "supported_actions": ["retry", "fix_rtl", "fix_tb", "revise", "abort"],
    }
    response = (await _resolve_interrupt(payload)) or {}
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
    elif action == "revise":
        revised = _apply_revise_uarch(
            pr, response, dict(payload.get("contract_audit") or {}),
            "validation_dv")
        dv_result["revised_blocks"] = revised
        log(f"  [VALIDATION-DV] Revise: uArch feedback appended to "
            f"{revised or '[] (no specs matched)'}; "
            + ("re-entering only those blocks" if revised
               else "re-running every tier"),
            YELLOW)
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
        **({"current_tier_index": 0,
            "revise_blocks": ({n: False for n in revised} or None)}
           if action == "revise" else {}),
    }


def route_after_validation_dv_decision(state: OrchestratorState) -> str:
    """Route the operator's decision back into validation DV or terminate."""
    result = state.get("validation_dv_result") or {}
    if result.get("action_taken") == "revise":
        return "init_tier"
    if result.get("action_taken") == "fix_rtl":
        return "integration_check"   # WP-76: edited RTL must be re-adopted first
    if result.get("action_taken") in ("retry", "fix_tb"):
        return "validation_dv"
    return END


route_after_validation_dv_decision.__edge_labels__ = {
    "validation_dv": "RETRY / FIX TB",
    "integration_check": "FIX RTL (re-adopt)",
    "init_tier": "REVISE",
    END: "DONE",
}


def route_after_validation_dv(state: OrchestratorState) -> str:
    """Route after validation DV: terminal frontend pipeline."""
    result = state.get("validation_dv_result") or {}
    if result.get("candidate_stale") or result.get("action_taken") == "fix_rtl":
        return "integration_check"   # WP-76
    if result.get("pending_decision"):
        return "validation_dv_decision"
    if result.get("action_taken") in ("retry", "fix_tb"):
        return "validation_dv"
    return END


route_after_validation_dv.__edge_labels__ = {
    "validation_dv_decision": "Park for decision",
    "integration_check": "Re-adopt (RTL changed)",
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
    # Candidate records belong to adoption/invalidation, and must survive
    # reporting so both automatic and later explicit backend starts can read them.
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
        # Opt-in labeled SFT dataset (CORESMITH_EMIT_SFT=1): publish
        # <run>/sft/ from the verified artifacts. Never fails the report.
        try:
            from orchestrator.langgraph.sft_export import (
                emit_sft_dataset,
                sft_enabled,
            )
            if sft_enabled():
                _sft = emit_sft_dataset(pr)
                if _sft:
                    _cnt = ", ".join(
                        f"{k}={v}" for k, v in _sft["counts"].items())
                    log(f"  [SFT] labeled dataset: {_sft['total_pairs']} "
                        f"pairs ({_cnt}) -> {root / 'sft'}", GREEN)
        except Exception as _sft_exc:  # noqa: BLE001
            log(f"  [SFT] dataset emission failed (non-fatal): {_sft_exc}",
                YELLOW)
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

    """
    block_subgraph = build_block_subgraph().compile()

    orchestrator = StateGraph(OrchestratorState)

    # Nodes (shared by both topologies)
    orchestrator.add_node("init_tier", init_tier_node)
    orchestrator.add_node("process_block", block_subgraph)
    orchestrator.add_node("integration_review_prepare", integration_review_prepare_node)
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
    orchestrator.add_edge("process_block", "integration_review_prepare")
    orchestrator.add_edge("integration_review_prepare", "integration_review")
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
            "integration_check": "integration_check",
            END: "final_report",
        },
    )
    orchestrator.add_conditional_edges(
        "integration_dv_decision", route_after_integration_dv_decision,
        {
            "integration_dv": "integration_dv",
            "integration_check": "integration_check",
            "init_tier": "init_tier",
            END: "final_report",
        },
    )
    orchestrator.add_conditional_edges(
        "validation_dv", route_after_validation_dv,
        {
            "validation_dv": "validation_dv",
            "validation_dv_decision": "validation_dv_decision",
            "integration_check": "integration_check",
            END: "final_report",
        },
    )
    orchestrator.add_conditional_edges(
        "validation_dv_decision", route_after_validation_dv_decision,
        {
            "validation_dv": "validation_dv",
            "integration_check": "integration_check",
            "init_tier": "init_tier",
            END: "final_report",
        },
    )

    orchestrator.add_conditional_edges("advance_tier", route_next_tier)
    orchestrator.add_conditional_edges("integration_check", route_after_integration)

    return orchestrator.compile(checkpointer=checkpointer)
