# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""A declared pin map RETIRES the pad-adapter block, before it is generated.

``orchestrator/architecture/pin_map.py`` made the shuttle's pin assignment DATA,
and ``generate_caravel_wrapper_top`` emits the routing from it -- then DROPS the
pad-adapter block, because the top now does the adapter's whole job. That fixed
assembly. It did not fix the flow: the block was still in the BLOCK QUEUE, so
the run kept paying to microarchitect, generate, lint and gate a module the chip
does not contain.

That cost is not theoretical. The adapter's module name is mandated to be
``user_project_wrapper``, which by convention means "the entire chip", and the
generator produced a full chip top instead of a leaf adapter 6 times out of 6
across three runs -- with explicit corrective feedback each time. The in-flow
conformance gate then correctly refused the shape (13 deviations: 6 missing
channel ports, 7 sibling instantiations) and parked the run twice
(``contract_conformance_unrepairable``). Both hands-off attempts died there.

Refusing the wrong shape after generating it is the right gate on the wrong
question. The design does not contain this block at all once a pin map covers
it, so the flow must not ask for it. This module answers, deterministically:

    Is block B the pad adapter, and does the PRD pin map COVER what B was
    contracted to translate?

Both halves are evidence, never a name:

* **Pad adapter** -- the block carrying the LOCKED Caravel boundary. Identified
  the way the rest of the engine already identifies it: an ``interfaces`` group
  in its architecture entry declaring all of io_in/io_out/io_oeb (the same
  ``_LOCKED_BOUNDARY_PORTS`` triple ``contract_conformance.check_block`` uses for
  ``locked_boundary``), or an ``rtl_target`` whose module is the Caravel top
  (the same ``CARAVEL_TOP_MODULE`` ``detect_wrapper_block`` prefers), or -- once
  RTL exists -- a module declaring that boundary. No block NAME is matched.

* **Coverage** -- every signal the interface contract says the block translates
  (each touching edge's ``fields`` + ``sideband_signals``, the union that
  ``contract_conformance._signal_names`` computes) must appear in the pin map,
  either as a mapped signal or as an output-enable. Six signals on the run that
  motivated this; the pin map maps five plus one ``oe``. Exactly covered.

PARTIAL coverage is NOT a retirement. A boundary where the top routes some pads
and a block routes the rest is a contradiction the operator has to resolve --
the two would fight over the same bus -- so :func:`plan_retirement` returns
``park=True`` and the caller raises an interrupt instead of guessing. ZERO
coverage means this pin map is not about this block; nothing is retired and the
flow is unchanged (loudly logged, because the assembler drops the adapter on a
valid pin map regardless and an operator should know the two disagree).

``CORESMITH_PINMAP_RETIRES_ADAPTER=0`` restores the previous behaviour: the
block is generated, and the assembler drops it afterwards as before.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

#: The externally mandated Caravel pad vector. Same triple as
#: ``contract_conformance._LOCKED_BOUNDARY_PORTS`` and
#: ``integration_helpers._PAD_IO_PORTS`` -- a block declaring all three carries
#: the locked boundary.
LOCKED_BOUNDARY_PORTS = ("io_in", "io_out", "io_oeb")

#: The shuttle-mandated top module name (mirrors
#: ``integration_helpers.CARAVEL_TOP_MODULE``; duplicated rather than imported so
#: this module stays import-light for the harness).
CARAVEL_TOP_MODULE = "user_project_wrapper"

#: Recorded in state / on disk as the block's skip reason.
RETIRE_REASON = "retired_by_pin_map"

#: Carried artifact, read by the final report.
ARTIFACT_NAME = "retired_blocks.json"

#: Written into the retired block's own directory so the reason is where a
#: reader of that block looks.
BLOCK_NOTE_NAME = "RETIRED_BY_PIN_MAP.md"


def retirement_enabled() -> bool:
    """Retire the pad adapter when a pin map covers it (default ON).

    ``CORESMITH_PINMAP_RETIRES_ADAPTER=0`` restores the previous behaviour.
    """
    return (os.environ.get("CORESMITH_PINMAP_RETIRES_ADAPTER", "1")
            or "1").strip().lower() not in {"0", "false", "no", "off"}


# ---------------------------------------------------------------------------
# Evidence: which block is the pad adapter?
# ---------------------------------------------------------------------------

def _architecture_blocks(project_root) -> dict:
    """``block_diagram.json`` entries by name (the richest block metadata).

    ``block_specs.json`` / ``block_queue.json`` carry only name/tier/rtl_target;
    the ``interfaces`` map that records the locked boundary lives here.
    """
    p = Path(project_root) / ".coresmith" / "block_diagram.json"
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return {}
    blocks = doc.get("blocks") if isinstance(doc, dict) else doc
    out: dict = {}
    for b in blocks or []:
        if isinstance(b, dict) and b.get("name"):
            out[str(b["name"])] = b
    return out


def _spec_declares_locked_boundary(spec: dict) -> bool:
    """Does this block's architecture entry declare the Caravel pad vector?

    Looks inside ``interfaces`` (``{group: {port: width}}``) and, defensively,
    at any top-level ``ports``/``signals`` mapping -- the pad triple is what is
    being looked for, not a particular schema.
    """
    if not isinstance(spec, dict):
        return False
    names: set[str] = set()

    def _harvest(obj) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                names.add(str(k))
                _harvest(v)
        elif isinstance(obj, list):
            for v in obj:
                if isinstance(v, str):
                    names.add(v)
                else:
                    _harvest(v)

    for key in ("interfaces", "ports", "signals", "io"):
        if key in spec:
            _harvest(spec.get(key))
    return all(p in names for p in LOCKED_BOUNDARY_PORTS)


def _rtl_target_module(spec: dict) -> str:
    """The module name a block's ``rtl_target`` file is named for."""
    target = str((spec or {}).get("rtl_target") or "").strip()
    return Path(target).stem if target else ""


def _rtl_declares_locked_boundary(project_root, spec: dict) -> bool:
    """Third-tier evidence: the block's RTL, when it already exists, declares
    the pad vector. Only consulted when the architecture metadata is silent --
    a retirement must not depend on RTL existing, since the point is to retire
    the block BEFORE any is generated."""
    target = str((spec or {}).get("rtl_target") or "").strip()
    if not target:
        return False
    for cand in (Path(target), Path(project_root) / target):
        try:
            if not cand.is_file():
                continue
            text = cand.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        from orchestrator.langgraph.contract_conformance import declared_ports
        ports = declared_ports(text, module=None)
        if all(p in ports for p in LOCKED_BOUNDARY_PORTS):
            return True
    return False


def pad_adapter_blocks(project_root, block_queue) -> list[str]:
    """Blocks carrying the locked Caravel pad boundary, in queue order.

    Structural, never by name. Returns [] for a non-Caravel design.
    """
    arch = _architecture_blocks(project_root)
    found: list[str] = []
    for entry in block_queue or []:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or entry.get("block_name") or "")
        if not name:
            continue
        spec = {**(arch.get(name) or {}), **entry}
        if (_spec_declares_locked_boundary(spec)
                or _rtl_target_module(spec) == CARAVEL_TOP_MODULE
                or _rtl_declares_locked_boundary(project_root, spec)):
            found.append(name)
    return found


# ---------------------------------------------------------------------------
# Evidence: what was the block contracted to translate, and is it covered?
# ---------------------------------------------------------------------------

def _contracts(project_root) -> list:
    p = Path(project_root) / ".coresmith" / "interface_contracts.json"
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return []
    c = d.get("contracts") if isinstance(d, dict) else d
    return c if isinstance(c, list) else []


def _edge_signals(edge: dict) -> list[str]:
    """Payload fields + sideband: a channel's full signal set.

    Same union ``contract_conformance._signal_names`` and
    ``integration_helpers._contract_signal_names`` compute; the schema splits
    them across two keys, which is why readers have concluded the contract does
    not record signal names at all.
    """
    out: list[str] = []
    for key in ("fields", "sideband_signals"):
        for f in edge.get(key) or []:
            n = f.get("name") if isinstance(f, dict) else f
            if n:
                out.append(str(n))
    seen, uniq = set(), []
    for n in out:
        if n not in seen:
            seen.add(n)
            uniq.append(n)
    return uniq


def contract_signals_for_block(project_root, block_name: str) -> list[str]:
    """Every signal the interface contract says this block carries.

    The union over every edge naming the block as producer or consumer -- i.e.
    exactly the set the conformance gate demands the block expose as ports, and
    therefore exactly what a pin map has to route for the block to be redundant.
    """
    out: list[str] = []
    seen: set[str] = set()
    for edge in _contracts(project_root):
        if not isinstance(edge, dict):
            continue
        if block_name not in (edge.get("producer_block"),
                              edge.get("consumer_block")):
            continue
        for n in _edge_signals(edge):
            if n not in seen:
                seen.add(n)
                out.append(n)
    return out


def pin_map_signal_names(pin_map) -> list[str]:
    """Every name a pin map binds: mapped signals plus their output-enables.

    The ``oe`` of an out entry is a real routed signal -- the top inverts it
    into ``io_oeb`` -- so a contract signal satisfied by an ``oe`` is covered.
    """
    out: list[str] = []
    seen: set[str] = set()
    for e in getattr(pin_map, "entries", []) or []:
        for n in (getattr(e, "signal", ""), getattr(e, "oe", "")):
            if n and n not in seen:
                seen.add(n)
                out.append(n)
    return out


@dataclass
class RetirementPlan:
    """What to do with the pad-adapter block, and the evidence for it."""

    block: str = ""
    retire: bool = False
    park: bool = False
    reason: str = ""
    contract_signals: list = field(default_factory=list)
    pin_map_signals: list = field(default_factory=list)
    covered: list = field(default_factory=list)
    uncovered: list = field(default_factory=list)
    message: str = ""

    @property
    def acted(self) -> bool:
        return self.retire or self.park


def plan_retirement(project_root, block_queue, pin_map=None) -> RetirementPlan:
    """Decide whether a declared pin map retires the pad-adapter block.

    ``pin_map`` defaults to ``load_pin_map(project_root)``. Returns a plan with
    ``retire`` (full coverage), ``park`` (partial coverage -- a contradiction),
    or neither (no pin map / no adapter / no contract edges / zero overlap).
    Never raises: every input is treated as untrusted data.
    """
    plan = RetirementPlan()
    if pin_map is None:
        from orchestrator.architecture.pin_map import load_pin_map
        pin_map = load_pin_map(project_root)
    if pin_map is None:
        plan.reason = "no pin_map declared in the PRD"
        return plan
    if not getattr(pin_map, "ok", False):
        plan.reason = ("pin_map is present but INVALID ("
                       + "; ".join(getattr(pin_map, "errors", []) or [])
                       + ") -- refusing to retire anything on a bad map")
        return plan

    adapters = pad_adapter_blocks(project_root, block_queue)
    if not adapters:
        plan.reason = ("no block carries the locked Caravel pad boundary -- "
                       "nothing for the pin map to replace")
        return plan
    plan.block = adapters[0]

    plan.contract_signals = contract_signals_for_block(project_root, plan.block)
    plan.pin_map_signals = pin_map_signal_names(pin_map)
    if not plan.contract_signals:
        plan.reason = (
            f"no interface-contract edge names '{plan.block}', so there is no "
            f"declared translation for the pin map to cover -- not retiring on "
            f"an unevidenced guess")
        return plan

    mapped = set(plan.pin_map_signals)
    plan.covered = [s for s in plan.contract_signals if s in mapped]
    plan.uncovered = [s for s in plan.contract_signals if s not in mapped]

    if not plan.uncovered:
        plan.retire = True
        plan.reason = RETIRE_REASON
        plan.message = (
            f"the PRD pin map routes all {len(plan.covered)} signal(s) "
            f"'{plan.block}' was contracted to translate "
            f"({', '.join(plan.covered)}) -- the chip top emits that routing "
            f"itself, so there is no adapter to generate")
        return plan

    if not plan.covered:
        # The pin map is about something else entirely. Not a partial boundary,
        # so not the contradiction the park exists for -- but say so, because
        # the assembler drops the adapter on ANY valid pin map and an operator
        # should not discover that disagreement at integration time.
        plan.reason = (
            f"the pin map maps {sorted(mapped)} and covers NONE of the "
            f"{len(plan.contract_signals)} signal(s) '{plan.block}' is "
            f"contracted to translate ({', '.join(plan.contract_signals)}) -- "
            f"not retiring; note that the integration assembler still drops a "
            f"pad adapter whenever a valid pin map is present")
        return plan

    plan.park = True
    plan.reason = "pin_map_partially_covers_pad_adapter"
    plan.message = (
        f"the PRD pin map covers {len(plan.covered)} of "
        f"{len(plan.contract_signals)} signal(s) '{plan.block}' is contracted "
        f"to translate: covered {plan.covered}, NOT covered {plan.uncovered}. "
        f"A boundary where the chip top routes some pads and a block routes "
        f"the rest is a contradiction -- both would drive the same bus. "
        f"Either extend prd.pin_map to cover {plan.uncovered}, or remove the "
        f"pin map and let the adapter own the whole boundary.")
    return plan


def apply_retirement(block_queue, plan: RetirementPlan) -> list:
    """The block queue with the retired block removed (a new list).

    Idempotent: re-applying to an already-reduced queue is a no-op, which is
    what makes it safe on every tier re-entry and every checkpoint resume.
    """
    if not plan or not plan.retire or not plan.block:
        return list(block_queue or [])
    return [b for b in (block_queue or [])
            if not (isinstance(b, dict)
                    and (b.get("name") or b.get("block_name")) == plan.block)]


# ---------------------------------------------------------------------------
# Artifacts: the reason has to survive the run
# ---------------------------------------------------------------------------

def _artifact_path(project_root) -> Path:
    return Path(project_root) / ".coresmith" / ARTIFACT_NAME


def read_retired_blocks(project_root) -> list:
    """Retirement records for this run (``[]`` when nothing was retired)."""
    try:
        doc = json.loads(_artifact_path(project_root).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return []
    rows = doc.get("retired") if isinstance(doc, dict) else doc
    return rows if isinstance(rows, list) else []


def record_retirement(project_root, plan: RetirementPlan) -> dict:
    """Persist the retirement: a carried artifact + a note in the block's dir.

    Returns the record (also the shape stored in orchestrator state). Writes are
    best-effort -- a bookkeeping failure must never fail the run -- but the
    record is returned either way so state still carries it.
    """
    record = {
        "block": plan.block,
        "reason": RETIRE_REASON,
        "skipped": True,
        "retired_at": time.time(),
        "contract_signals": list(plan.contract_signals),
        "pin_map_signals": list(plan.pin_map_signals),
        "covered_signals": list(plan.covered),
        "explanation": plan.message,
    }
    try:
        rows = [r for r in read_retired_blocks(project_root)
                if isinstance(r, dict) and r.get("block") != plan.block]
        rows.append(record)
        p = _artifact_path(project_root)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"retired": rows}, indent=2), encoding="utf-8")
    except OSError:
        pass
    try:
        bd = Path(project_root) / ".coresmith" / "blocks" / plan.block
        bd.mkdir(parents=True, exist_ok=True)
        (bd / BLOCK_NOTE_NAME).write_text(
            f"# {plan.block}: RETIRED ({RETIRE_REASON})\n"
            "\n"
            "This block was NOT microarchitected, generated, simulated or\n"
            "synthesized, and it is deliberately ABSENT from the assembled\n"
            "chip. It is not missing and it did not fail.\n"
            "\n"
            f"Why: {plan.message}.\n"
            "\n"
            "Contract signals this block was to translate:\n"
            + "".join(f"  - {s}\n" for s in plan.contract_signals)
            + "\nPin-map signals that route them instead:\n"
            + "".join(f"  - {s}\n" for s in plan.pin_map_signals)
            + "\nThe routing is emitted directly into the chip top by\n"
              "orchestrator/architecture/pin_map.py:emit_pin_routing, from\n"
              "prd_spec.json's structured `pin_map`. Set\n"
              "CORESMITH_PINMAP_RETIRES_ADAPTER=0 to generate this block\n"
              "again (the integration assembler then drops it after the fact,\n"
              "which is what this retirement replaces).\n",
            encoding="utf-8")
    except OSError:
        pass
    return record
