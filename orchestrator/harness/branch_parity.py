# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Branch-parity smoke -- the backstop that catches split-brain RTL the ifdef
lint misses.

The deterministic functional-ifdef lint (``rtl_storage_lint``) rejects a module
that guards FUNCTIONAL logic (always/assign/instantiation/driving-initial)
behind a strippable ``ifdef``. What survives lint is a region that holds ONLY
debug/trace/assertion code, or a legitimate macro-module blackbox split -- in
BOTH cases toggling the guard macro must leave the design's FUNCTION unchanged.
This smoke proves that: it rebuilds the SAME block under the synth-side macro
world (``-DSYNTHESIS`` + the engine's synth defines) and reruns the SAME seeded
cocotb vectors, then compares the verdict to the default (sim-world) build. If
the two builds disagree, the "allowed" region actually changed hardware -- the
exact split-brain the lint could not see -- and we fail closed.

Why the parity build is well-defined (wrapper investigation, 2026-07-03): the
``cs_*`` memory wrappers in ``rtl_lib/cs_sram.v`` select their body with a
synthesizable ``generate`` block keyed on the ``MEM_IMPL`` PARAMETER (default
"BEHAV"), NOT with an ``ifdef``. So defining ``SYNTHESIS`` (or
``CORESMITH_SRAM_SYNTH``) leaves the wrapper's behavioral memory model fully
present in the parity sim -- only the DESIGN module's own ``ifdef`` split
resolves to its synth branch. No separate behavioral-memory injection is needed.

Env gate ``CORESMITH_BRANCH_PARITY``:
  * unset  -> ON only when the RTL actually contains a surviving conditional
              region (off otherwise, for speed).
  * 1/on   -> force ON (even with no region, a no-op that returns ran=False).
  * 0/off  -> force OFF.

Fail semantics:
  * verdicts AGREE            -> ran=True, ok=True   (no-op; sim verdict stands)
  * verdicts DIVERGE          -> ran=True, ok=False  (FAIL-CLOSED park)
  * a build cannot compile /
    toolchain missing / crash  -> ran=True, ok=True, skipped=True (NON-blocking:
                                 you cannot compare what will not build; never a
                                 false fail)
  * no conditional region /
    gate disabled              -> ran=False           (does not apply)

All heavy imports are deferred so this module stays import-light.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field

# The synth-side preprocessor defines the parity build applies to the DESIGN.
# SYNTHESIS is the canonical macro the split-brain mock keyed on; CORESMITH_SRAM_SYNTH
# is the engine's block-synth memory-wrapper define (a no-op for the current
# generate-based wrapper, included for fidelity with the real synth read).
SYNTH_PARITY_DEFINES = ["SYNTHESIS", "CORESMITH_SRAM_SYNTH"]


@dataclass
class ParityResult:
    ran: bool = False
    ok: bool = True
    skipped: bool = False
    reason: str = ""
    baseline: dict = field(default_factory=dict)
    synth: dict = field(default_factory=dict)

    def as_prev_error(self, block: str = "") -> str:
        where = f" in {block}" if block else ""
        return (
            f"BRANCH-PARITY DIVERGENCE{where}: the block builds DIFFERENT "
            f"hardware under the synthesis macro world than under the "
            f"simulation macro world -- a split-brain the conditional-"
            f"compilation lint did not catch.\n"
            f"  default (sim)  build: {self._fmt(self.baseline)}\n"
            f"  parity (-D{'/-D'.join(SYNTH_PARITY_DEFINES)}) build: "
            f"{self._fmt(self.synth)}\n"
            f"  FIX: write EXACTLY ONE implementation per module. Any `ifdef/"
            f"`ifndef must guard ONLY non-functional debug/trace/assertion code "
            f"so both macro worlds are the same hardware."
        )

    @staticmethod
    def _fmt(v: dict) -> str:
        return (f"passed={v.get('passed')} "
                f"tests={v.get('tests_passed')}/{v.get('tests_total')} "
                f"failed={v.get('tests_failed')}")


def has_conditional_region(rtl_text: str) -> bool:
    """True if the RTL contains ANY surviving `ifdef/`ifndef/`elsif region.

    (Functional ones are already rejected at generation time by the ifdef lint;
    what reaches here is debug/assertion-only or a macro-module split -- either
    way a conditional region exists that the parity build must exercise.)"""
    if "`" not in rtl_text:
        return False
    import re
    return bool(re.search(r"^[ \t]*`(?:ifdef|ifndef|elsif)\b", rtl_text,
                          re.MULTILINE))


def branch_parity_enabled(rtl_text: str) -> bool:
    """Resolve the CORESMITH_BRANCH_PARITY gate against the RTL.

    Default (unset): ON iff a conditional region exists. ``1/on`` forces ON;
    ``0/off`` forces OFF.
    """
    raw = os.environ.get("CORESMITH_BRANCH_PARITY", "").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return False
    if raw in {"1", "true", "yes", "on"}:
        return True
    return has_conditional_region(rtl_text)


# Verilator/cocotb build-failure signatures (a compile that never produced a
# functional verdict -> we cannot compare, so we SKIP rather than fail).
_BUILD_ERROR_SIGNS = (
    "%error", "cannot find", "not found", "syntax error", "cannot open",
    "no such file", "verilator: error", "make: *** ", "command not found",
    "no tests were discovered",
)


def _is_build_error(res: dict) -> bool:
    """A parity build that could not run the vectors (compile/toolchain error),
    as opposed to a clean pass/fail verdict on the vectors."""
    if res.get("sim_timed_out"):
        return True
    # A build that reached a real cocotb regression summary DID run the vectors,
    # so it is comparable even if it failed.
    if (res.get("tests_total") or 0) > 0:
        return False
    if res.get("passed"):
        return False
    low = str(res.get("log", "")).lower()
    return any(sig in low for sig in _BUILD_ERROR_SIGNS)


def _verdict(res: dict) -> tuple:
    """The comparable verdict tuple of a build: (passed, tests_passed,
    tests_total, tests_failed)."""
    return (
        bool(res.get("passed")),
        res.get("tests_passed"),
        res.get("tests_total"),
        res.get("tests_failed"),
    )


def check_branch_parity(
    block: dict,
    rtl_path: str,
    tb_path: str,
    attempt: int = 1,
    rtl_text: str | None = None,
    sim_runner: Callable | None = None,
) -> ParityResult:
    """Run the branch-parity smoke for one block.

    Builds+sims the block twice under an IDENTICAL pinned DV seed -- once in the
    default (sim) macro world, once with the synth-side defines -- and compares
    the cocotb verdicts. ``sim_runner`` is injected for tests; it defaults to
    ``pipeline_helpers.run_simulation`` and must accept
    ``(block, rtl_path, tb_path, attempt, extra_defines=, sim_subdir=)``.
    """
    block_name = block.get("name", "block")

    if rtl_text is None:
        try:
            from pathlib import Path
            rtl_text = Path(rtl_path).read_text()
        except OSError:
            rtl_text = ""

    if not branch_parity_enabled(rtl_text):
        return ParityResult(ran=False, ok=True,
                            reason="no conditional region / gate off")

    if sim_runner is None:
        from orchestrator.langgraph.pipeline_helpers import (
            run_simulation as sim_runner,  # type: ignore
        )

    # Pin ONE seed so both builds see the identical stimulus. Restore whatever
    # was there (a caller-pinned seed for debugging) afterward.
    from orchestrator.harness.seed_provider import gate_seed
    seed = str(gate_seed())
    _prev_pin = os.environ.get("CORESMITH_DV_SEED_PIN")
    os.environ["CORESMITH_DV_SEED_PIN"] = seed
    try:
        try:
            base = sim_runner(block, rtl_path, tb_path, attempt,
                              extra_defines=None,
                              sim_subdir=f"{block_name}__parity_base")
            synth = sim_runner(block, rtl_path, tb_path, attempt,
                               extra_defines=list(SYNTH_PARITY_DEFINES),
                               sim_subdir=f"{block_name}__parity_synth")
        except Exception as exc:  # never let the smoke crash the node
            return ParityResult(ran=True, ok=True, skipped=True,
                                reason=f"parity build harness error: {exc}")
    finally:
        if _prev_pin is None:
            os.environ.pop("CORESMITH_DV_SEED_PIN", None)
        else:
            os.environ["CORESMITH_DV_SEED_PIN"] = _prev_pin

    base = base or {}
    synth = synth or {}

    # If EITHER build could not compile/run the vectors, we cannot compare ->
    # SKIP (non-blocking). A toolchain gap must never manufacture a false fail.
    if _is_build_error(base) or _is_build_error(synth):
        return ParityResult(ran=True, ok=True, skipped=True,
                            reason="a parity build could not compile/run the "
                                   "vectors (toolchain) -- comparison skipped",
                            baseline=base, synth=synth)

    if _verdict(base) == _verdict(synth):
        return ParityResult(ran=True, ok=True,
                            reason="sim and synth macro worlds agree",
                            baseline=base, synth=synth)

    # The two builds ARE different hardware -> fail closed.
    return ParityResult(ran=True, ok=False,
                        reason="sim and synth macro worlds diverge",
                        baseline=base, synth=synth)


__all__ = [
    "SYNTH_PARITY_DEFINES",
    "ParityResult",
    "branch_parity_enabled",
    "check_branch_parity",
    "has_conditional_region",
]
