# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Deterministic, contract-faithful bus BFM library for coresmith DV.

Replaces the LLM-authored pin-driving BFM (which co-tunes to the DUT, so a
non-conformant design passes CoreSmith DV yet fails the real fixed host) with a
DUT-blind driver generated purely from a bus contract. Vertical slice: the
QSPI-slave accelerator chassis (the ``qspi_host`` protocol). Other bus
families are a follow-on -- the classifier/codegen split is designed to grow a
new ``*_contract`` + ``*_master_bfm`` per protocol.

Public API:
  * ``classify_bus_verdict(project_root, top_rtl_source, connections,
    top_module, top_rtl_path)`` -> ``BusVerdict`` (is this a QSPI-slave chassis,
    WHERE is the graded Caravel pin boundary, and will the integration sim
    actually drive it?). Pin-driving callers must use this one: it separates
    "not the graded boundary" (fail closed) from "not a QSPI design" (advisory).
  * ``classify_chip_bus(project_root, top_rtl_source, connections)`` ->
    ``QSPIContract | None``  (is this chip a QSPI-slave? DUT-blind)
  * ``find_pin_boundary(...)`` -> ``PinBoundary | None`` (which module declares
    io_in/io_out/io_oeb, and in which file)
  * ``boundary_gate_enabled()`` -> the fail-closed pin-boundary gate's env flag
  * ``build_plan_from_run(project_root, contract)`` -> ``StimulusPlan | None``
    (host-flow writes + expected OUT from the LLM golden reference)
  * ``plan_deterministic_dv(...)`` -> ``(contract, plan) | (None, None)``
  * ``render_integration_tb(contract, design_name, plan)`` -> TB source
  * ``write_deterministic_integration_tb(...)`` -> ``{tb_path, test_count, ...}``
"""

from __future__ import annotations

import os
from pathlib import Path

from .classifier import (
    STATUS_BOUNDARY_OFF_TOP,
    STATUS_CONTRADICTION,
    STATUS_NOT_QSPI,
    STATUS_QSPI_TOP,
    BusVerdict,
    PinBoundary,
    arch_indicates_qspi_slave,
    classify_bus_verdict,
    classify_chip_bus,
    describe_unmodeled_roles,
    detect_bus_roles,
    detect_dut_mastered_buses,
    find_pin_boundary,
)
from .codegen import (
    render_bfm_module,
    render_conformance_tb,
    render_conformance_test_fn,
    render_contract_literal,
    render_integration_tb,
    render_rom_bfm_module,
    render_rom_contract_literal,
)
from .qspi_contract import QSPIContract
from .qspi_master_bfm import QSPIMasterBFM
from .qspi_rom_bfm import QSPIRomContract, QSPIRomResponderBFM
from .stimulus import StimulusPlan, build_plan_from_run

__all__ = [
    "STATUS_BOUNDARY_OFF_TOP",
    "STATUS_CONTRADICTION",
    "STATUS_NOT_QSPI",
    "STATUS_QSPI_TOP",
    "BusVerdict",
    "PinBoundary",
    "QSPIContract",
    "QSPIMasterBFM",
    "QSPIRomContract",
    "QSPIRomResponderBFM",
    "StimulusPlan",
    "arch_indicates_qspi_slave",
    "boundary_gate_enabled",
    "build_plan_from_run",
    "classify_bus_verdict",
    "classify_chip_bus",
    "conformance_enabled",
    "describe_unmodeled_roles",
    "detect_bus_roles",
    "detect_dut_mastered_buses",
    "deterministic_bfm_enabled",
    "find_pin_boundary",
    "plan_deterministic_dv",
    "render_bfm_module",
    "render_conformance_tb",
    "render_conformance_test_fn",
    "render_contract_literal",
    "render_integration_tb",
    "render_rom_bfm_module",
    "render_rom_contract_literal",
    "write_deterministic_integration_tb",
    "write_qspi_conformance_tb",
]


def deterministic_bfm_enabled() -> bool:
    """True when ``CORESMITH_DETERMINISTIC_BFM`` is set truthy (default OFF).

    Flag-off keeps the existing LLM-BFM path byte-identical, per repo
    convention for behavior-changing env gates.
    """
    return (os.environ.get("CORESMITH_DETERMINISTIC_BFM", "0") or "0").strip() not in (
        "",
        "0",
        "false",
        "False",
        "no",
    )


def conformance_enabled() -> bool:
    """True when the QSPI-slave BUS-PROTOCOL conformance DV should run.

    ON when ``CORESMITH_QSPI_CONFORMANCE`` is set truthy, OR (by default) whenever
    the deterministic BFM is enabled -- so the deterministic-BFM campaign ALWAYS
    exercises the raw QSPI-slave bus contract, even for a chip whose compute-lane
    oracle cannot be modeled (the image codec case, where a golden host-flow plan
    could not be derived and the run silently fell back to the co-tuned LLM BFM,
    so a frontend missing 0x05 / mistiming the read turnaround was never caught).

    Explicit ``CORESMITH_QSPI_CONFORMANCE=0`` forces it off. With neither flag set
    the conformance DV stays off and behavior is byte-identical to before.
    """
    v = (os.environ.get("CORESMITH_QSPI_CONFORMANCE", "") or "").strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return deterministic_bfm_enabled()


def boundary_gate_enabled() -> bool:
    """True when the QSPI PIN-BOUNDARY gate fails the run CLOSED (default ON).

    The gate fires when the run's spec says the external bus is QSPI but the
    module integration DV elaborates does NOT expose the graded Caravel pin
    boundary -- either because no ``io_in``/``io_out``/``io_oeb`` boundary exists
    anywhere (spec/RTL contradiction) or because it lives on another module (the
    ``user_project_wrapper`` the grader actually drives). Proceeding there means
    silently keeping the DUT-co-tuned LLM BFM in exactly the runs most likely to
    be non-conformant, so it is a gate, not a log line.

    Overridable per the repo's gate idiom: ``CORESMITH_QSPI_BOUNDARY_GATE=0``
    (or the single global rollback knob ``CORESMITH_GATE_FAIL_OPEN=1``) restores
    the pre-fix advisory-only behavior -- the caller then keeps the LLM BFM but
    records a carried-forward defect, so the bypass is never silent.

    Scope note: the gate only reaches a run that opted into the deterministic
    BFM campaign at all (see :func:`conformance_enabled`); with both BFM flags
    unset this module is never consulted and behavior is byte-identical.
    """
    try:
        from orchestrator.profile import flag_enabled
        on = flag_enabled("CORESMITH_QSPI_BOUNDARY_GATE", default=True)
    except Exception:  # noqa: BLE001 - a profile import hiccup must not break a gate
        raw = (os.environ.get("CORESMITH_QSPI_BOUNDARY_GATE") or "").strip().lower()
        on = raw not in ("0", "false", "no", "off")
    if not on:
        return False
    try:
        from orchestrator.langgraph.gate_guard import gate_fail_open_enabled
        if gate_fail_open_enabled():
            return False
    except Exception:  # noqa: BLE001
        pass
    return True


def plan_deterministic_dv(
    project_root: str,
    top_rtl_source: str,
    connections: list | None = None,
    top_module: str = "",
    top_rtl_path: str = "",
) -> tuple[QSPIContract | None, StimulusPlan | None]:
    """Classify the chip-top bus and, if QSPI-slave, build a host-flow plan.

    Returns ``(contract, plan)`` when a deterministic, contract-enforcing DV is
    possible, else ``(None, None)``. ``(None, None)`` alone does NOT tell you
    whether this is an honest non-QSPI design or a spec/RTL disagreement that
    must fail the run closed -- use :func:`classify_bus_verdict` for that.
    """
    verdict = classify_bus_verdict(
        project_root, top_rtl_source, connections, top_module, top_rtl_path
    )
    contract = verdict.contract if verdict.contract_enforcing else None
    if contract is None:
        return None, None
    plan = build_plan_from_run(project_root, contract)
    if plan is None:
        return contract, None
    return contract, plan


def write_deterministic_integration_tb(
    project_root: str,
    design_name: str,
    contract: QSPIContract,
    plan: StimulusPlan,
    output_path: str,
    include_conformance: bool = False,
) -> dict:
    """Render + persist the deterministic integration TB. Mirrors the LLM
    generator's return contract (``tb_path``, ``testbench_path``, ``test_count``).

    ``include_conformance`` also appends the compute-lane-independent QSPI-slave
    bus-protocol conformance test to the same module (CFG read-back through the
    dummy turnaround + 0x05 status + bad-opcode robustness), so one sim run gates
    both the golden host-flow output and the raw bus contract.
    """
    tb_src = render_integration_tb(
        contract, design_name, plan, include_conformance=include_conformance
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tb_src, encoding="utf-8")
    return {
        "tb_path": str(out),
        "testbench_path": str(out),
        "test_count": 2 if include_conformance else 1,
        "deterministic_bfm": True,
        "qspi_conformance": bool(include_conformance),
        "contract_fingerprint": contract.fingerprint(),
        "case_name": plan.case_name,
    }


def write_qspi_conformance_tb(
    project_root: str,
    design_name: str,
    contract: QSPIContract,
    output_path: str,
) -> dict:
    """Render + persist a standalone QSPI-slave bus-protocol conformance TB.

    Used when a golden host-flow plan cannot be derived (the compute-lane oracle
    is not modeled) but the chip-top IS a QSPI-slave: the deterministic master BFM
    still drives the pins and the conformance test enforces the standard command
    set + timing (COMPUTE-LANE INDEPENDENT). Mirrors the LLM generator's return
    contract so the caller treats it like any other integration TB.
    """
    tb_src = render_conformance_tb(contract, design_name)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tb_src, encoding="utf-8")
    return {
        "tb_path": str(out),
        "testbench_path": str(out),
        "test_count": 1,
        "deterministic_bfm": True,
        "qspi_conformance": True,
        "conformance_only": True,
        "contract_fingerprint": contract.fingerprint(),
        "case_name": "qspi_slave_protocol_conformance",
    }
