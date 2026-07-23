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
  * ``classify_chip_bus(project_root, top_rtl_source, connections)`` ->
    ``QSPIContract | None``  (is this chip-top a QSPI-slave? DUT-blind)
  * ``build_plan_from_run(project_root, contract)`` -> ``StimulusPlan | None``
    (host-flow writes + expected OUT from the LLM golden reference)
  * ``plan_deterministic_dv(...)`` -> ``(contract, plan) | (None, None)``
  * ``render_integration_tb(contract, design_name, plan)`` -> TB source
  * ``write_deterministic_integration_tb(...)`` -> ``{tb_path, test_count, ...}``
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from .classifier import (
    arch_indicates_qspi_slave,
    classify_chip_bus,
    describe_unmodeled_roles,
    detect_bus_roles,
    detect_dut_mastered_buses,
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
    "QSPIContract",
    "QSPIMasterBFM",
    "QSPIRomContract",
    "QSPIRomResponderBFM",
    "render_rom_bfm_module",
    "render_rom_contract_literal",
    "StimulusPlan",
    "classify_chip_bus",
    "arch_indicates_qspi_slave",
    "describe_unmodeled_roles",
    "detect_bus_roles",
    "detect_dut_mastered_buses",
    "build_plan_from_run",
    "render_bfm_module",
    "render_contract_literal",
    "render_integration_tb",
    "render_conformance_tb",
    "render_conformance_test_fn",
    "plan_deterministic_dv",
    "write_deterministic_integration_tb",
    "write_qspi_conformance_tb",
    "deterministic_bfm_enabled",
    "conformance_enabled",
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


def plan_deterministic_dv(
    project_root: str,
    top_rtl_source: str,
    connections: Optional[list] = None,
) -> tuple[Optional[QSPIContract], Optional[StimulusPlan]]:
    """Classify the chip-top bus and, if QSPI-slave, build a host-flow plan.

    Returns ``(contract, plan)`` when a deterministic, contract-enforcing DV is
    possible, else ``(None, None)`` -- the caller falls back to the LLM BFM with
    a loud advisory.
    """
    contract = classify_chip_bus(project_root, top_rtl_source, connections)
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
