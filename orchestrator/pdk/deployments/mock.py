# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""The ``mock`` deployment -- instant canned results, no subprocesses.

Formalizes the canned EDA shapes from ``orchestrator/testing/eda_stubs.py``
(LINT_CLEAN, SIM_PASS, SYNTH_OK, ...) as a real :class:`Deployment` so the whole
CLI / graph-routing surface can be exercised without Verilator/Yosys/OpenROAD
(fast, no fork-storm on a small box).

``eda_stubs.py`` itself is unchanged -- the ``pipeline_graph`` monkeypatch surface
and helper return-dict shapes are frozen for test-suite compatibility.

Verdict is deterministic from the design name (so the CLI exit-code matrix is
reproducible):

* name contains ``"infra"`` -> ``tool_ok=False``   (infra failure, CLI exit 3)
* name contains ``"fail"``  -> blocking checker fail (CLI exit 1)
* otherwise                 -> pass                 (CLI exit 0)

``run_lvs`` is intentionally **omitted** from ``tools()`` so an unsupported verb
demonstrates the honest capability-skip path (CLI exit 4).
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from orchestrator.pdk.base import (
    CheckResult,
    Deployment,
    EdaTool,
    ToolRequest,
    ToolResult,
)
from orchestrator.pdk.pdk_config import PDKConfig


def _verdict_name(req: ToolRequest) -> str:
    name = (req.design or "").lower()
    if not name:
        rtl = req.inputs.get("rtl")
        if rtl is not None:
            name = Path(rtl).stem.lower()
    return name


class _MockTool(EdaTool):
    """Returns a canned :class:`ToolResult` keyed off the design name."""

    def __init__(self, deployment: Deployment, verb: str,
                 metrics: dict | None = None) -> None:
        super().__init__(deployment)
        self._verb = verb
        self._metrics = metrics or {}

    @property
    def verb(self) -> str:  # type: ignore[override]
        return self._verb

    def run(self, req: ToolRequest) -> ToolResult:
        name = _verdict_name(req)
        if "infra" in name:
            return ToolResult.from_checks(
                tool_ok=False,
                checks=[CheckResult(self._verb, "not_run",
                                    details="mock: simulated infra failure")],
                verb=self._verb, design=req.design,
            )
        status = "fail" if "fail" in name else "pass"
        return ToolResult.from_checks(
            tool_ok=True,
            checks=[CheckResult(self._verb, status,
                                metrics=dict(self._metrics),
                                details=f"mock {self._verb}")],
            verb=self._verb, design=req.design,
            extra_metrics=dict(self._metrics),
        )

    def prompt_notes(self) -> str:
        return f"mock deployment: {self._verb} returns canned results instantly."


class MockDeployment(Deployment):
    name: ClassVar[str] = "mock"

    def __init__(self) -> None:
        # Metrics mirror eda_stubs.py canned shapes.
        self._tools: dict[str, EdaTool] = {
            "run_synth": _MockTool(self, "run_synth",
                                   {"cells": 1500, "gate_count": 1500, "ff_count": 64}),
            "run_lint": _MockTool(self, "run_lint", {"clean": True}),
            "run_sim": _MockTool(self, "run_sim", {"passed": True}),
            "run_pnr": _MockTool(self, "run_pnr",
                                 {"design_area_um2": 955.0, "wns_ns": 0.0}),
            "run_drc": _MockTool(self, "run_drc", {"violations": 0}),
            "run_sta": _MockTool(self, "run_sta", {"wns_ns": 0.0, "tns_ns": 0.0}),
            # run_lvs deliberately absent -> honest capability skip (exit 4).
        }
        self._pdk = PDKConfig(
            name="mock", process_nm=130, std_cell_library="mock_stdcell",
            site_name="mocksite", supply_voltage=1.8, default_corner="tt",
        )

    @property
    def pdk(self) -> PDKConfig:
        return self._pdk

    def tools(self) -> dict[str, EdaTool]:
        return self._tools


DEPLOYMENT = MockDeployment()
