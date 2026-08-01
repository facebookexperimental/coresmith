# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""PR6 honest-skip proof: a reduced deployment never fakes a green.

A minimal custom deployment that implements only ``run_synth`` (no ``run_lvs``,
no ``gen_macro``, and none of the gate-sim / macro accessors) MUST make the
characterizer + macro + CLI paths report skip / not_run / empty -- never a false
pass. The base :class:`Deployment` defaults are the honest empties that make
this hold.
"""

from __future__ import annotations

import types

import pytest

from orchestrator.pdk import registry
from orchestrator.pdk.base import (
    CheckResult,
    Deployment,
    EdaTool,
    ToolResult,
)
from orchestrator.pdk.pdk_config import PDKConfig


class _NoopSynth(EdaTool):
    verb = "run_synth"

    def run(self, req):
        return ToolResult.from_checks(
            tool_ok=True, checks=[CheckResult("synth", "pass")],
            verb=self.verb, design=req.design)


class MinimalDeployment(Deployment):
    """The smallest legal deployment: one verb, no macros, no gate-sim models."""

    name = "minimal-test"

    def __init__(self):
        self._tools = {"run_synth": _NoopSynth(self)}
        self._pdk = PDKConfig(
            name="minimal", process_nm=130, std_cell_library="x",
            site_name="s", supply_voltage=1.8, default_corner="tt")

    @property
    def pdk(self):
        return self._pdk

    def tools(self):
        return self._tools


@pytest.fixture
def minimal(monkeypatch):
    dep = MinimalDeployment()
    monkeypatch.setattr(registry, "_cache", dep)
    yield dep
    registry.reset_deployment_cache()


# ---------------------------------------------------------------------------
# Deployment ABC defaults are honest empties
# ---------------------------------------------------------------------------
def test_defaults_are_empty(minimal):
    assert minimal.cell_model_files("anything") == ([], ())
    assert minimal.udp_shim_source(("x",)) == ""
    assert minimal.sim_defines() == []
    assert minimal.macro_search_paths("/some/pdk") == []
    assert minimal.gds_layer_map() is None
    assert not minimal.supports("run_lvs")
    assert not minimal.supports("gen_macro")


# ---------------------------------------------------------------------------
# gate-sim: no cell models -> nothing to simulate -> not_run, never a pass
# ---------------------------------------------------------------------------
def test_gate_sim_cell_models_absent(minimal):
    from orchestrator.harness import gate_sim as gs
    assert gs.cell_model_files("/no/pdk", "netlist") == ([], ())
    assert gs.udp_shim_source(("x",)) == ""
    # sim_defines falls back to the sky130 default constant (build never breaks)
    # but the *deployment* itself reports none.
    assert minimal.sim_defines() == []


def test_gate_sim_verdict_missing_is_not_run(minimal, tmp_path):
    """A blocking gate-sim checker with no verdict file is not_run (fail-closed),
    never a false pass -- the honest-skip contract at the checker level."""
    from orchestrator.pdk.base import ToolRequest
    from orchestrator.pdk.checkers import GateSimVerdictChecker
    res = GateSimVerdictChecker().check(
        ToolRequest(verb="run_gate_sim", design="d"), tmp_path)
    assert res.status == "not_run"
    assert res.failed  # blocking not_run fails the verb


# ---------------------------------------------------------------------------
# macro discovery: no macro search paths -> no macros discovered
# ---------------------------------------------------------------------------
def test_macro_discovery_empty(minimal, tmp_path):
    from orchestrator.langgraph.macro_registry import discover_macros
    discover_macros.cache_clear()
    assert discover_macros(str(tmp_path)) == {}
    discover_macros.cache_clear()


# ---------------------------------------------------------------------------
# CLI: missing capabilities exit 4 (skip), never 0 (green)
# ---------------------------------------------------------------------------
def _args(verb, tmp_path, **kw):
    base = dict(
        verb=verb, project_root=str(tmp_path), design="d", out_dir=None,
        timeout_s=None, json=False, rtl=None, script=None, netlist=None,
        sdc=None, gds=None, spice=None, width=None, depth=None, ports=None)
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_cli_run_lvs_skips(minimal, tmp_path):
    from orchestrator.harness import cli_tool
    assert cli_tool.cmd_tool_run(_args("run_lvs", tmp_path)) == cli_tool.EXIT_SKIP


def test_cli_gen_macro_skips(minimal, tmp_path):
    from orchestrator.harness import cli_tool
    assert cli_tool.cmd_tool_run(
        _args("gen_macro", tmp_path, width=8, depth=64, ports="1rw1r")
    ) == cli_tool.EXIT_SKIP
