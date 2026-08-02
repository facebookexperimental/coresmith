# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""PR5 golden-diff: the moved TCL generators reproduce the pre-move output.

The OpenROAD TCL generation (``generate_pnr_tcl`` / ``_generate_floorplan_tcl``
/ ``generate_rcx_tcl`` and the wrapper PnR generator) moved from
``backend_helpers`` / ``tapeout_helpers`` into the sky130 deployment's tool
classes, parameterized from ``PDKConfig.cells``/``.pnr``. The fixtures under
``fixtures/tcl_golden/`` were captured from the PRE-MOVE code with every
env-dependent path tokenized (``@@TECH_LEF@@`` ...). This test builds a fresh
``Sky130Deployment`` against a SYNTHETIC PDK tree (no real PDK), renders the
same inputs through the new tool-class methods, tokenizes the same way, and
asserts BYTE-IDENTICAL output -- the core regression proof for the move.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from orchestrator.pdk.registry import reset_deployment_cache

_FIX = Path(__file__).resolve().parent / "fixtures" / "tcl_golden"
_STD = "sky130_fd_sc_hd"

# The fixed, env-independent inputs the fixtures were captured with.
_NETLIST = "/abs/design/blk_netlist.v"
_SDC = "/abs/design/blk.sdc"
_DEF = "/abs/design/blk_routed.def"
_WRAP = "/abs/design/openframe_project_wrapper_netlist.v"


@pytest.fixture(autouse=True)
def _reset_cache():
    reset_deployment_cache()
    yield
    reset_deployment_cache()


def _synthetic_deployment(tmp_path, monkeypatch):
    """A fresh Sky130Deployment resolving all paths under a synthetic tree."""
    pdk_root = tmp_path / "pdk"
    (pdk_root / "sky130A").mkdir(parents=True)
    monkeypatch.setenv("PDK_ROOT", str(pdk_root))
    reset_deployment_cache()
    from orchestrator.pdk.deployments.sky130 import Sky130Deployment
    return Sky130Deployment()


def _tokenize(text: str, dep) -> str:
    """Replace env-dependent PDK paths with the fixture tokens."""
    for real, tok in (
        (str(dep.paths.tech_lef), "@@TECH_LEF@@"),
        (str(dep.paths.cell_lef), "@@CELL_LEF@@"),
        (str(dep.paths.liberty), "@@LIBERTY@@"),
        (str(dep.paths.cell_gds), "@@CELL_GDS@@"),
        (str(dep.paths.rcx_rules), "@@RCX_RULES@@"),
    ):
        text = text.replace(real, tok)
    return (text.replace(_NETLIST, "@@NETLIST@@").replace(_SDC, "@@SDC@@")
            .replace(_DEF, "@@DEF@@").replace(_WRAP, "@@WRAP_NETLIST@@"))


@pytest.mark.parametrize("gc,util,dens,tag", [
    (300, 45, 0.6, "small"),   # needs_explicit_die branch (gate_count < 500)
    (2000, 60, 0.5, "large"),  # utilization branch
])
def test_pnr_tcl_byte_identical(tmp_path, monkeypatch, gc, util, dens, tag):
    dep = _synthetic_deployment(tmp_path, monkeypatch)
    pnr = dep.tools()["run_pnr"]
    got = _tokenize(
        pnr.render_pnr_tcl("blk", "blk", _NETLIST, _SDC,
                           utilization=util, density=dens, gate_count=gc),
        dep)
    expected = (_FIX / f"pnr_{tag}.tcl").read_text()
    assert got == expected, f"pnr_{tag} TCL drifted from the golden fixture"


def test_rcx_tcl_byte_identical(tmp_path, monkeypatch):
    dep = _synthetic_deployment(tmp_path, monkeypatch)
    sta = dep.tools()["run_sta"]
    got = _tokenize(sta.render_rcx_tcl("blk", _DEF, _SDC), dep)
    expected = (_FIX / "rcx.tcl").read_text()
    assert got == expected, "rcx TCL drifted from the golden fixture"


def test_wrapper_pnr_tcl_byte_identical(tmp_path, monkeypatch):
    from orchestrator.langgraph.tapeout_helpers import (
        OPENFRAME_CORE_MARGIN_UM,
        OPENFRAME_DIE_HEIGHT_UM,
        OPENFRAME_DIE_WIDTH_UM,
    )
    dep = _synthetic_deployment(tmp_path, monkeypatch)
    pnr = dep.tools()["run_pnr"]
    out = "/run/wrap"  # stands in for the tapeout out_dir
    got = pnr.render_wrapper_pnr_tcl(
        _WRAP, f"{out}/wrapper.sdc", "openframe_project_wrapper",
        OPENFRAME_DIE_WIDTH_UM, OPENFRAME_DIE_HEIGHT_UM, OPENFRAME_CORE_MARGIN_UM)
    got = _tokenize(got, dep).replace(out, "@@OUT_DIR@@")
    expected = (_FIX / "wrapper_pnr.tcl").read_text()
    assert got == expected, "wrapper PnR TCL drifted from the golden fixture"


def test_pdk_config_carries_cells_and_pnr(tmp_path, monkeypatch):
    """The moved generators are DATA-DRIVEN: the sky130 config supplies the
    cells/pnr sections (a BYO-PDK deployment supplies its own)."""
    dep = _synthetic_deployment(tmp_path, monkeypatch)
    pdk = dep.pdk
    assert pdk.cells.tapcell == f"{_STD}__tapvpwrvgnd_1"
    assert pdk.cells.clkbuf_root == f"{_STD}__clkbuf_8"
    assert len(pdk.cells.fillers) == 7
    assert len(pdk.pnr.tracks) == 6
    # A float-fragile token must round-trip verbatim (1.70, not 1.7).
    met5 = [t for t in pdk.pnr.tracks if t["layer"] == "met5"][0]
    assert met5["x_pitch"] == "3.40"
    # cells/pnr survive a to_dict/from_dict round-trip.
    from orchestrator.pdk.pdk_config import PDKConfig
    rt = PDKConfig.from_dict(pdk.to_dict())
    assert rt.cells.tapcell == pdk.cells.tapcell
    assert rt.pnr.tracks == pdk.pnr.tracks
    assert rt.pnr.pdn == pdk.pnr.pdn
