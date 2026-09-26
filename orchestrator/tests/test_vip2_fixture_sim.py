# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""A2/A3 end-to-end on the vip2 fixture with a real Verilator + cocotb run:
the generated VIP drives the responder, its assertions accept the
contract-conforming RTL and reject the late-responder twin."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from orchestrator.langgraph import vip_lib as V

_FX = Path(__file__).parent / "fixtures" / "vip2"
_HAVE_TOOLS = shutil.which("verilator") is not None and shutil.which("cocotb-config") is not None

pytestmark = pytest.mark.skipif(not _HAVE_TOOLS, reason="verilator/cocotb not installed")


def _run(tmp_path, rtl_name, monkeypatch, sva=True):
    import orchestrator.langgraph.pipeline_helpers as ph
    from orchestrator.state_store.project_db import open_project
    monkeypatch.setattr(ph, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("CORESMITH_VIP_SVA_BIND", "1" if sva else "0")
    monkeypatch.setenv("CORESMITH_LINE_COV_GATE", "0")
    monkeypatch.setenv("CORESMITH_COVERAGE", "0")
    edge = json.loads((_FX / "contract.json").read_text())
    db = open_project(tmp_path)
    db.import_contracts({"contracts": [edge]})
    V.write_all_vips(tmp_path, [edge], contract_version=db.contracts_version())
    rtl = tmp_path / "rtl" / "responder.v"
    rtl.parent.mkdir()
    shutil.copy(_FX / rtl_name, rtl)
    tb = tmp_path / "tb" / "test_responder.py"
    tb.parent.mkdir()
    shutil.copy(_FX / "test_responder.py", tb)
    monkeypatch.setattr(ph, "create_golden_model_wrapper", lambda *a, **k: None)
    return ph.run_simulation({"name": "responder"}, str(rtl), str(tb), project_root=str(tmp_path))


@pytest.mark.slow
def test_conforming_responder_passes_through_the_vip(tmp_path, monkeypatch):
    res = _run(tmp_path, "responder.v", monkeypatch)
    assert res["passed"], res.get("log", "")[-3000:]


@pytest.mark.slow
def test_late_responder_is_rejected_by_the_vip(tmp_path, monkeypatch):
    res = _run(tmp_path, "responder_late.v", monkeypatch, sva=False)
    assert not res["passed"]
    log = res.get("log", "")
    assert "VIPError" in log or "contract" in log, log[-3000:]
