# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""``coresmith contract add|set|rm|lock|unlock|show|list``, ``register
contracts --unlock``, and the contract auto-lock when ``interfaces`` completes."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import pytest

from orchestrator.harness import cli_contract as cc
from orchestrator.harness.tools.register import register
from orchestrator.state_store import stages as st
from orchestrator.state_store.project_db import open_project

_REPO = Path(__file__).resolve().parents[2]
_EID = "src__m_out__to__dst__s_in"

_DIAGRAM = {
    "blocks": [{"name": "src"}, {"name": "dst"}, {"name": "sink"}],
    "connections": [
        {"edge_id": _EID, "producer_block": "src", "producer_port": "m_out",
         "consumer_block": "dst", "consumer_port": "s_in", "handshake_protocol": "axi_stream",
         "data_width_bits": 12},
        {"edge_id": "dst__m_out__to__sink__s_in", "producer_block": "dst", "producer_port": "m_out",
         "consumer_block": "sink", "consumer_port": "s_in", "handshake_protocol": "axi_stream",
         "data_width_bits": 8},
    ],
}


@pytest.fixture
def proj(tmp_path, monkeypatch):
    monkeypatch.setenv("CORESMITH_PROJECT_ROOT", str(tmp_path))
    db = open_project(tmp_path)
    db.import_block_diagram(json.loads(json.dumps(_DIAGRAM)))
    return tmp_path, db


def _ns(root, **kw):
    base = {"project_root": str(root), "json": True, "unlock": False, "reason": ""}
    return argparse.Namespace(**{**base, **kw})


def _add_ns(root, producer="src.m_out", consumer="dst.s_in", **kw):
    base = {"producer": producer, "consumer": consumer, "protocol": "axi_stream", "width": None,
            "edge_id": None, "bus_param": [], "field": ["tdata:8", "tuser:4"], "sideband": ["tlast:1"],
            "timing": [], "policy": None, "semantic": None, "spec": None}
    return _ns(root, **{**base, **kw})


def _out(capsys):
    return json.loads(capsys.readouterr().out)


def test_add_writes_row_and_view(proj, capsys):
    root, db = proj
    assert cc.cmd_contract_add(_add_ns(root, semantic="pixels")) == 0
    out = _out(capsys)
    assert out["ok"] and out["version"] == 1
    row = db.contract_edge(_EID)
    assert row["version"] == 1 and not row["locked"]
    spec = row["spec"]
    assert spec["data_width_bits"] == 12 and spec["handshake_protocol"] == "axi_stream"
    assert [(f["name"], f["msb"], f["lsb"]) for f in spec["fields"]] == [("tdata", 7, 0), ("tuser", 11, 8)]
    assert spec["sideband_signals"] == [{"name": "tlast", "width": 1}] and spec["semantic_contract"] == "pixels"
    view = json.loads((root / ".coresmith" / "interface_contracts.json").read_text())
    assert [c["edge_id"] for c in view["contracts"]] == [_EID]
    # the uncovered second diagram connection is pre-existing, reported not fatal
    assert any(p["code"] == "CT_MISSING_EDGE" for p in out["problems"])
    assert cc.cmd_contract_list(_ns(root, block="dst", locked=False)) == 0
    assert _out(capsys)["contracts"] == [{"edge_id": _EID, "family": "axi_stream", "width": 12,
                                          "version": 1, "locked": False}]


def test_add_width_mismatch_refused(proj, capsys):
    root, db = proj
    rc = cc.cmd_contract_add(_add_ns(root, field=["tdata:16"]))
    out = _out(capsys)
    assert rc == 1 and not out["ok"]
    assert "CT_WIDTH_MISMATCH" in {p["code"] for p in out["problems"]}
    assert db.contract_rows() == [] and not (root / ".coresmith" / "interface_contracts.json").exists()


def test_set_lock_unlock(proj, capsys):
    root, db = proj
    assert cc.cmd_contract_add(_add_ns(root)) == 0
    capsys.readouterr()
    assert cc.cmd_contract_set(_ns(root, edge_id=_EID, key="timing.valid_to_ready_max_stall", value="8")) == 0
    out = _out(capsys)
    row = db.contract_edge(_EID)
    assert row["spec"]["timing"]["valid_to_ready_max_stall"] == 8 and row["version"] == 2 == out["version"]
    assert row["spec"]["timing"]["valid_hold_until_ready"] is True   # normaliser completed the object

    assert cc.cmd_contract_lock(_ns(root, verb="lock", block=None)) == 0
    assert _out(capsys)["edges"] == 1 and db.contract_edge(_EID)["locked"]
    rc = cc.cmd_contract_set(_ns(root, edge_id=_EID, key="timing.valid_to_ready_max_stall", value="4"))
    out = _out(capsys)
    assert rc == 1 and out["problems"][0]["code"] == "CT_LOCKED" and out["edge_ids"] == [_EID]
    assert db.contract_edge(_EID)["spec"]["timing"]["valid_to_ready_max_stall"] == 8
    assert cc.cmd_contract_rm(_ns(root, edge_id=_EID)) == 1
    assert _out(capsys)["problems"][0]["code"] == "CT_LOCKED" and db.contract_edge(_EID)
    # --unlock without --reason is a usage error
    assert cc.cmd_contract_set(_ns(root, edge_id=_EID, key="note", value="x", unlock=True)) == 2
    capsys.readouterr()
    assert cc.cmd_contract_set(_ns(root, edge_id=_EID, key="timing.valid_to_ready_max_stall", value="4",
                                   unlock=True, reason="x")) == 0
    out = _out(capsys)
    assert out["reason"] == "x" and out["locked"] is True   # the unlock is one-shot: the edge stays locked
    row = db.contract_edge(_EID)
    assert row["spec"]["timing"]["valid_to_ready_max_stall"] == 4 and row["locked"] and row["version"] == 3
    assert cc.cmd_contract_show(_ns(root, edge_id=_EID)) == 0
    assert _out(capsys)["version"] == 3
    assert cc.cmd_contract_lock(_ns(root, verb="unlock", block="src", reason="")) == 2
    assert cc.cmd_contract_rm(_ns(root, edge_id=_EID)) == 1   # still locked
    capsys.readouterr()
    assert cc.cmd_contract_rm(_ns(root, edge_id=_EID, unlock=True, reason="drop it")) == 0
    assert db.contract_rows() == []


def test_register_contracts_respects_locks(proj):
    root, db = proj
    edge = {"edge_id": _EID, "producer_block": "src", "producer_port": "m_out", "consumer_block": "dst",
            "consumer_port": "s_in", "handshake_protocol": "axi_stream", "data_width_bits": 12}
    other = {"edge_id": "dst__m_out__to__sink__s_in", "producer_block": "dst", "producer_port": "m_out",
             "consumer_block": "sink", "consumer_port": "s_in", "handshake_protocol": "axi_stream",
             "data_width_bits": 8}
    path = root / "ic.json"
    path.write_text(json.dumps({"contracts": [edge, other]}))
    res = register(db, root, "contracts", str(path))
    assert res["ok"], res["problems"]
    db.lock_contracts()
    path.write_text(json.dumps({"contracts": [{**edge, "semantic_contract": "changed"}, other]}))
    res = register(db, root, "contracts", str(path))
    assert not res["ok"] and [p["code"] for p in res["problems"]] == ["CT_LOCKED"]
    assert res["problems"][0]["where"] == _EID and db.artifact("contracts")["version"] == 1
    assert "semantic_contract" not in db.contract_edge(_EID)["spec"]
    assert register(db, root, "contracts", str(path), unlock=True)["problems"][0]["code"] == "CT_UNLOCK_NO_REASON"
    res = register(db, root, "contracts", str(path), unlock=True, reason="spec review")
    assert res["ok"], res["problems"]
    assert db.contract_edge(_EID)["spec"]["semantic_contract"] == "changed" and db.artifact("contracts")["version"] == 2


def _at_interfaces(root, db):
    from orchestrator.tests.build_fixtures import shared_stages_done
    shared_stages_done(db, "src")        # the done rows before interfaces hold on re-evaluation
    edge = {"edge_id": _EID, "producer_block": "src", "producer_port": "m_out", "consumer_block": "dst",
            "consumer_port": "s_in", "handshake_protocol": "axi_stream", "data_width_bits": 12}
    db.import_contracts({"contracts": [edge]})
    (root / "arch").mkdir()
    (root / "arch" / "ic.json").write_text("{}")
    (root / "arch" / "abi.md").write_text("abi " * 80)
    db.register_artifact("contracts", "arch/ic.json")
    db.register_artifact("abi", "arch/abi.md")
    (root / ".coresmith" / "vip_index.json").write_text("{}")
    db.pin_add({"name": "clk", "dir": "in", "kind": "clock"})
    db.add_integration_snapshot({"tier": "shell", "top": "chip_top", "elaborated": True, "boundary_ports": 1,
                                 "boundary": ["clk"]})
    for i, s in enumerate(st.STAGES[:st.STAGES.index("interfaces")]):
        db.stage_set(s, i, "done")
    db.stage_set("interfaces", st.STAGES.index("interfaces"), "active")
    assert st.current(db) == "interfaces" and st.entry(db, root, "interfaces") == []


def _stage_events(root):
    p = root / ".coresmith" / "pipeline_events.jsonl"
    rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()] if p.exists() else []
    return [r for r in rows if "stage_done" in json.dumps(r)]


def test_advance_from_interfaces_locks_contracts(proj, monkeypatch):
    root, db = proj
    monkeypatch.delenv("CORESMITH_CONTRACT_AUTOLOCK", raising=False)
    _at_interfaces(root, db)
    res = st.advance(db, root)
    assert res["advanced"] and res["done"] == "interfaces" and res["locked_edges"] == 1
    assert db.contract_edge(_EID)["locked"]
    assert _stage_events(root)[-1]["locked_edges"] == 1
    # the pins freeze with the edges
    assert res["locked_pins"] == 1 and db.pin("clk")["locked"]


def test_advance_autolock_disabled(proj, monkeypatch):
    root, db = proj
    monkeypatch.setenv("CORESMITH_CONTRACT_AUTOLOCK", "0")
    _at_interfaces(root, db)
    res = st.advance(db, root)
    assert res["advanced"] and "locked_edges" not in res and "locked_pins" not in res
    assert "locked_edges" not in _stage_events(root)[-1]
    assert not db.contract_edge(_EID)["locked"] and not db.pin("clk")["locked"]


def _run(root, *a):
    env = {"CORESMITH_PROJECT_ROOT": str(root), "PATH": "/usr/bin:/bin", "PYTHONPATH": str(_REPO)}
    return subprocess.run([sys.executable, str(_REPO / "bin" / "coresmith"), *a],
                          capture_output=True, text=True, env=env, timeout=120)


def test_cli_subprocess_add_lock_unlock(proj):
    root, db = proj
    p = _run(root, "contract", "add", "src.m_out", "dst.s_in", "--protocol", "axi_stream",
             "--field", "tdata:8", "--field", "tuser:4", "--timing", "valid_to_ready_max_stall=2", "--json")
    assert p.returncode == 0, p.stdout + p.stderr
    assert json.loads(p.stdout)["version"] == 1
    assert _run(root, "contract", "lock").returncode == 0
    p = _run(root, "contract", "set", _EID, "timing.valid_to_ready_max_stall", "3")
    assert p.returncode == 1 and "CT_LOCKED" in p.stdout
    p = _run(root, "contract", "set", _EID, "data_width_bits", "16")   # invalid: refused before the lock
    assert p.returncode == 1 and "CT_WIDTH_MISMATCH" in p.stdout
    p = _run(root, "contract", "set", _EID, "semantic_contract", "frames", "--unlock", "--reason", "ECO-7")
    assert p.returncode == 0, p.stdout + p.stderr
    assert db.contract_edge(_EID)["spec"]["semantic_contract"] == "frames"
    acts = db.actions()
    assert acts[-1]["rc"] == 0 and "ECO-7" in acts[-1]["summary"]
    p = _run(root, "contract", "list", "--locked")   # the --unlock was one-shot: still locked
    assert p.returncode == 0 and "LOCKED" in p.stdout and db.contract_edge(_EID)["locked"]
    assert _run(root, "contract", "unlock", "--reason", "ECO-8").returncode == 0   # the only way to open it
    p = _run(root, "contract", "list", "--locked")
    assert p.returncode == 0 and p.stdout.strip() == "no contract edges"


# -- fix 4: an --unlock is one-shot --------------------------------------------------
def test_unlock_flag_leaves_the_edge_locked(proj, capsys):
    root, db = proj
    assert cc.cmd_contract_add(_add_ns(root)) == 0
    assert cc.cmd_contract_lock(_ns(root, verb="lock", block=None)) == 0
    capsys.readouterr()
    # an unchanged write through --unlock: still locked (the evaluation's 4.13)
    assert cc.cmd_contract_set(_ns(root, edge_id=_EID, key="data_width_bits", value="12", unlock=True,
                                   reason="noop")) == 0
    capsys.readouterr()
    assert db.contract_edge(_EID)["locked"]
    # a changing add through --unlock: written, still locked
    assert cc.cmd_contract_add(_add_ns(root, semantic="v2", unlock=True, reason="ECO-1")) == 0
    out = _out(capsys)
    assert out["ok"] and out["locked"] is True
    row = db.contract_edge(_EID)
    assert row["locked"] and row["spec"]["semantic_contract"] == "v2"
    assert cc.cmd_contract_add(_add_ns(root, semantic="v3")) == 1   # the next write needs --unlock again
    assert _out(capsys)["problems"][0]["code"] == "CT_LOCKED"
    # register contracts --unlock re-locks too
    assert cc.cmd_contract_add(_add_ns(root, producer="dst.m_out", consumer="sink.s_in", field=["tdata:8"])) == 0
    doc = db.contracts()
    doc["contracts"] = [{**e, "semantic_contract": "v4"} if e["edge_id"] == _EID else e for e in doc["contracts"]]
    path = root / "ic.json"
    path.write_text(json.dumps(doc))
    res = register(db, root, "contracts", str(path), unlock=True, reason="bulk")
    assert res["ok"], res["problems"]
    assert db.contract_edge(_EID)["locked"] and db.contract_edge(_EID)["spec"]["semantic_contract"] == "v4"
    # contract unlock is the only way to leave it open
    assert cc.cmd_contract_lock(_ns(root, verb="unlock", block=None, reason="re-open")) == 0
    assert not db.contract_edge(_EID)["locked"]


# -- fix 6: contract add validates protocol, blocks, diagram ---------------------------
def test_contract_add_refuses_unknown_protocol(proj, capsys):
    root, db = proj
    assert cc.cmd_contract_add(_add_ns(root, protocol="wishbone", field=[], sideband=[], width=12)) == 1
    out = _out(capsys)
    assert [p["code"] for p in out["problems"] if p["severity"] == "error"][:1] == ["CT_BAD_PROTOCOL"]
    assert db.contract_rows() == [] and db.artifact("contracts") is None
    for fam in ("axi4", "apb", "req_resp", "valid_only", "static"):
        assert fam in cc._families()


def test_contract_add_refuses_unknown_block_and_warns_off_diagram(proj, capsys):
    root, db = proj
    rc = cc.cmd_contract_add(_add_ns(root, producer="fft64.foo", consumer="mcu.bar", field=[], sideband=[], width=8))
    out = _out(capsys)
    assert rc == 1 and "CT_UNKNOWN_BLOCK" in {p["code"] for p in out["problems"]}
    assert "fft64" in [p for p in out["problems"] if p["code"] == "CT_UNKNOWN_BLOCK"][0]["text"]
    assert db.contract_rows() == []
    # registered blocks with no diagram connection between them: a warning, not a refusal
    rc = cc.cmd_contract_add(_add_ns(root, producer="src.irq", consumer="sink.irq_in", protocol="static",
                                     field=["irq:1"], sideband=[]))
    out = _out(capsys)
    assert rc == 0 and out["ok"]
    off = [p for p in out["problems"] if p["code"] == "CT_OFF_DIAGRAM"]
    assert off and off[0]["severity"] == "warning"
    # a diagram-connected pair: no warning
    assert cc.cmd_contract_add(_add_ns(root)) == 0
    assert not any(p["code"] == "CT_OFF_DIAGRAM" for p in _out(capsys)["problems"])


def test_contract_add_without_registered_blocks_skips_block_check(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CORESMITH_PROJECT_ROOT", str(tmp_path))
    db = open_project(tmp_path)
    assert cc.cmd_contract_add(_add_ns(tmp_path, producer="a.m", consumer="b.s")) == 0
    assert _out(capsys)["ok"] and db.contract_edge("a__m__to__b__s")


# -- fix 7: line-item edges satisfy the stage machine -----------------------------------
def test_contract_add_makes_the_contracts_artifact_db_sourced(proj, capsys):
    root, db = proj
    assert db.artifact("contracts") is None
    assert any(b["code"] == "MISSING_ARTIFACT" and "contract" in b["text"] for b in st.entry(db, root, "interfaces"))
    assert cc.cmd_contract_add(_add_ns(root)) == 0
    art = db.artifact("contracts")
    assert art["path"] == "db:contracts"
    assert not any(b["code"] == "MISSING_ARTIFACT" and "contract" in b["text"] for b in st.entry(db, root, "interfaces"))
    # a registered document that is then edited line by line flips to db:contracts too
    assert cc.cmd_contract_add(_add_ns(root, producer="dst.m_out", consumer="sink.s_in", field=["tdata:8"])) == 0
    path = root / "ic.json"
    path.write_text(json.dumps(db.contracts()))
    assert register(db, root, "contracts", str(path))["ok"] and db.artifact("contracts")["path"] == "ic.json"
    capsys.readouterr()
    assert cc.cmd_contract_set(_ns(root, edge_id=_EID, key="semantic_contract", value="x")) == 0
    assert db.artifact("contracts")["path"] == "db:contracts"
