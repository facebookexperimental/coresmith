"""The canonical SQLite project state (state_store.project_db)."""
from __future__ import annotations

import json

from orchestrator.state_store.project_db import ProjectDB, open_project

DIAGRAM = {
    "blocks": [
        {"name": "ctrl", "description": "control", "tier": 1, "subsystem": "top",
         "python_source": "", "rtl_target": "rtl/ctrl.v", "testbench": "tb/test_ctrl.py",
         "interfaces": [{"name": "s_axis_in", "kind": "axi_stream", "direction": "in", "width": 8}],
         "estimated_gates": 1200, "flip_flop_budget": 300, "area_budget_um2": 12000.5},
        {"name": "dsp", "description": "math", "tier": 2, "subsystem": "dp",
         "python_source": "golden.py:dsp", "rtl_target": "rtl/dsp.v", "testbench": "tb/test_dsp.py",
         "interfaces": [], "custom_key": {"nested": [1, 2]}},
    ],
    "connections": [
        {"from": "ctrl", "to": "dsp", "from_port": "m_axis", "to_port": "s_axis", "interface": "axi_stream",
         "data_width": 16, "bus_name": "ctrl_to_dsp", "handshake_protocol": "axi_stream"},
    ],
    "reasoning": "two blocks", "questions": [], "system_invariants": ["one clock"],
}
SPECS = [
    {"name": "ctrl", "tier": "1", "python_source": "", "rtl_target": "rtl/ctrl.v",
     "testbench": "tb/test_ctrl.py", "description": "control"},
    {"name": "dsp", "tier": "2", "python_source": "golden.py:dsp", "rtl_target": "rtl/dsp.v",
     "testbench": "tb/test_dsp.py", "description": "math"},
]
CONTRACTS = {
    "design_summary": "demo",
    "contracts": [
        {"edge_id": "ctrl_to_dsp", "producer_block": "ctrl", "producer_port": "m_axis",
         "consumer_block": "dsp", "consumer_port": "s_axis", "handshake_protocol": "axi_stream",
         "data_width_bits": 16, "fields": [{"name": "sample", "width": 16}]},
    ],
}


def test_schema_and_round_trip(tmp_path):
    db = open_project(tmp_path)
    assert db.exists() and (tmp_path / ".coresmith" / "project.sqlite").is_file()
    db.import_block_diagram(DIAGRAM)
    db.import_block_specs(SPECS)
    db.import_contracts(CONTRACTS)
    bd = db.block_diagram()
    assert [b["name"] for b in bd["blocks"]] == ["ctrl", "dsp"]
    assert bd["blocks"][0]["interfaces"][0]["name"] == "s_axis_in"
    assert bd["blocks"][1]["custom_key"] == {"nested": [1, 2]}
    assert bd["blocks"][0]["area_budget_um2"] == 12000.5
    assert bd["connections"][0]["bus_name"] == "ctrl_to_dsp"
    assert bd["system_invariants"] == ["one clock"] and bd["reasoning"] == "two blocks"
    assert db.block_specs() == SPECS
    assert db.block_names() == ["ctrl", "dsp"]
    assert db.contracts()["contracts"][0]["fields"][0]["width"] == 16
    assert db.contracts()["design_summary"] == "demo"
    assert db.contract_edges_for_block("dsp")[0]["edge_id"] == "ctrl_to_dsp"
    # views are derived and read-only
    view = tmp_path / ".coresmith" / "block_diagram.json"
    assert json.loads(view.read_text())["blocks"][1]["name"] == "dsp"
    assert not (view.stat().st_mode & 0o222)
    bview = tmp_path / ".coresmith" / "blocks"
    db.add_constraint("ctrl", "rule one", source="human")
    assert json.loads((bview / "ctrl" / "constraints.json").read_text())[0]["rule"] == "rule one"
    assert not ((bview / "ctrl" / "constraints.json").stat().st_mode & 0o222)


def test_contract_versions_drive_staleness(tmp_path):
    db = open_project(tmp_path)
    db.import_block_diagram(DIAGRAM)
    db.import_block_specs(SPECS)
    assert db.import_contracts(CONTRACTS) == 1
    assert db.stamp_block_spec("dsp") == 1
    assert db.stale_spec_blocks() == []
    # unchanged re-import keeps the version
    assert db.import_contracts(CONTRACTS) == 1
    assert db.stale_spec_blocks() == []
    amended = json.loads(json.dumps(CONTRACTS))
    amended["contracts"][0]["data_width_bits"] = 24
    assert db.import_contracts(amended) == 2
    assert db.stale_spec_blocks() == ["dsp"]      # ctrl was never stamped -> not a question
    db.stamp_block_spec("dsp")
    assert db.stale_spec_blocks() == []
    assert db.block_contract_version("dsp") == 2


def test_per_block_lifecycle_state(tmp_path):
    db = open_project(tmp_path)
    db.record_attempt("ctrl", {"attempt": 1, "error": "sim failed", "category": "LOGIC_ERROR"})
    db.record_attempt("ctrl", {"attempt": 2, "error": "lint", "category": "SYNTAX", "extra": "x"})
    assert [a["attempt"] for a in db.attempt_history("ctrl")] == [1, 2]
    assert db.attempt_history("ctrl")[1]["extra"] == "x"
    assert db.attempt_history("dsp") == []
    db.set_diagnosis("ctrl", {"category": "LOGIC_ERROR", "confidence": 0.9, "constraints": ["a"]}, attempt=2)
    assert db.diagnosis("ctrl")["constraints"] == ["a"]
    db.add_constraint("ctrl", "keep reset synchronous", source="human", attempt=1)
    db.add_constraint("ctrl", "no latches", source="debug_agent", attempt=2)
    db.add_constraint("ctrl", "chip fix", source="chip_dv_fix", attempt=0)
    assert [c["source"] for c in db.constraints("ctrl")] == ["human", "debug_agent", "chip_dv_fix"]
    assert db.prune_constraints("ctrl", ("chip_dv_revise", "chip_dv_fix", "human")) == 1
    assert [c["rule"] for c in db.constraints("ctrl")] == ["keep reset synchronous", "chip fix"]
    db.set_constraints("ctrl", [{"rule": "only", "source": "human", "attempt": 3, "note": "n"}])
    assert db.constraints("ctrl") == [{"rule": "only", "source": "human", "attempt": 3, "note": "n"}]
    db.set_result("ctrl", "best", {"sim_passed": True, "attempt": 2}, report_path="rtl/ctrl.v")
    assert db.result("ctrl", "best")["sim_passed"] is True
    assert db.update_result("ctrl", "best", ppa_retry_attempt=3)["ppa_retry_attempt"] == 3
    assert db.results("ctrl")[0]["report_path"] == "rtl/ctrl.v"
    db.clear_result("ctrl", "best")
    assert db.result("ctrl", "best") is None
    db.set_setting("engine_sha", "abc123")
    assert db.settings()["engine_sha"] == "abc123"


def test_legacy_import_once(tmp_path):
    cdir = tmp_path / ".coresmith"
    (cdir / "blocks" / "ctrl").mkdir(parents=True)
    (cdir / "block_diagram.json").write_text(json.dumps(DIAGRAM))
    (cdir / "block_specs.json").write_text(json.dumps(SPECS))
    (cdir / "interface_contracts.json").write_text(json.dumps(CONTRACTS))
    (cdir / "blocks" / "ctrl" / "attempt_history.json").write_text(json.dumps([{"attempt": 1, "error": "e", "category": "X"}]))
    (cdir / "blocks" / "ctrl" / "constraints.json").write_text(json.dumps([{"rule": "r", "source": "human", "attempt": 1}]))
    (cdir / "blocks" / "ctrl" / "best_result.json").write_text(json.dumps({"sim_passed": True}))
    db = open_project(tmp_path)
    assert db.block_names() == ["ctrl", "dsp"]
    assert db.contracts_version() == 1
    assert db.attempt_history("ctrl")[0]["category"] == "X"
    assert db.constraints("ctrl")[0]["rule"] == "r"
    # A1: a pre-gate ``best`` (no ``done`` flag) is only a DV pass -> ``dv_best``.
    assert db.result("ctrl", "best") is None
    assert db.result("ctrl", "dv_best") == {"sim_passed": True}
    # second open does not re-import (would duplicate attempts)
    db2 = open_project(tmp_path)
    assert len(db2.attempt_history("ctrl")) == 1


def test_scoreboard_tables_share_the_database(tmp_path):
    from orchestrator.state_store import Scoreboard
    db = open_project(tmp_path)
    sb = Scoreboard(tmp_path)
    assert sb.db_path == db.path
    sb.ensure_schema()
    sb.record_dv(block="ctrl", scope="rtl", source="gate", attempt=1, passed=True)
    assert sb.latest_dv(block="ctrl")[0]["passed"] == 1
    assert isinstance(ProjectDB(tmp_path).results(), list)


def test_rounds_keep_history_and_scope_reads(tmp_path):
    db = open_project(tmp_path)
    assert db.current_round("ctrl") == 1
    db.record_attempt("ctrl", {"attempt": 1, "error": "e1", "category": "A"})
    db.set_diagnosis("ctrl", {"category": "A", "confidence": 0.5}, attempt=1)
    assert db.begin_round("ctrl") == 2
    # a fresh round reads empty, history is retained
    assert db.attempt_history("ctrl") == []
    assert db.diagnosis("ctrl") is None
    assert [a["round"] for a in db.attempt_history("ctrl", all_rounds=True)] == [1]
    assert db.diagnosis("ctrl", all_rounds=True)["category"] == "A"
    db.record_attempt("ctrl", {"attempt": 1, "error": "e2", "category": "B"})
    assert [a["category"] for a in db.attempt_history("ctrl")] == ["B"]
    assert [d["round"] for d in db.diagnoses("ctrl")] == [1]
    allv = json.loads((tmp_path / ".coresmith" / "blocks" / "ctrl" / "attempt_history_all_rounds.json").read_text())
    assert [a["category"] for a in allv] == ["A", "B"]


def test_ppa_history_records_tns_and_migrates(tmp_path):
    from orchestrator.state_store import Scoreboard
    open_project(tmp_path)
    sb = Scoreboard(tmp_path)
    assert sb.record_ppa(block="ctrl", probe="synth", cells=10, ff=2, wns_ns=1.5, tns_ns=-0.25)
    row = sb.latest_ppa("ctrl")
    assert row["wns_ns"] == 1.5 and row["tns_ns"] == -0.25
    # a database created before the column existed is migrated on open
    import sqlite3
    old = tmp_path / "old" / ".coresmith"
    old.mkdir(parents=True)
    conn = sqlite3.connect(old / "project.sqlite")
    conn.executescript("CREATE TABLE ppa_history (id INTEGER PRIMARY KEY, ts REAL, block TEXT);")
    conn.close()
    db2 = open_project(tmp_path / "old")
    with db2._conn() as c:
        assert "tns_ns" in {r[1] for r in c.execute("PRAGMA table_info(ppa_history)")}


def test_pre_layout_sta_persists_report(tmp_path, monkeypatch):
    import subprocess

    from orchestrator.langgraph import ppa_check
    netlist = tmp_path / "n.v"
    netlist.write_text("module top(input clk); endmodule\n")
    sdc = tmp_path / "t.sdc"
    sdc.write_text("create_clock -name clk -period 20 [get_ports clk]\n")
    lib = tmp_path / "l.lib"
    lib.write_text("library(x){}\n")
    monkeypatch.setattr(ppa_check.shutil, "which", lambda n: "/usr/bin/sta" if n == "sta" else None)
    captured = {}

    def fake_run(cmd, **kw):
        captured["tcl"] = open(cmd[-1]).read()
        return subprocess.CompletedProcess(cmd, 0, stdout=(
            "Startpoint: a_reg (rising edge-triggered flip-flop clocked by clk)\n"
            "Endpoint: b_reg\n  slack (MET)  1.25\nwns max 1.25\ntns max 0.00\n"), stderr="")
    monkeypatch.setattr(ppa_check.subprocess, "run", fake_run)
    rpt = tmp_path / "syn" / "output" / "top" / "top_sta.rpt"
    out = ppa_check.run_pre_layout_sta(str(netlist), str(sdc), str(lib), "top", report_path=str(rpt))
    assert out["wns_ns"] == 1.25 and out["tns_ns"] == 0.0
    assert "report_checks -path_delay max" in captured["tcl"]
    assert rpt.is_file() and "Startpoint: a_reg" in rpt.read_text()
