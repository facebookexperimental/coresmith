# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Database layer of the line-item CLI verbs: item bounds, verifiers, the
actions audit log, contract locking / single-edge edits, and fabric specs."""
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from orchestrator.state_store.ontology import CheckStatusConflict
from orchestrator.state_store.project_db import ContractLockedError, open_project

_REPO = Path(__file__).resolve().parents[2]

# The tables exactly as they were created before the bounds / value / locked
# columns existed (copied literally from the pre-change schema).
_OLD_ITEMS = """
CREATE TABLE IF NOT EXISTS items (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    artifact TEXT NOT NULL,
    section TEXT,
    text TEXT NOT NULL,
    priority TEXT,
    acceptance TEXT,
    model_check TEXT,
    status TEXT NOT NULL DEFAULT 'open',
    extra_json TEXT,
    artifact_sha TEXT,
    ts REAL NOT NULL
);
"""
_OLD_CHECKS = """
CREATE TABLE IF NOT EXISTS checks (
    id INTEGER PRIMARY KEY,
    item_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    evidence TEXT,
    sha TEXT,
    run_id TEXT,
    actor TEXT,
    ts REAL NOT NULL
);
"""
_OLD_CONTRACTS = """
CREATE TABLE IF NOT EXISTS contracts (
    edge_id TEXT PRIMARY KEY,
    ordinal INTEGER NOT NULL,
    producer_block TEXT,
    producer_port TEXT,
    consumer_block TEXT,
    consumer_port TEXT,
    handshake_protocol TEXT,
    data_width_bits INTEGER,
    spec_json TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 1
);
"""


def _cols(path, table):
    con = sqlite3.connect(str(path))
    try:
        return {r[1] for r in con.execute(f"PRAGMA table_info({table})")}
    finally:
        con.close()


def _frd(db, **item):
    db.ensure_db_artifact("frd")
    return db.upsert_item("frd", {"id": "PERF-001", "text": "cycles per frame", **item})


# (a) migration ----------------------------------------------------------------
def test_old_database_migrates(tmp_path):
    cdir = tmp_path / ".coresmith"
    cdir.mkdir()
    con = sqlite3.connect(str(cdir / "project.sqlite"))
    con.executescript(_OLD_ITEMS + _OLD_CHECKS + _OLD_CONTRACTS)
    con.execute("INSERT INTO items(id, kind, artifact, text, ts) VALUES ('PERF-009','PERF','frd','old',0)")
    con.execute("INSERT INTO contracts(edge_id, ordinal, spec_json) VALUES ('e0', 0, '{\"edge_id\": \"e0\"}')")
    con.commit()
    con.close()
    db = open_project(tmp_path)
    assert {"metric", "bound_min", "bound_max", "unit"} <= _cols(db.path, "items")
    assert "value" in _cols(db.path, "checks")
    assert "locked" in _cols(db.path, "contracts")
    assert {"verifiers", "actions", "fabric_specs"} <= {
        r[0] for r in sqlite3.connect(str(db.path)).execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert db.item("PERF-009")["bound_min"] is None
    assert db.edit_item("PERF-009", bound_max=5.0)["bound_max"] == 5.0
    db.add_check("PERF-009", "sim", None, value=4)
    assert db.checks("PERF-009")[-1]["value"] == 4.0
    assert db.contract_edge("e0")["locked"] is False
    db.ensure_schema()  # idempotent


# (b) bounds -> status ---------------------------------------------------------
def test_value_checks_derive_status_from_bounds(tmp_path):
    db = open_project(tmp_path)
    it = _frd(db, metric="cycles", bound_min=10, bound_max=20, unit="cyc")
    assert (it["metric"], it["bound_min"], it["bound_max"], it["unit"]) == ("cycles", 10.0, 20.0, "cyc")
    db.add_check("PERF-001", "sim", None, value=15)
    assert db.latest_check("PERF-001")["status"] == "pass" and db.item("PERF-001")["status"] == "verified"
    db.add_check("PERF-001", "sim", "", value=25)
    assert db.latest_check("PERF-001")["status"] == "fail" and db.item("PERF-001")["status"] == "failed"
    with pytest.raises(CheckStatusConflict):   # the bounds win: a contradicting status writes nothing
        db.add_check("PERF-001", "sim", "pass", value=99)
    n = len(db.checks("PERF-001"))
    db.add_check("PERF-001", "sim", "fail", value=99)  # an explicit status equal to the derived one is fine
    c = db.latest_check("PERF-001")
    assert (c["status"], c["value"]) == ("fail", 99.0) and len(db.checks("PERF-001")) == n + 1
    db.edit_item("PERF-001", bound_min=None)  # one open side: only the max applies
    db.add_check("PERF-001", "sim", None, value=-1e9)
    assert db.latest_check("PERF-001")["status"] == "pass"


def test_value_check_without_bounds_needs_status(tmp_path):
    db = open_project(tmp_path)
    _frd(db)
    with pytest.raises(ValueError, match="has no bounds"):
        db.add_check("PERF-001", "sim", None, value=1.0)
    assert db.add_check("PERF-001", "sim", "fail", value=1.0)
    assert db.add_check("PERF-001", "sim", "pass")  # the old path is unchanged
    assert db.latest_check("PERF-001")["value"] is None


def test_reregistering_items_keeps_cli_bounds(tmp_path):
    db = open_project(tmp_path)
    db.upsert_items("frd", [{"id": "PERF-001", "text": "a"}, {"id": "PERF-002", "text": "b"}])
    db.edit_item("PERF-001", metric="fps", bound_min=30, unit="fps")
    db.upsert_items("frd", [{"id": "PERF-001", "text": "a"}, {"id": "PERF-002", "text": "b"}])
    it = db.item("PERF-001")
    assert (it["metric"], it["bound_min"], it["unit"]) == ("fps", 30.0, "fps")
    db.upsert_items("frd", [{"id": "PERF-001", "text": "a", "bound_min": 60}, {"id": "PERF-002", "text": "b"}])
    assert db.item("PERF-001")["bound_min"] == 60.0


def test_single_item_upsert_edit_retire(tmp_path):
    db = open_project(tmp_path)
    art = db.ensure_db_artifact("frd")
    assert (art["path"], art["sha"], art["version"]) == ("db:frd", "db", 1)
    assert db.ensure_db_artifact("frd", registered_by="other") == art
    db.upsert_items("frd", [{"id": "PERF-001", "text": "a"}])
    db.upsert_item("frd", {"id": "PERF-002", "text": "b"})
    assert db.item("PERF-001")["status"] == "open"  # sibling not retired
    with pytest.raises(ValueError):
        db.upsert_item("frd", {"id": "not an id", "text": "x"})
    db.add_check("PERF-002", "sim", "pass")
    assert db.edit_item("PERF-002", acceptance="x")["status"] == "verified"
    assert db.edit_item("PERF-002", text="b2")["status"] == "open"
    db.set_item_status("PERF-002", "waived")
    assert db.edit_item("PERF-002", text="b3")["status"] == "waived"
    with pytest.raises(ValueError):
        db.edit_item("PERF-002", status="open")
    with pytest.raises(KeyError):
        db.edit_item("PERF-404", text="x")
    assert db.retire_item("PERF-002") and db.item("PERF-002")["status"] == "retired"
    assert not db.retire_item("PERF-404")


# (c) verifiers ----------------------------------------------------------------
def test_verifiers_add_list_link_remove(tmp_path):
    db = open_project(tmp_path)
    _frd(db)
    with pytest.raises(ValueError):
        db.add_verifier("PERF-001", "bogus")
    v1 = db.add_verifier("PERF-001", "cocotb", path="tb/test_x.py", entry="test_fps", args={"frames": 3}, block="core")
    v2 = db.add_verifier("PERF-001", "manual")
    assert db.add_verifier("PERF-001", "cocotb", path="tb/test_x.py", entry="test_fps", args={"frames": 5}) == v1
    assert db.verifier(v1)["args"] == {"frames": 5}
    assert [v["id"] for v in db.verifiers(item_id="PERF-001")] == [v1, v2]
    assert [v["id"] for v in db.verifiers(kind="manual")] == [v2]
    assert {lk["to_id"] for lk in db.links(from_id="PERF-001", rel="verified_by")} == {f"verifier:{v1}", f"verifier:{v2}"}
    assert db.remove_verifier(v1) and not db.remove_verifier(v1)
    assert db.verifier(v1) is None
    assert {lk["to_id"] for lk in db.links(from_id="PERF-001", rel="verified_by")} == {f"verifier:{v2}"}


# (d) actions ------------------------------------------------------------------
def test_actions_recorded_by_cli(tmp_path):
    cli = _REPO / "bin" / "coresmith"
    env = {"CORESMITH_PROJECT_ROOT": str(tmp_path), "PATH": "/usr/bin:/bin", "PYTHONPATH": str(_REPO)}

    def run(*a):
        return subprocess.run([sys.executable, str(cli), *a], capture_output=True, text=True, env=env, timeout=120)

    p = run("status", "--project-root", str(tmp_path))
    assert p.returncode == 0, p.stderr
    db = open_project(tmp_path)
    rows = db.actions()
    assert len(rows) == 1 and rows[0]["argv"][:1] == ["status"] and rows[0]["rc"] == 0
    assert rows[0]["summary"].startswith("stage:")
    p = run("actions", "--project-root", str(tmp_path), "--json")
    assert p.returncode == 0, p.stderr
    out = json.loads(p.stdout)
    assert [r["argv"][0] for r in out["actions"]] == ["status"]
    assert len(db.actions()) == 1  # `actions` itself is not recorded
    assert db.actions(since_id=rows[0]["id"]) == []


def test_record_action_api(tmp_path):
    db = open_project(tmp_path)
    db.begin_run("run-x")
    ids = [db.record_action(["item", "list", str(i)], 0, summary=f"s{i}") for i in range(3)]
    assert [r["id"] for r in db.actions(limit=2)] == ids[1:]
    assert db.actions()[0]["run_id"] == "run-x" and db.actions()[0]["actor"] == "cli"


# (e) contracts ----------------------------------------------------------------
_EDGE = {"producer_block": "a", "producer_port": "m", "consumer_block": "b", "consumer_port": "s",
         "handshake_protocol": "valid_ready", "data_width_bits": 32}
_EID = "a__m__to__b__s"


def test_contract_edge_upsert_lock_and_set_field(tmp_path):
    db = open_project(tmp_path)
    r = db.upsert_contract_edge(dict(_EDGE))
    assert r == {"edge_id": _EID, "version": 1, "changed": True, "locked": False}
    assert db.contracts_version() == 1
    assert db.upsert_contract_edge(dict(_EDGE))["changed"] is False
    assert db.contract_edge(_EID)["version"] == 1
    other = db.upsert_contract_edge({"edge_id": "x", "producer_block": "b", "consumer_block": "c"})
    assert other["version"] == 2 and [c["edge_id"] for c in db.contract_rows()] == [_EID, "x"]
    with pytest.raises(ValueError):
        db.upsert_contract_edge({"producer_block": "a"})

    r = db.set_contract_field(_EID, "timing.valid_to_ready_max_stall", "8")
    assert r["changed"] and r["version"] == 3
    assert db.contract_edge(_EID)["spec"]["timing"] == {"valid_to_ready_max_stall": 8}
    db.set_contract_field(_EID, "data_width_bits", "64")
    db.set_contract_field(_EID, "bus_params", '{"id_width": 4}')
    db.set_contract_field(_EID, "note", "hello world")
    spec = db.contract_edge(_EID)["spec"]
    assert spec["data_width_bits"] == 64 and spec["bus_params"] == {"id_width": 4} and spec["note"] == "hello world"
    assert db.contract_edges_for_block("a")[0]["data_width_bits"] == 64
    assert json.loads((tmp_path / ".coresmith" / "interface_contracts.json").read_text())["contracts"][0]["note"] == "hello world"
    with pytest.raises(KeyError):
        db.set_contract_field("nope", "a", 1)

    assert db.lock_contracts("a") == 1
    assert db.contract_edge(_EID)["locked"] and not db.contract_edge("x")["locked"]
    v = db.contract_edge(_EID)["version"]
    with pytest.raises(ContractLockedError) as ei:
        db.set_contract_field(_EID, "data_width_bits", 16)
    assert ei.value.edge_ids == [_EID] and db.contract_edge(_EID)["version"] == v
    assert db.upsert_contract_edge(db.contract_edge(_EID)["spec"])["locked"] is True  # unchanged: allowed
    r = db.set_contract_field(_EID, "data_width_bits", 16, unlock=True)
    assert r["changed"] and r["locked"] is True and db.contract_edge(_EID)["locked"]   # one-shot unlock
    assert db.lock_contracts() == 2 and db.lock_contracts(locked=False) == 2


def test_import_contracts_respects_locks(tmp_path):
    db = open_project(tmp_path)
    doc = {"contracts": [{**_EDGE, "edge_id": _EID}, {"edge_id": "x", "producer_block": "b", "consumer_block": "c"}]}
    db.import_contracts(doc)
    db.lock_contracts("a")
    assert db.import_contracts(doc) == 1  # unchanged: fine, lock kept
    assert db.contract_edge(_EID)["locked"]
    before = db.contract_rows()
    changed = {"contracts": [{**_EDGE, "edge_id": _EID, "data_width_bits": 8}, doc["contracts"][1]]}
    with pytest.raises(ContractLockedError) as ei:
        db.import_contracts(changed)
    assert ei.value.edge_ids == [_EID]
    with pytest.raises(ContractLockedError):
        db.import_contracts({"contracts": [doc["contracts"][1]]})  # dropping it is a change too
    assert db.contract_rows() == before
    db.import_contracts({"contracts": [doc["contracts"][1], {**doc["contracts"][0]}]})  # reorder only
    assert db.contract_edge(_EID)["locked"]
    db.import_contracts(changed, unlock=True)
    assert db.contract_edge(_EID)["spec"]["data_width_bits"] == 8
    assert db.contract_edge(_EID)["locked"] and not db.contract_edge("x")["locked"]   # re-locked after the import


# (f) fabric specs -------------------------------------------------------------
_FABRIC = {"masters": [{"name": "core0"}, {"name": "core1"}],
           "slaves": [{"name": "ram", "protocol": "axi4", "base": 0, "size": 4096},
                      {"name": "uart", "protocol": "apb", "base": 4096, "size": 4096}],
           "data_width": 64}
_BD = {"blocks": [{"name": "core", "tier": 1},
                  {"name": "noc", "tier": 0, "kind": "primitive", "primitive": "cs_fabric",
                   "fabric": {"masters": [{"name": "core0"}], "slaves": [{"name": "ram", "base": 0, "size": 4096}]}}],
       "connections": []}


def test_fabric_spec_set_view_and_merge(tmp_path):
    from orchestrator.architecture.specialists import fabric_resolution
    db = open_project(tmp_path)
    db.import_block_diagram(_BD)
    assert db.fabric_spec() is None
    r = db.set_fabric_spec("soc_fabric", dict(_FABRIC))
    assert r == {"ok": True, "name": "soc_fabric", "version": 1, "changed": True}
    view = tmp_path / ".coresmith" / "fabric_spec.json"
    assert json.loads(view.read_text())["name"] == "soc_fabric"
    from orchestrator.state_store.ontology import file_sha
    assert db.get_setting("fabric_spec_sha") == file_sha(view)
    noc = next(b for b in db.block_diagram()["blocks"] if b["name"] == "noc")
    assert noc["fabric"]["name"] == "soc_fabric" and len(noc["fabric"]["slaves"]) == 2
    assert next(b for b in db.block_specs() if b["name"] == "noc")["fabric"]["data_width"] == 64
    res = fabric_resolution.resolve(db.block_diagram())
    assert res["fabrics"] == ["noc"] and not res["errors"]
    rb = next(b for b in res["diagram"]["blocks"] if b["name"] == "noc")
    assert rb["fabric"]["data_width"] == 64 and rb["rtl_target"].endswith("cs_fabric_soc_fabric.v")

    assert db.set_fabric_spec("soc_fabric", dict(_FABRIC))["changed"] is False
    r = db.set_fabric_spec("soc_fabric", {**_FABRIC, "max_outstanding": 8})
    assert r["version"] == 2 and r["changed"]
    assert db.fabric_spec("soc_fabric")["spec"]["max_outstanding"] == 8
    assert [f["name"] for f in db.fabric_specs()] == ["soc_fabric"]


@pytest.mark.parametrize("bad", [
    {**_FABRIC, "slaves": [{"name": "ram", "base": 0, "size": 8192}, {"name": "rom", "base": 4096, "size": 4096}]},
    {**_FABRIC, "masters": [{"name": "core0", "protocol": "apb"}]},
])
def test_invalid_fabric_spec_refused(tmp_path, bad):
    db = open_project(tmp_path)
    r = db.set_fabric_spec("soc_fabric", bad)
    assert r["ok"] is False and r["problems"]
    assert db.fabric_specs() == [] and not (tmp_path / ".coresmith" / "fabric_spec.json").exists()


def test_fabric_merge_by_name_and_delete(tmp_path):
    db = open_project(tmp_path)
    db.import_block_diagram({"blocks": [{"name": "xbar", "kind": "primitive", "fabric": {}},
                                        {"name": "noc", "kind": "primitive", "fabric": {}}], "connections": []})
    db.set_fabric_spec("noc", dict(_FABRIC))
    blocks = {b["name"]: b for b in db.block_diagram()["blocks"]}
    assert blocks["noc"]["fabric"]["name"] == "noc" and blocks["xbar"]["fabric"] == {}
    assert (tmp_path / ".coresmith" / "fabric_spec.json").is_file()
    assert db.delete_fabric_spec("noc") and not db.delete_fabric_spec("noc")
    assert not (tmp_path / ".coresmith" / "fabric_spec.json").exists()
    assert {b["name"]: b for b in db.block_diagram()["blocks"]}["noc"]["fabric"] == {}


# (g) CLI verbs: ids, actions, links ------------------------------------------------
def _verb(root, *argv, json_out=True):
    """Run one harness verb in-process; (rc, payload-or-text)."""
    import argparse
    import contextlib
    import io

    from orchestrator.harness import cli as hcli
    ap = argparse.ArgumentParser(prog="coresmith")
    sub = ap.add_subparsers(dest="cmd")
    hcli.register_subcommands(sub)
    args = ap.parse_args([*argv, "--project-root", str(root), *(["--json"] if json_out else [])])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), pytest.raises(SystemExit) as exc:
        args.func(args)
    out = buf.getvalue()
    return exc.value.code, (json.loads(out) if json_out and out.strip().startswith("{") else out)


@pytest.fixture
def _root(tmp_path, monkeypatch):
    monkeypatch.setenv("CORESMITH_PROJECT_ROOT", str(tmp_path))
    return tmp_path


def test_question_and_ruling_ids_accept_the_printed_prefix(_root):
    db = open_project(_root)
    rc, out = _verb(_root, "question", "add", "which bus?", "--item", "PERF-003")
    assert rc == 0 and out["id"] == 1
    rc, out = _verb(_root, "question", "show", "Q1")
    assert rc == 0 and out["question"]["text"] == "which bus?"
    rc, out = _verb(_root, "question", "answer", "Q1", "--ruling", "APB port, back to back")
    assert rc == 0 and out["answered"] and out["ruling_id"]
    rid = out["ruling_id"]
    assert db.question(1)["ruling_id"] == rid and db.question(1)["status"] == "answered"
    # answering a closed / unknown / malformed question writes no (orphan) ruling
    n = len(db.rulings(active_only=False))
    assert _verb(_root, "question", "answer", "Q1", "--ruling", "again")[0] == 2
    assert _verb(_root, "question", "answer", "q99", "--ruling", "nope")[0] == 2
    assert _verb(_root, "question", "answer", "QX", "--ruling", "nope")[0] == 2
    assert len(db.rulings(active_only=False)) == n
    rc, out = _verb(_root, "ruling", "revoke", f"R{rid}", "--reason", "superseded")
    assert rc == 0 and out["revoked"] and out["id"] == rid
    assert _verb(_root, "ruling", "revoke", str(rid))[0] == 2   # already revoked
    assert _verb(_root, "ruling", "revoke", "Rx")[0] == 2


def test_interrupts_resolve_accepts_prefixed_and_short_ids(_root):
    from orchestrator.harness.cli import _interrupt_id
    db = open_project(_root)
    db.begin_run("r1")
    iid, _ = db.park_interrupt({"type": "uarch_spec_review", "block_name": "fft64"}, graph="pipeline", node="n")
    assert _interrupt_id(db, iid) == iid
    assert _interrupt_id(db, iid[:10]) == iid                  # unique prefix
    assert _interrupt_id(db, iid.split("-", 1)[1][:8]) == iid  # without the int- prefix
    assert _interrupt_id(db, "Q12") == "12"                    # Q/R + digits: stripped (no match -> as given)
    rc, out = _verb(_root, "interrupts", "--resolve", iid[:10], "--action", "approve")
    assert rc == 0 and out["resolved"] == iid


def test_actions_all_and_omitted_count(_root):
    db = open_project(_root)
    for i in range(5):
        db.record_action(["item", "list", str(i)], 0)
    rc, out = _verb(_root, "actions", "--limit", "2")
    assert rc == 0 and len(out["actions"]) == 2 and out["omitted"] == 3 and out["total"] == 5
    rc, out = _verb(_root, "actions", "--all")
    assert rc == 0 and len(out["actions"]) == 5 and out["omitted"] == 0 and out["limit"] is None
    rc, out = _verb(_root, "actions", "--since", "1", "--limit", "2")
    assert [r["id"] for r in out["actions"]] == [2, 3] and out["omitted"] == 2
    rc, text = _verb(_root, "actions", "--limit", "2", json_out=False)
    assert "3 earlier row(s) omitted" in text
    rc, text = _verb(_root, "actions", "--script", json_out=False)   # unchanged
    assert text.startswith("#!/bin/sh") and text.count("coresmith item list") == 5
    assert len(db.actions(limit=None)) == 5


def test_link_checks_blocks_and_unlink(_root):
    db = open_project(_root)
    assert _verb(_root, "link", "TIME-001", "block:nosuch", "owned_by")[0] == 0   # no blocks registered: no check
    db.import_block_diagram({"blocks": [{"name": "fft64"}, {"name": "uart"}], "connections": []})
    rc, out = _verb(_root, "link", "TIME-001", "block:nosuch2", "owned_by")
    assert rc == 2 and out["problems"][0]["code"] == "LINK_UNKNOWN_BLOCK"
    assert not db.links(to_id="block:nosuch2")
    assert _verb(_root, "link", "TIME-001", "block:fft64", "owned_by")[0] == 0
    rc, out = _verb(_root, "unlink", "TIME-001", "block:nosuch", "owned_by")
    assert rc == 0 and out["removed"] == 1
    assert [lk["to_id"] for lk in db.links(from_id="TIME-001")] == ["block:fft64"]
    assert _verb(_root, "unlink", "TIME-001", "block:nosuch", "owned_by")[0] == 2   # nothing to remove
    assert _verb(_root, "unlink", "TIME-001", "block:fft64", "bogus_rel")[0] == 2
    assert _verb(_root, "unlink", "TIME-001", "block:*", "owned_by")[1]["removed"] == 1
