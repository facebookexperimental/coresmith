# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""``coresmith fabric ...`` verbs over the fabric_specs rows (cli_fabric.py) and
the parameter system (orchestrator.fabric.params)."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import pytest

from orchestrator.fabric import params as fp
from orchestrator.harness import cli_fabric as cf
from orchestrator.state_store.project_db import open_project

_REPO = Path(__file__).resolve().parents[2]


def _ns(tmp_path, **kw):
    base = {"project_root": str(tmp_path), "json": True, "name": None, "param": []}
    return argparse.Namespace(**{**base, **kw})


def _out(capsys) -> dict:
    return json.loads(capsys.readouterr().out)


@pytest.fixture
def proj(tmp_path, monkeypatch):
    monkeypatch.setenv("CORESMITH_PROJECT_ROOT", str(tmp_path))
    return tmp_path, open_project(tmp_path)


def _build(tmp_path, db, capsys):
    """init (master core0 + slave ram) + master core1 + slave uart."""
    assert cf.cmd_fabric_init(_ns(tmp_path, name="soc", data_width="32", addr_width="32", master=["core0"],
                                  slave=["ram:axi4:0x0:0x2000"], replace=False), db) == 0
    assert _out(capsys)["version"] == 1
    assert cf.cmd_fabric_master_add(_ns(tmp_path, port="core1", protocol="axi4", param=["id_width=4"]), db) == 0
    assert _out(capsys)["version"] == 2
    assert cf.cmd_fabric_slave_add(_ns(tmp_path, port="uart", protocol="apb", base="0x10000000",
                                       size="0x1000"), db) == 0
    assert _out(capsys)["version"] == 3


def test_build_versions_and_view(proj, capsys):
    tmp_path, db = proj
    _build(tmp_path, db, capsys)
    row = db.fabric_spec("soc")
    assert row["version"] == 3
    assert [m["name"] for m in row["spec"]["masters"]] == ["core0", "core1"]
    assert {s["name"]: (s["protocol"], s["base"], s["size"]) for s in row["spec"]["slaves"]} == {
        "ram": ("axi4", 0, 0x2000), "uart": ("apb", 0x10000000, 0x1000)}
    view = tmp_path / ".coresmith" / "fabric_spec.json"
    assert json.loads(view.read_text()) == row["spec"]
    # an unchanged write does not bump the version
    assert cf.cmd_fabric_set(_ns(tmp_path, param="data_width", value="32"), db) == 0
    assert _out(capsys)["changed"] is False and db.fabric_spec()["version"] == 3


def test_init_requires_ports_and_refuses_existing(proj, capsys):
    tmp_path, db = proj
    rc = cf.cmd_fabric_init(_ns(tmp_path, name="soc", data_width="32", addr_width="32", master=[], slave=[],
                                replace=False), db)
    assert rc == 2 and _out(capsys)["code"] == "FABRIC_USAGE" and db.fabric_specs() == []
    _build(tmp_path, db, capsys)
    rc = cf.cmd_fabric_init(_ns(tmp_path, name="soc", data_width="32", addr_width="32", master=["a"],
                                slave=["r:axi4:0x0:0x1000"], replace=False), db)
    assert rc == 1 and _out(capsys)["code"] == "FABRIC_EXISTS" and db.fabric_spec()["version"] == 3


def test_overlap_refused_row_unchanged(proj, capsys):
    tmp_path, db = proj
    _build(tmp_path, db, capsys)
    before = db.fabric_spec()
    rc = cf.cmd_fabric_slave_add(_ns(tmp_path, port="rom", protocol="axi4", base="0x1000", size="0x1000"), db)
    res = _out(capsys)
    assert rc == 1 and res["code"] == "FABRIC_INVALID" and any("overlap" in p for p in res["problems"])
    assert db.fabric_spec() == before


def test_set_fabric_and_port_params(proj, capsys):
    tmp_path, db = proj
    _build(tmp_path, db, capsys)
    assert cf.cmd_fabric_set(_ns(tmp_path, param="data_width", value="64"), db) == 0
    assert db.fabric_spec()["spec"]["data_width"] == 64 and db.fabric_spec()["version"] == 4
    assert cf.cmd_fabric_set(_ns(tmp_path, param="slave.uart.base", value="0x10001000"), db) == 0
    uart = next(s for s in db.fabric_spec()["spec"]["slaves"] if s["name"] == "uart")
    assert uart["base"] == 0x10001000 and db.fabric_spec()["version"] == 5
    # a port-level data width that no longer matches the fabric's is refused by validate()
    capsys.readouterr()
    rc = cf.cmd_fabric_set(_ns(tmp_path, param="slave.ram.data_width", value="32"), db)
    assert rc == 1 and _out(capsys)["code"] == "FABRIC_INVALID" and db.fabric_spec()["version"] == 5
    # illegal value (choices) and malformed key
    assert cf.cmd_fabric_set(_ns(tmp_path, param="latency_mode", value="bogus"), db) == 1
    assert _out(capsys)["code"] == "FABRIC_INVALID"
    assert cf.cmd_fabric_set(_ns(tmp_path, param="slave.uart", value="1"), db) == 2
    assert _out(capsys)["code"] == "FABRIC_USAGE"
    assert cf.cmd_fabric_master_rm(_ns(tmp_path, port="core1"), db) == 0
    assert [m["name"] for m in db.fabric_spec()["spec"]["masters"]] == ["core0"]
    capsys.readouterr()
    assert cf.cmd_fabric_master_rm(_ns(tmp_path, port="core0"), db) == 1        # last master
    assert _out(capsys)["code"] == "FABRIC_INVALID"
    assert cf.cmd_fabric_slave_rm(_ns(tmp_path, port="nope"), db) == 1
    assert _out(capsys)["code"] == "FABRIC_PORT_NOT_FOUND"


def test_unknown_param_lists_allowed(proj, capsys):
    tmp_path, db = proj
    _build(tmp_path, db, capsys)
    rc = cf.cmd_fabric_set(_ns(tmp_path, param="slave.uart.id_width", value="4"), db)
    res = _out(capsys)
    assert rc == 1 and res["code"] == "FABRIC_INVALID"
    assert res["problems"] == ["unknown parameter id_width for slave/apb; allowed: base, size, max_outstanding"]
    assert db.fabric_spec()["version"] == 3
    with pytest.raises(ValueError, match="allowed: data_width, addr_width"):
        fp.coerce("fabric", None, "bogus", "1")


def test_params_schema_and_listing(capsys):
    assert fp.coerce("slave", "apb", "base", "0x10000000") == 0x10000000
    assert fp.coerce("fabric", None, "unique_ids", "1") is True
    assert fp.coerce("fabric", None, "err_slave", "false") is False
    with pytest.raises(ValueError, match="power of two"):
        fp.coerce("slave", "axi4", "size", "0x3000")
    assert fp.PARAM_SCHEMA["fabric"]["latency_mode"]["default"] == "cut_all_ports"
    assert fp.PARAM_SCHEMA["master"]["axi4"]["id_width"]["default"] == 4
    assert cf.cmd_fabric_params(argparse.Namespace(role="slave", protocol="apb", json=True)) == 0
    assert [r["param"] for r in _out(capsys)["params"]] == ["base", "size", "max_outstanding"]
    assert cf.cmd_fabric_params(argparse.Namespace(role="slave", protocol="apb", json=False)) == 0
    text = capsys.readouterr().out
    assert "base" in text and "size" in text and "data_width" not in text


def _stub_arch(monkeypatch, derived):
    from orchestrator.systemc_model import arch_model as am
    monkeypatch.setattr(am, "read_stats", lambda md: {"links": []})
    monkeypatch.setattr(am, "load_spec", lambda root: object())
    monkeypatch.setattr(am, "derive_fabric", lambda spec, stats, name=None, headroom=2.0: dict(derived))


_DERIVED = {"name": "soc", "masters": [{"name": "core0", "protocol": "axi4", "id_width": 4, "max_outstanding": 4}],
            "slaves": [{"name": "ram", "protocol": "axi4", "base": "0x0", "size": "0x2000"}],
            "data_width": 64, "addr_width": 32, "max_outstanding": 4,
            "derived_from": {"busiest_link_bytes_per_cycle": 6.0, "headroom": 2.0, "links": []}}


def test_derive_dry_run_then_write(proj, capsys, monkeypatch):
    tmp_path, db = proj
    _build(tmp_path, db, capsys)
    _stub_arch(monkeypatch, _DERIVED)
    before = db.fabric_spec()
    assert cf.cmd_fabric_derive(_ns(tmp_path, json=False, headroom=2.0, dry_run=True), db) == 0
    text = capsys.readouterr().out
    assert "--- fabric soc v3" in text and "+++ derived" in text and '-  "data_width": 32' in text
    assert db.fabric_spec() == before
    assert cf.cmd_fabric_derive(_ns(tmp_path, headroom=2.0, dry_run=False), db) == 0
    res = _out(capsys)
    assert res["ok"] and res["version"] == 4
    row = db.fabric_spec("soc")
    assert row["spec"]["data_width"] == 64 and [s["name"] for s in row["spec"]["slaves"]] == ["ram"]
    assert json.loads((tmp_path / ".coresmith" / "fabric_spec.json").read_text())["data_width"] == 64


def test_derive_dry_run_without_row(proj, capsys, monkeypatch):
    tmp_path, db = proj
    _stub_arch(monkeypatch, _DERIVED)
    assert cf.cmd_fabric_derive(_ns(tmp_path, json=False, headroom=2.0, dry_run=True), db) == 0
    assert "(none)" in capsys.readouterr().out and db.fabric_specs() == []


def test_block_diagram_renders_row(proj, capsys):
    from orchestrator.architecture.specialists import fabric_resolution
    tmp_path, db = proj
    stale = {"name": "old", "masters": [{"name": "x"}], "slaves": [{"name": "y", "base": 0, "size": 4096}]}
    db.import_block_diagram({"blocks": [{"name": "core0", "tier": 1},
                                        {"name": "soc_fabric", "tier": 0, "kind": "primitive",
                                         "primitive": "cs_fabric", "fabric": stale}],
                             "connections": []})
    assert cf.cmd_fabric_init(_ns(tmp_path, name="soc_fabric", data_width="32", addr_width="32",
                                  master=["core0"], slave=["ram:axi4:0x0:0x2000"], replace=False), db) == 0
    assert cf.cmd_fabric_slave_add(_ns(tmp_path, port="uart", protocol="apb", base="0x10000000",
                                       size="0x1000"), db) == 0
    capsys.readouterr()
    blk = next(b for b in db.block_diagram()["blocks"] if b["name"] == "soc_fabric")
    assert blk["fabric"]["name"] == "soc_fabric"
    assert [s["name"] for s in blk["fabric"]["slaves"]] == ["ram", "uart"]
    res = fabric_resolution.resolve(db.block_diagram())
    assert res["fabrics"] == ["soc_fabric"] and not res["errors"]
    rb = next(b for b in res["diagram"]["blocks"] if b["name"] == "soc_fabric")
    assert [s["name"] for s in rb["fabric"]["slaves"]] == ["ram", "uart"]
    assert cf.cmd_fabric_show(_ns(tmp_path), db) == 0
    shown = _out(capsys)["fabrics"][0]
    assert shown["blocks"] == ["soc_fabric"] and shown["warnings"] == [] and shown["version"] == 2


def test_arch_graph_fabric_node_sees_row(proj, capsys, monkeypatch):
    """The Fabric Resolution node resolves the LLM's document with the DB row merged in."""
    import asyncio

    from orchestrator.langgraph import architecture_graph as ag
    tmp_path, db = proj
    _build(tmp_path, db, capsys)
    monkeypatch.setattr(ag, "_event", lambda *a, **k: None)
    stale = {"name": "soc", "masters": [{"name": "x"}], "slaves": [{"name": "y", "base": 0, "size": 4096}]}
    doc = {"blocks": [{"name": "soc", "kind": "primitive", "primitive": "cs_fabric", "fabric": stale}],
           "connections": []}
    out = asyncio.run(ag.fabric_resolution_node({"project_root": str(tmp_path), "block_diagram": doc}))
    fab = out["block_diagram"]["blocks"][0]["fabric"]
    assert out["fabric_resolution"]["fabrics"] == ["soc"]
    assert [s["name"] for s in fab["slaves"]] == ["ram", "uart"]


def test_show_warns_without_primitive_block_and_cli(proj, capsys):
    tmp_path, db = proj
    _build(tmp_path, db, capsys)
    cli = _REPO / "bin" / "coresmith"
    env = {"CORESMITH_PROJECT_ROOT": str(tmp_path), "PATH": "/usr/bin:/bin", "PYTHONPATH": str(_REPO)}
    p = subprocess.run([sys.executable, str(cli), "fabric", "show"], capture_output=True, text=True,
                       env=env, timeout=120)
    assert p.returncode == 0, p.stderr
    assert "fabric 'soc' v3" in p.stdout and "uart" in p.stdout and "FABRIC_NO_PRIMITIVE_BLOCK" in p.stdout
    p = subprocess.run([sys.executable, str(cli), "fabric", "rm", "--name", "soc"], capture_output=True,
                       text=True, env=env, timeout=120)
    assert p.returncode == 0, p.stderr
    assert db.fabric_specs() == [] and not (tmp_path / ".coresmith" / "fabric_spec.json").exists()
