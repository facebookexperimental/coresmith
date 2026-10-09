# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Chip pins as rows: ``coresmith pin add|set|rm|lock|unlock|list``, the shell
boundary built from them, the interfaces-stage gate (``PINS_MISSING``), the
pinout rendering, and a regression on the MCU+FFT run-2 database (which died
at ``interfaces`` on ``SHELL_NO_BOUNDARY``)."""
import argparse
import json
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from orchestrator.harness import cli_contract as cc
from orchestrator.harness import cli_pins as cp
from orchestrator.langgraph import shell_integration as SI
from orchestrator.state_store import stages as st
from orchestrator.state_store.project_db import open_project

_REPO = Path(__file__).resolve().parents[2]
_RUN2 = Path("/home/ubuntu/coresmith-runs/mcufft-cli-tactics-v2-20260930")

_DIAGRAM = {
    "blocks": [
        {"name": "soc", "interfaces": ["m_uart"]},
        {"name": "uart", "interfaces": ["s_apb", "uart_tx", "uart_rx"]},
        {"name": "gpio", "interfaces": ["gpio_out", "gpio_in"]},
        {"name": "misc"},                                   # no interface rows: any port name
    ],
    "connections": [{"from": "soc", "to": "uart", "from_port": "m_uart", "to_port": "s_apb"}],
}
_APB_EDGE = {"edge_id": "soc__m_uart__to__uart__s_apb", "producer_block": "soc", "producer_port": "m_uart",
             "consumer_block": "uart", "consumer_port": "s_apb", "handshake_protocol": "static",
             "data_width_bits": 8, "fields": [{"name": "d", "width": 8, "msb": 7, "lsb": 0}]}


@pytest.fixture
def proj(tmp_path, monkeypatch):
    monkeypatch.setenv("CORESMITH_PROJECT_ROOT", str(tmp_path))
    monkeypatch.delenv("CORESMITH_SHELL_INFER_BOUNDARY", raising=False)
    monkeypatch.delenv("CORESMITH_CONTRACT_AUTOLOCK", raising=False)
    db = open_project(tmp_path)
    db.import_block_diagram(json.loads(json.dumps(_DIAGRAM)))
    db.import_contracts({"contracts": [dict(_APB_EDGE)]})
    return tmp_path, db


def _ns(root, **kw):
    base = {"project_root": str(root), "json": True, "unlock": False, "reason": ""}
    return argparse.Namespace(**{**base, **kw})


def _add(root, name, dir_, *, width=1, frm=None, kind=None, bus=None, msb=None, lsb=None, oe=None, **kw):
    return cp.cmd_pin_add(_ns(root, name=name, dir=dir_, width=width, from_=frm, kind=kind, bus=bus, msb=msb,
                              lsb=lsb, oe=oe, **kw))


def _out(capsys):
    return json.loads(capsys.readouterr().out)


def _codes(out):
    return sorted({p["code"] for p in out["problems"]})


# ---------------------------------------------------------------------------
# verbs

def test_add_list_and_views(proj, capsys):
    root, db = proj
    assert _add(root, "clk", "in", kind="clock") == 0
    assert _out(capsys)["version"] == 1
    assert _add(root, "uart_tx", "out", frm="uart.uart_tx") == 0
    capsys.readouterr()
    assert _add(root, "gpio_in", "in", width=8, frm="gpio.gpio_in") == 0
    capsys.readouterr()
    assert [p["name"] for p in db.pins()] == ["clk", "uart_tx", "gpio_in"]
    assert db.pin("gpio_in")["width"] == 8 and (db.pin("uart_tx")["block"], db.pin("uart_tx")["port"]) == ("uart", "uart_tx")
    assert db.artifact("pins")["path"] == "db:pins"
    view = json.loads((root / ".coresmith" / "pins.json").read_text())
    assert [p["name"] for p in view["pins"]] == ["clk", "uart_tx", "gpio_in"]
    assert cp.cmd_pin_list(_ns(root)) == 0
    assert [p["name"] for p in _out(capsys)["pins"]] == ["clk", "uart_tx", "gpio_in"]
    assert cp.cmd_pin_list(_ns(root, json=False)) == 0
    human = capsys.readouterr().out
    assert "gpio_in[7:0]  in    signal <- gpio.gpio_in" in human and "clk  in    clock  <- -" in human


@pytest.mark.parametrize("args,code", [
    (("uart_tx", "out", {"frm": "uart.uart_rx"}), "PIN_EXISTS"),
    (("x", "out", {"frm": "nosuch.p"}), "PIN_UNKNOWN_BLOCK"),
    (("x", "out", {"frm": "uart.nope"}), "PIN_UNKNOWN_PORT"),
    (("x", "in", {"frm": "uart.s_apb"}), "PIN_DOUBLE_DRIVEN"),          # the port is on a contract edge
    (("x", "out", {"frm": "uart.uart_tx"}), "PIN_DOUBLE_DRIVEN"),       # another pin exposes it
    (("x", "out", {}), "PIN_NO_SOURCE"),
    (("x", "out", {"frm": "misc.led", "width": 2, "bus": "io", "msb": 5, "lsb": 5}), "PIN_BAD_BUS"),
    (("x", "out", {"frm": "misc.led", "msb": 5, "lsb": 5}), "PIN_BAD_BUS"),
    (("9bad", "out", {"frm": "misc.led"}), "PIN_BAD_NAME"),
])
def test_add_refusals_write_nothing(proj, capsys, args, code):
    root, db = proj
    assert _add(root, "uart_tx", "out", frm="uart.uart_tx") == 0
    capsys.readouterr()
    name, dir_, kw = args
    assert _add(root, name, dir_, **kw) == 1
    out = _out(capsys)
    assert out["ok"] is False and code in _codes(out)
    assert [p["name"] for p in db.pins()] == ["uart_tx"]


def test_block_without_interface_rows_takes_any_port(proj, capsys):
    root, db = proj
    assert _add(root, "led", "out", width=2, frm="misc.led", bus="io", msb=6, lsb=5) == 0
    assert db.pin("led")["bus"] == "io"


def test_set_rm_and_versions(proj, capsys):
    root, db = proj
    assert _add(root, "uart_tx", "out", frm="uart.uart_tx") == 0
    assert cp.cmd_pin_set(_ns(root, name="uart_tx", field="from", value="uart.uart_rx")) == 0
    capsys.readouterr()
    assert db.pin("uart_tx")["port"] == "uart_rx" and db.pin("uart_tx")["version"] == 2
    assert cp.cmd_pin_set(_ns(root, name="uart_tx", field="from", value="uart.uart_rx")) == 0
    assert _out(capsys)["changed"] is False and db.pin("uart_tx")["version"] == 2
    assert cp.cmd_pin_set(_ns(root, name="uart_tx", field="from", value="nosuch.x")) == 1
    assert "PIN_UNKNOWN_BLOCK" in _codes(_out(capsys)) and db.pin("uart_tx")["port"] == "uart_rx"
    assert cp.cmd_pin_set(_ns(root, name="uart_tx", field="colour", value="red")) == 2
    assert cp.cmd_pin_set(_ns(root, name="nope", field="dir", value="in")) == 2
    assert cp.cmd_pin_rm(_ns(root, name="uart_tx")) == 0
    assert db.pins() == [] and not (root / ".coresmith" / "pins.json").exists()
    assert cp.cmd_pin_rm(_ns(root, name="uart_tx")) == 2


def test_lock_and_one_write_unlock(proj, capsys):
    root, db = proj
    assert _add(root, "uart_tx", "out", frm="uart.uart_tx") == 0
    assert cp.cmd_pin_lock(_ns(root, verb="lock")) == 0
    capsys.readouterr()
    assert db.pin("uart_tx")["locked"]
    assert cp.cmd_pin_set(_ns(root, name="uart_tx", field="width", value="2")) == 1
    assert _codes(_out(capsys)) == ["PIN_LOCKED"] and db.pin("uart_tx")["width"] == 1
    assert _add(root, "uart_rx", "in", frm="uart.uart_rx") == 1          # the set is locked
    assert _codes(_out(capsys)) == ["PIN_LOCKED"]
    assert cp.cmd_pin_rm(_ns(root, name="uart_tx")) == 1
    assert _codes(_out(capsys)) == ["PIN_LOCKED"]
    # --unlock needs a reason; it licenses ONE write and the pin stays locked
    assert cp.cmd_pin_set(_ns(root, name="uart_tx", field="width", value="2", unlock=True)) == 2
    assert cp.cmd_pin_set(_ns(root, name="uart_tx", field="width", value="2", unlock=True, reason="pad")) == 0
    assert _out(capsys)["reason"] == "pad"
    assert db.pin("uart_tx")["width"] == 2 and db.pin("uart_tx")["locked"]
    assert _add(root, "uart_rx", "in", frm="uart.uart_rx", unlock=True, reason="late pin") == 0
    assert db.pin("uart_rx")["locked"]                                    # joins the locked set
    assert cp.cmd_pin_lock(_ns(root, verb="unlock", reason="")) == 2
    assert cp.cmd_pin_lock(_ns(root, verb="unlock", reason="rework")) == 0
    capsys.readouterr()
    assert not any(p["locked"] for p in db.pins())
    assert cp.cmd_pin_set(_ns(root, name="uart_tx", field="width", value="1")) == 0


def test_contract_add_to_a_top_suggests_pins(proj, capsys):
    root, db = proj
    ns = _ns(root, producer="uart.uart_tx", consumer="chip_top.uart_tx", protocol="static", width=1,
             edge_id=None, bus_param=[], field=["d:1"], sideband=[], timing=[], policy=None, semantic=None, spec=None)
    assert cc.cmd_contract_add(ns) == 1
    out = _out(capsys)
    p = next(p for p in out["problems"] if p["code"] == "CT_UNKNOWN_BLOCK")
    assert "coresmith pin add" in p["text"]
    ns.consumer = "other.x"
    assert cc.cmd_contract_add(ns) == 1
    p = next(p for p in _out(capsys)["problems"] if p["code"] == "CT_UNKNOWN_BLOCK")
    assert "pin add" not in p["text"]


def test_cli_subprocess_pin_add_and_list(proj):
    root, db = proj
    env = {"CORESMITH_PROJECT_ROOT": str(root), "PATH": "/usr/bin:/bin", "PYTHONPATH": str(_REPO)}

    def run(*a):
        return subprocess.run([sys.executable, str(_REPO / "bin" / "coresmith"), *a],
                              capture_output=True, text=True, env=env, timeout=120)
    p = run("pin", "add", "gpio_out", "--dir", "out", "--width", "8", "--from", "gpio.gpio_out")
    assert p.returncode == 0 and "pin add gpio_out: OK v1" in p.stdout, p.stderr
    p = run("pin", "add", "bad", "--dir", "out", "--from", "gpio")
    assert p.returncode == 2 and "<block>.<port>" in p.stderr
    p = run("pin", "list", "--json")
    assert p.returncode == 0 and [x["name"] for x in json.loads(p.stdout)["pins"]] == ["gpio_out"]
    acts = open_project(root).actions()
    assert [a["argv"][:2] for a in acts][-3:] == [["pin", "add"], ["pin", "add"], ["pin", "list"]]


# ---------------------------------------------------------------------------
# shell assembly

_EDGE = {"edge_id": "req__m_q__to__rsp__s_q", "producer_block": "req", "producer_port": "m_q",
         "consumer_block": "rsp", "consumer_port": "s_q", "handshake_protocol": "req_resp",
         "data_width_bits": 8, "fields": [{"name": "addr", "width": 8, "msb": 7, "lsb": 0}],
         "sideband_signals": [{"name": "req_valid", "width": 1, "direction": "producer->consumer"},
                              {"name": "req_gnt", "width": 1, "direction": "consumer->producer"},
                              {"name": "rsp_valid", "width": 1, "direction": "consumer->producer"},
                              {"name": "rdata", "width": 8, "direction": "consumer->producer"}],
         "timing": {"req_to_rsp_cycles": {"exact": 1}}}
_PINS = [{"name": "clk", "dir": "in", "width": 1, "kind": "clock"},
         {"name": "rst_n", "dir": "in", "width": 1, "kind": "reset"},
         {"name": "led", "dir": "out", "width": 2, "block": "req", "port": "led", "kind": "signal"},
         {"name": "sw", "dir": "in", "width": 4, "block": "rsp", "port": "sw", "kind": "signal"}]


def _rsp_rtl(tmp_path, extra_port: bool):
    t = (Path(__file__).parent / "fixtures" / "vip2" / "responder.v").read_text().replace("module responder", "module rsp")
    t = t.replace("output reg  [7:0] s_q_rdata", "output reg  [7:0] s_q_rdata,\n  input  wire [3:0] sw"
                  + (",\n  output wire       irq" if extra_port else ""))
    if extra_port:
        t = t.replace("endmodule", "  assign irq = 1'b0;\nendmodule")
    p = tmp_path / "rsp.v"
    p.write_text(t)
    return p


def test_stubs_plus_pins_elaborate_with_boundary_equal_to_the_pins(tmp_path, monkeypatch):
    monkeypatch.delenv("CORESMITH_SHELL_INFER_BOUNDARY", raising=False)
    asm = SI.assemble_top(tmp_path, top_name="chip_top", blocks=["req", "rsp"], edges=[_EDGE], rtl_paths={},
                          out_dir=tmp_path / "shell", pins=_PINS)
    assert asm.wiring_errors == [] and asm.stubs == ["req", "rsp"]
    assert [b["name"] for b in asm.boundary_ports] == ["clk", "rst_n", "led", "sw"]
    # the stubs grew the pin ports; the top declares and wires them
    assert "output wire [1:0] led" in (tmp_path / "shell" / "stubs" / "req.v").read_text()
    assert "input  wire [3:0] sw" in (tmp_path / "shell" / "stubs" / "rsp.v").read_text()
    assert "output wire [1:0] led" in asm.verilog and ".led(led)" in asm.verilog and ".sw(sw)" in asm.verilog
    snap = SI.write_snapshot(tmp_path, asm, {"ok": True})
    assert snap["boundary"] == ["clk", "rst_n", "led", "sw"] and snap["boundary_ports"] == 4
    assert open_project(tmp_path).latest_integration_snapshot()["boundary"] == ["clk", "rst_n", "led", "sw"]
    if shutil.which("verilator"):
        res = SI.elaborate(asm)
        assert res["ran"] and res["ok"], res


def test_named_clock_pin_renames_the_clk_net(tmp_path):
    pins = [{"name": "sys_clk", "dir": "in", "kind": "clock"}, {"name": "sys_rst_n", "dir": "in", "kind": "reset"}]
    asm = SI.assemble_top(tmp_path, top_name="chip_top", blocks=["req", "rsp"], edges=[_EDGE], rtl_paths={},
                          out_dir=tmp_path / "shell", pins=pins)
    assert asm.wiring_errors == [] and "input  wire sys_clk" in asm.verilog and ".clk(sys_clk)" in asm.verilog
    assert ".rst_n(sys_rst_n)" in asm.verilog and [b["name"] for b in asm.boundary_ports] == ["sys_clk", "sys_rst_n"]


@pytest.mark.parametrize("infer", ["", "1"])
def test_undeclared_real_port_is_refused_unless_inferred(tmp_path, monkeypatch, infer):
    if infer:
        monkeypatch.setenv("CORESMITH_SHELL_INFER_BOUNDARY", infer)
    else:
        monkeypatch.delenv("CORESMITH_SHELL_INFER_BOUNDARY", raising=False)
    rsp = _rsp_rtl(tmp_path, extra_port=True)
    asm = SI.assemble_top(tmp_path, top_name="chip_top", blocks=["req", "rsp"], edges=[_EDGE],
                          rtl_paths={"rsp": str(rsp)}, out_dir=tmp_path / "shell", pins=_PINS)
    if infer:
        # pre-pins behaviour: the undeclared port becomes a boundary port
        assert asm.wiring_errors == []
        assert [b["name"] for b in asm.boundary_ports] == ["clk", "rst_n", "led", "sw", "rsp_irq"]
    else:
        assert [e.split(":")[0] for e in asm.wiring_errors] == ["SHELL_UNDECLARED_PORT rsp.irq"]
        assert ".irq()" in asm.verilog and "rsp_irq" not in asm.verilog
    # declaring it makes the assembly clean
    asm = SI.assemble_top(tmp_path, top_name="chip_top", blocks=["req", "rsp"], edges=[_EDGE],
                          rtl_paths={"rsp": str(rsp)}, out_dir=tmp_path / "shell",
                          pins=_PINS + [{"name": "irq", "dir": "out", "width": 1, "block": "rsp", "port": "irq"}])
    assert asm.wiring_errors == [] and ".irq(irq)" in asm.verilog


def test_pin_on_an_edge_net_and_dir_width_mismatch_are_reported(tmp_path, monkeypatch):
    monkeypatch.delenv("CORESMITH_SHELL_INFER_BOUNDARY", raising=False)
    rsp = _rsp_rtl(tmp_path, extra_port=False)
    bad = [{"name": "clk", "dir": "in", "kind": "clock"},
           {"name": "gnt", "dir": "out", "width": 1, "block": "rsp", "port": "s_q_req_gnt"},   # an edge port
           {"name": "sw", "dir": "out", "width": 4, "block": "rsp", "port": "sw"}]             # sw is an input
    asm = SI.assemble_top(tmp_path, top_name="chip_top", blocks=["req", "rsp"], edges=[_EDGE],
                          rtl_paths={"rsp": str(rsp)}, out_dir=tmp_path / "shell", pins=bad)
    errs = asm.wiring_errors
    assert any(e.startswith("PIN_DOUBLE_DRIVEN rsp.s_q_req_gnt") for e in errs)
    assert any("pin sw is out but rsp.sw is input" in e for e in errs)
    assert any(e.startswith("SHELL_BOUNDARY_MISMATCH") and "missing=['gnt', 'sw']" in e for e in errs)


def test_no_pins_keeps_the_inferred_boundary(tmp_path, monkeypatch):
    monkeypatch.delenv("CORESMITH_SHELL_INFER_BOUNDARY", raising=False)
    rsp = _rsp_rtl(tmp_path, extra_port=True)
    asm = SI.assemble_top(tmp_path, top_name="chip_top", blocks=["req", "rsp"], edges=[_EDGE],
                          rtl_paths={"rsp": str(rsp)}, out_dir=tmp_path / "shell")
    assert asm.wiring_errors == [] and [b["name"] for b in asm.boundary_ports] == ["rsp_sw", "rsp_irq"]


# ---------------------------------------------------------------------------
# interfaces stage

def _at_interfaces(root, db):
    from orchestrator.tests.build_fixtures import shared_stages_done
    shared_stages_done(db, "soc")        # the done rows before interfaces hold on re-evaluation
    (root / "arch").mkdir(exist_ok=True)
    (root / "arch" / "abi.md").write_text("abi " * 80)
    db.register_artifact("abi", "arch/abi.md")
    db.ensure_db_artifact("contracts")
    (root / ".coresmith" / "vip_index.json").write_text("{}")
    for i, s in enumerate(st.STAGES[:st.STAGES.index("interfaces")]):
        db.stage_set(s, i, "done")
    db.stage_set("interfaces", st.STAGES.index("interfaces"), "active")


def _codes_of(db, root):
    return [b["code"] for b in st.entry(db, root, "interfaces")]


def test_interfaces_stage_needs_pins_then_a_matching_shell(proj, capsys, monkeypatch):
    root, db = proj
    _at_interfaces(root, db)
    # the run-2 state: an elaborated stub shell with no boundary
    db.add_integration_snapshot({"top": "chip_top", "elaborated": True, "boundary_ports": 0, "boundary": []})
    assert _codes_of(db, root) == ["PINS_MISSING"]
    monkeypatch.setenv("CORESMITH_SHELL_INFER_BOUNDARY", "1")          # the old heuristic
    assert _codes_of(db, root) == ["SHELL_NO_BOUNDARY"]
    monkeypatch.delenv("CORESMITH_SHELL_INFER_BOUNDARY")
    assert _add(root, "clk", "in", kind="clock") == 0
    assert _add(root, "uart_tx", "out", frm="uart.uart_tx") == 0
    capsys.readouterr()
    b = st.entry(db, root, "interfaces")
    assert [x["code"] for x in b] == ["SHELL_BOUNDARY_MISMATCH"] and b[0]["ids"] == ["missing:clk", "missing:uart_tx"]
    # a shell that refused an undeclared port
    db.add_integration_snapshot({"top": "chip_top", "elaborated": None, "boundary_ports": 1, "boundary": ["clk"],
                                 "wiring_errors": ["SHELL_UNDECLARED_PORT gpio.gpio_in: on no contract edge and no pin"]})
    b = {x["code"]: x for x in st.entry(db, root, "interfaces")}
    assert set(b) == {"SHELL_NOT_ELABORATED", "SHELL_UNDECLARED_PORT"} and b["SHELL_UNDECLARED_PORT"]["ids"] == ["gpio.gpio_in"]
    db.add_integration_snapshot({"top": "chip_top", "elaborated": True, "boundary_ports": 2,
                                 "boundary": ["clk", "uart_tx"]})
    assert _codes_of(db, root) == []
    res = st.advance(db, root)
    assert res["advanced"] and res["locked_pins"] == 2 and all(p["locked"] for p in db.pins())


# ---------------------------------------------------------------------------
# rendering

def test_state_write_renders_pinout_and_caravel_pin_map(proj, capsys):
    from orchestrator.architecture.pin_map import load_pin_map
    from orchestrator.harness import cli_frd
    root, db = proj
    assert _add(root, "clk", "in", kind="clock") == 0
    assert _add(root, "uart_tx", "out", frm="uart.uart_tx", bus="io", msb=6, lsb=6, oe="tx_en") == 0
    assert _add(root, "gpio_in", "in", width=4, frm="gpio.gpio_in", bus="io", msb=5, lsb=2) == 0
    assert _add(root, "sda", "inout", frm="misc.sda", bus="io", msb=7, lsb=7, oe="sda_oe") == 0
    capsys.readouterr()
    assert cli_frd.cmd_state_write(_ns(root, force=False)) == 0
    out = _out(capsys)
    assert any(p.endswith("arch/pinout.md") for p in out["written"])
    md = (root / "arch" / "pinout.md").read_text()
    assert "rendered from project.sqlite" in md and "| `gpio_in` | in | 4 | signal | `gpio.gpio_in` | io[5:2] | - | open |" in md
    assert "| `clk` | in | 1 | clock | (shell clk/rst net) | - | - | open |" in md
    pm = load_pin_map(root)
    assert pm is not None and pm.ok, pm.errors if pm else None
    assert [(e.signal, e.dir, e.msb, e.lsb, e.oe) for e in pm.entries] == [
        ("uart_tx", "out", 6, 6, "tx_en"), ("gpio_in", "in", 5, 2, ""), ("sda_in", "in", 7, 7, ""),
        ("sda_out", "out", 7, 7, "sda_oe")]
    # a pin map written from the pins goes away with the last bus pin
    for n in ("uart_tx", "gpio_in", "sda"):
        assert cp.cmd_pin_set(_ns(root, name=n, field="bus", value="none")) == 1   # msb/lsb still set
        assert cp.cmd_pin_rm(_ns(root, name=n)) == 0
    capsys.readouterr()
    assert cli_frd.cmd_state_write(_ns(root, force=False)) == 0
    assert load_pin_map(root) is None


# ---------------------------------------------------------------------------
# regression: the MCU+FFT run-2 database

RUN2_PIN_COMMANDS = [
    ("clk", "in", {"kind": "clock"}),
    ("rst_n", "in", {"kind": "reset"}),
    ("uart_rx", "in", {"frm": "uart.uart_rx"}),
    ("uart_tx", "out", {"frm": "uart.uart_tx"}),
    ("gpio_in", "in", {"width": 8, "frm": "gpio.gpio_in"}),
    ("gpio_out", "out", {"width": 8, "frm": "gpio.gpio_out"}),
]


@pytest.mark.skipif(not (_RUN2 / ".coresmith" / "project.sqlite").exists(), reason="run-2 evidence not on this host")
def test_run2_database_leaves_interfaces_once_the_pins_are_declared(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CORESMITH_PROJECT_ROOT", str(tmp_path))
    for k in ("CORESMITH_SHELL_INFER_BOUNDARY", "CORESMITH_CONTRACT_AUTOLOCK", "CORESMITH_STAGE_FAIL_BLOCKS",
              "CORESMITH_ROLE"):
        monkeypatch.delenv(k, raising=False)
    (tmp_path / ".coresmith").mkdir()
    src = sqlite3.connect(f"file:{_RUN2 / '.coresmith' / 'project.sqlite'}?mode=ro", uri=True)
    dst = sqlite3.connect(str(tmp_path / ".coresmith" / "project.sqlite"))
    with dst:
        src.backup(dst)
    src.close()
    dst.close()
    shutil.copy(_RUN2 / ".coresmith" / "vip_index.json", tmp_path / ".coresmith" / "vip_index.json")
    db = open_project(tmp_path)              # migrates: pins table, snapshot boundary column
    assert st.current(db) == "interfaces"
    # what run 2 saw, and what it sees now
    monkeypatch.setenv("CORESMITH_SHELL_INFER_BOUNDARY", "1")
    assert [b["code"] for b in st.status(db, tmp_path)["blocked_by"]] == ["SHELL_NO_BOUNDARY"]
    monkeypatch.delenv("CORESMITH_SHELL_INFER_BOUNDARY")
    assert [b["code"] for b in st.status(db, tmp_path)["blocked_by"]] == ["PINS_MISSING"]
    # the six obvious pins (SAD Pinout: clk, rst_n, uart_rx, uart_tx, gpio_in[7:0], gpio_out[7:0])
    for name, dir_, kw in RUN2_PIN_COMMANDS:
        assert _add(tmp_path, name, dir_, **kw) == 0, capsys.readouterr().out
    capsys.readouterr()
    # the pseudo-block workaround gets the hint
    ns = _ns(tmp_path, producer="uart.uart_tx", consumer="chip_top.uart_tx", protocol="static", width=1,
             edge_id=None, bus_param=[], field=["d:1"], sideband=[], timing=[], policy=None, semantic=None, spec=None)
    assert cc.cmd_contract_add(ns) == 1
    assert "coresmith pin add" in json.dumps(_out(capsys))
    # re-assemble the stub shell from the contracts + pins
    if not shutil.which("verilator"):
        monkeypatch.setattr(SI, "elaborate", lambda asm, **k: {"ran": True, "ok": True, "errors": []})
    from orchestrator.harness.tools import integrate
    res = integrate.shell_assemble(db, tmp_path)
    assert res["ok"], res
    assert res["wiring_errors"] == [] and res["elaborated"] is True
    assert sorted(res["boundary"]) == sorted(n for n, _, _ in RUN2_PIN_COMMANDS)
    assert len(res["stubs"]) == 6 and res["wires"] == 121
    status = st.status(db, tmp_path)
    codes = [b["code"] for b in status["blocked_by"]]
    assert "SHELL_NO_BOUNDARY" not in codes and "PINS_MISSING" not in codes
    # run 2 opened a must-answer question (Q7) after requirements was marked
    # done: the done row is history, the stage is reported regressed until the
    # question has its ruling
    assert [r["stage"] for r in status["regressed"]] == ["requirements"]
    assert status["regressed"][0]["ids"] == ["OPEN_QUESTIONS"] and status["can_advance"] is False
    for q in db.questions(open_only=True, must_answer=True):
        assert db.answer_question(q["id"], "ruled for the regression test")
    status = st.status(db, tmp_path)
    assert status["can_advance"], (status["blocked_by"], status["regressed"])
    adv = st.advance(db, tmp_path)
    assert adv["advanced"] and adv["done"] == "interfaces" and adv["stage"] == "uarch"
    assert adv["locked_pins"] == 6 and adv["locked_edges"] == 8
