# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Real-toolchain smoke of ``coresmith build module``: a supplied RTL seed
goes through the REAL compiled block subgraph on a persistent checkpoint
with REAL Verilator lint, REAL cocotb/Verilator simulation with coverage,
REAL Yosys synthesis against the sky130 liberty and REAL OpenSTA timing; the
SystemC reference model is REALLY built, smoked and evaluated with g++. The
only substitution is the testbench author (a deterministic cocotb test is
written where the model worker would write one). Skipped when the tools are
absent; see the docstrings for the environment it needs.

Run::

    PATH=/home/ubuntu/oss-cad-suite/bin:$PATH \\
    CORESMITH_SMOKE_LIBERTY=/path/to/sky130_fd_sc_hd__tt_025C_1v80.lib \\
    CORESMITH_SMOKE_STA_DIR=/path/to/dir/with/sta \\
    python -m pytest orchestrator/tests/test_build_toolchain_smoke.py -m slow -s
"""
from __future__ import annotations

import asyncio
import os
import shutil
import sqlite3
import types
from pathlib import Path

import pytest

from orchestrator.state_store import builds as B
from orchestrator.state_store.store import Scoreboard
from orchestrator.tests.build_fixtures import make_build_lifecycle, ready_project

pytestmark = pytest.mark.slow

_LIB_CANDIDATES = [
    os.environ.get("CORESMITH_SMOKE_LIBERTY", ""),
    "/home/ubuntu/openroad-src/test/sky130hd/sky130_fd_sc_hd__tt_025C_1v80.lib",
]
_STA_CANDIDATES = [os.environ.get("CORESMITH_SMOKE_STA_DIR", ""), "/home/ubuntu/openroad-src/build/src/sta"]

TB = '''import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge


@cocotb.test()
async def test_counter_counts(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.rst_n.value = 0
    for _ in range(3):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    seen = []
    for _ in range(6):
        await RisingEdge(dut.clk)
        seen.append(int(dut.m_out.value))
    assert seen == sorted(seen) and seen[-1] > seen[0], seen
'''

MODEL = '''#include "tiny_model.h"
tiny_model::tiny_model(sc_core::sc_module_name n) : sc_module(n), count(0) { SC_THREAD(run); sensitive << clk.pos(); }
void tiny_model::reset() { count = 0; }
void tiny_model::dump_state(std::ostream& os) const { os << "count=" << count << "\\n"; }
void tiny_model::run() { while (true) { wait(); if (rst_n.read()) { ++count; m_out.write((cs_beat_t)count); } } }
'''
SINK_MODEL = '''#include "sink_model.h"
sink_model::sink_model(sc_core::sc_module_name n) : sc_module(n), beats(0) { SC_THREAD(run); sensitive << clk.pos(); }
void sink_model::reset() { beats = 0; }
void sink_model::dump_state(std::ostream& os) const { os << "beats=" << beats << "\\n"; }
void sink_model::run() { while (true) { (void)s_in.read(); ++beats; } }
'''
HARNESS = '''#include "soc_model_top.h"
// The declared check (PERF-001: cycles per op) is MEASURED on the executed
// model: the cycles the clock ran after reset divided by the beats the sink
// model actually consumed. Nothing is constant; the value and the verdict
// come from the simulation that ran.
int sc_main(int argc, char** argv) {
    sc_core::sc_clock clk("clk", cs_clock_period());
    sc_core::sc_signal<bool> rst_n("rst_n");
    soc_model_top top("top");
    top.clk(clk); top.rst_n(rst_n);
    rst_n.write(false); sc_core::sc_start(3 * cs_clock_period());
    rst_n.write(true); top.reset_all();
    const sc_core::sc_time run(500, sc_core::SC_NS);
    sc_core::sc_start(run);
    const double cycles = run / cs_clock_period();
    const unsigned beats = top.u_sink.beats;
    if (beats == 0) {
        std::cout << "FRD_EVAL {\\"id\\": \\"PERF-001\\", \\"status\\": \\"fail\\", \\"evidence\\": \\"the sink consumed no beat\\"}" << std::endl;
    } else {
        const double cycles_per_op = cycles / beats;
        std::cout << "FRD_EVAL {\\"id\\": \\"PERF-001\\", \\"status\\": \\"" << (cycles_per_op <= 100.0 ? "pass" : "fail")
                  << "\\", \\"value\\": " << cycles_per_op << ", \\"unit\\": \\"cycles\\", \\"evidence\\": \\"" << cycles
                  << " cycles / " << beats << " beats consumed by the sink model\\"}" << std::endl;
    }
    std::cout << "FRD_EVAL_DONE" << std::endl;
    return 0;
}
'''

FRD = """# FRD

## Performance Requirements

1. **ID**: PERF-001
   - **Requirement**: cycles per op <= 100.
   - **Acceptance criteria**: measured on the model.
   - **Priority**: must_have
   - **Model check**: cycle accounting on the reference model.
"""


def _tools():
    lib = next((p for p in _LIB_CANDIDATES if p and Path(p).is_file()), None)
    sta_dir = next((p for p in _STA_CANDIDATES if p and (Path(p) / "sta").is_file()), None)
    missing = [t for t in ("verilator", "yosys", "make", "g++") if not shutil.which(t)]
    try:
        from orchestrator.systemc_model import detect
        sc_ok = detect()["ok"]
    except Exception:  # noqa: BLE001
        sc_ok = False
    try:
        from orchestrator.langgraph.pipeline_helpers import _verilator_version
        vv = _verilator_version(shutil.which("verilator") or "verilator")
    except Exception:  # noqa: BLE001
        vv = None
    return lib, sta_dir, missing, sc_ok, vv


@pytest.mark.slow
def test_seeded_module_build_with_real_tools(tmp_path, monkeypatch):
    lib, sta_dir, missing, sc_ok, vv = _tools()
    if missing or not lib or not sta_dir or not sc_ok or not vv or vv < (5, 36):
        pytest.skip(f"real toolchain incomplete: missing={missing} liberty={lib} sta={sta_dir} systemc={sc_ok} verilator={vv}")
    from orchestrator.harness.tools import integrate as it
    from orchestrator.langgraph import pipeline_graph as pg
    from orchestrator.langgraph import pipeline_helpers as ph
    from orchestrator.systemc_model.frd_eval import extract_requirements

    monkeypatch.setenv("PATH", f"{sta_dir}:{os.environ['PATH']}")
    pdk = tmp_path / "pdk" / "sky130A" / "libs.ref" / "sky130_fd_sc_hd" / "lib"
    pdk.mkdir(parents=True)
    (pdk / Path(lib).name).symlink_to(lib)
    monkeypatch.setenv("PDK_ROOT", str(tmp_path / "pdk"))
    monkeypatch.setattr(ph, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(ph, "LIBERTY_FILE", pdk / Path(lib).name)
    monkeypatch.setattr(ph, "_LOG_DIR", tmp_path / ".coresmith" / "step_logs")
    monkeypatch.setattr(ph, "PDK_ROOT", tmp_path / "pdk")

    db = ready_project(tmp_path, monkeypatch, with_rtl=True, with_model=False)
    # the REAL PPA gate (pre-layout OpenSTA timing): a build needs its verdict
    monkeypatch.setenv("CORESMITH_PPA_GATE", "1")
    (tmp_path / "arch" / "frd_spec.md").write_text(FRD)
    assert extract_requirements(FRD)
    # --- the REAL reference model: generated shell + the models above, built and smoked by g++
    res = it.model_build(db, tmp_path)
    assert res["missing_models"] == ["sink", "tiny"], res
    (tmp_path / "model" / "tiny_model.cpp").write_text(MODEL)
    (tmp_path / "model" / "sink_model.cpp").write_text(SINK_MODEL)
    for name, extra in (("tiny", "  unsigned count;\n"), ("sink", "  unsigned beats;\n")):
        hp = tmp_path / "model" / f"{name}_model.h"
        hp.write_text(hp.read_text().replace("  void run();", extra + "  void run();", 1))
    res = it.model_build(db, tmp_path)
    assert res["ok"] and res["build_ok"] and res["smoke_ok"], res.get("build_log", "")[-2000:] + res.get("smoke_log", "")
    row = db.model_for("tiny")
    assert row["build_ok"] and row["smoke_ok"] and row["sha"] == B.model_sha16(tmp_path, "tiny")
    # --- the REAL declared model check: a harness evaluated on the assembled model
    (tmp_path / "model" / "frd_eval" / "frd_eval.cpp").write_text(HARNESS)
    res = it.model_eval(db, tmp_path)
    assert res["ok"], res
    chk = db.latest_check("PERF-001", "model_eval")
    assert chk["status"] == "pass" and chk["sha"] == B.model_check_sha(db, tmp_path, "PERF-001")
    assert res["model_check_scope"] == ["PERF-001"]
    # the verdict is the measured number from the executed model, within the bound
    assert chk["value"] is not None and 0 < chk["value"] <= 100 and "beats consumed" in chk["evidence"]
    from orchestrator.state_store import stages as st
    assert st.module_ready(db, tmp_path, "tiny") == []

    # --- the build: only the testbench author is substituted
    async def write_tb(block, callbacks=None):
        p = tmp_path / block["testbench"]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(TB)
        return {"testbench_path": str(p)}
    monkeypatch.setattr(pg, "generate_testbench", write_tb)
    monkeypatch.setattr(pg, "create_golden_model_wrapper", lambda *a, **k: None)

    from orchestrator.daemon import server
    lc = make_build_lifecycle(tmp_path)
    monkeypatch.setattr(server, "_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr(server, "_build", lc)
    monkeypatch.setattr(server, "_pipeline", types.SimpleNamespace(task=None, thread_id="pipeline", status="idle"))
    monkeypatch.setattr(server, "_apply_run_env", lambda where: [])

    async def run():
        out = await server._start_module_build(server.BuildModuleRequest(module="tiny", seed_rtl="rtl/tiny.v"),
                                               entry="build_module")
        assert isinstance(out, dict), getattr(out, "body", out)
        bid = out["build_id"]
        for _ in range(2400):           # up to 20 minutes of real EDA
            row = B.get_build(db, bid)
            if row["status"] in ("parked",) + B.BUILD_TERMINAL:
                await asyncio.sleep(0.2)
                break
            await asyncio.sleep(0.5)
        row = B.get_build(db, bid)
        parks, _ = await server.MB.live_parks(lc, f"build-{bid}") if row["status"] == "parked" else ([], [])
        await lc.cleanup()
        return bid, row, parks
    bid, row, parks = asyncio.run(run())
    assert row["status"] == "completed", (row, parks, (tmp_path / ".coresmith" / "blocks" / "tiny" / "previous_error.txt").read_text()
                                           if (tmp_path / ".coresmith" / "blocks" / "tiny" / "previous_error.txt").exists() else "")
    rows = Scoreboard(tmp_path).rows_for_build(bid)
    dv, ppa = rows["dv"][-1], rows["ppa"][-1]
    assert dv["passed"] == 1 and dv["tests_total"] == 1
    assert ppa["tool"] == "yosys+opensta" and ppa["pdk"] == "sky130A" and ppa["cells"] and ppa["wns_ns"] is not None
    assert ppa["power_mw"] is None and ppa["power_basis"] == "unavailable"
    assert rows["coverage"] and rows["coverage"][-1]["pct"] is not None
    best = db.result("tiny", "best")
    assert best["build_id"] == bid and best["timing_ok"] is True
    assert row["result"]["seed_modified_by_repair"] is False and row["seed"]["sha256"] == B.file_sha256(tmp_path / "rtl" / "tiny.v")
    con = sqlite3.connect(str(tmp_path / ".coresmith" / "build_checkpoint.db"))
    assert con.execute("SELECT COUNT(*) FROM checkpoints WHERE thread_id=?", (f"build-{bid}",)).fetchone()[0] > 0
    con.close()
    lin = B.lineage(db, tmp_path, "tiny")
    assert lin["modules"]["tiny"]["intended_workflow"] is True
    print("\nSMOKE OK:", {"build": bid, "cells": ppa["cells"], "wns_ns": ppa["wns_ns"], "coverage_pct": rows["coverage"][-1]["pct"],
                          "verilator": vv, "yosys": shutil.which("yosys"), "sta": sta_dir, "liberty": lib})
