# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""An unseeded module build judged by FRD targets, end to end through the
REAL compiled block subgraph (the daemon's ``build module`` entry, the
persistent checkpoint, the parks, the publication). Only the worker and the
EDA tools are faked -- and the fakes write the files the real tools write
(results.xml, measurements.jsonl, the yosys report, the netlist), so every
number the build judges is read back from a tool's file:

* no RTL on disk and no seed: the worker implements it; the Architect's
  acceptance testbench is run as-is and never authored;
* a measured throughput ABOVE its bound routes back to the worker with the
  gap; the next attempt BELOW it completes and stamps the FRD checks;
* a measurement the acceptance test did not record parks
  ``target_unmeasured`` -- never a pass -- and an abort leaves nothing
  published;
* the acceptance oracle cannot change during a build (park, restore);
* an FRD revision makes a parked build un-resumable and a completed one
  stale;
* coverage closure adds supplemental tests next to the unchanged oracle;
* the first candidate meeting the required targets completes the build."""
from __future__ import annotations

import asyncio
import json
import time
import types
from pathlib import Path

import pytest

from orchestrator.state_store import builds as B
from orchestrator.state_store import stages as st
from orchestrator.state_store.store import Scoreboard
from orchestrator.tests.build_fixtures import (
    add_module_targets,
    fake_graph_helpers,
    make_build_lifecycle,
    ready_project,
)

LIB = '''library(fake) {
  cell ("sky130_fd_sc_hd__dfxtp_1") { area : 20.0; }
  cell ("sky130_fd_sc_hd__inv_1") { area : 3.75; }
}
'''


class Tools:
    """Per-attempt tool outcomes. ``throughput[a]`` is what the acceptance
    test measures on attempt ``a`` (None: the test records nothing);
    ``area[a]`` the std-cell area yosys reports."""

    def __init__(self, root: Path, *, throughput, area, coverage_ok=True, record=True):
        self.root = Path(root)
        self.throughput = list(throughput)
        self.area = list(area)
        self.coverage_ok = coverage_ok
        self.record = record
        self.sims: list[dict] = []
        self.rtl_versions: list[str] = []

    def _pick(self, seq, attempt):
        return seq[min(attempt, len(seq)) - 1]

    def sim(self, block, rtl_path, tb_path, attempt=1, **kw):
        name = block["name"]
        sim_dir = self.root / "sim_build" / name
        sim_dir.mkdir(parents=True, exist_ok=True)
        extra = [Path(p) for p in kw.get("extra_tb_paths") or []]
        self.sims.append({"attempt": attempt, "tb": tb_path, "extra": [str(p) for p in extra]})
        cases = ['<testcase classname="test_tiny" name="test_count"/>',
                 '<testcase classname="test_tiny" name="test_throughput"/>']
        cases += [f'<testcase classname="{p.stem}" name="test_supp_a"/>' for p in extra]
        (sim_dir / "results.xml").write_text("<testsuites><testsuite>" + "".join(cases) + "</testsuite></testsuites>")
        m = sim_dir / "measurements.jsonl"
        m.unlink(missing_ok=True)
        value = self._pick(self.throughput, attempt)
        if self.record and value is not None:
            m.write_text(json.dumps({"item": "PERF-TINY-1", "value": value, "unit": "cycles/op",
                                     "test": "test_throughput", "module": "test_tiny", "ts": time.time()}) + "\n")
        covered = self.coverage_ok or bool(extra)
        cov = {"applicable": True, "pct": 95.0 if covered else 40.0, "floor": 70.0, "points_total": 20,
               "points_hit": 19 if covered else 8, "uncovered_count": 1 if covered else 12, "passed": covered}
        base = {"returncode": 0, "tests_total": len(cases), "coverage": cov, "log_path": "",
                "throughput": {"applicable": False, "reason": "none declared"}, "sim_dir": str(sim_dir),
                "results_xml": str(sim_dir / "results.xml"), "measurements_path": str(m)}
        if not covered:
            return {**base, "passed": False, "coverage_gate_failed": True, "tests_passed": len(cases),
                    "tests_failed": 0, "log": "LINE-COVERAGE GATE: 40% (floor 70%): FAIL"}
        return {**base, "passed": True, "log": "all passed", "tests_passed": len(cases), "tests_failed": 0}

    def synth(self, block, rtl_path, target_clock_mhz=50.0, attempt=1, **kw):
        name = block["name"]
        out = self.root / "syn" / "output" / name
        out.mkdir(parents=True, exist_ok=True)
        area = self._pick(self.area, attempt)
        (out / f"{name}_report.txt").write_text(f"=== {name} ===\n   Chip area for module '\\{name}': {area}\n")
        (out / f"{name}_netlist.v").write_text(
            f"module {name}(clk);\n  input clk;\n  sky130_fd_sc_hd__dfxtp_1 _1_ (.CLK(clk));\n"
            f"  // attempt {attempt}\nendmodule\n")
        (out / f"{name}.sdc").write_text("create_clock -name clk -period 20.0 [get_ports clk]\n")
        return {"success": True, "gate_count": 40 + attempt, "ff_count": 8, "chip_area_um2": area,
                "liberty_path": str(self.root / "lib" / "fake.lib"),
                "netlist_path": str(out / f"{name}_netlist.v"), "report_path": str(out / f"{name}_report.txt"),
                "sdc_path": str(out / f"{name}.sdc"), "log": "", "log_path": ""}


@pytest.fixture
def daemon(tmp_path, monkeypatch):
    from orchestrator.daemon import server
    from orchestrator.langgraph import pipeline_helpers as ph
    db = ready_project(tmp_path, monkeypatch)          # no RTL on disk: the build is unseeded
    ids = add_module_targets(db, tmp_path)
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib" / "fake.lib").write_text(LIB)
    monkeypatch.setattr(ph, "LIBERTY_FILE", tmp_path / "lib" / "fake.lib")
    lc = make_build_lifecycle(tmp_path)
    monkeypatch.setattr(server, "_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr(server, "_build", lc)
    monkeypatch.setattr(server, "_pipeline", types.SimpleNamespace(task=None, thread_id="pipeline", status="idle"))
    monkeypatch.setattr(server, "_apply_run_env", lambda where: [])
    monkeypatch.setattr(server, "_preflight_or_400", lambda: None)
    yield server, db, lc, ids
    asyncio.run(lc.cleanup())


def _install(monkeypatch, root, tools: Tools, *, rtl_hook=None, calls=None):
    from orchestrator.langgraph import pipeline_graph as pg
    calls = fake_graph_helpers(monkeypatch, root, calls=calls if calls is not None else {})
    seen_errors: list[str] = []

    async def gen_rtl(block, attempt, callbacks=None):
        calls["generate_rtl"] = calls.get("generate_rtl", 0) + 1
        err = root / ".coresmith" / "blocks" / block["name"] / "previous_error.txt"
        seen_errors.append(err.read_text() if err.exists() else "")
        p = root / "rtl" / f"{block['name']}.v"
        p.parent.mkdir(parents=True, exist_ok=True)
        text = f"module {block['name']}(input clk, input rst_n, output reg [7:0] m_out);  // attempt {attempt}\nendmodule\n"
        p.write_text(text)
        tools.rtl_versions.append(text)
        if rtl_hook:
            rtl_hook(attempt)
        return {"rtl_path": str(p)}

    async def supp(block_name, rtl_path, acceptance_path, supplemental_path, sim_log_path, callbacks=None):
        calls["author_supplemental_tests"] = calls.get("author_supplemental_tests", 0) + 1
        Path(supplemental_path).write_text("import cocotb\n\n@cocotb.test()\nasync def test_supp_a(dut):\n    pass\n")
        return True

    monkeypatch.setattr(pg, "generate_rtl", gen_rtl)
    monkeypatch.setattr(pg, "run_simulation", tools.sim)
    monkeypatch.setattr(pg, "synthesize_block", tools.synth)
    monkeypatch.setattr(pg, "author_supplemental_tests", supp)
    return calls, seen_errors


async def _settle(db, build_id: str, *, until=("parked",) + B.BUILD_TERMINAL, timeout_s: float = 60.0) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        row = B.get_build(db, build_id)
        if row and row["status"] in until:
            await asyncio.sleep(0.05)
            return B.get_build(db, build_id)
        await asyncio.sleep(0.05)
    raise AssertionError(f"build {build_id} did not settle: {B.get_build(db, build_id)}")


async def _start(server, module="tiny", **kw):
    resp = await server._start_module_build(server.BuildModuleRequest(module=module, **kw), entry="build_module")
    assert isinstance(resp, dict), getattr(resp, "body", resp)
    return resp


class TestClosure:
    async def test_a_miss_goes_back_to_the_worker_and_a_met_target_completes(self, daemon, tmp_path, monkeypatch):
        server, db, lc, ids = daemon
        tb = tmp_path / ids["tb"]
        tb_sha = B.file_sha256(tb)
        tools = Tools(tmp_path, throughput=[6.0, 3.0], area=[80.0])
        calls, errors = _install(monkeypatch, tmp_path, tools)
        assert not (tmp_path / "rtl" / "tiny.v").exists()
        out = await _start(server)
        assert out["max_attempts"] == 3 and out["retries"] == 2
        assert out["targets"]["required"] == [ids["perf"], ids["area"]] and out["targets"]["functional"] == ["FUNC-001"]
        assert out["targets"]["acceptance_tb"] == ids["tb"]
        bid = out["build_id"]
        row = await _settle(db, bid)
        assert row["status"] == "completed", row
        # the worker implemented the RTL twice: the second time with the measured gap
        assert calls["generate_rtl"] == 2
        assert "TARGET MISS" in errors[1] and ids["perf"] in errors[1] and "over by 2" in errors[1]
        # the acceptance testbench was run as-is: no testbench-author call, bytes unchanged
        assert calls.get("generate_testbench", 0) == 0 and calls.get("author_supplemental_tests", 0) == 0
        assert B.file_sha256(tb) == tb_sha and all(s["tb"] == str(tb) for s in tools.sims)
        # every candidate is in the ledger with its receipts
        cands = B.candidates_for(db, bid)
        assert [(c["attempt"], c["outcome"]) for c in cands] == [(1, "target_miss"), (2, "feasible")]
        t2 = {t["id"]: t for t in cands[1]["evaluation"]["targets"]}
        assert t2[ids["perf"]]["value"] == 3.0 and t2[ids["area"]]["value"] == 80.0
        rec = t2[ids["perf"]]["receipts"][0]["receipt"]
        assert rec["results_sha256"] and rec["measurements_sha256"]
        assert t2[ids["area"]]["receipts"][0]["receipt"]["report_sha256"]
        # the publication names the candidate; the FRD checks carry the measured values
        best = db.result("tiny", "best")
        assert best["build_id"] == bid and best["candidate_id"] == cands[1]["id"]
        assert best["targets"][ids["perf"]]["value"] == 3.0
        chk = db.latest_check(ids["perf"], "block_dv")
        assert chk["status"] == "pass" and chk["value"] == 3.0 and chk["actor"] == f"build:{bid}"
        assert db.latest_check(ids["area"], "block_dv")["value"] == 80.0
        assert db.latest_check("FUNC-001", "block_dv")["status"] == "pass"
        assert B.current_build_status(db, tmp_path, "tiny")["ok"]
        # the PPA row of the published attempt and the brief the worker read
        brief = (tmp_path / ".coresmith" / "blocks" / "tiny" / "build_targets.md").read_text()
        assert ids["perf"] in brief and "READ-ONLY" in brief
        state = await server.build_state(bid)
        assert [c["outcome"] for c in state["candidates"]] == ["target_miss", "feasible"]
        # the experiment record: compare and lineage carry the measured targets / candidates
        cmp = B.compare(db, tmp_path, bid, bid)
        assert cmp["a"]["targets"]["values"][ids["perf"]] == {"value": 3.0, "unit": "cycles/op", "status": "pass"}
        assert cmp["a"]["targets"]["candidate_id"] == cands[1]["id"]
        assert B.lineage(db, tmp_path, "tiny")["modules"]["tiny"]["builds"][0]["candidates"] == 2

    async def test_attempts_exhausted_on_misses_fail_without_publication(self, daemon, tmp_path, monkeypatch):
        server, db, lc, ids = daemon
        tools = Tools(tmp_path, throughput=[6.0], area=[80.0])
        calls, _ = _install(monkeypatch, tmp_path, tools)
        out = await _start(server, max_attempts=2)
        row = await _settle(db, out["build_id"])
        assert row["status"] == "failed" and "TARGET MISS" in row["error"]
        assert calls["generate_rtl"] == 2 and db.result("tiny", "best") is None
        assert [c["outcome"] for c in B.candidates_for(db, out["build_id"])] == ["target_miss", "target_miss"]

    async def test_missing_evidence_parks_and_abort_publishes_nothing(self, daemon, tmp_path, monkeypatch):
        server, db, lc, ids = daemon
        tools = Tools(tmp_path, throughput=[3.0], area=[80.0], record=False)
        _install(monkeypatch, tmp_path, tools)
        out = await _start(server)
        bid = out["build_id"]
        assert (await _settle(db, bid))["status"] == "parked"
        state = await server.build_state(bid)
        park = state["interrupts"][0]["payload"]
        assert park["type"] == "target_unmeasured" and park["unmeasured"] == [ids["perf"]]
        assert park["supported_actions"] == ["retry", "abort"] and "never a pass" in park["outer_agent_guidance"]
        assert B.candidates_for(db, bid)[-1]["outcome"] == "unmeasured"
        await server.build_resume(server.BuildResumeRequest(build_id=bid, action="abort"))
        row = await _settle(db, bid, until=B.BUILD_TERMINAL)
        assert row["status"] == "aborted" and "measurement missing" in row["error"]
        assert db.result("tiny", "best") is None

    async def test_retry_after_the_measurement_is_available_completes(self, daemon, tmp_path, monkeypatch):
        server, db, lc, ids = daemon
        tools = Tools(tmp_path, throughput=[3.0], area=[80.0], record=False)
        calls, _ = _install(monkeypatch, tmp_path, tools)
        out = await _start(server)
        bid = out["build_id"]
        assert (await _settle(db, bid))["status"] == "parked"
        tools.record = True                         # the environment now measures it
        await server.build_resume(server.BuildResumeRequest(build_id=bid, action="retry"))
        row = await _settle(db, bid, until=B.BUILD_TERMINAL)
        assert row["status"] == "completed" and calls["generate_rtl"] == 1
        assert [c["outcome"] for c in B.candidates_for(db, bid)] == ["unmeasured", "feasible"]

    async def test_an_frd_revision_makes_a_parked_build_unresumable(self, daemon, tmp_path, monkeypatch):
        server, db, lc, ids = daemon
        _install(monkeypatch, tmp_path, Tools(tmp_path, throughput=[3.0], area=[80.0], record=False))
        out = await _start(server)
        bid = out["build_id"]
        assert (await _settle(db, bid))["status"] == "parked"
        db.edit_item(ids["perf"], bound_max=2.0)            # the Architect revises the target
        state = await server.build_state(bid)
        assert state["resumable"] is False and any(r.startswith("targets:") for r in state["stale"])
        resp = await server.build_resume(server.BuildResumeRequest(build_id=bid, action="retry"))
        assert resp.status_code == 409 and json.loads(resp.body)["error"] == "BUILD_STALE"


class TestAcceptanceOracle:
    async def test_a_changed_oracle_parks_and_restore_puts_it_back(self, daemon, tmp_path, monkeypatch):
        server, db, lc, ids = daemon
        tb = tmp_path / ids["tb"]
        original = tb.read_text()

        def weaken(attempt):                             # the worker edits the oracle
            tb.write_text(original.replace("assert True", "pass  # weakened"))
        _install(monkeypatch, tmp_path, Tools(tmp_path, throughput=[3.0], area=[80.0]), rtl_hook=weaken)
        out = await _start(server)
        bid = out["build_id"]
        assert (await _settle(db, bid))["status"] == "parked"
        park = (await server.build_state(bid))["interrupts"][0]["payload"]
        assert park["type"] == "acceptance_changed" and park["files"] == [ids["tb"]]
        await server.build_resume(server.BuildResumeRequest(build_id=bid, action="restore"))
        row = await _settle(db, bid, until=B.BUILD_TERMINAL)
        assert row["status"] == "completed"
        assert tb.read_text() == original                                     # the oracle the build recorded
        assert "weakened" in (tb.parent / f"{tb.name}.changed-{bid}").read_text()   # the change is kept aside

    async def test_coverage_closure_adds_supplemental_tests_only(self, daemon, tmp_path, monkeypatch):
        server, db, lc, ids = daemon
        tb = tmp_path / ids["tb"]
        tb_sha = B.file_sha256(tb)
        tools = Tools(tmp_path, throughput=[3.0], area=[80.0], coverage_ok=False)
        calls, _ = _install(monkeypatch, tmp_path, tools)
        out = await _start(server)
        row = await _settle(db, out["build_id"])
        assert row["status"] == "completed", row
        supp = tb.parent / "test_tiny_supplemental.py"
        assert calls["author_supplemental_tests"] == 1 and calls.get("generate_testbench", 0) == 0
        assert supp.is_file() and B.file_sha256(tb) == tb_sha
        assert tools.sims[-1]["extra"] == [str(supp)]
        assert str(supp) in row["result"]["produced"]["files_sha256"] or str(supp) in (
            B.candidates_for(db, out["build_id"])[-1]["files"])


class TestCompletion:
    async def test_first_passing_candidate_completes_with_recorded_targets(self, daemon, tmp_path, monkeypatch):
        server, db, lc, ids = daemon
        tools = Tools(tmp_path, throughput=[3.0], area=[80.0, 60.0])
        calls, errors = _install(monkeypatch, tmp_path, tools)
        out = await _start(server)
        row = await _settle(db, out["build_id"])
        assert row["status"] == "completed", row
        assert row["result"]["attempt"] == 1 and calls["generate_rtl"] == 1
        assert ids["area"] in out["targets"]["required"] and ids["perf"] in out["targets"]["required"]
        brief = (tmp_path / ".coresmith/blocks/tiny/build_targets.md").read_text()
        assert ids["area"] in brief and ids["perf"] in brief and "REQUIRED" in brief
        candidates = B.candidates_for(db, out["build_id"])
        assert len(candidates) == 1 and candidates[0]["outcome"] == "feasible"
        assert row["result"]["candidate_id"] == candidates[0]["id"]
        assert db.latest_check(ids["area"], "block_dv")["value"] == 80.0
        assert B.current_build_status(db, tmp_path, "tiny")["ok"]
        receipt = next(t for t in candidates[0]["evaluation"]["targets"] if t["id"] == ids["area"])["receipts"][0]["receipt"]
        assert "/candidates/a1/evidence/" in receipt["report_path"]
        assert "80.0" in Path(receipt["report_path"]).read_text()


def _fake_sta(tmp_path, monkeypatch, total_w: str = "2.00e-03") -> Path:
    """An ``sta`` that checks the script it was given and prints a power report."""
    import os
    import stat as _stat
    bindir = tmp_path / "stabin"
    bindir.mkdir(exist_ok=True)
    sta = bindir / "sta"
    sta.write_text("#!/bin/sh\nscript=\"$3\"\n"
                   "grep -q 'set_power_activity -input -activity 0.2 -duty 0.5' \"$script\" || exit 3\n"
                   "grep -q 'read_sdc' \"$script\" || exit 4\n"
                   f"echo 'Total                  1.00e-03   1.00e-03   0.00e+00   {total_w} 100.0%'\n")
    sta.chmod(sta.stat().st_mode | _stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}")
    return sta


class TestEvidence:
    async def test_power_is_measured_by_the_sta_invocation_at_the_bound_condition(self, daemon, tmp_path,
                                                                                    monkeypatch):
        server, db, lc, ids = daemon
        add_module_targets(db, tmp_path, throughput=False, area=False, power=True, functional=False, tb=False)
        ids_power = "PWR-TINY-1"
        db.add_verifier(ids_power, "eda", entry="power_mw", block="tiny", args={"activity": 0.2, "duty": 0.5})
        _fake_sta(tmp_path, monkeypatch)
        _install(monkeypatch, tmp_path, Tools(tmp_path, throughput=[3.0], area=[80.0]))
        out = await _start(server)
        row = await _settle(db, out["build_id"])
        assert row["status"] == "completed", row
        t = {x["id"]: x for x in B.latest_candidate(db, out["build_id"])["evaluation"]["targets"]}[ids_power]
        rec = t["receipts"][0]["receipt"]
        assert t["value"] == pytest.approx(2.0) and rec["activity"] == 0.2 and rec["clock_period_ns"] == 20.0
        assert "/evidence/" in rec["report_path"] and "/evidence/" in rec["sdc_path"]
        ppa = Scoreboard(tmp_path).rows_for_build(out["build_id"])["ppa"][-1]
        assert ppa["power_basis"] == "estimated" and ppa["power_mw"] == pytest.approx(2.0)
        assert db.latest_check(ids_power, "block_dv")["value"] == pytest.approx(2.0)

    async def test_evidence_changed_before_publication_is_refused(self, daemon, tmp_path, monkeypatch):
        from orchestrator.langgraph import pipeline_graph as pg
        server, db, lc, ids = daemon
        add_module_targets(db, tmp_path, throughput=False, area=False, power=True, functional=False, tb=False)
        db.add_verifier("PWR-TINY-1", "eda", entry="power_mw", block="tiny", args={"activity": 0.2, "duty": 0.5})
        _fake_sta(tmp_path, monkeypatch)
        _install(monkeypatch, tmp_path, Tools(tmp_path, throughput=[3.0], area=[80.0]))
        evaluate = pg.evaluate_targets_node

        async def then_tamper(state):
            out = await evaluate(state)
            if out.get("targets_route") == "done":
                cand = B.latest_candidate(db, state["build_id"])
                recs = [r["receipt"] for t in cand["evaluation"]["targets"] for r in t["receipts"]
                        if r["kind"] == "eda" and r["measure"].startswith("power")]
                Path(recs[0]["sdc_path"]).write_text("create_clock -name clk -period 2.0 [get_ports clk]\n")
            return out
        monkeypatch.setattr(pg, "evaluate_targets_node", then_tamper)
        out = await _start(server)
        row = await _settle(db, out["build_id"])
        assert row["status"] == "failed" and "FRD_TARGETS_STALE" in row["error"], row
        assert db.result("tiny", "best") is None


class TestOneNetlist:
    """The PPA gate may select the fan-out-repaired netlist for timing. Then
    that netlist is THE block netlist: gate simulation, area, power and the
    published files all refer to it -- the timing of one netlist is never
    combined with the cheaper area of another."""

    def _select_buffered(self, monkeypatch, *, buffered_area: float):
        from orchestrator.langgraph import pipeline_graph as pg

        def fake_ppa(project_root, block_name, rtl_path, synth_result, *, require_gate_flag=True):
            out = Path(project_root) / "syn" / "output" / block_name
            buf = out / f"{block_name}_sta_buf.v"
            buf.write_text(f"module {block_name}(clk);\n  input clk;\n  sky130_fd_sc_hd__dfxtp_1 _1_ (.CLK(clk));\n"
                           "  sky130_fd_sc_hd__inv_1 _rep_ (.A(clk));\nendmodule\n")
            (out / f"{block_name}_sta_buf_stat.rpt").write_text(
                f"=== {block_name} ===\n   Chip area for module '\\{block_name}': {buffered_area}\n")
            rpt = out / f"{block_name}_sta_buf.rpt"
            rpt.write_text("wns max 1.00\n")
            return True, [], {"wns_ns": 1.0, "tns_ns": 0.0, "timing_required": True, "cells": 50, "ff": 8,
                              "area_um2": buffered_area, "ppa_variant": "buf", "netlist_repair_status": "repaired",
                              "ppa_netlist_path": str(buf), "ppa_netlist_sha256": B.file_sha256(buf),
                              "sta_report_path": str(rpt)}
        monkeypatch.setattr(pg, "_evaluate_ppa_gate", fake_ppa)
        simulated: list[str] = []

        def gate_sim(state, block, name, res, rtl):
            simulated.append(Path(res["netlist_path"]).read_text())
            return None, "disabled", "test"
        monkeypatch.setattr(pg, "_run_gate_sim_gate", gate_sim)
        return simulated

    async def test_buffered_timing_cannot_borrow_the_unbuffered_area(self, daemon, tmp_path, monkeypatch):
        server, db, lc, ids = daemon
        _install(monkeypatch, tmp_path, Tools(tmp_path, throughput=[3.0], area=[80.0]))   # yosys map: 80 um2
        simulated = self._select_buffered(monkeypatch, buffered_area=130.0)            # timing netlist: 130 um2
        out = await _start(server, max_attempts=1)
        row = await _settle(db, out["build_id"])
        assert row["status"] == "failed" and db.result("tiny", "best") is None, row
        cand = B.latest_candidate(db, out["build_id"])
        area = {t["id"]: t for t in cand["evaluation"]["targets"]}[ids["area"]]
        assert cand["outcome"] == "target_miss" and area["value"] == 130.0 and area["status"] == "fail"
        rec = area["receipts"][0]["receipt"]
        syn = tmp_path / "syn" / "output" / "tiny"
        buffered = (syn / "tiny_sta_buf.v").read_text()
        assert rec["netlist_sha256"] == rec["timing_netlist_sha256"] == B.file_sha256(syn / "tiny_sta_buf.v")
        assert rec["netlist_variant"] == "buf" and rec["original_netlist_sha256"]
        # the canonical files ARE the timing netlist and its statistics; the yosys map is kept as evidence
        assert (syn / "tiny_netlist.v").read_text() == buffered and simulated == [buffered]
        assert "130.0" in (syn / "tiny_report.txt").read_text()
        assert "80.0" in (syn / "tiny_report.synth.txt").read_text() and (syn / "tiny_netlist.synth.v").is_file()

    async def test_the_published_build_names_the_timing_netlist(self, daemon, tmp_path, monkeypatch):
        server, db, lc, ids = daemon
        _install(monkeypatch, tmp_path, Tools(tmp_path, throughput=[3.0], area=[80.0]))
        self._select_buffered(monkeypatch, buffered_area=90.0)
        out = await _start(server)
        row = await _settle(db, out["build_id"])
        assert row["status"] == "completed", row
        syn = tmp_path / "syn" / "output" / "tiny"
        canon = str(syn / "tiny_netlist.v")
        assert row["result"]["produced"]["files_sha256"][canon] == B.file_sha256(syn / "tiny_sta_buf.v")
        assert db.result("tiny", "best")["gate_count"] == 50 and db.latest_check(ids["area"], "block_dv")["value"] == 90.0
        ppa = Scoreboard(tmp_path).rows_for_build(out["build_id"])["ppa"][-1]
        assert ppa["cells"] == 50 and ppa["area_um2"] == 90.0


@pytest.fixture
def legacy_daemon(tmp_path, monkeypatch):
    """A module with no FRD targets at all (the pre-targets shape)."""
    from orchestrator.daemon import server
    db = ready_project(tmp_path, monkeypatch)
    lc = make_build_lifecycle(tmp_path)
    monkeypatch.setattr(server, "_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr(server, "_build", lc)
    monkeypatch.setattr(server, "_pipeline", types.SimpleNamespace(task=None, thread_id="pipeline", status="idle"))
    monkeypatch.setattr(server, "_apply_run_env", lambda where: [])
    monkeypatch.setattr(server, "_preflight_or_400", lambda: None)
    yield server, db, lc
    asyncio.run(lc.cleanup())


class TestMissingSelectedNetlist:
    def test_a_missing_buffered_selection_never_falls_back_to_the_yosys_map(self, tmp_path):
        from orchestrator.langgraph import pipeline_graph as pg
        syn = tmp_path / "syn" / "output" / "tiny"
        syn.mkdir(parents=True)
        (syn / "tiny_netlist.v").write_text("module tiny(clk); input clk; endmodule\n")
        (syn / "tiny_report.txt").write_text("Chip area for module '\\tiny': 80.0\n")
        meta = {"ppa_variant": "buf", "ppa_netlist_path": str(syn / "tiny_sta_buf.v"), "ppa_netlist_sha256": "a" * 64,
                "wns_ns": 1.0, "cells": 50, "area_um2": 130.0}
        out = pg._canonical_netlist(str(tmp_path), "tiny", {"netlist_path": str(syn / "tiny_netlist.v"),
                                                            "report_path": str(syn / "tiny_report.txt")}, meta)
        assert "not on disk" in out["netlist_selection"]["error"]
        assert out["timing_netlist_sha256"] == "a" * 64 != B.file_sha256(syn / "tiny_netlist.v")
        assert "130" not in (syn / "tiny_report.txt").read_text()       # nothing was adopted or blessed

    async def test_a_legacy_build_cannot_publish_timing_of_a_missing_netlist(self, legacy_daemon, tmp_path,
                                                                             monkeypatch):
        from orchestrator.langgraph import pipeline_graph as pg
        server, db, lc = legacy_daemon
        fake_graph_helpers(monkeypatch, tmp_path)

        def fake_ppa(project_root, block_name, rtl_path, synth_result, *, require_gate_flag=True):
            gone = Path(project_root) / "syn" / "output" / block_name / f"{block_name}_sta_buf.v"
            return True, [], {"wns_ns": 2.0, "tns_ns": 0.0, "timing_required": True, "cells": 50, "ff": 8,
                              "area_um2": 99.0, "ppa_variant": "buf", "ppa_netlist_path": str(gone),
                              "ppa_netlist_sha256": "b" * 64}
        monkeypatch.setattr(pg, "_evaluate_ppa_gate", fake_ppa)
        resp = await server._start_module_build(server.BuildModuleRequest(module="tiny"), entry="build_module")
        bid = resp["build_id"]
        row = await _settle(db, bid)
        assert row["status"] == "parked", row
        park = (await server.build_state(bid))["interrupts"][0]["payload"]
        assert park["type"] == "ppa_gate_unmeasurable" and "timing netlist unavailable" in park["outer_agent_guidance"]
        await server.build_resume(server.BuildResumeRequest(build_id=bid, action="abort"))
        row = await _settle(db, bid, until=B.BUILD_TERMINAL)
        assert row["status"] in ("failed", "aborted") and db.result("tiny", "best") is None
        ppa = Scoreboard(tmp_path).rows_for_build(bid)["ppa"][-1]
        assert ppa["wns_ns"] is None                     # no timing is recorded for an unpublishable netlist


class TestRefusals:
    async def test_unbound_and_invalid_refusals(self, daemon, tmp_path, monkeypatch):
        server, db, lc, ids = daemon
        _install(monkeypatch, tmp_path, Tools(tmp_path, throughput=[3.0], area=[80.0]))
        for v in db.verifiers(item_id=ids["area"]):
            db.remove_verifier(v["id"])
        resp = await server._start_module_build(server.BuildModuleRequest(module="tiny"), entry="build_module")
        body = json.loads(resp.body)
        assert resp.status_code == 409 and body["error"] == "MODULE_NOT_READY"
        assert any(b["code"] == "FRD_TARGET_UNBOUND" and b["ids"] == [ids["area"]] for b in body["blocked_by"])
        assert B.builds_for(db, "tiny") == []                 # refused before anything was recorded
        resp = await server._start_module_build(server.BuildModuleRequest(module="tiny", max_attempts=0),
                                                entry="build_module")
        assert json.loads(resp.body)["error"] == "BAD_MAX_ATTEMPTS"

    def test_default_attempts_allow_ordinary_retries(self):
        """The default keeps two retries; an explicit 1 is accepted (no retry)."""
        from orchestrator import module_build as MB
        from orchestrator.daemon import server
        assert server.BuildModuleRequest(module="x").max_attempts == MB.DEFAULT_MAX_ATTEMPTS == 3


def test_build_targets_cli_reports_the_allocation(tmp_path, monkeypatch):
    import subprocess
    import sys
    db = ready_project(tmp_path, monkeypatch)
    ids = add_module_targets(db, tmp_path)
    cli = Path(__file__).resolve().parents[2] / "bin" / "coresmith"
    env = {**__import__("os").environ, "CORESMITH_PROJECT_ROOT": str(tmp_path)}
    p = subprocess.run([sys.executable, str(cli), "build", "targets", "tiny", "--project-root", str(tmp_path),
                        "--json"], capture_output=True, text=True, env=env, timeout=120)
    doc = json.loads(p.stdout)
    assert {t["id"] for t in doc["targets"]} == {ids["perf"], ids["area"]}
    assert doc["acceptance"]["path"] == ids["tb"] and doc["deferred"] == ["PERF-001"]
    for v in db.verifiers(item_id=ids["perf"]):
        db.remove_verifier(v["id"])
    p = subprocess.run([sys.executable, str(cli), "build", "targets", "tiny", "--project-root", str(tmp_path),
                        "--json"], capture_output=True, text=True, env=env, timeout=120)
    assert p.returncode == 2 and json.loads(p.stdout)["problems"][0]["code"] == "FRD_TARGET_UNBOUND"
    assert st.module_ready(db, tmp_path, "tiny")
