# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""FRD-linked module targets (``state_store.module_targets``) and the
receipts they are judged by (``langgraph.target_closure``): the allocation
a build binds and every readable refusal, the build identity it adds, the
evaluation of one candidate from tool receipts (pass / gap / unmeasured,
never a pass without a measurement), plus the blocking defects fixed with
it -- rotated event logs and checkpoint ancestors in the lineage reader,
the role-to-port-direction mapping of the fabric check, a tooling identity
that never stores an error as a version, and aborted builds' parks."""
from __future__ import annotations

import json
import math
import os
import stat
import sys
from pathlib import Path

import pytest

from orchestrator.state_store import builds as B
from orchestrator.state_store import module_targets as MT
from orchestrator.state_store import stages as st
from orchestrator.tests.build_fixtures import add_module_targets, complete_build, ready_project


def _codes(problems):
    return sorted({p["code"] for p in problems})


@pytest.fixture
def proj(tmp_path, monkeypatch):
    db = ready_project(tmp_path, monkeypatch)
    ids = add_module_targets(db, tmp_path)
    return db, tmp_path, ids


# ---------------------------------------------------------------- allocation
class TestAllocation:
    def test_owned_bounded_items_are_bound_with_their_measurement(self, proj):
        db, root, ids = proj
        a = MT.allocation(db, root, "tiny")
        assert a["problems"] == []
        by_id = {t["id"]: t for t in a["targets"]}
        assert set(by_id) == {ids["perf"], ids["area"]}
        perf = by_id[ids["perf"]]
        assert perf["required"] and perf["bound_max"] == 4.0 and perf["unit"] == "cycles/op"
        assert perf["derives_from"] == ["KPI-FPS-1"]
        assert perf["bindings"][0]["kind"] == "cocotb" and perf["bindings"][0]["entry"] == "test_throughput"
        assert perf["bindings"][0]["args"] == {"workload": "reference-burst"}
        assert by_id[ids["area"]]["bindings"][0] == {**by_id[ids["area"]]["bindings"][0], "kind": "eda",
                                                     "entry": "area_um2"}
        # PERF-001 is measured at chip scope: deferred, not a module target
        assert a["deferred"] == ["PERF-001"]
        assert [f["id"] for f in a["functional"]] == ["FUNC-001"]
        acc = a["acceptance"]
        assert acc["path"] == ids["tb"] and acc["sha256"] == B.file_sha256(root / ids["tb"])
        assert acc["files"][0]["entries"] == ["test_count", "test_throughput"] and acc["module"] == "test_tiny"
        assert acc["supplemental"] == "tb/cocotb/test_tiny_supplemental.py"
        assert a["digest"] and a["frd_revision"]

    def test_advisory_targets_do_not_gate(self, proj):
        db, root, ids = proj
        db.edit_item(ids["area"], priority="should_have")
        a = MT.allocation(db, root, "tiny")
        assert {t["id"]: t["required"] for t in a["targets"]}[ids["area"]] is False

    def test_invalid_bounds_and_units_refuse(self, proj):
        db, root, ids = proj
        db.edit_item(ids["perf"], bound_min=10.0)           # min > max
        db.edit_item(ids["area"], unit="furlongs")           # not an area unit
        probs = MT.allocation(db, root, "tiny")["problems"]
        assert _codes(probs) == ["FRD_TARGET_INVALID"]
        assert {p["where"] for p in probs} == {ids["perf"], ids["area"]}
        db.edit_item(ids["perf"], bound_min=None, bound_max=math.inf)
        db.edit_item(ids["area"], unit="mm2")
        probs = MT.allocation(db, root, "tiny")["problems"]
        assert [p["where"] for p in probs] == [ids["perf"]] and "not finite" in probs[0]["text"]

    def test_unbound_required_target_refuses_readably(self, proj):
        db, root, ids = proj
        for v in db.verifiers(item_id=ids["area"]):
            db.remove_verifier(v["id"])
        probs = MT.allocation(db, root, "tiny")["problems"]
        assert _codes(probs) == ["FRD_TARGET_UNBOUND"] and probs[0]["where"] == ids["area"]
        assert "frd verifier" in probs[0]["text"]
        # module readiness carries it: build module / stage next refuse
        assert "FRD_TARGET_UNBOUND" in {b["code"] for b in st.module_ready(db, root, "tiny")}

    def test_wrongly_owned_target_refuses(self, proj):
        db, root, ids = proj
        db.unlink_items(ids["area"], "block:*", "owned_by")
        db.link_items(ids["area"], "block:sink", "owned_by")   # its eda verifier still names tiny
        probs = MT.allocation(db, root, "tiny")["problems"]
        assert "FRD_TARGET_OWNER_MISMATCH" in _codes(probs)
        db.link_items(ids["area"], "block:tiny", "owned_by")   # two owners for one bounded target
        probs = MT.allocation(db, root, "tiny")["problems"]
        assert any(p["code"] == "FRD_TARGET_OWNER_MISMATCH" and "one owner" in p["text"] for p in probs)

    def test_retired_items_are_not_targets(self, proj):
        db, root, ids = proj
        db.retire_item(ids["area"])
        a = MT.allocation(db, root, "tiny")
        assert ids["area"] not in {t["id"] for t in a["targets"]} and a["problems"] == []

    def test_acceptance_testbench_problems(self, proj):
        db, root, ids = proj
        (root / ids["tb"]).write_text("import cocotb\n\n@cocotb.test()\nasync def test_count(dut):\n    pass\n")
        probs = MT.allocation(db, root, "tiny")["problems"]
        assert _codes(probs) == ["ACCEPTANCE_TEST_MISSING"] and "test_throughput" in probs[0]["text"]
        (root / ids["tb"]).unlink()
        assert _codes(MT.allocation(db, root, "tiny")["problems"]) == ["ACCEPTANCE_TB_MISSING"]

    def test_several_acceptance_files_run_under_their_own_module_names(self, proj):
        db, root, ids = proj
        (root / "tb" / "protocol").mkdir(parents=True)
        (root / "tb" / "protocol" / "test_proto.py").write_text(
            "import cocotb\n\n@cocotb.test()\nasync def test_handshake(dut):\n    pass\n")
        db.add_verifier("FUNC-001", "cocotb", path="tb/protocol/test_proto.py", entry="test_handshake", block="tiny")
        a = MT.allocation(db, root, "tiny")
        assert a["problems"] == []
        files = MT.acceptance_files(a["acceptance"])
        assert [(f["path"], f["module"]) for f in files] == [(ids["tb"], "test_tiny"),
                                                             ("tb/protocol/test_proto.py", "test_proto")]
        func = {b["entry"]: b["module"] for b in a["functional"][0]["bindings"]}
        assert func == {"test_count": "test_tiny", "test_handshake": "test_proto"}
        # two files that would run under one cocotb module name are ambiguous
        (root / "tb" / "other").mkdir()
        (root / "tb" / "other" / "test_proto.py").write_text(
            "import cocotb\n\n@cocotb.test()\nasync def test_x(dut):\n    pass\n")
        db.add_verifier("FUNC-001", "cocotb", path="tb/other/test_proto.py", entry="test_x", block="tiny")
        assert "ACCEPTANCE_TB_AMBIGUOUS" in _codes(MT.allocation(db, root, "tiny")["problems"])

    def test_eda_measurement_needs_a_bound(self, proj):
        db, root, ids = proj
        db.add_verifier("FUNC-001", "eda", entry="area_um2", block="tiny")
        assert "FRD_TARGET_INVALID" in _codes(MT.allocation(db, root, "tiny")["problems"])

    @pytest.mark.parametrize("kind,entry,args,why", [
        ("eda", "power_mw", {"clock_mhz": 100}, "unsupported measurement condition"),
        ("eda", "power_mw", {"activity": float("nan")}, "out of range"),
        ("eda", "area_um2", {"area_scope": "die"}, "not one of"),
        ("cocotb", "test_throughput", {"frequency": 2}, "not measurement conditions"),
    ])
    def test_unsupported_conditions_are_refused_not_ignored(self, proj, kind, entry, args, why):
        db, root, ids = proj
        item = ids["area"] if entry == "area_um2" else ids["perf"]
        if entry == "power_mw":
            db.edit_item(item, unit="mW")
        db.add_verifier(item, kind, path=ids["tb"] if kind == "cocotb" else "", entry=entry, block="tiny", args=args)
        probs = MT.allocation(db, root, "tiny")["problems"]
        assert any(p["code"] == "FRD_TARGET_INVALID" and why in p["text"] for p in probs), probs

    def test_power_conditions_are_distinct_measurements(self, proj):
        db, root, ids = proj
        db.edit_item(ids["perf"], unit="mW")
        for v in db.verifiers(item_id=ids["perf"]):
            db.remove_verifier(v["id"])
        db.add_verifier(ids["perf"], "eda", entry="power_mw", block="tiny", args={"activity": 0.2})
        db.add_verifier(ids["area"], "eda", entry="area_um2", block="tiny", args={"area_scope": "std_cell"})
        a = MT.allocation(db, root, "tiny")
        keys = sorted(b["measure"] for t in a["targets"] for b in t["bindings"] if b["kind"] == "eda")
        assert keys == ["area_um2@std_cell", "power_mw@activity=0.2,duty=0.5"]
        assert MT.measure_key({"entry": "power_mw", "args": {}}) == "power_mw@activity=0.1,duty=0.5"


# ---------------------------------------------------------------- identity
class TestIdentity:
    def test_target_and_acceptance_revisions_are_new_input_identity(self, proj):
        db, root, ids = proj
        stored = B.module_inputs(db, root, "tiny", target_clock_mhz=50.0)

        def axes():
            live = B.module_inputs(db, root, "tiny", target_clock_mhz=50.0)
            return sorted({r.split(":")[0] for r in B.stale_reasons(stored, live)})
        assert axes() == []
        # an unrelated FRD item does not stale the module
        db.upsert_item("frd", {"id": "PERF-OTHER-9", "kind": "PERF", "text": "x", "priority": "must_have",
                               "metric": "x", "bound_max": 1.0})
        db.link_items("PERF-OTHER-9", "block:sink", "owned_by")
        assert axes() == []
        db.edit_item(ids["perf"], bound_max=3.5)                     # an explicit FRD revision of a target
        assert "targets" in axes()
        db.edit_item(ids["perf"], bound_max=4.0)
        assert axes() == []
        v = db.verifiers(item_id=ids["perf"])[0]                    # the workload condition of a binding
        db.add_verifier(ids["perf"], "cocotb", path=v["path"], entry=v["entry"], block="tiny",
                        args={"workload": "other"})
        assert axes() == ["targets"]
        db.add_verifier(ids["perf"], "cocotb", path=v["path"], entry=v["entry"], block="tiny", args=v["args"])
        assert axes() == []
        (root / ids["tb"]).write_text((root / ids["tb"]).read_text() + "\n# edited oracle\n")
        assert axes() == ["acceptance"]

    def test_a_completed_build_goes_stale_after_an_frd_revision(self, tmp_path, monkeypatch):
        db = ready_project(tmp_path, monkeypatch, stage="blocks", with_rtl=True)
        ids = add_module_targets(db, tmp_path)
        bid = complete_build(db, tmp_path)
        assert B.current_build_status(db, tmp_path, "tiny")["ok"]
        db.edit_item(ids["area"], bound_max=50.0)
        cur = B.current_build_status(db, tmp_path, "tiny")
        assert not cur["ok"] and any(r.startswith("targets:") for r in cur["reasons"])
        assert B.get_build(db, bid)["status"] == "completed"           # history stays

    def test_builds_recorded_before_targets_existed_compare_as_empty(self, proj):
        db, root, ids = proj
        live = B.module_inputs(db, root, "tiny", target_clock_mhz=50.0)
        old = {k: v for k, v in live.items() if k not in ("targets", "acceptance")}
        assert {r.split(":")[0] for r in B.stale_reasons(old, live)} == {"targets", "acceptance"}
        for v in db.verifiers(block="tiny"):
            db.remove_verifier(v["id"])
        for iid in (ids["perf"], ids["area"]):
            db.retire_item(iid)
        live = B.module_inputs(db, root, "tiny", target_clock_mhz=50.0)
        assert live["targets"]["digest"] is None and live["acceptance"] is None
        old = {k: v for k, v in live.items() if k not in ("targets", "acceptance")}
        assert B.stale_reasons(old, live) == []          # nothing bound: an old record is not stale


# ---------------------------------------------------------------- evaluation
_LIB = 'library(x) {\n cell ("sky130_fd_sc_hd__dfxtp_1") { area : 20.0; }\n}\n'


def _row(value, *, item="PERF-TINY-1", test="test_throughput", module="test_tiny", unit="cycles/op"):
    return {"item": item, "value": value, "unit": unit, "test": test, "module": module}


def _sim(root, a, rows, tests=("test_count", "test_throughput"), failing=()):
    """A real simulation receipt: results.xml and measurements.jsonl written
    to the sim dir, captured with the acceptance files and an RTL source as
    the simulated inputs (what the graph does after a passing run)."""
    from orchestrator.langgraph import target_closure as TC
    sim = Path(root) / "sim_build" / "tiny"
    sim.mkdir(parents=True, exist_ok=True)
    cases = "".join(f'<testcase classname="test_tiny" name="{t}">' + ("<failure/>" if t in failing else "")
                    + "</testcase>" for t in tests)
    (sim / "results.xml").write_text(f"<testsuites><testsuite>{cases}</testsuite></testsuites>")
    m = sim / "measurements.jsonl"
    m.unlink(missing_ok=True)
    if rows:
        m.write_text("".join(json.dumps(r) + "\n" for r in rows))
    rtl = Path(root) / "rtl" / "tiny.v"
    rtl.parent.mkdir(parents=True, exist_ok=True)
    if not rtl.exists():
        rtl.write_text("module tiny; endmodule\n")
    inputs = [str(rtl)] + [f["abs_path"] for f in MT.acceptance_files(a["acceptance"])]
    return {"receipt": TC.capture_sim_receipt(sim, inputs=inputs, module="tiny", build_id="b-x", attempt=1)}


def _area(root, um2, *, scope="total"):
    """A real yosys-style area measurement: report + netlist + liberty."""
    from orchestrator.langgraph import target_closure as TC
    d = Path(root) / "syn" / "output" / "tiny"
    d.mkdir(parents=True, exist_ok=True)
    (d / "fake.lib").write_text(_LIB)
    (d / "tiny_report.txt").write_text(f"=== tiny ===\n   Chip area for module '\\tiny': {um2}\n")
    (d / "tiny_netlist.v").write_text("module tiny(clk);\n  input clk;\n  sky130_fd_sc_hd__dfxtp_1 _1_ (.CLK(clk));\n"
                                      "endmodule\n")
    m = TC.measure_area({"netlist_path": str(d / "tiny_netlist.v"), "report_path": str(d / "tiny_report.txt"),
                         "liberty_path": str(d / "fake.lib")}, area_scope=scope)
    return {f"area_um2@{scope}": m}


class TestEvaluate:
    def test_met_targets_are_feasible_from_receipts(self, proj):
        db, root, ids = proj
        a = MT.allocation(db, root, "tiny")
        ev = MT.evaluate(a, sim=_sim(root, a, [_row(3.0)]), eda=_area(root, 80.0))
        assert ev["outcome"] == "feasible" and ev["missed"] == [] and ev["unmeasured"] == []
        t = {x["id"]: x for x in ev["targets"]}
        assert t[ids["perf"]]["value"] == 3.0 and t[ids["perf"]]["gap"]["margin"] == pytest.approx(1.0)
        assert t[ids["area"]]["receipts"][0]["receipt"]["kind"] == MT.AREA_RECEIPT
        assert ev["functional"] == [{"id": "FUNC-001", "required": True, "status": "pass",
                                     "tests": {"test_count": "pass"}}]
        assert MT.candidate_receipt_problems(a, ev) == []

    def test_caller_supplied_verdicts_and_rows_are_ignored(self, proj):
        """The verdicts and measurements come from the receipt's own files."""
        db, root, ids = proj
        a = MT.allocation(db, root, "tiny")
        sim = _sim(root, a, [])
        sim["tests"] = {"test_count": "pass", "test_throughput": "pass"}
        sim["rows"] = [_row(3.0)]
        ev = MT.evaluate(a, sim=sim, eda=_area(root, 80.0))
        assert ev["outcome"] == "unmeasured" and ev["unmeasured"] == [ids["perf"]]

    def test_a_miss_carries_the_gap(self, proj):
        db, root, ids = proj
        a = MT.allocation(db, root, "tiny")
        ev = MT.evaluate(a, sim=_sim(root, a, [_row(6.0)]), eda=_area(root, 80.0))
        assert ev["outcome"] == "target_miss" and ev["missed"] == [ids["perf"]]
        gap = {x["id"]: x for x in ev["targets"]}[ids["perf"]]["gap"]
        assert gap["side"] == "max" and gap["over"] == pytest.approx(2.0) and gap["relative"] == pytest.approx(0.5)
        report = MT.gap_report("tiny", ev)
        assert "TARGET MISS" in report and "over by 2 (50.0%)" in report and ids["perf"] in report

    @pytest.mark.parametrize("rows,why", [
        ([], "no measurement"),
        ([_row(3.0, test="test_count")], "not by the bound test"),
        ([_row(3.0, module="test_tiny_supplemental")], "acceptance module"),
        ([_row(3.0, unit="cycles")], "unit"),
        ([_row(float("nan"))], "non-finite"),
    ])
    def test_missing_or_foreign_evidence_is_unmeasured_never_a_pass(self, proj, rows, why):
        db, root, ids = proj
        a = MT.allocation(db, root, "tiny")
        ev = MT.evaluate(a, sim=_sim(root, a, rows), eda=_area(root, 80.0))
        assert ev["outcome"] == "unmeasured" and ev["unmeasured"] == [ids["perf"]]
        assert why in " ".join({x["id"]: x for x in ev["targets"]}[ids["perf"]]["reasons"])

    def test_a_tampered_or_foreign_receipt_is_not_evidence(self, proj):
        db, root, ids = proj
        a = MT.allocation(db, root, "tiny")
        sim = _sim(root, a, [_row(3.0)])
        eda = _area(root, 80.0)
        (root / "sim_build" / "tiny" / "measurements.jsonl").write_text(json.dumps(_row(1.0)) + "\n")
        ev = MT.evaluate(a, sim=sim, eda=eda)
        assert ev["outcome"] == "unmeasured" and "does not match its recorded hash" in ev["sim_problems"][0]
        sim = _sim(root, a, [_row(3.0)])
        (root / ids["tb"]).write_text((root / ids["tb"]).read_text() + "\n# weakened\n")   # the oracle that ran
        ev = MT.evaluate(a, sim=sim, eda=eda)
        assert ev["outcome"] == "unmeasured" and any("changed after the simulation" in p for p in ev["sim_problems"])
        m = dict(eda["area_um2@total"], value=50.0)                                          # a doctored scalar
        assert "does not match the yosys report" in " ".join(MT.eda_receipt_problems(m, {"entry": "area_um2"}))

    def test_a_failing_acceptance_test_is_a_miss_and_unrun_is_unmeasured(self, proj):
        db, root, ids = proj
        a = MT.allocation(db, root, "tiny")
        ev = MT.evaluate(a, sim=_sim(root, a, [_row(3.0)], failing=("test_count",)), eda=_area(root, 80.0))
        assert ev["outcome"] == "target_miss" and ev["missed"] == ["FUNC-001"]
        ev = MT.evaluate(a, sim=_sim(root, a, [_row(3.0)], tests=("test_throughput",)), eda=_area(root, 80.0))
        assert ev["outcome"] == "unmeasured" and ev["unmeasured"] == ["FUNC-001"]

    def test_unmeasured_eda_and_missing_receipt(self, proj):
        db, root, ids = proj
        a = MT.allocation(db, root, "tiny")
        ev = MT.evaluate(a, sim=None, eda={"area_um2@total": {"error": "black-box cells"}})
        assert ev["outcome"] == "unmeasured" and set(ev["unmeasured"]) == {ids["perf"], ids["area"], "FUNC-001"}

    def test_area_converts_to_the_item_unit(self, proj):
        db, root, ids = proj
        db.edit_item(ids["area"], unit="mm2", bound_max=0.0001)
        a = MT.allocation(db, root, "tiny")
        ev = MT.evaluate(a, sim=_sim(root, a, [_row(3.0)]), eda=_area(root, 80.0))
        t = {x["id"]: x for x in ev["targets"]}[ids["area"]]
        assert t["value"] == pytest.approx(80e-6) and t["status"] == "pass"


    def test_brief_names_targets_methods_and_the_fixed_oracle(self, proj):
        db, root, ids = proj
        a = MT.allocation(db, root, "tiny")
        text = MT.brief(a)
        assert ids["perf"] in text and "REQUIRED" in text and "test_throughput" in text and "reference-burst" in text
        assert "READ-ONLY" in text and "test_tiny_supplemental.py" in text and "yosys stat -liberty" in text

    def test_advisory_unmeasurable_targets_never_refuse(self, proj):
        db, root, ids = proj
        db.edit_item(ids["area"], priority="should_have")
        a = MT.allocation(db, root, "tiny")
        assert MT.measurability(a, liberty_present=False, synth_generic=False, sta_problem="x") == []
        db.edit_item(ids["area"], priority="must_have")
        a = MT.allocation(db, root, "tiny")
        assert [p["code"] for p in MT.measurability(a, liberty_present=False, synth_generic=False,
                                                    sta_problem=None)] == ["FRD_TARGET_UNMEASURABLE"]


# ---------------------------------------------------------------- receipts
class TestReceipts:
    def test_measure_records_the_recording_test_module(self, tmp_path, monkeypatch):
        from orchestrator.harness import measure as M
        monkeypatch.setenv(M.ENV, str(tmp_path / "m.jsonl"))
        mod = type(sys)("test_widget")
        exec("from orchestrator.harness import measure\n"
             "def run():\n    return measure.record('PERF-W-1', 2.5, unit='cycles/op', test='test_rate')\n",
             mod.__dict__)
        row = mod.run()
        assert row["module"] == "test_widget"
        assert M.read(tmp_path / "m.jsonl")[0]["module"] == "test_widget"

    def test_sim_receipt_detects_changed_evidence(self, tmp_path):
        from orchestrator.langgraph import target_closure as TC
        sim = tmp_path / "sim"
        sim.mkdir()
        (sim / "results.xml").write_text('<testsuites><testsuite><testcase classname="test_tiny" name="test_count"/>'
                                         '</testsuite></testsuites>')
        (sim / "measurements.jsonl").write_text(json.dumps(_row(3.0)) + "\n")
        rtl = tmp_path / "tiny.v"
        rtl.write_text("module tiny; endmodule\n")
        rec = TC.capture_sim_receipt(sim, inputs=[str(rtl)], module="tiny", build_id="b-1", attempt=2)
        assert rec["kind"] == MT.SIM_RECEIPT and rec["attempt"] == 2 and MT.sim_receipt_problems(rec) == []
        rtl.write_text("module tiny; /* edited after the simulation */ endmodule\n")
        assert "changed after the simulation" in MT.sim_receipt_problems(rec)[0]
        assert TC.capture_sim_receipt(tmp_path, inputs=[], module="tiny")["error"]
        for bogus in (None, {}, {"note": "x"}, {"kind": MT.SIM_RECEIPT}):
            assert MT.sim_receipt_problems(bogus)

    def test_evidence_snapshots_survive_the_next_attempt(self, proj):
        """Each candidate keeps its own copies; a later attempt overwriting the
        working files does not change what the recorded candidate was judged
        by -- and a tampered copy is caught at publication."""
        from orchestrator.langgraph import target_closure as TC
        db, root, ids = proj
        a = MT.allocation(db, root, "tiny")
        sim, eda = _sim(root, a, [_row(3.0)]), _area(root, 80.0)
        rec, eda2, errors = TC.snapshot_evidence(root, "b-1", "a1", sim["receipt"], eda)
        assert errors == [] and "/candidates/a1/evidence/" in rec["results_xml"]
        assert "/candidates/a1/evidence/" in eda2["area_um2@total"]["receipt"]["report_path"]
        ev = MT.evaluate(a, sim={"receipt": rec}, eda=eda2)
        _area(root, 95.0)                                        # the next attempt overwrites the working report
        _sim(root, a, [_row(9.0)])
        assert ev["outcome"] == "feasible" and MT.candidate_receipt_problems(a, ev) == []
        Path(eda2["area_um2@total"]["receipt"]["report_path"]).write_text("Chip area for module '\\tiny': 1.0\n")
        assert any("does not match its recorded hash" in p for p in MT.candidate_receipt_problems(a, ev))

    def test_supplemental_tests_may_not_record_measurements(self):
        from orchestrator.langgraph import target_closure as TC
        assert TC.supplemental_problem("from orchestrator.harness import measure\nmeasure.record('X-1', 1)\n")
        assert TC.supplemental_problem("import cocotb\n@cocotb.test()\nasync def test_supp_a(dut):\n    pass\n") is None

    def test_area_counts_bound_macros_and_refuses_unknown_black_boxes(self, tmp_path, monkeypatch):
        from orchestrator.langgraph import macro_sta
        from orchestrator.langgraph import target_closure as TC
        lib = tmp_path / "x.lib"
        lib.write_text(_LIB)
        rpt = tmp_path / "tiny_report.txt"
        rpt.write_text("=== tiny ===\n   Chip area for module '\\tiny': 80.025600\n")
        net = tmp_path / "tiny_netlist.v"
        net.write_text("module tiny(clk);\n  input clk;\n  sky130_fd_sc_hd__dfxtp_1 _1_ (.CLK(clk));\nendmodule\n")
        res = {"netlist_path": str(net), "report_path": str(rpt), "liberty_path": str(lib)}
        m = TC.measure_area(res)
        assert m["value"] == pytest.approx(80.0256) and m["receipt"]["report_sha256"] == B.file_sha256(rpt)
        assert m["receipt"]["macro_um2"] == 0 and m["receipt"]["total_um2"] == pytest.approx(80.0256)
        # an unknown black box: no total; the std-cell subtotal only when the target asks for it
        net.write_text(net.read_text().replace("endmodule", "  vendor_ip_x u_ip (.clk(clk));\nendmodule"))
        assert "no known area" in TC.measure_area(res)["error"]
        assert TC.measure_area(res, area_scope="std_cell")["value"] == pytest.approx(80.0256)
        # a memory wrapper bound to a macro with a Liberty area is counted
        mlib = tmp_path / "sram.lib"
        mlib.write_text('library(sram) {\n cell ("sky130_sram_1kbyte_1rw1r_32x256_8") {\n  area : 1000.5;\n }\n}\n')
        net.write_text("module tiny(clk);\n  input clk;\n  sky130_fd_sc_hd__dfxtp_1 _1_ (.CLK(clk));\n"
                       "  cs_sram_1rw1r #(\n    .WIDTH(32),\n    .DEPTH(256)\n  ) u_mem (.clk(clk));\nendmodule\n")
        binding = macro_sta.MacroBinding(netlist="", libs=[str(mlib)], instances=1,
                                         bound=[("cs_sram_1rw1r", 32, 256, "sky130_sram_1kbyte_1rw1r_32x256_8")])
        monkeypatch.setattr(macro_sta, "bind_netlist_macros", lambda text, **k: binding)
        from orchestrator.langgraph import macro_registry
        monkeypatch.setattr(macro_registry, "discover_macros", lambda *a, **k: {})
        m = TC.measure_area(res)
        assert m["value"] == pytest.approx(80.0256 + 1000.5)
        assert m["receipt"]["macros"][0]["count"] == 1 and m["receipt"]["std_cell_um2"] == pytest.approx(80.0256)
        assert MT.eda_receipt_problems(m, {"entry": "area_um2"}) == []
        assert TC.measure_area(res, area_scope="std_cell")["value"] == pytest.approx(80.0256)
        assert "no liberty" in TC.measure_area({**res, "liberty_path": str(tmp_path / "none.lib")})["error"]

    def test_registered_macro_instances_are_counted_by_name(self, tmp_path, monkeypatch):
        """A cell that IS a registered macro (explicit identity, no geometry
        guess) contributes its Liberty area; an unregistered one does not."""
        from types import SimpleNamespace

        from orchestrator.langgraph import macro_registry
        from orchestrator.langgraph import target_closure as TC
        lib = tmp_path / "x.lib"
        lib.write_text(_LIB)
        mlib = tmp_path / "rom.lib"
        mlib.write_text('library(rom) {\n cell ("rom_1r_32x1024") {\n  area : 500.0;\n  pin (clk) { }\n }\n}\n')
        rpt = tmp_path / "r.txt"
        rpt.write_text("Chip area for module '\\tiny': 100.0\n")
        net = tmp_path / "n.v"
        net.write_text("module tiny(clk);\n input clk;\n rom_1r_32x1024 u_rom0 (.clk(clk));\n"
                       " rom_1r_32x1024 u_rom1 (.clk(clk));\nendmodule\n")
        res = {"netlist_path": str(net), "report_path": str(rpt), "liberty_path": str(lib)}
        monkeypatch.setattr(macro_registry, "discover_macros", lambda *a, **k: {})
        assert "not a registered macro" in TC.measure_area(res)["error"]
        monkeypatch.setattr(macro_registry, "discover_macros",
                            lambda *a, **k: {"rom_1r_32x1024": SimpleNamespace(lib=str(mlib))})
        m = TC.measure_area(res)
        assert m["value"] == pytest.approx(100.0 + 2 * 500.0) and m["receipt"]["macros"][0]["count"] == 2
        assert MT.eda_receipt_problems(m, {"entry": "area_um2"}) == []
        # its Liberty has no power data: it is not a power model
        assert not MT.liberty_cell_has_power(str(mlib), "rom_1r_32x1024")
        sdc = tmp_path / "t.sdc"
        sdc.write_text("create_clock -name clk -period 20.0 [get_ports clk]\n")
        bindir = tmp_path / "bin"
        bindir.mkdir()
        sta = bindir / "sta"
        sta.write_text("#!/bin/sh\necho 'Total 1.0e-03 1.0e-03 0.0e+00 2.0e-03 100.0%'\n")
        sta.chmod(sta.stat().st_mode | stat.S_IEXEC)
        monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}")
        p = TC.measure_power({**res, "sdc_path": str(sdc)}, "tiny", activity=0.1, duty=0.5,
                             report_path=str(tmp_path / "p.rpt"))
        assert "without power characterization" in p["error"]
        mlib.write_text(mlib.read_text().replace("pin (clk) { }", "leakage_power () { value : 1.0; }"))
        p = TC.measure_power({**res, "sdc_path": str(sdc)}, "tiny", activity=0.1, duty=0.5,
                             report_path=str(tmp_path / "p.rpt"))
        assert p["value"] == pytest.approx(2.0) and p["receipt"]["macros"][0]["name"] == "rom_1r_32x1024"
        # OpenSTA creating a black box is a failure, even with a Total row and exit 0
        sta.write_text("#!/bin/sh\necho 'Warning: Creating black box for u_x.'\n"
                       "echo 'Total 1.0e-03 1.0e-03 0.0e+00 2.0e-03 100.0%'\n")
        p = TC.measure_power({**res, "sdc_path": str(sdc)}, "tiny", activity=0.1, duty=0.5,
                             report_path=str(tmp_path / "p.rpt"))
        assert "Creating black box" in p["error"]

    def test_an_area_measured_on_another_netlist_than_the_timing_is_rejected(self, proj):
        """The receipt names the netlist the timing verdict was measured on; a
        cheaper netlist's area is not the timing netlist's area."""
        db, root, ids = proj
        a = MT.allocation(db, root, "tiny")
        eda = _area(root, 80.0)
        m = eda["area_um2@total"]
        assert MT.eda_receipt_problems(m, {"entry": "area_um2"}) == []
        borrowed = {**m, "receipt": {**m["receipt"], "timing_netlist_sha256": "f" * 64}}
        probs = MT.eda_receipt_problems(borrowed, {"entry": "area_um2"})
        assert any("timing verdict was measured on another netlist" in p for p in probs)
        ev = MT.evaluate(a, sim=_sim(root, a, [_row(3.0)]), eda={"area_um2@total": borrowed})
        assert ev["outcome"] == "unmeasured" and ids["area"] in ev["unmeasured"]
        same = {**m, "receipt": {**m["receipt"], "timing_netlist_sha256": m["receipt"]["netlist_sha256"]}}
        assert MT.eda_receipt_problems(same, {"entry": "area_um2"}) == []

    def test_required_measurement_files_must_exist(self, tmp_path):
        rpt = tmp_path / "r.txt"
        rpt.write_text("Chip area for module 'tiny': 1.0\n")
        rec = {"kind": MT.AREA_RECEIPT, "area_scope": "total", "report_path": str(rpt),
               "report_sha256": B.file_sha256(rpt), "netlist_path": str(tmp_path / "gone.v"), "netlist_sha256": None,
               "liberty": str(tmp_path / "gone.lib"), "liberty_sha256": None, "macros": [], "macros_unresolved": []}
        probs = MT.eda_receipt_problems({"value": 1.0, "receipt": rec}, {"entry": "area_um2"})
        assert any("netlist_path" in p for p in probs) and any("liberty" in p for p in probs)
        sim = tmp_path / "sim"
        sim.mkdir()
        (sim / "results.xml").write_text("<testsuites/>")
        from orchestrator.langgraph import target_closure as TC
        r = TC.capture_sim_receipt(sim, inputs=[str(tmp_path / "missing.v")], module="tiny")
        assert any("had no hash" in p for p in MT.sim_receipt_problems(r))

    def test_power_report_is_parsed_from_opensta_output(self):
        from orchestrator.langgraph.ppa_check import parse_power_report
        text = ("Group                  Internal  Switching    Leakage      Total\n"
                "                          Power      Power      Power      Power (Watts)\n"
                "Sequential             1.00e-04   2.00e-05   1.00e-09   1.20e-04  69.0%\n"
                "----------------------------------------------------------------\n"
                "Total                  1.41e-04   3.24e-05   7.52e-10   1.74e-04 100.0%\n")
        p = parse_power_report(text)
        assert p["power_mw"] == pytest.approx(0.174) and p["leakage_mw"] == pytest.approx(7.52e-7)
        assert parse_power_report("no power here") is None

    def test_power_runs_opensta_and_records_its_receipt(self, tmp_path, monkeypatch):
        """The estimate is what the STA binary printed for the netlist, the
        liberty, the SDC clock and the declared activity -- through a fake
        ``sta`` that checks its script and prints a report."""
        from orchestrator.langgraph import target_closure as TC
        bindir = tmp_path / "bin"
        bindir.mkdir()
        sta = bindir / "sta"
        sta.write_text("#!/bin/sh\n"
                       "script=\"$3\"\n"
                       "grep -q 'set_power_activity -input -activity 0.2 -duty 0.5' \"$script\" || exit 3\n"
                       "grep -q 'report_power' \"$script\" || exit 4\n"
                       "echo 'Total                  1.00e-03   1.00e-03   0.00e+00   2.00e-03 100.0%'\n")
        sta.chmod(sta.stat().st_mode | stat.S_IEXEC)
        monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}")
        lib = tmp_path / "x.lib"
        lib.write_text('library(x) { cell ("sky130_fd_sc_hd__dfxtp_1") { } }\n')
        net = tmp_path / "n.v"
        net.write_text("module tiny(clk);\n input clk;\n sky130_fd_sc_hd__dfxtp_1 _1_ (.CLK(clk));\nendmodule\n")
        sdc = tmp_path / "t.sdc"
        sdc.write_text("create_clock -name clk -period 20.0 [get_ports clk]\n")
        res = {"netlist_path": str(net), "report_path": "", "liberty_path": str(lib), "sdc_path": str(sdc)}
        rpt = str(tmp_path / "p.rpt")
        m = TC.measure_power(res, "tiny", activity=0.2, duty=0.5, report_path=rpt)
        assert m["value"] == pytest.approx(2.0)
        rec = m["receipt"]
        assert rec["kind"] == MT.POWER_RECEIPT and rec["basis"] == "estimated" and rec["activity"] == 0.2
        assert rec["clock_period_ns"] == 20.0 and rec["report_sha256"] == B.file_sha256(rpt) and rec["liberty_sha256"]
        binding = {"entry": "power_mw", "args": {"activity": 0.2}}
        assert MT.eda_receipt_problems(m, binding) == []
        # the receipt is for its own condition: another activity is not this measurement
        assert "activity" in " ".join(MT.eda_receipt_problems(m, {"entry": "power_mw", "args": {"activity": 0.5}}))
        # an Error line is a failure even with exit 0; a clockless SDC is refused before running
        sta.write_text("#!/bin/sh\necho 'Error: link_design failed'\n"
                       "echo 'Total                  1.00e-03   1.00e-03   0.00e+00   2.00e-03 100.0%'\nexit 0\n")
        assert "link_design failed" in TC.measure_power(res, "tiny", activity=0.2, duty=0.5, report_path=rpt)["error"]
        sdc.write_text("# no clock\n")
        assert "no clock" in TC.measure_power(res, "tiny", activity=0.2, duty=0.5, report_path=rpt)["error"]
        sdc.write_text("create_clock -name clk -period 20.0 [get_ports clk]\n")
        sta.write_text("#!/bin/sh\necho 'Error: no liberty' ; exit 1\n")
        assert "no valid Total power" in TC.measure_power(res, "tiny", activity=0.2, duty=0.5,
                                                          report_path=rpt)["error"]

    def test_power_without_opensta_is_a_readable_refusal(self, proj, monkeypatch):
        from orchestrator import module_build as MB
        from orchestrator.langgraph import pipeline_helpers as ph
        db, root, ids = proj
        add_module_targets(db, root, throughput=False, area=False, power=True, functional=False, tb=False)
        lib = root / "x.lib"
        lib.write_text("library(x) {}\n")
        monkeypatch.setattr(ph, "LIBERTY_FILE", lib)
        monkeypatch.setattr(ph, "_sta_problem", lambda: "OpenSTA not reachable: no `sta` on PATH")
        refusal = MB.target_refusal(db, root, "tiny")
        assert refusal.code == "FRD_TARGET_UNMEASURABLE" and refusal.blockers[0]["ids"] == ["PWR-TINY-1"]
        assert "OpenSTA" in refusal.blockers[0]["text"]
        monkeypatch.setattr(ph, "_sta_problem", lambda: None)
        assert MB.target_refusal(db, root, "tiny") is None


# ---------------------------------------------------------------- blocking defects fixed with it
class TestToolingIdentity:
    def _tool(self, path: Path, body: str) -> Path:
        path.write_text("#!/bin/sh\n" + body)
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
        return path

    def test_an_error_is_never_a_version_and_the_shim_resolves_to_real_opensta(self, tmp_path, monkeypatch):
        bindir = tmp_path / "bin"
        bindir.mkdir()
        shim = Path(__file__).resolve().parents[2] / "bin" / "sta"
        (bindir / "sta").symlink_to(shim)
        monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}")
        monkeypatch.setenv("CORESMITH_REAL_STA", "sta-real-missing")
        B._TOOL_VERSION_CACHE.clear()
        t = B._tool_version("sta", tmp_path)
        assert t["version"] is None and t["sha256"] is None and "CORESMITH_REAL_STA" in t["error"]
        real = self._tool(bindir / "sta-real", 'echo "2.6.0"\n')
        monkeypatch.setenv("CORESMITH_REAL_STA", "sta-real")
        t = B._tool_version("sta", tmp_path)
        assert t["path"] == str(real.resolve()) and t["version"] == "2.6.0" and t["sha256"] == B.file_sha256(real)
        bad = self._tool(bindir / "yosys", 'echo "segfault while loading" ; exit 2\n')
        t = B._tool_version("yosys", tmp_path)
        assert t["version"] is None and "exited 2" in t["error"] and t["sha256"] == B.file_sha256(bad)

    def test_unknown_is_never_a_match_and_another_binary_is_a_change(self):
        a = {"verilator": {"sha256": "aa", "version": "5.0"}, "yosys": {"sha256": "bb"}, "sta": {"sha256": "cc"}}
        legacy = {"verilator": {"path": "/x", "version": "Verilator 5.0"},
                  "sta": {"path": "/s", "version": "OpenSTA command 'sta-real' is unavailable; set CORESMITH_REAL_STA"},
                  "yosys": {"path": None, "version": "unavailable"}}
        stored = {"tooling": a}
        unobservable = B.stale_reasons(stored, {"tooling": {**a, "sta": {"sha256": None, "error": "not on PATH"}}})
        assert any("cannot be identified" in r and "sta" in r for r in unobservable)
        assert any("changed since the build (sta)" in r
                   for r in B.stale_reasons(stored, {"tooling": {**a, "sta": {"sha256": "dd"}}}))
        assert B.stale_reasons(stored, {"tooling": dict(a)}) == []
        assert B._same_tool(legacy["verilator"], {"version": "Verilator 5.1"}) is False
        assert B._same_tool(legacy["sta"], {"sha256": "cc", "version": "2.6.0"}) is None     # an error is no version
        assert B._same_tool(legacy["yosys"], {"sha256": "bb"}) is None
        assert B._same_tool({"sha256": "cc"}, {"sha256": "cc"}) is True

    def test_other_processes_identify_the_binaries_the_daemon_recorded(self, tmp_path, monkeypatch):
        """The tool-running process records the resolved paths; a process with
        another PATH identifies the same binaries (no spurious staleness), and
        a recorded binary that disappeared is unknown -- never a match."""
        daemon_bin, cli_bin = tmp_path / "daemon_bin", tmp_path / "cli_bin"
        daemon_bin.mkdir()
        cli_bin.mkdir()
        self._tool(daemon_bin / "yosys", 'echo "Yosys 0.65"\n')
        self._tool(cli_bin / "yosys", 'echo "Yosys 0.10 (another one on the caller PATH)"\n')
        B._TOOL_VERSION_CACHE.clear()
        monkeypatch.setenv("PATH", f"{daemon_bin}{os.pathsep}/usr/bin:/bin")
        monkeypatch.setenv("CORESMITH_TOOL_AUTHORITY", "1")
        (tmp_path / ".coresmith").mkdir()
        recorded = B.tooling_snapshot(tmp_path, target_clock_mhz=50.0, persist=True)
        assert recorded["yosys"]["version"] == "Yosys 0.65"
        assert B.recorded_tool_paths(tmp_path)["yosys"] == str((daemon_bin / "yosys").resolve())
        monkeypatch.delenv("CORESMITH_TOOL_AUTHORITY")
        monkeypatch.setenv("PATH", f"{cli_bin}{os.pathsep}/usr/bin:/bin")
        live = B.tooling_snapshot(tmp_path, target_clock_mhz=50.0)
        assert live["yosys"]["resolved_by"] == "recorded" and live["yosys"]["version"] == "Yosys 0.65"
        assert [r for r in B.stale_reasons({"tooling": recorded}, {"tooling": live}) if "yosys" in r] == []
        (daemon_bin / "yosys").unlink()
        live = B.tooling_snapshot(tmp_path, target_clock_mhz=50.0)
        assert live["yosys"]["sha256"] is None and "no longer exists" in live["yosys"]["error"]
        assert any("cannot be identified" in r for r in B.stale_reasons({"tooling": recorded}, {"tooling": live}))


class TestLineage:
    def test_rotated_event_logs_are_read(self, tmp_path):
        cdir = tmp_path / ".coresmith"
        cdir.mkdir()
        (cdir / "pipeline_events.20261008-124808.jsonl").write_text(
            json.dumps({"build_id": "b-1", "node": "Init Block"}) + "\n")
        (cdir / "pipeline_events.jsonl").write_text(json.dumps({"build_id": "b-1", "node": "Block Done"}) + "\n")
        ev = B._events_for_build(tmp_path, "b-1")
        assert ev["count"] == 2 and ev["nodes"] == ["Init Block", "Block Done"]
        assert ev["files"] == ["pipeline_events.20261008-124808.jsonl", "pipeline_events.jsonl"]

    def test_a_nested_completion_namespace_is_accepted_on_segment_boundaries(self):
        rec = {"", "process_block:9539"}
        assert B.namespace_recorded("process_block:9539|block_done:26ad", rec)
        assert B.namespace_recorded("process_block:9539", rec)
        assert not B.namespace_recorded("process_block:953", rec)              # not a segment boundary
        assert not B.namespace_recorded("process_block:1234|block_done:26ad", rec)
        assert not B.namespace_recorded("", rec)


class TestFabricDirection:
    def test_role_and_flow_map_to_verilog_direction(self):
        from orchestrator.langgraph.contract_conformance import port_direction
        assert port_direction("producer", "producer->consumer", "awvalid") == "output"
        assert port_direction("consumer", "producer->consumer", "awvalid") == "input"
        assert port_direction("consumer", "consumer->producer", "awready") == "output"
        assert port_direction("producer", "", "tready") == "input"
        assert port_direction("consumer", "", "tdata") == "input"

    def test_axi_fabric_ports_conform(self, tmp_path, monkeypatch):
        """An AXI edge master->fabric: the materialization check compares the
        generated Verilog directions with the fabric's ROLE-derived direction,
        so a correct crossbar no longer fails FABRIC_CONTRACT_MISMATCH."""
        from orchestrator.langgraph.contract_conformance import contract_port_rows, port_direction
        edge = {"edge_id": "cpu__m_axi__to__fab__s_cpu", "producer_block": "cpu", "producer_port": "m_axi",
                "consumer_block": "fab", "consumer_port": "s_cpu", "handshake_protocol": "axi_lite",
                "data_width_bits": 32, "bus_params": {"addr_width": 32}}
        (tmp_path / ".coresmith").mkdir()
        (tmp_path / ".coresmith" / "interface_contracts.json").write_text(json.dumps({"contracts": [edge]}))
        monkeypatch.setenv("CORESMITH_PROJECT_ROOT", str(tmp_path))
        rows = contract_port_rows(tmp_path, "fab")
        assert rows, "the AXI-Lite channel set expands to ports"
        dirs = {r["signal"]: port_direction(r["role"], r["dir"], r["signal"]) for r in rows}
        assert dirs["awvalid"] == "input" and dirs["awready"] == "output"
        assert dirs["rdata"] == "output" and dirs["rready"] == "input"
        assert all(r["dir"] in ("producer->consumer", "consumer->producer") for r in rows)


class TestAbortParks:
    async def test_abort_abandons_the_builds_open_parks(self, tmp_path, monkeypatch):
        from orchestrator import module_build as MB
        db = ready_project(tmp_path, monkeypatch)
        inputs = B.module_inputs(db, tmp_path, "tiny", target_clock_mhz=50.0)
        bid = B.new_build_id("tiny")
        B.record_dispatch(db, build_id=bid, module="tiny", entry="build_module", graph="build",
                          thread_id=f"build-{bid}", inputs=inputs)
        B.mark_status(db, bid, "parked", terminal=False)
        iid, _ = db.park_interrupt({"type": "ppa_gate_unmeasurable", "block_name": "tiny"}, graph="build",
                                   node="synthesize", block="tiny")
        other, _ = db.park_interrupt({"type": "x", "block_name": "sink"}, graph="build", node="n", block="sink")

        class _Idle:
            task = None
            thread_id = ""
        out = await MB.abort_build(db, _Idle(), bid, reason="superseded")
        assert out["aborted"] and out["abandoned_interrupts"] == [iid]
        assert db.interrupt(iid)["status"] == "abandoned" and db.interrupt(other)["status"] == "pending"
