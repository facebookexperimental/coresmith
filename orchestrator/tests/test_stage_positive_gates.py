# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""The stages past ``blocks`` complete only on the graph's own recorded
gates for the current composition -- every block's current build plus the
integrated top's bytes, named by a digest on each row: an elaborated
real-RTL top plus a passing integration DV row (``integration``), a passing
validation DV row after it (``acceptance``), a backend signoff PPA row after
that (``backend``). Each gate reads the LATEST verdict of its scope for the
current run: a later failure outranks an earlier pass; failed, skipped,
agent-sourced, other-run, other-composition and stale rows are rejected;
power that was never measured is recorded as unavailable and is not the
signoff."""
from __future__ import annotations

import shutil
import sqlite3

import pytest

from orchestrator.state_store import builds as B
from orchestrator.state_store import stages as st
from orchestrator.state_store.store import Scoreboard
from orchestrator.tests.build_fixtures import ready_project
from orchestrator.tests.test_build_identity import _complete_build

_REQUIRES_YOSYS = pytest.mark.skipif(not shutil.which("yosys"), reason="requires yosys (hierarchy elaboration)")


def _codes(blockers):
    return {b["code"] for b in blockers}


def _composed(tmp_path, monkeypatch):
    db = ready_project(tmp_path, monkeypatch, stage="blocks", with_rtl=True)
    for m in ("tiny", "sink"):
        _complete_build(db, tmp_path, m)
    db.add_check("FUNC-001", "block_dv", "pass")
    db.add_check("PERF-001", "block_dv", None, value=50.0)
    res = st.advance(db, tmp_path)
    assert res["advanced"], res
    assert st.current(db) == "integration"
    (tmp_path / "rtl" / "soc.v").write_text("module soc(input clk, input rst_n); tiny u0(.clk(clk), .rst_n(rst_n), .m_out()); endmodule\n")
    return db


def _snapshot(db, **over):
    snap = {"tier": "1", "top": "soc", "rtl_path": "rtl/soc.v", "real_blocks": ["tiny", "sink"], "stub_blocks": [],
            "wires": 1, "boundary_ports": 2, "elaborated": True, "boundary": ["clk", "rst_n"]}
    snap.update(over)
    db.add_integration_snapshot(snap)


def _comp(db, root):
    return B.composition_identity(db, root)["sha256"]


def _backdate(root, table, ts):
    con = sqlite3.connect(str(root / ".coresmith" / "project.sqlite"))
    con.execute(f"UPDATE {table} SET ts=? WHERE id=(SELECT MAX(id) FROM {table})", (ts,))
    con.commit()
    con.close()


def test_integration_needs_real_top_and_a_passing_chip_row(tmp_path, monkeypatch):
    db = _composed(tmp_path, monkeypatch)
    codes = _codes(st.entry(db, tmp_path, "integration"))
    assert "INTEGRATION_STUBS" in codes and "INTEGRATION_DV_MISSING" in codes
    _snapshot(db)
    assert _codes(st.entry(db, tmp_path, "integration")) == {"INTEGRATION_DV_MISSING"}
    sb = Scoreboard(tmp_path)
    rid, comp = db.run_id(), _comp(db, tmp_path)
    # rejected evidence: failed, skipped, agent-sourced, another run, another composition, no composition
    sb.record_dv(block="soc", scope="chip", source="gate", passed=False, run_id=rid, composition_sha=comp)
    sb.record_dv(block="soc", scope="chip", source="gate", passed=True, skipped=True, run_id=rid, composition_sha=comp)
    sb.record_dv(block="soc", scope="chip", source="agent", passed=True, run_id=rid, composition_sha=comp)
    sb.record_dv(block="soc", scope="chip", source="gate", passed=True, run_id="run-other", composition_sha=comp)
    sb.record_dv(block="soc", scope="chip", source="gate", passed=True, run_id=rid, composition_sha="other")
    sb.record_dv(block="soc", scope="chip", source="gate", passed=True, run_id=rid)
    assert _codes(st.entry(db, tmp_path, "integration")) == {"INTEGRATION_DV_MISSING"}
    sb.record_dv(block="soc", scope="chip", source="gate", passed=True, run_id=rid, composition_sha=comp)
    assert st.entry(db, tmp_path, "integration") == []
    # the LATEST verdict decides: a later failure of the same scope negates the earlier pass
    sb.record_dv(block="soc", scope="chip", source="gate", passed=False, run_id=rid, composition_sha=comp)
    assert _codes(st.entry(db, tmp_path, "integration")) == {"INTEGRATION_DV_MISSING"}
    sb.record_dv(block="soc", scope="chip", source="gate", passed=True, run_id=rid, composition_sha=comp)
    assert st.advance(db, tmp_path)["advanced"] and st.current(db) == "acceptance"


@_REQUIRES_YOSYS
def test_an_adopted_candidate_is_the_integration_elaboration_evidence(tmp_path, monkeypatch):
    """The canonical adoption (``write_candidate_receipt``: a real hierarchy
    elaboration of the actual top) is the elaboration evidence when a receipt
    exists: a current valid candidate that instantiates every block passes
    the integration exit without a second shell snapshot; an edited
    candidate is a problem in itself; a candidate that leaves a registered
    block out is not integrated; the shell-only project is unchanged."""
    from orchestrator.harness.top_module import validated_candidate, write_candidate_receipt
    db = _composed(tmp_path, monkeypatch)
    assert "INTEGRATION_STUBS" in _codes(st.entry(db, tmp_path, "integration"))       # the fixture's stub shell
    top = tmp_path / "rtl" / "adopted_soc.v"
    top.write_text("module soc(input clk, input rst_n, output [7:0] last); wire [7:0] data; "
                   "tiny t(.clk(clk), .rst_n(rst_n), .m_out(data)); "
                   "sink s(.clk(clk), .rst_n(rst_n), .s_in(data), .last(last)); endmodule\n")
    rtls = {m: str(tmp_path / "rtl" / f"{m}.v") for m in ("tiny", "sink")}
    receipt = write_candidate_receipt(tmp_path, "soc", str(top), rtls, expected_blocks=["tiny", "sink"])
    assert validated_candidate(tmp_path) == receipt and sorted(receipt["elaborated_cells"]) == ["sink", "tiny"]
    assert _codes(st.entry(db, tmp_path, "integration")) == {"INTEGRATION_DV_MISSING"}      # no stubs: the receipt decides
    comp = B.composition_identity(db, tmp_path, top_rtl_path=str(top))
    assert comp["top_rtl_path"] == str(top) and B.composition_status(db, tmp_path, comp["sha256"])["ok"]
    Scoreboard(tmp_path).record_dv(block="soc", scope="chip", source="gate", passed=True, run_id=db.run_id(),
                                   composition_sha=comp["sha256"])
    assert st.entry(db, tmp_path, "integration") == []
    # the adopted top edited after adoption: the receipt no longer validates and the verdict is gone
    original = top.read_text()
    top.write_text(original + "// changed after adoption\n")
    blockers = st.entry(db, tmp_path, "integration")
    assert _codes(blockers) == {"INTEGRATION_CANDIDATE_STALE", "INTEGRATION_DV_MISSING"}
    assert any("no longer validates" in b["text"] for b in blockers)
    top.write_text(original)
    assert st.entry(db, tmp_path, "integration") == []
    # a candidate that leaves a registered block out is not an integration of this composition
    partial = tmp_path / "rtl" / "partial_soc.v"
    partial.write_text("module soc(input clk, input rst_n, output [7:0] last); "
                       "tiny t(.clk(clk), .rst_n(rst_n), .m_out(last)); endmodule\n")
    write_candidate_receipt(tmp_path, "soc", str(partial), {"tiny": rtls["tiny"]}, expected_blocks=["tiny"])
    blockers = st.entry(db, tmp_path, "integration")
    stubs = [b for b in blockers if b["code"] == "INTEGRATION_STUBS"]
    assert stubs and stubs[0]["ids"] == ["sink"] and "INTEGRATION_DV_MISSING" in _codes(blockers)


@_REQUIRES_YOSYS
def test_adopted_hierarchy_presence_resolves_modules_through_their_target_binding(tmp_path, monkeypatch):
    """A registered module's HDL top is what its existing target binding
    names, and the adopted root is part of the elaborated hierarchy: a
    module bound to another top name that IS the adopted leaf root, or a
    descendant under its bound name, is present; a registered module whose
    bound top is nowhere in the hierarchy is still absent."""
    from orchestrator.harness.targets import bind
    from orchestrator.harness.top_module import write_candidate_receipt
    from orchestrator.state_store.project_db import open_project
    monkeypatch.setenv("CORESMITH_PROJECT_ROOT", str(tmp_path))
    (tmp_path / "rtl").mkdir()
    db = open_project(tmp_path)
    db.import_block_diagram({"blocks": [{"name": "cpu", "tier": 0, "rtl_target": "rtl/core.v"}], "connections": []})
    core = tmp_path / "rtl" / "core.v"
    core.write_text("module my_cpu(input a, output z); assign z = a; endmodule\n")
    bind(tmp_path, "cpu", {"top": "my_cpu", "sources": ["rtl/core.v"]})
    # the leaf root: the receipt's elaborated cells list descendants only
    rec = write_candidate_receipt(tmp_path, "my_cpu", str(core), [], expected_blocks=[])
    assert rec["elaborated_cells"] == []
    ev, problem = st._elaboration(db, tmp_path)
    assert problem is None and ev["source"] == "candidate" and ev["stub_blocks"] == [] and ev["top_module"] == "my_cpu"
    # a descendant under its bound name
    chip = tmp_path / "rtl" / "chip.v"
    chip.write_text("module chip(input a, output z); my_cpu u(.a(a), .z(z)); endmodule\n")
    write_candidate_receipt(tmp_path, "chip", str(chip), [str(core)], expected_blocks=["my_cpu"])
    ev, problem = st._elaboration(db, tmp_path)
    assert problem is None and ev["stub_blocks"] == []
    # a registered module whose bound HDL top is nowhere in the hierarchy is absent
    db.import_block_diagram({"blocks": [{"name": "cpu", "tier": 0, "rtl_target": "rtl/core.v"},
                                        {"name": "missing_module", "tier": 0, "rtl_target": "rtl/missing.v"}],
                             "connections": []})
    (tmp_path / "rtl" / "missing.v").write_text("module missing_hdl(input a, output z); assign z = a; endmodule\n")
    bind(tmp_path, "missing_module", {"top": "missing_hdl", "sources": ["rtl/missing.v"]})
    ev, problem = st._elaboration(db, tmp_path)
    assert problem is None and ev["stub_blocks"] == ["missing_module"]
    # an unbound module is looked up under its own name
    db.import_block_diagram({"blocks": [{"name": "cpu", "tier": 0, "rtl_target": "rtl/core.v"},
                                        {"name": "chip", "tier": 0, "rtl_target": "rtl/chip.v"}], "connections": []})
    ev, problem = st._elaboration(db, tmp_path)
    assert problem is None and ev["stub_blocks"] == []                      # chip is the adopted root itself


def test_an_edited_chip_top_invalidates_the_chip_verdicts_without_a_module_rebuild(tmp_path, monkeypatch):
    db = _composed(tmp_path, monkeypatch)
    _snapshot(db)
    sb = Scoreboard(tmp_path)
    sb.record_dv(block="soc", scope="chip", source="gate", passed=True, run_id=db.run_id(), composition_sha=_comp(db, tmp_path))
    assert st.entry(db, tmp_path, "integration") == []
    (tmp_path / "rtl" / "soc.v").write_text("module soc(input clk, input rst_n); // edited by hand\nendmodule\n")
    assert _codes(st.entry(db, tmp_path, "integration")) == {"INTEGRATION_DV_MISSING"}
    assert all(B.current_build_status(db, tmp_path, m)["ok"] for m in ("tiny", "sink"))   # the modules did not change


def test_acceptance_needs_validation_after_integration_and_rebuilds_invalidate(tmp_path, monkeypatch):
    db = _composed(tmp_path, monkeypatch)
    sb = Scoreboard(tmp_path)
    rid = db.run_id()
    _snapshot(db)
    comp = _comp(db, tmp_path)
    sb.record_dv(block="soc", scope="chip", source="gate", passed=True, run_id=rid, composition_sha=comp)
    assert st.advance(db, tmp_path)["advanced"]
    assert _codes(st.entry(db, tmp_path, "acceptance")) == {"VALIDATION_DV_MISSING"}
    sb.record_dv(block="soc", scope="validation", source="gate", passed=True, run_id=rid, composition_sha=comp)
    newest_build = st._composition(db, tmp_path)[1]
    chip_ts = max(r["ts"] for r in sb.dv_rows() if r["scope"] == "chip")
    assert newest_build < chip_ts
    _backdate(tmp_path, "dv_results", (newest_build + chip_ts) / 2)   # after the builds, before the integration DV
    assert _codes(st.entry(db, tmp_path, "acceptance")) == {"VALIDATION_DV_STALE"}
    sb.record_dv(block="soc", scope="validation", source="gate", passed=True, run_id=rid, composition_sha=comp)
    assert st.entry(db, tmp_path, "acceptance") == []
    assert st.advance(db, tmp_path)["advanced"] and st.current(db) == "backend"
    # a module is rebuilt: the composition changed, every later gate is re-established
    _complete_build(db, tmp_path, "tiny")
    assert "INTEGRATION_DV_MISSING" in _codes(st.entry(db, tmp_path, "integration"))
    assert {r["stage"] for r in st.regressed_stages(db, tmp_path)} >= {"integration", "acceptance"}
    assert st.status(db, tmp_path)["can_advance"] is False
    # a module without a current build blocks every later stage
    (tmp_path / "rtl" / "sink.v").write_text("module sink(); // edited\nendmodule\n")
    assert "COMPOSITION_NOT_BUILT" in _codes(st.entry(db, tmp_path, "acceptance"))


def test_backend_needs_a_signoff_row_and_power_is_context_not_signoff(tmp_path, monkeypatch):
    db = _composed(tmp_path, monkeypatch)
    sb = Scoreboard(tmp_path)
    rid = db.run_id()
    _snapshot(db)
    comp = _comp(db, tmp_path)
    sb.record_dv(block="soc", scope="chip", source="gate", passed=True, run_id=rid, composition_sha=comp)
    assert st.advance(db, tmp_path)["advanced"]
    sb.record_dv(block="soc", scope="validation", source="gate", passed=True, run_id=rid, composition_sha=comp)
    assert st.advance(db, tmp_path)["advanced"] and st.current(db) == "backend"
    assert _codes(st.entry(db, tmp_path, "backend")) == {"BACKEND_SIGNOFF_MISSING"}
    # a failed signoff is not a signoff; a signoff of another composition is not this one's
    assert sb.record_ppa(block="soc", probe="backend", source="gate", ppa_ok=False, stage="signoff", tool="openroad",
                         pdk="sky130A", clock_mhz=50.0, power_mw=None, power_basis="unavailable", run_id=rid,
                         composition_sha=comp)
    assert _codes(st.entry(db, tmp_path, "backend")) == {"BACKEND_SIGNOFF_MISSING"}
    assert sb.record_ppa(block="soc", probe="backend", source="gate", ppa_ok=True, stage="signoff", tool="openroad",
                         pdk="sky130A", clock_mhz=50.0, power_mw=None, power_basis="unavailable", run_id=rid,
                         composition_sha="other")
    assert _codes(st.entry(db, tmp_path, "backend")) == {"BACKEND_SIGNOFF_MISSING"}
    # a passing signoff with power unavailable completes the stage: power is context
    assert sb.record_ppa(block="soc", probe="backend", source="gate", ppa_ok=True, stage="signoff", tool="openroad",
                         pdk="sky130A", clock_mhz=50.0, workload="vectorless", power_mw=None, power_basis="unavailable",
                         run_id=rid, area_um2=1234.0, wns_ns=0.2, composition_sha=comp)
    assert st.entry(db, tmp_path, "backend") == []
    # ... until a later signoff of the same composition fails
    assert sb.record_ppa(block="soc", probe="backend", source="gate", ppa_ok=False, stage="signoff", tool="openroad",
                         pdk="sky130A", clock_mhz=50.0, power_mw=None, power_basis="unavailable", run_id=rid,
                         composition_sha=comp)
    assert _codes(st.entry(db, tmp_path, "backend")) == {"BACKEND_SIGNOFF_MISSING"}
    assert sb.record_ppa(block="soc", probe="backend", source="gate", ppa_ok=True, stage="signoff", tool="openroad",
                         pdk="sky130A", clock_mhz=50.0, power_mw=None, power_basis="unavailable", run_id=rid,
                         composition_sha=comp)
    assert st.advance(db, tmp_path)["advanced"] and st.current(db) == "backend"   # the last stage stays current


def test_ppa_rows_never_carry_a_power_figure_without_its_basis(tmp_path, monkeypatch):
    ready_project(tmp_path, monkeypatch)
    sb = Scoreboard(tmp_path)
    assert sb.record_ppa(block="x", probe="synth", power_mw=None, power_basis="unavailable")
    assert sb.record_ppa(block="x", probe="backend", power_mw=0.0, power_basis="estimated")     # a real zero keeps its basis
    assert not sb.record_ppa(block="x", probe="backend", power_mw=1.5)                            # a figure without a basis
    assert not sb.record_ppa(block="x", probe="backend", power_mw=1.5, power_basis="unavailable")
    assert not sb.record_ppa(block="x", probe="backend", power_mw=None, power_basis="measured")
    rows = sb.ppa_rows("x")
    assert [r["power_basis"] for r in rows] == ["unavailable", "estimated"]
    assert rows[1]["power_mw"] == 0.0


def test_backend_metrics_report_unavailable_power_as_none(tmp_path):
    from orchestrator.langgraph.backend_helpers import parse_openroad_reports, parse_pnr_stdout
    out = tmp_path / "pnr"
    out.mkdir()
    m = parse_openroad_reports(str(out))
    assert m["total_power_mw"] is None and m["power_basis"] == "unavailable"
    (out / "power.rpt").write_text("Group Internal Switching Leakage Total\nTotal  1.0e-03 2.0e-03 1.0e-06 3.001e-03 100.0%\n")
    m = parse_openroad_reports(str(out))
    assert m["total_power_mw"] == pytest.approx(3.001) and m["power_basis"] == "estimated"
    s = parse_pnr_stdout("Design area 955 1830 49%\n")
    assert s["total_power_mw"] is None and s["power_basis"] == "unavailable"


def test_the_graph_records_the_composition_on_its_chip_verdicts(tmp_path, monkeypatch):
    """The integration / validation DV nodes and the backend signoff stamp
    the rows with the composition they measured (``_composition_sha``): the
    builds, the top, the chip testbench with its local imports, the block
    RTL compiled, the ERS for validation. Editing any of them after the
    verdict re-opens the stage even though no module was rebuilt."""
    from orchestrator.langgraph import pipeline_graph as pg
    db = _composed(tmp_path, monkeypatch)
    _snapshot(db)
    assert pg._composition_sha(str(tmp_path), "rtl/soc.v") == _comp(db, tmp_path)
    assert pg._composition_sha(str(tmp_path / "nowhere")) is None          # no project database: nothing to name
    tb = tmp_path / "tb" / "cocotb" / "test_soc.py"
    (tmp_path / "tb" / "cocotb" / "soc_helpers.py").write_text("X = 1\n")
    tb.write_text("import cocotb\nimport soc_helpers\n")
    (tmp_path / ".coresmith" / "ers_spec.json").write_text('{"ers": {"tiny": []}}')
    blocks = {"tiny": str(tmp_path / "rtl" / "tiny.v"), "sink": str(tmp_path / "rtl" / "sink.v")}
    inputs = pg._chip_verdict_inputs(str(tmp_path), str(tb), blocks, ers=True)
    assert str(tmp_path / "tb" / "cocotb" / "soc_helpers.py") in inputs and inputs[-1].endswith("ers_spec.json")
    # a declared input that is missing when measured never names a valid composition
    missing_sha = pg._composition_sha(str(tmp_path), "rtl/soc.v", inputs + [str(tmp_path / "tb" / "nope.py")])
    assert not B.composition_status(db, tmp_path, missing_sha)["ok"]
    pg._record_dv_row(str(tmp_path), block="soc", scope="chip", source="gate", passed=True, run_id=db.run_id(),
                      composition_sha=pg._composition_sha(str(tmp_path), "rtl/soc.v", inputs))
    assert st.entry(db, tmp_path, "integration") == []
    (tmp_path / "tb" / "cocotb" / "soc_helpers.py").write_text("X = 2\n")   # a test helper the chip TB imports
    assert _codes(st.entry(db, tmp_path, "integration")) == {"INTEGRATION_DV_MISSING"}
    (tmp_path / "tb" / "cocotb" / "soc_helpers.py").write_text("X = 1\n")
    assert st.entry(db, tmp_path, "integration") == []
    (tmp_path / ".coresmith" / "ers_spec.json").write_text("{}")             # the ERS the validation ran against
    assert _codes(st.entry(db, tmp_path, "integration")) == {"INTEGRATION_DV_MISSING"}
