"""WP-68: reporting and frontend completion preserve the adopted candidate."""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from orchestrator.harness import top_module as tm
from orchestrator.langgraph import backend_graph as bg
from orchestrator.langgraph import final_report as fr
from orchestrator.langgraph import integration_helpers as ih
from orchestrator.langgraph import pipeline_graph as pg


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.delenv("CORESMITH_TOP_MODULE", raising=False)
    monkeypatch.setenv("CORESMITH_UARCH_SINGLE_CONTEXT", "0")
    (tmp_path / "inputs").mkdir()
    (tmp_path / "inputs/task.yaml").write_text("top: chip_top\nchassis: none\n")
    top = tmp_path / "rtl/integration/top.v"
    top.parent.mkdir(parents=True)
    top.write_text("module chip_top(input a, output y); leaf u(a,y); endmodule\n")
    leaf = tmp_path / "rtl/leaf.v"
    leaf.write_text("module leaf(input a, output y); assign y=a; endmodule\n")
    # Tool-free hierarchy evidence; adoption, hashing, records and consumers are real.
    monkeypatch.setattr("orchestrator.harness.hierarchy.elaborate_hierarchy",
                        lambda *a, **k: {"leaf", "user_project_wrapper_pads"})
    monkeypatch.setattr(pg, "lint_top_level", Mock(return_value={"clean": True}))
    monkeypatch.setattr(pg, "load_architecture_connections", lambda *a: ([], "Friendly Design"))
    monkeypatch.setattr(ih, "load_architecture_connections", lambda *a: ([], "Friendly Design"))
    monkeypatch.setattr(pg, "discover_block_rtl", lambda *a: {"leaf": str(leaf)})
    monkeypatch.setattr("orchestrator.langchain.agents.integration_lead.IntegrationLeadAgent",
                        Mock(side_effect=AssertionError("single-block adoption called an agent")))
    monkeypatch.setattr(pg, "_resolve_interrupt", AsyncMock(side_effect=AssertionError("unexpected park")))
    rec = tm.write_candidate_receipt(
        tmp_path, "chip_top", str(top), {"leaf": str(leaf)},
        integration_result={"top_module": "chip_top", "top_rtl_path": str(top), "lint_clean": True},
    )
    state = {
        "project_root": str(tmp_path), "design_name": "Friendly Design",
        "block_queue": [{"name": "leaf", "rtl_target": "rtl/leaf.v"}],
        "completed_blocks": [{"name": "leaf", "success": True}],
        "pipeline_done": True,
        "integration_dv_result": {"passed": True},
        "validation_dv_result": {"passed": True},
    }
    return SimpleNamespace(root=tmp_path, top=top, leaf=leaf, rec=rec, state=state)


def _records(root):
    return {name: (root / ".coresmith" / name).read_bytes()
            for name in ("candidate.json", "integration_result.json")}


@pytest.mark.parametrize("aborted", [False, True])
async def test_terminal_reports_preserve_engine_records(project, aborted):
    p = project
    before = _records(p.root)
    state = {**p.state, "pipeline_aborted": aborted, "pipeline_done": not aborted}
    for _ in range(2):
        result = await pg.final_report_node(state)
        assert result.get("final_report"), result
        assert _records(p.root) == before
        assert tm.validated_candidate(p.root) == p.rec


async def test_report_shows_current_manifest_identity(project):
    p = project
    # Stale checkpoint hints must not outrank the persisted adoption.
    result = await pg.final_report_node({
        **p.state, "integration_result": {"top_module": "old_top", "candidate_sha": "old_sha"},
    })
    report = json.loads((p.root / "final_report.json").read_text())
    assert report == result["final_report"]
    assert report["top_module"] == p.rec["top_module"]
    assert report["top_rtl_path"] == p.rec["top_rtl_path"]
    assert report["candidate_sha"] == p.rec["candidate_sha"]
    assert report["candidate_error"] == ""
    markdown = (p.root / "final_report.md").read_text()
    for value in (p.rec["top_module"], p.rec["top_rtl_path"], p.rec["candidate_sha"]):
        assert value in markdown


async def test_report_failure_preserves_candidate(project, monkeypatch):
    before = _records(project.root)
    monkeypatch.setattr(fr, "build_final_report", Mock(side_effect=OSError("report unavailable")))
    assert await pg.final_report_node(project.state) == {}
    assert _records(project.root) == before
    assert tm.validated_candidate(project.root) == project.rec


async def test_backend_resolves_exact_manifest_after_report(project):
    p = project
    (p.top.parent / "unrecorded.v").write_text("module unrecorded(); endmodule\n")
    await pg.final_report_node(p.state)
    out = await bg.init_design_node({"project_root": str(p.root), "design_name": "old_top"})
    assert out["previous_error"] == "", out
    assert out["phase"] == "init"
    assert bg.route_after_init_design(out) == "flat_top_synthesis"
    assert out["design_name"] == p.rec["top_module"]
    assert out["integration_top_path"] == str(p.top)
    assert set(out["block_rtl_paths"].values()) == {str(p.leaf)}
    assert tm.validated_candidate(p.root) == p.rec


@pytest.mark.parametrize("entry", ["auto_backend", "backend_start"])
async def test_frontend_adoption_report_then_backend_launch(project, monkeypatch, entry):
    from orchestrator import mcp_server as mcp
    from orchestrator.langgraph import pipeline_helpers as ph

    p = project
    (p.root / "inputs/task.yaml").write_text("top: leaf\nchassis: none\n")
    integration = await pg.integration_check_node(p.state)
    assert integration["integration_result"]["lint_clean"] is True
    rec = tm.validated_candidate(p.root)
    assert rec["top_module"] == "leaf"
    assert rec["candidate_sha"] != p.rec["candidate_sha"]
    before = _records(p.root)
    await pg.final_report_node({**p.state, **integration})

    (p.root / ".coresmith/block_specs.json").write_text(json.dumps(p.state["block_queue"]))
    netlist = p.root / "syn/output/leaf/leaf_netlist.v"
    netlist.parent.mkdir(parents=True)
    netlist.write_text(p.leaf.read_text())
    monkeypatch.setattr(mcp, "_project_root", lambda: str(p.root))
    monkeypatch.setattr(mcp, "_pipeline", SimpleNamespace(graph=None))
    monkeypatch.setattr(ph, "preflight_check", lambda *a: {"ok": True})
    captured = {}

    async def start(initial_state, config):
        # Execute the real backend entry synchronously; stop before the synthesis agent.
        captured.update(await bg.init_design_node(initial_state))

    monkeypatch.setattr(mcp, "_backend", SimpleNamespace(
        status="idle", thread_id="backend", error_message="",
        ensure_graph=AsyncMock(), reset_for_new_run=AsyncMock(), safe_start=start,
    ))
    if entry == "auto_backend":
        result = await mcp.launch_backend(stop_after_gate_sim=True)
    else:
        result = json.loads(await mcp.start_backend())
    assert result["started"], result
    assert captured["previous_error"] == "", captured
    assert bg.route_after_init_design(captured) == "flat_top_synthesis"
    assert captured["design_name"] == rec["top_module"]
    assert captured["integration_top_path"] == rec["top_rtl_path"]
    assert _records(p.root) == before


@pytest.fixture
def assembled_project(project):
    p = project
    p.top = p.top.with_name("user_project_wrapper.v")
    p.top.write_text("module user_project_wrapper(input a, output y); leaf u(a,y); "
                     "user_project_wrapper_pads pads(); endmodule\n")
    p.pads = p.top.with_name("user_project_wrapper_pads.v")
    p.pads.write_text("module user_project_wrapper_pads(); endmodule\n")
    (p.root / "inputs/task.yaml").write_text("chassis: none\n")
    p.rec = tm.write_candidate_receipt(
        p.root, "user_project_wrapper", str(p.top),
        {"leaf": str(p.leaf), "user_project_wrapper_pads": str(p.pads)},
        integration_result={"caravel_wrapper_assembled": True,
                            "top_module": "user_project_wrapper", "top_rtl_path": str(p.top)},
    )
    return p


def test_retirement_protects_current_candidate_sources(assembled_project):
    p = assembled_project
    assert pg._retire_derived_integration_artifacts(str(p.root)) == []
    assert tm.validated_candidate(p.root) == p.rec


@pytest.mark.parametrize("stale", [False, True])
async def test_targeted_tier_reentry_does_not_retire_before_integration(assembled_project, stale):
    p = assembled_project
    if stale:
        p.leaf.write_text(p.leaf.read_text() + "// revised leaf\n")
    before = _records(p.root)
    await pg.init_tier_node({**p.state, "revise_blocks": {"leaf": True}})
    assert p.top.exists() and p.pads.exists()
    assert _records(p.root) == before
    if not stale:
        assert tm.validated_candidate(p.root) == p.rec


async def test_new_integration_retires_superseded_assembly(assembled_project):
    p = assembled_project
    p.leaf.write_text(p.leaf.read_text() + "// revised leaf\n")
    (p.root / "inputs/task.yaml").write_text("top: leaf\nchassis: none\n")
    result = await pg.integration_check_node(p.state)
    assert result["integration_result"]["lint_clean"] is True
    assert not p.top.exists() and not p.pads.exists()
    retired = {path.name for path in (p.top.parent / "_stale").glob("*/*.v")}
    assert retired == {p.top.name, p.pads.name}
    rec = tm.validated_candidate(p.root)
    assert rec["top_rtl_path"] == str(p.leaf)
    assert rec["candidate_sha"] != p.rec["candidate_sha"]


async def test_stale_manifest_remains_rejected_after_reporting(project):
    p = project
    p.leaf.write_text(p.leaf.read_text() + "// revised leaf\n")
    before = _records(p.root)
    result = await pg.final_report_node(p.state)
    assert result["final_report"]["top_module"] is None
    assert result["final_report"].get("candidate_sha") is None
    assert "stale" in result["final_report"]["candidate_error"]
    assert _records(p.root) == before
    out = await bg.init_design_node({"project_root": str(p.root)})
    assert out["phase"] == "candidate"
    assert "stale" in out["previous_error"]
    assert bg.route_after_init_design(out) == "ask_human"
