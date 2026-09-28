"""Round-4 candidate counterexamples through real adoption and consumer paths."""
import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from orchestrator.harness import task_adapter as ta
from orchestrator.harness import top_module as tm
from orchestrator.langgraph import backend_graph as bg
from orchestrator.langgraph import integration_helpers as ih
from orchestrator.langgraph import pipeline_graph as pg


@pytest.fixture
def candidate(tmp_path, monkeypatch):
    monkeypatch.delenv("CORESMITH_TOP_MODULE", raising=False)
    monkeypatch.delenv("CORESMITH_TASK_ADAPTER", raising=False)
    monkeypatch.setenv("CORESMITH_ADAPTER_SANDBOX", "none")
    # F4 covers the real tool; these tests exercise manifest decisions in CI.
    monkeypatch.setattr("orchestrator.harness.hierarchy.elaborate_hierarchy", lambda *a, **k: {"leaf"})
    root = tmp_path
    (root / "inputs").mkdir()
    (root / "rtl").mkdir()
    top = root / "rtl/top.v"
    leaf = root / "rtl/leaf.v"
    top.write_text("module chip_top(); leaf u(); endmodule\n")
    leaf.write_text("module leaf(); endmodule\n")
    (root / "inputs/task.yaml").write_text("top: chip_top\n")
    (root / "inputs/task_adapter.py").write_text(
        'CASES=["one"]\ndef grade(c,w):\n return {"cases":{"one":{"ok":True}}}\n')
    rec = tm.write_candidate_receipt(root, "chip_top", str(top), {"leaf": str(leaf)})
    (root / ".coresmith/integration_result.json").write_text(json.dumps({
        "top_module": "chip_top", "top_rtl_path": str(top), "lint_clean": True}))
    return root, top, leaf, rec


def test_stale_leaf_cannot_be_revived_by_integration_record(candidate):
    root, top, leaf, rec = candidate
    leaf.write_text("module leaf(); wire changed; endmodule\n")
    assert not tm.receipt_is_current(rec)
    with pytest.raises(ValueError, match="stale"):
        tm.resolve_top(root)
    result = ta.run_task_adapter(str(root), str(top), {"leaf": str(leaf)})
    assert not result["passed"] and result["kind"] == "candidate_mismatch"


def test_extra_source_is_rejected(candidate):
    root, top, leaf, rec = candidate
    extra = root / "rtl/extra.v"
    extra.write_text("module extra(); endmodule")
    result = ta.run_task_adapter(str(root), str(top), {"extra": str(extra)})
    assert not result["passed"] and result["kind"] == "candidate_mismatch"


def test_missing_leaf_is_infrastructure_error(candidate):
    root, top, leaf, rec = candidate
    leaf.unlink()
    result = ta.run_task_adapter(str(root), str(top), {})
    assert not result["passed"] and result["kind"] == "infrastructure_error"


def test_task_top_change_invalidates_manifest(candidate):
    root, top, leaf, rec = candidate
    (root / "inputs/task.yaml").write_text("top: other_chip\n")
    with pytest.raises(ValueError, match="declares"):
        tm.resolve_top(root)


@pytest.mark.parametrize("dependency", ["readmemh", "readmemb", "include"])
def test_dependency_closure_is_bound(candidate, dependency):
    root, top, leaf, rec = candidate
    dep = root / "inputs/asset.dat"
    dep.write_text("01\n" if dependency != "include" else "wire included;\n")
    ref = ('`include "asset.dat"\n' if dependency == "include"
           else f'initial ${dependency}("asset.dat", rom);\n')
    top.write_text('module chip_top(); reg [7:0] rom[0:0];\n' + ref + 'leaf u(); endmodule\n')
    rec = tm.write_candidate_receipt(root, "chip_top", str(top), {"leaf": str(leaf)})
    assert tm.receipt_is_current(rec)
    dep.write_text("10\n" if dependency != "include" else "wire changed;\n")
    assert not tm.receipt_is_current(rec)
    with pytest.raises(ValueError):
        tm.resolve_top(root)


def test_configuration_is_explicit_and_hashed(candidate):
    root, top, leaf, rec = candidate
    assert rec.get("defines") == "none" and rec.get("parameters") == "none"
    changed = tm.write_candidate_receipt(root, "chip_top", str(top), {"leaf": str(leaf)}, defines=["FEATURE=1"])
    assert changed["candidate_sha"] != rec["candidate_sha"]


def test_adapter_uses_exact_recorded_sources_and_sha(candidate):
    root, top, leaf, rec = candidate
    result = ta.assemble_candidate(str(root), str(top), {})
    assert result["sources"] == rec["sources"]
    assert result["candidate_sha"] == rec["candidate_sha"]


def test_no_manifest_has_no_helper_first_fallback(candidate):
    root, top, leaf, rec = candidate
    (root / tm.RECEIPT_REL).unlink()
    top.write_text("module helper(); endmodule\nmodule chip_top(); endmodule\n")
    result = ta.run_task_adapter(str(root), str(top), {})
    assert not result["passed"] and result["oracle_incomplete"]
    with pytest.raises(ValueError):
        ih.chip_rtl_sources(str(top), {})


def test_backend_uses_manifest_without_discovery(candidate, monkeypatch):
    root, top, leaf, rec = candidate
    extras = root / "rtl/integration"
    extras.mkdir()
    (extras / "unrecorded.v").write_text("module unrecorded(); endmodule\n")
    monkeypatch.setattr(ih, "discover_block_rtl", lambda *a: pytest.fail("backend rediscovered RTL"))
    out = asyncio.run(bg.init_design_node({"project_root": str(root), "block_queue": [{"name": "leaf"}]}))
    assert out["integration_top_path"] == str(top)
    assert set(out["block_rtl_paths"].values()) == {str(leaf)}


def test_backend_missing_manifest_refuses_integration_record(candidate):
    root, top, leaf, rec = candidate
    (root / tm.RECEIPT_REL).unlink()
    out = asyncio.run(bg.init_design_node({"project_root": str(root)}))
    assert not out["integration_top_path"] and out["previous_error"]


def test_single_block_mismatch_never_publishes_success(tmp_path, monkeypatch):
    (tmp_path / "inputs").mkdir()
    (tmp_path / "inputs/task.yaml").write_text("top: declared_chip\n")
    leaf = tmp_path / "leaf.v"
    leaf.write_text("module leaf(input a, output y); assign y=a; endmodule\n")
    monkeypatch.setattr(pg, "_current_phase_completed", lambda *a: [{"name": "leaf", "success": True}])
    monkeypatch.setattr(pg, "load_architecture_connections", lambda *a: ([], "friendly_name"))
    monkeypatch.setattr(pg, "discover_block_rtl", lambda *a: {"leaf": str(leaf)})
    monkeypatch.setattr(pg, "lint_top_level", lambda *a, **k: {"clean": True})
    monkeypatch.setattr(pg, "_resolve_interrupt", AsyncMock(return_value={"action": "abort"}))
    out = asyncio.run(pg.integration_check_node({"project_root": str(tmp_path), "block_queue": [{"name": "leaf"}]}))
    assert out["integration_result"].get("aborted")
    assert not (tmp_path / ".coresmith/integration_result.json").exists()
    assert not (tmp_path / tm.RECEIPT_REL).exists()


def test_sources_cannot_change_during_adoption(candidate, monkeypatch):
    root, top, leaf, rec = candidate

    def changing_elaborator(*a, **k):
        leaf.write_text("module leaf(); wire changed_after_elaboration; endmodule")
        return {"leaf"}

    monkeypatch.setattr("orchestrator.harness.hierarchy.elaborate_hierarchy", changing_elaborator)
    with pytest.raises(ValueError, match="during"):
        tm.write_candidate_receipt(root, "chip_top", str(top), {"leaf": str(leaf)})
    assert not (root / tm.RECEIPT_REL).exists()


def test_literal_memory_asset_in_inputs_elaborates(tmp_path):
    import shutil
    if not shutil.which("yosys"):
        pytest.skip("requires yosys")
    (tmp_path / "inputs").mkdir()
    (tmp_path / "inputs/rom.mem").write_text("01\n")
    top = tmp_path / "top.v"
    top.write_text('module chip_top(output [7:0] y); reg [7:0] rom[0:0]; '
                   'initial $readmemh("rom.mem",rom); assign y=rom[0]; endmodule')
    rec = tm.write_candidate_receipt(tmp_path, "chip_top", str(top), {})
    assert tm.receipt_is_current(rec)


def test_implicit_synthesis_define_cannot_change_candidate_hierarchy(tmp_path):
    import shutil
    if not shutil.which("yosys"):
        pytest.skip("requires yosys")
    top = tmp_path / "top.v"
    top.write_text('module chip_top();\n`ifdef SYNTHESIS\nleaf u();\n`endif\nendmodule\n')
    leaf = tmp_path / "leaf.v"
    leaf.write_text('module leaf(); endmodule\n')
    with pytest.raises(ValueError, match="leaf"):
        tm.write_candidate_receipt(tmp_path, "chip_top", str(top), {"leaf": str(leaf)})


def test_simulation_only_region_is_not_hierarchy_evidence(tmp_path):
    # Q26(b): `final` (simulation-only, under `ifndef SYNTHESIS) used to be a
    # Yosys syntax error. It is now outside the evidence, and so is a block
    # instantiated only there (silicon never has it).
    import shutil
    if not shutil.which("yosys"):
        pytest.skip("requires yosys")
    leaf = tmp_path / "leaf.v"
    leaf.write_text('module leaf(); endmodule\nmodule probe(); endmodule\n')
    top = tmp_path / "top.v"
    top.write_text('module chip_top();\nleaf u();\n`ifndef SYNTHESIS\nprobe p();\n'
                   'final begin $display("done"); end\n`endif\nendmodule\n')
    rec = tm.write_candidate_receipt(tmp_path, "chip_top", str(top), {"leaf": str(leaf)})
    assert "leaf" in rec["elaborated_cells"] and "probe" not in rec["elaborated_cells"]
    with pytest.raises(ValueError, match="probe"):
        tm.write_candidate_receipt(tmp_path, "chip_top", str(top), {"leaf": str(leaf)},
                                   expected_blocks=["leaf", "probe"])


def test_synthesis_branches_are_blanked_line_for_line():
    from orchestrator.harness.hierarchy import _without_synthesis_branches as strip
    text = ('a\n`ifndef SYNTHESIS\nsim\n`ifdef X\nx\n`endif\n`else\nsyn\n`endif\n'
            '`ifdef X\nxx\n`elsif SYNTHESIS\ns\n`else\nsim2\n`endif\nb\n')
    out = strip(text)
    assert out.count("\n") == text.count("\n")
    assert [l for l in out.splitlines() if l] == ["a", "`ifdef X", "xx", "`else", "`endif", "b"]


def test_hierarchy_timeout_is_configurable(tmp_path, monkeypatch):
    # Q26(c): the full SoC needs ~259 s; the budget is env/multiplier driven,
    # and a timeout is an explicit infrastructure failure, never a skip.
    import subprocess
    from orchestrator.harness import hierarchy
    monkeypatch.delenv(hierarchy.TIMEOUT_ENV, raising=False)
    monkeypatch.delenv("CORESMITH_TIMEOUT_MULTIPLIER", raising=False)
    assert hierarchy.hierarchy_timeout_s() >= 600
    monkeypatch.setenv("CORESMITH_TIMEOUT_MULTIPLIER", "2")
    assert hierarchy.hierarchy_timeout_s() >= 1200
    monkeypatch.setenv(hierarchy.TIMEOUT_ENV, "7")
    seen = {}

    def run(cmd, **kwargs):
        seen["timeout"] = kwargs["timeout"]
        raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])
    monkeypatch.setattr(hierarchy.shutil, "which", lambda _: "/bin/yosys")
    monkeypatch.setattr(hierarchy.subprocess, "run", run)
    top = tmp_path / "top.v"
    top.write_text("module chip_top(); endmodule\n")
    failure = hierarchy.elaborate_hierarchy([top], "chip_top", project_root=tmp_path)
    assert seen["timeout"] == 14
    assert failure.kind == "infrastructure_error" and hierarchy.TIMEOUT_ENV in failure
