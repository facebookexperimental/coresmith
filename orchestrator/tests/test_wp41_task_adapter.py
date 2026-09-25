"""WP-41: the task-adapter boundary -- run the task's own checker, refuse partial receipts."""
from __future__ import annotations

import inspect
import json
import os
import sys
from pathlib import Path

import pytest

from orchestrator.harness import task_adapter as ta
from orchestrator.tests.candidate_fixtures import adopt


@pytest.fixture(autouse=True)
def _adapter_policy(monkeypatch):
    monkeypatch.setenv("CORESMITH_ADAPTER_SANDBOX", "none")


def _run_adapter(root, *args, **kwargs):
    # Each case installs its own owner adapter; snapshot before that evaluation.
    from orchestrator.state_store.trust import capture_run_baseline
    capture_run_baseline(root)
    return ta.run_task_adapter(root, *args, **kwargs)


TOP_RTL = "module chip_top(input wire clk, input wire rst_n, output wire y);\n  assign y = 1'b1;\nendmodule\n"
BLOCK_RTL = "module leaf(input wire a, output wire b);\n  assign b = a;\nendmodule\n"


def _project(tmp_path, adapter_src: str):
    root = tmp_path / "proj"
    (root / "inputs").mkdir(parents=True)
    (root / "rtl").mkdir()
    top = root / "rtl" / "chip_top.v"
    top.write_text(TOP_RTL)
    blk = root / "rtl" / "leaf.v"
    blk.write_text(BLOCK_RTL)
    (root / "inputs" / "task_adapter.py").write_text(adapter_src)
    adopt(root, top, {"leaf": str(blk)})
    return root, top, blk


GOOD = '''
CASES = ["c1", "c2"]
TOP = "chip_top"
LABEL = "fake published grader"
def grade(candidate, workdir):
    assert candidate["top"] == "chip_top" and len(candidate["sources"]) == 2
    return {"cases": {"c1": {"ok": True, "cycles": 10},
                      "c2": {"ok": True, "cycles": 12}},
            "budgets": {"throughput": {"ok": True, "measured": 1.0, "budget": 2.0}},
            "detail": "all good"}
'''


def test_no_adapter_means_none(tmp_path, monkeypatch):
    monkeypatch.delenv("CORESMITH_TASK_ADAPTER", raising=False)
    root = tmp_path / "p"
    (root / "rtl").mkdir(parents=True)
    (root / "rtl" / "t.v").write_text(TOP_RTL)
    assert _run_adapter(str(root), str(root / "rtl" / "t.v"), {}) is None


def test_complete_passing_receipt(tmp_path, monkeypatch):
    monkeypatch.delenv("CORESMITH_TASK_ADAPTER", raising=False)
    monkeypatch.delenv("CORESMITH_TASK_ADAPTER_PYTHON", raising=False)
    root, top, blk = _project(tmp_path, GOOD)
    res = _run_adapter(str(root), str(top), {"leaf": str(blk)})
    assert res["passed"] is True and res["kind"] is None and not res["oracle_incomplete"] if "oracle_incomplete" in res else res["passed"]
    assert [r["name"] for r in res["cases"]] == ["c1", "c2"]
    assert res["label"] == "fake published grader" and res["top"] == "chip_top"
    assert res["requested_cases"] == res["completed_cases"] == 2
    keep = Path(res["captured_dir"])
    assert (keep / "receipt.json").exists() and (keep / "candidate.json").exists()
    assert json.loads((root / ".coresmith" / "acceptance_dv.json").read_text())["passed"] is True
    # identity changes with the RTL
    blk.write_text(BLOCK_RTL.replace("assign b = a", "assign b = ~a"))
    adopt(root, top, {"leaf": str(blk)})
    res2 = _run_adapter(str(root), str(top), {"leaf": str(blk)})
    assert res2["candidate_sha"] != res["candidate_sha"]


def test_adapter_application_cache_is_private(tmp_path, monkeypatch):
    parent_home = os.environ.get("HOME")
    parent_cache = tmp_path / "parent-cache"
    parent_cache.mkdir()
    monkeypatch.setenv("APPDATA", str(parent_cache))
    root, top, blk = _project(tmp_path, '''
import os
from pathlib import Path
CASES = ["cache"]
def grade(candidate, workdir):
    assert "HOME" not in os.environ, "read-only host home leaked into application cache selection"
    cache = Path(os.environ["APPDATA"])
    (cache / "wisdom.lock").write_text("application cache")
    assert cache.is_relative_to(Path(workdir)), "cache escaped the evaluator work directory"
    assert Path(os.environ["XDG_CACHE_HOME"]).is_dir()
    return {"cases": {"cache": {"ok": True}}}
''')
    res = _run_adapter(str(root), str(top), {"leaf": str(blk)})
    assert res["passed"], res
    assert not list(parent_cache.iterdir())
    assert os.environ["APPDATA"] == str(parent_cache)
    assert os.environ.get("HOME") == parent_home


def test_functional_and_budget_failures_are_typed(tmp_path, monkeypatch):
    monkeypatch.delenv("CORESMITH_TASK_ADAPTER", raising=False)
    root, top, blk = _project(tmp_path, '''
CASES = ["a", "b"]
def grade(candidate, workdir):
    return {"cases": {"a": {"ok": True, "cycles": 5},
                      "b": {"ok": False, "detail": "byte 3 differs", "cycles": 9}}}
''')
    res = _run_adapter(str(root), str(top), {"leaf": str(blk)})
    assert res["passed"] is False and res["kind"] == "functional_fail" and not res["skipped"]
    assert res["violations"][0]["criterion"] == "task_adapter_functional"
    assert res["violations"][0]["acceptance_case"] == "b"
    (root / "inputs" / "task_adapter.py").write_text('''
CASES = ["a"]
def grade(candidate, workdir):
    return {"cases": {"a": {"ok": True}},
            "budgets": {"throughput": {"ok": False, "measured": 5263.0, "budget": 4.21,
                                       "detail": "cycles per bit over the task cap"}}}
''')
    res = _run_adapter(str(root), str(top), {"leaf": str(blk)})
    assert res["passed"] is False and res["kind"] == "budget_fail"
    assert res["violations"][0]["criterion"] == "task_adapter_budget"
    assert res["violations"][0]["measured"]["measured"] == 5263.0


def test_incomplete_receipt_never_passes(tmp_path, monkeypatch):
    """Two cases declared, one reported -> oracle_incomplete, not a pass."""
    monkeypatch.delenv("CORESMITH_TASK_ADAPTER", raising=False)
    root, top, blk = _project(tmp_path, '''
CASES = ["one", "two"]
def grade(candidate, workdir):
    return {"cases": {"one": {"ok": True}}}
''')
    res = _run_adapter(str(root), str(top), {"leaf": str(blk)})
    assert res["passed"] is False and res["oracle_incomplete"] and res["kind"] == "oracle_incomplete"
    assert "missing ['two']" in res["reason"]
    (root / "inputs" / "task_adapter.py").write_text('''
CASES = ["one"]
def grade(candidate, workdir):
    return {"cases": {"one": {"ok": "yes"}, "extra": {"ok": True}}}
''')
    res = _run_adapter(str(root), str(top), {"leaf": str(blk)})
    assert res["oracle_incomplete"] and "undeclared ['extra']" in res["reason"] and "non-boolean" in res["reason"]


def test_adapter_exception_and_wrong_top_are_typed(tmp_path, monkeypatch):
    monkeypatch.delenv("CORESMITH_TASK_ADAPTER", raising=False)
    root, top, blk = _project(tmp_path, '''
CASES = ["x"]
def grade(candidate, workdir):
    raise RuntimeError("grader crashed")
''')
    res = _run_adapter(str(root), str(top), {"leaf": str(blk)})
    assert res["oracle_incomplete"] and res["kind"] == "adapter_defect" and "grader crashed" in res["reason"]
    (root / "inputs" / "task_adapter.py").write_text('''
CASES = ["x"]
TOP = "user_project_wrapper"
def grade(candidate, workdir):
    return {"cases": {"x": {"ok": True}}}
''')
    res = _run_adapter(str(root), str(top), {"leaf": str(blk)})
    assert res["passed"] is False and res["kind"] == "boundary_mismatch"
    assert not res.get("oracle_incomplete") and "top module is 'chip_top'" in res["reason"]
    assert res["violations"][0]["criterion"] == "task_adapter_boundary"
    (root / "inputs" / "task_adapter.py").write_text("import nonexistent_package_xyz\nCASES=['x']\n")
    res = _run_adapter(str(root), str(top), {"leaf": str(blk)})
    assert res["oracle_incomplete"] and res["kind"] == "adapter_defect" and "rc=3" in res["reason"]


def test_header_selects_interpreter_and_timeout(tmp_path, monkeypatch):
    monkeypatch.delenv("CORESMITH_TASK_ADAPTER", raising=False)
    monkeypatch.delenv("CORESMITH_TASK_ADAPTER_PYTHON", raising=False)
    root, top, blk = _project(tmp_path, f'''# coresmith-python: {sys.executable}
# coresmith-timeout-s: 1
import time
CASES = ["slow"]
def grade(candidate, workdir):
    time.sleep(5)
    return {{"cases": {{"slow": {{"ok": True}}}}}}
''')
    hdr = ta.read_header(str(root / "inputs" / "task_adapter.py"))
    assert hdr == {"python": sys.executable, "timeout-s": "1"}
    res = _run_adapter(str(root), str(top), {"leaf": str(blk)})
    assert res["oracle_incomplete"] and res["kind"] == "infrastructure_error" and "exceeded 1s" in res["reason"]
    (root / "inputs" / "task_adapter.py").write_text("# coresmith-python: /nonexistent/python\nCASES=['x']\ndef grade(c, w):\n    return {}\n")
    res = _run_adapter(str(root), str(top), {"leaf": str(blk)})
    assert res["kind"] == "adapter_defect" and "interpreter not found" in res["reason"]


def test_candidate_includes_only_explicitly_adopted_pads(tmp_path):
    root = tmp_path / "p"
    (root / "rtl" / "integration").mkdir(parents=True)
    top = root / "rtl" / "integration" / "user_project_wrapper.v"
    top.write_text("module user_project_wrapper(); endmodule\n")
    (root / "rtl" / "integration" / "user_project_wrapper_pads.v").write_text("module pads(); endmodule\n")
    adopt(root, top, {"pads": str(top.with_name("user_project_wrapper_pads.v"))}, name="user_project_wrapper")
    c = ta.assemble_candidate(str(root), str(top), {})
    assert c["top"] == "user_project_wrapper"
    assert [Path(s).name for s in c["sources"]] == ["user_project_wrapper.v", "user_project_wrapper_pads.v"]
    assert len(c["candidate_sha"]) == 64


def test_validation_node_prefers_the_adapter():
    from orchestrator.langgraph import pipeline_graph as pg
    src = inspect.getsource(pg.validation_dv_node)
    assert "run_task_adapter" in src
    assert src.index("run_task_adapter, pr, top_rtl_path, block_rtl_paths") < src.index("run_acceptance_dv, pr, top_rtl_path, block_rtl_paths")
