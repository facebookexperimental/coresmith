"""WP-66: policy dimensions cannot create an owner certification obligation."""
import json
from unittest.mock import AsyncMock, Mock

import pytest
import yaml

from orchestrator.langgraph import final_report as fr
from orchestrator.langgraph import pipeline_graph as pg

DIMS = {"queue_depth": 64, "address_limit": 255}
NOTICE = "maximum geometry not owner-certified (no max_geometry_cases declared)"
CASES = {"test_limits": DIMS}
TB = '''import cocotb

@cocotb.test()
async def test_limits(dut):
    # MAXGEO: queue_depth=64 address_limit=255
    assert dut.ready.value

@cocotb.test()
async def test_smoke(dut):
    """Check queue_depth at a small configuration."""
    assert dut.ready.value

def helper():
    # address_limit is also mentioned outside a test case.
    pass
'''


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setenv("CORESMITH_MAXGEO_GATE", "1")
    monkeypatch.setenv("CORESMITH_DETERMINISTIC_BFM", "0")
    (tmp_path / "inputs").mkdir()
    (tmp_path / "inputs/task.yaml").write_text("chassis: none\n")
    cs = tmp_path / ".coresmith"
    cs.mkdir()
    (cs / "block_specs.json").write_text(json.dumps([
        {"name": "leaf", "parameters": [
            {"name": name, "max": value} for name, value in DIMS.items()
        ]}
    ]))
    (cs / "ers_spec.json").write_text(json.dumps({"ers": {"summary": "Queue controller"}}))
    top = tmp_path / "leaf.v"
    top.write_text("module leaf(input clk, output ready); assign ready=clk; endmodule\n")
    tb = tmp_path / "test_leaf.py"
    tb.write_text(TB)
    generated = {"testbench_path": str(tb), "test_count": 2}
    monkeypatch.setattr(pg, "generate_integration_testbench", AsyncMock(return_value=generated))
    monkeypatch.setattr(pg, "generate_validation_testbench", AsyncMock(return_value=generated))
    monkeypatch.setattr(pg, "load_architecture_connections", lambda _: ([], "queue"))
    sim = Mock(return_value={"passed": True, "executed_cases": [], "log": "sim passed"})
    monkeypatch.setattr(pg, "run_integration_simulation", sim)
    monkeypatch.setattr("orchestrator.harness.task_adapter.run_task_adapter",
                        Mock(return_value={"passed": True}))
    monkeypatch.setattr(pg, "_chip_top_synth_ok", lambda *a: (True, ""))
    monkeypatch.setattr(pg, "_measured_die_rollup", lambda *a: None)
    monkeypatch.setattr(pg, "_run_top_level_contract_audit", AsyncMock(return_value={}))
    events = Mock()
    monkeypatch.setattr(pg, "write_graph_event", events)
    state = {"project_root": str(tmp_path), "integration_result": {
        "top_rtl_path": str(top), "design_name": "queue", "block_rtl_paths": {"leaf": str(top)},
    }}
    return tmp_path, tb, state, sim, events


@pytest.mark.parametrize("stage", ["integration", "validation"])
@pytest.mark.parametrize("action", [None, "fix_tb", "fix_rtl"])
@pytest.mark.parametrize("declaration", ["absent", "empty"])
async def test_no_owner_cases_preserves_dv_and_uncertified_evidence(project, stage, action, declaration):
    root, tb, state, sim, events = project
    if declaration == "empty":
        (root / "inputs/task.yaml").write_text("max_geometry_cases: {}\n")
    original = (root / "inputs/task.yaml").read_bytes()
    key = f"{stage}_dv_result"
    if action:
        state[key] = {"action_taken": action, "testbench_path": str(tb), "test_count": 2}
    # Even a successful test mentioning all dimensions is not owner certification.
    sim.return_value["executed_cases"] = ["test_limits", "test_smoke"]
    result = await getattr(pg, f"{stage}_dv_node")(state)
    dv = result[key]
    assert dv["passed"] is True, dv
    assert not dv.get("pending_decision")
    assert result["pipeline_done"] is (stage == "validation")
    gate = dv["max_geometry"]
    assert gate["verdict"] == "not_declared"
    assert gate["reason"] == NOTICE
    assert gate["declared_dims"] == DIMS
    assert gate["marker_pairs"] == DIMS
    assert gate["uncovered_dims"] == DIMS
    assert gate["executed_maximum_cases"] == []
    assert gate["testbench_case_mentions"] == {
        "test_limits": ["queue_depth", "address_limit"], "test_smoke": ["queue_depth"],
    }
    records = [c.args[3] for c in events.call_args_list if c.args[2] == "maxgeo_verdict"]
    assert records == [gate]
    assert not any(c.args[2] == "maxgeo_gate_pass" for c in events.call_args_list)
    report = fr.build_final_report({**state, **result}, str(root))
    assert report["chip"][f"{stage}_dv"]["max_geometry"] == gate
    text = fr.render_markdown(report)
    line = next(line for line in text.splitlines() if line.startswith(f"- Maximum geometry ({stage})"))
    assert "not_declared" in line and NOTICE in line
    assert "verified" not in line and "**pass**" not in line
    for term in (*DIMS, "test_limits", "test_smoke", "Marker pairs", "Policy-declared dimensions"):
        assert term in text
    assert (root / "inputs/task.yaml").read_bytes() == original


@pytest.mark.parametrize("stage", ["integration", "validation"])
@pytest.mark.parametrize("declaration,executed,verdict", [
    pytest.param(CASES, [], "unknown", id="declared-unexecuted"),
    pytest.param(CASES, ["test_limits"], "pass", id="declared-executed"),
    pytest.param(CASES, ["different_case"], "unknown", id="wrong-case"),
    pytest.param({"test_limits": {"queue_depth": 64}}, ["test_limits"], "unknown", id="partial"),
    pytest.param(["test_limits"], [], "unknown", id="malformed-list"),
    pytest.param(None, [], "unknown", id="malformed-null"),
    pytest.param(False, [], "unknown", id="malformed-false"),
    pytest.param([], [], "unknown", id="malformed-empty-list"),
    pytest.param({**CASES, "bad": "not maxima"}, ["test_limits"], "unknown", id="malformed-extra-case"),
])
async def test_owner_declarations_still_require_successful_execution(project, stage, declaration, executed, verdict):
    root, _, state, sim, events = project
    (root / "inputs/task.yaml").write_text(yaml.safe_dump({"max_geometry_cases": declaration}))
    sim.return_value["executed_cases"] = executed
    result = await getattr(pg, f"{stage}_dv_node")(state)
    dv = result[f"{stage}_dv_result"]
    assert dv["passed"] is (verdict == "pass"), dv
    assert dv["max_geometry"]["verdict"] == verdict
    assert any(c.args[2] == "maxgeo_gate_pass" for c in events.call_args_list) is (verdict == "pass")
    if verdict == "pass":
        assert dv["max_geometry"]["executed_maximum_cases"] == ["test_limits"]
        report = fr.build_final_report({**state, **result}, str(root))
        text = fr.render_markdown(report)
        assert f"- Maximum geometry ({stage}): **pass**" in text
    else:
        assert dv["pending_decision"] is True
