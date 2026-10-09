# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Regression tests for module input ownership and implementation repair."""
from __future__ import annotations

from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import AsyncMock

import pytest

from orchestrator import module_build as MB
from orchestrator.harness import targets
from orchestrator.langgraph import pipeline_graph as pg
from orchestrator.langgraph import pipeline_helpers as ph
from orchestrator.langgraph import shell_integration as shell
from orchestrator.state_store import builds as B
from orchestrator.state_store import stages
from orchestrator.tests.build_fixtures import ready_project


@pytest.mark.parametrize("existing", [False, True])
async def test_failed_spec_author_restores_input_and_rejects_partial_files(tmp_path, monkeypatch, existing):
    from orchestrator.langchain.agents.coresmith_llm import ClaudeLLM

    monkeypatch.setattr(ph, "PROJECT_ROOT", tmp_path)
    spec = tmp_path / "arch/uarch_specs/unit.md"
    spec.parent.mkdir(parents=True)
    original = "# Architect specification\n## Interfaces\ninput request\n"
    if existing:
        spec.write_text(original)

    async def failed_call(*args, **kwargs):
        spec.write_text("# Partial rewrite\n## Interfaces\nwrong ports\n")
        return "[ClaudeLLM error: generation budget exhausted (max turns)]\nI wrote the specification."

    monkeypatch.setattr(ClaudeLLM, "call", failed_call)
    result = await ph.generate_uarch_spec({"name": "unit"})
    assert "generation budget exhausted" in result["error"]
    assert spec.read_text() == original if existing else not spec.exists()


async def test_build_consumes_registered_spec_even_with_pending_revision_feedback(tmp_path, monkeypatch):
    db = ready_project(tmp_path, monkeypatch)
    inputs = B.module_inputs(db, tmp_path, "tiny")
    B.record_dispatch(db, build_id="b-fixed", module="tiny", entry="build_module", graph="build",
                      thread_id="t", inputs=inputs)
    feedback = tmp_path / ".coresmith/blocks/tiny/gate_feedback.txt"
    feedback.parent.mkdir(parents=True, exist_ok=True)
    feedback.write_text("Rewrite the specification")
    author = AsyncMock(side_effect=AssertionError("module build called the spec author"))
    monkeypatch.setattr(pg, "generate_uarch_spec", author)
    state = {"project_root": str(tmp_path), "current_block": {"name": "tiny"}, "build_id": "b-fixed",
             "human_response": {"action": "revise", "feedback": "Change the ports"}}
    out = await pg.generate_uarch_spec_node(state)
    assert out["phase"] == "uarch"
    author.assert_not_called()
    assert B.get_build(db, "b-fixed")["inputs"] == inputs
    Path(inputs["spec"]["path"]).write_text("# Changed by another process\n")
    with pytest.raises(RuntimeError, match="BUILD_STALE"):
        await pg.generate_uarch_spec_node(state)
    assert B.get_build(db, "b-fixed")["inputs"] == inputs


def test_spec_revision_requires_a_new_registered_input(tmp_path, monkeypatch):
    db = ready_project(tmp_path, monkeypatch)
    with pytest.raises(MB.BuildRefusal, match="unchanged") as err:
        MB.plan_module_build(db, tmp_path, "tiny", entry="build_module", uarch_feedback="Change the ports")
    assert err.value.code == "SPEC_REVISION_REQUIRED"
    assert B.builds_for(db) == []
    with pytest.raises(RuntimeError, match="SPEC_REVISION_REQUIRED"):
        pg.route_after_uarch_review({"build_id": "b", "human_response": {"action": "revise"}})


async def test_build_without_a_recorded_spec_digest_is_refused(tmp_path, monkeypatch):
    db = ready_project(tmp_path, monkeypatch)
    (tmp_path / "arch/uarch_specs/tiny.md").unlink()
    inputs = B.module_inputs(db, tmp_path, "tiny")
    assert inputs["spec"]["sha256"] is None
    B.record_dispatch(db, build_id="b-missing", module="tiny", entry="build_module", graph="build",
                      thread_id="t", inputs=inputs)
    with pytest.raises(RuntimeError, match="BUILD_STALE"):
        await pg.generate_uarch_spec_node({"project_root": str(tmp_path), "current_block": {"name": "tiny"},
                                          "build_id": "b-missing"})


def test_simulation_logs_follow_the_explicit_project_root(tmp_path, monkeypatch):
    imported = tmp_path / "engine"
    project = tmp_path / "project"
    monkeypatch.setattr(ph, "_LOG_DIR", imported / ".coresmith/step_logs")
    monkeypatch.delenv("CORESMITH_LOG_DIR", raising=False)
    result = CompletedProcess(["simulator"], 0, stdout="passed", stderr="")
    output = Path(ph._write_step_log("unit", "simulate", result.args, result, project_root=project))
    error = Path(ph._write_step_log_error("unit", "simulate", result.args, "timeout", 2, project_root=project))
    assert output.is_relative_to(project) and "passed" in output.read_text()
    assert error.is_relative_to(project) and "timeout" in error.read_text()
    assert not imported.exists()


def test_failed_shell_allows_module_repair_but_still_blocks_stage_advancement(tmp_path, monkeypatch):
    db = ready_project(tmp_path, monkeypatch, stage="blocks", with_rtl=True)
    db.add_integration_snapshot({"top": "chip_top", "elaborated": False,
                                 "real_blocks": ["tiny"], "stub_blocks": ["sink"],
                                 "wiring_errors": ["SHELL_UNDECLARED_PORT tiny.extra: not connected"]})
    assert MB.readiness_refusal(db, tmp_path, "tiny") is None
    assert any("SHELL_NOT_ELABORATED" in row["ids"] for row in stages.regressed_stages(db, tmp_path))
    assert not stages.advance(db, tmp_path)["advanced"]
    (tmp_path / ".coresmith/vip_index.json").unlink()
    assert MB.readiness_refusal(db, tmp_path, "tiny").code == "ARCHITECTURE_NOT_READY"


def test_missing_pins_still_prevent_module_builds(tmp_path, monkeypatch):
    db = ready_project(tmp_path, monkeypatch)
    with db._tx() as con:
        con.execute("DELETE FROM pins")
    refusal = MB.readiness_refusal(db, tmp_path, "tiny")
    assert refusal.code == "ARCHITECTURE_NOT_READY"
    assert "PINS_MISSING" in str(refusal.blockers)


def test_shell_uses_bound_top_despite_comments_and_helper_declaration_order(tmp_path):
    rtl = tmp_path / "implementation.v"
    rtl.write_text("// one module per stage; module chip_top is the boundary\n"
                   "/* module misleading */\n"
                   "module helper(input extra); endmodule\n"
                   "module chip_top(input clk, input rst_n, output result);\n"
                   "assign result = 1'b0; endmodule\n")
    targets.bind(tmp_path, "unit", {"top": "chip_top", "sources": ["implementation.v"]})
    asm = shell.assemble_top(tmp_path, top_name="chip_top", blocks=["unit"], edges=[],
                             rtl_paths={"unit": str(rtl)}, out_dir=tmp_path / "shell")
    assert not asm.wiring_errors
    assert asm.block_modules == {"unit": "chip_top_core"}
    renamed = Path(asm.block_sources["unit"]).read_text()
    assert "// one module per stage; module chip_top is the boundary" in renamed
    assert "module chip_top_core(input" in renamed
    assert "extra" not in asm.verilog


def test_ambiguous_unbound_top_reports_binding_required(tmp_path):
    rtl = tmp_path / "implementation.v"
    rtl.write_text("module helper(input x); endmodule\nmodule other(input y); endmodule\n")
    asm = shell.assemble_top(tmp_path, top_name="chip_top", blocks=["unit"], edges=[],
                             rtl_paths={"unit": str(rtl)}, out_dir=tmp_path / "shell")
    assert any("TARGET_UNBOUND" in error for error in asm.wiring_errors)


def test_missing_bound_top_cannot_borrow_another_modules_ports(tmp_path):
    rtl = tmp_path / "implementation.v"
    rtl.write_text("module implementation(input clk, output result); endmodule\n")
    targets.bind(tmp_path, "unit", {"top": "missing", "sources": ["implementation.v"]})
    asm = shell.assemble_top(tmp_path, top_name="chip_top", blocks=["unit"], edges=[],
                             rtl_paths={"unit": str(rtl)}, out_dir=tmp_path / "shell")
    assert any("could not parse" in error for error in asm.wiring_errors)
    assert not any(port["name"].endswith("result") for port in asm.boundary_ports)


def test_worker_receives_registered_model_top_and_derived_assertion_ids(tmp_path, monkeypatch):
    from orchestrator.langchain.agents.rtl_generator import build_user_message
    from orchestrator.langgraph.assertion_stage import contract_invariants

    db = ready_project(tmp_path, monkeypatch)
    targets.bind(tmp_path, "tiny", {"top": "implementation", "sources": ["rtl/tiny.v"]})
    text = build_user_message("tiny", project_root=str(tmp_path), rtl_target="rtl/tiny.v")
    assert db.model_for("tiny")["path"] in text
    assert "Registered reference model (systemc)" in text
    assert '"top": "implementation"' in text
    checks = contract_invariants(tmp_path, "tiny")
    assert any(check["id"].endswith("reset_idle") for check in checks)
    for check in checks:
        assert check["id"] in text and check["text"] in text
    assert "- Golden Model: \n" not in text


@pytest.mark.parametrize("tamper", [False, True])
async def test_primitive_coverage_closure_adds_tests_without_replacing_generated_tb(tmp_path, monkeypatch, tamper):
    from orchestrator.state_store.trust import write_oracle_manifest

    ready_project(tmp_path, monkeypatch)
    write_oracle_manifest(tmp_path)
    rtl = tmp_path / "fabric.v"
    rtl.write_text("module fabric(); endmodule\n")
    tb = tmp_path / "test_fabric.py"
    original = "# Generated fabric acceptance test\n"
    tb.write_text(original)
    monkeypatch.setattr(pg, "_write_contract_slice", lambda *a: None)
    calls = []

    def simulate(block, rtl_path, tb_path, attempt, **kwargs):
        calls.append(kwargs.get("extra_tb_paths", []))
        assert Path(tb_path).read_text() == original
        passed = bool(calls[-1])
        return {"passed": passed, "tests_passed": 1, "tests_total": 1, "tests_failed": 0,
                "coverage_gate_failed": not passed, "log": "coverage below floor" if not passed else "passed"}

    async def supplement(block, rtl_path, tb_path, output, *a, **k):
        if tamper:
            Path(tb_path).write_text("# Weakened generated tests\n")
            return False
        Path(output).write_text("import cocotb\n@cocotb.test()\nasync def additional(dut):\n    pass\n")
        return True

    monkeypatch.setattr(pg, "run_simulation", simulate)
    monkeypatch.setattr(pg, "author_supplemental_tests", supplement)
    monkeypatch.setattr(pg, "generate_testbench", AsyncMock(side_effect=AssertionError("replaced generated TB")))
    monkeypatch.setattr(pg, "fix_testbench_errors", AsyncMock(side_effect=AssertionError("edited generated TB")))
    state = {"project_root": str(tmp_path), "attempt": 1, "rtl_path": str(rtl), "current_block": {
        "name": "fabric", "kind": "primitive", "primitive": "cs_fabric", "testbench": str(tb)}}
    if tamper:
        with pytest.raises(RuntimeError, match="generated primitive testbench was changed"):
            await pg.generate_testbench_node(state)
        assert len(calls) == 1
    else:
        result = await pg.generate_testbench_node(state)
        assert result["sim_passed"]
        assert len(calls) == 2 and calls[0] == [] and len(calls[1]) == 1
    assert tb.read_text() == original
