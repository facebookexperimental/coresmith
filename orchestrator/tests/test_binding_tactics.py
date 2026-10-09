"""Regression cases from the two-arm run, without model or network calls."""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from orchestrator import processes
from orchestrator.architect.outcomes import outcome
from orchestrator.architect.runners import CodexRunner, make_runner, resolve_runner
from orchestrator.harness import targets


@pytest.fixture(autouse=True)
def clean_execution(monkeypatch, tmp_path):
    monkeypatch.setenv("CORESMITH_PROJECT_ROOT", str(tmp_path))
    for key in ("CORESMITH_ARCHITECT_PROVIDER", "CORESMITH_LLM_PROVIDER"):
        monkeypatch.delenv(key, raising=False)
    token = processes._owner.set("")
    yield
    processes.cancel(grace_s=.1)
    processes._owner.reset(token)


def test_codex_binding_never_falls_back_to_claude(monkeypatch):
    monkeypatch.setenv("CORESMITH_LLM_PROVIDER", "codex")
    assert resolve_runner() == "codex"
    assert isinstance(make_runner("codex", "gpt-6-sol", "/unused/codex"), CodexRunner)
    with pytest.raises(ValueError, match="AGENT_BINDING_INVALID"):
        make_runner("claude", "gpt-6-sol", "/unused/claude")
    monkeypatch.setenv("CORESMITH_LLM_PROVIDER", "typo")
    with pytest.raises(ValueError):
        resolve_runner()


def test_explicit_sitting_model_wins_over_engine_default(monkeypatch):
    from orchestrator.architect.runners import resolve_model
    monkeypatch.setenv("CORESMITH_CODEX_MODEL", "gpt-default")
    assert resolve_model("codex", "gpt-6-sol") == "gpt-6-sol"


def test_codex_runner_reads_real_cli_events(tmp_path):
    cli = tmp_path / "codex"
    cli.write_text(f'''#!{sys.executable}
import json,sys
assert "gpt-6-sol" in sys.argv
assert "workspace-write" in sys.argv
assert "binding instructions" in sys.stdin.read()
print(json.dumps({{"type":"thread.started","thread_id":"thread-1"}}))
print(json.dumps({{"type":"item.completed","item":{{"type":"agent_message","text":"complete"}}}}))
print(json.dumps({{"type":"turn.completed","usage":{{"input_tokens":10,"output_tokens":5}}}}))
''')
    cli.chmod(0o755)
    system = tmp_path / "system.md"
    system.write_text("binding instructions")
    result = CodexRunner(str(cli), "gpt-6-sol").run(
        "do work", system_file=system, resume="", transcript=tmp_path / "trace.jsonl",
        cwd=tmp_path, env=dict(os.environ), max_turns=10, timeout_s=10)
    assert result["ok"] and result["session_id"] == "thread-1"
    assert result["tokens"]["input_tokens"] == 10


def test_timeout_is_not_model_commentary(tmp_path):
    from orchestrator.architect.runners import _run_cli
    rc, err = _run_cli([sys.executable, "-c", "import time;time.sleep(20)"], "",
                      transcript=tmp_path / "trace", cwd=tmp_path, env=dict(os.environ), timeout_s=.1)
    result = outcome({"ok": False, "rc": rc, "stderr": err, "text": "I suspect a codec defect"})
    assert result["status"] == "timed_out"
    assert "codec" not in result["reason"]
    assert json.loads(next((tmp_path / ".coresmith/jobs").glob("*.json")).read_text())["status"] == "timed_out"


def test_cancel_fences_future_launch_and_only_kills_owner():
    first = processes.begin_owner("one")
    one = processes.popen([sys.executable, "-c", "import time;time.sleep(30)"])
    second = processes.begin_owner("two")
    two = processes.popen([sys.executable, "-c", "import time;time.sleep(30)"])
    assert processes.cancel(first, grace_s=.1) == 1
    assert one.poll() is not None and two.poll() is None
    token = processes._owner.set(first)
    try:
        with pytest.raises(RuntimeError, match="cancelled"):
            processes.run([sys.executable, "-c", "raise Exception('must not execute')"])
    finally:
        processes._owner.reset(token)
        processes.cancel(second, grace_s=.1)


@pytest.mark.asyncio
async def test_pause_joins_direct_simulator_and_prevents_stale_thread_launch(tmp_path):
    from orchestrator.graph_lifecycle import GraphLifecycle
    graph = GraphLifecycle("pipeline", str(tmp_path / "db"), "unused", "unused", str(tmp_path))
    ready = asyncio.Event()
    owner = processes.begin_owner("pipeline")
    graph._job_owner = owner
    process = processes.popen([sys.executable, "-c", "import time;time.sleep(30)"])
    async def waiting():
        ready.set()
        await asyncio.sleep(30)
    graph.task = asyncio.create_task(waiting())
    await ready.wait()
    assert await graph.safe_pause()
    assert process.poll() is not None
    assert not processes.active(owner)
    with pytest.raises(RuntimeError, match="cancelled"):
        await asyncio.to_thread(processes.run, [sys.executable, "-c", "pass"])


def test_output_directory_has_one_writer(tmp_path):
    with processes.output_lock(tmp_path):
        with pytest.raises(RuntimeError, match="active job"):
            processes.run([sys.executable, "-c", "pass"], output_dir=tmp_path)
    assert processes.run([sys.executable, "-c", "pass"], output_dir=tmp_path).returncode == 0


def target_files(tmp_path):
    (tmp_path / "rtl").mkdir()
    (tmp_path / "inc").mkdir()
    (tmp_path / "rtl/main.v").write_text("module actual(input a, output b); assign b=a; endmodule\n")
    (tmp_path / "inc/constants.vh").write_text("`define SIZE 4\n")
    return {"top": "actual", "sources": ["rtl/main.v"], "include_dirs": ["inc"],
            "defines": {"MODE": 1}, "parameters": {}, "cwd": "."}


def test_target_revision_tracks_include_and_configuration(tmp_path):
    doc = target_files(tmp_path)
    targets.bind(tmp_path, "logical", doc)
    target = targets.load(tmp_path, "logical")
    before = targets.revision(target)
    (tmp_path / "inc/constants.vh").write_text("`define SIZE 8\n")
    assert targets.revision(target) != before
    before = targets.revision(target)
    targets.bind(tmp_path, "logical", {**doc, "defines": {"MODE": 2}})
    assert targets.revision(targets.load(tmp_path, "logical")) != before


def test_invalid_binding_does_not_replace_prior_value(tmp_path):
    doc = target_files(tmp_path)
    targets.bind(tmp_path, "logical", doc)
    for invalid in ({**doc, "sources": []}, {**doc, "sources": ["rtl"]}, {**doc, "top": ""},
                    {**doc, "parameters": {"W": "4; delete"}}):
        with pytest.raises(ValueError, match="TARGET_INVALID"):
            targets.bind(tmp_path, "logical", invalid)
    assert targets.load(tmp_path, "logical")["top"] == "actual"


def test_lint_uses_bound_sources_top_flags_and_cwd(tmp_path, monkeypatch):
    from orchestrator.langgraph import pipeline_helpers as h
    doc = target_files(tmp_path)
    targets.bind(tmp_path, "logical", doc)
    monkeypatch.setattr(h, "PROJECT_ROOT", tmp_path)
    seen = []
    def run(cmd, **kw):
        seen.append((cmd, kw))
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(h, "run_process", run)
    assert h.lint_rtl("unbound.v", "logical")["clean"]
    cmd, kwargs = seen[0]
    assert cmd[cmd.index("--top-module")+1] == "actual"
    assert str(tmp_path / "rtl/main.v") in cmd
    assert "-DMODE=1" in cmd and str(tmp_path) == kwargs["cwd"]


def test_changed_input_cannot_earn_a_pass(tmp_path, monkeypatch):
    from orchestrator.langgraph import pipeline_helpers as h
    targets.bind(tmp_path, "logical", target_files(tmp_path))
    monkeypatch.setattr(h, "PROJECT_ROOT", tmp_path)
    def run(cmd, **kw):
        (tmp_path / "inc/constants.vh").write_text("`define SIZE 8\n")
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(h, "run_process", run)
    result = h.lint_rtl("unbound.v", "logical")
    assert not result["clean"] and result["input_revision"]
    assert "TARGET_CHANGED" in result["errors"]


def test_parity_build_uses_bound_configuration(tmp_path, monkeypatch):
    import shlex

    from orchestrator.langgraph import pipeline_helpers as h
    targets.bind(tmp_path, "logical", {**target_files(tmp_path), "parameters": {"W": "8'hff"}})
    tb = tmp_path / "tb.py"
    tb.write_text("# testbench\n")
    monkeypatch.setattr(h, "create_golden_model_wrapper", lambda *a, **k: None)
    class Stop(Exception):
        pass
    def run(*a, **kw):
        raise Stop()
    monkeypatch.setattr(h, "run_process", run)
    with pytest.raises(Stop):
        h.run_simulation({"name": "logical"}, "old.v", str(tb), project_root=tmp_path,
                         sim_subdir="logical__parity", extra_defines=["SYNTHESIS"])
    makefile = (tmp_path / "sim_build/logical__parity/Makefile").read_text()
    assert "TOPLEVEL = actual" in makefile and "-DSYNTHESIS" in makefile and "-DMODE=1" in makefile
    assert str(tmp_path / "rtl/main.v") in makefile and "old.v" not in makefile
    flags = [v.removeprefix("EXTRA_ARGS += ") for v in makefile.splitlines() if v.startswith("EXTRA_ARGS +=")]
    assert "-GW=8'hff" in shlex.split(" ".join(flags))


@pytest.mark.asyncio
async def test_generation_checks_bound_include_file_instead_of_old_filename(tmp_path, monkeypatch):
    from unittest.mock import AsyncMock, Mock

    from orchestrator.langgraph import pipeline_graph as pg
    doc = target_files(tmp_path)
    (tmp_path / "rtl/main.v").write_text('`include "implementation.v"\n')
    targets.bind(tmp_path, "logical", doc)
    monkeypatch.setattr(pg, "generate_rtl", AsyncMock(return_value={"ok": True}))
    lint = Mock(return_value={"clean": True})
    monkeypatch.setattr(pg, "lint_rtl", lint)
    result = await pg.generate_rtl_node({"project_root": str(tmp_path), "attempt": 1,
                                       "current_block": {"name": "logical", "rtl_target": "old.v"}})
    assert result["lint_clean"] and result["rtl_path"] == str(tmp_path / "rtl/main.v")
    assert lint.call_args.args[0] == result["rtl_path"]


def test_small_and_include_only_rtl_is_left_for_elaboration(tmp_path):
    from orchestrator.langgraph.pipeline_helpers import _assert_rtl_materialized
    p = tmp_path / "small.v"
    p.write_text('`include "core.v"\n')
    assert _assert_rtl_materialized(p, "core") is None
    assert "not a file" in _assert_rtl_materialized(tmp_path, "core")


def test_malformed_contract_policy_refused():
    from orchestrator.harness.tools.validate import validate_contracts
    errors = validate_contracts({"contracts": [{"edge_id": "e", "flow_control_policy": "skid"}]})
    assert any(e["code"] == "CT_POLICY_TYPE" for e in errors)


def test_supervisor_does_not_call_park_or_failure_finished():
    from orchestrator.daemon.supervisor import classify
    assert classify({"pipeline_done": True, "interrupts": [{}]}, None)[0] == "awaiting_decision"
    assert classify({"pipeline_done": True, "frontend_outcome": "failed"}, None)[0] == "blocked"
    assert classify({"frontend_outcome": "complete"}, None)[0] == "frontend_complete"
    assert classify({"frontend_outcome": "complete", "interrupts": [{"consumed_by_resume": True}]}, None)[0] == "frontend_complete"


def test_terminal_pipeline_with_missing_blocks_is_not_complete():
    from types import SimpleNamespace

    from orchestrator.daemon.server import _shape_state
    state = SimpleNamespace(values={"pipeline_done": True, "block_queue": [{"name": "core"}],
                                    "completed_blocks": []}, tasks=(), next=())
    assert _shape_state(state)["frontend_outcome"] == "failed"


def test_rebinding_invalidates_publication_without_erasing_evidence(tmp_path):
    from orchestrator.state_store.project_db import open_project
    doc = target_files(tmp_path)
    targets.bind(tmp_path, "logical", doc)
    db = open_project(tmp_path)
    db.set_result("logical", "best", {"done": True})
    targets.bind(tmp_path, "logical", doc)
    assert db.result("logical", "best")["done"]
    targets.bind(tmp_path, "logical", {**doc, "defines": {"MODE": 2}})
    assert db.result("logical", "best") is None
    assert db.result("logical", "target_invalidated_best")["done"]


def test_shell_refuses_equal_but_contract_wrong_widths(tmp_path):
    from orchestrator.langgraph import shell_integration as shell
    from orchestrator.tests.test_shell_integration import _EDGE, _rsp_rtl
    rsp = _rsp_rtl(tmp_path)
    edge = json.loads(json.dumps(_EDGE))
    # Real RTL and stub agree with each other; the explicit connection still owns the required width.
    asm = shell.assemble_top(tmp_path, top_name="chip_top", blocks=["req", "rsp"], edges=[edge],
                             rtl_paths={"rsp": str(rsp)}, out_dir=tmp_path / "shell")
    assert not asm.wiring_errors
    original = (tmp_path / "shell/chip_top.v").read_bytes()
    rsp.write_text(rsp.read_text().replace('[7:0]', '[3:0]'))
    bad = shell.assemble_top(tmp_path, top_name="chip_top", blocks=["req", "rsp"], edges=[edge],
                             rtl_paths={"rsp": str(rsp)}, out_dir=tmp_path / "shell")
    assert bad.wiring_errors and not bad.rtl_path
    assert (tmp_path / "shell/chip_top.v").read_bytes() == original
    assert shell.elaborate(bad)["ok"] is False


def test_daemon_does_not_publish_without_ownership(tmp_path, monkeypatch):
    from orchestrator.daemon import server
    class BrokenDB:
        def acquire_lease(self, *args, **kwargs):
            raise OSError("database unavailable")
    monkeypatch.setattr(server, "_project_db", lambda: BrokenDB())
    monkeypatch.setattr(server, "_daemon_file", lambda: tmp_path / "daemon.json")
    with pytest.raises(RuntimeError, match="ownership"):
        server._write_daemon_file(1234)
    assert not (tmp_path / "daemon.json").exists()


@pytest.mark.asyncio
async def test_fabric_width_mismatch_keeps_previous_publication(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from orchestrator import fabric
    from orchestrator.langgraph import contract_conformance as cc
    from orchestrator.langgraph import pipeline_graph as pg
    from orchestrator.state_store.project_db import open_project
    from orchestrator.tests.test_fabric_pipeline import _BLOCK, TestBusFamilies

    db = open_project(tmp_path)
    db.import_block_diagram({"blocks": [_BLOCK], "connections": []})
    db.import_contracts({"contracts": [TestBusFamilies._EDGE]})
    rtl = tmp_path / _BLOCK["rtl_target"]
    rtl.parent.mkdir(parents=True)
    rtl.write_text("previous published candidate\n")
    rows = cc.contract_port_rows(tmp_path, "fabric")
    def generate(spec, out_dir, **kw):
        out_dir.mkdir(parents=True, exist_ok=True)
        ports = [f"{p['dir']} [{int(p['width'])-1}:0] {p['port']}" for p in rows]
        source = f"module {spec.module_name}(" + ",".join(ports) + "); endmodule\n"
        source = source.replace("[63:0]", "[31:0]")
        (out_dir / "fabric.v").write_text(source)
        (out_dir / "tb.py").write_text("# candidate TB\n")
        return SimpleNamespace(module=spec.module_name, rtl_path=str(out_dir / "fabric.v"),
                               tb_path=str(out_dir / "tb.py"), ports=[], cached=False)
    monkeypatch.setattr(fabric, "generate_fabric", generate)
    result = await pg.materialize_primitive_node({"project_root": str(tmp_path), "current_block": _BLOCK})
    assert result["primitive_failed"]
    assert rtl.read_text() == "previous published candidate\n"
    assert "FABRIC_CONTRACT_MISMATCH" in (tmp_path / ".coresmith/blocks/fabric/previous_error.txt").read_text()


def test_detached_child_of_nested_executor_is_owned(tmp_path):
    # Each executor gives its child a distinct scope; ancestors still own cancellation.
    import time
    parent = tmp_path / "parent.py"
    child_pid = tmp_path / "child.pid"
    parent.write_text('''import subprocess,sys,time
from orchestrator.processes import popen
p=popen([sys.executable,'-c',"import os,time;open(sys.argv[1],'w').write(str(os.getpid()));time.sleep(30)",sys.argv[1]])
p.wait()
'''.replace('"import os,time;', '"import os,sys,time;'))
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2]))
    owner = processes.begin_owner("nested")
    process = processes.popen([sys.executable, str(parent), str(child_pid)], env=env)
    try:
        deadline = time.monotonic()+5
        while not child_pid.exists() and time.monotonic()<deadline:
            time.sleep(.01)
        assert child_pid.exists()
        pid = int(child_pid.read_text())
        processes.cancel(owner, grace_s=.2)
        assert process.poll() is not None
        deadline = time.monotonic()+3
        while time.monotonic()<deadline:
            stat = Path(f"/proc/{pid}/stat")
            if not stat.exists() or stat.read_text().split()[2] == 'Z':
                break
            time.sleep(.01)
        else:
            pytest.fail("detached nested tool survived cancellation")
    finally:
        processes.cancel(owner, grace_s=.1)


@pytest.mark.asyncio
async def test_fix_tb_checks_supplied_file_without_model_repair(tmp_path, monkeypatch):
    from unittest.mock import AsyncMock

    from orchestrator.langgraph import pipeline_graph as pg
    monkeypatch.setenv('CORESMITH_CONTRACT_PORT_GATE', '0')
    monkeypatch.setenv('CORESMITH_CONTRACT_CONFORMANCE_GATE', '0')
    rtl = tmp_path / 'core.v'
    rtl.write_text('module core(input clk); endmodule')
    tb = tmp_path / 'test_core.py'
    tb.write_text('# operator-supplied testbench\n')
    generate = AsyncMock(side_effect=AssertionError('must not generate'))
    repair = AsyncMock(side_effect=AssertionError('must not repair'))
    monkeypatch.setattr(pg, 'generate_testbench', generate)
    monkeypatch.setattr(pg, 'fix_testbench_errors', repair)
    monkeypatch.setattr(pg, '_vip_tb_lint', lambda *a: ['generated VIP available'])
    monkeypatch.setattr(pg, 'run_simulation', lambda *a, **k: {
        'passed': False, 'tests_total': 1, 'tests_failed': 1,
        'log': 'AttributeError: dut has no attribute missing_signal', 'log_path': ''})
    state = {'current_block': {'name': 'core', 'testbench': str(tb)}, 'rtl_path': str(rtl),
             'project_root': str(tmp_path), 'attempt': 1, 'human_response': {'action': 'fix_tb'}}
    result = await pg.generate_testbench_node(state)
    assert result['sim_passed'] is False
    assert tb.read_text() == '# operator-supplied testbench\n'
    generate.assert_not_called()
    repair.assert_not_called()
