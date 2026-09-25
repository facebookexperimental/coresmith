"""WP-67: authoritative simulation builds cannot inherit agent products."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from orchestrator.langgraph import integration_helpers as ih
from orchestrator.langgraph import pipeline_helpers as ph
from orchestrator.tests.candidate_fixtures import adopt

_PRODUCTS = (
    "verilator.o", "verilated_cov.o", "support.a", "support.d", "rules.mk",
    "Vtop", "Vtop.cpp", "Vtop.h", "verilator.cpp", "dump.vcd", "dump.fst",
    "results.xml", "sim_build/Vtop", "obj_dir/helper.o",
)


def _seed_products(sim_dir):
    for name in _PRODUCTS:
        path = sim_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("stale product")


def _inputs(root):
    top = root / "chip_top.v"
    top.write_text("module chip_top(input clk); endmodule\n")
    tb = root / "test_chip.py"
    tb.write_text("import cocotb\n@cocotb.test()\nasync def smoke(dut):\n    pass\n")
    inputs = root / "inputs"
    inputs.mkdir()
    (inputs / "image.memh").write_text("ab\n")
    return top, tb


@pytest.mark.parametrize("scope,bounded", [
    ("integration", False), ("validation", False), ("validation", True),
])
def test_authoritative_build_recreates_scope(tmp_path, monkeypatch, capsys, scope, bounded):
    top, tb = _inputs(tmp_path)
    if bounded:
        tb.write_text(tb.read_text().replace("smoke", "test_wavekit_bounded_semantic_trace"))
    adopt(tmp_path, top)
    log_dir = tmp_path / ".coresmith" / "step_logs"
    monkeypatch.setattr(ph, "_LOG_DIR", log_dir)
    monkeypatch.setenv("CORESMITH_INTEGRATION_SIM_TIMEOUT_S", "100")
    monkeypatch.setenv("CORESMITH_INTEGRATION_SIM_TIMEOUT_CAP_S", "1000")
    prior_log = log_dir / "integration" / f"{scope}_sim_attempt1.log"
    prior_log.parent.mkdir(parents=True)
    prior_log.write_text("prior attempt evidence")
    sim_dir = tmp_path / "sim_build" / scope
    _seed_products(sim_dir)
    # Unknown files and old staged directories must go too, not just a glob list.
    (sim_dir / "foreign_helper.cc").write_text("stale generated C++")
    (sim_dir / "inputs").mkdir()
    (sim_dir / "inputs" / "old.memh").write_text("stale input")
    (sim_dir / "test_old.py").write_text("stale testbench")
    timeout_state = '{"timeouts": 2}\n'
    (sim_dir / "sim_timeout_state.json").write_text(timeout_state)
    makefile = ih._compose_dv_makefile(scope, tb.read_text(), str(top), "chip_top", tb.stem)
    ph.apply_build_fingerprint(sim_dir, makefile, [str(top)])
    fingerprint = (sim_dir / ".build_fingerprint").read_bytes()
    _seed_products(sim_dir)
    seen = []

    class Make:
        pid = 12345
        returncode = 0

        def __init__(self, cmd, **kwargs):
            assert Path(cmd[2]) == sim_dir
            seen.append({p.name for p in sim_dir.iterdir()})
            assert seen[-1] == {
                "Makefile", tb.name, "inputs", ".build_fingerprint", "sim_timeout_state.json",
            }
            assert (sim_dir / "Makefile").read_text() == makefile
            assert (sim_dir / tb.name).read_text() == tb.read_text()
            assert (sim_dir / ".build_fingerprint").read_bytes() == fingerprint
            assert (sim_dir / "sim_timeout_state.json").read_text() == timeout_state
            assert (sim_dir / "inputs" / "image.memh").read_text() == "ab\n"
            assert not (sim_dir / "inputs" / "old.memh").exists()
            (sim_dir / "results.xml").write_text(
                '<testsuite><testcase name="smoke"/></testsuite>')

        def communicate(self, timeout):
            assert timeout == 225  # Retry bookkeeping survived the fresh build.
            return "** TESTS=1 PASS=1 FAIL=0 **", ""

    monkeypatch.setattr("orchestrator.langchain.agents.coresmith_llm._reap_process_group",
                        lambda *a, **k: None)
    monkeypatch.setattr(ih.subprocess, "Popen", Make)
    for attempt in (2, 3):
        result = ih.run_integration_simulation(
            "chip", str(top), {}, str(tb), attempt=attempt,
            sim_scope=scope, project_root=tmp_path)
        assert result["passed"]
        assert result["executed_cases"] == ["smoke"]
        assert Path(result["log_path"]).name == f"{scope}_sim_attempt{attempt}.log"
        assert seen[-1] == {
            "Makefile", tb.name, "inputs", ".build_fingerprint", "sim_timeout_state.json",
        }
        _seed_products(sim_dir)
    assert prior_log.read_text() == "prior attempt evidence"
    assert (prior_log.parent / f"{scope}_sim_attempt2.log").exists()
    assert "verilator.o" in capsys.readouterr().out


@pytest.mark.parametrize("scope", ["integration", "validation"])
@pytest.mark.parametrize("has_waveform", [False, True])
def test_sim_timeout_returns_partial_evidence(tmp_path, monkeypatch, scope, has_waveform):
    top, tb = _inputs(tmp_path)
    adopt(tmp_path, top)
    monkeypatch.setattr(ph, "_LOG_DIR", tmp_path / ".coresmith" / "step_logs")
    monkeypatch.setenv("CORESMITH_INTEGRATION_SIM_TIMEOUT_S", "100")
    monkeypatch.setenv("CORESMITH_INTEGRATION_SIM_TIMEOUT_CAP_S", "1000")
    sim_dir = tmp_path / "sim_build" / scope
    reaped = []

    class TimedOutMake:
        pid = 12345

        def __init__(self, cmd, **kwargs):
            self.cmd = cmd
            if has_waveform:
                (sim_dir / "dump.vcd").write_text("partial waveform")

        def communicate(self, timeout):
            raise subprocess.TimeoutExpired(
                self.cmd, timeout, output=b"last cycle: 37", stderr=b"waiting for ready")

    monkeypatch.setattr(ih.subprocess, "Popen", TimedOutMake)
    monkeypatch.setattr("orchestrator.langchain.agents.coresmith_llm._reap_process_group",
                        lambda proc, pid, **kwargs: reaped.append((proc, pid)))
    result = ih.run_integration_simulation(
        "chip", str(top), {}, str(tb), sim_scope=scope, project_root=tmp_path)

    assert result["passed"] is False
    assert result["sim_timed_out"] is True
    assert result["sim_timeout_s"] == 100
    assert "No functional verdict" in result["log"]
    assert "last cycle: 37" in result["log"]
    assert "waiting for ready" in Path(result["log_path"]).read_text()
    assert result["vcd_path"] == (str(sim_dir / "dump.vcd") if has_waveform else "")
    assert json.loads((sim_dir / "sim_timeout_state.json").read_text()) == {"timeouts": 1}
    assert len(reaped) == 1 and reaped[0][1] == TimedOutMake.pid


def test_clear_build_products_keeps_inputs(tmp_path):
    _seed_products(tmp_path)
    kept = {
        "Makefile": "SIM = verilator", "test_chip.py": "# TB", "image.memh": "ab",
        ".build_fingerprint": "fingerprint", "sim_timeout_state.json": '{"timeouts": 2}',
    }
    for name, content in kept.items():
        (tmp_path / name).write_text(content)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "source.o").write_text("staged input must not be traversed")
    # Unlink a generated directory symlink without deleting its target.
    (tmp_path / "Vlinked").symlink_to(inputs, target_is_directory=True)
    assert ph.clear_build_products(tmp_path)
    assert not any((tmp_path / name).exists() for name in _PRODUCTS)
    assert not (tmp_path / "Vlinked").is_symlink()
    assert {name: (tmp_path / name).read_text() for name in kept} == kept
    assert (inputs / "source.o").read_text() == "staged input must not be traversed"
    assert ph.clear_build_products(tmp_path) is False


@pytest.mark.parametrize("change", ["source", "coverage"])
def test_block_fingerprint_reuses_then_cleans_top_level(tmp_path, monkeypatch, change):
    top, tb = _inputs(tmp_path)
    monkeypatch.setattr(ph, "_LOG_DIR", tmp_path / ".coresmith" / "step_logs")
    monkeypatch.setenv("CORESMITH_COVERAGE", "1")
    monkeypatch.setenv("CORESMITH_LINE_COV_GATE", "0")
    seen = []

    class Make:
        pid = 12345
        returncode = 0

        def __init__(self, cmd, **kwargs):
            seen.append((Path(cmd[2]) / "verilator.o").exists())

        def communicate(self, timeout):
            return "** TESTS=1 PASS=1 FAIL=0 **", ""

    monkeypatch.setattr("orchestrator.langchain.agents.coresmith_llm._reap_process_group",
                        lambda *a, **k: None)
    monkeypatch.setattr(ph.subprocess, "Popen", Make)
    block = {"name": "chip_top"}
    ph.run_simulation(block, str(top), str(tb), project_root=tmp_path)
    sim_dir = tmp_path / "sim_build" / "chip_top"
    _seed_products(sim_dir)
    ph.run_simulation(block, str(top), str(tb), project_root=tmp_path)
    if change == "source":
        top.write_text(top.read_text() + "// changed compile input\n")
    else:
        monkeypatch.setenv("CORESMITH_COVERAGE", "0")
    ph.run_simulation(block, str(top), str(tb), project_root=tmp_path)
    assert seen == [False, True, False]


@pytest.mark.skipif(
    any(shutil.which(tool) is None for tool in ("make", "verilator", "cocotb-config", "g++")),
    reason="requires local Verilator/cocotb build tools",
)
def test_real_coverage_objects_cannot_contaminate_authoritative_build(tmp_path, monkeypatch):
    top, tb = _inputs(tmp_path)
    top.write_text("module chip_top(input a, output y); assign y = ~a; endmodule\n")
    tb.write_text(
        "import cocotb\nfrom cocotb.triggers import Timer\n"
        "@cocotb.test()\nasync def smoke(dut):\n"
        "    for a in (0, 1):\n"
        "        dut.a.value = a\n"
        "        await Timer(1, unit='ns')\n"
        "        assert int(dut.y.value) == 1 - a\n")
    adopt(tmp_path, top)
    monkeypatch.setattr(ph, "_LOG_DIR", tmp_path / ".coresmith" / "step_logs")
    monkeypatch.setenv("CORESMITH_COVERAGE", "1")
    monkeypatch.setenv("CORESMITH_LINE_COV_GATE", "0")
    instrumented = ph.run_simulation(
        {"name": "chip_top"}, str(top), str(tb), project_root=tmp_path)
    assert instrumented["passed"], instrumented["log"]
    coverage_log = Path(instrumented["log_path"]).read_text()
    assert "--coverage" in coverage_log
    assert set(re.findall(r"-DVM_COVERAGE=(\d)", coverage_log)) == {"1"}

    sim_dir = tmp_path / "sim_build" / "integration"
    # Reproduce a direct-in-scope coverage build using real compiled objects.
    shutil.copytree(tmp_path / "sim_build" / "chip_top" / "sim_build", sim_dir)
    assert (sim_dir / "verilator.o").is_file()
    assert (sim_dir / "verilated_cov.o").is_file()
    result = ih.run_integration_simulation(
        "chip", str(top), {}, str(tb), project_root=tmp_path)
    assert result["passed"], result["log"]
    assert result["tests_passed"] == 1
    build_log = Path(result["log_path"]).read_text()
    assert set(re.findall(r"-DVM_COVERAGE=(\d)", build_log)) == {"0"}
    assert "../verilator.o" not in build_log
    assert not (sim_dir / "verilator.o").exists()


@pytest.mark.parametrize("prompt", ["integration_testbench", "validation_dv", "chip_lead"])
def test_prompt_requires_agent_scratch_builds(prompt):
    text = (Path(__file__).parents[1] / "langchain" / "prompts" / f"{prompt}.md").read_text()
    assert "sim_build/agent_<name>/" in text
    assert "never `sim_build/integration` or `sim_build/validation`" in text


def test_chip_verify_lock_survives_scope_recreation(tmp_path, monkeypatch):
    import fcntl
    import shutil

    from orchestrator.harness.verify import verify_chip

    top, tb = _inputs(tmp_path)
    state = tmp_path / ".coresmith"
    state.mkdir()
    (state / "integration_result.json").write_text(json.dumps({
        "top_rtl_path": str(top), "tb_path": str(tb),
    }))

    def simulate(*args, sim_scope, **kwargs):
        sim_dir = tmp_path / "sim_build" / sim_scope
        shutil.rmtree(sim_dir)
        sim_dir.mkdir()
        locks = list((tmp_path / "sim_build").rglob("*.lock"))
        assert len(locks) == 1, "The held lock must live outside the disposable scope"
        with locks[0].open("w") as contender:
            with pytest.raises(BlockingIOError):
                fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return {"passed": True}

    monkeypatch.setattr(ih, "run_integration_simulation", simulate)
    assert verify_chip(tmp_path, record_source="gate").passed
