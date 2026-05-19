import json


def test_read_back_json_recovers_codex_call_artifact(tmp_path):
    from orchestrator.utils import read_back_json

    target = tmp_path / ".socmate" / "block_diagram.json"
    isolated = tmp_path / "codex-call-abc123" / ".socmate" / "block_diagram.json"
    isolated.parent.mkdir(parents=True)
    isolated.write_text(json.dumps({"blocks": [{"name": "stage0"}]}))

    result, ok = read_back_json(
        target,
        str(target),
        {"blocks": [], "connections": [], "reasoning": "", "questions": []},
        context="block_diagram",
    )

    assert ok is True
    assert result["blocks"] == [{"name": "stage0"}]
    assert json.loads(target.read_text())["blocks"] == [{"name": "stage0"}]


def test_uarch_markdown_recovery_prefers_codex_call_spec(tmp_path):
    from pathlib import Path

    from orchestrator.langgraph.pipeline_helpers import (
        _choose_generated_markdown,
        _recover_codex_call_artifact,
    )

    rich_spec = (
        "## 1. Block Overview\n"
        "output_fifo stores bytes.\n\n"
        "## 2. Interface Specification\n"
        "| Port | Direction | Width |\n"
        "|---|---:|---:|\n"
        "| aclk | Input | 1 |\n"
        + ("Detailed interface and behavior. " * 30)
    )
    isolated = tmp_path / "codex-call-abc123" / "arch" / "uarch_specs" / "output_fifo.md"
    isolated.parent.mkdir(parents=True)
    isolated.write_text(rich_spec)

    recovered = _recover_codex_call_artifact(
        tmp_path,
        Path("arch") / "uarch_specs" / "output_fifo.md",
    )

    assert recovered == rich_spec.strip()
    assert _choose_generated_markdown([
        "Created the microarchitecture spec at arch/uarch_specs/output_fifo.md",
        recovered,
    ]) == rich_spec.strip()


def test_rtl_recovery_finds_codex_call_verilog(tmp_path):
    from orchestrator.langchain.agents.rtl_generator import _recover_codex_artifact

    rtl = "module foo(input wire clk); endmodule\n"
    isolated = tmp_path / "codex-call-rtl" / "rtl" / "foo.v"
    isolated.parent.mkdir(parents=True)
    isolated.write_text(rtl)

    assert _recover_codex_artifact(str(tmp_path), "rtl/foo.v") == rtl


def test_testbench_recovery_finds_codex_call_cocotb(tmp_path):
    from orchestrator.langchain.agents.testbench_generator import _recover_codex_artifact

    tb = "import cocotb\n\n@cocotb.test()\nasync def test_smoke(dut):\n    pass\n"
    target = tmp_path / "tb" / "cocotb" / "test_foo.py"
    isolated = tmp_path / "codex-call-tb" / "tb" / "cocotb" / "test_foo.py"
    isolated.parent.mkdir(parents=True)
    isolated.write_text(tb)

    assert _recover_codex_artifact(str(target)) == tb
