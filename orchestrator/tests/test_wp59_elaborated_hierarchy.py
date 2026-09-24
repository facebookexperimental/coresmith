"""Hierarchy evidence must come from elaboration, including inactive generates."""
import json
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from orchestrator.langchain.agents.integration_lead import assert_blocks_instantiated


@pytest.mark.skipif(not shutil.which("yosys"), reason="requires yosys")
@pytest.mark.parametrize("body", [
    "generate if (0) begin:g leaf u(); end endgenerate",
    "genvar i; generate for(i=0;i<0;i=i+1) begin:g leaf u(); end endgenerate",
])
def test_inactive_generate_is_not_instantiated(body):
    result = assert_blocks_instantiated(
        f"module chip_top(); {body} endmodule", {"leaf"},
        sources=["module leaf(); endmodule"], top_module="chip_top")
    assert result and "leaf" in result


def test_missing_elaborator_is_infrastructure_failure():
    with patch("shutil.which", return_value=None):
        result = assert_blocks_instantiated(
            "module chip_top(); leaf u(); endmodule", {"leaf"},
            sources=["module leaf(); endmodule"], top_module="chip_top")
    assert result and result.kind == "infrastructure_error"


@pytest.mark.parametrize("cells,ok", [({}, False), ({"u": {"type": "leaf"}}, True)])
def test_elaborator_output_is_the_decision(tmp_path, cells, ok):
    # Captured JSON shape from hierarchy -check/write_json; the source text
    # deliberately contains a leaf in both cases.
    design = {"modules": {"chip_top": {"cells": cells}, "leaf": {"cells": {}}}}

    def run(argv, **kwargs):
        script = Path(argv[-1]).read_text()
        output = script.split("write_json ", 1)[1].strip().strip('"')
        Path(output).write_text(json.dumps(design))
        from subprocess import CompletedProcess
        return CompletedProcess(argv, 0, "", "")

    with patch("shutil.which", return_value="/fake/yosys"), patch("subprocess.run", side_effect=run):
        result = assert_blocks_instantiated(
            "module chip_top(); leaf u(); endmodule", {"leaf"},
            sources=["module leaf(); endmodule"], top_module="chip_top")
    assert (result is None) is ok


def test_top_must_be_explicit():
    result = assert_blocks_instantiated(
        "module chip_top(); endmodule module orphan(); leaf u(); endmodule", {"leaf"},
        sources=["module leaf(); endmodule"])
    assert result and "top" in result.lower()


@pytest.mark.skipif(not shutil.which("yosys"), reason="requires yosys")
def test_memory_backed_block_uses_candidate_library(tmp_path):
    top = tmp_path / "chip_top.v"
    leaf = tmp_path / "leaf.v"
    top.write_text("module chip_top(input clk, output [7:0] q); leaf u(clk, q); endmodule")
    leaf.write_text('''module leaf(input clk, output [7:0] q);
      cs_fpmem_1rw1r #(.WIDTH(8), .DEPTH(4)) mem (
        .clk(clk), .ce0(1'b1), .we0(1'b0), .addr0(2'b0),
        .wdata0(8'b0), .rdata0(q), .ce1(1'b0), .addr1(2'b0), .rdata1());
    endmodule''')
    kwargs = dict(source_paths=[str(top), str(leaf)],
                  top_module="chip_top", project_root=tmp_path)
    assert assert_blocks_instantiated(top.read_text(), {"leaf"}, **kwargs) is None
    missing = assert_blocks_instantiated(top.read_text(), {"leaf", "absent"}, **kwargs)
    assert missing and "absent" in missing and "does NOT instantiate" in missing


def test_missing_selected_source_is_infrastructure_failure(tmp_path):
    result = assert_blocks_instantiated(
        "module chip_top(); endmodule", {"leaf"},
        source_paths=[str(tmp_path / "missing.v")], top_module="chip_top",
        project_root=tmp_path)
    assert result and result.kind == "infrastructure_error"
