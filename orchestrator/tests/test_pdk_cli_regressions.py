# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Regressions for the Sky130 ``coresmith tool`` wrappers."""

from __future__ import annotations

from pathlib import Path

from orchestrator.langgraph import pipeline_helpers
from orchestrator.pdk.base import ToolRequest
from orchestrator.pdk.deployments import sky130


def _request(tmp_path: Path, verb: str, rtls: list[Path], **kwargs) -> ToolRequest:
    return ToolRequest(
        verb=verb,
        design="top",
        inputs={"rtl": rtls[0]},
        params={"rtls": [str(p) for p in rtls]},
        **kwargs,
    )


def test_run_synth_forwards_all_rtls_output_dir_and_timeout(tmp_path, monkeypatch):
    top = tmp_path / "top.v"
    leaf = tmp_path / "leaf.v"
    top.write_text("module top; leaf u(); endmodule\n")
    leaf.write_text("module leaf; endmodule\n")
    out_dir = tmp_path / "requested"
    seen = {}

    def fake_synthesize(block, rtl_path, **kwargs):
        seen.update(block=block, rtl_path=rtl_path, **kwargs)
        out_dir.mkdir()
        netlist = out_dir / "top_netlist.v"
        netlist.write_text("module top; endmodule\n")
        report = out_dir / "top_report.txt"
        report.write_text("Number of cells: 1\n")
        return {
            "success": True,
            "netlist_path": str(netlist),
            "report_path": str(report),
            "log": "",
        }

    monkeypatch.setattr(pipeline_helpers, "synthesize_block", fake_synthesize)
    req = _request(tmp_path, "run_synth", [top, leaf],
                   out_dir=out_dir, timeout_s=37)
    result = sky130.RunSynthYosys(object()).run(req)

    assert result.ok
    assert seen["rtl_path"] == str(top)
    assert seen["extra_rtl_paths"] == [str(leaf)]
    assert seen["output_dir"] == out_dir
    assert seen["timeout_s"] == 37


def test_run_lint_forwards_all_rtls_and_timeout(tmp_path, monkeypatch):
    top = tmp_path / "top.v"
    leaf = tmp_path / "leaf.v"
    top.write_text("module top; leaf u(); endmodule\n")
    leaf.write_text("module leaf; endmodule\n")
    seen = {}

    def fake_lint(rtl_path, block_name, **kwargs):
        seen.update(rtl_path=rtl_path, block_name=block_name, **kwargs)
        return {"clean": True}

    monkeypatch.setattr(pipeline_helpers, "lint_rtl", fake_lint)
    result = sky130.RunLintVerilator(object()).run(
        _request(tmp_path, "run_lint", [top, leaf], timeout_s=19))

    assert result.ok
    assert seen["rtl_path"] == str(top)
    assert seen["extra_rtl_paths"] == [str(leaf)]
    assert seen["timeout_s"] == 19


def test_script_synth_fails_closed_when_no_netlist_is_written(tmp_path, monkeypatch):
    script = tmp_path / "stats_only.ys"
    script.write_text("stat\n")
    out_dir = tmp_path / "out"
    monkeypatch.setattr(sky130, "_run_cmd", lambda *args, **kwargs: (0, "", "", ""))

    result = sky130.RunSynthYosys(object()).run(ToolRequest(
        verb="run_synth", design="top", inputs={"script": script},
        out_dir=out_dir,
    ))

    assert result.tool_ok
    assert not result.ok
    check = next(c for c in result.checks if c.name == "netlist")
    assert check.status == "not_run" and check.blocking
    assert result.artifacts == {}


def test_script_synth_does_not_accept_a_stale_netlist(tmp_path, monkeypatch):
    script = tmp_path / "stats_only.ys"
    script.write_text("stat\n")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    stale = out_dir / "top_netlist.v"
    stale.write_text("module stale; endmodule\n")
    monkeypatch.setattr(sky130, "_run_cmd", lambda *args, **kwargs: (0, "", "", ""))

    result = sky130.RunSynthYosys(object()).run(ToolRequest(
        verb="run_synth", design="top", inputs={"script": script},
        out_dir=out_dir,
    ))

    assert not result.ok
    assert result.artifacts == {}


def test_script_synth_fresh_fallback_is_not_hidden_by_stale_named_netlist(
        tmp_path, monkeypatch):
    script = tmp_path / "write_fallback.ys"
    script.write_text("write_verilog foo.v\n")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    stale = out_dir / "top_netlist.v"
    stale.write_text("module stale; endmodule\n")

    def fake_run(*args, **kwargs):
        (out_dir / "foo.v").write_text("module top; endmodule\n")
        return 0, "", "", ""

    monkeypatch.setattr(sky130, "_run_cmd", fake_run)
    result = sky130.RunSynthYosys(object()).run(ToolRequest(
        verb="run_synth", design="top", inputs={"script": script},
        out_dir=out_dir,
    ))

    assert result.ok
    assert result.artifacts["netlist"] == out_dir / "foo.v"


def test_magic_drc_rejects_blank_on_clean_list_count_before_tool_run(
        tmp_path, monkeypatch):
    from orchestrator.langgraph import backend_helpers

    script = tmp_path / "drc.tcl"
    script.write_text(
        "# set drc_count [drc listall count total]\n"
        "set drc_count [drc listall count]\n"
        "set drc_result [drc listall why]\n"
    )

    def must_not_run(*args, **kwargs):
        raise AssertionError("invalid DRC script reached Magic")

    monkeypatch.setattr(backend_helpers, "run_magic", must_not_run)
    result = sky130.RunDrcMagic(object()).run(ToolRequest(
        verb="run_drc", design="top", inputs={"script": script},
        out_dir=tmp_path,
    ))

    assert result.ok is False
    assert result.tool_ok is False
    assert result.checks[0].name == "script"
    assert "drc listall count total" in result.checks[0].details
    assert "blank when clean" in result.checks[0].details


def test_magic_drc_script_gate_requires_detailed_report(tmp_path):
    script = tmp_path / "drc.tcl"
    script.write_text("set drc_count [drc listall count total]\n")

    error = sky130._magic_drc_script_error(script)

    assert "drc listall why" in error


def test_helpers_include_extra_sources_and_explicit_limits(tmp_path, monkeypatch):
    top = tmp_path / "top.v"
    leaf = tmp_path / "leaf.v"
    top.write_text("module top; leaf u(); endmodule\n")
    leaf.write_text("module leaf; endmodule\n")
    synth_out = tmp_path / "synth-out"
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(pipeline_helpers, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("CORESMITH_SYNTH_GENERIC", "1")
    monkeypatch.setattr(pipeline_helpers, "run_process", fake_run)
    pipeline_helpers.synthesize_block(
        {"name": "top"}, str(top), extra_rtl_paths=[str(leaf)],
        output_dir=synth_out, timeout_s=23,
    )
    script_text = (synth_out / "synth_top.ys").read_text()
    assert f'"{top}"' in script_text and f'"{leaf}"' in script_text
    assert calls[-1][1]["timeout"] == 23

    calls.clear()
    pipeline_helpers.lint_rtl(
        str(top), "top", extra_rtl_paths=[str(leaf)], timeout_s=17)
    assert calls[-1][0][-2:] == [str(top), str(leaf)]
    assert calls[-1][1]["timeout"] == 17
