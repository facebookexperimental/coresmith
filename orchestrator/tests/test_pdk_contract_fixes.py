# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Focused regressions for measured Sky130 tool/prompt contracts."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace


def test_magic_drc_notes_use_supported_report_capture():
    from orchestrator.pdk.deployments.sky130 import RunDrcMagic

    notes = RunDrcMagic(SimpleNamespace()).prompt_notes()
    assert "set drc_result [drc listall why]" in notes
    assert "set drc_count [drc listall count total]" in notes
    assert "drc listall count` returns a per-cell Tcl list" in notes
    assert "non-integer or negative" in notes
    assert "drc listall why <report>" not in notes
    assert "magic-gds-write" in notes
    assert "written by the deployment" not in notes


def test_all_magic_drc_prompts_and_template_require_scalar_total():
    root = Path(__file__).parents[1]
    paths = (
        root / "langchain" / "prompts" / "backend_drc_llm.md",
        root / "langchain" / "prompts" / "tapeout_wrapper_drc.md",
        root / "pdk_templates" / "sky130" / "drc.tcl",
    )
    for path in paths:
        text = path.read_text()
        assert "drc listall count total" in text, path
        assert "set drc_count [drc listall count]\n" not in text, path


def test_pnr_prompt_keeps_pdn_pin_layer_connected():
    prompt = (Path(__file__).parents[1] / "langchain" / "prompts"
              / "backend_pnr_llm.md").read_text()
    assert "Do not\nchange only `-pins` to met5" in prompt
    assert "add_pdn_stripe -grid stdcell_grid -layer met5" in prompt
    assert "add_pdn_connect -grid stdcell_grid -layers {{met4 met5}}" in prompt
    assert "CORESMITH_PNR_TOOL_TIMEOUT:-1650" in prompt
    assert "1650-second inner default" in prompt
    assert "default 1800-second" in prompt


def test_prompts_exclude_clock_from_input_delay():
    prompt_dir = Path(__file__).parents[1] / "langchain" / "prompts"
    for name in ("backend_synth_llm.md", "sdc_generator.md"):
        text = (prompt_dir / name).read_text()
        assert "[all_inputs -no_clocks]" in text
        assert "-clock clk [all_inputs]" not in text


def test_all_generated_sdc_paths_use_opensta_clock_exclusion():
    root = Path(__file__).parents[1]
    for relative in (
        "langgraph/pipeline_helpers.py",
        "langgraph/backend_helpers.py",
        "langgraph/backend_graph.py",
        "mcp_server.py",
    ):
        text = (root / relative).read_text()
        assert "[all_inputs -no_clocks]" in text
        assert "remove_from_collection" not in text


def test_reference_reports_density_fill_not_run():
    ref = (Path(__file__).parents[1] / "pdk_templates" / "sky130"
           / "pnr_reference.tcl").read_text()
    assert "density_fill -rules $tech_lef" not in ref
    assert "Density fill: NOT RUN" in ref
    assert "Density fill done" not in ref


def test_generated_pnr_has_connected_default_pdn_and_honest_fill(tmp_path,
                                                                  monkeypatch):
    pdk_root = tmp_path / "pdk"
    (pdk_root / "sky130A").mkdir(parents=True)
    monkeypatch.setenv("PDK_ROOT", str(pdk_root))
    from orchestrator.pdk.deployments.sky130 import Sky130Deployment

    tcl = Sky130Deployment().tools()["run_pnr"].render_pnr_tcl(
        "blk", "blk", "/tmp/blk.v", "/tmp/blk.sdc", gate_count=300,
    )
    assert "-pins met4" in tcl
    assert "add_pdn_stripe -grid stdcell_grid -layer met4" in tcl
    assert "density_fill -rules $tech_lef" not in tcl
    assert "Density fill: NOT RUN" in tcl


def test_renderer_requests_offscreen_qt(tmp_path, monkeypatch):
    from orchestrator.langgraph import backend_helpers as bh

    source = tmp_path / "layout.def"
    output = tmp_path / "layout.png"
    source.write_text("VERSION 5.8 ;\n")
    captured = {}

    def fake_run(cmd, **kwargs):
        captured.update(kwargs)
        output.write_bytes(b"png")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)
    monkeypatch.setattr(bh.subprocess, "run", fake_run)
    assert Path(bh.RENDER_SCRIPT).is_file()
    assert bh.render_layout_image(str(source), str(output)) is True
    assert captured["env"]["QT_QPA_PLATFORM"] == "offscreen"
