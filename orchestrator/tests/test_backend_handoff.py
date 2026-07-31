# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""The frontend -> backend handoff, and the testbench the gate-sim grades.

The chip_top gate-sim is the only step that ever simulates the artifact that
becomes silicon. It lives in the backend graph, and until now the backend graph
was reachable only from an MCP client or a hand-written driver: a daemon run
could reach ``pipeline_done`` and stop, with nobody to press the next button.

These tests cover the two things that made the handoff untrustworthy:

  * the testbench lookup -- the gate looked up ``test_<design_name>.py``, but the
    testbench is named after the FRONTEND design while ``design_name`` is the top
    MODULE (``user_project_wrapper`` on a Caravel chip). The exact-name lookup
    then found nothing and reported ``not_run``.

The lookup tests build a real directory and call the real function.
"""

from __future__ import annotations

from pathlib import Path

from orchestrator.langgraph.backend_graph import find_integration_tb


# ---------------------------------------------------------------------------
# Integration-testbench lookup
# ---------------------------------------------------------------------------
def _tb_dir(root: Path) -> Path:
    d = root / "tb" / "integration"
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_exact_design_name_wins(tmp_path):
    d = _tb_dir(tmp_path)
    (d / "test_user_project_wrapper.py").write_text("# chip tb\n")
    (d / "test_raster_top.py").write_text("# frontend-named tb\n")
    tb, note = find_integration_tb(tmp_path, "user_project_wrapper")
    assert tb.endswith("test_user_project_wrapper.py")
    assert note == ""


def test_sim_build_copy_is_preferred_over_tb_dir(tmp_path):
    d = _tb_dir(tmp_path)
    (d / "test_chip.py").write_text("# tb dir\n")
    sb = tmp_path / "sim_build" / "integration"
    sb.mkdir(parents=True)
    (sb / "test_chip.py").write_text("# sim_build copy\n")
    tb, _ = find_integration_tb(tmp_path, "chip")
    assert "sim_build" in tb


def test_falls_back_to_the_only_testbench_present(tmp_path):
    """THE BUG: the TB is named after the frontend design, design_name is the
    top module. One candidate is not a guess -- it is the chip's testbench."""
    d = _tb_dir(tmp_path)
    (d / "test_raster2d_accelerator_top.py").write_text("# frontend-named\n")
    tb, note = find_integration_tb(tmp_path, "user_project_wrapper")
    assert tb.endswith("test_raster2d_accelerator_top.py")
    assert "no test_user_project_wrapper.py" in note
    assert "user_project_wrapper" in note


def test_refuses_to_guess_between_several(tmp_path):
    """Picking by sort order is how a gate grades the wrong stimulus and calls
    the result a pass."""
    d = _tb_dir(tmp_path)
    (d / "test_a_top.py").write_text("# a\n")
    (d / "test_b_top.py").write_text("# b\n")
    tb, note = find_integration_tb(tmp_path, "user_project_wrapper")
    assert tb == ""
    assert "AMBIGUOUS" in note
    assert "test_a_top.py" in note and "test_b_top.py" in note


def test_no_testbench_at_all_is_reported_not_silently_passed(tmp_path):
    tb, note = find_integration_tb(tmp_path, "chip_top")
    assert tb == ""
    assert "no integration-DV testbench found" in note
