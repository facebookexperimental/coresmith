"""WP-31: stale assembled integration artifacts are retired on a targeted re-entry."""
from __future__ import annotations

import json

from orchestrator.langgraph import pipeline_graph as pg


def _proj(tmp_path, assembled: bool):
    (tmp_path / ".coresmith").mkdir()
    (tmp_path / ".coresmith" / "integration_result.json").write_text(json.dumps(
        {"caravel_wrapper_assembled": assembled, "top_module": "user_project_wrapper"}))
    d = tmp_path / "rtl" / "integration"
    d.mkdir(parents=True)
    (d / "user_project_wrapper.v").write_text("module user_project_wrapper(); endmodule\n")
    (d / "user_project_wrapper_pads.v").write_text("module user_project_wrapper_pads(); endmodule\n")
    (d / "chip_top.v").write_text("module chip_top(); endmodule\n")
    return d


def test_assembled_wrapper_and_pads_are_retired(tmp_path):
    d = _proj(tmp_path, True)
    moved = pg._retire_derived_integration_artifacts(str(tmp_path))
    assert set(moved) == {"user_project_wrapper.v", "user_project_wrapper_pads.v"}
    assert not (d / "user_project_wrapper.v").exists() and (d / "chip_top.v").exists()
    assert list((d / "_stale").iterdir())


def test_self_assembled_wrapper_is_kept(tmp_path):
    d = _proj(tmp_path, False)
    moved = pg._retire_derived_integration_artifacts(str(tmp_path))
    assert moved == ["user_project_wrapper_pads.v"]
    assert (d / "user_project_wrapper.v").exists()


def test_no_integration_dir_is_noop(tmp_path):
    assert pg._retire_derived_integration_artifacts(str(tmp_path)) == []
