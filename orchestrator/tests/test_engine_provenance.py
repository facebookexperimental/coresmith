# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Section 7a: engine git-SHA provenance stamped into run state + final report."""
from __future__ import annotations

import json
from pathlib import Path

from orchestrator.utils import engine_git_sha
from orchestrator.langgraph import final_report as fr


def test_engine_git_sha_returns_str():
    sha = engine_git_sha()
    assert isinstance(sha, str)
    # In this checkout it is a short hex sha; tolerate "" in a non-git env.
    assert sha == "" or all(c in "0123456789abcdef" for c in sha)


def test_stamp_engine_sha_writes_and_is_stable(tmp_path):
    from orchestrator.langgraph.pipeline_graph import _stamp_engine_sha
    _stamp_engine_sha(str(tmp_path))
    p = tmp_path / ".coresmith" / "engine_sha.json"
    assert p.exists()
    rec = json.loads(p.read_text())
    assert "sha" in rec and rec["changed"] is False
    # second stamp with same live sha -> still not changed
    _stamp_engine_sha(str(tmp_path))
    assert json.loads(p.read_text())["changed"] is False


def test_stamp_detects_mid_run_change(tmp_path):
    from orchestrator.langgraph.pipeline_graph import _stamp_engine_sha
    _stamp_engine_sha(str(tmp_path))
    p = tmp_path / ".coresmith" / "engine_sha.json"
    # simulate that the run STARTED under a different sha than the live one
    rec = json.loads(p.read_text())
    rec["sha"] = "deadbeefdead"
    p.write_text(json.dumps(rec))
    _stamp_engine_sha(str(tmp_path))
    rec2 = json.loads(p.read_text())
    # only asserts change-detection when a live sha is resolvable
    if engine_git_sha():
        assert rec2["changed"] is True
        assert rec2["changes"] and rec2["changes"][0]["from"] == "deadbeefdead"


def test_final_report_includes_engine_sha(tmp_path):
    (tmp_path / ".coresmith").mkdir(parents=True)
    (tmp_path / ".coresmith" / "engine_sha.json").write_text(
        json.dumps({"sha": "abc123abc123", "changed": True,
                    "changes": [{"from": "x", "to": "abc123abc123"}]}))
    report = fr.build_final_report({}, str(tmp_path))
    assert report["engine_sha"] == "abc123abc123"
    assert report["engine_sha_changed_mid_run"] is True
    md = fr.render_markdown(report)
    assert "abc123abc123" in md and "CHANGED MID-RUN" in md
