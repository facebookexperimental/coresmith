# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Section 7a: engine git-SHA provenance stamped into run state + final report."""
from __future__ import annotations

import json

from orchestrator.langgraph import final_report as fr
from orchestrator.utils import engine_git_sha


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


# ---------------------------------------------------------------------------
# Engine source carries no benchmark-exercise vocabulary (run3-followups #5)
# ---------------------------------------------------------------------------
#
# Standing repo rule: engine source must be exercise-agnostic. Naming a specific
# benchmark exercise (or a specific reference design's internals) in a comment
# looks harmless and is not: it leaks the evaluation set into the engine, and it
# teaches every later reader -- human and model -- that this file is allowed to
# know which exercise it is running.
#
# Protocol-generic identifiers are FINE and deliberately not matched here: the
# QSPI contract fields in ``bfm_lib`` describe a bus protocol, not an exercise,
# and ``rasterise``/``TRIANGLES`` in the layout renderer are graphics primitives.
# The guard is scoped to the files this sweep cleaned, so it pins the fix rather
# than asserting a repo-wide invariant that other work would have to keep true.

_EXERCISE_WORDS = ("raster", "triangle", "h264", "h.264", "cavlc")

_SCRUBBED_FILES = (
    "orchestrator/langgraph/bfm_lib/maxgeo.py",
    "orchestrator/langgraph/bfm_lib/stimulus.py",
    "orchestrator/langgraph/drc_verdict.py",
    "orchestrator/langgraph/macro_prebind.py",
    "orchestrator/langgraph/tapeout_helpers.py",
    "orchestrator/langgraph/integration_helpers.py",
    "orchestrator/langchain/agents/contract_audit_agent.py",
    "orchestrator/langchain/agents/uarch_spec_generator.py",
    "orchestrator/architecture/model_integration.py",
)


def _engine_root():
    from pathlib import Path

    import orchestrator
    return Path(orchestrator.__file__).resolve().parent.parent


def test_scrubbed_engine_files_name_no_benchmark_exercise():
    root = _engine_root()
    offenders = []
    for rel in _SCRUBBED_FILES:
        p = root / rel
        assert p.exists(), rel
        for i, line in enumerate(
                p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            low = line.lower()
            for word in _EXERCISE_WORDS:
                if word in low:
                    offenders.append(f"{rel}:{i}: {line.strip()[:100]}")
    assert not offenders, (
        "benchmark-exercise vocabulary in engine source:\n  "
        + "\n  ".join(offenders)
    )
