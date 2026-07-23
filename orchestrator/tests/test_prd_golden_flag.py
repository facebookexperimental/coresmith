# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""PRD no-golden verification-risk flag."""
from __future__ import annotations

import json

from orchestrator.architecture.specialists.prd_spec import (
    _NO_GOLDEN_FLAG,
    _annotate_golden_risk,
    _golden_available,
)


def test_no_golden_detected_and_flagged(tmp_path):
    (tmp_path / "inputs").mkdir()
    (tmp_path / "inputs" / "requirements.md").write_text("a jpeg codec")
    avail, paths = _golden_available(str(tmp_path), "jpeg")
    assert avail is False and paths == []
    prd: dict = {"summary": "x"}
    target = tmp_path / ".coresmith" / "prd_spec.json"
    target.parent.mkdir(parents=True)
    _annotate_golden_risk(prd, str(tmp_path), "jpeg", target)
    assert prd["golden_model_available"] is False
    assert any(f["id"] == _NO_GOLDEN_FLAG and f["severity"] == "high"
               for f in prd["risk_flags"])
    # persisted to disk for PRD review
    on_disk = json.loads(target.read_text())["prd"]
    assert on_disk["golden_model_available"] is False


def test_golden_present_no_flag(tmp_path):
    (tmp_path / "inputs").mkdir()
    (tmp_path / "inputs" / "codec_golden.py").write_text("def encode(x): return x\n")
    avail, paths = _golden_available(str(tmp_path), "jpeg")
    assert avail is True and len(paths) == 1
    prd: dict = {}
    target = tmp_path / ".coresmith" / "prd_spec.json"
    target.parent.mkdir(parents=True)
    _annotate_golden_risk(prd, str(tmp_path), "jpeg", target)
    assert prd["golden_model_available"] is True
    assert "risk_flags" not in prd  # no caution when golden present


def test_testbench_py_not_counted_as_golden(tmp_path):
    (tmp_path / "inputs").mkdir()
    (tmp_path / "inputs" / "test_foo.py").write_text("x")
    avail, _ = _golden_available(str(tmp_path), "x")
    assert avail is False


def test_dangling_requirements_reference_not_golden(tmp_path):
    # requirements name a golden path that was never created -> NOT available
    avail, _ = _golden_available(str(tmp_path), "use model/jpeg_golden.py bit-exact")
    assert avail is False


def test_requirements_reference_that_exists_is_golden(tmp_path):
    (tmp_path / "examples").mkdir()
    g = tmp_path / "examples" / "codec_golden.py"
    g.write_text("def encode(x): return x\n")
    req = "grounded in the golden at `examples/codec_golden.py`"
    avail, paths = _golden_available(str(tmp_path), req)
    assert avail is True and any("codec_golden.py" in p for p in paths)


def test_config_golden_dir_detection_works(tmp_path, monkeypatch):
    # the load_config import must resolve (was a dead import before) so a
    # golden_model_dirs entry is honored
    gdir = tmp_path / "gold"
    gdir.mkdir()
    (gdir / "codec_golden.py").write_text("def encode(x): return x\n")
    import orchestrator.architecture.specialists.prd_spec as ps
    monkeypatch.setattr(
        "orchestrator.langgraph.pipeline_helpers.load_config",
        lambda: {"golden_model_dirs": [str(gdir)]},
    )
    avail, paths = ps._golden_available(str(tmp_path), "no .py refs here")
    assert avail is True and any("codec_golden.py" in p for p in paths)
