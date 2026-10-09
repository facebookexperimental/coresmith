# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""WP-78: an adapter may name its interpreter by bare name and have it resolved
on PATH, so a task adapter need not hardcode one host's absolute python path."""
from __future__ import annotations

import os
import shutil
import stat
import sys
from pathlib import Path

from orchestrator.harness import task_adapter as ta
from orchestrator.state_store import trust
from orchestrator.tests.candidate_fixtures import adopt


def _project(tmp_path: Path, monkeypatch, header: str) -> tuple[Path, Path]:
    monkeypatch.setenv("CORESMITH_TRUST_DIR", str(tmp_path / "owner-trust"))
    monkeypatch.setenv("CORESMITH_ADAPTER_SANDBOX", "none")
    monkeypatch.delenv("CORESMITH_TASK_ADAPTER_PYTHON", raising=False)
    root = tmp_path / "project"
    (root / "inputs").mkdir(parents=True)
    top = root / "top.v"
    top.write_text("module chip_top(); endmodule")
    (root / "inputs/task.yaml").write_text("top: chip_top\n")
    (root / "inputs/task_adapter.py").write_text(
        f"{header}\nCASES = ['only']\nTOP = 'chip_top'\nLABEL = 'x'\n\n"
        "def grade(candidate, workdir):\n"
        "    return {'cases': {'only': {'ok': True}}}\n")
    adopt(root, top)
    assert trust.write_oracle_manifest(root) is not None
    return root, top


def test_bare_interpreter_name_is_resolved_on_path(tmp_path, monkeypatch):
    root, top = _project(tmp_path, monkeypatch, "# coresmith-python: python3")
    result = ta.run_task_adapter(str(root), str(top))
    assert result["kind"] != "adapter_defect", result
    assert result["passed"] is True, result


def test_an_unresolvable_name_still_fails_closed(tmp_path, monkeypatch):
    root, top = _project(tmp_path, monkeypatch,
                         "# coresmith-python: definitely-not-a-real-python")
    result = ta.run_task_adapter(str(root), str(top))
    assert result["passed"] is False
    assert result["kind"] == "adapter_defect"
    assert "definitely-not-a-real-python" in result["reason"]


def test_an_absolute_path_is_used_verbatim(tmp_path, monkeypatch):
    root, top = _project(tmp_path, monkeypatch, f"# coresmith-python: {sys.executable}")
    seen: dict = {}
    real = ta.run_process

    def spy(cmd, **kw):
        seen.setdefault("argv0", cmd[0])
        return real(cmd, **kw)

    monkeypatch.setattr(ta, "run_process", spy)
    ta.run_task_adapter(str(root), str(top))
    assert seen["argv0"] == sys.executable


def test_a_relative_path_is_not_treated_as_a_bare_name(tmp_path, monkeypatch):
    fake = tmp_path / "bin" / "python3"
    fake.parent.mkdir(parents=True)
    fake.write_text("#!/bin/sh\nexit 0\n")
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", str(fake.parent) + os.pathsep + os.environ["PATH"])
    assert shutil.which("python3")
    root, top = _project(tmp_path, monkeypatch, "# coresmith-python: ./bin/python3")
    result = ta.run_task_adapter(str(root), str(top))
    assert result["kind"] == "adapter_defect"
