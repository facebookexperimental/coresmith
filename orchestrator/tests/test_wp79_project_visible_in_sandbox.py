# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""WP-79: the adapter sandbox gives the adapter private /tmp scratch without
hiding the project it must read. A project living under /tmp used to lose its
candidate manifest, sources and inputs behind the tmpfs."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from orchestrator.harness import task_adapter as ta


def _idx(argv: list[str], *pair: str) -> int:
    for i in range(len(argv) - len(pair) + 1):
        if argv[i:i + len(pair)] == list(pair):
            return i
    raise AssertionError(f"{pair} not in argv")


def test_project_is_rebound_after_the_tmpfs(tmp_path, monkeypatch):
    monkeypatch.delenv("CORESMITH_ADAPTER_SANDBOX", raising=False)
    monkeypatch.setenv("CORESMITH_BWRAP", "/fake/bwrap")
    root = tmp_path / "project"
    work = root / ".coresmith" / "acceptance" / "sha" / "attempt-000001" / "work"
    work.mkdir(parents=True)
    argv = ta._sandbox_argv(["/bin/true"], work, project_root=str(root))
    assert argv[0] == "/fake/bwrap"
    # order matters: a later mount wins, so the project must come after the
    # tmpfs and the writable work directory after the project.
    assert _idx(argv, "--tmpfs", "/tmp") < _idx(argv, "--ro-bind", str(root))
    assert _idx(argv, "--ro-bind", str(root)) < _idx(argv, "--bind", str(work))
    assert "--unshare-net" in argv and "--die-with-parent" in argv


def test_no_project_root_keeps_the_old_argv(tmp_path, monkeypatch):
    monkeypatch.delenv("CORESMITH_ADAPTER_SANDBOX", raising=False)
    monkeypatch.setenv("CORESMITH_BWRAP", "/fake/bwrap")
    work = tmp_path / "work"
    work.mkdir()
    argv = ta._sandbox_argv(["/bin/true"], work)
    assert "--ro-bind" in argv and _idx(argv, "--bind", str(work))
    # only the root filesystem (plus the engine's own checkout when it lives
    # under /tmp, which the sandbox tmpfs would otherwise mask)
    engine = str(Path(ta.__file__).resolve().parents[2])
    binds = [argv[i + 1] for i, a in enumerate(argv) if a == "--ro-bind"]
    assert [b for b in binds if b != engine] == ["/"]


def test_a_project_under_tmp_can_still_be_read(tmp_path, monkeypatch):
    """The real boundary: run bubblewrap and read a project file through it."""
    if not shutil.which("bwrap"):
        pytest.skip("bubblewrap not installed")
    root = Path("/tmp") / f"coresmith-wp79-{tmp_path.name}"
    shutil.rmtree(root, ignore_errors=True)
    (root / "inputs").mkdir(parents=True)
    (root / "inputs" / "marker.txt").write_text("visible")
    work = root / "work"
    work.mkdir()
    try:
        monkeypatch.delenv("CORESMITH_ADAPTER_SANDBOX", raising=False)
        monkeypatch.delenv("CORESMITH_BWRAP", raising=False)
        argv = ta._sandbox_argv(
            ["/bin/sh", "-c", f"cat {root}/inputs/marker.txt"], work, project_root=str(root))
        done = subprocess.run(argv, capture_output=True, text=True, timeout=60)
        if done.returncode != 0 and "namespace" in (done.stderr or ""):
            pytest.skip("user namespaces unavailable here")
        assert done.stdout.strip() == "visible", done.stderr[-300:]
    finally:
        shutil.rmtree(root, ignore_errors=True)
