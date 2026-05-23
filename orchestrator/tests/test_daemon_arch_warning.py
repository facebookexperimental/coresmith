# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the architecture-artifacts warning emitted by the daemon's
`/run/start` endpoint when the frontend pipeline is launched without an
upstream architecture phase run."""

from __future__ import annotations

import pytest

from orchestrator.daemon.server import _check_architecture_artifacts


class TestCheckArchitectureArtifacts:
    def test_no_warnings_when_all_present(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CORESMITH_SKIP_ARCH_WARN", raising=False)
        (tmp_path / ".coresmith").mkdir()
        (tmp_path / "arch").mkdir()
        (tmp_path / ".coresmith" / "prd_spec.json").write_text("{}")
        (tmp_path / "arch" / "ers_spec.md").write_text("# ERS")
        (tmp_path / ".coresmith" / "block_diagram.json").write_text("{}")

        assert _check_architecture_artifacts(str(tmp_path)) == []

    def test_warns_when_all_missing(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CORESMITH_SKIP_ARCH_WARN", raising=False)
        warnings = _check_architecture_artifacts(str(tmp_path))

        assert len(warnings) == 1
        w = warnings[0]
        assert "PRD spec" in w
        assert "ERS spec" in w
        assert "block diagram" in w
        assert "coresmith architecture start" in w
        assert "CORESMITH_SKIP_ARCH_WARN" in w

    def test_warns_when_only_ers_missing(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CORESMITH_SKIP_ARCH_WARN", raising=False)
        (tmp_path / ".coresmith").mkdir()
        (tmp_path / ".coresmith" / "prd_spec.json").write_text("{}")
        (tmp_path / ".coresmith" / "block_diagram.json").write_text("{}")

        warnings = _check_architecture_artifacts(str(tmp_path))
        assert len(warnings) == 1
        assert "ERS spec" in warnings[0]
        assert "PRD spec" not in warnings[0]

    @pytest.mark.parametrize("value", ["1", "true", "yes", "on", "TRUE", "Yes"])
    def test_skip_env_var_suppresses_warning(self, tmp_path, monkeypatch, value):
        monkeypatch.setenv("CORESMITH_SKIP_ARCH_WARN", value)
        # Even though all artifacts are missing, we suppress.
        assert _check_architecture_artifacts(str(tmp_path)) == []

    def test_skip_env_var_off_does_not_suppress(self, tmp_path, monkeypatch):
        # Values other than the truthy set should NOT suppress.
        monkeypatch.setenv("CORESMITH_SKIP_ARCH_WARN", "0")
        warnings = _check_architecture_artifacts(str(tmp_path))
        assert len(warnings) == 1

        monkeypatch.setenv("CORESMITH_SKIP_ARCH_WARN", "")
        warnings = _check_architecture_artifacts(str(tmp_path))
        assert len(warnings) == 1
