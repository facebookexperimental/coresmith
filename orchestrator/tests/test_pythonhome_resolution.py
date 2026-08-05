# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for ``orchestrator.harness.env.resolve_pythonhome``.

Regression cover for a daemon-won't-start failure: ``bin/coresmith`` used to
inject ``PYTHONHOME=sysconfig.get_config_var("prefix")`` unconditionally. That
is the prefix the interpreter was *configured* with, which relocated /
framework / vendor-repackaged builds report as a path holding no stdlib. Since
PYTHONHOME overrides stdlib discovery, the spawned daemon died before running
any Python with ``ModuleNotFoundError: No module named 'encodings'``.
"""

from __future__ import annotations

import sys
from pathlib import Path

from orchestrator.harness import env as harness_env


def _make_fake_prefix(root: Path, version: str = "3.12") -> Path:
    """Build a directory that looks like a real posix Python install root."""
    (root / "lib" / f"python{version}" / "encodings").mkdir(parents=True)
    (root / "lib" / f"python{version}" / "encodings" / "__init__.py").touch()
    return root


class TestPrefixHasStdlib:
    def test_real_base_prefix_is_accepted(self):
        assert harness_env._prefix_has_stdlib(sys.base_prefix)

    def test_posix_layout_accepted(self, tmp_path):
        assert harness_env._prefix_has_stdlib(_make_fake_prefix(tmp_path / "p"))

    def test_windows_layout_accepted(self, tmp_path):
        (tmp_path / "Lib" / "encodings").mkdir(parents=True)
        assert harness_env._prefix_has_stdlib(tmp_path)

    def test_directory_without_stdlib_rejected(self, tmp_path):
        (tmp_path / "lib").mkdir()
        assert not harness_env._prefix_has_stdlib(tmp_path)

    def test_nonexistent_path_rejected(self, tmp_path):
        assert not harness_env._prefix_has_stdlib(tmp_path / "nope")

    def test_venv_prefix_rejected_when_it_has_no_stdlib(self, tmp_path):
        # A venv's own prefix carries site-packages but no stdlib -- exactly
        # the shape that must never be handed to PYTHONHOME.
        (tmp_path / "lib" / "python3.12" / "site-packages").mkdir(parents=True)
        assert not harness_env._prefix_has_stdlib(tmp_path)


class TestResolvePythonhome:
    def test_returns_a_bootable_prefix_for_this_interpreter(self):
        resolved = harness_env.resolve_pythonhome()
        assert resolved is not None
        assert harness_env._prefix_has_stdlib(resolved)

    def test_skips_configured_prefix_that_holds_no_stdlib(self, tmp_path, monkeypatch):
        """The exact production bug: configured prefix != install prefix."""
        broken = tmp_path / "Library" / "Frameworks" / "Python.framework"
        broken.mkdir(parents=True)
        import sysconfig
        monkeypatch.setattr(
            sysconfig, "get_config_var",
            lambda name: str(broken) if name == "prefix" else None,
        )
        resolved = harness_env.resolve_pythonhome()
        assert resolved != str(broken)
        assert resolved == sys.base_prefix

    def test_returns_none_when_no_candidate_is_bootable(self, tmp_path, monkeypatch):
        """Better to inject nothing than a value that kills the child."""
        broken = tmp_path / "broken"
        broken.mkdir()
        monkeypatch.setattr(sys, "base_prefix", str(broken))
        import sysconfig
        monkeypatch.setattr(sysconfig, "get_config_var", lambda name: str(broken))
        assert harness_env.resolve_pythonhome() is None

    def test_prefers_the_prefix_of_the_interpreter_being_spawned(
        self, tmp_path, monkeypatch,
    ):
        """``bin/coresmith`` may run under one interpreter and spawn another;
        the value must describe the child, not the launcher."""
        child_prefix = _make_fake_prefix(tmp_path / "child")
        monkeypatch.setattr(
            harness_env, "_interpreter_base_prefix", lambda python: str(child_prefix),
        )
        assert harness_env.resolve_pythonhome("/some/other/python") == str(child_prefix)

    def test_falls_back_when_the_spawned_interpreter_cannot_be_probed(
        self, monkeypatch,
    ):
        monkeypatch.setattr(
            harness_env, "_interpreter_base_prefix", lambda python: None,
        )
        assert harness_env.resolve_pythonhome("/nonexistent/python") == sys.base_prefix

    def test_same_interpreter_is_not_re_probed(self, monkeypatch):
        """No subprocess when the target is the interpreter already running."""
        def _boom(python):  # pragma: no cover -- must not be reached
            raise AssertionError("should not probe the current interpreter")

        monkeypatch.setattr(harness_env, "_interpreter_base_prefix", _boom)
        assert harness_env.resolve_pythonhome(sys.executable) == sys.base_prefix


class TestInterpreterBasePrefix:
    def test_probes_a_real_interpreter(self):
        assert harness_env._interpreter_base_prefix(sys.executable) == sys.base_prefix

    def test_missing_interpreter_returns_none(self):
        assert harness_env._interpreter_base_prefix("/nonexistent/python") is None


class TestHarnessChildEnv:
    def test_injects_only_a_bootable_pythonhome(self, monkeypatch):
        monkeypatch.delenv("PYTHONHOME", raising=False)
        env = harness_env.harness_child_env()
        if "PYTHONHOME" in env:
            assert harness_env._prefix_has_stdlib(env["PYTHONHOME"])

    def test_explicit_pythonhome_is_left_alone(self, monkeypatch):
        monkeypatch.setenv("PYTHONHOME", "/operator/override")
        assert harness_env.harness_child_env()["PYTHONHOME"] == "/operator/override"
