# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Backend preflight must not report a silent false green without nix.

The OpenROAD/Magic/netgen checks only assert the configured path EXISTS. By
default those paths are ``scripts/{openroad,magic,netgen}-nix.sh``, committed
to this repo, so they exist on every checkout -- backend preflight returned
``ok: true`` on a box with no EDA toolchain at all. Each wrapper is a one-liner
that ``exec nix shell "nixpkgs#<tool>"``, so without nix every backend node
fails at its first invocation.
"""

from __future__ import annotations

import shutil

import pytest

import orchestrator.langgraph.pipeline_helpers as ph

_NIX_HINT = "nix wrappers"


def _warn_blob(result: dict) -> str:
    return "\n".join(result.get("warnings", []))


@pytest.fixture
def _no_pdk_noise(monkeypatch):
    """Backend preflight also checks PDK files; those errors are not the point."""
    monkeypatch.setenv("CORESMITH_SKIP_SYNTH", "1")
    monkeypatch.setenv("CORESMITH_ALLOW_NO_OPENRAM", "1")


def _fake_which(present: set[str]):
    def _which(name, *a, **kw):
        return f"/usr/bin/{name}" if name in present else None
    return _which


class TestNixWrapperWarning:
    def test_warns_when_wrappers_configured_but_nix_absent(
        self, monkeypatch, _no_pdk_noise,
    ):
        monkeypatch.setattr(ph.shutil, "which", _fake_which({"verilator", "yosys"}))
        result = ph.preflight_check(["backend"])
        blob = _warn_blob(result)
        assert _NIX_HINT in blob
        # names the wrappers so the operator knows which paths are affected
        assert "openroad-nix.sh" in blob

    def test_no_warning_when_nix_is_on_path(self, monkeypatch, _no_pdk_noise):
        monkeypatch.setattr(
            ph.shutil, "which", _fake_which({"verilator", "yosys", "nix"}),
        )
        assert _NIX_HINT not in _warn_blob(ph.preflight_check(["backend"]))

    def test_no_warning_when_backend_phase_not_requested(
        self, monkeypatch, _no_pdk_noise,
    ):
        monkeypatch.setattr(ph.shutil, "which", _fake_which({"verilator", "yosys"}))
        assert _NIX_HINT not in _warn_blob(ph.preflight_check(["pipeline"]))

    def test_no_warning_when_binaries_are_not_nix_wrappers(
        self, monkeypatch, _no_pdk_noise,
    ):
        """A site pointing config.yaml at real binaries must stay quiet."""
        import orchestrator.langgraph.backend_helpers as bh

        for attr in ("OPENROAD_BIN", "MAGIC_BIN", "NETGEN_BIN"):
            monkeypatch.setattr(bh, attr, f"/usr/bin/{attr.split('_')[0].lower()}")
        monkeypatch.setattr(ph.shutil, "which", _fake_which({"verilator", "yosys"}))
        assert _NIX_HINT not in _warn_blob(ph.preflight_check(["backend"]))

    def test_warning_does_not_flip_ok(self, monkeypatch, _no_pdk_noise):
        """It is a warning, not an error -- `ok` is decided by errors only."""
        monkeypatch.setattr(ph.shutil, "which", _fake_which({"verilator", "yosys"}))
        result = ph.preflight_check(["backend"])
        assert result["ok"] == (len(result["errors"]) == 0)


class TestWrappersReallyNeedNix:
    """Guard the premise: if the wrappers stop shelling to nix, drop the warning."""

    def test_shipped_wrappers_exec_nix(self):
        from pathlib import Path

        repo_root = Path(ph.__file__).resolve().parents[2]
        for name in ("openroad-nix.sh", "magic-nix.sh", "netgen-nix.sh"):
            script = repo_root / "scripts" / name
            if not script.exists():
                continue
            assert "nix shell" in script.read_text(), f"{name} no longer uses nix"

    def test_shutil_is_the_module_under_patch(self):
        assert ph.shutil is shutil
