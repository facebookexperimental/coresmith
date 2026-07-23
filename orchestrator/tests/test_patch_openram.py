# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""scripts/patch_openram.py: idempotent self-heal of a pip OpenRAM wheel
(missing __main__.py + NumPy-2 bbox). Uses a synthetic package dir; no real
OpenRAM needed."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import patch_openram as po  # noqa: E402


_BROKEN_BBOX = (
    "class hierarchy_layout:\n"
    "    def get_bbox(self):\n"
    "        ll = vector(boundary[0][0], boundary[0][1])\n"
    "        ur = vector(boundary[1][0], boundary[1][1])\n"
    "        return ll, ur\n"
)


def _fake_openram(tmp_path: Path, *, with_main: bool, broken_bbox: bool) -> Path:
    pkg = tmp_path / "openram"
    (pkg / "compiler" / "base").mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "sram_compiler.py").write_text("print('compile')\n")
    if with_main:
        (pkg / "__main__.py").write_text("# stock\n")
    hl = pkg / "compiler" / "base" / "hierarchy_layout.py"
    hl.write_text(_BROKEN_BBOX if broken_bbox
                  else po._bbox_sub(_BROKEN_BBOX)[0])
    return pkg


def _point_at(monkeypatch, pkg: Path):
    real = importlib.util.find_spec

    def fake(name, *a, **k):
        if name == "openram":
            class _S:
                origin = str(pkg / "__init__.py")
            return _S()
        if name == "openram.__main__":
            return object() if (pkg / "__main__.py").exists() else None
        return real(name, *a, **k)
    monkeypatch.setattr(po.importlib.util, "find_spec", fake)


def test_applies_both_fixes(tmp_path, monkeypatch):
    pkg = _fake_openram(tmp_path, with_main=False, broken_bbox=True)
    _point_at(monkeypatch, pkg)
    r = po.patch_openram()
    assert r.ok is True
    assert set(r.applied) == {"__main__.py", "bbox-item"}
    assert po._MAIN_MARKER in (pkg / "__main__.py").read_text()
    hl = (pkg / "compiler" / "base" / "hierarchy_layout.py").read_text()
    assert ".item()" in hl
    # backup written
    assert (pkg / "compiler" / "base"
            / "hierarchy_layout.py.coresmith-bak").exists()


def test_patches_regardless_of_indentation(tmp_path, monkeypatch):
    # regression: the real openram file indents the bbox lines 12 spaces, not
    # 8. An exact-string matcher missed it; the regex matcher must not.
    pkg = _fake_openram(tmp_path, with_main=False, broken_bbox=True)
    hl = pkg / "compiler" / "base" / "hierarchy_layout.py"
    hl.write_text(
        "class hierarchy_layout:\n"
        "    def get_bbox(self):\n"
        "            ll = vector(boundary[0][0], boundary[0][1])\n"
        "            ur = vector(boundary[1][0], boundary[1][1])\n"
        "            return ll, ur\n")
    _point_at(monkeypatch, pkg)
    r = po.patch_openram()
    assert "bbox-item" in r.applied
    txt = hl.read_text()
    assert "vector(boundary[0][0].item(), boundary[0][1].item())" in txt
    assert "vector(boundary[1][0].item(), boundary[1][1].item())" in txt
    # idempotent: a second pass makes no further change
    r2 = po.patch_openram()
    assert "bbox-item" in r2.already


def test_idempotent_second_run_is_noop(tmp_path, monkeypatch):
    pkg = _fake_openram(tmp_path, with_main=False, broken_bbox=True)
    _point_at(monkeypatch, pkg)
    po.patch_openram()
    r2 = po.patch_openram()
    assert r2.ok is True
    assert r2.applied == []
    assert set(r2.already) == {"__main__.py", "bbox-item"}


def test_stock_working_wheel_is_left_alone(tmp_path, monkeypatch):
    # a hypothetical future wheel that already ships __main__ + fixed bbox
    pkg = _fake_openram(tmp_path, with_main=True, broken_bbox=False)
    _point_at(monkeypatch, pkg)
    r = po.patch_openram()
    assert r.ok is True
    assert r.applied == []
    assert "__main__.py" in r.already and "bbox-item" in r.already


def test_check_only_never_writes(tmp_path, monkeypatch):
    pkg = _fake_openram(tmp_path, with_main=False, broken_bbox=True)
    _point_at(monkeypatch, pkg)
    r = po.patch_openram(check_only=True)
    assert r.applied == []
    assert not (pkg / "__main__.py").exists()
    assert any("missing" in e for e in r.errors)


def test_missing_openram_reports_error(monkeypatch):
    monkeypatch.setattr(po, "_openram_pkg_dir", lambda: None)
    r = po.patch_openram()
    assert r.ok is False and any("not importable" in e for e in r.errors)


def test_unknown_bbox_pattern_not_guessed(tmp_path, monkeypatch):
    pkg = _fake_openram(tmp_path, with_main=False, broken_bbox=True)
    hl = pkg / "compiler" / "base" / "hierarchy_layout.py"
    hl.write_text("class x:\n    pass\n")  # neither broken nor fixed pattern
    _point_at(monkeypatch, pkg)
    r = po.patch_openram()
    assert any("bbox lines not found" in e for e in r.errors)


def test_ensure_openram_patched_reaches_patcher(tmp_path, monkeypatch):
    """Regression: ensure_openram_patched used sys.path but sys was not a
    module-level import in openram_gen -> NameError swallowed as False. A
    fake scripts/patch_openram on the resolved path must be reached and its
    .ok returned (proves the sys.path line executes)."""
    import orchestrator.langgraph.openram_gen as og
    fake_scripts = tmp_path / "scripts"
    fake_scripts.mkdir()
    (fake_scripts / "patch_openram.py").write_text(
        "class _R:\n    ok = True\n"
        "def patch_openram(*a, **k):\n    return _R()\n")
    monkeypatch.setattr(og, "__file__",
                        str(tmp_path / "orchestrator" / "langgraph" / "x.py"))
    import sys as _sys
    monkeypatch.setitem(_sys.modules, "patch_openram", None)
    _sys.modules.pop("patch_openram", None)
    assert og.ensure_openram_patched() is True
