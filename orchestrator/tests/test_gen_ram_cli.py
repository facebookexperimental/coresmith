# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Regression tests for ``bin/gen_ram``.

The CLI shipped with a call that could never succeed:
``ensure_macro(ports, width, depth)`` against a
``ensure_macro(words, data_bits, *, ...)`` signature -- three positionals into
a two-positional function, in the wrong order, with a string where an int
belongs. It raised TypeError on every invocation and nothing caught it, because
nothing imported the module under test. These tests exist so that cannot recur.
"""
from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GEN_RAM = REPO_ROOT / "bin" / "gen_ram"


def _load():
    spec = importlib.util.spec_from_loader(
        "gen_ram_cli",
        importlib.machinery.SourceFileLoader("gen_ram_cli", str(GEN_RAM)),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gen_ram():
    return _load()


class TestCallSignature:
    def test_module_imports(self, gen_ram):
        assert hasattr(gen_ram, "stage_generate")

    def test_stage_generate_calls_ensure_macro_correctly(self, gen_ram, monkeypatch):
        """(words, data_bits) positionally -- depth first, then width."""
        seen = {}

        def fake_ensure_macro(words, data_bits, **kw):
            seen["words"] = words
            seen["data_bits"] = data_bits
            seen["kw"] = kw
            return type("M", (), {"name": "fake_macro", "lib": "/tmp/fake.lib"})()

        import orchestrator.langgraph.openram_gen as og
        monkeypatch.setattr(og, "ensure_macro", fake_ensure_macro)

        ok, detail, info = gen_ram.stage_generate(
            width=16, depth=128, ports="1rw1r", out=Path("/tmp"), timeout=60
        )
        assert ok, detail
        # depth -> words, width -> data_bits. Getting these backwards silently
        # generates a transposed macro, which is worse than a crash.
        assert seen["words"] == 128, "depth must map to words"
        assert seen["data_bits"] == 16, "width must map to data_bits"
        assert info["name"] == "fake_macro"

    def test_signature_is_actually_two_positionals(self):
        """Pin the contract this CLI depends on."""
        from orchestrator.langgraph.openram_gen import ensure_macro

        params = list(inspect.signature(ensure_macro).parameters.values())
        positional = [
            p for p in params
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        assert [p.name for p in positional] == ["words", "data_bits"]

    def test_bad_ports_rejected_not_crashed(self, gen_ram):
        ok, detail, _ = gen_ram.stage_generate(
            width=16, depth=128, ports="7rw", out=Path("/tmp"), timeout=60
        )
        assert not ok and "unsupported" in detail.lower()

    def test_ensure_macro_exception_is_reported_not_raised(self, gen_ram, monkeypatch):
        import orchestrator.langgraph.openram_gen as og

        def boom(*a, **k):
            raise TypeError("simulated bad call")

        monkeypatch.setattr(og, "ensure_macro", boom)
        ok, detail, _ = gen_ram.stage_generate(
            width=16, depth=128, ports="1rw1r", out=Path("/tmp"), timeout=60
        )
        assert not ok and "TypeError" in detail


class TestHonestVerdicts:
    def test_parse_drc_returns_none_when_no_verdict(self, gen_ram):
        """None is NOT zero -- an unparsed DRC log must never read as clean."""
        assert gen_ram.parse_drc("nothing useful here") is None
        assert gen_ram.parse_drc("MAGIC_DRC_ERRORS = 0") == 0
        assert gen_ram.parse_drc("Total DRC errors found: 7") == 7

    def test_parse_lvs_returns_none_when_no_verdict(self, gen_ram):
        assert gen_ram.parse_lvs("no verdict") is None
        assert gen_ram.parse_lvs("Circuits match uniquely") is True
        assert gen_ram.parse_lvs("Circuits do not match") is False

    def test_liberty_missing_file_fails(self, gen_ram):
        ok, detail, cyc = gen_ram.check_liberty(Path("/nonexistent/x.lib"))
        assert not ok and cyc is None


class TestOpenramAvailability:
    def test_package_exported_openram_home_is_not_trusted(self, monkeypatch, tmp_path):
        """Importing openram exports OPENRAM_HOME=<pkg>/compiler, which has no
        sram_compiler.py. Treating that as a source checkout made the probe
        return a false NEGATIVE and left generation unreachable."""
        import orchestrator.langgraph.openram_gen as og

        monkeypatch.setenv("OPENRAM_HOME", str(tmp_path))  # no sram_compiler.py
        monkeypatch.setattr(og, "_OPENRAM_RUNNABLE", False)
        # Must fall through to the real check rather than short-circuiting True.
        assert og.openram_available() is False

    def test_real_checkout_openram_home_is_honored(self, monkeypatch, tmp_path):
        import orchestrator.langgraph.openram_gen as og

        (tmp_path / "sram_compiler.py").write_text("# real checkout\n")
        monkeypatch.setenv("OPENRAM_HOME", str(tmp_path))
        assert og.openram_available() is True
