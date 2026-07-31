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
import json
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


class TestUnmeasuredStagesAreNotFailures:
    """A stage the tool never measured must not be rendered as a result.

    The macro DRC stage used to emit ``{"ok": false, "violations": -1}`` for a
    DRC that produced no report at all (see the ``exp-raster-macro-20260727``
    w32_d64 / w9_d512 reports, whose named precheck_magic_drc.rpt was never
    written): every consumer read that as "this macro HAS DRC violations". The
    emitted report must instead carry the engine's three-state vocabulary --
    ``status: not_run`` + a ``reason``, ``ok: null``, ``violations: null``.
    """

    def _emit_report(self, gen_ram, monkeypatch, tmp_path):
        lib = tmp_path / "fake.lib"
        lib.write_text("library (fake) { }\n")
        monkeypatch.setattr(gen_ram, "stage_patch",
                            lambda check_only: (True, "patched"))
        monkeypatch.setattr(
            gen_ram, "stage_generate",
            lambda width, depth, ports, out, timeout: (
                True, "generated", {"name": "fake_macro", "lib": str(lib)}))
        monkeypatch.setattr(gen_ram, "check_liberty",
                            lambda p: (True, "minimum_period 2.0000 ns", 2.0))

        rc = gen_ram.main(["--width", "16", "--depth", "128",
                           "--out", str(tmp_path)])
        report = json.loads((tmp_path / "gen_ram_report.json").read_text())
        return rc, report

    def test_drc_stage_is_not_run_with_a_reason(self, gen_ram, monkeypatch,
                                               tmp_path):
        rc, report = self._emit_report(gen_ram, monkeypatch, tmp_path)
        assert rc == 0 and report["verdict"] == "PASS"

        drc = report["stages"]["drc"]
        assert drc["status"] == "not_run"
        # NOT False -- False claims a measured violation count.
        assert drc["ok"] is None
        # NOT -1 and NOT 0 -- there is no count.
        assert drc["violations"] is None
        assert drc["measured"] is False
        assert "check_lvs=False" in drc["reason"]
        assert "-1" not in json.dumps(drc)

    def test_lvs_stage_uses_the_same_vocabulary(self, gen_ram, monkeypatch,
                                               tmp_path):
        _, report = self._emit_report(gen_ram, monkeypatch, tmp_path)
        lvs = report["stages"]["lvs"]
        assert lvs["status"] == "not_run" and lvs["ok"] is None

    def test_unmeasured_stages_are_named_next_to_the_verdict(
            self, gen_ram, monkeypatch, tmp_path):
        """`verdict: PASS` must not be readable as "DRC/LVS clean"."""
        _, report = self._emit_report(gen_ram, monkeypatch, tmp_path)
        assert report["unmeasured"] == ["drc", "lvs"]

    def test_measured_stages_keep_their_boolean_verdicts(self, gen_ram,
                                                        monkeypatch, tmp_path):
        """Only the unmeasured case changed: a real failure still fails."""
        monkeypatch.setattr(gen_ram, "stage_patch",
                            lambda check_only: (False, "patcher raised: boom"))
        rc = gen_ram.main(["--width", "16", "--depth", "128",
                           "--out", str(tmp_path)])
        report = json.loads((tmp_path / "gen_ram_report.json").read_text())
        assert rc == 1
        assert report["verdict"] == "FAIL"
        assert report["stages"]["patch"]["ok"] is False
        assert report["unmeasured"] == []


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
