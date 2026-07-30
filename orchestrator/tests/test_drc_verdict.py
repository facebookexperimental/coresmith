# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""A DRC that produced NO REPORT is not a DRC failure.

Two real macros in ``exp-raster-macro-20260727`` (``w32_d64``, ``w9_d512``)
carry ``stages.drc = {"ok": false, "violations": -1}`` naming a
``precheck_magic_drc.rpt`` that was never written -- Magic produced nothing.
``-1`` meant "could not measure", but every consumer read the negative count and
``ok: false`` as "this macro HAS DRC violations", so the macros were FAILed for
a measurement that never happened (and the KLayout sibling did the mirror-image
thing: a missing XML report counted as ``violation_count = 0`` = clean).

These tests pin the four states apart -- report MISSING, report EMPTY, report
with ZERO violations, report with N violations -- at both the classifier and the
two producers, plus the precheck consumer that turns them into a signoff verdict.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from orchestrator.langgraph import tapeout_helpers as th
from orchestrator.langgraph.drc_verdict import (
    STATUS_FAIL,
    STATUS_NOT_RUN,
    STATUS_PASS,
    classify_drc,
    drc_stage,
    drc_summary,
    unmeasured_drc,
)

# Real ``drc listall why`` shapes (mirror of test_backend_helpers fixtures).
REPORT_CLEAN = "Design: sram_flat\nDRC count: 0\n\n"
REPORT_3 = (
    "Design: sram_flat\n"
    "DRC count: \n"
    "{Local interconnect spacing < 0.17um (li.3)} "
    "{{10961 96611 10977 96627} {10961 96611 10981 96627} "
    "{28515 242879 28529 242899}}\n"
)
STDOUT_CLEAN = "DRC violations: 0\n"
STDOUT_BLANK = "Total DRC errors found: 0\nDRC violations: \n"


# ---------------------------------------------------------------------------
# The classifier: the four states, kept distinct
# ---------------------------------------------------------------------------

class TestClassifyDrcFourStates:
    def test_report_missing_is_not_run_not_failure(self, tmp_path):
        """The raster bug, reproduced: no report on disk, -1 from the parser."""
        rpt = tmp_path / "precheck_magic_drc.rpt"
        v = classify_drc(violation_count=-1, report_path=str(rpt))

        assert v["status"] == STATUS_NOT_RUN
        assert v["measured"] is False
        assert v["pass"] is False
        # The lie was "-1 violations". There is NO count.
        assert v["violation_count"] is None
        assert "NO DRC report" in v["reason"]
        assert str(rpt) in v["reason"]
        assert "NOT a violation count" in v["reason"]

    def test_report_present_but_empty_is_not_run(self, tmp_path):
        rpt = tmp_path / "precheck_magic_drc.rpt"
        rpt.write_text("")
        v = classify_drc(violation_count=0, report_path=str(rpt))

        assert v["status"] == STATUS_NOT_RUN
        assert v["violation_count"] is None
        assert "EMPTY" in v["reason"]

    def test_report_present_but_whitespace_only_is_not_run(self, tmp_path):
        rpt = tmp_path / "precheck_magic_drc.rpt"
        rpt.write_text("\n \n")
        assert classify_drc(violation_count=0,
                            report_path=str(rpt))["status"] == STATUS_NOT_RUN

    def test_zero_violations_with_a_real_report_is_a_pass(self, tmp_path):
        rpt = tmp_path / "precheck_magic_drc.rpt"
        rpt.write_text(REPORT_CLEAN)
        v = classify_drc(violation_count=0, report_path=str(rpt))

        assert v["status"] == STATUS_PASS
        assert v["pass"] is True
        assert v["measured"] is True
        assert v["violation_count"] == 0

    def test_n_violations_is_a_failure(self, tmp_path):
        rpt = tmp_path / "precheck_magic_drc.rpt"
        rpt.write_text(REPORT_3)
        v = classify_drc(violation_count=3, report_path=str(rpt))

        assert v["status"] == STATUS_FAIL
        assert v["pass"] is False
        assert v["measured"] is True
        assert v["violation_count"] == 3


class TestClassifyDrcEdges:
    def test_tool_that_never_ran_is_not_run_with_the_tool_error(self, tmp_path):
        rpt = tmp_path / "r.rpt"
        rpt.write_text(REPORT_CLEAN)
        v = classify_drc(violation_count=0, report_path=str(rpt),
                         tool_ran=False, tool_error="magic: cannot load GDS")

        assert v["status"] == STATUS_NOT_RUN
        assert v["violation_count"] is None
        assert "cannot load GDS" in v["reason"]

    def test_measured_violations_survive_a_crashed_tool(self, tmp_path):
        """A report full of rects is evidence even if the tool then died."""
        rpt = tmp_path / "r.rpt"
        rpt.write_text(REPORT_3)
        v = classify_drc(violation_count=3, report_path=str(rpt),
                         tool_ran=False, tool_error="segfault")

        assert v["status"] == STATUS_FAIL
        assert v["violation_count"] == 3

    def test_report_without_a_parseable_count_is_not_run(self, tmp_path):
        rpt = tmp_path / "r.rpt"
        rpt.write_text("Design: sram_flat\n(magic said nothing else)\n")
        v = classify_drc(violation_count=None, report_path=str(rpt))

        assert v["status"] == STATUS_NOT_RUN
        assert "no parseable violation count" in v["reason"]

    def test_no_report_path_at_all_is_not_run(self):
        v = classify_drc(violation_count=0, report_path="")
        assert v["status"] == STATUS_NOT_RUN
        assert "no DRC report path" in v["reason"]

    @pytest.mark.parametrize("bad", [-1, -999, None, "", "abc", True, False])
    def test_no_unmeasured_input_can_produce_a_pass_or_a_count(self, bad, tmp_path):
        """Every "could not measure" input shape lands in not_run with no count."""
        rpt = tmp_path / "missing.rpt"  # deliberately not created
        v = classify_drc(violation_count=bad, report_path=str(rpt))
        assert v["status"] == STATUS_NOT_RUN
        assert v["pass"] is False
        assert v["violation_count"] is None

    def test_unmeasured_drc_always_carries_a_reason(self):
        v = unmeasured_drc("openram never ran DRC", tool="OpenRAM")
        assert v["status"] == STATUS_NOT_RUN
        assert v["reason"] == "openram never ran DRC"
        assert v["pass"] is False and v["violation_count"] is None
        # `error` so readers that only print that field still say why.
        assert v["error"] == v["reason"]


class TestVerdictRendering:
    def test_summary_words_are_distinct_per_state(self, tmp_path):
        rpt = tmp_path / "r.rpt"
        rpt.write_text(REPORT_CLEAN)
        clean = drc_summary(classify_drc(violation_count=0, report_path=str(rpt)))
        dirty = drc_summary(classify_drc(violation_count=7, report_path=str(rpt)))
        unmeasured = drc_summary(unmeasured_drc("magic wrote nothing"))

        assert "clean" in clean and "COULD NOT" not in clean
        assert "7 violation(s)" in dirty
        assert "COULD NOT BE MEASURED" in unmeasured
        assert "magic wrote nothing" in unmeasured
        # The unmeasured line must never read as a count or as clean.
        assert "-1" not in unmeasured and "clean" not in unmeasured

    def test_stage_ok_is_tri_state(self, tmp_path):
        rpt = tmp_path / "r.rpt"
        rpt.write_text(REPORT_CLEAN)

        assert drc_stage(classify_drc(violation_count=0,
                                      report_path=str(rpt)))["ok"] is True
        assert drc_stage(classify_drc(violation_count=2,
                                      report_path=str(rpt)))["ok"] is False
        unmeasured = drc_stage(unmeasured_drc("no report"))
        # NOT False (that claims violations were found) and NOT True.
        assert unmeasured["ok"] is None
        assert unmeasured["status"] == STATUS_NOT_RUN
        assert unmeasured["violations"] is None
        assert not unmeasured["ok"], "must never read as success"


# ---------------------------------------------------------------------------
# Producer: _run_magic_drc_on_gds (the gen_ram / precheck DRC stage)
# ---------------------------------------------------------------------------

def _fake_run_magic(monkeypatch, *, success=True, drc_count=0, stdout="",
                    stderr="", writes: str | None = None, name="rpt"):
    """Stand in for backend_helpers.run_magic; optionally write the report."""
    import orchestrator.langgraph.backend_helpers as bh

    def fake(tcl_script, block_name, step="drc", attempt=1, timeout=600):
        if writes is not None:
            (Path(tcl_script).parent / "precheck_magic_drc.rpt").write_text(writes)
        return {
            "success": success,
            "drc_count": drc_count,
            "stdout": stdout,
            "stderr": stderr,
            "log_path": "/tmp/fake.log",
        }

    monkeypatch.setattr(bh, "run_magic", fake)


class TestRunMagicDrcOnGds:
    @pytest.fixture
    def gds(self, tmp_path):
        p = tmp_path / "sram.gds"
        p.write_bytes(b"\x00\x06\x00\x02\x00\x07")
        return str(p)

    def test_no_report_is_not_run_not_a_minus_one_failure(self, monkeypatch,
                                                          tmp_path, gds):
        """Exactly the raster case: magic ran, wrote no report, count -1."""
        _fake_run_magic(monkeypatch, success=False, drc_count=-1,
                        stderr="magic: no such layout")
        v = th._run_magic_drc_on_gds(gds, str(tmp_path))

        assert v["status"] == STATUS_NOT_RUN
        assert v["violation_count"] is None
        assert v["pass"] is False
        assert v["reason"]
        assert "precheck_magic_drc.rpt" in v["report_path"]

    def test_tool_ok_but_report_absent_is_still_not_run(self, monkeypatch,
                                                       tmp_path, gds):
        _fake_run_magic(monkeypatch, success=True, drc_count=-1)
        v = th._run_magic_drc_on_gds(gds, str(tmp_path))
        assert v["status"] == STATUS_NOT_RUN
        assert "NO DRC report" in v["reason"]

    def test_empty_report_is_not_run(self, monkeypatch, tmp_path, gds):
        _fake_run_magic(monkeypatch, success=True, drc_count=0, writes="")
        v = th._run_magic_drc_on_gds(gds, str(tmp_path))
        assert v["status"] == STATUS_NOT_RUN
        assert "EMPTY" in v["reason"]
        assert v["violation_count"] is None

    def test_clean_report_still_passes(self, monkeypatch, tmp_path, gds):
        _fake_run_magic(monkeypatch, success=True, drc_count=0,
                        stdout=STDOUT_CLEAN, writes=REPORT_CLEAN)
        v = th._run_magic_drc_on_gds(gds, str(tmp_path))
        assert v["status"] == STATUS_PASS
        assert v["pass"] is True
        assert v["violation_count"] == 0

    def test_violations_still_fail(self, monkeypatch, tmp_path, gds):
        _fake_run_magic(monkeypatch, success=True, drc_count=12,
                        stdout="DRC violations: 12\n", writes=REPORT_3)
        v = th._run_magic_drc_on_gds(gds, str(tmp_path))
        assert v["status"] == STATUS_FAIL
        assert v["violation_count"] == 12

    def test_blank_stdout_count_is_recounted_from_this_flows_report(
            self, monkeypatch, tmp_path, gds):
        """run_magic's recount only knows magic_drc.rpt, so the PRECHECK report
        was never consulted -- a blank stdout count read as a clean 0 over a
        report holding real rects."""
        monkeypatch.delenv("CORESMITH_DRC_REPORT_FALLBACK", raising=False)
        _fake_run_magic(monkeypatch, success=True, drc_count=0,
                        stdout=STDOUT_BLANK, writes=REPORT_3)
        v = th._run_magic_drc_on_gds(gds, str(tmp_path))
        assert v["status"] == STATUS_FAIL
        assert v["violation_count"] == 3


# ---------------------------------------------------------------------------
# Producer: _run_klayout_drc (advisory sibling, mirror-image bug)
# ---------------------------------------------------------------------------

class TestRunKlayoutDrc:
    @pytest.fixture
    def gds(self, tmp_path):
        p = tmp_path / "wrapper.gds"
        p.write_bytes(b"\x00\x06\x00\x02\x00\x07")
        return str(p)

    def _fake_subprocess(self, monkeypatch, *, returncode=0,
                         writes: str | None = None, out_dir: Path | None = None):
        import subprocess as sp

        class R:
            def __init__(self):
                self.returncode = returncode
                self.stdout = ""
                self.stderr = ""

        def fake_run(cmd, **kw):
            if writes is not None and out_dir is not None:
                (out_dir / "klayout_drc.xml").write_text(writes)
            return R()

        monkeypatch.setattr(sp, "run", fake_run)
        monkeypatch.setattr(th, "_write_step_log",
                            lambda *a, **k: "/tmp/klayout.log")

    def test_missing_report_no_longer_reads_as_clean(self, monkeypatch,
                                                    tmp_path, gds):
        """It returned violation_count=0 -> pass=True with NO report at all."""
        self._fake_subprocess(monkeypatch)
        v = th._run_klayout_drc(gds, str(tmp_path))
        assert v["status"] == STATUS_NOT_RUN
        assert v["pass"] is False
        assert v["violation_count"] is None

    def test_clean_xml_report_passes(self, monkeypatch, tmp_path, gds):
        self._fake_subprocess(monkeypatch, writes="<report></report>\n",
                              out_dir=tmp_path)
        v = th._run_klayout_drc(gds, str(tmp_path))
        assert v["status"] == STATUS_PASS
        assert v["violation_count"] == 0

    def test_items_in_xml_report_fail(self, monkeypatch, tmp_path, gds):
        self._fake_subprocess(monkeypatch,
                              writes="<report><item></item><item></item></report>\n",
                              out_dir=tmp_path)
        v = th._run_klayout_drc(gds, str(tmp_path))
        assert v["status"] == STATUS_FAIL
        assert v["violation_count"] == 2

    def test_missing_binary_stays_skipped_and_unmeasured(self, monkeypatch,
                                                        tmp_path, gds):
        import subprocess as sp

        def boom(cmd, **kw):
            raise FileNotFoundError("klayout")

        monkeypatch.setattr(sp, "run", boom)
        v = th._run_klayout_drc(gds, str(tmp_path))
        assert v["status"] == STATUS_NOT_RUN
        assert v["skipped"] is True
        # NOT the old fabricated 0 (which read as "clean").
        assert v["violation_count"] is None


# ---------------------------------------------------------------------------
# Consumer: run_mpw_precheck_native
# ---------------------------------------------------------------------------

class TestPrecheckConsumer:
    @pytest.fixture
    def sub(self, monkeypatch, tmp_path):
        """A submission dir whose non-DRC checks all pass."""
        ok = {"pass": True, "errors": [], "warnings": []}
        for fn in ("_check_submission_structure", "_check_and_generate_user_defines",
                   "_check_wrapper_port_names"):
            monkeypatch.setattr(th, fn, lambda *a, **k: dict(ok))
        monkeypatch.setattr(th, "_check_gds_file", lambda *a, **k: dict(ok))
        (tmp_path / "gds").mkdir()
        (tmp_path / "gds" / "top.gds").write_bytes(b"\x00")
        return tmp_path

    def _run(self, monkeypatch, sub, *, magic, klayout=None):
        if klayout is None:  # default: a clean, MEASURED advisory KLayout run
            (sub / "k.xml").write_text("<report></report>")
            klayout = classify_drc(violation_count=0,
                                   report_path=str(sub / "k.xml"),
                                   tool="KLayout")
        monkeypatch.setattr(th, "_run_klayout_drc", lambda *a, **k: dict(klayout))
        monkeypatch.setattr(th, "_run_magic_drc_on_gds", lambda *a, **k: dict(magic))
        return th.run_mpw_precheck_native(str(sub))

    def test_unmeasured_magic_drc_blocks_but_is_not_reported_as_violations(
            self, monkeypatch, sub):
        magic = unmeasured_drc(
            "Magic left NO DRC report at /x/precheck_magic_drc.rpt",
            report_path="/x/precheck_magic_drc.rpt")
        res = self._run(monkeypatch, sub, magic=magic)

        # Fail-closed: a shuttle submission is not signed off without a DRC
        # MEASUREMENT ...
        assert res["pass"] is False
        joined = " ".join(res["errors"])
        # ... but it is NOT described as a violation count.
        assert "COULD NOT BE MEASURED" in joined
        assert "-1 violations" not in joined
        assert res["checks"]["magic_drc"]["status"] == STATUS_NOT_RUN
        assert res["checks"]["magic_drc"]["violation_count"] is None

    def test_measured_violations_still_reported_as_violations(self, monkeypatch,
                                                             sub):
        rpt = sub / "m.rpt"
        rpt.write_text(REPORT_3)
        magic = classify_drc(violation_count=5, report_path=str(rpt))
        res = self._run(monkeypatch, sub, magic=magic)

        assert res["pass"] is False
        assert any("5 violation(s)" in e for e in res["errors"])

    def test_clean_magic_drc_still_passes(self, monkeypatch, sub):
        rpt = sub / "m.rpt"
        rpt.write_text(REPORT_CLEAN)
        magic = classify_drc(violation_count=0, report_path=str(rpt))
        res = self._run(monkeypatch, sub, magic=magic)

        assert res["pass"] is True, res["errors"]

    def test_unmeasured_advisory_klayout_warns_without_blocking(self, monkeypatch,
                                                                sub):
        rpt = sub / "m.rpt"
        rpt.write_text(REPORT_CLEAN)
        res = self._run(
            monkeypatch, sub,
            magic=classify_drc(violation_count=0, report_path=str(rpt)),
            klayout=unmeasured_drc("KLayout wrote no XML report",
                                   tool="KLayout"))

        # Advisory + unmeasured: warned, not a hard fail (same posture a
        # missing binary always had).
        assert res["pass"] is True, res["errors"]
        assert any("COULD NOT BE MEASURED" in w for w in res["warnings"])

    def test_measured_klayout_violations_still_block(self, monkeypatch, sub):
        rpt = sub / "m.rpt"
        rpt.write_text(REPORT_CLEAN)
        krpt = sub / "k.xml"
        krpt.write_text("<report><item></item></report>")
        res = self._run(
            monkeypatch, sub,
            magic=classify_drc(violation_count=0, report_path=str(rpt)),
            klayout=classify_drc(violation_count=1, report_path=str(krpt),
                                 tool="KLayout"))

        assert res["pass"] is False
        assert any("1 violation(s)" in w for w in res["warnings"])


# ---------------------------------------------------------------------------
# Consumer: the tapeout diagnosis agent's context formatters
# ---------------------------------------------------------------------------

class TestDiagnosisFormatting:
    """The diagnosis agent must not be told to hunt violations that were never
    measured -- it renders the precheck checks and the DRC result for an LLM."""

    def test_unmeasured_check_is_not_rendered_as_FAIL(self):
        from orchestrator.architecture.specialists.tapeout_diagnosis import (
            _format_precheck,
        )
        text = _format_precheck({
            "pass": False,
            "checks": {
                "structure": {"pass": True},
                "magic_drc": unmeasured_drc(
                    "Magic left NO DRC report at /x/precheck_magic_drc.rpt"),
            },
            "errors": ["Magic DRC COULD NOT BE MEASURED: ..."],
        })
        assert "magic_drc: NOT MEASURED" in text
        assert "magic_drc: FAIL" not in text
        assert "NO DRC report" in text
        assert "structure: PASS" in text

    def test_measured_violation_check_still_renders_FAIL(self, tmp_path):
        from orchestrator.architecture.specialists.tapeout_diagnosis import (
            _format_precheck,
        )
        rpt = tmp_path / "m.rpt"
        rpt.write_text(REPORT_3)
        text = _format_precheck({
            "pass": False,
            "checks": {"magic_drc": classify_drc(violation_count=4,
                                                 report_path=str(rpt))},
        })
        assert "magic_drc: FAIL" in text

    def test_format_drc_unmeasured_says_no_count_exists(self, tmp_path):
        from orchestrator.architecture.specialists.tapeout_diagnosis import (
            _format_drc,
        )
        text = _format_drc(unmeasured_drc("magic wrote nothing"), tmp_path)
        assert "COULD NOT BE MEASURED" in text
        assert "NO violation count" in text
        assert "-1" not in text

    def test_format_drc_legacy_minus_one_is_labelled_not_measured(self, tmp_path):
        from orchestrator.architecture.specialists.tapeout_diagnosis import (
            _format_drc,
        )
        text = _format_drc({"clean": False, "violation_count": -1}, tmp_path)
        assert "Violation count: NOT MEASURED" in text

    def test_format_drc_measured_count_is_unchanged(self, tmp_path):
        from orchestrator.architecture.specialists.tapeout_diagnosis import (
            _format_drc,
        )
        text = _format_drc({"clean": False, "violation_count": 9}, tmp_path)
        assert "Violation count: 9" in text
