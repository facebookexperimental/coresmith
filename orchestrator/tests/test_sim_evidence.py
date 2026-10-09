# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Receipts cannot borrow tests from another candidate or a later run."""
import os

import pytest

from orchestrator.harness.sim_evidence import capture, input_hashes, validate
from orchestrator.langgraph.final_report import _dv_latest
from orchestrator.state_store.project_db import open_project


def fixture(tmp_path):
    rtl = tmp_path / "decoder.v"
    tb = tmp_path / "test_decoder.py"
    xml = tmp_path / "results.xml"
    rtl.write_text("module decoder; endmodule")
    tb.write_text("# test input")
    xml.write_text('<testsuite><testcase name="first"/><testcase name="second"/></testsuite>')
    return rtl, tb, xml


def test_counts_come_from_results_and_survive_timestamp_changes(tmp_path):
    rtl, tb, xml = fixture(tmp_path)
    receipt = capture(xml, input_hashes([rtl, tb]))
    os.utime(xml, (1, 1))
    assert validate(receipt)["tests_total"] == 2


@pytest.mark.parametrize("which", [0, 1, 2])
def test_changed_candidate_test_or_result_invalidates_receipt(tmp_path, which):
    paths = fixture(tmp_path)
    receipt = capture(paths[2], input_hashes(paths[:2]))
    p = paths[which]
    p.write_text(p.read_text() + " ")
    with pytest.raises(ValueError):
        validate(receipt)


@pytest.mark.parametrize("text", ['<testsuite/>', '<broken'])
def test_empty_or_malformed_results_cannot_pass(tmp_path, text):
    rtl, tb, xml = fixture(tmp_path)
    xml.write_text(text)
    with pytest.raises(ValueError):
        capture(xml, input_hashes([rtl, tb]))


def test_failed_or_skipped_tests_are_not_passes(tmp_path):
    rtl, tb, xml = fixture(tmp_path)
    xml.write_text('<testsuite><testcase name="a"><failure/></testcase><testcase name="b"><skipped/></testcase></testsuite>')
    result = capture(xml, input_hashes([rtl, tb]))
    assert not result["passed"]
    assert result["tests_failed"] == result["tests_skipped"] == 1


class EmptyScoreboard:
    def dv_results(self, **kwargs):
        return []


def test_published_result_without_receipt_is_unverified(tmp_path):
    fixture(tmp_path)
    open_project(tmp_path).set_result("decoder", "dv_best", {"ts": 1, "sim_passed": True})
    result = _dv_latest(EmptyScoreboard(), str(tmp_path), "decoder")
    assert result["passed"] is False
    assert "evidence_error" in result


def test_published_result_requires_the_matching_receipt(tmp_path):
    rtl, tb, xml = fixture(tmp_path)
    evidence = capture(xml, input_hashes([rtl, tb]))
    open_project(tmp_path).set_result("decoder", "dv_best", {"ts": 1, "sim_passed": True, "simulation_evidence": evidence})
    assert _dv_latest(EmptyScoreboard(), str(tmp_path), "decoder")["tests_total"] == 2
    xml.write_text('<testsuite><testcase name="unrelated"/></testsuite>')
    assert _dv_latest(EmptyScoreboard(), str(tmp_path), "decoder")["passed"] is False
