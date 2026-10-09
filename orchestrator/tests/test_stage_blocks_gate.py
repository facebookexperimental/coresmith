# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""``coresmith stage next`` out of ``blocks`` needs the block deliverables:
a published ``best`` per block -- authored module or engine primitive --
that names a current ``completed`` recorded build, and every must-have item
owned by a published block verified at RTL level (with the measured value
when bounded). A pass published by hand or by ``block-done`` outside a
build is not a deliverable, and a primitive's materialized file alone is
not a graph receipt. There is no switch that opens the exit; the graph's
``record_entered`` never bypasses it."""
from __future__ import annotations

import pytest

from orchestrator.state_store import builds as B
from orchestrator.state_store import stages as st
from orchestrator.tests.build_fixtures import add_primitive, ready_project
from orchestrator.tests.test_build_identity import _complete_build


def _project(tmp_path, monkeypatch=None):
    """``blocks`` active with the earlier stages genuinely holding: two
    modules (sram owns PERF-001, uart owns FUNC-001) and an engine primitive."""
    db = ready_project(tmp_path, _project.mp, stage="blocks", modules=("sram", "uart"), with_rtl=True)
    add_primitive(db, tmp_path, "fab")
    # the assembled SoC model now has one more block: the declared check is
    # re-evaluated against the assembled model as it is
    db.add_check("PERF-001", "model_eval", None, value=50.0, sha=B.model_check_sha(db, tmp_path, "PERF-001"))
    db.unlink_items("FUNC-001", "block:sram", "owned_by")
    db.link_items("FUNC-001", "block:uart", "owned_by")
    # PERF-001 is sram's module-level target, measured by its acceptance test
    for v in db.verifiers(item_id="PERF-001"):
        db.remove_verifier(v["id"])
    (tmp_path / "tb" / "cocotb" / "test_sram.py").write_text(
        "import cocotb\n\n@cocotb.test()\nasync def test_perf(dut):\n    pass\n")
    db.add_verifier("PERF-001", "cocotb", path="tb/cocotb/test_sram.py", entry="test_perf", block="sram")
    assert st.current(db) == "blocks" and st.regressed_stages(db, tmp_path) == []
    return db


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    monkeypatch.setenv("CORESMITH_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("CORESMITH_FANOUT", "block")
    _project.mp = monkeypatch


def _codes(res):
    return {b["code"]: b["ids"] for b in res["blocked_by"]}


def _materialize_fabric(tmp_path):
    (tmp_path / "rtl" / "interconnect").mkdir(parents=True, exist_ok=True)
    (tmp_path / "rtl" / "interconnect" / "fab.v").write_text("module fab; endmodule\n")


def _build_fabric(db, tmp_path):
    _materialize_fabric(tmp_path)
    return _complete_build(db, tmp_path, "fab")


def test_zero_deliverables_refuses(tmp_path):
    db = _project(tmp_path)
    res = st.entry(db, tmp_path, "blocks")
    codes = {b["code"]: b["ids"] for b in res}
    assert codes["BLOCKS_UNPUBLISHED"] == ["sram", "uart"]
    assert codes["PRIMITIVE_UNMATERIALIZED"] == ["fab"]


def test_a_materialized_primitive_without_a_graph_build_is_not_a_receipt(tmp_path):
    db = _project(tmp_path)
    _materialize_fabric(tmp_path)
    codes = {b["code"]: b["ids"] for b in st.entry(db, tmp_path, "blocks")}
    assert "PRIMITIVE_UNMATERIALIZED" not in codes
    assert codes["PRIMITIVE_NOT_BUILT"] == ["fab (materialized, no graph build)"]
    db.set_result("fab", "best", {"done": True})                         # a hand claim is no better
    codes = {b["code"]: b["ids"] for b in st.entry(db, tmp_path, "blocks")}
    assert codes["PRIMITIVE_NOT_BUILT"][0].startswith("fab (") and "names no recorded build" in codes["PRIMITIVE_NOT_BUILT"][0]
    _complete_build(db, tmp_path, "fab")                                 # the graph's own build of the primitive
    assert "PRIMITIVE_NOT_BUILT" not in {b["code"] for b in st.entry(db, tmp_path, "blocks")}


def test_a_manual_publication_is_not_a_deliverable(tmp_path):
    db = _project(tmp_path)
    _build_fabric(db, tmp_path)
    db.set_result("sram", "best", {"done": True})                       # a hand-written / block-done claim
    _complete_build(db, tmp_path, "uart")
    codes = {b["code"]: b["ids"] for b in st.entry(db, tmp_path, "blocks")}
    assert "BLOCKS_UNPUBLISHED" not in codes
    assert len(codes["BLOCK_NOT_BUILT"]) == 1 and codes["BLOCK_NOT_BUILT"][0].startswith("sram (")
    assert "names no recorded build" in codes["BLOCK_NOT_BUILT"][0]


def test_published_builds_still_need_their_owned_items_measured(tmp_path):
    db = _project(tmp_path)
    _build_fabric(db, tmp_path)
    _complete_build(db, tmp_path, "sram")
    _complete_build(db, tmp_path, "uart")
    # a current model-level verdict is the uarch deliverable, not the block one
    db.add_check("PERF-001", "model_eval", "pass", value=12.0, sha=B.model_check_sha(db, tmp_path, "PERF-001"))
    db.add_check("FUNC-001", "block_dv", "pass")
    codes = _codes(st.advance(db, tmp_path))
    assert "BLOCKS_UNPUBLISHED" not in codes and "BLOCK_NOT_BUILT" not in codes
    assert codes["OWNED_ITEM_UNVERIFIED"] == ["PERF-001"]
    db.add_check("PERF-001", "block_dv", "pass")                     # bounded, verdict without the number
    assert _codes(st.advance(db, tmp_path))["BOUNDED_ITEM_UNMEASURED"] == ["PERF-001"]
    db.add_check("PERF-001", "block_dv", value=11.5)
    res = st.advance(db, tmp_path)
    assert res["advanced"] and res["done"] == "blocks" and res["stage"] == "integration"


def test_there_is_no_switch_that_opens_the_exit(tmp_path, monkeypatch):
    db = _project(tmp_path)
    monkeypatch.setenv("CORESMITH_STAGE_BLOCKS_GATE", "0")
    res = st.advance(db, tmp_path)
    assert not res["advanced"] and "BLOCKS_UNPUBLISHED" in _codes(res)
    assert not hasattr(st, "stage_blocks_gate_enabled")


def test_chip_scoped_item_is_deferred_to_later_stage(tmp_path):
    db = _project(tmp_path)
    # measured only at chip scope: not a module target of sram (a binding
    # change is a new target allocation, so it precedes the builds)
    for v in db.verifiers(item_id="PERF-001"):
        db.remove_verifier(v["id"])
    db.add_verifier("PERF-001", "chip", entry="test_system_rate")
    _build_fabric(db, tmp_path)
    _complete_build(db, tmp_path, "sram")
    _complete_build(db, tmp_path, "uart")
    db.add_check("FUNC-001", "block_dv", "pass")
    result = st.advance(db, tmp_path)
    assert result["advanced"] is True


def test_the_graph_cannot_record_integration_past_a_failed_blocks_gate(tmp_path):
    db = _project(tmp_path)
    assert st.entry(db, tmp_path, "blocks")                         # the CLI would refuse
    result = st.record_entered(db, tmp_path, "integration")
    assert result["recorded"] is False
    assert result["blocked_stage"] == "blocks"
    assert st.current(db) == "blocks"


def test_a_failed_must_have_blocks_the_exit_but_not_the_repair_build(tmp_path):
    """A failing RTL verdict is a downstream result: the stage machine does
    not advance past it, but it is not a regression of the architecture and
    never refuses the build that would repair it."""
    from orchestrator import module_build as MB
    db = _project(tmp_path)
    _build_fabric(db, tmp_path)
    _complete_build(db, tmp_path, "sram")
    _complete_build(db, tmp_path, "uart")
    db.add_check("FUNC-001", "block_dv", "pass")
    db.add_check("PERF-001", "block_dv", value=500.0)                 # 500 > bound_max 100: failed
    assert db.item("PERF-001")["status"] == "failed"
    res = st.advance(db, tmp_path)
    assert not res["advanced"] and _codes(res)["MUST_HAVE_FAILED"] == ["PERF-001"]
    assert st.regressed_stages(db, tmp_path) == []
    assert MB.readiness_refusal(db, tmp_path, "sram") is None
    assert MB.pipeline_start_refusal(db, tmp_path) is None
