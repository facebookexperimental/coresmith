# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""A bounded FRD item is decided at the model rank by a measured value, not
by the harness's own verdict (run 2: PERF-001..005 became ``verified`` from
``model_eval pass`` with ``value=NULL`` before any RTL existed).
``CORESMITH_MODEL_EVAL_REQUIRE_VALUE=0`` restores verdict-only judging."""
import asyncio
from pathlib import Path

import pytest

from orchestrator.state_store import stages as st
from orchestrator.state_store.project_db import open_project
from orchestrator.systemc_model import frd_eval as fe

# the harness output of a fake architecture model
_OUT = """[frd_eval] booting
FRD_EVAL {"id": "PERF-001", "status": "pass", "value": 207, "unit": "cycles", "evidence": "start->done"}
FRD_EVAL {"id": "PERF-002", "status": "pass", "evidence": "128+2+207+2+128 = 467 (arithmetic)"}
FRD_EVAL {"id": "PERF-003", "status": "pass", "evidence": "SQNR measured below"}
VALUE PERF-003 58.2 dB
FRD_EVAL {"id": "PERF-006", "status": "not_testable", "evidence": "cell count is a synthesis metric"}
FRD_EVAL {"id": "INV-001", "status": "pass", "evidence": "bit-exact vs golden"}
FRD_EVAL_DONE
"""


def _reqs():
    return [
        {"id": "PERF-001", "priority": "must_have", "metric": "latency", "bound_min": None, "bound_max": 400.0},
        {"id": "PERF-002", "priority": "must_have", "metric": "period", "bound_min": None, "bound_max": 500.0},
        {"id": "PERF-003", "priority": "must_have", "metric": "sqnr", "bound_min": 60.0, "bound_max": None},
        {"id": "PERF-006", "priority": "must_have", "metric": "cells", "bound_min": None, "bound_max": 60000.0},
        {"id": "INV-001", "priority": "must_have"},
    ]


def _db(tmp_path):
    db = open_project(tmp_path)
    db.ensure_db_artifact("frd")
    for q in _reqs():
        db.upsert_item("frd", {"id": q["id"], "kind": q["id"].split("-")[0], "text": q["id"],
                               "priority": "must_have", "metric": q.get("metric"),
                               "bound_min": q.get("bound_min"), "bound_max": q.get("bound_max")})
    return db


def test_parse_carries_json_values_and_value_lines():
    res = {r["id"]: r for r in fe.parse_results(_OUT)}
    assert res["PERF-001"]["value"] == 207 and res["PERF-001"]["unit"] == "cycles"
    assert res["PERF-002"]["value"] is None
    assert res["PERF-003"]["value"] == pytest.approx(58.2) and res["PERF-003"]["unit"] == "dB"
    assert res["INV-001"]["value"] is None


def test_bounded_items_are_judged_by_the_value(monkeypatch):
    monkeypatch.delenv("CORESMITH_MODEL_EVAL_REQUIRE_VALUE", raising=False)
    reqs = _reqs()
    res = fe.judge_values(reqs, fe.parse_results(_OUT))
    by = {r["id"]: r for r in res}
    assert by["PERF-001"]["status"] == "pass"                      # 207 <= 400
    assert by["PERF-002"]["status"] == "tool_error" and by["PERF-002"]["no_value"]
    assert by["PERF-002"]["evidence"].startswith("no measured value from the model harness")
    assert by["PERF-003"]["status"] == "fail" and by["PERF-003"]["harness_status"] == "pass"   # 58.2 < 60
    assert by["PERF-006"]["status"] == "not_testable"             # a reasoned not_testable stands
    assert by["INV-001"]["status"] == "pass"                      # unbounded: the verdict stands
    s = fe.summarize(reqs, res)
    assert s["no_value"] == ["PERF-002"] and s["failed"] == ["PERF-003"] and s["gate_ok"] is False
    assert s["counts"]["tool_error"] == 1


def test_checks_carry_the_value_and_a_model_failure_is_advisory(tmp_path, monkeypatch):
    monkeypatch.delenv("CORESMITH_MODEL_EVAL_REQUIRE_VALUE", raising=False)
    db = _db(tmp_path)
    reqs = fe.overlay_bounds([{"id": q["id"], "priority": "must_have"} for q in _reqs()], db)
    assert reqs[0]["bound_max"] == 400.0                          # the DB bounds are authoritative
    res = fe.judge_values(reqs, fe.parse_results(_OUT))
    fe.record_checks(db, reqs, res, sha="s", actor="frd_eval_arch")
    latest = {c["item_id"]: c for c in db.checks(kind="model_eval", latest=True)}
    assert (latest["PERF-001"]["status"], latest["PERF-001"]["value"]) == ("pass", 207.0)
    assert latest["PERF-001"]["evidence"].startswith("207 cycles")
    assert (latest["PERF-002"]["status"], latest["PERF-002"]["value"]) == ("tool_error", None)
    assert (latest["PERF-003"]["status"], latest["PERF-003"]["value"]) == ("fail", pytest.approx(58.2))
    assert db.item("PERF-001")["status"] == "verified" and db.item("PERF-002")["status"] == "open"
    # the model-level failure stays on the item (status failed) but is advisory:
    # no stage blocker, an advisory naming it
    assert db.item("PERF-003")["status"] == "failed"
    assert not any(b["code"] == "MUST_HAVE_FAILED" for s in st.STAGES for b in st.entry(db, tmp_path, s))
    assert {a["code"]: a["ids"] for a in st.advisories(db)} == {"MODEL_ONLY_FAILED": ["PERF-003"]}
    # the harness is fixed: the value arrives, the advisory goes
    db.add_check("PERF-002", "model_eval", value=467)
    db.add_check("PERF-003", "model_eval", value=74.5)
    assert st.advisories(db) == [] and db.item("PERF-003")["status"] == "verified"


def test_a_valueless_model_pass_never_satisfies_the_rtl_gate(tmp_path, monkeypatch):
    """A model pass (value or not) is below the RTL rank: an owned bounded
    must-have still needs a measured block_dv check at the blocks gate."""
    monkeypatch.delenv("CORESMITH_MODEL_EVAL_REQUIRE_VALUE", raising=False)
    monkeypatch.setenv("CORESMITH_PROJECT_ROOT", str(tmp_path))
    from orchestrator.tests.build_fixtures import complete_build
    db = _db(tmp_path)
    db.import_block_diagram({"blocks": [{"name": "fft", "tier": 1}], "connections": []})
    for q in _reqs():
        db.link_items(q["id"], "block:fft", "owned_by")
        db.add_check(q["id"], "model_eval", "pass", evidence="said so")
    (tmp_path / "rtl").mkdir()
    (tmp_path / "rtl" / "fft.v").write_text("module fft(); endmodule\n")
    for i, stage in enumerate(st.STAGES[:st.STAGES.index("blocks")]):
        db.stage_set(stage, i, "done")
    db.stage_set("blocks", st.STAGES.index("blocks"), "active")
    db.set_result("fft", "best", {"done": True})                      # a hand claim publishes nothing
    assert "BLOCK_NOT_BUILT" in {b["code"] for b in st.entry(db, tmp_path, "blocks")}
    complete_build(db, tmp_path, "fft")                               # the block's recorded build
    codes = {b["code"]: b["ids"] for b in st.entry(db, tmp_path, "blocks")}
    assert codes["OWNED_ITEM_UNVERIFIED"] == ["INV-001", "PERF-001", "PERF-002", "PERF-003", "PERF-006"]
    db.add_check("PERF-001", "block_dv", "pass")                     # bounded, RTL rank, no number
    assert "PERF-001" in st.entry(db, tmp_path, "blocks")[0]["ids"] or \
        "PERF-001" in {b["code"]: b["ids"] for b in st.entry(db, tmp_path, "blocks")}["BOUNDED_ITEM_UNMEASURED"]
    db.add_check("PERF-001", "block_dv", value=350)
    assert "PERF-001" not in {i for b in st.entry(db, tmp_path, "blocks") for i in b["ids"]}


def test_require_value_off_restores_verdict_only(tmp_path, monkeypatch):
    monkeypatch.setenv("CORESMITH_MODEL_EVAL_REQUIRE_VALUE", "0")
    db = _db(tmp_path)
    reqs = _reqs()
    res = fe.judge_values(reqs, fe.parse_results(_OUT))
    assert {r["id"]: r["status"] for r in res}["PERF-002"] == "pass"
    assert {r["id"]: r["status"] for r in res}["PERF-003"] == "pass"   # the harness verdict, not 58.2 vs 60
    assert fe.summarize(reqs, res)["gate_ok"] is True and fe.summarize(reqs, res)["no_value"] == []
    fe.record_checks(db, reqs, res)
    latest = {c["item_id"]: c for c in db.checks(kind="model_eval", latest=True)}
    assert latest["PERF-002"]["status"] == "pass" and latest["PERF-001"]["value"] is None


class _Agent:
    async def generate(self, *, project_root, **k):
        d = Path(project_root) / "model" / "frd_eval"
        d.mkdir(parents=True, exist_ok=True)
        (d / "frd_eval.cpp").write_text("// fake\n")
        return {}


def test_evaluate_end_to_end_on_a_fake_harness(tmp_path, monkeypatch):
    monkeypatch.delenv("CORESMITH_MODEL_EVAL_REQUIRE_VALUE", raising=False)
    db = _db(tmp_path)
    (tmp_path / "arch").mkdir()
    (tmp_path / "arch" / "frd_spec.md").write_text("\n".join(
        f"1. **ID**: {q['id']}\n   - **Requirement**: {q['id']}\n   - **Priority**: must_have\n" for q in _reqs()))
    md = tmp_path / "model"
    md.mkdir()
    monkeypatch.setattr(fe, "build_harness", lambda md, **k: {"ok": True, "log": ""})
    monkeypatch.setattr(fe, "run_harness", lambda md, **k: {"ok": True, "done": True, "rc": 0, "log": _OUT,
                                                             "results": fe.parse_results(_OUT)})
    rec = asyncio.run(fe.evaluate(tmp_path, md, ["fft"], arch=True, agent=_Agent(), db=db, record_sha="s"))
    assert rec["gate_ok"] is False and rec["summary"]["no_value"] == ["PERF-002"]
    import json
    reqs = json.loads((md / "frd_eval" / "requirements.json").read_text())["requirements"]
    assert {r["id"]: r.get("bound_max") for r in reqs}["PERF-001"] == 400.0   # the harness sees the bounds
    report = (md / "frd_eval" / "REPORT.md").read_text()
    assert "| PERF-001 | must_have | pass | 207 cycles | [..400] |" in report
    assert "| PERF-002 | must_have | tool_error |  | [..500] | no measured value" in report
    latest = {c["item_id"]: c for c in db.checks(kind="model_eval", latest=True)}
    assert latest["PERF-001"]["value"] == 207.0 and latest["PERF-002"]["status"] == "tool_error"
