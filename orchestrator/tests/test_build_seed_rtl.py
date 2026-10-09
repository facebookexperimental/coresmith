# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""``coresmith build module --seed-rtl``: an existing source of the bound
target is the starting implementation. It reaches lint, DV, coverage and
synthesis through the graph without an initial regeneration; its hash is
kept in the build record; a later repair in the normal loop may change the
working implementation and the record says so. The seed is validated before
any binding is touched or any build allocated, and the build never rebinds
a target to point at a seed."""
from __future__ import annotations

import asyncio
import json
import types

import pytest

from orchestrator.state_store import builds as B
from orchestrator.tests.build_fixtures import (
    events,
    fake_graph_helpers,
    make_build_lifecycle,
    ready_project,
)
from orchestrator.tests.test_build_lifecycle import _settle, _start


@pytest.fixture
def daemon(tmp_path, monkeypatch):
    from orchestrator.daemon import server
    db = ready_project(tmp_path, monkeypatch, with_rtl=True)
    lc = make_build_lifecycle(tmp_path)
    monkeypatch.setattr(server, "_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr(server, "_build", lc)
    monkeypatch.setattr(server, "_pipeline", types.SimpleNamespace(task=None, thread_id="pipeline", status="idle"))
    monkeypatch.setattr(server, "_apply_run_env", lambda where: [])
    monkeypatch.setattr(server, "_preflight_or_400", lambda: None)
    yield server, db, lc
    asyncio.run(lc.cleanup())


async def test_seed_is_linted_verified_and_measured_without_regeneration(daemon, tmp_path, monkeypatch):
    server, db, lc = daemon
    calls = fake_graph_helpers(monkeypatch, tmp_path, calls={})
    seed = tmp_path / "rtl" / "tiny.v"
    seed_sha = B.file_sha256(seed)
    out = await _start(server, seed_rtl="rtl/tiny.v")
    assert out["seed"]["sha256"] == seed_sha
    row = await _settle(db, out["build_id"])
    assert row["status"] == "completed", row
    assert "generate_rtl" not in calls                       # no regeneration
    assert calls["lint_rtl"] >= 1 and calls["run_simulation"] >= 1 and calls["synthesize_block"] >= 1
    assert row["seed"]["sha256"] == seed_sha and row["inputs"]["seed"]["sha256"] == seed_sha
    assert row["result"]["seed_modified_by_repair"] is False
    assert B.file_sha256(seed) == seed_sha                   # untouched
    assert any(e.get("event") == "seed_rtl_used" and e.get("build_id") == out["build_id"] for e in events(tmp_path))
    # the seeded target was bound by bytes: editing it makes the pass stale
    assert B.current_build_status(db, tmp_path, "tiny")["ok"]
    seed.write_text(seed.read_text() + "// edit\n")
    assert not B.current_build_status(db, tmp_path, "tiny")["ok"]


async def test_seed_repair_is_traced_not_silent(daemon, tmp_path, monkeypatch):
    from orchestrator.langgraph import pipeline_graph as pg
    server, db, lc = daemon
    seed = tmp_path / "rtl" / "tiny.v"
    seed_sha = B.file_sha256(seed)
    calls = fake_graph_helpers(monkeypatch, tmp_path, lint_clean=False, calls={})

    async def fix_lint(block_name, rtl_path, log_path, callbacks=None):
        seed.write_text("module tiny(input clk, input rst_n, output reg [7:0] m_out); endmodule // repaired\n")
        monkeypatch.setattr(pg, "lint_rtl", lambda *a, **k: {"clean": True, "warnings": ""})
        return "fixed"
    monkeypatch.setattr(pg, "fix_lint_errors", fix_lint)
    out = await _start(server, seed_rtl="rtl/tiny.v")
    row = await _settle(db, out["build_id"])
    assert row["status"] == "completed", row
    assert "generate_rtl" not in calls
    assert row["result"]["seed_modified_by_repair"] is True
    assert row["seed"]["sha256"] == seed_sha and B.file_sha256(seed) != seed_sha
    ev = events(tmp_path)
    assert any(e.get("event") == "seed_rtl_used" for e in ev)
    assert any(e.get("node") == "Lint Fix" and e.get("event") == "llm_end" and e.get("fix_produced") for e in ev)


async def test_seed_validation_precedes_allocation_and_never_rebinds(daemon, tmp_path, monkeypatch):
    from orchestrator.harness import targets as T
    server, db, lc = daemon
    fake_graph_helpers(monkeypatch, tmp_path)
    before = T.load(tmp_path, "tiny", require_files=False)
    resp = await server._start_module_build(server.BuildModuleRequest(module="tiny", seed_rtl="rtl/missing.v"),
                                            entry="build_module")
    assert resp.status_code == 400 and json.loads(resp.body)["error"] == "SEED_MISSING"
    (tmp_path / "rtl" / "other.v").write_text("module other(); endmodule\n")
    resp = await server._start_module_build(server.BuildModuleRequest(module="tiny", seed_rtl="rtl/other.v"),
                                            entry="build_module")
    assert resp.status_code == 400 and json.loads(resp.body)["error"] == "SEED_NOT_BOUND"
    assert T.load(tmp_path, "tiny", require_files=False) == before
    assert B.builds_for(db) == []
