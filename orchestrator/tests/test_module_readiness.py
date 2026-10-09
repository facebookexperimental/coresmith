# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Architecture readiness is deterministic and declared: a module is built
only with its registered uArch spec, its bound target, a built and smoked
SystemC reference model whose recorded hash is the current bytes at the
current contract version, every model check its requirements declare
passing on the current model + harness digest, and an explicit worker
binding. The shared stages must hold on re-evaluation; a project with no
stage rows starts at requirements; a refused build runs no worker and no
tool and records nothing."""
from __future__ import annotations

import asyncio
import json
import types

import pytest

from orchestrator import module_build as MB
from orchestrator.state_store import builds as B
from orchestrator.state_store import stages as st
from orchestrator.tests.build_fixtures import (
    fake_graph_helpers,
    ready_project,
    record_model_build,
    write_env,
)


def _codes(blockers):
    return {b["code"]: b["ids"] for b in blockers}


def test_ready_project_has_no_blockers(tmp_path, monkeypatch):
    db = ready_project(tmp_path, monkeypatch)
    assert st.shared_ready(db, tmp_path) == []
    assert st.module_ready(db, tmp_path, "tiny") == []
    assert MB.readiness_refusal(db, tmp_path, "tiny") is None
    s = st.status(db, tmp_path)
    assert s["stage"] == "uarch" and s["can_advance"] and s["regressed"] == []


def test_no_stage_rows_means_requirements_not_a_legacy_exemption(tmp_path, monkeypatch):
    from orchestrator.state_store.project_db import open_project
    monkeypatch.setenv("CORESMITH_PROJECT_ROOT", str(tmp_path))
    db = open_project(tmp_path)
    assert st.current(db) == "requirements"
    assert _codes(st.shared_ready(db, tmp_path)) == {"STAGE_MACHINE_UNUSED": []}
    refusal = MB.pipeline_start_refusal(db, tmp_path)
    assert refusal is not None and refusal.code == "STAGE_MACHINE_UNUSED"
    assert st.status(db, tmp_path)["unused"] is True


def test_shared_stage_before_uarch_refuses(tmp_path, monkeypatch):
    db = ready_project(tmp_path, monkeypatch, stage="interfaces")
    assert _codes(st.shared_ready(db, tmp_path)) == {"STAGE_BEFORE_UARCH": ["interfaces"]}
    refusal = MB.readiness_refusal(db, tmp_path, "tiny")
    assert refusal.code == "ARCHITECTURE_NOT_READY"


def test_done_stage_that_no_longer_holds_is_regressed_not_erased(tmp_path, monkeypatch):
    db = ready_project(tmp_path, monkeypatch)
    # an interfaces deliverable disappears after the stage was marked done
    (tmp_path / ".coresmith" / "vip_index.json").unlink()
    reg = st.regressed_stages(db, tmp_path)
    assert [r["stage"] for r in reg] == ["interfaces"]
    assert "VIP_MISSING" in reg[0]["ids"]
    rows = {r["name"]: r["status"] for r in db.stage_rows()}
    assert rows["interfaces"] == "done"                      # history is kept
    assert st.status(db, tmp_path)["can_advance"] is False   # effective completion is not
    assert not st.advance(db, tmp_path)["advanced"]
    assert MB.readiness_refusal(db, tmp_path, "tiny").code == "ARCHITECTURE_NOT_READY"


@pytest.mark.parametrize("break_it,code", [
    ("spec", "UARCH_MISSING"),
    ("target", "TARGET_UNBOUND"),
    ("model_file", "MODEL_MISSING"),
    ("model_row", "MODEL_NOT_BUILT"),
    ("model_bytes", "MODEL_STALE"),
    ("model_header", "MODEL_STALE"),
    ("model_row_without_deps", "MODEL_STALE"),
    ("contract_version", "MODEL_STALE"),
    ("worker", "WORKER_UNBOUND"),
])
def test_each_declared_input_is_required(tmp_path, monkeypatch, break_it, code):
    db = ready_project(tmp_path, monkeypatch)
    if break_it == "spec":
        with db._tx() as con:
            con.execute("DELETE FROM artifacts WHERE kind='uarch:tiny'")
    elif break_it == "target":
        with db._tx() as con:
            con.execute("DELETE FROM run_flags WHERE name='target:tiny'")
    elif break_it == "model_file":
        (tmp_path / "model" / "tiny_model.cpp").unlink()
    elif break_it == "model_row":
        record_model_build(db, tmp_path, "tiny", smoke_ok=False)
    elif break_it == "model_bytes":
        (tmp_path / "model" / "tiny_model.cpp").write_text("// edited after the recorded build\n")
    elif break_it == "model_header":
        # the header the model includes is compiled into it: an edit after the
        # recorded build is a stale build even though the .cpp is unchanged
        (tmp_path / "model" / "tiny_model.h").write_text("// header edited after the build\n")
    elif break_it == "model_row_without_deps":
        # a row recorded before dependency tracking cannot vouch for the headers
        db.upsert_model("tiny", path="model/tiny_model.cpp", sha=B.model_sha16(tmp_path, "tiny"),
                        spec_contract_version=str(db.block_contract_version("tiny")), build_ok=True, smoke_ok=True)
    elif break_it == "contract_version":
        edge = dict(db.contracts()["contracts"][0], data_width_bits=16)
        db.import_contracts({"contracts": [edge]})
    elif break_it == "worker":
        write_env(tmp_path, {"CORESMITH_MODEL": "claude-test-model"})
        monkeypatch.delenv("CORESMITH_LLM_PROVIDER", raising=False)
    codes = _codes(st.module_ready(db, tmp_path, "tiny"))
    assert code in codes, codes
    refusal = MB.readiness_refusal(db, tmp_path, "tiny")
    assert refusal.code == "MODULE_NOT_READY" and code in _codes(refusal.blockers)


def test_contract_version_zero_is_not_stale(tmp_path, monkeypatch):
    """An isolated, pin-only module has contract version 0; ``model build``
    stores it as ``"0"`` and the ontology reports ``0``/``None``: the same
    version, never a stale build."""
    db = ready_project(tmp_path, monkeypatch, modules=("lone", "other"))
    db.import_contracts({"contracts": []})
    assert int(db.block_contract_version("lone") or 0) == 0
    for stored in ("0", "", None):
        db.upsert_model("lone", path="model/lone_model.cpp", sha=B.model_sha16(tmp_path, "lone"),
                        deps_sha=B.model_deps_sha(tmp_path, "lone"), spec_contract_version=stored,
                        build_ok=True, smoke_ok=True)
        assert "MODEL_STALE" not in _codes(st.module_ready(db, tmp_path, "lone")), stored


def test_declared_model_check_is_required_current_and_not_masked_by_rtl(tmp_path, monkeypatch):
    db = ready_project(tmp_path, monkeypatch, with_model_check=False)
    # no declaration -> nothing required beyond the built model
    assert "MODEL_CHECK_MISSING" not in _codes(st.module_ready(db, tmp_path, "tiny"))
    db.edit_item("PERF-001", model_check="cycle accounting")
    assert _codes(st.module_ready(db, tmp_path, "tiny"))["MODEL_CHECK_MISSING"] == ["PERF-001"]
    # a failing verdict on the current model blocks, and a later block_dv pass never masks it
    db.add_check("PERF-001", "model_eval", None, value=500.0, sha=B.model_check_sha(db, tmp_path, "PERF-001"))
    db.add_check("PERF-001", "block_dv", None, value=10.0)
    assert db.item("PERF-001")["status"] == "verified"           # the RTL pass decides the item status...
    assert _codes(st.module_ready(db, tmp_path, "tiny"))["MODEL_CHECK_FAILED"] == ["PERF-001"]   # ...not readiness
    # a passing verdict bound to the current digest clears it
    db.add_check("PERF-001", "model_eval", None, value=40.0, sha=B.model_check_sha(db, tmp_path, "PERF-001"))
    assert "MODEL_CHECK_FAILED" not in _codes(st.module_ready(db, tmp_path, "tiny"))
    # editing the harness changes the digest: the verdict is stale until re-evaluated
    (tmp_path / "model" / "frd_eval" / "frd_eval.cpp").write_text("// edited harness\n")
    assert _codes(st.module_ready(db, tmp_path, "tiny"))["MODEL_CHECK_STALE"] == ["PERF-001"]
    db.add_check("PERF-001", "model_eval", None, value=40.0, sha=B.model_check_sha(db, tmp_path, "PERF-001"))
    assert "MODEL_CHECK_STALE" not in _codes(st.module_ready(db, tmp_path, "tiny"))
    # the verdict is bound to the requirement it judged: a bound change re-evaluates
    db.edit_item("PERF-001", bound_max=30.0)
    assert _codes(st.module_ready(db, tmp_path, "tiny"))["MODEL_CHECK_STALE"] == ["PERF-001"]
    # the old 40 would now fail; the history is intact, nothing is re-judged silently
    assert [c["value"] for c in db.checks("PERF-001", kind="model_eval")][-1] == 40.0
    # a verdict with the model digest alone (an old record) is also stale, never silently accepted
    db.add_check("PERF-001", "model_eval", None, value=20.0, sha=B.soc_model_digest(tmp_path))
    assert _codes(st.module_ready(db, tmp_path, "tiny"))["MODEL_CHECK_STALE"] == ["PERF-001"]
    db.add_check("PERF-001", "model_eval", None, value=20.0, sha="")
    assert _codes(st.module_ready(db, tmp_path, "tiny"))["MODEL_CHECK_STALE"] == ["PERF-001"]


def test_a_transitive_model_include_edit_invalidates_the_build_and_the_declared_check(tmp_path, monkeypatch):
    """``tiny_model.cpp`` includes ``tiny_model.h`` which includes ``defs.h``:
    editing ``defs.h`` and rebuilding the model (re-stamping its row) leaves
    the OLD declared verdict stale, because the assembled-model digest is
    built from the same dependency manifest the model identity follows -- not
    from a fixed file list. A registered model path outside the convention
    is covered the same way."""
    from orchestrator.tests.build_fixtures import record_model_build
    db = ready_project(tmp_path, monkeypatch)
    md = tmp_path / "model"
    (md / "defs.h").write_text("#define W 8\n")
    (md / "tiny_model.h").write_text("// header of tiny\n#include \"defs.h\"\n")
    record_model_build(db, tmp_path, "tiny")
    db.add_check("PERF-001", "model_eval", None, value=50.0, sha=B.model_check_sha(db, tmp_path, "PERF-001"))
    assert st.module_ready(db, tmp_path, "tiny") == []
    (md / "defs.h").write_text("#define W 16\n")
    codes = _codes(st.module_ready(db, tmp_path, "tiny"))
    assert codes["MODEL_STALE"] == ["tiny"] and codes["MODEL_CHECK_STALE"] == ["PERF-001"]
    record_model_build(db, tmp_path, "tiny")                                   # rebuilt and re-stamped
    codes = _codes(st.module_ready(db, tmp_path, "tiny"))
    assert "MODEL_STALE" not in codes and codes["MODEL_CHECK_STALE"] == ["PERF-001"]   # the old verdict stays stale
    db.add_check("PERF-001", "model_eval", None, value=50.0, sha=B.model_check_sha(db, tmp_path, "PERF-001"))
    assert st.module_ready(db, tmp_path, "tiny") == []
    # a registered implementation outside the convention, with its own include
    impl = md / "impl" / "tiny_impl.cpp"
    impl.parent.mkdir()
    (impl.parent / "tiny_private.h").write_text("// private\n")
    impl.write_text('#include "../tiny_model.h"\n#include "tiny_private.h"\nint y;\n')
    db.upsert_model("tiny", path="model/impl/tiny_impl.cpp", sha="x", build_ok=True, smoke_ok=True)
    record_model_build(db, tmp_path, "tiny")
    db.add_check("PERF-001", "model_eval", None, value=50.0, sha=B.model_check_sha(db, tmp_path, "PERF-001"))
    assert st.module_ready(db, tmp_path, "tiny") == []
    (impl.parent / "tiny_private.h").write_text("// private v2\n")
    assert "MODEL_CHECK_STALE" in _codes(st.module_ready(db, tmp_path, "tiny"))
    # the harness's own includes are part of the digest too
    record_model_build(db, tmp_path, "tiny")
    db.add_check("PERF-001", "model_eval", None, value=50.0, sha=B.model_check_sha(db, tmp_path, "PERF-001"))
    (md / "frd_eval" / "frd_eval.cpp").write_text('#include "frd_util.h"\n// harness\n')
    (md / "frd_eval" / "frd_util.h").write_text("// util\n")
    db.add_check("PERF-001", "model_eval", None, value=50.0, sha=B.model_check_sha(db, tmp_path, "PERF-001"))
    assert st.module_ready(db, tmp_path, "tiny") == []
    (md / "frd_eval" / "frd_util.h").write_text("// util v2\n")
    assert _codes(st.module_ready(db, tmp_path, "tiny"))["MODEL_CHECK_STALE"] == ["PERF-001"]


def test_model_hash_construction_matches_model_build(tmp_path, monkeypatch):
    import hashlib
    db = ready_project(tmp_path, monkeypatch)
    cp = tmp_path / "model" / "tiny_model.cpp"
    assert B.model_sha16(tmp_path, "tiny") == hashlib.sha256(cp.read_bytes()).hexdigest()[:16]
    row = db.model_for("tiny")
    assert row["sha"] == B.model_sha16(tmp_path, "tiny") and row["deps_sha"] == B.model_deps_sha(tmp_path, "tiny")
    ident = B.model_identity(tmp_path, "tiny", db)
    assert str(tmp_path / "model" / "tiny_model.h") in ident["deps"]


def test_a_primitive_is_buildable_once_the_shared_stages_hold(tmp_path, monkeypatch):
    from orchestrator.tests.build_fixtures import add_primitive
    db = ready_project(tmp_path, monkeypatch)
    add_primitive(db, tmp_path, "fab")
    assert st.module_ready(db, tmp_path, "fab") == []
    assert MB.readiness_refusal(db, tmp_path, "fab") is None
    with pytest.raises(MB.BuildRefusal) as exc:
        MB.plan_module_build(db, tmp_path, "fab", entry="build_module", seed_rtl="rtl/tiny.v")
    assert exc.value.code == "SEED_NOT_ALLOWED"
    plan = MB.plan_module_build(db, tmp_path, "fab", entry="build_module")
    assert plan["primitive"] and plan["inputs"]["spec"] is None and MB.initial_block_state(tmp_path, plan)["preserve_testbench"]
    assert "fab" not in " ".join(b["code"] for b in st.entry(db, tmp_path, "uarch"))


def test_uarch_exit_requires_every_module_ready(tmp_path, monkeypatch):
    db = ready_project(tmp_path, monkeypatch)
    with db._tx() as con:
        con.execute("DELETE FROM run_flags WHERE name='target:sink'")
    res = st.advance(db, tmp_path)
    assert not res["advanced"] and _codes(res["blocked_by"])["TARGET_UNBOUND"] == ["sink"]
    # the selected module can still be built: readiness is per module
    assert MB.readiness_refusal(db, tmp_path, "tiny") is None
    assert MB.readiness_refusal(db, tmp_path, "sink").code == "MODULE_NOT_READY"


def test_shared_reference_model_change_invalidates_every_declared_check(tmp_path, monkeypatch):
    """The FRD is evaluated on the ASSEMBLED SoC model: another block's model
    bytes are part of the model tiny's declared check ran on."""
    db = ready_project(tmp_path, monkeypatch)
    (tmp_path / "model" / "sink_model.cpp").write_text("// other bytes\n")
    codes = _codes(st.module_ready(db, tmp_path, "tiny"))
    assert codes["MODEL_CHECK_STALE"] == ["PERF-001"] and "MODEL_STALE" not in codes
    assert _codes(st.module_ready(db, tmp_path, "sink"))["MODEL_STALE"] == ["sink"]


def test_cluster_fanout_is_refused_as_a_qualified_build(tmp_path, monkeypatch):
    db = ready_project(tmp_path, monkeypatch, stage="blocks")
    monkeypatch.setenv("CORESMITH_FANOUT", "cluster")
    assert MB.readiness_refusal(db, tmp_path, "tiny").code == "CLUSTER_FANOUT_UNSUPPORTED"
    assert MB.pipeline_start_refusal(db, tmp_path).code == "CLUSTER_FANOUT_UNSUPPORTED"
    monkeypatch.setenv("CORESMITH_FANOUT", "block")
    assert MB.pipeline_start_refusal(db, tmp_path) is None


def test_refused_build_runs_no_worker_no_tool_and_records_nothing(tmp_path, monkeypatch):
    """The persisted env is applied FIRST (readiness and the recorded
    identity see the authoritative configuration); the tool preflight, the
    workers, the ledger and the checkpoint come only after every refusal."""
    from orchestrator.daemon import server
    from orchestrator.tests.build_fixtures import make_build_lifecycle
    db = ready_project(tmp_path, monkeypatch)
    (tmp_path / "model" / "tiny_model.cpp").unlink()
    calls = fake_graph_helpers(monkeypatch, tmp_path, calls={})
    order: list[str] = []
    monkeypatch.setattr(server, "_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr(server, "_build", make_build_lifecycle(tmp_path))
    monkeypatch.setattr(server, "_pipeline", types.SimpleNamespace(task=None, thread_id="pipeline"))
    monkeypatch.setattr(server, "_apply_run_env", lambda where: order.append("env") or [])
    monkeypatch.setattr(server, "_preflight_or_400", lambda: (_ for _ in ()).throw(AssertionError("preflight ran")))
    resp = asyncio.run(server._start_module_build(server.BuildModuleRequest(module="tiny"), entry="build_module"))
    assert resp.status_code == 409
    body = json.loads(resp.body)
    assert body["error"] == "MODULE_NOT_READY" and "MODEL_MISSING" in _codes(body["blocked_by"])
    assert order == ["env"] and B.builds_for(db) == [] and calls == {}
    assert not (tmp_path / ".coresmith" / "build_checkpoint.db").exists()
    # the stage machine was not touched either
    assert st.current(db) == "uarch"


def test_a_persisted_env_edit_is_seen_by_readiness_and_the_recorded_identity(tmp_path, monkeypatch):
    """The worker binding is resolved from the persisted file as applied,
    not from a stale ambient value: the same configuration decides the
    refusal, is recorded, and runs the build."""
    from orchestrator.daemon import server
    from orchestrator.tests.build_fixtures import make_build_lifecycle
    db = ready_project(tmp_path, monkeypatch)
    monkeypatch.setenv("CORESMITH_LLM_PROVIDER", "codex")        # ambient: stale
    write_env(tmp_path, {"CORESMITH_LLM_PROVIDER": "claude", "CORESMITH_MODEL": "claude-test-model"})
    assert B.worker_binding(tmp_path)["provider"] == "claude"    # the persisted file wins
    write_env(tmp_path, {"CORESMITH_LLM_PROVIDER": "claude"})    # no persisted model selector: not explicit
    assert _codes(st.module_ready(db, tmp_path, "tiny"))["WORKER_UNBOUND"] == ["tiny"]
    monkeypatch.setattr(server, "_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr(server, "_build", make_build_lifecycle(tmp_path))
    monkeypatch.setattr(server, "_pipeline", types.SimpleNamespace(task=None, thread_id="pipeline"))
    monkeypatch.setattr(server, "_preflight_or_400", lambda: None)
    resp = asyncio.run(server._start_module_build(server.BuildModuleRequest(module="tiny"), entry="build_module"))
    assert resp.status_code == 409 and "WORKER_UNBOUND" in _codes(json.loads(resp.body)["blocked_by"])


def test_run_start_refuses_before_blocks_and_force_does_not_bypass(tmp_path, monkeypatch):
    from orchestrator.daemon import server
    db = ready_project(tmp_path, monkeypatch, stage="uarch")
    monkeypatch.setattr(server, "_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr(server, "_pipeline", types.SimpleNamespace(task=None, thread_id="pipeline"))
    monkeypatch.setattr(server, "_build", types.SimpleNamespace(task=None, thread_id="build"))
    monkeypatch.setattr(server, "_apply_run_env", lambda where: [])
    monkeypatch.setattr(server, "_preflight_or_400", lambda: (_ for _ in ()).throw(AssertionError("preflight ran")))
    resp = asyncio.run(server.run_start(server.StartRequest(force=True)))
    assert resp.status_code == 409
    assert json.loads(resp.body)["error"] == "STAGE_BEFORE_BLOCKS"
    assert st.current(db) == "uarch"
    # a regressed earlier stage also refuses at blocks
    assert st.advance(db, tmp_path)["advanced"]
    (tmp_path / ".coresmith" / "vip_index.json").unlink()
    resp = asyncio.run(server.run_start(server.StartRequest(force=True)))
    assert json.loads(resp.body)["error"] == "STAGE_REGRESSED"
