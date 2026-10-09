# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""``coresmith model build|eval`` are checks of the files that exist and never
construct an authoring agent; ``model author --block`` and ``harness author``
are the explicit authoring verbs and touch only what they were asked for.
Exit contract: 0 pass, 1 build/run/check failure, 2 missing/invalid input
(``required`` names the path), 3 provider/toolchain failure."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path

import pytest

from orchestrator.harness import cli
from orchestrator.harness.tools import integrate as it
from orchestrator.state_store.project_db import open_project
from orchestrator.systemc_model import frd_eval as fe
from orchestrator.tests.test_frd_eval import _FRD, _TINY_HARNESS
from orchestrator.tests.test_systemc_model import (
    _BLOCKS,
    _EDGES,
    _PADS,
    _REQ,
    _RSP,
    _SINK,
    _extra_members,
)

_BODIES = {"req": _REQ, "rsp": _RSP, "sink": _SINK, "pads": _PADS}


class _Forbidden:
    """Constructed = a check authored something."""
    def __init__(self, *a, **k):
        raise AssertionError("an authoring agent was constructed by a check")


@pytest.fixture(autouse=True)
def _no_hidden_authoring(monkeypatch):
    import orchestrator.langchain.agents.frd_eval_generator as fgen
    import orchestrator.langchain.agents.systemc_model_generator as gen
    monkeypatch.setattr(gen, "SystemCModelGenerator", _Forbidden)
    monkeypatch.setattr(fgen, "FRDEvalGenerator", _Forbidden)
    import orchestrator.systemc_model as scm
    monkeypatch.setattr(scm, "detect", lambda: {"ok": True, "reason": "", "cxx": "g++", "systemc_home": ""})


def _project(tmp_path, blocks=_BLOCKS):
    (tmp_path / "inputs").mkdir(exist_ok=True)
    (tmp_path / "inputs" / "task.yaml").write_text("top: tiny\n")
    (tmp_path / "arch").mkdir(exist_ok=True)
    (tmp_path / "arch" / "frd_spec.md").write_text(_FRD)
    db = open_project(tmp_path)
    db.import_block_diagram({"blocks": [{"name": b, "tier": 1} for b in blocks], "connections": []})
    db.import_contracts({"contracts": _EDGES})
    return db


def _sha_tree(root: Path) -> dict:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def _inproc(root, argv):
    ap = argparse.ArgumentParser(prog="coresmith")
    sub = ap.add_subparsers(dest="cmd")
    cli.register_subcommands(sub)
    args = ap.parse_args([*argv, "--project-root", str(root), "--json"])
    args._argv = argv
    buf, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err), pytest.raises(SystemExit) as exc:
        args.func(args)
    out = buf.getvalue()
    return exc.value.code, (json.loads(out) if out.strip().startswith("{") else out)


# ---------------------------------------------------------------------------
# model build: assemble + compile what exists; list the missing, author nothing
# ---------------------------------------------------------------------------
def test_model_build_lists_missing_models_and_authors_nothing(tmp_path, monkeypatch):
    db = _project(tmp_path)
    res = it.model_build(db, tmp_path)
    assert res["ok"] is False and res["build_ok"] is False
    assert res["missing_models"] == sorted(_BLOCKS)
    assert res["required"] == [f"model/{b}_model.cpp" for b in sorted(_BLOCKS)] and "model author" in res["hint"]
    md = tmp_path / "model"
    assert (md / "soc_model_top.h").exists() and all((md / f"{b}_model.h").exists() for b in _BLOCKS)
    assert not list(md.glob("*.prev")) and not any((md / f"{b}_model.cpp").exists() for b in _BLOCKS)
    rc, out = _inproc(tmp_path, ["model", "build"])
    assert rc == 2 and out["missing_models"] == sorted(_BLOCKS)


def test_model_build_never_renames_an_existing_implementation(tmp_path, monkeypatch):
    db = _project(tmp_path)
    it.model_build(db, tmp_path)
    md = tmp_path / "model"
    (md / "req_model.cpp").write_text("// mine\n")
    before = _sha_tree(md)
    res = it.model_build(db, tmp_path)
    assert res["missing_models"] == [b for b in sorted(_BLOCKS) if b != "req"]
    assert res["blocks"]["req"] == "existing"
    assert not list(md.glob("*.prev")) and (md / "req_model.cpp").read_text() == "// mine\n"
    after = _sha_tree(md)
    assert after["req_model.cpp"] == before["req_model.cpp"]


def test_model_build_refuses_an_alternate_registered_path_instead_of_replacing_it(tmp_path):
    """The public binding is model/<block>_model.cpp: a models row that names
    another path is refused (nothing compiled, the row not overwritten),
    never silently replaced by the conventional one."""
    db = _project(tmp_path)
    it.model_build(db, tmp_path)
    md = tmp_path / "model"
    (md / "impl").mkdir()
    (md / "impl" / "req_custom.cpp").write_text("// elsewhere\n")
    db.upsert_model("req", path="model/impl/req_custom.cpp", sha="x", build_ok=True, smoke_ok=True)
    res = it.model_build(db, tmp_path)
    assert res["ok"] is False and res["error"] == "MODEL_PATH_UNSUPPORTED"
    assert res["unsupported_model_paths"] == {"req": str(md / "impl" / "req_custom.cpp")}
    assert res["required"] == ["model/req_model.cpp"] and "not a supported binding" in res["hint"]
    assert db.model_for("req")["path"] == "model/impl/req_custom.cpp"          # the row is not rewritten
    rc, out = _inproc(tmp_path, ["model", "build"])
    assert rc == 2 and out["error"] == "MODEL_PATH_UNSUPPORTED"
    # the supported migration: the implementation moves to the conventional
    # path and the other file goes away; the next build re-binds the row to
    # the conventional file before compiling (even when other models are
    # still missing), with the old file's verdicts cleared
    (md / "impl" / "req_custom.cpp").rename(md / "req_model.cpp")
    res = it.model_build(db, tmp_path)
    assert res.get("error") != "MODEL_PATH_UNSUPPORTED" and res["blocks"]["req"] == "existing"
    row = db.model_for("req")
    assert row["path"] == str(md / "req_model.cpp") and row["build_ok"] is None and row["smoke_ok"] is None
    assert row["sha"] == hashlib.sha256(b"// elsewhere\n").hexdigest()[:16]
    assert it.model_build(db, tmp_path).get("error") != "MODEL_PATH_UNSUPPORTED"   # and stays accepted


def test_model_refine_is_a_migration_error(tmp_path):
    _project(tmp_path)
    rc, out = _inproc(tmp_path, ["model", "refine", "--block", "req"])
    assert rc == 2 and out["error"] == "MODEL_REFINE_REMOVED" and "model author" in out["hint"]


# ---------------------------------------------------------------------------
# model eval: a pure check of the supplied harness
# ---------------------------------------------------------------------------
def test_model_eval_without_an_assembled_model_names_the_required_file(tmp_path):
    db = _project(tmp_path)
    res = it.model_eval(db, tmp_path)
    assert res["ok"] is False and res["error"] == "SOC_MODEL_NOT_ASSEMBLED"
    assert res["required"].endswith("model/soc_model_top.h") and "model build" in res["hint"]
    rc, out = _inproc(tmp_path, ["model", "eval"])
    assert rc == 2 and out["error"] == "SOC_MODEL_NOT_ASSEMBLED"


def test_model_eval_with_no_harness_makes_no_call_and_names_the_path(tmp_path):
    db = _project(tmp_path)
    it.model_build(db, tmp_path)
    before = _sha_tree(tmp_path / "model")
    res = it.model_eval(db, tmp_path)
    assert res["ok"] is False and res["gate_ok"] is None and res["error"] == "HARNESS_MISSING"
    assert res["required"].endswith("model/frd_eval/frd_eval.cpp") and "harness author" in res["hint"]
    assert res["authoring"] is False and res["harness_sources"] == []
    assert _sha_tree(tmp_path / "model") == {**before, "frd_eval/requirements.json": _sha_tree(tmp_path / "model")["frd_eval/requirements.json"]}
    rc, out = _inproc(tmp_path, ["model", "eval"])
    assert rc == 2 and out["error"] == "HARNESS_MISSING"
    assert db.checks(kind="model_eval") == []


def test_model_eval_build_failure_returns_diagnostics_and_repairs_nothing(tmp_path, monkeypatch):
    db = _project(tmp_path)
    it.model_build(db, tmp_path)
    md = tmp_path / "model"
    (md / "frd_eval").mkdir(exist_ok=True)
    (md / "frd_eval" / "frd_eval.cpp").write_text("int broken(\n")
    monkeypatch.setattr(fe, "build_harness", lambda md, **k: {"ok": False, "log": "error: expected ')'"})
    before = _sha_tree(md)
    res = it.model_eval(db, tmp_path)
    assert res["ok"] is False and res["error"] == "HARNESS_BUILD_FAILED" and res["built"] is False
    assert "expected ')'" in res["build_log"]
    after = _sha_tree(md)
    after.pop("frd_eval/requirements.json")              # the evaluation record is an expected side effect
    assert after == before                               # no edit, no repair of any source
    rc, out = _inproc(tmp_path, ["model", "eval"])
    assert rc == 1 and out["error"] == "HARNESS_BUILD_FAILED"


def test_model_eval_incomplete_run_is_reported_not_repaired(tmp_path, monkeypatch):
    db = _project(tmp_path)
    it.model_build(db, tmp_path)
    md = tmp_path / "model"
    (md / "frd_eval").mkdir(exist_ok=True)
    (md / "frd_eval" / "frd_eval.cpp").write_text("// hangs\n")
    monkeypatch.setattr(fe, "build_harness", lambda md, **k: {"ok": True, "log": ""})
    monkeypatch.setattr(fe, "run_harness", lambda md, **k: {"ok": False, "done": False, "rc": 139,
                                                             "log": "segfault", "results": []})
    res = it.model_eval(db, tmp_path)
    assert res["error"] == "HARNESS_RUN_INCOMPLETE" and res["done"] is False and "segfault" in res["run_log"]
    assert db.checks(kind="model_eval") == []            # an incomplete run records no verdict


def test_model_eval_pass_rows_then_nonzero_exit_is_a_failed_run_not_a_pass(tmp_path, monkeypatch):
    """FRD_EVAL pass rows and FRD_EVAL_DONE printed by a process that then
    exits non-zero are diagnostics: no gate_ok, no model_eval checks, exit 1."""
    db = _project(tmp_path)
    it.model_build(db, tmp_path)
    md = tmp_path / "model"
    (md / "frd_eval").mkdir(exist_ok=True)
    (md / "frd_eval" / "frd_eval.cpp").write_text("// returns 3 after DONE\n")
    rows = ('FRD_EVAL {"id":"PERF-001","status":"pass","value":100,"evidence":"a"}\n'
            'FRD_EVAL {"id":"PHYS-001","status":"not_testable","evidence":"b"}\n'
            'FRD_EVAL {"id":"INV-001","status":"pass","evidence":"c"}\nFRD_EVAL_DONE\n')
    monkeypatch.setattr(fe, "build_harness", lambda md, **k: {"ok": True, "log": ""})
    monkeypatch.setattr(fe, "run_harness", lambda md, **k: {"ok": False, "done": True, "rc": 3, "log": rows,
                                                             "results": fe.parse_results(rows)})
    res = it.model_eval(db, tmp_path)
    assert res["ok"] is False and res["gate_ok"] is False and res["error"] == "HARNESS_RUN_FAILED"
    assert res["done"] is True and res["rc"] == 3 and res["run_ok"] is False and "FRD_EVAL_DONE" in res["run_log"]
    assert res["summary"]["counts"]["pass"] == 1                 # the declared row is reported as a diagnostic ...
    assert db.checks(kind="model_eval") == []                    # ... never recorded as evidence
    rc, out = _inproc(tmp_path, ["model", "eval"])
    assert rc == 1 and out["error"] == "HARNESS_RUN_FAILED" and out["gate_ok"] is False
    # the same run through the explicit authoring path is a harness defect to repair, never a pass
    import orchestrator.langchain.agents.frd_eval_generator as fgen
    monkeypatch.setattr(fgen, "FRDEvalGenerator", _FakeHarnessAuthor)
    _FakeHarnessAuthor.calls, _FakeHarnessAuthor.error = [], ""
    monkeypatch.setenv("CORESMITH_FRD_EVAL_REPAIRS", "1")
    rec = __import__("asyncio").run(fe.evaluate(tmp_path, md, list(_BLOCKS), author=True, db=db,
                                                agent=_FakeHarnessAuthor()))
    assert rec["gate_ok"] is False and rec["error"] == "HARNESS_RUN_FAILED" and db.checks(kind="model_eval") == []
    assert [c[1] for c in _FakeHarnessAuthor.calls] == [2]      # one repair attempt with the run log, then stop


def test_model_eval_without_an_frd_is_an_input_error(tmp_path):
    db = _project(tmp_path)
    it.model_build(db, tmp_path)
    (tmp_path / "arch" / "frd_spec.md").unlink()
    res = it.model_eval(db, tmp_path)
    assert res["ok"] is False and res["gate_ok"] is None and res["error"] == "FRD_MISSING"
    assert res["required"].endswith("arch/frd_spec.md")
    rc, out = _inproc(tmp_path, ["model", "eval"])
    assert rc == 2 and out["error"] == "FRD_MISSING" and out["required"].endswith("arch/frd_spec.md")
    (tmp_path / "arch" / "frd_spec.md").write_text("# FRD\n\nNo identified requirements here.\n")
    res = it.model_eval(db, tmp_path)
    assert res["error"] == "FRD_NO_REQUIREMENTS" and res["required"].endswith("arch/frd_spec.md")
    rc, out = _inproc(tmp_path, ["model", "eval"])
    assert rc == 2 and out["error"] == "FRD_NO_REQUIREMENTS"
    rc, out = _inproc(tmp_path, ["harness", "author"])
    assert rc == 2 and out["error"] == "FRD_NO_REQUIREMENTS"        # no authoring call for an unusable FRD


def test_model_eval_with_a_supplied_harness_records_verdicts(tmp_path, monkeypatch):
    db = _project(tmp_path)
    it.model_build(db, tmp_path)
    md = tmp_path / "model"
    (md / "frd_eval").mkdir(exist_ok=True)
    (md / "frd_eval" / "frd_eval.cpp").write_text("// supplied\n")
    monkeypatch.setattr(fe, "build_harness", lambda md, **k: {"ok": True, "log": ""})
    monkeypatch.setattr(fe, "run_harness", lambda md, **k: {"ok": True, "done": True, "rc": 0, "log": "", "results": fe.parse_results(
        'FRD_EVAL {"id":"PERF-001","status":"pass","value":100,"evidence":"a"}\nFRD_EVAL {"id":"PHYS-001","status":"not_testable","evidence":"b"}\n'
        'FRD_EVAL {"id":"INV-001","status":"fail","evidence":"c"}')})
    res = it.model_eval(db, tmp_path)
    # the evaluation is SCOPED to the requirements that declare a model check
    # (PERF-001): INV-001's failure and PHYS-001's verdict are out of scope,
    # kept as diagnostics, never demanded of the model nor recorded
    assert res["model_check_scope"] == ["PERF-001"] and res["scope"]["excluded"] == ["PERF-002", "PHYS-001", "INV-001"]
    assert res["gate_ok"] is True and res["summary"]["failed"] == [] and res["authoring"] is False
    assert {r["id"]: r["status"] for r in res["out_of_scope_results"]} == {"PHYS-001": "not_testable", "INV-001": "fail"}
    assert res["harness_sources"] == [str(md / "frd_eval" / "frd_eval.cpp")]
    assert {c["item_id"]: c["status"] for c in db.checks(kind="model_eval", latest=True)} == {"PERF-001": "pass"}
    from orchestrator.state_store import builds as B
    assert db.latest_check("PERF-001", "model_eval")["sha"] == B.model_check_sha(db, tmp_path, "PERF-001")
    # the harness fails the declared one: the gate fails, exit 1
    monkeypatch.setattr(fe, "run_harness", lambda md, **k: {"ok": True, "done": True, "rc": 0, "log": "", "results": fe.parse_results(
        'FRD_EVAL {"id":"PERF-001","status":"fail","value":5000,"evidence":"slow"}\nFRD_EVAL {"id":"INV-001","status":"pass","evidence":"c"}')})
    rc, out = _inproc(tmp_path, ["model", "eval"])
    assert rc == 1 and out["gate_ok"] is False and out["summary"]["failed"] == ["PERF-001"]


def test_model_eval_never_demands_a_requirement_that_declares_no_model_check(tmp_path, monkeypatch):
    """An FRD with a model-checkable item and a Linux-boot must-have that
    declares no model check: the evaluator asks the harness for the first
    only. No proof of the second is demanded and no pass or waiver is
    manufactured for it; it keeps its chip-level acceptance."""
    from orchestrator.harness.tools.register import register
    frd = """# FRD

## Performance Requirements

1. **ID**: PERF-001
   - **Requirement**: Mean cycles per frame at most 2,133,333 [HARD].
   - **Acceptance criteria**: cycles_per_frame_mean <= 2133333.
   - **Priority**: must_have
   - **Model check**: LT cycle accounting.

## Software Requirements

1. **ID**: SW-001
   - **Requirement**: The SoC boots Linux to a shell prompt.
   - **Acceptance criteria**: a boot log with a shell prompt on the chip-level validation run.
   - **Priority**: must_have
"""
    db = _project(tmp_path)
    (tmp_path / "arch" / "frd_spec.md").write_text(frd)
    assert register(db, tmp_path, "frd", "arch/frd_spec.md")["ok"]
    it.model_build(db, tmp_path)
    md = tmp_path / "model"
    (md / "frd_eval").mkdir(exist_ok=True)
    (md / "frd_eval" / "frd_eval.cpp").write_text("// supplied\n")
    monkeypatch.setattr(fe, "build_harness", lambda md, **k: {"ok": True, "log": ""})
    seen = {}

    def run(md, **k):
        seen["requirements"] = json.loads((Path(md) / "frd_eval" / "requirements.json").read_text())
        return {"ok": True, "done": True, "rc": 0, "log": "", "results": fe.parse_results(
            'FRD_EVAL {"id":"PERF-001","status":"pass","evidence":"a"}')}
    monkeypatch.setattr(fe, "run_harness", run)
    res = it.model_eval(db, tmp_path)
    assert res["gate_ok"] is True and res["model_check_scope"] == ["PERF-001"]
    assert [q["id"] for q in seen["requirements"]["requirements"]] == ["PERF-001"]    # the harness's input list
    assert res["summary"]["unanswered_must"] == [] and res["scope"]["excluded"] == ["SW-001"]
    assert [c["item_id"] for c in db.checks(kind="model_eval")] == ["PERF-001"]
    assert db.item("SW-001")["status"] == "open"                                       # no fake pass, no waiver
    # with nothing declared, nothing is required of the model
    db.edit_item("PERF-001", model_check="")
    res = it.model_eval(db, tmp_path)
    assert res["gate_ok"] is True and res["requirements"] == 0 and "declares a model check" in res["skipped"]


# ---------------------------------------------------------------------------
# model author / harness author: explicit, selective, truthful about failures
# ---------------------------------------------------------------------------
class _FakeAuthor:
    calls: list = []
    error = ""
    write = True

    def __init__(self, *a, **k):
        pass

    async def generate(self, block_name, *, project_root, header_path, compiler_log="", attempt=1):
        _FakeAuthor.calls.append((block_name, attempt, compiler_log))
        p = Path(project_root) / "model" / f"{block_name}_model.cpp"
        if _FakeAuthor.write:
            p.write_text(f"// authored {block_name}\n")
        return {"files_written": [str(p)] if _FakeAuthor.write else [], "notes": "n",
                "written": p.exists(), "response_error": _FakeAuthor.error}


class _FakeHarnessAuthor:
    calls: list = []
    error = ""

    def __init__(self, *a, **k):
        pass

    async def generate(self, *, project_root, blocks, attempt=1, compiler_log="", run_log="", summary=None, arch=False):
        _FakeHarnessAuthor.calls.append((tuple(blocks), attempt, compiler_log, arch))
        d = Path(project_root) / "model" / ("arch/frd_eval" if arch else "frd_eval")
        d.mkdir(parents=True, exist_ok=True)
        (d / "frd_eval.cpp").write_text("// authored harness\n")
        return {"files_written": ["frd_eval.cpp"], "notes": "", "written": True, "sources": [str(d / "frd_eval.cpp")],
                "response_error": _FakeHarnessAuthor.error}


def test_model_author_authors_only_the_named_blocks(tmp_path, monkeypatch):
    import orchestrator.langchain.agents.systemc_model_generator as gen
    monkeypatch.setattr(gen, "SystemCModelGenerator", _FakeAuthor)
    _FakeAuthor.calls, _FakeAuthor.error, _FakeAuthor.write = [], "", True
    db = _project(tmp_path)
    res = it.model_author(db, tmp_path, ["req"])
    assert res["ok"] and res["written"] == ["req"] and res["not_written"] == [] and res["provider_errors"] == {}
    assert [c[0] for c in _FakeAuthor.calls] == ["req"]
    assert (tmp_path / "model" / "req_model.cpp").exists() and not (tmp_path / "model" / "rsp_model.cpp").exists()
    build = it.model_build(db, tmp_path)
    assert build["missing_models"] == [b for b in sorted(_BLOCKS) if b != "req"]
    events = (tmp_path / ".coresmith" / "pipeline_events.jsonl").read_text()
    assert '"model_authoring"' in events and '"blocks": ["req"]' in events
    rc, out = _inproc(tmp_path, ["model", "author", "--block", "rsp", "--block", "sink"])
    assert rc == 0 and out["written"] == ["rsp", "sink"]
    assert [c[0] for c in _FakeAuthor.calls] == ["req", "rsp", "sink"]


def test_model_author_diagnostics_make_it_a_repair_request(tmp_path, monkeypatch):
    import orchestrator.langchain.agents.systemc_model_generator as gen
    monkeypatch.setattr(gen, "SystemCModelGenerator", _FakeAuthor)
    _FakeAuthor.calls, _FakeAuthor.error, _FakeAuthor.write = [], "", True
    _project(tmp_path)
    log = tmp_path / "build.log"
    log.write_text("error: 'foo' was not declared\n")
    rc, out = _inproc(tmp_path, ["model", "author", "--block", "req", "--diagnostics", str(log)])
    assert rc == 0 and _FakeAuthor.calls == [("req", 2, "error: 'foo' was not declared\n")]
    rc, out = _inproc(tmp_path, ["model", "author", "--block", "req", "--diagnostics", str(tmp_path / "missing.log")])
    assert rc == 2 and out["error"] == "DIAGNOSTICS_UNREADABLE"


def test_model_author_provider_failure_is_exit_3_even_with_a_file_on_disk(tmp_path, monkeypatch):
    import orchestrator.langchain.agents.systemc_model_generator as gen
    monkeypatch.setattr(gen, "SystemCModelGenerator", _FakeAuthor)
    _FakeAuthor.calls, _FakeAuthor.write = [], True
    _FakeAuthor.error = "[ClaudeLLM error: codex CLI returned empty response. exit_code=1, stderr: Read-only file system]"
    db = _project(tmp_path)
    (tmp_path / "model").mkdir(exist_ok=True)
    (tmp_path / "model" / "req_model.cpp").write_text("// stale from an earlier attempt\n")
    res = it.model_author(db, tmp_path, ["req"])
    assert res["ok"] is False and res["written"] == [] and res["not_written"] == ["req"]
    assert "Read-only file system" in res["provider_errors"]["req"]
    rc, out = _inproc(tmp_path, ["model", "author", "--block", "req"])
    assert rc == 3 and "Read-only file system" in out["blocks"]["req"]["provider_error"]


def test_model_author_refuses_unknown_and_primitive_blocks_before_any_call(tmp_path, monkeypatch):
    db = _project(tmp_path)
    res = it.model_author(db, tmp_path, ["ghost"])
    assert res["ok"] is False and res["error"] == "UNKNOWN_BLOCK" and res["unknown"] == ["ghost"]
    rc, out = _inproc(tmp_path, ["model", "author", "--block", "ghost"])
    assert rc == 2
    rc, out = _inproc(tmp_path, ["model", "author"])
    assert rc == 2 and out["error"] == "NO_BLOCKS"
    db2 = open_project(tmp_path / "p2")
    db2.import_block_diagram({"blocks": [{"name": "fab", "tier": 0, "kind": "primitive", "primitive": "cs_fabric"}],
                              "connections": []})
    res = it.model_author(db2, tmp_path / "p2", ["fab"])
    assert res["ok"] is False and res["error"] == "PRIMITIVE_BLOCK" and res["primitive"] == ["fab"]


def test_harness_author_is_explicit_and_truthful(tmp_path, monkeypatch):
    import orchestrator.langchain.agents.frd_eval_generator as fgen
    monkeypatch.setattr(fgen, "FRDEvalGenerator", _FakeHarnessAuthor)
    _FakeHarnessAuthor.calls, _FakeHarnessAuthor.error = [], ""
    db = _project(tmp_path)
    res = it.harness_author(db, tmp_path)
    assert res["ok"] is False and res["error"] == "SOC_MODEL_NOT_ASSEMBLED" and _FakeHarnessAuthor.calls == []
    it.model_build(db, tmp_path)
    rc, out = _inproc(tmp_path, ["harness", "author"])
    assert rc == 0 and out["written"] and _FakeHarnessAuthor.calls == [(tuple(_BLOCKS), 1, "", False)]
    assert (tmp_path / "model" / "frd_eval" / "requirements.json").exists()
    _FakeHarnessAuthor.error = "[ClaudeLLM error: codex CLI timed out after 10s]"
    rc, out = _inproc(tmp_path, ["harness", "author"])
    assert rc == 3 and "timed out" in out["provider_error"] and out["ok"] is False
    rc, out = _inproc(tmp_path, ["harness", "author", "--arch"])
    assert rc == 2 and out["error"] == "ARCH_MODEL_NOT_BUILT"


def test_model_author_has_no_arch_form(tmp_path):
    """``model author --arch`` is rejected (exit 2) instead of being accepted
    and silently ignored: the architecture model is generated, never authored."""
    _project(tmp_path)
    ap = argparse.ArgumentParser(prog="coresmith")
    cli.register_subcommands(ap.add_subparsers(dest="cmd"))
    err = io.StringIO()
    with contextlib.redirect_stderr(err), pytest.raises(SystemExit) as exc:
        ap.parse_args(["model", "author", "--block", "req", "--arch", "--project-root", str(tmp_path), "--json"])
    assert exc.value.code == 2 and "--arch" in err.getvalue()
    ap.parse_args(["model", "eval", "--arch", "--project-root", str(tmp_path), "--json"])   # the checks keep --arch


# ---------------------------------------------------------------------------
# model --arch: the architecture model's exit contract
# ---------------------------------------------------------------------------
def test_arch_build_run_register_name_missing_or_invalid_input(tmp_path, monkeypatch):
    from orchestrator.harness.tools import model as mt
    _project(tmp_path)
    rc, out = _inproc(tmp_path, ["model", "build", "--arch"])
    assert rc == 2 and out["error"] == "ARCH_SPEC_MISSING" and out["required"].endswith("model/arch/arch_model.json")
    rc, out = _inproc(tmp_path, ["model", "run", "--arch"])
    assert rc == 2 and out["error"] == "ARCH_MODEL_NOT_BUILT" and out["required"].endswith("model/arch/arch_model")
    rc, out = _inproc(tmp_path, ["model", "register", "--arch"])
    assert rc == 2 and out["error"] == "ARCH_SPEC_MISSING" and out["required"].endswith("arch_model.json")
    rc, out = _inproc(tmp_path, ["model", "init", "--arch"])
    assert rc == 0 and out["created"] is True
    spec = Path(out["path"])
    spec.write_text("{not json")
    rc, out = _inproc(tmp_path, ["model", "build", "--arch"])
    assert rc == 2 and out["error"] == "ARCH_SPEC_INVALID" and out["required"] == str(spec) and out["problems"]
    spec.write_text(json.dumps({"name": "soc", "components": [], "links": [{"from": "ghost", "to": "ram"}]}))
    rc, out = _inproc(tmp_path, ["model", "build", "--arch"])
    assert rc == 2 and out["error"] == "ARCH_SPEC_INVALID" and out["problems"]
    # ordinary wrong shapes are input errors with the required path, never tracebacks
    for doc, needle in (([], "JSON object"), ({"components": [3]}, "components[0] must be an object"),
                        ({"fabric": 3}, "fabric must be an object"), ({"links": {"a": 1}}, "links must be a list"),
                        ({"components": [{"name": "x", "instances": "many"}]}, "'many'"),
                        ({"components": [{"name": "x", "energy_pj_per_txn": "lots"}]}, "ValueError")):
        spec.write_text(json.dumps(doc))
        rc, out = _inproc(tmp_path, ["model", "build", "--arch"])
        assert rc == 2 and out["error"] == "ARCH_SPEC_INVALID" and out["required"] == str(spec), doc
        assert any(needle in p for p in out["problems"]), (doc, out["problems"])
        rc, out = _inproc(tmp_path, ["model", "register", "--arch"])     # register needs only the file
        assert rc == 0
    # a valid spec: a missing toolchain is the only exit 3; compiler diagnostics are exit 1
    mt.arch_init(tmp_path) if not spec.exists() else None
    spec.write_text(json.dumps(mt.SPEC_TEMPLATE))
    monkeypatch.setattr("orchestrator.systemc_model.toolchain.detect", lambda: {"ok": False, "reason": "no g++"})
    rc, out = _inproc(tmp_path, ["model", "build", "--arch"])
    assert rc == 3 and out["tool_error"] is True and "no g++" in out["error"]
    monkeypatch.setattr("orchestrator.systemc_model.toolchain.detect",
                        lambda: {"ok": True, "reason": "", "cxx": "g++", "systemc_home": ""})
    from orchestrator.systemc_model import arch_model as am
    monkeypatch.setattr(am, "build", lambda md, **k: {"ok": False, "log": "error: 'sc_foo' was not declared"})
    rc, out = _inproc(tmp_path, ["model", "build", "--arch"])
    assert rc == 1 and out["error"] == "ARCH_MODEL_BUILD_FAILED" and out["tool_error"] is False
    assert "sc_foo" in out["log"] and (tmp_path / "model" / "arch" / "arch_model_top.h").exists()
    monkeypatch.setattr(am, "build", lambda md, **k: {"ok": True, "log": ""})
    (tmp_path / "model" / "arch" / "arch_model").write_text("#!/bin/sh\nexit 1\n")
    (tmp_path / "model" / "arch" / "arch_model").chmod(0o755)
    rc, out = _inproc(tmp_path, ["model", "run", "--arch"])
    assert rc == 1 and out["error"] == "ARCH_MODEL_RUN_FAILED"
    rc, out = _inproc(tmp_path, ["model", "register", "--arch"])
    assert rc == 0 and out["artifact"]["version"] >= 1


# ---------------------------------------------------------------------------
# Real tools: the fixture SoC built, smoked and evaluated through the checks
# ---------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.skipif(not __import__("orchestrator.systemc_model", fromlist=["detect"]).detect()["ok"],
                    reason="SystemC toolchain not available")
def test_real_toolchain_build_and_eval_without_any_agent(tmp_path, monkeypatch):
    import orchestrator.systemc_model as scm
    monkeypatch.setattr(scm, "detect", scm.toolchain.detect if hasattr(scm, "toolchain") else scm.detect)
    from orchestrator.systemc_model.toolchain import detect as real_detect
    monkeypatch.setattr(scm, "detect", real_detect)
    db = _project(tmp_path)
    first = it.model_build(db, tmp_path)
    assert first["missing_models"] == sorted(_BLOCKS)
    md = tmp_path / "model"
    for b in _BLOCKS:                                   # the Architect writes the implementations itself
        h = md / f"{b}_model.h"
        h.write_text(h.read_text().replace("  void run();", _extra_members(b) + "  void run();"))
        (md / f"{b}_model.cpp").write_text(_BODIES[b])
    res = it.model_build(db, tmp_path)
    assert res["ok"] and res["build_ok"] and res["smoke_ok"], res.get("build_log", "")[-2000:]
    assert all(m["build_ok"] == 1 and m["smoke_ok"] == 1 for m in db.models())
    ev = it.model_eval(db, tmp_path)
    assert ev["error"] == "HARNESS_MISSING"
    (md / "frd_eval" / "frd_eval.cpp").write_text(_TINY_HARNESS)
    ev = it.model_eval(db, tmp_path)
    # scoped to the one requirement that declares a model check (PERF-001)
    assert ev["gate_ok"] is True and ev["summary"]["counts"]["pass"] == 1 and ev["model_check_scope"] == ["PERF-001"], ev
    rc, out = _inproc(tmp_path, ["model", "eval"])
    assert rc == 0 and out["gate_ok"] is True
    passing = len(db.checks(kind="model_eval"))
    assert passing == 2                                  # two passing evaluations (tool + CLI), one declared verdict each
    # the same harness returning non-zero after FRD_EVAL_DONE: a failed run (exit 1), no new checks
    (md / "frd_eval" / "frd_eval.cpp").write_text(_TINY_HARNESS.replace("    return 0;\n}", "    return 3;\n}"))
    ev = it.model_eval(db, tmp_path)
    assert ev["gate_ok"] is False and ev["error"] == "HARNESS_RUN_FAILED" and ev["rc"] == 3 and ev["done"] is True, ev
    assert ev["summary"]["counts"]["pass"] == 1 and len(db.checks(kind="model_eval")) == passing
    rc, out = _inproc(tmp_path, ["model", "eval"])
    assert rc == 1 and out["error"] == "HARNESS_RUN_FAILED"
    # a harness that does not compile: diagnostics, exit 1, nothing repaired
    (md / "frd_eval" / "frd_eval.cpp").write_text("#include \"soc_model_top.h\"\nint sc_main(int, char**) { return undefined_symbol; }\n")
    rc, out = _inproc(tmp_path, ["model", "eval"])
    assert rc == 1 and out["error"] == "HARNESS_BUILD_FAILED" and "undefined_symbol" in out["build_log"]
