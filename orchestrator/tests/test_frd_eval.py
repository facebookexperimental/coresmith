"""B2: the FRD evaluated on the SystemC SoC model before RTL."""
import asyncio
import json
from pathlib import Path

import pytest

from orchestrator.langgraph import pipeline_graph as pg
from orchestrator.systemc_model import frd_eval as fe
from orchestrator.systemc_model import write_build

_FRD = """# FRD — tiny

## Performance Requirements

1. **ID**: PERF-001
   - **Requirement**: Mean cycles per frame at most 2,133,333 [HARD].
     - Unit: clk cycles.
   - **Acceptance criteria**: throughput.py reports cycles_per_frame_mean <= 2133333.
   - **Priority**: must_have
   - **Model check**: LT cycle accounting of the present counter.

2. **ID**: PERF-002
   - **Requirement**: Worst-case frame interval.
   - **Acceptance criteria**: max <= 2133333.
   - **Priority**: should_have

## Physical Design Requirements

1. **ID**: `PHYS-001`
   - **Requirement**: Die fits 3x3 mm.
   - **Acceptance criteria**: DRC clean.
   - **Priority**: must_have

## Semantic Invariants

- **ID**: INV-001
  - **Requirement**: Present order equals submit order.
  - **Acceptance criteria**: ordering check.
  - **Priority**: must_have
"""


def test_extract_requirements_ids_fields_and_sections():
    reqs = fe.extract_requirements(_FRD)
    ids = [r["id"] for r in reqs]
    assert ids == ["PERF-001", "PERF-002", "PHYS-001", "INV-001"]
    r = reqs[0]
    assert r["section"] == "Performance Requirements" and r["priority"] == "must_have"
    assert "2,133,333" in r["requirement"] and "cycles_per_frame_mean" in r["acceptance"]
    assert r["model_check"].startswith("LT cycle")
    assert fe.must_have(reqs[0]) and not fe.must_have(reqs[1]) and fe.must_have(reqs[3])


def test_parse_results_and_summary_gate():
    out = """[frd_eval] booting
FRD_EVAL {"id": "perf-001", "status": "pass", "evidence": "mean=1900000 (LT estimate)"}
FRD_EVAL {"id": "PERF-002", "status": "fail", "evidence": "max=2500000"}
FRD_EVAL {"id": "PHYS-001", "status": "not_testable", "evidence": "physical design; not observable on an LT model"}
FRD_EVAL {"id": "BOGUS-9", "status": "weird"}
not a verdict line FRD_EVAL {}
FRD_EVAL_DONE
"""
    res = fe.parse_results(out)
    assert [r["id"] for r in res] == ["PERF-001", "PERF-002", "PHYS-001", "BOGUS-9"]
    assert res[3]["status"] == "fail"   # unknown status is a failure, never a pass
    reqs = fe.extract_requirements(_FRD)
    s = fe.summarize(reqs, res)
    assert s["failed"] == ["BOGUS-9", "PERF-002"] and s["unanswered_must"] == ["INV-001"]
    assert s["unknown_ids"] == ["BOGUS-9"] and s["gate_ok"] is False
    good = fe.parse_results("""FRD_EVAL {"id":"PERF-001","status":"pass","evidence":"ok"}
FRD_EVAL {"id":"PHYS-001","status":"not_testable","evidence":"DRC"}
FRD_EVAL {"id":"INV-001","status":"pass","evidence":"ordered"}
FRD_EVAL_DONE""")
    assert fe.summarize(reqs, good)["gate_ok"] is True     # PERF-002 is should_have: unanswered is allowed
    noreason = fe.parse_results('FRD_EVAL {"id":"PERF-001","status":"not_testable","evidence":""}\n'
                                'FRD_EVAL {"id":"PHYS-001","status":"pass"}\nFRD_EVAL {"id":"INV-001","status":"pass"}')
    assert fe.summarize(reqs, noreason)["not_testable_without_reason"] == ["PERF-001"]
    assert fe.summarize(reqs, noreason)["gate_ok"] is False


def test_write_build_emits_shared_top_header_and_harness_target(tmp_path):
    edges = [{"edge_id": "a__m_q__to__b__s_q", "producer_block": "a", "producer_port": "m_q",
              "consumer_block": "b", "consumer_port": "s_q", "handshake_protocol": "req_resp", "data_width_bits": 8}]
    md = write_build(tmp_path, ["a", "b"], edges, top_name="tiny")
    assert (md / "soc_model_top.h").exists() and (md / "frd_eval").is_dir()
    mk = (md / "Makefile").read_text()
    assert "frd_eval/frd_eval:" in mk and "$(wildcard frd_eval/*.cpp)" in mk
    assert "MODEL_SRCS := a_model.cpp b_model.cpp" in mk
    assert 'cs_mem' in (md / "cs_model_common.h").read_text() and "load_bin" in (md / "cs_model_common.h").read_text()
    out = fe.write_requirements(md, fe.extract_requirements(_FRD))
    assert json.loads(out.read_text())["count"] == 4
    assert fe.build_harness(md)["ok"] is False   # no sources yet: a clear message, no make call


class _FakeHarnessAgent:
    calls: list = []
    output = ""

    def __init__(self, *a, **k):
        pass

    async def generate(self, *, project_root, blocks, attempt=1, compiler_log="", run_log="", summary=None, arch=False):
        _FakeHarnessAgent.calls.append((attempt, bool(compiler_log), bool(run_log)))
        d = Path(project_root) / "model" / "frd_eval"
        d.mkdir(parents=True, exist_ok=True)
        (d / "frd_eval.cpp").write_text("// fake harness\n")
        return {"written": True}


def _fake_project(tmp_path, frd=_FRD):
    (tmp_path / "arch").mkdir(parents=True, exist_ok=True)
    (tmp_path / "arch" / "frd_spec.md").write_text(frd)
    md = tmp_path / "model"
    md.mkdir(exist_ok=True)
    return md


def test_frd_evaluation_gates_on_verdicts(tmp_path, monkeypatch):
    import orchestrator.langchain.agents.frd_eval_generator as gen
    monkeypatch.setattr(gen, "FRDEvalGenerator", _FakeHarnessAgent)
    monkeypatch.setattr(fe, "build_harness", lambda md, **k: {"ok": True, "log": ""})
    runs = iter([{"ok": True, "done": True, "rc": 0, "log": "", "results": fe.parse_results(
        'FRD_EVAL {"id":"PERF-001","status":"pass","evidence":"x"}\nFRD_EVAL {"id":"PERF-002","status":"fail","evidence":"slow"}\n'
        'FRD_EVAL {"id":"PHYS-001","status":"not_testable","evidence":"DRC"}\nFRD_EVAL {"id":"INV-001","status":"pass","evidence":"y"}')}])
    monkeypatch.setattr(fe, "run_harness", lambda md, **k: next(runs))
    _FakeHarnessAgent.calls = []
    md = _fake_project(tmp_path)
    rec = asyncio.run(pg._frd_evaluation(str(tmp_path), md, ["a", "b"]))
    assert rec["requirements"] == 4 and rec["done"] is True and rec["gate_ok"] is False
    assert rec["summary"]["failed"] == ["PERF-002"]
    assert _FakeHarnessAgent.calls == [(1, False, False)]
    assert (tmp_path / ".coresmith" / "frd_eval.json").exists() and (md / "frd_eval" / "REPORT.md").exists()
    assert "PERF-002" in (md / "frd_eval" / "REPORT.md").read_text()


def test_frd_evaluation_repairs_compile_and_crash_then_passes(tmp_path, monkeypatch):
    import orchestrator.langchain.agents.frd_eval_generator as gen
    monkeypatch.setattr(gen, "FRDEvalGenerator", _FakeHarnessAgent)
    builds = iter([{"ok": False, "log": "error: x"}, {"ok": True, "log": ""}, {"ok": True, "log": ""}])
    monkeypatch.setattr(fe, "build_harness", lambda md, **k: next(builds))
    ok = fe.parse_results('FRD_EVAL {"id":"PERF-001","status":"pass","evidence":"a"}\nFRD_EVAL {"id":"PHYS-001","status":"not_testable","evidence":"DRC"}\n'
                          'FRD_EVAL {"id":"INV-001","status":"pass","evidence":"b"}')
    runs = iter([{"ok": False, "done": False, "rc": 139, "log": "segfault", "results": []},
                 {"ok": True, "done": True, "rc": 0, "log": "", "results": ok}])
    monkeypatch.setattr(fe, "run_harness", lambda md, **k: next(runs))
    _FakeHarnessAgent.calls = []
    md = _fake_project(tmp_path)
    rec = asyncio.run(pg._frd_evaluation(str(tmp_path), md, ["a"]))
    assert rec["gate_ok"] is True and rec["summary"]["counts"]["pass"] == 2
    # attempt 1 author, compile error -> attempt 2 with compiler log, crash -> attempt 3 with run log
    assert _FakeHarnessAgent.calls == [(1, False, False), (2, True, False), (3, False, True)]


def test_frd_evaluation_reuses_an_existing_harness_and_skips_without_frd(tmp_path, monkeypatch):
    import orchestrator.langchain.agents.frd_eval_generator as gen
    monkeypatch.setattr(gen, "FRDEvalGenerator", _FakeHarnessAgent)
    monkeypatch.setattr(fe, "build_harness", lambda md, **k: {"ok": True, "log": ""})
    monkeypatch.setattr(fe, "run_harness", lambda md, **k: {"ok": True, "done": True, "rc": 0, "log": "", "results": fe.parse_results(
        'FRD_EVAL {"id":"PERF-001","status":"pass","evidence":"a"}\nFRD_EVAL {"id":"PHYS-001","status":"pass","evidence":"b"}\nFRD_EVAL {"id":"INV-001","status":"pass","evidence":"c"}')})
    _FakeHarnessAgent.calls = []
    md = _fake_project(tmp_path)
    (md / "frd_eval").mkdir()
    (md / "frd_eval" / "frd_eval.cpp").write_text("// operator's harness\n")
    rec = asyncio.run(pg._frd_evaluation(str(tmp_path), md, ["a"]))
    assert rec["gate_ok"] is True and _FakeHarnessAgent.calls == []
    md2 = tmp_path / "other" / "model"
    md2.mkdir(parents=True)
    rec2 = asyncio.run(pg._frd_evaluation(str(tmp_path / "other"), md2, ["a"]))
    assert rec2["gate_ok"] is None and "no arch/frd_spec.md" in rec2["skipped"]


def test_phase_gate_parks_on_a_failed_frd_evaluation(tmp_path, monkeypatch):
    from orchestrator.tests.test_uarch_phase import _gate_run
    monkeypatch.delenv("CORESMITH_UARCH_PHASE_GATE", raising=False)
    parked = _gate_run(tmp_path, monkeypatch, {"build_ok": True, "smoke_ok": True,
                                               "frd_eval": {"gate_ok": False, "summary": {"failed": ["PERF-001"], "unanswered_must": []}}})
    assert len(parked) == 1 and "PERF-001" in parked[0][0]["outer_agent_guidance"]
    assert _gate_run(tmp_path, monkeypatch, {"build_ok": True, "smoke_ok": True, "frd_eval": {"gate_ok": True}}) == []
    assert _gate_run(tmp_path, monkeypatch, {"build_ok": True, "smoke_ok": True, "frd_eval": {"gate_ok": None, "skipped": "no FRD"}}) == []


def test_frd_eval_flag_off(monkeypatch):
    monkeypatch.setenv("CORESMITH_FRD_EVAL", "0")
    assert pg.frd_eval_enabled() is False
    monkeypatch.delenv("CORESMITH_FRD_EVAL")
    assert pg.frd_eval_enabled() is True


_TINY_HARNESS = r'''
#include "soc_model_top.h"
#include <cstdio>
using namespace sc_core;
int sc_main(int, char**) {
    sc_clock clk("clk", cs_clock_period());
    sc_signal<bool> rst_n("rst_n");
    soc_model_top top("top");
    top.clk(clk); top.rst_n(rst_n);
    rst_n.write(false); sc_start(3 * cs_clock_period()); rst_n.write(true); top.reset_all();
    sc_start(sc_time(2000, SC_NS));
    std::printf("FRD_EVAL {\"id\": \"PERF-001\", \"status\": \"pass\", \"evidence\": \"served=%s\"}\n", "4");
    std::printf("FRD_EVAL {\"id\": \"PHYS-001\", \"status\": \"not_testable\", \"evidence\": \"physical\"}\n");
    std::printf("FRD_EVAL {\"id\": \"INV-001\", \"status\": \"pass\", \"evidence\": \"ordered\"}\n");
    std::puts("FRD_EVAL_DONE");
    return 0;
}
'''


@pytest.mark.slow
@pytest.mark.skipif(not __import__("orchestrator.systemc_model", fromlist=["detect"]).detect()["ok"],
                    reason="SystemC toolchain not available")
def test_tiny_harness_builds_and_runs_against_the_fixture_soc(tmp_path):
    from orchestrator.tests.test_systemc_model import _BLOCKS, _EDGES, _extra_members
    from orchestrator.tests.test_uarch_phase import _BODIES
    from orchestrator.systemc_model import build, render_block_skeleton, smoke
    from orchestrator.systemc_model.conventions import model_name
    md = write_build(tmp_path, _BLOCKS, _EDGES, top_name="tiny")
    for b in _BLOCKS:
        h = render_block_skeleton(b, _EDGES).replace("  void run();", _extra_members(b) + "  void run();")
        (md / f"{model_name(b)}.h").write_text(h)
        (md / f"{model_name(b)}.cpp").write_text(_BODIES[b])
    assert build(md)["ok"] and smoke(md)["ok"]
    (md / "frd_eval" / "frd_eval.cpp").write_text(_TINY_HARNESS)
    b = fe.build_harness(md)
    assert b["ok"], b["log"]
    run = fe.run_harness(md, timeout_s=120)
    assert run["done"] and run["ok"], run["log"]
    s = fe.summarize(fe.extract_requirements(_FRD), run["results"])
    assert s["gate_ok"] is True and s["counts"]["pass"] == 2
