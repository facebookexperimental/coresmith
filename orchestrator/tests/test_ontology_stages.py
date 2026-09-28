# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Architect sitting step 1: ontology, extractors, validators, register, state machine, CLI."""
import json
import subprocess
import sys
from pathlib import Path

from orchestrator.harness.tools import extract, validate
from orchestrator.harness.tools.register import register
from orchestrator.state_store import stages as st
from orchestrator.state_store.ontology import is_item_id, item_must_have
from orchestrator.state_store.project_db import open_project

_PRD = {"prd": {
    "title": "t",
    "functional_requirements": ["FR-TOP-1: The top is soc_top with 51 ports.", "FR-CPU-2 [HARD, R2.1]: Two RV64GC harts.",
                                "no id here"],
    "validation_kpis": [{"id": "KPI-ISA-1", "metric": "riscv-tests", "threshold": "241/241", "test_method": "harness"},
                        {"id": "KPI-FPS-1", "metric": "fps", "threshold": ">=30", "test_method": "throughput.py"}],
    "constraints": ["Sky130 only"],
    "open_items": ["Triangles per frame must be measured from the oracle."],
}}

_FRD = """# FRD

## Performance Requirements

1. **ID**: PERF-001
   - **Requirement**: Mean cycles/frame <= 2,133,333 [HARD, KPI-FPS-1].
   - **Acceptance criteria**: throughput.py mean <= 2133333.
   - **Priority**: must_have
   - **Model check**: arch model cycle accounting over s1_hunt.

2. **ID**: PERF-002
   - **Requirement**: worst frame interval.
   - **Acceptance criteria**: max <= 2133333.
   - **Priority**: should_have

## Semantic Invariants

- **ID**: INV-001
  - **Requirement**: SWMR coherence (covers KPI-ISA-1).
  - **Acceptance criteria**: no two L1Ds in M.
  - **Priority**: must_have
  - **Model check**: coherence monitor in the arch model.

## Physical Design Requirements

1. **ID**: PHYS-001
   - **Requirement**: die 3x3.
   - **Acceptance criteria**: DRC clean.
   - **Priority**: must_have
   - **Model check**: not model-testable -- physical.
"""

_ERS = {"ers": {
    "functional_requirements": ["FR-TOP-1 [HARD, KPI-ISA-1]: top ports"],
    "system_invariants": [{"id": "INV-001", "description": "SWMR", "affected_blocks": ["l1d", "bus"]}],
    "validation_dv_requirements": [{"id": "VAL-001", "requirement": "[HARD] riscv-tests", "measurable_kpi": "count",
                                    "threshold": "241", "test_method": "h", "covers": ["KPI-ISA-1", "FR-CPU-2"]}],
    "per_block_requirements": [{"block": "l1d", "requirements": ["hit latency 2 cycles (PERF-001)"]}],
    "open_items": ["C-ERS-1: fog params missing"],
}}

_BD = {"blocks": [{"name": "core", "tier": 1, "instances": 2, "owns": ["PERF-001", "INV-001"]},
                  {"name": "l1d", "tier": 1}, {"name": "bus", "tier": 1},
                  {"name": "fabric", "tier": 0, "kind": "primitive", "primitive": "cs_fabric",
                   "fabric": {"masters": [{"name": "core0"}, {"name": "core1"}], "slaves": [{"name": "ram", "protocol": "axi4", "base": 0, "size": 4096}]}}],
       "connections": [
           {"edge_id": "core0__m_bus__to__bus__s_c0", "producer_block": "core", "producer_instance": 0, "producer_port": "m_bus",
            "consumer_block": "bus", "consumer_port": "s_c0", "handshake_protocol": "req_resp", "data_width_bits": 64},
           {"edge_id": "core1__m_bus__to__bus__s_c1", "producer_block": "core", "producer_instance": 1, "producer_port": "m_bus",
            "consumer_block": "bus", "consumer_port": "s_c1", "handshake_protocol": "req_resp", "data_width_bits": 64},
           {"edge_id": "bus__m_fab__to__fabric__s_core0", "producer_block": "bus", "producer_port": "m_fab",
            "consumer_block": "fabric", "consumer_port": "s_core0", "handshake_protocol": "axi4", "data_width_bits": 64},
       ]}


def test_ids_and_priority():
    assert is_item_id("PERF-001") and is_item_id("FR-CPU-2") and is_item_id("INV-RV_EXEC-003") and is_item_id("C-ERS-1")
    assert not is_item_id("perf") and not is_item_id("SI-COH-SWMR")
    assert item_must_have({"priority": "must_have"}) and item_must_have({"priority": "", "text": "x [HARD, R1]"})
    assert not item_must_have({"priority": "should_have", "text": ""})


def test_extract_prd_frd_ers():
    items, probs = extract.extract_prd_items(_PRD)
    ids = {i["id"] for i in items}
    assert {"FR-TOP-1", "FR-CPU-2", "KPI-ISA-1", "KPI-FPS-1", "CON-1", "Q-1"} <= ids
    assert any("no id" in p for p in probs)
    fr2 = next(i for i in items if i["id"] == "FR-CPU-2")
    assert fr2["priority"] == "must_have" and fr2["text"].startswith("Two RV64GC")
    fitems, fprobs = extract.extract_frd_items(_FRD)
    assert [i["id"] for i in fitems] == ["PERF-001", "PERF-002", "INV-001", "PHYS-001"] and fprobs == []
    assert "KPI-FPS-1" in fitems[0]["extra"]["refs"]
    eitems, elinks, eprobs = extract.extract_ers_items(_ERS)
    eids = {i["id"] for i in eitems}
    assert {"FR-TOP-1", "INV-001", "VAL-001", "ERS-l1d-1", "C-ERS-1"} <= eids
    assert ("VAL-001", "KPI-ISA-1", "covers") in elinks and ("INV-001", "block:l1d", "owned_by") in elinks
    assert ("ERS-l1d-1", "PERF-001", "derives_from") in elinks
    uitems, ulinks = extract.extract_uarch_items("## 6.1\n- INV-L1D-001 hit returns data in 2 cycles (PERF-001)\nMEETS PERF-001", "l1d")
    assert uitems[0]["id"] == "INV-L1D-001" and ("block:l1d", "PERF-001", "cites") in ulinks


def test_validators_catch_the_benchmark_defects():
    folded = {"blocks": [{"name": "fab", "interfaces": {"s_hart": "s_hart0_io / s_hart1_io (one per instance)"}}, {"name": "io"}],
              "connections": [{"producer_block": "io", "producer_port": "m", "consumer_block": "fab", "consumer_port": "s_hart"},
                              {"producer_block": "fab", "producer_port": "s_hart", "consumer_block": "io", "consumer_port": "m", "handshake_protocol": "req_resp"},
                              {"producer_block": "ghost", "producer_port": "x", "consumer_block": "io", "consumer_port": "y"}]}
    codes = {p["code"] for p in validate.validate_block_diagram(folded)}
    assert {"BD_FOLDED_INSTANCES", "BD_REVERSED_DUP", "BD_UNKNOWN_BLOCK"} <= codes
    multi = {"blocks": [{"name": "core", "instances": 2}, {"name": "bus"}],
             "connections": [{"producer_block": "core", "producer_port": "m", "consumer_block": "bus", "consumer_port": "s"}]}
    assert {p["code"] for p in validate.validate_block_diagram(multi)} == {"BD_INSTANCE_UNSPECIFIED"}
    assert validate.validate_block_diagram(_BD) == []
    contracts = {"contracts": [
        {"edge_id": "bus__m_fab__to__fabric__s_core0", "producer_block": "bus", "producer_port": "m_fab", "consumer_block": "fabric",
         "consumer_port": "s_core0", "handshake_protocol": "axi4", "data_width_bits": 32, "fields": [{"name": "data", "width": 32}],
         "timing": {}},
    ]}
    codes = {p["code"] for p in validate.validate_contracts(contracts, _BD)}
    assert {"CT_BUS_PHANTOM_FIELD", "CT_WIDTH_MISMATCH", "CT_MISSING_EDGE", "CT_FABRIC_PORT_UNREACHED"} <= codes


def _write(tmp_path):
    (tmp_path / ".coresmith").mkdir(parents=True, exist_ok=True)
    (tmp_path / "arch").mkdir(exist_ok=True)
    (tmp_path / "inputs").mkdir(exist_ok=True)
    (tmp_path / "inputs" / "task.yaml").write_text("top: soc_top\n")
    (tmp_path / ".coresmith" / "prd_spec.json").write_text(json.dumps(_PRD))
    (tmp_path / "arch" / "frd_spec.md").write_text(_FRD)
    (tmp_path / ".coresmith" / "ers_spec.json").write_text(json.dumps(_ERS))
    (tmp_path / ".coresmith" / "block_diagram.json").write_text(json.dumps(_BD))


def test_register_and_state_machine(tmp_path):
    _write(tmp_path)
    db = open_project(tmp_path)
    assert st.status(db, tmp_path)["stage"] == "requirements"
    r = register(db, tmp_path, "prd", ".coresmith/prd_spec.json")
    assert r["ok"] is False and any(p["code"] == "PRD_ITEM" for p in r["problems"])   # the id-less FR is refused
    _PRD["prd"]["functional_requirements"].pop()
    (tmp_path / ".coresmith" / "prd_spec.json").write_text(json.dumps(_PRD))
    r = register(db, tmp_path, "prd", ".coresmith/prd_spec.json")
    assert r["ok"] and r["items"]["items"] == 6
    assert db.artifact("prd")["version"] == 1 and len(db.questions()) == 1
    blk = st.entry(db, tmp_path, "requirements")
    assert [b["code"] for b in blk] == ["MISSING_ARTIFACT"]
    r = register(db, tmp_path, "frd", "arch/frd_spec.md")
    assert r["ok"] and r["links"] >= 1
    blk = {b["code"]: b for b in st.entry(db, tmp_path, "requirements")}
    # KPI-ISA-1 is only covered via the FRD INV-001 text -> derives_from link; PERF-002 has no model check but is should_have
    assert "FRD_NO_MODEL_CHECK" not in blk and "OPEN_QUESTIONS" in blk and blk["OPEN_QUESTIONS"]["ids"] == ["Q1"]
    assert st.advance(db, tmp_path)["advanced"] is False
    qid = db.questions()[0]["id"]
    assert db.answer_question(qid, "measured: mean 206, max 294")
    res = st.advance(db, tmp_path)
    assert res["advanced"] and res["stage"] == "arch_model"
    assert [b["code"] for b in st.entry(db, tmp_path, "arch_model")] == ["MISSING_ARTIFACT"]
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "arch.json").write_text("{}")
    db.register_artifact("arch_model", "model/arch.json")
    blk = {b["code"]: b for b in st.entry(db, tmp_path, "arch_model")}
    assert blk["MODEL_EVAL_MISSING"]["ids"] == ["INV-001", "PERF-001", "PHYS-001"]
    db.add_check("PERF-001", "model_eval", "pass", evidence="1.9M cyc/frame", sha="abc")
    db.add_check("INV-001", "model_eval", "fail", evidence="two M holders at t=12")
    db.add_check("PHYS-001", "model_eval", "not_testable", evidence="")
    blk = {b["code"]: b for b in st.entry(db, tmp_path, "arch_model")}
    assert blk["MODEL_EVAL_FAILED"]["ids"] == ["INV-001"] and blk["MODEL_EVAL_NO_REASON"]["ids"] == ["PHYS-001"]
    assert db.item("PERF-001")["status"] == "verified" and db.item("INV-001")["status"] == "failed"
    db.add_check("INV-001", "model_eval", "pass", evidence="fixed", sha="def")
    db.add_check("PHYS-001", "model_eval", "not_testable", evidence="physical design")
    assert st.advance(db, tmp_path)["advanced"] and st.current(db) == "decomposition"
    r = register(db, tmp_path, "block_diagram", ".coresmith/block_diagram.json")
    assert r["ok"], r["problems"]
    r = register(db, tmp_path, "ers", ".coresmith/ers_spec.json")
    assert r["ok"]
    blk = {b["code"]: b for b in st.entry(db, tmp_path, "decomposition")}
    assert "REQ_UNOWNED" not in blk, blk       # PERF-001/INV-001 owned via the diagram; PHYS excluded
    assert st.advance(db, tmp_path)["advanced"] and st.current(db) == "interfaces"
    codes = [b["code"] for b in st.entry(db, tmp_path, "interfaces")]
    assert codes == ["MISSING_ARTIFACT", "MISSING_ARTIFACT"]
    # re-registering an unchanged artifact keeps its version; a changed one bumps it
    assert register(db, tmp_path, "prd", ".coresmith/prd_spec.json")["artifact"]["version"] == 1
    _PRD["prd"]["title"] = "t2"
    (tmp_path / ".coresmith" / "prd_spec.json").write_text(json.dumps(_PRD))
    assert register(db, tmp_path, "prd", ".coresmith/prd_spec.json")["artifact"]["version"] == 2


def test_cli_round_trip(tmp_path):
    _write(tmp_path)
    _PRD["prd"]["functional_requirements"] = [f for f in _PRD["prd"]["functional_requirements"] if ":" in f]
    (tmp_path / ".coresmith" / "prd_spec.json").write_text(json.dumps(_PRD))
    cli = Path(__file__).resolve().parents[2] / "bin" / "coresmith"
    env = {"CORESMITH_PROJECT_ROOT": str(tmp_path), "PATH": "/usr/bin:/bin", "PYTHONPATH": str(Path(__file__).resolve().parents[2])}

    def run(*a):
        p = subprocess.run([sys.executable, str(cli), *a, "--json"], capture_output=True, text=True, env=env, timeout=120)
        return p.returncode, (json.loads(p.stdout) if p.stdout.strip().startswith("{") else p.stdout)

    rc, out = run("status")
    assert rc == 0 and out["stage"]["stage"] == "requirements"
    rc, out = run("register", "prd", ".coresmith/prd_spec.json")
    assert rc == 0 and out["ok"]
    rc, out = run("register", "frd", "arch/frd_spec.md")
    assert rc == 0
    rc, out = run("stage", "next")
    assert rc == 1 and out["advanced"] is False and out["blocked_by"][0]["code"] == "OPEN_QUESTIONS"
    rc, out = run("question", "answer", "1", "--ruling", "triangles: mean 206 max 294")
    assert rc == 0 and out["ruling_id"]
    rc, out = run("stage", "next")
    assert rc == 0 and out["stage"] == "arch_model"
    rc, out = run("item", "show", "PERF-001")
    assert rc == 0 and out["item"]["model_check"].startswith("arch model")
    rc, out = run("check", "add", "PERF-001", "model_eval", "pass", "--evidence", "1.9M")
    assert rc == 0
    rc, out = run("item", "list", "--must")
    assert rc == 0 and {i["id"] for i in out["items"]} >= {"PERF-001", "INV-001", "KPI-ISA-1"}
    rc, out = run("link", "PERF-002", "block:core", "owned_by")
    assert rc == 0
    rc, out = run("link", "PERF-002", "block:core", "bogus")
    assert rc == 2


def test_schema_verb_prints_every_kind(tmp_path):
    from orchestrator.harness.tools.schema import SCHEMAS, schema
    for k in ("prd", "frd", "ers", "block_diagram", "contracts", "abi", "uarch", "arch_model"):
        assert k in SCHEMAS and "register" in schema(k) or "coresmith" in schema(k)
    assert "unknown kind" in schema("nope")
    assert "instances" in schema("block_diagram") and "Model check" in schema("frd")
