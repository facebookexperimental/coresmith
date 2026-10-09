# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""The Architect's step 2: the executable SAD (abstract SystemC performance model) and fabric derivation."""
import json
from pathlib import Path

import pytest

from orchestrator.harness.tools import model as mt
from orchestrator.state_store.project_db import open_project
from orchestrator.systemc_model import arch_model as am
from orchestrator.systemc_model.toolchain import detect

_SPEC = {"name": "tiny", "clock_mhz": 100, "addr_width": 32,
         "components": [
             {"name": "cpu", "kind": "initiator", "instances": 2, "energy_pj_per_txn": 10},
             {"name": "dma", "kind": "both", "base": "0x10000000", "size": "0x1000", "latency_cycles": 1},
             {"name": "ram", "kind": "target", "base": "0x80000000", "size": "0x10000", "latency_cycles": 4,
              "bytes_per_cycle": 8, "protocol": "axi4", "energy_pj_per_byte": 1},
             {"name": "uart", "kind": "target", "base": "0x20000000", "size": "0x1000", "latency_cycles": 2, "protocol": "apb"}],
         "fabric": {"latency_cycles": 2, "bytes_per_cycle": 8},
         "links": [{"from": "dma", "to": "ram", "bytes_per_cycle": 16}]}


def test_spec_validation_and_codegen():
    spec = am.ArchSpec.from_json(_SPEC)
    assert spec.validate() == []
    bad = am.ArchSpec.from_json({**_SPEC, "components": _SPEC["components"] + [{"name": "ram2", "kind": "target", "base": "0x80001000", "size": "0x1000"}]})
    assert any("overlap" in e for e in bad.validate())
    bad2 = am.ArchSpec.from_json({**_SPEC, "components": [{"name": "x", "kind": "target", "base": "0x300", "size": "0x300"}]})
    errs = bad2.validate()
    assert any("power of two" in e for e in errs) and any("no initiator" in e for e in errs)
    top = am.render_arch_top(spec)
    assert "cs_arch_fabric fabric;" in top and "std::vector<cs_arch_initiator*> u_cpu;" in top
    assert "cs_arch_target u_ram;" in top and "cs_arch_target u_dma_t;" in top   # 'both' = initiator + target window
    assert 'fabric.add_target("ram", 0x80000000ULL, 0x10000ULL)' in top
    assert "fabric.set_link(kv.second, tidx[\"ram\"], 2, 16)" in top
    assert spec.link_params("dma", "ram") == (2, 16) and spec.link_params("cpu", "ram") == (2, 8)


def test_derive_fabric_from_stats():
    spec = am.ArchSpec.from_json(_SPEC)
    stats = {"total_cycles": 1000, "links": [
        {"from": "cpu", "to": "ram", "bytes": 4000, "bytes_per_cycle": 4.0, "utilization": 0.5, "max_outstanding": 3},
        {"from": "dma", "to": "ram", "bytes": 12000, "bytes_per_cycle": 12.0, "utilization": 0.9, "max_outstanding": 6},
        {"from": "cpu1", "to": "uart", "bytes": 8, "bytes_per_cycle": 0.008, "utilization": 0.01, "max_outstanding": 1}]}
    fs = am.derive_fabric(spec, stats)
    assert [m["name"] for m in fs["masters"]] == ["cpu0", "cpu1", "dma"]
    assert {s["name"]: s["protocol"] for s in fs["slaves"]} == {"dma_regs": "axi4", "ram": "axi4", "uart": "apb"}
    assert fs["data_width"] == 128 and fs["max_outstanding"] == 6    # 12 B/cyc * 2 headroom -> 192 b -> capped 128
    assert am.derive_fabric(spec, {"total_cycles": 10, "links": []})["data_width"] == 32
    from orchestrator.fabric import FabricSpec
    assert FabricSpec.from_json(fs).validate() == []


def test_cli_helpers_without_toolchain(tmp_path, monkeypatch):
    db = open_project(tmp_path)
    r = mt.arch_init(tmp_path)
    assert r["created"] and Path(r["path"]).exists()
    assert mt.arch_init(tmp_path)["created"] is False
    assert mt.fabric_derive(db, tmp_path)["ok"] is False   # no stats yet
    (tmp_path / "model" / "arch" / "arch_model.json").write_text(json.dumps({**_SPEC, "components": []}))
    r = mt.arch_build(tmp_path)
    assert r["ok"] is False and "no initiator" in " ".join(r["problems"])
    r = mt.arch_eval(db, tmp_path)
    assert r["ok"] is False and r["error"] == "ARCH_MODEL_NOT_BUILT" and r["required"].endswith("arch_model_top.h")


_HARNESS = r'''
#include "arch_model_top.h"
#include <cstdio>
using namespace sc_core;
int sc_main(int, char**) {
    sc_clock clk("clk", cs_clock_period()); sc_signal<bool> rst_n("rst_n");
    arch_model_top top("top"); top.clk(clk); top.rst_n(rst_n);
    for (auto* u : top.u_cpu) u->body = [&](cs_arch_initiator& me) {
        for (int i = 0; i < 100; ++i) { me.wr64(0x80000000ULL + i * 8, i); me.compute(3); }
        uint64_t sum = 0; for (int i = 0; i < 100; ++i) sum += me.rd64(0x80000000ULL + i * 8);
        std::printf("[cpu] sum=%llu\n", (unsigned long long)sum);
        me.wr64(0x20000000ULL, 'A');
    };
    top.u_dma[0]->body = [&](cs_arch_initiator& me) {
        uint8_t buf[64] = {0};
        for (int i = 0; i < 200; ++i) me.issue(true, 0x80008000ULL + (i % 64) * 64, buf, 64);
        me.issue(false, 0x30000000ULL, buf, 8);   // unmapped -> DECERR
    };
    rst_n.write(false); sc_start(3 * cs_clock_period()); rst_n.write(true);
    while (!top.all_done()) sc_start(1000 * cs_clock_period());
    uint64_t cyc = cs_cycles(sc_time_stamp());
    std::ofstream f("stats.json"); top.stats_json(f, cyc); f.close();
    std::printf("FRD_EVAL {\"id\": \"PERF-001\", \"status\": \"%s\", \"evidence\": \"cycles=%llu (LT)\"}\n", cyc < 200000 ? "pass" : "fail", (unsigned long long)cyc);
    std::printf("FRD_EVAL {\"id\": \"INV-001\", \"status\": \"pass\", \"evidence\": \"dma writes fenced\"}\n");
    std::puts("FRD_EVAL_DONE");
    return 0;
}
'''

_FRD = """# FRD
## Performance Requirements
1. **ID**: PERF-001
   - **Requirement**: mission under 200k cycles.
   - **Acceptance criteria**: cycles < 200000.
   - **Priority**: must_have
   - **Model check**: arch model cycle count.
## Semantic Invariants
- **ID**: INV-001
  - **Requirement**: dma writes visible.
  - **Acceptance criteria**: x.
  - **Priority**: must_have
  - **Model check**: x.
"""


class _FakeAgent:
    def __init__(self, *a, **k):
        pass

    async def generate(self, *, project_root, blocks, attempt=1, compiler_log="", run_log="", summary=None, arch=False):
        assert arch is True
        d = Path(project_root) / "model" / "arch" / "frd_eval"
        d.mkdir(parents=True, exist_ok=True)
        (d / "frd_eval.cpp").write_text('#include <fstream>\n' + _HARNESS)
        return {"written": True}


@pytest.mark.slow
@pytest.mark.skipif(not detect()["ok"], reason="SystemC toolchain not available")
def test_arch_model_builds_runs_evaluates_and_derives_a_fabric(tmp_path):
    db = open_project(tmp_path)
    (tmp_path / "arch").mkdir()
    (tmp_path / "arch" / "frd_spec.md").write_text(_FRD)
    (tmp_path / "model" / "arch").mkdir(parents=True)
    (tmp_path / "model" / "arch" / "arch_model.json").write_text(json.dumps(_SPEC))
    b = mt.arch_build(tmp_path)
    assert b["ok"], b["log"]
    r = mt.arch_run(tmp_path, ns=2000)
    assert r["ok"], r["log"]
    assert r["stats"]["total_cycles"] > 0 and r["stats"]["decerr"] == 0   # smoke: idle initiators
    ev = mt.arch_eval(db, tmp_path, agent=_FakeAgent(), repairs=0)
    assert ev["ok"], ev
    st = am.read_stats(tmp_path / "model" / "arch")
    links = {(lk["from"], lk["to"]): lk for lk in st["links"]}
    assert links[("dma", "ram")]["bytes"] == 200 * 64 and links[("cpu", "ram")]["txns"] == 200
    assert st["decerr"] == 1 and st["energy_pj"] > 0
    assert db.item("PERF-001") is None                       # FRD not registered here: checks still recorded
    assert {c["item_id"]: c["status"] for c in db.checks(kind="model_eval")} == {"PERF-001": "pass", "INV-001": "pass"}
    assert db.artifact("arch_model")["sha"]
    fd = mt.fabric_derive(db, tmp_path)
    assert fd["ok"], fd
    assert (tmp_path / ".coresmith" / "fabric_spec.json").exists()
    assert fd["fabric"]["data_width"] in (64, 128) and len(fd["fabric"]["masters"]) == 3
