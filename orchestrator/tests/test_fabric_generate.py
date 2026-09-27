# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""B1: the SoC fabric is generated from a FabricSpec over the vendored pulp
IP -- rendered, elaborated to plain Verilog by yosys-slang, and verified by
its own cocotbext-axi testbench."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from orchestrator.fabric import (
    CHANNELS,
    FabricMaster,
    FabricSlave,
    FabricSpec,
    generate_fabric,
    render_testbench,
    render_wrapper_sv,
    slang_available,
)
from orchestrator.fabric.amba import ports_for

_SV = Path(__file__).resolve().parents[1] / "langgraph" / "rtl_lib" / "fabric" / "sv"


def _spec():
    return FabricSpec(name="soc", masters=[FabricMaster("cpu0"), FabricMaster("gpu")],
                      slaves=[FabricSlave("ram", "axi4", 0x8000_0000, 0x1000_0000),
                              FabricSlave("uart", "apb", 0x1000_0000, 0x1000),
                              FabricSlave("gpu_regs", "axi_lite", 0x1000_2000, 0x1000)])


class TestSpec:
    def test_validation(self):
        assert _spec().validate() == []
        bad = _spec()
        bad.slaves.append(FabricSlave("ram2", "axi4", 0x8000_1000, 0x1000))   # overlaps ram
        bad.slaves.append(FabricSlave("odd", "apb", 0x3000, 0x3000))           # not pow2
        bad.masters.append(FabricMaster("cpu1", id_width=6))                   # id width mismatch
        errs = bad.validate()
        assert any("overlap" in e for e in errs) and any("power of two" in e for e in errs)
        assert any("id_width" in e for e in errs)

    def test_json_roundtrip_and_digest(self):
        s = _spec()
        d = json.loads(json.dumps(s.to_json()))
        s2 = FabricSpec.from_json(d)
        assert s2.digest() == s.digest() and s2.slaves[1].protocol == "apb"
        s2.slaves[1].size = 0x2000
        assert s2.digest() != s.digest()
        assert FabricSpec.from_json({"name": "x", "masters": [{"name": "m", "id_width": "0x4"}],
                                     "slaves": [{"name": "s", "base": "0x1000", "size": "0x1000"}]}).validate() == []

    def test_mst_id_width_grows_with_masters(self):
        s = _spec()
        assert s.mst_id_width == 5
        s.masters = s.masters[:1]
        assert s.mst_id_width == 4


class TestAmba:
    def test_channel_tables_and_port_naming(self):
        assert {c[0] for c in CHANNELS["apb"]} >= {"psel", "penable", "pready", "prdata", "pslverr"}
        p = ports_for("axi4", "s_cpu0", role="slave", AW=32, DW=32, IW=4, UW=1)
        by = {x["name"]: x for x in p}
        assert by["s_cpu0_awvalid"]["dir"] == "input" and by["s_cpu0_awready"]["dir"] == "output"
        assert by["s_cpu0_wstrb"]["width"] == 4 and by["s_cpu0_awid"]["width"] == 4
        m = {x["name"]: x for x in ports_for("apb", "m_uart", role="master", AW=32, DW=32, IW=5, UW=1)}
        assert m["m_uart_psel"]["dir"] == "output" and m["m_uart_prdata"]["dir"] == "input"


class TestRender:
    def test_wrapper_has_flat_amba_ports_and_the_address_map(self):
        sv = render_wrapper_sv(_spec())
        assert "module cs_fabric_soc (" in sv
        for p in ("s_cpu0_awvalid", "s_gpu_rdata", "m_ram_awid", "m_uart_psel", "m_gpu_regs_awaddr"):
            assert p in sv, p
        assert "start_addr: 32'h80000000, end_addr: 32'h90000000" in sv
        assert "axi_lite_to_apb" in sv and "axi_to_axi_lite" in sv and "axi_xbar #(" in sv
        assert sv == render_wrapper_sv(_spec())        # deterministic
        with pytest.raises(ValueError):
            render_wrapper_sv(FabricSpec(name="bad"))

    def test_testbench_renders_and_compiles(self):
        tb = render_testbench(_spec())
        compile(tb, "tb", "exec")
        assert "AxiRam" in tb and "ApbMem" in tb and "AxiLiteRam" in tb
        assert "UNMAPPED = 0x90001000" in tb

    def test_vendored_manifest_matches_files(self):
        man = json.loads((_SV / "MANIFEST.json").read_text())
        import hashlib
        for rel, sha in man["files"].items():
            assert hashlib.sha256((_SV / rel).read_bytes()).hexdigest() == sha, rel
        assert man["axi"]["tag"] == "v0.39.9" and man["common_cells"]["tag"] == "v1.37.0"
        assert (_SV / "axi" / "LICENSE").exists() and (_SV / "common_cells" / "LICENSE").exists()


_HAVE_SLANG = slang_available()
_HAVE_SIM = shutil.which("verilator") is not None and shutil.which("cocotb-config") is not None


@pytest.mark.slow
@pytest.mark.skipif(not _HAVE_SLANG, reason="yosys-slang not available")
def test_elaborate_to_plain_verilog_and_lint(tmp_path):
    art = generate_fabric(_spec(), tmp_path)
    v = Path(art.rtl_path)
    assert v.exists() and "module cs_fabric_soc" in v.read_text()
    assert not art.cached
    # plain Yosys (no -sv) reads it, and Verilator lints it
    yb = os.environ.get("CORESMITH_FABRIC_YOSYS") or "yosys"
    p = subprocess.run([yb, "-q", "-p", f"read_verilog {v}; hierarchy -check -top cs_fabric_soc"],
                       capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, (p.stdout + p.stderr)[-2000:]
    if shutil.which("verilator"):
        p = subprocess.run(["verilator", "--lint-only", "-Wno-fatal", "-Wno-WIDTH", "-Wno-UNUSED",
                            "-Wno-UNOPTFLAT", str(v), "--top-module", "cs_fabric_soc"],
                           capture_output=True, text=True, timeout=600)
        assert "%Error" not in p.stderr, p.stderr[-2000:]
    art2 = generate_fabric(_spec(), tmp_path)
    assert art2.cached


@pytest.mark.slow
@pytest.mark.skipif(not (_HAVE_SLANG and _HAVE_SIM), reason="yosys-slang/verilator/cocotb not available")
def test_generated_testbench_passes_in_simulation(tmp_path, monkeypatch):
    import orchestrator.langgraph.pipeline_helpers as ph
    monkeypatch.setattr(ph, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("CORESMITH_LINE_COV_GATE", "0")
    monkeypatch.setenv("CORESMITH_COVERAGE", "0")
    monkeypatch.setenv("CORESMITH_INTERFACE_VIP", "0")
    monkeypatch.setattr(ph, "create_golden_model_wrapper", lambda *a, **k: None)
    art = generate_fabric(_spec(), tmp_path / "rtl", tb_dir=tmp_path / "tb")
    res = ph.run_simulation({"name": "cs_fabric_soc"}, art.rtl_path, art.tb_path,
                            project_root=str(tmp_path))
    assert res["passed"], res.get("log", "")[-4000:]


def test_wrapper_declares_apb_types_once_for_many_apb_slaves():
    spec = FabricSpec(name="soc", masters=[FabricMaster("cpu0"), FabricMaster("gpu")],
                      slaves=[FabricSlave("ram", "axi4", 0x8000_0000, 0x1000_0000)]
                      + [FabricSlave(f"p{i}", "apb", 0x1000_0000 + i * 0x1000, 0x1000) for i in range(7)])
    assert spec.validate() == []
    sv = render_wrapper_sv(spec)
    assert sv.count("} apb_req_t;") == 1 and sv.count("} apb_resp_t;") == 1
    assert sv.count("axi_lite_to_apb #(") == 7
