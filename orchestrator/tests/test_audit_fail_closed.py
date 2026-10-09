import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

from orchestrator.harness import gate_sim
from orchestrator.langgraph import macro_prebind
from orchestrator.langgraph.macro_registry import ShellSpec


def _prebind(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "source.v"
    source.write_text("module source; endmodule\n")
    model = tmp_path / "sram.v"
    model.write_text("""module sram(input clk0,input csb0,input web0,input [3:0] wmask0,
input [8:0] addr0,input [31:0] din0,output [31:0] dout0,
input clk1,input csb1,input [8:0] addr1,output [31:0] dout1); endmodule\n""")
    spec = ShellSpec(kind="sram", width=32, depth=512, nport=2)
    macro = SimpleNamespace(name="sram", verilog=str(model), ports="1rw1r",
                            data_bits=32, words=512, mask_bits=4, kind="sram")
    result = macro_prebind.PrebindResult(bindings=[(spec, macro)],
                                         mask_lanes={(32, 512, 2): 4})
    shell = macro_prebind.write_bound_shell(result, tmp_path / "bound.v")
    manifest = macro_prebind.write_prebind_manifest(
        result, tmp_path / "manifest.json", sources=[source], bound_shell=shell)
    netlist = tmp_path / "netlist.v"
    netlist.write_text("module top; endmodule\n")
    macro_prebind.bind_manifest_to_netlist(manifest, netlist)
    return Path(manifest), netlist


def test_prebind_manifest_is_bound_to_exact_netlist_and_models(tmp_path):
    manifest, netlist = _prebind(tmp_path)
    doc, error = gate_sim._binding_manifest(manifest, netlist)
    assert not error and doc["bindings"][0]["nport"] == 2

    netlist.write_text("module changed; endmodule\n")
    assert "netlist hash" in gate_sim._binding_manifest(manifest, netlist)[1]


def test_prebind_manifest_rejects_stale_source_and_bound_shell(tmp_path):
    manifest, netlist = _prebind(tmp_path)
    doc = json.loads(manifest.read_text())
    Path(doc["sources"][0]["path"]).write_text("module changed; endmodule\n")
    assert "artifact hash mismatch" in gate_sim._binding_manifest(manifest, netlist)[1]

    manifest, netlist = _prebind(tmp_path / "second")
    doc = json.loads(manifest.read_text())
    Path(doc["bound_shell"]["path"]).write_text("module changed; endmodule\n")
    assert "artifact hash mismatch" in gate_sim._binding_manifest(manifest, netlist)[1]


def test_manifest_binding_uses_explicit_port_count_and_rejects_ambiguity(tmp_path):
    manifest, netlist = _prebind(tmp_path)
    doc, error = gate_sim._binding_manifest(manifest, netlist)
    assert not error
    shell = gate_sim.ShellBinding(module="cs_mem_macro_shell", width=32,
                                  depth=512, nmask=4)
    gate_sim.bind_macro_shells_for_sim([shell], doc)
    assert shell.macro == "sram"
    assert ".dout1(rdata1)" in shell.replacement

    ambiguous = json.loads(json.dumps(doc))
    other = dict(ambiguous["bindings"][0])
    other["nport"] = 1
    ambiguous["bindings"].append(other)
    shell2 = gate_sim.ShellBinding(module="cs_mem_macro_shell", width=32,
                                   depth=512, nmask=4)
    gate_sim.bind_macro_shells_for_sim([shell2], ambiguous)
    assert not shell2.replacement and "2 exact entries" in shell2.error

    wrong_mask = gate_sim.ShellBinding(module="cs_mem_macro_shell", width=32,
                                       depth=512, nmask=1)
    gate_sim.bind_macro_shells_for_sim([wrong_mask], doc)
    assert not wrong_mask.replacement and "0 exact entries" in wrong_mask.error


def test_openroad_discovery_never_guesses_home_checkout(tmp_path, monkeypatch):
    from orchestrator.pdk.deployments import sky130
    guessed = tmp_path / "openroad-src/build/bin/openroad"
    guessed.parent.mkdir(parents=True)
    guessed.write_text("binary")
    wrapper = str(tmp_path / "openroad-nix.sh")
    Path(wrapper).write_text("#!/bin/sh\n")
    monkeypatch.delenv("CORESMITH_BACKEND_OPENROAD", raising=False)
    monkeypatch.setattr(sky130.Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(sky130.shutil, "which", lambda _name, *a, **k: None)
    assert sky130.openroad_for_backend(wrapper) == (wrapper, "unreachable")


def test_explicit_openroad_command_resolves_from_path(monkeypatch):
    from orchestrator.pdk.deployments import sky130
    monkeypatch.setenv("CORESMITH_BACKEND_OPENROAD", "openroad-real")
    monkeypatch.setattr(
        sky130.shutil, "which",
        lambda name, *a, **k: "/run/tools/openroad-real"
        if name == "openroad-real" else None)
    assert sky130.DEPLOYMENT.resolve_openroad_bin() == "/run/tools/openroad-real"


def test_backend_preflight_exception_fails_closed(monkeypatch):
    from orchestrator.daemon import server
    from orchestrator.pdk.deployments import sky130

    def broken():
        raise RuntimeError("probe broke")

    monkeypatch.setattr(sky130, "backend_tools_preflight", broken)
    result = server._backend_preflight()
    assert result["ok"] is False
    assert result["warnings"] == []
    assert "RuntimeError: probe broke" in result["errors"][0]


def test_sta_shim_resolves_explicit_command_and_reports_missing(tmp_path):
    shim = Path(__file__).resolve().parents[2] / "bin/sta"
    real = tmp_path / "sta-real"
    real.write_text("#!/bin/sh\nexit 7\n")
    real.chmod(0o755)
    env = {**os.environ,
           "PATH": str(tmp_path) + os.pathsep + os.environ.get("PATH", ""),
           "CORESMITH_REAL_STA": "sta-real"}
    assert subprocess.run([str(shim), "--version"], env=env).returncode == 7

    env["CORESMITH_REAL_STA"] = "missing-sta"
    failed = subprocess.run([str(shim), "--version"], env=env,
                            capture_output=True, text=True)
    assert failed.returncode != 0
    assert "CORESMITH_REAL_STA" in failed.stderr and "unavailable" in failed.stderr
