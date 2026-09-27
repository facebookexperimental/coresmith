# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""soc_model.cpp + build files from the block diagram and contracts (B2)."""
from __future__ import annotations

from pathlib import Path

from .conventions import COMMON_HEADER, channel_binding, model_name


def render_soc_model(blocks: list[str], edges: list[dict], *, top_name: str = "soc",
                     smoke_ns: int = 1000) -> str:
    """``soc_model.cpp``: instantiate every block model, bind every edge,
    drive clk/rst_n, run for ``--ns N`` (default ``smoke_ns``) and dump state."""
    L = [f"// soc_model.cpp -- GENERATED SystemC TLM-2.0 LT model of '{top_name}' (B2). Do not edit.",
         "#include \"cs_model_common.h\""]
    for b in blocks:
        L.append(f"#include \"{model_name(b)}.h\"")
    L += ["", "using namespace sc_core;", "",
          "static void cs_clock(sc_signal<bool>& clk, sc_signal<bool>& rst_n, double ns) {",
          "    (void)ns; (void)clk; (void)rst_n;", "}", "",
          "int sc_main(int argc, char** argv) {",
          f"    double run_ns = {smoke_ns};",
          "    for (int i = 1; i + 1 < argc; ++i) if (std::string(argv[i]) == \"--ns\") run_ns = std::atof(argv[i + 1]);",
          "    sc_clock clk(\"clk\", cs_clock_period());",
          "    sc_signal<bool> rst_n(\"rst_n\");"]
    for b in blocks:
        L.append(f"    {model_name(b)} u_{b}(\"{b}\");")
        L.append(f"    u_{b}.clk(clk); u_{b}.rst_n(rst_n);")
    fifos, signals, signal_of = [], [], {}
    for e in edges:
        bd = channel_binding(e)
        if bd["producer"] not in blocks or bd["consumer"] not in blocks:
            continue
        p, c = f"u_{bd['producer']}", f"u_{bd['consumer']}"
        if bd["kind"] == "socket":
            L.append(f"    {p}.{bd['producer_member']}.bind({c}.{bd['consumer_member']});   // {bd['edge_id']}")
        elif bd["kind"] == "fifo":
            name = f"f_{len(fifos)}"
            fifos.append(name)
            L.append(f"    sc_fifo<cs_beat_t> {name}(\"{name}\", 16);   // {bd['edge_id']}")
            L.append(f"    {p}.{bd['producer_member']}({name}); {c}.{bd['consumer_member']}({name});")
        else:
            # One sc_signal per producer port: an sc_out binds exactly once, so a
            # static value fanning out to several blocks shares the signal.
            key = (p, bd["producer_member"])
            if key in signal_of:
                name = signal_of[key]
                L.append(f"    {c}.{bd['consumer_member']}({name});   // {bd['edge_id']} (fan-out)")
                continue
            name = f"s_{len(signals)}"
            signals.append(name)
            signal_of[key] = name
            L.append(f"    sc_signal<cs_word_t> {name}(\"{name}\");   // {bd['edge_id']}")
            L.append(f"    {p}.{bd['producer_member']}({name}); {c}.{bd['consumer_member']}({name});")
    L += ["    rst_n.write(false);",
          "    sc_start(3 * cs_clock_period());",
          "    rst_n.write(true);"]
    for b in blocks:
        L.append(f"    u_{b}.reset();")
    L += ["    sc_start(sc_time(run_ns, SC_NS));",
          "    std::cout << \"== soc_model state @ \" << sc_time_stamp() << std::endl;"]
    for b in blocks:
        L.append(f"    std::cout << \"[{b}]\" << std::endl; u_{b}.dump_state(std::cout);")
    L += ["    std::cout << \"SOC_MODEL_SMOKE_OK\" << std::endl;", "    return 0;", "}", ""]
    return "\n".join(L)


MAKEFILE = '''# GENERATED (B2): build the SystemC SoC model. Needs SystemC 2.3 (headers on the
# include path, libsystemc on the link path); override SYSTEMC_HOME as needed.
SYSTEMC_HOME ?= {systemc_home}
CXX ?= g++
CXXFLAGS ?= -std=c++17 -O1 -Wall -Wno-unused-parameter
INC := -I.
LIB := -lsystemc
ifneq ($(strip $(SYSTEMC_HOME)),)
INC += -I$(SYSTEMC_HOME)/include
LIB := -L$(SYSTEMC_HOME)/lib -L$(SYSTEMC_HOME)/lib-linux64 -Wl,-rpath,$(SYSTEMC_HOME)/lib -lsystemc
endif
SRCS := {srcs}
soc_model: $(SRCS) $(wildcard *.h)
\t$(CXX) $(CXXFLAGS) $(INC) $(SRCS) $(LIB) -o $@
clean:
\trm -f soc_model
.PHONY: clean
'''


def write_build(project_root, blocks: list[str], edges: list[dict], *, top_name: str = "soc",
                model_dir=None, systemc_home: str = "") -> Path:
    """Write ``cs_model_common.h``, ``soc_model.cpp`` and the Makefile into
    ``model/`` (block ``<block>_model.{h,cpp}`` are the agent's)."""
    md = Path(model_dir) if model_dir else Path(project_root) / "model"
    md.mkdir(parents=True, exist_ok=True)
    (md / "cs_model_common.h").write_text(COMMON_HEADER)
    (md / "soc_model.cpp").write_text(render_soc_model(blocks, edges, top_name=top_name))
    srcs = ["soc_model.cpp"] + [f"{model_name(b)}.cpp" for b in blocks]
    (md / "Makefile").write_text(MAKEFILE.format(systemc_home=systemc_home, srcs=" ".join(srcs)))
    return md
