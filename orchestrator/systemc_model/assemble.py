# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""soc_model_top.h / soc_model.cpp + build files from the block diagram and contracts (B2)."""
from __future__ import annotations

from pathlib import Path

from .conventions import COMMON_HEADER, channel_binding, model_name


def render_soc_top_header(blocks: list[str], edges: list[dict], *, top_name: str = "soc") -> str:
    """``soc_model_top.h``: ``SC_MODULE(soc_model_top)`` owning every block
    model (public ``u_<block>`` members), the channels and the bindings, plus
    ``reset_all()`` / ``dump_all()``. Shared by the smoke driver
    (``soc_model.cpp``) and the FRD evaluation harness (``frd_eval/``)."""
    L = [f"// soc_model_top.h -- GENERATED SystemC TLM-2.0 LT model of '{top_name}' (B2). Do not edit.",
         "#pragma once", "#include \"cs_model_common.h\""]
    for b in blocks:
        L.append(f"#include \"{model_name(b)}.h\"")
    L += ["", "SC_MODULE(soc_model_top) {", "  sc_core::sc_in<bool> clk;", "  sc_core::sc_in<bool> rst_n;"]
    for b in blocks:
        L.append(f"  {model_name(b)} u_{b};")
    binds, decls = [], []
    fifos, signals, signal_of = [], [], {}
    for e in edges:
        bd = channel_binding(e)
        if bd["producer"] not in blocks or bd["consumer"] not in blocks:
            continue
        p, c = f"u_{bd['producer']}", f"u_{bd['consumer']}"
        if bd["kind"] == "socket":
            binds.append(f"    {p}.{bd['producer_member']}.bind({c}.{bd['consumer_member']});   // {bd['edge_id']}")
        elif bd["kind"] == "fifo":
            name = f"f_{len(fifos)}"
            fifos.append(name)
            decls.append(f"  sc_core::sc_fifo<cs_beat_t> {name};   // {bd['edge_id']}")
            binds.append(f"    {p}.{bd['producer_member']}({name}); {c}.{bd['consumer_member']}({name});")
        else:
            # One sc_signal per producer port: an sc_out binds exactly once, so a
            # static value fanning out to several blocks shares the signal.
            key = (p, bd["producer_member"])
            if key in signal_of:
                binds.append(f"    {c}.{bd['consumer_member']}({signal_of[key]});   // {bd['edge_id']} (fan-out)")
                continue
            name = f"s_{len(signals)}"
            signals.append(name)
            signal_of[key] = name
            decls.append(f"  sc_core::sc_signal<cs_word_t> {name};   // {bd['edge_id']}")
            binds.append(f"    {p}.{bd['producer_member']}({name}); {c}.{bd['consumer_member']}({name});")
    L += decls
    inits = [f'u_{b}("{b}")' for b in blocks] + [f'{f}("{f}", 16)' for f in fifos] + [f'{s}("{s}")' for s in signals]
    L += ["", "  SC_CTOR(soc_model_top)", "    : " + ", ".join(inits) if inits else "  SC_CTOR(soc_model_top)", "  {"]
    for b in blocks:
        L.append(f"    u_{b}.clk(clk); u_{b}.rst_n(rst_n);")
    L += binds
    L += ["  }", "", "  void reset_all() {"]
    for b in blocks:
        L.append(f"    u_{b}.reset();")
    L += ["  }", "", "  void dump_all(std::ostream& os) const {"]
    for b in blocks:
        L.append(f'    os << "[{b}]" << std::endl; u_{b}.dump_state(os);')
    L += ["  }", "};", ""]
    return "\n".join(L)


def render_soc_model(blocks: list[str], edges: list[dict], *, top_name: str = "soc",
                     smoke_ns: int = 1000) -> str:
    """``soc_model.cpp``: the smoke driver -- instantiate ``soc_model_top``,
    drive clk/rst_n, run for ``--ns N`` (default ``smoke_ns``) and dump state."""
    L = [f"// soc_model.cpp -- GENERATED smoke driver of the '{top_name}' SystemC model (B2). Do not edit.",
         "#include \"soc_model_top.h\"", "", "using namespace sc_core;", "",
         "int sc_main(int argc, char** argv) {",
         f"    double run_ns = {smoke_ns};",
         "    for (int i = 1; i + 1 < argc; ++i) if (std::string(argv[i]) == \"--ns\") run_ns = std::atof(argv[i + 1]);",
         "    sc_clock clk(\"clk\", cs_clock_period());",
         "    sc_signal<bool> rst_n(\"rst_n\");",
         "    soc_model_top top(\"top\");",
         "    top.clk(clk); top.rst_n(rst_n);",
         "    rst_n.write(false);",
         "    sc_start(3 * cs_clock_period());",
         "    rst_n.write(true);",
         "    top.reset_all();",
         "    sc_start(sc_time(run_ns, SC_NS));",
         "    std::cout << \"== soc_model state @ \" << sc_time_stamp() << std::endl;",
         "    top.dump_all(std::cout);",
         "    std::cout << \"SOC_MODEL_SMOKE_OK\" << std::endl;", "    return 0;", "}", ""]
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
MODEL_SRCS := {model_srcs}
SRCS := soc_model.cpp $(MODEL_SRCS)
FRD_SRCS := $(wildcard frd_eval/*.cpp)
soc_model: $(SRCS) $(wildcard *.h)
\t$(CXX) $(CXXFLAGS) $(INC) $(SRCS) $(LIB) -o $@
# FRD evaluation harness (B2): the agent-authored frd_eval/*.cpp own sc_main.
frd_eval/frd_eval: $(MODEL_SRCS) $(FRD_SRCS) $(wildcard *.h) $(wildcard frd_eval/*.h)
\t$(CXX) $(CXXFLAGS) $(INC) -Ifrd_eval $(MODEL_SRCS) $(FRD_SRCS) $(LIB) -o $@
clean:
\trm -f soc_model frd_eval/frd_eval
.PHONY: clean
'''


def write_build(project_root, blocks: list[str], edges: list[dict], *, top_name: str = "soc",
                model_dir=None, systemc_home: str = "") -> Path:
    """Write ``cs_model_common.h``, ``soc_model.cpp`` and the Makefile into
    ``model/`` (block ``<block>_model.{h,cpp}`` are the agent's)."""
    md = Path(model_dir) if model_dir else Path(project_root) / "model"
    md.mkdir(parents=True, exist_ok=True)
    (md / "cs_model_common.h").write_text(COMMON_HEADER)
    (md / "soc_model_top.h").write_text(render_soc_top_header(blocks, edges, top_name=top_name))
    (md / "soc_model.cpp").write_text(render_soc_model(blocks, edges, top_name=top_name))
    (md / "frd_eval").mkdir(exist_ok=True)
    model_srcs = [f"{model_name(b)}.cpp" for b in blocks]
    (md / "Makefile").write_text(MAKEFILE.format(systemc_home=systemc_home, model_srcs=" ".join(model_srcs)))
    return md
