# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Render a fabric's SystemVerilog wrapper over the vendored pulp IP and
elaborate it to plain Verilog with yosys-slang."""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .amba import ports_for
from .spec import LATENCY_MODES, FabricSpec
from .tb_template import render_testbench

_HERE = Path(__file__).resolve().parent
SV_ROOT = _HERE.parent / "langgraph" / "rtl_lib" / "fabric" / "sv"
DEFINES = ("SYNTHESIS", "COMMON_CELLS_ASSERTS_OFF", "VERILATOR", "TARGET_SYNTHESIS")
# Elaboration order matters for packages; the rest is module-level.
COMMON_CELLS = ["cf_math_pkg", "rr_arb_tree", "lzc", "onehot_to_bin", "spill_register_flushable",
                "spill_register", "stream_register", "fifo_v3", "counter", "delta_counter",
                "addr_decode_dync", "addr_decode", "id_queue", "stream_fork", "stream_fork_dynamic",
                "stream_join", "stream_join_dynamic", "stream_mux", "stream_demux", "stream_arbiter",
                "stream_arbiter_flushable", "fall_through_register", "fifo_v2"]
AXI = ["axi_pkg", "axi_demux_simple", "axi_demux", "axi_mux", "axi_err_slv", "axi_id_prepend",
       "axi_xbar_unmuxed", "axi_xbar", "axi_atop_filter", "axi_cut", "axi_multicut", "axi_lite_xbar",
       "axi_lite_mux", "axi_lite_demux", "axi_to_axi_lite", "axi_lite_to_apb", "axi_id_remap",
       "axi_burst_splitter", "axi_burst_splitter_gran", "axi_dw_converter", "axi_dw_downsizer",
       "axi_dw_upsizer"]


@dataclass
class FabricArtifacts:
    module: str
    rtl_path: str
    sv_path: str
    tb_path: str
    digest: str
    ports: list[dict] = field(default_factory=list)
    cached: bool = False


def vendored_sources() -> list[Path]:
    return ([SV_ROOT / "common_cells" / f"{f}.sv" for f in COMMON_CELLS]
            + [SV_ROOT / "axi" / f"{f}.sv" for f in AXI])


def yosys_binary() -> str:
    return (os.environ.get("CORESMITH_FABRIC_YOSYS") or os.environ.get("CORESMITH_BACKEND_YOSYS")
            or shutil.which("yosys") or "yosys")


def slang_available(yosys: str | None = None) -> bool:
    """Whether this Yosys can load the slang frontend (oss-cad-suite ships it)."""
    yb = yosys or yosys_binary()
    if not shutil.which(yb) and not Path(yb).exists():
        return False
    try:
        p = subprocess.run([yb, "-q", "-p", "plugin -i slang"], capture_output=True, text=True,
                           timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return p.returncode == 0 and "ERROR" not in (p.stdout + p.stderr)


# --------------------------------------------------------------------------
# SystemVerilog wrapper
# --------------------------------------------------------------------------

def _flat_ports(spec: FabricSpec) -> list[dict]:
    IW, MIW = max(m.id_width for m in spec.masters), spec.mst_id_width
    out = [{"name": "clk", "dir": "input", "width": 1, "signal": "clk"},
           {"name": "rst_n", "dir": "input", "width": 1, "signal": "rst_n"}]
    for m in spec.masters:
        # the fabric is the SLAVE side of a master's bus: it drives s2m signals
        out += ports_for("axi4", f"s_{m.name}", role="slave", AW=spec.addr_width, DW=spec.data_width,
                         IW=IW, UW=spec.user_width)
    for s in spec.slaves:
        out += ports_for(s.protocol, f"m_{s.name}", role="master", AW=spec.addr_width,
                         DW=spec.data_width, IW=MIW, UW=spec.user_width)
    return out


def _emit_cut(L: list[str], j: int, kind: str = "mst", src: str | None = None,
              rsp: str | None = None) -> tuple[str, str]:
    """An axi_cut (spill register on all five channels) on slave port j.

    kind "mst" cuts the full-AXI crossbar master port; kind "lite" cuts the
    AXI-Lite side of the port's protocol converter (axi_cut only touches the
    *_valid/*_ready/channel fields, which the AXI-Lite structs share).
    """
    src, rsp = src or f"mst_req[{j}]", rsp or f"mst_rsp[{j}]"
    sfx = "" if kind == "mst" else f"_{kind}"
    L.append(f"  {kind}_req_t cut{sfx}_req_{j}; {kind}_resp_t cut{sfx}_rsp_{j};")
    L.append(f"  axi_cut #(.Bypass(1'b0), .aw_chan_t({kind}_aw_chan_t), .w_chan_t({kind}_w_chan_t), "
             f".b_chan_t({kind}_b_chan_t), .ar_chan_t({kind}_ar_chan_t), .r_chan_t({kind}_r_chan_t), "
             f".axi_req_t({kind}_req_t), .axi_resp_t({kind}_resp_t)) i_cut{sfx}_{j} (.clk_i(clk), .rst_ni(rst_n), "
             f".slv_req_i({src}), .slv_resp_o({rsp}), .mst_req_o(cut{sfx}_req_{j}), .mst_resp_i(cut{sfx}_rsp_{j}));")
    return f"cut{sfx}_req_{j}", f"cut{sfx}_rsp_{j}"


def render_wrapper_sv(spec: FabricSpec) -> str:
    errs = spec.validate()
    if errs:
        raise ValueError("invalid FabricSpec: " + "; ".join(errs))
    AW, DW, UW = spec.addr_width, spec.data_width, spec.user_width
    IW, MIW = max(m.id_width for m in spec.masters), spec.mst_id_width
    NM, NS = len(spec.masters), len(spec.slaves)
    ports = _flat_ports(spec)
    L: list[str] = []
    L.append(f"// {spec.module_name}: generated SoC fabric (B1). {NM} master(s) x {NS} slave(s),")
    L.append(f"// AXI4 {AW}-bit address / {DW}-bit data, master id width {IW}, slave-port id width {MIW}.")
    L.append(f"// Crossbar latency {LATENCY_MODES[spec.latency_mode]}; per-slave-port axi_cut "
             f"{'on' if spec.slave_cut else 'off'}.")
    L.append("// Rendered by orchestrator/fabric/generate.py over the vendored pulp-platform axi IP")
    L.append("// (SHL-0.51); elaborated to plain Verilog by yosys-slang. Do not edit.")
    L.append('`include "axi/typedef.svh"')
    L.append(f"module {spec.module_name} (")
    L.append(",\n".join(f"  {p['dir']:<6} logic " + (f"[{p['width'] - 1}:0] " if p["width"] > 1 else "") + p["name"]
                        for p in ports))
    L.append(");")
    L.append(f"  localparam int unsigned AW = {AW}, DW = {DW}, IW = {IW}, MIW = {MIW}, UW = {UW};")
    L.append("  typedef logic [AW-1:0] addr_t; typedef logic [DW-1:0] data_t; typedef logic [DW/8-1:0] strb_t;")
    L.append("  typedef logic [IW-1:0] id_t; typedef logic [MIW-1:0] mid_t; typedef logic [UW-1:0] user_t;")
    L.append("  `AXI_TYPEDEF_ALL(slv, addr_t, id_t, data_t, strb_t, user_t)")
    L.append("  `AXI_TYPEDEF_ALL(mst, addr_t, mid_t, data_t, strb_t, user_t)")
    L.append("  `AXI_LITE_TYPEDEF_ALL(lite, addr_t, data_t, strb_t)")
    L.append("  typedef struct packed { int unsigned idx; addr_t start_addr; addr_t end_addr; } rule_t;")
    L.append("  localparam axi_pkg::xbar_cfg_t Cfg = '{")
    L.append(f"    NoSlvPorts: {NM}, NoMstPorts: {NS}, MaxMstTrans: {spec.max_outstanding}, "
             f"MaxSlvTrans: {max(m.max_outstanding for m in spec.masters)}, FallThrough: 1'b0,")
    L.append(f"    LatencyMode: axi_pkg::{LATENCY_MODES[spec.latency_mode]}, PipelineStages: {int(spec.pipeline_stages)}, "
             "AxiIdWidthSlvPorts: IW, AxiIdUsedSlvPorts: IW,")
    L.append(f"    UniqueIds: 1'b{int(bool(spec.unique_ids))}, AxiAddrWidth: AW, AxiDataWidth: DW, NoAddrRules: {NS} }};")
    rules = ", ".join(f"'{{idx: {i}, start_addr: {AW}'h{s.base:X}, end_addr: {AW}'h{s.base + s.size:X}}}"
                      for i, s in reversed(list(enumerate(spec.slaves))))
    L.append(f"  localparam rule_t [{NS - 1}:0] AddrMap = '{{ {rules} }};")
    L.append(f"  slv_req_t  [{NM - 1}:0] slv_req;  slv_resp_t [{NM - 1}:0] slv_rsp;")
    L.append(f"  mst_req_t  [{NS - 1}:0] mst_req;  mst_resp_t [{NS - 1}:0] mst_rsp;")
    # master ports -> structs
    for i, m in enumerate(spec.masters):
        p = f"s_{m.name}_"
        L.append(f"  // master port {m.name}")
        L.append(f"  assign slv_req[{i}].aw_valid = {p}awvalid; assign {p}awready = slv_rsp[{i}].aw_ready;")
        L.append(f"  assign slv_req[{i}].aw = '{{id: {p}awid, addr: {p}awaddr, len: {p}awlen, size: {p}awsize, "
                 f"burst: {p}awburst, lock: {p}awlock, cache: {p}awcache, prot: {p}awprot, qos: {p}awqos, "
                 f"region: {p}awregion, atop: '0, user: {p}awuser}};")
        L.append(f"  assign slv_req[{i}].w_valid = {p}wvalid; assign {p}wready = slv_rsp[{i}].w_ready;")
        L.append(f"  assign slv_req[{i}].w = '{{data: {p}wdata, strb: {p}wstrb, last: {p}wlast, user: {p}wuser}};")
        L.append(f"  assign {p}bvalid = slv_rsp[{i}].b_valid; assign slv_req[{i}].b_ready = {p}bready;")
        L.append(f"  assign {p}bid = slv_rsp[{i}].b.id; assign {p}bresp = slv_rsp[{i}].b.resp; assign {p}buser = slv_rsp[{i}].b.user;")
        L.append(f"  assign slv_req[{i}].ar_valid = {p}arvalid; assign {p}arready = slv_rsp[{i}].ar_ready;")
        L.append(f"  assign slv_req[{i}].ar = '{{id: {p}arid, addr: {p}araddr, len: {p}arlen, size: {p}arsize, "
                 f"burst: {p}arburst, lock: {p}arlock, cache: {p}arcache, prot: {p}arprot, qos: {p}arqos, "
                 f"region: {p}arregion, user: {p}aruser}};")
        L.append(f"  assign {p}rvalid = slv_rsp[{i}].r_valid; assign slv_req[{i}].r_ready = {p}rready;")
        L.append(f"  assign {p}rdata = slv_rsp[{i}].r.data; assign {p}rid = slv_rsp[{i}].r.id; "
                 f"assign {p}rresp = slv_rsp[{i}].r.resp; assign {p}rlast = slv_rsp[{i}].r.last; assign {p}ruser = slv_rsp[{i}].r.user;")
    L.append("  axi_xbar #(.Cfg(Cfg), .ATOPs(1'b0), .slv_aw_chan_t(slv_aw_chan_t), .mst_aw_chan_t(mst_aw_chan_t),")
    L.append("    .w_chan_t(slv_w_chan_t), .slv_b_chan_t(slv_b_chan_t), .mst_b_chan_t(mst_b_chan_t),")
    L.append("    .slv_ar_chan_t(slv_ar_chan_t), .mst_ar_chan_t(mst_ar_chan_t), .slv_r_chan_t(slv_r_chan_t),")
    L.append("    .mst_r_chan_t(mst_r_chan_t), .slv_req_t(slv_req_t), .slv_resp_t(slv_resp_t),")
    L.append("    .mst_req_t(mst_req_t), .mst_resp_t(mst_resp_t), .rule_t(rule_t)) i_xbar (")
    L.append("    .clk_i(clk), .rst_ni(rst_n), .test_i(1'b0), .slv_ports_req_i(slv_req), .slv_ports_resp_o(slv_rsp),")
    L.append("    .mst_ports_req_o(mst_req), .mst_ports_resp_i(mst_rsp), .addr_map_i(AddrMap),")
    L.append("    .en_default_mst_port_i('0), .default_mst_port_i('0));")
    # slave ports
    for j, s in enumerate(spec.slaves):
        p = f"m_{s.name}_"
        L.append(f"  // slave port {s.name} ({s.protocol}) @ {s.base:#x} +{s.size:#x}")
        if s.protocol == "axi4":
            src = f"mst_req[{j}]"
            rsp = f"mst_rsp[{j}]"
            if spec.ordering == "none" or spec.slave_cut:
                src, rsp = _emit_cut(L, j)
            L.append(f"  assign {p}awvalid = {src}.aw_valid; assign {rsp}.aw_ready = {p}awready;")
            L.append(f"  assign {p}awaddr = {src}.aw.addr; assign {p}awid = {src}.aw.id; assign {p}awlen = {src}.aw.len; "
                     f"assign {p}awsize = {src}.aw.size; assign {p}awburst = {src}.aw.burst; assign {p}awlock = {src}.aw.lock; "
                     f"assign {p}awcache = {src}.aw.cache; assign {p}awprot = {src}.aw.prot; assign {p}awqos = {src}.aw.qos; "
                     f"assign {p}awregion = {src}.aw.region; assign {p}awuser = {src}.aw.user;")
            L.append(f"  assign {p}wvalid = {src}.w_valid; assign {rsp}.w_ready = {p}wready; assign {p}wdata = {src}.w.data; "
                     f"assign {p}wstrb = {src}.w.strb; assign {p}wlast = {src}.w.last; assign {p}wuser = {src}.w.user;")
            L.append(f"  assign {rsp}.b_valid = {p}bvalid; assign {p}bready = {src}.b_ready; "
                     f"assign {rsp}.b = '{{id: {p}bid, resp: {p}bresp, user: {p}buser}};")
            L.append(f"  assign {p}arvalid = {src}.ar_valid; assign {rsp}.ar_ready = {p}arready;")
            L.append(f"  assign {p}araddr = {src}.ar.addr; assign {p}arid = {src}.ar.id; assign {p}arlen = {src}.ar.len; "
                     f"assign {p}arsize = {src}.ar.size; assign {p}arburst = {src}.ar.burst; assign {p}arlock = {src}.ar.lock; "
                     f"assign {p}arcache = {src}.ar.cache; assign {p}arprot = {src}.ar.prot; assign {p}arqos = {src}.ar.qos; "
                     f"assign {p}arregion = {src}.ar.region; assign {p}aruser = {src}.ar.user;")
            L.append(f"  assign {rsp}.r_valid = {p}rvalid; assign {p}rready = {src}.r_ready; "
                     f"assign {rsp}.r = '{{id: {p}rid, data: {p}rdata, resp: {p}rresp, last: {p}rlast, user: {p}ruser}};")
        else:
            # slave_cut registers both sides of the protocol converters: a full
            # register slice between the crossbar and axi_to_axi_lite (so the
            # converter's response logic does not chain into the crossbar's R/B
            # muxes; axi4 ports get the same cut above), an AXI-Lite slice on
            # its other side (so the lite handshakes, e.g. bready, leave from a
            # flop) and, for APB, axi_lite_to_apb's own request/response
            # spill registers instead of its fall-through registers.
            src, rsp = _emit_cut(L, j) if spec.slave_cut else (f"mst_req[{j}]", f"mst_rsp[{j}]")
            L.append(f"  lite_req_t lite_req_{j}; lite_resp_t lite_rsp_{j};")
            L.append("  axi_to_axi_lite #(.AxiAddrWidth(AW), .AxiDataWidth(DW), .AxiIdWidth(MIW), .AxiUserWidth(UW),")
            L.append(f"    .AxiMaxWriteTxns({spec.max_outstanding}), .AxiMaxReadTxns({spec.max_outstanding}), .FallThrough(1'b0),")
            L.append("    .full_req_t(mst_req_t), .full_resp_t(mst_resp_t), .lite_req_t(lite_req_t), .lite_resp_t(lite_resp_t))")
            L.append(f"    i_to_lite_{j} (.clk_i(clk), .rst_ni(rst_n), .test_i(1'b0), .slv_req_i({src}), "
                     f".slv_resp_o({rsp}), .mst_req_o(lite_req_{j}), .mst_resp_i(lite_rsp_{j}));")
            lq, lr = (_emit_cut(L, j, "lite", f"lite_req_{j}", f"lite_rsp_{j}") if spec.slave_cut
                      else (f"lite_req_{j}", f"lite_rsp_{j}"))
            if s.protocol == "axi_lite":
                L.append(f"  assign {p}awvalid = {lq}.aw_valid; assign {lr}.aw_ready = {p}awready; "
                         f"assign {p}awaddr = {lq}.aw.addr; assign {p}awprot = {lq}.aw.prot;")
                L.append(f"  assign {p}wvalid = {lq}.w_valid; assign {lr}.w_ready = {p}wready; "
                         f"assign {p}wdata = {lq}.w.data; assign {p}wstrb = {lq}.w.strb;")
                L.append(f"  assign {lr}.b_valid = {p}bvalid; assign {p}bready = {lq}.b_ready; "
                         f"assign {lr}.b = '{{resp: {p}bresp}};")
                L.append(f"  assign {p}arvalid = {lq}.ar_valid; assign {lr}.ar_ready = {p}arready; "
                         f"assign {p}araddr = {lq}.ar.addr; assign {p}arprot = {lq}.ar.prot;")
                L.append(f"  assign {lr}.r_valid = {p}rvalid; assign {p}rready = {lq}.r_ready; "
                         f"assign {lr}.r = '{{data: {p}rdata, resp: {p}rresp}};")
            else:  # apb
                # The APB struct types are module-scoped: emit them once, at the
                # first APB slave (a second typedef is a redefinition in Verilator).
                if j == spec.slave_index(next(x.name for x in spec.slaves if x.protocol == "apb")):
                    L.append("  typedef struct packed { addr_t paddr; logic [2:0] pprot; logic psel; logic penable; "
                             "logic pwrite; data_t pwdata; strb_t pstrb; } apb_req_t;")
                    L.append("  typedef struct packed { logic pready; data_t prdata; logic pslverr; } apb_resp_t;")
                pipe = "1'b1" if spec.slave_cut else "1'b0"
                L.append(f"  apb_req_t apb_req_{j}; apb_resp_t apb_rsp_{j};")
                L.append(f"  localparam rule_t [0:0] ApbMap_{j} = '{{ '{{idx: 0, start_addr: {AW}'h{s.base:X}, "
                         f"end_addr: {AW}'h{s.base + s.size:X}}} }};")
                L.append("  axi_lite_to_apb #(.NoApbSlaves(1), .NoRules(1), .AddrWidth(AW), .DataWidth(DW), "
                         f".PipelineRequest({pipe}), .PipelineResponse({pipe}), .axi_lite_req_t(lite_req_t), "
                         ".axi_lite_resp_t(lite_resp_t), .apb_req_t(apb_req_t), .apb_resp_t(apb_resp_t), .rule_t(rule_t))")
                L.append(f"    i_to_apb_{j} (.clk_i(clk), .rst_ni(rst_n), .axi_lite_req_i({lq}), "
                         f".axi_lite_resp_o({lr}), .apb_req_o(apb_req_{j}), .apb_resp_i(apb_rsp_{j}), .addr_map_i(ApbMap_{j}));")
                L.append(f"  assign {p}psel = apb_req_{j}.psel; assign {p}penable = apb_req_{j}.penable; "
                         f"assign {p}pwrite = apb_req_{j}.pwrite; assign {p}paddr = apb_req_{j}.paddr; "
                         f"assign {p}pprot = apb_req_{j}.pprot; assign {p}pwdata = apb_req_{j}.pwdata; assign {p}pstrb = apb_req_{j}.pstrb;")
                L.append(f"  assign apb_rsp_{j} = '{{pready: {p}pready, prdata: {p}prdata, pslverr: {p}pslverr}};")
    L.append("endmodule")
    return "\n".join(x for x in L if x is not None) + "\n"


# --------------------------------------------------------------------------
# Elaboration
# --------------------------------------------------------------------------

def elaborate(spec: FabricSpec, out_dir, *, yosys: str | None = None, timeout_s: int = 1800) -> Path:
    """Write ``<module>.sv`` and elaborate it to ``<module>.v`` (plain Verilog).

    Cached on the spec digest: the same spec never re-elaborates.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sv = out / f"{spec.module_name}.sv"
    v = out / f"{spec.module_name}.v"
    sv.write_text(render_wrapper_sv(spec))
    stamp = out / f".{spec.module_name}.digest"
    key = spec.digest() + ":" + hashlib.sha256(sv.read_bytes()).hexdigest()[:16]
    if v.exists() and stamp.exists() and stamp.read_text().strip() == key:
        return v
    yb = yosys or yosys_binary()
    if not slang_available(yb):
        raise RuntimeError(f"yosys-slang is not available in {yb}: the fabric generator needs "
                           "oss-cad-suite's Yosys (plugin slang) -- set CORESMITH_FABRIC_YOSYS")
    defs = " ".join(f"-D{d}" for d in DEFINES)
    srcs = " ".join(str(p) for p in vendored_sources()) + f" {sv}"
    script = (f"plugin -i slang; read_slang {defs} -I {SV_ROOT / 'axi' / 'include'} "
              f"-I {SV_ROOT / 'common_cells' / 'include'} "
              f"--top {spec.module_name} {srcs}; hierarchy -top {spec.module_name}; proc; "
              f"bwmuxmap; opt_clean; write_verilog -noattr {v}")
    log = out / f"{spec.module_name}.slang.log"
    p = subprocess.run([yb, "-q", "-l", str(log), "-p", script], capture_output=True, text=True,
                       timeout=timeout_s)
    if p.returncode != 0 or not v.exists():
        raise RuntimeError("fabric elaboration failed:\n" + (p.stdout + p.stderr)[-4000:])
    header = (f"// {spec.module_name}: plain-Verilog elaboration of {sv.name} (yosys-slang). GENERATED, "
              f"do not edit. Sources: pulp-platform axi/common_cells (SHL-0.51), see "
              f"rtl_lib/fabric/sv/MANIFEST.json.\n")
    v.write_text(header + v.read_text())
    stamp.write_text(key)
    return v


def generate_fabric(spec: FabricSpec, out_dir, *, tb_dir=None, yosys: str | None = None) -> FabricArtifacts:
    """Elaborate the fabric and write its cocotb testbench."""
    out = Path(out_dir)
    stamp = out / f".{spec.module_name}.digest"
    before = stamp.read_text().strip() if stamp.exists() else None
    v = elaborate(spec, out, yosys=yosys)
    tbd = Path(tb_dir) if tb_dir else out
    tbd.mkdir(parents=True, exist_ok=True)
    tb = tbd / f"test_{spec.module_name}.py"
    tb.write_text(render_testbench(spec))
    return FabricArtifacts(module=spec.module_name, rtl_path=str(v), sv_path=str(out / f"{spec.module_name}.sv"),
                           tb_path=str(tb), digest=spec.digest(), ports=_flat_ports(spec),
                           cached=(before is not None and stamp.read_text().strip() == before))
