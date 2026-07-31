// Copyright (c) Meta Platforms, Inc. and affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
//
// ===========================================================================
// CoreSmith UNIFIED memory primitive: cs_mem_1rw / cs_mem_1rw1r
// ===========================================================================
// The implementation is selected by a `parameter MEM_IMPL` and a synthesizable
// `generate` block -- NOT by a strippable global `ifdef`. This is deliberate:
//
//   * "BEHAV" (default) and "FLOP" share ONE synchronous behavioral body
//     (write-first port0; for 1rw1r the port1 read sees the port0 write when
//     addr0==addr1 && we0; the read is captured into an output flop, so every
//     impl has IDENTICAL 1-cycle registered-read latency). The only difference
//     is intent/synth-policy: "FLOP" is the blessed flop tier (always flops),
//     "BEHAV" is the macro-backed wrapper's simulation model.
//
//   * "MACRO" instantiates a SEPARATELY-NAMED empty leaf, `cs_mem_macro_shell`,
//     that the CoreSmith PD macro flow (`macro_backend` / `openram_gen`) binds
//     to a pre-built sky130 / OpenRAM SRAM of matching geometry. Because the
//     empty cell has its OWN distinct name, a stripped/empty/blackbox cell can
//     NEVER masquerade as the whole `cs_mem_*` memory: if the behavioral body
//     were ever dropped, elaboration would still find the real `cs_mem_*`
//     module body here (the generate just picks BEHAV), not an empty stub.
//
// WHY a generate, not `ifdef CORESMITH_SRAM_SYNTH`: an ifdef-selected body is a
// single global switch that (a) a source deduper can strip wholesale (the
// behavioral body lives behind an inactive define, emitting nothing at lint),
// and (b) lets an LLM author its own empty blackbox of the same name that a
// first-wins deduper then locks in -- the exact defect that made every
// SRAM-backed block read all-zero in DV. A per-instance `MEM_IMPL` parameter
// keeps the behavioral body ALWAYS present in the source and selectable.
//
// cs_sram_* / cs_fpmem_* below are kept as THIN pass-throughs to cs_mem_* with
// a fixed MEM_IMPL, so already-generated RTL that instantiates them still
// elaborates unchanged.
// ===========================================================================

// Empty leaf bound by the PD macro flow. Intentionally has NO behavior and a
// UNIQUE name so it can never stand in for a whole cs_mem_* memory.
module cs_mem_macro_shell #(
    parameter integer WIDTH = 32,
    parameter integer DEPTH = 512,
    parameter integer NPORT = 1,                 // 1 = 1rw, 2 = 1rw1r
    parameter integer AW    = (DEPTH <= 1) ? 1 : $clog2(DEPTH)
) (
    input  wire             clk,
    // port 0 (read/write)
    input  wire             ce0,
    input  wire             we0,
    input  wire [AW-1:0]    addr0,
    input  wire [WIDTH-1:0] wdata0,
    output wire [WIDTH-1:0] rdata0,
    // port 1 (read-only; tied off / unused when NPORT==1)
    input  wire             ce1,
    input  wire [AW-1:0]    addr1,
    output wire [WIDTH-1:0] rdata1
);
    // Black box: the macro flow resolves WIDTH/DEPTH/NPORT to a real SRAM and
    // streams in its GDS/LEF/lib. No internal logic -> 0 flip-flops at synth.
    // (Reads are 0 in pure RTL sim -- "MACRO" is a backend impl, not a sim
    // model; use "BEHAV"/"FLOP" for simulation.)
    /* verilator lint_off UNDRIVEN */
    /* verilator lint_off UNUSED */
    assign rdata0 = {WIDTH{1'b0}};
    assign rdata1 = {WIDTH{1'b0}};
    /* verilator lint_on UNUSED */
    /* verilator lint_on UNDRIVEN */
endmodule

// ---------------------------------------------------------------------------
// Unified single-port (1RW) synchronous memory. Impl chosen by MEM_IMPL.
// ---------------------------------------------------------------------------
module cs_mem_1rw #(
    parameter            MEM_IMPL = "BEHAV",     // "BEHAV" | "FLOP" | "MACRO"
    parameter integer    WIDTH    = 32,
    parameter integer    DEPTH    = 512,
    parameter integer    AW       = (DEPTH <= 1) ? 1 : $clog2(DEPTH)
) (
    input  wire             clk,
    input  wire             ce,    // chip enable (gates read & write)
    input  wire             we,    // write enable (with ce)
    input  wire [AW-1:0]    addr,
    input  wire [WIDTH-1:0] wdata,
    output wire [WIDTH-1:0] rdata
);
    generate
        // verilator lint_off WIDTHEXPAND
        if (MEM_IMPL == "MACRO") begin : g_macro
        // verilator lint_on WIDTHEXPAND
            // Separately-named empty shell bound by the PD macro flow.
            cs_mem_macro_shell #(.WIDTH(WIDTH), .DEPTH(DEPTH), .NPORT(1)) u_shell (
                .clk(clk),
                .ce0(ce), .we0(we), .addr0(addr), .wdata0(wdata), .rdata0(rdata),
                .ce1(1'b0), .addr1({AW{1'b0}}), .rdata1()
            );
        end else begin : g_behav
            // Shared synchronous, write-first behavioral body for BEHAV + FLOP.
            // verilator lint_off MULTIDRIVEN
            reg [WIDTH-1:0] mem [0:DEPTH-1];
            // verilator lint_on MULTIDRIVEN
            reg [WIDTH-1:0] rdata_q;
            always @(posedge clk) begin
                if (ce) begin
                    if (we) mem[addr] <= wdata;             // 1 write port
                    rdata_q <= we ? wdata : mem[addr];      // write-first; captured
                end
            end
            assign rdata = rdata_q;
        end
    endgenerate
endmodule

// ---------------------------------------------------------------------------
// Unified one read-write + one read-only port (1RW1R) memory.
// ---------------------------------------------------------------------------
module cs_mem_1rw1r #(
    parameter            MEM_IMPL = "BEHAV",     // "BEHAV" | "FLOP" | "MACRO"
    parameter integer    WIDTH    = 32,
    parameter integer    DEPTH    = 512,
    parameter integer    AW       = (DEPTH <= 1) ? 1 : $clog2(DEPTH),
    parameter integer    READ_FIRST = 0,
    parameter            INIT_FILE = "",
    // Optional write mask on port 0. USE_WMASK=0 (default) is the legacy
    // full-word write and NEVER samples wmask0 (safe unconnected).
    // USE_WMASK=1: wmask0 is an active-high per-LANE enable -- a single-cycle
    // write that updates only the selected lanes and preserves the rest (no
    // RMW cycle). WMASK_GRAN sets the bits per mask lane: 8 = byte lanes
    // (matches the byte-write-enable of the sky130 1rw1r macro family; the
    // cs_sram tier), 1 = per-bit enables (flops have individual enables; the
    // cs_fpmem tier).
    parameter integer    USE_WMASK  = 0,
    parameter integer    WMASK_GRAN = 8,
    parameter integer    NB = (WIDTH + WMASK_GRAN - 1) / WMASK_GRAN
) (
    input  wire             clk,
    // read/write port
    input  wire             ce0,
    input  wire             we0,
    input  wire [AW-1:0]    addr0,
    input  wire [WIDTH-1:0] wdata0,
    /* verilator lint_off UNUSEDSIGNAL */
    input  wire [NB-1:0]    wmask0,  // sampled only when USE_WMASK=1
    /* verilator lint_on UNUSEDSIGNAL */
    output wire [WIDTH-1:0] rdata0,
    // read-only port
    input  wire             ce1,
    input  wire [AW-1:0]    addr1,
    output wire [WIDTH-1:0] rdata1
);
    generate
        // verilator lint_off WIDTHEXPAND
        // String-parameter compares warn when an instantiation overrides
        // MEM_IMPL with a string of a different length ("SRAM" = 32 bits vs
        // "MACRO" = 40) -- fatal under Verilator >=5.05 defaults; killed the
        // armD integration build (and earlier block rebuilds). Elaboration-
        // time constant compare; width expansion is exactly what we want.
        if (MEM_IMPL == "MACRO") begin : g_macro
        // verilator lint_on WIDTHEXPAND
            // NOTE (USE_WMASK): the black-box shell carries no mask pins; the
            // macro flow must resolve a USE_WMASK=1 geometry to a byte-write-
            // enable macro (e.g. the sky130 ..._1rw1r_32x512_8 family, wmask
            // granularity 8) or an OpenRAM config with write_size=8.
            cs_mem_macro_shell #(.WIDTH(WIDTH), .DEPTH(DEPTH), .NPORT(2)) u_shell (
                .clk(clk),
                .ce0(ce0), .we0(we0), .addr0(addr0), .wdata0(wdata0), .rdata0(rdata0),
                .ce1(ce1), .addr1(addr1), .rdata1(rdata1)
            );
        end else begin : g_behav
            // Shared synchronous behavioral body for BEHAV + FLOP. Matches the
            // existing cs_fpmem_1rw1r semantics so latency is identical across
            // impls: write-first port0, port1 read-bypasses the port0 write when
            // addr0==addr1 && we0, both reads captured (1-cycle latency).
            // verilator lint_off MULTIDRIVEN
            reg [WIDTH-1:0] mem [0:DEPTH-1];
            // verilator lint_on MULTIDRIVEN
            reg [WIDTH-1:0] rdata0_q;
            reg [WIDTH-1:0] rdata1_q;
            /* Immutable-ROM images are loaded only in this blessed wrapper. */
            initial begin
                if (INIT_FILE != "") begin
                    $readmemh(INIT_FILE, mem);
                end
            end
            // Effective byte-lane enable. A generate-if (not a ternary) so the
            // legacy USE_WMASK=0 configuration NEVER samples wmask0 -- an
            // unconnected mask port on an existing instantiation stays inert.
            wire [NB-1:0] wmask0_eff;
            if (USE_WMASK != 0) begin : g_wm
                assign wmask0_eff = wmask0;
            end else begin : g_nwm
                assign wmask0_eff = {NB{1'b1}};
            end
            // Bit-expanded lane mask + merged write view. With all lanes
            // enabled wr_merged == wdata0, so USE_WMASK=0 is bit-for-bit the
            // legacy behavior; with a partial mask the unselected byte lanes
            // preserve the stored word in the SAME single-cycle write (byte
            // enables, not read-modify-write). Both the write-first capture
            // and the port1 same-address bypass observe the merged word.
            genvar gb;
            wire [WIDTH-1:0] lane_mask;
            for (gb = 0; gb < WIDTH; gb = gb + 1) begin : g_lm
                assign lane_mask[gb] = wmask0_eff[gb / WMASK_GRAN];
            end
            wire [WIDTH-1:0] wr_merged =
                (mem[addr0] & ~lane_mask) | (wdata0 & lane_mask);
            always @(posedge clk) begin
                if (ce0) begin
                    if (we0) mem[addr0] <= wr_merged;
                    rdata0_q <= we0 ? wr_merged : mem[addr0];  // write-first; captured
                end
                if (ce1) begin
                    // Optional read-first mode suppresses same-edge bypass.
                    rdata1_q <= ((READ_FIRST == 0) && we0 && ce0 &&
                                 (addr0 == addr1)) ? wr_merged : mem[addr1];
                end
            end
            assign rdata0 = rdata0_q;
            assign rdata1 = rdata1_q;
        end
    endgenerate
endmodule

// CoreSmith generic SRAM wrappers.
//
// A block that needs on-chip storage above the flop threshold MUST instantiate
// one of these parametrized wrappers instead of writing a raw `reg [..] mem [..]`
// array.  The same source is:
//   * BEHAVIORAL in simulation  (cocotb/verilator) -- the `else` branch below,
//     so DV runs with no macro collateral, and
//   * an SRAM MACRO under synthesis/backend -- when `CORESMITH_SRAM_SYNTH` is
//     defined the body is empty (a black box); the CoreSmith macro flow
//     (`macro_backend` / `openram_gen`) resolves each instance's WIDTH/DEPTH to
//     a pre-built sky130 SRAM (or an OpenRAM-generated one) and streams in its
//     GDS/LEF/lib.  Block-level synth therefore measures ~0 storage flip-flops
//     for a properly wrapped memory; an unwrapped reg-array does not.
//
// Single source of truth: this file is appended to both the cocotb
// VERILOG_SOURCES and the block synth `read_verilog`, so every block sees the
// same wrapper without copy-pasting it into each netlist.

// ---------------------------------------------------------------------------
// Single-port (1 read OR 1 write per cycle) synchronous SRAM.
// ---------------------------------------------------------------------------
// Back-compat pass-through to cs_mem_1rw. Behavioral in sim; under the macro
// flow the geometry is resolved to an SRAM macro. MEM_IMPL DEFAULTS to "BEHAV"
// so the behavioral body is ALWAYS present in source (the deduper can never
// strip it to an empty cell) and every already-generated instantiation (no
// MEM_IMPL override) elaborates + simulates byte-for-byte as before. The
// wrapper is now SWAPPABLE: the backend/PD synth path overrides MEM_IMPL to
// "MACRO" (yosys `chparam`, see sram_wrapper.backend_sram_macro_directive),
// which selects the `cs_mem_macro_shell` leaf (0 storage flops); simulation
// keeps the default "BEHAV". This is a per-instance PARAMETER, never a
// strippable global ifdef.
module cs_sram_1rw #(
    parameter            MEM_IMPL = "BEHAV",     // "BEHAV" | "FLOP" | "MACRO"
    parameter integer    WIDTH = 32,
    parameter integer    DEPTH = 512,
    parameter integer    AW    = (DEPTH <= 1) ? 1 : $clog2(DEPTH)
) (
    input  wire             clk,
    input  wire             ce,    // chip enable (gates read & write)
    input  wire             we,    // write enable (with ce)
    input  wire [AW-1:0]    addr,
    input  wire [WIDTH-1:0] wdata,
    output wire [WIDTH-1:0] rdata
);
    cs_mem_1rw #(.MEM_IMPL(MEM_IMPL), .WIDTH(WIDTH), .DEPTH(DEPTH)) u_mem (
        .clk(clk), .ce(ce), .we(we), .addr(addr), .wdata(wdata), .rdata(rdata)
    );
endmodule

// ---------------------------------------------------------------------------
// One read-write port + one read-only port (1RW1R) synchronous SRAM.
// Matches the sky130 `1rw1r` pre-built and the OpenRAM `1rw1r` geometry.
// ---------------------------------------------------------------------------
// Back-compat pass-through to cs_mem_1rw1r. MEM_IMPL DEFAULTS to "BEHAV" (the
// behavioral body is always present in source and every existing instantiation
// with no override is byte-for-byte unchanged); the backend/PD synth path
// overrides it to "MACRO" via yosys `chparam`, selecting the cs_mem_macro_shell
// leaf (0 storage flops). Per-instance PARAMETER, never a strippable ifdef.
module cs_sram_1rw1r #(
    parameter         MEM_IMPL = "BEHAV",     // "BEHAV" | "FLOP" | "MACRO"
    parameter integer WIDTH = 32,
    parameter integer DEPTH = 512,
    parameter integer AW    = (DEPTH <= 1) ? 1 : $clog2(DEPTH),
    // READ_FIRST=1 makes a same-edge port-0 write/port-1 read collision
    // return the complete pre-write word, matching native OLD-data macros --
    // which is what real silicon does, so a design that cares should pass it
    // EXPLICITLY (every memory block in the raster design does).
    //
    // The DEFAULT stays 0 (write-first bypass) deliberately. Flipping a shared
    // library default silently changes the read semantics of every design
    // already built against it, including ones whose DV has already passed, and
    // it breaks the documented bypass contract that tb_cs_sram_wmask.v asserts.
    // A macro-faithful default is arguably the better long-term choice -- BEHAV
    // should not promise data the macro cannot deliver -- but that is a breaking
    // change to make deliberately, with the testbench updated to match, not as
    // a side effect of one design needing it.
    parameter integer READ_FIRST = 0,
    parameter         INIT_FILE = "",
    // Optional per-byte write mask on port 0 (see cs_mem_1rw1r). Legacy
    // instantiations (USE_WMASK=0, wmask0 unconnected) are unchanged.
    // USE_WMASK=1: wmask0[NB-1:0] active-high byte-lane enables -- selected
    // bytes written, unselected bytes preserved, single cycle (matches the
    // byte-write-enable of the sky130 1rw1r macro family).
    parameter integer USE_WMASK = 0,
    parameter integer NB        = (WIDTH + 7) / 8
) (
    input  wire             clk,
    // read/write port
    input  wire             ce0,
    input  wire             we0,
    input  wire [AW-1:0]    addr0,
    input  wire [WIDTH-1:0] wdata0,
    input  wire [NB-1:0]    wmask0,  // sampled only when USE_WMASK=1
    output wire [WIDTH-1:0] rdata0,
    // read-only port
    input  wire             ce1,
    input  wire [AW-1:0]    addr1,
    output wire [WIDTH-1:0] rdata1
);
    cs_mem_1rw1r #(
        .MEM_IMPL(MEM_IMPL), .WIDTH(WIDTH), .DEPTH(DEPTH),
        .READ_FIRST(READ_FIRST), .INIT_FILE(INIT_FILE),
        .USE_WMASK(USE_WMASK), .WMASK_GRAN(8)
    ) u_mem (
        .clk(clk),
        .ce0(ce0), .we0(we0), .addr0(addr0), .wdata0(wdata0), .wmask0(wmask0),
        .rdata0(rdata0),
        .ce1(ce1), .addr1(addr1), .rdata1(rdata1)
    );
endmodule

// ===========================================================================
// CoreSmith REFERENCE FLOP-BASED MEMORY (cs_fpmem) -- the blessed flop tier.
// MANDATORY for any on-chip array kept in flip-flops (below the SRAM-macro
// threshold). Storage is flops AND the read is ALWAYS captured into an output
// flop, so the read path is a clean storage-FF -> mux -> CAPTURE-FF reg-to-reg
// path (registered, 1-cycle latency, never a combinational N:1 mux chained into
// downstream logic). A raw `reg mem[]` read combinationally is FORBIDDEN -- that
// is the ~250ns/sub-10MHz read-mux + routing-congestion failure this prevents.
// Always flops in sim AND synth (NOT macro-backed -- use cs_sram_* for macros);
// latency-compatible with cs_sram_1rw1r so flop<->macro is a drop-in swap.
// ===========================================================================
// Back-compat pass-through to cs_mem_1rw (MEM_IMPL="FLOP" -- always flops, never
// macro-backed; registered read, 1-cycle latency, latency-compatible with the
// cs_sram tier so flop<->macro is a drop-in swap).
module cs_fpmem_1rw #(
    parameter integer WIDTH = 8,
    parameter integer DEPTH = 16,
    parameter integer AW    = (DEPTH <= 1) ? 1 : $clog2(DEPTH)
) (
    input  wire             clk,
    input  wire             ce,     // access enable (read & write)
    input  wire             we,     // write enable (with ce)
    input  wire [AW-1:0]    addr,
    input  wire [WIDTH-1:0] wdata,
    output wire [WIDTH-1:0] rdata   // CAPTURE FLOP -- registered read, 1-cycle latency
);
    cs_mem_1rw #(.MEM_IMPL("FLOP"), .WIDTH(WIDTH), .DEPTH(DEPTH)) u_mem (
        .clk(clk), .ce(ce), .we(we), .addr(addr), .wdata(wdata), .rdata(rdata)
    );
endmodule

// Back-compat pass-through to cs_mem_1rw1r (MEM_IMPL="FLOP").
module cs_fpmem_1rw1r #(
    parameter integer WIDTH = 8,
    parameter integer DEPTH = 16,
    parameter integer AW    = (DEPTH <= 1) ? 1 : $clog2(DEPTH),
    parameter integer READ_FIRST = 0,
    // Optional PER-BIT write mask (flop tier: flops have individual enables).
    // USE_WMASK=0 (default) is the legacy full-word write; wmask0 is never
    // sampled and may be left unconnected. USE_WMASK=1: wmask0[WIDTH-1:0]
    // active-high bit enables -- selected bits written, unselected preserved,
    // SINGLE-CYCLE commit (no 2-cycle read-modify-write state; ready/valid on
    // the write interface never needs to drop for a masked commit).
    parameter integer USE_WMASK = 0
) (
    input  wire             clk,
    // read/write port
    input  wire             ce0,
    input  wire             we0,
    input  wire [AW-1:0]    addr0,
    input  wire [WIDTH-1:0] wdata0,
    input  wire [WIDTH-1:0] wmask0,  // sampled only when USE_WMASK=1
    output wire [WIDTH-1:0] rdata0,  // CAPTURE FLOP (port 0)
    // read-only port
    input  wire             ce1,
    input  wire [AW-1:0]    addr1,
    output wire [WIDTH-1:0] rdata1   // CAPTURE FLOP (port 1)
);
    cs_mem_1rw1r #(
        .MEM_IMPL("FLOP"), .WIDTH(WIDTH), .DEPTH(DEPTH),
        .READ_FIRST(READ_FIRST), .USE_WMASK(USE_WMASK), .WMASK_GRAN(1)
    ) u_mem (
        .clk(clk),
        .ce0(ce0), .we0(we0), .addr0(addr0), .wdata0(wdata0), .wmask0(wmask0),
        .rdata0(rdata0),
        .ce1(ce1), .addr1(addr1), .rdata1(rdata1)
    );
endmodule

// ===========================================================================
// CoreSmith ROM primitive: cs_rom_1r (mask ROM / constant table)
// ===========================================================================
// A CONSTANT table (quant matrices, Huffman codebooks, header images, twiddle
// factors) MUST be a cs_rom_1r -- never a cs_sram_* with a tied-off write
// port. A tied-write SRAM (a) prices/places at SRAM bit density (~5-10x a mask
// ROM) and (b) is not fabrication-realistic: a real SRAM powers up UNKNOWN,
// so $readmemh contents exist only in simulation. The backend binds
// cs_rom_macro_shell to an OpenRAM rom_compiler mask-ROM whose contents come
// from the SAME INIT_FILE image, so the mask carries the data.
//
// INIT_FILE is a $readmemh hex image (one WIDTH-bit word per line), given as a
// path RELATIVE TO THE PROJECT ROOT (e.g. "inputs/rom_images/quant0.memh") --
// all engine sim/synth probes resolve it from there (C27).
// Timing matches cs_mem_* and the OpenRAM ROM verilog model: synchronous
// 1-cycle registered read (address sampled on posedge when ce=1).

// Empty leaf bound by the PD macro flow to the generated OpenROM macro.
// Same pattern as cs_mem_macro_shell: a UNIQUE name so an empty blackbox can
// never masquerade as the behavioral ROM.
module cs_rom_macro_shell #(
    parameter integer WIDTH = 32,
    parameter integer DEPTH = 1024,
    parameter integer AW    = (DEPTH <= 1) ? 1 : $clog2(DEPTH)
) (
    input  wire             clk,
    input  wire             ce,
    input  wire [AW-1:0]    addr,
    output wire [WIDTH-1:0] rdata
);
    // Black box: the macro flow resolves WIDTH/DEPTH/INIT_FILE to an OpenRAM
    // mask-ROM macro and streams in its GDS/LEF. No internal logic.
endmodule

module cs_rom_1r #(
    parameter            MEM_IMPL = "BEHAV",     // "BEHAV" | "MACRO"
    parameter integer    WIDTH    = 32,
    parameter integer    DEPTH    = 1024,
    parameter integer    AW       = (DEPTH <= 1) ? 1 : $clog2(DEPTH),
    parameter            INIT_FILE = ""
) (
    input  wire             clk,
    input  wire             ce,     // chip enable (gates the read)
    input  wire [AW-1:0]    addr,
    output wire [WIDTH-1:0] rdata
);
    generate
        // verilator lint_off WIDTHEXPAND
        if (MEM_IMPL == "MACRO") begin : g_macro
        // verilator lint_on WIDTHEXPAND
            cs_rom_macro_shell #(.WIDTH(WIDTH), .DEPTH(DEPTH)) u_shell (
                .clk(clk), .ce(ce), .addr(addr), .rdata(rdata)
            );
        end else begin : g_behav
            reg [WIDTH-1:0] mem [0:DEPTH-1];
            reg [WIDTH-1:0] rdata_q;
            /* Immutable-ROM images are loaded only in this blessed wrapper. */
            initial begin
                if (INIT_FILE != "") begin
                    $readmemh(INIT_FILE, mem);
                end
            end
            always @(posedge clk) begin
                if (ce) begin
                    rdata_q <= mem[addr];
                end
            end
            assign rdata = rdata_q;
        end
    endgenerate
endmodule
