# Skill: SRAM macro vs. flip-flops for on-chip memory

Whenever a block needs storage — an array, buffer, scratchpad, cache,
lookup table, FIFO, or any `mem[]`-style structure — you MUST make an
explicit, justified choice between **flip-flops** and an **SRAM macro**,
and record that decision in the uArch spec. This is not an implementation
detail left to the RTL author: pick wrong and the block either blows the
area budget or misses timing by an order of magnitude. **The chip will
still simulate and lint-pass; it will just be 5× too big and 50× too
slow.**

## Decision rule

- **Flip-flops are fine** for small register state: a handful up to
  ~64 entries — register files, status/config/CSR registers, small
  FIFOs, skid buffers, pipeline shadow registers. These are small enough
  that the read mux is cheap. **BUT a flop *memory* — any addressed array
  you index as `mem[addr]` — MUST be built with the reference flop-memory
  primitive `cs_fpmem_1rw` / `cs_fpmem_1rw1r`, never a raw
  `reg [W-1:0] mem [0:N-1]` read combinationally.** `cs_fpmem_1rw1r` supports
  an optional **per-bit write mask** (`.USE_WMASK(1)` + `wmask0[WIDTH-1:0]`,
  active-high; selected bits written, unselected preserved, SINGLE-CYCLE
  commit) — use it for masked/partial-field commits instead of a multi-cycle
  read-modify-write state that drops ready/valid on the write interface. See **"The flop tier"**
  below. (Truly random, same-cycle register state that isn't address-indexed
  — a few named CSRs, a 4-deep skid buffer — can stay plain flops.)
- **Use an SRAM macro** (`cs_sram_1rw` / `cs_sram_1rw1r`) for anything
  storage-like *that gets written at runtime*: scratchpads, caches,
  line/frame buffers, large FIFOs, runtime-loaded tables — roughly
  **≳256 words or ≳1 KiB**.
  - `cs_sram_1rw1r` supports an optional **per-byte write mask**: instantiate
    with `.USE_WMASK(1)` and drive `wmask0[(WIDTH+7)/8-1:0]` (active-high byte
    lane enables; selected bytes written, unselected preserved, single cycle —
    matches the byte-write-enable of the sky130 1rw1r macro family). Use it
    when narrow (e.g. byte-wise) host writes land in a wide word — do NOT
    split one logical wide memory into per-byte-lane macro instances just to
    get byte writes; that multiplies the macro census.
- **Use a mask ROM** (`cs_rom_1r`) for any **read-only CONSTANT table** above
  the same threshold: quantization matrices, Huffman/VLC codebooks, header
  images, twiddle/sine tables. Declare it `impl=rom` in the `# MEM` manifest
  and give the contents as `INIT_FILE` (a `$readmemh` hex image, path
  **relative to the project root**, e.g. `"inputs/rom_images/quant0.memh"`).
  The backend generates a sky130 mask ROM (OpenRAM `rom_compiler`) carrying
  those contents in the mask. **NEVER build a constant table as a `cs_sram`
  with a tied-off write port**: it prices/places at SRAM density (~5-10x a
  mask ROM per bit) and is not fabrication-realistic — a real SRAM powers up
  UNKNOWN, so the `$readmemh` contents would exist only in simulation.

```verilog
wire [31:0] q;                       // 1-cycle registered read, same as an SRAM
cs_rom_1r #(.WIDTH(32), .DEPTH(1024),
            .INIT_FILE("inputs/rom_images/quant_bank0.memh")) u_rom (
    .clk(clk), .ce(ce), .addr(addr), .rdata(q));
```
- **Depth drives the read-mux delay**, so a deep memory should *prefer* a
  macro (`cs_sram`) for density + single-cycle speed. But **a flop array is a
  perfectly fine memory** — it just has to be the registered `cs_fpmem`
  template, never a raw comb-read array. So: deep/large ⇒ **`cs_sram` preferred,
  `cs_fpmem` acceptable** (a registered deep flop array works, just at a lower
  Fmax); shallow ⇒ `cs_fpmem`. The hard rule is only that a raw
  `reg [] mem []` read combinationally is **never** allowed — wrap it.
- **Never** synthesize a multi-kilobyte or deep array as a raw flip-flop
  array. If you catch yourself writing `reg [W-1:0] mem [0:N-1]` with a
  large/deep N, stop and pick a macro.

## The flop tier — use `cs_fpmem` (registered read, capture flop). NEVER a raw array.

When a memory legitimately stays in flops (small AND shallow), you MUST
instantiate the reference primitive, not hand-roll the array:

```verilog
wire [7:0] q;                       // 1-cycle registered read (capture flop inside)
cs_fpmem_1rw #(.WIDTH(8), .DEPTH(16)) u_lut (
    .clk(clk), .ce(ce), .we(we), .addr(addr), .wdata(d), .rdata(q));
// q is valid the cycle AFTER addr/ce -- consume it in the next FSM state,
// exactly like an SRAM macro (so flop<->macro is a drop-in swap).
```

**Why this is mandatory (the failure it insures against):** a raw
`reg [W-1:0] mem [0:N-1]` read as `assign rdata = mem[addr];` (or `mem[addr]`
feeding logic) is an **unregistered N:1 combinational read mux**. For a deep
array that's a ~250 ns path (sub-10 MHz) **and** a huge-fanout net that
congests detailed routing — it lints and simulates fine and is silently fatal
at timing + P&R. `cs_fpmem` builds the **same storage flops but always captures
the read through an output flop**, so the path is a clean, bounded,
timing-analysable `storage-FF → mux → CAPTURE-FF` reg-to-reg path, with the
same 1-cycle latency as a macro. (This is the flop analogue of the
`cs_sram` rule: there is exactly one blessed way to build each tier of memory,
and a bare `reg mem[]` array is never it.)

## Why (the reasoning you must apply, not just the rule)

A flop array doesn't just cost the flops — it costs the flops **plus an
N:1 combinational read mux** to select one word out of N. That mux is a
deep combinational tree whose delay grows with N and destroys timing.
SRAM bit-cells are roughly **10× denser** than flops and give a
**clocked, single-cycle read** with no read mux at all (the address
decoder + sense amps are internal to the macro).

Concrete: a 4 KiB array built from flops ≈ **~33k flip-flops ≈ ~1 mm²**
with a **hundreds-of-ns critical path** through the read mux. The same
4 KiB as a sky130 SRAM macro is **~0.2–0.3 mm² per KiB** with
single-cycle access. Same function, ~5× the area and ~50× the cycle time
if you pick flops.

## MEASURED crossover — the numbers behind the rule (2026-07, sky130, 50 MHz)

The rule above used to rest on judgement. A 152-point characterization sweep
plus one fully signed-off generated macro now put numbers on it.

**The crossover is ~2 Kbit.** Like for like at 2048 bits, 1rw1r:

| implementation | Fmax | area |
|---|---|---|
| macro `sram_1rw1r_16_128_8_sky130` | **227.3 MHz** | 94,990 µm² |
| registered flops 8×256 | 189.4 MHz | 94,761 µm² |

At 2 Kbit the macro is already ~20% **faster at essentially identical area**.
Below that flops win on area (a macro carries fixed periphery overhead); above
it the macro wins outright, because flop Fmax collapses with size while the
macro's does not.

Measured **registered**-flop Fmax (a raw combinational-read array is far
worse):

| geometry | Fmax | | geometry | Fmax |
|---|---|---|---|---|
| 8×64 | 495.0 MHz | | 8×1024 | 59.3 MHz |
| 8×256 | 223.7 MHz | | 8×2048 | 30.5 MHz |
| 32×256 | 63.7 MHz | | 8×4096 | 16.1 MHz |
| 16×256 | 108.1 MHz | | 64×1024 | 8.0 MHz |

**43 of 76 registered-flop geometries cannot reach 50 MHz at all.**

**Total bits is what predicts the cliff** — fitted importances are bits=0.76,
impl=0.12, log_depth=0.06, depth=0.05. Width alone and depth alone do not:
32×256 and 8×1024 are both 8 Kbit and both land near 60 MHz, while 8×256 at
2 Kbit makes 223.7 MHz. **Equal bits → comparable Fmax; equal depth → wildly
different Fmax.** So reason about your array in *bits* first, and treat width
and depth as secondary structural concerns (read-mux fanout, sense-amp cost).

Two cautions when quoting macro speed. **Generated macros are much slower than
stock ones** — the stock 1kbyte/2kbyte macros give 511–558 MHz from
`minimum_period`, while a generated 16×128 gives 227 MHz; do not assume a macro
is "fast enough" without checking its own Liberty. And **read `minimum_period`,
never clock-to-output** — treating a clk→Q access arc as a legal clock period
produced ~2 GHz macro frequencies for a macro whose real limit was 558 MHz.

To generate a macro that actually passes DRC/LVS/characterization, use the
active deployment's memory-macro verb through the CLI
(`CS="${CORESMITH_CLI:-coresmith}"`; `"$CS" tool gen_macro ...`). It resolves
the memory compiler and its silent-repair recipe from the deployment; a
deployment that has no memory compiler exits 4 (skip) rather than producing a
plausible wrong answer.

## Available sky130 SRAM macros (reference one by name in the spec)

All are **1rw1r dual-port**: one read-write port (port A) + one
read-only port (port B).

| Macro | Capacity | Organization | Area |
|-------|----------|--------------|------|
| `sky130_sram_1kbyte_1rw1r_32x256_8` | 1 KiB | 256 words × 32 bits | ≈0.19 mm² |
| `sky130_sram_2kbyte_1rw1r_32x512_8` | 2 KiB | 512 words × 32 bits | ≈0.28 mm² |
| `sky130_sram_1kbyte_1rw1r_8x1024_8`  | 1 KiB | 1024 words × 8 bits | ≈0.20 mm² |

Pick the organization whose native word width / depth matches the
access pattern (e.g. a byte-addressed buffer favors the ×8 part; a
32-bit datapath favors a ×32 part).

### DO NOT TILE. Build the exact geometry instead.

If no listed part matches your geometry, **do not compose several macros**
into one logical memory — neither horizontally (concatenating `rdata`) nor
vertically (decoding high address bits to select a bank). Tiling is disabled
in the engine by default and `ensure_macro` will not return a tiled plan.

Ask for the geometry you actually need and let the deployment's memory compiler
build it (`"$CS" tool gen_macro --width W --depth D --ports 1rw1r`). A
purpose-built macro is smaller than an over-provisioned tile and arrives with its
own signed-off DRC/LVS/Liberty collateral.

Two reasons this matters, both measured:

- **Tiled timing is not modelled.** Composition Fmax is taken from the base
  macro and ignores tile count, so a 16-tile memory reports the same frequency
  as a 1-tile one. Deep tiling adds `ceil(log2(tiles))` output-mux levels that
  nothing accounts for.
- **Tiling existed only because the generator was broken.** An audit found 20
  OpenRAM launches across 11 geometries with zero successes, so tiling was the
  only way to avoid a flop array. The generator is now repaired.

**If the geometry genuinely cannot be built, that is a human decision, not a
fallback.** The engine returns no macro and escalates. Do not silently
substitute a flop array — a large one costs multiple mm² at single-digit MHz
(see the measured table above).

## uArch-spec implications you MUST record

1. **Pick a macro and map its ports.** The 1rw1r interface is roughly:
   `clk`, `addr`, `wdata`/`rdata`, `wmask` (per-byte/per-bit write
   enable), `web` (write-enable, active low), `csb` (chip-select,
   active low) on each port. Document the exact port mapping and, if
   composing multiple macros, the banking/concatenation scheme.
2. **Account for it in the design constraints.** Add the macro's
   capacity to the design's **on-chip SRAM budget
   (`max_onchip_sram_kb`)** and count it against **`max_macro_count`**.
   If the chosen macros exceed either constraint, that's a design-level
   issue to surface, not silently ignore.
3. **SRAM reads are registered — 1-cycle latency.** A read issued on
   cycle N returns `rdata` on cycle N+1. The pipeline / FSM that consumes
   the memory MUST account for this read delay. **Do not assume a
   combinational read** the way a flop array gives you — that's the most
   common bug when converting a flop array to a macro. Insert the wait
   state, pipeline stage, or read-address-ahead logic in the control FSM.

## Deterministic pre-synth gate (you cannot prompt your way past this)

A static memory-tier check runs BEFORE synthesis. It enumerates every
behavioral memory array (`reg [W-1:0] mem [0:D-1]`) and `cs_fpmem_*`/`cs_mem_*`
instantiation, computes WIDTH×DEPTH, and REJECTS — with `kind: "memory_tier"`,
without ever running yosys — any register-tier memory that is (a) over the SRAM
threshold (**≳256 words or ≳1 KiB**) and not backed by a `cs_sram_*` macro, or
(b) whose **single-cycle flat read** would exceed the clock period. The read
delay is estimated from the PDK op-delay model as `ceil(log2(DEPTH))` 2:1-mux
levels at the data width — a 1024-word flat read is ~10 mux levels and blows the
period. So a large single-cycle flop memory is caught structurally; the only
ways through are: instantiate an SRAM macro tier, **bank/pipeline the read into
log-depth stages**, or declare an **explicit multi-cycle read contract**. Do not
attempt to legalize a big single-cycle flop array — the gate rejects it before
RTL generation is even attempted.

## Port width follows the CONSUMER's access granularity

Size each memory port to the granularity of the logic that USES it, not to an
unrelated host/transport width. A byte-serial host interface (e.g. a QSPI/GPIO
register port that streams operands one byte at a time) must NOT force the
internal consumer — which reads a whole atomic record (a coefficient, a pixel, a
128-bit key) each access — through a byte-at-a-time loop. Give the internal
consumer its OWN correctly-sized read port (or a record-width mux over a
byte-addressed store) so it reads one record per access, while the host port
keeps its byte granularity for load/unload. Matching the port to the consumer's
record size is within the flop tier when the store is small; do not serialize a
record-parallel datapath down to the transport's width.

## Make the macro actually map — INSTANTIATE it, don't "infer" it

Picking a macro in the spec is not enough. The RTL must be written so the
macro is *actually* in the netlist. A behavioral `reg [W-1:0] mem [0:N-1]`
array that you *hope* synthesis turns into the macro is the trap: it will
simulate and lint fine, $mem-infer in yosys, and STILL come out as
thousands of flip-flops because the coding style blocked macro mapping.
**`$mem`-inferred is NOT the same as macro-mapped.**

### Strongly preferred: explicit instantiation

When the spec names an SRAM macro, **instantiate that module by name** with
the documented port mapping. Do not write a behavioral array and rely on
`memory_bram`/`memory_libmap` inference — sky130's OpenRAM macros do not
map cleanly through yosys inference, and you lose control over which macro
you get. Drive the macro from registered control signals and consume its
registered `dout` one cycle later. Canonical pattern:

```verilog
// Registered control, valid the cycle AFTER the access is accepted.
// dout0 is then valid the cycle after THAT -- absorb it with an FSM wait
// state (IDLE -> WAIT_SRAM -> RESP) or a pipeline bubble.
wire [31:0] rdata;
sky130_sram_1kbyte_1rw1r_32x256_8 u_mem (
    .clk0(clk), .csb0(csb_q), .web0(web_q), .wmask0(wmask_q),
    .addr0(addr_q), .din0(wdata_q), .dout0(rdata),
    .clk1(clk), .csb1(1'b1), .addr1(8'b0), .dout1()   // 2nd port tied off
);
// rdata is consumed in the RESP state, when it is valid.
```

### If a behavioral array is truly unavoidable, it MUST be macro-mappable

These are the exact coding mistakes that have blocked mapping on real runs.
Every one of them passes lint and sim and is silently fatal at backend:

1. **No `initial` / init block.** `initial mem[i] = 0;` (or an integer-loop
   zeroing block) makes the array un-mappable — real SRAM has no init.
   Reset the *control/output* registers, never the array contents.
2. **Exactly ONE write port.** Write with byte/bit enables as bare
   conditional writes — NEVER add `else mem[a] <= mem[a];` read-modify-write
   "holds". A held byte lane is counted as another write port; four lanes ×
   (write + hold) = up to 9 inferred write ports, and the macro has one.
   ```verilog
   if (we) begin                       // GOOD: 1 write port, byte-enabled
       if (wstrb[0]) mem[a][7:0]   <= din[7:0];
       if (wstrb[1]) mem[a][15:8]  <= din[15:8];
       ...                             // no else, no outer mem<=mem hold
   end
   ```
3. **A CLEAN registered read.** The read-output register may have a
   clock-enable, but it must NOT carry reset / self-hold / clear-to-zero
   muxing on the read DATA — that keeps the read port asynchronous and
   un-foldable into the macro's synchronous read. Put any
   clear-on-error/zeroing in a MUX *after* the read register, or handle it
   with a downstream valid/error qualifier, not on the read register input.

A synthesis/PPA check should treat "uArch named a macro but the netlist has
0 macro instances and a flop count near depth×width" as a FAILURE, not a
pass — `$mem` inference passing is not sufficient evidence the macro mapped.

## Cautionary example (real failed run — keep this in mind)

A fixed RV32I MCU wrote its 4 KiB data scratchpad as
`reg [31:0] ram_word [0:1023]`. Synthesis turned it into **~33k flops
plus a 1024:1 combinational read mux** → **1.23 mm²** and a **475 ns
critical path (~2 MHz)**. Re-specifying it as a
`sky130_sram_2kbyte_1rw1r_32x512_8` pair (or one 1 KiB ×32 macro per
bank) fixed both: the area dropped to a few tenths of a mm² and the read
became single-cycle, lifting the achievable clock by ~50×.

## If you define the macro's behavioral model in the SAME file, GUARD IT

cocotb sim needs a behavioral body for the named macro, so it is common to
append a `module sky130_sram_..._8 (...); reg [..] mem [..]; ... endmodule`
to the bottom of the block's `.v`. **If you do this WITHOUT a synthesis
guard, yosys elaborates that behavioral body and turns the macro into
depth×width flip-flops — the instantiation was correct, but the bundled
model defeated macro mapping.** This passes lint and sim and is silently
fatal at synth/backend (e.g. a 256×32 ×2-port model × 4 instances came out
as ~8200 flops with 0 macro cells, 3.3× the area).

**Rule:** any inline behavioral SRAM model MUST be wrapped so synthesis sees
an empty `(* blackbox *)` stub and only simulation sees the behavioral body.
yosys's `read_verilog` defines the `SYNTHESIS` macro; cocotb/iverilog does
not. Canonical, mirror this exactly:

```verilog
`ifndef SYNTHESIS
module sky130_sram_1kbyte_1rw1r_32x256_8 ( ...full port list... );
    reg [31:0] mem [0:255];
    // ...registered 1RW + 1R behavioral body for simulation...
endmodule
`else
(* blackbox *)
module sky130_sram_1kbyte_1rw1r_32x256_8 ( ...identical port list, all wire... );
endmodule   // empty body: synth maps the instance to the real .lib/.lef macro
`endif
```

A synth/PPA check that sees "macro named + 0 macro cells + flops ≈
depth×width" should suspect a MISSING synth guard on a bundled behavioral
model, not just a bad coding style on the instance itself.

## Quick checklist — paste into your spec's storage section

- [ ] Every storage structure names its choice: flip-flops or a specific SRAM macro.
- [ ] The choice follows the decision rule (≳256 words / ≳1 KiB → macro; small register state → flops).
- [ ] No memory is left as a raw `reg [...] mem [0:N]` array. Macro tier → `cs_sram_1rw`/`1rw1r`; flop tier → `cs_fpmem_1rw`/`1rw1r` (registered read / capture flop). A bare comb-read `reg mem[]` is never allowed.
- [ ] Every memory is a CoreSmith primitive: `cs_sram` (macro, preferred for deep/large) or `cs_fpmem` (registered flop array — fine at lower Fmax). Never a raw comb-read `reg mem[]`.
- [ ] Every read-only CONSTANT table above the macro threshold is a `cs_rom_1r` (`impl=rom`, INIT_FILE relative to project root) — never a `cs_sram` with its write port tied off.
- [ ] If a macro: a specific sky130 macro is named, with port mapping and any banking scheme.
- [ ] The macro capacity is counted against `max_onchip_sram_kb` and `max_macro_count`.
- [ ] The FSM / pipeline accounts for the **1-cycle registered read latency** of every SRAM macro.
- [ ] If a behavioral macro model is bundled in-file, it is wrapped in `` `ifndef SYNTHESIS `` (behavioral) / `` `else `` `(* blackbox *)` empty stub / `` `endif `` so synth maps the real macro instead of flopping the model.
