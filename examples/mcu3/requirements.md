# mcu3 — Minimal 3-stage Pipelined Microcontroller

Build a Sky130 Verilog-2005 soft IP block for a 3-stage pipelined 8-bit
microcontroller. This is the canonical CoreSmith smoke test: a single
block end-to-end through RTL gen → lint → cocotb sim → synthesis. The
goal is a tiny but real pipelined design, not a feature-complete CPU.

## Architecture decomposition

This is a **single-block** design. The architecture stage should emit
exactly one block named `mcu3` with the interface and ISA below. Memory
map, clock tree, and register-spec elaboration are not required and
should remain disabled.

For the generated block, include a Verilog target under `rtl/mcu3/` and
a cocotb testbench target under `tb/cocotb/test_mcu3.py`.

## Pipeline stages

All stages synchronous on `posedge clk`. Active-low **synchronous** reset
on `rst_n`. No latches, no async resets, no tri-state buffers. One
`always @(posedge clk)` per stage for readability.

1. **FETCH**       — 8-bit PC. Each cycle reads one instruction byte
                     from a 256-byte program ROM (`rom_data`, addressed
                     by `pc_o`). Holds when `stall_i` is high.
2. **DECODE/EXEC** — Decodes the instruction, reads operands from an
                     8 × 8-bit register file (`R0..R7`, where `R0` reads
                     as zero), performs the ALU op.
3. **WRITEBACK**   — Writes the ALU result back to the destination
                     register, or for the `OUT` instruction, drives
                     `out_o` (a memory-mapped output register).

Forward results from WRITEBACK to DECODE/EXEC operand reads to avoid a
1-cycle RAW hazard on consecutive `ADD`/`SUB`/`LDI` instructions.

## ISA (8-bit fixed-width opcodes, 8-bit data)

```
NOP        opcode 0x00                       do nothing
ADD rd,rs  0x10 | (rd<<2) | rs               rd <= rd + rs
SUB rd,rs  0x20 | (rd<<2) | rs               rd <= rd - rs
LDI rd     0x40 | (rd<<2), next byte = imm   rd <= imm  (multi-byte)
JMP        0x80, next byte = target PC       PC <= target  (multi-byte)
OUT rs     0xC0 | rs                         out_o <= rs
```

`R0` reads as 0; writes to `R0` are silently dropped.

## Top-level interface (module `mcu3`)

```
input              clk
input              rst_n      // active-low synchronous reset
input              stall_i    // freeze the fetch stage
input  [7:0]       rom_data   // combinational rom[pc]
output [7:0]       pc_o       // current program counter
output [7:0]       out_o      // last value written by OUT
output             halted_o   // 1 when an undefined opcode is latched;
                              //   held until reset
```

There is no AXI/AXI-Stream interface. The bus protocol is dedicated
pins; the ROM port is combinational read-only and `out_o` is a
memory-mapped output register.

## Constraints (Sky130 `sky130_fd_sc_hd`)

- Verilog-2005 only, synchronous design.
- No async resets, no latches, no tri-state buffers.
- Register file initialised to zero on reset; PC starts at 0.

## Validation (testbench expectations)

The cocotb testbench should:

- Drive a small ROM with: `LDI R1=5; LDI R2=3; ADD R1,R2; OUT R1; JMP 0`
  (looped).
- Assert that `out_o == 8` within ~10 cycles after reset deassertion.
- Toggle `stall_i` for 2 cycles and verify `pc_o` does not advance.
- Run for ~100 cycles total; pass on no `halted_o` assertion.

## KPIs

- **Functional**: cocotb passes all listed tests.
- **Synthesis**: passes Yosys with `sky130_fd_sc_hd`, < 2500 cells.
- **Target clock**: 50 MHz.
