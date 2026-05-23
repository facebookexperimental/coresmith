# 32-bit Adder Full-Flow Requirements

Build a Sky130 Verilog-2005 soft IP block for a 32-bit unsigned adder and use
it as a full frontend plus backend smoke test.

The design is intentionally small. It should exercise the complete CoreSmith flow
from architecture through RTL generation, cocotb simulation, synthesis,
OpenROAD placement/routing, Magic DRC, and LVS when the local toolchain is
available.

Required behavior:

- Combinational 32-bit unsigned addition.
- Inputs: `a[31:0]`, `b[31:0]`, and `cin`.
- Outputs: `sum[31:0]` and `cout`.
- Functional equation: `{cout, sum} = a + b + cin`.
- No latches, tri-states, gated clocks, asynchronous resets, memories, or
  inferred sequential state.
- Target 50 MHz in Sky130 `sky130_fd_sc_hd`.

Validation DV KPI:

- Exhaustive directed carry-chain cases and at least 256 randomized vectors
  must match the mathematical reference exactly.
- Backend success requires a generated GDS and zero Magic DRC violations.
