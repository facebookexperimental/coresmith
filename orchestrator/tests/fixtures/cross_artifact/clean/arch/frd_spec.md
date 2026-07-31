# FRD — raster2d_qspi

<!--
Fixture: the CORRECTED counterpart of contradictory/arch/frd_spec.md. Every
host-interface reference agrees with the ERS (6.25 MHz / 25 Mbit/s / 80 ns).
Used by the negative test that proves the gate reports nothing on consistent
artifacts.
-->

## Performance Requirements

### PERF-007 — sustained OUT bandwidth
   **Requirement**: At the supported 6.25 MHz QSPI clock, the OUT path shall sustain the raw active-direction data rate of 25 Mbit/s after turnaround.

   **Acceptance criteria**: At 6.25 MHz SCK, a complete OUT read shall have exactly 8,192 correct, ordered sample nibbles, zero stalls, zero duplicates, zero byte errors, and zero cycles of lane contention. The first data sample shall be the high nibble of address `0x002000`; consecutive pairs shall reconstruct framebuffer bytes 0 through 4,095.

## Timing Requirements

### TIME-002 — pin-level CDC fidelity
   **Acceptance criteria**: At SCK `<=6.25 MHz` with high and low phases `>=80 ns`, the ordered RTL event/nibble trace shall equal the pin-level BFM edge/nibble trace exactly for every randomized phase offset; event loss, duplication, or nibble mismatch count shall be zero.

### TIME-003 — asynchronous host operating range
   **Requirement**: The asynchronous host interface shall operate correctly at QSPI SCK frequencies from host-static operation through 6.25 MHz, provided each high and low phase is at least 80 ns. No fixed phase or integer divider relationship to `wb_clk_i` may be assumed.

   **Acceptance criteria**: Phase-randomized pin-level tests at 6.25 MHz and with `tHIGH>=80 ns`, `tLOW>=80 ns` shall complete all command classes and maximum transfers with zero nibble, event, or OEB errors. Tests shall cover all distinct SCK launch offsets over one 20 ns core period.

## Coverage

- **PRD functional requirement coverage — CDC and SCK limit (PRD-FR-008 and PRD-FR-009)**: Randomize SCK phase over the 20 ns core period at 6.25 MHz and at slower/non-integer ratios; require `tHIGH,tLOW>=80 ns`. Run structural CDC analysis for each two-flop chain and no derived clock. Pass requires IFACE-008, INV-001, TIME-002, and TIME-003.

## KPIs

- **KPI-007 — continuous full-frame QSPI integrity**: At 6.25 MHz, capture actual lanes and OEB for a complete OUT read. Metrics are nibble count, cadence, duplication, contention, and byte errors; thresholds are exactly 8,192 data nibbles, one per SCK cycle, and zero stalls, duplicate samples, wraps, contention cycles, or byte errors under PERF-007/IFACE-007.

- **KPI-011 — maximum QSPI/CDC capture**: Run asynchronous, phase-randomized transactions at 6.25 MHz with both SCK phases at least 80 ns. Metrics are captured/emitted nibble and event errors; threshold is zero under INV-001 and TIME-002. Structural evidence must also show two-flop synchronization and no SCK-derived clock.

## Signoff

- **On-silicon verification**: A host MCU or logic analyzer shall drive QSPI at and below 6.25 MHz, measure `wb_clk_i` and pin IRQ edges, verify lane turnaround/OEB indirectly through contention-free electrical behavior, run the public checksum and acceptance cases, and read all 4,096 bytes per operation.
