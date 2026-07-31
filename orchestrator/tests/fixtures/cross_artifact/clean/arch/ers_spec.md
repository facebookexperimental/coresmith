# ERS — raster2d_qspi
**Revision:** 1.0

<!--
Fixture: the CORRECTED counterpart of contradictory/arch/ers_spec.md. The
per-block qspi_slave_engine requirement now agrees with the READ-serializer
requirement (rise-scheduled), and the FRD next to it now states the same
6.25 MHz / 25 Mbit/s / 80 ns operating point.
-->

## Summary
This ERS defines the implementation and verification requirements for raster2d_qspi, a Verilog-2005 fixed-function 64x64 triangle rasterizer behind the locked Caravel user_project_wrapper. The design has one 50 MHz wb_clk_i domain. An asynchronous four-lane mode-0 QSPI slave loads 1 to 64 packed triangles, controls operation state, and streams the complete framebuffer.

## Speed & Feeds
- **Target clock:** 50 MHz
- **Throughput:** At qspi_sck=6.25 MHz, four bits per SCK provide 25 Mbit/s raw active-direction bandwidth. Compute shall accept one row-major pixel per wb_clk_i cycle with no scanline bubble.

## Functional Requirements
- qspi_csn, qspi_sck, and qspi_io[3:0] shall each pass through independent two-flop synchronizers. A registered previous synchronized SCK shall form one-cycle rising/falling pulses. Host nibbles are sampled only on rising pulses.
- Legal QSPI timing is asynchronous SCK <=6.25 MHz with each high and low phase >=80 ns (the published floor: host clock period >=8 wb_clk_i cycles, each phase >=4 wb_clk_i cycles; the harness may run slower, never faster). The RTL shall assume no fixed phase or divider relationship to wb_clk_i.
- The READ serializer shall prefetch the first byte during dummy and prefetch each following byte while the current byte is serialized. Output nibbles shall be SCHEDULED DURING THE LOW PHASE FROM THE ALREADY-CONSUMED PRECEDING SYNCHRONIZED RISING-SCK EVENT: preload the high nibble during dummy; assert qspi_drive_en once the SECOND dummy rising sample has been consumed; advance to the low nibble after the high data sample is consumed; promote the next byte after the low data sample is consumed. This gives a full SCK period from each raw rising edge to the following host sample and retains legal mode-0 setup/hold.

## Per-Block Requirements
### qspi_cdc_frontend
- Implement independent two-flop wb_clk_i synchronizers for CS, SCK, and each of four IO lanes; reset both stages synchronously to CS=1, SCK=0, IO=0.
- Register sck_prev from synchronized SCK and produce one-cycle sck_rise=(~sck_prev)&sck_sync2 and sck_fall=sck_prev&(~sck_sync2). sampled_io is the second-stage lane vector.
- **Interface:** Asynchronous dedicated pins to core-domain valid-only edge_event and static csn_sync

### qspi_slave_engine
- Receive two high-then-low nibbles per byte on sck_rise; issue one aperture write request on each complete WRITE data byte and increment the 24-bit byte address.
- For READ, issue the first aperture request early enough to capture its one-cycle response during dummy, then maintain current/next byte registers and valid bits so every CONSUMED data-phase sck_rise advances the serializer by one nibble without a bubble (rise-scheduled per the READ-serializer requirement above).
- **Interface:** Core-domain valid-only edge input; byte-wide request/response QSPI aperture; static registered nibble/drive output

## Constraints
- Constrain wb_clk_i to 20 ns.
- QSPI SCK is externally constrained to <=6.25 MHz with tHIGH and tLOW >=80 ns (published floor); this is a functional CDC assumption, not a synchronous STA path.
