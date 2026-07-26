# CoreSmith

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/facebookexperimental/coresmith)

Coresmith converts prompts to silicon. It uses LangGraph to drive the full RTL-to-GDS flow: architecture specification, RTL generation, verification, synthesis, and physical design. You start Coresmith through your agent (Claude or Codex), and it works as a daemon that spawns subagents for you until the GDS is created or your input is required. 

> **Try it:** click the Codespaces badge above for a pre-built sandbox with the full EDA toolchain (Yosys, OpenROAD, Magic, Sky130 PDK) and Claude CLI ready to go. Note that it takes up to ten minutes to boot. Once it boots, launch Claude or Codex in the terminal. You will need to log in to your Claude or Codex account within the codespace.  For your first time, keep it simple: ask for a 32 bit adder.

> \[!NOTE]
> Agentic silicon design is expensive. Every agent needs to reference the chip specification to accurately architect their block. Codex Pro (100$/month) is the recommended minimum viable inference provider. Claude Max (100$) is usable until June 15th, after which you will have to pay API rates. Codex is superior at prompt caching and a prerequisite for designs exceeding in-order MCU complexity. You can use local LLMs (unquantized only) to reduce cost, with severely degraded performance: https://coresmith.ai/blog/qwen-vs-gemma


## What It Does

Coresmith will ask questions about your requirements, then run these phases:

1. **Architecture** -- Generates a Product Requirements Document (PRD) and block diagram via a multi-step LangGraph state machine. (Memory-map, clock-tree, and register-spec stages also exist but are **off by default** — enable with `CORESMITH_ENABLE_MEMORY_MAP=1` / `_CLOCK_TREE=1` / `_REGISTER_SPEC=1`.)
2. **RTL Generation** -- An LLM agent converts specifications into synthesizable Verilog-2005
3. **Verification** -- Another LLM agent generates cocotb testbenches; Verilator lints and simulates
4. **Synthesis** -- Yosys synthesizes each block to a gate-level netlist targeting the SkyWater Sky130 130nm PDK
5. **Backend** -- OpenROAD/Magic/netgen handle place-and-route, DRC, and LVS
6. **Diagnosis** -- On failure at any step, a debug agent analyzes the root cause and retries with corrective constraints

The pipeline is interactive via a daemon that exposes endpoints to control the LangGraph pipeline.

### Architecture Phase

The architecture graph gathers requirements, generates specs, and gates the design through human review before RTL generation begins.

![Architecture graph](docs/images/architecture-graph.gif)

### Frontend Pipeline

Each block flows through RTL generation, testbench creation, simulation, and synthesis — with automatic diagnosis and retry on failure.

![Frontend pipeline](docs/images/frontend-pipeline.gif)

### Backend Pipeline

Post-synthesis, LLM agents drive place-and-route, DRC, GDS export, and LVS — each with tool-specific fix loops.

![Backend pipeline](docs/images/backend-pnr.gif)

## What's New

**Architecture → micro-architecture stage.** The architecture graph decomposes a
design into per-block *micro-architecture* specs before any RTL is written: each
block gets its own uArch spec (interfaces, latency/throughput intent, and a
byte-exact reference model) that the frontend pipeline then implements and verifies
block-by-block. An Amaranth block-model + composition methodology
(`CORESMITH_BLOCK_GOLDENS`) elaborates each per-block reference model and the
integrated chip model as `Elaboratable`s; pysim carries their real
clock/handshake/latency semantics and the composition gate compares the composed
output bit-exactly against the software reference.

**Complexity-aware decomposition into memory vs compute.** A deterministic,
AST-based pass scores each block's reference slice on four axes (flop count, latency,
data-locality, modeling complexity) and min-cut partitions an over-budget block along
function boundaries -- separating storage-heavy sub-blocks from compute so each stays
inside its per-block area / FF / SRAM budget. Storage that should be a macro is
mapped to an **SRAM macro** (with LEF/GDS/lib injection) rather than synthesized as a
flop array.

**Proxy / coverage metrics + honest signoff gates.** DV is functional- and
coverage-driven, and the backend signoff gates are deterministic and *fail closed on
blank or proxy signals* rather than trusting a tool's summary line:

- **PnR route-DRC** and a **DRC-count** gate (guards against a parser reading an
  empty tool line as "0 violations").
- **LVS** with a **benign-tie classifier** that reconciles constant-tie / replicated
  top-pin and physical-only (tap/fill/decap) device-count deltas, and fails closed
  when a delta is *not* provably benign.
- **Synth cell-budget**, **memory-price**, and **aggressive flop-vs-SRAM thresholds**
  (a three-way bits|width|depth policy) that push storage over budget onto macros.

These gates are the difference between "a tool printed success" and "the evidence
actually supports signoff."

## PPABench Results

Eight designs have been driven through the pipeline on the SkyWater Sky130 130nm
PDK. They sit at **two different levels of evidence**, and the table keeps them
apart rather than blending them:

- **Backend signoffs (5)** -- GEMM, AES, Raster, FFT, JPEG -- architecture through
  DRC/LVS/timing signoff. Four of five signed off; JPEG is backend-blocked. Timing
  is post-route STA setup slack at the 50 MHz target.
- **Frontend-only (3)** -- h264-style encode, Theora encode, Theora decode --
  functionally verified, but **no place-and-route was run**. Their DRC/LVS/power
  cells are empty as a matter of fact, not omission. These rows are **not**
  signoffs and must not be read as such.

| Design | Coverage | DRC | LVS | Area (util) | Power | Timing @50 MHz | Throughput ⁸ | 480p-equiv |
|--------|----------|-----|-----|-------------|-------|----------------|--------------|------------|
| GEMM   | 94.2%      | 0        | match       | 1.06 mm² (41%) | 12.9 mW     | +5.52 ns MET  | 33.3 MMAC/s (1.50 cyc/MAC) | n/a |
| AES    | 98-99% ¹   | 0 *asserted* ² | match ³ | 0.31 mm² (26%) | 16.6 mW     | +11.32 ns MET | 128 Mbit/s pin; 533 Mbit/s core ⁵ | n/a |
| Raster | 96.6%      | 0 *audited* ² | match ³ | 2.32 mm² (39%) | *invalid* ⁴ | +6.42 ns MET  | 3.13 Mpix/s readout; 50 Mpix/s fill ⁵ | 10.2 / 162.8 fps |
| FFT    | 95.7%      | 0        | match ³     | 1.37 mm² (30%) | 31.4 mW     | +7.23 ns MET  | 37.3k transforms/s (256-pt, 1341 cyc) | n/a |
| JPEG   | 84.6% (min)| **X** (4166) | **X** not proven | -- | --      | routed 50 MHz only | 3.29 Mpix/s (973 cyc / 8x8 block) | 10.7 fps |
| h264-style encode | 86.4% merged / 79.7% single-case ⁶ | **X** | **X** | 30.3 mm² (50%) ⁷ | **X** | +3.54 ns post-placement ⁷ | 22.2 fps @ 640x352 QP28 ⁷ | 16.3 fps ¹¹ |
| Theora encode | 89.8% agg / 77.3% min | **X** | **X** | 0.52 mm² ⁹ | **X** | MET, no margin ¹⁰ | 459 fps @ 64x48 | 4.6 fps ¹² |
| Theora decode | 84.2% agg / 73.8% min ¹³ | **X** | **X** | 0.73 mm² ⁹ | **X** | MET, no margin ¹⁰ | 161 frame/s @ 64x48 e2e ¹³ | 1.6 fps ¹² |

**X** = not run or not closed -- a blank would read as an oversight, these are known gaps.

### Pixel throughput, normalized to 480p

Five of the eight designs move pixels, so their rates can be put on one axis. The
480p column is **arithmetic on the measured pixel rate** (`Mpix/s ÷ 307,200`), not a
measured 480p run -- no design here was ever driven at 640x480. Read it as "what
this pixel rate would sustain," and check the reachable column before quoting it.

| Design | Measured at | Cycles/frame | Mpix/s | 480p-equivalent | 480p reachable today? |
|--------|-------------|--------------|--------|-----------------|-----------------------|
| Raster (fill) | -- | -- | 50.0 | 162.8 fps | yes -- rate is geometry-bound, not buffer-bound |
| h264-style encode, QP40 | 640x352 | 2,039,016 | 5.52 | 18.0 fps | plausible ¹¹ |
| h264-style encode, QP28 | 640x352 | 2,254,703 | 5.00 | **16.3 fps** | plausible ¹¹ |
| h264-style encode, QP16 | 640x352 | 3,224,740 | 3.49 | 11.4 fps | plausible ¹¹ |
| JPEG | 8x8 block | 973 / block | 3.29 | 10.7 fps | untested |
| Raster (QSPI readout) | -- | -- | 3.13 | 10.2 fps | yes |
| Theora encode | 64x48 | 108,901 | 1.41 | 4.6 fps | **no** ¹² |
| Theora decode (core) | 64x48 | 170,932 | 0.90 | 2.9 fps | **no** ¹² |
| Theora decode (end-to-end) | 64x48 | 310,601 | 0.49 | 1.6 fps | **no** ¹² |

- **¹** AES functional cores are 98-99% (round-core 98.35%, QSPI frontend 98.99%); the small structural wrapper block is 86.7%. FFT aggregate 95.7% (min applicable 90.9%); GEMM/JPEG shown as minimum applicable coverage.
- **²** Neither of these is a raw 0; both are a nonzero Magic count reduced to 0 by an interior exclusion, and **the two rest on different quality of evidence.** *Raster is audited*: Magic reports 136 `met4.4a` minimum-area boxes, and `drc_result.json` records `clean: true` with a per-macro attribution (`sram_1rw1r_32_64_8_sky130`: 64, `sram_1rw1r_9_512_8_sky130`: 72) placing every box inside a placed signed-off SRAM macro's interior. One gap remains: Magic's own log gates on 160 error *tiles* while the exclusion is accounted in 136 geometry *records*, so "all excluded" is not strictly proven in the unit that gates. *AES is asserted*: Magic reports 4192 error tiles (`li.3` + `li.c2`), and **its own `drc_result.json` still reads `clean: false, violation_count: 4192`** with no `excluded_count`, no per-cell attribution, and no audit field. The only artifact claiming 0 is a 145-byte hand-written summary. Standard-cell `li` spacing violations under flat DRC are a well-known Sky130 artifact and the exclusion is plausible, but plausible is not audited -- AES's 0 should be read as unproven pending the same coordinate-level audit raster received.
- **³** LVS match under benign-tie classification -- constant-tie / replicated top-pins plus zero-transistor tap/fill/decap device-count deltas, each explicitly identified (no unexplained residue).
- **⁴** Raster power is invalid: an OpenRAM macro-power table returned a nonphysical value; the finite non-macro + leakage subtotal is ~2.26 mW.
- **⁵** Two rates are given where the compute core and the chip pins differ materially. The **pin rate is quoted first because it is the honest system number** -- it is what crosses the QSPI boundary end-to-end; the core rate is only what the datapath sustains internally, behind that bottleneck.
- **⁶** h264 line coverage is `verilator_coverage`, merged across all 6 preserved testbenches (2438/2823 points); the single-case figure is one full 640x352 QP28 frame (2250/2823). Every measured run was byte-exact against the golden. **This does not cover the synthesized RTL** -- see below.
- **⁷** **Recorded, not endorsed -- these are real measurements of a build that was functionally wrong.** The 30.3 mm² / 50% util / +3.54 ns figures were measured on the `-DSYNTHESIS` build, later shown not to implement the encoder (see below); the area is modest and the slack comfortable partly *because* logic is missing. Area is also the placement die box rather than a routed result, and is ~13x the largest signed-off design because 74 1 KB SRAM macros dominate; timing is post-**placement**, so neither is comparable to the post-route rows above. The throughput figure comes from the behavioral build, which is functionally correct but has no netlist. A later `chip_top` sweep on a separate machine reportedly did synthesise; those results are not captured here and would supersede this row.
- **⁸** All Functionality figures are measured RTL-simulation cycle counts converted at the stated 50 MHz target. They are simulation results, not silicon measurements.
- **⁹** Theora area is a **sum of per-block syntheses with SRAM macros blackboxed** -- macro area is excluded and no whole-chip flat synthesis exists. Utilization was never measured. Not comparable to the routed mm² of the signoff rows.
- **¹⁰** Both Theora runs report WNS 0.0000 ns and Fmax 50.00 MHz for every block. This is **correct but uninformative**, not a fabrication: WNS is *worst negative slack*, so 0 is its defined value when no path is violating, and the pipeline does record real negatives when they exist (`dct_token_encoder` logged -2.83, -2.95 and -2.39 ns across attempts before it was re-pipelined to 0). What the flow never records is **positive** slack margin, so the reported "Fmax 50.00 MHz" is the 50 MHz target restated rather than a measured maximum -- these designs are proven to *meet* 50 MHz and have no measured headroom above it. TNS is not recorded at all (no column exists for it).
- **¹¹** h264 was measured at 640x352 -- the same *width* as 480p, differing only in row count -- and its verified width sweep covers 11 widths up to 640. Extrapolating along rows is therefore a modest step, but 640x480 was never run and no height sweep exists, so it remains untested rather than demonstrated.
- **¹³** Every Theora *decode* figure is **self-reported by the engine and cannot be independently checked**, because two of the design's 21 RTL files are missing from the only surviving archive -- see the decoder note below. The coverage and throughput numbers are what the run recorded; nothing external corroborates them.
- **¹²** Theora **cannot be configured for 480p without an RTL change.** `frame_memory.v` hardcodes `localparam [12:0] FRAME_BYTES = 13'd4608` -- one 64x48 4:2:0 frame. A 480p frame is 450 KiB, ~100x larger, and a 13-bit address cannot reach it. The 480p-equivalent figures for Theora are pixel-rate arithmetic describing a configuration the design does not support; they are not a claim that this hardware encodes or decodes 480p.

**JPEG is not signed off.** DRC genuinely does not close -- a from-scratch Magic run
finds 4166 real `li.*` violations that the engine's stdout parser mis-reported as 0
(an empty-string bug in `_parse_magic_drc_count()`; **fixed**, the parser now falls
back to counting concrete violation rects in the report) -- and its LVS net-delta is
architectural (the DCT row/column caches are flop arrays rather than SRAM macros, so
read-mux symmetry defeats a unique match), not a proven-benign tie.

**Re-audit of the four DRC-0 rows against the fixed parser.** All four were signed
off before the parser fix landed, so each was re-checked against Magic's own logs
and structured results rather than the stdout line:

| Design | Magic log | Structured result | Verdict |
|--------|-----------|-------------------|---------|
| GEMM   | `Total DRC errors found: 0` | `DRC count: 0` | **genuinely clean** |
| FFT    | `No errors found. Total DRC errors found: 0` | empty report, consistent | **genuinely clean** |
| Raster | `Total DRC errors found: 160` | `clean: true`, 136 boxes attributed per-macro | **audited exclusion**, unit gap noted ² |
| AES    | `Total DRC errors found: 4192` | **`clean: false, violation_count: 4192`** | **asserted exclusion, unproven** ² |

The parser bug did **not** retroactively corrupt GEMM or FFT -- both are clean in
Magic's own words, and FFT's empty report is consistent with its log rather than a
symptom of it. The residual risk is not the parser; it is that an exclusion can be
asserted in prose and reach the table as a 0 while the run's own machine-readable
result still says `clean: false`. That is what AES did.

**The h264 encoder's PPA numbers are withdrawn: they measure a stub, not an
encoder.** `intra_rd_encode_core.v` -- the RD encode core, the largest block in the
design -- is split by an `` `ifndef SYNTHESIS `` guard into two independent
implementations. Simulation compiles the behavioral branch (lines 36-1020);
`synth_rung3.ys` passes `-DSYNTHESIS` and compiles the FSM branch (lines 1022-1355).
The two were built and run side by side, and they are **not equivalent**:

- **0 of 21 test cases byte-exact.** Every spotcheck crop and every width-sweep case
  produced different output; the full 640x352 QP28 frame -- the one with a stored
  golden -- **hard-deadlocks**, stalling at pixel 51,201 and emitting 0 bytes with
  `error`, `status` and `frame_done` all held at 0. It hangs silently.
- **The FSM branch has no encoder in it.** Its entire datapath is two functions:
  `coeff_from_sample` (`sample - 128`, a raw residual) and `nz_from_coeff`. Every
  encoder primitive present in the behavioral branch is absent from it -- intra
  prediction, mode search, DCT, quantisation, zigzag, RD lambda and the complete
  CAVLC tables all appear 0 times. Its 18 state names (`STATE_DCT_ROW`,
  `STATE_QUANT`, `STATE_CAVLC_LEVELS`, `STATE_PRED_RESID`...) gate nothing, and all
  four SRAM read-data ports are declared and wired but **never read by any
  expression** -- the memories are write-only. Output size is QP-invariant
  (362/362/363 bytes at QP20/28/36, against the behavioral 94/45/16), which is the
  signature of a design with no quantiser.
- **This is exactly why the PPA looked good.** The 3,426 FF figure is reachable only
  because the FSM branch dropped the reconstruction and context state; the
  behavioral branch holds it in `reg [1884159:0] recon_flat_q` -- 1.88 Mbit of
  flops, memory-as-flops that cannot synthesise at any sane area. `WNS +3.535 ns` is
  comfortable because the arithmetic that would set the critical path is not there.

So the design has **no synthesisable encoder**: the version that was verified cannot
be synthesised, and the version that was synthesised does not encode. The functional
results (21/21 byte-exact, +2.56% BD-rate vs x264) are real but describe RTL for
which no netlist exists; the area and timing figures are therefore withdrawn rather
than merely re-qualified. The throughput figure is likewise from the behavioral
branch. The other six modules have no such guard and are sim-equals-synth.

The FSM branch was not under-tested -- under these runs it reaches **143/144 line
points (99.31%)**. Coverage was high and the code was wrong, which is the point: a
line-coverage gate cannot detect a module that executes fully while computing
nothing. What would have caught it is comparing the *synthesised* configuration
against the golden, and no gate did that.

**The two Theora paths rest on different quality of evidence.** The **encoder** was
independently re-verified clean-room (Verilator 5.036 + cocotb 2.0.1, outside the
engine), reproducing the engine's per-frame PSNRs exactly.

**The decoder's result is not independently verifiable, and no longer can be.** Its
byte-exactness against the golden rests solely on the engine's own validation
testbench; its *integration* DV is a QSPI bus-protocol conformance test explicitly
labelled compute-lane-independent, so it proves bus liveness rather than decoding.
An attempt to re-verify it clean-room established that **the final decoder RTL does
not exist in complete form anywhere.** The run directory is gone; the only surviving
copy is a backup whose `rtl/` tree holds 22 files, missing
`decode_frontend/header_parser.v` and `frame_pipeline/motion_qi_control.v` -- both
instantiated by `theora_dec_top.v` (lines 637 and 768) and both listed as PASS
blocks in the run's own final report. `tb/` is excluded from the backup entirely.
The only copies of the two missing files are from a snapshot taken ~3 hours before
the run finished, and they are provably incompatible with the final design: the last
RTL edit compacted `huffman_codebook_memory` from a 2600x8 array (`AW=12`, node
region < 2520) to 1360x8 (`AW=11`, `HUFF_NODE_WORDS = 1280`), and the stale
`header_parser` still emits the old 2520-word layout, so codebook write #1313 is
rejected as an illegal address and the setup header fails with
`ERR_MEMORY_RANGE`. A build from the archive therefore fails at packet 2 -- but that
failure is an artifact of the missing file, so it is evidence neither for nor
against the decoder. **The 4-frames-byte-exact claim is now unfalsifiable from the
archived artifacts.** Treat the Theora decode row as self-reported.

Two process findings fall out of this, independent of whether the decoder worked.
The backup that was meant to preserve a verified design **silently dropped two of
its RTL files**, so "we have a backup" did not mean "we can rebuild it." And the
archived decoder RTL **does not compile under Verilator at all**: the engine's final
edits inserted 20 `/* verilator lint_off MODMISSING */` pragmas, and `MODMISSING` is
not a real Verilator message code in either 5.020 or 5.036 -- Verilator rejects it as
`Unknown verilator lint message code`, and `-Wno-fatal` does not suppress it.

**The Theora encoder is functionally correct but rate-inefficient.** Against
libtheora on md5-identical clips it spends **4.25x** the bits on a gradient clip
(24,720 B vs 5,822 B) for +2.35 dB, 2.11x on synthetic motion, and 1.07x on noise.
Only a single operating point (q=6) was ever run, so **no BD-rate exists** for this
design -- unlike the h264 encoder, whose +2.56% BD-rate vs x264 comes from a real
multi-QP sweep. Root-cause and a rate sweep are open work.

> **Also in flight:** an AX.25 framer is in progress, and a TinyStories on-chip
> language-model design is mid-run. Neither has results to report yet.

## LLM Cost
You must have a Claude Code Max or OpenAI Codex Pro subscription. Codex is recommended and GPT 5.6 is superior at silicon design.

| Design  | Opus 4.8 | GPT 5.6 | 
|------|---------|---------|
| MCU | OK | OK |
| JPEG | Exceeds 5hr limit on Max 5x | OK |

You can use an API key, but it will be expensive.

## Setup

Three install paths -- see **[SETUP.md](SETUP.md)** for the full commands and a
reproducible Ubuntu 22.04 reference setup:

- **Option A -- Docker / RunPod / Codespace** (recommended for first-time users): a
  pre-built image with the full EDA toolchain (Yosys, OpenROAD, Magic, Sky130 PDK) +
  the Claude/Codex CLI. Click the Codespaces badge above, or pull the container.
- **Option B -- Local install (Nix-based backend)**: `nix develop` pins every EDA
  tool; run the MCP server or the `coresmithd` daemon + `bin/coresmith` CLI.
- **Option C -- Linux without Nix or Docker (OSS-CAD-Suite)**: install the frontend
  EDA toolchain, Node + Claude CLI, a Python venv, and the Sky130 PDK by hand.

After any path, run `make preflight` -- it checks the Sky130 PDK files and the
`yosys` / `verilator` binaries on `$PATH` and prints exactly what's missing.

## Architecture

The system is built around three LangGraph state machines:

```
Phase 1: ARCHITECTURE        Phase 2: RTL PIPELINE        Phase 3: BACKEND
------------------------    ------------------------     -----------------
User Requirements            Per-block loop:              Post-synthesis:
  |                            uArch Spec                   Place & Route
  v                            RTL + Lint                   DRC
PRD (sizing questions)         Testbench + Sim              LVS
  |                            Synthesis                    Timing Sign-off
  v                            Diagnose / Retry
Block Diagram
  |
  v
Constraint Check -> OK2DEV Gate
  (off by default:
   Memory Map -> Clock Tree -> Register Spec)
```

## Project Structure

```
coresmith/
  orchestrator/           # Core pipeline engine
    architecture/         #   Architecture phase (PRD, block diagram, constraints)
    langchain/            #   LLM agents (RTL gen, testbench, debug, timing)
    langgraph/            #   State machines (architecture, pipeline, backend, tapeout)
    pdk/                  #   PDK configuration
    pdk_templates/        #   EDA tool templates (Yosys, Magic, netgen)
    telemetry/            #   OpenTelemetry tracing
    mcp_server.py         #   MCP server for Claude Code integration
    config.yaml           #   Pipeline configuration
    tests/                #   Test suite
  scripts/                # Toolchain installer, Nix wrappers
  bin/coresmith           # CLI client for the coresmithd HTTP daemon
  orchestrator/daemon/    # coresmithd FastAPI daemon (one per project_root)
  Makefile                # Build targets
  requirements.txt        # Python dependencies
```

## Usage

### Interactive (Claude Code or Codex CLI)

The best way to use Coresmith is through Claude Code or Codex CLI. The CLAUDE.md has all the instructions your agent needs to get started. The coresmith daemon will build your ASIC and escalate any blocking questions up to you.

### Daemon mode (outer agent or human drives)

```bash
bin/coresmith daemon start --project-root $(pwd)
bin/coresmith run start --project-root $(pwd)
bin/coresmith state --project-root $(pwd)
bin/coresmith resume --project-root $(pwd) --action approve
```

The daemon does **not** auto-approve interrupts. An outer agent (Claude on
cron, a human, or another script) drives every decision via `coresmith
resume`. See [CLAUDE.md](CLAUDE.md) for the full decision contract.

### Web UI (live dashboard)

`orchestrator/vscode-ext/serve.py` serves the same ReactFlow dashboard the
VS Code extension provides, but as a plain web page in any browser — the
graph view, a Gantt timeline of every block/node, per-node LLM
trajectories (prompts, tool runs, reasoning), and a browser for generated
collateral (RTL, testbenches, waveforms, GDS reports). It's **read-only**:
it visualizes a daemon run by reading that run's `.coresmith/` event logs,
so it never drives the pipeline.

Because it reads the run's event logs, the webview must point at the **same
project root as the daemon** — it keys off the same `CORESMITH_PROJECT_ROOT`
the daemon uses.

```bash
# 1. Build the webview bundle once (produces dist/webview.js).
#    serve.py exits with an error until this exists.
cd orchestrator/vscode-ext && npm install && npm run build && cd -

# 2. With the daemon already running against $RUN_DIR (see above),
#    start the webview against the SAME project root:
CORESMITH_PROJECT_ROOT=$RUN_DIR python orchestrator/vscode-ext/serve.py --port 3000
```

Then open <http://127.0.0.1:3000>. The page polls the run's logs, so it
updates live as the daemon advances.

**Remote / headless box** (e.g. a cloud runner): `serve.py` binds
`127.0.0.1` by default. Either forward the port with
`ssh -L 3000:localhost:3000 <host>`, or expose it directly with
`--host 0.0.0.0` (or a `cloudflared tunnel --url http://127.0.0.1:3000`).
The daemon and the webview can run in separate shells as long as both
export the same `CORESMITH_PROJECT_ROOT`.

## Testing

```bash
# Run orchestrator tests
source venv/bin/activate
pytest orchestrator/tests/ -v

# Skip tests requiring live LLM
pytest orchestrator/tests/ -v -m "not live_llm"

# Skip tests requiring Nix/EDA tools
pytest orchestrator/tests/ -v -m "not requires_nix and not e2e"
```

## Further reading

- [docs/AUTHENTICATION.md](docs/AUTHENTICATION.md) — wiring up the Claude CLI (OAuth token, API key, GitHub Codespaces secret)
- [docs/LOCAL-DEV.md](docs/LOCAL-DEV.md) — running and iterating without containers
- [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) — common failures (Yosys version, missing PDK, OpenROAD OOM, …)
- [docs/RUNPOD.md](docs/RUNPOD.md) — hosted runs with a ready-to-paste pod template

## Maintainer

**Tim Balbekov** — balbekov@alum.mit.edu

## Citation

If CoreSmith is useful to you, please cite it:

```bibtex
@software{coresmith2026,
  author  = {Balbekov, Tim},
  title   = {{CoreSmith}: A Prompt to GDS Agentic Flow},
  year    = {2026},
  url      = {https://github.com/facebookexperimental/coresmith}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
