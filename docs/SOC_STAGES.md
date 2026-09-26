# SoC-grade stages (feat/soc-stages)

What the 2026-09-25 SoC benchmark showed, and what changed in the engine.

## Evidence
All 25 CoreSmith blocks passed block DV, but the chip was never assembled;
cross-block timing rules (e.g. a one-cycle snoop response) lived only in
prose and each testbench modelled its neighbour from its own reading; a
block with WNS −27.6 ns had a green `best_result.json`; the interconnect was
hand-written by the RTL agent (ready tied high, phantom assertions); 20 ABI
questions sat unanswered because adding a rulings file to `inputs/` counts
as oracle tampering; no executable model of the SoC existed.

## Stages and where they live

| Stage | Where | Flag |
|---|---|---|
| Fabric Resolution: the SoC bus is a generated primitive (pulp `axi` xbar + AXI-Lite/APB bridges), never a hand-written block | `architecture/specialists/fabric_resolution.py`, `fabric/` (FabricSpec, generator, cocotbext-axi TB), `langgraph/rtl_lib/fabric/sv` (vendored SHL-0.51 IP), skill `soc_fabric` | `CORESMITH_FABRIC_RESOLUTION` |
| Contract `timing` object (latency, stall, hold, reset idle) and the Interface VIP generated per edge (cocotb Driver/Monitor/Scoreboard/assertions + `$past` SVA binds) | `architecture/specialists/contract_timing.py`, `langgraph/vip_lib/`, node "Interface VIP" | `CORESMITH_CONTRACT_TIMING_GATE`, `CORESMITH_INTERFACE_VIP`, `CORESMITH_VIP_SVA_BIND` |
| Shell integration: the chip top assembled from the contracts before tier 0 (stubs) and after every tier (real RTL for published blocks); the final top is the same assembly | `langgraph/shell_integration.py`, nodes `shell_init` / `shell_update`, `integration_snapshots` table | `CORESMITH_SHELL_INTEGRATION`, `CORESMITH_DETERMINISTIC_TOP` |
| uArch phase delivers a SystemC TLM-2.0 LT model of the SoC (per-block models written to generated skeletons, assembled, built, smoked) | `systemc_model/`, agent `systemc_model_generator`, skill `systemc_tlm_lt`, node `uarch_phase`, `models` table | `CORESMITH_UARCH_PHASE`, `CORESMITH_SYSTEM_MODEL`, `CORESMITH_SYSTEMC_HOME` |
| Assertion stage: every §4a invariant and contract timing rule must exist as an assertion; phantom "checked by assertion" comments are rejected | `langgraph/assertion_stage.py`, node `assertion_check` | `CORESMITH_ASSERTION_STAGE` (1 / advisory / 0) |
| Timing false pass: `best` only when sim AND synth AND timing; `abc -D <period>`; `timing_fix` loop with `TimingClosureAgent` | `pipeline_graph.py` (`block_done`, `timing_fix_node`), `pipeline_helpers.py` | `CORESMITH_DONE_RESULT_GATE`, `CORESMITH_SYNTH_ABC_DELAY_TARGET`, `CORESMITH_TIMING_FIX(_MAX)` |
| Run state in the DB: leases (process locks), run flags, decisions, interrupts (single-branch resume, 202 while siblings run), LLM slots | `state_store/leases.py`, `state_store/interrupts.py`, `coresmith leases / interrupts` | `CORESMITH_INTERRUPT_WAIT_S`, `CORESMITH_LLM_SLOTS` |
| Operator rulings: additive policy injected into every prompt, resolving parked questions, never touching `inputs/` | `state_store/rulings.py`, `coresmith ruling add`, `POST /rulings`, MCP `add_ruling`, `.coresmith/OPERATOR_RULINGS.md` | — |

## Chip-cycle reading
Architecture → port lock (contracts + timing) → VIP → port-passing shell →
uArch + SystemC model → RTL per block against the VIPs and the model →
continuous hookup with contract checks → final integration on the same
netlist. Block DV, subsystem, chip -- in that order, with one BFM per
interface instead of one per author.

## Fabric primitive
Declare one block per bus (`kind: primitive`, `fabric: {masters, slaves,
widths}`); the engine renders a SystemVerilog wrapper over the vendored
pulp IP, elaborates it with yosys-slang (already in oss-cad-suite's Yosys)
to plain Verilog-2001, and verifies it with its generated cocotbext-axi
testbench (decode, DECERR, bursts, backpressure, fairness). Re-vendor with
`orchestrator/scripts/vendor_fabric.sh` (pins tags; writes a sha manifest).

## Follow-ups (not in this branch)
- Real-vs-real VIP stimulus on the assembled shell top (today: elaboration
  + per-block contract attribution; the VIPs run in block DV).
- SystemC model as the block golden (transaction server bridge) and
  RTL-vs-model checkpoint comparison in acceptance.
- FlooNoC for many-cluster designs; data-width converters in the fabric.
