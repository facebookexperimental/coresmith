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
| FRD evaluation on the model: every FRD requirement (`**ID**: XXX-NNN`) gets a verdict from an agent-authored SystemC harness (`model/frd_eval/`) that boots the mission stimulus on the assembled model through the fabric's `s_cs_tester` port; `fail`, an unanswered must-have or a `not_testable` without reason parks the phase. Report: `model/frd_eval/REPORT.md`, `.coresmith/frd_eval.json` | `systemc_model/frd_eval.py`, agent `frd_eval_generator`, `_frd_evaluation` in `pipeline_graph.py`; FRD prompt requires a `Model check` per requirement | `CORESMITH_FRD_EVAL`, `CORESMITH_FRD_EVAL_TIMEOUT_S` (1800), `CORESMITH_FRD_EVAL_REPAIRS` (3), `CORESMITH_UARCH_PHASE_GATE` (default on: build + smoke + FRD) |
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

## Architect sitting, step 1: the ontology and the state machine (CLI)

The architecture graph's deterministic work is also available as tools the
architect (one long-lived session) drives through the `coresmith` CLI; the
graph nodes call the same functions (`harness/tools/`), so both flows populate
the same tables (`state_store/ontology.py`: `artifacts`, `items`, `item_links`,
`checks`, `questions`, `stages`).

| verb | does |
|---|---|
| `coresmith register <kind> <path> [--block b]` | parse + validate + record a PRD/SAD/FRD/ERS/block_diagram/contracts/abi/uarch/arch_model/harness; refuses (exit 1) with stable problem codes (`BD_INSTANCE_UNSPECIFIED`, `BD_REVERSED_DUP`, `CT_WIDTH_MISMATCH`, `CT_FABRIC_PORT_UNREACHED`, `CT_BUS_PHANTOM_FIELD`, ...) |
| `coresmith item list|show|status` | the identified requirements (`PERF-001`, `FR-CPU-2`, `KPI-ISA-1`, `INV-003`, `VAL-007`, `ERS-<block>-n`) with links and checks |
| `coresmith link <from> <to> <rel>` | `derives_from` / `owned_by` / `verified_by` / `cites` / `covers` |
| `coresmith check add <item> <kind> <status>` | a tool verdict (pass / fail / not_testable / skipped / tool_error), hash-bound |
| `coresmith question add|answer|list` | item-scoped questions; `answer --ruling` records a ruling |
| `coresmith stage status|next` | the state machine: `requirements -> arch_model -> decomposition -> interfaces -> uarch -> model_eval -> blocks -> ...`; `next` refuses with the exact missing item ids |
| `coresmith status` | one screen |

Every verb takes `--json`. Registration is content-hashed (`artifacts.version`
bumps on change) and never edits the document.

### Step 2: the executable SAD (`coresmith model … --arch`, `coresmith fabric derive`)

`model/arch/arch_model.json` describes the chip as components (initiators,
targets with address windows / service latency / bandwidth, `both` for DMA-like
blocks, `instances`) and links (latency, bytes/cycle). `coresmith model build
--arch` generates an abstract SystemC performance model (`systemc_model/
arch_model.py`: generic initiators driven by a scenario body, targets, a fabric
that accounts bytes / utilization / queueing / outstanding depth per link,
energy proxies), `model run --arch` smokes it and writes `stats.json`, `model
eval --arch` has the harness agent write the mission scenario
(`model/arch/frd_eval/`) and records one `model_eval` check per FRD item, and
`coresmith fabric derive` turns the measured link table into
`.coresmith/fabric_spec.json` (masters, slaves, data width from the busiest
link with headroom, outstanding depth) -- the fabric is an output of analysis.
The `arch_model` stage cannot be left until every must-have FRD item has a
model_eval verdict.

### Step 3: the architect sitting (`coresmith architect start|status|stop`)

`orchestrator/architect/session.py` runs ONE `claude -p` session in the project
root (file tools + Bash, stream-json, `--max-turns` per sitting) with the
system prompt `langchain/prompts/architect.md` (the stage contract) plus the
specialist prompts as a reference appendix. Between sittings the runner reads
`coresmith stage status`; it resumes the same session id (`--resume`, cache
warm) with the exact blockers until the run enters `blocks`, is stopped
(`.coresmith/architect/STOP`) or hits `--max-sittings`. Transcripts, prompts
and `status.json` live under `.coresmith/architect/`. With
`CORESMITH_ARCHITECT_SITTING=1` the pipeline's `init_tier` parks
(`architect_stage_pending`) until the stage machine has reached `blocks`.

### Step 4: the block gate as a tool and cluster workers

`coresmith block-status <b>` / `coresmith block-done <b>` (`harness/tools/
block.py`): the gate a block must pass -- contract conformance, lint +
simulation with the VIPs (coverage), full synthesis on the frozen flow,
pre-layout timing -- publishes `best` (the same record `block_done_node`
writes) and a `block_dv` check on every FRD item the block owns; refusals come
with a stage report and tool failures are typed `tool_error`. With
`CORESMITH_FANOUT=cluster` (default when the architect sitting is on) the tier
loop sends ONE cluster worker per `cluster`/`subsystem` group
(`process_cluster_node` -> `architect/cluster.py`: a resumable claude session
with the cluster prompt `cluster_worker.md` + the RTL/TB/timing guidance as
appendix) instead of one subgraph per block; primitives still materialize on
the block path. `block_specs()` now carries `subsystem`, `cluster`,
`instances` and `owns`.

### Step 5: chip lead = the architect resumed

With `CORESMITH_ARCHITECT_SITTING=1`, `_resolve_interrupt` first resumes the
architect session (`architect/consult.py`: the interrupt payload + prior
decisions, answered in the chip-lead decision schema, `--resume` of the
recorded session id) and only falls back to the fresh chip-lead agent when
there is no session or the answer is malformed; action validation, decision
ledger and budgets are unchanged. Consult prompts/decisions are kept under
`.coresmith/architect/consult-N.*`.
