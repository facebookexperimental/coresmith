# SoC stages

## Stages and where they live

| Stage | Where | Flag |
|---|---|---|
| Fabric Resolution: the SoC bus is a generated primitive (pulp `axi` xbar + AXI-Lite/APB bridges), never a hand-written block | `architecture/specialists/fabric_resolution.py`, `fabric/` (FabricSpec, generator, cocotbext-axi TB), `langgraph/rtl_lib/fabric/sv` (vendored SHL-0.51 IP), skill `soc_fabric` | `CORESMITH_FABRIC_RESOLUTION` |
| Contract `timing` object (latency, stall, hold, reset idle) and the Interface VIP generated per edge (cocotb Driver/Monitor/Scoreboard/assertions + `$past` SVA binds) | `architecture/specialists/contract_timing.py`, `langgraph/vip_lib/`, node "Interface VIP" | `CORESMITH_CONTRACT_TIMING_GATE`, `CORESMITH_INTERFACE_VIP`, `CORESMITH_VIP_SVA_BIND` |
| Shell integration: the chip top assembled from the contracts before tier 0 (stubs) and after every tier (real RTL for published blocks); the final top is the same assembly. A tier whose top has design wiring/elaboration errors parks (`shell_not_elaborated`); engine/tool errors are reported, never a design verdict. The contract check is `pass` / `fail` (parks) / `tool_error` (only the engine's own primitives failed) / `unverified` (no elaborator ran: nothing was checked, the reason is recorded) / `inapplicable`: a project that declared NO contracts and NO pins (a bare `blocks.yaml`) has every real port "undeclared" by construction, so its check is recorded `inapplicable` -- never a pass, never a park -- and the real RTL is still judged by the block gate (lint, DV, synthesis) and the integration check. Only `pass` is a pass | `langgraph/shell_integration.py`, nodes `shell_init` / `shell_update`, `integration_snapshots` table | `CORESMITH_SHELL_INTEGRATION`, `CORESMITH_DETERMINISTIC_TOP` |
| uArch phase: every missing uArch spec authored in one session before the first RTL tier. Optionally (`CORESMITH_SYSTEM_MODEL=1`, default off) it also authors, assembles, builds and smokes a SystemC TLM-2.0 LT model of the SoC, announced by `model_authoring` events. The model is a declared input of every module build: `model build` records each block's implementation (`sha`) with its local includes (`deps_sha`) and the contract version, and readiness (`MODEL_*`) compares against the current bytes | `systemc_model/`, agent `systemc_model_generator`, skill `systemc_tlm_lt`, node `uarch_phase`, `models` table, `state_store/builds.py` (`model_identity`) | `CORESMITH_UARCH_PHASE`, `CORESMITH_SYSTEM_MODEL` (default `0`), `CORESMITH_SYSTEMC_HOME` |
| FRD evaluation on the model, scoped: the must-have requirements that DECLARE a model check (`--model-check`) get a verdict from a SystemC harness (`model/frd_eval/`) on the assembled model; the harness's input list (`frd_eval/requirements.json`) holds exactly those, an out-of-scope verdict is a diagnostic, and a requirement with no declared check (Linux boot, DRC) is never demanded of the model nor given a fake pass. `coresmith model eval` is a pure check of the supplied harness; `coresmith harness author` authors one explicitly; the graph authors it only inside the opted-in uArch phase. Verdicts are `model_eval` checks bound to the model + harness digest and the requirement as judged; a declared check must pass and be current for its module to be ready (`MODEL_CHECK_MISSING\|FAILED\|STALE`); a later RTL pass never masks it | `systemc_model/frd_eval.py` (`model_check_inputs`), agent `frd_eval_generator`, `_frd_evaluation` in `pipeline_graph.py`, `stages.module_ready` | `CORESMITH_FRD_EVAL`, `CORESMITH_FRD_EVAL_TIMEOUT_S` (1800), `CORESMITH_FRD_EVAL_REPAIRS` (3, authoring only), `CORESMITH_UARCH_PHASE_GATE` (parks the opted-in phase on build/smoke/FRD failure) |
| Assertion stage: every §4a invariant and contract timing rule must exist as an assertion; phantom "checked by assertion" comments are rejected | `langgraph/assertion_stage.py`, node `assertion_check` | `CORESMITH_ASSERTION_STAGE` (1 / advisory / 0) |
| Recorded module builds: one build per module through the block subgraph on a persistent checkpoint (`coresmith build module`, the pipeline's own fan-out, `restart_block`), recorded in the `builds` table before it starts with the inputs it started from (including the FRD target allocation and the acceptance testbench); every candidate is judged against the required FRD targets from tool receipts (`evaluate_targets`, `build_candidates`); `best` is published only by `block_done` as the committed result of that build, in one transaction with its `completed` record, after the evidence rows are read back, the target evaluation re-checked and the inputs re-checked. Timing false pass: `best` only when sim AND synth AND timing; `abc -D <period>`; `timing_fix` loop with `TimingClosureAgent` | `module_build.py`, `state_store/builds.py`, `state_store/module_targets.py`, `langgraph/target_closure.py`, `pipeline_graph.py` (`init_block_node`, `evaluate_targets_node`, `block_done_node`, `build_block_graph`, `timing_fix_node`), `pipeline_helpers.py`, daemon `/build/*` | `CORESMITH_SYNTH_ABC_DELAY_TARGET`, `CORESMITH_TIMING_FIX(_MAX)` (`CORESMITH_DONE_RESULT_GATE` is ignored) |
| Run state in the DB: leases (process locks), run flags, decisions, interrupts (single-branch resume, 202 while siblings run), LLM slots | `state_store/leases.py`, `state_store/interrupts.py`, `coresmith leases / interrupts` | `CORESMITH_INTERRUPT_WAIT_S`, `CORESMITH_LLM_SLOTS` |
| Operator rulings: additive policy injected into every prompt, resolving parked questions, never touching `inputs/` | `state_store/rulings.py`, `coresmith ruling add`, `POST /rulings`, MCP `add_ruling`, `.coresmith/OPERATOR_RULINGS.md` | — |

## Chip-cycle reading
Architecture → port lock (contracts + timing) → VIP → port-passing shell →
uArch (+ an optional SystemC model) → RTL per block against the VIPs →
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

## The Architect and the CLI state machine

The Architect is the coding agent the human is already talking to (Claude
Code, Codex, ...). It drives the run through the `coresmith` CLI; the engine
never launches, resumes, prompts or stops it, and nothing in the engine
decides a park for it. Line-item verbs (prd/frd/verifier/contract/fabric/
actions), refusal codes and env vars: [CLI_TACTICS.md](CLI_TACTICS.md).

The architecture graph's deterministic work is available as tools the
Architect drives through the CLI; the graph nodes call the same functions
(`harness/tools/`), so both flows populate the same tables
(`state_store/ontology.py`: `artifacts`, `items`, `item_links`, `checks`,
`questions`, `stages`).

| verb | does |
|---|---|
| `coresmith register <kind> <path> [--block b]` | parse + validate + record a PRD/SAD/FRD/ERS/block_diagram/contracts/abi/uarch/arch_model/harness; refuses (exit 1) with stable problem codes (`BD_INSTANCE_UNSPECIFIED`, `BD_REVERSED_DUP`, `CT_WIDTH_MISMATCH`, `CT_FABRIC_PORT_UNREACHED`, `CT_BUS_PHANTOM_FIELD`, ...) |
| `coresmith item list|show|status` | the identified requirements (`PERF-001`, `FR-CPU-2`, `KPI-ISA-1`, `INV-003`, `VAL-007`, `ERS-<block>-n`) with links and checks |
| `coresmith link <from> <to> <rel>` | `derives_from` / `owned_by` / `verified_by` / `cites` / `covers` |
| `coresmith check add <item> <kind> <status>` | a tool verdict (pass / fail / not_testable / skipped / tool_error), hash-bound |
| `coresmith question add|answer|list` | item-scoped questions; `answer --ruling` records a ruling |
| `coresmith stage status|next` | the state machine: `requirements -> decomposition -> interfaces -> uarch -> blocks -> integration -> acceptance -> backend`; `next` refuses with the exact missing item ids; `status` lists `regressed` done stages whose criteria no longer hold. `uarch` is per-module readiness, `blocks` needs a current recorded build per block, and `integration` / `acceptance` / `backend` complete only on the graph's own latest verdict for the current composition (every block's current build + the integrated top's bytes): the integration DV, the validation DV, the backend signoff row |
| `coresmith build module\|targets\|status\|resume\|pause\|abort\|list\|show\|compare\|lineage` | one recorded module build and its ledger (see [CLI_TACTICS.md](CLI_TACTICS.md), *Recorded module builds* and *Module targets*) |
| `coresmith status` | one screen |

Every verb takes `--json`. Registration is content-hashed (`artifacts.version`
bumps on change) and never edits the document.

### Executable models

The SystemC SoC model assembled from per-block implementations is a declared
input of every module build: `coresmith model build` compiles and smokes the
implementations that exist and records a `models` row per block (the
implementation's `sha`, the `deps_sha` over it and its local includes, the
contract version); `coresmith model eval` evaluates the requirements that
declare a model check with the harness the Architect supplied (or
`coresmith harness author` wrote) and records one `model_eval` check per
in-scope verdict, bound to the model + harness digest and the requirement.
A module is ready (`uarch` exit, `build module`) only with its model built
at the current bytes and its declared checks passing and current; a model,
harness or requirement edit re-opens that (`MODEL_STALE`, `MODEL_CHECK_STALE`)
and makes the module's published build stale. Requirements that declare no
model check keep their RTL / chip-level acceptance; a `model_eval` pass never
satisfies the RTL-level evidence the blocks gate demands, and a failing
`model_eval` that is not declared is an advisory (`MODEL_ONLY_FAILED`).

`model/arch/arch_model.json` describes the chip as components (initiators,
targets with address windows / service latency / bandwidth, `both` for DMA-like
blocks, `instances`) and links (latency, bytes/cycle). `coresmith model build
--arch` generates an abstract SystemC performance model (`systemc_model/
arch_model.py`), `model run --arch` smokes it and writes `stats.json`, `model
eval --arch` runs the architecture-level harness, and `coresmith fabric
derive` turns the measured link table into `.coresmith/fabric_spec.json`. The
architecture model remains a tactic no stage requires.

### Building modules and starting the frontend

A block is done only by a recorded build: `coresmith build module <b>` runs
the block subgraph on the build lifecycle's persistent checkpoint once the
shared stages hold and the module is ready; `block_done` publishes `best`
and the build's `completed` record together, bound to the inputs the build
started from and the files it produced. `coresmith run start` is refused
(409 `STAGE_MACHINE_UNUSED` / `STAGE_BEFORE_BLOCKS` / `STAGE_REGRESSED` /
`CLUSTER_FANOUT_UNSUPPORTED`, with the stage, its blockers and advisories)
before any tool preflight, baseline capture or state reset while the stage
machine is before `blocks` or a done stage no longer holds; `--force` only
replaces an existing run (its in-flight pipeline builds are aborted, with
history); a project that never used the stage machine is at `requirements`.
The pipeline's tier loop allocates the same recorded identity for every
block at its init node, reuses a module whose current build is valid instead
of regenerating it, and `pipeline_complete` counts a success only against
its receipt. Inside the graph, `init_tier` records the entry and
`record_entered` refuses to mark an unmet stage done. `run restart-block <b>`
and the MCP `restart_block` are the module build entry; the MCP
`mark_block_passed` override is removed.

Every module build -- every entry -- is judged by the FRD targets the module
owns (`state_store/module_targets.py`): the owned bounded items with a
module-scope cocotb or `eda` verifier, recorded in the build's identity
(`targets` and `acceptance` axes). After synthesis `evaluate_targets` judges
the candidate from tool receipts (the acceptance simulation's results and
measurements, yosys `stat -liberty` area, OpenSTA `report_power`), records a
`build_candidates` row, routes a miss back to the implementation worker with
the gap, and parks a missing measurement (`target_unmeasured`).
`block_done` publishes only a feasible evaluation and stamps the items'
`block_dv` checks with the measured values (CLI_TACTICS.md, *Module
targets*).

### Diagnostics and cluster workers

`coresmith block-status <b>` / `coresmith block-done <b>` (`harness/tools/
block.py`) run the block gate on the on-disk files -- contract conformance,
lint + simulation with the VIPs (coverage), full synthesis on the frozen
flow, pre-layout timing -- stamp a `block_dv` check on every FRD item the
block owns and report; what `block-done` publishes carries `published_by`
and no build id and satisfies no stage (build the implementation with
`build module <b> --seed-rtl <file>` instead). With `CORESMITH_FANOUT=cluster`
(an explicit opt-in) the tier loop would send ONE cluster worker per
`cluster`/`subsystem` group (`process_cluster_node` -> `architect/cluster.py`,
a native session on `CORESMITH_ARCHITECT_PROVIDER`); that mode is refused for
qualified builds (`CLUSTER_FANOUT_UNSUPPORTED` from `build module` and
`run start`) because the workers publish through `block-done`, outside the
module graph. The default block path is the migration.

### Parks

The graph never decides an interrupt: `_resolve_interrupt` only parks (through
the interrupts table). `coresmith state --json` (`interrupts[0].payload`) and
`coresmith interrupts --pending` show the park with its `type`, `block_name`,
`previous_error`, `supported_actions` and guidance; the Architect answers it
with ONE `coresmith resume --interrupt-id <id> --action <supported> [...]
--rationale "..."` (backend parks: `coresmith backend resume`). `actor`
(default `cli`) is recorded in `decisions.actor` and `interrupts.resolved_by`.
Interrupt ids carry the `tier` and the LangGraph task; an answered row stays
`consumed`.
