# CLI tactics: the SoC state as line items

For the Architect -- the coding agent that talks to the human and drives this
CLI -- and for the engine's own helpers (graph nodes, cluster workers).

## The Architect

The Architect is whatever coding agent the human is already using (Claude
Code, Codex, ...). It reads the human's intent, writes the design collateral,
and advances the run only through `coresmith`. The engine never launches,
resumes, prompts or stops an Architect: there is no `coresmith architect`
verb, no engine-owned Architect loop, no on-call loop and no no-progress counter. Parks
(`coresmith state`, `coresmith interrupts`) wait for the Architect's own
`coresmith resume`. `CORESMITH_ROLE=architect` is identity metadata for the
actions log and restricts nothing.

## Principle

* **`.coresmith/project.sqlite` is the SoC state.** Requirements (PRD/FRD
  items with metric and bounds), verifiers, checks, contract edges and locks,
  the fabric spec, stages and the actions log are rows. Every verb validates
  before it writes and refuses with stable problem codes; a refusal writes
  nothing.
* **Design collateral stays on disk.** RTL, testbenches, SystemC models, TCL,
  and long-form prose (SAD, uArch specs, HW/SW ABI, `DECISIONS.md`) are files
  you write and `coresmith register`.
* **Rendered files are views.** `arch/prd_spec.md`, `arch/frd_spec.md`,
  `.coresmith/prd_spec.json` and `.coresmith/fabric_spec.json` are produced
  from the DB (`coresmith state write`, `frd render`, the fabric verbs). They
  carry a `rendered from project.sqlite` header; never hand-edit one. Change
  the row, re-render.
* **Measurements decide.** PERF/TIME items carry `--metric` and `--min` /
  `--max`; `check add --value` derives pass/fail from those bounds (an explicit
  status that contradicts them is refused), and `block-done` stamps a bounded
  item from the value its testbench recorded, never from the cocotb verdict
  alone (see *Measurements* below).
* **The authoritative check decides an item's status** (see *Item status*
  below); a failed must-have blocks advancing past the current stage -- unless
  its only authoritative verdict is a `model_eval` check, which is an advisory
  for the RTL stages (`MODEL_ONLY_FAILED`). Where a requirement *declares* a
  model check, readiness requires that check to pass on the current model,
  harness and requirement (`MODEL_CHECK_*`); a later RTL pass never masks it.
* **A block is done only by a recorded build.** `coresmith build module <b>`
  runs the engine's block subgraph on a persistent checkpoint; the pass it
  publishes is bound to the inputs it started from and the files it produced
  (`build show`, `build lineage`). A pass written by hand, by `block-done`
  or by an override satisfies no stage, and no knob opens a stage exit.

Every verb takes `--project-root` (or `$CORESMITH_PROJECT_ROOT`) and `--json`.
Run `coresmith <verb> --help` for the exact flags.

## Verbs

| Group | Usage | Does |
|---|---|---|
| Requirements | `prd add "<text>" --id FR-X-n [--kind FR\|KPI\|CON] [--priority P] [--acceptance '<threshold> -- <method>']` | one PRD item (KPIs default must_have) |
| | `prd edit <id> [--text] [--priority] [--acceptance]` / `prd retire <id>` / `prd list [--kind]` | edit, retire, list PRD items |
| | `frd add "<text>" --id PERF-001 [--metric m --min x --max y --unit u] [--priority must_have] [--acceptance ..] [--model-check ..] [--owner <block>] [--derives-from ID,ID] [--section S]` | one FRD item; PERF/TIME need a metric and a bound |
| | `frd edit <id> [same flags] [--text]` | edit; `--min none` clears a bound; `--owner` replaces the owning block |
| | `frd retire <id>` / `frd list [--must] [--kind K] [--unverified]` / `frd show <id>` | retire, list (unverified = must-have with no verifier), show links/verifiers and the latest check per kind (which one decides the status) |
| | `register prd\|frd\|ers <path>` | bulk import of a written document; `register ers` never moves or rewrites an item another artifact owns (`ITEM_OWNED_ELSEWHERE`, its links are still added) and writes `.coresmith/ers_spec.json` (read by `validation_dv`) |
| | `state write [--force]` / `frd render [--out] [--force]` | render the DB-sourced PRD/FRD views (`--force` also overwrites a file-sourced one), `arch/pinout.md`, and the Caravel `prd["pin_map"]` in `.coresmith/prd_spec.json` when a pin carries `--bus` |
| Verification | `frd verifier <id> --kind cocotb\|eda\|python\|systemc\|judge\|manual\|chip [--path tb/..] [--entry test] [--block b] [--args JSON] [--replace]` | bind a verifier (cocotb needs `--path` and `--entry`; `eda` = a quantity the engine measures on the module build's synthesized netlist, `--entry area_um2\|power_mw`, the item needs a bound and a convertible unit; `chip` = a test of the integration/validation testbench, needs `--entry`); re-binding the same (item, path, entry) to another `--block` needs `--replace`. A module-scope cocotb or eda verifier makes a bounded owned item a **module target** (see *Module targets*) |
| | `frd verifiers [<id>] [--block] [--kind]` / `frd verifier-rm <vid>` | list / remove verifiers |
| | `check add <id> <kind> [status] [--value N] [--evidence ..]` | record a verdict; with `--value` on a bounded item the status comes from the bounds (a contradicting status: `CHECK_STATUS_CONFLICT`) |
| | `block-status <b>` / `block-done <b>` | the block gate as a DIAGNOSTIC: `block-done` runs conformance, lint + simulation and synthesis on the on-disk files, stamps `block_dv` per item from cocotb `results.xml` (bounded items: from the recorded measurement) and reports `verified/failed/unverified/deferred/unmeasured_items`; what it publishes carries `published_by` and no build id and satisfies no stage |
| Recorded builds | `build targets <b>` | read-only: the FRD target allocation a build of `<b>` binds (required / advisory / functional / deferred, the measurement binding and conditions of each, the acceptance testbench, the identity digest) and every refusal it would raise (exit 2) |
| | `build module <b> [--seed-rtl f] [--max-attempts N (3)] [--target-clock-mhz M]` | one recorded build of one module through the engine's block subgraph on a persistent checkpoint (`.coresmith/build_checkpoint.db`, thread `build-<id>`); refused before any tool runs unless the shared stages hold and the module is ready (see *Recorded module builds*); completes only when every required FRD target is measured and met (see *Module targets*) |
| | `build status [--build-id id]` / `build resume --build-id id --action A [..]` / `build pause` / `build abort --build-id id [--reason ..]` | the durable record with its live parks and `resumable`; answer a park (using its `supported_actions`) after the recorded inputs are re-validated; cancel the running task (it parks); mark a stopped build aborted so the module is buildable again |
| | `build list [--module b]` / `build show <id>` / `build compare <a> <b>` / `build lineage [--module b]` | the ledger (no daemon needed): rows; one record with its evidence rows and current-pass status; two builds side by side with their input differences and scoped numbers (stage, tool, PDK, clock, power basis); the machine-readable graph lineage (`intended_workflow`) |
| Ports | `contract add <blk>.<port> <blk>.<port> --protocol P [--width N] [--field NAME:W[:MSB:LSB]] [--sideband NAME:W] [--timing k=v] [--bus-param k=v] [--policy ..] [--semantic ..] [--spec JSON] [--edge-id ..]` | one contract edge, whole-document validation before the write |
| | `contract set <edge> <dotted.key> <json-value>` / `contract rm <edge>` | edit / remove an edge; locked edges need `--unlock --reason "<why>"`, which licenses that one write -- the edge stays locked |
| | `contract lock [--block b]` / `contract unlock [--block b] --reason ..` / `contract show <edge>` / `contract list [--block] [--locked]` | locks and inspection; `contract unlock` is the only way to leave an edge unlocked |
| | `register contracts <path> [--unlock --reason ..]` | bulk import; refuses to move a locked edge without `--unlock` (edges stay locked after it) |
| | `pin add <name> --dir in\|out\|inout [--width N] [--from <blk>.<port>] [--kind signal\|clock\|reset\|power] [--bus io --msb N --lsb N [--oe <sig>]]` | one chip pin: a block port that leaves the chip (a port is a contract edge OR a pin); `clk`/`rst_n` are `--kind clock\|reset` pins without `--from` (the shell's clk/rst nets, fanned to every block) |
| | `pin set <name> <field> <value>` / `pin rm <name>` / `pin list [--json]` | edit (`dir width from block port bus msb lsb oe kind`; `none` clears) / remove / the pinout |
| | `pin lock` / `pin unlock --reason ..` | pins lock with the edges when `interfaces` completes; a locked pin changes only with `--unlock --reason` (one write, it stays locked) |
| | `vip generate` / `shell assemble` | render the per-edge VIPs / assemble the chip top from the contracts and the pins: the top's ports ARE the pins; a block port on no edge and no pin is refused (`SHELL_UNDECLARED_PORT`) |
| Fabric | `fabric init --name F [--data-width] [--addr-width] [--master NAME]... [--slave NAME:PROTO:BASE:SIZE]... [--param k=v] [--replace]` | create the fabric row |
| | `fabric master add\|rm NAME [--protocol] [--param k=v]` / `fabric slave add NAME --protocol P --base 0x.. --size 0x.. [--param]` / `fabric slave rm NAME` | ports |
| | `fabric set <param> <value>` (`data_width`, `slave.<n>.base`, ...) / `fabric show` / `fabric params [fabric\|master\|slave] [proto]` / `fabric rm --name F` | edit, inspect, parameter schema, delete |
| | `fabric derive [--headroom 2.0] [--dry-run]` | FabricSpec from the architecture model's measured link table |
| Stage machine | `stage status` / `stage next` | `requirements -> decomposition -> interfaces -> uarch -> blocks -> integration -> acceptance -> backend`; `next` refuses with the missing items and prints advisories (`MODEL_ONLY_FAILED`) that never block. `status` also lists `regressed`: a done stage whose criteria no longer hold on re-evaluation (nothing moves past it). `uarch` is per-module readiness; `blocks` needs a current recorded build per block; `integration`, `acceptance` and `backend` need the graph's own latest verdict for the current composition (see *Stage exits*) |
| Executable models | `model build|run|eval|register [--arch]` / `model author --block <b>` / `harness author [--arch]` | the SystemC reference model is a declared input of every module build (`model build` records it); `model eval` evaluates exactly the requirements that declare a model check; checks never author (see *Executable models* below) |
| | `item list\|show\|status`, `link <from> <to> <rel>`, `unlink <from> <to> <rel>`, `question add\|answer\|show\|list`, `ruling add\|list\|revoke`, `interrupts [--resolve ID --action A]`, `status` | the ontology and one-screen status; question/ruling ids print and parse as `Q3` / `R3` (plain `3` also works), `--resolve` takes the full id or a unique prefix; `link` refuses a `block:<name>` that is not a registered block |
| | `state check` | rows-vs-disk audit (see *state check* below) |
| Log & replay | `actions [--limit N \| --all] [--since ID]` | every verb run against this project (argv, rc, summary, actor); `--json` reports `total` and `omitted` when the limit cut rows |
| | `actions --script` | the log as a shell script, one `coresmith <argv>` per line (`shlex`-quoted; refused rows marked `# rc=N`; `actions` rows skipped) |

## Refusal codes

Refused verbs exit non-zero (FRD/PRD, `CHECK_STATUS_CONFLICT`,
`LINK_UNKNOWN_BLOCK` and `ROLE_FORBIDDEN` refusals exit 2; contract refusals
(`CT_*`), pin refusals (`PIN_*`) and `FABRIC_INVALID` exit 1) and write nothing.

| Code | Meaning |
|---|---|
| `FRD_NO_BOUNDS` | a PERF/TIME item without `--min`/`--max` |
| `FRD_NO_METRIC` / `FRD_BAD_BOUND(S)` | a bound without `--metric`; non-numeric bound or `min > max` |
| `FRD_BAD_ID` / `PRD_BAD_ID` / `PRD_BAD_KIND` / `FRD_BAD_REF` | malformed ids or `--derives-from` references |
| `FRD_EXISTS` / `PRD_EXISTS` / `NO_ITEM` / `NO_VERIFIER` | add of an existing live id (use `frd edit` / `prd edit`); edit/show of a missing item or verifier |
| `VERIFIER_REBOUND` | the (item, path, entry) verifier is bound to another block; `--replace` to rebind (`VERIFIER_BLOCK_NOT_OWNER` is the advisory when the block does not own the item) |
| `CHECK_STATUS_CONFLICT` | `check add ... <status> --value N` where the bounds derive the other status; nothing written |
| `LINK_UNKNOWN_BLOCK` | `link` to `block:<name>` that is not a registered block |
| `ITEM_OWNED_ELSEWHERE` | (warning) `register` met ids another artifact owns; they were not moved/rewritten |
| `CT_BAD_PROTOCOL` / `CT_UNKNOWN_BLOCK` | `contract add` with a protocol family outside `axi_stream srdy_drdy axi4 axi_lite apb req_resp mem_write valid_only static`, or (once blocks are registered) an endpoint block that is not one (an endpoint named like the top -- `chip`, `chip_top`, `top`, `soc_top`, the declared top -- gets the hint "chip I/O is declared with `coresmith pin add`"); `CT_OFF_DIAGRAM` is the warning for an edge between blocks the diagram does not connect |
| `PIN_EXISTS` / `PIN_LOCKED` | `pin add` of an existing name (use `pin set`); a write to a locked pin (or an add to a locked pin set) without `--unlock --reason` |
| `PIN_UNKNOWN_BLOCK` / `PIN_UNKNOWN_PORT` | `--from` names a block that is not registered / a port its diagram `interfaces` rows do not list (a block without interface rows takes any port) |
| `PIN_DOUBLE_DRIVEN` | the `--from` port is on a contract edge, or another pin already exposes it (at `shell assemble`: a pin port on an edge net) |
| `PIN_NO_SOURCE` / `PIN_BAD_NAME\|DIR\|KIND\|WIDTH\|BUS\|FROM` | a `signal` pin without `--from`; a malformed field (`--bus` needs `--msb`/`--lsb` spanning `--width`) |
| `SHELL_UNDECLARED_PORT <blk.port>` / `SHELL_BOUNDARY_MISMATCH` | `shell assemble` (and the `interfaces` stage): a block port on no edge and no pin; the shell's boundary differs from the pin set (extra/missing listed) |
| `PINS_MISSING` | stage blocker at `interfaces`: no pins declared (`coresmith pin add ...`) |
| `MUST_HAVE_FAILED` | stage blocker: a must-have item whose authoritative check (block_dv or higher) failed; never a reason to refuse the build that repairs it |
| `MODEL_ONLY_FAILED` | advisory (never a blocker): a must-have whose only authoritative verdict is a failing `model_eval` check; an RTL-level check decides |
| `STAGE_REGRESSED` | a done stage's structural criteria no longer hold (`blocked_by` says which); stage advancement and pipeline start refuse until it is re-established; module builds still allow repairs when only the implementation shell fails |
| `FRD_TARGET_INVALID` / `FRD_TARGET_UNBOUND` / `FRD_TARGET_OWNER_MISMATCH` / `ACCEPTANCE_TB_MISSING` / `ACCEPTANCE_TEST_MISSING` / `ACCEPTANCE_TB_AMBIGUOUS` | module readiness from the FRD target allocation (`build targets`, `build module`, the `uarch` exit): an owned bounded item with a non-finite bound, min > max, no metric, a unit its eda measurement cannot convert to, or a verifier condition (`--args`) the engine does not apply; a required target with no module-scope measurement binding; a verifier bound to the module for an item another block owns, or a bounded item with several owners; an acceptance file the module's cocotb verifiers name does not exist / does not define a bound test / two acceptance files would run under one cocotb module name |
| `FRD_TARGET_UNMEASURABLE` / `BAD_MAX_ATTEMPTS` | `build module` only: a REQUIRED target's measurement cannot run in the daemon's environment (area: no liberty or `CORESMITH_SYNTH_GENERIC`; power: also no reachable OpenSTA; an advisory target is reported unmeasured, never refused); `--max-attempts` < 1 |
| `UARCH_MISSING` / `TARGET_UNBOUND` / `TARGET_INVALID` / `MODEL_MISSING` / `MODEL_INVALID` / `MODEL_NOT_BUILT` / `MODEL_STALE` / `MODEL_CHECK_MISSING` / `MODEL_CHECK_FAILED` / `MODEL_CHECK_STALE` / `WORKER_UNBOUND` | module readiness (the `uarch` exit, `build module`): no registered uArch spec; no (valid) bound target; the reference model is absent / its includes unreadable / not built+smoked / its `models` row is for other bytes or headers, an older contract version, or predates dependency tracking; a declared model check has no verdict / a failing one / one bound to another model, harness or requirement text; `.coresmith/env` does not declare the provider and its model selector |
| `BLOCKS_UNPUBLISHED` / `BLOCK_NOT_BUILT` / `PRIMITIVE_UNMATERIALIZED` / `PRIMITIVE_NOT_BUILT` / `OWNED_ITEM_UNVERIFIED` / `BOUNDED_ITEM_UNMEASURED` | the `blocks` exit: no published pass; a pass that names no recorded build, a build that is not completed, inputs changed since or an implementation edited after it; a primitive not materialized / materialized without a graph build; an owned must-have without an RTL-level pass / without its measured value |
| `COMPOSITION_NOT_BUILT` / `INTEGRATION_NOT_ELABORATED` / `INTEGRATION_STUBS` / `INTEGRATION_CANDIDATE_STALE` / `INTEGRATION_SNAPSHOT_STALE` / `INTEGRATION_DV_MISSING` / `VALIDATION_DV_MISSING` / `VALIDATION_DV_STALE` / `BACKEND_SIGNOFF_MISSING` | the exits past `blocks`: a block without a current build; the real-RTL top not elaborated / with stubs (or, for an adopted candidate, a registered module whose bound HDL top is not in the elaborated hierarchy) / an adopted candidate that no longer validates (re-adopt it) / older than a build; the graph's latest chip / validation DV verdict of this run is not a pass naming the current composition (or predates the integration DV); the latest backend signoff row is not `ppa_ok` for this composition |
| `ARCHITECTURE_NOT_READY` / `MODULE_NOT_READY` / `UNKNOWN_MODULE` / `BUILD_IN_FLIGHT` / `SEED_MISSING` / `SEED_NOT_BOUND` / `SEED_NOT_ALLOWED` / `CLUSTER_FANOUT_UNSUPPORTED` | `build module` refusals (exit 2 with `--json`; the body lists `blocked_by`): the shared stages / the module not ready; unknown block; another build of the module is dispatched, running or parked (resume, wait or `build abort`); the seed does not exist / is not one of the bound target's sources / the block is a primitive; `CORESMITH_FANOUT=cluster` (its publications bypass the module graph) |
| `BUILD_RUNNING` / `PIPELINE_RUNNING` / `BACKEND_RUNNING` / `WORKSPACE_BUSY` | a graph is driving the workspace: nothing else starts or resumes (and `run start\|resume\|continue\|restart-node`, `backend start\|resume` are refused while a build runs, in this process or -- through the `graph:build` lease -- in another one); `WORKSPACE_BUSY`: another process is inside its launch section (`graph_launch` lease; `coresmith leases`), retry |
| `BUILD_STALE` | `build resume`: the project no longer matches the inputs the parked build recorded (`blocked_by` lists the axes); start a new build |
| `BUILD_TERMINAL` / `UNKNOWN_BUILD` / `NO_SUCH_INTERRUPT` / `NOTHING_TO_RESUME` / `ACTION_UNSUPPORTED` / `NOT_A_MODULE_BUILD` | `build resume\|abort\|show`: the build is already terminal; no such build / parked interrupt; nothing parked and nothing scheduled; the action is not in the park's `supported_actions`; a pipeline-dispatched row (resume the pipeline instead) |
| `PERSISTENCE_FAILED` / `INPUTS_CHANGED` / `NO_GRAPH_THREAD` / `THREAD_MISMATCH` | `block_done` (in the build's `error` / the block result): the required rows or the produced files are missing, or the commit failed (status `failed_persistence`); the inputs changed while the build ran (status `failed`); the node ran outside a checkpointed graph thread / on another thread than the recorded one |
| `FRD_TARGETS_NOT_MET` (`FRD_TARGETS_NOT_EVALUATED` / `FRD_TARGETS_STALE`) | `block_done` of a build that binds FRD targets: the published attempt has no recorded target evaluation, its evaluation is not `feasible`, or its implementation files or any receipt it was judged by (simulation results/measurements, synthesis report, netlist, SDC, power report, liberty, acceptance files) changed after the evaluation (status `failed`, nothing published) |
| `target_unmeasured` / `acceptance_changed` (parks) | a required target or required acceptance test has no measurement for this candidate (`retry` re-measures, `abort` ends the build as incomplete); the acceptance testbench's bytes changed during the build (`restore` puts the recorded oracle back, keeping the change as `<file>.changed-<build>`; `abort`) |
| `MANUAL_COMPLETION_REMOVED` | MCP `mark_block_passed`: the manual completion override is gone; `restart_block` / `build module --seed-rtl` build the on-disk implementation |
| `HARNESS_MISSING` / `HARNESS_BUILD_FAILED` / `HARNESS_RUN_INCOMPLETE` | `model eval`: no harness supplied (exit 2, `required` names the path) / the supplied harness does not compile (exit 1, `build_log`) / did not print `FRD_EVAL_DONE` (exit 1, `run_log`); nothing is authored or repaired |
| `SOC_MODEL_NOT_ASSEMBLED` / `ARCH_MODEL_NOT_BUILT` / `UNKNOWN_BLOCK` / `PRIMITIVE_BLOCK` / `NO_BLOCKS` | model/harness verbs: missing or invalid input (exit 2, `required`/`hint`) |
| `MODEL_REFINE_REMOVED` | `model refine` (implicit authoring behind a check) is gone (exit 2); use `model build`, `model author`, `model eval`, `harness author` |
| `HARNESS_MISSING` / `HARNESS_BUILD_FAILED` / `HARNESS_RUN_INCOMPLETE` / `HARNESS_RUN_FAILED` | `model eval`: no `frd_eval/*.cpp` (exit 2, `required`); the harness does not compile (exit 1, `build_log`); it did not print `FRD_EVAL_DONE` (exit 1, `run_log`); it printed `FRD_EVAL_DONE` but exited non-zero (exit 1, `rc`, no checks recorded) |
| `FRD_MISSING` / `FRD_NO_REQUIREMENTS` | `model eval` / `harness author`: no `arch/frd_spec.md`; an FRD with no `**ID**: XXX-NNN` requirement blocks (exit 2, `required`) |
| `ARCH_SPEC_MISSING` / `ARCH_SPEC_INVALID` / `ARCH_MODEL_NOT_BUILT` | `model build\|run\|eval\|register --arch`: no `model/arch/arch_model.json`; it does not parse or validate (`problems`); the model is not built (exit 2, `required`) |
| `ARCH_MODEL_BUILD_FAILED` / `ARCH_MODEL_RUN_FAILED` | `model build\|run --arch`: compiler diagnostics / the smoke scenario failed (exit 1, `log`); only a missing toolchain is `tool_error` (exit 3) |
| `STAGE_MACHINE_UNUSED` / `STAGE_BEFORE_BLOCKS` / `STAGE_REGRESSED` | `POST /run/start` (409), before any preflight, baseline capture or reset: the project has no stage rows (it starts at `requirements`; nothing is exempt); the stage machine has not reached `blocks` (the body lists the stage, its blockers and advisories); a done stage no longer holds. `run start --force` only replaces an existing run; it never bypasses these |
| `ROLE_FORBIDDEN` | the verb is outside `$CORESMITH_ROLE`'s allow-list (exit 2) |
| `FRD_BAD_VERIFIER_KIND` / `FRD_VERIFIER_INCOMPLETE` | unknown verifier kind; a cocotb verifier without `--path`/`--entry` |
| `FRD_FILE_SOURCED` | `state write`/`frd render` would overwrite a registered hand-written FRD (use `--force`) |
| `FRD_UNVERIFIED`, `FRD_NO_ACCEPTANCE`, `REQ_UNOWNED`, `FRD_OWNER_UNKNOWN`, `FRD_REF_UNKNOWN` | advisories (warnings) printed after a successful write (a `--model-check` line is optional prose, never demanded) |
| `CT_LOCKED` | the edge is locked; repeat with `--unlock --reason "<why>"` |
| `CT_MISSING_EDGE`, `CT_WIDTH_MISMATCH`, `CT_FABRIC_PORT_UNREACHED`, ... | whole-document contract validation |
| `FABRIC_INVALID` | the FabricSpec does not validate (e.g. a slave size that is not a power of two) |
| `FABRIC_AMBIGUOUS` / `FABRIC_NOT_FOUND` / `FABRIC_EXISTS` / `FABRIC_PORT_NOT_FOUND` | several fabrics and no `--name`; unknown fabric or port; `init` without `--replace` |
| `requirements_not_registered` | `POST /run/start` (409): missing `PRD_NOT_REGISTERED`, `FRD_NOT_REGISTERED` or `FRD_NO_MUST_HAVE` |

## Environment

| Variable | Default | Effect |
|---|---|---|
| `CORESMITH_REQUIRE_REQUIREMENTS` | `1` | `/run/start` refuses (409) without a registered PRD and an FRD with a must-have item; `0` disables |
| `CORESMITH_SKIP_ARCH_WARN` | unset | `1` bypasses the requirements gate and the skipped-architecture warning (batch evaluation) |
| `CORESMITH_ROTATE_EVENTS` | `1` | `/run/start` rotates `pipeline_events.jsonl`; `0` truncates it (old behaviour) |
| `CORESMITH_CONTRACT_AUTOLOCK` | `1` | leaving the `interfaces` stage locks every contract edge; `0` disables |
| `CORESMITH_BLOCK_DV_BLANKET` | unset | `1` makes `block-done` pass every owned item (old behaviour) instead of per-item stamping |
| `CORESMITH_STAGE_FAIL_BLOCKS` | `1` | every stage is blocked by `MUST_HAVE_FAILED`; `0` disables |
| `CORESMITH_MEASUREMENTS` | set by every engine block simulation | the file `orchestrator.harness.measure.record` appends to: `sim_build/<block>/measurements.jsonl`, emptied before each run (default outside the engine: `./measurements.jsonl`) |
| `CORESMITH_CLOCK_MHZ` | set by a module build's simulation | the build clock, for an acceptance test that reports a time-based rate (cycle-based metrics need none) |
| `CORESMITH_ROLE` | unset | `worker` \| `watchdog`: restrict the CLI to that role's verbs; `architect`: identity only, no restriction (unset: no restriction) |
| `CORESMITH_SHELL_INFER_BOUNDARY` | `0` | `1`: `shell assemble` turns ports on no edge into boundary ports even when pins are declared, and the `interfaces` stage uses the old `SHELL_NO_BOUNDARY` check instead of `PINS_MISSING` (a project with no pin rows always infers) |
| `CORESMITH_MODEL_EVAL_REQUIRE_VALUE` | `1` | a bounded item's `model_eval` verdict must carry the measured value (recorded `tool_error`, item stays `open` otherwise); `0` restores verdict-only judging |
| `CORESMITH_DONE_RESULT_GATE` / `CORESMITH_STAGE_BLOCKS_GATE` | -- | no effect: `best` is published only by the graph's `block_done` as the committed result of a recorded build, and the `blocks` exit has no switch (`=0` logs one warning) |
| `CORESMITH_PPA_GATE` | profile (strict: `1`) | the synthesis PPA gate (pre-layout OpenSTA timing, budgets). A build completes only with its verdict: with the gate off the synthesis row carries no `ppa_ok` and `block_done` refuses to publish (`PERSISTENCE_FAILED: ... CORESMITH_PPA_GATE must be on`) |
| `CORESMITH_LINE_COV_GATE` / `CORESMITH_LINE_COV_FLOOR` | `1` / `70` | the line-coverage floor on block DV. A build completes only with MEASURED closure (a finite percentage against the floor, passed): a coverage row that says the gate was disabled, no `coverage.dat` was produced, `verilator_coverage` was absent or nothing was annotated is an unavailable measurement and `block_done` refuses to publish; there is no diagnostic reason that makes required closure inapplicable |
| `CORESMITH_SYNTH_GENERIC` | unset | generic synthesis (no liberty); part of every build's recorded tooling context, so flipping it makes earlier passes stale |
| `CORESMITH_ARCHITECT_PROVIDER` | follows `CORESMITH_LLM_PROVIDER` | historical name: `claude` \| `opencode` \| `codex`, the CLI the engine's cluster workers (`CORESMITH_FANOUT=cluster`) run on. It selects nothing for the Architect, which is outside the engine |
| `CORESMITH_ARCHITECT_MODEL` | per runner | historical name: the cluster workers' model; opencode falls back to `CORESMITH_OPENCODE_MODEL`, `CORESMITH_MODEL`, the `CORESMITH_OPENCODE_ENDPOINT` default |
| `CORESMITH_ARCHITECT_SITTING` / `CORESMITH_ARCHITECT_ONCALL` / `CORESMITH_ARCHITECT_NO_PROGRESS_SITTINGS` | -- | removed: no reader. The engine-owned Architect loop, on-call loop and no-progress stop are gone |
| `CORESMITH_FANOUT` | `block` | `cluster`: one native cluster-worker session per `cluster`/`subsystem` instead of a subgraph per block. Refused for qualified builds (`CLUSTER_FANOUT_UNSUPPORTED` from `build module` and `run start`): a worker's `block-done` publication bypasses the module graph; the migration is the default block path |
| `CORESMITH_SYSTEM_MODEL` | `0` | `1`: the frontend's uArch phase authors, builds and smokes the SystemC SoC model (announced by `model_authoring` events); default: the frontend constructs no model author -- the Architect runs `model build` / `model author` / `model eval` when a model is worth having |
| `CORESMITH_ACTOR` | set by cluster workers | who runs the CLI when no role says so (`worker:<cluster>`): `actions.actor` and `question add`'s `asked_by` |
| `CORESMITH_REVISE_PROSE_TARGETS` | unset | `1`: block names mentioned in a revise's `--feedback` become targets (old). Default: targets = `--affected-blocks` + `--block-actions` keys; `{"x": "keep"}` protects a block |
| `CORESMITH_INTERRUPT_ID_LEGACY` | unset | `1`: interrupt ids without the `tier` / task-round component, and an answered row re-opened on re-execution (old) |
| `CORESMITH_CHIP_ITEM_CHECKS` | `1` | chip-level stamping (`integration_dv` / `validation_dv` from the chip TB, `synth` cells, `sta` WNS); `0` disables |
| `CORESMITH_INTEGRATION_CHECK_PARK` | `1` | a clean `integration_check` parks for `accept` (`retry`, `abort`); `0` advances on its own (old) |
| `CORESMITH_SCORECARD_PUBLISHED_DV` | `1` | the signoff scorecard takes the newer of the latest `dv_results` row and the published `dv_best`; `0` = the `dv_results` row only (old) |
| `CORESMITH_DAEMON_KEEP_ROLE` | unset | `1`: `daemon start` keeps `CORESMITH_ROLE` / `CORESMITH_ACTOR` for the daemon and its children (old); default clears them |
| `CORESMITH_BACKEND_NATIVE_FALLBACK` | `1` | a `*-nix.sh` OpenROAD wrapper on a host without `nix` falls back to a native `openroad` (PATH, then `~/openroad-src/build/bin`); `0` keeps the wrapper |
| `CORESMITH_PNR_DEADLINE_S` | scaled | P&R worker deadline; default 1800 s + 900 s per mm^2 of die beyond 1 + 120 s per macro (cap 6 h); `CORESMITH_PNR_TIMEOUT` still honoured |
| `CORESMITH_PNR_THREADS` | all CPUs | `set_thread_count` written as the first line of the generated P&R script |
| `CORESMITH_GATE_SIM_MAX_CYCLES` | unset (no cap) | the chip gate-sim replays the whole recorded reference within `CORESMITH_GATE_SIM_TIMEOUT_S`; a positive value caps it (`200000` = old default); a replay over the time budget is `bounded`, not a fail |

## Measurements (bounded items)

At the model rank the same rule holds: the FRD harness prints the measured
number for every bounded item (`FRD_EVAL {"id": "PERF-001", "status": "pass",
"value": 207, "unit": "cycles", ...}` or a `VALUE PERF-001 207 cycles` line);
the value derives pass/fail from the bounds and is stored on the `model_eval`
check. A bounded `pass`/`fail` without a number is recorded `tool_error` ("no
measured value from the model harness") and leaves the item `open`; model
checks never satisfy an RTL-level requirement either way.

A PERF/TIME item with a metric and `--min`/`--max` is decided by a number, not
by a passing test. The testbench that verifies it records the measurement:

```python
from orchestrator.harness.measure import record   # stdlib only, no cocotb import
record("PERF-003", frames_per_s, unit="fps", test="test_throughput_b2b")
```

`record` appends `{"item", "value", "unit", "test", "ts"}` as one JSON line to
`$CORESMITH_MEASUREMENTS`, or to `./measurements.jsonl` when that is unset.
Block simulations run as `make -C <root>/sim_build/<block>`, so the cocotb
working directory is `sim_build/<block>/` and both resolve to the canonical
**`<root>/sim_build/<block>/measurements.jsonl`**. `block-done` deletes that file
before it simulates (a stale number never decides), exports
`CORESMITH_MEASUREMENTS` to it, and then, per owned bounded item whose bound
cocotb test passed: a measurement of the item (the one recorded by that test,
else the last one) becomes a `block_dv` check with `value` -- pass/fail from the
bounds, test and unit in the evidence; no measurement is `tool_error`
("no measurement recorded for a bounded item; TB must call
harness.measure.record") and the item is listed in `unmeasured_items`. A failed
test is a fail; unbounded items keep the cocotb verdict. Record in the test that
the verifier names, and record whether or not the value meets the bound -- the
test need not assert it.

### Chip-level numbers

The integration and validation testbenches use the same convention: the chip
simulation exports `CORESMITH_MEASUREMENTS=sim_build/<integration|validation>/measurements.jsonl`
and, after the run, every item with a `chip` verifier (or any verifier whose
`--path` is that chip TB) gets an `integration_dv` / `validation_dv` check from
its `results.xml` and recorded measurement, exactly as `block-done` stamps
`block_dv` (a path-less `chip` verifier is stamped by whichever chip TB holds its
test). After flat synthesis the backend records a `synth` check with the cell
count on bounded chip-level items whose metric is `cells`, and after chip STA a
`sta` check with the WNS on items whose metric is `wns_ns`/`ns` (items owned by a
block other than the top are never stamped with chip numbers).

## Item status

Checks are ranked by kind (`CHECK_KIND_RANK` in `state_store/ontology.py`):
`model_eval`=1 < `block_dv`=2 < `integration_dv`=3 < `validation_dv`=4 =
`acceptance`=4 < `signoff`=5; any other kind (`sta`, `synth`, ...) ranks 2. An
item's status is recomputed on every `check add`: the latest check of the
highest-ranked kind present decides (`pass` -> verified, `fail` -> failed,
`not_testable`; `skipped`/`tool_error` at that rank -> open). So a model pass
never overrides an RTL fail, an RTL pass after an RTL fail is a pass, and an
informative lower-rank number cannot flip a higher-rank verdict. Waived and
retired items are left alone. `frd show` / `frd list` print the latest check of
every kind; every stage's `stage status`/`stage next` lists `MUST_HAVE_FAILED`
for items whose authoritative check is `block_dv` or higher, and the advisory
`MODEL_ONLY_FAILED` (never a blocker) for items whose only authoritative verdict
is a failing `model_eval` check -- the item keeps its derived `failed` status
until an RTL-level check supersedes it.

## Roles

`CORESMITH_ROLE` scopes the CLI (checked in `bin/coresmith` before dispatch --
daemon-lifecycle verbs included -- and in the harness `_run` wrapper; a refused
verb exits 2 with `ROLE_FORBIDDEN <role> may not run <verb>`, and the actions
log records the role as `actor`; a refusal is logged too, rc 2). Unset: no
restriction; an unknown role may run nothing. The role scopes a shell, never
the engine: `daemon start` (and the daemon itself) drops `CORESMITH_ROLE` /
`CORESMITH_ACTOR`, so a watchdog-started daemon's agents are never refused. The
daemon-client verbs (`daemon`, `run`, `resume`, `backend`) are logged in
`actions` too (rc, first error line as summary).

| Role | May run |
|---|---|
| `watchdog` | `daemon`, `run start\|pause`, `state` (incl. `state check`), `stage status`, `question list`, `interrupts`, `leases`, `actions`, `resume`, `logs`, `status` |
| `worker` | `block-status`, `block-done`, `verify`, `frd verifier\|verifiers\|show\|list` (and bare `frd`), `check add`, `question add`, `constraints`, `blocks`, `results`, `contracts`, `dv-status`, `ppa`, `coverage`, `status`, `actions` |
| `architect` | everything (identity metadata only: the Architect starts the daemon and the frontend itself) |

## state check

`coresmith state check` audits the rows against the disk (read-only) and exits
1 when any error is found (warnings alone exit 0). It knows the stage: before
`blocks` (stage machine in use), `BLOCK_RTL_MISSING`, `BLOCK_TB_MISSING` and
`PRIMITIVE_UNMATERIALIZED` are expected and reported as `info`:

| Code | Meaning |
|---|---|
| `BLOCK_RTL_MISSING` / `BLOCK_TB_MISSING` | a block row's `rtl_target` / `testbench` file does not exist |
| `PRIMITIVE_UNMATERIALIZED` | a primitive block row with an empty `rtl_target` |
| `ARTIFACT_MISSING` / `ARTIFACT_STALE` | a file-sourced artifact's file is gone / changed since it was registered (re-register it) |
| `VIEW_MISSING` | a DB-sourced artifact's rendered view (`arch/frd_spec.md`, `arch/prd_spec.md` + `.coresmith/prd_spec.json`, `.coresmith/interface_contracts.json`, `.coresmith/pins.json`) or, with an ERS registered, `.coresmith/ers_spec.json` is missing |
| `UARCH_UNREGISTERED` | `arch/uarch_specs/<b>.md` exists but no `uarch:<b>` artifact |
| `FRD_UNVERIFIED` | (warning) a must-have FRD item with no verifier |
| `PINS_SHELL_MISMATCH` | (warning) the latest shell's boundary differs from the pins (re-run `shell assemble`) |

## Worked example: a PERF item from bound to verdict

```bash
coresmith prd add "Sustain 60 fps at 1080p" --id KPI-FPS-1 --kind KPI \
    --acceptance '>= 55 fps -- throughput.py on the reference clip'
coresmith frd add "Encoder sustains 60 fps at 1080p" --id PERF-001
#   REFUSED FRD_NO_BOUNDS: a PERF item must be measurable
coresmith frd add "Encoder sustains 60 fps at 1080p" --id PERF-001 \
    --metric fps --min 55 --unit fps --priority must_have \
    --acceptance "mean fps over 300 frames >= 55" \
    --model-check "arch model frame cycle count" \
    --derives-from KPI-FPS-1 --owner enc_core
#   OK fps[55..]; advisory FRD_UNVERIFIED
coresmith frd verifier PERF-001 --kind cocotb \
    --path tb/cocotb/test_enc_core.py --entry test_throughput_1080p
coresmith frd list --unverified            # (no FRD items)
# test_throughput_1080p calls harness.measure.record("PERF-001", fps, unit="fps", test="test_throughput_1080p")
coresmith block-done enc_core              # block_dv for PERF-001 = the recorded fps against min 55
coresmith check add PERF-001 integration_dv --value 58.3      # pass (fps min 55); outranks block_dv
coresmith check add PERF-001 integration_dv pass --value 41   # REFUSED CHECK_STATUS_CONFLICT -- the bound decides
coresmith state write                      # re-render arch/frd_spec.md for the next agent
coresmith actions --script > replay.sh     # the whole session, replayable
```

## Ports: edges and pins

Every block port is exactly one of: an end of a contract edge (`contract
add`), or a chip pin (`pin add ... --from <block>.<port>`). The shell top is
assembled from both: edges become nets, pins become the top's ports (a stub
block grows the port its pin names), `clk`/`rst_n` pins of kind
`clock`/`reset` name the nets fanned to every block's clk/rst. The
`interfaces` stage needs pins (`PINS_MISSING`) and a shell whose boundary
equals them; leaving it locks edges and pins together.

```bash
coresmith pin add clk      --dir in  --kind clock
coresmith pin add rst_n    --dir in  --kind reset
coresmith pin add uart_rx  --dir in  --from uart.uart_rx
coresmith pin add uart_tx  --dir out --from uart.uart_tx
coresmith pin add gpio_in  --dir in  --width 8 --from gpio.gpio_in
coresmith pin add gpio_out --dir out --width 8 --from gpio.gpio_out
coresmith shell assemble          # boundary_ports=6, elaborated=True
coresmith stage next              # interfaces DONE; 8 edges and 6 pins locked
coresmith state write             # arch/pinout.md
```

## Executable models (a declared input of every module build)

The SystemC SoC model assembled from per-block implementations is part of
what a module is built from: `uarch` completes, and `build module <b>` runs,
only when the module's `model/<b>_model.cpp` (or the path its `models` row
registers) is built and smoked by `model build` at its current bytes --
implementation AND the local headers it includes -- and at the current
contract version, and when every requirement the module owns that
*declares* a model check (`--model-check`) has a passing `model_eval`
verdict bound to the current model + harness digest and to the requirement
as it read when judged. A model edit, a harness edit or a bound change makes
the verdict stale (`MODEL_CHECK_STALE`) until `model eval` runs again; the old
checks stay as history. Requirements that declare no model check are never
demanded of the model: they keep their RTL / chip-level acceptance, and
`model eval` neither asks the harness about them nor records a verdict for
them (a verdict the harness prints anyway is `out_of_scope_results`, a
diagnostic). The abstract architecture model (`--arch`) remains a tactic.

A model edit after a module's build also makes that build's published pass
stale (`BLOCK_NOT_BUILT`: the model is a recorded input), so rebuild the
module after changing its reference.

Checks consume the files that exist and never construct an authoring agent;
authoring is its own verb and touches only what it was asked for:

| Verb | Does | Exit |
|---|---|---|
| `model init\|build\|run\|register --arch` | template / generate + compile / smoke / register the architecture model (generated from `arch_model.json`, never authored) | 0 pass, 1 `ARCH_MODEL_BUILD_FAILED` (compiler diagnostics) / `ARCH_MODEL_RUN_FAILED`, 2 `ARCH_SPEC_MISSING` / `ARCH_SPEC_INVALID` / `ARCH_MODEL_NOT_BUILT` (`required` names the path), 3 toolchain missing |
| `model build` | assemble the SoC model from the registered blocks and contracts (fabric model generated, skeleton headers and contract slices written), compile and smoke the implementations that exist, record a `models` row per block (`sha` of the implementation, `deps_sha` over it and its local includes, the contract version); `missing_models` lists the `model/<block>_model.cpp` files that do not exist. Never authors, repairs or renames an implementation | 0 pass, 1 build/smoke fail, 2 missing implementations (`required`), 3 toolchain |
| `model eval [--arch]` | the DECLARED model checks evaluated with the SUPPLIED harness (`model/frd_eval/*.cpp`, `model/arch/frd_eval/*.cpp`): the requirements that declare a model check are written to `frd_eval/requirements.json`, the harness is built once and run once, and one `model_eval` check per in-scope verdict is recorded, bound to the model + harness digest and the requirement (`model_check_scope`, `scope.excluded`, `out_of_scope_results` in the result). A missing harness, a compile error, an incomplete run (no `FRD_EVAL_DONE`) or a failed run (`FRD_EVAL_DONE` then a non-zero exit) is reported with its diagnostics; nothing is authored or repaired. Only a run that exits 0 AND prints `FRD_EVAL_DONE` can pass or record checks. With no declared model check nothing is required (`skipped`, `gate_ok`) | 0 gate pass, 1 gate fail / `HARNESS_BUILD_FAILED` / `HARNESS_RUN_INCOMPLETE` / `HARNESS_RUN_FAILED`, 2 `HARNESS_MISSING` / `SOC_MODEL_NOT_ASSEMBLED` / `ARCH_MODEL_NOT_BUILT` / `FRD_MISSING` / `FRD_NO_REQUIREMENTS` |
| `model author --block <b> [--block ...] [--diagnostics <log>]` | one SystemC model-author call per NAMED block (a block not named stays missing); `--diagnostics` turns it into a repair request. A provider that cannot run -- non-zero exit, a terminal failure event, an empty answer -- is exit 3 even if a (stale or partial) file exists; the partial answer and the provider's diagnostics are in `llm_calls.jsonl` / `codex_turns.jsonl`. No `--arch` form (rejected, exit 2) | 0 written, 1 not written, 2 unknown/primitive block / `--arch`, 3 provider failure |
| `harness author [--arch] [--diagnostics <log>]` | one FRD-harness author call for the SoC (or architecture) model | same |
| `model refine` | removed (`MODEL_REFINE_REMOVED`) | 2 |

With `--json` the result is the only document on stdout; engine progress lines
go to stderr. Every authoring call is one recorded helper call
(`llm_calls.jsonl`, a `model_authoring` event) and nothing else.

```bash
coresmith model build --json                     # exit 2: missing=[rv_core, ...], required paths named
# write model/rv_core_model.cpp against model/rv_core_model.h yourself, or:
coresmith model author --block rv_core --block sdram_ctrl --json
coresmith model build --json                     # exit 0: build + smoke pass, rows recorded (sha + deps_sha)
coresmith model eval --json                      # exit 2: HARNESS_MISSING -> write model/frd_eval/frd_eval.cpp, or:
coresmith harness author --json
coresmith model eval --json                      # the declared checks; a failing one blocks the module's readiness
coresmith stage status --json                    # MODEL_CHECK_* / MODEL_STALE name what to redo per module
```

The frontend graph's uArch phase authors models only with
`CORESMITH_SYSTEM_MODEL=1`, and then says so (`model_authoring` events).

## Recorded module builds

`coresmith build module <b>` is how a block gets done. The daemon runs the
engine's block subgraph (uArch review -> RTL -> lint -> assertions ->
testbench + DV + coverage -> synthesis + timing -> done) on a persistent
SQLite checkpoint, one thread per build, and records the build in the
`builds` table of `project.sqlite` before the graph starts. Before anything
runs the daemon applies the persisted env and refuses when the shared stages
are not done or no longer hold (`ARCHITECTURE_NOT_READY`), the module is not
ready (`MODULE_NOT_READY`, with the `MODEL_*` / `TARGET_*` / `WORKER_UNBOUND`
items), another build of it is in flight (`BUILD_IN_FLIGHT`), the seed is
not the bound target's own source (`SEED_*`), or cluster fan-out is on
(`CLUSTER_FANOUT_UNSUPPORTED`). A refusal runs no tool, touches no binding
and resets no checkpoint. The build consumes the registered spec unchanged.
To revise it, edit and register the spec before starting a new build. A failed
implementation shell does not block a repair build; stage advancement still
requires shell elaboration. Answer any park with one of its `supported_actions`. `build resume` first re-validates the
recorded inputs against the project (`BUILD_STALE` names the axes; start a
new build). `build pause` cancels the running task (it parks);
`build abort` marks a stopped build aborted, history kept.

The pass `block_done` publishes is bound to the architecture inputs the
build started from (the registered uArch spec, the module's contract edges,
the bound target's configuration, the reference model and its headers, the
FRD harness, the owned must-have items, the worker binding as the adapter
resolves it, the tool/PDK/clock/synthesis-mode context) and to the files it
produced (the complete final target revision -- sources, include-dir
headers, discovered `$readmem` assets -- the testbench with the local
modules it imports, the synthesis reports); immutable copies of those files
are kept under `.coresmith/builds/<id>/` because the next build of the
module overwrites the originals. `build show <id>` reports
whether that pass is still current and why not; the `blocks` exit
(`BLOCK_NOT_BUILT`) and every later stage (`COMPOSITION_NOT_BUILT`) read the
same answer. `best` and the build's `completed` record are committed in one
transaction only after the evidence rows are read back from the database
(the latest DV row a gate pass of the attempt, a coverage verdict of the
attempt, a synthesis row with a positive verdict and measured timing when
required) and the inputs are re-checked; otherwise the build is
`failed_persistence` / `failed` and nothing is published.

```bash
coresmith target bind rv_core --file targets/rv_core.json
coresmith model build --json && coresmith model eval --json
coresmith stage status --json                      # uarch: rv_core ready (no MODEL_*/TARGET_*/WORKER_UNBOUND)
coresmith build module rv_core --json              # {"started": true, "build_id": "b-rv_core-...", ...}
coresmith build status --json                      # completed; current.ok true; evidence rows
coresmith build module rv_core --seed-rtl rtl/rv_core.v --json   # a hand-verified implementation: no regeneration
coresmith build compare b-rv_core-1 b-rv_core-2 --json           # cells/WNS/coverage with stage, tool, PDK, clock, power basis
coresmith build lineage --json                     # intended_workflow per module
coresmith stage next                               # blocks -> integration once every block has a current build
coresmith run start                                # integrates: current builds are reused, never regenerated
```

`run start` runs the same subgraph per tier (every block gets the same
recorded identity at its init node) and skips a module whose current build
is valid; `coresmith run restart-block <b>` is the same build entry. The
MCP tools are `restart_block`, `get_build_state`, `resume_build`,
`abort_build`.

## Module targets (FRD-linked throughput, area, power)

The Architect allocates measurable goals to a module with the FRD items it
already writes; there is no separate target store. An owned, live item with
a `--metric` and a finite `--min`/`--max` is a **module target** when a
module-scope verifier says how it is measured: `must_have` = **required**
(a constraint: the build completes only when it is measured and met),
`should_have`/`nice_to_have` = **advisory** (measured where possible and
reported, never gating -- not even when its tool is missing). An item whose
verifiers are all chip-scope (`--kind chip`, or `--args '{"scope":"integration"}'`)
is deferred to integration/validation. Owned items with module-scope cocotb
verifiers and no bound are **functional** requirements: their acceptance
tests must pass.

| Target | Binding | Measured by (tool receipt) |
|---|---|---|
| throughput / latency / any sim number | `frd verifier PERF-CPU-1 --kind cocotb --path tb/cocotb/test_cpu.py --entry test_throughput [--args '{"workload":"dhrystone"}']` (`workload`/`note` are labels; the test defines its stimulus) | the engine's simulation of the acceptance testbench: the bound test passed in `results.xml` and recorded the value with `orchestrator.harness.measure.record("PERF-CPU-1", v, unit=.., test="test_throughput")` into `sim_build/<b>/measurements.jsonl` (`record` stamps the recording cocotb module; only the bound test of its own acceptance file counts). The receipt hashes both files and every simulated input |
| area | `frd verifier AREA-CPU-1 --kind eda --entry area_um2 [--args '{"area_scope":"std_cell"}']` (unit `um2` or `mm2`) | `total` (default): yosys `stat -liberty` standard-cell area of the build's synthesized netlist + the Liberty `area` of every memory macro the netlist binds by explicit identity (a `cs_sram` wrapper bound by geometry the way the pre-layout STA binds it, or an instance whose cell name is a registered macro). Any other black box makes the total unmeasured. `std_cell`: the explicit standard-cell subtotal, never presented as a total |
| power | `frd verifier PWR-CPU-1 --kind eda --entry power_mw [--args '{"activity":0.1,"duty":0.5}']` (unit `W`/`mW`/`uW`) | OpenSTA `report_power` on the mapped netlist with the build's liberty (plus the bound macros' Liberty, each of which must carry power data) and SDC clock, vectorless with the declared primary-input activity propagated by OpenSTA; `power_basis` `estimated` with method, library, clock and activity. One run per distinct condition. A clockless SDC, an `Error` line, `Creating black box` or a leftover black box is unmeasured, never a number; no cell-count proxy. Other `--args` (a clock, a library, ...) are refused, never ignored: the build's clock and liberty are its own |

**One netlist.** Timing, gate-level simulation, area and power are measured
on ONE netlist: the one the timing verdict was measured on. When the PPA gate
selects the fan-out-repaired netlist over yosys' map, that netlist becomes the
block netlist `syn/output/<b>/<b>_netlist.v` and its own `stat -liberty`
becomes `<b>_report.txt` (the published cell count and area); yosys' map and
report are kept as `<b>_netlist.synth.v` / `<b>_report.synth.txt`. Every area
and power receipt names the timing netlist's sha256 and is rejected when its
own netlist differs (a repaired netlist never borrows the unrepaired one's
cheaper area or power). A selected netlist that is not on disk as measured is
never replaced by yosys' map: the required timing is unmeasured (park
`ppa_gate_unmeasurable`) and nothing is published, with or without targets.

**The acceptance testbench** is the set of files the module's cocotb
verifiers name (protocol, functional and performance tests may live in
separate files). They are fixed inputs of the build: hashed with the local
modules they import into the build identity (`acceptance` axis) and all run
deterministically in one simulation -- the primary one (the block's declared
`testbench` when it is among them) as `test_<module>`, every other file under
its own file stem -- with no testbench-author call when they pass and close
coverage. The graph never authors, regenerates or "fixes" them. When they pass
but miss the line-coverage floor, one helper call adds **supplemental** tests
in `<primary's dir>/test_<module>_supplemental.py` (the same simulation, one
more cocotb module; it may not record measurements). If their bytes change
during the build the build parks `acceptance_changed` (`restore` / `abort`).
Without any module-scope cocotb verifier the worker owns the module testbench
as before. A default build is **unseeded**: the RTL is written by the
implementation worker from the uArch spec, the contracts, the target brief
(`.coresmith/blocks/<b>/build_targets.md`, inlined in its prompt) and the
acceptance tests; `--seed-rtl` stays the explicit reuse/import path. Imported
IP among the bound target's other sources is not rewritten.

**Closure.** After synthesis every candidate goes through `evaluate_targets`:
the targets the build RECORDED at dispatch (never the live FRD) are judged
only from this attempt's receipts, which the evaluation validates itself --
the receipt kind of the method, the hashes of every file the tool read or
wrote, the value re-derived from the tool's own report, the simulation being
this build's and this attempt's and having run the recorded acceptance files.
A scalar, a test name or a JSONL row without a valid receipt is never a
measurement. All required targets met -> `block_done`; a measured miss -> the
gap report (value, bound, over/under by how much) goes to `previous_error.txt`
and the attempt goes back to the implementation worker (no debug-agent call;
it consumes one of `--max-attempts`, default 3); a missing or unproven
measurement -> park `target_unmeasured` (`retry` re-measures after you fixed
the environment, `abort` ends the build incomplete) -- never a pass. Every
evaluation, whatever its outcome, is a row of `build_candidates` with an
immutable copy of its evidence files (`.coresmith/builds/<id>/candidates/
a<attempt>r<round>/evidence/`, the receipts point at the copies; `build show`
lists values, gaps and receipts). `block_done` re-validates the published
candidate's receipts (`FRD_TARGETS_NOT_MET` / `FRD_TARGETS_STALE`) and then
stamps `block_dv` checks with the measured values and receipts on the targets
and functional items, so the `blocks` exit needs no out-of-graph `block-done`.

**Identity.** The `targets` axis covers every target and functional
requirement (item content, bounds, unit, priority, `derives_from`, each
binding's kind/path/entry/args); the `acceptance` axis every acceptance file's
bytes and imports. An FRD revision of a bound target, a changed
workload/activity or a changed oracle is a new input identity: a parked build
is `BUILD_STALE`, a completed one is no longer current. An unrelated FRD edit
stales nothing; the FRD artifact revision is recorded as context. Tools are
identified by the resolved binary's sha256 and a version from a successful
query (the engine's `sta` shim is followed to the real OpenSTA); the daemon
records the paths it resolved (`.coresmith/tool_paths.json`) and every other
process identifies those binaries, so a caller's PATH never changes the
identity. A tool that cannot be identified is unknown -- reported as such,
never a match.

```bash
coresmith frd add "CPU sustains 1 op per 1.25 cycles" --id PERF-CPU-1 --metric cycles_per_op --max 1.25 \
    --unit cycles/op --owner cpu --derives-from KPI-PERF-1 --acceptance "test_throughput <= 1.25"
coresmith frd verifier PERF-CPU-1 --kind cocotb --path tb/cocotb/test_cpu_perf.py --entry test_throughput
coresmith frd verifier FUNC-CPU-3 --kind cocotb --path tb/cocotb/test_cpu.py --entry test_irq_protocol
coresmith frd add "CPU area incl. caches" --id AREA-CPU-1 --metric area_um2 --max 250000 --unit um2 --owner cpu
coresmith frd verifier AREA-CPU-1 --kind eda --entry area_um2
coresmith frd add "CPU power at the build clock" --id PWR-CPU-1 --metric power_mw --max 12 --unit mW --owner cpu
coresmith frd verifier PWR-CPU-1 --kind eda --entry power_mw --args '{"activity":0.1,"duty":0.5}'
coresmith build targets cpu --json                       # what the build binds; exit 2 = it would be refused
coresmith build module cpu --json                        # unseeded; completes only when every required target is met
coresmith build show <id> --json                         # candidates: values, gaps, receipts
```

## Stage exits

| Stage | `stage next` needs |
|---|---|
| `uarch` | every module ready: registered spec, cited PERF budgets, bound target, model built + smoked at the current bytes/headers/contract version, declared model checks passing and current, worker binding declared |
| `blocks` | every block's published pass names a current completed build (primitives included: a materialized file alone is not a receipt), every owned must-have verified at `block_dv` rank or higher with its measured value |
| `integration` | the real-RTL top of the current composition elaborated with no stubs, and the graph's LATEST chip-scope DV verdict of this run is a pass naming this composition |
| `acceptance` | the LATEST validation-scope DV verdict of this run is a pass naming this composition, after the integration DV |
| `backend` | the LATEST backend signoff row (`probe=backend`, DRC + LVS + extracted timing + chip gate-sim [+ precheck]) of this run is `ppa_ok` for this composition, after the validation DV; power is context (`unavailable` / `estimated`), never the signoff |

The composition is every block's current build plus the bytes of every
input the measuring node declares -- the integrated top, the chip testbench
with the local modules it imports, the block RTL it compiled, the ERS the
validation ran against, the flat netlist and SDC the backend signed off --
and the tooling it ran under (the liberty digest, the clock). The graph
records that manifest (`.coresmith/compositions/<sha>.json`) and stamps its
digest on the row (`composition_sha`); the gate re-checks every component.
A module rebuild, a hand edit of the chip top or of the chip testbench, an
ERS or constraint change, or a liberty change re-opens every later stage
(`stage status` lists them under `regressed`); a later failure of the same
scope outranks an earlier pass.

## Build targets

Bind compilation inputs once. The same target is consumed by lint, RTL simulation
and full synthesis; tools do not infer its top from filenames.

```json
{"top":"cpu_top","sources":["rtl/cpu.v","rtl/cache.v"],"cwd":".",
 "include_dirs":["rtl/include"],"defines":{"SIM_CONFIG":1},
 "parameters":{"XLEN":64},"assets":["inputs/boot.hex"]}
```

```bash
coresmith target bind cpu --file targets/cpu.json
coresmith target show cpu --json
coresmith tool run_lint --design cpu --json
coresmith verify rtl cpu --tb tb/cocotb/test_cpu.py --json
coresmith tool run_synth --design cpu --timeout-s 1800 --json
```

Paths in a target are relative to the project root. `cwd` selects execution's
working directory. Sources may be declared before generation; checks require
those files to exist. Defines and parameters accept numeric/token Verilog values.
Whitespace in build-tool paths is rejected before launch. Declare runtime data
in `assets`; includes and literal memory images also participate in the input
revision. A binding is project state (it survives `run start`'s run-id
rotation; a binding recorded under an old run id is preserved as the
project's). Changing a binding invalidates that target's published block pass
and preserves it as historical evidence; a recorded build stores the binding's
configuration as an input and the complete revision of what it produced, so a
later edit of a source, an include-dir header or an asset makes the pass
stale without a rebind. Checks return the input revision; a source change
during a check prevents publication of that check as passing. Simulation also
records the supplied testbench's hash and rejects a change during the check.
Parity/reference builds consume the same binding in their separate directories.

`verify` and `tool` check existing files. A block-level `resume --action fix_tb`
also checks the supplied testbench without generating or repairing it again.
Generated VIPs remain available; missing imports are advisory. Contract, protocol,
functional and measurement checks retain their own verdicts.

An unresolved connection reports the actual and required port widths; choose
an explicit adapter or change the binding. Assembly does not publish a top with
wiring errors. Process outcomes are recorded under `.coresmith/jobs/` with status,
exit code, elapsed time and execution budget. A timeout is distinct from a failed
check. Pause fences further launches from that graph and joins its owned jobs.
