# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added (feat/soc-stages -- see docs/SOC_STAGES.md)
- Fabric Resolution: the SoC bus is a generated primitive over vendored
  pulp-platform `axi` IP (FabricSpec -> yosys-slang -> plain Verilog +
  cocotbext-axi testbench); `axi4` / `axi_lite` / `apb` contract families.
- Structured contract `timing` and one generated Interface VIP per edge
  (cocotb driver/monitor/scoreboard/assertions, `$past` SVA binds); block
  testbenches must exercise every edge through its VIP.
- Shell integration: the chip top is assembled from the contracts before the
  first tier and after every tier; the final top is the same assembly.
- uArch phase delivers a SystemC TLM-2.0 loosely-timed SoC model.
- Architect sitting step 2: the executable SAD -- an abstract SystemC performance
  model built from `model/arch/arch_model.json` (`coresmith model init|build|run|eval
  --arch`), FRD verdicts recorded as `model_eval` checks, and `coresmith fabric derive`
  producing the FabricSpec from the measured link table.
- Architect sitting step 1: project ontology (artifacts / items / links / checks /
  questions / stages) and the deterministic stage machine, driven by `coresmith
  register|item|link|check|question|stage|status`; the graph nodes register too.
- The FRD is evaluated on that model before any RTL is lowered: an agent-authored
  SystemC harness answers every FRD requirement id (pass / fail / not_testable with
  reason); the uArch phase gate (default on) parks on build, smoke or FRD failure.
  The model prompts demand behaviourally complete models (full register maps, ISA
  execution for processor blocks, real memory contents).
- Assertion stage: spec invariants must exist as assertions; phantom claims
  are rejected.
- Timing false pass closed: `best` means sim AND synth AND timing; `abc -D`;
  a `timing_fix` loop.
- Run state in the project DB: leases, run flags, decisions, interrupts
  (single-branch resume), LLM slots; operator rulings channel.

### Changed
- **Breaking:** the architecture phase no longer runs the uArch exploration,
  Memory Map, Clock Tree, Register Spec or Complexity Review stages, and their
  artifacts are no longer produced. Project state is one SQLite database with the
  JSON files beside it as regenerated views.
- **Breaking:** acceptance is the task's own checker. A task ships
  `inputs/task_adapter.py` and its verdict outranks the engine's internal
  requirements; the built-in stream harness is the fallback and now requires an
  explicit `AXIS_MAPPING` instead of inferring packing and geometry.
- **Breaking:** the chip top and the chassis are declared in `inputs/task.yaml`
  rather than guessed from the RTL, and integration adopts one validated
  candidate manifest that every consumer reads.
- The oracle integrity baseline moved outside the project, by default under
  `~/.coresmith/trust/`; task adapters run inside a bubblewrap boundary.
- See [docs/migration-arm-e.md](docs/migration-arm-e.md) for migrating an
  existing project.

### Added
- Initial open-source release of coresmith
- LangGraph-based ASIC pipeline orchestration (architecture, RTL, verification, synthesis, backend)
- MCP server for interactive use with Claude Code
- Headless CI mode with auto-retry and auto-skip
- Sky130 PDK support via Volare
- OpenTelemetry tracing
