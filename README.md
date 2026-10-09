# CoreSmith

[![CI](https://github.com/facebookexperimental/coresmith/actions/workflows/ci.yml/badge.svg)](https://github.com/facebookexperimental/coresmith/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.11 | 3.12](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](requirements.txt)
[![Version](https://img.shields.io/github/v/tag/facebookexperimental/coresmith?label=version&sort=semver)](https://github.com/facebookexperimental/coresmith/tags)

Coresmith converts prompts to silicon. It uses LangGraph to drive the full RTL-to-GDS flow: architecture specification, RTL generation, verification, synthesis, and physical design. You start Coresmith through your agent (Claude or Codex), and it works as a daemon that manages the chip lifecycle and spawns subagents for you until the GDS is created or your input is required.


## What It Does

You provide Coresmith with a specification of the ASIC you want, ideally including a software model for some parts of the design. Coresmith will then decompose your requirements into an ASIC architecture and autonomously drive execution into a GDS.

1. **Architecture**: Generates a Product Requirements Document (PRD), functional requirements document and system architecture
2. **Microarchitecture**: Decomposes your requirements and software model into microarchitecture specifications and byte-exact software models using heuristics for data movement and locality
3. **RTL Generation**: An LLM agent converts specifications for every block into synthesizable Verilog
4. **Verification**: Another LLM agent generates cocotb testbenches; Verilator lints and simulates
5. **Synthesis**: Yosys synthesizes each block to a gate-level netlist targeting the SkyWater Sky130 130nm PDK, fixing timing if necessary
6. **Backend**: OpenROAD/Magic/netgen handle place-and-route, DRC, and LVS
7. **Diagnosis**: On failure at any step, a debug agent analyzes the root cause and retries with corrective constraints

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

## Setup

See [SETUP.md](SETUP.md) (Docker/RunPod, Nix, or OSS-CAD-Suite), then run `make preflight`.

## Usage

Open the repo in Claude Code or Codex and describe your chip. [CLAUDE.md](CLAUDE.md)
tells the agent how to drive CoreSmith.

To drive it by hand (one project root per chip):

    export CORESMITH_PROJECT_ROOT=/path/to/project
    coresmith daemon start
    coresmith stage status --json         # what blocks the next stage
    coresmith stage next --json           # advance once blockers are resolved
    coresmith build module <module> --json
    coresmith build status --json         # parks list their supported_actions
    coresmith build resume --build-id <id> --action <action>

Nothing is auto-approved. Every park waits for a `resume`. See
[CLI and operation](docs/gitbook/usage.md) for the full command inventory.

For a read-only live dashboard, see [docs/WEBVIEW.md](docs/WEBVIEW.md).

## Testing

    pytest orchestrator/tests/ -m "not live_llm and not requires_nix and not e2e"

## Docs

- **[The CoreSmith book](docs/gitbook/README.md)**: start here
- [CLAUDE.md](CLAUDE.md): agent decision contract, run conventions
- [docs/migration.md](docs/migration.md): task acceptance (`task_adapter.py`)
- [docs/AUTHENTICATION.md](docs/AUTHENTICATION.md) · [LOCAL-DEV](docs/LOCAL-DEV.md) · [TROUBLESHOOTING](docs/TROUBLESHOOTING.md) · [RUNPOD](docs/RUNPOD.md) · [WEBVIEW](docs/WEBVIEW.md)

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
