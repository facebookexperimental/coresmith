# CoreSmith
Coresmith converts prompts to silicon. It uses LangGraph to drive the full RTL-to-GDS flow: architecture specification, RTL generation, verification, synthesis, and physical design. You start Coresmith through your agent (Claude or Codex), and it works as a daemon that spawns subagents for you until the GDS is created or your input is required. 


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

## Design Philosophy
**Fewer agents, sharper tactics.** When CoreSmith was written in March 2026, frontier models did not have long horizon capability at chip design. To bridge this gap, we originally split the chip design process into dozens of policies: every chip design deliverable (like a floorplan or block diagram) had its own LangGraph agent. Modern models can run the entire chip design process autonomously, so we are *minimizing the number of agents* in the LangGraph pipeline and *sharpening the tactics*. 

For example, instead of a discrete LangGraph fabric design agent that writes a fabric to disk, we expose fabric creation as a tool action taken through the CoreSmith CLI by the monolithic architect agent.

This has two benefits:

1. A single architect agent stores SoC design state in context and explores constraints in latent space, instead of persisting state to disk and relying on separate review agents to reconstruct the state in every LangGraph node

2. It exposes chip architecture mutations as crisp tactics that can be directly correlated to power, performance and area outcomes.

We still retain adversarial review for design verification, chip integration, and chip validation because we believe black-box testing is able to find more bugs from an unbiased policy. 

We hope this pivot to a tactic-based CLI positions CoreSmith as an example IR for SoC design. 

## LLM Providers
For the best experience: 

* Claude Code Max (minimum $100/month, ideally $200/month): use Opus 5.5 as outer agent, Opus 5.5 for inner agents
* OpenAI Codex Pro (minimum $100/month, ideally $200/month): use Astra as outer agent, GPT 5.6 as inner agent

For research purposes:

* OpenCode/OpenRouter endpoints are supported, including Kimi K3. 
* Meta Muse Spark is supported via OpenCode: http://dev.meta.ai

Using API rates, budget about 3$-5$ for a simple 3-stage MCU. 

## Setup

Three install paths -- see **[SETUP.md](SETUP.md)** for the full commands and a
reproducible Ubuntu 22.04 reference setup:

- **Option A -- Docker / RunPod ** (recommended for first-time users): a
  pre-built image with the full EDA toolchain (Yosys, OpenROAD, Magic, Sky130 PDK) +
  the Claude/Codex CLI. 
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
Interface Contracts
  |
  v
Constraint Check -> OK2DEV Gate
```

Acceptance is the task's own checker: a task ships `inputs/task_adapter.py` and its
verdict outranks the engine's internal requirements. See
[docs/migration.md](docs/migration.md) for what a task declares and for
migrating a project created before that change.

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
VS Code extension provides, but as a plain web page in any browser. It is a
**run-review tool**: the **Overview** tab is a run-level dashboard (block table
with DV / coverage / cells / FF / area / WNS vs budgets, integration and
validation results, chip-lead decisions, interrupt history, engine SHA and
settings, token and wall-time totals); the **Blocks** tab scopes everything to
one block — a chronological **Trajectory** (rounds → graph nodes → every LLM
call with its full prompts, agent commands and captured output, file writes,
response, plus tool logs and engine events) and a **Design & results** page
(uArch spec versions, RTL with diffs between attempts, testbench, simulation,
synthesis, timing, gates, issues, decisions). The graph, Gantt timeline and
collateral browser remain. It's **read-only**: it visualizes a run by reading
that run's `.coresmith/` logs, `project.sqlite` (opened read-only) and
artefacts, so it never drives the pipeline or writes to the run directory.
See [docs/WEBVIEW.md](docs/WEBVIEW.md) for the views, where every number comes
from, the JSON endpoints, and the data the engine does not persist yet.

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

- [docs/AUTHENTICATION.md](docs/AUTHENTICATION.md) — Claude Code or OpenCode/OpenRouter (hosted Kimi K3) credentials
- [docs/LOCAL-DEV.md](docs/LOCAL-DEV.md) — running and iterating without containers
- [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) — common failures (Yosys version, missing PDK, OpenROAD OOM, …)
- [docs/RUNPOD.md](docs/RUNPOD.md) — hosted runs with a ready-to-paste pod template
- [docs/WEBVIEW.md](docs/WEBVIEW.md) — the run-review web UI: views, data sources, endpoints, known gaps

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
