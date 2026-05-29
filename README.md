# coresmith

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/facebookexperimental/coresmith)

Coresmith converts prompts to silicon. It uses LangGraph to drive the full RTL-to-GDS flow: architecture specification, RTL generation, verification, synthesis, and physical design. You start Coresmith through your agent (Claude or Codex), and it works as a daemon that spawns subagents for you until the GDS is created or your input is required. 

> **Try it:** click the Codespaces badge above for a pre-built sandbox with the full EDA toolchain (Yosys, OpenROAD, Magic, Sky130 PDK) and Claude CLI ready to go. You will need to log in to your Claude or Codex account within the codespace.

## What It Does

Coresmith will ask questions about your requirements, then run these phases:

1. **Architecture** -- Generates a Product Requirements Document (PRD), block diagram, memory map, clock tree, and register specification via a multi-step LangGraph state machine
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

## LLM Cost
You must have a Claude Code Max or OpenAI Codex Pro subscription. Codex is recommended and GPT 5.5 is superior at silicon design.

| Design  | Opus 4.7 | GPT 5.5 | 
|------|---------|---------|
| MCU | OK | OK |
| JPEG | Exceeds 5hr limit on Max 5x | OK |

You can use an API key, but it will be expensive.

## Quick Start

> After any install path below, run `make preflight` first — it checks
> the Sky130 PDK files and the `yosys` / `verilator` binaries on `$PATH`
> and prints exactly what's missing. Don't burn a real run on a broken
> toolchain.

### Option A -- Docker / RunPod / Codespace (recommended for first-time users)

The repo ships a `Dockerfile` that bundles the full EDA toolchain
(Yosys, OpenROAD, Magic, netgen, KLayout, Sky130 PDK, Verilator,
cocotb) plus the orchestrator and the Claude CLI. No Nix or local
EDA install needed.

```bash
git clone https://github.com/facebookexperimental/coresmith.git
cd coresmith
docker build -t coresmith:latest .

docker run --rm -it \
    -e ANTHROPIC_API_KEY=sk-ant-... \
    -e CORESMITH_MODE=shell \
    -v "$(pwd)/.coresmith:/coresmith/.coresmith" \
    coresmith:latest
# inside the container:
bin/coresmith daemon start --project-root /coresmith
bin/coresmith run start --project-root /coresmith
```

For a hosted run, see [docs/RUNPOD.md](docs/RUNPOD.md) for a
ready-to-paste pod template.

### Option B -- Local install (Nix-based backend)

```bash
git clone https://github.com/facebookexperimental/coresmith.git
cd coresmith

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e orchestrator/

cp .env.example .env  # then edit and add ANTHROPIC_API_KEY

# Optional: pin a non-default model without code edits
# export CORESMITH_MODEL=sonnet-4.6   # (cheaper than opus-4.7 default)

# Start the MCP server (for interactive use with Claude Code)
make mcp

# Or drive the pipeline via the coresmithd daemon + bin/coresmith CLI
bin/coresmith daemon start --project-root $(pwd)
bin/coresmith run start --project-root $(pwd)
# Then `bin/coresmith state` / `bin/coresmith resume --action approve`,
# or wire up the cron-Claude autochecker described in CLAUDE.md.
```

The local path uses `nix shell "nixpkgs#openroad"` etc. for the backend
EDA tools (see `scripts/*-nix.sh`), so Nix with flakes enabled must be
on `$PATH` for any post-synthesis step. The container image avoids
this entirely.

If you have Nix with flakes already enabled, the cleanest local setup
is `nix develop` -- the repo's `flake.nix` pins every EDA tool plus
Verilator and Node/Claude CLI to a single nixpkgs commit, drops them
on `$PATH`, and bypasses the per-call `nix shell` re-entry through the
`CORESMITH_BACKEND_*` env vars:

```bash
nix develop
# then, inside the dev shell:
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt && pip install -e orchestrator/
bin/coresmith daemon start --project-root $(pwd)
bin/coresmith run start --project-root $(pwd)
```

### Option C -- Linux without Nix or Docker (OSS-CAD-Suite)

If you're on a plain Linux box without Nix and don't want to use Docker,
the [OSS-CAD-Suite](https://github.com/YosysHQ/oss-cad-suite-build)
nightly tarball bundles Yosys, Verilator, OpenROAD, Magic, netgen, and
KLayout at modern, known-good versions. A single `tar xzf` gives you the
full frontend toolchain — apt's `yosys` (0.9) and `verilator` (4.038) on
Ubuntu 22.04 are *below* this README's stated minimums and will silently
break the pipeline.

```bash
git clone https://github.com/facebookexperimental/coresmith.git
cd coresmith

# 1. Frontend EDA toolchain (~2 GB extracted)
curl -L -o /tmp/oss-cad.tgz \
  https://github.com/YosysHQ/oss-cad-suite-build/releases/latest/download/oss-cad-suite-linux-x64-$(date -u +%Y%m%d).tgz \
  || curl -L -o /tmp/oss-cad.tgz \
       "$(curl -s https://api.github.com/repos/YosysHQ/oss-cad-suite-build/releases/latest \
          | grep -oP '"browser_download_url": "\K[^"]+linux-x64[^"]+')"
sudo tar --no-same-owner -C /opt -xzf /tmp/oss-cad.tgz
echo 'export PATH="/opt/oss-cad-suite/bin:$PATH"' | sudo tee /etc/profile.d/oss-cad-suite.sh
export PATH="/opt/oss-cad-suite/bin:$PATH"

# 2. Node + Claude Code CLI (apt nodejs is too old; use NodeSource)
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo bash -
sudo apt-get install -y nodejs
sudo npm install -g @anthropic-ai/claude-code
claude auth login   # interactive

# 3. Python venv + orchestrator
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e orchestrator/
cp .env.example .env   # then edit and add ANTHROPIC_API_KEY

# 4. Sky130 PDK (~2 GB; check `volare ls-remote --pdk sky130` for a current commit)
pip install volare
source scripts/pdk-version.env   # pins SKY130_PDK_COMMIT (single source of truth)
volare enable --pdk sky130 --pdk-root .pdk "$SKY130_PDK_COMMIT"

# 5. Verify before burning a real run
make preflight   # should print {"ok": true}
bin/coresmith daemon start --project-root $(pwd)
bin/coresmith run start --project-root $(pwd)
```

> The orchestrator drives the Claude CLI in headless mode via
> `--permission-mode auto`, which auto-approves tool use without prompting
> and works under any UID. (Older revisions used `--dangerously-skip-permissions`,
> which Claude Code refuses to honour when run as `root`.)

### Reproducible setup (Ubuntu 22.04 reference)

The exact stack used to validate this guide end-to-end on `adder8` and
`mcu3`. Pin the same versions if you want bit-identical reproduction;
otherwise the latest of each works for most blocks.

| Component | Version | Notes |
|---|---|---|
| OS | Ubuntu 22.04 LTS (jammy) | Any glibc >= 2.31 distro works |
| Python | 3.11.x | `python3.11 -m venv venv` |
| Node.js | 20.20.x | NodeSource repo, *not* apt's 12.x |
| Claude Code CLI | 2.x | `npm install -g @anthropic-ai/claude-code` |
| OSS-CAD-Suite | 2026-04-29 nightly | Bundles Yosys 0.64+, Verilator 5.049, OpenROAD, Magic, netgen, KLayout |
| Sky130 PDK | volare commit pinned in [`scripts/pdk-version.env`](scripts/pdk-version.env) | `volare ls-remote --pdk sky130` for current pins |
| Python deps | see `requirements-lock.txt` | `pip install -r requirements-lock.txt` for an exact replay |

To replicate the exact dev environment:

```bash
# 1-4 from "Option C" above, then:
pip install -r requirements-lock.txt   # exact wheels used during validation
pip install -e orchestrator/

# Optional knobs (defaults are sane; bump for very large blocks):
export CORESMITH_RTL_TIMEOUT=1800        # RTL agent LLM timeout (s)
export CORESMITH_TB_TIMEOUT=1800         # Testbench agent LLM timeout (s)
export CORESMITH_TB_FIX_TIMEOUT=600      # Local TB-fix loop timeout (s)
export CORESMITH_LINT_FIX_TIMEOUT=600    # Local lint-fix loop timeout (s)
export CORESMITH_SYNTH_FIX_TIMEOUT=600   # Local synth-fix loop timeout (s)
export CORESMITH_MODEL=opus-4.7          # default; sonnet-4.6 is cheaper

make preflight
bin/coresmith daemon start --project-root $(pwd)
bin/coresmith run start --project-root $(pwd)
make traces      # inspect OTel spans in .coresmith/traces.db
```

The daemon initialises OpenTelemetry at startup, so a SQLite span database
is written to `.coresmith/traces.db` for every run. `make traces` prints
span counts and the slowest 10 spans.

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
Memory Map -> Clock Tree -> Register Spec -> Constraint Check -> OK2DEV Gate
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

## License

MIT License. See [LICENSE](LICENSE) for details.
