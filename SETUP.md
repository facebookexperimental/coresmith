# CoreSmith — Setup & Installation

This guide covers the three install paths and a reproducible reference setup.
For what CoreSmith does, results, and usage, see the [README](README.md).

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
# export CORESMITH_MODEL=sonnet-4.6   # (cheaper than opus-4.8 default)

# Start the MCP server (for interactive use with Claude Code)
make mcp

# Or drive the pipeline via the coresmithd daemon + bin/coresmith CLI
bin/coresmith daemon start --project-root $(pwd)
bin/coresmith run start --project-root $(pwd)
# Then `bin/coresmith state` / `bin/coresmith resume --action approve`,
# or wire up the cron-Claude autochecker described in CLAUDE.md.
```

Backend (post-synthesis) steps need Nix with flakes on `$PATH`. The
cleanest setup is `nix develop` — the repo's `flake.nix` pins every EDA
tool plus Verilator and Node/Claude CLI to one nixpkgs commit and drops
them on `$PATH`, replacing the per-call `nix shell "nixpkgs#openroad"`
re-entry in `scripts/*-nix.sh`. (Option A's container image avoids Nix
entirely.)

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

# 2. Node + agent CLIs (apt nodejs is too old; use NodeSource)
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo bash -
sudo apt-get install -y nodejs
sudo npm install -g @anthropic-ai/claude-code opencode-ai
claude auth login   # default provider, interactive
# Hosted Kimi K3 alternative:
# opencode auth login   # select OpenRouter
# export CORESMITH_LLM_PROVIDER=opencode

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
| OpenCode CLI | 1.18+ | `npm install -g opencode-ai`; optional OpenRouter/Kimi K3 provider |
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
export CORESMITH_MODEL=opus-4.8          # default; sonnet-4.6 is cheaper

make preflight
bin/coresmith daemon start --project-root $(pwd)
bin/coresmith run start --project-root $(pwd)
make traces      # inspect OTel spans in .coresmith/traces.db
```

The daemon initialises OpenTelemetry at startup, so a SQLite span database
is written to `.coresmith/traces.db` for every run. `make traces` prints
span counts and the slowest 10 spans.
