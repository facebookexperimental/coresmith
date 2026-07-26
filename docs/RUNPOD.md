# Running coresmith on RunPod

coresmith ships a single Docker image (`Dockerfile` at the repo root) that
bundles the entire EDA toolchain (Yosys, OpenROAD, Magic, netgen, KLayout,
Sky130 PDK, Verilator + cocotb) with the Python orchestrator and the
Claude Code CLI. RunPod is the lowest-friction way to run the pipeline
end-to-end on a fresh machine -- no Nix install required.

## TL;DR

1. Build and push the image:
   ```bash
   docker build -t <your-registry>/coresmith:latest .
   docker push <your-registry>/coresmith:latest
   ```
2. Create a RunPod template (see template JSON below).
3. Deploy a pod, set `ANTHROPIC_API_KEY`, then either SSH in or watch the
   logs of the headless `pipeline` mode.

## Pod sizing

The workload is 100% CPU/RAM bound -- the LLM runs on Anthropic's servers,
and the heavy local work is Yosys/OpenROAD/Magic. A GPU pod is wasted money.

| Resource | Recommended |
|----------|-------------|
| CPU      | 8 vCPU      |
| RAM      | 32 GB (16 GB minimum; OpenROAD on a chip-finish run can spike) |
| Disk     | 60 GB persistent volume mounted at `/coresmith/.coresmith` |
| GPU      | None        |
| Image    | `<your-registry>/coresmith:latest` (or build inline; see below) |

Pick any RunPod CPU pod template. "Community Cloud" is fine; "Secure Cloud"
is fine. There's no GPU dependency.

## Environment variables

| Variable | Purpose | Required |
|----------|---------|----------|
| `ANTHROPIC_API_KEY` | API key from console.anthropic.com | one of API key / OAuth required for non-shell modes |
| `CLAUDE_CODE_OAUTH_TOKEN` | OAuth token from `claude setup-token` | alternate to API key |
| `CORESMITH_MODE` | `shell` (default), `pipeline`, `mcp`, `mcp-http`, `test` | optional |
| `CORESMITH_MODEL` | Override default model (e.g. `opus-4.8`, `sonnet-4.6`, `haiku-4.5`) | optional |
| `MCP_PORT` | Port for `mcp-http` mode (default `8765`) | optional |

## RunPod template (JSON)

Save as a "Pod Template" in the RunPod dashboard, or use it directly via
the RunPod GraphQL API:

```json
{
  "name": "coresmith",
  "imageName": "your-registry/coresmith:latest",
  "containerDiskInGb": 60,
  "volumeInGb": 50,
  "volumeMountPath": "/coresmith/.coresmith",
  "ports": "8765/http",
  "env": [
    { "key": "CORESMITH_MODE",    "value": "shell" },
    { "key": "CORESMITH_MODEL",   "value": "" },
    { "key": "ANTHROPIC_API_KEY","value": "" },
    { "key": "PYTHONUNBUFFERED","value": "1" }
  ],
  "dockerArgs": ""
}
```

Notes:

- `volumeMountPath` is `/coresmith/.coresmith` so SQLite checkpoints,
  generated RTL/testbenches, and the event log all survive pod restarts.
  If you want generated RTL/syn/pnr to persist as well, mount additional
  volumes at `/coresmith/rtl`, `/coresmith/syn`, `/coresmith/pnr`.
- `8765/http` is only needed if you plan to use `CORESMITH_MODE=mcp-http`
  (so an external Claude CLI / Cursor instance can talk to the in-pod MCP
  server). Drop it for headless `pipeline` mode.
- Leave `ANTHROPIC_API_KEY` blank in the template and fill it in when
  deploying so the secret doesn't end up in the template definition.

## Using the pod

### Headless: full pipeline run, no interaction

```
CORESMITH_MODE=pipeline
ANTHROPIC_API_KEY=sk-ant-...
```

The entrypoint runs `make pipeline` which auto-resumes interrupts (uArch
specs are auto-approved, failed blocks retry up to `MAX_ATTEMPTS=5` then
skip). Watch the pod's stdout for progress; results are written to
`.coresmith/pipeline_results.json`.

### Interactive: MCP over HTTP, drive from your laptop

In the pod, set `CORESMITH_MODE=mcp-http`. RunPod will expose port 8765 as
a public HTTPS endpoint -- e.g. `https://abc123-8765.proxy.runpod.net/`.
Point your local Claude CLI at it:

```bash
claude --mcp-server coresmith=https://abc123-8765.proxy.runpod.net/sse
> /mcp
> use the start_architecture tool with my requirements...
```

This is the recommended setup for design exploration: heavy EDA tools run
in the pod, your laptop just sees Claude conversations.

### Debug shell

The default mode is `CORESMITH_MODE=shell`. SSH or web-terminal in, then:

```bash
make help
make preflight    # verifies PDK + EDA tools are reachable
make pipeline     # run frontend pipeline
```

## Cost rough-cut

A complete frontend pipeline run on the FFT16 reference design takes
~30 minutes on an 8-vCPU pod (Yosys synth + Verilator sim + a few LLM
round trips per block). At RunPod's current ~$0.15/hr CPU pricing, that's
under $0.10 of compute. Most of the spend is the LLM calls themselves
(measured separately on your Anthropic account).

A full RTL-to-GDSII run (frontend + backend + tapeout for a small chip)
on the same pod takes 2-4 hours, dominated by OpenROAD PnR and DRC.

## Building the image without pushing

If you just want to test locally before pushing to a registry:

```bash
docker build -t coresmith:latest .
docker run --rm -it \
    -e ANTHROPIC_API_KEY=sk-ant-... \
    -e CORESMITH_MODE=shell \
    -v "$(pwd)/.coresmith:/coresmith/.coresmith" \
    coresmith:latest

# inside the container:
make preflight
make pipeline
```

The first build is slow (~20 min, mostly the Sky130 PDK download).
Subsequent code-only edits rebuild in seconds because dependencies and
the PDK are cached in earlier layers.
