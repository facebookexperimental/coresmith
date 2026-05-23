# Troubleshooting

Run `make preflight` first — it prints exactly which PDK files and binaries are missing or stale, in JSON. Most of the failures below show up there.

---

## Toolchain

### `yosys: command not found` or version mismatch

The pipeline requires Yosys ≥ 0.40. Ubuntu 22.04's apt ships **0.9** (silently breaks synthesis) and macOS Homebrew's `yosys` is usually current.

**Fix:** install via [OSS-CAD-Suite](https://github.com/YosysHQ/oss-cad-suite-build) (Linux), Homebrew (macOS), or use the Docker image which carries a pinned Yosys.

```bash
yosys -V    # must print "Yosys 0.40" or higher
```

### `verilator` too old

The pipeline requires Verilator ≥ 5.0. Ubuntu 22.04 ships **4.038** which lacks features cocotb 2.x relies on.

**Fix:** OSS-CAD-Suite tarball, Homebrew, or the Docker image.

```bash
verilator --version   # must print "Verilator 5.x"
```

### `OpenROAD: command not found` (backend pipeline)

Backend (`make backend`, place-and-route, DRC, LVS) requires OpenROAD, Magic, and netgen on `$PATH`. The frontend (`make pipeline`, RTL → synthesis) does **not**.

**Fix:** if you only need the frontend, ignore. If you need backend, use Docker, Nix (`nix develop`), or RunPod — these all bundle the EDA tools.

---

## PDK

### `Sky130 PDK not found at .pdk/sky130A/`

The pipeline expects the PDK at `$PDK_ROOT` (default `.pdk` in the project root).

**Fix:**

```bash
pip install volare
source scripts/pdk-version.env    # pins the commit hash we test against
volare enable --pdk sky130 --pdk-root .pdk "$SKY130_PDK_COMMIT"
```

The download is ~2 GB. The Docker image pre-installs it at build time.

### `volare: not found remotely`

Open-PDKs has rotated the pinned commit. Run `volare ls-remote --pdk sky130` to see current hashes, pick a recent one, and update `scripts/pdk-version.env` (single source of truth).

---

## Authentication

### `claude: error: invalid API key` or hangs at first agent call

The Claude CLI can't see your credentials. See [AUTHENTICATION.md](AUTHENTICATION.md). Quickest test:

```bash
echo "say hi" | claude -p   # should round-trip in <5 s
```

### Pipeline fails with `--dangerously-skip-permissions cannot be used with root/sudo privileges`

Older coresmith revisions used this flag; current code uses `--permission-mode auto` which works under any UID. Make sure your checkout is up to date.

---

## Pipeline behaviour

### A block hits the LLM timeout

Default is 1800 s (30 min) for RTL/TB and 2700 s (45 min) for uarch / integration / timing closure. Heavy blocks can need more.

**Fix:** bump the relevant env var before re-running:

```bash
export CORESMITH_RTL_TIMEOUT=3600
export CORESMITH_TB_TIMEOUT=3600
export CORESMITH_UARCH_TIMEOUT=3600
export CORESMITH_TIMING_CLOSURE_TIMEOUT=3600
export CORESMITH_INTEGRATION_LEAD_TIMEOUT=3600
export CORESMITH_INTEGRATION_TB_TIMEOUT=3600
export CORESMITH_INTEGRATION_REVIEW_TIMEOUT=3600
make pipeline    # auto-resumes from the last successful checkpoint
```

The graph is checkpointed in `.coresmith/pipeline_checkpoint.db`, so re-running picks up where the previous run left off — you don't lose work.

### `make pipeline` runs forever / one block dominates

`make traces` shows the slowest spans:

```bash
make traces      # span counts + slowest 10 spans
sqlite3 .coresmith/traces.db   # interactive SQL
```

Look for blocks with many `rtl_attempt_*` or `synth_attempt_*` spans — those are retry loops. The block-specific log under `.coresmith/step_logs/<block>/` has the verbatim LLM output.

### OpenROAD OOM on large blocks

PnR can need 8+ GB resident on blocks > 50k cells. Reduce target clock (less timing-driven optimization) or split into smaller blocks via the architecture phase.

---

## Cocotb / simulation

### `ModuleNotFoundError: No module named 'cocotb_bus'`

Older requirements.txt didn't pin cocotb-bus. Pull the latest and:

```bash
pip install -r requirements.txt
```

### Verilator dumps thousands of `WARNING-WIDTH` messages

These are real width-mismatch warnings from generated RTL. The pipeline's lint loop usually fixes them automatically; if you're seeing them on a final run, the LLM gave up and merged anyway. Inspect the block's `.coresmith/step_logs/<block>/lint_*.log` to see what was tried.

---

## Where to find more signal

- `.coresmith/pipeline_results.json` — final per-block status
- `.coresmith/pipeline_events.jsonl` — every state transition, line-delimited JSON
- `.coresmith/step_logs/<block>/` — verbatim LLM call inputs + outputs per step
- `.coresmith/traces.db` — OpenTelemetry spans (use `make traces`)
- `.coresmith/llm_calls.jsonl` — every LLM call with tokens + cost (recent runs)

If you find a reproducible bug not listed above, please open an issue with `.coresmith/pipeline_events.jsonl` and the relevant block's step logs attached.
