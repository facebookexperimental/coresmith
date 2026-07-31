# Authentication

coresmith supports Claude Code (default), Codex CLI, OpenCode with OpenRouter's
hosted Kimi K3, and the Kimi Code CLI. Select OpenCode with
`CORESMITH_LLM_PROVIDER=opencode`, or Kimi Code with `CORESMITH_LLM_PROVIDER=kimi`.

## OpenCode + OpenRouter (Kimi K3)

Install the official OpenCode CLI and authenticate OpenRouter locally:

```bash
npm install -g opencode-ai
opencode auth login     # select OpenRouter and enter the key locally
export CORESMITH_LLM_PROVIDER=opencode
```

For CI, Docker, or RunPod, use an environment secret instead of the interactive
credential store:

```bash
export OPENROUTER_API_KEY=...
export CORESMITH_LLM_PROVIDER=opencode
export CORESMITH_OPENCODE_MODEL=openrouter/moonshotai/kimi-k3  # optional; default
```

OpenCode stores interactive credentials at
`${XDG_DATA_HOME:-~/.local/share}/opencode/auth.json`. Never commit this file or
paste the API key into CoreSmith configuration. Verify the route with:

```bash
opencode --version
printf 'Reply with exactly: ready' | opencode --pure run --format json \
  --model openrouter/moonshotai/kimi-k3
```

CoreSmith passes prompts on stdin, adds OpenCode's `--thinking` flag, and
consumes the NDJSON event stream. Every valid event—including model-exposed
`reasoning` parts—is appended to `.coresmith/opencode_turns.jsonl`; final answers
and usage remain in `.coresmith/llm_calls.jsonl`. The trajectory file can contain
sensitive prompt, reasoning, and tool content, so protect it like other run
artifacts. CoreSmith uses `permission: "deny"` when tools are disabled.

## Kimi Code

Install the official CLI (Node.js 22.19 or newer is required):

```bash
npm install -g @moonshot-ai/kimi-code
kimi --version
```

For a developer machine or persistent worker, use Kimi's device-code login:

```bash
kimi login
export CORESMITH_LLM_PROVIDER=kimi
```

The CLI stores config, sessions, and OAuth credentials in `~/.kimi-code`. Set
`KIMI_CODE_HOME` to relocate that directory; mount it on persistent storage in
a container or ephemeral worker.

For headless API-key deployments, Kimi Code's explicit temporary-model channel
requires both variables (an ordinary exported `KIMI_API_KEY` is not read):

```bash
export KIMI_MODEL_NAME=kimi-for-coding
export KIMI_MODEL_API_KEY=...
export CORESMITH_LLM_PROVIDER=kimi
```

Optional coresmith overrides:

```bash
export CORESMITH_KIMI_MODEL=kimi-code/k3
export KIMI_CLI_PATH=/usr/local/bin/kimi
export CORESMITH_KIMI_WORKDIR=/path/to/project
```

coresmith communicates with `kimi acp` over stdin/stdout JSON-RPC. This avoids
command-line length limits for large RTL prompts and preserves streaming token
usage.

## Claude Code

The [Claude Code CLI](https://docs.claude.com/en/docs/claude-code/overview)
accepts three credential sources, checked in this order:

1. **`CLAUDE_CODE_OAUTH_TOKEN`** — long-lived OAuth token from `claude setup-token`. Recommended for CI, Docker, RunPod, GitHub Codespaces.
2. **`ANTHROPIC_API_KEY`** — raw API key from <https://console.anthropic.com/>. Bills your console workspace; *not* your Claude.ai/Pro subscription.
3. **Interactive browser login** — `claude auth login` opens a browser; only works when a desktop browser is reachable from the terminal.

If you have a Claude Pro/Max subscription, **option 1 is the right choice** — it bills against your subscription, not the API console. Option 2 is for users who want or need API-billed usage (typically heavier programmatic workloads).

---

## Option 1: OAuth token (recommended)

```bash
# On a machine with a browser, run once:
claude setup-token
# Copy the printed token (starts with `sk-ant-oat01-...`).

# On the machine that will run coresmith:
export CLAUDE_CODE_OAUTH_TOKEN=sk-ant-oat01-...
```

The token is long-lived; rotate it via the same command.

### In Docker

```bash
docker run --rm -it \
    -e CLAUDE_CODE_OAUTH_TOKEN=sk-ant-oat01-... \
    coresmith:latest
```

### In RunPod

Set `CLAUDE_CODE_OAUTH_TOKEN` as a pod environment variable in the template (see [docs/RUNPOD.md](RUNPOD.md)).

### In GitHub Codespaces

1. On your fork: **Settings → Secrets and variables → Codespaces → New secret**
2. Name: `CLAUDE_CODE_OAUTH_TOKEN`
3. Value: the token from `claude setup-token`

The devcontainer config (`.devcontainer/devcontainer.json`) forwards the secret automatically.

### In GitHub Actions (nightly e2e)

Same as Codespaces, but under **Settings → Secrets and variables → Actions**.

---

## Option 2: API key (console-billed)

```bash
export ANTHROPIC_API_KEY=sk-ant-api03-...
```

If both `CLAUDE_CODE_OAUTH_TOKEN` and `ANTHROPIC_API_KEY` are set, the OAuth token wins. To force API billing, unset the OAuth token.

---

## Option 3: Interactive login

Only useful on a developer machine with a desktop browser:

```bash
claude auth login
```

This will not work inside a headless Docker container, RunPod pod, or CI runner — use options 1 or 2 there.

---

## Verifying

Quick check that the CLI is wired up:

```bash
claude --version            # should print a 2.x version string
echo 'say hi' | claude -p   # should round-trip a short response
```

If `claude -p` hangs or returns an auth error, neither token nor key are being seen by the CLI — re-export them in the same shell.

---

## What model gets used

coresmith defaults to `opus-5` (the most capable model). Override with `CORESMITH_MODEL`:

```bash
export CORESMITH_MODEL=sonnet-5   # ~5x cheaper, slightly less reliable on hard blocks
export CORESMITH_MODEL=haiku-4.5    # cheapest; fine for trivial blocks
```

The mapping from short names (`opus-5`, `sonnet-5`, `haiku-4.5`, …) to full CLI model IDs lives in `orchestrator/langchain/agents/coresmith_llm.py`. Unknown short names pass through verbatim, so any model the CLI accepts works.
