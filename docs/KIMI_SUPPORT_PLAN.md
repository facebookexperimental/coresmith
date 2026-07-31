# Plan: Kimi support for Coresmith

Status: implemented; live generation blocked by account membership validation
Researched: 2026-07-26

## Executive decision

Kimi does offer an official coding CLI. The current product is **Kimi Code CLI**, installed with the official installer or `npm install -g @moonshot-ai/kimi-code`. It supports a non-interactive prompt mode and newline-delimited JSON output:

```sh
kimi -p "Summarize this repository" --output-format stream-json
```

That makes a CLI-backed Kimi provider the closest fit for Coresmith's existing
Claude Code and Codex adapters. The implementation uses Kimi Code's supported
`kimi acp` JSON-RPC transport rather than `-p`: prompts travel over stdin, ACP
streams assistant chunks and token usage, and Coresmith answers tool permission
requests without an interactive terminal.

The adapter, tests, packaging, and documentation are implemented against Kimi
Code 0.29.1. The real initialize/session handshake was verified through the
authentication boundary. Device authorization reached Kimi successfully, but
Kimi rejected the account because it could not verify active membership, so a
billable live model response remains an account-level validation item.

## Research summary

- Kimi Code CLI is a terminal coding agent with file, search, edit, shell, sub-agent, Skills, and MCP tools. In `-p` mode, normal tool calls use the non-interactive auto permission policy, so it should not block on an approval prompt.
- The current CLI emits one JSON object per stdout line with `--output-format stream-json`. Assistant messages use `role: "assistant"`; tool calls and tool results are separate messages. Thinking and progress can remain on stderr.
- Interactive or device-code OAuth is available through `kimi login`. Direct API-key configuration is also supported.
- Ordinary exported `KIMI_API_KEY` is **not** automatically consumed by current Kimi Code. Credentials normally live in `~/.kimi-code/config.toml`. The explicit `KIMI_MODEL_*` variables are the supported environment-only channel for ephemeral/CI model configuration.
- Current Kimi coding model IDs include `k3`, `k3-256k`, `kimi-for-coding`, and `kimi-for-coding-highspeed`. The exact aliases accepted by `kimi -m` depend on the installed catalog (the documentation example uses `kimi-code/kimi-for-coding`), so aliases must be discovered in the compatibility spike instead of guessed in production code.
- Moonshot also publishes a Python `kimi-agent-sdk` that exposes the same agent runtime programmatically. It is a reasonable fallback if the supported CLI cannot accept large prompts over stdin or cannot reliably express Coresmith's no-tools calls.

Primary sources:

- [Kimi Code CLI command reference](https://www.kimi.com/code/docs/en/kimi-code-cli/reference/kimi-command.html)
- [Kimi Code CLI installation guide](https://www.kimi.com/code/docs/en/kimi-code-cli/guides/getting-started.html)
- [Kimi Code environment variables](https://www.kimi.com/code/docs/en/kimi-code-cli/configuration/env-vars)
- [Kimi Code providers and models](https://www.kimi.com/code/docs/en/kimi-code-cli/configuration/providers.html)
- [Kimi Code model IDs](https://www.kimi.com/code/docs/en/kimi-code/models.html)
- [MoonshotAI/kimi-cli transition notice](https://github.com/MoonshotAI/kimi-cli)
- [Kimi Agent SDK](https://github.com/MoonshotAI/kimi-agent-sdk)

## Current Coresmith seam

All agents already go through `ClaudeLLM.call(system, prompt)`. Despite the class name, `orchestrator/langchain/agents/coresmith_llm.py` currently selects either `claude_cli` or `codex_cli` using `CORESMITH_LLM_PROVIDER`, resolves provider-specific model names, starts a subprocess, applies timeout/stall handling, extracts a final response, and records JSONL/OpenTelemetry data.

The Kimi change should extend that seam. It should not modify every agent or introduce provider conditionals in LangGraph nodes.

The relevant files are:

- `orchestrator/langchain/agents/coresmith_llm.py`: provider selection, model mapping, binary discovery, command construction, output parsing, process watchdog, telemetry.
- `orchestrator/tests/test_coresmith_llm.py`: provider, mapping, command, parser, watchdog, and telemetry unit tests.
- `.env.example` and `orchestrator/config.yaml`: runtime configuration and comments.
- `Dockerfile`, `flake.nix`, and `scripts/runpod_entrypoint.sh`: installed CLI and persistent authentication/config state.
- `README.md`, `docs/AUTHENTICATION.md`, `docs/RUNPOD.md`, and `docs/TROUBLESHOOTING.md`: installation and operator guidance.
- `orchestrator/tests/test_live_architecture.py`: pattern for an opt-in live-provider test.

## Proposed user-facing contract

Keep Claude as the default and add these settings:

```sh
CORESMITH_LLM_PROVIDER=kimi
CORESMITH_KIMI_MODEL=<validated-kimi-cli-alias>
KIMI_CLI_PATH=/usr/local/bin/kimi          # optional
CORESMITH_KIMI_WORKDIR=/path/to/project    # optional
KIMI_CODE_HOME=/persistent/kimi-home       # optional; config and OAuth state
```

Provider aliases `kimi` and `kimi_cli` should both normalize to `kimi_cli`. Model precedence should be:

1. `CORESMITH_KIMI_MODEL`
2. `CORESMITH_MODEL`
3. the provider-specific mapping of the agent's requested Coresmith tier

After phase 0 validates CLI aliases, map the expensive top-level tier (`opus-4.7`) to the strongest generally available Kimi coding model and the block tier (`sonnet-4.6`) to the standard coding model. Do not default to `*-highspeed`: its documented quota multiplier makes it an explicit opt-in.

For CI/API-key use, document one of these supported approaches rather than implying that bare `KIMI_API_KEY` works:

- provision `$KIMI_CODE_HOME/config.toml` as a secret, or
- set the complete `KIMI_MODEL_NAME`, `KIMI_MODEL_API_KEY`, `KIMI_MODEL_BASE_URL`, and related `KIMI_MODEL_*` configuration.

Never log the config file or any credential-bearing environment value.

## Implementation phases

### Phase 0: compatibility spike and fixtures

Pin a current Kimi Code version in a disposable environment and capture sanitized fixtures for:

1. `kimi --version` and `kimi --help`.
2. A no-tool one-line response in text and `stream-json` modes.
3. A tool-using response that reads a fixture file and writes another fixture file.
4. A permanent auth failure, a transient/rate-limit failure if reproducible, and a killed/timeout run.
5. The model aliases exposed after OAuth login and after `KIMI_MODEL_*` configuration.
6. At least 256 KiB and 1 MiB prompts delivered without placing the payload in argv.
7. Eight concurrent one-shot calls sharing one `KIMI_CODE_HOME` but using separate work directories.
8. A call in which tools are reliably unavailable.

Prefer a supported stdin/JSON input flag if the current binary exposes one. If current Kimi Code does not support large prompt input over stdin, evaluate the Python Kimi Agent SDK for the provider implementation. Treat argv-only transport as a release blocker, not as the production fallback.

The spike must also choose how Coresmith's `system` argument is represented:

- Preferred: a per-call temporary agent definition/system prompt if the stable CLI supports it without an experimental flag.
- Fallback: the same explicit `<system>...</system><user>...</user>` envelope used by the Codex adapter, followed by prompt-adherence tests.

For `disable_tools=True`, use a real Kimi tool policy/agent profile. A sentence asking the model not to call tools is not sufficient. If stable CLI flags cannot enforce it, use the SDK for those calls or introduce a provider capability error rather than silently enabling tools.

Commit the sanitized stdout fixtures under `orchestrator/tests/fixtures/kimi/`. They make parser tests deterministic and reveal upstream schema drift during upgrades.

Exit criteria: stdin-safe prompt transport, deterministic final-answer extraction, enforceable no-tools mode, known model aliases, and successful concurrent calls.

### Phase 1: add `kimi_cli` to the provider adapter

In `coresmith_llm.py`:

1. Add `_KIMI_MODEL_MAP` and a provider-specific default selected from the phase 0 catalog.
2. Extend `_detect_provider()` and its error message for `kimi`/`kimi_cli`.
3. Extend `_resolve_model()` with `CORESMITH_KIMI_MODEL` precedence.
4. Add `_find_kimi_binary()` with `KIMI_CLI_PATH`, `PATH`, and documented install locations. Its error should name both the official installer and npm package.
5. Add `kimi_path` to `ClaudeLLM.__init__` without breaking existing call sites. Keep `ClaudeLLM` as the compatibility name in the first patch; a broad rename can be a separate cleanup.
6. Add `_generate_via_kimi_cli()` and a Kimi-specific command builder. Set the subprocess working directory explicitly to `CORESMITH_KIMI_WORKDIR`, then `CORESMITH_PROJECT_ROOT`, then the existing project-root fallback.
7. Refactor `_run_cli_with_watchdog()` just enough to accept a provider parser and optional `cwd`; do not duplicate its timeout, stall, heartbeat, active-process, and live-stream logic.
8. Add `_parse_kimi_stream_json()`. It should ignore malformed/non-message lines, record tool events for trajectory diagnostics, and return the final non-empty assistant text rather than tool output or an intermediate assistant preamble.
9. Record `provider="kimi_cli"` in JSONL, graph events, live-stream records, and OpenTelemetry. Populate usage only when the captured schema provides trustworthy token fields; `{}` is better than invented counts.
10. Handle the CLI's documented retryable exit status separately from permanent failures. Preserve stderr and exit code in the error record, retry only transient failures with bounded backoff, and let the existing circuit breaker see the final outcome.
11. Make provider-neutral log/error text say “LLM CLI” or the selected provider instead of “Claude CLI.” Update `kill_active_cli_processes()` documentation and messages similarly.

Do not add a direct OpenAI-compatible HTTP client in this phase. Coresmith relies on the coding agent's filesystem and shell tools, not only text completion.

### Phase 2: tests

Extend `test_coresmith_llm.py` with:

- provider alias and invalid-provider tests;
- Kimi model precedence/pass-through tests;
- binary discovery and missing-binary diagnostics;
- exact command, stdin, cwd, and no-tools policy tests;
- fixture-based JSON parsing for plain responses, tools, multiple assistant messages, malformed lines, empty output, and usage when present;
- exit-code classification, retry limit, timeout, stall, and partial-output tests;
- telemetry assertions for `kimi_cli`;
- regression assertions for the existing Claude and Codex commands/parsers.

Add an opt-in live test, skipped unless a Kimi-specific environment flag is set. It should perform a cheap no-tool call and a temporary-directory file read/write call. Never run a billable live test in the default unit suite.

Then run one small Coresmith workflow (for example the architecture phase and an `adder8` frontend run) with Kimi. Acceptance is based on Coresmith artifacts and lint/simulation gates, not textual equality with Claude output.

### Phase 3: packaging, preflight, and docs

1. Install a pinned Kimi Code CLI in the container and verify `kimi --version` during the image build. The current npm package requires a newer Node than Coresmith's documented Node 20 setup, so prefer the official standalone binary/install path unless the base Node upgrade is independently validated for Claude Code and the rest of the image.
2. Persist or mount `KIMI_CODE_HOME` in RunPod/container deployments so OAuth and model configuration survive restarts. Do not bake credentials into an image layer.
3. Teach the preflight check to validate only the selected provider's binary and basic configuration, and print an actionable Kimi login/config error.
4. Add the user-facing variables above to `.env.example` and update the provider comment in `orchestrator/config.yaml`.
5. Generalize `docs/AUTHENTICATION.md` into provider sections covering Claude, Codex, and Kimi. Include `kimi login`, API-key/CI configuration, `kimi --version`, and a minimal headless smoke test.
6. Update README, RunPod, local-development, and troubleshooting instructions. Clearly distinguish the Coresmith daemon client `bin/coresmith` from the upstream model-provider CLIs.
7. Document the pinned Kimi Code version and the fixture-refresh/upgrade procedure.

### Phase 4: validation and rollout

Run the same small design through Claude and Kimi with identical Coresmith inputs. Compare:

- graph completion and interrupt behavior;
- required architecture/RTL/testbench artifacts;
- Yosys/Verilator/cocotb results;
- retry, timeout, and circuit-breaker behavior;
- elapsed time and token/usage data where available;
- parallel block generation and pause/kill behavior.

Roll out Kimi as opt-in (`CORESMITH_LLM_PROVIDER=kimi`) for at least one release. Claude remains the default. Promote Kimi in documentation only after a repeatable architecture-to-simulation run succeeds and no provider-specific branches have leaked outside the LLM adapter.

## Acceptance criteria

- `CORESMITH_LLM_PROVIDER=kimi` routes every existing agent through Kimi without agent-level code changes.
- A fresh user gets a precise missing-CLI or missing-auth error before an expensive pipeline run.
- Large prompts are transported over stdin or an SDK channel, never argv.
- Tool-enabled agents can read/write/run in the intended project directory; `disable_tools=True` actually prevents tool use.
- Final assistant output, tool trajectory, timeout/stall state, exit code, and provider identity are observable.
- Concurrent block calls do not corrupt Kimi configuration or session state.
- Unit tests require no Kimi credentials; live tests are explicit and skipped by default.
- Existing Claude and Codex unit tests and one smoke workflow continue to pass unchanged.
- Container/RunPod credentials remain runtime secrets and survive restarts when configured for persistence.

## Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Kimi CLI transition changes flags or JSON schema | Pin a tested Kimi Code version, keep captured fixtures, and document upgrades. Do not target the deprecated CLI as the default. |
| Large prompts exceed OS argv limits | Require stdin/JSON input or use Kimi Agent SDK; block release on argv-only transport. |
| Kimi has no separate stable system-prompt flag | Validate a stable agent-file route; otherwise envelope system/user text and add adherence tests. |
| `disable_tools=True` cannot be enforced | Use a tool-restricted agent profile or SDK; fail closed if neither is possible. |
| Kimi built-in model aliases vary by catalog/account | Discover and test aliases, allow `CORESMITH_KIMI_MODEL` pass-through, and avoid assuming marketing names are CLI aliases. |
| Auth works interactively but fails in CI | Test both `kimi login` state and complete `KIMI_MODEL_*` configuration; explicitly warn that bare `KIMI_API_KEY` is insufficient. |
| Parallel calls contend on shared Kimi state | Add an eight-call concurrency test; use unique sessions/workdirs while sharing read-only auth/config state. |
| Current Kimi print mode waits on background work for a long time | Set supported print/background bounds when available and retain Coresmith's hard timeout, stall watchdog, and external kill registry. |
| Container Node version is too old for npm package | Prefer the verified standalone artifact or separately qualify a Node upgrade before changing the base image. |

## Estimated patch sequence

1. Compatibility spike plus sanitized fixtures.
2. Provider/model/binary/parser unit patch.
3. Watchdog/cwd/retry/tool-policy integration patch.
4. Live smoke and one small Coresmith end-to-end run.
5. Container, preflight, environment template, and documentation patch.

Keeping these as separate commits makes it easy to review the provider behavior before taking on packaging changes and easy to revert a Kimi version bump independently.
