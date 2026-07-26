# OpenCode + OpenRouter Kimi K3 support

## Goal

Make OpenRouter's hosted Kimi K3 a first-class CoreSmith LLM route through the
official OpenCode CLI, without replacing Claude, Codex, agy, or test providers.

The selected OpenCode model identifier is:

```text
openrouter/moonshotai/kimi-k3
```

OpenRouter's live catalog reports the underlying model as
`moonshotai/kimi-k3` with a 1,048,576-token context window.

## CLI and authentication

OpenCode provides a headless command:

```bash
opencode --pure run --format json --model openrouter/moonshotai/kimi-k3
```

CoreSmith sends the combined system/user prompt on stdin rather than the
argument list. Authentication stays owned by OpenCode:

- developer machine: `opencode auth login`, then select OpenRouter;
- CI/container: `OPENROUTER_API_KEY`;
- persistent worker: mount OpenCode's `auth.json` data directory.

Secrets are never copied into CoreSmith logs or configuration.

## Implementation

1. Add `opencode`, `opencode_cli`, and `openrouter` provider aliases.
2. Resolve every CoreSmith model tier to hosted Kimi K3 by default, with
   `CORESMITH_OPENCODE_MODEL` as an explicit override.
3. Invoke `opencode run` with `--pure`, `--thinking`, `--format json`, the
   selected model and work directory, and `--auto`; pipe prompts through stdin.
4. Parse text, token, cache, reasoning, and cost fields from OpenCode NDJSON,
   and persist every valid raw event to `.coresmith/opencode_turns.jsonl`.
5. Preserve `OPENCODE_CONFIG_CONTENT`; when tools are disabled, merge
   `permission: "deny"` into that JSON object.
6. Reuse CoreSmith's timeout, stall detection, process-group cleanup,
   telemetry, and circuit breaker.
7. Install OpenCode in the container, expose its resolved binary path, and
   validate explicit OpenCode configurations during pipeline preflight.
8. Support OpenRouter environment credentials or a mounted OpenCode auth file
   in the RunPod entrypoint.

## Validation

- model mapping and provider aliases;
- captured OpenCode NDJSON parsing and raw reasoning-event persistence;
- exact command construction and stdin prompt transport;
- tool-deny config merging without dropping existing config;
- existing LLM provider regression suite;
- real authenticated `ready` smoke call against hosted Kimi K3;
- Docker/RunPod shell syntax and Python compilation checks.

## Acceptance criteria

- `CORESMITH_LLM_PROVIDER=opencode` constructs a hosted Kimi K3 call;
- the default `opus-4.8` and block-tier aliases all resolve to Kimi K3;
- missing OpenCode binaries and credentials fail with actionable messages;
- no API key appears in argv, logs, docs examples, or committed files;
- Claude, Codex, agy, fault, and replay behavior remains unchanged.
