#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# runpod_entrypoint.sh -- container entrypoint for the coresmith Docker image.
#
# Picks one of three modes based on the CORESMITH_MODE env var:
#
#   CORESMITH_MODE=pipeline   Start coresmithd in foreground (driven by an outer agent)
#   CORESMITH_MODE=mcp        Start the MCP server on stdio (for `claude --mcp ...`)
#   CORESMITH_MODE=mcp-http   Start the MCP server behind mcp-proxy on $MCP_PORT (default 8765)
#   CORESMITH_MODE=shell      Drop into an interactive bash (default for `docker run -it`)
#
# Authentication:
#   - If $CLAUDE_CODE_OAUTH_TOKEN is set, it is exported untouched -- the
#     Claude CLI picks it up via its standard env lookup.
#   - Else if $ANTHROPIC_API_KEY is set, it is exported as well.
#   - Else, if no token is found, the entrypoint prints a clear error and
#     exits non-zero before starting any pipeline work, so users don't burn
#     a RunPod hour on a misconfigured pod.
#
# Optional overrides:
#   CORESMITH_MODEL=sonnet-5   Pin a specific model (short name or full ID)
#   CORESMITH_REQUIREMENTS_FILE  Path to a text file with architecture requirements
#                              (used in mcp / pipeline modes that need a starter prompt)
#   PUBLIC_KEY                 SSH public key to install for root. If set, the
#                              entrypoint starts sshd in the background so the
#                              container is reachable on port 22 (RunPod sets
#                              this automatically from the pod template).
#   CORESMITH_KEEP_ALIVE=1       In pipeline mode, keep the container running
#                              after the pipeline exits (so SSH / RunPod web
#                              terminal can still attach for inspection).
#                              Implied when PUBLIC_KEY is set.

set -euo pipefail

# Pick up build-time-baked env (currently: CLAUDE_CLI_PATH so the orchestrator
# can't fail to resolve `claude` if a downstream PATH munge drops the nix
# profile dir).
if [[ -f /etc/coresmith.env ]]; then
    # shellcheck disable=SC1091
    . /etc/coresmith.env
    [[ -n "${CLAUDE_CLI_PATH:-}" ]] && export CLAUDE_CLI_PATH
    [[ -n "${OPENCODE_CLI_PATH:-}" ]] && export OPENCODE_CLI_PATH
    [[ -n "${KIMI_CLI_PATH:-}" ]] && export KIMI_CLI_PATH
fi

# --- Project root -----------------------------------------------------------
PROJECT_ROOT="${CORESMITH_PROJECT_ROOT:-/coresmith}"
cd "${PROJECT_ROOT}"

mode="${CORESMITH_MODE:-shell}"
echo "[coresmith] entrypoint: mode=${mode} project_root=${PROJECT_ROOT}"

# --- Optional sshd bootstrap (RunPod / interactive use) ---------------------
# Started for every mode -- harmless if no PUBLIC_KEY is set (returns early).
maybe_start_sshd() {
    if [[ -z "${PUBLIC_KEY:-}" ]]; then
        return 0
    fi
    local sshd_bin
    sshd_bin="$(command -v sshd 2>/dev/null || true)"
    if [[ -z "${sshd_bin}" ]]; then
        echo "[coresmith] PUBLIC_KEY set but sshd not installed; skipping ssh setup" >&2
        return 0
    fi
    mkdir -p /root/.ssh /var/run/sshd /run/sshd
    chmod 700 /root/.ssh
    if ! grep -qxF "${PUBLIC_KEY}" /root/.ssh/authorized_keys 2>/dev/null; then
        printf '%s\n' "${PUBLIC_KEY}" >> /root/.ssh/authorized_keys
    fi
    chmod 600 /root/.ssh/authorized_keys
    if "${sshd_bin}"; then
        echo "[coresmith] sshd started on :22 (PUBLIC_KEY accepted)"
    else
        echo "[coresmith] sshd failed to start (rc=$?)" >&2
    fi
}
maybe_start_sshd

# --- Auth check (skipped in shell mode so users can poke around) ------------
require_auth() {
    local provider="${CORESMITH_LLM_PROVIDER:-claude}"
    if [[ "${provider}" == "opencode" || "${provider}" == "opencode_cli" || "${provider}" == "openrouter" ]]; then
        if [[ -n "${OPENROUTER_API_KEY:-}" ]]; then
            echo "[coresmith] auth: using OPENROUTER_API_KEY with OpenCode"
            export OPENROUTER_API_KEY
            return 0
        fi
        local opencode_auth="${XDG_DATA_HOME:-${HOME:-/root}/.local/share}/opencode/auth.json"
        if [[ -f "${opencode_auth}" ]] && grep -q '"openrouter"' "${opencode_auth}"; then
            echo "[coresmith] auth: using OpenCode credentials at ${opencode_auth}"
            return 0
        fi
        cat <<'MSG' >&2

[coresmith] ERROR: no OpenRouter credentials found for OpenCode.

Set OPENROUTER_API_KEY in the pod/container environment, or persist the
OpenCode auth directory after running `opencode auth login` and selecting
OpenRouter. Set CORESMITH_MODE=shell to authenticate interactively.

MSG
        exit 2
    fi
    if [[ "${provider}" == "kimi" || "${provider}" == "kimi_cli" ]]; then
        if [[ -n "${KIMI_MODEL_NAME:-}" && -n "${KIMI_MODEL_API_KEY:-}" ]]; then
            echo "[coresmith] auth: using KIMI_MODEL_NAME/KIMI_MODEL_API_KEY"
            export KIMI_MODEL_NAME KIMI_MODEL_API_KEY
            return 0
        fi
        local kimi_home="${KIMI_CODE_HOME:-${HOME:-/root}/.kimi-code}"
        if [[ -f "${kimi_home}/config.toml" ]]; then
            echo "[coresmith] auth: using Kimi Code config at ${kimi_home}"
            export KIMI_CODE_HOME="${kimi_home}"
            return 0
        fi
        cat <<'MSG' >&2

[coresmith] ERROR: no Kimi Code credentials found.

Either persist a prior `kimi login` directory and set KIMI_CODE_HOME, or set:

  KIMI_MODEL_NAME       -- model id for the temporary provider
  KIMI_MODEL_API_KEY    -- Kimi Platform API key

Set CORESMITH_MODE=shell to enter the container and run `kimi login`.

MSG
        exit 2
    fi

    if [[ -n "${CLAUDE_CODE_OAUTH_TOKEN:-}" ]]; then

        echo "[coresmith] auth: using CLAUDE_CODE_OAUTH_TOKEN"
        export CLAUDE_CODE_OAUTH_TOKEN
        return 0
    fi
    if [[ -n "${ANTHROPIC_API_KEY:-}" ]]; then
        echo "[coresmith] auth: using ANTHROPIC_API_KEY"
        export ANTHROPIC_API_KEY
        return 0
    fi
    cat <<'MSG' >&2

[coresmith] ERROR: no Claude credentials found.

Set one of:

  ANTHROPIC_API_KEY        -- API key from https://console.anthropic.com
  CLAUDE_CODE_OAUTH_TOKEN  -- OAuth token from `claude setup-token`

For RunPod, add the variable in the pod template's "Environment Variables"
section. For docker run, pass it via -e:

  docker run -e ANTHROPIC_API_KEY=sk-ant-... coresmith:latest

Set CORESMITH_MODE=shell to drop into a debug shell without auth.

MSG
    exit 2
}

# --- Optional model override echo -------------------------------------------
if [[ -n "${CORESMITH_MODEL:-}" ]]; then
    echo "[coresmith] model override: CORESMITH_MODEL=${CORESMITH_MODEL}"
fi

# --- Preflight: bail early if the toolchain is broken (skipped in shell) ---
run_preflight() {
    echo "[coresmith] running preflight..."
    if ! python3 -c "
from orchestrator.langgraph.pipeline_helpers import preflight_check
import json, sys
r = preflight_check(['pipeline', 'backend'])
print(json.dumps(r, indent=2))
sys.exit(0 if r['ok'] else 1)
"; then
        echo "[coresmith] preflight failed; aborting." >&2
        exit 3
    fi
}

case "${mode}" in
    shell)
        # If extra args were passed (e.g. `docker run image bash -lc '...'`),
        # exec them directly rather than wrapping in another bash. This
        # also avoids the openlane2 base's missing /bin/bash hardlink --
        # we let exec resolve via PATH (Nix-store bash is on it).
        if [[ $# -gt 0 ]]; then
            exec "$@"
        fi
        echo "[coresmith] dropping into bash. Try: make help"
        exec bash
        ;;

    pipeline)
        require_auth
        run_preflight
        log_file="${CORESMITH_PIPELINE_LOG:-/coresmith/.coresmith/pipeline.log}"
        mkdir -p "$(dirname "${log_file}")"
        echo "[coresmith] starting coresmithd in the foreground; log -> ${log_file}"

        # The daemon parks on every interrupt and does NOT auto-approve. An
        # outer agent (Claude on cron, a human, or another script) is
        # expected to drive resume decisions via `bin/coresmith resume`.
        # If you want a fully unattended run, write a small driver that
        # polls /run/state and POSTs /run/resume; do not bake auto-approve
        # back into the daemon.
        set +e
        export CORESMITH_PROJECT_ROOT="${CORESMITH_PROJECT_ROOT:-/coresmith}"
        python3 -m orchestrator.daemon.server "$@" 2>&1 | tee "${log_file}"
        rc=${PIPESTATUS[0]}
        set -e
        echo "[coresmith] daemon exited rc=${rc}"

        # Keep PID 1 alive on RunPod (PUBLIC_KEY set) or when the operator
        # explicitly opts in via CORESMITH_KEEP_ALIVE=1, so SSH and the
        # RunPod web terminal can still attach for post-mortem. For plain
        # `docker run` users with neither set, exit cleanly with the
        # pipeline's rc.
        if [[ "${CORESMITH_KEEP_ALIVE:-0}" == "1" ]] || [[ -n "${PUBLIC_KEY:-}" ]]; then
            echo "[coresmith] keeping container alive for inspection (rc=${rc})."
            echo "[coresmith] tailing ${log_file} as PID 1; ssh in to inspect state."
            exec tail -F "${log_file}"
        fi
        exit "${rc}"
        ;;

    mcp)
        require_auth
        run_preflight
        echo "[coresmith] starting MCP server on stdio"
        exec python3 -m orchestrator.mcp_server "$@"
        ;;

    mcp-http)
        require_auth
        run_preflight
        port="${MCP_PORT:-8765}"
        if ! command -v mcp-proxy >/dev/null 2>&1; then
            echo "[coresmith] mcp-proxy not installed; installing now (npm)"
            npm install -g @modelcontextprotocol/proxy >/dev/null 2>&1 \
                || pip install mcp-proxy >/dev/null 2>&1 \
                || { echo "[coresmith] cannot install mcp-proxy" >&2; exit 4; }
        fi
        echo "[coresmith] starting MCP server behind HTTP proxy on :${port}"
        exec mcp-proxy --port "${port}" -- python3 -m orchestrator.mcp_server "$@"
        ;;

    test)
        run_preflight || true
        echo "[coresmith] running test suite (excluding live_llm and e2e)"
        exec python3 -m pytest orchestrator/tests/ -v \
            -m "not live_llm and not requires_nix and not e2e" "$@"
        ;;

    *)
        echo "[coresmith] unknown CORESMITH_MODE=${mode}" >&2
        echo "  valid: shell | pipeline | mcp | mcp-http | test" >&2
        exit 1
        ;;
esac
