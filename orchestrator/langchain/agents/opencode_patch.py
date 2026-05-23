# Minimal monkey-patch: replace ClaudeLLM._generate_via_cli with an opencode equivalent.
# Drop this into orchestrator/langchain/agents/ as opencode_patch.py and import it from
# wherever you'd normally import ClaudeLLM (or use sitecustomize.py to auto-load it).

from __future__ import annotations
from orchestrator._timeouts import scaled
import logging
import os
import shutil
import subprocess
import time

from orchestrator.langchain.agents import coresmith_llm as _slm

logger = logging.getLogger(__name__)

# Find opencode binary (npm-global install puts it at /usr/local/bin/opencode)
_OPENCODE_PATH = shutil.which("opencode") or "/usr/local/bin/opencode"

# How coresmith names models -> what opencode calls them.
# coresmith config.yaml lets you pin per-agent models; map them to whatever local
# provider you've registered in ~/.config/opencode/opencode.json.
# Examples assume a "local" provider pointing at a vLLM endpoint with
# Qwen/Qwen3.6-27B registered.
_MODEL_MAP = {
    "claude-opus-4-7":            "local/Qwen/Qwen3.6-27B",
    "claude-sonnet-4-6":          "local/Qwen/Qwen3.6-27B",
    "claude-haiku-4-5-20251001":  "local/Qwen/Qwen3.6-27B",
}


def _opencode_call(self, system_prompt: str, user_prompt: str) -> str:
    """Drop-in for ClaudeLLM._generate_via_cli, backed by `opencode run`."""
    resolved = _slm._resolve_model(self.model)
    opencode_model = _MODEL_MAP.get(resolved, "local/Qwen/Qwen3.6-27B")

    # opencode doesn't have --system-prompt; prepend as a system marker.
    full_prompt = (
        f"<system>{system_prompt}</system>\n\n{user_prompt}"
        if system_prompt else user_prompt
    )

    cmd = [
        _OPENCODE_PATH,
        "run",
        "--model", opencode_model,
        # opencode run reads positional message args
        full_prompt,
    ]
    logger.info("opencode invocation: model=%s prompt_len=%d", opencode_model, len(full_prompt))

    t0 = time.monotonic()
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=scaled(self._STALL_THRESHOLD_S),
        cwd=os.environ.get("CORESMITH_PROJECT_ROOT", os.getcwd()),
    )
    elapsed = time.monotonic() - t0

    if proc.returncode != 0:
        raise RuntimeError(
            f"opencode failed (rc={proc.returncode}, t={elapsed:.1f}s): "
            f"{proc.stderr[:500]}"
        )

    # opencode run prints the assistant's final reply to stdout in default mode.
    # Strip TUI escape sequences.
    import re
    text = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", proc.stdout).strip()

    logger.info("opencode reply: chars=%d elapsed=%.1fs", len(text), elapsed)
    return text


# Monkey-patch
_slm.ClaudeLLM._generate_via_cli = _opencode_call
_slm.ClaudeLLM.claude_path = _OPENCODE_PATH  # bypass the "Claude CLI not found" check
logger.warning("ClaudeLLM._generate_via_cli has been monkey-patched to use opencode")
