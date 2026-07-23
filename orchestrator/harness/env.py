# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Project-root + child-process environment bootstrap for the harness.

``pipeline_helpers.PROJECT_ROOT`` is frozen at import time from
``CORESMITH_PROJECT_ROOT``. Any ``coresmith verify`` invocation must therefore
resolve + export the project root BEFORE importing langgraph, and must NOT rely
on the process cwd (agents run from arbitrary directories).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]


def repo_root() -> Path:
    return _REPO_ROOT


def bootstrap_project_root(arg: Optional[str] = None) -> Path:
    """Resolve the project root and export ``CORESMITH_PROJECT_ROOT``.

    Resolution order: explicit ``arg`` -> ``$CORESMITH_PROJECT_ROOT`` -> error.
    Never falls back to cwd (agents are launched from arbitrary directories,
    per the plan's B5 risk note). Sets the env var so a *subsequent* langgraph
    import picks up the right root at freeze time.

    Raises ``ValueError`` when no project root can be determined.
    """
    candidate = (arg or "").strip() or os.environ.get("CORESMITH_PROJECT_ROOT", "").strip()
    if not candidate:
        raise ValueError(
            "no project root: pass --project-root or set CORESMITH_PROJECT_ROOT",
        )
    root = Path(candidate).expanduser().resolve()
    os.environ["CORESMITH_PROJECT_ROOT"] = str(root)
    return root


def harness_child_env(extra: Optional[dict] = None) -> dict:
    """Environment for spawned harness children (make/verilator/yosys/cocotb).

    Prepends the repo venv bin + the repo ``bin/`` to PATH so ``coresmith`` and
    the venv toolchain resolve, and preserves PYTHONHOME (cocotb 2.x pygpi needs
    it). Mirrors ``bin/coresmith`` daemon-spawn env handling.
    """
    env = os.environ.copy()
    parts: list[str] = []
    venv_bin = _REPO_ROOT / "venv" / "bin"
    if venv_bin.is_dir():
        parts.append(str(venv_bin))
        env.setdefault("VIRTUAL_ENV", str(_REPO_ROOT / "venv"))
    repo_bin = _REPO_ROOT / "bin"
    if repo_bin.is_dir():
        parts.append(str(repo_bin))
    if parts:
        env["PATH"] = os.pathsep.join(parts + [env.get("PATH", "/usr/bin:/bin")])
    # Preserve PYTHONHOME for cocotb's Verilator-embedded interpreter (pygpi).
    if "PYTHONHOME" not in env:
        try:
            import sysconfig
            prefix = sysconfig.get_config_var("prefix")
            if prefix:
                env["PYTHONHOME"] = prefix
        except Exception:  # noqa: BLE001
            pass
    if extra:
        env.update({str(k): str(v) for k, v in extra.items()})
    return env


def ensure_repo_on_syspath() -> None:
    """Make ``orchestrator.*`` importable when invoked via ``bin/coresmith``."""
    rr = str(_REPO_ROOT)
    if rr not in sys.path:
        sys.path.insert(0, rr)


def ensure_cli_symlink(cli_path, bindir: str = "/usr/local/bin"):
    """Best-effort: refresh a ``coresmith`` symlink in ``bindir`` -> ``cli_path``.

    Belt-and-suspenders for Defect 1: codex runs agent commands via
    ``/bin/bash -lc`` and ``/etc/profile`` can unconditionally reassign ``PATH``,
    dropping the repo ``bin/`` prefix the daemon injected. A stable
    ``/usr/local/bin/coresmith`` symlink survives that reset.

    NEVER raises (observability/robustness helper): returns the ``Path`` of the
    created/refreshed symlink on success, or ``None`` when ``bindir`` is
    absent/unwritable, a non-symlink file already occupies the target (we don't
    clobber it), or any OS error occurs. Idempotent -- a symlink already
    pointing at ``cli_path`` is a no-op success.
    """
    try:
        cli_path = Path(cli_path).resolve()
        bd = Path(bindir)
        if not bd.is_dir() or not os.access(bd, os.W_OK):
            return None
        link = bd / "coresmith"
        if link.is_symlink():
            try:
                if link.resolve() == cli_path:
                    return link
            except OSError:
                pass
            link.unlink()
        elif link.exists():
            # A real (non-symlink) file already lives here -- don't clobber it.
            return None
        link.symlink_to(cli_path)
        return link
    except OSError:
        return None
