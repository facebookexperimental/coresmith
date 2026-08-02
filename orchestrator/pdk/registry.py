# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Deployment resolution + process-wide cache.

``get_deployment()`` resolves the active :class:`~orchestrator.pdk.base.Deployment`
once per process. Resolution order (first match wins):

1. ``CORESMITH_DEPLOYMENT`` env var -- either a built-in name (``sky130``,
   ``mock``) OR a filesystem path to a single ``.py`` file that defines a
   module-level ``DEPLOYMENT`` attribute (loaded via ``importlib``).
2. ``deployment:`` key in ``orchestrator/config.yaml``.
3. Default ``"sky130"``.

Shape mirrors ``coresmith_llm._TESTING_PROVIDER_MODULES`` (name -> module). The
existing ``CORESMITH_BACKEND_*`` binary overrides keep working -- they are read
*inside* the sky130 deployment, not here.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
from pathlib import Path

from orchestrator.pdk.base import Deployment

# Built-in deployments live under ``orchestrator.pdk.deployments.<name>`` and
# expose a module-level ``DEPLOYMENT`` instance.
_BUILTIN_PACKAGE = "orchestrator.pdk.deployments"

_DEPLOYMENT_ATTR = "DEPLOYMENT"

_cache: Deployment | None = None


def reset_deployment_cache() -> None:
    """Clear the cached deployment (tests set env then re-resolve)."""
    global _cache
    _cache = None


def get_deployment() -> Deployment:
    """Return the active deployment, resolving + caching on first call."""
    global _cache
    if _cache is None:
        _cache = _resolve()
    return _cache


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------
def _resolve() -> Deployment:
    spec = os.environ.get("CORESMITH_DEPLOYMENT", "").strip()
    if not spec:
        spec = _config_deployment()
    if not spec:
        spec = "sky130"
    return load_deployment(spec)


def _config_deployment() -> str:
    """Read the ``deployment:`` key from the project config, best-effort."""
    try:
        # Deferred: load_config lives in langgraph and freezes PROJECT_ROOT at
        # import; only touch it at resolution time (never at module import).
        from orchestrator.langgraph.pipeline_helpers import load_config

        cfg = load_config()
        val = cfg.get("deployment", "")
        return str(val).strip() if val else ""
    except Exception:  # noqa: BLE001
        return ""


def load_deployment(spec: str) -> Deployment:
    """Load a deployment from a built-in name or a filesystem ``.py`` path."""
    if _looks_like_path(spec):
        return _load_from_path(Path(spec).expanduser())
    return _load_builtin(spec)


def _looks_like_path(spec: str) -> bool:
    if spec.endswith(".py") or os.sep in spec:
        return True
    return bool(os.altsep) and os.altsep in spec


def _load_builtin(name: str) -> Deployment:
    module_name = f"{_BUILTIN_PACKAGE}.{name}"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise ValueError(
            f"unknown deployment '{name}': no built-in module {module_name} "
            f"(set CORESMITH_DEPLOYMENT to a built-in name or a .py path)"
        ) from exc
    return _extract(module, module_name)


def _load_from_path(path: Path) -> Deployment:
    if not path.is_file():
        raise ValueError(
            f"deployment file not found: {path} "
            f"(CORESMITH_DEPLOYMENT must be a built-in name or an existing .py file)"
        )
    module_name = f"_coresmith_deployment_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load deployment module from {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"failed to import deployment {path}: {exc}") from exc
    return _extract(module, str(path))


def _extract(module, source: str) -> Deployment:
    dep = getattr(module, _DEPLOYMENT_ATTR, None)
    if dep is None:
        raise ValueError(
            f"deployment source {source} has no module-level "
            f"'{_DEPLOYMENT_ATTR}' attribute"
        )
    if not isinstance(dep, Deployment):
        raise ValueError(
            f"{source}.{_DEPLOYMENT_ATTR} is {type(dep).__name__}, "
            f"not a Deployment subclass instance"
        )
    return dep
