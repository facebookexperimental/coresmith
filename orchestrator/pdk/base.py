# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Bring-your-own-PDK / bring-your-own-EDA abstraction layer (ABCs).

A *Deployment* is the whole EDA/PDK environment expressed as one Python object:
it maps named tool *verbs* (``run_synth``, ``run_pnr``, ``run_drc``, ``run_lvs``,
``run_sta``, ``run_lint``, ...) to concrete :class:`EdaTool` implementations, each
carrying its own :class:`Checker` classes that parse that tool's output into a
normalized, three-state verdict. The engine and the CLI stop invoking
``yosys`` / ``openroad`` / ``magic`` by name and instead go through the verbs.

Design invariants (mirrors ``drc_verdict.classify_drc`` + ``gate_guard``):

* Checkers are **fail-closed**: a checker that cannot find its report returns
  ``not_run`` and, when ``blocking``, that fails the verb -- never a false clean.
* A missing capability is an **honest skip** (``status="skip"``), never a green.

Stdlib-only (plus the existing :class:`PDKConfig`) so this module stays importable
without pulling in ``orchestrator.langgraph`` (the harness CLI depends on that).
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal

from orchestrator.pdk.pdk_config import PDKConfig

# The closed verb set the engine + prompts key off. A deployment may expose
# extra verbs (surfaced via ``tool list``) but the engine only requires the
# ones its enabled gates need.
VERBS: tuple[str, ...] = (
    "run_synth",
    "run_pnr",
    "run_drc",
    "run_lvs",
    "run_sta",
    "run_lint",
    "run_sim",
    "run_gate_sim",
    "gen_macro",
)

CheckStatus = Literal["pass", "fail", "skip", "not_run"]


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------
@dataclass
class ToolRequest:
    """One invocation of a verb.

    ``inputs`` / ``params`` are verb-specific: ``inputs`` carries file paths
    (``rtl``, ``script``, ``netlist``, ``sdc``, ``gds``, ``spice``, ...) and
    ``params`` carries scalars (``clock_ns``, ``corner``, ``utilization``, ...).
    """

    verb: str
    design: str
    inputs: dict[str, Path] = field(default_factory=dict)
    out_dir: Path | None = None
    params: dict[str, Any] = field(default_factory=dict)
    timeout_s: int | None = None

    def input(self, key: str) -> Path | None:
        """Return an input path by key, or ``None`` if absent."""
        v = self.inputs.get(key)
        return Path(v) if v is not None else None


@dataclass
class CheckResult:
    """A single checker's normalized verdict.

    ``status`` mirrors ``drc_verdict``'s three-state model plus ``not_run``:
    a blocking ``not_run`` (report absent) fails the verb rather than passing.
    """

    name: str
    status: CheckStatus
    metrics: dict[str, float | int | str] = field(default_factory=dict)
    details: str = ""
    blocking: bool = True

    @property
    def passed(self) -> bool:
        return self.status == "pass"

    @property
    def failed(self) -> bool:
        """A blocking checker fails when it did not pass (fail or not_run)."""
        if not self.blocking:
            return False
        return self.status in ("fail", "not_run")

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "metrics": self.metrics,
            "details": self.details,
            "blocking": self.blocking,
        }


@dataclass
class ToolResult:
    """The result of running a verb + all its checkers.

    ``ok`` is the composed verdict (tool ran AND every blocking checker passed);
    ``tool_ok`` is process-level success only (rc == 0, no timeout, binary
    present). The CLI keys its exit code off both:
    ``0`` iff ``ok``; ``3`` when not ``tool_ok`` (infra); ``1`` otherwise.
    """

    ok: bool
    tool_ok: bool
    checks: list[CheckResult] = field(default_factory=list)
    artifacts: dict[str, Path] = field(default_factory=dict)
    log_path: Path | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    verb: str = ""
    design: str = ""

    @classmethod
    def from_checks(
        cls,
        *,
        tool_ok: bool,
        checks: list[CheckResult],
        artifacts: dict[str, Path] | None = None,
        log_path: Path | None = None,
        verb: str = "",
        design: str = "",
        extra_metrics: dict[str, Any] | None = None,
    ) -> ToolResult:
        """Compose a result: ``ok`` iff the tool ran and no blocking check failed.

        Merges every checker's ``metrics`` into the top-level ``metrics`` map
        (for gates/telemetry), then overlays ``extra_metrics``.
        """
        metrics: dict[str, Any] = {}
        for c in checks:
            metrics.update(c.metrics)
        if extra_metrics:
            metrics.update(extra_metrics)
        ok = tool_ok and not any(c.failed for c in checks)
        return cls(
            ok=ok,
            tool_ok=tool_ok,
            checks=list(checks),
            artifacts=dict(artifacts or {}),
            log_path=log_path,
            metrics=metrics,
            verb=verb,
            design=design,
        )

    @classmethod
    def skipped(cls, verb: str, reason: str, design: str = "") -> ToolResult:
        """An honest capability skip: the deployment does not support this verb."""
        return cls(
            ok=False,
            tool_ok=False,
            checks=[CheckResult(name=verb, status="skip", details=reason,
                                blocking=False)],
            metrics={"skipped": True, "reason": reason},
            verb=verb,
            design=design,
        )

    def to_json(self) -> dict[str, Any]:
        """The schema ``coresmith tool ... --json`` prints."""
        return {
            "verb": self.verb,
            "design": self.design,
            "ok": self.ok,
            "tool_ok": self.tool_ok,
            "checks": [c.to_json() for c in self.checks],
            "artifacts": {k: str(v) for k, v in self.artifacts.items()},
            "log_path": str(self.log_path) if self.log_path else None,
            "metrics": self.metrics,
        }

    def to_json_str(self) -> str:
        return json.dumps(self.to_json(), indent=2, default=str)


# ---------------------------------------------------------------------------
# ABCs
# ---------------------------------------------------------------------------
class Checker(ABC):
    """Parses one tool's output into a :class:`CheckResult`.

    Subclasses set ``name`` (and optionally ``blocking = False`` for advisory
    checks) and implement :meth:`check`. A checker that cannot find its report
    MUST return ``status="not_run"`` (fail-closed), never ``pass``.
    """

    name: ClassVar[str] = "check"
    blocking: ClassVar[bool] = True

    @abstractmethod
    def check(self, req: ToolRequest, run_dir: Path) -> CheckResult:
        ...


class EdaTool(ABC):
    """A concrete implementation of one verb (e.g. yosys for ``run_synth``)."""

    verb: ClassVar[str] = ""

    def __init__(self, deployment: "Deployment") -> None:
        self.deployment = deployment

    @abstractmethod
    def run(self, req: ToolRequest) -> ToolResult:
        """Invoke the tool and run :meth:`checkers`, returning a composed result."""
        ...

    def checkers(self) -> list[Checker]:
        """The checker classes attached to this tool (default: none)."""
        return []

    def reference_script(self) -> Path | None:
        """A template script an agent can copy + adapt (``tool emit-script``)."""
        return None

    def prompt_notes(self) -> str:
        """Tool-specific instructions injected into agent prompts."""
        return ""


class Deployment(ABC):
    """The whole EDA/PDK environment as one object."""

    name: ClassVar[str] = ""

    @property
    @abstractmethod
    def pdk(self) -> PDKConfig:
        """The PDK configuration (paths, corners, cell library)."""
        ...

    @abstractmethod
    def tools(self) -> dict[str, EdaTool]:
        """Map of verb -> implementation."""
        ...

    def tool(self, verb: str) -> EdaTool | None:
        """Return the tool for ``verb`` if supported, else ``None``."""
        return self.tools().get(verb)

    def supports(self, verb: str) -> bool:
        return verb in self.tools()

    def capabilities(self) -> set[str]:
        """The verbs this deployment implements (drives honest-skip)."""
        return set(self.tools().keys())

    def prompt_context(self) -> dict[str, str]:
        """Fields prompts ``.format()`` with (``pdk_summary``, ``tool_notes``...)."""
        try:
            summary = self.pdk.to_summary()
        except Exception:  # noqa: BLE001
            summary = self.name
        return {
            "deployment": self.name,
            "pdk_summary": summary,
            "tool_notes": "",
        }

    def sim_models(self) -> list[Path]:
        """Gate-sim cell verilog + UDP shims (default: none)."""
        return []

    def data_dir(self) -> Path | None:
        """Optional sidecar dir for TCL/yaml templates (default: none)."""
        return None

    def describe(self) -> dict[str, Any]:
        """Machine-readable summary for ``coresmith pdk info --json``."""
        tools = self.tools()
        try:
            pdk_dict = self.pdk.to_dict()
        except Exception:  # noqa: BLE001
            pdk_dict = {"name": self.name}
        return {
            "deployment": self.name,
            "capabilities": sorted(self.capabilities()),
            "tools": {
                verb: {
                    "impl": type(t).__name__,
                    "checkers": [type(c).__name__ for c in t.checkers()],
                }
                for verb, t in sorted(tools.items())
            },
            "pdk": pdk_dict,
            "data_dir": str(self.data_dir()) if self.data_dir() else None,
        }
