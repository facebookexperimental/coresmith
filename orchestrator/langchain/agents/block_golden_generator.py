# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
BlockGoldenGenerator -- emits a per-block **Amaranth block model**.

(The class name is kept for call-site compatibility.)

Given the design's **reference implementation** (the input software golden for
the whole chip) and a block's frozen interface contract, this agent writes an
executable Amaranth model for that single block to
``arch/block_models/<block>.py``. The emitted module defines an
``Elaboratable`` class NAMED EXACTLY ``<block>`` whose constructor signature is
``(clk, rst, <handshake + data ports from the contract>)`` and whose
``elaborate()`` method transcribes THIS block's reference math.

The point is to produce per-block models that a later **model-integration
agent** can wire into a top-level Amaranth chip model; a deterministic gate then
simulates that chip model and proves the composed chip output equals the
reference implementation BEFORE any RTL is trusted.

This mirrors UarchSpecGenerator's wiring: ClaudeLLM(model, timeout); system
prompt from prompts/block_golden_generator.md; ``await self.llm.call(...)``.
The emitted file is validated (ast.parse + a guarded import) to confirm it
imports and defines an ``Elaboratable`` class whose signature includes the
contract's ports before it is trusted.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import re
from pathlib import Path
from typing import Any

from opentelemetry import trace

from orchestrator._timeouts import scaled
from orchestrator.langchain.prompts.skills import load_skills as _load_skills

from .coresmith_llm import ClaudeLLM

_tracer = trace.get_tracer(__name__)

_PROMPT_FILE = (
    Path(__file__).resolve().parent.parent
    / "prompts"
    / "block_golden_generator.md"
)
if _PROMPT_FILE.exists():
    SYSTEM_PROMPT = _PROMPT_FILE.read_text()
else:  # pragma: no cover - prompt ships with the repo
    SYSTEM_PROMPT = (
        "Emit a per-block Amaranth block model: an Elaboratable class named "
        "after the block whose elaborate() method transcribes the reference "
        "implementation's exact math for this block, with clock/valid-ready/"
        "latency semantics."
    )

# Inject the shared streaming-protocol skills so generated block models follow
# the same handshake + framing (tvalid/tready, tlast/tuser) conventions every
# other coresmith agent uses -- this is what lets the blocks COMPOSE.
_SKILLS_TEXT = _load_skills(
    "axi_stream", "srdy_drdy", "arithmetic_precision", "serialization_contract",
    "buffer_stride_contract", "no_stimulus_keyed_memorization", "pipeline_contract")
if _SKILLS_TEXT:
    SYSTEM_PROMPT = (
        SYSTEM_PROMPT
        + "\n\n# Reference Skills (streaming protocol — follow these)\n\n"
        + _SKILLS_TEXT
    )


def _contract_port_names(block_ports: Any, interface_contract: Any) -> list[str]:
    """Best-effort extraction of data-port names from the contract / ports.

    Used only to make the validator lenient: we confirm the emitted factory
    references the block's contract ports, but we accept conventional clock/
    reset/handshake names the contract may not enumerate. Never raises.
    """
    names: list[str] = []

    def _collect(obj: Any) -> None:
        if isinstance(obj, dict):
            # block-diagram interfaces: {port_name: {...}}
            for k, v in obj.items():
                if isinstance(k, str):
                    names.append(k)
                _collect(v)
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict) and isinstance(item.get("name"), str):
                    names.append(item["name"])
                else:
                    _collect(item)

    try:
        _collect(block_ports)
    except Exception:  # noqa: BLE001
        pass
    try:
        if not isinstance(interface_contract, str):
            _collect(interface_contract)
    except Exception:  # noqa: BLE001
        pass
    # De-dup, keep order.
    seen: set[str] = set()
    out: list[str] = []
    for n in names:
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out


class BlockGoldenGenerator:
    """Agent that produces an Amaranth block model (.py) for one block."""

    def __init__(self, model: str | None = None, temperature: float = 0.1):
        from orchestrator.langchain.agents.coresmith_llm import arch_reasoning_effort, block_model

        model = model or block_model()
        # The block model is generated inside the uarch stage and is the
        # oracle every DV verdict rests on -- frontier blocks (full datapath
        # transcription) are exactly where extra reasoning pays. Same tier as
        # the other architecture-stage calls (codex-only; no-op elsewhere).
        self.llm = ClaudeLLM(
            model=model,
            timeout=scaled(2700, env="CORESMITH_BLOCK_GOLDEN_TIMEOUT"),
            reasoning_effort=arch_reasoning_effort(),
        )

    async def generate(
        self,
        block_name: str,
        block_ports: dict[str, Any],
        interface_contract: Any,
        reference_impl_source: str,
        reference_impl_path: str,
        project_root: str,
        output_path: str,
        slice_functions: list[str] | None = None,
        slice_regions: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Generate ``arch/block_models/<block>.py`` (an Amaranth block model).

        Args:
            block_name: HW block name. The emitted ``Elaboratable`` class MUST be
                named exactly this.
            block_ports: the block's interface ports (from block_diagram /
                interface contract).
            interface_contract: the per-block frozen contract slice (the
                authoritative bit-level ports / packing / handshake). May be any
                JSON-serialisable structure or a prompt-ready string.
            reference_impl_source: source of the reference implementation
                (the input software golden for the whole chip).
            reference_impl_path: path to the reference implementation (for
                citation context).
            project_root: project root path.
            output_path: absolute path to write the block model .py to.
            slice_functions: the authoritative, deterministic list of golden
                function names this block is responsible for (from
                ``block_complexity.resolve_block_slice_regions``). When present it
                is given to the LLM as a focused hint (transcribe THESE functions'
                math, not the whole golden) and persisted alongside the model.
            slice_regions: ``[{"fn", "start", "end"}]`` 1-based line spans for the
                slice, persisted to the ``.slice.json`` sidecar so downstream
                gates (golden-feasibility, complexity advisory) consume the exact
                block->golden mapping instead of re-deriving it.

        Returns:
            ``{"path", "slice_path", "golden_functions"}`` -- ``slice_path`` is
            None when no slice was resolved (the generator saw the whole golden,
            i.e. today's behaviour).

        Raises:
            RuntimeError: if the agent did not emit a valid Amaranth block model
                (missing file, syntax error, no ``Elaboratable`` class named
                ``<block>``, or failed guarded import).
        """
        import json as _json

        block_title = block_name.replace("_", " ").title()
        span_name = f"Block Model [{block_title}]"

        with _tracer.start_as_current_span(span_name) as span:
            span.set_attribute("block_name", block_name)

            try:
                ports_text = _json.dumps(block_ports, indent=2)
            except (TypeError, ValueError):
                ports_text = str(block_ports)
            try:
                contract_text = (
                    interface_contract
                    if isinstance(interface_contract, str)
                    else _json.dumps(interface_contract, indent=2)
                )
            except (TypeError, ValueError):
                contract_text = str(interface_contract)

            slice_hint_lines: list[str] = []
            if slice_functions:
                slice_hint_lines = [
                    "\n--- THIS BLOCK'S GOLDEN SLICE (authoritative) ---",
                    "The reference implementation below is the WHOLE chip golden. "
                    "This block is responsible for exactly these functions; "
                    "transcribe THEIR math and only theirs (other blocks own the "
                    "rest):",
                    ", ".join(slice_functions),
                ]

            user_message = "\n".join(
                [
                    "Emit an Amaranth block model for the following HW block.",
                    f"\nThe Elaboratable class MUST be named exactly: {block_name}",
                    "\n--- BLOCK INTERFACE PORTS (from block diagram) ---",
                    ports_text,
                    "\n--- FROZEN INTERFACE CONTRACT (this block's edges: "
                    "handshake protocol, data buses, packing) ---",
                    contract_text,
                    *slice_hint_lines,
                    f"\n--- REFERENCE IMPLEMENTATION ({reference_impl_path}) ---",
                    "```python",
                    reference_impl_source,
                    "```",
                    "\nEmit ONLY the Amaranth ```python code block per the CONTRACT. "
                    "The Elaboratable class name MUST be the block name and its "
                    "constructor signature MUST include the block's clock/reset ports AS "
                    "NAMED in the contract/port list (e.g. clk/rst, or "
                    "wb_clk_i/wb_rst_i on Caravel designs) plus <the "
                    "contract's valid/ready handshake + data bus ports>. "
                    "Transcribe the "
                    "reference implementation's EXACT math for this block's "
                    "responsibility -- no heuristics, no float substitution.",
                ]
            )

            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)

            # C8: snapshot the PRE-CALL artifact so disk-first arbitration can
            # tell "the tool-enabled CLI wrote this file during THIS call"
            # from "this is the superseded model of a PREVIOUS generation".
            pre_existing = ""
            if out.exists():
                try:
                    pre_existing = out.read_text(encoding="utf-8")
                except OSError:
                    pre_existing = ""

            run_name = f"Generate Block Model [{block_title}]"
            content = await self.llm.call(
                system=SYSTEM_PROMPT,
                prompt=user_message,
                run_name=run_name,
            )

            code = self._extract_python(content)

            # Disk-first: a tool-enabled CLI may have written the file itself
            # and returned only a status string. Prefer a real on-disk module
            # over an unusable stdout snippet.
            on_disk = ""
            if out.exists():
                try:
                    on_disk = out.read_text(encoding="utf-8")
                except OSError:
                    on_disk = ""
            chosen = self._arbitrate_disk_first(
                code, on_disk, pre_existing, block_name)
            if not chosen:
                raise RuntimeError(
                    f"Block model generation for '{block_name}' produced no "
                    f"usable Python (no fenced code block and no on-disk file "
                    f"at {output_path})."
                )

            out.write_text(chosen, encoding="utf-8")

            problem = _validate_block_model_file(str(out), block_name)
            if problem is not None:
                raise RuntimeError(
                    f"Block model for '{block_name}' at {output_path} is "
                    f"invalid: {problem}"
                )

            # Persist the authoritative block->golden-slice mapping alongside the
            # model so downstream gates (golden-feasibility, complexity advisory)
            # consume the exact slice instead of re-deriving it by name-matching.
            slice_path: str | None = None
            if slice_functions:
                sidecar = out.with_suffix(".slice.json")
                try:
                    sidecar.write_text(
                        _json.dumps(
                            {
                                "block": block_name,
                                "reference_impl_path": reference_impl_path,
                                "golden_functions": list(slice_functions),
                                "regions": list(slice_regions or []),
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                    slice_path = str(sidecar)
                    span.set_attribute("slice_path", slice_path)
                except OSError:
                    slice_path = None

            span.set_attribute("path", str(out))
            return {
                "path": str(out),
                "slice_path": slice_path,
                "golden_functions": list(slice_functions or []),
            }

    @staticmethod
    def _extract_python(content: str) -> str:
        """Extract the first ```python fenced block from an LLM response."""
        match = re.search(r"```python\s*\n(.*?)```", content, re.DOTALL)
        if match:
            return match.group(1).strip()
        # No fence -- the whole content may already be code.
        return content.strip()

    @staticmethod
    def _arbitrate_disk_first(
        extracted: str, on_disk: str, pre_existing: str, block_name: str
    ) -> str:
        """C8: disk-first arbitration that cannot resurrect a superseded model.

        The disk-first preference exists for tool-enabled CLIs that write the
        file during THIS call and echo only a status string. But on a
        REGENERATION the output path already holds the PREVIOUS model -- if
        the on-disk bytes are identical to the pre-call snapshot, the CLI did
        not write them now, and they must never beat the fresh extraction
        (observed: a regeneration "chose" the superseded stub over a fresh
        INFEASIBLE-INTERFACE-GAP declaration, masking the model's honest claim
        and resurrecting the stub the regen existed to replace).
        """
        if on_disk and on_disk == pre_existing:
            on_disk = ""
        return BlockGoldenGenerator._choose_block_model(
            extracted, on_disk, block_name)

    @staticmethod
    def _choose_block_model(extracted: str, on_disk: str, block_name: str) -> str:
        """Pick the better of an extracted snippet vs an on-disk artifact.

        Prefer whichever parses AND defines the contract; if both do, prefer
        the on-disk file (a tool-using CLI tends to write richer output than
        it echoes). Fall back to the longer non-empty candidate.
        """
        candidates = [c for c in (on_disk, extracted) if c and c.strip()]
        valid = [
            c for c in candidates if _validate_block_model_text(c, block_name) is None
        ]
        if valid:
            return valid[0]
        return max(candidates, key=len) if candidates else ""


def _block_model_class(tree: ast.AST, block_name: str) -> ast.ClassDef | None:
    """Find the contract ``class <block>(Elaboratable)``."""
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != block_name:
            continue
        for base in node.bases:
            name = getattr(base, "id", None) or getattr(base, "attr", None)
            if name == "Elaboratable":
                return node
    return None


def _class_method(cls: ast.ClassDef, name: str) -> ast.FunctionDef | None:
    return next(
        (n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name),
        None,
    )


def _validate_block_model_text(text: str, block_name: str) -> str | None:
    """Static-validate block model SOURCE TEXT (no import). None == OK."""
    if not text or not text.strip():
        return "empty source"
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return f"syntax error: {exc}"

    cls = _block_model_class(tree, block_name)
    if cls is None:
        return (
            f"missing class {block_name}(Elaboratable) "
            "(Amaranth block model contract)"
        )
    init = _class_method(cls, "__init__")
    elaborate = _class_method(cls, "elaborate")
    if init is None:
        return f"class {block_name} must define __init__(self, clk, rst, ports...)"
    if elaborate is None:
        return f"class {block_name} must define elaborate(self, platform)"
    params = [a.arg for a in init.args.args if a.arg != "self"]
    # C11: accept the design's ACTUAL clock name, not a literal 'clk'. Caravel
    # contracts mandate wb_clk_i; the literal check failed EVERY model
    # generation on such designs -- and the failure path discarded an
    # otherwise-valid model before gap detection / resolution could run.
    if not any("clk" in p.lower() for p in params):
        return (f"{block_name} constructor must include a clock port "
                f"(a parameter containing 'clk', e.g. clk / wb_clk_i)")
    forbidden = {"Simulator", "Tick", "Settle", "add_process", "add_testbench"}
    used = {
        (getattr(n, "id", None) or getattr(n, "attr", None))
        for n in ast.walk(cls)
        if isinstance(n, (ast.Name, ast.Attribute))
    }
    bad = sorted(forbidden & used)
    if bad:
        return ("simulation/testbench constructs do not belong in block hardware: "
                + ", ".join(bad))
    return None


def _validate_block_model_file(path: str, block_name: str) -> str | None:
    """Full validation: static text checks + a guarded import. None == OK."""
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        return f"could not read file: {exc}"

    static = _validate_block_model_text(text, block_name)
    if static is not None:
        return static

    # Guarded import from the file path under a private module name so the
    # import side-effects (if any) are sandboxed and never collide.
    mod_name = f"_coresmith_block_model_validate_{p.stem}"
    spec = importlib.util.spec_from_file_location(mod_name, str(p))
    if spec is None or spec.loader is None:
        return "could not build import spec"
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001 - any import error invalidates it
        return f"import failed: {type(exc).__name__}: {exc}"

    factory = getattr(module, block_name, None)
    if factory is None or not callable(factory):
        return f"Elaboratable class '{block_name}' not importable / not callable"
    try:
        from amaranth import Elaboratable
        if not inspect.isclass(factory) or not issubclass(factory, Elaboratable):
            return f"'{block_name}' is not an Amaranth Elaboratable class"
        sig = inspect.signature(factory)
        if not any("clk" in p.lower() for p in sig.parameters):
            return (f"Elaboratable '{block_name}' constructor missing a clock "
                    f"port (a parameter containing 'clk')")
    except (TypeError, ValueError):
        # The AST check already confirmed the constructor signature.
        pass
    return None
