# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
ModelIntegrationGenerator -- LLM agent that wires per-block Amaranth models
into a top-level Amaranth chip model with a deterministic ``simulate()`` driver.

This is the v2 "model integration" agent. It runs AFTER every per-block uarch /
block-model node has finished. Given the block diagram, the frozen per-block
interface contracts, the reference implementation, and the directory of Amaranth
block models, it writes ``arch/block_models/_chip_model.py`` defining:

- a top-level ``class chip_model(Elaboratable)`` that
  instantiates every block model and wires the Signals per the block diagram
  (chip ingress / egress, internal edges, feedback), and
- a module-level ``def simulate(stimulus) -> observed`` that drives the stimulus
  through ``chip_model`` in an Amaranth Simulator and returns the chip-level output
  in the SAME shape the reference implementation returns.

A generic harness cannot do this (chip I/O, handshakes, latency, feedback), so
this is an LLM agent. Wiring mirrors UarchSpecGenerator / IntegrationLeadAgent:
ClaudeLLM(model, timeout); system prompt from
prompts/model_integration_generator.md; ``await self.llm.call(...)``;
``_extract_python``. The emitted file is validated (import + callable simulate).
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from typing import Any

from opentelemetry import trace

from orchestrator._timeouts import scaled
from orchestrator.langchain.prompts.skills import load_skills as _load_skills

from .coresmith_llm import ClaudeLLM

_tracer = trace.get_tracer(__name__)


def _chip_model_attempts() -> int:
    """Chip-model generation rounds (default 2).

    A rejected model leaves the composition gate with nothing to compose, so it
    NO-OPS for the whole run -- an expensive outcome for a fault the validator
    already localised to one line. One retry carrying that exact complaint back
    is cheap next to losing the gate. ``CORESMITH_CHIP_MODEL_ATTEMPTS=1``
    restores single-shot.
    """
    import os as _os

    try:
        return max(1, int(
            _os.environ.get("CORESMITH_CHIP_MODEL_ATTEMPTS", "2") or "2"))
    except ValueError:
        return 2


_PROMPT_FILE = (
    Path(__file__).resolve().parent.parent
    / "prompts"
    / "model_integration_generator.md"
)
if _PROMPT_FILE.exists():
    SYSTEM_PROMPT = _PROMPT_FILE.read_text()
else:  # pragma: no cover - prompt ships with the repo
    SYSTEM_PROMPT = (
        "Wire the per-block Amaranth block models into a top-level Elaboratable "
        "chip_model and a module-level simulate(stimulus) driver that returns "
        "a (output, cycles) 2-tuple: output in the same shape the reference "
        "implementation returns, plus the integer clock-cycle count."
    )

# Inject the shared streaming-protocol skills so the chip wiring honours the same
# handshake + framing (tvalid/tready, tlast/tuser propagation) the block models do.
_SKILLS_TEXT = _load_skills("axi_stream", "srdy_drdy")
if _SKILLS_TEXT:
    SYSTEM_PROMPT = (
        SYSTEM_PROMPT
        + "\n\n# Reference Skills (streaming protocol — follow these)\n\n"
        + _SKILLS_TEXT
    )


class ModelIntegrationGenerator:
    """Agent that produces the integrated Amaranth chip model (_chip_model.py)."""

    def __init__(self, model: str | None = None, temperature: float = 0.1):
        from orchestrator.langchain.agents.coresmith_llm import block_model

        model = model or block_model()
        self.llm = ClaudeLLM(
            model=model,
            timeout=scaled(2700, env="CORESMITH_MODEL_INTEGRATION_TIMEOUT"),
        )

    async def generate(
        self,
        project_root: str,
        block_models_dir: str,
        block_diagram: Any,
        interface_contracts: Any,
        reference_impl_source: str,
        reference_entry_name: str,
        output_path: str,
        reference_output_shape: str = "",
    ) -> dict[str, Any]:
        """Generate ``arch/block_models/_chip_model.py``.

        Args:
            project_root: project root path.
            block_models_dir: directory holding the per-block Amaranth models.
            block_diagram: ``{"blocks": [...], "connections": [...]}``.
            interface_contracts: the frozen per-block interface contracts
                (any JSON-serialisable structure or a prompt-ready string).
            reference_impl_source: source of the reference implementation.
            reference_entry_name: name of the reference entry point (the oracle)
                whose return shape ``simulate`` must match.
            output_path: absolute path to write ``_chip_model.py`` to.
            reference_output_shape: a human-readable description of the EXACT
                container the reference returns (e.g. ``"a dict with keys
                [bitstream, recon, stats], each a list[int]"`` or
                ``"a flat list[int]"``). The composed ``simulate()`` output
                MUST match this container/keys structurally (the gate compares
                structurally, so a missing dict key fails even when the bytes
                are right). Derived by the caller from the real reference return.

        Returns:
            ``{"path": <written path>}``.

        Raises:
            RuntimeError: if the agent did not emit a valid chip model (missing
                file, syntax error, failed import, or no callable ``simulate``).
        """
        import json as _json

        span_name = "Model Integration [chip_model]"
        with _tracer.start_as_current_span(span_name) as span:
            try:
                bd_text = (
                    block_diagram
                    if isinstance(block_diagram, str)
                    else _json.dumps(block_diagram, indent=2)
                )
            except (TypeError, ValueError):
                bd_text = str(block_diagram)
            try:
                contracts_text = (
                    interface_contracts
                    if isinstance(interface_contracts, str)
                    else _json.dumps(interface_contracts, indent=2)
                )
            except (TypeError, ValueError):
                contracts_text = str(interface_contracts)

            # List the available block-model modules so the agent imports the
            # right module names.
            model_files = sorted(
                p.stem
                for p in Path(block_models_dir).glob("*.py")
                if p.name != "_chip_model.py" and not p.name.startswith("__")
            )

            user_message = "\n".join(
                [
                    "Wire the per-block Amaranth block models into a top-level "
                    "Amaranth chip model + simulate() driver.",
                    f"\nBlock-model directory: {block_models_dir}",
                    "Available block-model modules (import each by this name): "
                    + ", ".join(model_files),
                    f"\nReference entry point NAME (the EXTERNAL oracle the gate "
                    f"runs separately; do NOT import/call/reimplement it): "
                    f"{reference_entry_name}.",
                    (
                        # The reference OUTPUT CONTAINER is the authoritative
                        # shape simulate() must return.  The gate compares
                        # STRUCTURALLY, so the container type AND every field must
                        # match -- a flat list when the reference returns a dict
                        # (or a missing dict key) is a gap_class=contract FAILURE
                        # even when the bytes you DID return are byte-exact.
                        f"REFERENCE OUTPUT SHAPE (your simulate() output, the "
                        f"FIRST element of the (output, cycles) tuple, MUST be "
                        f"this SAME container with ALL these fields -- see system "
                        f"rule 6c): {reference_output_shape}. If this shape is a "
                        f"dict with multiple keys, return a dict with ALL those "
                        f"keys (NOT a flat list of just the streaming output); "
                        f"reconstruct every non-streaming field (e.g. an on-die "
                        f"SRAM/memory image such as recon or coeff_mem) by snooping "
                        f"the DUT real state via its debug read port or a write-bus "
                        f"snoop (NEVER by calling/embedding the oracle), and assert "
                        f"set(output.keys())==reference keys before returning. "
                        f"The gate compares it to {reference_entry_name}(stimulus)."
                        if reference_output_shape
                        else
                        f"Return your captured chip egress in the SAME container "
                        f"the reference {reference_entry_name}(stimulus) returns "
                        f"(a flat list / bytes for a single-stream design; a dict "
                        f"with ALL its keys for a multi-output design -- see "
                        f"system rule 6c), so the gate can compare them "
                        f"structurally."
                    ),
                    "\n--- BLOCK DIAGRAM (blocks + connections) ---",
                    bd_text,
                    "\n--- FROZEN INTERFACE CONTRACTS (handshake + bus widths) ---",
                    contracts_text,
                    "\nNOTE: the reference implementation source is deliberately "
                    "NOT provided. Your chip output must be produced by the wired "
                    "block models alone and captured from the chip egress ports -- "
                    "never embed, import, call, or reimplement the reference, and "
                    "never drive an egress/output port (see hard rules 6 & 7).",
                    "\nEmit ONLY the _chip_model.py Amaranth ```python code block "
                    "per the CONTRACT: import each block model, define class "
                    "chip_model(Elaboratable), and a module-level "
                    "simulate(stimulus) that runs an Amaranth Simulator and returns "
                    "a (output, cycles) 2-tuple where output is the captured chip "
                    "egress IN THE REFERENCE OUTPUT SHAPE above (a dict with ALL "
                    "keys if the reference returns a dict -- rule 6c) and cycles "
                    "is the integer clock-cycle count.",
                ]
            )

            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)

            on_disk = ""
            if out.exists():
                try:
                    on_disk = out.read_text(encoding="utf-8")
                except OSError:
                    on_disk = ""

            # RETRY ONCE with the rejection quoted back. A rejected chip model
            # leaves the composition gate with nothing to compose, so the gate
            # NO-OPS -- the most expensive possible outcome for a fault the
            # validator has already localised to one line. The rules were in
            # the prompt the first time; what was missing was the feedback.
            problem = None
            _attempts = _chip_model_attempts()
            for _round in range(_attempts):
                prompt = user_message if problem is None else "\n".join([
                    user_message,
                    "\n--- YOUR PREVIOUS ATTEMPT WAS REJECTED ---",
                    problem,
                    "Fix EXACTLY that and re-emit the COMPLETE _chip_model.py. "
                    "Do not change anything else, and do not respond with a "
                    "diff or a partial file.",
                ])
                content = await self.llm.call(
                    system=SYSTEM_PROMPT,
                    prompt=prompt,
                    run_name=("Generate Model Integration [chip_model]"
                              + (f" retry {_round}" if _round else "")),
                )
                code = self._extract_python(content)
                chosen = self._choose_chip_model(code, on_disk, block_models_dir)
                if not chosen:
                    raise RuntimeError(
                        "Model integration generation produced no usable Python "
                        f"(no fenced code block and no on-disk file at "
                        f"{output_path})."
                    )

                # Gate-observation hardening (the internal-memory snoop the
                # reference output keys require, and the frame-done/tlast count)
                # is implemented inside the generated Amaranth model via real
                # Signals / debug ports -- the engine no longer post-processes
                # the generated source to inject an observation shim.

                out.write_text(chosen, encoding="utf-8")

                problem = _validate_chip_model_file(
                    str(out), block_models_dir, reference_entry_name
                )
                if problem is None:
                    break
                span.set_attribute(f"rejected_round_{_round}", problem[:300])

            if problem is not None:
                raise RuntimeError(
                    f"Integrated chip model at {output_path} is invalid after "
                    f"{_attempts} attempt(s): {problem}"
                )

            span.set_attribute("path", str(out))
            span.set_attribute("attempts", _round + 1)
            return {"path": str(out)}

    @staticmethod
    def _extract_python(content: str) -> str:
        """Extract the first ```python fenced block from an LLM response."""
        match = re.search(r"```python\s*\n(.*?)```", content, re.DOTALL)
        if match:
            return match.group(1).strip()
        return content.strip()

    @staticmethod
    def _choose_chip_model(
        extracted: str, on_disk: str, block_models_dir: str
    ) -> str:
        """Pick the better of an extracted snippet vs an on-disk artifact."""
        candidates = [c for c in (on_disk, extracted) if c and c.strip()]
        valid = [
            c
            for c in candidates
            if _validate_chip_model_text(c) is None
        ]
        if valid:
            return valid[0]
        return max(candidates, key=len) if candidates else ""


def _validate_chip_model_text(
    text: str, reference_entry_name: str | None = None
) -> str | None:
    """Static-validate chip-model SOURCE TEXT (no import). None == OK.

    Includes the ANTI-CHEAT check: the chip model must not embed, import, or call
    the reference oracle. The gate runs the reference EXTERNALLY; a chip model
    that reproduces it (e.g. inlining the golden and replaying its bytes onto the
    egress port) makes the gate compare reference-to-reference and pass
    vacuously. See hard rules 6 & 7 in the prompt.
    """
    import ast

    if not text or not text.strip():
        return "empty source"
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return f"syntax error: {exc}"
    has_simulate = any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "simulate"
        for node in ast.walk(tree)
    )
    if not has_simulate:
        return "missing module-level def simulate(stimulus)"

    chip_classes = [
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "chip_model"
    ]
    if not chip_classes:
        return "missing class chip_model(Elaboratable)"
    if not any(
        (getattr(base, "id", None) or getattr(base, "attr", None))
        == "Elaboratable"
        for base in chip_classes[0].bases
    ):
        return "chip_model must inherit amaranth.Elaboratable"

    ren = (reference_entry_name or "").strip().lower()

    # Which names in this module are bound to a LOCAL object? A chip model
    # writes ``sim = Simulator(m)`` and then ``sim.run()`` -- and when the
    # reference entry happens to be called ``run``, the blanket attribute-name
    # match below read that as "the chip model calls the oracle" and rejected
    # the model. The composition gate then no-op'd, on two consecutive runs,
    # over the Amaranth simulator API. A method call on an object this module
    # constructed cannot be the oracle: reaching the oracle needs an import,
    # and importing it is already rejected above.
    #
    # The relaxation is scoped hard. Only an attribute call whose receiver is a
    # plain local name is spared, and only when the receiver is not itself
    # oracle-shaped and the module does no dynamic importing -- a module that
    # touches importlib/__import__/exec/eval could bind the oracle to a local,
    # so it keeps the strict rule. A bare ``run(...)``, an unbound receiver
    # (``mod.run()``), and a chained one (``a.b.run()``) all still fail.
    _local_names = _locally_bound_names(tree)
    _dynamic = bool(re.search(r"\b(?:importlib|__import__|exec|eval)\b", text))

    for node in ast.walk(tree):
        # No import of a *_golden module (the reference implementation).
        if isinstance(node, ast.Import):
            for alias in node.names:
                if "golden" in alias.name.lower():
                    return (
                        f"ANTI-CHEAT: chip model imports the reference oracle "
                        f"({alias.name!r}). The output must be produced by the "
                        f"block models only -- never embed the reference."
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.module and "golden" in node.module.lower():
                return (
                    f"ANTI-CHEAT: chip model imports from the reference oracle "
                    f"({node.module!r}). Never embed the reference."
                )
        # No function that defines/reimplements the reference entry, and no call
        # to it (catches an inlined ``_encode_image_v2`` and any call to it).
        if ren:
            # Match underscore-decorated reimplementations too: an agent that
            # inlines the oracle typically names it `_encode_image_v2` -- the
            # strip('_') normalisation catches that without substring-matching
            # (which would false-positive short entries like `run` in run_sim).
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name.lower().strip("_") == ren
            ):
                return (
                    f"ANTI-CHEAT: chip model defines {node.name!r}, which "
                    f"reimplements the reference entry {reference_entry_name!r}. "
                    f"The chip output must come from the wired block models, not "
                    f"a reproduced oracle."
                )
            if isinstance(node, ast.Call):
                fn = node.func
                called = getattr(fn, "id", None) or getattr(fn, "attr", None)
                if not called or called.lower().strip("_") != ren:
                    continue
                recv = (fn.value.id
                        if isinstance(fn, ast.Attribute)
                        and isinstance(fn.value, ast.Name) else None)
                if (recv and not _dynamic and recv in _local_names
                        and recv.lower().strip("_") not in _ORACLE_RECEIVERS):
                    continue    # method on a locally-built object, not the oracle
                return (
                    f"ANTI-CHEAT: chip model calls {called!r}, matching the "
                    f"reference entry {reference_entry_name!r}. Never call "
                    f"the oracle from the chip model."
                )
    return None


#: Receiver names that read as the reference even when locally bound, so
#: ``golden.run()`` is never spared by the local-object exception above.
_ORACLE_RECEIVERS = frozenset({
    "golden", "gold", "ref", "reference", "oracle", "impl", "model_ref",
    "reference_impl", "reference_module", "ref_mod", "golden_mod",
})


def _locally_bound_names(tree) -> set:
    """Names this module BINDS to a value it constructed.

    Assignment targets, walrus targets, ``with ... as``, ``for`` targets,
    comprehension targets, ``except ... as`` and function parameters. Import
    bindings are deliberately EXCLUDED -- an imported handle is exactly the way
    the oracle would arrive, and importing it is rejected separately.
    """
    import ast

    out: set = set()

    def _bind(target) -> None:
        if isinstance(target, ast.Name):
            out.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for e in target.elts:
                _bind(e)
        elif isinstance(target, ast.Starred):
            _bind(target.value)

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                _bind(t)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            _bind(node.target)
        elif isinstance(node, ast.NamedExpr):
            _bind(node.target)
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
            _bind(node.target)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars is not None:
                    _bind(item.optional_vars)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            out.add(node.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                               ast.Lambda)):
            a = node.args
            for arg in (list(a.posonlyargs) + list(a.args) + list(a.kwonlyargs)
                        + [a.vararg, a.kwarg]):
                if arg is not None:
                    out.add(arg.arg)
    return out


def _validate_chip_model_file(
    path: str, block_models_dir: str, reference_entry_name: str | None = None
) -> str | None:
    """Full validation: static checks + a guarded import. None == OK.

    The chip model imports its sibling block-model modules by plain name, so we
    must import it with ``block_models_dir`` on ``sys.path``.
    """
    import sys

    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        return f"could not read file: {exc}"

    static = _validate_chip_model_text(text, reference_entry_name)
    if static is not None:
        return static

    mod_name = "_coresmith_chip_model_validate"
    spec = importlib.util.spec_from_file_location(mod_name, str(p))
    if spec is None or spec.loader is None:
        return "could not build import spec"
    module = importlib.util.module_from_spec(spec)

    inserted = False
    if block_models_dir not in sys.path:
        sys.path.insert(0, block_models_dir)
        inserted = True
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001
        return f"import failed: {type(exc).__name__}: {exc}"
    finally:
        if inserted:
            try:
                sys.path.remove(block_models_dir)
            except ValueError:
                pass

    simulate = getattr(module, "simulate", None)
    if simulate is None or not callable(simulate):
        return "module-level simulate is missing / not callable"
    # simulate() must accept a single positional ``stimulus`` argument. We do
    # NOT execute it here (that needs real stimulus + block-model wiring); the
    # deterministic gate runs it. Its return contract is a ``(output, cycles)``
    # 2-tuple, but the gate tolerates a legacy 1-value (bare output) return too,
    # so static validation accepts either -- it only checks the signature.
    try:
        import inspect as _inspect
        sig = _inspect.signature(simulate)
        positional = [
            p
            for p in sig.parameters.values()
            if p.kind
            in (
                _inspect.Parameter.POSITIONAL_ONLY,
                _inspect.Parameter.POSITIONAL_OR_KEYWORD,
                _inspect.Parameter.VAR_POSITIONAL,
            )
        ]
        if not positional:
            return "simulate must accept a stimulus argument"
    except (TypeError, ValueError):
        pass
    return None
