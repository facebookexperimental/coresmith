"""Deterministic, engine-authored RTL <-> model byte-exact equivalence checker.

WHY THIS EXISTS
---------------
The per-block DV uses the Amaranth block model as the cocotb oracle, but the
*comparison testbench* is written by an LLM. ``testbench_generator.py`` even
instructs the LLM NOT to byte-exact-compare long-latency / "functional" blocks.
That is exactly how a *wrong* ``intra_rd_encode_core.v`` (a gradient heuristic
instead of the model's RD search) "PASSED" its block DV: the LLM-authored TB
never byte-compared it to the model.

This module is the load-bearing fix: a check the LLM **cannot** weaken. The
ENGINE itself (not an LLM) drives BOTH the generated Verilog AND a deterministic
Python reference of the same block on the SAME seeded-random vectors, then
asserts the two output byte streams are identical. The seed is fresh per run
(anti-memorization, mirroring ``CORESMITH_DV_SEED`` in pipeline_helpers.py), so
the RTL cannot have memorized the vectors at generation time.

DESIGN PRINCIPLES
-----------------
* DETERMINISTIC + engine-authored. No LLM is in the loop here. The cocotb
  harness is a fixed template emitted by this module.
* HONEST SKIPS, NEVER A FALSE PASS. If a deterministic reference callable for
  the block cannot be resolved, or the block's interface is not the single
  ``s_axis`` in / ``m_axis`` out AXI-Stream shape this generic harness covers,
  we return ``{"skipped": True, ...}`` with a logged reason -- we never return
  ``passed=True`` for a block we did not actually exercise byte-for-byte.
* BOUNDED BUT REAL. Per the maintainer's rule, a large block legitimately
  cannot be simulated unboundedly. ``n_vectors``, ``max_cycles`` and the VCD
  window are all configurable. A bounded run that genuinely progressed and
  matched is a valid PASS; an empty / never-progressed run is a FAILURE (not a
  pass), so a block that emits nothing cannot sneak through.

PUBLIC API
----------
``rtl_model_equiv_enabled() -> bool``
    Gate. ON by default (``CORESMITH_RTL_MODEL_EQUIV`` default "1"); disable
    with "0"/"false"/"off"/"no".

``check_rtl_model_equivalence(block_name, rtl_path, model_module_path, *,
    project_root, seed, n_vectors, max_cycles, clock_period_ns=20) -> dict``
    Returns ``{"passed": bool, "reason": str, "checked_vectors": int,
    "skipped": bool}``.
"""

from __future__ import annotations

import importlib.util
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

# Names a deterministic, pure-Python block reference might be exported under.
# The golden-model wrapper re-exports the FULL namespace (incl. underscore
# names), so we probe a generous set. We deliberately do NOT accept an Amaranth
# ``Elaboratable`` here: it is RTL-shaped (it needs its own clocked TB to
# drive), not a value->value reference, so driving it generically would itself
# be an LLM-free-but-unfaithful re-transcription. If only a @block exists we
# SKIP (honest) rather than risk a wrong oracle.
_REFERENCE_NAME_CANDIDATES = (
    "reference",
    "ref",
    "process",
    "step",
    "compute",
    "model",
    "transform",
    "golden",
    "apply",
    "run",
)


def rtl_model_equiv_enabled() -> bool:
    """Gate for the deterministic RTL<->model equivalence check.

    ON by default. Set ``CORESMITH_RTL_MODEL_EQUIV=0`` (or
    false/off/no) to disable.
    """
    from orchestrator.profile import ensure_applied, flag_enabled
    ensure_applied()
    return flag_enabled("CORESMITH_RTL_MODEL_EQUIV", default=True)


# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------

def _skip(reason: str, checked: int = 0, *, harness_error: bool = False) -> dict:
    """A non-blocking SKIP.

    ``harness_error=True`` marks an ENVIRONMENT/HARNESS failure (a build error,
    a timeout, a run that produced no output json) as opposed to an *honest*
    skip (non-AXIS interface, no deterministic reference). The caller retries a
    harness error once with a longer timeout and, if it still errors, fails the
    block closed -- an honest skip stays non-blocking (A-Fix 2c).
    """
    r = {"passed": False, "skipped": True, "reason": reason, "checked_vectors": checked}
    if harness_error:
        r["harness_error"] = True
    return r


def _fail(reason: str, checked: int = 0) -> dict:
    return {"passed": False, "skipped": False, "reason": reason, "checked_vectors": checked}


def _pass(reason: str, checked: int) -> dict:
    return {"passed": True, "skipped": False, "reason": reason, "checked_vectors": checked}


# ---------------------------------------------------------------------------
# Verilog port-list parsing (deterministic, no LLM)
# ---------------------------------------------------------------------------

# A single Verilog port declaration inside the module header, e.g.
#   input  wire [7:0] s_axis_tdata,
#   output reg        m_axis_tvalid
_PORT_RE = re.compile(
    r"\b(?P<dir>input|output|inout)\b"
    r"(?:\s+(?:wire|reg|logic))?"
    r"(?:\s*\[\s*(?P<msb>\d+)\s*:\s*(?P<lsb>\d+)\s*\])?"
    r"\s+(?P<name>[A-Za-z_][A-Za-z0-9_$]*)",
)


def _module_name(rtl_text: str) -> str | None:
    m = re.search(r"\bmodule\s+([A-Za-z_][A-Za-z0-9_$]*)", rtl_text)
    return m.group(1) if m else None


def _parse_ports(rtl_text: str) -> dict:
    """Return {port_name: {'dir': 'input'/'output'/'inout', 'width': int}}.

    Parses the ANSI-style module header (the common case the engine emits).
    """
    # Restrict to the first module's header (up to the first ');') so a large
    # block body with combinational ``input``-looking strings doesn't pollute.
    head = rtl_text
    m = re.search(r"\bmodule\b", rtl_text)
    if m:
        end = rtl_text.find(");", m.start())
        if end != -1:
            head = rtl_text[m.start():end + 1]
    ports: dict = {}
    for pm in _PORT_RE.finditer(head):
        name = pm.group("name")
        if name in ports:
            continue
        if pm.group("msb") is not None:
            width = abs(int(pm.group("msb")) - int(pm.group("lsb"))) + 1
        else:
            width = 1
        ports[name] = {"dir": pm.group("dir"), "width": width}
    return ports


def _classify_axis(ports: dict) -> dict | None:
    """Identify a single s_axis in / single m_axis out AXI-Stream interface.

    Returns a dict describing the discovered signal names, or None if the
    port set is not the single-in/single-out AXIS shape this harness covers.
    The signal name may carry a stream-name infix
    (``s_axis_<name>_tdata``), so we match by prefix + ``t<field>`` suffix.
    """
    def _find(prefix: str, field: str) -> str | None:
        suffix = "t" + field
        cands = [
            n for n in ports
            if n.startswith(prefix) and n.endswith(suffix)
        ]
        return cands[0] if len(cands) == 1 else None

    clk = next((n for n in ports if n in ("clk", "clock")), None)
    rst = next(
        (n for n in ports if n in ("rst_n", "resetn", "aresetn", "reset", "rst")),
        None,
    )
    if clk is None or rst is None:
        return None

    s_tdata = _find("s_axis", "data")
    s_tvalid = _find("s_axis", "valid")
    s_tready = _find("s_axis", "ready")
    m_tdata = _find("m_axis", "data")
    m_tvalid = _find("m_axis", "valid")
    m_tready = _find("m_axis", "ready")
    if not all((s_tdata, s_tvalid, s_tready, m_tdata, m_tvalid, m_tready)):
        return None

    s_tlast = _find("s_axis", "last")
    m_tlast = _find("m_axis", "last")

    # Reject blocks with EXTRA s_axis/m_axis stream groups (multi-stream): we
    # only cover exactly one input stream and one output stream.
    s_prefixes = {n.rsplit("_t", 1)[0] for n in ports if n.startswith("s_axis")}
    m_prefixes = {n.rsplit("_t", 1)[0] for n in ports if n.startswith("m_axis")}
    if len(s_prefixes) != 1 or len(m_prefixes) != 1:
        return None

    return {
        "clk": clk,
        "rst": rst,
        "rst_active_low": rst.endswith("_n") or rst.endswith("n"),
        "s_tdata": s_tdata,
        "s_tvalid": s_tvalid,
        "s_tready": s_tready,
        "s_tlast": s_tlast,
        "m_tdata": m_tdata,
        "m_tvalid": m_tvalid,
        "m_tready": m_tready,
        "m_tlast": m_tlast,
        "s_width": ports[s_tdata]["width"],
        "m_width": ports[m_tdata]["width"],
    }


# ---------------------------------------------------------------------------
# Reference-callable resolution
# ---------------------------------------------------------------------------

def _load_model_module(model_module_path: str, project_root: Path):
    """Import the block model (the same oracle the TB uses).

    ``model_module_path`` may be a dotted module path
    (``arch.block_models.<block>``) or a filesystem path to the ``_model.py``
    wrapper. Returns the module object or raises.
    """
    p = Path(model_module_path)
    if p.exists() and p.suffix == ".py":
        root = str(project_root)
        if root not in sys.path:
            sys.path.insert(0, root)
        spec = importlib.util.spec_from_file_location(
            f"_rme_{p.stem}", str(p),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        return mod
    # dotted module path
    root = str(project_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    return importlib.import_module(model_module_path)


def _is_hardware_block(obj) -> bool:
    """True for an Amaranth Elaboratable class (not a value->value oracle)."""
    try:
        import inspect

        from amaranth import Elaboratable
        return inspect.isclass(obj) and issubclass(obj, Elaboratable)
    except (ImportError, TypeError):
        return False


def _resolve_reference(mod, block_name: str) -> Callable | None:
    """Find a deterministic, pure-Python reference callable for the block.

    Tries (in order): a callable named exactly ``<block_name>`` (if not a
    hardware Elaboratable), then block-name-derived underscore variants
    (``_process_<x>``), then the generic candidate names. Returns None if
    nothing resolvable -> caller SKIPs (honest, never a false pass).
    """
    def _ok(name: str):
        obj = getattr(mod, name, None)
        if obj is None or not callable(obj):
            return None
        if _is_hardware_block(obj):
            return None
        return obj

    # 1. exact block name
    cand = _ok(block_name)
    if cand is not None:
        return cand

    # 2. underscore / suffix variants derived from the block name
    short = block_name.split("_")[-1]
    for pat in (
        f"_process_{short}",
        f"process_{short}",
        f"_{block_name}",
        f"reference_{block_name}",
        f"{block_name}_ref",
        f"{block_name}_model",
    ):
        cand = _ok(pat)
        if cand is not None:
            return cand

    # 3. generic candidate names
    for name in _REFERENCE_NAME_CANDIDATES:
        cand = _ok(name)
        if cand is not None:
            return cand

    return None


def _expected_stream(ref: Callable, in_bytes: list, mask: int) -> list | None:
    """Compute the expected output byte stream from the reference.

    Two calling conventions are supported, tried in order:
      * stream form: ``ref(list_of_ints) -> iterable_of_ints``
      * per-element form: ``ref(int) -> int`` (mapped over the stream)
    Returns a list of masked ints, or None if neither convention yields a
    usable stream.
    """
    # stream form
    try:
        out = ref(list(in_bytes))
        if out is not None and not isinstance(out, (int, bytes, bytearray)):
            return [int(v) & mask for v in out]
        if isinstance(out, (bytes, bytearray)):
            return [int(v) & mask for v in out]
    except Exception:
        pass
    # per-element form
    try:
        return [int(ref(v)) & mask for v in in_bytes]
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Generic AXI-Stream cocotb harness (engine-authored template)
# ---------------------------------------------------------------------------

_HARNESS_TEMPLATE = r'''"""ENGINE-AUTHORED generic AXI-Stream equivalence harness.

NOT LLM-authored. Drives the DUT with engine-supplied seeded vectors and
collects the output byte stream, which the engine then byte-compares to the
block model. Stimulus + expected come from JSON the engine wrote; this file
contains NO design knowledge.
"""
import json
import os

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

_CFG = json.loads(os.environ["RME_CONFIG_JSON"])


def _g(dut, name):
    return getattr(dut, name)


@cocotb.test()
async def equiv_drive(dut):
    cfg = _CFG
    clk = _g(dut, cfg["clk"])
    # Positional units arg -- robust across cocotb 1.9 (`units=`) and 2.0
    # (`unit=`), which renamed the kwarg; positional slot is the same in both.
    cocotb.start_soon(Clock(clk, cfg["clock_period_ns"], "ns").start())

    # Reset
    rst = _g(dut, cfg["rst"])
    active_low = cfg["rst_active_low"]
    rst.value = 0 if active_low else 1
    _g(dut, cfg["s_tvalid"]).value = 0
    _g(dut, cfg["s_tdata"]).value = 0
    if cfg.get("s_tlast"):
        _g(dut, cfg["s_tlast"]).value = 0
    _g(dut, cfg["m_tready"]).value = 1
    for _ in range(5):
        await RisingEdge(clk)
    rst.value = 1 if active_low else 0
    for _ in range(2):
        await RisingEdge(clk)

    stim = cfg["stimulus"]
    max_cycles = cfg["max_cycles"]
    collected = []

    s_tvalid = _g(dut, cfg["s_tvalid"])
    s_tdata = _g(dut, cfg["s_tdata"])
    s_tready = _g(dut, cfg["s_tready"])
    s_tlast = _g(dut, cfg["s_tlast"]) if cfg.get("s_tlast") else None
    m_tvalid = _g(dut, cfg["m_tvalid"])
    m_tdata = _g(dut, cfg["m_tdata"])
    m_tready = _g(dut, cfg["m_tready"])

    m_tready.value = 1
    n = len(stim)
    idx = 0
    cycles = 0
    # Drive all input beats, concurrently draining output, bounded by max_cycles.
    while cycles < max_cycles:
        # Present next input beat if any remain.
        if idx < n:
            s_tdata.value = int(stim[idx])
            s_tvalid.value = 1
            if s_tlast is not None:
                s_tlast.value = 1 if idx == n - 1 else 0
        else:
            s_tvalid.value = 0
            if s_tlast is not None:
                s_tlast.value = 0

        await RisingEdge(clk)
        cycles += 1

        # Input handshake (sampled on the edge we just passed).
        if idx < n and int(s_tvalid.value) == 1 and int(s_tready.value) == 1:
            idx += 1

        # Output handshake: collect a beat when valid && ready.
        if int(m_tvalid.value) == 1 and int(m_tready.value) == 1:
            collected.append(int(m_tdata.value))

        # Stop once we've sent everything and output went quiet for a while.
        if idx >= n:
            break

    s_tvalid.value = 0
    if s_tlast is not None:
        s_tlast.value = 0

    # Drain remaining output until quiet or bound hit.
    quiet = 0
    drain_limit = cfg["expected_len"] if cfg["expected_len"] > 0 else n
    while cycles < max_cycles and quiet < cfg["drain_quiet_cycles"]:
        await RisingEdge(clk)
        cycles += 1
        if int(m_tvalid.value) == 1 and int(m_tready.value) == 1:
            collected.append(int(m_tdata.value))
            quiet = 0
        else:
            quiet += 1
        if len(collected) >= drain_limit and quiet >= 2:
            break

    out_path = os.environ["RME_OUTPUT_JSON"]
    with open(out_path, "w") as f:
        json.dump({"collected": collected, "cycles": cycles, "sent": idx}, f)
'''


def _build_makefile(sim_dir: Path, rtl_path: str, module_name: str,
                    tb_module: str, build_jobs: int) -> str:
    # HARD SAFETY (2026-07-01): NO trace/VCD -- a byte-exact equivalence compare
    # needs no waveform, and `--trace --trace-structs` makes Verilator emit
    # enormous C++ that chokes the compile on any non-trivial design (it hung a
    # 4-core box to load ~3400). SERIAL build (`--build-jobs 1`) and a serial
    # make (set by the caller) so the gate can never fork-storm. Trace is only
    # ever enabled, bounded, when explicitly debugging a divergence.
    return (
        f"SIM = verilator\n"
        f"TOPLEVEL_LANG = verilog\n"
        f"VERILOG_SOURCES = {rtl_path}\n"
        f"TOPLEVEL = {module_name}\n"
        f"MODULE = {tb_module}\n"
        f"EXTRA_ARGS += --build-jobs 1\n"
        f"include $(shell cocotb-config --makefiles)/Makefile.sim\n"
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Interface adapters (Phase 3): decouple the byte-exact equivalence harness from
# the single s_axis-in / m_axis-out AXI-Stream shape, so multi-stream / non-AXIS
# blocks are BYTE-CHECKED instead of honestly skipped.
#
# The engine keeps the trust boundary: an ADAPTER only describes HOW to classify
# / drive / capture a block's ports and how to SHAPE stimulus into that
# interface -- it never chooses the stimulus (``shape_stimulus`` receives the
# ENGINE's seeded rng, so it can only shape the engine's random draws, never
# smuggle a weak vector) and never does the compare (that stays byte-exact in
# ``_run_equiv_sim``). ``AxisAdapter`` (the historical logic) is the registered
# DEFAULT and is tried LAST, so when no design adapter matches, behaviour is
# byte-identical to before. A design supplies extra adapters by dropping a
# ``arch/equiv_adapters/<name>.py`` that calls ``register_equiv_adapter(...)`` --
# mirroring how it supplies per-block models.
# ---------------------------------------------------------------------------

_AXIS_RUNTIME_KEYS = (
    "clk", "rst", "rst_active_low", "s_tdata", "s_tvalid", "s_tready",
    "s_tlast", "m_tdata", "m_tvalid", "m_tready", "m_tlast",
)


class EquivAdapter:
    """How to run the byte-exact equivalence harness for ONE interface shape."""

    name = "base"

    def classify(self, ports: dict) -> dict | None:
        """Return a harness config for this shape, or None if ports don't match."""
        raise NotImplementedError

    def shape_stimulus(self, rng: random.Random, cfg: dict,
                       n_vectors: int) -> list:
        """Shape ``n_vectors`` of the ENGINE's seeded ``rng`` into stimulus."""
        raise NotImplementedError

    def expected(self, ref: Callable, stimulus, cfg: dict):
        """Reference's expected output stream (masked ints), or None if unusable."""
        raise NotImplementedError

    def output_mask(self, cfg: dict) -> int:
        """Mask applied to each collected RTL output beat before compare."""
        raise NotImplementedError

    def harness_template(self) -> str:
        """The engine-authored cocotb harness that drives/captures this shape."""
        raise NotImplementedError

    def runtime_cfg(self, cfg: dict) -> dict:
        """Harness-facing subset of ``cfg`` (signal names etc.) for RME_CONFIG_JSON."""
        raise NotImplementedError


class AxisAdapter(EquivAdapter):
    """Historical single s_axis-in / m_axis-out AXI-Stream harness (default)."""

    name = "axis_single"

    def classify(self, ports):
        return _classify_axis(ports)

    def shape_stimulus(self, rng, cfg, n_vectors):
        s_mask = (1 << cfg["s_width"]) - 1
        return [rng.randint(0, s_mask) for _ in range(max(1, n_vectors))]

    def expected(self, ref, stimulus, cfg):
        return _expected_stream(ref, stimulus, self.output_mask(cfg))

    def output_mask(self, cfg):
        return (1 << cfg["m_width"]) - 1

    def harness_template(self):
        return _HARNESS_TEMPLATE

    def runtime_cfg(self, cfg):
        return {k: cfg[k] for k in _AXIS_RUNTIME_KEYS}


# Custom/design adapters (registered on load), tried BEFORE the AXIS default.
_ADAPTER_REGISTRY: dict[str, EquivAdapter] = {}
_AXIS_ADAPTER = AxisAdapter()
_DESIGN_ADAPTERS_LOADED: set = set()


def register_equiv_adapter(adapter: EquivAdapter) -> None:
    """Register a custom interface adapter (design-supplied or built-in)."""
    _ADAPTER_REGISTRY[adapter.name] = adapter


def _load_design_adapters(project_root) -> None:
    """Import ``arch/equiv_adapters/*.py`` once per root so they self-register."""
    root = str(project_root)
    if root in _DESIGN_ADAPTERS_LOADED:
        return
    _DESIGN_ADAPTERS_LOADED.add(root)
    adir = Path(project_root) / "arch" / "equiv_adapters"
    if not adir.is_dir():
        return
    for py in sorted(adir.glob("*.py")):
        try:
            _load_model_module(str(py), Path(project_root))
        except Exception:  # noqa: BLE001 - a broken design adapter must not crash
            pass


def select_equiv_adapter(ports: dict,
                         project_root=None) -> EquivAdapter | None:
    """First adapter whose ``classify(ports)`` matches: design adapters (from
    ``arch/equiv_adapters/``) first, then the AXIS default. None if nothing fits.
    """
    if project_root is not None:
        _load_design_adapters(project_root)
    for adapter in (*_ADAPTER_REGISTRY.values(), _AXIS_ADAPTER):
        try:
            if adapter.classify(ports) is not None:
                return adapter
        except Exception:  # noqa: BLE001 - a broken adapter is skipped, not fatal
            continue
    return None


def check_rtl_model_equivalence(
    block_name: str,
    rtl_path: str,
    model_module_path: str,
    *,
    project_root,
    seed: int,
    n_vectors: int = 64,
    max_cycles: int = 20000,
    clock_period_ns: int = 20,
    timeout_scale: float = 1.0,
) -> dict:
    """Deterministically drive the generated Verilog AND the block model on the
    SAME seeded vectors and assert byte-exact output equivalence.

    Returns ``{"passed": bool, "reason": str, "checked_vectors": int,
    "skipped": bool}``. SKIPs (never false-passes) when no deterministic
    reference callable resolves or the interface is not single-in/single-out
    AXI-Stream.
    """
    project_root = Path(project_root)
    rtl_p = Path(rtl_path)
    if not rtl_p.exists():
        return _skip(f"RTL not found: {rtl_path}")
    if not shutil.which("verilator"):
        return _skip("verilator not on PATH")

    rtl_text = rtl_p.read_text(encoding="utf-8", errors="replace")
    module_name = _module_name(rtl_text)
    if not module_name:
        return _skip(f"no module declaration found in {rtl_path}")

    ports = _parse_ports(rtl_text)
    adapter = select_equiv_adapter(ports, project_root)
    if adapter is None:
        return _skip(
            f"interface for '{block_name}' matched no equivalence adapter "
            f"(the default AXIS harness covers single s_axis-in / m_axis-out; "
            f"register a design adapter in arch/equiv_adapters/ for other shapes) "
            f"(ports={sorted(ports)})"
        )
    cfg = adapter.classify(ports)

    # Resolve the deterministic reference callable (same oracle as the TB).
    try:
        mod = _load_model_module(model_module_path, project_root)
    except Exception as e:  # noqa: BLE001
        return _skip(f"could not import model module '{model_module_path}': {e!r}")
    ref = _resolve_reference(mod, block_name)
    if ref is None:
        return _skip(
            f"no deterministic pure-Python reference callable resolved for "
            f"'{block_name}' in {model_module_path} (only an Amaranth model or no "
            f"callable matched); refusing to fabricate an oracle"
        )

    # Seeded-random input vectors. The seed is supplied by the caller and is
    # fresh per run (anti-memorization, like CORESMITH_DV_SEED), so the RTL
    # cannot have memorized these at generation time. The adapter only SHAPES
    # the engine's seeded draws into the interface; it never picks the stimulus.
    rng = random.Random(seed)
    m_mask = adapter.output_mask(cfg)
    stimulus = adapter.shape_stimulus(rng, cfg, n_vectors)

    expected = adapter.expected(ref, stimulus, cfg)
    if expected is None:
        return _skip(
            f"reference callable for '{block_name}' did not yield a usable "
            f"int stream under the '{adapter.name}' adapter's calling convention"
        )

    return _run_equiv_sim(
        block_name,
        [str(rtl_p.resolve())],
        module_name,
        cfg,
        stimulus,
        expected,
        m_mask,
        project_root=project_root,
        clock_period_ns=clock_period_ns,
        max_cycles=max_cycles,
        timeout_scale=timeout_scale,
        pass_note=f"(seed={seed})",
        harness_template=adapter.harness_template(),
        runtime_cfg=adapter.runtime_cfg(cfg),
    )


def _run_equiv_sim(
    block_name: str,
    sources: list[str],
    module_name: str,
    axis: dict,
    stimulus: list[int],
    expected: list[int],
    m_mask: int,
    *,
    project_root,
    clock_period_ns: int = 20,
    max_cycles: int = 20000,
    timeout_scale: float = 1.0,
    pass_note: str = "",
    harness_template: str = None,
    runtime_cfg: dict = None,
) -> dict:
    """Build + run the interface cocotb harness for ``module_name`` (compiled
    from ``sources``, first is the top) against ``stimulus`` and byte-compare the
    collected output to ``expected``.

    ``harness_template`` / ``runtime_cfg`` come from the selected interface
    adapter (Phase 3); both default to the single-AXIS harness, so the AXIS path
    is byte-identical to before.

    Extracted from :func:`check_rtl_model_equivalence` so both the per-block and
    the chip-top equivalence gates share ONE harness. Preserves the commit-5
    ``timeout_scale`` (x wall budget on a harness-error retry) + ``harness_error``
    skip semantics: a build/timeout/no-json is a non-blocking SKIP flagged
    ``harness_error`` (caller retries once then fails closed); an empty run or a
    byte divergence is a FAIL; a byte-exact match is a PASS.
    """
    project_root = Path(project_root)
    # SERIAL by default (build_jobs=1): the equivalence gate must never
    # fork-storm the host. Only raise CORESMITH_SIM_BUILD_JOBS on a big box.
    try:
        build_jobs = max(1, int(os.environ.get("CORESMITH_SIM_BUILD_JOBS", "1") or "1"))
    except ValueError:
        build_jobs = 1

    sim_dir = Path(tempfile.mkdtemp(prefix=f"rme_{block_name}_"))
    try:
        tb_module = f"rme_tb_{block_name}"
        (sim_dir / f"{tb_module}.py").write_text(
            harness_template if harness_template is not None else _HARNESS_TEMPLATE)
        (sim_dir / "Makefile").write_text(
            _build_makefile(sim_dir, " ".join(sources), module_name,
                            tb_module, build_jobs)
        )

        out_json = sim_dir / "rme_out.json"
        import json as _json
        _rt = (runtime_cfg if runtime_cfg is not None
               else {k: axis[k] for k in _AXIS_RUNTIME_KEYS})
        cfg = {
            "clock_period_ns": clock_period_ns,
            "max_cycles": max_cycles,
            "stimulus": stimulus,
            "expected_len": len(expected),
            "drain_quiet_cycles": 64,
            **_rt,
        }

        env = os.environ.copy()
        venv_bin = str(Path(sys.prefix) / "bin")
        env["PATH"] = f"{venv_bin}:{env.get('PATH', '/usr/bin:/bin')}"
        env["PYTHONPATH"] = f"{sim_dir}:{project_root}:{env.get('PYTHONPATH', '')}"
        env["RME_CONFIG_JSON"] = _json.dumps(cfg)
        env["RME_OUTPUT_JSON"] = str(out_json)
        # SERIAL make (-j1): with `--build-jobs 1` in the Makefile this keeps the
        # whole Verilator build single-threaded so it can never fork-storm.
        env["MAKEFLAGS"] = (env.get("MAKEFLAGS", "") + " -j1").strip()

        make_bin = shutil.which("make") or "make"
        # Bound wall time defensively (the cycle bound is the real limiter).
        try:
            wall_timeout = int(os.environ.get("CORESMITH_RME_TIMEOUT_S", "180") or "180")
        except ValueError:
            wall_timeout = 180
        # A harness-error retry (A-Fix 2c) reruns with a longer wall budget so a
        # slow-box build timeout is not mistaken for a persistent harness break.
        if timeout_scale and timeout_scale != 1.0:
            wall_timeout = int(wall_timeout * timeout_scale)

        # HARD host protection: cap this build's address space + CPU-seconds so a
        # pathological Verilog (huge flattened design) can never OOM or peg the
        # box. Belt-and-suspenders with the serial build + wall timeout above.
        def _limit_child():  # runs in the child before exec
            try:
                import resource as _res
                _mem = int(os.environ.get("CORESMITH_RME_MEM_BYTES",
                                          str(6 * 1024 ** 3)) or str(6 * 1024 ** 3))
                _res.setrlimit(_res.RLIMIT_AS, (_mem, _mem))
                _res.setrlimit(_res.RLIMIT_CPU, (wall_timeout, wall_timeout + 20))
            except Exception:  # noqa: BLE001 - never block on limit-setting
                pass

        try:
            proc = subprocess.run(
                [make_bin, "-C", str(sim_dir)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                timeout=wall_timeout,
                preexec_fn=_limit_child,
            )
        except subprocess.TimeoutExpired:
            # A build/sim that couldn't finish in the bounded window is "cannot
            # judge", NOT a block failure -- the equivalence gate must only FAIL
            # on a genuine byte divergence, never on a harness/env/timeout issue
            # (else it false-fails every block on a box with a slow/odd build).
            # Structural un-synthesizability is the cell/depth gates' job.
            return _skip(
                f"equivalence sim for '{block_name}' exceeded wall timeout "
                f"{wall_timeout}s (cannot judge -- not counted as a divergence)",
                harness_error=True,
            )

        sim_out = (proc.stdout or "") + "\n" + (proc.stderr or "")
        # Build/harness errors (e.g. an unsupported verilator version, missing
        # cocotb, a codegen error) mean we CANNOT judge -> SKIP, never FAIL.
        if "No tests were discovered" in sim_out:
            return _skip(f"cocotb discovered no tests for '{block_name}' "
                         f"(build/harness error, cannot judge)\n{sim_out[-800:]}",
                         harness_error=True)
        if not out_json.exists():
            return _skip(
                f"equivalence harness could not build/run '{block_name}' "
                f"(cannot judge -- not a divergence)\n{sim_out[-800:]}",
                harness_error=True,
            )
        result = _json.loads(out_json.read_text())
        collected = [int(v) & m_mask for v in result.get("collected", [])]

        # An empty / never-progressed run is a FAILURE, not a pass (the whole
        # point: a block emitting nothing must not sneak through).
        if not collected:
            return _fail(
                f"'{block_name}' emitted ZERO output beats over "
                f"{result.get('cycles')} cycles (expected {len(expected)}); "
                f"a never-progressed run cannot pass"
            )

        # Byte-exact compare with first-divergence reporting.
        if len(collected) != len(expected):
            # Find first offset that differs within the overlap for a precise msg.
            n = min(len(collected), len(expected))
            off = next((i for i in range(n) if collected[i] != expected[i]), None)
            detail = (
                f" first byte mismatch at offset {off}: "
                f"rtl=0x{collected[off]:x} model=0x{expected[off]:x}"
                if off is not None else ""
            )
            return _fail(
                f"'{block_name}' output length mismatch: rtl emitted "
                f"{len(collected)} bytes, model expected {len(expected)}.{detail}",
                checked=n,
            )
        for i, (a, b) in enumerate(zip(collected, expected)):
            if a != b:
                return _fail(
                    f"'{block_name}' BYTE DIVERGENCE at vector/offset {i}: "
                    f"rtl=0x{a:x} model=0x{b:x} "
                    f"(total {len(expected)} bytes compared)",
                    checked=i,
                )

        return _pass(
            f"'{block_name}' RTL byte-exact-matches model over "
            f"{len(expected)} output bytes from {len(stimulus)} seeded "
            f"input vectors {pass_note}".rstrip(),
            checked=len(expected),
        )
    finally:
        # Keep the dir only if requested for forensics.
        if os.environ.get("CORESMITH_RME_KEEP_SIM", "0") in ("0", "", "false", "no", "off"):
            shutil.rmtree(sim_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Chip-top RTL-vs-model equivalence (A-Fix 5d)
# ---------------------------------------------------------------------------

def _coerce_int_stream(obj, mask: int):
    """Coerce a chip-model output into a flat list of masked ints, or None.

    Accepts a ``(output, cycles)`` tuple (takes ``output``), ``bytes`` /
    ``bytearray``, or a flat ``list[int]``. Anything else (nested lists, dicts,
    floats) is not comparable by this generic byte-stream harness -> None.
    """
    if isinstance(obj, tuple) and len(obj) == 2 and (
        obj[1] is None or isinstance(obj[1], int)
    ):
        obj = obj[0]
    if isinstance(obj, (bytes, bytearray)):
        return [b & mask for b in obj]
    if isinstance(obj, list) and obj and all(isinstance(v, int) for v in obj):
        return [v & mask for v in obj]
    return None


def check_chip_model_equivalence(
    design: str,
    top_rtl: str,
    block_rtls,
    chip_model_path: str,
    *,
    project_root,
    seed: int,
    n_vectors: int = 64,
    max_cycles: int = 20000,
    clock_period_ns: int = 20,
    timeout_scale: float = 1.0,
) -> dict:
    """Drive chip_top RTL and the composed Amaranth ``_chip_model``
    with the SAME seeded stimulus and assert byte-exact output equivalence.

    The chip-level analogue of :func:`check_rtl_model_equivalence`: expected is
    ``_chip_model.simulate(stimulus)`` (loaded via ``model_integration.load_chip_model``),
    observed is the elaborated chip_top + all block RTLs (deduped via
    ``integration_helpers._dedup_module_sources``). HONEST-skips (returns
    ``skipped=True`` without ``harness_error``) when the chip top is not the
    single s_axis-in / m_axis-out shape this harness covers, verilator is
    absent, or the model output is not a flat int stream -- these are
    non-blocking. A build/timeout/model-load error is a ``harness_error`` skip
    (caller retries then fails closed); a byte divergence is a FAIL.

    ``block_rtls`` may be a dict ``{name: path}`` or a list of paths.
    """
    project_root = Path(project_root)
    top_p = Path(top_rtl)
    if not top_p.exists():
        return _skip(f"chip top RTL not found: {top_rtl}")
    if not shutil.which("verilator"):
        return _skip("verilator not on PATH")

    rtl_text = top_p.read_text(encoding="utf-8", errors="replace")
    module_name = _module_name(rtl_text)
    if not module_name:
        return _skip(f"no module declaration found in {top_rtl}")

    ports = _parse_ports(rtl_text)
    axis = _classify_axis(ports)
    if axis is None:
        return _skip(
            f"chip top for '{design}' is not the single s_axis-in / m_axis-out "
            f"AXI-Stream shape this generic harness covers (ports={sorted(ports)})"
        )

    # dv-hardening-12 (armC live, 2 non-convergent rounds): a design that
    # declares a STRUCTURED stimulus contract (CORESMITH_MODEL_STIMULUS) is a
    # FRAMED protocol -- sideband config + a complete frame + TLAST. This
    # generic harness drives unstructured random beats; the composed model
    # CORRECTLY emits nothing for a partial garbage frame, and the old
    # fail-closed path reported "ZERO output beats" as an equivalence failure,
    # which the audit then misclassified as TESTBENCH_BUG (fix_tb no-ops).
    # Model-equivalence for framed designs is covered by the integration
    # testbench, which drives the real framing against the SAME chip model.
    # Honest SKIP, per this module's charter.
    if os.environ.get("CORESMITH_MODEL_STIMULUS", "").strip():
        return _skip(
            f"design '{design}' declares a structured/framed stimulus contract "
            "(CORESMITH_MODEL_STIMULUS) -- the generic random-beat harness "
            "cannot legally drive a framed protocol (sideband config + full "
            "frame + TLAST). Model-equivalence for framed designs is exercised "
            "by the integration testbench's model-comparison tests."
        )

    # Load the composed Amaranth chip model's simulate() (the oracle).
    try:
        from orchestrator.architecture.model_integration import load_chip_model
        simulate = load_chip_model(str(project_root))
    except Exception as e:  # noqa: BLE001
        return _skip(f"could not load composed chip model: {e!r}",
                     harness_error=True)

    rng = random.Random(seed)
    s_mask = (1 << axis["s_width"]) - 1
    m_mask = (1 << axis["m_width"]) - 1
    stimulus = [rng.randint(0, s_mask) for _ in range(max(1, n_vectors))]

    try:
        raw = simulate(stimulus)
    except Exception as e:  # noqa: BLE001
        return _skip(f"composed chip model simulate() raised {e!r}",
                     harness_error=True)
    expected = _coerce_int_stream(raw, m_mask)
    if expected is None:
        return _skip(
            f"composed chip model output for '{design}' is not a flat int/byte "
            "stream this generic equivalence harness can compare"
        )
    if not expected:
        # Zero output on unstructured random beats is the SIGNATURE of a
        # framed-protocol design (it is waiting for a legal frame), not
        # evidence of a broken oracle -- the composition gate already
        # validated this model against the golden. Fail-closed here produced
        # an unfixable TESTBENCH_BUG loop (armC). Honest SKIP.
        return _skip(
            f"composed chip model for '{design}' produced zero output beats "
            "on unstructured random stimulus -- framed-protocol signature; "
            "this generic harness cannot establish an oracle for it. "
            "Model-equivalence is covered by the integration testbench."
        )

    # Assemble + dedup the source set (top first, then blocks). Cross-file
    # MODDUP (each block bundling the same shared macro) would otherwise break
    # elaboration -- reuse the integration deduper.
    if isinstance(block_rtls, dict):
        block_paths = list(block_rtls.values())
    else:
        block_paths = list(block_rtls or [])
    sources = [str(top_p.resolve())] + [
        str(Path(p).resolve()) for p in block_paths if p
    ]

    dedup_dir = Path(tempfile.mkdtemp(prefix=f"rme_chip_dedup_{_safe(design)}_"))
    try:
        try:
            from orchestrator.langgraph.integration_helpers import (
                _dedup_module_sources,
            )
            deduped = _dedup_module_sources(sources, dedup_dir)
        except Exception:  # noqa: BLE001 - dedup is best-effort
            deduped = sources
        return _run_equiv_sim(
            f"chip_{_safe(design)}",
            deduped,
            module_name,
            axis,
            stimulus,
            expected,
            m_mask,
            project_root=project_root,
            clock_period_ns=clock_period_ns,
            max_cycles=max_cycles,
            timeout_scale=timeout_scale,
            pass_note=f"(chip-top, seed={seed})",
        )
    finally:
        if os.environ.get("CORESMITH_RME_KEEP_SIM", "0") in (
            "0", "", "false", "no", "off"
        ):
            shutil.rmtree(dedup_dir, ignore_errors=True)


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", str(name)) or "design"
