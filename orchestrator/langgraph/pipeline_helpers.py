# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Reusable helper functions for the ASIC pipeline.

Extracted from run_pipeline.py so that both the LangGraph pipeline graph
and the CLI runner can share the same implementation.

Provides:
- Constants: PROJECT_ROOT, PDK_ROOT, LIBERTY_FILE, CONFIG_PATH
- Config: load_config(), get_blocks_by_tier(), get_sorted_block_queue()
- Golden model: create_golden_model_wrapper()
- RTL generation: generate_rtl()
- Lint: lint_rtl()
- Testbench: generate_testbench()
- Simulation: run_simulation()
- Synthesis: synthesize_block()
- Debug: diagnose_failure()
"""

from __future__ import annotations

import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import time as _time
from pathlib import Path

import yaml

from orchestrator._timeouts import scaled

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(
    os.environ.get(
        "CORESMITH_PROJECT_ROOT",
        str(Path(__file__).resolve().parent.parent.parent),
    )
)
CODE_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = Path(
    os.environ.get(
        "CORESMITH_CONFIG_PATH",
        str(PROJECT_ROOT / "orchestrator" / "config.yaml"),
    )
)
# dv-hardening-17 (armD defect #4): honor the standard PDK_ROOT env var --
# the hardcoded PROJECT_ROOT/.pdk forced operators to symlink every run dir
# even when the daemon env carried a valid PDK_ROOT (volare layout).
PDK_ROOT = Path(os.environ.get("PDK_ROOT", "").strip() or (PROJECT_ROOT / ".pdk"))
def _find_liberty_file() -> Path:
    """Locate the Sky130 liberty file, checking both sky130A and sky130B."""
    lib_name = "sky130_fd_sc_hd__tt_025C_1v80.lib"
    for variant in ("sky130A", "sky130B"):
        candidate = PDK_ROOT / variant / "libs.ref" / "sky130_fd_sc_hd" / "lib" / lib_name
        if candidate.exists():
            return candidate
    # Fallback to sky130A path (will fail at synthesis time with a clear error)
    return PDK_ROOT / "sky130A" / "libs.ref" / "sky130_fd_sc_hd" / "lib" / lib_name


LIBERTY_FILE = _find_liberty_file()


# ---------------------------------------------------------------------------
# Preflight check -- validate PDK/EDA tools before burning retry budgets
# ---------------------------------------------------------------------------

def preflight_check(phases: list[str] | None = None) -> dict:
    """Validate that required PDK files and EDA tools exist.

    Args:
        phases: List of phases to check. Options: "pipeline", "backend".
            Defaults to ["pipeline"] if not specified.

    Returns:
        {"ok": True} or {"ok": False, "errors": [...], "warnings": [...]}
    """
    if not phases:
        phases = ["pipeline"]

    errors: list[str] = []
    warnings: list[str] = []
    skip_synth = os.environ.get("CORESMITH_SKIP_SYNTH", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    # HOT-PATCH (chip-lead, synth-gated run): honor CORESMITH_SYNTH_GENERIC in
    # preflight. Generic (PDK-free) synth maps with abc -g and needs only yosys+
    # verilator -- NOT the sky130 Liberty/PDK. Preflight previously required the
    # PDK unconditionally whenever SKIP_SYNTH was unset, falsely blocking generic
    # synth on PDK-less boxes. SYNTH_GENERIC honored downstream (~L1181); align here.
    synth_generic = os.environ.get("CORESMITH_SYNTH_GENERIC", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    if "pipeline" in phases:
        provider = os.environ.get("CORESMITH_LLM_PROVIDER", "").strip().lower()
        if provider in {"opencode", "opencode_cli", "openrouter"}:
            opencode_path = os.environ.get("OPENCODE_CLI_PATH", "").strip()
            if opencode_path:
                if not (Path(opencode_path).is_file() and os.access(opencode_path, os.X_OK)):
                    errors.append(f"OpenCode CLI is not executable: {opencode_path}")
            elif not shutil.which("opencode"):
                errors.append(
                    "OpenCode CLI not found on PATH; install with "
                    "`npm install -g opencode-ai` or set OPENCODE_CLI_PATH"
                )
        if provider in {"kimi", "kimi_cli"}:
            configured_kimi = os.environ.get("KIMI_CLI_PATH", "").strip()
            kimi_ok = (
                bool(configured_kimi)
                and Path(configured_kimi).is_file()
                and os.access(configured_kimi, os.X_OK)
            ) or bool(shutil.which("kimi"))
            if not kimi_ok:
                errors.append("Kimi Code CLI not found; install @moonshot-ai/kimi-code or set KIMI_CLI_PATH")

        if not shutil.which("verilator"):
            errors.append("verilator not found on PATH")
        if skip_synth:
            _loud = (
                "!!! CORESMITH_SKIP_SYNTH=1 -- SYNTHESIS GATE DISABLED. "
                "Yosys/PDK checks SKIPPED; un-synthesizable RTL (e.g. a "
                "non-terminating combinational cloud) WILL NOT be caught. "
                "Unset CORESMITH_SKIP_SYNTH to enable the synth gate."
            )
            warnings.append(_loud)
            import sys as _sys
            print("\n" + "=" * 78 + "\n" + _loud + "\n" + "=" * 78,
                  file=_sys.stderr, flush=True)
        elif synth_generic:
            # Generic PDK-free synth: only yosys (+ verilator above) required.
            if not shutil.which("yosys"):
                errors.append("yosys not found on PATH")
            warnings.append(
                "CORESMITH_SYNTH_GENERIC=1 -- PDK-free generic gate-mapping synth "
                "(abc -g); sky130 Liberty/PDK checks skipped (real synth still runs)."
            )
        else:
            if not LIBERTY_FILE.exists():
                errors.append(f"Liberty file not found: {LIBERTY_FILE}")
            if not shutil.which("yosys"):
                errors.append("yosys not found on PATH")
            if not PDK_ROOT.exists():
                errors.append(f"PDK root directory not found: {PDK_ROOT}")
            elif not any((PDK_ROOT / v).is_dir() for v in ("sky130A", "sky130B")):
                errors.append(f"No sky130A or sky130B variant found in {PDK_ROOT}")

        # OpenRAM must be available: without it a block needing a non-pre-built
        # SRAM macro fails LATE (at backend) or gets flagged infeasible at uarch.
        # "no OpenRAM, no greenlight" -- fail preflight instead of proceeding into
        # a run that cannot realize its memories. Opt out with
        # CORESMITH_ALLOW_NO_OPENRAM=1 ONLY for a design that provably needs no
        # generated macros (every store fits a pre-built macro).
        #
        # This matters MORE now that macro tiling is off by default: generation
        # is the only route to a geometry the PDK does not ship exactly, so the
        # exemption's precondition ("every store fits a pre-built macro") is
        # strictly narrower than it used to be -- composing or over-provisioning
        # from prebuilt parts no longer counts as a fit.
        if os.environ.get("CORESMITH_ALLOW_NO_OPENRAM", "").strip().lower() not in {
            "1", "true", "yes", "on",
        }:
            try:
                from orchestrator.langgraph.openram_gen import openram_available
                if not openram_available():
                    errors.append(
                        "OpenRAM not available (pip install openram, or set "
                        "OPENRAM_HOME) -- SRAM macro generation is impossible, so a "
                        "block needing a non-pre-built macro cannot be realized. "
                        "Install OpenRAM, or set CORESMITH_ALLOW_NO_OPENRAM=1 if "
                        "this design provably needs no generated macros. NOTE: "
                        "with macro tiling disabled (the default), 'fits a "
                        "pre-built macro' means an EXACT geometry match -- a "
                        "geometry that used to be satisfied by composition or "
                        "over-provisioning now requires generation."
                    )
            except Exception as _oe:  # noqa: BLE001 - import failure == not available
                errors.append(f"OpenRAM availability probe failed: {_oe}")
        else:
            try:
                from orchestrator.langgraph.openram_gen import tiling_allowed
                _tiling = tiling_allowed()
            except ImportError:
                _tiling = False
            if not _tiling:
                warnings.append(
                    "CORESMITH_ALLOW_NO_OPENRAM=1 with macro tiling disabled: "
                    "this design must satisfy EVERY store with an exact "
                    "pre-built macro or an explicitly-accepted flop tier. Any "
                    "other geometry will escalate rather than being tiled."
                )

    if "backend" in phases:
        from orchestrator.langgraph.backend_helpers import (
            CELL_GDS,
            CELL_LEF,
            MAGIC_BIN,
            MAGIC_RC,
            NETGEN_BIN,
            OPENROAD_BIN,
            TECH_LEF,
        )
        if not TECH_LEF.exists():
            errors.append(f"Tech LEF not found: {TECH_LEF}")
        if not CELL_LEF.exists():
            errors.append(f"Cell LEF not found: {CELL_LEF}")
        if not CELL_GDS.exists():
            errors.append(f"Cell GDS not found: {CELL_GDS}")
        if not MAGIC_RC.exists():
            errors.append(f"Magic RC file not found: {MAGIC_RC}")
        if not Path(OPENROAD_BIN).exists():
            errors.append(f"OpenROAD binary/script not found: {OPENROAD_BIN}")
        if not Path(MAGIC_BIN).exists():
            errors.append(f"Magic binary/script not found: {MAGIC_BIN}")
        if not Path(NETGEN_BIN).exists():
            errors.append(f"Netgen binary/script not found: {NETGEN_BIN}")

    return {"ok": len(errors) == 0, "errors": errors, "warnings": warnings}

# ANSI colors for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def log(msg: str, color: str = "") -> None:
    """Print a coloured log line to stdout."""
    prefix = f"{color}{BOLD}" if color else ""
    suffix = RESET if color else ""
    print(f"{prefix}{msg}{suffix}", flush=True)


# ---------------------------------------------------------------------------
# Step log files  (<project>/.coresmith/step_logs/<block>/<step>_attempt<N>.log)
# ---------------------------------------------------------------------------

_LOG_DIR = Path(
    os.environ.get("CORESMITH_LOG_DIR", str(PROJECT_ROOT / ".coresmith" / "step_logs"))
)


def _write_step_log(
    block_name: str,
    step: str,
    cmd: list[str],
    result: subprocess.CompletedProcess,
    attempt: int = 1,
) -> str:
    """Write full subprocess output to /tmp and return the log file path."""
    log_dir = _LOG_DIR / block_name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{step}_attempt{attempt}.log"

    ts = _time.strftime("%Y-%m-%dT%H:%M:%S", _time.gmtime())
    content = (
        f"=== {step.upper()} LOG ===\n"
        f"Timestamp: {ts}\n"
        f"Block: {block_name}\n"
        f"Attempt: {attempt}\n"
        f"Command: {' '.join(cmd)}\n"
        f"Return code: {result.returncode}\n"
        f"\n=== STDOUT ===\n"
        f"{result.stdout}\n"
        f"\n=== STDERR ===\n"
        f"{result.stderr}\n"
    )
    log_file.write_text(content, encoding="utf-8")
    return str(log_file)


def _write_step_log_error(
    block_name: str,
    step: str,
    cmd: list[str],
    error_msg: str,
    attempt: int = 1,
) -> str:
    """Write an error-only log file when subprocess didn't complete normally."""
    log_dir = _LOG_DIR / block_name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{step}_attempt{attempt}.log"

    ts = _time.strftime("%Y-%m-%dT%H:%M:%S", _time.gmtime())
    content = (
        f"=== {step.upper()} LOG ===\n"
        f"Timestamp: {ts}\n"
        f"Block: {block_name}\n"
        f"Attempt: {attempt}\n"
        f"Command: {' '.join(cmd)}\n"
        f"Return code: N/A (exception)\n"
        f"\n=== ERROR ===\n"
        f"{error_msg}\n"
    )
    log_file.write_text(content, encoding="utf-8")
    return str(log_file)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config() -> dict:
    """Load the project config from orchestrator/config.yaml.

    If ``CORESMITH_BLOCKS_FILE`` is set, the ``blocks:`` section is replaced
    with the contents of that YAML file. Used by ``make demo`` and the
    nightly e2e job to swap in a small reference design without touching
    the canonical config.
    """
    config_path = CONFIG_PATH
    if not config_path.exists():
        repo_config = CODE_ROOT / "orchestrator" / "config.yaml"
        if repo_config.exists():
            config_path = repo_config

    with open(config_path) as f:
        config = yaml.safe_load(f)

    blocks_override = os.environ.get("CORESMITH_BLOCKS_FILE")
    if blocks_override:
        with open(blocks_override) as f:
            override = yaml.safe_load(f) or {}
        blocks = override.get("blocks", override) if isinstance(override, dict) else override
        if isinstance(blocks, list):
            config["blocks"] = {
                block["name"]: {k: v for k, v in block.items() if k != "name"}
                for block in blocks
                if isinstance(block, dict) and block.get("name")
            }
        else:
            config["blocks"] = blocks

    return config


def get_blocks_by_tier(config: dict) -> dict[int, list[dict]]:
    """Group blocks by tier, returning sorted dict."""
    tiers: dict[int, list[dict]] = {}
    for name, spec in config.get("blocks", {}).items():
        tier = spec.get("tier", 1)
        block = {"name": name, **spec}
        tiers.setdefault(tier, []).append(block)
    return dict(sorted(tiers.items()))


def get_sorted_block_queue(config: dict) -> list[dict]:
    """Return a flat list of blocks sorted by tier (1 -> 2 -> 3)."""
    tiers = get_blocks_by_tier(config)
    queue: list[dict] = []
    for _tier_num, blocks in tiers.items():
        queue.extend(blocks)
    return queue


def get_tier_list(block_queue: list[dict]) -> list[int]:
    """Return sorted unique tier values from a block queue.

    Example: ``[1, 2, 3]`` for blocks spanning three tiers.
    """
    return sorted(set(b.get("tier", 1) for b in block_queue))


def get_blocks_for_tier(block_queue: list[dict], tier: int) -> list[dict]:
    """Filter blocks by tier number."""
    return [b for b in block_queue if b.get("tier", 1) == tier]


# ---------------------------------------------------------------------------
# Golden-source slice resolution.
#
# The architecture phase assigns each block a per-block golden SLICE via a
# ``<file>.py:name1,name2,ClassName`` syntax in ``python_source`` (e.g.
# ``<golden>.py:BitReader`` for a bit-reader, ``<golden>.py:tbl_unpack,
# param_unpack`` for a table store). Historically the consumers did
# ``PROJECT_ROOT / python_source`` and treated the whole string as a path --
# so ``<golden>.py:BitReader`` did not exist and the block was fed an EMPTY
# golden, which the feasibility judge then flagged [capability]
# ("cannot be frozen -- golden slice is empty"). These helpers parse the slice
# syntax and extract the named defs (+ their local-helper closure + imports/
# module constants), so each block sees exactly its own golden math.
# ---------------------------------------------------------------------------

def _split_source_ref(python_source_ref: str) -> tuple[str, list[str]]:
    """Split ``<file>.py:name1,name2`` -> (file_path, [names]); ([names] empty
    when there is no ``:`` slice suffix)."""
    ref = (python_source_ref or "").strip()
    if not ref:
        return "", []
    marker = ".py:"
    idx = ref.find(marker)
    if idx == -1:
        return ref, []
    file_path = ref[: idx + len(".py")]
    names = [n.strip() for n in ref[idx + len(marker):].split(",") if n.strip()]
    return file_path, names


def _slice_python_source(file_text: str, names: list[str]) -> str:
    """Extract the named top-level functions/classes -- OR class *methods* --
    from ``file_text``, plus the code they transitively reach: local helper
    functions they call, and, when a named target (or a reached callee) is a
    class method, the ENTIRE enclosing class (a method cannot stand alone as
    valid, importable Python). Also carries the module imports + top-level
    constants. Returns a self-contained source string; falls back to the whole
    file on any parse error or when nothing resolves.

    Method-awareness matters because real goldens routinely put the datapath on
    a class (e.g. ``ReferenceCodecDecoder._decode_frame``) and expose only a few
    top-level helpers. The earlier version indexed *only* top-level defs, so a
    ref naming ``_decode_frame`` (a method) resolved to an EMPTY/partial slice --
    the feasibility judge then flagged ``[capability]`` ("golden slice cannot be
    frozen"). Emitting the whole enclosing class closes that gap honestly."""
    import ast as _ast
    try:
        tree = _ast.parse(file_text)
    except SyntaxError:
        return file_text
    lines = file_text.splitlines(keepends=True)

    top_defs: dict = {}        # name -> top-level Func/AsyncFunc/ClassDef node
    class_defs: dict = {}      # class name -> ClassDef node
    method_owners: dict = {}   # method name -> [class name, ...]
    imports: list = []
    consts: list = []
    for node in tree.body:
        if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef,
                             _ast.ClassDef)):
            top_defs[node.name] = node
        if isinstance(node, _ast.ClassDef):
            class_defs[node.name] = node
            for m in node.body:
                if isinstance(m, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                    method_owners.setdefault(m.name, []).append(node.name)
        elif isinstance(node, (_ast.Import, _ast.ImportFrom)):
            imports.append(node)
        elif isinstance(node, (_ast.Assign, _ast.AnnAssign)):
            consts.append(node)

    def _callees(node):
        for sub in _ast.walk(node):
            if isinstance(sub, _ast.Call):
                fn = sub.func
                yield (fn.id if isinstance(fn, _ast.Name)
                       else fn.attr if isinstance(fn, _ast.Attribute)
                       else None)

    want_top: set = set()      # top-level fn/class names emitted whole
    want_classes: set = set()  # class names emitted whole (a method was reached)

    # Seed: a name may be a top-level def OR a class method (prefer top-level).
    fn_frontier: list = []
    for nm in names:
        if nm in top_defs:
            fn_frontier.append(nm)
        elif nm in method_owners:
            for cls in method_owners[nm]:
                want_classes.add(cls)

    walked_classes: set = set()
    while fn_frontier or (want_classes - walked_classes):
        # 1) drain the top-level-function frontier
        while fn_frontier:
            nm = fn_frontier.pop()
            if nm in want_top:
                continue
            want_top.add(nm)
            for callee in _callees(top_defs[nm]):
                if callee in top_defs and callee not in want_top:
                    fn_frontier.append(callee)
                elif callee in method_owners:
                    for cls in method_owners[callee]:
                        want_classes.add(cls)
        # 2) walk each newly-reached class WHOLE to discover the top-level
        #    helpers it calls (method->method calls stay inside the emitted
        #    class, so no per-method bookkeeping is needed).
        for cls in list(want_classes - walked_classes):
            walked_classes.add(cls)
            for callee in _callees(class_defs[cls]):
                if callee in top_defs and callee not in want_top:
                    fn_frontier.append(callee)
                elif callee in method_owners:
                    for oc in method_owners[callee]:
                        want_classes.add(oc)

    if not want_top and not want_classes:
        return file_text  # named nothing resolvable -> whole file (safe)

    def _seg(node) -> str:
        end = getattr(node, "end_lineno", node.lineno) or node.lineno
        return "".join(lines[node.lineno - 1:end])

    # Emit imports, consts, then the reached top-level defs + whole classes in
    # source order (de-dup a class that was BOTH named directly and reached).
    emit_nodes = ([top_defs[n] for n in want_top]
                  + [class_defs[c] for c in want_classes])
    seen_lineno: set = set()
    uniq: list = []
    for n in sorted(emit_nodes, key=lambda n: n.lineno):
        if n.lineno in seen_lineno:
            continue
        seen_lineno.add(n.lineno)
        uniq.append(n)
    parts = [_seg(n) for n in imports] + [_seg(n) for n in consts]
    parts += [_seg(n) for n in uniq]
    return "\n".join(parts)


def _live_python_source_ref(block: dict, project_root=None) -> str:
    """Return the block's ``python_source`` ref, preferring the LIVE
    ``.coresmith/block_diagram.json`` on disk over the (possibly checkpointed)
    ``block`` dict. This makes a chip-lead's on-disk ``python_source`` slice
    edit a first-class triage lever on a feasibility revise -- exactly like the
    interface/area edits, which are already re-read from their frozen artifacts.
    Falls back to the ``block`` dict's ref when the live file is missing or the
    block is absent from it."""
    checkpoint_ref = (block or {}).get("python_source", "") or ""
    name = (block or {}).get("name", "")
    if not name:
        return checkpoint_ref
    import json as _json
    bd_path = Path(project_root or PROJECT_ROOT) / ".coresmith" / "block_diagram.json"
    if not bd_path.exists():
        return checkpoint_ref
    try:
        bd = _json.loads(bd_path.read_text(encoding="utf-8"))
    except (OSError, _json.JSONDecodeError):
        return checkpoint_ref
    for b in bd.get("blocks", []):
        if b.get("name") == name:
            return (b.get("python_source", "") or "") or checkpoint_ref
    return checkpoint_ref


def resolve_python_source(python_source_ref: str, project_root=None) -> str:
    """Resolve a ``python_source`` ref to source TEXT: the sliced defs when it
    carries a ``:name`` suffix, else the whole file. Returns "" if the file
    can't be read."""
    file_path, names = _split_source_ref(python_source_ref)
    if not file_path:
        return ""
    p = Path(file_path)
    if not p.is_absolute():
        p = Path(project_root or PROJECT_ROOT) / file_path
    if not p.exists() or p.is_dir():
        return ""
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return _slice_python_source(text, names) if names else text


def python_source_file(python_source_ref: str, project_root=None):
    """Resolve just the FILE of a ``python_source`` ref (stripping any ``:name``
    slice suffix). Returns a Path that exists, or None."""
    file_path, _ = _split_source_ref(python_source_ref)
    if not file_path:
        return None
    p = Path(file_path)
    if not p.is_absolute():
        p = Path(project_root or PROJECT_ROOT) / file_path
    return p if (p.exists() and not p.is_dir()) else None


# ---------------------------------------------------------------------------
# Golden model wrapper creation
# ---------------------------------------------------------------------------

def create_golden_model_wrapper(block_name: str, python_source_path: str) -> None:
    """Create a <block_name>_model.py wrapper on PYTHONPATH for cocotb import.

    The testbench generator expects to import ``from <block_name>_model import ...``.
    We create a thin wrapper that imports from the actual source location.
    """
    # Prefer the per-block Amaranth golden model as the cocotb oracle when the
    # block-goldens feature is on. The block model (arch/block_models/<block>.py)
    # exposes the block-level reference API (e.g. _process_mb, field masks),
    # whereas the flat python_source (a whole-chip golden) typically does NOT.
    # Importing the flat golden leaves the testbench's
    # `from <block>_model import <block-level-symbol>` unresolved, which silently
    # falls back to a wrong/missing module. (engine fix, 2026-06-21)
    block_model_module = ""
    try:
        from orchestrator.architecture import composition as _composition
        if _composition.block_goldens_enabled():
            _bm = (
                PROJECT_ROOT / "arch" / _composition.BLOCK_MODELS_DIRNAME
                / f"{block_name}.py"
            )
            if _bm.exists():
                block_model_module = ".".join(
                    _bm.relative_to(PROJECT_ROOT).with_suffix("").parts
                )
    except Exception:  # noqa: BLE001
        block_model_module = ""

    # armC pass-2 finding [dv-hardening-9]: architecture-driven runs have NO
    # per-block python_source (the golden is chip-level), so the early return
    # here left every block without its <block>_model.py oracle wrapper -- the
    # cocotb TB import crashed one block's SIM outright and the equivalence
    # check silently SKIPPED for all six. The block MODEL alone is a fully
    # valid oracle source: proceed when either source exists.
    have_python_source = bool(python_source_path and python_source_path.strip())
    # Strip any `:name1,name2` slice suffix to resolve the underlying golden
    # FILE (the wrapper imports the whole module; the slice is for the judge).
    source_path = python_source_file(python_source_path, PROJECT_ROOT) if have_python_source else None
    if source_path is not None and (
        not source_path.exists() or source_path.is_dir()
    ):
        source_path = None
    if source_path is None and not block_model_module:
        return

    wrapper_dir = PROJECT_ROOT / "tb" / "cocotb"
    wrapper_dir.mkdir(parents=True, exist_ok=True)
    wrapper_path = wrapper_dir / f"{block_name}_model.py"

    if block_model_module:
        module_path = block_model_module
    else:
        module_parts = source_path.relative_to(PROJECT_ROOT).with_suffix("").parts
        module_path = ".".join(module_parts)

    # Refresh a stale wrapper that points at the wrong module (e.g. an old
    # wrapper that imported the flat golden instead of the block model).
    if wrapper_path.exists():
        try:
            _cur = wrapper_path.read_text(encoding="utf-8")
        except OSError:
            _cur = ""
        if f'importlib.import_module("{module_path}")' in _cur:
            return
        # else fall through and rewrite with the correct module_path

    # Re-export EVERYTHING from the source module, INCLUDING underscore-prefixed
    # names. A bare `from <mod> import *` skips names starting with "_" (and
    # honors __all__), so block-level reference helpers like `_process_mb`
    # would be missing and the testbench's
    # `from <block>_model import _process_mb` would raise ImportError. We copy
    # the full module namespace instead. (engine fix, 2026-06-21)
    wrapper_content = f'''"""Auto-generated wrapper for {block_name} golden model.

Re-exports the full namespace of {module_path} (including private/underscore
names) so cocotb testbenches can import block-level reference helpers.
"""
import sys
import importlib
from pathlib import Path

# Add project root to path so we can import the golden model
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

_mod = importlib.import_module("{module_path}")
for _name in dir(_mod):
    if _name == "__name__":
        continue
    globals()[_name] = getattr(_mod, _name)
del _mod, _name
'''
    wrapper_path.write_text(wrapper_content)


# ---------------------------------------------------------------------------
# Hardware golden (microarchitecture restructure, Phase-3 rollout step 1)
# ---------------------------------------------------------------------------

def _rtl_from_hw_golden_enabled() -> bool:
    """When on, RTL is generated as a *lowering* of the per-block Amaranth hardware
    golden (``arch/block_models/<block>.py``) -- already in hardware semantics:
    fixed-point arithmetic, resolved feedback/state, derating applied -- instead
    of an independent re-transcription of the float reference golden
    (``python_source``).

    Opt-in (default off) until validated on a full run; set
    ``CORESMITH_RTL_FROM_HW_GOLDEN=1`` to enable. See
    coresmith_microarch_phase_design.html for the rationale (the two-golden
    model: reference golden = intent/fidelity target, hardware golden = the
    bit-exact RTL target).
    """
    from orchestrator.profile import ensure_applied, flag_enabled
    ensure_applied()
    return flag_enabled("CORESMITH_RTL_FROM_HW_GOLDEN", default=False)


def block_hw_golden_rel(block: dict) -> str:
    """Project-relative path to the block's Amaranth hardware-golden model, or ''.

    The model is emitted by the ``BlockGoldenGenerator`` into
    ``arch/block_models/<block>.py`` during the uArch-spec stage, so it already
    exists on disk by the time RTL is generated.
    """
    name = (block or {}).get("name", "")
    if not name:
        return ""
    try:
        from orchestrator.architecture import composition as _composition
        _dirname = _composition.BLOCK_MODELS_DIRNAME
    except Exception:  # noqa: BLE001
        _dirname = "block_models"
    bm = PROJECT_ROOT / "arch" / _dirname / f"{name}.py"
    if bm.is_file():
        try:
            return str(bm.relative_to(PROJECT_ROOT))
        except ValueError:
            return ""
    return ""


def rtl_reference_source(block: dict) -> tuple[str, bool]:
    """Return ``(relative_source_path, is_hw_golden)`` the RTL generator should
    transcribe.

    When the RTL-from-hardware-golden rollout is enabled and the block's Amaranth
    model exists, that model is the reference and ``is_hw_golden`` is True (the
    RTL becomes a mechanical, functionally byte-exact lowering of a proven,
    hardware-semantics model). Otherwise falls back to the float reference golden
    (``python_source``), preserving legacy behavior.
    """
    float_src = (block or {}).get("python_source", "") or ""
    if _rtl_from_hw_golden_enabled():
        hw = block_hw_golden_rel(block)
        if hw:
            return hw, True
    return float_src, False


# ---------------------------------------------------------------------------
# µarch composition gate -- honest exit banner
# ---------------------------------------------------------------------------

def uarch_gate_banner(model_integration_result: dict | None) -> tuple[str, str]:
    """``(banner_text, colour)`` for the µarch gate's exit banner.

    CLEAN means nothing fired. Measured live: the gate detected a model-level
    mismatch, logged it, DISMISSED it as advisory (the deterministic-BFM bypass,
    which is a legitimate non-blocking decision) -- and four lines later the run
    printed a green "µARCH GATE CLEAN". Two true statements were composed into a
    false one, and the green line is the one a reader carries away.

    This function does not change what the gate DOES -- the bypass stays
    advisory, the run still proceeds -- only what it SAYS. Every outcome that is
    not "nothing fired" gets a yellow banner naming the finding and stating
    explicitly that it is non-blocking.
    """
    res = model_integration_result if isinstance(model_integration_result, dict) else {}
    tail = " -> beginning RTL pass (pass 2)"

    if res.get("advisory_bypass"):
        n = len(res.get("violations") or [])
        where = res.get("first_divergence_block") or ""
        return (
            "µARCH GATE NOT CLEAN: "
            + (f"{n} model-level mismatch(es)" if n else "a model-level mismatch")
            + " were DISMISSED as ADVISORY (non-blocking; the deterministic "
            "integration DV on the real RTL is the authoritative check) and "
            "carried forward"
            + (f"; first-divergence block: {where}" if where else
               "; first-divergence block: (unlocalized)")
            + tail,
            YELLOW,
        )

    if res and res.get("passed") is False:
        return (
            "µARCH GATE NOT CLEAN: the gate did not pass"
            + (f" (action taken: {res.get('action_taken')})"
               if res.get("action_taken") else "")
            + tail,
            YELLOW,
        )

    if res.get("derate_signed_off"):
        return (
            "µARCH GATE PASSED WITH A SIGNED-OFF DERATE (within budget, "
            "recorded in the derate ledger)" + tail,
            YELLOW,
        )

    # Nothing fired (or no record at all -- the gate never ran on this path).
    return ("µARCH GATE CLEAN" + tail, CYAN)


# ---------------------------------------------------------------------------
# Microarchitecture Spec Generation
# ---------------------------------------------------------------------------

def _report_uarch_golden(block_name: str, ref: str, resolved: str) -> None:
    """D1: never let an unreadable / absent golden pass SILENTLY.

    Three cases, three different truths -- and the old code told none of them,
    because every reader of a golden returns "" on failure while the uArch
    prompt appended the golden section unconditionally. Measured on a
    validation run: all 8 uArch calls carried an empty golden block and nothing
    anywhere said so.

      * ref declared AND resolved  -> nothing to say.
      * ref declared, resolved ""  -> a BROKEN PROMISE. The block diagram names
        a transcription target that cannot be read; the spec author would have
        invented the math in its place. RED log + carried defect naming the ref.
      * no ref at all              -> this block legitimately has no reference
        golden (routinely most blocks of a design). Log it and carry it forward,
        because the spec -- and every RTL lowering below it -- is then
        unconstrained by any reference and the final report should say so.

    Neither empty case raises here: the hard failure belongs where a FLAG
    explicitly promised a golden (``CORESMITH_RTL_FROM_HW_GOLDEN`` in the RTL
    generator, which tells the model in as many words that its output must be
    byte-exact to a file). A block diagram ref is a weaker claim, and failing
    every such block would trade a silent gap for a stopped run.
    """
    if resolved.strip():
        return
    if ref:
        log(f"  [UARCH] {block_name}: DECLARED golden '{ref}' resolves to "
            "NOTHING (missing file, bad slice name, or unreadable). The spec "
            "author is being asked to design with no transcription target and "
            "will invent the math. FIX THE REF.", RED)
        kind, detail = "declared_golden_unreadable", (
            f"block '{block_name}' declares python_source '{ref}' but it "
            "resolves to nothing, so its uArch spec was written with an EMPTY "
            "golden model")
    else:
        log(f"  [UARCH] {block_name}: NO reference golden model (no "
            "python_source in the block diagram). The uArch spec -- and every "
            "RTL lowering below it -- is unconstrained by any reference for "
            "this block.", YELLOW)
        kind, detail = "no_reference_golden", (
            f"block '{block_name}' has no reference golden model "
            "(python_source absent), so its uArch spec and RTL were never "
            "checked against one")
    try:
        from orchestrator.langgraph.pipeline_graph import (
            record_carried_forward_defect,
        )
        record_carried_forward_defect(str(PROJECT_ROOT), {
            "gate": "uarch_golden",
            "kind": kind,
            "advisory": True,
            "unmodeled": detail,
            # The explanation the report renders. It was built right above and
            # then went nowhere the report reads, so every uarch_golden entry
            # in the ledger carried a gate/kind pair and no account of itself.
            "detail": detail,
            "first_divergence_block": block_name,
            "note": "",
        })
    except Exception:  # noqa: BLE001 - reporting must never block generation
        pass


async def generate_uarch_spec(
    block: dict,
    feedback: str = "",
    previous_spec: str = "",
    constraints: list[dict] = None,
    callbacks: list = None,
    resume_session_id: str | None = None,
) -> dict:
    """Generate a microarchitecture specification from Python golden model.

    Returns a dict with keys: spec_text, spec_summary, block_name, session_id.
    Also writes the spec to ``arch/uarch_specs/<block_name>.md``.

    ``resume_session_id`` (codex only, gated by CORESMITH_CODEX_RESUME) resumes
    the block's prior codex session so a convergent revise round continues its
    earlier reasoning; ``None`` starts fresh. The codex session id this call
    produced is returned as ``session_id`` so the caller can thread it into the
    next round (or drop it on a fresh-session escalation).
    """
    from orchestrator.langchain.agents.uarch_spec_generator import UarchSpecGenerator

    # Resolve the block's golden slice (handles the `<file>.py:name1,name2`
    # syntax the architecture phase assigns, so the feasibility judge sees this
    # block's OWN golden math -- not empty, not the whole chip golden). Read the
    # ref from the LIVE block_diagram.json so a chip-lead's on-disk slice edit is
    # honoured on a feasibility revise (not shadowed by the checkpoint).
    _golden_ref = _live_python_source_ref(block, PROJECT_ROOT)
    python_source = resolve_python_source(_golden_ref, PROJECT_ROOT)
    _report_uarch_golden(block.get("name", ""), _golden_ref, python_source)

    # C13: snapshot the pre-call canonical spec + call start time. The old
    # arbitration let (a) the PRE-CALL on-disk spec compete as a "candidate"
    # and win on length, and (b) recovery resurrect a PREVIOUS call's
    # codex-call-* artifact -- observed: a re-spec after a contract amendment
    # returned a byte-identical 110K spec still carrying the pre-amendment
    # widths, so the fresh RTL came out stale-width and deterministically
    # failed sim. Only artifacts produced DURING this call may win.
    from orchestrator.architecture.state import ARCH_DOC_DIR as _ADD

    _spec_path_pre = PROJECT_ROOT / _ADD / "uarch_specs" / f"{block['name']}.md"
    _spec_pre, _spec_pre_mtime = "", 0
    if _spec_path_pre.exists():
        try:
            _spec_pre = _spec_path_pre.read_text()
            _spec_pre_mtime = _spec_path_pre.stat().st_mtime_ns
        except OSError:
            pass
    _spec_call_start = _time.time()

    # Block-level: use the block_model() default (Sonnet); pipe DEFAULT_MODEL
    # would force Opus and is wasteful for per-block work.
    agent = UarchSpecGenerator(temperature=0.2)
    result = await agent.generate(
        block_name=block["name"],
        python_source=python_source,
        description=block.get("description", ""),
        feedback=feedback,
        previous_spec=previous_spec,
        constraints=constraints,
        callbacks=callbacks,
        project_root=str(PROJECT_ROOT),
        resume_session_id=resume_session_id,
    )
    # Surface the codex session id captured from this call (empty on non-codex /
    # no-session) so the node can persist it for the next convergent round.
    # Fully guarded: a stubbed generator may carry no ``.llm``.
    result["session_id"] = getattr(getattr(agent, "llm", None), "last_session_id", "") or ""

    from orchestrator.architecture.state import ARCH_DOC_DIR

    spec_dir = PROJECT_ROOT / ARCH_DOC_DIR / "uarch_specs"
    spec_dir.mkdir(parents=True, exist_ok=True)
    spec_path = spec_dir / f"{block['name']}.md"
    # Disk-first: if the agent wrote a richer spec via its write/edit tool,
    # prefer that artifact over stdout. Codex may write inside an isolated
    # codex-call-* workdir and return only a path/status message, so recover
    # that per-call artifact before deciding what to persist canonically.
    # C13: only artifacts produced DURING this call are candidates -- the
    # pre-call canonical spec used to compete (and WIN on length), and
    # recovery used to resurrect a previous call's workdir artifact.
    spec_returned = result.get("spec_text") or ""
    recovered = _recover_codex_call_artifact(
        PROJECT_ROOT,
        Path(ARCH_DOC_DIR) / "uarch_specs" / f"{block['name']}.md",
        min_mtime=_spec_call_start,
    )
    candidates: list[str] = []
    if spec_path.exists():
        try:
            _spec_now = spec_path.read_text()
            if (_spec_now != _spec_pre
                    or spec_path.stat().st_mtime_ns != _spec_pre_mtime):
                candidates.append(_spec_now)  # written during THIS call
        except OSError:
            pass
    if recovered:
        candidates.append(recovered)
    if spec_returned:
        candidates.append(spec_returned)

    spec_text = _choose_generated_markdown(candidates)
    if not spec_text:
        # Nothing fresh was produced. NEVER silently re-adopt the stale
        # pre-call spec as if it were this call's output -- that is exactly
        # how a re-spec after a contract amendment "returned" a byte-identical
        # pre-amendment spec and the downstream RTL failed deterministically.
        log(f"  [UARCH] {block['name']}: spec generation produced NO fresh "
            f"artifact (canonical spec unchanged, no per-call workdir "
            f"artifact, empty stdout) -- treating as a FAILED generation",
            RED)
        result["error"] = result.get("error") or (
            "spec generation produced no fresh artifact (stale spec is not "
            "success)")
        result["spec_text"] = _spec_pre
        result["spec_path"] = str(spec_path)
        return result
    spec_path.write_text(spec_text)
    result["spec_text"] = spec_text
    result["spec_path"] = str(spec_path)

    # C7(a): stamp WHICH contract this spec was generated against. The
    # integration staleness preflight compares this to the live contract to
    # catch a block whose spec predates a later contract amendment (observed:
    # bitstream_reader consumed 8-bit bit_req while post-amendment producers
    # emitted 9-bit; the Lead then bridged it with a semantics-destroying
    # truncation adapter).
    try:
        _bd = PROJECT_ROOT / ".coresmith" / "blocks" / block["name"]
        _bd.mkdir(parents=True, exist_ok=True)
        _ct = block_contract_sha1(str(PROJECT_ROOT), block["name"])
        if _ct:
            (_bd / "uarch_spec_contract_sha1").write_text(_ct, encoding="utf-8")
    except OSError:
        pass

    # Block model (env-gated). When CORESMITH_BLOCK_GOLDENS is on, emit a
    # per-block Amaranth model (arch/block_models/<block>.py) transcribing the
    # reference implementation's exact math for this block with real clock /
    # handshake / latency semantics, so the model-integration agent can wire the
    # block models into a top-level chip model and the deterministic
    # model-integration gate can prove the simulated chip output == the
    # reference implementation BEFORE end-of-pipeline DV. Best-effort: a failure
    # here is logged but does not crash the spec step -- the gate is the hard
    # gate. Flag off => byte-identical to before (no new file).
    await _maybe_generate_block_golden(block, callbacks=callbacks)

    return result


def gate_scoped_reuse_reason(project_root, block_name: str) -> str:
    """Non-empty reason when this block's spec/model must be REUSED verbatim
    during a µarch-gate revise iteration (disk-first signals, no state needed):

    - ``.coresmith/_last_gate_signature.txt`` exists  => a composition-gate
      FAILURE iteration is in progress (written on gate fail, cleared on pass);
    - ``.coresmith/blocks/<b>/gate_feedback.txt`` is ABSENT => init_tier did
      NOT implicate this block (precise localization writes feedback only for
      affected blocks and clears it for the rest; a broadcast writes it for
      every tier block, so broadcasts are unaffected by this skip).

    Rationale (armC live, 2026-07-05): each gate revise round re-drew specs,
    reviews, and models for every NON-implicated block (~5 LLM rounds of pure
    waste per iteration) because regen is unconditional on tier re-entry --
    and the integration reviewer's edits kept bumping spec mtimes, cascading
    model regens that even clobbered an operator hand-patch.

    ``CORESMITH_GATE_SCOPED_REVISE=0`` disables (old behavior).
    """
    if os.environ.get(
        "CORESMITH_GATE_SCOPED_REVISE", "1"
    ).strip().lower() in ("0", "false", "no", "off"):
        return ""
    root = Path(project_root)
    if not (root / ".coresmith" / "_last_gate_signature.txt").exists():
        return ""
    if (root / ".coresmith" / "blocks" / block_name / "gate_feedback.txt").exists():
        return ""
    return (
        "gate-scoped revise: the µarch gate did not implicate this block "
        "(no gate_feedback.txt) -- reusing the on-disk spec/model verbatim"
    )


# C6: markers a generated block model uses to declare it CANNOT realize the
# datapath from the frozen interface. "INFEASIBLE-INTERFACE-GAP" is the
# structured convention the generator prompt mandates; "ERROR_INTERFACE_GAP"
# is the observed stub idiom (residual_recon_engine emitted a model routing
# every well-formed job to that error state -- and the stub-consistent TB+RTL
# then passed per-block DV 6/6 while the chip could not decode one frame).
_MODEL_GAP_MARKERS = ("INFEASIBLE-INTERFACE-GAP", "ERROR_INTERFACE_GAP")


def _gap_resolution_enabled() -> bool:
    """C10 gap->resolution loop, default ON; opt out with
    ``CORESMITH_GAP_RESOLUTION=0``."""
    return os.environ.get("CORESMITH_GAP_RESOLUTION", "").strip().lower() \
        not in {"0", "false", "no", "off"}


def _gap_resolution_rounds_cap() -> int:
    """Max auto-resolution rounds per block-model generation (default 3);
    ``CORESMITH_GAP_RESOLUTION_ROUNDS`` overrides."""
    try:
        return max(0, int(os.environ.get(
            "CORESMITH_GAP_RESOLUTION_ROUNDS", "3")))
    except ValueError:
        return 3


def detect_model_interface_gap(model_text: str) -> str:
    """First line of a generated block model that declares an interface gap
    ('' when none). See ``_MODEL_GAP_MARKERS`` for the recognized forms."""
    for line in (model_text or "").splitlines():
        for marker in _MODEL_GAP_MARKERS:
            if marker in line:
                return line.strip()[:500]
    return ""


def stale_uarch_spec_blocks(project_root, block_names) -> list:
    """C7(b): blocks whose uarch spec was generated against an OLDER interface
    contract than the live one (recorded ``uarch_spec_contract_sha1`` sidecar
    != current contract hash). Returns ``[{"block", "recorded", "current"}]``.
    Blocks with no sidecar (older runs) are never flagged."""
    out = []
    for name in block_names:
        p = (Path(project_root) / ".coresmith" / "blocks" / name
             / "uarch_spec_contract_sha1")
        if not p.exists():
            continue
        try:
            rec = p.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        cur = block_contract_sha1(project_root, name)
        if rec and cur and rec != cur:
            out.append({"block": name, "recorded": rec, "current": cur})
    return out


def refresh_current_sidecars(project_root, block_names,
                             changed_edge_substrings=None) -> list:
    """Engine follow-up #6: re-sync the ``uarch_spec_contract_sha1`` sidecar (+
    ``best_result.contract_sha1``) to the LIVE per-block contract hash for
    blocks whose passing RTL is still intact -- WITHOUT regenerating them.

    The staleness pileup: a ``--force`` regen wave (or a mid-run contract edit)
    touches many blocks' recorded provenance, and blocks regenerated BEFORE a
    later edit re-stale at the next integration preflight, forcing an
    all-blocks mass-regen decision. This is the safe automation of the manual
    per-block resync: a block is refreshed ONLY when its ``best_result`` is
    ``sim_passed`` AND the on-disk RTL still hashes to the recorded
    ``rtl_sha1`` (the passing RTL is present, not overwritten). integration_dv
    + validation_dv (byte-exact) remain the backstop for any block wrongly
    refreshed. When ``changed_edge_substrings`` is given, a block that
    participates in an edge matching any substring is SKIPPED (its slice may
    have genuinely changed -- let it regenerate). Returns the refreshed block
    names. Never raises."""
    refreshed: list = []
    changed = list(changed_edge_substrings or [])
    for name in block_names:
        try:
            bd = Path(project_root) / ".coresmith" / "blocks" / name
            brj = bd / "best_result.json"
            if not brj.exists():
                continue
            best = json.loads(brj.read_text(encoding="utf-8"))
            if not best.get("sim_passed"):
                continue
            # passing RTL must still be on disk (hash intact)
            rec_rtl = best.get("rtl_sha1")
            if rec_rtl:
                # locate the block's rtl_target via best_result or skip the check
                rtl_rel = best.get("rtl_target") or ""
                if rtl_rel:
                    import hashlib as _hl
                    try:
                        cur = _hl.sha1(
                            (Path(project_root) / rtl_rel).read_bytes()
                        ).hexdigest()
                    except OSError:
                        cur = ""
                    if cur and cur != rec_rtl:
                        continue  # RTL was overwritten -- do NOT vouch for it
            # skip blocks touching a genuinely-changed edge
            if changed:
                try:
                    from orchestrator.langchain.agents.contract_lookup import (
                        load_block_contracts,
                    )
                    edges = load_block_contracts(str(project_root), name) or []
                    eids = [e.get("edge_id", "") if isinstance(e, dict) else str(e)
                            for e in edges]
                    if any(any(s in eid for s in changed) for eid in eids):
                        continue
                except Exception:  # noqa: BLE001
                    pass
            live = block_contract_sha1(str(project_root), name)
            if not live:
                continue
            (bd / "uarch_spec_contract_sha1").write_text(live, encoding="utf-8")
            best["contract_sha1"] = live
            brj.write_text(json.dumps(best, indent=2))
            refreshed.append(name)
        except (OSError, json.JSONDecodeError):
            continue
    return refreshed


def check_rtl_contract_ports(project_root, block_name: str,
                             rtl_path: str) -> list:
    """C15: deterministic RTL-ports-vs-frozen-contract check.

    Per-block DV proved hollow against a stale-width RTL: residual came out
    with s_axis_frame_job_tdata[33:0] against a 41-bit frozen contract (and
    942 cells against a ~52K-gate architecture estimate) yet passed its own
    TB 6/6 -- the TB asserted neither port widths nor the contract's NORMAL
    path, and the mismatch only surfaced at integration (the most expensive
    place). This check is TB-independent: parse the RTL's module ports and
    compare each contract edge's data port width against the frozen
    ``data_width_bits``. Naming per the interface-definition conventions:
    ``<port>_tdata`` for axi_stream, ``<port minus _srdy/_drdy>_data`` for
    srdy_drdy. Returns a list of error strings (empty = clean / no contract /
    unparseable RTL -- other gates own those cases).
    """
    errors: list = []
    advisories: list = []  # C22: name-not-found notes (non-fatal)
    try:
        from orchestrator.langchain.agents.contract_lookup import (
            load_block_contracts,
        )
        view = load_block_contracts(str(project_root), block_name)
        edges = view.get("edges") or []
        if not edges:
            return []
        # Lazy import: integration_helpers imports FROM this module at top
        # level, so importing it lazily here avoids the cycle.
        from orchestrator.langgraph.integration_helpers import (
            parse_verilog_ports,
        )
        mod = parse_verilog_ports(str(rtl_path))
        if not mod.name or not mod.ports:
            return []
        by_name = {p.name: p for p in mod.ports}
        # A shared consumer service may have several producer edges with legacy
        # payload widths and one widened canonical request port.  The integration
        # fabric owns the lossless zero-fill adapters on the narrower edges; the
        # consumer RTL must expose the widest record.  Group those fan-in edges
        # so the per-block gate does not demand mutually incompatible widths for
        # the same physical input port (the reference codec bitstream_reader: 11/26 bits).
        consumer_widths: dict[str, set[int]] = {}
        for e in edges:
            if e.get("role") != "consumer":
                continue
            base = str(e.get("consumer_port") or "").strip()
            width = e.get("data_width_bits")
            proto = str(e.get("handshake_protocol") or "").strip()
            if not base or not isinstance(width, int) or width <= 0:
                continue
            name = (base if base.endswith("_tdata") else f"{base}_tdata") \
                if proto == "axi_stream" else base
            consumer_widths.setdefault(name, set()).add(width)
        for e in edges:
            port_base = (e.get("producer_port")
                         if e.get("role") == "producer"
                         else e.get("consumer_port")) or ""
            port_base = str(port_base).strip()
            width = e.get("data_width_bits")
            if not port_base or not isinstance(width, int) or width <= 0:
                continue
            proto = str(e.get("handshake_protocol") or "").strip()
            # C22: the RTL data-port name IS the contract's producer/consumer
            # port name, VERBATIM. The RTL generator emits the contract port
            # name directly -- whether it's a bare native bundle
            # (`qspi_rx_internal`, `control_events`), a `_data`-suffixed payload
            # (`m_byte_request_data`), or an axi_stream base. Only axi_stream
            # carries a `_tdata` payload suffix the contract base omits. Earlier
            # revisions GUESSED a `_data` suffix (C15 doubled it -> C19; this
            # still appended one to bare native ports -> false-failing whole
            # lint-clean blocks). Verbatim matches every observed convention
            # (matmul `_data`, aes bare, the reference codec axi_stream) -- so "missing port"
            # now fires only on a genuinely absent port, and the width check
            # (the gate's real value: the reference codec's 34-vs-41 frame_job) still holds.
            if proto == "axi_stream":
                data_port = (port_base if port_base.endswith("_tdata")
                             else f"{port_base}_tdata")
            else:  # srdy_drdy (and any other): contract names the port verbatim
                data_port = port_base
            p = by_name.get(data_port)
            # C22: hard-fail ONLY on a genuine WIDTH MISMATCH -- a port present
            # under the contract's name but the WRONG width (the gate's real,
            # low-false-positive value: it caught the reference codec's residual
            # s_axis_frame_job_tdata[33:0] vs a 41-bit contract, which passed a
            # hollow per-block TB and only died at integration). A port NOT
            # FOUND by name is NOT a hard failure: RTL port naming for
            # wrappers/native bundles/pad groups legitimately diverges from the
            # contract endpoint name, so name-matching over-flags lint-clean
            # blocks (proven across a regression sweep: _data_data -> C19, bare
            # native ports -> here). A truly-missing port is still caught by the
            # testbench/sim (which references it) and integration. Record it as
            # advisory context only.
            if p is None:
                advisories.append(
                    f"contract port '{data_port}' not found by name in the RTL "
                    f"(edge {e.get('edge_id', '?')}); naming may differ -- "
                    f"width not verifiable here, TB/integration will confirm")
            elif (e.get("role") == "consumer"
                  and len(consumer_widths.get(data_port, set())) > 1
                  and p.width == max(consumer_widths[data_port])):
                # Narrow producer edges are adapted at integration into this
                # canonical widest consumer record; producer-side widths remain
                # checked independently on their own blocks.
                continue
            elif p.width != width:
                errors.append(
                    f"port '{data_port}' is [{p.msb}:{p.lsb}] ({p.width} "
                    f"bits) but the frozen contract requires {width} bits "
                    f"(edge {e.get('edge_id', '?')})")
    except Exception:  # noqa: BLE001 - deterministic gate must never crash
        return []
    return errors


def block_contract_sha1(project_root, block_name: str) -> str:
    """sha1 of THIS block's frozen interface-contract slice ('' on any error).

    Single source of truth for contract-provenance hashing (C5): used by the
    sim-pass provenance in pipeline_graph AND the block-model sidecar below.
    Hashing the block's OWN slice (not the whole interface_contracts.json)
    keeps a chip-lead edit to one block's contract from invalidating every
    other block's recorded provenance.
    """
    try:
        import hashlib

        from orchestrator.langchain.agents.contract_lookup import (
            load_block_contracts,
        )

        contract = load_block_contracts(str(project_root), block_name)
        # No edges -> this block has no contract participation; return "" so
        # the provenance axis is simply not recorded (load_block_contracts
        # returns a truthy empty view when the file is missing).
        if not contract or not contract.get("edges"):
            return ""
        return hashlib.sha1(
            json.dumps(contract, sort_keys=True).encode("utf-8")
        ).hexdigest()
    except Exception:  # noqa: BLE001 - provenance is best-effort
        return ""


async def _maybe_generate_block_golden(block: dict, callbacks: list = None) -> None:
    """Emit ``arch/block_models/<block>.py`` (Amaranth) when the flag is on.

    Resolves the reference implementation and this block's interface contract,
    then calls BlockGoldenGenerator (which emits an Amaranth Elaboratable).
    Best-effort: any failure is logged and swallowed (the model-integration gate
    is the hard gate).

    Skipped (model REUSED) when: the operator PINNED the on-disk model
    (``.coresmith/blocks/<b>/OPERATOR_MODEL_PIN`` -- protects hand-patches from
    being clobbered by regen; the OPERATOR_SPEC_PIN analog, same
    CORESMITH_IGNORE_SPEC_PINS escape), or the block is outside a
    gate-localized revise scope (see :func:`gate_scoped_reuse_reason`).
    """
    from orchestrator.architecture import composition as _composition

    if not _composition.block_goldens_enabled():
        return

    block_name = block["name"]

    _model_path = (
        PROJECT_ROOT / "arch" / _composition.BLOCK_MODELS_DIRNAME
        / f"{block_name}.py"
    )
    if _model_path.exists():
        _pin = (PROJECT_ROOT / ".coresmith" / "blocks" / block_name
                / "OPERATOR_MODEL_PIN")
        if _pin.exists() and os.environ.get(
            "CORESMITH_IGNORE_SPEC_PINS", ""
        ).strip() != "1":
            log(f"  [BLOCK-MODEL] {block_name}: OPERATOR_MODEL_PIN present -- "
                f"keeping the on-disk (hand-patched) model, SKIPPING regen",
                YELLOW)
            return
        _scope = gate_scoped_reuse_reason(str(PROJECT_ROOT), block_name)
        if _scope:
            # C5(b): the scoped-reuse shortcut only holds while the frozen
            # contract this model was generated against is unchanged. The
            # fragment_metadata_memory stale-oracle livelock: the contract
            # widened 48->56 bits but the on-disk model was reused, so DV
            # judged the (correct) new RTL against an obsolete oracle. PIN
            # still wins above (explicit operator intent). No sidecar recorded
            # (older runs) -> reuse as before.
            _sc_path = (PROJECT_ROOT / ".coresmith" / "blocks" / block_name
                        / "block_model_contract_sha1")
            _rec_ct = ""
            try:
                if _sc_path.exists():
                    _rec_ct = _sc_path.read_text(encoding="utf-8").strip()
            except OSError:
                pass
            _cur_ct = block_contract_sha1(str(PROJECT_ROOT), block_name)
            if _rec_ct and _cur_ct and _cur_ct != _rec_ct:
                log(f"  [BLOCK-MODEL] {block_name}: interface contract changed "
                    f"since this model was generated -- overriding scoped "
                    f"reuse, REGENERATING the model", YELLOW)
            else:
                log(f"  [BLOCK-MODEL] {block_name}: {_scope}", YELLOW)
                return
    try:
        from orchestrator.langchain.agents.block_golden_generator import (
            BlockGoldenGenerator,
        )
        from orchestrator.langchain.agents.contract_lookup import (
            load_block_contracts,
        )

        project_root = str(PROJECT_ROOT)

        # Generators need the FULL golden (per-block math), not the gate's
        # bytes-only wrapper -- use the generator-specific reference.
        ref_path = _composition.resolve_generator_reference(project_root)
        if not ref_path:
            log(
                f"  [BLOCK-MODEL] {block_name}: no reference implementation "
                f"found; skipping block model (gate will no-op)",
                YELLOW,
            )
            return
        try:
            reference_impl_source = Path(ref_path).read_text(encoding="utf-8")
        except OSError as exc:
            log(f"  [BLOCK-MODEL] {block_name}: cannot read {ref_path}: {exc}",
                YELLOW)
            return

        # C9: prefer the block's OWN golden slice (the architecture's
        # `python_source` ref, resolved by the C2/C3 method-aware slicer) over
        # the whole-chip golden. The hint-based slicing below
        # (resolve_block_slice_regions) is gated on BLOCK_ENTRY_HINTS, which is
        # codec-specific -- so on every other design each model generation saw the
        # ENTIRE golden: wrong scope for authoring one block's model, and a
        # real tractability tax on frontier blocks. Only narrows (never
        # widens); falls back to the whole file when the ref has no slice
        # suffix or fails to resolve.
        _ps_ref = _live_python_source_ref(block, PROJECT_ROOT)
        if ".py:" in (_ps_ref or ""):
            _sliced = resolve_python_source(_ps_ref, PROJECT_ROOT)
            if _sliced and len(_sliced) < len(reference_impl_source):
                log(f"  [BLOCK-MODEL] {block_name}: scoping the reference to "
                    f"the block's own golden slice ({len(_sliced)} of "
                    f"{len(reference_impl_source)} chars)", YELLOW)
                reference_impl_source = _sliced

        # This block's frozen interface contract slice + its ports.
        interface_contract = load_block_contracts(project_root, block_name)
        block_ports = block.get("interfaces", {}) or {}
        if not block_ports:
            # Fall back to the block-diagram interfaces on disk.
            import json as _json
            bd_path = PROJECT_ROOT / ".coresmith" / "block_diagram.json"
            if bd_path.exists():
                try:
                    bd = _json.loads(bd_path.read_text(encoding="utf-8"))
                    for b in bd.get("blocks", []):
                        if b.get("name") == block_name:
                            block_ports = b.get("interfaces", {}) or {}
                            break
                except (OSError, _json.JSONDecodeError):
                    pass

        out_path = (
            PROJECT_ROOT / "arch" / _composition.BLOCK_MODELS_DIRNAME
            / f"{block_name}.py"
        )

        # Authoritative, deterministic block->golden-slice mapping (line spans
        # already computed by the AST parse). Passed to the generator as a
        # focused hint and persisted as a .slice.json sidecar. Empty for an
        # unmapped block, in which case the generator sees the whole golden
        # (today's behaviour).
        slice_functions: list[str] = []
        slice_regions: list[dict] = []
        try:
            from orchestrator.langgraph.block_complexity import (
                resolve_block_slice_regions,
            )
            slice_functions, slice_regions = resolve_block_slice_regions(
                block_name, reference_impl_source
            )
            if slice_functions:
                log(f"  [BLOCK-MODEL] {block_name}: golden slice = "
                    f"{len(slice_functions)} fn(s): "
                    f"{', '.join(slice_functions[:8])}"
                    f"{' ...' if len(slice_functions) > 8 else ''}", YELLOW)
        except Exception as _exc:  # noqa: BLE001 - slice is advisory context
            log(f"  [BLOCK-MODEL] {block_name}: slice resolution skipped "
                f"({_exc})", YELLOW)

        agent = BlockGoldenGenerator(temperature=0.1)
        _gap_path = (PROJECT_ROOT / ".coresmith" / "blocks" / block_name
                     / "model_interface_gap.txt")

        # C10: generate -> detect gap -> RESOLVE from the committed corpus ->
        # freeze the answer into the contract -> regenerate, bounded. The gap
        # mechanism is non-terminating under regeneration when resolutions
        # only exist in spec prose (proven: 6 rounds, 6 different marginal
        # asks, and a regen even LOST a previously-earned resolution) -- the
        # resolver gives it MEMORY: every answered fact lands in
        # interface_contracts.json, the one document the generator reads.
        # A genuinely-new design decision still stops the loop and parks at
        # the C6 feasibility interrupt, now with the resolver's analysis.
        _rounds_cap = _gap_resolution_rounds_cap()
        _gap = ""
        for _round in range(_rounds_cap + 1):
            log(f"  [BLOCK-MODEL] Generating Amaranth block model for "
                f"{block_name}"
                + (f" (gap-resolution round {_round})" if _round else "")
                + "...", YELLOW)
            # C6: regenerating -- clear any prior gap declaration so the
            # marker always reflects THIS model (the pin/scope early-returns
            # above keep the old model AND its old marker, correctly).
            try:
                _gap_path.unlink(missing_ok=True)
            except OSError:
                pass
            await agent.generate(
                block_name=block_name,
                block_ports=block_ports,
                interface_contract=interface_contract,
                reference_impl_source=reference_impl_source,
                reference_impl_path=ref_path,
                project_root=project_root,
                output_path=str(out_path),
                slice_functions=slice_functions or None,
                slice_regions=slice_regions or None,
            )
            log(f"  [BLOCK-MODEL] Wrote {out_path}", GREEN)

            # C5(b): record WHICH contract this model was generated against,
            # so a later scoped-reuse skip can detect the frozen contract
            # moved and regenerate instead of reusing a stale oracle.
            try:
                _bd = PROJECT_ROOT / ".coresmith" / "blocks" / block_name
                _bd.mkdir(parents=True, exist_ok=True)
                _ct = block_contract_sha1(project_root, block_name)
                if _ct:
                    (_bd / "block_model_contract_sha1").write_text(
                        _ct, encoding="utf-8")
            except OSError:
                pass

            # C6: a model that declares it CANNOT realize the datapath from
            # the frozen interface is a model/spec feasibility CONFLICT.
            try:
                _model_text = Path(out_path).read_text(encoding="utf-8")
            except OSError:
                _model_text = ""
            _gap = detect_model_interface_gap(_model_text)
            if not _gap:
                break
            log(f"  [BLOCK-MODEL] {block_name}: model declares an INTERFACE "
                f"GAP -- {_gap}", RED)
            try:
                _gap_path.parent.mkdir(parents=True, exist_ok=True)
                _gap_path.write_text(_gap, encoding="utf-8")
            except OSError:
                pass
            if _round >= _rounds_cap or not _gap_resolution_enabled():
                break

            # C10: try to answer the gap from the committed corpus.
            try:
                from orchestrator.langchain.agents.gap_resolver import (
                    GapResolver,
                    apply_contract_amendments,
                    build_gap_corpus,
                )
                _corpus = build_gap_corpus(project_root, block_name, _gap)
                _verdict = await GapResolver().resolve(
                    block_name, _gap, _corpus)
            except Exception as _rexc:  # noqa: BLE001 - resolver best-effort
                log(f"  [GAP-RESOLVE] {block_name}: resolver errored "
                    f"({_rexc}) -- leaving the gap for the feasibility "
                    f"interrupt", RED)
                break
            if not _verdict.get("resolved"):
                log(f"  [GAP-RESOLVE] {block_name}: NOT resolvable from the "
                    f"committed corpus -- a real design decision is needed: "
                    f"{_verdict.get('unresolved_decision', '')}", RED)
                try:  # enrich the marker for the feasibility interrupt
                    _gap_path.write_text(
                        _gap + "\n\n[gap-resolver] " +
                        (_verdict.get("unresolved_decision") or "") +
                        ("\nRationale: " + _verdict.get("rationale", "")
                         if _verdict.get("rationale") else ""),
                        encoding="utf-8")
                except OSError:
                    pass
                break
            _cpath = PROJECT_ROOT / ".coresmith" / "interface_contracts.json"
            try:
                _cdoc = json.loads(_cpath.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as _cexc:
                log(f"  [GAP-RESOLVE] {block_name}: cannot load contracts "
                    f"({_cexc})", RED)
                break
            _cdoc, _applied = apply_contract_amendments(
                _cdoc, _verdict.get("amendments") or [])
            if not _applied:
                log(f"  [GAP-RESOLVE] {block_name}: resolver returned no "
                    f"applicable amendments -- leaving the gap for the "
                    f"feasibility interrupt", RED)
                break
            _cpath.write_text(json.dumps(_cdoc, indent=2), encoding="utf-8")
            try:  # audit trail -- every auto-frozen fact is reviewable
                with open(PROJECT_ROOT / ".coresmith" /
                          "gap_resolutions.jsonl", "a",
                          encoding="utf-8") as _gf:
                    _gf.write(json.dumps({
                        "ts": _time.time(), "block": block_name,
                        "round": _round, "gap": _gap,
                        "applied": _applied,
                        "rationale": _verdict.get("rationale", ""),
                    }) + "\n")
            except OSError:
                pass
            log(f"  [GAP-RESOLVE] {block_name}: RESOLVED from the committed "
                f"corpus -- froze {len(_applied)} amendment(s) into the "
                f"contract ({'; '.join(_applied[:4])}); regenerating", GREEN)
            log("  [GAP-RESOLVE] NOTE: contract amended -- partner blocks' "
                "recorded passes may be invalidated (C5/C7 catch this on "
                "their next entry)", YELLOW)
            # Reload the (now richer) contract slice for the next round.
            interface_contract = load_block_contracts(
                project_root, block_name)

        # Close the swallow (Phase 2C): probe the freshly generated model for
        # degeneracy and RECORD the verdict (never silently swallowed). Advisory
        # by default; hard-fails only under CORESMITH_GOLDEN_FEASIBILITY_GATE.
        try:
            from orchestrator.architecture.model_integration import (
                check_golden_feasibility,
                golden_feasibility_gate_enabled,
            )
            fr = check_golden_feasibility(project_root, block_name)
            if fr.get("ran") and not fr.get("passed"):
                gate_on = golden_feasibility_gate_enabled()
                log(f"  [BLOCK-MODEL] {block_name}: golden-feasibility "
                    f"{'FAIL (gate on)' if gate_on else 'FAIL (advisory)'}: "
                    f"{fr.get('reason')}", RED)
                if gate_on:
                    (PROJECT_ROOT / ".coresmith" / "blocks" / block_name
                     / "golden_feasibility_failed").write_text(
                        fr.get("reason", "degenerate golden"))
            elif fr.get("verdict") == "pass":
                _reach = (fr.get("checks", {})
                          .get("slice_reachability", {}) or {})
                log(f"  [BLOCK-MODEL] {block_name}: golden-feasibility PASS -- "
                    f"the block's golden slice is REACHED by the reference "
                    f"({_reach.get('reason') or 'slice exercised'})", GREEN)
            elif fr.get("ran"):
                # NOT RUN, reported the way the gate-sim gate reports it: full
                # reason, no green. The old line printed "OK (skipped)" in
                # GREEN -- an OK whose own parenthetical said the discriminating
                # check had not run. Every block of the first hands-off run got
                # that line.
                log(f"  [BLOCK-MODEL] {block_name}: golden-feasibility NOT RUN "
                    f"-- {fr.get('not_run_reason') or 'no discriminating check concluded'}",
                    YELLOW)
        except Exception as _fexc:  # noqa: BLE001 - probe is best-effort
            log(f"  [BLOCK-MODEL] {block_name}: feasibility probe skipped "
                f"({_fexc})", YELLOW)
    except Exception as exc:  # noqa: BLE001 - generation itself failed
        # No longer a silent swallow: record the failure so it is VISIBLE (the
        # model-integration gate is still the hard backstop downstream).
        log(
            f"  [BLOCK-MODEL] {block_name}: generation FAILED ({exc}); recorded. "
            f"The model-integration gate will flag any resulting divergence",
            RED,
        )
        try:
            _bd = PROJECT_ROOT / ".coresmith" / "blocks" / block_name
            _bd.mkdir(parents=True, exist_ok=True)
            (_bd / "block_golden_generation_failed.txt").write_text(str(exc))
        except OSError:
            pass
        # C11 backstop: generate() writes the model to disk BEFORE validating,
        # so a validation failure can discard an otherwise-substantive model
        # that DECLARES an interface gap (observed: an 82KB near-implementation
        # model rejected for its clock-port name -- the gap marker was never
        # written, so neither the C6 interrupt nor the C10 resolver ever saw
        # the model's claim). Even on a failed generation, surface a declared
        # gap so the feasibility gate engages instead of silently proceeding.
        try:
            _failed_model = (
                PROJECT_ROOT / "arch" / _composition.BLOCK_MODELS_DIRNAME
                / f"{block_name}.py"
            )
            if _failed_model.exists():
                _fgap = detect_model_interface_gap(
                    _failed_model.read_text(encoding="utf-8"))
                if _fgap:
                    _gp = (PROJECT_ROOT / ".coresmith" / "blocks" / block_name
                           / "model_interface_gap.txt")
                    _gp.parent.mkdir(parents=True, exist_ok=True)
                    _gp.write_text(
                        _fgap + f"\n\n[generation-failed] {exc}",
                        encoding="utf-8")
                    log(f"  [BLOCK-MODEL] {block_name}: failed-generation "
                        f"model still DECLARES a gap -- recorded for the "
                        f"feasibility gate: {_fgap}", RED)
        except Exception:  # noqa: BLE001 - backstop is best-effort
            pass


def _recover_codex_call_artifact(
    project_root: Path, rel_path: Path, min_mtime: float = 0.0
) -> str:
    """Return newest matching artifact written in a Codex isolated workdir.

    ``min_mtime`` (C13): only artifacts written at/after this timestamp are
    eligible. codex-call-* dirs from PREVIOUS calls persist on disk, and an
    unfiltered newest-first glob resurrected a prior call's spec as if this
    call produced it (a re-spec after a contract amendment "returned" a
    byte-identical pre-amendment spec).
    """
    try:
        candidates = sorted(
            project_root.glob(f"codex-call-*/{rel_path.as_posix()}"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return ""

    for candidate in candidates:
        try:
            if min_mtime and candidate.stat().st_mtime < min_mtime:
                continue
            text = candidate.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if text:
            return text
    return ""


def _looks_like_uarch_markdown(text: str) -> bool:
    lowered = text.lower()
    return (
        len(text) >= 512
        and "interface" in lowered
        and ("##" in text or "| port" in lowered or "module " in lowered)
    )


def _choose_generated_markdown(candidates: list[str]) -> str:
    """Pick the richest generated Markdown artifact, avoiding path messages."""
    non_empty = [c.strip() for c in candidates if c and c.strip()]
    if not non_empty:
        return ""
    rich = [c for c in non_empty if _looks_like_uarch_markdown(c)]
    return max(rich or non_empty, key=len)


# ---------------------------------------------------------------------------
# RTL Generation
# ---------------------------------------------------------------------------

async def generate_rtl(
    block: dict, attempt: int,
    callbacks: list = None,
) -> dict:
    """Generate Verilog RTL -- disk-first, agent reads/writes all files.

    The agent reads the uArch spec, ERS, constraints, golden model, and
    previous error from disk, and writes the Verilog to block["rtl_target"].
    """
    from orchestrator.langchain.agents.coresmith_llm import DEFAULT_MODEL
    from orchestrator.langchain.agents.rtl_generator import RTLGeneratorAgent

    rtl_path = PROJECT_ROOT / block["rtl_target"]
    rtl_path.parent.mkdir(parents=True, exist_ok=True)

    agent = RTLGeneratorAgent(model=DEFAULT_MODEL, temperature=0.1)
    # Step 1 of the microarchitecture restructure: prefer the per-block Amaranth
    # hardware golden as the reference the RTL lowers (when enabled), so the RTL
    # inherits the hardware-lowering decisions instead of re-deriving them from
    # the float reference golden. Falls back to python_source otherwise.
    _ref_src, _ref_is_hw = rtl_reference_source(block)
    try:
        result = await agent.generate(
            block_name=block["name"],
            description=block.get("description", ""),
            attempt=attempt,
            rtl_target=block["rtl_target"],
            python_source_path=_ref_src,
            reference_is_hw_golden=_ref_is_hw,
            project_root=str(PROJECT_ROOT),
            callbacks=callbacks,
        )
    except (ValueError, Exception) as e:
        log(f"  [RTL-GEN] Error: {e}", RED)
        return {"error": str(e)}

    # Postcondition: the agent must have materialized RTL at the canonical
    # path with a real Verilog module. Codex in particular has a strong
    # bias to write into its isolated codex-call-*/ scratch workdir and
    # then return a JSON status that points at that scratch path rather
    # than copying to rtl/. Failing fast here -- with a specific error
    # the retry prompt can act on -- is cheaper than letting lint discover
    # an empty or stub file.
    postcond = _assert_rtl_materialized(rtl_path, block["name"])
    if postcond is not None:
        log(f"  [RTL-GEN] postcondition failed: {postcond}", RED)
        return {"error": postcond, "postcondition_failed": True}

    return result


def rtl_module_name(rtl_path: str | Path, block_name: str) -> str:
    """Resolve the Verilog module name to drive for ``block_name``.

    **A block's architectural name is not always its RTL module name.** When a
    module name is fixed by an external contract -- a Caravel
    ``user_project_wrapper``, a vendor-locked top, a pad ring -- the
    architecture keeps its own block identifier while ``rtl_target`` carries
    the mandated name. Every ordinary block has
    ``rtl_target == <...>/<block_name>.v``, so the two coincide and this
    returns ``block_name`` unchanged.

    Resolution is evidence-based: whichever candidate the file actually
    declares wins. Preferring ``block_name`` keeps existing behaviour intact
    for every normal block; falling back to the ``rtl_target`` stem is safe
    because that path comes from the block spec, which the engine controls --
    an agent cannot use it to smuggle in an arbitrarily-named module.

    Conflating the two caused a correct, lint-clean block to fail twice in one
    run: first the RTL-generation postcondition rejected it, then the cocotb
    Makefile set ``TOPLEVEL`` to the block name and Verilator's
    ``--top-module`` found no such module, so simulation never elaborated and
    produced no VCD -- reported as a simulation failure rather than a
    configuration error.
    """
    p = Path(rtl_path)
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return block_name
    if f"module {block_name}" in text:
        return block_name
    if p.stem != block_name and f"module {p.stem}" in text:
        return p.stem
    return block_name


def _assert_rtl_materialized(rtl_path: Path, block_name: str) -> str | None:
    """Return None if rtl_path contains a real Verilog module; otherwise
    return a one-line diagnostic explaining what's wrong.

    Three checks: file exists; file is non-trivially-sized (manylinux empty
    file is 0 bytes, an `include "<path>"` redirect stub is typically
    < 200 bytes); and the file declares a module whose name is one we expect,
    so we know the LLM didn't write the wrong block's RTL or a placeholder.

    **An architectural block name is not always its Verilog module name.** A
    block whose RTL module name is fixed by an external contract -- a Caravel
    ``user_project_wrapper``, a vendor-locked top, a pad ring -- is declared in
    the architecture with its own block name while ``rtl_target`` carries the
    mandated file/module name. Requiring ``module <block_name>`` rejected such
    a block even though its RTL was correct and lint-clean, stopping the flow
    before testbench generation.

    So the expected name is ``block_name`` OR the stem of ``rtl_target``. That
    keeps the check strong: ``rtl_target`` comes from the block spec, which the
    engine controls, not from the agent -- so this cannot be used to smuggle in
    an arbitrarily-named module. Every ordinary block has
    ``rtl_target == <...>/<block_name>.v``, making the two identical.
    """
    if not rtl_path.exists():
        return (
            f"agent did not write {rtl_path}. The RTL likely lives in a "
            f"codex-call-*/ scratch workdir. Materialize it at "
            f"{rtl_path} via your file-write tool before returning."
        )
    size = rtl_path.stat().st_size
    if size < 200:
        return (
            f"{rtl_path} exists but is only {size} bytes -- likely a stub "
            f"or `include` redirect, not real RTL. Write the full module "
            f"body inline."
        )
    try:
        text = rtl_path.read_text(encoding="utf-8")
    except OSError as e:
        return f"{rtl_path}: read failed ({e})"
    expected = {block_name, rtl_path.stem}  # see rtl_module_name()
    if not any(f"module {name}" in text for name in expected):
        want = "` or `module ".join(sorted(expected))
        return (
            f"{rtl_path} ({size} bytes) does not contain `module {want}`. "
            f"The agent likely wrote the wrong module name or a different "
            f"block's RTL. The module declaration must use either the block "
            f"name ({block_name}) or the rtl_target file stem "
            f"({rtl_path.stem}) when the module name is fixed by an external "
            f"contract."
        )
    return None


# ---------------------------------------------------------------------------
# Lint
# ---------------------------------------------------------------------------

def lint_rtl(rtl_path: str, block_name: str, attempt: int = 1) -> dict:
    """Run Verilator lint on a Verilog file (read-only, no file mutation).

    Uses -Wno-fatal so style warnings (unused signals, EOF newline, etc.)
    don't block the pipeline.  Real errors (%Error) still cause failure.
    """
    rtl_text = Path(rtl_path).read_text() if Path(rtl_path).exists() else ""

    cmd = [
        "verilator", "--lint-only", "-Wall", "-Wno-fatal",
        "-Wno-EOFNEWLINE",
    ]
    if block_name == "viterbi_decoder":
        cmd.append("-Wno-BLKSEQ")
    # If the block instantiates the generic SRAM wrapper, lint it together with
    # the shared wrapper lib (behavioral view) so `cs_sram_*` resolves.
    try:
        from orchestrator.langgraph.sram_wrapper import (
            uses_wrapper as _uses_wrapper,
        )
        from orchestrator.langgraph.sram_wrapper import (
            wrapper_lib_path as _wrapper_lib_path,
        )
        if _uses_wrapper(rtl_text):
            cmd += ["--top-module", rtl_module_name(rtl_path, block_name),
                    _wrapper_lib_path()]
    except Exception:
        pass
    cmd.append(rtl_path)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=scaled(60))
        log_path = _write_step_log(block_name, "lint", cmd, result, attempt)
        stderr = result.stderr.strip()
        has_errors = "%Error" in stderr
        verilator_clean = (result.returncode == 0 and not has_errors)

        # Hard memory-wrapper gate: a large raw `reg [..] mem [..]` array that
        # should be an SRAM macro is a synthesizability defect (it either flops
        # to a giant array or hides as a $mem the backend can't place blindly).
        # Block it at lint so the fix happens before sim/synth. Disable with
        # CORESMITH_SRAM_GATE=0. Best-effort: never let the check break lint.
        mem_ok, mem_reasons = True, []
        if os.environ.get("CORESMITH_SRAM_GATE", "1") not in ("0", "false", "False"):
            try:
                from orchestrator.langgraph.sram_wrapper import gate_memory_wrapping
                mem_ok, mem_reasons = gate_memory_wrapping(rtl_text)
            except Exception:
                mem_ok, mem_reasons = True, []

        if verilator_clean and mem_ok:
            return {"clean": True, "warnings": stderr, "log_path": log_path}
        errs = stderr[-2000:]
        if not mem_ok:
            errs = (
                errs
                + "\n\n%Error: non-synthesizable memory -- must use the cs_sram "
                "wrapper (behavioral in sim, OpenRAM/sky130 SRAM macro in "
                "synth/backend):\n  - " + "\n  - ".join(mem_reasons)
            ).strip()
        return {"clean": False, "errors": errs, "log_path": log_path}
    except subprocess.TimeoutExpired:
        log_path = _write_step_log_error(block_name, "lint", cmd, "Verilator lint timed out", attempt)
        return {"clean": False, "errors": "Verilator lint timed out", "log_path": log_path}
    except FileNotFoundError:
        log_path = _write_step_log_error(block_name, "lint", cmd, "Verilator not installed", attempt)
        return {"clean": False, "errors": "Verilator not installed", "log_path": log_path}


# ---------------------------------------------------------------------------
# Testbench Generation
# ---------------------------------------------------------------------------

async def generate_testbench(
    block: dict,
    callbacks: list = None,
) -> dict:
    """Generate cocotb testbench -- disk-first, agent reads/writes all files."""
    from orchestrator.langchain.agents.coresmith_llm import DEFAULT_MODEL
    from orchestrator.langchain.agents.testbench_generator import TestbenchGeneratorAgent

    rtl_path = str(PROJECT_ROOT / block["rtl_target"])
    tb_path_str = str(PROJECT_ROOT / block["testbench"])
    Path(tb_path_str).parent.mkdir(parents=True, exist_ok=True)

    # When the block-goldens feature is on AND this block has a block-level
    # golden model on disk, pass it so the testbench uses it as the per-block
    # oracle. Flag off (or no golden file) => unchanged behavior.
    block_golden_path = ""
    try:
        from orchestrator.architecture import composition as _composition

        if _composition.block_goldens_enabled():
            bg = (
                PROJECT_ROOT / "arch" / _composition.BLOCK_MODELS_DIRNAME
                / f"{block['name']}.py"
            )
            if bg.exists():
                block_golden_path = str(bg)
    except Exception:  # noqa: BLE001 - never let this break TB generation
        block_golden_path = ""

    agent = TestbenchGeneratorAgent(model=DEFAULT_MODEL, temperature=0.1)
    result = await agent.generate(
        block_name=block["name"],
        rtl_path=rtl_path,
        python_source_path=block.get("python_source", ""),
        testbench_path=tb_path_str,
        project_root=str(PROJECT_ROOT),
        callbacks=callbacks,
        block_golden_path=block_golden_path,
    )

    # Postcondition: cocotb TB must exist with at least one @cocotb.test().
    # Same Codex-writes-to-scratch failure mode as generate_rtl; surface it
    # early instead of waiting for simulate to discover an empty TB.
    postcond = _assert_testbench_materialized(Path(tb_path_str), block["name"])
    if postcond is not None:
        log(f"  [TB-GEN] postcondition failed: {postcond}", RED)
        return {"error": postcond, "postcondition_failed": True}

    return result


def _assert_testbench_materialized(tb_path: Path, block_name: str) -> str | None:
    """Return None if tb_path is a usable cocotb file; else a one-line reason."""
    if not tb_path.exists():
        return (
            f"agent did not write {tb_path}. Likely written to a codex-call-*/ "
            f"scratch dir. Materialize at {tb_path}."
        )
    size = tb_path.stat().st_size
    if size < 200:
        return f"{tb_path} exists but is only {size} bytes -- likely a stub."
    try:
        text = tb_path.read_text(encoding="utf-8")
    except OSError as e:
        return f"{tb_path}: read failed ({e})"
    if "@cocotb.test" not in text:
        return (
            f"{tb_path} ({size} bytes) has no @cocotb.test() decorator. "
            f"The cocotb runner will report zero testcases. Write at least "
            f"one @cocotb.test() async function."
        )
    return None


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def run_wavekit_vcd_audit(vcd_path: Path, audit_path: Path, clock_hint: str = "clk") -> dict:
    """Inspect a Verilator VCD with WaveKit and persist a small audit report."""
    if not vcd_path.exists() or vcd_path.stat().st_size == 0:
        result = {
            "ok": False,
            "error": f"missing or empty VCD: {vcd_path}",
            "vcd_path": str(vcd_path),
        }
        audit_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    def has_wavekit(python: str) -> bool:
        check = subprocess.run(
            [python, "-c", "import wavekit"],
            capture_output=True,
            text=True,
            timeout=scaled(30),
        )
        return check.returncode == 0

    def wavekit_python() -> str:
        if has_wavekit(sys.executable):
            return sys.executable

        venv_dir = PROJECT_ROOT / ".coresmith" / "tools" / "wavekit-venv"
        python = venv_dir / "bin" / "python"
        if not python.exists():
            subprocess.run(
                [sys.executable, "-m", "venv", str(venv_dir)],
                capture_output=True,
                text=True,
                timeout=scaled(120),
                check=True,
            )
        if not has_wavekit(str(python)):
            subprocess.run(
                [str(python), "-m", "pip", "install", "-q", "wavekit>=0.5.6"],
                capture_output=True,
                text=True,
                timeout=scaled(300),
                check=True,
            )
        return str(python)

    script = r"""
import json
import sys
from pathlib import Path

from wavekit import VcdReader

vcd_path = Path(sys.argv[1])
clock_hint = sys.argv[2]

with VcdReader(str(vcd_path)) as reader:
    top_scopes = reader.top_scope_list()
    signals = []
    clocks = []

    def walk(scope):
        for sig in getattr(scope, "signal_list", []):
            name = sig.full_name
            signals.append({"name": name, "width": int(sig.width)})
            base = name.split(".")[-1].split("[")[0]
            if base in {clock_hint, "clk", "clock", "i_clk"}:
                clocks.append(name)
        for child in getattr(scope, "child_scope_list", []):
            walk(child)

    for top in top_scopes:
        walk(top)

    if not signals:
        raise RuntimeError("VCD contains no signals")
    if int(reader.end_time) <= int(reader.begin_time):
        raise RuntimeError(
            f"VCD contains no value-change time range: begin={reader.begin_time} end={reader.end_time}"
        )

    report = {
        "ok": True,
        "vcd_path": str(vcd_path),
        "begin_time": int(reader.begin_time),
        "end_time": int(reader.end_time),
        "signal_count": len(signals),
        "sample_signals": signals[:64],
        "clock_candidates": clocks[:16],
    }
    print(json.dumps(report))
"""
    try:
        audit_python = wavekit_python()
        proc = subprocess.run(
            [audit_python, "-c", script, str(vcd_path), clock_hint],
            capture_output=True,
            text=True,
            timeout=scaled(180),
        )
    except subprocess.CalledProcessError as exc:
        # WaveKit could not be SET UP (no prebuilt wheel for this arch + missing
        # native build deps like python3-dev/cmake, etc.). The WaveKit VCD audit
        # is a *supplementary* analysis layered on top of the cocotb regression
        # result -- a missing optional tool must NOT masquerade as a DV failure
        # (that produced a spurious DV_PROCESS_ERROR on arm64 workers lacking
        # build deps). Skip gracefully so DV is decided by the cocotb pass/fail.
        result = {
            "ok": True,
            "skipped": True,
            "reason": (
                "WaveKit unavailable; VCD audit skipped -- DV relies on cocotb "
                "results. Install WaveKit (needs python3-dev + cmake to build "
                "pylibfst from sdist on platforms without a prebuilt wheel)."
            ),
            "detail": (exc.stderr or exc.stdout or str(exc))[-1000:],
            "vcd_path": str(vcd_path),
        }
        audit_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    except subprocess.TimeoutExpired:
        result = {
            "ok": False,
            "error": "WaveKit VCD audit timed out",
            "vcd_path": str(vcd_path),
        }
        audit_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    if proc.returncode != 0:
        result = {
            "ok": False,
            "error": (proc.stderr or proc.stdout)[-2000:],
            "vcd_path": str(vcd_path),
        }
    else:
        result = json.loads(proc.stdout)
    audit_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _build_products_present(sim_dir: Path) -> bool:
    """True when ``sim_dir`` holds a prior Verilator/cocotb BUILD.

    Distinguishes real build products -- the cocotb obj dir (``sim_build/``), a
    ``V*`` sim binary, or a ``results.xml`` -- from mere config inputs (Makefile,
    the copied TB, ``.build_fingerprint``, ``wavekit_audit.json``, the flock).
    Cheap: a couple of stat/glob calls, so the no-products first-call fast path
    stays inexpensive. Best-effort (any error -> False)."""
    try:
        obj = sim_dir / "sim_build"
        if obj.is_dir():
            try:
                if any(obj.iterdir()):
                    return True
            except OSError:
                pass
        if (sim_dir / "results.xml").exists():
            return True
        for base in (obj, sim_dir):
            try:
                if any(base.glob("V*")):
                    return True
            except OSError:
                continue
    except OSError:
        return False
    return False


def clear_build_products(sim_dir: Path) -> bool:
    """Wipe cocotb/Verilator BUILD products under ``sim_dir`` (obj dir + outputs).

    Removes the ``sim_build/`` obj dir and the ``dump.vcd`` / ``dump.fst`` /
    ``results.xml`` outputs, leaving config inputs (Makefile, TB, fingerprint)
    intact so the next ``make`` does a clean rebuild with the current flags.
    Best-effort; returns True when it removed something."""
    removed = False
    try:
        obj = sim_dir / "sim_build"
        if obj.is_dir():
            shutil.rmtree(obj, ignore_errors=True)
            removed = True
        for out in ("dump.vcd", "dump.fst", "results.xml"):
            try:
                (sim_dir / out).unlink()
                removed = True
            except OSError:
                pass
    except Exception:  # noqa: BLE001
        pass
    return removed


def apply_build_fingerprint(
    sim_dir: Path,
    makefile_content: str,
    sources: list[str] | None = None,
) -> bool:
    """Clear a stale Verilator/cocotb build when its compile inputs changed.

    Verilator bakes compile-time flags -- notably ``VM_TRACE`` (from
    ``--trace``) -- into the compiled ``Vtop`` binary at BUILD time, but cocotb's
    ``Makefile.sim`` only rebuilds when a *source* is newer than that binary; it
    does NOT rebuild when ``EXTRA_ARGS`` / ``WAVES`` change. So a run that flips
    tracing (or swaps the source list) after an earlier build silently REUSES the
    stale binary (the ``Vtop`` mtime predates the regenerated Makefile) -- the
    2026-07-02 case where post-env-fix integration runs still used a traceless
    binary and the mandatory VCD audit fail-closed on a phantom-missing dump.vcd.

    We fingerprint the build inputs -- the full Makefile text (which embeds
    ``EXTRA_ARGS``, ``WAVES``, ``TOPLEVEL`` and the ``VERILOG_SOURCES`` list) plus
    the bytes of every source file -- into ``<sim_dir>/.build_fingerprint``. On a
    mismatch we wipe the stale build products (cocotb's ``sim_build/`` obj dir and
    the ``dump.vcd`` / ``results.xml`` outputs) so the next ``make`` does a clean
    rebuild with the new flags. Returns True when a stale build was cleared.
    Best-effort: any error leaves the tree untouched (make's own mtime logic still
    applies) so this can never wedge a build.
    """
    import hashlib

    try:
        sim_dir = Path(sim_dir)
        h = hashlib.sha256()
        h.update(makefile_content.encode("utf-8", "replace"))
        for src in sources or []:
            try:
                p = Path(src)
                if p.is_file():
                    h.update(b"\0src\0")
                    h.update(str(src).encode("utf-8", "replace"))
                    h.update(p.read_bytes())
            except OSError:
                continue
        new_fp = h.hexdigest()

        fp_path = sim_dir / ".build_fingerprint"
        old_fp = None
        try:
            if fp_path.exists():
                old_fp = fp_path.read_text(encoding="utf-8").strip()
        except OSError:
            old_fp = None

        cleared = False
        if old_fp is not None and old_fp != new_fp:
            # Inputs changed since the last build -> nuke the stale products so
            # the compile-time flags actually take effect on the next make.
            clear_build_products(sim_dir)
            cleared = True
        elif old_fp is None and _build_products_present(sim_dir):
            # First fingerprint next to a PRE-EXISTING (unfingerprinted) build:
            # those products were laid down by something OUTSIDE the engine's
            # build -- e.g. a TB-generation agent's in-context `verify` running
            # the cocotb make itself in this dir before the engine wrote its
            # traced Makefile. An unfingerprinted build is by definition
            # untrusted (cocotb's make would REUSE a stale/traceless Vtop whose
            # mtime predates our Makefile -> no dump.vcd -> the mandatory WaveKit
            # audit fail-closes a passing sim; the 2026-07-02 integration-DV
            # failure). Wipe so the next make rebuilds cleanly. The no-products
            # first call (per-block fast path) skips this and stays cheap.
            clear_build_products(sim_dir)
            cleared = True

        try:
            sim_dir.mkdir(parents=True, exist_ok=True)
            fp_path.write_text(new_fp, encoding="utf-8")
        except OSError:
            pass
        return cleared
    except Exception:  # noqa: BLE001
        return False


_COCOTB_SUMMARY_RE = re.compile(
    r"TESTS=(?P<tests>\d+)\s+PASS=(?P<passed>\d+)\s+FAIL=(?P<failed>\d+)",
    re.IGNORECASE,
)


def _parse_cocotb_summary(output: str) -> dict:
    """Extract cocotb regression counts from stdout/stderr."""
    matches = list(_COCOTB_SUMMARY_RE.finditer(output or ""))
    if not matches:
        return {"found": False, "tests_total": 0, "tests_passed": 0, "tests_failed": 0}
    match = matches[-1]
    return {
        "found": True,
        "tests_total": int(match.group("tests")),
        "tests_passed": int(match.group("passed")),
        "tests_failed": int(match.group("failed")),
    }


def _cocotb_timing_keyword_mode() -> str:
    """Which timing keyword(s) the installed cocotb's ``Timer`` accepts:
    ``"both"`` (cocotb 2.x: ``unit`` preferred + ``units`` deprecated alias),
    ``"units"`` (cocotb 1.x: plural only), or ``"unit"`` (singular only).

    C4 (exp-reference_codec-20260713): the old predicate was ``"units" in parameters``,
    which misfired on cocotb 2.x (BOTH spellings present) and rewrote the
    generated TB's correct ``unit=`` to the deprecated ``units=`` -- ON THE
    STAGED COPY ONLY. The TB's ``_assert_staged_source_current`` SHA-256
    staged-vs-canonical self-check then mismatched and aborted every sim at
    t=0 before a single functional test ran. When both spellings are accepted
    the normalizer must be a strict NO-OP so staged stays byte-identical to
    canonical."""
    try:
        from cocotb.triggers import Timer

        params = inspect.signature(Timer).parameters
        has_unit, has_units = "unit" in params, "units" in params
        if has_unit and has_units:
            return "both"
        if has_units:
            return "units"
        return "unit"
    except Exception:
        return "both"  # unknown -> don't touch the TB


def _normalize_cocotb_timing_keywords(tb_file: Path) -> None:
    """Make generated cocotb timing calls match the installed cocotb API.
    NO-OP when the installed cocotb accepts both spellings (cocotb 2.x) --
    rewriting would break the TB's staged-vs-canonical integrity self-check."""
    mode = _cocotb_timing_keyword_mode()
    if mode == "both":
        return
    try:
        text = tb_file.read_text(encoding="utf-8")
    except OSError:
        return
    if mode == "units":
        normalized = re.sub(r"(?<!\w)unit\s*=", "units=", text)
    else:
        normalized = re.sub(r"(?<!\w)units\s*=", "unit=", text)
    if normalized != text:
        tb_file.write_text(normalized, encoding="utf-8")


def run_simulation(block: dict, rtl_path, tb_path: str, attempt: int = 1,
                   extra_defines: list | None = None,
                   sim_subdir: str | None = None,
                   extra_args: list | None = None) -> dict:
    """Run cocotb simulation with Verilator.

    ``extra_defines`` (e.g. ``["SYNTHESIS"]``) are added as Verilator ``-D``
    preprocessor defines, and ``sim_subdir`` overrides the ``sim_build/<name>``
    directory -- both used by the branch-parity smoke (harness.branch_parity) to
    build the SAME RTL under the synth-side macro world in an isolated dir.
    ``extra_args`` appends raw Verilator flags (e.g. ``["--trace-depth 1"]``),
    used by the gate-sim harness (harness.gate_sim) to record a PORT-ONLY
    waveform for post-synthesis vector replay. With all three omitted the
    Makefile and build dir are byte-identical to the default.
    """
    block_name = block["name"]
    sim_dir = PROJECT_ROOT / "sim_build" / (sim_subdir or block_name)
    sim_dir.mkdir(parents=True, exist_ok=True)

    # Bound Verilator's C++ build parallelism (engine fix 2026-06-24). A huge
    # block (e.g. the consolidated full-RD intra_rd_encode_core) Verilates into
    # many translation units; an unbounded parallel compile fans out into a
    # large number of g++/cc1plus processes. Combined with the orphaned-on-
    # timeout builds the killpg fix below now prevents, that fork-bombed RAM and
    # OOM-killed the daemon. Cap it (override via CORESMITH_SIM_BUILD_JOBS).
    try:
        _build_jobs = max(1, int(os.environ.get("CORESMITH_SIM_BUILD_JOBS", "2") or "2"))
    except ValueError:
        _build_jobs = 2
    # ``rtl_path`` may be a single file OR a sequence of sources. An ASSEMBLED
    # top is not one file: it needs the integration top plus every block's RTL
    # plus the wrapper lib, and a lone path yields an empty/partial source list
    # ("No input Verilog file specified") and therefore NO waveform -- which is
    # why the chip_top gate-sim could only ever report not_run. A plain string
    # is normalized to a 1-element list, so " ".join() reproduces it exactly and
    # every existing caller's Makefile is byte-identical.
    _srcs = [rtl_path] if isinstance(rtl_path, str) else [str(p) for p in rtl_path]
    _srcs = [p for p in _srcs if p]
    # The FIRST source is the primary: it is the file whose declared module the
    # TOPLEVEL is resolved from. Callers passing a list must put the top first.
    _primary = _srcs[0] if _srcs else ""
    # Include the generic SRAM wrapper lib (behavioral view) when the design
    # instantiates it, so cocotb/verilator has the cs_sram_* modules.
    _verilog_sources = " ".join(_srcs)
    try:
        from orchestrator.langgraph.sram_wrapper import (
            uses_wrapper as _uses_wrapper,
        )
        from orchestrator.langgraph.sram_wrapper import (
            wrapper_lib_path as _wrapper_lib_path,
        )
        # Scan EVERY source, not just the primary: with an assembled top the
        # cs_sram instantiation lives in a leaf block, so looking only at the
        # top would miss it and the build would fail on a missing module.
        _rtl_text = "".join(
            Path(p).read_text(errors="replace") for p in _srcs if Path(p).exists()
        )
        _wlib = _wrapper_lib_path()
        if _uses_wrapper(_rtl_text) and _wlib not in _srcs:
            _verilog_sources = " ".join([_wlib] + _srcs)
    except Exception:
        pass
    # Coverage injection. Instrumentation is added when EITHER the explicit
    # coverage opt-in (CORESMITH_COVERAGE=1) OR the line-coverage floor gate
    # (CORESMITH_LINE_COV_GATE, default ON -- see harness.coverage) needs it,
    # so the gate can annotate coverage.dat after a passing DV run.
    from orchestrator.harness import coverage as _cov

    _cov_line = ""
    if _cov.coverage_enabled() or _cov.line_cov_gate_enabled():
        _cov_line = "EXTRA_ARGS += --coverage\n"

    # Branch-parity: force the DESIGN onto its synth-side `ifdef world (e.g.
    # -DSYNTHESIS) while the cs_* wrapper stays behavioral (it selects its body
    # by a `generate` on MEM_IMPL, NOT by this macro, so the memory model is
    # unaffected). Empty by default -> no line emitted -> byte-identical build.
    _def_line = ""
    for _d in (extra_defines or []):
        _dn = str(_d).strip()
        if _dn:
            _def_line += f"EXTRA_ARGS += -D{_dn}\n"
    # Raw Verilator flags (gate-sim reference run uses --trace-depth 1 so the
    # recorded waveform holds the top-level PORTS and little else). Empty by
    # default -> no line emitted -> byte-identical build.
    for _a in (extra_args or []):
        _an = str(_a).strip()
        if _an:
            _def_line += f"EXTRA_ARGS += {_an}\n"

    # TOPLEVEL must be the module Verilator will find with --top-module, which
    # is not always the architectural block name (locked Caravel/vendor tops).
    _toplevel = rtl_module_name(_primary, block_name)
    if _toplevel != block_name:
        log(f"  [SIM] TOPLEVEL={_toplevel} (block {block_name} declares an "
            f"externally-mandated module name)", CYAN)

    makefile_content = f"""
SIM = verilator
TOPLEVEL_LANG = verilog
VERILOG_SOURCES = {_verilog_sources}
TOPLEVEL = {_toplevel}
MODULE = test_{block_name}
WAVES = 1
EXTRA_ARGS += --trace --trace-structs
EXTRA_ARGS += --build-jobs {_build_jobs}
{_cov_line}{_def_line}include $(shell cocotb-config --makefiles)/Makefile.sim
"""
    # Clear a stale Verilator build if the flags/sources changed since last run
    # (compile-time VM_TRACE etc. would otherwise be baked into a reused Vtop).
    apply_build_fingerprint(sim_dir, makefile_content, _verilog_sources.split())
    (sim_dir / "Makefile").write_text(makefile_content)

    sim_tb_path = sim_dir / f"test_{block_name}.py"
    shutil.copy2(tb_path, sim_tb_path)
    _normalize_cocotb_timing_keywords(sim_tb_path)

    create_golden_model_wrapper(block_name, block.get("python_source", ""))

    wrapper_src = PROJECT_ROOT / "tb" / "cocotb" / f"{block_name}_model.py"
    if wrapper_src.exists():
        shutil.copy2(wrapper_src, sim_dir / f"{block_name}_model.py")

    env = os.environ.copy()
    import sys
    venv_bin = str(Path(sys.prefix) / "bin")
    env["PATH"] = f"{venv_bin}:{env.get('PATH', '/usr/bin:/bin')}"
    env["SHELL"] = shutil.which("bash") or "/bin/bash"
    env["PYTHONPATH"] = f"{sim_dir}:{PROJECT_ROOT}:{env.get('PYTHONPATH', '')}"

    # ANTI-MEMORIZATION DV SEED (engine fix, 2026-06-21).
    # Per-block DV stimulus must be UNPREDICTABLE at RTL-generation time, so a
    # data-transforming block cannot pass by memorizing a fixed (seed-pinned)
    # set of golden vectors keyed on metadata. We inject a fresh, high-entropy
    # seed into the simulation environment on EVERY run. The generated cocotb
    # testbench is required (see testbench_generator.md rule 16) to derive ALL
    # randomized stimulus (pixel/sample content AND geometry/QP scenario
    # selection) from this seed, computing expected outputs from the imported
    # golden model at runtime. Because the seed did not exist when the RTL was
    # generated, a finite stimulus-keyed LUT cannot cover it -- only a genuine
    # datapath implementation can match the model. Honor a caller-pinned seed
    # (CORESMITH_DV_SEED_PIN) for reproducible debugging when set.
    from orchestrator.harness.seed_provider import mint_dv_seed
    _dv_seed = mint_dv_seed(env)
    try:
        _write_step_log_error(
            block_name, "dv_seed",
            ["CORESMITH_DV_SEED", _dv_seed],
            f"per-run DV seed for {block_name}: {_dv_seed}", attempt,
        )
    except Exception:  # noqa: BLE001
        pass

    make_bin = shutil.which("make") or "make"

    # --- SIM TIMEOUT RESOLUTION (engine fix 2026-06-23) ---------------------
    # The per-block cocotb/Verilator sim is DISTINCT from the synth probe
    # (CORESMITH_SYNTH_TIMEOUT_S). A slow-but-correct full-datapath block
    # (e.g. intra_rd_encode_core) needs more than the old 10-min cap or its
    # diagnose->revise loop is starved: a TIMEOUT carries no PSNR/divergence
    # verdict, so codex has nothing to root-cause. We resolve the cap as the
    # MAX of three channels and auto-extend on repeated timeouts:
    #   1) env default            CORESMITH_SIM_TIMEOUT_S (default 900s / 15min)
    #   2) per-block declared     block["sim_timeout_s"] / ["runtime_target_s"]
    #                             / ["requested_sim_timeout_s"] (agent-set)
    #   3) auto-extend            x1.5 per prior TIMEOUT for THIS block,
    #                             capped at CORESMITH_SIM_TIMEOUT_CAP_S (1800s).
    _sim_to_cap = scaled(1800, env="CORESMITH_SIM_TIMEOUT_CAP_S")
    _sim_to_env = scaled(900, env="CORESMITH_SIM_TIMEOUT_S")
    _sim_to_decl = 0
    for _k in ("sim_timeout_s", "runtime_target_s", "requested_sim_timeout_s"):
        try:
            _v = int(float(block.get(_k) or 0))
        except (TypeError, ValueError):
            _v = 0
        _sim_to_decl = max(_sim_to_decl, _v)
    # Prior-timeout count for THIS block persists across attempts / daemon
    # restarts via a tiny state file under the block dir.
    _to_state = sim_dir / "sim_timeout_state.json"
    _prior_timeouts = 0
    try:
        if _to_state.exists():
            _prior_timeouts = int(json.loads(_to_state.read_text()).get("timeouts", 0))
    except Exception:  # noqa: BLE001
        _prior_timeouts = 0
    _sim_to_base = max(_sim_to_env, _sim_to_decl)
    _sim_timeout = _sim_to_base
    if _prior_timeouts > 0:
        _sim_timeout = int(_sim_to_base * (1.5 ** _prior_timeouts))
    _sim_timeout = min(_sim_timeout, _sim_to_cap)
    _sim_to_src = "default"
    if _sim_to_decl > _sim_to_env:
        _sim_to_src = "declared"
    if _prior_timeouts > 0:
        _sim_to_src = f"extended(x1.5^{_prior_timeouts})"
    print(f"[SIM] timeout={_sim_timeout}s ({_sim_to_src}; "
          f"env={_sim_to_env}s declared={_sim_to_decl}s "
          f"prior_timeouts={_prior_timeouts} cap={_sim_to_cap}s) "
          f"block={block_name}", flush=True)
    try:
        _write_step_log_error(
            block_name, "sim_timeout",
            ["CORESMITH_SIM_TIMEOUT_S", str(_sim_timeout)],
            f"resolved sim timeout for {block_name}: {_sim_timeout}s "
            f"(src={_sim_to_src} env={_sim_to_env} declared={_sim_to_decl} "
            f"prior_timeouts={_prior_timeouts} cap={_sim_to_cap})", attempt,
        )
    except Exception:  # noqa: BLE001
        pass

    import signal as _signal
    env["MAKEFLAGS"] = (env.get("MAKEFLAGS", "") + " -j2").strip()
    try:
        # Run the build+sim in its OWN process group (start_new_session) so a
        # timeout can kill the ENTIRE tree -- make -> verilator -> all the
        # parallel g++/cc1plus. subprocess.run()/proc.kill() SIGKILL only the
        # direct `make` child, orphaning the compilers (reparented to PID 1);
        # combined with the retry loop those orphans piled up into thousands of
        # processes -> OOM-killed the daemon (engine fix 2026-06-24).
        _proc = subprocess.Popen(
            [make_bin, "-C", str(sim_dir)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            start_new_session=True,
        )
        try:
            _out, _err = _proc.communicate(timeout=_sim_timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(_proc.pid), _signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            _proc.wait()
            raise
        result = subprocess.CompletedProcess(
            [make_bin, "-C", str(sim_dir)], _proc.returncode, _out, _err,
        )
        log_path = _write_step_log(block_name, "simulate", [make_bin, "-C", str(sim_dir)], result, attempt)
        full_output = result.stdout + "\n" + result.stderr
        output = full_output[-5000:]
        no_tests = "No tests were discovered" in output
        summary = _parse_cocotb_summary(full_output)
        if no_tests:
            output = (
                "COCOTB ERROR: No tests were discovered. Treating simulation "
                "as failed to prevent DV false pass.\n" + output
            )
        if summary["found"] and summary["tests_failed"]:
            output = (
                "COCOTB ERROR: Regression summary reports failing tests. "
                "Treating simulation as failed even if make returned 0.\n" + output
            )

        vcd_path = sim_dir / "dump.vcd"
        audit_path = sim_dir / "wavekit_audit.json"
        wavekit_audit = run_wavekit_vcd_audit(vcd_path, audit_path)
        passed = (
            result.returncode == 0
            and not no_tests
            and (
                not summary["found"]
                or (summary["tests_total"] > 0 and summary["tests_failed"] == 0)
            )
        )
        if not wavekit_audit.get("ok"):
            output = (
                "WAVEKIT VCD AUDIT WARNING: "
                f"{wavekit_audit.get('error', 'unknown error')}\n" + output
            )

        # --- LINE-COVERAGE FLOOR GATE (weak-TB rejector) --------------------
        # Only on the PRIMARY block-DV run (not the branch-parity smoke): a
        # DV pass whose TB exercises < floor% of the block's coverage points
        # is demoted to FAIL with the uncovered regions as feedback for the
        # TB-fix loop. REJECT-only: a pct above the floor never upgrades or
        # certifies anything (execution is not observation); a missing
        # coverage.dat / verilator_coverage skips the gate (verdict None).
        coverage_gate_failed = False
        coverage_pct = None
        # PERSISTENCE record: an ALWAYS-present coverage fact for the signoff
        # report -- {applicable, pct, floor, points_total, points_hit,
        # uncovered_count, passed} on the primary block-DV run, or
        # {applicable:False, reason} when coverage could not be measured (no
        # coverage.dat / verilator_coverage absent / gate off). Recorded on the
        # PRIMARY run (not the branch-parity smoke / -D define builds) regardless
        # of the DV verdict, so a run with no coverage is VISIBLE downstream
        # rather than silently blanked. NEVER fails a block for missing tooling.
        coverage_record = None
        if sim_subdir is None and not extra_defines:
            _na_reason = None
            try:
                from orchestrator.harness.coverage import (
                    coverage_na_reason as _na_reason,
                )
                from orchestrator.harness.coverage import (
                    line_cov_gate_verdict,
                )

                _verdict = line_cov_gate_verdict(sim_dir)
            except Exception:  # noqa: BLE001 - never fail DV on gate plumbing
                _verdict = None
            if _verdict is not None:
                coverage_pct = _verdict.get("pct")
                coverage_record = {
                    "applicable": True,
                    "pct": _verdict.get("pct"),
                    "floor": _verdict.get("floor"),
                    "points_total": _verdict.get("points_total"),
                    "points_hit": _verdict.get("points_hit"),
                    "uncovered_count": _verdict.get("uncovered_count"),
                    "passed": _verdict.get("passed"),
                }
                # REJECT-only gate: demote a PASSING DV whose TB is below floor.
                if passed and not _verdict["passed"]:
                    passed = False
                    coverage_gate_failed = True
                    output = _verdict["report"] + "\n\n" + output
                    # The TB fixer / diagnose agent reads the step log file:
                    # append the uncovered-region report there too.
                    try:
                        if log_path:
                            with open(log_path, "a", encoding="utf-8") as _lf:
                                _lf.write("\n" + _verdict["report"] + "\n")
                    except OSError:
                        pass
            else:
                # Coverage not measurable -- record a VISIBLE reason (never blank,
                # never a block failure) so a run with no coverage is auditable.
                try:
                    _reason = (_na_reason(sim_dir) if _na_reason
                               else "coverage unavailable")
                except Exception:  # noqa: BLE001
                    _reason = "coverage unavailable"
                coverage_record = {"applicable": False, "reason": _reason}

        # --- MEASURED-THROUGHPUT GATE (v3) ---------------------------------
        # After functional DV PASSES, MEASURE the block's actual cyc/op from the
        # TB's ``test_throughput_measure`` artifact and FAIL the block when the
        # measured rate exceeds the uArch-DECLARED §6.1 cyc/op x 1.1 -- the
        # plan->RTL drift the AES serial 21-cyc key schedule proved (plan
        # declared 11 word-parallel; nothing re-measured the RTL). A too-slow
        # RTL is an RTL PERFORMANCE defect (escalates to diagnose), NOT a TB
        # weakness -- unless the artifact is MISSING (declared a rate but the TB
        # emitted no measurement), which fails CLOSED and routes to TB
        # regeneration (``throughput_needs_tb``), mirroring the coverage gate.
        # Recorded on the PRIMARY run regardless of verdict; fail-open on any
        # tooling absence (never crashes a DV run). Only the block-DV verdict is
        # gated -- the branch-parity smoke / -D builds skip it.
        throughput_gate_failed = False
        throughput_needs_tb = False
        throughput_record = None
        if sim_subdir is None and not extra_defines:
            try:
                from orchestrator.langgraph.throughput_gate import (
                    evaluate_block_throughput,
                )
                if passed:
                    # engine-v31 step 4: pass the block's current RTL so a STALE
                    # throughput_measured.json (older than the RTL -- e.g. a
                    # prior DV run's number on the retry path) is ignored and
                    # re-measured, not trusted.
                    throughput_record = evaluate_block_throughput(
                        str(PROJECT_ROOT), block_name, sim_dir, rtl_path
                    )
                else:
                    throughput_record = {
                        "gate": "measured_throughput", "scope": "block",
                        "applicable": False, "passed": None,
                        "reason": ("functional DV failed; throughput not "
                                   "measured"),
                    }
            except Exception:  # noqa: BLE001 - never fail DV on gate plumbing
                throughput_record = {
                    "gate": "measured_throughput", "scope": "block",
                    "applicable": False, "passed": None,
                    "reason": "throughput gate plumbing error",
                }
            if (passed and isinstance(throughput_record, dict)
                    and throughput_record.get("applicable")
                    and throughput_record.get("passed") is False):
                passed = False
                throughput_gate_failed = True
                throughput_needs_tb = bool(
                    throughput_record.get("artifact_missing")
                )
                _trep = throughput_record.get("report", "") or ""
                output = _trep + "\n\n" + output
                try:
                    if log_path:
                        with open(log_path, "a", encoding="utf-8") as _lf:
                            _lf.write("\n" + _trep + "\n")
                except OSError:
                    pass

        return {
            "passed": passed,
            "log": output,
            "returncode": result.returncode,
            "tests_passed": summary["tests_passed"],
            "tests_total": summary["tests_total"],
            "tests_failed": summary["tests_failed"],
            "log_path": log_path,
            "vcd_path": str(vcd_path) if vcd_path.exists() else "",
            "wavekit_audit_path": str(audit_path),
            "wavekit_audit": wavekit_audit,
            "coverage_gate_failed": coverage_gate_failed,
            "coverage_pct": coverage_pct,
            "coverage": coverage_record,
            "throughput_gate_failed": throughput_gate_failed,
            "throughput_needs_tb": throughput_needs_tb,
            "throughput": throughput_record,
        }
    except subprocess.TimeoutExpired:
        cmd = [make_bin, "-C", str(sim_dir)]
        # Persist the timeout so the NEXT attempt auto-extends (x1.5, capped).
        try:
            _to_state.write_text(json.dumps({"timeouts": _prior_timeouts + 1}))
        except Exception:  # noqa: BLE001
            pass
        _mins = _sim_timeout // 60
        # SIM_TIMEOUT (not the generic "timed out" INFRASTRUCTURE marker): a
        # pure timeout produced NO functional verdict, so the diagnose router
        # retries this block with an EXTENDED cap instead of burning the
        # INFRASTRUCTURE_ERROR escalation budget (see pipeline_graph route).
        _msg = (f"SIM_TIMEOUT: Simulation exceeded {_sim_timeout}s ({_mins} min). "
                f"No functional verdict produced. Next attempt will use an "
                f"extended timeout (x1.5, cap {_sim_to_cap}s).")
        log_path = _write_step_log_error(block_name, "simulate", cmd, _msg, attempt)
        return {"passed": False, "log": _msg, "log_path": log_path,
                "sim_timed_out": True, "sim_timeout_s": _sim_timeout}
    except FileNotFoundError as e:
        cmd = [make_bin, "-C", str(sim_dir)]
        log_path = _write_step_log_error(block_name, "simulate", cmd, f"Tool not found: {e}", attempt)
        return {"passed": False, "log": f"Tool not found: {e}", "log_path": log_path}


# ---------------------------------------------------------------------------
# SDC Generation
# ---------------------------------------------------------------------------

def _detect_clock_port(rtl_source: str) -> str:
    """Regex-based clock port detection from Verilog source.

    Scans the module port declarations for common clock port names.
    Returns the detected clock port name, or 'clk' as fallback.
    """
    import re

    port_pattern = re.compile(
        r'\binput\s+(?:wire\s+)?(\w+)', re.MULTILINE
    )
    ports = port_pattern.findall(rtl_source)

    for name in ("clk", "clk_in", "clock", "CLK", "CLOCK"):
        if name in ports:
            return name

    for p in ports:
        if "clk" in p.lower() or "clock" in p.lower():
            return p

    return "clk"


# Word-boundary reset token: matches rst / rst_n / reset / arst_n / aresetn /
# s_axi_aresetn / cpu_rst, but NOT data ports that merely CONTAIN the letters
# (burst, first, wrst_count -> "u_rst"? no: requires (^|_) before the token).
_RESET_TOKEN_RE = re.compile(r"(?:^|_)(?:a?rst|a?reset)(?:_?n)?(?:$|_)")


def _detect_reset_port(rtl_source: str) -> str:
    """Regex-based reset port detection (mirrors ``_detect_clock_port``).

    Returns the detected reset port name, or the ``rst_n`` convention as the
    fallback. The SDC false-path that consumes this is ALWAYS wrapped in a
    ``[get_ports -quiet ...]`` existence guard, so a fallback that is not an
    actual port simply no-ops -- the reset false-path is reset-name-agnostic.
    """
    port_pattern = re.compile(r"\binput\s+(?:wire\s+)?(\w+)", re.MULTILINE)
    ports = port_pattern.findall(rtl_source)

    for name in ("rst_n", "resetn", "reset_n", "rstn", "arst_n", "aresetn",
                 "rst", "reset", "arst", "areset"):
        if name in ports:
            return name

    for p in ports:
        if _RESET_TOKEN_RE.search(p.lower()):
            return p

    return "rst_n"


def _sdc_reset_false_path_enabled() -> bool:
    """``CORESMITH_SDC_RESET_FALSE_PATH`` (default ON).

    The pre-layout reset tree is unbuffered: a blanket ``set_input_delay`` on
    the reset port makes the reset net masquerade as the block WNS (a measured
    -13.58 ns artifact vs -2.72 ns functional on ``intra_rd_encode_core``),
    masking the real timing. Default ON exempts the reset from timing via a
    guarded ``set_false_path``. Set 0/false/no/off to restore the old
    blanket-input-delay-on-all-inputs SDC.
    """
    return os.environ.get(
        "CORESMITH_SDC_RESET_FALSE_PATH", "1"
    ).strip().lower() not in {"0", "false", "no", "off", ""}


def _build_sdc_content(rtl_source: str, target_clock_mhz: float) -> str:
    """THE single SDC generator.

    Unifies ``generate_sdc`` and ``synthesize_block``'s inline SDC copy so BOTH
    sites emit byte-identical constraints. Detects the clock port, applies the
    20%-period I/O delays, and (unless CORESMITH_SDC_RESET_FALSE_PATH is off)
    exempts the reset tree from timing with a guarded ``set_false_path`` so the
    unbuffered pre-layout reset net can't masquerade as the block WNS.
    """
    period_ns = 1000.0 / target_clock_mhz
    clock_port = _detect_clock_port(rtl_source)

    if clock_port:
        sdc_content = (
            f"create_clock -name clk -period {period_ns} [get_ports {clock_port}]\n"
            f"set_input_delay -clock clk {period_ns * 0.2} [all_inputs]\n"
            f"set_output_delay -clock clk {period_ns * 0.2} [all_outputs]\n"
        )
    else:
        sdc_content = (
            f"create_clock -name vclk -period {period_ns}\n"
            f"set_input_delay -clock vclk {period_ns * 0.2} [all_inputs]\n"
            f"set_output_delay -clock vclk {period_ns * 0.2} [all_outputs]\n"
        )

    if _sdc_reset_false_path_enabled():
        reset_port = _detect_reset_port(rtl_source)
        # Reset distribution is built/buffered at PnR, so pre-layout the reset
        # is exempt from timing. Guarded so blocks without the port no-op.
        sdc_content += (
            f"if {{[llength [get_ports -quiet {reset_port}]] > 0}} "
            f"{{ set_false_path -from [get_ports {reset_port}] }}\n"
        )

    return sdc_content


async def generate_sdc(
    block_name: str,
    rtl_source: str,
    target_clock_mhz: float,
    sdc_path: str,
) -> str:
    """Generate SDC constraints by detecting the clock port from RTL.

    Uses a regex-based detector (fast, no LLM cost). Falls back to 'clk'
    if no clock port is found. Also creates a virtual clock for pure
    combinational modules.

    Returns the SDC file path.
    """
    Path(sdc_path).write_text(_build_sdc_content(rtl_source, target_clock_mhz))
    return sdc_path


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------

def synthesize_block(
    block: dict, rtl_path: str, target_clock_mhz: float = 50.0,
    attempt: int = 1,
) -> dict:
    """Run Yosys synthesis targeting Sky130."""
    block_name = block["name"]
    output_dir = PROJECT_ROOT / "syn" / "output" / block_name
    output_dir.mkdir(parents=True, exist_ok=True)

    liberty = str(LIBERTY_FILE)
    netlist_path = output_dir / f"{block_name}_netlist.v"
    report_path = output_dir / f"{block_name}_report.txt"

    # ------------------------------------------------------------------
    # GENERIC vs PDK-mapped synthesis.
    #
    # When the Sky130 liberty / PDK is present we map to real standard
    # cells (area/STA-capable). When it is absent (most non-backend
    # hosts) we fall back to a PDK-FREE *generic* technology-mapping
    # synth: it still proves the design ELABORATES, has no combinational
    # loops, and MAPS TO GATES that TERMINATE -- the synthesizability
    # gate that catches a non-terminating combinational cloud. Forced on
    # with CORESMITH_SYNTH_GENERIC=1.
    # ------------------------------------------------------------------
    _force_generic = os.environ.get("CORESMITH_SYNTH_GENERIC", "").strip().lower() in {
        "1", "true", "yes", "on",
    }
    _have_pdk = LIBERTY_FILE.exists()
    generic = _force_generic or not _have_pdk

    # Generic SRAM wrapper: read its synth (black-box) view first so every
    # `cs_sram_*` instance is an opaque macro (0 storage flops) -- the backend
    # streams in the real SRAM collateral. Only when the block instantiates it.
    _wrapper_read = ""
    try:
        from orchestrator.langgraph.sram_wrapper import (
            uses_wrapper as _uw,
        )
        from orchestrator.langgraph.sram_wrapper import (
            wrapper_lib_path as _wlp,
        )
        _rtl_src = Path(rtl_path).read_text() if Path(rtl_path).exists() else ""
        if _uw(_rtl_src):
            _wrapper_read = (
                f"read_verilog -DCORESMITH_SRAM_SYNTH -sv {_wlp()}\n"
                "blackbox cs_sram_1rw cs_sram_1rw1r\n"
            )
    except Exception:
        _wrapper_read = ""

    if generic:
        # No liberty: full generic synth + generic gate mapping (abc -g),
        # then a plain stat (cell counts). This TERMINATES iff the design
        # is real, finite, loop-free logic.
        script = f"""# Auto-generated GENERIC synthesis script for {block_name}
read_verilog -sv {rtl_path}
{_wrapper_read}hierarchy -top {block_name}
proc
flatten
opt
synth -top {block_name}
memory_map
opt -full
techmap
abc -g AND,OR,XOR,MUX
opt_clean
stat
write_verilog -noattr {netlist_path}
"""
    else:
        script = f"""# Auto-generated synthesis script for {block_name}
read_verilog {rtl_path}
{_wrapper_read}hierarchy -top {block_name}
proc
flatten
opt
synth -run begin:fine
memory_bram
memory_map
synth -run fine:
dfflibmap -liberty {liberty}
abc -liberty {liberty}
opt_clean
stat -liberty {liberty}
write_verilog -noattr {netlist_path}
"""
    script_path = output_dir / f"synth_{block_name}.ys"
    script_path.write_text(script)

    rtl_source = Path(rtl_path).read_text() if Path(rtl_path).exists() else ""
    # Unified with generate_sdc: one SDC generator so both sites emit identical
    # constraints (reset false-path + I/O delays).
    sdc_content = _build_sdc_content(rtl_source, target_clock_mhz)
    sdc_path = output_dir / f"{block_name}.sdc"
    sdc_path.write_text(sdc_content)

    # Wall-clock synth timeout (the KEY synthesizability gate). A
    # non-terminating combinational design (e.g. an unrolled RD-search
    # cloud) blows this -> success=False -> route_after_synth -> diagnose.
    _synth_timeout = scaled(600, env="CORESMITH_SYNTH_TIMEOUT_S")
    try:
        # cwd=PROJECT_ROOT so a PROJECT-RELATIVE artifact path inside the RTL
        # resolves exactly as it does in simulation. Block RTL legitimately
        # carries `$readmemh("inputs/rom_images/<image>.memh", ...)` (and the
        # cs_rom_1r wrapper's INIT_FILE parameter is the same relative path);
        # yosys resolves those against ITS OWN cwd, which was whatever the
        # daemon happened to be started in -- so the image was unreadable at
        # block synth even though the identical path worked in DV. Both flat
        # synth and the memory-flop probe already run rooted at the project.
        result = subprocess.run(
            ["yosys", "-s", str(script_path)],
            cwd=str(PROJECT_ROOT.resolve()),
            capture_output=True,
            text=True,
            timeout=_synth_timeout,
        )

        gate_count = 0
        chip_area = 0.0
        for line in result.stdout.split("\n"):
            if "Number of cells:" in line:
                # Plain stat format: "   Number of cells:   178"
                try:
                    gate_count = int(line.split(":")[-1].strip())
                except ValueError:
                    pass
            else:
                stripped = line.strip()
                if stripped.endswith("cells") and not stripped.startswith("-"):
                    # Liberty stat format: "      178 1.73E+03 cells"
                    parts = stripped.split()
                    if len(parts) >= 3:
                        try:
                            gate_count = int(parts[0])
                        except ValueError:
                            pass
            if "Chip area for module" in line:
                # "   Chip area for module '\adder_16bit': 1727.907200"
                try:
                    chip_area = float(line.split(":")[-1].strip())
                except ValueError:
                    pass

        report_path.write_text(result.stdout)

        from orchestrator.langgraph.ppa_check import count_flops_from_stat
        ff_count = count_flops_from_stat(result.stdout)

        log_path = _write_step_log(block_name, "synthesize", ["yosys", "-s", str(script_path)], result, attempt)

        return {
            "success": result.returncode == 0,
            "gate_count": gate_count,
            "chip_area_um2": chip_area,
            "ff_count": ff_count,
            "netlist_path": str(netlist_path),
            "sdc_path": str(sdc_path),
            "report_path": str(report_path),
            "liberty_path": liberty,
            "log": result.stdout[-3000:] + "\n" + result.stderr[-1000:],
            "log_path": log_path,
        }
    except subprocess.TimeoutExpired:
        cmd = ["yosys", "-s", str(script_path)]
        _msg = (
            f"SYNTH FAILED: Yosys did not terminate within {_synth_timeout}s "
            f"(CORESMITH_SYNTH_TIMEOUT_S). This is an UNSYNTHESIZABLE design "
            f"signal -- typically a combinational loop or an enormous unrolled "
            f"combinational cloud (e.g. an un-pipelined RD-search encoder) that "
            f"never maps to finite gates. Pipeline/FSM-sequentialize it."
        )
        log_path = _write_step_log_error(block_name, "synthesize", cmd, _msg, attempt)
        return {"success": False, "log": _msg, "log_path": log_path}
    except FileNotFoundError:
        cmd = ["yosys", "-s", str(script_path)]
        log_path = _write_step_log_error(block_name, "synthesize", cmd, "Yosys not installed", attempt)
        return {"success": False, "log": "Yosys not installed", "log_path": log_path}


# ---------------------------------------------------------------------------
# Lint Fixer (local LLM iteration)
# ---------------------------------------------------------------------------

async def fix_lint_errors(
    block_name: str, rtl_path: str, lint_log_path: str,
    callbacks: list = None,
) -> bool | None:
    """Call an LLM to fix Verilator lint errors in the RTL.

    Disk-first: the agent reads the RTL and lint log from disk, uses
    the Edit tool to fix in-place.  Returns True if the agent modified
    the file, None if it couldn't fix.
    """
    from orchestrator.langchain.agents.coresmith_llm import DEFAULT_MODEL, ClaudeLLM

    prompt_file = Path(__file__).resolve().parent.parent / "langchain" / "prompts" / "lint_fixer.md"
    if prompt_file.exists():
        system_prompt = prompt_file.read_text()
    else:
        system_prompt = (
            "You are an expert Verilog lint fixer. Read the RTL file and "
            "lint error log, then use the Edit tool to fix the errors in-place."
        )

    user_message = (
        f"Block: {block_name}\n\n"
        f"## Working Files\n"
        f"- RTL file: {rtl_path}\n"
        f"- Lint log: {lint_log_path}\n"
        f"- Constraints: .coresmith/blocks/{block_name}/constraints.json\n"
        f"  ({_naming_precedence_line()})\n\n"
        f"Read the lint errors, then use the Edit tool to fix the RTL file "
        f"in-place. Do NOT rewrite the entire file -- make targeted fixes."
    )

    block_title = block_name.replace("_", " ").title()
    llm = ClaudeLLM(
        model=DEFAULT_MODEL,
        timeout=scaled(600, env="CORESMITH_LINT_FIX_TIMEOUT"),
    )

    try:
        await llm.call(
            system=system_prompt,
            prompt=user_message,
            run_name=f"Lint Fix [{block_title}]",
        )
        return True

    except Exception as e:
        log(f"  [LINT-FIX] LLM error: {e}", RED)
        return None


# ---------------------------------------------------------------------------
# Synthesis Fixer (local LLM iteration)
# ---------------------------------------------------------------------------

async def fix_synth_errors(
    block_name: str, rtl_path: str, synth_log_path: str,
    callbacks: list = None,
) -> bool | None:
    """Call an LLM to fix Yosys synthesis errors in the RTL.

    Disk-first: the agent reads the RTL and synth log from disk, uses
    the Edit tool to fix in-place.  Returns True if the agent modified
    the file, None if it couldn't fix.
    """
    from orchestrator.langchain.agents.coresmith_llm import DEFAULT_MODEL, ClaudeLLM

    prompt_file = Path(__file__).resolve().parent.parent / "langchain" / "prompts" / "synth_fixer.md"
    if prompt_file.exists():
        system_prompt = prompt_file.read_text()
    else:
        system_prompt = (
            "You are an expert synthesis engineer. Read the RTL file and "
            "synthesis log, then use the Edit tool to fix the errors in-place."
        )

    # A STRUCTURAL fix (wide flat-packed storage -> a barrel-shifter cloud)
    # touches dozens of sites; targeted hunk-patches go stale and fail to apply
    # (the codec RD-core's apply_patch failed exactly this way). For those,
    # mandate a full-file rewrite; targeted Edit only for trivial fixes.
    structural = False
    rtl_before = ""
    try:
        rtl_before = Path(rtl_path).read_text()
        from orchestrator.langgraph.rtl_storage_lint import (
            find_flat_packed_dynamic_storage,
        )
        structural = not find_flat_packed_dynamic_storage(rtl_before).ok
    except Exception:  # noqa: BLE001
        pass

    if structural:
        edit_instr = (
            "This is a STRUCTURAL fix (wide flat-packed storage sliced by a "
            "runtime index -> a combinational cloud). It touches many sites, so "
            "REWRITE THE ENTIRE MODULE with the Write tool -- do NOT make "
            "scattered targeted edits (they go stale and fail to apply). Convert "
            "the flagged flat-packed regs to cs_fpmem/cs_sram addressed memories "
            "(or a per-element array indexed by a REGISTERED address with a "
            "1-cycle registered read); keep the datapath math byte-exact."
        )
    else:
        edit_instr = (
            "Use the Edit tool to make targeted fixes in-place; do NOT rewrite "
            "the whole file for a small change."
        )

    user_message = (
        f"Block: {block_name}\n\n"
        f"## Working Files\n"
        f"- RTL file: {rtl_path}\n"
        f"- Synthesis log: {synth_log_path}\n"
        f"- Diagnosis / prior error context (READ THIS FIRST -- when a prior "
        f"diagnose ran it names the specific root cause + the fix): "
        f".coresmith/blocks/{block_name}/previous_error.txt\n"
        f"- Constraints: .coresmith/blocks/{block_name}/constraints.json\n"
        f"  ({_naming_precedence_line()})\n\n"
        f"Read previous_error.txt and the synthesis errors, then fix the RTL. "
        f"{edit_instr}"
    )

    block_title = block_name.replace("_", " ").title()
    llm = ClaudeLLM(
        model=DEFAULT_MODEL,
        timeout=scaled(600, env="CORESMITH_SYNTH_FIX_TIMEOUT"),
    )

    try:
        await llm.call(
            system=system_prompt,
            prompt=user_message,
            run_name=f"Synth Fix [{block_title}]",
        )
    except Exception as e:
        log(f"  [SYNTH-FIX] LLM error: {e}", RED)
        return None

    # VERIFY THE FIX LANDED. The LLM call returning without raising does NOT
    # mean the file changed -- a failed apply_patch / stale-context Edit leaves
    # the RTL untouched, and re-synthesizing it just burns another ~600 s on a
    # guaranteed-identical timeout (the codec RD-core did this repeatedly).
    if os.environ.get("CORESMITH_VERIFY_FIX_LANDED", "1").strip() != "0":
        try:
            rtl_after = Path(rtl_path).read_text()
        except Exception:  # noqa: BLE001
            rtl_after = rtl_before
        if rtl_before and rtl_after == rtl_before:
            log("  [SYNTH-FIX] LLM produced NO change to the RTL -- not "
                "re-synthesizing (would be an identical failure)", RED)
            return None
    return True


# ---------------------------------------------------------------------------
# Testbench Fixer (local LLM iteration for sim failures)
# ---------------------------------------------------------------------------

async def fix_testbench_errors(
    block_name: str, rtl_path: str, tb_path: str, sim_log_path: str,
    callbacks: list = None,
) -> bool | None:
    """Call an LLM to fix simulation errors by editing the testbench.

    Disk-first: the agent reads the testbench, RTL, sim log, uArch spec,
    and DV rules from disk, uses the Edit tool to fix in-place.
    Returns True if the agent modified the file, None if it couldn't fix.
    """
    from orchestrator.langchain.agents.coresmith_llm import DEFAULT_MODEL, ClaudeLLM

    system_prompt = (
        "You are an expert verification engineer. A cocotb testbench is "
        "failing during simulation. Read the testbench, RTL, simulation log, "
        "and uArch spec, then fix the testbench in-place using the Edit tool.\n\n"
        "Common issues:\n"
        "- Wrong port/signal names (check RTL module ports)\n"
        "- Import errors (use the <block>_model wrapper, not direct imports)\n"
        "- Timer(0) usage (use RisingEdge/FallingEdge instead)\n"
        "- Wrong timing assumptions (check pipeline latency in uArch spec)\n"
        "- Golden model mismatches (check algorithm implementation)\n"
        "- Type errors (cast numpy types to int before DUT assignment)\n"
        "- Cocotb API issues (use unit= not units=, start_soon not start_fork)\n"
        "- Missing, empty, or header-only dump.vcd / failed WaveKit audit "
        "(tests must advance time and exercise real DUT activity)\n\n"
        "Make targeted fixes. Do NOT rewrite the entire testbench unless "
        "the structure is fundamentally broken."
    )

    user_message = (
        f"Block: {block_name}\n\n"
        f"## Working Files\n"
        f"- Testbench (fix this): {tb_path}\n"
        f"- RTL Verilog: {rtl_path}\n"
        f"- Simulation log: {sim_log_path}\n"
        f"- VCD waveform: sim_build/{block_name}/dump.vcd\n"
        f"- WaveKit audit: sim_build/{block_name}/wavekit_audit.json\n"
        f"- uArch Spec: arch/uarch_specs/{block_name}.md\n"
        f"- Constraints: .coresmith/blocks/{block_name}/constraints.json\n"
        f"- DV Rules: arch/DV_RULES.md\n\n"
        f"Read the simulation log to understand the failure, then read the "
        f"testbench and RTL. Fix the testbench in-place using the Edit tool."
    )

    block_title = block_name.replace("_", " ").title()
    # 600s default; bump via CORESMITH_TB_FIX_TIMEOUT for complex blocks
    # whose TB rewrite genuinely needs more than 10 minutes. The previous
    # 300s default consistently timed out for non-trivial blocks (mcu3
    # 3-stage CPU, multi-stage pipelines) and produced partial fixes that
    # didn't address the root cause.
    llm = ClaudeLLM(
        model=DEFAULT_MODEL,
        timeout=scaled(600, env="CORESMITH_TB_FIX_TIMEOUT"),
    )

    try:
        await llm.call(
            system=system_prompt,
            prompt=user_message,
            run_name=f"TB Fix [{block_title}]",
        )
        return True
    except Exception as e:
        log(f"  [TB-FIX] LLM error: {e}", RED)
        return None


def _naming_precedence_line() -> str:
    """Naming precedence for a block's ACCUMULATED constraints ('' on error).

    A learned constraint is one debug agent's read of one failure; the frozen
    interface contract is design intent. When they disagree about a port NAME,
    the contract wins -- otherwise a constraint that memorialised a collapsed
    name keeps re-introducing it on every fixer pass.
    """
    try:
        from orchestrator.langgraph.contract_conformance import (
            CONSTRAINT_PRECEDENCE_LINE,
        )
        return CONSTRAINT_PRECEDENCE_LINE
    except Exception:  # noqa: BLE001 - prompt garnish, never blocks a fix
        return ""


# ---------------------------------------------------------------------------
# Debug Agent
# ---------------------------------------------------------------------------

async def diagnose_failure(
    block_name: str,
    phase: str = "sim",
    project_root: str = "",
    callbacks: list = None,
) -> dict:
    """Run DebugAgent to analyze failure -- disk-first, agent reads all files."""
    from orchestrator.langchain.agents.debug_agent import DebugAgent

    # Diagnose runs per-block on sim/synth failures; use Sonnet not Opus.
    agent = DebugAgent(temperature=0.1)
    return await agent.analyze(
        block_name=block_name,
        phase=phase,
        project_root=project_root or str(PROJECT_ROOT),
        mode="debug",
        callbacks=callbacks,
    )
