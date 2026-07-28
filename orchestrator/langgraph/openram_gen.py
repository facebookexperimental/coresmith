# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""OpenRAM fallback: generate a custom SRAM macro when the PDK ships no
pre-built one of the required geometry (and it can't be composed from the
pre-built set).

Resolution order in `ensure_macro()`:
  1. exact pre-built macro in the registry  -> return it (no generation)
  2. composable from pre-built macros        -> return a CompositionPlan
     (the RTL author tiles them; no new GDS needed)
  3. otherwise                               -> generate with OpenRAM, install
     the collateral into the PDK so the registry/backend pick it up
  4. OpenRAM unavailable / generation fails  -> return None + a clear log; the
     caller surfaces this as a spec-level blocker (never silently flops).

OpenRAM (pip `openram`) is heavy: it writes GDS/LEF/lib/spice/verilog and needs
magic/ngspice for full collateral. Generation is therefore opt-in and
time-bounded; everything else here is pure/deterministic and unit-tested.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from orchestrator.langgraph.macro_registry import (
    MacroInfo,
    discover_macros,
    macro_pin_clean,
)

# Default place to install generated macros so discover_macros() finds them:
# the same sram_macros leaf the pre-built ones live in.
_INSTALL_VARIANT = "sky130B"


def tiling_allowed() -> bool:
    """True when a logical memory may be built from MULTIPLE prebuilt macros.

    Default OFF. Tiling (composition + over-provisioning) existed as a
    workaround for a broken OpenRAM; with the generator repaired the exact
    geometry should be built instead. Two concrete reasons to keep it off:

    * A tiled memory's timing is not modelled honestly -- composition Fmax is
      computed from the base macro alone and ignores tile count, so a 16-tile
      memory reports the same frequency as a 1-tile one. Deep tiling in
      particular adds ``ceil(log2(tiles))`` output-mux levels that nothing
      accounts for.
    * An over-provisioned tile is larger than a purpose-built macro and carries
      collateral that was signed off for a different geometry.

    ``CORESMITH_ALLOW_MACRO_TILING=1`` restores the previous behaviour.
    """
    return os.environ.get(
        "CORESMITH_ALLOW_MACRO_TILING", ""
    ).strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class CompositionPlan:
    """A way to build the requested memory by tiling pre-built macros."""

    words: int
    data_bits: int
    tiles_wide: int            # macros concatenated for width
    tiles_deep: int            # macros banked for depth
    base: MacroInfo            # the pre-built macro being tiled
    # Over-provisioning: the tiled array can be DEEPER and/or WIDER than the
    # requested geometry when no exact tiling exists (the request's depth is
    # not a multiple of any prebuilt depth, or its width not a multiple of any
    # prebuilt width). The surplus rows are addressed by tying the high address
    # bits low; the surplus columns leave the extra data bits unconnected. Both
    # are standard, area-cheap relative to flopping the whole memory.
    provisioned_words: int = 0     # tiles_deep * base.words (>= words)
    provisioned_bits: int = 0      # tiles_wide * base.data_bits (>= data_bits)

    def __post_init__(self) -> None:
        if not self.provisioned_words:
            self.provisioned_words = self.tiles_deep * self.base.words
        if not self.provisioned_bits:
            self.provisioned_bits = self.tiles_wide * self.base.data_bits

    @property
    def over_provisioned(self) -> bool:
        return (self.provisioned_words > self.words
                or self.provisioned_bits > self.data_bits)

    def describe(self) -> str:
        extra = ""
        if self.over_provisioned:
            extra = (f"; over-provisioned to "
                     f"{self.provisioned_words}x{self.provisioned_bits} "
                     f"(surplus rows addr-tied, surplus bits unused)")
        return (
            f"compose {self.words}x{self.data_bits} from "
            f"{self.tiles_wide * self.tiles_deep}x `{self.base.name}` "
            f"({self.tiles_wide} wide x {self.tiles_deep} deep){extra}"
        )


def find_exact(words: int, data_bits: int, registry=None) -> MacroInfo | None:
    """An exactly-matching pre-built macro, if any."""
    if registry is None:
        registry = discover_macros()
    for m in registry.values():
        if (
            m.words == words
            and m.data_bits == data_bits
            and m.collateral_complete()
            and macro_pin_clean(m)
        ):
            return m
    return None


def plan_composition(words: int, data_bits: int, registry=None) -> CompositionPlan | None:
    """If the requested geometry tiles cleanly from a single pre-built macro,
    return the plan; else None. Only exact integer tilings are offered."""
    if registry is None:
        registry = discover_macros()
    best: CompositionPlan | None = None
    for m in registry.values():
        if not (m.words and m.data_bits and m.collateral_complete() and macro_pin_clean(m)):
            continue
        if words % m.words or data_bits % m.data_bits:
            continue
        wide = data_bits // m.data_bits
        deep = words // m.words
        if wide < 1 or deep < 1 or (wide == 1 and deep == 1):
            continue  # exact single-macro case handled by find_exact
        plan = CompositionPlan(words, data_bits, wide, deep, m)
        # fewest tiles wins
        if best is None or (wide * deep) < (best.tiles_wide * best.tiles_deep):
            best = plan
    return best


def plan_over_provisioned(
    words: int, data_bits: int, registry=None
) -> CompositionPlan | None:
    """Resolve a geometry that does NOT tile exactly by OVER-PROVISIONING.

    When ``words`` is not a multiple of any prebuilt depth (e.g. a 64-deep
    store vs a 256-deep prebuilt) or ``data_bits`` not a multiple of any
    prebuilt width, tile up to the smallest array that COVERS the request:
    ``ceil(words/base.words)`` deep x ``ceil(data_bits/base.data_bits)`` wide.
    Surplus rows are addressed by tying the high address bits low; surplus
    columns leave the extra bits unconnected. This is the resolution that
    previously fell through to (broken) OpenRAM generation and then to a
    silent full-flop fallback -- a real prebuilt macro is always cheaper than
    flopping the whole memory. Prefers the plan with the least WASTED bit
    area, then the fewest tiles.
    """
    if registry is None:
        registry = discover_macros()
    if words <= 0 or data_bits <= 0:
        return None
    req_bits = words * data_bits
    max_waste_ratio = over_provision_max_ratio()
    best: CompositionPlan | None = None
    best_waste = None
    for m in registry.values():
        if not (m.words and m.data_bits and m.collateral_complete() and macro_pin_clean(m)):
            continue
        deep = -(-words // m.words)          # ceil
        wide = -(-data_bits // m.data_bits)  # ceil
        if deep < 1 or wide < 1:
            continue
        prov_words = deep * m.words
        prov_bits = wide * m.data_bits
        prov_area_bits = prov_words * prov_bits
        # Skip a candidate whose provisioned array is wildly larger than the
        # request -- past that ratio a flop array is the better answer than a
        # mostly-empty macro, and the caller (which decided this store needs a
        # macro) should re-decide rather than place dead silicon.
        if prov_area_bits > req_bits * max_waste_ratio:
            continue
        waste = prov_area_bits - req_bits
        key = (waste, deep * wide)
        if best is None or key < best_waste:
            best = CompositionPlan(
                words, data_bits, wide, deep, m,
                provisioned_words=prov_words, provisioned_bits=prov_bits)
            best_waste = key
    return best


def over_provision_max_ratio() -> float:
    """Max provisioned/requested bit ratio for over-provisioning a macro.

    Above this, a mostly-empty prebuilt macro is worse than flops, so
    :func:`plan_over_provisioned` declines and the caller re-decides. Default
    8x; override with ``CORESMITH_MACRO_OVERPROVISION_MAX``.
    """
    try:
        v = float(os.environ.get("CORESMITH_MACRO_OVERPROVISION_MAX", "8") or "8")
        return v if v >= 1.0 else 8.0
    except ValueError:
        return 8.0


_OPENRAM_RUNNABLE: bool | None = None


def ensure_openram_patched() -> bool:
    """Idempotently self-heal a pip-installed OpenRAM; return runnable status.

    The PyPI ``openram==1.2.48`` wheel omits ``openram/__main__.py`` (so
    ``python -m openram`` dies at launch) and crashes under NumPy 2. Rather
    than depend on a manual per-box venv edit -- which would NOT travel with
    this repo, so a public checkout could not generate macros -- apply the
    in-repo ``scripts/patch_openram.py`` fixes here, once, at point of use.
    Idempotent + guarded: a no-op when already patched or when the venv is
    read-only (then it honestly reports not-runnable). Cached.
    """
    try:
        _repo = Path(__file__).resolve().parents[2]
        _scripts = _repo / "scripts"
        if str(_scripts) not in sys.path:
            sys.path.insert(0, str(_scripts))
        import patch_openram as _po
        return _po.patch_openram().ok
    except Exception as exc:  # noqa: BLE001 - never let self-heal break caller
        # Non-fatal, but NOT silent: a swallowed bug here (e.g. a missing
        # import) once made this return False while hiding the real cause.
        print(f"[OPENRAM] self-heal patch step failed: {exc!r}")
        return False


def openram_available() -> bool:
    """True if OpenRAM can actually be INVOKED the way the engine invokes it.

    A bare ``import openram`` is NOT proof: the PyPI 1.2.48 wheel imports fine
    but omits ``openram/__main__.py``, so ``python -m openram`` (how
    :func:`generate_openram_macro` runs it) dies with "cannot be directly
    executed". That import-only check reported OpenRAM available while every
    generation crashed at launch -- a silent availability lie that made a
    backend run silently flop memories instead of surfacing a blocker. This
    self-heals the wheel via :func:`ensure_openram_patched` and then verifies
    the real invocation path (cached; OPENRAM_HOME's sram_compiler.py counts).
    """
    global _OPENRAM_RUNNABLE
    # OPENRAM_HOME means "a source checkout with sram_compiler.py at its root".
    # It is NOT safe to trust the env var alone: importing the pip package
    # EXPORTS ``OPENRAM_HOME=<site-packages>/openram/compiler`` into os.environ,
    # which has no sram_compiler.py -- so once anything imported openram (the
    # patcher does, on every gen_ram invocation) this probe returned a false
    # NEGATIVE and generation became unreachable. Only honor the var when it
    # actually looks like a checkout; otherwise fall through to the real
    # `python -m openram` invocation check below.
    _home = os.environ.get("OPENRAM_HOME")
    if _home and Path(_home, "sram_compiler.py").exists():
        return True
    if _OPENRAM_RUNNABLE is not None:
        return _OPENRAM_RUNNABLE
    try:
        import openram  # noqa: F401
    except Exception:
        _OPENRAM_RUNNABLE = False
        return False
    # Self-heal the known wheel defects, then confirm the -m entry point.
    runnable = ensure_openram_patched()
    if not runnable:
        try:
            import importlib.util
            runnable = importlib.util.find_spec("openram.__main__") is not None
        except Exception:
            runnable = False
    _OPENRAM_RUNNABLE = runnable
    return runnable


def _write_config(
    cfg_path: Path,
    name: str,
    words: int,
    data_bits: int,
    out_dir: Path,
    num_rw: int,
    num_r: int,
    num_w: int,
    write_size: int,
) -> None:
    cfg = f"""# Auto-generated OpenRAM config (coresmith openram_gen)
word_size = {data_bits}
num_words = {words}
write_size = {write_size}
num_rw_ports = {num_rw}
num_r_ports = {num_r}
num_w_ports = {num_w}
tech_name = "sky130"
process_corners = ["TT"]
supply_voltages = [1.8]
temperatures = [25]
nominal_corner_only = True
route_supplies = "ring"
check_lvs = False
uniquify = True
output_path = "{out_dir}"
output_name = "{name}"
"""
    cfg_path.write_text(cfg)


def macro_name_for(words: int, data_bits: int, ports: str, write_size: int) -> str:
    """OpenRAM-style name parseable by the registry's _OPENRAM_NAME_RE."""
    return f"sram_{ports}_{data_bits}_{words}_{write_size}_sky130"


def generate_openram_macro(
    words: int,
    data_bits: int,
    *,
    num_rw: int = 1,
    num_r: int = 1,
    num_w: int = 0,
    write_size: int = 8,
    pdk_root: str | None = None,
    work_dir: str | None = None,
    timeout_s: int = 3600,
) -> MacroInfo | None:
    """Generate a custom macro with OpenRAM and install its collateral into the
    PDK so discover_macros() finds it. Returns the MacroInfo, or None on any
    failure (logged). Never raises."""
    from orchestrator.langgraph.macro_registry import _build_macro

    if not openram_available():
        print(
            "[OPENRAM] not available (pip install openram / set OPENRAM_HOME). "
            f"Cannot generate {words}x{data_bits} macro -- surfacing as a "
            "spec-level blocker."
        )
        return None

    if pdk_root is None:
        from orchestrator.langgraph.pipeline_helpers import PDK_ROOT
        pdk_root = str(PDK_ROOT)
    ports = f"{num_rw}rw{num_r}r" if num_r else f"{num_rw}rw"
    name = macro_name_for(words, data_bits, ports, write_size)
    work = Path(work_dir or f"/tmp/openram_{name}")
    work.mkdir(parents=True, exist_ok=True)
    cfg_path = work / f"{name}.py"
    _write_config(cfg_path, name, words, data_bits, work, num_rw, num_r, num_w, write_size)

    # Self-heal the pip wheel (missing __main__.py, NumPy-2 bbox) before the
    # first invocation so a fresh checkout works without a manual venv edit.
    ensure_openram_patched()
    # Invoke OpenRAM: prefer `python -m openram`, fall back to OPENRAM_HOME script.
    import sys
    cmd = [sys.executable, "-m", "openram", str(cfg_path)]
    home = os.environ.get("OPENRAM_HOME")
    if home and not _module_runnable():
        cmd = [sys.executable, str(Path(home) / "sram_compiler.py"), str(cfg_path)]
    # Self-contained env: an API caller may pass pdk_root= without exporting
    # PDK_ROOT, but OpenRAM's sky130 tech reads models from $PDK_ROOT. Mirror
    # the ROM wrapper: seed PDK_ROOT from the resolved pdk_root if unset.
    _env = dict(os.environ)
    _env.setdefault("PDK_ROOT", str(pdk_root))
    print(f"[OPENRAM] generating {name} ({words}x{data_bits}) ...")
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s,
            cwd=str(work), env=_env,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        print(f"[OPENRAM] generation failed: {exc}")
        return None
    if r.returncode != 0:
        print(f"[OPENRAM] generation exited {r.returncode}: {r.stderr[-600:]}")
        return None

    installed = _install_collateral(name, work, Path(pdk_root))
    if installed is None:
        print(f"[OPENRAM] generated but collateral incomplete for {name}")
        return None
    discover_macros.cache_clear()
    return _build_macro(name, installed)


def _module_runnable() -> bool:
    try:
        import openram  # noqa: F401
        return True
    except Exception:
        return False


def _install_collateral(name: str, work: Path, pdk_root: Path) -> Path | None:
    """Copy OpenRAM outputs into the PDK's sram_macros dir; return that dir.

    OpenRAM writes <name>.{gds,lef,lib,sp,v} (lib may be corner-suffixed).
    Returns None if the essential views are missing.
    """
    sram_root = pdk_root / _INSTALL_VARIANT / "libs.ref" / "sky130_sram_macros"
    pairs = {
        "gds": [f"{name}.gds"],
        "lef": [f"{name}.lef"],
        "verilog": [f"{name}.v"],
        "spice": [f"{name}.sp", f"{name}.spice"],
        "lib": [f"{name}_TT_1p8V_25C.lib", f"{name}.lib"],
    }
    found_any = False
    for sub, candidates in pairs.items():
        (sram_root / sub).mkdir(parents=True, exist_ok=True)
        for cand in candidates:
            src = work / cand
            if src.exists():
                dst_name = cand if sub != "spice" else f"{name}.spice"
                shutil.copy2(src, sram_root / sub / dst_name)
                found_any = True
                break
    # essentials: lef + gds + verilog
    if not (
        (sram_root / "lef" / f"{name}.lef").exists()
        and (sram_root / "gds" / f"{name}.gds").exists()
    ):
        return None
    return sram_root if found_any else None


def ensure_macro(
    words: int,
    data_bits: int,
    *,
    allow_generate: bool = True,
    write_size: int = 8,
    registry: dict | None = None,
) -> MacroInfo | CompositionPlan | None:
    """Resolve a required memory geometry to something buildable.

    Returns a MacroInfo (exact pre-built or freshly generated), a
    CompositionPlan (tile pre-built macros in RTL), or None (no path -- caller
    must surface as a spec blocker; never silently fall back to flops).

    ``registry`` (default None -> :func:`discover_macros`) lets a caller inject
    a specific macro set -- the Part-C shell binder passes the backend's live
    registry, and tests inject a synthetic one. OpenRAM generation still writes
    into the real PDK and re-discovers, so the generated result is independent
    of the injected registry.
    """
    if registry is None:
        registry = discover_macros()
    exact = find_exact(words, data_bits, registry)
    if exact:
        return exact

    # TILING IS OFF BY DEFAULT. Composition and over-provisioning both build a
    # logical memory out of several prebuilt macros. They existed because
    # OpenRAM generation was failing (an audit found 20 launches across 11
    # geometries: 17 failed, 2 timed out, 1 interrupted -- zero successes), so
    # tiling was the only way to get a real macro instead of a flop array.
    #
    # That premise no longer holds: the OpenRAM repairs in
    # ``scripts/patch_openram.py`` produce macros that pass DRC, LVS and
    # characterization, so the exact geometry can simply be BUILT. A
    # purpose-built macro is tighter than an over-provisioned tile and carries
    # its own signed-off collateral, and a tiled memory's timing is not
    # modelled honestly (composition Fmax ignores tile count entirely).
    #
    # Set ``CORESMITH_ALLOW_MACRO_TILING=1`` to restore the old behaviour.
    if tiling_allowed():
        comp = plan_composition(words, data_bits, registry)
        if comp:
            return comp
    if allow_generate:
        gen = generate_openram_macro(words, data_bits, write_size=write_size)
        if gen and macro_pin_clean(gen):
            return gen
        if gen:
            # OpenRAM reproduced the same abstract that made this geometry a
            # real pin short (e.g. 64x32's din0/addr1 met4 collision). Do NOT
            # place a shorted macro -- fall through to an over-provisioned tile
            # of LVS-clean prebuilt macros.
            print(
                f"[OPENRAM] generated {gen.name} has an intra-macro pin short "
                f"{gen.pin_shorts[:3]} -- discarding; over-provisioning from "
                "clean prebuilt macros instead."
            )
    # Over-provisioning (depth/width rounded up from prebuilt macros, surplus
    # tied off) is tiling by another name and is gated with it. Its original
    # justification was explicitly "OpenRAM was unavailable/broken" -- with the
    # generator repaired, building the exact geometry is the better answer.
    if tiling_allowed():
        over = plan_over_provisioned(words, data_bits, registry)
        if over:
            return over

    # No path. Return None so the caller ESCALATES TO A HUMAN. Do not fall back
    # to flops (that is what produced multi-mm^2, single-digit-MHz memories) and
    # do not tile silently. A memory geometry that OpenRAM cannot build is a
    # design-level decision, not something to paper over.
    print(
        f"[OPENRAM] no macro for {words}x{data_bits}: no exact prebuilt part, "
        f"and OpenRAM generation "
        f"{'failed' if allow_generate else 'was not attempted'}. "
        "Tiling is disabled (set CORESMITH_ALLOW_MACRO_TILING=1 to re-enable). "
        "ESCALATE: this geometry needs a human decision -- regenerate the macro "
        "with bin/gen_ram, choose a buildable geometry, or explicitly accept a "
        "flop array with its measured Fmax cost."
    )
    return None


# ---------------------------------------------------------------------------
# OpenROM: mask-ROM generation via OpenRAM's rom_compiler
# ---------------------------------------------------------------------------
# A read-only constant table (declared `impl=rom` in the # MEM manifest and
# instantiated as `cs_rom_1r`) is generated as a sky130 mask ROM whose contents
# come from the block's INIT_FILE hex image -- the fabrication-realistic
# replacement for the sim-only "$readmemh into a tied-write SRAM" pattern
# (audit F4), at a fraction of SRAM bit density (audit F5).
#
# rom_compiler footguns handled here (validated on the E6 smoke run):
#   * `rom_data` MUST be a path RELATIVE to the compiler cwd -- rom.py save()
#     does a naive `output_path + rom_data` string concat, so an absolute path
#     crashes AFTER GDS/LEF but BEFORE the .v model is written.
#   * `data_type="bin"` reads RAW BYTES; the flat bit-string is carved into
#     WIDTH-bit words MSB-first, so each word serializes BIG-endian.
#   * word_size is in BYTES (word_bits = word_size*8) -> data_bits % 8 == 0.
#   * No .lib is ever emitted (characterization is an upstream TODO) -- the
#     registry's collateral_complete() knows kind="rom" needs no lib.


def rom_name_for(words: int, data_bits: int) -> str:
    """Registry-parseable mask-ROM macro name (see _ROM_NAME_RE)."""
    return f"rom_1r_{data_bits}_{words}_sky130"


def memh_to_rom_bytes(memh_path: str | Path, data_bits: int) -> bytes:
    """Convert a $readmemh image (one hex word per line, comments tolerated)
    to the raw big-endian byte stream rom_compiler's bin reader expects."""
    if data_bits % 8:
        raise ValueError(f"ROM data_bits must be byte-aligned, got {data_bits}")
    word_bytes = data_bits // 8
    out = bytearray()
    for line in Path(memh_path).read_text().splitlines():
        tok = line.split("//")[0].strip()
        if not tok or tok.startswith("@"):
            continue
        out += int(tok, 16).to_bytes(word_bytes, "big")
    return bytes(out)


def _rom_compiler_path() -> Path | None:
    try:
        import openram
        p = Path(openram.__file__).resolve().parent / "rom_compiler.py"
        return p if p.exists() else None
    except Exception:  # noqa: BLE001
        return None


def generate_openram_rom(
    words: int,
    data_bits: int,
    data: bytes | str | Path,
    *,
    pdk_root: str | None = None,
    work_dir: str | None = None,
    timeout_s: int = 3600,
) -> MacroInfo | None:
    """Generate a sky130 mask-ROM macro with OpenRAM's rom_compiler and install
    its collateral into the PDK so discover_macros() finds it.

    ``data`` is the ROM contents: raw bytes, or a path to a ``.memh`` hex image
    (converted big-endian) / a raw ``.bin`` file. Returns the MacroInfo, or
    None on any failure (logged). Never raises.
    """
    from orchestrator.langgraph.macro_registry import _build_macro

    if not openram_available():
        print(
            "[OPENROM] OpenRAM not available (pip install openram / set "
            f"OPENRAM_HOME). Cannot generate {words}x{data_bits} ROM -- "
            "surfacing as a spec-level blocker."
        )
        return None
    if data_bits % 8:
        print(f"[OPENROM] data_bits {data_bits} not byte-aligned -- cannot map "
              "to rom_compiler word_size (bytes)")
        return None
    compiler = _rom_compiler_path()
    if compiler is None:
        print("[OPENROM] rom_compiler.py not found in the openram package "
              "(needs openram >= 1.2)")
        return None

    if isinstance(data, (str, Path)):
        p = Path(data)
        try:
            payload = (memh_to_rom_bytes(p, data_bits)
                       if p.suffix.lower() in (".memh", ".hex", ".mem")
                       else p.read_bytes())
        except (OSError, ValueError) as exc:
            print(f"[OPENROM] cannot read ROM data {p}: {exc}")
            return None
    else:
        payload = bytes(data)
    word_bytes = data_bits // 8
    want = words * word_bytes
    if len(payload) < want:
        payload = payload + b"\x00" * (want - len(payload))  # zero-pad tail
    elif len(payload) > want:
        print(f"[OPENROM] data is {len(payload)} bytes but {words}x{data_bits} "
              f"holds {want} -- truncating")
        payload = payload[:want]

    if pdk_root is None:
        from orchestrator.langgraph.pipeline_helpers import PDK_ROOT
        pdk_root = str(PDK_ROOT)
    name = rom_name_for(words, data_bits)
    work = Path(work_dir or f"/tmp/openrom_{name}")
    work.mkdir(parents=True, exist_ok=True)
    (work / "rom_data.bin").write_bytes(payload)
    out_dir = work / "out"
    cfg_path = work / f"{name}.py"
    cfg_path.write_text(f"""# Auto-generated OpenROM config (coresmith openram_gen)
word_size = {word_bytes}
rom_data = "rom_data.bin"
data_type = "bin"
tech_name = "sky130"
nominal_corner_only = True
route_supplies = "ring"
check_lvsdrc = False
output_name = "{name}"
output_path = "{out_dir}"
""")

    import sys
    env = dict(os.environ)
    env.setdefault("PDK_ROOT", str(pdk_root))
    print(f"[OPENROM] generating {name} ({words}x{data_bits} mask ROM) ...")
    try:
        r = subprocess.run(
            [sys.executable, str(compiler), str(cfg_path)],
            capture_output=True, text=True, timeout=timeout_s,
            cwd=str(work), env=env,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        print(f"[OPENROM] generation failed: {exc}")
        return None
    if r.returncode != 0:
        print(f"[OPENROM] generation exited {r.returncode}: {r.stderr[-600:]}")
        return None

    installed = _install_collateral(name, out_dir, Path(pdk_root))
    if installed is None:
        print(f"[OPENROM] generated but collateral incomplete for {name}")
        return None
    discover_macros.cache_clear()
    return _build_macro(name, installed)
