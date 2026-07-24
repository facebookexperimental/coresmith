# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""SRAM-macro registry: discover pre-built hard macros from the PDK and detect
which ones a synthesized netlist instantiates.

This is the deterministic backbone of the "5th fix": the frontend agent now
*instantiates* a named SRAM macro (see prompts/skills/memory_macro_vs_flops.md),
but the backend flow only ever read standard-cell collateral. This module lets
both sides see the *actual* macros on disk:

* `discover_macros()` scans `<pdk>/<variant>/libs.ref/sky130_sram_macros/` and
  parses each macro's LEF (SIZE + power pins) + name (geometry) into a
  `MacroInfo` carrying every collateral path (gds/lef/lib/spice/verilog).
* `macro_menu_markdown()` formats the discovered set for the agent prompt, so
  the menu reflects what is really available (not a hardcoded 3) -- including
  any macro generated on the fly by the OpenRAM fallback.
* `detect_instantiated_macros()` greps a netlist for instantiated macro modules
  so the backend knows which LEF/GDS/lib/spice to inject into PnR/DRC/LVS.

Pure stdlib, no EDA tools required -- safe to import anywhere.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

# sky130 SRAM macros live under this leaf dir inside a PDK variant. efabless
# ships them in sky130B; some installs mirror them into sky130A.
_SRAM_DIR = "sky130_sram_macros"
_VARIANTS = ("sky130A", "sky130B")
# Preferred liberty corner (typical, 1.8V, 25C). Fall back to any .lib.
_PREF_LIB_CORNER = "TT_1p8V_25C"

# efabless pre-built: sky130_sram_<cap>kbyte_<ports>_<width>x<depth>_<maskbits>
_NAME_RE = re.compile(
    r"^sky130_sram_(?P<cap>\d+)kbyte_(?P<ports>\w+?)_(?P<width>\d+)x(?P<depth>\d+)_(?P<mask>\d+)$"
)
# raw OpenRAM output: sram_<ports>_<width>_<depth>_<maskbits>_sky130
_OPENRAM_NAME_RE = re.compile(
    r"^sram_(?P<ports>\w+?)_(?P<width>\d+)_(?P<depth>\d+)_(?P<mask>\d+)_sky130$"
)
# OpenRAM rom_compiler output (coresmith naming): rom_<ports>_<width>_<depth>_sky130
_ROM_NAME_RE = re.compile(
    r"^rom_(?P<ports>\d+r)_(?P<width>\d+)_(?P<depth>\d+)_sky130$"
)


def _parse_geometry(name: str):
    """Return (ports, data_bits, words, mask_bits) from any naming scheme,
    or None if none matches. ROM macros have no write mask (mask_bits=0)."""
    m = _NAME_RE.match(name) or _OPENRAM_NAME_RE.match(name)
    if m:
        return (
            m.group("ports"),
            int(m.group("width")),
            int(m.group("depth")),
            int(m.group("mask")),
        )
    m = _ROM_NAME_RE.match(name)
    if m:
        return (m.group("ports"), int(m.group("width")), int(m.group("depth")), 0)
    return None


def _macro_kind(name: str) -> str:
    return "rom" if _ROM_NAME_RE.match(name) else "sram"


@dataclass
class MacroInfo:
    """One pre-built hard macro and all its collateral paths + geometry."""

    name: str
    lef: str = ""
    gds: str = ""
    lib: str = ""
    spice: str = ""
    verilog: str = ""
    width_um: float = 0.0          # physical LEF SIZE width
    height_um: float = 0.0         # physical LEF SIZE height
    words: int = 0                 # depth (number of addressable words)
    data_bits: int = 0             # word width in bits
    bits: int = 0                  # total capacity in bits (words * data_bits)
    ports: str = ""               # e.g. "1rw1r"
    mask_bits: int = 0             # write-mask granularity in bits
    power_pin: str = "vccd1"      # macro power pin (tie to VPWR)
    ground_pin: str = "vssd1"     # macro ground pin (tie to VGND)
    kind: str = "sram"            # "sram" | "rom" (mask ROM, read-only)
    # Distinct-pin name pairs whose PORT metal overlaps on the SAME layer in the
    # macro's own LEF abstract -- an intra-macro pin short. Some OpenRAM sky130
    # macros ship this defect (e.g. sram_1rw1r_64_32_8's din0[50]/addr1[4] met4
    # stripes overlap in BOTH the LEF abstract and the real GDS), which no
    # honest LVS can pass. Populated at discovery; empty for a clean macro.
    pin_shorts: tuple[tuple[str, str], ...] = ()

    @property
    def lvs_clean_pins(self) -> bool:
        """True iff the macro's LEF has no intra-macro pin short."""
        return not self.pin_shorts

    @property
    def kib(self) -> float:
        return self.bits / 8192.0

    @property
    def area_um2(self) -> float:
        return self.width_um * self.height_um

    def collateral_complete(self) -> bool:
        """True iff every view needed by the full backend flow exists.

        ROM macros: OpenRAM's rom_compiler does not characterize (no .lib is
        ever emitted -- upstream TODO), so a ROM's required set is
        lef/gds/spice/verilog only; STA treats it as a constrained blackbox.
        """
        views = (self.lef, self.gds, self.spice, self.verilog)
        if self.kind != "rom":
            views = views + (self.lib,)
        return all(p and Path(p).exists() for p in views)


def _parse_lef(lef_path: Path) -> tuple[float, float, str, str]:
    """Return (width_um, height_um, power_pin, ground_pin) from a macro LEF."""
    w = h = 0.0
    power_pin = "vccd1"
    ground_pin = "vssd1"
    try:
        text = lef_path.read_text(errors="ignore")
    except OSError:
        return w, h, power_pin, ground_pin
    m = re.search(r"^\s*SIZE\s+([\d.]+)\s+BY\s+([\d.]+)\s*;", text, re.MULTILINE)
    if m:
        w, h = float(m.group(1)), float(m.group(2))
    # Power/ground pins carry USE POWER / USE GROUND inside their PIN block.
    for pm in re.finditer(
        r"PIN\s+(\S+)(.*?)END\s+\1", text, re.DOTALL
    ):
        pin_name, body = pm.group(1), pm.group(2)
        if re.search(r"USE\s+POWER", body):
            power_pin = pin_name
        elif re.search(r"USE\s+GROUND", body):
            ground_pin = pin_name
    return w, h, power_pin, ground_pin


def _lef_pin_rects(lef_text: str) -> dict[str, list[tuple[str, tuple]]]:
    """Parse ``{pin_name: [(layer, (x0,y0,x1,y1)), ...]}`` from a LEF's PIN
    section. Stops at OBS (obstruction metal is not a pin). Power/ground pins
    are included but harmless -- only DISTINCT-pin overlaps are ever a short."""
    pins: dict[str, list[tuple[str, tuple]]] = {}
    cur: str | None = None
    layer: str | None = None
    for line in lef_text.splitlines():
        s = line.split()
        if not s:
            continue
        head = s[0]
        if head == "OBS":
            break
        if head == "PIN" and len(s) >= 2:
            cur = s[1]
            pins.setdefault(cur, [])
            layer = None
        elif head == "END" and len(s) >= 2 and cur is not None and s[1] == cur:
            cur = None
            layer = None
        elif head == "LAYER" and len(s) >= 2:
            layer = s[1]
        elif head == "RECT" and cur is not None and layer is not None:
            # `RECT x0 y0 x1 y1 ;`, also `RECT MASK <n> x0 y0 x1 y1 ;` -- take the
            # first four float tokens (skips a MASK keyword/index). ITERATE forms
            # (stepped arrays) rarely apply to SRAM pins; ignored if ambiguous.
            nums: list[float] = []
            for tok in s[1:]:
                try:
                    nums.append(float(tok))
                except ValueError:
                    continue
                if len(nums) == 4:
                    break
            if len(nums) == 4:
                x0, y0, x1, y1 = nums
                pins[cur].append(
                    (layer, (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)))
                )
    return pins


def _rects_overlap(a: tuple[str, tuple], b: tuple[str, tuple]) -> bool:
    """Two (layer, rect) tuples share metal: same layer + positive-area overlap."""
    la, (ax0, ay0, ax1, ay1) = a
    lb, (bx0, by0, bx1, by1) = b
    if la != lb:
        return False
    return min(ax1, bx1) > max(ax0, bx0) and min(ay1, by1) > max(ay0, by0)


def lef_pin_shorts(lef: str | Path) -> tuple[tuple[str, str], ...]:
    """DISTINCT-pin name pairs whose PORT rectangles overlap on the same layer.

    ``lef`` is a LEF file path OR its text. A returned pair is an intra-macro
    pin short: the two pins are galvanically one node in the abstract (and, for
    the OpenRAM sky130 macros whose abstract mirrors their GDS, in silicon too).
    Same-pin multi-rect stripes and different-layer crossings are NOT shorts.
    Pure/deterministic -- no EDA tools. Returns a sorted tuple.
    """
    text = ""
    try:
        p = Path(lef)
        if len(str(lef)) < 4096 and p.exists():
            text = p.read_text(errors="ignore")
    except (OSError, ValueError):
        text = ""
    if not text:
        text = lef if isinstance(lef, str) else ""
    pins = _lef_pin_rects(text)
    names = list(pins)
    shorts: set[tuple[str, str]] = set()
    for i in range(len(names)):
        ra_list = pins[names[i]]
        if not ra_list:
            continue
        for j in range(i + 1, len(names)):
            rb_list = pins[names[j]]
            if not rb_list:
                continue
            if any(_rects_overlap(ra, rb) for ra in ra_list for rb in rb_list):
                a, b = names[i], names[j]
                shorts.add((a, b) if a <= b else (b, a))
    return tuple(sorted(shorts))


def macro_pin_short_guard_enabled() -> bool:
    """Whether macro SELECTION excludes a macro with an intra-macro pin short.

    Default ON. Set ``CORESMITH_MACRO_PIN_SHORT_GUARD=0`` to restore the prior
    behavior (a pin-shorted macro stays selectable -> the historical, real LVS
    short a wide macro like ``sram_1rw1r_64_32_8`` produces). The guard NEVER
    weakens LVS: it removes a genuinely-shorted macro from the candidate set so
    the resolver realizes the memory from LVS-clean collateral instead; it does
    not teach LVS to accept a short.
    """
    from orchestrator.profile import ensure_applied, flag_enabled
    ensure_applied()
    return flag_enabled("CORESMITH_MACRO_PIN_SHORT_GUARD", default=True)


def macro_pin_clean(m: MacroInfo) -> bool:
    """True unless the guard is ON and this macro carries an intra-macro pin
    short. The selection predicate shared by the openram_gen resolvers."""
    return not (m.pin_shorts and macro_pin_short_guard_enabled())


def _first(paths: list[Path]) -> str:
    for p in paths:
        if p.exists():
            return str(p)
    return ""


def _build_macro(name: str, sram_root: Path) -> MacroInfo | None:
    """Assemble a MacroInfo from the collateral subdirs of one sram_macros dir."""
    lef = sram_root / "lef" / f"{name}.lef"
    if not lef.exists():
        return None
    gds = sram_root / "gds" / f"{name}.gds"
    verilog = sram_root / "verilog" / f"{name}.v"
    spice = _first([
        sram_root / "spice" / f"{name}.spice",
        sram_root / "cdl" / f"{name}.cdl",
    ])
    # liberty: prefer the typical corner, else any matching .lib
    lib = sram_root / "lib" / f"{name}_{_PREF_LIB_CORNER}.lib"
    if not lib.exists():
        cands = sorted((sram_root / "lib").glob(f"{name}*.lib"))
        lib = cands[0] if cands else lib

    w, h, ppin, gpin = _parse_lef(lef)
    try:
        shorts = lef_pin_shorts(lef)
    except Exception:  # noqa: BLE001 - never let LEF parsing break discovery
        shorts = ()
    info = MacroInfo(
        name=name,
        lef=str(lef),
        gds=str(gds) if gds.exists() else "",
        lib=str(lib) if lib.exists() else "",
        spice=spice,
        verilog=str(verilog) if verilog.exists() else "",
        width_um=w,
        height_um=h,
        power_pin=ppin,
        ground_pin=gpin,
        pin_shorts=shorts,
    )
    geom = _parse_geometry(name)
    if geom is not None:
        info.ports, info.data_bits, info.words, info.mask_bits = geom
        info.bits = info.data_bits * info.words
    info.kind = _macro_kind(name)
    return info


@lru_cache(maxsize=8)
def discover_macros(pdk_root: str | None = None) -> dict[str, MacroInfo]:
    """Scan the PDK for pre-built SRAM macros. Cached per pdk_root.

    Returns a {macro_name: MacroInfo} map. Scans both sky130A and sky130B
    (efabless ships the macros in sky130B); the first variant that yields a
    given macro wins. Returns {} if the PDK is absent -- callers degrade to the
    flop path / OpenRAM fallback.
    """
    if pdk_root is None:
        try:
            from orchestrator.langgraph.pipeline_helpers import PDK_ROOT
            pdk_root = str(PDK_ROOT)
        except Exception:
            return {}
    root = Path(pdk_root)
    out: dict[str, MacroInfo] = {}
    for variant in _VARIANTS:
        sram_root = root / variant / "libs.ref" / _SRAM_DIR
        lef_dir = sram_root / "lef"
        if not lef_dir.is_dir():
            continue
        for lef in sorted(lef_dir.glob("*.lef")):
            name = lef.stem
            if name in out:
                continue
            info = _build_macro(name, sram_root)
            if info is not None:
                out[name] = info
    return out


def detect_instantiated_macros(
    netlist_path: str, registry: dict[str, MacroInfo] | None = None
) -> list[MacroInfo]:
    """Return the macros (from `registry`) that `netlist_path` instantiates.

    Matches whole-word module names so a substring can't false-trigger.
    """
    if registry is None:
        registry = discover_macros()
    if not registry:
        return []
    try:
        text = Path(netlist_path).read_text(errors="ignore")
    except OSError:
        return []
    found: list[MacroInfo] = []
    for name, info in registry.items():
        if re.search(rf"(?<![\w]){re.escape(name)}(?![\w])", text):
            found.append(info)
    return found


# ---------------------------------------------------------------------------
# Part C: bind the cs_mem_macro_shell / cs_rom_macro_shell leaves to real macros
# ---------------------------------------------------------------------------
# The Part-B backend synth emits the empty `cs_mem_macro_shell` (SRAM) and
# `cs_rom_macro_shell` (mask ROM) leaves in place of a flop array (0 storage
# flops). This is the MISSING physical-design step that resolves each shell's
# WIDTH/DEPTH/NPORT to a CONCRETE on-disk macro (a pre-built one, an OpenRAM
# composition/generation, or -- as a hard, reported error -- nothing), so the
# existing LEF/GDS/lib injection (backend_graph DRC/LVS/PnR) can flow it into
# the flow as a placed black box. Reuses the exact same resolver the frontend
# uses (openram_gen.ensure_macro) -- nothing here reinvents macro generation.

# A shell instance in EITHER the pre-synth wrapper source (explicit
# `#(.WIDTH(..),.DEPTH(..),.NPORT(..))`) OR the derived netlist (yosys bakes the
# params into a `$paramod$<hash>\cs_mem_macro_shell` module and drops the `#()`,
# so we recover the geometry from the DERIVED MODULE'S port widths). Both forms
# are handled by :func:`detect_macro_shells`.
_SHELL_INST_RE = re.compile(
    r"\bcs_(?P<kind>mem|rom)_macro_shell\b\s*#\s*\((?P<params>[^;]*?)\)\s*\w+\s*\("
)
_SHELL_PARAM_RE = {
    "WIDTH": re.compile(r"\.WIDTH\s*\(\s*(\d+)\s*\)"),
    "DEPTH": re.compile(r"\.DEPTH\s*\(\s*(\d+)\s*\)"),
    "NPORT": re.compile(r"\.NPORT\s*\(\s*(\d+)\s*\)"),
}
# Derived module: `module $paramod$..\cs_mem_macro_shell (clk, ce0, ...);` with
# `input [W-1:0] wdata0;` / `input [A-1:0] addr0;` giving WIDTH and DEPTH=2**A.
_SHELL_DERIVED_MOD_RE = re.compile(
    r"module\s+([\\\w$.]*cs_(?P<kind>mem|rom)_macro_shell)\b(?P<body>.*?)\bendmodule",
    re.DOTALL,
)
_PORT_RANGE_RE = re.compile(
    r"\b(?:input|output)\b[^;]*?\[\s*(\d+)\s*:\s*0\s*\]\s*(\w+)\s*;"
)


@dataclass
class ShellSpec:
    """One macro-shell instance's resolved geometry request."""

    kind: str          # "sram" | "rom"
    width: int         # data word width in bits
    depth: int         # number of addressable words
    nport: int = 1     # 1 = 1rw, 2 = 1rw1r (SRAM only; ROM is always 1r)

    @property
    def ports(self) -> str:
        if self.kind == "rom":
            return "1r"
        return "1rw1r" if self.nport >= 2 else "1rw"

    def key(self) -> tuple:
        return (self.kind, self.width, self.depth, self.nport)

    def describe(self) -> str:
        return (f"{self.kind} shell {self.width}b x {self.depth} "
                f"({self.ports})")


def _clog2(n: int) -> int:
    if n <= 1:
        return 1
    b = 0
    v = n - 1
    while v:
        b += 1
        v >>= 1
    return b


def detect_macro_shells(text: str) -> list[ShellSpec]:
    """Every cs_mem/cs_rom macro-shell geometry a source OR netlist instantiates.

    Handles the explicit-parameter form (behavioral wrapper source /
    pre-derivation netlist) and the yosys-derived form (params baked into the
    `$paramod$..\\cs_*_macro_shell` module, recovered from its port widths).
    Deduped by geometry.
    """
    specs: list[ShellSpec] = []
    seen: set[tuple] = set()

    def _add(kind: str, width: int, depth: int, nport: int) -> None:
        if width <= 0 or depth <= 0:
            return
        sp = ShellSpec(kind="rom" if kind == "rom" else "sram",
                       width=width, depth=depth,
                       nport=1 if kind == "rom" else max(1, nport))
        if sp.key() not in seen:
            seen.add(sp.key())
            specs.append(sp)

    # 1) Explicit `#(.WIDTH(..),.DEPTH(..),.NPORT(..))` instantiations.
    for m in _SHELL_INST_RE.finditer(text):
        params = m.group("params")
        w = _SHELL_PARAM_RE["WIDTH"].search(params)
        d = _SHELL_PARAM_RE["DEPTH"].search(params)
        n = _SHELL_PARAM_RE["NPORT"].search(params)
        if not (w and d):
            continue
        _add(m.group("kind"),
             int(w.group(1)), int(d.group(1)),
             int(n.group(1)) if n else 1)

    # 2) Derived modules: recover WIDTH from wdata0/rdata0, DEPTH from addr0,
    #    NPORT from a connected second read port (rdata1 present).
    for m in _SHELL_DERIVED_MOD_RE.finditer(text):
        body = m.group("body")
        ranges = {name: int(msb) + 1 for msb, name in _PORT_RANGE_RE.findall(body)}
        width = ranges.get("wdata0") or ranges.get("rdata0") or ranges.get("rdata")
        aw = ranges.get("addr0") or ranges.get("addr")
        if not width:
            continue
        depth = (1 << aw) if aw else 0
        # width-1 single-bit ports omit the [msb:0] range; treat missing addr as
        # depth unknown -> skip (an explicit-param instance, if any, covers it).
        if not depth:
            continue
        nport = 2 if ("rdata1" in ranges or "addr1" in ranges) else 1
        _add(m.group("kind"), width, depth, nport)

    return specs


@dataclass
class BindResult:
    """Outcome of binding every macro shell in a design to a concrete macro."""

    resolved: list[tuple[ShellSpec, MacroInfo]] = field(default_factory=list)
    plans: list[tuple[ShellSpec, object]] = field(default_factory=list)  # CompositionPlan
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def macros(self) -> list[MacroInfo]:
        return [mi for _, mi in self.resolved]


def resolve_shell(
    spec: ShellSpec,
    *,
    registry: dict[str, MacroInfo] | None = None,
    allow_generate: bool = True,
):
    """Resolve one shell geometry to a concrete macro (or composition plan).

    Delegates to the SAME resolver the frontend uses
    (:func:`openram_gen.ensure_macro`): exact pre-built -> composable ->
    OpenRAM-generate -> over-provision. Returns a ``MacroInfo`` /
    ``CompositionPlan`` / ``None`` (unresolvable -- the caller surfaces a hard
    error, never a silent flop fallback). ROMs need contents to generate, so a
    ROM with no pre-built match returns ``None`` (a reported blocker) rather
    than guessing an image.
    """
    if registry is None:
        registry = discover_macros()
    if spec.kind == "rom":
        from orchestrator.langgraph.openram_gen import find_exact
        return find_exact(words=spec.depth, data_bits=spec.width, registry=registry)
    from orchestrator.langgraph.openram_gen import ensure_macro
    return ensure_macro(
        words=spec.depth, data_bits=spec.width,
        allow_generate=allow_generate, write_size=8, registry=registry,
    )


def bind_macro_shells(
    netlist_path_or_text: str,
    *,
    registry: dict[str, MacroInfo] | None = None,
    allow_generate: bool = True,
    is_text: bool = False,
) -> BindResult:
    """Resolve every macro shell in a synthesized design to a concrete macro.

    Reads the netlist (or raw text), detects each shell geometry, and resolves
    it via :func:`resolve_shell`. A geometry that can be neither matched nor
    generated becomes a HARD, reported error in :attr:`BindResult.errors` (the
    backend surfaces it to the outer agent) -- it is NEVER silently left as
    flops. The resolved macros' collateral is what the existing
    :func:`detect_instantiated_macros` + LEF/GDS/lib injection stream into
    PnR/DRC/LVS.
    """
    if is_text:
        text = netlist_path_or_text
    else:
        try:
            text = Path(netlist_path_or_text).read_text(errors="ignore")
        except OSError:
            return BindResult(errors=[f"cannot read netlist {netlist_path_or_text!r}"])
    if registry is None:
        registry = discover_macros()
    res = BindResult()
    for spec in detect_macro_shells(text):
        macro = resolve_shell(spec, registry=registry, allow_generate=allow_generate)
        if macro is None:
            res.errors.append(
                f"macro-shell {spec.describe()} could NOT be bound to a concrete "
                f"SRAM/ROM macro: no pre-built {spec.width}x{spec.depth} "
                f"{spec.ports} macro in the PDK and OpenRAM "
                f"{'declined/failed' if allow_generate else 'not attempted'}. "
                f"This memory has no placeable macro -- it must NOT fall back to "
                f"a flop array. Provide a matching pre-built macro, enable "
                f"OpenRAM generation, or resize the memory to a buildable geometry."
            )
        elif hasattr(macro, "collateral_complete"):  # MacroInfo
            res.resolved.append((spec, macro))
        else:  # CompositionPlan (tile pre-built macros in RTL)
            res.plans.append((spec, macro))
    return res


def macro_menu_markdown(registry: dict[str, MacroInfo] | None = None) -> str:
    """Markdown table of the available macros, for injection into the agent
    prompt so the menu reflects the real PDK (not a hardcoded list)."""
    if registry is None:
        registry = discover_macros()
    if not registry:
        return (
            "_No pre-built SRAM macros found in the PDK. If a block needs an "
            "on-chip memory, an SRAM macro must be generated (OpenRAM) before "
            "backend; flag this in the spec._"
        )
    rows = [
        "| Macro | Capacity | Organization (words x bits) | Ports | Area |",
        "|-------|----------|-----------------------------|-------|------|",
    ]
    for info in sorted(registry.values(), key=lambda m: (m.bits, m.name)):
        # Don't advertise a macro the backend selection will refuse to place
        # (an intra-macro pin short -> a real LVS short). Same guard as the
        # openram_gen resolvers, so the menu matches what is actually buildable.
        if not macro_pin_clean(info):
            continue
        org = f"{info.words} x {info.data_bits}"
        cap = f"{info.kib:.0f} KiB" if info.bits else "?"
        area = f"~{info.area_um2 / 1e6:.2f} mm^2" if info.area_um2 else "?"
        ports = (info.ports or "?") + (" (mask ROM)" if info.kind == "rom" else "")
        rows.append(
            f"| `{info.name}` | {cap} | {org} | {ports} | {area} |"
        )
    return "\n".join(rows)


def openram_live() -> bool:
    """Whether OpenRAM is available to generate a non-pre-built SRAM geometry
    on demand at backend. Best-effort; False on any import/probe error."""
    try:
        from orchestrator.langgraph.openram_gen import openram_available
        return bool(openram_available())
    except Exception:  # noqa: BLE001
        return False


def sram_policy_context() -> str:
    """The live SRAM-macro menu + OpenRAM-availability policy as a prompt block.

    Shared by the uArch spec author AND the Block Diagram author so BOTH stages
    size/hoist memory blocks against what the backend can actually place --
    the block diagram needs this at DECOMPOSITION time (to decide which stores
    to hoist into their own memory_subsystem block and how to budget their
    area), not only later at uArch. Returns '' on any error (best-effort;
    never breaks prompt construction)."""
    try:
        menu = macro_menu_markdown(discover_macros())
        if openram_live():
            policy = (
                "OpenRAM IS available on this flow: a WIDTH x DEPTH geometry "
                "NOT in the pre-built list below is generated on demand by "
                "OpenRAM at backend and is fully placeable. Treat any "
                "reasonable SRAM geometry as buildable -- an [area] concern is "
                "valid ONLY if the SRAM's PRICED area (~1.7 um^2/bit) busts a "
                "block's area budget (a decomposition/budget issue, not a "
                "macro-availability one). OpenRAM's ROM COMPILER is also "
                "available: a READ-ONLY CONSTANT table (quant matrices, "
                "Huffman codebooks, header images, twiddle factors) must be "
                "declared `impl=rom` in the # MEM manifest and instantiated as "
                "`cs_rom_1r` (INIT_FILE = a $readmemh hex image, path relative "
                "to the project root); the backend generates a sky130 mask-ROM "
                "macro carrying those contents in the mask. A mask ROM prices "
                "FAR below SRAM per bit -- NEVER declare a constant table as a "
                "cs_sram with a tied-off write port: it wastes ~5-10x area AND "
                "a real SRAM powers up unknown, so the table would exist only "
                "in simulation."
            )
        else:
            policy = (
                "OpenRAM is NOT available on this flow: ONLY the pre-built "
                "macros listed below are placeable. A required WIDTH x DEPTH "
                "geometry NOT in the list below is a genuine memory-availability "
                "problem -- surface it."
            )
        return (
            "\n\n# On-chip memory (SRAM) policy\n\n"
            "Any storage >= 16384 bits AND >= 256 words deep is an SRAM, not "
            "flops: hoist it into its OWN memory_subsystem block with an "
            "area_budget priced at ~1.7 um^2/bit and req/resp channels (never "
            "embed it in a functional block's budget). Do NOT pin a specific "
            "macro name; give the WIDTH x DEPTH the block needs so resolution "
            "can match. " + policy + "\n\n" + menu
        )
    except Exception:  # noqa: BLE001 - discovery is best-effort
        return ""
