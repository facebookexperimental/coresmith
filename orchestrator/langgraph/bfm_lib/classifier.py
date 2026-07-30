# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Contract classifier: is a chip-top's external bus a QSPI-slave?

Detects the QSPI-slave accelerator chassis from DUT-BLIND evidence --
the chip-top port *shape* and the architecture artifacts -- and, when it
matches, produces the :class:`QSPIContract` the deterministic BFM needs. It
never inspects internal DUT logic (only the top-level port list, which is part
of the public contract), so a non-conformant DUT cannot fool it into relaxing
the contract.

WHERE the boundary lives matters as much as whether it exists. The external
grader drives the LOCKED Caravel pinout -- ``io_in``/``io_out``/``io_oeb`` on
``user_project_wrapper`` -- so that, and only that, is the *graded boundary*. An
assembled integration top whose own ports are design-prefixed (``qspi_io_in``,
``qspi_io_out``, ``qspi_drive_en``) is **not** the graded boundary even though
the chassis genuinely is QSPI-slave: those are different pins on a different
module. :func:`find_pin_boundary` therefore looks past the assembled top into
the project's Caravel wrapper (``rtl/user_project_wrapper.v`` and friends), and
counts a wrapper only when it REALLY DECLARES the three pad busses as ports --
a doc comment naming them (every hand-written wrapper has one) is not evidence.

That look-past is DIAGNOSTIC, never compensating. Finding the wrapper does NOT
re-point DV at it and does NOT let the run proceed: a boundary that is not the
module the sim elaborates is reported as a FAILURE
(:data:`STATUS_BOUNDARY_OFF_TOP`, identical fail-closed treatment to "no
boundary at all"), with the wrapper's path in the message. The reason to look is
that the wrapper's existence NAMES the upstream breakage -- the graded module was
written but never made the top, typically because the Caravel pad-adapter block
was dropped during assembly -- instead of leaving the operator with a vague "not
a QSPI bus". Compensating for a mis-assembled chip by quietly driving a
different module would hide exactly the defect this reports.

:func:`classify_bus_verdict` reports the whole picture so the caller can never
silently downgrade to the DUT-co-tuned LLM BFM:

``qspi_slave_on_simulated_top``       the graded pins are on the module the
                                     integration sim will elaborate -> the
                                     deterministic BFM can drive them.
``qspi_boundary_not_on_simulated_top`` the graded pins exist, but on another
                                     module (typically the wrapper) -> the
                                     integration sim is NOT driving the graded
                                     boundary. Fail closed.
``spec_says_qspi_but_no_pin_boundary`` the spec declares a QSPI bus and NO pad
                                     boundary exists anywhere -> spec/RTL
                                     CONTRADICTION. Fail closed.
``not_qspi_slave``                    genuinely not a QSPI-slave chassis (no pad
                                     boundary and nothing claims QSPI, or a pad
                                     boundary with no QSPI corroboration) -> the
                                     LLM BFM stays, with the historical advisory.

:func:`classify_chip_bus` keeps its original contract-or-None shape. Callers
that DRIVE pins must use :func:`classify_bus_verdict` and honor
``boundary.is_simulated_top``: a contract alone no longer implies the sim's
toplevel exposes ``io_in``/``io_out``/``io_oeb``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .qspi_contract import QSPIContract

# ---------------------------------------------------------------------------
# the graded Caravel pin boundary
# ---------------------------------------------------------------------------

# The three pad busses the external harness drives. Matched with ``\b`` anchors,
# so a design-prefixed lookalike (``qspi_io_in``) deliberately does NOT match:
# it is a DIFFERENT boundary, and loosening this to a substring/suffix match
# would classify a non-Caravel top as Caravel and then drive the wrong pins.
_PAD_IO_PORTS: tuple[str, ...] = ("io_in", "io_out", "io_oeb")

# Direction each pad bus must be DECLARED with (``inout`` tolerated for all
# three -- some hand-written wrappers model the pads as bidirectional).
_PAD_IO_DIRECTIONS: dict[str, tuple[str, ...]] = {
    "io_in": ("input", "inout"),
    "io_out": ("output", "inout"),
    "io_oeb": ("output", "inout"),
}

# Mirrors ``integration_helpers.CARAVEL_TOP_MODULE`` (kept local: bfm_lib is
# deliberately dependency-light and must import without the LangGraph stack).
CARAVEL_TOP_MODULE = "user_project_wrapper"

# Where a graded Caravel wrapper is conventionally written, relative to the
# project root. Probed in this order, then a bounded sorted sweep of the RTL
# tree, so discovery is deterministic (same project in -> same boundary out).
_WRAPPER_CANDIDATES: tuple[str, ...] = (
    "rtl/user_project_wrapper.v",
    "verilog/rtl/user_project_wrapper.v",
    "rtl/integration/user_project_wrapper.v",
    "rtl/chip_top.v",
)
_WRAPPER_GLOBS: tuple[str, ...] = ("rtl/**/*.v", "verilog/rtl/**/*.v")
_WRAPPER_SCAN_CAP = 400          # bound the sweep on a pathological tree
_WRAPPER_MAX_BYTES = 4_000_000   # skip an implausibly large "RTL" file

# classification statuses (see the module docstring)
STATUS_QSPI_TOP = "qspi_slave_on_simulated_top"
STATUS_BOUNDARY_OFF_TOP = "qspi_boundary_not_on_simulated_top"
STATUS_CONTRADICTION = "spec_says_qspi_but_no_pin_boundary"
STATUS_NOT_QSPI = "not_qspi_slave"

_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.S)
_LINE_COMMENT_RE = re.compile(r"//[^\n]*")
_MODULE_START_RE = re.compile(r"\bmodule\s+([A-Za-z_][\w$]*)")
_DIR_RE = re.compile(r"\b(input|output|inout)\b")
_NONANSI_DECL_RE = re.compile(r"\b(?:input|output|inout)\b[^;]*;")
_IDENT_RE = re.compile(r"[A-Za-z_][\w$]*")

# Type/attribute words that may sit between a direction keyword and the port
# name; never port names themselves.
_DECL_KEYWORDS = frozenset({
    "input", "output", "inout", "wire", "reg", "logic", "bit", "signed",
    "unsigned", "tri", "integer", "real", "supply0", "supply1", "var",
    "parameter", "localparam", "genvar", "byte", "shortint", "int", "longint",
})


def _strip_comments(src: str) -> str:
    """Comment-free copy of ``src`` (evidence must be code, not prose)."""
    return _LINE_COMMENT_RE.sub("", _BLOCK_COMMENT_RE.sub(" ", src or ""))


def _paren_group(text: str, open_idx: int) -> tuple[str, int]:
    """``(inner_text, index_after_close)`` for the paren group at ``open_idx``."""
    depth = 0
    for i in range(open_idx, len(text)):
        ch = text[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return text[open_idx + 1:i], i + 1
    return text[open_idx + 1:], len(text)


def _iter_modules(src: str) -> list[tuple[str, str]]:
    """``[(module_name, module_text)]`` for each ``module .. endmodule``.

    Comments are stripped first, so a header comment describing the Caravel
    interface can never be mistaken for a port declaration.
    """
    text = _strip_comments(src)
    out: list[tuple[str, str]] = []
    for m in _MODULE_START_RE.finditer(text):
        end = text.find("endmodule", m.end())
        out.append((m.group(1), text[m.start():end if end != -1 else len(text)]))
    return out


def _port_decl_chunks(module_text: str) -> list[str]:
    """The module's PORT DECLARATION text: the ANSI header + non-ANSI statements.

    Internal ``wire``/``reg`` declarations and body code are excluded -- an
    internal net named ``io_out`` is not a boundary.
    """
    chunks: list[str] = []
    body = module_text
    open_idx = module_text.find("(")
    if open_idx != -1:
        hash_idx = module_text.find("#")
        if hash_idx != -1 and hash_idx < open_idx:
            # `module m #(params) (ports);` -- skip the parameter list
            _params, after = _paren_group(module_text, open_idx)
            open_idx = module_text.find("(", after)
        if open_idx != -1:
            header, after = _paren_group(module_text, open_idx)
            chunks.append(header)
            body = module_text[after:]
    chunks.extend(_NONANSI_DECL_RE.findall(body))
    return chunks


def _decl_identifiers(item: str) -> list[str]:
    """Declared names in one declaration item (ranges/keywords removed)."""
    txt = re.sub(r"\[[^\]]*\]", " ", item)
    return [t for t in _IDENT_RE.findall(txt) if t not in _DECL_KEYWORDS]


def declared_pad_ports(module_text: str) -> dict[str, str]:
    """``{pad bus -> declared direction}`` for the Caravel pads this module DECLARES.

    Evidence-based: only a real port declaration counts (an ANSI header entry or
    a non-ANSI ``input``/``output`` statement) and the direction must be the one
    the Caravel pinout mandates. A prose mention or an internal net does not
    count, and ``qspi_io_in`` is not ``io_in``.
    """
    found: dict[str, str] = {}
    for chunk in _port_decl_chunks(module_text):
        direction = ""
        for item in re.split(r"[;,]", chunk):
            dirs = _DIR_RE.findall(item)
            if dirs:
                direction = dirs[-1]
            if not direction:
                continue
            names = set(_decl_identifiers(item))
            for sig in _PAD_IO_PORTS:
                if sig in found or sig not in names:
                    continue
                if direction in _PAD_IO_DIRECTIONS[sig]:
                    found[sig] = direction
    return found


def module_declares_pin_boundary(module_text: str) -> bool:
    """True iff this module declares ALL THREE Caravel pad busses as ports."""
    return len(declared_pad_ports(module_text)) == len(_PAD_IO_PORTS)


def declared_port_names(module_text: str) -> list[str]:
    """The module's declared port names, in declaration order (for diagnostics)."""
    out: list[str] = []
    for chunk in _port_decl_chunks(module_text):
        direction = ""
        for item in re.split(r"[;,]", chunk):
            dirs = _DIR_RE.findall(item)
            if dirs:
                direction = dirs[-1]
            if not direction:
                continue
            for name in _decl_identifiers(item):
                if name not in out:
                    out.append(name)
    return out


def _top_has_gpio_pin_boundary(top_rtl_source: str) -> bool:
    """True if the given source declares the Caravel ``io_in``/``io_out``/``io_oeb``.

    This is the QSPI-slave chassis pin boundary (no AXI-Stream): the whole
    external contract is pins on a standard off-chip bus. All three (in/out/oeb)
    must be DECLARED AS PORTS of some module in the source, so neither a design
    that merely names ``io_in`` for something else nor a header comment
    describing the Caravel interface is misclassified.
    """
    return any(
        module_declares_pin_boundary(text)
        for _name, text in _iter_modules(top_rtl_source)
    )


def _bus_protocol_says_qspi(project_root: str) -> bool:
    root = Path(project_root)
    for name in ("prd_spec.json", "ers_spec.json"):
        p = root / ".coresmith" / name
        if not p.exists():
            continue
        try:
            d = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        doc = d.get("prd", d.get("ers", {}))
        df = doc.get("dataflow", {}) if isinstance(doc, dict) else {}
        proto = str(df.get("bus_protocol", "")).lower()
        if "qspi" in proto:
            return True
        # also scan a free-text summary
        summ = str(doc.get("summary", "")).lower() if isinstance(doc, dict) else ""
        if "qspi" in summ and "slave" in summ:
            return True
    return False


def _connections_have_qspi(connections: list | None) -> bool:
    if not connections:
        return False
    for c in connections:
        blob = json.dumps(c).lower()
        if "qspi" in blob:
            return True
    return False


def _reg_map_overrides(project_root: str) -> dict:
    """Best-effort register-map overrides from arch artifacts.

    The defaults already encode the PROTOCOL.md map; this only overrides when
    the run declares an explicit numeric map (register_spec/memory_map). Silent
    on absence -- absence keeps the standard-contract defaults.
    """
    root = Path(project_root)
    out: dict = {}
    mm = root / ".coresmith" / "memory_map.json"
    if mm.exists():
        try:
            d = json.loads(mm.read_text())
        except (OSError, json.JSONDecodeError):
            d = {}
        regs = d.get("registers", d.get("regmap", d.get("map", {})))
        name_to_field = {
            "ctrl": "ctrl_addr", "status": "status_addr", "cfg0": "cfg0_addr",
            "cfg1": "cfg1_addr", "in": "in_addr", "out": "out_addr",
        }
        if isinstance(regs, dict):
            for k, v in regs.items():
                fk = name_to_field.get(str(k).strip().lower())
                addr = v.get("addr") if isinstance(v, dict) else v
                if fk and isinstance(addr, int):
                    out[fk] = addr
        elif isinstance(regs, list):
            for entry in regs:
                if not isinstance(entry, dict):
                    continue
                fk = name_to_field.get(str(entry.get("name", "")).strip().lower())
                addr = entry.get("addr", entry.get("offset"))
                if fk and isinstance(addr, int):
                    out[fk] = addr
    return out


def arch_indicates_qspi_slave(
    project_root: str, connections: list | None = None
) -> bool:
    """RTL-free QSPI-slave hint from the architecture artifacts alone.

    Used at composition time (the ``chip_top`` RTL does not exist yet) to decide
    whether to warn that the composition pin driver is LLM-authored while the
    contract-enforcing deterministic BFM only kicks in at integration_dv.
    """
    return _bus_protocol_says_qspi(project_root) or _connections_have_qspi(connections)


# Pin-group prefixes that mark a DUT-MASTERED second bus (the DUT drives
# csn/sck/io OUT to an external device -- a ROM/flash from which it reads
# operands). DUT-blind: read from the top-level port names only.
_SECOND_BUS_PREFIXES = (
    "rom", "flash", "spiflash", "spi_flash", "mem", "sram_ext", "sdram",
    "psram", "ext", "operand", "src_mem",
)
_MASTER_PIN_TOKENS = (
    "csn", "cs_n", "csb", "sck", "sclk", "io0", "io1", "io2", "io3",
    "mosi", "miso", "d0", "d1", "d2", "d3", "dq0", "dq1",
)


def detect_dut_mastered_buses(top_rtl_source: str) -> list[str]:
    """Names of DUT-MASTERED second buses on the chip top (DUT-blind).

    A QSPI-slave accelerator that ALSO reads operands from an external ROM
    exposes a SECOND pin group (e.g. ``rom_csn``/``rom_sck``/``rom_io0``) that
    the DUT DRIVES as a master. The deterministic QSPI-SLAVE BFM models the host
    side only; a DUT-mastered ROM port is UNMODELED unless a ROM-responder BFM is
    wired. Returns the distinct pin-group prefixes found on the top boundary,
    e.g. ``["rom"]``. Reads only the port-name shape, never internal DUT logic.
    """
    src = (top_rtl_source or "").lower()
    found: list[str] = []
    for pre in _SECOND_BUS_PREFIXES:
        if any(re.search(rf"\b{pre}_{tok}\b", src) for tok in _MASTER_PIN_TOKENS):
            if pre not in found:
                found.append(pre)
    return found


def detect_bus_roles(
    project_root: str,
    top_rtl_source: str = "",
    connections: list | None = None,
) -> dict:
    """Multi-role, DUT-blind classification of the chip-top external bus.

    Beyond the single QSPI-slave role, report whether the DUT ALSO masters a
    second bus (a ROM/flash operand port). Returns::

        {dut_slave: bool,
         dut_master_buses: [prefix, ...],   # e.g. ["rom"]
         modeled:   [role, ...],            # roles the deterministic TB answers
         unmodeled: [role, ...],            # roles it does NOT answer (risk)
         summary:   "<specific description>"}
    """
    if top_rtl_source:
        dut_slave = _top_has_gpio_pin_boundary(top_rtl_source) and (
            _bus_protocol_says_qspi(project_root)
            or _connections_have_qspi(connections)
        )
    else:
        # RTL not yet assembled (composition time) -> arch-only hint.
        dut_slave = arch_indicates_qspi_slave(project_root, connections)
    master_buses = detect_dut_mastered_buses(top_rtl_source)
    modeled = ["qspi_slave_host"] if dut_slave else []
    unmodeled = [f"dut_mastered_{b}_bus" for b in master_buses]
    if unmodeled:
        summary = (
            "DUT-mastered second bus ("
            + ", ".join(f"{b}_*" for b in master_buses)
            + ") not modeled by the deterministic QSPI-slave TB -- the DUT reads "
            "operands from an external device the host-only BFM does not answer"
        )
    elif dut_slave:
        summary = "QSPI-slave host bus (single-role; fully modeled)"
    else:
        summary = "external bus role not classified"
    return {
        "dut_slave": bool(dut_slave),
        "dut_master_buses": master_buses,
        "modeled": modeled,
        "unmodeled": unmodeled,
        "summary": summary,
    }


def describe_unmodeled_roles(
    project_root: str,
    top_rtl_source: str = "",
    connections: list | None = None,
) -> str:
    """Name the SPECIFIC unmodeled bus role(s), for the advisory / carried-defect.

    Never a generic single-role label: when the DUT masters a second bus, say
    WHICH pin group is unanswered (e.g. "DUT-mastered second bus (rom_*) ...").
    """
    roles = detect_bus_roles(project_root, top_rtl_source, connections)
    return roles["summary"]


# ---------------------------------------------------------------------------
# WHERE the graded boundary lives
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PinBoundary:
    """The module that DECLARES the graded Caravel pad boundary, and where it is.

    ``is_simulated_top`` is the load-bearing field: the deterministic BFM drives
    ``dut.io_in`` / samples ``dut.io_out``, so it can only enforce the bus
    contract when the module the integration sim elaborates is this one.
    """

    module: str
    path: str = ""                       # file it was found in ("" -> from source text)
    ports: tuple[str, ...] = ()          # the pad busses found, sorted
    in_top_source: bool = False          # found in the assembled top's own source
    is_simulated_top: bool = False       # the module the integration sim elaborates

    def describe(self) -> str:
        where = self.path or "the assembled top source"
        return f"module '{self.module}' in {where}"


@dataclass(frozen=True)
class BusVerdict:
    """Full, DUT-blind classification of the chip's external bus + its boundary."""

    status: str
    contract: QSPIContract | None = None
    boundary: PinBoundary | None = None
    spec_says_qspi: bool = False
    connections_say_qspi: bool = False
    simulated_top: str = ""
    reason: str = ""
    top_ports: tuple[str, ...] = ()

    @property
    def contract_enforcing(self) -> bool:
        """True when a deterministic, contract-enforcing DV is actually possible."""
        return self.status == STATUS_QSPI_TOP and self.contract is not None

    @property
    def fails_closed(self) -> bool:
        """True when proceeding would silently keep the DUT-co-tuned LLM BFM.

        Both cases are spec/RTL disagreements, not design choices: either the
        graded pins are on a module the DV does not drive, or the spec claims a
        QSPI bus that no RTL boundary implements.
        """
        return self.status in (STATUS_BOUNDARY_OFF_TOP, STATUS_CONTRADICTION)


def resolve_simulated_top_module(
    top_rtl_source: str, top_module: str = "", top_rtl_path: str = ""
) -> str:
    """The module cocotb will elaborate as ``TOPLEVEL`` for the integration sim.

    MIRRORS ``run_integration_simulation``: the caller-supplied design/top module
    when that module is declared in the top file, else the top file's stem.
    Returns ``""`` when neither is known -- callers then accept a boundary found
    ANYWHERE in the top source (the historical 3-argument behavior).
    """
    src = top_rtl_source or ""
    if top_module and re.search(
        rf"^\s*module\s+{re.escape(top_module)}\b", src, re.MULTILINE
    ):
        return top_module
    if top_rtl_path:
        return Path(top_rtl_path).stem
    return ""


def _wrapper_search_paths(project_root: str, skip: str = "") -> list[Path]:
    """Deterministic candidate list of files that may declare the graded boundary."""
    root = Path(project_root)
    skip_resolved = ""
    if skip:
        try:
            skip_resolved = str(Path(skip).resolve())
        except OSError:
            skip_resolved = skip
    out: list[Path] = []
    seen: set[str] = set()

    def _add(p: Path) -> None:
        try:
            key = str(p.resolve())
        except OSError:
            key = str(p)
        if key in seen or key == skip_resolved or not p.is_file():
            return
        seen.add(key)
        out.append(p)

    for rel in _WRAPPER_CANDIDATES:
        _add(root / rel)
    for pattern in _WRAPPER_GLOBS:
        try:
            hits = sorted(root.glob(pattern))
        except OSError:
            hits = []
        for p in hits:
            if len(out) >= _WRAPPER_SCAN_CAP:
                return out
            _add(p)
    return out


def find_pin_boundary(
    project_root: str,
    top_rtl_source: str = "",
    *,
    top_module: str = "",
    top_rtl_path: str = "",
) -> PinBoundary | None:
    """Locate the GRADED Caravel pad boundary for this project, or None.

    Looks in the order that matters for DV:

    1. the assembled top's own source -- preferring the module the integration
       sim will actually elaborate (that is the only boundary the deterministic
       BFM can drive);
    2. failing that, the project's Caravel wrapper
       (``rtl/user_project_wrapper.v`` and the other conventional locations,
       then a bounded sorted sweep of the RTL tree).

    A candidate counts ONLY if it really declares ``io_in``/``io_out``/``io_oeb``
    as ports with the mandated directions -- never because a comment names them,
    an internal net is called ``io_out``, or a port is *called something like*
    ``qspi_io_in``. Returning a boundary whose ``is_simulated_top`` is False is
    the honest answer "the graded pins exist, but not where DV is looking".

    Step 2 does NOT nominate a replacement top. Its only effect is that the
    fail-closed message can name the file the graded module was written to, which
    points straight at the upstream defect (a pad-adapter block that never got
    instantiated). Callers must not treat an ``is_simulated_top=False`` boundary
    as drivable.
    """
    sim_top = resolve_simulated_top_module(top_rtl_source, top_module, top_rtl_path)
    fallback: PinBoundary | None = None
    for name, text in _iter_modules(top_rtl_source):
        pads = declared_pad_ports(text)
        if len(pads) != len(_PAD_IO_PORTS):
            continue
        is_sim_top = (not sim_top) or name == sim_top
        cand = PinBoundary(
            module=name,
            path=top_rtl_path,
            ports=tuple(sorted(pads)),
            in_top_source=True,
            is_simulated_top=is_sim_top,
        )
        if is_sim_top:
            return cand
        if fallback is None:
            fallback = cand
    if fallback is not None:
        # A pad boundary sits in the top FILE but not on the simulated module --
        # already conclusive (and more relevant than any other wrapper copy).
        return fallback
    for path in _wrapper_search_paths(project_root, skip=top_rtl_path):
        try:
            if path.stat().st_size > _WRAPPER_MAX_BYTES:
                continue
            src = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        mods = _iter_modules(src)
        # Prefer the module literally named user_project_wrapper: on a Caravel
        # chassis that is the graded module by definition.
        for name, text in sorted(
            mods, key=lambda kv: (0 if kv[0] == CARAVEL_TOP_MODULE else 1, kv[0])
        ):
            pads = declared_pad_ports(text)
            if len(pads) != len(_PAD_IO_PORTS):
                continue
            return PinBoundary(
                module=name,
                path=str(path),
                ports=tuple(sorted(pads)),
                in_top_source=False,
                is_simulated_top=bool(sim_top) and name == sim_top,
            )
    return None


def _spec_evidence(project_root: str, connections: list | None) -> str:
    bits = []
    if _bus_protocol_says_qspi(project_root):
        bits.append(".coresmith/{prd,ers}_spec.json dataflow.bus_protocol names QSPI")
    if _connections_have_qspi(connections):
        bits.append("an architecture connection references a qspi interface")
    return "; ".join(bits) or "no QSPI evidence in the architecture artifacts"


_OVERRIDE_NOTE = (
    "Override (records a carried-forward defect instead of failing): "
    "CORESMITH_QSPI_BOUNDARY_GATE=0, or the global CORESMITH_GATE_FAIL_OPEN=1."
)

_WHY_IT_MATTERS = (
    "Without the graded pin boundary the only integration TB available is the "
    "LLM-authored BFM, which CO-TUNES to the DUT: that is exactly how a "
    "non-conformant read serializer passed CoreSmith DV and failed the real "
    "fixed host (the AES bug the deterministic BFM exists to prevent)."
)


def classify_bus_verdict(
    project_root: str,
    top_rtl_source: str,
    connections: list | None = None,
    top_module: str = "",
    top_rtl_path: str = "",
) -> BusVerdict:
    """Classify the external bus AND where its graded boundary lives.

    This is the API a pin-driving caller must use: it distinguishes "not the
    graded boundary" (loud, fail-closed) from "not a QSPI-slave design at all"
    (an honest advisory), instead of collapsing both into ``None``.
    """
    spec_qspi = _bus_protocol_says_qspi(project_root)
    conn_qspi = _connections_have_qspi(connections)
    corroborated = spec_qspi or conn_qspi
    sim_top = resolve_simulated_top_module(top_rtl_source, top_module, top_rtl_path)
    boundary = find_pin_boundary(
        project_root,
        top_rtl_source,
        top_module=top_module,
        top_rtl_path=top_rtl_path,
    )
    top_ports: tuple[str, ...] = ()
    for name, text in _iter_modules(top_rtl_source):
        if not sim_top or name == sim_top:
            top_ports = tuple(declared_port_names(text))
            break

    def _mk(status: str, contract: QSPIContract | None, reason: str) -> BusVerdict:
        return BusVerdict(
            status=status,
            contract=contract,
            boundary=boundary,
            spec_says_qspi=spec_qspi,
            connections_say_qspi=conn_qspi,
            simulated_top=sim_top,
            reason=reason,
            top_ports=top_ports,
        )

    shown_ports = ", ".join(top_ports[:12]) or "(none parsed)"
    if boundary is None:
        if corroborated:
            return _mk(
                STATUS_CONTRADICTION,
                None,
                "SPEC/RTL CONTRADICTION: the architecture declares a QSPI bus ("
                + _spec_evidence(project_root, connections)
                + ") but NO Caravel pin boundary (io_in/io_out/io_oeb declared as "
                "ports) exists anywhere in this project -- not on the simulated "
                f"top '{sim_top or '(unknown)'}' (ports: {shown_ports}) and not in "
                "any of rtl/user_project_wrapper.v, verilog/rtl/"
                "user_project_wrapper.v, rtl/chip_top.v or the rtl/ tree. The "
                "external grader drives io_in/io_out/io_oeb on "
                f"{CARAVEL_TOP_MODULE}, so integration DV cannot enforce the bus "
                "contract on any module that exists. " + _WHY_IT_MATTERS + " Fix "
                "the disagreement: assemble/keep the graded "
                f"{CARAVEL_TOP_MODULE} (pad-adapter block + "
                "CORESMITH_DETERMINISTIC_CARAVEL_TOP=1), or correct the spec's "
                "bus_protocol if this design genuinely has no QSPI pin boundary. "
                + _OVERRIDE_NOTE,
            )
        return _mk(
            STATUS_NOT_QSPI,
            None,
            "not a QSPI-slave chassis: no Caravel pin boundary "
            "(io_in/io_out/io_oeb) is declared anywhere and nothing in the "
            "architecture artifacts claims a QSPI bus "
            f"({_spec_evidence(project_root, connections)}).",
        )
    if not corroborated:
        return _mk(
            STATUS_NOT_QSPI,
            None,
            f"a Caravel pin boundary exists ({boundary.describe()}) but the "
            "architecture artifacts never name a QSPI bus "
            f"({_spec_evidence(project_root, connections)}) -- not classified as "
            "QSPI-slave, so the standard QSPI contract is not imposed.",
        )
    contract = QSPIContract(**_reg_map_overrides(project_root))
    if boundary.is_simulated_top:
        return _mk(
            STATUS_QSPI_TOP,
            contract,
            f"QSPI-slave chassis on the simulated top: {boundary.describe()} "
            f"declares {', '.join(boundary.ports)}.",
        )
    return _mk(
        STATUS_BOUNDARY_OFF_TOP,
        contract,
        "GRADED BOUNDARY IS NOT THE SIMULATED TOP: this IS a QSPI-slave chassis "
        f"({_spec_evidence(project_root, connections)}) and the graded Caravel "
        f"pin boundary is declared by {boundary.describe()} "
        f"({', '.join(boundary.ports)}) -- but integration DV elaborates "
        f"'{sim_top or '(unknown)'}', whose ports are: {shown_ports}. Those are "
        "DIFFERENT pins on a DIFFERENT module: a design-prefixed group such as "
        "qspi_io_in/qspi_io_out/qspi_drive_en is not the graded io_in/io_out/"
        "io_oeb, so the deterministic BFM cannot drive the boundary the grader "
        "drives. " + _WHY_IT_MATTERS + " LIKELY UPSTREAM CAUSE: the graded "
        f"module was written but never instantiated -- the {CARAVEL_TOP_MODULE} "
        "pad-adapter block was dropped during assembly (e.g. its RTL never "
        "resolved from the block spec's rtl_target, so integration_check parsed "
        "N-1 blocks and the deterministic Caravel assembler never ran). Fix "
        f"THERE, by making the graded {CARAVEL_TOP_MODULE} the integration top "
        "(pad-adapter block present + CORESMITH_DETERMINISTIC_CARAVEL_TOP=1) -- "
        "not here: this classifier deliberately does NOT retarget DV at the "
        f"wrapper it found in {boundary.path or 'the top source'}, because "
        "driving a module the assembled chip does not use would hide the "
        "mis-assembly. " + _OVERRIDE_NOTE,
    )


def classify_chip_bus(
    project_root: str,
    top_rtl_source: str,
    connections: list | None = None,
    top_module: str = "",
    top_rtl_path: str = "",
) -> QSPIContract | None:
    """Return a QSPIContract if this chip's external bus is QSPI-slave, else None.

    Decision (all DUT-blind):
      * REQUIRED: the GRADED Caravel ``io_in``/``io_out``/``io_oeb`` pin boundary
        is declared somewhere in the project -- on the assembled top, or (when
        the top's own pins are design-prefixed) in the project's Caravel wrapper.
        A wrapper counts only if it really declares the three pad busses.
      * plus at least one corroborating signal: the PRD/ERS bus_protocol names
        QSPI, or an architecture connection references a qspi interface.

    NOTE for pin-driving callers: a contract here does NOT promise that the
    module your sim elaborates exposes the pads. Use :func:`classify_bus_verdict`
    and honor ``boundary.is_simulated_top`` / ``verdict.fails_closed``.
    """
    verdict = classify_bus_verdict(
        project_root,
        top_rtl_source,
        connections,
        top_module=top_module,
        top_rtl_path=top_rtl_path,
    )
    return verdict.contract
