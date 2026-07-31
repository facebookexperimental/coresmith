# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Resolve memory macros BEFORE synthesis and emit a bound shell.

Why this exists
---------------
``cs_sram.v`` selects its implementation with a ``MEM_IMPL`` parameter, and the
``"MACRO"`` arm instantiates ``cs_mem_macro_shell`` -- a leaf that *hard-assigns
zero*::

    // Reads are 0 in pure RTL sim -- "MACRO" is a backend impl, not a sim model
    assign rdata0 = {WIDTH{1'b0}};

Synthesis faithfully preserves that tie-off, so the flat netlist's memories read
zero and a gate-level simulation of a MACRO-mode design can never pass: the
first read of real data diverges. Macro *binding* today runs on the finished
netlist (``bind_macro_shells``) and only records collateral for PnR -- it never
gives the shell a body.

The fix is ordering. Resolve each shell geometry to a concrete macro *before*
synthesis, then emit a ``cs_mem_macro_shell`` whose body instantiates that
macro. One artifact then serves every consumer:

===============  =============================================================
yosys            ``read_verilog -lib <macro>.v`` -- interface only, 0 flops,
                 exactly the property the zero tie-off was protecting
PnR              LEF/GDS keyed to the macro name already in the netlist
gate-sim         the same ``<macro>.v`` read normally -- a real memory
integration DV   ditto, so DV and silicon share one memory model
===============  =============================================================

An unresolvable geometry must never degrade to zeros or to a flop array, so the
fallback arm references a deliberately undefined module: both yosys and
Verilator fail on an unknown module, which makes the failure loud in every tool
rather than silent in one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Undefined on purpose -- see module docstring. Any tool elaborating this arm
# errors out, which is the whole point.
UNBOUND_SENTINEL = "cs_mem_macro_shell__UNBOUND_GEOMETRY__see_macro_prebind_py"


#: Memory wrappers a block may instantiate, mapped to their port count.
#: ``detect_macro_shells`` only matches ``cs_*_macro_shell`` instantiations,
#: which exist solely AFTER the backend re-derives MEM_IMPL="MACRO" -- block
#: RTL instantiates the wrapper instead (``cs_sram_1rw1r #(.WIDTH(8),
#: .DEPTH(4096))``), so pre-synthesis binding has to read that form.
_WRAPPERS = {
    "cs_sram_1rw1r": 2,
    "cs_mem_1rw1r": 2,
    "cs_sram_1rw": 1,
    "cs_mem_1rw": 1,
}


def _balanced(text: str, open_at: int) -> tuple[str, int]:
    """Return the text inside the parens starting at ``open_at`` and the index
    just past the closer. Parameter lists contain nested parens (``.WIDTH(8)``),
    so a non-greedy regex would stop at the first inner ``)``."""
    depth = 0
    for i in range(open_at, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return text[open_at + 1:i], i + 1
    return "", len(text)


def _param(params: str, name: str) -> int | None:
    import re
    m = re.search(r"\.\s*%s\s*\(\s*([0-9]+)\s*\)" % name, params)
    return int(m.group(1)) if m else None


def detect_memory_instances(text: str) -> list[tuple[int, int, int, int]]:
    """Every ``(width, depth, nport, mask_bits)`` the source instantiates.

    Reads the *wrapper* form used by block RTL. ``mask_bits`` is derived from
    ``USE_WMASK``/``WMASK_GRAN`` so a masked write binds to a macro that has a
    ``wmask0`` port. Deduped by geometry, order-stable.
    """
    import re

    out: list[tuple[int, int, int, int]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for name, nport in _WRAPPERS.items():
        # `\b` on both sides: cs_sram_1rw must not match inside cs_sram_1rw1r.
        for m in re.finditer(r"\b%s\b\s*#\s*\(" % re.escape(name), text):
            params, after = _balanced(text, m.end() - 1)
            w, d = _param(params, "WIDTH"), _param(params, "DEPTH")
            if not w or not d:
                continue
            mask = 0
            if (_param(params, "USE_WMASK") or 0) == 1:
                gran = _param(params, "WMASK_GRAN") or 8
                mask = (w + gran - 1) // gran
            key = (w, d, nport, mask)
            if key not in seen:
                seen.add(key)
                out.append(key)
    return out


@dataclass
class PrebindResult:
    """Outcome of pre-synthesis macro resolution."""

    bindings: list = field(default_factory=list)      # [(ShellSpec, MacroInfo)]
    # geometry key (w, d, nport) -> write-mask lanes the RTL drives. 0/1 mean a
    # whole-word mask. Needed by emit_bound_shell to size the shell's wmask0 and
    # to decide whether the macro's own mask port can carry it.
    mask_lanes: dict = field(default_factory=dict)
    unresolved: list = field(default_factory=list)    # [ShellSpec]
    errors: list = field(default_factory=list)        # [str]
    warnings: list = field(default_factory=list)      # [str] -- do not block

    @property
    def ok(self) -> bool:
        """True only when every detected geometry bound to a real macro."""
        return not self.unresolved and not self.errors

    def model_paths(self) -> list[str]:
        """Macro Verilog models, deduped, order-stable.

        Read these with ``-lib`` for synthesis and normally for simulation.
        """
        out: list[str] = []
        for _spec, macro in self.bindings:
            v = getattr(macro, "verilog", "") or ""
            if v and v not in out:
                out.append(v)
        return out


#: Type/sign/net keywords that may appear between a direction keyword and the
#: port name. Anything else at depth zero is a port name.
_PORT_NOISE = frozenset({
    "wire", "reg", "logic", "bit", "signed", "unsigned", "tri", "tri0", "tri1",
    "wand", "wor", "triand", "trior", "trireg", "supply0", "supply1", "var",
    "integer", "real", "realtime", "time", "byte", "shortint", "int", "longint",
    "input", "output", "inout",
})


def _strip_comments(text: str) -> str:
    """Remove // and /* */ comments so prose cannot contribute a port name."""
    import re
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//[^\n]*", " ", text)


def macro_ports(verilog_path) -> set[str]:
    """Port names the macro's Verilog model actually declares.

    Trust the model, not registry metadata: ``MacroInfo.mask_bits`` can be
    non-zero for a macro whose ``.v`` has no ``wmask0`` port, and connecting a
    pin that does not exist is a hard elaboration error.

    Scanned rather than regexed, because this parser MUST NOT fail by omission.
    A missing name here is read by ``_ports_for`` as "the macro has no such pin",
    so the binder leaves it unconnected and it floats low -- a dropped ``addr0``
    is a memory stuck at word zero, and a dropped ``wmask0`` is what suppressed
    every framebuffer write in exp-raster-macro-20260727. The previous regex
    (``[^;)]*``) could not cross a ``)``, so it lost the name in
    ``input [$clog2(DEPTH)-1:0] addr0;`` -- a form the sky130 macro family and
    plenty of generated models use.

    Handles non-ANSI (``input clk0;``), multi-name (``input a, b;``) and ANSI
    (``module m(input wire [3:0] a, output b);``) declarations.

    CONTRACT: the result is a SUPERSET of the module's ports, never a subset.
    Function argument declarations inside the module body are collected too --
    the sky130 SRAM models declare ``input [DATA_WIDTH-1:0] new_word;`` inside
    ``merge_write0``, so ``write_mask``/``new_word``/``old_word`` come back as
    well. That is deliberate and safe for the consumer that matters:
    ``_ports_for`` connects a pin only if it is BOTH in its own connection list
    AND in this set, so a spurious extra name can never create a connection,
    while a missing real name silently drops one. Do not use this set as an
    authoritative port list for anything that must be exact -- read the module
    header for that.
    """
    import re
    try:
        text = _strip_comments(Path(verilog_path).read_text(errors="ignore"))
    except OSError:
        return set()

    ports: set[str] = set()
    for m in re.finditer(r"\b(?:input|output|inout)\b", text):
        depth_paren = depth_brack = 0
        chunk: list[str] = []
        i = m.end()
        while i < len(text):
            c = text[i]
            if c == "[":
                depth_brack += 1
            elif c == "]":
                depth_brack = max(0, depth_brack - 1)
            elif c == "(":
                depth_paren += 1
            elif c == ")":
                if depth_paren == 0:
                    break          # closing paren of an ANSI port list
                depth_paren -= 1
            elif c == ";" and depth_paren == 0 and depth_brack == 0:
                break              # end of a non-ANSI declaration
            elif depth_paren == 0 and depth_brack == 0:
                chunk.append(c)
            i += 1
        for tok in re.findall(r"[A-Za-z_]\w*", "".join(chunk)):
            if tok not in _PORT_NOISE and not tok.startswith("$"):
                ports.add(tok)
    return ports


def macro_mask_lanes(verilog_path) -> int | None:
    """Write-mask lanes the macro's Verilog actually declares, or None.

    OpenRAM emits ``parameter NUM_WMASKS = N;`` next to
    ``input [NUM_WMASKS-1:0] wmask0;``. A literal range is accepted too. Returns
    None when the macro declares no ``wmask0`` OR when the width cannot be
    resolved -- callers must treat None as "cannot verify" and refuse, never as
    a default, because this number decides which bytes a write touches.

    Deliberately does NOT consult ``MacroInfo.mask_bits``: that field reports
    the write granularity for some macros and the lane count for others, and it
    read 8 for a macro whose mask is 2 bits wide.
    """
    import re
    try:
        text = _strip_comments(Path(verilog_path).read_text(errors="ignore"))
    except OSError:
        return None
    if not re.search(r"\bwmask0\b", text):
        return None
    m = re.search(r"\bparameter\b[^;]*?\bNUM_WMASKS\b\s*=\s*(\d+)", text)
    if m:
        return int(m.group(1))
    # Literal range, e.g. `input [3:0] wmask0;`
    m = re.search(r"\b(?:input|output|inout)\b[^;()]*\[\s*(\d+)\s*:\s*(\d+)\s*\][^;()]*\bwmask0\b",
                  text)
    if m:
        return abs(int(m.group(1)) - int(m.group(2))) + 1
    # Declared, but the width is an expression we will not evaluate.
    return None


def _ports_for(macro, spec, nmask: int = 1) -> list[tuple[str, str]]:
    """Port connections from the OpenRAM macro to the shell's signals.

    OpenRAM sky130 SRAMs use ACTIVE-LOW chip-select and write-enable
    (``csb0``/``web0``), while the shell's ``ce0``/``we0`` are active high --
    so these are inverted, not renamed. Getting that backwards yields a memory
    that is selected exactly when it should be idle, which reads as plausible
    garbage rather than an obvious failure.
    """
    have = macro_ports(getattr(macro, "verilog", "") or "")
    conns = [
        ("clk0", "clk"),
        ("csb0", "~ce0"),
        ("web0", "~we0"),
        ("addr0", "addr0"),
        ("din0", "wdata0"),
        ("dout0", "rdata0"),
    ]
    if "wmask0" in have:
        # Connect the SHELL's mask, not a constant. Tying this high was the
        # defect: it silently turned every partial write into a full-word write.
        conns.insert(3, ("wmask0", "wmask0"))
    else:
        # The macro has no mask port at all -- OpenRAM omits it when
        # word_size == write_size, i.e. the mask is a single whole-word bit.
        # That bit is exactly a write-enable qualifier, so fold it into web0
        # here. Doing it structurally means no design has to remember to write
        # `we0 = write_fire & wmask` by hand (framebuffer_sram had to be
        # patched by hand to do precisely this).
        for i, (port, _sig) in enumerate(conns):
            if port == "web0":
                conns[i] = ("web0", "~(we0 & (&wmask0))")
                break
    if int(getattr(spec, "nport", 1) or 1) >= 2:
        conns += [
            ("clk1", "clk"),
            ("csb1", "~ce1"),
            ("addr1", "addr1"),
            ("dout1", "rdata1"),
        ]
    # Never connect a pin the model does not declare -- that is a hard
    # elaboration error, and it is how the 4096x8 framebuffer macro (no
    # wmask0 port) was caught.
    return [(p, s) for p, s in conns if not have or p in have]


def emit_bound_shell(result: PrebindResult) -> str:
    """Verilog defining ``cs_mem_macro_shell`` bound to concrete macros.

    Dispatch is a ``generate`` over the elaborated WIDTH/DEPTH/NPORT, so one
    module serves every geometry in the design and the selection is visible to
    both simulator and synthesiser -- no preprocessor, matching the reasoning
    already documented in ``cs_sram.v``.
    """
    lines = [
        "// GENERATED by orchestrator/langgraph/macro_prebind.py -- do not edit.",
        "//",
        "// Replaces the zero-driving cs_mem_macro_shell with one bound to the",
        "// concrete macros resolved for this design. Synthesis should read the",
        "// macro models with `read_verilog -lib` (interface only, 0 flops);",
        "// simulation reads them normally to get a real memory.",
        "",
        "module cs_mem_macro_shell #(",
        "    parameter integer WIDTH = 32,",
        "    parameter integer DEPTH = 512,",
        "    parameter integer NPORT = 1,",
        "    parameter integer NMASK = 1,",
        "    parameter integer AW    = (DEPTH <= 1) ? 1 : $clog2(DEPTH)",
        ") (",
        "    input  wire             clk,",
        "    input  wire             ce0,",
        "    input  wire             we0,",
        "    input  wire [NMASK-1:0] wmask0,",
        "    input  wire [AW-1:0]    addr0,",
        "    input  wire [WIDTH-1:0] wdata0,",
        "    output wire [WIDTH-1:0] rdata0,",
        "    input  wire             ce1,",
        "    input  wire [AW-1:0]    addr1,",
        "    output wire [WIDTH-1:0] rdata1",
        ");",
        "  generate",
    ]

    branch = "if"
    for spec, macro in result.bindings:
        w, d = int(spec.width), int(spec.depth)
        n = int(getattr(spec, "nport", 1) or 1)
        lines.append(
            f"    {branch} (WIDTH == {w} && DEPTH == {d} && NPORT == {n}) begin : "
            f"g_w{w}_d{d}_p{n}"
        )
        lines.append(f"      {macro.name} u_macro (")
        key = (w, d, n)
        nmask = max(1, int(result.mask_lanes.get(key, 1) or 1))
        conns = _ports_for(macro, spec, nmask)
        for i, (port, sig) in enumerate(conns):
            comma = "," if i < len(conns) - 1 else ""
            lines.append(f"        .{port}({sig}){comma}")
        lines.append("      );")
        if n < 2:
            # Single-port macro: the shell still exposes rdata1. Drive it from
            # port 0 rather than inventing data.
            lines.append("      assign rdata1 = rdata0;")
        lines.append("    end")
        branch = "else if"

    lines += [
        "    else begin : g_unbound",
        "      // No macro was resolved for this geometry. Referencing an",
        "      // undefined module makes yosys AND Verilator fail here, rather",
        "      // than silently reading zeros (the old behaviour) or inferring",
        "      // a flop array (worse).",
        f"      {UNBOUND_SENTINEL} u_unbound_geometry ();",
        "    end",
        "  endgenerate",
        "endmodule",
        "",
    ]
    return "\n".join(lines)


def resolve_prebindings(sources, *, allow_generate: bool = True,
                        registry=None) -> PrebindResult:
    """Detect every macro-shell geometry in RTL SOURCE and bind each one.

    ``detect_macro_shells`` already accepts source as well as netlists (its
    explicit-``#(.WIDTH..)`` form is the behavioral wrapper), so no new parsing
    is needed -- only running it earlier.
    """
    res = PrebindResult()
    try:
        from orchestrator.langgraph.macro_registry import (
            detect_macro_shells,
            discover_macros,
            resolve_shell,
        )
    except Exception as exc:  # pragma: no cover - import guard
        res.errors.append(f"macro_registry unavailable: {exc!r}")
        return res

    text_parts = []
    for s in sources:
        p = Path(s)
        if p.is_file():
            text_parts.append(p.read_text(errors="ignore"))
    if not text_parts:
        res.errors.append("no readable RTL sources given")
        return res
    text = "\n".join(text_parts)

    # Union of both forms: post-derivation shells (netlists / already-derived
    # sources) and the wrapper instantiations block RTL actually writes.
    specs = list(detect_macro_shells(text))
    have = {(int(s.width), int(s.depth), int(getattr(s, "nport", 1) or 1))
            for s in specs}
    req_mask: dict[tuple[int, int, int], int] = {}
    try:
        from orchestrator.langgraph.macro_registry import ShellSpec
        for w, d, n, mask in detect_memory_instances(text):
            if mask:
                req_mask[(w, d, n)] = mask
            if (w, d, n) not in have:
                have.add((w, d, n))
                specs.append(ShellSpec(kind="sram", width=w, depth=d, nport=n))
    except Exception as exc:  # pragma: no cover - construction guard
        res.errors.append(f"could not build ShellSpec for wrapper form: {exc!r}")

    if not specs:
        return res          # genuinely no memories; nothing to bind

    if registry is None:
        registry = discover_macros()

    for spec in specs:
        macro = resolve_shell(spec, registry=registry,
                              allow_generate=allow_generate)
        if macro is None or not getattr(macro, "name", ""):
            res.unresolved.append(spec)
            continue
        if not getattr(macro, "verilog", ""):
            # A macro with no simulation model cannot be gate-simulated. Say so
            # here rather than letting the sim read an unbound instance.
            res.unresolved.append(spec)
            res.errors.append(
                f"macro {macro.name} for {spec.describe()} has no Verilog model "
                f"-- cannot simulate; regenerate it so a .v view exists")
            continue
        key = (int(spec.width), int(spec.depth),
               int(getattr(spec, "nport", 1) or 1))
        if key in req_mask:
            nb = req_mask[key]
            macro_has_mask = "wmask0" in macro_ports(macro.verilog)
            result_lanes = nb
            if macro_has_mask and nb > 1:
                # The shell routes the mask to the macro's own port, so this
                # binds -- PROVIDED both sides agree on lane count. OpenRAM
                # lanes = ceil(width / write_size); a disagreement would write
                # the wrong bytes, which is the corruption we are removing.
                macro_lanes = macro_mask_lanes(macro.verilog)
                if macro_lanes != nb:
                    res.unresolved.append(spec)
                    res.errors.append(
                        f"{spec.describe()}: RTL drives {nb} write-mask lane(s) "
                        f"but macro {macro.name} declares "
                        f"{macro_lanes if macro_lanes is not None else 'an unresolvable number of'}"
                        f" lane(s). Connecting mismatched lanes would write the "
                        f"wrong bytes, so this is refused rather than truncated "
                        f"(the previous binder replicated a constant and let "
                        f"yosys truncate it). Regenerate the macro with a "
                        f"write_size that yields {nb} lanes.")
                    continue
                res.mask_lanes[key] = nb
                res.bindings.append((spec, macro))
                continue
            if nb > 1:
                # A real per-byte mask survives ONLY if it can travel all the way
                # from the RTL to the macro. Two independent things can break
                # that, and both end the same way -- the mask is replaced by
                # all-ones, every partial write becomes a full-word write, and
                # neighbouring bytes are clobbered. That is data corruption which
                # passes RTL DV (the BEHAV arm honours the mask) and appears only
                # in the macro-backed design, so it must be refused here.
                if macro_has_mask:
                    # The macro could honour it; the SHELL cannot express it.
                    # emit_bound_shell's cs_mem_macro_shell has no mask pins and
                    # _ports_for hardwires wmask0 to all-ones.
                    detail = (
                        f"macro {macro.name} HAS a wmask0 port, but "
                        f"cs_mem_macro_shell carries no mask pins and the "
                        f"binding ties wmask0 to all-ones, so the RTL's mask is "
                        f"discarded. Route wmask0 through the shell (shell port "
                        f"+ _ports_for connection + the cs_sram MACRO arm), or "
                        f"change the RTL to fold the mask into we0")
                else:
                    detail = (
                        f"macro {macro.name} has no wmask0 port at all. "
                        f"Regenerate this geometry with a smaller write_size so "
                        f"OpenRAM emits a mask, or change the RTL to "
                        f"read-modify-write")
                res.unresolved.append(spec)
                res.errors.append(
                    f"{spec.describe()}: RTL instantiates USE_WMASK with {nb} "
                    f"mask bits and the mask would NOT reach the memory -- "
                    f"{detail}. Refusing to bind: a dropped mask turns masked "
                    f"writes into full-word writes, which RTL DV cannot see.")
                continue
            # nb == 1: a whole-word mask, exactly equivalent to a write-enable
            # qualifier. The shell now folds it into web0 itself, so this is
            # correct by construction and no longer depends on the RTL author
            # remembering to write `we0 = write_fire & wmask`.
            res.mask_lanes[key] = 1
            res.warnings.append(
                f"{spec.describe()}: whole-word write mask folded into the "
                f"macro's write enable by the bound shell"
                f"{'' if macro_has_mask else f' (macro {macro.name} exposes no wmask0)'}"
                f" -- masked writes are suppressed correctly without any RTL "
                f"change.")
        res.bindings.append((spec, macro))
    return res


def strip_module(text: str, name: str) -> tuple[str, int]:
    """Remove every ``module <name> ... endmodule`` definition. Returns
    ``(text, removed_count)``.

    Used to drop the zero-driving ``cs_mem_macro_shell`` from a COPY of the
    wrapper library so the bound one can take its place. The shared
    ``cs_sram.v`` is never modified -- other flows still get the documented
    "MACRO reads zero in pure RTL sim" behaviour.
    """
    import re
    pat = re.compile(r"\bmodule\s+%s\b.*?\bendmodule\b" % re.escape(name),
                     re.DOTALL)
    out, n = pat.subn("", text)
    return out, n


def prepare_synth_sources(wrapper_lib: str, result: PrebindResult,
                          work_dir) -> dict:
    """Source set for a synthesis that should use real macros.

    Returns ``{"wrapper_lib", "bound_shell", "models"}``:

    * ``wrapper_lib`` -- a copy of the library with the zero-driving shell
      removed (or the original path when there was nothing to strip)
    * ``bound_shell`` -- the generated shell instantiating concrete macros
    * ``models`` -- macro Verilog to be read with ``-lib`` (interface only)

    Reading both the original library and the bound shell would be a duplicate
    definition of ``cs_mem_macro_shell``, so the strip is not optional.
    """
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    out = {"wrapper_lib": wrapper_lib, "bound_shell": "", "models": []}
    if not result.bindings:
        return out

    if wrapper_lib and Path(wrapper_lib).is_file():
        text = Path(wrapper_lib).read_text(errors="ignore")
        stripped, n = strip_module(text, "cs_mem_macro_shell")
        if n:
            filtered = work / "cs_sram__prebind.v"
            filtered.write_text(
                "// GENERATED: copy of %s with the zero-driving\n"
                "// cs_mem_macro_shell removed; the bound shell replaces it.\n"
                % wrapper_lib + stripped)
            out["wrapper_lib"] = str(filtered)

    out["bound_shell"] = write_bound_shell(
        result, work / "cs_mem_macro_shell__bound.v")
    out["models"] = result.model_paths()
    return out


def write_bound_shell(result: PrebindResult, out_path) -> str:
    """Write the bound shell next to the design. Returns the path written."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(emit_bound_shell(result))
    return str(out)
