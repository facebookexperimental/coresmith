# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Does a block's RTL expose the ports its interface contract declares?

The contract already says exactly what every channel's signals are called. Each
edge in ``.coresmith/interface_contracts.json`` carries ``fields[]`` (the
payload) and ``sideband_signals[]`` (everything else); their UNION is the port
set. A block may expose a declared signal either as ``<channel>_<field>`` or
as bare ``<field>`` -- both styles are in use, and the prefixed form is what
disambiguates a generic name shared by two channels. Nothing checked that the
generated RTL agreed with the contract at all, and it frequently does not:

    contract framebuffer_read: rdata + req_addr, read_enable, rvalid, fault
    framebuffer_sram  ->  framebuffer_read_read_enable      CONFORMS
    control_status_aperture -> framebuffer_read_enable      DEVIATES

The consequences are not cosmetic. The deterministic Caravel assembler resolves
contract edges by name; a deviating port makes the edge unresolvable, the
assembler refuses (correctly -- it will not guess at wiring), and the whole
design falls back to an LLM-authored top. That is how a chip shipped with no
graded pin boundary at all.

Two observations decided the shape of this check.

* **The deviation is not consistently on one side.** Measured across two runs:
  3 producer-side and 1 consumer-side in the first, plus a block that dropped
  channel prefixes entirely in the second. So a stitcher-side heuristic cannot
  repair it -- both ends must conform.
* **It is systematic, not sampling noise.** It survived a full fresh-context
  re-spec: the regenerated aperture still emitted ``host_write_enable`` ten
  times. ``contract_lookup`` already injects the contract with the instruction
  "sideband signals ... MUST match in your output exactly. Do not invent new
  fields", and the generator collapsed the duplicated token anyway. A prompt
  rule with no gate is not enforcement.

So this is deterministic and it runs per block, right after RTL generation --
not at integration. An integration-time failure has already paid for seven other
blocks; a block-time failure is one cheap regeneration with an exact expected
name to fix.

:func:`run_conformance_stage` is the production entry point the per-block RTL
flow calls (see ``generate_testbench_node``): check -> unambiguous repair ->
re-check, with the testbench's port references carried along. Everything here is
pure file I/O and returns a record; the caller owns logging, the block verdict
and the park.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

#: A block whose module declares the full Caravel pad boundary has externally
#: MANDATED port names (io_in/io_out/io_oeb[37:0] are fixed by the shuttle), so
#: the <channel>_<field> convention cannot apply to it.
_LOCKED_BOUNDARY_PORTS = ("io_in", "io_out", "io_oeb")

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Deriving a canonical port name from contract text
# ---------------------------------------------------------------------------
#
# The contract's channel and signal strings are NOT always bare identifiers.
# The schema template the interface-definition agent is handed literally reads
#
#     "producer_port": "<m_axis_<name> or m_<name>_srdy/m_<name>_data>"
#
# so a real contract carries either a bare channel or a SLASH-SEPARATED
# ENUMERATION of the concrete ports on that channel -- and a sideband is
# sometimes declared under two names ("srdy/in_valid" = the generic protocol
# role plus the spelling mandated on that edge). Concatenating those strings
# verbatim into `<channel>_<field>` emitted port names containing `/`. No legal
# UNESCAPED Verilog identifier can contain one, so the repair pass rewrote six
# blocks' RTL to names nothing downstream can parse and the loop could never
# converge. Everything below exists so a `/` can never reach an identifier.

#: A legal, unescaped Verilog identifier. `\s_read_req/addr ` is legal only as
#: an ESCAPED identifier, which no stitcher / cocotb handle lookup / netlist
#: reader in this flow handles -- so it is not an acceptable port name here.
_LEGAL_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")


def is_legal_identifier(name) -> bool:
    """Whether ``name`` may be emitted as an unescaped Verilog identifier."""
    return bool(_LEGAL_IDENT_RE.match(str(name or "")))


#: Illegal derivations already reported, so a 25-edge contract logs each defect
#: once instead of once per block that touches it.
_ILLEGAL_REPORTED: set = set()


def _report_illegal(edge: dict, chan_raw, signal, derived: str) -> None:
    """Loudly refuse a derived name that is not an identifier (never emit it).

    Skipping is the only safe action: the alternative is rewriting a block's
    RTL -- and its testbench -- to a name that cannot be declared, which is
    what corrupted six blocks. It is logged at ERROR *and* on stderr because
    the caller of this module owns logging and would otherwise swallow it.
    """
    key = (str(chan_raw), str(signal), derived)
    if key in _ILLEGAL_REPORTED:
        return
    _ILLEGAL_REPORTED.add(key)
    msg = (
        "contract signal DROPPED from the port set -- the derived name is not "
        f"a legal Verilog identifier. edge={edge.get('edge_id')!r} "
        f"channel={chan_raw!r} signal={signal!r} derived={derived!r}. Fix the "
        "CONTRACT (a channel/signal must reduce to an identifier); the RTL is "
        "not at fault and must not be rewritten to this name."
    )
    _logger.error(msg)
    print(f"[contract-conformance] {msg}", file=sys.stderr)


def channel_base(raw) -> str:
    """The channel PREFIX named by ``producer_port`` / ``consumer_port``.

    A bare value is the channel. A slash-separated value ENUMERATES the
    concrete ports on the channel; all three shapes occur in one 12-block
    design::

        m_coeff_out_srdy/m_coeff_out_data   -> m_coeff_out   (both qualified)
        m_transform_read_req/addr           -> m_transform_read  (tail bare)
        in_valid/in_data/in_last            -> in            (short prefix)

    So the channel is the segments' longest common ``_``-token prefix. When no
    later segment shares a leading token with the first (the ``.../addr``
    shape), the segments are suffixes of one base and the first segment's
    trailing token is its own signal, so the channel is everything before it.
    """
    segs = [s.strip() for s in str(raw or "").strip().split("/") if s.strip()]
    if not segs:
        return ""
    if len(segs) == 1:
        return segs[0]
    head = segs[0].split("_")
    common = None
    for seg in segs[1:]:
        toks = seg.split("_")
        n = 0
        while n < min(len(head), len(toks)) and head[n] == toks[n]:
            n += 1
        if n:
            common = n if common is None else min(common, n)
    if common:
        return "_".join(head[:common])
    return "_".join(head[:-1]) or segs[0]


def signal_name(raw, chan: str = "") -> str:
    """One concrete signal name from a (possibly slash-aliased) declaration.

    ``srdy/in_valid`` declares ONE wire under two names -- the generic
    handshake role and the spelling this edge mandates -- and only one of them
    can be a port. Prefer the segment that already carries the channel prefix
    (that is this end's fully-qualified spelling: on channel ``in`` the
    contract's own port list says ``in_valid``, on channel ``s_source`` it says
    ``s_source_srdy``), otherwise the first.
    """
    segs = [s.strip() for s in str(raw or "").split("/") if s.strip()]
    if not segs:
        return ""
    if chan:
        for seg in segs:
            if seg == chan or seg.startswith(chan + "_"):
                return seg
    return segs[0]


def canonical_port(chan: str, signal) -> tuple[str, str]:
    """``(port, bare)`` -- the canonical flattened name for one channel signal.

    ``<channel>_<signal>``, IDEMPOTENT in the channel prefix: a signal the
    contract already spells with the channel on it (``in_last`` on channel
    ``in``) is not prefixed a second time. Without that the derivation emitted
    ``s_chan_s_chan_addr`` and then demanded the RTL rename to it.

    The doubled-TOKEN rule is untouched, because it is a different thing:
    channel ``data_write`` + signal ``write_enable`` is still
    ``data_write_write_enable`` -- ``write_enable`` does not start with
    ``data_write_``. Only a whole-prefix repeat is idempotent.

    ``bare`` is the unprefixed spelling a block may legitimately use instead;
    it equals ``port`` when the signal already carries the prefix (there is
    then only ONE acceptable name, not two).
    """
    sig = signal_name(signal, chan)
    if not sig:
        return "", ""
    if not chan or sig == chan or sig.startswith(chan + "_"):
        return sig, sig
    return f"{chan}_{sig}", sig


def channel_signals(edge: dict, chan_raw) -> list[dict]:
    """Canonical port rows for ONE END of a contract edge.

    THE single derivation: :func:`check_block` (the gate) and
    :func:`contract_port_rows` (what the RTL generator is shown) both go
    through it, so the gate can never demand a spelling the prompt did not
    advertise -- and a slash-alias cannot be fixed in one and not the other.

    Returns ``[{channel, signal, port, bare, width, dir, kind,
    doubled_token}]``, de-duplicated by port. A row whose derived name is not a
    legal identifier is DROPPED and reported; nothing downstream ever sees it.
    """
    chan = channel_base(chan_raw)
    if chan and not is_legal_identifier(chan):
        _report_illegal(edge, chan_raw, "<channel>", chan)
        return []
    rows: list[dict] = []
    seen: set = set()
    for spec in signal_specs(edge):
        port, bare = canonical_port(chan, spec["name"])
        if not port:
            continue
        if not is_legal_identifier(port):
            _report_illegal(edge, chan_raw, spec["name"], port)
            continue
        if port in seen:
            continue
        seen.add(port)
        chan_tail = chan.rsplit("_", 1)[-1]
        sig_head = bare.split("_", 1)[0]
        rows.append({
            "channel": chan,
            "signal": bare,
            "port": port,
            "bare": bare,
            "width": spec["width"],
            "dir": spec["dir"],
            "kind": spec["kind"],
            "doubled_token": bool(chan_tail) and chan_tail == sig_head,
        })
    return rows


@dataclass
class ConformanceResult:
    """Per-block verdict. ``ok`` only when every declared signal is present."""

    block: str = ""
    checked_edges: int = 0
    missing: list = field(default_factory=list)     # [(channel, expected_port)]
    undeclared: list = field(default_factory=list)  # [port] -- <channel>_* not in the contract
    ambiguous: list = field(default_factory=list)   # [(channel, explanation)]
    instantiates: list = field(default_factory=list)  # sibling blocks wired in
    accounted: set = field(default_factory=set)     # ports bound to a declared signal
    ports: set = field(default_factory=set)         # every port the module declares
    exempt: bool = False
    locked_boundary: bool = False   # carries externally-mandated pad names
    reason: str = ""

    @property
    def ok(self) -> bool:
        return not (self.missing or self.undeclared or self.ambiguous
                    or self.instantiates)

    def as_feedback(self) -> str:
        """An actionable message for the RTL generator.

        States the EXACT required name. The failure mode this guards is a
        generator collapsing a duplicated token (channel ``host_write`` +
        signal ``write_enable`` -> ``host_write_enable``), so a vague "port
        names must match the contract" would not change the outcome -- the
        prompt already says that.
        """
        lines = []
        if self.instantiates:
            lines.append(
                "This block INSTANTIATES other blocks: "
                + ", ".join(self.instantiates)
                + ". It is a leaf, not the chip top. Delete those "
                "instantiations and expose the channel signals below as PORTS "
                "instead -- the integration stage wires the blocks together, "
                "and a block that assembles the design itself cannot be wired "
                "into anything.")
        if self.missing:
            lines.append(
                "The interface contract declares these ports and the RTL does "
                "not expose them. Use these names EXACTLY -- do not shorten a "
                "repeated word (channel 'host_write' + signal 'write_enable' is "
                "'host_write_write_enable', NOT 'host_write_enable'), and do "
                "not drop the channel prefix:")
            for chan, port in self.missing:
                lines.append(f"  - {port}    (channel '{chan}')")
        if self.ambiguous:
            lines.append(
                "These channel signals cannot be resolved by name. Give each "
                "declared signal exactly one port, and prefix it with the "
                "channel when two channels share a signal name:")
            for chan, why in self.ambiguous:
                lines.append(f"  - channel '{chan}': {why}")
        if self.undeclared:
            lines.append(
                "These ports use a channel prefix but are not in the contract. "
                "Either they are misspellings of a declared signal, or they are "
                "new signals that the contract does not permit:")
            for p in self.undeclared:
                lines.append(f"  - {p}")
        return "\n".join(lines)


def _load_contracts(project_root) -> list[dict]:
    p = Path(project_root) / ".coresmith" / "interface_contracts.json"
    try:
        d = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    c = d.get("contracts") if isinstance(d, dict) else d
    return c if isinstance(c, list) else []


def signal_specs(edge: dict) -> list[dict]:
    """Every signal on a channel, with what the contract says about it.

    The payload fields plus the sideband -- split across two keys in the
    schema, which is precisely why several readers concluded the contract did
    not record them at all. Their UNION is the port set.

    Returns ``[{"name", "width", "dir", "kind"}]`` in contract order,
    de-duplicated by name. ``width``/``dir`` are ``""`` when the contract does
    not state them. This is the single union implementation: both the
    conformance checker and the prompt-side port table read it, so the
    generator can never be shown a port set the gate does not check.
    """
    out: list[dict] = []
    for kind, key in (("field", "fields"), ("sideband", "sideband_signals")):
        for f in edge.get(key) or []:
            if isinstance(f, dict):
                name = f.get("name")
                width = f.get("width")
                if width is None:
                    msb, lsb = f.get("msb"), f.get("lsb")
                    if isinstance(msb, int) and isinstance(lsb, int):
                        width = abs(msb - lsb) + 1
                direction = (
                    f.get("dir") or f.get("direction") or f.get("towards") or ""
                )
            else:
                name, width, direction = f, None, ""
            if not name:
                continue
            out.append({
                "name": str(name),
                "width": "" if width is None else str(width),
                "dir": str(direction),
                "kind": kind,
            })
    seen, uniq = set(), []
    for spec in out:
        if spec["name"] not in seen:
            seen.add(spec["name"])
            uniq.append(spec)
    return uniq


def _signal_names(edge: dict) -> list[str]:
    """Every signal name on a channel (payload fields plus sideband)."""
    return [s["name"] for s in signal_specs(edge)]


def contract_port_rows(project_root, block_name: str) -> list[dict]:
    """The ports the contract DECLARES for one block, as canonical rows.

    Walks exactly the edges :func:`check_block` walks, in the same order, and
    builds the same ``f"{chan}_{n}"`` expectation -- so what the generator is
    shown and what the gate demands are derived from one place. Returns::

        [{"channel", "role", "signal", "port", "width", "dir", "kind",
          "peer", "doubled_token"}]

    ``port`` is the canonical flattened name. ``doubled_token`` marks the rows
    whose channel suffix and signal prefix share a token (``data_write`` +
    ``write_enable`` -> ``data_write_write_enable``), i.e. exactly the rows a
    generator "tidies up" into an unwireable name.
    """
    rows: list[dict] = []
    for edge in _load_contracts(project_root):
        for role, key, peer_key in (
            ("producer", "producer_port", "consumer_block"),
            ("consumer", "consumer_port", "producer_block"),
        ):
            role_key = "producer_block" if role == "producer" else "consumer_block"
            if edge.get(role_key) != block_name:
                continue
            chan_raw = str(edge.get(key) or "")
            if not chan_raw:
                continue
            for row in channel_signals(edge, chan_raw):
                rows.append(dict(
                    row, role=role, peer=str(edge.get(peer_key) or "")))
    return rows


#: Prepended wherever a block's ACCUMULATED constraints (constraints.json) are
#: put in front of a generator. Learned constraints are written by a debug
#: agent looking at one failure; the contract is frozen design intent. When
#: they disagree about a NAME, the contract wins -- silently, without asking.
CONSTRAINT_PRECEDENCE_LINE = (
    "PRECEDENCE: these accumulated constraints are subordinate to the "
    "interface contract's port table on anything to do with NAMING. If a "
    "constraint (or a previous attempt's RTL, or the golden model, or the "
    "uArch spec's prose) spells a port differently from the contract's "
    "AUTHORITATIVE PORT NAMES table, the contract wins -- use the contract's "
    "spelling and ignore the constraint's. Constraints remain authoritative "
    "for everything that is not a port name."
)

_PORT_TABLE_HEADER = (
    "## AUTHORITATIVE PORT NAMES (from the frozen interface contract)\n"
    "The golden model's port identifiers may be collapsed or abbreviated; "
    "transcribe the model's BEHAVIOR byte-exact but take every port NAME from "
    "this table. Each row is a port your module MUST declare, spelled exactly "
    "as shown. A deterministic pre-simulation gate checks this list against "
    "your module header and FAILS the block on any deviation -- there is no "
    "sim to reach if a name is wrong.\n"
)


def format_contract_port_table(project_root, block_name: str) -> str:
    """Render :func:`contract_port_rows` as a prompt fragment ('' when empty).

    Grouped by channel so the ``<channel>_<field>`` construction is visible,
    with the doubled-token rows called out by name (the exact class the
    conformance gate keeps catching).
    """
    rows = contract_port_rows(project_root, block_name)
    if not rows:
        return ""
    lines = ["", _PORT_TABLE_HEADER]
    seen_channels: list[tuple[str, str, str]] = []
    for r in rows:
        keyed = (r["channel"], r["role"], r["peer"])
        if keyed not in seen_channels:
            seen_channels.append(keyed)
    for chan, role, peer in seen_channels:
        peer_txt = f" <-> {peer}" if peer else ""
        lines.append(f"\n**channel `{chan}`** (this block is the {role}{peer_txt})")
        for r in rows:
            if (r["channel"], r["role"], r["peer"]) != (chan, role, peer):
                continue
            bits = []
            if r["width"]:
                bits.append(f"width {r['width']}")
            if r["dir"]:
                bits.append(f"dir {r['dir']}")
            bits.append(r["kind"])
            note = ""
            if r["doubled_token"]:
                note = (
                    "   <-- DOUBLED TOKEN IS CORRECT: channel "
                    f"`{chan}` + signal `{r['signal']}`. Do NOT collapse it."
                )
            lines.append(
                f"- `{r['port']}`  ({', '.join(bits)}; signal "
                f"`{r['signal']}`){note}"
            )
    lines.append(
        "\nRules: the canonical port name is `<channel>_<signal>`; never "
        "shorten a repeated token, never drop the channel prefix, never "
        "expose the same signal twice (both prefixed and bare), and never "
        "invent a port that wears a channel prefix but is not in this table."
    )
    return "\n".join(lines)


_PORT_RE = re.compile(r"\b(?:input|output|inout)\b([^;)]*)", re.MULTILINE)


def declared_ports(rtl_text: str, module: str | None = None) -> set[str]:
    """Port names ONE module declares. Tolerant of both Verilog styles.

    Scoped to a single module deliberately. A generated file routinely carries
    stub declarations of other modules after the real one, and unioning their
    ports produced a FALSE PASS for the pad block: it looked like it exposed
    qspi_csn/qspi_sck because the stubs at the bottom of its own file declare
    them, while the top module itself has only the Caravel boundary. Defaults to
    the FIRST module, which is the one the file is named for.
    """
    text = re.sub(r"/\*.*?\*/", " ", rtl_text, flags=re.S)
    text = re.sub(r"//[^\n]*", " ", text)
    if module:
        m = re.search(r"\bmodule\s+" + re.escape(module) + r"\b", text)
    else:
        m = re.search(r"\bmodule\s+[A-Za-z_]\w*", text)
    if m:
        end = text.find("endmodule", m.start())
        text = text[m.start():end if end != -1 else len(text)]
    noise = {"wire", "reg", "logic", "signed", "unsigned", "input", "output",
             "inout", "bit", "tri", "var", "integer", "real"}
    ports: set[str] = set()
    for m in _PORT_RE.finditer(text):
        seg = re.sub(r"\[[^\]]*\]", " ", m.group(1))
        for tok in re.findall(r"[A-Za-z_]\w*", seg):
            if tok not in noise:
                ports.add(tok)
    return ports


def check_block(project_root, block_name: str, rtl_path,
                siblings=()) -> ConformanceResult:
    """Verify one block's ports against every contract edge that touches it."""
    res = ConformanceResult(block=block_name)
    try:
        rtl = Path(rtl_path).read_text(errors="ignore")
    except OSError as exc:
        res.reason = f"cannot read {rtl_path}: {exc}"
        return res

    # Same trap: parse the BLOCK's module. Defaulting to the first
    # module made this checker report 22 phantom missing ports for a
    # block whose real module is declared last in its own file.
    _mod = block_name
    if not re.search(r"\bmodule\s+" + re.escape(block_name) + r"\b", rtl):
        _mod = Path(rtl_path).stem
    ports = declared_ports(rtl, module=_mod)

    # Structural: a leaf must not assemble the design. Scoped to this block's
    # own module -- a file may legitimately carry an alias wrapper after it.
    body = rtl
    _bm = re.search(r"\bmodule\s+" + re.escape(_mod) + r"\b", rtl)
    if _bm:
        _be = rtl.find("endmodule", _bm.start())
        body = rtl[_bm.start():_be if _be != -1 else len(rtl)]
    body = re.sub(r"//[^\n]*", " ", re.sub(r"/\*.*?\*/", " ", body, flags=re.S))
    for sib in siblings or ():
        if sib == block_name:
            continue
        if re.search(r"\b" + re.escape(str(sib)) + r"\s+[A-Za-z_\\]", body):
            res.instantiates.append(str(sib))
    # A block carrying the Caravel pad boundary is NOT exempt from the contract.
    # Its io_in/io_out/io_oeb names are externally mandated, so those specific
    # ports are never reported as undeclared -- but the channel signals the
    # contract says this block exposes INWARD must still be there.
    #
    # A blanket exemption hid the largest defect in the design: the pad block was
    # generated as a complete chip top that instantiates the other blocks
    # internally, with the channel signals as internal wires and NO inward ports
    # at all. The architecture specifies a pin ADAPTER with ports; the RTL
    # produced a competing top. Nothing can wire that, which is why the
    # deterministic assembler always fell back to an LLM-authored integration.
    res.locked_boundary = all(p in ports for p in _LOCKED_BOUNDARY_PORTS)

    bare_owner: dict[str, str] = {}      # bare port -> channel that claimed it
    accepted: set[str] = set()

    for edge in _load_contracts(project_root):
        for role, key in (("producer_block", "producer_port"),
                          ("consumer_block", "consumer_port")):
            if edge.get(role) != block_name:
                continue
            chan_raw = str(edge.get(key) or "")
            if not chan_raw:
                continue
            res.checked_edges += 1
            # ONE derivation, shared with the prompt's port table: a channel
            # or signal spelled as a slash enumeration/alias is reduced here,
            # never concatenated into an unparseable name.
            chan = channel_base(chan_raw)
            for row in channel_signals(edge, chan_raw):
                prefixed, bare = row["port"], row["bare"]
                has_p = prefixed in ports
                # When the contract's signal already carries the channel
                # prefix there is only ONE acceptable spelling, so the bare
                # form is not a second candidate (and cannot be "ambiguous"
                # with itself).
                has_b = bare != prefixed and bare in ports
                if has_p and has_b:
                    # Two candidate ports for one declared signal: the stitcher
                    # would have to pick, and picking is how channels get
                    # cross-wired.
                    res.ambiguous.append(
                        (chan, f"{prefixed} and {bare} both exist"))
                    accepted.update((prefixed, bare))
                elif has_p:
                    accepted.add(prefixed)
                elif has_b:
                    # A bare name is only unambiguous while ONE channel claims
                    # it. Two channels sharing `req_addr` on one block cannot be
                    # told apart by name.
                    prev = bare_owner.get(bare)
                    if prev is not None and prev != chan:
                        res.ambiguous.append(
                            (chan, f"{bare} is claimed by both '{prev}' and "
                                   f"'{chan}'; prefix it"))
                    bare_owner[bare] = chan
                    accepted.add(bare)
                else:
                    res.missing.append((chan, prefixed))

    res.accounted = set(accepted)
    res.ports = set(ports)

    # A port wearing a channel prefix that no declared signal accounts for is
    # either a misspelling of one of the above or an invented signal; both break
    # name-based edge resolution.
    channels = set()
    for edge in _load_contracts(project_root):
        for role, key in (("producer_block", "producer_port"),
                          ("consumer_block", "consumer_port")):
            if edge.get(role) == block_name and edge.get(key):
                base = channel_base(edge[key])
                if base:
                    channels.add(base)
    for port in sorted(ports):
        if port in accepted or port in _LOCKED_BOUNDARY_PORTS:
            continue
        for chan in channels:
            if port.startswith(chan + "_"):
                res.undeclared.append(port)
                break
    return res


def _rename_in_module(text: str, module: str, renames: dict) -> str:
    """Apply identifier renames inside one module body only."""
    m = re.search(r"\bmodule\s+" + re.escape(module) + r"\b", text)
    if not m:
        return text
    end = text.find("endmodule", m.start())
    end = end + len("endmodule") if end != -1 else len(text)
    body = text[m.start():end]
    for old, new in renames.items():
        body = re.sub(r"\b" + re.escape(old) + r"\b", new, body)
    return text[:m.start()] + body + text[end:]


def plan_port_repairs(result: "ConformanceResult") -> dict:
    """Map each undeclared port to the declared port it is a near-miss for.

    Only unambiguous pairs are returned. A declared name and an existing port
    match when they agree on the trailing signal name, which covers both
    observed shapes -- a collapsed duplicate token and a wrong channel prefix --
    without needing to know which one it is.

    Ambiguity means no repair. Renaming the wrong wire cross-wires a channel,
    and a silent cross-wire is worse than a loud deviation.
    """
    # Tier 3 needs no undeclared ports, so only `missing` gates the pass. The
    # earlier guard also required undeclared to be non-empty, which meant a
    # block whose prefix-collapsed ports had already been repaired could never
    # reach tier 3.
    if not result.missing:
        return {}

    pairs: dict = {}
    for chan, want in result.missing:
        # Only ports already wearing this channel's prefix are candidates. The
        # channel is what makes the match unambiguous: `host_read_enable` can
        # only be repairing a `host_read` signal, even though it shares
        # `read_enable` with framebuffer_read's.
        cands = [h for h in result.undeclared if h.startswith(chan + "_")]
        if len(cands) == 1:
            pairs.setdefault(cands[0], []).append(want)
            continue
        if cands:
            continue        # ambiguous within the channel -- leave it alone
        # Tier 3: a port spelling this channel with a DIFFERENT prefix. Match on
        # the trailing signal name, but only among ports not already bound to
        # some declared signal -- without that exclusion, three ports on the
        # real block end in `_req_addr` and the match is a coin flip.
        sig = want[len(chan) + 1:] if want.startswith(chan + "_") else want
        if not sig:
            continue
        free = [q for q in result.ports
                if q not in result.accounted and q.endswith("_" + sig)]
        if len(free) == 1:
            pairs.setdefault(free[0], []).append(want)

    # A source port wanted by two declared names is ambiguous -- drop it.
    plan = {have: wants[0] for have, wants in pairs.items() if len(wants) == 1}

    # Last line of defence before RTL is rewritten: never rename a port to
    # something that is not an identifier. `check_block` already refuses to
    # derive one, so reaching here means a new derivation path was added --
    # and a rename is the step that turns a bad name into corrupted source.
    safe = {}
    for have, want in plan.items():
        if is_legal_identifier(want) and is_legal_identifier(have):
            safe[have] = want
        else:
            _logger.error(
                "REFUSING port repair %r -> %r: not a legal Verilog "
                "identifier. The RTL is left untouched.", have, want)
            print(f"[contract-conformance] REFUSING port repair {have!r} -> "
                  f"{want!r}: not a legal Verilog identifier", file=sys.stderr)
    return safe


def repair_block_ports(project_root, block_name: str, rtl_path,
                       siblings=(), apply: bool = False) -> dict:
    """Compute (and optionally apply) port-name repairs for one block.

    Returns ``{"renames": {...}, "before": n, "after": n, "conforms": bool}``.
    The result is RE-CHECKED after applying, so a repair can never be reported
    as success unless the checker agrees.
    """
    before = check_block(project_root, block_name, rtl_path, siblings=siblings)
    renames = plan_port_repairs(before)
    out = {"renames": renames, "before": len(before.missing),
           "after": len(before.missing), "conforms": before.ok}
    if not renames or not apply:
        return out

    text = Path(rtl_path).read_text(errors="ignore")
    mod = block_name
    if not re.search(r"\bmodule\s+" + re.escape(block_name) + r"\b", text):
        mod = Path(rtl_path).stem
    Path(str(rtl_path) + ".pre_portrepair").write_text(text)
    Path(rtl_path).write_text(_rename_in_module(text, mod, renames))

    after = check_block(project_root, block_name, rtl_path, siblings=siblings)
    out["after"] = len(after.missing)
    out["conforms"] = after.ok
    return out


# ---------------------------------------------------------------------------
# Production stage: check -> repair -> re-check, wired into the block RTL flow
# ---------------------------------------------------------------------------

def conformance_gate_enabled() -> bool:
    """Contract-conformance stage in the per-block RTL flow (default ON).

    ``CORESMITH_CONTRACT_CONFORMANCE_GATE=0`` disables it -- the same env-gate
    convention as the storage / ifdef / stage lints.
    """
    return (os.environ.get("CORESMITH_CONTRACT_CONFORMANCE_GATE", "1")
            or "1").strip().lower() not in {"0", "false", "no", "off"}


#: How a cocotb testbench names a DUT port. A rename is applied to THESE forms
#: only. A blanket word-boundary substitution over the TB would also rewrite a
#: local variable that happens to share a generic signal name (`read_enable` is
#: not distinctive), and corrupting a testbench to fix a port name trades a
#: loud failure for a silent one.
def _tb_ref_patterns(old: str) -> list[tuple[str, str]]:
    o = re.escape(old)
    return [
        (rf"(?<![\w.])dut\.{o}\b", "dut.{new}"),
        (rf"(?<![\w.])dut\._id\(\s*(['\"]){o}\1", 'dut._id("{new}"'),
        (rf"getattr\(\s*dut\s*,\s*(['\"]){o}\1", 'getattr(dut, "{new}"'),
    ]


def repair_testbench_refs(tb_path, renames: dict) -> dict:
    """Rewrite a testbench's DUT port references after a port rename.

    Returns ``{"applied": {old: n_sites}, "residual": [old, ...],
    "changed": bool, "needs_regen": bool}``. ``residual`` lists renamed ports
    whose OLD name still appears as a bare word in the testbench after the
    narrow rewrite -- either a harmless local, or a port reference in a form
    this function deliberately does not touch (a generated testbench really does
    drive ``getattr(dut, field)`` over a tuple of port-name STRINGS, and the
    same strings key its stimulus dict, which is also how it feeds the Amaranth
    block model; blanket-renaming quoted strings would corrupt the model side).

    So residual references are not guessed at -- they set ``needs_regen``, and
    the caller regenerates the testbench against the repaired RTL. That is one
    cheap testbench call instead of a sim failure that costs an RTL attempt,
    and it never invents a mapping the checker cannot prove.
    """
    out: dict = {"applied": {}, "residual": [], "changed": False,
                 "needs_regen": False}
    p = Path(tb_path)
    if not renames or not tb_path or not p.exists():
        return out
    try:
        text = original = p.read_text(errors="ignore")
    except OSError:
        return out
    for old, new in renames.items():
        n = 0
        for pat, repl in _tb_ref_patterns(old):
            text, k = re.subn(pat, repl.format(new=new), text)
            n += k
        if n:
            out["applied"][old] = n
        if re.search(r"\b" + re.escape(old) + r"\b", text):
            out["residual"].append(old)
    out["needs_regen"] = bool(out["residual"])
    if text != original:
        try:
            Path(str(tb_path) + ".pre_portrepair").write_text(original)
            p.write_text(text)
            out["changed"] = True
        except OSError:
            return {"applied": {}, "residual": sorted(renames),
                    "changed": False, "needs_regen": True}
    return out


def run_conformance_stage(project_root, block_name: str, rtl_path,
                          siblings=(), tb_path: str = "") -> dict:
    """Check one block against its contract, repair what is unambiguous, re-check.

    This is what the per-block flow calls. It NEVER decides the block's fate --
    it returns the record and the caller (``generate_testbench_node``) logs it,
    fails the block, or parks.

    Returns::

        {ran, ok, reason, checked_edges, renames{old:new}, rename_channels,
         before_missing, after_missing, deviations[], feedback, tb{...}}

    ``ran`` is False (and ``ok`` True) when the stage does not apply: no
    contract edge names this block, or the RTL is unreadable. Those are other
    gates' jobs -- this one must not invent a verdict it has no evidence for.
    """
    out: dict = {
        "block": block_name, "ran": False, "ok": True, "reason": "",
        "checked_edges": 0, "renames": {}, "rename_channels": {},
        "before_missing": 0, "after_missing": 0, "deviations": [],
        "feedback": "",
        "tb": {"applied": {}, "residual": [], "changed": False,
               "needs_regen": False},
    }
    before = check_block(project_root, block_name, rtl_path, siblings=siblings)
    if before.reason:
        out["reason"] = before.reason
        return out
    out["checked_edges"] = before.checked_edges
    if not before.checked_edges:
        out["reason"] = ("no interface contract edge names this block -- "
                         "nothing to conform to")
        return out

    out["ran"] = True
    out["before_missing"] = len(before.missing)
    out["deviations"] = _deviation_lines(before)
    if before.ok:
        out["ok"] = True
        return out

    # Repair only the unambiguous renames, then let the checker -- not the
    # repairer -- say whether the block conforms.
    want_channel = {port: chan for chan, port in before.missing}
    rep = repair_block_ports(project_root, block_name, rtl_path,
                             siblings=siblings, apply=True)
    renames = rep.get("renames") or {}
    out["renames"] = dict(renames)
    out["rename_channels"] = {
        old: want_channel.get(new, "") for old, new in renames.items()
    }

    after = check_block(project_root, block_name, rtl_path, siblings=siblings)
    out["after_missing"] = len(after.missing)
    out["ok"] = after.ok
    if not after.ok:
        out["deviations"] = _deviation_lines(after)
        out["feedback"] = after.as_feedback()

    if renames:
        out["tb"] = repair_testbench_refs(tb_path, renames)
    return out


def _deviation_lines(res: ConformanceResult) -> list[str]:
    """One flat, greppable line per deviation (for logs + the run artifact)."""
    lines = [f"missing {port} (channel '{chan}')" for chan, port in res.missing]
    lines += [f"undeclared port {p}" for p in res.undeclared]
    lines += [f"ambiguous channel '{c}': {why}" for c, why in res.ambiguous]
    lines += [f"instantiates sibling block {s}" for s in res.instantiates]
    return lines
