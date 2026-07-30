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
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

#: A block whose module declares the full Caravel pad boundary has externally
#: MANDATED port names (io_in/io_out/io_oeb[37:0] are fixed by the shuttle), so
#: the <channel>_<field> convention cannot apply to it.
_LOCKED_BOUNDARY_PORTS = ("io_in", "io_out", "io_oeb")


@dataclass
class ConformanceResult:
    """Per-block verdict. ``ok`` only when every declared signal is present."""

    block: str = ""
    checked_edges: int = 0
    missing: list = field(default_factory=list)     # [(channel, expected_port)]
    undeclared: list = field(default_factory=list)  # [port] -- <channel>_* not in the contract
    ambiguous: list = field(default_factory=list)   # [(channel, explanation)]
    exempt: bool = False
    locked_boundary: bool = False   # carries externally-mandated pad names
    reason: str = ""

    @property
    def ok(self) -> bool:
        return not (self.missing or self.undeclared or self.ambiguous)

    def as_feedback(self) -> str:
        """An actionable message for the RTL generator.

        States the EXACT required name. The failure mode this guards is a
        generator collapsing a duplicated token (channel ``host_write`` +
        signal ``write_enable`` -> ``host_write_enable``), so a vague "port
        names must match the contract" would not change the outcome -- the
        prompt already says that.
        """
        lines = []
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


def _signal_names(edge: dict) -> list[str]:
    """Every signal on a channel: the payload fields plus the sideband.

    Split across two keys in the schema, which is precisely why several readers
    concluded the contract did not record them at all.
    """
    out: list[str] = []
    for f in edge.get("fields") or []:
        n = f.get("name") if isinstance(f, dict) else f
        if n:
            out.append(str(n))
    for s in edge.get("sideband_signals") or []:
        n = s.get("name") if isinstance(s, dict) else s
        if n:
            out.append(str(n))
    seen, uniq = set(), []
    for n in out:
        if n not in seen:
            seen.add(n)
            uniq.append(n)
    return uniq


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


def check_block(project_root, block_name: str, rtl_path) -> ConformanceResult:
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
            chan = str(edge.get(key) or "")
            if not chan:
                continue
            res.checked_edges += 1
            for n in _signal_names(edge):
                prefixed, bare = f"{chan}_{n}", n
                has_p, has_b = prefixed in ports, bare in ports
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

    # A port wearing a channel prefix that no declared signal accounts for is
    # either a misspelling of one of the above or an invented signal; both break
    # name-based edge resolution.
    channels = set()
    for edge in _load_contracts(project_root):
        for role, key in (("producer_block", "producer_port"),
                          ("consumer_block", "consumer_port")):
            if edge.get(role) == block_name and edge.get(key):
                channels.add(str(edge[key]))
    for port in sorted(ports):
        if port in accepted or port in _LOCKED_BOUNDARY_PORTS:
            continue
        for chan in channels:
            if port.startswith(chan + "_"):
                res.undeclared.append(port)
                break
    return res
