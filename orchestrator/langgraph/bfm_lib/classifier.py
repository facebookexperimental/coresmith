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

If the bus is not QSPI-slave (or the evidence is ambiguous), returns None -- the
caller then keeps the existing LLM BFM and logs a loud advisory that the run's
DV is NOT contract-enforcing.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from .qspi_contract import QSPIContract


def _top_has_gpio_pin_boundary(top_rtl_source: str) -> bool:
    """True if the top exposes the Caravel ``io_in``/``io_out``/``io_oeb`` bus.

    This is the QSPI-slave chassis pin boundary (no AXI-Stream): the whole
    external contract is pins on a standard off-chip bus. We require all three
    (in/out/oeb) so a design that merely names ``io_in`` for something else is
    not misclassified.
    """
    src = top_rtl_source
    have = 0
    for sig in ("io_in", "io_out", "io_oeb"):
        # a port declaration like `... io_in`  (input/output [..]/reg/wire)
        if re.search(rf"\b{sig}\b", src):
            have += 1
    return have == 3


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


def _connections_have_qspi(connections: Optional[list]) -> bool:
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
    project_root: str, connections: Optional[list] = None
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
    connections: Optional[list] = None,
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
    connections: Optional[list] = None,
) -> str:
    """Name the SPECIFIC unmodeled bus role(s), for the advisory / carried-defect.

    Never a generic single-role label: when the DUT masters a second bus, say
    WHICH pin group is unanswered (e.g. "DUT-mastered second bus (rom_*) ...").
    """
    roles = detect_bus_roles(project_root, top_rtl_source, connections)
    return roles["summary"]


def classify_chip_bus(
    project_root: str,
    top_rtl_source: str,
    connections: Optional[list] = None,
) -> Optional[QSPIContract]:
    """Return a QSPIContract if the chip-top external bus is QSPI-slave, else None.

    Decision (all DUT-blind):
      * REQUIRED: the top exposes the Caravel ``io_in``/``io_out``/``io_oeb``
        pin boundary (the QSPI-slave chassis; no AXI-Stream).
      * plus at least one corroborating signal: the PRD/ERS bus_protocol names
        QSPI, or an architecture connection references a qspi interface.
    """
    if not _top_has_gpio_pin_boundary(top_rtl_source):
        return None
    corroborated = _bus_protocol_says_qspi(project_root) or _connections_have_qspi(
        connections
    )
    if not corroborated:
        return None
    overrides = _reg_map_overrides(project_root)
    return QSPIContract(**overrides)
