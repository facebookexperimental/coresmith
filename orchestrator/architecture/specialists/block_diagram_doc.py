# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Block diagram documentation specialist.

Converts the finalized architecture (block diagram, memory map, clock tree)
into a structured JSON document that can be consumed by the ReactFlow-based
Block Diagram viewer tab in CoreSmith.

The output format matches the taskgraph_dash_component architecture graph
schema so the same node types and styles (compute, bus, memory, sensor,
hwa, power_domain, pll, pcie, gpio, pmic) can be reused.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC
from pathlib import Path

log = logging.getLogger(__name__)


def _hashable_label(value) -> str:
    """Coerce a connection field (interface / data_width) to a hashable,
    human-readable label.

    Block-diagram connection records sometimes carry the ``interface`` or
    ``data_width`` field as a structured ``dict`` (e.g. ``{"name": ...,
    "data_width_bits": ...}``) or a ``list`` rather than a plain string.
    ``set.add()`` on a raw dict/list raises ``TypeError: unhashable type``.
    Normalize to a string so bus-hub note aggregation never crashes,
    regardless of the upstream schema shape.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("name", "interface", "id", "label", "bundle"):
            v = value.get(key)
            if isinstance(v, str) and v:
                return v
        # Fall back to a stable serialization of the dict.
        try:
            return json.dumps(value, sort_keys=True)
        except (TypeError, ValueError):
            return str(value)
    if isinstance(value, (list, tuple)):
        parts = [_hashable_label(v) for v in value]
        return ", ".join(p for p in parts if p)
    return str(value)


# ---------------------------------------------------------------------------
# Node type classification
# ---------------------------------------------------------------------------

# Maps block name keywords/patterns to architecture node types.
# These match the taskgraph_dash_component node type vocabulary:
#   compute, bus, memory, sensor, hwa, power_domain, pll, pcie, i3c, gpio, pmic
# Ordered by specificity: more specific types checked first to avoid
# false positives (e.g. "axi" in bus matching an "axi_lite csr bridge").
_TYPE_KEYWORDS = {
    "pll": [
        "pll", "clock_gen", "clk_gen", "oscillator", "dcm",
    ],
    "pcie": [
        "pcie", "pci",
    ],
    "i3c": [
        "i3c", "i2c", "spi",
    ],
    "gpio": [
        "gpio", "io_pad", "pad_ring", "io_mux",
    ],
    "pmic": [
        "pmic", "ldo", "regulator", "power_supply",
    ],
    "sensor": [
        "sensor", "adc", "dac", "analog", "comparator",
    ],
    "memory": [
        "fifo", "sram", "rom", "buffer", "cache", "ddr", "flash", "queue",
    ],
    "hwa": [
        "csr", "register", "config", "control", "apb_bridge",
        "axi_lite", "decoder", "arbiter", "dma",
    ],
    "bus": [
        "bus", "interconnect", "switch", "fabric", "crossbar",
        "axi_bus", "apb", "ahb", "wishbone", "noc",
    ],
}


def _classify_node_type(block: dict) -> str:
    """Classify a block into an architecture node type.

    Uses block name and description keywords to select from the
    taskgraph_dash_component node type vocabulary. Keywords are
    matched as whole words (using underscore/space as word boundaries)
    to avoid false positives like "scrambler" matching "ram".
    """
    import re

    name = block.get("name", "").lower()
    desc = (block.get("description", "") or "").lower()
    # Split into words (by underscores, spaces, hyphens)
    words = set(re.split(r"[\s_\-/]+", f"{name} {desc}"))
    text = f" {name} {desc} "

    # Check each type's keywords
    for node_type, keywords in _TYPE_KEYWORDS.items():
        for kw in keywords:
            # Match as a whole word in the word set, or as a substring
            # for multi-word keywords (e.g. "clock_gen")
            if kw in words or (len(kw) > 3 and kw in text):
                return node_type

    # Default: compute (processing blocks, DSP, etc.)
    return "compute"


# ---------------------------------------------------------------------------
# JSON generation (deterministic, no LLM needed)
# ---------------------------------------------------------------------------

def generate_block_diagram_doc(
    block_diagram: dict,
    memory_map: dict | None = None,
    clock_tree: dict | None = None,
    register_spec: dict | None = None,
    ers_spec: dict | None = None,
    design_name: str = "",
) -> dict:
    """Generate the ReactFlow-compatible block diagram JSON.

    Converts the architecture specialist outputs into the same JSON format
    used by taskgraph_dash_component's architecture graph view:
    - systemNodes: list of ReactFlow node objects
    - systemEdges: list of ReactFlow edge objects
    - systemLayout: ELK.js layout configuration

    Args:
        block_diagram: The finalized block diagram dict from the specialist.
        memory_map: Optional memory map for enriching node data.
        clock_tree: Optional clock tree for frequency annotations.
        register_spec: Optional register spec for CSR annotations.
        ers_spec: Optional PRD for design name and metadata.
        design_name: Override design name.

    Returns:
        A validated JSON-serializable dict matching the block diagram schema.
    """
    blocks = block_diagram.get("blocks", [])
    connections = block_diagram.get("connections", [])

    # Extract design name
    if not design_name and ers_spec:
        ers = ers_spec.get("prd", ers_spec.get("ers", {}))
        design_name = ers.get("title", "ASIC Design")

    if not design_name:
        design_name = "ASIC Design"

    # Build clock frequency lookup from clock_tree
    clock_lookup: dict[str, str] = {}
    if clock_tree:
        ct_result = clock_tree.get("result", clock_tree)
        for domain in ct_result.get("clock_domains", []):
            domain.get("name", "")
            freq = domain.get("frequency_mhz", "")
            for blk in domain.get("blocks", []):
                if freq:
                    clock_lookup[blk] = f"{freq} MHz"

    # Build memory info lookup
    mem_lookup: dict[str, dict] = {}
    if memory_map:
        mm_result = memory_map.get("result", memory_map)
        for periph in mm_result.get("peripherals", []):
            pname = periph.get("name", "")
            mem_lookup[pname] = {
                "base_address": periph.get("base_address", ""),
                "size": periph.get("size", ""),
            }

    # Build register info lookup
    reg_lookup: dict[str, int] = {}
    if register_spec:
        rs_result = register_spec.get("result", register_spec)
        for rb in rs_result.get("register_blocks", []):
            rbname = rb.get("block", "")
            reg_lookup[rbname] = len(rb.get("registers", []))

    # ── Detect subsystems (group nodes) ──
    # If blocks have a "subsystem" field, create group nodes.
    subsystems: dict[str, list[str]] = {}
    for block in blocks:
        sub = block.get("subsystem", "")
        if sub:
            subsystems.setdefault(sub, []).append(block.get("name", ""))

    # ── Build nodes ──
    nodes = []

    # Add subsystem group nodes
    for sub_name in subsystems:
        nodes.append({
            "id": sub_name,
            "type": "nodeArchGraph",
            "position": {"x": 0, "y": 0},
            "data": {
                "node_type": "",
                "node_name": sub_name,
                "node_parentId": "graph_root",
                "connect_to": [],
                "is_subsystem": True,
                "is_power_domain": False,
                "device_name": sub_name,
                "frequency": "",
                "module_type": "",
                "node_notes": f"Subsystem: {sub_name}",
            },
        })

    # Add block nodes
    for block in blocks:
        name = block.get("name", "")
        node_type = _classify_node_type(block)
        subsystem = block.get("subsystem", "")
        parent_id = subsystem if subsystem else "graph_root"
        node_id = f"{subsystem}.{name}" if subsystem else name

        # Build connect_to from connections
        connect_to = []
        for conn in connections:
            if conn.get("from") == name:
                target = conn.get("to", "")
                target_sub = ""
                for b in blocks:
                    if b.get("name") == target:
                        target_sub = b.get("subsystem", "")
                        break
                abs_path = f"{target_sub}.{target}" if target_sub else target
                connect_to.append({
                    "absolute_path": abs_path,
                    "local_path": target,
                })

        # Frequency from clock tree
        freq = clock_lookup.get(name, "")

        # Build notes from multiple sources
        notes_parts = []
        desc = block.get("description", "")
        if desc:
            notes_parts.append(desc)
        tier = block.get("tier", 0)
        if tier:
            notes_parts.append(f"Tier {tier}")
        gates = block.get("estimated_gates", 0)
        if gates:
            if gates >= 1000:
                notes_parts.append(f"~{gates // 1000}K gates")
            else:
                notes_parts.append(f"~{gates} gates")
        if name in mem_lookup:
            mi = mem_lookup[name]
            notes_parts.append(f"Addr: {mi['base_address']}, Size: {mi['size']}")
        if name in reg_lookup:
            notes_parts.append(f"{reg_lookup[name]} registers")

        nodes.append({
            "id": node_id,
            "type": "nodeArchGraph",
            "position": {"x": 0, "y": 0},
            "data": {
                "node_type": node_type,
                "node_name": node_id,
                "node_parentId": parent_id,
                "connect_to": connect_to,
                "is_subsystem": False,
                "is_power_domain": False,
                "device_name": name,
                "frequency": freq,
                "module_type": node_type,
                "node_notes": " | ".join(notes_parts) if notes_parts else "",
            },
        })

    # ── Build bus hub nodes ──
    # Collect all unique bus_name values from connections and create a
    # bus hub node for each.  Connections that route through a bus are
    # split into two edges: source -> bus_node, bus_node -> target.
    bus_names: dict[str, list[dict]] = {}
    for conn in connections:
        bus = conn.get("bus_name", "")
        if bus:
            bus_names.setdefault(bus, []).append(conn)

    # Track which bus nodes we've already created
    bus_node_ids: set[str] = set()
    for bus_name, bus_conns in bus_names.items():
        bus_id = f"bus__{bus_name}"
        if bus_id in bus_node_ids:
            continue
        bus_node_ids.add(bus_id)

        # Gather the set of blocks that connect through this bus
        connected_blocks = set()
        bus_interfaces = set()
        bus_widths = set()
        for c in bus_conns:
            connected_blocks.add(c.get("from", ""))
            connected_blocks.add(c.get("to", ""))
            iface = _hashable_label(c.get("interface", ""))
            if iface:
                bus_interfaces.add(iface)
            dw = _hashable_label(c.get("data_width", ""))
            if dw:
                bus_widths.add(str(dw))

        notes_parts = []
        if bus_interfaces:
            notes_parts.append(", ".join(sorted(bus_interfaces)))
        if bus_widths:
            notes_parts.append("Width: " + "/".join(sorted(bus_widths)) + "b")
        notes_parts.append(f"{len(connected_blocks)} ports")

        nodes.append({
            "id": bus_id,
            "type": "nodeArchGraph",
            "position": {"x": 0, "y": 0},
            "data": {
                "node_type": "bus",
                "node_name": bus_name,
                "node_parentId": "graph_root",
                "connect_to": [],
                "is_subsystem": False,
                "is_power_domain": False,
                "device_name": bus_name,
                "frequency": "",
                "module_type": "bus",
                "node_notes": " | ".join(notes_parts) if notes_parts else "",
            },
        })

    # ── Resolve block name -> full ID (with subsystem prefix) ──
    block_id_map: dict[str, str] = {}
    for block in blocks:
        name = block.get("name", "")
        subsystem = block.get("subsystem", "")
        block_id_map[name] = f"{subsystem}.{name}" if subsystem else name

    # ── Build edges ──
    edges = []
    # Track edges already added to avoid duplicates (e.g. same bus pair)
    edge_id_set: set[str] = set()

    for conn in connections:
        src = conn.get("from", "")
        tgt = conn.get("to", "")
        if not src or not tgt:
            continue

        src_id = block_id_map.get(src, src)
        tgt_id = block_id_map.get(tgt, tgt)
        interface = _hashable_label(conn.get("interface", ""))
        data_width = _hashable_label(conn.get("data_width", ""))
        bus_name = conn.get("bus_name", "")

        if bus_name:
            # Route through bus hub node: source -> bus, bus -> target
            bus_id = f"bus__{bus_name}"

            # Edge: source -> bus
            eid_src = f"e{src}-{bus_name}"
            if eid_src not in edge_id_set:
                edge_id_set.add(eid_src)
                label_parts_src = [f"{src}->{bus_name}"]
                if interface:
                    label_parts_src.append(interface)
                if data_width:
                    label_parts_src.append(f"{data_width}b")
                edges.append({
                    "id": eid_src,
                    "source": src_id,
                    "target": bus_id,
                    "type": "edgeArchGraph",
                    "data": {
                        "label": " ".join(label_parts_src),
                        "connection_type": "bus_connect",
                    },
                })

            # Edge: bus -> target
            eid_tgt = f"e{bus_name}-{tgt}"
            if eid_tgt not in edge_id_set:
                edge_id_set.add(eid_tgt)
                label_parts_tgt = [f"{bus_name}->{tgt}"]
                if interface:
                    label_parts_tgt.append(interface)
                if data_width:
                    label_parts_tgt.append(f"{data_width}b")
                edges.append({
                    "id": eid_tgt,
                    "source": bus_id,
                    "target": tgt_id,
                    "type": "edgeArchGraph",
                    "data": {
                        "label": " ".join(label_parts_tgt),
                        "connection_type": "bus_connect",
                    },
                })
        else:
            # Direct point-to-point connection
            eid = f"e{src}-{tgt}"
            if eid in edge_id_set:
                continue
            edge_id_set.add(eid)

            label_parts = [f"{src}->{tgt}"]
            if interface:
                label_parts.append(interface)
            if data_width:
                label_parts.append(f"{data_width}b")

            edges.append({
                "id": eid,
                "source": src_id,
                "target": tgt_id,
                "type": "edgeArchGraph",
                "data": {
                    "label": " ".join(label_parts),
                    "connection_type": "depends_on",
                },
            })

    # ── Build the full document ──
    from datetime import datetime

    doc = {
        "version": {"id": "reactflow_json_1.0.0"},
        "metadata": {
            "design_name": design_name,
            "timestamp": datetime.now(UTC).isoformat(),
            "source": "coresmith_architecture",
            "block_count": len(blocks),
            "connection_count": len(connections),
        },
        "architecture": {
            "designName": design_name,
            "systemNodes": nodes,
            "systemEdges": edges,
            "systemLayout": {
                "elk_layoutOptions": {
                    "elk.algorithm": "layered",
                    "org.eclipse.elk.direction": "DOWN",
                    "elk.direction": "DOWN",
                    "org.eclipse.elk.spacing.nodeNode": "50",
                    "org.eclipse.elk.spacing.edgeNode": "50",
                    "org.eclipse.elk.hierarchyHandling": "INCLUDE_CHILDREN",
                    "elk.layered.spacing.nodeNodeBetweenLayers": "60",
                },
            },
            "moduleTypeOptions": [
                "bus", "compute", "hwa", "gpio", "memory",
                "pmic", "power_domain", "sensor", "pll",
                "pcie", "i3c",
            ],
        },
    }

    return doc


def persist_block_diagram_doc(doc: dict, project_root: str) -> Path:
    """Write the block diagram doc JSON to disk.

    Args:
        doc: The validated block diagram document.
        project_root: Project root directory.

    Returns:
        Path to the written file.
    """
    out_path = Path(project_root) / ".coresmith" / "block_diagram_viz.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=2, default=str))
    log.info("Block diagram doc written to %s", out_path)
    return out_path
