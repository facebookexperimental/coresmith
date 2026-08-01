# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
GDS Layout Viewer Generator (3D and 2D).

3D: Converts a GDS file to a glTF binary (.glb) and embeds it in a
standalone HTML file for interactive 3D visualization using Three.js.

2D: Renders a GDS file to SVG via gdstk and optionally converts to
PNG via cairosvg for embedding in the chip-finish dashboard.

The GDS is parsed with gdstk, polygons are triangulated per layer
with mapbox-earcut, and the resulting meshes are assembled into a
glTF 2.0 binary via pygltflib.  The .glb is base64-encoded and
injected into a Jinja2 HTML template that loads Three.js from CDN.

Dependencies (gdstk, pygltflib, mapbox-earcut) are optional -- the
module degrades gracefully when they are not installed.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    import gdstk
    import mapbox_earcut
    from pygltflib import (
        GLTF2,
        Accessor,
        Asset,
        Attributes,
        Buffer,
        BufferView,
        Material,
        Mesh,
        Node,
        PbrMetallicRoughness,
        Primitive,
        Scene,
    )

    _HAS_DEPS = True
except ImportError:
    _HAS_DEPS = False


# ── glTF numeric constants (defined locally to avoid import when deps missing)

_FLOAT = 5126
_UINT32 = 5125
_VEC3 = "VEC3"
_SCALAR = "SCALAR"
_ARRAY_BUF = 34962
_ELEM_BUF = 34963
_TRIANGLES = 4


# ── Sky130 layer mapping ────────────────────────────────────────
#
# (gds_layer, datatype) → rendering parameters.
#   z / h   – relative stack height (scaled to design extent at render time)
#   color   – RGB [0..1] for the material
#   pri     – render priority (1 = most important, kept first under budget)
#   alpha   – material opacity

SKY130_LAYERS: dict[tuple[int, int], dict[str, Any]] = {
    (64, 20): {"name": "nwell", "z": 0.00, "h": 0.08, "color": [0.20, 0.20, 0.20], "pri": 9,  "alpha": 1.0},
    (65, 20): {"name": "diff",  "z": 0.10, "h": 0.08, "color": [0.13, 0.55, 0.13], "pri": 8,  "alpha": 1.0},
    (65, 44): {"name": "tap",   "z": 0.10, "h": 0.08, "color": [0.13, 0.55, 0.13], "pri": 10, "alpha": 1.0},
    (66, 20): {"name": "poly",  "z": 0.25, "h": 0.12, "color": [0.75, 0.35, 0.46], "pri": 3,  "alpha": 1.0},
    (66, 44): {"name": "licon", "z": 0.37, "h": 0.23, "color": [0.50, 0.50, 0.50], "pri": 11, "alpha": 1.0},
    (67, 20): {"name": "li1",   "z": 0.60, "h": 0.15, "color": [1.00, 0.81, 0.55], "pri": 4,  "alpha": 0.90},
    (67, 44): {"name": "mcon",  "z": 0.75, "h": 0.25, "color": [0.50, 0.50, 0.50], "pri": 11, "alpha": 1.0},
    (68, 20): {"name": "met1",  "z": 1.00, "h": 0.20, "color": [0.16, 0.38, 0.83], "pri": 1,  "alpha": 0.85},
    (68, 44): {"name": "via",   "z": 1.20, "h": 0.25, "color": [0.50, 0.50, 0.50], "pri": 11, "alpha": 1.0},
    (69, 20): {"name": "met2",  "z": 1.45, "h": 0.20, "color": [0.65, 0.75, 0.90], "pri": 2,  "alpha": 0.85},
    (69, 44): {"name": "via2",  "z": 1.65, "h": 0.25, "color": [0.50, 0.50, 0.50], "pri": 11, "alpha": 1.0},
    (70, 20): {"name": "met3",  "z": 1.90, "h": 0.20, "color": [0.45, 0.65, 0.45], "pri": 5,  "alpha": 0.85},
    (70, 44): {"name": "via3",  "z": 2.10, "h": 0.25, "color": [0.50, 0.50, 0.50], "pri": 11, "alpha": 1.0},
    (71, 20): {"name": "met4",  "z": 2.35, "h": 0.25, "color": [0.75, 0.55, 0.35], "pri": 6,  "alpha": 0.85},
    (71, 44): {"name": "via4",  "z": 2.60, "h": 0.25, "color": [0.50, 0.50, 0.50], "pri": 11, "alpha": 1.0},
    (72, 20): {"name": "met5",  "z": 2.85, "h": 0.25, "color": [0.55, 0.35, 0.75], "pri": 7,  "alpha": 0.85},
}

_MAX_Z = max(v["z"] + v["h"] for v in SKY130_LAYERS.values())
DEFAULT_POLY_BUDGET = 300_000
Z_EXAGGERATION = 0.15


def _active_layer_map() -> dict[tuple[int, int], dict[str, Any]]:
    """The GDS layer map for the active deployment (PR6, byo-pdk).

    A deployment may supply its own ``gds_layer_map()``; when it returns None
    (the sky130 default) the engine's built-in ``SKY130_LAYERS`` is used, so
    behavior is unchanged. Best-effort -- any resolution error falls back to
    the sky130 map so the viewer never hard-fails."""
    try:
        from orchestrator.pdk.registry import get_deployment
        m = get_deployment().gds_layer_map()
        if m:
            return m
    except Exception:  # noqa: BLE001
        pass
    return SKY130_LAYERS


# ── Geometry helpers ────────────────────────────────────────────


def _triangulate(verts: np.ndarray) -> np.ndarray | None:
    """Earcut triangulation of a 2-D polygon.  Returns flat index array."""
    if len(verts) < 3:
        return None
    try:
        coords = np.ascontiguousarray(verts[:, :2].astype(np.float64))
        rings = np.array([len(verts)], dtype=np.uint32)
        idx = mapbox_earcut.triangulate_float64(coords, rings)
        return idx if len(idx) >= 3 else None
    except Exception:
        return None


def _build_layer_mesh(
    polygons: list[np.ndarray],
    z: float,
    h: float,
    scale: float,
    center: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Triangulate *polygons* and extrude from *z* to *z + h*.

    Generates top face, bottom face, and side walls for each polygon,
    producing solid 3D geometry instead of flat planes.
    Returns ``(positions_Nx3, indices_flat)`` or ``None``.
    """
    all_pos: list[np.ndarray] = []
    all_idx: list[np.ndarray] = []
    offset = 0
    z_top = z + h

    for verts in polygons:
        tri = _triangulate(verts)
        if tri is None:
            continue
        n = len(verts)
        xy = (verts - center) * scale
        xy32 = xy.astype(np.float32)

        if h < 1e-4:
            pos = np.column_stack([xy32, np.full(n, z_top, dtype=np.float32)])
            all_pos.append(pos)
            all_idx.append(tri.astype(np.uint32) + offset)
            offset += n
            continue

        # -- Top face (at z_top, original winding) --
        top_pos = np.column_stack([xy32, np.full(n, z_top, dtype=np.float32)])
        top_idx = tri.astype(np.uint32) + offset

        # -- Bottom face (at z, reversed winding for outward normals) --
        bot_pos = np.column_stack([xy32, np.full(n, z, dtype=np.float32)])
        bot_offset = offset + n
        bot_tri = tri.astype(np.uint32).reshape(-1, 3)[:, ::-1].ravel()
        bot_idx = bot_tri + bot_offset

        # -- Side walls (one quad = 2 triangles per polygon edge) --
        wall_pos_list = []
        wall_idx_list = []
        wall_offset = offset + 2 * n
        for i in range(n):
            j = (i + 1) % n
            v0 = np.array([xy32[i, 0], xy32[i, 1], z], dtype=np.float32)
            v1 = np.array([xy32[j, 0], xy32[j, 1], z], dtype=np.float32)
            v2 = np.array([xy32[j, 0], xy32[j, 1], z_top], dtype=np.float32)
            v3 = np.array([xy32[i, 0], xy32[i, 1], z_top], dtype=np.float32)
            wall_pos_list.extend([v0, v1, v2, v3])
            base = wall_offset + i * 4
            wall_idx_list.extend([base, base + 1, base + 2,
                                  base, base + 2, base + 3])

        wall_pos = np.array(wall_pos_list, dtype=np.float32).reshape(-1, 3)
        wall_idx = np.array(wall_idx_list, dtype=np.uint32)

        all_pos.extend([top_pos, bot_pos, wall_pos])
        all_idx.extend([top_idx, bot_idx, wall_idx])
        offset += 2 * n + 4 * n

    if not all_pos:
        return None
    return np.vstack(all_pos), np.concatenate(all_idx)


# ── GDS → glTF ──────────────────────────────────────────────────


def gds_to_gltf(
    gds_path: str,
    max_polygons: int = DEFAULT_POLY_BUDGET,
    layer_map: dict[tuple[int, int], dict[str, Any]] | None = None,
) -> tuple[bytes, list[dict[str, Any]]] | None:
    """Convert a GDS file to GLB binary + layer metadata.

    Returns ``(glb_bytes, layer_info_list)`` or ``None`` on failure.
    Each entry in *layer_info_list* has keys ``name``, ``color``, ``count``.

    ``layer_map`` is the (gds_layer, datatype) -> render-param map; when None it
    is resolved from the active deployment (PR6), defaulting to the sky130 map
    so behavior is unchanged.
    """
    if not _HAS_DEPS:
        logger.warning("3D viewer deps not installed (gdstk / pygltflib / mapbox-earcut)")
        return None

    if layer_map is None:
        layer_map = _active_layer_map()
    max_z = max(v["z"] + v["h"] for v in layer_map.values())

    lib = gdstk.read_gds(gds_path)
    tops = lib.top_level()
    if not tops:
        logger.warning("GDS has no top-level cell")
        return None
    top = tops[0]

    # ── Collect polygons per layer ──────────────────────────────
    layer_polys: dict[tuple[int, int], list[np.ndarray]] = {}
    for key in layer_map:
        raw = top.get_polygons(depth=None, layer=key[0], datatype=key[1])
        if raw:
            pts = [np.asarray(p.points) if hasattr(p, "points") else np.asarray(p) for p in raw]
            layer_polys[key] = pts

    if not layer_polys:
        logger.warning("No recognised PDK layer polygons in GDS")
        return None

    total = sum(len(v) for v in layer_polys.values())
    logger.info("GDS: %d polygons across %d layers", total, len(layer_polys))

    # ── Budget cap (keep highest-priority layers first) ─────────
    if total > max_polygons:
        sorted_keys = sorted(layer_polys, key=lambda k: layer_map[k]["pri"])
        kept: dict[tuple[int, int], list[np.ndarray]] = {}
        count = 0
        for key in sorted_keys:
            polys = layer_polys[key]
            room = max_polygons - count
            if room <= 0:
                break
            if len(polys) <= room:
                kept[key] = polys
                count += len(polys)
            else:
                kept[key] = polys[:room]
                count += room
                break
        layer_polys = kept
        logger.info("Budget-capped to %d polygons (%d layers)", count, len(kept))

    # ── Bounding box & scale ────────────────────────────────────
    all_pts = np.vstack([np.vstack(p) for p in layer_polys.values()])
    bb_min, bb_max = all_pts.min(axis=0), all_pts.max(axis=0)
    center = (bb_min + bb_max) / 2.0
    extent = max(float(bb_max[0] - bb_min[0]), float(bb_max[1] - bb_min[1]), 1.0)
    scale = 10.0 / extent
    z_scale = 10.0 * Z_EXAGGERATION / max_z

    # ── Materials, meshes, buffer ───────────────────────────────
    materials: list[Any] = []
    mat_map: dict[tuple[int, int], int] = {}
    layer_info: list[dict[str, Any]] = []

    binary = bytearray()
    buffer_views: list[Any] = []
    accessors_list: list[Any] = []
    meshes: list[Any] = []
    child_ids: list[int] = []

    sorted_keys = sorted(layer_polys, key=lambda k: layer_map[k]["z"])

    for key in sorted_keys:
        info = layer_map[key]
        r, g, b = info["color"]
        a: float = info.get("alpha", 1.0)

        mi = len(materials)
        mat_map[key] = mi
        materials.append(Material(
            pbrMetallicRoughness=PbrMetallicRoughness(
                baseColorFactor=[r, g, b, a],
                metallicFactor=0.1,
                roughnessFactor=0.8,
            ),
            alphaMode="BLEND" if a < 1.0 else "OPAQUE",
            name=info["name"],
        ))
        layer_info.append({"name": info["name"], "color": [r, g, b], "count": len(layer_polys[key])})

        # Build extruded mesh for this layer
        z = info["z"] * z_scale
        h = info["h"] * z_scale
        result = _build_layer_mesh(layer_polys[key], z, h, scale, center)
        if result is None:
            continue
        positions, indices = result

        # Pack positions into binary buffer (4-byte aligned)
        while len(binary) % 4:
            binary.append(0)
        pos_bv_idx = len(buffer_views)
        pos_bytes = positions.tobytes()
        buffer_views.append(BufferView(
            buffer=0, byteOffset=len(binary),
            byteLength=len(pos_bytes), target=_ARRAY_BUF,
        ))
        binary.extend(pos_bytes)

        # Pack indices
        while len(binary) % 4:
            binary.append(0)
        idx_bv_idx = len(buffer_views)
        idx_bytes = indices.tobytes()
        buffer_views.append(BufferView(
            buffer=0, byteOffset=len(binary),
            byteLength=len(idx_bytes), target=_ELEM_BUF,
        ))
        binary.extend(idx_bytes)

        # Accessors
        pos_acc = len(accessors_list)
        accessors_list.append(Accessor(
            bufferView=pos_bv_idx, componentType=_FLOAT,
            count=len(positions), type=_VEC3,
            max=positions.max(axis=0).tolist(),
            min=positions.min(axis=0).tolist(),
        ))
        idx_acc = len(accessors_list)
        accessors_list.append(Accessor(
            bufferView=idx_bv_idx, componentType=_UINT32,
            count=len(indices), type=_SCALAR,
            max=[int(indices.max())], min=[int(indices.min())],
        ))

        # Mesh (one per layer)
        len(meshes)
        meshes.append(Mesh(
            primitives=[Primitive(
                attributes=Attributes(POSITION=pos_acc),
                indices=idx_acc, material=mi, mode=_TRIANGLES,
            )],
            name=info["name"],
        ))
        child_ids.append(len(child_ids) + 1)  # +1 because root is node 0

    if not meshes:
        logger.warning("No renderable geometry produced from GDS")
        return None

    # Final buffer padding
    while len(binary) % 4:
        binary.append(0)

    # ── Assemble GLTF2 ─────────────────────────────────────────
    root_node = Node(name="root", children=child_ids)
    child_nodes = [Node(mesh=i, name=meshes[i].name) for i in range(len(meshes))]

    gltf = GLTF2(
        asset=Asset(version="2.0", generator="coresmith-asic-pipeline"),
        scene=0,
        scenes=[Scene(nodes=[0])],
        nodes=[root_node] + child_nodes,
        meshes=meshes,
        materials=materials,
        accessors=accessors_list,
        bufferViews=buffer_views,
        buffers=[Buffer(byteLength=len(binary))],
    )
    gltf.set_binary_blob(bytes(binary))

    # Save to .glb via temp file (pygltflib's save_to_bytes API varies by version)
    fd, tmp_path = tempfile.mkstemp(suffix=".glb")
    os.close(fd)
    try:
        gltf.save(tmp_path)
        glb_data = Path(tmp_path).read_bytes()
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    logger.info("GLB: %.2f MB, %d meshes, %d materials", len(glb_data) / 1e6, len(meshes), len(materials))
    return glb_data, layer_info


# ── HTML generation ─────────────────────────────────────────────


def generate_3d_html(
    gds_path: str,
    block_name: str,
    project_root: str,
) -> str | None:
    """Generate a standalone 3D viewer HTML file from a GDS.

    Returns the complete HTML string, or ``None`` on failure.
    Best-effort: never raises.
    """
    if not _HAS_DEPS:
        logger.warning("3D viewer deps not installed -- skipping 3D view")
        return None

    try:
        result = gds_to_gltf(gds_path)
    except Exception as exc:
        logger.error("GDS → glTF conversion failed: %s", exc, exc_info=True)
        return None

    if result is None:
        return None

    glb_data, layer_info = result
    b64 = base64.b64encode(glb_data).decode("ascii")

    from jinja2 import Environment, FileSystemLoader

    tmpl_dir = Path(__file__).resolve().parents[2] / "langchain" / "prompts"
    env = Environment(loader=FileSystemLoader(str(tmpl_dir)), autoescape=False)
    template = env.get_template("3d_viewer.html.j2")

    return template.render(
        design_name=block_name,
        gltf_base64=b64,
        layer_info=json.dumps(layer_info),
        total_polygons=sum(li["count"] for li in layer_info),
        glb_size_mb=round(len(glb_data) / 1_000_000, 2),
    )


# ── 2D layout rendering (SVG + PNG) ─────────────────────────────

_SKY130_SVG_STYLE: dict[tuple[int, int], dict[str, str]] = {
    (64, 20):  {"fill": "rgba(80,80,80,0.3)",     "stroke": "none"},
    (64, 5):   {"fill": "rgba(80,80,80,0.15)",    "stroke": "none"},
    (64, 59):  {"fill": "rgba(80,80,80,0.1)",     "stroke": "none"},
    (65, 20):  {"fill": "rgba(34,139,34,0.4)",    "stroke": "none"},
    (65, 44):  {"fill": "rgba(34,139,34,0.35)",   "stroke": "none"},
    (66, 20):  {"fill": "rgba(190,90,120,0.5)",   "stroke": "none"},
    (66, 44):  {"fill": "rgba(128,128,128,0.3)",  "stroke": "none"},
    (67, 20):  {"fill": "rgba(255,207,140,0.5)",  "stroke": "none"},
    (67, 5):   {"fill": "rgba(255,207,140,0.2)",  "stroke": "none"},
    (67, 44):  {"fill": "rgba(128,128,128,0.25)", "stroke": "none"},
    (68, 20):  {"fill": "rgba(41,98,212,0.55)",   "stroke": "none"},
    (68, 5):   {"fill": "rgba(41,98,212,0.25)",   "stroke": "none"},
    (68, 44):  {"fill": "rgba(128,128,128,0.3)",  "stroke": "none"},
    (69, 20):  {"fill": "rgba(167,191,230,0.5)",  "stroke": "none"},
    (69, 16):  {"fill": "rgba(167,191,230,0.2)",  "stroke": "none"},
    (69, 44):  {"fill": "rgba(128,128,128,0.25)", "stroke": "none"},
    (70, 20):  {"fill": "rgba(115,166,115,0.5)",  "stroke": "none"},
    (70, 16):  {"fill": "rgba(115,166,115,0.2)",  "stroke": "none"},
    (70, 44):  {"fill": "rgba(128,128,128,0.2)",  "stroke": "none"},
    (71, 20):  {"fill": "rgba(191,140,89,0.5)",   "stroke": "none"},
    (71, 16):  {"fill": "rgba(191,140,89,0.2)",   "stroke": "none"},
    (78, 44):  {"fill": "rgba(128,128,128,0.15)", "stroke": "none"},
    (81, 4):   {"fill": "rgba(200,200,200,0.05)", "stroke": "none"},
    (93, 44):  {"fill": "rgba(200,200,200,0.03)", "stroke": "none"},
    (94, 20):  {"fill": "rgba(180,180,180,0.05)", "stroke": "none"},
    (95, 20):  {"fill": "rgba(180,180,180,0.05)", "stroke": "none"},
}

DEFAULT_PNG_WIDTH = 2048


def generate_2d_layout(
    gds_path: str,
    block_name: str,
    project_root: str,
    png_width: int = DEFAULT_PNG_WIDTH,
) -> tuple[str, str] | None:
    """Render a GDS file to SVG and PNG (2D floorplan view).

    Uses gdstk's ``write_svg`` with Sky130-themed layer colors on a dark
    background.  The SVG is then rasterized to PNG via cairosvg (if
    available).

    Args:
        gds_path: Path to the ``.gds`` file.
        block_name: Design / block name (used in filenames).
        project_root: Project root directory.
        png_width: Output PNG width in pixels.

    Returns:
        ``(svg_path, png_path)`` on success, ``None`` on failure.
        If cairosvg is not installed the PNG path will be empty but the
        SVG is still produced.  Best-effort: never raises.
    """
    if not _HAS_DEPS:
        logger.warning("gdstk not installed -- skipping 2D layout render")
        return None

    try:
        lib = gdstk.read_gds(gds_path)
    except Exception as exc:
        logger.error("Failed to read GDS for 2D layout: %s", exc)
        return None

    tops = lib.top_level()
    if not tops:
        logger.warning("GDS has no top-level cell -- skipping 2D layout")
        return None
    top = tops[0]

    output_dir = Path(project_root) / "chip_finish"
    output_dir.mkdir(parents=True, exist_ok=True)

    svg_path = str(output_dir / f"{block_name}_layout.svg")
    png_path = ""

    try:
        top.write_svg(
            svg_path,
            scaling=50,
            precision=3,
            shape_style=_SKY130_SVG_STYLE,
            background="#1a1a2e",
            pad="2%",
            sort_function=lambda p1, p2: p1.layer < p2.layer,
        )
        logger.info(
            "2D SVG: %s (%.1f MB)",
            svg_path,
            os.path.getsize(svg_path) / 1e6,
        )
    except Exception as exc:
        logger.error("SVG generation failed: %s", exc, exc_info=True)
        return None

    try:
        import cairosvg

        png_file = str(output_dir / f"{block_name}_layout.png")
        cairosvg.svg2png(
            url=svg_path,
            write_to=png_file,
            output_width=png_width,
        )
        png_path = png_file
        logger.info(
            "2D PNG: %s (%.1f MB)",
            png_path,
            os.path.getsize(png_path) / 1e6,
        )
    except ImportError:
        logger.info("cairosvg not installed -- PNG conversion skipped")
    except Exception as exc:
        logger.warning("PNG conversion failed (non-fatal): %s", exc)

    return svg_path, png_path
