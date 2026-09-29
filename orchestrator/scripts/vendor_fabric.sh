#!/usr/bin/env bash
# Vendor the pulp-platform AXI fabric IP (SHL-0.51) into
# orchestrator/langgraph/rtl_lib/fabric/sv/ at pinned tags (B1).
#
# What is vendored: the SystemVerilog sources of the closure the fabric
# generator instantiates (axi_xbar, axi_mux/demux, axi_lite_xbar,
# axi_to_axi_lite, axi_lite_to_apb, axi_err_slv, axi_id_remap,
# axi_dw_converter, axi_cut, axi_atop_filter and the common_cells they use).
# The generator elaborates them per fabric instance with yosys-slang
# (bundled in oss-cad-suite's Yosys) and writes plain Verilog-2001, so the
# engine's `read_verilog` synthesis and Verilator simulation never see
# SystemVerilog. Interface-based wrapper modules (`*_intf`) are stripped:
# they need `AXI_BUS` interfaces the generator does not use.
#
# Usage: scripts/vendor_fabric.sh [--from <dir with axi/ and common_cells/>]
set -euo pipefail
AXI_TAG=v0.39.9
CC_TAG=v1.37.0            # the common_cells version axi $AXI_TAG's Bender.yml pins
HERE="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$HERE/langgraph/rtl_lib/fabric/sv"
WORK="$(mktemp -d)"
FROM=""
if [ "${1:-}" = "--from" ]; then FROM="$2"; fi
if [ -n "$FROM" ]; then
  cp -r "$FROM/axi" "$WORK/axi"; cp -r "$FROM/common_cells" "$WORK/common_cells"
else
  git clone -q --depth 1 -b "$AXI_TAG" https://github.com/pulp-platform/axi.git "$WORK/axi"
  git clone -q --depth 1 -b "$CC_TAG" https://github.com/pulp-platform/common_cells.git "$WORK/common_cells"
fi
AXI_SHA=$(git -C "$WORK/axi" rev-parse HEAD)
CC_SHA=$(git -C "$WORK/common_cells" rev-parse HEAD)

rm -rf "$DEST"; mkdir -p "$DEST/axi/include/axi" "$DEST/common_cells/include/common_cells"
cp "$WORK/axi/include/axi/typedef.svh" "$WORK/axi/include/axi/assign.svh" "$DEST/axi/include/axi/"
cp "$WORK/common_cells/include/common_cells"/*.svh "$DEST/common_cells/include/common_cells/"
cp "$WORK/axi/LICENSE" "$DEST/axi/LICENSE"
cp "$WORK/common_cells/LICENSE" "$DEST/common_cells/LICENSE"
CC_FILES="cf_math_pkg rr_arb_tree lzc onehot_to_bin spill_register_flushable spill_register stream_register fifo_v3 counter delta_counter addr_decode_dync addr_decode id_queue stream_fork stream_fork_dynamic stream_join stream_join_dynamic stream_mux stream_demux stream_arbiter stream_arbiter_flushable fall_through_register"
for f in $CC_FILES; do cp "$WORK/common_cells/src/$f.sv" "$DEST/common_cells/$f.sv"; done
cp "$WORK/common_cells/src/deprecated/fifo_v2.sv" "$DEST/common_cells/fifo_v2.sv"
AXI_FILES="axi_pkg axi_demux_simple axi_demux axi_mux axi_err_slv axi_id_prepend axi_xbar axi_xbar_unmuxed axi_atop_filter axi_cut axi_multicut axi_lite_xbar axi_lite_mux axi_lite_demux axi_to_axi_lite axi_lite_to_apb axi_id_remap axi_burst_splitter axi_burst_splitter_gran axi_dw_converter axi_dw_downsizer axi_dw_upsizer"
for f in $AXI_FILES; do cp "$WORK/axi/src/$f.sv" "$DEST/axi/$f.sv"; done

# Patch: drop the interface-wrapper modules (module *_intf ... endmodule).
python3 - "$DEST/axi" <<'PY'
import pathlib, re, sys
pat = re.compile(r"\nmodule\s+\w+_intf\b.*?\nendmodule\b", re.DOTALL)
for f in sorted(pathlib.Path(sys.argv[1]).glob("*.sv")):
    s = f.read_text()
    s2, n = pat.subn("\n// [coresmith vendoring] interface wrapper module removed (needs AXI_BUS interfaces)", s)
    if n:
        f.write_text(s2)
        print(f"  stripped {n} *_intf module(s) from {f.name}")
PY

# Manifest with per-file hashes (the regression checks for drift).
python3 - "$DEST" "$AXI_TAG" "$AXI_SHA" "$CC_TAG" "$CC_SHA" <<'PY'
import hashlib, json, pathlib, sys, time
dest, axi_tag, axi_sha, cc_tag, cc_sha = sys.argv[1:6]
root = pathlib.Path(dest)
files = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
         for p in sorted(root.rglob("*")) if p.is_file() and p.name != "MANIFEST.json"}
manifest = {
    "axi": {"repo": "https://github.com/pulp-platform/axi", "tag": axi_tag, "commit": axi_sha, "license": "SHL-0.51"},
    "common_cells": {"repo": "https://github.com/pulp-platform/common_cells", "tag": cc_tag, "commit": cc_sha, "license": "SHL-0.51"},
    "converter": "yosys-slang (plugin slang in oss-cad-suite Yosys) at fabric-generation time",
    "patches": ["*_intf wrapper modules stripped from axi/*.sv"],
    "defines": ["SYNTHESIS", "COMMON_CELLS_ASSERTS_OFF", "VERILATOR", "TARGET_SYNTHESIS"],
    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "files": files,
}
(root / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
print(f"vendored {len(files)} files -> {root}")
PY
rm -rf "$WORK"
