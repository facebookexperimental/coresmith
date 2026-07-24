# Sky130 backend flow — hardening notes (armD GDS campaign)

Findings from driving the armD codec to clean GDS + LVS on Sky130 (OpenROAD /
klayout / magic / netgen). Each is a real, live-proven fix. Items marked
**APPLIED** are in the reference flow; items marked **CHECKLIST** are
per-run / tool-level actions the backend agent must apply (they are
ordering-sensitive or environment-specific and must be validated on hardware —
the local test env has no OpenROAD, so P&R-flow edits are not blind-committed).

Numbering matches the armD DRIVER_LOG backend defect ledger (B1–B10).

## B1 — SDC `remove_from_collection` unsupported  · CHECKLIST
OpenROAD's SDC reader rejected `remove_from_collection`. Constraint scripts must
avoid it; build the exclusion set directly (e.g. filter with `get_ports`/`lassign`)
rather than subtracting collections.

## B2 — `place_macro` name lookup breaks on generate-bracket names  · CHECKLIST
`place_macro -macro_name g_payload_bank[0].u_mem ...` fails MPL-0020: the bracketed
hierarchical instance name from a Verilog `generate` block is not resolvable by
name. Fix: place via the ODB API instead —
`set inst [[ord::get_db_block] findInst $name]; $inst setOrigin $x $y; $inst
setOrient $orient; $inst setPlacementStatus FIRM`. The reference `place_macro`
call (§2 of pnr_reference.tcl) works for flat names; the ODB path is required
whenever a macro instance lives inside a generate scope.

## B3 — no `repair_design` → high-fanout nets stall routing  · CHECKLIST
Without a max-fanout repair, a 12–14k-terminal unbuffered net (reset / broadcast
bus) stalled detailed routing at ~255k DRC violations. Fix: run
`repair_design -max_fanout 20` (and let it fix max-cap/max-slew) AFTER
`set_wire_rc` + `estimate_parasitics -placement` and BEFORE CTS — the ordering
matters (repair_design needs parasitics; it is a datapath/pre-CTS step, not a
post-CTS one), which is why it is a checklist item rather than a blind edit to
the reference flow. Add `set_max_fanout 20 [current_design]` in the SDC too.

## B4 — narrow macro channels → GRT congestion  · CHECKLIST
40 µm channels between the SRAM macros congested global route. Widen to ~130 µm
and add placement blockages/halos around macros. Design-specific (depends on
macro count/size), so it is a floorplan-tuning knob, not a fixed reference value.

## B5 — no placement padding → local routing hotspots  · CHECKLIST
A wide read-mux (e.g. a framer's 25:1 met1 mux) created a local hotspot. Fix:
`set_placement_padding -global 3` (or targeted `-masters`/`-instances`) before
detailed placement. Global padding hurts density on designs that don't need it,
so apply per-run rather than baking a global default.

## B6 — apt magic too old for the current sky130A tech  · CHECKLIST
apt `magic` 8.3.105 could not read the current `sky130A.tech`. Build magic from
source (8.3.670+). Environment/tooling action.

## B7 — flatten-DRC too slow at scale → use klayout + hierarchical extract  · CHECKLIST
The engine's `generate_drc_tcl` flattens ~490k instances → magic DRC ran >19 min
unfinished. Use klayout for the sign-off DRC deck, and a LEAN hierarchical magic
`extract` (extract all → ext2spice) for LVS instead of a flat extract. Magic FEOL
slivers on the stripped PDK are known artifacts — do not chase them.

## B8 — `catch {write_spef}` emits no SPEF; RCX rules absent from PDK  · CHECKLIST
`write_spef` silently produced nothing (no RCX rules in the volare PDK). Final
timing sign-off must not depend on a SPEF that may be empty; run STA on the
routed DB (`estimate_parasitics -global_routing`) and treat missing-SPEF as a
degraded-but-explicit condition, not a silent pass.

## B9 — apt netgen 1.5.133 Verilog parser breaks on OpenROAD escaped ids  · CHECKLIST
netgen-lvs 1.5.133 choked on OpenROAD's escaped-identifier bit-selects
(`.B2(\u_x.w [84])`) with "unknown module". Build netgen 1.5.322+ from source;
the compare then runs in seconds. For a fast logical LVS, filter fillers/decaps/
taps + normalize power symmetrically on both sides before the compare.

## B10 — `write_verilog -include_pwr_gnd` omits power pins post-global_connect  · APPLIED
CTS + repair insert buffers AFTER the PDN-stage `global_connect`; the exported
`-include_pwr_gnd` netlist then lacks VPWR/VGND on those cells and they
LVS-mismatch as power-disconnected. Fix (now in pnr_reference.tcl §14): re-run
`global_connect` immediately before `write_verilog`.

---

## B11 — backend synth does not exclude `lpflow_*` isolation cells  · CHECKLIST
Confirmed NOT currently in the flow (no `dont_use` in the backend synth path as
of this writing). sky130 `lpflow_*` isolation cells caused an mcu3 LVS mismatch
after an otherwise-clean synth/PnR/DRC. The backend synth must set
`dont_use {sky130_fd_sc_hd__lpflow_* ...}` (plus probe cells). This is a safe,
high-value addition once the backend synth dont-use list is located (it is
LLM-driven in `backend_synth*.md` today, so the exclusion belongs in that prompt
or a synth constraints file).
- Related PDN policy from a prior video_codec backend: PDN `-max_columns 2` +
  `set_dont_use` on probe cells were needed to reach 0 DRC.
