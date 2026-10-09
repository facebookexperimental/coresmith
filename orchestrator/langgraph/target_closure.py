# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tool receipts for a module build's FRD targets (the block graph's side
of :mod:`orchestrator.state_store.module_targets`).

Every number a target is judged by comes from a file a tool wrote during
this build, read back and hashed by the engine:

* the acceptance simulation: cocotb's ``results.xml`` and the
  ``measurements.jsonl`` the acceptance tests appended to through
  :func:`orchestrator.harness.measure.record`, with the hashes of every
  simulated input (:func:`capture_sim_receipt`);
* the synthesized netlist's area from yosys ``stat -liberty`` plus, for the
  ``total`` scope, the Liberty area of the memory macros the netlist's
  wrappers bind to (:func:`measure_area`);
* OpenSTA ``report_power`` on the mapped netlist with the build's liberty
  (and the bound macros' liberties), SDC clock and the declared activity
  (:func:`measure_power`).

Each evaluated candidate keeps an immutable copy of its evidence files and
its receipts are rebound to those copies (:func:`snapshot_evidence`), so a
later attempt that overwrites the working files never changes what a
recorded candidate was judged by. Validation lives with the evaluation
(``module_targets.sim_receipt_problems`` / ``eda_receipt_problems``).
Nothing here decides routing; the graph nodes do.
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path

from orchestrator.state_store import module_targets as MT
from orchestrator.state_store.builds import file_sha256, snapshot_artifacts


# ---------------------------------------------------------------- simulation
def capture_sim_receipt(sim_dir, *, inputs: list[str], module: str, build_id: str = "", attempt: int = 0) -> dict:
    """The receipt of one passing block simulation, captured by the engine
    right after the run from the files it wrote: ``results.xml``,
    ``measurements.jsonl`` and the hashes of every simulated input
    (sources, acceptance files, supplemental tests, imported modules)."""
    sim_dir = Path(sim_dir)
    xml = sim_dir / "results.xml"
    mfile = sim_dir / "measurements.jsonl"
    out = {"kind": MT.SIM_RECEIPT, "build_id": build_id, "attempt": int(attempt or 0), "module": module,
           "sim_dir": str(sim_dir), "results_xml": str(xml), "results_sha256": file_sha256(xml),
           "measurements": str(mfile), "measurements_sha256": file_sha256(mfile),
           "inputs": {str(Path(p).resolve()): file_sha256(p) for p in dict.fromkeys(inputs) if p},
           "captured_at": time.time(), "error": None}
    if out["results_sha256"] is None:
        out["error"] = f"the simulation wrote no {xml.name}"
    return out


_MEASURE_REF = re.compile(r"harness\.measure|harness\s+import\s+measure|measurements\.jsonl|CORESMITH_MEASUREMENTS")


def supplemental_problem(text: str) -> str | None:
    """Supplemental tests close coverage; they never produce target evidence."""
    if _MEASURE_REF.search(text or ""):
        return ("supplemental tests may not record measurements (harness.measure / measurements.jsonl): targets are "
                "measured only by the Architect's acceptance tests")
    return None


def failing_modules(results_xml) -> set[str]:
    """The cocotb modules (``testcase classname``) with a failing or erroring
    test in ``results_xml``."""
    import xml.etree.ElementTree as ET
    try:
        root = ET.parse(str(results_xml)).getroot()
    except (OSError, ET.ParseError):
        return set()
    out = set()
    for tc in root.iter("testcase"):
        if tc.find("failure") is not None or tc.find("error") is not None:
            out.add(str(tc.get("classname") or "").split(".")[-1])
    return out


# ---------------------------------------------------------------- synthesis
_KEYWORDS = {"module", "input", "output", "inout", "wire", "reg", "assign", "always", "initial", "endmodule",
             "parameter", "localparam", "function", "task", "begin", "end", "if", "else", "case", "for", "generate",
             "supply0", "supply1", "tri", "integer", "genvar", "defparam", "specify", "endspecify"}
_LIB_CELLS: dict[tuple, set[str]] = {}
_IDENT = r"(?:\\\S+|[A-Za-z_][\w$]*)"


def _liberty_cells(liberty: str) -> set[str]:
    try:
        st = os.stat(liberty)
        key = (liberty, st.st_mtime_ns, st.st_size)
    except OSError:
        return set()
    if key not in _LIB_CELLS:
        text = Path(liberty).read_text(errors="replace")
        _LIB_CELLS[key] = set(re.findall(r'cell\s*\(\s*"?([^")\s]+)"?\s*\)\s*\{', text))
    return _LIB_CELLS[key]


def netlist_instances(netlist_text: str) -> list[str]:
    """The cell/module type of every instance in a structural netlist
    (parameter blocks ``#( ... )`` may span lines)."""
    text = re.sub(r"//[^\n]*", "", netlist_text or "")
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    out = []
    for m in re.finditer(rf"(?m)^\s*({_IDENT})\s*(#\s*\()?", text):
        typ = m.group(1)
        if typ in _KEYWORDS or typ.startswith("$") or typ.startswith("`"):
            continue
        i = m.end()
        if m.group(2):
            depth = 1
            while i < len(text) and depth:
                depth += {"(": 1, ")": -1}.get(text[i], 0)
                i += 1
        im = re.match(rf"\s*({_IDENT})\s*(\[[^\]]*\]\s*)?\(", text[i:])
        if im:
            out.append(typ)
    return out


def netlist_blackboxes(netlist_text: str, liberty: str, extra_libs: list[str] | None = None) -> list[str]:
    """Instantiated types of a mapped netlist that are neither a cell of the
    liberties nor a module the netlist defines: black boxes (memory macros,
    unmapped IP) whose area and power the module netlist does not contain."""
    defined = set(re.findall(rf"(?m)^\s*module\s+({_IDENT})", netlist_text or ""))
    cells = set(_liberty_cells(liberty))
    for lib in extra_libs or []:
        cells |= _liberty_cells(lib)
    return sorted({t for t in netlist_instances(netlist_text) if t not in defined and t not in cells})


def _selection_fields(synth_result: dict) -> dict:
    """The netlist-selection record every eda receipt carries: the netlist the
    timing verdict was measured on (its sha must be the receipt's netlist)
    and, when a repaired netlist was selected, the yosys original kept as
    evidence."""
    sel = synth_result.get("netlist_selection") or {}
    out = {"timing_netlist_sha256": synth_result.get("timing_netlist_sha256"),
           "netlist_variant": sel.get("variant"), "netlist_selection_error": sel.get("error")}
    if sel.get("timing_report_path") and Path(sel["timing_report_path"]).is_file():
        out.update(timing_report_path=sel["timing_report_path"], timing_report_sha256=file_sha256(sel["timing_report_path"]))
    for key, path_key, sha_key in (("original_netlist", "original_netlist_path", "original_netlist_sha256"),
                                   ("original_report", "original_report_path", "original_report_sha256")):
        if sel.get(key) and Path(sel[key]).is_file():
            out.update({path_key: sel[key], sha_key: file_sha256(sel[key])})
    return out


def _synth_context(synth_result: dict) -> tuple[str, str, str, str | None]:
    """``(netlist, report, liberty, problem)`` of a synthesis result."""
    netlist = str(synth_result.get("netlist_path") or "")
    report = str(synth_result.get("report_path") or "")
    liberty = str(synth_result.get("liberty_path") or "")
    generic = (os.environ.get("CORESMITH_SYNTH_GENERIC", "") or "").strip().lower() in {"1", "true", "yes", "on"}
    if generic:
        return netlist, report, liberty, "CORESMITH_SYNTH_GENERIC: generic gates carry no liberty area or power"
    if not liberty or not Path(liberty).is_file():
        return netlist, report, liberty, "no liberty file: synthesis was generic"
    if not netlist or not Path(netlist).is_file():
        return netlist, report, liberty, "no synthesized netlist"
    return netlist, report, liberty, None


def resolve_macros(net_text: str, liberty: str) -> dict:
    """The memory macros of a mapped netlist, by explicit identity only:
    ``cs_sram`` wrapper instances bound to a macro the way the pre-layout STA
    binds them (``macro_sta``, by geometry), and instances whose cell name IS
    a registered macro (``macro_registry``). ``{macros: [{name, lib, count,
    kind?, width?, depth?}], unresolved: [...], libs: [...], netlist: <text
    with wrappers rebound>, wrappers: <wrapper modules>}``; every other black
    box is unresolved -- no geometry is guessed."""
    from collections import Counter

    from orchestrator.langgraph import macro_sta
    out = {"macros": [], "unresolved": [], "libs": [], "netlist": net_text, "wrappers": ""}
    boxes = netlist_blackboxes(net_text, liberty)
    if not boxes:
        return out
    try:
        from orchestrator.langgraph.macro_registry import discover_macros
        registry = discover_macros() or {}
    except Exception:  # noqa: BLE001 - no registry: every black box stays unresolved
        registry = {}
    counts = Counter(netlist_instances(net_text))
    wrappers = set(macro_sta._WRAPPERS)
    if set(boxes) & wrappers:
        binding = macro_sta.bind_netlist_macros(net_text)
        out["netlist"], out["wrappers"] = binding.netlist, binding.wrappers
        geo = Counter((k, w, d) for _s, _e, k, w, d in macro_sta.find_instances(net_text))
        out["unresolved"] += [f"{k} {w}x{d} (no macro)" for k, w, d in binding.unresolved]
        for kind, w, d, name in binding.bound:
            info = registry.get(name)
            lib = getattr(info, "lib", "") or next((p for p in binding.libs if MT.liberty_cell_area(p, name) is not None
                                                    or MT._liberty_cell_body(p, name) is not None), "")
            out["macros"].append({"name": name, "lib": lib, "count": geo.get((kind, w, d), 0), "kind": kind,
                                  "width": w, "depth": d})
    for box in boxes:
        if box in wrappers:
            continue
        info = registry.get(box)
        lib = getattr(info, "lib", "") or ""
        if info is None or not lib or MT._liberty_cell_body(lib, box) is None:
            out["unresolved"].append(f"{box} (not a registered macro with a Liberty model)")
            continue
        out["macros"].append({"name": box, "lib": lib, "count": counts.get(box, 0)})
    for m in out["macros"]:
        if m["lib"] and m["lib"] not in out["libs"]:
            out["libs"].append(m["lib"])
    out["unresolved"] = sorted(set(out["unresolved"]))
    return out


def measure_area(synth_result: dict, *, area_scope: str = "total") -> dict:
    """Area (um2) of the build's synthesized netlist: the standard-cell area
    yosys ``stat -liberty`` reported, plus -- for ``area_scope`` ``total`` --
    the Liberty ``area`` of every memory macro the netlist's wrappers bind to
    (the binding the pre-layout STA uses). A black box that binds to no macro
    with a known area makes the TOTAL unmeasured; the ``std_cell`` scope is
    the explicit standard-cell subtotal. ``{value, receipt}`` or ``{error}``."""
    netlist, report, liberty, problem = _synth_context(synth_result)
    if problem:
        return {"error": problem}
    try:
        text = Path(report).read_text(errors="replace")
    except OSError:
        return {"error": f"no synthesis report at {report}"}
    std = MT.parse_chip_area(text)
    if std is None:
        return {"error": "the yosys report has no 'Chip area' line (stat -liberty did not run)"}
    net_text = Path(netlist).read_text(errors="replace")
    res = resolve_macros(net_text, liberty)
    macros, unresolved = [], list(res["unresolved"])
    for m in res["macros"]:
        area = MT.liberty_cell_area(m["lib"], m["name"]) if m["lib"] else None
        if area is None:
            unresolved.append(f"{m['name']} (no Liberty area)")
            continue
        macros.append({**m, "area_um2": area})
    macro_area = sum(m["area_um2"] * m["count"] for m in macros)
    receipt = {"kind": MT.AREA_RECEIPT, "tool": "yosys stat -liberty", "area_scope": area_scope,
               "report_path": report, "report_sha256": file_sha256(report), "netlist_path": netlist,
               "netlist_sha256": file_sha256(netlist), "liberty": liberty, "liberty_sha256": file_sha256(liberty),
               "std_cell_um2": std, "macros": macros, "macro_um2": macro_area, "macros_unresolved": sorted(unresolved),
               "macro_libs": [{"path": p, "sha256": file_sha256(p)} for p in res["libs"]],
               **_selection_fields(synth_result)}
    if area_scope == "std_cell":
        return {"value": std, "receipt": receipt}
    if unresolved:
        return {"error": "the netlist instantiates cells with no known area (" + ", ".join(unresolved[:4])
                         + (" ..." if len(unresolved) > 4 else "") + "): the total area is not measured; bind the "
                         "target with --args '{\"area_scope\":\"std_cell\"}' to judge the standard-cell subtotal",
                "receipt": receipt}
    receipt["total_um2"] = std + macro_area
    return {"value": std + macro_area, "receipt": receipt}


def measure_power(synth_result: dict, top: str, *, activity: float, duty: float, report_path: str) -> dict:
    """Estimated power (mW) of the mapped netlist from OpenSTA
    ``report_power``: the build's liberty (plus the Liberty of every memory
    macro the netlist's wrappers bind to) and SDC clock, vectorless with
    ``activity`` toggles per clock and ``duty`` on the primary inputs,
    propagated by OpenSTA. ``{value, receipt}`` (kind opensta_report_power:
    basis estimated, method, library, clock, activity) or ``{error}``."""
    from orchestrator.langgraph.ppa_check import run_power_estimate
    netlist, _report, liberty, problem = _synth_context(synth_result)
    if problem:
        return {"error": problem}
    sdc = str(synth_result.get("sdc_path") or "")
    if not sdc or not Path(sdc).is_file():
        return {"error": "no SDC: the clock the power is estimated at is unknown"}
    res = run_power_estimate(netlist, sdc, liberty, top, activity=activity, duty=duty, report_path=report_path)
    if not isinstance(res, dict) or res.get("error") or res.get("power_mw") is None:
        return {"error": (res or {}).get("error") or "OpenSTA produced no power figure"}
    return {"value": float(res["power_mw"]),
            "receipt": {"kind": MT.POWER_RECEIPT, "tool": "OpenSTA report_power", "basis": "estimated",
                        "method": res.get("method"), "activity": float(activity), "duty": float(duty),
                        "clock_period_ns": res.get("clock_period_ns"),
                        "liberty": liberty, "liberty_sha256": file_sha256(liberty),
                        "macro_libs": res.get("macro_libs") or [], "macros": res.get("macros") or [],
                        "netlist_path": netlist, "netlist_sha256": file_sha256(netlist),
                        "sdc_path": sdc, "sdc_sha256": file_sha256(sdc),
                        "report_path": res.get("report_path"), "report_sha256": res.get("report_sha256"),
                        "internal_mw": res.get("internal_mw"), "switching_mw": res.get("switching_mw"),
                        "leakage_mw": res.get("leakage_mw"), "sta": res.get("sta"),
                        **_selection_fields(synth_result)}}


# ---------------------------------------------------------------- candidates
_EVIDENCE_KEYS = (("results_xml", "results_sha256"), ("measurements", "measurements_sha256"),
                  ("report_path", "report_sha256"), ("netlist_path", "netlist_sha256"), ("sdc_path", "sdc_sha256"),
                  ("timing_report_path", "timing_report_sha256"), ("original_netlist_path", "original_netlist_sha256"),
                  ("original_report_path", "original_report_sha256"))


def snapshot_evidence(root, build_id: str, tag: str, sim_receipt: dict | None,
                      eda: dict | None) -> tuple[dict | None, dict, list[str]]:
    """Keep an immutable copy of every evidence file a candidate's receipts
    name (results.xml, measurements.jsonl, synthesis report, netlist, SDC,
    power report) under ``.coresmith/builds/<build>/candidates/<tag>/evidence``
    and return the receipts REBOUND to those copies (the working files are
    overwritten by the next attempt). A file whose bytes no longer match the
    hash the receipt recorded is not copied: the error is returned and the
    receipt keeps pointing at the changed file, so it fails validation."""
    files: dict[str, str | None] = {}
    recs = ([sim_receipt] if isinstance(sim_receipt, dict) else []) + [
        m.get("receipt") for m in (eda or {}).values() if isinstance(m, dict) and isinstance(m.get("receipt"), dict)]
    for rec in recs:
        for path_key, sha_key in _EVIDENCE_KEYS:
            p, sha = rec.get(path_key), rec.get(sha_key)
            if isinstance(p, str) and p and sha:
                files[p] = sha
    snap = snapshot_artifacts(root, f"{build_id}/candidates/{tag}/evidence", files) if files else \
        {"files": {}, "errors": [], "dir": ""}

    def rebind(rec: dict | None) -> dict | None:
        if not isinstance(rec, dict):
            return rec
        out = dict(rec)
        for path_key, _sha_key in _EVIDENCE_KEYS:
            p = rec.get(path_key)
            if isinstance(p, str) and p in snap["files"]:
                out[path_key] = snap["files"][p]
                out.setdefault("source_paths", {})[path_key] = p
        return out
    eda_out = {k: ({**m, "receipt": rebind(m.get("receipt"))} if isinstance(m, dict) else m)
               for k, m in (eda or {}).items()}
    return rebind(sim_receipt), eda_out, list(snap["errors"])
