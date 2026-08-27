"""First-class SFT dataset emission from a coresmith run.

Every completed run writes a self-describing ``<run>/sft/`` folder: JSONL
pair files under ``sft/pairs/``, a ``manifest.json`` with counts and
provenance, and a ``README.md`` that tells a human exactly what the folder
is -- a LABELED dataset, where the label is the pipeline's own verification
verdict (``chip`` when the block shipped inside a chip that passed
integration_dv + validation_dv with a PASS signoff; ``block_dv`` when the
block passed its own DV).

Emission is generic (any design), deterministic (reads only on-disk run
artifacts -- no LLM, no state), and non-fatal (the pipeline never fails on
an emission error). Default OFF; opt in with ``CORESMITH_EMIT_SFT=1``. For
archived runs, call ``emit_sft_dataset(run_dir)`` directly.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Optional

_SYSTEM_RTL = (
    "You are an expert digital design engineer. Implement the block described "
    "by the microarchitecture specification as synthesizable Verilog-2005. "
    "Honor every mandatory constraint verbatim."
)
_SYSTEM_TB = (
    "You are an expert verification engineer. Write a cocotb testbench for "
    "the block described by the microarchitecture specification and RTL below."
)
_SYSTEM_CHIP_TOP = (
    "You are an expert integration engineer. Assemble the chip-level top "
    "module that instantiates and wires the blocks below according to the "
    "block diagram. Synthesizable Verilog-2005, no logic beyond wiring/glue."
)
_SYSTEM_SPEC = (
    "You are an expert microarchitect. Write the complete microarchitecture "
    "specification for the block described below, covering interfaces, "
    "timing, storage, FSMs, and a verification plan."
)


def sft_enabled() -> bool:
    """CORESMITH_EMIT_SFT (default OFF): opt in with =1 to emit
    ``<run>/sft/`` at the final_report stage."""
    return (os.environ.get("CORESMITH_EMIT_SFT", "0") or "0") == "1"


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _find_block_rtl(root: Path, block: str) -> Optional[Path]:
    hits = sorted(root.glob(f"rtl/**/{block}.v"))
    return hits[0] if hits else None


def _module_header(rtl_text: str) -> str:
    """The first ``module ... );`` header -- the block's port contract."""
    m = re.search(r"^\s*module\s+[\s\S]*?\)\s*;", rtl_text, re.M)
    return m.group(0) if m else ""


def _chip_verified(root: Path) -> bool:
    report = _read_json(root / "final_report.json") or {}
    return (report.get("signoff") or {}).get("status") == "PASS"


def _block_labels(root: Path, block: str) -> dict:
    bdir = root / ".coresmith" / "blocks" / block
    attempts = _read_json(bdir / "attempt_history.json")
    cov = _read_json(bdir / "coverage.json") or {}
    cov_pct = None
    if isinstance(cov, dict):
        for k in ("percent", "pct", "aggregate", "line_pct"):
            if isinstance(cov.get(k), (int, float)):
                cov_pct = cov[k]
                break
    return {
        "attempts": len(attempts) if isinstance(attempts, list) else None,
        "coverage_pct": cov_pct,
    }


def _constraints_text(root: Path, block: str) -> str:
    entries = _read_json(
        root / ".coresmith" / "blocks" / block / "constraints.json")
    if not isinstance(entries, list) or not entries:
        return ""
    rules = [str(e.get("rule", "")).strip() for e in entries
             if isinstance(e, dict) and e.get("rule")]
    if not rules:
        return ""
    return ("\n\n## MANDATORY BLOCK CONSTRAINTS\n\n"
            + "\n".join(f"- {r}" for r in rules) + "\n")


def _block_diagram_context(root: Path, block: str) -> str:
    diagram = _read_json(root / ".coresmith" / "block_diagram.json")
    if not isinstance(diagram, dict):
        return ""
    parts: list[str] = []
    for key in ("blocks", "nodes"):
        for entry in diagram.get(key) or []:
            if isinstance(entry, dict) and entry.get("name") == block:
                parts.append("Block diagram entry:\n"
                             + json.dumps(entry, indent=2))
    conns = [c for c in diagram.get("connections") or []
             if isinstance(c, dict)
             and block in (c.get("from_block", c.get("from")),
                           c.get("to_block", c.get("to")))]
    if conns:
        parts.append("Connections touching this block:\n"
                     + json.dumps(conns, indent=2))
    return "\n\n".join(parts)


def _row(system: str, user: str, assistant: str, labels: dict) -> str:
    return json.dumps({
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "labels": labels,
    }, ensure_ascii=False)


_README = """# SFT DATASET (labeled) -- generated by coresmith

This folder is a supervised-fine-tuning dataset emitted automatically from
this run's VERIFIED artifacts. Each `pairs/*.jsonl` row is a chat-format
sample: `{"messages": [system, user, assistant], "labels": {...}}`.

## The label is the point

`labels.verified` records how strongly the pipeline verified the assistant
artifact:

- `chip` -- the block shipped inside a chip that passed integration_dv AND
  validation_dv with a PASS signoff (strongest label).
- `block_dv` -- the block passed its own cocotb DV (sim green, coverage
  floor met) but the chip-level verdict was not (or not yet) PASS.

Blocks with no verified artifact are excluded (counted in
`manifest.json.skipped_unverified`).

## Files

- `pairs/uarch_to_rtl.jsonl` -- microarchitecture spec (+ pinned
  constraints) -> verified Verilog module.
- `pairs/uarch_to_testbench.jsonl` -- spec + RTL -> cocotb testbench that
  the verification verdict was earned with.
- `pairs/integration_to_chip_top.jsonl` -- block diagram + per-block port
  contracts -> chip-level integration top.
- `pairs/spec_generation.jsonl` -- block-diagram context (+ constraints) ->
  microarchitecture spec.

## Caveats

- USER content is reconstructed from the canonical run artifacts (spec,
  constraints, diagram), not the verbatim generator prompt of the original
  LLM call. It is deterministic and self-contained; if you need verbatim
  agentic trajectories, extract them from `.coresmith/codex_turns.jsonl`.
- Provenance: see `manifest.json` (engine SHA, provider/model, run id).
  Check your LLM provider's terms before training on its outputs.
- Regeneration rounds mean sibling runs can contain near-duplicate pairs;
  deduplicate across runs by `labels.block` + assistant-content hash.
"""


def emit_sft_dataset(project_root: str) -> Optional[dict]:
    """Write ``<project_root>/sft/`` from on-disk run artifacts.

    Returns the manifest dict, or None when the run has no extractable
    verified artifacts at all. Never raises for a malformed single artifact
    -- rows are best-effort per block.
    """
    root = Path(project_root)
    blocks_dir = root / ".coresmith" / "blocks"
    spec_dir = root / "arch" / "uarch_specs"
    chip_ok = _chip_verified(root)

    blocks = sorted(p.name for p in blocks_dir.iterdir() if p.is_dir()) \
        if blocks_dir.is_dir() else []
    run_id = root.name
    engine_sha = (_read_json(root / ".coresmith" / "engine_sha.json")
                  or {}).get("sha", "")
    base_labels = {
        "run": run_id,
        "engine_sha": engine_sha,
        "provider": os.environ.get("CORESMITH_LLM_PROVIDER", ""),
        "model": os.environ.get("CORESMITH_MODEL", ""),
    }

    rows: dict[str, list[str]] = {
        "uarch_to_rtl": [], "uarch_to_testbench": [],
        "integration_to_chip_top": [], "spec_generation": [],
    }
    skipped: list[str] = []

    for block in blocks:
        verified = (blocks_dir / block / "best_result.json").exists()
        if not verified:
            skipped.append(block)
            continue
        tier = "chip" if chip_ok else "block_dv"
        spec = _read(spec_dir / f"{block}.md")
        rtl_path = _find_block_rtl(root, block)
        rtl = _read(rtl_path) if rtl_path else ""
        tb = _read(root / "tb" / "cocotb" / f"test_{block}.py")
        labels = {**base_labels, **_block_labels(root, block),
                  "block": block, "verified": tier}
        constraints = _constraints_text(root, block)
        if spec and rtl:
            rows["uarch_to_rtl"].append(_row(
                _SYSTEM_RTL, spec + constraints, rtl,
                {**labels, "task": "uarch_to_rtl"}))
        if spec and rtl and tb:
            rows["uarch_to_testbench"].append(_row(
                _SYSTEM_TB,
                spec + "\n\n## RTL UNDER TEST\n\n```verilog\n" + rtl
                + "\n```\n",
                tb, {**labels, "task": "uarch_to_testbench"}))
        diagram_ctx = _block_diagram_context(root, block)
        if spec and diagram_ctx:
            rows["spec_generation"].append(_row(
                _SYSTEM_SPEC,
                f"Design a microarchitecture spec for block `{block}`.\n\n"
                + diagram_ctx + constraints,
                spec, {**labels, "task": "spec_generation"}))

    # Chip-top pair: manifest-first (the deduped filelist's head is the
    # assembled top), glob fallback.
    top_path: Optional[Path] = None
    flist = _read(root / ".coresmith" / "chip_top_sources.f").splitlines()
    if flist and Path(flist[0]).exists():
        top_path = Path(flist[0])
    else:
        tops = sorted(root.glob("rtl/integration/*.v"))
        top_path = tops[0] if tops else None
    diagram_raw = _read(root / ".coresmith" / "block_diagram.json")
    if top_path and diagram_raw:
        headers = []
        for block in blocks:
            rp = _find_block_rtl(root, block)
            h = _module_header(_read(rp)) if rp else ""
            if h:
                headers.append(h)
        rows["integration_to_chip_top"].append(_row(
            _SYSTEM_CHIP_TOP,
            "## BLOCK DIAGRAM\n\n```json\n" + diagram_raw + "\n```\n\n"
            "## BLOCK PORT CONTRACTS\n\n```verilog\n"
            + "\n\n".join(headers) + "\n```\n",
            _read(top_path),
            {**base_labels, "block": "__chip_top__",
             "task": "integration_to_chip_top",
             "verified": "chip" if chip_ok else "assembled"}))

    total = sum(len(v) for v in rows.values())
    if total == 0:
        return None

    sft_dir = root / "sft"
    pairs_dir = sft_dir / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)
    for name, lines in rows.items():
        if lines:
            (pairs_dir / f"{name}.jsonl").write_text(
                "\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "dataset": "coresmith-sft",
        "design_run": run_id,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "engine_sha": engine_sha,
        "provider": base_labels["provider"],
        "model": base_labels["model"],
        "chip_verified": chip_ok,
        "counts": {k: len(v) for k, v in rows.items() if v},
        "total_pairs": total,
        "skipped_unverified": skipped,
    }
    (sft_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    (sft_dir / "README.md").write_text(_README, encoding="utf-8")
    return manifest
