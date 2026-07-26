# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""C10: Interface-gap resolver -- answer a block model's INFEASIBLE-INTERFACE-GAP
declaration from the design's COMMITTED documents, and freeze the answer into
the interface contract so it can never be lost again.

Why this exists (exp-reference_codec-20260713, rounds 15-20): the block-model generator
reads ONLY the golden slice + the block's contract slice. Most representation
decisions (numeric enums, region-selector values, address maps, phase
allocations, validity semantics) were made during architecture but recorded in
uarch-spec PROSE or Q&A -- invisible to the generator. Six model-generation
rounds each parked on a DIFFERENT unrecorded fact, and a fresh regeneration
even lost a previously-resolved question (zero-length semantics, resolved at
architecture time, re-asked by a regen). Hand-freezing facts one at a time is
whack-a-mole; this resolver automates it WITH MEMORY: every resolution lands
in ``interface_contracts.json`` (the one document the generator reads), so the
resolved-fact set grows monotonically across regenerations.

Trust boundary: the resolver moves INTERFACE facts from committed prose into
the contract. It must never invent a numeric value the corpus does not ground
-- an invented constant would silently corrupt the oracle. The model's MATH
still comes exclusively from the golden slice (oracle independence preserved).
Every applied amendment is logged to ``.coresmith/gap_resolutions.jsonl``.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from orchestrator._timeouts import scaled

from .coresmith_llm import DEFAULT_MODEL, ClaudeLLM, arch_reasoning_effort

SYSTEM_PROMPT = """\
You are the INTERFACE-GAP RESOLVER for an ASIC design flow.

A per-block model generator (which reads ONLY the software golden slice and
this block's frozen interface contract) declared it cannot realize the block's
datapath because a specific interface fact is missing. Your job: decide
whether that fact ALREADY EXISTS in the design's committed documents (uarch
specs, block diagram, existing contracts) -- explicitly, or by unambiguous
arithmetic derivation (e.g. a region base address that is the sum of the
preceding regions' extents) -- and if so, FREEZE it into the interface
contract as machine-readable amendments.

HARD RULES:
1. NEVER invent a numeric value, enum member, address, or layout the corpus
   does not determine. An invented constant silently corrupts the design's
   oracle. If the corpus is ambiguous or silent, return resolved=false.
2. Amendments must target the SPECIFIC contract edges the fact belongs to,
   identified by their `edge_id` (preferred) or producer/consumer block pair.
3. Use the `representations` schema: `enums` (explicit numeric values for
   every member), `address_maps` (explicit selector_value / base_address /
   extent per region), `record_layouts` (bit-exact msb/lsb fields),
   `state_semantics` (exact validity/lifetime rules incl. reset state).
4. Resolve ONLY what the gap asks for (plus facts inseparable from it).
   Do not restate the whole design.
5. If the gap describes a fact that genuinely requires a NEW design decision
   (no document makes it, no derivation determines it), return resolved=false
   and state the decision crisply in `unresolved_decision` -- that goes to the
   chip-lead as an interrupt.

OUTPUT: exactly one fenced ```json block:
{
  "resolved": true | false,
  "rationale": "<where in the corpus each fact came from, 2-5 sentences>",
  "amendments": [
    {
      "edge_id": "<edge_id from the contracts, preferred>",
      "producer_block": "<alternative selector when edge_id unknown>",
      "consumer_block": "<alternative selector>",
      "add_representations": {
        "enums": [...], "address_maps": [...],
        "record_layouts": [...], "state_semantics": [...]
      },
      "append_note": "<optional one-line note>"
    }
  ],
  "unresolved_decision": "<only when resolved=false: the exact decision needed>"
}
"""


def _extract_json(text: str) -> dict:
    """Parse the first fenced ```json block (or bare JSON object) in ``text``.
    Returns {} on failure -- callers treat that as an unresolved verdict."""
    if not text:
        return {}
    m = re.search(r"```json\s*(.*?)```", text, re.DOTALL)
    candidates = [m.group(1)] if m else []
    # fall back to the largest {...} span
    first, last = text.find("{"), text.rfind("}")
    if first != -1 and last > first:
        candidates.append(text[first:last + 1])
    for cand in candidates:
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return {}


def build_gap_corpus(
    project_root: str | Path,
    block_name: str,
    gap_text: str,
    max_chars_per_doc: int = 40000,
    max_total_chars: int = 200000,
) -> dict[str, str]:
    """Assemble the committed documents that can ground a resolution:
    this block's uarch spec, the specs of its edge partners and of any block
    the gap text names, the block's contract slice (with edge_ids), and the
    relevant block-diagram entries (where architecture Q&A resolutions live).
    """
    root = Path(project_root)
    sections: dict[str, str] = {}
    total = 0

    def _add(key: str, text: str) -> None:
        nonlocal total
        text = (text or "").strip()
        if not text or total >= max_total_chars:
            return
        text = text[:max_chars_per_doc]
        sections[key] = text
        total += len(text)

    # Which blocks matter, in PRIORITY order: this block, then blocks the gap
    # text NAMES (these hold the answer more often than not), then generic
    # edge partners. Gap text abbreviates ("coeff_mem" for
    # coefficient_token_memory), so match on name segments, not full names.
    def _gap_names(bname: str) -> bool:
        if bname in gap_text:
            return True
        segs = [s for s in bname.split("_") if len(s) >= 3]
        hits = 0
        for s in segs:
            # a >=3-char prefix of the segment appearing in the gap counts
            # (coefficient -> "coeff", memory -> "mem")
            for ln in range(len(s), 2, -1):
                if s[:ln] in gap_text:
                    hits += 1
                    break
        return hits >= 2

    contracts_path = root / ".coresmith" / "interface_contracts.json"
    contract_doc: dict = {}
    try:
        contract_doc = json.loads(contracts_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        pass
    all_blocks: set[str] = set()
    partners: list[str] = []
    my_edges: list[dict] = []
    for c in contract_doc.get("contracts") or []:
        p = str(c.get("producer_block", "")).strip()
        q = str(c.get("consumer_block", "")).strip()
        all_blocks.update(x for x in (p, q) if x)
        if block_name in (p, q):
            my_edges.append(c)
            for x in (p, q):
                if x and x != block_name and x not in partners:
                    partners.append(x)
    named = [b for b in sorted(all_blocks)
             if b != block_name and _gap_names(b)]
    relevant: list[str] = [block_name]
    for b in named + partners:
        if b not in relevant:
            relevant.append(b)

    # 1) the block's own contract slice, verbatim (carries the edge_ids the
    #    amendments must target).
    _add("interface_contracts (this block's edges, amend THESE)",
         json.dumps({"defaults": {k: contract_doc.get(k) for k in (
             "default_packing_convention", "default_endianness_rationale")
             if contract_doc.get(k)}, "edges": my_edges}, indent=2))

    # 2) uarch specs: own first, then partners/named blocks.
    for b in relevant:
        spec = root / "arch" / "uarch_specs" / f"{b}.md"
        if spec.exists():
            try:
                _add(f"uarch_spec:{b}", spec.read_text(encoding="utf-8",
                                                       errors="replace"))
            except OSError:
                pass

    # 3) block-diagram entries for the relevant blocks (semantic contracts /
    #    Q&A resolutions often live here).
    bd_path = root / ".coresmith" / "block_diagram.json"
    try:
        bd = json.loads(bd_path.read_text(encoding="utf-8"))
        ents = [b for b in bd.get("blocks", [])
                if b.get("name") in relevant]
        if ents:
            _add("block_diagram (relevant blocks)", json.dumps(ents, indent=2))
    except (OSError, json.JSONDecodeError):
        pass

    return sections


def apply_contract_amendments(
    contract_doc: dict, amendments: list
) -> tuple[dict, list[str]]:
    """Merge resolver amendments into an ``interface_contracts.json`` document
    (pure function -- no I/O). Returns ``(updated_doc, applied_descriptions)``;
    an amendment matching no edge is skipped (never guessed).

    Merge semantics: within each ``representations`` list, an entry REPLACES an
    existing same-``name`` entry (re-resolution updates rather than
    duplicates); notes are appended once.
    """
    applied: list[str] = []
    contracts = contract_doc.get("contracts") or []

    def _matches(c: dict, amend: dict) -> bool:
        eid = (amend.get("edge_id") or "").strip()
        if eid:
            return str(c.get("edge_id", "")).strip() == eid
        p = (amend.get("producer_block") or "").strip()
        q = (amend.get("consumer_block") or "").strip()
        if not (p or q):
            return False
        ok = True
        if p:
            ok = ok and str(c.get("producer_block", "")).strip() == p
        if q:
            ok = ok and str(c.get("consumer_block", "")).strip() == q
        return ok

    for amend in amendments or []:
        if not isinstance(amend, dict):
            continue
        targets = [c for c in contracts if _matches(c, amend)]
        if not targets:
            continue
        reps = amend.get("add_representations") or {}
        note = (amend.get("append_note") or "").strip()
        for c in targets:
            added_bits: list[str] = []
            if reps:
                dst = c.setdefault("representations", {})
                for kind in ("enums", "address_maps", "record_layouts",
                             "state_semantics"):
                    entries = reps.get(kind) or []
                    if not isinstance(entries, list):
                        continue
                    lst = dst.setdefault(kind, [])
                    for entry in entries:
                        if not isinstance(entry, dict):
                            continue
                        nm = str(entry.get("name", "")).strip()
                        # replace-by-name (update, don't duplicate)
                        lst[:] = [e for e in lst
                                  if str(e.get("name", "")).strip() != nm
                                  or not nm]
                        lst.append(entry)
                        added_bits.append(f"{kind}[{nm or '?'}]")
            if note:
                cur = str(c.get("notes", "") or "")
                if note not in cur:
                    c["notes"] = (cur + "\n" if cur else "") + \
                        f"[gap-resolution] {note}"
                    added_bits.append("note")
            if added_bits:
                applied.append(
                    f"{c.get('edge_id', '<no-edge-id>')}: "
                    + ", ".join(added_bits))
    return contract_doc, applied


class GapResolver:
    """LLM wrapper for the resolution verdict (architecture-stage tier)."""

    def __init__(self, model: str | None = None):
        self.llm = ClaudeLLM(
            model=model or DEFAULT_MODEL,
            timeout=scaled(1200, env="CORESMITH_GAP_RESOLVER_TIMEOUT"),
            reasoning_effort=arch_reasoning_effort(),
        )

    async def resolve(
        self, block_name: str, gap_text: str, corpus: dict[str, str]
    ) -> dict[str, Any]:
        parts = [
            f"BLOCK: {block_name}",
            f"\nGAP DECLARATION (verbatim from the generated model):\n{gap_text}",
        ]
        for key, text in corpus.items():
            parts.append(f"\n--- {key} ---\n{text}")
        content = await self.llm.call(
            system=SYSTEM_PROMPT,
            prompt="\n".join(parts),
            run_name=f"Resolve Interface Gap [{block_name}]",
        )
        verdict = _extract_json(content)
        if not verdict:
            return {"resolved": False, "rationale": "",
                    "amendments": [],
                    "unresolved_decision":
                        "resolver returned no parseable verdict"}
        verdict.setdefault("resolved", False)
        verdict.setdefault("amendments", [])
        verdict.setdefault("rationale", "")
        verdict.setdefault("unresolved_decision", "")
        return verdict
