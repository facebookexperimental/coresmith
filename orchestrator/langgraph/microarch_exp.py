# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Microarchitecture (pass-1) experiment graph -- STANDALONE LangGraph pipeline.

This is a self-contained redesign of the CoreSmith microarchitecture / model-
building stage as a real feedback loop that MIRRORS the RTL/frontend stage,
minus RTL-write and synth. It is NOT wired into the daemon or pipeline_graph;
run it directly via ``python -m orchestrator.langgraph.microarch_exp <root>``.

Design (REAL bounded per-block codex fan-out for BOTH build and verify; build
threads a PER-BLOCK codex session for resume across retries):

    START
      -> build_models   (AGENT 1, codex)   : REAL fan-out -- one codex call PER
                                              block, run concurrently under a
                                              bounded asyncio.Semaphore(N)
                                              (N=CORESMITH_MICROARCH_FANOUT,
                                              default 4; BOX SAFETY, never
                                              unbounded). Each call builds ONE
                                              arch/block_models/<block>.py from
                                              its spec + golden slice and runs its
                                              own smoke check. PER-BLOCK RESUME:
                                              each block threads its own prior
                                              codex session id on a retry so the
                                              rebuild continues that block's
                                              conversation + diagnose feedback.
      -> lint_models    (DETERMINISTIC)    : import + elaborate every
                                              Elaboratable (Amaranth Verilog
                                              conversion) to catch
                                              import/syntax/missing-include/
                                              missing-macro errors. (The check the
                                              current flow lacks.)
      -> verify_models  (AGENT 2, codex)   : REAL fan-out -- one focused DV codex
                                              call PER block, concurrent under the
                                              same bounded semaphore, each
                                              verifying THAT block's model
                                              vs its golden slice byte-exact
                                              (MODEL-vs-GOLDEN) with all anti-cheat
                                              rules scoped to one block.
      -> size           (DETERMINISTIC)    : real per-block datapath DFG
                                              (pipeline_scheduler) + register +
                                              memory area (mem_characterize) vs
                                              the block area budget.
      -> ppa_judge      (AGENT, codex)     : IMPARTIAL verdict -- does the
                                              microarch meet ALL FRD reqs + area
                                              budget + Fmax + synthesizability?
                                              pass -> END; fail -> diagnose;
                                              escalate -> ask_human.
      -> (any gate failed) diagnose (AGENT, codex): seeds structured MUST /
                                              MUST-NOT rules into each block's
                                              .coresmith/microarch/<b>/constraints.json
                                              (disk-first memory) and routes:
                                              rebuild -> build_models,
                                              ask_human -> ask_human (interrupt).

    Bounded retries (default max 4).

The three agents all use ``ClaudeLLM`` (honours CORESMITH_LLM_PROVIDER=codex) with
distinct run_names ("Build Models", "Verify Models", "Diagnose Models") so their
codex turns land in ``.coresmith/codex_turns.jsonl`` for the webview.

State (a plain TypedDict / dict):
    project_root, blocks, attempt, feedback, lint_errors, verify_results,
    size_results, status
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

logger = logging.getLogger(__name__)

# Directory layout (mirrors orchestrator.architecture.state / composition).
ARCH_DIR = "arch"
UARCH_SPECS_DIRNAME = "uarch_specs"
BLOCK_MODELS_DIRNAME = "block_models"

DEFAULT_TARGET_CLOCK_MHZ = 50.0
DEFAULT_MAX_ATTEMPTS = 4
# The runner (run_microarch_exp) gets more room to converge now that a rebuild
# resumes the builder's prior codex conversation (continuity across retries).
RUN_MICROARCH_MAX_ATTEMPTS = 6

MICROARCH_DIRNAME = "microarch"       # .coresmith/microarch/<block>/constraints.json


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class MicroarchState(TypedDict, total=False):
    project_root: str
    blocks: list[str]
    attempt: int
    feedback: str
    lint_errors: dict[str, str]          # block -> error text (empty == all clean)
    verify_results: dict[str, dict]      # block -> {passed, first_divergence, detail}
    size_results: dict[str, dict]        # block -> {feasible, depth, fmax_mhz, detail}
    mem_results: dict[str, dict]         # block -> {mem_area_um2, memories:[...], detail}
    ppa_verdict: dict                    # {passed, verdict, violations, first_divergence, recommended_action}
    debug_action: str                    # rebuild | ask_human (routed out of diagnose)
    build_session_id: str                # codex session id of the build_models call (for resume, legacy/grouped)
    build_session_ids: dict[str, str]    # per-block codex session id (for PER-BLOCK resume across retries)
    build_cluster_session_ids: dict[str, str]  # per-CLUSTER codex session id (cluster_id -> sid), for PER-CLUSTER resume
    failed_blocks: list[str]             # blocks the last diagnose flagged (incremental rebuild)
    passing_models: dict[str, float]     # block -> model mtime when it last passed ALL gates
    max_attempts: int                    # retry budget; MUST be declared here or LangGraph drops the
                                         # init key and every run silently caps at DEFAULT_MAX_ATTEMPTS
    status: str                          # running | passed | failed | error


# ---------------------------------------------------------------------------
# Small utilities (deterministic; no LLM, no MCP)
# ---------------------------------------------------------------------------

def _pr(state: dict) -> str:
    return state.get("project_root", ".")


def _env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean env flag (``1/true/yes/on`` == True). Missing -> default."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _prompt_slim_enabled() -> bool:
    """CORESMITH_PROMPT_SLIM (default ON): the build-models prompt points at the
    golden path + a HEAD instead of inlining the whole (up to 60 KB) source, and
    tells the builder to `coresmith verify model <block>` before finishing. The
    per-block model gate re-checks the model, so the full dump is redundant. Set
    to 0 to restore the full inlined golden (pre-B4 behavior)."""
    return os.environ.get("CORESMITH_PROMPT_SLIM", "1").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _microarch_fanout() -> int:
    """CORESMITH_MICROARCH_FANOUT (default 4): the MAX number of per-block codex
    calls in flight AT ONCE during the build/verify fan-out.

    BOX SAFETY IS CRITICAL. A prior unbounded fork-storm hung the machine: each
    ``codex exec`` spawns ~7 processes, so an unbounded fan-out over N blocks is
    ~7N concurrent procs. Every fan-out in this module is gated by an
    ``asyncio.Semaphore(_microarch_fanout())`` -- NEVER unbounded. The value is
    clamped to >= 1 so a bad env value can never disable the bound.
    """
    try:
        n = int(os.environ.get("CORESMITH_MICROARCH_FANOUT", "4"))
    except ValueError:
        n = 4
    return max(1, n)


def _microarch_incremental_enabled() -> bool:
    """CORESMITH_MICROARCH_INCREMENTAL (default ON): on a rebuild, regenerate
    ONLY the blocks that failed a gate, and skip re-lint/re-verify of blocks
    whose model file is unchanged since it last passed all gates. Set to 0 to
    rebuild + re-verify ALL blocks every attempt (pre-B4 behavior)."""
    return os.environ.get("CORESMITH_MICROARCH_INCREMENTAL", "1").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _model_mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime if path.exists() else None
    except OSError:
        return None


def _unchanged_since_pass(path: Path, prev_mtime: float | None) -> bool:
    """True when the model file is unchanged since it last passed a gate.

    ``prev_mtime`` is the model mtime recorded the last time the block passed
    (None -> never passed -> not skippable). A tiny epsilon absorbs FS mtime
    granularity so an equal-mtime file counts as unchanged.
    """
    if prev_mtime is None:
        return False
    mt = _model_mtime(path)
    return mt is not None and mt <= float(prev_mtime) + 1e-6


def _arch_dir(project_root: str) -> Path:
    return Path(project_root) / ARCH_DIR


def _block_models_dir(project_root: str) -> Path:
    return _arch_dir(project_root) / BLOCK_MODELS_DIRNAME


def _uarch_specs_dir(project_root: str) -> Path:
    return _arch_dir(project_root) / UARCH_SPECS_DIRNAME


# ---------------------------------------------------------------------------
# Disk-first per-block constraint loop (mirrors the frontend
# ``.coresmith/blocks/<b>/constraints.json`` accumulation, under microarch/).
# ---------------------------------------------------------------------------

def _microarch_dir(project_root: str) -> Path:
    return Path(project_root) / ".coresmith" / MICROARCH_DIRNAME


def _block_constraints_path(project_root: str, block: str) -> Path:
    return _microarch_dir(project_root) / block / "constraints.json"


def _read_block_constraints(project_root: str, block: str) -> list[dict]:
    """Return the accumulated constraint rules for a block (never raises)."""
    p = _block_constraints_path(project_root, block)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (ValueError, OSError):
        return []


def _reset_block_constraints(project_root: str, block: str) -> None:
    """Reset a block's constraints.json to ``[]`` (called once at run start)."""
    p = _block_constraints_path(project_root, block)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        p.write_text("[]", encoding="utf-8")
    except OSError:
        pass


def _normalize_constraint(text: str) -> str:
    """Lowercase / strip punctuation / collapse whitespace for dedup."""
    t = (text or "").lower().strip()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^\w\s]", "", t)
    return t


def _append_block_constraints(
    project_root: str, block: str, rules: list[dict], attempt: int,
) -> int:
    """Append structured MUST/MUST-NOT rules to a block's constraints.json.

    Each rule is ``{"rule": str, "kind": "must"|"must_not"|str, "source": str}``.
    Deduplicates against existing rules by normalized text. Returns the count
    added. This is the disk-first memory that survives across ALL retries.
    """
    if not rules:
        return 0
    existing = _read_block_constraints(project_root, block)
    seen = {_normalize_constraint(r.get("rule", "")) for r in existing}
    added = 0
    for r in rules:
        rule_text = (r.get("rule") or "").strip()
        if not rule_text:
            continue
        norm = _normalize_constraint(rule_text)
        if norm in seen:
            continue
        seen.add(norm)
        existing.append({
            "rule": rule_text,
            "kind": r.get("kind", "must"),
            "source": r.get("source", "diagnose"),
            "attempt": attempt,
        })
        added += 1
    if added:
        p = _block_constraints_path(project_root, block)
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            p.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        except OSError:
            pass
    return added


def _log_path(project_root: str) -> Path:
    d = Path(project_root) / ".coresmith"
    d.mkdir(parents=True, exist_ok=True)
    return d / "microarch_exp.log"


def _log(project_root: str, msg: str) -> None:
    """Log a progress line to stdout + ``.coresmith/microarch_exp.log``."""
    line = f"[microarch_exp] {msg}"
    print(line, flush=True)
    try:
        with _log_path(project_root).open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass


def _read_target_clock_mhz(project_root: str) -> float:
    """Best-effort target clock from ERS / uArch specs; default 50 MHz.

    Scans the ERS doc and any uArch spec markdown for a ``NN MHz`` mention and
    takes the first plausible hit. Never raises; falls back to the default.
    """
    candidates: list[Path] = []
    arch = _arch_dir(project_root)
    for name in ("ers_spec.md", "ers.md"):
        candidates.append(arch / name)
    specs = _uarch_specs_dir(project_root)
    if specs.is_dir():
        candidates.extend(sorted(specs.glob("*.md")))
    pat = re.compile(r"(\d+(?:\.\d+)?)\s*MHz", re.IGNORECASE)
    for p in candidates:
        try:
            text = p.read_text(encoding="utf-8")
        except OSError:
            continue
        m = pat.search(text)
        if m:
            try:
                val = float(m.group(1))
                if 1.0 <= val <= 5000.0:
                    return val
            except ValueError:
                pass
    return DEFAULT_TARGET_CLOCK_MHZ


def discover_blocks(project_root: str) -> list[str]:
    """Read the block list from arch/block_diagram.json | .md, else uArch specs.

    Returns block names in declaration order. Never raises: an empty list means
    the graph will report ``status=error`` from build_models.
    """
    arch = _arch_dir(project_root)

    # 1. block_diagram.json -- authoritative.
    bd_json = arch / "block_diagram.json"
    if bd_json.exists():
        try:
            doc = json.loads(bd_json.read_text(encoding="utf-8"))
            blocks = doc.get("blocks") if isinstance(doc, dict) else None
            names = []
            for b in (blocks or []):
                if isinstance(b, dict) and isinstance(b.get("name"), str):
                    names.append(b["name"])
                elif isinstance(b, str):
                    names.append(b)
            if names:
                return names
        except (ValueError, OSError):
            pass

    # 2. block_diagram.md -- parse block headings / a bullet list of names.
    bd_md = arch / "block_diagram.md"
    if bd_md.exists():
        try:
            text = bd_md.read_text(encoding="utf-8")
            names = re.findall(r"^#{1,4}\s+`?([a-zA-Z_][\w]*)`?", text, re.MULTILINE)
            if names:
                # de-dup preserving order
                seen: set[str] = set()
                out = []
                for n in names:
                    if n not in seen:
                        seen.add(n)
                        out.append(n)
                return out
        except OSError:
            pass

    # 3. Fallback: one block per uArch spec file.
    specs = _uarch_specs_dir(project_root)
    if specs.is_dir():
        return [p.stem for p in sorted(specs.glob("*.md"))]

    return []


def _read_uarch_specs(project_root: str, blocks: list[str]) -> dict[str, str]:
    """Return {block: uArch spec markdown} for every block that has one."""
    specs_dir = _uarch_specs_dir(project_root)
    out: dict[str, str] = {}
    for b in blocks:
        p = specs_dir / f"{b}.md"
        if p.exists():
            try:
                out[b] = p.read_text(encoding="utf-8")
            except OSError:
                out[b] = ""
    return out


def _read_block_diagram(project_root: str) -> dict:
    """Load arch/block_diagram.json as a dict (empty on any failure)."""
    p = _arch_dir(project_root) / "block_diagram.json"
    if not p.exists():
        return {}
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
        return doc if isinstance(doc, dict) else {}
    except (ValueError, OSError):
        return {}


def _expected_ports_for_block(block_diagram: dict, block: str) -> list[str]:
    """Return the expected port/interface names for a block from the diagram.

    Reads the block's ``interfaces`` map (each key is an interface = a logical
    port group) plus any connection endpoints that name it. Never raises.
    """
    for b in (block_diagram.get("blocks") or []):
        if not isinstance(b, dict) or b.get("name") != block:
            continue
        ifaces = b.get("interfaces")
        if isinstance(ifaces, dict):
            return [str(k) for k in ifaces.keys()]
    return []


# ---------------------------------------------------------------------------
# COUPLING CLUSTERS -- the fan-out UNIT (a coupling cluster, not always a
# singleton). exp13 fixed oscillation + closed the framer entropy wall by
# fanning out ONE codex agent PER BLOCK (per-block resume). But that REGRESSED
# the 5 tightly-coupled intra_rd sub-blocks (mode_decision, transform_quant,
# reconstruct, chroma_encode, syntax_pack): they share a tight dataflow contract
# (the reconstructed-neighbour feedback loop + the intra-pipeline byte contract),
# so building them in ISOLATION loses the cross-block context that let them agree
# under the earlier GROUPED build. Fix: coupled blocks build TOGETHER in one
# shared codex context; independent blocks fan out solo. All with resume.
#
# CLUSTERING SIGNAL (priority order, most-robust-available wins):
#   (a) FEEDBACK CYCLE / non-byte-aligned shared-state edge between two blocks
#       -> TIGHT. A bidirectional (A->B and B->A) pair of edges whose interface
#       is a direct/handshake-free shared-state port (the intra recon-neighbour
#       loop: writeback + neighbor-read, "no handshake -- tight per-block-cycle
#       dependency") means the two blocks share cycle-accurate internal state and
#       MUST agree on it -- build them together.
#   (b) SHARED DECOMPOSITION PARENT whose mutual boundary is NOT byte/handshake-
#       aligned -> TIGHT. Decomposition is not stamped in block_diagram.json (the
#       only structural field is `subsystem`), so the robust available proxy for
#       "shared decomposition parent" is a shared NAME PREFIX: `intra_rd_*` sub-
#       blocks share parent `intra_rd`; `annexb_*` share parent `framer`. Two
#       same-parent blocks are coupled ONLY when a direct edge between them is
#       NOT byte-aligned (a tight internal boundary). bytestream_entropy<->
#       bytestream_framing share parent `framer` BUT their boundary is a clean byte-
#       aligned AXI-Stream/FIFO handoff -> they stay SEPARATE singletons.
#   (c) FALLBACK: no coupling signal -> each block is its own singleton (== the
#       exp13 per-block behavior).
#
# A cluster is capped at CLUSTER_SIZE_CAP so a cluster can't reintroduce the
# context-overload that a whole-chip single build caused. Overridable via
# CORESMITH_MICROARCH_CLUSTERING (auto default | off -> all singletons).
# ---------------------------------------------------------------------------

CLUSTER_SIZE_CAP = 6

# An edge whose interface/semantic-contract shows one of these tokens is a CLEAN,
# BYTE-ALIGNED, HANDSHAKED boundary (AXI-Stream tdata/tvalid/tready/tlast, a FIFO
# hop). Two blocks that only talk over such a boundary are DECOUPLED: each can be
# built in isolation because the contract between them is an explicit, latched,
# self-describing byte handshake -- exactly the bytestream_entropy<->bytestream_framing
# case. These are NOT a coupling signal.
_BYTE_ALIGNED_TOKENS = (
    "_axis", "axi-stream", "axi stream", "axistream",
    "tdata", "tvalid", "tready", "tlast", "tuser",
    "fifo", "handshake", "back-pressure", "backpressure",
    "ready/valid", "valid/ready", "srdy", "drdy",
)

# An edge whose interface/semantic-contract shows one of these tokens is a TIGHT,
# non-byte-aligned shared-state boundary: a direct combinational/registered port
# with no handshake, a per-cycle dependency, a feedback/writeback/neighbour path.
# Two blocks joined by such an edge share cycle-accurate internal state.
_TIGHT_EDGE_TOKENS = (
    "no handshake", "no back", "single-cycle", "single cycle",
    "per-block-cycle", "per-cycle", "per cycle", "direct port",
    "combinational", "tight", "shared state", "shared-state",
    "feedback", "writeback", "write-back", "neighbor", "neighbour",
    "recon", "loop",
)


def _clustering_mode() -> str:
    """CORESMITH_MICROARCH_CLUSTERING (default ``auto``): the fan-out unit.

    ``auto`` -> derive coupling clusters (coupled blocks build together, others
    fan out solo). ``off`` -> every block is its own singleton cluster (the
    exp13 per-block behavior; the current tests/behavior are preserved).
    """
    raw = (os.environ.get("CORESMITH_MICROARCH_CLUSTERING") or "auto").strip().lower()
    return "off" if raw == "off" else "auto"


def _edge_endpoints(conn: dict) -> tuple[str | None, str | None]:
    """Return (from_block, to_block) from a connection record (schema-tolerant)."""
    if not isinstance(conn, dict):
        return None, None
    src = conn.get("from") or conn.get("source") or conn.get("src")
    dst = conn.get("to") or conn.get("target") or conn.get("dst")
    return (str(src) if src else None, str(dst) if dst else None)


def _edge_text(conn: dict) -> str:
    """Lowercased blob of an edge's interface + semantic-contract + bus name.

    This is the signal we classify: whether the boundary is a clean byte-aligned
    handshake (decoupled) or a tight non-byte-aligned shared-state port (coupled).
    """
    if not isinstance(conn, dict):
        return ""
    parts = [
        str(conn.get("interface") or ""),
        str(conn.get("semantic_contract") or ""),
        str(conn.get("bus_name") or ""),
        str(conn.get("interface_name") or ""),
    ]
    return " ".join(parts).lower()


def _edge_is_byte_aligned(conn: dict) -> bool:
    """True when an edge's contract shows a clean byte-aligned handshake boundary.

    A byte-aligned edge (AXI-Stream / FIFO / ready-valid) is DECOUPLED even if it
    is a feedback edge, UNLESS the contract ALSO explicitly says the boundary is
    tight/handshake-free (then the tight signal wins). This is what keeps
    bytestream_entropy<->bytestream_framing (clean AXI/FIFO handoff) as separate
    singletons while the intra recon loop (direct no-handshake port) is coupled.
    """
    text = _edge_text(conn)
    if not text:
        return False
    tight = any(tok in text for tok in _TIGHT_EDGE_TOKENS)
    aligned = any(tok in text for tok in _BYTE_ALIGNED_TOKENS)
    # An explicit "no handshake" / "direct port" phrasing overrides a stray
    # mention of a byte width -- the boundary is tight, not clean.
    if "no handshake" in text or "no back" in text or "direct port" in text:
        return False
    return aligned and not tight


def _edge_is_tight(conn: dict) -> bool:
    """True when an edge is a tight, non-byte-aligned shared-state boundary."""
    text = _edge_text(conn)
    if not text:
        # No metadata at all -> DEFENSIVE: not tight (missing metadata -> singleton).
        return False
    if _edge_is_byte_aligned(conn):
        return False
    return any(tok in text for tok in _TIGHT_EDGE_TOKENS)


def _shared_parent_prefix(a: str, b: str) -> str | None:
    """Return the shared decomposition-parent prefix of two block names, or None.

    Decomposition is not stamped in block_diagram.json, so a shared NAME PREFIX
    (split on '_') is the robust proxy: ``intra_rd_mode_decision`` +
    ``intra_rd_transform_quant`` -> ``intra_rd``; ``bytestream_entropy`` +
    ``bytestream_framing`` -> ``framer``. Requires at least ONE shared leading
    underscore-delimited segment. Returns the longest shared prefix (>= 1 seg).
    """
    pa, pb = str(a).split("_"), str(b).split("_")
    shared: list[str] = []
    for sa, sb in zip(pa, pb):
        if sa == sb and sa:
            shared.append(sa)
        else:
            break
    if not shared:
        return None
    # Two shared segments (e.g. `intra_rd`) is an unambiguous shared parent.
    if len(shared) >= 2:
        return "_".join(shared)
    # A single shared segment counts as a shared decomposition parent when it is
    # specific enough (a real parent name like `framer`, not a 1-2 char stub) OR
    # when the whole of one name is the parent of the other (`intra` + `intra_x`).
    # This keeps the `bytestream_entropy`/`bytestream_framing` pair recognised as sharing
    # parent `framer` (they then stay SEPARATE only because their boundary is
    # byte-aligned, decided in derive_coupling_clusters -- not here).
    only = shared[0]
    if only == a or only == b or len(only) >= 4:
        return only
    return None


def _union_find_partition(
    blocks: list[str], coupled_pairs: set[tuple[str, str]],
) -> list[list[str]]:
    """Partition ``blocks`` into clusters by the transitive closure of coupling.

    ``coupled_pairs`` is a set of unordered (a, b) block pairs that are TIGHT.
    Uses union-find; returns clusters as lists in first-appearance order, each
    cluster's members kept in ``blocks`` order.
    """
    parent: dict[str, str] = {b: b for b in blocks}

    def _find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(x: str, y: str) -> None:
        rx, ry = _find(x), _find(y)
        if rx != ry:
            parent[ry] = rx

    bset = set(blocks)
    for a, b in coupled_pairs:
        if a in bset and b in bset:
            _union(a, b)

    groups: dict[str, list[str]] = {}
    order: list[str] = []
    for b in blocks:
        root = _find(b)
        if root not in groups:
            groups[root] = []
            order.append(root)
        groups[root].append(b)
    return [groups[r] for r in order]


def _split_oversized_clusters(
    clusters: list[list[str]], cap: int,
) -> list[list[str]]:
    """Cap each cluster at ``cap`` members (chunk any oversized cluster in order).

    A cluster larger than the cap would reintroduce the context-overload the
    whole-chip single build caused, so it is chunked into cap-sized pieces (order
    preserved). This is a coarse but safe bound.
    """
    cap = max(1, int(cap))
    out: list[list[str]] = []
    for cl in clusters:
        if len(cl) <= cap:
            out.append(cl)
        else:
            for i in range(0, len(cl), cap):
                out.append(cl[i:i + cap])
    return out


def derive_coupling_clusters(
    project_root: str, blocks: list[str],
) -> list[list[str]]:
    """Partition ``blocks`` into COUPLING CLUSTERS -- the build fan-out unit.

    Tightly-coupled blocks land in the SAME cluster (built together in one shared
    codex context); independent blocks are singletons (fanned out solo). See the
    module-level design note for the signal priority. Behavior:

    - ``CORESMITH_MICROARCH_CLUSTERING=off`` -> every block a singleton (== exp13
      per-block behavior; preserves the current tests).
    - ``auto`` (default): read ``arch/block_diagram.json``, build the block
      dataflow graph, and mark a block PAIR as TIGHT when EITHER
        (a) they form a FEEDBACK CYCLE (edges A->B and B->A) AND at least one of
            those edges is a non-byte-aligned tight shared-state port, OR
        (b) they share a DECOMPOSITION PARENT (shared name prefix) AND a direct
            edge between them is NOT byte-aligned (a tight internal boundary).
      The transitive closure of TIGHT pairs forms the clusters; every other block
      is a singleton. Clusters are capped at ``CLUSTER_SIZE_CAP``.

    DEFENSIVE: a missing/empty block_diagram, missing edge metadata, or any parse
    failure yields all-singletons (never raises).
    """
    blocks = [b for b in (blocks or []) if b]
    if not blocks:
        return []
    if _clustering_mode() == "off":
        return [[b] for b in blocks]

    diagram = _read_block_diagram(project_root)
    conns = diagram.get("connections")
    if not isinstance(conns, list):
        conns = diagram.get("edges") if isinstance(diagram.get("edges"), list) else []
    bset = set(blocks)

    # Index directed edges between IN-SCOPE blocks, keyed by (from, to) -> the
    # list of connection records (there can be multiple ports between a pair).
    directed: dict[tuple[str, str], list[dict]] = {}
    for conn in conns:
        src, dst = _edge_endpoints(conn)
        if src in bset and dst in bset and src != dst:
            directed.setdefault((src, dst), []).append(conn)

    coupled: set[tuple[str, str]] = set()

    def _mark(a: str, b: str) -> None:
        coupled.add((a, b) if a <= b else (b, a))

    # (a) FEEDBACK CYCLE with a tight edge: A->B and B->A, at least one tight.
    for (src, dst), recs in directed.items():
        if src >= dst:
            continue  # consider each unordered pair once
        back = directed.get((dst, src))
        if not back:
            continue
        both = recs + back
        # A feedback cycle is a coupling signal on its own UNLESS every edge in
        # the cycle is a clean byte-aligned handshake (a FIFO'd loop is decoupled).
        if any(_edge_is_tight(c) for c in both) or not all(
            _edge_is_byte_aligned(c) for c in both
        ):
            _mark(src, dst)

    # (b) SHARED DECOMPOSITION PARENT + a tight (non-byte-aligned) direct edge.
    for (src, dst), recs in directed.items():
        if _shared_parent_prefix(src, dst) is None:
            continue
        if any(_edge_is_tight(c) for c in recs):
            _mark(src, dst)
        elif not any(_edge_is_byte_aligned(c) for c in recs):
            # same-parent + a direct edge with NO byte-aligned handshake marker
            # (and no explicit tight marker) -> treat as a tight internal boundary
            # (decomposition split rarely inserts a full AXI/FIFO between halves).
            _mark(src, dst)

    clusters = _union_find_partition(blocks, coupled)
    clusters = _split_oversized_clusters(clusters, CLUSTER_SIZE_CAP)
    return clusters


def _cluster_id(cluster: list[str]) -> str:
    """Stable cluster id for the per-cluster resume session store.

    Keyed by the sorted-block-join so the same set of blocks resumes its own
    codex session across attempts regardless of discovery order.
    """
    return "+".join(sorted(cluster))


# port-name aliases: a factory signature routinely splits one logical interface
# into its constituent handshake/data signals (e.g. ``s_axis_in`` ->
# ``s_axis_in_tdata`` / ``..._tvalid`` / ``..._tready``). The interface check is
# satisfied when SOME factory param carries the interface name as a prefix.
_CLOCK_RESET_NAMES = {"clk", "clock", "rst", "reset", "rst_n", "resetn"}


def check_interface_constraint(
    factory_params: list[str], expected_ports: list[str],
) -> list[str]:
    """Deterministic anti-cheat: which expected interfaces has the model dropped?

    Compares a model's ``@block`` factory signature (its port/param names) to the
    expected interfaces from ``arch/block_diagram.json``. An expected interface
    is considered PRESENT when some factory param equals it or carries it as a
    prefix (``s_axis_in`` matched by ``s_axis_in_tdata``). Returns the list of
    MISSING expected interface names (empty == interface honoured). Clock/reset
    interfaces are ignored (the factory always has clk/rst). A non-empty result
    is a lint-level failure that seeds a constraint.
    """
    params = [str(p).lower() for p in (factory_params or [])]
    missing: list[str] = []
    for port in (expected_ports or []):
        pl = str(port).lower()
        if pl in _CLOCK_RESET_NAMES:
            continue
        # present if an exact match OR a param starts with "<port>_" / contains it
        hit = any(
            p == pl or p.startswith(pl + "_") or p.startswith(pl)
            or pl.startswith(p) and len(p) >= 3
            for p in params
        )
        if not hit:
            missing.append(str(port))
    return missing


def _factory_params(model_path: str, block_name: str) -> list[str] | None:
    """Best-effort import of a model and return its @block factory param names.

    Returns None when the file can't be imported / no factory found (the lint
    elaboration check owns those failures; the interface check just skips).
    """
    p = Path(model_path)
    if not p.exists():
        return None
    import importlib.util
    import inspect
    mod_name = f"_coresmith_uarch_iface_{p.stem}"
    try:
        spec = importlib.util.spec_from_file_location(mod_name, str(p))
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception:  # noqa: BLE001 - import errors are lint's job
        return None
    factory = _find_block_factory(module, block_name)
    if factory is None:
        return None
    try:
        return list(inspect.signature(factory).parameters.keys())
    except (TypeError, ValueError):
        return None


def _resolve_golden(project_root: str) -> tuple[str | None, str]:
    """Locate the reference software golden. Returns (path, source_text)."""
    path = None
    # Prefer inputs/golden.py -- the user names the golden explicitly, and the
    # generic reference resolver can grab a sibling like inputs/fidelity_metric.py
    # (glob/alpha order), which would make the consistency gate hash the wrong
    # file. The explicit golden wins; only fall back to the resolver if absent.
    cand = Path(project_root) / "inputs" / "golden.py"
    if cand.exists():
        path = str(cand.resolve())
    if not path:
        try:
            from orchestrator.architecture.composition import (
                resolve_reference_implementation,
            )
            path = resolve_reference_implementation(project_root)
        except Exception:  # noqa: BLE001 - stay standalone / import-robust
            path = None
    src = ""
    if path and Path(path).exists():
        try:
            src = Path(path).read_text(encoding="utf-8")
        except OSError:
            src = ""
    return path, src


# ---------------------------------------------------------------------------
# GOLDEN <-> SPEC CONSISTENCY GATE (robustness fix)
#
# The block uArch specs are written by the ARCHITECTURE phase against a
# particular reference golden. If someone later ENRICHES the golden (e.g. adds
# full-RDO mode search) WITHOUT regenerating the specs, the specs silently go
# STALE: they describe an encoder the golden no longer implements, so some
# blocks can NEVER be byte-exact. This gate hashes the resolved golden and the
# spec directory the first time it sees a run, and on every subsequent run
# FAILS a ``golden_spec_consistency`` check when the golden's hash changed AND
# the specs were NOT regenerated (their newest mtime is older than the golden's
# mtime). Wired as a startup precondition in ``run_microarch_exp`` so a stale
# run is flagged UP FRONT, not after N failed byte-exact attempts.
# ---------------------------------------------------------------------------

GOLDEN_SPEC_HASH_FILENAME = "golden_spec_hash.json"


def _golden_spec_hash_path(project_root: str) -> Path:
    return _microarch_dir(project_root) / GOLDEN_SPEC_HASH_FILENAME


def _sha256_file(path: str) -> str:
    """sha256 of a file's bytes (empty string on any failure)."""
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return ""


def _uarch_specs_dir_sha(project_root: str) -> str:
    """Stable sha256 over the uArch spec directory contents (name + bytes).

    Hashes every ``*.md`` spec file's relative name and content in sorted order
    so the digest changes iff a spec's name set OR any spec's bytes change.
    """
    specs = _uarch_specs_dir(project_root)
    h = hashlib.sha256()
    if specs.is_dir():
        for p in sorted(specs.glob("*.md")):
            h.update(p.name.encode("utf-8"))
            h.update(b"\0")
            try:
                h.update(p.read_bytes())
            except OSError:
                pass
            h.update(b"\0")
    return h.hexdigest()


def _newest_spec_mtime(project_root: str) -> float:
    """Newest mtime across the uArch spec files (0.0 when none exist)."""
    specs = _uarch_specs_dir(project_root)
    newest = 0.0
    if specs.is_dir():
        for p in specs.glob("*.md"):
            try:
                newest = max(newest, p.stat().st_mtime)
            except OSError:
                pass
    return newest


def _golden_mtime(golden_path: str | None) -> float:
    if not golden_path:
        return 0.0
    try:
        return Path(golden_path).stat().st_mtime
    except OSError:
        return 0.0


def check_golden_spec_consistency(
    golden_path: str | None,
    golden_sha: str,
    stored: dict | None,
    golden_mtime: float,
    newest_spec_mtime: float,
) -> dict:
    """Pure decision core for the golden<->spec consistency gate (unit-testable).

    Returns ``{"passed": bool, "reason": str, "should_store": bool}``.

    - No stored record yet (first run) -> PASS + should_store (record the
      baseline so a future desync is detectable).
    - Stored hash == current golden hash -> PASS (golden unchanged since specs
      were written; nothing to do).
    - Stored hash != current golden hash:
        * specs regenerated after the golden (newest_spec_mtime >= golden_mtime)
          -> PASS + should_store (the desync was resolved; refresh the record).
        * specs OLDER than the golden -> FAIL: the golden changed but the specs
          were not regenerated, so they are STALE.
    """
    if not golden_path or not golden_sha:
        # No resolvable golden -> gate is a logged no-op (cannot judge).
        return {"passed": True, "reason": "no resolvable golden; gate skipped",
                "should_store": False}
    if not stored or not isinstance(stored, dict) or not stored.get("golden_sha"):
        return {"passed": True,
                "reason": "no stored golden/spec hash yet; recording baseline",
                "should_store": True}
    if stored.get("golden_sha") == golden_sha:
        return {"passed": True, "reason": "golden unchanged since specs written",
                "should_store": False}
    # Hash changed. Were the specs regenerated after the golden?
    if newest_spec_mtime >= golden_mtime > 0.0:
        return {"passed": True,
                "reason": ("golden changed AND specs were regenerated after it "
                           "-- consistent; refreshing recorded hash"),
                "should_store": True}
    return {
        "passed": False,
        "reason": (
            "golden changed since specs written -- specs are STALE, regenerate "
            "architecture for the affected blocks. The reference golden "
            f"({golden_path}) hash changed from "
            f"{str(stored.get('golden_sha'))[:12]}.. to {golden_sha[:12]}.. but "
            "the uArch specs were NOT regenerated (their newest mtime "
            f"{newest_spec_mtime:.0f} is older than the golden's "
            f"{golden_mtime:.0f}). A model built to these specs can never be "
            "byte-exact to the current golden."
        ),
        "should_store": False,
    }


def golden_spec_consistency_gate(project_root: str) -> dict:
    """Run the golden<->spec consistency gate for a project (deterministic).

    Resolves the golden, computes hashes/mtimes, compares against the stored
    baseline in ``.coresmith/microarch/golden_spec_hash.json``, and (on a PASS
    that ``should_store``) writes/refreshes the baseline. Returns
    ``{"passed", "reason", "golden_sha", "uarch_specs_dir_sha"}``. Never raises.
    """
    golden_path, _ = _resolve_golden(project_root)
    golden_sha = _sha256_file(golden_path) if golden_path else ""
    specs_sha = _uarch_specs_dir_sha(project_root)
    golden_mt = _golden_mtime(golden_path)
    spec_mt = _newest_spec_mtime(project_root)

    stored: dict | None = None
    hp = _golden_spec_hash_path(project_root)
    if hp.exists():
        try:
            data = json.loads(hp.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                stored = data
        except (ValueError, OSError):
            stored = None

    decision = check_golden_spec_consistency(
        golden_path, golden_sha, stored, golden_mt, spec_mt,
    )
    if decision.get("should_store"):
        try:
            hp.parent.mkdir(parents=True, exist_ok=True)
            hp.write_text(json.dumps({
                "golden_sha": golden_sha,
                "uarch_specs_dir_sha": specs_sha,
                "golden_path": golden_path,
                "golden_mtime": golden_mt,
                "newest_spec_mtime": spec_mt,
            }, indent=2), encoding="utf-8")
        except OSError:
            pass
    return {
        "passed": bool(decision.get("passed", True)),
        "reason": decision.get("reason", ""),
        "golden_sha": golden_sha,
        "uarch_specs_dir_sha": specs_sha,
    }


# ---------------------------------------------------------------------------
# Deterministic block-model elaboration (used by lint_models and shared)
# ---------------------------------------------------------------------------

def _find_block_factory(module: Any, block_name: str):
    """Return the @block factory callable named ``block_name`` (or a lone one)."""
    fn = getattr(module, block_name, None)
    if callable(fn):
        return fn
    # A model whose factory name drifted: accept a single public callable.
    cands = [
        v for k, v in vars(module).items()
        if callable(v) and not k.startswith("_")
        and getattr(v, "__module__", None) == module.__name__
    ]
    if len(cands) == 1:
        return cands[0]
    return None


def elaborate_block_model(model_path: str, block_name: str) -> str | None:
    """Import, elaborate, and convert one Amaranth block model.

    A guarded import catches source/import errors. ``Fragment.get`` then walks
    the complete Elaboratable, and the Verilog backend exercises the same
    lowering boundary used by a real Amaranth-to-RTL flow. None == clean.
    """
    p = Path(model_path)
    if not p.exists():
        return f"model file missing: {model_path}"
    text = ""
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        return f"could not read model: {exc}"
    if not text.strip():
        return "model file is empty"

    # 1. Static parse (fast syntax check with a clear message).
    import ast
    try:
        ast.parse(text)
    except SyntaxError as exc:
        return f"syntax error: {exc}"

    # 2. Guarded import under a private module name.
    import importlib.util
    mod_name = f"_coresmith_uarch_exp_{p.stem}"
    spec = importlib.util.spec_from_file_location(mod_name, str(p))
    if spec is None or spec.loader is None:
        return "could not build import spec"
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001 - any import error invalidates it
        return f"import failed: {type(exc).__name__}: {exc}"

    factory = _find_block_factory(module, block_name)
    if factory is None or not callable(factory):
        return (
            f"no callable Elaboratable class named '{block_name}' "
            "(and not a lone public model)"
        )

    # 3. Amaranth elaboration + conversion with best-effort port Signals.
    try:
        from amaranth import Elaboratable, Fragment, Signal
        from amaranth.back import verilog
    except Exception as exc:  # noqa: BLE001 - amaranth not installed
        return f"amaranth unavailable: {type(exc).__name__}: {exc}"

    import inspect
    if not inspect.isclass(factory) or not issubclass(factory, Elaboratable):
        return f"'{block_name}' is not an Amaranth Elaboratable class"

    try:
        params = list(inspect.signature(factory).parameters.keys())
    except (TypeError, ValueError):
        params = ["clk", "rst"]

    def _mk_signal(name: str):
        n = name.lower()
        if n in ("clk", "clock"):
            return Signal(name=name)
        if n in ("rst", "reset", "rst_n", "resetn"):
            return Signal(name=name)
        if "vld" in n or "valid" in n or "ready" in n or "rdy" in n or "last" in n \
                or "en" == n or n.endswith("_en"):
            return Signal(name=name)
        return Signal(32, name=name)

    try:
        args = [_mk_signal(nm) for nm in params]
        inst = factory(*args)
        Fragment.get(inst, platform=None)

        # System Yosys is selected by AMARANTH_USE_YOSYS=system in the
        # experiment run environment; conversion proves the generated model
        # crosses the real Amaranth codegen boundary.
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rtl = verilog.convert(inst, name=block_name, ports=args)
        out_dir = Path(model_path).parent / "_uarch_exp_elab"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{block_name}.v").write_text(rtl, encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 - elaboration failure is the signal
        return f"elaboration failed: {type(exc).__name__}: {exc}"
    return None


# ---------------------------------------------------------------------------
# Agent prompt builders (kept module-level so tests can inspect without codex)
# ---------------------------------------------------------------------------

def _load_shared_skills() -> str:
    """Streaming/serialization discipline skills shared by all CoreSmith agents."""
    try:
        from orchestrator.langchain.prompts.skills import load_skills
        return load_skills(
            "axi_stream", "srdy_drdy", "arithmetic_precision",
            "serialization_contract", "buffer_stride_contract",
            "no_stimulus_keyed_memorization", "pipeline_contract",
            "verify_in_context",
        )
    except Exception:  # noqa: BLE001
        return ""


BUILD_MODELS_SYSTEM = """\
You are CoreSmith's MICROARCHITECTURE MODEL BUILDER (pass-1). This invocation
builds a COUPLING CLUSTER: the 1..N blocks named in the prompt. Usually that is
ONE block, but when the prompt lists several they are TIGHTLY COUPLED (a
reconstructed-neighbour feedback loop and/or a shared decomposition parent whose
internal boundary is NOT byte-aligned) and MUST be built TOGETHER in this one
shared context so they agree on that shared contract -- building them in
isolation loses the cross-block context. Focus all of your attention on THIS
cluster's blocks; write a model file for EACH block the prompt lists, and do NOT
touch any block's model file that is not in this cluster.

Discipline (based on CoreSmith's BlockGoldenGenerator):
- For EACH block named in the prompt, write `arch/block_models/<block>.py`
  containing `class <block>(Elaboratable)` NAMED EXACTLY after that block. Its
  constructor signature is `(clk, rst, <the handshake + data ports from that
  block's uArch spec / contract>)`; save all ports on `self`, and implement
  hardware in `elaborate(self, platform)`.
  When the cluster has multiple blocks, make their SHARED-BOUNDARY contract
  (the coupling edge -- feedback/writeback port widths, the intra-pipeline byte
  record, cycle timing) IDENTICAL on both sides; that mutual agreement is the
  whole reason they are built together.
- `elaborate()` TRANSCRIBES that block's slice of the reference software
  golden's EXACT math: `m.d.comb` for ready/glue and `m.d.sync` for registers.
  No heuristics, float substitution, or re-deriving the algorithm. Expressions
  in a sync branch read pre-edge state; use explicit next-state logic for
  same-cycle forwarding.
- Keep simulator processes OUT of block hardware. `Simulator`, `add_testbench`,
  clock generation, stimulus, and monitoring belong only in a testbench.
- MEMORIES ARE Amaranth `Array`s of fixed-width `Signal`s at this stage. Do NOT
  instantiate SRAM macros or reference macro
  LEF/GDS; macro concerns belong to the backend, not here.
- MEMORY-TIER DECISION (MANDATORY, per memory). For EVERY memory array you
  declare you MUST choose an implementation TIER by geometry using CoreSmith's
  rule and DOCUMENT the choice in the module docstring. The rule
  (`mem_characterize.recommend_impl_rule(width, depth, ports, d_crit=256)`):
    * depth > 256                 -> `macro`  (a flop read mux this deep is a
                                      multi-ns N:1 path -- reshape to a macro at
                                      backend; here it stays a plain array but
                                      DOCUMENT it as macro-tier + registered read)
    * depth <= 32 AND width >= 128 -> `registered_flop` (wide-shallow register file)
    * otherwise                    -> `registered_flop`
  The read of ANY flop memory MUST be REGISTERED (never a combinational N:1
  mux). JUSTIFY line-buffer vs frame-buffer explicitly: a LINE buffer is a few
  lines x line_width (shallow) -> `registered_flop`; a FRAME buffer is
  whole-frame depth (deep) -> `macro`/reshape. State, per memory in the module
  docstring: `# MEM <name>: WIDTHxDEPTH ports=<...> tier=<macro|registered_flop>
  role=<line_buffer|frame_buffer|...> reason=<...>`.
- SYNTHESIZABLE Amaranth ONLY -- the model IS the hardware description, not a
  Python reference with a hardware wrapper. HARD BANS (each is auto-rejected):
  * Do not hand-write or inject Verilog. `elaborate()` is the single source of
    truth and MUST lower with `amaranth.back.verilog.convert`.
  * NO Python behavioral objects / dynamic data structures in the runtime path:
    no `self.bits = []` / `.append()` / `.pop()` / `while len(buf) ...` / helper
    class instances (`ctx = _SomeContext()`) called as runtime hardware. Variable-
    length / bit-packing logic (entropy coding, byte-framing, entropy) MUST be a BOUNDED FSM with
    fixed-width registers + shift registers, not a growable Python list.
  * Elaboration loops must be STATICALLY BOUNDED (`for i in range(N)` with
    constant N). Signal-indexed lookup requires `Array`; never use a Signal as a
    Python-list index or a data-dependent Python `while`.
  If an algorithm genuinely cannot be expressed as bounded registered Amaranth,
  declare an `INFEASIBLE-INTERFACE-GAP`; do not emit a constant stub.
- Every model MUST import, elaborate, and convert with the Amaranth Verilog
  backend with no missing symbols. Use exact `Signal(W)` / `Signal(signed(W))`
  shapes; Amaranth silently wraps, so widen intermediates and explicitly
  slice/clamp only where the hardware contract requires it.

BYTE-EXACT PITFALLS -- the EXACT failure classes that walled prior runs. Obey each:
  1. RESET DISCIPLINE: EVERY handshake/status output (`*_tready`, `*_tvalid`,
     `*_tlast`, status/valid flags) MUST be DEASSERTED (0) while reset is active.
     Never assert `s_axis_*_tready`/`m_axis_*_tvalid` during reset. (A prior GPIO
     shell failed: `s_axis_host_in_tready` asserted while reset was active.)
  2. INTERFACE COMPLETENESS: the class constructor MUST expose EVERY port
     from `block_diagram.json` for that block -- every `s_axis_*`/`m_axis_*`/config
     interface, exactly named. A dropped interface fails the gate. (A prior shell
     dropped `s_axis_host_in`/`m_axis_host_out`.)
  3. FIXED-HEADER BYTE-EXACT: header/parameter-set emitters (parameter-set/byte-framing start
     codes) MUST reproduce the golden's `build_header_a`/`build_header_b` byte sequence
     EXACTLY -- `config_field`, `tier_field`, every ue(v)/u(n) field, emulation-
     prevention -- byte for byte. Transcribe the golden's emission; do NOT re-derive
     the header. (A prior entropy streamer emitted the wrong/missing `config_field`.)
  4. QUANT/TRANSFORM EXACT: the forward transform + quantize MUST reproduce the
     golden EXACTLY -- the same `_MF`/LevelScale entry per coefficient class, the
     SAME dead-zone offset `f = (1<<qbits)//3`, the SAME `>>qbits` shift, and sign
     handling. One wrong scaling corrupts the first coefficient and the entire
     syntax record. Copy the golden's `quantize`/`_fdct4` arithmetic verbatim into
     registered form; do not approximate. (A prior intra core mismatched the first
     quantized coefficient of the 6252-bit record.)
  5. DECISION COMPLETENESS (mode/RDO) -- the coefficient levels a block emits are a
     function of the MODE-DECISION it makes, so the decision MUST be the golden's,
     not a hardware-friendlier subset. If the golden's per-unit encode
     (`_encode_mb`) evaluates the FULL candidate set -- all 9 Intra_4x4 modes via
     `avail_modes_4x4`, the Intra_16x16 path (`_try_intra16x16`), chroma-mode RD
     (`decide_chroma_mode`) -- and selects on the golden's cost (lambda-weighted
     `_rd_cost(SSD, bits)` incl. the entropy coding rate term, NOT SSD alone), then your
     block MUST reproduce that EXACT search and EXACT cost. A restricted decision
     (e.g. "3 candidates DC/V/H, SSD-only, Intra_4x4-only") is a DIFFERENT, lossy
     encoder: it agrees on edge blocks that have no neighbours but diverges the
     moment a neighbour makes a non-DC/16x16 mode win -- corrupting that block's
     levels and every downstream entropy coding byte. Isolation proof from a walled run: the
     3-mode/SSD subset first diverged at luma block 3 (coeff slot 48), golden=-1 vs
     subset=0, because the golden picked mode 2 there and the subset picked mode 1.
     If (and ONLY if) the block's OWN uArch spec mandates the restricted decision
     while the golden uses the full one, the spec is unbuildable-to-golden: DO NOT
     fake byte-exactness -- report the spec/golden decision mismatch explicitly
     (name the missing modes/paths + the cost term) and flag it as an ARCHITECTURE
     defect, because no faithful model can close it.
  6. FULL SYNTAX ENGINE, not per-field bytes (entropy/byte-framing) -- an entropy
     streamer whose fixed header (parameter-set/frame-markers/unit header) matches the golden
     is NOT byte-exact if the SLICE PAYLOAD is faked. After the unit header the
     golden emits an unary/exp coding SLICE HEADER (`ue(first_block_in_group) ue(slice_type)
     ue(pps_id) u(frame_num,4) ...se(qp-26) ue(filter_mode)`) then the entropy coding-coded
     MB layer (`_encode_mb` -> `entropy_encode` with the real coeff-token / level /
     total_zeros / run_before tables and the neighbour-derived `nC` context and MPM
     flags), then `rbsp_trailing_bits` + byte-escaped byte-stuffing. Emitting the raw
     metadata fields as bytes (e.g. one qp byte, one mb_index byte, one coeff byte)
     is the exact cheat this bans: it passes the header check then diverges on the
     FIRST slice-payload byte. Isolation proof: golden slice byte 24 = 0x88 (the
     unary/exp coding slice header) vs a faked-field model's 0x1c (raw qp). You MUST
     port the golden's BitWriter (`ue`/`se`/`u`), `entropy_encode` tables, bit-accumulator
     trailing bits, and `frame_pack` into a BOUNDED registered FSM (per rule in
     the HARD BANS above -- fixed-width shift-register accumulator, no growable
     Python list). NOTE the coupling: this block is byte-exact ONLY when its input
     coefficient records are already golden-exact; if the upstream mode-decision
     block is lossy (rule 5) even a perfect entropy coding engine here emits wrong bytes.

After writing each block's model, RUN YOUR OWN SANITY CHECK: import the file and do a
quick smoke elaboration (instantiate signals + a short pysim or a Verilog backend
conversion) to confirm it imports and elaborates. Fix anything that fails before you
finish. Report which block(s) you wrote and the result of each smoke check.
"""

VERIFY_MODELS_SYSTEM = """\
You are CoreSmith's MICROARCHITECTURE MODEL VERIFIER (pass-1), a rigorous DV
engineer. You verify EXACTLY ONE block: the single block named in the prompt. You
are one of a bounded, concurrent per-block fan-out of focused DV agents. You
mirror CoreSmith's Validation-DV anti-cheat: functional equivalence is proven,
never asserted.

Step 1 -- DESIGN a test plan for YOUR block against that block's slice of the
reference software golden. Build a REQUIREMENT_COVERAGE-style manifest: enumerate
your block's FRD functional vectors (FUNC-NNN) and directed edge cases (min/max
operands, empty/'full' streams, back-to-back handshakes, boundary framing, reset
mid-stream) AND a randomized-stimulus regime. State which vectors you cover.

Step 2 -- IMPLEMENT + RUN. Drive your block's Amaranth `Elaboratable` model with
`Simulator.add_clock(20e-9)` and an async `add_testbench`. `await ctx.tick()`
before post-edge sampling; keep reset/stimulus/monitor logic in the simulator,
not `elaborate()`. Compare its
output BYTE/VALUE-WISE to the golden reference for that block. This is
MODEL-vs-GOLDEN (the model is the DUT, the software golden is the oracle) -- NOT
RTL-vs-model.

NON-NEGOTIABLE ANTI-CHEAT RULES (a plausible but wrong model MUST fail here):
- FUNCTIONAL CHECK IS NON-DEFERRABLE. You MUST assert the model's output == the
  golden output BYTE/VALUE-EXACT on BOTH the directed edge cases AND randomized
  stimulus. Cover EVERY FRD vector for the block.
- A structural / trace / existence / byte-count / handshake-only check does NOT
  satisfy a functional requirement. A datapath that emits a constant, flat, or
  structurally-correct-but-WRONG value MUST fail. If your bounded check would
  pass such a model, it is not a functional check.
- The comparison is EXTERNAL: your harness compares the model's output against
  the golden. NEVER accept the model's own self-report of correctness.
- You may defer ONLY an EXHAUSTIVE / full-dataset / fuzz sweep (mark it
  `deferred_to_golden_sweep`). You may NEVER defer the bounded directed
  output-equals-golden equivalence itself.

Report your block's pass/fail and, on failure, the FIRST DIVERGENCE as
`{"summary": ..., "golden_observation": ..., "model_observation": ...,
"vector": "FUNC-NNN or the stimulus"}`. Emit a JSON summary object mapping
YOUR block -> {passed, first_divergence, detail}. A block ABSENT from the JSON is
treated as FAILED (unverified == not proven).
"""

DIAGNOSE_MODELS_SYSTEM = """\
You are CoreSmith's MICROARCHITECTURE DIAGNOSTICIAN (pass-1). One or more block
models failed lint (import/elaboration), verify (model-vs-golden divergence), or size
(pipeline/Fmax infeasibility).

Read the failing block models, the collected failures, and the reference golden.
For EACH failing block, diagnose the ROOT CAUSE (not the symptom): which line / which
contract / which math step is wrong, and the SPECIFIC change the model builder must
make. Do NOT hand-patch the models yourself -- your job is to produce structured
FEEDBACK that the model builder acts on next round.

Emit clear, per-block, actionable feedback (a bullet per failing block). Be concrete:
name the block, the failure class (lint/verify/size), the divergence or error, and the
fix.
"""


def _spec_digest(specs: dict[str, str], limit: int = 6000) -> str:
    """Compact per-block uArch spec bundle for a prompt."""
    parts = []
    for b, text in specs.items():
        chunk = text.strip()
        if len(chunk) > limit:
            chunk = chunk[:limit] + "\n... [truncated]"
        parts.append(f"\n===== uArch spec: {b} =====\n{chunk}")
    return "\n".join(parts)


def build_build_models_prompt(
    project_root: str,
    blocks: list[str],
    specs: dict[str, str],
    golden_path: str | None,
    golden_src: str,
    feedback: str,
) -> str:
    out_dir = _block_models_dir(project_root)
    _n = len(blocks)
    _lead = (
        "Write an Amaranth Elaboratable model for the SINGLE block below."
        if _n == 1 else
        "Write an Amaranth Elaboratable model for EACH block below."
    )
    lines = [
        _lead,
        f"\nProject root: {project_root}",
        f"Write each model to: {out_dir}/<block>.py",
        f"\nBLOCKS ({_n}): {', '.join(blocks)}",
        _spec_digest(specs),
    ]
    if _prompt_slim_enabled():
        # B4 prompt-slim: reference the golden by path + a HEAD (not the whole
        # 60 KB source). The per-block model gate re-checks each model.
        _head = "\n".join((golden_src or "").splitlines()[:40])
        lines += [
            f"\n===== REFERENCE SOFTWARE GOLDEN ({golden_path or 'unknown'}) =====",
            "READ THE FULL FILE at the path above (first 40 lines shown for "
            "orientation). After writing each model, run "
            "`coresmith verify model <block>` and fix it until it passes before "
            "you finish this turn.",
            "```python",
            _head or "# (golden source not found -- use the uArch specs)",
            "```",
        ]
    else:
        lines += [
            f"\n===== REFERENCE SOFTWARE GOLDEN ({golden_path or 'unknown'}) =====",
            "```python",
            (golden_src or "# (golden source not found -- use the uArch specs)")[:60000],
            "```",
        ]
    # DISK-FIRST CONSTRAINT LOOP: point the builder at each block's persistent
    # constraints.json (accumulated MUST / MUST-NOT rules that survive ALL
    # retries), and inline the current rules so they are honoured even if the
    # file read is skipped. This mirrors the frontend's
    # .coresmith/blocks/<b>/constraints.json memory.
    constr_lines = [
        "\n===== PER-BLOCK CONSTRAINTS (READ THESE FILES FIRST -- accumulated "
        "MUST / MUST-NOT rules that PERSIST across every retry) =====",
    ]
    any_constraints = False
    for b in blocks:
        cpath = _block_constraints_path(project_root, b)
        rules = _read_block_constraints(project_root, b)
        constr_lines.append(f"\n- {b}: read {cpath}")
        for r in rules:
            any_constraints = True
            kind = str(r.get("kind", "must")).upper().replace("_", "-")
            constr_lines.append(f"    [{kind}] {r.get('rule', '')}")
    if not any_constraints:
        constr_lines.append(
            "  (no constraints accumulated yet -- but you MUST still read the "
            "files above; they will fill up on retries.)"
        )
    try:
        from orchestrator.langgraph.contract_conformance import (
            CONSTRAINT_PRECEDENCE_LINE as _PRECEDENCE,
        )
        constr_lines.append("\n" + _PRECEDENCE)
    except Exception:  # noqa: BLE001 - prompt garnish, never blocks a build
        pass
    lines += constr_lines
    if feedback.strip():
        lines += [
            "\n===== DIAGNOSE FEEDBACK FROM THE PREVIOUS ROUND (transient; the "
            "durable rules are in the constraints.json files above) =====",
            feedback.strip(),
        ]
    if _n == 1:
        lines.append(
            "\nWrite this block's model file now, run your own smoke check, and "
            "report the block you wrote plus your smoke-check result."
        )
    else:
        lines.append(
            "\nWrite ALL block model files now, run your own smoke checks, and "
            "report which blocks you wrote plus your smoke-check results."
        )
    return "\n".join(lines)


def build_verify_models_prompt(
    project_root: str,
    blocks: list[str],
    golden_path: str | None,
) -> str:
    models_dir = _block_models_dir(project_root)
    _n = len(blocks)
    _lead = (
        "Verify this Amaranth block model against the reference golden "
        "(MODEL-vs-GOLDEN)."
        if _n == 1 else
        "Verify these Amaranth block models against the reference golden "
        "(MODEL-vs-GOLDEN)."
    )
    return "\n".join([
        _lead,
        f"\nProject root: {project_root}",
        f"Block models are in: {models_dir}/<block>.py",
        f"Reference golden: {golden_path or 'unknown (locate it under the run dir)'}",
        f"\nBLOCKS ({_n}): {', '.join(blocks)}",
        "\nStep 1: DESIGN the exhaustive test plan for the block(s) above.",
        "Step 2: IMPLEMENT + RUN the plan in Amaranth pysim and compare "
        "byte/value-wise to the golden.",
        "\nEnd with a JSON object: {block: {\"passed\": bool, "
        "\"first_divergence\": ..., \"detail\": ...}}.",
    ])


def build_diagnose_prompt(
    project_root: str,
    lint_errors: dict[str, str],
    verify_results: dict[str, dict],
    size_results: dict[str, dict],
    golden_path: str | None,
) -> str:
    models_dir = _block_models_dir(project_root)
    fails = _collect_failures(lint_errors, verify_results, size_results)
    return "\n".join([
        "One or more block models failed. Diagnose root cause per failing block and "
        "emit structured feedback for the model builder.",
        f"\nProject root: {project_root}",
        f"Block models are in: {models_dir}/<block>.py",
        f"Reference golden: {golden_path or 'unknown'}",
        "\n===== FAILURES =====",
        json.dumps(fails, indent=2, default=str),
        "\nEmit per-block, actionable feedback (failure class + root cause + fix).",
    ])


def _collect_failures(
    lint_errors: dict[str, str],
    verify_results: dict[str, dict],
    size_results: dict[str, dict],
) -> dict[str, dict]:
    """Merge the three stages' failures into a per-block record."""
    fails: dict[str, dict] = {}
    for b, err in (lint_errors or {}).items():
        if err:
            fails.setdefault(b, {})["lint"] = err
    for b, res in (verify_results or {}).items():
        if isinstance(res, dict) and not res.get("passed", False):
            fails.setdefault(b, {})["verify"] = res
    for b, res in (size_results or {}).items():
        if isinstance(res, dict) and not res.get("feasible", True):
            fails.setdefault(b, {})["size"] = res
    return fails


# ---------------------------------------------------------------------------
# Bounded per-block codex fan-out (BOX SAFETY: never unbounded)
# ---------------------------------------------------------------------------

async def _bounded_fanout(
    blocks: list[str],
    per_block,
    concurrency: int,
) -> dict[str, Any]:
    """Run ``per_block(block)`` for every block, at most ``concurrency`` in flight.

    ``per_block`` is an async callable ``(block) -> Any``. Each block runs inside
    an ``asyncio.Semaphore(concurrency)`` so the number of concurrent codex execs
    is HARD-CAPPED at ``concurrency`` (BOX SAFETY: a prior unbounded fork-storm
    hung the machine; every codex exec is ~7 procs). A single block raising does
    NOT sink the batch -- its exception is captured and returned in place of its
    result, so ``asyncio.gather`` never propagates one failure over the others.

    Returns ``{block: result_or_Exception}`` in ``blocks`` order.
    """
    import asyncio

    concurrency = max(1, int(concurrency))
    sem = asyncio.Semaphore(concurrency)

    async def _guarded(block: str):
        async with sem:
            try:
                return await per_block(block)
            except Exception as exc:  # noqa: BLE001 - one block failing is isolated
                return exc

    coros = [_guarded(b) for b in blocks]
    settled = await asyncio.gather(*coros)  # never raises: exceptions are values
    return dict(zip(blocks, settled))


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

async def build_models_node(state: dict) -> dict:
    """AGENT 1 (codex): CLUSTER-AWARE fan-out -- one call per COUPLING CLUSTER.

    The fan-out UNIT is a coupling cluster (``derive_coupling_clusters``), NOT
    always a singleton: tightly-coupled blocks (the intra recon-neighbour feedback
    loop / a shared decomposition parent with a non-byte-aligned boundary) are
    built TOGETHER in ONE shared codex context so they agree on their shared
    contract, while independent blocks fan out solo. ``CLUSTERING=off`` -> every
    block is its own singleton (== the exp13 per-block behavior). Each cluster runs
    under the SAME bounded ``asyncio.Semaphore`` (BOX SAFETY, never unbounded) with
    PER-CLUSTER resume (a cluster resumes its own codex session across attempts).
    """
    pr = _pr(state)
    attempt = int(state.get("attempt", 0)) + 1
    blocks = state.get("blocks") or discover_blocks(pr)
    # FOCUSED/INCREMENTAL BUILD: on a rebuild, regenerate ONLY the blocks that
    # failed a gate last round -- a single build context can only give so many
    # blocks enough attention. Regenerating ALL blocks each attempt dilutes
    # per-block quality and regresses passing blocks: the 14-block decomposed run
    # stalled at 11/14 with even simple blocks (that close in the 8-block runs)
    # never converging, because 14-in-one-context overloaded the builder. Passing
    # blocks' models stay on disk untouched; lint/verify still check ALL blocks.
    build_targets = blocks
    if attempt > 1 and _microarch_incremental_enabled():
        # Prefer the diagnose-set failed_blocks; fall back to recomputing from
        # the last round's gate results (belt-and-suspenders / older state).
        failing = set(state.get("failed_blocks") or [])
        if not failing:
            failing = set(_collect_failures(
                state.get("lint_errors") or {},
                state.get("verify_results") or {},
                state.get("size_results") or {},
            ).keys())
        if failing:
            build_targets = [b for b in blocks if b in failing]
    _log(pr, f"build_models: attempt {attempt}, building {len(build_targets)} of "
             f"{len(blocks)} block(s): {build_targets}")

    if not blocks:
        _log(pr, "build_models: NO BLOCKS discovered -- aborting.")
        return {
            "attempt": attempt, "blocks": [], "status": "error",
            "feedback": "no blocks discovered from arch/block_diagram.* or uarch_specs/",
        }

    _block_models_dir(pr).mkdir(parents=True, exist_ok=True)
    golden_path, golden_src = _resolve_golden(pr)

    system = BUILD_MODELS_SYSTEM
    skills = _load_shared_skills()
    if skills:
        system = system + "\n\n# Reference Skills (streaming protocol)\n\n" + skills

    # CLUSTER-AWARE FAN-OUT: partition the WHOLE block list into coupling clusters
    # (a cluster is the tightly-coupled unit; independent blocks are singletons),
    # then keep only the clusters that INTERSECT the build_targets. INCREMENTAL
    # RULE: a cluster is rebuilt as a WHOLE if ANY of its blocks failed -- shared
    # context is the whole point of clustering, so we never split the cluster on a
    # retry. Clusters with all-passing blocks are skipped (their models stay on
    # disk). Each cluster is built by ONE codex call with ALL its blocks' specs,
    # run concurrently under the SAME BOUNDED semaphore (BOX SAFETY).
    all_clusters = derive_coupling_clusters(pr, blocks)
    target_set = set(build_targets)
    clusters = [cl for cl in all_clusters if any(b in target_set for b in cl)]
    # Rebuild the WHOLE cluster (all its blocks) for shared context even if only
    # one member failed.
    cluster_blocks = sorted({b for cl in clusters for b in cl})
    specs = _read_uarch_specs(pr, cluster_blocks)

    fanout = _microarch_fanout()
    feedback = state.get("feedback", "") or ""

    # PER-CLUSTER RESUME store (cluster_id -> codex session id). Seeded from the
    # prior round's cluster store; falls back to the per-block store (and the
    # legacy singular id) so a singleton cluster still resumes an earlier
    # per-block session across the cluster refactor.
    prior_cluster_sids: dict[str, str] = dict(state.get("build_cluster_session_ids") or {})
    prior_block_sids: dict[str, str] = dict(state.get("build_session_ids") or {})
    legacy_sid = (state.get("build_session_id") or "").strip()
    if legacy_sid and len(build_targets) == 1 and build_targets[0] not in prior_block_sids:
        prior_block_sids[build_targets[0]] = legacy_sid
    new_cluster_sids: dict[str, str] = dict(prior_cluster_sids)
    new_block_sids: dict[str, str] = dict(prior_block_sids)

    _log(pr, f"build_models: cluster-aware fan-out over {len(clusters)} cluster(s) "
             f"[{[len(c) for c in clusters]}] covering {len(cluster_blocks)} "
             f"block(s), max {fanout} concurrent codex call(s).")

    def _prior_cluster_sid(cluster: list[str]) -> str:
        cid = _cluster_id(cluster)
        sid = (prior_cluster_sids.get(cid) or "").strip()
        if sid:
            return sid
        # Singleton cluster: fall back to that block's prior per-block session id.
        if len(cluster) == 1:
            return (prior_block_sids.get(cluster[0]) or "").strip()
        return ""

    # Map a stable cluster_id -> its block list so the bounded fan-out (keyed by
    # cluster_id) can recover the cluster's members inside the per-unit callable.
    cluster_by_id: dict[str, list[str]] = {_cluster_id(cl): cl for cl in clusters}

    async def _build_one(cid: str):
        cluster = cluster_by_id[cid]
        cspecs = {b: specs[b] for b in cluster if b in specs}
        prompt = build_build_models_prompt(
            pr, cluster, cspecs, golden_path, golden_src, feedback,
        )
        prior_sid = _prior_cluster_sid(cluster)
        resume_sid = prior_sid if attempt > 1 else None
        label = cluster[0] if len(cluster) == 1 else cid
        if resume_sid:
            _log(pr, f"build_models[{label}]: resuming codex session {resume_sid}")
        # A fresh LLM per cluster so last_session_id capture never races across the
        # concurrent tasks (each call resets + stashes it on its own instance).
        llm = _make_llm()
        await llm.call(
            system=system, prompt=prompt, run_name=f"Build Models [{label}]",
            resume_session_id=resume_sid,
        )
        return getattr(llm, "last_session_id", "") or prior_sid

    settled = await _bounded_fanout(list(cluster_by_id.keys()), _build_one, fanout)
    for cid, res in settled.items():
        cluster = cluster_by_id[cid]
        label = cluster[0] if len(cluster) == 1 else cid
        if isinstance(res, Exception):
            _log(pr, f"build_models[{label}]: agent error: "
                     f"{type(res).__name__}: {res}")
            continue
        if res:
            new_cluster_sids[cid] = res
            # Mirror the cluster's session id onto EACH of its blocks so per-block
            # resume/back-compat callers keep working.
            for b in cluster:
                new_block_sids[b] = res

    written = [b for b in blocks if (_block_models_dir(pr) / f"{b}.py").exists()]
    _log(pr, f"build_models: {len(written)}/{len(blocks)} model files present on disk.")
    out: dict = {
        "attempt": attempt,
        "blocks": blocks,
        "status": "running",
        "build_session_ids": new_block_sids,
        "build_cluster_session_ids": new_cluster_sids,
        # Clear feedback now that the builder consumed it.
        "feedback": "",
    }
    # Back-compat mirror: expose the sole target's session id under the legacy
    # singular key so older callers/tests keying on `build_session_id` still work.
    if len(build_targets) == 1:
        out["build_session_id"] = new_block_sids.get(build_targets[0], "")
    return out


def lint_models_node(state: dict) -> dict:
    """DETERMINISTIC: import + elaborate every @block; collect per-block errors.

    Also runs the anti-cheat INTERFACE-CONSTRAINT check: the model's @block
    factory signature must carry every interface the block owns in
    ``arch/block_diagram.json``. A dropped interface is a lint-level failure.
    """
    pr = _pr(state)
    blocks = state.get("blocks") or []
    models_dir = _block_models_dir(pr)
    block_diagram = _read_block_diagram(pr)
    # B4 incremental: skip re-linting blocks whose model is unchanged since it
    # last passed a gate (they stay clean; absent from errors == clean).
    incremental = _microarch_incremental_enabled()
    passing = state.get("passing_models") or {}
    _skipped: list[str] = []
    errors: dict[str, str] = {}
    for b in blocks:
        path = models_dir / f"{b}.py"
        if incremental and _unchanged_since_pass(path, passing.get(b)):
            _skipped.append(b)
            continue
        err = elaborate_block_model(str(path), b)
        if err:
            errors[b] = err
            continue
        # Interface check only when elaboration is clean (so the params import).
        expected = _expected_ports_for_block(block_diagram, b)
        if expected:
            params = _factory_params(str(path), b)
            if params is not None:
                missing = check_interface_constraint(params, expected)
                if missing:
                    errors[b] = (
                        "interface mismatch: the @block factory signature is "
                        f"missing expected interface(s) {missing} from "
                        f"arch/block_diagram.json (factory params: {params}). "
                        "The model must expose every port the block owns."
                    )
    status = "running" if not errors else "failed"
    if _skipped:
        _log(pr, f"lint_models: skipped {len(_skipped)} unchanged passing "
                 f"block(s): {_skipped}")
    if errors:
        _log(pr, f"lint_models: {len(errors)}/{len(blocks)} FAILED: "
                 + ", ".join(f"{b}: {e[:80]}" for b, e in errors.items()))
    else:
        _log(pr, f"lint_models: all {len(blocks)} model(s) elaborate clean.")
    return {"lint_errors": errors, "status": status}


async def verify_models_node(state: dict) -> dict:
    """AGENT 2 (codex): design exhaustive test plan, sub-agent fan-out runs it.

    B4 incremental: re-verify ONLY blocks whose model changed since they last
    passed; carry forward prior passing results for unchanged blocks (and skip
    the LLM entirely when nothing changed). Blocks that pass are stamped in
    ``passing_models`` (model mtime) so the next lint/verify can skip them.
    """
    pr = _pr(state)
    blocks = state.get("blocks") or []
    golden_path, _ = _resolve_golden(pr)
    models_dir = _block_models_dir(pr)
    incremental = _microarch_incremental_enabled()
    passing = dict(state.get("passing_models") or {})
    prior_results = state.get("verify_results") or {}
    lint_errors = state.get("lint_errors") or {}

    to_verify = blocks
    skipped: list[str] = []
    if incremental:
        to_verify = [
            b for b in blocks
            if not _unchanged_since_pass(models_dir / f"{b}.py", passing.get(b))
        ]
        skipped = [b for b in blocks if b not in to_verify]

    # Carry forward the prior passing result for unchanged (skipped) blocks.
    results: dict[str, dict] = {}
    for b in skipped:
        results[b] = prior_results.get(b) or {
            "passed": True, "detail": "unchanged since last pass (skipped re-verify)",
        }
    if skipped:
        _log(pr, f"verify_models: skipped {len(skipped)} unchanged passing "
                 f"block(s): {skipped}")

    if not to_verify:
        _log(pr, "verify_models: nothing changed -- all blocks carried forward.")
        return {"verify_results": results, "passing_models": passing,
                "status": "running"}

    # REAL PER-BLOCK FAN-OUT: one focused DV codex call PER block, run
    # concurrently under a BOUNDED asyncio.Semaphore (BOX SAFETY -- never
    # unbounded; each codex exec is ~7 procs). Each call is a focused DV agent
    # verifying THAT ONE block's Amaranth model vs its golden slice byte-exact, with
    # all anti-cheat rules scoped to the single block. Results are parsed per
    # block and aggregated into the same shape _parse_verify_results produces.
    # (Verify is kept per-block stateless -- the resume win is on build.)
    fanout = _microarch_fanout()
    _log(pr, f"verify_models: per-block fan-out over {len(to_verify)} model(s) vs "
             f"golden, max {fanout} concurrent codex call(s).")

    async def _verify_one(block: str):
        prompt = build_verify_models_prompt(pr, [block], golden_path)
        llm = _make_llm()
        content = await llm.call(
            system=VERIFY_MODELS_SYSTEM, prompt=prompt,
            run_name=f"Verify Models [{block}]",
        )
        # Parse this ONE block's record out of its own JSON summary.
        return _parse_verify_results(content, [block])[block]

    settled = await _bounded_fanout(to_verify, _verify_one, fanout)
    for block, res in settled.items():
        if isinstance(res, Exception):
            _log(pr, f"verify_models[{block}]: agent error: "
                     f"{type(res).__name__}: {res}")
            results[block] = {
                "passed": False,
                "first_divergence": _normalize_first_divergence(None),
                "detail": f"agent error: {res}",
            }
        else:
            results[block] = res
    # Stamp passing_models for blocks that passed verify AND were lint-clean, so
    # the next round's lint/verify can skip them while their model is unchanged.
    if incremental:
        for b in to_verify:
            if results.get(b, {}).get("passed") and not lint_errors.get(b):
                mt = _model_mtime(models_dir / f"{b}.py")
                if mt is not None:
                    passing[b] = mt

    failed = [b for b, r in results.items() if not r.get("passed", False)]
    status = "running" if not failed else "failed"
    if failed:
        _log(pr, f"verify_models: {len(failed)}/{len(blocks)} FAILED: {failed}")
    else:
        _log(pr, f"verify_models: all {len(blocks)} model(s) match golden.")
    return {"verify_results": results, "passing_models": passing, "status": status}


def size_node(state: dict) -> dict:
    """DETERMINISTIC: real per-block DFG + register/memory area vs budget.

    For each block: build a datapath DFG from ALL arithmetic in the
    ``@always_seq`` (not just the worst single line), schedule it for Fmax, size
    declared memory arrays via ``mem_characterize.predict_mem``, sum
    datapath+memory area, and compare against the block's ``area_budget_um2``.
    Over Fmax OR over area == infeasible.
    """
    pr = _pr(state)
    blocks = state.get("blocks") or []
    target_mhz = _read_target_clock_mhz(pr)
    specs = _read_uarch_specs(pr, blocks)
    _log(pr, f"size: sizing {len(blocks)} model(s) at {target_mhz:.0f} MHz.")
    results: dict[str, dict] = {}
    mem_results: dict[str, dict] = {}
    for b in blocks:
        path = _block_models_dir(pr) / f"{b}.py"
        res = _size_one_model(str(path), b, target_mhz, specs.get(b, ""))
        results[b] = res
        mem_results[b] = {
            "memories": res.get("memories", []),
            "mem_area_um2": res.get("mem_area_um2", 0.0),
            "datapath_area_um2": res.get("datapath_area_um2", 0.0),
            "total_area_um2": res.get("total_area_um2"),
            "area_budget_um2": res.get("area_budget_um2"),
        }
    infeasible = [b for b, r in results.items() if not r.get("feasible", True)]
    status = "running" if not infeasible else "failed"
    if infeasible:
        _log(pr, f"size: {len(infeasible)}/{len(blocks)} INFEASIBLE: {infeasible}")
    else:
        _log(pr, f"size: all {len(blocks)} model(s) feasible at {target_mhz:.0f} MHz.")
    return {"size_results": results, "mem_results": mem_results, "status": status}


# ---------------------------------------------------------------------------
# PPA JUDGE (NEW) -- impartial verdict node (mirrors OutputContractReviewAgent)
# ---------------------------------------------------------------------------

_PPA_JUDGE_PROMPT_FILE = (
    Path(__file__).resolve().parent.parent
    / "langchain" / "prompts" / "microarch_ppa_judge.md"
)


def _load_ppa_judge_prompt() -> str:
    try:
        return _PPA_JUDGE_PROMPT_FILE.read_text(encoding="utf-8")
    except OSError:
        return "You are the microarch PPA judge. Emit the verdict JSON."


def _read_frd_vectors(project_root: str) -> list[dict]:
    """Parse ALL FRD FUNC-NNN requirement vectors (never raises)."""
    frd = _arch_dir(project_root) / "frd_spec.md"
    if not frd.exists():
        return []
    try:
        from orchestrator.architecture.composition import parse_func_vectors
        return parse_func_vectors(frd.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return []


def build_ppa_judge_prompt(
    project_root: str,
    blocks: list[str],
    frd_vectors: list[dict],
    budgets: dict[str, dict],
    size_results: dict[str, dict],
    mem_results: dict[str, dict],
    target_mhz: float,
    output_path: str,
) -> str:
    vec_lines = [
        f"- {v.get('id')}: block={v.get('block') or '?'} "
        f"stimulus={str(v.get('stimulus'))[:120]} "
        f"expected={str(v.get('expected_output'))[:120]}"
        for v in frd_vectors
    ] or ["- (no FRD FUNC vectors parsed -- coverage cannot be judged)"]
    return "\n".join([
        f"Project root: {project_root}",
        f"Output path: {output_path}",
        f"Target clock: {target_mhz:.0f} MHz",
        f"\nBLOCKS ({len(blocks)}): {', '.join(blocks)}",
        "\n## FRD functional requirements (ALL must be covered)",
        *vec_lines,
        "\n## Per-block budgets (area_budget_um2 / flip_flop_budget)",
        json.dumps(budgets, indent=2, default=str),
        "\n## Measured sizing (datapath area / Fmax / depth)",
        json.dumps(size_results, indent=2, default=str),
        "\n## Measured memory sizing (impl / area / macro_feasible)",
        json.dumps(mem_results, indent=2, default=str),
        "\nRender the IMPARTIAL verdict and write ONLY the JSON to the output "
        "path (also print it).",
    ])


def _normalize_ppa_verdict(result: dict) -> dict:
    """Coerce a PPA-judge JSON into the structured schema (tolerant defaults).

    Schema: ``{passed, verdict in {pass,fail,escalate}, violations[],
    first_divergence{summary,golden_observation,model_observation,vector},
    recommended_action in {rebuild, ask_human}}``.
    """
    verdict = str((result or {}).get("verdict", "")).strip().lower()
    if verdict not in ("pass", "fail", "escalate"):
        # infer from `passed` when verdict missing/garbled
        verdict = "pass" if (result or {}).get("passed") is True else "fail"
    violations = (result or {}).get("violations") or []
    if not isinstance(violations, list):
        violations = [violations]
    passed = verdict == "pass" and not violations
    action = str((result or {}).get("recommended_action", "")).strip().lower()
    if action not in ("rebuild", "ask_human"):
        action = "ask_human" if verdict == "escalate" else "rebuild"
    return {
        "passed": bool(passed),
        "verdict": verdict,
        "violations": violations,
        "first_divergence": _normalize_first_divergence(
            (result or {}).get("first_divergence")
        ),
        "recommended_action": action,
    }


def _parse_ppa_json(content: str) -> dict:
    """Pull the last balanced JSON object out of the judge's text."""
    blobs = re.findall(r"```json\s*\n(.*?)```", content or "", re.DOTALL)
    if not blobs:
        blobs = re.findall(r"(\{.*\})", content or "", re.DOTALL)
    for blob in reversed(blobs):
        try:
            cand = json.loads(blob)
            if isinstance(cand, dict):
                return cand
        except (ValueError, TypeError):
            continue
    return {}


async def ppa_judge_node(state: dict) -> dict:
    """PPA JUDGE (codex): impartial FRD-coverage + area + Fmax + mappability verdict.

    Reads the full FRD requirement set, per-block area/FF budgets, and the
    measured sizing/memory results, then renders a pass|fail|escalate verdict.
    Writes ``.coresmith/microarch/ppa_judge.json`` and returns
    ``{ppa_verdict, status}``.
    """
    pr = _pr(state)
    blocks = state.get("blocks") or []
    target_mhz = _read_target_clock_mhz(pr)
    frd_vectors = _read_frd_vectors(pr)
    specs = _read_uarch_specs(pr, blocks)

    budgets: dict[str, dict] = {}
    try:
        from orchestrator.langgraph.ppa_check import (
            parse_area_budget,
            parse_ff_budget,
        )
        for b in blocks:
            budgets[b] = {
                "area_budget_um2": parse_area_budget(specs.get(b, "")),
                "flip_flop_budget": parse_ff_budget(specs.get(b, "")),
            }
    except Exception:  # noqa: BLE001
        budgets = {b: {} for b in blocks}

    out_path = _microarch_dir(pr) / "ppa_judge.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    _log(pr, f"ppa_judge: judging {len(blocks)} block(s) vs {len(frd_vectors)} "
             f"FRD vector(s) @ {target_mhz:.0f} MHz.")

    prompt = build_ppa_judge_prompt(
        pr, blocks, frd_vectors, budgets,
        state.get("size_results", {}), state.get("mem_results", {}),
        target_mhz, str(out_path),
    )
    llm = _make_llm()
    try:
        content = await llm.call(
            system=_load_ppa_judge_prompt(), prompt=prompt,
            run_name="PPA Judge",
        )
        if out_path.exists():
            raw = json.loads(out_path.read_text(encoding="utf-8"))
        else:
            raw = _parse_ppa_json(content)
        verdict = _normalize_ppa_verdict(raw)
    except Exception as exc:  # noqa: BLE001 - fail to escalate, never crash
        _log(pr, f"ppa_judge: agent error: {type(exc).__name__}: {exc}")
        verdict = _normalize_ppa_verdict({
            "verdict": "escalate",
            "violations": [{"kind": "judge_error", "block": "chip",
                            "detail": f"PPA judge failed: {exc}"}],
            "recommended_action": "ask_human",
        })
    try:
        out_path.write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    except OSError:
        pass

    status = "running" if verdict["passed"] else "failed"
    _log(pr, f"ppa_judge: verdict={verdict['verdict']} "
             f"action={verdict['recommended_action']} passed={verdict['passed']}")
    return {"ppa_verdict": verdict, "status": status}


async def diagnose_node(state: dict) -> dict:
    """AGENT 3 (codex): diagnose root cause per failing block -> feedback.

    Also emits a ``debug_action`` (``rebuild`` | ``ask_human``) from the failure
    buckets, and APPENDS structured MUST / MUST-NOT rules to each failing block's
    persistent constraints.json (disk-first memory, not just prose feedback).
    """
    pr = _pr(state)
    attempt = int(state.get("attempt", 0))
    golden_path, _ = _resolve_golden(pr)
    fails = _collect_failures(
        state.get("lint_errors", {}), state.get("verify_results", {}),
        state.get("size_results", {}),
    )
    ppa_verdict = state.get("ppa_verdict") or {}
    debug_action = _decide_debug_action(fails, ppa_verdict)
    _log(pr, f"diagnose: {len(fails)} failing block(s): {list(fails)} "
             f"-> debug_action={debug_action}")

    # Seed durable, structured constraints per failing block (survive all retries).
    for b, rec in fails.items():
        rules = _constraints_from_failure(b, rec)
        added = _append_block_constraints(pr, b, rules, attempt)
        if added:
            _log(pr, f"diagnose: +{added} constraint(s) for {b}")

    # B4 incremental: publish the failing set so build_models rebuilds ONLY these.
    failed_blocks = list(fails.keys())

    # A PPA-driven ask_human short-circuits the LLM diagnosis (nothing to rebuild).
    if debug_action == "ask_human":
        return {
            "feedback": state.get("feedback", "") or "",
            "debug_action": "ask_human",
            "failed_blocks": failed_blocks,
            "status": "running",
        }

    llm = _make_llm()
    prompt = build_diagnose_prompt(
        pr, state.get("lint_errors", {}), state.get("verify_results", {}),
        state.get("size_results", {}), golden_path,
    )
    try:
        feedback = await llm.call(
            system=DIAGNOSE_MODELS_SYSTEM, prompt=prompt, run_name="Diagnose Models",
        )
    except Exception as exc:  # noqa: BLE001
        feedback = (
            "diagnose agent failed; raw failures:\n"
            + json.dumps(fails, indent=2, default=str)
        )
        _log(pr, f"diagnose: agent error: {type(exc).__name__}: {exc}")
    return {"feedback": feedback, "debug_action": "rebuild",
            "failed_blocks": failed_blocks, "status": "running"}


def _decide_debug_action(fails: dict[str, dict], ppa_verdict: dict) -> str:
    """Bucket failures -> ``rebuild`` (fixable) or ``ask_human`` (impasse).

    PPA ``escalate`` / an unmappable ``reshape`` memory -> ask_human. Lint /
    verify / size (Fmax/area) failures -> rebuild.
    """
    if ppa_verdict:
        if ppa_verdict.get("verdict") == "escalate" or \
                ppa_verdict.get("recommended_action") == "ask_human":
            return "ask_human"
    # an unmappable memory (recommended reshape) is an architectural impasse
    for rec in fails.values():
        size = rec.get("size") if isinstance(rec, dict) else None
        if isinstance(size, dict):
            for mem in (size.get("memories") or []):
                if str(mem.get("recommended_impl")) == "reshape":
                    return "ask_human"
    return "rebuild"


def _constraints_from_failure(block: str, rec: dict) -> list[dict]:
    """Turn a per-block failure record into structured MUST / MUST-NOT rules."""
    rules: list[dict] = []
    lint = rec.get("lint")
    if lint:
        if "override" in str(lint):
            rules.append({
                "rule": "MUST NOT assign `.verilog_code`/`.vhdl_code`; the "
                        "@always_seq body is the only source of truth.",
                "kind": "must_not", "source": "lint",
            })
        elif "interface mismatch" in str(lint):
            rules.append({
                "rule": f"MUST expose every interface from block_diagram.json: "
                        f"{str(lint)[:200]}",
                "kind": "must", "source": "lint",
            })
        else:
            rules.append({
                "rule": f"MUST elaborate cleanly (toVerilog): {str(lint)[:200]}",
                "kind": "must", "source": "lint",
            })
    verify = rec.get("verify")
    if isinstance(verify, dict) and not verify.get("passed", False):
        fd = verify.get("first_divergence") or {}
        detail = fd.get("summary") or verify.get("detail") or ""
        rules.append({
            "rule": f"MUST match golden byte/value-exact; first divergence: "
                    f"{str(detail)[:200]}",
            "kind": "must", "source": "verify",
        })
    size = rec.get("size")
    if isinstance(size, dict) and not size.get("feasible", True):
        if size.get("infeasible_ops"):
            rules.append({
                "rule": "MUST pipeline/decompose the datapath so no single op "
                        f"exceeds the clock period (ops {size['infeasible_ops'][:6]}).",
                "kind": "must", "source": "size",
            })
        if size.get("area_budget_um2") is not None and \
                (size.get("total_area_um2") or 0) > (size.get("area_budget_um2") or 0):
            rules.append({
                "rule": f"MUST fit area budget {size['area_budget_um2']} um2 "
                        f"(currently {size.get('total_area_um2')} um2).",
                "kind": "must", "source": "size",
            })
    return rules


# ---------------------------------------------------------------------------
# size: coarse per-model sizing from declared arithmetic + characterized delays
# ---------------------------------------------------------------------------

_ARITH_TOKENS = (
    (re.compile(r"[^*]\*[^*]"), "mul"),   # single '*' (not '**')
    (re.compile(r"\+"), "add"),
    (re.compile(r"(?<![<>=!])-(?![>])"), "sub"),
    (re.compile(r"<<|>>"), "shift"),
    (re.compile(r"[<>]=?|==|!="), "cmp"),
)


# A rough per-FF std-cell area (um^2) for the register-area estimate. sky130
# HD DFF cells are ~5 um^2 each; used only to give a non-arithmetic block a real
# (non-zero) area from its registers + to price the datapath registers.
_FF_AREA_UM2 = float(os.environ.get("CORESMITH_MICROARCH_FF_AREA_UM2", "5.0"))

# Amaranth memory-array declaration pattern:
#   mem = Array(Signal(W) for _ in range(DEPTH))
_MEM_LIST_RE = re.compile(
    r"(?:Array\s*\(\s*)?Signal\(\s*(\d+)\s*(?:,|\))[^\n]*?"
    r"for\s+\w+\s+in\s+range\(\s*(\d+)\s*\)"
)


def _detect_memories(text: str) -> list[dict]:
    """Detect declared Amaranth Array-of-Signal memories with width/depth.

    AST-based so it resolves NAMED depth/width constants -- e.g.
    ``Y_DEPTH = 307200; mem = Array(Signal(8) for _ in range(Y_DEPTH))`` --
    which the old numeric-literal-only regex silently missed, pricing a 2.4 Mbit
    frame buffer at 0 um2. Falls back to the regex on a parse failure.
    Returns ``[{width, depth, ports}]`` (ports default 1rw at this coarse stage).
    """
    import ast as _ast

    def _regex_fallback() -> list[dict]:
        out = []
        for m in _MEM_LIST_RE.finditer(text or ""):
            w, d = int(m.group(1)), int(m.group(2))
            if w > 0 and d > 0:
                out.append({"width": w, "depth": d, "ports": "1rw"})
        return out

    try:
        tree = _ast.parse(text or "")
    except SyntaxError:
        return _regex_fallback()

    consts: dict[str, int] = {}

    def _v(node):
        if isinstance(node, _ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, _ast.Name) and node.id in consts:
            return consts[node.id]
        if isinstance(node, _ast.BinOp):
            lv, r = _v(node.left), _v(node.right)
            if lv is None or r is None:
                return None
            if isinstance(node.op, _ast.Mult):
                return lv * r
            if isinstance(node.op, _ast.Add):
                return lv + r
            if isinstance(node.op, _ast.Sub):
                return lv - r
            if isinstance(node.op, _ast.FloorDiv) and r:
                return lv // r
        return None

    for n in _ast.walk(tree):
        if (isinstance(n, _ast.Assign) and len(n.targets) == 1
                and isinstance(n.targets[0], _ast.Name)):
            val = _v(n.value)
            if val is not None:
                consts[n.targets[0].id] = val

    def _elt_width(elt):
        if (isinstance(elt, _ast.Call)
                and (getattr(elt.func, "id", None)
                     or getattr(elt.func, "attr", None)) == "Signal"
                and elt.args):
            shape = elt.args[0]
            if isinstance(shape, _ast.Call) and shape.args:
                # Signal(signed(W)) / Signal(unsigned(W))
                shape = shape.args[0]
            w = _v(shape)
            if w:
                return w
        for sub in _ast.walk(elt):
            if isinstance(sub, _ast.Subscript) and isinstance(sub.slice, _ast.Slice):
                w = _v(sub.slice.lower) if sub.slice.lower is not None else None
                if w:
                    return w
        return None

    mems: list[dict] = []
    for n in _ast.walk(tree):
        if isinstance(n, (_ast.ListComp, _ast.GeneratorExp)) and n.generators:
            gen = n.generators[0]
            if (isinstance(gen.iter, _ast.Call)
                    and isinstance(gen.iter.func, _ast.Name)
                    and gen.iter.func.id == "range" and gen.iter.args):
                depth = _v(gen.iter.args[-1])
                width = _elt_width(n.elt) if isinstance(n.elt, _ast.Call) else None
                if depth and depth > 1 and width and width > 0:
                    mems.append({"width": width, "depth": depth, "ports": "1rw"})
    return mems or _regex_fallback()


def _register_count(text: str, width: int) -> int:
    """Coarse register-bit count from Amaranth ``target.eq(...)`` assignments.

    Every distinct signal assigned with ``<sig>.eq(...)`` is conservatively a
    possible register output;
    approximate its width by the block's max declared Signal width. A block with
    zero arithmetic still carries these registers (so it is NOT free area).
    """
    targets = set(re.findall(r"(?:self\.)?(\w+)\s*\.eq\s*\(", text or ""))
    return len(targets) * max(1, width)


def _size_one_model(
    model_path: str, block_name: str, target_mhz: float, spec_text: str = "",
) -> dict:
    """Real per-block sizing: full datapath DFG + register + memory area vs budget.

    * DATAPATH: collect ALL arithmetic ops in the model (chained mul/add/sub/
      shift/cmp), build one DFG, and schedule it with
      ``pipeline_scheduler.schedule_dfg`` -> Fmax + pipeline depth. A single op
      exceeding the period is infeasible.
    * A block with NO datapath arithmetic is NOT auto-feasible: it still gets
      area from its registers + memories.
    * MEMORIES: each declared array is sized via
      ``mem_characterize.predict_mem(width, depth, ports, target_mhz)`` ->
      ``{recommended_impl, pred_area_um2, macro_feasible}``.
    * AREA: datapath+register+memory area is summed and compared to the block's
      ``area_budget_um2`` (``ppa_check.parse_area_budget``). Over budget -> fail.
    """
    from orchestrator.langgraph.pipeline_scheduler import (
        Node,
        pipeline_contract_text,
        schedule_dfg,
    )

    p = Path(model_path)
    if not p.exists():
        return {"feasible": False, "detail": f"model missing: {model_path}"}
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        return {"feasible": False, "detail": f"unreadable: {exc}"}

    period_ns = 1000.0 / max(1e-6, target_mhz)

    # Estimate operand widths from Signal(W) and Signal(signed(W)).
    widths = [
        int(w) for w in re.findall(
            r"Signal\(\s*(?:signed\s*\(\s*)?(\d+)", text
        )
    ]
    width = max(widths) if widths else 16

    # ---- DATAPATH DFG: collect ALL arithmetic ops (not just the worst line) ----
    ops: list[str] = []
    per_line_chains: list[list[str]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("#") or not line:
            continue
        if "=" not in line and "return" not in line:
            continue
        line_ops: list[str] = []
        for pat, op in _ARITH_TOKENS:
            line_ops.extend([op] * len(pat.findall(line)))
        if line_ops:
            per_line_chains.append(line_ops)
            ops.extend(line_ops)

    # Build a DFG where each source-line's ops form a data-dependent chain (a
    # realistic combinational cluster), all chains sharing the block's datapath.
    order = {"mul": 0, "add": 1, "sub": 2, "shift": 3, "cmp": 4}
    nodes: list[Node] = []
    edges: list[tuple[str, str]] = []
    nid = 0
    for chain in per_line_chains:
        chain = sorted(chain, key=lambda o: order.get(o, 9))
        prev: str | None = None
        for op in chain:
            this = f"n{nid}"
            nid += 1
            nodes.append(Node(this, op, width))
            if prev is not None:
                edges.append((prev, this))
            prev = this

    # ---- MEMORY sizing ----
    memories = _detect_memories(text)
    mem_area = 0.0
    mem_records: list[dict] = []
    for mem in memories:
        rec = {"width": mem["width"], "depth": mem["depth"], "ports": mem["ports"]}
        try:
            from orchestrator.langgraph import mem_characterize as _mc
            pred = _mc.predict_mem(
                mem["width"], mem["depth"], mem["ports"], target_mhz=target_mhz,
            )
            rec["recommended_impl"] = pred.get("recommended_impl")
            rec["pred_area_um2"] = pred.get("pred_area_um2")
            rec["macro_feasible"] = pred.get("macro_feasible")
        except Exception as exc:  # noqa: BLE001 - degrade to a bit-count estimate
            rec["recommended_impl"] = None
            rec["pred_area_um2"] = None
            rec["macro_feasible"] = None
            rec["error"] = f"{type(exc).__name__}: {exc}"
        area = rec.get("pred_area_um2")
        if not area:
            # fall back to a flop bit-area estimate so memory is never "free"
            area = mem["width"] * mem["depth"] * _FF_AREA_UM2
            rec["pred_area_um2"] = area
            rec["area_source"] = "flop_bit_estimate"
        mem_area += float(area)
        mem_records.append(rec)

    # ---- REGISTER area (so a zero-arithmetic block still has area) ----
    reg_bits = _register_count(text, width)
    reg_area = reg_bits * _FF_AREA_UM2

    # ---- schedule the datapath ----
    if nodes:
        try:
            sched = schedule_dfg(nodes, edges, period_ns)
        except Exception as exc:  # noqa: BLE001
            return {"feasible": False, "detail": f"schedule error: {exc}"}
        depth = sched.depth
        fmax = round(sched.fmax_mhz, 1)
        infeasible_ops = sched.infeasible
        dp_detail = pipeline_contract_text(sched, title=block_name)
        # datapath cell area ~ op-count-weighted; a coarse but non-zero proxy.
        datapath_area = len(nodes) * width * _FF_AREA_UM2
    else:
        depth = 1
        fmax = target_mhz
        infeasible_ops = []
        dp_detail = "no combinational arithmetic (control/FIFO/memory block)."
        datapath_area = 0.0

    total_area = datapath_area + reg_area + mem_area

    # ---- AREA BUDGET gate ----
    budget = None
    try:
        from orchestrator.langgraph.ppa_check import parse_area_budget
        budget = parse_area_budget(spec_text)
    except Exception:  # noqa: BLE001
        budget = None

    fmax_ok = not infeasible_ops
    area_ok = budget is None or total_area <= budget
    feasible = bool(fmax_ok and area_ok)

    reason_parts = []
    if not fmax_ok:
        reason_parts.append(
            f"Fmax infeasible: op(s) {infeasible_ops[:6]} exceed the "
            f"{period_ns:.2f} ns period at {target_mhz:.0f} MHz.")
    if not area_ok:
        reason_parts.append(
            f"area {total_area:,.0f} um2 exceeds budget {budget:,.0f} um2 "
            f"(datapath {datapath_area:,.0f} + regs {reg_area:,.0f} + "
            f"mem {mem_area:,.0f}).")

    return {
        "feasible": feasible,
        "depth": depth,
        "fmax_mhz": fmax,
        "target_mhz": target_mhz,
        "datapath_area_um2": round(datapath_area, 1),
        "reg_area_um2": round(reg_area, 1),
        "mem_area_um2": round(mem_area, 1),
        "total_area_um2": round(total_area, 1),
        "area_budget_um2": budget,
        "memories": mem_records,
        "infeasible_ops": infeasible_ops,
        "reason": " ".join(reason_parts),
        "detail": dp_detail,
    }


# ---------------------------------------------------------------------------
# verify result parsing (tolerant)
# ---------------------------------------------------------------------------

def _parse_verify_results(content: str, blocks: list[str]) -> dict[str, dict]:
    """Parse the verifier's JSON summary; default to a conservative per-block record.

    Tolerant: pulls the last ```json / { ... } object out of the agent text. A block
    absent from the parsed object is treated as FAILED (unverified == not proven).
    """
    parsed: dict = {}
    # Prefer a fenced ```json block; else the last balanced {...}.
    m = re.findall(r"```json\s*\n(.*?)```", content or "", re.DOTALL)
    blobs = list(m)
    if not blobs:
        blobs = re.findall(r"(\{.*\})", content or "", re.DOTALL)
    for blob in reversed(blobs):
        try:
            cand = json.loads(blob)
            if isinstance(cand, dict):
                parsed = cand
                break
        except (ValueError, TypeError):
            continue

    results: dict[str, dict] = {}
    for b in blocks:
        rec = parsed.get(b) if isinstance(parsed, dict) else None
        if isinstance(rec, dict):
            results[b] = {
                "passed": bool(rec.get("passed", False)),
                "first_divergence": _normalize_first_divergence(
                    rec.get("first_divergence")
                ),
                "detail": rec.get("detail"),
            }
        else:
            # A block absent from the JSON = FAILED (unverified == not proven).
            results[b] = {
                "passed": False,
                "first_divergence": _normalize_first_divergence(None),
                "detail": "no machine-readable verify result for this block",
            }
    return results


def _normalize_first_divergence(fd: Any) -> dict:
    """Coerce a verifier's ``first_divergence`` into the structured schema.

    Mirrors ``contract_audit_agent``'s tolerant-default schema:
    ``{summary, golden_observation, model_observation, vector}``. Accepts a dict
    (fills missing keys), a bare string (-> summary), or None (empty record).
    """
    schema = {
        "summary": "",
        "golden_observation": "",
        "model_observation": "",
        "vector": "",
    }
    if isinstance(fd, dict):
        out = dict(schema)
        for k in schema:
            if fd.get(k) is not None:
                out[k] = fd.get(k)
        # tolerate alternate key names the agent might emit
        if not out["model_observation"] and fd.get("got") is not None:
            out["model_observation"] = fd.get("got")
        if not out["golden_observation"] and fd.get("expected") is not None:
            out["golden_observation"] = fd.get("expected")
        if not out["summary"] and fd.get("detail") is not None:
            out["summary"] = str(fd.get("detail"))
        return out
    if isinstance(fd, str) and fd.strip():
        return {**schema, "summary": fd.strip()}
    return schema


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def _at_retry_limit(state: dict) -> bool:
    return int(state.get("attempt", 0)) >= int(
        state.get("max_attempts", DEFAULT_MAX_ATTEMPTS)
    )


def route_after_build(state: dict) -> str:
    """build -> lint, unless the build itself errored out (e.g. no blocks)."""
    if state.get("status") == "error":
        return END
    return "lint_models"


def route_after_stage(state: dict) -> str:
    """Generic gate: a failing stage short-circuits to diagnose (or END if spent).

    Used after lint, verify, and size. On failure, route to diagnose unless the
    retry budget is exhausted (then END with status=failed).
    """
    if state.get("status") == "error":
        return END
    if state.get("status") == "failed":
        if _at_retry_limit(state):
            return END
        return "diagnose"
    return "__continue__"  # replaced per-edge by _next_after helpers


def route_after_lint(state: dict) -> str:
    r = route_after_stage(state)
    return "verify_models" if r == "__continue__" else r


def route_after_verify(state: dict) -> str:
    r = route_after_stage(state)
    return "size" if r == "__continue__" else r


def route_after_size(state: dict) -> str:
    """size gate: all-pass -> ppa_judge; failure -> diagnose or END."""
    if state.get("status") == "error":
        return END
    if state.get("status") == "failed":
        return END if _at_retry_limit(state) else "diagnose"
    # sizing feasible -> hand to the impartial PPA judge
    return "ppa_judge"


def route_after_ppa_judge(state: dict) -> str:
    """PPA judge: pass -> END; escalate -> ask_human; fail -> diagnose or END."""
    if state.get("status") == "error":
        return END
    verdict = state.get("ppa_verdict") or {}
    if verdict.get("passed"):
        return END
    if verdict.get("verdict") == "escalate" or \
            verdict.get("recommended_action") == "ask_human":
        return "ask_human"
    # fail -> rebuild loop (unless retry budget spent)
    return END if _at_retry_limit(state) else "diagnose"


def route_after_diagnose(state: dict) -> str:
    """Route out of diagnose by the emitted ``debug_action``.

    ``rebuild`` -> build_models (retry with accumulated constraints);
    ``ask_human`` -> ask_human (architectural impasse / unmappable).
    """
    if state.get("debug_action") == "ask_human":
        return "ask_human"
    return "build_models"


# ---------------------------------------------------------------------------
# ask_human: human-in-the-loop escalation (LangGraph interrupt)
# ---------------------------------------------------------------------------

def ask_human_node(state: dict) -> dict:
    """Interrupt the graph for an architectural impasse (can't map to hardware).

    Payload names the offending block + reason and the supported human actions
    (relax_requirement, retry, abort). The outer agent resumes with an action.
    """
    from langgraph.types import interrupt

    pr = _pr(state)
    fails = _collect_failures(
        state.get("lint_errors", {}), state.get("verify_results", {}),
        state.get("size_results", {}),
    )
    verdict = state.get("ppa_verdict") or {}
    # pick the offending block: first fail, else the PPA violation's block
    block = next(iter(fails), "")
    reason = "microarchitecture cannot be mapped to hardware"
    if verdict:
        viols = verdict.get("violations") or []
        if viols and isinstance(viols[0], dict):
            block = block or str(viols[0].get("block", ""))
            reason = str(viols[0].get("detail") or reason)
        fd = verdict.get("first_divergence") or {}
        if fd.get("summary"):
            reason = fd["summary"]

    payload = {
        "type": "microarch_unmappable",
        "block": block,
        "reason": reason,
        "supported_actions": ["relax_requirement", "retry", "abort"],
        "ppa_verdict": verdict,
    }
    _log(pr, f"ask_human: escalating block={block!r} reason={reason[:120]!r}")

    response = interrupt(payload)
    action = (response or {}).get("action", "abort") if isinstance(
        response, dict) else "abort"

    if action == "retry":
        return {"human_response": response, "status": "running",
                "debug_action": "rebuild"}
    # relax_requirement / abort both terminate this experiment run
    return {"human_response": response, "status": "failed"}


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def _make_llm():
    """Build a ClaudeLLM (honours CORESMITH_LLM_PROVIDER=codex via the env)."""
    from orchestrator._timeouts import scaled
    from orchestrator.langchain.agents.coresmith_llm import ClaudeLLM, block_model
    return ClaudeLLM(
        model=block_model(),
        timeout=scaled(2700, env="CORESMITH_MICROARCH_EXP_TIMEOUT"),
    )


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_microarch_graph(checkpointer=None):
    """Build + compile the standalone microarchitecture (pass-1) experiment graph.

    Nodes: build_models -> lint_models -> verify_models -> size -> ppa_judge.
    ppa_judge: pass -> END, fail -> diagnose, escalate -> ask_human. Every gate
    short-circuits to diagnose; diagnose routes via route_after_diagnose
    (rebuild -> build_models, ask_human -> ask_human).
    """
    graph = StateGraph(MicroarchState)

    graph.add_node("build_models", build_models_node)
    graph.add_node("lint_models", lint_models_node)
    graph.add_node("verify_models", verify_models_node)
    graph.add_node("size", size_node)
    graph.add_node("ppa_judge", ppa_judge_node)
    graph.add_node("diagnose", diagnose_node)
    graph.add_node("ask_human", ask_human_node)

    graph.add_edge(START, "build_models")
    graph.add_conditional_edges(
        "build_models", route_after_build,
        {"lint_models": "lint_models", END: END},
    )
    graph.add_conditional_edges(
        "lint_models", route_after_lint,
        {"verify_models": "verify_models", "diagnose": "diagnose", END: END},
    )
    graph.add_conditional_edges(
        "verify_models", route_after_verify,
        {"size": "size", "diagnose": "diagnose", END: END},
    )
    graph.add_conditional_edges(
        "size", route_after_size,
        {"ppa_judge": "ppa_judge", "diagnose": "diagnose", END: END},
    )
    graph.add_conditional_edges(
        "ppa_judge", route_after_ppa_judge,
        {"diagnose": "diagnose", "ask_human": "ask_human", END: END},
    )
    # diagnose routes by failure class: rebuild -> build_models, else ask_human.
    graph.add_conditional_edges(
        "diagnose", route_after_diagnose,
        {"build_models": "build_models", "ask_human": "ask_human"},
    )
    # ask_human terminates the experiment (or the outer agent resumes with retry).
    graph.add_edge("ask_human", END)

    return graph.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def run_microarch_exp(
    project_root: str, max_attempts: int = RUN_MICROARCH_MAX_ATTEMPTS,
) -> dict:
    """Compile + run the standalone microarchitecture experiment graph.

    Returns the final graph state dict. Progress is logged to stdout and to
    ``<project_root>/.coresmith/microarch_exp.log``.

    Codex SESSION RESUME is enabled here by DEFAULT (``CORESMITH_CODEX_RESUME``
    is set to ``1`` unless the env explicitly disables it) so a rebuilt
    build_models agent continues its prior conversation across retries. The
    global ClaudeLLM default stays OFF -- only this runner opts in.
    """
    os.environ.setdefault("CORESMITH_PROJECT_ROOT", project_root)

    # Default-ON codex resume for the microarch experiment (more continuity
    # across the rebuild loop). An explicit CORESMITH_CODEX_RESUME wins -- set
    # it to 0/false/off to disable.
    if not os.environ.get("CORESMITH_CODEX_RESUME", "").strip():
        os.environ["CORESMITH_CODEX_RESUME"] = "1"

    _log(project_root, f"run_microarch_exp START (max_attempts={max_attempts}, "
                       f"codex_resume={os.environ.get('CORESMITH_CODEX_RESUME')})")

    # STARTUP PRECONDITION: golden<->spec consistency gate (default ON). If the
    # reference golden changed since the block specs were written but the specs
    # were NOT regenerated, the specs are STALE and no model can be byte-exact --
    # flag it UP FRONT instead of burning N failed build/verify attempts.
    if _env_flag("CORESMITH_GOLDEN_SPEC_GATE", default=True):
        gate = golden_spec_consistency_gate(project_root)
        if not gate["passed"]:
            _log(project_root,
                 f"golden_spec_consistency: FAIL -- {gate['reason']}")
            return {
                "project_root": project_root,
                "blocks": discover_blocks(project_root),
                "attempt": 0,
                "status": "failed",
                "golden_spec_consistency": gate,
                "feedback": gate["reason"],
            }
        _log(project_root,
             f"golden_spec_consistency: PASS -- {gate['reason']}")

    blocks = discover_blocks(project_root)
    _log(project_root, f"discovered {len(blocks)} block(s): {blocks}")

    # DISK-FIRST CONSTRAINT LOOP: reset each block's constraints.json to [] at
    # start so accumulated rules are per-run (they grow across retries within
    # this run and survive every rebuild).
    for b in blocks:
        _reset_block_constraints(project_root, b)

    graph = build_microarch_graph()
    init: MicroarchState = {
        "project_root": project_root,
        "blocks": blocks,
        "attempt": 0,
        "max_attempts": max_attempts,
        "feedback": "",
        "lint_errors": {},
        "verify_results": {},
        "size_results": {},
        "mem_results": {},
        "ppa_verdict": {},
        "debug_action": "",
        "build_session_id": "",
        "build_session_ids": {},
        "build_cluster_session_ids": {},
        "status": "running",
    }
    # Allow more supersteps than attempts (6 nodes/attempt + slack).
    config = {"recursion_limit": max(30, max_attempts * 9)}
    final = await graph.ainvoke(init, config=config)

    # Final status normalisation: passed only when every gate AND the PPA judge
    # are clean.
    status = final.get("status", "running")
    lint_ok = not any(final.get("lint_errors", {}).values())
    verify_ok = all(
        r.get("passed", False) for r in final.get("verify_results", {}).values()
    ) if final.get("verify_results") else False
    size_ok = all(
        r.get("feasible", True) for r in final.get("size_results", {}).values()
    ) if final.get("size_results") else False
    ppa_ok = bool((final.get("ppa_verdict") or {}).get("passed"))
    if status not in ("error",):
        if lint_ok and verify_ok and size_ok and ppa_ok and blocks:
            final["status"] = "passed"
        elif status == "running":
            final["status"] = "failed"

    _log(project_root, f"run_microarch_exp END status={final.get('status')} "
                       f"attempts={final.get('attempt')}")
    return dict(final)


def _main(argv: list[str]) -> int:
    import asyncio

    if len(argv) < 2:
        print("usage: python -m orchestrator.langgraph.microarch_exp "
              "<project_root> [max_attempts]", flush=True)
        return 2
    project_root = argv[1]
    max_attempts = int(argv[2]) if len(argv) > 2 else RUN_MICROARCH_MAX_ATTEMPTS
    logging.basicConfig(level=logging.INFO)
    final = asyncio.run(run_microarch_exp(project_root, max_attempts=max_attempts))
    print(json.dumps({
        "status": final.get("status"),
        "attempt": final.get("attempt"),
        "blocks": final.get("blocks"),
        "lint_errors": final.get("lint_errors"),
        "verify_results": final.get("verify_results"),
        "size_results": {
            b: {k: v for k, v in r.items() if k != "detail"}
            for b, r in (final.get("size_results") or {}).items()
        },
    }, indent=2, default=str), flush=True)
    return 0 if final.get("status") == "passed" else 1


if __name__ == "__main__":  # pragma: no cover
    import sys
    raise SystemExit(_main(sys.argv))
