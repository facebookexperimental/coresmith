# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Deterministic cross-artifact quantity consistency.

This is the *deterministic half* of the ``cross_artifact_consistency``
constraint (the LLM half lives in the constraint catalog in
``constraints.py``).  It answers one question, arithmetically:

    Do two different architecture artifacts state two DIFFERENT values for
    the same named physical quantity?

That is the numeric root-cause class behind the arch-phase contradiction
set this gate was built for: an FRD that still said ``12.5 MHz`` /
``50 Mbit/s`` / ``40 ns`` phases while the ERS said ``6.25 MHz`` /
``25 Mbit/s`` / ``80 ns``.  Each stale reference was caught one-at-a-time
by a *downstream* gate, days later, at the cost of a full re-spec.

Design rules (in priority order):

1. **Never guess.**  A quantity that cannot be parsed *confidently* is
   SKIPPED and recorded as a note -- never reported as a violation.  The
   skip classes are: an ambiguous unit (``KB``/``MB``: decimal or binary?),
   an approximate value (``~30 ns``), a range (``4-8 ns``), and -- the big
   one -- a value the document never NAMES adjacently.
2. **Cross-artifact only.**  Two mentions are compared only when they come
   from two DIFFERENT artifacts.  Intra-document contradictions are the LLM
   half's job -- they are usually semantic, not arithmetic.
3. **Named comparison, not co-occurrence.**  Two mentions are comparable
   only when the document writes the quantity's NAME immediately next to the
   number (``tHIGH>=80 ns``, ``SCK <=6.25 MHz``, ``12.5 MHz QSPI clock``) and
   the two names are identical.  Binding a number to identifiers merely
   *nearby* was tried and measured: on 58 real architecture runs it flagged
   48 of them, because one clause routinely mentions the interface clock, the
   core clock and a latency together.  See ``_resolve_name``.
4. **Dimension-matched.**  Only mentions with the same physical dimension
   (frequency / time / data rate / data size) are compared, after
   normalizing to that dimension's base unit -- so ``9 KiB`` and
   ``73728 bits`` agree, and ``6.25 MHz`` and ``6250 kHz`` agree.

Everything else (semantic scheduling statements, prose contradictions,
"rise-scheduled here / fall-scheduled there") is out of scope here and is
owned by the LLM half.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------
#
# Every accepted unit maps to (dimension, factor-to-base).  Base units:
#   frequency -> Hz, time -> ns, data_rate -> bit/s, data_size -> bit.
# Units whose base is genuinely ambiguous (KB: 1000 or 1024 bytes?) are
# listed in _AMBIGUOUS_UNITS and produce a NOTE, never a comparison.

_UNITS: dict[str, tuple[str, float]] = {
    # frequency
    "hz": ("frequency", 1.0),
    "khz": ("frequency", 1e3),
    "mhz": ("frequency", 1e6),
    "ghz": ("frequency", 1e9),
    # time
    "ps": ("time", 1e-3),
    "ns": ("time", 1.0),
    "us": ("time", 1e3),
    "µs": ("time", 1e3),
    "μs": ("time", 1e3),
    "ms": ("time", 1e6),
    "s": ("time", 1e9),
    "sec": ("time", 1e9),
    "secs": ("time", 1e9),
    "second": ("time", 1e9),
    "seconds": ("time", 1e9),
    # data rate (bit-based only -- byte-based rates are ambiguous below)
    "bps": ("data_rate", 1.0),
    "bit/s": ("data_rate", 1.0),
    "bits/s": ("data_rate", 1.0),
    "kbps": ("data_rate", 1e3),
    "kbit/s": ("data_rate", 1e3),
    "kbits/s": ("data_rate", 1e3),
    "kb/s": ("data_rate", 1e3),
    "mbps": ("data_rate", 1e6),
    "mbit/s": ("data_rate", 1e6),
    "mbits/s": ("data_rate", 1e6),
    "mb/s": ("data_rate", 1e6),
    "gbps": ("data_rate", 1e9),
    "gbit/s": ("data_rate", 1e9),
    "gbits/s": ("data_rate", 1e9),
    "gb/s": ("data_rate", 1e9),
    # data size (binary prefixes only -- KB/MB/GB are ambiguous below)
    "bit": ("data_size", 1.0),
    "bits": ("data_size", 1.0),
    "byte": ("data_size", 8.0),
    "bytes": ("data_size", 8.0),
    "kbit": ("data_size", 1e3),
    "kbits": ("data_size", 1e3),
    "mbit": ("data_size", 1e6),
    "mbits": ("data_size", 1e6),
    "kib": ("data_size", 8.0 * 1024),
    "mib": ("data_size", 8.0 * 1024 * 1024),
    "gib": ("data_size", 8.0 * 1024 * 1024 * 1024),
}

# Recognized as "this is a quantity" but deliberately NOT normalized: the
# base is ambiguous (decimal vs binary, bits vs bytes).  Noted, never flagged.
_AMBIGUOUS_UNITS = frozenset({
    "kb", "mb", "gb", "tb",
    "kbyte", "kbytes", "mbyte", "mbytes", "gbyte", "gbytes",
    "b/s", "kb/sec", "mb/sec", "byte/s", "bytes/s",
})

# Longest-first alternation so "mbit/s" wins over "mbit", "khz" over "hz".
_UNIT_ALTERNATION = "|".join(
    re.escape(u) for u in sorted(
        set(_UNITS) | set(_AMBIGUOUS_UNITS), key=len, reverse=True,
    )
)

# number (optional thousands separators / decimals), optional symbolic
# comparator, optional hyphen or space before the unit.
_QUANTITY_RE = re.compile(
    r"(?P<sym>(?:<=|>=|=<|=>|<|>|≤|≥)\s*)?"
    r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?P<sep>[\s\u00a0-])?"
    r"(?P<unit>" + _UNIT_ALTERNATION + r")"
    r"(?![A-Za-z0-9_/])",
    re.IGNORECASE,
)

# Word comparators, searched in the text immediately preceding a mention.
_WORD_COMPARATORS: tuple[tuple[str, str, bool], ...] = (
    # (phrase, comparator, strict)
    ("no more than", "le", False),
    ("no faster than", "le", False),
    ("at most", "le", False),
    ("up to", "le", False),
    ("maximum of", "le", False),
    ("max of", "le", False),
    ("not exceed", "le", False),
    ("less than or equal to", "le", False),
    ("no less than", "ge", False),
    ("no slower than", "ge", False),
    ("at least", "ge", False),
    ("minimum of", "ge", False),
    ("min of", "ge", False),
    ("greater than or equal to", "ge", False),
    ("exactly", "eq", False),
)
# Deliberately NOT comparators: "over", "under", "above", "below", "less
# than", "greater than", "more than". English uses them as prepositions at
# least as often as bounds ("Tests shall cover all SCK launch offsets over one
# 20 ns core period"), and a misread comparator turns a passing mention into a
# fabricated bound. Left as plain point values.

_APPROX_MARKERS: tuple[str, ...] = (
    "~", "≈", "∼", "about ", "approximately ", "approx", "roughly ",
    "circa ", "nominally ", "order of ", "around ",
)

# A sentence that is an explicit anti-pattern warning is a FORBIDDEN pattern
# being described, not a claim being made.  Same rule the constraint subagent
# prompt already applies to the requirements doc; applied here to every source.
_NEGATION_MARKERS: tuple[str, ...] = (
    "do not ", "don't ", "must not ", "shall not ", "should not ", "never ",
    "incorrect", "avoid ", "rather than", "instead of", "not from",
    "no longer", "was previously", "superseded",
)

# Generic tokens that must never become a comparison anchor: they appear in
# every design and would link two unrelated quantities.
_ANCHOR_STOPWORDS = frozenset({
    "the", "and", "for", "with", "from", "into", "per", "any", "all", "one",
    "two", "new", "old", "top", "sub", "aux", "gen", "tmp", "val", "var",
    "data", "out", "outp", "output", "outputs", "in", "inp", "input", "inputs",
    "req", "request", "resp", "response", "bus", "port", "ports", "block",
    "blocks", "core", "chip", "clk", "clock", "rst", "reset", "en", "enable",
    "sig", "signal", "wire", "reg", "mem", "memory", "addr", "address",
    "write", "writes", "read", "reads", "max", "min", "word", "words",
    "byte", "bytes", "bit", "bits", "cnt", "count", "idx", "index", "num",
    "level", "levels", "state", "states", "event", "events", "mode", "cfg",
    "config", "ctrl", "control", "status", "start", "stop", "done", "error",
    "fault", "sync", "async", "first", "last", "next", "prev", "value",
    "values", "size", "sizes", "width", "widths", "depth", "spec", "specs",
    "unit", "units", "time", "rate", "rates", "phase", "phases", "edge",
    "edges", "host", "user", "wrapper", "interface", "interfaces", "if",
    "io", "pin", "pins", "pad", "pads", "gpio", "sel", "ack", "valid",
    "ready", "strobe", "src", "dst", "dest", "id", "ids", "map", "maps",
    "type", "types", "name", "names", "info", "misc", "flag", "flags",
})

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Quantity:
    """One confidently-parsed numeric quantity found in one artifact."""

    artifact: str          # "frd" | "ers" | "block_diagram" | "interface_contracts" | ...
    location: str          # "line 122" | "contracts[7].rate_description"
    sentence: str          # the sentence the mention was read from
    dimension: str         # "frequency" | "time" | "data_rate" | "data_size"
    unit: str              # as written
    raw_value: float       # as written
    value: float           # normalized to the dimension base unit
    comparator: str        # "eq" | "le" | "ge"
    strict: bool           # True for < / > (as opposed to <= / >=)
    name: str              # the adjacent identifier that names this quantity

    def claim(self) -> str:
        """The claim as written, e.g. ``<= 6.25 MHz``."""
        cmp_txt = {"eq": "", "le": "< " if self.strict else "<= ",
                   "ge": "> " if self.strict else ">= "}[self.comparator]
        return f"{cmp_txt}{self.raw_value:g} {self.unit}"


@dataclass
class ArtifactSource:
    """One artifact to scan: either markdown/prose text or a JSON document."""

    artifact: str
    kind: str                       # "text" | "json"
    text: str = ""
    payload: Any = None
    label: str = ""                 # human-facing file label, e.g. "arch/ers_spec.md"

    def display(self) -> str:
        return self.label or self.artifact


@dataclass
class ScanResult:
    violations: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    quantities: list[Quantity] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Anchor vocabulary (STRUCTURED artifacts only -- never prose)
# ---------------------------------------------------------------------------

def _add_identifier(vocab: set[str], raw: Any) -> None:
    if not isinstance(raw, str):
        return
    ident = raw.strip().lower()
    if not ident:
        return
    parts = [p for p in re.split(r"[^a-z0-9]+", ident) if p]
    if not parts:
        return
    joined = "_".join(parts)
    if len(parts) > 1 and joined not in _ANCHOR_STOPWORDS:
        vocab.add(joined)
    for p in parts:
        if len(p) >= 3 and p not in _ANCHOR_STOPWORDS and not p.isdigit():
            vocab.add(p)


def build_anchor_vocabulary(
    block_diagram: dict | None = None,
    interface_contracts: dict | None = None,
    memory_map: dict | None = None,
    clock_tree: dict | None = None,
    register_spec: dict | None = None,
) -> frozenset[str]:
    """Harvest the design's own identifiers from the STRUCTURED artifacts.

    A token in here is accepted as a quantity NAME even when it does not look
    like an identifier (see :func:`_is_named_quantity`), because a structured
    artifact declared it.  Generic words are filtered out, so a phrase that
    exists only in generated prose cannot become a name by itself.
    """
    vocab: set[str] = set()

    bd = block_diagram or {}
    for b in bd.get("blocks") or []:
        if not isinstance(b, dict):
            continue
        _add_identifier(vocab, b.get("name"))
        ifaces = b.get("interfaces")
        if isinstance(ifaces, dict):
            for k in ifaces:
                _add_identifier(vocab, k)
        elif isinstance(ifaces, list):
            for it in ifaces:
                _add_identifier(vocab, it.get("name") if isinstance(it, dict) else it)
    for c in (bd.get("connections") or []) + (bd.get("edges") or []):
        if not isinstance(c, dict):
            continue
        for key in ("interface", "from", "to", "from_block", "to_block",
                    "from_port", "to_port", "bus_name"):
            _add_identifier(vocab, c.get(key))

    ic = interface_contracts or {}
    for c in ic.get("contracts") or []:
        if not isinstance(c, dict):
            continue
        for key in ("edge_id", "producer_block", "producer_port",
                    "consumer_block", "consumer_port"):
            _add_identifier(vocab, c.get(key))
        for group in ("fields", "sideband_signals"):
            for f in c.get(group) or []:
                if isinstance(f, dict):
                    _add_identifier(vocab, f.get("name"))

    def _unwrap(d: Any) -> dict:
        if not isinstance(d, dict):
            return {}
        inner = d.get("result")
        return inner if isinstance(inner, dict) else d

    for p in _unwrap(memory_map).get("peripherals") or []:
        if isinstance(p, dict):
            _add_identifier(vocab, p.get("name"))
    for d in _unwrap(clock_tree).get("domains") or []:
        if isinstance(d, dict):
            _add_identifier(vocab, d.get("name"))
    for r in _unwrap(register_spec).get("register_blocks") or []:
        if isinstance(r, dict):
            _add_identifier(vocab, r.get("name"))

    return frozenset(vocab)


# ---------------------------------------------------------------------------
# Naming a quantity
# ---------------------------------------------------------------------------
#
# The hardest part of this check is deciding WHICH quantity a number is. An
# earlier revision bound a number to every design identifier within a
# clause-sized window. Measured against 58 real architecture runs that flagged
# 48 of them -- because a clause routinely mentions the interface clock and the
# core clock and a latency in the same breath, and co-occurrence cannot tell
# them apart. Co-occurrence is guessing, and guessing is forbidden here.
#
# So a quantity is only judged when the document NAMES it adjacently, in one of
# the two forms specs actually use:
#
#     <NAME> [`'"(=:] <cmp> <value> <unit>      e.g.  tHIGH>=80 ns
#                                                     SCK `<=6.25 MHz`
#                                                     qspi_sck=6.25 MHz
#     <value> <unit> <NAME>                     e.g.  12.5 MHz QSPI clock
#                                                     50 MHz wb_clk_i domain
#
# Anything else ("the raw active-direction data rate of 50 Mbit/s") is left
# unnamed and skipped with a note. That loses real coverage on purpose: an
# operating point that is genuinely stale restates itself in a named form
# somewhere in the same document set.

# Left-side filler that may sit between the name and the number.
_LEFT_FILLER_RE = re.compile(r"[\s`'\"“”\(\[=:~<>≤≥\-]*$")
# Right-side filler between the unit and a trailing name. A comma, period or
# semicolon ENDS the binding -- "6.25 MHz, four bits per SCK" does not name the
# frequency "four".
_RIGHT_FILLER_RE = re.compile(r"^[\s`'\"”\)\]]*")
_TRAILING_IDENT_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)$")
_LEADING_IDENT_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)")

# Function words that can never name a quantity, whatever their shape.
# Disjoint from _ANCHOR_STOPWORDS below it, which is unioned in.
_NAME_STOPWORDS = frozenset({
    "a", "an", "is", "are", "was", "were", "be", "been", "being", "at",
    "to", "of", "on", "or", "than", "then", "each", "every", "no", "not",
    "up", "about", "exactly", "least", "most", "supported", "provide",
    "provides", "provided", "sustain", "shall", "must", "should", "will",
    "may", "can", "run", "runs", "capture", "test", "tests", "require",
    "requires", "required", "using", "use", "uses", "used", "through",
    "during", "while", "when", "its", "it", "this", "that", "these",
    "those", "both", "only", "also", "still", "raw", "total", "full",
    "half", "high", "low", "second", "third", "same", "other", "such",
    "more", "less", "above", "below", "within", "without", "across",
    "between", "over", "under", "onto", "via", "but", "so", "as", "has",
    "have", "had", "does", "do", "did", "give", "gives", "giving",
    "target", "targets", "nominal", "typical", "worst", "best", "case",
    "measured", "assume", "assumed", "assumes", "operate", "operates",
    "operating", "supports", "support", "roughly",
}) | _ANCHOR_STOPWORDS


def _is_named_quantity(token: str, vocab: frozenset[str]) -> bool:
    """Is ``token`` shaped like the NAME of a design quantity?

    Accepts (a) anything a structured artifact actually declares, (b) a
    snake_case identifier, and (c) a token with an interior capital -- the
    ``SCK`` / ``tHIGH`` / ``IRQ`` convention. A merely sentence-initial capital
    (``The``, ``Compute``) is NOT interior, so ordinary prose cannot pose as a
    signal name.
    """
    low = token.lower()
    if low in _NAME_STOPWORDS:
        return False
    if low in vocab:
        return True
    if "_" in token:
        return True
    return any(c.isupper() for c in token[1:])


def _resolve_name(
    sentence: str, start: int, end: int, vocab: frozenset[str],
) -> str:
    """Resolve the adjacent name of the quantity at ``sentence[start:end]``."""
    left = _LEFT_FILLER_RE.sub("", sentence[:start])
    m = _TRAILING_IDENT_RE.search(left)
    if m and _is_named_quantity(m.group(1), vocab):
        return m.group(1).lower()
    right = _RIGHT_FILLER_RE.sub("", sentence[end:])
    m = _LEADING_IDENT_RE.match(right)
    if m and _is_named_quantity(m.group(1), vocab):
        return m.group(1).lower()
    return ""


# ---------------------------------------------------------------------------
# Quantity extraction
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.;:!?])\s+(?=[A-Za-z(\"'“])")


def _split_sentences(line: str) -> list[str]:
    """Split a line into sentences without breaking decimal numbers.

    The lookahead requires a letter/quote/paren after the boundary, so
    ``6.25 MHz`` and ``0.5`` survive intact.
    """
    line = line.strip()
    if not line:
        return []
    return [s for s in (p.strip() for p in _SENTENCE_SPLIT_RE.split(line)) if s]


def _is_negated(sentence: str) -> bool:
    low = sentence.lower()
    if any(m in low for m in _NEGATION_MARKERS):
        return True
    # Emphatic upper-case NOT (documents use it for corrections).
    return bool(re.search(r"\bNOT\b", sentence))


def _leading_comparator(prefix: str) -> tuple[str, bool] | None:
    """Word comparator immediately preceding a mention (longest match wins)."""
    low = prefix.lower()
    best: tuple[int, str, bool] | None = None
    for phrase, cmp_, strict in _WORD_COMPARATORS:
        idx = low.rfind(phrase)
        if idx == -1:
            continue
        tail = low[idx + len(phrase):]
        # Only a short, number-free gap counts as "immediately preceding".
        if len(tail) > 24 or re.search(r"\d", tail):
            continue
        if best is None or idx > best[0]:
            best = (idx, cmp_, strict)
    if best is None:
        return None
    return best[1], best[2]


def _is_approximate(prefix: str) -> bool:
    low = prefix.lower()
    tail = low[-18:]
    return any(m in tail for m in _APPROX_MARKERS)


def _is_range_member(sentence: str, start: int) -> bool:
    """``4-8 ns`` / ``6.25 to 12.5 MHz``: the value is one end of a range."""
    prefix = sentence[max(0, start - 12):start]
    return bool(re.search(r"\d\s*(?:-|–|to|through|\.\.)\s*$", prefix))


# ``at 6.25, 5, 3.125, and 1 MHz SCK`` -- the trailing member carries the unit
# and the name, but it is one option in an enumeration, not THE value of the
# quantity. Comparing it against another document's single figure is a
# guaranteed false positive.
_LIST_MEMBER_RE = re.compile(
    r"\d[\d.,]*\s*(?:[A-Za-z/]{1,6})?\s*(?:,|,?\s+(?:and|or))\s*$"
)


def _is_list_member(sentence: str, start: int) -> bool:
    return bool(_LIST_MEMBER_RE.search(sentence[max(0, start - 40):start]))


def extract_quantities(
    text: str,
    artifact: str,
    location: str,
    vocab: frozenset[str],
    notes: list[str],
) -> list[Quantity]:
    """Extract every confidently-parsed quantity from one blob of text.

    Anything that is recognizable as a quantity but not confidently
    comparable is appended to ``notes`` and dropped.
    """
    out: list[Quantity] = []
    if not text:
        return out

    for sentence in _split_sentences(text):
        if _is_negated(sentence):
            continue
        for m in _QUANTITY_RE.finditer(sentence):
            unit_raw = m.group("unit")
            unit = unit_raw.lower()
            num_txt = m.group("num").replace(",", "")
            where = f"{artifact}:{location}"

            if unit in _AMBIGUOUS_UNITS:
                notes.append(
                    f"{where}: skipped '{num_txt} {unit_raw}' -- ambiguous unit "
                    f"(decimal vs binary / bit vs byte base is not determinable)."
                )
                continue
            prefix = sentence[:m.start()]
            if _is_approximate(prefix):
                notes.append(
                    f"{where}: skipped '{num_txt} {unit_raw}' -- approximate "
                    f"value (not a specification claim)."
                )
                continue
            if _is_range_member(sentence, m.start()):
                notes.append(
                    f"{where}: skipped '{num_txt} {unit_raw}' -- one end of a "
                    f"range, not a single named value."
                )
                continue
            if _is_list_member(sentence, m.start()):
                notes.append(
                    f"{where}: skipped '{num_txt} {unit_raw}' -- the last member "
                    f"of an enumeration, not THE value of the quantity."
                )
                continue
            if re.search(r"[A-Za-z0-9_]$", prefix) and not m.group("sym"):
                notes.append(
                    f"{where}: skipped '{num_txt} {unit_raw}' -- embedded in an "
                    f"identifier."
                )
                continue
            # `N-bit` / `N-byte` is a type width, not a budget -- silently
            # out of scope rather than noted, or every ledger in the bundle
            # would emit a note.
            if m.group("sep") == "-" and unit in (
                    "bit", "bits", "byte", "bytes"):
                continue
            name = _resolve_name(sentence, m.start(), m.end(), vocab)
            if not name:
                notes.append(
                    f"{where}: skipped '{num_txt} {unit_raw}' -- no adjacent "
                    f"identifier names this quantity; refusing to guess which "
                    f"parameter it is."
                )
                continue

            dimension, factor = _UNITS[unit]
            try:
                raw_value = float(num_txt)
            except ValueError:  # pragma: no cover - regex guarantees numeric
                continue

            comparator, strict = "eq", False
            sym = (m.group("sym") or "").strip()
            if sym in ("<=", "=<", "≤"):
                comparator, strict = "le", False
            elif sym == "<":
                comparator, strict = "le", True
            elif sym in (">=", "=>", "≥"):
                comparator, strict = "ge", False
            elif sym == ">":
                comparator, strict = "ge", True
            else:
                word = _leading_comparator(prefix)
                if word:
                    comparator, strict = word

            out.append(Quantity(
                artifact=artifact,
                location=location,
                sentence=sentence[:400],
                dimension=dimension,
                unit=unit_raw,
                raw_value=raw_value,
                value=raw_value * factor,
                comparator=comparator,
                strict=strict,
                name=name,
            ))
    return out


def scan_source(
    source: ArtifactSource, vocab: frozenset[str], notes: list[str],
) -> list[Quantity]:
    """Extract quantities from one artifact, tagging each with its location."""
    quantities: list[Quantity] = []
    if source.kind == "text":
        for lineno, line in enumerate(source.text.splitlines(), start=1):
            quantities.extend(extract_quantities(
                line, source.artifact, f"line {lineno}", vocab, notes,
            ))
        return quantities

    # JSON: scan string leaves only.  Numeric JSON fields are structured
    # declarations with their own dedicated gates -- mixing them in here
    # would compare a bit-width int against a prose sentence.
    def _walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                _walk(v, f"{path}.{k}" if path else str(k))
        elif isinstance(node, list):
            for i, v in enumerate(node):
                _walk(v, f"{path}[{i}]")
        elif isinstance(node, str):
            quantities.extend(extract_quantities(
                node, source.artifact, path or "(root)", vocab, notes,
            ))

    _walk(source.payload, "")
    return quantities


# ---------------------------------------------------------------------------
# Conflict detection
# ---------------------------------------------------------------------------

_REL_TOL = 1e-6

# Dimensions where a BARE value ("at 6.25 MHz", "25 Mbit/s") reliably names
# the operating point of the identifier next to it: a design has one clock per
# domain and one line rate per port, so two different bare figures anchored on
# the same identifier really are a disagreement.
#
# Durations and sizes are NOT in this set. The same clause neighbourhood
# routinely carries a latency, a period, a budget, a buffer depth and a field
# width, so two bare figures sharing an anchor are usually two DIFFERENT
# quantities, not one quantity stated twice ("a 20 ns core period" next to an
# "SCK phase >= 80 ns"; "5 bits" of drive group next to a "4096 bytes"
# window). Measured on real architecture artifacts, bare time/size comparison
# was pure noise. For those dimensions only like-kind DECLARED LIMITS
# (cap-vs-cap, floor-vs-floor, empty interval) are judged -- a limit is an
# explicit claim about one named quantity, so it does not suffer the ambiguity.
_POINT_COMPARABLE_DIMENSIONS = frozenset({"frequency", "data_rate"})


def _close(a: float, b: float) -> bool:
    scale = max(abs(a), abs(b), 1.0)
    return abs(a - b) <= _REL_TOL * scale


def _conflict(a: Quantity, b: Quantity) -> str:
    """Return a human-readable conflict reason, or "" when consistent.

    Only *like-for-like* comparisons are judged: point vs point, cap vs cap,
    floor vs floor, and an empty cap/floor interval. A POINT-vs-BOUND
    comparison is deliberately NOT judged -- a value merely mentioned in
    passing frequently sits outside an unrelated declared limit that happens
    to carry the same name, and that produced pure noise on real artifacts.
    A genuinely stale operating point restates itself as a point AND as a
    bound somewhere in the same document set, so the like-for-like pair
    still catches it.
    """
    ca, cb = a.comparator, b.comparator
    if ca == "eq" and cb == "eq":
        if a.dimension not in _POINT_COMPARABLE_DIMENSIONS:
            return ""
        return "" if _close(a.value, b.value) else "two different stated values"
    if ca == cb == "le":
        return "" if _close(a.value, b.value) else "two different upper limits"
    if ca == cb == "ge":
        return "" if _close(a.value, b.value) else "two different lower limits"
    if ca == "eq" or cb == "eq":
        return ""
    # le vs ge -- only a conflict when the permitted interval is empty.
    lo = a if ca == "ge" else b
    hi = b if ca == "ge" else a
    if hi.value < lo.value:
        return "upper limit is below the stated lower limit"
    return ""


_DIMENSION_LABEL = {
    "frequency": "frequency",
    "time": "time/duration",
    "data_rate": "data rate",
    "data_size": "data size",
}


def find_quantity_conflicts(
    quantities: Iterable[Quantity], max_findings: int = 25,
) -> list[dict]:
    """Cross-artifact conflicts, one violation per (name, dimension) group.

    Only pairs from two DIFFERENT artifacts are compared.  Within a group the
    first conflicting pair is reported (with the other members as evidence)
    so a stale value repeated nine times produces one actionable finding, not
    nine.
    """
    grouped: dict[tuple[str, str], list[Quantity]] = {}
    for q in quantities:
        grouped.setdefault((q.name, q.dimension), []).append(q)

    def _first_conflicting_pair(
        members: list[Quantity],
    ) -> tuple[Quantity, Quantity, str] | None:
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                if a.artifact == b.artifact:
                    continue
                reason = _conflict(a, b)
                if reason:
                    return a, b, reason
        return None

    findings: list[dict] = []
    for (name, dimension), members in sorted(grouped.items()):
        if len({q.artifact for q in members}) < 2:
            continue
        pair = _first_conflicting_pair(members)
        if pair is None:
            continue
        a, b, reason = pair
        findings.append({
            "name": name,
            "dimension": dimension,
            "reason": reason,
            "a": a,
            "b": b,
            "related": [q for q in members if q is not a and q is not b][:6],
        })
    return findings[:max_findings]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def check_cross_artifact_quantities(
    sources: list[ArtifactSource],
    block_diagram: dict | None = None,
    interface_contracts: dict | None = None,
    memory_map: dict | None = None,
    clock_tree: dict | None = None,
    register_spec: dict | None = None,
) -> ScanResult:
    """Deterministic half of ``cross_artifact_consistency``.

    Returns a :class:`ScanResult` carrying structural violations, the skip
    notes (quantities deliberately NOT judged), and every parsed quantity
    (useful for tests and for the trace span).
    """
    vocab = build_anchor_vocabulary(
        block_diagram=block_diagram,
        interface_contracts=interface_contracts,
        memory_map=memory_map,
        clock_tree=clock_tree,
        register_spec=register_spec,
    )
    result = ScanResult()
    if not vocab or len(sources) < 2:
        if not vocab:
            result.notes.append(
                "cross_artifact_consistency: no structured design identifiers "
                "available; the deterministic quantity check did not run."
            )
        return result

    labels = {s.artifact: s.display() for s in sources}
    for source in sources:
        result.quantities.extend(scan_source(source, vocab, result.notes))

    for f in find_quantity_conflicts(result.quantities):
        a: Quantity = f["a"]
        b: Quantity = f["b"]
        a_label = labels.get(a.artifact, a.artifact)
        b_label = labels.get(b.artifact, b.artifact)
        related = "".join(
            f"\n    also: {labels.get(q.artifact, q.artifact)}:{q.location} "
            f"→ {q.raw_value:g} {q.unit}"
            for q in f["related"]
        )
        result.violations.append({
            "violation": (
                f"CROSS-ARTIFACT CONTRADICTION ({_DIMENSION_LABEL[f['dimension']]}"
                f", quantity named '{f['name']}'): {f['reason']}. "
                f"{a_label}:{a.location} states "
                f"{a.claim()}; {b_label}:{b.location} states "
                f"{b.claim()}. Two architecture artifacts cannot both be "
                f"authoritative -- pick one value and update every artifact that "
                f"repeats the other."
            ),
            "category": "structural",
            "check": "cross_artifact_consistency",
            "severity": "error",
            "source_doc": "",
            "finding_kind": "deterministic_quantity",
            "quantity_name": f["name"],
            "dimension": f["dimension"],
            "locations": [
                {
                    "artifact": a.artifact, "file": a_label,
                    "location": a.location, "claim": a.claim(),
                    "quote": a.sentence,
                },
                {
                    "artifact": b.artifact, "file": b_label,
                    "location": b.location, "claim": b.claim(),
                    "quote": b.sentence,
                },
            ],
            "evidence": (
                f"A: {a_label}:{a.location} — \"{a.sentence}\"\n"
                f"B: {b_label}:{b.location} — \"{b.sentence}\"{related}"
            )[:1000],
            "suggested_fix": (
                f"Decide which value is authoritative for '{f['name']}' "
                f"({_DIMENSION_LABEL[f['dimension']]}) and re-issue EVERY "
                f"artifact that repeats the stale one (the same stale number "
                f"is usually restated in several places)."
            ),
        })
    return result


def load_json_source(
    artifact: str, path: Any, label: str = "",
) -> ArtifactSource | None:
    """Build an :class:`ArtifactSource` from a JSON file, or None if unusable."""
    from pathlib import Path as _Path

    p = _Path(path)
    if not p.exists():
        return None
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return ArtifactSource(
        artifact=artifact, kind="json", payload=payload,
        label=label or p.name,
    )


def load_text_source(
    artifact: str, path: Any, label: str = "",
) -> ArtifactSource | None:
    """Build an :class:`ArtifactSource` from a text/markdown file."""
    from pathlib import Path as _Path

    p = _Path(path)
    if not p.exists():
        return None
    try:
        text = p.read_text(encoding="utf-8")
    except OSError:
        return None
    return ArtifactSource(
        artifact=artifact, kind="text", text=text, label=label or p.name,
    )
