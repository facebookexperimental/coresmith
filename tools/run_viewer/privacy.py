"""What never leaves the private snapshot.

Model hidden reasoning (Claude ``thinking`` / ``redacted_thinking`` blocks and
their signatures, Codex ``reasoning`` items, ``raw_content`` / ``summary_text``,
encrypted inter-agent payloads, streamed ``thinking_delta`` /
``signature_delta`` events) is replaced by a marker that keeps only its length
and whether a signature was present. Credential-like values are redacted.

Parsers export content by allowlist; :func:`scrub` is the second line of
defence for metadata dicts, and :class:`Ledger` remembers fingerprints of
every private payload it was shown so the exporter can prove none of them
reached the output (:meth:`Ledger.leaks`).

``decisions.reasoning`` / an interrupt resolution's ``reasoning`` in the
engine database are operator-supplied public rationales, not model reasoning;
they are kept (the parsers of engine rows never call :func:`scrub` with the
``reasoning`` key).
"""
from __future__ import annotations

import collections
import json
import re

_B64_FRAG = re.compile(r"[A-Za-z0-9+/=_\-]{40}")
_B64_RUN = re.compile(r"[A-Za-z0-9+/=_\-]{40,}")


def _is_b64_kind(kind: str) -> bool:
    """Payload kinds that are opaque base64 (signatures, encrypted content)."""
    k = kind or ""
    return ("signature" in k or "encrypted" in k or k.startswith(("codex_reasoning_copy", "key:signature",
                                                                   "key:encrypted_content")))

REASONING_MARKER = "[private reasoning not exported: {n} characters{sig}]"
ENCRYPTED_MARKER = "[encrypted payload not exported: {n} characters]"
SECRET_MARKER = "[secret-like value redacted]"

ENCRYPTED_RE = re.compile(r"gAAAA[A-Za-z0-9_\-=]{20,}")
SECRET_RES = (
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}"),
    re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_\-]{32,}"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{30,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}"),
    re.compile(r"\beyJ[A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-]{10,}"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._\-]{24,}"),
)
SECRET_ENV_RE = re.compile(r"(?i)(KEY|TOKEN|SECRET|PASSWORD|PASSWD|AUTH|CREDENTIAL|COOKIE|SESSION_KEY)")

# Keys whose values are private payloads wherever they appear in a native
# provider record. ``reasoning`` is deliberately absent: see the module doc.
PRIVATE_KEYS = frozenset({
    "signature", "encrypted_content", "raw_content", "summary_text", "thinking", "redacted_thinking",
    "creator_account_id", "creator_user_id", "credits", "balance",
})
PRIVATE_BLOCK_TYPES = frozenset({
    "thinking", "redacted_thinking", "reasoning", "agent_reasoning", "reasoning_text",
    "thinking_delta", "signature_delta", "reasoning_summary", "agent_reasoning_raw_content",
})


class Ledger:
    """Counts what was withheld and keeps short fingerprints of private
    payloads (never the payloads themselves in the output)."""

    def __init__(self):
        self.counts = collections.Counter()
        self._fingerprints: set[str] = set()
        self.kinds: dict[str, str] = {}         # fingerprint -> the kind of payload it came from
        self._prefixes = {}
        self._prefix_count = -1
        self._b64: set[str] = set()
        self._b64_count = -1
        self._pending: list[tuple[str, str, object]] = []    # (kind, text, scope) awaiting resolve()
        self._public: dict[object, list[str]] = collections.defaultdict(list)

    def private(self, kind: str, value) -> int:
        """Record one withheld payload; returns its length."""
        text = value if isinstance(value, str) else ("" if value is None else str(value))
        self.counts[kind] += 1
        self.counts[kind + ":chars"] += len(text)
        self._fingerprint(kind, text)
        return len(text)

    def _fingerprint(self, kind: str, text: str) -> None:
        if len(text) >= 40 and _is_b64_kind(kind):
            # every aligned 40-character block: any copied run of 79+
            # characters of the payload contains at least one of them
            for start in range(0, len(text) - 39, 40):
                frag = text[start:start + 40]
                if _B64_FRAG.fullmatch(frag):
                    self._fingerprints.add(frag)
                    self.kinds.setdefault(frag, kind)
            return
        if len(text) >= 32:
            for start in (0, len(text) // 2, max(0, len(text) - 40)):
                frag = text[start:start + 40]
                if len(frag) >= 32 and frag.strip():
                    self._fingerprints.add(frag)
                    self.kinds.setdefault(frag, kind)

    def private_text(self, kind: str, value, scope) -> int:
        """A withheld reasoning *text* whose fingerprint is decided later:
        Claude "narration" thinking blocks repeat text the same message also
        shows as a visible text block, so their text is not private and must
        not trip the leak check. Signatures are never deferred."""
        text = value if isinstance(value, str) else ("" if value is None else str(value))
        self.counts[kind] += 1
        self.counts[kind + ":chars"] += len(text)
        if text.strip():
            self._pending.append((kind, text, scope))
        return len(text)

    def public_text(self, scope, text: str) -> None:
        if text and text.strip():
            self._public[scope].append(text)

    def resolve(self) -> None:
        """Fingerprint every deferred reasoning text unless the whole text
        (whitespace-normalised) is part of the same agent's own visible
        output: a Claude narration block is shown to the user verbatim, so
        it is not private. A partial overlap never exempts anything, and
        signature / encrypted fingerprints are never deferred."""
        joined = {scope: "\x00".join(" ".join(v.split()) for v in texts) for scope, texts in self._public.items()}
        for kind, text, scope in self._pending:
            core = " ".join(text.split())
            if core and core in joined.get(scope, ""):
                self.counts["thinking_text_fully_visible_in_same_agent"] += 1
                continue
            self._fingerprint(kind, text)
        self._pending = []
        self._prefix_count = -1

    def leaks(self, blob: str) -> list[str]:
        """Fingerprints of private payloads found in ``blob`` (must be empty).

        Signatures and encrypted payloads are base64-like, so their
        fingerprints can only occur inside a base64 run of the blob: those
        runs are found by regex and every 40-character window of a run is
        looked up in a set. Other (text) fingerprints are searched directly,
        raw and JSON-escaped; when there are many, a prefix index visits
        each blob position once."""
        b64 = {f for f in self._fingerprints if _B64_FRAG.fullmatch(f)}
        text = self._fingerprints - b64
        found = set()
        if b64:
            for m in _B64_RUN.finditer(blob):
                run = m.group(0)
                for i in range(0, len(run) - 39):
                    w = run[i:i + 40]
                    if w in b64:
                        found.add(w)
        if not text:
            return sorted(found)
        variants = {}
        for f in text:
            variants[f] = f
            esc = json.dumps(f, ensure_ascii=False)[1:-1]
            if esc != f:
                variants[esc] = f
        if len(variants) < 512:
            found.update(orig for v, orig in variants.items() if v in blob)
            return sorted(found)
        # Thousands of independent substring scans over a large transcript
        # are quadratic in corpus size. Index a short prefix, then inspect
        # each position once; confirm every candidate with the full fragment.
        if self._prefix_count != len(variants):
            prefixes = collections.defaultdict(list)
            for v in variants:
                prefixes[v[:12]].append(v)
            self._prefixes = dict(prefixes)
            self._prefix_count = len(variants)
        candidates = self._prefixes.get
        for i in range(max(0, len(blob) - 11)):
            for v in candidates(blob[i:i + 12], ()):
                if blob.startswith(v, i):
                    found.add(variants[v])
        return sorted(found)

    @property
    def fingerprint_count(self) -> int:
        return len(self._fingerprints)

    def scrub_private_runs(self, text: str) -> str:
        """Withhold every base64 run of ``text`` that contains a fingerprint
        of a signature or encrypted payload (a fragment an agent printed from
        a raw record, e.g. a ``head -c`` of a live stream cut mid-value)."""
        if not text or len(text) < 40:
            return text
        if self._b64_count != len(self._fingerprints):
            self._b64 = {f for f in self._fingerprints if _is_b64_kind(self.kinds.get(f, ""))}
            self._b64_count = len(self._fingerprints)
        if not self._b64:
            return text
        out, pos = [], 0
        for m in _B64_RUN.finditer(text):
            run = m.group(0)
            if any(run[i:i + 40] in self._b64 for i in range(len(run) - 39)):
                out.append(text[pos:m.start()])
                out.append(f"[private signature/encrypted fragment not exported: {len(run)} characters]")
                pos = m.end()
                self.counts["private_run_withheld"] += 1
                self.counts["private_run_withheld:chars"] += len(run)
        if not out:
            return text
        out.append(text[pos:])
        return "".join(out)


def reasoning_marker(ledger: Ledger, kind: str, raw, signature=None, scope=None) -> str:
    raw = raw if raw is not None else ""
    n = ledger.private_text(kind, raw, scope) if scope is not None else ledger.private(kind, raw)
    if signature:
        ledger.private(kind + ":signature", signature)
    return REASONING_MARKER.format(n=n, sig=", signature present" if signature else "")


# A private key's JSON string value embedded in visible text (an agent that
# printed a raw session or live-stream record), at any escaping depth: the
# value ends at the first quote preceded by exactly as many backslashes as
# the opening quote.
_EMBED_KEYS = ("signature", "thinking", "redacted_thinking", "encrypted_content", "raw_content", "summary_text")
_EMBED_STR = re.compile(r'(?P<q>\\*)"(?P<k>' + "|".join(_EMBED_KEYS) + r')(?P=q)"\s*:\s*(?P=q)"(?P<v>.*?)(?<!\\)(?P=q)"',
                        re.S)
_EMBED_LIST = re.compile(r'(?P<q>\\*)"(?P<k>raw_content|summary_text)(?P=q)"\s*:\s*\[')
# a signature / encrypted value whose closing quote is missing (output cut by
# the shell or the tool mid-value): the base64 run itself is withheld
_EMBED_B64 = re.compile(r'(?P<pre>(?:signature|encrypted_content|"data)\\*"\s*:\s*\\*")(?P<v>[A-Za-z0-9+/=_\-]{16,})')
_EMBED_OPEN_THINKING = re.compile(r'(?P<q>\\*)"thinking(?P=q)"\s*:\s*(?P=q)"(?!\[private field)')
EMBEDDED_MARKER = "[private field not exported: {n} characters]"


def _redact_embedded(text: str, ledger: Ledger | None) -> str:
    def _str(m):
        v = m.group("v")
        if not v:
            return m.group(0)
        if ledger is not None:
            ledger.counts["embedded_private_field:" + m.group("k")] += 1
            ledger.counts["embedded_private_field:chars"] += len(v)
        q = m.group("q")
        return f'{q}"{m.group("k")}{q}":{q}"{EMBEDDED_MARKER.format(n=len(v))}{q}"'
    text = _EMBED_STR.sub(_str, text)

    def _b64(m):
        if ledger is not None:
            ledger.counts["embedded_private_field:unterminated_base64"] += 1
        return m.group("pre") + EMBEDDED_MARKER.format(n=len(m.group("v")))
    text = _EMBED_B64.sub(_b64, text)
    m = _EMBED_OPEN_THINKING.search(text)
    while m is not None:
        q = m.group("q")
        close = re.compile(r"(?<!\\)" + re.escape(q) + '"').search(text, m.end())
        if close is None:            # cut off: everything after the opening quote is the private value
            if ledger is not None:
                ledger.counts["embedded_private_field:unterminated_thinking"] += 1
            text = text[:m.end()] + EMBEDDED_MARKER.format(n=len(text) - m.end()) + " (value cut off in the source)"
            break
        if close.start() == m.end():   # empty value
            m = _EMBED_OPEN_THINKING.search(text, close.end())
            continue
        m = _EMBED_OPEN_THINKING.search(text, close.end())
    if "raw_content" in text or "summary_text" in text:
        out, pos = [], 0
        for m in _EMBED_LIST.finditer(text):
            if m.start() < pos:
                continue
            depth, j = 1, m.end()
            while j < len(text) and depth:
                depth += {"[": 1, "]": -1}.get(text[j], 0)
                j += 1
            body = text[m.end():j - 1]
            if body.strip():
                if ledger is not None:
                    ledger.counts["embedded_private_field:" + m.group("k")] += 1
                out.append(text[pos:m.end()] + EMBEDDED_MARKER.format(n=len(body)) + "]")
            else:
                out.append(text[pos:j])
            pos = j
        out.append(text[pos:])
        text = "".join(out)
    return text


def redact(text: str, ledger: Ledger | None = None) -> str:
    """Encrypted blobs become length markers; private-key values embedded as
    JSON inside visible text are withheld; credential-like tokens are
    redacted."""
    if not text:
        return text
    if any(k in text for k in _EMBED_KEYS):
        text = _redact_embedded(text, ledger)
    if "gAAAA" in text:
        def _enc(m):
            if ledger is not None:
                ledger.private("encrypted_inline", m.group(0))
            return ENCRYPTED_MARKER.format(n=len(m.group(0)))
        text = ENCRYPTED_RE.sub(_enc, text)
    for rx in SECRET_RES:
        if rx.search(text):
            def _sec(m):
                if ledger is not None:
                    ledger.private("secret_like", m.group(0))
                return SECRET_MARKER
            text = rx.sub(_sec, text)
    return text


def scrub(value, ledger: Ledger | None = None, *, drop=PRIVATE_KEYS):
    """A deep copy of a JSON value with private keys replaced by markers and
    every string passed through :func:`redact`."""
    if isinstance(value, str):
        return redact(value, ledger)
    if isinstance(value, list):
        return [scrub(v, ledger, drop=drop) for v in value]
    if isinstance(value, dict):
        kind = str(value.get("type", "")).lower()
        if kind in PRIVATE_BLOCK_TYPES or (
            value.get("role") == "assistant" and value.get("channel") == "analysis"
        ):
            if ledger is not None:
                ledger.private("nested_private_block", repr(value))
            return {"type": kind or "message", "omitted": "private reasoning not exported"}
        out = {}
        for k, v in value.items():
            if k in drop:
                if v in (None, "", [], {}):
                    continue
                if ledger is not None:
                    ledger.private("key:" + k, v if isinstance(v, str) else repr(v))
                out[k] = "[withheld: private field]"
                continue
            out[k] = scrub(v, ledger, drop=drop)
        return out
    return value


def mask_env_argv(argv: list) -> list:
    """``--env NAME=VALUE`` pairs of a launcher argv with credential-like
    names masked."""
    out = []
    for tok in argv or []:
        s = str(tok)
        if "=" in s and not s.startswith("-"):
            name, _, val = s.partition("=")
            if SECRET_ENV_RE.search(name) and val:
                s = f"{name}={SECRET_MARKER}"
        out.append(redact(s))
    return out
