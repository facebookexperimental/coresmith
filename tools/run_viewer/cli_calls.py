"""The CLI ledger: audited ``coresmith`` calls joined to the native shell calls
that ran them.

The engine's ``actions`` table is authoritative for completed audited calls
(the row is written when the verb returns, so ``ts`` is the end time). Native
shell calls add the command line, its output and timing, and the calls the
audit never sees (``--help``, parser failures, verbs dispatched without the
audit hook). Invocation candidates are parsed only from the *input* of a
shell tool call; a command quoted inside output, a prompt or a document is
never parsed. A candidate inside a conditional or loop is not proof that
the branch ran or how often it ran. Audit rows confirm completed calls.

There is no shared identifier between an audit row and a shell call, so
every association is labelled:

* ``strong``  -- one candidate: verb, sub-verb and every literal argument of
  the native invocation match the audit argv, and the audit time lies inside
  the shell call's window;
* ``weak``    -- several compatible windows resolved by the closest end time,
  or only a window (a script, loop or variable that hides the argv);
* ``ambiguous`` -- equally plausible candidates; all listed, none chosen;
* ``unlinked``.
"""
from __future__ import annotations

import re
import shlex

UNAUDITED_VERBS = {
    "state": "bin/coresmith dispatches `state` without the audit hook",
    "logs": "bin/coresmith dispatches `logs` without the audit hook",
    "architecture": "bin/coresmith dispatches `architecture` verbs without the audit hook",
    "supervisor": "bin/coresmith dispatches `supervisor` without the audit hook",
    "actions": "the `actions` verb is excluded from its own log by design",
}
_CS_VAR = re.compile(r'^"?\$\{?CORESMITH_CLI(?::-[^}]*)?\}?"?$')
_VAR_REF = re.compile(r'^"?\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?"?$')
_ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_REDIR = re.compile(r"^(?:\d*>>?|<|\d*>&\d+|&>>?|2>&1|>&2)")
_LOOP_RE = re.compile(r"(?:^|[;\n&|(]\s*|\bdo\s+)(?:for|while|until)\s")
_SCRIPT_RE = re.compile(r"(?:^|[\s;&|(])(?:python3?|bash|sh|make|\./[\w./-]+\.sh|[\w./-]+\.py)\b")
_HEREDOC_RE = re.compile(r"<<-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1")
_VALUED = {"--project-root"}


_WRITE_HEREDOC = re.compile(r"(?:cat|tee)\s+(?:-a\s+)?>?\s*(?P<path>[^\s<>;|&]+)\s*<<-?\s*(['\"]?)(?P<tag>[A-Za-z_]\w*)\2[^\n]*\n"
                            r"(?P<body>.*?)\n\s*(?P=tag)\s*(?:\n|$)", re.S)
_MOVE = re.compile(r"\b(?:mv|cp|ln\s+-s[f]?|install(?:\s+-m\s*\d+)?)\s+(?:-\w+\s+)*(?P<src>[^\s;|&]+)\s+(?P<dst>[^\s;|&]+)")
_FORWARDS = re.compile(r"""(?:^|\s|exec\s+)(?:\S*/)?coresmith\s+(?:"\$@"|\$@|"\$\*"|\$\*)""", re.M)


def find_wrappers(tool_inputs) -> list[dict]:
    """Scripts an agent wrote whose body forwards its arguments to
    coresmith (``exec coresmith "$@"``), followed through ``mv`` / ``cp`` /
    ``ln -s``. ``tool_inputs``: ``[(call_id, tool_name, text, write_path)]``
    (``write_path`` for a file-writing tool, else None). Only what a tool
    input shows is used; nothing is inferred from outputs."""
    found: dict[str, dict] = {}
    for cid, tool, text, wpath in tool_inputs:
        if wpath and _FORWARDS.search(text or ""):
            found.setdefault(wpath, {"path": wpath, "defined_by": cid, "via": f"{tool} tool"})
        for m in _WRITE_HEREDOC.finditer(text or ""):
            if _FORWARDS.search(m.group("body")):
                found.setdefault(m.group("path"), {"path": m.group("path"), "defined_by": cid, "via": "heredoc"})
    changed = True
    while changed:
        changed = False
        for cid, tool, text, wpath in tool_inputs:
            for m in _MOVE.finditer(text or ""):
                if m.group("src") in found and m.group("dst") not in found:
                    found[m.group("dst")] = {"path": m.group("dst"), "defined_by": cid,
                                             "via": f"copied/moved from {m.group('src')}"}
                    changed = True
    return list(found.values())


def _wrapper_match(tok: str, wrappers) -> str | None:
    """The wrapper path a command token runs (exact path, or a relative
    form of it with at least one directory component)."""
    if not wrappers or "/" not in tok:
        return None
    t = tok[2:] if tok.startswith("./") else tok
    for w in wrappers:
        p = w["path"] if isinstance(w, dict) else str(w)
        if t == p or (p.endswith("/" + t) and t.count("/") >= 1):
            return p
    return None


def is_coresmith_token(tok: str) -> bool:
    t = tok.strip()
    for pre in ("$(", "`", "(", "{"):
        while t.startswith(pre):
            t = t[len(pre):]
    t = _ASSIGN.sub("", t, count=1) if _ASSIGN.match(t) else t
    for pre in ("$(", "`", "("):
        while t.startswith(pre):
            t = t[len(pre):]
    if _CS_VAR.match(t):
        return True
    base = t.rsplit("/", 1)[-1]
    return base == "coresmith"


def _strip_heredocs(cmd: str) -> tuple[str, bool]:
    """Remove heredoc bodies (data, or a script for another interpreter):
    commands inside them are not the shell's own command line."""
    lines = cmd.split("\n")
    out, i, found = [], 0, False
    while i < len(lines):
        ln = lines[i]
        out.append(ln)
        m = _HEREDOC_RE.search(ln)
        i += 1
        if m:
            found = True
            term = m.group(2)
            while i < len(lines) and lines[i].strip() != term:
                i += 1
            i += 1
    return "\n".join(out), found


def _segments(cmd: str) -> list[str]:
    """Split a shell command at unquoted ``;``, newlines, ``&&``, ``||``,
    ``|``, ``&``, ``$(``, backticks and parentheses."""
    segs, cur, q, i = [], [], None, 0
    while i < len(cmd):
        c = cmd[i]
        if q:
            cur.append(c)
            if c == q:
                q = None
            elif c == "\\" and q == '"' and i + 1 < len(cmd):
                cur.append(cmd[i + 1])
                i += 1
            i += 1
            continue
        if c in ("'", '"'):
            q = c
            cur.append(c)
            i += 1
            continue
        if c == "\\" and i + 1 < len(cmd):
            cur.append(cmd[i:i + 2])
            i += 2
            continue
        if c in ";\n|&`()" or cmd.startswith("$(", i):
            segs.append("".join(cur))
            cur = []
            i += 2 if cmd.startswith("$(", i) else 1
            continue
        cur.append(c)
        i += 1
    segs.append("".join(cur))
    return [s.strip() for s in segs if s.strip()]


def _command_position(tokens: list[str]) -> int | None:
    """Recognise command position through common shell wrappers.

    A token in an argument position (``which coresmith``, ``echo coresmith``)
    is data. Unknown wrappers remain unlinked rather than inventing a call.
    """
    i = 0
    controls = {"then", "do", "else", "if", "elif", "while", "until", "!", "{"}
    valued = {
        "sudo": {"-u", "--user", "-g", "--group", "-h", "--host", "-p", "--prompt", "-C"},
        "env": {"-u", "--unset", "-C", "--chdir"},
        "timeout": {"-s", "--signal", "-k", "--kill-after"},
        "nice": {"-n", "--adjustment"}, "exec": {"-a"},
        "command": set(), "nohup": set(), "time": {"-f", "--format", "-o", "--output"},
    }
    while i < len(tokens):
        tok = tokens[i]
        if tok in controls or _ASSIGN.match(tok):
            i += 1
            continue
        base = tok.rsplit("/", 1)[-1]
        if base not in valued:
            return i
        i += 1
        if base == "command" and any(t in ("-v", "-V") for t in tokens[i:i+2]):
            return None
        while i < len(tokens) and tokens[i].startswith("-"):
            option = tokens[i]
            i += 1
            if option == "--":
                break
            if option in valued[base]:
                i += 1
        if base == "timeout":
            i += 1  # timeout duration is an argument, not an executable
    return None


def parse_invocations(cmd: str, wrappers=None) -> dict:
    """``{invocations: [{argv, segment}], loop, scripted, heredoc}`` for the
    ``coresmith`` invocations visible in a shell command line (directly or
    through a known wrapper script, see :func:`find_wrappers`)."""
    body, heredoc = _strip_heredocs(cmd or "")
    invs = []
    segs = []
    aliases: set[str] = set()      # NAME in NAME="${CORESMITH_CLI:-coresmith}" earlier in the same command line
    for seg in _segments(body):
        try:
            toks = shlex.split(seg, posix=True)
        except ValueError:
            toks = seg.split()
        segs.append((seg, toks))
    for seg, toks in segs:
        i = _command_position(toks)
        for tok in (toks[:i] if i is not None else toks):
            m = _ASSIGN.match(tok)
            if m:
                name, value = tok.split("=", 1)
                if is_coresmith_token(value) or "CORESMITH_CLI" in value:
                    aliases.add(name)
        if i is not None and i < len(toks):
            tok = toks[i]
            via = _wrapper_match(tok, wrappers)
            vm = _VAR_REF.match(tok)
            if vm and vm.group(1) in aliases:
                via = via or f"${vm.group(1)} (assigned to coresmith in this command)"
            if is_coresmith_token(tok) or via:
                args = []
                skip = False
                for a in toks[i + 1:]:
                    if skip:
                        skip = False
                        continue
                    if _REDIR.match(a):
                        skip = a in (">", ">>", "<", "&>", "&>>", "1>", "2>")
                        continue
                    args.append(a)
                inv = {"argv": args, "segment": seg, "execution": "candidate"}
                if via:
                    inv["via_wrapper"] = via
                invs.append(inv)
    loop = bool(_LOOP_RE.search(body))
    expanded_all = False
    if loop:
        invs, expanded_all = _expand_for_loops(body, invs)
    return {"invocations": invs, "loop": loop, "loop_expanded": expanded_all, "heredoc": heredoc,
            "scripted": bool(_SCRIPT_RE.search(body)), "mentions_coresmith": "coresmith" in body.lower()
            or "CORESMITH_CLI" in body or any(i.get("via_wrapper") for i in invs)}


_FOR_RE = re.compile(r"\bfor\s+([A-Za-z_][A-Za-z0-9_]*)\s+in\s+([^;\n]*?)\s*(?:;|\n)\s*do\b")


def _expand_for_loops(body: str, invs: list[dict]) -> tuple[list[dict], bool]:
    """``for v in a b c; do coresmith x $v; done`` -> one invocation per
    literal value (``loop_var`` / ``loop_value`` recorded). Loops over a
    command substitution, a glob or a variable are left as they are.
    Returns ``(invocations, every_loop_was_expandable)``."""
    loops = []
    ok = True
    for m in _FOR_RE.finditer(body):
        words = m.group(2)
        if any(ch in words for ch in "$`*?["):
            ok = False
            continue
        try:
            values = shlex.split(words)
        except ValueError:
            ok = False
            continue
        loops.append((m.group(1), values))
    if not loops:
        return invs, False
    out = []
    for inv in invs:
        done = False
        for var, values in loops:
            refs = ("$" + var, "${" + var + "}")
            if any(r in tok for tok in inv["argv"] for r in refs):
                seg = inv.get("segment") or ""
                quoted = any(f'"{r}' in seg for r in refs)
                for v in values:
                    argv = []
                    for tok in inv["argv"]:
                        if tok in refs and not quoted:
                            argv.extend(v.split())          # bash word-splits an unquoted expansion
                        else:
                            argv.append(tok.replace("${" + var + "}", v).replace("$" + var, v))
                    out.append({**inv, "argv": argv, "loop_var": var, "loop_value": v})
                done = True
                break
        if not done:
            out.append(inv)
    literal = all(_literal(t) for inv in out for t in inv["argv"][:2])
    return out, ok and literal


def verb_pair(argv: list[str]) -> tuple[str | None, str | None]:
    pos, skip = [], False
    for tok in argv or []:
        if skip:
            skip = False
            continue
        if tok.startswith("-"):
            skip = tok in _VALUED
            continue
        pos.append(tok)
        if len(pos) == 2:
            break
    return (pos[0] if pos else None), (pos[1] if len(pos) > 1 else None)


def _literal(tok: str) -> bool:
    return "$" not in tok and "`" not in tok and "*" not in tok


def argv_match(native: list[str], audit: list[str]) -> bool:
    """Every literal native token is in the audit argv and the first two
    positional tokens agree (a ``$VAR`` token matches anything)."""
    nv, ns = verb_pair(native)
    av, as_ = verb_pair(audit)
    if nv is None or av is None:
        return False
    if _literal(nv) and nv != av:
        return False
    if ns is not None and _literal(ns) and ns != as_:
        return False
    if ns is None and as_ is not None and av in ("build", "run", "daemon", "backend", "stage", "item", "check"):
        return False
    audit_set = set(audit)
    for tok in native:
        if _literal(tok) and tok not in audit_set:
            if "=" in tok and tok.split("=", 1)[1] in audit_set:
                continue
            return False
    return True


def classify_missing(argv: list[str], call: dict) -> tuple[str, str]:
    """Why an observed invocation has no audit row: ``(code, explanation)``."""
    out = (call.get("output") or "")[:20000]
    verb, sub = verb_pair(argv)
    if not argv or "-h" in argv or "--help" in argv or verb in ("help",):
        return "help", "help/usage output: argparse exits before the audit hook runs"
    if verb in UNAUDITED_VERBS:
        return "unaudited_verb", UNAUDITED_VERBS[verb]
    if call.get("exit_code") == 2 or ("usage: coresmith" in out and "error:" in out) or "invalid choice" in out:
        return "parser_failure", "parser failure: argparse exits (rc 2) before dispatch and the audit hook"
    if "command not found" in out or "No such file or directory" in out[:300]:
        return "not_started", "the shell could not start coresmith"
    if call.get("interrupted") or call.get("end_ts") is None:
        return "incomplete", "the call was interrupted or its completion was not observed (rows are written at the end)"
    if call.get("loop"):
        return "loop", "inside a shell loop: the number of executions is not visible in the command line"
    return "unexplained", "no audit row in the call window"


def match(actions: list[dict], calls: list[dict], *, slack_before: float = 2.0, slack_after: float = 3.0) -> dict:
    """Join audit rows to native shell calls. ``calls`` carry ``start_ts``,
    ``end_ts`` (None: not observed), ``invocations``, ``loop``, ``scripted``.
    Mutates both: ``action['native']`` and ``call['audit']`` link lists, and
    ``call['missing']`` for invocations no audit row was attributed to."""
    stats = {"strong": 0, "weak": 0, "ambiguous": 0, "unlinked": 0}
    for c in calls:
        c.setdefault("audit", [])

    def window(c):
        start = c.get("start_ts") if c.get("start_ts") is not None else c.get("end_ts")
        start = start if start is not None else 0.0
        hi = c.get("end_ts")
        hi = (hi + slack_after) if hi is not None else start + (c.get("open_window_s") or 0)
        return start - slack_before, hi

    cs_calls = [c for c in calls if c.get("invocations")]
    for a in actions:
        a["native"] = []
        ts = a.get("ts")
        if ts is None:
            a["link"] = "unlinked"
            stats["unlinked"] += 1
            continue
        strong = []
        for c in cs_calls:
            lo, hi = window(c)
            if not (lo <= ts <= hi):
                continue
            if any(argv_match(inv["argv"], a["argv"]) for inv in c["invocations"]):
                strong.append(c)
        strict = [c for c in strong if window(c)[0] + slack_before - 0.5 <= ts <= (
            (c["end_ts"] + 0.5) if c.get("end_ts") is not None else window(c)[1])]
        context = []
        if len(strong) == 1:
            label, chosen, alts = "strong", strong[0], []
        elif len(strict) == 1:
            label, chosen, alts = "strong", strict[0], [c for c in strong if c is not strict[0]]
        elif len(strong) > 1:
            def dist(c):
                end = c.get("end_ts")
                return abs(ts - end) if end is not None else 1e9
            ranked = sorted(strict or strong, key=dist)
            if len(ranked) == 1 or dist(ranked[1]) - dist(ranked[0]) > 2.0:
                label, chosen, alts = "weak", ranked[0], ranked[1:]
            else:
                label, chosen, alts = "ambiguous", None, ranked
        else:
            loose = []
            for c in calls:
                lo, hi = window(c)
                if not (lo <= ts <= hi) or not (c.get("loop") or c.get("scripted") or
                                                 any("$" in t for inv in c.get("invocations") or [] for t in inv["argv"][:2])):
                    continue
                (loose if c.get("mentions_coresmith") else context).append(c)
            if len(loose) == 1:
                label, chosen, alts = "weak", loose[0], []
            elif loose:
                label, chosen, alts = "ambiguous", None, loose
            else:
                label, chosen, alts = "unlinked", None, []
        a["link"] = label
        stats[label] += 1
        if context:
            a["context"] = [{"call": c["id"], "basis": "a script or loop running at the audit time; its command line "
                                                         "does not show coresmith, so it is not linked"} for c in context[:6]]
        basis = {"strong": "argv tokens match and the audit time is inside the shell call window"
                           + (" (the only call whose exact window contains it)" if strict and len(strong) > 1 else ""),
                 "weak": "closest compatible shell call window (several matched, or the argv is hidden by a "
                         "script/loop/variable in a command that names coresmith)",
                 "ambiguous": "several equally plausible shell calls"}.get(label)
        if chosen is not None:
            a["native"].append({"call": chosen["id"], "label": label, "basis": basis})
            chosen["audit"].append({"action": a["id"], "label": label})
        for c in alts:
            a["native"].append({"call": c["id"], "label": "ambiguous" if chosen is None else "alternative",
                                "basis": basis})
    for c in cs_calls:
        strong_rows = [x for x in c["audit"] if x["label"] in ("strong", "weak")]
        c["missing"] = []
        if c.get("loop") and not c.get("loop_expanded"):
            if not strong_rows:
                for inv in c["invocations"]:
                    code, why = classify_missing(inv["argv"], c)
                    c["missing"].append({"argv": inv["argv"], "code": code, "why": why})
            continue
        # one audit row per visible invocation, in order
        rows_by_inv = []
        remaining = list(strong_rows)
        action_by_id = {a["id"]: a for a in actions}
        for inv in c["invocations"]:
            hit = next((x for x in remaining if argv_match(inv["argv"], action_by_id[x["action"]]["argv"])), None)
            if hit is not None:
                remaining.remove(hit)
                rows_by_inv.append(hit["action"])
            else:
                rows_by_inv.append(None)
                code, why = classify_missing(inv["argv"], c)
                c["missing"].append({"argv": inv["argv"], "code": code, "why": why})
        c["invocation_actions"] = rows_by_inv
    return stats
