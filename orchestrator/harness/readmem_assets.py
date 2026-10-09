"""Conservative textual binding of include and parameterised readmem assets.

This is deliberately not a Verilog evaluator: only whole string literals bind.
Token spans keep comments, strings and nested override expressions distinct and
let elaboration stage exactly the same literals that the manifest hashes.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

_TOKEN = re.compile(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\\\S+|[A-Za-z_$][\w$]*|\S', re.S)
_IDENTIFIER = re.compile(r'[A-Za-z_][\w$]*\Z')


@dataclass(frozen=True)
class Token:
    text: str
    path: Path
    start: int
    end: int


def tokens(path: Path, text: str) -> list[Token]:
    return [Token(m[0], path, m.start(), m.end()) for m in _TOKEN.finditer(text)
            if not m[0].startswith(('//', '/*'))]


def group(ts: list[Token], start: int) -> tuple[list[Token], int]:
    """Read one balanced group, including nested expressions and strings."""
    pairs = {'(': ')', '[': ']', '{': '}'}
    stack = [pairs[ts[start].text]]
    for i in range(start + 1, len(ts)):
        word = ts[i].text
        if word in pairs:
            stack.append(pairs[word])
        elif word in pairs.values():
            if word != stack.pop():
                break
            if not stack:
                return ts[start + 1:i], i + 1
    raise ValueError('Unbalanced HDL group')


def split(ts: list[Token], separator=',') -> list[list[Token]]:
    parts, start, i = [], 0, 0
    while i < len(ts):
        if ts[i].text in ('(', '[', '{'):
            _, i = group(ts, i)
            continue
        if ts[i].text == separator:
            parts.append(ts[start:i])
            start = i + 1
        i += 1
    return [*parts, ts[start:]]


@dataclass
class Module:
    name: str
    body: list[Token]
    parameters: dict[str, list[Token]] = field(default_factory=dict)
    private_parameters: set[str] = field(default_factory=set)


def modules(ts: list[Token]) -> list[Module]:
    result, i = [], 0
    while i < len(ts):
        if ts[i].text != 'module':
            i += 1
            continue
        begin = i
        i += 1
        if ts[i].text in ('automatic', 'static'):
            i += 1
        name = ts[i].text
        while i < len(ts) and ts[i].text != 'endmodule':
            i += 1
        if i == len(ts):
            raise ValueError(f'Unterminated module {name}')
        mod = Module(name, ts[begin:i + 1])
        j, scope = 0, 0
        while j < len(mod.body):
            word = mod.body[j].text
            if word in ('begin', 'fork', 'function', 'task', 'generate'):
                scope += 1
            elif word in ('end', 'join', 'join_any', 'join_none', 'endfunction', 'endtask', 'endgenerate'):
                scope = max(0, scope - 1)
            if word not in ('parameter', 'localparam'):
                j += 1
                continue
            j += 1
            end = j
            while end < len(mod.body) and mod.body[end].text not in (';', ')'):
                if mod.body[end].text in ('(', '[', '{'):
                    _, end = group(mod.body, end)
                else:
                    end += 1
            public = word == 'parameter' and not scope
            for part in split(mod.body[j:end]):
                if part and part[0].text in ('parameter', 'localparam'):
                    public = part[0].text == 'parameter' and not scope
                eq = next((k for k, t in enumerate(part) if t.text == '='), None)
                if eq is not None and eq > 0:
                    param = part[eq - 1].text
                    if not public:
                        mod.private_parameters.add(param)
                        continue
                    if param in mod.parameters:
                        raise ValueError(f'Duplicate parameter {name}.{param}')
                    mod.parameters[param] = part[eq + 1:]
            j = end
        result.append(mod)
        i += 1
    return result


@dataclass
class Assets:
    root: Path
    include_dirs: tuple[Path, ...] = ()
    texts: dict[Path, str] = field(default_factory=dict)
    dependencies: set[Path] = field(default_factory=set)
    # Replacements point to files; includes point to the recursively staged file.
    replacements: dict[Token, tuple[Path, bool]] = field(default_factory=dict)
    empty_guards: set[tuple[Token, Token, str]] = field(default_factory=set)

    def resolve(self, token: Token, name: str, *, include=False) -> Path:
        choices = {p.resolve() for p in (token.path.parent / name, self.root / name,
                   self.root / 'inputs' / name,
                   *[p / name for p in self.include_dirs if include]) if p.is_file()}
        if len(choices) != 1:
            from orchestrator.harness.top_module import CandidateError
            raise CandidateError(f'Dependency {name!r} from {token.path} is '
                                 + ('missing' if not choices else 'ambiguous'), 'infrastructure_error')
        return choices.pop()

    def literal(self, value: list[Token], site: str, *, include=False):
        from orchestrator.harness.top_module import CandidateError
        if len(value) != 1 or not re.fullmatch(r'"[^"\\\n]*"', value[0].text):
            raise CandidateError(f'Nonliteral or unresolved asset at {site}; dependency cannot be bound')
        token = value[0]
        name = token.text[1:-1]
        if not name and not include:
            return  # The parameter convention explicitly means no image.
        dep = self.resolve(token, name, include=include)
        self.dependencies.add(dep)
        self.replacements[token] = (dep, include)
        return dep

    def load(self, path: Path, ancestors=()) -> list[Token]:
        from orchestrator.harness.top_module import CandidateError
        if path in ancestors:
            raise CandidateError(f'Recursive include in {path}')
        if path not in self.texts:
            self.texts[path] = path.read_text()
        ts = tokens(path, self.texts[path])
        expanded, i = [], 0
        while i < len(ts):
            if ts[i].text == '`' and i + 1 < len(ts) and ts[i + 1].text == 'include':
                value = ts[i + 2:i + 3]
                dep = self.literal(value, f'include in {path}', include=True)
                expanded.extend(self.load(dep, (*ancestors, path)))
                i += 3
            else:
                expanded.append(ts[i])
                i += 1
        return expanded

    def rewritten(self, path: Path, staged=None, *, guard_empty=False) -> str:
        text = self.texts[path]
        edits = [(t.start, t.end, json.dumps(str(staged[dep] if inc and staged else dep)))
                 for t, (dep, inc) in self.replacements.items() if t.path == path]
        if guard_empty:
            for first, last, parameter in self.empty_guards:
                if first.path == path:
                    edits.extend([(first.start, first.start, f'begin if ({parameter} != "") '),
                                  (last.end, last.end, ' end')])
        for start, end, replacement in sorted(edits, reverse=True):
            text = text[:start] + replacement + text[end:]
        return text

    def stage(self, paths: list[Path], directory: Path) -> list[Path]:
        staged = {p: directory / f'input_{i}.v' for i, p in enumerate(self.texts)}
        for path in self.texts:
            staged[path].write_text(self.rewritten(path, staged, guard_empty=True))
        return [staged[p] for p in paths]


def _engine_supplied(mod: Module, root: Path) -> bool:
    """Module text the engine wrote: its SRAM library or a snapshot of it."""
    from orchestrator.langgraph.sram_wrapper import wrapper_lib_path
    path = mod.body[0].path
    return path == Path(wrapper_lib_path()).resolve() or (
        path.parent == root / '.coresmith/candidate-library'
        and path.name.split('-', 1)[0] == mod.name)


def _forwarded(value: list[Token], parent: Module) -> str | None:
    """The parent's own public parameter when ``value`` is exactly its name."""
    if len(value) == 1 and value[0].text in parent.parameters and value[0].text not in parent.private_parameters:
        return value[0].text
    return None


def _instantiations(parent: Module, targets: dict):
    """(target parameters, override values) per resolvable target instance."""
    for i, token in enumerate(parent.body):
        if token.text not in targets or (i and parent.body[i - 1].text == 'module'):
            continue
        mod, params = targets[token.text]
        if i + 1 < len(parent.body) and parent.body[i + 1].text == '#':
            try:
                parts = split(group(parent.body, i + 2)[0])
                if parts and parts[0] and parts[0][0].text == '.':
                    values = {p[1].text: group(p, 2)[0] for p in parts if len(p) >= 4 and p[2].text == '('}
                else:
                    values = dict(zip(mod.parameters, parts))
            except (ValueError, IndexError):
                continue  # the binding loop reports the malformed override
            yield params, values


def bind_assets(paths, project_root, *, top_module="", parameters=None, texts=None, include_dirs=()) -> Assets:
    from orchestrator.harness.top_module import CandidateError
    assets = Assets(Path(project_root).resolve(), include_dirs=tuple(Path(p).resolve() for p in include_dirs), texts=dict(texts or {}))
    try:
        ts = [t for path in paths for t in assets.load(Path(path).resolve())]
        mods = modules(ts)
        targets = {}
        covered = set()
        for mod in mods:
            for i, token in enumerate(mod.body):
                if not token.text.startswith('$readmem'):
                    continue
                covered.add(token)
                args, past = group(mod.body, i + 1)
                value = split(args)[0]
                site = f'module {mod.name} in {token.path}'
                if len(value) == 1 and _IDENTIFIER.fullmatch(value[0].text):
                    param = value[0].text
                    site += f', parameter {param}, default instantiation'
                    if param not in mod.parameters or param in mod.private_parameters:
                        raise CandidateError(f'Unresolved readmem at {site}')
                    assets.literal(mod.parameters[param], site)
                    if past >= len(mod.body) or mod.body[past].text != ';':
                        raise CandidateError(f'Unresolved readmem statement at {site}')
                    if token.path != mod.body[past].path:
                        raise CandidateError(f'Readmem statement spans include files at {site}')
                    assets.empty_guards.add((token, mod.body[past], param))
                    targets.setdefault(mod.name, (mod, set()))[1].add(param)
                else:
                    assets.literal(value, site + ', parameter/expression ' + ''.join(t.text for t in value)
                                   + ', instantiation <default>')
        # An engine-supplied wrapper (e.g. cs_sram_1rw1r) forwards its own
        # public parameter by name to a readmem target. The forward is engine
        # text, not candidate content: the wrapper becomes a target for that
        # parameter, so its default and every candidate instantiation are bound
        # by the same literal rules. Candidate-authored forwards stay unresolved.
        forwards = set()
        while True:
            added = {(parent.name, param) for parent in mods if _engine_supplied(parent, assets.root)
                     for inst_params, values in _instantiations(parent, targets)
                     for target_param in inst_params & values.keys()
                     if (param := _forwarded(values[target_param], parent)) is not None} - forwards
            if not added:
                break
            forwards |= added
            for name, param in added:
                mod = next(m for m in mods if m.name == name)
                assets.literal(mod.parameters[param], f'module {name} in {mod.body[0].path}, '
                               f'parameter {param}, default instantiation')
                targets.setdefault(name, (mod, set()))[1].add(param)
        for token in ts:
            if token.text.startswith('$readmem') and token not in covered:
                raise CandidateError(f'Readmem outside a resolvable module in {token.path}')
        for name in targets:
            if sum(mod.name == name for mod in mods) != 1:
                raise CandidateError(f'Ambiguous module {name} asset parameter declarations')
        # Macro expansion and defparam can hide or replace #() arguments. They
        # are not textual string-literal overrides; do not silently use defaults.
        directives = {'define', 'undef', 'ifdef', 'ifndef', 'elsif', 'else', 'endif',
                      'timescale', 'default_nettype', 'resetall', 'celldefine',
                      'endcelldefine', 'line', 'unconnected_drive', 'nounconnected_drive',
                      'begin_keywords', 'end_keywords'}
        for i, token in enumerate(ts):
            if targets and token.text == '`' and (i + 1 == len(ts) or ts[i + 1].text not in directives):
                tail = ts[i + 1:]
                end = next((n for n, t in enumerate(tail) if t.text == ';'), len(tail))
                sites = '; '.join(f'module {name}, parameter {", ".join(sorted(params))}'
                                  for name, (_, params) in targets.items())
                raise CandidateError(f'Cannot bind {sites}, instantiation '
                                     f'{" ".join(t.text for t in tail[:end])}: unresolved macro')
            if token.text == 'defparam':
                tail = ts[i + 1:]
                end = next((n for n, t in enumerate(tail) if t.text == ';'), len(tail))
                for assignment in split(tail[:end]):
                    eq = next((n for n, t in enumerate(assignment) if t.text == '='), 0)
                    if not eq:
                        continue
                    param = assignment[eq - 1].text
                    for name, (_, params) in targets.items():
                        if param in params:
                            site = ''.join(t.text for t in assignment[:eq])
                            raise CandidateError(f'Cannot bind module {name}, parameter {param}, '
                                                 f'instantiation {site}: defparam is unsupported')
        if top_module in targets:
            _, params = targets[top_module]
            for param in params & (parameters or {}).keys():
                raise CandidateError(f'Cannot bind module {top_module}, parameter {param}, '
                                     'instantiation <top>: external asset overrides are unsupported')
        for parent in mods:
            for i, token in enumerate(parent.body):
                if token.text not in targets or (i and parent.body[i - 1].text == 'module'):
                    continue
                mod, params = targets[token.text]
                j = i + 1
                overrides = []
                if j < len(parent.body) and parent.body[j].text == '#':
                    overrides, j = group(parent.body, j + 1)
                # A module type must be followed by an instance name and ports.
                if j + 1 >= len(parent.body) or not (_IDENTIFIER.fullmatch(parent.body[j].text)
                            or parent.body[j].text.startswith('\\')):
                    continue
                instance = parent.body[j].text
                site = (f'module {mod.name}, parameter {", ".join(sorted(params))}, '
                        f'instantiation {parent.name}.{instance} in {token.path}')
                try:
                    parts = split(overrides) if overrides else []
                    values = {}
                    if parts and parts[0] and parts[0][0].text == '.':
                        for part in parts:
                            if len(part) < 4 or part[0].text != '.' or part[2].text != '(':
                                raise ValueError('unresolved named override')
                            name = part[1].text
                            value, end = group(part, 2)
                            if end != len(part) or name in values or name not in mod.parameters:
                                raise ValueError('duplicate or unknown override')
                            values[name] = value
                    else:
                        if len(parts) > len(mod.parameters):
                            raise ValueError('unresolved positional override')
                        values = dict(zip(mod.parameters, parts))
                    for param in params & values.keys():
                        if (parent.name, _forwarded(values[param], parent)) not in forwards:
                            assets.literal(values[param], site)
                except CandidateError:
                    raise
                except (ValueError, IndexError) as exc:
                    raise CandidateError(f'Cannot bind {site}: {exc}') from exc
        return assets
    except CandidateError:
        raise
    except (ValueError, IndexError, KeyError) as exc:
        raise CandidateError(f'Cannot resolve candidate asset syntax: {exc}') from exc
