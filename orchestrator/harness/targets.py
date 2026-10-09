"""Explicit build targets shared by lint, simulation and synthesis."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_$]*\Z")


def validate(root, doc, *, require_files=False):
    if not isinstance(doc, dict):
        raise ValueError("TARGET_INVALID: expected an object")
    unknown = set(doc) - {"top", "sources", "include_dirs", "defines", "parameters", "assets", "cwd"}
    if unknown:
        raise ValueError(f"TARGET_INVALID: unknown fields {sorted(unknown)}")
    root = Path(root).resolve()
    top = doc.get("top", "")
    if not isinstance(top, str) or not _IDENTIFIER.fullmatch(top):
        raise ValueError("TARGET_INVALID: top must name a module")
    out = {"top": top}
    for key in ("sources", "include_dirs", "assets"):
        values = doc.get(key, [])
        if not isinstance(values, list) or (key == "sources" and not values):
            raise ValueError(f"TARGET_INVALID: {key} must be a {'nonempty ' if key == 'sources' else ''}list")
        out[key] = []
        for value in values:
            if not isinstance(value, str) or not value.strip() or any(c in value for c in '\n\r"$`'):
                raise ValueError(f"TARGET_INVALID: invalid {key} path {value!r}")
            path = (root / value).resolve()
            if any(c.isspace() for c in str(path)):
                raise ValueError(f"TARGET_INVALID: build-tool paths cannot contain whitespace: {path}")
            if not re.fullmatch(r"[A-Za-z0-9_./+@:-]+", str(path)):
                raise ValueError(f"TARGET_INVALID: unsupported build-tool path characters: {path}")
            if key != "include_dirs" and path.is_dir():
                raise ValueError(f"TARGET_INVALID: {key} entry is a directory: {path}")
            if require_files and not (path.is_dir() if key == "include_dirs" else path.is_file()):
                raise ValueError(f"TARGET_UNAVAILABLE: {path}")
            out[key].append(str(path))
    cwd = doc.get("cwd", ".")
    if not isinstance(cwd, str) or not cwd.strip():
        raise ValueError("TARGET_INVALID: cwd must be a directory")
    out["cwd"] = str((root / cwd).resolve())
    if not Path(out["cwd"]).is_dir():
        raise ValueError(f"TARGET_INVALID: cwd does not exist: {out['cwd']}")
    for key in ("defines", "parameters"):
        values = doc.get(key, {})
        if not isinstance(values, dict):
            raise ValueError(f"TARGET_INVALID: {key} must be an object")
        for name, value in values.items():
            # Numeric/config tokens only; never allow arguments/script fragments through values.
            if not isinstance(name, str) or not _IDENTIFIER.fullmatch(name) or type(value) not in (str, int) or not re.fullmatch(r"[A-Za-z0-9_+'hHbBdDoOxX.-]+", str(value)):
                raise ValueError(f"TARGET_INVALID: invalid {key} entry {name!r}: {value!r}")
        out[key] = values
    return out


def load(root, name, *, require_files=True):
    from orchestrator.state_store.project_db import open_project
    doc = open_project(root).target_binding(name)
    return validate(root, doc, require_files=require_files) if doc is not None else None


def bind(root, name, doc):
    from orchestrator.state_store.project_db import open_project
    if not _IDENTIFIER.fullmatch(name):
        raise ValueError("TARGET_INVALID: name must be an identifier")
    value = validate(root, doc)
    open_project(root).bind_target(name, value)
    return value


def revision_payload(target):
    """Everything :func:`revision` hashes: the configuration, the discovered
    dependencies and ``{path: sha256}`` of every file the target compiles
    (sources, declared assets, every file of every include dir, discovered
    ``include`` / ``$readmem*`` files), plus the digest itself."""
    files = set(target["sources"] + target["assets"])
    for directory in target["include_dirs"]:
        files.update(str(p) for p in Path(directory).rglob("*") if p.is_file())
    # Reuse candidate dependency discovery for relative `include/$readmem* files.
    from orchestrator.harness.readmem_assets import bind_assets
    assets = bind_assets(target["sources"], Path(target["cwd"]), top_module=target["top"],
                         parameters=target["parameters"] or "none", include_dirs=target["include_dirs"])
    dependencies = sorted(str(p) for p in assets.dependencies)
    files.update(dependencies)
    payload = {"target": target, "dependencies": dependencies,
               "files": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sorted(files)}}
    payload["sha256"] = hashlib.sha256(json.dumps({k: payload[k] for k in ("target", "dependencies", "files")},
                                                  sort_keys=True).encode()).hexdigest()
    return payload


def revision(target):
    """Hash configuration, sources, declared runtime assets and include contents."""
    return revision_payload(target)["sha256"]


def verilator_args(target):
    return ["--top-module", target["top"],
            *["-I" + p for p in target["include_dirs"]],
            *[f"-D{k}={v}" for k, v in target["defines"].items()],
            *[f"-G{k}={v}" for k, v in target["parameters"].items()]]


def cmd_target(args):
    from orchestrator.harness.env import bootstrap_project_root
    root = bootstrap_project_root(args.project_root)
    try:
        if args.target_action == "bind":
            value = bind(root, args.name, json.loads(Path(args.file).read_text()))
        else:
            value = load(root, args.name, require_files=False)
            if value is None:
                raise ValueError(f"TARGET_UNBOUND: {args.name}")
        print(json.dumps({"name": args.name, **value}, indent=2))
        return 0
    except (ValueError, OSError) as exc:
        print(json.dumps({"error": str(exc)}))
        return 2


def register_cli(sub, run_wrap, add_root, add_json):
    parser = sub.add_parser("target", help="bind one build configuration used by lint/sim/synth")
    commands = parser.add_subparsers(dest="target_action", required=True)
    for verb in ("bind", "show"):
        p = commands.add_parser(verb)
        p.add_argument("name")
        if verb == "bind":
            p.add_argument("--file", required=True, help="JSON with top, sources, cwd, include_dirs, defines, parameters, assets")
        add_root(p)
        add_json(p)
        p.set_defaults(func=run_wrap(cmd_target))
