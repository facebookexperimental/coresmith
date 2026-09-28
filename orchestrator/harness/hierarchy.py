"""Candidate hierarchy evidence from Yosys elaboration, never source matching."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


class HierarchyFailure(str):
    """A truthy, serializable postcondition failure with a machine-readable kind."""

    def __new__(cls, reason: str, kind: str = "hierarchy_error"):
        obj = super().__new__(cls, reason)
        obj.kind = kind
        return obj


def _reachable_hierarchy(design: dict, top_module: str) -> set[str]:
    modules = design["modules"]
    if top_module not in modules:
        raise ValueError(f"elaborator did not return top {top_module!r}")
    reached, pending, instantiated = set(), [top_module], set()
    while pending:
        name = pending.pop()
        if name in reached:
            continue
        reached.add(name)
        for cell in modules[name]["cells"].values():
            child = cell["type"]
            if child not in modules:
                if child.startswith("$"):
                    continue  # Yosys primitive, not a design block.
                raise ValueError(f"elaborator returned unresolved cell {child!r}")
            instantiated.add(child)
            # Parameter specialization records the original HDL module name.
            origin = modules[child].get("attributes", {}).get("hdlname", "")
            instantiated.update(origin.split())
            pending.append(child)
    return instantiated


_CONDITIONAL = re.compile(r"^[ \t]*`(ifdef|ifndef|elsif|else|endif)\b[ \t]*([A-Za-z_][A-Za-z0-9_]*)?")


def _without_synthesis_branches(text: str) -> str:
    """Blank every branch that depends on SYNTHESIS (either polarity).

    Hierarchy evidence is then only the structure simulation and synthesis
    share: a block under `ifdef SYNTHESIS is not counted (the define-free
    candidate never simulates it), a block under `ifndef SYNTHESIS is not
    counted (silicon never has it), and simulation-only constructs such as
    `final` never reach the Yosys parser. Line numbers are preserved.
    """
    out, stack = [], []  # per conditional: [whole chain on SYNTHESIS, rest of chain dropped]
    for line in text.splitlines(keepends=True):
        m = _CONDITIONAL.match(line)
        kw, name = (m.group(1), m.group(2)) if m else ("", "")
        if kw in ("ifdef", "ifndef"):
            keep = not any(f[1] for f in stack) and name != "SYNTHESIS"
            stack.append([name == "SYNTHESIS"] * 2)
        elif kw in ("elsif", "else") and stack:
            keep = not any(f[1] for f in stack)
            if keep and kw == "elsif" and name == "SYNTHESIS":
                stack[-1][1] = True  # every remaining branch depends on SYNTHESIS
                line = line[:m.start(1)] + "else\n"
        elif kw == "endif" and stack:
            frame = stack.pop()
            keep = not any(f[1] for f in stack) and not frame[0]
        else:
            keep = not any(f[1] for f in stack)
        out.append(line if keep else "\n" * line.endswith("\n"))
    return "".join(out)


def _stage_sources(paths, stage: Path, project_root=None, *, defines=()):
    """Stage the same include/data literals that candidate adoption binds."""
    from orchestrator.harness.readmem_assets import bind_assets
    staged = bind_assets(paths, project_root or paths[0].parent).stage(paths, stage)
    if "SYNTHESIS" not in {d.split("=", 1)[0] for d in defines}:
        for path in stage.glob("input_*.v"):
            path.write_text(_without_synthesis_branches(path.read_text()))
    return staged


TIMEOUT_ENV = "CORESMITH_HIERARCHY_CHECK_TIMEOUT_S"


def hierarchy_timeout_s() -> int:
    """Yosys budget; a full SoC with a flattened fabric needs minutes, not 60 s."""
    from orchestrator._timeouts import scaled
    return max(1, scaled(900, env=TIMEOUT_ENV))


def elaborate_hierarchy(source_paths, top_module: str, *, defines=(), parameters=None,
                        project_root=None) -> set[str] | HierarchyFailure:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_$]*", top_module or ""):
        return HierarchyFailure("Integration postcondition failed: an explicit valid top is required")
    yosys = shutil.which("yosys")
    if not yosys:
        return HierarchyFailure("Hierarchy elaborator yosys is unavailable", "infrastructure_error")
    # Only simple tokens enter Yosys commands; paths are quoted separately.
    defines = list(defines or ())
    parameters = dict(parameters or {})
    if any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(?:=[A-Za-z0-9_'hHbBdDxX+-]+)?", d)
           for d in defines) or any(
               not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k)
               or not re.fullmatch(r"[A-Za-z0-9_'hHbBdD+-]+", str(v))
               for k, v in parameters.items()):
        return HierarchyFailure("Unsupported elaboration define/parameter token")
    try:
        paths = [Path(p).resolve(strict=True) for p in source_paths]
        if not paths:
            return HierarchyFailure("Hierarchy has no selected sources")
        with tempfile.TemporaryDirectory(prefix="coresmith-hierarchy-") as td:
            output = Path(td) / "design.json"
            script = Path(td) / "elaborate.ys"
            include_dirs = sorted({str(p.parent) for p in paths})
            if project_root:
                include_dirs.append(str(Path(project_root).resolve() / "inputs"))
            opts = " ".join([*("-D" + d for d in defines),
                             *("-I" + json.dumps(d) for d in include_dirs)])
            staged = _stage_sources(paths, Path(td), project_root, defines=defines)
            commands = [f"read_verilog -sv -nosynthesis {opts} " + " ".join(json.dumps(str(p)) for p in staged)]
            for key, value in sorted(parameters.items()):
                commands.append(f"chparam -set {key} {value} {top_module}")
            commands += [f"hierarchy -check -top {top_module}", "proc", f"write_json {json.dumps(str(output))}"]
            script.write_text("\n".join(commands) + "\n")
            result = subprocess.run([yosys, "-Q", "-T", "-s", str(script)],
                                    capture_output=True, text=True, timeout=hierarchy_timeout_s(),
                                    cwd=str(project_root or paths[0].parent))
            if result.returncode:
                return HierarchyFailure("Integration postcondition failed: elaboration rejected candidate: "
                                        + (result.stderr or result.stdout)[-2000:])
            return _reachable_hierarchy(json.loads(output.read_text()), top_module)
    except subprocess.TimeoutExpired as exc:
        return HierarchyFailure(f"Hierarchy elaboration unavailable: {exc} (raise {TIMEOUT_ENV} or "
                                "CORESMITH_TIMEOUT_MULTIPLIER)", "infrastructure_error")
    except OSError as exc:
        return HierarchyFailure(f"Hierarchy elaboration unavailable: {exc}", "infrastructure_error")
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        return HierarchyFailure(f"Invalid elaborator evidence: {exc}", "infrastructure_error")


def missing_blocks(instantiated: set[str], expected, top_module: str) -> HierarchyFailure | None:
    # The assembler's explicit top/block collision rename is part of its file contract.
    missing = sorted(name for name in expected if name not in instantiated
                     and not (name == top_module and name + "_pads" in instantiated))
    if missing:
        return HierarchyFailure(f"Integration postcondition failed: top {top_module!r} does NOT "
                                f"instantiate expected block(s): {missing}")
    return None
