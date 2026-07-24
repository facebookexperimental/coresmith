# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the LICENSE file.

"""Static composition auditor for the integrated Amaranth chip model.

The audit is deliberately AST-only: generated code is not executed here. It
checks constructor signatures, provable zero-width Signals, forgotten or
multiply-driven internal nets, and simulator/private-hierarchy introspection in
hardware. Port direction is derived from ``self.<port>.eq(...)`` targets in each
block's ``elaborate()`` method.
"""
from __future__ import annotations

import ast
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)
CHIP_MODEL_FUNC = "chip_model"
_SIGNAL_CTORS = {"Signal", "ClockSignal", "ResetSignal"}


def composition_audit_enabled() -> bool:
    return os.environ.get("CORESMITH_COMPOSITION_AUDIT", "1").strip().lower() not in (
        "0", "false", "no", "off",
    )


@dataclass
class BlockPortInfo:
    name: str
    params: list[str]
    n_defaults: int = 0
    flexible: bool = False
    driven: set[str] = field(default_factory=set)

    @property
    def required(self) -> list[str]:
        return self.params[:-self.n_defaults] if self.n_defaults else self.params

    def signature_str(self) -> str:
        return f"{self.name}({', '.join(self.params)})"


def _method(cls: ast.ClassDef, name: str) -> ast.FunctionDef | None:
    return next(
        (n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name),
        None,
    )


def _params(fn: ast.FunctionDef) -> list[str]:
    args = list(fn.args.posonlyargs) + list(fn.args.args)
    return [a.arg for a in args if a.arg != "self"]


def _expr_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
            and node.value.id == "self"):
        return f"self.{node.attr}"
    return None


def _eq_targets(node: ast.AST) -> list[str]:
    """Return names on the LHS of Amaranth ``value.eq(rhs)`` calls."""
    out: list[str] = []
    for sub in ast.walk(node):
        if not (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)
                and sub.func.attr == "eq"):
            continue
        name = _expr_name(sub.func.value)
        if name:
            out.append(name)
    return out


def _assignment_pairs(target: ast.AST, value: ast.AST):
    """Yield scalar pairs from ``a,b = x,y`` as well as simple assignments."""
    if (isinstance(target, (ast.Tuple, ast.List))
            and isinstance(value, (ast.Tuple, ast.List))
            and len(target.elts) == len(value.elts)):
        for left, right in zip(target.elts, value.elts):
            yield from _assignment_pairs(left, right)
    else:
        yield target, value


def _analyze_block_module(path: Path) -> BlockPortInfo | None:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError) as exc:
        logger.info("composition audit: cannot parse %s (%s)", path, exc)
        return None
    cls = next(
        (n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == path.stem),
        None,
    )
    if cls is None:
        return None
    init = _method(cls, "__init__")
    elaborate = _method(cls, "elaborate")
    if init is None or elaborate is None:
        return None
    params = _params(init)

    # __init__ normally binds self.port = port. Record those aliases, then map
    # self.port.eq(...) in elaborate back to constructor parameters.
    attrs: dict[str, str] = {}
    for sub in ast.walk(init):
        if not isinstance(sub, ast.Assign) or len(sub.targets) != 1:
            continue
        for tgt, val in _assignment_pairs(sub.targets[0], sub.value):
            if (isinstance(tgt, ast.Attribute) and isinstance(tgt.value, ast.Name)
                    and tgt.value.id == "self" and isinstance(val, ast.Name)
                    and val.id in params):
                attrs[tgt.attr] = val.id
    driven = {
        attrs[name.split(".", 1)[1]]
        for name in _eq_targets(elaborate)
        if name.startswith("self.") and name.split(".", 1)[1] in attrs
    }
    return BlockPortInfo(
        name=path.stem,
        params=params,
        n_defaults=len(init.args.defaults),
        flexible=bool(init.args.vararg or init.args.kwarg),
        driven=driven,
    )


def collect_block_port_info(models_dir: str | Path) -> dict[str, BlockPortInfo]:
    out: dict[str, BlockPortInfo] = {}
    d = Path(models_dir)
    if not d.is_dir():
        return out
    for p in sorted(d.glob("*.py")):
        if p.name.startswith("_"):
            continue
        info = _analyze_block_module(p)
        if info:
            out[p.stem] = info
    return out


def block_signature_appendix(models_dir: str | Path, max_chars: int = 4000) -> str:
    infos = collect_block_port_info(models_dir)
    if not infos:
        return ""
    text = "Actual block-model constructors (wire exactly):\n  " + "\n  ".join(
        infos[k].signature_str() for k in sorted(infos)
    )
    return text[:max_chars]


def _map_call_args(call: ast.Call, params: list[str]) -> tuple[dict[str, ast.expr], list[str]]:
    bound: dict[str, ast.expr] = {}
    problems: list[str] = []
    has_star = any(isinstance(a, ast.Starred) for a in call.args) or any(
        kw.arg is None for kw in call.keywords
    )
    pos = [a for a in call.args if not isinstance(a, ast.Starred)]
    if not has_star and len(pos) > len(params):
        problems.append(f"{len(pos)} positional args but constructor takes {len(params)}")
    for idx, arg in enumerate(pos):
        if idx < len(params):
            bound[params[idx]] = arg
    for kw in call.keywords:
        if kw.arg is None:
            continue
        if kw.arg not in params:
            problems.append(f"unexpected keyword argument {kw.arg!r}")
        elif kw.arg in bound:
            problems.append(f"parameter {kw.arg!r} bound twice")
        else:
            bound[kw.arg] = kw.value
    return bound, problems


def _const_int(node: ast.AST, consts: dict[str, int]) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return int(node.value)
    if isinstance(node, ast.Name):
        return consts.get(node.id)
    if isinstance(node, ast.BinOp):
        left, right = _const_int(node.left, consts), _const_int(node.right, consts)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.FloorDiv) and right:
            return left // right
    return None


def _signal_width(call: ast.Call, consts: dict[str, int]) -> int | None:
    if not call.args:
        return 1
    shape = call.args[0]
    if isinstance(shape, ast.Call):
        fn = getattr(shape.func, "id", None) or getattr(shape.func, "attr", None)
        if fn in ("signed", "unsigned") and shape.args:
            shape = shape.args[0]
        elif fn == "range":
            return None
    return _const_int(shape, consts)


def _signal_init_nonzero(call: ast.Call) -> bool | None:
    for kw in call.keywords:
        if kw.arg in ("init", "reset") and isinstance(kw.value, ast.Constant):
            return bool(kw.value.value)
    return False


@dataclass
class _Endpoint:
    block_var: str
    block: str
    param: str
    direction: str


@dataclass
class AuditResult:
    violations: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _violation(check: str, block: str, observed: str, expected: str, fix: str) -> dict:
    return {
        "type": "model_integration_failure",
        "criterion": "composition_audit",
        "audit_check": check,
        "first_divergence_block": block,
        "expected": expected,
        "observed": observed,
        "gap_class": "contract",
        "suggested_fix": fix,
    }


def audit_chip_model(chip_model_path: str | Path, models_dir: str | Path) -> AuditResult:
    res = AuditResult()
    try:
        tree = ast.parse(Path(chip_model_path).read_text(encoding="utf-8"))
    except (OSError, SyntaxError) as exc:
        logger.info("composition audit: cannot parse chip model (%s)", exc)
        return res
    infos = collect_block_port_info(models_dir)
    if not infos:
        return res

    aliases: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module in infos:
            for alias in node.names:
                if alias.name == node.module:
                    aliases[alias.asname or alias.name] = node.module

    cls = next(
        (n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == CHIP_MODEL_FUNC),
        None,
    )
    if cls is None:
        return res
    init, elaborate = _method(cls, "__init__"), _method(cls, "elaborate")
    if init is None or elaborate is None:
        return res

    primary_attrs: set[str] = set()
    for sub in ast.walk(init):
        if not (isinstance(sub, ast.Assign) and len(sub.targets) == 1):
            continue
        for tgt, _ in _assignment_pairs(sub.targets[0], sub.value):
            if (isinstance(tgt, ast.Attribute) and isinstance(tgt.value, ast.Name)
                    and tgt.value.id == "self"):
                primary_attrs.add(f"self.{tgt.attr}")

    consts: dict[str, int] = {}
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            value = _const_int(node.value, consts)
            if value is not None:
                consts[node.targets[0].id] = value

    nets: dict[str, ast.Call] = {}
    for sub in ast.walk(elaborate):
        if not (isinstance(sub, ast.Assign) and len(sub.targets) == 1
                and isinstance(sub.value, ast.Call)):
            continue
        ctor = getattr(sub.value.func, "id", None) or getattr(sub.value.func, "attr", None)
        name = _expr_name(sub.targets[0])
        if name and ctor in _SIGNAL_CTORS:
            nets[name] = sub.value

    endpoints: dict[str, list[_Endpoint]] = {}
    instance_no = 0
    for call in (n for n in ast.walk(elaborate) if isinstance(n, ast.Call)):
        fn = getattr(call.func, "id", None) or getattr(call.func, "attr", None)
        if fn not in aliases:
            continue
        stem, info = aliases[fn], infos[aliases[fn]]
        bound, problems = _map_call_args(call, info.params)
        if not info.flexible:
            missing = [p for p in info.required if p not in bound]
            if missing:
                problems.append("missing required parameter(s): " + ", ".join(missing))
        for problem in problems:
            res.violations.append(_violation(
                "instantiation_signature", stem,
                f"instantiation line {call.lineno}: {problem}", info.signature_str(),
                f"Wire exactly {info.signature_str()}.",
            ))
        instance_no += 1
        for param, arg in bound.items():
            name = _expr_name(arg)
            if name:
                endpoints.setdefault(name, []).append(_Endpoint(
                    f"i{instance_no}", stem, param,
                    "out" if param in info.driven else "in",
                ))

    # Hardware cannot use simulator/private hierarchy introspection.
    for sub in ast.walk(elaborate):
        if isinstance(sub, ast.Attribute) and sub.attr in ("symdict", "_fragment", "_engine"):
            res.violations.append(_violation(
                "unlowerable_introspection", "",
                f"private hierarchy access .{sub.attr} at line {sub.lineno}",
                "structural Signal wiring", "Expose the state through a real port.",
            ))
            break

    for name, ctor in nets.items():
        width = _signal_width(ctor, consts)
        if width is not None and width <= 0:
            res.violations.append(_violation(
                "zero_width_signal", "", f"net {name!r} has width {width}",
                "positive contract width", "Use the frozen interface width.",
            ))

    glue = set(_eq_targets(elaborate))
    for net, eps in endpoints.items():
        if net in primary_attrs or net not in nets:
            continue
        drivers = [f"glue:{net}" for _ in [0] if net in glue]
        drivers += [f"{e.block}.{e.param}" for e in eps if e.direction == "out"]
        consumers = [e for e in eps if e.direction == "in"]
        if len(drivers) > 1:
            res.violations.append(_violation(
                "multi_driven_net", consumers[0].block if consumers else eps[0].block,
                f"net {net!r} drivers: {', '.join(drivers)}", "one driver",
                "Keep one block/glue driver per Amaranth Signal.",
            ))
        elif not drivers and consumers and not _signal_init_nonzero(nets[net]):
            distinct = {e.block_var for e in consumers}
            desc = ", ".join(f"{e.block}.{e.param}" for e in consumers)
            if len(distinct) >= 2:
                res.violations.append(_violation(
                    "undriven_net", consumers[0].block,
                    f"net {net!r} consumed by {desc} has no driver", "one driver",
                    "Wire the producer or add explicit combinational glue.",
                ))
            else:
                res.warnings.append(f"net {net!r} consumed by {desc} is tied low or forgotten")
    return res


def audit_violations(chip_model_path: str | Path, models_dir: str | Path) -> list[dict]:
    if not composition_audit_enabled():
        return []
    try:
        return audit_chip_model(chip_model_path, models_dir).violations
    except Exception as exc:  # noqa: BLE001
        logger.warning("composition audit: internal error (%s) -- skipping", exc)
        return []
