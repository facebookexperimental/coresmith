# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""``coresmith tool <verb>`` + ``coresmith pdk info``.

Runs a deployment's EDA verbs the same way agents already run
``coresmith verify ...``: uniform exit codes, ``--json`` machine output, and one
JSONL telemetry record per invocation. The deployment (resolved by
``orchestrator.pdk.registry``) owns binary resolution, PDK env, checkers, and
timeouts, so the caller only supplies verb I/O.

Exit codes (shared with ``harness.cli``):

    0  ToolResult.ok            (tool ran AND all blocking checkers pass)
    1  tool-or-checker fail     (ran, but a blocking check failed)
    2  usage
    3  infra                    (missing binary / timeout -> not tool_ok)
    4  capability skip          (deployment does not implement the verb)

Kept langgraph-free at import (like ``cli_verify``): the registry/deployment
imports -- which reach into langgraph -- are deferred into the handler bodies.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Exit codes (kept in sync with harness.cli).
EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_USAGE = 2
EXIT_INFRA = 3
EXIT_SKIP = 4

# The verbs exposed as ``coresmith tool <verb>`` subcommands, with the input
# flags each accepts (a verb-appropriate subset of the shared flag set).
_VERB_INPUTS: dict[str, tuple[str, ...]] = {
    "run_synth": ("rtl", "script"),
    "run_pnr": ("script", "netlist", "sdc"),
    "run_drc": ("script", "gds"),
    "run_lvs": ("spice", "netlist"),
    "run_sta": ("script", "netlist", "sdc"),
    "run_lint": ("rtl",),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _bootstrap(args) -> Path:
    from orchestrator.harness.env import bootstrap_project_root
    return bootstrap_project_root(getattr(args, "project_root", None))


def _get_deployment():
    from orchestrator.pdk.registry import get_deployment
    return get_deployment()


def _emit(args, payload: dict, human: str) -> None:
    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(human)


def _record_run(root: Path, verb: str, design: str, ok: bool,
                metrics: dict) -> None:
    """Append one JSONL telemetry record (best-effort; never raises)."""
    try:
        d = root / ".coresmith" / "tool_runs"
        d.mkdir(parents=True, exist_ok=True)
        rec = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "verb": verb,
            "design": design,
            "ok": ok,
            "metrics": metrics,
        }
        with (d / "tool_runs.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
    except Exception:  # noqa: BLE001
        pass


def _build_request(verb: str, args):
    """Assemble a ToolRequest from parsed CLI args."""
    from orchestrator.pdk.base import ToolRequest

    inputs: dict[str, Path] = {}
    params: dict = {}
    rtls = getattr(args, "rtl", None)
    if rtls:
        inputs["rtl"] = Path(rtls[0])
        params["rtls"] = [str(Path(r)) for r in rtls]
    for key in ("script", "netlist", "sdc", "gds", "spice"):
        val = getattr(args, key, None)
        if val:
            inputs[key] = Path(val)
    design = getattr(args, "design", None) or (
        Path(rtls[0]).stem if rtls else "")
    out_dir = getattr(args, "out_dir", None)
    return ToolRequest(
        verb=verb,
        design=design,
        inputs=inputs,
        out_dir=Path(out_dir) if out_dir else None,
        params=params,
        timeout_s=getattr(args, "timeout_s", None),
    )


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------
def cmd_tool_list(args) -> int:
    try:
        dep = _get_deployment()
    except Exception as exc:  # noqa: BLE001
        print(f"could not resolve deployment: {exc}", file=sys.stderr)
        return EXIT_INFRA
    tools = dep.tools()
    payload = {
        "deployment": dep.name,
        "capabilities": sorted(dep.capabilities()),
        "verbs": {
            verb: {
                "impl": type(t).__name__,
                "checkers": [type(c).__name__ for c in t.checkers()],
                "prompt_notes": t.prompt_notes(),
            }
            for verb, t in sorted(tools.items())
        },
    }
    human = [f"deployment: {dep.name}"]
    for verb, info in payload["verbs"].items():
        chk = ", ".join(info["checkers"]) or "-"
        human.append(f"  {verb:<12} {info['impl']:<20} checkers=[{chk}]")
    _emit(args, payload, "\n".join(human))
    return EXIT_PASS


def cmd_tool_run(args) -> int:
    verb = args.verb
    try:
        root = _bootstrap(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_USAGE
    try:
        dep = _get_deployment()
    except Exception as exc:  # noqa: BLE001
        print(f"could not resolve deployment: {exc}", file=sys.stderr)
        return EXIT_INFRA

    if not dep.supports(verb):
        from orchestrator.pdk.base import ToolResult
        reason = (f"deployment '{dep.name}' does not implement '{verb}' "
                  f"(capabilities: {sorted(dep.capabilities())})")
        result = ToolResult.skipped(verb, reason,
                                    design=getattr(args, "design", "") or "")
        _record_run(root, verb, result.design, result.ok, result.metrics)
        _emit(args, result.to_json(), f"SKIP: {reason}")
        return EXIT_SKIP

    req = _build_request(verb, args)
    try:
        result = dep.tool(verb).run(req)
    except Exception as exc:  # noqa: BLE001
        print(f"{verb} raised: {exc}", file=sys.stderr)
        _record_run(root, verb, req.design, False, {"error": str(exc)})
        return EXIT_INFRA

    _record_run(root, verb, req.design, result.ok, result.metrics)
    _emit(args, result.to_json(), _human_result(result))
    if result.ok:
        return EXIT_PASS
    if not result.tool_ok:
        return EXIT_INFRA
    return EXIT_FAIL


def _human_result(result) -> str:
    verdict = "PASS" if result.ok else ("INFRA" if not result.tool_ok else "FAIL")
    lines = [f"{result.verb} {result.design}: {verdict}"]
    for c in result.checks:
        lines.append(f"  [{c.status}] {c.name}"
                     + (f" -- {c.details}" if c.details else ""))
    if result.metrics:
        lines.append(f"  metrics: {result.metrics}")
    return "\n".join(lines)


def cmd_tool_emit_script(args) -> int:
    verb = args.emit_verb
    try:
        dep = _get_deployment()
    except Exception as exc:  # noqa: BLE001
        print(f"could not resolve deployment: {exc}", file=sys.stderr)
        return EXIT_INFRA
    tool = dep.tool(verb)
    if tool is None:
        print(f"deployment '{dep.name}' does not implement '{verb}'",
              file=sys.stderr)
        return EXIT_SKIP
    ref = tool.reference_script()
    if ref is None or not Path(ref).exists():
        print(f"no reference script for '{verb}' in deployment '{dep.name}'",
              file=sys.stderr)
        return EXIT_SKIP
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(Path(ref).read_text(), encoding="utf-8")
    print(f"wrote {verb} reference script -> {out}")
    return EXIT_PASS


def cmd_pdk_info(args) -> int:
    try:
        dep = _get_deployment()
    except Exception as exc:  # noqa: BLE001
        print(f"could not resolve deployment: {exc}", file=sys.stderr)
        return EXIT_INFRA
    info = dep.describe()
    human = [
        f"deployment: {info['deployment']}",
        f"capabilities: {', '.join(info['capabilities'])}",
    ]
    pdk = info.get("pdk", {})
    if pdk:
        human.append(f"pdk: {pdk.get('name')} ({pdk.get('process_nm')}nm), "
                     f"library {pdk.get('std_cell_library')}, "
                     f"corners {list((pdk.get('corners') or {}).keys())}")
    if info.get("data_dir"):
        human.append(f"data_dir: {info['data_dir']}")
    _emit(args, info, "\n".join(human))
    return EXIT_PASS


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
def register_tool(sub, run_wrap, add_project_root, add_json) -> None:
    """Add the ``tool`` command tree to the CLI subparser action."""
    tp = sub.add_parser("tool", help="run a deployment EDA verb (synth/pnr/drc/...)")
    tsub = tp.add_subparsers(dest="tool_cmd", required=True)

    # tool list
    tl = tsub.add_parser("list", help="list verbs, impl classes, checkers")
    add_json(tl)
    tl.set_defaults(func=run_wrap(cmd_tool_list))

    # tool <verb>
    for verb, keys in _VERB_INPUTS.items():
        vp = tsub.add_parser(verb, help=f"run the {verb} verb")
        add_project_root(vp)
        add_json(vp)
        # --design (optional for lint; derived from --rtl stem)
        vp.add_argument("--design", help="top module / block name"
                        + (" (default: --rtl stem)" if verb == "run_lint" else ""))
        if "rtl" in keys:
            vp.add_argument("--rtl", action="append", metavar="FILE",
                            help="RTL source (repeatable)")
        for key in ("script", "netlist", "sdc", "gds", "spice"):
            if key in keys:
                vp.add_argument(f"--{key}", metavar="FILE", help=f"{key} input")
        vp.add_argument("--out-dir", dest="out_dir", metavar="DIR",
                        help="output/working directory")
        vp.add_argument("--timeout-s", dest="timeout_s", type=int,
                        help="tool timeout in seconds")
        vp.set_defaults(func=run_wrap(cmd_tool_run), verb=verb)

    # tool emit-script <verb> --out <path>
    es = tsub.add_parser("emit-script",
                         help="write a verb's reference script to adapt")
    es.add_argument("emit_verb", metavar="verb",
                    help="verb whose reference script to emit")
    es.add_argument("--out", required=True, help="destination path")
    es.set_defaults(func=run_wrap(cmd_tool_emit_script))


def register_pdk(sub, run_wrap, add_project_root, add_json) -> None:
    """Add the ``pdk`` command tree (``pdk info``)."""
    pp = sub.add_parser("pdk", help="inspect the active PDK / deployment")
    psub = pp.add_subparsers(dest="pdk_cmd", required=True)
    pi = psub.add_parser("info", help="resolved deployment, paths, corners, caps")
    add_json(pi)
    pi.set_defaults(func=run_wrap(cmd_pdk_info))
