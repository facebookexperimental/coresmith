# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Cluster workers (architect sitting, step 4): few, large, long-lived block
owners instead of 29 x {spec, RTL, TB, diagnose} micro-calls.

A cluster is a set of blocks that belong together (the diagram's ``cluster``
or ``subsystem``: cpu, gpu, mem_fabric_periph ...). One session owns them for
the whole tier: it writes RTL / assertions / testbenches with its file tools,
iterates against ``coresmith verify`` and the edge VIPs, re-runs the shell
smoke after every change, and can only finish a block through ``coresmith
block-done <b>`` (conformance -> DV -> synth -> timing publishes ``best``).
The runner resumes the session (cache warm) until every block is done.
"""
from __future__ import annotations

import json
import subprocess

from .session import PROMPT_DIR, ArchitectSession, _read, _strip_format_fields

CLUSTER_APPENDIX = ["rtl_generator.md", "testbench_generator.md", "timing_closure.md",
                    "skills/verify_in_context.md", "skills/srdy_drdy.md", "skills/throughput_budget_contract.md",
                    "skills/serialization_contract.md", "skills/soc_fabric.md"]


def build_cluster_prompt() -> str:
    head = _read(PROMPT_DIR / "cluster_worker.md")
    parts = [head, "", "# Appendix: RTL / testbench / timing guidance (reference; the gate above wins)"]
    for name in CLUSTER_APPENDIX:
        t = _read(PROMPT_DIR / name)
        if t:
            parts += ["", f"## {name}", "", _strip_format_fields(t)]
    return "\n".join(parts)


class ClusterSession(ArchitectSession):
    def __init__(self, project_root, cluster: str, blocks: list[str], *, target_clock_mhz: float = 50.0, **kw):
        super().__init__(project_root, **kw)
        self.cluster = cluster
        self.blocks = list(blocks)
        self.target_clock_mhz = target_clock_mhz
        self.dir = self.root / ".coresmith" / "clusters" / cluster
        self.dir.mkdir(parents=True, exist_ok=True)

    # the system prompt is the cluster contract, not the architect's
    def sit(self, prompt: str, *, resume: str = "", index: int = 1) -> dict:
        import orchestrator.architect.session as _s
        orig = _s.build_system_prompt
        _s.build_system_prompt = build_cluster_prompt
        try:
            return super().sit(prompt, resume=resume, index=index)
        finally:
            _s.build_system_prompt = orig

    def _write_status(self, **kw) -> dict:
        st = self.status()
        st.update(kw)
        st["cluster"] = self.cluster
        st["blocks"] = self.blocks
        import time
        st["ts"] = time.time()
        self.status_path.write_text(json.dumps(st, indent=2))
        return st

    def block_statuses(self) -> dict:
        out = {}
        for b in self.blocks:
            try:
                p = subprocess.run([self.coresmith_bin, "block-status", b, "--project-root", str(self.root), "--json"],
                                   capture_output=True, text=True, timeout=120)
                out[b] = json.loads(p.stdout) if p.stdout.strip().startswith("{") else {"done": False, "error": p.stderr[-300:]}
            except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
                out[b] = {"done": False, "error": str(exc)}
        return out

    def stage_status(self) -> dict:
        sts = self.block_statuses()
        pending = [b for b, s in sts.items() if not s.get("done")]
        return {"stage": "blocks", "cluster": self.cluster, "pending": pending, "blocks": sts,
                "blocked_by": [{"code": "BLOCK_NOT_DONE", "text": "blocks without a published pass", "ids": pending, "count": len(pending)}] if pending else []}

    def done(self, st: dict | None = None) -> bool:
        st = st or self.stage_status()
        return not st.get("pending")

    def feedback_lines(self, sts: dict) -> list[str]:
        """Pending revision feedback (``gate_feedback.txt``: integration review,
        operator ``run revise-blocks``) for blocks without a published pass."""
        lines = []
        for b in self.blocks:
            fb = self.root / ".coresmith" / "blocks" / b / "gate_feedback.txt"
            if sts.get(b, {}).get("done") or not fb.is_file():
                continue
            text = fb.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                lines += ["", f"## MANDATORY revision feedback for {b} (`{fb.relative_to(self.root)}`)",
                          "", text[-6000:]]
        return lines

    def opening_prompt(self) -> str:
        sts = self.block_statuses()
        lines = [f"You are the cluster worker for cluster `{self.cluster}` in this project. Work in this directory.",
                 f"Target clock: {self.target_clock_mhz} MHz.", "", f"## Your blocks ({len(self.blocks)})"]
        for b in self.blocks:
            s = sts.get(b, {})
            lines.append(f"- **{b}** tier {s.get('tier')} {'(primitive: generated, do not write RTL)' if s.get('primitive') else ''}"
                         f" rtl `{s.get('rtl_path')}` tb `{s.get('tb_path')}` spec `{s.get('uarch_spec')}` "
                         f"slice `{s.get('contract_slice')}`; edges {len(s.get('edges') or [])}, VIPs {len(s.get('vips') or [])}; "
                         f"owns {', '.join(s.get('owned_items') or []) or '-'}")
        lines += ["", "## Shared context", "- HW/SW ABI: arch/hw_sw_abi.md   - DV rules: arch/DV_RULES.md   - FRD: arch/frd_spec.md",
                  "- SoC model: model/ (soc_model_top.h, per-block <block>_model.h/.cpp -- the golden your RTL must match)",
                  "- Shell top (stubs for other clusters' blocks): .coresmith/shell/",
                  "", "## Start", "Run `coresmith block-status <b> --json` for each block, then work block by block: RTL -> "
                  "`coresmith verify rtl <b> --lint-only` -> assertions -> testbench -> `coresmith verify rtl <b>` -> "
                  "`coresmith verify synth <b> --full` -> `coresmith block-done <b>`. After every published block run "
                  "`coresmith shell assemble` and report the cluster's cycles/frame estimate in notes/CLUSTER_<cluster>.md."]
        return "\n".join(lines + self.feedback_lines(sts))

    def resume_prompt(self, st: dict, sitting: int) -> str:
        lines = [f"Sitting {sitting}: continue. Blocks without a published pass: {', '.join(st.get('pending') or []) or 'none'}."]
        for b, s in (st.get("blocks") or {}).items():
            lines.append(f"- {b}: {'DONE' if s.get('done') else 'pending'}; rtl {'present' if s.get('rtl_exists') else 'MISSING'}, "
                         f"tb {'present' if s.get('tb_exists') else 'MISSING'}, attempts {s.get('attempts')}")
        lines.append("Finish every pending block through `coresmith block-done <b>`; read its stage report when it refuses.")
        return "\n".join(lines + self.feedback_lines(st.get("blocks") or {}))
