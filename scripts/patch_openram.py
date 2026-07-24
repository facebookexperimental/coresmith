# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Idempotently repair a pip-installed OpenRAM so ``python -m openram`` runs
under NumPy 2.

The PyPI ``openram==1.2.48`` wheel has two defects that block sky130 macro
generation on a stock modern install:

1. It omits ``openram/__main__.py``, so ``python -m openram <cfg>`` (how the
   engine invokes it) dies with "cannot be directly executed" -- even though
   ``import openram`` succeeds.
2. Its gdsMill bounding-box path calls ``float()`` on size-one NumPy arrays,
   which NumPy >= 2 removed (``TypeError: only 0-dimensional arrays can be
   converted to Python scalars``). OpenRAM only pins ``numpy>=1.17.4``, so pip
   selects NumPy 2 and generation crashes mid-layout.

Both are patched HERE, in-repo, so a fresh checkout on any box self-heals the
first time the macro flow runs (see ``openram_gen.ensure_openram_patched``) --
rather than depending on a manual venv edit that would not travel with a
public commit. Every edit is idempotent (a marker/exact-pattern check), backs
the target up once, and is a no-op when already applied. Downgrading NumPy is
deliberately NOT attempted: a shared venv's SciPy/scikit-learn need NumPy 2.

Usable as a library (``patch_openram() -> PatchResult``) or a CLI
(``python scripts/patch_openram.py``; ``--check`` reports status without
writing).
"""
from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

_MAIN_MARKER = "coresmith openram __main__ shim"
_MAIN_SHIM = f'''"""Module entry point for ``python -m openram`` ({_MAIN_MARKER}).

The PyPI 1.2.48 wheel ships sram_compiler.py but omits __main__.py; recreate
it with the compiler's script semantics so the package is executable.
"""
from pathlib import Path
import runpy
import sys

_pkg = Path(__file__).resolve().parent
sys.path.insert(0, str(_pkg))
runpy.run_path(str(_pkg / "sram_compiler.py"), run_name="__main__")
'''

# The NumPy-2-incompatible bbox pattern, matched INDENTATION- and
# variable-name-agnostically: any `vector(boundary[i][j], boundary[k][l])`
# whose args are bare (no .item()). Adds `.item()` to each subscript so the
# size-one ndarray converts under NumPy 2. Idempotent: an already-fixed call
# has `.item()` before the comma/paren, so the bare-arg group won't match.
_BBOX_RE = re.compile(
    r"vector\(\s*"
    r"(boundary\[\d+\]\[\d+\])\s*,\s*"
    r"(boundary\[\d+\]\[\d+\])\s*\)"
)


def _bbox_sub(text: str) -> tuple[str, int]:
    """Return (patched_text, n_substitutions) -- adds .item() to bare
    ``vector(boundary[..], boundary[..])`` calls; leaves fixed ones alone."""
    return _BBOX_RE.subn(
        lambda m: f"vector({m.group(1)}.item(), {m.group(2)}.item())", text)


@dataclass
class PatchResult:
    ok: bool                       # openram is runnable after this call
    applied: list[str] = field(default_factory=list)
    already: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        parts = []
        if self.applied:
            parts.append("applied: " + ", ".join(self.applied))
        if self.already:
            parts.append("already-ok: " + ", ".join(self.already))
        if self.errors:
            parts.append("errors: " + "; ".join(self.errors))
        return f"openram runnable={self.ok} [{'; '.join(parts) or 'no-op'}]"


def _openram_pkg_dir() -> Path | None:
    try:
        spec = importlib.util.find_spec("openram")
    except Exception:
        return None
    if not spec or not spec.origin:
        return None
    return Path(spec.origin).resolve().parent


def _find_hierarchy_layout(pkg: Path) -> Path | None:
    cand = pkg / "compiler" / "base" / "hierarchy_layout.py"
    if cand.exists():
        return cand
    hits = list(pkg.rglob("hierarchy_layout.py"))
    return hits[0] if hits else None


def patch_openram(*, check_only: bool = False) -> PatchResult:
    """Apply (or, with ``check_only``, just report) both OpenRAM repairs."""
    res = PatchResult(ok=False)
    pkg = _openram_pkg_dir()
    if pkg is None:
        res.errors.append("openram package not importable")
        return res

    # (1) __main__.py launcher --------------------------------------------
    main_py = pkg / "__main__.py"
    main_ok = main_py.exists() and _MAIN_MARKER in main_py.read_text(
        errors="ignore") if main_py.exists() else False
    # A wheel that already ships a working __main__ (future fix) also counts.
    stock_main_ok = main_py.exists() and _MAIN_MARKER not in main_py.read_text(
        errors="ignore")
    if main_ok or stock_main_ok:
        res.already.append("__main__.py")
    elif check_only:
        res.errors.append("__main__.py missing")
    else:
        try:
            main_py.write_text(_MAIN_SHIM, encoding="utf-8")
            res.applied.append("__main__.py")
        except OSError as e:
            res.errors.append(f"__main__.py write failed ({e}); "
                              "venv may be read-only")

    # (2) NumPy-2 bbox .item() fix ----------------------------------------
    hl = _find_hierarchy_layout(pkg)
    if hl is None:
        res.errors.append("hierarchy_layout.py not found")
    else:
        text = hl.read_text(errors="ignore")
        patched, n = _bbox_sub(text)
        has_fixed = ".item()" in text and "vector(boundary" in text.replace(
            ".item()", "")  # a fixed call still contains vector(boundary..
        if n == 0:
            # No bare vector(boundary..) call. Either already fixed (a
            # .item() form present) or an unexpected version (neither).
            if "vector(boundary" in text or has_fixed:
                res.already.append("bbox-item")
            else:
                res.errors.append(
                    "bbox lines not found (unexpected openram version -- "
                    "not patched; verify generation manually)")
        elif check_only:
            res.errors.append("bbox-item not applied")
        else:
            try:
                bak = hl.with_suffix(".py.coresmith-bak")
                if not bak.exists():
                    bak.write_text(text, encoding="utf-8")
                hl.write_text(patched, encoding="utf-8")
                res.applied.append("bbox-item")
            except OSError as e:
                res.errors.append(f"bbox patch write failed ({e})")

    # Runnable iff __main__ importable now.
    try:
        importlib.invalidate_caches()
        res.ok = importlib.util.find_spec("openram.__main__") is not None
    except Exception:
        res.ok = False
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Idempotently repair pip OpenRAM")
    ap.add_argument("--check", action="store_true",
                    help="report status without writing")
    args = ap.parse_args(argv)
    r = patch_openram(check_only=args.check)
    print(r.summary())
    return 0 if r.ok else 1


if __name__ == "__main__":
    sys.exit(main())
