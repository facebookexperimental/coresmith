#!/usr/bin/env python3
"""Load every major webview view in headless Chromium, save screenshots and
fail on console / page errors.

Usage:
    python3 scripts/screenshot_views.py --url http://127.0.0.1:3000 \
        --out /tmp/shots --prefix h264 [--block cavlc_macroblock_encoder]

Requires the Python ``playwright`` package with a Chromium build installed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:3000")
    ap.add_argument("--out", default="screenshots")
    ap.add_argument("--prefix", default="run")
    ap.add_argument("--block", default=None, help="block to open in the Blocks view")
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height", type=int, default=1000)
    ap.add_argument("--theme", default="light", choices=("light", "dark"))
    args = ap.parse_args()

    from playwright.sync_api import sync_playwright

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    shots: list[str] = []

    def shot(page, name):
        path = out / f"{args.prefix}-{name}.png"
        page.screenshot(path=str(path), full_page=False)
        shots.append(str(path))
        print(f"  saved {path}")

    def settle(page, ms=1200):
        page.wait_for_timeout(ms)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport={"width": args.width, "height": args.height},
                                  color_scheme="dark" if args.theme == "dark" else "light")
        page = ctx.new_page()
        page.on("console", lambda m: errors.append(f"console.{m.type}: {m.text}") if m.type in ("error",) else None)
        page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
        page.on("requestfailed", lambda r: errors.append(f"requestfailed: {r.url} {r.failure}"))
        page.on("response", lambda r: errors.append(f"http {r.status}: {r.url}") if r.status >= 500 else None)

        page.add_init_script(f"localStorage.setItem('coresmith-theme', '{args.theme}')")
        page.goto(args.url, wait_until="networkidle")
        settle(page, 1500)
        shot(page, "overview")
        # Scroll the overview to the decisions section.
        page.evaluate("() => { const el = document.getElementById('ov-decisions'); if (el) el.scrollIntoView(); }")
        settle(page, 600)
        shot(page, "overview-decisions")

        # Blocks -> trajectory
        page.get_by_role("button", name="Blocks", exact=True).click()
        settle(page, 1500)
        if args.block:
            nav = page.locator(".rv-nav-item", has_text=args.block.replace("_", " "))
            if nav.count():
                nav.first.click()
                settle(page, 1500)
        shot(page, "blocks-trajectory")
        # Open the first tool run if present
        tool = page.locator(".rv-step-tool_run")
        if tool.count():
            tool.first.click()
            settle(page, 1500)
            shot(page, "blocks-trajectory-toolrun")
        # Open a node outcome
        seg = page.locator(".rv-seg-head")
        if seg.count():
            seg.first.click()
            settle(page, 800)
            shot(page, "blocks-trajectory-node")

        # Blocks -> design
        page.get_by_role("tab", name="Design & results").click()
        settle(page, 2500)
        shot(page, "blocks-design-summary")
        for sec, name in (("sec-rtl", "blocks-design-rtl"), ("sec-sim", "blocks-design-sim"),
                          ("sec-synth", "blocks-design-synth"), ("sec-timing", "blocks-design-timing"),
                          ("sec-issues", "blocks-design-issues")):
            page.evaluate(f"() => {{ const el = document.getElementById('{sec}'); if (el) el.scrollIntoView(); }}")
            settle(page, 900)
            shot(page, name)

        # LLM call drawer from the overview decisions (if any call link exists)
        page.get_by_role("button", name="Overview", exact=True).click()
        settle(page, 1200)
        link = page.locator("#ov-decisions .rv-link", has_text="call #")
        if link.count():
            link.first.click()
            settle(page, 1800)
            shot(page, "call-drawer")
            page.keyboard.press("Escape")
            settle(page, 300)

        # Timeline + detail panel
        page.get_by_role("button", name="Timeline", exact=True).click()
        settle(page, 1500)
        shot(page, "timeline")
        segs = page.locator(".gantt-segment")
        if segs.count():
            # click a Generate RTL segment if there is one, else the first
            target = page.locator(".gantt-segment", has_text="Generate RTL")
            (target.first if target.count() else segs.first).click()
            settle(page, 3000)
            shot(page, "timeline-detail")

        # Graph
        page.get_by_role("button", name="Frontend", exact=True).click()
        settle(page, 2500)
        shot(page, "graph-frontend")

        # Block diagram + collateral
        page.get_by_role("button", name="Block Diagram", exact=True).click()
        settle(page, 2000)
        shot(page, "block-diagram")
        page.get_by_role("button", name="Collateral", exact=True).click()
        settle(page, 1500)
        shot(page, "collateral")

        browser.close()

    # Requests aborted by navigation are not real failures.
    real = [e for e in errors if "net::ERR_ABORTED" not in e]
    print(json.dumps({"screenshots": shots, "errors": real}, indent=2))
    if real:
        print(f"FAILED: {len(real)} browser error(s)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
