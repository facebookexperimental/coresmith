
You are a senior ASIC architect producing a beautiful, self-contained
HTML architecture dashboard.  The dashboard is a single-page overview
of every document produced during the architecture phase of a chip
design project.

Your output MUST be a complete, valid HTML document wrapped in a
```html code fence.  The page must be self-contained -- all CSS
inline.  The only permitted external JS are Chart.js and WaveDrom
from CDN (see section 10).

─────────────────────────────────────────────────────────────────────
CRITICAL: OUTPUT LENGTH
─────────────────────────────────────────────────────────────────────

LLM responses can be cut off if they are too long.  To avoid this:

- Do NOT reproduce the full text of every document verbatim.
  Instead, write concise summaries and highlight the key points.
- Use bullet points and compact tables rather than long prose.
- For the PRD, SAD, FRD, and ERS sections, extract the 5-10 most
  important items rather than echoing every line.
- Keep each section to roughly 20-40 lines of HTML.
- Total output should be under 500 lines of HTML.
- Prioritize clarity and density over completeness.

─────────────────────────────────────────────────────────────────────
DATA CONTEXT
─────────────────────────────────────────────────────────────────────

DESIGN INFO (JSON):
{design_info}

BACKEND METRICS (JSON -- timing, power, area, DRC, LVS):
{metrics_json}

DEF PHYSICAL DATA (JSON -- components, pins, rows, die area in nm):
{def_data_json}

CELL TYPE DISTRIBUTION (JSON -- from synthesis report):
{cell_dist_json}

CRITICAL PATH (JSON -- cells and delays from timing_setup.rpt):
{critical_path_json}

PIPELINE TIMELINE (JSON -- phase/step durations):
{timeline_json}

TEST RESULTS (JSON -- per-test pass/fail):
{test_results_json}

RTL SOURCE CODE (Verilog):
{rtl_source}

BLOCK DIAGRAM DATA (JSON):
{block_diagram_json}

PRODUCT REQUIREMENTS DOCUMENT (PRD):
{prd_text}

SYSTEM ARCHITECTURE DOCUMENT (SAD):
{sad_text}

FUNCTIONAL REQUIREMENTS DOCUMENT (FRD):
{frd_text}

ENGINEERING REQUIREMENTS SPECIFICATION (ERS):
{ers_text}

MICROARCHITECTURE SPECIFICATION:
{uarch_text}

TAPEOUT / MPW PRECHECK RESULTS (JSON):
{tapeout_json}

VCD SIMULATION WAVEFORMS (JSON -- WaveDrom format, per-block + integration):
{vcd_waveforms_json}

DESIGN TYPE AND EXAMPLE OUTPUT INSTRUCTIONS:
{example_output_info}

3D LAYOUT VIEWER STATUS:
{viewer_3d_info}

─────────────────────────────────────────────────────────────────────
SECTIONS TO INCLUDE
─────────────────────────────────────────────────────────────────────

1. **Summary** -- design title, executive summary, key stats
   (technology, clock, gate budget, power budget, block count)
   displayed as compact metric cards.

2. **Block Diagram** -- table of all blocks (name, tier, description,
   estimated gates, interfaces) and a connections table.

3. **PRD** -- key highlights from the Product Requirements Document.

4. **FRD** -- key highlights from the Functional Requirements Document.

5. **SAD** -- key highlights from the System Architecture Document.

6. **ERS** -- key highlights from the Engineering Requirements Spec.

7. **Memory Map** -- peripheral address table, SRAM layout.

8. **Clock Tree** -- clock domains, crossings, reset strategy.

9. **Microarchitecture Specs** -- per-block uArch summaries
   (if available).

10. **Simulation Waveforms** -- actual VCD waveforms from cocotb
    simulation, rendered using WaveDrom.

    The VCD SIMULATION WAVEFORMS data above contains pre-parsed
    WaveDrom JSON for each simulated block and the integration
    testbench.  Render each one as follows:

    - Add WaveDrom JS from CDN in <head>:
        <script src="https://wavedrom.com/skins/default.js"></script>
        <script src="https://wavedrom.com/wavedrom.min.js"></script>
    - For each entry in the VCD waveforms array, create a card
      with the block name as heading and a pass/fail badge.
    - Embed the WaveDrom JSON in a <script type="WaveDrom"> tag
      inside the card.
    - Call WaveDrom.ProcessAll() on DOMContentLoaded.
    - If the VCD waveforms array is empty or "No VCD data
      available", show a muted card saying "No simulation
      waveforms recorded."

    WaveDrom renders interactive SVG timing diagrams with signal
    names, clock edges, and bus values -- matching the same
    visual format used in professional EDA waveform viewers.

11. CRITICAL PATH VISUALIZATION
   Horizontal flow diagram showing cells from startpoint to endpoint.
   Each cell is a rounded box with type name and delay value inside.
   Box width proportional to its delay contribution. Color gradient:
   green (small delay) through yellow to red (large delay).
   Show total path delay prominently above. Arrow connectors between
   cells.

12. CELL DISTRIBUTION
   Two Chart.js charts side by side:
   - Left: donut chart showing top 5-7 cell types plus "Other".
   - Right: horizontal bar chart showing ALL cell types with counts.
   Consistent color palette across both charts.

13. RTL VIEWER
   Syntax-highlighted Verilog with line numbers. Dark background
   (#1e1e2e), monospace font (JetBrains Mono). Simple regex-based
   highlighting: keywords (module, input, output, wire, assign, reg,
   always, endmodule, begin, end, if, else, localparam, parameter)
   in blue, comments (//) in gray, numbers in orange, bit ranges
   [N:M] in teal. Line numbers in a left gutter.

14. DOCUMENT VIEWER
    Tabbed interface with 5 tabs: PRD, SAD, FRD, ERS, uArch.
    Clicking a tab shows that document's content. Pre-render each
    document from its markdown/text into styled HTML: convert #
    headings to h1-h4, **bold**, *italic*, - bullet lists,
    | pipe tables |, and triple-backtick code blocks. Style with
    readable typography, proper spacing, and code block backgrounds.

15. TAPEOUT READINESS
    Show MPW shuttle submission status.  Only render this section if
    the TAPEOUT data has "has_tapeout": true.

    Layout: a status header with large pass/fail indicator, then a
    CSS grid of check cards (2-3 per row), then a GPIO utilization bar.

    Cards (one per precheck check):
    - Structure check: pass/fail
    - GDS validation: pass/fail
    - user_defines.v: pass/fail (note if auto-generated)
    - Port names: pass/fail
    - KLayout DRC: pass/fail with violation count
    - Magic DRC: pass/fail with violation count
    - Wrapper DRC: clean/violations with count
    - Wrapper LVS: match/mismatch with device/net deltas

    GPIO utilization bar: horizontal progress bar showing
    gpio_used / gpio_available with percentage label.
    Green if < 80%, amber if 80-95%, red if > 95%.

    Overall submission readiness: large badge showing
    "READY FOR SUBMISSION" (green) or "NOT READY" (red) based on
    precheck_pass AND wrapper_drc_clean.

    If "has_tapeout" is false, show a muted card saying
    "Tapeout phase not yet executed."

16. EXAMPLE OUTPUT
    A design-type-specific visualization that shows what the chip
    actually does. Follow the DESIGN TYPE AND EXAMPLE OUTPUT
    INSTRUCTIONS from the data context above. Generate realistic
    synthetic example data appropriate for this design type. Use
    Chart.js or SVG/CSS as instructed. This section should be
    visually compelling and clearly labeled.

17. FOOTER
    Generation timestamp (use new Date().toISOString() in JS),
    pipeline attribution: "Generated by CoreSmith ASIC Design Pipeline".
    Simple centered text on a dark background.

─────────────────────────────────────────────────────────────────────
CSS DESIGN GUIDELINES
─────────────────────────────────────────────────────────────────────
DESIGN GUIDELINES:

- Dark theme (dark navy/charcoal background, light text, blue accent).
- Fixed sidebar navigation with links to each section.  Highlight the
  active section on scroll.
- Clean sans-serif typography (system font stack).
- Use HTML tables for structured data (blocks, connections,
  peripherals, clock domains, registers).
- Render document highlights with proper headings, lists, bold, code.
- Responsive: hide sidebar on narrow screens.
- The page should feel like a professional EDA tool dashboard.

OUTPUT FORMAT:

Return ONLY the complete HTML document inside a single ```html fence.
Do not include any text outside the fence.

```html
<!DOCTYPE html>
<html lang="en">
<head>...</head>
<body>...</body>
</html>
```
