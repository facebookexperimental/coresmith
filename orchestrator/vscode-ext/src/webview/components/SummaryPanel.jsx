import React, { useMemo, useRef, useLayoutEffect, useState, useCallback } from 'react';


/**
 * Lightweight markdown-to-HTML converter.
 * Handles headers, bold, italic, code, fenced code blocks, tables, lists,
 * and horizontal rules.
 */
function markdownToHtml(md) {
  if (!md) return '';

  // Split by fenced code blocks first to protect them from inline processing
  const segments = md.split(/(```[\s\S]*?```)/g);

  return segments.map((seg) => {
    // Render fenced code blocks as <pre><code>
    if (seg.startsWith('```') && seg.endsWith('```')) {
      const match = seg.match(/```(\w*)\n?([\s\S]*?)```/);
      const lang = match?.[1] || '';
      const code = (match?.[2] || seg.slice(3, -3)).trimEnd()
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
      const langTag = lang ? `<span class="summary-code-lang">${lang}</span>` : '';
      return `<pre class="summary-code-block">${langTag}<code>${code}</code></pre>`;
    }

    // Process non-code segments with block-level markdown
    const lines = seg.split('\n');
    const html = [];
    let inTable = false;
    let inList = false;
    let listType = null;

    function closeLists() {
      if (inList) {
        html.push(listType === 'ol' ? '</ol>' : '</ul>');
        inList = false;
        listType = null;
      }
    }

    function closeTable() {
      if (inTable) {
        html.push('</tbody></table>');
        inTable = false;
      }
    }

    function inlineFormat(text) {
      return text
        .replace(/`([^`]+)`/g, '<code>$1</code>')
        .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
        .replace(/\*([^*]+)\*/g, '<em>$1</em>')
        .replace(/_([^_]+)_/g, '<em>$1</em>');
    }

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const trimmed = line.trim();

      // Empty line
      if (!trimmed) {
        closeLists();
        closeTable();
        continue;
      }

      // Table separator row (|---|---|)
      if (/^\|[\s\-:|]+\|$/.test(trimmed)) {
        continue;
      }

      // Table row
      if (trimmed.startsWith('|') && trimmed.endsWith('|')) {
        closeLists();
        const cells = trimmed.slice(1, -1).split('|').map((c) => c.trim());

        if (!inTable) {
          inTable = true;
          html.push('<table><thead><tr>');
          cells.forEach((c) => html.push(`<th>${inlineFormat(c)}</th>`));
          html.push('</tr></thead><tbody>');
          // Skip the separator row if next
          if (i + 1 < lines.length && /^\|[\s\-:|]+\|$/.test(lines[i + 1].trim())) {
            i++;
          }
        } else {
          html.push('<tr>');
          cells.forEach((c) => html.push(`<td>${inlineFormat(c)}</td>`));
          html.push('</tr>');
        }
        continue;
      }

      closeTable();

      // Headers
      const headerMatch = trimmed.match(/^(#{1,4})\s+(.+)$/);
      if (headerMatch) {
        closeLists();
        const level = headerMatch[1].length;
        html.push(`<h${level}>${inlineFormat(headerMatch[2])}</h${level}>`);
        continue;
      }

      // Horizontal rule
      if (/^[-*_]{3,}$/.test(trimmed)) {
        closeLists();
        html.push('<hr/>');
        continue;
      }

      // Unordered list
      if (/^[-*+]\s+/.test(trimmed)) {
        if (!inList || listType !== 'ul') {
          closeLists();
          html.push('<ul>');
          inList = true;
          listType = 'ul';
        }
        const content = trimmed.replace(/^[-*+]\s+/, '');
        html.push(`<li>${inlineFormat(content)}</li>`);
        continue;
      }

      // Ordered list
      const olMatch = trimmed.match(/^\d+\.\s+(.+)$/);
      if (olMatch) {
        if (!inList || listType !== 'ol') {
          closeLists();
          html.push('<ol>');
          inList = true;
          listType = 'ol';
        }
        html.push(`<li>${inlineFormat(olMatch[1])}</li>`);
        continue;
      }

      // Regular paragraph
      closeLists();
      html.push(`<p>${inlineFormat(trimmed)}</p>`);
    }

    closeLists();
    closeTable();

    return html.join('\n');
  }).join('');
}

function formatTimestamp(epoch) {
  if (!epoch) return '';
  const d = new Date(epoch * 1000);
  const hh = String(d.getHours()).padStart(2, '0');
  const mm = String(d.getMinutes()).padStart(2, '0');
  const ss = String(d.getSeconds()).padStart(2, '0');
  return `${hh}:${mm}:${ss}`;
}

// ── Collapsible Card Component ──────────────────────────────────────────────

function SummaryCard({ title, badge, defaultOpen = true, children }) {
  const [open, setOpen] = useState(defaultOpen);
  const toggle = useCallback(() => setOpen((v) => !v), []);

  return (
    <div className={`summary-card ${open ? 'summary-card-open' : ''}`}>
      <button className="summary-card-header" onClick={toggle}>
        <span className="summary-card-chevron">{open ? '\u25BC' : '\u25B6'}</span>
        <span className="summary-card-title">{title}</span>
        {badge && <span className="summary-card-badge">{badge}</span>}
      </button>
      {open && <div className="summary-card-body">{children}</div>}
    </div>
  );
}

// ── Architecture Cards ──────────────────────────────────────────────────────

function MarkdownDocCard({ content, emptyMessage }) {
  const html = useMemo(() => markdownToHtml(content), [content]);
  if (!content) {
    return <div className="summary-empty">{emptyMessage || 'Not yet generated.'}</div>;
  }
  return <div className="summary-markdown" dangerouslySetInnerHTML={{ __html: html }} />;
}

function ArchitectureCards({ cardData, updated }) {
  const {
    summary = '',
    prd_content = '',
    sad_content = '',
    frd_content = '',
    ers_content = '',
    clock_tree_content = '',
    memory_map_content = '',
  } = cardData || {};

  return (
    <>
      <SummaryCard title="Summary" defaultOpen={true}>
        <MarkdownDocCard
          content={summary}
          emptyMessage="No summary available yet. The observer will generate one as the architecture runs."
        />
      </SummaryCard>

      <SummaryCard title="PRD" defaultOpen={true}>
        <MarkdownDocCard
          content={prd_content}
          emptyMessage="PRD not yet generated. Waiting for Gather Requirements to complete."
        />
      </SummaryCard>

      <SummaryCard title="SAD" defaultOpen={false}>
        <MarkdownDocCard
          content={sad_content}
          emptyMessage="SAD not yet generated. Waiting for System Architecture phase."
        />
      </SummaryCard>

      <SummaryCard title="FRD" defaultOpen={false}>
        <MarkdownDocCard
          content={frd_content}
          emptyMessage="FRD not yet generated. Waiting for Functional Requirements phase."
        />
      </SummaryCard>

      <SummaryCard title="ERS" defaultOpen={false}>
        <MarkdownDocCard
          content={ers_content}
          emptyMessage="ERS not yet generated. Waiting for Create Documentation to complete."
        />
      </SummaryCard>

      <SummaryCard title="Clock Tree" defaultOpen={false}>
        <MarkdownDocCard
          content={clock_tree_content}
          emptyMessage="Clock tree not yet generated."
        />
      </SummaryCard>

      <SummaryCard title="Memory Map" defaultOpen={false}>
        <MarkdownDocCard
          content={memory_map_content}
          emptyMessage="Memory map not yet generated."
        />
      </SummaryCard>
    </>
  );
}

// ── Frontend Cards ──────────────────────────────────────────────────────────

function UarchSpecCard({ blockName, spec }) {
  const { full_content = '' } = spec;
  const specHtml = useMemo(() => markdownToHtml(full_content), [full_content]);

  return (
    <SummaryCard title={blockName} defaultOpen={true}>
      {full_content ? (
        <div className="summary-markdown" dangerouslySetInnerHTML={{ __html: specHtml }} />
      ) : (
        <div className="summary-empty">Spec content not available.</div>
      )}
    </SummaryCard>
  );
}

function FrontendCards({ summary, uarchSpecs, updated }) {
  const summaryHtml = useMemo(() => markdownToHtml(summary), [summary]);
  const specEntries = useMemo(
    () => Object.entries(uarchSpecs || {}).sort(([a], [b]) => a.localeCompare(b)),
    [uarchSpecs],
  );

  return (
    <>
      <SummaryCard title="Summary" defaultOpen={true}>
        {summary ? (
          <div className="summary-markdown" dangerouslySetInnerHTML={{ __html: summaryHtml }} />
        ) : (
          <div className="summary-empty">
            No summary available yet. The observer will generate one as the pipeline runs.
          </div>
        )}
      </SummaryCard>

      {specEntries.length > 0 ? (
        specEntries.map(([blockName, spec]) => (
          <UarchSpecCard key={blockName} blockName={blockName} spec={spec} />
        ))
      ) : (
        <SummaryCard title="Microarchitecture" defaultOpen={true}>
          <div className="summary-empty">
            No uArch specs generated yet. They will appear here as blocks are processed.
          </div>
        </SummaryCard>
      )}
    </>
  );
}

// ── Backend Cards ───────────────────────────────────────────────────────────

function StatCard({ label, value, unit, status }) {
  const statusClass = status === 'pass' ? 'stat-pass' : status === 'fail' ? 'stat-fail' : '';
  return (
    <div className={`backend-stat-card ${statusClass}`}>
      <div className="backend-stat-label">{label}</div>
      <div className="backend-stat-value">
        {value}
        {unit && <span className="backend-stat-unit">{unit}</span>}
      </div>
    </div>
  );
}

function SignoffBadge({ label, passed, detail }) {
  return (
    <div className={`signoff-badge ${passed ? 'signoff-pass' : 'signoff-fail'}`}>
      <span className="signoff-icon">{passed ? '\u2713' : '\u2717'}</span>
      <span className="signoff-label">{label}</span>
      {detail && <span className="signoff-detail">{detail}</span>}
    </div>
  );
}

function BlockBackendCard({ block, targetClock }) {
  const {
    name = '',
    success = false,
    gate_count,
    design_area_um2 = 0,
    utilization_pct = 0,
    max_freq_mhz,
    total_power_mw = 0,
    dynamic_power_mw = 0,
    leakage_power_mw = 0,
    wns_ns = 0,
    tns_ns = 0,
    setup_slack_ns = 0,
    hold_slack_ns = 0,
    drc_clean,
    lvs_match,
    timing_met,
    wire_length_um = 0,
    via_count = 0,
    floorplan_image = '',
    gds_image = '',
    attempts = 0,
  } = block;

  const formatNum = (n, decimals = 1) => {
    if (n === undefined || n === null) return '--';
    if (typeof n !== 'number') return String(n);
    if (Math.abs(n) >= 1e6) return (n / 1e6).toFixed(1) + 'M';
    if (Math.abs(n) >= 1e3) return (n / 1e3).toFixed(1) + 'k';
    return n.toFixed(decimals);
  };

  const fmtGates = (n) => {
    if (!n) return '--';
    return n.toLocaleString();
  };

  const timingStatus = timing_met === true ? 'pass' : timing_met === false ? 'fail' : undefined;
  const freq = max_freq_mhz || (targetClock > 0 ? targetClock : undefined);

  return (
    <SummaryCard
      title={name}
      badge={success ? 'PASSED' : `attempt ${attempts}`}
      defaultOpen={true}
    >
      <div className="backend-stats-grid">
        <StatCard
          label="Gates"
          value={fmtGates(gate_count)}
        />
        <StatCard
          label="Area"
          value={formatNum(design_area_um2, 0)}
          unit={'\u00B5m\u00B2'}
        />
        <StatCard
          label="Utilization"
          value={utilization_pct ? `${formatNum(utilization_pct, 1)}` : '--'}
          unit="%"
        />
        <StatCard
          label="Max Freq"
          value={freq ? formatNum(freq, 1) : '--'}
          unit="MHz"
          status={timingStatus}
        />
      </div>

      <div className="backend-power-row">
        <div className="backend-power-header">Power</div>
        <div className="backend-power-grid">
          <div className="backend-power-item">
            <span className="backend-power-label">Total</span>
            <span className="backend-power-value">{formatNum(total_power_mw, 3)} mW</span>
          </div>
          <div className="backend-power-item">
            <span className="backend-power-label">Dynamic</span>
            <span className="backend-power-value">{formatNum(dynamic_power_mw, 3)} mW</span>
          </div>
          <div className="backend-power-item">
            <span className="backend-power-label">Leakage</span>
            <span className="backend-power-value">{formatNum(leakage_power_mw, 4)} mW</span>
          </div>
        </div>
      </div>

      <div className="backend-timing-row">
        <div className="backend-timing-header">Timing</div>
        <div className="backend-timing-grid">
          <div className="backend-timing-item">
            <span className="backend-timing-label">WNS</span>
            <span className={`backend-timing-value ${wns_ns >= 0 ? 'timing-met' : 'timing-violated'}`}>
              {wns_ns >= 0 ? '+' : ''}{formatNum(wns_ns, 2)} ns
            </span>
          </div>
          <div className="backend-timing-item">
            <span className="backend-timing-label">TNS</span>
            <span className={`backend-timing-value ${tns_ns >= 0 ? 'timing-met' : 'timing-violated'}`}>
              {tns_ns >= 0 ? '+' : ''}{formatNum(tns_ns, 2)} ns
            </span>
          </div>
          <div className="backend-timing-item">
            <span className="backend-timing-label">Setup</span>
            <span className="backend-timing-value">{formatNum(setup_slack_ns, 2)} ns</span>
          </div>
          <div className="backend-timing-item">
            <span className="backend-timing-label">Hold</span>
            <span className="backend-timing-value">{formatNum(hold_slack_ns, 2)} ns</span>
          </div>
        </div>
      </div>

      {(wire_length_um > 0 || via_count > 0) && (
        <div className="backend-routing-row">
          {wire_length_um > 0 && (
            <span className="backend-routing-item">Wire: {formatNum(wire_length_um, 0)} \u00B5m</span>
          )}
          {via_count > 0 && (
            <span className="backend-routing-item">Vias: {via_count.toLocaleString()}</span>
          )}
        </div>
      )}

      <div className="backend-signoff-row">
        {drc_clean !== undefined && (
          <SignoffBadge label="DRC" passed={drc_clean} />
        )}
        {lvs_match !== undefined && (
          <SignoffBadge label="LVS" passed={lvs_match} />
        )}
        {timing_met !== undefined && (
          <SignoffBadge label="Timing" passed={timing_met} />
        )}
      </div>

      {(floorplan_image || gds_image) && (
        <div className="backend-images">
          {floorplan_image && (
            <div className="backend-image-card">
              <div className="backend-image-label">Floorplan (post-PnR)</div>
              <img
                src={`/api/artifacts/${floorplan_image}`}
                alt={`${name} floorplan`}
                className="backend-image"
                loading="lazy"
              />
            </div>
          )}
          {gds_image && (
            <div className="backend-image-card">
              <div className="backend-image-label">GDS Layout</div>
              <img
                src={`/api/artifacts/${gds_image}`}
                alt={`${name} GDS`}
                className="backend-image"
                loading="lazy"
              />
            </div>
          )}
        </div>
      )}
    </SummaryCard>
  );
}

function BackendCards({ cardData, updated }) {
  const {
    summary = '',
    blocks = [],
    target_clock_mhz = 0,
  } = cardData || {};

  const summaryHtml = useMemo(() => markdownToHtml(summary), [summary]);
  const passed = blocks.filter((b) => b.success).length;
  const total = blocks.length;

  return (
    <>
      {total > 0 && (
        <SummaryCard
          title="Backend Results"
          badge={`${passed}/${total} passed`}
          defaultOpen={true}
        >
          <div className="backend-overview">
            <div className="backend-overview-stat">
              <span className="backend-overview-label">Blocks</span>
              <span className="backend-overview-value">{total}</span>
            </div>
            <div className="backend-overview-stat">
              <span className="backend-overview-label">Passed</span>
              <span className="backend-overview-value backend-overview-pass">{passed}</span>
            </div>
            {total - passed > 0 && (
              <div className="backend-overview-stat">
                <span className="backend-overview-label">Failed</span>
                <span className="backend-overview-value backend-overview-fail">{total - passed}</span>
              </div>
            )}
            {target_clock_mhz > 0 && (
              <div className="backend-overview-stat">
                <span className="backend-overview-label">Target</span>
                <span className="backend-overview-value">{target_clock_mhz} MHz</span>
              </div>
            )}
          </div>
        </SummaryCard>
      )}

      {blocks.map((block) => (
        <BlockBackendCard
          key={block.name}
          block={block}
          targetClock={target_clock_mhz}
        />
      ))}

      <SummaryCard title="Summary" defaultOpen={blocks.length === 0}>
        {summary ? (
          <div className="summary-markdown" dangerouslySetInnerHTML={{ __html: summaryHtml }} />
        ) : (
          <div className="summary-empty">
            No backend summary available yet. Results will appear as blocks complete PnR, DRC, and LVS.
          </div>
        )}
      </SummaryCard>
    </>
  );
}

// ── Main Panel ──────────────────────────────────────────────────────────────

const SummaryPanel = React.memo(function SummaryPanel({
  stage, content, updated, width, cardData,
}) {
  const renderedHtml = useMemo(() => markdownToHtml(content), [content]);
  const scrollRef = useRef(null);
  const savedScrollRef = useRef(0);

  // Save scroll position before DOM update, restore after
  useLayoutEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const saved = savedScrollRef.current;
    if (saved > 0) {
      el.scrollTop = Math.min(saved, el.scrollHeight - el.clientHeight);
    }
  }, [renderedHtml, cardData]);

  const handleScroll = () => {
    if (scrollRef.current) {
      savedScrollRef.current = scrollRef.current.scrollTop;
    }
  };

  const renderCards = () => {
    if (stage === 'architecture') {
      return (
        <ArchitectureCards
          cardData={cardData || { summary: content }}
          updated={updated}
        />
      );
    }

    if (stage === 'frontend') {
      return (
        <FrontendCards
          summary={cardData ? cardData.summary : content}
          uarchSpecs={cardData ? cardData.uarch_specs : null}
          updated={updated}
        />
      );
    }

    if (stage === 'backend') {
      return (
        <BackendCards
          cardData={cardData || { summary: content }}
          updated={updated}
        />
      );
    }

    // Fallback: plain markdown
    return content ? (
      <div
        className="summary-markdown"
        dangerouslySetInnerHTML={{ __html: renderedHtml }}
      />
    ) : (
      <div className="summary-empty">
        No summary available yet. The observer LLM will generate one as the pipeline runs.
      </div>
    );
  };

  return (
    <div className="summary-sidebar" style={width ? { width, minWidth: width, maxWidth: width } : undefined}>
      <div className="summary-content" ref={scrollRef} onScroll={handleScroll}>
        {renderCards()}
      </div>

      {updated && (
        <div className="summary-footer">
          <span className="summary-updated">
            Updated {formatTimestamp(updated)}
          </span>
        </div>
      )}
    </div>
  );
});

export default SummaryPanel;
