import React, { useState, useRef, useCallback, useEffect, useLayoutEffect } from 'react';
import DetailPanel from './DetailPanel';

const SEGMENT_COLORS = {
  // Pipeline stages
  'Init Block':              { bg: '#94a3b8', border: '#64748b' },
  'Generate RTL':            { bg: '#60a5fa', border: '#3b82f6' },
  'Lint Check':              { bg: '#2dd4bf', border: '#14b8a6' },
  'Generate Testbench':      { bg: '#818cf8', border: '#6366f1' },
  'Simulate':                { bg: '#34d399', border: '#10b981' },
  'Synthesize':              { bg: '#a78bfa', border: '#8b5cf6' },
  'Diagnose Failure':        { bg: '#fb923c', border: '#f97316' },
  'Decide Route':            { bg: '#fbbf24', border: '#f59e0b' },
  'Advance Block':           { bg: '#94a3b8', border: '#64748b' },
  'Review Uarch Spec':       { bg: '#f87171', border: '#dc2626' },
  'Generate Uarch Spec':     { bg: '#93c5fd', border: '#3b82f6' },
  'Ask Human':               { bg: '#f87171', border: '#dc2626' },
  'Block Done':              { bg: '#94a3b8', border: '#64748b' },
  'Increment Attempt':       { bg: '#d4d4d8', border: '#a1a1aa' },
  // Architecture stages
  'Block Diagram':           { bg: '#f472b6', border: '#ec4899' },
  'Memory Map':              { bg: '#c084fc', border: '#a855f7' },
  'Clock Tree':              { bg: '#22d3ee', border: '#06b6d4' },
  'Register Spec':           { bg: '#fb923c', border: '#f97316' },
  'Constraint Check':        { bg: '#facc15', border: '#eab308' },
  'Finalize Architecture':   { bg: '#818cf8', border: '#6366f1' },
  'Architecture Complete':   { bg: '#4ade80', border: '#22c55e' },
  'Constraint Iteration':    { bg: '#d4d4d8', border: '#a1a1aa' },
  'Escalate PRD':            { bg: '#fb923c', border: '#f97316' },
  'Escalate Diagram':        { bg: '#fb923c', border: '#f97316' },
  'Escalate Constraints':    { bg: '#fb923c', border: '#f97316' },
  'Escalate Exhausted':      { bg: '#fb923c', border: '#f97316' },
  'Final Review':            { bg: '#fb923c', border: '#f97316' },
  'Abort':                   { bg: '#ef4444', border: '#dc2626' },
  // Legacy aliases
  'Check Constraints':       { bg: '#facc15', border: '#eab308' },
  'Benchmark':               { bg: '#4ade80', border: '#22c55e' },
  'PDK Characterize':        { bg: '#2dd4bf', border: '#14b8a6' },
  'Finalize':                { bg: '#818cf8', border: '#6366f1' },
};

const DEFAULT_COLOR = { bg: '#cbd5e1', border: '#94a3b8' };

const STATUS_LABELS = {
  done: 'Completed',
  failed: 'Failed',
  running: 'Running',
  waiting: 'Waiting for Human',
};

const HITL_NODES = new Set([
  'Review Uarch Spec', 'Ask Human',
  'Escalate PRD', 'Escalate Diagram', 'Escalate Constraints', 'Escalate Exhausted',
  'Final Review',
]);

function getSegColor(nodeName) {
  return SEGMENT_COLORS[nodeName] || DEFAULT_COLOR;
}

function formatDuration(seconds) {
  if (seconds == null) return '--';
  if (seconds === 0) return '< 1ms';
  if (seconds < 1) return `${Math.round(seconds * 1000)}ms`;
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return `${m}m ${s}s`;
}

function formatTime(ts, startTs) {
  const offset = ts - startTs;
  return formatDuration(offset);
}

function generateTicks(totalDuration) {
  if (totalDuration <= 0) return [];
  const targets = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600];
  let interval = targets.find((t) => totalDuration / t <= 12) || Math.ceil(totalDuration / 10);
  const ticks = [];
  for (let t = 0; t <= totalDuration; t += interval) {
    ticks.push(t);
  }
  return ticks;
}

function Tooltip({ segment, style }) {
  const color = getSegColor(segment.node);
  return (
    <div className="gantt-tooltip" style={style}>
      <div className="gantt-tooltip-header" style={{ borderLeftColor: color.bg }}>
        {segment.node}
      </div>
      <div className="gantt-tooltip-body">
        <div className="gantt-tooltip-row">
          <span className="gantt-tooltip-label">Status</span>
          <span className={`gantt-tooltip-status status-${segment.status}`}>
            {STATUS_LABELS[segment.status] || segment.status}
          </span>
        </div>
        <div className="gantt-tooltip-row">
          <span className="gantt-tooltip-label">Duration</span>
          <span>{formatDuration(segment.duration_s)}</span>
        </div>
        <div className="gantt-tooltip-row">
          <span className="gantt-tooltip-label">Attempt</span>
          <span>#{segment.attempt}</span>
        </div>
        {segment.category && (
          <div className="gantt-tooltip-row">
            <span className="gantt-tooltip-label">Category</span>
            <span className="gantt-tooltip-category">{segment.category}</span>
          </div>
        )}
        {segment.chars != null && (
          <div className="gantt-tooltip-row">
            <span className="gantt-tooltip-label">Output</span>
            <span>{segment.chars.toLocaleString()} chars</span>
          </div>
        )}
        {segment.gate_count != null && (
          <div className="gantt-tooltip-row">
            <span className="gantt-tooltip-label">Gates</span>
            <span>{segment.gate_count.toLocaleString()}</span>
          </div>
        )}
      </div>
    </div>
  );
}

function GanttSegment({ segment, left, width, onHover, onLeave, onClick, isSelected }) {
  const color = getSegColor(segment.node);
  const isFailed = segment.status === 'failed';
  const isRunning = segment.status === 'running';
  const isWaiting = segment.status === 'waiting';
  const showLabel = width > 5;

  const segStyle = {
    left: `${left}%`,
    width: `${Math.max(width, 0.3)}%`,
    backgroundColor: isFailed ? '#fecaca' : color.bg,
    borderColor: isFailed ? '#ef4444' : isWaiting ? '#dc2626' : color.border,
  };

  const cls = [
    'gantt-segment',
    isRunning && 'gantt-segment-running',
    isFailed && 'gantt-segment-failed',
    isWaiting && 'gantt-segment-waiting',
    isSelected && 'gantt-segment-selected',
  ].filter(Boolean).join(' ');

  return (
    <div
      className={cls}
      style={segStyle}
      onMouseEnter={(e) => onHover(segment, e)}
      onMouseLeave={onLeave}
      onClick={() => onClick && onClick(segment)}
    >
      {showLabel && (
        <span className="gantt-segment-label" style={{ color: isFailed ? '#991b1b' : '#fff' }}>
          {isWaiting ? '\u26A0 ' : ''}{segment.node}
        </span>
      )}
    </div>
  );
}

function GanttRow({ block, toPercent, effectiveEnd, onSegmentClick, selectedSegKey }) {
  const [tooltip, setTooltip] = useState(null);
  const rowRef = useRef(null);
  const allSegments = block.attempts.flatMap((a) => a.segments);
  const maxAttempt = Math.max(...block.attempts.map((a) => a.attempt), 1);
  const isRunning = block.status === 'running';
  const isDone = block.status === 'done';
  const isFailed = block.status === 'failed';

  const handleHover = useCallback((seg, e) => {
    const rect = rowRef.current?.getBoundingClientRect();
    if (!rect) return;
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    setTooltip({ segment: seg, x, y });
  }, []);

  const handleLeave = useCallback(() => setTooltip(null), []);

  return (
    <div className="gantt-row" ref={rowRef}>
      <div className="gantt-label">
        <div className="gantt-block-info">
          <span className="gantt-block-name">{block.name.replace(/_/g, ' ')}</span>
          <div className="gantt-block-meta">
            {block.tier && <span className="gantt-tier-badge">T{block.tier}</span>}
            {maxAttempt > 1 && (
              <span className="gantt-retry-count" title={`${maxAttempt} attempts`}>
                {maxAttempt}x
              </span>
            )}
            <span
              className={[
                'gantt-status-indicator',
                isDone && 'done',
                isFailed && 'failed',
                isRunning && 'running',
              ].filter(Boolean).join(' ')}
            />
          </div>
        </div>
      </div>

      <div className="gantt-track">
        {/* Attempt boundary markers */}
        {block.attempts.slice(1).map((a) => {
          const firstSeg = a.segments[0];
          if (!firstSeg) return null;
          return (
            <div
              key={`retry-${a.attempt}`}
              className="gantt-retry-marker"
              style={{ left: `${toPercent(firstSeg.start_ts)}%` }}
            >
              <span className="gantt-retry-label">#{a.attempt}</span>
            </div>
          );
        })}

        {/* Segment bars */}
        {allSegments.map((seg, i) => {
          const segKey = `${block.name}:${seg.node}:${seg.attempt}`;
          const segLeft = toPercent(seg.start_ts);
          const segEnd = seg.end_ts ? toPercent(seg.end_ts) : toPercent(effectiveEnd);
          return (
            <GanttSegment
              key={`${seg.node}-${seg.attempt}-${i}`}
              segment={seg}
              left={segLeft}
              width={segEnd - segLeft}
              onHover={handleHover}
              onLeave={handleLeave}
              onClick={() => onSegmentClick && onSegmentClick(seg, block)}
              isSelected={selectedSegKey === segKey}
            />
          );
        })}

        {tooltip && (
          <Tooltip
            segment={tooltip.segment}
            style={{
              left: Math.min(tooltip.x + 12, 400),
              top: tooltip.y - 10,
            }}
          />
        )}
      </div>
    </div>
  );
}

const LEGEND_ITEMS = {
  frontend: [
    { label: 'Generate RTL', color: SEGMENT_COLORS['Generate RTL'].bg },
    { label: 'Lint Check', color: SEGMENT_COLORS['Lint Check'].bg },
    { label: 'Testbench', color: SEGMENT_COLORS['Generate Testbench'].bg },
    { label: 'Simulate', color: SEGMENT_COLORS['Simulate'].bg },
    { label: 'Synthesize', color: SEGMENT_COLORS['Synthesize'].bg },
    { label: 'Diagnose', color: SEGMENT_COLORS['Diagnose Failure'].bg },
    { label: 'Init/Advance', color: SEGMENT_COLORS['Init Block'].bg },
    { label: 'Human Review', color: SEGMENT_COLORS['Review Uarch Spec'].bg },
  ],
  architecture: [
    { label: 'Block Diagram', color: SEGMENT_COLORS['Block Diagram'].bg },
    { label: 'Memory Map', color: SEGMENT_COLORS['Memory Map'].bg },
    { label: 'Clock Tree', color: SEGMENT_COLORS['Clock Tree'].bg },
    { label: 'Register Spec', color: SEGMENT_COLORS['Register Spec'].bg },
    { label: 'Constraints', color: SEGMENT_COLORS['Constraint Check'].bg },
    { label: 'Escalate', color: SEGMENT_COLORS['Escalate Diagram'].bg },
    { label: 'Finalize', color: SEGMENT_COLORS['Finalize Architecture'].bg },
  ],
  backend: [
    { label: 'Floorplan', color: '#60a5fa' },
    { label: 'Place', color: '#34d399' },
    { label: 'CTS', color: '#818cf8' },
    { label: 'Route', color: '#a78bfa' },
    { label: 'DRC/LVS', color: '#2dd4bf' },
    { label: 'Timing', color: '#fbbf24' },
  ],
};

const GRAPH_TITLES = {
  frontend: 'Pipeline Execution Timeline',
  architecture: 'Architecture Execution Timeline',
  backend: 'Backend Execution Timeline',
  all: 'Execution Timeline',
};

function Legend({ graphType }) {
  const items = LEGEND_ITEMS[graphType] || LEGEND_ITEMS.frontend;

  return (
    <div className="gantt-legend">
      {items.map((item) => (
        <div key={item.label} className="gantt-legend-item">
          <div className="gantt-legend-swatch" style={{ backgroundColor: item.color }} />
          <span>{item.label}</span>
        </div>
      ))}
      <div className="gantt-legend-divider" />
      <div className="gantt-legend-item">
        <div className="gantt-legend-swatch gantt-legend-running" />
        <span>Running</span>
      </div>
      <div className="gantt-legend-item">
        <div className="gantt-legend-swatch gantt-legend-failed" />
        <span>Failed</span>
      </div>
      <div className="gantt-legend-item">
        <div className="gantt-legend-swatch gantt-legend-retry" />
        <span>Retry</span>
      </div>
    </div>
  );
}

export default function GanttTimeline({ timelineData, traceData, onRequestTraces, graphName, detailWidth, onDetailResize }) {
  const [now, setNow] = useState(Date.now() / 1000);
  const [selectedSeg, setSelectedSeg] = useState(null);
  const [selectedSegKey, setSelectedSegKey] = useState(null);

  // Drag-to-zoom state: null means full view, otherwise { start, end } as fractions [0,1]
  const [zoomRange, setZoomRange] = useState(null);
  const [dragState, setDragState] = useState(null); // { startX, trackWidth, anchorFrac }
  const chartRef = useRef(null);
  const chartScrollRef = useRef(0);

  // Only tick the clock when blocks are actively processing (not just waiting for human)
  const hasActiveWork = timelineData?.blocks?.some((b) => {
    if (b.status !== 'running') return false;
    const segs = b.attempts?.flatMap((a) => a.segments) ?? [];
    const hasOpenWaiting = segs.some((s) => !s.end_ts && s.status === 'waiting');
    const hasOpenRunning = segs.some((s) => !s.end_ts && s.status === 'running');
    return !hasOpenWaiting || hasOpenRunning;
  }) ?? false;

  // Preserve chart scroll position across re-renders
  const handleChartScroll = useCallback(() => {
    if (chartRef.current) {
      chartScrollRef.current = chartRef.current.scrollTop;
    }
  }, []);

  useLayoutEffect(() => {
    const el = chartRef.current;
    if (el && chartScrollRef.current > 0) {
      el.scrollTop = Math.min(chartScrollRef.current, el.scrollHeight - el.clientHeight);
    }
  });

  useEffect(() => {
    if (!hasActiveWork) return;
    const interval = setInterval(() => setNow(Date.now() / 1000), 1000);
    return () => clearInterval(interval);
  }, [hasActiveWork]);

  const handleSegmentClick = useCallback((segment, block) => {
    const key = `${block.name}:${segment.node}:${segment.attempt}`;
    if (selectedSegKey === key) {
      setSelectedSeg(null);
      setSelectedSegKey(null);
      return;
    }
    const isHITL = HITL_NODES.has(segment.node);
    setSelectedSeg({
      label: segment.node,
      id: segment.node,
      type: isHITL ? 'human_review' : 'activity',
      uses_interrupt: isHITL,
      description: isHITL
        ? 'Pauses for human review. Click to see pending specs.'
        : undefined,
      function: segment.node.toLowerCase().replace(/\s+/g, '_') + '_node',
      status: segment.status,
      block: block.name,
      attempt: segment.attempt,
      duration_s: segment.duration_s,
      metadata: segment.metadata || null,
    });
    setSelectedSegKey(key);
    if (onRequestTraces) {
      onRequestTraces(segment.node);
    }
  }, [selectedSegKey, onRequestTraces]);

  const handleClosePanel = useCallback(() => {
    setSelectedSeg(null);
    setSelectedSegKey(null);
  }, []);

  // ── Drag-to-zoom on the chart area (X axis only) ──
  const handleChartMouseDown = useCallback((e) => {
    // Only respond to left button on the track area, not on segments
    if (e.button !== 0) return;
    if (e.target.closest('.gantt-segment') || e.target.closest('.gantt-label')) return;

    const track = e.currentTarget.querySelector('.gantt-time-track') || e.currentTarget;
    const rect = track.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const frac = Math.max(0, Math.min(1, x / rect.width));

    setDragState({ startX: e.clientX, trackLeft: rect.left, trackWidth: rect.width, anchorFrac: frac });
  }, []);

  useEffect(() => {
    if (!dragState) return;

    const onMouseMove = (e) => {
      const { trackLeft, trackWidth, anchorFrac } = dragState;
      const currentFrac = Math.max(0, Math.min(1, (e.clientX - trackLeft) / trackWidth));
      const lo = Math.min(anchorFrac, currentFrac);
      const hi = Math.max(anchorFrac, currentFrac);

      // Show selection overlay
      setDragState((prev) => prev ? { ...prev, selLo: lo, selHi: hi } : null);
    };

    const onMouseUp = (e) => {
      if (dragState.selLo != null && dragState.selHi != null) {
        const lo = dragState.selLo;
        const hi = dragState.selHi;
        // Only zoom if selection is meaningful (>1% of width)
        if (hi - lo > 0.01) {
          // Map selection fractions back through current zoom
          const curStart = zoomRange ? zoomRange.start : 0;
          const curEnd = zoomRange ? zoomRange.end : 1;
          const span = curEnd - curStart;
          setZoomRange({
            start: curStart + lo * span,
            end: curStart + hi * span,
          });
        }
      }
      setDragState(null);
    };

    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);
    return () => {
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseup', onMouseUp);
    };
  }, [dragState, zoomRange]);

  const handleResetZoom = useCallback(() => {
    setZoomRange(null);
  }, []);

  if (!timelineData || !timelineData.blocks || timelineData.blocks.length === 0) {
    return (
      <div className="gantt-empty">
        <div className="gantt-empty-icon">&#x23F1;</div>
        <div className="gantt-empty-title">No execution data yet</div>
        <div className="gantt-empty-sub">
          Start a run to see the execution timeline here.
        </div>
      </div>
    );
  }

  const { blocks, pipeline_start, pipeline_end } = timelineData;
  const hasRunning = hasActiveWork;
  const effectiveEnd = hasActiveWork
    ? now
    : pipeline_end || blocks.reduce((latest, b) => {
        for (const a of b.attempts || []) {
          for (const s of a.segments || []) {
            const ts = s.end_ts || s.start_ts;
            if (ts > latest) latest = ts;
          }
        }
        return latest;
      }, pipeline_start);
  const totalDuration = effectiveEnd - pipeline_start;

  // Compute visible time window based on zoom
  const zoomStart = zoomRange ? pipeline_start + zoomRange.start * totalDuration : pipeline_start;
  const zoomEnd = zoomRange ? pipeline_start + zoomRange.end * totalDuration : effectiveEnd;
  const zoomDuration = zoomEnd - zoomStart;

  const toPercent = (ts) => {
    if (zoomDuration <= 0) return 0;
    return ((ts - zoomStart) / zoomDuration) * 100;
  };

  const ticks = generateTicks(zoomDuration).map((t) => t + (zoomStart - pipeline_start)).filter((t) => t >= (zoomStart - pipeline_start) && t <= (zoomEnd - pipeline_start));
  const ticksAdjusted = generateTicks(zoomDuration);

  const completedCount = blocks.filter((b) => b.status === 'done').length;
  const failedCount = blocks.filter((b) => b.status === 'failed').length;
  const runningCount = blocks.filter((b) => b.status === 'running').length;
  const waitingCount = timelineData.waiting_for_human || 0;
  const totalRetries = blocks.reduce(
    (sum, b) => sum + Math.max(0, b.attempts.length - 1),
    0,
  );

  const panelWidth = detailWidth || 420;

  return (
    <div className="gantt-wrapper">
      {/* Main timeline area -- shrinks when detail panel opens */}
      <div className="gantt-container">
        {/* Summary header */}
        <div className="gantt-summary">
          <div className="gantt-summary-title">Execution Timeline</div>
          <div className="gantt-summary-stats">
            <div className="gantt-stat">
              <span className="gantt-stat-value">{formatDuration(totalDuration)}</span>
              <span className="gantt-stat-label">Elapsed</span>
            </div>
            <div className="gantt-stat">
              <span className="gantt-stat-value gantt-stat-done">{completedCount}</span>
              <span className="gantt-stat-label">Completed</span>
            </div>
            {runningCount > 0 && (
              <div className="gantt-stat">
                <span className="gantt-stat-value gantt-stat-running">{runningCount}</span>
                <span className="gantt-stat-label">Running</span>
              </div>
            )}
            {failedCount > 0 && (
              <div className="gantt-stat">
                <span className="gantt-stat-value gantt-stat-failed">{failedCount}</span>
                <span className="gantt-stat-label">Failed</span>
              </div>
            )}
            {waitingCount > 0 && (
              <div className="gantt-stat">
                <span className="gantt-stat-value gantt-stat-waiting">{'\u26A0'} {waitingCount}</span>
                <span className="gantt-stat-label">Waiting for Human</span>
              </div>
            )}
            {totalRetries > 0 && (
              <div className="gantt-stat">
                <span className="gantt-stat-value gantt-stat-retry">{totalRetries}</span>
                <span className="gantt-stat-label">Retries</span>
              </div>
            )}
            <div className="gantt-stat">
              <span className="gantt-stat-value">{blocks.length}</span>
              <span className="gantt-stat-label">Blocks</span>
            </div>
          </div>
        </div>

        {/* Chart area */}
        <div className="gantt-chart" ref={chartRef} onMouseDown={handleChartMouseDown} onScroll={handleChartScroll}>
          {/* Zoom controls */}
          {zoomRange && (
            <div className="gantt-zoom-bar">
              <span className="gantt-zoom-label">
                Zoomed: {formatDuration(zoomStart - pipeline_start)} &ndash; {formatDuration(zoomEnd - pipeline_start)}
              </span>
              <button className="gantt-zoom-reset" onClick={handleResetZoom}>
                Reset zoom
              </button>
            </div>
          )}

          {/* Time axis */}
          <div className="gantt-time-axis">
            <div className="gantt-label-spacer" />
            <div className="gantt-time-track">
              {ticksAdjusted.map((t) => (
                <div
                  key={t}
                  className="gantt-tick"
                  style={{ left: `${toPercent(zoomStart + t)}%` }}
                >
                  <span className="gantt-tick-label">{formatDuration((zoomStart - pipeline_start) + t)}</span>
                </div>
              ))}
              {/* Now marker when pipeline is running */}
              {hasRunning && toPercent(now) >= 0 && toPercent(now) <= 100 && (
                <div className="gantt-now-marker" style={{ left: `${toPercent(now)}%` }}>
                  <span className="gantt-now-label">now</span>
                </div>
              )}

              {/* Drag selection overlay */}
              {dragState && dragState.selLo != null && (
                <div
                  className="gantt-drag-selection"
                  style={{
                    left: `${dragState.selLo * 100}%`,
                    width: `${(dragState.selHi - dragState.selLo) * 100}%`,
                  }}
                />
              )}
            </div>
          </div>

          {/* Block rows */}
          <div className="gantt-rows">
            {blocks.map((block) => (
              <GanttRow
                key={block.name}
                block={block}
                toPercent={toPercent}
                effectiveEnd={effectiveEnd}
                onSegmentClick={handleSegmentClick}
                selectedSegKey={selectedSegKey}
              />
            ))}
          </div>
        </div>

        <Legend graphType={graphName || timelineData.graph || 'frontend'} />
      </div>

      {/* Detail panel as flex sibling so timeline rescales */}
      {selectedSeg && (
        <>
          <div
            className="resize-handle resize-handle-right"
            onMouseDown={onDetailResize}
          />
          <DetailPanel
            node={selectedSeg}
            traceData={traceData}
            onRequestTraces={onRequestTraces}
            onClose={handleClosePanel}
            width={panelWidth}
            flowLayout
          />
        </>
      )}
    </div>
  );
}
