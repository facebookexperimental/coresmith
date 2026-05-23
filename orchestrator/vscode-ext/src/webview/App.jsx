import React, { Component, useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { createRoot } from 'react-dom/client';
import { ReactFlowProvider } from 'reactflow';
import GraphCanvas from './components/GraphCanvas';
import GanttTimeline from './components/GanttTimeline';
import BlockDiagramCanvas from './components/BlockDiagramCanvas';
import CollateralViewer from './components/CollateralViewer';
import NodePromptModal from './components/NodePromptModal';
import SummaryPanel from './components/SummaryPanel';
import StatusBar from './components/StatusBar';
import 'reactflow/dist/style.css';
import './styles/theme.css';
import './styles/nodes.css';
import './styles/edges.css';
import './styles/execution.css';
import './styles/timeline.css';


class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error('SoCMate ErrorBoundary caught:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{
          padding: '2rem',
          textAlign: 'center',
          color: '#e0e0e0',
          fontFamily: 'system-ui, sans-serif',
        }}>
          <h2 style={{ color: '#ff6b6b' }}>Something went wrong</h2>
          <p style={{ color: '#aaa', maxWidth: 500, margin: '1rem auto' }}>
            {this.state.error?.message || 'An unexpected error occurred.'}
          </p>
          <button
            onClick={() => this.setState({ hasError: false, error: null })}
            style={{
              padding: '0.5rem 1.5rem',
              background: '#4a9eff',
              color: '#fff',
              border: 'none',
              borderRadius: 6,
              cursor: 'pointer',
              fontSize: '0.9rem',
            }}
          >
            Retry
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

const vscode = typeof acquireVsCodeApi === 'function' ? acquireVsCodeApi() : null;
const isStandalone = !vscode;

function useTheme() {
  const [theme, setThemeState] = useState(() => {
    try {
      const stored = localStorage.getItem('socmate-theme');
      if (stored === 'dark' || stored === 'light') return stored;
    } catch {}
    return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  });

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    try { localStorage.setItem('socmate-theme', theme); } catch {}
  }, [theme]);

  const toggle = useCallback(() => {
    setThemeState((prev) => (prev === 'dark' ? 'light' : 'dark'));
  }, []);

  return [theme, toggle];
}

/** Only update state if JSON content actually changed. */
function useStableState(initial) {
  const [value, setValue] = useState(initial);
  const jsonRef = useRef(JSON.stringify(initial));
  const setStable = useCallback((newVal) => {
    const j = JSON.stringify(newVal);
    if (j !== jsonRef.current) {
      jsonRef.current = j;
      setValue(newVal);
    }
  }, []);
  return [value, setStable];
}

async function fetchGraph(graphName) {
  const res = await fetch(`/api/graph/${graphName}`);
  if (!res.ok) throw new Error(`Graph fetch failed: ${res.status}`);
  return res.json();
}

async function fetchStatus(graphName) {
  try {
    const url = graphName ? `/api/status?graph=${graphName}` : '/api/status';
    const res = await fetch(url);
    if (!res.ok) return {};
    return res.json();
  } catch {
    return {};
  }
}

async function fetchTimeline(graphName) {
  try {
    const url = graphName ? `/api/timeline?graph=${graphName}` : '/api/timeline';
    const res = await fetch(url);
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}

async function fetchTracesHttp(nodeId) {
  try {
    const res = await fetch(`/api/traces/${encodeURIComponent(nodeId)}`);
    if (!res.ok) return [];
    return res.json();
  } catch {
    return [];
  }
}

async function fetchLiveCallsHttp(nodeId) {
  try {
    const res = await fetch(`/api/live_calls/${encodeURIComponent(nodeId)}`);
    if (!res.ok) return [];
    return res.json();
  } catch {
    return [];
  }
}

async function fetchNodeTrajectoryHttp(nodeId) {
  try {
    const res = await fetch(`/api/node_trajectory/${encodeURIComponent(nodeId)}`);
    if (!res.ok) return [];
    return res.json();
  } catch {
    return [];
  }
}

async function fetchNodeDescriptions() {
  try {
    const res = await fetch('/api/node_descriptions');
    if (!res.ok) return {};
    return res.json();
  } catch {
    return {};
  }
}

async function fetchSummaryHttp(stage) {
  try {
    const res = await fetch(`/api/summary/${stage}`);
    if (!res.ok) return { stage, content: '', updated: null };
    return res.json();
  } catch {
    return { stage, content: '', updated: null };
  }
}

async function fetchSummaryCardsHttp(stage) {
  try {
    const res = await fetch(`/api/summary_cards/${stage}`);
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}

async function fetchBlockDiagramViz() {
  try {
    const res = await fetch('/api/block_diagram_viz');
    if (!res.ok) return null;
    const data = await res.json();
    // Return null if empty object (no data generated yet)
    if (!data || !data.architecture) return null;
    return data;
  } catch {
    return null;
  }
}

/** Map the current view tab to the summary stage name. */
function getSummaryStage(viewMode, graphName) {
  if (graphName === 'timeline') return 'frontend';
  if (graphName === 'block_diagram') return 'architecture';
  return graphName || 'frontend';
}

function App() {
  const [theme, toggleTheme] = useTheme();
  const [graphData, setGraphData] = useState(null);
  const [graphName, setGraphName] = useState('frontend');
  const [viewMode, setViewMode] = useState('graph'); // 'graph' | 'timeline' | 'block_diagram'
  const [selectedNode, setSelectedNode] = useState(null);
  const [blockDiagramData, setBlockDiagramData] = useState(null);
  const [modalOpen, setModalOpen] = useState(false);
  const [executionStatus, setExecutionStatus] = useStableState({});
  const [timelineData, setTimelineData] = useStableState(null);
  const [traceData, setTraceData] = useState(null);
  const [traceNodeId, setTraceNodeId] = useState(null);
  const viewModeRef = useRef(viewMode);
  const graphNameRef = useRef(graphName);

  // Summary sidebar state
  const [showSummary, setShowSummary] = useState(true);
  const [summaryContent, setSummaryContent] = useState('');
  const [summaryStage, setSummaryStage] = useState('frontend');
  const [summaryUpdated, setSummaryUpdated] = useState(null);
  const [summaryCardData, setSummaryCardData] = useStableState(null);

  // Resizable sidebar widths
  const [summaryWidth, setSummaryWidth] = useState(360);
  const [detailWidth, setDetailWidth] = useState(420);
  const draggingRef = useRef(null); // 'summary' | 'detail' | null

  // Node descriptions for the detail panel header (fetched once)
  const [nodeDescriptions, setNodeDescriptions] = useState({});
  useEffect(() => {
    if (!isStandalone) return;
    fetchNodeDescriptions().then(setNodeDescriptions);
  }, []);

  // Server-side feature flags (e.g. observer_enabled) so UI hides
  // controls for features that won't produce data.
  const [serverConfig, setServerConfig] = useState({ observer_enabled: false });
  useEffect(() => {
    if (!isStandalone) return;
    fetch('/api/server_config')
      .then((r) => r.ok ? r.json() : null)
      .then((c) => { if (c) setServerConfig(c); })
      .catch(() => {});
  }, []);


  useEffect(() => { viewModeRef.current = viewMode; }, [viewMode]);
  useEffect(() => { graphNameRef.current = graphName; }, [graphName]);

  // Update summary stage when tab changes
  useEffect(() => {
    const newStage = getSummaryStage(viewMode, graphName);
    setSummaryStage(newStage);
    // Request summary for new stage immediately
    if (isStandalone) {
      fetchSummaryHttp(newStage).then((data) => {
        setSummaryContent(data.content || '');
        setSummaryUpdated(data.updated);
      });
      fetchSummaryCardsHttp(newStage).then((data) => {
        if (data) setSummaryCardData(data);
      });
    } else {
      vscode.postMessage({ type: 'requestSummary', stage: newStage });
    }
  }, [viewMode, graphName]);

  const loadGraph = useCallback((name) => {
    if (isStandalone) {
      fetchGraph(name)
        .then((data) => { setGraphData(data); setGraphName(name); })
        .catch((err) => console.error('Graph load error:', err));
    } else {
      vscode.postMessage({ type: 'requestGraph', graphName: name });
    }
  }, []);

  useEffect(() => {
    if (isStandalone) {
      loadGraph('frontend');
      const interval = setInterval(() => {
        // Always poll status so the status indicator stays current
        const statusGraph = graphNameRef.current === 'timeline' ? 'frontend' : graphNameRef.current;
        fetchStatus(statusGraph).then(setExecutionStatus);

        // Always fetch timeline for StatusBar waiting/active counts
        fetchTimeline('').then((data) => { if (data) setTimelineData(data); });
        if (viewModeRef.current === 'block_diagram') {
          fetchBlockDiagramViz().then((data) => { if (data) setBlockDiagramData(data); });
        }
        // Poll summary for the current stage -- only update state if content changed.
        // Skip markdown summary polling on the timeline view to avoid
        // re-rendering the sidebar and resetting its scroll position.
        const stage = getSummaryStage(viewModeRef.current, graphNameRef.current);
        if (viewModeRef.current !== 'timeline') {
          fetchSummaryHttp(stage).then((data) => {
            const newContent = data.content || '';
            setSummaryContent((prev) => prev === newContent ? prev : newContent);
            setSummaryUpdated((prev) => prev === data.updated ? prev : data.updated);
            setSummaryStage((prev) => {
              const s = data.stage || stage;
              return prev === s ? prev : s;
            });
          });
        }
        // Always poll structured card data (uses useStableState so no
        // unnecessary re-renders / scroll resets).
        if (stage === 'architecture' || stage === 'frontend' || stage === 'backend') {
          fetchSummaryCardsHttp(stage).then((data) => {
            if (data) setSummaryCardData(data);
          });
        }
      }, 3000);
      return () => clearInterval(interval);
    } else {
      const handler = (event) => {
        const msg = event.data;
        if (msg.type === 'graphUpdate') {
          setGraphData(msg.data);
          if (msg.graphName) setGraphName(msg.graphName);
        }
        if (msg.type === 'executionUpdate') {
          setExecutionStatus(msg.status);
        }
        if (msg.type === 'timelineUpdate') {
          setTimelineData(msg.data);
        }
        if (msg.type === 'traceUpdate') {
          setTraceData(msg.traces);
        }
        if (msg.type === 'summaryUpdate') {
          setSummaryContent(msg.content || '');
          setSummaryUpdated(msg.updated);
          if (msg.stage) setSummaryStage(msg.stage);
        }
      };
      window.addEventListener('message', handler);
      vscode.postMessage({ type: 'requestGraph', graphName: 'frontend' });
      vscode.postMessage({ type: 'requestSummary', stage: 'frontend' });
      return () => window.removeEventListener('message', handler);
    }
  }, [loadGraph]);

  // ── Resizable drag handlers ──
  const handleDragStart = useCallback((panel) => (e) => {
    e.preventDefault();
    draggingRef.current = panel;
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';

    const onMouseMove = (ev) => {
      if (draggingRef.current === 'summary') {
        const newW = Math.max(200, Math.min(ev.clientX, 600));
        setSummaryWidth(newW);
      } else if (draggingRef.current === 'detail') {
        const newW = Math.max(280, Math.min(window.innerWidth - ev.clientX, 800));
        setDetailWidth(newW);
      }
    };

    const onMouseUp = () => {
      draggingRef.current = null;
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseup', onMouseUp);
    };

    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);
  }, []);

  // ── Status bar data derived from executionStatus + timelineData ──
  const statusBarInfo = useMemo(() => {
    const status = executionStatus || {};
    let pipelineStatus = status.status || status.pipeline_status || 'idle';
    const currentBlock = status.current_block || status.block || null;
    const completed = status.completed_blocks || status.completed || [];
    const total = status.total_blocks ?? status.total ?? null;
    const completedCount = Array.isArray(completed) ? completed.length : (typeof completed === 'number' ? completed : 0);

    const tl = timelineData || {};
    const waitingForHuman = tl.waiting_for_human || 0;
    const activeBlocks = tl.active_blocks || 0;
    const inFlight = activeBlocks || (currentBlock ? 1 : 0);
    const remaining = total != null ? Math.max(0, total - completedCount - inFlight - waitingForHuman) : null;

    if (pipelineStatus === 'idle' && (inFlight > 0 || waitingForHuman > 0)) {
      pipelineStatus = 'running';
    }

    return { pipelineStatus, currentBlock, completedCount, inFlight, waitingForHuman, remaining, total };
  }, [graphName, executionStatus, timelineData]);

  const traceNodeIdRef = useRef(null);

  const handleRequestTraces = useCallback((nodeId) => {
    // Only show loading state when switching to a different node.
    // Refreshing the same node keeps existing data so the DetailPanel
    // doesn't unmount its content and lose scroll position.
    const isRefresh = nodeId === traceNodeIdRef.current;
    traceNodeIdRef.current = nodeId;
    setTraceNodeId(nodeId);
    if (!isRefresh) {
      setTraceData(null);
    }
    if (isStandalone) {
      // Prefer the unified trajectory endpoint which interleaves LLM calls
      // with tool runs (lint/simulate/synthesize step_logs). Fall back to
      // OTel traces / live_calls if that endpoint returns nothing (older
      // runs, or nodes with no LLM activity).
      Promise.all([
        fetchNodeTrajectoryHttp(nodeId),
        fetchTracesHttp(nodeId),
        fetchLiveCallsHttp(nodeId),
      ]).then(([trajectory, traces, live]) => {
        const hasTraj = trajectory && trajectory.length > 0;
        const hasLive = live && live.length > 0;
        const hasTraces = traces && traces.length > 0;

        if (hasTraj) {
          // Convert trajectory to the same {attempt, spans, steps} shape
          // DetailPanel already consumes, but tag steps so we can render
          // LLM calls and tool runs differently.
          setTraceData(trajectory.map((a) => ({
            attempt: a.attempt,
            block: a.block,
            duration_s: a.duration_s,
            status: a.status,
            exit_event: a.exit_event,
            steps: a.steps,
            trajectory: true,
          })));
        } else if (hasLive) {
          setTraceData(live.map((g) => ({ ...g, live: true })));
        } else if (hasTraces) {
          setTraceData(traces);
        } else {
          setTraceData([]);
        }
      });
    } else {
      vscode.postMessage({ type: 'requestTraces', nodeId });
    }
  }, []);

  const handleNodeCogwheel = useCallback((node) => {
    setSelectedNode(node);
    setModalOpen(true);
  }, []);

  const handleTabSwitch = useCallback((tab) => {
    // Close the per-node modal on tab switch; otherwise it lingers showing
    // a node from the previous graph, on top of unrelated content.
    setModalOpen(false);
    if (tab === 'timeline') {
      setViewMode('timeline');
      setGraphName('timeline');
      if (isStandalone) {
        fetchTimeline('').then((data) => { if (data) setTimelineData(data); });
      } else {
        vscode.postMessage({ type: 'requestTimeline' });
      }
    } else if (tab === 'collateral') {
      setViewMode('collateral');
      setGraphName('collateral');
    } else if (tab === 'block_diagram') {
      setViewMode('block_diagram');
      setGraphName('block_diagram');
      if (isStandalone) {
        fetchBlockDiagramViz().then((data) => { setBlockDiagramData(data); });
      }
    } else {
      setViewMode('graph');
      setGraphName(tab);
      loadGraph(tab);
    }
  }, [loadGraph]);

  return (
    <div className="app-shell">
      {/* ── Header bar with view selector tabs + inline status ── */}
      <div className="app-header">
        <span className="app-title">SoCMate</span>
        <div className="graph-selector">
          <button
            className={graphName === 'architecture' ? 'active' : ''}
            onClick={() => handleTabSwitch('architecture')}
          >
            Architecture
          </button>
          <button
            className={graphName === 'frontend' ? 'active' : ''}
            onClick={() => handleTabSwitch('frontend')}
          >
            Frontend
          </button>
          <button
            className={graphName === 'backend' ? 'active' : ''}
            onClick={() => handleTabSwitch('backend')}
          >
            Backend
          </button>
          <button
            className={graphName === 'block_diagram' ? 'active' : ''}
            onClick={() => handleTabSwitch('block_diagram')}
          >
            Block Diagram
          </button>
          <button
            className={graphName === 'timeline' ? 'active' : ''}
            onClick={() => handleTabSwitch('timeline')}
          >
            Timeline
          </button>
          <button
            className={graphName === 'collateral' ? 'active' : ''}
            onClick={() => handleTabSwitch('collateral')}
          >
            Collateral
          </button>
          <span className="selector-divider" />
          <button
            className={`summary-toggle-btn ${showSummary ? 'active' : ''}`}
            onClick={() => setShowSummary(!showSummary)}
            title={showSummary ? 'Hide summary' : 'Show summary'}
          >
            Summary
          </button>
        </div>
        <StatusBar info={statusBarInfo} />
        <button
          className="theme-toggle"
          onClick={toggleTheme}
          title={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
        >
          {theme === 'dark' ? '\u2600' : '\u263E'}
        </button>
      </div>

      {/* ── Main content area with optional summary sidebar ── */}
      <div className="app-body">
        {showSummary && viewMode !== 'collateral' && (
          <>
            <SummaryPanel
              stage={summaryStage}
              content={summaryContent}
              updated={summaryUpdated}
              width={summaryWidth}
              cardData={summaryCardData}
              observerEnabled={serverConfig.observer_enabled}
            />
            <div
              className="resize-handle resize-handle-right"
              onMouseDown={handleDragStart('summary')}
            />
          </>
        )}

        {viewMode === 'timeline' ? (
          <div className="canvas-container">
            <GanttTimeline
              timelineData={timelineData}
              traceData={traceData}
              onRequestTraces={handleRequestTraces}
              graphName={graphName}
              detailWidth={detailWidth}
              onDetailResize={handleDragStart('detail')}
              nodeDescriptions={nodeDescriptions}
            />
          </div>
        ) : viewMode === 'block_diagram' ? (
          <div className="canvas-container">
            <ReactFlowProvider>
              <BlockDiagramCanvas diagramData={blockDiagramData} />
            </ReactFlowProvider>
          </div>
        ) : viewMode === 'collateral' ? (
          <div className="canvas-container">
            <CollateralViewer />
          </div>
        ) : (
          <div className="canvas-container">
            <ReactFlowProvider>
              {graphData ? (
                <GraphCanvas
                  graphData={graphData}
                  graphName={graphName}
                  executionStatus={executionStatus}
                  onNodeCogwheel={handleNodeCogwheel}
                  traceData={traceData}
                  onRequestTraces={handleRequestTraces}
                  detailWidth={detailWidth}
                  onDetailResize={handleDragStart('detail')}
                />
              ) : (
                <div className="loading">Loading graph...</div>
              )}
            </ReactFlowProvider>
          </div>
        )}
      </div>

      {modalOpen && selectedNode && (
        <NodePromptModal
          node={selectedNode}
          onClose={() => setModalOpen(false)}
        />
      )}
    </div>
  );
}

const root = createRoot(document.getElementById('root'));
root.render(
  <ErrorBoundary>
    <App />
  </ErrorBoundary>
);
