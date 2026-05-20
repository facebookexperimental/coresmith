import React, { useMemo, useState, useCallback, useEffect, useRef } from 'react';
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  MarkerType,
  BackgroundVariant,
  useNodesState,
  useEdgesState,
} from 'reactflow';

import DecideNode from './DecideNode';
import ConditionNode from './ConditionNode';
import AgentNode from './AgentNode';
import ActivityNode from './ActivityNode';
import HumanReviewNode from './HumanReviewNode';
import TerminalNode from './TerminalNode';
import GroupNode from './GroupNode';
import NamedEdge from './NamedEdge';
import DetailPanel from './DetailPanel';
import { applyElkLayout } from '../utils/layout';

const NODE_TYPE_MAP = {
  activity: 'activityNode',
  agent: 'agentNode',
  decide: 'decideNode',
  terminal: 'terminalNode',
  internal: 'activityNode',
  human_review: 'humanReviewNode',
};

const nodeTypes = {
  activityNode: ActivityNode,
  agentNode: AgentNode,
  decideNode: DecideNode,
  conditionNode: ConditionNode,
  humanReviewNode: HumanReviewNode,
  terminalNode: TerminalNode,
  groupNode: GroupNode,
};

const edgeTypes = {
  named: NamedEdge,
};

const NODE_COLORS = {
  activity: '#32cdcd',
  agent: '#80aaff',
  decide: '#ffa500',
  terminal: '#b4b4b4',
  internal: '#c8c8c8',
  human_review: '#dc3c3c',
};

const MARKER = (color) => ({
  type: MarkerType.ArrowClosed,
  color,
  strokeWidth: 3,
  width: 30,
});

const EDGE_STROKE = {
  flow: '#22DD22',
  fail: '#FF1111',
  retry: '#FF6666',
  agent: '#1111FF',
};

const HIDDEN_NODES = new Set(['Abort']);

export default function GraphCanvas({ graphData, graphName, executionStatus, onNodeCogwheel, traceData, onRequestTraces, detailWidth, onDetailResize }) {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [selectedNode, setSelectedNode] = useState(null);
  const [layoutDone, setLayoutDone] = useState(false);
  const [direction, setDirection] = useState('RIGHT');

  // Track whether we've already done initial layout for this graphData
  const layoutGraphRef = useRef(null);

  // Build RF nodes from graphData ONLY (no executionStatus in deps).
  // This prevents layout re-triggering when status polls arrive.
  const { rfNodes, rfEdges } = useMemo(() => {
    if (!graphData) return { rfNodes: [], rfEdges: [] };

    const rfNodes = graphData.nodes
      .filter((n) => !HIDDEN_NODES.has(n.id))
      .map((n) => ({
        id: n.id,
        type: NODE_TYPE_MAP[n.type] || 'activityNode',
        data: {
          ...n,
          graphName,
          onCogwheel: onNodeCogwheel,
          direction,
        },
        position: { x: 0, y: 0 },
      }));

    const rfEdges = graphData.edges
      .filter((e) => !HIDDEN_NODES.has(e.source) && !HIDDEN_NODES.has(e.target))
      .map((e, i) => {
      const style = e.style || 'flow';
      const strokeColor = EDGE_STROKE[style] || EDGE_STROKE.flow;
      return {
        id: `e-${e.source}-${e.target}-${i}`,
        source: e.source === 'START' ? '__start__' : e.source,
        target: e.target === 'END' ? '__end__' : e.target,
        type: e.label ? 'named' : 'default',
        markerEnd: MARKER(strokeColor),
        data: { label: e.label, style },
        style: {
          stroke: strokeColor,
          strokeWidth: 3,
          ...(style === 'retry' ? { strokeDasharray: '5 5' } : {}),
          ...(style === 'agent' ? { strokeDasharray: '6 3' } : {}),
        },
        animated: style === 'retry',
      };
    });

    const nodeIds = new Set(rfNodes.map((n) => n.id));
    const filteredEdges = rfEdges.filter(
      (e) =>
        (nodeIds.has(e.source) || e.source === '__start__') &&
        (nodeIds.has(e.target) || e.target === '__end__')
    );

    return { rfNodes, rfEdges: filteredEdges };
  }, [graphData, graphName, onNodeCogwheel, direction]);

  // Apply execution status as className updates WITHOUT re-triggering layout.
  useEffect(() => {
    if (!executionStatus || Object.keys(executionStatus).length === 0) return;
    setNodes((prev) =>
      prev.map((n) => {
        const status = executionStatus[n.id];
        const cls = status ? `status-${status}` : '';
        if (n.className === cls) return n;
        return { ...n, className: cls, data: { ...n.data, status } };
      })
    );
  }, [executionStatus, setNodes]);

  // Run ELK layout (only called explicitly or on graph data change)
  const doLayout = useCallback(
    (dir) => {
      if (rfNodes.length === 0) return;
      setLayoutDone(false);

      applyElkLayout(rfNodes, rfEdges, dir || direction)
        .then((layoutedNodes) => {
          setNodes(layoutedNodes);
          setEdges(rfEdges);
          setLayoutDone(true);
        })
        .catch((err) => {
          console.error('Layout error:', err);
          const isHoriz = (dir || direction) === 'RIGHT';
          const fallback = rfNodes.map((n, i) => ({
            ...n,
            position: isHoriz ? { x: i * 280, y: 100 } : { x: 200, y: i * 120 },
          }));
          setNodes(fallback);
          setEdges(rfEdges);
          setLayoutDone(true);
        });
    },
    [rfNodes, rfEdges, direction, setNodes, setEdges]
  );

  // Auto-layout only when graphData changes (not on status polls)
  useEffect(() => {
    const key = graphData ? JSON.stringify(graphData.nodes.map((n) => n.id)) : '';
    if (key && key !== layoutGraphRef.current) {
      layoutGraphRef.current = key;
      doLayout(direction);
    }
  }, [graphData, doLayout, direction]);

  const onNodeClick = useCallback((_event, node) => {
    setSelectedNode(node.data);
    if (onRequestTraces) {
      onRequestTraces(node.data.label || node.data.id);
    }
  }, [onRequestTraces]);

  const handleAutoLayout = useCallback(() => {
    doLayout(direction);
  }, [doLayout, direction]);

  const handleDirectionToggle = useCallback(() => {
    const next = direction === 'RIGHT' ? 'DOWN' : 'RIGHT';
    setDirection(next);
  }, [direction]);

  // Re-layout when direction changes
  useEffect(() => {
    if (rfNodes.length > 0 && layoutDone) {
      doLayout(direction);
    }
  }, [direction]); // eslint-disable-line react-hooks/exhaustive-deps

  if (!layoutDone && rfNodes.length > 0) {
    return <div className="loading">Laying out graph...</div>;
  }

  return (
    <div className="graph-canvas-wrapper">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        onNodeClick={onNodeClick}
        fitView
        fitViewOptions={{ padding: 0.15 }}
        minZoom={0.02}
        maxZoom={1.5}
        nodesDraggable={true}
        nodesConnectable={false}
        elementsSelectable={true}
        proOptions={{ hideAttribution: true }}
      >
        <Controls />
        <MiniMap
          style={{ width: 170, height: 110 }}
          nodeColor={(n) => NODE_COLORS[n.data?.type] || '#aaa'}
        />
        <Background variant={BackgroundVariant.Dots} />
      </ReactFlow>

      {/* Layout toolbar -- rendered OUTSIDE ReactFlow, inside positioned wrapper */}
      <div className="layout-toolbar">
        <button className="layout-btn" onClick={handleAutoLayout} title="Re-run auto layout">
          ⟳ Layout
        </button>
        <button
          className="layout-btn direction-btn"
          onClick={handleDirectionToggle}
          title={`Switch to ${direction === 'RIGHT' ? 'top-down' : 'left-right'}`}
        >
          {direction === 'RIGHT' ? '→ LR' : '↓ TB'}
        </button>
      </div>

      {selectedNode && (
        <>
          <div
            className="resize-handle resize-handle-left"
            style={{ right: (detailWidth || 420) - 2 }}
            onMouseDown={onDetailResize}
          />
          <DetailPanel
            node={selectedNode}
            traceData={traceData}
            onRequestTraces={onRequestTraces}
            onClose={() => setSelectedNode(null)}
            width={detailWidth}
          />
        </>
      )}
    </div>
  );
}
