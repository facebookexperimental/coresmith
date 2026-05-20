import React from 'react';
import { getBezierPath, EdgeLabelRenderer, BaseEdge } from 'reactflow';

const STYLE_CONFIG = {
  flow: { stroke: '#22DD22', strokeWidth: 3, labelColor: '#22DD22', labelBg: '#f0fff0' },
  fail: { stroke: '#FF1111', strokeWidth: 3, labelColor: '#FF1111', labelBg: '#fff0f0' },
  retry: { stroke: '#FF6666', strokeWidth: 3, strokeDasharray: '5 5', animated: true, labelColor: '#FF6600', labelBg: '#fff8f0' },
  agent: { stroke: '#1111FF', strokeWidth: 3, strokeDasharray: '6 3', labelColor: '#1111FF', labelBg: '#f0f0ff' },
};

export default function NamedEdge({
  id,
  sourceX, sourceY,
  targetX, targetY,
  sourcePosition, targetPosition,
  data,
  markerEnd,
}) {
  const edgeStyle = data?.style || 'flow';
  const label = data?.label;
  const config = STYLE_CONFIG[edgeStyle] || STYLE_CONFIG.flow;

  const [edgePath, labelX, labelY] = getBezierPath({
    sourceX, sourceY, targetX, targetY,
    sourcePosition, targetPosition,
  });

  return (
    <>
      <BaseEdge
        id={id}
        path={edgePath}
        markerEnd={markerEnd}
        style={{
          stroke: config.stroke,
          strokeWidth: config.strokeWidth,
          strokeDasharray: config.strokeDasharray,
        }}
        className={config.animated ? 'animated-edge' : ''}
      />
      {label && (
        <EdgeLabelRenderer>
          <div
            className={`edge-label-badge ${edgeStyle}`}
            style={{
              position: 'absolute',
              transform: `translate(-50%, -50%) translate(${labelX}px, ${labelY}px)`,
              color: config.labelColor,
              backgroundColor: config.labelBg,
              border: `1px solid ${config.stroke}`,
              pointerEvents: 'none',
            }}
          >
            {label}
          </div>
        </EdgeLabelRenderer>
      )}
    </>
  );
}
