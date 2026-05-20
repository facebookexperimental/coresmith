import React from 'react';
import { Handle, Position } from 'reactflow';
import { getPersona, personaAvatarUrl } from '../utils/personas';

export default function ActivityNode({ data, selected }) {
  const targetPos = data.direction === 'RIGHT' ? Position.Left : Position.Top;
  const sourcePos = data.direction === 'RIGHT' ? Position.Right : Position.Bottom;
  const info = getPersona(data.graphName, data.id);
  const persona = info?.persona;
  const avatarUrl = persona
    ? personaAvatarUrl(persona, info.avatarStyle, 43)
    : null;

  return (
    <div className={`workflow-node activity-node ${selected ? 'selected' : ''}`}>
      <Handle type="target" position={targetPos} />
      <div className="node-header activity-header">
        {!persona && <span className="node-icon">⚡</span>}
        <span className="node-title">{data.label}</span>
        {data.status === 'running' && <span className="active-badge">Working</span>}
        <button
          className="cogwheel-btn"
          onClick={(e) => { e.stopPropagation(); data.onCogwheel?.(data); }}
          title="View details"
        >
          ⚙
        </button>
      </div>
      <div className="node-body">
        <div className="node-body-inner">
          {avatarUrl && (
            <div className="node-persona">
              <img
                className="node-avatar-body"
                src={avatarUrl}
                alt=""
                width={43}
                height={43}

              />
              <span className="node-persona-title">{persona}</span>
            </div>
          )}
          <div className="node-body-content">
            {data.description && (
              <div className="node-description">{data.description}</div>
            )}
            {data.uses_interrupt && (
              <span className="interrupt-badge">interrupt()</span>
            )}
          </div>
        </div>
      </div>
      <Handle type="source" position={sourcePos} />
    </div>
  );
}
