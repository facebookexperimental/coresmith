import React, { useState } from 'react';
import { getPersona, personaAvatarUrl } from '../utils/personas';

const TYPE_LABELS = {
  agent: 'Agent',
  activity: 'Activity',
  decide: 'Decision',
  terminal: 'Terminal',
  internal: 'Internal',
  human_review: 'Human Review',
};

export default function NodePromptModal({ node, onClose }) {
  const [activeTab, setActiveTab] = useState('prompt');

  if (!node) return null;

  const typeLabel = TYPE_LABELS[node.type] || node.type;
  const info = getPersona(node.graphName, node.id);
  const persona = info?.persona;
  const avatarUrl = persona
    ? personaAvatarUrl(persona, info.avatarStyle, 40)
    : null;

  const tabs = [
    { key: 'prompt', label: 'Prompt' },
    { key: 'config', label: 'Config' },
    { key: 'source', label: 'Source' },
  ];

  return (
    <div className="prompt-panel-mask" onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}>
      <div className="prompt-panel">
        {/* Header */}
        <div className="prompt-panel-header">
          <div className="modal-title">
            {avatarUrl ? (
              <img
                className="modal-avatar"
                src={avatarUrl}
                alt=""
                width={32}
                height={32}

              />
            ) : (
              <span className="cogwheel-icon">⚙</span>
            )}
            <span className="modal-title-text">
              <span className="modal-title-name">{node.label}</span>
              {persona && <span className="modal-persona-badge">{persona}</span>}
              <span className={`modal-type-badge type-badge ${node.type}`}>{typeLabel}</span>
            </span>
          </div>
          <button className="modal-close" onClick={onClose}>×</button>
        </div>

        {/* Description banner */}
        {node.description && (
          <div className="modal-description">
            {node.description}
          </div>
        )}

        {/* Tab bar */}
        <div className="modal-tabs">
          {tabs.map((tab) => (
            <button
              key={tab.key}
              className={`modal-tab ${activeTab === tab.key ? 'active' : ''}`}
              onClick={() => setActiveTab(tab.key)}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Body */}
        <div className="prompt-panel-body">
          {activeTab === 'prompt' && (
            <div className="prompt-tab">
              {node.prompt_file ? (
                <>
                  <div className="prompt-file-path">
                    <code>{node.prompt_file}</code>
                  </div>
                  <div className="prompt-content">
                    <pre>{node.prompt_full || 'No prompt content available.'}</pre>
                  </div>
                </>
              ) : (
                <div className="prompt-empty">
                  This node has no associated prompt file.
                  {node.type === 'internal' && ' It is a deterministic internal node.'}
                  {node.type === 'terminal' && ' It is a terminal state node.'}
                  {node.type === 'human_review' && ' It pauses for human input via interrupt().'}
                </div>
              )}
            </div>
          )}

          {activeTab === 'config' && (
            <div className="config-tab">
              <div className="config-row">
                <span className="config-label">Node Type</span>
                <span className={`config-value type-badge ${node.type}`}>{typeLabel}</span>
              </div>
              {persona && (
                <div className="config-row">
                  <span className="config-label">Persona</span>
                  <span className="config-value">{persona}</span>
                </div>
              )}
              <div className="config-row">
                <span className="config-label">Node ID</span>
                <code className="config-value">{node.id}</code>
              </div>
              <div className="config-row">
                <span className="config-label">Function</span>
                <code className="config-value">{node.function}</code>
              </div>
              <div className="config-row">
                <span className="config-label">Uses Interrupt</span>
                <span className="config-value">{node.uses_interrupt ? 'Yes (Temporal dispatch)' : 'No (in-process)'}</span>
              </div>
              {node.possible_outcomes && (
                <div className="config-row">
                  <span className="config-label">Possible Outcomes</span>
                  <div className="config-value">
                    {node.possible_outcomes.map((o) => (
                      <span key={o} className={`outcome-chip ${o}`}>{o}</span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {activeTab === 'source' && (
            <div className="source-tab">
              <div className="source-info">
                <div className="config-row">
                  <span className="config-label">Function</span>
                  <code className="config-value">{node.function}</code>
                </div>
                {node.prompt_file && (
                  <div className="config-row">
                    <span className="config-label">Prompt File</span>
                    <code className="config-value">{node.prompt_file}</code>
                  </div>
                )}
                <p className="source-hint">
                  Use Cmd+P in VS Code and search for the function name to jump to source.
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="modal-footer">
          <button className="btn btn-default" onClick={onClose}>Close</button>
        </div>
      </div>
    </div>
  );
}
