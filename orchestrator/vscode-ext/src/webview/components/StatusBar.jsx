import React from 'react';

const STATUS_META = {
  running:     { label: 'Running',     cls: 'pill-running' },
  completed:   { label: 'Completed',   cls: 'pill-completed' },
  interrupted: { label: 'Interrupted', cls: 'pill-interrupted' },
  failed:      { label: 'Failed',      cls: 'pill-failed' },
  paused:      { label: 'Paused',      cls: 'pill-paused' },
  idle:        { label: 'Idle',        cls: 'pill-idle' },
};

export default function StatusBar({ info }) {
  if (!info) return null;

  const {
    pipelineStatus,
    currentBlock,
    completedCount,
    inFlight,
    waitingForHuman,
    total,
  } = info;

  const { label, cls } = STATUS_META[pipelineStatus] || STATUS_META.idle;

  return (
    <div className="status-bar">
      <span className={`status-pill ${cls}`}>{label}</span>

      {inFlight > 0 && (
        <span className="status-pill pill-active">{inFlight} active</span>
      )}

      {waitingForHuman > 0 && (
        <span className="status-pill pill-waiting">{waitingForHuman} waiting</span>
      )}

      {total != null && (
        <span className="status-pill pill-progress">{completedCount}/{total}</span>
      )}

      {currentBlock && (
        <span className="status-pill pill-block">{currentBlock}</span>
      )}
    </div>
  );
}
