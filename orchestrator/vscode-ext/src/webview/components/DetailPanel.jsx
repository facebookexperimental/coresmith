import React, { useState, useEffect, useCallback, useRef, useLayoutEffect } from 'react';

/* ── Helpers ─────────────────────────────────────────────── */

function formatDuration(ms) {
  if (ms == null) return '--';
  if (ms < 1) return '<1ms';
  if (ms < 1000) return `${Math.round(ms)}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  const m = Math.floor(ms / 60000);
  const s = Math.round((ms % 60000) / 1000);
  return `${m}m ${s}s`;
}

function formatTokens(count) {
  if (count == null) return null;
  const n = Number(count);
  if (isNaN(n)) return null;
  if (n >= 1000) return `${(n / 1000).toFixed(1)}k`;
  return String(n);
}

function tryParseJson(str) {
  if (typeof str !== 'string') return null;
  const trimmed = str.trim();
  if (
    (trimmed.startsWith('{') && trimmed.endsWith('}')) ||
    (trimmed.startsWith('[') && trimmed.endsWith(']'))
  ) {
    try {
      return JSON.parse(trimmed);
    } catch {
      return null;
    }
  }
  return null;
}

/* ── LLM content extraction ──────────────────────────────── */

/**
 * Extract role and content from a LangChain serialized message object.
 * Format: { lc: 1, type: "constructor", id: [..., "HumanMessage"], kwargs: { content: "..." } }
 * Returns { role, content } or null if not a LangChain message.
 */
function parseLangChainMessage(obj) {
  if (
    typeof obj !== 'object' || obj === null ||
    obj.lc == null || obj.type !== 'constructor' ||
    !Array.isArray(obj.id) || !obj.kwargs
  ) {
    return null;
  }

  const idTail = (obj.id[obj.id.length - 1] || '').toLowerCase();
  const role = idTail.includes('system') ? 'system' : 'user';

  let content = obj.kwargs.content;
  if (content == null) return null;

  if (Array.isArray(content)) {
    // Multi-part content (text + images etc.) — extract text parts
    content = content
      .filter((p) => typeof p === 'string' || (p && p.type === 'text'))
      .map((p) => (typeof p === 'string' ? p : p.text || ''))
      .join('\n\n');
  }

  if (typeof content !== 'string') content = JSON.stringify(content);
  return { role, content };
}

function extractContent(inputValue, outputValue) {
  let prompt = '';
  let systemPrompt = '';
  let response = '';

  // Normalize non-string values to JSON strings so tryParseJson can handle them
  if (inputValue && typeof inputValue !== 'string') {
    inputValue = JSON.stringify(inputValue, null, 2);
  }
  if (outputValue && typeof outputValue !== 'string') {
    outputValue = JSON.stringify(outputValue, null, 2);
  }

  // --- Parse input ---
  if (inputValue) {
    const parsed = tryParseJson(inputValue);
    if (parsed) {
      // Check if the top-level object is a single LangChain message
      const lcSingle = parseLangChainMessage(parsed);
      if (lcSingle) {
        if (lcSingle.role === 'system') {
          systemPrompt = lcSingle.content;
        } else {
          prompt = lcSingle.content;
        }
      } else {
        const msgs = parsed.messages || (Array.isArray(parsed) ? parsed : null);
        if (msgs && Array.isArray(msgs)) {
          const sysParts = [];
          const userParts = [];
          for (const m of msgs) {
            // Try LangChain serialized format first
            const lcMsg = parseLangChainMessage(m);
            if (lcMsg) {
              if (lcMsg.role === 'system') {
                sysParts.push(lcMsg.content);
              } else {
                userParts.push(lcMsg.content);
              }
            } else if (Array.isArray(m)) {
              // Tuple format: [role, content]
              const content = typeof m[1] === 'string' ? m[1] : JSON.stringify(m[1], null, 2);
              if (m[0] === 'system' || m[0] === 'SystemMessage') {
                sysParts.push(content);
              } else {
                userParts.push(content);
              }
            } else if (typeof m === 'object' && m !== null) {
              const role = (m.role || m.type || '').toLowerCase();
              const content =
                typeof m.content === 'string'
                  ? m.content
                  : JSON.stringify(m.content);
              if (role === 'system' || role === 'systemmessage') {
                sysParts.push(content);
              } else {
                userParts.push(content);
              }
            }
          }
          systemPrompt = sysParts.join('\n\n');
          prompt = userParts.join('\n\n');
        } else if (typeof parsed === 'object' && !Array.isArray(parsed)) {
          // Try extracting content from kwargs (partial LangChain format)
          if (parsed.kwargs?.content) {
            const kc = parsed.kwargs.content;
            prompt = typeof kc === 'string' ? kc : JSON.stringify(kc);
          } else {
            prompt =
              parsed.prompt || parsed.input || parsed.query || inputValue;
            if (typeof prompt !== 'string') prompt = JSON.stringify(prompt);
          }
        }
      }
    } else {
      prompt = inputValue;
    }
  }

  // --- Parse output ---
  if (outputValue) {
    const parsed = tryParseJson(outputValue);
    if (parsed) {
      // Check if the output is a single LangChain message
      const lcOut = parseLangChainMessage(parsed);
      if (lcOut) {
        response = lcOut.content;
      } else if (parsed.generations) {
        // OpenInference / LangChain generation format
        const gen = parsed.generations?.[0];
        if (Array.isArray(gen) && gen.length > 0) {
          const first = gen[0];
          // Generation item may wrap a LangChain message
          const lcGen = parseLangChainMessage(first?.message || first);
          if (lcGen) {
            response = lcGen.content;
          } else {
            response = first?.text || first?.message?.content || '';
          }
        } else if (gen?.text || gen?.message?.content) {
          response = gen.text || gen.message?.content || '';
        }
      }
      if (!response) {
        if (parsed.kwargs?.content) {
          const kc = parsed.kwargs.content;
          response = typeof kc === 'string' ? kc : JSON.stringify(kc);
        } else {
          response =
            parsed.content || parsed.text || parsed.output || outputValue;
        }
      }
      if (typeof response !== 'string') response = JSON.stringify(response);
    } else {
      response = outputValue;
    }
  }

  return { prompt, systemPrompt, response };
}

/**
 * Detect whether an LLM call is from the observer/summarizer.
 * Observer calls have distinctive system prompts and should not
 * appear in the timeline trace detail view.
 */
const _OBSERVER_SYSTEM_RE =
  /you are an? (?:ASIC architecture|RTL pipeline|physical design) observer/i;

function _isObserverCall(systemPrompt) {
  return systemPrompt && _OBSERVER_SYSTEM_RE.test(systemPrompt);
}

function extractLLMCalls(spans) {
  const calls = [];

  function walk(span) {
    // Handle streaming (in-progress) LLM calls
    if (span.attributes?.streaming || span.status === 'streaming') {
      const partial = span.attributes?.['output.value'] || '';
      calls.push({
        id: span.span_id,
        model: span.attributes?.['llm.model_name'] || 'Claude',
        streaming: true,
        duration_ms: span.duration_ms,
        response: partial,
        prompt: '',
        systemPrompt: '',
        status: 'streaming',
      });
      return;
    }

    const kind = span.attributes?.['openinference.span.kind'];
    const name = (span.name || '').toLowerCase();
    const isLLM =
      kind === 'LLM' ||
      /model|llm|chatmodel|claude|anthropic/i.test(name);

    if (
      isLLM &&
      (span.attributes?.['input.value'] || span.attributes?.['output.value'])
    ) {
      const { prompt, systemPrompt, response } = extractContent(
        span.attributes['input.value'],
        span.attributes['output.value']
      );

      // Skip observer/summarizer LLM calls -- only show actual
      // pipeline activity in the trace detail view
      if (_isObserverCall(systemPrompt)) return;

      calls.push({
        id: span.span_id,
        model: span.attributes['llm.model_name'] || 'Claude',
        promptTokens: span.attributes['llm.token_count.prompt'],
        completionTokens: span.attributes['llm.token_count.completion'],
        totalTokens: span.attributes['llm.token_count.total'],
        duration_ms: span.duration_ms,
        status: span.status,
        prompt,
        systemPrompt,
        response,
      });
    }

    (span.children || []).forEach(walk);
  }

  (spans || []).forEach(walk);
  return calls;
}

/**
 * Extract block_name from span attributes within an attempt group.
 * Multiple blocks may share the same graph node (e.g. generate_rtl is used
 * for scrambler, viterbi_decoder, etc.). This function extracts the block
 * name from span attributes or the full span name.
 */
function extractBlockName(spans) {
  for (const span of (spans || [])) {
    const bn = span.attributes?.block_name || span.attributes?.['block_name'];
    if (bn) return bn;
    // Try parsing from span name: "Generate RTL [scrambler] attempt 1"
    const m = (span.name || '').match(/\[([^\]]+)\]/);
    if (m) return m[1];
    // Check children
    for (const child of (span.children || [])) {
      const cbn = child.attributes?.block_name || child.attributes?.['block_name'];
      if (cbn) return cbn;
      const cm = (child.name || '').match(/\[([^\]]+)\]/);
      if (cm) return cm[1];
    }
  }
  return null;
}

/**
 * Re-group trace data by (block_name, attempt) when multiple blocks
 * share the same graph node. Returns an array of { key, label, attempt,
 * blockName, spans } groups for the tab bar.
 */
function regroupTraces(traceData) {
  if (!traceData || traceData.length === 0) return [];

  // Extract block names from each attempt group
  const groups = traceData.map((group) => ({
    ...group,
    blockName: extractBlockName(group.spans),
  }));

  // Check if there are multiple distinct block names
  const blockNames = new Set(groups.map((g) => g.blockName).filter(Boolean));
  const hasMultipleBlocks = blockNames.size > 1;

  if (!hasMultipleBlocks) {
    // Single block (or no block names found) -- use original attempt-only tabs
    return groups.map((g) => ({
      key: `a${g.attempt}`,
      label: groups.length === 1 ? 'Run' : `#${g.attempt}`,
      attempt: g.attempt,
      blockName: g.blockName,
      spans: g.spans,
    }));
  }

  // Multiple blocks -- create (block, attempt) tabs
  return groups.map((g) => {
    const name = (g.blockName || 'unknown').replace(/_/g, ' ');
    return {
      key: `${g.blockName || 'unknown'}:${g.attempt}`,
      label: `${name} #${g.attempt}`,
      attempt: g.attempt,
      blockName: g.blockName,
      spans: g.spans,
    };
  });
}

function getAttemptDuration(spans) {
  if (!spans || spans.length === 0) return null;
  return spans.reduce((sum, s) => sum + (s.duration_ms || 0), 0);
}

function getAttemptStatus(spans) {
  if (!spans || spans.length === 0) return 'unset';
  return spans.some((s) => s.status === 'error') ? 'error' : 'ok';
}

/* ── Markdown-to-HTML converter for LLM output ──────────── */

function markdownToHtml(md) {
  if (!md) return '';

  // Split by code fences first to protect them from inline processing
  const segments = md.split(/(```[\s\S]*?```)/g);

  return segments.map((seg) => {
    if (seg.startsWith('```') && seg.endsWith('```')) {
      const match = seg.match(/```(\w*)\n?([\s\S]*?)```/);
      const lang = match?.[1] || '';
      const code = (match?.[2] || seg.slice(3, -3)).trimEnd()
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
      const langTag = lang ? `<span class="llm-code-lang">${lang}</span>` : '';
      return `<pre class="llm-code-block">${langTag}<code>${code}</code></pre>`;
    }

    // Process line-by-line for block elements
    const lines = seg.split('\n');
    const html = [];
    let inList = false;
    let listType = null;

    function closeLists() {
      if (inList) {
        html.push(listType === 'ol' ? '</ol>' : '</ul>');
        inList = false;
        listType = null;
      }
    }

    function escapeHtml(text) {
      return text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
    }

    function inlineFmt(text) {
      return escapeHtml(text)
        .replace(/`([^`]+)`/g, '<code>$1</code>')
        .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
        .replace(/\*([^*]+)\*/g, '<em>$1</em>')
        .replace(/_([^_]+)_/g, '<em>$1</em>');
    }

    for (const line of lines) {
      const trimmed = line.trim();

      if (!trimmed) {
        closeLists();
        continue;
      }

      // Headers
      const hMatch = trimmed.match(/^(#{1,4})\s+(.+)$/);
      if (hMatch) {
        closeLists();
        const lvl = hMatch[1].length;
        html.push(`<h${lvl}>${inlineFmt(hMatch[2])}</h${lvl}>`);
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
        html.push(`<li>${inlineFmt(trimmed.replace(/^[-*+]\s+/, ''))}</li>`);
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
        html.push(`<li>${inlineFmt(olMatch[1])}</li>`);
        continue;
      }

      // Regular paragraph
      closeLists();
      html.push(`<p>${inlineFmt(trimmed)}</p>`);
    }

    closeLists();
    return html.join('\n');
  }).join('');
}

/* ── Formatted text renderer with markdown ───────────────── */

function FormattedText({ text, maxCollapsed = 2000 }) {
  const [expanded, setExpanded] = useState(false);

  if (!text) return <span className="llm-no-content">No content</span>;

  const isLong = text.length > maxCollapsed;
  const display = isLong && !expanded ? text.slice(0, maxCollapsed) : text;

  const html = markdownToHtml(display);

  return (
    <div className="llm-formatted llm-markdown">
      <div dangerouslySetInnerHTML={{ __html: html }} />
      {isLong && (
        <button
          className="llm-expand-btn"
          onClick={() => setExpanded(!expanded)}
        >
          {expanded
            ? 'Show less'
            : `Show more (${(text.length / 1000).toFixed(1)}k chars)`}
        </button>
      )}
    </div>
  );
}

/* ── Collapsible section ─────────────────────────────────── */

function Collapsible({ label, icon, defaultOpen, children, className }) {
  const [open, setOpen] = useState(defaultOpen ?? false);

  return (
    <div className={`llm-collapsible ${className || ''}`}>
      <button
        className="llm-collapsible-header"
        onClick={() => setOpen(!open)}
      >
        <span className="llm-collapsible-arrow">{open ? '\u25BC' : '\u25B6'}</span>
        {icon && <span className="llm-collapsible-icon">{icon}</span>}
        <span className="llm-collapsible-label">{label}</span>
      </button>
      {open && <div className="llm-collapsible-body">{children}</div>}
    </div>
  );
}

/* ── LLM Call Card ───────────────────────────────────────── */

function LLMCallCard({ call }) {
  const statusSymbol =
    call.status === 'ok' ? '\u2713' : call.status === 'error' ? '\u2717' : '\u2014';
  const statusCls =
    call.status === 'ok' ? 'ok' : call.status === 'error' ? 'error' : 'unset';

  const tokenLabel = formatTokens(call.totalTokens);

  return (
    <div
      className={`llm-card ${call.status === 'error' ? 'llm-card-error' : ''}`}
    >
      {/* Header bar */}
      <div className="llm-card-header">
        <span className="llm-model-name">{call.model}</span>
        <span className="llm-card-meta">
          <span className="llm-dur">{formatDuration(call.duration_ms)}</span>
          {tokenLabel && (
            <span className="llm-tok">{tokenLabel} tok</span>
          )}
          <span className={`llm-stat llm-stat-${statusCls}`}>
            {statusSymbol}
          </span>
        </span>
      </div>

      {/* System prompt (collapsed) */}
      {call.systemPrompt && (
        <Collapsible label="System Prompt" icon="\u2699" className="llm-sys">
          <FormattedText text={call.systemPrompt} />
        </Collapsible>
      )}

      {/* User prompt (collapsed) */}
      {call.prompt && (
        <Collapsible label="Prompt" icon="&#x25B6;" className="llm-usr">
          <FormattedText text={call.prompt} />
        </Collapsible>
      )}

      {/* Response (always visible) */}
      {call.response && (
        <div className="llm-response">
          <div className="llm-response-label">
            <span className="llm-response-icon">&#x25C0;</span> Response
          </div>
          <div className="llm-response-body">
            <FormattedText text={call.response} />
          </div>
        </div>
      )}

      {/* Error message */}
      {call.status === 'error' && !call.response && (
        <div className="llm-error-msg">LLM call failed</div>
      )}
    </div>
  );
}

/* ── Streaming LLM Call Card ──────────────────────────────── */

function StreamingLLMCard({ call }) {
  const responseRef = useRef(null);
  const prevLenRef = useRef(0);

  // Auto-scroll to bottom only when new content arrives
  useEffect(() => {
    if (responseRef.current && call.response) {
      const newLen = call.response.length;
      if (newLen > prevLenRef.current) {
        prevLenRef.current = newLen;
        const el = responseRef.current;
        el.scrollTop = el.scrollHeight;
      }
    }
  }, [call.response]);

  const elapsedStr = call.duration_ms != null
    ? formatDuration(call.duration_ms)
    : '--';

  const charCount = call.response ? call.response.length : 0;
  const charLabel = charCount >= 1000
    ? `${(charCount / 1000).toFixed(1)}k`
    : String(charCount);

  return (
    <div className="llm-card llm-card-streaming">
      {/* Header bar */}
      <div className="llm-card-header llm-card-header-streaming">
        <span className="llm-model-name">{call.model}</span>
        <span className="llm-card-meta">
          <span className="llm-dur">{elapsedStr}</span>
          <span className="llm-tok">{charLabel} chars</span>
          <span className="llm-streaming-badge">LIVE</span>
        </span>
      </div>

      {/* Streaming response */}
      <div className="llm-response llm-response-streaming">
        <div className="llm-response-label">
          <span className="llm-response-icon">&#x25C0;</span>
          Response
          <span className="llm-streaming-dots">
            <span>.</span><span>.</span><span>.</span>
          </span>
        </div>
        <div className="llm-response-body llm-response-body-streaming" ref={responseRef}>
          {call.response ? (
            <div className="llm-formatted llm-markdown">
              <div dangerouslySetInnerHTML={{ __html: markdownToHtml(call.response) }} />
              <span className="llm-streaming-cursor" />
            </div>
          ) : (
            <div className="llm-streaming-waiting">
              <div className="trace-spinner" style={{ width: 16, height: 16 }} />
              <span>Waiting for response...</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

/* ── Attempt summary bar ─────────────────────────────────── */

function AttemptSummary({ spans, llmCount, hasStreamingCalls }) {
  const duration = getAttemptDuration(spans);
  const status = getAttemptStatus(spans);
  const isError = status === 'error';
  const isStreaming = hasStreamingCalls;

  return (
    <div className={`llm-summary-bar ${isStreaming ? 'streaming' : isError ? 'error' : 'ok'}`}>
      <span className={`llm-summary-status ${isStreaming ? 'streaming' : isError ? 'error' : 'ok'}`}>
        {isStreaming ? '\u25CF Generating' : isError ? '\u2717 Failed' : '\u2713 Completed'}
      </span>
      <span className="llm-summary-detail">
        {!isStreaming && formatDuration(duration)}
        {llmCount > 0 && (
          <span className="llm-summary-count">
            {llmCount} LLM call{llmCount !== 1 ? 's' : ''}
            {isStreaming ? ' (1 active)' : ''}
          </span>
        )}
      </span>
    </div>
  );
}

/* ── Metadata Summary for non-LLM nodes ──────────────────── */

const META_LABELS = {
  node_count: 'Nodes',
  edge_count: 'Edges',
  validation_errors: 'Validation errors',
  path: 'Output path',
  peripheral_count: 'Peripherals',
  questions: 'Questions asked',
  round: 'Round',
  error: 'Error',
};

function NodeMetadataSummary({ node }) {
  const meta = node.metadata || {};
  const entries = Object.entries(meta).filter(([, v]) => v != null);

  return (
    <div className="trace-metadata-summary">
      <div className="trace-metadata-header">
        <span className="trace-metadata-icon">{'\u2699'}</span>
        <span>Completed without LLM calls</span>
      </div>
      {node.duration_s != null && (
        <div className="trace-metadata-duration">
          Duration: {node.duration_s < 0.001
            ? '< 1ms'
            : node.duration_s < 1
              ? `${Math.round(node.duration_s * 1000)}ms`
              : `${node.duration_s.toFixed(1)}s`}
        </div>
      )}
      {entries.length > 0 && (
        <div className="trace-metadata-table">
          {entries.map(([key, value]) => (
            <div key={key} className="trace-metadata-row">
              <span className="trace-metadata-label">
                {META_LABELS[key] || key.replace(/_/g, ' ')}
              </span>
              <span className="trace-metadata-value">
                {key === 'error' ? (
                  <span className="trace-metadata-error">{String(value)}</span>
                ) : key === 'path' ? (
                  <code className="trace-metadata-path">{String(value)}</code>
                ) : typeof value === 'number' ? (
                  value.toLocaleString()
                ) : (
                  String(value)
                )}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/* ── HITL Interrupt Panel ─────────────────────────────────── */

function HITLPanel({ node }) {
  const [interruptData, setInterruptData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    fetch('/api/interrupts')
      .then((r) => r.json())
      .then((data) => {
        if (!cancelled) {
          setInterruptData(data);
          setLoading(false);
        }
      })
      .catch(() => {
        if (!cancelled) setLoading(false);
      });

    const interval = setInterval(() => {
      fetch('/api/interrupts')
        .then((r) => r.json())
        .then((data) => {
          if (!cancelled) setInterruptData(data);
        })
        .catch(() => {});
    }, 5000);

    return () => { cancelled = true; clearInterval(interval); };
  }, [node?.id]);

  if (loading) {
    return (
      <div className="trace-loading">
        <div className="trace-spinner" />
        Loading interrupt data&hellip;
      </div>
    );
  }

  const allInterrupts = interruptData?.interrupts || [];

  // Separate pipeline vs architecture entries
  const pipelineInterrupts = allInterrupts.filter(i => i.type !== 'architecture_escalation');
  const archEscalations = allInterrupts.filter(i => i.type === 'architecture_escalation');

  // Filter architecture escalations by the selected node (if any)
  const relevantEscalations = node?.id
    ? archEscalations.filter(e => e.node === node.id)
    : archEscalations;

  // For pipeline nodes, show pipeline interrupts; for arch nodes, show escalations
  const isArchNode = node?.id && archEscalations.some(e => e.node === node.id);
  const visiblePipeline = isArchNode ? [] : pipelineInterrupts;
  const visibleArch = isArchNode ? relevantEscalations : (pipelineInterrupts.length === 0 ? archEscalations : []);

  if (visiblePipeline.length === 0 && visibleArch.length === 0) {
    return (
      <div className="trace-empty">
        <span className="trace-empty-icon">{'\u23F3'}</span>
        No blocks currently waiting at this node.
      </div>
    );
  }

  return (
    <div className="hitl-panel">
      {/* Architecture escalation entries */}
      {visibleArch.map((esc, i) => (
        <ArchEscalationCard key={`arch-${i}`} escalation={esc} singleItem={visibleArch.length === 1} />
      ))}

      {/* Pipeline HITL entries */}
      {visiblePipeline.length > 0 && (
        <>
          <div className="hitl-header">
            <span className="hitl-warning-icon">{'\u26A0'}</span>
            <span className="hitl-header-text">
              {visiblePipeline.length} block{visiblePipeline.length !== 1 ? 's' : ''} waiting for human review
            </span>
          </div>
          <div className="hitl-actions-hint">
            <strong>Actions:</strong> {visiblePipeline[0]?.supported_actions?.join(', ')}
          </div>
          <div className="hitl-actions-hint hitl-resume-hint">
            Use <code>resume_pipeline(action="approve")</code> to approve all, or <code>"skip"</code> to skip.
          </div>
          {visiblePipeline.map((intr) => (
            <div key={intr.block_name} className="hitl-block-card">
              <div className="hitl-block-header">
                <span className="hitl-block-name">{intr.block_name.replace(/_/g, ' ')}</span>
                <span className="hitl-block-type">
                  {intr.type === 'uarch_spec_review' ? 'uArch Spec Review' : 'Human Intervention'}
                </span>
              </div>
              {intr.spec_content ? (
                <Collapsible
                  label={`${intr.block_name} uArch Spec`}
                  icon={'\u{1F4DD}'}
                  defaultOpen={visiblePipeline.length === 1}
                >
                  <FormattedText text={intr.spec_content} maxCollapsed={3000} />
                </Collapsible>
              ) : (
                <div className="hitl-no-spec">No spec content available</div>
              )}
            </div>
          ))}
        </>
      )}
    </div>
  );
}

/* ── Architecture Escalation Card ─────────────────────────── */

const PHASE_LABELS = {
  prd: 'PRD Sizing Questions',
  block_diagram: 'Block Diagram Review',
  constraints: 'Constraint Violations',
  max_rounds_exhausted: 'Max Iterations Exhausted',
};

function ArchEscalationCard({ escalation: esc, singleItem }) {
  const isWaiting = esc.status === 'waiting';
  const phaseLabel = PHASE_LABELS[esc.phase] || esc.node;

  return (
    <div className="hitl-block-card">
      <div className="hitl-block-header">
        <span className="hitl-block-name">{phaseLabel}</span>
        <span className={`hitl-block-type ${isWaiting ? 'hitl-block-waiting' : 'hitl-block-done'}`}>
          {isWaiting ? 'Waiting' : 'Resolved'}
        </span>
      </div>

      <div className="esc-round-badge">Round {esc.round || 1}</div>

      {/* Phase-specific content */}
      {esc.phase === 'prd' && <PRDContent esc={esc} defaultOpen={singleItem} />}
      {esc.phase === 'constraints' && <ConstraintsContent esc={esc} defaultOpen={singleItem} />}
      {esc.phase === 'block_diagram' && <DiagramContent esc={esc} defaultOpen={singleItem} />}
      {esc.phase === 'max_rounds_exhausted' && <ExhaustedContent esc={esc} defaultOpen={singleItem} />}

      {/* Response section (for completed escalations) */}
      {esc.response && <EscalationResponse response={esc.response} />}

      {/* Actions hint (for waiting escalations) */}
      {isWaiting && esc.supported_actions?.length > 0 && (
        <div className="esc-actions">
          <strong>Available actions:</strong>{' '}
          {esc.supported_actions.map((a, i) => (
            <code key={i} className="esc-action-tag">{a}</code>
          ))}
          <div className="hitl-actions-hint hitl-resume-hint">
            Use <code>resume_architecture(action="...")</code> to respond.
          </div>
        </div>
      )}
    </div>
  );
}

function PRDContent({ esc, defaultOpen }) {
  const questions = esc.questions || [];
  const answers = esc.prd_answers || esc.ers_answers || {};

  if (questions.length === 0) {
    return <div className="esc-summary">{esc.question_count || 0} question(s) for architect review</div>;
  }

  return (
    <Collapsible
      label={`${questions.length} Sizing Question${questions.length !== 1 ? 's' : ''}`}
      icon={'\u2753'}
      defaultOpen={defaultOpen}
    >
      <div className="esc-questions-list">
        {questions.map((q, i) => (
          <div key={q.id || i} className="esc-question-row">
            <div className="esc-q-header">
              <span className="esc-q-category">{q.category || 'general'}</span>
              <span className="esc-q-text">{q.question}</span>
            </div>
            {q.options && q.options.length > 0 && (
              <div className="esc-q-options">Options: {q.options.join(' | ')}</div>
            )}
            {answers[q.id] && (
              <div className="esc-q-answer">
                <span className="esc-q-answer-label">{'\u2705'} Answer:</span> {answers[q.id]}
              </div>
            )}
          </div>
        ))}
      </div>
    </Collapsible>
  );
}

function ConstraintsContent({ esc, defaultOpen }) {
  const violations = esc.violations || [];
  const structural = esc.structural_violations || [];

  return (
    <>
      <div className="esc-summary">
        {esc.total_violations || violations.length} violation{(esc.total_violations || violations.length) !== 1 ? 's' : ''} found
        {esc.structural_count > 0 && (
          <span className="esc-structural-badge"> ({esc.structural_count} structural)</span>
        )}
      </div>
      {violations.length > 0 && (
        <Collapsible
          label={`${violations.length} Violation${violations.length !== 1 ? 's' : ''}`}
          icon={'\u26A0'}
          defaultOpen={defaultOpen}
        >
          <div className="esc-violations-list">
            {violations.map((v, i) => (
              <div key={i} className={`esc-violation-row ${v.category === 'structural' ? 'esc-violation-structural' : ''}`}>
                {v.category && <span className="esc-v-category">{v.category}</span>}
                <span className="esc-v-text">{v.violation}</span>
              </div>
            ))}
          </div>
        </Collapsible>
      )}
    </>
  );
}

function DiagramContent({ esc, defaultOpen }) {
  const questions = esc.questions || [];

  return (
    <>
      <div className="esc-summary">
        {esc.question_count || questions.length} question{(esc.question_count || questions.length) !== 1 ? 's' : ''} from block diagram specialist
      </div>
      {questions.length > 0 && (
        <Collapsible
          label={`${questions.length} Question${questions.length !== 1 ? 's' : ''}`}
          icon={'\u2753'}
          defaultOpen={defaultOpen}
        >
          <div className="esc-questions-list">
            {questions.map((q, i) => (
              <div key={i} className="esc-question-row">
                <span className="esc-q-text">{typeof q === 'string' ? q : q.question || JSON.stringify(q)}</span>
              </div>
            ))}
          </div>
        </Collapsible>
      )}
    </>
  );
}

function ExhaustedContent({ esc, defaultOpen }) {
  const violations = esc.violations || [];

  return (
    <>
      <div className="esc-summary esc-exhausted-warning">
        Max iterations reached ({esc.max_rounds} rounds) with {esc.remaining_violations || violations.length} violation{(esc.remaining_violations || violations.length) !== 1 ? 's' : ''} remaining
      </div>
      {violations.length > 0 && (
        <Collapsible
          label={`${violations.length} Remaining Violation${violations.length !== 1 ? 's' : ''}`}
          icon={'\u26A0'}
          defaultOpen={defaultOpen}
        >
          <div className="esc-violations-list">
            {violations.map((v, i) => (
              <div key={i} className="esc-violation-row">
                {v.category && <span className="esc-v-category">{v.category}</span>}
                <span className="esc-v-text">{v.violation}</span>
              </div>
            ))}
          </div>
        </Collapsible>
      )}
    </>
  );
}

function EscalationResponse({ response }) {
  return (
    <div className="esc-response">
      <div className="esc-response-header">
        <span className="esc-response-icon">{'\u2705'}</span>
        <span>Resolved with action: <code>{response.action}</code></span>
      </div>
      {response.has_answers && (
        <div className="esc-response-detail">
          {response.answer_count} answer{response.answer_count !== 1 ? 's' : ''} provided
          {response.answer_keys?.length > 0 && (
            <span className="esc-response-keys"> ({response.answer_keys.join(', ')})</span>
          )}
        </div>
      )}
      {response.feedback && (
        <div className="esc-response-feedback">
          <div className="esc-response-feedback-label">Feedback:</div>
          <div className="esc-response-feedback-text">{response.feedback}</div>
        </div>
      )}
    </div>
  );
}


/* ── Main DetailPanel ────────────────────────────────────── */

const DetailPanel = React.memo(function DetailPanel({
  node,
  traceData,
  onRequestTraces,
  onClose,
  width,
  flowLayout,
}) {
  const [activeTabKey, setActiveTabKey] = useState(null);
  const scrollRef = useRef(null);
  const savedScrollRef = useRef(0);

  const isHITLNode = node?.uses_interrupt === true;

  // Re-group traces by (block_name, attempt) for multi-block nodes
  const tabGroups = React.useMemo(() => regroupTraces(traceData), [traceData]);

  // Save scroll position before data changes cause a re-render
  useLayoutEffect(() => {
    if (scrollRef.current) {
      savedScrollRef.current = scrollRef.current.scrollTop;
    }
  });

  // Restore scroll position after render
  useEffect(() => {
    if (scrollRef.current && savedScrollRef.current > 0) {
      scrollRef.current.scrollTop = savedScrollRef.current;
    }
  });

  // Auto-select tab matching the clicked segment's attempt, or fall
  // back to the latest non-streaming tab so historical segments don't
  // jump to the currently-active streaming call.
  useEffect(() => {
    if (tabGroups.length === 0) {
      setActiveTabKey(null);
      return;
    }
    // If current selection is still valid, keep it
    if (activeTabKey && tabGroups.some((g) => g.key === activeTabKey)) {
      return;
    }
    // Try to match the clicked segment's attempt number
    if (node?.attempt) {
      const match = tabGroups.find((g) => g.attempt === node.attempt);
      if (match) {
        setActiveTabKey(match.key);
        return;
      }
    }
    // Fall back to the latest non-streaming tab (avoid jumping to
    // an active call when the user clicked a historical segment).
    const nonStreaming = tabGroups.filter(
      (g) => !g.spans?.some((s) => s.children?.some((c) => c.status === 'streaming'))
    );
    if (nonStreaming.length > 0) {
      setActiveTabKey(nonStreaming[nonStreaming.length - 1].key);
    } else {
      setActiveTabKey(tabGroups[tabGroups.length - 1].key);
    }
  }, [tabGroups]);

  const handleRefresh = useCallback(() => {
    if (onRequestTraces && node) {
      onRequestTraces(node.label || node.id);
    }
  }, [onRequestTraces, node]);

  // Auto-refresh when viewing live data (node still running).
  // Use faster polling (2s) when streaming data is present for
  // realtime trajectory updates; fall back to 5s for regular live.
  // Also poll when traceData is empty but the node is still running
  // so we pick up the first LLM response as soon as it appears.
  const isLive = traceData?.some((g) => g.live);
  const hasStreaming = traceData?.some((g) => g.has_streaming);
  const isWaitingForData = Array.isArray(traceData) && traceData.length === 0 && node?.status === 'running';
  const shouldPoll = isLive || isWaitingForData;
  const refreshMs = hasStreaming ? 2000 : isWaitingForData ? 3000 : 5000;
  useEffect(() => {
    if (!shouldPoll || !onRequestTraces || !node) return;
    const interval = setInterval(() => {
      onRequestTraces(node.label || node.id);
    }, refreshMs);
    return () => clearInterval(interval);
  }, [shouldPoll, onRequestTraces, node, refreshMs]);

  if (!node) return null;

  const activeGroup = tabGroups.find((g) => g.key === activeTabKey);
  const llmCalls = activeGroup ? extractLLMCalls(activeGroup.spans) : [];

  return (
    <div
      className={`detail-panel ${flowLayout ? 'detail-panel-flow' : ''}`}
      style={width ? { width } : undefined}
    >
      {/* Header */}
      <div className="detail-header">
        <div className="detail-header-title">
          <h3>{node.label}</h3>
          {activeGroup?.blockName && (
            <span className="detail-block-badge">{activeGroup.blockName.replace(/_/g, ' ')}</span>
          )}
          {isLive && (
            <span className="detail-live-badge">LIVE</span>
          )}
        </div>
        <div className="detail-header-actions">
          <button
            className="detail-refresh"
            onClick={handleRefresh}
            title="Refresh"
          >
            &#x21bb;
          </button>
          <button className="detail-close" onClick={onClose}>
            &times;
          </button>
        </div>
      </div>

      {/* Description */}
      {node.description && (
        <div className="llm-description">{node.description}</div>
      )}

      {/* Trace section */}
      <div className="trace-section">
        {/* Loading */}
        {traceData === null && (
          <div className="trace-loading">
            <div className="trace-spinner" />
            Loading&hellip;
          </div>
        )}

        {/* Empty -- show HITL panel for interrupt nodes, metadata summary
            for completed non-LLM nodes, generic message otherwise */}
        {traceData && traceData.length === 0 && (
          isHITLNode ? (
            <HITLPanel node={node} />
          ) : node.status === 'running' ? (
            <div className="trace-empty">
              <div className="trace-spinner" />
              <span>Waiting for first LLM response...</span>
            </div>
          ) : node.metadata && Object.keys(node.metadata).length > 0 ? (
            <NodeMetadataSummary node={node} />
          ) : (
            <div className="trace-empty">
              <span className="trace-empty-icon">{'\u{1F4ED}'}</span>
              No activity recorded yet.
            </div>
          )
        )}

        {/* Has data */}
        {tabGroups.length > 0 && (
          <>
            {/* Tab bar */}
            <div className="trace-tabs">
              {tabGroups.map((group) => {
                const hasError = group.spans?.some(
                  (s) => s.status === 'error'
                );
                return (
                  <button
                    key={group.key}
                    className={`trace-tab ${
                      activeTabKey === group.key ? 'trace-tab-active' : ''
                    } ${hasError ? 'trace-tab-error' : ''}`}
                    onClick={() => setActiveTabKey(group.key)}
                  >
                    {hasError && (
                      <span className="trace-tab-icon">{'\u26A0'}</span>
                    )}
                    {group.label}
                  </button>
                );
              })}
            </div>

            {/* Summary bar */}
            <AttemptSummary
              spans={activeGroup?.spans}
              llmCount={llmCalls.length}
              hasStreamingCalls={llmCalls.some((c) => c.streaming)}
            />

            {/* LLM calls */}
            <div className="trace-content" ref={scrollRef}>
              {llmCalls.length > 0 ? (
                <div className="llm-call-list">
                  {llmCalls.map((call, i) => (
                    call.streaming
                      ? <StreamingLLMCard key={call.id || `stream-${i}`} call={call} />
                      : <LLMCallCard key={call.id || i} call={call} />
                  ))}
                </div>
              ) : node.metadata && Object.keys(node.metadata).length > 0 ? (
                <NodeMetadataSummary node={node} />
              ) : (
                <div className="trace-empty-tab">
                  <span className="trace-empty-icon">{'\u{1F4CB}'}</span>
                  No LLM interactions in this run.
                </div>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
});

export default DetailPanel;
