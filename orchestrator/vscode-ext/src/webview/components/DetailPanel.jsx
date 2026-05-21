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
/**
 * Convert traceData (in either the OTel/live-calls shape with `spans[]`, or
 * the unified-trajectory shape with `steps[]`) into a flat list of tab
 * descriptors with a normalized `steps[]` field that always interleaves
 * LLM calls and tool runs in chronological order.
 */
function regroupTraces(traceData) {
  if (!traceData || traceData.length === 0) return [];

  const isTrajectory = traceData[0]?.trajectory === true ||
                       Array.isArray(traceData[0]?.steps);

  // Extract block names from each attempt group
  const groups = traceData.map((group) => ({
    ...group,
    blockName: group.block || group.blockName || extractBlockName(group.spans),
  }));

  // Build a `steps` list per group: trajectory data is already steps;
  // span data is converted via extractLLMCalls (LLM-only).
  for (const g of groups) {
    if (Array.isArray(g.steps) && g.steps.length > 0) {
      // already normalized
      continue;
    }
    const calls = extractLLMCalls(g.spans);
    g.steps = calls.map((c) => ({
      type: c.streaming ? 'llm_call_streaming' : 'llm_call',
      ts: c.ts || 0,
      model: c.model,
      duration_s: c.duration_ms ? c.duration_ms / 1000 : null,
      duration_ms: c.duration_ms,
      system_prompt: c.systemPrompt,
      user_prompt: c.prompt,
      response: c.response,
      status: c.status,
      streaming: c.streaming,
      _span: c,
    }));
  }

  const blockNames = new Set(groups.map((g) => g.blockName).filter(Boolean));
  const hasMultipleBlocks = blockNames.size > 1;

  // Pick a per-attempt duration: prefer trajectory's own duration_s field
  function attemptDurMs(g) {
    if (g.duration_s) return g.duration_s * 1000;
    if (g.duration_ms) return g.duration_ms;
    return getAttemptDuration(g.spans);
  }

  // Some pipelines re-enter the same node multiple times within a single
  // "attempt" (Constraint Check fires once per architecture round). When
  // a result_summary carries a round number, prefer it for the label so
  // the user sees "Attempt 1 · Round 2" instead of an opaque "(#2)".
  const seenKeys = new Map();
  return groups.map((g) => {
    const steps = g.steps || [];
    const toolCount = steps.filter((s) => s.type === 'tool_run').length;
    const llmCount = steps.filter((s) => s.type === 'llm_call' || s.type === 'llm_call_streaming' || !s.type).length;
    // Probe step metrics for a useful round / iteration discriminator.
    let roundNum = null;
    for (const s of steps) {
      if (s.type === 'result_summary' && s.metrics) {
        const r = s.metrics.round ?? s.metrics.new_round;
        if (r != null) { roundNum = r; break; }
      }
    }
    const baseKey = hasMultipleBlocks
      ? `${g.blockName || 'unknown'}:${g.attempt}`
      : `a${g.attempt}`;
    const count = (seenKeys.get(baseKey) || 0) + 1;
    seenKeys.set(baseKey, count);
    const repeatCount = groups.filter((og) => {
      const k = hasMultipleBlocks
        ? `${og.blockName || 'unknown'}:${og.attempt}`
        : `a${og.attempt}`;
      return k === baseKey;
    }).length;
    const isRepeat = repeatCount > 1;
    let label = hasMultipleBlocks
      ? `${(g.blockName || 'unknown').replace(/_/g, ' ')} · Attempt ${g.attempt}`
      : (groups.length === 1 ? 'Run' : `Attempt ${g.attempt}`);
    if (isRepeat) {
      // Build a discriminator that includes round when known and
      // always appends an iteration counter so two calls in the same
      // round still get distinct labels (Block Diagram fires multiple
      // times per architecture round during escalation feedback).
      const parts = [];
      if (roundNum != null) parts.push(`Round ${roundNum}`);
      parts.push(`Iter ${count}`);
      label += ` · ${parts.join(' · ')}`;
    }
    return {
      key: `${baseKey}#${count}`,
      label,
      durMs: attemptDurMs(g),
      attempt: g.attempt,
      blockName: g.blockName,
      round: roundNum,
      iteration: count,
      status: g.status,
      exitEvent: g.exit_event,
      spans: g.spans,
      steps,
      llmCount,
      toolCount,
      isTrajectory: isTrajectory,
    };
  });
}

function getAttemptDuration(spans) {
  if (!spans || spans.length === 0) return null;
  // For OTel traces the parent span carries the duration; for live_calls the
  // synthetic root span has duration_ms=null and the per-call children carry
  // it. Fall back to summing children when the parent is missing a duration
  // so tab labels still show a number in either case.
  let total = 0;
  for (const s of spans) {
    if (s.duration_ms) {
      total += s.duration_ms;
    } else if (s.children && s.children.length) {
      for (const c of s.children) {
        if (c.duration_ms) total += c.duration_ms;
      }
    }
  }
  return total || null;
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
      const escaped = escapeHtml(text);
      // Split on inline-code spans first so bold/italic regex never runs
      // inside `code` -- otherwise `dedicated_pins_` and similar snake_case
      // identifiers get their underscores eaten by the italic rule and
      // styling breaks mid-token.
      const parts = escaped.split(/(`[^`\n]+`)/g);
      return parts.map((part) => {
        if (part.length >= 2 && part.startsWith('`') && part.endsWith('`')) {
          return `<code>${part.slice(1, -1)}</code>`;
        }
        return part
          .replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>')
          .replace(/\*([^*\n]+)\*/g, '<em>$1</em>')
          // Underscore italic must sit at a word boundary (whitespace or
          // punctuation) on both sides -- this leaves snake_case tokens
          // and file paths like arch/uarch_specs/foo_bar.md alone.
          .replace(/(^|[^A-Za-z0-9_])_([^_\n\s][^_\n]*?)_(?=[^A-Za-z0-9_]|$)/g,
                   '$1<em>$2</em>');
      }).join('');
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

/* ── JSON detection + syntax highlighting ─────────────────── */

function _tryParseJson(text) {
  if (!text || typeof text !== 'string') return null;
  const trimmed = text.trim();
  if (!trimmed) return null;
  // Cheap shape check before paying parse cost
  const first = trimmed[0];
  const last = trimmed[trimmed.length - 1];
  if (!((first === '{' && last === '}') || (first === '[' && last === ']'))) {
    return null;
  }
  try {
    return JSON.parse(trimmed);
  } catch {
    return null;
  }
}

// Minimal regex-based JSON syntax highlighter -- escapes HTML, then wraps
// keys/strings/numbers/booleans/nulls in styled spans.
function _highlightJsonHtml(jsonStr) {
  const escaped = jsonStr
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
  return escaped.replace(
    /("(\\.|[^"\\])*")\s*:|("(\\.|[^"\\])*")|\b(true|false|null)\b|\b(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\b/g,
    (match, key, _k2, str, _s2, kw, num) => {
      if (key) {
        const k = key.replace(/:$/, '').trim();
        return `<span class="json-key">${k}</span>:`;
      }
      if (str) return `<span class="json-string">${str}</span>`;
      if (kw) return `<span class="json-${kw}">${kw}</span>`;
      if (num) return `<span class="json-number">${num}</span>`;
      return match;
    },
  );
}

const _REASONING_KEYS = [
  'reasoning', 'thinking', 'diagnosis', 'analysis', 'rationale',
  'explanation', 'summary',
];
const _ACTION_KEYS = ['action', 'decision', 'next_step', 'suggested_fix'];

/* ── Formatted text renderer with markdown ───────────────── */

function FormattedText({ text, maxCollapsed = 2000, autoJson = true }) {
  const [expanded, setExpanded] = useState(false);

  if (!text) return <span className="llm-no-content">No content</span>;

  // If the entire response is a JSON object, surface the reasoning/diagnosis
  // field above the JSON body so the user sees the "thinking" without
  // hunting through escaped JSON.
  const parsed = autoJson ? _tryParseJson(text) : null;
  if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
    let reasoningKey = null;
    let actionKey = null;
    for (const k of _REASONING_KEYS) {
      if (typeof parsed[k] === 'string' && parsed[k].trim()) {
        reasoningKey = k; break;
      }
    }
    for (const k of _ACTION_KEYS) {
      if (parsed[k] != null && (typeof parsed[k] === 'string' || typeof parsed[k] === 'number')) {
        actionKey = k; break;
      }
    }
    const pretty = JSON.stringify(parsed, null, 2);
    const isLong = pretty.length > maxCollapsed;
    const display = isLong && !expanded ? pretty.slice(0, maxCollapsed) : pretty;
    return (
      <div className="llm-formatted llm-json-response">
        {(reasoningKey || actionKey) && (
          <div className="llm-thinking">
            {reasoningKey && (
              <div className="llm-thinking-section">
                <div className="llm-thinking-label">{reasoningKey === 'reasoning' ? 'Thinking' : reasoningKey.replace(/_/g,' ').replace(/\b\w/g, c => c.toUpperCase())}</div>
                <div className="llm-thinking-body">{parsed[reasoningKey]}</div>
              </div>
            )}
            {actionKey && (
              <div className="llm-thinking-section llm-thinking-action">
                <div className="llm-thinking-label">{actionKey.replace(/_/g,' ').replace(/\b\w/g, c => c.toUpperCase())}</div>
                <div className="llm-thinking-body">{String(parsed[actionKey])}</div>
              </div>
            )}
          </div>
        )}
        <pre className="llm-code-block json-code-block">
          <span className="llm-code-lang">json</span>
          <code dangerouslySetInnerHTML={{ __html: _highlightJsonHtml(display) }} />
        </pre>
        {isLong && (
          <button
            className="llm-expand-btn"
            onClick={() => setExpanded(!expanded)}
          >
            {expanded
              ? 'Show less'
              : `Show more (${(pretty.length / 1000).toFixed(1)}k chars)`}
          </button>
        )}
      </div>
    );
  }

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

/* ── Tool Run Card ───────────────────────────────────────── */

function _classifyToolStep(step) {
  // Step name → display label + icon
  const map = {
    lint: { label: 'Lint', icon: '🔍' },
    simulate: { label: 'Simulate', icon: '▶' },
    synthesize: { label: 'Synthesize', icon: '⚙' },
    integration_lint: { label: 'Integration Lint', icon: '🔍' },
    integration_sim: { label: 'Integration Sim', icon: '▶' },
    flat_top_synth: { label: 'Flat Top Synth', icon: '⚙' },
    place: { label: 'Place', icon: '◫' },
    route: { label: 'Route', icon: '⫸' },
    cts: { label: 'CTS', icon: '🕒' },
    drc: { label: 'DRC', icon: '✓' },
    lvs: { label: 'LVS', icon: '✓' },
  };
  return map[step] || { label: step.replace(/_/g, ' '), icon: '🔧' };
}

function ToolRunCard({ run, index, total }) {
  const [open, setOpen] = useState(false);
  const { label, icon } = _classifyToolStep(run.step || 'tool');
  const rc = run.return_code;
  const ok = rc === 0;
  const statusIcon = ok ? '✓' : rc != null ? '✗' : '—';
  const statusCls = ok ? 'ok' : rc != null ? 'error' : 'unset';

  // Parse useful chunks from the log content -- our pipeline logs have
  // === STDOUT === / === STDERR === markers we can split on.
  const content = run.content || '';
  const stdoutMatch = content.match(/=== STDOUT ===([\s\S]*?)(?:=== STDERR ===|$)/);
  const stderrMatch = content.match(/=== STDERR ===([\s\S]*?)$/);
  const stdout = (stdoutMatch?.[1] || '').trim();
  const stderr = (stderrMatch?.[1] || '').trim();
  const hasOnlyStdout = stdout && !stderr;
  const hasOnlyStderr = !stdout && stderr;

  return (
    <div className={`llm-card tool-card tool-card-${statusCls}`}>
      <div className="llm-card-header">
        {total > 1 && (
          <span className="llm-call-index" title={`Step ${index + 1} of ${total}`}>
            {`${index + 1}/${total}`}
          </span>
        )}
        <span className="tool-card-icon" aria-hidden="true">{icon}</span>
        <span className="llm-model-name tool-card-label">{label}</span>
        <span className="llm-card-meta">
          {run.command && (
            <span className="tool-card-cmd" title={run.command}>
              {(() => {
                const first = run.command.split(/\s+/)[0] || '';
                const tool = first.split('/').pop() || first;
                return tool;
              })()}
            </span>
          )}
          {rc != null && (
            <span className={`tool-card-rc tool-card-rc-${statusCls}`}>
              rc={rc}
            </span>
          )}
          <span className={`llm-stat llm-stat-${statusCls}`}>{statusIcon}</span>
        </span>
      </div>

      {run.command && (
        <div className="tool-card-command">
          <span className="tool-card-command-label">$</span>
          <code className="tool-card-command-text">{run.command}</code>
        </div>
      )}

      <button
        className="tool-card-toggle"
        onClick={() => setOpen((v) => !v)}
      >
        {open ? '▼ Hide output' : '▶ Show output'}
        {run.size != null && (
          <span className="tool-card-size">
            {run.size > 1024 ? `${(run.size / 1024).toFixed(1)} KB` : `${run.size} B`}
          </span>
        )}
      </button>
      {open && (
        <div className="tool-card-output">
          {stdout && (
            <details open={hasOnlyStdout} className="tool-output-section">
              <summary>stdout</summary>
              <pre className="tool-output-pre">{stdout}</pre>
            </details>
          )}
          {stderr && (
            <details open={hasOnlyStderr || !ok} className="tool-output-section tool-output-stderr">
              <summary>stderr</summary>
              <pre className="tool-output-pre">{stderr}</pre>
            </details>
          )}
          {!stdout && !stderr && content && (
            <pre className="tool-output-pre">{content}</pre>
          )}
        </div>
      )}
    </div>
  );
}

function ResultSummaryCard({ metrics, index, total }) {
  const entries = Object.entries(metrics || {}).filter(([, v]) => v != null);
  if (!entries.length) return null;

  function fmt(key, value) {
    if (value == null) return '—';
    if (typeof value === 'boolean') return value ? '✓ yes' : '✗ no';
    if (Array.isArray(value)) {
      if (value.length === 0) return '(none)';
      if (value.length <= 4) return value.join(', ');
      return `${value.slice(0, 3).join(', ')}, +${value.length - 3} more`;
    }
    if (key === 'dashboard_path' || key === 'path' || key === 'log_path'
        || key === 'layout_2d_png_path' || key === 'viewer_path') {
      const s = String(value);
      return s.length > 80 ? '…' + s.slice(-80) : s;
    }
    if (key === 'html_size' || key === 'size' || key === 'stdout_bytes') {
      const n = Number(value);
      if (!Number.isFinite(n)) return String(value);
      return n > 1024 ? `${(n / 1024).toFixed(1)} KB` : `${n} B`;
    }
    if (key === 'utilization_pct') {
      const n = Number(value);
      if (Number.isFinite(n)) return `${n.toFixed(1)}%`;
    }
    if (key === 'confidence') {
      const n = Number(value);
      if (Number.isFinite(n)) {
        const pct = n <= 1 ? n * 100 : n;
        return `${pct.toFixed(0)}%`;
      }
    }
    if (typeof value === 'number') {
      if (Math.abs(value) >= 1000) return value.toLocaleString();
      if (Number.isInteger(value)) return String(value);
      return value.toFixed(3);
    }
    return String(value);
  }

  const LABEL = {
    gate_count: 'Gate count',
    chip_area_um2: 'Chip area (µm²)',
    design_area_um2: 'Design area (µm²)',
    utilization_pct: 'Utilization',
    wns_ns: 'WNS (ns)',
    tns_ns: 'TNS (ns)',
    total_power_mw: 'Total power (mW)',
    max_freq_mhz: 'Max freq (MHz)',
    violations: 'Violations',
    violation_count: 'Violations',
    error_count: 'Errors',
    sim_passed: 'Sim passed',
    lint_passed: 'Lint passed',
    lint_clean: 'Lint clean',
    success: 'Success',
    passed: 'Passed',
    clean: 'Clean',
    all_pass: 'All passed',
    match: 'LVS match',
    dashboard_path: 'Dashboard',
    log_path: 'Log file',
    path: 'Output path',
    html_size: 'Dashboard size',
    layout_2d_png_path: 'Layout image',
    viewer_path: '3D viewer',
    ers_generated: 'ERS generated',
    node_count: 'Nodes',
    edge_count: 'Edges',
    block_count: 'Blocks',
    blocks: 'Blocks',
    tb_fixes_attempted: 'TB fixes',
    local_fixes_attempted: 'Local fixes',
    validation_errors: 'Validation errors',
    tier: 'Tier',
    round: 'Round',
    new_round: 'New round',
    max_rounds: 'Max rounds',
    total_rounds: 'Total rounds',
    design_name: 'Design',
    integration_top: 'Integration top',
    category: 'Category',
    confidence: 'Confidence',
    action: 'Action',
    decision: 'Decision',
    phase: 'Phase',
    skipped: 'Skipped',
    needs_human: 'Needs human',
    diagnosis_preview: 'Diagnosis',
    suggested_fix: 'Suggested fix',
    last_error: 'Last error',
    text_len: 'Text length',
    has_structural: 'Structural violations',
    has_feedback: 'Feedback',
    feedback: 'Feedback',
    issues_found: 'Issues',
    device_delta: 'Device delta',
    net_delta: 'Net delta',
    analysis: 'Analysis',
    answer_count: 'Answers',
    answer_keys: 'Answered',
    has_answers: 'Has answers',
    attempt: 'Attempt',
    new_tier_index: 'Next tier',
    question_count: 'Questions',
    total: 'Total',
    expected: 'Expected',
    completed_so_far: 'Completed',
    passed_so_far: 'Passed',
  };

  return (
    <div className="llm-card result-card">
      <div className="llm-card-header">
        {total > 1 && (
          <span className="llm-call-index" title={`Step ${index + 1} of ${total}`}>
            {`${index + 1}/${total}`}
          </span>
        )}
        <span className="tool-card-icon" aria-hidden="true">📋</span>
        <span className="llm-model-name tool-card-label">Outcome</span>
      </div>
      <div className="result-card-grid">
        {entries.map(([k, v]) => {
          // Wide strings / arrays / paths get a full-width row so they
          // don't truncate to "32 Bit Adder Full Flow Sm...".
          const isWide = k === 'dashboard_path' || k === 'path'
                       || k === 'log_path' || k === 'design_name'
                       || k === 'category' || k === 'analysis'
                       || k === 'diagnosis_preview' || k === 'suggested_fix'
                       || k === 'feedback' || k === 'last_error'
                       || k === 'layout_2d_png_path' || k === 'viewer_path'
                       || Array.isArray(v)
                       || (typeof v === 'string' && v.length > 28);
          const isBool = typeof v === 'boolean';
          const isNumeric = typeof v === 'number';
          return (
            <div key={k} className={`result-card-row ${isWide ? 'full-width' : ''}`}>
              <span className="result-card-key">{LABEL[k] || k.replace(/_/g, ' ')}</span>
              <span className={`result-card-val ${isBool ? (v ? 'ok' : 'error') : ''} ${isNumeric ? 'numeric' : ''}`}>
                {fmt(k, v)}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function TrajectorySteps({ steps }) {
  // Count for the N/total badge; combine LLM + tool but show them as a
  // single ordered list.
  const total = steps.length;
  return (
    <div className="llm-call-list trajectory-steps">
      {steps.map((step, i) => {
        if (step.type === 'tool_run') {
          return <ToolRunCard key={`tool-${i}`} run={step} index={i} total={total} />;
        }
        if (step.type === 'result_summary') {
          return <ResultSummaryCard key={`res-${i}`} metrics={step.metrics} index={i} total={total} />;
        }
        // llm_call / llm_call_streaming
        const call = {
          id: step._span?.id || `step-${i}`,
          model: step.model || 'LLM',
          promptTokens: step.usage?.prompt_tokens || step.usage?.input_tokens,
          completionTokens: step.usage?.completion_tokens || step.usage?.output_tokens,
          totalTokens: step.usage?.total_tokens,
          duration_ms: step.duration_ms || (step.duration_s ? step.duration_s * 1000 : null),
          status: step.status || (step.error ? 'error' : 'ok'),
          prompt: step.user_prompt || '',
          systemPrompt: step.system_prompt || '',
          response: step.response || '',
          streaming: step.streaming,
        };
        return call.streaming
          ? <StreamingLLMCard key={`stream-${i}`} call={call} />
          : <LLMCallCard key={`llm-${i}`} call={call} index={i} total={total} />;
      })}
    </div>
  );
}

/* ── LLM Call Card ───────────────────────────────────────── */

function LLMCallCard({ call, index, total }) {
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
        {total > 1 && (
          <span className="llm-call-index" title={`Call ${index + 1} of ${total}`}>
            {`${index + 1}/${total}`}
          </span>
        )}
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
        <Collapsible label="System Prompt" icon={'\u2699'} className="llm-sys">
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

function AttemptSummary({ spans, steps, llmCount, hasStreamingCalls }) {
  // Prefer step-aware tallies when we have trajectory data
  let llmN = llmCount || 0;
  let toolN = 0;
  let toolErr = 0;
  let resultN = 0;
  let resultErr = 0;
  let stepDurMs = 0;
  if (Array.isArray(steps) && steps.length) {
    llmN = 0;
    for (const s of steps) {
      if (s.type === 'tool_run') {
        toolN++;
        if (s.return_code != null && s.return_code !== 0) toolErr++;
      } else if (s.type === 'result_summary') {
        resultN++;
        const m = s.metrics || {};
        if (m.success === false || m.passed === false || m.clean === false
            || m.lint_passed === false || m.sim_passed === false
            || m.match === false || m.all_pass === false) {
          resultErr++;
        }
      } else {
        // llm_call / llm_call_streaming / undefined (legacy span path)
        llmN++;
      }
      const d = s.duration_ms || (s.duration_s ? s.duration_s * 1000 : 0);
      if (d) stepDurMs += d;
    }
  }
  const duration = stepDurMs || getAttemptDuration(spans);
  const status = getAttemptStatus(spans);
  const isError = status === 'error' || toolErr > 0 || resultErr > 0;
  const isStreaming = hasStreamingCalls;
  const noActivity = llmN === 0 && toolN === 0 && resultN === 0;

  return (
    <div className={`llm-summary-bar ${isStreaming ? 'streaming' : isError ? 'error' : 'ok'}`}>
      <span className={`llm-summary-status ${isStreaming ? 'streaming' : isError ? 'error' : 'ok'}`}>
        {isStreaming ? '\u25CF Generating' : isError ? '\u2717 Failed' : '\u2713 Completed'}
      </span>
      <span className="llm-summary-detail">
        {!isStreaming && duration ? formatDuration(duration) : null}
        {llmN > 0 && (
          <span className="llm-summary-count">
            {llmN} LLM call{llmN !== 1 ? 's' : ''}
            {isStreaming ? ' (1 active)' : ''}
          </span>
        )}
        {toolN > 0 && (
          <span className="llm-summary-count tool-summary-count">
            {toolN} tool run{toolN !== 1 ? 's' : ''}
            {toolErr > 0 ? ` (${toolErr} failed)` : ''}
          </span>
        )}
        {llmN === 0 && toolN === 0 && resultN > 0 && (
          <span className="llm-summary-count">result-only</span>
        )}
        {noActivity && (
          <span className="llm-summary-count">no activity</span>
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
    // Try to match the clicked segment's attempt number, but prefer a
    // sibling attempt that actually has tool runs -- otherwise the user
    // sees only LLM cards and assumes the trajectory is incomplete.
    if (node?.attempt) {
      const sameAttempt = tabGroups.filter((g) => g.attempt === node.attempt);
      const withTools = sameAttempt.find((g) => (g.toolCount || 0) > 0);
      const exact = sameAttempt[0];
      const pick = withTools || exact;
      if (pick) {
        setActiveTabKey(pick.key);
        return;
      }
    }

    // No clicked attempt -- prefer the latest tab that has tool runs so
    // tool data is visible by default; fall back to latest non-streaming.
    const nonStreaming = tabGroups.filter(
      (g) => !g.spans?.some((s) => s.children?.some((c) => c.status === 'streaming'))
    );
    const candidates = nonStreaming.length ? nonStreaming : tabGroups;
    const withTools = [...candidates].reverse().find((g) => (g.toolCount || 0) > 0);
    setActiveTabKey((withTools || candidates[candidates.length - 1]).key);
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
                // Detect failure across both legacy span data and the
                // unified trajectory steps. Attempts that failed (e.g.
                // Flat Top Synthesis on success=false) used to render as
                // a normal tab, hiding the failure from a casual look.
                const spanError = group.spans?.some(
                  (s) => s.status === 'error'
                );
                const stepError = group.steps?.some((s) => {
                  if (s.type === 'tool_run') {
                    return s.return_code != null && s.return_code !== 0;
                  }
                  if (s.type === 'result_summary') {
                    const m = s.metrics || {};
                    return m.success === false
                        || m.passed === false
                        || m.clean === false
                        || m.lint_passed === false
                        || m.sim_passed === false
                        || m.match === false;
                  }
                  return s.status === 'error';
                });
                const groupStatusError = group.status === 'failed';
                const hasError = spanError || stepError || groupStatusError;
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
                    <span className="trace-tab-label">{group.label}</span>
                    {group.durMs ? (
                      <span className="trace-tab-dur">
                        {formatDuration(group.durMs)}
                      </span>
                    ) : null}
                    {(group.llmCount > 0 || group.toolCount > 0) && (
                      <span className="trace-tab-counts">
                        {group.llmCount > 0 && (
                          <span className="trace-tab-count-llm" title={`${group.llmCount} LLM call${group.llmCount !== 1 ? 's' : ''}`}>
                            ✦{group.llmCount}
                          </span>
                        )}
                        {group.toolCount > 0 && (
                          <span className="trace-tab-count-tool" title={`${group.toolCount} tool run${group.toolCount !== 1 ? 's' : ''}`}>
                            🔧{group.toolCount}
                          </span>
                        )}
                      </span>
                    )}
                  </button>
                );
              })}
            </div>

            {/* Summary bar */}
            <AttemptSummary
              spans={activeGroup?.spans}
              steps={activeGroup?.steps}
              llmCount={llmCalls.length}
              hasStreamingCalls={llmCalls.some((c) => c.streaming)}
            />

            {/* Steps (LLM calls + tool runs interleaved) */}
            <div className="trace-content" ref={scrollRef}>
              {activeGroup?.steps && activeGroup.steps.length > 0 ? (
                <TrajectorySteps steps={activeGroup.steps} />
              ) : llmCalls.length > 0 ? (
                <div className="llm-call-list">
                  {llmCalls.map((call, i) => (
                    call.streaming
                      ? <StreamingLLMCard key={call.id || `stream-${i}`} call={call} />
                      : <LLMCallCard
                          key={call.id || i}
                          call={call}
                          index={i}
                          total={llmCalls.length}
                        />
                  ))}
                </div>
              ) : node.metadata && Object.keys(node.metadata).length > 0 ? (
                <NodeMetadataSummary node={node} />
              ) : (
                <div className="trace-empty-tab">
                  <span className="trace-empty-icon">{'\u{1F4CB}'}</span>
                  No activity recorded in this run.
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
