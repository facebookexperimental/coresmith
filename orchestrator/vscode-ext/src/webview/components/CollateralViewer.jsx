import React, { useState, useEffect, useMemo } from 'react';

const KIND_META = {
  rtl: { label: 'RTL (Verilog)', icon: '⚡' },
  tb: { label: 'Testbench', icon: '⌭' },
  syn: { label: 'Synthesis', icon: '⚙' },
  pnr: { label: 'PnR', icon: '◫' },
  waveforms: { label: 'Waveforms', icon: '〰' },
  reports: { label: 'Reports / Docs', icon: '📋' },
};

const TEXT_EXTS = new Set([
  '.v', '.sv', '.py', '.sdc', '.ys', '.tcl', '.txt', '.json', '.md',
  '.rpt', '.log', '.def', '.html', '.csv',
]);

function _extOf(name) {
  const i = name.lastIndexOf('.');
  return i < 0 ? '' : name.slice(i).toLowerCase();
}

function _formatBytes(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

// Very lightweight Verilog highlighter -- escapes HTML then wraps keywords,
// strings, numbers, comments in styled spans. Not a full parser, but enough
// to make a file readable.
const VERILOG_KEYWORDS = new Set([
  'module', 'endmodule', 'input', 'output', 'inout', 'wire', 'reg',
  'logic', 'assign', 'always', 'always_ff', 'always_comb', 'begin',
  'end', 'if', 'else', 'case', 'endcase', 'casex', 'casez', 'default',
  'parameter', 'localparam', 'integer', 'genvar', 'generate',
  'endgenerate', 'for', 'while', 'function', 'endfunction', 'task',
  'endtask', 'return', 'posedge', 'negedge', 'or', 'and', 'not', 'xor',
  'initial', 'forever', 'repeat', 'do', 'fork', 'join', 'disable',
  'wait', 'macromodule', 'primitive', 'endprimitive', 'specify',
  'endspecify', 'typedef', 'enum', 'struct', 'packed', 'unique',
  'priority', 'unique0', 'package', 'endpackage', 'import', 'export',
  'class', 'endclass', 'extends', 'virtual', 'pure', 'static',
  'automatic', 'rand', 'randc', 'constraint', 'property', 'sequence',
]);

function _escHtml(s) {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

// Single-pass Verilog highlighter. Walks the text once, finds the longest
// applicable token at each position, escapes it, and emits a styled span.
// The previous regex-cascade approach was broken because the string regex
// matched class="..." attributes inserted by the comment regex.
function highlightVerilog(text) {
  if (!text) return '';
  const out = [];
  let i = 0;
  const n = text.length;
  while (i < n) {
    const c = text[i];
    const c2 = text[i + 1];

    // Line comment
    if (c === '/' && c2 === '/') {
      let j = text.indexOf('\n', i);
      if (j < 0) j = n;
      out.push(`<span class="syntax-comment">${_escHtml(text.slice(i, j))}</span>`);
      i = j;
      continue;
    }
    // Block comment
    if (c === '/' && c2 === '*') {
      let j = text.indexOf('*/', i + 2);
      if (j < 0) j = n;
      else j += 2;
      out.push(`<span class="syntax-comment">${_escHtml(text.slice(i, j))}</span>`);
      i = j;
      continue;
    }
    // String
    if (c === '"') {
      let j = i + 1;
      while (j < n) {
        if (text[j] === '\\') { j += 2; continue; }
        if (text[j] === '"') { j++; break; }
        j++;
      }
      out.push(`<span class="syntax-string">${_escHtml(text.slice(i, j))}</span>`);
      i = j;
      continue;
    }
    // Sized Verilog number (8'hAB / 16'b... )
    const sizedMatch = text.slice(i).match(/^\d+'(?:[bdoh])[0-9a-fA-F_xz?]+/);
    if (sizedMatch) {
      out.push(`<span class="syntax-number">${_escHtml(sizedMatch[0])}</span>`);
      i += sizedMatch[0].length;
      continue;
    }
    // Plain number
    if (/[0-9]/.test(c)) {
      const numMatch = text.slice(i).match(/^\d+(?:\.\d+)?(?:[eE][+-]?\d+)?/);
      if (numMatch) {
        out.push(`<span class="syntax-number">${_escHtml(numMatch[0])}</span>`);
        i += numMatch[0].length;
        continue;
      }
    }
    // Identifier / keyword
    if (/[a-zA-Z_]/.test(c)) {
      const idMatch = text.slice(i).match(/^[a-zA-Z_][a-zA-Z0-9_$]*/);
      if (idMatch) {
        const ident = idMatch[0];
        if (VERILOG_KEYWORDS.has(ident)) {
          out.push(`<span class="syntax-keyword">${ident}</span>`);
        } else {
          out.push(_escHtml(ident));
        }
        i += ident.length;
        continue;
      }
    }
    // Plain character
    out.push(_escHtml(c));
    i++;
  }
  return out.join('');
}

function highlightJson(text) {
  if (!text) return '';
  let escaped = text
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

function FileViewer({ file }) {
  const [content, setContent] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!file) return;
    const ext = _extOf(file.name);
    if (!TEXT_EXTS.has(ext)) {
      setContent(null);
      setError(null);
      return;
    }
    setLoading(true);
    setError(null);
    fetch(`/api/artifacts/${file.rel_path}`)
      .then(async (r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.text();
      })
      .then((txt) => {
        const MAX = 256 * 1024;
        if (txt.length > MAX) {
          setContent(txt.slice(0, MAX) + '\n\n... [truncated]');
        } else {
          setContent(txt);
        }
        setLoading(false);
      })
      .catch((e) => {
        setError(e.message || String(e));
        setLoading(false);
      });
  }, [file?.rel_path]);

  if (!file) {
    return (
      <div className="collateral-empty">
        <div className="collateral-empty-icon">📂</div>
        <div>Select a file on the left to view its content.</div>
      </div>
    );
  }

  const ext = _extOf(file.name);
  const isText = TEXT_EXTS.has(ext);

  let highlighted = null;
  if (content) {
    if (ext === '.v' || ext === '.sv') highlighted = highlightVerilog(content);
    else if (ext === '.json') highlighted = highlightJson(content);
    else highlighted = content
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');
  }

  return (
    <div className="collateral-content">
      <div className="collateral-content-header">
        <div className="collateral-file-name">{file.name}</div>
        <div className="collateral-file-meta">
          <span>{file.rel_path}</span>
          <span>·</span>
          <span>{_formatBytes(file.size)}</span>
          <a
            className="collateral-download"
            href={`/api/artifacts/${file.rel_path}`}
            download={file.name}
          >Open raw</a>
        </div>
      </div>
      <div className="collateral-content-body">
        {!isText && (
          <div className="collateral-empty">
            <div className="collateral-empty-icon">📦</div>
            <div>Binary file ({ext || 'unknown'}) — open via the link above.</div>
          </div>
        )}
        {loading && <div className="collateral-loading">Loading…</div>}
        {error && <div className="collateral-error">Failed to load: {error}</div>}
        {isText && content != null && !error && (
          <pre className="collateral-code">
            <code dangerouslySetInnerHTML={{ __html: highlighted }} />
          </pre>
        )}
      </div>
    </div>
  );
}

export default function CollateralViewer() {
  const [data, setData] = useState(null);
  const [selected, setSelected] = useState(null);
  const [filter, setFilter] = useState('');
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch('/api/collateral')
      .then((r) => r.ok ? r.json() : Promise.reject(`HTTP ${r.status}`))
      .then(setData)
      .catch((e) => setError(String(e)));
  }, []);

  const sections = useMemo(() => {
    if (!data) return [];
    const out = [];
    const kinds = ['rtl', 'tb', 'syn', 'pnr', 'waveforms', 'reports'];
    for (const k of kinds) {
      const items = (data[k] || []).filter((f) => {
        if (!filter) return true;
        const q = filter.toLowerCase();
        return f.name.toLowerCase().includes(q) || f.rel_path.toLowerCase().includes(q);
      });
      if (items.length === 0) continue;
      out.push({ kind: k, meta: KIND_META[k], items });
    }
    return out;
  }, [data, filter]);

  if (error) {
    return (
      <div className="collateral-shell">
        <div className="collateral-empty">Failed to load collateral: {error}</div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="collateral-shell">
        <div className="collateral-empty">Loading collateral…</div>
      </div>
    );
  }

  const totalFiles = sections.reduce((s, sec) => s + sec.items.length, 0);

  return (
    <div className="collateral-shell">
      <div className="collateral-sidebar">
        <div className="collateral-toolbar">
          <input
            className="collateral-filter"
            placeholder={`Filter ${totalFiles} files…`}
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
          />
        </div>
        <div className="collateral-tree">
          {sections.length === 0 && (
            <div className="collateral-empty-mini">No matching files.</div>
          )}
          {sections.map(({ kind, meta, items }) => (
            <div key={kind} className="collateral-section">
              <div className="collateral-section-header">
                <span className="collateral-section-icon">{meta.icon}</span>
                <span>{meta.label}</span>
                <span className="collateral-section-count">{items.length}</span>
              </div>
              <ul className="collateral-file-list">
                {items.map((f) => (
                  <li
                    key={f.rel_path}
                    className={`collateral-file ${selected?.rel_path === f.rel_path ? 'collateral-file-selected' : ''}`}
                    onClick={() => setSelected(f)}
                    title={f.rel_path}
                  >
                    <span className="collateral-file-iname">{f.name}</span>
                    <span className="collateral-file-isize">{_formatBytes(f.size)}</span>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </div>
      <div className="collateral-viewer">
        <FileViewer file={selected} />
      </div>
    </div>
  );
}
