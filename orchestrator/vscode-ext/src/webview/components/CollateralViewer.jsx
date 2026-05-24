import React, { useState, useEffect, useRef, useMemo } from 'react';

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
  '.rpt', '.log', '.def', '.html', '.htm', '.csv', '.map',
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

const TCL_KEYWORDS = new Set([
  // Core Tcl
  'set', 'unset', 'if', 'else', 'elseif', 'for', 'foreach', 'while',
  'switch', 'proc', 'return', 'break', 'continue', 'expr', 'puts',
  'gets', 'string', 'lindex', 'llength', 'lappend', 'list', 'array',
  'dict', 'global', 'upvar', 'variable', 'namespace', 'source',
  'package', 'catch', 'error', 'eval', 'exec', 'open', 'close', 'read',
  'regexp', 'regsub', 'format', 'scan', 'incr', 'append', 'concat',
  // SDC commands
  'create_clock', 'create_generated_clock', 'set_input_delay',
  'set_output_delay', 'set_clock_uncertainty', 'set_clock_latency',
  'set_clock_groups', 'set_false_path', 'set_multicycle_path',
  'set_max_delay', 'set_min_delay', 'set_disable_timing',
  'set_load', 'set_drive', 'set_driving_cell', 'set_max_fanout',
  'set_max_transition', 'set_max_capacitance', 'set_min_capacitance',
  'set_case_analysis', 'set_propagated_clock', 'group_path',
  'all_inputs', 'all_outputs', 'all_clocks', 'all_registers',
  'get_ports', 'get_pins', 'get_nets', 'get_cells', 'get_clocks',
  'current_design', 'read_sdc', 'write_sdc', 'check_timing',
  // Yosys-ish
  'design', 'read_verilog', 'synth', 'flatten', 'opt', 'abc', 'write_verilog',
  'hierarchy', 'proc', 'memory', 'opt_clean', 'stat', 'check',
]);

function highlightTcl(text) {
  if (!text) return '';
  const out = [];
  let i = 0;
  const n = text.length;
  while (i < n) {
    const c = text[i];
    // Line comment '#' at start of line or after whitespace
    if (c === '#' && (i === 0 || /\s/.test(text[i - 1]))) {
      let j = text.indexOf('\n', i);
      if (j < 0) j = n;
      out.push(`<span class="syntax-comment">${_escHtml(text.slice(i, j))}</span>`);
      i = j;
      continue;
    }
    if (c === '"') {
      let j = i + 1;
      while (j < n && text[j] !== '"') {
        if (text[j] === '\\') j += 2; else j++;
      }
      j = Math.min(j + 1, n);
      out.push(`<span class="syntax-string">${_escHtml(text.slice(i, j))}</span>`);
      i = j;
      continue;
    }
    if (c === '$' && /[A-Za-z_]/.test(text[i + 1] || '')) {
      const m = text.slice(i).match(/^\$\{?[A-Za-z_][A-Za-z0-9_]*\}?/);
      if (m) {
        out.push(`<span class="syntax-attr">${_escHtml(m[0])}</span>`);
        i += m[0].length;
        continue;
      }
    }
    if (/[0-9]/.test(c)) {
      const m = text.slice(i).match(/^-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?/);
      if (m) {
        out.push(`<span class="syntax-number">${_escHtml(m[0])}</span>`);
        i += m[0].length;
        continue;
      }
    }
    if (/[A-Za-z_]/.test(c)) {
      const m = text.slice(i).match(/^[A-Za-z_][A-Za-z0-9_]*/);
      if (m) {
        const ident = m[0];
        if (TCL_KEYWORDS.has(ident)) {
          out.push(`<span class="syntax-keyword">${ident}</span>`);
        } else {
          out.push(_escHtml(ident));
        }
        i += ident.length;
        continue;
      }
    }
    out.push(_escHtml(c));
    i++;
  }
  return out.join('');
}

function highlightMarkdown(text) {
  if (!text) return '';
  // Process line by line so headings / lists / fences are easy to detect.
  const lines = text.split('\n');
  const out = [];
  let inFence = false;
  let fenceLang = '';
  let fenceBuf = [];

  function flushFence() {
    const inner = _escHtml(fenceBuf.join('\n'));
    const langTag = fenceLang ? ` <span class="syntax-attr">${_escHtml(fenceLang)}</span>` : '';
    out.push(`<span class="syntax-md-code-block"><span class="syntax-md-fence">\`\`\`${langTag}</span>\n${inner}\n<span class="syntax-md-fence">\`\`\`</span></span>`);
    fenceBuf = [];
    fenceLang = '';
  }

  for (const line of lines) {
    if (inFence) {
      const m = line.match(/^```\s*$/);
      if (m) {
        flushFence();
        inFence = false;
      } else {
        fenceBuf.push(line);
      }
      continue;
    }
    const fenceStart = line.match(/^```(\w*)/);
    if (fenceStart) {
      inFence = true;
      fenceLang = fenceStart[1];
      continue;
    }
    // Heading
    const heading = line.match(/^(#{1,6})\s+(.*)$/);
    if (heading) {
      out.push(`<span class="syntax-md-heading">${_escHtml(line)}</span>`);
      continue;
    }
    // List marker
    const list = line.match(/^(\s*)([-*+]|\d+\.)\s/);
    if (list) {
      const lead = line.slice(0, list[0].length);
      const rest = line.slice(list[0].length);
      out.push(`<span class="syntax-md-list">${_escHtml(lead)}</span>${_inlineMd(rest)}`);
      continue;
    }
    out.push(_inlineMd(line));
  }
  if (inFence) flushFence();
  return out.join('\n');
}

function _inlineMd(line) {
  // Process inline tokens left-to-right: backtick code → bold → italic → link
  // Use a small state-machine to avoid the JSX-style escape collisions.
  const out = [];
  let i = 0;
  const n = line.length;
  while (i < n) {
    const c = line[i];
    if (c === '`') {
      const end = line.indexOf('`', i + 1);
      if (end > i) {
        out.push(`<span class="syntax-md-code">${_escHtml(line.slice(i, end + 1))}</span>`);
        i = end + 1;
        continue;
      }
    }
    if (c === '*' && line[i + 1] === '*') {
      const end = line.indexOf('**', i + 2);
      if (end > i + 2) {
        const inner = _escHtml(line.slice(i + 2, end));
        out.push(`<span class="syntax-md-bold">**${inner}**</span>`);
        i = end + 2;
        continue;
      }
    }
    if (c === '*') {
      const end = line.indexOf('*', i + 1);
      if (end > i + 1 && !/^\s/.test(line[i + 1])) {
        out.push(`<span class="syntax-md-em">${_escHtml(line.slice(i, end + 1))}</span>`);
        i = end + 1;
        continue;
      }
    }
    if (c === '[') {
      const closeBr = line.indexOf('](', i + 1);
      const endParen = closeBr > i ? line.indexOf(')', closeBr + 2) : -1;
      if (closeBr > i && endParen > closeBr) {
        out.push(`<span class="syntax-md-link">${_escHtml(line.slice(i, endParen + 1))}</span>`);
        i = endParen + 1;
        continue;
      }
    }
    out.push(_escHtml(c));
    i++;
  }
  return out.join('');
}

function highlightHtml(text) {
  if (!text) return '';
  // Wrap tags and attribute names in spans; everything else escaped.
  let escaped = text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  // Tag opening / closing
  escaped = escaped.replace(/(&lt;\/?)([a-zA-Z][a-zA-Z0-9-]*)/g,
    (m, lead, tag) => `${lead}<span class="syntax-tag">${tag}</span>`);
  // Attribute name="value"
  escaped = escaped.replace(/([a-zA-Z-]+)=&quot;([^&]*?)&quot;/g,
    (m, attr, val) => `<span class="syntax-attr">${attr}</span>=<span class="syntax-string">"${val}"</span>`);
  // Comments
  escaped = escaped.replace(/&lt;!--([\s\S]*?)--&gt;/g,
    (m) => `<span class="syntax-comment">${m}</span>`);
  return escaped;
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

// Embed the self-hosted Surfer (Rust→WASM) waveform viewer for .vcd / .fst
// files. The bundle lives under /waveform-demos/surfer-local/ and Surfer's
// integration.js listens for {command: "LoadUrl", url} via postMessage.
function SurferWaveformViewer({ relPath }) {
  const iframeRef = useRef(null);
  const [status, setStatus] = useState('loading');
  const vcdUrl = `${location.origin}/api/artifacts/${relPath}`;
  useEffect(() => {
    const iframe = iframeRef.current;
    if (!iframe) return;
    const handleLoad = () => {
      // Give Surfer's WASM a beat to register its message listener.
      setTimeout(() => {
        try {
          iframe.contentWindow.postMessage({ command: 'LoadUrl', url: vcdUrl }, '*');
          setStatus('loaded');
        } catch (e) {
          setStatus('error: ' + e.message);
        }
      }, 1500);
    };
    iframe.addEventListener('load', handleLoad);
    return () => iframe.removeEventListener('load', handleLoad);
  }, [vcdUrl]);
  return (
    <div className="collateral-waveform">
      <div className="collateral-waveform-bar">
        <span className="collateral-waveform-engine">Surfer</span>
        <span className="collateral-waveform-status">{status}</span>
      </div>
      <iframe
        ref={iframeRef}
        className="collateral-waveform-frame"
        src="/waveform-demos/surfer-local/index.html"
        title="Surfer waveform viewer"
      />
    </div>
  );
}

function FileViewer({ file }) {
  const [content, setContent] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!file) return;
    const ext = _extOf(file.name);
    // HTML / images / waveforms are rendered via iframe or <img>, not
    // through the fetched text content. Skip the text fetch entirely.
    if (ext === '.html' || ext === '.htm' || ext === '.vcd' || ext === '.fst'
        || ext === '.ghw' || ext === '.png' || ext === '.jpg'
        || ext === '.jpeg' || ext === '.gif' || ext === '.svg'
        || ext === '.webp') {
      setContent(null);
      setError(null);
      setLoading(false);
      return;
    }
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
  const isWaveform = ext === '.vcd' || ext === '.fst' || ext === '.ghw';
  const isHtmlPreview = ext === '.html' || ext === '.htm';
  const isImage = ext === '.png' || ext === '.jpg' || ext === '.jpeg'
                 || ext === '.gif' || ext === '.svg' || ext === '.webp';

  let highlighted = null;
  if (content && !isHtmlPreview) {
    if (ext === '.v' || ext === '.sv') highlighted = highlightVerilog(content);
    else if (ext === '.json' || ext === '.map') highlighted = highlightJson(content);
    else if (ext === '.sdc' || ext === '.tcl' || ext === '.ys') highlighted = highlightTcl(content);
    else if (ext === '.md') highlighted = highlightMarkdown(content);
    else if (ext === '.html' || ext === '.htm') highlighted = highlightHtml(content);
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
        {isWaveform && (
          <SurferWaveformViewer relPath={file.rel_path} key={file.rel_path} />
        )}
        {isHtmlPreview && (
          <iframe
            key={file.rel_path}
            className="collateral-html-frame"
            src={`/api/artifacts/${file.rel_path}`}
            title={file.name}
            sandbox="allow-same-origin allow-popups allow-scripts"
          />
        )}
        {isImage && (
          <div className="collateral-image">
            <img src={`/api/artifacts/${file.rel_path}`} alt={file.name} />
          </div>
        )}
        {!isWaveform && !isHtmlPreview && !isImage && !isText && (
          <div className="collateral-empty">
            <div className="collateral-empty-icon">📦</div>
            <div>Binary file ({ext || 'unknown'}) — open via the link above.</div>
          </div>
        )}
        {!isWaveform && !isHtmlPreview && !isImage && loading && (
          <div className="collateral-loading">Loading…</div>
        )}
        {!isWaveform && !isHtmlPreview && !isImage && error && (
          <div className="collateral-error">Failed to load: {error}</div>
        )}
        {!isWaveform && !isHtmlPreview && !isImage && isText && content != null && !error && (
          <pre className="collateral-code">
            <code dangerouslySetInnerHTML={{ __html: highlighted }} />
          </pre>
        )}
      </div>
    </div>
  );
}

/** Build a nested folder tree from a flat list of {rel_path, ...} files. */
function _buildTree(files) {
  const root = { name: '', children: new Map(), files: [], isDir: true };
  for (const f of files) {
    const parts = f.rel_path.split('/').filter(Boolean);
    let node = root;
    for (let i = 0; i < parts.length - 1; i++) {
      const part = parts[i];
      if (!node.children.has(part)) {
        node.children.set(part, {
          name: part,
          path: parts.slice(0, i + 1).join('/'),
          children: new Map(),
          files: [],
          isDir: true,
        });
      }
      node = node.children.get(part);
    }
    node.files.push({ ...f, displayName: parts[parts.length - 1] });
  }
  // Sort: directories first (alphabetical), then files (alphabetical).
  const sortNode = (n) => {
    n.children = new Map([...n.children.entries()].sort((a, b) => a[0].localeCompare(b[0])));
    n.files.sort((a, b) => a.displayName.localeCompare(b.displayName));
    for (const child of n.children.values()) sortNode(child);
  };
  sortNode(root);
  return root;
}

function TreeNode({ node, depth, selected, onSelect, defaultOpen, filterTokens }) {
  const [open, setOpen] = useState(defaultOpen);
  if (!node) return null;
  const indent = { paddingLeft: 6 + depth * 12 };
  const matchesFilter = (text) => {
    if (!filterTokens || filterTokens.length === 0) return true;
    const t = text.toLowerCase();
    return filterTokens.every((q) => t.includes(q));
  };
  // Recursive count + filter so a search shows the surviving subtree only.
  const visibleFiles = node.files.filter((f) => matchesFilter(f.rel_path));
  const visibleChildren = [...node.children.values()].filter((c) => _treeHasMatches(c, filterTokens));
  if (filterTokens && filterTokens.length > 0 && visibleFiles.length === 0 && visibleChildren.length === 0) {
    return null;
  }
  return (
    <div className="collateral-tree-node">
      {node.path != null && node.name && (
        <div
          className="collateral-tree-folder"
          style={indent}
          onClick={() => setOpen((v) => !v)}
        >
          <span className="collateral-tree-chev">{open ? '▾' : '▸'}</span>
          <span className="collateral-tree-folder-icon">{'📁'}</span>
          <span className="collateral-tree-folder-name">{node.name}</span>
          <span className="collateral-tree-folder-count">
            {visibleFiles.length + visibleChildren.reduce((s, c) => s + _treeFileCount(c, filterTokens), 0)}
          </span>
        </div>
      )}
      {(open || node.path == null) && (
        <>
          {visibleChildren.map((child) => (
            <TreeNode
              key={child.path}
              node={child}
              depth={node.path != null ? depth + 1 : depth}
              selected={selected}
              onSelect={onSelect}
              defaultOpen={depth < 1}
              filterTokens={filterTokens}
            />
          ))}
          {visibleFiles.map((f) => (
            <div
              key={f.rel_path}
              className={`collateral-tree-file ${selected?.rel_path === f.rel_path ? 'collateral-tree-file-selected' : ''}`}
              style={{ paddingLeft: indent.paddingLeft + 14 }}
              onClick={() => onSelect(f)}
              title={f.rel_path}
            >
              <span className="collateral-tree-file-icon">{_fileIcon(f.displayName)}</span>
              <span className="collateral-tree-file-name">{f.displayName}</span>
              <span className="collateral-tree-file-size">{_formatBytes(f.size)}</span>
            </div>
          ))}
        </>
      )}
    </div>
  );
}

function _treeHasMatches(node, filterTokens) {
  if (!filterTokens || filterTokens.length === 0) return true;
  for (const f of node.files) {
    const t = f.rel_path.toLowerCase();
    if (filterTokens.every((q) => t.includes(q))) return true;
  }
  for (const child of node.children.values()) {
    if (_treeHasMatches(child, filterTokens)) return true;
  }
  return false;
}

function _treeFileCount(node, filterTokens) {
  let n = node.files.filter((f) => {
    if (!filterTokens || filterTokens.length === 0) return true;
    const t = f.rel_path.toLowerCase();
    return filterTokens.every((q) => t.includes(q));
  }).length;
  for (const child of node.children.values()) {
    n += _treeFileCount(child, filterTokens);
  }
  return n;
}

function _fileIcon(name) {
  const ext = _extOf(name);
  if (ext === '.v' || ext === '.sv') return '⚡';
  if (ext === '.vcd' || ext === '.fst' || ext === '.ghw') return '〰';
  if (ext === '.html' || ext === '.htm') return '🌐';
  if (ext === '.json') return '{ }';
  if (ext === '.md') return '📄';
  if (ext === '.rpt' || ext === '.log' || ext === '.txt') return '📋';
  if (ext === '.sdc' || ext === '.tcl' || ext === '.ys') return '⚙';
  if (ext === '.png' || ext === '.jpg' || ext === '.jpeg' || ext === '.svg' || ext === '.gif') return '🖼';
  return '·';
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

  // Flatten the kind-bucketed response into a single list keyed by rel_path
  // so we can build a real folder tree. Tag each file with its kind so we
  // can colour the icon by category later if we want.
  const { tree, totalFiles } = useMemo(() => {
    if (!data) return { tree: null, totalFiles: 0 };
    const all = [];
    const seen = new Set();
    for (const k of ['rtl', 'tb', 'syn', 'pnr', 'waveforms', 'reports']) {
      for (const f of (data[k] || [])) {
        if (seen.has(f.rel_path)) continue;
        seen.add(f.rel_path);
        all.push({ ...f, kind: k });
      }
    }
    return { tree: _buildTree(all), totalFiles: all.length };
  }, [data]);

  const filterTokens = useMemo(() => {
    return filter
      ? filter.toLowerCase().split(/\s+/).filter(Boolean)
      : [];
  }, [filter]);

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
          {tree && (
            <TreeNode
              node={tree}
              depth={0}
              selected={selected}
              onSelect={setSelected}
              defaultOpen={true}
              filterTokens={filterTokens}
            />
          )}
          {totalFiles === 0 && (
            <div className="collateral-empty-mini">No collateral yet.</div>
          )}
        </div>
      </div>
      <div className="collateral-viewer">
        <FileViewer file={selected} />
      </div>
    </div>
  );
}
