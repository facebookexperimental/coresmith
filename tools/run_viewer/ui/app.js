/* CoreSmith run viewer. Dependency-free; every piece of record content is
   inserted with textContent (never innerHTML), links are internal hash
   routes only, and data shards are loaded on demand with <script> tags so
   the viewer also works from file://. */
'use strict';
(function () {
  // ------------------------------------------------------------------ data loading
  const store = Object.create(null);
  const waiting = Object.create(null);
  const loading = Object.create(null);
  window.RV = {
    load(key, value) {
      store[key] = value;
      (waiting[key] || []).forEach(f => f(value));
      delete waiting[key];
    }
  };
  function shard(file, key) {
    if (key in store) return Promise.resolve(store[key]);
    if (loading[key]) return loading[key];
    loading[key] = new Promise((resolve, reject) => {
      (waiting[key] = waiting[key] || []).push(resolve);
      const s = document.createElement('script');
      s.src = file;
      s.async = true;
      s.onerror = () => { delete loading[key]; reject(new Error('could not load ' + file)); };
      document.head.appendChild(s);
    });
    return loading[key];
  }

  // ------------------------------------------------------------------ DOM helpers
  function h(tag, attrs, ...kids) {
    const el = document.createElement(tag);
    if (attrs) {
      for (const k of Object.keys(attrs)) {
        const v = attrs[k];
        if (v === null || v === undefined || v === false) continue;
        if (k === 'class') el.className = v;
        else if (k === 'href') { const sv = String(v); if (sv.startsWith('#')) el.setAttribute('href', sv); }
        else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2), v);
        else if (k === 'checked' || k === 'selected' || k === 'disabled') el[k] = !!v;
        else if (k === 'value') el.value = v;
        else if (['title', 'id', 'type', 'placeholder', 'colspan', 'rowspan', 'name', 'for', 'min', 'max', 'step'].includes(k) || k.startsWith('data-'))
          el.setAttribute(k, String(v));
      }
    }
    add(el, kids);
    return el;
  }
  function add(el, kids) {
    for (const k of kids) {
      if (k === null || k === undefined || k === false) continue;
      if (Array.isArray(k)) add(el, k);
      else if (k instanceof Node) el.appendChild(k);
      else el.appendChild(document.createTextNode(String(k)));
    }
    return el;
  }
  const SVGNS = 'http://www.w3.org/2000/svg';
  function sv(tag, attrs, ...kids) {
    const el = document.createElementNS(SVGNS, tag);
    for (const k of Object.keys(attrs || {})) {
      const v = attrs[k];
      if (v === null || v === undefined) continue;
      if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2), v);
      else el.setAttribute(k, String(v));
    }
    for (const c of kids) {
      if (c === null || c === undefined) continue;
      if (c instanceof Node) el.appendChild(c); else el.appendChild(document.createTextNode(String(c)));
    }
    return el;
  }
  const link = (href, ...kids) => h('a', { href }, ...kids);
  const enc = s => encodeURIComponent(String(s));

  // ------------------------------------------------------------------ formatting
  function fmtTs(ts, short) {
    if (ts === null || ts === undefined || ts === '') return '';
    const d = new Date(Number(ts) * 1000);
    if (isNaN(d)) return String(ts);
    const iso = d.toISOString();
    return short ? iso.slice(11, 19) : iso.slice(0, 19).replace('T', ' ') + 'Z';
  }
  function tsCell(ts) { return ts == null ? h('span', { class: 'muted' }, '—') : h('span', { title: fmtTs(ts) }, fmtTs(ts, true)); }
  function fmtDur(s) {
    if (s === null || s === undefined || isNaN(s)) return '';
    s = Number(s);
    if (s < 60) return s.toFixed(s < 10 ? 1 : 0) + 's';
    const m = Math.floor(s / 60), r = Math.round(s % 60);
    if (m < 60) return m + 'm ' + String(r).padStart(2, '0') + 's';
    return Math.floor(m / 60) + 'h ' + String(m % 60).padStart(2, '0') + 'm';
  }
  const num = v => (v === null || v === undefined) ? '' : (typeof v === 'number' ? v.toLocaleString('en-US') : String(v));
  const money = v => (typeof v === 'number') ? '$' + v.toFixed(2) : 'unknown';
  function badge(label, text) { return h('span', { class: 'badge b-' + (label || 'unlinked'), title: label || '' }, text || label || 'unlinked'); }
  function stat(text, cls) { return h('span', { class: 'badge s-' + String(text || '').replace(/[^a-z_-]/gi, '') + (cls ? ' ' + cls : '') }, text || '—'); }
  function kindBadge(k) { return h('span', { class: 'badge k-' + k }, k.replace('_', ' ')); }
  function jsonText(v) { try { return JSON.stringify(v, null, 1); } catch (e) { return String(v); } }

  // A text with an explicit full-detail control: nothing is cut silently.
  function textBlock(text, preview) {
    text = text === null || text === undefined ? '' : String(text);
    preview = preview || 4000;
    const wrap = h('div', { class: 'textblock' });
    const pre = h('pre');
    wrap.appendChild(pre);
    if (text.length <= preview) { pre.textContent = text; return wrap; }
    pre.textContent = text.slice(0, preview);
    const more = h('div', { class: 'more' });
    const btn = h('button', {
      onclick: () => {
        if (pre.dataset.full === '1') { pre.textContent = text.slice(0, preview); pre.dataset.full = '0'; btn.textContent = 'Show all ' + num(text.length) + ' characters'; note.textContent = ' preview: first ' + num(preview) + ' of ' + num(text.length) + ' characters'; }
        else { pre.textContent = text; pre.dataset.full = '1'; btn.textContent = 'Collapse to preview'; note.textContent = ' showing all ' + num(text.length) + ' characters'; }
      }
    }, 'Show all ' + num(text.length) + ' characters');
    const note = h('span', null, ' preview: first ' + num(preview) + ' of ' + num(text.length) + ' characters');
    add(more, [btn, note]);
    wrap.appendChild(more);
    return wrap;
  }
  function jsonBlock(v, label) {
    const d = h('details', null, h('summary', null, label || 'JSON'));
    d.addEventListener('toggle', () => { if (d.open && d.childNodes.length === 1) d.appendChild(textBlock(jsonText(v), 20000)); }, { once: false });
    return d;
  }
  function kv(pairs) {
    const g = h('div', { class: 'kv' });
    for (const [k, v] of pairs) {
      if (v === undefined) continue;
      g.appendChild(h('div', null, k));
      g.appendChild(h('div', null, v === null || v === '' ? h('span', { class: 'muted' }, '—') : v));
    }
    return g;
  }
  function section(title, ...kids) { return h('div', { class: 'panel' }, title ? h('h3', null, title) : null, ...kids); }

  // A sortable, paginated table. columns: [{t: title, v: row=>sortValue, r: row=>node, cls}]
  function table(columns, rows, opts) {
    opts = opts || {};
    const size = opts.pageSize || 200;
    let page = 0, sortCol = opts.sort === undefined ? -1 : opts.sort, desc = !!opts.desc;
    const wrap = h('div');
    function render() {
      wrap.textContent = '';
      let rs = rows.slice();
      if (sortCol >= 0 && columns[sortCol].v) {
        const f = columns[sortCol].v;
        rs.sort((a, b) => { const x = f(a), y = f(b); if (x === y) return 0; if (x === null || x === undefined) return 1; if (y === null || y === undefined) return -1; return (x < y ? -1 : 1) * (desc ? -1 : 1); });
      }
      const pages = Math.max(1, Math.ceil(rs.length / size));
      if (page >= pages) page = pages - 1;
      const pager = h('div', { class: 'pager' },
        h('span', { class: 'muted' }, num(rs.length) + ' rows'),
        pages > 1 ? [h('button', { onclick: () => { page = Math.max(0, page - 1); render(); }, disabled: page === 0 }, '‹ prev'),
          h('span', null, 'page ' + (page + 1) + ' / ' + pages),
          h('button', { onclick: () => { page = Math.min(pages - 1, page + 1); render(); }, disabled: page >= pages - 1 }, 'next ›')] : null);
      const thead = h('tr', null, columns.map((c, i) => h('th', {
        class: c.cls || '', title: c.v ? 'sort' : '',
        onclick: () => { if (!c.v) return; if (sortCol === i) desc = !desc; else { sortCol = i; desc = false; } render(); }
      }, c.t + (sortCol === i ? (desc ? ' ▾' : ' ▴') : ''))));
      const body = h('tbody');
      for (const r of rs.slice(page * size, page * size + size)) {
        const tr = h('tr', { class: opts.rowClass ? opts.rowClass(r) : '' }, columns.map(c => h('td', { class: c.cls || '' }, c.r(r))));
        body.appendChild(tr);
      }
      add(wrap, [pager, h('table', null, h('thead', null, thead), body), rs.length > size ? pager.cloneNode(true) : null]);
      if (rs.length > size) {
        const p2 = wrap.lastChild;
        const btns = p2.querySelectorAll('button');
        if (btns[0]) btns[0].onclick = () => { page = Math.max(0, page - 1); render(); };
        if (btns[1]) btns[1].onclick = () => { page = Math.min(pages - 1, page + 1); render(); };
      }
    }
    render();
    return wrap;
  }

  // ------------------------------------------------------------------ index lookups
  let IDX = null;
  const SRC = Object.create(null);
  const L = Object.create(null);     // per-arm lookup maps
  function arm(name) { return IDX.arms[name]; }
  function lk(name) {
    if (L[name]) return L[name];
    const a = arm(name), m = { actions: {}, shell: {}, builds: {}, agents: {}, parks: {}, runs: {}, steps: {}, ev: {}, shellBySeq: {} };
    a.actions.forEach(x => m.actions[x.id] = x);
    a.shell.forEach(x => { m.shell[x.id] = x; m.shellBySeq[x.agent + ':' + x.seq] = x; });
    a.builds.forEach(x => m.builds[x.id] = x);
    a.agents.forEach(x => m.agents[x.id] = x);
    a.parks.forEach(x => m.parks[x.id] = x);
    a.graph_runs.forEach(x => m.runs[x.id] = x);
    a.steps.forEach(x => m.steps[x.id] = x);
    ['dv', 'coverage', 'ppa'].forEach(k => { m['ev_' + k] = {}; (a.evidence[k] || []).forEach(r => m['ev_' + k][r.id] = r); });
    L[name] = m;
    return m;
  }
  function srcText(r) {
    if (!r) return '';
    const s = SRC[r.s];
    const p = s ? s.path : r.s;
    return p + (r.l !== undefined ? ':' + r.l : '') + (r.k !== undefined ? ' [' + r.k + ']' : '');
  }
  function srcRef(r) { return r ? h('span', { class: 'mono small muted', title: 'private snapshot source (not a download link)' }, srcText(r)) : null; }
  function refsList(refs) {
    if (!refs || !refs.length) return null;
    if (refs.length === 1) return srcRef(refs[0]);
    const d = h('details', { class: 'small' }, h('summary', null, refs.length + ' copies (first is canonical)'));
    refs.forEach(r => d.appendChild(h('div', null, srcRef(r))));
    return d;
  }
  const agentLabel = (armName, id) => { const a = lk(armName).agents[id]; return a ? a.label : id; };
  const aLink = (armName, id, seq) => link('#/a/' + enc(armName) + '/agent/' + enc(id) + (seq ? '/' + seq : ''), id + (seq ? ' #' + seq : ''));
  const bLink = (armName, id) => link('#/a/' + enc(armName) + '/build/' + enc(id), id);
  const cliLink = (armName, id) => link('#/a/' + enc(armName) + '/cli/' + id, '#' + id);
  const callLink = (armName, id) => link('#/a/' + enc(armName) + '/call/' + enc(id), id);
  const evLink = (armName, id) => link('#/a/' + enc(armName) + '/event/' + enc(id), id);
  const runLink = (armName, id) => link('#/a/' + enc(armName) + '/graph/' + enc(id), id);

  // ------------------------------------------------------------------ chrome
  const TABS = [['', 'Overview'], ['cli', 'CLI ledger'], ['builds', 'Builds'], ['graphs', 'Graph runs'], ['events', 'Graph events'],
    ['agents', 'Trajectories'], ['helpers', 'Helper calls'], ['state', 'Parks & state']];
  function chrome(route) {
    const arms = document.getElementById('arms');
    arms.textContent = '';
    const cur = route.arm;
    Object.keys(IDX.arms).forEach(n => arms.appendChild(h('a', { href: '#/a/' + enc(n) + (route.tab ? '/' + route.tab : ''), class: n === cur ? 'on' : '' }, n)));
    const tabs = document.getElementById('tabs');
    tabs.textContent = '';
    const armName = cur || Object.keys(IDX.arms)[0];
    const tabOf = { build: 'builds', call: 'cli', agent: 'agents', graph: 'graphs', event: 'events' };
    const curTab = tabOf[route.tab] !== undefined ? tabOf[route.tab] : (route.tab || '');
    TABS.forEach(([t, label]) => tabs.appendChild(h('a', { href: '#/a/' + enc(armName) + (t ? '/' + t : ''), class: cur && curTab === t ? 'on' : '' }, label)));
    tabs.appendChild(h('a', { href: '#/provenance', class: route.view === 'provenance' ? 'on' : '' }, 'Provenance'));
    if (IDX.has_analysis) tabs.appendChild(h('a', { href: '#/analysis', class: route.view === 'analysis' ? 'on' : '' }, 'Analysis'));
    const bar = document.getElementById('snapbar');
    bar.textContent = '';
    const sn = IDX.snapshot, man = sn.manifest || {};
    add(bar, ['Snapshot collected ', h('b', null, fmtTs(Date.parse(man.started_at) / 1000)), ' – ', h('b', null, fmtTs(Date.parse(man.finished_at) / 1000)),
      sn.ready ? [' · READY.json present'] : [' · ', h('span', { class: 'err' }, 'no READY.json (collection not marked complete)')],
      ' · not an atomic cross-file snapshot (SQLite files are online backups, logs are bounded copies) · exported ', fmtTs(Date.parse(IDX.generated_at) / 1000),
      ' · privacy check ', IDX.privacy.ok ? h('span', { class: 'badge b-exact' }, 'passed') : h('span', { class: 'badge b-missing' }, 'FAILED')]);
  }

  // ------------------------------------------------------------------ router
  function parse() {
    const raw = location.hash.replace(/^#/, '') || '/';
    const [path, qs] = raw.split('?');
    const parts = path.split('/').filter(Boolean).map(decodeURIComponent);
    const q = new URLSearchParams(qs || '');
    if (parts[0] === 'a') return { arm: parts[1], tab: parts[2] || '', id: parts[3], sub: parts[4], q };
    return { view: parts[0] || 'home', id: parts[1], q };
  }
  async function route() {
    const r = parse();
    chrome(r);
    const main = document.getElementById('main');
    let node;
    try {
      if (r.arm) {
        if (!IDX.arms[r.arm]) node = h('p', { class: 'err' }, 'Unknown arm ' + r.arm);
        else node = await (VIEWS[r.tab || ''] || viewMissing)(r.arm, r);
      } else node = await (PAGES[r.view] || viewMissing)(r);
    } catch (e) {
      node = h('div', { class: 'badbox' }, 'Could not render this view: ' + (e && e.message ? e.message : String(e)));
      console.error(e);
    }
    main.textContent = '';
    main.appendChild(node);
    if (r.q && r.q.get('t')) {
      const el = document.getElementById('turn-' + r.q.get('t'));
      if (el) { el.classList.add('hl'); el.scrollIntoView({ block: 'center' }); }
    } else window.scrollTo(0, 0);
  }
  function viewMissing() { return h('p', { class: 'err' }, 'Unknown view.'); }

  // ------------------------------------------------------------------ overview
  function stagesRow(a) {
    return h('div', { class: 'chips' }, (a.stages || []).map(s => h('span', { class: 'badge s-' + s.status, title: s.name + ': ' + s.status + (s.done_ts ? ' at ' + fmtTs(s.done_ts) : '') }, s.name)));
  }
  function buildCounts(a) {
    const c = {};
    a.builds.forEach(b => c[b.status] = (c[b.status] || 0) + 1);
    return h('div', { class: 'chips' }, Object.keys(c).sort().map(k => h('span', { class: 'badge s-' + k }, k + ' ' + c[k])));
  }
  function costLine(a) {
    const arch = (a.usage.architect || [])[0] || {};
    const snap = arch.latest_cumulative_snapshot;
    const codex = arch.codex_thread_total_latest;
    const out = [];
    if (snap) out.push(h('div', null, 'Architect provider-reported cost: ', h('b', null, money(snap.totalCostUSD ?? snap.total_cost_usd)), h('span', { class: 'muted small' }, ' (latest cumulative session snapshot; never summed)')));
    else if (codex) out.push(h('div', null, 'Architect thread tokens: ', h('b', null, num((codex.total_token_usage || {}).total_tokens)), h('span', { class: 'muted small' }, ' (latest cumulative Codex counter; provider cost unavailable)')));
    else out.push(h('div', { class: 'muted' }, 'Architect usage: unavailable'));
    out.push(h('div', null, 'Engine helper cost: ', h('b', null, a.usage.helper_cost_usd_sum_over_calls === null ? 'unknown' : money(a.usage.helper_cost_usd_sum_over_calls)),
      h('span', { class: 'muted small' }, ' over finished calls; ' + a.usage.helper_calls_without_cost + ' call(s) without a cost are unknown, not zero')));
    return out;
  }
  function evidenceIssues(a) {
    const out = [];
    const rot = a.builds.filter(b => b.status === 'completed' && !b.lineage.engine.ok && b.lineage.viewer.ok);
    if (rot.length) {
      const byReason = {};
      rot.forEach(b => b.lineage.engine.reasons.forEach(r => (byReason[r] = byReason[r] || []).push(b)));
      out.push(h('div', { class: 'warnbox' }, h('b', null, 'Engine lineage false negatives: '), rot.length + ' completed build(s) fail the engine\'s lineage predicate but pass when every event file and the checkpointed ancestor namespace are read.',
        h('ul', null, Object.keys(byReason).map(r => h('li', null, r + ' — ', byReason[r].map((b, i) => [i ? ', ' : '', bLink(a.arm, b.id)]))))));
    }
    const pend = a.parks.filter(p => p.status === 'pending');
    if (pend.length) out.push(h('div', { class: 'infobox' }, h('b', null, pend.length + ' open park(s): '), pend.map((p, i) => [i ? ', ' : '', p.kind + (p.block ? ' [' + p.block + ']' : '') + ' since ' + fmtTs(p.ts, true)])));
    const unf = a.agents.filter(x => x.engine_call && x.engine_call.kind === 'unfinished');
    if (unf.length) out.push(h('div', { class: 'infobox' }, h('b', null, unf.length + ' helper call(s) unfinished at the snapshot: '), unf.map((x, i) => [i ? ', ' : '', aLink(a.arm, x.id)])));
    return out;
  }
  function armCard(a) {
    const st = a.status || {};
    const inv = (a.invocations || []);
    return h('div', { class: 'panel' },
      h('div', { class: 'row' }, h('h2', { style: null }, link('#/a/' + enc(a.arm), a.arm)), h('span', { class: 'spacer' }), stat(st.state || 'unknown')),
      kv([['provider / model', (a.config.provider || a.provider) + ' · ' + (a.config.model || '') + (a.config.effort ? ' · ' + a.config.effort : '')],
        ['Architect session', h('span', { class: 'mono small' }, st.session_id || '—')],
        ['run started', fmtTs(Date.parse(st.run_started_at) / 1000)], ['last event', fmtTs(Date.parse(st.last_event_at) / 1000)],
        ['invocations', inv.length + (st.exit_code !== undefined ? ' · last exit ' + st.exit_code : '') + (st.provider_error ? ' · provider error' : '')],
        ['stages', stagesRow(a)], ['builds', buildCounts(a)],
        ['CLI audit rows', num(a.counts.actions) + ' (' + num(a.counts.observed_invocations_missing_from_audit) + ' observed calls not audited)'],
        ['graph events', num(a.counts.events) + ' in ' + a.counts.event_files + ' file(s)'],
        ['helper calls', num(a.counts.helper_calls) + (a.counts.helpers_unfinished ? ' + ' + a.counts.helpers_unfinished + ' unfinished' : '')],
        ['trajectories', num(a.counts.agents) + ' · ' + num(a.counts.turns) + ' turns']]),
      h('div', { style: null }, costLine(a)),
      evidenceIssues(a));
  }
  function home() {
    const arms = Object.values(IDX.arms);
    const sn = IDX.snapshot;
    return h('div', null,
      h('h1', null, 'CoreSmith treatment runs'),
      h('p', { class: 'muted' }, 'Every CLI call, graph build and thread, node event, helper trajectory and park in the snapshot, with the relationship between them labelled exact, strong, weak, ambiguous or unlinked (see Provenance).'),
      sn.manifest && sn.manifest.consistency ? h('div', { class: 'infobox small' }, h('b', null, 'Snapshot consistency: '), sn.manifest.consistency, ' Records written after a source file was copied are absent; an in-flight build, park or helper call shows its state at copy time.') : null,
      h('div', { class: 'grid two' }, arms.map(armCard)),
      h('h2', null, 'Timelines'),
      arms.map(a => h('div', { class: 'panel' }, h('h3', null, a.arm), timeline(a))));
  }

  // ------------------------------------------------------------------ timeline
  function timeline(a) {
    const items = [];
    const st = a.status || {};
    let t0 = Date.parse(st.run_started_at) / 1000, t1 = Date.parse(st.last_event_at) / 1000;
    const lanes = [];
    const laneOf = name => { let i = lanes.indexOf(name); if (i < 0) { lanes.push(name); i = lanes.length - 1; } return i; };
    laneOf('stages');
    (a.stages || []).forEach(s => { if (s.entered_ts) items.push({ lane: 'stages', from: s.entered_ts, to: s.done_ts || t1, label: s.name + ' (' + s.status + ')', cls: s.status === 'done' ? '#9bd1a8' : '#9cbcf0', href: '#/a/' + enc(a.arm) + '/state' }); });
    const mods = [...new Set(a.builds.map(b => b.module))];
    mods.forEach(m => laneOf('build ' + m));
    const colors = { completed: '#3f9a5c', failed: '#d4483b', aborted: '#9aa1ab', running: '#3a78d6', parked: '#e0a33a', error: '#d4483b' };
    a.builds.forEach(b => items.push({ lane: 'build ' + b.module, from: b.started_at || b.requested_at, to: b.finished_at || t1, label: b.id + ' ' + b.status + ' (' + b.graph + ')', cls: colors[b.status] || '#888', href: '#/a/' + enc(a.arm) + '/build/' + enc(b.id) }));
    laneOf('helpers');
    a.agents.filter(x => x.engine_call).forEach(x => { const c = x.engine_call; if (c.start_ts) items.push({ lane: 'helpers', from: c.start_ts, to: c.ts || t1, label: x.label + (c.duration_s ? ' · ' + fmtDur(c.duration_s) : ''), cls: c.error ? '#d98a83' : '#a98ee0', href: '#/a/' + enc(a.arm) + '/agent/' + enc(x.id) }); });
    (a.graph_runs || []).forEach(r => { if (r.first_ts) { const ln = 'graph ' + r.graph; laneOf(ln); items.push({ lane: ln, from: r.first_ts, to: r.last_ts || r.first_ts, label: r.id + ' (' + r.checkpoints + ' checkpoints)', cls: '#7fb3c9', href: '#/a/' + enc(a.arm) + '/graph/' + enc(r.id) }); } });
    laneOf('parks');
    a.parks.forEach(p => items.push({ lane: 'parks', from: p.ts, to: p.resolved_ts || (p.status === 'pending' ? t1 : p.ts + 30), label: p.kind + ' ' + (p.block || '') + ' ' + p.status, cls: p.status === 'pending' ? '#e0a33a' : '#c9b26b', href: '#/a/' + enc(a.arm) + '/state' }));
    a.agents.filter(x => x.kind !== 'engine-helper').forEach(x => { const s = x.stats || {}; if (s.first_ts) { const ln = x.kind === 'architect' ? 'Architect' : 'sub-agents'; laneOf(ln); items.push({ lane: ln, from: s.first_ts, to: s.last_ts || s.first_ts, label: x.label + ' · ' + num(s.turns) + ' turns', cls: x.kind === 'architect' ? '#5d6b7d' : '#93a1b3', href: '#/a/' + enc(a.arm) + '/agent/' + enc(x.id) }); } });
    if (isNaN(t0)) t0 = Infinity;
    if (isNaN(t1)) t1 = -Infinity;
    items.forEach(i => { if (i.from && i.from < t0) t0 = i.from; if (i.to && i.to > t1) t1 = i.to; });
    if (!isFinite(t0) || !isFinite(t1)) return h('p', { class: 'muted' }, 'No timed records.');
    const W = 1400, left = 150, lh = 18, top = 22;
    const H = top + lanes.length * lh + 6;
    const x = t => left + (W - left - 10) * ((t - t0) / Math.max(1, t1 - t0));
    const svg = sv('svg', { width: W, height: H, viewBox: '0 0 ' + W + ' ' + H });
    for (let hr = Math.ceil(t0 / 3600) * 3600; hr <= t1; hr += 3600) {
      svg.appendChild(sv('line', { x1: x(hr), x2: x(hr), y1: 14, y2: H, stroke: '#e3e6ea' }));
      svg.appendChild(sv('text', { x: x(hr) + 2, y: 11 }, fmtTs(hr, true).slice(0, 5)));
    }
    lanes.forEach((ln, i) => {
      svg.appendChild(sv('rect', { class: 'lane', x: left, y: top + i * lh, width: W - left - 10, height: lh - 3 }));
      svg.appendChild(sv('text', { x: 4, y: top + i * lh + 12 }, ln.length > 24 ? ln.slice(0, 23) + '…' : ln));
    });
    items.forEach(i => {
      if (!i.from) return;
      const li = lanes.indexOf(i.lane);
      const r = sv('rect', { class: 'bar', x: x(i.from), y: top + li * lh + 2, width: Math.max(2, x(i.to || i.from) - x(i.from)), height: lh - 7, fill: i.cls, rx: 2, onclick: () => { location.hash = i.href; } },
        sv('title', {}, i.label + '\n' + fmtTs(i.from) + ' → ' + (i.to ? fmtTs(i.to) : '…')));
      svg.appendChild(r);
    });
    return h('div', { class: 'timeline' }, svg);
  }

  // ------------------------------------------------------------------ arm overview
  function viewArm(name) {
    const a = arm(name);
    const modules = {};
    a.builds.forEach(b => (modules[b.module] = modules[b.module] || []).push(b));
    const resultsBest = {};
    (a.results || []).forEach(r => { if (r.kind === 'best' && r.value) resultsBest[r.block] = r.value.build_id; });
    return h('div', null,
      h('h1', null, name),
      armCard(a),
      h('h2', null, 'Timeline'), timeline(a),
      h('h2', null, 'Modules'),
      table([
        { t: 'module', v: r => r[0], r: r => r[0] },
        { t: 'published at snapshot (results.best)', r: r => resultsBest[r[0]] ? bLink(name, resultsBest[r[0]]) : h('span', { class: 'muted' }, 'none') },
        { t: 'builds (oldest first)', r: r => h('div', { class: 'chips' }, r[1].slice().sort((x, y) => (x.requested_at || 0) - (y.requested_at || 0)).map(b => h('a', { href: '#/a/' + enc(name) + '/build/' + enc(b.id), class: 'badge s-' + b.status, title: b.id + ' · ' + b.graph + ' · ' + b.status }, b.status + ' ' + fmtTs(b.requested_at, true)))) }
      ], Object.entries(modules)),
      h('h2', null, 'Architect invocations'),
      (a.invocations || []).map(inv => section('invocation ' + inv.name,
        kv([['agent', inv.agent ? aLink(name, inv.agent) : '—'], ['state', (inv.status || {}).state], ['started', (inv.status || {}).invocation_started_at], ['finished', (inv.status || {}).invocation_finished_at],
          ['exit code', (inv.status || {}).exit_code], ['launcher argv', h('span', { class: 'mono small' }, (inv.command || []).join(' '))],
          ['results', (inv.results || []).map(x => h('div', { class: 'small' }, x.subtype + (x.is_error ? ' (error)' : '') + ' · turns ' + x.num_turns + ' · cumulative cost ' + money(x.total_cost_usd) + ' · ', srcRef(x.src)))]]),
        inv.prompt_txt ? h('details', null, h('summary', null, 'prompt'), textBlock(inv.prompt_txt)) : null,
        inv.response_txt ? h('details', null, h('summary', null, 'response.txt'), textBlock(inv.response_txt)) : null,
        inv.stderr_log ? h('details', null, h('summary', null, 'stderr.log'), textBlock(inv.stderr_log)) : null)),
      (a.interventions || []).length ? section('Coordinator interventions', textBlock(jsonText(a.interventions))) : null);
  }

  // ------------------------------------------------------------------ CLI ledger
  function argvText(argv) { return (argv || []).map(t => /\s/.test(t) ? JSON.stringify(t) : t).join(' '); }
  function viewCli(name, r) {
    const a = arm(name), m = lk(name);
    if (r.id) return viewAction(name, r.id);
    const rows = a.actions.map(x => ({ t: 'audit', ts: x.ts, x }));
    a.missing.forEach(x => rows.push({ t: 'observed', ts: x.ts, x }));
    rows.sort((p, q) => (p.ts || 0) - (q.ts || 0));
    const verbs = [...new Set(rows.map(row => (row.x.argv || [])[0]).filter(Boolean))].sort();
    const st = { q: r.q.get('q') || '', verb: r.q.get('verb') || '', rc: r.q.get('rc') || '', label: r.q.get('label') || '', kind: r.q.get('kind') || '' };
    const host = h('div');
    function draw() {
      const ql = st.q.toLowerCase();
      const rs = rows.filter(row => {
        const x = row.x;
        if (st.kind && row.t !== st.kind) return false;
        if (st.verb && (x.argv || [])[0] !== st.verb) return false;
        if (st.rc === '0' && !(row.t === 'audit' && x.rc === 0)) return false;
        if (st.rc === 'nz' && !(row.t === 'audit' && x.rc !== 0)) return false;
        if (st.label && !(row.t === 'audit' && x.link === st.label)) return false;
        if (ql) {
          const hay = (argvText(x.argv) + ' ' + (x.summary || '') + ' ' + (x.actor || '') + ' ' + (x.run_id || '') + ' #' + (x.id || '') + ' ' + (x.why || '') + ' ' + (x.agent || '')).toLowerCase();
          if (!hay.includes(ql)) return false;
        }
        return true;
      });
      host.textContent = '';
      host.appendChild(table([
        { t: 'row', v: row => row.t === 'audit' ? row.x.id : 1e9, r: row => row.t === 'audit' ? cliLink(name, row.x.id) : h('span', { class: 'badge b-missing', title: row.x.why }, 'not audited') },
        { t: 'time (UTC)', v: row => row.ts, r: row => tsCell(row.ts) },
        { t: 'actor', v: row => row.x.actor || row.x.agent, r: row => row.t === 'audit' ? (row.x.actor || '') : aLink(name, row.x.agent) },
        { t: 'argv', cls: 'wrap mono', r: row => argvText(row.x.argv) },
        { t: 'rc', cls: 'num', v: row => row.t === 'audit' ? row.x.rc : null, r: row => row.t === 'audit' ? h('span', { class: row.x.rc ? 'err' : '' }, String(row.x.rc)) : '' },
        { t: 'summary / reason', cls: 'wrap', r: row => row.t === 'audit' ? (row.x.summary || '') : h('span', { class: 'small' }, row.x.code + ': ' + row.x.why) },
        { t: 'run id', r: row => h('span', { class: 'mono small' }, row.x.run_id || '') },
        { t: 'native call', r: row => row.t === 'audit' ? nativeCell(name, row.x) : callLink(name, row.x.call) },
        { t: 'graph', r: row => row.t === 'audit' ? h('div', { class: 'chips' }, (row.x.builds || []).map(b => [badge(b.label), ' ', bLink(name, b.build)]), (row.x.graph_runs || []).map(g => [badge(g.label), ' ', runLink(name, g.run)])) : '' }
      ], rs, { rowClass: row => row.t === 'observed' ? 'observed' : '' }));
    }
    const sel = (key, opts) => h('select', { onchange: e => { st[key] = e.target.value; draw(); } }, opts.map(([v, t]) => h('option', { value: v, selected: st[key] === v }, t)));
    const qbox = h('input', { type: 'search', placeholder: 'filter argv, summary, actor, run id…', value: st.q, oninput: e => { st.q = e.target.value; draw(); } });
    draw();
    const j = a.cli_join;
    return h('div', null,
      h('h1', null, name + ' — CLI ledger'),
      h('p', { class: 'muted' }, 'Audited rows come from project.sqlite actions (written when the verb returns: the time is the end time). Rows marked "not audited" are coresmith invocations visible in a native shell command line with no audit row: help output, parser failures and verbs dispatched without the audit hook. A command quoted inside output or a document is never counted.'),
      h('div', { class: 'row small' }, 'audit ↔ native shell joins: ', badge('strong', 'strong ' + j.strong), badge('weak', 'weak ' + j.weak), badge('ambiguous', 'ambiguous ' + j.ambiguous), badge('unlinked', 'unlinked ' + j.unlinked),
        ' · not audited: ', Object.entries(a.missing_by_code || {}).map(([k, v]) => h('span', { class: 'badge b-missing' }, k + ' ' + v)),
        (a.cli_wrappers || []).length ? [' · wrapper scripts recognised: ', a.cli_wrappers.map(w => h('span', { class: 'mono small', title: 'defined by ' + w.defined_by + ' via ' + w.via }, w.path + ' '))] : null),
      h('div', { class: 'filters' }, qbox,
        h('label', null, 'rows ', sel('kind', [['', 'audited + not audited'], ['audit', 'audited only'], ['observed', 'not audited only']])),
        h('label', null, 'verb ', sel('verb', [['', 'any']].concat(verbs.map(v => [v, v])))),
        h('label', null, 'rc ', sel('rc', [['', 'any'], ['0', '0'], ['nz', 'non-zero']])),
        h('label', null, 'native join ', sel('label', [['', 'any'], ['strong', 'strong'], ['weak', 'weak'], ['ambiguous', 'ambiguous'], ['unlinked', 'unlinked']]))),
      host);
  }
  function nativeCell(name, x) {
    const n = x.native || [];
    if (!n.length) return h('span', null, badge('unlinked'), (x.context || []).length ? h('span', { class: 'small muted', title: x.context.map(c => c.call).join(', ') }, ' ' + x.context.length + ' context') : null);
    const main = n.find(y => y.label !== 'alternative' && y.label !== 'ambiguous') || n[0];
    return h('span', null, badge(main.label), ' ', callLink(name, main.call), n.length > 1 ? h('span', { class: 'small muted' }, ' +' + (n.length - 1)) : null);
  }
  function viewAction(name, id) {
    const a = arm(name), m = lk(name), x = m.actions[id];
    if (!x) return h('p', { class: 'err' }, 'No audit row #' + id);
    const prev = m.actions[Number(id) - 1], next = m.actions[Number(id) + 1];
    return h('div', null,
      h('div', { class: 'row' }, h('h1', null, name + ' — CLI audit row #' + id), h('span', { class: 'spacer' }), prev ? link('#/a/' + enc(name) + '/cli/' + prev.id, '‹ #' + prev.id) : null, ' ', next ? link('#/a/' + enc(name) + '/cli/' + next.id, '#' + next.id + ' ›') : null),
      section(null, kv([['argv', h('span', { class: 'mono' }, argvText(x.argv))], ['time (end of the call)', fmtTs(x.ts)], ['actor', x.actor], ['exit code', String(x.rc)], ['summary', x.summary], ['run id', x.run_id], ['source', srcRef(x.src)], ['note', x.context_note]])),
      section('Native shell call', (x.native || []).length ? (x.native || []).map(n => {
        const c = m.shell[n.call];
        return h('div', { class: 'panel' }, h('div', { class: 'row' }, badge(n.label), callLink(name, n.call), h('span', { class: 'small muted' }, n.basis || '')),
          c ? kv([['agent', aLink(name, c.agent, c.seq)], ['tool', c.tool], ['window', fmtTs(c.start_ts) + ' → ' + (c.end_ts ? fmtTs(c.end_ts) : 'not observed') + ' (' + c.end_basis + ')'], ['command', textBlock(c.command_preview + (c.command_len > c.command_preview.length ? '\n… ' + (c.command_len - c.command_preview.length) + ' more characters in the turn' : ''), 1000)]]) : h('span', { class: 'muted' }, 'call details not exported'));
      }) : [h('p', null, badge('unlinked'), ' No native shell call shows this invocation.'), (x.context || []).length ? h('div', null, h('p', { class: 'small muted' }, 'Scripts or loops running at the time (context only, not a link):'), x.context.map(c => h('div', null, callLink(name, c.call)))) : null]),
      (x.builds || []).length ? section('Builds', x.builds.map(b => h('div', null, badge(b.label), ' ', bLink(name, b.build), h('span', { class: 'small muted' }, ' ' + b.basis)))) : null,
      (x.graph_runs || []).length ? section('Graph runs', x.graph_runs.map(g => h('div', null, badge(g.label), ' ', runLink(name, g.run)))) : null,
      (x.interrupts || []).length ? section('Parks', x.interrupts.map(p => h('div', null, badge(p.label), ' ', link('#/a/' + enc(name) + '/state', p.interrupt)))) : null);
  }
  function viewCall(name, r) {
    const m = lk(name), c = m.shell[r.id];
    if (!c) return h('p', { class: 'err' }, 'No exported shell call ' + r.id + ' (only calls that show a coresmith invocation or are linked to an audit row are indexed; every call is in its trajectory).');
    const host = h('div', null, h('p', { class: 'muted' }, 'loading the turn…'));
    const ag = m.agents[c.agent];
    loadTurns(name, ag, [c.seq, c.result_seq].filter(Boolean)).then(turns => {
      host.textContent = '';
      const tu = turns[c.seq], tr = turns[c.result_seq];
      add(host, [section('Command (tool input)', tu ? textBlock(tu.text, 8000) : 'not found'), section('Output (tool result)', tr ? textBlock(tr.text, 8000) : h('span', { class: 'muted' }, 'no result record'))]);
    }).catch(e => { host.textContent = 'could not load: ' + e.message; });
    return h('div', null,
      h('h1', null, name + ' — native shell call ' + c.id),
      section(null, kv([['agent', aLink(name, c.agent, c.seq)], ['tool', c.tool], ['start', fmtTs(c.start_ts)], ['end', c.end_ts ? fmtTs(c.end_ts) : 'not observed'], ['end basis', c.end_basis],
        ['exit code', c.exit_code === null ? 'unknown' : String(c.exit_code)], ['background', c.background ? 'yes' : 'no'], ['loop / script / heredoc', [c.loop ? 'loop' : null, c.loop_expanded ? '(literal for-loop expanded)' : null, c.scripted ? 'script' : null, c.heredoc ? 'heredoc' : null].filter(Boolean).join(' ') || '—']])),
      section('Visible coresmith invocations', table([
        { t: '#', r: (inv) => String(c.invocations.indexOf(inv) + 1) },
        { t: 'argv', cls: 'wrap mono', r: inv => argvText(inv.argv) },
        { t: 'via', r: inv => inv.via_wrapper ? h('span', { class: 'mono small' }, 'wrapper ' + inv.via_wrapper) : (inv.loop_var ? 'for ' + inv.loop_var + '=' + inv.loop_value : 'direct') },
        { t: 'audit row', r: inv => { const i = c.invocations.indexOf(inv); const id = (c.invocation_actions || [])[i]; return id ? cliLink(name, id) : (c.missing || []).find(mm => argvText(mm.argv) === argvText(inv.argv)) ? h('span', { class: 'badge b-missing' }, 'not audited') : '—'; } }
      ], c.invocations, { pageSize: 500 })),
      (c.audit || []).length ? section('Audit rows linked to this call', c.audit.map(x => h('div', null, badge(x.label), ' ', cliLink(name, x.action), ' ', h('span', { class: 'mono small' }, argvText((m.actions[x.action] || {}).argv))))) : null,
      (c.missing || []).length ? section('Observed but not audited', c.missing.map(x => h('div', null, h('span', { class: 'badge b-missing' }, x.code), ' ', h('span', { class: 'mono small' }, argvText(x.argv)), ' — ', x.why))) : null,
      host);
  }

  // ------------------------------------------------------------------ builds
  function lineageChip(b, which) {
    const L2 = b.lineage[which];
    return h('span', { class: 'badge ' + (L2.ok ? 'b-exact' : 'b-missing'), title: L2.reasons.join('\n') || 'all predicates hold' }, (which === 'engine' ? 'engine ' : 'all files ') + (L2.ok ? 'ok' : 'fails'));
  }
  function viewBuilds(name, r) {
    if (r.id) return viewBuild(name, r.id);
    const a = arm(name);
    return h('div', null,
      h('h1', null, name + ' — module and pipeline builds'),
      h('p', { class: 'muted' }, '"engine" re-evaluates state_store/builds.py lineage on the snapshot (active event file only, exact checkpoint namespace). "all files" reads every rotated event file and accepts the recorded namespace\'s checkpointed ancestor. Neither reading is a chip-level acceptance verdict.'),
      table([
        { t: 'module', v: b => b.module, r: b => b.module },
        { t: 'build', v: b => b.id, r: b => bLink(name, b.id) },
        { t: 'entry / graph', r: b => b.entry + ' / ' + b.graph },
        { t: 'status', v: b => b.status, r: b => stat(b.status) },
        { t: 'requested', v: b => b.requested_at, r: b => tsCell(b.requested_at) },
        { t: 'duration', v: b => (b.finished_at || 0) - (b.started_at || 0), r: b => b.finished_at ? fmtDur(b.finished_at - (b.started_at || b.requested_at)) : h('span', { class: 'muted' }, 'open') },
        { t: 'node iterations', cls: 'num', v: b => b.segments.length, r: b => String(b.segments.length) },
        { t: 'events exact / inferred', cls: 'num', r: b => b.events_exact.length + ' / ' + (b.events_inferred.length + b.events_llm.length) + (b.events_in_rotated_files ? ' (' + b.events_in_rotated_files + ' rotated)' : '') },
        { t: 'checkpoints', cls: 'num', v: b => b.checkpoints.count, r: b => String(b.checkpoints.count) },
        { t: 'helpers', cls: 'num', v: b => b.helpers.length, r: b => String(b.helpers.length) },
        { t: 'evidence dv/cov/ppa', r: b => b.evidence.dv.length + '/' + b.evidence.coverage.length + '/' + b.evidence.ppa.length },
        { t: 'published at snapshot', v: b => b.published ? 1 : 0, r: b => b.published ? h('span', { class: 'badge s-ok' }, 'published') : (b.superseded ? h('span', { class: 'muted small' }, 'superseded') : '') },
        { t: 'lineage', r: b => h('div', { class: 'chips' }, lineageChip(b, 'engine'), lineageChip(b, 'viewer')) }
      ], a.builds, { sort: 3 }));
  }
  function evidenceTables(name, b) {
    const m = lk(name);
    const dv = b.evidence.dv.map(id => m.ev_dv[id]).filter(Boolean), cov = b.evidence.coverage.map(id => m.ev_coverage[id]).filter(Boolean), ppa = b.evidence.ppa.map(id => m.ev_ppa[id]).filter(Boolean);
    const unc = r => { let u = r.uncovered; if (typeof u === 'string') { try { u = JSON.parse(u); } catch (e) { u = {}; } } return u || {}; };
    return [
      h('h3', null, 'DV rows (dv_results, build_id exact)'),
      table([{ t: 'id', r: r => String(r.id) }, { t: 'time', r: r => tsCell(r.ts) }, { t: 'scope', r: r => r.scope }, { t: 'source', r: r => r.source }, { t: 'attempt', r: r => String(r.attempt) },
        { t: 'passed', r: r => r.passed ? stat('pass') : stat('fail') }, { t: 'tests', r: r => r.tests_passed + '/' + r.tests_total }, { t: 'duration', r: r => fmtDur(r.duration_s) }, { t: 'log', cls: 'wrap mono small', r: r => r.log_path || '' }], dv),
      h('h3', null, 'Coverage rows'),
      table([{ t: 'id', r: r => String(r.id) }, { t: 'time', r: r => tsCell(r.ts) }, { t: 'attempt', r: r => String(r.attempt) }, { t: 'pct', r: r => r.pct === null ? 'not measured' : String(r.pct) },
        { t: 'points', r: r => r.points_hit + '/' + r.points_total }, { t: 'floor', r: r => String(unc(r).floor) }, { t: 'passed', r: r => String(unc(r).passed) }], cov),
      h('h3', null, 'PPA rows'),
      table([{ t: 'id', r: r => String(r.id) }, { t: 'time', r: r => tsCell(r.ts) }, { t: 'attempt', r: r => String(r.attempt) }, { t: 'probe/stage', r: r => r.probe + '/' + (r.stage || '') },
        { t: 'cells', cls: 'num', r: r => num(r.cells) }, { t: 'area µm²', cls: 'num', r: r => num(r.area_um2) }, { t: 'WNS ns', cls: 'num', r: r => r.wns_ns === null ? '—' : String(r.wns_ns) },
        { t: 'ppa_ok', r: r => r.ppa_ok === null ? 'NULL' : String(r.ppa_ok) }, { t: 'pdk / clock', r: r => (r.pdk || '') + ' / ' + (r.clock_mhz || '') }], ppa)];
  }
  function viewBuild(name, id) {
    const a = arm(name), m = lk(name), b = m.builds[id];
    if (!b) return h('p', { class: 'err' }, 'No build ' + id);
    const res = b.result || {};
    const segHost = h('div', null, h('p', { class: 'muted' }, 'loading events…'));
    const ckHost = h('div', null, h('p', { class: 'muted' }, 'loading checkpoints…'));
    shard('data/' + name + '/events.js', 'events:' + name).then(evs => {
      const byId = {}; evs.forEach(e => byId[e.id] = e);
      segHost.textContent = '';
      const exact = new Set(b.events_exact);
      if (!b.segments.length) segHost.appendChild(h('p', { class: 'muted' }, 'No node events.'));
      b.segments.forEach(s => {
        const evRows = s.events.map(i => byId[i]).filter(Boolean);
        segHost.appendChild(h('div', { class: 'seg' },
          h('div', { class: 'when' }, (s.enter_ts || s.exit_ts) ? fmtTs(s.enter_ts || s.exit_ts, true) : '—', h('br'), s.duration_s !== undefined && s.duration_s !== null ? fmtDur(s.duration_s) : ''),
          h('div', null,
            h('div', { class: 'row' }, h('b', null, s.node), s.block ? h('span', { class: 'muted small' }, s.block) : null, s.attempt !== null && s.attempt !== undefined ? h('span', { class: 'small' }, 'attempt ' + s.attempt) : null, stat(s.status || 'open'), s.instant ? h('span', { class: 'flag' }, 'instant') : null,
              (s.helpers || []).map(hid => h('span', null, '⇢ helper ', aLink(name, hid)))),
            h('details', { class: 'small' }, h('summary', null, evRows.length + ' event(s): ' + evRows.map(e => e.event).join(', ')),
              evRows.map(e => h('div', { class: 'panel' }, h('div', { class: 'row' }, badge(exact.has(e.id) ? 'exact' : 'strong', exact.has(e.id) ? 'build_id' : 'block+window'), evLink(name, e.id), h('span', null, e.event), tsCell(e.ts), e.active_file ? null : h('span', { class: 'flag' }, 'rotated file'), (e.copies || []).length ? h('span', { class: 'flag' }, (e.copies.length) + ' copy') : null),
                textBlock(jsonText(e.fields), 3000)))))));
      });
      const llm = b.events_llm.map(i => byId[i]).filter(Boolean);
      if (llm.length) segHost.appendChild(h('details', { class: 'small' }, h('summary', null, llm.length + ' helper lifecycle event(s) (llm_start/llm_end; strong: block in run name + window)'), llm.map(e => h('div', null, evLink(name, e.id), ' ', e.event, ' ', tsCell(e.ts), ' ', (e.fields || {}).run_name || ''))));
    }).catch(e => { segHost.textContent = 'could not load events: ' + e.message; });
    shard('data/' + name + '/checkpoints.js', 'checkpoints:' + name).then(cks => {
      const byId = {}; cks.forEach(c => byId[c.thread + '|' + c.ns + '|' + c.id] = c);
      const list = cks.filter(c => c.thread === b.checkpoints.thread && b.checkpoints.list.includes(c.id));
      ckHost.textContent = '';
      ckHost.appendChild(table([
        { t: 'time', v: c => c.ts, r: c => tsCell(c.ts) }, { t: 'step', cls: 'num', r: c => String(c.step) }, { t: 'source', r: c => c.source_kind || '' },
        { t: 'namespace', cls: 'wrap mono small', r: c => c.ns || '(root)' }, { t: 'nodes that ran (versions_seen diff)', cls: 'wrap', r: c => (c.ran || []).join(', ') },
        { t: 'pending writes (channels)', cls: 'wrap small', r: c => c.writes.length ? c.writes.map(w => w.channel).join(', ') : '' },
        { t: 'park/resume', r: c => (c.interrupt_writes ? 'interrupt ' : '') + (c.resume_writes ? 'resume' : '') },
        { t: 'id', cls: 'mono small', r: c => h('span', { title: srcText(c.src) }, c.id.slice(0, 13) + '…') }
      ], list, { sort: 0, pageSize: 300 }));
    }).catch(e => { ckHost.textContent = 'could not load checkpoints: ' + e.message; });
    const ns = b.completion_ns;
    return h('div', null,
      h('div', { class: 'row' }, h('h1', null, b.module + ' — ' + b.id), stat(b.status), b.published ? h('span', { class: 'badge s-ok', title: 'results.best names this build in the snapshot' }, 'published at snapshot') : null),
      section(null, kv([['entry / graph', b.entry + ' / ' + b.graph], ['thread', h('span', { class: 'mono small' }, b.thread_id)], ['run id', b.run_id],
        ['requested / started / finished', fmtTs(b.requested_at) + ' / ' + fmtTs(b.started_at) + ' / ' + (b.finished_at ? fmtTs(b.finished_at) : 'open')],
        ['duration', b.finished_at ? fmtDur(b.finished_at - (b.started_at || b.requested_at)) : 'open at snapshot'],
        ['attempts seen in events', (b.attempts_seen || []).join(', ') || '—'], ['completion attempt', res.attempt],
        ['error', b.error ? textBlock(b.error, 2000) : null], ['published at snapshot (results.best)', h('span', null, b.published ? 'yes' : (b.superseded ? 'superseded (results.best_superseded)' : 'no'), h('span', { class: 'small muted' }, ' · currency against the live inputs is not re-evaluated here: RTL and build artifacts are not in the snapshot'))],
        ['source', srcRef(b.src)]])),
      section('Lineage readings', h('div', { class: 'grid two' },
        ['engine', 'viewer'].map(w => h('div', { class: b.lineage[w].ok ? 'okbox' : 'badbox' }, h('b', null, w === 'engine' ? 'As the engine reads it (active event file; exact namespace)' : 'Every event file; checkpointed ancestor namespace'),
          b.lineage[w].ok ? h('p', null, 'all predicates hold') : h('ul', null, b.lineage[w].reasons.map(x => h('li', null, x)))))),
        kv([['event window', fmtTs(b.event_window.from) + ' → ' + (b.event_window.to ? fmtTs(b.event_window.to) : 'open') + ' (daemon pid ' + b.event_window.daemon_pid + '; ' + b.event_window.basis + ')'],
          ['events carrying the build id', b.events_exact.length + ' (' + b.events_in_rotated_files + ' only in a rotated file); nodes in the active file: ' + (b.lineage.active_file_nodes.join(', ') || 'none') + '; in all files: ' + (b.lineage.all_file_nodes.join(', ') || 'none')],
          ['dispatch namespace (builds.checkpoint_ns)', h('span', { class: 'mono small' }, b.dispatch_ns || '—')],
          ['completion namespace (result.checkpoint_ns)', ns ? h('span', { class: 'mono small' }, ns.recorded || '—') : '—'],
          ['exact namespace membership', ns ? (ns.exact_member ? 'yes' : 'no') : '—'],
          ['checkpointed ancestor (segment boundaries)', ns ? h('span', { class: 'mono small' }, ns.checkpointed_ancestor === null ? 'none' : (ns.checkpointed_ancestor || '(root)')) : '—'],
          ['checkpointed namespaces of the thread', h('div', null, Object.entries(b.checkpoints.namespaces).map(([k, v]) => h('div', { class: 'mono small' }, (k || '(root)') + ' — ' + v + ' checkpoints')))]])),
      section('Node iterations (graph events, ordered)', segHost),
      section('Checkpoints of this build (' + b.checkpoints.count + ' in the thread' + (b.checkpoints.list.length !== b.checkpoints.count ? '; ' + b.checkpoints.list.length + ' under its namespace' : '') + ')', ckHost),
      section('Helper calls', b.helpers.length ? b.helpers.map(x => h('div', null, badge(x.label), ' ', aLink(name, x.agent), ' ', agentLabel(name, x.agent), h('span', { class: 'small muted' }, ' — ' + x.basis))) : h('span', { class: 'muted' }, 'none')),
      section('CLI calls', b.actions.length ? b.actions.map(x => { const ac = m.actions[x.action] || {}; return h('div', null, badge(x.label), ' ', cliLink(name, x.action), ' ', tsCell(ac.ts), ' ', h('span', { class: 'mono small' }, argvText(ac.argv)), x.dispatch ? h('span', { class: 'flag' }, 'dispatch') : null); }) : h('span', { class: 'muted' }, 'none')),
      section('Parks', b.interrupts.length ? b.interrupts.map(x => parkCard(name, m.parks[x.interrupt], x.label)) : h('span', { class: 'muted' }, 'none')),
      section('Evidence', evidenceTables(name, b)),
      section('Step logs', b.step_logs.length ? b.step_logs.map(sid => stepLogRow(name, m.steps[sid])) : h('span', { class: 'muted' }, 'none attributed')),
      section('Recorded inputs and result', jsonBlock(b.inputs, 'inputs_json'), jsonBlock(res, 'result_json'), jsonBlock(b.worker, 'worker_json'), b.seed ? jsonBlock(b.seed, 'seed_json') : null));
  }
  function stepLogRow(name, s) {
    if (!s) return null;
    const host = h('div');
    const d = h('details', null, h('summary', null, s.block + ' / ' + s.name + ' · ' + num(s.bytes) + ' bytes · ' + (s.iso || 'no header timestamp')), host);
    d.addEventListener('toggle', () => {
      if (!d.open || host.childNodes.length) return;
      shard(s.file, 'step:' + name + ':' + s.id).then(v => { host.appendChild(kv([['command', s.header.command], ['return code', s.header.return_code], ['source', srcRef(s.src)]])); host.appendChild(textBlock(v.text, 6000)); });
    });
    return d;
  }
  function parkCard(name, p, label) {
    if (!p) return null;
    return h('div', { class: 'panel' },
      h('div', { class: 'row' }, label ? badge(label) : null, h('b', null, p.kind), p.block ? h('span', null, p.block) : null, stat(p.status), h('span', { class: 'muted small' }, p.graph + ' / ' + p.node), h('span', { class: 'mono small' }, p.id)),
      kv([['parked', fmtTs(p.ts)], ['resolved', p.resolved_ts ? fmtTs(p.resolved_ts) + ' by ' + (p.resolved_by || '?') : '—'], ['consumed', p.consumed_ts ? fmtTs(p.consumed_ts) : '—'], ['run id', p.run_id],
        ['decision', p.decision ? h('div', null, badge(p.decision.label), ' ', p.decision.action + ' by ' + (p.decision.actor || '?'), h('div', { class: 'small' }, h('span', { class: 'muted' }, 'public rationale (decisions.reasoning, operator-supplied): '), p.decision.reasoning || '')) : '—'],
        ['resume calls', p.actions.length ? p.actions.map(x => h('span', null, badge(x.label), ' ', cliLink(name, x.action), ' ')) : '—'],
        ['build', p.build ? h('span', null, badge(p.build.label), ' ', bLink(name, p.build.build)) : '—'], ['source', srcRef(p.src)]]),
      jsonBlock(p.payload, 'park payload'), p.resolution ? jsonBlock(p.resolution, 'resolution') : null);
  }

  // ------------------------------------------------------------------ graph runs
  function viewGraphs(name, r) {
    const a = arm(name), m = lk(name);
    if (r.id) {
      const g = m.runs[r.id];
      if (!g) return h('p', { class: 'err' }, 'No graph run ' + r.id);
      const evHost = h('div', null, h('p', { class: 'muted' }, 'loading events…'));
      shard('data/' + name + '/events.js', 'events:' + name).then(evs => {
        const ids = new Set(g.events);
        evHost.textContent = '';
        evHost.appendChild(eventTable(name, evs.filter(e => ids.has(e.id))));
      });
      return h('div', null, h('h1', null, name + ' — ' + g.graph + ' thread ' + g.thread),
        section(null, kv([['checkpoints', String(g.checkpoints)], ['span', fmtTs(g.first_ts) + ' → ' + fmtTs(g.last_ts)], ['run ids', g.run_ids.join(', ') || '—'],
          ['namespaces', h('div', null, Object.entries(g.namespaces).map(([k, v]) => h('div', { class: 'mono small' }, (k || '(root)') + ' — ' + v)))],
          ['builds on this thread', g.builds.length ? g.builds.map(x => h('div', null, badge('exact', 'thread'), ' ', bLink(name, x))) : '—'],
          ['parks', g.parks.length ? g.parks.map(x => h('div', null, badge('exact', 'graph'), ' ', x)) : '—'],
          ['helper calls', (g.helpers || []).length ? g.helpers.map(x => h('div', null, badge('strong'), ' ', aLink(name, x), ' ', agentLabel(name, x))) : '—'],
          ['CLI calls', (g.actions || []).length ? g.actions.map(x => h('span', null, badge(x.label), ' ', cliLink(name, x.action), ' ')) : '—'], ['source', srcRef({ s: g.source })]])),
        section('Root-namespace steps (from checkpoints)', table([{ t: 'time', r: c => tsCell(c.ts) }, { t: 'step', cls: 'num', r: c => String(c.step) }, { t: 'source', r: c => c.source_kind },
          { t: 'nodes that ran', cls: 'wrap', r: c => (c.ran || []).join(', ') }, { t: 'park/resume', r: c => (c.interrupt_writes ? 'interrupt ' : '') + (c.resume_writes ? 'resume' : '') }], g.root_steps, { pageSize: 300 })),
        section('Events outside any block in the thread span (' + g.events_basis + ')', evHost),
        g.parks.length ? section('Parks', g.parks.map(id => parkCard(name, m.parks[id]))) : null);
    }
    return h('div', null, h('h1', null, name + ' — pipeline, backend and architecture graph threads'),
      h('p', { class: 'muted' }, 'Module builds run on their own build-graph threads (see Builds). These are the other LangGraph threads with checkpoints.'),
      table([{ t: 'thread', r: g => runLink(name, g.id) }, { t: 'checkpoints', cls: 'num', r: g => String(g.checkpoints) }, { t: 'first', r: g => tsCell(g.first_ts) }, { t: 'last', r: g => tsCell(g.last_ts) },
        { t: 'namespaces', r: g => String(Object.keys(g.namespaces).length) }, { t: 'builds', r: g => g.builds.map(x => bLink(name, x)) }, { t: 'parks', cls: 'num', r: g => String(g.parks.length) }, { t: 'events', cls: 'num', r: g => String(g.events.length) }], a.graph_runs),
      a.graph_runs.length ? null : h('p', { class: 'muted' }, 'No pipeline/backend/architecture checkpoints in this arm.'),
      h('h2', null, 'Checkpoint databases'),
      table([{ t: 'graph', r: g => g[0] }, { t: 'checkpoints', cls: 'num', r: g => num(g[1]) }], Object.entries(a.counts.checkpoints)));
  }

  // ------------------------------------------------------------------ events
  function eventTable(name, evs) {
    return table([
      { t: 'time', v: e => e.ts, r: e => tsCell(e.ts) }, { t: 'event', v: e => e.event, r: e => evLink(name, e.id) }, { t: 'type', v: e => e.event, r: e => e.event },
      { t: 'node', v: e => e.node, r: e => e.node }, { t: 'block', v: e => e.block, r: e => e.block || '' },
      { t: 'build id', r: e => e.build_id ? bLink(name, e.build_id) : '' }, { t: 'pid', cls: 'num', r: e => String(e.pid) },
      { t: 'file', r: e => h('span', { class: 'small' }, e.active_file ? 'active' : 'rotated', (e.copies || []).length ? ' +' + e.copies.length + ' copy' : '') },
      { t: 'fields', cls: 'wrap small mono', r: e => { const t = jsonText(e.fields); return t.length > 240 ? t.slice(0, 240) + ' … (open the event)' : t; } }
    ], evs, { pageSize: 200 });
  }
  function viewEvents(name, r) {
    if (r.tab === 'event') return viewEvent(name, r.id);
    const host = h('div', null, h('p', { class: 'muted' }, 'loading events…'));
    const a = arm(name);
    shard('data/' + name + '/events.js', 'events:' + name).then(evs => {
      const types = [...new Set(evs.map(e => e.event))].sort(), nodes = [...new Set(evs.map(e => e.node))].sort(), blocks = [...new Set(evs.map(e => e.block).filter(Boolean))].sort();
      const st = { q: '', type: '', node: '', block: '', file: '', beats: false };
      const out = h('div');
      function draw() {
        const ql = st.q.toLowerCase();
        const rs = evs.filter(e => (st.beats || e.event !== 'llm_call_heartbeat') && (!st.type || e.event === st.type) && (!st.node || e.node === st.node) && (!st.block || e.block === st.block)
          && (!st.file || (st.file === 'active') === !!e.active_file) && (!ql || (e.id + ' ' + (e.build_id || '') + ' ' + jsonText(e.fields)).toLowerCase().includes(ql)));
        out.textContent = '';
        out.appendChild(eventTable(name, rs));
      }
      const sel = (key, opts) => h('select', { onchange: e => { st[key] = e.target.value; draw(); } }, [['', 'any']].concat(opts.map(o => [o, o])).map(([v, t]) => h('option', { value: v }, t)));
      host.textContent = '';
      add(host, [h('div', { class: 'filters' }, h('input', { type: 'search', placeholder: 'filter fields, build id…', oninput: e => { st.q = e.target.value; draw(); } }),
        h('label', null, 'type ', sel('type', types)), h('label', null, 'node ', sel('node', nodes)), h('label', null, 'block ', sel('block', blocks)),
        h('label', null, 'file ', h('select', { onchange: e => { st.file = e.target.value; draw(); } }, h('option', { value: '' }, 'any'), h('option', { value: 'active' }, 'active'), h('option', { value: 'rotated' }, 'rotated'))),
        h('label', null, h('input', { type: 'checkbox', onchange: e => { st.beats = e.target.checked; draw(); } }), 'show helper heartbeats')), out]);
      draw();
    }).catch(e => { host.textContent = 'could not load: ' + e.message; });
    return h('div', null, h('h1', null, name + ' — graph events'),
      h('p', { class: 'muted' }, 'Every line of every pipeline_events*.jsonl (the rotated logs included). An identical record found in a later file is shown once with its copies; identical lines inside one file are kept as separate events.'),
      table([{ t: 'file', cls: 'mono small', r: f => f.path }, { t: 'state', r: f => f.active ? 'active' : 'rotated at run start' }, { t: 'events', cls: 'num', r: f => num(f.events) }, { t: 'first', r: f => tsCell(f.first_ts) }, { t: 'last', r: f => tsCell(f.last_ts) }], a.event_files),
      h('p', { class: 'small muted' }, 'Daemon processes (writer pid of graph events): ', a.epochs.map(e => 'epoch ' + e.epoch + ' pid ' + e.pid + ' ' + fmtTs(e.first_ts, true) + '–' + fmtTs(e.last_ts, true)).join(' · ')),
      host);
  }
  function viewEvent(name, id) {
    const host = h('div', null, 'loading…');
    shard('data/' + name + '/events.js', 'events:' + name).then(evs => {
      const e = evs.find(x => x.id === id);
      host.textContent = '';
      if (!e) { host.textContent = 'No event ' + id; return; }
      const b = e.build_id ? e.build_id : null;
      const inBuilds = arm(name).builds.filter(x => x.events_exact.includes(id) || x.events_inferred.includes(id) || x.events_llm.includes(id));
      add(host, [h('h1', null, name + ' — event ' + id), section(null, kv([['time', fmtTs(e.ts)], ['event', e.event], ['node', e.node], ['block', e.block], ['writer pid', String(e.pid)],
        ['build id field', b ? bLink(name, b) : '—'], ['attributed to builds', inBuilds.length ? inBuilds.map(x => h('span', null, badge(x.events_exact.includes(id) ? 'exact' : 'strong'), ' ', bLink(name, x.id), ' ')) : '—'],
        ['source', srcRef(e.src)], ['copies', (e.copies || []).length ? e.copies.map(c => h('div', null, srcRef(c))) : '—'], ['file', e.active_file ? 'active log' : 'rotated log']])),
      section('Fields', textBlock(jsonText(e.fields), 20000))]);
    });
    return host;
  }

  // ------------------------------------------------------------------ trajectories
  function viewAgents(name, r) {
    if (r.tab === 'agent') return viewAgent(name, r);
    const a = arm(name), m = lk(name);
    const roots = a.agents.filter(x => !x.parent || !x.parent.agent || !m.agents[x.parent.agent]);
    const node = x => h('li', null, h('div', { class: 'row small' }, h('span', { class: 'badge k-' + (x.kind === 'architect' ? 'user' : x.kind === 'engine-helper' ? 'tool_use' : 'assistant') }, x.kind), aLink(name, x.id), x.label,
      x.parent ? badge(x.parent.label, 'parent ' + x.parent.label) : null, h('span', { class: 'muted' }, num(x.stats.turns) + ' turns')),
      x.children.length ? h('ul', null, x.children.map(c => m.agents[c]).filter(Boolean).map(node)) : null);
    return h('div', null, h('h1', null, name + ' — trajectories'),
      h('p', { class: 'muted' }, 'The Architect, its native sub-agents and every engine helper call, with provider, models observed in the records, session ids and parent links. Private reasoning is shown only as a length marker.'),
      section('Lineage', h('div', { class: 'tree' }, h('ul', null, roots.filter(x => x.kind !== 'engine-helper').map(node)), h('p', { class: 'small muted' }, 'Engine helpers are spawned by the daemon, not by the Architect: see the table below and the Helper calls view.'))),
      table([
        { t: 'agent', v: x => x.id, r: x => aLink(name, x.id) },
        { t: 'kind', v: x => x.kind, r: x => x.kind },
        { t: 'label', cls: 'wrap', r: x => x.label },
        { t: 'provider / models', r: x => x.provider + ' · ' + (Object.keys(x.models || {}).join(', ') || (x.model_requested ? 'requested ' + x.model_requested : '—')) },
        { t: 'session', cls: 'mono small', r: x => x.session_id || '—' },
        { t: 'parent', r: x => x.parent && x.parent.agent ? h('span', null, badge(x.parent.label), ' ', aLink(name, x.parent.agent)) : '—' },
        { t: 'first', v: x => x.stats.first_ts, r: x => tsCell(x.stats.first_ts) },
        { t: 'last', v: x => x.stats.last_ts, r: x => tsCell(x.stats.last_ts) },
        { t: 'turns', cls: 'num', v: x => x.stats.turns, r: x => num(x.stats.turns) },
        { t: 'shell calls', cls: 'num', v: x => x.shell_calls, r: x => num(x.shell_calls) },
        { t: 'reasoning markers', cls: 'num', v: x => x.stats.reasoning_markers, r: x => num(x.stats.reasoning_markers) }
      ], a.agents, { pageSize: 300 }));
  }
  function turnPageFor(ag, seq) { return ag.pages.findIndex(p => seq >= p.from && seq <= p.to); }
  function loadTurns(name, ag, seqs) {
    const pages = [...new Set(seqs.map(s => turnPageFor(ag, s)).filter(i => i >= 0))];
    return Promise.all(pages.map(i => shard(ag.pages[i].file, 'turns:' + name + ':' + ag.id + ':' + i))).then(lists => {
      const out = {}; lists.forEach(l => l.forEach(t => out[t.seq] = t)); return out;
    });
  }
  const KINDS = ['user', 'assistant', 'tool_use', 'tool_result', 'thinking', 'event', 'attachment', 'system'];
  function renderTurn(name, ag, t, byTid) {
    const m = lk(name);
    const sc = m.shellBySeq[ag.id + ':' + t.seq];
    const head = h('div', { class: 'th' },
      link('#/a/' + enc(name) + '/agent/' + enc(ag.id) + '?t=' + t.seq, '#' + t.seq), t.ts ? h('span', { title: t.iso || '' }, fmtTs(t.ts, true)) : h('span', { class: 'muted' }, 'no time'),
      kindBadge(t.kind), t.name ? h('b', { class: 'small' }, t.name) : null,
      (t.flags || []).map(f => h('span', { class: 'flag' }, f.replace(/_/g, ' '))),
      t.err ? h('span', { class: 'badge s-error' }, 'error') : null, t.exit !== undefined && t.exit !== null ? h('span', { class: 'small' }, 'exit ' + t.exit) : null,
      t.tid && byTid && byTid[t.tid] && byTid[t.tid] !== t ? link('#/a/' + enc(name) + '/agent/' + enc(ag.id) + '?t=' + byTid[t.tid].seq, t.kind === 'tool_use' ? '→ result #' + byTid[t.tid].seq : '← call #' + byTid[t.tid].seq) : null,
      t.child ? h('span', null, 'spawned ', aLink(name, t.child)) : null,
      sc ? h('span', { class: 'chips' }, (sc.audit || []).map(x => h('span', null, badge(x.label, 'audit'), ' ', cliLink(name, x.action))), (sc.missing || []).length ? h('span', { class: 'badge b-missing' }, sc.missing.length + ' not audited') : null, callLink(name, sc.id)) : null,
      h('span', { class: 'spacer' }), h('span', null, refsList(t.refs)));
    const metaBits = [];
    const mt = t.meta || {};
    ['description', 'run_in_background', 'subagent_type', 'model', 'to', 'origin', 'status', 'stop_reason', 'error', 'agentId', 'resolvedModel', 'backgroundTaskId', 'interrupted', 'duration_s', 'cwd', 'author', 'recipient'].forEach(k => {
      if (mt[k] !== undefined && mt[k] !== null && mt[k] !== '' && mt[k] !== false) metaBits.push(h('span', { class: 'small muted' }, k + ': ' + (typeof mt[k] === 'object' ? jsonText(mt[k]) : String(mt[k]))));
    });
    if (mt.usage) metaBits.push(h('span', { class: 'small muted' }, 'usage: ' + jsonText(mt.usage).replace(/\s+/g, ' ')));
    return h('div', { class: 'turn k-' + t.kind + '-b', id: 'turn-' + t.seq }, head,
      h('div', { class: 'tb' }, metaBits.length ? h('div', { class: 'row' }, metaBits) : null, textBlock(t.text, t.kind === 'tool_result' ? 2500 : 5000)));
  }
  function viewAgent(name, r) {
    const a = arm(name), m = lk(name), ag = m.agents[r.id];
    if (!ag) return h('p', { class: 'err' }, 'No agent ' + r.id);
    const focus = Number(r.q.get('t') || r.sub || 0) || null;
    const st = { page: focus ? Math.max(0, turnPageFor(ag, focus)) : 0, kinds: new Set(KINDS), noise: true, q: '', all: null };
    const listHost = h('div');
    const info = h('div', { class: 'small muted' });
    function filt(ts) {
      const ql = st.q.toLowerCase();
      return ts.filter(t => st.kinds.has(t.kind) && (!st.noise || !(t.flags || []).includes('noise') || t.seq === focus) && (!ql || (t.text || '').toLowerCase().includes(ql) || (t.name || '').toLowerCase().includes(ql)));
    }
    function draw(turns, label) {
      const uses = {}, results = {};
      turns.forEach(t => { if (t.tid && t.kind === 'tool_use' && !uses[t.tid]) uses[t.tid] = t; if (t.tid && t.kind === 'tool_result' && !results[t.tid]) results[t.tid] = t; });
      const pairOf = {};
      Object.keys(uses).forEach(k => { if (results[k]) { pairOf[uses[k].seq] = results[k]; pairOf[results[k].seq] = uses[k]; } });
      const shown = filt(turns);
      listHost.textContent = '';
      info.textContent = label + ' · ' + num(shown.length) + ' of ' + num(turns.length) + ' turns shown';
      const MAX = 400;
      shown.slice(0, MAX).forEach(t => listHost.appendChild(renderTurn(name, ag, t, { [t.tid]: pairOf[t.seq] || null })));
      if (shown.length > MAX) {
        const more = h('button', { onclick: () => { more.remove(); shown.slice(MAX).forEach(t => listHost.appendChild(renderTurn(name, ag, t, { [t.tid]: pairOf[t.seq] || null }))); } }, 'Render the remaining ' + num(shown.length - MAX) + ' matching turns');
        listHost.appendChild(more);
      }
      if (focus) { const el = document.getElementById('turn-' + focus); if (el) { el.classList.add('hl'); setTimeout(() => el.scrollIntoView({ block: 'center' }), 30); } }
    }
    function loadPage(i) {
      st.page = i;
      listHost.textContent = 'loading page ' + (i + 1) + '…';
      return shard(ag.pages[i].file, 'turns:' + name + ':' + ag.id + ':' + i).then(ts => { pagerSel.value = String(i); draw(ts, 'page ' + (i + 1) + ' of ' + ag.pages.length + ' (turns ' + ag.pages[i].from + '–' + ag.pages[i].to + ')'); });
    }
    function loadAll() {
      listHost.textContent = 'loading all ' + ag.pages.length + ' pages…';
      return Promise.all(ag.pages.map((p, i) => shard(p.file, 'turns:' + name + ':' + ag.id + ':' + i))).then(ls => { st.all = [].concat(...ls); draw(st.all, 'all pages'); });
    }
    const pagerSel = h('select', { onchange: e => loadPage(Number(e.target.value)) }, ag.pages.map((p, i) => h('option', { value: String(i) }, 'page ' + (i + 1) + ' · #' + p.from + '–' + p.to + (p.first_ts ? ' · ' + fmtTs(p.first_ts, true) : ''))));
    const qbox = h('input', { type: 'search', placeholder: 'search this trajectory (loads every page)…', onchange: e => { st.q = e.target.value; (st.all ? Promise.resolve(draw(st.all, 'all pages')) : loadAll()); } });
    const kindBoxes = KINDS.map(k => h('label', null, h('input', { type: 'checkbox', checked: true, onchange: e => { e.target.checked ? st.kinds.add(k) : st.kinds.delete(k); redraw(); } }), k.replace('_', ' ')));
    function redraw() { if (st.all) draw(st.all, 'all pages'); else loadPage(st.page); }
    const ec = ag.engine_call;
    const header = h('div', null,
      h('div', { class: 'row' }, h('h1', null, name + ' — ' + ag.label)),
      section(null, kv([['agent id', ag.id], ['kind', ag.kind], ['provider', ag.provider], ['requested model', ag.model_requested], ['models in records', Object.entries(ag.models || {}).map(([k, v]) => k + ' ×' + v).join(', ') || '—'],
        ['session / thread', h('span', { class: 'mono small' }, ag.session_id || '—')], ['parent', ag.parent && ag.parent.agent ? h('span', null, badge(ag.parent.label), ' ', aLink(name, ag.parent.agent, null), ' ', h('span', { class: 'small muted' }, ag.parent.basis || '')) : '—'],
        ['children', ag.children.length ? ag.children.map(c => h('span', null, aLink(name, c), ' ')) : '—'],
        ['span', fmtTs(ag.stats.first_ts) + ' → ' + fmtTs(ag.stats.last_ts)], ['turns', num(ag.stats.turns) + ' (' + Object.entries(ag.stats.by_kind).map(([k, v]) => k + ' ' + v).join(', ') + ')'],
        ['tools', Object.entries(ag.stats.tools || {}).map(([k, v]) => k + ' ×' + v).join(', ') || '—'],
        ['copies referenced', num(ag.stats.duplicates_referenced) + ' duplicate record copies folded into turns'],
        ['usage', usageNode(ag)], ['sources', h('div', null, ag.sources.map(s => h('div', null, srcRef({ s }))))], ['notes', ag.notes.length ? h('ul', null, ag.notes.map(n => h('li', null, n))) : null]])),
      ec ? section('Engine call of record', kv([['kind', ec.kind === 'unfinished' ? 'unfinished at the snapshot (no llm_calls record)' : 'finished (llm_calls.jsonl)'],
        ['run name', ec.run_name], ['node / block', (ec.node || '') + ' / ' + (ec.block || '')], ['graph', ec.graph], ['daemon epoch · call_index', ec.epoch + ' · ' + ec.call_index],
        ['start → end', fmtTs(ec.start_ts) + ' → ' + (ec.ts ? fmtTs(ec.ts) : 'open') + (ec.duration_s ? ' (' + fmtDur(ec.duration_s) + ')' : '')],
        ['timeout / timed out', ec.timeout + ' / ' + ec.timed_out], ['error', ec.error || '—'], ['child pid · heartbeats', ec.child_pid + ' · ' + ec.heartbeats],
        ['joins', h('div', null, (ec.joins || []).map(j => h('div', null, badge(j.label), ' ', j.to, ' ', h('span', { class: 'mono small' }, j.id), h('span', { class: 'small muted' }, ' — ' + j.basis))))],
        ['build', (ec.builds || []).length ? ec.builds.map(b => h('div', null, badge(b.label), ' ', bLink(name, b.build), h('span', { class: 'small muted' }, ' ' + (b.basis || '')))) : ((ec.graph_runs || []).length ? ec.graph_runs.map(g => h('div', null, badge(g.label), ' ', runLink(name, g.run))) : '—')],
        ['usage (this call)', ec.usage ? textBlock(jsonText(ec.usage), 1500) : 'unknown'], ['source', srcRef(ec.src)]])) : null,
      (ag.tasks || []).length ? h('details', { class: 'panel' }, h('summary', null, ag.tasks.length + ' background task(s)'), textBlock(jsonText(ag.tasks), 4000)) : null);
    const controls = h('div', { class: 'filters' }, h('label', null, 'page ', pagerSel), h('button', { onclick: () => loadAll() }, 'Load all pages'), qbox,
      h('label', null, h('input', { type: 'checkbox', checked: true, onchange: e => { st.noise = e.target.checked; redraw(); } }), 'hide noise records'), kindBoxes, info);
    loadPage(st.page);
    return h('div', null, header, controls, listHost);
  }
  function usageNode(ag) {
    const u = ag.usage || {};
    const out = [];
    if (u.latest_cumulative_snapshot) { const s = u.latest_cumulative_snapshot; out.push(h('div', null, 'latest cumulative provider snapshot: ', h('b', null, money(s.totalCostUSD ?? s.total_cost_usd)), ' ', h('span', { class: 'small muted' }, (s.scope || '') + ' · ' + u.snapshots + ' snapshot(s) · '), srcRef(s.src))); }
    if (u.claude_sum_unique_messages) out.push(h('div', { class: 'small' }, 'tokens over ' + num(u.claude_sum_unique_messages.messages) + ' unique message ids: ' + Object.entries(u.claude_sum_unique_messages.usage).map(([k, v]) => k + ' ' + num(v)).join(', ')));
    if (u.codex_thread_total_latest) out.push(h('div', { class: 'small' }, 'latest Codex thread total: ' + Object.entries(u.codex_thread_total_latest.total_token_usage || {}).map(([k, v]) => k + ' ' + num(v)).join(', ') + ' (cumulative; never summed)'));
    if (u.codex_sum_unique_responses) out.push(h('div', { class: 'small' }, 'sum over ' + num(u.codex_sum_unique_responses.responses) + ' unique responses: ' + Object.entries(u.codex_sum_unique_responses.usage).map(([k, v]) => k + ' ' + num(v)).join(', ')));
    return out.length ? out : h('span', { class: 'muted' }, 'unavailable');
  }

  // ------------------------------------------------------------------ helpers
  function viewHelpers(name) {
    const a = arm(name);
    return h('div', null, h('h1', null, name + ' — engine helper calls'),
      h('p', { class: 'muted' }, 'One row per llm_calls.jsonl record (and per llm_call_start without a record: unfinished). call_index restarts with every daemon process, so a call is identified by daemon epoch and call_index together.'),
      table([
        { t: 'helper', v: x => x.agent, r: x => aLink(name, x.agent) },
        { t: 'epoch · idx', v: x => (x.epoch || 0) * 1000 + (x.call_index || 0), r: x => x.epoch + ' · ' + x.call_index },
        { t: 'run name', cls: 'wrap', v: x => x.run_name, r: x => x.run_name || h('span', { class: 'muted' }, 'unknown') },
        { t: 'graph', v: x => x.graph, r: x => x.graph || '' },
        { t: 'start', v: x => x.start_ts, r: x => tsCell(x.start_ts) },
        { t: 'duration', cls: 'num', v: x => x.duration_s, r: x => x.duration_s ? fmtDur(x.duration_s) : (x.kind === 'unfinished' ? 'open' : '') },
        { t: 'model', r: x => x.model || '' },
        { t: 'cost', cls: 'num', v: x => (x.usage || {}).total_cost_usd, r: x => typeof (x.usage || {}).total_cost_usd === 'number' ? money(x.usage.total_cost_usd) : 'unknown' },
        { t: 'tokens in/out', cls: 'num', r: x => { const u = x.usage || {}; return (u.input_tokens !== undefined ? num(u.input_tokens + (u.cache_read_input_tokens || 0) + (u.cache_creation_input_tokens || 0)) : '?') + ' / ' + (u.output_tokens !== undefined ? num(u.output_tokens) : '?'); } },
        { t: 'error', cls: 'wrap small', r: x => (x.timed_out ? 'timed out; ' : '') + (x.error || '') },
        { t: 'build / thread', r: x => (x.builds || []).length ? x.builds.map(b => h('span', null, badge(b.label), ' ', bLink(name, b.build))) : (x.graph_runs || []).map(g => h('span', null, badge(g.label), ' ', runLink(name, g.run))) },
        { t: 'joins', r: x => h('div', { class: 'chips' }, (x.joins || []).map(j => badge(j.label, j.to))) }
      ], a.helper_calls, { sort: 4, pageSize: 300 }));
  }

  // ------------------------------------------------------------------ parks & state
  function viewState(name) {
    const a = arm(name);
    const host = h('div', null, h('p', { class: 'muted' }, 'loading project tables…'));
    shard('data/' + name + '/project.js', 'project:' + name).then(p => {
      host.textContent = '';
      add(host, [
        h('h2', null, 'Results (published passes)'),
        table([{ t: 'block', v: r => r.block, r: r => r.block }, { t: 'kind', v: r => r.kind, r: r => r.kind }, { t: 'build', r: r => (r.value || {}).build_id ? bLink(name, r.value.build_id) : '' }, { t: 'time', r: r => tsCell(r.ts) }, { t: 'value', r: r => jsonBlock(r.value, 'value') }], a.results, { sort: 0 }),
        h('h2', null, 'Requirement items and checks'),
        table([{ t: 'item', v: r => r.id, r: r => r.id }, { t: 'kind', r: r => r.kind }, { t: 'status', v: r => r.status, r: r => stat(r.status) }, { t: 'priority', r: r => r.priority || '' }, { t: 'text', cls: 'wrap', r: r => textBlock(r.text, 300) }], p.items || []),
        table([{ t: 'check', r: r => String(r.id) }, { t: 'item', r: r => r.item_id }, { t: 'kind', r: r => r.kind }, { t: 'status', r: r => stat(r.status) }, { t: 'actor', r: r => r.actor || '' }, { t: 'time', r: r => tsCell(r.ts) }, { t: 'evidence', cls: 'wrap', r: r => textBlock(r.evidence || '', 300) }], p.checks || []),
        h('h2', null, 'Other project tables'),
        Object.entries(p).filter(([k, v]) => Array.isArray(v) && v.length && !['items', 'checks', 'interrupts', 'decisions', 'stages', 'results', 'dv_results', 'coverage_results', 'ppa_history'].includes(k)).map(([k, v]) => h('details', { class: 'panel' }, h('summary', null, k + ' (' + v.length + ' rows)'), textBlock(jsonText(v), 6000)))]);
    }).catch(e => { host.textContent = 'could not load: ' + e.message; });
    return h('div', null, h('h1', null, name + ' — parks, decisions and project state'),
      section('Stages', table([{ t: 'stage', r: s => s.name }, { t: 'status', r: s => stat(s.status) }, { t: 'entered', r: s => tsCell(s.entered_ts) }, { t: 'done', r: s => tsCell(s.done_ts) }, { t: 'blocked by', cls: 'wrap small', r: s => (s.blocked_by || []).length ? jsonText(s.blocked_by) : '' }], a.stages)),
      h('h2', null, 'Parks (interrupts) with decisions and resumes'),
      a.parks.length ? a.parks.map(p => parkCard(name, p)) : h('p', { class: 'muted' }, 'No parks recorded.'),
      h('h2', null, 'All evidence rows'),
      evidenceTables(name, { evidence: { dv: a.evidence.dv.map(r => r.id), coverage: a.evidence.coverage.map(r => r.id), ppa: a.evidence.ppa.map(r => r.id) } }),
      host);
  }

  // ------------------------------------------------------------------ provenance
  function provenance() {
    const sn = IDX.snapshot, man = sn.manifest || {}, pv = IDX.privacy;
    const arms = Object.values(IDX.arms);
    return h('div', null, h('h1', null, 'Provenance and coverage'),
      section('Snapshot', kv([['path (private)', h('span', { class: 'mono small' }, sn.path)], ['collection', (man.started_at || '?') + ' → ' + (man.finished_at || '?')], ['READY.json', sn.ready ? textBlock(jsonText(sn.ready), 2000) : h('span', { class: 'err' }, 'absent: the collector had not marked the snapshot complete')],
        ['consistency', man.consistency], ['scope', man.scope], ['collector errors', (man.errors || []).length ? textBlock(jsonText(man.errors)) : 'none'], ['files in manifest', num(sn.manifest_files)],
        ['manifest files not read by the exporter', (sn.unread_manifest_files || []).length ? h('details', null, h('summary', null, sn.unread_manifest_files.length + ' file(s)'), sn.unread_manifest_files.map(f => h('div', { class: 'mono small' }, f.path + ' (' + f.mode + ', ' + num(f.bytes) + ' B)'))) : 'none']])),
      section('Relationship labels',
        h('ul', null,
          h('li', null, badge('exact'), ' an equal stable identifier: build_id field, thread id, interrupt id, provider session/thread id, call_index within one daemon process, a build id string inside an audited argv, a Claude record uuid.'),
          h('li', null, badge('strong'), ' a unique candidate on content and time: argv tokens + the audit time inside a shell call window; a module named in a helper run name and the call inside that module\'s only build window; a block-scoped event of the same daemon process between a build\'s Init Block and Block Done events.'),
          h('li', null, badge('weak'), ' time only, or the closest of several compatible candidates (alternatives listed).'),
          h('li', null, badge('ambiguous'), ' several equally plausible candidates; all are listed and none is chosen.'),
          h('li', null, badge('unlinked'), ' no candidate. A command string quoted in output, a prompt or a document is never treated as an execution.')),
        table([{ t: 'arm', r: a => a.arm }, { t: 'audit rows', cls: 'num', r: a => num(a.counts.actions) }, { t: 'strong', cls: 'num', r: a => num(a.cli_join.strong) }, { t: 'weak', cls: 'num', r: a => num(a.cli_join.weak) }, { t: 'ambiguous', cls: 'num', r: a => num(a.cli_join.ambiguous) }, { t: 'unlinked', cls: 'num', r: a => num(a.cli_join.unlinked) },
          { t: 'observed, not audited', cls: 'wrap', r: a => Object.entries(a.missing_by_code).map(([k, v]) => k + ' ' + v).join(', ') }], arms)),
      section('Deduplication', table([{ t: 'arm', r: a => a.arm }, { t: 'event copies folded', cls: 'num', r: a => num(a.counts.event_copies_deduplicated) }, { t: 'identical lines kept within one file', cls: 'num', r: a => num(a.counts.within_file_repeats_kept) },
        { t: 'trajectory record copies folded', cls: 'num', r: a => num(a.agents.reduce((s, x) => s + (x.stats.duplicates_referenced || 0), 0)) }], arms),
        h('p', { class: 'small muted' }, 'Claude records fold by uuid across the native session, the invocation stream and engine live-stream captures; Codex items by item id and, for exec --json copies, by output digest in order; graph events by identical whole record across files. Cumulative usage snapshots are never summed.')),
      section('Privacy', pv.ok ? h('div', { class: 'okbox' }, 'No fingerprint of any withheld payload (' + num(pv.fingerprints_checked) + ' fingerprints) and no encrypted-payload pattern was found in the ' + num(pv.files_scanned) + ' exported data files.') : h('div', { class: 'badbox' }, 'Privacy check failed: ' + jsonText(pv.leaks)),
        h('p', { class: 'small' }, 'Withheld (counts and characters only): '), textBlock(jsonText(pv.withheld), 3000)),
      h('h2', null, 'Sources'),
      table([{ t: 'id', v: s => Number(s.id.slice(1)), r: s => s.id }, { t: 'arm', v: s => s.arm, r: s => s.arm || '' }, { t: 'path (private snapshot)', cls: 'wrap mono small', v: s => s.path, r: s => s.path }, { t: 'role', cls: 'wrap small', r: s => s.role },
        { t: 'bytes', cls: 'num', v: s => s.bytes, r: s => num(s.bytes) }, { t: 'copy', r: s => s.manifest ? h('span', { class: 'small' }, s.manifest.mode + (s.manifest.changed_during_copy ? ' · changed during copy' : '') + (s.manifest.partial_final_line ? ' · partial final line' : '')) : h('span', { class: 'muted small' }, 'not in manifest') },
        { t: 'sha matches manifest', r: s => s.manifest_sha_matches === null ? '—' : (s.manifest_sha_matches ? 'yes' : h('span', { class: 'err' }, 'no')) },
        { t: 'records', cls: 'num', v: s => s.records, r: s => num(s.records) }, { t: 'kept', cls: 'num', r: s => num(s.kept) }, { t: 'copies', cls: 'num', r: s => num(s.duplicates) }, { t: 'unparseable', cls: 'num', r: s => num(s.unparseable) },
        { t: 'not exported (why)', cls: 'wrap small', r: s => Object.entries(s.skipped || {}).map(([k, v]) => k + ' ' + v).join('; ') }], IDX.sources, { pageSize: 400 }));
  }

  // ------------------------------------------------------------------ search
  function search(r) {
    const q = (r.q.get('q') || '').trim();
    const ql = q.toLowerCase();
    const out = h('div', null, h('h1', null, 'Search: ' + q));
    if (!q) return out;
    for (const a of Object.values(IDX.arms)) {
      const name = a.arm;
      const acts = a.actions.filter(x => (argvText(x.argv) + ' ' + (x.summary || '') + ' #' + x.id).toLowerCase().includes(ql));
      const builds = a.builds.filter(b => (b.id + ' ' + b.module + ' ' + (b.error || '') + ' ' + b.status).toLowerCase().includes(ql));
      const agents = a.agents.filter(x => (x.id + ' ' + x.label + ' ' + (x.session_id || '')).toLowerCase().includes(ql));
      const parks = a.parks.filter(p => (p.id + ' ' + p.kind + ' ' + (p.block || '') + ' ' + jsonText(p.resolution || '')).toLowerCase().includes(ql));
      add(out, [h('h2', null, name),
        acts.length ? section('CLI audit rows (' + acts.length + ')', acts.slice(0, 200).map(x => h('div', { class: 'small' }, cliLink(name, x.id), ' ', tsCell(x.ts), ' ', h('span', { class: 'mono' }, argvText(x.argv)), ' rc ' + x.rc))) : null,
        builds.length ? section('Builds (' + builds.length + ')', builds.map(b => h('div', null, bLink(name, b.id), ' ', stat(b.status)))) : null,
        agents.length ? section('Trajectories (' + agents.length + ')', agents.map(x => h('div', null, aLink(name, x.id), ' ', x.label))) : null,
        parks.length ? section('Parks (' + parks.length + ')', parks.map(p => h('div', null, p.id, ' ', p.kind, ' ', stat(p.status)))) : null]);
      const evHost = h('div', { class: 'small muted' }, 'searching graph events…');
      out.appendChild(evHost);
      shard('data/' + name + '/events.js', 'events:' + name).then(evs => {
        const hits = evs.filter(e => (e.id + ' ' + e.event + ' ' + e.node + ' ' + (e.block || '') + ' ' + (e.build_id || '') + ' ' + jsonText(e.fields)).toLowerCase().includes(ql));
        evHost.textContent = '';
        evHost.className = '';
        if (hits.length) evHost.appendChild(section('Graph events (' + hits.length + ')', hits.slice(0, 200).map(e => h('div', { class: 'small' }, evLink(name, e.id), ' ', tsCell(e.ts), ' ', e.event, ' ', e.node, ' ', e.block || ''))));
      });
    }
    const deep = h('div', { class: 'panel' }, h('p', null, 'Transcript text is loaded on demand. '),
      h('button', { class: 'primary', onclick: () => deepSearch(ql, deep) }, 'Search every trajectory turn'));
    out.appendChild(deep);
    return out;
  }
  function deepSearch(ql, host) {
    host.textContent = '';
    const prog = h('p', { class: 'muted' }, 'loading…');
    const res = h('div');
    add(host, [prog, res]);
    const jobs = [];
    for (const a of Object.values(IDX.arms)) for (const ag of a.agents) ag.pages.forEach((p, i) => jobs.push([a.arm, ag, p, i]));
    let done = 0, hits = 0;
    const MAXH = 1000;
    function next() {
      if (!jobs.length) { prog.textContent = 'searched ' + done + ' pages · ' + hits + ' matching turns' + (hits > MAXH ? ' (first ' + MAXH + ' shown)' : ''); return; }
      const [name, ag, p, i] = jobs.shift();
      shard(p.file, 'turns:' + name + ':' + ag.id + ':' + i).then(ts => {
        done++;
        prog.textContent = 'searched ' + done + ' pages · ' + hits + ' matches…';
        ts.forEach(t => {
          const tx = (t.text || '');
          const at = tx.toLowerCase().indexOf(ql);
          if (at < 0) return;
          hits++;
          if (hits > MAXH) return;
          res.appendChild(h('div', { class: 'small hit' }, link('#/a/' + enc(name) + '/agent/' + enc(ag.id) + '?t=' + t.seq, name + ' · ' + ag.id + ' #' + t.seq), ' ', kindBadge(t.kind), ' ', tsCell(t.ts), ' … ',
            tx.slice(Math.max(0, at - 80), at), h('mark', null, tx.slice(at, at + ql.length)), tx.slice(at + ql.length, at + ql.length + 120), ' …'));
        });
        next();
      }).catch(() => { done++; next(); });
    }
    next();
  }

  // ------------------------------------------------------------------ analysis (safe markdown subset)
  function inline(text) {
    const out = [];
    const re = /(`[^`]+`)|(\*\*[^*]+\*\*)|(\[[^\]]+\]\((#[^)\s]*)\))/g;
    let pos = 0, m;
    while ((m = re.exec(text))) {
      if (m.index > pos) out.push(text.slice(pos, m.index));
      if (m[1]) out.push(h('code', null, m[1].slice(1, -1)));
      else if (m[2]) out.push(h('b', null, inline(m[2].slice(2, -2))));
      else if (m[3]) { const label = m[3].slice(1, m[3].indexOf('](')); out.push(link(m[4], label)); }
      pos = re.lastIndex;
    }
    if (pos < text.length) out.push(text.slice(pos));
    return out;
  }
  function markdown(md) {
    const root = h('div', { class: 'md panel' });
    const lines = md.split('\n');
    let i = 0;
    while (i < lines.length) {
      const ln = lines[i];
      if (/^```/.test(ln)) { const buf = []; i++; while (i < lines.length && !/^```/.test(lines[i])) buf.push(lines[i++]); i++; root.appendChild(h('pre', null, buf.join('\n'))); continue; }
      const hm = /^(#{1,4})\s+(.*)$/.exec(ln);
      if (hm) { root.appendChild(h('h' + Math.min(4, hm[1].length), null, inline(hm[2]))); i++; continue; }
      if (/^\s*\|/.test(ln)) {
        const rows = [];
        while (i < lines.length && /^\s*\|/.test(lines[i])) { rows.push(lines[i].trim().replace(/^\||\|$/g, '').split('|').map(c => c.trim())); i++; }
        const body = rows.filter((r2, k) => !(k === 1 && r2.every(c => /^:?-+:?$/.test(c))));
        root.appendChild(h('table', null, h('thead', null, h('tr', null, body[0].map(c => h('th', null, inline(c))))), h('tbody', null, body.slice(1).map(r2 => h('tr', null, r2.map(c => h('td', null, inline(c))))))));
        continue;
      }
      if (/^\s*([-*]|\d+\.)\s+/.test(ln)) {
        // one level of nesting: an item indented by two or more spaces
        // belongs to the list of the previous top-level item
        const ordered = /^\s*\d+\./.test(ln);
        const list = h(ordered ? 'ol' : 'ul');
        let lastLi = null, sub = null;
        while (i < lines.length && /^\s*([-*]|\d+\.)\s+/.test(lines[i])) {
          const nested = /^\s{2,}/.test(lines[i]);
          let item = lines[i].replace(/^\s*([-*]|\d+\.)\s+/, ''); i++;
          while (i < lines.length && /^\s{2,}\S/.test(lines[i]) && !/^\s*([-*]|\d+\.)\s+/.test(lines[i])) item += ' ' + lines[i++].trim();
          const li = h('li', null, inline(item));
          if (nested && lastLi) { if (!sub) { sub = h('ul'); lastLi.appendChild(sub); } sub.appendChild(li); }
          else { list.appendChild(li); lastLi = li; sub = null; }
        }
        root.appendChild(list);
        continue;
      }
      if (!ln.trim()) { i++; continue; }
      const buf = [ln];
      i++;
      while (i < lines.length && lines[i].trim() && !/^(#{1,4}\s|```|\s*\||\s*([-*]|\d+\.)\s)/.test(lines[i])) buf.push(lines[i++]);
      root.appendChild(h('p', null, inline(buf.join(' '))));
    }
    return root;
  }
  function analysis() {
    const host = h('div', null, 'loading analysis…');
    shard('data/analysis.js', 'analysis').then(a => { host.textContent = ''; host.appendChild(markdown(a.markdown)); }).catch(e => { host.textContent = 'No analysis exported (' + e.message + ').'; });
    return host;
  }

  const VIEWS = { '': viewArm, cli: viewCli, call: viewCall, builds: viewBuilds, build: (n, r) => viewBuild(n, r.id), graphs: viewGraphs, graph: viewGraphs, events: viewEvents, event: viewEvents, agents: viewAgents, agent: viewAgents, helpers: viewHelpers, state: viewState };
  const PAGES = { home, provenance, analysis, search };

  // ------------------------------------------------------------------ boot
  document.getElementById('searchform').addEventListener('submit', e => {
    e.preventDefault();
    const q = document.getElementById('q').value.trim();
    if (q) location.hash = '#/search?q=' + enc(q);
  });
  window.addEventListener('hashchange', route);
  shard('data/index.js', 'index').then(idx => {
    IDX = idx;
    (idx.sources || []).forEach(s => SRC[s.id] = s);
    route();
  }).catch(e => {
    const main = document.getElementById('main');
    main.textContent = '';
    main.appendChild(h('div', { class: 'badbox' }, 'Could not load data/index.js: ' + e.message + '. Run the exporter first (see README).'));
  });
})();
