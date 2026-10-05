// scout frontend (T014, D1): plain ES2020, no imports, no build step, no external URLs.
// Talks only to POST /api/scans and GET /api/scans/{id}?since=N.
// All data text goes through textContent / setAttribute (tensor names come from untrusted headers).
(function () {
  "use strict";

  const POLL_MS = 250;
  const SVG_NS = "http://www.w3.org/2000/svg";
  const PALETTE = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#b07aa1", "#edc948", "#76b7b2", "#ff9da7"];
  const COL_W = 180;      // graph: x = depth * 180
  const ROW_H = 22;       // graph: y = leaf order * 22
  const BOX_H = 18;
  const CHAR_W = 6.7;     // approx. width of one 11px monospace glyph
  const MAX_LABEL = 24;   // longer labels are truncated with an ellipsis (full id in <title>)
  const STRIP_H = 26;
  const MIN_CELL = 4;

  const $ = (id) => document.getElementById(id);
  const el = {
    form: $("scan-form"), target: $("target"), scan: $("scan"), status: $("status"),
    meta: $("counter-meta"), header: $("counter-header"), weight: $("counter-weight"),
    log: $("log"), cards: $("cards"),
  };

  let runId = 0;          // bumps on every new scan; stale poll loops stop themselves
  let uid = 0;            // unique ids for SVG <pattern> elements
  const stripRenderers = [];

  // ------------------------------------------------------------------ helpers

  function h(tag, attrs, text) {
    const n = document.createElement(tag);
    if (attrs) for (const k in attrs) n.setAttribute(k, attrs[k]);
    if (text !== undefined && text !== null) n.textContent = String(text);
    return n;
  }

  function s(tag, attrs, text) {
    const n = document.createElementNS(SVG_NS, tag);
    if (attrs) for (const k in attrs) n.setAttribute(k, String(attrs[k]));
    if (text !== undefined && text !== null) n.textContent = String(text);
    return n;
  }

  function svgTitle(parent, text) {
    parent.appendChild(s("title", null, text));
  }

  // Exact byte count with thousands separators: "0 B", "12,345 B".
  function fmtBytes(n) {
    return String(n || 0).replace(/\B(?=(\d{3})+(?!\d))/g, ",") + " B";
  }

  // Parameter count in human units (K/M/B, 2 decimals); below 1000 it is shown as-is.
  function fmtParams(n) {
    const a = Math.abs(n);
    if (a >= 1e9) return (n / 1e9).toFixed(2) + " B";
    if (a >= 1e6) return (n / 1e6).toFixed(2) + " M";
    if (a >= 1e3) return (n / 1e3).toFixed(2) + " K";
    return String(n);
  }

  // Same one-line format as the CLI's format_event (T012).
  function formatEvent(e) {
    const t = e.totals || {};
    let line = "[" + String(e.stage).padEnd(8) + "] " + String(e.event).padEnd(12) + " " +
      (e.path || "-") + " +" + e.bytes + "B  meta=" + t.meta + " header=" + t.header + " weight=" + t.weight;
    if (e.note) line += "  (" + e.note + ")";
    return line;
  }

  // ------------------------------------------------------------------ status, counter, log

  function setStatus(kind, elapsed, error) {
    let text = kind;
    if (typeof elapsed === "number") text += " · " + elapsed.toFixed(1) + " s";
    if (error) text += " · " + error;
    el.status.textContent = text;
    el.status.className = kind;
  }

  function setCounter(totals) {
    const t = totals || { meta: 0, header: 0, weight: 0 };
    el.meta.textContent = "meta " + fmtBytes(t.meta);
    el.header.textContent = "header " + fmtBytes(t.header);
    el.weight.textContent = "weight " + fmtBytes(t.weight);
    el.weight.className = "counter " + (t.weight > 0 ? "nonzero" : "zero");
  }

  function appendEvents(events) {
    if (!events.length) return;
    const log = el.log;
    const atBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 24;
    const frag = document.createDocumentFragment();
    for (const e of events) {
      const li = h("li", null, formatEvent(e));
      const w = e.bytes_by_class ? e.bytes_by_class.weight : 0;
      if (e.event === "error" || e.event === "refused" || w > 0) li.className = "bad";
      else if (e.event === "stage_start") li.className = "stage";
      frag.appendChild(li);
    }
    log.appendChild(frag);
    if (atBottom) log.scrollTop = log.scrollHeight;
  }

  // ------------------------------------------------------------------ scan + polling

  function sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }

  async function startScan() {
    const target = el.target.value.trim();
    if (!target) { setStatus("error", null, "enter a Hub repo (org/name[@rev]) or a local folder"); return; }
    const my = ++runId;
    el.scan.disabled = true;
    el.log.textContent = "";
    el.cards.textContent = "";
    stripRenderers.length = 0;
    setCounter(null);
    setStatus("running");
    try {
      const r = await fetch("/api/scans", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ target: target }),
      });
      const body = await r.json().catch(() => ({}));
      if (r.status !== 202) { setStatus("error", null, body.error || ("HTTP " + r.status)); return; }
      await poll(my, body.scan_id);
    } catch (err) {
      if (my === runId) setStatus("error", null, String(err && err.message ? err.message : err));
    } finally {
      if (my === runId) el.scan.disabled = false;
    }
  }

  async function poll(my, scanId) {
    let lastSeq = 0;
    let rendered = false;
    for (;;) {
      if (my !== runId) return;
      const r = await fetch("/api/scans/" + encodeURIComponent(scanId) + "?since=" + lastSeq, { cache: "no-store" });
      if (my !== runId) return;
      const body = await r.json().catch(() => ({}));
      if (r.status !== 200) { setStatus("error", null, body.error || ("HTTP " + r.status)); return; }
      const events = body.events || [];
      appendEvents(events);
      if (events.length) lastSeq = events[events.length - 1].seq;
      setCounter(body.totals);
      if (body.status === "running") {
        setStatus("running", body.elapsed_s);
        await sleep(POLL_MS);
        continue;
      }
      const err = body.error ? body.error.type + ": " + body.error.message : null;
      setStatus(body.status, body.elapsed_s, err);
      if (!rendered && body.status === "done") { renderViews(body.views || []); rendered = true; }
      // The server caps events per poll; drain until a poll returns nothing new.
      if (!events.length) return;
    }
  }

  el.form.addEventListener("submit", (ev) => { ev.preventDefault(); if (!el.scan.disabled) startScan(); });

  // ------------------------------------------------------------------ views

  function renderViews(views) {
    el.cards.textContent = "";
    stripRenderers.length = 0;
    if (!views.length) el.cards.appendChild(h("p", { class: "muted" }, "scan finished with no views"));
    for (const v of views) el.cards.appendChild(renderView(v));
  }

  function renderView(v) {
    const sec = h("section", { class: "view panel" });

    // a0. disclaimers: always visible at the top of every view section (invariant 5)
    const notes = h("ul", { class: "disclaimers", role: "note", "aria-label": "disclaimers" });
    for (const d of v.disclaimers || []) notes.appendChild(h("li", null, d));
    sec.appendChild(notes);

    // a. title + claims
    sec.appendChild(h("h2", null, v.title));
    sec.appendChild(renderClaims(v.model_card || {}));

    // b. summary line
    const sm = v.summary || {};
    const p = h("p", { class: "summary" });
    p.appendChild(document.createTextNode(
      "params " + fmtParams(sm.params_total) + " · tensors " + sm.n_tensors + " · stacks " + sm.n_stacks +
      " · expert groups " + sm.n_expert_groups + " · "));
    p.appendChild(h("span", { class: sm.weight_bytes_read > 0 ? "nonzero" : "zero" }, "weight " + fmtBytes(sm.weight_bytes_read)));
    p.appendChild(document.createTextNode(
      " read · header " + fmtBytes(sm.header_bytes_read) + " · meta " + fmtBytes(sm.meta_bytes_read)));
    sec.appendChild(p);

    // c. depth strips
    const strips = v.depth_strips || [];
    sec.appendChild(h("h3", null, "depth strips"));
    if (!strips.length) sec.appendChild(h("p", { class: "muted" }, "no repeated stacks"));
    for (const st of strips) sec.appendChild(renderStrip(st));

    // d. graph
    sec.appendChild(h("h3", null, "graph"));
    sec.appendChild(renderGraphBlock(v.nodes || []));
    return sec;
  }

  function renderClaims(mc) {
    const dl = h("dl", { class: "claims" });
    dl.appendChild(h("div", { class: "caption" }, "claimed by model card"));
    const row = (k, val) => { dl.appendChild(h("dt", null, k)); dl.appendChild(h("dd", null, val)); };
    if (!mc.present) {
      row("model card", "no model card");
      return dl;
    }
    const list = (a) => (Array.isArray(a) && a.length ? a.join(", ") : "none claimed");
    row("base_model", list(mc.base_model) + (mc.base_model_relation ? "  (" + mc.base_model_relation + ")" : ""));
    row("license", (mc.license || "none claimed") + (mc.license_name ? "  (" + mc.license_name + ")" : ""));
    row("tags", list(mc.tags));
    row("pipeline_tag", mc.pipeline_tag || "none claimed");
    if (mc.parse_error) row("parse error", mc.parse_error);
    return dl;
  }

  // ---- depth strip

  function signatureColours(cells) {
    const counts = new Map();
    for (const c of cells) counts.set(c.signature, (counts.get(c.signature) || 0) + 1);
    let common = null, best = -1;
    for (const [sig, n] of counts) if (n > best) { best = n; common = sig; }   // ties: first seen wins
    const colours = new Map();
    let i = 0;
    for (const sig of counts.keys()) {
      colours.set(sig, sig === common ? "var(--neutral)" : PALETTE[i++ % PALETTE.length]);
    }
    return { colours, counts };
  }

  function renderStrip(st) {
    const wrap = h("div", { class: "strip" });
    wrap.appendChild(h("div", { class: "strip-label" }, st.prefix + " · " + st.depth));
    const box = h("div", { class: "strip-svg" });
    wrap.appendChild(box);
    const cells = st.cells || [];
    const { colours, counts } = signatureColours(cells);
    const hasMoe = cells.some((c) => c.moe);
    const pid = "hatch" + (++uid);

    function draw() {
      const avail = Math.max(box.clientWidth || 600, 50);
      const cw = Math.max(MIN_CELL, avail / Math.max(cells.length, 1));
      const width = Math.ceil(cw * cells.length);
      const svg = s("svg", { width: width, height: STRIP_H, viewBox: "0 0 " + width + " " + STRIP_H, role: "img" });
      svgTitle(svg, st.prefix + " · " + st.depth + " blocks");
      if (hasMoe) {
        const defs = s("defs");
        const pat = s("pattern", { id: pid, width: 5, height: 5, patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)" });
        pat.appendChild(s("line", { x1: 0, y1: 0, x2: 0, y2: 5, class: "hatch-line" }));
        defs.appendChild(pat);
        svg.appendChild(defs);
      }
      cells.forEach((c, j) => {
        const g = s("g");
        const x = j * cw;
        const w = Math.max(cw - (cw >= 8 ? 1 : 0), 1);   // 1 px gap between cells when there is room
        g.appendChild(s("rect", { x: x, y: 0, width: w, height: STRIP_H, style: "fill: " + colours.get(c.signature) }));
        if (c.moe) g.appendChild(s("rect", { x: x, y: 0, width: w, height: STRIP_H, fill: "url(#" + pid + ")" }));
        svgTitle(g, "index " + c.index + "\nparams " + fmtParams(c.params) + " (" + c.params + ")\nsignature " +
          c.signature + "\nn_experts " + (c.n_experts === null || c.n_experts === undefined ? "-" : c.n_experts));
        svg.appendChild(g);
      });
      box.textContent = "";
      box.appendChild(svg);
    }
    stripRenderers.push(draw);
    // Drawn once attached (needs the real width); redrawn on resize.
    requestAnimationFrame(draw);

    const legend = h("div", { class: "legend" });
    for (const [sig, n] of counts) {
      const item = h("span");
      const sw = h("span", { class: "swatch" });
      sw.style.background = colours.get(sig);
      item.appendChild(sw);
      item.appendChild(document.createTextNode(sig + " ×" + n));
      legend.appendChild(item);
    }
    if (hasMoe) legend.appendChild(h("span", null, "hatched = MoE block"));
    wrap.appendChild(legend);
    return wrap;
  }

  let resizeTimer = 0;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => stripRenderers.forEach((f) => f()), 120);
  });

  // ---- graph

  function renderGraphBlock(nodes) {
    const block = h("div");
    const tools = h("div", { class: "graph-tools" });
    const lab = h("label");
    const cb = h("input", { type: "checkbox" });
    lab.appendChild(cb);
    lab.appendChild(document.createTextNode(" show tensors"));
    tools.appendChild(lab);
    tools.appendChild(h("span", { class: "muted" }, "click a node to collapse / expand"));
    block.appendChild(tools);
    const wrap = h("div", { class: "graph-wrap" });
    block.appendChild(wrap);

    const byId = new Map();
    const children = new Map();
    for (const n of nodes) { byId.set(n.id, n); children.set(n.id, []); }
    for (const n of nodes) {
      if (n.parent !== null && n.parent !== undefined && children.has(n.parent)) children.get(n.parent).push(n);
    }
    const root = nodes.find((n) => n.kind === "root") || nodes[0];
    const collapsed = new Set();

    function draw() {
      wrap.textContent = "";
      if (!root) { wrap.appendChild(h("p", { class: "muted" }, "no nodes")); return; }
      const showTensors = cb.checked;
      const visKids = (n) => (collapsed.has(n.id) ? [] : children.get(n.id).filter((c) => showTensors || c.kind !== "tensor"));
      const hasKids = (n) => children.get(n.id).some((c) => showTensors || c.kind !== "tensor");

      // Layout: leaves take consecutive rows; a parent sits midway between its first and last child.
      const pos = new Map();
      let row = 0, maxRight = 0;
      (function place(n, depth) {
        const kids = visKids(n);
        let y;
        if (!kids.length) y = row++ * ROW_H;
        else {
          for (const c of kids) place(c, depth + 1);
          y = (pos.get(kids[0].id).y + pos.get(kids[kids.length - 1].id).y) / 2;
        }
        const label = n.label.length > MAX_LABEL ? n.label.slice(0, MAX_LABEL - 1) + "…" : n.label;
        const w = Math.min(COL_W - 12, Math.ceil(label.length * CHAR_W) + 12);
        const x = depth * COL_W;
        pos.set(n.id, { x: x, y: y, w: w, label: label });
        maxRight = Math.max(maxRight, x + w);
      })(root, 0);

      const pad = 8;
      const width = maxRight + pad * 2;
      const height = row * ROW_H + pad * 2;
      const svg = s("svg", { width: width, height: height, viewBox: (-pad) + " " + (-pad - BOX_H / 2) + " " + width + " " + height });

      const edges = s("g");
      const boxes = s("g");
      svg.appendChild(edges);
      svg.appendChild(boxes);
      (function emit(n) {
        const p = pos.get(n.id);
        const kids = visKids(n);
        for (const c of kids) {
          const q = pos.get(c.id);
          const mx = q.x - 10;
          edges.appendChild(s("path", { class: "edge", d: "M" + (p.x + p.w) + " " + p.y + " H" + mx + " V" + q.y + " H" + q.x }));
          emit(c);
        }
        const isCollapsed = collapsed.has(n.id) && hasKids(n);
        const g = s("g", { class: n.kind + (isCollapsed ? " collapsed" : "") + (hasKids(n) ? " clickable" : "") });
        const y0 = p.y - BOX_H / 2;
        if (n.kind === "stack" || n.kind === "expert_group") {
          // double outline
          g.appendChild(s("rect", { class: "box", x: p.x - 3, y: y0 - 3, width: p.w + 6, height: BOX_H + 6, rx: 4 }));
        }
        g.appendChild(s("rect", { class: "box", x: p.x, y: y0, width: p.w, height: BOX_H, rx: 3 }));
        g.appendChild(s("text", { x: p.x + 6, y: p.y + 4 }, p.label + (isCollapsed ? " +" : "")));
        let tip = (n.id || "(root)") + "\nkind " + n.kind + "\nparams " + fmtParams(n.params) + " (" + n.params + ")";
        if (n.count > 1) tip += "\ncount ×" + n.count;
        if (n.kind === "tensor") tip += "\nshape " + (n.shape ? "[" + n.shape.join(", ") + "]" : "varies") + "\ndtype " + n.dtype;
        svgTitle(g, tip);
        if (hasKids(n)) {
          g.addEventListener("click", () => {
            if (collapsed.has(n.id)) collapsed.delete(n.id); else collapsed.add(n.id);
            draw();
          });
        }
        boxes.appendChild(g);
      })(root);
      wrap.appendChild(svg);
    }

    cb.addEventListener("change", draw);
    draw();
    return block;
  }

  setCounter(null);
})();
