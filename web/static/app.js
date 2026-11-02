// p-bit chip viewer: talks to web/server.py, which drives the Python model,
// the Verilator build of rtl/top.v or a Tang Nano 9K with the same UART bytes.
"use strict";

const $ = (s) => document.querySelector(s);
const SVG = "http://www.w3.org/2000/svg";
const S = {
  chips: [], chip: null, connected: false, prog: null, programs: null, mode: "pin", sel: null, lastState: null,
  samples: [], playing: false, pm: null, pmTraj: [], pmPlaying: false, selBlock: null,
};

// ------------------------------------------------------------------ helpers
function h(tag, attrs = {}, text) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v; else e.setAttribute(k, v);
  }
  if (text !== undefined) e.textContent = text;
  return e;
}
function sv(tag, attrs = {}, parent) {
  const e = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}
function svText(parent, x, y, str, attrs = {}) {
  const t = sv("text", { x, y, ...attrs }, parent);
  t.textContent = str;
  return t;
}
function clear(e) { while (e.firstChild) e.removeChild(e.firstChild); return e; }
function css(name) { return getComputedStyle($(".viz-root")).getPropertyValue(name).trim(); }
const fmt = (v, d = 3) => {
  if (Math.abs(v) >= 1000) return Math.round(v).toLocaleString();
  const s = (+v).toFixed(d);
  return s.includes(".") ? s.replace(/0+$/, "").replace(/\.$/, "") : s;
};

async function api(path, body) {
  const opt = body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  const r = await fetch(path, opt);
  const j = await r.json();
  if (!r.ok || j.error) throw new Error(j.error || r.statusText);
  return j;
}
function showError(e) {
  const box = $("#error");
  if (!e) { box.hidden = true; return; }
  box.textContent = e.message || String(e);
  box.hidden = false;
}
function showWarn(list) {
  const box = $("#warn");
  box.hidden = !list || !list.length;
  if (!box.hidden) box.textContent = list.join(" ");
}
async function guarded(btn, fn) {
  showError(null);
  if (btn) btn.disabled = true;
  try { return await fn(); } catch (e) { showError(e); } finally { if (btn) btn.disabled = false; }
}
function setStatus(txt, cls = "") { const s = $("#status"); s.textContent = txt; s.className = "status " + cls; }

// tooltip (values lead, labels follow; built with textContent only)
function tip(evt, rows) {
  const t = $("#tooltip");
  clear(t);
  for (const [k, v] of rows) {
    const d = h("div");
    if (v !== undefined) { d.appendChild(h("b", {}, String(v))); d.appendChild(document.createTextNode(" ")); }
    d.appendChild(h("span", { class: "tk" }, k));
    t.appendChild(d);
  }
  t.hidden = false;
  const x = Math.min(evt.clientX + 14, window.innerWidth - t.offsetWidth - 8);
  const y = Math.min(evt.clientY + 14, window.innerHeight - t.offsetHeight - 8);
  t.style.left = x + "px";
  t.style.top = y + "px";
}
function untip() { $("#tooltip").hidden = true; }

function bit(sHex, i) { return Number((BigInt("0x" + sHex) >> BigInt(i)) & 1n); }
function bitString(sHex, n) { let o = ""; for (let i = 0; i < n; i++) o += bit(sHex, i); return o; }

// ------------------------------------------------------------------ chips
function renderChips() {
  const box = clear($("#chips"));
  for (const c of S.chips) {
    const b = h("button", { class: "chipcard", "aria-pressed": String(S.chip && S.chip.id === c.id) });
    b.appendChild(h("strong", {}, c.title));
    const f = c.fpga;
    if (f) {
      const r = f.resources;
      const pct = (k) => (r[k] ? `${Math.round(100 * r[k][0] / r[k][1])}%` : "–");
      b.appendChild(h("span", { class: "meta" }, `LUT4 ${pct("LUT4")} · BSRAM ${pct("BSRAM")} · MULT ${r.MULT18X18 ? r.MULT18X18[0] : 0} · Fmax ${f.fmax_mhz} MHz`));
    }
    const badges = h("div", { class: "badges" });
    for (const [be, lab] of [["model", "model"], ["rtl", "RTL"], ["board", "board"]]) {
      badges.appendChild(h("span", { class: "badge" + (c.backends.includes(be) ? "" : " off") }, lab));
    }
    b.appendChild(badges);
    b.addEventListener("click", () => selectChip(c.id));
    box.appendChild(b);
  }
}

function selectChip(id) {
  stopPlay();
  S.chip = S.chips.find((c) => c.id === id);
  S.connected = false;
  S.connWarn = [];
  $("#cur-chip").textContent = S.chip.title;
  for (const inp of document.querySelectorAll('input[name="backend"]')) {
    inp.disabled = !S.chip.backends.includes(inp.value);
    if (inp.disabled && inp.checked) document.querySelector('input[name="backend"][value="model"]').checked = true;
  }
  showWarn(S.chip.note ? [S.chip.note] : []);
  setStatus("not connected");
  renderChips();
  renderArch();
  renderFpga();
  if (ML.list.length) { ML.demo = null; clear($("#ml-panel")); renderMlCards(); }
  const pmode = id === "pmode_chain4";
  $("#spin-controls").hidden = pmode;
  $("#pm-controls").hidden = !pmode;
  $("#results-spin").hidden = pmode;
  $("#results-pm").hidden = !pmode;
  $("#chart1-title").textContent = pmode ? "Soft spins x_i(t)" : "Energy per sweep";
  $("#chart2-title").hidden = pmode;
  $("#chart-beta").hidden = pmode;
  const pysim = id === "pysim";
  for (const t of ["program", "wire"]) document.querySelector(`[data-tab="${t}"]`).disabled = pmode || pysim;
  for (const t of ["run", "results"]) document.querySelector(`[data-tab="${t}"]`).disabled = pysim;
  const cur = document.querySelector('.tabs button[aria-selected="true"]');
  if (cur && cur.disabled) showTab("learn");
  clear($("#graph")); clear($("#chart-energy")); clear($("#chart-beta")); clear($("#now"));
  renderLeds([0, 0, 0, 0, 0], false);
}

// ------------------------------------------------------------------ architecture
const BLOCKS = {
  uart_rx: { t: "UART rx", f: "uart_rx.v", d: "Receives host bytes (115200 8N1 on the board)." },
  fsm: { t: "Command FSM", s: "W H O B X R D S Q", f: "top.v", d: "Decodes the UART protocol: programs RAMs, starts runs, dumps readout." },
  uart_tx: { t: "UART tx", f: "uart_tx.v", d: "Sends replies: acks, histogram, counters, streamed spins." },
  wram: { t: "J / h / order RAM", s: "dense, 2^(2·IW) words", f: "pbit_array.v", d: "Dense couplings J_ij, biases h_i and the colour-ordered visit list." },
  nbr: { t: "Neighbour lists", s: "≤ D per node, per lane", f: "sparse_array.v", d: "Sparse graph: each node keeps ≤ D (neighbour, J) pairs; rounds of P nodes." },
  sched: { t: "β schedule", s: "const / linear / table", f: "schedule_ctrl.v", d: "Annealing: β (Q4.4) per sweep." },
  mac: { t: "MAC  Σ ±J + h", s: "no multipliers", f: "weight_mac.v", d: "Synapse: adds +J or −J per neighbour depending on its spin." },
  pbit: { t: "p-bit", s: "2β·field → σ → compare", f: "pbit.v", d: "Shift-add β×field, sigmoid lookup, compare with a 16-bit random word: P(+1) = σ(2βΣ)." },
  lut: { t: "σ LUT", s: "1024 × 16 BSRAM", f: "sigmoid_lut.v", d: "Sigmoid table (or piecewise-linear shift-add, sigmoid_pwl.v)." },
  lfsr: { t: "LFSR noise", s: "4 primitive polys", f: "noise_bank.v", d: "Galois LFSR lanes, leap-16: the stand-in for transistor noise." },
  lanes: { t: "× P parallel lanes", s: "lock-step engines", f: "sparse_array.v", d: "P MAC + p-bit engines update P same-colour nodes at once." },
  noisectl: { t: "Noise options 'Y'", s: "persistence τ0, shared", f: "sparse_array.v", d: "Slow RNG (refresh prob. q) and shared-noise experiments." },
  temper: { t: "Parallel tempering", s: "K = 4 replicas", f: "temper_ctrl.v", d: "Replica banks at a β ladder; swaps decided by a p-bit on the energy gap." },
  readout: { t: "Readout", s: "histogram + up counters", f: "readout_counter.v", d: "4096-bin state histogram and per-spin counters, dumped by D / U." },
  leds: { t: "LEDs · buttons", s: "led[4:0] = spins", f: "top.v", d: "Stand-alone demo: run button anneals the power-on program." },
};
const PM_BLOCKS = {
  pm_x: { t: "State RAM x_i", s: "Q4.12 soft spins", f: "pmode_array.v", d: "Continuous site values, saturated to [−8, 8)." },
  pm_tanh: { t: "tanh LUT", s: "s = tanh x", f: "pmode_array.v", d: "Sigmoid table reused as tanh." },
  pm_mac: { t: "Coupling MAC", s: "F = Σ A_ij s_j + h_i", f: "pmode_array.v", d: "DSP multipliers (MULT18X18)." },
  pm_cpl: { t: "Chain rule", s: "(1 − s²)·F", f: "pmode_array.v", d: "Gradient of the coupling energy through tanh." },
  pm_quart: { t: "Double well", s: "λ·x(x² − 1)", f: "pmode_array.v", d: "Gradient of the quartic confining potential." },
  pm_noise: { t: "Gaussian noise", s: "LFSR → Gauss LUT · σ", f: "pmode_array.v", d: "σ = √(2Tε) times a table-shaped normal draw." },
  pm_upd: { t: "Langevin update", s: "x ← x − ε∇V + σξ", f: "pmode_array.v", d: "Unadjusted Langevin step (ε = 2^−k)." },
  pm_led: { t: "LEDs · buttons", s: "led[k] = sign x_k", f: "pmode_top.v", d: "Run button toggles stepping at 26 steps/s; no UART." },
};

const PY_BLOCKS = {
  py_state: { t: "State s", s: "C chains × n, ±1", f: "model/pysim.py", d: "numpy array: every chain is one copy of the p-bit array (no 16 / 256 limit)." },
  py_col: { t: "Colour classes", s: "greedy independent sets", f: "model/pysim.py", d: "Nodes of one colour do not interact, so they update together (as the chip's rounds)." },
  py_field: { t: "Local field", s: "h_i + Σ J_ij s_j", f: "model/pysim.py", d: "Float matmul; scipy CSR above 1024 nodes. No Q3.4 rounding, no ±8 saturation." },
  py_sig: { t: "Exact sigmoid", s: "σ(2β·field)", f: "model/pysim.py", d: "Float logistic (the chip uses a 16-bit LUT)." },
  py_rng: { t: "numpy RNG", s: "PCG64 uniforms", f: "model/pysim.py", d: "One uniform per p-bit per sweep (the chip uses 32-bit LFSR lanes)." },
  py_pin: { t: "Hard pins", s: "clamped nodes skipped", f: "model/pysim.py", d: "A pinned p-bit is never updated (on the chip a pin is an extreme bias)." },
  py_train: { t: "Learning", s: "SimTrainer · RBM", f: "model/pysim.py", d: "Same contrastive rule as model/train.py, float weights, many parallel chains." },
  py_ml: { t: "Demos", s: "digits · MRF any size", f: "model/ml.py", d: "Data sets and builders shared with the chip demos." },
};

function renderArch() {
  const c = S.chip;
  $("#arch-title").textContent = c.title;
  $("#arch-summary").textContent = c.summary;
  const box = clear($("#arch-svg"));
  const svg = sv("svg", { viewBox: "0 0 720 340", role: "img", "aria-label": `Block diagram of ${c.title}` }, box);
  const defs = sv("defs", {}, svg);
  const m = sv("marker", { id: "arrow", viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 6, markerHeight: 6, orient: "auto-start-reverse" }, defs);
  sv("path", { d: "M0,0 L10,5 L0,10 z", fill: css("--axis") }, m);
  const on = new Set(c.blocks);
  const place = (id, x, y, w, hh, tab) => {
    const b = tab[id];
    const g = sv("g", { class: "blk" + (on.has(id) || tab !== BLOCKS ? " on" : "") + (S.selBlock === id ? " sel" : ""), tabindex: 0 }, svg);
    sv("rect", { x, y, width: w, height: hh, rx: 6 }, g);
    svText(g, x + w / 2, y + (b.s ? hh / 2 - 3 : hh / 2 + 4), b.t, { "text-anchor": "middle" });
    if (b.s) svText(g, x + w / 2, y + hh / 2 + 11, b.s, { "text-anchor": "middle", class: "small" });
    if (on.has(id) || tab !== BLOCKS) {
      const pick = () => showBlock(id, b);
      g.addEventListener("click", pick);
      g.addEventListener("keydown", (e) => { if (e.key === "Enter") pick(); });
      g.addEventListener("pointermove", (e) => tip(e, [[b.d], [b.f, "file"]]));
      g.addEventListener("pointerleave", untip);
    }
  };
  const flow = (d) => sv("path", { d, class: "flow" }, svg);
  if (c.id === "pysim") {
    place("py_state", 20, 130, 130, 56, PY_BLOCKS);
    place("py_col", 200, 40, 140, 50, PY_BLOCKS);
    place("py_field", 200, 130, 140, 56, PY_BLOCKS);
    place("py_sig", 390, 130, 130, 56, PY_BLOCKS);
    place("py_rng", 390, 40, 130, 50, PY_BLOCKS);
    place("py_pin", 200, 240, 140, 50, PY_BLOCKS);
    place("py_train", 570, 90, 130, 56, PY_BLOCKS);
    place("py_ml", 570, 200, 130, 56, PY_BLOCKS);
    flow("M150,158 L200,158"); flow("M270,90 L270,130"); flow("M340,158 L390,158"); flow("M455,90 L455,130");
    flow("M455,186 C455,320 85,320 85,186"); flow("M270,240 L270,186");
    flow("M520,150 L570,125"); flow("M635,200 L635,146");
    return;
  }
  if (c.id === "pmode_chain4") {
    place("pm_x", 20, 140, 120, 50, PM_BLOCKS);
    place("pm_tanh", 180, 60, 120, 50, PM_BLOCKS);
    place("pm_mac", 340, 60, 140, 50, PM_BLOCKS);
    place("pm_cpl", 520, 60, 120, 50, PM_BLOCKS);
    place("pm_quart", 340, 150, 140, 50, PM_BLOCKS);
    place("pm_noise", 340, 240, 140, 50, PM_BLOCKS);
    place("pm_upd", 520, 170, 140, 60, PM_BLOCKS);
    place("pm_led", 20, 260, 120, 46, PM_BLOCKS);
    flow("M140,155 C160,155 160,85 180,85"); flow("M300,85 L340,85"); flow("M480,85 L520,85");
    flow("M580,110 L590,170"); flow("M140,175 L340,175"); flow("M480,175 L520,190");
    flow("M480,265 C500,265 500,215 520,215"); flow("M520,215 C300,320 120,260 80,190");
    flow("M80,190 L80,260");
    return;
  }
  sv("rect", { x: 150, y: 20, width: 420, height: 300, rx: 10, class: "grp" }, svg);
  svText(svg, 160, 314, c.blocks.includes("nbr") ? "p-bit array: sparse_array.v" : "p-bit array: pbit_array.v", { class: "grptext" });
  place("uart_rx", 20, 40, 105, 40, BLOCKS);
  place("fsm", 20, 130, 105, 60, BLOCKS);
  place("uart_tx", 20, 250, 105, 40, BLOCKS);
  place("wram", 165, 40, 125, 44, BLOCKS);
  place("nbr", 300, 40, 125, 44, BLOCKS);
  place("sched", 435, 40, 120, 44, BLOCKS);
  place("mac", 165, 130, 125, 50, BLOCKS);
  place("pbit", 300, 130, 125, 50, BLOCKS);
  place("lut", 435, 130, 120, 50, BLOCKS);
  place("lanes", 165, 225, 125, 50, BLOCKS);
  place("lfsr", 300, 225, 125, 50, BLOCKS);
  place("noisectl", 435, 225, 120, 50, BLOCKS);
  place("temper", 590, 40, 122, 50, BLOCKS);
  place("readout", 590, 140, 122, 50, BLOCKS);
  place("leds", 590, 250, 122, 44, BLOCKS);
  flow("M72,80 L72,130"); flow("M72,190 L72,250");
  flow("M125,150 C145,150 145,62 165,62");
  flow("M228,84 L228,130"); flow("M362,84 C362,105 250,105 245,130");
  flow("M290,155 L300,155"); flow("M435,155 L425,155"); flow("M362,225 L362,180");
  flow("M450,84 L410,130");
  flow("M425,140 C450,108 560,108 590,158"); flow("M425,175 C460,203 560,203 590,268");
  flow("M590,180 C570,182 580,334 360,334 C160,334 150,285 125,272");
  if (on.has("temper")) flow("M651,90 L651,140");
}

async function showBlock(id, b) {
  S.selBlock = id;
  renderArch();
  const path = b.f.includes("/") ? b.f : `rtl/${b.f}`;
  $("#block-title").textContent = `${b.t}: ${path}`;
  $("#block-desc").textContent = b.d;
  const pre = $("#block-src");
  pre.textContent = "loading…";
  try {
    const r = await fetch("/" + path);
    pre.textContent = r.ok ? await r.text() : "source not found";
  } catch (e) { pre.textContent = String(e); }
}

function renderFpga() {
  const c = S.chip, f = c.fpga, t = clear($("#fpga-table"));
  if (!f) { t.appendChild(h("tr")).appendChild(h("td", {}, "no P&R log in fpga/build")); return; }
  const hr = t.appendChild(h("tr"));
  ["Resource", "Used", "Available", "%"].forEach((k, i) => hr.appendChild(h("th", i ? { class: "num" } : {}, k)));
  for (const [k, [u, a]] of Object.entries(f.resources)) {
    const r = t.appendChild(h("tr"));
    r.appendChild(h("td", {}, k));
    r.appendChild(h("td", { class: "num" }, u.toLocaleString()));
    r.appendChild(h("td", { class: "num" }, a.toLocaleString()));
    r.appendChild(h("td", { class: "num" }, `${Math.round(100 * u / a)}%`));
  }
  const r = t.appendChild(h("tr"));
  r.appendChild(h("td", {}, "Fmax"));
  r.appendChild(h("td", { class: "num" }, `${f.fmax_mhz} MHz`));
  r.appendChild(h("td", { class: "num" }, "27 MHz target"));
  r.appendChild(h("td", { class: "num" }, f.fmax_mhz >= 27 ? "PASS" : "FAIL"));
  $("#fpga-flash").textContent = `bitstream: ${f.bitstream}${f.bitstream_built ? "" : " (not built)"} · flash: cd fpga && ${c.flash}`;
}

function showTab(name) { const b = document.querySelector(`.tabs button[data-tab="${name}"]`); if (b && !b.disabled) b.click(); }
const engWord = () => (S.chip && S.chip.id === "pysim" ? "simulator" : "chip");

// ------------------------------------------------------------------ connect
async function connect() {
  stopPlay();
  const backend = document.querySelector('input[name="backend"]:checked').value;
  const port = $("#port").value;
  setStatus(backend === "rtl" ? "building / starting the Verilog simulator…" : "connecting…", "busy");
  await guarded($("#connect"), async () => {
    const r = await api("/api/connect", { chip: S.chip.id, backend, port });
    S.connected = true;
    S.backend = backend;
    S.samples = [];
    ML.demo = null; ML.training = false;
    clear($("#ml-panel"));
    S.connWarn = r.warnings || [];
    showWarn(S.connWarn.concat(S.chip.note ? [S.chip.note] : []));
    if (S.chip.id === "pysim") {
      S.prog = null;
      setStatus("ideal Python simulator ready (Learn tab)", "ok");
    } else if (S.chip.id === "pmode_chain4") {
      S.pm = await api("/api/pmode/info", {});
      S.pmTraj = [];
      setStatus("p-mode Python model ready", "ok");
      renderPmGraph(S.pm.x0);
      await pmStep(200);
    } else {
      const p = r.ping;
      setStatus(`${backend} · N=${p.N} HIST_BITS=${p.HIST_BITS}${p.sparse ? ` · SPARSE P=${p.sparse.P} D=${p.sparse.D}` : ""} · ${r.connect_s}s`, "ok");
      setProgram(r.program);
      await stepOnce();
    }
  });
  renderMlCards();
  if (!S.connected) setStatus("not connected");
}

// ------------------------------------------------------------------ program
function setProgram(p) {
  // a freshly loaded program: reset the run history, then draw it
  S.samples = [];
  S.sel = null;
  S.lastState = null;
  $("#run-beta").value = p.beta;
  $("#smp-beta").value = p.beta;
  clear($("#smp-tiles")); clear($("#smp-hist")); clear($("#smp-mag")); clear($("#sol-out"));
  showProgram(p);
  renderCharts();
}

function showProgram(p) {
  // a program summary from the server (load or live edit): redraw everything that depends on it
  S.prog = p;
  showWarn((S.connWarn || []).concat(p.warnings || [], S.chip && S.chip.note ? [S.chip.note] : []));
  $("#prog-title").textContent = `Loaded: ${p.name} (${p.kind})${p.edited ? " · edited live" : ""}`;
  const tiles = clear($("#prog-stats"));
  const tile = (k, v) => { const d = h("div", { class: "tile" }); d.appendChild(h("div", { class: "v" }, String(v))); d.appendChild(h("div", { class: "k" }, k)); tiles.appendChild(d); };
  tile("p-bits", p.n); tile("couplings", p.edges.length); tile("β", p.beta); tile("colours", p.colors.length);
  tile("pinned", Object.keys(p.pins).length);
  if (p.ladder) tile("tempering ladder β", p.ladder.map((b) => b / 16).join(" · "));
  const g = clear($("#prog-ground"));
  if (p.ground) {
    g.appendChild(h("p", { class: "muted" }, `${p.ground.count} ground state(s), energy ${fmt(p.ground.energy, 4)}`));
    for (const st of p.ground.states) g.appendChild(h("div", { class: "mono" }, st.lines.join("  |  ")));
  } else g.appendChild(h("p", { class: "muted" }, "Too many p-bits to enumerate (2^n states)."));
  const jb = clear($("#prog-j"));
  const t = jb.appendChild(h("table", { class: "tbl mono" }));
  const hr = t.appendChild(h("tr"));
  ["i", "j", "J_ij"].forEach((k, i) => hr.appendChild(h("th", i === 2 ? { class: "num" } : {}, k)));
  for (const [i, j, w] of p.edges.slice(0, 300)) {
    const r = t.appendChild(h("tr"));
    r.appendChild(h("td", {}, p.names[i])); r.appendChild(h("td", {}, p.names[j])); r.appendChild(h("td", { class: "num" }, fmt(w, 4)));
  }
  if (p.edges.length > 300) jb.appendChild(h("p", { class: "muted" }, `… ${p.edges.length - 300} more`));
  renderGraph(S.lastState || null);
  renderGroups();
  renderInspector();
  $("#edit-status").textContent = p.edited ? `edited live · revision ${p.rev}` : "as loaded";
  $("#reset-program").disabled = !p.edited;
}

function fillPrograms() {
  const ex = clear($("#prog-example"));
  for (const e of S.programs.examples) ex.appendChild(h("option", { value: e.name }, `${e.name} (${e.type})`));
  const cs = clear($("#prog-case"));
  for (const c of S.programs.cases) cs.appendChild(h("option", { value: c.name }, `${c.name} (n=${c.n})`));
  showExample();
}
function showExample() {
  const e = S.programs.examples.find((x) => x.name === $("#prog-example").value);
  if (e) $("#prog-json").value = JSON.stringify(e.spec, null, 2);
}
async function loadProgram(req, btn) {
  if (!S.connected) return showError(new Error("connect to a chip first"));
  stopPlay();
  if (req.example) { $("#prog-example").value = req.example; showExample(); }
  if (req.case) $("#prog-case").value = req.case;
  await guarded(btn, async () => { setProgram(await api("/api/program", req)); await stepOnce(); });
}

// ------------------------------------------------------------------ live graph
function nodeRadius(n) { return n <= 16 ? 15 : n <= 64 ? 9 : 8; }
const sigma = (x) => 1 / (1 + Math.exp(-x));
function curBeta() { return S.samples.length ? S.samples[S.samples.length - 1].beta : (S.prog ? S.prog.beta : 1); }
function pinOf(i) { const v = S.prog.pins[String(i)]; return v === undefined ? null : v; }
function neighbours(i) {
  const out = [];
  for (const [a, b, w] of S.prog.edges) { if (a === i) out.push([b, w]); else if (b === i) out.push([a, w]); }
  return out;
}
function holdProb(i) {
  // worst case: every neighbour pulls against the pin; the chip can only bias, not force
  const p = S.prog, against = neighbours(i).reduce((s, [, w]) => s + Math.abs(w), 0);
  return sigma(2 * curBeta() * (Math.abs(p.h[i]) - against));
}

function renderGraph(sHex) {
  const p = S.prog;
  if (!p) return;
  const box = $("#graph");
  let svg = box.querySelector("svg");
  const W = 640, H = 440, pad = 34;
  const X = (u) => pad + u * (W - 2 * pad), Y = (v) => pad + v * (H - 2 * pad);
  const key = `${p.name}#${p.rev}`;
  if (!svg || svg.dataset.key !== key) {
    clear(box);
    svg = sv("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": `Coupling graph of ${p.name}` }, box);
    svg.dataset.key = key;
    const editing = () => S.mode === "edit";
    for (const [i, j, w] of p.edges) {
      const sel = S.sel && S.sel.type === "edge" && ((S.sel.i === i && S.sel.j === j) || (S.sel.i === j && S.sel.j === i));
      const at = { x1: X(p.pos[i][0]), y1: Y(p.pos[i][1]), x2: X(p.pos[j][0]), y2: Y(p.pos[j][1]) };
      sv("line", { ...at, class: "e" + (w < 0 ? " neg" : "") + (sel ? " sel" : ""), "stroke-width": Math.min(4, 0.6 + Math.abs(w) * (p.n > 64 ? 0.5 : 0.9)) }, svg);
      const hit = sv("line", { ...at, class: "ehit" }, svg);
      hit.addEventListener("pointermove", (e) => tip(e, [[`${p.names[i]} — ${p.names[j]}`], [fmt(w, 4), "J"], [editing() ? "click to edit" : "switch to Edit to change"]]));
      hit.addEventListener("pointerleave", untip);
      hit.addEventListener("click", () => { if (editing()) { S.sel = { type: "edge", i, j }; showProgram(S.prog); } });
    }
    const r = nodeRadius(p.n);
    for (let i = 0; i < p.n; i++) {
      const g = sv("g", { class: "node" }, svg);
      const cx = X(p.pos[i][0]), cy = Y(p.pos[i][1]);
      const pin = pinOf(i);
      const sel = S.sel && S.sel.type === "node" && S.sel.i === i;
      if (pin !== null) sv("circle", { cx, cy, r: r + 4, class: "pin" }, g);
      if (sel) sv("circle", { cx, cy, r: r + 7, class: "selring" }, g);
      const c = sv("circle", { cx, cy, r, class: "n", id: "n" + i }, g);
      if (p.n <= 64) svText(g, cx, cy + 4, "", { class: "nl sign", id: "t" + i, "text-anchor": "middle", fill: "#fff", "font-weight": 700 });
      if (p.n <= 32 && p.names[i] !== String(i)) svText(g, cx, cy + r + 12, p.names[i], { class: "nl", "text-anchor": "middle" });
      else if (p.n <= 16) svText(g, cx, cy + r + 12, String(i), { class: "nl", "text-anchor": "middle" });
      if (pin !== null) {
        const bx = cx + r * 0.75, by = cy - r * 0.75;
        sv("rect", { x: bx - 6, y: by - 7, width: 12, height: 13, rx: 3, class: "pinbadge" }, g);
        svText(g, bx, by + 3, String(pin), { class: "pinbadge-t", "text-anchor": "middle" });
      }
      const hit = sv("circle", { cx, cy, r: r + 6, fill: "transparent", class: "nhit" }, g);
      hit.addEventListener("pointermove", (e) => {
        const s = c.dataset.s;
        const rows = [[`p-bit ${i}${p.names[i] !== String(i) ? " · " + p.names[i] : ""}`], [s === "1" ? "+1" : s === "0" ? "−1" : "?", "spin"],
          [fmt(p.h[i], 4), "bias h"], [neighbours(i).length, "neighbours"]];
        if (pin !== null) rows.push([pin, "pinned to"], [`${(100 * holdProb(i)).toFixed(2)}%`, `holds per update at β ${fmt(curBeta(), 3)} (worst case)`]);
        rows.push([S.mode === "edit" ? "click to edit h and couplings" : "click: free → 1 → 0 → free"]);
        tip(e, rows);
      });
      hit.addEventListener("pointerleave", untip);
      hit.addEventListener("click", () => nodeClick(i));
    }
    const lg = clear($("#graph-legend"));
    const item = (cls, style, lab) => { const s = h("span"); const sw = h("span", { class: cls }); if (style) sw.style.background = style; s.appendChild(sw); s.appendChild(document.createTextNode(lab)); lg.appendChild(s); };
    item("sw", css("--pos"), "spin +1 (bit 1)");
    item("sw", css("--neg"), "spin −1 (bit 0)");
    item("ln", null, "J > 0 (align)");
    item("ln dash", null, "J < 0 (anti-align)");
    item("sw ring", null, "ring + badge = pinned (clamp)");
  }
  if (sHex == null) return;
  S.lastState = sHex;
  for (let i = 0; i < p.n; i++) {
    const b = bit(sHex, i);
    const c = svg.querySelector("#n" + i);
    c.dataset.s = String(b);
    c.setAttribute("fill", b ? css("--pos") : css("--neg"));
    const t = svg.querySelector("#t" + i);
    if (t) t.textContent = b ? "+" : "−";
  }
}

// ------------------------------------------------------------------ live edits
async function liveEdit(path, body) {
  if (!S.connected) return showError(new Error("connect to a chip first"));
  await guarded(null, async () => {
    showProgram(await api(path, body));
    if (!S.playing) await stepOnce();
  });
}
function nodeClick(i) {
  if (S.mode === "edit") { S.sel = { type: "node", i }; showProgram(S.prog); return; }
  const pin = pinOf(i);
  const next = pin === null ? 1 : pin === 1 ? 0 : null;     // free -> 1 -> 0 -> free
  liveEdit("/api/pin", { node: i, value: next });
}

function renderGroups() {
  const box = clear($("#groups"));
  const p = S.prog;
  const groups = Object.entries(p.show || {});
  box.hidden = !groups.length;
  if (!groups.length) return;
  box.appendChild(h("h3", {}, "Clamp a group (little-endian integer)"));
  for (const [g, wires] of groups) {
    const idx = wires.map((w) => p.names.indexOf(w));
    const pins = idx.map((i) => pinOf(i));
    const all = pins.every((v) => v !== null);
    const row = h("div", { class: "row" });
    const lab = h("label", {}, `${g} =`);
    const inp = h("input", { type: "number", min: 0, max: (1 << wires.length) - 1, value: all ? pins.reduce((s, v, k) => s + (v << k), 0) : "" });
    inp.placeholder = "free";
    lab.appendChild(inp);
    row.appendChild(lab);
    const bc = h("button", {}, "Clamp");
    bc.addEventListener("click", () => liveEdit("/api/pin_group", { group: g, value: inp.value === "" ? null : +inp.value }));
    const bf = h("button", { class: "ghost" }, "Free");
    bf.addEventListener("click", () => liveEdit("/api/pin_group", { group: g, value: null }));
    inp.addEventListener("keydown", (e) => { if (e.key === "Enter") bc.click(); });
    row.appendChild(bc); row.appendChild(bf);
    row.appendChild(h("span", { class: "muted" }, `${wires.join(", ")} · ${all ? "clamped" : pins.some((v) => v !== null) ? "partly pinned" : "free"}`));
    box.appendChild(row);
  }
}

function numInput(value, step = 0.0625) {
  return h("input", { type: "number", step, min: -7.9375, max: 7.9375, value });
}
function renderInspector() {
  const box = clear($("#inspector"));
  const p = S.prog;
  if (S.mode !== "edit") { box.hidden = true; return; }
  box.hidden = false;
  if (!S.sel) { box.appendChild(h("p", { class: "muted" }, "Edit mode: click a p-bit or a coupling in the graph.")); return; }
  if (S.sel.type === "edge") {
    const { i, j } = S.sel;
    const w = (p.edges.find(([a, b]) => (a === i && b === j) || (a === j && b === i)) || [0, 0, 0])[2];
    box.appendChild(h("h3", {}, `Coupling ${p.names[i]} — ${p.names[j]}`));
    const row = h("div", { class: "row" });
    const lab = h("label", {}, "J"); const inp = numInput(w); lab.appendChild(inp); row.appendChild(lab);
    const ap = h("button", {}, "Apply"); ap.addEventListener("click", () => liveEdit("/api/coupling", { i, j, J: +inp.value }));
    const rm = h("button", { class: "ghost" }, "Remove"); rm.addEventListener("click", () => { S.sel = null; liveEdit("/api/coupling", { i, j, J: 0 }); });
    row.appendChild(ap); row.appendChild(rm); box.appendChild(row);
    box.appendChild(h("p", { class: "hint" }, "J > 0 makes the two spins agree, J < 0 disagree. Stored in 1/16 steps, |J| ≤ 7.94."));
    return;
  }
  const i = S.sel.i, pin = pinOf(i);
  box.appendChild(h("h3", {}, `p-bit ${i}${p.names[i] !== String(i) ? " · " + p.names[i] : ""}`));
  const pr = h("div", { class: "row" });
  pr.appendChild(h("span", { class: "muted" }, "Pin:"));
  for (const [lab, v] of [["free", null], ["1", 1], ["0", 0]]) {
    const b = h("button", { class: pin === v ? "primary" : "" }, lab);
    b.addEventListener("click", () => liveEdit("/api/pin", { node: i, value: v }));
    pr.appendChild(b);
  }
  box.appendChild(pr);
  const hr = h("div", { class: "row" });
  const hl = h("label", {}, "bias h"); const hin = numInput(p.h_free[i]); hl.appendChild(hin); hr.appendChild(hl);
  const ha = h("button", {}, "Apply"); ha.addEventListener("click", () => liveEdit("/api/bias", { i, h: +hin.value }));
  hr.appendChild(ha);
  if (pin !== null) hr.appendChild(h("span", { class: "hint" }, "(the pin overrides h while pinned)"));
  box.appendChild(hr);
  const nb = neighbours(i);
  if (nb.length) {
    const t = box.appendChild(h("table", { class: "tbl" }));
    const head = t.appendChild(h("tr"));
    for (const k of ["neighbour", "J", ""]) head.appendChild(h("th", {}, k));
    for (const [j, w] of nb.slice(0, 40)) {
      const tr = t.appendChild(h("tr"));
      tr.appendChild(h("td", {}, p.names[j]));
      const td = tr.appendChild(h("td")); const inp = numInput(w); td.appendChild(inp);
      const td2 = tr.appendChild(h("td"));
      const ap = h("button", { class: "small" }, "Apply"); ap.addEventListener("click", () => liveEdit("/api/coupling", { i, j, J: +inp.value }));
      const rm = h("button", { class: "small ghost" }, "Remove"); rm.addEventListener("click", () => liveEdit("/api/coupling", { i, j, J: 0 }));
      td2.appendChild(ap); td2.appendChild(rm);
    }
  }
  const ar = h("div", { class: "row" });
  const sel = h("select");
  for (let j = 0; j < p.n; j++) if (j !== i && !nb.some(([k]) => k === j)) sel.appendChild(h("option", { value: j }, p.names[j]));
  const al = h("label", {}, "connect to"); al.appendChild(sel); ar.appendChild(al);
  const jl = h("label", {}, "J"); const jin = numInput(1); jl.appendChild(jin); ar.appendChild(jl);
  const ad = h("button", {}, "Add"); ad.addEventListener("click", () => liveEdit("/api/coupling", { i, j: +sel.value, J: +jin.value }));
  ar.appendChild(ad);
  box.appendChild(ar);
}

function renderLeds(bits, busy) {
  const box = clear($("#leds"));
  bits.concat([busy ? 1 : 0]).forEach((b, k) => {
    const d = h("div", { class: "ledbox", title: k < 5 ? `led[${k}] = spin ${k}` : "led[5] = busy" });
    d.appendChild(h("div", { class: "led" + (b ? " on" : "") }));
    d.appendChild(h("div", { class: "ledlab" }, k < 5 ? `L${k}` : "busy"));
    box.appendChild(d);
  });
}

// ------------------------------------------------------------------ run loop
async function stepOnce() {
  const chunk = Math.max(1, +$("#run-chunk").value || 1);
  const r = await api("/api/stream", { sweeps: chunk });
  for (const s of r.samples) S.samples.push(s);
  if (S.samples.length > 800) S.samples.splice(0, S.samples.length - 800);
  const last = r.samples[r.samples.length - 1];
  renderGraph(last.s);
  renderLeds(r.leds, S.playing);
  const now = clear($("#now"));
  now.appendChild(h("div", {}, `sweep ${last.t} · β ${fmt(last.beta, 4)} · energy ${fmt(last.e, 4)}`));
  for (const line of r.decode.slice(1)) now.appendChild(h("div", {}, line.length > 120 ? line.slice(0, 120) + "…" : line));
  renderCharts();
}
async function playLoop() {
  while (S.playing) {
    try { await stepOnce(); } catch (e) { showError(e); stopPlay(); break; }
    await new Promise((res) => setTimeout(res, +$("#run-ms").value));
  }
}
function togglePlay() {
  if (!S.connected) return showError(new Error("connect to a chip first"));
  S.playing = !S.playing;
  $("#run-play").textContent = S.playing ? "Pause" : "Play";
  if (S.playing) playLoop();
}
function stopPlay() {
  S.playing = false; S.pmPlaying = false;
  $("#run-play").textContent = "Play";
  $("#pm-play").textContent = "Play";
}

// ------------------------------------------------------------------ charts
function niceTicks(lo, hi, n = 5) {
  if (lo === hi) { lo -= 1; hi += 1; }
  const span = hi - lo, step0 = span / n, mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= step0);
  const t = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) t.push(+v.toFixed(10));
  return { ticks: t, lo: Math.min(lo, t[0]), hi: Math.max(hi, t[t.length - 1]) };
}

// line chart: one y-axis, crosshair + one tooltip listing every series at the x
function lineChart(box, { series, height = 200, xLabel = "", yFmt = (v) => fmt(v, 2), endLabels = false, step = false }) {
  clear(box);
  const W = 560, H = height, L = 46, R = endLabels ? 36 : 12, T = 8, B = 24;
  const all = series.flatMap((s) => s.pts);
  if (!all.length) return;
  const xs = all.map((p) => p[0]), ys = all.map((p) => p[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs, x0 + 1);
  const yt = niceTicks(Math.min(...ys), Math.max(...ys), 4);
  const X = (v) => L + (v - x0) / (x1 - x0) * (W - L - R), Y = (v) => T + (1 - (v - yt.lo) / (yt.hi - yt.lo)) * (H - T - B);
  const svg = sv("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" }, box);
  for (const t of yt.ticks) {
    sv("line", { x1: L, x2: W - R, y1: Y(t), y2: Y(t), class: "gridl" }, svg);
    svText(svg, L - 6, Y(t) + 3, yFmt(t), { "text-anchor": "end" });
  }
  sv("line", { x1: L, x2: W - R, y1: H - B, y2: H - B, class: "base" }, svg);
  for (const t of niceTicks(x0, x1, 5).ticks.filter((v) => v >= x0 && v <= x1 && Number.isInteger(v))) svText(svg, X(t), H - B + 14, fmt(t, 0), { "text-anchor": "middle" });
  if (xLabel) svText(svg, W - R, H - 2, xLabel, { "text-anchor": "end" });
  for (const s of series) {
    let d = "";
    s.pts.forEach(([x, y], k) => {
      if (step && k) d += `H${X(x).toFixed(1)}`;
      d += `${k ? "L" : "M"}${X(x).toFixed(1)},${Y(y).toFixed(1)}`;
    });
    sv("path", { d, class: "line", stroke: s.color }, svg);
    if (endLabels && s.pts.length) {
      const [lx, ly] = s.pts[s.pts.length - 1];
      svText(svg, X(lx) + 5, Y(ly) + 4, s.name, { class: "lab" });
    }
  }
  if (series.length >= 2) {
    const lg = h("div", { class: "legend" });
    for (const s of series) { const it = h("span"); const sw = h("span", { class: "sw" }); sw.style.background = s.color; it.appendChild(sw); it.appendChild(document.createTextNode(s.name)); lg.appendChild(it); }
    box.appendChild(lg);
  }
  const hair = sv("line", { y1: T, y2: H - B, class: "hair", visibility: "hidden" }, svg);
  const dots = series.map((s) => sv("circle", { r: 4, fill: s.color, stroke: css("--surface-1"), "stroke-width": 2, visibility: "hidden" }, svg));
  const hit = sv("rect", { x: L, y: T, width: W - L - R, height: H - T - B, class: "hit" }, svg);
  hit.addEventListener("pointermove", (e) => {
    const r = svg.getBoundingClientRect();
    const xv = x0 + ((e.clientX - r.left) * W / r.width - L) / (W - L - R) * (x1 - x0);
    const rows = [];
    let xs0 = null;
    series.forEach((s, k) => {
      if (!s.pts.length) return;
      let best = s.pts[0];
      for (const p of s.pts) if (Math.abs(p[0] - xv) < Math.abs(best[0] - xv)) best = p;
      xs0 = best[0];
      dots[k].setAttribute("cx", X(best[0])); dots[k].setAttribute("cy", Y(best[1])); dots[k].setAttribute("visibility", "visible");
      rows.push([s.name, yFmt(best[1])]);
    });
    hair.setAttribute("x1", X(xs0)); hair.setAttribute("x2", X(xs0)); hair.setAttribute("visibility", "visible");
    tip(e, [[`${xLabel || "x"} ${fmt(xs0, 0)}`], ...rows.map(([n, v]) => [n, v])]);
  });
  hit.addEventListener("pointerleave", () => { untip(); hair.setAttribute("visibility", "hidden"); dots.forEach((d) => d.setAttribute("visibility", "hidden")); });
}

function roundedBar(x, y0, w, y1, r = 4) {
  // rectangle from the baseline y0 to the data end y1, rounded at the data end only
  const up = y1 < y0, hgt = Math.abs(y0 - y1);
  r = Math.min(r, w / 2, hgt);
  if (hgt < 0.5) return `M${x},${y0}h${w}`;
  if (up) return `M${x},${y0}V${y1 + r}Q${x},${y1} ${x + r},${y1}H${x + w - r}Q${x + w},${y1} ${x + w},${y1 + r}V${y0}Z`;
  return `M${x},${y0}V${y1 - r}Q${x},${y1} ${x + r},${y1}H${x + w - r}Q${x + w},${y1} ${x + w},${y1 - r}V${y0}Z`;
}

// grouped / diverging bars, per-mark tooltip
function barChart(box, { cats, series, height = 220, yFmt = (v) => fmt(v, 3), catLabel = (c) => c, legend = true, diverging = false, tipRows, overlay }) {
  clear(box);
  const W = 560, H = height, L = 46, R = 8, T = 8, B = catLabel(cats[0]) === "" ? 10 : cats.length > 16 ? 44 : 26;
  const vals = series.flatMap((s) => s.vals).concat(overlay ? overlay.vals : []);
  const yt = niceTicks(Math.min(0, ...vals), Math.max(0, ...vals), 4);
  const Y = (v) => T + (1 - (v - yt.lo) / (yt.hi - yt.lo)) * (H - T - B);
  const svg = sv("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" }, box);
  for (const t of yt.ticks) {
    sv("line", { x1: L, x2: W - R, y1: Y(t), y2: Y(t), class: t === 0 ? "base" : "gridl" }, svg);
    svText(svg, L - 6, Y(t) + 3, yFmt(t), { "text-anchor": "end" });
  }
  const bw = (W - L - R) / cats.length, gap = 2, inner = Math.max(1, (bw - 4) / series.length);
  cats.forEach((c, i) => {
    series.forEach((s, k) => {
      const v = s.vals[i];
      const x = L + i * bw + 2 + k * inner;
      const color = diverging ? (v >= 0 ? css("--pos") : css("--neg")) : s.color;
      const p = sv("path", { d: roundedBar(x, Y(0), Math.max(1, inner - gap), Y(v)), fill: color }, svg);
      const hit = sv("rect", { x: L + i * bw, y: T, width: bw, height: H - T - B, fill: "transparent" }, svg);
      hit.addEventListener("pointermove", (e) => tip(e, tipRows ? tipRows(i) : [[catLabel(c)], ...series.map((ss) => [ss.name, yFmt(ss.vals[i])])]));
      hit.addEventListener("pointerleave", untip);
      void p;
    });
    if (cats.length <= 32) {
      const lab = catLabel(c);
      const t = svText(svg, L + i * bw + bw / 2, H - B + 12, lab, { "text-anchor": cats.length > 16 ? "end" : "middle", class: "mono" });
      if (cats.length > 16) t.setAttribute("transform", `rotate(-45 ${L + i * bw + bw / 2} ${H - B + 12})`);
    }
  });
  if (overlay) {
    const d = overlay.vals.map((v, i) => `${i ? "L" : "M"}${(L + i * bw + bw / 2).toFixed(1)},${Y(v).toFixed(1)}`).join("");
    sv("path", { d, class: "line", stroke: overlay.color, "pointer-events": "none" }, svg);
  }
  const all = overlay ? series.concat([overlay]) : series;
  if (legend && all.length >= 2) {
    const lg = h("div", { class: "legend" });
    for (const s of all) { const it = h("span"); const sw = h("span", { class: "sw" }); sw.style.background = s.color; it.appendChild(sw); it.appendChild(document.createTextNode(s.name)); lg.appendChild(it); }
    box.appendChild(lg);
  }
}

function renderCharts() {
  if (S.chip && S.chip.id === "pmode_chain4") return renderPmCharts();
  const pts = S.samples.map((s) => [s.t, s.e]);
  lineChart($("#chart-energy"), { series: [{ name: "energy H(s)", color: css("--series-1"), pts }], xLabel: "sweep", yFmt: (v) => fmt(v, 2) });
  lineChart($("#chart-beta"), { series: [{ name: "β", color: css("--series-1"), pts: S.samples.map((s) => [s.t, s.beta]) }], height: 110, xLabel: "sweep", step: true });
}

// ------------------------------------------------------------------ results
async function sample(btn) {
  if (!S.connected) return showError(new Error("connect to a chip first"));
  stopPlay();
  await guarded(btn, async () => {
    const r = await api("/api/sample", { sweeps: +$("#smp-sweeps").value, beta: +$("#smp-beta").value });
    const tiles = clear($("#smp-tiles"));
    const tile = (k, v) => { const d = h("div", { class: "tile" }); d.appendChild(h("div", { class: "v" }, String(v))); d.appendChild(h("div", { class: "k" }, k)); tiles.appendChild(d); };
    if (r.tv !== undefined) { tile("total variation vs exact", r.tv); tile("KL (nats)", r.kl); }
    tile("sweeps", r.sweeps.toLocaleString()); tile("sweeps / s (incl. link)", r.rate.toLocaleString());
    const n = r.n;
    const lab = (hx) => (n <= 16 ? bitString(hx, Math.min(n, r.hist_bits)) : hx);
    const series = [{ name: "chip", color: css("--series-1"), vals: r.top.map((t) => t.chip) }];
    if (r.top.length && r.top[0].exact !== undefined) series.push({ name: "exact Boltzmann", color: css("--series-2"), vals: r.top.map((t) => t.exact) });
    barChart($("#smp-hist"), {
      cats: r.top.map((t) => t.state), series, height: 260, catLabel: lab, yFmt: (v) => fmt(v, 3),
      tipRows: (i) => {
        const t = r.top[i];
        const rows = [[`state ${lab(t.state)} (bits 0→${Math.min(n, r.hist_bits) - 1})`], ["chip", fmt(t.chip, 4)]];
        if (t.exact !== undefined) rows.push(["exact", fmt(t.exact, 4)], ["energy", fmt(t.energy, 4)]);
        return rows;
      },
    });
    if (r.note) $("#smp-hist").appendChild(h("p", { class: "hint" }, r.note));
    barChart($("#smp-mag"), {
      cats: r.mag.map((_, i) => (S.prog.names[i] || String(i))), series: [{ name: "⟨s_i⟩", color: css("--pos"), vals: r.mag }],
      diverging: true, legend: false, height: 180, yFmt: (v) => fmt(v, 2),
    });
    await stepOnce();
  });
}

async function solve(btn) {
  if (!S.connected) return showError(new Error("connect to a chip first"));
  stopPlay();
  await guarded(btn, async () => {
    const r = await api("/api/solve", { runs: +$("#sol-runs").value });
    const out = clear($("#sol-out"));
    out.appendChild(h("p", {}, `${r.hits}/${r.runs} anneals (${r.sweeps} sweeps each) reached the lowest energy found.`));
    const best = h("div", { class: "now" });
    for (const l of r.best) best.appendChild(h("div", {}, l));
    out.appendChild(best);
    const t = out.appendChild(h("table", { class: "tbl" }));
    const hr = t.appendChild(h("tr"));
    for (const k of ["runs", "energy", "state"]) hr.appendChild(h("th", {}, k));
    for (const s of r.states) {
      const tr = t.appendChild(h("tr"));
      tr.appendChild(h("td", { class: "num" }, s.count));
      tr.appendChild(h("td", { class: "num" }, fmt(s.energy, 4)));
      tr.appendChild(h("td", { class: "mono" }, s.lines.slice(1).join("  |  ")));
    }
    renderGraph(r.states[0].state);
  });
}

// ------------------------------------------------------------------ wire + compare
const CMD = { P: "ping", W: "weight J", H: "bias h", O: "visit order", T: "β table", B: "schedule", Z: "restart schedule",
  X: "reseed", R: "run sweeps", S: "stream sweeps", C: "clear readout", D: "dump histogram", U: "dump counters",
  Q: "read spins", E: "neighbour word", M: "node word", G: "rounds", Y: "noise options", L: "ladder β", N: "raw noise" };
async function refreshWire() {
  if (!S.connected) return;
  const rows = await api("/api/wire?n=300");
  const t = clear($("#wire-table"));
  const hr = t.appendChild(h("tr"));
  for (const k of ["#", "cmd", "meaning", "sent", "reply"]) hr.appendChild(h("th", {}, k));
  rows.slice().reverse().forEach((e, k) => {
    const tr = t.appendChild(h("tr"));
    tr.appendChild(h("td", { class: "num" }, rows.length - k));
    tr.appendChild(h("td", {}, e.cmd));
    tr.appendChild(h("td", {}, CMD[e.cmd] || "?"));
    tr.appendChild(h("td", {}, e.tx.match(/../g).join(" ")));
    tr.appendChild(h("td", {}, `${e.nrx} B: ${(e.rx.match(/../g) || []).join(" ")}${e.nrx > 48 ? " …" : ""}`));
  });
}
async function compare(btn) {
  const out = clear($("#cmp-out"));
  out.appendChild(h("p", { class: "muted" }, "running the session on both engines (the Verilog build may take a few seconds the first time)…"));
  await guarded(btn, async () => {
    const r = await api("/api/compare", { chip: S.chip.id });
    clear(out);
    out.appendChild(h("p", { class: r.equal ? "ok-text" : "bad-text" },
      r.equal ? `Identical: ${r.commands} commands, every reply byte equal.` : `Different at command #${r.first_mismatch.index} (${r.first_mismatch.cmd}).`));
    out.appendChild(h("p", { class: "muted" }, `model ${r.seconds.model} s · RTL ${r.seconds.rtl} s · ${r.note}`));
    const t = out.appendChild(h("table", { class: "tbl" }));
    const hr = t.appendChild(h("tr"));
    for (const k of ["cmd", "meaning", "count", "reply bytes", "equal"]) hr.appendChild(h("th", {}, k));
    for (const row of r.rows) {
      const tr = t.appendChild(h("tr"));
      tr.appendChild(h("td", {}, row.cmd)); tr.appendChild(h("td", {}, CMD[row.cmd] || ""));
      tr.appendChild(h("td", { class: "num" }, row.count)); tr.appendChild(h("td", { class: "num" }, row.bytes.toLocaleString()));
      tr.appendChild(h("td", { class: row.equal === row.count ? "ok-text" : "bad-text" }, `${row.equal}/${row.count}`));
    }
  });
}

// ------------------------------------------------------------------ p-modes
function renderPmGraph(xs) {
  const info = S.pm;
  const box = $("#graph");
  let svg = box.querySelector("svg");
  const W = 640, H = 440, cx0 = 320, cy0 = 220, R = 140;
  const pos = info.names.map((_, i) => [cx0 + R * Math.cos(Math.PI / 2 + i * 2 * Math.PI / info.m), cy0 - R * Math.sin(Math.PI / 2 + i * 2 * Math.PI / info.m)]);
  if (!svg || svg.dataset.prog !== "pmode") {
    clear(box);
    svg = sv("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "p-mode sites" }, box);
    svg.dataset.prog = "pmode";
    for (let i = 0; i < info.m; i++) for (let j = i + 1; j < info.m; j++) {
      const a = info.A[i][j];
      if (a) sv("line", { x1: pos[i][0], y1: pos[i][1], x2: pos[j][0], y2: pos[j][1], class: "e" + (a < 0 ? " neg" : ""), "stroke-width": 1 + 2 * Math.abs(a) }, svg);
    }
    for (let i = 0; i < info.m; i++) {
      sv("circle", { cx: pos[i][0], cy: pos[i][1], r: 34, class: "n", id: "n" + i }, svg);
      svText(svg, pos[i][0], pos[i][1] - 4, info.names[i], { class: "nl", "text-anchor": "middle", fill: "#fff", "font-weight": 700 });
      svText(svg, pos[i][0], pos[i][1] + 12, "", { class: "nl", id: "t" + i, "text-anchor": "middle", fill: "#fff" });
      svText(svg, pos[i][0], pos[i][1] + 52, `h = ${info.h[i]}`, { class: "nl", "text-anchor": "middle" });
    }
    const lg = clear($("#graph-legend"));
    for (const [c, l] of [[css("--pos"), "x > 0"], [css("--neg"), "x < 0 (opacity ∝ |x|)"]]) { const s = h("span"); const sw = h("span", { class: "sw" }); sw.style.background = c; s.appendChild(sw); s.appendChild(document.createTextNode(l)); lg.appendChild(s); }
    for (const [cls, l] of [["ln", "A_ij > 0"], ["ln dash", "A_ij < 0"]]) { const s = h("span"); s.appendChild(h("span", { class: cls })); s.appendChild(document.createTextNode(l)); lg.appendChild(s); }
  }
  xs.forEach((x, i) => {
    const c = svg.querySelector("#n" + i);
    c.setAttribute("fill", x >= 0 ? css("--pos") : css("--neg"));
    c.setAttribute("fill-opacity", String(0.35 + 0.65 * Math.min(1, Math.abs(x) / 1.5)));
    svg.querySelector("#t" + i).textContent = (x >= 0 ? "+" : "") + x.toFixed(2);
  });
}
async function pmStep(k) {
  const r = await api("/api/pmode/step", { steps: k });
  for (const p of r.traj) S.pmTraj.push(p);
  if (S.pmTraj.length > 1200) S.pmTraj.splice(0, S.pmTraj.length - 1200);
  S.pmLast = r;
  const last = r.traj[r.traj.length - 1];
  renderPmGraph(last.x);
  renderLeds(r.leds.concat([0]).slice(0, 5), S.pmPlaying);
  const now = clear($("#now"));
  now.appendChild(h("div", {}, `step ${last.t} · ` + last.x.map((x, i) => `x${i} = ${x.toFixed(3)}`).join("  ")));
  now.appendChild(h("div", {}, `λ = ${S.pm.lam} · ε = ${S.pm.eps} · T = ${S.pm.T}`));
  renderPmCharts();
}
function renderPmCharts() {
  if (!S.pm) return;
  const cols = ["--series-1", "--series-2", "--series-3", "--series-4"].map(css);
  lineChart($("#chart-energy"), {
    series: S.pm.names.map((nm, i) => ({ name: nm, color: cols[i], pts: S.pmTraj.map((p) => [p.t, p.x[i]]) })),
    xLabel: "step", endLabels: true, height: 240, yFmt: (v) => fmt(v, 2),
  });
  const r = S.pmLast;
  if (!r) return;
  const box = clear($("#pm-hists"));
  const mids = r.edges.slice(0, -1).map((e, k) => (e + r.edges[k + 1]) / 2);
  S.pm.names.forEach((nm, i) => {
    const tot = r.hist[i].reduce((a, b) => a + b, 0) || 1;
    const d = h("div");
    d.appendChild(h("h3", {}, `${nm}: ${tot.toLocaleString()} steps`));
    const c = h("div", { class: "chart" });
    d.appendChild(c);
    box.appendChild(d);
    barChart(c, {
      cats: mids.map((m) => m.toFixed(2)), catLabel: () => "",
      series: [{ name: "chip histogram", color: css("--series-1"), vals: r.hist[i].map((v) => v / tot) }],
      overlay: { name: "exact e^(−V/T)", color: css("--series-2"), vals: r.exact[i] }, height: 220,
      tipRows: (k) => [[`x ∈ [${r.edges[k].toFixed(2)}, ${r.edges[k + 1].toFixed(2)})`], ["chip", fmt(r.hist[i][k] / tot, 4)], ["exact", fmt(r.exact[i][k], 4)]],
    });
  });
}
async function pmLoop() {
  while (S.pmPlaying) {
    try { await pmStep(100); } catch (e) { showError(e); stopPlay(); break; }
    await new Promise((res) => setTimeout(res, 30));
  }
}

// ------------------------------------------------------------------ circuit editor
const N_IN = { AND: 2, OR: 2, XOR: 2, NOT: 1, COPY: 1 };
const E = { inputs: [], gates: [], clamp: {}, tool: "AND", pick: [], selGate: null };

function stripAux(g) { return g[0] === "XOR" && g.length === 5 && String(g[4]).endsWith("_aux") ? g.slice(0, 4) : g.slice(); }
function edWires() {
  const out = [];
  const add = (w) => { if (!out.includes(w)) out.push(w); };
  E.inputs.forEach(add);
  for (const g of E.gates) g.slice(1).forEach(add);
  return out;
}
function edDriver() { const d = {}; E.gates.forEach((g, k) => { d[g[1 + N_IN[g[0]]]] = k; }); return d; }
function edIO() {
  const drv = edDriver(), used = new Set(E.gates.flatMap((g) => g.slice(1, 1 + N_IN[g[0]])));
  const wires = edWires();
  return { inputs: wires.filter((w) => drv[w] === undefined), outputs: wires.filter((w) => drv[w] !== undefined && !used.has(w)) };
}
function edShow() {
  const show = {};
  for (const part of $("#ed-show").value.split(";")) {
    const m = part.split("=");
    if (m.length !== 2 || !m[0].trim()) continue;
    show[m[0].trim()] = m[1].split(",").map((x) => x.trim()).filter(Boolean);
  }
  return show;
}
function edValidate() {
  const errs = [], wires = new Set(edWires()), drv = {};
  E.gates.forEach((g, k) => {
    if (!(g[0] in N_IN)) errs.push(`gate ${k + 1}: unknown type ${g[0]}`);
    else if (g.length !== 2 + N_IN[g[0]]) errs.push(`gate ${k + 1} (${g[0]}) needs ${N_IN[g[0]]} input(s) and 1 output`);
    else {
      const o = g[1 + N_IN[g[0]]];
      if (drv[o] !== undefined) errs.push(`wire ${o} is driven by two gates`);
      drv[o] = k;
      if (g.slice(1, 1 + N_IN[g[0]]).includes(o)) errs.push(`gate ${k + 1} feeds its own output`);
    }
  });
  for (const [g, ws] of Object.entries(edShow())) for (const w of ws) if (!wires.has(w)) errs.push(`group ${g}: no wire ${w}`);
  if (!E.gates.length) errs.push("add at least one gate");
  return errs;
}
function edSpec() {
  const io = edIO(), wires = new Set(edWires());
  const clamp = {};
  for (const [w, b] of Object.entries(E.clamp)) if (wires.has(w)) clamp[w] = b;
  return { type: "circuit", inputs: io.inputs, outputs: io.outputs, gates: E.gates.map((g) => g.slice()), clamp,
    show: edShow(), beta: +$("#ed-beta").value || 2 };
}
function edLoadSpec(spec, name) {
  if (!spec || spec.type !== "circuit") throw new Error("the editor opens circuit programs only (\"type\": \"circuit\")");
  E.inputs = (spec.inputs || []).slice();
  E.gates = (spec.gates || []).map(stripAux);
  E.clamp = { ...(spec.clamp || {}) };
  E.pick = []; E.selGate = null;
  $("#ed-show").value = Object.entries(spec.show || {}).map(([g, ws]) => `${g}=${ws.join(",")}`).join("; ");
  $("#ed-beta").value = spec.beta || 2;
  $("#ed-name").value = name || "circuit";
  renderEditor();
}
function edFromProgram() {
  const p = S.prog;
  if (!p || !p.gates || !p.gates.length) throw new Error("the loaded program is not a circuit with a gate list");
  const clamp = {};
  for (const [i, b] of Object.entries(p.pins)) { const w = p.names[+i]; if (!w.endsWith("_aux")) clamp[w] = b; }
  edLoadSpec({ type: "circuit", inputs: p.ins, gates: p.gates, clamp, show: p.show, beta: p.beta }, p.name);
}
function edFreshName(base) {
  const wires = new Set(edWires());
  for (let k = 1; ; k++) if (!wires.has(`${base}${k}`)) return `${base}${k}`;
}
function edValidName(w) { return /^[A-Za-z_][A-Za-z0-9_]*$/.test(w) && !w.endsWith("_aux"); }

function edClickWire(w) {
  if (E.pick.includes(w)) { E.pick = E.pick.filter((x) => x !== w); renderEditor(); return; }
  E.pick.push(w);
  if (E.pick.length === N_IN[E.tool]) {
    let out = $("#ed-outname").value.trim() || edFreshName("w");
    if (!edValidName(out)) { showError(new Error(`bad wire name ${out}`)); E.pick = []; renderEditor(); return; }
    if (edWires().includes(out)) { showError(new Error(`wire ${out} exists already: pick another output name`)); E.pick = []; renderEditor(); return; }
    E.gates.push([E.tool, ...E.pick, out]);
    E.pick = [];
    $("#ed-outname").value = "";
  }
  renderEditor();
}
function edRename(w) {
  const nw = (prompt(`Rename wire ${w} to:`, w) || "").trim();
  if (!nw || nw === w) return;
  if (!edValidName(nw) || edWires().includes(nw)) return showError(new Error(`cannot rename to ${nw} (invalid or taken)`));
  E.inputs = E.inputs.map((x) => (x === w ? nw : x));
  E.gates = E.gates.map((g) => g.map((x, k) => (k && x === w ? nw : x)));
  if (w in E.clamp) { E.clamp[nw] = E.clamp[w]; delete E.clamp[w]; }
  $("#ed-show").value = $("#ed-show").value.replace(new RegExp(`(^|[=,\\s])${w}(?=$|[,;\\s])`, "g"), `$1${nw}`);
  renderEditor();
}
function edDeleteGate() {
  if (E.selGate === null) return;
  const g = E.gates[E.selGate];
  const out = g[1 + N_IN[g[0]]];
  E.gates.splice(E.selGate, 1);
  // keep the gate's input wires and, if something still reads it, its output wire (as an input)
  for (const w of g.slice(1, 1 + N_IN[g[0]])) if (!edWires().includes(w)) E.inputs.push(w);
  if (edWires().includes(out) && !E.inputs.includes(out)) E.inputs.push(out);
  E.selGate = null;
  renderEditor();
}

function renderEditor() {
  const kinds = clear($("#ed-kinds"));
  for (const k of Object.keys(N_IN)) {
    const lab = h("label", { class: E.tool === k ? "on" : "" });
    const r = h("input", { type: "radio", name: "edkind", value: k });
    r.checked = E.tool === k;
    r.addEventListener("change", () => {
      E.tool = k;
      // with a gate selected, a type of the same arity changes that gate
      if (E.selGate !== null && N_IN[E.gates[E.selGate][0]] === N_IN[k]) E.gates[E.selGate][0] = k;
      E.pick = [];
      renderEditor();
    });
    lab.appendChild(r); lab.appendChild(document.createTextNode(` ${k} (${N_IN[k]} in)`));
    kinds.appendChild(lab);
  }
  $("#ed-del").disabled = E.selGate === null;

  const box = clear($("#ed-svg"));
  const wires = edWires(), drv = edDriver();
  const col = {}, gcol = {};
  const wcol = (w, depth = 0) => {
    if (col[w] !== undefined) return col[w];
    if (drv[w] === undefined || depth > 200) return (col[w] = 0);
    const g = E.gates[drv[w]];
    const c = 1 + Math.max(0, ...g.slice(1, 1 + N_IN[g[0]]).map((x) => wcol(x, depth + 1)));
    gcol[drv[w]] = c;
    return (col[w] = c);
  };
  wires.forEach((w) => wcol(w));
  const rows = {}, wy = {};
  for (const w of wires) { const c = col[w]; rows[c] = (rows[c] || 0) + 1; wy[w] = 36 + (rows[c] - 1) * 54; }
  const maxCol = Math.max(0, ...Object.values(col)), maxRows = Math.max(1, ...Object.values(rows));
  const CW = 160, W = 80 + maxCol * CW + 110, H = 30 + maxRows * 54;
  const wx = (w) => 40 + col[w] * CW;
  const svg = sv("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Circuit schematic" }, box);
  E.gates.forEach((g, k) => {
    const n = N_IN[g[0]];
    if (n === undefined) return;
    const out = g[1 + n], gx = 40 + (gcol[k] - 1) * CW + 62, gy = wy[out];
    for (const x of g.slice(1, 1 + n)) {
      const x0 = wx(x) + 8, y0 = wy[x];
      sv("path", { d: `M${x0},${y0} C${x0 + 30},${y0} ${gx - 30},${gy} ${gx},${gy}`, class: "w" }, svg);
    }
    sv("path", { d: `M${gx + 52},${gy} L${wx(out) - 8},${gy}`, class: "w" }, svg);
    const gg = sv("g", { class: "gate" + (E.selGate === k ? " sel" : "") }, svg);
    sv("rect", { x: gx, y: gy - 14, width: 52, height: 28, rx: 6 }, gg);
    svText(gg, gx + 26, gy + 4, g[0], { "text-anchor": "middle" });
    gg.addEventListener("click", () => { E.selGate = E.selGate === k ? null : k; renderEditor(); });
  });
  const io = edIO();
  for (const w of wires) {
    const g = sv("g", { class: "wire" + (E.pick.includes(w) ? " pick" : "") }, svg);
    const cl = E.clamp[w];
    const fill = cl === 1 ? css("--pos") : cl === 0 ? css("--neg") : css("--muted");
    sv("circle", { cx: wx(w), cy: wy[w], r: 8, fill }, g);
    const role = io.inputs.includes(w) ? " (in)" : io.outputs.includes(w) ? " (out)" : "";
    svText(g, wx(w), wy[w] - 13, w + role, { "text-anchor": "middle" });
    if (cl !== undefined) svText(g, wx(w) + 12, wy[w] + 4, `= ${cl}`, { class: "cl" });
    g.addEventListener("click", () => edClickWire(w));
    g.addEventListener("dblclick", (e) => { e.preventDefault(); edRename(w); });
    g.addEventListener("contextmenu", (e) => {
      e.preventDefault();
      const v = E.clamp[w];
      if (v === undefined) E.clamp[w] = 1; else if (v === 1) E.clamp[w] = 0; else delete E.clamp[w];
      renderEditor();
    });
  }
  if (!wires.length) svText(svg, 20, 40, "Empty: add input wires, then gates.", { class: "lab" });
  const nx = E.gates.filter((g) => g[0] === "XOR").length;
  const nb = wires.length + nx;
  const chipN = S.chip && S.chip.params ? S.chip.params.N : null;
  const errs = edValidate();
  const info = $("#ed-info");
  info.textContent = `${nb} p-bits (${wires.length} wires + ${nx} XOR aux)` + (chipN ? ` · chip has ${chipN}` : "") +
    ` · inputs ${io.inputs.join(", ") || "–"} · outputs ${io.outputs.join(", ") || "–"}` +
    (E.pick.length ? ` · picking ${E.tool} inputs: ${E.pick.join(", ")} (${N_IN[E.tool] - E.pick.length} more)` : "") +
    (errs.length ? ` · ✕ ${errs.join("; ")}` : "") + (chipN && nb > chipN ? " · ✕ too many p-bits for this chip" : "");
  info.className = errs.length || (chipN && nb > chipN) ? "bad" : "muted";
  $("#ed-load").disabled = !!errs.length || !!(chipN && nb > chipN);
}

function setupEditor() {
  $("#ed-add-input").addEventListener("click", () => {
    const w = (prompt("Name of the new input wire:", edFreshName("x")) || "").trim();
    if (!w) return;
    if (!edValidName(w) || edWires().includes(w)) return showError(new Error(`cannot add wire ${w} (invalid or taken)`));
    E.inputs.push(w);
    renderEditor();
  });
  $("#ed-del").addEventListener("click", edDeleteGate);
  $("#ed-clear").addEventListener("click", () => { E.inputs = []; E.gates = []; E.clamp = {}; E.pick = []; E.selGate = null; $("#ed-show").value = ""; renderEditor(); });
  $("#ed-show").addEventListener("input", renderEditor);
  const open = (fn) => { try { showError(null); fn(); } catch (e) { showError(e); } };
  $("#ed-from-example").addEventListener("click", () => open(() => {
    const e = S.programs.examples.find((x) => x.name === $("#prog-example").value);
    edLoadSpec(e && e.spec, e && e.name);
  }));
  $("#ed-from-current").addEventListener("click", () => open(edFromProgram));
  $("#ed-from-json").addEventListener("click", () => open(() => edLoadSpec(JSON.parse($("#prog-json").value), $("#prog-example").value)));
  $("#ed-to-json").addEventListener("click", () => { $("#prog-json").value = JSON.stringify(edSpec(), null, 2); });
  $("#ed-load").addEventListener("click", (e) => {
    const spec = edSpec();
    $("#prog-json").value = JSON.stringify(spec, null, 2);
    loadProgram({ problem: spec, name: $("#ed-name").value.trim() || "circuit" }, e.target);
  });
  renderEditor();
}

// ------------------------------------------------------------------ learn tab (machine learning)
const ML = { list: [], demo: null, status: null, training: false, draw: null, label: 0 };
const GLYPHS = { T: "111010010", L: "100100111", X: "101010101", O: "111101111" };
const GLYPH_NAMES = Object.keys(GLYPHS);

function hex2rgb(x) { const v = parseInt(x.replace("#", ""), 16); return [(v >> 16) & 255, (v >> 8) & 255, v & 255]; }
function mixColor(a, b, t) {
  const A = hex2rgb(a), B = hex2rgb(b);
  return `rgb(${A.map((v, k) => Math.round(v + (B[k] - v) * t)).join(",")})`;
}
function divColor(t) {          // t in [0, 1]: 0 -> spin -1 pole, 0.5 -> neutral, 1 -> spin +1 pole
  const neg = css("--neg"), mid = css("--mid"), pos = css("--pos");
  return t < 0.5 ? mixColor(neg, mid, t * 2) : mixColor(mid, pos, (t - 0.5) * 2);
}

// pixel grid: values 1 / 0 / null (unknown) or probabilities (prob: true); onClick(k) makes it editable
function pixelGrid(box, { side, values, prob = false, onClick, size = 24, label, tipText }) {
  const fig = h("div", { class: "pixfig" });
  const wrap = h("div", { class: "pix" + (onClick ? " click" : "") + (size < 10 ? " thumb" : "") });
  const W = side * size;
  const show = (v) => (prob ? `${(100 * v).toFixed(1)}%` : String(v));
  if (side > 32 && !onClick) {          // 4096+ pixels: one canvas instead of an SVG rect per pixel
    const cv = h("canvas", { width: W, height: W, role: "img", "aria-label": label || "image" });
    wrap.appendChild(cv);
    const ctx = cv.getContext("2d");
    values.forEach((v, k) => { ctx.fillStyle = prob ? divColor(v) : v ? css("--pos") : css("--neg"); ctx.fillRect((k % side) * size, Math.floor(k / side) * size, size, size); });
    cv.addEventListener("pointermove", (e) => {
      const r = cv.getBoundingClientRect(), x = Math.floor((e.clientX - r.left) / size), y = Math.floor((e.clientY - r.top) / size);
      if (x >= 0 && y >= 0 && x < side && y < side) tip(e, tipText ? tipText(y * side + x) : [[`pixel ${y * side + x} (row ${y}, col ${x})`], [show(values[y * side + x]), prob ? "P(pixel = 1)" : "value"]]);
    });
    cv.addEventListener("pointerleave", untip);
    fig.appendChild(wrap);
    if (label) fig.appendChild(h("span", { class: "cap" }, label));
    box.appendChild(fig);
    return fig;
  }
  const svg = sv("svg", { width: W, height: W, viewBox: `0 0 ${W} ${W}`, role: "img", "aria-label": label || "image" }, wrap);
  values.forEach((v, k) => {
    const x = (k % side) * size, y = Math.floor(k / side) * size;
    const unk = v === null || v === undefined;
    const pad = size >= 10 ? 1 : 0;      // thumbnails: no gap, or the pixels shrink to dots
    const r = sv("rect", { x: x + pad, y: y + pad, width: size - 2 * pad, height: size - 2 * pad, rx: pad ? Math.min(4, size / 6) : 0, class: "px" + (unk ? " unk" : "") }, svg);
    if (!unk) r.setAttribute("fill", prob ? divColor(v) : v ? css("--pos") : css("--neg"));
    r.addEventListener("pointermove", (e) => tip(e, tipText ? tipText(k) : [[`pixel ${k} (row ${Math.floor(k / side)}, col ${k % side})`],
      [unk ? "unknown" : prob ? `${(100 * v).toFixed(1)}%` : String(v), prob ? "P(pixel = 1)" : "value"],
      ...(onClick ? [["click to change"]] : [])]));
    r.addEventListener("pointerleave", untip);
    if (onClick) r.addEventListener("click", () => onClick(k));
  });
  fig.appendChild(wrap);
  if (label) fig.appendChild(h("span", { class: "cap" }, label));
  box.appendChild(fig);
  return fig;
}
function probScale(box) {
  const d = h("div", { class: "scale" });
  d.appendChild(h("span", {}, "P(1) = 0"));
  const bar = h("span", { class: "bar" });
  bar.style.background = `linear-gradient(90deg, ${divColor(0)}, ${divColor(0.5)}, ${divColor(1)})`;
  d.appendChild(bar);
  d.appendChild(h("span", {}, "1 (blue = pixel on / spin +1)"));
  box.appendChild(d);
}

function heatmap(box, M, names, title, scaleMax) {
  clear(box);
  const n = M.length, cell = n > 13 ? 18 : 22, L = 44, T = 40;
  const W = L + n * cell + 4, H = T + n * cell + 4;
  const mx = scaleMax || Math.max(1e-9, ...M.flat().map(Math.abs));
  const svg = sv("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, class: "heat", role: "img", "aria-label": title }, box);
  names.forEach((nm, i) => {
    svText(svg, L - 4, T + i * cell + cell / 2 + 3, nm, { "text-anchor": "end" });
    const t = svText(svg, L + i * cell + cell / 2, T - 4, nm, { "text-anchor": "start" });
    t.setAttribute("transform", `rotate(-60 ${L + i * cell + cell / 2} ${T - 4})`);
  });
  M.forEach((row, i) => row.forEach((v, j) => {
    const r = sv("rect", { x: L + j * cell, y: T + i * cell, width: cell, height: cell, fill: i === j ? css("--page") : divColor(0.5 + v / mx / 2) }, svg);
    r.addEventListener("pointermove", (e) => tip(e, [[`${names[i]} — ${names[j]}`], [fmt(v, 4), "J"]]));
    r.addEventListener("pointerleave", untip);
  }));
  const d = h("div", { class: "scale" });
  d.appendChild(h("span", {}, `J = −${fmt(mx, 2)}`));
  const bar = h("span", { class: "bar" });
  bar.style.background = `linear-gradient(90deg, ${divColor(0)}, ${divColor(0.5)}, ${divColor(1)})`;
  d.appendChild(bar);
  d.appendChild(h("span", {}, `+${fmt(mx, 2)}  (red: disagree, blue: agree)`));
  box.appendChild(d);
}

// Which engine runs demo d: the connected chip if it is in d.chips; otherwise a click connects one
// (on the Python model, or when nothing is connected yet: the selected chip if it fits, else d.home).
function mlFits(d) {
  if (S.connected && d.chips.includes(S.chip.id)) return { ok: true, why: d.train ? `trains on the ${engWord()}` : `inference on the ${engWord()}` };
  const picked = document.querySelector('input[name="backend"]:checked').value;
  const target = !S.connected && d.chips.includes(S.chip.id) ? S.chip.id : d.home;
  const tc = S.chips.find((c) => c.id === target);
  const be = !S.connected && target === S.chip.id && tc.backends.includes(picked) ? picked : "model";
  if (!S.connected || S.backend === "model") {
    return { ok: true, switchTo: target, backend: be, why: `click: connects ${tc.title} (${target === "pysim" ? "Python" : be === "model" ? "Python model" : be})` };
  }
  return { ok: false, why: `needs ${d.needs}` };
}
function renderMlCards() {
  const box = clear($("#ml-demos"));
  for (const d of ML.list) {
    const f = mlFits(d);
    const b = h("button", { class: "chipcard", "aria-pressed": String(ML.demo === d.id) });
    b.disabled = !f.ok;
    b.appendChild(h("strong", {}, d.title));
    b.appendChild(h("span", { class: "meta" }, d.summary));
    b.appendChild(h("span", { class: "meta" }, `runs on: ${d.chips.map((c) => (S.chips.find((x) => x.id === c) || { title: c }).title).join(" · ")}`));
    b.appendChild(h("span", { class: "why" + (f.switchTo ? " switch" : "") }, f.why));
    b.addEventListener("click", () => mlOpen(d));
    box.appendChild(b);
  }
  $("#ml-engine").textContent = S.connected
    ? `Connected: ${S.chip.title} · ${S.chip.id === "pysim" ? "Python" : S.backend}. Cards for other chips connect them on the Python model.`
    : "Not connected: a demo card connects its chip (Python model) when clicked.";
}
async function mlOpen(d) {
  const f = mlFits(d);
  if (!f.ok) return;
  if (f.switchTo) {
    selectChip(f.switchTo);
    const r = document.querySelector(`input[name="backend"][value="${f.backend}"]`);
    if (r && !r.disabled) r.checked = true;
    showTab("learn");
    await connect();
    if (!S.connected) return;
  }
  await mlStart(d.id);
}

async function mlStart(id) {
  stopPlay();
  ML.training = false;
  await guarded(null, async () => {
    const r = await api("/api/ml/start", { demo: id });
    ML.demo = id;
    renderMlCards();
    const panel = clear($("#ml-panel"));
    if (id === "glyphs" || id === "bars3") {
      S.prog = null; S.lastState = null; clear($("#graph"));
      setStatus(`${r.engine} · training ${id}`, "ok");
      ML.status = r;
      ML.render = mlRenderStatus;
      mlTrainPanel(panel);
    } else if (id === "digits") {
      ML.status = r;
      ML.render = mlRenderDigits;
      await mlDigitsPanel(panel);
    } else if (id === "pmode") {
      ML.status = r;
      ML.render = mlRenderPmode;
      mlPmodePanel(panel);
    } else if (id === "hopfield") {
      setProgram(r.program);
      mlHopfieldPanel(panel, r);
    } else if (id === "mrf") {
      if (r.program) setProgram(r.program);
      mlMrfPanel(panel, r);
    }
  });
}

// ---------------------------------------------------------- training demos
function mlTrainPanel(panel) {
  const g = panel.appendChild(h("div", { class: "grid2" }));
  const left = g.appendChild(h("div", { class: "card" }));
  left.appendChild(h("h2", {}, ML.demo === "glyphs" ? "Train: glyph classifier" : "Train: bars & stripes"));
  left.appendChild(h("p", { class: "muted" }, ML.demo === "glyphs"
    ? "13 visible p-bits: 9 pixels + 4 one-hot labels (T, L, X, O). Each epoch: exact positive phase from the data, negative phase = 1000 free samples from the chip, then the new weights are uploaded."
    : "9 visible + 6 hidden p-bits. Each epoch: the chip samples the hidden units with every image clamped (positive phase) and runs free (negative phase)."));
  const row = left.appendChild(h("div", { class: "row" }));
  const bt = row.appendChild(h("button", { class: "primary", id: "ml-train" }, "Train"));
  row.appendChild(h("span", { id: "ml-epoch", class: "muted" }));
  const use = row.appendChild(h("button", { id: "ml-use" }, "Use trained model"));
  left.appendChild(h("div", { id: "ml-tiles", class: "tiles" }));
  left.appendChild(h("h3", {}, "KL(data ‖ model) per epoch, nats (exact, from the uploaded weights)"));
  left.appendChild(h("div", { id: "ml-kl", class: "chart" }));
  const right = g.appendChild(h("div", { class: "card" }));
  right.appendChild(h("h2", {}, "Learned couplings J (as programmed, Q3.4)"));
  right.appendChild(h("div", { id: "ml-heat", class: "scrollbox tall" }));
  panel.appendChild(h("div", { id: "ml-infer" }));
  bt.addEventListener("click", () => {
    ML.training = !ML.training;
    bt.textContent = ML.training ? "Pause" : "Train";
    if (ML.training) mlTrainLoop();
  });
  use.addEventListener("click", (e) => guarded(e.target, async () => {
    ML.training = false; bt.textContent = "Train";
    const r = await api("/api/ml/use", {});
    ML.status = r;
    setProgram(r.program);
    setStatus(`${S.backend} · trained ${ML.demo} is the program`, "ok");
    mlRenderStatus();
    mlInferPanel();
  }));
  mlRenderStatus();
}
async function mlTrainLoop() {
  while (ML.training) {
    try {
      ML.status = await api("/api/ml/train", { epochs: ML.demo === "digits" ? 2 : 5 });
      (ML.render || mlRenderStatus)();
      if (ML.status.epoch >= ML.status.target) { ML.training = false; $("#ml-train").textContent = "Train more"; }
    } catch (e) { showError(e); ML.training = false; $("#ml-train").textContent = "Train"; }
  }
}
function mlRenderStatus() {
  const st = ML.status;
  if (!st || !$("#ml-epoch")) return;
  $("#ml-epoch").textContent = `epoch ${st.epoch} / ${st.target} (suggested)`;
  const tiles = clear($("#ml-tiles"));
  const tile = (k, v) => { const d = h("div", { class: "tile" }); d.appendChild(h("div", { class: "v" }, v)); d.appendChild(h("div", { class: "k" }, k)); tiles.appendChild(d); };
  tile("KL now (nats)", fmt(st.kl[st.kl.length - 1], 3));
  tile("engine", st.engine);
  for (const [k, v] of Object.entries(st.metrics)) tile(k, `${(100 * v).toFixed(1)}%`);
  lineChart($("#ml-kl"), { series: [{ name: "KL", color: css("--series-1"), pts: st.kl.map((v, i) => [i, v]) }], xLabel: "epoch", yFmt: (v) => fmt(v, 2), height: 180 });
  heatmap($("#ml-heat"), st.J, st.names, "learned couplings");
}

function mlInferPanel() {
  const box = clear($("#ml-infer"));
  const g = box.appendChild(h("div", { class: "grid2" }));
  if (ML.demo === "glyphs") {
    // classify
    const a = g.appendChild(h("div", { class: "card" }));
    a.appendChild(h("h2", {}, "Classify: pin the image, read the label p-bits"));
    a.appendChild(h("p", { class: "muted" }, "Click pixels to draw, or start from a glyph and add noise. The chip samples the 4 label p-bits (and nothing else) with the 9 pixels pinned."));
    const pre = a.appendChild(h("div", { class: "row" }));
    ML.draw = GLYPHS.T.split("").map(Number);
    for (const nm of GLYPH_NAMES) {
      const b = pre.appendChild(h("button", {}, nm));
      b.addEventListener("click", () => { ML.draw = GLYPHS[nm].split("").map(Number); drawCls(); });
    }
    const nz = pre.appendChild(h("button", { class: "ghost" }, "+ flip a random pixel"));
    nz.addEventListener("click", () => { const k = Math.floor(Math.random() * 9); ML.draw[k] ^= 1; drawCls(); });
    const gridBox = a.appendChild(h("div", { class: "pixrow" }));
    const go = a.appendChild(h("button", { class: "primary" }, `Classify on the ${engWord()}`));
    a.appendChild(h("div", { id: "ml-cls-out", class: "chart" }));
    const drawCls = () => { clear(gridBox); pixelGrid(gridBox, { side: 3, values: ML.draw, size: 40, label: "input (click to toggle)", onClick: (k) => { ML.draw[k] ^= 1; drawCls(); } }); };
    drawCls();
    go.addEventListener("click", (e) => guarded(e.target, async () => {
      const pins = {};
      ML.draw.forEach((v, k) => { pins[k] = v; });
      const r = await api("/api/ml/infer", { pins, sweeps: 1500, beta: 1 });
      const out = clear($("#ml-cls-out"));
      const chip = r.marg.slice(9, 13), ex = r.marg_exact.slice(9, 13);
      const best = chip.indexOf(Math.max(...chip));
      out.appendChild(h("p", {}, `${engWord() === "chip" ? "Chip" : "Simulator"} says: ${GLYPH_NAMES[best]} (P(L_${GLYPH_NAMES[best]} = 1) = ${(100 * chip[best]).toFixed(1)}%, ${r.sweeps} samples after a ${r.warmup}-sweep warm-up)`));
      const c = h("div"); out.appendChild(c);
      barChart(c, { cats: GLYPH_NAMES, catLabel: (x) => `L_${x}`, height: 180, yFmt: (v) => fmt(v, 2),
        series: [{ name: `${engWord()} (sampled)`, color: css("--series-1"), vals: chip }, { name: "exact", color: css("--series-2"), vals: ex }] });
      showProgram(S.prog);
    }));
    // generate
    const b = g.appendChild(h("div", { class: "card" }));
    b.appendChild(h("h2", {}, "Generate: pin a label, sample the image"));
    const lr = b.appendChild(h("div", { class: "row" }));
    const gen = (c) => guarded(null, async () => {
      const pins = {};
      GLYPH_NAMES.forEach((_, k) => { pins[9 + k] = k === c ? 1 : 0; });
      const r = await api("/api/ml/infer", { pins, sweeps: 1500, beta: 1 });
      const out = clear($("#ml-gen-out"));
      const row = out.appendChild(h("div", { class: "pixrow" }));
      pixelGrid(row, { side: 3, values: r.marg.slice(0, 9), prob: true, size: 34, label: `${engWord()}: P(pixel = 1)` });
      pixelGrid(row, { side: 3, values: r.marg_exact.slice(0, 9), prob: true, size: 34, label: "exact" });
      probScale(out);
      out.appendChild(h("h3", {}, `Last 12 samples from the ${engWord()}`));
      const gal = out.appendChild(h("div", { class: "gallery" }));
      for (const st of r.recent) pixelGrid(gal, { side: 3, values: Array.from({ length: 9 }, (_, k) => bit(st, k)), size: 14 });
      showProgram(S.prog);
    });
    GLYPH_NAMES.forEach((nm, c) => { const bb = lr.appendChild(h("button", {}, `label ${nm}`)); bb.addEventListener("click", () => gen(c)); });
    b.appendChild(h("div", { id: "ml-gen-out" }));
  } else {
    const a = g.appendChild(h("div", { class: "card" }));
    a.appendChild(h("h2", {}, "Generate: free-running samples"));
    const go = a.appendChild(h("button", { class: "primary" }, "Sample 2000 images"));
    a.appendChild(h("div", { id: "ml-bars-gen" }));
    go.addEventListener("click", (e) => guarded(e.target, async () => {
      const r = await api("/api/ml/infer", { pins: {}, sweeps: 2000, beta: 1, top: 200 });
      const out = clear($("#ml-bars-gen"));
      const valid = new Set(barsImages().map((v) => v.join("")));
      const vis = (st) => Array.from({ length: 9 }, (_, k) => bit(st, k));
      const counts = {};
      let ok = 0, tot = 0;
      for (const t of r.top) { const key = vis(t.state).join(""); counts[key] = (counts[key] || 0) + t.count; }
      const allTop = Object.entries(counts).sort((x, y) => y[1] - x[1]);
      for (const [k, c] of allTop) { tot += c; if (valid.has(k)) ok += c; }
      out.appendChild(h("p", {}, `${ok} of ${tot} samples (${(100 * ok / Math.max(tot, 1)).toFixed(1)}%) are valid bars/stripes; ${allTop.length} distinct images seen. Most frequent (✕ = not a valid image):`));
      const gal = out.appendChild(h("div", { class: "gallery" }));
      for (const [k, c] of allTop.slice(0, 18)) pixelGrid(gal, { side: 3, values: k.split("").map(Number), size: 16, label: `${c}×${valid.has(k) ? "" : " ✕"}` });
      showProgram(S.prog);
    }));
    const b = g.appendChild(h("div", { class: "card" }));
    b.appendChild(h("h2", {}, "Complete: pin what you know"));
    b.appendChild(h("p", { class: "muted" }, "Click a pixel to cycle unknown → 1 → 0. Unknown pixels (dashed) are sampled by the chip."));
    ML.draw = [1, 0, 1, null, null, null, null, null, null];     // a 1 0 1 row can only be vertical stripes
    const gridBox = b.appendChild(h("div", { class: "pixrow" }));
    const drawC = () => { clear(gridBox); pixelGrid(gridBox, { side: 3, values: ML.draw, size: 40, label: "known pixels", onClick: (k) => { const v = ML.draw[k]; ML.draw[k] = v === null ? 1 : v === 1 ? 0 : null; drawC(); } }); };
    drawC();
    const go2 = b.appendChild(h("button", { class: "primary" }, `Complete on the ${engWord()}`));
    b.appendChild(h("div", { id: "ml-bars-comp" }));
    go2.addEventListener("click", (e) => guarded(e.target, async () => {
      const pins = {};
      ML.draw.forEach((v, k) => { if (v !== null) pins[k] = v; });
      const r = await api("/api/ml/infer", { pins, sweeps: 1500, beta: 1 });
      const out = clear($("#ml-bars-comp"));
      const row = out.appendChild(h("div", { class: "pixrow" }));
      pixelGrid(row, { side: 3, values: r.marg.slice(0, 9), prob: true, size: 34, label: `${engWord()}: P(pixel = 1)` });
      pixelGrid(row, { side: 3, values: r.marg_exact.slice(0, 9), prob: true, size: 34, label: "exact" });
      probScale(out);
      showProgram(S.prog);
    }));
  }
}
function barsImages() {
  const out = new Set();
  for (let m = 0; m < 8; m++) {
    const rows = [0, 1, 2].map((k) => (m >> k) & 1);
    out.add([0, 1, 2].flatMap((r) => [rows[r], rows[r], rows[r]]).join(""));
    out.add([0, 1, 2].flatMap(() => rows).join(""));
  }
  return [...out].map((s) => s.split("").map(Number));
}

// ---------------------------------------------------------- shared bits of the training panels
function trainControls(card) {
  const row = card.appendChild(h("div", { class: "row" }));
  const bt = row.appendChild(h("button", { class: "primary", id: "ml-train" }, "Train"));
  row.appendChild(h("span", { id: "ml-epoch", class: "muted" }));
  bt.addEventListener("click", () => {
    ML.training = !ML.training;
    bt.textContent = ML.training ? "Pause" : "Train";
    if (ML.training) mlTrainLoop();
  });
  card.appendChild(h("div", { id: "ml-tiles", class: "tiles" }));
}
function fillTiles(box, rows) {
  clear(box);
  for (const [k, v] of rows) {
    const d = h("div", { class: "tile" });
    d.appendChild(h("div", { class: "v" }, String(v)));
    d.appendChild(h("div", { class: "k" }, k));
    box.appendChild(d);
  }
}

// ---------------------------------------------------------- digits RBM (ideal simulator)
async function mlDigitsPanel(panel) {
  const g = panel.appendChild(h("div", { class: "grid2" }));
  const left = g.appendChild(h("div", { class: "card" }));
  left.appendChild(h("h2", {}, "Train: digits RBM (ideal simulator)"));
  left.appendChild(h("p", { class: "muted" }, "74 visible p-bits (64 pixels + 10 one-hot labels) and 100 hidden ones, coupled only across the two layers. Each epoch: 72 minibatches of 20 training images. Positive phase exact (the hidden units are independent given the image); negative phase from 100 persistent block-Gibbs chains (PCD-1). Accuracy: the label with the lowest free energy F(image, label), computed exactly."));
  trainControls(left);
  left.appendChild(h("h3", {}, ML.status.curve_title));
  left.appendChild(h("div", { id: "ml-kl", class: "chart" }));
  const right = g.appendChild(h("div", { class: "card" }));
  right.appendChild(h("h2", {}, "What the hidden units learned"));
  right.appendChild(h("p", { class: "muted" }, "Pixel weights of the 32 hidden units with the largest weights. Blue: the unit turns the pixel on; red: off."));
  right.appendChild(h("div", { id: "ml-filters", class: "gallery" }));
  panel.appendChild(h("div", { id: "ml-infer" }));
  mlRenderDigits();
  await mlDigitsInfer();
}
function mlRenderDigits() {
  const st = ML.status;
  if (!st || !$("#ml-epoch")) return;
  $("#ml-epoch").textContent = `epoch ${st.epoch} / ${st.target} (suggested)`;
  fillTiles($("#ml-tiles"), [...Object.entries(st.metrics).map(([k, v]) => [k, `${(100 * v).toFixed(1)}%`]),
    ["p-bits (visible + hidden)", `${st.nv} + ${st.nh}`], ["engine", st.engine]]);
  const cols = [css("--series-1"), css("--series-2")];
  lineChart($("#ml-kl"), { series: st.curves.map((c, k) => ({ name: c.name, color: cols[k], pts: c.values.map((v, i) => [i, v]) })),
    xLabel: "epoch", yFmt: (v) => `${Math.round(100 * v)}%`, height: 180 });
  const fb = clear($("#ml-filters"));
  st.filters.forEach((w, j) => {
    const mx = Math.max(1e-9, ...w.map(Math.abs));
    pixelGrid(fb, { side: 8, values: w.map((v) => 0.5 + v / mx / 2), prob: true, size: 6, tipText: (k) => [[`hidden unit ${j + 1}, pixel ${k}`], [fmt(w[k], 3), "weight"]] });
  });
}
async function mlDigitsInfer() {
  const box = clear($("#ml-infer"));
  const t = await api("/api/ml/digits", { action: "tests", k: 16, seed: 1 });
  ML.draw = t.images[0].slice();
  const g = box.appendChild(h("div", { class: "grid2" }));
  const a = g.appendChild(h("div", { class: "card" }));
  a.appendChild(h("h2", {}, "Classify: pin the image, sample the labels"));
  a.appendChild(h("p", { class: "muted" }, "Pick a held-out test digit (never trained on) or draw one. The simulator pins the 64 pixels and samples the label and hidden p-bits. Exact: the free-energy softmax over the 10 one-hot labels. Train first: an untrained model guesses."));
  const gal = a.appendChild(h("div", { class: "gallery" }));
  const gridBox = a.appendChild(h("div", { class: "pixrow" }));
  const drawD = () => { clear(gridBox); pixelGrid(gridBox, { side: 8, values: ML.draw, size: 26, label: "input (click to toggle)", onClick: (k) => { ML.draw[k] ^= 1; drawD(); } }); };
  t.images.forEach((im, k) => pixelGrid(gal, { side: 8, values: im, size: 6, label: `${t.labels[k]}`, onClick: () => { ML.draw = im.slice(); drawD(); },
    tipText: () => [[`test image ${t.index[k]}, label ${t.labels[k]}`], ["click to use it"]] }));
  drawD();
  const row = a.appendChild(h("div", { class: "row" }));
  const go = row.appendChild(h("button", { class: "primary" }, "Classify on the simulator"));
  const hide = row.appendChild(h("button", {}, "Hide the bottom half and complete"));
  const clr = row.appendChild(h("button", { class: "ghost" }, "Clear"));
  const out = a.appendChild(h("div", { id: "ml-dig-cls" }));
  clr.addEventListener("click", () => { ML.draw = Array(64).fill(0); drawD(); });
  const digits = [...Array(10).keys()];
  go.addEventListener("click", (e) => guarded(e.target, async () => {
    const r = await api("/api/ml/digits", { action: "classify", image: ML.draw, seed: Math.floor(Math.random() * 1e6) });
    clear(out);
    const best = r.exact.indexOf(Math.max(...r.exact));
    out.appendChild(h("p", {}, `Simulator says: ${best} (exact P = ${(100 * r.exact[best]).toFixed(1)}%; sampled P(L_${best} = 1) = ${(100 * r.sampled[best]).toFixed(1)}%)`));
    const c = out.appendChild(h("div", { class: "chart" }));
    barChart(c, { cats: digits, catLabel: (x) => String(x), height: 180, yFmt: (v) => fmt(v, 2),
      series: [{ name: "sampled P(L_c = 1)", color: css("--series-1"), vals: r.sampled }, { name: "exact P(label = c | image)", color: css("--series-2"), vals: r.exact }] });
  }));
  hide.addEventListener("click", (e) => guarded(e.target, async () => {
    const pins = {};
    for (let k = 0; k < 32; k++) pins[k] = ML.draw[k];
    const r = await api("/api/ml/digits", { action: "complete", pins, seed: Math.floor(Math.random() * 1e6) });
    clear(out);
    const pr = out.appendChild(h("div", { class: "pixrow" }));
    pixelGrid(pr, { side: 8, values: ML.draw.map((v, k) => (k < 32 ? v : null)), size: 18, label: "known: top half" });
    pixelGrid(pr, { side: 8, values: r.marg, prob: true, size: 18, label: "simulator: P(pixel = 1)" });
    pixelGrid(pr, { side: 8, values: r.completed, size: 18, label: "completed" });
    probScale(out);
    const best = r.exact.indexOf(Math.max(...r.exact));
    out.appendChild(h("p", {}, `The completed image reads as ${best} (exact P = ${(100 * r.exact[best]).toFixed(1)}%).`));
  }));
  const b = g.appendChild(h("div", { class: "card" }));
  b.appendChild(h("h2", {}, "Generate: pin a label, sample the pixels"));
  b.appendChild(h("p", { class: "muted" }, "32 chains of block Gibbs with the one-hot label pinned. Left: P(pixel = 1) over all samples; then the last state of 12 chains."));
  const lr = b.appendChild(h("div", { class: "row" }));
  const gout = b.appendChild(h("div"));
  for (const c of digits) {
    const bb = lr.appendChild(h("button", {}, String(c)));
    bb.addEventListener("click", (e) => guarded(e.target, async () => {
      const r = await api("/api/ml/digits", { action: "generate", label: c, seed: Math.floor(Math.random() * 1e6) });
      clear(gout);
      const pr = gout.appendChild(h("div", { class: "pixrow" }));
      pixelGrid(pr, { side: 8, values: r.marg, prob: true, size: 22, label: `label ${c}: P(pixel = 1)` });
      const gl = gout.appendChild(h("div", { class: "gallery" }));
      for (const im of r.samples) pixelGrid(gl, { side: 8, values: im, size: 7 });
      probScale(gout);
    }));
  }
}

// ---------------------------------------------------------- p-mode chip learns a p-mode circuit
function mlPmodePanel(panel) {
  const st = ML.status;
  const g = panel.appendChild(h("div", { class: "grid2" }));
  const left = g.appendChild(h("div", { class: "card" }));
  left.appendChild(h("h2", {}, "Train: a student p-mode chip copies a teacher"));
  left.appendChild(h("p", { class: "muted" }, `Teacher: the chain4 circuit, hidden from the student. Data: ${st.n_data.toLocaleString()} soft-spin samples from ${st.chains} copies of the teacher's bit-exact Langevin model. Each epoch the student (A = h = 0 at the start; same λ = ${st.lam}, ε = ${st.eps}, T = ${st.T}) runs ${st.steps} Langevin steps on ${st.chains} copies, then moves A and h by the moment gap ⟨tanh x_i tanh x_j⟩_data − ⟨…⟩_student. Its words are rounded to Q3.4 before every run, as the chip stores them.`));
  trainControls(left);
  left.appendChild(h("h3", {}, st.curve_title));
  left.appendChild(h("div", { id: "ml-kl", class: "chart" }));
  const right = g.appendChild(h("div", { class: "card" }));
  right.appendChild(h("h2", {}, "Couplings A: student vs teacher (same colour scale)"));
  const hr = right.appendChild(h("div", { class: "pixrow" }));
  const ha = hr.appendChild(h("div", { class: "pixfig" })), ht = hr.appendChild(h("div", { class: "pixfig" }));
  ha.appendChild(h("div", { id: "ml-heatA" })); ha.appendChild(h("span", { class: "cap" }, "student (learned)"));
  ht.appendChild(h("div", { id: "ml-heatT" })); ht.appendChild(h("span", { class: "cap" }, "teacher (true)"));
  right.appendChild(h("h3", {}, "Fields h"));
  right.appendChild(h("table", { id: "ml-htab", class: "tbl" }));
  const bot = panel.appendChild(h("div", { class: "card" }));
  bot.appendChild(h("h2", {}, "Marginal density of each x_i: teacher data vs student samples"));
  bot.appendChild(h("div", { id: "ml-pmhist", class: "multiples" }));
  mlRenderPmode();
}
function mlRenderPmode() {
  const st = ML.status;
  if (!st || !$("#ml-epoch")) return;
  $("#ml-epoch").textContent = `epoch ${st.epoch} / ${st.target} (suggested)`;
  fillTiles($("#ml-tiles"), [...Object.entries(st.metrics_raw), ["data samples", st.n_data.toLocaleString()], ["engine", st.engine]]);
  lineChart($("#ml-kl"), { series: [{ name: st.curves[0].name, color: css("--series-1"), pts: st.curves[0].values.map((v, i) => [i, v]) }],
    xLabel: "epoch", yFmt: (v) => fmt(v, 1), height: 180 });
  const mx = Math.max(1e-9, ...st.A.flat().map(Math.abs), ...st.A_true.flat().map(Math.abs));
  heatmap($("#ml-heatA"), st.A, st.names, "student couplings", mx);
  heatmap($("#ml-heatT"), st.A_true, st.names, "teacher couplings", mx);
  const t = clear($("#ml-htab"));
  const hr = t.appendChild(h("tr"));
  ["site", "student h", "teacher h"].forEach((k, i) => hr.appendChild(h("th", i ? { class: "num" } : {}, k)));
  st.names.forEach((nm, i) => {
    const r = t.appendChild(h("tr"));
    r.appendChild(h("td", {}, nm));
    r.appendChild(h("td", { class: "num" }, fmt(st.h[i], 4)));
    r.appendChild(h("td", { class: "num" }, fmt(st.h_true[i], 4)));
  });
  const box = clear($("#ml-pmhist"));
  const mid = st.edges.slice(0, -1).map((e, k) => (e + st.edges[k + 1]) / 2);
  st.names.forEach((nm, i) => {
    const d = box.appendChild(h("div"));
    d.appendChild(h("h3", {}, nm));
    const c = d.appendChild(h("div", { class: "chart" }));
    lineChart(c, { series: [{ name: "teacher (data)", color: css("--series-1"), pts: mid.map((x, k) => [x, st.data[i][k]]) },
      { name: "student", color: css("--series-2"), pts: mid.map((x, k) => [x, st.model[i][k]]) }], xLabel: nm, yFmt: (v) => fmt(v, 2), height: 140 });
  });
}

// ---------------------------------------------------------- Hopfield
function mlHopfieldPanel(panel, r) {
  const g = panel.appendChild(h("div", { class: "grid2" }));
  const a = g.appendChild(h("div", { class: "card" }));
  a.appendChild(h("h2", {}, "Stored memories (one-shot Hebbian: J = ⅛ Σ ξ ξᵀ)"));
  const st = a.appendChild(h("div", { class: "pixrow" }));
  for (const l of r.letters) pixelGrid(st, { side: 4, values: r.patterns[l], size: 22, label: l });
  const row = a.appendChild(h("div", { class: "row" }));
  const sel = h("select");
  for (const l of r.letters) sel.appendChild(h("option", { value: l }, l));
  const ls = h("label", {}, "letter"); ls.appendChild(sel); row.appendChild(ls);
  const nz = h("input", { type: "range", min: 0, max: 0.4, step: 0.05, value: 0.15 });
  const nl = h("label", {}, "noise"); nl.appendChild(nz); const nv = h("span", { class: "muted" }, "15%"); nl.appendChild(nv); row.appendChild(nl);
  nz.addEventListener("input", () => { nv.textContent = `${Math.round(100 * nz.value)}%`; });
  const tr = h("input", { type: "number", min: 1, max: 50, value: 10 });
  const tl = h("label", {}, "trials"); tl.appendChild(tr); row.appendChild(tl);
  const go = a.appendChild(h("button", { class: "primary" }, `Corrupt and recall on the ${engWord()}`));
  a.appendChild(h("p", { class: "hint" }, "Each trial flips each pixel with the noise probability, feeds the noisy image as soft evidence h = 0.5·y, warms up from β/8 and samples at β = 3. Click the noisy image to edit it, then 'Recall this input'."));
  const b = g.appendChild(h("div", { class: "card" }));
  b.appendChild(h("h2", {}, "Recall"));
  const out = b.appendChild(h("div"));
  let lastNoisy = null;
  const show = (res) => {
    clear(out);
    lastNoisy = res.noisy.slice();
    const row2 = out.appendChild(h("div", { class: "pixrow" }));
    pixelGrid(row2, { side: 4, values: res.stored, size: 26, label: `stored ${res.letter}` });
    const noisyFig = () => pixelGrid(row2, { side: 4, values: lastNoisy, size: 26, label: `input (${lastNoisy.filter((v, k) => v !== res.stored[k]).length} wrong px)`,
      onClick: (k) => { lastNoisy[k] ^= 1; const f = row2.children[1]; row2.removeChild(f); row2.insertBefore(noisyFig(), row2.children[1]); } });
    noisyFig();
    pixelGrid(row2, { side: 4, values: res.marg, prob: true, size: 26, label: `${engWord()} P(1): ${res.recalled_errors} wrong px` });
    pixelGrid(row2, { side: 4, values: res.marg_exact, prob: true, size: 26, label: "exact posterior" });
    probScale(out);
    const tiles = out.appendChild(h("div", { class: "tiles" }));
    const t = h("div", { class: "tile" }); t.appendChild(h("div", { class: "v" }, `${res.successes} / ${res.trials}`)); t.appendChild(h("div", { class: "k" }, "trials recalled exactly")); tiles.appendChild(t);
    const again = out.appendChild(h("button", {}, "Recall this input"));
    again.addEventListener("click", (e) => guarded(e.target, async () => show(await api("/api/ml/hopfield", { letter: sel.value, input: lastNoisy, trials: 1 }))));
    showProgram(S.prog);
  };
  go.addEventListener("click", (e) => guarded(e.target, async () => show(await api("/api/ml/hopfield",
    { letter: sel.value, noise: +nz.value, trials: +tr.value, seed: Math.floor(Math.random() * 1e6) }))));
}

// ---------------------------------------------------------- MRF denoising
function mlMrfPanel(panel, r) {
  const card = panel.appendChild(h("div", { class: "card" }));
  const simE = engWord() === "simulator";
  card.appendChild(h("h2", {}, simE ? "Image denoising with a Markov random field (ideal simulator, any size)" : "Image denoising with a Markov random field (256 p-bits)"));
  card.appendChild(h("p", { class: "muted" }, "Prior: neighbouring pixels agree (J on the 4-neighbour grid). Evidence: each noisy pixel biases its p-bit by h = ½ ln((1−p)/p)·y. " +
    (simE ? "The simulator samples the posterior P(image | noisy image) with 4 parallel checkerboard chains; the denoised pixel is the majority over the samples. Larger sides upsample the same test images." : "The chip samples the posterior P(image | noisy image); the denoised pixel is the majority over the samples (up-counters, command U).")));
  const row = card.appendChild(h("div", { class: "row" }));
  const sel = h("select");
  for (const im of r.images) sel.appendChild(h("option", { value: im }, im));
  const l1 = h("label", {}, "image"); l1.appendChild(sel); row.appendChild(l1);
  const sideSel = h("select");
  for (const sd of r.sides || [16]) sideSel.appendChild(h("option", { value: sd }, `${sd} × ${sd} (${(sd * sd).toLocaleString()} p-bits)`));
  if ((r.sides || [16]).length > 1) { const l0 = h("label", {}, "size"); l0.appendChild(sideSel); row.appendChild(l0); }
  const nz = h("input", { type: "range", min: 0.05, max: 0.3, step: 0.05, value: 0.15 });
  const nl = h("label", {}, "noise"); nl.appendChild(nz); const nv = h("span", { class: "muted" }, "15%"); nl.appendChild(nv); row.appendChild(nl);
  nz.addEventListener("input", () => { nv.textContent = `${Math.round(100 * nz.value)}%`; });
  const jj = h("input", { type: "number", min: 0, max: 2, step: 0.0625, value: 0.5 });
  const jl = h("label", {}, "J"); jl.appendChild(jj); row.appendChild(jl);
  const sw = h("input", { type: "number", min: 50, max: 5000, step: 50, value: 300 });
  const sl = h("label", {}, "sweeps"); sl.appendChild(sw); row.appendChild(sl);
  const go = row.appendChild(h("button", { class: "primary" }, `Denoise on the ${engWord()}`));
  card.appendChild(h("p", { class: "hint" }, "J ≈ 0.5 works best here; above the 2-D Ising critical coupling (≈ 0.44 at β = 1) the prior starts to wash out thin shapes (try J = 1 on the ring)."));
  const out = card.appendChild(h("div"));
  const show = (d) => {
    clear(out);
    const pr = out.appendChild(h("div", { class: "pixrow" }));
    const sd = d.side || 16, px = { 16: 13, 32: 8, 64: 4, 128: 2 }[sd] || 2;
    pixelGrid(pr, { side: sd, values: d.truth, size: px, label: "truth" });
    pixelGrid(pr, { side: sd, values: d.noisy, size: px, label: `noisy (${(100 * d.error_noisy).toFixed(1)}% wrong)` });
    pixelGrid(pr, { side: sd, values: d.marg, prob: true, size: px, label: `${engWord()}: P(pixel = 1)` });
    pixelGrid(pr, { side: sd, values: d.denoised, size: px, label: `denoised (${(100 * d.error_denoised).toFixed(1)}% wrong)` });
    probScale(out);
    const tiles = out.appendChild(h("div", { class: "tiles" }));
    for (const [k, v] of [["pixel error before", `${(100 * d.error_noisy).toFixed(1)}%`], ["pixel error after", `${(100 * d.error_denoised).toFixed(1)}%`],
      ["evidence h = ½ ln((1−p)/p)", fmt(d.lam, 3)], ["sweeps (after warm-up)", d.sweeps], ["p-bits", (sd * sd).toLocaleString()], ["seconds on this engine", d.seconds]]) {
      const t = h("div", { class: "tile" }); t.appendChild(h("div", { class: "v" }, String(v))); t.appendChild(h("div", { class: "k" }, k)); tiles.appendChild(t);
    }
    if (d.program) setProgram(d.program);
  };
  show(r);
  go.addEventListener("click", (e) => guarded(e.target, async () => show(await api("/api/ml/denoise",
    { image: sel.value, side: +sideSel.value, noise: +nz.value, J: +jj.value, sweeps: +sw.value, seed: Math.floor(Math.random() * 1e6) }))));
}

// ------------------------------------------------------------------ boot
function setupTabs() {
  for (const b of document.querySelectorAll(".tabs button")) {
    b.addEventListener("click", () => {
      for (const o of document.querySelectorAll(".tabs button")) o.setAttribute("aria-selected", String(o === b));
      for (const t of document.querySelectorAll(".tab")) t.hidden = t.id !== "tab-" + b.dataset.tab;
      if (b.dataset.tab === "arch") renderFpga();
      if (b.dataset.tab === "wire") refreshWire().catch(showError);
      if (b.dataset.tab === "run" || b.dataset.tab === "results") renderCharts();
      if (b.dataset.tab === "learn") renderMlCards();
    });
  }
}
async function refreshPorts() {
  const ports = await api("/api/ports");
  const sel = clear($("#port"));
  if (!ports.length) sel.appendChild(h("option", { value: "/dev/ttyUSB1" }, "/dev/ttyUSB1 (none detected)"));
  for (const p of ports) sel.appendChild(h("option", { value: p.device }, `${p.device} ${p.description || ""}`));
  const usb1 = ports.find((p) => p.device === "/dev/ttyUSB1");
  if (usb1) sel.value = usb1.device;
}

async function boot() {
  setupTabs();
  $("#theme").addEventListener("click", () => {
    const d = document.documentElement;
    const dark = d.dataset.theme ? d.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    d.dataset.theme = dark ? "light" : "dark";
    if (S.chip) { renderArch(); if (S.prog) { const s = S.samples.length ? S.samples[S.samples.length - 1].s : null; $("#graph").firstChild && clear($("#graph")); renderGraph(s); } renderCharts(); }
  });
  for (const r of document.querySelectorAll('input[name="backend"]')) {
    r.addEventListener("change", () => { $("#port-wrap").hidden = r.value !== "board" || !r.checked; if (r.value === "board" && r.checked) refreshPorts().catch(showError); });
  }
  $("#ports-refresh").addEventListener("click", () => refreshPorts().catch(showError));
  $("#connect").addEventListener("click", connect);
  $("#prog-example").addEventListener("change", showExample);
  $("#prog-load-json").addEventListener("click", (e) => {
    let spec;
    try { spec = JSON.parse($("#prog-json").value); } catch (err) { return showError(new Error("JSON: " + err.message)); }
    loadProgram({ problem: spec, name: $("#prog-example").value || "custom" }, e.target);
  });
  $("#prog-load-case").addEventListener("click", (e) => loadProgram({ case: $("#prog-case").value }, e.target));
  $("#run-play").addEventListener("click", togglePlay);
  $("#run-step").addEventListener("click", (e) => S.connected ? guarded(e.target, stepOnce) : showError(new Error("connect to a chip first")));
  $("#run-anneal").addEventListener("click", (e) => S.connected && guarded(e.target, async () => { await api("/api/restart", { mode: "anneal" }); S.samples = []; await stepOnce(); }));
  $("#run-const").addEventListener("click", (e) => S.connected && guarded(e.target, async () => { await api("/api/restart", { mode: "const", beta: +$("#run-beta").value }); S.samples = []; await stepOnce(); }));
  for (const r of document.querySelectorAll('input[name="mode"]')) {
    r.addEventListener("change", () => { S.mode = r.value; S.sel = null; if (S.prog) showProgram(S.prog); });
  }
  $("#reset-program").addEventListener("click", (e) => S.connected && guarded(e.target, async () => {
    S.sel = null; setProgram(await api("/api/reset_program", {})); await stepOnce();
  }));
  setupEditor();
  $("#smp-run").addEventListener("click", (e) => sample(e.target));
  $("#sol-run").addEventListener("click", (e) => solve(e.target));
  $("#wire-refresh").addEventListener("click", () => refreshWire().catch(showError));
  $("#cmp-run").addEventListener("click", (e) => compare(e.target));
  $("#pm-play").addEventListener("click", () => {
    if (!S.connected) return showError(new Error("connect to the p-mode chip first"));
    S.pmPlaying = !S.pmPlaying; $("#pm-play").textContent = S.pmPlaying ? "Pause" : "Play"; if (S.pmPlaying) pmLoop();
  });
  $("#pm-step").addEventListener("click", (e) => S.connected && guarded(e.target, () => pmStep(500)));
  $("#pm-reset").addEventListener("click", (e) => S.connected && guarded(e.target, async () => { S.pm = await api("/api/pmode/reset", {}); S.pmTraj = []; await pmStep(50); }));
  try {
    S.chips = await api("/api/chips");
    S.programs = await api("/api/programs");
    ML.list = await api("/api/ml/list");
    fillPrograms();
    // deep link: ?chip=factor6&backend=rtl&tab=run&play=1&example=qubo_small
    const q = new URLSearchParams(location.search);
    selectChip(S.chips.some((c) => c.id === q.get("chip")) ? q.get("chip") : S.chips[0].id);
    renderFpga();
    if (q.get("backend")) {
      const r = document.querySelector(`input[name="backend"][value="${q.get("backend")}"]`);
      if (r && !r.disabled) r.checked = true;
      await connect();
      if (q.get("example") && S.connected) await loadProgram({ example: q.get("example") });
      if (q.get("sample") && S.connected) await sample($("#smp-run"));
      if (q.get("solve") && S.connected) await solve($("#sol-run"));
      if (q.get("compare")) await compare($("#cmp-run"));
      if (q.get("mode") === "edit") document.querySelector('input[name="mode"][value="edit"]').click();
      for (const pv of (q.get("pin") || "").split(",").filter(Boolean)) {
        const [n, v] = pv.split(":");
        await liveEdit("/api/pin", { node: +n, value: v === "free" ? null : +v });
      }
      if (q.get("group") && S.connected) { const [g, v] = q.get("group").split(":"); await liveEdit("/api/pin_group", { group: g, value: +v }); }
      if (q.get("select") && S.prog) { S.sel = { type: "node", i: +q.get("select") }; showProgram(S.prog); }
      if (q.get("ml") && S.connected) {
        await mlStart(q.get("ml"));
        if (q.get("train") && ML.status) {          // train=<epochs>: run that many epochs, then use the model
          while (ML.status.epoch < +q.get("train")) { ML.status = await api("/api/ml/train", { epochs: 10 }); }
          mlRenderStatus();
          $("#ml-use").click();
          await new Promise((res) => setTimeout(res, 1500));
        }
      }
      if (q.get("play") && S.connected) (S.chip.id === "pmode_chain4" ? $("#pm-play") : $("#run-play")).click();
    }
    const ed = q.get("editor");
    if (ed) {
      try {
        if (ed === "current") edFromProgram();
        else { const e = S.programs.examples.find((x) => x.name === ed.replace(/^example:/, "")); edLoadSpec(e && e.spec, e && e.name); }
      } catch (err) { showError(err); }
    }
    const tab = document.querySelector(`.tabs button[data-tab="${q.get("tab")}"]`);
    if (tab) tab.click();
  } catch (e) { showError(e); }
}
boot();
