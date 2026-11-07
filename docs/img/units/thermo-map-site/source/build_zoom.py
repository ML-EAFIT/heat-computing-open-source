#!/usr/bin/env python3
"""Builds the zoomable map: every view nested inside its parent's box, semantic zoom from host PC to transistor."""
import json
import re
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_map as bm      # noqa: E402
import zoom_views as zv     # noqa: E402

TYPES = zv.build_all()

# SVG styling from the reference page, applied to the map canvas as well
dia_rules = "\n".join(l for l in bm.CSS.splitlines() if l.startswith(".dia"))
root_rules = bm.CSS.split("*{box-sizing:border-box}")[0]

JUMPS = [("top", "Board top (top.v)"), ("cmdfsm", "Command FSM"), ("dense", "Dense engine"),
         ("pbit", "p-bit decision"), ("sigrom", "Sigmoid ROM"), ("lfsr0", "LFSR noise lane 0"),
         ("sparse", "Sparse lanes"), ("temper", "Parallel tempering"), ("readout", "Readout counters"),
         ("pmode", "Langevin p-modes"), ("wmac", "weight_mac"), ("carry", "Carry chain"),
         ("lut4", "LUT4"), ("dff", "D flip-flop"), ("bsram", "Block SRAM"), ("sram6t", "6T SRAM cell"),
         ("inverter", "CMOS inverter"), ("nmos_x", "NMOS transistor")]

CSS = root_rules + r"""
*{box-sizing:border-box}
html,body{height:100%;margin:0}
body{background:var(--paper);color:var(--ink);font:15px/1.5 var(--font-body);overflow:hidden}
button,select{font:inherit;color:var(--ink)}
a{color:var(--signal)}
.app{height:100%;display:grid;grid-template-rows:auto minmax(0,1fr)}
.bar{display:flex;align-items:center;gap:14px;padding:8px 14px;border-bottom:1px solid var(--rule);background:var(--panel);flex-wrap:wrap}
.brand h1{font:600 1.15rem/1.1 var(--font-display);margin:0}
.brand span{font:500 .72rem var(--font-mono);color:var(--muted)}
.crumbs{display:flex;flex-wrap:wrap;gap:2px;align-items:center;flex:1 1 320px;min-width:0;font:500 .8rem var(--font-mono)}
.crumbs button{border:0;background:none;padding:3px 6px;border-radius:4px;cursor:pointer;color:var(--muted);white-space:nowrap}
.crumbs button:last-child{color:var(--ink);background:color-mix(in srgb,var(--signal) 12%,var(--panel))}
.crumbs button:hover{color:var(--signal)}
.crumbs i{color:var(--rule);font-style:normal}
.tools{display:flex;gap:6px;align-items:center;flex-wrap:wrap}
.tools button,.tools select{border:1px solid var(--rule);background:var(--paper);border-radius:5px;padding:5px 10px;cursor:pointer;min-height:32px}
.tools button:hover,.tools select:hover{border-color:var(--signal)}
.tools .navlink{border:1px solid var(--rule);background:var(--paper);border-radius:5px;padding:5px 10px;min-height:32px;display:inline-flex;align-items:center;text-decoration:none;color:var(--ink)}
.tools .navlink:hover{border-color:var(--signal);color:var(--signal)}
.tools .ic{width:34px;padding:5px 0;font:600 1rem var(--font-mono)}
button:focus-visible,select:focus-visible{outline:2px solid var(--signal);outline-offset:2px}
.main{display:grid;grid-template-columns:58px minmax(0,1fr) 400px;min-height:0}
.ladder{list-style:none;margin:0;padding:12px 6px;display:flex;flex-direction:column;gap:4px;border-right:1px solid var(--rule);background:var(--panel);overflow:auto}
.ladder button{width:100%;border:1px solid transparent;background:none;border-radius:5px;padding:6px 0;cursor:pointer;font:600 .78rem var(--font-mono);color:var(--muted);display:grid;gap:1px}
.ladder button small{font:500 .58rem var(--font-body);letter-spacing:.02em}
.ladder button.on{color:var(--paper);background:var(--ink)}
.ladder button:hover:not(.on){border-color:var(--rule);color:var(--ink)}
.canvas{position:relative;min-width:0;min-height:0;overflow:hidden;touch-action:none;cursor:grab;user-select:none;-webkit-user-select:none;
  background-color:var(--paper);background-image:radial-gradient(color-mix(in srgb,var(--muted) 30%,transparent) 1px,transparent 1.3px)}
.canvas.drag{cursor:grabbing}
.canvas.ptr{cursor:zoom-in}
#zm{position:absolute;inset:0;width:100%;height:100%;min-width:0;display:block}
.hint{position:absolute;left:14px;bottom:12px;font:500 .78rem var(--font-mono);color:var(--muted);background:var(--panel);border:1px solid var(--rule);border-radius:5px;padding:6px 10px;transition:opacity .6s;pointer-events:none}
.hint.gone{opacity:0}
.scale{position:absolute;right:14px;bottom:12px;font:500 .74rem var(--font-mono);color:var(--muted);background:var(--panel);border:1px solid var(--rule);border-radius:5px;padding:5px 9px;pointer-events:none;font-variant-numeric:tabular-nums}
.panel{border-left:1px solid var(--rule);background:var(--panel);overflow:auto;padding:16px 18px 40px;min-height:0}
.panel .lay{font:500 .74rem var(--font-mono);color:var(--muted);margin:0}
.panel h2{font:600 1.35rem/1.2 var(--font-display);margin:2px 0 4px;text-wrap:balance}
.panel .inst{font:500 .78rem var(--font-mono);color:var(--muted);margin:0 0 10px}
.panel h4{font:600 .82rem var(--font-display);letter-spacing:.02em;margin:18px 0 6px;text-transform:uppercase;color:var(--muted)}
.panel p{margin:8px 0;font-size:.92rem}
.kids{display:flex;flex-wrap:wrap;gap:5px}
.kids button{border:1px solid var(--rule);background:var(--paper);border-radius:4px;padding:3px 8px;font:500 .76rem var(--font-mono);cursor:pointer}
.kids button:hover{border-color:var(--signal);color:var(--signal)}
.kids button span{color:var(--muted)}
.eq{font-family:var(--font-mono);font-size:.82rem;background:var(--paper);border-left:3px solid var(--signal);padding:8px 12px;margin:10px 0}
.kv{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:1px;background:var(--rule);border:1px solid var(--rule);border-radius:5px;overflow:hidden;margin:10px 0}
.kv div{background:var(--paper);padding:7px 9px}
.kv div:last-child:nth-child(odd){grid-column:1/-1}
.kv dt{font-size:.7rem;color:var(--muted)}
.kv dd{margin:1px 0 0;font:600 .86rem var(--font-display);font-variant-numeric:tabular-nums}
.tscroll{overflow-x:auto;margin:8px 0;border:1px solid var(--rule);border-radius:5px;background:var(--paper)}
table{border-collapse:collapse;width:100%;font-size:.74rem;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:5px 7px;border-bottom:1px solid var(--rule);vertical-align:top}
th{font-weight:600;color:var(--muted);background:var(--panel)}
td.n,th.n{text-align:right}
td code{white-space:nowrap}
code{font-family:var(--font-mono);font-size:.86em}
tr.grp td{background:var(--panel);font:600 .72rem var(--font-mono);color:var(--muted)}
.meter{display:inline-block;width:46px;height:6px;background:color-mix(in srgb,var(--signal) 14%,var(--panel));border-radius:3px;vertical-align:middle;margin-left:4px;overflow:hidden}
.meter i{display:block;height:100%;background:var(--signal);border-radius:3px}
.finding{border-left:3px solid var(--heat);background:var(--paper);padding:8px 11px;margin:8px 0;font-size:.86rem}
.finding strong{display:block;font-family:var(--font-display)}
.mini{margin:10px 0;background:var(--paper);border:1px solid var(--rule);border-radius:5px;padding:6px;overflow-x:auto}
.mini .dia,.panel .dia{min-width:0}
.ptog{display:none}
@media (max-width:900px){
  .main{grid-template-columns:minmax(0,1fr);grid-template-rows:minmax(0,1fr) auto}
  .ladder{display:none}
  .panel{border-left:0;border-top:1px solid var(--rule);max-height:42vh}
  .panel.closed{display:none}
  .ptog{display:inline-block}
  .brand span{display:none}
  .hint{font-size:.68rem;right:14px;left:10px;bottom:8px}
  .scale{display:none}
}
""" + dia_rules + r"""
.zm .sheet{fill:var(--panel);stroke:var(--rule);stroke-width:1.2}
.zm .t.htl{font:600 17px var(--font-display);fill:currentColor}
.zm .t.hly{font:500 11px var(--font-mono);fill:var(--muted)}
.zm .st{fill:var(--panel)}
.zm .t.stt{font-size:11px;font-weight:600}
.zm .led{fill:color-mix(in srgb,var(--heat) 55%,var(--panel));stroke:currentColor;stroke-width:1}
.zm .wl{stroke:var(--signal);stroke-width:.8;opacity:.55}
.zm .bl{stroke:var(--muted);stroke-width:.6;opacity:.5}
.zm .curve{fill:none;stroke:var(--signal);stroke-width:2.2}
.zm .si{fill:color-mix(in srgb,var(--muted) 16%,var(--panel));stroke:var(--muted)}
.zm .well{fill:color-mix(in srgb,var(--signal) 12%,var(--panel));stroke:var(--muted);stroke-dasharray:4 3}
.zm .dopes{fill:color-mix(in srgb,var(--heat) 30%,var(--panel));stroke:var(--heat)}
.zm .ox{fill:color-mix(in srgb,var(--signal) 45%,var(--panel))}
.zm .poly{fill:color-mix(in srgb,var(--ink) 16%,var(--panel));stroke:currentColor}
.zm .metal{fill:color-mix(in srgb,var(--muted) 60%,var(--panel));stroke:currentColor}
.zm .carrier{fill:var(--signal)}
.zm .pbg{fill:var(--panel);fill-opacity:0;stroke:none}
.zm #hov{fill:color-mix(in srgb,var(--signal) 7%,transparent);stroke:var(--signal);stroke-width:2;rx:4}
.zm #hovt{font:600 12px var(--font-mono);fill:var(--signal);paint-order:stroke;stroke:var(--paper);stroke-width:4px}
.sr{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0)}
"""

JS = r"""
(function(){
'use strict';
const TYPES = window.__TYPES__;
const LAYERS = {6:'L6 · host + board',5:'L5 · bitstream top',4:'L4 · block / engine',3:'L3 · mechanism',2:'L2 · datapath element',1:'L1 · FPGA primitive',0:'L0 · transistor'};
const NS = 'http://www.w3.org/2000/svg';
const canvas = document.getElementById('canvas'), svg = document.getElementById('zm'), stage = document.getElementById('stage');
const hov = document.getElementById('hov'), hovt = document.getElementById('hovt');
const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
let vw = 800, vh = 600;
function mk(n, a){ const e = document.createElementNS(NS, n); for (const k in a) e.setAttribute(k, a[k]); return e; }

class Inst {
  constructor(tid, parent, idx){
    this.tid = tid; this.T = TYPES[tid]; this.parent = parent; this.idx = idx; this.kids = null; this.el = null;
    if (parent){
      const p = parent.T.portals[idx]; this.p = p; this.name = p.n;
      const pad = Math.min(p.w, p.h) * 0.06;
      const s = Math.min((p.w - 2*pad) / this.T.w, (p.h - 2*pad) / this.T.h);
      this.s = s; this.ox = p.x + (p.w - this.T.w*s)/2; this.oy = p.y + (p.h - this.T.h*s)/2;
      const a = parent.A; this.A = {S: a.S*s, X: a.X + a.S*this.ox, Y: a.Y + a.S*this.oy};
      this.depth = parent.depth + 1;
    } else { this.s = 1; this.ox = 0; this.oy = 0; this.name = this.T.title; this.A = {S:1, X:0, Y:0}; this.depth = 0; }
  }
  children(){ if (!this.kids) this.kids = this.T.portals.map((p, i) => new Inst(p.t, this, i)); return this.kids; }
  build(){
    if (this.el) return this.el;
    const g = mk('g', {'class':'inst'}); const own = mk('g', {'class':'own'}); own.innerHTML = this.T.svg; g.appendChild(own); this.own = own;
    for (const c of this.children()){
      const p = c.p, sg = mk('g', {'class':'slot'});
      const bg = mk('rect', {x:p.x+0.6, y:p.y+0.6, width:Math.max(p.w-1.2,0.1), height:Math.max(p.h-1.2,0.1), rx:2.5, 'class':'pbg'});
      const h = mk('g', {transform:`translate(${c.ox} ${c.oy}) scale(${c.s})`}); h.style.display = 'none';
      sg.appendChild(bg); sg.appendChild(h); g.appendChild(sg); c.bg = bg; c.holder = h;
    }
    this.el = g; return g;
  }
}
const world = new Inst('world', null, 0);
let cam = {k:1, x:0, y:0}, frame = null, focus = null;

function fitK(T){ return Math.min(vw*0.94/T.w, vh*0.9/T.h); }
function scr(I){ return {k: cam.k*I.A.S, x: cam.x + cam.k*I.A.X, y: cam.y + cam.k*I.A.Y}; }
function covers(I){ const s = scr(I); return s.x <= 0.5 && s.y <= 0.5 && s.x + s.k*I.T.w >= vw-0.5 && s.y + s.k*I.T.h >= vh-0.5; }
function locate(){ let F = world; for(;;){ let n = null; for (const c of F.children()) if (covers(c)) { n = c; break; } if (!n) return F; F = n; } }

function setFrame(F){
  if (frame === F) return;
  if (frame){
    if (frame.parent){ if (!frame.parent.el) frame.parent.build(); frame.holder.appendChild(frame.el); }
    else if (frame.el && frame.el.parentNode === stage) stage.removeChild(frame.el);
  }
  F.build(); stage.appendChild(F.el); frame = F;
}

// fade an instance's own drawing while one of its children takes over the screen
function ghost(I, k, x, y){
  let dom = 0;
  for (const c of I.children()){
    const kc = k*c.s, xc = x + k*c.ox, yc = y + k*c.oy, w = kc*c.T.w, h = kc*c.T.h;
    const ix = Math.max(0, Math.min(vw, xc + w) - Math.max(0, xc)), iy = Math.max(0, Math.min(vh, yc + h) - Math.max(0, yc));
    dom = Math.max(dom, ix*iy/(vw*vh));
  }
  let op = 1;
  if (dom >= 0.2){
    const r = k / fitK(I.T);
    const fd = 1 - 0.93*Math.min(1, Math.max(0, (dom - 0.25)/0.35));
    const fr = r <= 1.8 ? 1 : Math.max(0.07, 1 - (r - 1.8)/2.2);
    op = Math.min(fd, fr);
  }
  const v = op.toFixed(3);
  if (I.own.style.opacity !== v) I.own.style.opacity = v;
}
function lod(I, k, x, y, d){
  for (const c of I.children()){
    const kc = k*c.s, xc = x + k*c.ox, yc = y + k*c.oy, w = kc*c.T.w, h = kc*c.T.h;
    let a = 0;
    if (xc < vw && yc < vh && xc + w > 0 && yc + h > 0){ a = Math.min(1, Math.max(0, (Math.max(w, h*1.7) - 120) / 190)); }
    c.bg.style.fillOpacity = (a*0.97).toFixed(3);
    if (a > 0){
      if (!c.el) c.build();
      if (c.el.parentNode !== c.holder) c.holder.appendChild(c.el);
      c.holder.style.display = ''; c.holder.style.opacity = a.toFixed(3);
      ghost(c, kc, xc, yc);
      if (d < 12) lod(c, kc, xc, yc, d+1);
    } else c.holder.style.display = 'none';
  }
}

function render(){
  const F = locate(); setFrame(F);
  const s = scr(F);
  stage.setAttribute('transform', `matrix(${s.k} 0 0 ${s.k} ${s.x} ${s.y})`);
  ghost(F, s.k, s.x, s.y);
  lod(F, s.k, s.x, s.y, 0);
  const g = 22 * Math.pow(2, Math.log2(s.k) - Math.floor(Math.log2(s.k)));
  canvas.style.backgroundSize = `${g}px ${g}px`;
  canvas.style.backgroundPosition = `${((s.x % g) + g) % g}px ${((s.y % g) + g) % g}px`;
  updateUI();
}

// ------------------------------------------------------------ camera moves
function zoomAt(px, py, f){
  cam.x = px - (px - cam.x)*f; cam.y = py - (py - cam.y)*f; cam.k *= f;
  // cap the zoom relative to the deepest view under the cursor, so zooming toward a tiny
  // component keeps going while zooming into empty space stops at 6x that view's fit
  const at = deepestAt(px, py), r = at.s.k / fitK(at.I.T), rw = cam.k / fitK(world.T);
  let g = 1;
  if (r > 6) g = 6 / r;
  if (rw < 0.45) g = 0.45 / rw;
  if (g !== 1){ cam.x = px - (px - cam.x)*g; cam.y = py - (py - cam.y)*g; cam.k *= g; }
  clampPan();
}
function deepestAt(px, py){
  let I = locate(), s = scr(I);
  for (let d = 0; d < 24; d++){
    let n = null;
    for (const c of I.children()){
      const p = c.p, rx = s.x + s.k*p.x, ry = s.y + s.k*p.y;
      if (px >= rx && px <= rx + s.k*p.w && py >= ry && py <= ry + s.k*p.h){ n = c; break; }
    }
    if (!n) break;
    s = {k: s.k*n.s, x: s.x + s.k*n.ox, y: s.y + s.k*n.oy}; I = n;
  }
  return {I, s};
}
function clampPan(){
  const s = scr(world), w = s.k*world.T.w, h = s.k*world.T.h;
  cam.x = Math.min(vw*0.85, Math.max(vw*0.15 - w, cam.x));
  cam.y = Math.min(vh*0.85, Math.max(vh*0.15 - h, cam.y));
}
let flight = null;
function stopFlight(){ if (flight){ cancelAnimationFrame(flight.raf); flight = null; } }
function view(){ return [(vw/2 - cam.x)/cam.k, (vh/2 - cam.y)/cam.k, vw/cam.k]; }
function setView(p){ cam.k = vw/p[2]; cam.x = vw/2 - cam.k*p[0]; cam.y = vh/2 - cam.k*p[1]; }
function interp(p0, p1){
  const [x0, y0, w0] = p0, [x1, y1, w1] = p1, dx = x1-x0, dy = y1-y0, d1 = Math.hypot(dx, dy), lr = Math.log(w1/w0);
  let f, dur;
  if (d1 < 0.6*Math.max(w0, w1) && Math.abs(lr) > 0.05){
    // pure zoom about the fixed point of the two views: an exact similarity, smooth at any depth
    const fx = (x1*w0 - x0*w1)/(w0 - w1), fy = (y1*w0 - y0*w1)/(w0 - w1);
    f = t => { const w = w0*Math.exp(lr*t), q = w/w0; return [fx + (x0 - fx)*q, fy + (y0 - fy)*q, w]; };
    dur = Math.abs(lr)*330;
  } else {
    const rho = Math.SQRT2, r2 = 2, r4 = 4;
    if (d1 < 1e-12){ f = t => [x0, y0, w0*Math.exp(lr*t)]; dur = Math.abs(lr)*330; }
    else {
      const b0 = (w1*w1 - w0*w0 + r4*d1*d1)/(2*w0*r2*d1), b1 = (w1*w1 - w0*w0 - r4*d1*d1)/(2*w1*r2*d1);
      const z0 = -Math.asinh(b0), z1 = -Math.asinh(b1), S = (z1 - z0)/rho;
      f = t => { const s = t*S, c0 = Math.cosh(z0), u = w0/(r2*d1)*(c0*Math.tanh(rho*s + z0) - Math.sinh(z0)); return [x0 + u*dx, y0 + u*dy, w0*c0/Math.cosh(rho*s + z0)]; };
      dur = Math.abs(S)*1000;
    }
  }
  f.dur = Math.min(2600, Math.max(380, dur));
  return f;
}
function flyTo(target){
  stopFlight(); hideHover();
  const f = interp(view(), target), dur = reduce ? 0 : f.dur, t0 = performance.now();
  const step = now => {
    const t = dur ? Math.min(1, (now - t0)/dur) : 1, e = t < .5 ? 4*t*t*t : 1 - Math.pow(-2*t + 2, 3)/2;
    setView(t >= 1 ? target : f(e)); render();
    if (t < 1) flight.raf = requestAnimationFrame(step); else flight = null;
  };
  flight = {raf: requestAnimationFrame(step)};
}
function targetFor(I){
  const w = I.T.w*I.A.S, h = I.T.h*I.A.S, k = Math.min(vw*0.92/w, vh*0.88/h);
  return [I.A.X + w/2, I.A.Y + h/2, vw/k];
}
function flyToInst(I){ flyTo(targetFor(I)); }

// ------------------------------------------------------------ picking
function pick(px, py){
  let I = frame, s = scr(I);
  for (let d = 0; d < 16; d++){
    let hit = null;
    for (const c of I.children()){
      const p = c.p, rx = s.x + s.k*p.x, ry = s.y + s.k*p.y, rw = s.k*p.w, rh = s.k*p.h;
      if (rw < 9 || rh < 7) continue;
      if (px >= rx && px <= rx + rw && py >= ry && py <= ry + rh){ hit = {c, rx, ry, rw, rh}; break; }
    }
    if (!hit) return null;
    const kc = s.k*hit.c.s;
    if (fitK(hit.c.T)/kc > 1.35) return hit;
    I = hit.c; s = {k: kc, x: s.x + s.k*hit.c.ox, y: s.y + s.k*hit.c.oy};
  }
  return null;
}
function hideHover(){ hov.setAttribute('width', 0); hovt.textContent = ''; canvas.classList.remove('ptr'); }
function showHover(px, py){
  const h = pick(px, py);
  if (!h){ hideHover(); return; }
  hov.setAttribute('x', h.rx); hov.setAttribute('y', h.ry); hov.setAttribute('width', h.rw); hov.setAttribute('height', h.rh);
  hovt.setAttribute('x', h.rx + 2); hovt.setAttribute('y', Math.max(14, h.ry - 6)); hovt.textContent = h.c.name + '  →';
  canvas.classList.add('ptr');
}

// ------------------------------------------------------------ input
const ptrs = new Map(); let moved = 0, pinch = null, hinted = false;
function pt(e){ const r = canvas.getBoundingClientRect(); return {x: e.clientX - r.left, y: e.clientY - r.top}; }
function unhint(){ if (!hinted){ hinted = true; document.getElementById('hint').classList.add('gone'); } }
canvas.addEventListener('pointerdown', e => {
  canvas.setPointerCapture(e.pointerId); const p = pt(e); ptrs.set(e.pointerId, p);
  if (ptrs.size === 1){ moved = 0; } else { pinch = null; moved = 99; }
  stopFlight(); canvas.classList.add('drag');
});
canvas.addEventListener('pointermove', e => {
  const p = pt(e);
  if (!ptrs.has(e.pointerId)){ if (e.pointerType === 'mouse' && !flight) showHover(p.x, p.y); return; }
  const prev = ptrs.get(e.pointerId); ptrs.set(e.pointerId, p);
  if (ptrs.size === 1){
    const dx = p.x - prev.x, dy = p.y - prev.y; moved += Math.abs(dx) + Math.abs(dy);
    if (moved > 4){ cam.x += dx; cam.y += dy; clampPan(); hideHover(); render(); unhint(); }
  } else if (ptrs.size === 2){
    const [a, b] = [...ptrs.values()], m = {x:(a.x + b.x)/2, y:(a.y + b.y)/2}, d = Math.hypot(a.x - b.x, a.y - b.y);
    if (pinch){ cam.x += m.x - pinch.m.x; cam.y += m.y - pinch.m.y; zoomAt(m.x, m.y, d/pinch.d); render(); unhint(); }
    pinch = {m, d};
  }
});
function endPtr(e){
  const p = pt(e); const had = ptrs.delete(e.pointerId);
  if (ptrs.size < 2) pinch = null;
  if (ptrs.size === 0){ canvas.classList.remove('drag'); if (had && moved <= 4 && e.type === 'pointerup') clickAt(p.x, p.y); }
}
canvas.addEventListener('pointerup', endPtr);
canvas.addEventListener('pointercancel', endPtr);
canvas.addEventListener('pointerleave', () => { if (!ptrs.size) hideHover(); });
function clickAt(x, y){ const h = pick(x, y); if (h){ unhint(); flyToInst(h.c); } }
canvas.addEventListener('dblclick', e => {
  const p = pt(e); if (pick(p.x, p.y)) return;
  const v = view(), nx = v[0] + (p.x - vw/2)/cam.k*0.5, ny = v[1] + (p.y - vh/2)/cam.k*0.5;
  flyTo([nx, ny, v[2]/2]); unhint();
});
canvas.addEventListener('wheel', e => {
  e.preventDefault(); stopFlight(); hideHover();
  const p = pt(e), unit = e.deltaMode === 1 ? 0.05 : e.deltaMode === 2 ? 1 : 0.0022;
  zoomAt(p.x, p.y, Math.exp(-e.deltaY*unit*(e.ctrlKey ? 3 : 1))); render(); unhint();
}, {passive: false});

function zoomBtn(f){ const v = view(); flyTo([v[0], v[1], v[2]/f]); }
function upOne(){ const I = focus && focus.parent ? focus.parent : (frame.parent || world); flyToInst(I); }
document.getElementById('zin').onclick = () => zoomBtn(2);
document.getElementById('zout').onclick = () => zoomBtn(0.5);
document.getElementById('up').onclick = upOne;
document.getElementById('home').onclick = () => flyToInst(world);
document.getElementById('ptog').onclick = () => document.getElementById('panel').classList.toggle('closed');
window.addEventListener('keydown', e => {
  if (e.target.tagName === 'SELECT' || e.target.tagName === 'INPUT') return;
  const k = e.key;
  if (k === '+' || k === '='){ zoomBtn(2); }
  else if (k === '-' || k === '_'){ zoomBtn(0.5); }
  else if (k === 'Escape' || k === 'Backspace'){ upOne(); }
  else if (k === '0' || k === 'Home'){ flyToInst(world); }
  else if (k.startsWith('Arrow')){ stopFlight(); const d = 70; if (k === 'ArrowLeft') cam.x += d; if (k === 'ArrowRight') cam.x -= d; if (k === 'ArrowUp') cam.y += d; if (k === 'ArrowDown') cam.y -= d; clampPan(); render(); }
  else return;
  e.preventDefault();
});

// ------------------------------------------------------------ paths for jumps and the ladder
function pathTo(fromTid, test){
  const seen = new Set([fromTid]), q = [[fromTid, []]];
  while (q.length){
    const [t, path] = q.shift();
    if (path.length && test(t)) return path;
    TYPES[t].portals.forEach((p, i) => { if (!seen.has(p.t)){ seen.add(p.t); q.push([p.t, path.concat(i)]); } });
  }
  return null;
}
function follow(I, path){ for (const i of path) I = I.children()[i]; return I; }
const jump = document.getElementById('jump');
jump.addEventListener('change', () => {
  const t = jump.value; if (!t) return;
  const path = pathTo('world', x => x === t); if (path) { flyToInst(follow(world, path)); unhint(); }
  jump.value = '';
});

// ------------------------------------------------------------ UI: breadcrumb, ladder, panel
function computeFocus(){
  let I = frame, s = scr(I);
  for (let d = 0; d < 16; d++){
    let n = null;
    for (const c of I.children()){
      const kc = s.k*c.s, xc = s.x + s.k*c.ox, yc = s.y + s.k*c.oy, w = kc*c.T.w, h = kc*c.T.h;
      if (xc <= vw/2 && xc + w >= vw/2 && yc <= vh/2 && yc + h >= vh/2 && w*h >= 0.2*vw*vh){ n = {c, s:{k:kc, x:xc, y:yc}}; break; }
    }
    if (!n) break; I = n.c; s = n.s;
  }
  return I;
}
const crumbs = document.getElementById('crumbs'), ladder = document.getElementById('ladder'), pinfo = document.getElementById('pinfo'), scaleEl = document.getElementById('scale');
function chain(I){ const a = []; while (I){ a.unshift(I); I = I.parent; } return a; }
function updateUI(){
  const F = computeFocus();
  const z = cam.k / fitK(world.T);
  scaleEl.textContent = 'zoom ×' + (z < 1000 ? z.toFixed(z < 10 ? 1 : 0) : z.toExponential(1).replace('e+', 'e'));
  if (F === focus) return;
  focus = F;
  crumbs.innerHTML = '';
  chain(F).forEach((I, i) => {
    if (i){ const s = document.createElement('i'); s.textContent = '›'; crumbs.appendChild(s); }
    const b = document.createElement('button'); b.textContent = I.parent ? I.name : 'Map'; b.title = I.T.title;
    b.onclick = () => flyToInst(I); crumbs.appendChild(b);
  });
  ladder.querySelectorAll('button').forEach(b => b.classList.toggle('on', +b.dataset.l === F.T.layer));
  const kids = {}, order = [];
  F.children().forEach(c => { if (!kids[c.tid]){ kids[c.tid] = {c, n: 0}; order.push(c.tid); } kids[c.tid].n++; });
  const kidHtml = order.map(t => { const k = kids[t]; return `<button data-i="${k.c.idx}">${esc(k.n > 1 ? TYPES[t].title : k.c.name)}${k.n > 1 ? ` <span>×${k.n}</span>` : ''}</button>`; }).join('');
  pinfo.innerHTML = `<p class="lay">${LAYERS[F.T.layer]}</p><h2>${esc(F.T.title)}</h2>` +
    (F.parent ? `<p class="inst">instance: ${esc(F.name)}${F.parent.parent ? ' in ' + esc(F.parent.name) : ''}</p>` : '<p class="inst">the whole machine</p>') +
    (order.length ? `<h4>Zoom into</h4><div class="kids">${kidHtml}</div>` : '<p class="inst">deepest level: nothing further inside</p>') + F.T.info;
  pinfo.querySelectorAll('.kids button').forEach(b => b.onclick = () => { flyToInst(F.children()[+b.dataset.i]); unhint(); });
  document.getElementById('panel').scrollTop = 0;
}
function esc(s){ return String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }
ladder.querySelectorAll('button').forEach(b => b.onclick = () => {
  const L = +b.dataset.l, F = focus || world;
  if (L === F.T.layer) return flyToInst(F);
  if (L > F.T.layer){ let I = F; while (I.parent && I.T.layer < L) I = I.parent; return flyToInst(I); }
  const path = pathTo(F.tid, t => TYPES[t].layer === L);
  if (path) flyToInst(follow(F, path));
});

// ------------------------------------------------------------ size
function resize(){
  const r = canvas.getBoundingClientRect(); const ow = vw, oh = vh;
  vw = Math.max(200, r.width); vh = Math.max(200, r.height);
  svg.setAttribute('viewBox', `0 0 ${vw} ${vh}`);
  if (!frame){ setView(targetFor(world)); }
  else { cam.x += (vw - ow)/2; cam.y += (vh - oh)/2; }
  render();
}
if (innerWidth < 900) document.getElementById('panel').classList.add('closed');
new ResizeObserver(resize).observe(canvas);
resize();
window.__map = {flyToInst, world, follow, pathTo, get frame(){ return frame; }, get focus(){ return focus; }, setView, render, view};
})();
"""


def ladder_html():
    names = {6: "host", 5: "top", 4: "block", 3: "mech", 2: "data", 1: "prim", 0: "trans"}
    return "".join(f'<li><button data-l="{l}" title="{bm.E(zv.LAYERS[l])}">L{l}<small>{names[l]}</small></button></li>' for l in range(6, -1, -1))


def build_html():
    data = json.dumps(TYPES, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    jump_opts = "".join(f'<option value="{t}">{bm.E(n)}</option>' for t, n in JUMPS)
    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Thermodynamic Computer Map</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans:wght@400;600&display=swap" rel="stylesheet">
<style>{CSS}</style></head><body>
<div class="app">
<header class="bar">
 <div class="brand"><h1>Thermodynamic Computer Map</h1><span>Tang Nano 9K · p-bit sampler · host PC to transistor</span></div>
 <nav class="crumbs" id="crumbs" aria-label="Zoom path"></nav>
 <div class="tools">
  <label class="sr" for="jump">Jump to a component</label>
  <select id="jump"><option value="">Jump to…</option>{jump_opts}</select>
  <a class="navlink" href="reference.html" title="Full reference page: tables, findings, signal dictionary">Reference</a>
  <button class="ic" id="zin" aria-label="Zoom in" title="Zoom in (+)">+</button>
  <button class="ic" id="zout" aria-label="Zoom out" title="Zoom out (−)">−</button>
  <button id="up" title="Up one level (Esc)">Up</button>
  <button id="home" title="Whole map (0)">Whole map</button>
  <button id="ptog" class="ptog">Info</button>
 </div>
</header>
<div class="main">
 <ol class="ladder" id="ladder" aria-label="Abstraction layers">{ladder_html()}</ol>
 <div class="canvas" id="canvas">
  <svg id="zm" class="zm dia" role="img" aria-label="Zoomable schematic of the thermodynamic computer, from the host PC down to single transistors" xmlns="http://www.w3.org/2000/svg">
   <defs>
    <marker id="z-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,1 L10,5 L0,9 z" class="ah"/></marker>
    <marker id="z-m" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,1 L10,5 L0,9 z" class="ahm"/></marker>
    <marker id="z-n" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,1 L10,5 L0,9 z" class="ahn"/></marker>
    <marker id="z-c" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,1 L10,5 L0,9 z" class="ahc"/></marker>
   </defs>
   <g id="stage"></g>
   <g id="ovl" pointer-events="none"><rect id="hov" width="0" height="0"/><text id="hovt"></text></g>
  </svg>
  <div class="hint" id="hint">Scroll or pinch to zoom · drag to pan · click a block to dive in · Esc goes up</div>
  <div class="scale" id="scale"></div>
 </div>
 <aside class="panel" id="panel" aria-label="Component details"><div id="pinfo"></div></aside>
</div>
</div>
<script>window.__TYPES__ = {data};</script>
<script>{JS}</script>
</body></html>"""
    return html


def main(out):
    html = build_html()
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"wrote {out}: {len(html):,} bytes, {len(TYPES)} views")
    for w in bm.WARN:
        print("WARN", w)


if __name__ == "__main__":
    main(sys.argv[1])
