"use strict";

// Shared by the range-only tools of post 01b. Data and model: assets/code/optimization/post01/linear_vs_nonlinear.py.
const RANGE = (() => {
  const BEACONS = [[0, 0], [3, 0], [0, 2], [2.5, 2.5]];
  const P_TRUE = [7, 5];
  const SIGMA = 0.3;
  // The script's noise is rng(7).normal(0, 0.3, 4); these are those draws divided by 0.3,
  // so z_i = |p_true - b_i| + SIGMA * NOISE[i] reproduces its ranges exactly.
  const NOISE = [0.00123015, 0.29874554, -0.27413786, -0.89059184];

  const sub = (p, q) => [p[0] - q[0], p[1] - q[1]];
  const norm = (v) => Math.hypot(v[0], v[1]);
  const measure = (beacons, pTrue, sigma, noise) => beacons.map((b, i) => norm(sub(pTrue, b)) + sigma * noise[i]);

  // r_i(p) = |p - b_i| - z_i, and row i of J is the unit vector (p - b_i)^T / |p - b_i|.
  function problem(beacons, z) {
    const residuals = (p) => beacons.map((b, i) => norm(sub(p, b)) - z[i]);
    const jacobian = (p) => beacons.map((b) => {
      const d = sub(p, b), n = Math.max(norm(d), 1e-9);
      return [d[0] / n, d[1] / n];
    });
    const cost = (p) => 0.5 * residuals(p).reduce((s, r) => s + r * r, 0);
    return { beacons, z, residuals, jacobian, cost };
  }

  // JᵀJ and Jᵀr for a 2D state.
  function normal(J, r) {
    let a = 0, b = 0, c = 0, g0 = 0, g1 = 0;
    J.forEach(([j0, j1], i) => { a += j0 * j0; b += j0 * j1; c += j1 * j1; g0 += j0 * r[i]; g1 += j1 * r[i]; });
    return { H: [a, b, c], g: [g0, g1] };
  }
  // Eigenvalues (big, small) and the unit eigenvector of the big one, for symmetric [[a, b], [b, c]].
  function eig([a, b, c]) {
    const mid = (a + c) / 2, rad = Math.hypot((a - c) / 2, b);
    const l1 = mid + rad, l2 = mid - rad;
    const v = Math.abs(b) > 1e-12 ? [l1 - c, b] : (a >= c ? [1, 0] : [0, 1]);
    const n = norm(v);
    return { l1, l2, v1: [v[0] / n, v[1] / n] };
  }
  // Solve [[a, b], [b, c]] x = -g. Returns null when the system is (nearly) singular.
  function solveStep([a, b, c], g) {
    const det = a * c - b * b;
    if (Math.abs(det) < 1e-12 * Math.max(1, a * c)) return null;
    return [-(c * g[0] - b * g[1]) / det, -(a * g[1] - b * g[0]) / det];
  }

  // A small Levenberg-Marquardt, only to find the minima to draw (posts 03-04 explain it).
  function solve(prob, p) {
    let lambda = 1e-3, F = prob.cost(p);
    for (let k = 0; k < 200; k++) {
      const { H, g } = normal(prob.jacobian(p), prob.residuals(p));
      const step = solveStep([H[0] + lambda, H[1], H[2] + lambda], g);
      if (!step) break;
      const q = [p[0] + step[0], p[1] + step[1]], Fq = prob.cost(q);
      if (Fq < F) {
        const done = norm(step) < 1e-10;
        p = q; F = Fq; lambda = Math.max(lambda / 3, 1e-9);
        if (done) break;
      } else lambda *= 4;
    }
    return p;
  }

  // Every local minimum inside the view: grid local minima, polished by LM, then de-duplicated.
  function minima(prob, view, n = 90) {
    const xs = (i) => view.xMin + (i / (n - 1)) * (view.xMax - view.xMin);
    const ys = (j) => view.yMin + (j / (n - 1)) * (view.yMax - view.yMin);
    const F = [];
    for (let j = 0; j < n; j++) { F.push([]); for (let i = 0; i < n; i++) F[j].push(prob.cost([xs(i), ys(j)])); }
    const found = [];
    for (let j = 1; j < n - 1; j++) for (let i = 1; i < n - 1; i++) {
      let lowest = true;
      for (let dj = -1; dj <= 1 && lowest; dj++) for (let di = -1; di <= 1; di++) {
        if ((di || dj) && F[j + dj][i + di] < F[j][i]) { lowest = false; break; }
      }
      if (!lowest) continue;
      const p = solve(prob, [xs(i), ys(j)]);
      const inside = p[0] >= view.xMin && p[0] <= view.xMax && p[1] >= view.yMin && p[1] <= view.yMax;
      if (inside && !found.some((q) => norm(sub(p, q)) < 0.05)) found.push(p);
    }
    return found.sort((p, q) => prob.cost(p) - prob.cost(q));
  }

  // ---- drawing ------------------------------------------------------------
  const add = (parent, tag, attrs = {}, text) => {
    const e = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (text !== undefined) e.textContent = text;
    parent.append(e);
    return e;
  };
  const set = (e, attrs) => { for (const k in attrs) e.setAttribute(k, attrs[k]); return e; };

  // A map with equal scales on both axes, so circles look round and right angles look right.
  function map(svg, view, rect, step = 2) {
    const sx = (x) => rect.x0 + ((x - view.xMin) / (view.xMax - view.xMin)) * (rect.x1 - rect.x0);
    const sy = (y) => rect.y1 - ((y - view.yMin) / (view.yMax - view.yMin)) * (rect.y1 - rect.y0);
    const k = (rect.x1 - rect.x0) / (view.xMax - view.xMin); // pixels per meter
    const heat = add(svg, "image", { x: rect.x0, y: rect.y0, width: rect.x1 - rect.x0, height: rect.y1 - rect.y0, preserveAspectRatio: "none" });
    for (let x = Math.ceil(view.xMin / step) * step; x <= view.xMax; x += step) {
      add(svg, "line", { class: "gridline", x1: sx(x), y1: rect.y0, x2: sx(x), y2: rect.y1 });
      add(svg, "text", { x: sx(x), y: rect.y1 + 22, "text-anchor": "middle" }, x);
    }
    for (let y = Math.ceil(view.yMin / step) * step; y <= view.yMax; y += step) {
      add(svg, "line", { class: "gridline", x1: rect.x0, y1: sy(y), x2: rect.x1, y2: sy(y) });
      add(svg, "text", { x: rect.x0 - 8, y: sy(y) + 6, "text-anchor": "end" }, y);
    }
    add(svg, "line", { class: "axis", x1: rect.x0, y1: rect.y1, x2: rect.x1, y2: rect.y1 });
    add(svg, "line", { class: "axis", x1: rect.x0, y1: rect.y0, x2: rect.x0, y2: rect.y1 });
    add(svg, "text", { x: (rect.x0 + rect.x1) / 2, y: rect.y1 + 46, "text-anchor": "middle" }, "pₓ [m]");
    add(svg, "text", { x: 16, y: (rect.y0 + rect.y1) / 2, "text-anchor": "middle", transform: `rotate(-90 16 ${(rect.y0 + rect.y1) / 2})` }, "p_y [m]");
    const id = `clip${Math.random().toString(36).slice(2)}`;
    add(add(svg, "clipPath", { id }), "rect", { x: rect.x0, y: rect.y0, width: rect.x1 - rect.x0, height: rect.y1 - rect.y0 });
    const layer = add(svg, "g", { "clip-path": `url(#${id})` });
    // Screen point -> map coordinates.
    const toMap = (e) => {
      const m = svg.getScreenCTM().inverse();
      const px = e.clientX * m.a + e.clientY * m.c + m.e, py = e.clientX * m.b + e.clientY * m.d + m.f;
      return [view.xMin + (px - rect.x0) / k, view.yMin + (rect.y1 - py) / k];
    };
    return { sx, sy, k, heat, layer, toMap };
  }

  // Resolve a CSS color token (light-dark() included) to [r, g, b].
  function color(token) {
    const probe = document.body.appendChild(document.createElement("i"));
    probe.style.color = `var(${token})`;
    const rgb = getComputedStyle(probe).color.match(/[\d.]+/g).slice(0, 3).map(Number);
    probe.remove();
    return rgb;
  }

  // The cost as a banded heatmap: log F in 12 bands, light = low, dark = high, so bands read as contours.
  const canvas = document.createElement("canvas");
  function drawHeat(m, prob, view, n = 150) {
    const [lo, hi] = [color("--panel"), color("--heat")];
    canvas.width = canvas.height = n;
    const ctx = canvas.getContext("2d"), img = ctx.createImageData(n, n);
    const vals = [];
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      const x = view.xMin + ((i + 0.5) / n) * (view.xMax - view.xMin);
      const y = view.yMax - ((j + 0.5) / n) * (view.yMax - view.yMin);
      vals.push(Math.log(prob.cost([x, y]) + 0.01));
    }
    const sorted = [...vals].sort((a, b) => a - b);
    const vMin = sorted[0], vMax = sorted[Math.floor(sorted.length * 0.98)];
    vals.forEach((v, k) => {
      const t = Math.floor(Math.min(Math.max((v - vMin) / (vMax - vMin), 0), 0.999) * 12) / 11;
      for (let c = 0; c < 3; c++) img.data[4 * k + c] = lo[c] + t * (hi[c] - lo[c]);
      img.data[4 * k + 3] = 255;
    });
    ctx.putImageData(img, 0, 0);
    m.heat.setAttribute("href", canvas.toDataURL());
  }

  // An arrow from p to q, in map coordinates.
  function arrow(m, el, p, q) {
    return set(el, { x1: m.sx(p[0]), y1: m.sy(p[1]), x2: m.sx(q[0]), y2: m.sy(q[1]) });
  }
  function arrowHead(svg, id, token) {
    const marker = add(add(svg, "defs"), "marker", { id, viewBox: "0 0 10 10", refX: 6, refY: 5, markerWidth: 4, markerHeight: 4, orient: "auto-start-reverse" });
    add(marker, "path", { d: "M0 0L10 5L0 10z", style: `fill: var(${token})` });
  }

  // A draggable, keyboard-movable point. onMove gets the new map position.
  function handle(m, parent, label, onMove, getPos) {
    const g = add(parent, "g", { class: "handle", tabindex: 0, role: "slider", "aria-label": label });
    add(g, "circle", { class: "hit", r: 18 });
    g.addEventListener("pointerdown", (e) => { e.stopPropagation(); g.setPointerCapture(e.pointerId); });
    g.addEventListener("pointermove", (e) => { if (g.hasPointerCapture(e.pointerId)) onMove(m.toMap(e)); });
    g.addEventListener("keydown", (e) => {
      const d = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, 1], ArrowDown: [0, -1] }[e.key];
      if (!d) return;
      e.preventDefault();
      const p = getPos();
      onMove([p[0] + 0.25 * d[0], p[1] + 0.25 * d[1]]);
    });
    const place = (p) => {
      g.setAttribute("transform", `translate(${m.sx(p[0])} ${m.sy(p[1])})`);
      g.setAttribute("aria-valuetext", `${label} at (${p[0].toFixed(2)}, ${p[1].toFixed(2)}) m. Arrow keys move it.`);
    };
    return { g, place };
  }

  // Embedded in a post (same origin): follow the site's light/dark toggle and fit the iframe to the content.
  function embed(explorer) {
    try { document.documentElement.style.colorScheme = parent.document.documentElement.dataset.theme || ""; } catch {}
    if (frameElement) new ResizeObserver(() => { frameElement.style.height = `${explorer.offsetHeight}px`; }).observe(explorer);
  }

  const fmt = (v, d = 2) => Number(v).toFixed(d);
  const fmtP = (p, d = 2) => `(${fmt(p[0], d)}, ${fmt(p[1], d)})`;
  return { BEACONS, P_TRUE, SIGMA, NOISE, sub, norm, measure, problem, normal, eig, solveStep, solve, minima, add, set, map, drawHeat, arrow, arrowHead, handle, embed, fmt, fmtP };
})();
