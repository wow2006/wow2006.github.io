"use strict";

// Linearization explorer for post 01b (section 7).
(() => {
  const R = RANGE;
  const explorer = document.getElementById("explorer");
  R.embed(explorer);
  const z = R.measure(R.BEACONS, R.P_TRUE, R.SIGMA, R.NOISE);
  const prob = R.problem(R.BEACONS, z);
  const fmtE = (v) => v.toExponential(3);

  // ---- part 1: the tangent at p0 along u, as in the script's "taylor" region ----
  const P0 = [1, 10], U = [0.6, -0.8], B = 2; // beacon 3 (index 2) bends the most along this line
  const at = (s) => [P0[0] + s * U[0], P0[1] + s * U[1]];
  const r0 = prob.residuals(P0), J0 = prob.jacobian(P0);
  const slope = J0.map((row) => row[0] * U[0] + row[1] * U[1]); // J u: each residual's slope along u
  const gaps = (s) => prob.residuals(at(s)).map((r, i) => r - (r0[i] + s * slope[i]));
  const gapAll = (s) => Math.hypot(...gaps(s));

  const curve = document.getElementById("curve");
  const C = { x0: 64, x1: 590, y0: 10, y1: 360 };
  const [sMin, sMax] = [-4, 10];
  const ss = Array.from({ length: 141 }, (_, k) => sMin + (k / 140) * (sMax - sMin));
  const ys = ss.flatMap((s) => [prob.residuals(at(s))[B], r0[B] + s * slope[B]]);
  const [yLo, yHi] = [Math.floor(Math.min(...ys)), Math.ceil(Math.max(...ys))];
  const cx = (s) => C.x0 + ((s - sMin) / (sMax - sMin)) * (C.x1 - C.x0);
  const cy = (v) => C.y1 - ((v - yLo) / (yHi - yLo)) * (C.y1 - C.y0);
  for (let s = sMin; s <= sMax; s += 2) {
    R.add(curve, "line", { class: "gridline", x1: cx(s), y1: C.y0, x2: cx(s), y2: C.y1 });
    R.add(curve, "text", { x: cx(s), y: C.y1 + 24, "text-anchor": "middle" }, s);
  }
  const yStep = Math.max(1, Math.ceil((yHi - yLo) / 6));
  for (let v = yLo; v <= yHi; v += yStep) {
    R.add(curve, "line", { class: "gridline", x1: C.x0, y1: cy(v), x2: C.x1, y2: cy(v) });
    R.add(curve, "text", { x: C.x0 - 8, y: cy(v) + 6, "text-anchor": "end" }, v);
  }
  R.add(curve, "line", { class: "axis", x1: C.x0, y1: C.y1, x2: C.x1, y2: C.y1 });
  R.add(curve, "line", { class: "axis", x1: C.x0, y1: C.y0, x2: C.x0, y2: C.y1 });
  R.add(curve, "text", { x: (C.x0 + C.x1) / 2, y: 412, "text-anchor": "middle" }, "distance moved along the line [m]");
  R.add(curve, "text", { x: 16, y: (C.y0 + C.y1) / 2, "text-anchor": "middle", transform: `rotate(-90 16 ${(C.y0 + C.y1) / 2})` }, "r₃ [m]");
  R.add(curve, "polyline", { points: ss.map((s) => `${cx(s)},${cy(prob.residuals(at(s))[B])}`).join(" "), style: "fill: none; stroke: var(--text); stroke-width: 2.5" });
  R.add(curve, "line", { x1: cx(sMin), y1: cy(r0[B] + sMin * slope[B]), x2: cx(sMax), y2: cy(r0[B] + sMax * slope[B]), style: "stroke: var(--blue); stroke-width: 2.5; stroke-dasharray: 9 6" });
  const gapEl = R.add(curve, "line", { style: "stroke: var(--orange); stroke-width: 4" });
  R.add(curve, "circle", { cx: cx(0), cy: cy(r0[B]), r: 7, style: "fill: var(--blue); stroke: var(--panel); stroke-width: 2" });
  R.add(curve, "text", { class: "label", x: cx(0) + 10, y: cy(r0[B]) - 12 }, "p");
  const onCurve = R.add(curve, "circle", { r: 6, style: "fill: var(--text); stroke: var(--panel); stroke-width: 2" });
  const onLine = R.add(curve, "circle", { r: 6, style: "fill: var(--panel); stroke: var(--blue); stroke-width: 2.5" });

  const loglog = document.getElementById("loglog");
  const L = { x0: 74, x1: 570, y0: 10, y1: 360 };
  const [ex0, ex1, ey0, ey1] = [-3, 1, -9, 1];
  const lx = (d) => L.x0 + ((Math.log10(d) - ex0) / (ex1 - ex0)) * (L.x1 - L.x0);
  const ly = (e) => L.y1 - ((Math.log10(e) - ey0) / (ey1 - ey0)) * (L.y1 - L.y0);
  const sup = (n) => String(n).replace("-", "⁻").replace(/\d/g, (d) => "⁰¹²³⁴⁵⁶⁷⁸⁹"[d]);
  for (let e = ex0; e <= ex1; e++) {
    R.add(loglog, "line", { class: "gridline", x1: lx(10 ** e), y1: L.y0, x2: lx(10 ** e), y2: L.y1 });
    R.add(loglog, "text", { x: lx(10 ** e), y: L.y1 + 24, "text-anchor": "middle" }, `10${sup(e)}`);
  }
  for (let e = ey0 + 1; e <= ey1; e += 2) {
    R.add(loglog, "line", { class: "gridline", x1: L.x0, y1: ly(10 ** e), x2: L.x1, y2: ly(10 ** e) });
    R.add(loglog, "text", { x: L.x0 - 8, y: ly(10 ** e) + 6, "text-anchor": "end" }, `10${sup(e)}`);
  }
  R.add(loglog, "line", { class: "axis", x1: L.x0, y1: L.y1, x2: L.x1, y2: L.y1 });
  R.add(loglog, "line", { class: "axis", x1: L.x0, y1: L.y0, x2: L.x0, y2: L.y1 });
  R.add(loglog, "text", { x: (L.x0 + L.x1) / 2, y: 412, "text-anchor": "middle" }, "step size |δ| [m]");
  R.add(loglog, "text", { x: 16, y: (L.y0 + L.y1) / 2, "text-anchor": "middle", transform: `rotate(-90 16 ${(L.y0 + L.y1) / 2})` }, "gap [m]");
  // The slope-2 guide sits one decade below the data, so both stay visible.
  const c2 = gapAll(1e-3) / 1e-6 / 10;
  R.add(loglog, "line", { x1: lx(1e-3), y1: ly(c2 * 1e-6), x2: lx(10), y2: ly(c2 * 100), style: "stroke: var(--muted); stroke-width: 2; stroke-dasharray: 6 6" });
  const ds = Array.from({ length: 121 }, (_, k) => 10 ** (ex0 + (k / 120) * (ex1 - ex0)));
  R.add(loglog, "polyline", { points: ds.map((d) => `${lx(d)},${ly(gapAll(d))}`).join(" "), style: "fill: none; stroke: var(--orange); stroke-width: 2.5" });
  [1, 0.1, 0.01, 0.001].forEach((d) => R.add(loglog, "circle", { cx: lx(d), cy: ly(gapAll(d)), r: 6, style: "fill: var(--text); stroke: var(--panel); stroke-width: 2" }));
  const here = R.add(loglog, "circle", { r: 9, style: "fill: none; stroke: var(--orange); stroke-width: 3" });

  const stepInput = document.getElementById("step");
  function drawPart1() {
    const d = 10 ** Number(stepInput.value);
    const [rc, rl] = [prob.residuals(at(d))[B], r0[B] + d * slope[B]];
    R.set(gapEl, { x1: cx(d), y1: cy(rc), x2: cx(d), y2: cy(rl) });
    R.set(onCurve, { cx: cx(d), cy: cy(rc) });
    R.set(onLine, { cx: cx(d), cy: cy(rl) });
    R.set(here, { cx: lx(d), cy: ly(gapAll(d)) });
    const label = d >= 0.1 ? d.toFixed(2) : d.toPrecision(2);
    document.getElementById("step-value").value = `${label} m`;
    document.getElementById("m-step").textContent = `${label} m`;
    document.getElementById("m-gap3").textContent = fmtE(Math.abs(rc - rl));
    document.getElementById("m-gap").textContent = fmtE(gapAll(d));
    document.getElementById("m-ratio").textContent = (gapAll(d) / d ** 2).toFixed(4);
  }
  stepInput.addEventListener("input", drawPart1);
  drawPart1();

  // ---- part 2: the model ½‖r + Jδ‖² around a guess, and the step to its minimum ----
  const view = { xMin: -6, xMax: 14, yMin: -8, yMax: 12 };
  const svg = document.getElementById("map");
  const m = R.map(svg, view, { x0: 56, x1: 590, y0: 8, y1: 542 });
  R.drawHeat(m, prob, view);
  R.arrowHead(svg, "head-step", "--blue");
  const best = R.minima(prob, view)[0];
  const ellipses = [1, 1 / 4, 1 / 16].map(() => R.add(m.layer, "polyline", { style: "fill: none; stroke: var(--blue); stroke-width: 2; stroke-dasharray: 7 5" }));
  R.add(m.layer, "path", { class: "minimum", transform: `translate(${m.sx(best[0])} ${m.sy(best[1])})`, d: "M0 -12L3.5 -4L12 -3.7L5.5 2L7.4 10.5L0 6L-7.4 10.5L-5.5 2L-12 -3.7L-3.5 -4z" });
  R.BEACONS.forEach((b, i) => {
    R.add(m.layer, "path", { transform: `translate(${m.sx(b[0])} ${m.sy(b[1])})`, d: "M0 -11L10 7L-10 7z", style: "fill: var(--muted); stroke: var(--panel); stroke-width: 2" });
    R.add(m.layer, "text", { class: "label", x: m.sx(b[0]) + 12, y: m.sy(b[1]) - 8 }, `b${i + 1}`);
  });
  const stepEl = R.add(m.layer, "line", { class: "arrow-blue", "marker-end": "url(#head-step)" });
  const landEl = R.add(m.layer, "circle", { r: 7, style: "fill: var(--panel); stroke: var(--blue); stroke-width: 3" });
  let x = [-2, 8], iteration = 0;
  const guess = R.handle(m, m.layer, "Your guess x", (p) => { x = p; iteration = 0; drawPart2(); }, () => x);
  R.add(guess.g, "circle", { class: "mark", r: 9, style: "fill: var(--blue); stroke: var(--panel); stroke-width: 3" });

  const status = document.getElementById("status");
  let delta = null;
  function drawPart2() {
    const r = prob.residuals(x), J = prob.jacobian(x);
    const { H, g } = R.normal(J, r);
    delta = R.solveStep(H, g); // Gauss-Newton: JᵀJ δ = −Jᵀr, the normal equations of the model
    guess.place(x);
    const F = prob.cost(x);
    document.getElementById("f-now").textContent = F.toFixed(4);
    document.getElementById("iter").textContent = `Step ${iteration}`;
    [stepEl, landEl, ...ellipses].forEach((e) => { e.style.display = delta ? "" : "none"; });
    if (!delta) {
      ["f-model", "f-real", "f-step"].forEach((id) => { document.getElementById(id).textContent = "—"; });
      status.textContent = "Here all the rows of J point the same way, so the model has no single minimum. Move the guess.";
      status.classList.add("warn");
      return;
    }
    const c = [x[0] + delta[0], x[1] + delta[1]];
    const model = 0.5 * r.reduce((s, ri, i) => s + (ri + J[i][0] * delta[0] + J[i][1] * delta[1]) ** 2, 0);
    const Fnew = prob.cost(c);
    // Model contours: ½‖r + Jδ‖² = model + ½ (δ − δ*)ᵀ JᵀJ (δ − δ*), ellipses around the landing point.
    const { l1, l2, v1 } = R.eig(H);
    [1, 1 / 4, 1 / 16].forEach((f, k) => {
      const dF = Math.max(F - model, 1e-9) * f, pts = [];
      for (let i = 0; i <= 120; i++) {
        const th = (i / 120) * 2 * Math.PI;
        const a = Math.sqrt((2 * dF) / l1) * Math.cos(th), b = Math.sqrt((2 * dF) / Math.max(l2, 1e-9)) * Math.sin(th);
        pts.push(`${m.sx(c[0] + a * v1[0] - b * v1[1]).toFixed(1)},${m.sy(c[1] + a * v1[1] + b * v1[0]).toFixed(1)}`);
      }
      ellipses[k].setAttribute("points", pts.join(" "));
    });
    R.arrow(m, stepEl, x, c);
    R.set(landEl, { cx: m.sx(c[0]), cy: m.sy(c[1]) });
    document.getElementById("f-model").textContent = model.toFixed(4);
    document.getElementById("f-real").textContent = Fnew.toFixed(4);
    document.getElementById("f-step").textContent = `${R.norm(delta).toFixed(3)} m`;

    const len = R.norm(delta);
    let text, warn = false;
    if (len < 1e-4) text = "The step is zero: the model's minimum is the real minimum. Gauss-Newton has converged.";
    else if (Fnew > F) { text = `Overshoot: the model promised ${model.toFixed(2)}, but the real cost goes up from ${F.toFixed(2)} to ${Fnew.toFixed(2)}. The model is only good for small steps, and this one is ${len.toFixed(1)} m long.`; warn = true; }
    else text = `Good step: the real cost goes down from ${F.toFixed(4)} to ${Fnew.toFixed(4)}; the model promised ${model.toFixed(4)}.`;
    status.textContent = text;
    status.classList.toggle("warn", warn);
  }
  document.getElementById("take").addEventListener("click", () => {
    if (!delta) return;
    x = [x[0] + delta[0], x[1] + delta[1]]; iteration++; drawPart2();
  });
  document.getElementById("far").addEventListener("click", () => { x = [-2, 8]; iteration = 0; drawPart2(); });
  document.getElementById("near").addEventListener("click", () => { x = [5.5, 6.5]; iteration = 0; drawPart2(); });
  drawPart2();
})();
