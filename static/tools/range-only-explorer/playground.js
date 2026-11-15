"use strict";

// Range-only localization playground for post 01b (sections 4-6).
(() => {
  const R = RANGE;
  const explorer = document.getElementById("explorer");
  R.embed(explorer);

  const view = { xMin: -6, xMax: 14, yMin: -8, yMax: 12 };
  const svg = document.getElementById("map");
  const m = R.map(svg, view, { x0: 56, x1: 590, y0: 8, y1: 542 });
  R.arrowHead(svg, "head-blue", "--blue");

  const CORNER = R.BEACONS.map((b) => [...b]);
  const AROUND = [[3, 1], [11, 1], [11, 9], [3, 9]];
  const state = { beacons: CORNER.map((b) => [...b]), robot: [...R.P_TRUE], noise: [...R.NOISE], sigma: R.SIGMA, count: 4 };

  // Layers, back to front.
  const circles = state.beacons.map(() => R.add(m.layer, "circle", { class: "circle" }));
  const flatEl = R.add(m.layer, "line", { class: "arrow-orange" });
  const rowEls = state.beacons.map(() => R.add(m.layer, "line", { class: "arrow-blue", "marker-end": "url(#head-blue)" }));
  const starEls = [];
  const robot = R.handle(m, m.layer, "True robot position", (p) => { state.robot = p; draw(); }, () => state.robot);
  R.add(robot.g, "circle", { class: "mark", r: 9, style: "fill: none; stroke: var(--text); stroke-width: 3" });
  const beaconHandles = state.beacons.map((_, i) => {
    const h = R.handle(m, m.layer, `Beacon ${i + 1}`, (p) => { state.beacons[i] = p; draw(); }, () => state.beacons[i]);
    R.add(h.g, "path", { class: "mark", d: "M0 -11L10 7L-10 7z", style: "fill: var(--muted); stroke: var(--panel); stroke-width: 2" });
    R.add(h.g, "text", { class: "label", x: 12, y: -8 }, `b${i + 1}`);
    return h;
  });

  const showJ = document.getElementById("jac");
  const status = document.getElementById("status");

  function draw() {
    const beacons = state.beacons.slice(0, state.count);
    const z = R.measure(beacons, state.robot, state.sigma, state.noise);
    const prob = R.problem(beacons, z);
    R.drawHeat(m, prob, view);

    state.beacons.forEach((b, i) => {
      const on = i < state.count;
      beaconHandles[i].g.style.display = on ? "" : "none";
      circles[i].style.display = on ? "" : "none";
      beaconHandles[i].place(b);
      if (on) R.set(circles[i], { cx: m.sx(b[0]), cy: m.sy(b[1]), r: Math.max(z[i], 0) * m.k });
    });
    robot.place(state.robot);

    const mins = R.minima(prob, view);
    while (starEls.length < mins.length) starEls.push(R.add(m.layer, "path", { class: "minimum", d: "M0 -12L3.5 -4L12 -3.7L5.5 2L7.4 10.5L0 6L-7.4 10.5L-5.5 2L-12 -3.7L-3.5 -4z" }));
    starEls.forEach((el, i) => {
      el.style.display = i < mins.length ? "" : "none";
      if (i < mins.length) el.setAttribute("transform", `translate(${m.sx(mins[i][0])} ${m.sy(mins[i][1])})`);
    });
    m.layer.append(robot.g, ...starEls, ...beaconHandles.map((h) => h.g)); // keep handles on top

    // The rows of J at the best minimum, and the direction they barely see.
    const best = mins[0];
    const J = best ? prob.jacobian(best) : [];
    const { H } = best ? R.normal(J, prob.residuals(best)) : { H: [1, 0, 1] };
    const { l1, l2, v1 } = R.eig(H);
    const kappa = l2 > 1e-12 ? l1 / l2 : Infinity;
    const jOn = showJ.checked && best;
    document.querySelectorAll(".jac-key").forEach((e) => { e.hidden = !jOn; });
    rowEls.forEach((el, i) => {
      el.style.display = jOn && i < state.count ? "" : "none";
      if (jOn && i < state.count) R.arrow(m, el, best, [best[0] + 2.2 * J[i][0], best[1] + 2.2 * J[i][1]]);
    });
    flatEl.style.display = jOn ? "" : "none";
    if (jOn) {
      const flat = [-v1[1], v1[0]]; // the small eigenvector is at right angles to the big one
      R.arrow(m, flatEl, [best[0] - 3.5 * flat[0], best[1] - 3.5 * flat[1]], [best[0] + 3.5 * flat[0], best[1] + 3.5 * flat[1]]);
    }

    document.getElementById("sigma-value").value = `${R.fmt(state.sigma)} m`;
    document.getElementById("est").textContent = best ? R.fmtP(best) : "—";
    document.getElementById("err").textContent = best ? `${R.fmt(R.norm(R.sub(best, state.robot)))} m` : "—";
    document.getElementById("cost").textContent = best ? R.fmt(prob.cost(best), 4) : "—";
    document.getElementById("kappa").textContent = !best ? "—" : kappa > 1e6 ? "huge" : R.fmt(kappa, 1);

    let text, warn = true;
    if (mins.length > 1) text = `${mins.length} minima: several positions explain the ranges about equally well. Which one a search finds depends on where it starts.`;
    else if (kappa > 10) text = `One minimum, in a long, curved valley: near the star, the cost curves ${R.fmt(kappa, 0)} times less along the valley than across it, because the beacons all see the robot from about the same direction.`;
    else { text = `A round bowl (κ = ${R.fmt(kappa, 1)}): the beacons see the robot from different directions, so every way of moving changes some distance.`; warn = false; }
    status.textContent = text;
    status.classList.toggle("warn", warn);
  }

  const sigma = document.getElementById("sigma");
  sigma.addEventListener("input", () => { state.sigma = Number(sigma.value); draw(); });
  document.getElementById("redraw").addEventListener("click", () => {
    // Box-Muller: standard normal draws, scaled by σ in measure().
    state.noise = state.noise.map(() => Math.sqrt(-2 * Math.log(1 - Math.random())) * Math.cos(2 * Math.PI * Math.random()));
    draw();
  });
  const four = document.getElementById("four"), two = document.getElementById("two");
  const setCount = (n) => { state.count = n; four.setAttribute("aria-pressed", n === 4); two.setAttribute("aria-pressed", n === 2); draw(); };
  four.addEventListener("click", () => setCount(4));
  two.addEventListener("click", () => setCount(2));
  document.getElementById("corner").addEventListener("click", () => { state.beacons = CORNER.map((b) => [...b]); draw(); });
  document.getElementById("around").addEventListener("click", () => { state.beacons = AROUND.map((b) => [...b]); draw(); });
  showJ.addEventListener("change", draw);
  draw();
})();
