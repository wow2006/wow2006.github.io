"use strict";

// Line-fit explorer for post 00b; ?view=bowl adds the cost bowl over (a, b) for post 01a. Data: assets/code/optimization/post00/line_fit.py (seed 42).
(() => {
  const t = [0.6381725610417532, 0.9417734788764953, 1.2811363267554587, 2.272387217847769, 3.7079802423258124, 4.388784397520523, 4.4341419882733115, 4.503859378955672, 5.545847870158348, 6.316643991220649, 6.438651200806645, 6.973680290593639, 7.61139701990353, 7.739560485559633, 7.860643052769538, 8.227616132708299, 8.27631171992582, 8.585979199113824, 9.267649888486018, 9.756223516367559];
  const z = [1.8996206307062462, 1.8448610564588315, 3.258433864608785, 3.2251920850369338, 4.209387500562785, 4.799747387723482, 5.366699101950659, 5.322353941782985, 6.184317863084272, 6.81180779478125, 7.935909521167593, 6.835095222644142, 7.2817719784799015, 7.20338475149898, 8.158102095760928, 8.759476281799175, 8.052680901347733, 7.864689473113543, 8.41943118137407, 9.695334485788868];
  const truth = { a: 0.8, b: 1.5 };
  const FAULTY = 0; // the "faulty reading" is the first point: at the end of the line it pulls hardest
  const FAULT = 5; // how far up the faulty reading is pushed
  const NOISY_SIGMA = 5; // a noisy sensor; trusted ones have σ = 1, as in the post

  // Arabic strings keep formulas left-to-right with Unicode isolates (like the {{< en >}} shortcode).
  const iso = (s) => `⁦${s}⁩`;
  const translations = {
    en: {
      pageTitle: "Interactive line-fit explorer",
      title: "Change the line, watch the residuals",
      intro: "Move the slope and intercept. The 20 measurements stay fixed; click any point to change how much you trust it.",
      controlsLabel: "Line parameters",
      slope: "Slope a",
      intercept: "Intercept b",
      best: "Jump to best fit",
      truth: "Show true line",
      faultyAdd: "Add a faulty reading",
      faultyRemove: "Remove the faulty reading",
      keyYours: "your line",
      keyBest: "best fit",
      keyTrue: "true line",
      keyResidual: "residual rᵢ",
      keyTrusted: "trusted (σ = 1)",
      keyNoisy: "noisy (σ = 5)",
      plotLabel: "Measurements, your line, the best fit and the true line",
      cost: "Your cost F(x)",
      bestCost: "Best possible F",
      trueCost: "F of the true line",
      atMin: "You are on the best line: any change to a or b makes F bigger.",
      above: (gap) => `Your F is ${gap} above the minimum. The dashed line shows where the minimum is.`,
      faultyPull: "The faulty reading drags the best line toward it. Click that point to mark it as noisy.",
      faultyCalm: "The faulty point is marked noisy: its squared residual counts 25 times less, so the best line stays near the other points.",
      explanation: "Each pink segment is the residual rᵢ = hᵢ(x) − zᵢ, with hᵢ(x) = a tᵢ + b. The cost is F(x) = ½ Σ (rᵢ/σᵢ)². A trusted point has σᵢ = 1 (weight 1); a noisy one has σᵢ = 5 (weight 1/25), so the same residual's square counts 25 times less.",
      step1: "Move the sliders to make F(x) as small as you can. You cannot go below the dashed best line: it is the minimum.",
      step2: "Press “Show true line”. Its cost (2.2048) is higher than the best (2.1610): the best line fits this noise, not the truth.",
      step3: "Press “Add a faulty reading”. One bad point drags the best line toward it.",
      step4: "Click that point to mark it as noisy (σ = 5, weight 1/25). Its squared residual counts 25 times less, and the best line comes back.",
      xAxis: "t (where we measure)",
      yAxis: "z (measured value)",
      point: (i, ti, zi, r, term, noisy) =>
        `Point ${i}: t = ${ti}, z = ${zi}, residual ${r}, cost term ${term}, ${noisy ? "noisy (σ = 5)" : "trusted (σ = 1)"}. Click to change.`,
      keyYourPoint: "your (a, b)",
      keyBottom: "bottom of the bowl",
      keyContour: "equal cost",
      keyDownhill: "downhill, −∇F",
      bowlLabel: "The cost F over slope a and intercept b: drag the dot to change the line",
      gradient: "Slope of the bowl ∇F",
      aAxis: "slope a",
      bAxis: "intercept b",
      bowl: {
        title: "Every line is a point in a bowl",
        intro: "Each point (a, b) on the right is one line on the left, and the contours show its cost F. Drag the blue dot, or move the sliders.",
        atMin: "You are at the bottom of the bowl: ∇F = 0 in both directions. Those two equations are the normal equations AᵀA x = Aᵀb.",
        above: (gap) => `Your F is ${gap} above the minimum. Follow the pink arrow downhill.`,
        explanation: "The two numbers of ∇F are ∂F/∂a = Σ wᵢ rᵢ tᵢ and ∂F/∂b = Σ wᵢ rᵢ, with weight wᵢ = 1/σᵢ². Stacked, they are AᵀA x − Aᵀb (with weights, AᵀWA x − AᵀWb). F is quadratic in (a, b), so every contour is an exact ellipse, all around the same center.",
        step1: "Drag the blue dot. The pink arrow points straight downhill, across the contours, and shrinks as you get close to the bottom.",
        step2: "Press “Jump to best fit”. The arrow vanishes: ∇F = (0, 0). Solving ∇F = 0 is solving the normal equations.",
        step3: "Look at the shape: long and thin. Along its long axis, the bowl curves 223 times less than across it. That ratio is κ(AᵀA).",
        step4: "Press “Add a faulty reading”. The bottom moves, but the bowl keeps its shape: AᵀA depends only on where we measure (tᵢ), not on what we read (zᵢ). Mark the point noisy, and the weights change AᵀWA, so the shape changes a little too.",
      },
    },
    ar: {
      pageTitle: "مستكشف ملاءمة الخط",
      title: "غيّر الخط وشاهد البواقي",
      intro: "حرّك الميل والمقطع. تبقى القياسات العشرون ثابتة؛ انقر أي نقطة لتغيير مقدار الثقة بها.",
      controlsLabel: "معاملات الخط",
      slope: `الميل ${iso("a")}`,
      intercept: `المقطع الصادي ${iso("b")}`,
      best: "انتقل إلى أفضل ملاءمة",
      truth: "عرض الخط الحقيقي",
      faultyAdd: "أضف قراءة معيبة",
      faultyRemove: "أزل القراءة المعيبة",
      keyYours: "خطّك",
      keyBest: "أفضل ملاءمة",
      keyTrue: "الخط الحقيقي",
      keyResidual: `الباقي ${iso("rᵢ")}`,
      keyTrusted: `موثوقة ${iso("(σ = 1)")}`,
      keyNoisy: `كثيرة الضجيج ${iso("(σ = 5)")}`,
      plotLabel: "القياسات وخطّك وأفضل ملاءمة والخط الحقيقي",
      cost: `تكلفتك ${iso("F(x)")}`,
      bestCost: `أصغر ${iso("F")} ممكنة`,
      trueCost: `${iso("F")} للخط الحقيقي`,
      atMin: `أنت على أفضل خط: أي تغيير في ${iso("a")} أو ${iso("b")} يزيد ${iso("F")}.`,
      above: (gap) => `تكلفتك أعلى من الحد الأدنى بمقدار ${iso(gap)}. يبيّن الخط المتقطع موضع الحد الأدنى.`,
      faultyPull: "القراءة المعيبة تسحب أفضل خط نحوها. انقر تلك النقطة لتصنيفها كثيرة الضجيج.",
      faultyCalm: "النقطة المعيبة مصنّفة كثيرة الضجيج: يُحسب مربع باقيها أقل بخمس وعشرين مرة، فيبقى أفضل خط قريبًا من النقاط الأخرى.",
      explanation: `كل قطعة وردية هي الباقي ${iso("rᵢ = hᵢ(x) − zᵢ")}، حيث ${iso("hᵢ(x) = a tᵢ + b")}. والتكلفة هي ${iso("F(x) = ½ Σ (rᵢ/σᵢ)²")}. للنقطة الموثوقة ${iso("σᵢ = 1")} (الوزن 1)، وللنقطة كثيرة الضجيج ${iso("σᵢ = 5")} (الوزن ${iso("1/25")})، فمربع الباقي نفسه يُحسب أقل بخمس وعشرين مرة.`,
      step1: `حرّك المنزلقين لتجعل ${iso("F(x)")} أصغر ما يمكن. لن تنزل تحت الخط المتقطع: إنه الحد الأدنى.`,
      step2: "اضغط «عرض الخط الحقيقي». تكلفته (2.2048) أعلى من تكلفة أفضل خط (2.1610): أفضل خط يلائم هذا الضجيج بعينه، لا الحقيقة.",
      step3: "اضغط «أضف قراءة معيبة». نقطة سيئة واحدة تسحب أفضل خط نحوها.",
      step4: `انقر تلك النقطة لتصنيفها كثيرة الضجيج (${iso("σ = 5")}، والوزن ${iso("1/25")}). يُحسب مربع باقيها أقل بخمس وعشرين مرة، فيعود أفضل خط.`,
      xAxis: "t (موضع القياس)",
      yAxis: "z (القيمة المقيسة)",
      point: (i, ti, zi, r, term, noisy) =>
        `النقطة ${i}: ${iso(`t = ${ti}, z = ${zi}`)}، الباقي ${iso(r)}، حدّ التكلفة ${iso(term)}، ${noisy ? `كثيرة الضجيج ${iso("(σ = 5)")}` : `موثوقة ${iso("(σ = 1)")}`}. انقر للتغيير.`,
      keyYourPoint: `نقطتك ${iso("(a, b)")}`,
      keyBottom: "قاع الوعاء",
      keyContour: "تكلفة متساوية",
      keyDownhill: `اتجاه النزول ${iso("−∇F")}`,
      bowlLabel: `التكلفة ${iso("F")} بدلالة الميل ${iso("a")} والمقطع ${iso("b")}: اسحب النقطة لتغيير الخط`,
      gradient: `ميل الوعاء ${iso("∇F")}`,
      aAxis: "a (الميل)",
      bAxis: "b (المقطع)",
      bowl: {
        title: "كل خط نقطة في وعاء",
        intro: `كل نقطة ${iso("(a, b)")} في الرسم الثاني هي خط في الرسم الأول، وتبيّن الخطوط الكنتورية تكلفته ${iso("F")}. اسحب النقطة الزرقاء أو حرّك المنزلقين.`,
        atMin: `أنت في قاع الوعاء: ${iso("∇F = 0")} في الاتجاهين. هاتان المعادلتان هما المعادلات الناظمية ${iso("AᵀA x = Aᵀb")}.`,
        above: (gap) => `تكلفتك أعلى من الحد الأدنى بمقدار ${iso(gap)}. اتبع السهم الوردي نزولًا.`,
        explanation: `عددا ${iso("∇F")} هما ${iso("∂F/∂a = Σ wᵢ rᵢ tᵢ")} و${iso("∂F/∂b = Σ wᵢ rᵢ")}، حيث الوزن ${iso("wᵢ = 1/σᵢ²")}. معًا يساويان ${iso("AᵀA x − Aᵀb")} (ومع الأوزان ${iso("AᵀWA x − AᵀWb")}). ولأن ${iso("F")} تربيعية في ${iso("(a, b)")}، فكل خط كنتوري قطع ناقص تام، وكلها حول المركز نفسه.`,
        step1: "اسحب النقطة الزرقاء. يشير السهم الوردي إلى أسرع نزول، عموديًا على الخطوط الكنتورية، ويقصر كلما اقتربت من القاع.",
        step2: `اضغط «انتقل إلى أفضل ملاءمة». يختفي السهم: ${iso("∇F = (0, 0)")}. حلّ ${iso("∇F = 0")} هو حلّ المعادلات الناظمية.`,
        step3: `انظر إلى الشكل: طويل ورفيع. على محوره الطويل ينحني الوعاء أقل بـ 223 مرة مما ينحني عرضًا. هذه النسبة هي ${iso("κ(AᵀA)")}.`,
        step4: `اضغط «أضف قراءة معيبة». يتحرك القاع لكن الوعاء يحتفظ بشكله: ${iso("AᵀA")} تعتمد على مواضع القياس ${iso("tᵢ")} فقط، لا على القيم المقيسة ${iso("zᵢ")}. صنّف النقطة كثيرة الضجيج، فتتغير الأوزان في ${iso("AᵀWA")} ويتغير الشكل قليلًا.`,
      },
    },
  };
  const params = new URLSearchParams(location.search);
  const language = params.get("lang") === "ar" ? "ar" : "en";
  const bowlView = params.get("view") === "bowl";
  const words = bowlView ? { ...translations[language], ...translations[language].bowl } : translations[language];
  const root = document.documentElement;
  root.lang = language;
  root.dir = language === "ar" ? "rtl" : "ltr";
  document.title = words.pageTitle;
  document.querySelectorAll("[data-i18n]").forEach((e) => { e.textContent = words[e.dataset.i18n]; });
  document.querySelectorAll("[data-i18n-label]").forEach((e) => e.setAttribute("aria-label", words[e.dataset.i18nLabel]));

  // Embedded in a post (same origin): follow the site's light/dark toggle and fit the iframe to the content.
  try { root.style.colorScheme = parent.document.documentElement.dataset.theme || ""; } catch {}
  const explorer = document.getElementById("explorer");
  explorer.classList.toggle("bowl-view", bowlView);
  if (frameElement) new ResizeObserver(() => { frameElement.style.height = `${explorer.offsetHeight}px`; }).observe(explorer);

  // The model, the weights and the cost, as in the post.
  const noisy = t.map(() => false);
  let faulty = false;
  const zAt = (i) => z[i] + (faulty && i === FAULTY ? FAULT : 0);
  const weight = (i) => (noisy[i] ? 1 / NOISY_SIGMA ** 2 : 1);
  const residual = (p, i) => p.a * t[i] + p.b - zAt(i);
  const cost = (p) => t.reduce((sum, _, i) => sum + 0.5 * weight(i) * residual(p, i) ** 2, 0);
  // Weighted least squares for a line: the 2x2 normal equations, solved directly.
  function bestFit() {
    let s = 0, st = 0, stt = 0, sz = 0, stz = 0;
    t.forEach((ti, i) => {
      const w = weight(i);
      s += w; st += w * ti; stt += w * ti * ti; sz += w * zAt(i); stz += w * ti * zAt(i);
    });
    const det = s * stt - st * st;
    return { a: (s * stz - st * sz) / det, b: (stt * sz - st * stz) / det };
  }

  // Plot: built once, then draw() only moves things.
  const svg = document.getElementById("plot");
  const [x0, x1, y0, y1, minY, maxY] = [72, 664, 14, 350, -1, 12];
  const px = (v) => x0 + (v / 10) * (x1 - x0);
  const py = (v) => y1 - ((v - minY) / (maxY - minY)) * (y1 - y0);
  const add = (parent, tag, attrs, text) => {
    const e = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (text !== undefined) e.textContent = text;
    parent.append(e);
    return e;
  };
  [0, 2, 4, 6, 8, 10].forEach((v) => {
    add(svg, "line", { class: "gridline", x1: px(v), y1: y0, x2: px(v), y2: y1 });
    add(svg, "text", { x: px(v), y: y1 + 28, "text-anchor": "middle" }, v);
  });
  [0, 3, 6, 9, 12].forEach((v) => {
    add(svg, "line", { class: "gridline", x1: x0, y1: py(v), x2: x1, y2: py(v) });
    add(svg, "text", { x: x0 - 10, y: py(v) + 7, "text-anchor": "end" }, v);
  });
  add(svg, "line", { class: "axis", x1: x0, y1, x2: x1, y2: y1 });
  add(svg, "line", { class: "axis", x1: x0, y1: y0, x2: x0, y2: y1 });
  add(svg, "text", { x: (x0 + x1) / 2, y: 414, "text-anchor": "middle" }, words.xAxis);
  add(svg, "text", { x: 20, y: (y0 + y1) / 2, "text-anchor": "middle", transform: `rotate(-90 20 ${(y0 + y1) / 2})` }, words.yAxis);
  add(add(svg, "clipPath", { id: "area" }), "rect", { x: x0, y: y0, width: x1 - x0, height: y1 - y0 });
  const lines = add(svg, "g", { "clip-path": "url(#area)" });
  const residualEls = t.map(() => add(lines, "line", { class: "residual" }));
  const trueEl = add(lines, "line", { class: "truth" });
  const bestEl = add(lines, "line", { class: "best" });
  const yoursEl = add(lines, "line", { class: "yours" });
  const pointEls = t.map((ti, i) => {
    const g = add(svg, "g", { class: "pt", tabindex: 0, role: "button" });
    add(g, "circle", { class: "hit", cx: px(ti), r: 16 });
    const dot = add(g, "circle", { class: "dot", cx: px(ti), r: 6.5 });
    const toggle = () => { noisy[i] = !noisy[i]; draw(); };
    g.addEventListener("click", toggle);
    g.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggle(); }
    });
    return { g, dot, hit: g.firstChild, title: add(g, "title") };
  });
  const placeLine = (el, p) => {
    el.setAttribute("x1", px(0)); el.setAttribute("y1", py(p.b));
    el.setAttribute("x2", px(10)); el.setAttribute("y2", py(p.a * 10 + p.b));
  };

  const slope = document.getElementById("slope");
  const intercept = document.getElementById("intercept");
  const faultyButton = document.getElementById("faulty");
  const status = document.getElementById("status");
  const format = (v, digits = 3) => Number(v).toFixed(digits);

  // Bowl view: the same cost, drawn over (a, b). F is quadratic, so F = F* + ½ dᵀ H d with
  // d = (a, b) − best and H = AᵀWA: every contour is an exact ellipse, drawn from H's eigenvectors.
  const bowl = document.getElementById("bowl");
  const [aMin, aMax, bMin, bMax] = [0.4, 1.2, -1, 4];
  const qx = (a) => x0 + ((a - aMin) / (aMax - aMin)) * (x1 - x0);
  const qy = (b) => y1 - ((b - bMin) / (bMax - bMin)) * (y1 - y0);
  let contourEls = [], meEl, bottomEl, arrowEl;
  if (bowlView) {
    [slope.min, slope.max, intercept.min, intercept.max] = [aMin, aMax, bMin, bMax];
    slope.step = intercept.step = "any"; // so "Jump to best fit" lands exactly where ∇F = 0
    const start = bestFit();
    [slope.value, intercept.value] = [start.a, start.b];
    [0.4, 0.6, 0.8, 1, 1.2].forEach((v) => {
      add(bowl, "line", { class: "gridline", x1: qx(v), y1: y0, x2: qx(v), y2: y1 });
      add(bowl, "text", { x: qx(v), y: y1 + 28, "text-anchor": "middle" }, v);
    });
    [-1, 0, 1, 2, 3, 4].forEach((v) => {
      add(bowl, "line", { class: "gridline", x1: x0, y1: qy(v), x2: x1, y2: qy(v) });
      add(bowl, "text", { x: x0 - 10, y: qy(v) + 7, "text-anchor": "end" }, v);
    });
    add(bowl, "line", { class: "axis", x1: x0, y1, x2: x1, y2: y1 });
    add(bowl, "line", { class: "axis", x1: x0, y1: y0, x2: x0, y2: y1 });
    add(bowl, "text", { x: (x0 + x1) / 2, y: 414, "text-anchor": "middle" }, words.aAxis);
    add(bowl, "text", { x: 20, y: (y0 + y1) / 2, "text-anchor": "middle", transform: `rotate(-90 20 ${(y0 + y1) / 2})` }, words.bAxis);
    add(add(add(bowl, "defs"), "marker", { id: "arrowhead", viewBox: "0 0 10 10", refX: 5, refY: 5, markerWidth: 4, markerHeight: 4, orient: "auto" }), "path", { d: "M0 0L10 5L0 10z" });
    add(add(bowl, "clipPath", { id: "bowl-area" }), "rect", { x: x0, y: y0, width: x1 - x0, height: y1 - y0 });
    const area = add(bowl, "g", { "clip-path": "url(#bowl-area)" });
    contourEls = [0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128].map((dF) => ({ dF, el: add(area, "polyline", { class: "contour" }) }));
    bottomEl = add(area, "circle", { class: "bottom", r: 7 });
    arrowEl = add(area, "line", { class: "downhill", "marker-end": "url(#arrowhead)" });
    meEl = add(area, "circle", { class: "me", r: 9 });

    // Drag anywhere on the map: the point under the pointer becomes the line.
    const pick = (e) => {
      const m = bowl.getScreenCTM().inverse();
      const sx = e.clientX * m.a + e.clientY * m.c + m.e, sy = e.clientX * m.b + e.clientY * m.d + m.f;
      slope.value = aMin + ((sx - x0) / (x1 - x0)) * (aMax - aMin);
      intercept.value = bMin + ((y1 - sy) / (y1 - y0)) * (bMax - bMin);
      draw();
    };
    bowl.addEventListener("pointerdown", (e) => { bowl.setPointerCapture(e.pointerId); pick(e); });
    bowl.addEventListener("pointermove", (e) => { if (bowl.hasPointerCapture(e.pointerId)) pick(e); });
  }

  function drawBowl(mine, best) {
    let haa = 0, hab = 0, hbb = 0, ga = 0, gb = 0;
    t.forEach((ti, i) => {
      const w = weight(i), r = residual(mine, i);
      haa += w * ti * ti; hab += w * ti; hbb += w;
      ga += w * r * ti; gb += w * r;
    });
    // Eigen-decomposition of the 2x2 symmetric H.
    const mid = (haa + hbb) / 2, rad = Math.hypot((haa - hbb) / 2, hab);
    const [l1, l2] = [mid + rad, mid - rad];
    const v1 = Math.abs(hab) > 1e-12 ? [l1 - hbb, hab] : [1, 0];
    const n1 = Math.hypot(...v1), [c, s] = [v1[0] / n1, v1[1] / n1];
    contourEls.forEach(({ dF, el }) => {
      const pts = [];
      for (let k = 0; k <= 120; k++) {
        const th = (k / 120) * 2 * Math.PI;
        const u = Math.sqrt((2 * dF) / l1) * Math.cos(th), v = Math.sqrt((2 * dF) / l2) * Math.sin(th);
        pts.push(`${qx(best.a + u * c - v * s).toFixed(1)},${qy(best.b + u * s + v * c).toFixed(1)}`);
      }
      el.setAttribute("points", pts.join(" "));
    });
    bottomEl.setAttribute("cx", qx(best.a)); bottomEl.setAttribute("cy", qy(best.b));
    meEl.setAttribute("cx", qx(mine.a)); meEl.setAttribute("cy", qy(mine.b));
    // Steepest descent on the screen: the gradient with respect to screen pixels, so the arrow
    // crosses the drawn contours at right angles even though a and b have different scales.
    const kx = (x1 - x0) / (aMax - aMin), ky = (y1 - y0) / (bMax - bMin);
    const [dx, dy] = [-ga / kx, gb / ky];
    const size = Math.hypot(dx, dy), len = 90 * (1 - Math.exp(-size / 0.03));
    arrowEl.style.display = len < 4 ? "none" : "";
    if (len >= 4) {
      arrowEl.setAttribute("x1", qx(mine.a)); arrowEl.setAttribute("y1", qy(mine.b));
      arrowEl.setAttribute("x2", qx(mine.a) + (dx / size) * len); arrowEl.setAttribute("y2", qy(mine.b) + (dy / size) * len);
    }
    const g = `(${format(ga, 2)}, ${format(gb, 2)})`;
    document.getElementById("gradient-value").textContent = language === "ar" ? iso(g) : g;
  }

  function draw() {
    const mine = { a: Number(slope.value), b: Number(intercept.value) };
    const best = bestFit();
    const [myCost, bestCost] = [cost(mine), cost(best)];
    placeLine(yoursEl, mine);
    placeLine(bestEl, best);
    placeLine(trueEl, truth);
    t.forEach((ti, i) => {
      const r = residual(mine, i);
      const res = residualEls[i];
      res.setAttribute("x1", px(ti)); res.setAttribute("x2", px(ti));
      res.setAttribute("y1", py(zAt(i))); res.setAttribute("y2", py(zAt(i) + r));
      res.classList.toggle("noisy", noisy[i]);
      const p = pointEls[i];
      p.dot.setAttribute("cy", py(zAt(i)));
      p.hit.setAttribute("cy", py(zAt(i)));
      p.g.classList.toggle("noisy", noisy[i]);
      p.g.setAttribute("aria-pressed", noisy[i]);
      const label = words.point(i + 1, format(ti, 2), format(zAt(i), 2), format(r), format(0.5 * weight(i) * r * r, 4), noisy[i]);
      p.g.setAttribute("aria-label", label);
      p.title.textContent = label;
    });

    document.getElementById("slope-value").value = format(mine.a);
    document.getElementById("intercept-value").value = format(mine.b);
    document.getElementById("cost-value").textContent = format(myCost, 4);
    document.getElementById("best-cost-value").textContent = format(bestCost, 4);
    document.getElementById("true-cost-value").textContent = format(cost(truth), 4);
    if (bowlView) drawBowl(mine, best);

    // Sliders move in steps of 0.001, so "on the minimum" allows a tiny gap.
    const gap = myCost - bestCost;
    const parts = [gap < 5e-4 ? words.atMin : words.above(format(gap, 4))];
    if (faulty) parts.push(noisy[FAULTY] ? words.faultyCalm : words.faultyPull);
    status.textContent = parts.join(" ");
  }

  const jump = (p) => { slope.value = p.a; intercept.value = p.b; draw(); };
  [slope, intercept].forEach((input) => input.addEventListener("input", draw));
  document.getElementById("best").addEventListener("click", () => jump(bestFit()));
  document.getElementById("truth").addEventListener("click", () => jump(truth));
  faultyButton.addEventListener("click", () => {
    faulty = !faulty;
    faultyButton.setAttribute("aria-pressed", faulty);
    faultyButton.textContent = faulty ? words.faultyRemove : words.faultyAdd;
    draw();
  });
  draw();
})();
