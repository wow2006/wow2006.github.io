"""Post 00: three top-view pictures of a 2D robot localizing itself from beacons.

Pose = (x, y, theta). Each beacon gives a range z and a bearing phi (angle from the
robot's heading to the beacon). Gauss-Newton shows how the pose and heading are refined.
Run:  python robot_topview.py [output_dir]   (writes 3 SVG files; numpy only)
"""
import sys
from pathlib import Path
import numpy as np

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
B = np.array([[1.0, 6.5], [8.5, 5.5], [6.0, 0.0]])        # beacon positions (known)
truth = np.array([4.0, 3.0, np.deg2rad(35)])               # true pose (x, y, theta)
wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi


def predict(p):
    """Measurement model h: (range, bearing) to every beacon, if the pose were p."""
    d = B - p[:2]
    return np.column_stack([np.hypot(*d.T), wrap(np.arctan2(d[:, 1], d[:, 0]) - p[2])])


rng = np.random.default_rng(3)
Z = predict(truth) + rng.normal(0, [0.12, 0.03], (3, 2))   # noisy measurements z


def resid(p):
    r = predict(p) - Z                                     # predicted - measured
    r[:, 1] = wrap(r[:, 1])
    return r


def cost(p):
    r = resid(p)
    return 0.5 * np.sum((r / [0.12, 0.03]) ** 2)           # weighted by 1/sigma


def gauss_newton(p, n):
    path = [p.copy()]
    for _ in range(n):
        w = 1 / np.array([0.12, 0.03] * 3)
        r = (resid(p).ravel()) * w
        J = np.column_stack([(resid(p + e).ravel() * w - r) / 1e-6 for e in 1e-6 * np.eye(3)])
        p = p + np.linalg.solve(J.T @ J, -J.T @ r)
        p[2] = wrap(p[2])
        path.append(p.copy())
    return path


start = np.array([3.0, 3.9, np.deg2rad(15)])             # bad first guess
path = gauss_newton(start, 3)

# ---------------- tiny SVG helpers (world y points up) ----------------
S, X0, Y0 = 52, 40, 30                                     # px per metre, margins
W, H = 10.5 * S + 2 * X0, 8 * S + 2 * Y0 + 80      # +80: caption strip under the map
sx = lambda x: X0 + x * S
sy = lambda y: Y0 + (7.4 - y) * S
INK, BLUE, ORANGE, RED, GREY, GREEN = "#222", "#0072b2", "#d55e00", "#c0152f", "#999", "#2a9d5c"


def txt(x, y, s, c=INK, size=15, anchor="start", w="normal"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" fill="{c}" font-size="{size}" text-anchor="{anchor}" '
            f'font-weight="{w}" font-family="DejaVu Sans,Arial,sans-serif">{s}</text>')


def line(a, b, c=INK, w=2, dash="", arrow=False):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    m = f' marker-end="url(#ar{c[1:]})"' if arrow else ""
    return (f'<line x1="{sx(a[0]):.1f}" y1="{sy(a[1]):.1f}" x2="{sx(b[0]):.1f}" y2="{sy(b[1]):.1f}" '
            f'stroke="{c}" stroke-width="{w}"{d}{m}/>')


def robot(p, c, fill_op=1.0, label=""):
    """Triangle pointing along the heading, plus the heading arrow."""
    x, y, t = p
    pts = [(x + .32 * np.cos(t), y + .32 * np.sin(t)),
           (x + .26 * np.cos(t + 2.5), y + .26 * np.sin(t + 2.5)),
           (x + .26 * np.cos(t - 2.5), y + .26 * np.sin(t - 2.5))]
    poly = " ".join(f"{sx(a):.1f},{sy(b):.1f}" for a, b in pts)
    out = [f'<polygon points="{poly}" fill="{c}" fill-opacity="{fill_op}" stroke="{c}" stroke-width="2"/>',
           line((x + .3 * np.cos(t), y + .3 * np.sin(t)), (x + 1.0 * np.cos(t), y + 1.0 * np.sin(t)), c, 2.5, arrow=True)]
    return out + ([txt(sx(x) - 14, sy(y) + 26, label, c, 14, "end", "bold")] if label else [])


def arc(c, r, a0, a1, col, n=24):
    t = np.linspace(a0, a1, n)
    pts = " ".join(f"{sx(c[0] + r * np.cos(a)):.1f},{sy(c[1] + r * np.sin(a)):.1f}" for a in t)
    return f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2"/>'


def frame(body, title, w=W, h=H):
    marks = "".join(f'<marker id="ar{c[1:]}" markerWidth="9" markerHeight="9" refX="8" refY="4.5" '
                    f'orient="auto"><path d="M0,0 L9,4.5 L0,9 z" fill="{c}"/></marker>'
                    for c in (INK, BLUE, ORANGE, RED, GREY, GREEN))
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.0f} {h + 34:.0f}" width="{w:.0f}">'
            f'<defs>{marks}</defs><rect width="100%" height="100%" fill="#fff"/>'
            + txt(w / 2, 24, title, INK, 18, "middle", "bold") + f'<g transform="translate(0,34)">{body}</g></svg>')


def world(body=None):
    g = [f'<rect x="{sx(0)}" y="{sy(7.4)}" width="{10.5 * S}" height="{7.4 * S}" fill="#fafafa" stroke="{GREY}"/>']
    g += [line((i, 0), (i, 7.4), "#e6e6e6", 1) for i in range(1, 11)]
    g += [line((0, j), (10.5, j), "#e6e6e6", 1) for j in range(1, 8)]
    g += [line((0, 0), (10.2, 0), INK, 2, arrow=True), line((0, 0), (0, 7.2), INK, 2, arrow=True),
          txt(sx(10.3), sy(0) + 5, "x", INK, 15), txt(sx(0) + 6, sy(7.2) + 4, "y", INK, 15)]
    return g


def beacons(labels=True):
    g = []
    for i, b in enumerate(B, 1):
        g.append(f'<rect x="{sx(b[0]) - 7:.1f}" y="{sy(b[1]) - 7:.1f}" width="14" height="14" fill="{GREEN}"/>')
        if labels:
            g.append(txt(sx(b[0]) + 11, sy(b[1]) - 9, f"B{i}", GREEN, 15, "start", "bold"))
    return g


def dot(p, c, r=5):
    return f'<circle cx="{sx(p[0]):.1f}" cy="{sy(p[1]):.1f}" r="{r}" fill="{c}"/>'


def mid(a, b, f=.5):
    return (a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]))


def meas_tip(p, i):
    """Where the measurement of beacon i says the beacon is, seen from pose p."""
    a = p[2] + Z[i, 1]
    return p[:2] + Z[i, 0] * np.array([np.cos(a), np.sin(a)])


# ---------------- figure 1: the setup ----------------
g = world() + beacons()
for i, b in enumerate(B):
    g.append(line(truth[:2], b, BLUE, 1.8, "6 5"))
    g.append(txt(*[(sx(m) , sy(n)) for m, n in [mid(truth[:2], b, .55)]][0], f"z{i+1}", BLUE, 15, "middle", "bold"))
g += robot(truth, INK)
g.append(dot(truth[:2], INK))
g.append(line(truth[:2], (truth[0] + 1.6, truth[1]), GREY, 1.5, "3 3"))          # parallel to world x
g.append(arc(truth[:2], 1.1, 0, truth[2], RED))
g.append(txt(sx(truth[0] + 1.25), sy(truth[1] + .28), "θ", RED, 17, "start", "bold"))
g.append(arc(truth[:2], .7, truth[2], truth[2] + Z[0, 1], ORANGE))
g.append(txt(sx(truth[0] - .55), sy(truth[1] + 1.05), "φ1", ORANGE, 15, "start", "bold"))
g.append(txt(sx(truth[0] + .35), sy(truth[1]) + 22, "robot (x, y)", INK, 14, "start", "bold"))
g.append(txt(sx(.15), sy(0) + 50, "Robot pose = (x, y, θ): position + heading.", INK, 14))
g.append(txt(sx(.15), sy(0) + 70, "Each beacon Bi (known position) gives a range zi and a", INK, 14))
g.append(txt(sx(.15), sy(0) + 90, "bearing φi: the angle from the robot's heading to Bi.", INK, 14))
(OUT / "robot_setup.svg").write_text(frame("".join(g), "The setup: pose, heading, beacons and measurements"))

# ---------------- figure 2: predict and residual ----------------
g = world() + beacons()
g += robot(truth, GREY, .25)
g.append(txt(sx(truth[0]) + 14, sy(truth[1]) + 4, "truth (unknown)", GREY, 13))
p = start
for i, b in enumerate(B):
    tip = meas_tip(p, i)
    g.append(line(p[:2], b, BLUE, 2, "7 5"))                      # prediction h(x)
    g.append(line(p[:2], tip, ORANGE, 2.5))                       # what z says, from the guessed pose
    g.append(dot(tip, ORANGE, 5))
    g.append(line(tip, b, RED, 3.5))                              # residual gap
    g.append(txt(sx(tip[0]) + 8, sy(tip[1]) - 8, f"r{i+1}", RED, 15, "start", "bold"))
g += robot(p, BLUE)
g.append(dot(p[:2], BLUE))
g.append(txt(sx(p[0]) - 14, sy(p[1]) + 26, "our guess x", BLUE, 14, "end", "bold"))
leg = [(BLUE, "predicted h(x): where the guess says Bi is"),
       (ORANGE, "measured z: where the sensor says Bi is, from the guess"),
       (RED, "residual r = h(x) - z: the gap")]
for k, (c, s) in enumerate(leg):
    y = sy(0) + 58 + 20 * k
    g.append(f'<rect x="{sx(.2)}" y="{y - 10:.1f}" width="16" height="5" fill="{c}"/>')
    g.append(txt(sx(.2) + 24, y - 3, s, INK, 13))
(OUT / "robot_predict_residual.svg").write_text(frame("".join(g), "Predict, measure, residual"))

# ---------------- figure 3: refinement, 3 panels ----------------
pw = 6.0 * S + 20
panels = []
for k in (0, 1, 2):
    p = path[k]
    g = world() + beacons(False)
    g += robot(truth, GREY, .25)
    for i, b in enumerate(B):
        tip = meas_tip(p, i)
        g += [line(p[:2], tip, ORANGE, 2), dot(tip, ORANGE, 4), line(tip, b, RED, 3.5)]
    g += robot(p, BLUE)
    g.append(txt(sx(.2), sy(0) + 50, f"iteration {k}:  F = {cost(p):.1f}", INK, 15, "start", "bold"))
    g.append(txt(sx(.2), sy(0) + 72, f"position error {np.hypot(*(p[:2] - truth[:2])):.2f} m,"
                                       f"  heading error {abs(np.rad2deg(wrap(p[2] - truth[2]))):.1f}°", INK, 13))
    panels.append("".join(g))
sc = 0.62
inner = "".join(f'<g transform="translate({j * (W * sc + 8):.1f},0) scale({sc})">{pn}</g>' for j, pn in enumerate(panels))
(OUT / "robot_refine.svg").write_text(
    frame(inner, "Refining: pose and heading move so the red gaps shrink", 3 * W * sc + 16, H * sc))
print("start", start, "->", path[-1], "truth", truth)
print([round(float(cost(q)), 2) for q in path])
