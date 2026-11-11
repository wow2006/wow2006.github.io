"""Post 00: the least-squares problem on the smallest example we know.

Fit a line y = a*x + b to noisy points and draw the residuals.
Run:  python line_fit.py [output_dir]
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")

# #region data
rng = np.random.default_rng(42)            # fixed seed: same points every run
x_true = np.array([0.8, 1.5])              # true slope a and intercept b
t = np.sort(rng.uniform(0.0, 10.0, 20))    # where we measure
z = x_true[0] * t + x_true[1] + rng.normal(0.0, 0.6, t.size)  # observations z_i
# #endregion data


# #region model
def h(x, t):
    """Measurement model: what we expect to measure at t if the state is x."""
    a, b = x
    return a * t + b


def residuals(x):
    """r_i(x) = h_i(x) - z_i  (predicted minus measured)."""
    return h(x, t) - z


def cost(x):
    """F(x) = 1/2 * sum r_i^2  (all measurements equally trusted, Omega = I)."""
    r = residuals(x)
    return 0.5 * r @ r
# #endregion model


# The best x. Post 01 derives how to get it; here we only need the answer.
A = np.column_stack([t, np.ones_like(t)])
x_best = np.linalg.lstsq(A, z, rcond=None)[0]

print(f"true x  = {x_true}")
print(f"best x  = {x_best.round(4)}")
print(f"F(true) = {cost(x_true):.4f}")
print(f"F(best) = {cost(x_best):.4f}")

# Self-checks.
assert cost(x_best) <= cost(x_true)                     # the best line beats the true one on this data
for d in ([1e-3, 0], [0, 1e-3], [-1e-3, 0], [0, -1e-3]):
    assert cost(x_best + np.array(d)) > cost(x_best)    # any small move makes F worse
assert abs(residuals(x_best).sum()) < 1e-9              # with an intercept, residuals sum to zero

# Figure: points, fitted line, residuals as vertical segments.
plt.rcParams.update({"font.size": 11})
fig, ax = plt.subplots(figsize=(7, 4.2))
tt = np.linspace(0, 10, 2)
ax.vlines(t, z, h(x_best, t), color="#CC79A7", lw=1.5, label="residual $r_i = h_i(x) - z_i$")
ax.plot(tt, h(x_best, tt), color="#333333", lw=2, label=f"fitted line  a={x_best[0]:.2f}, b={x_best[1]:.2f}")
ax.plot(t, z, "o", color="#777777", ms=6, label="observations $z_i$")
ax.set_xlabel("t (where we measure)")
ax.set_ylabel("measured value")
ax.set_title("Least squares: make the pink segments small")
ax.grid(alpha=0.3)
ax.legend(loc="upper left", frameon=False)
fig.tight_layout()
fig.savefig(OUT / "line_fit_residuals.png", dpi=150)
print("saved", OUT / "line_fit_residuals.png")


# Vocabulary figures: the same example, with every word of the post pointed at.
from matplotlib.patches import FancyBboxPatch

OI = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
      "blue": "#0072B2", "red": "#D55E00", "pink": "#CC79A7"}   # Okabe-Ito, colour-blind safe
BOX = dict(boxstyle="round,pad=0.4", fc="white", ec="#888888")
ARROW = dict(arrowstyle="->", color="#444444", lw=1.3)


def spring(ax, t0, y0, y1, color, lw):
    """Zig-zag from (t0, y0) to (t0, y1): a residual drawn as a vertical spring."""
    coils = max(2, int(abs(y1 - y0) * 6))
    width = min(0.14, 0.4 * abs(y1 - y0))               # short springs stay thin
    y = np.linspace(y0, y1, 2 * coils + 3)
    dx = np.r_[0, np.tile([-width, width], coils), 0, 0]
    ax.plot(t0 + dx, y, color=color, lw=lw)


def plot_vocabulary_annotated(path):
    plt.rcParams.update({"font.size": 12})
    fig = plt.figure(figsize=(7.5, 8.4))
    ax = fig.add_axes([0.1, 0.3, 0.87, 0.64])
    r = residuals(x_best)
    k = int(np.argmax(np.abs(r)))                      # the residual we point at
    for i in range(t.size):
        spring(ax, t[i], z[i], h(x_best, t[i]), OI["red"] if i == k else OI["pink"], 2.4 if i == k else 1.4)
    tt = np.linspace(0, 10, 2)
    ax.plot(tt, h(x_best, tt), color=OI["blue"], lw=2.5)
    ax.plot(t, z, "o", color=OI["black"], ms=6, zorder=3)
    ax.set(xlim=(-0.3, 10.3), ylim=(-0.5, 12.5), xlabel="t (where we measure)", ylabel="measured value",
           title="The words of least squares, on the line-fit example")
    ax.grid(alpha=0.3)

    ax.text(0.1, 12.1, "variables x = (a, b) = (slope, intercept)\n"
            f"the knobs we turn; best: a = {x_best[0]:.2f}, b = {x_best[1]:.2f}", va="top", bbox=BOX)
    ax.annotate("model $h_i(x) = a\\,t_i + b$\nwhat we predict", xy=(3.0, h(x_best, 3.0)), xytext=(0.1, 8.0),
                va="top", bbox=BOX, arrowprops=ARROW)
    j = 11                                             # a point below the line
    ax.annotate("observations $z_i$\nwhat we measured", xy=(t[j], z[j]), xytext=(3.2, 1.3),
                va="center", bbox=BOX, arrowprops=ARROW)
    ym = 0.5 * (z[k] + h(x_best, t[k]))
    ax.annotate("residual $r_i = h_i(x) - z_i$\nthe gap", xy=(t[k] + 0.2, ym), xytext=(10.1, 11.6),
                ha="right", va="top", bbox=BOX, color=OI["red"], arrowprops=dict(ARROW, color=OI["red"]))
    ax.text(10.1, -0.1, "cost $F = \\frac{1}{2} \\sum\\, r_i^2$\ntotal spring energy\n"
            f"= {cost(x_best):.4f} here", ha="right", va="bottom", bbox=dict(BOX, ec=OI["red"]))

    fig.text(0.04, 0.215, "Springs: least squares = let the springs pull the line until it\n"
             "rests. A spring stretched by $r_i$ stores $\\frac{1}{2} r_i^2$ of energy, so the line\n"
             "rests where the total energy $F$ is smallest.", va="top")
    fig.text(0.04, 0.115, "Soft constraint: every observation (spring) pulls a little.\n"
             "Hard constraint: must hold exactly, e.g. a pin forcing the line\nthrough a point.", va="top")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("saved", path)


def plot_vocabulary_loop(path):
    plt.rcParams.update({"font.size": 12})
    fig, ax = plt.subplots(figsize=(7.5, 7.8))
    ax.set(xlim=(-0.1, 10.1), ylim=(0, 10.3))
    ax.axis("off")

    def box(cx, cy, w, hgt, title, words, ec="#444444", ls="-"):
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - hgt / 2), w, hgt, boxstyle="round,pad=0.05",
                                    fc="white", ec=ec, lw=2, ls=ls))
        ax.text(cx, cy + hgt / 2 - 0.2, title, ha="center", va="top", fontsize=13, fontweight="bold")
        ax.text(cx, cy + hgt / 2 - 0.75, words, ha="center", va="top", color="#333333", style="italic")

    def arrow(p, q, color="#444444"):
        ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", color=color, lw=2, mutation_scale=20))

    box(8.5, 9.0, 2.9, 1.4, "observation $z_i$", "what the sensor read")
    box(1.5, 6.6, 2.9, 1.6, "variables $x$", "the knobs we turn")
    box(5.0, 6.6, 2.9, 1.6, "model $h_i(x)$", "turns x into a\nprediction")
    box(8.5, 6.6, 2.9, 1.6, "residual $r_i$", "$h_i(x) - z_i$: prediction\nminus measurement")
    box(8.5, 3.6, 2.9, 1.8, "cost $F$", "$\\frac{1}{2} \\sum\\, r_i^2$: one number,\nhow bad this x is")
    box(3.25, 3.6, 6.3, 1.8, "optimizer", "changes x to make F smaller;\nthis is what GD / GN / LM do (posts 02–04)",
        ec=OI["blue"])
    box(5.0, 1.0, 9.9, 1.7, "constraints", "soft: residual terms in F; the optimizer may break them a little\n"
        "hard: must hold exactly, e.g. fix the first pose (post 05)", ec=OI["orange"], ls="--")

    arrow((8.5, 8.3), (8.5, 7.4))                      # z joins
    arrow((2.95, 6.6), (3.55, 6.6))                    # x -> model
    arrow((6.45, 6.6), (7.05, 6.6))                    # prediction -> residual
    arrow((8.5, 5.8), (8.5, 4.5))                      # residuals -> cost
    ax.text(8.3, 5.15, "square and add", ha="right", va="center")
    arrow((7.05, 3.6), (6.4, 3.6))                     # cost -> optimizer
    arrow((1.5, 4.5), (1.5, 5.8), OI["blue"])          # new x, and around again
    ax.text(1.7, 5.15, "new x, repeat", va="center", color=OI["blue"])
    ax.set_title("The least-squares loop", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("saved", path)


plot_vocabulary_annotated(OUT / "vocabulary_annotated.png")
plot_vocabulary_loop(OUT / "vocabulary_loop.png")
