"""Post 03: Gauss-Newton.

Problem: range-only localization from posts 01-02.
Case A: clean ranges, GN converges in a few steps.
Case B: one range is 4 m too long (a bounced signal), GN runs away.
Run:  python gauss_newton.py [output_dir]
"""
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
GN_COLOR = "#0072B2"

# --- the problem from post 01 ----------------------------------------------
beacons = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 2.0], [2.5, 2.5]])
p_true = np.array([7.0, 5.0])
rng = np.random.default_rng(7)
ranges_clean = np.linalg.norm(p_true - beacons, axis=1) + rng.normal(0.0, 0.3, len(beacons))
ranges_bad = ranges_clean.copy()
ranges_bad[0] += 4.0     # case B: beacon 0's signal bounced off a wall, 4 m too long


def make_problem(ranges):
    def residuals(p):
        return np.linalg.norm(p - beacons, axis=1) - ranges

    def jacobian(p):
        d = p - beacons
        return d / np.linalg.norm(d, axis=1, keepdims=True)

    def cost(p):
        r = residuals(p)
        return 0.5 * r @ r

    return residuals, jacobian, cost


# #region gn
def gauss_newton(residuals, jacobian, x0, max_iter=50, tol_g=1e-8, tol_step=1e-10):
    x = np.asarray(x0, float).copy()
    r = residuals(x)
    path, costs = [x.copy()], [0.5 * r @ r]
    for _ in range(max_iter):
        J = jacobian(x)
        g = J.T @ r                       # gradient
        if np.linalg.norm(g, np.inf) < tol_g:
            break
        H = J.T @ J                       # Gauss-Newton Hessian approximation
        delta = np.linalg.solve(H, -g)    # the normal equations of the linearized problem
        x = x + delta                     # pure GN: take the full step, no questions asked
        r = residuals(x)
        path.append(x.copy())
        costs.append(0.5 * r @ r)
        if np.linalg.norm(delta) < tol_step or not np.isfinite(costs[-1]):
            break
    return np.array(path), np.array(costs)
# #endregion gn


# #region hessian_terms
def hessian_parts(residuals, jacobian, p, eps=1e-5):
    """Split the true Hessian of F into J^T J and the dropped term S = sum r_i * Hess(r_i)."""
    J = jacobian(p)
    grad = lambda q: jacobian(q).T @ residuals(q)
    H_true = np.column_stack([(grad(p + eps * e) - grad(p - eps * e)) / (2 * eps) for e in np.eye(2)])
    return J.T @ J, H_true - J.T @ J
# #endregion hessian_terms


x0 = np.array([-2.0, 8.0])
results = {}
for name, ranges in [("A (clean)", ranges_clean), ("B (one bad range)", ranges_bad)]:
    residuals, jacobian, cost = make_problem(ranges)
    path, costs = gauss_newton(residuals, jacobian, x0)
    ref = least_squares(residuals, x0, xtol=1e-15, ftol=1e-15, gtol=1e-15)
    JtJ, S = hessian_parts(residuals, jacobian, ref.x)
    # Near the minimum, GN behaves like x_{k+1} - x* = -(J^T J)^{-1} S (x_k - x*).
    rate = np.max(np.abs(np.linalg.eigvals(np.linalg.solve(JtJ, S))))
    results[name] = (path, costs, ref, residuals, cost)
    print(f"\ncase {name}")
    print(f"  GN: {len(path) - 1} iterations, final x = {path[-1].round(4)}, F = {costs[-1]:.6g}")
    print(f"  scipy least_squares: x = {ref.x.round(4)}, F* = {ref.cost:.6g}")
    print(f"  at x*: eig(J^T J) = {np.linalg.eigvalsh(JtJ).round(3)}, |S| = {np.linalg.norm(S, 2):.3f}, "
          f"GN local rate = {rate:.3f}  (< 1 needed, near 0 is fast)")
    print("  cost per iteration:", np.array2string(costs[:15], precision=4, max_line_width=200))

# Self-checks: in case A, GN lands on scipy's minimum; in case B, GN runs away from it.
pA, cA, refA, _, _ = results["A (clean)"]
assert np.allclose(pA[-1], refA.x, atol=1e-6) and abs(cA[-1] - refA.cost) < 1e-12
pB, cB, refB, _, _ = results["B (one bad range)"]
assert cB[-1] > 100 * refB.cost

# --- figures ----------------------------------------------------------------
plt.rcParams.update({"font.size": 11})


def contour(ax, ranges, F_star, lim):
    X, Y = np.meshgrid(np.linspace(*lim[0], 400), np.linspace(*lim[1], 400))
    P = np.stack([X, Y], -1)
    Fg = 0.5 * ((np.linalg.norm(P[..., None, :] - beacons, axis=-1) - ranges) ** 2).sum(-1)
    ax.contour(X, Y, Fg, levels=np.geomspace(F_star * 1.2, Fg.max(), 20), cmap="viridis", linewidths=1)
    ax.plot(beacons[:, 0], beacons[:, 1], "^", color="#555555", ms=9, label="beacons")


fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))
ax = axes[0]
contour(ax, ranges_clean, refA.cost, ((-3, 13), (-3, 13)))
ax.plot(pA[:, 0], pA[:, 1], "-s", color=GN_COLOR, ms=5, lw=1.5, label=f"GN ({len(pA) - 1} iterations)")
ax.plot(*x0, "ks", ms=8, label="start")
ax.plot(*refA.x, "k*", ms=13, label="minimum")
ax.set(xlim=(-3, 13), ylim=(-3, 13), xlabel="px [m]", ylabel="py [m]", aspect="equal",
       title="A: clean ranges, GN converges")
ax.legend(loc="upper right", frameon=False)

ax = axes[1]
lim = ((-5, 30), (-5, 30))
contour(ax, ranges_bad, refB.cost, lim)
k = np.argmax(np.abs(pB).max(1) > 30)          # first iterate that leaves the plot
shown = pB[: k + 1] if k else pB
ax.plot(shown[:, 0], shown[:, 1], "-s", color=GN_COLOR, ms=5, lw=1.5, label="GN")
for i in range(1, min(k, 6)):
    ax.annotate(str(i), shown[i], textcoords="offset points", xytext=(6, 4), fontsize=9)
ax.plot(*x0, "ks", ms=8, label="start")
ax.plot(*refB.x, "k*", ms=13, label="minimum")
ax.text(0.98, 0.02, f"after {len(pB) - 1} steps: x = ({pB[-1, 0]:.1e}, {pB[-1, 1]:.1e}) m\nF = {cB[-1]:.1e}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=10,
        bbox=dict(facecolor="white", edgecolor="#999999", boxstyle="round"))
ax.set(xlim=lim[0], ylim=lim[1], xlabel="px [m]", ylabel="py [m]", aspect="equal",
       title="B: one range 4 m too long, GN runs away")
ax.legend(loc="upper right", frameon=False)
fig.tight_layout()
fig.savefig(OUT / "gn_path.png", dpi=150)

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.semilogy(np.maximum(cA - refA.cost, 1e-16), "-s", color=GN_COLOR, lw=2, ms=5, label="GN, case A")
ax.set(xlabel="iteration", ylabel="F(x) - F*  (log scale)", title="Gauss-Newton on the clean problem")
ax.grid(alpha=0.3, which="both")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUT / "gn_cost.png", dpi=150)
print("\nsaved gn_path.png, gn_cost.png")
