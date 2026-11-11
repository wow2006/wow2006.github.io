"""Post 02: gradient descent with a backtracking (Armijo) line search.

Problem: range-only localization from post 01.
Run:  python gradient_descent.py [output_dir]
"""
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
GD_COLOR = "#D55E00"

# --- the problem from post 01 ----------------------------------------------
beacons = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 2.0], [2.5, 2.5]])
p_true = np.array([7.0, 5.0])
rng = np.random.default_rng(7)
ranges = np.linalg.norm(p_true - beacons, axis=1) + rng.normal(0.0, 0.3, len(beacons))


def residuals(p):
    return np.linalg.norm(p - beacons, axis=1) - ranges


def jacobian(p):
    d = p - beacons
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def cost(p):
    r = residuals(p)
    return 0.5 * r @ r


# #region gradient
def gradient(p):
    """g = J^T r  (Omega = I here)."""
    return jacobian(p).T @ residuals(p)
# #endregion gradient


# #region gd
def gradient_descent(cost, gradient, x0, max_iter=1000, c=1e-4, tol_g=1e-8, tol_step=1e-10):
    x = np.asarray(x0, float).copy()
    path, costs = [x.copy()], [cost(x)]
    alpha = 1.0
    for _ in range(max_iter):
        g = gradient(x)
        if np.linalg.norm(g, np.inf) < tol_g:
            break
        F = costs[-1]
        alpha *= 2.0                       # try a bit bigger than last time ...
        while cost(x - alpha * g) > F - c * alpha * (g @ g):   # Armijo test
            alpha *= 0.5                   # ... and halve until F drops enough
        step = -alpha * g
        x = x + step
        path.append(x.copy())
        costs.append(cost(x))
        if np.linalg.norm(step) < tol_step:
            break
    return np.array(path), np.array(costs)
# #endregion gd


x0 = np.array([-2.0, 8.0])
path, costs = gradient_descent(cost, gradient, x0)
ref = least_squares(residuals, x0, xtol=1e-15, ftol=1e-15, gtol=1e-15)
F_star = ref.cost

print(f"start {x0}, F = {costs[0]:.4f}")
print(f"GD: {len(path) - 1} iterations, x = {path[-1].round(6)}, F = {costs[-1]:.8f}")
print(f"scipy least_squares: x = {ref.x.round(6)}, F = {F_star:.8f}")
for k in (0, 1, 2, 5, 10, 50, 100, 200):
    if k < len(costs):
        print(f"  iter {k:4d}: F - F* = {costs[k] - F_star:.3e}")

# #region rate
# Near the minimum F is almost a quadratic with Hessian ~ J^T J. Theory for GD on a quadratic:
# F - F* shrinks per step by about ((kappa - 1) / (kappa + 1))^2.
eig = np.linalg.eigvalsh(jacobian(ref.x).T @ jacobian(ref.x))
kappa = eig[-1] / eig[0]
measured = ((costs[200] - F_star) / (costs[100] - F_star)) ** (1 / 100)
print(f"eig(J^T J) at x* = {eig.round(3)}, kappa = {kappa:.1f}")
print(f"predicted rate ((k-1)/(k+1))^2 = {((kappa - 1) / (kappa + 1)) ** 2:.3f}, measured (iter 100->200) = {measured:.3f}")
# #endregion rate

# Self-checks: analytic gradient vs finite differences, and GD reaches scipy's minimum.
eps = 1e-6
g_fd = np.array([(cost(x0 + eps * e) - cost(x0 - eps * e)) / (2 * eps) for e in np.eye(2)])
assert np.allclose(gradient(x0), g_fd, atol=1e-6)
assert np.allclose(path[-1], ref.x, atol=1e-5)

# --- condition number demo: GD on the linear line fit of posts 00/01 ---------
# #region kappa
rng = np.random.default_rng(42)
t = np.sort(rng.uniform(0.0, 10.0, 20))
z = 0.8 * t + 1.5 + rng.normal(0.0, 0.6, t.size)
print("\nGD on the line fit (tolerance |g| < 1e-8):")
for name, tt in [("raw t", t), ("centered t", t - t.mean())]:
    A = np.column_stack([tt, np.ones_like(tt)])
    kappa = np.linalg.cond(A.T @ A)
    p, _ = gradient_descent(lambda x: 0.5 * np.sum((A @ x - z) ** 2),
                            lambda x: A.T @ (A @ x - z), np.zeros(2), max_iter=100000)
    print(f"  {name:11s} kappa(A^T A) = {kappa:7.1f}   iterations = {len(p) - 1}")
# #endregion kappa

# --- figures ----------------------------------------------------------------
plt.rcParams.update({"font.size": 11})
X, Y = np.meshgrid(np.linspace(-3, 13, 300), np.linspace(-3, 13, 300))
P = np.stack([X, Y], -1)
Fg = 0.5 * ((np.linalg.norm(P[..., None, :] - beacons, axis=-1) - ranges) ** 2).sum(-1)
levels = np.geomspace(F_star * 1.5, Fg.max(), 18)

fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
for ax, (xl, yl) in zip(axes, [((-3, 13), (-3, 13)), ((4.0, 7.5), (4.5, 8.0))]):
    ax.contour(X, Y, Fg, levels=levels, cmap="viridis", linewidths=1)
    ax.plot(path[:, 0], path[:, 1], "-o", color=GD_COLOR, ms=3, lw=1.2, label=f"GD ({len(path) - 1} iterations)")
    ax.plot(beacons[:, 0], beacons[:, 1], "^", color="#555555", ms=9, label="beacons")
    ax.plot(*x0, "ks", ms=8, label="start")
    ax.plot(*ref.x, "k*", ms=13, label="minimum")
    ax.set(xlim=xl, ylim=yl, xlabel="px [m]", ylabel="py [m]", aspect="equal")
axes[0].set_title("Gradient descent path")
axes[1].set_title("Zoom: zig-zag down the valley")
axes[0].legend(loc="upper right", frameon=False)
fig.tight_layout()
fig.savefig(OUT / "gd_path.png", dpi=150)

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.semilogy(np.maximum(costs - F_star, 1e-16), color=GD_COLOR, lw=2, label="GD")
ax.set(xlabel="iteration", ylabel="F(x) - F*  (log scale)", title="Gradient descent: slow, steady (linear) convergence")
ax.grid(alpha=0.3, which="both")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUT / "gd_cost.png", dpi=150)
print("\nsaved gd_path.png, gd_cost.png")
