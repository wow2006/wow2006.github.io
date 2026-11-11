"""Post 01: linear vs nonlinear least squares.

Part 1: solve the line fit in one shot (normal equations, Cholesky, QR).
Part 2: range-only localization, a nonlinear problem with no closed form.
Run:  python linear_vs_nonlinear.py [output_dir]
"""
import sys
from pathlib import Path

import numpy as np
import scipy.linalg
from scipy.optimize import least_squares
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")

# ---------------------------------------------------------------- part 1 ---
# Same data as post 00.
rng = np.random.default_rng(42)
t = np.sort(rng.uniform(0.0, 10.0, 20))
z = 0.8 * t + 1.5 + rng.normal(0.0, 0.6, t.size)

# #region linear
# r(x) = A x - b  with  x = (a, b_line),  row i of A = [t_i, 1],  b = z.
A = np.column_stack([t, np.ones_like(t)])
b = z

# 1) Normal equations, solved with a Cholesky factorization of A^T A.
AtA = A.T @ A
Atb = A.T @ b
x_chol = scipy.linalg.cho_solve(scipy.linalg.cho_factor(AtA), Atb)

# 2) QR: A = Q R, then R x = Q^T b. Never forms A^T A.
Q, R = np.linalg.qr(A)
x_qr = scipy.linalg.solve_triangular(R, Q.T @ b)
# #endregion linear

x_ref = np.linalg.lstsq(A, b, rcond=None)[0]
print("line fit, Cholesky :", x_chol.round(6))
print("line fit, QR       :", x_qr.round(6))
print("numpy lstsq        :", x_ref.round(6))
print(f"cond(A)      = {np.linalg.cond(A):.1f}")
print(f"cond(A^T A)  = {np.linalg.cond(AtA):.1f}   (= cond(A)^2 = {np.linalg.cond(A)**2:.1f})")
assert np.allclose(x_chol, x_ref, atol=1e-10) and np.allclose(x_qr, x_ref, atol=1e-10)

# ---------------------------------------------------------------- part 2 ---
# #region range
beacons = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 2.0], [2.5, 2.5]])  # known positions
p_true = np.array([7.0, 5.0])                                          # robot, unknown to us
rng = np.random.default_rng(7)
ranges = np.linalg.norm(p_true - beacons, axis=1) + rng.normal(0.0, 0.3, len(beacons))


def residuals(p):
    """r_i(p) = ||p - b_i|| - z_i : predicted distance minus measured distance."""
    return np.linalg.norm(p - beacons, axis=1) - ranges


def jacobian(p):
    """Row i is d r_i / d p = (p - b_i)^T / ||p - b_i||, a unit vector."""
    d = p - beacons
    return d / np.linalg.norm(d, axis=1, keepdims=True)
# #endregion range


def cost(p):
    r = residuals(p)
    return 0.5 * r @ r


print("\nranges z_i =", ranges.round(3))

# Self-check: analytic Jacobian vs central finite differences.
p0 = np.array([1.0, 10.0])
eps = 1e-6
J_fd = np.column_stack([(residuals(p0 + eps * e) - residuals(p0 - eps * e)) / (2 * eps) for e in np.eye(2)])
assert np.allclose(jacobian(p0), J_fd, atol=1e-8)

# #region taylor
# How good is the linear model r(p + d) ~ r(p) + J d ?  The error should shrink like |d|^2.
print("\n|d|      |r(p+d) - (r(p) + J d)|")
direction = np.array([0.6, -0.8])
for s in [1.0, 0.1, 0.01, 0.001]:
    d = s * direction
    err = np.linalg.norm(residuals(p0 + d) - (residuals(p0) + jacobian(p0) @ d))
    print(f"{s:<8} {err:.3e}")
# #endregion taylor

# ---------------------------------------------------------------- figure ---
plt.rcParams.update({"font.size": 11})
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

# (a) linear: the cost over (a, b) is a quadratic bowl.
aa, bb = np.meshgrid(np.linspace(0.4, 1.2, 200), np.linspace(-1.0, 4.0, 200))
r_grid = aa[..., None] * t + bb[..., None] - z
F_lin = 0.5 * (r_grid ** 2).sum(-1)
ax = axes[0]
ax.contour(aa, bb, F_lin, levels=np.geomspace(F_lin.min() * 1.05, F_lin.max(), 15), cmap="viridis", linewidths=1)
ax.plot(*x_ref, "k*", ms=12, label="minimum")
ax.set(xlabel="slope a", ylabel="intercept b", title="(a) Linear: line fit, one bowl")
ax.legend(loc="upper right", frameon=False)


def range_cost_grid(bcn, rng_meas, X, Y):
    P = np.stack([X, Y], -1)
    r = np.linalg.norm(P[..., None, :] - bcn, axis=-1) - rng_meas
    return 0.5 * (r ** 2).sum(-1)


def draw_range_panel(ax, bcn, rng_meas, title, mins, ylim=(-3, 13)):
    X, Y = np.meshgrid(np.linspace(-3, 13, 300), np.linspace(*ylim, 300))
    Fg = range_cost_grid(bcn, rng_meas, X, Y)
    ax.contour(X, Y, Fg, levels=np.geomspace(max(Fg.min(), 1e-3) * 1.5, Fg.max(), 18), cmap="viridis", linewidths=1)
    ax.plot(bcn[:, 0], bcn[:, 1], "^", color="#555555", ms=9, label="beacons")
    for m in mins:
        ax.plot(*m, "k*", ms=12)
    ax.plot([], [], "k*", ms=12, label="minimum" if len(mins) == 1 else "minima")
    ax.set(xlabel="px [m]", ylabel="py [m]", title=title, aspect="equal")
    ax.legend(loc="upper left", frameon=False)


# (b) nonlinear, 4 beacons: one minimum, but a curved, stretched valley.
p_min = least_squares(residuals, p0, xtol=1e-14, ftol=1e-14, gtol=1e-14).x
draw_range_panel(axes[1], beacons, ranges, "(b) Nonlinear: 4 beacons, curved valley", [p_min])

# (c) nonlinear, only 2 beacons: two circles cross twice, so two perfect answers.
b2 = beacons[[0, 1]]
r2 = ranges[[0, 1]]
mins2 = [least_squares(lambda p: np.linalg.norm(p - b2, axis=1) - r2, s).x for s in ([5, 8], [5, -2])]
draw_range_panel(axes[2], b2, r2, "(c) Nonlinear: 2 beacons, two minima", mins2, ylim=(-8, 8))

fig.tight_layout()
fig.savefig(OUT / "cost_contours.png", dpi=150)
print("\nrange-only minimum (4 beacons):", p_min.round(4), " F =", f"{cost(p_min):.5f}")
print("two-beacon minima:", [m.round(3) for m in mins2])
print("saved", OUT / "cost_contours.png")
