"""Post 04: Levenberg-Marquardt, and GD vs GN vs LM on the range-only problem.

LM uses Nielsen's damping update, from Madsen, Nielsen & Tingleff,
"Methods for Non-Linear Least Squares Problems" (2nd ed., 2004), Algorithm 3.16.
Run:  python levenberg_marquardt.py [output_dir]
"""
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
COLORS = {"GD": "#D55E00", "GN": "#0072B2", "LM": "#009E73"}
MARKERS = {"GD": "o", "GN": "s", "LM": "^"}

# --- the problem from posts 01-03 -------------------------------------------
beacons = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 2.0], [2.5, 2.5]])
p_true = np.array([7.0, 5.0])
rng = np.random.default_rng(7)
ranges_clean = np.linalg.norm(p_true - beacons, axis=1) + rng.normal(0.0, 0.3, len(beacons))
ranges_bad = ranges_clean.copy()
ranges_bad[0] += 4.0


def make_problem(ranges):
    def residuals(p):
        return np.linalg.norm(p - beacons, axis=1) - ranges

    def jacobian(p):
        d = p - beacons
        return d / np.linalg.norm(d, axis=1, keepdims=True)

    return residuals, jacobian


# --- the three solvers, same stopping rules ---------------------------------
TOL_G, TOL_STEP = 1e-8, 1e-10


def gradient_descent(residuals, jacobian, x0, max_iter=1000, c=1e-4):
    cost = lambda x: 0.5 * np.sum(residuals(x) ** 2)
    x = np.asarray(x0, float).copy()
    path, costs, alpha = [x.copy()], [cost(x)], 1.0
    for _ in range(max_iter):
        g = jacobian(x).T @ residuals(x)
        if np.linalg.norm(g, np.inf) < TOL_G:
            break
        alpha *= 2.0
        while cost(x - alpha * g) > costs[-1] - c * alpha * (g @ g):
            alpha *= 0.5
        x = x - alpha * g
        path.append(x.copy())
        costs.append(cost(x))
        if alpha * np.linalg.norm(g) < TOL_STEP:
            break
    return np.array(path), np.array(costs)


def gauss_newton(residuals, jacobian, x0, max_iter=50):
    x = np.asarray(x0, float).copy()
    r = residuals(x)
    path, costs = [x.copy()], [0.5 * r @ r]
    for _ in range(max_iter):
        J = jacobian(x)
        g = J.T @ r
        if np.linalg.norm(g, np.inf) < TOL_G:
            break
        delta = np.linalg.solve(J.T @ J, -g)
        x = x + delta
        r = residuals(x)
        path.append(x.copy())
        costs.append(0.5 * r @ r)
        if np.linalg.norm(delta) < TOL_STEP or not np.isfinite(costs[-1]):
            break
    return np.array(path), np.array(costs)


# #region lm
def levenberg_marquardt(residuals, jacobian, x0, max_iter=200, tau=1e-3):
    x = np.asarray(x0, float).copy()
    r = residuals(x)
    J = jacobian(x)
    H, g, F = J.T @ J, J.T @ r, 0.5 * r @ r
    lam = tau * H.diagonal().max()        # start damping relative to the size of H
    nu = 2.0
    path, costs, lams = [x.copy()], [F], [lam]
    for _ in range(max_iter):
        if np.linalg.norm(g, np.inf) < TOL_G:
            break
        delta = np.linalg.solve(H + lam * np.eye(len(x)), -g)
        if np.linalg.norm(delta) < TOL_STEP:
            break
        x_new = x + delta
        r_new = residuals(x_new)
        F_new = 0.5 * r_new @ r_new
        predicted = 0.5 * delta @ (lam * delta - g)   # L(0) - L(delta), always > 0
        rho = (F - F_new) / predicted                 # gain ratio
        if rho > 0:                                   # cost went down: accept
            x, r, F = x_new, r_new, F_new
            J = jacobian(x)
            H, g = J.T @ J, J.T @ r
            lam *= max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3)
            nu = 2.0
            path.append(x.copy())
        else:                                         # cost went up: reject, damp more
            lam *= nu
            nu *= 2.0
        costs.append(F)
        lams.append(lam)
    return np.array(path), np.array(costs), np.array(lams)
# #endregion lm


x0 = np.array([-2.0, 8.0])
residuals, jacobian = make_problem(ranges_clean)
ref = least_squares(residuals, x0, method="lm", xtol=1e-15, ftol=1e-15, gtol=1e-15)
F_star = ref.cost

runs = {
    "GD": gradient_descent(residuals, jacobian, x0),
    "GN": gauss_newton(residuals, jacobian, x0),
    "LM": levenberg_marquardt(residuals, jacobian, x0),
}
print(f"start {x0}, F = {runs['GD'][1][0]:.4f};  scipy (MINPACK lm) F* = {F_star:.10f}")
for name, out in runs.items():
    path, costs = out[0], out[1]
    print(f"  {name}: {len(costs) - 1:4d} iterations ({len(path) - 1} accepted steps), "
          f"x = {path[-1].round(6)}, F - F* = {costs[-1] - F_star:.1e}")
    assert np.allclose(path[-1], ref.x, atol=1e-5)          # self-check: all reach scipy's minimum
lm_path, lm_costs, lm_lams = runs["LM"]
print("  LM lambda per iteration:", np.array2string(lm_lams, precision=3, max_line_width=200))
print("  LM cost per iteration:  ", np.array2string(lm_costs, precision=4, max_line_width=200))

# Self-check: the two limits of the damped system.
J, r = jacobian(x0), residuals(x0)
H, g = J.T @ J, J.T @ r
step = lambda lam: np.linalg.solve(H + lam * np.eye(2), -g)
assert np.allclose(step(1e-12), np.linalg.solve(H, -g), rtol=1e-6)     # lambda -> 0: Gauss-Newton
assert np.allclose(1e8 * step(1e8), -g, rtol=1e-6)                     # lambda -> inf: -g / lambda
print("  limits OK: lambda->0 gives the GN step, lambda->inf gives -g/lambda")

# Case B from post 03: one range 4 m too long.
res_b, jac_b = make_problem(ranges_bad)
ref_b = least_squares(res_b, x0, method="lm", xtol=1e-15, ftol=1e-15, gtol=1e-15)
pb, cb, _ = levenberg_marquardt(res_b, jac_b, x0)
print(f"\ncase B (bad range): LM {len(cb) - 1} iterations ({len(pb) - 1} accepted), "
      f"x = {pb[-1].round(4)}, F = {cb[-1]:.6f}, scipy F* = {ref_b.cost:.6f}")
assert np.allclose(pb[-1], ref_b.x, atol=1e-5)

# --- figures ----------------------------------------------------------------
plt.rcParams.update({"font.size": 11})
X, Y = np.meshgrid(np.linspace(-3, 13, 300), np.linspace(-3, 13, 300))
P = np.stack([X, Y], -1)
Fg = 0.5 * ((np.linalg.norm(P[..., None, :] - beacons, axis=-1) - ranges_clean) ** 2).sum(-1)

fig, ax = plt.subplots(figsize=(7, 6.4))
ax.contour(X, Y, Fg, levels=np.geomspace(F_star * 1.5, Fg.max(), 18), cmap="Greys", linewidths=0.8)
for name, out in runs.items():
    p = out[0]
    ax.plot(p[:, 0], p[:, 1], "-" + MARKERS[name], color=COLORS[name], ms=4, lw=1.4,
            label=f"{name} ({len(out[1]) - 1} iterations)")
ax.plot(beacons[:, 0], beacons[:, 1], "^", color="#555555", ms=9, label="beacons")
ax.plot(*x0, "ks", ms=8, label="start")
ax.plot(*ref.x, "k*", ms=13, label="minimum")
ax.set(xlim=(-3, 13), ylim=(-3, 13), xlabel="px [m]", ylabel="py [m]", aspect="equal",
       title="Same start, three methods")
ax.legend(loc="upper right", frameon=False, fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "all_paths.png", dpi=150)

fig, ax = plt.subplots(figsize=(7.5, 4.4))
for name, out in runs.items():
    c = out[1]
    ax.semilogy(np.arange(len(c)), np.maximum(c - F_star, 1e-16), "-" + MARKERS[name], color=COLORS[name],
                ms=4, lw=1.8, markevery=1 if len(c) < 60 else 10, label=name)
ax.set(xlabel="iteration", ylabel="F(x) - F*  (log scale)", title="Cost vs iteration, same start")
ax.grid(alpha=0.3, which="both")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUT / "all_costs.png", dpi=150)

# --- the LM step as a function of lambda, at the start x0 ---------------------
J0, r0 = jacobian(x0), residuals(x0)
H0, g0 = J0.T @ J0, J0.T @ r0
lm_step = lambda lam: np.linalg.solve(H0 + lam * np.eye(2), -g0)
L0 = lambda d: 0.5 * np.sum((r0 + J0 @ d) ** 2)            # the linear model of the cost
lam_grid = np.geomspace(1e-4, 1e4, 400)
ends = np.array([x0 + lm_step(lam) for lam in lam_grid])
assert np.all(np.diff(np.linalg.norm(ends - x0, axis=1)) < 0)   # larger lambda, shorter step
d_gn = np.linalg.solve(H0, -g0)

# (a) trust region: the step for one lambda is the model minimum inside a circle.
lam_tr = lm_lams[3]                                         # the first lambda LM accepted (post text)
d_tr = lm_step(lam_tr)
radius = np.linalg.norm(d_tr)
phi = np.linspace(0, 2 * np.pi, 3600)
circle = radius * np.stack([np.cos(phi), np.sin(phi)], 1)
assert L0(d_tr) <= min(L0(d) for d in circle) + 1e-9       # self-check: best model point on the circle
Lg = 0.5 * (((P - x0) @ J0.T + r0) ** 2).sum(-1)          # model L on the grid, centred at x0

fig, ax = plt.subplots(figsize=(7, 6.4))
ax.contour(X, Y, Fg, levels=np.geomspace(F_star * 1.5, Fg.max(), 18), cmap="Greys", linewidths=0.8)
ax.contour(X, Y, Lg, levels=sorted([L0(d_tr), L0(0.5 * d_gn), L0(0.8 * d_gn)]), colors=COLORS["LM"],
           linewidths=0.8, linestyles="dotted")
ax.plot(*(x0 + circle).T, "--", color="#333333", lw=1.4, label=f"trusted region, radius {radius:.1f} m")
ax.annotate("", xy=x0 + d_gn, xytext=x0, arrowprops=dict(arrowstyle="->", color=COLORS["GN"], lw=1.8))
ax.annotate("", xy=x0 + d_tr, xytext=x0, arrowprops=dict(arrowstyle="->", color=COLORS["LM"], lw=2.2))
ax.plot([], [], color=COLORS["GN"], lw=1.8, label="GN step (model minimum, no limit)")
ax.plot([], [], color=COLORS["LM"], lw=2.2, label=f"LM step, lambda = {lam_tr:.2f}")
ax.plot([], [], ":", color=COLORS["LM"], lw=0.8, label="contours of the linear model")
ax.plot(beacons[:, 0], beacons[:, 1], "^", color="#555555", ms=9, label="beacons")
ax.plot(*x0, "ks", ms=8, label="current point")
ax.plot(*ref.x, "k*", ms=13, label="true minimum")
ax.set(xlim=(-3, 13), ylim=(-3, 13), xlabel="px [m]", ylabel="py [m]", aspect="equal",
       title="Trust-region view of one LM step")
ax.legend(loc="lower right", frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "trust_region.png", dpi=150)

# (b) the lambda dial: where the step ends, for every lambda.
fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))
g_dir = -g0 / np.linalg.norm(g0)
zoom = (-2.2, -0.4, 6.9, 8.7)                               # the box around x0 shown on the right
for ax, labelled, arrow, off in [(axes[0], (1e-2, 1e-1, 1.0), 2.5, (8, -14)),
                                 (axes[1], (1.0, 3.0, 10.0, 30.0, 100.0), 1.2, (7, 4))]:
    ax.contour(X, Y, Fg, levels=np.geomspace(F_star * 1.5, Fg.max(), 18), cmap="Greys", linewidths=0.8)
    ax.plot(*ends.T, "-", color=COLORS["LM"], lw=2.2, label="end of the LM step, lambda from 1e-4 to 1e4")
    for lam in labelled:
        e = x0 + lm_step(lam)
        ax.plot(*e, "o", color=COLORS["LM"], ms=6)
        ax.annotate(f"lambda = {lam:g}", e, xytext=off, textcoords="offset points", fontsize=9)
    ax.annotate("", xy=x0 + arrow * g_dir, xytext=x0,
                arrowprops=dict(arrowstyle="->", color=COLORS["GD"], lw=1.8, ls="--"))
    ax.plot([], [], "--", color=COLORS["GD"], lw=1.8, label="direction -g (steepest descent)")
    ax.plot(*x0, "ks", ms=8, label="current point")
axes[0].plot(*(x0 + d_gn), "s", color=COLORS["GN"], ms=8, label="GN step (lambda = 0)")
axes[0].plot(beacons[:, 0], beacons[:, 1], "^", color="#555555", ms=9, label="beacons")
axes[0].plot(*ref.x, "k*", ms=13, label="true minimum")
axes[0].add_patch(plt.Rectangle(zoom[::2], zoom[1] - zoom[0], zoom[3] - zoom[2], fill=False, ls=":", lw=1))
axes[0].set(xlim=(-3, 13), ylim=(-3, 13), xlabel="px [m]", ylabel="py [m]", aspect="equal",
            title="The lambda dial: one start, every lambda")
axes[0].legend(loc="lower right", frameon=False, fontsize=9)
axes[1].set(xlim=zoom[:2], ylim=zoom[2:], xlabel="px [m]", aspect="equal",
            title="Zoom: large lambda, short step along -g")
fig.tight_layout()
fig.savefig(OUT / "lambda_dial.png", dpi=150)
print("\nsaved all_paths.png, all_costs.png, trust_region.png, lambda_dial.png")
