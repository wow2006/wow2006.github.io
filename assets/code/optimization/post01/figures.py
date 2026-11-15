"""Post 01: two explanatory pictures.

normal_equations.png : why "normal" -- A x* is the projection of b onto the columns of A,
                       and the residual is perpendicular to that plane (post 01a).
linearization.png    : a residual along a line vs its tangent, and the error shrinking
                       like |d|^2 (post 01b). Same beacons and noise as linear_vs_nonlinear.py.
Run:  python figures.py [output_dir]   (numpy + matplotlib only)
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
plt.rcParams.update({"font.size": 11})

# ------------------------------------------------------- normal equations ---
# Three measurements, two unknowns: b lives in 3D, and the columns of A span a plane.
# (Columns chosen for a clear picture; the line fit works the same way with m dimensions.)
A = np.array([[2.0, 0.4], [0.2, 2.0], [0.3, 0.2]])
b = np.array([1.6, 1.4, 2.4])
x_star = np.linalg.solve(A.T @ A, A.T @ b)
Ax = A @ x_star
r = Ax - b
assert np.allclose(A.T @ r, 0)  # the residual is normal to every column

fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(projection="3d")
s, u = np.meshgrid(np.linspace(-0.3, 1.4, 2), np.linspace(-0.3, 1.4, 2))
P = s[..., None] * A[:, 0] + u[..., None] * A[:, 1]
ax.plot_surface(P[..., 0], P[..., 1], P[..., 2], color="#4c72b0", alpha=0.15, edgecolor="#4c72b0", lw=0.5)
o = np.zeros(3)
for v, c, lab, off in [(A[:, 0], "#4c72b0", "column 1 of A", [0.05, -0.35, -0.15]),
                       (A[:, 1], "#55a868", "column 2 of A", [-0.6, 0.15, 0.05]),
                       (b, "k", "b  (measurements)", [-0.9, 0, 0.15]),
                       (Ax, "#dd8452", "A x*  (best fit)", [0.1, 0.05, -0.3])]:
    ax.quiver(*o, *v, color=c, lw=2, arrow_length_ratio=0.08)
    ax.text(*(v + off), lab, color=c)
ax.plot(*np.column_stack([b, Ax]), "--", color="#c44e52", lw=2)
# right-angle mark at A x*: a small square between r and a direction inside the plane
e_r, e_p = -r / np.linalg.norm(r), A[:, 0] / np.linalg.norm(A[:, 0])
sq = Ax + 0.18 * np.array([e_r, e_r + e_p, e_p])
ax.plot(*sq.T, color="#c44e52", lw=1.2)
ax.text(*((b + Ax) / 2 + [0.12, 0, 0]), "residual r = A x* - b\nmeets the plane at 90°", color="#c44e52")
ax.text(*(1.4 * A[:, 0] + 1.4 * A[:, 1] + [0, 0, 0.1]), "plane of all A x", color="#4c72b0")
ax.set_axis_off()
ax.set_box_aspect(np.ptp(np.column_stack([P.reshape(-1, 3).T, b[:, None], o[:, None]]), axis=1))
ax.view_init(elev=14, azim=-70)
fig.tight_layout()
fig.savefig(OUT / "normal_equations.png", dpi=150, bbox_inches="tight")

# ---------------------------------------------------------- linearization ---
beacons = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 2.0], [2.5, 2.5]])
p_true = np.array([7.0, 5.0])
rng = np.random.default_rng(7)
ranges = np.linalg.norm(p_true - beacons, axis=1) + rng.normal(0.0, 0.3, len(beacons))
residuals = lambda p: np.linalg.norm(p - beacons, axis=-1) - ranges
p0, direction = np.array([1.0, 10.0]), np.array([0.6, -0.8])   # as in the taylor check
J = (p0 - beacons) / np.linalg.norm(p0 - beacons, axis=1, keepdims=True)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
steps = np.linspace(-3, 12, 300)
curve = np.array([residuals(p0 + k * direction)[0] for k in steps])
tangent = residuals(p0)[0] + (J[0] @ direction) * steps
a1.plot(steps, curve, color="#4c72b0", lw=2, label="true residual  r1(p + s u)")
a1.plot(steps, tangent, "--", color="#dd8452", lw=2, label="linear model  r1(p) + J1 u s")
a1.fill_between(steps, curve, tangent, color="#c44e52", alpha=0.15, label="error of the linear model")
a1.plot(0, residuals(p0)[0], "ko", ms=7)
a1.annotate("current guess p", (0, residuals(p0)[0]), (2.5, 3), arrowprops={"arrowstyle": "->"})
a1.set(xlabel="step length s along u = (0.6, -0.8)  [m]", ylabel="residual of beacon 1  [m]",
       title="(a) Near p the tangent is good, far away it is not")
a1.legend(frameon=False, loc="lower left")

sizes = np.geomspace(1e-3, 1, 40)
err = [np.linalg.norm(residuals(p0 + k * direction) - (residuals(p0) + J @ (k * direction))) for k in sizes]
marks = [1.0, 0.1, 0.01, 0.001]
err_marks = [np.linalg.norm(residuals(p0 + k * direction) - (residuals(p0) + J @ (k * direction))) for k in marks]
assert abs(err_marks[0] - 4.435e-2) < 1e-4  # same numbers as the post's printed table
a2.loglog(sizes, err, color="#4c72b0", lw=2, label="|r(p+d) - (r(p) + J d)|")
a2.loglog(marks, err_marks, "ko", ms=6, label="values printed in the post")
a2.loglog(sizes, err_marks[0] * sizes ** 2, ":", color="gray", label="slope 2:  10x smaller step, 100x smaller error")
a2.set(xlabel="step size |d|  [m]", ylabel="error  [m]", title="(b) The error shrinks like |d|^2")
a2.legend(frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(OUT / "linearization.png", dpi=150)
print("saved", OUT / "normal_equations.png", "and", OUT / "linearization.png")
