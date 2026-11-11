"""Post 05: a 2D (SE(2)) pose graph, solved with GD, GN and LM.

Builds a synthetic Manhattan-style trajectory, adds noisy odometry and loop-closure
edges, starts from chained odometry, and compares the three methods with the same
stopping rules. Checks the analytic Jacobian against finite differences and the
final cost against scipy.optimize.least_squares.
Run:  python pose_graph.py [output_dir]
"""
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import least_squares
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
COLORS = {"GD": "#D55E00", "GN": "#0072B2", "LM": "#009E73"}
MARKERS = {"GD": "o", "GN": "s", "LM": "^"}


# #region wrap
def wrap(a):
    """Wrap angles to (-pi, pi]."""
    return np.pi - np.mod(np.pi - a, 2 * np.pi)
# #endregion wrap


# ------------------------------------------------------------ the dataset ---
# #region dataset
def make_dataset(n_poses=400, grid=8, sigma_xy=0.05, sigma_theta_deg=1.0, seed=3):
    rng = np.random.default_rng(seed)
    # Ground truth: walk on a grid with 1 m cells, turning left/right at random.
    gt = np.zeros((n_poses, 3))
    for k in range(1, n_poses):
        x, y, th = gt[k - 1]
        options = [th, th + np.pi / 2, th - np.pi / 2]
        probs = [0.6, 0.2, 0.2]
        while True:
            th_new = wrap(rng.choice(options, p=probs))
            nx, ny = round(x + np.cos(th_new)), round(y + np.sin(th_new))
            if 0 <= nx <= grid and 0 <= ny <= grid:   # stay inside the grid
                break
        gt[k] = nx, ny, th_new

    # Edges: odometry (k -> k+1) plus loop closures when the robot revisits a grid node.
    I = list(range(n_poses - 1))
    J = list(range(1, n_poses))
    for j in range(n_poses):
        earlier = [i for i in range(j - 10) if np.allclose(gt[i, :2], gt[j, :2])]
        if earlier and rng.random() < 0.5:
            I.append(rng.choice(earlier))
            J.append(j)
    I, J = np.array(I), np.array(J)

    sigma = np.array([sigma_xy, sigma_xy, np.deg2rad(sigma_theta_deg)])
    Z = predict(gt, I, J) + rng.normal(0.0, 1.0, (len(I), 3)) * sigma   # z = h(x_true) + noise
    Z[:, 2] = wrap(Z[:, 2])
    sqrt_info = 1.0 / sigma          # Omega = diag(1/sigma^2), so sqrt(Omega) = diag(1/sigma)

    # Initial guess: chain the noisy odometry from the true first pose.
    init = np.zeros_like(gt)
    init[0] = gt[0]
    for k in range(n_poses - 1):
        c, s = np.cos(init[k, 2]), np.sin(init[k, 2])
        dx, dy, dth = Z[k]
        init[k + 1] = init[k, 0] + c * dx - s * dy, init[k, 1] + s * dx + c * dy, wrap(init[k, 2] + dth)
    return gt, init, I, J, Z, sqrt_info
# #endregion dataset


# ------------------------------------------------- residuals and Jacobians ---
# #region residual
def predict(X, I, J):
    """h_ij(x): pose j seen from pose i, (dx, dy, dtheta) in frame i."""
    d = X[J, :2] - X[I, :2]
    c, s = np.cos(X[I, 2]), np.sin(X[I, 2])
    return np.column_stack([c * d[:, 0] + s * d[:, 1],
                            -s * d[:, 0] + c * d[:, 1],
                            wrap(X[J, 2] - X[I, 2])])


def residuals(X, I, J, Z):
    """r_ij = h_ij(x) - z_ij, with the angle part wrapped to (-pi, pi]."""
    r = predict(X, I, J) - Z
    r[:, 2] = wrap(r[:, 2])
    return r
# #endregion residual


# #region jacobian
def jacobian_blocks(X, I, J):
    """A = d r_ij / d x_i  and  B = d r_ij / d x_j, each (E, 3, 3)."""
    d = X[J, :2] - X[I, :2]
    c, s = np.cos(X[I, 2]), np.sin(X[I, 2])
    E = len(I)
    A = np.zeros((E, 3, 3))
    A[:, 0, 0], A[:, 0, 1], A[:, 0, 2] = -c, -s, -s * d[:, 0] + c * d[:, 1]
    A[:, 1, 0], A[:, 1, 1], A[:, 1, 2] = s, -c, -c * d[:, 0] - s * d[:, 1]
    A[:, 2, 2] = -1.0
    B = np.zeros((E, 3, 3))
    B[:, 0, 0], B[:, 0, 1] = c, s
    B[:, 1, 0], B[:, 1, 1] = -s, c
    B[:, 2, 2] = 1.0
    return A, B
# #endregion jacobian


class PoseGraph:
    """Whitened residual/Jacobian of the graph, with pose 0 held fixed (the gauge)."""

    def __init__(self, I, J, Z, sqrt_info, x_fixed0):
        self.I, self.J, self.Z, self.w = I, J, Z, sqrt_info
        self.x0 = x_fixed0                     # pose 0, never changes
        E = len(I)
        # Sparsity pattern of the full Jacobian (3E x 3N): every edge row block touches two poses.
        rows = np.repeat(np.arange(3 * E).reshape(E, 3), 3, axis=1)                   # (E, 9)
        cols_i = (3 * I[:, None] + np.arange(3)[None, :])[:, None, :].repeat(3, 1).reshape(E, 9)
        cols_j = (3 * J[:, None] + np.arange(3)[None, :])[:, None, :].repeat(3, 1).reshape(E, 9)
        self.rows = np.concatenate([rows, rows], 1).ravel()
        self.cols = np.concatenate([cols_i, cols_j], 1).ravel()
        self.shape = (3 * E, 3 * (max(I.max(), J.max()) + 1))

    def poses(self, v):
        """Free variables v (poses 1..N-1, flattened) -> all poses (N, 3)."""
        return np.vstack([self.x0, v.reshape(-1, 3)])

    def r(self, v):
        return (residuals(self.poses(v), self.I, self.J, self.Z) * self.w).ravel()

    def cost(self, v):
        r = self.r(v)
        return 0.5 * r @ r

    def jac_full(self, v):
        A, B = jacobian_blocks(self.poses(v), self.I, self.J)
        A = A * self.w[None, :, None]           # whiten: sqrt(Omega) * A
        B = B * self.w[None, :, None]
        vals = np.concatenate([A.reshape(-1, 9), B.reshape(-1, 9)], 1).ravel()
        return sp.csr_matrix((vals, (self.rows, self.cols)), shape=self.shape)

    def jac(self, v):
        return self.jac_full(v)[:, 3:]          # drop pose 0's columns: hard constraint delta_0 = 0


def boxplus(v, delta):
    """x <- x [+] delta: add, then wrap every heading."""
    out = (v + delta).reshape(-1, 3)
    out[:, 2] = wrap(out[:, 2])
    return out.ravel()


# ------------------------------------------------------------- the solvers ---
# #region stopping
MAX_ITER, TIME_BUDGET, REPEATS = 100, 60.0, 5
TOL_REL_COST, TOL_STEP = 1e-10, 1e-8


def converged(F_old, F_new, delta, v):
    small_change = abs(F_old - F_new) <= TOL_REL_COST * F_old
    small_step = np.linalg.norm(delta) <= TOL_STEP * (np.linalg.norm(v) + TOL_STEP)
    return small_change or small_step
# #endregion stopping


# #region solvers
def solve_gd(pg, v, max_iter=MAX_ITER, c=1e-4):
    t0, F = time.perf_counter(), pg.cost(v)
    hist, alpha = [(0.0, F)], 1.0
    for _ in range(max_iter):
        g = pg.jac(v).T @ pg.r(v)
        alpha *= 2.0
        while True:                                  # backtracking (Armijo) line search
            v_new = boxplus(v, -alpha * g)
            F_new = pg.cost(v_new)
            if F_new <= F - c * alpha * (g @ g) or alpha < 1e-20:
                break
            alpha *= 0.5
        done = converged(F, F_new, alpha * g, v)
        v, F = v_new, F_new
        hist.append((time.perf_counter() - t0, F))
        if done or hist[-1][0] > TIME_BUDGET:
            break
    return v, np.array(hist)


def solve_gn(pg, v, max_iter=MAX_ITER):
    t0, F = time.perf_counter(), pg.cost(v)
    hist = [(0.0, F)]
    for _ in range(max_iter):
        Jm, r = pg.jac(v), pg.r(v)
        H, g = (Jm.T @ Jm).tocsc(), Jm.T @ r
        delta = spla.spsolve(H, -g)                  # sparse LU (SuperLU, COLAMD ordering)
        v_new = boxplus(v, delta)
        F_new = pg.cost(v_new)
        done = converged(F, F_new, delta, v) or not np.isfinite(F_new)
        v, F = v_new, F_new
        hist.append((time.perf_counter() - t0, F))
        if done or hist[-1][0] > TIME_BUDGET:
            break
    return v, np.array(hist)


def solve_lm(pg, v, max_iter=MAX_ITER, tau=1e-3):
    t0, F = time.perf_counter(), pg.cost(v)
    hist = [(0.0, F)]
    Jm, r = pg.jac(v), pg.r(v)
    H, g = (Jm.T @ Jm).tocsc(), Jm.T @ r
    lam, nu = tau * H.diagonal().max(), 2.0
    I_n = sp.identity(len(v), format="csc")
    for _ in range(max_iter):
        delta = spla.spsolve(H + lam * I_n, -g)
        v_new = boxplus(v, delta)
        F_new = pg.cost(v_new)
        rho = (F - F_new) / (0.5 * delta @ (lam * delta - g))
        done = False
        if rho > 0:                                   # accept
            done = converged(F, F_new, delta, v)
            v, F = v_new, F_new
            Jm, r = pg.jac(v), pg.r(v)
            H, g = (Jm.T @ Jm).tocsc(), Jm.T @ r
            lam *= max(1 / 3, 1 - (2 * rho - 1) ** 3)
            nu = 2.0
        else:                                         # reject
            lam *= nu
            nu *= 2.0
            done = np.linalg.norm(delta) <= TOL_STEP * (np.linalg.norm(v) + TOL_STEP)
        hist.append((time.perf_counter() - t0, F))
        if done or hist[-1][0] > TIME_BUDGET:
            break
    return v, np.array(hist)
# #endregion solvers


# ----------------------------------------------------------------- metrics ---
# #region metrics
def align_errors(est, gt):
    """Rigid 2D alignment (rotation + translation, no scale), then position RMSE and heading RMSE."""
    mu_e, mu_g = est[:, :2].mean(0), gt[:, :2].mean(0)
    U, _, Vt = np.linalg.svd((gt[:, :2] - mu_g).T @ (est[:, :2] - mu_e))
    D = np.diag([1.0, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    p = (est[:, :2] - mu_e) @ R.T + mu_g
    ate = np.sqrt(np.mean(np.sum((p - gt[:, :2]) ** 2, 1)))
    phi = np.arctan2(R[1, 0], R[0, 0])
    head = np.sqrt(np.mean(wrap(est[:, 2] + phi - gt[:, 2]) ** 2))
    return ate, np.rad2deg(head)
# #endregion metrics


# -------------------------------------------------------------------- main ---
def check_jacobian(pg, v, eps=1e-6):
    Jd = pg.jac(v).toarray()
    Jfd = np.empty_like(Jd)
    for k in range(len(v)):
        e = np.zeros_like(v)
        e[k] = eps
        Jfd[:, k] = (pg.r(v + e) - pg.r(v - e)) / (2 * eps)
    return np.abs(Jd - Jfd).max() / np.abs(Jd).max()


def run_case(label, sigma_theta_deg):
    gt, init, I, J, Z, w = make_dataset(sigma_theta_deg=sigma_theta_deg)
    n_odo = len(gt) - 1
    print(f"\n=== {label}: {len(gt)} poses, {n_odo} odometry edges, {len(I) - n_odo} loop closures, "
          f"sigma_xy = 0.05 m, sigma_theta = {sigma_theta_deg} deg ===")
    pg = PoseGraph(I, J, Z, w, init[0])
    v0 = init[1:].ravel()

    # Self-check 1: analytic Jacobian vs finite differences (at the initial guess).
    rel = check_jacobian(pg, v0)
    print(f"Jacobian check: max |J - J_fd| / max |J| = {rel:.1e}")
    assert rel < 1e-6

    # Gauge: the full H (no pose fixed) has 3 zero eigenvalues; fixing pose 0 removes them.
    Jf = pg.jac_full(v0)
    eig_full = np.linalg.eigvalsh((Jf.T @ Jf).toarray())
    eig_fixed = np.linalg.eigvalsh((pg.jac(v0).T @ pg.jac(v0)).toarray())
    print(f"smallest eigenvalues of H, all poses free : {eig_full[:4]}")
    print(f"smallest eigenvalues of H, pose 0 fixed   : {eig_fixed[:2]}   (largest {eig_full[-1]:.3e})")
    assert np.all(np.abs(eig_full[:3]) < 1e-8 * eig_full[-1]) and eig_fixed[0] > 1e-6
    Jm = pg.jac(v0)
    H = Jm.T @ Jm
    print(f"J: {Jm.shape}, {Jm.nnz} non-zeros ({100 * Jm.nnz / np.prod(Jm.shape):.2f}%);  "
          f"H: {H.shape}, {H.nnz} non-zeros ({100 * H.nnz / np.prod(H.shape):.2f}%)")
    print(f"LM start: lambda_0 = 1e-3 * max diag(H) = {1e-3 * H.diagonal().max():.2f}")

    # Each solver runs REPEATS times; the table reports the median wall time.
    results, times = {}, {}
    for name, solver in [("GD", solve_gd), ("GN", solve_gn), ("LM", solve_lm)]:
        runs = [solver(pg, v0.copy()) for _ in range(REPEATS)]
        results[name] = runs[0]
        times[name] = np.median([h[-1, 0] for _, h in runs])
    v_gd_long, hist_gd_long = solve_gd(pg, v0.copy(), max_iter=10000)
    times["GD, 10000 iter"] = hist_gd_long[-1, 0]

    # Self-check 2 (sanity): scipy's trust-region solver with finite-difference sparse Jacobian.
    pattern = (pg.jac(v0) != 0).astype(int)
    t0 = time.perf_counter()
    ref = least_squares(pg.r, v0, jac_sparsity=pattern, method="trf",
                        xtol=1e-12, ftol=1e-12, gtol=1e-12, tr_options={"atol": 1e-12, "btol": 1e-12})
    t_ref = time.perf_counter() - t0

    def row(name, v, hist):
        ate, head = align_errors(pg.poses(v), gt)
        return f"| {name} | {len(hist) - 1} | {times[name]:.3f} | {hist[-1, 1]:.6g} | {ate:.4f} | {head:.3f} |"

    ate0, head0 = align_errors(init, gt)
    print("\n| method | iterations | time [s] | final cost F | ATE RMSE [m] | heading RMSE [deg] |")
    print("|---|---|---|---|---|---|")
    print(f"| initial (odometry) | - | - | {pg.cost(v0):.6g} | {ate0:.4f} | {head0:.3f} |")
    for name, (v, hist) in results.items():
        print(row(name, v, hist))
    print(row("GD, 10000 iter", v_gd_long, hist_gd_long))
    ate_r, head_r = align_errors(pg.poses(ref.x), gt)
    print(f"| scipy trf (finite-diff. J) | {ref.nfev} | {t_ref:.3f} | {ref.cost:.6g} | {ate_r:.4f} | {head_r:.3f} |")
    F_gn, F_lm = results["GN"][1][-1, 1], results["LM"][1][-1, 1]
    print(f"relative difference to scipy: GN {abs(F_gn - ref.cost) / ref.cost:.1e}, "
          f"LM {abs(F_lm - ref.cost) / ref.cost:.1e}")

    # --- figures ---
    tag = label.split()[0].lower()
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.8), sharex=True, sharey=True, layout="constrained")
    panels = [("ground truth + loop closures", gt, "#000000"), ("initial (chained odometry)", init, "#888888")]
    panels += [(f"{n} ({len(h) - 1} it)", pg.poses(v), COLORS[n]) for n, (v, h) in results.items()]
    for ax, (title, P, col) in zip(axes, panels):
        if title.startswith("ground"):
            # A loop closure links two visits of the same grid node, so draw it as a ring there.
            ax.plot(*gt[J[n_odo:], :2].T, "o", mfc="none", mec="#CC79A7", ms=7, mew=1.5,
                    label="loop closure", zorder=3)
            ax.legend(loc="upper right", frameon=False, fontsize=9)
        else:
            ax.plot(gt[:, 0], gt[:, 1], color="#BBBBBB", lw=1.0, label="ground truth")
        ax.plot(P[:, 0], P[:, 1], color=col, lw=1.2)
        ax.plot(P[0, 0], P[0, 1], "ko", ms=5)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")
    axes[1].legend(loc="upper right", frameon=False, fontsize=9)
    fig.savefig(OUT / f"trajectories_{tag}.png", dpi=150, bbox_inches="tight")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    for name, (v, hist) in results.items():
        kw = dict(color=COLORS[name], marker=MARKERS[name], ms=4, lw=1.8, label=name,
                  markevery=1 if len(hist) < 30 else 10)
        axes[0].semilogy(np.arange(len(hist)), hist[:, 1], **kw)
        axes[1].semilogy(hist[:, 0], hist[:, 1], **kw)
    for ax in axes:
        ax.axhline(ref.cost, color="#555555", ls="--", lw=1, label="scipy trf final cost")
        ax.grid(alpha=0.3, which="both")
        ax.set_ylabel("cost F (log scale)")
        ax.legend(frameon=False)
    axes[0].set(xlabel="iteration", title=f"{label}: cost vs iteration")
    axes[1].set(xlabel="wall time [s]", title=f"{label}: cost vs wall time")
    fig.tight_layout()
    fig.savefig(OUT / f"costs_{tag}.png", dpi=150)
    return results, ref


# #region sweep
def sweep():
    """Push the heading noise up. Reference: LM started from the ground truth."""
    print("\n=== Sweep: more heading noise, two random graphs, up to 500 iterations ===")
    print("| sigma_theta [deg] | seed | init ATE [m] | GN it | GN F | GN ATE [m] | LM it | LM F | LM ATE [m] | F from GT start |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for sigma_theta_deg in (8.0, 10.0, 15.0, 20.0):
        for seed in (3, 4):
            gt, init, I, J, Z, w = make_dataset(sigma_theta_deg=sigma_theta_deg, seed=seed)
            pg = PoseGraph(I, J, Z, w, init[0])
            cells = [f"{sigma_theta_deg:g}", str(seed), f"{align_errors(init, gt)[0]:.2f}"]
            for solver in (solve_gn, solve_lm):
                v, hist = solver(pg, init[1:].ravel(), max_iter=500)
                cells += [str(len(hist) - 1), f"{hist[-1, 1]:.1f}", f"{align_errors(pg.poses(v), gt)[0]:.3f}"]
            _, hist = solve_lm(pg, gt[1:].ravel(), max_iter=500)
            cells.append(f"{hist[-1, 1]:.1f}")
            print("| " + " | ".join(cells) + " |")
# #endregion sweep


def chain(start, Z_odo):
    """Dead reckoning: compose relative odometry measurements from a start pose."""
    P = [np.asarray(start, float)]
    for dx, dy, dth in Z_odo:
        x, y, th = P[-1]
        c, s = np.cos(th), np.sin(th)
        P.append(np.array([x + c * dx - s * dy, y + s * dx + c * dy, wrap(th + dth)]))
    return np.array(P)


def schematic_figure(seed=4):
    """A 20-pose toy graph: one lap around a 4 m square, then 4 poses along the first side again."""
    pts = [(k, 0) for k in range(5)] + [(4, k) for k in range(1, 5)] + [(4 - k, 4) for k in range(1, 5)]
    pts += [(0, 4 - k) for k in range(1, 4)] + [(k, 0.5) for k in range(4)]     # 2nd visit, 0.5 m to the side
    pts = np.array(pts, float)
    heading = np.arctan2(*np.diff(pts, axis=0)[:, ::-1].T)                   # direction of travel
    gt = np.column_stack([pts, np.append(heading[0], heading)])
    I_odo = np.arange(len(gt) - 1)
    loops = np.array([[0, 16], [1, 17], [2, 18], [3, 19]])
    Z_odo = predict(gt, I_odo, I_odo + 1)
    assert np.allclose(chain(gt[0], Z_odo), gt)                 # self-check: noise-free chaining is exact
    rng = np.random.default_rng(seed)
    noisy = Z_odo + rng.normal(0.0, 1.0, Z_odo.shape) * [0.03, 0.03, np.deg2rad(4.0)]
    drift = chain(gt[0], noisy)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), layout="constrained")
    ax = axes[0]
    ax.plot(*gt[:, :2].T, "-", color="#333333", lw=1.6, zorder=1, label="odometry edge")
    for i, j in loops:
        ax.annotate("", xy=gt[j, :2], xytext=gt[i, :2], zorder=1,
                    arrowprops=dict(arrowstyle="-", color="#CC79A7", lw=1.8, ls="--",
                                    connectionstyle="arc3,rad=0.6"))
    ax.plot([], [], "--", color="#CC79A7", lw=1.8, label="loop-closure edge")
    ax.plot(*gt[:, :2].T, "o", ms=13, mfc="white", mec="#333333", zorder=2, label="pose (node)")
    ax.quiver(gt[:, 0], gt[:, 1], np.cos(gt[:, 2]), np.sin(gt[:, 2]), color="#0072B2", zorder=3,
              angles="xy", scale_units="xy", scale=3.6, width=0.006)
    for k in (0, 4, 8, 12, 16, 19):
        ax.annotate(str(k), gt[k, :2], xytext=(-12, 9), textcoords="offset points", fontsize=9)
    ax.set(title="(a) A pose graph: 20 poses, 19 odometry edges, 4 loop closures",
           xlim=(-1, 5), ylim=(-1.2, 5), aspect="equal", xlabel="x [m]", ylabel="y [m]")
    ax.legend(loc="center", frameon=False, fontsize=9)

    ax = axes[1]
    ax.plot(*gt[:, :2].T, "-o", color="#888888", ms=4, lw=1.2, label="true path")
    ax.plot(*drift[:, :2].T, "-o", color="#333333", ms=4, lw=1.4, label="chained odometry (dead reckoning)")
    for i, j in loops:
        ax.plot(*np.stack([drift[i, :2], drift[j, :2]]).T, "--", color="#CC79A7", lw=1.8)
    ax.plot([], [], "--", color="#CC79A7", lw=1.8, label="loop closures: these should be 0.5 m apart")
    gap = np.linalg.norm(drift[16, :2] - drift[0, :2])
    ax.annotate(f"pose 16: {gap:.1f} m from pose 0", drift[16, :2], xytext=(10, -4),
                textcoords="offset points", fontsize=9)
    ax.set(title="(b) Odometry alone drifts; loop closures pull it back",
           aspect="equal", xlabel="x [m]")
    ax.legend(loc="center", frameon=False, fontsize=9)
    fig.savefig(OUT / "pose_graph_schematic.png", dpi=150, bbox_inches="tight")
    print(f"schematic: odometry drift at pose 16 = {gap:.2f} m (should be 0.5 m)")


if __name__ == "__main__":
    schematic_figure()
    print(f"MAX_ITER = {MAX_ITER}, time budget = {TIME_BUDGET} s, "
          f"stop if |dF|/F <= {TOL_REL_COST} or |delta| <= {TOL_STEP}(|x| + {TOL_STEP})")
    for label, sigma_theta_deg in [("Easy init", 1.0), ("Hard init", 5.0)]:
        res, ref = run_case(label, sigma_theta_deg)
        for name in ("GN", "LM"):
            F = res[name][1][-1, 1]
            assert abs(F - ref.cost) <= 1e-6 * ref.cost, name           # self-check: matches scipy
    sweep()
