"""Post 00b: what covariance means, in pictures.

Figure 1: two sensors measuring the same 0 m target, noise sigma = 1 cm vs 5 cm.
Figure 2: 2D noise clouds with three different covariance matrices + 1/2-sigma ellipses.
Run:  python covariance.py [output_dir]
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
rng = np.random.default_rng(7)

# ---- Figure 1: variance in one dimension -------------------------------
fig, ax = plt.subplots(figsize=(7, 3.2))
u = np.linspace(-15, 15, 600)
for sigma, color in [(1.0, "tab:blue"), (5.0, "tab:orange")]:
    pdf = np.exp(-0.5 * (u / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    ax.plot(u, pdf, color=color, lw=2,
            label=rf"$\sigma={sigma:g}$ cm  (variance $\sigma^2={sigma**2:g}$, weight $1/\sigma^2={1/sigma**2:g}$)")
    ax.fill_between(u, pdf, color=color, alpha=0.15)
ax.axvline(0, color="k", lw=0.8, ls="--")
ax.set_xlabel("measurement error (cm)")
ax.set_ylabel("how likely")
ax.set_yticks([])
ax.legend(frameon=False, fontsize=8)
ax.set_title("Same target, two sensors: narrow = trusted, wide = not trusted")
fig.tight_layout()
fig.savefig(OUT / "covariance_1d.png", dpi=150)
plt.close(fig)

# ---- Figure 2: covariance in two dimensions ----------------------------
# #region covs
covs = {
    "independent, equal\nΣ = [[1, 0], [0, 1]]": np.array([[1.0, 0.0], [0.0, 1.0]]),
    "independent, x noisier\nΣ = [[4, 0], [0, 0.25]]": np.array([[4.0, 0.0], [0.0, 0.25]]),
    "correlated\nΣ = [[2, 1.6], [1.6, 2]]": np.array([[2.0, 1.6], [1.6, 2.0]]),
}
# #endregion covs

fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharex=True, sharey=True)
theta = np.linspace(0, 2 * np.pi, 200)
circle = np.stack([np.cos(theta), np.sin(theta)])
for ax, (title, S) in zip(axes, covs.items()):
    pts = rng.multivariate_normal([0, 0], S, 400)
    ax.scatter(pts[:, 0], pts[:, 1], s=6, alpha=0.4, color="tab:gray")
    vals, vecs = np.linalg.eigh(S)  # ellipse axes = eigenvectors, lengths = sqrt(eigenvalues)
    for k, ls in [(1, "-"), (2, "--")]:
        e = vecs @ (k * np.sqrt(vals)[:, None] * circle)
        ax.plot(e[0], e[1], color="tab:red", ls=ls, lw=1.5,
                label=f"{k}σ" if ax is axes[0] else None)
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_xlabel("error in x")
axes[0].set_ylabel("error in y")
axes[0].legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(OUT / "covariance_2d.png", dpi=150)
