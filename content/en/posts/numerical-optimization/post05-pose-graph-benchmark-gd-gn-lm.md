---
title: "Post 05: A 2D Pose-Graph Benchmark: GD vs GN vs LM"
date: 2026-11-10T14:00:00+03:00
weight: 7
draft: true
toc: false
images:
math: true
series: ["numerical-optimization"]
tags:
  - optimization
  - least-squares
  - python
  - slam
  - pose-graph
---

So far, our test problem had two unknowns. Real SLAM problems have thousands. In this last post, we build a top-view (2D) **pose graph**: a robot drives 400 steps around a grid, its odometry drifts, and a set of loop closures says "you have been here before". We write the residuals and their Jacobians by hand, solve with sparse matrices, and run gradient descent, Gauss-Newton and Levenberg-Marquardt under the same rules. Then we check the answer against SciPy, and push the noise up until things break.

## What We Will Learn

* What a 2D pose graph is: poses \((x, y, \theta)\), odometry edges, loop-closure edges.{{< sidenote >}}\(\theta\) (theta) is the usual letter for an angle; here, which way the robot faces.{{< /sidenote >}}
* The relative-pose residual, angle wrapping, and the \(\boxplus\) update.{{< sidenote >}}\(\boxplus\) ("box-plus") is addition with one extra rule: after adding the step, wrap every angle back into \((-\pi, \pi]\), that is, between -180° and 180°.{{< /sidenote >}}
* How to derive the analytic Jacobians of an SE(2) edge with respect to both poses.
* Why \(J^\top \Omega J\) is sparse, and how to solve with it.{{< sidenote >}}\(J\) is the Jacobian, a table of slopes of the residuals, and \(J^\top\) ("J transpose") is that table turned on its side. \(\Omega\) (omega) weights each measurement by how much we trust it.{{< /sidenote >}}
* Gauge freedom: why \(H\) is singular until we fix the first pose.{{< sidenote >}}\(H\) is short for \(J^\top \Omega J\), the Gauss-Newton stand-in for the curvature of the cost. "Singular" means it cannot be inverted: solving with it is like dividing by zero.{{< /sidenote >}}
* How GD, GN and LM compare on cost, time and accuracy, from an easy and a hard start.

## Table of Contents

1. The Pose Graph.
2. The Synthetic Dataset.
3. The Residual of an Edge.
4. The Jacobians of an Edge.
5. Sparse Matrices.
6. Gauge Freedom: Fixing the First Pose.
7. A Fair Comparison.
8. Results: Easy Start.
9. Results: Hard Start.
10. Sanity Check Against SciPy.
11. Pushing the Noise Further.
12. Honest Caveats.
13. The Full Program.

---

## 1. The Pose Graph

A **pose** is where the robot is and which way it faces: \(x_k = (x_k, y_k, \theta_k)\) for pose \(k\). In 2D, the set of all such poses is called **SE(2)**, the special Euclidean group: all rotations and translations of the plane. The state is all poses stacked: with 400 poses, \(x \in \mathbb{R}^{1200}\).{{< sidenote >}}\(x \in \mathbb{R}^{1200}\) reads "\(x\) is a list of 1200 real numbers": 3 numbers for each of the 400 poses. The small \(k\) in \(x_k\) is just a counter.{{< /sidenote >}}

The robot never measures where it *is*. It measures where it moved *relative to where it was*. Each such measurement links two poses \(i\) and \(j\), so we draw it as an **edge** in a graph whose nodes are poses. Two kinds of edges:

* **Odometry edges** connect pose \(k\) to pose \(k+1\). They come from wheel encoders or scan matching. Each one is a bit wrong, and errors add up: the further we drive, the further off we are. This is **drift**.
* **Loop-closure edges** connect pose \(j\) to some much earlier pose \(i\), when the robot recognizes a place it has seen before. They are what lets us correct the drift.

Every edge is a *soft constraint* in the sense of post 00: it adds one residual term \(\frac{1}{2} r_{ij}^\top \Omega_{ij} r_{ij}\) to the cost.{{< sidenote >}}\(r_{ij}\) is the residual (predicted minus measured) of the edge between poses \(i\) and \(j\). In general, \(r^\top \Omega\, r\) also has cross terms between entries, but here \(\Omega\) is diagonal (independent noise, section 2), so it is just a weighted sum of the squared entries of \(r\). The \(\frac{1}{2}\) only makes derivatives tidier.{{< /sidenote >}} Solving the pose graph means finding the poses that disagree the least with all edges at once.

![(a) A toy pose graph with 20 poses, heading arrows, solid odometry edges and dashed loop closures; (b) the true path vs the drifting chained-odometry path](/images/numerical-optimization/post05/pose_graph_schematic.png)

*(a) A toy pose graph with 20 poses: the robot drives once around a 4 m square, then along the first side again, 0.5 m to the side. Each circle is a pose and its arrow the heading; solid lines are odometry edges, dashed ones loop closures ("pose 16 is 0.5 m from pose 0"). (b) Why loop closures matter: chaining the noisy odometry alone drifts, and after one lap pose 16 is 1.4 m from pose 0 instead of 0.5 m. The loop-closure edges (dashed) are what pull the two ends back together. The graph in the rest of this post is the same idea, with 400 poses and 157 loop closures.*

## 2. The Synthetic Dataset

We make the data ourselves, so we know the truth:

{{< code file="optimization/post05/pose_graph.py" region="dataset" title="Generate the pose graph" >}}

* **Ground truth**: a walk on a grid of 1 m cells, 8 by 8 cells, 400 poses. At each node, the robot goes straight (60%) or turns left or right (20% each), and never leaves the grid. This "Manhattan world" walk revisits many nodes.
* **Edges**: 399 odometry edges, plus a loop closure on about half of the revisits: 157 loop closures in total. The measurement is the true relative pose plus Gaussian noise: \(\sigma_{xy} = 0.05\) m on each position part, and \(\sigma_\theta = 1°\) on the angle (5° for the hard start).{{< sidenote >}}\(\sigma\) (sigma) is the standard deviation: the typical size of the noise.{{< /sidenote >}}
* **Information matrix**: \(\Omega = \operatorname{diag}(1/\sigma_{xy}^2, 1/\sigma_{xy}^2, 1/\sigma_\theta^2)\) for every edge.{{< sidenote >}}\(\operatorname{diag}(a, b, c)\) is a \(3 \times 3\) table with \(a, b, c\) on its diagonal and zeros elsewhere. Small noise \(\sigma\) means a big weight \(1/\sigma^2\).{{< /sidenote >}}
* **Initial guess**: chain the noisy odometry from the true first pose. This is exactly what a robot would believe if it had no loop closures. It is visibly drifted.

Everything uses a fixed seed (3). The easy and the hard case use the *same* trajectory, the same edges and the same random draws; only the heading noise is 5 times larger in the hard case.

## 3. The Residual of an Edge

An edge \((i, j)\) carries a measurement \(z_{ij} = (\Delta x, \Delta y, \Delta\theta)\): where pose \(j\) is, *as seen from pose \(i\)*, in pose \(i\)'s own frame.{{< sidenote >}}\(\Delta\) (capital delta) means "change in": \(\Delta\theta\) is how much the heading changed between the two poses.{{< /sidenote >}}

The measurement model predicts that from the current poses. Write \(t_i = (x_i, y_i)\) for the position and \(R_i = R(\theta_i)\) for the 2D rotation matrix of pose \(i\):

$$
R(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix} .
$$

To see the vector from \(i\) to \(j\) in pose \(i\)'s frame, we rotate it backward by \(\theta_i\), using \(R_i^\top\). The heading change is just the difference of angles:

$$
h_{ij}(x) = \begin{bmatrix} R_i^\top (t_j - t_i) \\ \theta_j - \theta_i \end{bmatrix} .
$$

The residual is predicted minus measured, as always:

$$
r_{ij}(x) = \begin{bmatrix} R_i^\top (t_j - t_i) - t_z \\ \operatorname{wrap}(\theta_j - \theta_i - \Delta\theta) \end{bmatrix},
\qquad t_z = (\Delta x, \Delta y) .
$$

**Angle wrapping.** Angles live on a circle: \(179°\) and \(-179°\) are only \(2°\) apart, not \(358°\). If we forget this, a robot that turned a little past \(180°\) gets an enormous residual, and the optimizer tries to "unwind" it with a full turn. So every angle difference is wrapped to \((-\pi, \pi]\):

{{< code file="optimization/post05/pose_graph.py" region="wrap" >}}

{{< code file="optimization/post05/pose_graph.py" region="residual" title="Pose-graph residuals" >}}

For the same reason, the update is not plain addition. We write it \(x \leftarrow x \boxplus \delta\) ("box-plus"): add \(\delta\), then wrap every heading back to \((-\pi, \pi]\).{{< sidenote >}}\(\delta\) (delta) is the step the solver wants to take, and \(\leftarrow\) means "replace by".{{< /sidenote >}}

A note for readers who know g2o: it defines the edge error as the relative pose between \(Z_{ij}\) and \(X_i^{-1}X_j\), which rotates our position residual by \(R(\Delta\theta)^\top\). A rotation does not change a vector's length, and our position noise is the same in \(x\) and \(y\), so the cost is exactly the same with either definition.

## 4. The Jacobians of an Edge

Each residual \(r_{ij}\) depends on only two poses, so its Jacobian has only two non-zero \(3 \times 3\) blocks:{{< sidenote >}}\(\partial r_{ij} / \partial x_i\) is a partial derivative: how much the residual changes when we nudge only pose \(i\). A \(3 \times 3\) block is a small table, 3 rows by 3 columns.{{< /sidenote >}}

$$
A_{ij} = \frac{\partial r_{ij}}{\partial x_i}, \qquad B_{ij} = \frac{\partial r_{ij}}{\partial x_j} .
$$

Let \(c = \cos\theta_i\), \(s = \sin\theta_i\) and \(d = t_j - t_i = (d_x, d_y)\). Then

$$
R_i^\top d = \begin{bmatrix} c\, d_x + s\, d_y \\ -s\, d_x + c\, d_y \end{bmatrix} .
$$

Now we differentiate, one variable at a time.

**With respect to \(t_j\).** \(d\) contains \(+t_j\), so the derivative of \(R_i^\top d\) is \(R_i^\top\). The angle part does not depend on \(t_j\).

**With respect to \(t_i\).** \(d\) contains \(-t_i\), so the derivative is \(-R_i^\top\).

**With respect to \(\theta_i\).** Differentiate each entry of \(R_i^\top d\) with respect to \(\theta_i\) (using \(c' = -s\), \(s' = c\)):

$$
\frac{\partial}{\partial \theta_i} R_i^\top d = \begin{bmatrix} -s\, d_x + c\, d_y \\ -c\, d_x - s\, d_y \end{bmatrix} .
$$

The angle residual \(\theta_j - \theta_i - \Delta\theta\) has derivative \(-1\). (The wrap only adds a constant multiple of \(2\pi\), so it does not change the derivative.)

**With respect to \(\theta_j\).** The position part does not depend on it; the angle part has derivative \(+1\).

Putting it together, with columns ordered \((x, y, \theta)\):

$$
A_{ij} = \begin{bmatrix} -c & -s & -s\, d_x + c\, d_y \\ s & -c & -c\, d_x - s\, d_y \\ 0 & 0 & -1 \end{bmatrix},
\qquad
B_{ij} = \begin{bmatrix} c & s & 0 \\ -s & c & 0 \\ 0 & 0 & 1 \end{bmatrix} .
$$

In code, all edges are handled at once with NumPy arrays:

{{< code file="optimization/post05/pose_graph.py" region="jacobian" title="Build the Jacobian blocks" >}}

Hand-derived Jacobians are the most common source of bugs in this kind of code, so the script compares the full Jacobian with central finite differences at the initial guess. The largest difference, relative to the largest entry, is \(7.5 \times 10^{-10}\) (easy) and \(2.2 \times 10^{-10}\) (hard): the derivation is right.

**Whitening.** Instead of carrying \(\Omega\) around, we multiply each residual and each Jacobian row by \(\Omega^{1/2} = \operatorname{diag}(1/\sigma_{xy}, 1/\sigma_{xy}, 1/\sigma_\theta)\). Then \(\frac{1}{2}\lVert \Omega^{1/2} r \rVert^2 = \frac{1}{2} r^\top \Omega r\), and all formulas from posts 02 to 04 apply unchanged, with \(\Omega = I\).{{< sidenote >}}\(\Omega^{1/2}\) is the square root of \(\Omega\): here, \(1/\sigma\) for each part. \(\lVert \cdot \rVert^2\) is a squared length: square every entry and add them up.{{< /sidenote >}}

## 5. Sparse Matrices

The stacked Jacobian \(J\) has \(3 \times 556 = 1668\) rows (one per residual number) and 1197 columns (3 per pose, for the 399 poses we optimize; section 6 explains why not 400). But each row block touches only 2 poses. The script counts: \(J\) has 9990 non-zeros, 0.50% of its entries.

The same holds for \(H = J^\top \Omega J\): 8329 non-zeros, 0.58%. Its block \((k, l)\) is non-zero only if poses \(k\) and \(l\) share an edge (or \(k = l\)). The pattern of \(H\) is the pattern of the graph itself. Storing it as a dense \(1200 \times 1200\) matrix and solving with dense Cholesky would cost \(O(n^3)\); for a city-sized map with \(10^6\) poses, that is impossible.{{< sidenote >}}\(O(n^3)\) means the work grows like the cube of the size: twice as many poses, about eight times the work.{{< /sidenote >}} Sparse solvers only store and touch the non-zeros.

We build \(J\) as a `scipy.sparse` matrix (the row and column indices are computed once, only the values change per iteration), form \(H = J^\top J\) as a sparse product, and solve \(H\delta = -g\) with `scipy.sparse.linalg.spsolve`.{{< sidenote >}}\(g = J^\top \Omega r\) is the gradient: the slope of the cost, pointing uphill.{{< /sidenote >}} SciPy uses SuperLU here, a sparse LU factorization, with a COLAMD *fill-reducing ordering*: it reorders the variables so that the factors stay sparse. Real SLAM libraries do the same with sparse Cholesky (CHOLMOD, or Eigen's), and the choice of variable ordering matters a lot for their speed.

## 6. Gauge Freedom: Fixing the First Pose

Every edge measures a *relative* pose. If we take the whole solution and rotate and shift it, every relative pose stays the same, so every residual stays the same, and the cost stays the same. In 2D, that is 3 free directions: shift in \(x\), shift in \(y\), rotate. This is called **gauge freedom**.

For the solver, it means \(J v = 0\) for those 3 directions \(v\), so \(H = J^\top \Omega J\) has 3 zero eigenvalues and is singular. The script shows this at the initial guess:

```text
smallest eigenvalues of H, all poses free : [-2.25560368e-12 -3.24696446e-13  3.99184243e-12  9.78378356e+00]
smallest eigenvalues of H, pose 0 fixed   : [0.16401103 0.64829431]   (largest 2.615e+04)
```

With all poses free, exactly 3 eigenvalues are zero (up to rounding, \(10^{-12}\) against a largest of \(2.6 \times 10^4\)), and the 4th one is 9.78. GN would have to divide by zero.

The fix is the one **hard constraint** of this series: we fix the first pose to its known value, \(\delta_0 = 0\) exactly. The simplest way to enforce it is to remove pose 0 from the variables: we drop its 3 columns from \(J\), and the solver never sees it.

```python
def jac(self, v):
    return self.jac_full(v)[:, 3:]          # drop pose 0's columns: hard constraint delta_0 = 0
```

Now the smallest eigenvalue is 0.164, and \(H\) is invertible. (LM's \(\lambda I\) would also make this \(H\) invertible, but GN would still fail, and where the map ends up would depend on the damping rather than on a clear rule.{{< sidenote >}}\(\lambda\) is LM's damping knob from post 04, and \(I\) the identity matrix (the matrix version of 1), so \(\lambda I\) adds \(\lambda\) to the diagonal of \(H\).{{< /sidenote >}} Fixing the gauge is the clean solution.)

Look at those numbers again: the largest eigenvalue is about \(2.6 \times 10^4\) and the smallest about 0.16, so the condition number of \(H\) is roughly \(10^5\). In post 02, a condition number of 29 cost GD 258 iterations. Keep that in mind.

## 7. A Fair Comparison

All three methods get exactly the same conditions:

* **Same start**: the chained odometry.
* **Same budget**: at most 100 iterations, and at most 60 s of wall time.
* **Same stopping rules**: stop when the relative change in cost is tiny, \(|F_{\text{old}} - F_{\text{new}}| \le 10^{-10} F_{\text{old}}\), or when the step is tiny, \(\lVert\delta\rVert \le 10^{-8}(\lVert x\rVert + 10^{-8})\).{{< sidenote >}}\(|\cdot|\) is the absolute value: the size of a number, ignoring its sign. \(F\) is the cost.{{< /sidenote >}}

{{< code file="optimization/post05/pose_graph.py" region="stopping" title="Shared stopping rules" >}}

* **GD** uses the backtracking Armijo line search of post 02. **GN** takes the full step, as in post 03. **LM** uses \(\lambda I\) damping with Nielsen's rule and \(\tau = 10^{-3}\), as in post 04.{{< sidenote >}}\(\tau\) (tau) sets LM's starting damping, relative to the size of \(H\).{{< /sidenote >}} A rejected LM step counts as an iteration.
* Each solver runs 5 times; the table shows the median wall time.

{{< code file="optimization/post05/pose_graph.py" region="solvers" title="GD, GN and LM solvers" >}}

**Accuracy metrics.** The cost tells us how well the poses agree with the *measurements*. Since we know the truth, we can also measure how close they are to the *ground truth*:

* **ATE RMSE** (Absolute Trajectory Error): align the estimated positions to the true ones with the best rotation and translation (no scaling), then take the root mean square of the position errors.
* **Heading RMSE**: the same for the headings, after the same alignment.

{{< code file="optimization/post05/pose_graph.py" region="metrics" title="Align and measure errors" >}}

**What is a good final cost?** At the true optimum of a correct model, \(2F\) behaves like a chi-squared variable with (residuals − variables) degrees of freedom: \(1668 - 1197 = 471\). So we expect \(F\) around 235. A much larger final cost means the solver is not at the global optimum.

## 8. Results: Easy Start

\(\sigma_\theta = 1°\). Timings are on the author's machine (AMD Ryzen Threadripper PRO 5975WX, Python 3.14.4, NumPy 2.5.3, SciPy 1.18.1).

| method | iterations | time [s] | final cost F | ATE RMSE [m] | heading RMSE [deg] |
|---|---|---|---|---|---|
| initial (odometry) | - | - | 29268.6 | 0.6487 | 4.283 |
| GD | 100 | 0.051 | 1357.15 | 0.2872 | 1.722 |
| GN | 5 | 0.017 | 245.429 | 0.0778 | 0.895 |
| LM | 9 | 0.031 | 245.429 | 0.0778 | 0.895 |
| GD, 10000 iter | 10000 | 4.770 | 264.158 | 0.0788 | 0.897 |
| scipy trf (finite-diff. J) | 7 | 0.197 | 245.429 | 0.0778 | 0.895 |

![Top view: ground truth with loop closures, the drifted initial guess, and the GD, GN and LM results for the easy start](/images/numerical-optimization/post05/trajectories_easy.png)

![Cost vs iteration and cost vs wall time for GD, GN and LM, easy start](/images/numerical-optimization/post05/costs_easy.png)

* **GN** converges in 5 iterations, to a cost of 245.4 (close to the expected 235). The position error drops from 0.65 m to 0.078 m.
* **LM** reaches exactly the same cost and the same poses in 9 iterations. Unlike in post 04, it never rejects a step here (its cost goes 29269, 750, 361, 288, 253, 246, ...). It is slower for a different reason: its starting damping. The script prints \(\lambda_0 = 10^{-3} \max_j H_{jj} = 20.4\). The largest diagonal entries belong to heading variables (a heading change swings the far end of every edge attached to that pose), but the weakest directions of \(H\) have eigenvalues around 0.16. So the first LM steps are heavily damped exactly in the directions that need the biggest correction, and it takes a few iterations of \(\lambda\) shrinking by up to 3 each time before LM behaves like GN. (Marquardt's \(\operatorname{diag}(J^\top J)\) scaling from post 04 is designed for this kind of imbalance. We did not test it here.)
* **GD** uses its whole budget of 100 iterations and is still at a cost of 1357, more than 5 times the optimum. The trajectory has improved, but it is visibly bent. Even with **10,000 iterations** (4.8 s, about 280 times GN's time), GD is still almost 8% above the optimum cost (264.2 vs 245.4). This is a condition number of roughly \(10^5\) at work: GD makes only a tiny bit of progress along the weakest directions at each step.

## 9. Results: Hard Start

Same graph, \(\sigma_\theta = 5°\). The odometry now drifts by 2.2 m and 19° RMS, and the initial cost is 12 times higher:

| method | iterations | time [s] | final cost F | ATE RMSE [m] | heading RMSE [deg] |
|---|---|---|---|---|---|
| initial (odometry) | - | - | 357556 | 2.2306 | 18.634 |
| GD | 100 | 0.051 | 3039.07 | 0.6517 | 12.037 |
| GN | 8 | 0.028 | 242.373 | 0.1060 | 3.023 |
| LM | 14 | 0.048 | 242.373 | 0.1060 | 3.023 |
| GD, 10000 iter | 10000 | 4.755 | 243.22 | 0.1119 | 3.082 |
| scipy trf (finite-diff. J) | 11 | 0.218 | 242.373 | 0.1060 | 3.023 |

![Top view for the hard start: the initial guess is badly drifted and rotated; GN and LM recover the grid, GD after 100 iterations does not](/images/numerical-optimization/post05/trajectories_hard.png)

![Cost vs iteration and cost vs wall time, hard start](/images/numerical-optimization/post05/costs_hard.png)

We expected this case to show GN failing and LM saving the day. **It did not.** GN still converged, in 8 iterations, to the same optimum as LM and SciPy, and LM needed 14. On this problem, from this start, the extra safety of LM only cost time. The cost plot shows GN's second step making almost no progress (it nearly stalls at a cost of about \(10^4\)) before it finds its way, but it never goes up.

GD, after 100 iterations, has turned the map by a large angle and is still far off (12° heading error). With 10,000 iterations, it gets close to the optimum (243.2 vs 242.4).

The final costs are again close to the expected 235, and the ATE (0.106 m) is a bit larger than in the easy case, because the heading measurements are 5 times noisier.

## 10. Sanity Check Against SciPy

To check our GN and LM against an independent implementation, we also solve every case with `scipy.optimize.least_squares` (method `trf`, a trust-region method). To keep it truly independent of our hand-derived Jacobian, we do **not** pass our Jacobian: we only pass its sparsity pattern (`jac_sparsity`), and SciPy computes the Jacobian with finite differences.

```python
pattern = (pg.jac(v0) != 0).astype(int)
ref = least_squares(pg.r, v0, jac_sparsity=pattern, method="trf",
                    xtol=1e-12, ftol=1e-12, gtol=1e-12, tr_options={"atol": 1e-12, "btol": 1e-12})
```

```text
easy: relative difference to scipy: GN 8.0e-15, LM 2.3e-14
hard: relative difference to scipy: GN 9.7e-14, LM 8.2e-16
```

The final costs agree to 13 or more digits, and so do the ATE and heading errors in the tables. The script asserts agreement to \(10^{-6}\).

One practical note: for a sparse Jacobian, `trf` solves its inner linear problems approximately, with the iterative LSMR method. With the default LSMR tolerances, it took thousands of outer iterations here (up to about 100 s). Tightening LSMR's `atol` and `btol` to \(10^{-12}\) brought it down to 7 and 11 iterations. The SciPy timings in the tables also include the finite-difference Jacobians, so they are not comparable to our solvers.

## 11. Pushing the Noise Further

The hard start did not break GN. So how far can we push? The script runs a small sweep: heading noise of 8°, 10°, 15° and 20°, on two different random graphs (seeds 3 and 4), with up to 500 iterations. As a reference for "the good answer", we also start LM from the ground truth.

{{< code file="optimization/post05/pose_graph.py" region="sweep" title="Heading-noise sweep" >}}

| \(\sigma_\theta\) [deg] | seed | init ATE [m] | GN it | GN F | GN ATE [m] | LM it | LM F | LM ATE [m] | F from GT start |
|---|---|---|---|---|---|---|---|---|---|
| 8 | 3 | 3.44 | 10 | 239.7 | 0.132 | 36 | 239.7 | 0.132 | 239.7 |
| 8 | 4 | 5.59 | 500 | 4324.2 | 0.538 | 159 | 3107.9 | 0.686 | 221.9 |
| 10 | 3 | 4.16 | 12 | 238.5 | 0.150 | 69 | 383.0 | 0.179 | 238.5 |
| 10 | 4 | 6.61 | 500 | 2606.8 | 0.599 | 358 | 2837.4 | 2.102 | 221.7 |
| 15 | 3 | 5.87 | 18 | 237.1 | 0.198 | 154 | 516.6 | 0.346 | 237.1 |
| 15 | 4 | 8.36 | 57 | 1055.1 | 1.049 | 170 | 801.2 | 0.570 | 221.0 |
| 20 | 3 | 7.99 | 500 | 2164.8 | 1.620 | 233 | 719.0 | 1.092 | 236.4 |
| 20 | 4 | 8.19 | 500 | 932.1 | 1.108 | 195 | 1351.8 | 1.783 | 219.8 |

Here the picture changes, and it is not a simple "LM wins":

* **GN stops converging.** In 4 of the 8 runs, GN hit the 500-iteration limit. In the two we looked at more closely (8°/seed 4 and 20°/seed 3), GN ends up flipping between two states forever: the cost alternates 4549.4, 4324.2, 4549.4, 4324.2, ... This is the overshoot problem of post 03, at scale.
* **LM always stops**, and its cost never goes up. That is the seat belt working.
* **But stopping is not the same as succeeding.** Where the good answer has a cost of 220 to 240, LM often stops at 383, 517, 719 or even 3108. It has found a **local minimum**: a wrong solution where some loops are "folded" the wrong way, and no small change can lower the cost. Damping makes the search safe, not global.
* **Neither method is reliably better.** At 10° and 15° on seed 3, GN found the good answer and LM did not. At 15°/seed 4 and 20°/seed 3, LM ended lower than GN. On seed 4, both failed from 8° up.

The lesson matches post 01's two-beacon picture: when the initial guess is far enough off, especially in heading, the cost has many valleys, and *any* local method can land in the wrong one. Real systems attack this with a better initial guess, not a better local optimizer: for example, by first solving for the headings alone, or with chordal or other convex relaxations of the rotations, and with robust cost functions that limit the pull of bad loop closures.

## 12. Honest Caveats

* **Timings measure this implementation, not the algorithms.** Everything is Python with NumPy and SciPy. A GD iteration here is cheap mostly because it avoids a sparse factorization; a GN iteration is dominated by building \(J\), forming \(J^\top J\), and SuperLU. In C++, all of these numbers change, and so do their ratios. The wall-time plots compare these three scripts on one machine, nothing more.
* **Real systems use dedicated solvers.** g2o, Ceres and GTSAM are written in C++, use sparse Cholesky factorizations (often CHOLMOD) with careful variable ordering, exploit the block structure of the problem, and work directly on the SE(2)/SE(3) manifold. GTSAM can also update the solution incrementally as new measurements arrive (iSAM2). They solve a 400-pose 2D graph in milliseconds.
* **GD is here to show why nobody uses it for pose graphs.** With a condition number around \(10^5\), it needed 10,000 iterations to get within 8% of the optimum cost on the easy case, where GN needed 5. Plain GD is the wrong tool for this job, no matter how well it is implemented.
* **One small synthetic problem.** The grid world, the noise levels and the loop-closure rate are our choices. Real data has outliers (wrong loop closures), correlated noise, and much larger graphs. The *qualitative* lessons (GD is hopeless, GN and LM are fast near the answer, all local methods need a good start) are well known to hold in general; the exact numbers are not.

## 13. The Full Program

{{< code file="optimization/post05/pose_graph.py" title="Full benchmark program" >}}

```bash
python3 pose_graph.py .
```

It prints both tables, the sweep, the gauge and Jacobian checks, and writes the four figures. On the author's machine, the whole run, including the sweep and the 10,000-iteration GD runs, takes about 35 s.

## Key Takeaways

* A 2D pose graph has one residual per edge: the measured relative pose minus the predicted one, with the angle wrapped to \((-\pi, \pi]\). Each edge's Jacobian has two \(3 \times 3\) blocks, \(A_{ij}\) and \(B_{ij}\), which we derived and checked against finite differences.
* \(J\) and \(H = J^\top \Omega J\) are sparse, with the structure of the graph. Use sparse matrices and a sparse factorization.
* Relative measurements leave 3 gauge directions free: \(H\) has 3 zero eigenvalues. Fixing the first pose, a hard constraint, makes it invertible.
* On the easy start: GN 5 iterations, LM 9, both reaching the same optimum as SciPy to 13+ digits. GD was still at 5.5 times the optimal cost after 100 iterations.
* On the hard start, GN did *not* fail: it beat LM (8 vs 14 iterations). Only at much higher heading noise did GN stop converging, and there LM, while always stopping safely, often stopped in a wrong local minimum. A good initial guess matters more than the choice between GN and LM.

**Previous:** [Post 04: Levenberg-Marquardt](../post04-levenberg-marquardt/)
**Back to the start:** [Post 00a: The Least-Squares Problem, Words and Cost](../post00a-the-least-squares-problem/)
