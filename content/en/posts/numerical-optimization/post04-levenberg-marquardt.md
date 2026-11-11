---
title: "Post 04: Levenberg-Marquardt"
date: 2026-11-10T13:00:00+03:00
weight: 6
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
---

We now have two methods with opposite strengths. Gradient descent ([post 02](../post02-gradient-descent/)) is safe: with a line search, every step lowers the cost. But it is slow. Gauss-Newton ([post 03](../post03-gauss-newton/)) is fast near the answer, but it trusts its linear model blindly: it can raise the cost, and with one bad range it ran off to infinity. **Levenberg-Marquardt** (LM) blends the two with a single knob, \(\lambda\), and turns that knob automatically, based on how well the linear model predicted the last step. It is the default nonlinear least-squares method in most libraries, and in this post we build it from scratch.

## What We Will Learn

* The damped normal equations \((J^\top J + \lambda I)\delta = -J^\top r\), and where they come from.{{< sidenote >}}\(J\) is the Jacobian, a table of slopes of the residuals \(r\) (predicted minus measured). \(J^\top\) ("J transpose") is that table turned on its side.{{< /sidenote >}}{{< sidenote >}}\(\delta\) (delta) is the step we solve for. \(I\) is the identity matrix, the matrix version of 1, so \(\lambda I\) just adds the number \(\lambda\) to each diagonal entry.{{< /sidenote >}}
* Why \(\lambda \to 0\) gives Gauss-Newton and \(\lambda \to \infty\) gives a small gradient-descent step.{{< sidenote >}}\(\lambda \to 0\) reads "\(\lambda\) shrinks toward zero"; \(\lambda \to \infty\) means "\(\lambda\) grows without limit".{{< /sidenote >}}
* Marquardt's scaling with \(\operatorname{diag}(J^\top J)\).{{< sidenote >}}\(\operatorname{diag}(J^\top J)\) keeps only the diagonal entries of \(J^\top J\) (top-left to bottom-right) and sets the rest to zero.{{< /sidenote >}}
* The gain ratio \(\rho\), and Nielsen's rule for updating \(\lambda\).{{< sidenote >}}\(\rho\) (rho) is the actual drop in cost divided by the drop the model promised. Section 4 defines it.{{< /sidenote >}}
* The trust-region view of LM.
* GD, GN and LM side by side on the same problem, from the same start.

## Table of Contents

1. Adding a Penalty for Long Steps.
2. The Two Limits.
3. Marquardt's Scaling.
4. Did the Model Predict Well? The Gain Ratio.
5. Turning the Knob: Nielsen's Rule.
6. The Algorithm.
7. The Trust-Region View.
8. Three Methods, One Problem.
9. The Bad-Range Case Again.
10. The Full Program.

---

## 1. Adding a Penalty for Long Steps

GN finds the step by minimizing the linear model of the cost:{{< sidenote >}}\(\lVert \cdot \rVert^2\) is a squared length: square every entry and add them up. The \(\frac{1}{2}\) only makes derivatives tidier; it does not move the minimum.{{< /sidenote >}}

$$
L(\delta) = \frac{1}{2} \lVert r + J\delta \rVert^2 .
$$

The trouble is that \(L\) is only a good model for *small* \(\delta\) (post 01: its error grows like \(\lVert\delta\rVert^2\)). GN ignores this and happily takes a 10 m step if the model says so.

Levenberg's idea (1944) was simple: keep the model, but add a cost for long steps:{{< sidenote >}}\(\min_\delta\) means "find the \(\delta\) that makes this as small as possible".{{< /sidenote >}}

$$
\min_\delta \; \frac{1}{2} \lVert r + J\delta \rVert^2 + \frac{1}{2} \lambda \lVert \delta \rVert^2 , \qquad \lambda > 0 .
$$

The new term is like a spring pulling the step back toward zero. \(\lambda\) is the **damping** parameter: how stiff the spring is.

To find the minimum, set the gradient with respect to \(\delta\) to zero:

$$
J^\top (r + J\delta) + \lambda \delta = 0
\qquad\Rightarrow\qquad
\boxed{(J^\top J + \lambda I)\, \delta = -J^\top r = -g}
$$

These are the **damped normal equations**.{{< sidenote >}}\(g = J^\top r\) is the gradient: the slope of the cost, pointing uphill. In the line above, \(\Rightarrow\) means "which leads to".{{< /sidenote >}} Compared with GN, we only add \(\lambda\) to the diagonal of \(J^\top J\). With an information matrix, it is \((J^\top \Omega J + \lambda I)\delta = -J^\top \Omega r\).{{< sidenote >}}\(\Omega\) (omega), the information matrix, weights each measurement by how much we trust it.{{< /sidenote >}}

Adding \(\lambda I\) also fixes the first failure of GN from post 03. \(J^\top J\) can be singular. \(J^\top J + \lambda I\) cannot: for any \(v \neq 0\), \(v^\top(J^\top J + \lambda I)v = \lVert Jv \rVert^2 + \lambda\lVert v \rVert^2 > 0\). So the system always has exactly one solution, and, as in post 03, that solution always points downhill (\(g^\top\delta < 0\)).

## 2. The Two Limits

**\(\lambda \to 0\).** The spring disappears, and the equation becomes \(J^\top J\delta = -g\): the **Gauss-Newton** step.

**\(\lambda \to \infty\).** \(\lambda I\) dominates \(J^\top J\), so \(\lambda\delta \approx -g\), or:{{< sidenote >}}\(\approx\) means "approximately equal".{{< /sidenote >}}

$$
\delta \approx -\frac{1}{\lambda}\, g .
$$

That is a **gradient-descent** step with step size \(1/\lambda\): it goes straight downhill, and it is very short.

So \(\lambda\) slides LM between the two methods. Small \(\lambda\): fast, trusting GN steps. Large \(\lambda\): short, careful GD steps. Everything in between is a blend: the direction turns smoothly from the GN direction toward \(-g\) while the step shrinks. The script checks both limits with `assert`s (\(\lambda = 10^{-12}\) gives the GN step, \(\lambda = 10^{8}\) gives \(-g/\lambda\)).

![The lambda dial: where the LM step ends for every lambda, from the same start, bending from the GN step to a short step along -g](/images/numerical-optimization/post04/lambda_dial.png)

*The \(\lambda\) dial, at the start point of section 8. The green curve shows where the LM step ends for every \(\lambda\) from \(10^{-4}\) to \(10^{4}\). With a tiny \(\lambda\), it is the long GN step (blue square). As \(\lambda\) grows, the step gets shorter and turns, and for a very large \(\lambda\) it is a short step along \(-g\), straight downhill (orange arrow, zoomed in on the right).*

## 3. Marquardt's Scaling

There is one problem with \(\lambda I\): it treats all variables the same. If one variable is an angle in radians and another is a position in millimeters, "a step of length 1" means very different things for them. In the GD limit, the step is \(-g/\lambda\), which is exactly the badly scaled GD we saw in post 02.

Marquardt (1963) proposed to damp each variable relative to its own curvature:

$$
\left(J^\top J + \lambda \operatorname{diag}(J^\top J)\right) \delta = -g .
$$

Now the large-\(\lambda\) step is \(\delta_j \approx -g_j / (\lambda (J^\top J)_{jj})\): each variable is divided by its own curvature, so the step no longer depends on the units we choose. This is a cheap form of the preconditioning we wished for at the end of post 02.

Both versions are used in practice. In this series, we use the plain \(\lambda I\) version, because that is the one the update rule below was designed and analyzed for in the Madsen, Nielsen and Tingleff notes [1]. In post 05 we whiten the residuals with \(\Omega^{1/2}\), which already puts all residuals in the same "sigma" units, so the plain version behaves well there.

## 4. Did the Model Predict Well? The Gain Ratio

Now the important part: how do we choose \(\lambda\)? We let the problem tell us. After computing a step \(\delta\), we compare two numbers:

* **Predicted reduction**: how much the linear model said the cost would drop, \(L(0) - L(\delta)\).
* **Actual reduction**: how much it really dropped, \(F(x) - F(x + \delta)\).{{< sidenote >}}\(F\) is the true cost and \(L\) is the linear model's guess of it, so the ratio below compares reality with the prediction.{{< /sidenote >}}

Their ratio is the **gain ratio**:

$$
\rho = \frac{F(x) - F(x + \delta)}{L(0) - L(\delta)} .
$$

* \(\rho \approx 1\): the model was right. We can trust it more, so lower \(\lambda\) (more GN-like).
* \(\rho\) small or negative: the model was wrong, maybe the cost even went up. Reject the step, raise \(\lambda\) (more GD-like, shorter), and try again from the same \(x\).

The predicted reduction has a neat closed form. Expand \(L\):

$$
L(0) - L(\delta) = -\delta^\top J^\top r - \frac{1}{2}\delta^\top J^\top J \delta .
$$

From the damped equations, \(J^\top J\delta = -g - \lambda\delta\). Substitute (and use \(J^\top r = g\)):

$$
L(0) - L(\delta) = -\delta^\top g + \frac{1}{2}\delta^\top g + \frac{1}{2}\lambda \delta^\top\delta = \frac{1}{2}\delta^\top(\lambda\delta - g) .
$$

Both parts are positive (\(\lambda\delta^\top\delta > 0\), and \(-\delta^\top g > 0\) because \(\delta\) points downhill), so the predicted reduction is always positive and the ratio is well defined.

## 5. Turning the Knob: Nielsen's Rule

The classic rule is "if the step worked, divide \(\lambda\) by 10; if not, multiply by 10". It works, but it jumps. We use the smoother rule from Nielsen (1999), as given in Algorithm 3.16 of Madsen, Nielsen and Tingleff [1]:

* **If \(\rho > 0\)** (the cost went down): accept the step, and set{{< sidenote >}}\(\leftarrow\) means "replace by": compute the right side and store it. \(\max(a, b)\) is the larger of the two numbers.{{< /sidenote >}}
  $$
  \lambda \leftarrow \lambda \cdot \max\left(\tfrac{1}{3},\; 1 - (2\rho - 1)^3\right), \qquad \nu \leftarrow 2 .
  $$
* **Otherwise**: reject the step, and set \(\lambda \leftarrow \lambda\,\nu\), \(\nu \leftarrow 2\nu\).

Some values of the factor: \(\rho = 1\) gives \(\tfrac{1}{3}\) (great prediction, damp less); \(\rho = 0.5\) gives 1 (keep \(\lambda\)); \(\rho\) close to 0 gives almost 2 (poor prediction, damp more). The extra variable \(\nu\) makes repeated failures grow \(\lambda\) faster and faster: \(\times 2\), then \(\times 4\), then \(\times 8\), and so on.

The starting value is \(\lambda_0 = \tau \cdot \max_j (J^\top J)_{jj}\), relative to the size of \(J^\top J\), so it does not depend on the units of the problem.{{< sidenote >}}\(\max_j (J^\top J)_{jj}\) is the largest diagonal entry of \(J^\top J\): the small \(j\) counts through the unknowns. \(\tau\) (tau) is a small fixed number.{{< /sidenote >}} The notes suggest a small \(\tau\) like \(10^{-6}\) if we believe the start is good, and \(10^{-3}\) or larger otherwise. We use \(\tau = 10^{-3}\).

## 6. The Algorithm

{{< code file="optimization/post04/levenberg_marquardt.py" region="lm" title="Levenberg-Marquardt loop" >}}

Note one detail: a rejected step still counts as an iteration, because it costs a linear solve and a cost evaluation, just like an accepted one. When we count iterations below, we count both.

## 7. The Trust-Region View

There is a second way to see LM. For any \(\lambda > 0\), the LM step \(\delta_\lambda\) is also the solution of

$$
\min_\delta L(\delta) \quad \text{subject to} \quad \lVert\delta\rVert \le \Delta, \qquad \Delta = \lVert \delta_\lambda \rVert .
$$

In words: LM minimizes the linear model, but only inside a ball of radius \(\Delta\) where we *trust* it. A large \(\lambda\) means a small ball. The gain ratio decides whether the ball grows (good prediction) or shrinks (bad prediction). Methods that control \(\Delta\) directly instead of \(\lambda\) are called **trust-region methods**. Dogleg (used in g2o and GTSAM as an option) and SciPy's `trf` are in this family, and Ceres also describes its LM as a trust-region method. They all share the same core idea: accept a step only when the model has earned it.

![One LM step drawn as a trust-region step: a dashed circle around the current point, the long GN step leaving it, and the shorter LM step ending on the circle](/images/numerical-optimization/post04/trust_region.png)

*One LM step, seen as a trust-region step. The current point is the start of section 8, and \(\lambda = 0.20\) is the first value LM accepts there. The dashed circle is the region where we trust the linear model; its radius is the length of that step, 3.4 m. The green dotted lines are contours of the linear model \(L\). Without a limit, its minimum is the GN step (blue). Inside the circle, the best we can do is the green step: it ends where a model contour just touches the circle.*

## 8. Three Methods, One Problem

Same beacons, same ranges, same start \((-2, 8)\), same stopping rules (\(\lVert g\rVert_\infty < 10^{-8}\) or step \(< 10^{-10}\)):

```text {title="Output: GD vs GN vs LM"}
start [-2.  8.], F = 7.5966;  scipy (MINPACK lm) F* = 0.0180216191
  GD:  258 iterations (258 accepted steps), x = [6.661476 5.342091], F - F* = 4.2e-17
  GN:   10 iterations (10 accepted steps), x = [6.661477 5.342091], F - F* = -4.2e-17
  LM:   13 iterations (10 accepted steps), x = [6.661477 5.342091], F - F* = 1.0e-16
```

All three reach the minimum that SciPy's MINPACK LM finds (the script asserts it). The tiny negative \(F - F^*\) only means GN is \(10^{-17}\) below SciPy, at the level of rounding noise.

![GD, GN and LM paths from the same start on the range-only cost](/images/numerical-optimization/post04/all_paths.png)

![Cost gap vs iteration for GD, GN and LM](/images/numerical-optimization/post04/all_costs.png)

The paths tell the story:

* **GD** zig-zags along the valley, 258 iterations.
* **GN** jumps far past the valley on its first step (the cost goes from 7.60 to 33.46), then comes back down fast, 10 iterations.
* **LM** never goes uphill. Look at its first iterations:

```text
  LM lambda per iteration: [3.159e-03 6.319e-03 2.527e-02 2.022e-01 1.395e-01 1.155e-01 5.058e-02 2.286e-02 ...]
  LM cost per iteration:   [7.5966 7.5966 7.5966 7.5966 3.9146 1.6155 0.33   0.0495 0.0182 0.018 ...]
```

The first step, with a tiny \(\lambda\), is almost the GN step that overshoots. LM computes it, sees that the cost went up (\(\rho < 0\)), and rejects it: \(\lambda\) doubles. Rejected again: \(\times 4\). Then \(\times 8\), to \(\lambda = 0.20\). Now the step is shorter and bent toward the gradient, and it is accepted. From then on, \(\rho\) is good at every step, \(\lambda\) shrinks by up to 3 per step, and LM turns into GN for the final fast approach.

So on this clean problem, LM needs 3 more iterations than GN: the 3 rejected steps are the price of never going uphill. The cost curve shows that after the start, LM and GN converge at the same fast rate. That is the general pattern: LM is "GN with a seat belt". When GN would have been fine, the seat belt costs a little. When GN would have crashed, it saves the run. That is the next section.

## 9. The Bad-Range Case Again

In post 03, one range was 4 m too long, and GN ran away to \(2.6 \times 10^8\) m. Same data, same start, LM:

```text
case B (bad range): LM 28 iterations (25 accepted), x = [7.4584 5.9917], F = 6.212942, scipy F* = 6.212942
```

LM converges to the same minimum as SciPy. Near that minimum, GN's local rate was 1.039 (post 03): pure GN steps make the error slightly *bigger* each time. LM's gain ratio sees the cost go up, raises \(\lambda\), and the damped steps converge. The extra 15 iterations compared with the clean case are the cost of fighting a large-residual problem; but the answer is correct, and the cost never increased on the way.

One thing LM does *not* fix: wrong basins. Like GN and GD, it finds the minimum of the valley it starts in. With two beacons (post 01), starting below them still leads to the mirror solution. Damping makes the search safe, not global.

## 10. The Full Program

{{< code file="optimization/post04/levenberg_marquardt.py" title="Full Levenberg-Marquardt program" >}}

```bash
python3 levenberg_marquardt.py .
```

## Key Takeaways

* LM solves \((J^\top \Omega J + \lambda I)\delta = -J^\top \Omega r\): the GN model plus a penalty \(\frac{1}{2}\lambda\lVert\delta\rVert^2\) on long steps. The matrix is always invertible.
* \(\lambda \to 0\) gives the GN step; \(\lambda \to \infty\) gives a short GD step \(-g/\lambda\). Marquardt's \(\operatorname{diag}(J^\top J)\) version makes the damping independent of units.
* The gain ratio \(\rho\) (actual over predicted reduction, with predicted \(= \frac{1}{2}\delta^\top(\lambda\delta - g)\)) decides whether to accept a step and how to change \(\lambda\). Nielsen's rule does this smoothly.
* On our problem: GD 258 iterations, GN 10, LM 13 (3 of them rejected steps). With one bad range, GN diverged and LM converged in 28 iterations to the same minimum as SciPy.
* LM is a trust-region method in disguise. It makes the search safe, but it still finds only a local minimum.

## References

[1] K. Madsen, H. B. Nielsen, O. Tingleff, *Methods for Non-Linear Least Squares Problems*, 2nd ed., IMM, Technical University of Denmark, 2004. Section 3.2 and Algorithm 3.16.

**Previous:** [Post 03: Gauss-Newton](../post03-gauss-newton/)
**Next:** [Post 05: A 2D Pose-Graph Benchmark](../post05-pose-graph-benchmark-gd-gn-lm/)
