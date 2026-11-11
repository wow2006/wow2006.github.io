---
title: "Post 03: Gauss-Newton"
date: 2026-11-10T12:00:00+03:00
weight: 5
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

Gradient descent in [post 02](../post02-gradient-descent/) only knew the *slope* of the cost, so it could not tell a steep wall from a long valley floor, and it zig-zagged for 258 iterations. If we also knew the *curvature*, we could jump straight to the bottom of the local bowl. That is Newton's method. For least squares, there is a cheap and clever version of it that needs nothing more than the Jacobian we already have: **Gauss-Newton** (GN). We will derive it in two different ways and get the same equation, run it on the range-only problem, and then break it on purpose.

## What We Will Learn

* Newton's method, and the exact Hessian of a least-squares cost.
* Gauss-Newton, derived twice: by dropping a term from Newton, and by solving the linearized problem.
* Why GN is fast when residuals are small, measured.
* When GN fails: singular \(J^\top J\), overshooting, and large residuals, with a run that goes off to infinity.{{< sidenote >}}\(J\) is the Jacobian, a table of slopes of the residuals. \(J^\top\) ("J transpose") is that table turned on its side, and \(J^\top J\) is the small square table we get by multiplying the two.{{< /sidenote >}}

## Table of Contents

1. Newton's Method.
2. The Hessian of a Least-Squares Cost.
3. Derivation 1: Drop the Second-Order Term.
4. Derivation 2: Linearize, Then Solve.
5. The Algorithm.
6. GN on the Range-Only Problem.
7. When Is GN Fast? The Dropped Term Decides.
8. When GN Fails.
9. The Full Program.

---

## 1. Newton's Method

GD used a first-order (linear) model of the cost. Newton's method uses a second-order (quadratic) one. Around the current \(x\), the Taylor expansion of \(F\) to second order is:{{< sidenote >}}As in post 00, \(x\) is the list of unknowns and \(F(x)\) the cost: one number that says how bad the guess \(x\) is.{{< /sidenote >}}

$$
F(x + \delta) \approx F(x) + g^\top \delta + \frac{1}{2} \delta^\top \nabla^2 F(x)\, \delta ,
$$

where \(\nabla^2 F\) is the **Hessian**, the \(n \times n\) matrix of second derivatives. It describes the curvature: how fast the slope itself changes.{{< sidenote >}}\(\delta\) (delta) is the step we are about to take, a small change of \(x\). \(\approx\) means "approximately equal".{{< /sidenote >}}{{< sidenote >}}\(g\) is the gradient: the slope of the cost, pointing uphill. \(g^\top \delta\) multiplies matching entries of \(g\) and \(\delta\) and adds them up.{{< /sidenote >}}{{< sidenote >}}The \(\frac{1}{2}\) is part of Taylor's formula: the slope of \(\delta^2\) is \(2\delta\), and the \(\frac{1}{2}\) cancels that 2.{{< /sidenote >}}{{< sidenote >}}An \(n \times n\) matrix is a square table of numbers, with one row and one column per unknown.{{< /sidenote >}}

This model is a quadratic bowl in \(\delta\), and from post 01 we know how to find the bottom of a bowl: set its gradient to zero. That gives \(g + \nabla^2 F\, \delta = 0\), or:

$$
\nabla^2 F(x)\, \delta = -g .
$$

<img src="/images/numerical-optimization/post03/newton_iteration.svg" alt="A curve y = f(x) with its tangent line at the point (x_n, f(x_n)); the tangent crosses the x-axis at x_{n+1}" width="400" class="center">

*Newton's method as Newton first used it: to find a root, the \(x\) where a curve \(f(x)\) crosses zero. At the current guess \(x_n\), replace the curve by its tangent line and jump to where the tangent hits zero; that is the next guess \(x_{n+1}\). Minimizing a cost is the same job one level up: at the minimum, the slope \(g\) is zero. So we apply the tangent trick to the slope. The slope of the slope is the Hessian, and "where the tangent of \(g\) hits zero" is exactly the equation above.*

<small>Image: Olegalexandrov (original) and Pbroks13 (vector version), via [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Newton_iteration.svg), public domain. Modified: no.</small>

Solve it, take the step, repeat. On a true quadratic, Newton lands on the minimum in one step, whatever the condition number. Near a minimum of any smooth function, it converges *quadratically*: the number of correct digits roughly doubles each iteration. The steep-wall-versus-flat-floor problem of GD is gone, because the Hessian measures both curvatures and scales the step for each direction separately.

<img src="/images/numerical-optimization/post03/newton_vs_gradient_descent.svg" alt="Contour lines of a cost; a green curved path (gradient descent) and a red, more direct path (Newton's method) from the same start to the minimum x" width="300" class="center">

*Gradient descent (green) and Newton's method (red) from the same start, both drawn with very small steps, so each path looks like a smooth curve. Gradient descent follows the steepest slope, which bends with the contour lines; Newton's method also uses the curvature and heads much more directly to the minimum \(x\).*

<small>Image: Oleg Alexandrov, via [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Newton_optimization_vs_grad_descent.svg), public domain. Modified: no.</small>

The price: we need the Hessian.

**A little history.** The two names in "Gauss-Newton" belong to people born more than a century apart. Isaac Newton described his method for finding the roots of an equation around 1669; Joseph Raphson published a simpler version in 1690, which is why it is also called Newton-Raphson. Carl Friedrich Gauss used least squares in 1801 to predict where the newly found dwarf planet Ceres would reappear after it was lost in the Sun's glare, and published the method in 1809 (Adrien-Marie Legendre had published it first, in 1805). The method of this post puts the two together: Newton's tangent trick, applied to Gauss's sum of squares.

<div style="display: flex; gap: 1.5rem; justify-content: center; flex-wrap: wrap; text-align: center;">
<figure style="margin: 0;"><img src="/images/numerical-optimization/post03/newton.jpg" alt="Portrait of Isaac Newton by Godfrey Kneller, 1689" width="150"><figcaption>Isaac Newton (1643&ndash;1727)</figcaption></figure>
<figure style="margin: 0;"><img src="/images/numerical-optimization/post03/gauss.jpg" alt="Portrait of Carl Friedrich Gauss by Gottlieb Biermann, 1887" width="150"><figcaption>Carl Friedrich Gauss (1777&ndash;1855)</figcaption></figure>
</div>

<small>Image (Newton): Godfrey Kneller, 1689, via [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:GodfreyKneller-IsaacNewton-1689.jpg), public domain. Modified: scaled down to 300 px wide.<br>
Image (Gauss): Gottlieb Biermann, 1887, after Christian Albrecht Jensen, via [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Carl_Friedrich_Gauss.jpg), public domain. Modified: scaled down to 300 px wide.</small>

## 2. The Hessian of a Least-Squares Cost

Let us compute it (with \(\Omega = I\)).{{< sidenote >}}\(\Omega\) (omega) weights each measurement by how much we trust it. \(\Omega = I\) means all weights are equal: \(I\) is the matrix version of the number 1.{{< /sidenote >}} From post 02, the gradient is{{< sidenote >}}\(\partial F / \partial x_j\) is a partial derivative: how much \(F\) changes when we nudge only the unknown \(x_j\). \(\partial^2\) means doing that twice.{{< /sidenote >}}

$$
\frac{\partial F}{\partial x_j} = \sum_i r_i \frac{\partial r_i}{\partial x_j} .
$$

Differentiate once more, with respect to \(x_k\).{{< sidenote >}}\(\sum_i\) means "add up over all residuals \(i\)", and \(r_i\) is residual number \(i\) (predicted minus measured). \(j\) and \(k\) are counters too: \(x_k\) is unknown number \(k\).{{< /sidenote >}} By the product rule, there are two terms:

$$
\frac{\partial^2 F}{\partial x_j \partial x_k} = \sum_i \frac{\partial r_i}{\partial x_k}\frac{\partial r_i}{\partial x_j} + \sum_i r_i \frac{\partial^2 r_i}{\partial x_j \partial x_k} .
$$

The first sum is entry \((j, k)\) of \(J^\top J\). The second one uses the Hessian of each residual, \(\nabla^2 r_i\). So:

$$
\nabla^2 F = \underbrace{J^\top J}_{\text{first derivatives only}} + \underbrace{\sum_i r_i \, \nabla^2 r_i}_{S \text{: second derivatives}} .
$$

We will call the second term \(S\).

## 3. Derivation 1: Drop the Second-Order Term

Gauss-Newton is Newton's method with \(S\) thrown away:{{< sidenote >}}In the formula below, \(\Rightarrow\) means "which leads to", and the box marks the result to remember.{{< /sidenote >}}

$$
\nabla^2 F \approx J^\top J
\qquad\Rightarrow\qquad
\boxed{J^\top J\, \delta = -J^\top r}
$$

With an information matrix, the same steps give \(H = J^\top \Omega J\) and \(H\delta = -J^\top \Omega r = -g\). This \(H\) is the **Gauss-Newton approximation of the Hessian**, and it is the \(H\) you will see in every SLAM paper.

Why is dropping \(S\) a good idea? Three reasons:

1. **It is free.** \(J^\top J\) needs only the Jacobian, which we already compute for the gradient. \(S\) needs a second-derivative matrix *for every residual*, which is expensive and easy to get wrong.
2. **It is small when the fit is good.** Each \(\nabla^2 r_i\) is multiplied by its residual \(r_i\). Near a solution that fits the data well, the residuals are small, so \(S\) is small, and \(J^\top J\) is close to the true Hessian.
3. **It always points downhill.** \(J^\top J\) is positive semi-definite (\(v^\top J^\top J v = \lVert Jv \rVert^2 \ge 0\)).{{< sidenote >}}\(\lVert Jv \rVert\) is the length of the vector \(Jv\). A squared length can never be negative.{{< /sidenote >}} If it is invertible, \(g^\top \delta = -g^\top (J^\top J)^{-1} g < 0\), so \(\delta\) is a descent direction.{{< sidenote >}}\(^{-1}\) means inverse, the matrix version of "divide by": \((J^\top J)^{-1}\) undoes multiplication by \(J^\top J\).{{< /sidenote >}} The full Hessian can have negative curvature far from the minimum, and then Newton's step can point uphill.

## 4. Derivation 2: Linearize, Then Solve

Now the second route, which we already prepared at the end of post 01. Replace each residual by its linear model:

$$
r(x + \delta) \approx r(x) + J\delta .
$$

Then the cost becomes

$$
F(x + \delta) \approx \frac{1}{2} \lVert J\delta + r \rVert^2 ,
$$

which is a *linear* least-squares problem in \(\delta\), of the form \(\lVert A\delta - b \rVert^2\) with \(A = J\) and \(b = -r\). The normal equations \(A^\top A \delta = A^\top b\) from post 01 give:

$$
J^\top J\, \delta = -J^\top r .
$$

The same equation. This second view is the more useful one in practice:

* GN does not approximate \(F\) directly. It approximates the **residuals** linearly, and then solves the resulting least-squares problem *exactly*.
* So each GN iteration is just a linear least-squares solve. We can use Cholesky on \(J^\top J\), or QR on \(J\) when the conditioning is bad, exactly like in post 01.
* The approximation is good when the residual functions are close to linear over the length of the step. It does not care whether the residuals are *small*. But derivation 1 shows that it *does* care, through \(S\), and section 7 will make this precise.

## 5. The Algorithm

{{< code file="optimization/post03/gauss_newton.py" region="gn" title="Gauss-Newton loop" >}}

Compared with GD, the only change is how the step is computed: instead of \(-\alpha g\), we solve the \(n \times n\) system \(H\delta = -g\). There is no step size and no line search. This is the *pure* Gauss-Newton method, the textbook version. It trusts its model completely and takes the full step, even if the cost goes up. Keep that in mind for section 8. The stopping rules are the same as for GD.

## 6. GN on the Range-Only Problem

Same beacons, same ranges, same start \((-2, 8)\) as in post 02 (case A):

```text {title="Output: case A (clean)"}
case A (clean)
  GN: 10 iterations, final x = [6.6615 5.3421], F = 0.0180216
  scipy least_squares: x = [6.6615 5.3421], F* = 0.0180216
  cost per iteration: [7.5966e+00 3.3456e+01 4.7852e+00 8.4401e-02 1.8381e-02 1.8022e-02 1.8022e-02 ...]
```

10 iterations instead of 258, and the same minimum as SciPy (the script asserts it). But look at the first step: the cost went **up**, from 7.60 to 33.46. GN made a big jump to \((6.1, 10.9)\), past the valley. From there, the linear model was good, and the next steps came straight down.

![Gauss-Newton paths: (A) clean ranges, converges in 10 iterations; (B) one range 4 m too long, the iterates run away from the minimum](/images/numerical-optimization/post03/gn_path.png)

![Cost gap vs iteration for Gauss-Newton on the clean problem, dropping about two orders of magnitude per iteration](/images/numerical-optimization/post03/gn_cost.png)

After the first wild step, the cost gap drops by about two orders of magnitude per iteration. Compare that with GD, which needed about 17 iterations for the same drop.

## 7. When Is GN Fast? The Dropped Term Decides

Can we predict that speed? Yes. Near the minimum \(x^*\), one GN step does this to the error \(e = x - x^*\) (a short Taylor argument, using \(\nabla^2 F = J^\top J + S\)):{{< sidenote >}}\(x^*\) ("x star") is the minimum we are looking for, so \(e\) is how far we still are from it. \(e_k\) is that error after \(k\) steps.{{< /sidenote >}}

$$
e_{k+1} \approx -(J^\top J)^{-1} S \; e_k .
$$

So the error is multiplied by the matrix \(-(J^\top J)^{-1} S\) at every step:

* If \(S = 0\), which happens when the residuals at the solution are zero (a perfect fit), GN behaves like Newton and converges **quadratically**.
* If \(S\) is small compared with \(J^\top J\), GN converges **linearly, but with a tiny factor**: very fast in practice.
* If \(S\) is as large as \(J^\top J\) in some direction, the factor is close to or above 1. Then GN is slow, or it **does not converge at all**, not even when started next to the answer.

The size of that factor is the largest eigenvalue (in absolute value) of \((J^\top J)^{-1} S\). The script computes it at the minimum. It gets the true Hessian by finite differences of the gradient, and subtracts \(J^\top J\) to get \(S\):

{{< code file="optimization/post03/gauss_newton.py" region="hessian_terms" title="Split the Hessian" >}}

For case A:

```text
  at x*: eig(J^T J) = [0.134 3.866], |S| = 0.011, GN local rate = 0.084  (< 1 needed, near 0 is fast)
```

The error is multiplied by 0.084 per step, so the cost gap (which is quadratic in the error) by about \(0.084^2 \approx 0.007\): two orders of magnitude, exactly what the plot shows. Note how the condition number, which made GD slow, does not appear here at all: GN undoes the stretch of the valley by solving with \(J^\top J\).

## 8. When GN Fails

Pure GN has no safety net. Here are the ways it breaks.

**1. \(J^\top J\) is singular or nearly singular.** Then \(H\delta = -g\) has no unique solution, or a huge one. This happens when the measurements do not pin down all the variables. Examples: too few beacons; a robot sitting exactly on a beacon (the unit vector is undefined); and, in every pose graph, the freedom to move the whole map (post 05). A nearly singular \(J^\top J\) is worse than an exactly singular one, because the solver does not complain. It just returns an enormous step.

**2. Overshoot from a far start.** The linear model is only good for small steps (post 01 showed its error grows like \(\lVert\delta\rVert^2\)). Far from the minimum, GN may take a step that the linear model loves and the real cost hates. We saw it in case A: the cost went from 7.60 to 33.46 in the first step. Here GN was lucky and recovered. Nothing guarantees that.

**3. Large residuals.** This is the subtle one. Suppose one beacon's signal bounces off a wall, so its range is 4 m too long (case B). Nothing else changes: same beacons, same start. Now the best fit cannot explain all four ranges, so the residuals at the minimum are large, and \(S\) is not small any more:

```text {title="Output: case B (bad range)"}
case B (one bad range)
  GN: 50 iterations, final x = [ 2.57386613e+08 -1.48854377e+07], F = 1.32939e+17
  scipy least_squares: x = [7.4584 5.9917], F* = 6.21294
  at x*: eig(J^T J) = [0.101 3.899], |S| = 0.105, GN local rate = 1.039  (< 1 needed, near 0 is fast)
  cost per iteration: [1.7023e+01 1.7465e+02 3.3513e+01 1.2933e+02 1.4270e+02 3.9746e+02 6.3499e+02 1.7282e+03 ...]
```

A minimum exists, at \((7.46, 5.99)\), and SciPy finds it. But GN bounces from side to side across it, further each time, and after 50 iterations it is \(2.6 \times 10^8\) m away. The right panel of the figure above shows the first steps, numbered.

Why? \(\lVert S \rVert = 0.105\) looks small next to the large eigenvalue of \(J^\top J\) (3.9). But what matters is the *weak* direction, where \(J^\top J\) has eigenvalue 0.101, as small as \(S\) itself. In that direction, dropping \(S\) is a big mistake, and the local rate becomes 1.039. A rate above 1 means GN is not even locally convergent: started a millimeter from the minimum, it would still drift away, while flipping sides at each step (the negative sign in \(e_{k+1} \approx -(J^\top J)^{-1} S\, e_k\)).

This is not a strange corner case. Bad geometry (a weak direction) plus one bad measurement (a large residual) is everyday robotics.

**4. Wrong basin.** Even when GN converges, it converges to the minimum of the basin it lands in. With the two-beacon problem of post 01, starting below the beacons leads to the mirror solution. This is true for every local method, not just GN.

All four failures come from the same root: GN trusts its linear model fully, everywhere, no matter how far the step goes. The fix is to trust it *only within a region where it has proved itself*, and to fall back toward gradient descent where it has not. That is Levenberg-Marquardt, next.

## 9. The Full Program

{{< code file="optimization/post03/gauss_newton.py" title="Full Gauss-Newton program" >}}

```bash
python3 gauss_newton.py .
```

## Key Takeaways

* The Hessian of a least-squares cost is \(J^\top J + S\), with \(S = \sum_i r_i \nabla^2 r_i\). Gauss-Newton drops \(S\) and solves \(J^\top \Omega J\, \delta = -J^\top \Omega r\).
* The same equation comes from linearizing the residuals and solving the linear least-squares problem exactly. Each GN iteration is a linear solve.
* Near the minimum, the error is multiplied by \(-(J^\top J)^{-1} S\) per step. Small residuals mean GN is very fast: 10 iterations instead of 258 on our problem, with a local rate of 0.084.
* Pure GN has no step control. It can increase the cost (7.60 to 33.46 in our first step), fail on a singular \(J^\top J\), and with one bad range (local rate 1.039) it ran \(2.6 \times 10^8\) m away from a perfectly good minimum.

**Previous:** [Post 02: Gradient Descent](../post02-gradient-descent/)
**Next:** [Post 04: Levenberg-Marquardt](../post04-levenberg-marquardt/)
