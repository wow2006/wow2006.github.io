---
title: "Post 02: Gradient Descent"
date: 2026-11-10T11:00:00+03:00
weight: 4
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

At the end of [post 01](../post01-linear-vs-nonlinear-least-squares/), we saw that nonlinear least squares has no formula for the answer. We have to search: start somewhere, and improve step by step. The simplest search there is: look at which way is downhill, and take a step that way. That is **gradient descent** (GD). In this post, we derive it for least squares, add a line search so it never makes things worse, and watch it zig-zag down the curved valley of our range-only problem. Then we see exactly why it is slow.

## What We Will Learn

* How to derive the gradient of the least-squares cost: \(g = J^\top \Omega r\).{{< sidenote >}}\(g\) is the gradient: for each unknown, how fast the cost rises if we nudge it. It points uphill, so \(-g\) points downhill.{{< /sidenote >}}{{< sidenote >}}\(J\) is the Jacobian, a table of slopes of the residuals. \(J^\top\) ("J transpose") is that table turned on its side: rows become columns.{{< /sidenote >}}{{< sidenote >}}\(r\) is the list of residuals (predicted minus measured), and \(\Omega\) (omega) weights each one by how much we trust its sensor.{{< /sidenote >}}
* Why \(-g\) is the steepest way down.
* How to pick the step size with a backtracking (Armijo) line search.
* Why GD zig-zags, and what *linear convergence* means.
* How the condition number decides the speed, measured on two problems.

## Table of Contents

1. The Gradient of a Least-Squares Cost.
2. The Steepest Way Down.
3. How Big a Step? Backtracking Line Search.
4. GD on the Range-Only Problem.
5. Why It Zig-Zags: Linear Convergence and the Condition Number.
6. The Full Program.

---

## 1. The Gradient of a Least-Squares Cost

The **gradient** \(\nabla F(x)\) is the vector of partial derivatives of \(F\). Each entry says how fast the cost rises if we nudge one variable. We will call it \(g\).

Start with \(\Omega = I\):{{< sidenote >}}\(I\) is the identity matrix, the matrix version of the number 1. \(\Omega = I\) means every measurement gets the same weight.{{< /sidenote >}}

$$
F(x) = \frac{1}{2} \sum_i r_i(x)^2 .
$$

Take the derivative with respect to one variable \(x_j\).{{< sidenote >}}\(\sum_i\) means "add up over all measurements \(i\)". Small letters like \(i\) and \(j\) are just counters: \(x_j\) is unknown number \(j\).{{< /sidenote >}} By the chain rule, the derivative of \(\frac{1}{2} r_i^2\) is \(r_i \cdot \partial r_i / \partial x_j\) (the \(\frac{1}{2}\) cancels the 2, as promised in post 00):{{< sidenote >}}\(\partial r_i / \partial x_j\) is a partial derivative: how much \(r_i\) changes when we nudge only \(x_j\), keeping the other unknowns fixed.{{< /sidenote >}}

$$
\frac{\partial F}{\partial x_j} = \sum_i r_i \, \frac{\partial r_i}{\partial x_j} = \sum_i J_{ij}\, r_i .
$$

The sum over \(i\) of \(J_{ij} r_i\) is entry \(j\) of the matrix-vector product \(J^\top r\). So, for all \(j\) at once:

$$
\boxed{g = \nabla F(x) = J^\top r}
$$

With an information matrix, the cost is \(\frac{1}{2} \sum_i r_i^\top \Omega_i r_i\), and the same steps give \(g = J^\top \Omega\, r\) (where \(\Omega\) is block-diagonal with the \(\Omega_i\) on its diagonal). In code, we usually *whiten* the problem instead: multiply each residual and its Jacobian rows by \(\Omega_i^{1/2}\). Then \(\Omega\) disappears and every formula looks like the \(\Omega = I\) case. We will do that in post 05.

Notice what we need: only the residuals and the Jacobian. We already have both from post 01. For our range-only problem, the gradient is one line:

{{< code file="optimization/post02/gradient_descent.py" region="gradient" >}}

The script checks it against finite differences of \(F\).

## 2. The Steepest Way Down

Stand at \(x\) and take a tiny step in direction \(d\) (a unit vector).{{< sidenote >}}Below, \(\epsilon\) (epsilon) is the tiny length of that step, and \(g^\top d\) multiplies matching entries of \(g\) and \(d\) and adds them up. \(\approx\) means "approximately equal".{{< /sidenote >}} By the first-order Taylor expansion, the cost changes by about:

$$
F(x + \epsilon d) - F(x) \approx \epsilon\, g^\top d .
$$

To go down as fast as possible, we want \(g^\top d\) as negative as possible. A dot product of two vectors is smallest when they point in opposite directions, so the best \(d\) is \(-g / \lVert g \rVert\).{{< sidenote >}}\(\lVert g \rVert\) is the length of the vector \(g\). Dividing by it keeps the direction and makes the length 1.{{< /sidenote >}} The negative gradient is the **direction of steepest descent**. This gives the update:

$$
x \leftarrow x - \alpha\, g ,
$$

where \(\alpha > 0\) is the **step size** (in machine learning, it is called the *learning rate*).{{< sidenote >}}The arrow \(\leftarrow\) means "replace by": compute the right side, then store the result in \(x\).{{< /sidenote >}}

<img src="/images/numerical-optimization/post02/gradient_descent_commons.svg" alt="Contour lines of a bowl-shaped cost with gradient-descent steps x0, x1, x2, x3, x4 heading toward the centre" width="360" class="center">

*Gradient descent on a round bowl. The blue lines are contour lines (points of equal cost), and each red arrow is one step, \(x_0 \to x_1 \to x_2 \to \dots\), straight downhill. The steepest direction always crosses the contour line at a right angle. On a round bowl like this, that points almost straight at the minimum; section 5 shows what happens in a narrow one.*

<small>Image: Zerodamage, based on a figure by Oleg Alexandrov, via [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Gradient_descent.svg), public domain. Modified: no.</small>

## 3. How Big a Step? Backtracking Line Search

The direction is easy. The step size is the hard part.

* Too small, and we crawl.
* Too big, and we jump over the valley and land higher than where we started.

And the "right" size changes from place to place. So instead of fixing \(\alpha\), we search for a good one at every iteration. This is a **line search**: we search along the line \(x - \alpha g\) for a good \(\alpha\).

What counts as good? Not just "the cost went down". A step that reduces the cost by \(10^{-15}\) technically went down, but it is useless. The **Armijo condition** asks for a reduction that is at least a small fraction \(c\) of what the slope promised:

$$
F(x - \alpha g) \le F(x) - c\, \alpha\, \lVert g \rVert^2 , \qquad c = 10^{-4} .
$$

The term \(\alpha \lVert g \rVert^2\) is the drop that the linear model predicts. We ask for at least 0.01% of it. (Small \(c\) is the standard choice. The condition mainly rules out steps that are too long.)

**Backtracking** is the simplest way to satisfy it: try a step, and while the Armijo condition fails, halve \(\alpha\). Because \(-g\) points downhill, a small enough \(\alpha\) always works, so the loop always ends.

One small trick: at each iteration, we start from twice the last accepted step, not from 1. If the last step was fine, the next one probably is too, and this way the step can also grow again when the valley gets wider.

{{< code file="optimization/post02/gradient_descent.py" region="gd" title="Gradient descent with backtracking" >}}

Stopping rules: we stop when the gradient is tiny (\(\lVert g \rVert_\infty < 10^{-8}\)), when the step is tiny (\(< 10^{-10}\)), or after 1000 iterations.{{< sidenote >}}\(\lVert g \rVert_\infty\) is the largest entry of \(g\), ignoring its sign.{{< /sidenote >}} We will use the same rules for Gauss-Newton and Levenberg-Marquardt, so the comparison in post 04 is fair.

## 4. GD on the Range-Only Problem

We use the same four beacons and the same noisy ranges as in post 01, and start at \((-2, 8)\), far from the minimum at \((6.66, 5.34)\):

```text {title="Output: gradient descent run"}
start [-2.  8.], F = 7.5966
GD: 258 iterations, x = [6.661476 5.342091], F = 0.01802162
scipy least_squares: x = [6.661477 5.342091], F = 0.01802162
  iter    0: F - F* = 7.579e+00
  iter    1: F - F* = 6.562e+00
  iter    2: F - F* = 5.725e+00
  iter    5: F - F* = 3.598e+00
  iter   10: F - F* = 1.656e+00
  iter   50: F - F* = 4.387e-03
  iter  100: F - F* = 2.838e-06
  iter  200: F - F* = 1.489e-12
```

It works: GD reaches the same minimum as SciPy's solver. But it needs 258 iterations for a problem with only two unknowns. The path shows why:

![Gradient descent path on the range-only cost contours: a zig-zag down the curved valley](/images/numerical-optimization/post02/gd_path.png)

GD never walks *along* the valley. At each point, the steepest direction points mostly *across* the valley, toward the nearest wall. So it steps across, overshoots a bit, and the next gradient points back across. The result is a zig-zag that makes only a little progress along the valley floor with each step. The zoom on the right shows that the zig-zag continues all the way to the minimum.

![Cost gap F - F* vs iteration for gradient descent, a straight line on a log scale](/images/numerical-optimization/post02/gd_cost.png)

On a log scale, the cost gap is a straight line. A straight line on a log plot means the error is multiplied by the same factor at every step. This is called **linear convergence**. (The wiggle at the very end is rounding noise: we are at the limit of double precision.)

## 5. Why It Zig-Zags: Linear Convergence and the Condition Number

Close to the minimum, any smooth cost looks like a quadratic bowl, with curvatures given by the eigenvalues of the Hessian, \(\lambda_{\min}\) and \(\lambda_{\max}\).{{< sidenote >}}The Hessian is the table of second derivatives: how curved the cost is. Each eigenvalue \(\lambda\) (lambda) is its curvature along one main direction of the bowl. (In post 04, the same letter gets a different job.){{< /sidenote >}} Imagine a long, thin bowl:

* Across the bowl (the \(\lambda_{\max}\) direction), the walls are steep. A step longer than \(2/\lambda_{\max}\) in that direction would bounce up the other wall higher than we started. So the step size must stay below about \(2/\lambda_{\max}\).
* Along the bowl (the \(\lambda_{\min}\) direction), the floor is almost flat. With a step of about \(1/\lambda_{\max}\), we only remove a fraction \(\lambda_{\min}/\lambda_{\max} = 1/\kappa\) of the remaining error per step.

One step size must serve both directions, and the steep one wins. The classic result for GD on a quadratic (with the best possible step size) is that the error shrinks per iteration by at most

$$
\frac{\kappa - 1}{\kappa + 1}, \qquad \kappa = \frac{\lambda_{\max}}{\lambda_{\min}},
$$

and the cost gap, which is quadratic in the error, by the square of that. If \(\kappa = 1\) (a round bowl), one step is enough. If \(\kappa = 1000\), each step removes only about 0.2% of the error.

Let us check this on our problem. Near the minimum, the Hessian is close to \(J^\top J\) (post 03 explains why), so the script computes its eigenvalues there:

{{< code file="optimization/post02/gradient_descent.py" region="rate" title="Predict the convergence rate" >}}

```text
eig(J^T J) at x* = [0.134 3.866], kappa = 28.9
predicted rate ((k-1)/(k+1))^2 = 0.871, measured (iter 100->200) = 0.865
```

The prediction and the measurement agree well. The valley is 29 times steeper across than along, and so each GD step only keeps about 87% of the cost gap, that is, it removes 13%. To remove 12 orders of magnitude, we need about \(\ln(10^{-12}) / \ln(0.87) \approx 200\) steps, which is what we saw.{{< sidenote >}}\(\ln\) is the natural logarithm. Here it simply counts how many times we must multiply by 0.87 to shrink the gap by a factor \(10^{12}\).{{< /sidenote >}}

The beacon geometry causes this. All four beacons are on one side, so the ranges pin down the *distance* to them well, but the *direction* poorly. Bad geometry means a high condition number, and a high condition number means slow GD.

**A second test: the line fit.** The condition number of the line fit was 223 in post 01. A well-known trick reduces it: center the input, \(t \to t - \bar{t}\).{{< sidenote >}}\(\bar t\) ("t bar") is the average of all the \(t_i\). Subtracting it slides the data so that it is centered on zero.{{< /sidenote >}} This does not change the problem (the fitted line is the same, only the intercept is measured at \(\bar t\) instead of at 0), but it makes the two columns of \(A\) uncorrelated:

{{< code file="optimization/post02/gradient_descent.py" region="kappa" title="Raw vs centered line fit" >}}

```text
GD on the line fit (tolerance |g| < 1e-8):
  raw t       kappa(A^T A) =   223.2   iterations = 1601
  centered t  kappa(A^T A) =     7.7   iterations = 67
```

Same data, same answer, same algorithm. The condition number dropped 29 times, and the number of iterations dropped 24 times. This is the main lesson of GD: its speed is not set by how hard the problem is, but by how badly *scaled* it is. Rescaling the variables (called *preconditioning*) helps a lot, but finding a good scaling by hand for a problem with thousands of variables is not realistic.

In a pose graph, angles and positions have different units and different information weights, and the graph is long and thin. Its condition number is huge. In post 05, we will see what that does to GD. The fix is to use the curvature itself to choose the step, and that is Gauss-Newton, next.

## 6. The Full Program

{{< code file="optimization/post02/gradient_descent.py" title="Full gradient descent program" >}}

```bash
python3 gradient_descent.py .
```

## Key Takeaways

* The gradient of a least-squares cost is \(g = J^\top \Omega r\): only residuals and the Jacobian are needed.
* Gradient descent steps along \(-g\). A backtracking line search with the Armijo condition picks a step that is guaranteed to lower the cost enough.
* GD converges *linearly*: the cost gap shrinks by about the same factor each step, roughly \(((\kappa-1)/(\kappa+1))^2\). We predicted 0.871 and measured 0.865.
* In a narrow valley (high condition number), GD zig-zags across instead of walking along. Our 2-variable problem needed 258 iterations; centering the line-fit data cut GD's iterations from 1601 to 67.

**Previous:** [Post 01: Linear vs Nonlinear Least Squares](../post01-linear-vs-nonlinear-least-squares/)
**Next:** [Post 03: Gauss-Newton](../post03-gauss-newton/)
