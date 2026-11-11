---
title: "Post 01: Linear vs Nonlinear Least Squares"
date: 2026-11-10T10:00:00+03:00
weight: 3
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

In [post 00a](../post00a-the-least-squares-problem/) and [post 00b](../post00b-weights-constraints-and-a-first-fit/), we defined the cost \(F(x) = \frac{1}{2}\lVert r(x) \rVert^2\) and fitted a line, but we let NumPy find the answer for us.{{< sidenote >}}Reminder from post 00: \(x\) is the list of unknowns we want to find, \(r(x)\) the list of residuals (predicted minus measured), and \(F\) the cost: one number that says how bad a guess is.{{< /sidenote >}}{{< sidenote >}}\(\lVert r \rVert\) is the length of the list \(r\), so \(\lVert r \rVert^2\) means "square every residual and add them up". The \(\frac{1}{2}\) only makes derivatives tidier; it does not move the minimum.{{< /sidenote >}} In this post, we find it ourselves. When the residuals are *linear* in \(x\), there is a formula that gives the exact answer in one step. When they are *nonlinear*, there is no such formula, and we need a new tool: linearization. That tool is the bridge to every method in the rest of the series.

## What We Will Learn

* What makes a least-squares problem linear or nonlinear.
* How to derive the normal equations \(A^\top A x = A^\top b\), and why we solve them with Cholesky or QR instead of a matrix inverse.{{< sidenote >}}\(A^\top\) ("A transpose") is the table \(A\) turned on its side: its rows become columns.{{< /sidenote >}}
* What a condition number is, and why \(A^\top A\) squares it.
* A nonlinear robotics example: finding a robot from noisy distances to beacons.
* The Jacobian, and the first-order Taylor approximation \(r(x + \delta) \approx r(x) + J\delta\).{{< sidenote >}}The Jacobian \(J\) is a table of slopes: how much each residual changes when we nudge each unknown.{{< /sidenote >}}{{< sidenote >}}\(\delta\) (delta) is a small step, a change we add to \(x\). \(\approx\) means "approximately equal".{{< /sidenote >}}

## Table of Contents

1. Linear Least Squares.
2. The Normal Equations.
3. Solving Them Well: Cholesky, QR and Conditioning.
4. A Nonlinear Problem: Range-Only Localization.
5. Why There Is No Formula.
6. Pictures: a Bowl vs a Curved Valley.
7. The Bridge: Linearize, Then Solve.
8. The Full Program.

---

## 1. Linear Least Squares

A problem is **linear least squares** when every residual is a linear function of the state. Then we can write all of them at once with one matrix:

$$
r(x) = A x - b,
$$

where \(A\) is an \(m \times n\) matrix and \(b\) is a vector of \(m\) numbers.{{< sidenote >}}An \(m \times n\) matrix is a table of numbers with \(m\) rows and \(n\) columns: here, one row per measurement and one column per unknown.{{< /sidenote >}} Both are known and fixed.

The line fit from post 00 is linear. Its residual is \(r_i = a\, t_i + b_{\text{line}} - z_i\), and the state is \(x = (a, b_{\text{line}})\).{{< sidenote >}}The small \(i\) is a counter, like a seat number: \(r_1\) belongs to the first point, \(r_2\) to the second, and so on.{{< /sidenote >}} (Sorry for the two \(b\)'s: in this section, \(b\) without a label is the vector on the right side, and the intercept is \(b_{\text{line}}\).) Row \(i\) of \(A\) is \([\,t_i,\ 1\,]\), and the vector \(b\) is just the measurements \(z\):

$$
A = \begin{bmatrix} t_1 & 1 \\ t_2 & 1 \\ \vdots & \vdots \\ t_m & 1 \end{bmatrix}, \qquad
b = \begin{bmatrix} z_1 \\ z_2 \\ \vdots \\ z_m \end{bmatrix}.
$$

Note that "linear" is about \(x\), not about the data. We could fit a parabola \(a t^2 + b_{\text{line}} t + c\) and it would still be linear least squares, because the unknowns \(a, b_{\text{line}}, c\) appear only multiplied by known numbers.

## 2. The Normal Equations

Let us write out the cost:

$$
F(x) = \frac{1}{2}(Ax - b)^\top (Ax - b) = \frac{1}{2} x^\top A^\top A\, x - b^\top A\, x + \frac{1}{2} b^\top b .
$$

This is a **quadratic** function of \(x\): it has a squared term, a linear term, and a constant. In one dimension, that is a parabola. In two dimensions, it is a bowl.

At the bottom of a smooth bowl, the slope is zero in every direction. The slope of \(F\) is its gradient, the vector of partial derivatives.{{< sidenote >}}We write the gradient as \(\nabla F\) ("nabla F"). For each unknown, it says how fast the cost rises if we nudge that one; all together, it points uphill.{{< /sidenote >}} Taking the derivative of each term:

$$
\nabla F(x) = A^\top A\, x - A^\top b .
$$

Setting it to zero gives the **normal equations**:

$$
\boxed{A^\top A\, x = A^\top b}
$$

This is an ordinary \(n \times n\) linear system. Solve it, and we have the exact minimum. No iterations, no starting guess.

Is it really a minimum and not a maximum? The second derivative of \(F\) (the *Hessian*) is \(A^\top A\). For any vector \(v\), \(v^\top A^\top A v = \lVert A v \rVert^2 \ge 0\), so the bowl always opens upward. If the columns of \(A\) are independent, it is strictly positive, and the minimum is unique.

The name "normal" comes from geometry. At the solution, the residual \(Ax - b\) is perpendicular (normal) to every column of \(A\): that is exactly what \(A^\top (Ax - b) = 0\) says. For the line fit, the column of ones gives \(\sum_i r_i = 0\), which is the check we used in post 00.{{< sidenote >}}\(\sum_i\) (a capital Greek sigma) just means "add them all up": \(\sum_i r_i = r_1 + r_2 + \dots + r_m\).{{< /sidenote >}}

## 3. Solving Them Well: Cholesky, QR and Conditioning

The formula \(x = (A^\top A)^{-1} A^\top b\) is correct on paper, but we should never compute the inverse in code.{{< sidenote >}}\(^{-1}\) means inverse. For a number, it is "one over": \(4^{-1} = 1/4\). For a matrix, it is the matrix that undoes it, the matrix version of dividing.{{< /sidenote >}} It is slower and less accurate than solving the system directly. There are two standard ways.

**Cholesky.** \(A^\top A\) is symmetric and positive definite, so it can be factored as \(A^\top A = L L^\top\), where \(L\) is lower-triangular. Then we solve two triangular systems, which is cheap. This is the fastest method, and it is what large solvers (g2o, Ceres, GTSAM) use, in a sparse form.

**QR.** Factor \(A\) itself as \(A = QR\), where \(Q\) has orthonormal columns and \(R\) is upper-triangular. Then \(\lVert Ax - b \rVert = \lVert Rx - Q^\top b \rVert\) (plus a part that does not depend on \(x\)), and the solution is \(R x = Q^\top b\). QR never builds \(A^\top A\). This matters because of conditioning.

**Conditioning.** The **condition number** \(\kappa\) of a matrix is the ratio of its largest to its smallest stretching factor (singular value). It tells us how much small errors in the input, like rounding, can grow in the output. A rule of thumb: with \(\kappa = 10^k\), we can lose about \(k\) digits of accuracy. And here is the catch:

$$
\kappa(A^\top A) = \kappa(A)^2 .
$$

Forming \(A^\top A\) squares the condition number. If \(\kappa(A) = 10^6\), Cholesky on the normal equations can lose 12 of the 16 digits a double has, while QR loses about 6. For well-conditioned problems, Cholesky is fine and faster. For badly conditioned ones, QR (or SVD) is safer.

Here is the line fit, solved both ways:

{{< code file="optimization/post01/linear_vs_nonlinear.py" region="linear" title="Line fit via Cholesky and QR" >}}

Output:

```text {title="Output: Cholesky vs QR"}
line fit, Cholesky : [0.796935 1.583263]
line fit, QR       : [0.796935 1.583263]
numpy lstsq        : [0.796935 1.583263]
cond(A)      = 14.9
cond(A^T A)  = 223.2   (= cond(A)^2 = 223.2)
```

All three agree, and the script `assert`s that they match to \(10^{-10}\). The condition number of \(A^\top A\) is exactly the square of the condition number of \(A\). Here 223 is harmless, but remember this number: in post 02, it will decide how slow gradient descent is.

## 4. A Nonlinear Problem: Range-Only Localization

Now a robotics problem. A robot is somewhere in a room. Four radio beacons are at known positions. Each beacon measures its distance to the robot, with about 0.3 m of noise. Where is the robot?

The state is the robot position \(x = p = (p_x, p_y)\). The measurement model is the distance to beacon \(b_i\), so the residual is:

$$
r_i(p) = \lVert p - b_i \rVert - z_i = \sqrt{(p_x - b_{i,x})^2 + (p_y - b_{i,y})^2} - z_i .
$$

This residual is **nonlinear** in \(p\): the unknowns are inside a square root. We cannot write it as \(Ap - b\).

{{< code file="optimization/post01/linear_vs_nonlinear.py" region="range" title="Range residuals and Jacobian" >}}

The beacons sit close together in one corner (between 0 and 3 m), and the true robot position is \((7, 5)\), a few meters away. This is on purpose: it makes the problem harder in an interesting way, as we will see.

<img src="/images/numerical-optimization/post01/trilateration.svg" alt="Three circles around beacons P1, P2 and P3. Circles 1 and 2 cross at two points, A and B; circle 3 also passes through B." width="330" class="center">

*The idea behind range-only localization, drawn with exact distances: the distance to beacon P1 says "the robot is somewhere on this circle". Two circles cross at two points (A and B), and a third distance picks one of them (B). With noisy distances, the circles no longer meet in a single point, and least squares finds the best compromise.*

<small>Image: Rossi (original PNG) and Braindrain0000 (SVG), via [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Trilateration.svg), [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/). Modified: no.</small>

## 5. Why There Is No Formula

Let us try what worked before: set the gradient to zero. We need the derivative of a distance. For \(d(p) = \lVert p - b \rVert\):

$$
\frac{\partial d}{\partial p} = \frac{(p - b)^\top}{\lVert p - b \rVert},
$$

which is the unit vector pointing from the beacon to the robot.{{< sidenote >}}\(\partial d / \partial p\) ("partial d by partial p") asks: how much does \(d\) change when we nudge \(p\) a tiny bit, keeping everything else fixed?{{< /sidenote >}} So the gradient of \(F\) is:

$$
\nabla F(p) = \sum_i r_i(p)\, \frac{p - b_i}{\lVert p - b_i \rVert} = \sum_i \left(1 - \frac{z_i}{\lVert p - b_i \rVert}\right)(p - b_i) = 0 .
$$

These are two equations in two unknowns, but \(p\) appears inside square roots, in denominators, and multiplied with itself. There is no way to isolate \(p\) on one side. (There are algebraic tricks for trilateration that subtract squared ranges to get a linear system, but they minimize a *different* cost and weight the noise wrongly. They are good for a first guess, not for the least-squares answer.)

So we cannot jump to the answer. We must **search** for it: start from a guess and improve it step by step. Every method in posts 02 to 04 is such a search.

## 6. Pictures: a Bowl vs a Curved Valley

The best way to feel the difference is to draw \(F\) as a contour map, like a hiking map where each line connects points of equal cost.

![Cost contours: (a) the line fit is a bowl with one minimum, (b) range-only localization with 4 beacons is a curved valley, (c) with 2 beacons there are two minima](/images/numerical-optimization/post01/cost_contours.png)

**(a) Linear.** The line fit cost, over slope and intercept. Every contour is an ellipse, all centered on the same point. This is a perfect bowl: from anywhere, "go downhill" leads to the one minimum. The ellipses are stretched and tilted: along one direction, the cost rises much faster than along the other. (The tilt is because a steeper slope can be partly undone by a lower intercept.) The ratio of the two curvatures is the condition number of \(A^\top A\) again.

**(b) Nonlinear, 4 beacons.** The contours are no longer ellipses. Near the minimum at \((6.66, 5.34)\), there is a long, narrow, *curved* valley. Why? All four beacons are far away and in the same direction. Moving the robot toward or away from them changes all four distances, so the cost rises fast. Moving it *sideways*, along a circle around the beacons, barely changes any distance, so the cost rises slowly. The valley bends because the circle bends. A straight step "downhill" will soon climb the opposite wall.

**(c) Nonlinear, only 2 beacons.** Two circles cross at two points, so there are two positions that fit both ranges perfectly: \((6.81, 5.26)\) and its mirror image \((6.81, -5.26)\). The cost has **two minima**, and which one we find depends on where we start. A linear problem can never do this: a bowl has one bottom.

In short, nonlinear least squares can have curved valleys, flat regions, and several minima. Our methods will only promise to find a *local* minimum, the bottom of the valley we start in. A good starting guess matters.

![The Rosenbrock function: a long, curved, banana-shaped valley, with the paths of two coordinate-descent methods](/images/numerical-optimization/post01/rosenbrock_2d.png)

*A famous hard landscape: the Rosenbrock function, \(F = (1 - x_1)^2 + 100\,(x_2 - x_1^2)^2\). It is a sum of two squares, so it is a tiny nonlinear least-squares problem, and its minimum at \((1, 1)\) sits at the end of a long, curved, banana-shaped valley. The dots are two coordinate-descent methods (not used in this series) creeping along that valley: the plain one needs 22,231 cost evaluations, the adaptive one 325.*

<small>Image: Ilya Loshchilov, via [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Rosenbrock2D.png), public domain. Modified: scaled down to 1200 px wide.</small>

## 7. The Bridge: Linearize, Then Solve

We cannot solve the nonlinear problem directly, but we *can* solve linear ones (section 2). So here is the plan: near the current guess \(x\), pretend the problem is linear.

The **Jacobian** \(J\) is the matrix of all first derivatives of the residuals:

$$
J = \frac{\partial r}{\partial x}, \qquad J_{ij} = \frac{\partial r_i}{\partial x_j}, \qquad J \in \mathbb{R}^{m \times n}.
$$

Row \(i\) says how residual \(i\) changes when we nudge each variable.{{< sidenote >}}\(J \in \mathbb{R}^{m \times n}\) reads "\(J\) is a table of real numbers with \(m\) rows and \(n\) columns". \(J_{ij}\) is the entry in row \(i\), column \(j\).{{< /sidenote >}} For range-only localization, we already computed it in section 5: row \(i\) is the unit vector \((p - b_i)^\top / \lVert p - b_i \rVert\). The script checks this against finite differences.

With \(J\), the first-order Taylor expansion says that for a small step \(\delta\):

$$
r(x + \delta) \approx r(x) + J\,\delta .
$$

The right side is **linear** in \(\delta\). How good is it? The script measures the error for steps of shrinking size:

{{< code file="optimization/post01/linear_vs_nonlinear.py" region="taylor" title="Check the linearization error" >}}

```text {title="Output: linearization error"}
|d|      |r(p+d) - (r(p) + J d)|
1.0      4.435e-02
0.1      4.092e-04
0.01     4.060e-06
0.001    4.057e-08
```

Each time the step gets 10 times smaller, the error gets 100 times smaller. The error shrinks like \(\lVert\delta\rVert^2\), which is what "first-order" means: the linear model is very good for small steps, and worse for big ones.

Now put the linear model into the cost:

$$
F(x + \delta) \approx \frac{1}{2} \lVert r(x) + J\delta \rVert^2 .
$$

Look closely: this is a **linear least-squares problem in \(\delta\)**, with \(A = J\) and \(b = -r(x)\). We already know how to solve it in one shot. So we can: linearize at \(x\), solve for the best \(\delta\), move to \(x + \delta\), linearize again, and repeat. That is the Gauss-Newton method of post 03. Before that, post 02 looks at the simplest idea of all: just walk downhill.

## 8. The Full Program

{{< code file="optimization/post01/linear_vs_nonlinear.py" title="Full linear vs nonlinear program" >}}

```bash
python3 linear_vs_nonlinear.py .
```

## Key Takeaways

* Linear least squares (\(r = Ax - b\)) has a quadratic bowl as its cost and is solved in one step by the normal equations \(A^\top A x = A^\top b\).
* Solve them with Cholesky (fast) or QR (more accurate), never with an explicit inverse. Forming \(A^\top A\) squares the condition number: 14.9 became 223.2 in our line fit.
* Nonlinear least squares, like range-only localization, has no closed-form answer, and its cost can have curved valleys and several minima.
* The Jacobian \(J = \partial r / \partial x\) gives the linear model \(r(x + \delta) \approx r(x) + J\delta\), with an error that shrinks like \(\lVert\delta\rVert^2\). That turns one hard nonlinear problem into a series of easy linear ones.

**Previous:** [Post 00b: Weights, Constraints and a First Fit](../post00b-weights-constraints-and-a-first-fit/)
**Next:** [Post 02: Gradient Descent](../post02-gradient-descent/)
