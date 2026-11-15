---
title: "Post 01b: Nonlinear Least Squares and Linearization"
date: 2026-11-10T10:30:00+03:00
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
This is part two of post 01. In [post 01a](../post01a-linear-least-squares/), we solved linear least squares, \(r(x) = Ax - b\), in one step with the normal equations \(A^\top A x = A^\top b\). Here the residuals become *nonlinear* in \(x\), that formula is gone, and we need a new tool: linearization. That tool is the bridge to every method in the rest of the series.

## What We Will Learn

* A nonlinear robotics example: finding a robot from noisy distances to beacons.
* Why nonlinear least squares has no formula, and what its cost looks like.
* The Jacobian, and the first-order Taylor approximation \(r(x + \delta) \approx r(x) + J\delta\).{{< sidenote >}}The Jacobian \(J\) is a table of slopes: how much each residual changes when we nudge each unknown.{{< /sidenote >}}{{< sidenote >}}\(\delta\) (delta) is a small step, a change we add to \(x\). \(\approx\) means "approximately equal".{{< /sidenote >}}

## Table of Contents

4. A Nonlinear Problem: Range-Only Localization.
5. Why There Is No Formula.
6. Pictures: a Bowl vs a Curved Valley.
7. The Bridge: Linearize, Then Solve.

---

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

Try it yourself: drag the robot and the beacons, switch between 4 and 2 beacons, and show the rows of the Jacobian (section 7) to see why the valley is flat sideways.

<details><summary style="cursor:pointer;margin:.5em 0">Interactive: range-only localization, live</summary>

<iframe src="/tools/range-only-explorer/" title="Interactive range-only localization: drag the robot and beacons, and watch the range circles, the cost and its minima" style="width:100%;height:1250px;border:0" loading="lazy"></iframe>

<p><a href="/tools/range-only-explorer/">Open the range-only explorer in its own page</a>.</p>

</details>

In short, nonlinear least squares can have curved valleys, flat regions, and several minima. Our methods will only promise to find a *local* minimum, the bottom of the valley we start in. A good starting guess matters.

![The Rosenbrock function: a long, curved, banana-shaped valley, with the paths of two coordinate-descent methods](/images/numerical-optimization/post01/rosenbrock_2d.png)

*A famous hard landscape: the Rosenbrock function, \(F = (1 - x_1)^2 + 100\,(x_2 - x_1^2)^2\). It is a sum of two squares, so it is a tiny nonlinear least-squares problem, and its minimum at \((1, 1)\) sits at the end of a long, curved, banana-shaped valley. The dots are two coordinate-descent methods (not used in this series) creeping along that valley: the plain one needs 22,231 cost evaluations, the adaptive one 325.*

<small>Image: Ilya Loshchilov, via [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Rosenbrock2D.png), public domain. Modified: scaled down to 1200 px wide.</small>

## 7. The Bridge: Linearize, Then Solve

We cannot solve the nonlinear problem directly, but we *can* solve linear ones ([post 01a](../post01a-linear-least-squares/)). So here is the plan: near the current guess \(x\), pretend the problem is linear.

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

![(a) The residual of beacon 1 along a straight line through the current guess, with its tangent: they agree near the guess and drift apart further away, the gap shaded. (b) The linearization error against the step size on log-log axes: a straight line of slope 2 through the four printed values](/images/numerical-optimization/post01/linearization.png)

*(a) Walking along one direction from the current guess, the true residual (blue) bends, while the linear model (dashed) is its tangent and stays straight. Close to the guess they agree; a few meters away they are far apart. (b) The same gap, for the whole residual vector, on log-log axes: a straight line of slope 2, through the four numbers printed above.*

Now put the linear model into the cost:

$$
F(x + \delta) \approx \frac{1}{2} \lVert r(x) + J\delta \rVert^2 .
$$

Look closely: this is a **linear least-squares problem in \(\delta\)**, with \(A = J\) and \(b = -r(x)\). We already know how to solve it in one shot. So we can: linearize at \(x\), solve for the best \(\delta\), move to \(x + \delta\), linearize again, and repeat. That is the Gauss-Newton method of post 03. Before that, post 02 looks at the simplest idea of all: just walk downhill.

Both halves of this section, live: shrink the step and watch the gap fall like \(\lVert\delta\rVert^2\), then let the linear model choose a step, and see it overshoot from a far start.

<details><summary style="cursor:pointer;margin:.5em 0">Interactive: linearize, then solve</summary>

<iframe src="/tools/range-only-explorer/linearization.html" title="Interactive linearization: the tangent and its error, then the linear model's contours and the step to its minimum" style="width:100%;height:1750px;border:0" loading="lazy"></iframe>

<p><a href="/tools/range-only-explorer/linearization.html">Open the linearization explorer in its own page</a>.</p>

</details>

The full program for both parts of post 01 is at the end of [post 01a](../post01a-linear-least-squares/#4-the-full-program).

## Key Takeaways

* Nonlinear least squares, like range-only localization, has no closed-form answer, and its cost can have curved valleys and several minima.
* The Jacobian \(J = \partial r / \partial x\) gives the linear model \(r(x + \delta) \approx r(x) + J\delta\), with an error that shrinks like \(\lVert\delta\rVert^2\). That turns one hard nonlinear problem into a series of easy linear ones.

{{< quiz >}}
- q: 'Why is the range residual \(r_i(p) = \lVert p - b_i \rVert - z_i\) nonlinear in \(p\)?'
  options:
    - 'Because the beacon positions \(b_i\) are noisy.'
    - 'Because the unknowns \(p_x, p_y\) are inside a square root, so it cannot be written as \(Ap - b\).'
    - 'Because there are four beacons.'
  answer: 1
- q: 'What is row \(i\) of the Jacobian for range-only localization?'
  options:
    - 'The unit vector \((p - b_i)^\top / \lVert p - b_i \rVert\), pointing from beacon \(i\) to the robot.'
    - 'The beacon position \(b_i\).'
    - 'The measured distance \(z_i\).'
  answer: 0
  explain: 'Moving the robot along this direction changes the distance fastest; moving it sideways barely changes it.'
- q: 'Why is the 4-beacon cost a long, curved valley instead of a round bowl?'
  options:
    - 'Because the noise is too large.'
    - 'Because the beacons are all far away in one direction: moving sideways, along a circle around them, barely changes any distance.'
    - 'Because the robot is outside the room.'
  answer: 1
  explain: 'Toward or away from the beacons, all distances change and the cost rises fast. The valley bends because the circle bends.'
- q: 'With only 2 beacons, the cost has two minima. Which one does a search find?'
  options:
    - 'Always the true robot position.'
    - 'Always the one with the lower cost.'
    - 'It depends on where the search starts.'
  answer: 2
  explain: 'Our methods only promise a *local* minimum, the bottom of the valley we start in. A linear problem can never do this: a bowl has one bottom.'
- q: 'Halving the step \(\lVert\delta\rVert\) shrinks the linearization error \(\lVert r(x+\delta) - (r(x) + J\delta) \rVert\) by about how much?'
  options: ['2 times', '4 times', '10 times', 'Not at all']
  answer: 1
  explain: 'The error shrinks like \(\lVert\delta\rVert^2\). In the script, 10 times smaller steps gave 100 times smaller errors.'
- q: 'Putting the linear model into the cost gives \(\frac{1}{2}\lVert r(x) + J\delta \rVert^2\). What kind of problem is this in \(\delta\)?'
  options:
    - 'Nonlinear least squares, as hard as before.'
    - 'Linear least squares, with \(A = J\) and \(b = -r(x)\), solvable in one step.'
    - 'A problem with no minimum.'
  answer: 1
  explain: 'Linearize, solve for \(\delta\), step, and repeat: that is Gauss-Newton (post 03).'
- q: 'Why not use the classic trilateration trick (subtract squared ranges to get a linear system) as the final answer?'
  options:
    - 'It only works in 3D.'
    - 'It minimizes a different cost and weights the noise wrongly; it is good only as a first guess.'
    - 'It needs at least 10 beacons.'
  answer: 1
{{< /quiz >}}

**Previous:** [Post 01a: Linear Least Squares and the Normal Equations](../post01a-linear-least-squares/)
**Next:** [Post 02: Gradient Descent](../post02-gradient-descent/)
