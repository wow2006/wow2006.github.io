---
title: "Post 00b: Weights, Constraints and a First Fit"
date: 2026-11-10T09:00:00+03:00
weight: 2
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
This is part two of post 00. In [post 00a](../post00a-the-least-squares-problem/), we defined the state \(x\), the measurements \(z_i\), the residuals \(r_i\) and the cost \(F\). Here we add weights, clear up the word "constraint", and fit a first line.

## What We Will Learn

* What the *information matrix* is, and why some measurements count more than others.
* The two meanings of the word *constraint*, and which one this series is about.
* A tiny example: fitting a line to noisy points.

## Table of Contents

4. Trusting Some Measurements More: the Information Matrix.
5. Two Meanings of "Constraint".
6. Example: Fitting a Line.
7. Notation for the Whole Series.

---

## 4. Trusting Some Measurements More: the Information Matrix

Not all sensors are equally good. A laser rangefinder may be accurate to 1 cm; wheel odometry may be off by 5 cm per meter. If we add their squared residuals with the same weight, the bad sensor can pull the answer as much as the good one.

The fix is to weight each residual by how much we trust it. If measurement \(i\) has noise with covariance \(\Sigma_i\) (for a single number, \(\Sigma_i = \sigma_i^2\), the variance), we use its inverse:{{< sidenote >}}Careful: this \(\Sigma_i\) is not a sum. It is the covariance: how spread out sensor \(i\)'s noise is. Its small cousin \(\sigma\) (sigma) is the standard deviation, the typical size of the noise.{{< /sidenote >}}{{< sidenote >}}\(^{-1}\) means inverse. For one number, it is "one over": \(4^{-1} = 1/4\). So small noise gives a big weight.{{< /sidenote >}}

$$
\Omega_i = \Sigma_i^{-1}.
$$

\(\Omega_i\) is called the **information matrix**. Small noise means big information. The cost becomes:

$$
F(x) = \frac{1}{2} \sum_i r_i(x)^\top \, \Omega_i \, r_i(x).
$$

For a single number, this is just \(\frac{1}{2} \sum_i r_i^2 / \sigma_i^2\): each residual is measured in units of its own standard deviation.{{< sidenote >}}\(^\top\) (transpose) turns a column of numbers into a row. \(r^\top \Omega\, r\) is a weighted sum of the squared entries of \(r\) when \(\Omega\) is diagonal (independent noise); a full \(\Omega\) also adds cross terms \(r_j \Omega_{jk} r_k\) with \(j \ne k\). With all weights 1, it is just \(\lVert r \rVert^2\).{{< /sidenote >}} A 5 cm error on a sensor with \(\sigma = 1\) cm counts as "5 sigmas", which is a lot. The same 5 cm error on a sensor with \(\sigma = 5\) cm counts as "1 sigma", which is normal.

When all measurements are equally good, \(\Omega_i = I\) (the identity matrix) and we are back to \(\frac{1}{2}\lVert r \rVert^2\).{{< sidenote >}}\(I\) is the matrix version of the number 1: multiplying by it changes nothing.{{< /sidenote >}} In posts 00 to 04, we use \(\Omega = I\) to keep the math short. In post 05, the pose graph has position and angle residuals with very different units, and \(\Omega\) becomes important.

### What is covariance, really?

**Covariance is a number (or a small table of numbers) that says how much a measurement wanders around the truth.** Imagine measuring a target that is exactly 0 cm away, many times. A good sensor gives answers close to 0; a bad one gives answers all over the place. The *variance* \(\sigma^2\) is the average squared distance of the answers from the truth. Small variance means a narrow bump, large variance means a wide one:

![Two bell curves centred on zero: a tall narrow one for sigma = 1 cm and a short wide one for sigma = 5 cm, with the weight 1/sigma^2 in the legend](/images/numerical-optimization/post00/covariance_1d.png)

*The narrow sensor (blue) has variance 1 and weight 1. The wide sensor (orange) has variance 25 and weight 1/25, so its residuals count 25 times less.*

That is the whole idea for one number. When a measurement has *several* numbers (for example an \(x\) and a \(y\) position), noise in one can be linked to noise in the other, so we need a table. For two numbers it is a \(2 \times 2\) matrix:

$$
\Sigma = \begin{pmatrix} \sigma_x^2 & \sigma_{xy} \\ \sigma_{xy} & \sigma_y^2 \end{pmatrix}.
$$

* The **diagonal** entries are the variances of each number on its own: how much \(x\) wanders, how much \(y\) wanders.
* The **off-diagonal** entry \(\sigma_{xy}\) is the *covariance proper*: do the two errors move together? Zero means "independent". Positive means "when \(x\) is too big, \(y\) tends to be too big as well".

The picture below shows 400 noisy readings for three different \(\Sigma\). Each grey dot is one measurement error; the red ellipse is the 1-sigma (solid) and 2-sigma (dashed) contour, which contains most of the dots.

![Three clouds of dots with red ellipses: a round cloud for the identity covariance, a wide flat cloud when x is noisier than y, and a tilted thin cloud when x and y errors are correlated](/images/numerical-optimization/post00/covariance_2d.png)

*Left: equal, independent noise gives a circle. Middle: \(x\) is noisier than \(y\), so the cloud is stretched along \(x\). Right: the errors are correlated, so the cloud leans along the diagonal.*

Now the link to the cost. The inverse \(\Omega = \Sigma^{-1}\) turns the picture upside down: directions where the cloud is **long** (lots of noise) get a **small** weight, and directions where it is **thin** (little noise) get a **big** weight. So \(r^\top \Omega\, r\) measures how far the residual \(r\) is from zero *in units of the ellipse*, not in plain centimetres. A residual of 3 cm along the long axis of the ellipse is unremarkable; the same 3 cm along the thin axis is a surprise, and the cost punishes it much more.

For a diagonal \(\Sigma\) (no correlation) this reduces exactly to the per-number weights \(1/\sigma^2\) above. You can ignore the off-diagonal part until post 05.

## 5. Two Meanings of "Constraint"

In robotics papers and libraries, you will see the word **constraint** all the time. Be careful: it is used for two different things.

**(a) Soft constraints: residual terms.** In a pose graph, every measurement between two poses is drawn as an edge, and these edges are usually called *constraints*. "Pose 7 is about 1 m in front of pose 6" is a constraint in this sense. But it does not have to hold exactly. It only adds a term \(\frac{1}{2} r_i^\top \Omega_i r_i\) to the cost, and the optimizer is free to break it a little if other measurements disagree. The information matrix says how expensive breaking it is. That is why it is *soft*.

**(b) Hard constraints: rules that must hold exactly.** These come from classical constrained optimization: *equality* constraints like \(c(x) = 0\) and *inequality* constraints like \(c(x) \le 0\). They are not part of the cost. No amount of cost reduction is allowed to break them. Examples: "the first pose is at the origin", or "the speed is at most 2 m/s".

This series is almost entirely about **(a)**. Every method we build minimizes a sum of soft terms. We meet **(b)** only once, in post 05: a pose graph only contains *relative* measurements, so we can move the whole map without changing any residual. To get a single answer, we fix the first pose exactly. This is called **gauge fixing**, and it is a hard equality constraint. We will handle it the simple way: we just do not let the solver change that pose.

![Flow diagram: variables x go into the model h_i(x); its prediction and the observation z_i give the residual r_i = h_i(x) - z_i; squaring and adding gives the cost F; the optimizer changes x to make F smaller and the loop repeats; a side box says soft constraints are residual terms and hard constraints must hold exactly](/images/numerical-optimization/post00/vocabulary_loop.png)

*The least-squares loop: everything up to the cost F is defined in this post, and posts 02–04 are three different ways to fill in the optimizer box.*

## 6. Example: Fitting a Line

Let us make this concrete with the smallest problem we know. We measure a quantity at 20 places \(t_1, \dots, t_{20}\), and we believe it grows along a straight line. The state is the slope and the intercept of that line:

$$
x = (a, b), \qquad h_i(x) = a\, t_i + b, \qquad r_i(x) = a\, t_i + b - z_i .
$$

Here \(t_i\) is *where* we measured (it is known, not part of the state), and \(z_i\) is what we measured there. We generate the data ourselves from a true line \(a = 0.8\), \(b = 1.5\), plus Gaussian noise with \(\sigma = 0.6\), using a fixed random seed so every run gives the same points:

{{< code file="optimization/post00/line_fit.py" region="data" title="Generate noisy line data" >}}

The model, the residuals and the cost are three tiny functions that follow the definitions above word for word:

{{< code file="optimization/post00/line_fit.py" region="model" title="Model, residuals and cost" >}}

We will learn how to *find* the best line in the next post. For now, the script uses `numpy.linalg.lstsq` and prints:

```text {title="Output: true vs best line"}
true x  = [0.8 1.5]
best x  = [0.7969 1.5833]
F(true) = 2.2048
F(best) = 2.1610
```

![Noisy points, the fitted line, and the residuals drawn as vertical segments](/images/numerical-optimization/post00/line_fit_residuals.png)

The pink segments are the residuals. Least squares picks the line that makes the sum of their squared lengths as small as possible. Note that the residuals are *vertical*, not perpendicular to the line: we assume the \(t_i\) are exact and only the \(z_i\) are noisy.

![The line-fit example with labels: black dots are the observations z_i, the blue line is the model h_i(x) = a t_i + b, a box gives the variables x = (a, b), and the residuals r_i = h_i(x) - z_i are drawn as springs, one highlighted, with the cost F as their total energy](/images/numerical-optimization/post00/vocabulary_annotated.png)

*The same fit with every word labelled: if each residual is a spring that stores ½ r_i² of energy, the least-squares line is where the springs leave it at rest, at the smallest total energy F.*

One result may surprise you: the best line has a **lower** cost than the true line (2.1610 vs 2.2048). This is not a bug. The best line fits *this particular* noise as well as possible, and the true line does not care about our noise. With more data, the best line gets closer to the true one, but it fits the data it has, not the truth.

The script has a few `assert` checks at the end: moving the best \(x\) a little in any direction makes the cost worse, and the residuals of a line with an intercept sum to zero.

{{< code file="optimization/post00/line_fit.py" title="Full line-fit program" >}}

To run it:

```bash
python3 line_fit.py .
```

## 7. Notation for the Whole Series

We will use these symbols in every post:{{< sidenote >}}\(J = \partial r / \partial x\) ("partial r by partial x") is a table of slopes: how much each residual changes when we nudge each unknown. It has \(m\) rows (residuals) and \(n\) columns (unknowns).{{< /sidenote >}}{{< sidenote >}}The gradient \(g\) points uphill: for each unknown, how fast the cost rises if we nudge it.{{< /sidenote >}}{{< sidenote >}}\(H\) describes the curvature of the cost: how fast the slope itself changes.{{< /sidenote >}}{{< sidenote >}}\(\delta\) (delta) is a small step, and \(\leftarrow\) means "replace by". \(\boxplus\) ("box-plus") is addition that also wraps angles back into one turn.{{< /sidenote >}}

| symbol | meaning |
|---|---|
| \(x \in \mathbb{R}^n\) | the state (variables) |
| \(z_i\) | observation \(i\) |
| \(h_i(x)\) | measurement model: what we predict sensor \(i\) reads |
| \(r_i(x) = h_i(x) - z_i\) | residual (predicted minus measured); stacked: \(r(x) \in \mathbb{R}^m\) |
| \(\Omega_i = \Sigma_i^{-1}\) | information matrix (inverse covariance) |
| \(F(x) = \frac{1}{2}\sum_i r_i^\top \Omega_i r_i\) | cost; \(= \frac{1}{2}\lVert r \rVert^2\) when \(\Omega = I\) |
| \(J = \partial r / \partial x\) | Jacobian, an \(m \times n\) matrix (post 01) |
| \(g = J^\top \Omega r\) | gradient of \(F\) (post 02) |
| \(H = J^\top \Omega J\) | Gauss-Newton approximation of the Hessian (post 03) |
| \(x \leftarrow x + \delta\) | update step; for poses \(x \leftarrow x \boxplus \delta\), with angle wrapping (post 05) |

## Key Takeaways

* "Constraint" means two things: a soft residual term (a pose-graph edge), or a hard rule that must hold exactly. This series is about soft terms; the only hard one is fixing the first pose.
* With noisy data, the best fit is not the truth. It is the best explanation of the data we have.

{{< quiz >}}
- q: 'Sensor A has \(\sigma = 1\) cm and sensor B has \(\sigma = 5\) cm. With \(\Omega_i = \Sigma_i^{-1}\), how much does a squared residual of B count compared with A?'
  options: ["5 times more", "The same", "1/5 as much", "1/25 as much"]
  answer: 3
  explain: 'For one number the weight is \(1/\sigma^2\): \(1/1\) for A and \(1/25\) for B. Small noise means big information.'
- q: 'In a \(2 \times 2\) covariance \(\Sigma\), what does a positive off-diagonal \(\sigma_{xy}\) mean?'
  options:
    - 'When the \(x\) error is too big, the \(y\) error tends to be too big as well.'
    - 'The \(x\) and \(y\) errors are independent.'
    - '\(y\) is noisier than \(x\).'
  answer: 0
  explain: 'Zero means independent. A non-zero \(\sigma_{xy}\) tilts the noise ellipse along the diagonal.'
- q: 'A pose-graph edge says "pose 7 is about 1 m in front of pose 6". What kind of constraint is it?'
  options:
    - 'A hard equality constraint: it must hold exactly.'
    - 'A soft constraint: a residual term the optimizer may break a little, at a price set by \(\Omega_i\).'
    - 'An inequality constraint \(c(x) \le 0\).'
  answer: 1
- q: 'What is the only hard constraint in this series, and why is it needed?'
  options:
    - 'Fixing the first pose (gauge fixing): with only relative measurements, the whole map can move without changing any residual.'
    - 'Setting \(\Omega = I\), so that all measurements count the same.'
    - 'A speed limit of 2 m/s, so the robot cannot jump.'
  answer: 0
- q: 'In the line fit, the best line has a lower cost than the true line (2.1610 vs 2.2048). Why?'
  options:
    - 'A bug in `numpy.linalg.lstsq`.'
    - 'The true line was generated with the wrong slope.'
    - 'The best line fits this particular noise; the true line does not care about our noise.'
  answer: 2
  explain: 'With noisy data, the best fit is the best explanation of the data we have, not the truth. More data brings it closer to the true line.'
- q: 'Why are the residuals in the line fit drawn vertical, not perpendicular to the line?'
  options:
    - 'We assume the \(t_i\) are exact and only the \(z_i\) are noisy.'
    - 'Vertical segments are always shorter.'
    - 'Perpendicular residuals cannot be squared.'
  answer: 0
  explain: '\(r_i = a\, t_i + b - z_i\) only measures the error in \(z\), at the known place \(t_i\).'
{{< /quiz >}}

**Previous:** [Post 00a: The Least-Squares Problem, Words and Cost](../post00a-the-least-squares-problem/)
**Next:** [Post 01: Linear vs Nonlinear Least Squares](../post01-linear-vs-nonlinear-least-squares/)
