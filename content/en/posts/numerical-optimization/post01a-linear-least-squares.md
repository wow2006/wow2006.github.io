---
title: "Post 01a: Linear Least Squares and the Normal Equations"
date: 2026-11-10T10:00:00+03:00
weight: 3
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

In [post 00a](../post00a-the-least-squares-problem/) and [post 00b](../post00b-weights-constraints-and-a-first-fit/), we defined the cost \(F(x) = \frac{1}{2}\lVert r(x) \rVert^2\) and fitted a line, but we let NumPy find the answer for us.{{< sidenote >}}Reminder from post 00: \(x\) is the list of unknowns we want to find, \(r(x)\) the list of residuals (predicted minus measured), and \(F\) the cost: one number that says how bad a guess is.{{< /sidenote >}}{{< sidenote >}}\(\lVert r \rVert\) is the length of the list \(r\), so \(\lVert r \rVert^2\) means "square every residual and add them up". The \(\frac{1}{2}\) only makes derivatives tidier; it does not move the minimum.{{< /sidenote >}} In this post, we find it ourselves. When the residuals are *linear* in \(x\), there is a formula that gives the exact answer in one step. In [post 01b](../post01b-nonlinear-least-squares/), we see what happens when they are *nonlinear*: there is no such formula, and we need a new tool.

## What We Will Learn

* What makes a least-squares problem linear or nonlinear.
* How to derive the normal equations \(A^\top A x = A^\top b\), and why we solve them with Cholesky or QR instead of a matrix inverse.{{< sidenote >}}\(A^\top\) ("A transpose") is the table \(A\) turned on its side: its rows become columns.{{< /sidenote >}}
* What a condition number is, and why \(A^\top A\) squares it.

## Table of Contents

1. Linear Least Squares.
2. The Normal Equations.
3. Solving Them Well: Cholesky, QR and Conditioning.
4. The Full Program.

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

![Three arrows from the origin: column 1 and column 2 of A lie in a shaded plane, b points up out of it, and A x* is the point of the plane right below b; the dashed residual from b down to A x* meets the plane at a right angle](/images/numerical-optimization/post01/normal_equations.png)

*The same idea in a picture, for three measurements and two unknowns. Every possible \(Ax\) lies in the plane spanned by the columns of \(A\). The measurements \(b\) usually lie off that plane, so no \(x\) fits them exactly. The best \(Ax^*\) is the point of the plane closest to \(b\): the foot of the perpendicular. So the residual meets the plane at 90°, which is what "normal" means.*

Try it: each line is a point in the bowl. Drag it downhill to where \(\nabla F = 0\).

<details><summary style="cursor:pointer;margin:.5em 0">Interactive: the cost bowl</summary>

<iframe src="/tools/line-fit-explorer/?lang=en&view=bowl" title="Interactive cost bowl: each point (a, b) is a line; drag it, follow the downhill arrow, and see the gradient vanish at the bottom" style="width:100%;height:1000px;border:0" loading="lazy"></iframe>

<p><a href="/tools/line-fit-explorer/?lang=en&view=bowl">Open it in its own page</a>.</p>

</details>

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

## 4. The Full Program

One script covers both parts of post 01: the line fit from this part, and the range-only problem from [post 01b](../post01b-nonlinear-least-squares/).

{{< code file="optimization/post01/linear_vs_nonlinear.py" title="Full linear vs nonlinear program" >}}

```bash
python3 linear_vs_nonlinear.py .
```

## Key Takeaways

* Linear least squares (\(r = Ax - b\)) has a quadratic bowl as its cost and is solved in one step by the normal equations \(A^\top A x = A^\top b\).
* Solve them with Cholesky (fast) or QR (more accurate), never with an explicit inverse. Forming \(A^\top A\) squares the condition number: 14.9 became 223.2 in our line fit.

{{< quiz >}}
- q: 'We fit a parabola \(z = a t^2 + b_{\text{line}} t + c\) to data. Is this linear least squares?'
  options:
    - 'No: \(t^2\) is not linear.'
    - 'Yes: the unknowns \(a, b_{\text{line}}, c\) only appear multiplied by known numbers.'
    - 'Only if the data lie on a straight line.'
  answer: 1
  explain: '"Linear" is about the unknowns \(x\), not the data. Row \(i\) of \(A\) is just \([\,t_i^2,\ t_i,\ 1\,]\).'
- q: 'For the line fit with \(m\) points, what are the sizes of \(A\) and of \(A^\top A\)?'
  options: ['\(A\) is \(2 \times m\), \(A^\top A\) is \(m \times m\)', '\(A\) is \(m \times 2\), \(A^\top A\) is \(2 \times 2\)', 'Both are \(m \times m\)']
  answer: 1
  explain: 'One row per measurement, one column per unknown. The normal equations are a small \(n \times n\) system, no matter how many points we have.'
- q: 'Why is the solution of \(A^\top A x = A^\top b\) a minimum and not a maximum?'
  options:
    - 'Because \(b\) is positive.'
    - 'Because the Hessian \(A^\top A\) satisfies \(v^\top A^\top A v = \lVert Av \rVert^2 \ge 0\): the bowl opens upward.'
    - 'Because we start the search from a good guess.'
  answer: 1
  explain: 'There is no search at all: the normal equations give the bottom of the bowl in one step.'
- q: 'What does the word "normal" in "normal equations" refer to?'
  options:
    - 'The noise is assumed to be normally distributed.'
    - 'The equations are the usual, standard ones.'
    - 'At the solution, the residual \(Ax - b\) is perpendicular (normal) to every column of \(A\).'
  answer: 2
  explain: 'That is exactly what \(A^\top (Ax - b) = 0\) says. For the line fit, the column of ones gives \(\sum_i r_i = 0\).'
- q: 'Why should code not compute \(x = (A^\top A)^{-1} A^\top b\) literally?'
  options:
    - 'The formula is wrong.'
    - 'An explicit inverse is slower and less accurate than solving the system with Cholesky or QR.'
    - 'An inverse cannot be computed for a \(2 \times 2\) matrix.'
  answer: 1
- q: 'If \(\kappa(A) = 10^6\), about how many of the 16 digits of a double can Cholesky on the normal equations lose?'
  options: ['About 3', 'About 6', 'About 12', 'None']
  answer: 2
  explain: '\(\kappa(A^\top A) = \kappa(A)^2 = 10^{12}\), so about 12 digits. QR works on \(A\) directly and loses about 6.'
- q: 'When is QR worth its extra cost over Cholesky?'
  options:
    - 'When \(A\) is badly conditioned: QR never forms \(A^\top A\), so it does not square \(\kappa\).'
    - 'Always: Cholesky gives a different answer.'
    - 'Only when \(A\) is square.'
  answer: 0
  explain: 'In our line fit, \(\kappa(A) = 14.9\) and all three solvers agree to \(10^{-10}\). Cholesky is fine there, and faster.'
{{< /quiz >}}

**Previous:** [Post 00b: Weights, Constraints and a First Fit](../post00b-weights-constraints-and-a-first-fit/)
**Next:** [Post 01b: Nonlinear Least Squares and Linearization](../post01b-nonlinear-least-squares/)
