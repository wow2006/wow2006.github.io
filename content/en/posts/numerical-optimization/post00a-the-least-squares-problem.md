---
title: "Post 00a: The Least-Squares Problem, Words and Cost"
date: 2026-11-10T09:00:00+03:00
weight: 1
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

A robot drives around and measures things: how far it moved, how far away a beacon is, which way it turned. Every measurement is a little bit wrong. From all these wrong numbers, we want one good answer: where the robot really is, and where it has been. Almost every method that does this, from fitting a line in a spreadsheet to SLAM (Simultaneous Localization and Mapping) in a self-driving car, is the same idea: **least squares**.

This series builds that idea from the ground up. We start with the words and the cost function (this post), solve the easy linear case in one shot (post 01), then build three iterative methods for the hard nonlinear case: Gradient Descent (post 02), Gauss-Newton (post 03) and Levenberg-Marquardt (post 04). In post 05, we run all three on a 2D pose graph with 400 robot poses and compare them on speed and accuracy.

All code is plain Python with NumPy, SciPy and Matplotlib. Every number in the series comes from running these scripts.


## What We Will Learn

* What *variables*, *observations*, a *measurement model* and *residuals* are.
* How residuals become one number, the *cost*, and why we square them.

## Table of Contents

1. The Words.
2. Residuals.
3. The Cost.

---

## 1. The Words

**Variables (the state).** The things we want to find. We put them all in one vector \(x \in \mathbb{R}^n\), where \(n\) is how many numbers we are looking for.{{< sidenote >}}A vector is just a list of numbers, like a shopping list of everything we want to know. \(x \in \mathbb{R}^n\) reads "\(x\) is a list of \(n\) real numbers".{{< /sidenote >}} For a robot in 2D, the state of one pose is \(x = (x, y, \theta)\): position and heading. For a whole trajectory of 400 poses, \(n = 1200\).

**Observations (measurements).** The numbers our sensors gave us. We call the \(i\)-th one \(z_i\).{{< sidenote >}}The small \(i\) is a label, like a seat number: \(z_1\) is the first measurement, \(z_2\) the second, and so on.{{< /sidenote >}} It can be a single number (a distance) or a small vector (a relative motion \((\Delta x, \Delta y, \Delta\theta)\)).

**Measurement model.** A function \(h_i(x)\) that answers: *if the state were \(x\), what would sensor \(i\) read?* For a range sensor and a beacon at position \(b\), it is the distance:{{< sidenote >}}The double bars \(\lVert \cdot \rVert\) mean length. Here, the straight-line distance between \(p\) and \(b\); later, \(\lVert r \rVert^2\) means "square every entry of \(r\) and add them up".{{< /sidenote >}}

$$
h_i(x) = \lVert p - b \rVert,
$$

where \(p\) is the robot position inside \(x\). The model has no noise in it. It is our idea of the perfect sensor.

So we have two numbers for every measurement: what we *predict* (\(h_i(x)\)) and what we *measured* (\(z_i\)). If \(x\) were the true state and the sensor were perfect, they would be equal. In real life, they are not.

**A picture, top view.** Here is the same story for a robot driving on a flat floor. Its state is a position and a heading, \(x = (x, y, \theta)\). Three beacons \(B_1, B_2, B_3\) sit at known places. Each one gives two measurements: a range \(z_i\) (distance) and a bearing \(\varphi_i\) (the angle from the robot's nose to the beacon). The bearing is what makes the heading \(\theta\) visible: ranges alone cannot tell which way the robot faces.

<img src="/images/numerical-optimization/post00/robot_setup.svg" alt="Top view of a robot at (x, y) with heading angle theta from the world x axis, three beacons B1 to B3, dashed lines labelled with ranges z1 to z3, and the bearing angle phi1 measured from the heading to B1" class="center">

*The setup. The robot (black) is somewhere we do not know. Heading \(\theta\) is measured from the world \(x\) axis; bearing \(\varphi_i\) is measured from the heading.*

Now we make a guess \(x\) for the pose. For each beacon, the model \(h_i(x)\) says what range and bearing the guess *predicts*. The sensor says what it *measured*. If we draw the measured range and bearing starting from our guessed pose, the measurement points at a spot near the beacon, but not exactly on it. The gap is the residual:

<img src="/images/numerical-optimization/post00/robot_predict_residual.svg" alt="A guessed robot pose in blue with dashed blue lines to the beacons (predicted), orange lines drawn with the measured range and bearing ending near each beacon, and short red segments r1, r2, r3 between the orange tips and the beacons" class="center">

*Predict, measure, residual. Blue dashed: what the guess predicts. Orange: what the sensor measured, drawn from the guess. Red: the gaps, the residuals \(r_i = h_i(x) - z_i\).*

Refining the pose means moving \(x, y\) *and* turning \(\theta\) until the red gaps are as small as possible together. Moving the robot changes the ranges; turning it changes every bearing. Post 03 shows the step that does this (Gauss-Newton). Here is what it looks like on this example: the cost \(F\) falls from 348.9 to 1.3 in one step and to 0.8 in two (the position and heading errors are in the panels; the rest is measurement noise):

<img src="/images/numerical-optimization/post00/robot_refine.svg" alt="Three panels of the same scene at iterations 0, 1 and 2: the blue robot starts away from the grey true pose with long red gaps, then moves and turns until the orange lines hit the beacons and the red gaps vanish" class="center">

*Refinement. Blue: our current guess. Grey: the true pose we cannot see. After each step the orange lines land closer to the beacons.*

The code that draws these three pictures, with the beacons, noise and Gauss-Newton loop:

{{< code file="optimization/post00/robot_topview.py" title="Top-view pictures" >}}

## 2. Residuals

The **residual** is the difference between the two:

$$
r_i(x) = h_i(x) - z_i \qquad \text{(predicted minus measured)}.
$$

A residual is a measure of disagreement. If \(r_i = 0\), the state \(x\) explains measurement \(i\) perfectly. If \(r_i\) is large, \(x\) and measurement \(i\) do not agree.

We stack all \(m\) residual numbers into one long vector \(r(x) \in \mathbb{R}^m\). In general, \(m\) is larger than \(n\): we have more measurements than unknowns. This is on purpose. Many noisy measurements of the same thing average out the noise. But it also means no \(x\) can make every residual zero at once. We have to settle for the \(x\) that makes them **all small together**.

## 3. The Cost

"All small together" needs to be a single number, so we can compare two guesses and say which one is better. The standard choice is the sum of squares:

$$
F(x) = \frac{1}{2} \sum_i r_i(x)^2 = \frac{1}{2} \lVert r(x) \rVert^2 .
$$

\(F\) is the **cost** (also called the *objective* or *loss*).{{< sidenote >}}\(\sum_i\) (a capital Greek sigma) just means "add them all up": \(\sum_i r_i^2 = r_1^2 + r_2^2 + \dots\){{< /sidenote >}} The best state is the one with the smallest cost:{{< sidenote >}}\(\arg\min_x F(x)\) reads "the \(x\) that makes \(F\) smallest": not the smallest cost itself, but where it happens. The star in \(x^*\) marks that best \(x\).{{< /sidenote >}}

$$
x^* = \arg\min_x F(x).
$$

Why squares, and not, say, absolute values?

* Squares are never negative, so a residual of \(+2\) and one of \(-2\) are equally bad. They cannot cancel out.
* Large errors are punished much more than small ones: a residual of 3 costs 9 times as much as a residual of 1.
* The square is smooth: it has a derivative everywhere. All the methods in this series use derivatives.
* If the noise is Gaussian (bell shaped), the least-squares answer is also the *most likely* answer. This is the real reason it is everywhere.

The \(\frac{1}{2}\) in front is only there to make the derivative cleaner (the 2 from the square cancels it). It does not change where the minimum is.

## Key Takeaways

* Least squares finds the state \(x\) whose predictions \(h_i(x)\) agree best with the measurements \(z_i\).
* A residual is predicted minus measured. The cost is half the sum of the squared residuals, each measured with its information matrix \(\Omega_i = \Sigma_i^{-1}\): \(F = \frac{1}{2}\sum_i r_i^\top \Omega_i r_i\).

{{< quiz >}}
- q: 'What is the residual \(r_i(x)\)?'
  options:
    - 'Measured minus true: \(z_i - x\).'
    - 'The noise that sensor \(i\) added.'
    - 'Predicted minus measured: \(h_i(x) - z_i\).'
    - 'The distance from the robot to beacon \(i\).'
  answer: 2
  explain: 'The model \(h_i(x)\) says what sensor \(i\) would read if the state were \(x\). The residual is how far that is from what the sensor really read, \(z_i\).'
- q: 'Why can we usually not make every residual zero at once?'
  options:
    - 'We have more measurements than unknowns (\(m > n\)), and they are noisy, so they disagree with each other.'
    - 'The measurement model \(h_i(x)\) has noise in it.'
    - 'The \(\frac{1}{2}\) in the cost keeps it above zero.'
  answer: 0
  explain: 'The model is noise-free; the measurements are not. Many noisy measurements of the same thing cannot all be matched exactly, so we look for the \(x\) that makes them all small together.'
- q: 'In \(F(x) = \frac{1}{2} \sum_i r_i^2\), how many times as much does a residual of 3 cost as a residual of 1?'
  options: ["3", "9", "6", "The same"]
  answer: 1
  explain: '\(3^2 = 9\) and \(1^2 = 1\). Squaring punishes large errors much more than small ones.'
- q: 'Why is there a \(\frac{1}{2}\) in front of the cost?'
  options:
    - 'It moves the minimum closer to the true state.'
    - 'It averages the residuals.'
    - 'It cancels the 2 from the derivative of the square. It does not change where the minimum is.'
  answer: 2
- q: 'What is the real reason least squares is used everywhere?'
  options:
    - 'It is the only cost that has a minimum.'
    - 'If the noise is Gaussian, the least-squares answer is also the most likely answer.'
    - 'It always finds the true state exactly.'
  answer: 1
  explain: 'Smoothness and "no cancelling" help too, but the link to Gaussian noise is why it is the default choice.'
{{< /quiz >}}

**Next:** [Post 00b: Weights, Constraints and a First Fit](../post00b-weights-constraints-and-a-first-fit/)
