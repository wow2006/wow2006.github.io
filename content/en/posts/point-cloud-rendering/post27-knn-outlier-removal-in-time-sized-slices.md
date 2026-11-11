---
title: "Post 27: kNN Outlier Removal on a GPU Grid, in Time-Sized Slices"
date: 2026-10-15T21:00:00+03:00
weight: 28
draft: true
toc: false
images:
series: ["point-cloud-rendering"]
tags:
  - opengl
  - cpp
  - gpu
  - point-cloud
  - knn
  - compute-shader
---

Selection, hide and delete are single-pass operations. Statistical outlier removal is not: for every point it needs its *k nearest neighbours*, and then a global statistic over all points. It is the first editor feature that does real work per point, and the one that taught the project its most important rule about GPUs: **a long dispatch can hang the machine's graphics**.

## The algorithm

For every live point (neither hidden nor deleted, read from the pool's flags, which are the truth, [Post 24](../post24-flags-and-class-edits-as-compute-passes/)):

1. `m_i` = the mean distance to its `k` nearest live neighbours within a radius (k 8, radius 1 m by default); no neighbour in range makes it an outlier.
2. `mu` and `sigma` of the `m_i` (population statistics) over the points that have neighbours.
3. A point is an outlier if `m_i > mu + n_sigma * sigma` (n_sigma 2).

The result is a selection mask, so it composes with hide and delete, and undoes like any selection op (`kOpSelectMask`, one XOR delta, [Post 25](../post25-undo-redo-as-gpu-xor-deltas/)).

## First, a CPU version as the reference

The CPU implementation uses a kd-tree. It is deterministic for every thread count (the statistics are reduced over fixed blocks). Measured, k 8, radius 1 m, 2 sigma:

| cloud | threads | time | throughput |
|---|---|---|---|
| synthetic surface, 10 M points | 1 / 8 / 32 | 9901 / 1706 / 833 ms | 1.0 / 5.9 / 12.0 M pts/s |
| Toronto L002, 10 283 800 points | 1 / 8 / 32 | 10713 / 1801 / 831 ms | 0.96 / 5.7 / 12.4 M pts/s |

Roughly 0.83 s for a 10 M-point tile on 32 threads. Fine as a job, too slow to feel interactive, and a perfect oracle for a GPU version: the GPU mask must equal it point for point.

## A GPU neighbour grid

A kd-tree is pointer-chasing and a poor fit for a GPU. The GPU twin uses **hashed uniform grids built by a counting sort**:

1. Count points per cell (cells hashed into a table of at least as many buckets as points).
2. Prefix-sum the counts into cell starts (a multi-level scan).
3. Scatter the points into cell order.
4. For each point, search the cells around its own in expanding shells, keeping the `k` best distances in registers (`K` is a compile-time constant, one program per `k`), and stop when the `k`-th distance is closer than any unvisited cell, or the radius is covered.

The kNN is **exact**: a shell is only searched while a closer point could be in it.

**One grid was not enough.** The dense tiles have strongly varying density within a tile. Small cells make sparse queries walk many shells; big cells make dense queries test thousands of candidates. So there are **three grids, each cell 3x finer**, and a query picks its level from how full its cell is in the coarsest grid. The sweep over layouts, in GPU ms summed over the four tiles:

| grid | sum |
|---|---|
| 1 level, factor 0.4 | 3518 |
| 1 level, factor 1.5 | 1075 |
| 3 levels, target 8, factor 3 | 663 |
| **3 levels, target 4, factor 4 (default)** | **580** |
| 3 levels, target 4, factor 5 | 557 |

The default is not the absolute minimum; it was chosen for the best balance, with L002 alone fastest on a single level (35 ms). Capping the shells at two (wrong results, as an experiment) saved only 10 % on the densest tile, which tells you that the cost is the dense cells, not the far shells.

### Making the GPU match the CPU exactly

The mask must be identical, and the notes list how:

- squared distances in float, with **no fused multiply-add** (`precise`),
- distances as correctly rounded float square roots,
- summed nearest-first in double, the mean rounded to float,
- the statistics in double.

Ties between exactly equal distances may pick different neighbours, which changes no distance. Results against the CPU on the four Toronto tiles:

| cloud | points | GPU ms | CPU kd-tree ms | GPU vs CPU | points that differ |
|---|---|---|---|---|---|
| L004 | 6.7 M | 20.2-20.6 | 504 | 25x | 0 |
| **L002** | **10.3 M** | **37.5-38.4** | 729 | 19x | 0 |
| L001 | 21.6 M | 132.5-139.3 | 1597 | 12x | 0 |
| L003 | 39.7 M | 388.9-392.0 | 2886 | 7.4x | 0 |

The research goal was "outliers on L002 in under 200 ms on the GPU": 38 ms. `mu` and `sigma` are bit-identical on two tiles and equal to 15 digits on the other two. The harness is also run with 2^20 buckets (many cells per bucket, which must not mix) and with the kNN pass in slices.

The checks fail when the code breaks (mutation-tested): no cell filter on shared buckets (6 points differ), no `precise` on the squared distance (mu and sigma differ in the tenth digit), hidden points in the grid (3 141 differ), every level reading level 0's cells (9.5 M differ). Two mutations were judged *equivalent here* and left: `sqrt` instead of the corrected version, since the 4090's sqrt is already correctly rounded on these inputs, and no rounding margin in the shell bound.

## The problem: one long dispatch

On a Linux machine with the same GPU, whole-cloud outliers on a larger scan (S3DIS Area_5, 78.6 M points) measured:

| case | L002 | Area_3 (18.7 M) | Area_5 (78.6 M) |
|---|---|---|---|
| segment step, all passes in one call (wall) | 339 ms | 1136 ms | 14 405 ms, then the GPU reset |
| whole cloud, kNN as one dispatch | 2.35 s | 2.29 s (20 s with 2^20 buckets: GPU reset) | 13.6 s: GPU reset |

A **GPU reset** (`GL_GUILTY_CONTEXT_RESET`) kills the context: every later readback returns zeros. The resets came with the longest dispatches (13.6 s, and 20 s with 2^20 buckets), while the 2.3 s ones survived; the project treats about 2 s as the danger zone. And a UI thread that waits for a 14-second call has frozen the app anyway. This is why the project has a written rule: **a user-triggered operation that can take more than 100 ms runs as a job with progress and Cancel, and GPU work is cut into time-sized slices so no frame stalls more than about 50 ms.**

## The fix: slices sized by measurement

The run is a small state machine, advanced one step per frame:

```cpp
struct GpuOutlierRun {
    int stage;                 // grid level l: count 3l, scan 3l+1, scatter 3l+2; then the kNN; then done
    std::size_t next;          // the stage's first point not dispatched yet
    std::size_t slice[3];      // points in the next count / scatter / kNN slice
    bool pending;              // a GL_TIME_ELAPSED query in flight
    double max_slice_ms;       // the longest unit
};
bool gpu_outliers_step(GpuNeighbours& g, const GpuPool& pool, GpuOutlierRun& r);   // true once none is left
```

Each step allocates one scratch buffer or dispatches one **unit**: a slice of a per-point pass, sized from the *measured time of the previous slice of that pass* to take about 10 ms of GPU, or one grid scan. The slice size adapts: it grows when a slice was quick and shrinks when it was slow. The app issues one step per frame, with a progress bar and a Cancel button; conflicting operations (save, merge, remove) are refused while it runs.

After, on the same Linux machine (`--bench-edit`, every check passes):

| | L002 | Area_3 | Area_5 |
|---|---|---|---|
| longest frame while the job runs | 32.5 ms | 34.1 ms | 44.6 ms |
| segment step: GPU, units, wall | 258 ms, 58 units, 0.28 s | 874 ms, 120 units, 0.94 s | 5422 ms, 554 units, 5.91 s |
| whole cloud, GPU ms (5 runs) | 1276-1303 | 1631-1645 | 8642-8670 |
| longest single unit | 30.2 ms | 18.4 ms | 37.1 ms |

No frame over 45 ms, no reset on any set. The bench fails any frame over 100 ms.

Slicing even made the whole-cloud pass faster on that machine (L002: 2.35 s as one dispatch to 1.28-1.30 s). On the Windows machine from the R10 work the opposite was measured: 4 M-point slices cost 1.6-2.2x and 16 M-point slices 1.3-1.4x, because each slice waits for its slowest warps, so there only clouds above 2^26 points are sliced (against a 2 s reset). Two drivers, two answers: **measure slicing on your target drivers**. The notes also record that the same kernel is about 35x slower on that Linux driver than on Windows (L002 38 ms vs about 1.3 s), reason not investigated.

## Checklist

1. Write the **CPU reference first**; make the GPU match it, including float semantics (`precise`, no FMA), and test the match on real data.
2. Use a **hashed grid with multi-level cells** rather than a tree for kNN on a GPU, and tune on dense and sparse data.
3. **Never run a dispatch longer than the driver watchdog**, and design every long GPU operation as a *job over frames*.
4. **Size slices from measured time** (a `GL_TIME_ELAPSED` query on the previous slice), not from point counts.
5. Give the job **progress, Cancel and a conflict list**.
6. Test the **mutations** that matter and name the ones you judged equivalent.

Next: how the numbers in this part (and the whole series) were measured: [Post 28](../post28-benchmarking-a-renderer-honestly/).

## Where this lives in the code

`src/render/neighbours.hpp`, `neighbours.cpp` (`GpuNeighbours`, `gpu_outliers_begin/step/end`), `src/ops/outlier.cpp` (CPU reference), `src/app/segmentation/` (the job). Numbers: BENCH.md "R10 - statistical outlier removal" and "Task 2.3". Commits 238fa9d, c614267, 97f447d, 1123ef7.
