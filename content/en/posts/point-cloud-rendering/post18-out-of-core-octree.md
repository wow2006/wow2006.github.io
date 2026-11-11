---
title: "Post 18: An Out-of-Core Octree for Point Clouds"
date: 2026-10-15T12:00:00+03:00
weight: 19
draft: true
toc: false
images:
series: ["point-cloud-rendering"]
tags:
  - opengl
  - cpp
  - gpu
  - point-cloud
  - octree
  - lod
---

Everything so far assumed the whole cloud lives in the pool. That works to about 700 M points in this design (the resolve's four chunks, [Post 03](../post03-one-big-pool/)) and to however much VRAM the card has, whichever comes first. The test set for this part is `toronto16`: 16 shifted copies of the four Toronto tiles, **1 253 123 360 points, a 32.6 GB LAS**, and in the pool's 20-byte layout it needs 25 GB, more than the 24 GB card. To draw it you must stop uploading everything and draw only what the current view needs, at the detail it needs.

The editor's answer is the Potree 2 design (Schuetz, Ohrhallinger and Wimmer, 2020): a multi-resolution octree on disk, converted once, streamed per frame within a fixed VRAM budget.

## The key property: every point is stored once

A classic LOD tree duplicates data: each level stores a coarser average of its children. This one does not. A node at level `l` holds roughly **one real source point per cell of a 128³ grid** laid over its cube: the point closest to the cell's centre, taken from its children's points. That point is then *removed* from the child. Every point appears in exactly one node, the shallowest that sampled it, and leaves hold the rest.

```
root:   a sparse sample of the whole cloud (one point per 128³ cell of the root cube)
level 1: fills in around it, in each octant
...
leaves: the remaining points, up to 20 000 per node
```

A node together with its ancestors shows the node's cube at spacing `cube / 128`. Drawing more levels in a region means more detail there, and a point is never drawn twice. This is also what makes editing possible without rebuilding the tree ([Post 26](../post26-edits-on-streamed-clouds/)): a deleted point disappears at every level, because it lives in exactly one node.

## Converting a file bigger than RAM

The converter reads the inputs three times through the streaming reader from [Post 07](../post07-loading-100m-points-without-freezing/) (`read_batches`), then indexes in memory in parallel:

1. **Bounds** (and intensity percentiles).
2. **Counts** on a 256³ grid, merged top-down into *chunks* of at most 8 M points along the octree.
3. **Distribute**: each point is written into a temp file in per-batch chunk blocks.
4. **Index**, chunk by chunk, in waves of at most 16 workers: sort the chunk's points by a 63-bit Morton key (the radix sort of [Post 05](../post05-morton-order/)), split while a node has more than 20 000 points (down to level 14), and sample bottom-up as described above.
5. **Top**: the levels above the chunks are sampled from the chunk roots.

Written in chunk order, the output is **byte-identical for any thread count**, which the tests assert (1 thread against 16). Measured on 64 threads with the input in the page cache:

| input | points | time | result |
|---|---|---|---|
| Toronto 4 tiles | 78.3 M | 10 s | 14 823 nodes, 1.5 GB |
| toronto16 | 1.25 B | 153 s | 234 832 nodes, depth 14, 673 chunks (the largest 14.5 M), 23.8 GB (v1) |

toronto16 by pass: bounds 15.9 s, count 17.1, distribute 32.0, index 82.9, top 0.7. The indexing pass dominates. One design detail came from the data: on the densest street a single 256³ cell held 14.5 M points. The first 128³-grid attempt gave a 23.2 M chunk and a 196 s conversion; the final grid brought the largest chunk to 14.5 M and the total to 153 s. A synthetic 10 M-point build takes 4347, 1687 and 1677 ms at 1, 8 and 32 threads (`BM_BuildOctree`), so it stops scaling around 8 threads, like the Morton sort.

## The file

A `.pco` is a header plus per-node SoA data at 64-byte-aligned offsets, a breadth-first hierarchy table, and a manifest:

```cpp
struct OctNode {
    uint64_t offset;            // bytes: the node's data
    uint32_t count;             // points (0: an inner node whose samples all went to its ancestors)
    uint32_t first_child;       // index of its first child (0: none)
    uint32_t x, y, z;           // cube coordinates at `level`
    uint8_t level, children;    // children: octant mask
    uint16_t pad;
};                               // 32 B per node
```

Nodes are ordered by level, then by Morton order of the cube coordinates, so a node's children are **contiguous in octant order** and `children` is just a bit mask. The manifest records `size mtime path` for each input, and a cached `.pco` next to the inputs is reused while its manifest matches; opening a cached file takes about 30 ms (the file is memory-mapped and validated, not read).

## Choosing what to draw: one priority queue

Each frame the app calls `lod_select`, a pure function of the octree, the camera and three parameters:

```cpp
struct LodParams {
    std::size_t budget = 30'000'000; // points
    double spacing_px = 1.5;         // refine a node while its spacing (cube / grid) projects to more than this
    std::size_t page = 4096;
    std::size_t max_pages = ~std::size_t(0);
};
```

The priority of a node is its **projected spacing in pixels**:

```cpp
double lod_spacing_px(const Octree& o, const OctNode& n, const LodCamera& cam)
{
    const double s = o.node_size(n), spacing = s / double(o.h.grid);
    if (cam.ortho) return spacing * cam.px;
    const glm::dvec3 c = o.node_min(n) + 0.5 * s;
    const double d = glm::length(glm::dvec3(cam.view * glm::dvec4(c, 1.0))) - 0.8660254037844386 * s; // minus the half-diagonal
    return spacing * cam.px / std::max(d, cam.min_dist);
}
```

and the algorithm is a best-first search:

1. Push the root. Pop the node with the **largest projected spacing**.
2. Stop when adding it would exceed the point budget or the page budget.
3. Select it. If its spacing is already `<= spacing_px` (the "Detail" slider, default 1.5 px) or it has no children, do not refine it.
4. Otherwise push its children that pass the **frustum test** (the cube against six planes extracted from the clip matrix, in double).

The invariants fall out of the order: a selected node's parent is always selected (it was popped first), refinement spends the budget where the screen is coarsest, and a budget that is too small yields a uniformly coarser picture instead of holes. Tests check determinism (the same inputs give the same selection), parents-first, looking away, and the budget.

## What it costs and what it gives

`toronto16`, budget 30 M points, detail 1.5 px, 2400 x 1350, warm cache:

| view | settle | points drawn | GPU scene median / p95 |
|---|---|---|---|
| fit all (16 copies) | 0.02-0.04 s | 0.27 M | 0.115 / 0.15 ms |
| aerial 300 m | 0.02-0.15 s | 1.92 M | 0.136 / 0.17 ms |
| street 60 m | 0.03-0.22 s | 3.73 M | 0.155 / 0.19 ms |
| street 15 m | 0.01-0.09 s | 3.13 M | 0.145 / 0.17 ms |

624 nodes (0.11 GB) streamed over the run's five views (the four above, then back to aerial 300 m), and the CPU side (selection plus cache) is under 0.06 ms per frame. A cloud of 1.25 billion points, which does not fit in VRAM, renders in about 0.15 ms of GPU time. The cache holds 9 220 pages (37.8 M points, 720 MB), and the same views with a 3 M-point cache (76 MB) still settle, with 387 evictions.

Does it draw the same picture as a direct load? The acceptance test: on the four tiles with a 100 M budget and detail 0 (all 78.3 M points drawn), the compute path differed from a direct load in **34 of 1.44 M pixels, 0 of them coverage**, only depth ties because the points are in a different index order. The LOD path never applies the 128-point block shuffle (nodes keep their sampling order), which is why `GL_POINTS` is slower on it: 59.0 vs 51.3 ms.

## Choosing when to stream

The behaviour is a setting, "Large files", with Auto as the default: a dropped set that would not fit the VRAM budget is converted from all its files into one LOD cloud. Always and Never are also offered, and a `.pco` dropped directly just opens. `pce --convert out.pco files...` converts without a window.

## Checklist

1. Store **real points, once**: edits and exports stay simple.
2. **Sample bottom-up to a grid**, nearest the cell centre, so each level is evenly spread.
3. Convert in **bounded passes** over a streaming reader; make the output independent of the thread count.
4. Order nodes **breadth-first, children contiguous**, so the hierarchy is a flat 32-byte table.
5. Select by **projected spacing in a priority queue** under a point and page budget; test determinism.
6. Verify the **end state equals a direct load**.

Next: the file is 23.8 GB. Can we make it smaller without losing a millimetre? [Post 19](../post19-quantized-nodes/).

## Where this lives in the code

`src/io/octree.hpp` and `octree.cpp` (`build_octree`, `open_octree`, `lod_select`, `lod_spacing_px`), `src/app/lod/` (conversion trigger, `LodJob`), `tools/` (`pce_tiles` builds the test set). Numbers: ROADMAP 6.1 and BENCH.md F0.1.
